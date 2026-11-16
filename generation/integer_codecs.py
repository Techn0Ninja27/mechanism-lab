"""Independent scalar implementations of known reversible integer operators.

Not wire-compatible with Protobuf/Parquet containers; headers and bounds are
our explicit test format. No claim of new compression algorithms or SIMD speed.
"""
from __future__ import annotations
import hashlib
import lzma
from pathlib import Path
import struct
import zlib
from .operator_composition import Operator, OperatorRegistry

MAX_ITEMS = 1_000_000
MAX_BITS = 128
MAX_BYTES = 64_000_000


def delta_encode(xs):
    if len(xs)>MAX_ITEMS:raise ValueError('too many integers')
    return [xs[0]] + [b-a for a,b in zip(xs,xs[1:])] if xs else []


def delta_decode(xs):
    if len(xs)>MAX_ITEMS:raise ValueError('too many integers')
    out=[];total=0
    for x in xs:total+=x;out.append(total)
    return out


def delta2_encode(xs):return delta_encode(delta_encode(xs))
def delta2_decode(xs):return delta_decode(delta_decode(xs))
def zigzag_encode(xs):
    if len(xs)>MAX_ITEMS:raise ValueError('too many integers')
    return [2*x if x>=0 else -2*x-1 for x in xs]
def zigzag_decode(xs):
    if len(xs)>MAX_ITEMS:raise ValueError('too many integers')
    return [x//2 if x%2==0 else -(x//2)-1 for x in xs]


def varint_encode(xs, *, _limit=None):
    if len(xs)>(MAX_ITEMS if _limit is None else _limit):raise ValueError('too many integers')
    out=bytearray()
    for x in xs:
        if isinstance(x,bool) or not isinstance(x,int) or x<0 or x.bit_length()>MAX_BITS:raise ValueError('unsigned integer exceeds codec domain')
        while x>=128:out.append((x&127)|128);x>>=7
        out.append(x)
    return bytes(out)


def varint_decode(data, *, _limit=None):
    if not isinstance(data,bytes) or len(data)>MAX_BYTES:raise ValueError('invalid varint payload')
    out=[];value=0;shift=0
    for byte in data:
        value|=(byte&127)<<shift
        if value.bit_length()>MAX_BITS:raise ValueError('varint exceeds bit limit')
        if byte&128:
            shift+=7
            if shift>=MAX_BITS:raise ValueError('varint exceeds bit limit')
        else:
            if shift and byte==0:raise ValueError('noncanonical varint')
            out.append(value);value=0;shift=0
            if len(out)>(MAX_ITEMS if _limit is None else _limit):raise ValueError('too many integers')
    if shift:raise ValueError('truncated varint')
    return out


def raw64_encode(xs):
    if len(xs)>MAX_ITEMS or any(isinstance(x,bool) or not isinstance(x,int) or not -(1<<63)<=x<(1<<63) for x in xs):raise ValueError('raw64 needs signed int64 values')
    return struct.pack('<'+str(len(xs))+'q',*xs)


def raw64_decode(data):
    if len(data)%8 or len(data)>8*MAX_ITEMS:raise ValueError('invalid raw64 payload')
    return list(struct.unpack('<'+str(len(data)//8)+'q',data))


def rle_encode(xs):
    if len(xs)>MAX_ITEMS:raise ValueError('too many integers')
    pairs=[]
    for x in xs:
        if pairs and pairs[-2]==x:pairs[-1]+=1
        else:pairs.extend([x,1])
    return varint_encode(pairs,_limit=2*MAX_ITEMS)


def rle_decode(data):
    pairs=varint_decode(data,_limit=2*MAX_ITEMS)
    if len(pairs)%2:raise ValueError('truncated RLE pair')
    out=[]
    for value,count in zip(pairs[::2],pairs[1::2]):
        if count<1 or len(out)+count>MAX_ITEMS:raise ValueError('invalid RLE count')
        out.extend([value]*count)
    return out


def bitpack_encode(xs):
    if len(xs)>MAX_ITEMS or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in xs):raise ValueError('invalid bitpack input')
    width=max((x.bit_length() for x in xs),default=0)
    if width>MAX_BITS:raise ValueError('bitpack width too large')
    out=bytearray(struct.pack('<BI',width,len(xs)));buffer=0;bits=0
    for x in xs:
        buffer|=x<<bits;bits+=width
        while bits>=8:out.append(buffer&255);buffer>>=8;bits-=8
    if bits:out.append(buffer)
    return bytes(out)


def bitpack_decode(data):
    if len(data)<5:raise ValueError('truncated bitpack header')
    width,n=struct.unpack('<BI',data[:5])
    if width>MAX_BITS or n>MAX_ITEMS or len(data)!=5+(width*n+7)//8:raise ValueError('invalid bitpack size')
    if width==0:return [0]*n
    out=[];buffer=0;bits=0;offset=5;mask=(1<<width)-1
    for _ in range(n):
        while bits<width:buffer|=data[offset]<<bits;offset+=1;bits+=8
        out.append(buffer&mask);buffer>>=width;bits-=width
    if buffer:raise ValueError('nonzero bitpack padding')
    return out


def blockpack_encode(xs):
    if len(xs)>MAX_ITEMS:raise ValueError('too many integers')
    out=bytearray(struct.pack('<I',len(xs)))
    for start in range(0,len(xs),128):
        block=bitpack_encode(xs[start:start+128]);out.extend(struct.pack('<I',len(block)));out.extend(block)
    return bytes(out)


def blockpack_decode(data):
    if len(data)<4:raise ValueError('truncated block count')
    n=struct.unpack('<I',data[:4])[0]
    if n>MAX_ITEMS:raise ValueError('too many integers')
    out=[];offset=4
    while len(out)<n:
        if offset+4>len(data):raise ValueError('truncated block size')
        size=struct.unpack('<I',data[offset:offset+4])[0];offset+=4
        if offset+size>len(data):raise ValueError('truncated block payload')
        block=bitpack_decode(data[offset:offset+size]);offset+=size
        if len(block)!=min(128,n-len(out)):raise ValueError('invalid block count')
        out.extend(block)
    if offset!=len(data):raise ValueError('trailing block bytes')
    return out


def for_encode(xs):
    if len(xs)>MAX_ITEMS:raise ValueError('too many integers')
    base=min(xs,default=0);encoded_base=varint_encode(zigzag_encode([base]))
    return bytes([len(encoded_base)])+encoded_base+bitpack_encode([x-base for x in xs])


def for_decode(data):
    if not data or not 1<=data[0]<=19 or len(data)<1+data[0]+5:raise ValueError('invalid reference header')
    base=zigzag_decode(varint_decode(data[1:1+data[0]]))
    if len(base)!=1:raise ValueError('invalid reference')
    return [base[0]+x for x in bitpack_decode(data[1+data[0]:])]


def block_for_encode(xs):
    if len(xs)>MAX_ITEMS:raise ValueError('too many integers')
    out=bytearray(struct.pack('<I',len(xs)))
    for start in range(0,len(xs),128):
        block=for_encode(xs[start:start+128]);out.extend(struct.pack('<I',len(block)));out.extend(block)
    return bytes(out)


def block_for_decode(data):
    if len(data)<4:raise ValueError('truncated reference blocks')
    n=struct.unpack('<I',data[:4])[0]
    if n>MAX_ITEMS:raise ValueError('too many integers')
    out=[];offset=4
    while len(out)<n:
        if offset+4>len(data):raise ValueError('truncated block size')
        size=struct.unpack('<I',data[offset:offset+4])[0];offset+=4
        if offset+size>len(data):raise ValueError('truncated block')
        block=for_decode(data[offset:offset+size]);offset+=size
        if len(block)!=min(128,n-len(out)):raise ValueError('invalid reference block count')
        out.extend(block)
    if offset!=len(data):raise ValueError('trailing block bytes')
    return out


def zlib_encode(data):
    if not isinstance(data,bytes) or len(data)>MAX_BYTES:raise ValueError('invalid byte payload')
    return zlib.compress(data,level=6)
def lzma_encode(data):
    if not isinstance(data,bytes) or len(data)>MAX_BYTES:raise ValueError('invalid byte payload')
    return lzma.compress(data,preset=6)


def zlib_decode(data):
    obj=zlib.decompressobj();decoded=obj.decompress(data,MAX_BYTES+1)
    if len(decoded)>MAX_BYTES or not obj.eof or obj.unused_data or obj.unconsumed_tail:raise ValueError('invalid or oversized zlib stream')
    return decoded


def lzma_decode(data):
    obj=lzma.LZMADecompressor(memlimit=128_000_000);decoded=obj.decompress(data,max_length=MAX_BYTES+1)
    if len(decoded)>MAX_BYTES or not obj.eof or obj.unused_data:raise ValueError('invalid or oversized lzma stream')
    return decoded


def registry():
    registry=OperatorRegistry();digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protobuf='https://protobuf.dev/programming-guides/encoding/'
    parquet='https://parquet.apache.org/docs/file-format/data-pages/encodings/'
    gorilla='https://www.vldb.org/pvldb/vol8/p1816-teller.pdf'
    specs=[
        ('delta','signed_ints','signed_ints',delta_encode,delta_decode,parquet),
        ('delta2','signed_ints','signed_ints',delta2_encode,delta2_decode,gorilla),
        ('zigzag','signed_ints','unsigned_ints',zigzag_encode,zigzag_decode,protobuf),
        ('varint','unsigned_ints','bytes',varint_encode,varint_decode,protobuf),
        ('rle','unsigned_ints','bytes',rle_encode,rle_decode,parquet),
        ('bitpack','unsigned_ints','bytes',bitpack_encode,bitpack_decode,parquet),
        ('blockpack','unsigned_ints','bytes',blockpack_encode,blockpack_decode,parquet),
        ('for','signed_ints','bytes',for_encode,for_decode,parquet),
        ('block_for','signed_ints','bytes',block_for_encode,block_for_decode,parquet),
        ('raw64','signed_ints','bytes',raw64_encode,raw64_decode,'https://docs.python.org/3/library/struct.html'),
        ('zlib','bytes','bytes',zlib_encode,zlib_decode,'https://docs.python.org/3/library/zlib.html'),
        ('lzma','bytes','bytes',lzma_encode,lzma_decode,'https://docs.python.org/3/library/lzma.html')]
    for name,inp,out,encode,decode,url in specs:
        registry.register(Operator(name,inp,out,encode,decode,source=url,
            provenance=({'source_url':url,'implementation':'independent scalar implementation; not format-compatible container'},),
            assumptions=('Input and decoded sequences bounded to one million values; encoded unsigned scalars at most 128 bits.',
                         'raw64 additionally requires signed64 range; type compatibility does not prove numeric preconditions.'),
            contract_status='untested',implementation_digest=digest))
    return registry
