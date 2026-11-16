import random
import unittest
from generation import integer_codecs as c
from generation.operator_composition import compose_chain

class IntegerCodecTests(unittest.TestCase):
    def test_chains_roundtrip_edges_and_random(self):
        rng=random.Random(19001)
        samples=[[],[0],[1],[-1],[0]*100,[-(1<<63),(1<<63)-1]*4,[rng.randint(-10000,10000) for _ in range(259)]]
        reg=c.registry()
        for ids in [('for',),('block_for',),('delta','block_for','zlib'),('raw64',),('raw64','zlib'),('raw64','lzma'),('delta','zigzag','varint'),('delta2','zigzag','rle'),('delta2','zigzag','bitpack'),('delta','zigzag','blockpack','zlib')]:
            chain=compose_chain(reg,ids)
            for xs in samples:
                with self.subTest(ids=ids,n=len(xs)):self.assertEqual(chain.decode(chain.execute(xs)),xs)
    def test_truncation_and_invalid_counts(self):
        for decode,payload in [(c.varint_decode,b'\x80'),(c.varint_decode,b'\x80\x00'),(c.rle_decode,b'\x01\x00'),(c.bitpack_decode,b'\x00'),(c.blockpack_decode,b'\x01\x00\x00\x00'),(c.raw64_decode,b'a')]:
            with self.subTest(decode=decode),self.assertRaises(ValueError):decode(payload)
        for encode,decode in [(c.zlib_encode,c.zlib_decode),(c.lzma_encode,c.lzma_decode)]:
            good=encode(b'payload')
            with self.assertRaises((ValueError,EOFError,c.lzma.LZMAError)):decode(good[:-1])
            with self.assertRaises((ValueError,EOFError,c.lzma.LZMAError)):decode(good+b'trailing')
    def test_raw64_range_and_bitpack_padding(self):
        with self.assertRaises(ValueError):c.raw64_encode([1<<63])
        packed=c.bitpack_encode([1]);bad=packed[:-1]+bytes([packed[-1]|128])
        with self.assertRaises(ValueError):c.bitpack_decode(bad)

    def test_count_limits_and_rle_token_overhead(self):
        from unittest.mock import patch
        with patch.object(c,'MAX_ITEMS',4):
            xs=[1,2,3,4]
            self.assertEqual(c.rle_decode(c.rle_encode(xs)),xs)
            for encode in (c.delta_encode,c.zigzag_encode,c.varint_encode,c.rle_encode,c.bitpack_encode,c.blockpack_encode,c.raw64_encode):
                with self.subTest(encode=encode),self.assertRaises(ValueError):encode([0]*5)
