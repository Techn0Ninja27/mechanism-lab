"""Small, deterministic reference codecs for fixed-length graph walks.

The framing is deliberately self-contained: a 48-byte little-endian header
(``GWC1``, version 1, codec id, N, L, node_count, k, bitwidth), followed by
codec-specific sections.  Packed streams are bit-packed continuously across
the complete corpus (the final byte is zero padded).  FaBC bitmap rows are
each independently padded to a byte; its uint64 tail offsets are item offsets
into the concatenated tail stream.  Varint offsets are byte offsets.
"""

from __future__ import annotations

import struct
from typing import Iterable

import numpy as np

MAGIC = b"GWC1"
VERSION = 1
_HEADER = struct.Struct("<4sBBHQQQQQ")
_CODECS = {"raw": 1, "packed": 2, "fabc32": 3, "fabc_packed": 4, "varint": 5}
_NAMES = {v: k for k, v in _CODECS.items()}


def _width(node_count: int) -> int:
    return max(1, (node_count - 1).bit_length())


def _check_walks(walks: np.ndarray, node_count: int) -> np.ndarray:
    a = np.asarray(walks)
    if a.ndim != 2 or a.dtype != np.uint32:
        raise ValueError("walks must be a 2-D uint32 array")
    if not isinstance(node_count, (int, np.integer)) or int(node_count) <= 0:
        raise ValueError("node_count must be a positive integer")
    if a.size and int(a.max()) >= int(node_count):
        raise ValueError("walk contains an ID >= node_count")
    return np.ascontiguousarray(a)


def _pack(values: Iterable[int], width: int) -> bytes:
    out = bytearray()
    acc = 0
    bits = 0
    mask = (1 << width) - 1
    for value in values:
        v = int(value)
        if v < 0 or v > mask:
            raise ValueError("value does not fit packed width")
        acc |= v << bits
        bits += width
        while bits >= 8:
            out.append(acc & 255)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 255)
    return bytes(out)


def _unpack(data: bytes, count: int, width: int) -> np.ndarray:
    need = (count * width + 7) // 8
    if len(data) != need:
        raise ValueError("invalid packed payload length")
    result = np.empty(count, dtype=np.uint32)
    acc = 0
    bits = 0
    pos = 0
    mask = (1 << width) - 1
    for i in range(count):
        while bits < width:
            acc |= data[pos] << bits
            pos += 1
            bits += 8
        result[i] = acc & mask
        acc >>= width
        bits -= width
    if bits and (acc & ((1 << bits) - 1)):
        raise ValueError("non-zero padding in packed payload")
    return result


def _header(codec: str, n: int, l: int, nodes: int, k: int, width: int) -> bytes:
    try:
        cid = _CODECS[codec]
    except KeyError as e:
        raise ValueError(f"unknown codec: {codec}") from e
    return _HEADER.pack(MAGIC, VERSION, cid, 0, n, l, nodes, k, width)


def _parse(blob: bytes):
    if len(blob) < _HEADER.size:
        raise ValueError("truncated codec header")
    magic, version, cid, reserved, n, l, nodes, k, width = _HEADER.unpack_from(blob)
    if magic != MAGIC or version != VERSION or reserved != 0 or cid not in _NAMES:
        raise ValueError("invalid codec header")
    codec = _NAMES[cid]
    if nodes == 0 or n > (1 << 63) or l > (1 << 63):
        raise ValueError("invalid dimensions")
    expected = _width(nodes)
    if codec in ("packed", "fabc_packed") and width != expected:
        raise ValueError("invalid packed bitwidth")
    if codec in ("raw", "fabc32") and width != 32:
        raise ValueError("invalid uint32 bitwidth")
    if codec == "varint" and width != 0:
        raise ValueError("invalid varint bitwidth")
    if codec != "fabc32" and codec != "fabc_packed" and k != 0:
        raise ValueError("k is only valid for FaBC")
    if k > nodes:
        raise ValueError("invalid hot count")
    return codec, int(n), int(l), int(nodes), int(k), int(width), memoryview(blob)[_HEADER.size:]


def _leb128(v: int) -> bytes:
    out = bytearray()
    while v >= 128:
        out.append((v & 127) | 128)
        v >>= 7
    out.append(v)
    return bytes(out)


def encode(walks: np.ndarray, codec: str, node_count: int, hot_ids: list[int] = [], rank_to_id: np.ndarray | None = None) -> bytes:
    a = _check_walks(walks, int(node_count))
    n, l = a.shape
    nodes = int(node_count)
    if codec in ("raw", "packed"):
        if hot_ids:
            raise ValueError("hot_ids only applies to FaBC")
        w = 32 if codec == "raw" else _width(nodes)
        payload = a.astype("<u4", copy=False).tobytes() if codec == "raw" else _pack(a.ravel(), w)
        return _header(codec, n, l, nodes, 0, w) + payload
    if codec in ("fabc32", "fabc_packed"):
        hot = [int(x) for x in hot_ids]
        if len(set(hot)) != len(hot) or any(x < 0 or x >= nodes for x in hot):
            raise ValueError("hot_ids must be unique valid trainer IDs")
        k = len(hot)
        bw = 32 if codec == "fabc32" else _width(nodes)
        bpr = (l + 7) // 8
        ids = np.tile(np.asarray(hot, dtype="<u4"), n).tobytes()
        maps = bytearray(n * k * bpr)
        tails: list[int] = []
        offsets = [0]
        for row in a:
            for j, h in enumerate(hot):
                positions = np.flatnonzero(row == h)
                base = (len(offsets) - 1) * k * bpr + j * bpr
                for p in positions:
                    maps[base + int(p) // 8] |= 1 << (int(p) % 8)
            hotset = set(hot)
            tails.extend(int(x) for x in row if int(x) not in hotset)
            offsets.append(len(tails))
        off = np.asarray(offsets, dtype="<u8").tobytes()
        tail = _pack(tails, bw) if codec == "fabc_packed" else np.asarray(tails, dtype="<u4").tobytes()
        return _header(codec, n, l, nodes, k, bw) + ids + bytes(maps) + off + tail
    if codec == "varint":
        if hot_ids:
            raise ValueError("hot_ids does not apply to varint")
        if rank_to_id is None:
            raise ValueError("rank_to_id is required for varint")
        ranks = np.asarray(rank_to_id)
        if ranks.ndim != 1 or ranks.dtype != np.uint32 or len(ranks) != nodes:
            raise ValueError("rank_to_id must be uint32 with node_count entries")
        inverse = {int(v): i for i, v in enumerate(ranks)}
        payload = bytearray(); offsets = [0]
        for row in a:
            for value in row:
                try:
                    payload.extend(_leb128(inverse[int(value)]))
                except KeyError as e:
                    raise ValueError("walk ID missing from rank_to_id") from e
            offsets.append(len(payload))
        return _header(codec, n, l, nodes, 0, 0) + np.asarray(offsets, dtype="<u8").tobytes() + bytes(payload)
    raise ValueError(f"unknown codec: {codec}")


def decode(blob: bytes, rank_to_id: np.ndarray | None = None) -> np.ndarray:
    codec, n, l, nodes, k, width, rest = _parse(blob)
    count = n * l
    if codec == "raw":
        need = count * 4
        if len(rest) != need:
            raise ValueError("invalid raw payload length")
        return np.frombuffer(rest, dtype="<u4").copy().reshape(n, l)
    if codec == "packed":
        return _unpack(rest, count, width).reshape(n, l)
    if codec == "varint":
        if rank_to_id is None:
            raise ValueError("rank_to_id is required for varint")
        ranks = np.asarray(rank_to_id)
        if ranks.ndim != 1 or ranks.dtype != np.uint32 or len(ranks) != nodes:
            raise ValueError("rank_to_id must be uint32 with node_count entries")
        if len(rest) < (n + 1) * 8:
            raise ValueError("truncated varint offsets")
        offsets = np.frombuffer(rest[: (n + 1) * 8], dtype="<u8")
        payload = rest[(n + 1) * 8 :]
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]) or int(offsets[-1]) != len(payload):
            raise ValueError("invalid varint offsets")
        out = np.empty((n, l), dtype=np.uint32)
        for i in range(n):
            pos = int(offsets[i]); end = int(offsets[i + 1])
            for j in range(l):
                value = 0; shift = 0
                while True:
                    if pos >= end:
                        raise ValueError("truncated varint")
                    byte = payload[pos]; pos += 1
                    value |= (byte & 127) << shift
                    if byte < 128:
                        break
                    shift += 7
                    if shift > 63:
                        raise ValueError("varint overflow")
                if value >= nodes:
                    raise ValueError("rank outside dictionary")
                out[i, j] = ranks[value]
            if pos != end:
                raise ValueError("varint sequence has extra items")
        return out
    bpr = (l + 7) // 8
    rec = n * k * 4 + n * k * bpr + (n + 1) * 8
    if len(rest) < rec:
        raise ValueError("truncated FaBC payload")
    hot = np.frombuffer(rest[: n * k * 4], dtype="<u4").reshape(n, k)
    maps_start = n * k * 4
    maps = rest[maps_start : maps_start + n * k * bpr]
    offsets_start = maps_start + n * k * bpr
    offsets = np.frombuffer(rest[offsets_start : offsets_start + (n + 1) * 8], dtype="<u8")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid FaBC offsets")
    tails_blob = rest[rec:]
    tail_count = int(offsets[-1])
    if codec == "fabc32":
        if len(tails_blob) != tail_count * 4:
            raise ValueError("invalid FaBC tail length")
        tails = np.frombuffer(tails_blob, dtype="<u4")
    else:
        tails = _unpack(tails_blob, tail_count, width)
    out = np.empty((n, l), dtype=np.uint32)
    for i in range(n):
        row = out[i]
        present = np.zeros(l, dtype=bool)
        cursor = 0
        for j in range(k):
            bm = maps[(i * k + j) * bpr : (i * k + j + 1) * bpr]
            for p in range(l):
                if bm[p // 8] & (1 << (p % 8)):
                    row[p] = hot[i, j]
                    present[p] = True
        for p in range(l):
            if not present[p]:
                if cursor >= int(offsets[i + 1] - offsets[i]):
                    raise ValueError("FaBC bitmap/tail mismatch")
                row[p] = tails[int(offsets[i]) + cursor]
                cursor += 1
        if cursor != int(offsets[i + 1] - offsets[i]):
            raise ValueError("FaBC tail has unused items")
    if len(tails_blob) != ((tail_count * width + 7) // 8 if codec == "fabc_packed" else tail_count * 4):
        raise ValueError("invalid FaBC trailing bytes")
    return out
