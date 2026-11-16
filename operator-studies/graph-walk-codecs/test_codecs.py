import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("walk_codecs", Path(__file__).with_name("codecs.py"))
codecs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codecs)


WALKS = np.array([[0, 5, 2, 5, 7], [7, 7, 1, 3, 0], [9, 4, 5, 8, 2]], dtype=np.uint32)


class CodecTests(unittest.TestCase):
  def test_all_codecs_roundtrip_sparse_ids(self):
    walks = WALKS
    ranks = np.array([7, 2, 9, 0, 5, 1, 3, 4, 6, 8], dtype=np.uint32)
    for name, hot, mapping in [("raw", [], None), ("packed", [], None),
                               ("fabc32", [5, 7], None), ("fabc_packed", [5, 7], None),
                               ("varint", [], ranks)]:
        blob = codecs.encode(walks, name, 10, hot, mapping)
        self.assertTrue(np.array_equal(codecs.decode(blob, mapping), walks))
        self.assertEqual(blob, codecs.encode(walks, name, 10, hot, mapping))


  def test_boundary_lengths_and_empty_tail(self):
    walks = np.array([[2] * 13, [3] * 13], dtype=np.uint32)
    ranks = np.array([2, 3, 0, 1], dtype=np.uint32)
    for name in ["raw", "packed", "fabc32", "fabc_packed", "varint"]:
      hot = [2, 3] if name.startswith("fabc") else []
      mapping = ranks if name == "varint" else None
      blob = codecs.encode(walks, name, 4, hot, mapping)
      self.assertTrue(np.array_equal(codecs.decode(blob, mapping), walks))


  def test_varint_leb128_boundaries(self):
    for value in [127, 128, 16383, 16384]:
      ranks = np.arange(value + 1, dtype=np.uint32)
      walks = np.array([[value]], dtype=np.uint32)
      blob = codecs.encode(walks, "varint", value + 1, rank_to_id=ranks)
      self.assertTrue(np.array_equal(codecs.decode(blob, ranks), walks))


  def test_invalid_and_truncated_streams(self):
    walks = WALKS
    ranks = np.arange(10, dtype=np.uint32)
    for name, hot, mapping in [("raw", [], None), ("packed", [], None),
                               ("fabc32", [5], None), ("fabc_packed", [5], None),
                               ("varint", [], ranks)]:
        blob = codecs.encode(walks, name, 10, hot, mapping)
        with self.assertRaises(ValueError):
            codecs.decode(blob[:-1], mapping)
        with self.assertRaises(ValueError):
            codecs.decode(blob + b"x", mapping)


  def test_varint_requires_external_dictionary(self):
    walks = WALKS
    blob = codecs.encode(walks, "varint", 10, rank_to_id=np.arange(10, dtype=np.uint32))
    with self.assertRaises(ValueError):
        codecs.decode(blob)


  def test_input_validation(self):
    walks = WALKS
    with self.assertRaises(ValueError):
        codecs.encode(walks.astype(np.uint64), "raw", 10)
    with self.assertRaises(ValueError):
        codecs.encode(walks, "packed", 5)
    with self.assertRaises(ValueError):
        codecs.encode(walks, "fabc32", 10, [1, 1])


if __name__ == "__main__":
    unittest.main()
