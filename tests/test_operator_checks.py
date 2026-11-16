import unittest
from generation.operator_checks import check_chain

class OperatorCheckTests(unittest.TestCase):
    def test_real_lossless_molecule(self):
        result=check_chain(['for','lzma'],[-5,0,10]*10)
        self.assertTrue(result['round_trip']);self.assertEqual(result['input_count'],30)
        self.assertFalse(result['scientific_novelty_claim']);self.assertFalse(result['macro_atom_promoted'])
        self.assertEqual(len(result['molecule']['atoms']),2)
    def test_bounded_types_and_known_registry(self):
        for ids,values in [(['unknown'],[1]),(['varint'],[1]),(['delta'],[1]),(['raw64'],[True]),(['raw64'],[1.2]),(['raw64'],[1<<63]),(['raw64'],[0]*4097),(['delta']*5,[1])]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):check_chain(ids,values)
