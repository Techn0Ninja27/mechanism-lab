import unittest
from pathlib import Path
from scripts import prepare_chunk_representations as prep
class PromptContractTests(unittest.TestCase):
 def test_constraints_are_explicit_in_both_arms(self):
  pack=prep._pack(Path(__file__).resolve().parents[1]/'results/gnn-useful-proposal-v1/pack.json')
  for kind in ('atoms','distillations'):
   prompt=prep._prompt(pack,kind).split('CANONICAL SOURCES')[0]
   for rule in ('exactly the one requested key','unique nonempty strings','zero to eight','distinct within that record','belong to the record','nonempty exact substring','18000 UTF-8 bytes','24000-byte canonical-result cap'):
    self.assertIn(rule,prompt)
if __name__=='__main__': unittest.main()
