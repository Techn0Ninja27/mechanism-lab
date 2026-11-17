import unittest
from generation.contract_molecules import contract_connection
from tests import test_interface_contracts as fixtures

class ContractMoleculeTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ContractTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
    def connect(self,registry):
        return contract_connection(registry,self.fixture.pack,'one','tensor','two','tensor',
            mechanism='Pass declared tensor',assumptions=['Declared tensor meaning holds'],falsifiable_prediction='Consumer accepts returned tensor')
    def test_matching_declarations_form_only_a_hypothesis(self):
        r=self.connect(self.fixture.registry())
        self.assertIsNotNone(r['molecule'])
        self.assertEqual(r['molecule']['status'],'combination')
        self.assertFalse(r['molecule']['scientific_novelty_claim'])
        self.assertFalse(r['executable_compatibility_proven'])
    def test_lower_bound_cannot_silently_become_exact(self):
        reg=self.fixture.registry();reg['contracts'][0]['outputs'][0]['representation']='lower_bound'
        reg['contracts'][1]['inputs'][0]['representation']='exact'
        r=self.connect(reg);self.assertIsNone(r['molecule']);self.assertEqual(len(r['atoms']),2)
        self.assertEqual(r['assessment']['mismatches'][0]['field'],'representation')
    def test_missing_auxiliary_input_blocks_direct_connection(self):
        reg=self.fixture.registry();reg['contracts'][1]['inputs'].append(dict(self.fixture.port,id='query'))
        self.assertIsNone(self.connect(reg)['molecule'])
    def test_tampered_source_quote_does_not_produce_atoms(self):
        reg=self.fixture.registry();reg['contracts'][0]['evidence'][0]['quote']='invented'
        with self.assertRaises(ValueError):self.connect(reg)

if __name__=='__main__':unittest.main()
