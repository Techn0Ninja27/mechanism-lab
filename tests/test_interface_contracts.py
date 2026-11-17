import copy, tempfile, unittest
from pathlib import Path
from generation.sourcepack import build_source_pack
from generation.interface_contracts import validate_contracts,assess_connection,inspect_contracts,SCHEMA

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name)
        (root/'source.txt').write_text('The operation returns a tensor. The consumer accepts a tensor.')
        self.pack=build_source_pack({'sources':[{'id':'s','title':'Source','url':'https://example.test','path':'source.txt','evidence_level':'full_text'}]},root)
        chunk=self.pack['chunks'][0]
        self.port={'id':'tensor','artifact':'tensor','representation':'dense','granularity':'sample','phase':'offline','role':'artifact','exposure':'required','conditions':[],'evidence_ids':['e']}
        self.card={'id':'one','source_id':'s','interpretation':'curated_not_formally_verified',
                   'evidence':[{'id':'e','chunk_id':chunk['id'],'source_sha256':chunk['source_sha256'],'quote':'returns a tensor'}],
                   'inputs':[copy.deepcopy(self.port)],'outputs':[dict(self.port,exposure='returned')]}
        self.other=copy.deepcopy(self.card);self.other['id']='two'
    def registry(self):return {'schema':SCHEMA,'contracts':[copy.deepcopy(self.card),copy.deepcopy(self.other)]}
    def assess(self,r):return assess_connection(validate_contracts(r,self.pack),'one','tensor','two','tensor')
    def test_equal_declarations_are_not_a_proof(self):
        r=self.assess(self.registry());self.assertEqual(r['status'],'declared_match_unverified')
        self.assertFalse(r['executable_compatibility_proven']);self.assertFalse(r['novelty_established'])
    def test_unknown_fields_and_missing_conditions_are_not_wildcards(self):
        r=self.registry();r['contracts'][0]['outputs'][0]['representation']=None
        r['contracts'][1]['inputs'][0]['conditions']=['unbiased']
        result=self.assess(r);self.assertEqual(result['status'],'unresolved_requirements')
        self.assertEqual(result['unknown_fields'],['representation']);self.assertEqual(result['unestablished_conditions'],['unbiased'])
    def test_intermediate_and_other_inputs_are_explicit(self):
        r=self.registry();r['contracts'][0]['outputs'][0]['exposure']='intermediate'
        r['contracts'][1]['inputs'].append(dict(self.port,id='query'))
        result=self.assess(r);self.assertEqual(result['status'],'requires_bridge')
        self.assertEqual(result['other_required_inputs'],['query']);self.assertTrue(result['adapter_or_exposure_required'])
    def test_metrics_can_feed_metric_consumers_but_not_artifact_ports(self):
        r=self.registry();r['contracts'][0]['outputs'][0]['role']='metric'
        self.assertEqual(self.assess(r)['mismatches'][0]['field'],'role')
        r['contracts'][1]['inputs'][0]['role']='metric'
        self.assertEqual(self.assess(r)['status'],'declared_match_unverified')
    def test_evidence_and_duplicate_ids_fail_closed(self):
        for field,value in [('quote','invented'),('source_sha256','f'*64),('chunk_id',[])]:
            r=self.registry();r['contracts'][0]['evidence'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_contracts(r,self.pack)
        r=self.registry();r['contracts'][1]['id']='one'
        self.assertEqual(inspect_contracts(r,self.pack)['matched_count'],0)
    def test_unknown_sentinels_are_not_concrete_types(self):
        r=self.registry();r['contracts'][0]['outputs'][0]['phase']='unknown'
        with self.assertRaisesRegex(ValueError,'JSON null'):validate_contracts(r,self.pack)

    def test_partial_source_coverage_keeps_valid_card(self):
        r=self.registry();r['contracts'][1]['source_id']='absent'
        result=inspect_contracts(r,self.pack);self.assertEqual(result['matched_count'],1);self.assertTrue(result['warnings'])
        r['contracts'][1]['source_id']=[]
        self.assertEqual(inspect_contracts(r,self.pack)['matched_count'],1)

if __name__=='__main__':unittest.main()
