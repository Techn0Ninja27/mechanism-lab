import copy,unittest
from scripts.extract_operational_notes import validate

class OperationalNotesTests(unittest.TestCase):
    def setUp(self):
        self.paper={'source_id':'arxiv:0000.00001','abstract':'The estimator returns an accuracy estimate. Accuracy improved by 2 percent.',
                    'source_hash':'a'*64,'url':'https://example.test'}
        self.note={'source_id':self.paper['source_id'],'operation':'Estimate accuracy','operation_quote':'estimator returns an accuracy estimate',
                   'inputs':[],'outputs':[{'name':'accuracy estimate','quote':'accuracy estimate'}],
                   'reported_outcomes':[{'name':'measured improvement','quote':'Accuracy improved by 2 percent.'}],
                   'unknowns':['Estimator input and representation are unspecified.'],'ready_for_composition':False}
    def test_metric_valued_artifact_is_not_confused_with_reported_result(self):
        result=validate(self.note,self.paper)
        self.assertEqual(result['outputs'][0]['name'],'accuracy estimate')
        self.assertFalse(result['train_eligible']);self.assertFalse(result['ready_for_composition'])
    def test_invented_evidence_or_promotion_fails(self):
        for field,value in [('ready_for_composition',True),('operation_quote','invented'),('source_id','other')]:
            n=copy.deepcopy(self.note);n[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate(n,self.paper)
        n=copy.deepcopy(self.note);n['inputs']=[{'name':'input','quote':'not in the source'}]
        with self.assertRaises(ValueError):validate(n,self.paper)
    def test_abstention_and_bounds(self):
        n=copy.deepcopy(self.note);n['outputs']=[];validate(n,self.paper)
        n['unknowns']=[]
        with self.assertRaises(ValueError):validate(n,self.paper)
        n=copy.deepcopy(self.note);n['outputs']*=5
        with self.assertRaises(ValueError):validate(n,self.paper)

if __name__=='__main__':unittest.main()
