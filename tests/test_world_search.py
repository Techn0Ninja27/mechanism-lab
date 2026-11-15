import importlib.util,json,pathlib,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from world_search import plan,prompts,parse_json,Client

class WorldSearchTests(unittest.TestCase):
 def test_balanced_output_caps(self):
  p=plan();self.assertEqual(len(p['cases']),24)
  for arm in ['direct','random_seed','world']:self.assertEqual(sum(r['arm']==arm for r in p['cases']),8)
  self.assertEqual(sum(p['max_tokens_per_stage'])*24,p['max_output_tokens_total'])
 def test_world_roles_are_isolated(self):
  c={'problem':'SECRET_TARGET','arm':'world','seed_word':'UNIQUE_SEED'}
  self.assertNotIn('SECRET_TARGET',prompts(c,0,''));self.assertIn('UNIQUE_SEED',prompts(c,0,''))
  self.assertNotIn('UNIQUE_SEED',prompts(c,1,'laws'));self.assertIn('SECRET_TARGET',prompts(c,1,'laws'))
 def test_pending_call_is_not_retried(self):
  with tempfile.TemporaryDirectory() as d,patch('world_search.key',return_value='test'):
   client=Client(pathlib.Path(d),plan())
   with patch('world_search.urllib.request.urlopen',side_effect=TimeoutError('test')) as net:
    with self.assertRaises(TimeoutError):client.call({'id':'test'},0,'prompt')
    with self.assertRaises(ValueError):client.call({'id':'test'},0,'prompt')
    self.assertEqual(net.call_count,1)
 def test_budget_blocks_network(self):
  with tempfile.TemporaryDirectory() as d,patch('world_search.key',return_value='test'):
   config=plan();config['budget_usd']=0;client=Client(pathlib.Path(d),config)
   with patch('world_search.urllib.request.urlopen') as net:
    with self.assertRaises(ValueError):client.call({'id':'test'},0,'prompt')
    net.assert_not_called()
 def test_schema_missing_query_fails(self):
  with self.assertRaises(ValueError):parse_json('{}')
