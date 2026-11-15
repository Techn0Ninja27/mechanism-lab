"""Validate grounded reconstructed proposals without imposing arbitrary ID names."""
import copy,re

def canonical(value):
 s=value.removeprefix('arxiv:')
 return re.sub(r'v\d+$','',s) if re.fullmatch(r'\d{4}\.\d{4,5}(?:v\d+)?',s) else value

def validate(proposal,source_ids):
 d=copy.deepcopy(proposal);aliases={canonical(s):s for s in source_ids}
 primitives=d.get('proposed_primitives',[])
 if not isinstance(primitives,list):raise ValueError('proposed_primitives must be a list')
 names=set()
 for p in primitives:
  name=p.get('id')
  if not isinstance(name,str) or not name.strip() or name in names or canonical(name) in aliases:raise ValueError('Invalid/duplicate proposed primitive identity')
  if not isinstance(p.get('definition'),str) or not p['definition'].strip() or p.get('novelty_established') is not False:raise ValueError('Primitive needs definition and explicit unvalidated novelty')
  support=p.get('source_support_ids')
  if not isinstance(support,list) or not support:raise ValueError('Primitive needs grounded source support')
  if any(canonical(s) not in aliases for s in support):raise ValueError('Unknown source support')
  p['source_support_ids']=[aliases[canonical(s)] for s in support];names.add(name)
 def resolve(s):
  if s in names:return s
  if canonical(s) in aliases:return aliases[canonical(s)]
  raise ValueError('Unknown component '+str(s))
 if not isinstance(d.get('components'),list) or not d['components']:raise ValueError('Missing components')
 d['components']=[resolve(s) for s in d['components']]
 if not isinstance(d.get('relations'),list):raise ValueError('Missing relations')
 for edge in d['relations']:
  edge['source']=resolve(edge['source']);edge['target']=resolve(edge['target'])
  for endpoint in [edge['source'],edge['target']]:
   if endpoint not in d['components']:d['components'].append(endpoint)
  if not edge.get('type') or not edge.get('counterfactual_prediction'):raise ValueError('Unspecified relation')
 return d
