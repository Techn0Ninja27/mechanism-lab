"""Harmonize existing study metrics without selecting/tuning any model."""
import json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
PAIRS={'abstracts-0':('admib','completed_aug_unweighted'),'abstracts-1':('candidate_mixed_k8','random_mask_augmentation_hgb_200_trees'),'atoms-0':('stable_gate','single_conditional_mean'),'atoms-1':('candidate','native_nan_hgb'),'molecules-0':('candidate_selfmask_dro','randommask_dro_matched'),'molecules-1':('anchor_full','native_nan_duplicate')}
CONDITIONS=('mnar_own_feature','mnar_reversed_sign','block_mcar_2_8','block_mcar_8_14')
def rows(case,d):
 out=[]
 def add(seed,method,condition,r):
  if 'ap' in r:out.append({'seed':int(seed),'method':method,'condition':condition,'ap':r['ap'],'brier':r.get('brier')})
 if case=='abstracts-0':
  for r in d['records']:add(r['seed'],r['arm'],r['condition'],r)
 elif case=='abstracts-1':
  for sd in d['results']:
   for method,cases in sd['methods'].items():
    for condition,r in cases.items():
     if isinstance(r,dict):add(sd['seed'],method,condition,r)
 elif case=='atoms-0':
  for sd in d['records']:
   for condition,methods in sd['results'].items():
    for method,r in methods.items():add(sd['seed'],method,condition,r)
 elif case=='atoms-1':
  for seed,sd in d['seeds'].items():
   for condition,methods in sd['metrics'].items():
    for method,r in methods.items():add(seed,method,condition,r)
 elif case=='molecules-0':
  for sd in d['records']:
   for method,md in sd['arms'].items():
    for condition,r in md['scores'].items():add(sd['seed'],method,condition,r)
 else:
  for sd in d['seed_results']:
   for method,md in sd['methods'].items():
    for condition,r in md['cases'].items():add(sd['seed'],method,condition,r)
 return out

def main():
 root=ROOT/'reconstruction-studies';summary=[];allrows=[]
 for case,pair in PAIRS.items():
  data=json.loads((root/case/'results.json').read_text());normalized=rows(case,data);allrows.extend({'study':case,**r} for r in normalized)
  values={}
  for role,method in zip(['candidate','control'],pair):
   selected=[r for r in normalized if r['method']==method and r['condition'] in CONDITIONS]
   if len(selected)!=12:raise ValueError(f'{case}/{method}: expected12 rows, got{len(selected)}')
   values[role]=statistics.mean(r['ap'] for r in selected)
  summary.append({'id':case,'candidate_method':pair[0],'control_method':pair[1],'mean_ap':values,'delta':values['candidate']-values['control'],'conditions':CONDITIONS,'rows_per_method':12,'original_verdict':json.loads((root/case/'verdict.json').read_text())['status']})
 (root/'harmonized-results.json').write_text(json.dumps({'purpose':'Secondary descriptive harmonization to four new mask mechanisms; not a replacement for study preregistration, not a cross-arm causal effect estimate.','comparisons':summary,'rows':allrows},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
