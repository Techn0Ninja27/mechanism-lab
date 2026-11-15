"""Central validation/freeze/final runner for reviewed estimator plugins."""
import argparse,hashlib,importlib.util,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SEEDS=(9401,9403,9407)
FINAL_CONDITIONS=('mnar_own_feature','mnar_reversed_sign','block_mcar_2_8','block_mcar_8_14')
VALIDATION_CONDITIONS=('mcar_0.2','mcar_0.4','mar_anchor_x0')
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def fixture_source():return ROOT/'scripts/missingness_shift_fixture.py'
def get_fixture(seed,mode):
 from scripts.missingness_shift_fixture import load_fixture
 return load_fixture(seed,mode)
def score(y,p):
 from scripts.missingness_shift_fixture import evaluate
 return {'ap':evaluate(y,p),'brier':evaluate(y,p,'brier')}
def read_config(path):
 c=json.loads(pathlib.Path(path).read_text());vs=c.get('variants',[])
 if not vs or len({v['id'] for v in vs})!=len(vs) or {v['role'] for v in vs}!={'candidate','control'}:raise ValueError('Need unique candidate/control variants')
 plugin=pathlib.Path(c['plugin']);plugin=plugin if plugin.is_absolute() else ROOT/plugin
 hashes={'config':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),'plugin':sha(plugin),'fixture':sha(fixture_source())}
 return c,plugin,hashes
def load_plugin(path):
 spec=importlib.util.spec_from_file_location('reviewed_study_plugin',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def run(config_path,out,mode):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);config,plugin_path,hashes=read_config(config_path)
 path=out/('validation.json' if mode=='validation' else 'results.json')
 if path.exists():raise FileExistsError('Refusing to overwrite '+str(path))
 selected=config['variants'];freeze=None
 if mode=='final':
  freeze=json.loads((out/'freeze.json').read_text())
  if freeze['source_hashes']!=hashes or freeze['validation_sha256']!=sha(out/'validation.json'):raise ValueError('Frozen artifacts changed')
  selected=[v for v in selected if v['id'] in freeze['selection'].values()]
 module=load_plugin(plugin_path);record={'mode':mode,'source_hashes':hashes,'rows':[],'errors':[],'status':'running'}
 with path.open('x') as f:json.dump(record,f)
 try:
  for seed in SEEDS:
   fixture=get_fixture(seed,mode);cases=fixture.validation_cases if mode=='validation' else fixture.final_cases
   for variant in selected:
    started=time.perf_counter();estimator=module.build(variant.get('params',{}),seed)
    estimator.fit(fixture.train_x.copy(),fixture.train_y.copy());fit_seconds=time.perf_counter()-started
    for condition,(x,y) in cases.items():
     t=time.perf_counter();p=np.asarray(estimator.predict_proba(x.copy()))
     if p.ndim==2:p=p[:,1]
     if p.ndim!=1 or len(p)!=len(y) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
     # Labels become visible only to central scoring, after prediction.
     record['rows'].append({'seed':seed,'variant':variant['id'],'role':variant['role'],'condition':condition,**score(y,p),'fit_seconds':fit_seconds,'predict_seconds':time.perf_counter()-t})
  record['status']='complete'
  if mode=='final':
   names=freeze['selection'];values={k:mean(record['rows'],v,FINAL_CONDITIONS) for k,v in names.items()}
   clean={k:mean(record['rows'],v,('clean',)) for k,v in names.items()}
   passed=freeze['validation_passed'] and values['candidate']-values['control']>=.005 and clean['candidate']-clean['control']>=-.01
   record['decision']={'status':'supported_pilot' if passed else 'no_gain','primary':values,'clean':clean,'validation_passed':freeze['validation_passed'],'novelty_established':False,'human_adjudicated':False}
 except Exception as exc:
  record.update(status='failed');record['errors'].append(type(exc).__name__+': '+str(exc));raise
 finally:path.write_text(json.dumps(record,indent=2))
 return record

def mean(rows,variant,conditions):
 selected=[r for r in rows if r['variant']==variant and r['condition'] in conditions]
 if len(selected)!=len(SEEDS)*len(conditions):raise ValueError('Incomplete metric coverage')
 return float(np.mean([r['ap'] for r in selected]))
def freeze(config_path,out):
 out=pathlib.Path(out);path=out/'freeze.json'
 if path.exists() or (out/'results.json').exists():raise FileExistsError('Freeze/final already exists')
 config,plugin,hashes=read_config(config_path);validation=json.loads((out/'validation.json').read_text())
 if validation['status']!='complete' or validation['source_hashes']!=hashes:raise ValueError('Validation incomplete or source changed')
 choices={}
 for role in ['candidate','control']:
  variants=[v['id'] for v in config['variants'] if v['role']==role]
  choices[role]=max(sorted(variants),key=lambda v:mean(validation['rows'],v,VALIDATION_CONDITIONS))
 delta=mean(validation['rows'],choices['candidate'],VALIDATION_CONDITIONS)-mean(validation['rows'],choices['control'],VALIDATION_CONDITIONS)
 clean=mean(validation['rows'],choices['candidate'],('clean',))-mean(validation['rows'],choices['control'],('clean',))
 record={'source_hashes':hashes,'validation_sha256':sha(out/'validation.json'),'selection':choices,'validation_passed':delta>=.005 and clean>=-.01,'thresholds':{'minimum_ap_gain':.005,'maximum_clean_loss':.01},'final_primary_conditions':FINAL_CONDITIONS,'note':'Freezing authorizes one descriptive final evaluation; a failed validation screen cannot be reversed by final scores.'}
 with path.open('x') as f:json.dump(record,f,indent=2)
 return record
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['validation','freeze','final']);p.add_argument('--config',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();r=freeze(a.config,a.output) if a.mode=='freeze' else run(a.config,a.output,a.mode);print(json.dumps({'mode':a.mode,'status':r.get('status','frozen'),'selection':r.get('selection'),'decision':r.get('decision')},indent=2))
