"""Limit local candidate experiments to three concurrent CPU processes."""
import argparse,fcntl,os,pathlib,subprocess,tempfile,time
p=argparse.ArgumentParser();p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command
if cmd and cmd[0]=='--':cmd=cmd[1:]
if not cmd:raise SystemExit('Expected command after --')
root=pathlib.Path(tempfile.gettempdir())/'alien-candidate-compute';root.mkdir(exist_ok=True)
handles=[(root/f'slot{i}.lock').open('w') for i in range(3)]
while True:
 for h in handles:
  try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:continue
  env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
  try:raise SystemExit(subprocess.call(cmd,env=env))
  finally:fcntl.flock(h,fcntl.LOCK_UN)
 time.sleep(.5)
