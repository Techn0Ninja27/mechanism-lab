"""Exact integer checks for the target in the accompanying analytical proof."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

HERE=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(output):
    if output.exists(): raise FileExistsError('Preserve the previous result')
    protocol=json.loads((HERE/'protocol.json').read_text())
    q=protocol['q']
    square=[[sum(q[i][k]*q[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    checks={'symmetric':q==list(map(list,zip(*q))),
            'q_squared_equals_3q':square==[[3*x for x in row] for row in q],
            'q_annihilates_ones':all(sum(row)==0 for row in q),
            'q_trace_is_6':sum(q[i][i] for i in range(3))==6}
    result={'schema':'pera-rank-counterexample-result.v1','status':'exact_target_checks_pass' if all(checks.values()) else 'checks_failed',
            'checks':checks,'protocol':protocol,'protocol_sha256':sha(HERE/'protocol.json'),
            'script_sha256':sha(Path(__file__)),'python':platform.python_version(),
            'target_singular_values_exact':[1,1,0],
            'claimed_error_upper_bound_for_target':0,
            'analytical_error_lower_bound':'1/sqrt(3)',
            'analytical_error_lower_bound_decimal':1/math.sqrt(3),
            'proof_status':'written_analytical_argument_reviewed_by_native_agents_not_formal_verification',
            'novelty_established':False,'task_usefulness_established':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    if not all(checks.values()): raise AssertionError('Exact target checks failed; result retained')
    return {'output':str(output),'checks':checks}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(check(p.parse_args().output)))
