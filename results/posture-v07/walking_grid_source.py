import sys,json,multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'training'))
from clearance_policy import WalkingPolicy
from evaluate_motion import trial,session,COMMANDS
p=np.array(json.loads(Path('results/posture-v07/policies/sitstand/parameters.json').read_text())['stand_parameters'])
def job(scale):
 actor=WalkingPolicy(session('results/precision-v06/policies/selected/policy.onnx'),p*scale)
 rows=[trial(actor,COMMANDS[c],seed,8,revision='v07',noise=seed%2==1,randomization=seed%2==1) for c in ['stop','forward','backward','turn_left','curve_left'] for seed in [0,1]]
 error=np.mean([np.mean(r['mean_absolute_twist_error']) for r in rows]);success=sum(r['survived'] and r['payload_selfcontact_fraction']==0 for r in rows)
 return dict(scale=scale,successes=success,fitness=success-5*error,error=float(error),trials=rows)
if __name__=='__main__':
 with mp.get_context('spawn').Pool(4) as pool:rows=pool.map(job,[0.,.25,.5,.75,1.])
 Path('results/posture-v07/walking-bias-grid.json').write_text(json.dumps(dict(training_seeds=[0,1],randomized_seeds=[1],candidates=rows),indent=2)+'\n')
 for r in rows:print(r['scale'],r['successes'],r['error'],flush=True)
