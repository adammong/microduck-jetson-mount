import sys,json,multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'training'))
from evaluate_motion import trial,session
from evaluate_transitions import SlewedPolicy,SCHEDULE

def job(args):
 name,path,seed,acceleration=args
 p=SlewedPolicy(session(path));p.limiter.acceleration=np.array(acceleration,np.float32)
 r=trial(p,[0,0,0],seed,69,schedule=SCHEDULE);r['actor']=name;r['acceleration']=acceleration;return r
if __name__=='__main__':
 old='results/motion-v06/policies/selected/policy.onnx';new='results/precision-v06/policies/selected/policy.onnx'
 jobs=[('old',old,s,[.5,.3,1.5]) for s in range(650,654)]+[('new',new,s,a) for a in [[.25,.2,1.],[.15,.1,.8],[.1,.08,.6]] for s in range(650,654)]
 with mp.get_context('spawn').Pool(4) as pool:rows=pool.map(job,jobs)
 Path('results/precision-v06/ramp-study.json').write_text(json.dumps(rows,indent=2)+'\n')
 for r in rows:print(r['actor'],r['seed'],r['acceleration'],r['survived'],r['payload_selfcontact_fraction'],flush=True)
