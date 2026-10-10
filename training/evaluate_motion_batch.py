from pathlib import Path
import sys,json,argparse,multiprocessing as mp
sys.path.insert(0,str(Path.cwd()/'training'))
import numpy as np
from evaluate_motion import trial,session,COMMANDS
from evaluate_transitions import SlewedPolicy
from paths import ROOT
_policy=None

def init(path):
 global _policy
 _policy=session(path)
def job(args):
 name,cmd,seed,seconds,dr,slew=args
 r=trial(SlewedPolicy(_policy) if slew else _policy,cmd,seed,seconds,noise=dr,randomization=dr);r['task']=name;r['command_slew']=slew;return r
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--out',required=True);ap.add_argument('--noise-dr',action='store_true');ap.add_argument('--seeds',type=int,default=8);ap.add_argument('--seed-start',type=int,default=20);ap.add_argument('--seconds',type=float,default=30);ap.add_argument('--extra',action='store_true');ap.add_argument('--tasks',nargs='+');ap.add_argument('--slew',action='store_true');a=ap.parse_args()
 tasks=dict(COMMANDS)
 if a.extra:
  tasks.update(forward_half=[.1,0,0],backward_half=[-.075,0,0],left_half=[0,.05,0],right_half=[0,-.05,0],turn_left_half=[0,0,.2],turn_right_half=[0,0,-.2],forward_curve_left=[.2,0,.4],forward_curve_right=[.2,0,-.4],backward_curve_left=[-.15,0,.4],backward_curve_right=[-.15,0,-.4])
 if a.tasks:tasks={name:tasks[name] for name in a.tasks}
 jobs=[(name,cmd,seed,a.seconds,a.noise_dr,a.slew) for name,cmd in tasks.items() for seed in range(a.seed_start,a.seed_start+a.seeds)]
 with mp.get_context('spawn').Pool(4,initializer=init,initargs=(a.policy,)) as pool:rows=pool.map(job,jobs)
 Path(a.out).write_text(json.dumps(dict(policy=a.policy,revision='v06',noise_dr=a.noise_dr,command_slew=a.slew,trials=rows),indent=2)+'\n')
 for name in tasks:
  group=[r for r in rows if r['task']==name]
  print(name,sum(r['survived'] for r in group),'/',len(group),'twist',np.round(np.mean([r['sustained_twist'] for r in group],axis=0),3).tolist(),'contacts',max(r['payload_selfcontact_fraction'] for r in group),flush=True)
