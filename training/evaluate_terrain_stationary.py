"""Stand/turn after a cold start on actual rough/sloping ground."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from evaluate_motion import trial,session,COMMANDS
_actor=None

def job(args):
 label,path,terrain,task,seed,terrain_seed,dr=args
 global _actor
 if _actor is None or _actor._model_path!=path:_actor=session(path)
 cmd=COMMANDS[task];r=trial(_actor,cmd,seed,12,noise=dr,randomization=dr,schedule=[(2,[0,0,0]),(12,cmd)],revision='v07',terrain=terrain,terrain_seed=terrain_seed,terrain_spawn=[.7,.5])
 r['task']=task;r['passed']=r['survived'] and r['nonfoot_ground_fraction']==0 and r['payload_selfcontact_fraction']==0
 v=np.asarray(r['sustained_twist']);r['direction_followed']=bool(np.all(abs(v)<=np.array([.04,.04,.12]))) if task=='stop' else bool(v[2]*cmd[2]>=.25*cmd[2]**2 and np.max(abs(v[:2]))<=.04)
 r['direction_qualified']=r['passed'] and r['direction_followed'];return label,r
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();policies=dict(baseline='results/posture-v07/policies/walking-adjusted/policy.onnx',candidate='results/terrain-v07/policies/candidate/policy.onnx')
 jobs=[(label,path,t,task,seed+i,terrain_seed+i,dr) for label,path in policies.items() for t in ['bumps_5mm','uphill_3deg','cross_left_3deg','mixed_5mm_3deg'] for task in ['stop','turn_left','turn_right'] for seed,terrain_seed,dr in [(800,1800,False),(820,1820,True)] for i in range(2)]
 with mp.get_context('spawn').Pool(4) as pool:rows=pool.map(job,jobs)
 data=dict(simulation_only=True,policies=policies,spawn_xy_m=[.7,.5],extra_initial_ground_clearance_m=.012,standing_warmup_s=2,trials={label:[r for name,r in rows if name==label] for label in policies})
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(data,indent=2)+'\n')
 for label,rr in data['trials'].items():print(label,'survives/contact-free',sum(r['passed'] for r in rr),'direction',sum(r['direction_qualified'] for r in rr),'/',len(rr))
