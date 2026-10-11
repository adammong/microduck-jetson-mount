"""Long start/stop/directional sequences on the mounted terrain model."""
import argparse,json,multiprocessing as mp
from pathlib import Path
from evaluate_motion import trial,session
from evaluate_transitions import SCHEDULE,SlewedPolicy
_actor=None

def init(path):
 global _actor
 _actor=session(path)
def job(args):
 terrain,seed,terrain_seed,dr=args
 row=trial(SlewedPolicy(_actor),[0,0,0],seed,SCHEDULE[-1][0],noise=dr,randomization=dr,schedule=SCHEDULE,revision='v07',terrain=terrain,terrain_seed=terrain_seed)
 row['command_slew']=True;row['passed']=row['survived'] and row['nonfoot_ground_fraction']==0 and row['payload_selfcontact_fraction']==0
 return row
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 jobs=[(terrain,seed+i,terrain_seed+i,dr) for terrain in ['flat','bumps_5mm','uphill_3deg','mixed_5mm_3deg'] for seed,terrain_seed,dr in [(400,600,False),(420,620,True)] for i in range(2)]
 with mp.get_context('spawn').Pool(4,initializer=init,initargs=(a.policy,)) as pool:rows=pool.map(job,jobs)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(policy=a.policy,schedule=SCHEDULE,trials=rows),indent=2)+'\n');print('Sequences',sum(r['passed'] for r in rows),'/',len(rows))
