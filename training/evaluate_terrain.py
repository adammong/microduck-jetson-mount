"""Paired terrain tests with independent terrain and physics random seeds."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from evaluate_motion import trial,session,COMMANDS
from terrain_env import TERRAINS

_actor=None

def init(policy):
    global _actor
    _actor=session(policy)

def job(args):
    terrain,task,seed,terrain_seed,seconds,randomized,scale=args
    row=trial(_actor,COMMANDS[task],seed,seconds,noise=randomized,randomization=randomized,payload_scale=scale,revision='v07',terrain=terrain,terrain_seed=terrain_seed)
    row['task']=task
    row['contact_free']=row['payload_selfcontact_fraction']==0 and row['nonfoot_ground_fraction']==0
    actual=np.asarray(row['sustained_twist']);command=np.asarray(row['command'])
    # All requested directions must make useful progress; survival cannot
    # silently count a stalled gait as successful terrain walking.
    row['direction_followed']=bool(np.all(np.where(command!=0,actual*command>=.25*command**2,abs(actual)<=np.array([.04,.04,.12]))))
    row['passed']=bool(row['survived'] and row['contact_free'] and row['direction_followed'])
    row['scaled_tracking_error']=float(np.mean(np.abs(np.asarray(row['mean_absolute_twist_error']))/np.array([.2,.1,.4])))
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--terrains',nargs='+',choices=list(TERRAINS),default=list(TERRAINS));ap.add_argument('--tasks',nargs='+',choices=list(COMMANDS),default=list(COMMANDS))
    ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--seed-start',type=int,default=200);ap.add_argument('--terrain-seed-start',type=int,default=500)
    ap.add_argument('--seconds',type=float,default=15);ap.add_argument('--noise-dr',action='store_true');ap.add_argument('--payload-scales',nargs='+',type=float,default=[1.]);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    jobs=[(terrain,task,a.seed_start+i,a.terrain_seed_start+i,a.seconds,a.noise_dr,scale) for terrain in a.terrains for task in a.tasks for i in range(a.seeds) for scale in a.payload_scales]
    with mp.get_context('spawn').Pool(a.workers,initializer=init,initargs=(a.policy,)) as pool:rows=pool.map(job,jobs)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(policy=a.policy,revision='v07',physics_seeds=list(range(a.seed_start,a.seed_start+a.seeds)),terrain_seeds=list(range(a.terrain_seed_start,a.terrain_seed_start+a.seeds)),trials=rows),indent=2)+'\n')
    for terrain in a.terrains:
        group=[r for r in rows if r['terrain']==terrain]
        print(terrain,'pass',sum(r['passed'] for r in group),'/',len(group),'survive',sum(r['survived'] for r in group),'error',round(np.mean([r['scaled_tracking_error'] for r in group]),3),flush=True)
if __name__=='__main__':main()
