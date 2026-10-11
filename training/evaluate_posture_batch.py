"""Independent held-out posture trials, including replayed physical falls."""
import argparse,json,multiprocessing as mp
from pathlib import Path
from evaluate_motion import session
from evaluate_posture import trial
from posture_policy import PosturePolicy,STAND,SIT
from recovery_policy import RecoveryPolicy
from seated_policy import SeatedPolicy
actor=None
def load(path):
    path=Path(path)
    if str(path)=='limp':return 'limp'
    if path.suffix=='.onnx':return session(path)
    data=json.loads(path.read_text());p=data['parameters']
    if data.get('method')=='gravity_conditioned_recovery':return RecoveryPolicy(session(STAND),p)
    if data.get('method')=='neutral_head_sitstand':
        from clearance_policy import SitStandPolicy
        return SitStandPolicy(data.get('stand_parameters'),data.get('sit_parameters'))
    if data.get('method')=='seated_target':return SeatedPolicy(session(STAND),p)
    return PosturePolicy(session(STAND),session(SIT),p if data['stage']=='recovery' else data['stand_parameters'],p if data['stage']=='sitstand' else None)
def init(path):
    global actor
    actor=load(path)
def job(args):return trial(actor,**args)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--poses',nargs='+',choices=['stand','sitstand','back','front','left','right'],default=['stand','back','front','left','right'])
    ap.add_argument('--seed-start',type=int,default=100);ap.add_argument('--seeds',type=int,default=8);ap.add_argument('--noise-dr',action='store_true')
    ap.add_argument('--payload-scales',nargs='+',type=float,default=[1.]);ap.add_argument('--fall-recordings',type=Path);ap.add_argument('--revision',default='v06');a=ap.parse_args()
    if a.fall_recordings:
        falls=json.loads(a.fall_recordings.read_text())['fall_poses']
        jobs=[dict(pose=f['requested_tip'],seed=a.seed_start+i,randomized=a.noise_dr,payload_scale=scale,replay=f['qpos']) for i,f in enumerate(falls) for scale in a.payload_scales]
    else:jobs=[dict(pose=p,seed=s,randomized=a.noise_dr,payload_scale=scale) for p in a.poses for s in range(a.seed_start,a.seed_start+a.seeds) for scale in a.payload_scales]
    for item in jobs:item['revision']=a.revision
    rows=[];a.out.parent.mkdir(parents=True,exist_ok=True)
    with mp.get_context('spawn').Pool(4,initializer=init,initargs=(a.policy,)) as pool:
        for row in pool.imap(job,jobs):
            rows.append(row);a.out.write_text(json.dumps(dict(policy=a.policy,fall_recordings=str(a.fall_recordings) if a.fall_recordings else None,trials=rows),indent=2)+'\n')
            print(row['pose'],row['seed'],row['payload_scale'],row['recovered'],row['sitstand_passed'],flush=True)
if __name__=='__main__':main()
