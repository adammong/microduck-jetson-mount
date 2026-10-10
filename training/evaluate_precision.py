"""Paired slow-command/start-stop and physical shove benchmarks for v06."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from evaluate_motion import trial,session,COMMANDS
from evaluate_transitions import SlewedPolicy
from motion_policy import NAMES,REFERENCES

def cases(suite='slow'):
    if suite=='all':return cases('slow')+cases('push')+cases('regression')
    if suite=='slow':
        return [dict(name=f'{name}_{ratio:g}',command=(cmd*ratio).tolist(),
            schedule=[(3.,[0.,0.,0.]),(11.,(cmd*ratio).tolist()),(16.,[0.,0.,0.])],seconds=16.,pushes=[])
            for name,cmd in zip(NAMES,REFERENCES) for ratio in [.25,.5,.75]]
    if suite=='push':
        return [dict(name=f'{name}_{force:g}N_{direction}',command=cmd,schedule=None,seconds=16.,
            pushes=[(5.,.12,(np.array(axis)*force).tolist()),(10.,.12,(-np.array(axis)*force).tolist())])
            for name,cmd in [('stand',[0.,0.,0.]),('forward',[.2,0.,0.]),('left',[0.,.1,0.])]
            for force in [2.,4.,6.] for direction,axis in [('x',[1.,0.,0.]),('y',[0.,1.,0.])]]
    return [dict(name=name,command=cmd,schedule=None,seconds=20.,pushes=[]) for name,cmd in COMMANDS.items()]

def measure(policy,case,seed,dr=False,payload_scale=1.,render=None):
    r=trial(SlewedPolicy(policy),case['command'],seed,case['seconds'],render=render,
        noise=dr,randomization=dr,schedule=case['schedule'],pushes=case['pushes'],payload_scale=payload_scale,trace=True)
    a=np.array(r.pop('trace')['actual_twist']);r['case']=case['name']
    r['safe']=bool(r['survived'] and r['nonfoot_ground_fraction']==0 and r['payload_selfcontact_fraction']==0)
    if case['schedule']:
        moving=a[250:550];c=np.array(case['command']);r['moving_twist']=moving.mean(axis=0).tolist() if len(moving) else [0.,0.,0.]
        r['tracking_error']=float(np.linalg.norm((np.array(r['moving_twist'])-c)/[.04,.025,.12]))
        stop=a[550:];r['stop_residual_twist']=stop[-100:].mean(axis=0).tolist() if len(stop) else [0.,0.,0.]
        r['stop_distance_m']=float(np.linalg.norm(stop[:,:2],axis=1).sum()*.02)
        r['settle_s']=None
        for i in range(max(0,len(stop)-49)):
            if np.all(np.linalg.norm(stop[i:i+50,:2],axis=1)<.02) and np.all(abs(stop[i:i+50,2])<.12):
                r['settle_s']=i*.02;break
    else:
        r['tracking_error']=float(np.linalg.norm((np.array(r['sustained_twist'])-case['command'])/[.08,.05,.25]))
    r['post_push_errors']=[]
    for start,duration,force in case['pushes']:
        recovery=a[round((start+duration+1.5)/.02):round((start+duration+3.5)/.02)]
        error=float(np.linalg.norm((recovery.mean(axis=0)-case['command'])/[.08,.05,.25])) if len(recovery) else 20.
        r['post_push_errors'].append(error)
    return r

_policy=None
def init(path):
    global _policy
    _policy=session(path)
def job(args):
    case,seed,dr,scale=args
    return measure(_policy,case,seed,dr,scale)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--suite',choices=['slow','push','regression','all'],default='slow');ap.add_argument('--seeds',type=int,default=4)
    ap.add_argument('--seed-start',type=int,default=0);ap.add_argument('--noise-dr',action='store_true')
    ap.add_argument('--payload-scales',nargs='+',type=float,default=[1.]);ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();jobs=[(c,s,a.noise_dr,scale) for c in cases(a.suite) for s in range(a.seed_start,a.seed_start+a.seeds) for scale in a.payload_scales]
    rows=[];a.out.parent.mkdir(parents=True,exist_ok=True)
    with mp.get_context('spawn').Pool(a.workers,initializer=init,initargs=(a.policy,)) as pool:
        for r in pool.imap(job,jobs):
            rows.append(r);a.out.write_text(json.dumps(dict(policy=a.policy,suite=a.suite,trials=rows),indent=2)+'\n')
            print(r['case'],r['seed'],r['safe'],np.round(r.get('moving_twist',r['sustained_twist']),3).tolist(),flush=True)
if __name__=='__main__':main()
