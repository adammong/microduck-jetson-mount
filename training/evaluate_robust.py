"""Slope precision and actual applied-disturbance recovery benchmarks for v07."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from evaluate_motion import session,trial
from evaluate_transitions import SlewedPolicy

def cases(suite='all',training=False):
    if suite=='all':return sum([cases(s,training) for s in ('slope','push','trip_slip','regression')],[])
    if suite=='slope':
        return [dict(name=t+'_'+name,suite=suite,terrain=t,spawn=[.7,.5],command=c,events=[])
            for t in (['downhill_5deg','cross_left_3deg','uphill_3deg'] if training else ['uphill_3deg','downhill_3deg','uphill_5deg','downhill_5deg','cross_left_3deg','cross_right_3deg','mixed_5mm_3deg'])
            for name,c in [('forward',[.15,0,0]),('backward',[-.10,0,0]),('turn_left',[0,0,.4]),('turn_right',[0,0,-.4])]]
    if suite=='push':
        return [dict(name=f'{t}_{name}_{force:g}N_{axis_name}',suite=suite,terrain=t,spawn=None,command=c,
            events=[dict(kind='push',start=4.,duration=.12,force_body=(np.asarray(axis)*force).tolist()),dict(kind='push',start=8.,duration=.12,force_body=(-np.asarray(axis)*force).tolist())])
            for t in (['flat'] if training else ['flat','bumps_5mm'])
            for name,c in [('stand',[0,0,0]),('forward',[.2,0,0])]
            for force in ([6.] if training else [2.,4.,6.]) for axis_name,axis in [('x',[1,0,0]),('y',[0,1,0])]]
    if suite=='trip_slip':
        return [dict(name=f'{t}_snag_{foot}_{force:g}N',suite=suite,terrain=t,spawn=None,command=[.2,0,0],events=[dict(kind='snag',start=4.,duration=.08,window=1.,foot=foot,force_body=[-force,0,0])])
            for t in (['flat'] if training else ['flat','bumps_5mm']) for foot in ('left','right') for force in ([2.] if training else [1.,2.,4.])] + [dict(name=f'{t}_slip_mu{mu:g}',suite=suite,terrain=t,spawn=None,command=[.2,0,0],events=[dict(kind='slip',start=4.,duration=.5,friction=mu),dict(kind='slip',start=8.,duration=.5,friction=mu)]) for t in (['flat'] if training else ['flat','bumps_5mm']) for mu in ([.15] if training else [.1,.2,.35])]
    return [dict(name=t+'_'+name,suite=suite,terrain=t,spawn=None,command=c,events=[]) for t in (['flat'] if training else ['flat','bumps_5mm']) for name,c in [('stop',[0,0,0]),('forward',[.2,0,0]),('left',[0,.1,0]),('turn_left',[0,0,.4])]]

def measure(policy,case,seed,terrain_seed,randomized=False,render=None,seconds=12,keep_trace=False):
    command=case['command'];schedule=[(2,[0,0,0]),(seconds,command)]
    r=trial(SlewedPolicy(policy),command,seed,seconds,noise=randomized,randomization=randomized,schedule=schedule,trace=True,revision='v07',terrain=case['terrain'],terrain_seed=terrain_seed,terrain_spawn=case['spawn'],events=case['events'],render=render)
    trace=r['trace'];a=np.asarray(trace['actual_twist']);tilt=np.asarray(trace['tilt_deg']);target=np.asarray(command)
    r.update(case=case['name'],suite=case['suite'],command_slew=True,safe=bool(r['survived'] and r['nonfoot_ground_fraction']==0 and r['payload_selfcontact_fraction']==0))
    steady=a[150:];mean=steady.mean(0) if len(steady) else np.zeros(3)
    r['steady_twist']=mean.tolist();r['tracking_error']=float(np.linalg.norm((mean-target)/[.08,.05,.2]))
    r['direction_followed']=bool(np.all(np.where(target!=0,mean*target>=.25*target**2,abs(mean)<=[.04,.04,.12])))
    recoveries=[]
    for d in r.get('disturbances',[]):
        e=d['event'];end=None if d['trigger_s'] is None else d['trigger_s']+e['duration']
        applied=d['applied_seconds']>=e['duration']-.021
        experienced=applied and (e['kind']!='slip' or (d['peak_supported_foot_speed_m_s']>=.05 and d['supported_foot_travel_m']>=.004))
        settle=None;error=20.
        if end is not None:
            for i in range(round(end/.02),min(len(a)-24,round((end+3)/.02))):
                if np.all(abs(a[i:i+25].mean(0)-target)<=[.08,.05,.2]) and np.max(tilt[i:i+25])<12:
                    settle=max(0.,i*.02-end);break
            window=a[round((end+1)/.02):round((end+3)/.02)]
            if len(window):error=float(np.linalg.norm((window.mean(0)-target)/[.08,.05,.2]))
        recoveries.append(dict(kind=e['kind'],applied=applied,experienced=bool(experienced),recovered=bool(r['safe'] and experienced and settle is not None),settle_s=settle,post_event_tracking_error=error))
    r['recoveries']=recoveries;r['recovery_complete']=bool(recoveries and all(x['recovered'] for x in recoveries))
    r['passed']=r['safe'] and (r['recovery_complete'] if recoveries else r['direction_followed'])
    if not keep_trace:r.pop('trace')
    return r

_policy=None
def init(path):
    global _policy
    _policy=session(path)
def job(args):return measure(_policy,*args)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--suite',choices=['all','slope','push','trip_slip','regression'],default='all');ap.add_argument('--training-cases',action='store_true');ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--seed-start',type=int,default=100);ap.add_argument('--terrain-seed-start',type=int,default=1100);ap.add_argument('--noise-dr',action='store_true');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    jobs=[(c,a.seed_start+i,a.terrain_seed_start+i,a.noise_dr) for c in cases(a.suite,a.training_cases) for i in range(a.seeds)]
    with mp.get_context('spawn').Pool(a.workers,initializer=init,initargs=(a.policy,)) as pool:rows=pool.map(job,jobs)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(policy=a.policy,simulation_only=True,trials=rows),indent=2)+'\n')
    for s in sorted({r['suite'] for r in rows}):
        rr=[r for r in rows if r['suite']==s];print(s,'passes',sum(r['passed'] for r in rr),'safe',sum(r['safe'] for r in rr),'/',len(rr),'tracking',round(np.mean([r['tracking_error'] for r in rr]),3),flush=True)
