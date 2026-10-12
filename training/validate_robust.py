"""Freeze a correction profile on independent validation seeds before final tests."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from robust_policy import BASE,INITIAL,RobustPolicy,export
from evaluate_motion import session
from evaluate_robust import cases,measure

_base=None
def init():
    global _base
    _base=session(BASE)
def job(args):
    label,parameters=args;p=RobustPolicy(_base,parameters)
    rows=[measure(p,c,30+i,1030+i,randomized=bool(i)) for c in cases() for i in range(2)]
    summary={}
    for suite in sorted({r['suite'] for r in rows}):
        rr=[r for r in rows if r['suite']==suite];ee=[e for r in rr for e in r['recoveries']]
        summary[suite]=dict(passed=sum(r['passed'] for r in rr),safe=sum(r['safe'] for r in rr),trials=len(rr),tracking_error=float(np.mean([r['tracking_error'] for r in rr])),post_event_error=float(np.mean([min(10.,e['post_event_tracking_error']) for e in ee])) if ee else 0.)
    fitness=float(np.mean([14*s['safe']/s['trials']+4*s['passed']/s['trials']-s['tracking_error']-.5*s['post_event_error'] for s in summary.values()]))
    return dict(label=label,parameters=list(map(float,parameters)),fitness=fitness,summary=summary,trials=rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--curriculum',type=Path,required=True);ap.add_argument('--slip-curriculum',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--policy-out',type=Path,required=True);a=ap.parse_args();jobs=[('baseline',INITIAL.tolist())]
    for stage in ('slope','recovery'):
        p=np.asarray(json.loads((a.curriculum/(stage+'-candidate')/'parameters.json').read_text())['parameters'])
        jobs.extend((f'{stage}_{scale:g}',(INITIAL+scale*(p-INITIAL)).tolist()) for scale in (.25,.5,1.))
    if a.slip_curriculum:
        p=np.asarray(json.loads((a.slip_curriculum/'candidate/parameters.json').read_text())['parameters'])
        jobs.extend((f'slip_{scale:g}',(INITIAL+scale*(p-INITIAL)).tolist()) for scale in (.25,.5,1.))
    with mp.get_context('spawn').Pool(4,initializer=init) as pool:rows=pool.map(job,jobs)
    baseline=rows[0]
    eligible=[r for r in rows if all(r['summary'][s]['safe']>=baseline['summary'][s]['safe'] for s in baseline['summary']) and r['summary']['regression']['passed']>=baseline['summary']['regression']['passed'] and r['summary']['slope']['tracking_error']<=baseline['summary']['slope']['tracking_error']*1.02]
    best=max(eligible,key=lambda r:r['fitness']);export(best['parameters'],a.policy_out)
    record=dict(physics_seeds=[30,31],terrain_seeds=[1030,1031],noise_dr_seed=31,criterion='Maximize equal-skill fitness among profiles with no per-suite safety or flat/rough regression-pass reduction, and slope error no more than 2% worse.',selected=best['label'],candidates=rows)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(record,indent=2)+'\n');print('Validation selected',best['label'],'fitness',round(best['fitness'],4),'baseline',round(baseline['fitness'],4),flush=True)
