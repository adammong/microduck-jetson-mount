"""CPU CEM slope-precision curriculum followed by joint disturbance training."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from evaluate_motion import session
from evaluate_robust import cases,measure
from robust_policy import BASE,INITIAL,LOW,HIGH,RobustPolicy,export

_base=None
def init():
    global _base
    _base=session(BASE)

def score(args):
    parameters,selected=args;policy=RobustPolicy(_base,parameters)
    rows=[measure(policy,c,seed,1000+seed,randomized=seed==1,seconds=12 if c['events'] else 8) for c in selected for seed in (0,1)]
    groups=[]
    for suite in sorted({r['suite'] for r in rows}):
        rr=[r for r in rows if r['suite']==suite]
        tracking=np.mean([min(10.,r['tracking_error']) for r in rr])
        event_error=np.mean([min(10.,e['post_event_tracking_error']) for r in rr for e in r['recoveries']]) if any(r['recoveries'] for r in rr) else 0.
        # Each skill has equal weight, so easy ordinary walking cannot hide falls.
        value=np.mean([14*r['safe']+4*r['passed'] for r in rr])-tracking-.5*event_error
        groups.append(value)
    fitness=float(np.mean(groups)-.005*np.linalg.norm((np.asarray(parameters)-INITIAL)/(HIGH-LOW)))
    return dict(parameters=list(map(float,parameters)),fitness=fitness,passed=sum(r['passed'] for r in rows),safe=sum(r['safe'] for r in rows),trials=rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--slope-generations',type=int,default=6);ap.add_argument('--recovery-generations',type=int,default=8);ap.add_argument('--population',type=int,default=16);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();assert a.population>=4 and a.slope_generations>0 and a.recovery_generations>0;a.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(4401);mean=INITIAL.astype(float).copy();std=(HIGH-LOW)*.10;history=[];best=None
    slope=cases('slope',True)+cases('regression',True)
    recovery=[c for c in cases('slope',True) if c['command'][2]<=0]+cases('push',True)+cases('trip_slip',True)+cases('regression',True)
    with mp.get_context('spawn').Pool(a.workers,initializer=init) as pool:
        for stage,selected,generations,active in [('slope',slope,a.slope_generations,[0,1,2,3,4,5,6,7,8,10]),('recovery',recovery,a.recovery_generations,list(range(16)))]:
            prior=best;best=None
            if prior:mean=np.asarray(prior['parameters']);std=np.maximum(std,(HIGH-LOW)*.07)
            for generation in range(generations):
                pop=np.tile(mean,(a.population,1));pop[:,active]=np.clip(rng.normal(mean[active],std[active],(a.population,len(active))),LOW[active],HIGH[active]);pop[0]=mean if best is None else best['parameters'];pop[1]=INITIAL
                scores=pool.map(score,[(p.tolist(),selected) for p in pop]);scores.sort(key=lambda r:r['fitness'],reverse=True)
                if best is None or scores[0]['fitness']>best['fitness']:best=scores[0]
                elite=np.array([r['parameters'] for r in scores[:max(3,a.population//4)]])
                mean[active]=.3*mean[active]+.7*elite[:,active].mean(0);std[active]=np.maximum((HIGH-LOW)[active]*.015,.3*std[active]+.7*elite[:,active].std(0))
                history.append(dict(stage=stage,generation=generation,cases=selected,candidates=scores))
                record=dict(method='Two-stage CPU CEM, frozen terrain actor',random_seed=4401,physics_seeds=[0,1],terrain_seeds=[1000,1001],noise_dr_seed=1,best=best,history=history)
                (a.out/'search.json').write_text(json.dumps(record,indent=2)+'\n');export(best['parameters'],a.out/'candidate')
                export(best['parameters'],a.out/(stage+'-candidate'))
                print(stage,generation,'safe',best['safe'],'passed',best['passed'],'/',len(best['trials']),'fitness',round(best['fitness'],4),flush=True)
