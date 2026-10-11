"""Paired CEM mounted sit/stand or floor-recovery feedback calibration."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from evaluate_posture import trial
from evaluate_motion import session
from posture_policy import PosturePolicy,STAND,SIT

stand=sit=None
def init():
    global stand,sit
    stand=session(STAND);sit=session(SIT)

def score(job):
    stage,params,fixed,seeds=job
    actor=PosturePolicy(stand,sit,params if stage=='recovery' else fixed,params if stage=='sitstand' else None)
    poses=['back','front','left','right'] if stage=='recovery' else ['sitstand']
    rows=[trial(actor,p,s,randomized=s%2==1) for p in poses for s in seeds]
    if stage=='recovery':
        fit=np.mean([4*r['recovered']+2*r['stand_fraction']+r['final_quality']-.05*r['payload_internal_fraction'] for r in rows])
    else:
        fit=np.mean([4*r['sitstand_passed']+2*r['sit_quality']+r['final_quality']+r['recovered']-.5*r['payload_internal_fraction']-.5*r['payload_floor_fraction'] for r in rows])
    return dict(parameters=list(map(float,params)),fitness=float(fit),successes=sum(r['recovered'] if stage=='recovery' else r['sitstand_passed'] for r in rows),trials=rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['recovery','sitstand'],required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--init',type=Path);ap.add_argument('--generations',type=int,default=8);ap.add_argument('--population',type=int,default=20);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);fixed=json.loads(a.init.read_text())['parameters'] if a.init else np.zeros(8)
    dim=8 if a.stage=='recovery' else 9;mean=np.zeros(dim);std=np.r_[np.ones(6)*.18,[.35,.18]]
    lower=np.r_[np.ones(6)*-.5,[-1.,-1.]];upper=-lower
    if a.stage=='sitstand':mean[0]=1.;std=np.r_[.1,std];lower=np.r_[.85,lower];upper=np.r_[1.,upper]
    rng=np.random.default_rng(1850 if a.stage=='recovery' else 1851);best=None;history=[]
    with mp.get_context('spawn').Pool(4,initializer=init) as pool:
        for generation in range(a.generations):
            population=np.clip(rng.normal(mean,std,(a.population,dim)),lower,upper)
            population[0]=mean if best is None else best['parameters'];population[1]=0
            if a.stage=='sitstand':population[1,0]=1.
            scores=pool.map(score,[(a.stage,p,fixed,[0,1]) for p in population]);scores.sort(key=lambda s:s['fitness'],reverse=True)
            if best is None or scores[0]['fitness']>best['fitness']:best=scores[0]
            elite=np.array([s['parameters'] for s in scores[:max(3,a.population//4)]])
            mean=.2*mean+.8*elite.mean(axis=0);std=np.maximum(.025,.25*std+.75*elite.std(axis=0))
            history.append(dict(generation=generation,candidates=scores))
            (a.out/'search.json').write_text(json.dumps(dict(stage=a.stage,training_seeds=[0,1],randomized_seeds=[1],best=best,history=history),indent=2)+'\n')
            (a.out/'parameters.json').write_text(json.dumps(dict(parameters=best['parameters'],stage=a.stage,stand_parameters=list(map(float,best['parameters'] if a.stage=='recovery' else fixed))),indent=2)+'\n')
            print(a.stage,generation,round(best['fitness'],3),best['successes'],np.round(best['parameters'],3).tolist(),flush=True)
if __name__=='__main__':main()
