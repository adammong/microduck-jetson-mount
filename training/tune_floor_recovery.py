"""CEM observed roll/pitch feedback on four settled mounted fall orientations."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from evaluate_posture import trial
from evaluate_motion import session
from posture_policy import STAND
from recovery_policy import RecoveryPolicy

base=None
def init():
    global base
    base=session(STAND)
def score(job):
    params,seeds=job;actor=RecoveryPolicy(base,params)
    rows=[trial(actor,p,s,randomized=s%2==1) for p in ['back','front','left','right'] for s in seeds]
    fit=np.mean([4*r['recovered']+2*r['stand_fraction']+r['final_quality']-.05*r['payload_internal_fraction'] for r in rows])
    return dict(parameters=list(map(float,params)),fitness=float(fit),successes=sum(r['recovered'] for r in rows),trials=rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--init',type=Path)
    ap.add_argument('--generations',type=int,default=10);ap.add_argument('--population',type=int,default=24);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);mean=np.zeros(14);std=np.r_[np.ones(6)*.22,np.ones(8)*.4]
    if a.init:mean[:8]=json.loads(a.init.read_text())['parameters']
    rng=np.random.default_rng(1852);history=[];best=None
    with mp.get_context('spawn').Pool(4,initializer=init) as pool:
        for gen in range(a.generations):
            pop=np.clip(rng.normal(mean,std,(a.population,14)),-1.5,1.5);pop[0]=mean if best is None else best['parameters'];pop[1]=0
            scores=pool.map(score,[(p,[0,1]) for p in pop]);scores.sort(key=lambda x:x['fitness'],reverse=True)
            if best is None or scores[0]['fitness']>best['fitness']:best=scores[0]
            elite=np.array([s['parameters'] for s in scores[:max(3,a.population//4)]])
            mean=.2*mean+.8*elite.mean(axis=0);std=np.maximum(.04,.3*std+.7*elite.std(axis=0))
            history.append(dict(generation=gen,candidates=scores))
            (a.out/'search.json').write_text(json.dumps(dict(training_seeds=[0,1],randomized_seeds=[1],best=best,history=history),indent=2)+'\n')
            (a.out/'parameters.json').write_text(json.dumps(dict(parameters=best['parameters'],method='gravity_conditioned_recovery'),indent=2)+'\n')
            print(gen,round(best['fitness'],3),best['successes'],np.round(best['parameters'],3).tolist(),flush=True)
if __name__=='__main__':main()
