"""Search a real seated pose that clears the existing rigid mount."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from evaluate_posture import trial
from evaluate_motion import session
from posture_policy import STAND
from seated_policy import SeatedPolicy
stand=None
def init():
    global stand
    stand=session(STAND)
def score(p):
    rows=[trial(SeatedPolicy(stand,p),'sitstand',s,randomized=s%2==1) for s in [0,1]]
    fitness=np.mean([5*r['sitstand_passed']+3*r['sit_quality']+r['final_quality']+r['recovered']-.5*r['payload_floor_fraction']-.5*r['payload_internal_fraction'] for r in rows])
    return dict(parameters=list(map(float,p)),fitness=float(fitness),successes=sum(r['sitstand_passed'] for r in rows),trials=rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--generations',type=int,default=10);ap.add_argument('--population',type=int,default=24);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);mean=np.array([-.524,1.047,0,.18,0,0,0]);std=np.array([.22,.22,.3,.12,.15,.35,.15])
    lower=np.array([-1.3,.6,-1.2,0,-.35,-1,-1]);upper=np.array([-.15,1.5,.8,.37,.35,1,1]);rng=np.random.default_rng(1853);best=None;history=[]
    with mp.get_context('spawn').Pool(4,initializer=init) as pool:
        for gen in range(a.generations):
            pop=np.clip(rng.normal(mean,std,(a.population,7)),lower,upper);pop[0]=mean if best is None else best['parameters']
            scores=pool.map(score,pop);scores.sort(key=lambda r:r['fitness'],reverse=True)
            if best is None or scores[0]['fitness']>best['fitness']:best=scores[0]
            elite=np.array([s['parameters'] for s in scores[:6]]);mean=.2*mean+.8*elite.mean(axis=0);std=np.maximum(.025,.3*std+.7*elite.std(axis=0))
            history.append(dict(generation=gen,candidates=scores));record=dict(training_seeds=[0,1],randomized_seeds=[1],best=best,history=history)
            (a.out/'search.json').write_text(json.dumps(record,indent=2)+'\n');(a.out/'parameters.json').write_text(json.dumps(dict(parameters=best['parameters'],method='seated_target'),indent=2)+'\n')
            print(gen,round(best['fitness'],3),best['successes'],np.round(best['parameters'],3).tolist(),flush=True)
if __name__=='__main__':main()
