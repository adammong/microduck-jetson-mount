"""Jointly tune seated and standing feedback on the collision-enabled v07 CAD."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from clearance_policy import SitStandPolicy
from evaluate_posture import trial

def unpack(p):
    stand=np.zeros(8,np.float32);sit=np.r_[p[0],np.zeros(8)].astype(np.float32)
    stand[[0,1,2,3,6,7]]=p[1:7];sit[[1,2,3,4,7,8]]=p[7:13]
    return stand,sit

def score(p):
    stand,sit=unpack(p);actor=SitStandPolicy(stand,sit)
    rows=[trial(actor,'sitstand',s,randomized=s%2==1,revision='v07') for s in [0,1,2,3]]
    fitness=np.mean([8*r['sitstand_passed']+2*r['sit_quality']+r['final_quality']+r['recovered']-2*r['payload_internal_fraction']-2*r['payload_floor_fraction'] for r in rows])
    return dict(parameters=list(map(float,p)),fitness=float(fitness),successes=sum(r['sitstand_passed'] for r in rows),trials=rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--generations',type=int,default=10);ap.add_argument('--population',type=int,default=24);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(1911);mean=np.r_[1.,np.zeros(12)];std=np.r_[.03,[.07,.07,.07,.04,.2,.12]*2]
    lower=np.r_[.9,[-.3,-.3,-.3,-.2,-1.,-1.]*2];upper=np.r_[1.,[.3,.3,.3,.2,1.,1.]*2];best=None;history=[]
    assert unpack(mean)[0].shape==(8,) and unpack(mean)[1].shape==(9,)
    with mp.get_context('spawn').Pool(4) as pool:
        for gen in range(a.generations):
            pop=np.clip(rng.normal(mean,std,(a.population,13)),lower,upper);pop[0]=mean if best is None else best['parameters'];pop[1]=np.r_[1.,np.zeros(12)]
            candidates=pool.map(score,pop);candidates.sort(key=lambda x:x['fitness'],reverse=True)
            if best is None or candidates[0]['fitness']>best['fitness']:best=candidates[0]
            elite=np.array([c['parameters'] for c in candidates[:max(3,a.population//4)]])
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.015,.3*std+.7*elite.std(0));history.append(dict(generation=gen,candidates=candidates))
            (a.out/'search.json').write_text(json.dumps(dict(revision='v07',training_seeds=[0,1,2,3],randomized_seeds=[1,3],best=best,history=history),indent=2)+'\n')
            stand,sit=unpack(best['parameters']);(a.out/'parameters.json').write_text(json.dumps(dict(method='neutral_head_sitstand',parameters=best['parameters'],stand_parameters=stand.tolist(),sit_parameters=sit.tolist()),indent=2)+'\n')
            print(gen,round(best['fitness'],3),best['successes'],np.round(best['parameters'],3).tolist(),flush=True)
if __name__=='__main__':main()
