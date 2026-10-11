"""Small-bump then slope/roughness curriculum for a frozen walking actor."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from evaluate_terrain import job
from evaluate_motion import session
from terrain_policy import TerrainPolicy,BASE,BOUNDS,export
import evaluate_terrain

base=None

def init():
 global base
 base=session(BASE)

EARLY=[('flat','forward'),('bumps_2mm','forward'),('bumps_2mm','left'),('bumps_2mm','turn_left')]
MAIN=[('flat','forward'),('flat','backward'),('flat','left'),('flat','turn_right'),('bumps_5mm','forward'),('bumps_5mm','backward'),('bumps_5mm','left'),('bumps_5mm','right'),('uphill_3deg','forward'),('downhill_3deg','forward'),('cross_left_3deg','forward'),('cross_right_3deg','forward'),('mixed_5mm_3deg','curve_left'),('mixed_5mm_3deg','curve_right')]

def score(args):
 parameters,cases=args;evaluate_terrain._actor=TerrainPolicy(base,parameters)
 rows=[job((terrain,task,seed,10+seed,8.,seed==1,1.)) for terrain,task in cases for seed in [0,1]]
 value=np.mean([10*r['passed']+2*r['seconds']/r['requested_seconds']-r['scaled_tracking_error']-.002*r['max_tilt_deg'] for r in rows])-.003*float(np.linalg.norm(np.asarray(parameters)/BOUNDS))
 return dict(parameters=list(map(float,parameters)),fitness=float(value),passed=sum(r['passed'] for r in rows),trials=rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--early-generations',type=int,default=3);ap.add_argument('--generations',type=int,default=8);ap.add_argument('--population',type=int,default=20);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(118);mean=np.zeros(8);std=np.array([.015,.015,.015,.12,.035,.12,.035,.035]);history=[];best=None
 with mp.get_context('spawn').Pool(a.workers,initializer=init) as pool:
  for stage,cases,generations in [('bumps_2mm',EARLY,a.early_generations),('bumps_5mm_slopes_3deg',MAIN,a.generations)]:
   previous=best;best=None
   for g in range(generations):
    pop=np.clip(rng.normal(mean,std,(a.population,8)),-BOUNDS,BOUNDS);pop[0]=mean if best is None else best['parameters'];pop[1]=0.
    if g==0 and previous is not None:pop[0]=previous['parameters']
    candidates=pool.map(score,[(p.tolist(),cases) for p in pop]);candidates.sort(key=lambda r:r['fitness'],reverse=True)
    if best is None or candidates[0]['fitness']>best['fitness']:best=candidates[0]
    elite=np.array([r['parameters'] for r in candidates[:max(3,a.population//4)]]);mean=.3*mean+.7*elite.mean(0);std=np.maximum(BOUNDS*.02,.3*std+.7*elite.std(0))
    history.append(dict(stage=stage,generation=g,cases=cases,candidates=candidates));record=dict(method='Two-stage CPU CEM calibration, frozen neural actor',random_seed=118,physics_seeds=[0,1],terrain_seeds=[10,11],randomized_seeds=[1],parameter_bounds=BOUNDS.tolist(),correction_bound_rad=.12,fitness='10*pass + 2*completion_fraction - scaled_tracking_error - .002*max_tilt - .003*normalized_parameter_norm',best=best,history=history)
    (a.out/'search.json').write_text(json.dumps(record,indent=2)+'\n');export(best['parameters'],a.out/'candidate')
    print(stage,g,'pass',best['passed'],'/',len(best['trials']),'fitness',round(best['fitness'],4),'parameters',np.round(best['parameters'],4).tolist(),flush=True)
if __name__=='__main__':main()
