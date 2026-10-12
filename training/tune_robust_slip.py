"""Progressive slip calibration after the initial joint-recovery search."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from tune_robust import init,score
from robust_policy import LOW,HIGH,export
from evaluate_robust import cases
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--init',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 starting=json.loads(a.init.read_text());mean=np.asarray(starting['parameters']);initial=mean.copy();std=(HIGH-LOW)*.12;active=[0,5,6,7,8,9,10,11,12,13,14,15];rng=np.random.default_rng(4402);history=[];best=None
 slope=[c for c in cases('slope',True) if c['name'] in ['downhill_5deg_forward','downhill_5deg_backward','cross_left_3deg_turn_right']]
 ordinary=cases('regression',True)
 with mp.get_context('spawn').Pool(4,initializer=init) as pool:
  for stage,levels in [('moderate',[.35,.25]),('harder',[.25,.15])]:
   selected=slope+ordinary
   for terrain in ['flat','bumps_5mm']:
    for mu in levels:selected.append(dict(name=f'{terrain}_slip_mu{mu:g}',suite='trip_slip',terrain=terrain,spawn=None,command=[.2,0,0],events=[dict(kind='slip',start=4.,duration=.5,friction=mu),dict(kind='slip',start=8.,duration=.5,friction=mu)]))
   selected += [c for c in cases('push',True) if '_forward_' in c['name']]
   best=None
   for g in range(3):
    pop=np.tile(mean,(16,1));pop[:,active]=np.clip(rng.normal(mean[active],std[active],(16,len(active))),LOW[active],HIGH[active]);pop[0]=mean if best is None else best['parameters'];pop[1]=initial
    scores=pool.map(score,[(p.tolist(),selected) for p in pop]);scores.sort(key=lambda r:r['fitness'],reverse=True)
    if best is None or scores[0]['fitness']>best['fitness']:best=scores[0]
    elite=np.array([r['parameters'] for r in scores[:4]]);mean[active]=.3*mean[active]+.7*elite[:,active].mean(0);std[active]=np.maximum((HIGH-LOW)[active]*.03,.3*std[active]+.7*elite[:,active].std(0))
    history.append(dict(stage=stage,generation=g,cases=selected,candidates=scores));record=dict(method='Progressive slip CPU CEM calibration, frozen terrain actor',random_seed=4402,initial_parameters=initial.tolist(),initial_source=starting,physics_seeds=[0,1],terrain_seeds=[1000,1001],noise_dr_seed=1,best=best,history=history)
    export(best['parameters'],a.out/'candidate');(a.out/'search.json').write_text(json.dumps(record,indent=2)+'\n')
    print(stage,g,'safe',best['safe'],'passed',best['passed'],'/',len(best['trials']),'fitness',round(best['fitness'],4),flush=True)
   mean=np.array(best['parameters']);std=np.maximum(std,(HIGH-LOW)*.08)
