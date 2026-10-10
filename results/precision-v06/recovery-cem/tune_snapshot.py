import sys,json,multiprocessing as mp
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'training'))
from precision_policy import PrecisionPolicy,BASE,save
from evaluate_precision import measure,cases
from evaluate_motion import session
base=None
def init():
 global base
 base=session(BASE)
def score(job):
 p,params,seeds=job
 actor=PrecisionPolicy(base,p['exponents'],p['balance'],p['gains'],p['turn_feedback'],params,True)
 selected=[c for c in cases('push') if c['name'].startswith('stand_')]
 rows=[measure(actor,c,s,dr=s%2==1,payload_scale=[.9,1.,1.1][s%3]) for c in selected for s in seeds]
 bad=sum(not r['safe'] for r in rows)
 err=np.mean([r['tracking_error'] for r in rows]);rec=np.mean([e for r in rows for e in r['post_push_errors']])
 return dict(recovery=list(map(float,params)),fitness=float(-20*bad/len(rows)-err-.5*rec),failures=bad,trials=rows)
if __name__=='__main__':
 p=json.load(open('results/precision-v06/policies/combined-v1/parameters.json'))
 out=Path('results/precision-v06/recovery-cem');out.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(1718);mean=np.array([2.,.2,2.,.2]);std=np.array([1.2,.4,1.2,.4]);best=None;history=[]
 with mp.get_context('spawn').Pool(4,initializer=init) as pool:
  for gen in range(6):
   pop=np.clip(rng.normal(mean,std,(12,4)),0,8);pop[0]=mean if best is None else best['recovery'];pop[1]=0
   scores=pool.map(score,[(p,v,[0,1,2,3]) for v in pop]);scores.sort(key=lambda x:x['fitness'],reverse=True)
   if best is None or scores[0]['fitness']>best['fitness']:best=scores[0]
   elite=np.array([x['recovery'] for x in scores[:3]]);mean=.3*mean+.7*elite.mean(axis=0);std=np.maximum([.08,.03,.08,.03],.3*std+.7*elite.std(axis=0))
   history.append(dict(generation=gen,candidates=scores));save(p['exponents'],p['balance'],out/'candidate',p['gains'],p['turn_feedback'],best['recovery'],True)
   (out/'search.json').write_text(json.dumps(dict(training_seeds=[0,1,2,3],noise_dr='odd seeds',payload_scales=[.9,1.,1.1],best=best,history=history),indent=2)+'\n')
   print(gen,best['fitness'],best['failures'],best['recovery'],flush=True)
