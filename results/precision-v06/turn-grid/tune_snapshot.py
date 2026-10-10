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
 p,mode,exp,gain=job
 ex=p['exponents'].copy();ex[mode]=exp;turn=p.get('turn_feedback',[0,0]).copy();turn[mode-4]=gain
 selected=[c for c in cases('slow') if c['name'].rsplit('_',1)[0]==['turn_left','turn_right'][mode-4]]
 rows=[measure(PrecisionPolicy(base,ex,p['balance'],p['gains'],turn),c,s) for c in selected for s in [0,1]]
 bad=sum(not r['safe'] for r in rows)
 fit=-np.mean([r['tracking_error'] for r in rows])-20*bad/len(rows)
 return dict(exponents=ex,turn_feedback=turn,gain=gain,fitness=float(fit),failures=bad,trials=rows)
if __name__=='__main__':
 p=json.load(open('results/precision-v06/amplitude-grid/candidate/parameters.json'));history=[]
 out=Path('results/precision-v06/turn-grid');out.mkdir(parents=True,exist_ok=True)
 with mp.get_context('spawn').Pool(4,initializer=init) as pool:
  for mode in [4,5]:
   scores=pool.map(score,[(p,mode,e,g) for e in [.3,.5,.65,1.] for g in [0.,.5,1.,2.,3.,4.,6.]])
   scores.sort(key=lambda x:x['fitness'],reverse=True);b=scores[0];p['exponents']=b['exponents'];p['turn_feedback']=b['turn_feedback']
   history.append(dict(mode=mode,candidates=scores));save(p['exponents'],p['balance'],out/'candidate',p['gains'],p['turn_feedback'])
   (out/'search.json').write_text(json.dumps(dict(training_seeds=[0,1],history=history),indent=2)+'\n')
   print(mode,b['fitness'],b['failures'],p['exponents'],p['turn_feedback'],flush=True)
