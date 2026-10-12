from pathlib import Path
import sys,json,multiprocessing as mp,numpy as np
sys.path.insert(0,'training')
from evaluate_motion import session
from robust_policy import RobustPolicy,BASE,INITIAL,export
from evaluate_robust import measure,cases
_actor=None

def init():
 global _actor
 _actor=session(BASE)
def job(args):
 label,p,cc=args
 actor=RobustPolicy(_actor,p)
 rows=[measure(actor,c,seed,1000+seed,randomized=seed==3) for c in cc for seed in [2,3]]
 return dict(label=label,parameters=p,trials=rows)
if __name__=='__main__':
 root=Path('results/robust-v07');p=np.array(json.loads((root/'policies/candidate/parameters.json').read_text())['parameters']);models=[('baseline',INITIAL.tolist()),('full',p.tolist())]
 for label,indices in [('without_pitch_roll_gyro',[9,11,12,13,15]),('pose_and_command_only',list(range(8,16))),('without_all_new_gyro',[4,9,11,12,13,15])]:
  q=p.copy();q[indices]=0;models.append((label,q.tolist()))
 cc=cases('push',True)+cases('trip_slip',True)+[c for c in cases('slope',True) if c['name'] in ['downhill_5deg_forward','downhill_5deg_backward','cross_left_3deg_turn_right']]+cases('regression',True)
 with mp.get_context('spawn').Pool(4,initializer=init) as pool:rows=pool.map(job,[(label,p,cc) for label,p in models])
 for d in rows:
  print(d['label'],[(s,sum(r['passed'] for r in d['trials'] if r['suite']==s),sum(r['safe'] for r in d['trials'] if r['suite']==s),sum(r['suite']==s for r in d['trials'])) for s in ['slope','push','trip_slip','regression']],flush=True)
 (root/'ablation-probe.json').write_text(json.dumps(dict(description='Exploratory feature ablations on calibration seeds 2/3; final acceptance requires fresh paired tests.',trials=rows),indent=2)+'\n')
 for d in rows[2:]:export(d['parameters'],root/'policies'/('ablation-'+d['label']))
