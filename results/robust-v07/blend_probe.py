from pathlib import Path
import sys,json,multiprocessing as mp,numpy as np
sys.path.insert(0,'training')
from evaluate_motion import session
from robust_policy import BASE
from evaluate_robust import measure,cases
_base=None;_full=None
class Blend:
 def __init__(self,base,full,tilt,gyro):self.base=base;self.full=full;self.tilt=tilt;self.gyro=gyro;self.peak_gate=0.
 def get_inputs(self):return self.base.get_inputs()
 def get_modelmeta(self):return self.base.get_modelmeta()
 def run(self,names,inputs):
  o=next(iter(inputs.values()));g=np.maximum(np.clip((abs(o[:,3:5]).max(1,keepdims=True)-self.tilt)/.12,0,1),np.clip((abs(o[:,:2]).max(1,keepdims=True)-self.gyro)/1.5,0,1));self.peak_gate=max(self.peak_gate,float(g.max()))
  b=self.base.run(names,inputs)[0];f=self.full.run(names,inputs)[0];return [f+g*(b-f)]
def init():
 global _base,_full
 _base=session(BASE);_full=session('results/robust-v07/policies/candidate/policy.onnx')
def job(args):
 label,gate,cc=args;rows=[]
 for c in cc:
  for seed in [2,3]:
   p=_base if label=='baseline' else _full if label=='full' else Blend(_base,_full,*gate)
   r=measure(p,c,seed,1000+seed,randomized=seed==3);rows.append(r)
 return dict(label=label,guard=gate,trials=rows)
if __name__=='__main__':
 cc=cases('push',True)+cases('trip_slip',True)+[c for c in cases('slope',True) if c['name'] in ['downhill_5deg_forward','downhill_5deg_backward','cross_left_3deg_turn_right']]+cases('regression',True)
 models=[('baseline',None),('full',None)]+[(f'guard_{t:g}_{g:g}',[t,g]) for t in [.10,.16] for g in [.5,1.,1.5]]
 with mp.get_context('spawn').Pool(4,initializer=init) as pool:rows=pool.map(job,[(label,gate,cc) for label,gate in models])
 for d in rows:print(d['label'],[(s,sum(r['passed'] for r in d['trials'] if r['suite']==s),sum(r['safe'] for r in d['trials'] if r['suite']==s),sum(r['suite']==s for r in d['trials'])) for s in ['slope','push','trip_slip','regression']],flush=True)
 Path('results/robust-v07/blend-probe.json').write_text(json.dumps(dict(description='Observable IMU envelope blending toward the existing actor; exploratory calibration seeds 2/3, no final data.',candidates=rows),indent=2)+'\n')
