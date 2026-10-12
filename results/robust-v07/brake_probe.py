from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,'training')
from evaluate_motion import session
from evaluate_robust import cases,measure
path=Path('results/robust-v07/policies/brake-control-snapshot/policy.onnx');actor=session(path)
class Brake:
 def __init__(self,base,strength):self.base=base;self.strength=strength;self.maximum=0.
 def get_inputs(self):return self.base.get_inputs()
 def get_modelmeta(self):return self.base.get_modelmeta()
 def run(self,names,inputs):
  name=self.get_inputs()[0].name;o=inputs[name].copy()
  gate=np.maximum(np.clip((abs(o[:,3:5]).max(1,keepdims=True)-.12)/.12,0,1),np.clip((abs(o[:,:2]).max(1,keepdims=True)-1.5)/1.5,0,1))
  self.maximum=max(self.maximum,float(gate.max()));o[:,48:51]*=1-self.strength*gate
  return self.base.run(names,{name:o})
rows=[]
for strength in [0.,.5,1.]:
 for mu in [.1,.15,.2,.35]:
  case=next(c for c in cases('trip_slip') if c['name']==f'flat_slip_mu{.2:g}');case=dict(case);case['events']=[dict(e,friction=mu) for e in case['events']];case['name']=f'flat_slip_mu{mu:g}'
  for seed in [2,3]:
   p=Brake(actor,strength);r=measure(p,case,seed,1000+seed,randomized=seed==3);r['brake_strength']=strength;r['peak_brake_gate']=p.maximum;rows.append(r)
 print('brake',strength,'safe',sum(r['safe'] for r in rows if r['brake_strength']==strength),'passed',sum(r['passed'] for r in rows if r['brake_strength']==strength),flush=True)
Path('results/robust-v07/brake-probe.json').write_text(json.dumps(dict(description='Exploratory observable emergency command-brake calibration; distinct seeds 2/3, not final tests.',base_parameters=json.loads(path.with_name('parameters.json').read_text()),trials=rows),indent=2)+'\n')
