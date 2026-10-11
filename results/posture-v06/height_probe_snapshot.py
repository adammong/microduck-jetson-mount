import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'training'))
from evaluate_posture import trial
from evaluate_motion import session
from posture_policy import STAND
class Height:
 def __init__(self,base,drop):self.base=base;self.drop=drop
 def get_inputs(self):return self.base.get_inputs()
 def run(self,names,inputs):
  obs=next(iter(inputs.values())).copy();flag=obs[:,48]>=.5;obs[:,48:61]=0;obs[:,57]=-self.drop*flag
  return self.base.run(None,{self.get_inputs()[0].name:obs})
base=session(STAND);rows=[]
for drop in [.035,.05,.065,.08,.1]:
 for seed in [0,1]:
  r=trial(Height(base,drop),'sitstand',seed);r['drop']=drop;rows.append(r)
  print(drop,seed,r['recovered'],r['sitstand_passed'],r['sit_mean_height_m'],r['sit_mean_tilt_deg'],r['payload_floor_fraction'],r['payload_internal_fraction'],flush=True)
Path('results/posture-v06/height-grid.json').write_text(json.dumps(rows,indent=2)+'\n')
