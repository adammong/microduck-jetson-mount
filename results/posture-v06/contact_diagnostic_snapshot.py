import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training'))
import evaluate_posture as e
from evaluate_motion import session
original=e.environment
def diagnostic(*args,**kwargs):
 env=original(*args,**kwargs)
 for i in range(env.model.ngeom):
  if (env.model.geom(i).name or '').startswith('easy_'):
   env.model.geom_contype[i]=0;env.model.geom_conaffinity[i]=0
 return env
e.environment=diagnostic;rows=[]
for name,poses in [('alpha_sitstand',['sitstand']),('alpha_stand',['front','back','left','right'])]:
 for pose in poses:
  for seed in [0,1]:
   r=e.trial(session(f'tmp/sim-inputs/{name}.onnx'),pose,seed);r['counterfactual']='Payload mass/inertia retained; mount collision disabled. Not a valid deployable result.';rows.append(r)
   print(name,pose,seed,r['recovered'],r['sitstand_passed'],r['sit_mean_height_m'],flush=True)
Path('results/posture-v06/collision-disabled-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
