import sys,json,multiprocessing as mp
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path.cwd()/'training'))
from evaluate_posture import trial
from evaluate_motion import session
actors={}
def job(args):
 group,path,kwargs=args
 if path not in actors:actors[path]='limp' if path=='limp' else session(path)
 return group,trial(actors[path],revision='v07',**kwargs)
if __name__=='__main__':
 root=Path('results/posture-v07');falls=json.loads((root/'physical-falls.json').read_text())['fall_poses'];jobs=[];policies={}
 for label,path in [('recovery','results/posture-v07/policies/recovery-candidate/policy.onnx'),('recovery-baseline','tmp/sim-inputs/alpha_stand.onnx')]:
  for condition in ['nominal','randomized']:
   name=label+'-'+condition;policies[name]=path
   for pose in ['stand','back','front','left','right']:
    for seed in range(200,208) if condition=='nominal' else range(220,224):
     for scale in [1.] if condition=='nominal' else [.9,1.1]:jobs.append((name,path,dict(pose=pose,seed=seed,randomized=condition=='randomized',payload_scale=scale)))
  name=label+'-physical-falls';policies[name]=path
  for i,f in enumerate(falls):jobs.append((name,path,dict(pose=f['requested_tip'],seed=940+i,replay=f['qpos'])))
 policies['limp-physical-falls']='limp'
 for i,f in enumerate(falls):jobs.append(('limp-physical-falls','limp',dict(pose=f['requested_tip'],seed=940+i,replay=f['qpos'])))
 groups=defaultdict(list)
 with mp.get_context('spawn').Pool(4) as pool:
  for name,row in pool.imap_unordered(job,jobs):groups[name].append(row)
 for name,rows in groups.items():
  rows.sort(key=lambda r:(r['pose'],r['seed'],r['payload_scale']))
  (root/(name+'.json')).write_text(json.dumps(dict(policy=policies[name],revision='v07',trials=rows),indent=2)+'\n')
  fallen=[r for r in rows if r['pose']!='stand'];print(name,sum(r['recovered'] for r in fallen),'/',len(fallen),flush=True)
