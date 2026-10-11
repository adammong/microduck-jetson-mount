"""Run once after the curriculum has finished; freezes export before testing."""
from pathlib import Path
Path('tmp/terrain-study').mkdir(parents=True,exist_ok=True)
import sys,json,subprocess,shutil
sys.path.insert(0,'training')
from terrain_policy import export
root=Path('results/terrain-v07');record=json.loads((root/'curriculum/search.json').read_text());subprocess.run([sys.executable,'training/validate_terrain.py','--search',str(root/'curriculum/search.json'),'--out',str(root/'validation.json'),'--policy-out',str(root/'policies/candidate')],check=True)
for folder in [root/'policies/candidate',root/'curriculum/candidate']:
 shutil.copyfile('results/posture-v07/policies/walking-adjusted/LICENSE',folder/'LICENSE')
 (folder/'NOTICE').write_text('Derived from the Pollen Robotics MicroDuck walking actor at policy revision 088524a64e2557dc453256b6071dbb9d23888802 (Apache 2.0), via the v07 mounted precision walking adaptation. Original neural tensors and normalizer are retained. Adds bounded IMU and commanded-motion feedback, gated off at zero command. See parameters and REPORT.md for selection and simulation limits. Hardware untested.\n')
py=sys.executable
for name,start,terrain_start,dr in [('nominal',200,500,False),('randomized',220,520,True)]:
 args=[py,'training/evaluate_terrain.py','--policy',str(root/'policies/candidate/policy.onnx'),'--out',str(root/('candidate-'+name+'.json')),'--seeds','4','--seed-start',str(start),'--terrain-seed-start',str(terrain_start),'--seconds','15']
 if dr:args+=['--noise-dr']
 with open('tmp/terrain-study/candidate-'+name+'.log','w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Candidate',name,'complete',flush=True)
subprocess.run([py,'training/verify_terrain.py','--policy',str(root/'policies/candidate/policy.onnx'),'--out',str(root/'verification.json')],check=True)
