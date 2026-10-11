import subprocess,sys
from pathlib import Path
Path('tmp/terrain-study').mkdir(parents=True,exist_ok=True)
for name,start,terrain_start,dr in [('nominal',200,500,False),('randomized',220,520,True)]:
 args=[sys.executable,'training/evaluate_terrain.py','--policy','results/posture-v07/policies/walking-adjusted/policy.onnx','--out','results/terrain-v07/baseline-'+name+'.json','--seeds','4','--seed-start',str(start),'--terrain-seed-start',str(terrain_start),'--seconds','15']
 if dr:args+=['--noise-dr']
 with open('tmp/terrain-study/baseline-'+name+'.log','w') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'complete',flush=True)
