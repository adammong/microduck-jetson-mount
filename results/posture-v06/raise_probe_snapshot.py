import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'training'))
import evaluate_posture as e
from posture_policy import PosturePolicy,STAND,SIT
from evaluate_motion import session
original=e.environment;front_shift=rear_shift=0.
def diagnostic(*args,**kw):
 env=original(*args,**kw)
 for i in range(env.model.ngeom):
  n=env.model.geom(i).name or ''
  if n in ['easy_jetson','easy_tray_plate'] or n.startswith(('easy_tray_rail','easy_corner_gate','easy_front_dock_')) and not n.startswith(('easy_front_dock_arm','easy_front_dock_riser')):env.model.geom_pos[i,2]+=front_shift
  if n in ['easy_battery','easy_rear_battery_pocket'] or n.startswith('easy_rear_battery_'):env.model.geom_pos[i,2]+=rear_shift
 # Layout diagnostic only: exact print mass/inertia need regenerated CAD.
 env.model.body_ipos[env.model.body('easy_mount_payload').id,2]+=(.175*front_shift+.073*rear_shift)/.442094081211853
 return env
e.environment=diagnostic;actor=PosturePolicy(session(STAND),session(SIT));rows=[]
for front_shift,rear_shift in [(f,r) for f in [.01,.02,.03] for r in [0.,.015]]:
 for seed in [0,1]:
  row=e.trial(actor,'sitstand',seed);row['front_shift_mm']=front_shift*1000;row['rear_shift_mm']=rear_shift*1000;row['diagnostic_only']=True;rows.append(row)
  print(front_shift,rear_shift,seed,row['sitstand_passed'],row['sit_hold'],row['recovered'],round(row['payload_internal_fraction'],3),round(row['payload_floor_fraction'],3),flush=True)
Path('results/posture-v06/raised-layout-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
