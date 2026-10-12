"""Actual mid-posture stop/command overrides, without episode resets."""
from pathlib import Path
import sys,json,multiprocessing as mp
sys.path.insert(0,'training')
from evaluate_motion import session
from evaluate_posture import environment,state
from evaluate_skills import WALKING,POSTURE
from skill_controller import SkillController,parse_command
EVENTS=[(2,'sit'),(3,'stop'),(6,'walk forward'),(6.5,'stop'),(12,'walk forward for ten seconds'),(13,'sit'),(14,'stop'),(17,'sit'),(19,'walk forward'),(20,'stop'),(27,'walk forward'),(31,'sit'),(31.8,'stop'),(38,'stand')]
def trial(args):
 seed,randomized,scale=args;env=environment(seed,randomized,scale,revision='v07');ctl=SkillController(session(WALKING),session(POSTURE));index=0;records=[];rows=[]
 for k in range(2200):
  now=k*.02;ctl.heartbeat(now)
  if index<len(EVENTS) and now>=EVENTS[index][0]:
   text=EVENTS[index][1];records.append(dict(time_s=now,text=text,phase_before=ctl.phase));ctl.accept(parse_command(text),now);index+=1
  try:action=ctl.action(env,state(env),now)
  except ValueError:break
  env.step(action);m=state(env);ctl.check(m,(k+1)*.02);rows.append(m)
  if ctl.fault:break
 safe=not ctl.fault and not any(r['payload_floor'] or r['payload_internal'] for r in rows)
 passed=safe and index==len(EVENTS) and all(r['standing'] for r in rows[-100:]) and ctl.pending is None and any(r['sitting'] for r in rows)
 r=dict(seed=seed,randomized=randomized,payload_scale=scale,passed=bool(passed),safe=bool(safe),fault=ctl.fault,seconds=len(rows)*.02,events=records,transitions=ctl.transitions,final=rows[-1])
 env.close();return r
if __name__=='__main__':
 jobs=[(s,False,1.) for s in range(880,884)]+[(s,True,scale) for s in [890,891] for scale in [.9,1.1]]
 with mp.get_context('spawn').Pool(4) as pool:rows=pool.map(trial,jobs)
 Path('results/voice-v07/override-verification.json').write_text(json.dumps(dict(simulation_only=True,trials=rows),indent=2)+'\n')
 print('Actual mid-posture overrides',sum(r['passed'] for r in rows),'/',len(rows),flush=True)
 for r in rows:
  if not r['passed']:print(r['seed'],r['payload_scale'],r['fault'],flush=True)
