from pathlib import Path
import sys,json,multiprocessing as mp,numpy as np
sys.path.insert(0,'training')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_skills as ev
import skill_controller as sc
class Tuned(sc.SkillController):
 def __init__(self,w,p,blend_s):
  super().__init__(w,p,blend_s);self.settle_s=0.
 def action(self,env,measurement,now,dt=.02):
  action=super().action(env,measurement,now,dt)
  if self.active_actor=='walking' and self.command[0]<0:
   hip,knee=BIAS;gain=min(-float(self.command[0])/.05,1.)
   delta=np.zeros(14,np.float32);delta[[2,11]]=np.array([hip,-hip])*gain;delta[[3,12]]=np.array([knee,-knee])*gain;delta[[4,13]]=np.array([-hip-knee,hip+knee])*gain
   action=action+np.clip(delta,-.12,.12);self.last_action=action.copy()
  return action
BIAS=(0,0)
def job(args):
 global BIAS
 BIAS=args;ev.SkillController=Tuned
 rows=[ev.trial(name,seed,randomized=seed in [11,13],blend_s=.3) for name in ['cycles','interrupts'] for seed in [10,11,12,13]]
 return dict(bias=list(args),passed=sum(r['passed'] for r in rows),safe=sum(r['safe'] for r in rows),trials=rows)
if __name__=='__main__':
 jobs=[(hip,knee) for hip in [-.04,-.06,-.08] for knee in [-.04,-.06,-.08]]
 with mp.get_context('spawn').Pool(4) as pool:
  rows=[]
  for r in pool.imap(job,jobs):
   rows.append(r);print(r['bias'],r['passed'],r['safe'],'/8',flush=True)
   Path('tmp/voice-study/reverse-refine.json').write_text(json.dumps(dict(training_seeds=[10,11,12,13],candidates=rows),indent=2)+'\n')
