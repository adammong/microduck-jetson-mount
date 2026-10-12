from pathlib import Path
import sys,json,multiprocessing as mp
sys.path.insert(0,'training')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_skills as ev
import skill_controller as sc
class Tuned(sc.SkillController):
 def __init__(self,w,p,blend_s):
  super().__init__(w,p,blend_s);self.settle_s=SETTLE
SETTLE=.8
def job(args):
 global SETTLE
 actor,reverse,settle,blend=args;SETTLE=settle
 ev.WALKING=Path('results/'+actor+'/policies/'+('selected' if actor=='terrain-v07' else 'walking-adjusted')+'/policy.onnx')
 ev.SkillController=Tuned;sc.COMMANDS['backward'][0]=reverse
 rows=[ev.trial(name,seed,randomized=seed in [11,13],blend_s=blend) for name in ['cycles','interrupts'] for seed in [10,11,12,13]]
 return dict(actor=actor,reverse=reverse,settle=settle,blend=blend,passed=sum(r['passed'] for r in rows),safe=sum(r['safe'] for r in rows),trials=rows)
if __name__=='__main__':
 jobs=[(actor,reverse,settle,blend) for actor in ['terrain-v07','posture-v07'] for reverse in [-.15,-.1] for settle in [0.,.8] for blend in [0.,.3]]
 with mp.get_context('spawn').Pool(4) as pool:
  rows=[]
  for r in pool.imap(job,jobs):
   rows.append(r);print(r['actor'],r['reverse'],r['settle'],r['blend'],r['passed'],r['safe'],'/8',flush=True)
   Path('tmp/voice-study/handoff-grid.json').write_text(json.dumps(dict(training_seeds=[10,11,12,13],candidates=rows),indent=2)+'\n')
