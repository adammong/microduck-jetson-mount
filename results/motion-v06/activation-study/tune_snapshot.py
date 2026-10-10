"""Paired finite grid study of one gyro-based turn-activation parameter."""
import argparse,json,multiprocessing as mp,shutil
from pathlib import Path
import numpy as np
from motion_activation import ActivationPolicy,save
from evaluate_motion import session,trial
from evaluate_transitions import SlewedPolicy,SCHEDULE
from paths import ROOT
_teacher=None
def init():
    global _teacher
    _teacher=session(ROOT/'tmp/sim-inputs/alpha_walking.onnx')
def score(job):
    gain,parameters,seeds=job;rows=[];errors=[];missed=0
    for dr in [False,True]:
        for seed in seeds:
            policy=SlewedPolicy(ActivationPolicy(_teacher,parameters,gain))
            r=trial(policy,[0.,0.,0.],seed,SCHEDULE[-1][0],noise=dr,randomization=dr,schedule=SCHEDULE);rows.append(r)
            for s in r['segments']:
                if s['command'] not in [[0,0,.4],[0,0,-.4]]:continue
                c=np.array(s['command']);v=np.array(s['mean_twist']);errors.append(np.sum(((v-c)/[.08,.05,.25])**2))
                missed+=v[2]*np.sign(c[2])<abs(c[2])*.5
    bad=sum(not r['survived'] or r['payload_selfcontact_fraction']>0 or r['nonfoot_ground_fraction']>0 for r in rows)
    return dict(boost=gain,fitness=float(-float(np.mean(errors))-10*(bad+missed)/len(rows)),feasible=bool(bad==0 and missed==0),missed_turns=int(missed),trials=rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--init',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(8)));ap.add_argument('--gains',nargs='+',type=float,default=[0,.15,.3,.45,.6]);a=ap.parse_args()
    parameters=json.loads(a.init.read_text())['parameters'];a.out.mkdir(parents=True,exist_ok=True)
    with mp.get_context('spawn').Pool(4,initializer=init) as pool:results=pool.map(score,[(g,parameters,a.seeds) for g in a.gains])
    results.sort(key=lambda r:r['fitness'],reverse=True);best=results[0];save(parameters,best['boost'],a.out/'candidate')
    (a.out/'search.json').write_text(json.dumps(dict(method='Paired one-parameter turn-start activation grid, full production-slew movement sequences in nominal and noise+DR physics',training_seeds=a.seeds,gains=a.gains,best=best,candidates=results),indent=2)+'\n');shutil.copyfile(__file__,a.out/'tune_snapshot.py')
    print([(r['boost'],r['fitness'],r['feasible'],r['missed_turns']) for r in results],flush=True)
if __name__=='__main__':main()
