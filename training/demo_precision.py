"""Continuous slow-control and standing-shove demonstration; actual physics."""
import argparse,json
from pathlib import Path
from evaluate_motion import session,trial
from evaluate_transitions import SlewedPolicy

SCHEDULE=[(2.,[0.,0.,0.]),(8.,[.1,0.,0.]),(10.,[0.,0.,0.]),
          (16.,[0.,0.,.1]),(18.,[0.,0.,0.]),(24.,[0.,0.,-.1]),(35.,[0.,0.,0.])]
PUSHES=[(27.,.12,[0.,4.,0.]),(30.,.12,[0.,-4.,0.])]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed-start',type=int,default=640);ap.add_argument('--seeds',type=int,default=2)
    ap.add_argument('--noise-dr',action='store_true');ap.add_argument('--render',action='store_true');a=ap.parse_args()
    actor=session(a.policy);rows=[];a.out.parent.mkdir(parents=True,exist_ok=True)
    for seed in range(a.seed_start,a.seed_start+a.seeds):
        dest=a.out.parent/(a.out.stem+f'-seed{seed}') if a.render else None
        r=trial(SlewedPolicy(actor),[0.,0.,0.],seed,35.,render=dest,noise=a.noise_dr,randomization=a.noise_dr,schedule=SCHEDULE,pushes=PUSHES)
        rows.append(r);a.out.write_text(json.dumps(dict(policy=str(a.policy),schedule=SCHEDULE,pushes=PUSHES,trials=rows),indent=2)+'\n')
        assert r['survived'] and r['payload_selfcontact_fraction']==0 and r['nonfoot_ground_fraction']==0,r
        print(seed,r['survived'],[s['mean_twist'] for s in r['segments']],flush=True)
if __name__=='__main__':main()
