"""Drive the exported ONNX through the same command limiter as the control page."""
import argparse,json
from pathlib import Path
import numpy as np
from evaluate_motion import session,trial
from motion_controller import CommandLimiter

SCHEDULE=[(2,[0.,0.,0.]),(8,[.2,0.,0.]),(10,[0.,0.,0.]),
    (15,[-.15,0.,0.]),(17,[0.,0.,0.]),(22,[0.,.1,0.]),(24,[0.,0.,0.]),
    (29,[0.,-.1,0.]),(31,[0.,0.,0.]),(37,[0.,0.,.4]),(39,[0.,0.,0.]),
    (44,[.2,0.,0.]),(46,[0.,0.,0.]),
    (52,[0.,0.,-.4]),(54,[0.,0.,0.]),(60,[.15,0.,.3]),(66,[.15,0.,-.3]),(69,[0.,0.,0.])]

class SlewedPolicy:
    def __init__(self,policy):self.policy=policy;self.limiter=CommandLimiter();self.steps=0
    def get_inputs(self):return self.policy.get_inputs()
    def run(self,names,inputs):
        name=self.get_inputs()[0].name;obs=inputs[name].copy();now=self.steps*.02
        self.limiter.set(obs[0,48:51],now);obs[0,48:51]=self.limiter.step(now)
        self.steps+=1;return self.policy.run(names,{name:obs})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed-start',type=int,default=60);ap.add_argument('--seeds',type=int,default=4)
    ap.add_argument('--render',action='store_true');ap.add_argument('--noise-dr',action='store_true');ap.add_argument('--abrupt',action='store_true')
    a=ap.parse_args();policy=session(a.policy);rows=[];a.out.parent.mkdir(parents=True,exist_ok=True)
    for seed in range(a.seed_start,a.seed_start+a.seeds):
        controller=policy if a.abrupt else SlewedPolicy(policy)
        dest=a.out.parent/(a.out.stem+f'-seed{seed}') if a.render and seed<a.seed_start+2 else None
        r=trial(controller,[0.,0.,0.],seed,SCHEDULE[-1][0],render=dest,noise=a.noise_dr,randomization=a.noise_dr,schedule=SCHEDULE)
        r['command_slew']=not a.abrupt;rows.append(r)
        print('Sequence',seed,r['survived'],'seconds',r['seconds'],'mount contacts',r['payload_selfcontact_fraction'],flush=True)
        a.out.write_text(json.dumps(dict(policy=str(a.policy),schedule=SCHEDULE,command_slew=not a.abrupt,trials=rows),indent=2)+'\n')
if __name__=='__main__':main()
