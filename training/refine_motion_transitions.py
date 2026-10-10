"""Calibrate turn profiles from settled standing through real command ramps."""
import argparse,json,multiprocessing as mp,shutil
from pathlib import Path
import numpy as np
from motion_policy import MotionPolicy,NAMES,REFERENCES,save
from evaluate_motion import trial,session
from evaluate_transitions import SlewedPolicy
from paths import ROOT
_teacher=None
def init():
    global _teacher
    _teacher=session(ROOT/'tmp/sim-inputs/alpha_walking.onnx')
def score(job):
    index,mode,params,allparams,seeds,seconds=job
    p=np.array(allparams,np.float32);p[mode]=params;rows=[];measured=[]
    schedule=[(4.,[0.,0.,0.]),(seconds,REFERENCES[mode].tolist())]
    for seed in seeds:
        policy=SlewedPolicy(MotionPolicy(_teacher,p));r=trial(policy,REFERENCES[mode].tolist(),seed,seconds,schedule=schedule)
        rows.append(r);measured.append(r['segments'][-1]['mean_twist'] if len(r['segments'])==2 else [0.,0.,0.])
    errors=np.array(measured)-REFERENCES[mode]
    bad=sum(not r['survived'] or r['payload_selfcontact_fraction']>0 or r['nonfoot_ground_fraction']>0 for r in rows)
    fitness=-float(np.mean(np.sum((errors/[.08,.05,.25])**2,axis=1)))-10.*bad/len(rows)-.025*float(np.mean((np.array(params[4:])/.08)**2))
    return dict(index=index,parameters=list(params),fitness=fitness,feasible=bad==0,mean_twist=np.mean(measured,axis=0).tolist(),trials=rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--init',type=Path,required=True)
    ap.add_argument('--generations',type=int,default=8);ap.add_argument('--population',type=int,default=24);ap.add_argument('--seconds',type=float,default=16)
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(8)));ap.add_argument('--seed',type=int,default=811)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);parameters=np.array(json.loads(a.init.read_text())['parameters'],np.float32);rng=np.random.default_rng(a.seed);history=[]
    with mp.get_context('spawn').Pool(4,initializer=init) as pool:
        for name in ['turn_left','turn_right']:
            mode=NAMES.index(name);mean=parameters[mode].astype(float);std=np.array([.025,.025,.2,.25]+[.018]*6+[.02]*5);best=None;generations=[]
            low=np.array([-.6,-.6,-1.8,0]+[-.08]*11);high=np.array([.55,.6,1.8,2]+[.08]*11)
            for g in range(a.generations):
                pop=np.clip(rng.normal(mean,std,(a.population,15)),low,high);pop[0]=mean if best is None else best['parameters'];pop[1]=parameters[mode]
                results=pool.map(score,[(i,mode,p.tolist(),parameters.tolist(),a.seeds,a.seconds) for i,p in enumerate(pop)]);results.sort(key=lambda r:r['fitness'],reverse=True)
                elite=np.array([r['parameters'] for r in results[:max(3,a.population//4)]]);mean=.3*mean+.7*elite.mean(axis=0);std=np.maximum(np.array([.006,.006,.03,.04]+[.004]*11),.3*std+.7*elite.std(axis=0))
                if best is None or results[0]['fitness']>best['fitness']:best=results[0]
                parameters[mode]=best['parameters'];generations.append(dict(generation=g,candidates=results));save(parameters,a.out/'candidate')
                (a.out/'search.json').write_text(json.dumps(dict(method='CEM observable turn calibration after four seconds standing, using the production command slew',training_seeds=a.seeds,seconds=a.seconds,seed=a.seed,fitness='Same scaled body-twist error / failure-contact rejection / parameter regularizer as the original study, evaluated after the command ramp.',completed_profiles=history,active_profile=name,generations=generations,parameters=parameters.tolist()),indent=2)+'\n')
                print(name,g,round(best['fitness'],3),best['feasible'],np.round(best['mean_twist'],3).tolist(),flush=True)
            history.append(dict(profile=name,best=best,generations=generations))
    shutil.copyfile(__file__,a.out/'tune_snapshot.py')
if __name__=='__main__':main()
