"""Bounded, paired CEM calibration of six command-conditioned motion modes."""
import argparse,json,multiprocessing as mp,shutil
from pathlib import Path
import numpy as np
from motion_policy import MotionPolicy,INITIAL,REFERENCES,NAMES,save
from evaluate_motion import trial,session
from paths import ROOT

_teacher=None
def init():
    global _teacher
    _teacher=session(ROOT/'tmp/sim-inputs/alpha_walking.onnx')
def score(job):
    index,mode,params,allparams,seeds,seconds=job
    p=np.array(allparams,np.float32);p[mode]=params;policy=MotionPolicy(_teacher,p)
    rows=[trial(policy,REFERENCES[mode].tolist(),s,seconds) for s in seeds]
    errors=np.array([np.array(r['sustained_twist'])-REFERENCES[mode] for r in rows])
    error=float(np.mean(np.sum((errors/[.08,.05,.25])**2,axis=1)))
    bad=sum(not r['survived'] or r['payload_selfcontact_fraction']>0 or r['nonfoot_ground_fraction']>0 for r in rows)
    regularization=.025*float(np.mean((np.array(params[4:])/.08)**2))
    fitness=-error-10.*bad/len(rows)-regularization
    return dict(index=index,parameters=list(params),fitness=fitness,feasible=bad==0,mean_twist=np.mean([r['sustained_twist'] for r in rows],axis=0).tolist(),trials=rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--generations',type=int,default=8)
    ap.add_argument('--population',type=int,default=20);ap.add_argument('--seconds',type=float,default=12);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3])
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--seed',type=int,default=122);ap.add_argument('--modes',nargs='+',choices=NAMES,default=NAMES)
    ap.add_argument('--init',type=Path);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    parameters=INITIAL.copy() if args.init is None else np.array(json.loads(args.init.read_text())['parameters'],np.float32)
    rng=np.random.default_rng(args.seed);history=[]
    with mp.get_context('spawn').Pool(args.workers,initializer=init) as pool:
        for name in args.modes:
            mode=NAMES.index(name);mean=parameters[mode].astype(float);std=np.array([.025,.03,.15,.3]+[.018]*6+[.02]*5)
            low=np.array([-.6,-.6,-1.8,0]+[-.08]*11);high=np.array([.55,.6,1.8,2]+[.08]*11)
            best=None;generations=[]
            for generation in range(args.generations):
                pop=np.clip(rng.normal(mean,std,(args.population,15)),low,high);pop[0]=mean if best is None else best['parameters'];pop[1]=parameters[mode]
                results=pool.map(score,[(i,mode,p.tolist(),parameters.tolist(),args.seeds,args.seconds) for i,p in enumerate(pop)])
                results.sort(key=lambda r:r['fitness'],reverse=True)
                elite=np.array([r['parameters'] for r in results[:max(3,args.population//4)]])
                mean=.3*mean+.7*elite.mean(axis=0);std=np.maximum(np.array([.006,.006,.03,.04]+[.004]*11),.3*std+.7*elite.std(axis=0))
                if best is None or results[0]['fitness']>best['fitness']:best=results[0]
                generations.append(dict(generation=generation,candidates=results));parameters[mode]=best['parameters'];save(parameters,args.out/'candidate')
                (args.out/'search.json').write_text(json.dumps(dict(method='CEM command-conditioned teacher calibration',training_seeds=args.seeds,seconds=args.seconds,seed=args.seed,fitness='Negative squared sustained body-twist error scaled by [0.08 m/s, 0.05 m/s, 0.25 rad/s], minus 10 per failure/contact fraction and parameter regularization. Same objective throughout.',completed_profiles=history,active_profile=name,generations=generations,parameters=parameters.tolist()),indent=2)+'\n')
                print(name,generation,'fit',round(best['fitness'],3),'feasible',best['feasible'],'twist',np.round(best['mean_twist'],3).tolist(),flush=True)
            history.append(dict(profile=name,best=best,generations=generations))
    save(parameters,args.out/'candidate');shutil.copyfile(__file__,args.out/'tune_snapshot.py')
if __name__=='__main__':main()
