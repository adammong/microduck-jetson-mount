"""Small paired command-shaping grid, then CEM IMU balance calibration."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from precision_policy import PrecisionPolicy,BASE,save
from evaluate_precision import cases,measure
from evaluate_motion import session
_base=None
def init():
    global _base
    _base=session(BASE)
def score(job):
    exponents,balance,gains,selected,seeds=job;rows=[]
    for case in selected:
        for seed in seeds:
            rows.append(measure(PrecisionPolicy(_base,exponents,balance,gains),case,seed))
    failures=sum(not r['safe'] for r in rows)
    error=np.mean([r['tracking_error'] for r in rows])
    stop=np.mean([np.linalg.norm(np.array(r.get('stop_residual_twist',[0,0,0]))/[.02,.02,.12]) for r in rows])
    recovery=np.mean([e for r in rows for e in r['post_push_errors']]) if any(r['post_push_errors'] for r in rows) else 0.
    fit=-float(error+.5*stop+.5*recovery+20*failures/len(rows))
    return dict(exponents=list(map(float,exponents)),gains=list(map(float,gains)),balance=list(map(float,balance)),fitness=fit,failures=failures,trials=rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['slow','balance'],default='slow')
    ap.add_argument('--init',type=Path);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1]);ap.add_argument('--generations',type=int,default=8);ap.add_argument('--population',type=int,default=20)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);exponents=np.ones(6);gains=np.ones(6);balance=np.zeros(5);history=[]
    if a.init:
        p=json.loads(a.init.read_text());exponents=np.array(p['exponents']);balance=np.array(p['balance'])
        gains=np.array(p.get('gains',gains))
    with mp.get_context('spawn').Pool(4,initializer=init) as pool:
        if a.stage=='slow':
            for i,name in enumerate(['forward','backward','left','right','turn_left','turn_right']):
                selected=[c for c in cases('slow') if c['name'].rsplit('_',1)[0]==name]
                jobs=[]
                for value in [1.,.8,.65,.5,.4,.3,.2,.1]:
                    for gain in [1.,.85,.7,.55]:
                        p=exponents.copy();p[i]=value;amp=gains.copy();amp[i]=gain;jobs.append((p,balance,amp,selected,a.seeds))
                scores=pool.map(score,jobs);scores.sort(key=lambda r:r['fitness'],reverse=True);best=scores[0]
                exponents=np.array(best['exponents']);gains=np.array(best['gains']);history.append(dict(profile=name,candidates=scores));save(exponents,balance,a.out/'candidate',gains)
                (a.out/'search.json').write_text(json.dumps(dict(method='Paired slow-command exponent grid',training_seeds=a.seeds,history=history),indent=2)+'\n')
                print(name,best['exponents'][i],best['gains'][i],best['fitness'],best['failures'],flush=True)
        else:
            rng=np.random.default_rng(1701);mean=balance.copy();std=np.ones(5)*.025;best=None
            selected=[c for c in cases('push') if '4N' in c['name'] or '6N' in c['name']]
            selected += [c for c in cases('slow') if c['name'].endswith('_0.5')]
            for gen in range(a.generations):
                pop=np.clip(rng.normal(mean,std,(a.population,5)),-.15,.15);pop[0]=balance if best is None else best['balance'];pop[1]=0
                scores=pool.map(score,[(exponents,p,gains,selected,a.seeds) for p in pop]);scores.sort(key=lambda r:r['fitness'],reverse=True)
                if best is None or scores[0]['fitness']>best['fitness']:best=scores[0]
                elite=np.array([r['balance'] for r in scores[:max(3,a.population//4)]])
                mean=.3*mean+.7*elite.mean(axis=0);std=np.maximum(.004,.3*std+.7*elite.std(axis=0));balance=np.array(best['balance'])
                history.append(dict(generation=gen,candidates=scores));save(exponents,balance,a.out/'candidate',gains)
                (a.out/'search.json').write_text(json.dumps(dict(method='Paired CEM bounded IMU balance feedback',training_seeds=a.seeds,history=history),indent=2)+'\n')
                print(gen,best['fitness'],best['failures'],balance.tolist(),flush=True)
if __name__=='__main__':main()
