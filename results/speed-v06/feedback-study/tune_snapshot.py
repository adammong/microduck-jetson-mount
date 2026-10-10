"""Bounded CEM study of an observable, angle-limited feedback correction."""
import argparse,json,multiprocessing as mp,shutil
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from tune_fast_bias import BASIS,export_bias
from evaluate_speed import trial
from paths import ROOT

DYNAMIC=np.zeros((5,14),np.float32)
DYNAMIC[0,[0,9]]=1
DYNAMIC[1,[2,11]]=[1,-1]
DYNAMIC[2,[1,10]]=[1,-1]
DYNAMIC[3]=DYNAMIC[1];DYNAMIC[4]=DYNAMIC[2]
_teacher=None

def matrix(params):
    weights=np.zeros((61,14),np.float32)
    for gain,slot,basis in zip(params[6:],[2,1,0,3,4],DYNAMIC):weights[slot]+=gain*basis
    weights[50]-=params[6]*DYNAMIC[0]
    return weights

class FeedbackPolicy:
    def __init__(self,teacher,params):
        self.teacher=teacher;self.bias=np.array(params[:6],np.float32)@BASIS;self.weights=matrix(params)
    def get_inputs(self):return self.teacher.get_inputs()
    def run(self,names,inputs):
        obs=next(iter(inputs.values()));gate=np.clip((obs[:,48:49]-.35)/.1,0,1)
        correction=np.clip(obs@self.weights+self.bias[None],-.08,.08)
        return [self.teacher.run(names,inputs)[0]+gate*correction]

def worker_init():
    global _teacher
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    _teacher=ort.InferenceSession(str(ROOT/'tmp/sim-inputs/alpha_walking.onnx'),opts,providers=['CPUExecutionProvider'])

def score(job):
    index,params,command,seconds,seeds=job;policy=FeedbackPolicy(_teacher,params)
    rows=[trial(policy,command,seed,seconds) for seed in seeds]
    value=float(np.mean([r['mean_body_forward_m_s']-1.5*abs(r['mean_body_yaw_rate_rad_s']) for r in rows]))
    failed=any(not r['survived'] or r['nonfoot_ground_fraction']>0 or r['payload_selfcontact_fraction']>0 for r in rows)
    if failed:value-=1.
    value-=2.*float(np.mean([r['payload_selfcontact_fraction'] for r in rows]))
    value-=.01*float(np.linalg.norm(params)/.06)
    return dict(index=index,params=list(params),fitness=value,feasible=not failed,trials=rows)

def export(params,path):
    export_bias(params[:6],path);model=onnx.load(path)
    # The static export gates fast_bias. Replace that input with the bounded
    # sum of the bias and a linear map of raw gyro/gravity/command slots.
    model.graph.initializer.extend([onnx.numpy_helper.from_array(matrix(params),'feedback_weights'),
        onnx.numpy_helper.from_array(np.array(-.08,np.float32),'feedback_min'),
        onnx.numpy_helper.from_array(np.array(.08,np.float32),'feedback_max')])
    inp=model.graph.input[0].name;nodes=list(model.graph.node)
    index=next(i for i,n in enumerate(nodes) if n.op_type=='Mul' and 'fast_bias' in n.input)
    nodes[index].input[1]='feedback_bounded'
    nodes[index:index]=[onnx.helper.make_node('MatMul',[inp,'feedback_weights'],['feedback_linear']),
        onnx.helper.make_node('Add',['feedback_linear','fast_bias'],['feedback_raw']),
        onnx.helper.make_node('Clip',['feedback_raw','feedback_min','feedback_max'],['feedback_bounded'])]
    del model.graph.node[:];model.graph.node.extend(nodes);onnx.checker.check_model(model);onnx.save(model,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--generations',type=int,default=8);ap.add_argument('--population',type=int,default=20)
    ap.add_argument('--command',type=float,default=.5);ap.add_argument('--seconds',type=float,default=25.)
    ap.add_argument('--training-seeds',nargs='+',type=int,default=list(range(12)))
    ap.add_argument('--seed',type=int,default=74);ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed);mean=np.zeros(11);std=np.ones(11)*.018;best=None;history=[]
    with mp.get_context('spawn').Pool(a.workers,initializer=worker_init) as pool:
        for g in range(a.generations):
            pop=np.clip(rng.normal(mean,std,(a.population,11)),-.06,.06)
            pop[0]=mean if best is None else best['params'];pop[1]=0.
            results=pool.map(score,[(i,p.tolist(),a.command,a.seconds,a.training_seeds) for i,p in enumerate(pop)])
            results.sort(key=lambda r:r['fitness'],reverse=True)
            elite=np.array([r['params'] for r in results[:max(3,a.population//4)]])
            mean=.3*mean+.7*elite.mean(axis=0);std=np.maximum(.004,.3*std+.7*elite.std(axis=0))
            if best is None or results[0]['fitness']>best['fitness']:best=results[0]
            history.append(dict(generation=g,candidates=results));export(best['params'],a.out/'candidate.onnx')
            record=dict(method='CEM bounded observable feedback',training_seeds=a.training_seeds,command_m_s=a.command,
                seconds=a.seconds,seed=a.seed,parameter_bound=.06,correction_bound_rad=.08,
                fitness='mean body forward - 1.5*abs(mean gyro yaw) - 0.01*norm(params)/0.06; reject falls/contacts, penalize contact time',
                feature_slots=[2,1,0,3,4,50],gate='clip((raw forward command-0.35)/0.1,0,1)',
                static_basis=BASIS.tolist(),dynamic_basis=DYNAMIC.tolist(),best=best,history=history)
            (a.out/'search.json').write_text(json.dumps(record,indent=2)+'\n')
            print('Generation',g,'fitness',round(best['fitness'],3),'feasible',best['feasible'],
                'body',round(np.mean([r['mean_body_forward_m_s'] for r in best['trials']]),3),
                'net',round(np.mean([r['net_forward_m_s'] for r in best['trials']]),3),
                'yaw',round(np.mean([abs(r['yaw_change_rad']) for r in best['trials']]),3),flush=True)
    worker_init();wrapped=FeedbackPolicy(_teacher,best['params'])
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    exported=ort.InferenceSession(str(a.out/'candidate.onnx'),opts,providers=['CPUExecutionProvider']);error=0.
    for i in range(100):
        obs=rng.normal(0,.3,(1,61)).astype(np.float32);obs[0,48]=[0,.3,.5][i%3]
        data={_teacher.get_inputs()[0].name:obs}
        error=max(error,float(np.max(abs(exported.run(None,data)[0]-wrapped.run(None,data)[0]))))
    assert error<1e-6
    (a.out/'export-verification.json').write_text(json.dumps(dict(max_action_error_rad=error,correction_bound_rad=.08),indent=2)+'\n')
    shutil.copyfile(__file__,a.out/'tune_snapshot.py')
if __name__=='__main__':main()
