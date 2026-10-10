"""Bounded evolutionary study of small gait corrections; real payload/BAM.

This is an alternative to PPO, with a trajectory speed/gyro fitness. Every
candidate remains the teacher plus <=0.06 rad joint corrections. Separate
held-out rollouts must validate the chosen candidate before it is selected.
"""
import argparse,json,os,hashlib,multiprocessing as mp
from pathlib import Path
import numpy as np
import onnx,onnxruntime as ort
from evaluate_speed import trial
from paths import ROOT

# Independent yaw corrections, mirrored roll/pitch/ankle corrections, and
# a paired neck/head pitch correction. These are actions in radians.
BASIS=np.zeros((6,14),np.float32)
BASIS[0,0]=1;BASIS[1,9]=1
BASIS[2,[1,10]]=[1,-1]
BASIS[3,[2,11]]=[1,-1]
BASIS[4,[4,13]]=[1,-1]
BASIS[5,[5,6]]=1
_teacher=None

class CorrectedPolicy:
    def __init__(self,teacher,bias):self.teacher=teacher;self.bias=bias
    def get_inputs(self):return self.teacher.get_inputs()
    def run(self,names,inputs):
        obs=next(iter(inputs.values()))
        # The stand command retains the original teacher exactly.
        gate=np.minimum(1.,np.abs(obs[:,48:49])/.35)
        action=self.teacher.run(names,inputs)[0]+gate*self.bias[None]
        return [action]

def worker_init():
    global _teacher
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    _teacher=ort.InferenceSession(str(ROOT/'tmp/sim-inputs/alpha_walking.onnx'),opts,providers=['CPUExecutionProvider'])

def score(job):
    index,params,command,seconds,seeds=job
    bias=np.array(params,np.float32)@BASIS
    policy=CorrectedPolicy(_teacher,bias);rows=[trial(policy,command,seed,seconds) for seed in seeds]
    # Body-forward speed and relative yaw drift are trajectory measurements,
    # not hidden observations or added motor assistance.
    value=float(np.mean([r['mean_body_forward_m_s']-.16*abs(r['yaw_change_rad'])/r['seconds'] for r in rows]))
    if any(not r['survived'] or r['nonfoot_ground_fraction']>0 or r['payload_selfcontact_fraction']>0 for r in rows):value-=1.
    value-=.04*float(np.linalg.norm(params)/.06)
    return dict(index=index,params=list(params),bias_rad=bias.tolist(),fitness=value,trials=rows)

def export_bias(params,path):
    source=ROOT/'tmp/sim-inputs/alpha_walking.onnx';model=onnx.load(source)
    output=model.graph.output[0].name
    # Rename just the teacher output, then add the correction gated by the
    # existing raw forward-command slot. Observations/actions stay 61/14.
    for node in model.graph.node:
        for i,name in enumerate(node.output):
            if name==output:node.output[i]='fast_teacher_actions'
    bias=np.array(params,np.float32)@BASIS
    for name,array in [('fast_bias',bias[None]),('fast_index',np.array([48],np.int64)),
        ('fast_scale',np.array(.35,np.float32)),('fast_min',np.array(0,np.float32)),('fast_max',np.array(1,np.float32))]:
        model.graph.initializer.append(onnx.numpy_helper.from_array(array,name))
    inp=model.graph.input[0].name
    nodes=[onnx.helper.make_node('Gather',[inp,'fast_index'],['fast_command'],axis=1),
        onnx.helper.make_node('Abs',['fast_command'],['fast_abs']),
        onnx.helper.make_node('Div',['fast_abs','fast_scale'],['fast_ratio']),
        onnx.helper.make_node('Clip',['fast_ratio','fast_min','fast_max'],['fast_gate']),
        onnx.helper.make_node('Mul',['fast_gate','fast_bias'],['fast_correction']),
        onnx.helper.make_node('Add',['fast_teacher_actions','fast_correction'],[output])]
    model.graph.node.extend(nodes);onnx.checker.check_model(model)
    path.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,path)
    return bias

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/speed-v06/bias-study')
    ap.add_argument('--generations',type=int,default=8);ap.add_argument('--population',type=int,default=16)
    ap.add_argument('--command',type=float,default=.65);ap.add_argument('--seconds',type=float,default=10.)
    ap.add_argument('--seed',type=int,default=71);ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed);mean=np.zeros(6);std=np.ones(6)*.025;best=None;history=[]
    with mp.get_context('spawn').Pool(a.workers,initializer=worker_init) as pool:
        for generation in range(a.generations):
            pop=np.clip(rng.normal(mean,std,(a.population,6)),-.06,.06)
            pop[0]=np.zeros(6) if best is None else best['params']
            jobs=[(i,p.tolist(),a.command,a.seconds,[0,1]) for i,p in enumerate(pop)]
            scores=pool.map(score,jobs);scores.sort(key=lambda s:s['fitness'],reverse=True)
            elite=np.array([s['params'] for s in scores[:max(3,a.population//4)]])
            mean=.3*mean+.7*elite.mean(axis=0);std=np.maximum(.004,.3*std+.7*elite.std(axis=0))
            if best is None or scores[0]['fitness']>best['fitness']:best=scores[0]
            history.append(dict(generation=generation,candidates=scores))
            bias=export_bias(best['params'],a.out/'candidate.onnx')
            (a.out/'search.json').write_text(json.dumps(dict(method='bounded CEM gait bias',training_seeds=[0,1],
                command_m_s=a.command,seconds=a.seconds,seed=a.seed,max_joint_bias_rad=.06,
                basis=BASIS.tolist(),best=best,history=history),indent=2)+'\n')
            print('Generation',generation,'fitness',round(best['fitness'],3),
                'body',round(np.mean([r['mean_body_forward_m_s'] for r in best['trials']]),3),
                'net',round(np.mean([r['net_forward_m_s'] for r in best['trials']]),3),
                'yaw',round(np.mean([abs(r['yaw_change_rad']) for r in best['trials']]),3),flush=True)
    # Export agreement on fresh inputs includes both idle and moving gates.
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    teacher=ort.InferenceSession(str(ROOT/'tmp/sim-inputs/alpha_walking.onnx'),opts,providers=['CPUExecutionProvider'])
    exported=ort.InferenceSession(str(a.out/'candidate.onnx'),opts,providers=['CPUExecutionProvider'])
    wrapped=CorrectedPolicy(teacher,bias);error=0.
    for i in range(100):
        obs=rng.normal(0,.3,(1,61)).astype(np.float32);obs[0,48]=[0,.2,.65][i%3]
        data={teacher.get_inputs()[0].name:obs}
        error=max(error,float(np.max(abs(exported.run(None,data)[0]-wrapped.run(None,data)[0]))))
    assert error<1e-6
    (a.out/'export-verification.json').write_text(json.dumps(dict(max_action_error_rad=error,teacher_sha256=hashlib.sha256((ROOT/'tmp/sim-inputs/alpha_walking.onnx').read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
