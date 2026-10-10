"""Command-conditioned calibration around the pinned walking network.

No hidden state, clock, external forces or new observation slots. Six profiles
map body-frame commands into the teacher's gait range and add bounded IMU
feedback. Stop preserves the original network exactly.
"""
import json
from pathlib import Path
import numpy as np
import onnx
from tune_fast_bias import BASIS
from tune_fast_feedback import DYNAMIC
from paths import ROOT

NAMES=['forward','backward','left','right','turn_left','turn_right']
REFERENCES=np.array([[.2,0,0],[-.15,0,0],[0,.1,0],[0,-.1,0],[0,0,.4],[0,0,-.4]],np.float32)
INITIAL=np.zeros((6,15),np.float32)
INITIAL[:,:3]=[[.32,0,0],[-.45,0,0],[0,.4,0],[0,-.4,0],[0,0,.85],[0,0,-1.2]]
INITIAL[:,3]=.7
LIMIT=.12

def weights(obs):
    c=obs[:,48:51]
    return np.clip(np.column_stack([c[:,0]/.2,-c[:,0]/.15,c[:,1]/.1,-c[:,1]/.1,c[:,2]/.4,-c[:,2]/.4]),0,1)

def feedback_matrix(p):
    w=np.zeros((61,14),np.float32)
    for gain,slot,basis in zip(p[10:],[2,1,0,3,4],DYNAMIC):w[slot]+=gain*basis
    w[50]-=p[10]*DYNAMIC[0]
    return w

class MotionPolicy:
    def __init__(self,teacher,parameters):
        self.teacher=teacher;self.parameters=np.array(parameters,np.float32)
        assert self.parameters.shape==(6,15)
        self.bias=self.parameters[:,4:10]@BASIS
        self.feedback=np.array([feedback_matrix(p) for p in self.parameters])
    def get_inputs(self):return self.teacher.get_inputs()
    def run(self,names,inputs):
        obs=next(iter(inputs.values()));w=weights(obs)
        mix=w/np.maximum(w.sum(axis=1,keepdims=True),1.)
        proxy=w@self.parameters[:,:3]
        proxy[:,2]+=(mix@self.parameters[:,3])*(obs[:,50]-obs[:,2])
        proxy=np.clip(proxy,[-.65,-.7,-2.],[.6,.7,2.]).astype(np.float32)
        warped=obs.copy();warped[:,48:51]=proxy
        action=self.teacher.run(names,{self.get_inputs()[0].name:warped})[0]
        correction=mix@self.bias
        for i in range(6):correction+=mix[:,i:i+1]*(obs@self.feedback[i])
        return [action+np.clip(correction,-LIMIT,LIMIT)]

def export(parameters,path):
    """Build the same command map and correction into one portable ONNX."""
    p=np.array(parameters,np.float32);model=onnx.load(ROOT/'tmp/sim-inputs/alpha_walking.onnx')
    inp=model.graph.input[0].name;output=model.graph.output[0].name
    nodes=[];initializers=[]
    def const(name,v,dtype=np.float32):
        initializers.append(onnx.numpy_helper.from_array(np.array(v,dtype=dtype),'motion_'+name));return 'motion_'+name
    def node(op,args,out,**kw):nodes.append(onnx.helper.make_node(op,args,['motion_'+out],**kw));return 'motion_'+out
    zero=const('zero',0.);one=const('one',1.)
    cmdmap=np.zeros((61,6),np.float32)
    for i,(slot,scale) in enumerate([(48,5.),(48,-1/.15),(49,10.),(49,-10.),(50,2.5),(50,-2.5)]):cmdmap[slot,i]=scale
    raww=node('MatMul',[inp,const('cmdmap',cmdmap)],'raw_weights')
    w=node('Clip',[raww,zero,one],'weights')
    total=node('ReduceSum',[w,const('sum_axes',[1],np.int64)],'total',keepdims=1)
    denom=node('Max',[total,one],'denom');mix=node('Div',[w,denom],'mix')
    proxy=node('MatMul',[w,const('proxy_parameters',p[:,:3])],'proxy')
    gain=node('MatMul',[mix,const('yaw_gain',p[:,3:4])],'yaw_gain_value')
    error=node('MatMul',[inp,const('yaw_error',np.eye(61,dtype=np.float32)[:,50:51]-np.eye(61,dtype=np.float32)[:,2:3])],'yaw_error_value')
    yawfix=node('Mul',[gain,error],'yaw_feedback');yawvec=node('MatMul',[yawfix,const('yaw_axis',[[0,0,1.]])],'yaw_vector')
    corrected=node('Add',[proxy,yawvec],'proxy_corrected')
    proxylo=node('Max',[corrected,const('proxy_min',[-.65,-.7,-2.])],'proxy_lower')
    proxybounded=node('Min',[proxylo,const('proxy_max',[.6,.7,2.])],'proxy_bounded')
    mask=np.ones(61,np.float32);mask[48:51]=0
    kept=node('Mul',[inp,const('obs_mask',mask)],'kept_observation')
    insert=np.zeros((3,61),np.float32);insert[:,48:51]=np.eye(3)
    proxyfull=node('MatMul',[proxybounded,const('insert',insert)],'proxy_full')
    warped=node('Add',[kept,proxyfull],'warped_observation')
    oldnodes=list(model.graph.node)
    for n in oldnodes:
        for i,name in enumerate(n.input):
            if name==inp:n.input[i]=warped
        for i,name in enumerate(n.output):
            if name==output:n.output[i]='motion_teacher_actions'
    nodes.extend(oldnodes)
    static=node('MatMul',[mix,const('bias',p[:,4:10]@BASIS)],'static_correction');correction=static
    for i in range(6):
        m=node('Gather',[mix,const(f'index_{i}',[i],np.int64)],f'mix_{i}',axis=1)
        dyn=node('MatMul',[inp,const(f'feedback_{i}',feedback_matrix(p[i]))],f'dynamic_{i}')
        scaled=node('Mul',[m,dyn],f'scaled_{i}');correction=node('Add',[correction,scaled],f'sum_{i}')
    bounded=node('Clip',[correction,const('angle_min',-LIMIT),const('angle_max',LIMIT)],'bounded_correction')
    nodes.append(onnx.helper.make_node('Add',['motion_teacher_actions',bounded],[output]))
    del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(initializers)
    # Retain the teacher's opset and normalizer without reinterpretation.
    for key,value in [('mount_revision','v06'),('deployment_status','simulation_only'),('motion_interface','body-frame vx m/s, vy m/s, wz rad/s')]:
        prop=model.metadata_props.add();prop.key=key;prop.value=value
    onnx.checker.check_model(model);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,path)

def save(parameters,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);export(parameters,out/'policy.onnx')
    (out/'parameters.json').write_text(json.dumps(dict(profiles=NAMES,reference_commands=REFERENCES.tolist(),parameters=np.array(parameters).tolist(),correction_limit_rad=LIMIT,proxy_limits=[[-.65,-.7,-2],[.6,.7,2]],description='Paired CEM calibration; exact teacher at stop; raw observable IMU feedback; no assistance.'),indent=2)+'\n')
