"""Bounded observable terrain adaptation; no terrain map enters the actor."""
from pathlib import Path
import json,hashlib
import numpy as np
import onnx

BASE=Path(__file__).resolve().parents[1]/'results/posture-v07/policies/walking-adjusted/policy.onnx'
BOUNDS=np.array([.08,.08,.08,1.,.25,1.,.25,.25],np.float32)

def coefficients(parameters):
    p=np.asarray(parameters,np.float32);assert p.shape==(8,) and np.isfinite(p).all() and np.all(abs(p)<=BOUNDS+1e-6)
    from posture_policy import pose_bias
    bias=pose_bias([*p[:3],0,0,0]);weights=np.zeros((61,14),np.float32)
    weights[3,[4,13]]=[p[3],-p[3]];weights[1,[4,13]]=[p[4],-p[4]]
    weights[4,[1,10]]=[-p[5],p[5]];weights[0,[1,10]]=[-p[6],p[6]]
    weights[50,[0,9]]=p[7];weights[2,[0,9]]=-p[7]
    return bias,weights

class TerrainPolicy:
    def __init__(self,base,parameters):self.base=base;self.bias,self.weights=coefficients(parameters)
    def get_inputs(self):return self.base.get_inputs()
    def run(self,names,inputs):
        obs=next(iter(inputs.values()));gate=np.clip(np.sum(abs(obs[:,48:51]),axis=1,keepdims=True)/.05,0,1)
        action=self.base.run(names,inputs)[0]
        return [action+gate*np.clip(obs@self.weights+self.bias,-.12,.12)]

def export(parameters,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);model=onnx.load(BASE);output=model.graph.output[0].name;inp=model.graph.input[0].name
    for n in model.graph.node:
        for i,name in enumerate(n.output):
            if name==output:n.output[i]='terrain_base_action'
    bias,weights=coefficients(parameters);nodes=[];tensors=[]
    def const(name,value,dtype=np.float32):
        name='terrain_'+name;tensors.append(onnx.numpy_helper.from_array(np.asarray(value,dtype),name));return name
    def node(op,args,name,**kw):
        name='terrain_'+name;nodes.append(onnx.helper.make_node(op,args,[name],**kw));return name
    command=node('Gather',[inp,const('command_indices',[48,49,50],np.int64)],'command',axis=1)
    command=node('Abs',[command],'absolute_command');total=node('ReduceSum',[command,const('axis',[1],np.int64)],'command_total',keepdims=1)
    gate=node('Clip',[node('Div',[total,const('gate_scale',.05)],'motion_fraction'),const('zero',0.),const('one',1.)],'motion_gate')
    linear=node('MatMul',[inp,const('feedback',weights)],'linear');correction=node('Add',[linear,const('bias',bias)],'correction')
    bounded=node('Clip',[correction,const('min',-.12),const('max',.12)],'bounded');active=node('Mul',[gate,bounded],'active')
    nodes.append(onnx.helper.make_node('Add',['terrain_base_action',active],[output]))
    model.graph.node.extend(nodes);model.graph.initializer.extend(tensors)
    for key,value in [('terrain_adaptation','Bounded IMU/command correction; no terrain sensor; simulation only'),('terrain_correction_bound_rad','0.12')]:
        prop=model.metadata_props.add();prop.key=key;prop.value=value
    onnx.checker.check_model(model);onnx.save(model,folder/'policy.onnx')
    (folder/'parameters.json').write_text(json.dumps(dict(method='CEM bounded IMU and command feedback',parameters=list(map(float,parameters)),base_policy='results/posture-v07/policies/walking-adjusted/policy.onnx',base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),correction_bound_rad=.12,stopped_actor_unchanged=True,head_pitch_unchanged=True,status='unselected_simulation_candidate'),indent=2)+'\n')
