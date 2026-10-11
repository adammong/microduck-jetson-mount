"""Gravity-conditioned mounted get-up adaptation; no state teleport or assist."""
import numpy as np
from posture_policy import pose_bias

class RecoveryPolicy:
    def __init__(self,base,parameters):
        self.base=base;self.parameters=np.asarray(parameters,np.float32)
        assert self.parameters.shape==(14,) and np.isfinite(self.parameters).all() and np.max(abs(self.parameters))<=1.5
    def get_inputs(self):return self.base.get_inputs()
    def run(self,names,inputs):
        obs=next(iter(inputs.values())).copy();obs[:,48:61]=0
        action=self.base.run(None,{self.get_inputs()[0].name:obs})[0]
        for i,o in enumerate(obs):
            p=self.parameters;gate=np.clip((np.max(abs(o[3:5]))-.12)/.4,0,1)
            correction=pose_bias(p[:6])*gate
            pitch=p[6]*o[3]+p[7]*o[1];roll=p[8]*o[4]+p[9]*o[0]
            correction[[4,13]]+=[pitch,-pitch];correction[[1,10]]+=roll
            correction[[2,11]]+=gate*p[10]*o[4];correction[[3,12]]+=gate*p[11]*o[4]
            correction[5]+=gate*p[12]*o[3];correction[6]+=gate*p[13]*o[3]
            action[i]+=np.clip(correction,-.75,.75)
        return [action]

def export(parameters,out,revision="v06",scope=None):
    import json
    from pathlib import Path
    import onnx
    from posture_policy import STAND
    p=np.asarray(parameters,np.float32);assert p.shape==(14,)
    model=onnx.load(STAND);inp=model.graph.input[0].name;output=model.graph.output[0].name
    nodes=[];initializers=[]
    def const(name,value,dtype=np.float32):
        name='recovery_'+name;initializers.append(onnx.numpy_helper.from_array(np.asarray(value,dtype=dtype),name));return name
    def node(op,args,name,**kw):
        name='recovery_'+name;nodes.append(onnx.helper.make_node(op,args,[name],**kw));return name
    mask=np.ones(61,np.float32);mask[48:61]=0;kept=node('Mul',[inp,const('mask',mask)],'observations')
    gravity=node('Gather',[kept,const('gravity_indices',[3,4],np.int64)],'gravity',axis=1)
    tilt=node('ReduceMax',[node('Abs',[gravity],'abs_gravity'),const('axes',[1],np.int64)],'tilt',keepdims=1)
    gate=node('Clip',[node('Div',[node('Sub',[tilt,const('tilt_start',.12)],'tilt_offset'),const('tilt_range',.4)],'tilt_fraction'),const('zero',0.),const('one',1.)],'gate')
    gated=np.zeros((61,14),np.float32)
    gated[4,[2,11]]=p[10];gated[4,[3,12]]=p[11];gated[3,5]=p[12];gated[3,6]=p[13]
    correction=node('Add',[node('MatMul',[kept,const('gated_matrix',gated)],'gated_linear'),const('pose_bias',pose_bias(p[:6]))],'gated_correction')
    correction=node('Mul',[correction,gate],'active_correction')
    feedback=np.zeros((61,14),np.float32)
    feedback[3,4]=p[6];feedback[3,13]=-p[6];feedback[1,4]=p[7];feedback[1,13]=-p[7]
    feedback[4,[1,10]]=p[8];feedback[0,[1,10]]=p[9]
    correction=node('Add',[correction,node('MatMul',[kept,const('feedback_matrix',feedback)],'feedback')],'total_correction')
    correction=node('Clip',[correction,const('minimum',-.75),const('maximum',.75)],'bounded_correction')
    original=list(model.graph.node)
    for n in original:
        for i,name in enumerate(n.input):
            if name==inp:n.input[i]=kept
        for i,name in enumerate(n.output):
            if name==output:n.output[i]='recovery_base_action'
    nodes.extend(original);nodes.append(onnx.helper.make_node('Add',['recovery_base_action',correction],[output]))
    del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(initializers)
    prop=model.metadata_props.add();prop.key='simulation_only';prop.value='true'
    prop=model.metadata_props.add();prop.key='validated_scope';prop.value=scope or 'Limited mounted front recovery; full sitting and back/side recovery unresolved'
    if revision!='v06':
        prop=model.metadata_props.add();prop.key='cad_revision';prop.value=revision
    onnx.checker.check_model(model);out=Path(out);out.mkdir(parents=True,exist_ok=True);onnx.save(model,out/'policy.onnx')
    (out/'parameters.json').write_text(json.dumps(dict(parameters=list(map(float,p)),method='gravity_conditioned_recovery',status='limited_simulation_candidate',additional_correction_limit_rad=.75,base_policy='tmp/sim-inputs/alpha_stand.onnx'),indent=2)+'\n')
