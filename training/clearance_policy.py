"""Sit/stand candidate with a neutral neck/head target for v07 clearance."""
import numpy as np
from posture_policy import PosturePolicy,STAND,SIT
from evaluate_motion import session

class SitStandPolicy(PosturePolicy):
    def __init__(self,stand_parameters=None,sit_parameters=None):
        super().__init__(session(STAND),session(SIT),stand_parameters,sit_parameters)
    def run(self,names,inputs):
        action=super().run(names,inputs)[0]
        # Offset zero holds the pinned default neck/head pitch, both 0.3491 rad.
        action[:,5:7]=0.
        return [action]

def export(out,stand_parameters=None,sit_parameters=None):
    """Frozen actors plus explicit posture selection; observation slot 48 is sit."""
    import json,onnx
    from pathlib import Path
    from posture_policy import pose_bias
    sp=np.zeros(8,np.float32) if stand_parameters is None else np.asarray(stand_parameters,np.float32)
    qp=np.r_[1.,np.zeros(8)].astype(np.float32) if sit_parameters is None else np.asarray(sit_parameters,np.float32)
    assert sp.shape==(8,) and qp.shape==(9,) and np.isfinite(sp).all() and np.isfinite(qp).all()
    base=onnx.load(STAND);seat=onnx.load(SIT);nodes=[];initializers=[]
    def const(name,value,dtype=np.float32):
        initializers.append(onnx.numpy_helper.from_array(np.asarray(value,dtype),name));return name
    def node(op,args,name,**kw):nodes.append(onnx.helper.make_node(op,args,[name],**kw));return name
    mask=np.ones(61,np.float32);mask[48:61]=0
    obs=node('Mul',['observation',const('command_mask',mask)],'stand_observation')
    flag=np.zeros(61,np.float32);flag[48]=qp[0]
    sit_obs=node('Add',[obs,const('sit_flag',flag)],'sit_observation')
    outputs=[]
    for prefix,model,input_name,p in [('stand_',base,obs,sp),('sit_',seat,sit_obs,qp[1:])]:
        model=onnx.compose.add_prefix(model,prefix);inp=model.graph.input[0].name;output=model.graph.output[0].name
        for n in model.graph.node:
            for i,v in enumerate(n.input):
                if v==inp:n.input[i]=input_name
        nodes.extend(model.graph.node);initializers.extend(model.graph.initializer)
        mat=np.zeros((61,14),np.float32);mat[3,4]=p[6];mat[3,13]=-p[6];mat[1,4]=p[7];mat[1,13]=-p[7]
        bias=node('Add',[node('MatMul',[obs,const(prefix+'feedback_matrix',mat)],prefix+'feedback'),const(prefix+'pose_bias',pose_bias(p[:6]))],prefix+'correction')
        bias=node('Clip',[bias,const(prefix+'min',-.5),const(prefix+'max',.5)],prefix+'bounded')
        outputs.append(node('Add',[output,bias],prefix+'adjusted'))
    flag=node('Gather',['observation',const('flag_index',[48],np.int64)],'requested_posture',axis=1)
    choose=node('GreaterOrEqual',[flag,const('threshold',.5)],'sit_requested')
    selected=node('Where',[choose,outputs[1],outputs[0]],'selected_action')
    head_mask=np.ones(14,np.float32);head_mask[5:7]=0
    node('Mul',[selected,const('head_mask',head_mask)],'action')
    graph=onnx.helper.make_graph(nodes,'v07_neutral_head_sitstand',[onnx.helper.make_tensor_value_info('observation',onnx.TensorProto.FLOAT,[1,61])],[onnx.helper.make_tensor_value_info('action',onnx.TensorProto.FLOAT,[1,14])],initializers)
    model=onnx.helper.make_model(graph,opset_imports=base.opset_import);model.ir_version=base.ir_version
    for key,value in [('simulation_only','true'),('interface','observation[48]: 1=sit, 0=stand; not a velocity command'),('clearance_constraint','neutral neck/head pitch during posture transitions')]:
        prop=model.metadata_props.add();prop.key=key;prop.value=value
    onnx.checker.check_model(model);out=Path(out);out.mkdir(parents=True,exist_ok=True);onnx.save(model,out/'policy.onnx')
    (out/'parameters.json').write_text(json.dumps(dict(method='neutral_head_sitstand',stand_parameters=sp.tolist(),sit_parameters=qp.tolist(),parameters=[],status='simulation_candidate'),indent=2)+'\n')

def export_walking(source,out,parameters=None):
    """Retain velocity commands and leg control, holding neutral head pitch."""
    import json,onnx
    from pathlib import Path
    source=Path(source);out=Path(out);model=onnx.load(source);output=model.graph.output[0].name
    for n in model.graph.node:
        for i,name in enumerate(n.output):
            if name==output:n.output[i]='clearance_original_walk_action'
    p=np.zeros(8,np.float32) if parameters is None else np.asarray(parameters,np.float32)
    assert p.shape==(8,) and np.isfinite(p).all()
    from posture_policy import pose_bias
    inp=model.graph.input[0].name;feedback=np.zeros((61,14),np.float32)
    feedback[3,4]=p[6];feedback[3,13]=-p[6];feedback[1,4]=p[7];feedback[1,13]=-p[7]
    for name,value in [('clearance_walk_feedback',feedback),('clearance_walk_bias',pose_bias(p[:6])),('clearance_walk_min',np.float32(-.5)),('clearance_walk_max',np.float32(.5))]:
        model.graph.initializer.append(onnx.numpy_helper.from_array(value,name))
    model.graph.node.extend([onnx.helper.make_node('MatMul',[inp,'clearance_walk_feedback'],['clearance_walk_linear']),onnx.helper.make_node('Add',['clearance_walk_linear','clearance_walk_bias'],['clearance_walk_correction']),onnx.helper.make_node('Clip',['clearance_walk_correction','clearance_walk_min','clearance_walk_max'],['clearance_walk_bounded']),onnx.helper.make_node('Add',['clearance_original_walk_action','clearance_walk_bounded'],['clearance_walk_adjusted'])])
    mask=np.ones(14,np.float32);mask[5:7]=0
    model.graph.initializer.append(onnx.numpy_helper.from_array(mask,'clearance_walk_head_mask'))
    model.graph.node.append(onnx.helper.make_node('Mul',['clearance_walk_adjusted','clearance_walk_head_mask'],[output]))
    prop=model.metadata_props.add();prop.key='cad_revision';prop.value='v07'
    prop=model.metadata_props.add();prop.key='head_constraint';prop.value='Neutral neck/head pitch; velocity command interface retained'
    onnx.checker.check_model(model);out.mkdir(parents=True,exist_ok=True);onnx.save(model,out/'policy.onnx')
    (out/'parameters.json').write_text(json.dumps(dict(method='neutral_head_walking',base_policy=str(source),neutral_head_offset=0.,parameters=p.tolist(),status='simulation_candidate'),indent=2)+'\n')


class WalkingPolicy:
    def __init__(self,base,parameters):
        self.base=base;self.parameters=np.asarray(parameters,np.float32)
        assert self.parameters.shape==(8,) and np.isfinite(self.parameters).all()
    def get_inputs(self):return self.base.get_inputs()
    def run(self,names,inputs):
        from posture_policy import pose_bias
        obs=next(iter(inputs.values()));action=self.base.run(None,inputs)[0].copy();p=self.parameters
        for i,o in enumerate(obs):
            bias=pose_bias(p[:6]);pitch=p[6]*o[3]+p[7]*o[1];bias[[4,13]]+=[pitch,-pitch]
            action[i]+=np.clip(bias,-.5,.5)
        action[:,5:7]=0.
        return [action]
