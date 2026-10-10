"""Observable turn-start boost to escape the inherited standing attractor."""
import json
from pathlib import Path
import numpy as np
import onnx
from motion_policy import MotionPolicy,save as save_base

class ActivationPolicy(MotionPolicy):
    def __init__(self,teacher,parameters,boost):super().__init__(teacher,parameters);self.boost=float(boost)
    def run(self,names,inputs):
        from motion_policy import weights,LIMIT
        obs=next(iter(inputs.values()));w=weights(obs);mix=w/np.maximum(w.sum(axis=1,keepdims=True),1.)
        proxy=w@self.parameters[:,:3];proxy[:,2]+=(mix@self.parameters[:,3])*(obs[:,50]-obs[:,2])
        activation=np.clip(1-np.abs(obs[:,2])/.2,0,1)*np.clip(obs[:,50]/.4,-1,1)
        activation*=np.sum(np.abs(obs[:,48:50]),axis=1)<=.01
        proxy[:,2]+=self.boost*activation
        warped=obs.copy();warped[:,48:51]=np.clip(proxy,[-.65,-.7,-2.],[.6,.7,2.])
        action=self.teacher.run(names,{self.get_inputs()[0].name:warped})[0];correction=mix@self.bias
        for i in range(6):correction+=mix[:,i:i+1]*(obs@self.feedback[i])
        return [action+np.clip(correction,-LIMIT,LIMIT)]

def save(parameters,boost,out):
    save_base(parameters,out);out=Path(out);path=out/'policy.onnx';model=onnx.load(path);inp=model.graph.input[0].name
    initializers=[];extra=[]
    def const(n,v,dtype=np.float32):initializers.append(onnx.numpy_helper.from_array(np.array(v,dtype=dtype),'activation_'+n));return 'activation_'+n
    def node(op,args,n,**kw):extra.append(onnx.helper.make_node(op,args,['activation_'+n],**kw));return 'activation_'+n
    zero=const('zero',0.);one=const('one',1.)
    gyro=node('Gather',[inp,const('gyro_index',[2],np.int64)],'gyro',axis=1);absw=node('Abs',[gyro],'abs_gyro')
    scaled=node('Div',[absw,const('gyro_threshold',.2)],'scaled_gyro');low=node('Sub',[one,scaled],'low_gyro')
    gate=node('Clip',[low,zero,one],'gyro_gate')
    command=node('Gather',[inp,const('command_index',[50],np.int64)],'command',axis=1)
    direction=node('Div',[command,const('command_scale',.4)],'direction_unclipped')
    direction=node('Clip',[direction,const('minus_one',-1.),one],'direction')
    xy=node('Gather',[inp,const('xy_indices',[48,49],np.int64)],'xy',axis=1);xyabs=node('Abs',[xy],'xy_abs')
    total=node('ReduceSum',[xyabs,const('sum_axis',[1],np.int64)],'xy_total',keepdims=1)
    stationary=node('LessOrEqual',[total,const('xy_threshold',.01)],'pure_turn')
    stationary=node('Cast',[stationary],'pure_turn_float',to=onnx.TensorProto.FLOAT)
    value=node('Mul',[gate,direction],'signed_gate');value=node('Mul',[value,stationary],'gated_boost')
    value=node('Mul',[value,const('boost',boost)],'boost_value');vector=node('MatMul',[value,const('yaw_axis',[[0.,0.,1.]])],'vector')
    nodes=list(model.graph.node);index=next(i for i,n in enumerate(nodes) if 'motion_proxy_corrected' in n.output)
    nodes[index].output[0]='motion_proxy_before_activation'
    extra.append(onnx.helper.make_node('Add',['motion_proxy_before_activation',vector],['motion_proxy_corrected']))
    nodes[index+1:index+1]=extra;del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model);onnx.save(model,path)
    p=json.loads((out/'parameters.json').read_text());p['turn_activation_boost']=float(boost)
    p['activation_rule']='During pure turns only: boost*clip(1-abs(raw gyroZ)/0.2,0,1)*clip(raw wz/0.4,-1,1), added to teacher yaw command; no hidden state or clock.'
    (out/'parameters.json').write_text(json.dumps(p,indent=2)+'\n')
