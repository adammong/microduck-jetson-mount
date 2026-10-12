"""Observable command/IMU adaptation of the frozen mounted terrain actor."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import onnx
from posture_policy import pose_bias

BASE=Path(__file__).resolve().parents[1]/'results/terrain-v07/policies/selected/policy.onnx'
INITIAL=np.r_[1.,1.,np.zeros(14)].astype(np.float32)
LOW=np.array([.7,.8,-4.,-.5,-.3,-.05,-.05,-.05,-1.,-.35,-1.,-.35,-.3,-.3,-.5,0.],np.float32)
HIGH=np.array([1.25,1.5,4.,.5,.6,.05,.05,.05,1.,.35,1.,.35,.3,.3,.5,.05],np.float32)
NAMES=['forward_gain','reverse_gain','pitch_command_gain','roll_lateral_command_gain','yaw_joint_feedback','hip_bias','knee_bias','ankle_bias','ankle_pitch_gravity','ankle_pitch_gyro','hip_roll_gravity','hip_roll_gyro','hip_pitch_gyro','knee_pitch_gyro','knee_pitch_gravity','disturbed_joint_velocity_damping']

def coefficients(p):
    p=np.asarray(p,np.float32);assert p.shape==(16,) and np.isfinite(p).all() and np.all(p>=LOW-1e-6) and np.all(p<=HIGH+1e-6)
    bias=pose_bias([*p[5:8],0,0,0]);w=np.zeros((61,14),np.float32)
    for slot,joints,values in [(3,[4,13],[p[8],-p[8]]),(1,[4,13],[p[9],-p[9]]),(4,[1,10],[-p[10],p[10]]),(0,[1,10],[-p[11],p[11]]),(1,[2,11],[p[12],-p[12]]),(1,[3,12],[p[13],-p[13]]),(3,[3,12],[p[14],-p[14]]),(50,[0,9],[p[4],p[4]]),(2,[0,9],[-p[4],-p[4]])]:w[slot,joints]=values
    mask=np.ones(14,np.float32);mask[5:9]=0
    return p,bias,w,mask

class RobustPolicy:
    def __init__(self,base,parameters):self.base=base;self.p,self.bias,self.w,self.mask=coefficients(parameters)
    def get_inputs(self):return self.base.get_inputs()
    def get_modelmeta(self):return self.base.get_modelmeta()
    def run(self,names,inputs):
        name=self.get_inputs()[0].name;obs=inputs[name];p=self.p;warped=obs.copy()
        move=np.clip(abs(obs[:,48:51]).sum(1,keepdims=True)/.05,0,1)
        translate=np.clip(abs(obs[:,48:50]).sum(1,keepdims=True)/.05,0,1)
        disturbed=np.maximum(np.clip((abs(obs[:,3:5]).max(1,keepdims=True)-.06)/.08,0,1),np.clip((abs(obs[:,:2]).max(1,keepdims=True)-.5)/1.,0,1))
        gain=np.where(obs[:,48:49]>=0,p[0],p[1])*(1+p[2]*obs[:,3:4])
        warped[:,48:49]=np.clip(obs[:,48:49]*gain,-.25,.3)
        warped[:,49:50]=np.clip(obs[:,49:50]+p[3]*obs[:,4:5]*translate,-.15,.15)
        action=self.base.run(names,{name:warped})[0]
        correction=move*self.bias+np.maximum(move,disturbed)*(obs@self.w)-p[15]*disturbed*obs[:,20:34]*self.mask
        return [action+np.clip(correction,-.12,.12)]

def export(parameters,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);p,bias,w,mask=coefficients(parameters);model=onnx.load(BASE);inp=model.graph.input[0].name;out=model.graph.output[0].name;nodes=[];tensors=[]
    def const(name,value,dtype=np.float32):
        name='robust_'+name;tensors.append(onnx.numpy_helper.from_array(np.asarray(value,dtype),name));return name
    def node(op,args,name,**kw):
        name='robust_'+name;nodes.append(onnx.helper.make_node(op,args,[name],**kw));return name
    zero=const('zero',0.);one=const('one',1.);axis=const('axis',[1],np.int64)
    def gather(indices,name):return node('Gather',[inp,const(name+'_indices',indices,np.int64)],name,axis=1)
    def gate(indices,offset,divisor,name):
        a=node('Abs',[gather(indices,name+'_data')],name+'_abs')
        a=node('ReduceMax',[a,axis],name+'_max',keepdims=1)
        a=node('Sub',[a,const(name+'_offset',offset)],name+'_offset_data')
        a=node('Div',[a,const(name+'_scale',divisor)],name+'_ratio')
        return node('Clip',[a,zero,one],name+'_gate')
    def motion(indices,name):
        a=node('Abs',[gather(indices,name+'_command')],name+'_abs');a=node('ReduceSum',[a,axis],name+'_sum',keepdims=1);a=node('Div',[a,const(name+'_scale',.05)],name+'_ratio');return node('Clip',[a,zero,one],name+'_gate')
    move=motion([48,49,50],'move');translate=motion([48,49],'translate');disturbed=node('Max',[gate([3,4],.06,.08,'tilt'),gate([0,1],.5,1.,'gyro')],'disturbed')
    vx=gather([48],'vx');gravity_x=gather([3],'gravity_x');vy=gather([49],'vy');gravity_y=gather([4],'gravity_y')
    positive=node('GreaterOrEqual',[vx,zero],'forward');gain=node('Where',[positive,const('forward_gain',p[0]),const('reverse_gain',p[1])],'gain')
    pitch=node('Add',[one,node('Mul',[gravity_x,const('pitch_command_gain',p[2])],'pitch_command_delta')],'pitch_scale')
    vx=node('Clip',[node('Mul',[vx,node('Mul',[gain,pitch],'pitch_gain')],'scaled_vx'),const('vx_min',-.25),const('vx_max',.3)],'warped_vx')
    lateral=node('Mul',[node('Mul',[gravity_y,const('roll_lateral_gain',p[3])],'lateral_delta'),translate],'active_lateral')
    vy=node('Clip',[node('Add',[vy,lateral],'scaled_vy'),const('vy_min',-.15),const('vy_max',.15)],'warped_vy')
    keep=np.ones(61,np.float32);keep[48:50]=0;kept=node('Mul',[inp,const('keep',keep)],'kept')
    for index,command in [(48,vx),(49,vy)]:
        insert=np.zeros((1,61),np.float32);insert[0,index]=1;kept=node('Add',[kept,node('MatMul',[command,const(f'insert_{index}',insert)],f'command_{index}')],f'kept_{index}')
    warped=kept
    old=list(model.graph.node)
    for n in old:
        for i,name in enumerate(n.input):
            if name==inp:n.input[i]=warped
        for i,name in enumerate(n.output):
            if name==out:n.output[i]='robust_base_action'
    nodes.extend(old)
    correction=node('Mul',[move,const('bias',bias)],'pose_bias');feedback=node('MatMul',[inp,const('feedback_weights',w)],'feedback_raw')
    feedback=node('Mul',[feedback,node('Max',[move,disturbed],'feedback_gate')],'feedback')
    correction=node('Add',[correction,feedback],'pose_and_feedback')
    damping=node('Mul',[gather(list(range(20,34)),'joint_velocity'),const('damping',p[15]*mask)],'damping_raw');damping=node('Mul',[damping,disturbed],'active_damping')
    correction=node('Sub',[correction,damping],'correction');correction=node('Clip',[correction,const('min',-.12),const('max',.12)],'bounded')
    nodes.append(onnx.helper.make_node('Add',['robust_base_action',correction],[out]));del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(tensors)
    prop=model.metadata_props.add();prop.key='robust_adaptation';prop.value='Bounded observable IMU/joint/command adaptation; no disturbance or terrain oracle; simulation only'
    onnx.checker.check_model(model);onnx.save(model,folder/'policy.onnx')
    (folder/'parameters.json').write_text(json.dumps(dict(method='CPU CEM, frozen mounted terrain neural actor',names=NAMES,parameters=p.tolist(),bounds=dict(low=LOW.tolist(),high=HIGH.tolist()),base_policy=str(BASE.relative_to(BASE.parents[4])),base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),correction_bound_rad=.12,simulated_only=True,status='unselected_simulation_candidate'),indent=2)+'\n')
    shutil.copyfile(BASE.with_name('LICENSE'),folder/'LICENSE')
    (folder/'NOTICE').write_text('Derived from Pollen Robotics MicroDuck walking actor revision 088524a64e2557dc453256b6071dbb9d23888802 (Apache 2.0), via the v07 mounted terrain adaptation. Retains all inherited neural/normalizer tensors. Adds command shaping and bounded observable IMU/joint feedback. See REPORT.md for selection and simulation limits. Hardware untested.\n')
