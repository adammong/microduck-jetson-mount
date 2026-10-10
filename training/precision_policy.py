"""Continuous command shaping and bounded observable balance feedback."""
import json
from pathlib import Path
import numpy as np
import onnx
from motion_policy import weights,NAMES
from tune_fast_feedback import matrix

BASE=Path(__file__).resolve().parents[1]/'results/motion-v06/policies/selected/policy.onnx'

class PrecisionPolicy:
    def __init__(self,base,exponents,balance=None,gains=None,turn_feedback=None,recovery=None,balance_forward_only=False):
        self.base=base;self.exponents=np.asarray(exponents,np.float32)
        self.balance=np.zeros(5,np.float32) if balance is None else np.asarray(balance,np.float32)
        self.gains=np.ones(6,np.float32) if gains is None else np.asarray(gains,np.float32)
        self.turn_feedback=np.zeros(2,np.float32) if turn_feedback is None else np.asarray(turn_feedback,np.float32)
        self.recovery=np.zeros(4,np.float32) if recovery is None else np.asarray(recovery,np.float32)
        self.balance_forward_only=balance_forward_only
        assert self.exponents.shape==(6,) and np.all((self.exponents>=.1)&(self.exponents<=1.5))
        assert self.balance.shape==(5,) and np.all(abs(self.balance)<=.15)
        assert self.gains.shape==(6,) and np.all((self.gains>=.4)&(self.gains<=1.2))
        assert self.turn_feedback.shape==(2,) and np.all((self.turn_feedback>=0)&(self.turn_feedback<=6))
        assert self.recovery.shape==(4,) and np.all((self.recovery>=0)&(self.recovery<=8))
        self.feedback=matrix(np.r_[np.zeros(6),self.balance])
    def get_inputs(self):return self.base.get_inputs()
    def run(self,names,inputs):
        obs=next(iter(inputs.values()));raw=weights(obs);w=raw**self.exponents
        warped=obs.copy();warped[:,48:51]=w@np.array([[.2,0,0],[-.15,0,0],[0,.1,0],[0,-.1,0],[0,0,.4],[0,0,-.4]],np.float32)
        mix=raw/np.maximum(raw.sum(axis=1,keepdims=True),1e-6)
        turn_gain=(mix[:,4:6]@self.turn_feedback)*(1-raw[:,4:6].sum(axis=1))
        turn_gain*=np.abs(obs[:,48:50]).sum(axis=1)<=.01
        warped[:,50]=np.clip(warped[:,50]+turn_gain*(obs[:,50]-obs[:,2]),-.4,.4)
        gate=np.clip((np.max(abs(obs[:,3:5]),axis=1)-.06)/.08,0,1)
        gate*=np.sum(abs(obs[:,48:51]),axis=1)<1e-6
        catch=np.column_stack([self.recovery[0]*obs[:,3]+self.recovery[1]*obs[:,1],self.recovery[2]*obs[:,4]-self.recovery[3]*obs[:,0]])
        catch_obs=np.zeros_like(obs);catch_obs[:,48:50]=gate[:,None]*np.clip(catch,[-.15,-.1],[.2,.1])
        catch_weights=weights(catch_obs)**self.exponents
        warped[:,48:50]+=catch_weights@np.array([[.2,0],[-.15,0],[0,.1],[0,-.1],[0,0],[0,0]],np.float32)
        action=self.base.run(names,{self.get_inputs()[0].name:warped})[0]
        idle=obs.copy();idle[:,48:51]=0
        standing=self.base.run(names,{self.get_inputs()[0].name:idle})[0]
        mix=raw/np.maximum(raw.sum(axis=1,keepdims=True),1e-6)
        amplitude=1-np.sum(mix*(1-raw)*(1-self.gains),axis=1,keepdims=True)
        action=standing+amplitude*(action-standing)
        # Raw gyro/gravity only; no simulator velocity/position in the actor.
        balance=obs@self.feedback
        if self.balance_forward_only:balance*=raw[:,0:1]
        return [action+np.clip(balance,-.12,.12)]

def save(exponents,balance,out,gains=None,turn_feedback=None,recovery=None,balance_forward_only=False):
    import copy
    gains=np.ones(6,np.float32) if gains is None else np.asarray(gains,np.float32)
    turn_feedback=np.zeros(2,np.float32) if turn_feedback is None else np.asarray(turn_feedback,np.float32)
    recovery=np.zeros(4,np.float32) if recovery is None else np.asarray(recovery,np.float32)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);model=onnx.load(BASE)
    inp=model.graph.input[0].name;output=model.graph.output[0].name
    initializers=[];nodes=[]
    def const(name,value,dtype=np.float32):
        name='precision_'+name;initializers.append(onnx.numpy_helper.from_array(np.array(value,dtype=dtype),name));return name
    def node(op,args,name,**kw):
        name='precision_'+name;nodes.append(onnx.helper.make_node(op,args,[name],**kw));return name
    cmdmap=np.zeros((61,6),np.float32)
    for i,(slot,scale) in enumerate([(48,5.),(48,-1/.15),(49,10.),(49,-10.),(50,2.5),(50,-2.5)]):cmdmap[slot,i]=scale
    w=node('MatMul',[inp,const('map',cmdmap)],'raw_weights')
    w=node('Clip',[w,const('zero',0.),const('one',1.)],'weights')
    w=node('Pow',[w,const('exponents',exponents)],'shaped_weights')
    insert=np.zeros((6,61),np.float32);insert[:,48:51]=[[.2,0,0],[-.15,0,0],[0,.1,0],[0,-.1,0],[0,0,.4],[0,0,-.4]]
    mapped=node('MatMul',[w,const('insert',insert)],'mapped')
    mask=np.ones(61,np.float32);mask[48:51]=0
    kept=node('Mul',[inp,const('mask',mask)],'kept')
    raw=node('Clip',[node('MatMul',[inp,'precision_map'],'turn_raw_weights'),'precision_zero','precision_one'],'turn_weights')
    total=node('ReduceSum',[raw,const('turn_axis',[1],np.int64)],'turn_total',keepdims=1)
    mix=node('Div',[raw,node('Max',[total,const('turn_epsilon',1e-6)],'turn_denominator')],'turn_mix')
    gain=node('MatMul',[mix,const('turn_gains',np.r_[np.zeros(4),turn_feedback][:,None])],'turn_gain')
    q=node('MatMul',[raw,const('turn_selector',[[0],[0],[0],[0],[1],[1]])],'turn_fraction')
    gain=node('Mul',[gain,node('Sub',['precision_one',q],'turn_remaining')],'turn_scaled_gain')
    xy=node('Abs',[node('Gather',[inp,const('xy_indices',[48,49],np.int64)],'xy',axis=1)],'xy_abs')
    xy=node('ReduceSum',[xy,'precision_turn_axis'],'xy_sum',keepdims=1)
    pure=node('Cast',[node('LessOrEqual',[xy,const('xy_max',.01)],'pure_turn')],'pure_turn_float',to=onnx.TensorProto.FLOAT)
    gain=node('Mul',[gain,pure],'gated_turn_gain')
    selector=np.zeros((61,1),np.float32);selector[50]=1;selector[2]=-1
    error=node('MatMul',[inp,const('yaw_error_selector',selector)],'yaw_error')
    correction=node('Mul',[gain,error],'turn_correction')
    mapped_yaw=node('Gather',[mapped,const('yaw_index',[50],np.int64)],'mapped_yaw',axis=1)
    yaw=node('Clip',[node('Add',[mapped_yaw,correction],'corrected_yaw'),const('yaw_min',-.4),const('yaw_max',.4)],'yaw_bounded')
    no_yaw=np.ones(61,np.float32);no_yaw[50]=0
    mapped=node('Mul',[mapped,const('no_yaw',no_yaw)],'mapped_without_yaw')
    yaw_insert=np.zeros((1,61),np.float32);yaw_insert[0,50]=1
    mapped=node('Add',[mapped,node('MatMul',[yaw,const('yaw_insert',yaw_insert)],'yaw_full')],'mapped_with_yaw')
    gravity=node('Abs',[node('Gather',[inp,const('gravity_xy_indices',[3,4],np.int64)],'gravity_xy',axis=1)],'gravity_xy_abs')
    tilt=node('ReduceMax',[gravity,'precision_turn_axis'],'tilt',keepdims=1)
    gate=node('Clip',[node('Div',[node('Sub',[tilt,const('tilt_start',.06)],'tilt_offset'),const('tilt_range',.08)],'tilt_ratio'),'precision_zero','precision_one'],'tilt_gate')
    command=node('Abs',[node('Gather',[inp,const('command_indices',[48,49,50],np.int64)],'commands',axis=1)],'commands_abs')
    command=node('ReduceSum',[command,'precision_turn_axis'],'command_total',keepdims=1)
    stopped=node('Cast',[node('Less',[command,const('stand_epsilon',1e-6)],'stopped')],'stopped_float',to=onnx.TensorProto.FLOAT)
    gate=node('Mul',[gate,stopped],'catch_gate')
    catch_matrix=np.zeros((61,2),np.float32)
    catch_matrix[3,0]=recovery[0];catch_matrix[1,0]=recovery[1];catch_matrix[4,1]=recovery[2];catch_matrix[0,1]=-recovery[3]
    catch=node('MatMul',[inp,const('catch_matrix',catch_matrix)],'catch_raw')
    catch=node('Max',[catch,const('catch_min',[-.15,-.1])],'catch_lower')
    catch=node('Min',[catch,const('catch_max',[.2,.1])],'catch_bounded')
    catch=node('Mul',[catch,gate],'catch_active')
    catch_insert=np.zeros((2,61),np.float32);catch_insert[:,48:50]=np.eye(2)
    catch=node('MatMul',[catch,const('catch_insert',catch_insert)],'catch_full')
    catch=node('Clip',[node('MatMul',[catch,'precision_map'],'catch_weights_raw'),'precision_zero','precision_one'],'catch_weights')
    catch=node('Pow',[catch,'precision_exponents'],'catch_weights_shaped')
    catch=node('MatMul',[catch,'precision_insert'],'catch_shaped_full')
    mapped=node('Add',[mapped,catch],'mapped_with_catch')
    warped=node('Add',[kept,mapped],'warped')
    old=list(model.graph.node)
    for n in old:
        for i,name in enumerate(n.input):
            if name==inp:n.input[i]=warped
        for i,name in enumerate(n.output):
            if name==output:n.output[i]='precision_base_action'
    nodes.extend(old)
    # Both actors share all original tensors; only zero-command intermediates differ.
    idle_nodes=copy.deepcopy(list(onnx.load(BASE).graph.node))
    intermediates={name for n in idle_nodes for name in n.output}
    for node_index,n in enumerate(idle_nodes):
        n.name=f'precision_idle_{node_index}_{n.name}'
        for i,name in enumerate(n.input):
            if name==inp:n.input[i]=kept
            elif name in intermediates:n.input[i]='precision_idle_'+name
        for i,name in enumerate(n.output):n.output[i]='precision_idle_'+name
    nodes.extend(idle_nodes)
    total=node('ReduceSum',[node('Clip',[node('MatMul',[inp,'precision_map'],'amplitude_weights_raw'),'precision_zero','precision_one'],'amplitude_weights'),const('axis',[1],np.int64)],'weight_total',keepdims=1)
    mix=node('Div',['precision_amplitude_weights',node('Max',[total,const('epsilon',1e-6)],'denominator')],'mix')
    remaining=node('Sub',['precision_one','precision_amplitude_weights'],'remaining')
    decay=node('Mul',[mix,remaining],'decay_weights')
    decay=node('MatMul',[decay,const('gain_decay',(1-gains)[:,None])],'decay')
    amplitude=node('Sub',['precision_one',decay],'amplitude')
    difference=node('Sub',['precision_base_action','precision_idle_'+output],'action_difference')
    scaled=node('Mul',[difference,amplitude],'scaled_action')
    blended=node('Add',['precision_idle_'+output,scaled],'blended_action')
    feedback=matrix(np.r_[np.zeros(6),balance])
    raw=node('MatMul',[inp,const('feedback',feedback)],'feedback_raw')
    if balance_forward_only:
        gate=node('Gather',['precision_amplitude_weights',const('forward_index',[0],np.int64)],'forward_gate',axis=1)
        raw=node('Mul',[raw,gate],'forward_feedback')
    correction=node('Clip',[raw,const('minimum',-.12),const('maximum',.12)],'bounded')
    nodes.append(onnx.helper.make_node('Add',[blended,correction],[output]))
    del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(initializers)
    prop=model.metadata_props.add();prop.key='command_limiter';prop.value='precision_v1'
    onnx.checker.check_model(model);onnx.save(model,out/'policy.onnx')
    (out/'parameters.json').write_text(json.dumps(dict(profiles=NAMES,exponents=list(map(float,exponents)),gains=list(map(float,gains)),turn_feedback=list(map(float,turn_feedback)),balance=list(map(float,balance)),recovery=list(map(float,recovery)),balance_forward_only=balance_forward_only,base_policy=str(BASE.relative_to(BASE.parents[4])),balance_correction_limit_rad=.12,description='Continuous command power map and gait amplitude relative to the standing actor, observed yaw-rate feedback at low pure-turn commands; bounded gyro/gravity balance feedback and tilt-triggered standing catch steps. Simulation only.'),indent=2)+'\n')
