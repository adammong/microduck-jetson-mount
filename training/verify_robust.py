"""Assert disturbance physics, identity control and exported actor equivalence."""
import argparse,json,hashlib,tempfile
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort,mujoco
from evaluate_motion import scene,session,trial
from robust_env import RobustTerrainEnv
from robust_policy import BASE,INITIAL,LOW,HIGH,RobustPolicy,export
from evaluate_robust import cases,measure

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--policy',type=Path);a=ap.parse_args();base=session(BASE);name=base.get_inputs()[0].name
    rng=np.random.default_rng(8101);identity=RobustPolicy(base,INITIAL);maximum=0.
    for _ in range(100):
        obs=rng.normal(0,.3,(1,61)).astype(np.float32);obs[:,48:51]=rng.uniform([-.15,-.1,-.4],[.2,.1,.4])
        assert np.array_equal(identity.run(None,{name:obs})[0],base.run(None,{name:obs})[0])
    parameters=json.loads(a.policy.with_name('parameters.json').read_text())['parameters'] if a.policy else rng.uniform(LOW,HIGH).tolist()
    with tempfile.TemporaryDirectory(dir='tmp') as tmp:
        export(parameters,tmp);wrapped=RobustPolicy(base,parameters);deployed=session(Path(tmp)/'policy.onnx')
        checked=onnx.load(Path(tmp)/'policy.onnx')
        checked.graph.output.append(onnx.helper.make_tensor_value_info('robust_bounded',onnx.TensorProto.FLOAT,[None,14]))
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        bounded=ort.InferenceSession(checked.SerializeToString(),options,providers=['CPUExecutionProvider'])
        for i in range(1000):
            obs=rng.normal(0,.3,(1,61)).astype(np.float32);obs[:,48:51]=0 if i%4==0 else rng.uniform([-.15,-.1,-.4],[.2,.1,.4]);before=obs.copy()
            x=wrapped.run(None,{name:obs})[0];y=deployed.run(None,{name:obs})[0];maximum=max(maximum,float(np.max(abs(x-y))));assert np.array_equal(before,obs) and np.isfinite(y).all()
            assert np.array_equal(y[:,5:7],base.run(None,{name:obs})[0][:,5:7])
            assert np.max(abs(bounded.run(['robust_bounded'],{name:obs})[0]))<=.120001
        assert maximum<2e-6,maximum
        tensors={t.name:onnx.numpy_helper.to_array(t) for t in onnx.load(Path(tmp)/'policy.onnx').graph.initializer}
        for t in onnx.load(BASE).graph.initializer:assert np.array_equal(tensors[t.name],onnx.numpy_helper.to_array(t))
        assert deployed.get_modelmeta().custom_metadata_map['command_limiter']==base.get_modelmeta().custom_metadata_map['command_limiter']
        if a.policy:assert a.policy.read_bytes()==(Path(tmp)/'policy.onnx').read_bytes()
    path=scene('loaded',rev='v07');events=[dict(kind='push',start=.2,duration=.08,force_body=[2,0,0]),dict(kind='slip',start=.5,duration=.1,friction=.1)]
    e=RobustTerrainEnv(events=events,terrain='flat',terrain_seed=1,model=mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb'))),scene_xml=str(path),seed=3,actuator_force='bam',domain_rand=False,obs_noise=False,action_delay=True,random_yaw=False,max_episode_s=3,command_resample_s=1000)
    e.reset(seed=3);e.twist_cmd[:]=0;e.head_cmd[:]=0;e.body_cmd[:]=0;old_mu=e.model.geom_friction.copy();effective=[]
    for k in range(40):
        obs=e._get_obs();e.step(base.run(None,{name:obs[None].astype(np.float32)})[0].reshape(14))
        assert not np.any(e.data.xfrc_applied), 'No residual recovery/perturbation force between steps'
        if 25<=k<30:
            assert all(e.model.geom_friction[g,0]==.1 for g in e._feet|{e._floor_id})
            effective.extend(float(c.friction[0]) for c in e.data.contact if e._floor_id in (int(c.geom1),int(c.geom2)))
        if k>=31:assert np.array_equal(e.model.geom_friction,old_mu)
    assert effective and max(effective)<=.100001,effective
    assert np.isclose(e.disturbance_records[0]['applied_seconds'],.08)
    assert np.isclose(np.linalg.norm(e.disturbance_records[0]['impulse_world_ns']),.16,atol=.002)
    assert e.bam.max_current==1.75 and not e.spotter and not e.push_robot;e.close()
    legacy=trial(base,[.2,0,0],3,3,revision='v07',terrain='flat')
    no_event=trial(base,[.2,0,0],3,3,revision='v07',terrain='flat',events=[dict(kind='push',start=100,duration=.1,force_body=[2,0,0])])
    for key in legacy:assert legacy[key]==no_event[key],key
    snag=measure(base,cases('trip_slip',True)[0],20,1020)
    assert snag['disturbances'][0]['applied_seconds']==.08 and snag['disturbances'][0]['trigger_s'] is not None
    result=dict(passed=True,identity_matches_base_exactly=True,inherited_tensors_unchanged=True,head_pitch_unchanged=True,export_samples=1000,max_export_error_rad=maximum,correction_bound_rad=.12,exported_correction_bound_checked=True,slip_contact_friction_checked=True,friction_restoration_checked=True,actual_force_impulse_checked=True,no_residual_external_forces=True,no_event_control_matches_original_exactly=True,swing_foot_snag_trigger_checked=True,actuator_current_a=1.75,base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest())
    if a.policy:result['policy_sha256']=hashlib.sha256(a.policy.read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print('Robust physics and actor self-checks passed',maximum)
if __name__=='__main__':main()
