"""Verify the exported feedback policy, its bounds, and start/stop rollouts."""
import json,hashlib
from pathlib import Path
import numpy as np,mujoco,onnxruntime as ort
from microduck_local.robots import registry
from microduck_local.robots.policy_contract import PolicyContract
from fast_env import FastRunEnv
from tune_fast_feedback import FeedbackPolicy,matrix,BASIS,DYNAMIC
from evaluate import scene
from paths import ROOT

def main():
 out=ROOT/'results/speed-v06/policies/selected';source=ROOT/'results/speed-v06/feedback-study'
 search=json.loads((source/'search.json').read_text());params=search['best']['params']
 assert search['best']['feasible'] and max(abs(x) for x in params)<=.06
 contract=registry.get('microduck').contract();contract.write_onnx_metadata(out/'policy.onnx')
 import onnx
 model=onnx.load(out/'policy.onnx')
 for k,v in [('mount_revision','v06'),('deployment_status','simulation_only'),('fast_command_m_s','0.5')]:
  p=model.metadata_props.add();p.key=k;p.value=v
 onnx.save(model,out/'policy.onnx');onnx.checker.check_model(model)
 opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
 teacher=ort.InferenceSession(str(ROOT/'tmp/sim-inputs/alpha_walking.onnx'),opts,providers=['CPUExecutionProvider'])
 policy=ort.InferenceSession(str(out/'policy.onnx'),opts,providers=['CPUExecutionProvider']);wrapped=FeedbackPolicy(teacher,params)
 rng=np.random.default_rng(816);error=0.;max_correction=0.;idle_error=0.
 for i in range(1000):
  obs=rng.normal(0,.4,(1,61)).astype(np.float32);obs[0,48]=[0,.3,.5,-.2][i%4]
  data={teacher.get_inputs()[0].name:obs}
  actual=policy.run(None,data)[0];expected=wrapped.run(None,data)[0];base=teacher.run(None,data)[0]
  error=max(error,float(np.max(abs(actual-expected))));max_correction=max(max_correction,float(np.max(abs(actual-base))))
  if obs[0,48]<=.35:idle_error=max(idle_error,float(np.max(abs(actual-base))))
 assert error<1e-6 and max_correction<=.080001 and idle_error==0.
 assert PolicyContract.from_onnx(out/'policy.onnx').id==contract.id
 schedule=[('stand',0.,2.),('fast',.5,8.),('stop',0.,3.),('normal',.3,5.),('fast_again',.5,8.),('stop_again',0.,4.)]
 rows=[];p=scene('loaded',rev='v06')
 for seed in range(60,64):
  env=FastRunEnv(model=mujoco.MjModel.from_binary_path(str(p.with_suffix('.mjb'))),scene_xml=str(p),seed=seed,
      actuator_force='bam',domain_rand=False,obs_noise=False,action_delay=True,random_yaw=False,command_resample_s=1000,max_episode_s=40.)
  assert env.bam.max_current==1.75 and not env.spotter
  obs,_=env.reset(seed=seed);start=env.data.qpos[:3].copy();phases=[];failed=False
  for name,command,seconds in schedule:
   env.twist_cmd[:]=[command,0,0];env.head_cmd[:]=0;env.body_cmd[:]=0;obs=env._get_obs();speeds=[]
   for _ in range(round(seconds/.02)):
    action=policy.run(None,{policy.get_inputs()[0].name:obs[None].astype(np.float32)})[0].reshape(14)
    obs,_,done,_,info=env.step(action);speeds.append(float(env.body_lin_vel()[0]))
    if done:failed=True;break
   phases.append(dict(phase=name,command_m_s=command,seconds=len(speeds)*.02,mean_body_forward_m_s=float(np.mean(speeds)),last_second_mean_body_forward_m_s=float(np.mean(speeds[-50:]))))
   if failed:break
  rows.append(dict(seed=seed,survived=not failed,seconds=env.step_count*.02,phases=phases,
      failure_reason=info.get('failure_reason','tilt' if failed else None),mass_kg=float(env.model.body_mass.sum())))
  env.close()
 record=dict(passed_export=True,max_export_action_error_rad=error,max_test_correction_rad=max_correction,
     unchanged_normal_and_idle_max_error_rad=idle_error,contract=contract.as_dict(),
     transitions=rows,transition_survived=sum(r['survived'] for r in rows),transition_trials=len(rows))
 (ROOT/'results/speed-v06/feedback-verification.json').write_text(json.dumps(record,indent=2)+'\n')
 parameters=dict(params=params,static_basis=BASIS.tolist(),dynamic_basis=DYNAMIC.tolist(),raw_obs_linear_weights=matrix(params).tolist(),
     correction_clip_rad=[-.08,.08],command_gate='clip((obs[48]-0.35)/0.10,0,1)')
 (out/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
 record=dict(run_name='v06-fast-feedback',robot='microduck',task='walk',contract=contract.as_dict(),mount_revision='v06',
     status='selected for simulation at command [0.5,0,0]; hardware unverified',method=search['method'],
     training_seeds=search['training_seeds'],training_episode_seconds=search['seconds'],search_seed=search['seed'],
     teacher_sha256=hashlib.sha256((ROOT/'tmp/sim-inputs/alpha_walking.onnx').read_bytes()).hexdigest(),
     policy_sha256=hashlib.sha256((out/'policy.onnx').read_bytes()).hexdigest(),
     parent_search='results/speed-v06/feedback-study/search.json',
     env_kwargs=dict(actuator='bam',domain_rand=False,obs_noise=False,action_delay=True),
     physics_hz=200,policy_hz=50,model_mass_kg=1.179337261211853,servo_current_limit_a=1.75,
     no_assistance=True,unmodified_teacher_at_forward_commands_le_0_35=True,
     feedback_parameter_file='parameters.json')
 (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
 print('Export max error',error,'correction bound',max_correction,'transitions',sum(r['survived'] for r in rows),'/',len(rows),flush=True)
if __name__=='__main__':main()
