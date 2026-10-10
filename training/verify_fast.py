"""Integration checks for mounted speed training and exact teacher lineage."""
import sys,json,pickle
from pathlib import Path
import numpy as np,mujoco,onnxruntime as ort
from fast_env import FastRunEnv,install_recipe
from evaluate import scene
from paths import ROOT

def main():
 p=scene('loaded',rev='v06');model=mujoco.MjModel.from_binary_path(str(p.with_suffix('.mjb')))
 env=FastRunEnv(model=model,scene_xml=str(p),seed=9,actuator_force='bam',domain_rand=False,obs_noise=False,action_delay=True,
     spawn_overrides={'MICRODUCK_FAST_SPEED_LO':'.42','MICRODUCK_FAST_SPEED_HI':'.43'})
 assert abs(model.body_mass.sum()-1.179337261211853)<1e-8
 assert env.speed_lo==.42 and env.speed_hi==.43 and env.max_steps==750
 assert env.action_space.shape==(14,) and env.observation_space.shape==(61,)
 assert env.bam is not None and not env.spotter and env.action_delay
 assert not env.behavior.symmetric and env.push_robot is False
 assert dict((t.key,t.weight) for t in env.behavior.terms)['keep_pace']==8.
 commands=[]
 for i in range(100):
  obs,_=env.reset(seed=i)
  assert obs.shape==(61,)
  commands.append(float(env.twist_cmd[0]))
  assert env.twist_cmd[1]==env.twist_cmd[2]==0
  assert env.twist_cmd[0]==0 or .42<=env.twist_cmd[0]<=.43
 assert any(v==0 for v in commands) and any(v>.42 for v in commands)
 teacher=ROOT/'tmp/speed-runs/teacher-fast-critic'
 original=ROOT/'results/mac-v06/policies/teacher-import'
 with (teacher/'vecnormalize.pkl').open('rb') as f:a=pickle.load(f)
 with (original/'vecnormalize.pkl').open('rb') as f:b=pickle.load(f)
 assert np.array_equal(a.obs_rms.mean,b.obs_rms.mean) and np.array_equal(a.obs_rms.var,b.obs_rms.var)
 sess1=ort.InferenceSession(str(teacher/'policy.onnx'),providers=['CPUExecutionProvider'])
 sess2=ort.InferenceSession(str(original/'policy.onnx'),providers=['CPUExecutionProvider'])
 rng=np.random.default_rng(13);states=(b.obs_rms.mean+rng.normal(size=(1000,61))*np.sqrt(b.obs_rms.var)).astype(np.float32)
 x=np.concatenate([sess1.run(None,{sess1.get_inputs()[0].name:s[None]})[0] for s in states])
 y=np.concatenate([sess2.run(None,{sess2.get_inputs()[0].name:s[None]})[0] for s in states])
 assert np.max(abs(x-y))<2e-5
 # Null control must collapse or contact the floor under this real plant.
 obs,_=env.reset(seed=0);failed=False
 for i in range(500):
  _,_,done,_,info=env.step(np.zeros(14,np.float32))
  assert all(v<=1e-6 for k,v in info.get('episode_rewards',{}).items() if k.endswith('_penalty'))
  if done:failed=True;break
 assert failed,'Null control was artificially supported'
 result=dict(payload_model_mass_kg=float(model.body_mass.sum()),contract=[61,14],action_delay=True,asymmetric=True,
     preview_command_knob_override_verified=True,episode_s=env.max_steps*.02,critic_fit_actor_max_error=float(np.max(abs(x-y))),
     normalizer_identical=True,null_control_failed=True,null_failure_reason=info.get('failure_reason','tilt'),recipe_version=2)
 (ROOT/'results/speed-v06/verification.json').write_text(json.dumps(result,indent=2)+'\n')
 env.close();print(json.dumps(result,indent=2))
if __name__=='__main__':main()
