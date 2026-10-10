from pathlib import Path
import sys,json,shutil
root=Path.cwd();sys.path.insert(0,str(root/'training'))
import numpy as np,torch,mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from fast_env import FastRunEnv
from evaluate import scene
from microduck_local.export_onnx import export
from microduck_local import contract as C
import onnxruntime as ort

def main():
 torch.set_num_threads(4);torch.manual_seed(41)
 source=root/'results/mac-v06/policies/teacher-import';out=root/'tmp/speed-runs/teacher-fast-critic'
 if out.exists():raise SystemExit('Refusing to replace existing donor')
 p=scene('loaded',rev='v06')
 env=FastRunEnv(model=mujoco.MjModel.from_binary_path(str(p.with_suffix('.mjb'))),scene_xml=str(p),seed=41,actuator_force='bam',obs_noise=False,domain_rand=False,action_delay=True,random_yaw=True,max_episode_s=15.)
 vn=VecNormalize.load(str(source/'vecnormalize.pkl'),DummyVecEnv([lambda:env]));vn.training=False
 model=PPO.load(str(source/'model.sb3'),env=vn,device='cpu')
 teacher=ort.InferenceSession(str(source/'policy.onnx'),providers=['CPUExecutionProvider'])
 states=[];returns=[]
 for ep in range(80):
  obs,_=env.reset(seed=100+ep);rewards=[]
  for step in range(750):
   action=teacher.run(None,{'obs':obs[None].astype(np.float32)})[0].reshape(14)
   states.append(obs.copy());obs,r,done,truncated,_=env.step(action);rewards.append(r)
   if done or truncated:break
  ret=0;epreturns=[]
  for reward in reversed(rewards):ret=reward+.99*ret;epreturns.append(ret)
  returns.extend(reversed(epreturns))
  if ep%20==0:print('Collected',ep+1,'episodes',len(states),'states',flush=True)
 x=torch.from_numpy(vn.normalize_obs(np.array(states,np.float32)));y=torch.tensor(returns,dtype=torch.float32)
 with torch.no_grad():before=model.policy.action_net(model.policy.mlp_extractor.forward_actor(x)).clone()
 params=list(model.policy.mlp_extractor.value_net.parameters())+list(model.policy.value_net.parameters())
 optim=torch.optim.Adam(params,lr=1e-3)
 for epoch in range(20):
  ids=torch.randperm(len(x));losses=[]
  for batch in ids.split(2048):
   value=model.policy.value_net(model.policy.mlp_extractor.forward_critic(x[batch])).squeeze(-1)
   loss=torch.nn.functional.mse_loss(value,y[batch]);optim.zero_grad();loss.backward();optim.step();losses.append(float(loss.detach()))
  if epoch%5==0:print('Critic epoch',epoch,'MSE',np.mean(losses),flush=True)
 with torch.no_grad():after=model.policy.action_net(model.policy.mlp_extractor.forward_actor(x))
 assert torch.equal(before,after)
 # Mean actor unchanged. Explore with ~0.03 policy-output-unit SD.
 with torch.no_grad():model.policy.log_std.fill_(-3.5)
 out.mkdir(parents=True);model.save(out/'model.zip');vn.save(out/'vecnormalize.pkl')
 record=json.loads((source/'run.json').read_text());record.update(run_name=out.name,critic_recipe='mounted_fast_walk',critic_states=len(states),critic_epochs=20,log_std=-3.5,actor_mean_unchanged=True,critic_seed=41)
 (out/'run.json').write_text(json.dumps(record,indent=2)+'\n');export(out,out/'policy.onnx')
 print('Saved',out,flush=True)
if __name__=='__main__':main()
