"""Import the pinned ONNX actor exactly; fit only a critic on mounted rollouts.

No approximate actor cloning or changed command normalization. The actor's
numerical equivalence is checked on raw states before any PPO fine tuning.
"""
from pathlib import Path
import argparse,json,os,time
import numpy as np,onnx,onnxruntime as ort,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from microduck_local.symmetry import FastActorCriticPolicy
from microduck_local.export_onnx import export
from microduck_local.robots import registry
from strict_env import StrictWalkEnv
from evaluate import scene
from paths import ROOT,revision

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episodes',type=int,default=120)
    ap.add_argument('--critic-epochs',type=int,default=30);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--name',default='teacher-import');a=ap.parse_args();torch.set_num_threads(4)
    torch.manual_seed(a.seed);out=ROOT/'tmp/mac-runs'/f'{revision()}-{a.name}-s{a.seed}'
    if out.exists():raise SystemExit(f'Run already exists: {out}')
    teacher=ROOT/'tmp/sim-inputs/alpha_walking.onnx';graph=onnx.load(teacher)
    assert [n.op_type for n in graph.graph.node]==['Sub','Div','Gemm','Elu','Gemm','Elu','Gemm','Elu','Gemm']
    init={p.name:onnx.numpy_helper.to_array(p).copy() for p in graph.graph.initializer}
    mean=init['obs_normalizer._mean'].reshape(61).astype(np.float64)
    std=init['onnx::Div_24'].reshape(61).astype(np.float64)
    assert (std>0).all()
    model_path=scene('loaded');compiled=mujoco_model(model_path)
    env=StrictWalkEnv(model=compiled,scene_xml=str(model_path),seed=a.seed,actuator_force='bam',
        obs_noise=False,domain_rand=False,action_delay=True,random_yaw=True,max_episode_s=10)
    venv=VecNormalize(DummyVecEnv([lambda:env]),norm_obs=True,norm_reward=False,clip_obs=1e6)
    venv.obs_rms.mean=mean;venv.obs_rms.var=np.maximum(std**2-venv.epsilon,0);venv.obs_rms.count=1e6;venv.training=False
    model=PPO(FastActorCriticPolicy,venv,policy_kwargs=dict(net_arch=dict(pi=[512,256,128],vf=[512,256,128]),activation_fn=torch.nn.ELU,log_std_init=-4),
        n_steps=256,batch_size=256,n_epochs=5,learning_rate=1e-5,ent_coef=0.,seed=a.seed,device='cpu',verbose=0)
    layers=list(model.policy.mlp_extractor.policy_net)
    with torch.no_grad():
        for i in (0,2,4):
            layers[i].weight.copy_(torch.from_numpy(init[f'mlp.{i}.weight']));layers[i].bias.copy_(torch.from_numpy(init[f'mlp.{i}.bias']))
        model.policy.action_net.weight.copy_(torch.from_numpy(init['mlp.6.weight']));model.policy.action_net.bias.copy_(torch.from_numpy(init['mlp.6.bias']))
    sess=ort.InferenceSession(str(teacher),providers=['CPUExecutionProvider'])
    obs_buf=[];act_buf=[];returns=[];survived=0;start=time.time()
    for ep in range(a.episodes):
        obs,_=env.reset(seed=a.seed+ep);rewards=[]
        for k in range(500):
            act=sess.run(None,{'obs':obs[None]})[0][0];obs_buf.append(obs.copy());act_buf.append(act.copy())
            obs,r,done,trunc,info=env.step(act);rewards.append(r)
            if done or trunc:break
        survived+=not done
        ret=0.;ep_returns=[]
        for r in reversed(rewards):ret=float(r)+.99*ret;ep_returns.append(ret)
        returns.extend(reversed(ep_returns))
        if ep%20==0:print('Collected',ep+1,'episodes',len(obs_buf),'states',flush=True)
    raw=np.array(obs_buf,np.float32);target=np.array(act_buf,np.float32)
    normalized=venv.normalize_obs(raw);x=torch.from_numpy(normalized)
    with torch.no_grad():pred=model.policy.action_net(model.policy.mlp_extractor.forward_actor(x)).numpy()
    actor_before=pred.copy()
    error=float(np.max(np.abs(pred-target)));assert error<2e-5,error
    # Separate critic parameters: the exact imported actor never enters this optimizer.
    params=list(model.policy.mlp_extractor.value_net.parameters())+list(model.policy.value_net.parameters())
    optim=torch.optim.Adam(params,lr=1e-3);y=torch.tensor(returns,dtype=torch.float32)
    for epoch in range(a.critic_epochs):
        idx=torch.randperm(len(x));losses=[]
        for ids in idx.split(2048):
            pred=model.policy.value_net(model.policy.mlp_extractor.forward_critic(x[ids])).squeeze(-1)
            loss=torch.nn.functional.mse_loss(pred,y[ids]);optim.zero_grad();loss.backward();optim.step();losses.append(float(loss.detach()))
        if epoch%10==0:print('Critic epoch',epoch,'MSE',sum(losses)/len(losses),flush=True)
    with torch.no_grad():after=model.policy.action_net(model.policy.mlp_extractor.forward_actor(x)).numpy()
    assert np.array_equal(actor_before,after), 'Critic fitting changed the actor'
    assert float(np.max(np.abs(after-target)))==error
    out.mkdir(parents=True);model.save(out/'model');venv.save(out/'vecnormalize.pkl')
    record=dict(run_name=out.name,robot='microduck',task='walk',contract=registry.get('microduck').contract().as_dict(),
        pinned_command=False,teacher=str(teacher.relative_to(ROOT)),exact_actor_import=True,keep_command_normalizer=True,mount_revision=revision(),
        episodes=a.episodes,states=len(raw),critic_epochs=a.critic_epochs,max_action_error_rad=error,teacher_survived=survived,
        elapsed_s=time.time()-start,env_kwargs=dict(actuator_force='bam',domain_rand=False,obs_noise=False,action_delay=True))
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n');export(out,out/'policy.onnx')
    print('Saved',out,'max action error',error,flush=True)

def mujoco_model(path):
    import mujoco
    return mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb')))
if __name__=='__main__':main()
