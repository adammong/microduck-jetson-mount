"""Verify the normalizer/actor, exported map, physics and command deadman."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import mujoco,onnx
from microduck_local.robots import registry
from microduck_local.robots.policy_contract import PolicyContract
from motion_policy import MotionPolicy,weights,LIMIT
from motion_controller import CommandLimiter
from evaluate_motion import session,trial
from evaluate import scene
from fast_env import FastRunEnv
from paths import ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,default=ROOT/'results/motion-v06/policies/selected')
    ap.add_argument('--out',type=Path,default=ROOT/'results/motion-v06/verification.json');args=ap.parse_args()
    path=args.folder/'policy.onnx';contract=registry.get('microduck').contract();contract.write_onnx_metadata(path)
    teacher=session(ROOT/'tmp/sim-inputs/alpha_walking.onnx');policy=session(path)
    record=json.loads((args.folder/'parameters.json').read_text());p=np.array(record['parameters'],np.float32)
    boost=float(record.get('turn_activation_boost',0.))
    from motion_activation import ActivationPolicy
    wrapped=ActivationPolicy(teacher,p,boost) if boost else MotionPolicy(teacher,p)
    original={i.name:onnx.numpy_helper.to_array(i) for i in onnx.load(ROOT/'tmp/sim-inputs/alpha_walking.onnx').graph.initializer}
    exported={i.name:onnx.numpy_helper.to_array(i) for i in onnx.load(path).graph.initializer}
    assert all(np.array_equal(value,exported[name]) for name,value in original.items())
    assert PolicyContract.from_onnx(path)==contract and policy.get_inputs()[0].shape==[1,61] and policy.get_outputs()[0].shape==[1,14]
    rng=np.random.default_rng(909);error=idle=bound=0.;activation_hits=0
    for i in range(1000):
        obs=rng.normal(0,.3,(1,61)).astype(np.float32);obs[0,48:51]=rng.uniform([-.15,-.1,-.4],[.2,.1,.4])
        if i%4==2:obs[0,48:50]=0;obs[0,50]=.4 if (i//4)%2 else -.4;obs[0,2]=rng.uniform(-.15,.15)
        if i%10==0:obs[0,48:51]=0
        activation_hits+=bool(boost and abs(obs[0,2])<.2 and abs(obs[0,50])>0 and np.sum(np.abs(obs[0,48:50]))<=.01)
        data={'obs':obs};actual=policy.run(None,data)[0];expected=wrapped.run(None,data)[0]
        error=max(error,float(np.max(abs(actual-expected))))
        w=weights(obs);mix=w/np.maximum(w.sum(axis=1,keepdims=True),1)
        proxy=w@p[:,:3];proxy[:,2]+=(mix@p[:,3])*(obs[:,50]-obs[:,2])
        proxy[:,2]+=boost*np.clip(1-np.abs(obs[:,2])/.2,0,1)*np.clip(obs[:,50]/.4,-1,1)*(np.sum(np.abs(obs[:,48:50]),axis=1)<=.01)
        warped=obs.copy();warped[:,48:51]=np.clip(proxy,[-.65,-.7,-2.],[.6,.7,2.])
        bound=max(bound,float(np.max(abs(actual-teacher.run(None,{'obs':warped})[0]))))
        if i%10==0:idle=max(idle,float(np.max(abs(actual-teacher.run(None,data)[0]))))
    assert error<3e-6 and idle==0 and bound<=LIMIT+3e-6
    assert not boost or activation_hits>=100
    limiter=CommandLimiter();limiter.set([100,-100,100],0)
    assert np.array_equal(limiter.target,np.array([.2,-.1,.4],np.float32))
    for k in range(30):limited=limiter.step(k*.02)
    assert np.all(limited>=limiter.lower) and np.all(limited<=limiter.upper)
    for k in range(31,100):limited=limiter.step(k*.02)
    assert np.max(abs(limited))<1e-7
    limiter.set([.2,.1,.4],2);limiter.step(2);limiter.stop();assert np.array_equal(limiter.step(2),np.zeros(3))
    for c in [[float('nan'),0,0],[0,0],[float('inf'),0,0]]:
        try:limiter.set(c,2)
        except ValueError:pass
        else:raise AssertionError('Invalid command accepted')
    pathscene=scene('loaded',rev='v06');env=FastRunEnv(model=mujoco.MjModel.from_binary_path(str(pathscene.with_suffix('.mjb'))),scene_xml=str(pathscene),actuator_force='bam',action_delay=True,obs_noise=False,domain_rand=False,push_robot=False)
    mass=float(env.model.body_mass.sum());current=float(env.bam.max_current)
    assert abs(mass-1.179337261211853)<1e-8 and current==1.75 and not env.spotter and env.action_delay
    env.close()
    class Limp:
        def get_inputs(self):return teacher.get_inputs()
        def run(self,names,inputs):return [next(iter(inputs.values()))[:,6:20].copy()]
    nulls=[trial(Limp(),[0,0,0],seed,10) for seed in [90,91]]
    assert all(not r['survived'] for r in nulls)
    record=dict(passed=True,max_export_error_rad=error,max_correction_rad=bound,stop_teacher_error_rad=idle,teacher_initializers_unchanged=True,contract=contract.as_dict(),model_mass_kg=mass,servo_current_limit_a=current,action_delay=True,no_assistance=True,command_watchdog_s=.6,activation_hits=activation_hits,limiter_checks_passed=True,null_controls=nulls,policy_sha256=hashlib.sha256((args.folder/'policy.onnx').read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(record,indent=2)+'\n')
    print('Verified export',error,'bound',bound,'stop',idle,'deadman and physics',flush=True)
if __name__=='__main__':main()
