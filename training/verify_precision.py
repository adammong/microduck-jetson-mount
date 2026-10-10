"""Verify executable ONNX agreement, immutable base tensors and limits."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import onnx
import mujoco
from precision_policy import PrecisionPolicy,BASE,save
from evaluate_motion import session
from motion_controller import limiter_for_policy
from fast_env import FastRunEnv
from evaluate import scene
from evaluate_motion import trial

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    p=json.loads((a.folder/'parameters.json').read_text());base=session(BASE);actor=session(a.folder/'policy.onnx')
    wrapped=PrecisionPolicy(base,p['exponents'],p['balance'],p.get('gains'),p.get('turn_feedback'),p.get('recovery'),p.get('balance_forward_only',False))
    original={i.name:onnx.numpy_helper.to_array(i) for i in onnx.load(BASE).graph.initializer}
    actual={i.name:onnx.numpy_helper.to_array(i) for i in onnx.load(a.folder/'policy.onnx').graph.initializer}
    assert all(np.array_equal(v,actual[k]) for k,v in original.items())
    assert actor.get_inputs()[0].shape==[1,61] and actor.get_outputs()[0].shape==[1,14]
    rng=np.random.default_rng(1918);error=correction=0.
    for i in range(1000):
        obs=rng.normal(0,.3,(1,61)).astype(np.float32);obs[0,48:51]=rng.uniform([-.15,-.1,-.4],[.2,.1,.4])
        if i%7==0:obs[0,48:51]=0
        if i%3==1:obs[0,48:50]=0
        data={actor.get_inputs()[0].name:obs};out=actor.run(None,data)[0];expect=wrapped.run(None,data)[0]
        assert np.isfinite(out).all();error=max(error,float(abs(out-expect).max()))
        correction=max(correction,float(abs(np.clip(obs@wrapped.feedback,-.12,.12)).max()))
    assert error<4e-6 and correction<=.120001
    assert np.all(np.asarray(p['exponents'])>0) and np.all(np.asarray(p['gains'])>=.4)
    limiter=limiter_for_policy(actor);limiter.set([.1,.05,.2],0)
    assert np.array_equal(limiter.acceleration,np.array([.25,.2,1.],np.float32))
    for k in range(100):c=limiter.step(k*.02)
    assert np.max(abs(c))<1e-7
    path=scene('loaded',rev='v06');env=FastRunEnv(model=mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb'))),scene_xml=str(path),actuator_force='bam',action_delay=True,obs_noise=False,domain_rand=False)
    mass=float(env.model.body_mass.sum());current=float(env.bam.max_current)
    assert abs(mass-1.179337261211853)<1e-8 and current==1.75 and not env.spotter and not env.push_robot
    env.close()
    class Limp:
        def get_inputs(self):return base.get_inputs()
        def run(self,names,inputs):return [next(iter(inputs.values()))[:,6:20].copy()]
    nulls=[trial(Limp(),[0,0,0],seed,10) for seed in [390,391]]
    assert all(not r['survived'] for r in nulls)
    result=dict(passed=True,max_export_error_rad=error,max_balance_correction_rad=correction,base_tensors_unchanged=True,observation_dim=61,action_dim=14,watchdog_passed=True,command_acceleration=limiter.acceleration.tolist(),nominal_mass_kg=mass,servo_current_limit_a=current,no_assistance=True,null_controls=nulls,base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),policy_sha256=hashlib.sha256((a.folder/'policy.onnx').read_bytes()).hexdigest())
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
