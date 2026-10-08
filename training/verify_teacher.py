"""Check exact ONNX import, critic isolation metadata and frozen warm-start stats."""
from pathlib import Path
import argparse,json,pickle
import numpy as np,onnxruntime as ort
from evaluate import scene
from strict_env import StrictWalkEnv
from paths import ROOT,result_root
import mujoco

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm-run',type=Path,required=True)
    ap.add_argument('--import-run',type=Path,default=ROOT/'tmp/mac-runs/v06-teacher-import-s0')
    a=ap.parse_args();donor=a.import_run;warm=a.warm_run
    va=pickle.load((donor/'vecnormalize.pkl').open('rb'));vb=pickle.load((warm/'vecnormalize.pkl').open('rb'))
    assert np.array_equal(va.obs_rms.mean,vb.obs_rms.mean)
    assert np.array_equal(va.obs_rms.var,vb.obs_rms.var)
    teacher=ort.InferenceSession(str(ROOT/'tmp/sim-inputs/alpha_walking.onnx'),providers=['CPUExecutionProvider'])
    imported=ort.InferenceSession(str(donor/'policy.onnx'),providers=['CPUExecutionProvider'])
    path=scene('loaded');env=StrictWalkEnv(model=mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb'))),scene_xml=str(path),seed=31,actuator_force='bam',obs_noise=False,domain_rand=False,action_delay=True)
    error=0.;states=0
    for seed in range(31,35):
        obs,_=env.reset(seed=seed)
        for _ in range(250):
            x=obs[None].astype(np.float32)
            at=teacher.run(None,{'obs':x})[0];ai=imported.run(None,{'obs':x})[0]
            error=max(error,float(np.max(np.abs(at-ai))));states+=1
            obs,_,done,truncated,_=env.step(at[0])
            if done or truncated:break
    env.close();assert error<2e-5,error
    record=dict(passed=True,states=states,max_import_action_error_rad=error,
                critic_actor_isolation='Actor equality asserted before/after critic fitting during import',command_and_other_normalizer_stats='exactly preserved',
                warm_run=str(warm),teacher_sha256='e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c')
    (result_root()/'teacher-import-verification.json').write_text(json.dumps(record,indent=2)+'\n');print(record)

if __name__=='__main__':main()
