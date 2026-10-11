"""v07 fit, geometry, physics, and export self-checks for the posture study."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco,onnx
from evaluate_posture import environment,trial
from clearance_policy import SitStandPolicy
from evaluate_motion import session
from posture_policy import STAND,SIT
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    payload=json.loads((ROOT/'clearance_mount/simulation/mass_properties.json').read_text())['mass_kg']
    envs=[environment(50,payload_scale=s,revision='v07') for s in [.9,1.,1.1]];masses=[float(e.model.body_mass.sum()) for e in envs]
    assert np.allclose(masses,[.73724318+s*payload for s in [.9,1.,1.1]],atol=1e-8)
    e=envs[1];assert e.bam.max_current==1.75 and e.model.opt.timestep==.005 and not e.spotter and not e.push_robot
    poses=[]
    for pose in ['SIT_neutral_head','FOLD']:
        d=mujoco.MjData(e.model);mujoco.mj_resetDataKeyframe(e.model,d,e.model.key('SIT' if pose.startswith('SIT') else pose).id)
        if pose.startswith('SIT'):
            for name in ['neck_pitch','head_pitch']:d.qpos[e.model.jnt_qposadr[e.model.joint(name).id]]=.3491
        mujoco.mj_forward(e.model,d);hits=[]
        for c in d.contact:
            labels=[e.model.geom(g).name or e.model.body(e.model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
            if c.dist<-.0002 and any(n.startswith('easy_') for n in labels):hits.append(dict(pair=labels,penetration_m=-float(c.dist)))
        assert not hits,(pose,hits);poses.append(dict(pose=pose,mount_penetrations_over_02mm=hits))
    validation=json.loads((ROOT/'clearance_mount/simulation/validation.json').read_text());assert validation['connector_keepout']['near_HOME_access_passed']
    assert not validation['near_home_joint_sweep']['interferences'] and not validation['wiring']['HOME_centerline_capsule_robot_interferences']
    parameters=json.loads((ROOT/'results/posture-v07/policies/sitstand/parameters.json').read_text())
    actor=SitStandPolicy(parameters['stand_parameters'],parameters['sit_parameters']);deployed=session(ROOT/'results/posture-v07/policies/sitstand/policy.onnx');rng=np.random.default_rng(1908);error=0.
    for _ in range(1000):
        obs=rng.normal(0,.2,(1,61)).astype(np.float32);obs[0,48]=rng.integers(0,2);before=obs.copy()
        x=actor.run(None,{'observation':obs})[0];y=deployed.run(None,{'observation':obs})[0]
        error=max(error,float(np.max(abs(x-y))));assert np.array_equal(obs,before) and np.isfinite(y).all() and np.all(y[:,5:7]==0)
    assert error<1e-6
    candidate=onnx.load(ROOT/'results/posture-v07/policies/sitstand/policy.onnx');tensors={p.name:onnx.numpy_helper.to_array(p) for p in candidate.graph.initializer}
    for prefix,path in [('stand_',STAND),('sit_',SIT)]:
        for tensor in onnx.load(path).graph.initializer:assert np.array_equal(tensors[prefix+tensor.name],onnx.numpy_helper.to_array(tensor))
    for env in envs:env.close()
    x=trial('limp','front',51,seconds=2,revision='v07');y=trial('limp','front',51,seconds=2,revision='v07');assert x==y
    out=dict(passed=True,isolated_mass_kg=masses,nominal_current_a=1.75,physics_timestep_s=.005,no_assistance=True,geometry_poses=poses,connector_near_HOME_passed=True,neutral_head_export_error_rad=error,export_samples=1000,base_neural_tensors_unchanged=True,repeatability_passed=True,scene_sha256=hashlib.sha256((ROOT/'tmp/mac-models/v07/loaded/src/mjlab_microduck/robot/microduck/scene_walk.mjb').read_bytes()).hexdigest(),scope='Reference-model geometry and simulation only. Stock SIT head-down pose and unrestricted head articulation are not supported; neutral head is required for sitting.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print('Clearance checks passed')
if __name__=='__main__':main()
