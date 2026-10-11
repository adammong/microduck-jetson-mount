"""Frame, replay, isolation and physics checks for mounted posture studies."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from evaluate_posture import environment,trial
from evaluate_motion import session
from seated_policy import SeatedPolicy,DEFAULT
from posture_policy import STAND

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    envs=[environment(50,payload_scale=s) for s in [.9,1.,1.1]]
    masses=[float(e.model.body_mass.sum()) for e in envs]
    assert np.allclose(masses,[.73724318+s*.442094081211853 for s in [.9,1.,1.1]],atol=1e-8)
    e=envs[1];assert np.array_equal(e.robot.default_pose,DEFAULT)
    assert e.bam.max_current==1.75 and not e.spotter and not e.push_robot and e.model.opt.timestep==.005
    obs=e._get_obs();assert np.allclose(obs[6:20],e._joint_qpos()-DEFAULT,atol=1e-6)
    actor=SeatedPolicy(session(STAND),[-.524,1.047,0,.18,0,0,0]);obs[48]=1
    action=actor.run(None,{actor.get_inputs()[0].name:obs[None]})[0][0]
    assert np.max(abs(action-obs[6:20]))<=.030001
    # Lock the unit convention which invalidated the first seated-target search.
    assert np.max(abs((action+DEFAULT)-(e._joint_qpos()+np.clip(action-obs[6:20],-.03,.03))))<1e-6
    data=mujoco.MjData(e.model);mujoco.mj_resetDataKeyframe(e.model,data,e.model.key('SIT').id);mujoco.mj_forward(e.model,data)
    overlaps=[dict(a=e.model.geom(c.geom1).name or '',b=e.model.geom(c.geom2).name or '',penetration_m=-float(c.dist)) for c in data.contact if c.dist<0 and any((e.model.geom(g).name or '').startswith('easy_') for g in [c.geom1,c.geom2])]
    assert any('easy_front_dock' in r['a']+r['b'] and r['penetration_m']>.008 for r in overlaps)
    for e in envs:e.close()
    x=trial('limp','front',51,seconds=2);y=trial('limp','front',51,seconds=2);assert x==y
    try:trial('limp','front',51,replay=np.zeros(21),seconds=1)
    except ValueError:pass
    else:raise AssertionError('Invalid replay accepted')
    record=dict(passed=True,relative_joint_observation_checked=True,measured_joint_target_increment_limit_rad=.03,isolated_mass_kg=masses,nominal_current_limit_a=1.75,repeatability_passed=True,replay_validation_passed=True,stock_SIT_mount_overlaps=overlaps,policy_sha256=hashlib.sha256(Path('results/posture-v06/policies/front-recovery-candidate/policy.onnx').read_bytes()).hexdigest())
    a.out.write_text(json.dumps(record,indent=2)+'\n');print('Posture checks passed')
if __name__=='__main__':main()
