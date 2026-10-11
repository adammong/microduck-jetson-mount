"""Record resting poses after physical tip-over impulses on the loaded duck."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from evaluate_posture import environment,state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed-start',type=int,default=900);ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--revision',default='v06');a=ap.parse_args()
    rows=[]
    for name,axis in [('front',[0,1,0]),('back',[0,-1,0]),('left',[1,0,0]),('right',[-1,0,0])]:
        for seed in range(a.seed_start,a.seed_start+a.seeds):
            env=environment(seed,revision=a.revision);env.model.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_ENERGY)
            for k in range(1200):
                env.data.xfrc_applied[:]=0
                if k<10:env.data.xfrc_applied[env.trunk_body_id,3:]=np.array(axis)*.35
                action=(env._joint_qpos()-env.robot.default_pose).astype(np.float32);env.step(action)
            s=state(env);kinetic=float(env.data.energy[1])
            # A real fall can end collapsed upright against the mount, not flat.
            assert not s['standing'] and np.linalg.norm(env.data.qvel[:6])<.1,(name,seed,s,env.data.qvel[:6])
            rows.append(dict(requested_tip=name,seed=seed,qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),root_speed_norm=float(np.linalg.norm(env.data.qvel[:6])),kinetic_energy_j=kinetic,state=s,torque_nm=(np.array(axis)*.35).tolist(),torque_duration_s=.2,settle_after_impulse_s=23.8))
            env.close()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(simulation_only=True,revision=a.revision,fall_poses=rows),indent=2)+'\n');print('Recorded',len(rows),'settled physical falls')
if __name__=='__main__':main()
