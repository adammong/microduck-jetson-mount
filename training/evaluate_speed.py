"""Deterministic, paired-seed forward-speed validation on the mounted duck."""
import argparse, json, math
from pathlib import Path
import numpy as np
import mujoco
import onnxruntime as ort
from fast_env import FastRunEnv
from evaluate import scene
from paths import ROOT

def trial(policy, command, seed, seconds, render=None, noise=False, randomization=False):
    path=scene('loaded',rev='v06')
    env=FastRunEnv(model=mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb'))),
        scene_xml=str(path),seed=seed,actuator_force='bam',domain_rand=randomization,
        obs_noise=noise,action_delay=True,random_yaw=False,
        command_resample_s=1000,max_episode_s=seconds+1)
    obs,_=env.reset(seed=seed)
    env.twist_cmd[:]=[command,0,0];env.head_cmd[:]=0;env.body_cmd[:]=0
    obs=env._get_obs();start=env.data.qpos[:3].copy()
    velocities=[];yaws=[];gyros=[];tilts=[];heights=[];airborne=0;payload_frames=0;nonfoot=0
    max_depth=0.;max_force=0.;pairs=set();frames=[];renderer=None
    if render:
        renderer=mujoco.Renderer(env.model,height=480,width=640)
        cam=mujoco.MjvCamera();cam.distance=.75;cam.azimuth=120;cam.elevation=-18
        opts=mujoco.MjvOption();opts.geomgroup[3:]=0
    def label(g):
        return env.model.geom(g).name or env.model.body(int(env.model.geom_bodyid[g])).name
    for k in range(round(seconds/.02)):
        action=policy.run(None,{policy.get_inputs()[0].name:obs[None].astype(np.float32)})[0].reshape(14)
        obs,reward,done,truncated,info=env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(env.data.qpos).all()
        velocities.append(env.body_lin_vel().copy())
        gyros.append(env._gyro.copy())
        rot=env.data.xmat[env.trunk_body_id].reshape(3,3)
        yaws.append(math.atan2(rot[1,0],rot[0,0]))
        tilts.append(math.degrees(math.acos(np.clip(-env._projected_gravity()[2],-1,1))))
        heights.append(float(env.data.xpos[env.trunk_body_id,2]))
        feet=set();payload=False;ground=False
        for ci,c in enumerate(env.data.contact):
            force=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,ci,force)
            if abs(force[0])<=.05:continue
            a,b=int(c.geom1),int(c.geom2);names=[label(a),label(b)]
            if env._floor_id in (a,b):
                other=b if a==env._floor_id else a
                if other in env._feet:feet.add(other)
                else:ground=True
            if any(n.startswith('easy_') for n in names):
                max_depth=max(max_depth,-float(c.dist));max_force=max(max_force,abs(float(force[0])))
                pairs.add(' / '.join(names))
                if env._floor_id not in (a,b):payload=True
        airborne+=not feet;payload_frames+=payload;nonfoot+=ground
        if renderer and k%2==0:
            from PIL import Image,ImageDraw
            cam.lookat[:]=[env.data.qpos[0],env.data.qpos[1],.14]
            renderer.update_scene(env.data,camera=cam,scene_option=opts)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,67),fill=(20,20,20))
            draw.text((9,6),f'v06 mounted duck | {render.stem} | t={(k+1)*.02:.2f}s',fill='white')
            v=np.mean(np.array(velocities[-25:])[:,0])
            draw.text((9,27),f'Forward {v:.2f} m/s | command {command:.2f} | BAM | no assistance',fill='white')
            yaw=float(np.unwrap(yaws)[-1]-np.unwrap(yaws)[0])
            draw.text((9,47),f'Trunk {heights[-1]*100:.1f} cm | tilt {tilts[-1]:.1f} deg | yaw {math.degrees(yaw):+.1f} deg | non-foot floor {ground}',fill='white')
            frames.append(frame)
        if done:break
    elapsed=(k+1)*.02;delta=env.data.qpos[:3]-start;v=np.array(velocities)
    sustained=v[min(100,len(v)-1):]
    result=dict(seed=seed,command_m_s=command,seconds=elapsed,survived=not done,
        requested_seconds=seconds,mean_body_forward_m_s=float(v[:,0].mean()),
        sustained_body_forward_m_s=float(sustained[:,0].mean()),
        forward_distance_m=float(delta[0]),net_forward_m_s=float(delta[0]/seconds),
        lateral_distance_m=float(delta[1]),mean_body_lateral_m_s=float(v[:,1].mean()),
        yaw_change_rad=float(np.unwrap(yaws)[-1]-np.unwrap(yaws)[0]),
        mean_body_yaw_rate_rad_s=float(np.mean(np.array(gyros)[:,2])),
        max_tilt_deg=max(tilts),min_height_m=min(heights),mean_height_m=float(np.mean(heights)),
        both_feet_airborne_fraction=airborne/(k+1),payload_selfcontact_fraction=payload_frames/(k+1),
        nonfoot_ground_fraction=nonfoot/(k+1),max_payload_penetration_m=max_depth,
        max_payload_contact_force_n=max_force,payload_contact_pairs=sorted(pairs),
        failure_reason=info.get('failure_reason','height_or_tilt' if done else None),
        mass_kg=float(env.model.body_mass.sum()),action_delay=True,
        observation_noise=noise,domain_randomization=randomization)
    if renderer:
        import imageio.v2 as imageio
        from PIL import Image
        render.parent.mkdir(parents=True,exist_ok=True)
        imageio.mimsave(str(render.with_suffix('.mp4')),[np.array(f) for f in frames],fps=25,macro_block_size=1)
        sheet=Image.new('RGB',(640*4,480*2))
        for i,j in enumerate(np.linspace(0,len(frames)-1,8,dtype=int)):
            sheet.paste(frames[j],((i%4)*640,(i//4)*480))
        sheet.save(render.with_suffix('.png'));renderer.close()
    env.close();return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commands',nargs='+',type=float,default=[.3,.45,.55])
    ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--seed-start',type=int,default=0)
    ap.add_argument('--seconds',type=float,default=15);ap.add_argument('--render',action='store_true')
    ap.add_argument('--obs-noise',action='store_true');ap.add_argument('--domain-rand',action='store_true')
    a=ap.parse_args();opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    policy=ort.InferenceSession(str(a.policy),opts,providers=['CPUExecutionProvider'])
    rows=[]
    for cmd in a.commands:
        for seed in range(a.seed_start,a.seed_start+a.seeds):
            dest=a.out.parent/(a.out.stem+f'-cmd{cmd:.2f}-seed{seed}'.replace('.','p')) if a.render and seed<a.seed_start+2 else None
            rows.append(trial(policy,cmd,seed,a.seconds,dest,a.obs_noise,a.domain_rand))
        group=rows[-a.seeds:]
        print(cmd,sum(r['survived'] for r in group),'/',a.seeds,
            'body',round(np.mean([r['mean_body_forward_m_s'] for r in group]),3),
            'net',round(np.mean([r['net_forward_m_s'] for r in group]),3),
            'yaw',round(np.mean([abs(r['yaw_change_rad']) for r in group]),3),
            'contacts',round(max(r['payload_selfcontact_fraction'] for r in group),3),flush=True)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(dict(policy=str(a.policy.resolve().relative_to(ROOT)),
            revision='v06',evaluation='FastRunEnv; 200 Hz physics / 50 Hz policy; real BAM limits; no assistance',trials=rows),indent=2)+'\n')
if __name__=='__main__':main()
