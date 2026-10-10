"""Measure signed body-frame twist tracking, contacts and command transitions."""
import argparse, json, math
from pathlib import Path
import numpy as np
import mujoco
import onnxruntime as ort
from evaluate import scene
from fast_env import FastRunEnv
from paths import ROOT

COMMANDS = {
    'stop': [0., 0., 0.], 'forward': [.2, 0., 0.],
    'backward': [-.15, 0., 0.], 'left': [0., .1, 0.],
    'right': [0., -.1, 0.], 'turn_left': [0., 0., .4],
    'turn_right': [0., 0., -.4], 'curve_left': [.15, 0., .3],
    'curve_right': [.15, 0., -.3], 'diagonal_left': [.15, .07, 0.],
    'diagonal_right': [.15, -.07, 0.],
}

def session(path):
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    return ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])

def trial(policy, command, seed, seconds, render=None, noise=False, randomization=False, schedule=None):
    path=scene('loaded',rev='v06')
    env=FastRunEnv(model=mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb'))),
        scene_xml=str(path),seed=seed,actuator_force='bam',domain_rand=randomization,
        obs_noise=noise,action_delay=True,random_yaw=False,
        command_resample_s=1000,max_episode_s=seconds+1)
    obs,_=env.reset(seed=seed);env.head_cmd[:]=0;env.body_cmd[:]=0
    start=env.data.qpos[:3].copy();vs=[];ws=[];yaws=[];heights=[];tilts=[];commands=[]
    payload=0;ground=0;force_max=0.;depth_max=0.;pairs=set();frames=[];renderer=None
    if render:
        renderer=mujoco.Renderer(env.model,height=360,width=640)
        cam=mujoco.MjvCamera();cam.distance=.8;cam.azimuth=125;cam.elevation=-24
        opts=mujoco.MjvOption();opts.geomgroup[3:]=0
    for k in range(round(seconds/.02)):
        cmd=np.array(command,dtype=np.float32)
        if schedule:
            cmd=np.array(next(c for until,c in schedule if k*.02<until),dtype=np.float32)
        env.twist_cmd[:]=cmd;obs=env._get_obs()
        action=policy.run(None,{policy.get_inputs()[0].name:obs[None].astype(np.float32)})[0].reshape(14)
        obs,reward,done,truncated,info=env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(env.data.qpos).all()
        v=env.body_lin_vel().copy();w=env._gyro.copy();vs.append(v);ws.append(w);commands.append(cmd)
        rot=env.data.xmat[env.trunk_body_id].reshape(3,3)
        yaws.append(math.atan2(rot[1,0],rot[0,0]))
        heights.append(float(env.data.xpos[env.trunk_body_id,2]))
        tilts.append(math.degrees(math.acos(np.clip(-env._projected_gravity()[2],-1,1))))
        p=False;g=False
        for ci,c in enumerate(env.data.contact):
            f=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,ci,f)
            if abs(f[0])<=.05:continue
            a,b=int(c.geom1),int(c.geom2)
            names=[env.model.geom(i).name or env.model.body(int(env.model.geom_bodyid[i])).name for i in (a,b)]
            if env._floor_id in (a,b) and (b if a==env._floor_id else a) not in env._feet:g=True
            if any(n.startswith('easy_') for n in names):
                force_max=max(force_max,float(abs(f[0])));depth_max=max(depth_max,-float(c.dist))
                pairs.add(' / '.join(names))
                if env._floor_id not in (a,b):p=True
        payload+=p;ground+=g
        if renderer and k%2==0:
            from PIL import Image,ImageDraw
            cam.lookat[:]=[env.data.qpos[0],env.data.qpos[1],.14]
            renderer.update_scene(env.data,camera=cam,scene_option=opts)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,64),fill=(20,20,20))
            measured=np.mean(np.array(vs[-25:]),axis=0);yawrate=np.mean(np.array(ws[-25:])[:,2])
            draw.text((8,4),f'v06 mounted Microduck | {render.stem} | t={(k+1)*.02:.2f}s | seed {seed}',fill='white')
            draw.text((8,23),f'Command vx/vy/wz {cmd[0]:+.2f}/{cmd[1]:+.2f}/{cmd[2]:+.2f} | actual {measured[0]:+.2f}/{measured[1]:+.2f}/{yawrate:+.2f}',fill='white')
            draw.text((8,42),f'BAM, no assistance | trunk {heights[-1]*100:.1f}cm | tilt {tilts[-1]:.1f}deg | floor {g} | mount contact {p}',fill='white')
            frames.append(frame)
        if done:break
    vs=np.array(vs);ws=np.array(ws);cmds=np.array(commands);n=len(vs);warm=min(100,n-1)
    actual=np.column_stack((vs[:,:2],ws[:,2]));error=actual-cmds
    result=dict(seed=seed,command=list(command),schedule=schedule,seconds=n*.02,requested_seconds=seconds,
        survived=not done,mean_twist=actual.mean(axis=0).tolist(),
        sustained_twist=actual[warm:].mean(axis=0).tolist(),
        mean_absolute_twist_error=np.abs(error).mean(axis=0).tolist(),
        displacement_world_m=(env.data.qpos[:3]-start).tolist(),yaw_change_rad=float(np.unwrap(yaws)[-1]-np.unwrap(yaws)[0]),
        min_height_m=min(heights),mean_height_m=float(np.mean(heights)),max_tilt_deg=max(tilts),
        payload_selfcontact_fraction=payload/n,nonfoot_ground_fraction=ground/n,
        max_payload_contact_force_n=force_max,max_payload_penetration_m=depth_max,payload_contact_pairs=sorted(pairs),
        failure_reason=info.get('failure_reason','height_or_tilt' if done else None),
        mass_kg=float(env.model.body_mass.sum()),action_delay=True,observation_noise=noise,domain_randomization=randomization)
    if schedule:
        result['segments']=[];begin=0.
        for end,c in schedule:
            lo=round((begin+min(1.,(end-begin)/2))/.02);hi=min(n,round(end/.02))
            if hi>lo:result['segments'].append(dict(start_s=begin,end_s=end,command=c,mean_twist=actual[lo:hi].mean(axis=0).tolist(),mean_absolute_twist_error=np.abs(error[lo:hi]).mean(axis=0).tolist()))
            begin=end
    if renderer:
        import imageio.v2 as imageio
        from PIL import Image
        render.parent.mkdir(parents=True,exist_ok=True)
        imageio.mimsave(str(render.with_suffix('.mp4')),[np.array(f) for f in frames],fps=25,macro_block_size=1)
        sheet=Image.new('RGB',(640*4,360*3))
        for i,j in enumerate(np.linspace(0,len(frames)-1,12,dtype=int)):sheet.paste(frames[j],((i%4)*640,(i//4)*360))
        sheet.save(render.with_suffix('.png'));renderer.close()
    env.close();return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tasks',nargs='+',choices=list(COMMANDS),default=list(COMMANDS));ap.add_argument('--seeds',type=int,default=4)
    ap.add_argument('--seed-start',type=int,default=20);ap.add_argument('--seconds',type=float,default=15.)
    ap.add_argument('--render',action='store_true');ap.add_argument('--obs-noise',action='store_true');ap.add_argument('--domain-rand',action='store_true')
    args=ap.parse_args();policy=session(args.policy);rows=[];args.out.parent.mkdir(parents=True,exist_ok=True)
    for task in args.tasks:
        for seed in range(args.seed_start,args.seed_start+args.seeds):
            dest=args.out.parent/(args.out.stem+f'-{task}-seed{seed}') if args.render and seed<args.seed_start+2 else None
            r=trial(policy,COMMANDS[task],seed,args.seconds,dest,args.obs_noise,args.domain_rand);r['task']=task;rows.append(r)
        group=rows[-args.seeds:]
        print(task,sum(r['survived'] for r in group),'/',args.seeds,'twist',np.round(np.mean([r['mean_twist'] for r in group],axis=0),3).tolist(),'contacts',max(r['payload_selfcontact_fraction'] for r in group),flush=True)
        args.out.write_text(json.dumps(dict(policy=str(args.policy.relative_to(ROOT)) if args.policy.is_absolute() else str(args.policy),revision='v06',trials=rows),indent=2)+'\n')
if __name__=='__main__':main()
