"""Mounted sit/stand and settled-floor recovery on unchanged BAM physics."""
import argparse,json,math
from pathlib import Path
import numpy as np
import mujoco
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.getup import _getup_spawn,_getup_place
from evaluate import scene
from evaluate_motion import session

def environment(seed,randomized=False,payload_scale=1.,variant='loaded'):
    path=scene(variant,rev='v06');model=mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb')))
    assert .8<=payload_scale<=1.2
    if variant=='loaded':
        body=model.body('easy_mount_payload').id;model.body_mass[body]*=payload_scale;model.body_inertia[body]*=payload_scale
    else:assert variant=='stock' and payload_scale==1.
    env=BehaviorEnv('getup',model=model,scene_xml=str(path),seed=seed,actuator_force='bam',bam_current_scale=1.,
        random_yaw=False,domain_rand=randomized,obs_noise=randomized,action_delay=True,push_robot=False,spotter=False,
        standing_spawns=True,terminate_on_fall=False,height_termination=False,max_episode_s=60.)
    env.reset(seed=seed);env.head_cmd[:]=0;env.body_cmd[:]=0;env.twist_cmd[:]=0
    assert env.bam.max_current==1.75 and not env.spotter and not env.push_robot
    return env

def prime(env):
    action=(env._joint_qpos()-env.robot.default_pose).astype(np.float32)
    env.last_action=action.copy();env.prev_action=action.copy();env._delayed_action=action.copy()
    env.prev_joint_vel=env._joint_vel().copy();env._refresh_derived()

def spawn(env,pose):
    if pose in ('stand','sitstand'):return
    kind=pose if pose in ('back','front') else 'side'
    _getup_spawn(env,kind)
    if pose in ('left','right'):
        # Same folded-joint distribution, with each side explicitly tested.
        roll=(1 if pose=='left' else -1)*np.deg2rad(env._rng.uniform(80,115))
        env.data.qpos[3:7]=[np.cos(roll/2),np.sin(roll/2),0,0];_getup_place(env)
    prime(env)

def contacts(env):
    feet=set();props=set();payload=set();internal=set();maximum=0.
    for i,c in enumerate(env.data.contact):
        f=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,i,f)
        if abs(f[0])<=.05:continue
        a,b=int(c.geom1),int(c.geom2);names=[env.model.geom(g).name or '' for g in (a,b)]
        mount=any(n.startswith('easy_') for n in names)
        if env.floor_geom in (a,b):
            other=b if a==env.floor_geom else a
            if other in env.foot_geoms.values():feet.add(other)
            else:props.add(env.model.geom(other).name or '')
            if mount:payload.add(' / '.join(names))
        elif mount:internal.add(' / '.join(names))
        if mount:maximum=max(maximum,float(abs(f[0])))
    return dict(both_feet=len(feet)==2,props=sorted(props),payload_floor=sorted(payload),payload_internal=sorted(internal),max_payload_force_n=maximum)

def state(env):
    c=contacts(env);gravity=env._projected_gravity();height=float(env._trunk_xpos[2]);jaw=float(env.data.xpos[env.model.body('jaw_soft').id,2])
    tilt=math.degrees(math.acos(np.clip(-gravity[2],-1,1)))
    # Full stand excludes the low, head-down tripod that can fool height alone.
    standing=bool(c['both_feet'] and not c['props'] and not c['payload_internal'] and tilt<20 and height>=env.stand_z-.025 and jaw>=.18)
    sitting=bool(height<.07 and tilt<35 and c['props'] and not c['payload_floor'] and not c['payload_internal'])
    return dict(height_m=height,head_height_m=jaw,tilt_deg=tilt,standing=standing,sitting=sitting,**c)

def trial(actor,pose,seed,randomized=False,payload_scale=1.,render=None,replay=None,seconds=None,variant='loaded'):
    env=environment(seed,randomized,payload_scale,variant)
    if replay is None:spawn(env,pose)
    else:
        replay=np.asarray(replay,float)
        if replay.shape!=(env.model.nq,) or not np.isfinite(replay).all() or not np.isclose(np.linalg.norm(replay[3:7]),1.,atol=1e-5):
            raise ValueError('Replay must contain the finite 21-position pose and a unit root quaternion.')
        env.data.qpos[:]=replay;env.data.qpos[:2]=0;env.data.qvel[:]=0;mujoco.mj_forward(env.model,env.data)
        env.data.ctrl[:]=env._joint_qpos();env.bam.reset(env._joint_qpos());prime(env)
    initial=state(env);initial_qpos=env.data.qpos.tolist();rows=[];frames=[];renderer=None
    seconds=seconds or (19. if pose=='sitstand' else 10.)
    if render:
        renderer=mujoco.Renderer(env.model,height=360,width=640);camera=mujoco.MjvCamera();camera.distance=.8;camera.azimuth=125;camera.elevation=-24
        options=mujoco.MjvOption();options.geomgroup[3:]=0
    for k in range(round(seconds/.02)):
        t=k*.02;flag=float(pose=='sitstand' and 3<=t<11)
        env.twist_cmd[:]=[flag,0,0];obs=env._get_obs()
        if actor=='limp':action=(env._joint_qpos()-env.robot.default_pose).astype(np.float32)
        else:action=actor.run(None,{actor.get_inputs()[0].name:obs[None].astype(np.float32)})[0].reshape(14)
        env.step(action);r=state(env);r['time_s']=(k+1)*.02;r['sit_requested']=bool(flag);rows.append(r)
        assert np.isfinite(env.data.qpos).all() and np.isfinite(action).all()
        if renderer and k%2==0:
            from PIL import Image,ImageDraw
            camera.lookat[:]=[env.data.qpos[0],env.data.qpos[1],.14];renderer.update_scene(env.data,camera=camera,scene_option=options)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,640,64),fill='#141414')
            draw.text((8,5),f'v06 mount | {pose} | seed {seed} | t={t:.2f}s',fill='white')
            draw.text((8,25),f'height {r["height_m"]:.3f}m | tilt {r["tilt_deg"]:.1f} | stand {r["standing"]} | sit {r["sitting"]}',fill='white')
            draw.text((8,45),f'BAM 1.75 A, no assistance | mount floor {bool(r["payload_floor"])}',fill='white');frames.append(frame)
    stand=np.array([r['standing'] for r in rows]);sit=np.array([r['sitting'] for r in rows]);run=best=0
    for value in stand:run=run+1 if value else 0;best=max(best,run)
    final_stand=bool(stand[-100:].all());sit_hold=bool(sit[400:550].all()) if pose=='sitstand' else False
    result=dict(pose=pose,variant=variant,seed=seed,randomized=randomized,payload_scale=payload_scale,seconds=seconds,mass_kg=float(env.model.body_mass.sum()),initial=initial,initial_qpos=initial_qpos,final=rows[-1],
        recovered=final_stand and best>=150,full_stand_streak_s=best*.02,sit_hold=sit_hold,sit_fraction=float(sit[250:550].mean()) if pose=='sitstand' else 0.,
        stand_fraction=float(stand.mean()),final_mean_height_m=float(np.mean([r['height_m'] for r in rows[-100:]])),
        sit_mean_height_m=float(np.mean([r['height_m'] for r in rows[250:550]])) if pose=='sitstand' else None,
        sit_mean_tilt_deg=float(np.mean([r['tilt_deg'] for r in rows[250:550]])) if pose=='sitstand' else None,
        sit_quality=float(np.mean([np.exp(-((r['height_m']-.055)/.03)**2)*np.exp(-(r['tilt_deg']/30)**2)*(.2+.8*float(not(r['payload_floor'] or r['payload_internal']))) for r in rows[250:550]])) if pose=='sitstand' else 0.,
        final_quality=float(np.mean([np.exp(-((r['height_m']-.115)/.035)**2)*np.exp(-(r['tilt_deg']/25)**2)*(.1+.9*float(not(r['props'] or r['payload_internal']))) for r in rows[-150:]])),
        payload_floor_fraction=float(np.mean([bool(r['payload_floor']) for r in rows])),payload_internal_fraction=float(np.mean([bool(r['payload_internal']) for r in rows])),
        peak_payload_contact_force_n=max(r['max_payload_force_n'] for r in rows),payload_pairs=sorted({p for r in rows for p in r['payload_floor']+r['payload_internal']}),
        final_qpos=env.data.qpos.tolist(),first_full_stand_s=next((r['time_s'] for r in rows if r['standing']),None),
        sitstand_passed=bool(pose=='sitstand' and sit_hold and final_stand and not any(r['payload_floor'] or r['payload_internal'] for r in rows)))
    if renderer:
        import imageio.v2 as imageio
        from PIL import Image
        render=Path(render);render.parent.mkdir(parents=True,exist_ok=True)
        imageio.mimsave(str(render.with_suffix('.mp4')),[np.array(f) for f in frames],fps=25,macro_block_size=1)
        sheet=Image.new('RGB',(2560,1080))
        for i,j in enumerate(np.linspace(0,len(frames)-1,12,dtype=int)):sheet.paste(frames[j],((i%4)*640,(i//4)*360))
        sheet.save(render.with_suffix('.png'));renderer.close()
    env.close();return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--poses',nargs='+',choices=['stand','sitstand','back','front','left','right'],default=['back','front','left','right'])
    ap.add_argument('--seed-start',type=int,default=100);ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--noise-dr',action='store_true')
    ap.add_argument('--payload-scales',type=float,nargs='+',default=[1.]);ap.add_argument('--render',action='store_true');a=ap.parse_args()
    actor='limp' if a.policy=='limp' else session(a.policy);rows=[];a.out.parent.mkdir(parents=True,exist_ok=True)
    for pose in a.poses:
        for seed in range(a.seed_start,a.seed_start+a.seeds):
            for scale in a.payload_scales:
                dest=a.out.parent/(a.out.stem+f'-{pose}-{seed}-{scale:g}') if a.render and seed==a.seed_start else None
                r=trial(actor,pose,seed,a.noise_dr,scale,render=dest);rows.append(r)
                a.out.write_text(json.dumps(dict(policy=a.policy,trials=rows),indent=2)+'\n')
                print(pose,seed,scale,r['recovered'],r['sitstand_passed'],round(r['final_mean_height_m'],3),round(r['payload_floor_fraction'],3),flush=True)
if __name__=='__main__':main()
