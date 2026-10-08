"""Paired-seed no-assist BAM rollouts, metrics, and optional rendered video."""
from pathlib import Path
import argparse,json,math
import numpy as np
import mujoco,onnxruntime as ort
from microduck_local.walk_env import MicroduckWalkEnv
ROOT=Path(__file__).resolve().parents[1]

def scene(variant,feet=False):
    return ROOT/'tmp/mac-models'/variant/'src/mjlab_microduck/robot/microduck'/('scene_feet_only.xml' if feet else 'scene_walk.xml')

def rollout(variant,policy,cmd,seed,seconds=10,render=None,feet=False,strict=False):
    path=scene(variant,feet)
    compiled=mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb'))) if path.with_suffix('.mjb').exists() else None
    from strict_env import StrictWalkEnv
    cls=StrictWalkEnv if strict else MicroduckWalkEnv
    env=cls(model=compiled,scene_xml=str(path),seed=seed,actuator_force='bam',
            obs_noise=False,domain_rand=False,action_delay=False,random_yaw=False,
            command_resample_s=1000,max_episode_s=seconds+1)
    obs,_=env.reset(seed=seed)
    env.twist_cmd[:]=cmd;env.head_cmd[:]=0;env.body_cmd[:]=0
    obs=env._get_obs()
    assert obs.shape==(61,) and env.action_space.shape==(14,)
    start=env.data.qpos[:3].copy();n=int(seconds/.02);tilts=[];zs=[];errors=[];contacts=set()
    frames=[];renderer=None;nonfoot_frames=0;selfcontact_frames=0;max_penetration=0.;max_contact_force=0.
    def geom_label(g):
        name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,g)
        if name:return name
        body=env.model.body(int(env.model.geom_bodyid[g])).name
        mesh=int(env.model.geom_dataid[g])
        return body+":"+(env.model.mesh(mesh).name if env.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH else str(g))
    if render:
        renderer=mujoco.Renderer(env.model,height=360,width=480)
        cam=mujoco.MjvCamera();cam.distance=.65;cam.azimuth=125;cam.elevation=-16
        opts=mujoco.MjvOption();opts.geomgroup[3:]=0
    for k in range(n):
        action=np.zeros(14,dtype=np.float32) if policy is None else policy.run(None,{policy.get_inputs()[0].name:obs[None].astype(np.float32)})[0].reshape(14)
        obs,reward,done,truncated,info=env.step(action)
        assert np.isfinite(env.data.qpos).all() and np.isfinite(obs).all()
        assert all(v<=1e-6 for name,v in info.get('episode_rewards',{}).items() if name.endswith('_penalty'))
        tilt=math.degrees(math.acos(np.clip(-env._projected_gravity()[2],-1,1)))
        z=float(env.data.xpos[env.trunk_body_id,2]);tilts.append(tilt);zs.append(z)
        errors.append(float(np.linalg.norm(env.body_lin_vel()[:2]-np.array(cmd[:2]))))
        nonfoot=False;selfcontact=False
        for ci,contact in enumerate(env.data.contact):
            names=[geom_label(g) for g in (contact.geom1,contact.geom2)]
            force=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,ci,force)
            if abs(force[0])<.05:continue
            if 'floor' in names and not any('_foot_collision' in name for name in names):nonfoot=True
            if any(name.startswith('easy_') for name in names):
                contacts.add(' / '.join(names));max_penetration=max(max_penetration,-float(contact.dist))
                max_contact_force=max(max_contact_force,float(abs(force[0])))
                if 'floor' not in names:selfcontact=True
        nonfoot_frames+=nonfoot;selfcontact_frames+=selfcontact
        if renderer and k%2==0:
            from PIL import Image,ImageDraw
            cam.lookat[:]=[env.data.qpos[0],env.data.qpos[1],.14]
            renderer.update_scene(env.data,camera=cam,scene_option=opts)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,480,43),fill=(20,20,20));draw.text((8,6),f'{variant}   t={(k+1)*.02:.2f}s   tilt={tilt:.1f}deg   trunk={z*100:.1f}cm',fill='white')
            draw.text((8,24),f'BAM | cmd={cmd} | seed={seed} | no assistance',fill='white');frames.append(frame)
        if done: break
    elapsed=(k+1)*.02;delta=env.data.qpos[:3]-start
    result=dict(variant=variant,seed=seed,command=cmd,policy='null' if policy is None else str(Path(policy._model_path).resolve().relative_to(ROOT)) if Path(policy._model_path).resolve().is_relative_to(ROOT) else Path(policy._model_path).name,
        seconds=elapsed,survived=not done,forward_m=float(delta[0]),lateral_m=float(delta[1]),
        max_tilt_deg=max(tilts),final_height_m=zs[-1],mean_height_m=float(np.mean(zs)),
        mean_xy_velocity_error_m_s=float(np.mean(errors)),payload_contact_pairs=sorted(contacts),
        mass_kg=float(env.model.body_mass.sum()),contact_model='feet_only' if feet else 'groundcontact',nonfoot_ground_fraction=nonfoot_frames/(k+1),
        payload_selfcontact_fraction=selfcontact_frames/(k+1),max_payload_penetration_m=max_penetration,
        max_payload_contact_force_n=max_contact_force,strict_nonfoot_floor=strict,
        failure_reason=info.get('failure_reason','height_or_tilt' if done else None))
    if renderer:
        import imageio.v2 as imageio
        render.parent.mkdir(parents=True,exist_ok=True)
        imageio.mimsave(str(render.with_suffix('.mp4')),[np.array(f) for f in frames],fps=25,macro_block_size=1)
        from PIL import Image,ImageDraw
        count=min(8,len(frames));indices=np.linspace(0,len(frames)-1,count,dtype=int)
        sheet=Image.new('RGB',(480*4,360*math.ceil(count/4)),(30,30,30))
        for i,j in enumerate(indices):sheet.paste(frames[j],((i%4)*480,(i//4)*360))
        sheet.save(render.with_suffix('.png'));renderer.close()
    env.close();return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',default=str(ROOT/'tmp/sim-inputs/alpha_walking.onnx'))
    ap.add_argument('--seeds',type=int,default=8);ap.add_argument('--seconds',type=float,default=10)
    ap.add_argument('--variant',choices=['stock','loaded','both'],default='both');ap.add_argument('--feet-only',action='store_true');ap.add_argument('--strict',action='store_true')
    ap.add_argument('--out',type=Path,default=ROOT/'results/mac-v05/baseline.json');ap.add_argument('--render',action='store_true')
    args=ap.parse_args();policy=None if args.policy=='null' else ort.InferenceSession(args.policy,providers=['CPUExecutionProvider'])
    results=[]
    for variant in ('stock','loaded') if args.variant=='both' else [args.variant]:
        for name,cmd in [('stand',[0.,0.,0.]),('forward',[.3,0.,0.]),('turn',[0.,0.,.5])]:
            for seed in range(args.seeds):
                dest=args.out.parent/(args.out.stem+'-'+variant+'-'+name) if args.render and seed==0 else None
                r=rollout(variant,policy,cmd,seed,args.seconds,dest,args.feet_only,args.strict);r['task']=name;results.append(r)
            group=results[-args.seeds:];print(variant,name,sum(r['survived'] for r in group),'/',args.seeds,'mean seconds',round(np.mean([r['seconds'] for r in group]),3),flush=True)
    args.out.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
