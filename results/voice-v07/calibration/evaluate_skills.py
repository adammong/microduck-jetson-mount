"""Continuous mounted skill sequences; no physics resets between commands."""
import argparse,json,multiprocessing as mp,hashlib
from pathlib import Path
import numpy as np
from evaluate_motion import session
from evaluate_posture import environment,state
from skill_controller import SkillController,parse_command
from paths import ROOT

WALKING=ROOT/'results/terrain-v07/policies/selected/policy.onnx'
POSTURE=ROOT/'results/posture-v07/policies/sitstand/policy.onnx'
SEQUENCES={
    'voice_demo':['walk forward for three seconds','stop','sit down','walk backward for three seconds','turn left for three seconds','stop','sit','stand up'],
    'directions':['walk forward','walk backward','sidestep left','sidestep right','turn left','turn right','sit','stand','walk forward','stop'],
    'cycles':['sit','stand','walk forward','sit','stand','walk backward','sit','stand'],
    'interrupts':['walk forward for ten seconds','sit','stop','walk backward','sit','walk forward','stop','stand'],
    'lease_loss':['walk forward for ten seconds','stand','sit','walk forward for ten seconds','stand'],
}

def trial(name,seed,randomized=False,payload_scale=1.,blend_s=.3,render=None):
    env=environment(seed,randomized,payload_scale,revision='v07')
    controller=SkillController(session(WALKING),session(POSTURE),blend_s)
    words=SEQUENCES[name];records=[];rows=[];frames=[];renderer=None
    if render:
        import mujoco
        renderer=mujoco.Renderer(env.model,height=360,width=640)
        camera=mujoco.MjvCamera();camera.distance=.8;camera.azimuth=125;camera.elevation=-24
        options=mujoco.MjvOption();options.geomgroup[3:]=0
    index=-1;began=0.;intent=None;stable=0.;current=None;issued=False
    for k in range(8000):
        now=k*.02;m=state(env)
        if name!='lease_loss' or current is None or now-began<.4:controller.heartbeat(now)
        if not issued and now>=2:
            index+=1
            if index==len(words):break
            intent=parse_command(words[index]);controller.accept(intent,now)
            began=now;stable=0.;issued=True
            current=dict(text=words[index],intent=intent.kind,issued_s=now,samples=[],phases=[],completed=False)
            records.append(current)
        try:action=controller.action(env,m,now)
        except ValueError:break
        env.step(action);m=state(env);controller.check(m,(k+1)*.02)
        v=[float(x) for x in [*env.body_lin_vel()[:2],env._gyro[2]]]
        row=dict(time_s=(k+1)*.02,phase=controller.phase,goal=controller.goal,command=controller.command.tolist(),actual=v,**m)
        rows.append(row)
        if current:
            current['samples'].append(row)
            if controller.phase not in current['phases']:current['phases'].append(controller.phase)
            ready=(controller.phase=='seated' and m['sitting']) if intent.kind=='sit' else (controller.phase=='idle' and m['standing'] and max(abs(x) for x in v[:2])<.05 and abs(v[2])<.15)
            if intent.kind=='stop' and controller.phase=='seated':ready=m['sitting']
            if name=='lease_loss' and intent.kind=='move' and not controller.pending and controller.phase=='seated':ready=m['sitting']
            stable=stable+.02 if ready and now-began>=.5 else 0.
            # Interrupt a ten-second walk after one second; remaining cases wait
            # for completed postures, so no mid-transition teleport is hidden.
            interrupted=name=='interrupts' and index==0 and now-began>=1.
            if stable>=.8 or interrupted:
                current['completed']=True;current['ended_s']=(k+1)*.02
                movement=[r['actual'] for r in current['samples'] if r['phase']=='moving' and max(abs(x) for x in r['command'])>.03]
                mean=np.mean(movement,axis=0) if movement else np.zeros(3)
                current['mean_moving_twist']=mean.tolist()
                target=np.asarray(intent.command)
                current['direction_followed']=bool(movement and np.all(np.where(target!=0,mean*target>=.25*target**2,abs(mean)<=[.05,.05,.15]))) if intent.kind=='move' else None
                current['automatic_stand']=bool(intent.kind=='move' and 'standing' in current['phases'])
                current['movement_seconds']=sum(r['phase']=='moving' for r in current['samples'])*.02
                if name=='lease_loss' and intent.kind=='move':
                    current['lease_stopped']=current['movement_seconds']<=1.04 and controller.pending is None
                current.pop('samples');issued=False;current=None
            elif now-began>25:controller.fail('Sequence command timed out',now)
        if renderer and k%2==0:
            from PIL import Image,ImageDraw
            camera.lookat[:]=[env.data.qpos[0],env.data.qpos[1],.14];renderer.update_scene(env.data,camera=camera,scene_option=options)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,64),fill='#141414')
            text=words[max(0,index)]
            draw.text((8,5),f'Mounted skill controller | {name} | {row["time_s"]:.2f}s | seed {seed}',fill='white')
            draw.text((8,25),f'{text} | {controller.phase} | tilt {m["tilt_deg"]:.1f}deg',fill='white')
            draw.text((8,45),f'BAM 1.75 A, no assistance | actual {v[0]:+.2f}/{v[1]:+.2f}/{v[2]:+.2f}',fill='white');frames.append(frame)
        if controller.fault:break
    no_contact=not any(r['payload_floor'] or r['payload_internal'] for r in rows)
    complete=len(records)==len(words) and all(r['completed'] for r in records)
    motion_ok=all(r.get('lease_stopped') if name=='lease_loss' else r.get('direction_followed') for r in records if r['intent']=='move')
    result=dict(sequence=name,seed=seed,randomized=randomized,payload_scale=payload_scale,blend_s=blend_s,
                passed=bool(complete and no_contact and not controller.fault and motion_ok),completed=complete,
                safe=bool(no_contact and not controller.fault),fault=controller.fault,commands=records,transitions=controller.transitions,
                seconds=len(rows)*.02,mass_kg=float(env.model.body_mass.sum()),actuator_current_a=env.bam.max_current,
                no_spotter=not env.spotter,max_tilt_deg=max(r['tilt_deg'] for r in rows),
                max_payload_force_n=max(r['max_payload_force_n'] for r in rows),final=rows[-1],simulation_only=True)
    if renderer:
        import imageio.v2 as imageio
        from PIL import Image
        render=Path(render);render.parent.mkdir(parents=True,exist_ok=True)
        imageio.mimsave(render.with_suffix('.mp4'),[np.asarray(f) for f in frames],fps=25,macro_block_size=1)
        sheet=Image.new('RGB',(2560,1080))
        for i,j in enumerate(np.linspace(0,len(frames)-1,12,dtype=int)):sheet.paste(frames[j],((i%4)*640,(i//4)*360))
        sheet.save(render.with_suffix('.png'));renderer.close()
        result['trace']=rows
    env.close();return result

def job(args):return trial(*args)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--sequences',nargs='+',choices=list(SEQUENCES),default=list(SEQUENCES));ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--seed-start',type=int,default=700);ap.add_argument('--noise-dr',action='store_true');ap.add_argument('--payload-scales',type=float,nargs='+',default=[1.]);ap.add_argument('--blend',type=float,default=.3);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--render',action='store_true');a=ap.parse_args()
    jobs=[(name,a.seed_start+i,a.noise_dr,scale,a.blend,a.out.parent/(name+'-demo') if a.render and i==0 and scale==a.payload_scales[0] else None) for name in a.sequences for scale in a.payload_scales for i in range(a.seeds)]
    if a.render:rows=[trial(*j) for j in jobs]
    else:
        with mp.get_context('spawn').Pool(a.workers) as pool:rows=pool.map(job,jobs)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(simulation_only=True,walking_sha256=hashlib.sha256(WALKING.read_bytes()).hexdigest(),posture_sha256=hashlib.sha256(POSTURE.read_bytes()).hexdigest(),trials=rows),indent=2)+'\n')
    for r in rows:print(r['sequence'],r['seed'],r['payload_scale'],'pass',r['passed'],'safe',r['safe'],r['fault'],[(x['text'],x.get('direction_followed'),x.get('movement_seconds')) for x in r['commands']],flush=True)
if __name__=='__main__':main()
