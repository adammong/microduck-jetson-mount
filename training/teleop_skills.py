"""Local voice/text/button control of mounted walking and sit/stand simulation."""
import argparse,hashlib,io,json,os,subprocess,tempfile,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
from evaluate_motion import session,COMMANDS
from evaluate_posture import environment,state
from evaluate_skills import WALKING,POSTURE
from skill_controller import SkillController,parse_command
from language_plan import PlanRunner,validate_plan,MOVEMENT,intent_for,request_clauses,validate_request_plan
from paths import ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8798);ap.add_argument('--seed',type=int,default=900)
    ap.add_argument('--no-render',action='store_true');ap.add_argument('--idle-exit',type=float,default=15.)
    ap.add_argument('--voice-python',type=Path,default=ROOT/'tmp/voice-venv/bin/python');ap.add_argument('--voice-model',type=Path,default=ROOT/'tmp/voice-model')
    ap.add_argument('--language-python',type=Path,default=ROOT/'tmp/language-venv/bin/python');ap.add_argument('--language-model',type=Path,default=ROOT/'tmp/language-model')
    ap.add_argument('--language-adapter',type=Path,default=ROOT/'results/language-v07/adapter')
    a=ap.parse_args();env=environment(a.seed,revision='v07');controller=SkillController(session(WALKING),session(POSTURE))
    plan=PlanRunner()
    assert abs(env.model.body_mass.sum()-1.1847989141502486)<1e-8 and env.bam.max_current==1.75 and not env.spotter
    lock=threading.RLock();speech_lock=threading.Lock();shared=dict(epoch=0,reset=False,touched=time.monotonic(),jpeg=b'',state={},last_text='',last_message='Ready')
    voice_ready=a.voice_python.is_file() and (a.voice_model/'model.bin').is_file()
    language_ready=a.language_python.is_file() and (a.language_model/'model.safetensors').is_file() and (a.language_adapter/'adapters.safetensors').is_file()
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('walking',WALKING),('posture',POSTURE)]}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,data,status=200,kind='application/json'):
            body=data if isinstance(data,bytes) else json.dumps(data).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            route=self.path.split('?')[0]
            if route=='/':
                with lock:shared['touched']=time.monotonic()
                return self.respond((ROOT/'training/skill_control.html').read_bytes(),kind='text/html; charset=utf-8')
            with lock:
                if route=='/state':return self.respond(shared['state'])
                if route=='/frame.jpg':return self.respond(shared['jpeg'],kind='image/jpeg')
            self.respond(dict(error='Not found'),404)
        def accept(self,text):
            intent=parse_command(text)
            plan.cancel()
            controller.accept(intent,time.monotonic());shared['epoch']+=1
            shared['touched']=time.monotonic();shared['last_text']=text;shared['last_message']='Accepted: '+text
            return dict(ok=True,text=text,intent=intent.kind,seconds=intent.seconds)
        def begin_interpretation(self):
            with lock:
                if controller.fault:raise ValueError('Reset the simulation after a fault.')
                heartbeat_at=controller.heartbeat_at
                plan.cancel();controller.accept(parse_command('stop'),time.monotonic());controller.heartbeat_at=heartbeat_at
                shared['epoch']+=1;shared['touched']=time.monotonic()
                shared['last_message']='Understanding locally…'
                self.interpretation_epoch=shared['epoch']
                return shared['epoch']
        def interpret(self,text,epoch,speech=None):
            if not isinstance(text,str) or not 0<len(text.strip())<=160:raise ValueError('Use a short request, up to 160 characters.')
            try:
                intent=parse_command(text)
                action=next(name for name in MOVEMENT if tuple(COMMANDS[name])==intent.command) if intent.kind=='move' else intent.kind
                step=dict(action=action)
                if intent.kind=='move':step['seconds']=intent.seconds
                steps=[step];language=None
            except ValueError:
                if not language_ready:raise ValueError('Run training/setup_language.py for natural-language requests.')
                request_clauses(text)
                options=os.environ.copy();options['HF_HUB_OFFLINE']='1';options['TOKENIZERS_PARALLELISM']='false'
                output=subprocess.run([str(a.language_python),str(ROOT/'training/language_command.py'),'--model',str(a.language_model),'--adapter',str(a.language_adapter)],input=json.dumps(dict(text=text))+'\n',capture_output=True,text=True,timeout=20,env=options)
                if output.returncode:raise ValueError('Language runtime failed; check the local LFM setup.')
                language=json.loads(output.stdout)
                if not language.get('accepted'):raise ValueError('Could not turn that request into supported actions.')
                steps=validate_request_plan(text,language['steps'])
            with lock:
                if shared['epoch']!=epoch or time.monotonic()-controller.heartbeat_at>.6:
                    return self.respond(dict(error='Request cancelled or controller disconnected'),409)
                plan.start(controller,steps,time.monotonic());shared['epoch']+=1
                shared['touched']=time.monotonic();shared['last_text']=text
                shared['last_message']='Plan: '+' → '.join(s['action'].replace('_',' ')+(f" ({s['seconds']:g}s)" if 'seconds' in s else '') for s in steps)
                intent=intent_for(steps[0]);result=dict(ok=True,text=text,intent=intent.kind,seconds=intent.seconds,steps=steps)
                if language:result['language']=dict(model=language['model'],elapsed_s=language['elapsed_s'],load_s=language['load_s'])
                if speech:result['speech']=speech
            return self.respond(result)
        def do_POST(self):
            self.connection.settimeout(5.)
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{a.port}',f'http://localhost:{a.port}'):
                return self.respond(dict(error='Wrong origin'),403)
            try:
                n=int(self.headers.get('Content-Length','0'))
                limit=200000 if self.path=='/voice' else 4096
                if not 0<=n<=limit:raise ValueError('Request too large')
                if self.path=='/heartbeat':
                    with lock:controller.heartbeat(time.monotonic())
                    return self.respond(dict(ok=True))
                if self.path=='/stop':
                    with lock:result=self.accept('stop')
                    return self.respond(result)
                if self.path=='/reset':
                    with lock:plan.cancel();shared['epoch']+=1;shared['reset']=True;shared['touched']=time.monotonic()
                    return self.respond(dict(ok=True))
                if self.path=='/command':
                    data=json.loads(self.rfile.read(n))
                    text=data['text']
                    try:parse_command(text)
                    except ValueError:
                        if not speech_lock.acquire(blocking=False):return self.respond(dict(error='Already interpreting a request'),409)
                        try:return self.interpret(text,self.begin_interpretation())
                        finally:speech_lock.release()
                    with lock:result=self.accept(text)
                    return self.respond(result)
                if self.path=='/voice':
                    if not voice_ready:raise ValueError('Run training/setup_voice.py first.')
                    if not speech_lock.acquire(blocking=False):return self.respond(dict(error='Already recognizing speech'),409)
                    try:
                        audio=self.rfile.read(n)
                        if len(audio)!=n:raise ValueError('Incomplete recording')
                        epoch=self.begin_interpretation()
                        folder=ROOT/'tmp/voice-audio';folder.mkdir(parents=True,exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=folder,suffix='.wav') as f:
                            f.write(audio);f.flush()
                            options=os.environ.copy();options['HF_HUB_OFFLINE']='1';options['OMP_NUM_THREADS']='2'
                            output=subprocess.run([str(a.voice_python),str(ROOT/'training/speech_command.py'),f.name,'--model',str(a.voice_model)],capture_output=True,text=True,timeout=20,env=options)
                        if output.returncode:raise ValueError('Speech runtime failed; check the local voice setup.')
                        speech=json.loads(output.stdout)
                        if not speech['accepted']:
                            with lock:
                                if shared['epoch']==epoch:shared['last_message']=speech['reason']
                            return self.respond(dict(error=speech['reason'],speech=speech),422)
                        return self.interpret(speech['text'],epoch,speech)
                    finally:speech_lock.release()
                self.respond(dict(error='Not found'),404)
            except (ValueError,KeyError,TypeError,UnicodeError,json.JSONDecodeError) as e:
                with lock:
                    if getattr(self,'interpretation_epoch',None)==shared['epoch']:shared['last_message']='Request not accepted: '+str(e)
                self.respond(dict(error=str(e)),400)
            except (TimeoutError,subprocess.TimeoutExpired):
                with lock:
                    if getattr(self,'interpretation_epoch',None)==shared['epoch']:shared['last_message']='Request timed out'
                self.respond(dict(error='Request timed out'),408)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    renderer=None
    if not a.no_render:
        import mujoco
        renderer=mujoco.Renderer(env.model,height=360,width=640);camera=mujoco.MjvCamera();camera.distance=.8;camera.azimuth=125;camera.elevation=-24
        options=mujoco.MjvOption();options.geomgroup[3:]=0
    print(f'Mounted voice/skill simulation: http://127.0.0.1:{a.port} | offline voice ready: {voice_ready}',flush=True)
    tick=0
    try:
        while True:
            now=time.monotonic()
            with lock:
                if a.idle_exit>0 and now-shared['touched']>a.idle_exit*60:break
                if shared['reset']:
                    env.reset(seed=a.seed);env.head_cmd[:]=0;env.body_cmd[:]=0;env.twist_cmd[:]=0
                    controller.reset();shared['reset']=False;shared['last_text']='';shared['last_message']='Reset to standing'
                if not controller.fault:
                    try:action=controller.action(env,state(env),now)
                    except ValueError:action=None
                    if action is not None:env.step(action);controller.check(state(env),time.monotonic())
                m=state(env)
                m['motion_settled']=bool(np.max(abs(env.body_lin_vel()[:2]))<.05 and abs(env._gyro[2])<.15)
                plan.update(controller,m,time.monotonic())
                if controller.fault and shared['last_message']!=controller.fault:
                    shared['epoch']+=1;shared['last_message']=controller.fault
                shared['state']=dict(**controller.status(time.monotonic()),actual=[float(x) for x in [*env.body_lin_vel()[:2],env._gyro[2]]],
                                     seconds=float(env.data.time),measurement=m,voice_ready=voice_ready,last_text=shared['last_text'],message=shared['last_message'],
                                     mass_kg=float(env.model.body_mass.sum()),actuator_current_a=env.bam.max_current,policy_sha256=hashes,simulation_only=True)
                shared['state'].update(language_ready=language_ready,language_model='LFM2.5-350M' if language_ready else None,plan=plan.status())
            if renderer and tick%5==0:
                from PIL import Image
                camera.lookat[:]=[env.data.qpos[0],env.data.qpos[1],.14];renderer.update_scene(env.data,camera=camera,scene_option=options)
                buf=io.BytesIO();Image.fromarray(renderer.render()).save(buf,format='JPEG',quality=82)
                with lock:shared['jpeg']=buf.getvalue()
            tick+=1;time.sleep(max(0,.02-(time.monotonic()-now)))
    except KeyboardInterrupt:pass
    finally:
        server.shutdown();server.server_close()
        if renderer:renderer.close()
        env.close()
if __name__=='__main__':main()
