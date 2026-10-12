"""Local voice/text/button control of mounted walking and sit/stand simulation."""
import argparse,hashlib,io,json,os,subprocess,tempfile,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
from evaluate_motion import session
from evaluate_posture import environment,state
from evaluate_skills import WALKING,POSTURE
from skill_controller import SkillController,parse_command
from paths import ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8798);ap.add_argument('--seed',type=int,default=900)
    ap.add_argument('--no-render',action='store_true');ap.add_argument('--idle-exit',type=float,default=15.)
    ap.add_argument('--voice-python',type=Path,default=ROOT/'tmp/voice-venv/bin/python');ap.add_argument('--voice-model',type=Path,default=ROOT/'tmp/voice-model')
    a=ap.parse_args();env=environment(a.seed,revision='v07');controller=SkillController(session(WALKING),session(POSTURE))
    assert abs(env.model.body_mass.sum()-1.1847989141502486)<1e-8 and env.bam.max_current==1.75 and not env.spotter
    lock=threading.RLock();speech_lock=threading.Lock();shared=dict(epoch=0,reset=False,touched=time.monotonic(),jpeg=b'',state={},last_text='',last_message='Ready')
    voice_ready=a.voice_python.is_file() and (a.voice_model/'model.bin').is_file()
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
            controller.accept(intent,time.monotonic());shared['epoch']+=1
            shared['touched']=time.monotonic();shared['last_text']=text;shared['last_message']='Accepted: '+text
            return dict(ok=True,text=text,intent=intent.kind,seconds=intent.seconds)
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
                    with lock:shared['epoch']+=1;shared['reset']=True;shared['touched']=time.monotonic()
                    return self.respond(dict(ok=True))
                if self.path=='/command':
                    data=json.loads(self.rfile.read(n))
                    with lock:result=self.accept(data['text'])
                    return self.respond(result)
                if self.path=='/voice':
                    if not voice_ready:raise ValueError('Run training/setup_voice.py first.')
                    if not speech_lock.acquire(blocking=False):return self.respond(dict(error='Already recognizing speech'),409)
                    try:
                        audio=self.rfile.read(n)
                        if len(audio)!=n:raise ValueError('Incomplete recording')
                        with lock:shared['epoch']+=1;epoch=shared['epoch']
                        folder=ROOT/'tmp/voice-audio';folder.mkdir(parents=True,exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=folder,suffix='.wav') as f:
                            f.write(audio);f.flush()
                            options=os.environ.copy();options['HF_HUB_OFFLINE']='1';options['OMP_NUM_THREADS']='2'
                            output=subprocess.run([str(a.voice_python),str(ROOT/'training/speech_command.py'),f.name,'--model',str(a.voice_model)],capture_output=True,text=True,timeout=20,env=options)
                        if output.returncode:raise ValueError('Speech runtime failed; check the local voice setup.')
                        speech=json.loads(output.stdout)
                        if not speech['accepted']:return self.respond(dict(error=speech['reason'],speech=speech),422)
                        with lock:
                            if shared['epoch']!=epoch or time.monotonic()-controller.heartbeat_at>.6:
                                return self.respond(dict(error='Voice command cancelled or controller disconnected',speech=speech),409)
                            result=self.accept(speech['text']);result['speech']=speech
                        return self.respond(result)
                    finally:speech_lock.release()
                self.respond(dict(error='Not found'),404)
            except (ValueError,KeyError,TypeError,UnicodeError,json.JSONDecodeError) as e:self.respond(dict(error=str(e)),400)
            except (TimeoutError,subprocess.TimeoutExpired):self.respond(dict(error='Voice request timed out'),408)
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
                if controller.fault and shared['last_message']!=controller.fault:
                    shared['epoch']+=1;shared['last_message']=controller.fault
                shared['state']=dict(**controller.status(time.monotonic()),actual=[float(x) for x in [*env.body_lin_vel()[:2],env._gyro[2]]],
                                     seconds=float(env.data.time),measurement=m,voice_ready=voice_ready,last_text=shared['last_text'],message=shared['last_message'],
                                     mass_kg=float(env.model.body_mass.sum()),actuator_current_a=env.bam.max_current,policy_sha256=hashes,simulation_only=True)
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
