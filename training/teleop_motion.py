"""Local keyboard/button control of the actual loaded MuJoCo duck and ONNX."""
import argparse,hashlib,io,json,threading,time
from collections import deque
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
import mujoco
from evaluate import scene
from evaluate_motion import session
from fast_env import FastRunEnv
from motion_controller import CommandLimiter
from paths import ROOT

HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Drive Microduck</title>
<style>body{margin:0;background:#f7f6f2;color:#25352f;font:16px system-ui,sans-serif}main{max-width:1040px;margin:30px auto;padding:0 20px}h1{font-size:32px;margin:0 0 8px}.muted{color:#66776d}section{display:grid;grid-template-columns:2fr 1fr;gap:22px;margin-top:24px}.view,.panel{background:white;border:1px solid #e0e6df;border-radius:20px;padding:18px}.view img{width:100%;border-radius:12px;background:#dfe8e1}.tag{display:inline-block;background:#e8eee6;border-radius:20px;padding:6px 12px;font-size:13px}button{border:1px solid #d5ddd6;background:#f3f6f1;border-radius:12px;padding:18px 8px;color:inherit;font:inherit;cursor:pointer;touch-action:none}button.active,button:active{background:#34664b;color:white}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:16px}small{font-size:12px;color:#66776d}.stop{background:#f3dfcf;border-color:#e1c7b3}#reset{width:100%;margin-top:14px}dl{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:20px}dd{text-align:right;margin:0;font-variant-numeric:tabular-nums}@media(max-width:760px){section{grid-template-columns:1fr}}</style>
<main><span class="tag">Simulation · Jetson + battery mount</span><h1 style="margin-top:14px">Drive Microduck</h1><div class="muted">Hold a button or a key to move. Release to stop.</div>
<section><div class="view"><img id="scene" alt="Live Microduck physics simulation"><p id="status" class="muted">Starting simulation…</p><small>Real motor limits and communication delay. Hardware performance is untested.</small></div><div class="panel"><strong>Movement</strong><div class="grid">
<button data-key="q">Turn left<br><small>Q</small></button><button data-key="w">Forward<br><small>W / ↑</small></button><button data-key="e">Turn right<br><small>E</small></button>
<button data-key="a">Left<br><small>A / ←</small></button><button class="stop" id="stop">Stop<br><small>Space</small></button><button data-key="d">Right<br><small>D / →</small></button>
<span></span><button data-key="s">Backward<br><small>S / ↓</small></button><span></span></div>
<dl><dt>Forward</dt><dd id="vx">—</dd><dt>Sideways</dt><dd id="vy">—</dd><dt>Turning</dt><dd id="wz">—</dd><dt>Time</dt><dd id="time">—</dd></dl><button id="reset">Reset duck</button><p class="muted">Combine movement and turn keys to follow curves.</p></div></section></main>
<script>const held=new Set(),map={ArrowUp:'w',ArrowDown:'s',ArrowLeft:'a',ArrowRight:'d'};let busy=false,stopped=false;
function command(){if(stopped)return [0,0,0];return [(held.has('w')?.2:0)-(held.has('s')?.15:0),(held.has('a')?.1:0)-(held.has('d')?.1:0),(held.has('q')?.4:0)-(held.has('e')?.4:0)]}
async function send(){if(busy)return;busy=true;try{await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command:command()})})}finally{busy=false}document.querySelectorAll('[data-key]').forEach(b=>b.classList.toggle('active',held.has(b.dataset.key)))}
function clear(){held.clear();stopped=true;fetch('/stop',{method:'POST'});send()}
document.querySelectorAll('[data-key]').forEach(b=>{b.onpointerdown=e=>{b.setPointerCapture(e.pointerId);held.add(b.dataset.key);stopped=false;send()};b.onpointerup=b.onpointercancel=()=>{held.delete(b.dataset.key);send()}});
window.onkeydown=e=>{const k=map[e.key]||e.key.toLowerCase();if(k===' '){e.preventDefault();clear();return}if('wasdqe'.includes(k)&&k.length===1){e.preventDefault();held.add(k);stopped=false;send()}};
window.onkeyup=e=>{held.delete(map[e.key]||e.key.toLowerCase());send()};window.onblur=clear;document.onvisibilitychange=()=>{if(document.hidden)clear()};document.querySelector('#stop').onclick=clear;document.querySelector('#reset').onclick=()=>{clear();fetch('/reset',{method:'POST'})};
setInterval(send,180);let image=document.querySelector('#scene');function frame(){image.src='/frame.jpg?t='+Date.now()}image.onload=()=>setTimeout(frame,100);image.onerror=()=>setTimeout(frame,500);frame();
setInterval(async()=>{try{let s=await(await fetch('/state')).json();document.querySelector('#status').textContent=s.status;['vx','vy','wz'].forEach((k,i)=>document.querySelector('#'+k).textContent=(s.actual[i]>=0?'+':'')+s.actual[i].toFixed(2)+(i===2?' rad/s':' m/s'));document.querySelector('#time').textContent=s.seconds.toFixed(1)+' s'}catch{}},250);
</script></html>'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--policy',type=Path,default=ROOT/'results/motion-v06/policies/selected/policy.onnx')
    ap.add_argument('--port',type=int,default=8798);ap.add_argument('--seed',type=int,default=90)
    ap.add_argument('--idle-exit',type=float,default=15.,help='Exit after this many minutes without input activity (0 disables).');args=ap.parse_args()
    policy=session(args.policy);policy_hash=hashlib.sha256(args.policy.read_bytes()).hexdigest()
    limiter=CommandLimiter();lock=threading.Lock();shared=dict(jpeg=b'',state=dict(status='Starting',actual=[0.,0.,0.],seconds=0.),reset=False,touched=time.monotonic())
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,body,kind='application/json',status=200):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            route=self.path.split('?')[0]
            if route=='/':
                with lock:shared['touched']=time.monotonic()
            if route=='/':return self.respond(HTML.encode(),'text/html; charset=utf-8')
            with lock:
                if route=='/state':return self.respond(json.dumps(shared['state']).encode())
                if route=='/frame.jpg':return self.respond(shared['jpeg'],'image/jpeg')
            self.respond(b'{"error":"Not found"}',status=404)
        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{args.port}',f'http://localhost:{args.port}'):
                return self.respond(b'{"error":"Wrong origin"}',status=403)
            try:
                with lock:
                    if self.path=='/command':
                        n=int(self.headers.get('Content-Length','0'))
                        if n>4096:raise ValueError('Request too large')
                        limiter.set(json.loads(self.rfile.read(n))['command'],time.monotonic())
                        if np.any(limiter.target):shared['touched']=time.monotonic()
                    elif self.path=='/stop':limiter.stop();shared['touched']=time.monotonic()
                    elif self.path=='/reset':shared['reset']=True;limiter.stop();shared['touched']=time.monotonic()
                    else:return self.respond(b'{"error":"Not found"}',status=404)
                self.respond(b'{"ok":true}')
            except (ValueError,KeyError,TypeError):self.respond(b'{"error":"Invalid command"}',status=400)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    path=scene('loaded',rev='v06')
    env=FastRunEnv(model=mujoco.MjModel.from_binary_path(str(path.with_suffix('.mjb'))),scene_xml=str(path),seed=args.seed,actuator_force='bam',domain_rand=False,obs_noise=False,action_delay=True,random_yaw=False,max_episode_s=36000,command_resample_s=1e9)
    assert abs(env.model.body_mass.sum()-1.179337261211853)<1e-8 and env.bam.max_current==1.75 and not env.spotter
    env.reset(seed=args.seed);env.head_cmd[:]=0;env.body_cmd[:]=0
    renderer=mujoco.Renderer(env.model,height=360,width=640);cam=mujoco.MjvCamera();cam.distance=.8;cam.azimuth=125;cam.elevation=-24
    opts=mujoco.MjvOption();opts.geomgroup[3:]=0;history=deque(maxlen=25);paused=False;tick=0
    print(f'Control simulation: http://127.0.0.1:{args.port} | W/S forward/back, A/D sideways, Q/E turn, Space stop',flush=True)
    try:
        while True:
            began=time.monotonic()
            with lock:
                reset=shared['reset'];shared['reset']=False;command=limiter.step(began)
                if args.idle_exit>0 and began-shared['touched']>args.idle_exit*60:break
            if reset:env.reset(seed=args.seed);env.head_cmd[:]=0;env.body_cmd[:]=0;history.clear();paused=False
            if not paused:
                env.twist_cmd[:]=command;obs=env._get_obs()
                action=policy.run(None,{policy.get_inputs()[0].name:obs[None].astype(np.float32)})[0].reshape(14)
                _,_,done,_,info=env.step(action)
                history.append([*env.body_lin_vel()[:2],float(env._gyro[2])])
                if done:
                    paused=True
                    with lock:limiter.stop()
            if tick%5==0:
                from PIL import Image
                cam.lookat[:]=[env.data.qpos[0],env.data.qpos[1],.14];renderer.update_scene(env.data,camera=cam,scene_option=opts)
                buf=io.BytesIO();Image.fromarray(renderer.render()).save(buf,format='JPEG',quality=82)
                with lock:
                    shared['jpeg']=buf.getvalue();shared['state']=dict(status='Duck fell. Reset to continue.' if paused else 'Ready — hold keys or buttons to move',actual=np.mean(history,axis=0).tolist() if history else [0.,0.,0.],command=command.tolist(),seconds=float(env.data.time),paused=paused,simulation_only=True,mass_kg=float(env.model.body_mass.sum()),servo_current_limit_a=env.bam.max_current,policy_sha256=policy_hash)
            tick+=1;time.sleep(max(0,.02-(time.monotonic()-began)))
    except KeyboardInterrupt:pass
    finally:server.shutdown();server.server_close();renderer.close();env.close()
if __name__=='__main__':main()
