import subprocess,sys,time,json,urllib.request
from pathlib import Path
Path("tmp/terrain-study").mkdir(parents=True,exist_ok=True)
port=8799;root=Path('results/terrain-v07');log=open('tmp/terrain-study/controller.log','w')
p=subprocess.Popen([sys.executable,'training/teleop_motion.py','--revision','v07','--policy',str(root/'policies/selected/policy.onnx'),'--terrain','bumps_5mm','--terrain-seed','1701','--port',str(port),'--seed','711','--idle-exit','.3'],stdout=log,stderr=log)
def req(route,data=None):
 body=None if data is None else json.dumps(data).encode()
 return json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}'+route,data=body,headers={'Content-Type':'application/json'}),timeout=3))
try:
 for _ in range(150):
  try:
   state=req('/state')
   if state.get('mass_kg'):break
  except Exception:pass
  time.sleep(.1)
 else:raise AssertionError('Controller did not start')
 assert state['terrain']=='bumps_5mm' and state['revision']=='v07' and not state['paused']
 for _ in range(30):req('/command',{'command':[.2,0,0]});time.sleep(.15)
 moving=req('/state');assert not moving['paused'] and moving['actual'][0]>.05,moving
 req('/stop',{});time.sleep(2);stopped=req('/state');assert not stopped['paused'] and abs(stopped['actual'][0])<.07,stopped
 req('/reset',{});time.sleep(.3);reset=req('/state');assert not reset['paused'] and reset['seconds']<1.
 (root/'controller-verification.json').write_text(json.dumps(dict(passed=True,moving=moving,stopped=stopped,reset=reset),indent=2)+'\n');print('Terrain controller start/move/stop/reset passed')
finally:
 p.terminate()
 try:p.wait(timeout=5)
 except subprocess.TimeoutExpired:p.kill();p.wait()
 log.close()
