"""Speech-to-action, stale voice cancellation and HTTP validation on localhost."""
from pathlib import Path
import subprocess,sys,time,json,urllib.request,urllib.error,threading,wave
root=Path('results/voice-v07');port=8801;base=f'http://127.0.0.1:{port}'
log=(root/'http-smoke.log').open('w');p=subprocess.Popen([sys.executable,'training/teleop_skills.py','--no-render','--port',str(port),'--seed','910','--idle-exit','2'],stdout=log,stderr=subprocess.STDOUT)
def req(route,data=None,raw=False,headers=None):
 body=data if raw else None if data is None else json.dumps(data).encode()
 try:
  with urllib.request.urlopen(urllib.request.Request(base+route,data=body,headers=headers or {'Content-Type':'audio/wav' if raw else 'application/json'}),timeout=25) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as e:return e.code,json.load(e)
def wait(check,seconds=15):
 until=time.monotonic()+seconds
 while time.monotonic()<until:
  _,s=req('/state')
  assert not s.get('fault'),s
  if check(s):return s
  time.sleep(.08)
 raise AssertionError(('Timed out',s))
active=threading.Event();active.set();exit_beat=threading.Event()
def beat():
 while not exit_beat.is_set():
  if active.is_set():
   try:req('/heartbeat',{})
   except Exception:pass
  exit_beat.wait(.15)
records=[]
try:
 for _ in range(100):
  try:
   if req('/state')[1].get('voice_ready'):break
  except Exception:pass
  time.sleep(.1)
 else:raise AssertionError('Server not ready')
 thread=threading.Thread(target=beat,daemon=True);thread.start()
 for i,(label,expected,phase) in enumerate([('walk forward','move','idle'),('sit down','sit','seated'),('stand up','stand','idle'),('turn left','move','idle'),('stop','stop','idle'),('walk forward for three seconds','move','idle')]):
  audio=(root/'audio'/f'command-{i:02d}.wav').read_bytes();started=time.monotonic();status,result=req('/voice',audio,True)
  assert status==200 and result['intent']==expected,(label,status,result)
  if expected=='move':wait(lambda s:s['phase']=='moving')
  final=wait(lambda s:s['phase']==phase and not s['pending_movement'])
  records.append(dict(expected=label,response=result,final=final,end_to_end_s=time.monotonic()-started))
  print('Spoken command',label,'passed',flush=True)
 for i in [6,7]:
  status,result=req('/voice',(root/'audio'/f'command-{i:02d}.wav').read_bytes(),True);assert status in [400,422],(status,result)
 for bad in [{'text':'walk forward and sit'},{'text':'do not walk forward'},{'text':'walk forward for 100 seconds'},{'text':None},[]]:
  status,_=req('/command',bad);assert status==400,status
 status,_=req('/command',{'text':'sit'},headers={'Origin':'https://example.com','Content-Type':'application/json'});assert status==403
 silence=root/'audio/silence.wav'
 with wave.open(str(silence),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(bytes(32000))
 status,result=req('/voice',silence.read_bytes(),True);assert status==422 and not result['speech']['accepted']
 # A later stop/reset must supersede an in-flight utterance.
 cancellation=[]
 for route in ['/stop','/reset']:
  response={}
  def voice():response['value']=req('/voice',(root/'audio/command-00.wav').read_bytes(),True)
  job=threading.Thread(target=voice);job.start();time.sleep(.15);req(route,{});job.join(25)
  assert not job.is_alive() and response['value'][0]==409,response
  final=wait(lambda s:s['phase']=='idle' and not s['pending_movement']);cancellation.append(dict(route=route,response=response['value']))
 # An expired client lease cannot arm motion from otherwise valid speech.
 active.clear();time.sleep(.7)
 response={};job=threading.Thread(target=voice);job.start();job.join(25)
 assert response['value'][0]==409,response
 active.set();wait(lambda s:s['phase']=='idle' and not s['pending_movement'])
 req('/command',{'text':'walk forward for ten seconds'});wait(lambda s:s['phase']=='moving');active.clear();time.sleep(1.2)
 lost=wait(lambda s:s['phase']=='idle' and not s['pending_movement']);assert not lost['fault'];active.set()
 data=dict(passed=True,synthetic_audio_only=True,commands=records,negated_and_unknown_speech_rejected=True,silence_rejected=True,invalid_commands_rejected=True,foreign_origin_rejected=True,stale_voice_cancelled=cancellation,disconnected_voice_rejected=True,lost_client_stops=lost)
 (root/'http-verification.json').write_text(json.dumps(data,indent=2)+'\n');print('HTTP speech/action/cancellation checks passed',flush=True)
finally:
 exit_beat.set();p.terminate()
 try:p.wait(timeout=5)
 except subprocess.TimeoutExpired:p.kill();p.wait()
 log.close()
