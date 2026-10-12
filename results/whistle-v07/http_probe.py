"""Exercise Whistle through the real local controller, including cancelled voice plans."""
import json,time,threading,urllib.request,urllib.error
from pathlib import Path
BASE='http://127.0.0.1:8801';OUT=Path('results/whistle-v07')
def req(route,data=None,raw=False):
    body=data if raw else None if data is None else json.dumps(data).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE+route,data=body,headers={'Content-Type':'audio/wav' if raw else 'application/json'}),timeout=30) as r:return r.status,json.load(r)
    except urllib.error.HTTPError as e:return e.code,json.load(e)
def state():return req('/state')[1]
quit=threading.Event()
def beat():
    while not quit.is_set():req('/heartbeat',{});quit.wait(.15)
def complete():
    end=time.monotonic()+35;peak=0.;phases=[]
    while time.monotonic()<end:
        s=state();assert not s['fault'],s;peak=max(peak,s['measurement']['tilt_deg'])
        if not phases or phases[-1]!=s['phase']:phases.append(s['phase'])
        if s['plan']['current'] is None and not s['plan']['remaining'] and not s['pending_movement'] and s['phase'] in ('idle','seated'):
            return dict(final_phase=s['phase'],peak_tilt_deg=peak,phases=phases)
        time.sleep(.06)
    raise AssertionError(('Timed out',s))
def main():
    s=state();assert s['voice_ready'] and s['language_ready'] and s['speech_backend']=='whistle'
    hashes=s['policy_sha256'];threading.Thread(target=beat,daemon=True).start();records=[]
    try:
        cases=[('results/voice-v07/audio/command-00.wav',[dict(action='forward',seconds=3)]),('results/language-v07/audio/command-00.wav',[dict(action='backward',seconds=2),dict(action='sit')]),('results/language-v07/audio/command-01.wav',[dict(action='stand'),dict(action='turn_right',seconds=2)])]
        for path,steps in cases:
            start=time.perf_counter();code,response=req('/voice',Path(path).read_bytes(),True)
            assert code==200 and response['steps']==steps,(path,code,response)
            assert response['speech']['backend']=='whistle'
            row=dict(audio=path,status=code,response=response,request_wall_s=time.perf_counter()-start,execution=complete());records.append(row);print(path,'completed',flush=True)
        rejected=[]
        for path in ['results/voice-v07/audio/command-06.wav','results/voice-v07/audio/command-07.wav','results/whistle-v07/audio/noise.wav','results/whistle-v07/audio/tone.wav','results/whistle-v07/audio/silence.wav','results/whistle-v07/audio/noisy-06.wav']:
            code,response=req('/voice',Path(path).read_bytes(),True);assert code in (400,422),(path,code,response)
            assert state()['plan']['current'] is None
            rejected.append(dict(audio=path,status=code,response=response))
        cancelled=[]
        for route in ['/stop','/reset']:
            out={}
            def voice():out['response']=req('/voice',Path('results/language-v07/audio/command-00.wav').read_bytes(),True)
            job=threading.Thread(target=voice);job.start();time.sleep(.15);req(route,{});job.join(25)
            assert not job.is_alive() and out['response'][0]==409,out
            complete();cancelled.append(dict(route=route,response=out['response']))
        req('/reset',{});time.sleep(.2);code,response=req('/voice',b'not a WAV',True);assert code==422,(code,response)
        assert state()['policy_sha256']==hashes
        (OUT/'http-verification.json').write_text(json.dumps(dict(passed=True,synthetic_audio_only=True,microphone_tested=False,completed=records,rejected=rejected,cancelled=cancelled,invalid_wav_rejected=True,unchanged_policy_sha256=hashes),indent=2)+'\n')
        print('Whistle speech-to-motion and cancellation checks passed',flush=True)
    finally:quit.set();req('/stop',{})
if __name__=='__main__':main()
