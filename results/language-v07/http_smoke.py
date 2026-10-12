"""Test actual local text/audio requests, ordered physics execution and cancellation."""
import json,time,threading,urllib.request,urllib.error
from pathlib import Path
ROOT=Path('results/language-v07');BASE='http://127.0.0.1:8801'
def req(route,data=None,raw=False,headers=None):
    body=data if raw else None if data is None else json.dumps(data).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE+route,data=body,headers=headers or {'Content-Type':'audio/wav' if raw else 'application/json'}),timeout=45) as r:return r.status,json.load(r)
    except urllib.error.HTTPError as e:return e.code,json.load(e)
def state():return req('/state')[1]
active=threading.Event();active.set();quit=threading.Event()
def beat():
    while not quit.is_set():
        if active.is_set():req('/heartbeat',{})
        quit.wait(.15)
def wait(check,seconds=30):
    end=time.monotonic()+seconds;phases=[];actions=[];peak_tilt=0.;samples=0
    while time.monotonic()<end:
        s=state();samples+=1;peak_tilt=max(peak_tilt,s['measurement']['tilt_deg'])
        assert not s['fault'],s
        if s['phase'] not in phases:phases.append(s['phase'])
        step=s['plan']['current']
        if step and (not actions or actions[-1]!=step):actions.append(step)
        if check(s):return dict(final=s,phases=phases,actions_seen=actions,peak_tilt_deg=peak_tilt,sampled_frames=samples)
        time.sleep(.06)
    raise AssertionError(('Timed out',s))
def complete(s):return s['plan']['current'] is None and not s['plan']['remaining'] and s['pending_movement'] is False and s['phase'] in ('idle','seated')
def main():
    assert state()['language_ready'] and state()['voice_ready']
    threading.Thread(target=beat,daemon=True).start();records=[]
    cases=[('back up a little, then sit down',[dict(action='backward',seconds=1),dict(action='sit')]),('could you stand up and turn right for two seconds',[dict(action='stand'),dict(action='turn_right',seconds=2)]),('please move ahead briefly and then take a seat',[dict(action='forward',seconds=1),dict(action='sit')]),('take a seat, then rise to your feet',[dict(action='sit'),dict(action='stand')])]
    try:
        for text,expected in cases:
            start=time.monotonic();code,response=req('/command',dict(text=text));record=dict(text=text,status=code,response=response,interpretation_wall_s=time.monotonic()-start)
            if code==200:
                assert response['steps']==expected,(text,response,expected)
                record['execution']=wait(complete);record['completed']=True
            else:assert code in (400,422);record['completed']=False
            records.append(record);print(text,code,flush=True)
        for i,expected in enumerate([[dict(action='backward',seconds=2),dict(action='sit')],[dict(action='stand'),dict(action='turn_right',seconds=2)]]):
            start=time.monotonic();code,response=req('/voice',(ROOT/f'audio/command-{i:02d}.wav').read_bytes(),True);record=dict(audio=f'audio/command-{i:02d}.wav',synthetic_audio_only=True,status=code,response=response,interpretation_wall_s=time.monotonic()-start)
            if code==200:
                assert response['steps']==expected,(i,response,expected)
                record['execution']=wait(complete);record['completed']=True
            else:assert code in (400,422);record['completed']=False
            records.append(record);print('Synthetic voice',i,code,flush=True)
        for text in ["don't walk forward and then sit",'what happens when you stand?','go to my chair and sit','move forward for 99 seconds','please jump and sit','walk forward for ten seconds then move back for ten seconds then walk left for ten seconds']:
            code,response=req('/command',dict(text=text));assert code in (400,422),(text,code,response)
            assert state()['plan']['current'] is None,(text,state())
        code,_=req('/command',dict(text='sit'),headers={'Origin':'https://example.com','Content-Type':'application/json'});assert code==403
        cancellation=[]
        for route in ['/stop','/reset']:
            out={}
            def interpret():out['response']=req('/command',dict(text='please move ahead for two seconds, then take a seat'))
            job=threading.Thread(target=interpret);job.start();time.sleep(.15);req(route,{});job.join(30)
            assert not job.is_alive() and out['response'][0]==409,out
            wait(complete);cancellation.append(dict(route=route,response=out['response']))
        # A queued sit must not execute after losing the client's heartbeat.
        req('/reset',{});time.sleep(.2)
        code,response=req('/command',dict(text='please move ahead for two seconds, then take a seat'))
        assert code==200,response
        wait(lambda s:s['phase']=='moving');active.clear();time.sleep(.8)
        cancelled=wait(complete);assert cancelled['final']['phase']=='idle';active.set()
        # An expired client lease must not be renewed by language inference.
        active.clear();time.sleep(.7);code,response=req('/command',dict(text='could you move ahead briefly, then sit down'))
        assert code==409,(code,response);active.set()
        wait(complete)
        result=dict(passed=True,records=records,completed_plans=sum(r['completed'] for r in records),tested_plans=len(records),unsupported_requests_rejected=True,foreign_origin_rejected=True,stop_reset_supersede_inference=cancellation,lost_client_cancels_pending_sit=cancelled,expired_lease_rejected=True,microphone_capture_tested=False)
        assert result['completed_plans']>=2,result
        (ROOT/'http-verification.json').write_text(json.dumps(result,indent=2)+'\n');print('HTTP language/voice/physics checks passed',flush=True)
    finally:quit.set();req('/stop',{})
if __name__=='__main__':main()
