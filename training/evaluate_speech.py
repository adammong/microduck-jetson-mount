"""Compare actual offline recognizers on identical synthetic clean/noisy WAVs."""
from pathlib import Path
import argparse,hashlib,json,math,os,re,statistics,subprocess,time,wave
import numpy as np
from paths import ROOT

def words(text):return re.findall(r"[a-z0-9]+",text.lower())
def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]
def write_wav(path,audio):
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(np.rint(np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/whistle-v07');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'audio').mkdir(exist_ok=True)
    phrases=['walk forward','sit down','stand up','turn left','stop','walk forward for three seconds','do not walk forward','hello duck']
    phrases+=['Please back up for two seconds, then sit down.','Could you stand up and turn right for two seconds?']
    cases=[];rng=np.random.default_rng(7301)
    for i,text in enumerate(phrases):
        path=ROOT/('results/voice-v07/audio' if i<8 else 'results/language-v07/audio')/f'command-{i if i<8 else i-8:02d}.wav'
        cases.append(dict(text=text,kind='clean',path=path))
        with wave.open(str(path),'rb') as w:audio=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(float)/32768
        rms=math.sqrt(float(np.mean(audio**2)));noise=rng.normal(size=len(audio));noise*=rms/(10**(15/20)*math.sqrt(float(np.mean(noise**2))))
        noisy=a.out/'audio'/f'noisy-{i:02d}.wav';write_wav(noisy,audio+noise)
        cases.append(dict(text=text,kind='white_noise_15db',path=noisy))
    for kind,audio in [('silence',np.zeros(32000)),('noise',rng.normal(0,.03,32000)),('tone',.1*np.sin(np.arange(32000)*2*np.pi*440/16000))]:
        path=a.out/'audio'/f'{kind}.wav';write_wav(path,audio);cases.append(dict(text='',kind=kind,path=path))
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',NEEDLE_TELEMETRY='0',DO_NOT_TRACK='1',OMP_NUM_THREADS='2')
    rows=[]
    for case in cases:
        row=dict(expected_text=case['text'],kind=case['kind'],audio=str(case['path'].relative_to(ROOT)),audio_sha256=hashlib.sha256(case['path'].read_bytes()).hexdigest(),results={})
        for backend,prefix in [('whisper','voice'),('whistle','whistle')]:
            started=time.perf_counter()
            run=subprocess.run([str(ROOT/f'tmp/{prefix}-venv/bin/python'),str(ROOT/'training/speech_command.py'),str(case['path']),'--model',str(ROOT/f'tmp/{prefix}-model'),'--backend',backend],capture_output=True,text=True,timeout=20,env=env,check=True)
            result=json.loads(run.stdout);result['process_wall_s']=time.perf_counter()-started
            expected=words(case['text']);actual=words(result['text']);result['word_edits']=distance(expected,actual);result['reference_words']=len(expected)
            result['exact_transcript']=expected==actual
            result['recognized_correctly']=result['exact_transcript'] and (result['accepted'] if expected else not result['accepted'])
            row['results'][backend]=result
        rows.append(row);print(case['kind'],case['text'],[(b,r['text'],r['accepted'],round(r['process_wall_s'],3)) for b,r in row['results'].items()],flush=True)
    summary={}
    for backend in ['whisper','whistle']:
        summary[backend]={}
        for group in ['clean','white_noise_15db','non_speech']:
            selected=[r['results'][backend] for r in rows if (r['kind'] in ('silence','noise','tone') if group=='non_speech' else r['kind']==group)]
            reference_words=sum(r['reference_words'] for r in selected)
            summary[backend][group]=dict(trials=len(selected),correct=sum(r['recognized_correctly'] for r in selected),exact_transcripts=sum(r['exact_transcript'] for r in selected),rejected=sum(not r['accepted'] for r in selected),word_error_rate=sum(r['word_edits'] for r in selected)/reference_words if reference_words else None,median_process_wall_s=statistics.median(r['process_wall_s'] for r in selected))
    report=dict(synthetic_audio_only=True,human_microphone_tested=False,noise='Seeded additive white noise at 15 dB, not robot/motor noise',keyword_biasing=False,runtime='Fresh process per request, matching simulator integration',summary=summary,trials=rows)
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
