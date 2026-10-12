"""Offline, one-utterance speech recognition. Run in the isolated voice venv."""
import argparse,array,json,math,os,sys,time,wave
from pathlib import Path

def transcribe(path,model_path,backend='whisper'):
    started=time.perf_counter()
    with wave.open(str(path),'rb') as w:
        if (w.getnchannels(),w.getsampwidth(),w.getframerate())!=(1,2,16000):raise ValueError('Audio must be 16 kHz mono PCM16 WAV.')
        frames=w.getnframes()
        if not 1600<=frames<=16000*6:raise ValueError('Record 0.1 to 6 seconds.')
        samples=array.array('h');samples.frombytes(w.readframes(frames))
        if sys.byteorder!='little':samples.byteswap()
    if len(samples)!=frames:raise ValueError('Incomplete audio')
    audio=array.array('f',(x/32768 for x in samples))
    rms=math.sqrt(sum(x*x for x in audio)/frames)
    if rms<.005:return dict(text='',accepted=False,reason='No speech detected',rms=rms,backend=backend,elapsed_s=time.perf_counter()-started)
    if backend=='whistle':
        os.environ.update(NEEDLE_TELEMETRY='0',DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',NEEDLE3_LIB_PATH=str(Path(model_path)/'libneedle.dylib'))
        from needle import Whistle
        result=Whistle(weights=Path(model_path)/'whistle.cact').transcribe(audio,language='en',word_timestamps=True)
        words=result.get('words',[]);text=result.get('text','').strip()
        probabilities=[float(w['probability']) for w in words]
        # shortcut: word scores are uncalibrated; validate thresholds on human/noisy speech before hardware use.
        confidence=bool(text and probabilities and all(math.isfinite(p) and .2<=p<=1 for p in probabilities)
                        and math.exp(sum(math.log(p) for p in probabilities)/len(probabilities))>=.6)
        return dict(text=text,accepted=confidence,reason=None if confidence else 'Speech was unclear',backend=backend,
                    rms=rms,duration_s=frames/16000,elapsed_s=time.perf_counter()-started,words=words,
                    ttft_ms=result.get('ttft_ms'),decode_tps=result.get('decode_tps'))
    if backend!='whisper':raise ValueError('Unknown speech backend')
    import numpy as np
    from faster_whisper import WhisperModel
    model=WhisperModel(str(model_path),device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
    segments,info=model.transcribe(np.asarray(audio),language='en',beam_size=3,condition_on_previous_text=False,temperature=0.,vad_filter=False)
    segments=list(segments);text=' '.join(s.text.strip() for s in segments).strip()
    confidence=bool(segments and all(s.avg_logprob>=-.8 and s.no_speech_prob<.6 for s in segments))
    return dict(text=text,accepted=confidence,reason=None if confidence else 'Speech was unclear',backend=backend,
                rms=rms,duration_s=frames/16000,elapsed_s=time.perf_counter()-started,
                segments=[dict(text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('audio',type=Path);ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--backend',choices=['whisper','whistle'],default='whisper');a=ap.parse_args()
    try:result=transcribe(a.audio,a.model,a.backend)
    except (ValueError,wave.Error) as e:result=dict(text='',accepted=False,reason=str(e))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
