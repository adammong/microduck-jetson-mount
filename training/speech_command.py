"""Offline, one-utterance speech recognition. Run in the isolated voice venv."""
import argparse,json,time,wave
from pathlib import Path
import numpy as np

def transcribe(path,model_path):
    started=time.perf_counter()
    with wave.open(str(path),'rb') as w:
        if (w.getnchannels(),w.getsampwidth(),w.getframerate())!=(1,2,16000):raise ValueError('Audio must be 16 kHz mono PCM16 WAV.')
        frames=w.getnframes()
        if not 1600<=frames<=16000*6:raise ValueError('Record 0.1 to 6 seconds.')
        audio=np.frombuffer(w.readframes(frames),dtype='<i2').astype(np.float32)/32768
    if len(audio)!=frames or not np.isfinite(audio).all():raise ValueError('Incomplete audio')
    rms=float(np.sqrt(np.mean(audio**2)))
    if rms<.005:return dict(text='',accepted=False,reason='No speech detected',rms=rms)
    from faster_whisper import WhisperModel
    model=WhisperModel(str(model_path),device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
    segments,info=model.transcribe(audio,language='en',beam_size=3,condition_on_previous_text=False,temperature=0.,vad_filter=False)
    segments=list(segments);text=' '.join(s.text.strip() for s in segments).strip()
    confidence=bool(segments and all(s.avg_logprob>=-.8 and s.no_speech_prob<.6 for s in segments))
    return dict(text=text,accepted=confidence,reason=None if confidence else 'Speech was unclear',
                rms=rms,duration_s=frames/16000,elapsed_s=time.perf_counter()-started,
                segments=[dict(text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('audio',type=Path);ap.add_argument('--model',type=Path,required=True);a=ap.parse_args()
    try:result=transcribe(a.audio,a.model)
    except (ValueError,wave.Error) as e:result=dict(text='',accepted=False,reason=str(e))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
