"""Check shared WAV validation without loading either speech model."""
from pathlib import Path
import io,json,tempfile,wave
from speech_command import transcribe
from paths import ROOT
def main():
    folder=ROOT/'tmp/voice-validation';folder.mkdir(parents=True,exist_ok=True)
    checks=[]
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        for backend in ['whisper','whistle']:
            for channels,width,rate,frames in [(2,2,16000,16000),(1,1,16000,16000),(1,2,8000,8000),(1,2,16000,1599),(1,2,16000,96001)]:
                p=Path(tmp)/'invalid.wav'
                with wave.open(str(p),'wb') as w:
                    w.setnchannels(channels);w.setsampwidth(width);w.setframerate(rate);w.writeframes(bytes(channels*width*frames))
                try:transcribe(p,Path(tmp),backend)
                except ValueError:checks.append(dict(backend=backend,format=[channels,width,rate,frames],rejected=True))
                else:raise AssertionError('Invalid WAV accepted')
            p=Path(tmp)/'silent.wav'
            with wave.open(str(p),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(bytes(32000))
            result=transcribe(p,Path(tmp),backend);assert not result['accepted'] and result['text']==''
            data=p.read_bytes();p.write_bytes(data[:-2])
            try:transcribe(p,Path(tmp),backend)
            except ValueError:checks.append(dict(backend=backend,incomplete_rejected=True))
            else:raise AssertionError('Truncated audio accepted')
    out=ROOT/'results/whistle-v07';out.mkdir(parents=True,exist_ok=True)
    (out/'validation.json').write_text(json.dumps(dict(passed=True,checks=checks,silence_rejected_before_model_loading=True),indent=2)+'\n')
    print('Shared WAV validation checks passed')
if __name__=='__main__':main()
