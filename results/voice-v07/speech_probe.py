from pathlib import Path
import subprocess,json,sys,wave
import imageio_ffmpeg,numpy as np
sys.path.insert(0,'training')
from skill_controller import parse_command
root=Path('results/voice-v07');rows=[]
phrases=['walk forward','sit down','stand up','turn left','stop','walk forward for three seconds','do not walk forward','hello duck']
for i,text in enumerate(phrases):
 name=root/'audio'/f'command-{i:02d}'
 subprocess.run(['/usr/bin/say','-o',str(name.with_suffix('.aiff')),text],check=True)
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(name.with_suffix('.aiff')),'-ar','16000','-ac','1','-c:a','pcm_s16le',str(name.with_suffix('.wav'))],check=True)
 name.with_suffix('.aiff').unlink()
 output=subprocess.check_output(['tmp/voice-venv/bin/python','training/speech_command.py',str(name.with_suffix('.wav')),'--model','tmp/voice-model'],text=True)
 d=json.loads(output)
 try:expected=parse_command(text);actual=parse_command(d['text']) if d['accepted'] else None;passed=actual==expected
 except ValueError:
  try:parse_command(d['text']);passed=not d['accepted']
  except ValueError:passed=True
 rows.append(dict(expected_text=text,audio=str(name.with_suffix('.wav')),speech=d,passed=passed))
 print(text,repr(d['text']),d['accepted'],passed,round(d.get('elapsed_s',0),2),flush=True)
(root/'speech-probe.json').write_text(json.dumps(dict(synthetic_audio=True,trials=rows),indent=2)+'\n')
