"""Install a separate local speech runtime and pinned English tiny model."""
from pathlib import Path
import os,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
MODEL_REVISION='0d3d19a32d3338f10357c0889762bd8d64bbdeba'
def main():
 env=os.environ.copy();env['UV_CACHE_DIR']=str(ROOT/'tmp/uv-cache')
 target=ROOT/'tmp/voice-venv'
 if not target.exists():subprocess.run(['uv','venv','--python',sys.executable,str(target)],check=True,env=env)
 py=target/'bin/python'
 subprocess.run(['uv','pip','install','--python',str(py),'-r',str(ROOT/'training/voice-requirements.txt')],check=True,env=env)
 code='''from huggingface_hub import snapshot_download
from pathlib import Path
import hashlib,json,importlib.metadata
p=Path(snapshot_download('Systran/faster-whisper-tiny.en',revision=REVISION,local_dir=TARGET,allow_patterns=['config.json','model.bin','tokenizer.json','vocabulary.txt','preprocessor_config.json']))
files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()}
assert all(n in files for n in ['config.json','model.bin','tokenizer.json','vocabulary.txt'])
(p/'download.json').write_text(json.dumps(dict(repository='Systran/faster-whisper-tiny.en',revision=REVISION,sha256=files,packages={n:importlib.metadata.version(n) for n in ['faster-whisper','ctranslate2','av','numpy']}),indent=2)+'\\n')
print('Offline English tiny speech model ready')
'''
 subprocess.run([str(py),'-c',f'REVISION={MODEL_REVISION!r}\nTARGET={str(ROOT/"tmp/voice-model")!r}\n'+code],check=True,cwd=ROOT)
 print('Ready: run training/teleop_skills.py with the simulation Python.')
if __name__=='__main__':main()
