"""Download a pinned official 4-bit LFM2.5 model into an isolated Mac runtime."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPOSITORY='LiquidAI/LFM2.5-350M-MLX-4bit'
REVISION='f6cb4e006bb7a2d8a6afa14ec0a53e0586f65a5b'

def main():
    env=os.environ.copy();env['UV_CACHE_DIR']=str(ROOT/'tmp/uv-cache')
    target=ROOT/'tmp/language-venv'
    if not target.exists():subprocess.run(['uv','venv','--python',sys.executable,str(target)],check=True,env=env)
    py=target/'bin/python'
    subprocess.run(['uv','pip','install','--python',str(py),'-r',str(ROOT/'training/language-requirements.txt')],check=True,env=env)
    code='''from huggingface_hub import snapshot_download
from pathlib import Path
import hashlib,json,importlib.metadata
p=Path(snapshot_download(REPOSITORY,revision=REVISION,local_dir=TARGET,allow_patterns=['*.json','*.safetensors','*.jinja']))
files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file()}
assert all(n in files for n in ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json'])
(p/'download.json').write_text(json.dumps(dict(repository=REPOSITORY,revision=REVISION,sha256=files,packages={n:importlib.metadata.version(n) for n in ['mlx-lm','mlx','transformers','numpy']}),indent=2)+'\\n')
print('Offline LFM2.5-350M language model ready')
'''
    subprocess.run([str(py),'-c',f'REPOSITORY={REPOSITORY!r}\nREVISION={REVISION!r}\nTARGET={str(ROOT/"tmp/language-model")!r}\n'+code],check=True,cwd=ROOT)
if __name__=='__main__':main()
