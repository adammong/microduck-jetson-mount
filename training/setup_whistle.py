"""Install pinned, isolated Whistle weights and the native Apple Silicon runtime."""
from pathlib import Path
import os,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
MODEL_REVISION='ca5287601bef25af26dcf2e1b2bdc0843a7c19e5'
ENGINE_REVISION='93e88ede867e4311c39151285e74df513e4b835b'
def main():
    import platform
    if sys.platform!='darwin' or platform.machine()!='arm64':raise SystemExit('This setup targets Apple Silicon Macs.')
    env=os.environ.copy();env.update(UV_CACHE_DIR=str(ROOT/'tmp/uv-cache'),HF_HOME=str(ROOT/'tmp/hf-cache'),NEEDLE_TELEMETRY='0',DO_NOT_TRACK='1')
    target=ROOT/'tmp/whistle-venv'
    if not target.exists():subprocess.run(['uv','venv','--python',sys.executable,str(target)],check=True,env=env)
    py=target/'bin/python'
    subprocess.run(['uv','pip','install','--python',str(py),'-r',str(ROOT/'training/whistle-requirements.txt')],check=True,env=env)
    code='''from huggingface_hub import hf_hub_download
from pathlib import Path
import hashlib,json,zipfile,importlib.metadata
p=Path(TARGET);p.mkdir(parents=True,exist_ok=True)
weights=Path(hf_hub_download('Cactus-Compute/whistle','whistle.cact',revision=MODEL_REVISION))
assert hashlib.sha256(weights.read_bytes()).hexdigest()=='b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb'
(p/'whistle.cact').write_bytes(weights.read_bytes())
wheel=Path(hf_hub_download('Cactus-Compute/needle3','python/cactus_needle-3.3.1-py3-none-macosx_11_0_arm64.whl',revision=ENGINE_REVISION))
assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='19534f411ad71312031d6df5418d0e55b8d16075c28a217aef4b921349bdfb4a'
with zipfile.ZipFile(wheel) as z:(p/'libneedle.dylib').write_bytes(z.read('needle/libneedle3.dylib'))
(p/'LICENSE').write_bytes(Path(hf_hub_download('Cactus-Compute/whistle','LICENSE',revision=MODEL_REVISION)).read_bytes())
files={f.name:dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in p.iterdir() if f.is_file() and f.name!='download.json'}
(p/'download.json').write_text(json.dumps(dict(repository='Cactus-Compute/whistle',revision=MODEL_REVISION,engine_repository='Cactus-Compute/needle3',engine_revision=ENGINE_REVISION,engine_version='3.3.1',files=files,packages={n:importlib.metadata.version(n) for n in ['cactus-needle','huggingface-hub']}),indent=2)+'\\n')
print('Whistle and native Mac runtime ready for offline use')
'''
    subprocess.run([str(py),'-c',f'MODEL_REVISION={MODEL_REVISION!r}\nENGINE_REVISION={ENGINE_REVISION!r}\nTARGET={str(ROOT/"tmp/whistle-model")!r}\n'+code],check=True,cwd=ROOT,env=env)
    print('Run training/teleop_skills.py --speech-backend whistle with the simulation Python.')
if __name__=='__main__':main()
