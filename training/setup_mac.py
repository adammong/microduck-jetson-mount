"""Fetch a pinned Apple Silicon MuJoCo/PPO harness, inputs, and build CAD scenes.
Run from the repository root: uv run --python 3.12 training/setup_mac.py
"""
from pathlib import Path
import hashlib,io,json,os,subprocess,tarfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]
COMMIT='bbf0326ef97975f7062368914e0729504d17226f'
URL=f'https://api.github.com/repos/jonathanhawkins/microduck-lab/tarball/{COMMIT}'

def main():
    target=ROOT/'tmp/microduck-lab';target.parent.mkdir(exist_ok=True)
    if not target.exists():
        archive=urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'microduck-mount-simulation'})).read()
        with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
            members=tar.getmembers();prefix=members[0].name.split('/')[0]
            for m in members:
                m.name=m.name.removeprefix(prefix+'/')
                if m.name==prefix:continue
                tar.extract(m,target,filter='data')
        (target/'.source-commit').write_text(COMMIT+'\n')
    for item in json.loads((ROOT/'training/inputs.json').read_text()):
        dest=ROOT/'tmp/sim-inputs'/item['file'];dest.parent.mkdir(exist_ok=True)
        if not dest.exists():urllib.request.urlretrieve(item['url'],dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256'],f'Input checksum mismatch: {dest}'
    env=os.environ.copy();env['UV_CACHE_DIR']=str(ROOT/'tmp/uv-cache')
    subprocess.run(['uv','sync','--locked','--directory',str(target/'microduck_local')],check=True,env=env)
    subprocess.run([str(target/'microduck_local/.venv/bin/python'),str(ROOT/'training/build_scenes.py')],check=True,cwd=ROOT)
    print('Ready: tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate.py --render')
if __name__=='__main__':main()
