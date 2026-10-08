"""Package only final metrics and small policy artifacts; omit runtimes and CAD meshes."""
from pathlib import Path
import json,shutil,hashlib,importlib.metadata as md
import onnx
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/mac-v05'

def main():
    policies=OUT/'policies';policies.mkdir(exist_ok=True)
    runs=[]
    for variant in ('stock','loaded'):
        name=f'{variant}-bam-s0-1048576';source=ROOT/'tmp/mac-runs'/name
        dest=policies/variant;dest.mkdir(exist_ok=True)
        model=onnx.load(source/'policy.onnx')
        props={p.key:p.value for p in model.metadata_props};props.update(
            mount_revision='v05',experiment='CPU PPO nominal-physics pilot',
            status='REJECTED walking policy; simulation study only',
            strict_nonfoot_floor='false',harness_commit='bbf0326ef97975f7062368914e0729504d17226f')
        onnx.helper.set_model_props(model,props);onnx.save(model,dest/'policy.onnx')
        for filename in ('run.json','study.json','progress.jsonl','vecnormalize.pkl'):
            shutil.copy2(source/filename,dest/filename)
        study=json.loads((dest/'study.json').read_text());study['strict_nonfoot_floor']=False
        study['status']='Rejected: stock learned to stand rather than walk; loaded exploits non-foot floor support.'
        (dest/'study.json').write_text(json.dumps(study,indent=2)+'\n')
        # Native models are ZIP containers; retain them with a non-ignored suffix.
        shutil.copy2(source/'model.zip',dest/'model.sb3')
        runs.append(study)
    for p in OUT.glob('*.json'):
        data=json.loads(p.read_text())
        if isinstance(data,list):
            for r in data:
                if isinstance(r,dict) and r.get('policy','').startswith(str(ROOT)):
                    r['policy']=str(Path(r['policy']).relative_to(ROOT))
            p.write_text(json.dumps(data,indent=2)+'\n')
    for p in list(OUT.glob('baseline-*.mp4'))+list(OUT.glob('baseline-*.png')):
        p.unlink()
    for name in ('baseline-fresh.json','pilot-loaded.json'):
        p=OUT/name
        if p.exists():p.unlink()
    versions={k:md.version(k) for k in ('mujoco','torch','stable-baselines3','gymnasium','onnx','onnxruntime','numpy')}
    strict=[]
    for variant in ('stock','loaded'):
        source=ROOT/'tmp/mac-runs'/f'{variant}-bam-s0-16384-strict'
        strict.append(json.loads((source/'study.json').read_text()))
    (OUT/'training_summary.json').write_text(json.dumps(dict(runs=runs,strict_smoke_tests=strict,
        platform='Apple Silicon Mac; native CPU training',versions=versions,
        limitation='One training seed and short nominal-physics budgets; no hardware validation'),indent=2)+'\n')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in policies.rglob('*') if p.is_file()}
    (OUT/'policy_checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('Saved policy artifacts',len(hashes))
if __name__=='__main__':main()
