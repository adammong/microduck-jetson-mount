"""Archive a native LAB checkpoint with its normalizer and ONNX export."""
import argparse,json,shutil,tempfile,hashlib
from pathlib import Path
from microduck_local.export_onnx import export
from paths import ROOT
from fast_env import install_recipe

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--steps',type=int,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    checkpoint=a.run/'checkpoints'/f'model_{a.steps}_steps.zip'
    normalizer=a.run/'checkpoints'/f'model_vecnormalize_{a.steps}_steps.pkl'
    record=json.loads((a.run/'run.json').read_text())
    record.update(checkpoint_steps=a.steps,mount_revision='v06',keep_command_normalizer=True,
        speed_lo=.35,speed_hi=.55,episode_s=15.,action_delay=True,recipe_version=2,
        reward_terms={t.key:t.weight for t in install_recipe().terms},yaw_tracking_variance=.025,
        symmetry_loss=False,strict_nonfoot_floor=True,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as td:
        folder=Path(td);shutil.copyfile(checkpoint,folder/'model.zip')
        shutil.copyfile(normalizer,folder/'vecnormalize.pkl')
        (folder/'run.json').write_text(json.dumps(record))
        export(folder,a.out/'policy.onnx')
    shutil.copyfile(checkpoint,a.out/'model.sb3');shutil.copyfile(normalizer,a.out/'vecnormalize.pkl')
    (a.out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    shutil.copyfile(ROOT/'training/fast_env.py',a.out/'fast_env_snapshot.py')
    print(a.out)
if __name__=='__main__':main()
