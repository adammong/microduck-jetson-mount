"""LAB-launched PPO training; workers fork before Torch is imported."""
import sys,json
from pathlib import Path
from microduck_local import train
from fast_env import FastRunEnv,install_recipe
install_recipe()
original=train.env_class
train.env_class=lambda robot,task='walk':FastRunEnv if robot=='microduck' and task=='walk' else original(robot,task)
train._pass_through_command_dims=lambda venv,spec:None

if __name__=='__main__':
    train.main()
    name=sys.argv[sys.argv.index('--run-name')+1];out=train.RUNS_DIR/name
    p=out/'run.json';record=json.loads(p.read_text())
    record.update(keep_command_normalizer=True,mount_revision='v06',speed_recipe='mounted_fast_walk',
        strict_nonfoot_floor=True,symmetry_loss=False)
    p.write_text(json.dumps(record,indent=2)+'\n')
    from microduck_local.export_onnx import export
    export(out,out/'policy.onnx')
