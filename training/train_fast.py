"""LAB-launched PPO training; workers fork before Torch is imported."""
import sys,json,os
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
        strict_nonfoot_floor=True,symmetry_loss=False,
        speed_lo=float(os.environ.get('MICRODUCK_FAST_SPEED_LO','.35')),
        speed_hi=float(os.environ.get('MICRODUCK_FAST_SPEED_HI','.55')),
        episode_s=15.,action_delay=True,
        reward_terms={t.key:t.weight for t in install_recipe().terms},
        yaw_tracking_variance=.025,recipe_version=2)
    p.write_text(json.dumps(record,indent=2)+'\n')
    from microduck_local.export_onnx import export
    export(out,out/'policy.onnx')
