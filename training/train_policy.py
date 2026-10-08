"""Use the pinned trainer with our optional walking fall guard."""
import os
from microduck_local import train
if os.environ.get('MOUNT_KEEP_COMMAND_NORMALIZER')=='1':
    # The official actor was trained with normalized command slots. Freezing
    # its statistics must preserve those slots too, rather than resetting them.
    train._pass_through_command_dims=lambda venv,spec:None
if os.environ.get('MOUNT_STRICT_FLOOR','1')=='1':
    from strict_env import StrictWalkEnv
    original=train.env_class
    def env_class(robot,task='walk'):
        return StrictWalkEnv if robot=='microduck' and task=='walk' else original(robot,task)
    train.env_class=env_class
if __name__=='__main__':train.main()
