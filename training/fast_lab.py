"""Launch the existing LAB with the same mounted speed env as its trainer."""
from microduck_local import behaviors
from microduck_local import viz_server,motion
from fast_env import FastRunEnv,BEHAVIOR,install_recipe,BehaviorEnv
install_recipe()
# This dedicated LAB uses a pinned recipe snapshot. Restart while idle for
# changes, rather than hot-reloading away its external recipe/env registration.
motion.reload_self=lambda:None
behaviors.reload_library=install_recipe
def preview(behavior_id,**kwargs):
    return FastRunEnv(behavior_id,**kwargs) if behavior_id==BEHAVIOR else BehaviorEnv(behavior_id,**kwargs)
behaviors.BehaviorEnv=preview
original_preview=viz_server.trainee_env_kwargs
def trainee_kwargs(b,stage_env=None):
    kw=original_preview(b,stage_env)
    if b.id==BEHAVIOR:kw.update(domain_rand=False,obs_noise=False,action_delay=True)
    return kw
viz_server.trainee_env_kwargs=trainee_kwargs
# The upstream LAB normally overwrites locomotion commands with its demo
# script (including 0.9 m/s, sideways and turns). Keep the trainee's own
# sampled command mix instead, so the live view mirrors this experiment.
original_set_cmd=viz_server.Duck.set_cmd
def set_cmd(duck,cmd):
    if isinstance(duck.env,FastRunEnv):return
    return original_set_cmd(duck,cmd)
viz_server.Duck.set_cmd=set_cmd
original_reset=viz_server.Duck.reset
def reset(duck):
    original_reset(duck)
    if isinstance(duck.env,FastRunEnv):
        duck.env._sample_commands()
        duck.obs=duck.env._get_obs()
viz_server.Duck.reset=reset
if __name__=='__main__':viz_server.main()
