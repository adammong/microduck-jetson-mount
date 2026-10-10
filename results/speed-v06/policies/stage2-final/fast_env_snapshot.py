"""Mounted forward-speed curriculum using the upstream running reward recipe."""
import os
from dataclasses import replace
import numpy as np
import mujoco
from microduck_local import behaviors
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.behaviors.core import BEHAVIORS, _register
from paths import ROOT

BEHAVIOR='mounted_fast_walk'

def yaw_tracking(env):
    # Gyro and the requested yaw rate are actor observations; no absolute
    # heading or world-position reward is introduced.
    error=float(env._gyro[2])-float(env.twist_cmd[2])
    return float(np.exp(-error*error/.025))

def install_recipe():
    weights={'keep_pace':8.,'track_turn':4.,'air_time':.5,'pose':.5}
    terms=tuple(replace(t,weight=weights.get(t.key,t.weight),
        fn=yaw_tracking if t.key=='track_turn' else t.fn) for t in BEHAVIORS['run'].terms)
    recipe=replace(BEHAVIORS['run'],id=BEHAVIOR,title='Faster walking with the Jetson mount',
        keywords=('mounted fast walk','walk faster','run faster'),suggest='mounted fast walk',
        description='Practise faster forward walking with the v06 payload and real BAM servo limits.',
        how_it_learns='Ninety percent of commands ask for forward walking; ten percent ask for standing. Speed tracking and gyro yaw control dominate the score. Non-foot floor support ends the attempt; no assistance or stronger motors.',
        terms=terms,episode_s=15.,
        trainer=(str(ROOT/'training/train_fast.py'),'--no-domain-rand','--no-obs-noise','--lr','0.000005'),
        symmetric=False,default_steps=3_000_000)
    _register(recipe)
    return recipe

class FastRunEnv(BehaviorEnv):
    def __init__(self,behavior_id=BEHAVIOR,**kwargs):
        install_recipe()
        knobs=kwargs.get('spawn_overrides') or {}
        self.speed_lo=float(knobs.get('MICRODUCK_FAST_SPEED_LO',os.environ.get('MICRODUCK_FAST_SPEED_LO','0.35')))
        self.speed_hi=float(knobs.get('MICRODUCK_FAST_SPEED_HI',os.environ.get('MICRODUCK_FAST_SPEED_HI','0.55')))
        assert 0<self.speed_lo<=self.speed_hi<=1.1
        # Explicit arguments from evaluation/preview win; no force scaling or assist.
        kwargs.setdefault('max_episode_s',15.)
        kwargs.setdefault('push_robot',False)
        self._nonfoot_steps=0
        super().__init__(behavior_id,**kwargs)
        self._floor_id=self.model.geom('floor').id
        self._feet={self.model.geom('left_foot_collision').id,self.model.geom('right_foot_collision').id}

    def _sample_commands(self):
        r=self._rng
        self.twist_cmd[:]=0.
        if r.uniform()>=.10:self.twist_cmd[0]=r.uniform(self.speed_lo,self.speed_hi)
        # Retain observable head/body command slots and the teacher normalizer.
        self.head_cmd[:]=r.uniform(-.015,.015,4)
        self.body_cmd[:]=r.uniform(-.015,.015,6)

    def reset(self,**kwargs):
        self._nonfoot_steps=0
        return super().reset(**kwargs)

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        supported=False
        for i,c in enumerate(self.data.contact):
            a,b=int(c.geom1),int(c.geom2)
            if self._floor_id not in (a,b):continue
            other=b if a==self._floor_id else a
            if other in self._feet:continue
            f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f)
            if abs(f[0])>.05:supported=True;break
        self._nonfoot_steps=self._nonfoot_steps+1 if supported else 0
        if self._nonfoot_steps>=3:
            done=True;info['failure_reason']='nonfoot_floor_support'
            info['episode_rewards']=dict(self.reward_sums)
        return obs,reward,done,truncated,info
