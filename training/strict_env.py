"""Walking guard: prolonged non-foot floor support ends the episode.

The large rear board can hold a collapsed robot above the usual height/tilt
fall thresholds. Preserve all rewards/observations and reject that support.
"""
import mujoco
import numpy as np
from microduck_local.walk_env import MicroduckWalkEnv

class StrictWalkEnv(MicroduckWalkEnv):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._floor_id=self.model.geom('floor').id
        self._feet={self.model.geom('left_foot_collision').id,self.model.geom('right_foot_collision').id}
        self._nonfoot_steps=0
    def reset(self,**kwargs):
        self._nonfoot_steps=0
        return super().reset(**kwargs)
    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        supported=False
        for i,contact in enumerate(self.data.contact):
            a,b=int(contact.geom1),int(contact.geom2)
            if self._floor_id not in (a,b):continue
            other=b if a==self._floor_id else a
            if other in self._feet:continue
            force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,force)
            if abs(force[0])>.05:supported=True;break
        self._nonfoot_steps=self._nonfoot_steps+1 if supported else 0
        if self._nonfoot_steps>=3:
            terminated=True;info['failure_reason']='nonfoot_floor_support'
            info['episode_rewards']=dict(self.reward_sums)
        return obs,reward,terminated,truncated,info
