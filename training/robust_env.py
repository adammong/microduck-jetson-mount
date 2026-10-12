"""Measured external disturbances; no recovery forces or privileged observations."""
import copy
import numpy as np
import mujoco
from terrain_env import TerrainEnv

class RobustTerrainEnv(TerrainEnv):
    def __init__(self,events,**kwargs):
        self.events=copy.deepcopy(events)
        for e in self.events:
            assert e['kind'] in ('push','snag','slip')
            assert np.isfinite([e['start'],e['duration']]).all() and e['start']>=0 and e['duration']>0
            if e['kind']=='slip':assert 0<e['friction']<=1
            else:assert np.asarray(e['force_body']).shape==(3,) and np.isfinite(e['force_body']).all()
            if e['kind']=='snag':assert e['foot'] in ('left','right') and e.get('window',1)>0
        super().__init__(**kwargs)
        self.foot_ids={side:self.model.geom(side+'_foot_collision').id for side in ('left','right')}
        self.foot_bodies={side:int(self.model.geom_bodyid[g]) for side,g in self.foot_ids.items()}

    def reset(self,**kwargs):
        obs,info=super().reset(**kwargs)
        self.normal_friction=self.model.geom_friction.copy()
        self.disturbance_records=[dict(event=copy.deepcopy(e),trigger_s=None,applied_seconds=0.,impulse_world_ns=[0.,0.,0.],peak_supported_foot_speed_m_s=0.,supported_foot_travel_m=0.) for e in self.events]
        return obs,info

    def foot_motion(self,side):
        g=self.foot_ids[side];b=self.foot_bodies[side];force=0.
        for i,c in enumerate(self.data.contact):
            if {int(c.geom1),int(c.geom2)}=={g,self._floor_id}:
                f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f);force+=max(0.,float(f[0]))
        velocity=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_GEOM,g,velocity,0)
        return force,velocity[3:]

    def step(self,action):
        t=float(self.data.time);dt=.02;rotation=self.data.xmat[self.trunk_body_id].reshape(3,3)
        self.model.geom_friction[:]=self.normal_friction
        active_slips=[];applied=[]
        for e,r in zip(self.events,self.disturbance_records):
            if r['trigger_s'] is None and t+1e-8>=e['start']:
                if e['kind']!='snag':r['trigger_s']=t
                elif t<=e['start']+e.get('window',1):
                    normal,v=self.foot_motion(e['foot'])
                    # A snag begins only while the selected foot swings forward.
                    if normal<.05 and float(v@rotation[:,0])>.025:r['trigger_s']=t
            if r['trigger_s'] is None or not (r['trigger_s']<=t<r['trigger_s']+e['duration']-1e-8):continue
            r['applied_seconds']+=dt
            if e['kind']=='slip':
                for g in self._feet|{self._floor_id}:self.model.geom_friction[g,0]=e['friction']
                active_slips.append(r)
            else:
                body=self.trunk_body_id if e['kind']=='push' else self.foot_bodies[e['foot']]
                force=rotation@np.asarray(e['force_body'])
                self.data.xfrc_applied[body,:3]+=force
                applied.append((body,force))
                r['impulse_world_ns']=(np.asarray(r['impulse_world_ns'])+force*dt).tolist()
        result=super().step(action)
        for body,force in applied:self.data.xfrc_applied[body,:3]-=force
        for r in active_slips:
            speeds=[]
            for i,c in enumerate(self.data.contact):
                a,b=int(c.geom1),int(c.geom2)
                if self._floor_id not in (a,b):continue
                foot=b if a==self._floor_id else a
                if foot not in self._feet:continue
                f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f)
                if f[0]<=.05:continue
                velocity=np.zeros(6)
                mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_GEOM,foot,velocity,0)
                v=velocity[3:]+np.cross(velocity[:3],c.pos-self.data.geom_xpos[foot])
                normal=c.frame[:3];v-=normal*np.dot(v,normal)
                speeds.append(np.linalg.norm(v))
            speed=max(speeds,default=0.)
            r['peak_supported_foot_speed_m_s']=max(r['peak_supported_foot_speed_m_s'],float(speed))
            r['supported_foot_travel_m']+=float(speed*dt)
        return result
