"""Seeded terrain under the unchanged v07 robot and real BAM servo limits."""
from pathlib import Path
import json,hashlib,os
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from evaluate import scene
from fast_env import FastRunEnv
from paths import ROOT

TERRAINS={
    'flat':{},
    'bumps_2mm':{'relief_mm':2.},'bumps_5mm':{'relief_mm':5.},
    'bumps_10mm':{'relief_mm':10.},
    'uphill_3deg':{'slope_x_deg':3.},'downhill_3deg':{'slope_x_deg':-3.},
    'cross_left_3deg':{'slope_y_deg':3.},'cross_right_3deg':{'slope_y_deg':-3.},
    'uphill_5deg':{'slope_x_deg':5.},'downhill_5deg':{'slope_x_deg':-5.},
    'mixed_5mm_3deg':{'relief_mm':5.,'slope_x_deg':3.,'slope_y_deg':-2.},
    'threshold_5mm':{'threshold_mm':5.},
    'low_grip':{'friction':.5},
}
HALF_SIZE=5.;N=641

def build(training_grid=False):
    half_size,n=(4.,513) if training_grid else (HALF_SIZE,N)
    source=scene('loaded',rev='v07');target=scene('loaded',rev='v07-terrain');target.parent.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(source);root=tree.getroot()
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file',os.path.relpath((source.parent/mesh.get('file')).resolve(),target.parent))
    ET.SubElement(root.find('asset'),'hfield',name='terrain',nrow=str(n),ncol=str(n),size=f'{half_size} {half_size} 1 .1')
    floor=root.find(".//geom[@name='floor']");floor.set('type','hfield');floor.set('hfield','terrain');floor.set('pos','0 0 -.5');floor.attrib.pop('size',None)
    ET.indent(root);tree.write(target,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(target));model.hfield_data[:]=.5
    mujoco.mj_saveModel(model,str(target.with_suffix('.mjb')))
    base=mujoco.MjModel.from_binary_path(str(source.with_suffix('.mjb')))
    assert (model.nq,model.nv,model.nu)==(base.nq,base.nv,base.nu)
    assert np.array_equal(model.body_mass,base.body_mass) and np.array_equal(model.body_inertia,base.body_inertia)
    assert np.array_equal(model.key_qpos,base.key_qpos)
    for field in ["body_ipos","body_iquat","jnt_range","jnt_axis","dof_armature"]:
        assert np.allclose(getattr(model,field),getattr(base,field),rtol=0,atol=1e-12),field
    record=dict(revision='v07',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),terrain_scene_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),grid=[n,n],half_size_m=half_size,terrain_seed_independent_of_physics_seed=True,terrain_catalog=TERRAINS,collision_model='Original v07 groundcontact robot; existing floor replaced by a heightfield. Flat controls retain the original plane.',model_mass_kg=float(model.body_mass.sum()))
    folder=ROOT/'results/terrain-v07';folder.mkdir(parents=True,exist_ok=True);(folder/'scenes.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Terrain scene built, mass',record['model_mass_kg'])

def ramp(x):
    """Flat spawn then C1 slope entry; derivative never exceeds one."""
    a=np.maximum(abs(x)-.16,0);return np.sign(x)*np.where(a<.16,a*a/.32,a-.08)

def heights(name,seed,half_size=HALF_SIZE,n=N):
    config=TERRAINS[name];axis=np.linspace(-half_size,half_size,n);x,y=np.meshgrid(axis,axis);z=np.zeros_like(x)
    if 'relief_mm' in config:
        rng=np.random.default_rng(seed);waves=np.zeros_like(x)
        for _ in range(8):
            angle=rng.uniform(0,2*np.pi);wavelength=rng.uniform(.07,.18);phase=rng.uniform(0,2*np.pi)
            waves+=np.sin(2*np.pi/wavelength*(x*np.cos(angle)+y*np.sin(angle))+phase)/8
        blend=np.clip((np.hypot(x,y)-.16)/.14,0,1);blend=blend*blend*(3-2*blend)
        z+=waves*config['relief_mm']*.0005*blend
    z+=np.tan(np.radians(config.get('slope_x_deg',0)))*ramp(x)
    z+=np.tan(np.radians(config.get('slope_y_deg',0)))*ramp(y)
    if 'threshold_mm' in config:
        # One grid-cell beveled rise; vertical stairs are outside this study.
        z+=config['threshold_mm']*.001*(x>=.35)
    assert np.max(abs(z))<.49 and np.isfinite(z).all()
    return z

class TerrainEnv(FastRunEnv):
    def __init__(self,terrain,terrain_seed,spawn_xy=None,**kwargs):
        self.spawn_xy=None if spawn_xy is None else np.asarray(spawn_xy,float)
        assert self.spawn_xy is None or (self.spawn_xy.shape==(2,) and np.isfinite(self.spawn_xy).all())
        self.terrain=terrain;self.terrain_seed=int(terrain_seed)
        model=kwargs['model']
        self.half_size=HALF_SIZE;self.n=N
        if terrain!='flat':
            assert model.nhfield==1
            self.half_size=float(model.hfield_size[0,0]);self.n=int(model.hfield_nrow[0]);assert self.n==model.hfield_ncol[0]
        self.surface=heights(terrain,terrain_seed,self.half_size,self.n)
        if terrain!='flat':model.hfield_data[:]=(self.surface+.5).ravel()
        assert self.spawn_xy is None or np.max(abs(self.spawn_xy))<self.half_size-.5
        kwargs['height_termination']=False
        super().__init__(**kwargs)
        assert self.bam.max_current==1.75 and not self.spotter and not self.push_robot

    def terrain_height(self,xy):
        if self.terrain=='flat':return 0.
        q=np.clip((np.asarray(xy)+self.half_size)/(2*self.half_size)*(self.n-1),0,self.n-1-1e-6);i,j=np.floor(q).astype(int);a,b=q-[i,j]
        # The collider triangulates cells; bilinear lookup is only a conservative
        # height guard/metric, with sub-mm difference on these gentle surfaces.
        return float((1-b)*((1-a)*self.surface[j,i]+a*self.surface[j,i+1])+b*((1-a)*self.surface[j+1,i]+a*self.surface[j+1,i+1]))

    def reset(self,**kwargs):
        obs,info=super().reset(**kwargs)
        if self.spawn_xy is not None:
            self.data.qpos[:2]=self.spawn_xy
            self.data.qpos[2]+=self.terrain_height(self.spawn_xy)+.012
            mujoco.mj_forward(self.model,self.data);self.bam.reset(self.data.qpos[self.joint_qpos_adr])
            obs=self._get_obs()
        if 'friction' in TERRAINS[self.terrain]:
            for g in self._feet|{self._floor_id}:self.model.geom_friction[g,0]=TERRAINS[self.terrain]['friction']
        return obs,info

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        # Detect collapsed robots relative to local ground, not absolute world Z.
        if self.data.xpos[self.trunk_body_id,2]-self.terrain_height(self.data.qpos[:2])<.07:
            done=True;info['failure_reason']='local_ground_clearance'
        if max(abs(self.data.qpos[:2]))>self.half_size-.3:
            done=True;info['failure_reason']='terrain_boundary'
        return obs,reward,done,truncated,info

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--training-grid',action='store_true',help='Rebuild original 8m training grid; omit for final 10m tests.');a=ap.parse_args();build(a.training_grid)
