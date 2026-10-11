"""Terrain geometry, guard, actuator and baseline self-checks."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco
from terrain_env import heights,TERRAINS,TerrainEnv,HALF_SIZE
from evaluate import scene
from evaluate_motion import session,trial
from paths import ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--policy',type=Path);a=ap.parse_args()
 base=mujoco.MjModel.from_binary_path(str(scene('loaded',rev='v07').with_suffix('.mjb')))
 terrain=mujoco.MjModel.from_binary_path(str(scene('loaded',rev='v07-terrain').with_suffix('.mjb')))
 assert np.array_equal(base.body_mass,terrain.body_mass) and np.array_equal(base.body_inertia,terrain.body_inertia)
 for field in ["body_ipos","body_iquat","jnt_range","jnt_axis","dof_armature"]:
  assert np.allclose(getattr(base,field),getattr(terrain,field),rtol=0,atol=1e-12),field
 for name in TERRAINS:
  small=heights(name,50,4.,513);large=heights(name,50,5.,641)
  assert np.allclose(small,large[64:-64,64:-64],rtol=0,atol=1e-12),name
  z=heights(name,50);assert np.array_equal(z,heights(name,50))
  assert z[z.shape[0]//2,z.shape[1]//2]==0 and np.max(abs(z))<.49
  if 'bumps' in name:assert not np.array_equal(z,heights(name,51))
 e=TerrainEnv(terrain='mixed_5mm_3deg',terrain_seed=50,model=terrain,scene_xml=str(scene('loaded',rev='v07-terrain')),seed=50,actuator_force='bam',domain_rand=False,obs_noise=False,action_delay=True,random_yaw=False,max_episode_s=15,command_resample_s=1000)
 e.reset(seed=50);e.twist_cmd[:]=0;e.head_cmd[:]=0;e.body_cmd[:]=0
 assert e.model.opt.timestep==.005 and e.bam.max_current==1.75 and not e.push_robot and not e.spotter
 rays=[]
 for x,y in [(0,0),(.618,.241),(-.751,-.379),(1.,-.5)]:
  gid=np.array([-1],dtype=np.int32);distance=mujoco.mj_ray(e.model,e.data,np.array([x,y,1.]),np.array([0.,0.,-1.]),np.array([1,0,0,0,0,0],dtype=np.uint8),1,-1,gid)
  assert gid[0]==e._floor_id,(x,y,gid)
  error=abs(1-distance-e.terrain_height([x,y]));assert error<.001,error;rays.append(error)
 e.data.qpos[2]=.045;mujoco.mj_forward(e.model,e.data)
 _,_,done,_,info=e.step(np.zeros(14,dtype=np.float32));assert done
 e.reset(seed=50);e.data.qpos[0]=e.half_size-.1;e.data.qpos[2]=e.terrain_height(e.data.qpos[:2])+.12;mujoco.mj_forward(e.model,e.data)
 _,_,done,_,info=e.step(np.zeros(14,dtype=np.float32));assert done and info['failure_reason']=='terrain_boundary'
 e.reset(seed=50);spawn_base_z=float(e.data.qpos[2])
 e.spawn_xy=np.array([.7,.5]);e.reset(seed=50)
 assert np.array_equal(e.data.qpos[:2],e.spawn_xy)
 assert np.isclose(e.data.qpos[2],spawn_base_z+e.terrain_height(e.spawn_xy)+.012)
 assert not np.any(e.data.xfrc_applied)
 e.close()
 actor=session(ROOT/'results/posture-v07/policies/walking-adjusted/policy.onnx')
 legacy=trial(actor,[.2,0,0],50,3,revision='v07');flat=trial(actor,[.2,0,0],50,3,revision='v07',terrain='flat',terrain_seed=999)
 for key in legacy:assert legacy[key]==flat[key],key
 result=dict(passed=True,model_mass_kg=float(base.body_mass.sum()),full_inertials_and_joint_limits_unchanged=True,cad_mass_sha256=hashlib.sha256((ROOT/'clearance_mount/simulation/mass_properties.json').read_bytes()).hexdigest(),actuator_current_a=1.75,physics_timestep_s=.005,no_assistance=True,on_terrain_start_placement_checked=True,flat_control_matches_original_exactly=True,seeded_surface_repeatability=True,extended_grid_preserves_central_heights=True,ray_height_max_error_m=max(rays),collapsed_pose_rejected=True,boundary_rejected=True,terrain_boundary_half_size_m=float(terrain.hfield_size[0,0]))
 if a.policy:
  import onnx
  from terrain_policy import TerrainPolicy,BASE
  parameters=json.loads(a.policy.with_name('parameters.json').read_text())['parameters']
  baseline=session(BASE);input_name=baseline.get_inputs()[0].name;wrapped=TerrainPolicy(baseline,parameters);deployed=session(a.policy);rng=np.random.default_rng(445);error=0.
  for i in range(1000):
   obs=rng.normal(0,.3,(1,61)).astype(np.float32);obs[0,48:51]=0 if i%4==0 else rng.uniform([-.15,-.1,-.4],[.2,.1,.4]);before=obs.copy()
   inherited=baseline.run(None,{input_name:obs})[0];x=wrapped.run(None,{input_name:obs})[0];y=deployed.run(None,{input_name:obs})[0]
   error=max(error,float(np.max(abs(x-y))));assert np.array_equal(obs,before) and np.isfinite(y).all()
   assert np.max(abs(y-inherited))<=.120001 and np.array_equal(y[:,5:7],inherited[:,5:7])
   if i%4==0:assert np.array_equal(inherited,y)
  assert error<1e-6
  tensors={t.name:onnx.numpy_helper.to_array(t) for t in onnx.load(a.policy).graph.initializer}
  for t in onnx.load(BASE).graph.initializer:assert np.array_equal(tensors[t.name],onnx.numpy_helper.to_array(t))
  assert deployed.get_modelmeta().custom_metadata_map['command_limiter']==baseline.get_modelmeta().custom_metadata_map['command_limiter']
  result.update(export_samples=1000,max_export_error_rad=error,correction_bound_rad=.12,stopped_actor_exactly_unchanged=True,inherited_tensors_unchanged=True,head_unchanged=True,policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest())
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print('Terrain self-checks passed')
if __name__=='__main__':main()
