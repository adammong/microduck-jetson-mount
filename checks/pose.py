"""The upstream HOME pose, for static inspection only (not a learned policy)."""
import mujoco
import numpy as np

HOME={"left_hip_yaw":0,"right_hip_yaw":0,"left_hip_roll":-0.0873,"right_hip_roll":0.0873,"left_hip_pitch":-0.4579,"right_hip_pitch":0.4579,"left_knee":-0.0049,"right_knee":0.0049,"left_ankle":0.4530,"right_ankle":-0.4530,"neck_pitch":0.3491,"head_pitch":0.3491,"head_yaw":0,"head_roll":0}

def home_pose(model):
    data=mujoco.MjData(model)
    for name,value in HOME.items():
        idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if idx>=0:
            data.qpos[model.jnt_qposadr[idx]]=value
    mujoco.mj_forward(model,data)
    # Normalize placement onto the inspection floor using both foot meshes.
    lowest=[]
    for i in range(model.ngeom):
        body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])) or ""
        if "foot" not in body or model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid=int(model.geom_dataid[i])
        verts=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        world=verts@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
        lowest.append(world[:,2].min())
    if lowest:
        data.qpos[2]+=0.001-min(lowest)
        mujoco.mj_forward(model,data)
    for i in range(model.nu):
        jid=int(model.actuator_trnid[i,0])
        data.ctrl[i]=data.qpos[model.jnt_qposadr[jid]]
    return data
