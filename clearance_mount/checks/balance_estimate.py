"""Static COM projection against the reference HOME sole footprint."""
from pathlib import Path
import json,sys
import numpy as np,mujoco
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent
sys.path.insert(0,str(PARENT/"checks"))
from pose import home_pose

def main():
    out={}
    for label,path in [("stock",PARENT/"simulation/reference/robot_groundcontact.xml"),("loaded",ROOT/"simulation/microduck_easy_mount_groundcontact.xml")]:
        model=mujoco.MjModel.from_xml_path(str(path));data=home_pose(model)
        mass=model.body_mass.sum();com=(model.body_mass[:,None]*data.xipos).sum(axis=0)/mass
        feet=[]
        for name in ("left_foot_collision","right_foot_collision"):
            g=model.geom(name).id;k=model.geom_dataid[g]
            vertices=model.mesh_vert[model.mesh_vertadr[k]:model.mesh_vertadr[k]+model.mesh_vertnum[k]]
            feet.append(vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
        vertices=np.vstack(feet);hull=ConvexHull(vertices[:,:2]);eq=hull.equations
        margin=-(eq[:,:2]@com[:2]+eq[:,2]).max()
        out[label]={"mass_kg":float(mass),"com_world_m":com.tolist(),"footprint_x_limits_m":[float(vertices[:,0].min()),float(vertices[:,0].max())],"rear_static_margin_mm":float((com[0]-vertices[:,0].min())*1000),"projected_sole_hull_margin_mm":float(margin*1000),"sole_z_limits_m":[[float(v[:,2].min()),float(v[:,2].max())] for v in feet]}
    out["com_shift_world_mm"]=((np.array(out["loaded"]["com_world_m"])-np.array(out["stock"]["com_world_m"]))*1000).tolist()
    out["scope"]="Static COM for reference HOME. The hull uses projected entire sole meshes and assumes full usable sole contact, so margin is optimistic. Not a policy rollout, gait test, measured hardware result or payload rating."
    (ROOT/"simulation/balance_estimate.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"com_shift_mm":out["com_shift_world_mm"],"rear_margin_mm":out["loaded"]["rear_static_margin_mm"]}))

if __name__=="__main__":main()
