"""Sample saved print solids along their intended shell-attachment paths."""
from pathlib import Path
import json,sys
import numpy as np,mujoco
from cadgen import read_step
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent
sys.path.insert(0,str(PARENT/'checks'));sys.path.insert(0,str(ROOT/'src'))
from pose import home_pose
from lib.design import print_to_assembly

def main():
    m=mujoco.MjModel.from_xml_path(str(PARENT/'simulation/reference/robot_groundcontact.xml'));d=home_pose(m);t=m.body('trunk_base').id;R=d.xmat[t].reshape(3,3);origin=d.xpos[t];points=[]
    for g in range(m.ngeom):
        if m.geom_bodyid[g]!=t or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        k=m.geom_dataid[g]
        if m.mesh(k).name not in ['right_shell','left_shell','trunk_base']:continue
        v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]];v=((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-origin)@R)*1000
        f=m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]];points.extend(v);points.extend(v[f].mean(axis=1))
    points=np.unique(np.round(points,5),axis=0);assert len(points)>1000;rows=[]
    for name,steps in [('saddle',np.linspace(-80,0,17)),('front_gate',np.linspace(60,0,13))]:
        shape=read_step(ROOT/'STEP'/(name+'.step')).moved(print_to_assembly(name));bounds=shape.bounding_box();lo=np.array([bounds.min.X,bounds.min.Y,bounds.min.Z]);hi=np.array([bounds.max.X,bounds.max.Y,bounds.max.Z]);hits=[];tested=0;classifier=BRepClass3d_SolidClassifier(shape.wrapped)
        for dx in steps:
            local=points-np.array([dx,0,0]);subset=local[np.all((local>=lo)&(local<=hi),axis=1)]
            for p in subset:
                tested+=1
                classifier.Perform(gp_Pnt(*p),.001)
                if classifier.State()==TopAbs_IN or classifier.IsOnAFace():hits.append(dict(shift_x_mm=float(dx),point_local_mm=p.tolist()))
        assert not hits,(name,hits[:5]);rows.append(dict(print=name,steps=len(steps),shift_x_limits_mm=[float(steps[0]),float(steps[-1])],shell_points_tested=tested,interferences=hits))
    out=dict(passed=True,units='mm',reference_shell_sample_points=len(points),paths=rows,scope='5mm translation samples against pinned shell vertices/face centers, saved CAD solids. Gate and screws removed for saddle installation. Does not prove all triangle interiors, tolerance/foam compression, grip force, production fit or physical assembly.')
    (ROOT/'simulation/attachment_check.json').write_text(json.dumps(out,indent=2)+'\n');print('Attachment path samples passed',flush=True)
if __name__=='__main__':main()
