"""Sample front-shell cross sections against the new underside capture tongues."""
from pathlib import Path
import json,sys
import numpy as np,mujoco,trimesh
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent
sys.path.insert(0,str(PARENT/'checks'))
from pose import home_pose

def main():
    p=json.loads((ROOT/'parameters.json').read_text());m=mujoco.MjModel.from_xml_path(str(PARENT/'simulation/reference/robot_groundcontact.xml'));d=home_pose(m)
    t=m.body('trunk_base').id;R=d.xmat[t].reshape(3,3);origin=d.xpos[t];rows=[]
    inner=p['front_capture_inner_y_mm'];outer=p['front_capture_outer_y_mm'];top=p['front_capture_lip_top_z_mm'];thick=p['front_capture_lip_thickness_mm']
    for side,shell in [(-1,'right_shell'),(1,'left_shell')]:
        g=next(i for i in range(m.ngeom) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[i]).name==shell)
        k=m.geom_dataid[g];vertices=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]]
        vertices=((vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-origin)@R)*1000
        faces=m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]];mesh=trimesh.Trimesh(vertices,faces,process=False);samples=[]
        for x in np.linspace(24,34,21):
            lines=trimesh.intersections.mesh_plane(mesh,[1,0,0],[x,0,0])
            for a,b in lines:samples.extend(a[None]+np.linspace(0,1,11)[:,None]*(b-a)[None])
        pts=np.array(samples);assert len(pts)>100
        center=np.array([30,side*(inner+outer)/2,top-thick/2]);half=np.array([6,(outer-inner)/2,thick/2])
        distance=np.linalg.norm(np.maximum(abs(pts-center)-half,0),axis=1)
        minimum=float(distance.min());assert minimum>=.7,(shell,minimum)
        under=pts[(abs(pts[:,0]-30)<.01)&(pts[:,2]<27)&(abs(pts[:,1])>=inner)&(abs(pts[:,1])<=outer)]
        assert len(under)>0,(shell,'tongue has no shell surface above it')
        rows.append(dict(shell=shell,sampled_points=len(pts),minimum_tongue_shell_gap_mm=minimum,front_lobe_capture_overlap_y_mm=float(abs(under[:,1]).max()-inner)))
    out=dict(passed=True,units='mm',required_gap_mm=.7,shells=rows,scope='21 X cross sections of pinned shell surface, 11 samples per edge. Reference fit only; no shell strength, tolerance, foam compression or physical verification.')
    (ROOT/'simulation/shell_capture_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
