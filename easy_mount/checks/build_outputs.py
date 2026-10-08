"""Saved-CAD checks, payload inertials, MJCF integration and assembled previews."""
from pathlib import Path
import json,sys,os,copy,tempfile
import xml.etree.ElementTree as ET
import numpy as np
import trimesh,mujoco
from scipy.spatial.transform import Rotation
from cadgen import read_step,build123d as bd

ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(PARENT/"checks"))
from lib.design import parameters,print_to_assembly,jetson_tray_shape,cube,tray_location,battery_location
from lib.wiring import wiring_groups,route_metrics,route
from pose import home_pose
YUP=np.array([[1,0,0],[0,0,1],[0,-1,0]])

def vec(v):return np.array([v.X,v.Y,v.Z])
def values(v):return " ".join(f"{x:.12g}" for x in np.asarray(v).reshape(-1))
def properties(s,m,name):
    return {"name":name,"mass_kg":float(m),"com_m":(vec(s.center(bd.CenterOf.MASS))*.001).tolist(),
            "inertia_kg_m2":(np.asarray(s.matrix_of_inertia)/s.volume*m*1e-6).tolist()}
def sum_properties(parts):
    mass=sum(p["mass_kg"] for p in parts)
    com=sum(p["mass_kg"]*np.array(p["com_m"]) for p in parts)/mass
    I=np.zeros((3,3))
    for p in parts:
        d=np.array(p["com_m"])-com
        I+=p["inertia_kg_m2"]+p["mass_kg"]*((d@d)*np.eye(3)-np.outer(d,d))
    return mass,com,I
def write_xml(tree,path):
    ET.indent(tree);ET.ElementTree(tree).write(path,encoding="utf-8",xml_declaration=True)

def main():
    p=parameters();parts=[];posed={};checks={"printed_parts":[],"printed_part_overlaps_mm3":[]}
    sim=ROOT/"simulation";meshes=sim/"meshes";meshes.mkdir(exist_ok=True)
    for name in ["saddle","front_gate","jetson_tray","jetson_gate"]:
        s=read_step(ROOT/"STEP"/(name+".step"))
        assert s.is_valid and len(s.solids())==1,(name,"invalid/disconnected")
        target=ROOT/"STL"/(name+".stl")
        # Direct OCCT export avoids scene-meshing T-junction seams.
        bd.export_stl(s,str(target),tolerance=.025,angular_tolerance=.15)
        mesh=trimesh.load(target,force="mesh")
        assert mesh.is_watertight and mesh.is_winding_consistent,name
        assert abs(mesh.volume-s.volume)/s.volume<.002,name
        assert mesh.bounds[0,2]>-1e-5,name
        positioned=s.moved(print_to_assembly(name));posed[name]=positioned
        bd.export_stl(positioned,str(meshes/(name+".stl")),tolerance=.025,angular_tolerance=.15)
        parts.append(properties(positioned,s.volume*p["petg_density_g_cm3"]/1e6*p["printed_mass_scale"],name))
        checks["printed_parts"].append({"name":name,"valid_solid":True,"watertight_stl":True,"volume_mm3":s.volume,"print_bounds_mm":mesh.bounds.tolist()})
    names=list(posed)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            volume=0
            for sa in posed[a].solids():
                for sb in posed[b].solids():
                    hit=sa.intersect(sb)
                    items=list(hit) if isinstance(hit,(list,bd.ShapeList)) else ([hit] if hit else [])
                    volume+=sum(solid.volume for item in items for solid in item.solids())
            assert volume<.001,(a,b,volume)
            checks["printed_part_overlaps_mm3"].append({"parts":[a,b],"volume":float(volume)})
    # P3766 maximum vendor envelope: the original CAD is 0.377 mm longer than
    # the rounded published size. The OEM base itself remains 90.5 mm long.
    jetson=cube((103,90.876875,34.77),(0,0,4+34.77/2)).moved(tray_location())
    battery=cube(p["battery_size"],p["battery_local_center_mm"]).moved(battery_location())
    corridor=cube(p["connector_keepout_size_mm"],p["connector_keepout_center_mm"]).moved(tray_location())
    checks["connector_keepout"]={"size_tray_mm":p["connector_keepout_size_mm"],"center_tray_mm":p["connector_keepout_center_mm"],"printed_intersections_mm3":[],"scope":"Reserved 50 mm straight plug corridor ABOVE the connector edge; actual plugs and insertion motions still require physical verification."}
    for name,shape in posed.items():
        hits=[]
        for solid in shape.solids():
            hit=solid.intersect(corridor)
            hits.extend(list(hit) if isinstance(hit,(list,bd.ShapeList)) else ([hit] if hit else []))
        volume=sum(s.volume for item in hits for s in item.solids())
        assert volume<.001,(name,"connector corridor blocked",volume)
        checks["connector_keepout"]["printed_intersections_mm3"].append({"name":name,"volume":float(volume)})
    parts.append(properties(jetson,p["jetson_mass_g"]/1000,"jetson_spec_mass_uniform_envelope"))
    parts.append(properties(battery,p["battery_mass_g"]/1000,p["battery_model"]+"_specified_mass_uniform_body_including_factory_leads"))
    battery_visual=read_step(ROOT/"STEP/battery_pack.step")
    assert battery_visual.is_valid
    checks["battery"]={"model":p["battery_model"],"body_size_mm":p["battery_size"],"specified_mass_g":p["battery_mass_g"],"print_intersections_mm3":[]}
    for name,shape in posed.items():
        volume=0.
        for a in battery_visual.solids():
            for b in shape.solids():
                hit=a.intersect(b)
                items=list(hit) if isinstance(hit,(list,bd.ShapeList)) else ([hit] if hit else [])
                volume+=sum(solid.volume for item in items for solid in item.solids())
        assert volume<.001,("battery or factory lead/strap through print",name,volume)
        checks["battery"]["print_intersections_mm3"].append({"print":name,"volume":float(volume)})
    wiring=wiring_groups()
    parts.append(properties(cube((30,60,40),(-60,0,25)),p["hardware_pads_straps_mass_g"]/1000,"estimated_knobs_nuts_pads_straps"))
    for name,key in [("data","data_cable_mass_g"),("power","power_cable_mass_g"),("converter","converter_mass_g")]:
        assert wiring[name].is_valid,(name,"invalid wiring envelope")
        parts.append(properties(wiring[name],p[key]/1000,"estimated_"+name+"_wiring_envelope"))
    assert sum(p[k] for k in ["hardware_pads_straps_mass_g","data_cable_mass_g","power_cable_mass_g","converter_mass_g"])==p["ancillary_mass_g"]
    checks["wiring"]={"data":route_metrics(p["data_cable_route_trunk_mm"],(0,0,1),(0,-1,0)),"power":route_metrics(p["power_cable_route_trunk_mm"],(0,0,1),(-1,0,0)),"battery_extension":route_metrics(p["battery_extension_route_trunk_mm"],(0,-1,0),(-1,0,0)),"scope":"Approximate purchased cable envelopes in HOME. Head plug is deliberately free; production socket access unverified. No flexible-cable tension or simultaneous-motion validation.","print_intersections_mm3":[]}
    for name,wshape in wiring.items():
        for pname,pshape in posed.items():
            volume=0.
            for a in wshape.solids():
                for b in pshape.solids():
                    hit=a.intersect(b)
                    items=list(hit) if isinstance(hit,(list,bd.ShapeList)) else ([hit] if hit else [])
                    volume+=sum(s.volume for item in items for s in item.solids())
            assert volume<.001,(name,pname,"wiring through print",volume)
            checks["wiring"]["print_intersections_mm3"].append({"wiring":name,"print":pname,"volume":float(volume)})
    mass,com,I=sum_properties(parts)
    assert np.linalg.eigvalsh(I).min()>0
    asset=ET.Element("asset");body=ET.Element("body",name="easy_mount_payload")
    ET.SubElement(body,"inertial",mass=str(mass),pos=values(com),fullinertia=values([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
    for name in names:
        ET.SubElement(asset,"mesh",name="easy_"+name,file="meshes/"+name+".stl",scale="0.001 0.001 0.001")
        rgba="0.15 0.4 0.48 1" if name=="jetson_tray" else "0.91 0.65 0.2 1"
        ET.SubElement(body,"geom",name="easy_"+name+"_visual",type="mesh",mesh="easy_"+name,group="2",mass="0",contype="0",conaffinity="0",rgba=rgba)
    for name,shape in wiring.items():
        bd.export_stl(shape,str(meshes/("wiring_"+name+".stl")),tolerance=.06,angular_tolerance=.2)
        ET.SubElement(asset,"mesh",name="easy_wiring_"+name,file="meshes/wiring_"+name+".stl",scale="0.001 0.001 0.001")
        rgba={"data":"0.19 0.55 0.86 1","power":"0.26 0.68 0.46 1","converter":"0.44 0.39 0.69 1"}[name]
        ET.SubElement(body,"geom",name="easy_wiring_"+name+"_visual",type="mesh",mesh="easy_wiring_"+name,group="2",mass="0",contype="0",conaffinity="0",rgba=rgba)
    # Use the saved CAD display tessellation: native swept-lead STL meshing
    # exceeds MuJoCo's 200,000-facet STL limit at print-level tolerance.
    battery_scene=trimesh.load(ROOT/"GLB/battery_pack.glb",force="scene")
    battery_mesh=battery_scene.to_mesh()
    battery_mesh.vertices=battery_mesh.vertices@YUP*1000
    assert 0<len(battery_mesh.faces)<200000
    battery_mesh.export(meshes/"battery_pack.stl")
    ET.SubElement(asset,"mesh",name="easy_battery_pack",file="meshes/battery_pack.stl",scale="0.001 0.001 0.001")
    ET.SubElement(body,"geom",name="easy_battery_pack_visual",type="mesh",mesh="easy_battery_pack",group="2",mass="0",contype="0",conaffinity="0",rgba="0.22 0.25 0.28 1")
    boxes=[]
    def box(name,size,center):
        boxes.append({"name":name,"size_mm":list(size),"center_trunk_mm":list(center)})
        ET.SubElement(body,"geom",name="easy_"+name,type="box",size=values(np.array(size)*.0005),pos=values(np.array(center)*.001),group="3",mass="0",rgba="0.7 0.5 0.2 1")
    box("rear_beam",(4,74.2,20),(-52,0,22))
    box("front_gate",(4,84.2,12),(42,0,24))
    for sign in (-1,1):
        box("side_arm_"+str(sign),(94,4,12),(-7,sign*35.1,24))
        box("lower_capture_lip_"+str(sign),(12,7.1,1.6),(-18,sign*33.55,2))
        box("outer_clip_wall_"+str(sign),(12,4,35),(-18,sign*35.1,19.3))
        box("upper_seat_"+str(sign),(12,6,3),(-18,sign*24,43.5))
        box("rear_dock_"+str(sign),(18,12,26),(-60,sign*59,22))
    R=Rotation.from_euler("xyz",p["tray_euler_xyz_deg"],degrees=True).as_matrix()
    origin=np.array(p["tray_origin_trunk_mm"])
    def pack_box(name,size,center):
        # The 90-degree placement is axis aligned in the trunk frame.
        box(name,abs(R)@np.array(size),R@np.array(center)+origin)
    pack_box("jetson",(103,90.876875,34.77),(0,0,21.385))
    pack_box("tray_plate",(113,104,3),(0,0,1.5))
    pack_box("open_gate_bridge",(122,3,3),(0,57.5,-3.5))
    for sign in (-1,1):
        pack_box("open_gate_upright_"+str(sign),(4,3,14),(sign*59,61.5,5))
        pack_box("open_gate_backlink_"+str(sign),(4,7,3),(sign*59,59.5,-3.5))
    BR=Rotation.from_euler("xyz",p["battery_cradle_euler_xyz_deg"],degrees=True).as_matrix()
    BO=np.array(p["battery_cradle_origin_trunk_mm"])
    box("battery",abs(BR)@np.array(p["battery_size"]),BR@np.array(p["battery_local_center_mm"])+BO)
    box("front_battery_pocket",(p["battery_pocket_depth_mm"],p["battery_size"][0]+2*p["battery_clearance_per_side"]+4,p["battery_size"][1]+2*p["battery_clearance_per_side"]+4),(44+p["battery_pocket_depth_mm"]/2,0,-10))
    box("front_battery_back_panel",(4,40,52),(42,0,4))
    box("converter_pad",p["converter_pad_size_trunk_mm"],p["converter_pad_center_trunk_mm"])
    box("converter_envelope",p["converter_size_trunk_mm"],p["converter_center_trunk_mm"])
    for sign in (-1,1):pack_box("tray_rail_"+str(sign),(4.4,97,10.5),(sign*52.9,-1,8.25))
    patched=[]
    for variant in ("walk","groundcontact"):
        source=PARENT/"simulation/reference"/("robot_"+variant+".xml")
        root=ET.parse(source).getroot();compiler=root.find("compiler")
        meshdir=compiler.get("meshdir","");compiler.attrib.pop("meshdir",None)
        for mesh in root.findall(".//asset/mesh"):
            mesh.set("file",os.path.relpath((source.parent/meshdir/mesh.get("file")).resolve(),sim))
        for mesh in asset:root.find("asset").append(copy.deepcopy(mesh))
        root.find(".//body[@name='trunk_base']").append(copy.deepcopy(body))
        out=sim/("microduck_easy_mount_"+variant+".xml");write_xml(root,out);patched.append(out)
        baseline=mujoco.MjModel.from_xml_path(str(source));model=mujoco.MjModel.from_xml_path(str(out))
        assert (baseline.nq,baseline.nv,baseline.nu)==(model.nq,model.nv,model.nu)
        assert abs(model.body_mass.sum()-baseline.body_mass.sum()-mass)<1e-10
    # Thin flexible routes use local capsule probes, not a convex hull over
    # the complete loop. They are fit checks only, with no cable constraint.
    cable_root=ET.parse(patched[1]).getroot();cable_body=cable_root.find(".//body[@name='easy_mount_payload']")
    probe_names=[]
    for cname,end in [("data",(0,-1,0)),("power",(-1,0,0))]:
        spline=route(p[cname+"_cable_route_trunk_mm"],(0,0,1),end)
        points=spline(np.linspace(spline.x[0],spline.x[-1],85))
        for i,(start,finish) in enumerate(zip(points[:-1],points[1:])):
            name=f"probe_cable_{cname}_{i}";probe_names.append(name)
            ET.SubElement(cable_body,"geom",name=name,type="capsule",size=str(p[cname+"_cable_diameter_mm"]*.0005),fromto=values(np.r_[start,finish]*.001),mass="0",contype="0",conaffinity="0")
    with tempfile.NamedTemporaryFile(dir=sim,suffix=".xml") as temp:
        ET.ElementTree(cable_root).write(temp.name);cable_model=mujoco.MjModel.from_xml_path(temp.name)
    cable_data=home_pose(cable_model);payload_id=cable_model.body("easy_mount_payload").id
    robot_geoms=[g for g in range(cable_model.ngeom) if cable_model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and cable_model.geom_group[g]==2 and cable_model.geom_bodyid[g]!=payload_id]
    wire_hits=[];wire_gap=1.
    for name in probe_names:
        a=cable_model.geom(name).id
        for b in robot_geoms:
            distance=mujoco.mj_geomDistance(cable_model,cable_data,a,b,.12,None);wire_gap=min(wire_gap,distance)
            if distance<-.0002:wire_hits.append({"probe":name,"robot_mesh":cable_model.mesh(cable_model.geom_dataid[b]).name,"penetration_m":float(distance)})
    checks["wiring"].update({"HOME_centerline_capsule_robot_interferences":wire_hits,"HOME_minimum_robot_convex_gap_m":float(wire_gap),"scope_robot_fit":"HOME only, approximate tube centerline capsules versus reference robot convex meshes; free head plug, head articulation and insertion remain unverified"})
    fragment=ET.Element("mujoco",model="easy_mount_payload");fragment.append(asset)
    ET.SubElement(fragment,"worldbody").append(body);write_xml(fragment,sim/"payload.xml")
    model=mujoco.MjModel.from_xml_path(str(patched[1]));data=home_pose(model)
    pack=model.body("easy_mount_payload").id;trunk=model.body("trunk_base").id
    candidates=[i for i in range(model.ngeom) if model.geom_bodyid[i]==pack and model.geom_group[i]==3]
    # Convex distances against MOVING robot links only; intended fixed-shell
    # mating is tested separately against raw shell surfaces below.
    robot=[i for i in range(model.ngeom) if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[i]==2 and model.geom_bodyid[i] not in (pack,trunk)]
    contacts=[];closest=1.;base_qpos=data.qpos.copy()
    poses=[("HOME",base_qpos.copy())]
    for j in range(model.njnt):
        if not model.jnt_limited[j]:continue
        for q in np.linspace(*model.jnt_range[j],9):
            v=base_qpos.copy();v[model.jnt_qposadr[j]]=q
            poses.append((mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)+":"+str(round(q,4)),v))
    for label,qpos in poses:
        data.qpos[:]=qpos;mujoco.mj_forward(model,data)
        for a in candidates:
            for b in robot:
                d=mujoco.mj_geomDistance(model,data,a,b,.12,None);closest=min(closest,d)
                if d<-.0002:
                    contacts.append({"pose":label,"mount":model.geom(a).name,"robot_mesh":model.mesh(model.geom_dataid[b]).name,"penetration_m":float(d)})
    checks["sampled_joint_sweep"]={"poses":len(poses),"minimum_convex_distance_m":float(closest),"interferences":contacts,"scope":"HOME plus nine samples per limited joint, one joint at a time; not combined-motion or load validation"}
    nearby=[];near_contacts=[];near_closest=1.
    for j in range(model.njnt):
        if not model.jnt_limited[j]:continue
        adr=model.jnt_qposadr[j];q0=base_qpos[adr]
        low=max(model.jnt_range[j,0],q0-.2);high=min(model.jnt_range[j,1],q0+.2)
        for q in np.linspace(low,high,9):
            v=base_qpos.copy();v[adr]=q
            nearby.append((mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)+":"+str(round(q,4)),v))
    for label,qpos in nearby:
        data.qpos[:]=qpos;mujoco.mj_forward(model,data)
        for a in candidates:
            for b in robot:
                d=mujoco.mj_geomDistance(model,data,a,b,.12,None);near_closest=min(near_closest,d)
                if d<-.0002:
                    near_contacts.append({"pose":label,"mount":model.geom(a).name,"robot_mesh":model.mesh(model.geom_dataid[b]).name,"penetration_m":float(d)})
    checks["near_home_joint_sweep"]={"poses":len(nearby),"offset_limit_rad":.2,"minimum_convex_distance_m":float(near_closest),"interferences":near_contacts,"scope":"Nine samples per joint within HOME +/- 0.2 radians, one joint at a time; not a verified operating envelope or gait test"}
    # The reserved plug corridor must also clear the duck, not only the prints.
    probe_root=ET.parse(patched[1]).getroot()
    probe_body=probe_root.find(".//body[@name='easy_mount_payload']")
    ET.SubElement(probe_body,"geom",name="connector_access_probe",type="box",size=values(abs(R)@np.array(p["connector_keepout_size_mm"])*.0005),pos=values((R@np.array(p["connector_keepout_center_mm"])+origin)*.001),mass="0",contype="0",conaffinity="0")
    with tempfile.NamedTemporaryFile(dir=sim,suffix=".xml") as temp:
        ET.ElementTree(probe_root).write(temp.name)
        probe_model=mujoco.MjModel.from_xml_path(temp.name)
    probe_data=mujoco.MjData(probe_model);a=probe_model.geom("connector_access_probe").id
    payload_id=probe_model.body("easy_mount_payload").id
    robot_geoms=[g for g in range(probe_model.ngeom) if probe_model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and probe_model.geom_group[g]==2 and probe_model.geom_bodyid[g]!=payload_id]
    access_hits=[];access_gap=1.
    for label,qpos in [("HOME",base_qpos)]+nearby:
        probe_data.qpos[:]=qpos;mujoco.mj_forward(probe_model,probe_data)
        for b in robot_geoms:
            distance=mujoco.mj_geomDistance(probe_model,probe_data,a,b,.12,None);access_gap=min(access_gap,distance)
            if distance<-.0002:access_hits.append({"pose":label,"robot_mesh":probe_model.mesh(probe_model.geom_dataid[b]).name,"penetration_m":float(distance)})
    assert not access_hits,("robot obstructs connector corridor",access_hits[:3])
    checks["connector_keepout"].update({"robot_poses_checked":len(nearby)+1,"robot_interferences":access_hits,"minimum_robot_convex_gap_m":float(access_gap)})
    (sim/"validation.json").write_text(json.dumps(checks,indent=2)+"\n")
    (sim/"mass_properties.json").write_text(json.dumps({"mass_kg":mass,"com_trunk_m":com.tolist(),"inertia_trunk_kg_m2":I.tolist(),"parts":parts,"collision_boxes":boxes,"assumptions":["Fully dense PETG scaled by printed_mass_scale","Jetson and GNB8504S60AHV battery use uniform specified-mass envelopes; battery 73g includes stock leads at body COM, +/-2g manufacturer tolerance","74 g ancillary allowance split into 22 g hardware, 14 g data cable, 18 g power wiring and 20 g conditioning/cutoff allowance","Cable/plug mass uses HOME geometry fixed to trunk; head motion and flexible-cable tension are not modeled","Rigid fit to reference robot; physical shell strength and production fit unverified"]},indent=2)+"\n")
    print(json.dumps({"payload_g":round(mass*1000,1),"plastic_g":round(sum(x["mass_kg"] for x in parts[:4])*1000,1),"sweep_interferences":len(contacts),"near_home_interferences":len(near_contacts)}))
    make_previews(p,R)
    from balance_estimate import main as balance_estimate
    balance_estimate()

def make_previews(p,R):
    model=mujoco.MjModel.from_xml_path(str(PARENT/"simulation/reference/robot_groundcontact.xml"));data=home_pose(model)
    trunk=model.body("trunk_base").id;worldR=data.xmat[trunk].reshape(3,3);worldorigin=data.xpos[trunk]
    scene=trimesh.Scene()
    for i in range(model.ngeom):
        if model.geom_group[i]!=2 or model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        k=model.geom_dataid[i];mesh=trimesh.Trimesh(vertices=model.mesh_vert[model.mesh_vertadr[k]:model.mesh_vertadr[k]+model.mesh_vertnum[k]].copy(),faces=model.mesh_face[model.mesh_faceadr[k]:model.mesh_faceadr[k]+model.mesh_facenum[k]].copy(),process=False)
        mesh.vertices=(mesh.vertices@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i])@YUP.T
        color=model.mat_rgba[model.geom_matid[i]] if model.geom_matid[i]>=0 else model.geom_rgba[i]
        mesh.visual=trimesh.visual.ColorVisuals(mesh=mesh,face_colors=np.uint8(color*255));scene.add_geometry(mesh,node_name="microduck_"+model.mesh(k).name)
    def add_local(mesh,name):
        mesh.vertices=(mesh.vertices@worldR.T+worldorigin)@YUP.T;scene.add_geometry(mesh,node_name=name)
    mount=trimesh.load(ROOT/"GLB/easy_mount_assembly.glb",force="scene")
    for node in mount.graph.nodes_geometry:
        T,key=mount.graph[node];mesh=mount.geometry[key].copy();mesh.apply_transform(T)
        mesh.vertices=mesh.vertices@YUP;add_local(mesh,"mount_"+str(node))
    vendor=trimesh.load(PARENT/"GLB/jetson_devkit.glb",force="scene")
    bench=trimesh.Scene(mount)
    for node in vendor.graph.nodes_geometry:
        T,key=vendor.graph[node];mesh=vendor.geometry[key].copy();mesh.apply_transform(T)
        pack=mesh.vertices@YUP-np.array([0,.018,.0044])
        pack=pack@Rotation.from_euler("z",p["jetson_inplane_rotation_deg"],degrees=True).as_matrix().T
        mesh.vertices=pack@R.T+np.array(p["tray_origin_trunk_mm"])*.001
        benchmesh=mesh.copy();benchmesh.vertices=benchmesh.vertices@YUP.T
        bench.add_geometry(benchmesh,node_name="Jetson_"+str(node));add_local(mesh,"Jetson_"+str(node))
    for name,data in [("microduck_easy_mount",scene.export(file_type="glb")),("equipped_mount",bench.export(file_type="glb"))]:
        (ROOT/"GLB"/(name+".glb")).write_bytes(data)
        (ROOT/"GLB"/(name+"_"+p["revision"]+".glb")).write_bytes(data)
    inspection=trimesh.Scene()
    for node in scene.graph.nodes_geometry:
        if "top_head_shell" in node:continue
        T,key=scene.graph[node];mesh=scene.geometry[key].copy();mesh.apply_transform(T)
        if "pcb__raspberry_pi_zero" in node:
            mesh.visual=trimesh.visual.ColorVisuals(mesh=mesh,face_colors=[60,220,100,255])
            node="REFERENCE_old_compute_PCB_footprint_NOT_production_port_geometry"
        inspection.add_geometry(mesh,node_name=node)
    (ROOT/"GLB"/("head_wiring_inspection_"+p["revision"]+".glb")).write_bytes(inspection.export(file_type="glb"))

if __name__=="__main__":main()
