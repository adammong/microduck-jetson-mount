from __future__ import annotations
from pathlib import Path
import json
import math
from cadgen import build123d as bd, srgb

ROOT=Path(__file__).resolve().parents[2]

def parameters(): return json.loads((ROOT/"parameters.json").read_text())
def cube(size,center):return bd.Pos(*center)*bd.Box(*size)
def cylinder(radius,length,center,axis="Z"):
    rot={"X":(0,90,0),"Y":(90,0,0),"Z":(0,0,0)}[axis]
    return bd.Pos(*center)*bd.Rot(*rot)*bd.Cylinder(radius,length)
def hex_pocket(af,length,center,axis="Z"):
    rot={"X":(0,90,0),"Y":(90,0,0),"Z":(0,0,0)}[axis]
    poly=bd.RegularPolygon(af/math.sqrt(3),6)
    return bd.Pos(*center)*bd.Rot(*rot)*bd.extrude(poly,amount=length,both=True)
def color(shape,label,hexcolor):
    shape.label=label;shape.color=srgb(hexcolor);return shape.clean()

def saddle_shape():
    p=parameters();inner=p["torso_side_inner_y"];outer=inner+p["torso_arm_wall"]
    part=cube((4,2*outer,20),(-52,0,22))
    # The two C-shaped clips capture rounded shell shoulders and the underside.
    # Inner upper coordinates bound the official mesh over X=-24..-12 mm,
    # plus 1.2 mm clearance; 1 mm foam occupies most of that clearance.
    roof=[(22,42.22),(24,42.03),(27,41.46),(28,40.99),(29,40.30),(30,39.31),(31,37.77)]
    outline=roof+[(inner,34.5),(inner,2.8),(30,2.8),(30,1.2),
                  (outer,1.2),(outer,37.5),(31,40.77),(30,42.31),
                  (29,43.30),(28,43.99),(27,44.46),(24,45.03),(22,45.22)]
    for sign in (-1,1):
        profile=bd.Plane.YZ*bd.Polygon(*[(sign*y,z) for y,z in outline],align=None)
        clip=bd.extrude(profile,amount=12,dir=(1,0,0))
        part+=bd.Pos(-24,0,0)*clip
        part+=cube((94,4,12),(-7,sign*(inner+2),24))
        # Captive nuts installed once on the bench; standard screws remain steel.
        y=sign*p["gate_screw_y"]
        part+=cylinder(6,12,(34,y,24),"X")
        part-=cylinder(2.25,16,(34,y,24),"X")
        part-=hex_pocket(p["m4_nut_pocket_af"],p["m4_nut_pocket_depth"]/2,(29.75,y,24),"X")
        # Outriggers clear the stock battery and support the tray independently
        # of the two retention screws.
        dy=sign*p["dock_x"]
        part+=cube((4,26,8),(-52,sign*46,24))
        part+=cube((14,12,22),(-58,dy,24))
        part+=cube((8,8,4),(-65,sign*61,11))  # support ears, clear central plate
        part+=cylinder(p["dock_pin_diameter"]/2,4,(-67,dy,18),"X")
        part-=cylinder(2.25,17,(-58,dy,25),"X")
        part-=hex_pocket(p["m4_nut_pocket_af"],p["m4_nut_pocket_depth"]/2,(-52.75,dy,25),"X")
    # Small cable-tie points hold cables, not the robot's weight.
    for y in (-15,15):part-=cube((6,8,3),(-52,y,20))
    return color(part,"external_captive_torso_saddle","#E8AA39")

def front_gate_shape():
    p=parameters()
    part=cube((4,70.2,12),(42,0,24))
    for sign in (-1,1):
        y=sign*p["gate_screw_y"]
        part+=cylinder(7,4,(42,y,24),"X")
        part-=cylinder(2.25,6,(42,y,24),"X")
        part+=cube((3.5,14,12),(38.25,sign*15,24))
    # Carry the battery forward, below the front gate's accessible thumb knobs.
    # The back panel transfers its weight to the gate and the two side screws.
    pocket=battery_pocket().moved(battery_location())
    bottom=p["battery_cradle_origin_trunk_mm"][2]-(p["battery_size"][1]+2*p["battery_clearance_per_side"]+4)/2
    part+=cube((4,40,30-bottom),(42,0,(30+bottom)/2))
    part+=pocket
    # Strap-mounted converter allowance, beside rather than over the battery.
    part+=cube(p["converter_pad_size_trunk_mm"],p["converter_pad_center_trunk_mm"])
    for y in p["converter_strap_slot_y_mm"]:part-=cube((6,3,10),(42,y,-10))
    return color(part,"front_gate_with_forward_battery_cradle","#E8AA39")

def battery_pocket():
    p=parameters();gap=p["battery_clearance_per_side"]
    iw,ih=[v+2*gap for v in p["battery_size"][:2]];ow,oh=iw+4,ih+4
    depth=p["battery_pocket_depth_mm"]
    shape=bd.extrude(bd.RectangleRounded(ow,oh,3),amount=depth)
    shape-=bd.Pos(0,0,2)*bd.extrude(bd.RectangleRounded(iw,ih,1),amount=depth)
    for x in (-ow/2,ow/2):shape-=cube((6,12,3.2),(x,0,5))
    shape-=cube((10,6,9),(0,oh/2,depth-3.5))
    # Open short-edge exit for factory power and balance leads; no pinching.
    shape-=cube((6,12,12),(-ow/2,0,depth-4))
    return shape

def jetson_tray_shape():
    p=parameters();t=p["tray_thickness"]
    part=bd.extrude(bd.RectangleRounded(113,104,4),amount=t)
    # Slots are only under the OEM base: rails carry and capture that base.
    for x in (-28,0,28):
        for y in (-28,0,28):
            part-=bd.Pos(x,y,-1)*bd.extrude(bd.SlotOverall(20,12),amount=t+2)
    for sign in (-1,1):
        # 104.2 mm internal width, 0.8 mm lips overlap the plastic base edges.
        part+=cube((3,97,10.5),(sign*53.6,-1,8.25))
        part+=cube((4.4,97,2.5),(sign*52.9,-1,12.25))
        x=sign*p["dock_x"]
        part+=cube((14,24,t),(x,0,t/2))
        part-=cylinder(2.25,t+2,(x,0,t/2))
        part-=cylinder(2.8,t+2,(x,-7,t/2))
        # End-gate screws are outboard of both the base and carrier connectors.
        part+=cube((12,12,8),(sign*59,54,4))
        part-=cylinder(2.25,10,(sign*59,54,4))
        part-=hex_pocket(p["m4_nut_pocket_af"],1.75,(sign*59,54,1.75))
        # Outboard cable-tie eyes leave the connector corridor open.
        part+=cube((8,12,4),(sign*67,54,6))
        part-=cube((4,7,6),(sign*67,54,6))
    # Corner tabs support the OEM base; the connector edge remains open.
    # Each tab captures 1.7 mm of the base width outside the cable corridor.
    for sign in (-1,1):part+=cube((4,3,4.5),(sign*51.8,-47.3,5.25))
    return color(part,"slide_in_OEM_base_tray_with_open_connector_edge","#2E6477")

def jetson_gate_shape():
    # One-piece U-shaped end gate: bridge behind the base/plate, open port edge.
    part=cube((122,3,3),(0,57.5,-3.5))
    for sign in (-1,1):
        part+=cube((4,7,3),(sign*59,59.5,-3.5))
        part+=cube((4,3,14),(sign*59,61.5,5))
        part+=cube((12,5,4),(sign*59,60,10))
        part+=cube((2.1,3,7.5),(sign*50.85,47.3,6.75))
        part+=cube((6,4,4),(sign*53.5,50,10))
        part+=cube((12,12,4),(sign*59,54,10))
        part-=cylinder(2.25,6,(sign*59,54,10))
    return color(part,"removable_open_connector_edge_Jetson_gate","#E8AA39")

def tray_location():
    p=parameters();rx,ry,rz=p["tray_euler_xyz_deg"]
    # Explicit extrinsic XYZ, matching scipy and the documented pack->trunk
    # map (X,Y,Z)->(-Z,-X,Y). build123d's three-angle Location is intrinsic.
    return bd.Location(p["tray_origin_trunk_mm"])*bd.Location((0,0,0),(0,0,rz))*bd.Location((0,0,0),(0,ry,0))*bd.Location((0,0,0),(rx,0,0))

def battery_location():
    p=parameters();rx,ry,rz=p["battery_cradle_euler_xyz_deg"]
    return bd.Location(p["battery_cradle_origin_trunk_mm"])*bd.Location((0,0,0),(0,0,rz))*bd.Location((0,0,0),(0,ry,0))*bd.Location((0,0,0),(rx,0,0))

def manufacturing(shape,name):
    # Each print lies with its broadest usable face on the build plate.
    if name=="front_gate":shape=shape.rotate(bd.Axis.Y,-90)
    bounds=shape.bounding_box()
    return shape.moved(bd.Location((-bounds.center().X,-bounds.center().Y,-bounds.min.Z)))

def print_to_assembly(name):
    factory={"saddle":saddle_shape,"front_gate":front_gate_shape,"jetson_tray":jetson_tray_shape,"jetson_gate":jetson_gate_shape}[name]
    native=factory();rotated=native.rotate(bd.Axis.Y,-90) if name=="front_gate" else native
    b=rotated.bounding_box();restore=bd.Location((b.center().X,b.center().Y,b.min.Z))
    if name=="front_gate":restore=bd.Location((0,0,0),(0,90,0))*restore
    if name in ("jetson_tray","jetson_gate"):restore=tray_location()*restore
    return restore
