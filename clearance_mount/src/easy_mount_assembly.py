from cadgen import build123d as bd, step, glb, srgb
from saddle import saddle
from front_gate import front_gate
from jetson_tray import jetson_tray
from jetson_gate_left import jetson_gate_left
from jetson_gate_right import jetson_gate_right
from wiring_layout import wiring_layout
from battery_pack import battery_pack
from lib.design import print_to_assembly, cylinder, parameters, tray_location

@step(out="../STEP/easy_mount_assembly.step")
@glb(out="../GLB/easy_mount_assembly.glb",mesh_tolerance=0.0005)
def easy_mount_assembly():
    parts=[wiring_layout(),battery_pack()]
    for name,model in [("saddle",saddle),("front_gate",front_gate),("jetson_tray",jetson_tray),("jetson_gate_left",jetson_gate_left),("jetson_gate_right",jetson_gate_right)]:
        parts.append(model().moved(print_to_assembly(name)))
    # Simplified hand-knob/screw outlines: M4 screws are purchased, not printed.
    # Threads, washers and product-specific knobs are omitted in the preview.
    p=parameters()
    for sign in (-1,1):
        y=sign*p["gate_screw_y"]
        screw=cylinder(2,20,(34,y,24),"X")+cylinder(7,5,(46.5,y,24),"X")
        screw.label="M4x20_torso_thumb_screw";screw.color=srgb("#888C90");parts.append(screw)
        y=sign*p["dock_x"]
        tx,ty,tz=p["tray_origin_trunk_mm"]
        screw=cylinder(2,20,(tx-7,y,tz),"X")+cylinder(7,5,(tx+6.5,y,tz),"X")
        screw.label="M4x20_tray_thumb_screw";screw.color=srgb("#888C90");parts.append(screw)
        x=sign*p["dock_x"]
        screw=cylinder(2,12,(x,54,6))+cylinder(7,5,(x,54,14.5))
        screw.label="M4x12_Jetson_gate_thumb_screw";screw.color=srgb("#888C90");parts.append(screw.moved(tray_location()))
    return bd.Compound(children=parts,label="external_captive_mount_with_removable_Jetson_tray")

if __name__=="__main__":easy_mount_assembly()
