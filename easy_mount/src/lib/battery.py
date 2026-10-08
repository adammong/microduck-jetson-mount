"""GNB8504S60AHV specified body; approximate factory leads and strap outlines."""
from cadgen import build123d as bd
from .design import parameters,cube,color,battery_location
from .wiring import tube

def battery_shape():
    p=parameters();loc=battery_location()
    body=color(cube(p["battery_size"],p["battery_local_center_mm"]).moved(loc),"GNB8504S60AHV_73x18x32mm_73g_spec_envelope","#30383E")
    label=color(cube((55,14,.15),(0,0,34.675)).moved(loc),"GNB_4S_850mAh_label_illustration","#EE762C")
    pad=color(cube((73,18,.6),(0,0,2.3)).moved(loc),"nonconductive_bottom_pad_0_6mm","#65717C")
    # Strap runs between the pack ends and pocket walls, exits both slots.
    # Loose fastening tails/buckle are omitted from this purchased-part outline.
    strap=cube((75,8,1),(0,0,35.5))
    for sign in (-1,1):
        strap+=cube((1,8,30.5),(sign*37,0,20.25))
        strap+=cube((5,8,1),(sign*39,0,5))
    strap=color(strap.moved(loc),"purchased_retaining_strap_illustrative","#656A70")
    items=[body,label,pad,strap]
    # Relocate the original illustrative factory leads into the rear-facing frame.
    bx,by,bz=p["battery_cradle_origin_trunk_mm"]
    leads_loc=bd.Location((bx+44,by,bz+10))*bd.Location((0,0,0),(0,0,180))
    discharge=[[75,-36.5,-9],[76,-47,-9],[77,-58,-11],[75,-69.5,-13]]
    for dz,c in [(-1.5,"#BF3B34"),(1.5,"#25282B")]:
        points=[[x,y,z+dz] for x,y,z in discharge]
        items.append(color(tube(points,.95,(0,-1,0),(0,-1,0)).moved(leads_loc),"approx_factory_discharge_lead",c))
    items.append(color(cube((8,11,7),(75,-75,-13)).moved(leads_loc),"XT30_factory_plug_approximate","#F0BE37"))
    balance=[[73,-36.5,-13],[77,-43,-14],[84,-44,-15]]
    items.append(color(tube(balance,1.2,(0,-1,0),(1,0,0)).moved(leads_loc),"approx_factory_balance_bundle","#D8DADF"))
    items.append(color(cube((6,14,5),(87,-44,-15)).moved(leads_loc),"JST_XH_5pin_approximate_verify_pack","#E6E9E9"))
    return bd.Compound(children=items,label="purchased_GNB_4S_850mAh_pack_and_retention_envelopes")
