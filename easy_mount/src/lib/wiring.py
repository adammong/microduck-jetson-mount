"""Fit envelopes for purchased wiring, in the robot trunk frame, millimeters.

The free head-side plug is intentionally NOT connected to an invented socket.
These are approximate cable/plug outlines, not electrically selected hardware.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from cadgen import build123d as bd
from .design import parameters,cube,cylinder,color

def route(points,start,end):
    points=np.asarray(points,float)
    stations=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    return CubicSpline(stations,points,bc_type=((1,np.asarray(start,float)),(1,np.asarray(end,float))))

def tube(points,radius,start,end):
    spline=route(points,start,end)
    t=np.linspace(spline.x[0],spline.x[-1],100)
    path=bd.Edge.make_spline([tuple(p) for p in spline(t)],tangents=[tuple(spline(t[0],1)),tuple(spline(t[-1],1))])
    section=bd.Plane(origin=tuple(spline(t[0])),z_dir=tuple(spline(t[0],1)))*bd.Circle(radius)
    # Circular pipes can fail with one OCCT sweep frame at spline inflections.
    try:
        result=bd.sweep(section,path=path)
        if result.is_valid:return result
    except Exception:
        pass
    result=bd.sweep(section,path=path,is_frenet=True)
    if not result.is_valid:raise ValueError("Invalid routed cable sweep")
    return result

def route_metrics(points,start,end):
    spline=route(points,start,end);t=np.linspace(spline.x[0],spline.x[-1],2001)
    d=spline(t,1);dd=spline(t,2)
    curvature=np.linalg.norm(np.cross(d,dd),axis=1)/np.linalg.norm(d,axis=1)**3
    return {"centerline_length_mm":float(np.sum(np.linalg.norm(np.diff(spline(t),axis=0),axis=1))),
            "minimum_sampled_bend_radius_mm":float(1/curvature.max()),
            "sample_count":len(t)}

def wiring_groups():
    p=parameters()
    R=Rotation.from_euler("xyz",p["tray_euler_xyz_deg"],degrees=True).as_matrix()
    origin=np.array(p["tray_origin_trunk_mm"])
    usb=R@np.array(p["jetson_usb_c_face_tray_mm"])+origin
    dc=R@np.array(p["jetson_dc_face_tray_mm"])+origin
    def at(face,dz):return tuple(face+np.array([0,0,dz]))
    # NVIDIA J5 USB-C and J16 DC positions are derived from its vendor STEP.
    # Mouth planes and overmolds remain estimates until a cable is selected.
    data=[tube(p["data_cable_route_trunk_mm"],p["data_cable_diameter_mm"]/2,(0,0,1),(0,1,0)),
          cube((7,12,21),at(usb,11.05)),
          cube((2.6,8.3,7),at(usb,-1.95)),
          cylinder(2.25,8,at(usb,25.55)),
          cube((12,18,7),p["head_usb_plug_body_center_trunk_mm"]),
          cube((8.3,8,2.6),(50,-53,120))]
    power=[tube(p["power_cable_route_trunk_mm"],p["power_cable_diameter_mm"]/2,(0,0,1),(1,0,0)),
           cylinder(4.5,22,at(dc,11.9)),
           cylinder(2.75,7,at(dc,-1.6)),
           cylinder(1.5,3,at(dc,24.4))]
    # Custom XT30 extension mates with the stock battery plug; stock leads
    # are counted in the specified battery mass, not this cable allowance.
    power.append(tube(p["battery_extension_route_trunk_mm"],1.5,(0,1,0),(1,0,0)))
    power.append(cube((8,11,7),(p["battery_cradle_origin_trunk_mm"][0]-31,86,p["battery_cradle_origin_trunk_mm"][2]-3)))
    return {"data":color(bd.Compound(children=data),"USB_C_data_to_FREE_unverified_head_endpoint","#318CDC"),
            "power":color(bd.Compound(children=power),"battery_converter_J16_power_wiring_ENVELOPE","#43AD76"),
            "converter":color(cube(p["converter_size_trunk_mm"],p["converter_center_trunk_mm"]),"unselected_regulator_fit_allowance","#7064AF")}

def wiring_shape():
    groups=wiring_groups()
    # Amber marker means survey this area, not a socket or a shell cutout.
    marker=cylinder(4,1,p:=parameters()["head_usb_access_marker_trunk_mm"],axis="Y")-cylinder(2.8,2,p,axis="Y")
    marker=color(marker,"UNVERIFIED_head_access_survey_marker_NOT_A_PORT","#E8AA39")
    return bd.Compound(children=[*groups.values(),marker],label="purchased_wiring_fit_envelopes")
