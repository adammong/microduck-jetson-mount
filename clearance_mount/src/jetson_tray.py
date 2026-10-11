from cadgen import step, stl, glb
from lib.design import jetson_tray_shape, manufacturing

@step(out="../STEP/jetson_tray.step")
@stl(out="../STL/jetson_tray.stl",mesh_tolerance=0.0005)
@glb(out="../GLB/jetson_tray.glb",mesh_tolerance=0.0005)
def jetson_tray():
    return manufacturing(jetson_tray_shape(),"jetson_tray")

if __name__=="__main__":jetson_tray()
