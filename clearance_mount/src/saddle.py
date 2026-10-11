from cadgen import step, stl, glb
from lib.design import saddle_shape, manufacturing

@step(out="../STEP/saddle.step")
@stl(out="../STL/saddle.stl",mesh_tolerance=0.0005)
@glb(out="../GLB/saddle.glb",mesh_tolerance=0.0005)
def saddle():
    return manufacturing(saddle_shape(),"saddle")

if __name__=="__main__":saddle()
