from cadgen import step, stl, glb
from lib.design import front_gate_shape, manufacturing

@step(out="../STEP/front_gate.step")
@stl(out="../STL/front_gate.stl",mesh_tolerance=0.0005)
@glb(out="../GLB/front_gate.glb",mesh_tolerance=0.0005)
def front_gate():
    return manufacturing(front_gate_shape(),"front_gate")

if __name__=="__main__":front_gate()
