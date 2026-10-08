from cadgen import step, stl, glb
from lib.design import jetson_gate_shape, manufacturing

@step(out="../STEP/jetson_gate.step")
@stl(out="../STL/jetson_gate.stl",mesh_tolerance=0.0005)
@glb(out="../GLB/jetson_gate.glb",mesh_tolerance=0.0005)
def jetson_gate():
    return manufacturing(jetson_gate_shape(),"jetson_gate")

if __name__=="__main__":jetson_gate()
