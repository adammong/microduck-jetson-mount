from cadgen import step, stl, glb
from lib.design import jetson_gate_shape, manufacturing

@step(out="../STEP/jetson_gate_right.step")
@stl(out="../STL/jetson_gate_right.stl",mesh_tolerance=0.0005)
@glb(out="../GLB/jetson_gate_right.glb",mesh_tolerance=0.0005)
def jetson_gate_right():
    return manufacturing(jetson_gate_shape(1),"jetson_gate_right")

if __name__=="__main__":jetson_gate_right()
