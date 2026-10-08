from cadgen import step,glb
from lib.wiring import wiring_shape

@step(out="../STEP/wiring_layout.step")
@glb(out="../GLB/wiring_layout.glb",mesh_tolerance=0.0005)
def wiring_layout():
    return wiring_shape()

if __name__=="__main__":wiring_layout()
