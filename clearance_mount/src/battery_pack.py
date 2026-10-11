from cadgen import step,glb
from lib.battery import battery_shape

@step(out="../STEP/battery_pack.step")
@glb(out="../GLB/battery_pack.glb",mesh_tolerance=0.0005)
def battery_pack():
    return battery_shape()

if __name__=="__main__":battery_pack()
