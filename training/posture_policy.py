"""Mounted posture adaptation; observable IMU feedback, frozen neural actors."""
from pathlib import Path
import numpy as np
from evaluate_motion import session

STAND=Path(__file__).resolve().parents[1]/'tmp/sim-inputs/alpha_stand.onnx'
SIT=STAND.with_name('alpha_sitstand.onnx')

def pose_bias(values):
    hip,knee,ankle,roll,neck,head=values
    out=np.zeros(14,np.float32)
    out[[2,11]]=[hip,-hip];out[[3,12]]=[knee,-knee];out[[4,13]]=[ankle,-ankle]
    out[[1,10]]=[-roll,roll];out[5]=neck;out[6]=head
    return out

class PosturePolicy:
    def __init__(self,stand,sit,stand_parameters=None,sit_parameters=None):
        self.stand=stand;self.sit=sit
        self.stand_parameters=np.zeros(8,np.float32) if stand_parameters is None else np.array(stand_parameters,np.float32)
        self.sit_parameters=np.r_[1.,np.zeros(8)].astype(np.float32) if sit_parameters is None else np.array(sit_parameters,np.float32)
        assert self.stand_parameters.shape==(8,) and self.sit_parameters.shape==(9,) and np.isfinite(self.stand_parameters).all() and np.isfinite(self.sit_parameters).all()
    def get_inputs(self):return self.stand.get_inputs()
    def run(self,names,inputs):
        obs=next(iter(inputs.values())).copy();sit_requested=obs[:,48]>=.5
        obs[:,48:61]=0
        standing=self.stand.run(None,{self.stand.get_inputs()[0].name:obs})[0]
        sit_obs=obs.copy();sit_obs[:,48]=self.sit_parameters[0]
        seated=self.sit.run(None,{self.sit.get_inputs()[0].name:sit_obs})[0]
        action=np.where(sit_requested[:,None],seated,standing)
        params=np.where(sit_requested[:,None],self.sit_parameters[1:],self.stand_parameters)
        for i,p in enumerate(params):
            correction=pose_bias(p[:6])
            # IMU-derived fore-aft balance acts on both ankles with mirrored signs.
            pitch=p[6]*obs[i,3]+p[7]*obs[i,1];correction[[4,13]]+=[pitch,-pitch]
            action[i]+=np.clip(correction,-.5,.5)
        return [action]
