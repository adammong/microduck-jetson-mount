"""Bounded seated joint target with observable feedback and learned stand handoff."""
import numpy as np

DEFAULT=np.array([0,-.0873,-.4579,-.0049,.453,.3491,.3491,0,0,0,.0873,.4579,.0049,-.453],np.float32)
class SeatedPolicy:
    def __init__(self,stand,parameters):
        self.stand=stand;self.parameters=np.asarray(parameters,np.float32)
        assert self.parameters.shape==(7,) and np.isfinite(self.parameters).all()
    def get_inputs(self):return self.stand.get_inputs()
    def run(self,names,inputs):
        obs=next(iter(inputs.values())).copy();flag=obs[:,48]>=.5;obs[:,48:61]=0
        action=self.stand.run(None,{self.get_inputs()[0].name:obs})[0]
        p=self.parameters;target=DEFAULT.copy();target[[2,11]]=[p[0],-p[0]];target[[3,12]]=[p[1],-p[1]]
        target[[4,13]]=[p[2],-p[2]];target[[1,10]]=[-p[3],p[3]];target[[0,9]]=[p[4],-p[4]];target[5:7]=[.5,.6]
        for i,o in enumerate(obs):
            if flag[i]:
                desired=target-DEFAULT;pitch=np.clip(p[5]*o[3]+p[6]*o[1],-.4,.4);desired[[4,13]]+=[pitch,-pitch]
                action[i]=desired
            # Direct stand handoff and sitting are both rate-limited from measured joints.
            # These observation slots already contain q - DEFAULT_POSE.
            action[i]=o[6:20]+np.clip(action[i]-o[6:20],-.03,.03)
        return [action]
