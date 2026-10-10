"""Body-frame command limits, slew limiting and a stale-command stop."""
import numpy as np

class CommandLimiter:
    lower=np.array([-.15,-.1,-.4],np.float32)
    upper=np.array([.2,.1,.4],np.float32)
    acceleration=np.array([.5,.3,1.5],np.float32)
    def __init__(self,timeout_s=.6):
        self.timeout_s=timeout_s;self.target=np.zeros(3,np.float32)
        self.current=np.zeros(3,np.float32);self.received=-float('inf')
    def set(self,command,now):
        c=np.asarray(command,dtype=np.float32)
        if c.shape!=(3,) or not np.isfinite(c).all():raise ValueError('Command must contain three finite numbers.')
        self.target[:]=np.clip(c,self.lower,self.upper);self.received=float(now)
    def stop(self):
        self.target[:]=0;self.current[:]=0;self.received=-float('inf')
    def step(self,now,dt=.02):
        target=self.target if float(now)-self.received<=self.timeout_s else np.zeros(3,np.float32)
        self.current+=np.clip(target-self.current,-self.acceleration*dt,self.acceleration*dt)
        return self.current.copy()

def limiter_for_policy(policy):
    """An exported precision actor requires its gentler tested command ramp."""
    limiter=CommandLimiter()
    metadata=policy.get_modelmeta().custom_metadata_map if hasattr(policy,'get_modelmeta') else {}
    profile=metadata.get('command_limiter','default')
    if profile not in ('default','precision_v1'):raise ValueError(f'Unknown command limiter: {profile}')
    if profile=='precision_v1':limiter.acceleration=np.array([.25,.2,1.],np.float32)
    return limiter

