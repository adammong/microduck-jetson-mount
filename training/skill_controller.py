"""Bounded commands and explicit walking/posture handoffs for the mounted duck."""
from dataclasses import dataclass
import re
import numpy as np
from evaluate_motion import COMMANDS
from motion_controller import limiter_for_policy

@dataclass(frozen=True)
class Intent:
    kind: str
    command: tuple = (0.,0.,0.)
    seconds: float = 0.

PHRASES = {
    'forward': 'forward', 'walk forward': 'forward', 'move forward': 'forward', 'go forward': 'forward',
    'backward': 'backward', 'walk backward': 'backward', 'walk backwards': 'backward', 'move backward': 'backward',
    'left': 'left', 'walk left': 'left', 'move left': 'left', 'sidestep left': 'left',
    'right': 'right', 'walk right': 'right', 'move right': 'right', 'sidestep right': 'right',
    'turn left': 'turn_left', 'turn right': 'turn_right',
    'sit': 'sit', 'sit down': 'sit', 'stand': 'stand', 'stand up': 'stand',
    'stop': 'stop', 'stop moving': 'stop', 'halt': 'stop',
}
NUMBERS = dict(zip('one two three four five six seven eight nine ten'.split(),range(1,11)))

def parse_command(text):
    if not isinstance(text,str) or not 0<len(text)<=160:raise ValueError('Say one short command.')
    text=text.lower().strip().rstrip('.!?').strip()
    text=re.sub(r'\s+',' ',text)
    text=re.sub(r'^(?:duck|microduck)[, ]+','',text)
    text=re.sub(r'^please ','',text)
    match=re.fullmatch(r'(.+?) for (\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten) seconds?',text)
    seconds=3.
    if match:
        text,n=match.groups();seconds=float(NUMBERS[n] if n in NUMBERS else n)
        if not .5<=seconds<=10:raise ValueError('Movement duration must be 0.5 to 10 seconds.')
    kind=PHRASES.get(text)
    if kind is None:raise ValueError('Use walk forward/backward, sidestep left/right, turn left/right, stop, sit, or stand.')
    if kind in ('sit','stand','stop'):
        if match:raise ValueError('Durations apply only to movement.')
        return Intent(kind)
    return Intent('move',tuple(COMMANDS[kind]),seconds)

class SkillController:
    """One interface; separate actors retain their different command meanings.

    shortcut: posture readiness uses simulation measurements; hardware needs
    a validated estimator and its own actuator watchdog before deployment.
    """
    def __init__(self,walking,posture,blend_s=.3):
        if not np.isfinite(blend_s) or not 0<=blend_s<=1:raise ValueError('Invalid blend time')
        self.walking=walking;self.posture=posture;self.blend_s=float(blend_s)
        self.settle_s=0.
        self.reverse_bias=np.zeros(14,np.float32)
        self.reverse_bias[[2,11]]=[-.04,.04]
        self.reverse_bias[[3,12]]=[-.04,.04]
        self.reverse_bias[[4,13]]=[.08,-.08]
        self.limiter=limiter_for_policy(walking)
        self.reset()

    def reset(self):
        self.phase='idle';self.goal='stand';self.pending=None;self.expires=None
        self.heartbeat_at=-float('inf');self.entered=0.;self.stable=0.
        self.last_action=None;self.blend_from=None;self.blend_age=0.
        self.active_actor='walking';self.fault=None;self.transitions=[]
        self.command=np.zeros(3,np.float32);self.limiter.stop()
        self.nonfoot_steps=0

    def heartbeat(self,now):self.heartbeat_at=float(now)

    def accept(self,intent,now):
        if not isinstance(intent,Intent):raise ValueError('Invalid intent')
        if intent.kind not in ('move','sit','stand','stop'):raise ValueError('Invalid intent')
        if intent.kind=='move' and (len(intent.command)!=3 or not np.isfinite(intent.command).all() or not .5<=intent.seconds<=10):raise ValueError('Invalid movement')
        if self.fault:raise ValueError('Reset the simulation after a fault.')
        self.heartbeat(now)
        if intent.kind=='stop':
            self.pending=None;self.expires=None;self.limiter.stop();self.command[:]=0
            self.goal='sit' if self.phase in ('sitting','seated') else 'stand'
        elif intent.kind=='move':
            self.pending=intent;self.goal='move';self.expires=None
            if self.phase=='moving':self.expires=float(now)+intent.seconds
        else:
            self.pending=None;self.expires=None;self.goal=intent.kind

    def _phase(self,phase,now):
        if phase==self.phase:return
        self.transitions.append(dict(time_s=round(float(now),6),from_phase=self.phase,to_phase=phase))
        self.phase=phase;self.entered=float(now);self.stable=0.

    def fail(self,reason,now):
        self.fault=str(reason);self.goal='stand';self.pending=None;self.expires=None
        self.limiter.stop();self.command[:]=0;self._phase('fault',now)

    def action(self,env,measurement,now,dt=.02):
        if self.fault:raise ValueError('Controller is faulted')
        if not np.isfinite(now) or not 0<dt<=.1:raise ValueError('Invalid clock')
        if self.goal=='move' and now-self.heartbeat_at>.6:
            self.pending=None;self.goal='sit' if self.phase in ('sitting','seated') else 'stand';self.expires=None
        if self.phase=='moving' and self.expires is not None and now>=self.expires:
            self.pending=None;self.goal='stand';self.expires=None
        v=env.body_lin_vel();gyro=env._gyro
        settled=measurement['standing'] and np.max(abs(v[:2]))<.04 and abs(gyro[2])<.12
        if self.phase in ('idle','moving'):
            if self.goal=='sit':self._phase('stopping',now)
            elif self.goal=='move' and self.pending:
                if self.phase!='moving':self.expires=now+self.pending.seconds
                self._phase('moving',now)
            else:self._phase('idle',now)
        elif self.phase=='stopping':
            self.stable=self.stable+dt if settled and np.max(abs(self.command))<.005 else 0.
            if self.stable>=.6:self._phase('sitting' if self.goal=='sit' else 'idle',now)
        elif self.phase=='sitting':
            self.stable=self.stable+dt if measurement['sitting'] else 0.
            if self.stable>=.5:self._phase('seated',now)
        elif self.phase=='seated':
            if self.goal!='sit':self._phase('standing',now)
        elif self.phase=='standing':
            self.stable=self.stable+dt if settled else 0.
            if self.stable>=1.:self._phase('settling',now)
        elif self.phase=='settling':
            self.stable=self.stable+dt if settled else 0.
            if self.stable>=self.settle_s:self._phase('idle',now)
        if self.phase in ('stopping','sitting','standing','settling') and now-self.entered>8.:
            self.fail('Posture transition timed out',now);raise ValueError(self.fault)
        moving=self.phase=='moving' and self.pending is not None
        self.limiter.set(self.pending.command if moving else [0,0,0],now)
        self.command=self.limiter.step(now,dt)
        actor='posture' if self.phase in ('sitting','seated','standing') else 'walking'
        if actor!=self.active_actor:
            self.blend_from=None if self.last_action is None else self.last_action.copy()
            self.blend_age=0.;self.active_actor=actor
        env.head_cmd[:]=0;env.body_cmd[:]=0
        env.twist_cmd[:]=self.command if actor=='walking' else [float(self.phase in ('sitting','seated')),0,0]
        policy=self.walking if actor=='walking' else self.posture
        obs=env._get_obs().astype(np.float32)
        action=policy.run(None,{policy.get_inputs()[0].name:obs[None]})[0].reshape(14)
        if self.blend_from is not None and self.blend_s>0:
            self.blend_age+=dt;alpha=min(1.,self.blend_age/self.blend_s)
            action=(1-alpha)*self.blend_from+alpha*action
            if alpha==1:self.blend_from=None
        if actor=='walking' and self.command[0]<0:
            action=action+min(-float(self.command[0])/.05,1.)*self.reverse_bias
        if not np.isfinite(action).all():self.fail('Non-finite actor output',now);raise ValueError(self.fault)
        self.last_action=action.copy()
        return action

    def check(self,measurement,now):
        if measurement['payload_floor'] or measurement['payload_internal']:self.fail('Mount contact',now)
        elif measurement['tilt_deg']>60:self.fail('Duck fell',now)
        elif self.phase in ('idle','moving','stopping','settling'):
            self.nonfoot_steps=self.nonfoot_steps+1 if measurement['props'] else 0
            if measurement['height_m']<.07 or self.nonfoot_steps>=3:self.fail('Duck fell or used non-foot support',now)
        else:self.nonfoot_steps=0

    def status(self,now):
        return dict(phase=self.phase,goal=self.goal,active_actor=self.active_actor,command=self.command.tolist(),
                    remaining_s=max(0.,self.expires-now) if self.expires is not None else None,
                    pending_movement=self.pending is not None,fault=self.fault)
