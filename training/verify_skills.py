"""Small contract checks for command validation and the policy supervisor."""
import argparse,json
from pathlib import Path
import numpy as np
from skill_controller import SkillController,Intent,parse_command

class Model:
    def __init__(self):self.inputs=[]
    def get_inputs(self):return [type('Input',(),dict(name='obs'))()]
    def run(self,names,inputs):
        obs=next(iter(inputs.values()));self.inputs.append(obs.copy());return [np.zeros((1,14),np.float32)]

class Env:
    head_cmd=np.zeros(4,np.float32);body_cmd=np.zeros(6,np.float32);twist_cmd=np.zeros(3,np.float32);_gyro=np.zeros(3,np.float32)
    def body_lin_vel(self):return np.zeros(3,np.float32)
    def _get_obs(self):
        obs=np.zeros(61,np.float32);obs[48:51]=self.twist_cmd;return obs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert parse_command('Please walk forward for three seconds!')==Intent('move',(.2,0.,0.),3.)
    assert parse_command('Duck, sit down.')==Intent('sit')
    assert parse_command('stand up').kind=='stand' and parse_command('stop').kind=='stop'
    invalid=['','do not walk forward','don\'t sit','walk forward and sit','walk forward; stop','walk forward for 11 seconds','walk forward for 0 seconds','walk forward for nan seconds','sit for three seconds','hello duck',None,{},'x'*161]
    for text in invalid:
        try:parse_command(text)
        except ValueError:pass
        else:raise AssertionError(('Invalid phrase accepted',text))
    w,p=Model(),Model();env=Env();ctl=SkillController(w,p)
    standing=dict(standing=True,sitting=False,payload_floor=[],payload_internal=[],tilt_deg=0.,props=[],height_m=.12)
    seated=dict(**{**standing,'standing':False,'sitting':True,'height_m':.055,'props':['body']})
    ctl.accept(parse_command('walk forward for ten seconds'),0)
    ctl.action(env,standing,0);assert ctl.phase=='moving' and w.inputs[-1][0,48]>0 and not p.inputs
    ctl.action(env,standing,.62);assert ctl.pending is None and ctl.goal=='stand'
    ctl.accept(parse_command('sit'),1.)
    for i in range(80):
        t=1+i*.02;ctl.action(env,standing,t)
        if ctl.phase=='sitting':break
    assert ctl.phase=='sitting' and p.inputs[-1][0,48]==1 and np.max(abs(p.inputs[-1][0,49:51]))==0
    for i in range(30):ctl.action(env,seated,3+i*.02)
    assert ctl.phase=='seated'
    ctl.accept(parse_command('walk forward'),4.)
    ctl.action(env,seated,4.);assert ctl.phase=='standing' and p.inputs[-1][0,48]==0 and ctl.expires is None
    ctl.accept(parse_command('stop'),4.1)
    for i in range(120):ctl.action(env,standing,4.1+i*.02)
    assert ctl.phase=='idle' and ctl.pending is None and ctl.expires is None
    assert any(t['to_phase']=='settling' for t in ctl.transitions)
    ctl.accept(parse_command('walk forward'),7.)
    ctl.action(env,standing,7.);ctl.accept(parse_command('stop'),7.1)
    assert ctl.pending is None and np.max(abs(ctl.command))==0
    ctl.check({**standing,'payload_internal':['mount / leg']},7.2);assert ctl.fault and ctl.phase=='fault'
    try:ctl.accept(parse_command('stand'),8.)
    except ValueError:pass
    else:raise AssertionError('Fault permitted command')
    ctl.reset();assert not ctl.fault and ctl.phase=='idle'
    ctl.accept(parse_command('walk backward'),9.)
    for i in range(40):ctl.heartbeat(9+i*.02);action=ctl.action(env,standing,9+i*.02)
    assert np.max(abs(action))<=.080001 and np.array_equal(action,ctl.reverse_bias)
    assert np.max(abs(action[5:9]))==0
    d=dict(passed=True,invalid_phrases_rejected=len(invalid),command_slots_isolated=True,lease_cancellation_checked=True,automatic_stand_before_movement=True,
           stop_cancels_pending_movement=True,walking_settle_after_posture=True,mount_contact_fault_checked=True,reset_checked=True,reverse_correction_bound_rad=.08,head_unmodified_by_reverse_correction=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2)+'\n');print('Skill command/supervisor checks passed')
if __name__=='__main__':main()
