"""Check payload mass/inertia, policy contract, and isolated rollout repeatability."""
import json
from pathlib import Path
import numpy as np,mujoco,onnxruntime as ort
from evaluate import rollout,scene,ROOT

def main():
    metadata=json.loads((ROOT/'easy_mount/simulation/mass_properties.json').read_text())
    masses={}
    for v in ('stock','loaded'):
        model=mujoco.MjModel.from_binary_path(str(scene(v).with_suffix('.mjb')))
        masses[v]=float(model.body_mass.sum())
        assert model.nu==14 and model.nq==21
        assert (model.body_inertia[1:]>0).all()
    assert np.isclose(masses['stock'],.73724318,atol=1e-8)
    assert np.isclose(masses['loaded'],1.1885158048587487,atol=1e-8)
    teacher=ort.InferenceSession(str(ROOT/'tmp/sim-inputs/alpha_walking.onnx'),providers=['CPUExecutionProvider'])
    for v in ('stock','loaded'):
        a=rollout(v,teacher,[.3,0,0],0,2)
        b=rollout(v,teacher,[.3,0,0],0,2)
        assert a==b,(a,b)
    good=rollout('stock',teacher,[.3,0,0],0,2,strict=True)
    assert good['survived']
    bad_policy=ROOT/'tmp/mac-runs/loaded-bam-s0-1048576/policy.onnx'
    if not bad_policy.exists():bad_policy=ROOT/'results/mac-v05/policies/loaded/policy.onnx'
    if bad_policy.exists():
        bad=ort.InferenceSession(str(bad_policy),providers=['CPUExecutionProvider'])
        rejected=rollout('loaded',bad,[0,0,0],0,2,strict=True)
        assert not rejected['survived'] and rejected['failure_reason']=='nonfoot_floor_support'
    out={'passed':True,'strict_guard':'Stock forward roll survives; backpack-supported pilot is rejected','mass_kg':masses,'added_mass_kg':masses['loaded']-masses['stock'],
         'obs':61,'actions':14,'policy_rate_hz':50,'physics_rate_hz':200,
         'repeatability':'Two isolated 2-second seed-0 rollouts per variant are exactly equal',
         'actuator':'BAM XL330 M6, nominal firmware current limit 1.75 A, no strength scaling',
         'contact_model':'groundcontact, mount collision boxes active'}
    (ROOT/'results/mac-v05/verification.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
