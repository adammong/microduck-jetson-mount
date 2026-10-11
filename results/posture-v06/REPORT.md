# Mounted sit/stand and floor recovery — experimental

CPU training improves **front recovery from 3/16 to 14/16** in paired fresh nominal/randomized tests with the complete v06 mount. **Full sit/stand remains unresolved.** Synthetic back/left/right recovery remains **0/48**, and a separate recorded-fall benchmark improves only **7/16 → 8/16**. The exported ONNX is a limited simulation candidate, not a general recovery policy or a hardware release.

[Successful front-recovery video](front-recovery-success.mp4) · [Failed front-recovery video](front-recovery-failure.mp4) · [Sit/stand interference video](sitstand-interference.mp4) · [Candidate ONNX](policies/front-recovery-candidate/policy.onnx) · [Summary](summary.json)

The successful 10-second recording is fresh seed 251; seed 250 fails and is retained alongside it. Both begin in settled front-down poses, with collisions and ordinary motors active. The videos remove only the upper 64-pixel text strip. Original captioned recordings and frame sheets remain available. One attractive clip does not establish reliable recovery.

## Sit and stand

The official `alpha_sitstand` uses observation slot 48 as a posture flag: 1 means sit, 0 means stand. This differs from its forward-velocity meaning in the walking policy. The [upstream inference implementation](https://github.com/pollen-robotics/microduck_rl/blob/develop/scripts/infer_policy.py) documents that distinction. The actual model downloads are pinned to revision `088524a64e2557dc453256b6071dbb9d23888802`; checksums are in [posture_inputs.json](../../training/posture_inputs.json).

Each sit/stand test stands for 3 seconds, requests sitting for 8 seconds, then standing for 8 seconds. A proper sit must have trunk height below 0.07 m, tilt below 35°, robot-body support, and no detected mount-floor or mount-robot contact. Returning to stand requires both feet on the floor, no other floor support or mount interference, tilt below 20°, trunk height at least 0.095 m and jaw height at least 0.18 m, continuously for the final two seconds. These criteria reject a shallow crouch or a mount-supported heap.

The stock robot passes **2/2** baseline sit/stand cycles and sits at about **0.059 m** trunk height. The same actor on the mounted robot passes **0/2**: it settles against the rear battery pocket and the folded feet hit the front dock/tray. In the unchanged model's `SIT` keyframe, each foot penetrates a front-dock collision box by about **8.6 mm**. [Verification](verification.json) records those intersections; they are simulation geometry findings, not measurements on a printed robot.

Three bounded searches were attempted:

- Sit/stand actor adaptation: 8 generations × 20 candidates × 2 seeds. It finds a shallower crouch around 0.096–0.102 m, with occasional tray contact. **No proper seated cycle passes.**
- Explicit seated joint targets with observable feedback: an initial search had a joint-angle frame error and is rejected. Its source snapshot is retained under `seated-pose-cem`; it incorrectly subtracted the default pose from already-relative observations.
- Corrected target search: 10 generations × 24 candidates × 2 seeds, with nominal seed 0 and randomized seed 1. It reaches lower poses but remains supported/interfered with by the mount and cannot reliably return to stand. **No proper seated cycle passes.** The corrected frame and measured-joint target increment are checked in `verify_posture.py`.

A body-height command grid also fails to produce a proper seat. These records are retained as rejected experiments, not working sit policies. No Sit button is enabled in the existing controller.

As a diagnostic only, disabling mount collision while retaining payload mass/inertia allows a combination of the sit actor for sitting and `alpha_stand` for rising to pass **2/2** pilot cycles. This isolates clearance as a major obstacle; it is not a usable physical result. Simple collision-layout probes lifting the front tray by 10/20/30 mm and the battery by 0/15 mm still fail. Their inertia/geometry approximations are labeled diagnostic and no revised CAD is selected. A broader mount redesign must address foot, dock, rear-web and head clearance together.

## Floor recovery

The candidate preserves every neural actor and normalizer tensor in official `alpha_stand`. CEM trains 14 parameters for gravity-gated pose offsets, gyro/gravity ankle and hip feedback, differential leg motion and head/neck correction. Additional joint correction is clipped to ±0.75 rad. The 61-observation/14-action interface remains unchanged; command slots are zeroed for this recovery actor. It does not observe simulator position, height, contact forces or true velocity, and has no hidden phase timer. It applies no balancing force or teleport.

Training uses seeds 0/1 over four settled lying orientations; seed 1 includes observation noise and randomized physics. The gravity-conditioned search runs **10 generations × 24 candidates × 8 trials**. A preceding 8-parameter search runs **8 × 20 × 8 trials**. Fresh validation seeds 100–107 compare both approaches; both improve front recovery but fail the other orientations. The gravity-conditioned variant is exported as the limited candidate because it keeps a more level standing posture. This is numerical CEM calibration, **not a new PPO neural fine-tune**.

Final paired evaluation uses nominal seeds 200–207 and randomized seeds 220–223, each at payload scales 0.9 and 1.1. Each recovery trial lasts 10 seconds. Success requires a qualifying stand for at least three consecutive seconds and throughout the final two seconds. Floor support is allowed during recovery but excluded from the completed stand; simple survival is insufficient.

| Settled synthetic orientation | Official stand actor | Trained candidate |
|---|---:|---:|
| Front, nominal | 2/8 | 7/8 |
| Front, randomized and ±10% payload | 1/8 | 7/8 |
| Back, both conditions | 0/16 | 0/16 |
| Left, both conditions | 0/16 | 0/16 |
| Right, both conditions | 0/16 | 0/16 |
| Already standing, both conditions | 16/16 | 16/16 |

The standing trials are reported separately and are not counted as recoveries. Across the four fallen orientations, the candidate succeeds **14/64**, versus **3/64** for the baseline. The two additional demo seeds produce one success and one failure; those are separate from this benchmark.

Synthetic starts use the pinned harness's folded-leg distribution and 80–115° tilt range. They settle under BAM physics before the policy starts; root velocity is reset to zero, and joint/action delay state is primed at the resting pose. The nominal one-second settle is inherited from the upstream method. It is not a claim that these starts cover every real fall.

A separate test records **16 physically generated resting poses**. A 0.35 N·m torso torque lasts 0.2 seconds in each requested tipping direction, followed by 23.8 seconds with no external torque and limp joint targets. The resulting distribution matters: four trials end front-down, while the other twelve collapse mostly upright against the mount at about 0.052 m height. They are not twelve flat back/side lies. Maximum recorded root-speed norm is 0.000159 and kinetic energy is 2.60e−7 J. Replays zero velocity and preserve recorded joint/root orientation; their environment seeds are 940–955.

| Requested physical tip, resulting pose distribution | Official stand actor | Candidate |
|---|---:|---:|
| Front, front-down | 1/4 | 3/4 |
| Back, collapsed upright | 0/4 | 0/4 |
| Left, collapsed upright | 3/4 | 2/4 |
| Right, mostly collapsed upright | 3/4 | 3/4 |
| Total | 7/16 | 8/16 |

Sixteen paired limp negative controls recover **0/16**. The candidate regresses one left-tip replay; it is therefore not installed as an automatic recovery controller. Recorded poses and all successes/failures remain in the repository. The improvement on synthetic front lies does not establish general fall-recovery reliability.

## Physics and limitations

All accepted measurements keep the **1.1793 kg** nominal robot, including its **442.1 g** estimated payload, unchanged v06 collision geometry, **1.75 A BAM XL330 motors**, 200 Hz physics, 50 Hz policy execution and bus delay. Randomized tests vary the pinned harness's physical parameters and add observation noise; ±10% payload scales mass and inertia with fixed center of mass. There is no motor-strength scaling, spotter, assistance or active perturbation during recovery. Controlled floor contact is necessary while getting up, so the walking harness's immediate non-foot-support termination is intentionally not used here.

Contact metrics sample at 50 Hz and count normal forces above 0.05 N; substep impacts can be missed. Printed fit/strength, damage from falls, flexible cables, mount slip, motor heat and hardware execution remain untested. Collision boxes are approximations. More training alone is not proof that an intersecting seated path is mechanically usable.

[ONNX verification](export-verification.json) checks unchanged base tensors and numeric/export agreement on 1,000 fresh observations (maximum error **1.19e−7 rad**). [Physics/frame verification](verification.json) checks mass isolation, current, timestep, repeated rollout equality, replay validation, relative-joint conventions and the seated mount intersections. [Run provenance](run.json) and [selection status](selection.json) distinguish the candidate, rejected searches and collision-disabled/layout diagnostics.

Candidate SHA-256: `feffaef64c303b4d5f1b41980a01a3d349fc6bd527585d37b5136f7de7bd35db`.

## Reproduce

Complete [Mac setup](../../training/README.md), which now downloads both pinned posture actors. From the repository root:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_posture.py --out tmp/posture-check.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_posture_batch.py --policy results/posture-v06/policies/front-recovery-candidate/policy.onnx --out tmp/recovery-nominal.json --seed-start 200 --seeds 8
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_posture_batch.py --policy results/posture-v06/policies/front-recovery-candidate/policy.onnx --out tmp/recovery-randomized.json --seed-start 220 --seeds 4 --noise-dr --payload-scales .9 1.1
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_posture_batch.py --policy results/posture-v06/policies/front-recovery-candidate/policy.onnx --out tmp/recovery-recorded.json --seed-start 940 --fall-recordings results/posture-v06/physical-falls.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_posture.py --policy results/posture-v06/policies/front-recovery-candidate/policy.onnx --out tmp/recovery-demo.json --poses front --seed-start 251 --seeds 1 --render
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/tune_floor_recovery.py --out tmp/repeat-recovery --generations 10 --population 24
```

Re-export without retraining:

```sh
PYTHONPATH=training tmp/microduck-lab/microduck_local/.venv/bin/python - <<'PY'
import json
from recovery_policy import export
p=json.load(open('results/posture-v06/policies/front-recovery-candidate/parameters.json'))
export(p['parameters'], 'tmp/reexport-recovery')
PY
```

The existing walking controller is not a recovery preview: its walking termination guard stops after non-foot support. Use the dedicated posture evaluator above. Full sitting and general recovery need further mechanical/policy work; no completed solution is claimed for those tasks.
