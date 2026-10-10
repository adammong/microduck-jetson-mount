# Mounted slow-motion and push-recovery study

The selected simulation policy reduces average slow-command tracking error by **45.3%** relative to the previous directional controller. Push survival improves modestly, from **213/288 (74.0%) to 222/288 (77.1%)**. The v06 Jetson/battery mount is included throughout. Tiny translations still stall, and stronger pushes still cause falls or mount contacts. Hardware is untested.

[Continuous 35-second demonstration](motion-demo.mp4) · [15-second clip, top text removed](policy-15s.mp4) · [Selected ONNX](policies/selected/policy.onnx) · [Machine-readable summary](summary.json)

The demonstration is fresh seed 640: slow forward at 0.10 m/s, stops, left/right turns at ±0.10 rad/s, then two opposite 4 N side pushes while standing. Pushes occur at 27 and 30 seconds and last 0.12 seconds. The short clip is the continuous 18–33 second interval; it shows the right turn, stopping and both pushes. Seed 641 also completes the same sequence. These are simulation recordings; external forces have no visible pusher. The clean videos remove only the upper 64-pixel text band. Original recordings and both frame sheets are retained.

## Measurements

Each condition has 8 samples per case: nominal seeds 600–607; randomized seeds 620–623 at each payload scale 0.9 and 1.1. Baseline and selected controllers use matching cases, seeds and mass scales. These seeds were not used in training or controller calibration. Both use their respective tested command ramps, so this compares complete controllers, not isolated neural weights.

| Final paired benchmark | Previous directional controller | Selected controller |
|---|---:|---:|
| Slow-command mean normalized error, nominal | 1.462 | 0.770 |
| Slow-command mean normalized error, randomized | 1.339 | 0.762 |
| Slow start/stop trials without falls or detected contacts | 288/288 | 288/288 |
| Push trials surviving | 213/288 | 222/288 |
| Push trials without falls or detected contacts | 156/288 | 165/288 |
| Full-command regression trials without falls or detected contacts | 176/176 | 176/176 |

Thus **464/464 non-push trials** passed the contact and survival checks. Separately, **16/16 fresh 69-second control sequences** passed without detected contacts: nominal seeds 800–807 and randomized seeds 820–827. These sequences exercise starts, stops, reverse, sidesteps, both turns and curves. They use nominal payload mass; ±10% payload testing is in the paired benchmark above.

For the 18 slow cases, commands are 25%, 50% and 75% of each of six directional presets. Each 16-second trial stands for 3 seconds, moves for 8, then stops for 5. Tracking error is the Euclidean norm of the 5–11 second mean body-twist error divided componentwise by `[0.04 m/s, 0.025 m/s, 0.12 rad/s]`. The 45.3% reduction is the combined mean across all 288 trials, not a success rate or a guarantee at every command.

Selected mean tracking examples across both conditions:

| Requested command | Measured matching component, mean ± standard deviation |
|---|---:|
| Forward 0.10 m/s | 0.113 ± 0.033 m/s |
| Backward −0.075 m/s | −0.077 ± 0.061 m/s |
| Left 0.050 m/s | 0.046 ± 0.036 m/s |
| Right −0.050 m/s | −0.040 ± 0.023 m/s |
| Turn left 0.10 rad/s | 0.106 ± 0.044 rad/s |
| Turn right −0.10 rad/s | −0.080 ± 0.016 rad/s |

The 25% translation commands (forward 0.05, reverse −0.0375 and sidesteps ±0.025 m/s) remain essentially stationary. Some 50% translation trials also stall; others overshoot or drift. Slow turning improved more consistently. Full case means and variation are in [summary.json](summary.json), and individual trials remain in the JSON records.

All 288 slow trials settled during the five-second stop window. Settling means one continuous second with instantaneous linear speed below 0.02 m/s and yaw rate below 0.12 rad/s. Selected median settling time is **0.56 seconds**, maximum **1.60 seconds**. Mean integrated travel after the stop request is **7.9 cm**, maximum **21.8 cm**, including gait transients and sideways drift. The previous controller averages 3.3 cm, but many previous low commands never started moving. This is not evidence of better braking distance. The final two-second average absolute residual components are approximately 0.00015 m/s, 0.00023 m/s and 0.0036 rad/s; averaging can hide oscillations.

Push tests apply two opposite world-axis forces to the torso body at 5 and 10 seconds, each for 0.12 seconds (six 50 Hz control ticks). Cases cover standing, forward 0.20 m/s and left sidestepping 0.10 m/s, both world x/y axes, and 2/4/6 N forces. These correspond to 0.24/0.48/0.72 N·s impulses per pulse. Survival counts can include transient mount or non-foot contact; the stricter column excludes those contacts.

| Force per pulse | Previous survival / no-contact | Selected survival / no-contact |
|---|---:|---:|
| 2 N | 96/96 · 96/96 | 96/96 · 96/96 |
| 4 N | 75/96 · 46/96 | 80/96 · 50/96 |
| 6 N | 42/96 · 14/96 | 46/96 · 19/96 |

There are still **66 falls and 57 surviving trials with detected contacts** in the selected push benchmark. In particular, larger fore-aft pushes while standing and side pushes while sidestepping remain difficult. The modest aggregate gain does not establish reliable recovery from arbitrary disturbances or statistical significance.

## What was trained

This is CPU numerical training on the Mac: paired command/gait grids followed by cross-entropy-method (CEM) searches for bounded observable feedback. The original actor and normalization tensors remain unchanged. This adds continuous command-power shaping, gait amplitude relative to the same actor at zero command, low pure-turn feedback from the gyro, small forward-gated gyro/gravity joint corrections, and tilt-triggered catch steps while standing. It is not another PPO neural fine-tune or a resumable PPO checkpoint.

The standing catch rule activates above a projected-gravity threshold and requests bounded x/y stepping through the same learned command map. It does not apply stabilizing forces. The actor still uses **61 observations and 14 joint actions at 50 Hz**. It uses gyro/gravity and command slots, not simulator velocity, position, force, a hidden timer or privileged state. Additional IMU joint corrections are clipped to ±0.12 rad; the inherited directional actor has its own separate correction bound. The total action is not claimed to have a combined ±0.12 rad bound.

The retained searches include:

- `slow-grid`: preliminary command-power grid, superseded by the amplitude grid.
- `amplitude-grid`: six directional power/amplitude grids, 1,152 training rollouts on seeds 0/1.
- `balance-cem`: 6 generations × 16 candidates, using 18 cases × 2 seeds. Applying this correction globally regressed some sidestep pushes; the final actor gates it to forward commands.
- `turn-grid`: 28 candidates per turn direction, six low-turn trials per candidate on seeds 0/1.
- `recovery-cem`: initial direct catch commands, superseded. Its original numerical policy and driver snapshots are retained.
- `recovery-shaped-cem`: 6 generations × 12 candidates × 24 standing-push trials, seeds 0–3, with noise/randomization on odd seeds and payload scales 0.9/1.0/1.1. Catch commands pass through command-power shaping. The selected parameter vector is retained.

Training used the original command acceleration `[0.5, 0.3, 1.5]`. An initial sequence confirmation using that ramp fell at **3.48 seconds on seed 651**, while the previous actor survived the same sequence. That failure remains in `sequences-nominal.json`. A paired controller study on seeds 650–653 selected the gentler `[0.25, 0.20, 1.0]` ramp; three tested gentler variants all passed 4/4 sequences. The least restrictive passing variant was chosen, then all **752 selected benchmark trials** and **16 fresh sequences** were rerun with it.

The final ONNX metadata specifies `command_limiter=precision_v1`; the local controller and evaluation wrapper read it. An external robot runtime must implement the equivalent ramp and stale-input stopping explicitly. Loading the actor into another runtime does not itself implement these controls. Command bounds are forward +0.20/−0.15 m/s, sideways ±0.10 m/s and yaw ±0.40 rad/s; the stale-command timeout is 0.6 seconds. The explicit Stop button zeros the command immediately, but physical motion takes time to settle.

## Physics, verification and artifacts

The nominal loaded robot mass is **1.1793 kg**, including an estimated **442.1 g** mount/electronics/battery payload. Payload mass and inertia scale together by ±10%, with its center of mass held fixed. Randomized trials additionally use the pinned harness's trunk mass/inertia/center-of-mass, friction, armature and motor-voltage/sag randomization, plus observation noise. All tests use BAM XL330 motors at **1.75 A**, 200 Hz physics, 50 Hz control and 15–30 ms bus delay. There is no spotter, assistance or motor-force increase. Two limp negative controls fall; this distinguishes successful policy control from a model that stands regardless of actions.

Contact checks sample at 50 Hz and count forces above 0.05 N. The strict check excludes any detected non-foot floor support or mount-to-robot contact. The termination guard requires successive non-foot support, so survival alone is insufficient. Substep impacts can be missed. The mount is rigid, collision shapes are approximate and cables exert no forces; printed fit, mount slip/flex, thermal limits, uneven terrain and physical walking are unverified. Existing full joint-travel CAD collisions remain unresolved.

[Final verification](final-verification.json) checks unchanged base tensors, executable ONNX/numerical agreement on 1,000 fresh observations (maximum error 3.58e−7 rad), finite outputs, dimensions, command ramp, watchdog, mass/current and negative controls. [Live controller check](live-controller-check.json) runs the selected actor through the actual local control server, including forward motion, a low-speed turn, Stop, stale-input stopping and nonfinite-input rejection. [Control-page checks](controls-verification.json) verify speed scaling, combined keys and Stop behavior. Both demo seeds complete without detected contacts.

Selected SHA-256: `b507144a2c4a01205aa38382c67f6117724f00e2c7b9f5910cd7a141b8321920`.

Base directional SHA-256: `92268b6202b3f732b0f5e883fe14b718855077c24a7508d9f32fe20f573df3ee`.

[Selection index](selection.json), [parameters and source hashes](policies/selected/run.json), [source input provenance](../../training/inputs.json) and [scene provenance](../mac-v06/scenes.json) accompany the exports. `policies/recovery-v2` preserves the same actor before the ramp metadata change. Its historical evaluations remain linked to that immutable export. `confirmation-selected-*`, `sequences-*` and `demo-nominal*` are pre-ramp confirmations, not final deployment measurements. Authoritative results start with `final-`; baseline comparisons are `confirmation-baseline-*`.

Search histories include rejected candidates. Some preliminary pilots used earlier source versions that were not all snapshotted; these are measurement records, not claims of exact replay from today's source. Current sources and the selected parameters reproduce the final export. Retained snapshots document the initial catch method and final catch search; their original driver imports refer to the training module, so replaying an old method requires using its matching policy snapshot and original ramp.

## Reproduce and control

First complete the pinned [Mac setup](../../training/README.md). Run from the repository root:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_precision.py --folder results/precision-v06/policies/selected --out tmp/precision-verification.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_precision.py --policy results/precision-v06/policies/selected/policy.onnx --out tmp/precision-nominal.json --suite all --seed-start 600 --seeds 8
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_precision.py --policy results/precision-v06/policies/selected/policy.onnx --out tmp/precision-randomized.json --suite all --seed-start 620 --seeds 4 --noise-dr --payload-scales .9 1.1
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_transitions.py --policy results/precision-v06/policies/selected/policy.onnx --out tmp/precision-sequences.json --seed-start 800 --seeds 8
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/demo_precision.py --policy results/precision-v06/policies/selected/policy.onnx --out tmp/precision-demo.json --seed-start 640 --render
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_motion.py --policy results/precision-v06/policies/selected/policy.onnx
```

Open http://127.0.0.1:8798. Use W/S, A/D, Q/E and Space; combine movement/turn keys for curves. The Speed slider scales commands. Try 50% for translations and 25% for slow turns; 25% translations mostly stall. The previous directional actor remains the page's default unless the selected precision actor is passed explicitly.

Repeat the final catch search from recorded pre-catch parameters into a fresh output directory:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/tune_balance_recovery.py --init results/precision-v06/policies/combined-v1/parameters.json --out tmp/repeat-standing-recovery
```

Export the selected parameters without retraining:

```sh
PYTHONPATH=training tmp/microduck-lab/microduck_local/.venv/bin/python - <<'PY'
import json
from precision_policy import save
p=json.load(open('results/precision-v06/policies/selected/parameters.json'))
save(p['exponents'], p['balance'], 'tmp/reexport-precision', p['gains'], p['turn_feedback'], p['recovery'], p['balance_forward_only'])
PY
```

License and modification notices accompany the exported policies. All selected and rejected artifacts are simulation studies, not hardware deployment approval.
