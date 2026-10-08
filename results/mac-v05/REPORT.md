# V05 mounted MicroDuck: Mac simulation pilot

**Local simulation and training work. The current mount is not ready for a reliable walking policy.**

The model weighs 0.737 kg stock and 1.189 kg loaded: **451 g added, a 61% increase**. Payload mass includes the Jetson, GNB 850 mAh pack, fully dense printed mount, and ancillary wiring/hardware allowances. The earlier static calculation puts the loaded center of mass about 12 mm farther back.

## Existing walking policy

Same official `alpha_walking.onnx`, paired seeds 0–7, ten-second trials, no assistance. Forward command is 0.30 m/s; stationary command is zero. Motor voltage/sag draws and initial-pose draws are matched by constructing each robot with the same seed. Separate pristine models prevent one trial's motor state affecting another.

| Measure | Stock | Loaded |
|---|---:|---:|
| Forward trials without a fall | 8/8 | 8/8 |
| Mean forward displacement in 10 s | 1.095 m | 0.232 m |
| Effective world-forward speed | 0.109 m/s | 0.023 m/s |
| Stationary trials without a fall | 8/8 | 7/8 |
| Mean world-forward displacement at stationary command | +0.002 m | −0.573 m |

The loaded robot can move but has pronounced backward drift at zero command. Forward displacement is projected onto the starting world-forward axis; it is not total path length. These short nominal-physics trials are not a reliability estimate for hardware.

The mount's upper gate bridge contacts the head-shell collision meshes during motion. MuJoCo uses convex robot meshes and conservative mount boxes, so this flags a clearance problem requiring exact CAD inspection; it does not establish an exact physical penetration. No collision geometry was disabled to improve the result.

[Mounted forward video](preview-loaded-forward.mp4) · [stationary-command video](preview-loaded-stand.mp4) · [raw paired results](baseline.json).

## CPU training

Eight parallel environments per variant, same seed 0, 1,048,576 PPO transitions each, BAM motors, no assistance or mirror loss. Stock completed in 95 seconds; loaded in 125 seconds, including startup/export. Normalization is baked into the exported ONNX policies. This is one training seed and a small sample budget.

The stock pilot learned to stand rather than walk. The loaded pilot collapsed onto its backpack: under the original height/tilt fall thresholds it appeared to survive all standing trials, despite using non-foot floor support in about 94% of steps. It fell in 7/8 forward trials. **Both pilot policies are rejected.**

A strict floor-support guard now rejects all 8/8 of the loaded pilot's standing, forward, and turning trials. The guard preserves the 61/14 interface, reward terms, and motor physics. A stock teacher rollout passes it; the backpack-supported pilot fails it. Both guarded trainers also completed a 16,384-step smoke run.

[Rejected training-policy video](trained-preview-loaded-stand.mp4) · [training summary](training_summary.json) · [strict evaluation](trained-loaded-strict.json) · [experimental checkpoints](policies/).

## Verification and next work

Masses, positive inertias, 14 actuators, 61 observations, policy timing, exact repeatability of isolated seed-0 rollouts, and the floor-support guard passed [verification](verification.json). Rendered contact sheets were inspected alongside the metrics. Setup was re-run successfully; an archived native checkpoint resumed for 2,048 guarded PPO steps and exported successfully. Fixed-pose servo targets without the walking policy fall in roughly 1.38 s stock / 0.64 s loaded, so the policy supplies essential stabilization.

Reduce the rearward payload moment, clear the moving head and limbs, then distill/fine-tune a competent walking policy with the stricter fall check. Longer runs should include payload uncertainty, observation noise, servo supply/sag variation, delays, pushes, and hardware parameter checks. Print flex, attachment slip, cable dynamics, thermal behavior, and battery runtime are outside this rigid-body pilot.

Early exploratory outputs without a construction seed were superseded by the reproducible paired results above; BAM samples supply parameters at construction, and resetting the episode seed alone does not fix those draws. Failed or backpack-supported policies were not labeled as walking successes.

[Mac setup and reproduction](../../training/README.md).
