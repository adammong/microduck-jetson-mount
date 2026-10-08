# v06 Mac simulation report

**The mount redesign substantially improves nominal balance and forward walking with the unchanged official actor. Neither PPO fine-tuning attempt is selected. This is a fit/simulation prototype, not a hardware-ready walking system.**

## What changed

The Jetson moves low in front of the torso, with the connector edge upward; the 73 g battery moves behind it. Separate corner retainers replace the bar that touched the head. The tray has larger weight-saving windows, and wiring follows the new positions. The two-screw torso capture and two-screw removable equipped tray remain. Five print parts replace the old four; obsolete top-gate print files were removed.

CAD validation reports five valid single solids and watertight STLs, no native print/print, battery/print or wire/print overlaps, and a clear 50 mm plug-access corridor at sampled poses. All five print snapshots and the equipped/full-robot previews were rendered and reviewed. No head bar remains, and ports are visibly upward and accessible in the reference pose.

Payload estimate falls from **451.3 to 442.1 g**; fully dense PETG falls from **129.3 to 120.1 g**. Loaded reference mass is **1.1793 kg**, about 60% above stock. Ancillary hardware is now distributed around its approximate real fastening sites rather than placed in one lump. Mass is estimated, not weighed.

Static HOME COM changes from 12.1 mm behind stock to **2.1 mm ahead**, with a 9.0 mm lower height. Optimistic rear sole margin increases from **8.5 to 22.7 mm** (stock 20.5 mm). These footprint estimates assume contact over the projected sole and do not prove balance.

## Matched CAD comparison

Pinned official `alpha_walking.onnx`, construction/reset seeds 0–7, requested 20 seconds per trial, nominal BAM XL330 motor physics, no assistance, active full groundcontact and mount collision boxes. Strict termination rejects sustained non-foot support. Observation noise and communication delay are off in this first comparison. Commands are zero for standing and +0.30 m/s for forward. Every trial uses a fresh model copy.

| Robot | Stand survival | Mean stand X drift, m | Forward survival | Mean forward X travel, m | Mean forward X speed, m/s |
|---|---:|---:|---:|---:|---:|
| Stock | 8/8 | 0.0016 | 8/8 | 2.059 | 0.103 |
| Loaded v05 | 7/8 | -0.8065 | 8/8 | -0.177 | -0.009 |
| Loaded v06 | 8/8 | 0.0031 | 8/8 | 3.191 | 0.160 |

One v05 standing trial falls early; its endpoint is included in the mean. All v06 nominal standing, forward and turn-command trials survive: **24/24**. Standing and forward trials have zero recorded loaded-payload self contact and zero non-foot ground frames. Forward speed still undershoots the 0.30 m/s command.

**Survival is not task success.** The teacher barely moves on reverse, lateral and pure turning commands. Turn survival in the baseline table must not be described as successful turning. The extended measurements include yaw displacement so this failure is visible.

## Delay, noise and randomized tests

Each suite uses eight seeds, eight commands and 20 seconds: stand, forward, reverse, left/right lateral, left/right pure turns and a forward curve.

- With communication delay and observation noise: teacher **64/64 survives**, mean forward X speed about **0.151 m/s**. One forward trial has a brief mount contact (0.1% of control frames; 0.59 mm maximum penetration), so nominal contact-free results are not a guaranteed clearance envelope.
- With the harness's additional domain randomization: teacher **59/64 survives**, including **6/8 forward** and **7/8 curved** trials. Some falls/contacts remain. This is a diagnostic of the harness's perturbations, not a calibrated payload-mass or motor-temperature robustness study.
- Reverse and lateral speed remains near zero; pure-turn yaw rate remains near zero without domain randomization. The baseline actor does respond to a combined forward/turn command, but it does not reliably cover every requested motion.

Full joint-limit sampling still records **140 interfering pairs**, including large head/leg motions. The narrower HOME ±0.2 rad one-joint sample has no penetration beyond the 0.2 mm reporting threshold; its smallest convex gap is only **0.51 mm**. Combined articulation and flexible cable travel are unvalidated. Do not treat this mount as compatible with the entire stock motion range.

## Exact actor import and CPU fine-tuning

The ONNX normalizer and 512/256/128 ELU actor are copied directly into native PPO. Only the critic is fitted initially, using 40,000 states over 80 mounted teacher episodes. Actor equality is asserted before/after critic fitting: maximum Torch-vs-original action difference **7.2e-7 rad** on collected states. A separate 1,000-state test finds **zero exported ONNX difference**. Warm-start normalizer means and variances, including command slots, remain exactly equal. The 61-observation / 14-action, 50 Hz contract and BAM limits are unchanged.

Two bounded CPU studies use eight workers, nominal geometry/mass, no observation noise during training, bus delay and startup voltage/sag sampling. No symmetry or assistance is used. Both use the strict floor-support guard:

| Actor | Additional PPO steps | Training seed | Learning rate | Training time | Forward survival with delay/noise | Mean forward X speed |
|---|---:|---:|---:|---:|---:|---:|
| Teacher | 0 | — | — | — | 8/8 | 0.151 m/s |
| Short fine-tune | 262,144 | 0 | 1e-5 | 33.8 s | 8/8 | 0.040 m/s |
| Longer fine-tune | 1,048,576 | 1 | 3e-6 | 155.0 s | 7/8 | 0.029 m/s |

Fine-tuned actors veer/turn during a forward command, and fail to improve reverse/lateral/pure-turn behavior. Both are **rejected**. Different training budgets/seeds prevent attributing their difference to learning rate alone. Paired evaluation favors retaining the unchanged teacher as the simulation baseline. Checkpoints are published for reproducibility with explicit status metadata, not deployment approval.

Standing, forward and pure-turn rollouts were recorded and their contact sheets reviewed for both the teacher and the longer fine-tune. The teacher shows alternate foot motion and forward travel; pure turning shows essentially standing. The longer fine-tune visibly turns during a requested straight walk. No spotter or external stabilizer was used.

## Files and reproducibility

- [Baseline](baseline.json), [v05 control](v05-control.json), [comparison summary](comparison.json).
- [Noise/delay](teacher-robustness.json), [domain randomization](teacher-randomized.json).
- [Short fine-tune](finetuned.json), [longer fine-tune](finetuned-long.json).
- [Mass/interface checks](verification.json), [teacher import checks](teacher-import-verification.json), [scene hashes](scenes.json).
- [Checkpoint artifacts and statuses](policies/), [checksums](policy_checksums.json), [training metadata](training_summary.json), [teacher provenance](teacher-provenance.json).
- [Teacher forward video](preview-loaded-forward.mp4), [teacher contact sheet](preview-loaded-forward.png), [rejected fine-tune video](finetuned-preview-loaded-forward.mp4).
- [Setup and commands](../../training/README.md), [CAD validation](../../easy_mount/simulation/validation.json).

Hardware fit, mount stiffness/slip, printed strength, actual mass/inertia, power electronics, flexible cable loads and motor heat remain unmeasured. Training and testing happened on an Apple Silicon M5 Mac using native CPU simulation. The preferred next experiment is a calibrated payload/actuator randomization study and an explicit command-tracking curriculum, after a physical fit check; no hardware test was performed here.
