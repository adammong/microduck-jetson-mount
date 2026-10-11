# v07 clearance, posture and walking study

The revised mount raises the front Jetson and rear battery to clear folded legs. Separate calibrated sit/stand and walking actors pass the fresh tests below. **General floor recovery remains unreliable and the recovery candidate is rejected for that purpose.** These are Mac CPU simulations; no hardware deployment or automatic recovery/posture switching is enabled.

[Current CAD and attachment instructions](../../clearance_mount/README.md) · [Continuous 19-second sit/stand video, top text removed](sitstand-demo.mp4) · [Summary](summary.json) · [Selection](selection.json) · [Checksums/runtime](run.json).

## Final geometry and physics

Jetson tray: +40 mm forward and +33 mm upward from v06, ports upward. Battery pocket: 61 mm rearward and 35 mm upward. Raised side arms replace the low arm/dock interference. Front hooks are on the removable gate; torso screw bosses sit outboard at ±40 mm. Remove the gate, slide the saddle forward from behind, then fit the gate from ahead. The OEM Jetson base slides into a separate tray, retained by two corner gates. Six thumb screws serve the torso, removable tray and corner gates.

The five saved STEP parts are valid single solids, with watertight print STLs. There are no detected print overlaps or battery/wiring/connector-corridor intersections. Shell-hook surface samples give ≥1.24 mm clearance; 5 mm insertion-path samples pass. HOME ±0.2 rad single-joint samples and a 50 mm straight connector corridor pass. **Full joint travel still collides.** These samples cannot certify combined motion, grip, strength, production tolerances or real assembly. [Detailed checks](../../clearance_mount/simulation/validation.json), [attachment path](../../clearance_mount/simulation/attachment_check.json).

Payload is 447.56 g, including 125.56 g fully dense PETG, 175 g Jetson, 73 g GNB8504S60AHV battery and 74 g ancillary allowance. Total robot mass is 1.18480 kg, about 5.46 g more than v06. CAD volume/COM and full inertia feed the rigid payload model. Every constituent inertia is checked for positive principal moments and the physical triangle inequality. Cable routes use short positive-volume sweeps: a final check rejected a topologically valid long sweep with invalid inertia, and all final tests were repeated after its correction. Cable/plug density and specified-mass envelopes remain approximations; cable flex, mount slip and breakage are absent.

Tests use the pinned MicroDuck Lab harness, MuJoCo 3.10.0, 1.75 A BAM XL330 limits, 200 Hz physics, 50 Hz policy, bus delay and **no spotter, balancing force or motor-strength boost**. Randomized trials include observation noise and harness friction/mass/inertia/COM/armature/voltage/sag variations. Sit/stand additionally tests ±10% payload mass/inertia.

## Fresh final tests

| Skill/test | Nominal | Noise/randomized | Interpretation |
| --- | --- | --- | --- |
| Sit/stand cycle | 8/8 | 8/8 | Full floor-supported sitting then unsupported standing; no mount contact throughout |
| Walking, 11 commands × 4 seeds | 44/44 | 44/44 | 15 seconds each, no detected mount contact or non-foot floor support |
| Synthetic fallen recovery, candidate | 1/32 | 0/32 | Unreliable |
| Synthetic fallen recovery, official baseline | 1/32 | 0/32 | Candidate does not improve general recovery |

The same selected sit/stand actor on the unchanged v06 geometry passes **0/8** paired nominal cycles. This comparison supports the clearance redesign, not a claim that geometry alone is sufficient.

Sit/stand seeds are 200–207 nominal and 220–223 at payload scales 0.9/1.1 randomized. Each 19-second cycle stands for 3 seconds, requests sitting until second 11, then stands until second 19. Sitting must hold through seconds 8–11: trunk below 70 mm, tilt below 35°, robot-body floor support and no mount support/interference. Completion requires both feet supporting, no other floor prop or mount interference, tilt below 20°, trunk ≥95 mm and jaw ≥180 mm in the last 2 seconds. Mount contacts fail the entire cycle. The selected actor holds neutral neck/head pitch (default 0.3491 rad); the original head-down SIT pose remains incompatible.

Walking uses stop, forward +0.20 m/s, reverse −0.15 m/s, sidesteps ±0.10 m/s, turns ±0.40 rad/s, curves [0.15,0,±0.30] and diagonals [0.15,±0.07,0]. Seeds are 200–203 nominal and 220–223 randomized; commands step directly in this benchmark. [Per-command measured motion](summary.json) records cross-axis drift and slower curves/diagonals. Survival alone does not guarantee precise tracking. Tiny-command dead zones, strong pushes, long control sequences and high-speed walking were not retested on v07. Earlier v06 counts do not transfer to this mount. A separate interactive start/move/stop smoke check passes with the correct v07 mass and actor.

Recovery uses settled synthetic front/back/left/right starts, 8 trials per direction per regime, 10 seconds per trial. Successful recovery requires a full stand held for ≥3 seconds and throughout the last 2 seconds. Eight already-standing trials per regime are reported separately and excluded from the fallen denominators. The candidate and official baseline both recover 11/16 recorded tip-over poses; limp control recovers 0/16. Recordings use a 0.35 Nm impulse for 0.2 seconds followed by 23.8 seconds settling, then replay at zero root velocity. 13/16 are collapsed poses below 20° tilt; this set is not equivalent to flat front/back/side falls. Replay seeds 940–955 are fresh; tip recordings use 900–903. The candidate provides no demonstrated general-recovery improvement, so no automatic get-up is enabled.

## Training and artifacts

Frozen official `alpha_stand`/`alpha_sitstand` actors retain their neural tensors and normalization. CPU CEM calibrates bounded pose/gyro/gravity feedback and sitting depth; this is not PPO network fine-tuning. The selected sit/stand search jointly tunes 13 parameters, 10 generations ×24 candidates, seeds 0–3, with seeds 1/3 randomized. It passes 4/4 training cycles. Earlier 8×20 seated searches and the 10×24×8 gravity-conditioned recovery search remain archived. Recovery achieves only 2/8 training starts and fails broad fresh evaluation.

Walking retains the previous precision actor and velocity-command interface, holds the head neutral and adds a bounded standing correction. A five-scale grid (0, 0.25, 0.5, 0.75, 1) covers five commands ×2 training seeds for 8 seconds, selecting scale 0.75 by tracking error. All inherited neural tensors are unchanged. Wrapper/export checks use 1,000 observations, preserve input arrays and agree below 1e−6 rad. [Verification](verification.json), [additional exports](additional-export-verification.json), [controller smoke check](controller-verification.json).

Searches ran on an earlier v07 iteration, before the hooks moved to the gate and cable sweep inertia was corrected. Its exact self-contained compiled MuJoCo 3.10.0 scene is retained as `training-model/scene_walk.mjb.gz` (~50 MiB), with parameters, inertials, checksums and prototype evaluations. Three archived-scene replay checks match original metrics and final joint positions exactly. Its old cable inertia approximation is not the final model. Final tests and the video use the rebuilt final geometry and corrected inertia; source/design fixes were followed by fresh reevaluation, without selecting new policy parameters on held-out seeds. The compiled training scene contains upstream geometry and retains its applicable terms.

Selected exports: [sit/stand](policies/sitstand/), [walking](policies/walking-adjusted/). The recovery export is retained as [rejected general recovery](policies/recovery-candidate/). Baselines, searches and prototype evaluations are retained for comparison. Both selected actors use 61 observations/14 actions, but **slot 48 means sit/stand flag for posture and forward velocity for walking**. Do not swap them into the same controller without explicit interface/state management.

## Reproduce on the Mac

Follow [setup](../../training/README.md); install CAD requirements separately. From the repository root:

```sh
CAD_PYTHON=.venv/bin/python sh clearance_mount/build.sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/build_scenes.py --mount clearance_mount
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_clearance.py --out tmp/recheck-v07.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_posture_batch.py --revision v07 --policy results/posture-v07/policies/sitstand/policy.onnx --poses sitstand --seed-start 200 --seeds 8 --out tmp/sitstand-v07-nominal.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_posture_batch.py --revision v07 --policy results/posture-v07/policies/sitstand/policy.onnx --poses sitstand --noise-dr --payload-scales .9 1.1 --seed-start 220 --seeds 4 --out tmp/sitstand-v07-randomized.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_motion_batch.py --revision v07 --policy results/posture-v07/policies/walking-adjusted/policy.onnx --seed-start 200 --seeds 4 --seconds 15 --out tmp/walking-v07-nominal.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_motion_batch.py --revision v07 --policy results/posture-v07/policies/walking-adjusted/policy.onnx --noise-dr --seed-start 220 --seeds 4 --seconds 15 --out tmp/walking-v07-randomized.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_motion.py --revision v07 --policy results/posture-v07/policies/walking-adjusted/policy.onnx
```

Open http://127.0.0.1:8798 for walking. The page reads the inherited command ramp and watchdog metadata. For posture video, use `training/evaluate_posture.py` with the same posture arguments, `--seeds 1 --seed-start 250 --render` and a fresh output path. `recovery_benchmark_source.py` records the exact final paired recovery jobs; it writes this results folder, so copy the study before repeating. `walking_grid_source.py` records the scale grid.

For exact original search reproduction, first back up `tmp/mac-models/v07/loaded/src/mjlab_microduck/robot/microduck/scene_walk.mjb`, replace it with the decompressed archived training scene, run `training/tune_clearance.py --out tmp/repeat-v07-search --generations 10 --population 24`, and restore/rebuild final v07 scenes afterward. The archive is MuJoCo-version-specific and represents the documented training iteration, not printable final CAD.

## Limits

Physical shell fit, screw grip/torque, long battery-arm strength, infill/support removal, thermal endurance and electrical integration remain untested. The illustrated blue cable endpoint is free/unverified; the converter is an unselected fit allowance. Full head/joint motion and general fall recovery remain unresolved. Neither simulation pass rates nor a successful video authorize real autonomous get-up.
