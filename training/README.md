# Local Mac simulation and training

Uses [MicroDuck Lab](https://github.com/jonathanhawkins/microduck-lab/tree/bbf0326ef97975f7062368914e0729504d17226f), pinned to `bbf0326ef97975f7062368914e0729504d17226f`. Native CPU MuJoCo and PPO work on Apple Silicon; CUDA is unnecessary for these experiments.

The explicitly selected CAD revision is the source of payload mass, inertia and collision boxes. Stock and loaded scenes share the robot, floor, STAND keyframe, BAM XL330 motor model, 200 Hz physics and 50 Hz policy interface: **61 observations / 14 actions**. Both use the fuller groundcontact robot. The Jetson visual is an envelope; cables are fixed and the mount cannot slip or break in this model.

## Slope precision and push/trip/slip recovery

The [mounted robustness study](../results/robust-v07/REPORT.md) adds measured external torso shoves, swing-foot snag proxies and temporary low-grip patches. Applied force impulses, effective contact friction, sliding exposure and return-to-command windows are checked separately from survival. Two initial CEM stages and a progressive slip curriculum train a frozen-network adaptation, with independent validation/final comparisons and ordinary-terrain regression checks. Selected and rejected profiles are labeled; hardware and general fallen-pose get-up remain unvalidated.

Use the selected actor with the existing terrain control page:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_motion.py --revision v07 --policy results/robust-v07/policies/selected/policy.onnx --terrain downhill_3deg
```

The report provides scene setup, training, validation and exact disturbance tests. These are simulated environmental challenges, never hidden balance assistance or actor inputs.

## Uneven-terrain study

The [mounted terrain study](../results/terrain-v07/REPORT.md) adds seeded small bumps, gentle slopes/cross slopes, mixed roughness, a beveled threshold and low-grip conditions. It retains the current mount, servo limits, 61/14 interface and neutral head constraint. Two-stage CPU CEM trains bounded IMU/command corrections; separate validation selects their strength before paired final tests. Selected and rejected artifacts remain clearly labeled. No general fall recovery or real-world terrain claim is made.

After setup and building the v07 scenes:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/terrain_env.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_terrain.py --policy results/terrain-v07/policies/candidate/policy.onnx --out tmp/terrain-check.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_motion.py --revision v07 --policy results/terrain-v07/policies/selected/policy.onnx --terrain bumps_5mm --terrain-seed 700
```

Open http://127.0.0.1:8798; the surface is shown beside movement measurements. Terrain seeds independently change roughness. The controller stops at the finite terrain boundary. Omit `--terrain` for the existing flat-floor controller. The core stop/turn tests begin in a flat spawn patch; separate cold starts and long sequences exercise actual rough/sloping ground. See the report for exact tests, thresholds, training commands and scope.

## Current v07 mount

The [clearance study](../results/posture-v07/REPORT.md) selects separate sit/stand and walking ONNX actors. Fresh tests pass 16/16 sit/stand cycles and 88/88 walking trials. Head pitch stays neutral. General recovery remains rejected; the live controller does not enable automatic recovery or posture switching. v06 policy defaults and historical results below are preserved.

After the Mac setup command below, build the current scenes explicitly:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/build_scenes.py --mount clearance_mount
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_clearance.py --out tmp/v07-check.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_motion.py --revision v07 --policy results/posture-v07/policies/walking-adjusted/policy.onnx
```

Open http://127.0.0.1:8798. The walking actor retains velocity commands and the inherited command ramp/watchdog. The sit/stand actor instead reads observation slot 48 as a posture flag: 0=stand, 1=sit. Never use that actor in the walking command page. CAD and simulation alone do not validate real assembly, servo heat or hardware control.

## Run

From the repository root, with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```sh
uv run --python 3.12 training/setup_mac.py
tmp/microduck-lab/microduck_local/.venv/bin/python training/verify.py
tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate.py --strict --seeds 8 --seconds 20 --render
tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate.py --variant loaded --strict --extended --bus-delay --obs-noise
```

The legacy commands below use the revision in `easy_mount/parameters.json`: v06 metrics/videos go into `results/mac-v06/`, and compiled scenes into `tmp/mac-models/v06/`. `--revision v05` evaluates preserved local v05 scenes if present; a fresh checkout must build those from the historical CAD revision first. Existing historical results are committed in `results/mac-v05/`.

## Exact teacher import and bounded fine-tuning

The official ONNX actor can now be copied exactly into a resumable PPO checkpoint. Its normalizer and ELU layers are retained; only the initial critic is fitted to mounted rollouts. Equality is asserted before and after critic fitting. This avoids approximate actor cloning and preserves normalized command slots during later warm starts.

```sh
tmp/microduck-lab/microduck_local/.venv/bin/python training/import_teacher.py --episodes 80 --critic-epochs 20
tmp/microduck-lab/microduck_local/.venv/bin/python training/train_pair.py --variant loaded --steps 262144 --init-from tmp/mac-runs/v06-teacher-import-s0 --lr 0.00001 --run-tag teacher
tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_teacher.py --warm-run tmp/mac-runs/v06-loaded-bam-s0-262144-strict-warm-teacher
```

The training driver runs bounded paired numerical studies with eight workers, CPU PPO and a strict floor-support guard. It uses nominal mass and no observation noise; BAM still samples startup voltage/sag and uses bus delay. This is not the full hardware-transfer recipe. Use the upstream live LAB for general skill training. Logs, checkpoints and exports are stored in `tmp/mac-runs/`. `--seed`, `--run-tag`, `--lr` and `--init-from` control repeat studies. Existing outputs are preserved; choose a fresh run tag to repeat. Archived checkpoints use `model.sb3`; the driver restores this native ZIP container to temporary `model.zip` before loading it.

The guard ends an episode after three successive control steps with a non-foot floor contact carrying more than 0.05 N. Observations, actions, rewards and motor limits remain intact. `--allow-nonfoot-support` exists only to reproduce the historical backpack-supported failure. Evaluation defaults to nominal physics, with optional communication delay, observation noise and domain randomization. Trials load isolated model copies and seed both construction and reset.

## Results and selection

The [mounted posture study](../results/posture-v06/REPORT.md) adds sit/stand and settled-floor recovery evaluation, CEM adaptation, physical-fall recordings and an exported front-recovery candidate. Front recovery improves 3/16 → 14/16, but full sitting and synthetic back/side recovery remain unresolved due to mount interference. A separate recorded-fall benchmark improves only 7/16 → 8/16. Rejected seated-pose searches and successful/failed videos are retained. The existing walking controller is not modified to enable these unfinished skills.

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_posture.py --policy results/posture-v06/policies/front-recovery-candidate/policy.onnx --out tmp/recovery-preview.json --poses front --seed-start 251 --seeds 1 --render
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_posture.py --out tmp/posture-check.json
```

Setup now fetches both official posture actors at their pinned revision/checksums in [posture_inputs.json](posture_inputs.json). Sit/stand uses slot 48 as a 0/1 posture flag, not a walking velocity. The dedicated posture evaluator permits floor contact during recovery and checks an unsupported full stand at completion; the walking evaluator's support termination would prevent a get-up. These are simulation experiments; no hardware or general-recovery claim is made.

The [mounted precision/recovery study](../results/precision-v06/REPORT.md) trains continuous command shaping, gait amplitude, observable low-turn feedback and standing catch steps while preserving the original neural actor. Final paired tests show 45% lower slow-command tracking error, 464/464 non-push trials without detected contacts, and modest push survival improvement from 74% to 77%. Sixteen fresh full control sequences pass. Tiny translations and stronger push recovery remain unresolved. This is CPU grid/CEM calibration, with all searches and rejected candidates retained.

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_motion.py --policy results/precision-v06/policies/selected/policy.onnx
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_precision.py --folder results/precision-v06/policies/selected --out tmp/precision-check.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_precision.py --policy results/precision-v06/policies/selected/policy.onnx --out tmp/precision-cases.json --suite all --seed-start 600 --seeds 8
```

The control page's Speed slider scales keyboard/button commands. Try 50% translations or 25% turns; the smallest translations can stall. ONNX metadata selects the required gentler command acceleration `[0.25, 0.20, 1.0]`; other runtimes must implement it and the 0.6-second watchdog explicitly. The old ramp caused a retained sequence failure before this controller change. Full final evaluation, rendering, search and export instructions are in the report. Hardware is untested; the default interactive actor remains the earlier directional policy.

The [directional-motion study](../results/motion-v06/REPORT.md) adds a simulation controller for forward/backward, both sidesteps, both turns, diagonals and curves, plus a keyboard/button page. Six command-conditioned maps and bounded gyro/gravity corrections retain the teacher's exact neural weights and normalization. Paired CEM studies also calibrate turn startup from settled standing and a gyro-based start boost. Final checks use the same command slew as the page: 176/176 fresh 30-second nominal/randomized trials and 8/8 full 69-second sequences, with separate direction-tracking checks and no detected mount contact. Low-command dead zones and hardware transfer remain unresolved.

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_motion.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_motion.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_motion_batch.py --policy results/motion-v06/policies/selected/policy.onnx --out tmp/directional.json --seeds 8 --seed-start 200 --seconds 30 --slew
```

Open http://127.0.0.1:8798. W/S forward/back, A/D sideways, Q/E turn, Space stop. Body-frame presets are +0.20/-0.15 m/s forward/back, ±0.10 m/s sideways and ±0.40 rad/s yaw; movement/turn keys combine. The controller requires its tested command ramp and 0.6-second stale-input watchdog. It exports one 61-input/14-output ONNX; CEM parameters and all searches are retained. This is a bounded numerical calibration study, rather than another PPO fine-tune. Earlier candidates, cold-start failures and missed turns remain in the report.

The [faster-walking study](../results/speed-v06/REPORT.md) selects a **simulation-only** feedback correction policy for forward command `[0.5, 0, 0]`. It averages 0.266 m/s net forward progress, with 16/16 fresh 30-second nominal/randomized trials surviving without detected mount contacts. The original teacher network and normalization are retained; bounded gyro/gravity corrections are trained with CEM. Normal/idle commands at or below 0.35 retain the original teacher's output exactly. The study records the rejected 2M/3M-step LAB PPO attempts and the complete bounded calibration history.

```sh
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/tune_fast_feedback.py --out tmp/repeat-fast-feedback --generations 8 --population 20 --command .5 --seconds 25 --training-seeds 0 1 2 3 4 5 6 7 8 9 10 11
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_speed.py --policy results/speed-v06/policies/selected/policy.onnx --out tmp/recheck-fast.json --commands .5 --seeds 8 --seed-start 20 --seconds 30 --render
```

The selected graph still uses 61 observations / 14 actions at 50 Hz. It is an ONNX policy, with parameters and provenance under `results/speed-v06/policies/selected/`; it is not a resumable PPO checkpoint. CEM resumes conceptually from its recorded parameters/search, and PPO comparison checkpoints remain separately archived. Published measurements validate the 0.5 forward operating point, not higher speeds or general turning. See the report for sampling limits and hardware gaps.

[The v06 report](../results/mac-v06/REPORT.md) shows a substantial **mount balance improvement** with the unchanged walking teacher. The teacher remains the preferred comparison actor: both fine-tuning attempts failed to beat it reliably. Their checkpoints are marked rejected. Surviving a reverse, lateral or turn-in-place trial does not mean following its command; these motions remain ineffective. Randomized forward/curved trials still expose falls and occasional mount contacts.

The [v05 pilots](../results/mac-v05/REPORT.md) are also rejected. Retained artifacts support reproducibility, not deployment. Printed mass, motor heating, shell strength and real fit remain unverified. Inputs and checksums are in [inputs.json](inputs.json); scene source hashes in [scenes.json](../results/mac-v06/scenes.json). The pinned harness's `uv.lock` fixes training dependencies; CAD dependencies remain separate.
