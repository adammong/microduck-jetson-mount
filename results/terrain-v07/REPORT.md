# Mounted uneven-terrain walking study

**Selected: candidate.** Candidate direction/contact-qualified passes: **1106/1144**, versus **1077/1144** for the prior v07 walking policy. Staying upright alone is reported separately: 1144/1144 candidate, 1144/1144 previous policy. Hardware is untested.

[Continuous simulation video](terrain-demo.mp4) · [Selected ONNX](policies/selected/policy.onnx) · [Candidate](policies/candidate/) · [Summary](summary.json) · [Selection](selection.json) · [Runtime and checksums](run.json).

## Terrain and unchanged payload

The robot uses the final v07 CAD payload: 447.56 g, total 1.18480 kg. Geometry, full inertials, head constraints, 1.75 A BAM XL330 servo limits, bus delay, 200 Hz physics and 50 Hz policy are retained. There is no spotter, extra balancing force or motor-strength increase. Mount collision shapes remain active.

Terrain replaces the existing named floor with a 641×641 heightfield covering 10×10 m. Flat controls retain the original plane and match the original simulation exactly in the self-check. Seeded roughness sums eight sine waves with 70–180 mm wavelengths; “2/5/10 mm” labels denote upper bounds on peak-to-peak relief, not measured constant obstacle heights. A smooth flat spawn patch extends 160 mm from the center, blending into roughness over 140 mm. Slopes enter smoothly beyond 160 mm and reach the named 3° or 5° grade; cross slopes and combined roughness/grades are included. Roughness uses independent terrain seeds; smooth slopes and thresholds are deterministic.

The 5 mm threshold is a heightfield rise at x≈350 mm, beveled over one 15.625 mm grid cell; it is **not a vertical stair**. Low grip is a uniform friction coefficient of 0.5 at both foot/floor surfaces; other trials retain harness friction behavior. Terrain-aware support guards detect non-foot floor support, trunk clearance below 70 mm relative to local ground, and the terrain edge. Ray-height checks verify axis/height agreement; coarse collision and surface sampling remain simulation approximations.

## Fresh paired tests

Each actor runs 13 terrain conditions ×11 commands ×4 seeds ×2 regimes = **1,144 trials**, each 15 seconds. Nominal physics seeds: 200–203, terrain seeds: 500–503. Noise/randomized physics seeds: 220–223, terrain seeds: 520–523. These are separate from training and correction-strength validation seeds. Noise/DR includes the pinned harness's mass, inertia, COM, friction, armature and voltage/sag variations. The same paired scenes and seeds are used for both actors.

| Terrain | Previous policy | Candidate | Candidate survival |
| --- | --- | --- | --- |
| flat | 87/88 | 88/88 | 88/88 |
| bumps 2mm | 85/88 | 88/88 | 88/88 |
| bumps 5mm | 85/88 | 88/88 | 88/88 |
| bumps 10mm | 87/88 | 88/88 | 88/88 |
| uphill 3deg | 75/88 | 80/88 | 88/88 |
| downhill 3deg | 88/88 | 88/88 | 88/88 |
| cross left 3deg | 80/88 | 83/88 | 88/88 |
| cross right 3deg | 86/88 | 87/88 | 88/88 |
| uphill 5deg | 73/88 | 78/88 | 88/88 |
| downhill 5deg | 82/88 | 77/88 | 88/88 |
| mixed 5mm 3deg | 81/88 | 87/88 | 88/88 |
| threshold 5mm | 84/88 | 86/88 | 88/88 |
| low grip | 84/88 | 88/88 | 88/88 |

“Pass” requires surviving the full trial, no detected mount/internal or non-foot ground support above 0.05 N at the evaluator's 50 Hz samples, and useful direction following. Every commanded nonzero twist component must reach ≥25% of its target with the correct sign; uncommanded translation must remain within ±0.04 m/s and yaw within ±0.12 rad/s, using mean motion after the first 2 seconds. This is a broad direction/drift criterion, **not precise tracking or path accuracy**. Signed measured twist and continuous error remain in every trial. Scaled error averages absolute tracking errors normalized by [0.2 m/s, 0.1 m/s, 0.4 rad/s].

Commands are stop, forward +0.20 m/s, reverse −0.15 m/s, sidesteps ±0.10 m/s, turns ±0.40 rad/s, curves [0.15,0,±0.30] and diagonals [0.15,±0.07,0]. Core tests use direct command steps and start in the flat center patch. Stop and turn-in-place trials can remain in that patch; they do not alone establish rough/sloped-ground standing or turning. [Long sequences](sequences.json) separately use the interactive controller's inherited ramp and fresh seeds. No general recovery/get-up is enabled when a terrain trial falls.

## Starts on terrain and longer control sequences

The selected actor survives **16/16 fresh 69-second sequences** without sampled mount/internal or non-foot support contacts, using flat, 5 mm bumps, 3° uphill and mixed 5 mm/3° ground, two nominal and two randomized trials per surface. These combine walking, reversing, sidesteps, turns, curves and stops after traversing terrain. Physics seeds are 400/401 and 420/421; terrain seeds are 600/601 and 620/621. Sequence pass counts measure survival/contact behavior; signed motion and errors for every segment remain in [sequences.json](sequences.json), and some stop/curve segments exceed the core drift criterion.

A separate [on-terrain cold-start test](on-terrain-starts.json) starts at [0.7, 0.5] m, beyond the flat patch, on 5 mm bumps, 3° uphill, 3° left cross slope and mixed terrain. Each actor runs stop and both turns, with two nominal and two noisy/randomized seeds, for **48 trials per actor**. The initial standing pose is translated to local ground height plus 12 mm of drop clearance, with no forces or pose corrections after reset. Trials stand for 2 seconds, then hold the requested command for 10 seconds. Physics seeds are 800/801 and 820/821; terrain seeds are 1800/1801 and 1820/1821. Both actors survive all 48 without sampled unwanted support/contact. Direction/drift qualification improves **44/48 →45/48**, using the core thresholds on mean motion after warmup. Three candidate right turns on the left cross slope stall below the 25% yaw criterion; stopping and the other tested surfaces pass this limited test.

The improvement is not uniform: core **5° downhill passes regress 82/88 →77/88**, despite all runs staying upright. Overall normalized tracking error decreases by 2.45%; directional control on slopes still needs work. Parameters remain frozen throughout these follow-up tests.

## Training and selection

The actor retains the complete previous v07 neural graph, normalizer and calibration. Eight CPU CEM parameters add mirrored hip/knee/ankle pose bias, ankle gravity/gyro feedback, hip-roll gravity/gyro feedback and yaw-command/gyro feedback. Correction is limited to ±0.12 rad and smoothly gated by command magnitude; zero commands preserve the original actor exactly, including its standing catch behavior. Head/neck outputs remain unchanged and neutral. **The actor receives only the existing 61 observations**, including IMU, joint state and commands; it receives no terrain height map, simulator velocity, camera image or contact oracle. Output remains 14 joint actions.

Training used the original 8×8 m, 513×513 grid. Final comparisons extend it to 10×10 m at the same 15.625 mm spacing, preserving central heights; a self-check compares all terrain types. Initial 8 m diagnostics are retained in `diagnostic-8m/`: six downhill-forward runs reached the terrain boundary, rather than falling. Both final actors are reevaluated on the extended grid.

Curriculum: 3 generations on flat/small 2 mm bumps, then 8 generations on 5 mm bumps, 3° slopes and mixed roughness. Each generation evaluates 20 candidates. Early cases: 4 commands/terrains ×2 seeds; later cases: 14 ×2. Training physics seeds 0/1, terrain seeds 10/11; seed 1 includes noise/DR. The recorded CEM seed is 118. Objective rewards contact/direction-qualified passes, completion, tracking and low tilt, with a small parameter penalty. All candidates and training trials are retained in [curriculum/search.json](curriculum/search.json). This is calibrated frozen-network feedback, **not PPO neural-weight fine-tuning**.

A separate [validation grid](validation.json) chooses correction strength from 0, 0.25, 0.5, 0.75 and 1, using physics seeds 100/101 and terrain seeds 1000/1001, with noise/DR on seed 101. It selects by direction-qualified passes and tracking error without reducing flat validation passes. Parameters are frozen before final tests. Final acceptance requires improved aggregate passes, no aggregate survival or flat-pass regression, and no more than 2% worse average tracking error. [Export/geometry self-checks](verification.json) verify 1,000 wrapper/ONNX comparisons, unchanged inherited tensors, unchanged input arrays, the correction limit, identical stopped/head outputs and preserved command-ramp metadata.

## Run on the Mac

Follow the existing [Mac setup](../../training/README.md), then from the repository root:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/build_scenes.py --mount clearance_mount
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/terrain_env.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_terrain.py --policy results/terrain-v07/policies/candidate/policy.onnx --out tmp/terrain-check.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_motion.py --revision v07 --policy results/terrain-v07/policies/selected/policy.onnx --terrain bumps_5mm --terrain-seed 700
```

Open http://127.0.0.1:8798; W/S forward/back, A/D sideways, Q/E turn and Space stop. Replace `bumps_5mm` with `uphill_3deg`, `cross_left_3deg` or another named surface. The controller retains stale-input stopping, shows the selected surface and stops at the finite terrain boundary.

The retained benchmark drivers overwrite the study folder; copy it before repeating them. To repeat training without overwriting published results:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/terrain_env.py --training-grid
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/tune_terrain.py --out tmp/repeat-terrain --early-generations 3 --generations 8 --population 20
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/validate_terrain.py --search tmp/repeat-terrain/search.json --out tmp/repeat-terrain-validation.json --policy-out tmp/repeat-terrain-policy
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/terrain_env.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_terrain.py --policy tmp/repeat-terrain-policy/policy.onnx --out tmp/repeat-terrain-nominal.json --seed-start 200 --terrain-seed-start 500 --seeds 4 --seconds 15
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_terrain.py --policy tmp/repeat-terrain-policy/policy.onnx --out tmp/repeat-terrain-randomized.json --seed-start 220 --terrain-seed-start 520 --seeds 4 --seconds 15 --noise-dr
```

To repeat the additional checks without replacing results:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_terrain_stationary.py --out tmp/repeat-terrain-starts.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_terrain_sequences.py --policy results/terrain-v07/policies/selected/policy.onnx --out tmp/repeat-terrain-sequences.json
```

## Limits

Small smooth roughness and gentle grades are tested; stairs, sharp risers, loose gravel, deformable carpet/grass, obstacles, vision-based foothold planning and arbitrary terrain are not. Finite heightfields and approximate collision geometry do not certify real grip or foot placement. Existing shell fit, mount-arm strength, servo heating and electrical integration remain untested. A simulation video is one run; aggregate direction passes do not imply precise tracking or safe hardware autonomy. Fall recovery remains unreliable and disabled.
