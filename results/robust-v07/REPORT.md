# Mounted slope precision, push and trip/slip recovery

**Selected: existing_terrain_baseline.** The trained candidate changes average slope tracking error by **4.9% lower error**. Its final selection status is **rejected_due_to_regression**. All results are simulation only.

| Skill | Previous actor passes | Candidate passes | Previous safe | Candidate safe |
| --- | --- | --- | --- | --- |
| slope | 196/224 | 209/224 | 224/224 | 224/224 |
| push | 163/192 | 152/192 | 163/192 | 152/192 |
| trip_slip | 104/144 | 107/144 | 104/144 | 110/144 |
| regression | 64/64 | 64/64 | 64/64 | 64/64 |

[Selected ONNX](policies/selected/policy.onnx) · [Candidate](policies/candidate/) · [Summary](summary.json) · [Selection](selection.json) · [Validation](validation.json) · [Export and physics checks](verification.json) · [Runtime/checksums](run.json).

## Unchanged mounted robot

The v07 CAD includes the Jetson, 850 mAh pack, mount and wiring: 447.56 g added payload and 1.18480 kg nominal robot mass. Robot meshes, payload collision boxes and full inertials are retained; randomization varies the existing harness fields. The model uses BAM XL330 servos at 1.75 A, modeled bus delay, 200 Hz physics and 50 Hz policy. No spotter, balancing force, stronger motor, velocity teleport or get-up assist is enabled. Head pitch stays neutral. General fallen-pose recovery remains unreliable and disabled.

The previous [terrain study](../terrain-v07/REPORT.md) defines the 10 m heightfield, smooth spawn transitions, roughness bounds and geometry approximation. This study retains those definitions.

## Physical disturbance tests

Pushes apply external force at the trunk center: 2, 4 or 6 N for 120 ms, at 4 and 8 seconds, in opposite body-relative directions. Both fore/aft and side pushes are tested while standing and walking, on flat ground and 5 mm roughness. Actual applied durations and world-frame impulses are recorded, including interrupted or unreached second pushes.

A **simulated snag proxy** applies a 1, 2 or 4 N backward foot-force pulse for 80 ms to either foot during forward walking, on flat or rough ground. It triggers only during a forward swing with less than 0.05 N foot support and forward foot speed above 0.025 m/s, within a 1-second window beginning at 4 seconds. This models brief foot interference; it is **not an actual obstacle collision or a guarantee of curb clearance**. Actor-dependent swing timing is recorded; a missed or interrupted disturbance cannot count as a recovery pass.

Slip tests reduce **both foot and floor sliding friction** to 0.10, 0.20 or 0.35 for 500 ms, twice, then restore the episode's original friction. Self-checks verify effective contact friction and restoration. A slip exposure qualifies only if the patch was applied and supported contact-point tangential speed exceeds 0.05 m/s with at least 4 mm integrated peak contact-point sliding. This readout is a sampled sliding proxy, not a tracked sole material point. Unreached patches and patches without measured sliding remain visible and do not count as successful slip recovery. Safe runs with too little sliding to qualify can represent slip avoidance; their events are counted separately in the summary and are not described as falls. Forces and friction changes are environmental challenges; none are actor observations.

## Commands and recovery criteria

Every trial stands for 2 seconds, then uses the same inherited command ramp as the interactive controller. Slope tests start at [0.7,0.5] m beyond the flat spawn patch, translated to local ground plus 12 mm initial drop clearance. They test forward +0.15 m/s, reverse −0.10 m/s and turns ±0.40 rad/s on seven surfaces: 3°/5° up/down grades, both 3° cross slopes and mixed 5 mm/3° terrain. Push/trip/slip trials walk at +0.20 m/s or stand; ordinary controls additionally test stop, sidestep +0.10 m/s and left turning on flat/rough ground.

“Safe” means the full 12-second run stays upright with no sampled mount/internal or non-foot ground contact force above 0.05 N. Direction qualification uses mean motion after 3 seconds: each requested component must reach at least 25% of target with correct sign; unrequested translation stays within ±0.04 m/s and yaw ±0.12 rad/s. Slope precision reports the mean Euclidean error of each run's mean signed twist, normalized by [0.08 m/s,0.05 m/s,0.20 rad/s]. This is mean tracking, not exact path tracking or instantaneous gait smoothness.

Recovery requires the disturbance to be applied within one 20 ms control step of the requested duration and experienced, an entirely safe run, and a 0.5-second window within 3 seconds after the event: mean twist error per component ≤[0.08 m/s,0.05 m/s,0.20 rad/s], with tilt below 12° throughout that window. All planned events must qualify for the trial to pass. Mild shoves may never leave this tolerance; settle time can therefore be zero. Median settle times apply only to recovered events; full failures and unexercised events are also counted. No upright-survival count alone is presented as successful recovery.

## Training and independent tests

The actor retains all neural weights/normalization of the selected v07 terrain actor. Sixteen bounded CPU CEM parameters shape forward/reverse commands using observed body tilt, compensate lateral tilt, add yaw-rate feedback and mirrored leg pose/IMU feedback, and damp leg joint velocity during large observed tilt/gyro disturbances. Joint correction is clipped to ±0.12 rad. Inputs remain 61 observations and outputs 14 actions; no terrain heights, simulator velocity, contact state, disturbance timing or event labels enter the actor. This is frozen-network feedback calibration, not PPO weight training.

Six slope generations optimize ten parameters; eight joint recovery generations optimize all sixteen, sixteen candidates per generation. Slope and ordinary training runs last 8 seconds; disturbance runs last 12 seconds. Physics seeds 0/1 and terrain seeds 1000/1001 are used, with noise/DR on seed 1. CEM seed is 4401. Four workers run on the Mac CPU. Each skill has equal objective weight: 14×safe fraction +4×pass fraction −mean normalized tracking error −0.5×clipped post-event error, with a small parameter penalty. Missing post-event windows receive a failure penalty. All candidates and rejected searches remain in [curriculum/search.json](curriculum/search.json).

A follow-up [slip curriculum](slip-curriculum/search.json), initialized from the recorded early recovery checkpoint, adds three generations at friction 0.35/0.25, then three at 0.25/0.15, on flat and rough ground while preserving slope, push and ordinary controls. It uses sixteen candidates, CEM seed 4402, and the same paired training seeds. An exploratory IMU-triggered command-brake probe on separate calibration seeds 2/3 failed to improve slip passes (2/8 at each strength); it is retained but not added to the actor.

A separate validation comparison uses physics seeds 30/31 and terrain seeds 1030/1031, with noise/DR on seed 31. It evaluates baseline and 0.25/0.5/1 strengths of both stage checkpoints and the follow-up slip checkpoint on all 78 cases. The best equal-skill fitness is chosen among profiles without per-suite safety loss, ordinary-control pass loss or more than 2% worse slope error. The chosen parameters are frozen before final tests.

Final comparisons run **78 cases ×4 seeds ×2 regimes =624 trials per actor**, on paired scenes and rules. Nominal physics seeds are 100–103 and terrain seeds 1100–1103; noise/randomized seeds are 120–123 and 1120–1123. Seeded roughness changes independently of physics. Smooth slopes are deterministic. The previous actor's nominal/randomized files and the candidate files are retained separately.

Final acceptance requires: No per-skill safety regression; no push/trip pass or ordinary-regression pass decrease; lower slope error; more total robust passes; no core survival or flat-pass decrease; at most 10 fewer core direction passes; all 16 long sequences survive without sampled unwanted contacts. The acceptance criterion was prepared before final candidate results. The independent prior terrain benchmark additionally checks all 13 surfaces/11 commands: previous 1106/1144 direction/contact passes, candidate 1124/1144; candidate survival 1144/1144. Candidate long control sequences pass 16/16; each is 69 seconds. Per-segment signed motion remains in [sequences.json](sequences.json), and sequence pass checks survival/contacts rather than perfect tracking.

## Rejected follow-up profiles

The next eligible validation profile, **slope_0.25**, was frozen and compared against the previous actor on new paired scenes: nominal physics seeds 300–303/terrain 1300–1303 and randomized physics 320–323/terrain 1320–1323, 624 trials per actor. It is also **rejected**: push recoveries fall from 156 to 154/192 and safe trip/slip runs from 106 to 103/144. This fresh baseline differs from the first comparison; the two sets must not be mixed. [Plan](fallback-plan.json) · [Measured summary](conservative-summary.json) · [Rejected profile](policies/conservative/).

| Skill | Fresh previous passes | Conservative passes | Fresh previous safe | Conservative safe |
| --- | --- | --- | --- | --- |
| slope | 198/224 | 201/224 | 224/224 | 224/224 |
| push | 156/192 | 154/192 | 156/192 | 154/192 |
| trip_slip | 106/144 | 101/144 | 106/144 | 103/144 |
| regression | 64/64 | 64/64 | 64/64 | 64/64 |

Exploratory feature ablations and six observed-IMU blending guards used separate calibration seeds 2/3. The full candidate and all useful ablations/guards remained at 4/8 push passes; removing all new gyro terms or the loosest guard worsened it to 3/8. Trip/slip remained 4/6; ordinary controls 8/8. These small probes provided no push gain and are **not integrated or selected**. [Ablation measurements](ablation-probe.json) · [Blending measurements](blend-probe.json). They do not substitute for fresh full acceptance tests.

## Reproduce on the Mac

Follow the existing [setup](../../training/README.md), build v07 scenes and the final terrain grid, then run from the repository root:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/build_scenes.py --mount clearance_mount
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/terrain_env.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_robust.py --policy results/robust-v07/policies/candidate/policy.onnx --out tmp/robust-check.json
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_motion.py --revision v07 --policy results/robust-v07/policies/selected/policy.onnx --terrain downhill_3deg
```

Open http://127.0.0.1:8798; W/S walk, A/D sidestep, Q/E turn, Space stop. The required ramp and stale-input stop remain in the controller. The live page uses regular terrain, not hidden disturbance assistance.

To repeat without overwriting published results:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/tune_robust.py --out tmp/repeat-robust
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/tune_robust_slip.py --init tmp/repeat-robust/recovery-candidate/parameters.json --out tmp/repeat-slip
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/validate_robust.py --curriculum tmp/repeat-robust --slip-curriculum tmp/repeat-slip --out tmp/repeat-robust-validation.json --policy-out tmp/repeat-robust-policy
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_robust.py --policy tmp/repeat-robust-policy/policy.onnx --out tmp/repeat-robust-nominal.json --seed-start 100 --terrain-seed-start 1100
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_robust.py --policy tmp/repeat-robust-policy/policy.onnx --out tmp/repeat-robust-randomized.json --seed-start 120 --terrain-seed-start 1120 --noise-dr
```

## Continuous demonstration

[Video, top text removed](recovery-demo.mp4) · [Captioned recording](recovery-demo-captioned.mp4) · [Full trace](recovery-demo.json) · [Controller start/move/stop/reset check](controller-verification.json).

The video is one continuous fresh 15-second run of the **experimental full candidate, rejected as a replacement** on 5 mm roughness, with a 4 N side shove at 4 s, a 2 N right swing-foot pulse in the window starting at 8 s, and a 0.35-friction patch at 11 s. Physics seed 600 and terrain seed 1600 are separate from training, validation and final paired tests. It reports safety=True and recovery qualification=True; aggregate performance is above. Red arrows, where visible, illustrate applied forces and have no physics effect.

## Paired push failure demonstration

[Previous controller, 12 seconds](push-baseline.mp4) · [Rejected candidate, 8.8 seconds](push-candidate.mp4) · [Full traces and policy hashes](push-regression-demo.json) · [Reproduction](render_failure.py).

These render the same final-test scene `flat_forward_6N_y`, physics seed 100 and terrain seed 1100. Both receive opposite 6 N lateral shoves for 120 ms at 4 and 8 seconds while commanded forward at 0.20 m/s. The previous controller remains safe for all 12 seconds; the candidate reaches the height/tilt failure threshold at 8.8 seconds. Its recording stops there, with no padding or reset. This pair was chosen to illustrate a measured regression, not as an independent additional test.

To render the continuous success demo or this failure pair, run `results/robust-v07/render_demo.py` or `results/robust-v07/render_failure.py` with the setup Python from the repository root. Both overwrite their corresponding published recordings and traces.

## Limits

Strong disturbances still may cause falls. This is recovery during standing/walking, not reliable get-up from the floor. Foot-force proxies do not reproduce every trip; timed global friction drops do not represent all real patches. Collision/contact readouts are sampled at 50 Hz. Grip, mount strength/slip, motor heat, cables snagging externally, stairs and hardware are unvalidated. Camera navigation and perception are not added. A video is a single trial, not proof of general recovery.
