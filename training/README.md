# Local Mac simulation and training

Uses [MicroDuck Lab](https://github.com/jonathanhawkins/microduck-lab/tree/bbf0326ef97975f7062368914e0729504d17226f), pinned to `bbf0326ef97975f7062368914e0729504d17226f`. Native CPU MuJoCo and PPO work on Apple Silicon; CUDA is unnecessary for these experiments.

The current CAD revision is the source of payload mass, inertia and collision boxes. Stock and loaded scenes share the robot, floor, STAND keyframe, BAM XL330 motor model, 200 Hz physics and 50 Hz policy interface: **61 observations / 14 actions**. Both use the fuller groundcontact robot. The Jetson visual is an envelope; cables are fixed and the mount cannot slip or break in this model.

## Run

From the repository root, with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```sh
uv run --python 3.12 training/setup_mac.py
tmp/microduck-lab/microduck_local/.venv/bin/python training/verify.py
tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate.py --strict --seeds 8 --seconds 20 --render
tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate.py --variant loaded --strict --extended --bus-delay --obs-noise
```

Outputs use the revision in `easy_mount/parameters.json`: current metrics/videos go into `results/mac-v06/`, and compiled scenes into `tmp/mac-models/v06/`. `--revision v05` evaluates preserved local v05 scenes if present; a fresh checkout must build those from the historical CAD revision first. Existing historical results are committed in `results/mac-v05/`.

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

[The v06 report](../results/mac-v06/REPORT.md) shows a substantial **mount balance improvement** with the unchanged walking teacher. The teacher remains the preferred comparison actor: both fine-tuning attempts failed to beat it reliably. Their checkpoints are marked rejected. Surviving a reverse, lateral or turn-in-place trial does not mean following its command; these motions remain ineffective. Randomized forward/curved trials still expose falls and occasional mount contacts.

The [v05 pilots](../results/mac-v05/REPORT.md) are also rejected. Retained artifacts support reproducibility, not deployment. Printed mass, motor heating, shell strength and real fit remain unverified. Inputs and checksums are in [inputs.json](inputs.json); scene source hashes in [scenes.json](../results/mac-v06/scenes.json). The pinned harness's `uv.lock` fixes training dependencies; CAD dependencies remain separate.
