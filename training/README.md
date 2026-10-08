# Local Mac simulation and training

Uses [MicroDuck Lab](https://github.com/jonathanhawkins/microduck-lab/tree/bbf0326ef97975f7062368914e0729504d17226f), pinned to `bbf0326ef97975f7062368914e0729504d17226f`. Native CPU MuJoCo and PPO work on Apple Silicon; CUDA is unnecessary for these experiments.

The v05 CAD remains the source of payload mass, inertia, mounting position, and collision boxes. Both variants use the same robot assets, floor, STAND keyframe, BAM XL330 motor model, 200 Hz physics, and 50 Hz policy interface (61 observations / 14 actions). The fuller groundcontact robot is used for training and evaluation. Electronics visuals are simplified; the Jetson is a box envelope. Geometry is rigid; cables do not flex and the mount cannot slip or break in this model.

## Run

From the repository root, with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```sh
uv run --python 3.12 training/setup_mac.py
```

Then:

```sh
tmp/microduck-lab/microduck_local/.venv/bin/python training/verify.py
tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate.py --render
tmp/microduck-lab/microduck_local/.venv/bin/python training/train_pair.py --steps 1048576
```

The last command runs a paired stock/loaded experiment with eight workers per robot, seed 0, CPU PPO, and a strict check against non-foot floor support. Training uses nominal mass and no observation noise; the BAM model still samples startup supply voltage/sag and uses bus delay. This is a pilot recipe, not the full hardware-transfer recipe. Runs, logs, native resumable checkpoints, and ONNX exports go into `tmp/mac-runs/`. Rendered video and metrics go into `results/mac-v05/`.

Use `--variant loaded` for a single robot. Use `--seed 1` for another training seed, `--run-tag retry` for a new output directory, or `--init-from DIRECTORY` to resume a suitable native checkpoint. ONNX alone cannot resume PPO. Public archived native checkpoints use `.sb3` instead of `.zip`; the driver restores them to a temporary run directory before loading.

`--allow-nonfoot-support` reproduces the original pilot's permissive fall checks. It should only be used to reproduce the documented failure, where the robot rests on the backpack. The default strict wrapper ends an episode after three successive control steps with a non-foot floor contact carrying more than 0.05 N. It changes termination, keeping observations, actions, rewards, and motor limits intact. Both strict trainers completed a 16,384-step smoke test.

## Results

See [the report](../results/mac-v05/REPORT.md). Both million-step pilot policies are **rejected walking policies**. They are retained for comparison, not recommended warm starts. Fix mount clearance/balance, then import/distill a competent walking teacher and train with the strict guard and hardware randomization. Confirm actual printed masses and mounting rigidity before interpreting sim behavior as hardware behavior.

Inputs and download checksums: [inputs.json](inputs.json). Generated model source checksums: [scenes.json](../results/mac-v05/scenes.json). The pinned harness's `uv.lock` fixes dependency versions; CAD dependencies stay separate.
