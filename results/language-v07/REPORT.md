# LFM2.5-350M natural requests for the mounted duck

The local control page now combines English Whisper tiny transcription, a task-adapted LFM2.5-350M language model, and the existing walking/sit/stand supervisor. Try **“please move ahead briefly and then take a seat”** or **“could you stand up and turn right for two seconds.”** Buttons and exact single commands keep their fast parser path. The page displays the interpreted plan.

This is a simulation prototype with limited language reliability. The final test produced exact plans for **14/24 supported requests** and rejected **24/24 unsupported requests**. The other ten supported requests were refused; no incorrect plan was accepted in this test. These finite checks do not establish general language safety. The untuned 4-bit model produced zero exact supported plans in the same final test; many outputs had invalid formatting. The adapter alone produced 14/24 exact supported plans but wrongly accepted eight unsupported requests. The input/output guards provide those additional rejections.

Four of six end-to-end requests completed their ordered actions without a controller fault: two typed requests and both synthetic voice requests. Two supported typed requests were refused. Stop/reset superseded inference; disconnecting the client cancelled the queued sit; an expired client lease could not execute a plan. Contract checks cover malformed plans, durations, explicit directions/postures, queue order, timeout and cancellation. Browser WAV checks cover 16/44.1/48 kHz inputs and the six-second recording limit. Human microphone capture, background noise and hardware are untested.

## Model and training

Official base: [LiquidAI/LFM2.5-350M-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-350M-MLX-4bit), pinned to `f6cb4e006bb7a2d8a6afa14ec0a53e0586f65a5b`. Base weights occupy 221,809,325 bytes on disk; the adapter occupies 12,012,680 bytes. These are storage figures, not runtime RAM measurements. Inference uses MLX/Metal on Apple Silicon, greedy decoding and local files only. Each request starts a fresh language subprocess; successful text interpretation took about 2.2 seconds and synthetic speech about 2.6–2.7 seconds after upload, excluding recording and motion execution.

Supervised LoRA trains 2.998M parameters at rank eight across all 16 layers. The first stage used 3,828 synthetic training requests, 160 validation requests and 250 steps. Its diagnostic test exposed missed actions and poor rejection. The second stage added rejection examples and more sequence separators: 4,827 training requests, 160 validation requests, 600 continuation steps. Batch size four, learning rate 0.0001, seed 3525. Continuation checkpoint 550 was selected by minimum validation loss (0.229) before the final 48 text requests were evaluated. Exact final requests are disjoint from training and validation; vocabulary and task families overlap. The original 44-request diagnostic set informed development. Input guards and capitalization/punctuation normalization were subsequently corrected using test failures, so the final guarded result is a development evaluation, not a blind safety certification. All failures remain in the JSON reports.

The adapter and synthetic datasets are included. Base weights and isolated runtime remain in ignored `tmp/` directories. [Model provenance](model.json), [training record](training.json), [final guarded results](final-guarded.json), [adapter-only results](final-adapter-only.json), [untuned results](final-untuned.json), [HTTP/physics results](http-verification.json) and [prototype diagnostics](prototype/) record the comparison. Synthetic audio contains no user recording. The adapter is a modified derivative of Liquid AI's model; its [license](adapter/LICENSE) and [notice](adapter/NOTICE.md) apply.

## Execution limits

Supported actions: forward/backward, sidestep left/right, turn left/right, sit, stand, stop. Up to six sequential actions and 20 total movement seconds, 0.5–10 seconds per movement. Default three seconds; “briefly,” “a little” and “a bit” request one second. Code checks action count, explicit direction/posture and duration against request clauses. It rejects explicit negation, conditionals, explanatory questions, unsupported navigation, speed/distance/angle requests and rule changes. The language model still selects actions; this is bounded task interpretation, not an unrestricted conversational assistant.

A new request stops the previous plan during interpretation. Stop/reset/new accepted commands invalidate older inference. Losing the 0.6-second client lease clears every pending action. The next action waits for the preceding posture or movement to settle. A 25-second action timeout or a supervisor fault cancels the plan. Stop during a sit/stand transition finishes to a stable posture. Inference runs outside the physics lock.

The v07 mount, 1.184798914 kg robot mass, 1.75 A motor model, 200 Hz physics and 50 Hz actors are retained. Walking and posture neural policies are unchanged. Earlier heavier stand-up failures and limited walking robustness still apply. There is no physical robot communication driver, reliable general fall recovery or hidden assistance.

## Run on the Mac

After the existing [simulation setup](../../training/README.md):

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/build_scenes.py --mount clearance_mount
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/setup_voice.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/setup_language.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_skills.py --port 8800 --seed 950 --idle-exit 60
```

Open http://127.0.0.1:8800. Speak/type a short request; speech recording ends on silence, a second click, or six seconds. Microphone access requires the user's browser permission. Text works without speech setup. Natural-language interpretation requires the bundled adapter and separate language setup. The server binds only localhost; the configured idle timeout excludes heartbeat-only traffic.

## Reproduce training and checks

The first-stage dataset is preserved under `prototype/`; the final generator rebuilds `data/`. All commands run from the repository root:

```sh
tmp/language-venv/bin/python training/build_language_data.py
HF_HUB_OFFLINE=1 tmp/language-venv/bin/python -m mlx_lm lora -c training/language_lora.yaml --data results/language-v07/prototype --adapter-path tmp/language-study/initial
HF_HUB_OFFLINE=1 tmp/language-venv/bin/python -m mlx_lm lora -c training/language_lora.yaml --iters 600 --resume-adapter-file tmp/language-study/initial/0000250_adapters.safetensors --adapter-path tmp/language-study/retrained
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_language.py --out tmp/language-check.json
HF_HUB_OFFLINE=1 tmp/voice-venv/bin/python training/evaluate_language.py --adapter results/language-v07/adapter --cases results/language-v07/final-holdout.json --out tmp/language-evaluation.json
```

Select the continuation checkpoint with minimum validation loss and copy it as `adapters.safetensors` alongside its `adapter_config.json`. To replay HTTP checks, start `teleop_skills.py --no-render --port 8801 --seed 950` and run `results/language-v07/http_smoke.py` using the simulation Python.

## Jetson

LFM2.5 supports cross-platform runtimes, but this implementation uses Apple's MLX runtime and adapter format. It does **not** currently run unchanged on a Jetson. Exporting/merging the adapter into a Jetson-compatible runtime and measuring voice/policy latency, RAM and power remain required. No Jetson or physical duck validation is claimed. [Liquid deployment documentation](https://docs.liquid.ai/deployment/on-device/mlx) describes the Mac runtime.
