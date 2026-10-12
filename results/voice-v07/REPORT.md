# Voice and combined mounted motion controller

**Experimental simulation controller: 77/80 sequence passes, 78/80 contact/fault-free.** Nominal: 39/40 passes, 40/40 contact/fault-free. Noise/randomized physics with ±10% payload: 38/40 passes and contact/fault-free. Two heavier randomized trials contact the Jetson with the floor during stand-up; one nominal left turn falls below the minimum tracking threshold. A contact/tilt/timeout fault pauses simulation and requires reset. This controller is not hardware-qualified or a replacement for the selected terrain walking actor.

The local interface accepts microphone, text or buttons for forward/backward, both sidesteps, both turns, stop, sit and stand. Movement defaults to three seconds, accepts 0.5–10 seconds, and begins its duration after automatic standing completes. It stops before sitting. Stop cancels active/pending movement; an active sit/stand transition completes to its current stable posture. Unknown/compound/negated phrases are rejected. “Walk forward for three seconds” is supported; arbitrary natural-language tasks and distance/angle navigation are not.

| Sequence | Passes | Contact/fault-free |
| --- | --- | --- |
| voice_demo | 13/16 | 14/16 |
| directions | 16/16 | 16/16 |
| cycles | 16/16 | 16/16 |
| interrupts | 16/16 | 16/16 |
| lease_loss | 16/16 | 16/16 |

[Summary and failures](summary.json) · [Controller configuration](controller.json) · [Contracts](verification.json) · [Speech-to-action HTTP checks](http-verification.json) · [Actual mid-posture overrides](override-verification.json) · [Source/runtime hashes](run.json).

## Run on the Mac

From the repository root, complete the [simulation setup](../../training/README.md), then:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/build_scenes.py --mount clearance_mount
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/setup_voice.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/teleop_skills.py
```

Open http://127.0.0.1:8798. Click **Speak a command** and allow microphone access to this local page. Recording finishes after speech followed by 0.5 seconds of quiet, after four seconds, or when you click again. No microphone is enabled until you click. Typed commands/buttons work without the optional speech setup. Chrome or Safari can be used if the in-app browser cannot access the microphone. Voice runs locally after model download; temporary audio is removed after processing. No user microphone recording was taken for this study.

## Architecture and handoffs

Audio → offline English Whisper tiny → strict command parser → skill supervisor → walking or sit/stand ONNX → unchanged MuJoCo/BAM motors. The speech subprocess does not hold the physics lock while recognizing. Stop/reset or another command invalidates an older in-flight utterance. A 0.6-second expired browser heartbeat cancels movement, including a pending movement after standing; valid recognized speech cannot arm movement with an expired client lease. This is a supervisor over two actors, not one newly trained multitask neural network.

The v07 mounted scene retains 447.56 g added payload, 1.18480 kg total nominal mass, BAM XL330 at 1.75 A, 200 Hz physics, 50 Hz policy, motor/action delay, and neutral head pitch. No balance force, motor boost, root teleport or episode reset is used between commands. Walking and posture use separate meanings for observation slot 48; the supervisor writes the right command before evaluating each actor. Policy tensors are unchanged. Handoffs blend from the preceding applied action target for 0.3 seconds. Stand readiness requires one second of full standing with body XY speed below 0.04 m/s and yaw rate below 0.12 rad/s; sitting needs 0.5 seconds of qualified floor-supported sitting; stopping needs 0.6 seconds of settled standing. A transition taking over eight seconds faults.

Simulation contact and velocity measurements drive supervisor guards, not new neural observations. A hardware implementation needs validated state estimation and its own independent actuator watchdog. The simulator pauses physics on fault; that is not a hardware stop implementation.

## Calibration and fresh tests

Initial paired prototype tests passed 75/80 sequences with 79/80 contact/fault-free; occasional backwards starts stalled after a posture handoff. Those results/source are retained in [prototype](prototype/). A 16-profile grid on calibration seeds 10–13 compared terrain/posture walking actors, backward commands −0.15/−0.10 m/s, zero/0.8-second additional walking-settle holds, and zero/0.3-second action blending. The terrain actor at −0.15 m/s with 0.3-second blending was retained. Two nine-profile reverse-leg bias grids on those same seeds compared mirrored hip/knee offsets and an opposing ankle offset, clipped to 0.12 rad in the refinement. [Calibration](calibration/).

The smallest winning reverse profile adds −0.04 rad to the left hip/knee, +0.08 rad to the left ankle, with mirrored right-leg signs, only under the walking actor and negative requested forward velocity. The gain ramps with the command to full effect at −0.05 m/s; the added correction never exceeds 0.08 rad and never touches head joints. Forward/side/turn/stop/posture commands receive no reverse bias. This is bounded motor-target calibration, not neural PPO training or an external recovery assist.

An overly strict initial check compared a deliberately interrupted one-second forward startup with its full-speed target. Final evaluation instead compares that interrupted startup with its actual mean ramped command. Other movement commands retain the 25% full-target direction threshold and unrequested mean translation/yaw limits [0.05 m/s,0.05 m/s,0.15 rad/s]. After this criterion correction, the selected profile passed all 20 calibration sequences. Final source and parameters were then frozen before new nominal seeds 800–807 and noise/randomized seeds 820–823 at 0.9/1.1 payload scales: five sequences, 80 continuous trials. Pass requires every intended command/posture to complete, signed movement qualification, and no contact/fault. Lease-loss cases require cancellation rather than completing the requested ten-second movement.

Eight additional continuous scheduled override tests on nominal seeds 880–883 and randomized seeds 890–891 at 0.9/1.1 payload scales all pass, including stop during sitting/standing, pending-walk cancellation, and a final unsupported stand. These do not erase the two stand-up failures in the main benchmark.

## Speech and interface checks

The pinned [SYSTRAN faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.1 runtime runs English `tiny.en` locally on Mac CPU with int8 weights in a separate virtual environment. Model revision and file hashes are in [speech-model.json](speech-model.json). The [model repository](https://huggingface.co/Systran/faster-whisper-tiny.en) derives from OpenAI Whisper. Downloads happen only during setup; inference uses the local model in offline mode. Jetson execution/latency has not been tested.

Six Mac-generated synthetic command recordings decode to the expected intents, including a duration; two negated/unrelated recordings are rejected by the parser. Silence is rejected, and confidence checks require segment average log probability ≥−0.8 and no-speech probability <0.6. Warm recognizer computation was approximately 0.33–0.36 seconds on the probe, with a roughly four-second first call; process startup and microphone capture add latency. This is synthetic speech validation, not a human/noisy-microphone accuracy benchmark. [Recognition measurements](speech-probe.json) · [End-to-end HTTP speech/action checks](http-verification.json). The browser PCM resampler was checked at 16/44.1/48 kHz inputs; real microphone capture remains untested.

## Video and reproduction

[15-second continuous excerpt, top text removed](skills-15s.mp4) · [Full 29.96-second demo, top text removed](skills-demo.mp4) · [Captioned full demo](voice_demo-demo.mp4) · [Scene and phase trace](demo.json) · [Media hashes](media.json).

The demo uses the mounted physics and scripted text commands through the same parser/controller; it does not contain microphone recognition or recorded user speech. Fresh seed 850 completes forward, stop, sit, automatic stand before reverse, left turn, stop, sit, stand without a fault or sampled mount contact. The first 15 seconds are an unpadded continuous excerpt. The UI also was checked through real button/text interactions.

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/verify_skills.py --out tmp/skill-contract-check.json
node training/verify_voice.cjs
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_skills.py --out tmp/skills-nominal.json --seeds 8 --seed-start 800
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/evaluate_skills.py --out tmp/skills-randomized.json --noise-dr --payload-scales .9 1.1 --seeds 4 --seed-start 820
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python results/voice-v07/http_smoke.py
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python results/voice-v07/override_checks.py
```

The last three checks write their own corresponding result files. Calibration scripts import the archived pre-calibration controller/evaluator next to them and write under `tmp/voice-study`; create that directory before replay. Prototype files are retained as historical evidence, not the current controller.

## Limits

Combined sit/stand handoffs are tested on flat ground only. Existing terrain walking results do not validate sitting on rough ground. Stand-up can fail under randomized heavier payloads; slow turns can under-track. Voice is a finite English command vocabulary, with no wake word, conversational model, obstacle avoidance or distance/angle guarantees. Fall/get-up recovery, physical microphones, Jetson runtime, actual motor/communications integration, mount strength, thermal endurance and hardware behavior remain unvalidated.
