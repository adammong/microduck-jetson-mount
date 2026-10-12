# Third-party assets

## MicroDuck reference robot

Source: https://github.com/pollen-robotics/microduck_rl at commit `273afe0b31c4ab365b9ff806a927b63ac92b5ddd`.

The upstream README states that software is licensed under Apache 2.0, and 3D model files under Creative Commons BY-SA-NC. Preserve upstream attribution and applicable terms; no license version beyond the upstream declaration is inferred here. The original upstream README and software LICENSE are retained in `simulation/reference/`. Per-file source URLs and SHA-256 checksums are in `simulation/reference/provenance.json`. Assembled previews contain these robot model assets.

## NVIDIA developer-kit geometry

Source: NVIDIA Jetson Orin Nano developer-kit 3D STEP download. `GLB/jetson_devkit.glb` is a derived preview of NVIDIA geometry, also included in equipped assembly previews. Original download provenance and checksum are in `STEP/imported/provenance.json`. NVIDIA retains rights to its supplied geometry and applicable download terms; this repository does not grant a new license to those assets.

## Purchased-part illustrations

Battery body dimensions and mass are based on GAONENG GNB specifications linked in `easy_mount/battery_selection.json`. Battery label, lead placement and connectors are simplified illustrations. Standard fasteners, cable overmolds and the power-conditioning allowance are approximate fit geometry, not product manufacturing models. Model sources and documentation distinguish these from the five printable mount parts.

No blanket license is assigned to this mixed-source repository.

The directional-motion policies in `results/motion-v06/` also derive from the pinned Apache-2.0 walking actor. They retain its weights and normalizer and add command maps, bounded gyro/gravity joint feedback and an observable turn-start activation rule. License copies and modification notices accompany the exported policies. Full calibration and selection records distinguish selected simulation artifacts from rejected candidates; hardware performance is unverified.

## Mac simulation and policies

The setup downloads [jonathanhawkins/microduck-lab](https://github.com/jonathanhawkins/microduck-lab) at `bbf0326ef97975f7062368914e0729504d17226f` (Apache 2.0). The simulator scene wrappers are from `pollen-robotics/microduck_rl` at `badc4e7ffe5507fd7acb1a21487bd2925c1afe5a`. The official `alpha_walking.onnx` is downloaded from `pollen-robotics/microduck-policies` revision `088524a64e2557dc453256b6071dbb9d23888802`; it is not redistributed here. Source URLs and SHA-256 hashes are recorded in `training/inputs.json`. The archived stock/loaded pilot policies were trained locally for this study and are marked rejected experimental policies. Source harness and dependency caches stay outside Git.

The v06 native teacher import and fine-tuned derivatives retain that actor's weights. The pinned upstream [model metadata](https://huggingface.co/api/models/pollen-robotics/microduck-policies/revision/088524a64e2557dc453256b6071dbb9d23888802) declares Apache 2.0. Copies of the license and attribution are in `results/mac-v06/policies/LICENSE` and `NOTICE`; modifications and policy selection status are recorded with each artifact. The original downloaded ONNX file remains in the ignored input cache.

The fast-walking study also retains the teacher actor and adds command-gated joint corrections. Its PPO derivatives, numerical studies and selected feedback policy are attributed under `results/speed-v06/policies/LICENSE` and `NOTICE`. The selected policy is validated for the documented simulation cases; hardware performance is unverified.

The slow-motion/recovery exports in `results/precision-v06/` derive from the same actor and directional calibration. They retain all inherited neural tensors and add command powers, standing-relative gait amplitude, observable yaw-rate feedback, bounded IMU corrections and tilt-triggered catch commands. Apache-2.0 license copies and modification notices accompany exported policies. Their report distinguishes superseded candidates from the selected simulation controller and its required command ramp. Hardware performance is unverified.

The floor-recovery candidate in `results/posture-v06/` derives from official `alpha_stand.onnx` at the same pinned policy revision. It retains all neural actor/normalizer tensors and adds bounded pose and observable gravity/gyro corrections. The pinned `alpha_sitstand.onnx` is downloaded for evaluation but its original file is not redistributed. Checksums/source URLs are in `training/posture_inputs.json`; Apache-2.0 license and modification notices accompany the exported derivative. Full sitting and general fall recovery remain unresolved; the report records rejected searches and diagnostic-only geometry changes.

The v07 clearance geometry and previews under `clearance_mount/` retain the same upstream robot/NVIDIA assets and purchased-component assumptions. Policies in `results/posture-v07/` derive from the pinned `alpha_stand`, `alpha_sitstand` and prior precision walking actor. They preserve inherited neural/normalizer tensors and add posture selection, neutral neck/head pitch and bounded calibrated feedback. License copies and modification notices accompany every export. The compiled training-scene snapshot also contains those reference geometry assets and retains their applicable terms. Sitting, walking and recovery have separate selection status and simulation limits.

The uneven-terrain study in `results/terrain-v07/` derives from the same Apache-2.0 walking actor and v07 calibration. It retains inherited neural tensors/normalization and adds bounded observable IMU/command corrections; correction strength and final selection are recorded separately. Every exported policy includes LICENSE and NOTICE. Procedurally generated terrain adds no external model assets; the robot/mount/Jetson assets and their original terms remain unchanged.

The robustness study in `results/robust-v07/` derives from the same pinned Apache-2.0 walking actor and selected mounted terrain adaptation. It retains all neural and normalizer tensors, adding observable command shaping and bounded IMU/joint corrections. LICENSE and modification NOTICE accompany every exported actor. Procedural force/friction tests add no external assets; existing robot, mount and Jetson geometry retains its original terms. Training checkpoints, validation and rejected experiments remain distinct from the selected simulation controller.

The optional voice controller uses SYSTRAN faster-whisper (MIT) and the English Whisper tiny model converted by SYSTRAN from OpenAI Whisper (MIT), downloaded separately at the recorded model revision. Dependencies and speech weights are not vendored. Synthetic test audio was generated on macOS from the short project command phrases; no user microphone recording is included. The combined controller reuses previously attributed ONNX policies unchanged, with a separately documented bounded reverse motor-target correction.

## Local LFM language interpretation

LFM2.5-350M is copyright Liquid AI, Inc. and governed by the LFM Open License v1.0. The official 4-bit MLX model is downloaded separately at a pinned revision. The included duck-command LoRA adapter is a modified derivative, trained locally on synthetic requests; see `results/language-v07/adapter/LICENSE` and `NOTICE.md`. MLX-LM and MLX are MIT-licensed. Original model weights and runtime dependencies are not vendored. Synthetic spoken requests were generated on macOS; no human microphone recordings are included.

## Optional Whistle speech recognition

[Cactus-Compute/whistle](https://huggingface.co/Cactus-Compute/whistle) and the [Needle runtime](https://github.com/cactus-compute/needle) are Apache-2.0 licensed. Pinned weights/native runtime are downloaded separately by `training/setup_whistle.py`, with provenance and checksums in `results/whistle-v07/speech-model.json`; neither is vendored. Recognition uses the native Apple Silicon library through cactus-needle 3.2.1. Synthetic/noise test audio contains no user microphone recordings. Existing LFM adapter and motion policy attribution still applies.
