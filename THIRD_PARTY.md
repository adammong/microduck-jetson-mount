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
