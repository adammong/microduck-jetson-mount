# v07 clearance mount

The Jetson stays in front with ports upward. Its tray moves **40 mm forward and 33 mm upward** from v06. The same GNB8504S60AHV battery moves **61 mm rearward and 35 mm upward**. Raised side arms clear folded legs. Upper shoulder seats support the saddle; removable front-gate hooks capture the front shell lobes from below. The torso screw bosses are now outboard at ±40 mm so the saddle can slide onto the shell.

Estimated payload: **447.6 g**, including **125.6 g** fully dense PETG, 175 g Jetson, 73 g battery and 74 g ancillary allowance. The simulation uses this CAD-derived center of mass and full inertia, with collisions enabled. Head pitch must stay neutral for the tested sitting path. This is a reference-model prototype; physical fit, grip and strength are untested.

## Attach the mount

1. Print the five [STLs](STL/), remove supports and insert six M4 nuts. Use nonconductive pads: nominally 1 mm at the shoulder seats and 2 mm at the front gate pads. Support the powered-off duck with its head upright.
2. Remove the front gate and its two torso thumb screws. Slide the saddle forward onto the torso from behind. Its upper seats rest on the shoulders; the saddle has no underside hooks obstructing that slide.
3. Fit the front gate **from the front**. Its two tongues slide beneath the front shell lobes. Tighten the two **M4×20** torso thumb screws gently. No torso opening or original robot screw removal is required by this mockup.
4. On the bench, retain the Jetson's white OEM base. Remove the two small corner gates, orient the connectors upward, and slide the base into the tray rails. Refit the corner gates with two **M4×12** thumb screws. The gates capture the plastic base; the PCB is not drilled.
5. Engage the tray's locating holes on the front dock pins and secure two **M4×20** thumb screws. These two screws release the equipped tray independently of the torso clamp.
6. Strap the battery into the rear pocket, route the illustrated cables and secure their strain relief. The blue cable's head endpoint is free and unverified; it is not a modeled production socket. Power conditioning/cutoff hardware remains an allowance, not a selected electrical design.

The hooks and upper seats limit vertical motion geometrically; foam and the front gate locate the torso. Real shell dimensions, screw torque, slip under gait loads and damage during falls still require physical testing.

## Files and checks

- [Robot + Jetson + battery preview](GLB/microduck_easy_mount_v07.glb)
- [Equipped mount](GLB/equipped_mount_v07.glb) · [Editable assembly](STEP/easy_mount_assembly.step)
- [Five printable parts](STL/) · [Parameters](parameters.json) · [MuJoCo robot](simulation/microduck_easy_mount_groundcontact.xml)
- [Print, wiring, port and joint-sweep checks](simulation/validation.json)
- [Front-shell hook fit](simulation/shell_capture_check.json) · [Attachment path samples](simulation/attachment_check.json)
- [Mass/inertia](simulation/mass_properties.json) · [Policy tests and video](../results/posture-v07/REPORT.md)

Saved STEP parts are valid single solids; their print STLs are watertight. Sampled front-hook shell clearance is at least **1.24 mm**. The attachment check samples saddle insertion from behind and gate insertion from ahead against the pinned shell surface. It does not prove every triangle interior or production tolerances. HOME ±0.2 rad single-joint clearance and the 50 mm connector corridor pass their sampled checks. Full joint travel still has recorded interference; unrestricted head/get-up motion is not approved by these results.

Print in PETG using the saved print orientations; the saddle and gate need supports. Do not print the battery, wiring, electronics, knobs or straps. Dimensions are millimeters. Infill/strength, support removal and printer-specific clearances are unverified; weigh the finished parts before replacing the simulation's conservative fully dense estimate.

## Rebuild

After installing the repository's CAD requirements, from the repository root:

```sh
CAD_PYTHON=.venv/bin/python sh clearance_mount/build.sh
```

The CAD runtime and pinned Mac policy runtime are separate. After [Mac setup](../training/README.md), build v07 scenes explicitly:

```sh
OMP_NUM_THREADS=1 tmp/microduck-lab/microduck_local/.venv/bin/python training/build_scenes.py --mount clearance_mount
```

v06 CAD and its earlier policy studies remain under `easy_mount/`. The GNB capacity/runtime arithmetic is unchanged; see its [selection record](../easy_mount/battery_selection.json). Third-party robot and NVIDIA geometry retain their [original terms](../THIRD_PARTY.md).
