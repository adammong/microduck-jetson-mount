# Model catalog

Four scripts build print parts with STEP/STL/GLB outputs of the same name.
The wiring and purchased battery models have STEP/GLB only because it represents purchased components.
The assembly calls the four child models and restores their manufacturing poses.

| Script | Purpose |
| --- | --- |
| saddle.py | External torso capture and tray support/docking |
| front_gate.py | Two-screw removable torso closure and forward battery cradle |
| jetson_tray.py | OEM-base capture rails, corner support tabs and open connector edge |
| jetson_gate.py | Open U gate retaining the OEM base; bridge below head |
| wiring_layout.py | Approximate USB/power cables and unselected converter envelope |
| battery_pack.py | GNB8504S60AHV specified body and approximate stock leads, connectors and strap |
| easy_mount_assembly.py | Installed mechanism, purchased screws, battery and wiring envelopes |

Factory geometry, print transforms and assembly transforms are in `lib/design.py`.

Cable routing and plug assumptions are in `lib/wiring.py` and `../WIRING.md`.

Battery geometry is in `lib/battery.py`; source specifications and inertia assumptions are in `../battery_selection.json`.
