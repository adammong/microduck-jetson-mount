# External captive Jetson mount — v06 fit prototype

The Jetson Orin Nano developer kit sits **low in front**, with USB/Ethernet/DC ports upward. The GNB 4S 850 mAh battery sits **behind the torso**, counterbalancing the Jetson. Two independent corner retainers replace the head-interfering crossbar. Tray windows reduce plastic mass. The external torso capture remains; mounting does not require opening the torso or removing its screws.

## Attach the mount and Jetson

1. Prepare the five prints below, insert six M4 nuts, and fit nonconductive foam: nominally 1 mm at the upper saddle seats and 2 mm at the torso gate pads. Clear supports and tune nut-pocket clearance to the printer.
2. Support the powered-off duck. Remove the yellow front torso gate. Slide the saddle forward onto the torso from behind: upper seats rest on the shell shoulders; lower lips capture its side undersides. Refit the gate with **two M4×20 thumb screws**, seated gently.
3. On the bench, remove the tray's **two small corner retainers**. Keep NVIDIA's white plastic base attached. Orient its connector edge upward and slide the base down between the rails to the lower stops. Refit the retainers with **two M4×12 thumb screws**. These retain the plastic base; no PCB holes are used.
4. Engage the tray's locating holes on the front gate's outboard pins. Secure **two M4×20 thumb screws**. Removing these two screws releases the equipped tray later.
5. Place the battery on the 0.6 mm bottom pad in the rear pocket, then strap it through the pocket slots. Its leads leave through the short-edge notch. Strap a verified power-conditioning/protection device to the rear side pad. Restrain wiring with slack at moving joints.

These are intended operations, not a physically demonstrated installation. Production shell fit, pad compression and fastener access need a fit print. Connecting the duck's compute board may require separate head access. The amber ring is a survey marker, **not a stock port**; the blue head-end plug is deliberately disconnected. See [WIRING.md](WIRING.md).

## Parts and printing

Print only these five files:

- `STL/saddle.stl`
- `STL/front_gate.stl`
- `STL/jetson_tray.stl`
- `STL/jetson_gate_left.stl`
- `STL/jetson_gate_right.stl`

Hardware: four M4×20 and two M4×12 thumb screws; six ISO 4032 M4 nuts; nonconductive pads and retaining straps. Electronics, cables, battery, converter and gray knobs are purchased-component illustrations.

Screw holes are 4.5 mm; nut pockets are 7.25 mm across flats and 3.5 mm deep. The OEM base is nominally 103×90.5 mm. The rail internal width is 104.2 mm, with 0.8 mm lip overlap and nominal 0.5 mm clearance above the base. PETG is the mass assumption; capture seats, rail lips, ears and nut pockets still require support and strength review. All print STLs are watertight, with their bottom at Z=0. No slicer or strength certification is implied.

Battery: **GNB8504S60AHV**, 73×18×32 mm, 73 g ±2 g, 15.2 V, 850 mAh, 12.92 Wh. Body center `[-122.6,0,0]` mm in trunk coordinates. Pocket interior 75×20 mm, 36 mm deep. End straps occupy the nominal 1 mm end clearance; tune their thickness to the actual pack. Factory lead placement and XT30/balance connectors are approximate. The converter remains an **unselected** 14×24×12 mm allowance.

## CAD and simulation

- [Complete robot preview](GLB/microduck_easy_mount_v06.glb)
- [Equipped mount](GLB/equipped_mount_v06.glb)
- [Head wiring inspection](GLB/head_wiring_inspection_v06.glb): older reference PCB footprint, with the cover omitted; production Radxa socket geometry is unverified.
- [Editable assembly](STEP/easy_mount_assembly.step), plus five individual print STEP files.
- `simulation/microduck_easy_mount_{walk,groundcontact}.xml`: both baseline robot variants, with explicit payload inertia, zero-mass visuals and active rigid mount collision boxes.
- `simulation/{mass_properties,validation,balance_estimate}.json`: generated numerical results.
- [Mac simulation report](../results/mac-v06/REPORT.md).

CAD units are millimeters; simulation uses meters and kilograms. Trunk X is forward, Y left, Z up. Tray local `(X,Y,Z)` maps to trunk `(Z,X,Y)` plus `[50,0,-18]` mm; the vendor kit rotates 180° around tray Z first. `parameters.json` controls positions, dimensions and mass assumptions.

Payload mass is **442.1 g**: **120.1 g** fully dense PETG, Jetson 175 g, battery 73 g, and 74 g ancillary allowance (fasteners/pads/straps 22 g, data cable 14 g, power wiring/extension 18 g, conditioning/protection 20 g). The payload is counted once; the stock robot battery remains. Weigh sliced prints and purchased hardware before calibrating inertia. Cable mass is fixed in HOME; head articulation and cable tension are not modeled.

## Verification and limits

Five valid single print solids and five watertight STLs pass native print/print, battery/strap/factory-lead/print and wire/print intersection checks. The 50 mm upward plug corridor clears prints and the sampled robot poses. HOME plus 126 individual-joint samples within ±0.2 rad has no reported penetration beyond 0.2 mm. **Full-range motion still records 140 interfering pairs**; this design does not clear the complete stock motion range or combined head/leg motions.

Static HOME COM shifts **2.1 mm forward, 1.5 mm left and 9.0 mm down**. Optimistic rear sole margin is **22.7 mm**, versus 8.5 mm in v05 and 20.5 mm stock. This assumes usable contact over the entire projected sole; it is not a payload rating. Paired walking rollouts support the balance improvement, but hardware strength, fit, motor temperature and power remain unverified.

## Battery and power

[Manufacturer specification](https://www.gaoneng.shop/products/gaoneng-gnb-lihv-4s-15.2v-850mah-60c-xt30-lipo-battery-long-range) and [selection record](battery_selection.json) document the pack. No exact battery CAD was available: its body uses published dimensions, with illustrative leads and connectors.

At 15–25 W total Jetson draw, 80% usable energy and 90% conditioning efficiency imply **22–37 minutes**. This does not predict robot motor runtime. Full-charge 17.4 V is within NVIDIA's published 9–20 V carrier input range, but protection, transients, cutoff and the actual circuit are unvalidated. Use suitable LiHV charging equipment and verify actual lead exit, plugs and power hardware before fitting.

## Rebuild

From the repository root, using the Python 3.12 CAD environment:

```sh
sh easy_mount/build.sh
.venv/bin/python checks/package_easy_mount.py
```

`CAD_PYTHON` can select an existing CAD interpreter. Model sources are in `src/`, shared geometry in `src/lib/`, checks in `checks/`. The packaging command creates a versioned ZIP under `dist/` containing the current design and pinned rebuild inputs. See [third-party attribution](../THIRD_PARTY.md); robot and NVIDIA assets retain their respective terms.
