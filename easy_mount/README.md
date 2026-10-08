# External captive Jetson mount — v05 fit prototype

The Jetson Orin Nano developer kit now faces **USB/Ethernet/DC ports upward**.
Blue USB data and green power cable envelopes show the proposed wiring, with
service loops, two tray cable-tie eyes and a strap pad for a converter allowance.
The selected GNB 4S 850 mAh LiHV battery sits in a wider, lower pocket on the torso gate. The open Jetson gate's crossbar
sits behind the tray and below its top edge to improve head clearance.

The amber head-side ring is a **survey marker, not a port**. The blue cable's
head-end plug is deliberately disconnected: board USB capability is confirmed,
but access through the production enclosure and exact socket position are not.
See [WIRING.md](WIRING.md) for sources, port roles, assumptions and measurements.

## Attach the Jetson and mount

1. Prepare four prints, insert six M4 nuts, and fit nonconductive foam: nominally
   1 mm at the upper saddle seats and 2 mm at the front gate pads. Clear holes
   and supports; tune nut clearances to the printer.
2. With the duck supported and powered off, remove the yellow front torso gate.
   Slide the saddle forward onto the torso from behind. Its upper seats rest
   on the shell shoulders; lower lips capture the side undersides. Refit the
   front gate with **two M4×20 thumb screws**, seated gently.
3. On the bench, remove the Jetson tray's yellow top gate. Leave NVIDIA's white
   plastic base attached. Orient the connector edge upward and slide the base
   down between the tray rails to the lower corner stops. The rail lips capture
   the base edges. Refit the open top gate with **two M4×12 thumb screws**.
   This holds the plastic base; it does not fasten into the PCB or remove it.
4. Place the tray ears on the saddle ledges, engage locating pins, and secure
   **two M4×20 thumb screws**. These two screws release the equipped tray later.
5. Place the GNB8504S60AHV pack on the 0.6 mm nonconductive bottom pad and strap it into the front pocket. Its power/balance leads leave through the short-edge notch. Strap a dimensionally and
   electrically verified converter to the side pad. Connect its output to
   Jetson J16; connect USB data from J5 to the duck's confirmed host port.
   Route and restrain cables with slack at moving joints. The illustrated head
   endpoint and converter terminations must first be measured. Verify the pack lead exit and plugs against the purchased battery.

The torso saddle requires no stock screws to be removed and no torso opening.
Connecting to the compute board may require separate head access. These are
intended operations, not a physically demonstrated fit or installation.

## Parts and printing

Print only `STL/saddle.stl`, `front_gate.stl`, `jetson_tray.stl`, and
`jetson_gate.stl`. Gray knobs, NVIDIA hardware, battery, purple converter and
colored wiring are purchased-hardware illustrations, not printable parts.

- Hardware: four M4×20 and two M4×12 thumb screws; six ISO 4032 M4 nuts.
- Screw holes: 4.5 mm. Nut pockets: 7.25 mm across flats, 3.5 mm depth.
- OEM base: nominal 103×90.5 mm. Rail internal width: 104.2 mm; lips overlap
  its side edges by 0.8 mm, with nominal 0.5 mm above-base clearance.
- Battery: **GNB8504S60AHV**, 73×18×32 mm, 73 g ±2 g; 15.2 V, 850 mAh, 12.92 Wh. Center `[62.6,0,-10]` mm in trunk coordinates. Pocket interior 75×20 mm, 36 mm deep, with 1 mm clearance per side and a 0.6 mm bottom pad.
- The modeled retaining strap occupies the 1 mm clearance at the pack ends; tune pocket/strap thickness to the actual components.
- Approximate factory XT30 discharge plug and balance connector are shown; geometry/lead length are illustrative. `battery_selection.json` records manufacturer specifications and sources.
- Converter: **unselected** 14×24×12 mm allowance beside the front battery.
- PETG is the material assumption. The saddle arms/capture seats, tray rail
  lips, gate ears and some nut pockets require support review. All supplied
  print STLs are closed, watertight and positioned with their bottom at Z=0.
  This is not a strength or slicer certification.

## CAD and simulation

- `GLB/microduck_easy_mount_v05.glb`: complete reference robot and wiring.
- `GLB/equipped_mount_v05.glb`: mount and Jetson without the robot.
- `GLB/head_wiring_inspection_v05.glb`: head top cover omitted to reveal the
  **older reference compute PCB footprint**, not production Radxa port geometry.
- `STEP/easy_mount_assembly.step`: printable mechanism, simplified screws and
  generic wiring/converter envelopes and the specified battery body. NVIDIA geometry is added to the equipped
  GLB from the parent vendor source. Four individual STEP files remain editable.
- `simulation/microduck_easy_mount_{walk,groundcontact}.xml`: both baseline
  robot variants with an explicit payload inertial and zero-mass visuals.
- `simulation/mass_properties.json`, `validation.json`, `balance_estimate.json`:
  current mass/inertia, fit checks and static balance estimate.

Payload mass uses fully dense PETG, specified Jetson 175 g and battery 73 g,
plus an estimated **74 g** ancillary budget: hardware/pads/straps 22 g, data
cable 14 g, power wiring/custom XT30 extension 18 g, converter/protection allowance 20 g. Cable/converter mass is distributed
at its modeled location, counted once. Calibrate these estimates by weighing the
chosen hardware and sliced prints. The specified battery mass includes its stock leads/connectors, approximated at the body COM; they are not counted again in cable mass. The original robot battery mass is preserved.
Wiring is fixed in HOME for this inertia approximation; flexible tension and
moving-head cable mass redistribution are not modeled.

CAD uses millimeters in the robot trunk frame: X forward, Y left, Z up. Tray
placement maps local `(X,Y,Z)` to trunk `(-Z,-X,Y)` plus `[-65,0,25]` mm; the kit
rotates 180° around tray Z first. `parameters.json` controls dimensions and masses.

## Verification and limits

The maintained build checks saved STEP solids, watertight print meshes, native
print/print and wire/print intersections, the 50 mm upward connector corridor,
and individual-joint sweeps. Tube centerline capsules are checked against the
reference robot in HOME; blue and main green modeled bends exceed 10 mm radius.
The generic plug envelopes and insertion motions still require physical checks.

Large joint motions still collide with the hard mount. The build records full
sweep interference rather than silently limiting the robot's joints. A static
sole-footprint margin assumes full usable sole contact and is optimistic. Neither
collision samples nor COM geometry prove standing or walking with this payload.
Combined motions, shell strength, cable articulation and hardware fit are unverified.
The existing policies must account for this substantial extra mass before trials.

See the numerical results appended below; these are fit-model estimates, not
measured hardware performance.

## Rebuild and provenance

From the parent project in its Python 3.12 environment:

```sh
sh easy_mount/build.sh
.venv/bin/python checks/package_easy_mount.py
```

Model sources are in `src/`, shared geometry in `src/lib/`, checks in `checks/`.
The packet includes the parent NVIDIA preview source and pinned robot mesh assets.
Robot geometry retains upstream CC BY-SA-NC terms; NVIDIA retains its download
terms. See parent `simulation/reference/provenance.json` and
`STEP/imported/provenance.json`. The M4 nut was fetched from step.parts with
checksum verification; unmatched battery/XT30 hardware, thumb screws and USB cables are explicitly
simplified envelopes.

## Battery selection and power

[Manufacturer specification](https://www.gaoneng.shop/products/gaoneng-gnb-lihv-4s-15.2v-850mah-60c-xt30-lipo-battery-long-range): GNB8504S60AHV, 4S LiHV, 73 g ±2 g, body 73×18×32 mm. [US reseller listing](https://www.racedayquads.com/products/gaoneng-gnb-15-2v-4s-850mah-60c-lihv-micro-battery-long-type-xt30) reported in stock on 2026-10-05; the manufacturer currently reports paused US shipping. The 12.92 Wh nominal energy is voltage × capacity. No exact vendor STEP was available from the catalog search; this is a specified body envelope with approximate label, leads and connectors.

At 15–25 W **total Jetson draw**, 80% usable energy and 90% conditioning efficiency give an estimated **22–37 minutes**. This does not power or predict runtime for the robot motors. Full-charge 17.4 V lies within the NVIDIA carrier's published 9–20 V input range, but wiring, transient behavior and low-voltage cutoff remain unvalidated. Use an appropriate LiHV balance charger. The purple 20 g allowance reserves space/mass for power conditioning/protection; no circuit has been selected. Verify the pack, actual cable exit and all power hardware before printing/fitting.

## Current numerical results

Generated checks and current mass/balance results are in `simulation/`. Physical fit and gait remain unverified; full joint travel is not cleared.

- Payload: **451.3 g**, including **129.3 g** fully dense PETG; loaded reference robot **1.1885 kg**. Actual sliced plastic and purchased hardware must be weighed.
- Four valid single print solids and four watertight print STLs. Native print/print, battery/strap/factory-lead/print, and wiring/print intersection checks report zero volume above 0.001 mm³.
- No reported hard-mount penetration beyond 0.2 mm in HOME and 126 individual-joint samples within ±0.2 rad. Full-range sweep still records **128 interfering pairs**.
- Static HOME COM shifts 12.1 mm rearward and 1.4 mm right. Optimistic rear sole margin is **8.5 mm**, versus 20.5 mm stock; this does not prove balance or walking.
- Main blue/green cable and XT30 extension sampled bend radii exceed 10 mm. Factory battery leads and connector shapes are approximate and not validated for moving joints.
