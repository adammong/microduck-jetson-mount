# Wiring fit study — v05

Blue: USB-C data cable from Jetson J5 to a **FREE head-side plug**. The amber
ring marks an area to survey, not a modeled socket, opening or drilling target.
Green: custom XT30 extension → purple conditioning/protection allowance → Jetson J16 barrel plug. The GNB8504S60AHV pack has an XT30 discharge plug; approximate red/black factory leads and a balance connector are now shown. No conditioning or low-voltage protection circuit is selected.

## Which ports exist?

The current MicroDuck software identifies its computer as a Radxa ZERO 3W.
Radxa documents two USB-C connectors: USB-C 1 is power/USB 2.0 OTG; USB-C 2 is
USB 3.0 HOST. The proposed data link uses the **host port**, with a data-capable
USB-C-to-USB-C cable, into Jetson J5 in USB device networking mode. NVIDIA
supports virtual Ethernet on this connector. The Jetson still requires separate
DC barrel power; its USB-C connector does not power the developer kit.

This confirms board capability, **not access through the duck's enclosure**.
The pinned public simulation model puts two compute-board meshes in the head,
and one is explicitly an older Raspberry Pi Zero 2 W footprint. It does not
model the production Radxa USB sockets. Consequently this revision does not
invent an exterior port or drill/cut the stock shell. The head-side plug is
parked outside the reference shell, visibly disconnected.

Measure the actual board orientation, free host socket, shell openings and
available plug clearance before finalizing this end. Access may require opening
the head once and fitting a short extension through a verified cable exit.
Keep the robot's existing motor controller and power wiring intact. Transporting
policy observations/actions also requires the software bridge discussed earlier;
a cable alone does not add that feature.

## Geometric assumptions

- Jetson rotates 180° **in its tray plane**; USB, Ethernet and DC connector edge
  now faces upward. Its existing plastic base stays attached.
- J5/J16 connector centers come from NVIDIA's P3766/P3768/P3767 vendor STEP
  centroid annotations. Connector face planes are estimated from the vendor
  envelope, and the modeled plug bodies are generic overmold allowances.
- Approximate USB plug bodies: 12×21×7 mm at Jetson, 12×18×7 mm at the free
  head end. Cable diameter 4.5 mm. Main blue centerline is about 251 mm.
- Approximate DC plug body: 9 mm diameter ×22 mm. Power cable diameter 3 mm.
  Main power centerline is about 432 mm, plus a custom XT30 extension. Exact spline lengths are in `simulation/validation.json`.
- Curved service loops target at least 10 mm bend radius. These are sampled
  geometric radii, not cable-manufacturer approvals or a head travel test.
- Two outboard tray eyes accept cable ties for strain relief. Leave head-side
  slack free to move. Do not tie it across a joint or tension a board socket.
- Converter allowance 14×24×12 mm, strapped to the new front gate side pad;
  its electrical suitability is completely unverified. Replace its dimensions
  and mass with the selected converter before printing that pad.

Catalog search: step.parts queries `USB-C` and `USB C cable` returned sockets,
USB-C-to-micro-B, magnetic, panel-mount and micro-USB parts, but no matching
USB-C-to-USB-C cable. These simplified envelopes are labeled as such.

## Simulation and validation

The 74 g ancillary budget comprises hardware/pads/straps 22 g, USB cable 14 g,
power wiring 18 g (including the custom XT30 extension) and converter 20 g. Wiring mass and inertia use their modeled
HOME locations, with no duplicate ancillary lump. Cable visual meshes have zero
mass; stock battery leads/connectors are included in its specified 73 g and approximated at the body COM. The payload's explicit inertial contains all mass once. Cables are fixed
visuals in this approximation, without flexible tension, moving-head mass
redistribution or rigid cable collision forces.

`simulation/validation.json` records native cable/print intersection tests,
sampled spline lengths/radii and local capsule-to-robot distances in HOME.
It also records the hard mount's individual-joint sweeps. Full joint travel is
not cleared. No combination-motion, insertion, durability or live electrical
test has been performed.

## Primary references

- [Radxa ZERO 3 hardware interfaces](https://docs.radxa.com/en/zero/zero3/hardware-design/hardware-interface)
- [NVIDIA Orin Nano developer kit hardware layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/hardware_layout.html)
- [MicroDuck robotd design](https://github.com/pollen-robotics/microduck/blob/main/docs/design/robotd-design.md)
- Pinned robot and NVIDIA CAD provenance: parent `simulation/reference/provenance.json`
  and `STEP/imported/provenance.json`.

Battery source, power assumptions and runtime estimate: `battery_selection.json` and `README.md`. Factory lead lengths, XT30 and balance plug shapes are approximate fit envelopes.
