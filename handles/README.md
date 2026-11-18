# YAM kinesthetic handles v2: no velcro, fully 3D printed

A closed thumb ring on one jaw and a closed four-finger loop on the other, for the i2rt YAM `linear_4310` gripper. Squeeze your hand and the gripper closes; open your hand and it opens. This replaces an earlier v1 design (thumb ring + finger bar with velcro).

Start with `photos/00_contact_sheet.png`.

## What to print (`print_stl/`)

| File | Use |
|---|---|
| `RH_thumb_ring_jawL` + `RH_finger_loop_jawR` | right-hand set (right arm) |
| `LH_finger_loop_jawL` + `LH_thumb_ring_jawR` | left-hand set (left arm) |
| `fit_test_spacer_jawL`, `fit_test_spacer_jawR` | print these first (10 min each) to check the hole pattern on the real gripper |
| `fitkit_thumb_liner_1p5mm`, `fitkit_thumb_liner_3mm` | optional: snap-in sleeve that makes the thumb ring smaller |
| `fitkit_finger_pad_3mm`, `fitkit_finger_pad_6mm` | optional: clip for the loop's back bar that makes the finger slot shallower (2 per loop) |

- Every STL is already oriented for printing, sitting on the bed. **No supports.**
- The two jaw-L parts have one bridge: 23 mm long, 2 mm above the bed, held at both ends. It is the relief for the other jaw's rack arm.
- PETG or PLA+, 0.2 mm layers, 4 walls, 40–50 % infill.
- Print the finger pad clips in PETG, standing on end with a brim. They flex to snap on, and PLA may crack.
- Weight at that infill: about 50 g per finger loop and 30 g per thumb ring (solid volumes are 45–50 cm³ and 25–30 cm³).
- `step/` has the same parts for Onshape (mm, gripper frame). `gripper_frame_stl/` has them as meshes in the gripper frame.

## Hardware

- 4 × **M4×16** socket head cap screws per gripper (2 per jaw).
  - The length comes from the stack: 5 mm finger plate + 6 mm spacer + 5 mm carriage foot.
  - Do not use anything longer than M4×18: the other jaw's rack passes 4 mm above the foot.
- No straps.

## Fitting

1. Hold the gripper with the robot fingers pointing down and the D405 on the far side. The handles go on the side facing you. Jaw L is on your left, jaw R on your right.
2. Remove one stock finger: 2 screws, straight in from the fingertip side.
3. Put the part's spacer between the jaw's carriage foot and the finger. Refit with M4×16 and tighten until snug.
4. Repeat on the other jaw. Slide the jaws by hand through the full stroke; the handles stay 4 mm apart when closed.
5. Add 6 mm to the TCP / grasp site in your FK, and re-tune gravity compensation.
6. Leave the handles on during policy rollouts, so the cameras see the same gripper as in the demos.

## How to hold it

- Thumb through the ring, four fingers through the loop. Both go in from the robot-wrist side and point the same way as the robot's fingers.
- Your hand sits behind the robot's fingers and your forearm runs parallel to them, the way Flash ALOHA operators hold their leads.
- Pushing forward, the web of your hand bears on the back of the handles. Pulling back, curl your fingertips under the loop's inner bar.
- Thumb-to-finger distance is 15 mm with the gripper closed and 110 mm fully open. Small hands may not reach full open; sliding the loops toward your fingertips gives more reach.
- If the fit is loose, add fit-kit parts until it is snug. A loose fit rattles in fast motions.

## What changed from v1, and why

| | Flash ALOHA leads (Seeed arm) | v1 (velcro) | v2 (this) |
|---|---|---|---|
| Thumb | printed C-hook + 1 velcro strap | closed ring 30 × 24 mm | closed ring 27 × 23 mm, optional liners |
| Fingers | 86 mm bar + 2 velcro straps | 86 mm bar + velcro strap around 2 fingers | closed loop, 86 × 22 mm slot for all 4 fingers, optional pads |
| Opening the gripper | thumb and fingers push on velcro | thumb pushes on its ring, fingers on velcro | thumb pushes on its ring, fingers on a rigid back bar |
| Thumb-to-finger distance, closed | about 35 mm | 17 mm | 15 mm |
| Attachment | plate between slider and finger, 4 × M3×20 | 6 mm spacer, 2 × M4 | same spacer, now tied to the handle by a rib and a tongue on both jaws |
| Handle start, from the gripper centre | about 48 mm | 44 mm | 35 mm (ring centre 59 mm, was 63 mm) |

Measured from Flash ALOHA's own 3D model (`research/`), their demo videos and i2rt's meshes:

- **Same layout as theirs.** Handles hang off the side opposite the wrist camera, the operator's fingers run parallel to the robot's, and thumb and fingers oppose along the jaw axis. The YAM kinematics put that side toward an operator standing behind the arm.
- **Rigid loops replace the straps.** In their design the printed part only backs the squeeze; velcro does the opening. Here a printed back bar and a full ring do both.
- **Spacer stays 6 mm.** It makes the screw exactly M4×16 with full thread engagement.
- **Stiffer mount.** In v1 each handle joined its spacer through a single path: a side strip on one jaw, the spacer's short end on the other. Both jaws now have a rib along the finger's outer face plus a tongue from the spacer.
- **Not copied:** their leads sit 20–60 mm further back toward the wrist and tilt back by about 12°. That shape needs supports to print and gives a weaker mount on this gripper. The cost: your fingertips end about 30 mm short of the robot's fingertips instead of roughly 65 mm on theirs (my estimate from their model).

## Check results (`test_results/`)

Run against i2rt's `linear_4310` meshes with the fingers moved down 6 mm, at 17 jaw openings from closed to fully open, both hand sets.

- No overlap with the gripper body, the D405, either jaw or the other handle.
- Smallest clearances: body 6.9 mm, D405 17.7 mm, other jaw's finger 5.1 mm, handle to handle 4.0 mm (closed) to 99 mm (open).
- Closest pass is 1.3 mm sideways between each spacer and the other jaw's rack arm. The stock gripper runs the same 1.3 mm there.
- All print STLs are watertight and sit flat on the bed; all STEP files re-import as valid solids.
- The fit-kit liners and pad clips were checked seated in the handles: no overlap.

## Not verified

- **The real gripper.** The meshes are simplified (about ±0.3 mm). Print a fit-test spacer first.
- **Screw length.** M4×16 assumes the carriage foot is 5 mm thick, as the mesh shows. Check that the screw tip does not stand proud of the foot by more than 2 mm.
- **Backdrive force.** The YAM gripper moves 14.6 mm per radian of motor travel; the ARX grippers Flash ALOHA uses move about 17.5 mm. Expect about 20 % more hand force, and i2rt has no friction compensation for the gripper motor.
- **Fit-kit snap fits.** Printed clips depend on your printer's tolerances; they may need a light sand.

## Safety

Rigid loops do not release the way velcro does. Never run the gripper under position control, or replay a trajectory, with your hand in the loops.

## Editing

`source/yam_handles_v2.py` is parametric (CadQuery): ring size, loop length and depth, spacer thickness and gaps are constants at the top. `source/check_v2.py` reruns the interference check and `source/render_v2.py` remakes the photos.
