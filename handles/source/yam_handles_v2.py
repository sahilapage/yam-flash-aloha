"""Velcro-free kinesthetic "scissor" handles for the i2rt YAM linear_4310 gripper (v2).

Everything is 3D printed: a closed thumb ring on one jaw and a closed four-finger loop on the
other. Squeezing the hand closes the gripper, opening the hand opens it.

Gripper frame (mm), same as i2rt's linear_4310 MJCF:
  y  = jaw slide axis; jaw L (tip_left) sits at -y, jaw R (tip_right) at +y
  -z = the way the robot fingers point (fingertips at z = -144.7 when stock)
  +x = HANDLE SIDE, the side opposite the D405 wrist camera (the camera is at -x)

Each part is a 6 mm spacer clamped between a jaw's carriage foot and its stock finger by the
finger's own two M4 screws (use M4x16). The handle hangs off the spacer on the +x side.
Every feature is an extrusion along z, so each part prints with the carriage face on the bed
and needs no supports.

usage: python yam_handles_v2.py OUT_DIR
"""
import sys, os
import cadquery as cq

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---------------- measured from i2rt's linear_4310 meshes ----------------
ZT = -70.4            # carriage foot / finger interface plane
T = 6.0               # spacer thickness: 5 mm finger plate + 6 + 5 mm foot = M4x16, flush with the foot top
ZS = ZT - T           # top of the stock finger once the spacer is in
HOLE_D = 4.5          # M4 clearance
FOOT = {"L": dict(x=(-33.9, 12.6), y=(-23.2, -3.0), holes=[(8.29, -17.11), (-12.71, -17.11)]),
        "R": dict(x=(-12.6, 33.9), y=(3.0, 23.2), holes=[(-8.29, 17.11), (12.71, 17.11)])}
FINGER_X = 34.1       # stock finger plate reaches |x| = 34.1
FINGER_Y = 23.2       # ... and |y| = 23.2 on its outer face
ARM_X = (13.9, 33.9)  # jaw R's rack arm sweeps over jaw L's side here; its underside is at ZT
ARM_CLR = 2.0         # vertical clearance kept under that arm (the stock gripper runs 1.5 mm)

# ---------------- handle layout ----------------
H = 20.0              # handle height along the fingers
ZB = ZT - H
GAP = 2.0             # each handle stays this far from the jaw midplane -> 4 mm between handles when closed
X0 = FINGER_X + 1.1   # handles start here, clear of both stock fingers
RIB = (FINGER_Y + 1.0, 32.0)   # stiffening rib along the finger's outer face (|y| range)

# thumb ring (hole is a rounded rectangle, long axis x)
TH_L, TH_W, TH_R = 27.0, 23.0, 10.0
TH_WALL = 3.2         # wall between the thumb and the midplane
TH_XC = 59.0          # opposes the index / middle finger

# four-finger loop
SLOT_X = (42.0, 128.0)         # 86 mm: four fingers side by side
SLOT_D = 22.0                  # finger thickness room; reduce with the clip-on pads
SLOT_R = 8.0
BAR_T = 8.0                    # palm-side bar (takes the squeeze)
BACK_T = 6.0                   # back bar (takes the opening push)
END_T = 5.0

C_IN, C_OUT = 2.0, 1.5         # chamfers on the rims the hand touches: entry side, exit side
# -----------------------------------------------------------------------


def box(x, y, z):
    return (cq.Workplane("XY").box(x[1] - x[0], y[1] - y[0], z[1] - z[0], centered=False)
            .translate((x[0], y[0], z[0])))


def ybox(x, sign, yabs, z):
    """box given by |y| range on the side of the midplane picked by sign"""
    y = (yabs[0], yabs[1]) if sign > 0 else (-yabs[1], -yabs[0])
    return box(x, y, z)


def rrect_sketch(L, W, r):
    return cq.Sketch().rect(L, W).vertices().fillet(r)


def rprism(xc, yc, L, W, r, z0, z1):
    return (cq.Workplane("XY").workplane(offset=z0).center(xc, yc)
            .placeSketch(rrect_sketch(L, W, r)).extrude(z1 - z0))


def flare(xc, yc, L, W, r, z_face, c, direction):
    """cutter that turns a sharp hole rim into a c x 45 deg chamfer; direction = +1 if the solid lies above z_face"""
    eps = 0.02
    big = rrect_sketch(L + 2 * (c + eps), W + 2 * (c + eps), r + c + eps)
    small = rrect_sketch(L, W, r).moved(cq.Location(cq.Vector(0, 0, direction * (c + eps))))
    return (cq.Workplane("XY").workplane(offset=z_face - direction * eps).center(xc, yc)
            .placeSketch(big, small).loft())


def hole(part, xc, yc, L, W, r, z_top, z_bot):
    part = part.cut(rprism(xc, yc, L, W, r, z_bot - 1, z_top + 1))
    part = part.cut(flare(xc, yc, L, W, r, z_top, C_IN, -1))    # entry rim (hand comes in from +z)
    part = part.cut(flare(xc, yc, L, W, r, z_bot, C_OUT, +1))   # exit rim
    return part


def screw_holes(part, jaw):
    for hx, hy in FOOT[jaw]["holes"]:
        part = part.cut(cq.Workplane("XY").workplane(offset=ZB - 5).center(hx, hy).circle(HOLE_D / 2).extrude(40))
    return part


def spacer(jaw):
    f = FOOT[jaw]
    return box(f["x"], f["y"], (ZS, ZT))


def mount(jaw):
    """spacer + rib along the finger's outer face (+ the tongue under the other jaw's rack arm on jaw L)"""
    sign = -1 if jaw == "L" else +1
    f = FOOT[jaw]
    p = spacer(jaw)
    # rib: full height beside the finger, tied to the spacer along its whole outer edge
    p = p.union(ybox((f["x"][0], X0 + 0.01), sign, (FINGER_Y - 0.01, RIB[1]), (ZS, ZT)))
    p = p.union(ybox((f["x"][0], X0 + 0.01), sign, RIB, (ZB, ZT)))
    # tongue: the spacer plate carries on to the handle side. On jaw L it crosses the finger plate's
    # lower step, under jaw R's rack arm (finish() lowers its top there); on jaw R it is 1.3 mm long.
    ya = (-f["y"][1], -f["y"][0]) if jaw == "L" else f["y"]
    p = p.union(ybox((f["x"][1] - 0.01, X0 + 0.01), sign, ya, (ZS, ZT)))
    return p


def finish(p, jaw):
    sign = -1 if jaw == "L" else +1
    # stay clear of the stock finger (now 6 mm lower) everywhere below the spacer
    p = p.cut(ybox((-FINGER_X - 1.0, FINGER_X + 1.0), sign, (0.5, FINGER_Y + 1.0), (ZB - 1, ZS)))
    if jaw == "L":
        # relief under jaw R's rack arm, which slides over this part when the gripper is nearly closed
        p = p.cut(box((FOOT["L"]["x"][1], ARM_X[1] + 1.7), (-60, 60), (ZT - ARM_CLR, ZT + 1)))
    return screw_holes(p, jaw)


def thumb_part(jaw):
    sign = -1 if jaw == "L" else +1
    yc_hole = sign * (GAP + TH_WALL + TH_W / 2)
    x_out = (TH_XC - TH_L / 2 - TH_WALL, TH_XC + TH_L / 2 + TH_WALL)
    y_out = (GAP, RIB[1])
    p = mount(jaw)
    p = p.union(ybox((X0, TH_XC), sign, y_out, (ZB, ZT)))                                   # root block
    ring = rprism((x_out[0] + x_out[1]) / 2, sign * (y_out[0] + y_out[1]) / 2, x_out[1] - x_out[0], y_out[1] - y_out[0],
                  TH_R + TH_WALL, ZB, ZT)
    p = p.union(ring)
    p = hole(p, TH_XC, yc_hole, TH_L, TH_W, TH_R, ZT, ZB)
    return finish(p, jaw)


def loop_part(jaw):
    sign = -1 if jaw == "L" else +1
    y_slot = (GAP + BAR_T, GAP + BAR_T + SLOT_D)
    y_out = (GAP, y_slot[1] + BACK_T)
    x_out = (X0, SLOT_X[1] + END_T)
    p = mount(jaw)
    body = ybox(x_out, sign, y_out, (ZB, ZT))
    body = body.edges("|Z").edges(">X").fillet(9.0)                                          # round the far end
    p = p.union(body)
    p = hole(p, (SLOT_X[0] + SLOT_X[1]) / 2, sign * (y_slot[0] + y_slot[1]) / 2, SLOT_X[1] - SLOT_X[0], SLOT_D, SLOT_R, ZT, ZB)
    return finish(p, jaw)


def soften(p, name):
    """small chamfers on the outer edges of the handle (x > X0) at the entry and exit faces"""
    solid = p.val()
    k = 0.6 if "thumb" in name else 1.0          # the thumb wall is only 3.2 mm, keep its outer chamfers small
    for zf, c in ((ZB, 1.0 * k), (ZT, 0.8 * k)):
        yback = GAP + BAR_T + SLOT_D + BACK_T          # outer face of the finger loop's back bar
        edges = [e for e in solid.Edges()
                 if abs(e.BoundingBox().zmin - zf) < 1e-3 and abs(e.BoundingBox().zmax - zf) < 1e-3 and e.BoundingBox().xmin > X0 + 0.5
                 and not ("loop" in name and min(abs(e.BoundingBox().ymin), abs(e.BoundingBox().ymax)) > yback - 0.05)]
        try:
            solid2 = solid.chamfer(c, None, edges)
            if solid2.isValid():
                solid = solid2
            else:
                print(f"  {name}: outer chamfer at z={zf} gave an invalid solid, skipped")
        except Exception as e:
            print(f"  {name}: outer chamfer at z={zf} skipped ({type(e).__name__})")
    return cq.Workplane("XY").newObject([solid])


# ---------------- optional fit kit (also printed) ----------------
def thumb_liner(t):
    """split sleeve that snaps into the thumb ring and makes the hole 2t smaller each way"""
    L, W, r = TH_L - 0.3, TH_W - 0.3, TH_R - 0.15
    s = rprism(0, 0, L, W, r, 0, H)                                                          # z=0 is the entry face
    s = s.union(rprism(0, 0, L + 5.0, W + 5.0, r + 2.5, H, H + 1.2))                         # flange on the entry rim
    s = s.union(flare(0, 0, L, W, r, 0, 0.6, +1).intersect(rprism(0, 0, L + 1.3, W + 1.3, r + 0.65, 0, 0.62)))   # snap flare at the exit
    s = s.cut(rprism(0, 0, L - 2 * t, W - 2 * t, max(r - t, 2.0), -1, H + 3))
    s = s.cut(flare(0, 0, L - 2 * t, W - 2 * t, max(r - t, 2.0), H + 1.2, 1.2, -1))          # soft entry
    s = s.cut(box((L / 2 - t - 2, L / 2 + 4), (-2.5, 2.5), (-1, H + 3)))                     # split, on the far-x end
    return s


def finger_pad(t, length=32.0):
    """clip that snaps over the back bar of the finger loop and makes the slot t shallower"""
    ch, lip, fl = H + 0.3, 0.5, 1.2                                                          # channel height, lip, flange thickness
    prof = (cq.Workplane("YZ")
            .polyline([(-t, -fl), (BACK_T + 0.2 + fl, -fl), (BACK_T + 0.2 + fl, lip), (BACK_T + 0.2, lip), (BACK_T + 0.2, 0),
                       (0, 0), (0, ch), (BACK_T + 0.2, ch), (BACK_T + 0.2, ch - lip), (BACK_T + 0.2 + fl, ch - lip),
                       (BACK_T + 0.2 + fl, ch + fl), (-t, ch + fl)]).close().extrude(length))
    return prof.edges("|X").edges("<Y").fillet(min(1.5, t / 2 - 0.1))                         # round the two edges the fingers touch


PARTS = {
    "RH_thumb_ring_jawL": (thumb_part("L"), "L"),
    "RH_finger_loop_jawR": (loop_part("R"), "R"),
    "LH_finger_loop_jawL": (loop_part("L"), "L"),
    "LH_thumb_ring_jawR": (thumb_part("R"), "R"),
    "fit_test_spacer_jawL": (screw_holes(spacer("L"), "L"), "L"),
    "fit_test_spacer_jawR": (screw_holes(spacer("R"), "R"), "R"),
}
KIT = {
    "fitkit_thumb_liner_1p5mm": thumb_liner(1.5),
    "fitkit_thumb_liner_3mm": thumb_liner(3.0),
    "fitkit_finger_pad_3mm": finger_pad(3.0),
    "fitkit_finger_pad_6mm": finger_pad(6.0),
}

for d in ("step", "print_stl", "gripper_frame_stl"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

print(f"interface z={ZT}  spacer {T} mm  handle height {H} mm  closed gap {2*GAP} mm")
print(f"thumb-to-finger pad distance: {TH_WALL + 2*GAP + BAR_T:.1f} mm closed, {TH_WALL + 2*GAP + BAR_T + 95:.1f} mm fully open")
for name, (p, jaw) in PARTS.items():
    if "spacer" not in name:
        p = soften(p, name)
        PARTS[name] = (p, jaw)
    solid = p.val()
    bb = solid.BoundingBox()
    print(f"{name:24s} valid={solid.isValid()} solids={len(p.solids().vals())} vol={solid.Volume()/1000:5.1f} cm3 "
          f"x[{bb.xmin:.1f},{bb.xmax:.1f}] y[{bb.ymin:.1f},{bb.ymax:.1f}] z[{bb.zmin:.1f},{bb.zmax:.1f}]")
    cq.exporters.export(p, f"{OUT}/step/{name}.step")
    cq.exporters.export(p, f"{OUT}/gripper_frame_stl/{name}.stl", tolerance=0.02, angularTolerance=0.1)
    pr = p.rotate((0, 0, 0), (1, 0, 0), 180)                       # carriage face down on the bed
    b = pr.val().BoundingBox()
    pr = pr.translate((-b.xmin, -b.ymin, -b.zmin))
    cq.exporters.export(pr, f"{OUT}/print_stl/{name}.stl", tolerance=0.02, angularTolerance=0.1)

for name, p in KIT.items():
    solid = p.val()
    print(f"{name:24s} valid={solid.isValid()} solids={len(p.solids().vals())} vol={solid.Volume()/1000:5.1f} cm3")
    cq.exporters.export(p, f"{OUT}/step/{name}.step")
    if "liner" in name:
        pr = p.rotate((0, 0, 0), (1, 0, 0), 180)                   # flange down
    else:
        pr = p.rotate((0, 0, 0), (0, 1, 0), -90)                   # C profile on the bed, standing
    b = pr.val().BoundingBox()
    pr = pr.translate((-b.xmin, -b.ymin, -b.zmin))
    cq.exporters.export(pr, f"{OUT}/print_stl/{name}.stl", tolerance=0.02, angularTolerance=0.1)

for setname, a, b in (("RH_set", "RH_thumb_ring_jawL", "RH_finger_loop_jawR"), ("LH_set", "LH_finger_loop_jawL", "LH_thumb_ring_jawR")):
    asm = cq.Assembly(name=setname)
    asm.add(PARTS[a][0], name=a, color=cq.Color(0.93, 0.78, 0.13))
    asm.add(PARTS[b][0], name=b, color=cq.Color(0.95, 0.55, 0.15))
    asm.save(f"{OUT}/step/{setname}_closed_assembly.step")
print("done")
