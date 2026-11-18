"""Photos of the v2 (velcro-free) YAM handles.
usage: render_v2.py ROBOT_MODELS_DIR BUILD_DIR OUT_DIR SCRIPTS_DIR [FLASH_GLB] [draft] [only=01 ...] [hold=x,y,z] [cmp=x,y,z]"""
import sys, os, numpy as np, trimesh, tempfile
RM, BD, OUT, SD = sys.argv[1:5]
GLB = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5].endswith(".glb") else None
DRAFT = "draft" in sys.argv
ONLY = [a[5:] for a in sys.argv if a.startswith("only=")]
def want(tag): return not ONLY or tag in ONLY
def vec(key, default): return tuple(float(v) for v in next((a[len(key) + 1:] for a in sys.argv if a.startswith(key + "=")), default).split(","))
sys.path.insert(0, SD)
import rlib, cadquery as cq
from yam import Yam
from PIL import Image, ImageDraw
os.makedirs(OUT, exist_ok=True)
Y = Yam(RM); GF = f"{BD}/gripper_frame_stl"
T, ZT, H = 6.0, -70.4, 20.0; ZB = ZT - H; ZS = ZT - T
GAP, BAR_T, SLOT_D, BACK_T, TH_WALL, TH_W, TH_XC = 2.0, 8.0, 22.0, 6.0, 3.2, 23.0, 59.0
YEL, BODY, FING, CAMC, STEEL, SKIN, KIT = (0.93, 0.78, 0.13), (0.20, 0.20, 0.21), (0.10, 0.10, 0.11), (0.14, 0.14, 0.15), (0.62, 0.63, 0.66), (0.95, 0.76, 0.64), (0.25, 0.52, 0.85)
SETS = {"RH": [("RH_thumb_ring_jawL", -1, "thumb"), ("RH_finger_loop_jawR", 1, "loop")],
        "LH": [("LH_finger_loop_jawL", -1, "loop"), ("LH_thumb_ring_jawR", 1, "thumb")]}
HOLES = {-1: [(8.29, -17.11), (-12.71, -17.11)], 1: [(-8.29, 17.11), (12.71, 17.11)]}
SC = 0.5 if DRAFT else 1.0
UP = (0, 0, 1)
VIEW = (1.0, 0.55, 0.62)                    # from the handle (operator) side, a little above
VIEW2 = (1.0, -0.75, -0.55)                 # from the handle side, looking up from the fingertip end
HOLDVIEW = vec("hold", "0.9,-1.0,0.35")
CMPVIEW = vec("cmp", "0.9,-0.2,1.0")


def step_mesh(name):
    w = cq.importers.importStep(f"{BD}/step/{name}.step"); f = tempfile.mktemp(suffix=".stl")
    cq.exporters.export(w, f, tolerance=0.03, angularTolerance=0.1); m = trimesh.load(f); os.remove(f); return m


def capsule(p0, p1, r):
    p0, p1 = np.array(p0, float), np.array(p1, float); v = p1 - p0; L = np.linalg.norm(v)
    c = trimesh.creation.capsule(height=L, radius=r, count=[24, 24]); c.apply_translation([0, 0, -L / 2])
    c.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], v / L)); c.apply_translation((p0 + p1) / 2); return c


def screw():
    head = trimesh.creation.cylinder(3.5, 4, sections=48); head.apply_translation([0, 0, -2])
    shaft = trimesh.creation.cylinder(2.0, 16, sections=32); shaft.apply_translation([0, 0, 8])
    return trimesh.util.concatenate([head, shaft])


def hand(SET, s, op=0.32):
    """rough see-through hand: fingers through the loop, thumb through the ring, palm and wrist behind (toward the robot's wrist)"""
    st = [sg for _, sg, k in SETS[SET] if k == "thumb"][0]; sl = -st
    yf = sl * (GAP + BAR_T + SLOT_D / 2 + s); yt = st * (GAP + TH_WALL + TH_W / 2 + s)
    it = []
    for x, tip, r in ((53, -116, 9.0), (73, -121, 9.0), (93, -116, 8.5), (112, -103, 7.5)):
        it.append(capsule((x, yf, tip + r), (x, yf, -22), r))
    palm = trimesh.creation.icosphere(subdivisions=3); palm.apply_scale([44, 15, 46]); palm.apply_translation([83, yf, 20]); it.append(palm)
    it.append(capsule((TH_XC, yt, -96), (TH_XC, yt, -42), 10.0))                      # thumb through the ring
    it.append(capsule((TH_XC, yt, -42), (72, yf - sl * 6, 42), 11.5))                 # thumb base
    it.append(capsule((86, yf, 58), (93, yf + sl * 3, 128), 25))                      # wrist stub
    return [dict(m=m, c=SKIN, op=op, edges=False) for m in it]


def scene(SET, s, hand_on=False, explode=False, camera=True, kit=False):
    E = (36.0, 84.0, 180.0) if explode else (0.0, 0.0, 0.0)
    it, an = [], {}
    it.append(dict(m=Y.body, c=BODY, spec=0.25)); an["YAM linear_4310 gripper"] = Y.body.vertices[Y.body.vertices[:, 2] > -45]
    if camera: it.append(dict(m=Y.cam, c=CAMC, spec=0.3)); an["D405 wrist camera (far side)"] = Y.cam.vertices
    for nm, sg in (("tip_left", -1), ("tip_right", 1)):
        car, fin = Y.split(nm, s, shift=T + E[1]); it += [dict(m=car, c=FING, spec=0.3), dict(m=fin, c=FING, spec=0.3)]
        if sg == 1: an["stock finger (sits 6 mm lower)"] = fin.vertices[fin.vertices[:, 2] < -108 - E[1]]
    for n, sg, kind in SETS[SET]:
        m = trimesh.load(f"{GF}/{n}.stl"); m.apply_translation([0, sg * s, -E[0]]); it.append(dict(m=m, c=YEL, spec=0.35))
        v = m.vertices; sel = v[(np.abs(v[:, 0]) < 33) & (v[:, 2] > ZS - E[0] - 0.5)]
        k = "6 mm spacer (clamped under the finger)"
        an[k] = sel if k not in an else np.vstack([an[k], sel])
        if kind == "thumb":
            an["thumb ring"] = v[(v[:, 0] > TH_XC - 4)]
            if kit:
                l = step_mesh("fitkit_thumb_liner_1p5mm"); l.apply_translation([TH_XC, sg * (GAP + TH_WALL + TH_W / 2 + s), ZB - E[0]]); it.append(dict(m=l, c=KIT, spec=0.3))
                an["thumb liner (fit kit)"] = l.vertices[l.vertices[:, 2] > ZT - E[0]]
        else:
            an["four-finger loop"] = v[(v[:, 0] > 100)]
            if kit:
                for x0 in (50.0, 88.0):
                    p = step_mesh("fitkit_finger_pad_3mm")
                    if sg < 0: p.apply_transform(np.diag([1, -1, 1, 1])); p.invert()
                    p.apply_translation([x0, sg * (GAP + BAR_T + SLOT_D + s), ZB - 0.15 - E[0]]); it.append(dict(m=p, c=KIT, spec=0.3))
                an["finger pad clips (fit kit)"] = p.vertices
        if explode:
            for hx, hy in HOLES[sg]:
                p0 = np.array([hx, hy + sg * s, ZS - 5 - E[2]]); sc = screw(); sc.apply_translation(p0); it.append(dict(m=sc, c=STEEL, spec=0.8))
                it.append(dict(line=(p0 + [0, 0, 16], [p0[0], p0[1], ZT + 6]), c=(0.6, 0.6, 0.6)))
            an["M4x16 screws (2 per jaw)"] = np.array([p0 + [0, 0, -3]])
    if hand_on:
        it += hand(SET, s)
    return it, an


made = []
def go(name, items, d, up, title=None, sub=None, anchors=None, **kw):
    kw.setdefault("W", 2600); kw.setdefault("H", 1900); kw["W"] = int(kw["W"] * SC); kw["H"] = int(kw["H"] * SC)
    pts = rlib.render(items, d, up, f"{OUT}/{name}", anchors=anchors, **kw)
    if title or anchors: rlib.callouts(f"{OUT}/{name}", pts, title, sub)
    made.append(name); print("wrote", name, flush=True)


if want("01"):
    for SET, nm in (("RH", "right hand"), ("LH", "left hand")):
        it, an = scene(SET, 20)
        go(f"01_hero_{SET}.png", it, VIEW if SET == "RH" else (1.0, -0.55, 0.62), UP, f"YAM kinesthetic handles v2 ({nm} set): no velcro, fully 3D printed",
           "yellow = printed  |  each jaw 20 mm open  |  handles sit on the side opposite the D405", anchors=an, zoom=1.35)
if want("02"):
    it, an = scene("RH", 20)
    go("02_underside_RH.png", it, VIEW2, UP, "Same set seen from the fingertip end", "your fingers and thumb come in from the wrist side and point the same way as the robot's fingers",
       anchors={k: v for k, v in an.items() if k in ("thumb ring", "four-finger loop", "stock finger (sits 6 mm lower)")}, zoom=1.3)
if want("03"):
    for s, tag, ttl in ((38, "open", "hand open = gripper open"), (2, "closed", "hand squeezed = gripper closed")):
        it, an = scene("RH", s, hand_on=True)
        go(f"03_hold_RH_{tag}.png", it, HOLDVIEW, UP, "How to hold it: " + ttl, "thumb through the ring, four fingers through the loop; your hand sits behind the robot's fingers, like a pistol grip",
           anchors={k: v for k, v in an.items() if k in ("thumb ring", "four-finger loop", "D405 wrist camera (far side)")}, zoom=1.1)
if want("04"):
    it, an = scene("RH", 30, explode=True)
    go("04_exploded_RH.png", it, (1, 0.9, -0.35), (0, 1, 0), "Fitting: the spacer goes between the jaw carriage and the stock finger",
       "same 2 holes per jaw; M4x16 socket screws go in straight from the fingertip side", anchors={k: v for k, v in an.items() if "gripper" not in k}, zoom=1.25, W=3000)
if want("05"):
    for SET in ("RH", "LH"):
        for s in (0, 24, 47.5):
            it, _ = scene(SET, s); go(f"05_stroke_{SET}_s{s:g}.png", it, VIEW if SET == "RH" else (1.0, -0.55, 0.62), UP, W=1800, H=1400, zoom=1.3)
if want("06"):
    it, an = scene("RH", 20, kit=True)
    go("06_fit_kit_RH.png", it, VIEW, UP, "Optional fit kit (also printed): snap-in parts that tighten the fit for smaller hands",
       "blue = thumb liner (1.5 or 3 mm) and finger pad clips (3 or 6 mm)", anchors={k: v for k, v in an.items() if "fit kit" in k}, zoom=1.35)
if want("07"):
    for nm, d, up in (("handle_side", (1, 0, 0), (0, 0, 1)), ("from_wrist", (0, 0, 1), (-1, 0, 0)), ("along_slide", (0, -1, 0), (0, 0, 1))):
        it, _ = scene("RH", 20); go(f"07_ortho_RH_{nm}.png", it, d, up, W=1800, H=1400, zoom=1.2, parallel=True)
if want("08"):
    # parts as they sit on the print bed
    bed = trimesh.creation.box([256, 256, 2]); bed.apply_translation([128, 128, -1]); items = [dict(m=bed, c=(0.83, 0.84, 0.86), spec=0.1, edges=False)]
    y = 14
    for row in (["LH_finger_loop_jawL"], ["RH_finger_loop_jawR"], ["RH_thumb_ring_jawL", "LH_thumb_ring_jawR"], ["fit_test_spacer_jawL", "fit_test_spacer_jawR", "fitkit_thumb_liner_1p5mm", "fitkit_thumb_liner_3mm"],
                ["fitkit_finger_pad_3mm", "fitkit_finger_pad_3mm", "fitkit_finger_pad_6mm", "fitkit_finger_pad_6mm"]):
        x = 14; h = 0
        for n in row:
            m = trimesh.load(f"{BD}/print_stl/{n}.stl"); b = m.bounds; m.apply_translation([x - b[0, 0], y - b[0, 1], -b[0, 2]]); items.append(dict(m=m, c=KIT if "fitkit" in n else YEL, spec=0.35))
            x += (b[1, 0] - b[0, 0]) + 12; h = max(h, b[1, 1] - b[0, 1])
        y += h + 10
    go("08_print_bed.png", items, (0.45, -1, 1.25), (0, 0, 1), "Everything as it prints (256 mm bed)", "carriage face down, no supports  |  yellow = handles and fit-test spacers, blue = optional fit kit", W=2200, H=1700, zoom=1.2)
if GLB and want("09"):
    # side by side with Flash ALOHA's own leads (their left-hand export), rotated into the same frame
    sc = trimesh.load(GLB); M = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    COL = {"frame": (0.55, 0.57, 0.6), "adapter": (0.35, 0.37, 0.4), "finger": (0.10, 0.10, 0.11), "lead": YEL, "strap": (0.25, 0.27, 0.32)}
    fa = []
    for name in sc.graph.nodes_geometry:
        Tm, g = sc.graph[name]; m = sc.geometry[g].copy(); m.apply_transform(Tm); m.apply_scale(1000); m.apply_transform(M)
        fa.append(dict(m=m, c=next(v for k, v in COL.items() if name.startswith(k)), spec=0.25))
    rlib.render(fa, CMPVIEW, UP, f"{OUT}/_fa.png", W=int(2000 * SC), H=int(1700 * SC), zoom=1.2)
    it, _ = scene("LH", 3, camera=False); rlib.render(it, CMPVIEW, UP, f"{OUT}/_ours.png", W=int(2000 * SC), H=int(1700 * SC), zoom=1.2)
    a, b = Image.open(f"{OUT}/_fa.png"), Image.open(f"{OUT}/_ours.png"); hh = max(a.height, b.height); fs = int(46 * SC) + 10
    f1, f2 = rlib.font(fs), rlib.font(int(fs * 0.75)); dr0 = ImageDraw.Draw(a)
    wl = int(max(a.width, dr0.textlength("Flash ALOHA leads (their model, Seeed arm gripper)", font=f1) + 60))
    wr = int(max(b.width, dr0.textlength("closed thumb ring + closed four-finger loop, no straps", font=f2) + 60, dr0.textlength("This design (YAM linear_4310 gripper)", font=f1) + 60))
    can = Image.new("RGB", (wl + wr + 120, hh + fs * 4), "white"); can.paste(a, ((wl - a.width) // 2, fs * 4 + (hh - a.height) // 2)); can.paste(b, (wl + 120 + (wr - b.width) // 2, fs * 4 + (hh - b.height) // 2))
    dr = ImageDraw.Draw(can); dr.text((30, 20), "Flash ALOHA leads (their model, Seeed arm gripper)", fill=(15, 15, 15), font=f1)
    dr.text((30, 20 + fs * 1.5), "printed hook + bar, each closed by velcro straps (3 in all)", fill=(90, 90, 90), font=f2)
    dr.text((wl + 150, 20), "This design (YAM linear_4310 gripper)", fill=(15, 15, 15), font=f1)
    dr.text((wl + 150, 20 + fs * 1.5), "closed thumb ring + closed four-finger loop, no straps", fill=(90, 90, 90), font=f2)
    can.save(f"{OUT}/09_vs_flash_aloha.png"); os.remove(f"{OUT}/_fa.png"); os.remove(f"{OUT}/_ours.png"); made.append("09_vs_flash_aloha.png"); print("wrote 09_vs_flash_aloha.png")
if not ONLY:
    rlib.grid([f"{OUT}/{n}" for n in made], f"{OUT}/00_contact_sheet.png", cols=4, cell=(900, 680), labels=[n[:-4] for n in made])
    print("wrote 00_contact_sheet.png")
