"""Interference check of the v2 handles against i2rt's linear_4310 gripper over the full jaw stroke.
usage: check_v2.py ROBOT_MODELS_DIR PARTS_DIR(gripper_frame_stl) SCRIPTS_DIR [openings...]
For every opening it reports, per part:
  in  = surface samples of the obstacle that lie more than 0.15 mm inside the part, plus part samples inside the obstacle
  gap = smallest distance between the part and the obstacle (mm), from dense surface samples (resolution ~0.4 mm)
The spacer touches its own carriage foot and its own finger by design, so those two show gap 0 and are judged on `in` only."""
import sys, numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, sys.argv[3]); from yam import Yam
Y = Yam(sys.argv[1]); PD = sys.argv[2]; T = 6.0
ZI = -70.4   # carriage foot / finger interface plane
OPEN = [float(a) for a in sys.argv[4:]] or [0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 47.5]
SETS = {"RH": [("RH_thumb_ring_jawL", -1), ("RH_finger_loop_jawR", 1)], "LH": [("LH_finger_loop_jawL", -1), ("LH_thumb_ring_jawR", 1)]}
rng = np.random.default_rng(0)
def samp(m, n):
    p, _ = trimesh.sample.sample_surface(m, n, seed=1); return p
def capsplit(name):
    m = max(Y.tip(name, 0).split(only_watertight=False), key=lambda c: len(c.faces))
    car = trimesh.intersections.slice_mesh_plane(m, [0, 0, 1], [0, 0, ZI + 0.02], cap=True)       # carriage: everything above the interface
    fin = trimesh.intersections.slice_mesh_plane(m, [0, 0, -1], [0, 0, ZI - 0.02], cap=True); fin.apply_translation([0, 0, -T])
    return car, fin
class Ob:
    def __init__(s, name, mesh, jaw=0):
        s.name, s.m, s.jaw = name, mesh, jaw; s.wt = mesh.is_watertight
        s.pts = samp(mesh, 200000); s.tree = cKDTree(s.pts)
carL, finL = capsplit("tip_left"); carR, finR = capsplit("tip_right")
OBS = [Ob("body", Y.body), Ob("D405", Y.cam), Ob("carriageL", carL, -1), Ob("fingerL", finL, -1), Ob("carriageR", carR, 1), Ob("fingerR", finR, 1)]
print("watertight:", {o.name: bool(o.wt) for o in OBS})
worst = {}; cache = {}
for SET, parts in SETS.items():
    P = {n: trimesh.load(f"{PD}/{n}.stl") for n, _ in parts}
    PS = {n: samp(P[n], 60000) for n in P}; PT = {n: cKDTree(PS[n]) for n in P}
    for s in OPEN:
        row = []
        for n, sg in parts:
            off = np.array([0, sg * s, 0.0])
            for o in OBS:
                ooff = np.array([0, o.jaw * s, 0.0]); rel = off - ooff            # part position in the obstacle's frame
                own = o.jaw == sg
                ck = (n, o.name)
                if own and ck in cache:
                    gap, n_in = cache[ck]
                    w = worst[(SET, n, o.name)]; w["n_in"] += n_in
                    row.append(f"{o.name}:own in={n_in},gap={gap:.2f}"); continue
                d, _ = o.tree.query(PS[n] + rel); gap = float(d.min())
                # obstacle samples inside the part (part is a clean watertight CAD mesh)
                q = o.pts - rel; near = q[np.all((q > P[n].bounds[0]) & (q < P[n].bounds[1]), axis=1)]
                n_in = 0
                if gap > 2.5:       # far apart: only make sure neither sits wholly inside the other
                    near = near[:200]
                if len(near):
                    ins = P[n].contains(near)
                    if ins.any():
                        dd, _ = PT[n].query(near[ins]); n_in = int((dd > 0.15).sum())
                # part samples inside the obstacle
                if o.wt:
                    cand = (PS[n][d < 6.0] if gap <= 2.5 else PS[n][::300]) + rel
                    if len(cand):
                        ins = o.m.contains(cand)
                        if ins.any():
                            dd, _ = o.tree.query(cand[ins]); n_in += int((dd > 0.15).sum())
                if own: cache[ck] = (gap, n_in)
                key = (SET, n, o.name)
                w = worst.setdefault(key, dict(gap=1e9, s=None, n_in=0, own=own)); w["n_in"] += n_in
                if gap < w["gap"]: w["gap"], w["s"] = gap, s
                row.append(f"{o.name}:{'own ' if own else ''}in={n_in},gap={gap:.2f}")
            print(f"{SET} s={s:5.1f} {n:22s} " + "  ".join(row[-len(OBS):]), flush=True)
        # the two handle parts against each other
        (a, sa), (b, sb) = parts
        d, _ = PT[b].query(PS[a] + [0, (sa - sb) * s, 0]); g = float(d.min())
        ins = P[b].contains(PS[a] + [0, (sa - sb) * s, 0]).sum()
        w = worst.setdefault((SET, "handle-to-handle", ""), dict(gap=1e9, s=None, n_in=0, own=False)); w["n_in"] += int(ins)
        if g < w["gap"]: w["gap"], w["s"] = g, s
        print(f"{SET} s={s:5.1f} handle-to-handle gap={g:.2f} in={int(ins)}", flush=True)
print("\n=== SUMMARY (worst case over all openings) ===")
bad = 0
for (SET, n, o), w in worst.items():
    flag = "OK"
    if w["n_in"] > 0: flag = "PENETRATION"; bad += 1
    elif not w["own"] and w["gap"] < 0.8: flag = "TIGHT"; bad += 1
    print(f"{SET} {n:22s} vs {o:10s} {'(own jaw, touches by design) ' if w['own'] else ''}min gap {w['gap']:.2f} mm at s={w['s']}  samples inside: {w['n_in']}  {flag}")
print("RESULT:", "PASS" if bad == 0 else f"{bad} PROBLEM(S)")
