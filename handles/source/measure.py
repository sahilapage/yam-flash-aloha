import sys, numpy as np, trimesh
sys.path.insert(0, sys.argv[2]); from yam import Yam
Y = Yam(sys.argv[1])
def levels(m, box, up=True, tol=0.97, name=""):
    c = m.triangles_center; n = m.face_normals; a = m.area_faces
    (x0, x1), (y0, y1), (z0, z1) = box
    k = (c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1) & (c[:, 2] > z0) & (c[:, 2] < z1) & ((n[:, 2] > tol) if up else (n[:, 2] < -tol))
    zs = np.round(c[k, 2], 1); out = {}
    for z, ar in zip(zs, a[k]): out[z] = out.get(z, 0) + ar
    print(f"  {name:46s}", {float(z): round(float(ar), 1) for z, ar in sorted(out.items(), key=lambda t: -t[1])[:6] if ar > 3})
for nm, jaw, ys in (("tip_left", "L", (-24, -2)), ("tip_right", "R", (2, 24))):
    m = Y.tip(nm, 0); big = max(m.split(only_watertight=False), key=lambda c: len(c.faces)); scr = [c for c in m.split(only_watertight=False) if len(c.faces) < 1000]
    print(nm, "(jaw", jaw + ")  screws:", [(c.bounds.mean(0).round(2).tolist(), (c.bounds[1] - c.bounds[0]).round(2).tolist()) for c in scr])
    xin = (13, 34) if jaw == "L" else (-34, -13)       # finger-plate region NOT under own foot
    xft = (-34, 12.6) if jaw == "L" else (-12.6, 34)   # own foot footprint
    levels(big, (xin, ys, (-80, -68)), True, name="finger plate top, beyond own foot (up faces)")
    levels(big, (xft, ys, (-74, -68)), True, name="up faces under own foot footprint")
    levels(big, (xft, ys, (-74, -62)), False, name="down faces in own foot footprint")
    levels(big, ((-34, 34), ys, (-90, -72)), False, name="finger plate underside (down faces)")
    rk = (-34, -13) if jaw == "L" else (13, 34); yr = (0, 50) if jaw == "L" else (-50, 0)
    levels(big, (rk, yr, (-75, -55)), False, name="own rack bar underside (down faces)")
    levels(big, (rk, yr, (-75, -50)), True, name="own rack bar top (up faces)")
    v = big.vertices; f = v[(v[:, 2] < -70.4)]
    print("   finger (below -70.4) bounds:", f.min(0).round(1).tolist(), f.max(0).round(1).tolist())
    k = (v[:, 2] > -70.2); print("   carriage (above -70.2) bounds:", v[k].min(0).round(1).tolist(), v[k].max(0).round(1).tolist())
    rb = v[(v[:, 2] > -70.2) & ((v[:, 1] > 2) if jaw == "L" else (v[:, 1] < -2))]; print("   rack bar beyond the midplane: x", rb[:, 0].min().round(1), rb[:, 0].max().round(1), " y", rb[:, 1].min().round(1), rb[:, 1].max().round(1), " z", rb[:, 2].min().round(2), rb[:, 2].max().round(1))
    for z in (-72.5, -76, -79, -81, -83, -86, -89, -92, -96, -100):
        seg = trimesh.intersections.mesh_plane(big, [0, 0, 1], [0, 0, z]); p = seg.reshape(-1, 3)
        print(f"   z={z:6.1f}: x[{p[:,0].min():6.1f},{p[:,0].max():6.1f}] y[{p[:,1].min():6.1f},{p[:,1].max():6.1f}]", end="")
        mid = p[np.abs(p[:, 0]) < 9]; print(f"   centre (|x|<9) y range: {('[%.1f, %.1f]' % (mid[:,1].min(), mid[:,1].max())) if len(mid) else 'open'}")
b = Y.body; v = b.vertices
print("body: bottom z", v[:, 2].min().round(2), " +x side max x for z<-40:", v[v[:, 2] < -40][:, 0].max().round(2), " for -63.5<z<-55:", v[(v[:, 2] < -55)][:, 0].max().round(2))
