"""Small offscreen CAD-style renderer (VTK) + label helper."""
import numpy as np, trimesh, vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from PIL import Image, ImageOps, ImageDraw, ImageFont
FONT = "/System/Library/Fonts/Helvetica.ttc"
def font(sz):
    try: return ImageFont.truetype(FONT, sz)
    except Exception: return ImageFont.load_default()
def poly(m):
    pts = vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(m.vertices, dtype=np.float64), deep=1))
    f = np.hstack([np.full((len(m.faces), 1), 3), m.faces]).astype(np.int64).ravel()
    ca = vtk.vtkCellArray(); ca.SetCells(len(m.faces), numpy_to_vtkIdTypeArray(f, deep=1)); p = vtk.vtkPolyData(); p.SetPoints(pts); p.SetPolys(ca); return p
def render(items, d, up, png, W=2400, H=1800, zoom=1.0, parallel=False, anchors=None, margin=90, bg=(1, 1, 1), crop=True, focus=None):
    """items: dict(m=trimesh, c=(r,g,b), op=1.0, spec=0.2, edges=True) or dict(line=(p0,p1)). Returns label pixel positions."""
    ren = vtk.vtkRenderer(); ren.SetBackground(*bg); ren.SetUseFXAA(True); ren.SetUseDepthPeeling(1); ren.SetMaximumNumberOfPeels(8)
    camera = vtk.vtkCamera(); ren.SetActiveCamera(camera)
    for it in items:
        if "line" in it:
            v = it["line"]; ls = vtk.vtkLineSource(); ls.SetPoint1(*v[0]); ls.SetPoint2(*v[1])
            mp = vtk.vtkPolyDataMapper(); mp.SetInputConnection(ls.GetOutputPort()); a = vtk.vtkActor(); a.SetMapper(mp)
            a.GetProperty().SetColor(*it.get("c", (0.55, 0.55, 0.55))); a.GetProperty().SetLineWidth(it.get("w", W / 1200)); a.GetProperty().LightingOff(); ren.AddActor(a); continue
        m, col, op = it["m"], it["c"], it.get("op", 1.0); p = poly(m)
        nrm = vtk.vtkPolyDataNormals(); nrm.SetInputData(p); nrm.SetFeatureAngle(it.get("fa", 32)); nrm.SplittingOn(); nrm.ConsistencyOn(); nrm.Update()
        mp = vtk.vtkPolyDataMapper(); mp.SetInputConnection(nrm.GetOutputPort()); a = vtk.vtkActor(); a.SetMapper(mp)
        pr = a.GetProperty(); pr.SetColor(*col); pr.SetOpacity(op); pr.SetInterpolationToPhong(); pr.SetAmbient(it.get("amb", 0.18)); pr.SetDiffuse(0.85); pr.SetSpecular(it.get("spec", 0.2)); pr.SetSpecularPower(40)
        ren.AddActor(a)
        if it.get("edges", True):
            fe = vtk.vtkFeatureEdges(); fe.SetInputData(p); fe.BoundaryEdgesOff(); fe.ManifoldEdgesOff(); fe.NonManifoldEdgesOff(); fe.SetFeatureAngle(55); fe.ColoringOff()
            em = vtk.vtkPolyDataMapper(); em.SetInputConnection(fe.GetOutputPort()); em.SetResolveCoincidentTopologyToPolygonOffset()
            ea = vtk.vtkActor(); ea.SetMapper(em); ea.GetProperty().SetColor(*(np.array(col) * 0.45)); ea.GetProperty().SetLineWidth(max(1, W / 1600)); ea.GetProperty().LightingOff(); ren.AddActor(ea)
            sil = vtk.vtkPolyDataSilhouette(); sil.SetInputData(p); sil.SetCamera(camera); sil.BorderEdgesOff()
            sm = vtk.vtkPolyDataMapper(); sm.SetInputConnection(sil.GetOutputPort()); sm.SetResolveCoincidentTopologyToPolygonOffset()
            sa = vtk.vtkActor(); sa.SetMapper(sm); sa.GetProperty().SetColor(0.05, 0.05, 0.05); sa.GetProperty().SetLineWidth(max(1.5, W / 900)); sa.GetProperty().LightingOff(); ren.AddActor(sa)
    d = np.array(d, float); d /= np.linalg.norm(d)
    allv = np.vstack([it["m"].vertices for it in items if "m" in it and not it.get("nofit")]); c = (allv.min(0) + allv.max(0)) / 2 if focus is None else np.array(focus, float)
    camera.SetFocalPoint(*c); camera.SetPosition(*(c + 800 * d)); camera.SetViewUp(*up); camera.SetViewAngle(18)
    if parallel: camera.ParallelProjectionOn()
    lk = vtk.vtkLightKit(); lk.AddLightsToRenderer(ren); ren.ResetCamera(); camera.Zoom(zoom)
    rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1); rw.SetAlphaBitPlanes(1); rw.SetMultiSamples(0); rw.AddRenderer(ren); rw.SetSize(W, H); rw.Render()
    wi = vtk.vtkWindowToImageFilter(); wi.SetInput(rw); wi.ReadFrontBufferOff(); wi.Update()
    w = vtk.vtkPNGWriter(); w.SetFileName(png); w.SetInputConnection(wi.GetOutputPort()); w.Write()
    pts = {}
    if anchors:
        zb = vtk.vtkFloatArray(); rw.GetZbufferData(0, 0, W - 1, H - 1, zb); Z = np.frombuffer(memoryview(zb), dtype=np.float32).reshape(H, W)
        def disp(p):
            ren.SetWorldPoint(*p, 1.0); ren.WorldToDisplay(); return ren.GetDisplayPoint()
        for k, p in anchors.items():
            p = np.atleast_2d(p)
            if len(p) > 3000: p = p[np.random.default_rng(0).choice(len(p), 3000, replace=False)]
            D = np.array([disp(q) for q in p]); xi, yi = np.clip(D[:, 0].round().astype(int), 0, W - 1), np.clip(D[:, 1].round().astype(int), 0, H - 1)
            vis = D[:, 2] <= Z[yi, xi] + 4e-4
            if vis.sum() >= 1:
                V = D[vis][:, :2]; c2 = V.mean(0); x, y = V[np.argmin(((V - c2) ** 2).sum(1))]
            else:
                dd = p @ d; x, y, _ = disp(p[dd.argmax()])
            pts[k] = (x, H - y)
    x0 = y0 = 0
    if crop:
        im = Image.open(png).convert("RGB"); bgc = tuple(int(v * 255) for v in bg)
        diff = Image.eval(ImageOps.grayscale(Image.fromarray(np.abs(np.asarray(im).astype(int) - np.array(bgc)).clip(0, 255).astype(np.uint8))), lambda v: 255 if v > 6 else 0)
        bb = diff.getbbox() or (0, 0, W, H); x0, y0 = max(0, bb[0] - margin), max(0, bb[1] - margin)
        im = im.crop((x0, y0, min(W, bb[2] + margin), min(H, bb[3] + margin))); im.save(png)
    return {k: (x - x0, y - y0) for k, (x, y) in pts.items()}
def callouts(png, pts, title=None, sub=None, fs=None):
    im = Image.open(png); fs = fs or max(30, im.width // 55); f = font(fs)
    pad = int(max([ImageDraw.Draw(im).textlength(k, font=f) for k in pts] + [0]) + fs * 3)
    top = int(fs * 3.6) if title else 0
    tw_need = int(max(ImageDraw.Draw(im).textlength(title or "", font=font(int(fs * 1.4))), ImageDraw.Draw(im).textlength(sub or "", font=f)) + 2 * fs)
    pad = max(pad, (tw_need - im.width + 1) // 2, 0)
    can = Image.new("RGB", (im.width + 2 * pad, im.height + top), "white"); can.paste(im, (pad, top)); dr = ImageDraw.Draw(can)
    pts = {k: (x + pad, y + top) for k, (x, y) in pts.items()}; cx = pad + im.width / 2
    for side in (-1, 1):
        ks = sorted([k for k, (x, y) in pts.items() if (x < cx) == (side < 0)], key=lambda k: pts[k][1]); last = -1e9
        for k in ks:
            x, y = pts[k]; ty = max(y, last + fs * 2.2); last = ty
            tx = pad - fs * 0.6 if side < 0 else pad + im.width + fs * 0.6
            tw = dr.textlength(k, font=f)
            dr.line([(x, y), (tx, ty)], fill=(40, 40, 40), width=max(2, fs // 12)); r = fs // 5; dr.ellipse([x - r, y - r, x + r, y + r], fill=(40, 40, 40))
            dr.text((tx - tw - fs * 0.3 if side < 0 else tx + fs * 0.3, ty - fs * 0.6), k, fill=(20, 20, 20), font=f)
    bb = ImageOps.invert(can).getbbox(); m = fs
    if title:   # shift so the title starts at the left margin of the cropped picture
        left = max(0, bb[0] - m); can2 = Image.new("RGB", (max(bb[2] + m - left, tw_need), min(can.height, bb[3] + m)), "white"); can2.paste(can.crop((left, top, min(can.width, bb[2] + m), min(can.height, bb[3] + m))), (0, top))
        d2 = ImageDraw.Draw(can2); d2.text((fs, fs * 0.5), title, fill=(15, 15, 15), font=font(int(fs * 1.4)))
        if sub: d2.text((fs, fs * 2.3), sub, fill=(90, 90, 90), font=f)
        can2.save(png)
    else:
        can.crop((max(0, bb[0] - m), 0, min(can.width, bb[2] + m), min(can.height, bb[3] + m))).save(png)
def grid(pngs, out, cols=2, cell=(1200, 900), labels=None):
    rows = -(-len(pngs) // cols); f = font(30); sheet = Image.new("RGB", (cols * cell[0], rows * (cell[1] + 50)), "white"); dr = ImageDraw.Draw(sheet)
    for i, p in enumerate(pngs):
        im = Image.open(p).convert("RGB"); im.thumbnail(cell, Image.LANCZOS); cx, cy = (i % cols) * cell[0], (i // cols) * (cell[1] + 50)
        sheet.paste(im, (cx + (cell[0] - im.width) // 2, cy + (cell[1] - im.height) // 2))
        if labels: dr.text((cx + 16, cy + cell[1] + 6), labels[i], fill=(50, 50, 50), font=f)
    sheet.save(out)
