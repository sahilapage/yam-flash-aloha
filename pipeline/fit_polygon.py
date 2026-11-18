"""Fit the gripper-following mask polygon from occ_yaw.npy and score it on every segmented frame.

Score per threshold: share of operator pixels covered (person pixels whose back-projection is
within 0.6 m of the wrist), share of all person pixels covered, and mean image area blacked out.
"""
import json, sys, numpy as np, cv2
from scipy.spatial import ConvexHull
from common import YamFK, episode_data
from yam_mask import YamTopMask

root = sys.argv[1]
occ = np.load('occ_yaw.npy'); res, ext = 0.01, 0.8; nb = occ.shape[0]
cam = json.load(open('top_camera.json'))
s = np.load('seg_top.npz'); masks, eps, idxs = s['masks'], s['ep'], s['idx']
sel = np.arange(0, len(masks), 3)  # score on a third of the frames, fast enough
K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]])
R, _ = cv2.Rodrigues(np.array(cam['rvec_base_to_cam'])); C = -R.T @ np.array(cam['tvec_base_to_cam'])
h, w = masks.shape[1:]
uu, vv = np.meshgrid((np.arange(w) + 0.5) * 848 / w, (np.arange(h) + 0.5) * 480 / h)
rays = (R.T @ (np.linalg.inv(K) @ np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)]))).T
fk = YamFK(); cache = {}
qs = []
for e, i in zip(eps[sel], idxs[sel]):
    if e not in cache: cache[e] = np.stack(episode_data(root, e)['observation.state'].to_numpy())
    qs.append(cache[e][i][:6])
near = []
for k, q in zip(sel, qs):
    fk.set(q); p0 = fk.body('link_6')[:3, 3]
    sc = (p0[2] - C[2]) / rays[:, 2]; P = C + sc[:, None] * rays
    near.append((np.linalg.norm(P[:, :2] - p0[:2], axis=1) < 0.6).reshape(h, w) & masks[k])
near = np.array(near)
cyx = np.argwhere(np.ones_like(occ, bool))
results = []
for frac in (0.3, 0.15, 0.08, 0.04, 0.02):
    thr = frac * np.percentile(occ[occ > 0], 99.5)
    pts = (np.argwhere(occ >= thr) + 0.5) * res - ext
    pts = pts[np.linalg.norm(pts, axis=1) < 0.5]
    hull = pts[ConvexHull(pts).vertices]
    for margin in (0.02, 0.04):
        cen = hull.mean(0); d = hull - cen; poly = hull + margin * d / np.linalg.norm(d, axis=1, keepdims=True)
        cfg = {'camera': cam, 'operations': [{'project': {'frame': 'link_6_yaw', 'z': 0.0, 'corners': poly.round(3).tolist()}}]}
        json.dump(cfg, open('_tmp_mask.json', 'w'))
        m = YamTopMask('_tmp_mask.json', fk=fk)
        cov_n = cov_a = area = 0; tn = ta = 0
        for j, (k, q) in enumerate(zip(sel, qs)):
            img = np.full((h, w), 255, np.uint8)
            blk = m(img[..., None].repeat(3, 2), q)[..., 0] == 0
            cov_n += (blk & near[j]).sum(); tn += near[j].sum()
            cov_a += (blk & masks[k]).sum(); ta += masks[k].sum(); area += blk.mean()
        r = dict(frac=frac, margin=margin, n=len(poly), operator_cov=cov_n / tn, all_cov=cov_a / ta, area=area / len(sel), poly=poly.round(3).tolist())
        results.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'poly'}, flush=True)
json.dump(results, open('poly_fits.json', 'w'), indent=1)
