"""Where are the operator's hands relative to the gripper? Fits the FK mask polygon from data.

Person pixels (seg_top.npz) are back-projected onto the horizontal plane through link_6 and
expressed relative to link_6, in base axes and in gripper-yaw axes. Prints occupancy stats
and saves occupancy maps (occ_*.png) for both choices.
"""
import json, sys, numpy as np, cv2
from common import YamFK, episode_data

root = sys.argv[1]
cam = json.load(open('top_camera.json'))
K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]])
R, _ = cv2.Rodrigues(np.array(cam['rvec_base_to_cam'])); t = np.array(cam['tvec_base_to_cam'])
C = -R.T @ t
s = np.load('seg_top.npz'); masks, eps, idxs = s['masks'], s['ep'], s['idx']
h, w = masks.shape[1:]
uu, vv = np.meshgrid((np.arange(w) + 0.5) * 848 / w, (np.arange(h) + 0.5) * 480 / h)
rays = (R.T @ (np.linalg.inv(K) @ np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)]))).T  # (P,3) base frame
fk = YamFK(); cache = {}
res, ext = 0.01, 0.8  # 1 cm bins, +-0.8 m
nb = int(2 * ext / res)
occ = {'base': np.zeros((nb, nb)), 'yaw': np.zeros((nb, nb))}
frames_with = 0; total_px = 0; near_px = 0
for m, ep, i in zip(masks, eps, idxs):
    if ep not in cache: cache[ep] = np.stack(episode_data(root, ep)['observation.state'].to_numpy())
    q = cache[ep][i]; fk.set(q[:6]); T = fk.body('link_6'); p0 = T[:3, 3]
    sel = m.ravel()
    if not sel.any(): continue
    frames_with += 1
    d = rays[sel]; sc = (p0[2] - C[2]) / d[:, 2]; P = C + sc[:, None] * d
    rel = P[:, :2] - p0[:2]
    x = T[:3, 0]; yaw = np.arctan2(x[1], x[0])  # link_6 x axis heading
    c, s_ = np.cos(-yaw), np.sin(-yaw); rel_y = rel @ np.array([[c, s_], [-s_, c]])
    total_px += len(rel); near_px += (np.linalg.norm(rel, axis=1) < 0.6).sum()
    for k, r in (('base', rel), ('yaw', rel_y)):
        ij = ((r + ext) / res).astype(int); ok = ((ij >= 0) & (ij < nb)).all(1)
        np.add.at(occ[k], (ij[ok, 0], ij[ok, 1]), 1)
print('frames', len(masks), 'with person px', frames_with, 'person px within 0.6 m of wrist (plane)', near_px / max(total_px, 1))
for k, o in occ.items():
    o = o / max(frames_with, 1)
    np.save(f'occ_{k}.npy', o)
    img = (255 * np.clip(o / np.percentile(o[o > 0], 99), 0, 1)).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    cv2.circle(img, (nb // 2, nb // 2), 3, (0, 255, 0), -1)  # wrist; rows = x (down = +x), cols = y
    for r_ in (10, 20, 30, 40):
        cv2.circle(img, (nb // 2, nb // 2), int(r_ / 100 / res), (80, 80, 80), 1)
    cv2.imwrite(f'occ_{k}.png', cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
