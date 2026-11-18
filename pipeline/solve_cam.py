"""Fit the top camera (pose + focal) to hand-picked gripper pixels, using FK of each frame.

The picked pixel is the centre of the gripper housing, an unknown fixed point in the link_6
frame, so that offset is fitted too. Writes top_camera.json and an overlay sheet for checking.
"""
import json, sys, numpy as np, cv2
from scipy.optimize import least_squares
from common import YamFK

W, H = 848, 480
picks = {  # name: (u, v) picked on full-res frames
 'ep005_f0255': (660, 140), 'ep010_f0217': (775, 170), 'ep015_f0303': (615, 175), 'ep020_f0358': (515, 125),
 'ep025_f0129': (710, 180), 'ep030_f0368': (550, 355), 'ep035_f0180': (610, 280),
 'ep003_f0291': (667, 233), 'ep007_f0171': (735, 225), 'ep012_f0392': (550, 295), 'ep017_f0243': (585, 320),
 'ep022_f0218': (583, 323), 'ep027_f0181': (618, 215), 'ep032_f0162': (642, 210), 'ep037_f0343': (760, 202),
 'ep042_f0244': (658, 158), 'ep047_f0114': (693, 233), 'ep052_f0292': (592, 270), 'ep055_f0158': (677, 245),
 'ep008_f0408': (758, 190), 'ep018_f0086': (693, 215), 'ep028_f0333': (533, 293), 'ep048_f0059': (752, 243)}
recs = {r['name']: (d, r) for d in ('calib', 'calib2') for r in json.load(open(f'{d}/frames.json'))}
fk = YamFK()
T6 = []
uv = []
names = []
for n, p in picks.items():
    fk.set(recs[n][1]['q'][:6]); T6.append(fk.body('link_6')); uv.append(p); names.append(n)
T6, uv = np.array(T6), np.array(uv, float)

def K(f): return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.]])
def proj(x, T=T6):
    rvec, tvec, off, f = x[:3], x[3:6], x[6:9], x[9]
    pts = (T[:, :3, :3] @ off) + T[:, :3, 3]
    px, _ = cv2.projectPoints(pts, rvec, tvec, K(f), None)
    return px.reshape(-1, 2)
def res(x):
    r = (proj(x) - uv).ravel()
    return np.concatenate([r, [(x[9] - 610) / 20.0], x[6:9] / 0.05])  # weak priors

ok, rvec, tvec = cv2.solvePnP(T6[:, :3, 3], uv, K(610), None, flags=cv2.SOLVEPNP_SQPNP)
best = None
for off0 in ([0, 0, 0], [0, 0, 0.05], [0, -0.03, 0.06]):
    x0 = np.r_[rvec.ravel(), tvec.ravel(), off0, 610]
    s = least_squares(res, x0, loss='soft_l1', f_scale=15)
    if best is None or s.cost < best.cost: best = s
x = best.x
e = np.linalg.norm(proj(x) - uv, axis=1)
for n, ei in zip(names, e): print(f'{n} err {ei:5.1f}px')
print('median', np.median(e), 'max', e.max(), 'f', x[9], 'offset', x[6:9])
R, _ = cv2.Rodrigues(x[:3]); C = -R.T @ x[3:6]
print('camera centre in arm base frame (m):', C.round(3), 'optical axis:', (R.T @ [0, 0, 1]).round(3))
json.dump({'image_size': [W, H], 'fx': x[9], 'fy': x[9], 'cx': W / 2, 'cy': H / 2, 'rvec_base_to_cam': x[:3].tolist(),
           'tvec_base_to_cam': x[3:6].tolist(), 'pick_offset_in_link6': x[6:9].tolist(),
           'median_px_err': float(np.median(e)), 'note': 'fitted from hand-picked gripper pixels; refine with factory intrinsics'},
          open('top_camera.json', 'w'), indent=1)
