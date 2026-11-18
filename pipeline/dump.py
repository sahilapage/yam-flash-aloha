"""Dump full-res top frames with a pixel grid plus joint states, for picking calibration points."""
import sys, json, cv2, numpy as np
from common import episode_meta, episode_data, frame_at, YamFK
root, out = sys.argv[1], sys.argv[2]
picks = [tuple(map(float, p.split(':'))) for p in sys.argv[3].split(',')]
fk = YamFK()
meta = episode_meta(root)
rec = []
import os; os.makedirs(out, exist_ok=True)
for ep, frac in picks:
    ep = int(ep); n = int(meta.loc[ep, 'length']); idx = int(frac * (n - 1))
    d = episode_data(root, ep)
    q = np.array(d.loc[idx, 'observation.state'], dtype=float)
    img = frame_at(root, ep, 'top', idx)
    g = img.copy()
    for x in range(0, g.shape[1], 50):
        cv2.line(g, (x, 0), (x, g.shape[0]), (0, 0, 255) if x % 100 == 0 else (0, 160, 255), 1)
        if x % 100 == 0: cv2.putText(g, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for y in range(0, g.shape[0], 50):
        cv2.line(g, (0, y), (g.shape[1], y), (0, 0, 255) if y % 100 == 0 else (0, 160, 255), 1)
        if y % 100 == 0: cv2.putText(g, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    name = f'ep{ep:03d}_f{idx:04d}'
    cv2.imwrite(f'{out}/{name}.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(f'{out}/{name}_raw.png', img)
    fk.set(q[:6])
    rec.append(dict(name=name, ep=ep, idx=idx, q=q.tolist(), link6=fk.body('link_6')[:3, 3].tolist()))
json.dump(rec, open(f'{out}/frames.json', 'w'), indent=1)
for r in rec: print(r['name'], np.round(r['link6'], 3))
