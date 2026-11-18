"""Draw the FK skeleton of the right arm through top_camera.json on picked frames (checking the fit)."""
import json, sys, cv2, numpy as np
from common import YamFK
cam = json.load(open(sys.argv[1])); out = sys.argv[2]; names = sys.argv[3].split(',')
recs = {r['name']: (d, r) for d in ('calib', 'calib2') for r in json.load(open(f'{d}/frames.json'))}
K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]])
rv, tv = np.array(cam['rvec_base_to_cam']), np.array(cam['tvec_base_to_cam'])
fk = YamFK(); tiles = []
for n in names:
    d, r = recs[n]; img = cv2.imread(f'{d}/{n}_raw.png'); fk.set(r['q'][:6])
    pts = [np.zeros(3)] + [fk.body(f'link_{i}')[:3, 3] for i in range(1, 7)] + [fk.site('grasp_site')[:3, 3], fk.body('camera_d405')[:3, 3]]
    px, _ = cv2.projectPoints(np.array(pts), rv, tv, K, None); px = px.reshape(-1, 2).astype(int)
    for a, b in zip(px[:7], px[1:8]): cv2.line(img, tuple(a), tuple(b), (0, 0, 255), 2)
    for p in px[:8]: cv2.circle(img, tuple(p), 4, (0, 255, 255), -1)
    cv2.circle(img, tuple(px[8]), 6, (255, 0, 255), 2)
    cv2.putText(img, n, (10, 470), 0, 0.6, (255, 255, 0), 2)
    tiles.append(cv2.resize(img, None, fx=0.5, fy=0.5))
while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
