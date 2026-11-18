"""Hand mask for the YAM top camera, a pure function of (image, joint angles).

Flash ALOHA's rule: the mask must be computed from qpos alone and applied identically when
converting training data and at rollout, so the policy never sees a difference between a
demo frame (operator's hand blacked out) and a rollout frame (no hand, same black shape).

The config (yam_mask.json) holds the top camera calibration and the operations:
  {"project": {"frame": "link_6" | "link_6_yaw", "z": metres, "corners": [[x, y], ...]}}
      polygon on a horizontal plane at the link's height + z, corners in metres relative to the
      link origin, in base axes ("link_6") or base axes turned by the gripper heading ("link_6_yaw").
  {"region": {"x": [a, b], "y": [c, d]}}   image fractions.
contract() hashes the config; a checkpoint stores it and rollout refuses a different one.
"""
import hashlib, json
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


class YamTopMask:
    def __init__(self, config=HERE / 'yam_mask.json', fk=None):
        self.config = json.loads(Path(config).read_text())
        cam = self.config['camera']
        self.size0 = cam['image_size']
        self.K0 = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1.0]])
        self.rvec, self.tvec = np.array(cam['rvec_base_to_cam']), np.array(cam['tvec_base_to_cam'])
        if fk is None:
            from common import YamFK
            fk = YamFK()
        self.fk = fk

    def contract(self):
        blob = json.dumps({'camera': self.config['camera'], 'operations': self.config['operations']}, sort_keys=True)
        return {'name': 'yam_top_fk_mask', 'sha256': hashlib.sha256(blob.encode()).hexdigest()}

    def polygons(self, q6, size):
        """Pixel polygons for an image of size (width, height)."""
        K = np.diag([size[0] / self.size0[0], size[1] / self.size0[1], 1.0]) @ self.K0
        self.fk.set(np.asarray(q6, dtype=np.float64)[:6])
        T = self.fk.body('link_6')
        out = []
        for op in self.config['operations']:
            if 'project' not in op:
                continue
            p = op['project']
            c = np.asarray(p['corners'], dtype=np.float64)
            if p['frame'] == 'link_6_yaw':
                yaw = np.arctan2(T[1, 0], T[0, 0])
                rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                c = c @ rot.T
            elif p['frame'] != 'link_6':
                raise ValueError(p['frame'])
            pts = np.c_[c + T[:2, 3], np.full(len(c), T[2, 3] + p['z'])]
            px, _ = cv2.projectPoints(pts, self.rvec, self.tvec, K, None)
            out.append(np.clip(px.reshape(-1, 2), -5000, 5000).astype(np.int32))
        return out

    def __call__(self, img, q6):
        """Masked copy of a top-camera image (any resolution, HWC uint8)."""
        img = np.array(img, copy=True, order='C')
        h, w = img.shape[:2]
        for poly in self.polygons(q6, (w, h)):
            cv2.fillPoly(img, [poly.reshape(-1, 1, 2)], (0, 0, 0))
        for op in self.config['operations']:
            if 'region' in op:
                r = op['region']
                x0, x1 = (int(round(v * w)) for v in r['x'])
                y0, y1 = (int(round(v * h)) for v in r['y'])
                img[y0:y1, x0:x1] = 0
        return img
