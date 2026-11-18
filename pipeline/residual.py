"""Image-space heatmaps: where person pixels are, and where they remain after a candidate mask."""
import json, sys, numpy as np, cv2
from common import YamFK, episode_data
from yam_mask import YamTopMask
root, pick = sys.argv[1], int(sys.argv[2])
fit = json.load(open('poly_fits.json'))[pick]
cam = json.load(open('top_camera.json'))
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
json.dump({'camera': cam, 'operations': [{'project': {'frame': 'link_6_yaw', 'z': 0.0, 'corners': fit['poly']}}] + extra}, open('_tmp_mask.json', 'w'))
s = np.load('seg_top.npz'); masks, eps, idxs = s['masks'], s['ep'], s['idx']
fk = YamFK(); m = YamTopMask('_tmp_mask.json', fk=fk); cache = {}
h, w = masks.shape[1:]
allp = np.zeros((h, w)); left = np.zeros((h, w)); area = 0
sel = np.arange(0, len(masks), 3)
for k in sel:
    e, i = eps[k], idxs[k]
    if e not in cache: cache[e] = np.stack(episode_data(root, e)['observation.state'].to_numpy())
    blk = m(np.full((h, w, 3), 255, np.uint8), cache[e][i][:6])[..., 0] == 0
    allp += masks[k]; left += masks[k] & ~blk; area += blk.mean()
print('covered share of all person px', 1 - left.sum() / allp.sum(), 'mean area', area / len(sel))
def hm(a): 
    a = (255 * np.clip(a / len(sel) / 0.3, 0, 1)).astype(np.uint8)
    return cv2.resize(cv2.applyColorMap(a, cv2.COLORMAP_INFERNO), (424, 240), interpolation=cv2.INTER_NEAREST)
out = np.hstack([hm(allp), hm(left)])
for x in range(0, 424, 42): cv2.line(out, (x, 0), (x, 240), (90, 90, 90), 1); cv2.line(out, (x + 424, 0), (x + 424, 240), (90, 90, 90), 1)
for y in range(0, 240, 24): cv2.line(out, (0, y), (848, y), (90, 90, 90), 1)
cv2.imwrite('residual.png', out)
