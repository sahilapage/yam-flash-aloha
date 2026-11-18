"""Stress test of the FK mask on every 4th top frame:
 - ball (yellow) pixels hidden by the mask
 - operator pixels (DeepLab person, right of the roller zone, below the top strip) left visible
Writes worst-frame sheets for eyeballing."""
import sys, numpy as np, cv2
from common import episode_meta, episode_data, episode_frames
from yam_mask import YamTopMask
root = sys.argv[1]
s = np.load('seg_top.npz'); segm, seps, sidx = s['masks'], s['ep'], s['idx']
key = {(e, i): k for k, (e, i) in enumerate(zip(seps, sidx))}
m = YamTopMask(); rows = []
for ep in episode_meta(root).index:
    if ep == 2: continue
    q = np.stack(episode_data(root, ep)['observation.state'].to_numpy())
    for i, img in enumerate(episode_frames(root, ep, 'top')):
        if (ep, i) not in key: continue
        out = m(img, q[i, :6]); hidden = (out.sum(2) == 0) & (img.sum(2) > 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ball = (hsv[..., 0] >= 20) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 120)
        ball[:int(0.2 * 480)] = False
        person = cv2.resize(segm[key[(ep, i)]].astype(np.uint8), (848, 480), interpolation=cv2.INTER_NEAREST) > 0
        zone = np.zeros_like(person); zone[int(0.2 * 480):, int(0.3 * 848):] = True
        op = person & zone
        rows.append((ep, i, ball.sum(), (ball & hidden).sum(), op.sum(), (op & ~hidden).sum()))
r = np.array(rows, float)
np.save('check_rows.npy', r)
vis = r[:, 2] > 150
frac = r[vis, 3] / r[vis, 2]
print('frames', len(r), 'with ball visible', vis.sum())
print('ball pixels hidden: mean %.3f; frames with >25%% ball hidden %.3f, >50%% %.3f' % (frac.mean(), (frac > .25).mean(), (frac > .5).mean()))
opv = r[:, 4] > 0
print('operator px left visible overall %.3f; frames with >2000 visible operator px %.3f' % (r[opv, 5].sum() / r[opv, 4].sum(), (r[:, 5] > 2000).mean()))
def sheet(sel, name):
    tiles = []
    for k in sel:
        ep, i = int(r[k, 0]), int(r[k, 1])
        q = np.stack(episode_data(root, ep)['observation.state'].to_numpy())
        for j, img in enumerate(episode_frames(root, ep, 'top')):
            if j == i: break
        out = m(img, q[i, :6]); person = cv2.resize(segm[key[(ep, i)]].astype(np.uint8), (848, 480), interpolation=cv2.INTER_NEAREST) > 0
        ov = out.copy(); ov[person & (out.sum(2) > 0)] = (0, 0, 255)
        t = np.hstack([cv2.resize(img, (424, 240)), cv2.resize(ov, (424, 240))])
        cv2.putText(t, f'ep{ep} f{i} ball_hidden {r[k,3]:.0f}/{r[k,2]:.0f} op_visible {r[k,5]:.0f}', (6, 16), 0, 0.45, (0, 255, 255), 1)
        tiles.append(t)
    cv2.imwrite(name, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
bh = np.where(vis)[0][np.argsort(-frac)[:6]]
sheet(bh, 'worst_ball.jpg')
sheet(np.argsort(-r[:, 5])[:6], 'worst_operator.jpg')
