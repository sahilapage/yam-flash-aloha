"""2-frame model checks on recorded episodes: 0.5 s-ahead error with true images, and how much the
plan changes when top_prev is replaced by the current top frame (no motion info) or by black."""
import sys, os, pickle, numpy as np, torch, av, zarr
sys.path.insert(0, os.environ.get('FLASH_ALOHA_SRC', os.path.expanduser('~/flash-aloha/src')))
from contracts.standard_image_processing import ImageSpec, image_to_chw
from policy.model_util import make_policy
ck, name, eps = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = pickle.load(open(f'{ck}/config.pkl', 'rb')); st = pickle.load(open(f'{ck}/dataset_stats.pkl', 'rb')); pc = cfg['policy_config']
cams = cfg['camera_names']; assert cams == ['top', 'top_prev', 'right_wrist'], cams
pol = make_policy(cfg['policy_class'], pc); pol.deserialize(torch.load(f'{ck}/{name}', map_location='cuda:0')); pol.cuda().eval()
spec = ImageSpec(width=pc['width'], height=pc['height'], color='RGB')
def plan(q, im):
    with torch.no_grad():
        return pol(torch.from_numpy((q - st['qpos_mean']) / st['qpos_std']).float()[None].cuda(), torch.from_numpy(im)[None].cuda())[0].float().cpu().numpy() * st['action_std'] + st['action_mean']
H = 30; err, d_same, d_black = [], [], []
for ep in eps:
    c = zarr.open_group(f'{ep}/control.zarr', mode='r'); q, act = np.asarray(c['qpos']), np.asarray(c['commanded_action'])
    vids = []
    for cam in cams:
        with av.open(f'{ep}/{cam}.zarr/main.mp4') as f:
            vids.append(np.stack([image_to_chw(x.to_ndarray(format='bgr24'), spec) for x in f.decode(video=0)]))
    ims = np.stack(vids, 1)
    for t in range(10, len(q) - H, 10):
        p0 = plan(q[t], ims[t])
        same = ims[t].copy(); same[1] = same[0]
        blk = ims[t].copy(); blk[1] = 0
        err.append(np.abs(p0[H, :6] - act[t + H, :6]).mean())
        d_same.append(np.abs(plan(q[t], same)[H, :6] - p0[H, :6]).mean()); d_black.append(np.abs(plan(q[t], blk)[H, :6] - p0[H, :6]).mean())
print(f'{name:30s} 0.5 s-ahead error {np.mean(err):.3f} rad | plan change if prev=current {np.mean(d_same):.3f} (max {np.max(d_same):.3f}), prev=black {np.mean(d_black):.3f}')
