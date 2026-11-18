"""Does the policy use the cameras? Plan with the true images vs with images from another moment
(another held-out episode, same qpos), vs all-black images. If swapping images barely changes the
plan, the policy is driving from qpos alone (copycat) and cannot react to the ball."""
import sys, os, pickle, numpy as np, torch, av, zarr
sys.path.insert(0, os.environ.get('FLASH_ALOHA_SRC', os.path.expanduser('~/flash-aloha/src')))
from contracts.standard_image_processing import ImageSpec, image_to_chw
from policy.model_util import make_policy
ck, name, eps = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = pickle.load(open(f'{ck}/config.pkl', 'rb')); st = pickle.load(open(f'{ck}/dataset_stats.pkl', 'rb')); pc = cfg['policy_config']
pol = make_policy(cfg['policy_class'], pc); pol.deserialize(torch.load(f'{ck}/{name}', map_location='cuda:0')); pol.cuda().eval()
spec = ImageSpec(width=pc['width'], height=pc['height'], color='RGB')
data = []
for ep in eps:
    c = zarr.open_group(f'{ep}/control.zarr', mode='r')
    vids = []
    for cam in cfg['camera_names']:
        with av.open(f'{ep}/{cam}.zarr/main.mp4') as f:
            vids.append(np.stack([image_to_chw(x.to_ndarray(format='bgr24'), spec) for x in f.decode(video=0)]))
    data.append((np.asarray(c['qpos']), np.asarray(c['commanded_action']), np.stack(vids, 1)))
def plan(q, im):
    with torch.no_grad():
        return pol(torch.from_numpy((q - st['qpos_mean']) / st['qpos_std']).float()[None].cuda(), torch.from_numpy(im)[None].cuda())[0].float().cpu().numpy() * st['action_std'] + st['action_mean']
rng = np.random.default_rng(0); H = 30
e_true, d_swap, d_black, e_swap = [], [], [], []
for k, (q, act, ims) in enumerate(data):
    other = data[(k + 1) % len(data)][2]
    for t in range(0, len(q) - H, 10):
        p0 = plan(q[t], ims[t]); p1 = plan(q[t], other[rng.integers(len(other))]); p2 = plan(q[t], np.zeros_like(ims[t]))
        e_true.append(np.abs(p0[H, :6] - act[t + H, :6]).mean()); e_swap.append(np.abs(p1[H, :6] - act[t + H, :6]).mean())
        d_swap.append(np.abs(p1[H, :6] - p0[H, :6]).mean()); d_black.append(np.abs(p2[H, :6] - p0[H, :6]).mean())
print(f'{name:32s} 0.5 s ahead: error true imgs {np.mean(e_true):.3f} | wrong imgs {np.mean(e_swap):.3f} rad '
      f'| plan change wrong imgs {np.mean(d_swap):.3f}, black imgs {np.mean(d_black):.3f} rad')
