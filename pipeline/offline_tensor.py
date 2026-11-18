"""Training-path tensors for the converted (masked) episode frame (4090), compared with the live ones."""
import sys, os, pickle, numpy as np, torch, av, zarr
sys.path.insert(0, os.environ.get('FLASH_ALOHA_SRC', os.path.expanduser('~/flash-aloha/src')))
from contracts.standard_image_processing import ImageSpec, image_to_chw
from policy.model_util import make_policy
ck, name, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = pickle.load(open(f'{ck}/config.pkl', 'rb')); st = pickle.load(open(f'{ck}/dataset_stats.pkl', 'rb')); pc = cfg['policy_config']
spec = ImageSpec(width=pc['width'], height=pc['height'], color='RGB')
E = os.environ.get('YAM_EPISODE', os.path.expanduser('~/datasets/yam_sponge_flash60_all/episode_00057'))
def fr(cam):
    with av.open(f'{E}/{cam}.zarr/main.mp4') as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == idx: return f.to_ndarray(format='bgr24')
t_off = np.stack([image_to_chw(fr(c), spec) for c in cfg['camera_names']])
L = np.load('/tmp/live_tensor.npz')
t_live = L['t']
print('offline tensor', t_off.shape, t_off.dtype, 'per-channel mean top', t_off[0].reshape(3, -1).mean(1).round(4), 'wrist', t_off[1].reshape(3, -1).mean(1).round(4))
print('mean |live - offline| top %.4f wrist %.4f' % (np.abs(t_live[0] - t_off[0]).mean(), np.abs(t_live[1] - t_off[1]).mean()))
print('if live had R/B swapped: top %.4f wrist %.4f' % (np.abs(t_live[0][::-1] - t_off[0]).mean(), np.abs(t_live[1][::-1] - t_off[1]).mean()))
pol = make_policy(cfg['policy_class'], pc); pol.deserialize(torch.load(f'{ck}/{name}', map_location='cuda:0')); pol.cuda().eval()
q = L['q']
with torch.no_grad():
    out = pol(torch.from_numpy((q - st['qpos_mean']) / st['qpos_std']).float()[None].cuda(), torch.from_numpy(t_off)[None].cuda())[0].float().cpu().numpy() * st['action_std'] + st['action_mean']
print('chunk max |live - offline| (rad) rows 0/20/59:', [round(float(np.abs(L['chunk'][r, :6] - out[r, :6]).max()), 4) for r in (0, 20, 59)])
c = zarr.open_group(f'{E}/control.zarr', mode='r'); act = np.asarray(c['commanded_action'])
print('demo vs offline row20 %.4f, demo vs live row20 %.4f' % (np.abs(act[idx + 20, :6] - out[20, :6]).max(), np.abs(act[idx + 20, :6] - L['chunk'][20, :6]).max()))
