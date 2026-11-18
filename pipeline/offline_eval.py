"""Offline check of an ACT checkpoint on held-out converted episodes (masked videos), no robot.

Per episode: open-loop chunk error vs the demo at several horizons, chunk-boundary jumps when a
new chunk is taken every K steps (the discontinuity Flash ALOHA's sigmoid blend smooths), and a
plot of demo actions vs the stitched plan. Uses upstream's exact image/qpos preprocessing.

    python offline_eval.py CKPT_DIR CKPT_NAME EPISODE_DIR [EPISODE_DIR ...] --out DIR
"""
import argparse, os, pickle, sys, time
import av, numpy as np, torch, zarr
sys.path.insert(0, os.environ.get('FLASH_ALOHA_SRC', os.path.expanduser('~/flash-aloha/src')))
from policy.model_util import make_policy
from contracts.standard_image_processing import ImageSpec, image_to_chw

p = argparse.ArgumentParser()
p.add_argument('ckpt_dir'); p.add_argument('ckpt'); p.add_argument('episodes', nargs='+')
p.add_argument('--out', required=True); p.add_argument('--stride', type=int, default=3)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
cfg = pickle.load(open(f'{a.ckpt_dir}/config.pkl', 'rb')); stats = pickle.load(open(f'{a.ckpt_dir}/dataset_stats.pkl', 'rb'))
pc = cfg['policy_config']; cams = cfg['camera_names']; T = pc['num_queries']
policy = make_policy(cfg['policy_class'], pc); print(policy.deserialize(torch.load(f'{a.ckpt_dir}/{a.ckpt}', map_location='cuda:0')))
policy.cuda().eval()
spec = ImageSpec(width=pc['width'], height=pc['height'], color='RGB')
qm, qs = stats['qpos_mean'], stats['qpos_std']; am, as_ = stats['action_mean'], stats['action_std']

def frames(path):
    with av.open(path) as c:
        for f in c.decode(video=0): yield image_to_chw(f.to_ndarray(format='bgr24'), spec)

@torch.no_grad()
def predict(q, imgs):
    qn = torch.from_numpy((q - qm) / qs).float()[None].cuda()
    im = torch.from_numpy(np.stack(imgs))[None].cuda()
    return policy(qn, im)[0].cpu().numpy() * as_ + am

names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'grip']
summary = []
for ep in a.episodes:
    ctl = zarr.open_group(f'{ep}/control.zarr', mode='r')
    qpos, act = np.asarray(ctl['qpos']), np.asarray(ctl['commanded_action']); n = len(qpos)
    vids = [list(frames(f'{ep}/{c}.zarr/main.mp4')) for c in cams]
    assert all(len(v) == n for v in vids), [len(v) for v in vids]
    chunks, lat = {}, []
    for t in range(n):
        if t % a.stride and t % 15: continue
        s = time.perf_counter(); chunks[t] = predict(qpos[t], [v[t] for v in vids]); torch.cuda.synchronize(); lat.append(time.perf_counter() - s)
    # open-loop error vs demo at horizons
    hz = [0, 14, 29, T - 1]; err = {h: [] for h in hz}
    for t, c in chunks.items():
        for h in hz:
            if t + h < n: err[h].append(np.abs(c[h] - act[t + h]))
    # chunk handoff: new chunk every K steps; jump = new[0] vs old[K] (arm joints, rad)
    demo_step = np.abs(np.diff(act[:, :6], axis=0)).max(1)
    res = dict(ep=os.path.basename(ep), n=n, latency_ms=1000 * np.median(lat[3:]))
    for h in hz:
        e = np.array(err[h]); res[f'err_h{h}_arm_rad'] = e[:, :6].mean(); res[f'err_h{h}_grip'] = e[:, 6].mean()
    for K in (15, 30, 45):
        jumps = [np.abs(chunks[t][0, :6] - chunks[t - K][K, :6]).max() for t in range(K, n, K) if t in chunks and t - K in chunks]
        res[f'jump_K{K}_mean'] = float(np.mean(jumps)); res[f'jump_K{K}_max'] = float(np.max(jumps))
    res['demo_step_p99'] = float(np.percentile(demo_step, 99))
    summary.append(res)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    # stitched plan with K=30, no blend, for plotting
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plan = np.full_like(act, np.nan)
    for t in range(0, n, 30):
        if t in chunks: plan[t:t + 30] = chunks[t][:min(30, n - t)]
    fig, ax = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
    for j in range(7):
        ax[j].plot(np.arange(n) / 60, act[:, j], 'k', lw=1.2, label='demo action')
        ax[j].plot(np.arange(n) / 60, plan[:, j], 'C1', lw=1, label='policy (new chunk every 0.5 s)')
        ax[j].set_ylabel(names[j])
    ax[0].legend(loc='upper right', fontsize=8); ax[-1].set_xlabel('s'); fig.suptitle(f'{res["ep"]} {a.ckpt}')
    fig.tight_layout(); fig.savefig(f'{a.out}/{res["ep"]}.png', dpi=80); plt.close(fig)
pickle.dump(summary, open(f'{a.out}/summary.pkl', 'wb'))
