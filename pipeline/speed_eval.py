"""Is the policy's plan as fast as the demos? Displacement over the next 1/3 s, plan vs demo (held-out eps)."""
import sys, os, pickle, numpy as np, torch, zarr, av
sys.path.insert(0, os.environ.get('FLASH_ALOHA_SRC', os.path.expanduser('~/flash-aloha/src')))
from policy.model_util import make_policy
from contracts.standard_image_processing import ImageSpec, image_to_chw
D, name, eps = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = pickle.load(open(f"{D}/config.pkl", "rb")); st = pickle.load(open(f"{D}/dataset_stats.pkl", "rb")); pc = cfg["policy_config"]
pol = make_policy(cfg["policy_class"], pc); pol.deserialize(torch.load(f"{D}/{name}", map_location="cuda:0")); pol.cuda().eval()
spec = ImageSpec(width=pc["width"], height=pc["height"], color="RGB")
H = 20
rp, rd, gp, gd = [], [], [], []
for ep in eps:
    c = zarr.open_group(f"{ep}/control.zarr", mode="r"); q = np.asarray(c["qpos"]); act = np.asarray(c["commanded_action"])
    vids = []
    for cam in cfg["camera_names"]:
        with av.open(f"{ep}/{cam}.zarr/main.mp4") as f:
            vids.append([image_to_chw(x.to_ndarray(format="bgr24"), spec) for x in f.decode(video=0)])
    for t in range(0, len(q) - H, 6):
        with torch.no_grad():
            qn = torch.from_numpy((q[t] - st["qpos_mean"]) / st["qpos_std"]).float()[None].cuda()
            im = torch.from_numpy(np.stack([v[t] for v in vids]))[None].cuda()
            out = pol(qn, im)[0].cpu().numpy() * st["action_std"] + st["action_mean"]
        rd.append(np.linalg.norm(act[t + H, :6] - act[t, :6])); rp.append(np.linalg.norm(out[H, :6] - out[0, :6]))
        gd.append(act[t + H, 6]); gp.append(out[H, 6])
rd, rp = np.array(rd), np.array(rp)
def ratio(m): return rp[m].sum() / rd[m].sum()
print(f"{name:30s} plan/demo motion in next 1/3 s: all {ratio(rd >= 0):.2f}  faster half {ratio(rd > np.percentile(rd, 50)):.2f}  fastest 10pc {ratio(rd > np.percentile(rd, 90)):.2f} | gripper demo range {min(gd):.2f}-{max(gd):.2f} plan {min(gp):.2f}-{max(gp):.2f}")
