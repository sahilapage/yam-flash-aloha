import sys, numpy as np, mujoco
from scipy.optimize import least_squares
m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
print("joints:", names)
print("ranges:", {n: m.jnt_range[i].round(2).tolist() for i, n in enumerate(names)})
for side in ("left", "right"):
    jn = [n for n in names if n.startswith(side) and "tip" not in n and "finger" not in n and "grip" not in n][:6]
    qi = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn]
    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_gripper"); bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_base")
    cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_camera")
    def fk(q):
        d.qpos[:] = 0; d.qpos[qi] = q; mujoco.mj_forward(m, d); R = d.xmat[gb].reshape(3, 3).copy(); return d.xpos[gb].copy(), R
    mujoco.mj_forward(m, d); base = d.xpos[bb].copy()
    p0, R0 = fk(np.zeros(6)); print(f"\n[{side}] base at {base.round(3)}; zero pose gripper pos {p0.round(3)} finger_dir {R0[:,2].round(2)} handle_dir {R0[:,1].round(2)}")
    # task poses: fingers pointing straight down, and tilted 45 deg forward; target in front of the base
    fwd = p0 - base; fwd[2] = 0; fwd /= np.linalg.norm(fwd) + 1e-9
    for label, tilt in (("top-down", 0.0), ("45deg forward-down", np.pi / 4)):
        tgt = base + fwd * 0.33 + np.array([0, 0, 0.18]); fdir = -np.array([0, 0, 1.0]) * np.cos(tilt) + fwd * np.sin(tilt)
        best = None
        for q6 in (0.0,):
            for seed in range(30):
                rng = np.random.default_rng(seed); q0 = rng.uniform(-1.2, 1.2, 5)
                lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][0] for n in jn[:5]]); hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)][1] for n in jn[:5]])
                q0 = np.clip(q0, lo + 1e-3, hi - 1e-3)
                def res(q):
                    p, R = fk(np.r_[q, q6]); return np.r_[(p + R[:, 2] * 0.10 - tgt) * 5, R[:, 2] - fdir]
                s = least_squares(res, q0, bounds=(lo, hi))
                if best is None or s.cost < best[0]: best = (s.cost, np.r_[s.x, q6])
        p, R = fk(best[1]); cam = d.xpos[cb] - p
        side_dir = np.cross([0, 0, 1.0], fwd)
        def desc(v): return f"forward(away from base) {v@fwd:+.2f}, left {v@side_dir:+.2f}, up {v[2]:+.2f}"
        print(f"  {label}: cost {best[0]:.1e} q={best[1].round(2).tolist()}")
        print(f"     finger_dir : {desc(R[:,2])}")
        print(f"     HANDLE side (+y gripper = opposite camera): {desc(R[:,1])}")
        print(f"     slide axis : {desc(R[:,0])}")
        print(f"     camera offset from gripper origin: {desc(cam/np.linalg.norm(cam))}")
