"""Per-episode audit of the raw YAM sponge-ball dataset: ball track (top cam, yellow blob), ball size in
the wrist cam, arm motion onset, gripper close/open events (regrasps), place, speeds. Writes audit.npz
+ audit.csv + overlay images."""
import glob, json, os, sys, numpy as np, pandas as pd, av, cv2
R = os.environ.get('YAM_DATASET', os.path.expanduser('~/datasets/yam-right-sponge')); OUT = sys.argv[1]
meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{R}/meta/episodes/chunk-*/*.parquet'))]).set_index('episode_index')
data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{R}/data/chunk-*/*.parquet'))])
LO, HI = np.array([20, 110, 110]), np.array([38, 255, 255])

def frames(ep, cam):
    m = meta.loc[ep]; k = f'videos/observation.images.{cam}'
    path = f"{R}/videos/observation.images.{cam}/chunk-{int(m[k+'/chunk_index']):03d}/file-{int(m[k+'/file_index']):03d}.mp4"
    t0, t1 = float(m[k + '/from_timestamp']), float(m[k + '/to_timestamp'])
    with av.open(path) as c:
        s = c.streams.video[0]; c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t < t0 - 1e-3: continue
            if t >= t1 - 1e-3: break
            yield f.to_ndarray(format='bgr24')

def ball(img, min_area):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); m = cv2.inRange(hsv, LO, HI)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1: return (np.nan, np.nan, 0.0)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); a = st[i, cv2.CC_STAT_AREA]
    return (cen[i][0], cen[i][1], float(a)) if a >= min_area else (np.nan, np.nan, 0.0)

rows, tracks = [], {}
bg = None
for ep in meta.index:
    d = data[data.episode_index == ep].sort_values('frame_index'); q = np.stack(d['observation.state'].values).astype(float)
    top = np.array([ball(f, 150) for f in frames(ep, 'top')]); wr = np.array([ball(f, 400) for f in frames(ep, 'right_wrist')])
    if bg is None:
        bg = next(frames(ep, 'top'))
    n = min(len(q), len(top), len(wr)); q, top, wr = q[:n], top[:n], wr[:n]
    hz = 60.0; v = np.r_[0, np.linalg.norm(np.diff(q[:, :6], axis=0), axis=1) * hz]
    vs = pd.Series(v).rolling(9, center=True, min_periods=1).mean().values
    onset = int(np.argmax(vs > 0.25)) if np.any(vs > 0.25) else -1
    g = q[:, 6]; g0 = np.median(g[:30])
    closed = g > g0 + 0.06
    edges = np.flatnonzero(np.diff(closed.astype(int)) == 1) + 1; opens = np.flatnonzero(np.diff(closed.astype(int)) == -1) + 1
    # merge close events separated by < 0.15 s
    ev = [e for i, e in enumerate(edges) if i == 0 or e - edges[i - 1] > 9]
    first_close = int(ev[0]) if ev else -1
    vis = np.isfinite(top[:, 0]); first_vis = int(np.argmax(vis)) if vis.any() else -1
    # ball speed in top cam (px/s) over visible frames before the first close
    bs = np.nan
    if first_vis >= 0 and first_close > first_vis + 6:
        seg = top[first_vis:first_close, :2]; ok = np.isfinite(seg[:, 0])
        if ok.sum() > 6:
            p = seg[ok]; bs = float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)) * hz)
    wr_big = wr[:, 2] > 8000  # ball large in wrist view: between / at the fingers
    wr_first = int(np.argmax(wr_big)) if wr_big.any() else -1
    catch_xy = top[first_close, :2] if first_close >= 0 else (np.nan, np.nan)
    rows.append(dict(ep=ep, sec=n / hz, onset_s=onset / hz, ball_first_top_s=first_vis / hz if first_vis >= 0 else np.nan,
                     wrist_ball_s=wr_first / hz if wr_first >= 0 else np.nan, first_close_s=first_close / hz if first_close >= 0 else np.nan,
                     n_close_events=len(ev), close_events_s=';'.join(f'{e / hz:.2f}' for e in ev), grip_start=g0, grip_max=float(g.max()),
                     ball_speed_px_s=bs, catch_x=catch_xy[0], catch_y=catch_xy[1], peak_joint_speed=float(vs.max()),
                     ball_visible_frac=float(vis.mean())))
    tracks[f'ep{ep}'] = top; tracks[f'q{ep}'] = q; tracks[f'wr{ep}'] = wr
    print(ep, end=' ', flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{OUT}/audit.csv', index=False); np.savez_compressed(f'{OUT}/audit.npz', **tracks)
# overlay: every ball track on the top view, catch point marked
img = bg.copy() // 2
cmap = cv2.applyColorMap(np.linspace(0, 255, len(df)).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
for k, r in df.iterrows():
    t = tracks[f'ep{int(r.ep)}']; c = tuple(int(x) for x in cmap[k])
    pts = [(int(x), int(y)) for x, y, a in t if np.isfinite(x)]
    for a, b in zip(pts, pts[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 60: cv2.line(img, a, b, c, 1)
    if np.isfinite(r.catch_x): cv2.circle(img, (int(r.catch_x), int(r.catch_y)), 6, c, 2); cv2.putText(img, str(int(r.ep)), (int(r.catch_x) + 6, int(r.catch_y)), 0, 0.35, c, 1)
cv2.imwrite(f'{OUT}/ball_tracks.jpg', img)
print('\ndone')
