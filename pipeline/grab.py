"""Contact sheets: for each episode, frames at several fractions, top row = top cam, bottom = wrist."""
import glob, sys, os
import av, cv2, numpy as np, pandas as pd

root, out = sys.argv[1], sys.argv[2]
eps = [int(e) for e in sys.argv[3].split(',')]
fracs = [0.05, 0.25, 0.45, 0.65, 0.85]
os.makedirs(out, exist_ok=True)
meta = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{root}/meta/episodes/chunk-*/*.parquet'))])

def grab(path, times):
    res = {}
    with av.open(path) as c:
        s = c.streams.video[0]
        for t in times:
            c.seek(int(max(t - 0.3, 0) / s.time_base), stream=s, backward=True)
            for f in c.decode(s):
                if float(f.pts * s.time_base) >= t - 1e-3:
                    res[t] = f.to_ndarray(format='bgr24'); break
    return [res[t] for t in times]

for ep in eps:
    row = meta[meta.episode_index == ep].iloc[0]
    rows = []
    for cam in ['top', 'right_wrist']:
        k = f'videos/observation.images.{cam}'
        p = f"{root}/videos/observation.images.{cam}/chunk-{int(row[k+'/chunk_index']):03d}/file-{int(row[k+'/file_index']):03d}.mp4"
        t0, t1 = float(row[k+'/from_timestamp']), float(row[k+'/to_timestamp'])
        imgs = grab(p, [t0 + f * (t1 - t0) for f in fracs])
        imgs = [cv2.resize(i, (424, 240)) for i in imgs]
        rows.append(np.hstack(imgs))
    sheet = np.vstack(rows)
    cv2.putText(sheet, f'ep {ep}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(f'{out}/ep{ep:03d}.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('ep', ep, int(row['length']))
