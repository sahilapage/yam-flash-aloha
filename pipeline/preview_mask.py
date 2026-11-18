"""Preview: top camera original | masked | wrist, every episode except dropped ones, 2x speed."""
import sys, subprocess, numpy as np, cv2
from common import episode_meta, episode_data, episode_frames
from yam_mask import YamTopMask
root, out = sys.argv[1], sys.argv[2]
drop = {2}
m = YamTopMask(); meta = episode_meta(root)
W, H = 424, 240
proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{3 * W}x{H}',
                         '-r', '60', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24', out], stdin=subprocess.PIPE)
for ep in meta.index:
    if ep in drop: continue
    q = np.stack(episode_data(root, ep)['observation.state'].to_numpy())
    for i, (top, wr) in enumerate(zip(episode_frames(root, ep, 'top'), episode_frames(root, ep, 'right_wrist'))):
        if i % 2: continue
        a = cv2.resize(top, (W, H)); b = cv2.resize(m(top, q[i, :6]), (W, H)); c = cv2.resize(wr, (W, H))
        f = np.hstack([a, b, c])
        cv2.putText(f, f'ep {ep}  t {i / 60:4.1f}s', (8, 20), 0, 0.55, (0, 255, 255), 2)
        cv2.putText(f, 'top (raw)', (8, H - 8), 0, 0.5, (255, 255, 0), 1)
        cv2.putText(f, 'top (masked = policy input)', (W + 8, H - 8), 0, 0.5, (255, 255, 0), 1)
        cv2.putText(f, 'wrist (unmasked)', (2 * W + 8, H - 8), 0, 0.5, (255, 255, 0), 1)
        proc.stdin.write(f.tobytes())
    print('ep', ep, flush=True)
proc.stdin.close(); proc.wait()
