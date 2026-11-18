"""Shared helpers: dataset episodes, frame decoding, YAM forward kinematics (MuJoCo)."""
import glob
from pathlib import Path
import av, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent

def episode_meta(root):
    return pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{root}/meta/episodes/chunk-*/*.parquet'))]).set_index('episode_index')

def episode_data(root, ep):
    m = episode_meta(root).loc[ep]
    p = f"{root}/data/chunk-{int(m['data/chunk_index']):03d}/file-{int(m['data/file_index']):03d}.parquet"
    d = pd.read_parquet(p)
    return d[d.episode_index == ep].sort_values('frame_index').reset_index(drop=True)

def video_span(root, ep, cam):
    m = episode_meta(root).loc[ep]
    k = f'videos/observation.images.{cam}'
    p = f"{root}/videos/observation.images.{cam}/chunk-{int(m[k+'/chunk_index']):03d}/file-{int(m[k+'/file_index']):03d}.mp4"
    return p, float(m[k + '/from_timestamp']), float(m[k + '/to_timestamp'])

def episode_frames(root, ep, cam):
    """All frames (BGR) of one episode, in order."""
    path, t0, t1 = video_span(root, ep, cam)
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = 'AUTO'
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t < t0 - 1e-3:
                continue
            if t >= t1 - 1e-3:
                return
            yield f.to_ndarray(format='bgr24')

def frame_at(root, ep, cam, idx, fps=60):
    path, t0, _ = video_span(root, ep, cam)
    t = t0 + idx / fps
    with av.open(path) as c:
        s = c.streams.video[0]
        c.seek(int(max(t - 0.5, 0) / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if float(f.pts * s.time_base) >= t - 1e-3:
                return f.to_ndarray(format='bgr24')

class YamFK:
    """Body/site world poses of the YAM arm (i2rt MJCF; joint angles as the i2rt SDK reports them)."""
    def __init__(self, xml=HERE / 'model' / 'yam.xml'):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.jnt_qposadr[self.model.joint(f'joint{i}').id] for i in range(1, 7)]

    def set(self, q6):
        self.data.qpos[:] = 0
        self.data.qpos[self.qadr] = q6
        self.mj.mj_kinematics(self.model, self.data)

    def body(self, name):
        b = self.data.body(name)
        T = np.eye(4); T[:3, :3] = b.xmat.reshape(3, 3); T[:3, 3] = b.xpos
        return T

    def site(self, name):
        s = self.data.site(name)
        T = np.eye(4); T[:3, :3] = s.xmat.reshape(3, 3); T[:3, 3] = s.xpos
        return T
