"""LeRobot v3 (yam-right-sponge, 60 Hz) -> flash-aloha raw episodes, with the FK hand mask
(yam_mask.py) burned into the top camera. The wrist camera is left as is (no hands in view).

Same output layout as the lab's lerobot_to_flash.py; differences:
the mask is a function of qpos only (so rollout applies the exact same one), episodes can be
dropped, and every manifest records the mask contract.

    python convert_flash.py --dataset DIR --out DIR [--drop 2] [--shard I/N] [--no-mask]
"""
import argparse, glob, json, os, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd, zarr
from common import episode_frames
from yam_mask import YamTopMask

CAMERAS = ['top', 'right_wrist']


def encode(path, frames, size, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                             '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
                             '-pix_fmt', 'yuv420p', '-crf', '18', '-g', '15', str(path)], stdin=subprocess.PIPE)
    n = 0
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes()); n += 1
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f'ffmpeg failed for {path}')
    return n


def convert_episode(root, row, data, out_dir, masker, fps):
    ep = int(row.name)
    target = Path(out_dir) / f'episode_{ep:05d}'
    if (target / 'manifest.json').exists():
        return False
    tmp = Path(out_dir) / f'.tmp_episode_{ep:05d}'
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    n = int(row['length'])
    block = data.iloc[int(row['dataset_from_index']):int(row['dataset_to_index'])]
    assert (block.episode_index == ep).all() and len(block) == n, ep
    qpos = np.stack(block['observation.state'].to_numpy()).astype(np.float64)
    action = np.stack(block['action'].to_numpy()).astype(np.float64)
    times = np.arange(n, dtype=np.float64) / fps
    qvel = np.gradient(qpos, times, axis=0)
    control = zarr.open_group(str(tmp / 'control.zarr'), mode='w')
    for name, value in (('timestamps', times), ('elapsed_seconds', times), ('qpos', qpos), ('qvel', qvel),
                        ('commanded_action', action), ('continuity_id', np.zeros(n, dtype=np.int64))):
        control.create_dataset(name, data=value, chunks=False)
    for cam in CAMERAS:
        frames = episode_frames(root, ep, cam)
        first = next(frames); h, w = first.shape[:2]

        def processed():
            for i, img in enumerate([first, *frames]):
                yield masker(img, qpos[min(i, n - 1), :6]) if (masker is not None and cam == 'top') else img

        count = encode(tmp / f'{cam}.zarr' / 'main.mp4', processed(), (w, h), fps)
        if count != n:
            raise ValueError(f'episode {ep} {cam}: {count} video frames, {n} rows')
        g = control.require_group(f'camera_observation/{cam}')
        g.create_dataset('frame_number', data=np.arange(n, dtype=np.int64), chunks=False)
        g.create_dataset('capture_timestamp', data=times, chunks=False)
        g.create_dataset('consumer_received_timestamp', data=times, chunks=False)
        with open(tmp / f'{cam}.timing.jsonl', 'w') as f:
            for i in range(n):
                f.write(json.dumps(dict(frame_number=i, video_frame=i, capture_timestamp=float(times[i]), continuity_id=0)) + '\n')
    (tmp / 'alignment_report.json').write_text(json.dumps(dict(schema_version=1, alignable=True, errors=[], retained_robot_samples=n)))
    (tmp / 'manifest.json').write_text(json.dumps(dict(
        complete=True, retained=True, alignable=True, sample_count=n, camera_names=CAMERAS, control_freq=fps,
        source_episode_index=ep, hand_masked=masker is not None,
        mask_contract=masker.contract() if masker is not None else None, metadata=dict(control_freq=fps))))
    os.rename(tmp, target)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True); p.add_argument('--out', required=True)
    p.add_argument('--drop', type=int, nargs='*', default=[2]); p.add_argument('--only', type=int, nargs='*')
    p.add_argument('--shard', default='0/1'); p.add_argument('--no-mask', action='store_true')
    a = p.parse_args()
    fps = float(json.load(open(f'{a.dataset}/meta/info.json'))['fps'])
    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{a.dataset}/meta/episodes/chunk-*/*.parquet'))]).set_index('episode_index')
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{a.dataset}/data/chunk-*/*.parquet'))], ignore_index=True)
    k, total = (int(x) for x in a.shard.split('/'))
    masker = None if a.no_mask else YamTopMask()
    os.makedirs(a.out, exist_ok=True)
    for ep, row in meta.iterrows():
        if ep in a.drop or ep % total != k or (a.only and ep not in a.only):
            continue
        if convert_episode(a.dataset, row, data, a.out, masker, fps):
            print(f'converted episode {ep} ({int(row["length"])} frames)', flush=True)


if __name__ == '__main__':
    main()
