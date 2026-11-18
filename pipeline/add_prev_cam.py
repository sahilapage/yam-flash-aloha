"""Add a 'top_prev' camera = the masked top video delayed by K frames (first K repeat frame 0).

Gives ACT two top frames 100 ms apart (K=6 at 60 Hz) so it can see which way and how fast the
ball rolls. Everything else is hard-linked from the source episode; timing and per-sample frame
indices are top's (frame i of top_prev shows top frame i-K).
    python add_prev_cam.py SRC_DATASET DST_DATASET --k 6 [--skip-source-episodes 20]
"""
import argparse, json, os, shutil, subprocess
from pathlib import Path
import zarr

p = argparse.ArgumentParser()
p.add_argument('src'); p.add_argument('dst'); p.add_argument('--k', type=int, default=6)
p.add_argument('--skip-source-episodes', type=int, nargs='*', default=[])
a = p.parse_args()
src, dst = Path(a.src), Path(a.dst)
dst.mkdir(parents=True, exist_ok=False)
for ep in sorted(src.glob('episode_*')):
    man = json.loads((ep / 'manifest.json').read_text())
    if man['source_episode_index'] in a.skip_source_episodes:
        print('skip', ep.name, 'source', man['source_episode_index']); continue
    out = dst / ep.name
    shutil.copytree(ep, out, copy_function=os.link)
    n = man['sample_count']
    (out / 'top_prev.zarr').mkdir()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(ep / 'top.zarr' / 'main.mp4'),
                    '-vf', f'tpad=start={a.k}:start_mode=clone', '-frames:v', str(n), '-r', '60',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-g', '15',
                    str(out / 'top_prev.zarr' / 'main.mp4')], check=True)
    shutil.copy(ep / 'top.timing.jsonl', out / 'top_prev.timing.jsonl')
    ctl = zarr.open_group(str(out / 'control.zarr'), mode='a')
    # control.zarr was hard-linked: write the new group into a real copy so the source is untouched
    zarr.copy(ctl['camera_observation/top'], ctl['camera_observation'], name='top_prev')
    man['camera_names'] = ['top', 'top_prev', 'right_wrist']
    man['top_prev'] = {'source': 'top', 'delay_frames': a.k}
    (out / 'manifest.json').unlink()
    (out / 'manifest.json').write_text(json.dumps(man))
    print(ep.name, 'ok', n, flush=True)
