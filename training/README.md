# ☁️ training/

Configs for the **upstream Flash ALOHA trainer** (unmodified). Copy them into its `src/config/` tree, or let `train_cloud.sh` do it.

| File | What it is |
|---|---|
| `yam_sponge.yaml` | Task: one YAM arm (7-dim state/action), cameras `[top, right_wrist]`, 60 Hz. `DATA_DIR` is filled in by the launcher. |
| `yam_sponge_act.yaml` | ACT: ResNet18, hidden 512, 7 decoder layers, 8 heads, **chunk 60 (1 s)**, batch 128, lr 1.2e-4, 100k steps, crop augmentation. |
| `yam_sponge_prev.yaml` / `yam_sponge_prev_act.yaml` | The 2-frame variant: cameras `[top, top_prev, right_wrist]`, 5 % of episodes held out. |
| `train_cloud.sh` | `setup` (repo, env, data, config check) → `smoke` (200 steps) → `train` (in tmux, logged). |

**Notes from our runs**
- On one H100: about 11 it/s at batch 128, so 100k steps takes about 2.5 h.
- Hold out episodes (`data.train_ratio < 1`) and choose the checkpoint by held-out error. Without that, our first model memorized its training set, and nothing showed it.
- The paper trains for 200k steps at batch 256. The upstream trainer also supports a `proprio_noise` curriculum, which pushes the policy to rely on the cameras.
