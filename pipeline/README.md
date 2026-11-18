# 🔧 pipeline/

Each script is standalone. They share two modules: `common.py` (dataset access, frame decoding, YAM forward kinematics via MuJoCo) and `yam_mask.py` (the operator mask). Scripts that load checkpoints use the upstream Flash ALOHA source tree, set via `FLASH_ALOHA_SRC`.

## 📐 Calibration and masking

| Script | Purpose |
|---|---|
| `common.py` | Episode metadata and frames from a LeRobot v3 dataset; `YamFK` (MuJoCo forward kinematics of the arm). |
| `dump.py`, `zoom.py`, `sheet.py` | Full-resolution frames with a pixel grid, for hand-picking gripper pixels. |
| `solve_cam.py` | Solves the top camera's pose and focal length from FK plus picked pixels → `top_camera.json`. |
| `overlay.py` | Draws the FK skeleton through the solved camera to check the fit. |
| `analyze.py`, `residual.py` | Where the operator's pixels sit relative to the gripper, and what a candidate mask leaves uncovered. |
| `fit_polygon.py` | Fits the gripper-following mask polygon; candidates are scored in `poly_fits.json`. |
| `yam_mask.py` + `yam_mask.json` | **The mask**: a pure function of (image, joint angles). Its SHA-256 is the training/inference contract. |
| `check_mask.py`, `preview_mask.py` | Stress test on every 4th frame; side-by-side preview video. |

## 📦 Data

| Script | Purpose |
|---|---|
| `convert_flash.py` | LeRobot v3 → Flash ALOHA raw episodes, with the mask burned into the top camera. Episodes can be dropped and the work sharded. |
| `add_prev_cam.py` | Builds a 2-frame dataset: adds `top_prev`, the masked top video delayed K frames. |
| `grab.py` | Contact sheets per episode. |
| `audit_data.py` | **Per-episode audit**: yellow-ball track in the top camera, ball size in the wrist camera, arm motion onset, gripper close events (regrasps), catch location, roll speed. Writes `audit.csv` and an overlay image. |
| `demo_envelope.json` | Per-joint min/max over the demos. The runner refuses targets outside it plus a margin. |

## 🔬 Evaluation

| Script | Question it answers |
|---|---|
| `offline_eval.py` | How far is the plan from the demo on held-out episodes? |
| `speed_eval.py` | Does the plan move as fast as the demos? |
| `vision_reliance.py` | Does the policy use the cameras? (true vs swapped vs black images) |
| `prev_reliance.py` | Does a 2-frame model use the earlier frame? |
| `offline_tensor.py` | Does the live preprocessing path produce the same tensor as training? |
| `tel_report.py` | From robot telemetry: lag behind the plan (cross-correlation) and RMS tracking error per joint. |
