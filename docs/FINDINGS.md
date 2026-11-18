# 🔬 Findings

What we measured while bringing Flash ALOHA up on a YAM arm, in the order we found it. Every number comes from a script in [`pipeline/`](../pipeline) or from robot telemetry.

---

## 1. Arm tracking was the first bottleneck ✅ fixed

The arm was replaying demo trajectories, open loop, with the stock i2rt PD gains.

- **The measured joints trailed the commands by about 100–130 ms.** A 10-step lookahead hid that by sending targets early; the arm then ran 33–67 ms *ahead* of the demo.
- **The wrist reached only 75–80 % of the demo's speed.** The PD loop's damping brakes all motion when no target velocity is given.

**Fix: velocity feedforward.** Each tick sends the plan's joint velocity (central difference) together with the position target, the way the arm's native `move(pos, vel, tor)` interface works. Lookahead goes from 10 down to 3.

| | Lag | RMS j1–j6 (rad) | Worst (rad) | Wrist speed |
|---|---|---|---|---|
| Before | 100–130 ms | .015–.035 | .126 | 75–80 % |
| After | ≈ 0 | .009–.018 | .064 | 94–102 % |

**Gains.** Stiffer shoulder and elbow (kp 130) brought j2/j3 down to 0.008/0.007 rad without measurable buzz. Stiffer wrist gains (kp 40 on j5/j6) oscillated j6 in the MuJoCo replica, so the wrist stays at kp 10.

**Jerk.** Velocity feedforward turns ACT's small row-to-row wobble into shake. A zero-phase Hann filter (7 rows) on each predicted chunk cuts command jerk 2–8× per joint, with under 6 mrad change to the path.

## 2. Runner bugs that looked like policy failures ✅ fixed

- **The gripper was never reset between runs.** The move to the start pose only moved the joints. Every demo starts with the gripper about 0.37 closed, but live runs started fully open (run 1) or almost fully closed (every run after). With the gripper closed, the policy acts as if it already holds the ball and heads for the ring.
- **A "squeeze" offset ratcheted the gripper shut.** We added +0.15 to gripper commands above 0.38 so the gripper would actually grip, but the start value is 0.37. The policy reads the more-closed gripper back as its state and plans to close further, and within about 1 s the gripper was fully closed on empty air.

## 3. The perception path is correct ✅ ruled out

| Check | Result |
|---|---|
| Live vs training tensor, same recorded frame | mean abs diff **0.006**; a red/blue swap would give 0.015–0.041 |
| Live vs training plan, same frame | max 0.007 rad (row 0) to 0.018 rad (row 59) |
| Wrist-camera brightness, live vs demo | identical (V mean 75.8 vs 72.9–75.4) |
| Does the policy use the images? | true images: 0.060 rad error; images from another moment: plan changes **0.13–0.16 rad** |

## 4. The data, audited episode by episode

`audit_data.py` tracks the yellow ball in every top-camera frame and times the arm and the gripper. The full table is [`results/episode_audit.csv`](../results/episode_audit.csv).

**Good**
- Timing alignment is clean in every episode.
- Rolls are consistent: always from the left edge, across the table toward the robot.
- **The arm reaches out to meet the ball**, 0.5–1.0 rad before contact, and is idle only 10–30 % of the time before the catch.
- **Recovery demos exist.** Episodes 4, 16, 24, 33 and 36 regrasp the ball.
- Live rolls fall inside the demo distribution: same start area and paths, 60–570 px/s live vs 35–860 px/s in the demos.

**Problems**
- **Two protocols mixed together.** Episodes 0–15 are slow rolls (35–176 px/s), often with the ball already on the table at the start. Episodes 16–58 are fast rolls (230–860 px/s).
- **Variable length and start:** episodes run 4–13.5 s, and the ball first appears 0–4.5 s after the start.
- **The gripper start varies from 0.0 to 0.99.** Episodes 13, 29, 30, 31 and 34 start closed.
- **The operator's head enters the top camera,** outside the mask, in episodes 16 and 33. Episode 20, which has the same problem, was already dropped.

## 5. The root cause: memorization 🎯

| 0.5 s-ahead joint error | |
|---|---|
| Episodes the model trained on | **0.011 rad** |
| Held-out episodes | **0.061 rad** (5.5×) |

Each roll in the dataset is unique in angle, speed and catch point; catch locations spread over about 250 × 230 px of the top view. 54 training demos can't cover the mapping from ball path to catch pose, so the model reproduces its training rolls and is several cm off on a new one. The ball is about 6 cm wide.

**A 2-frame model doesn't help.** We added `top_prev` (the masked top frame from 100 ms earlier) so the model could see the ball's velocity. It learned to ignore it: replacing the earlier frame with the current one changes the plan by 0.006 rad, against a 0.06 rad error.

## 6. Why the paper can catch a ball and we can't (yet)

| | Flash ALOHA | Here |
|---|---|---|
| Collection | kinesthetic, operator masked | same |
| Demos | 50 per task | 57 |
| Task variance | objects along **15 cm of tape**; **2–7 s** timed demos | free 2-D roll paths; 4–13.5 s |
| Ball source | **the robot's own left arm tosses it**, so the flight follows from its own joint motion | a human rolls it, so the cameras are the only signal |
| Catch arm rate | 200 Hz | 60 Hz |
| Training | 200k steps, batch 256 | 100k steps, batch 128 |

Their catch works because the ball comes from their own arm, the same way every time. Intercepting a ball the robot didn't throw, from vision alone and about 50 demos, is a harder problem than anything in the paper.

## 7. What's next

1. **Re-record with low variance.** Roll from a fixed ramp and vary only the release point, over about 15 cm. Release on a cue about 0.5 s after the start. Keep one roll speed, the same gripper start (saved in the start pose) and 3–4 s demos. Record 100+ episodes, about 10–15 % of them deliberate regrasps.
2. **Train like the paper:** 200k steps at batch 256, 10 % held out with the checkpoint chosen by held-out error, plus the `proprio_noise` curriculum.
3. **Optionally, give the policy the ball's state** (image position and velocity from the colour detector) as extra inputs. A low-dimensional "where is the ball going" input generalizes from 100 demos far better than pixels do.
4. **GPU inference server** for faster re-planning.
5. **Catching in the air:** a repeatable launcher or self-toss, 100–200 Hz control on the catching arm, and the same low-variance protocol.
