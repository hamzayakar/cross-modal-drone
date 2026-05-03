# Cross-Modal Drone RL

PyBullet-based quadrotor simulation for curriculum reinforcement learning. A PPO teacher policy is trained through a 4-stage curriculum (hover → random-coin navigation), then distilled into a CNN-based student that navigates from camera pixels only.

## Current Status

**Distillation phase** — teacher curriculum complete. Collecting RGB demonstration data for Student A (BC) vs Student B (RL-from-pixels) comparison.

| Stage | Name | Status | Result |
|---|---|---|---|
| 0 | Hover | ✓ Solved | v5, 4.48M steps, mean 6200, 20/20 × 3 evals |
| 1 | Scout | ✓ Solved | v7, ~4M steps, 19/20, ~5% structural crash rate |
| 2 | Navigator | ✓ Solved | v5, ~8.3M steps, 19/20, mean 3675 |
| 3 | Hunter | ✓ Solved | v4, ~10M steps, 20/20, mean 4074 (4 random coins, 4m area) |
| 4–6 | — | Skipped | Scope reduced: distillation begins from Stage 3 |
| Distillation | Student A (BC) | ◉ In progress | Collecting RGB demonstrations |
| Distillation | Student B (RL) | — | PPO from pixels, baseline comparison |

## Project Structure

```
configs/teacher_ppo.yaml        — curriculum stage configs + reward weights
drone_env/drone_sim.py          — Gymnasium env (PyBullet 240Hz, PD controller, 50-D obs)
drone_env/reward_functions.py   — hover + navigation reward functions
drone_env/visual_drone_env.py   — CollectionDroneEnv + VisualDroneEnv (panoramic camera obs)
scripts/train_teacher.py        — PPO teacher training entry point
scripts/collect_teacher_data.py — deploy teacher, record (panorama, proprioception, action) pairs
scripts/train_student_a.py      — Student A: behavioral cloning on teacher demonstrations
scripts/train_student_b.py      — Student B: PPO from pixels directly (baseline)
scripts/evaluate_student.py     — side-by-side A vs B evaluation
scripts/debug_camera.py         — render one frame and save PNG (visual sanity check)
student/student_cnn.py          — StudentNet (CircPadConv2d + GRU-256 + proprioception head)
student/loss_functions.py       — bc_loss (MSE on teacher actions)
viewers/watch_live.py           — watch raw training state (reloads every episode)
viewers/watch_best.py           — watch best_model during training (hot-reload on EvalCallback)
viewers/watch_any.py            — load any completed model (--stage N --model best|final)
models/stage_N/<run_name>/      — active training artifacts: best/final model + monitor.csv
models/best/<run_name>/         — canonical best per run (train script loads from here)
models/archive/<run_name>/      — eval logs (.npz, .csv) for completed/failed runs
logs/teacher_ppo/stage_N/       — TensorBoard events (tensorboard --logdir logs/teacher_ppo)
docs/reward_evo.md              — index of full training history
docs/history/                   — detailed history: pre_curriculum / curriculum_v1 / curriculum_v5
docs/design_notes.md            — stage design decisions + open questions
```

## Architecture

### Teacher (MLP)

**Action space (4-D):** `[Target Pitch, Target Roll, Target Yaw Rate, Target Thrust]`
- Pitch/Roll: ±30°, Yaw Rate: ±2 rad/s, Thrust: 0–19.62 N
- Low-level PD controller (Kp_ang=5, Kd_ang=3, Kp_yaw=2) handles motor mixing at 240 Hz

**Observation space (50-D ego-centric):**
```
1D  Z altitude
2D  Body-frame roll, pitch
2D  sin(yaw), cos(yaw)
3D  Local linear velocity   (body frame)
3D  Local angular velocity  (body frame)
3D  Ego-centric compass     (body-frame vector to nearest coin)
36D Gimbal-stabilized LiDAR (yaw-only rotation)
```

### Student (CNN+GRU)

**Observation:** 360° panoramic camera (3 × 120° cameras at 0°, 120°, 240° relative to drone yaw) + proprioception.

**Why 360°:** After 6 failed attempts to activate yaw on the teacher (action[2] dormant after hover pretraining), the decision was made to change the student's sensors instead. Teacher navigates omnidirectionally; panoramic FOV guarantees the target is always visible regardless of teacher heading.

**Camera spec:**
- Per camera: 64W × 24H RGB — 3 channels, FOV 120°
- Panorama: 192W × 24H (concatenated horizontally), shape `(3, 24, 192)`
- Coin visual radius: 0.25m — ensures ≥4px angular size at 3m distance

**Student architecture:**
```
3×(64×24) RGB cameras → concatenate → (3, 24, 192) panorama
→ CircPadConv2d (circular horizontal padding) 3→32→64→64
→ AdaptiveAvgPool(4, 16) → flatten (4096)
→ concat proprioception (23D: IMU 11D + action history 12D)
→ MLP 4160→256→4 (Tanh)
```
GRU-256 maintains temporal context across steps (velocity estimation, seam continuity).

## Reward Design

**Stage 0 (Hover):**
```
R = max(0, 2 − dist²)
  − tilt_penalty     × (pitch² + roll²)
  − ang_vel_penalty  × |ω|
  − velocity_penalty × √(vx² + vy²)
  − collision_penalty
```

**Stage 1+ (Navigation):**
```
R = progress_weight × (prev_dist − curr_dist)   ← 150 × metres closed per step
  − smoothness_penalty × mean(Δaction²)
  − lidar_penalty      × max(0, 0.1 − min_lidar)
  − survival_penalty                              ← −0.01/step (favours fast collection)
  + coin_reward                                   ← +300 per coin
  + success_bonus                                 ← +1000 if all coins collected
  − collision_penalty                             ← −300 if crash
```

## Key Hyperparameters

```python
gamma        = 0.9995   # Extended for 240Hz: 1-step horizon ≈ 8.3s (vs 0.42s with γ=0.99)
n_steps      = 4096     # Per env per rollout
batch_size   = 1792     # 4096 × 14 envs / 32 mini-batches
N_ENVS       = 14       # SubprocVecEnv (16-thread WSL2: 14 envs + 2 for OS)
ent_coef     = 0.005
log_std_init = -1.2     # Initial std ≈ 0.3
LR           = linear_schedule(3e-4)
net_arch     = [256, 256]
```

## Training

```bash
# Teacher curriculum
python scripts/train_teacher.py --stage 0
python scripts/train_teacher.py --stage 1
python scripts/train_teacher.py --stage 3

# Distillation
python scripts/collect_teacher_data.py --episodes 600
python scripts/train_student_a.py --epochs 100 --eval_every 20
python scripts/train_student_b.py
python scripts/evaluate_student.py --agent both --episodes 20

# Monitoring
tensorboard --logdir logs/teacher_ppo
python scripts/debug_camera.py   # verify coin visibility before collection
```

## Viewers

```bash
python viewers/watch_any.py --stage 3 --model best   # watch Stage 3 Hunter
python viewers/watch_any.py --stage 1 --stride 1     # slow-motion
```

## Key Design Decisions

| Decision | Why |
|---|---|
| Hierarchical PD control | Raw motor control = cognitive overload; attitude commands separate flight physics from navigation |
| gamma=0.9995 | At 240Hz with gamma=0.99, a coin 5s away has value ≈ 0. Horizon must match time domain |
| Body-frame observations | World-frame compass changes meaning as drone rotates — breaks ego-centric student CNN |
| Gimbal-stabilized LiDAR | Full rotation on 2D LiDAR causes projection shrinkage at pitch: 5m range → 3.5m at 45° tilt |
| Progress reward only | velocity_direction (per-step alignment bonus) exploited by zigzag trajectories — removed |
| 360° student camera | Teacher approaches coins from any direction; 3×120° panorama ensures coin always in view |
| VecNorm reset on stage transition | Compass dims (11–13) and ret_rms reset at hover→nav: prevents compass saturation and gradient vanishing |
| Scope: stop at Stage 3 | Professor confirmed: random coins + Stage 3 sufficient for distillation research question |
