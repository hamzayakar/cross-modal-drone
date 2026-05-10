# Cross-Modal Drone RL

PyBullet-based quadrotor simulation for curriculum reinforcement learning. A PPO teacher policy is trained through a 4-stage curriculum (hover → random-coin navigation), then distilled into a CNN-based student that navigates from camera pixels only.

## Current Status

**Distillation phase** — teacher curriculum complete (Stages 0–3). Student A (BC) training in progress; Student B (RL from pixels) queued.

| Stage | Name | Status | Result |
|---|---|---|---|
| 0 | Hover | ✓ Solved | v5, 4.48M steps, mean 6200, 20/20 × 3 evals |
| 1 | Scout | ✓ Solved | v7, ~4M steps, 19/20, mean 1634 |
| 2 | Navigator | ✓ Solved | v5, ~8.3M steps, 19/20, mean 3675 |
| 3 | Hunter | ✓ Solved | v4, ~10M steps, 20/20, mean 4074 (4 random coins, 4m area) |
| 4–6 | — | Skipped | Scope reduced: distillation begins from Stage 3 |
| Student A (BC) | Behavioral Cloning | ◉ Training | epoch 33/100, best SR=45% @ epoch 20 |
| Student B (RL) | PPO from pixels | — | Baseline: quantifies value of distillation |

## Project Structure

```
configs/teacher_ppo.yaml        — curriculum stage configs + reward weights
drone_env/drone_sim.py          — Gymnasium env (PyBullet 240Hz, PD controller, 50-D obs)
drone_env/reward_functions.py   — hover + navigation reward functions
drone_env/visual_drone_env.py   — VisualDroneEnv (panoramic camera obs, VECTOR_DIM=23)
scripts/train_teacher.py        — PPO teacher training entry point
scripts/collect_teacher_data.py — deploy teacher, record (panorama, proprioception, action) pairs
scripts/train_student_a.py      — Student A: behavioral cloning on teacher demonstrations
scripts/train_student_b.py      — Student B: PPO from pixels directly (no teacher demos)
scripts/evaluate_student.py     — side-by-side A vs B evaluation
scripts/debug_camera.py         — render one frame and save PNG (visual sanity check)
student/student_cnn.py          — StudentNet (CircPadConv2d + MLP) + StudentFeatureExtractor (SB3)
student/loss_functions.py       — bc_loss (MSE on teacher actions)
viewers/watch_student.py        — watch student inference live (panoramic camera feed)
viewers/watch_live.py           — watch raw training state (reloads every episode)
viewers/watch_best.py           — watch best_model during training (hot-reload)
viewers/watch_any.py            — load any completed model (--stage N --model best|final)
models/stage_N/<run_name>/      — training artifacts: best/final model + monitor.csv
models/best/<run_name>/         — canonical best per run (train script loads from here)
models/student_a/v1/            — grayscale BC run (baseline)
models/student_a/v2/            — RGB BC run (active)
logs/teacher_ppo/stage_N/       — TensorBoard events (tensorboard --logdir logs/teacher_ppo)
docs/reward_evo.md              — full training history and reward evolution log
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

### Student (CNN + MLP)

**Observation:** 360° panoramic camera (3 × 120° cameras at 0°, 120°, 240°) + proprioception vector. No LiDAR, no compass — all navigation must be inferred visually.

**Why 360°:** After 8 failed attempts to activate yaw on the teacher (action[2] dormant after hover pretraining from 54M "yaw=0" gradient updates), the decision was made to change the student's sensors instead. Teacher navigates omnidirectionally; panoramic FOV guarantees the target is always visible regardless of teacher heading.

**Camera spec:**
- Per camera: 64W × 24H RGB, FOV 120°, nearVal=0.3m (drone arms excluded)
- Panorama: 192W × 24H (concatenated), shape `(3, 24, 192)`

**Proprioception vector (23-D, no privileged info):**
```
11D  IMU: altitude, roll, pitch, sin/cos(yaw), linear vel (3D), angular vel (3D)
12D  Action history: last 3 actions × 4D  (temporal context without RNN)
```

**Student architecture:**
```
3×(64×24) RGB cameras → concatenate → (3, 24, 192) panorama
→ CircPadConv2d (circular padding on W, zero on H): 3→32→64→64
→ AdaptiveAvgPool(4, 16) → flatten → 4096-D
→ concat proprioception (23-D → Linear → 64-D)
→ MLP: 4160 → 256 → 4 (Tanh) → action in [-1, 1]⁴
```

Circular padding on the horizontal axis prevents boundary artifacts at the 0°/360° panorama seam.

## Distillation Methods

The project compares two distillation approaches, chosen to answer the core research question: *does teacher-guided distillation outperform direct RL from pixels?*

| Method | How | Covariate shift | Can exceed teacher | Used for |
|---|---|---|---|---|
| Behavioral Cloning (BC) | Offline: copy teacher actions (MSE) | Yes — student drifts from training distribution | No | Student A |
| Policy Distillation (Rusu 2016) | Offline: copy teacher distribution (KL) | Yes | No | — (marginal gain over BC given small teacher σ) |
| DAgger | Iterative BC: label student-visited states | No | No | — (future work if BC SR < 20%) |
| Distillation-PPO (liu2025) | Online RL + KL regularisation toward teacher | No | Yes | — (future work) |
| Pure RL from pixels | PPO, no teacher | No | Yes | Student B (baseline) |

**Why BC for Student A:** Simplest and most interpretable. Teacher trained with smoothness penalty → low variance → KL ≈ MSE, so Policy Distillation offers no practical gain.

**Why pure RL for Student B:** Establishes a lower bound. If BC (Student A) >> RL (Student B), distillation is proven effective. A Distillation-PPO Student B would muddy this comparison.

**Why not DAgger:** Trigger condition (SR < 20%) not met — Student A reached 45% SR at epoch 20.

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

**Teacher (PPO):**
```python
gamma        = 0.9995   # Extended for 240Hz: 1-step horizon ≈ 8.3s (vs 0.42s with γ=0.99)
n_steps      = 4096     # Per env per rollout
batch_size   = 1792     # 4096 × 14 envs / 32 mini-batches
N_ENVS       = 14       # SubprocVecEnv (16-thread WSL2)
ent_coef     = 0.005
log_std_init = -1.2     # Initial std ≈ 0.3
LR           = linear_schedule(3e-4)
net_arch     = [256, 256]
```

**Student A (BC):**
```python
batch_size   = 256
lr           = 3e-4     # Cosine annealing, T_max = epochs
epochs       = 100
eval_every   = 20       # Live evaluation on 20 episodes every N epochs
```

## Training

```bash
# Teacher curriculum
python scripts/train_teacher.py --stage 0
python scripts/train_teacher.py --stage 1
python scripts/train_teacher.py --stage 3

# Distillation — data collection
python scripts/collect_teacher_data.py --episodes 600
# → filters: successful episodes only, length ≤ p90 (4200 steps)
# → output: data/distill/chunks_v2_rgb/ (24 chunks, 1.17M steps)

# Student A — behavioral cloning
python scripts/train_student_a.py --epochs 100 --eval_every 20 --chunk_dir data/distill/chunks_v2_rgb
# Resume after interruption:
python scripts/train_student_a.py --resume --epochs 100 --eval_every 20 --chunk_dir data/distill/chunks_v2_rgb

# Student B — RL from pixels (run after Student A completes)
python scripts/train_student_b.py

# Evaluation
python scripts/evaluate_student.py --agent both --episodes 20

# Monitoring
tensorboard --logdir logs/teacher_ppo
python scripts/debug_camera.py   # verify coin visibility before collection
```

## Viewers

```bash
python viewers/watch_student.py --model models/student_a/v2/best_model.pt
python viewers/watch_any.py --stage 3 --model best
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
| VecNorm reset on stage transition | Compass dims (11–13) and ret_rms reset at hover→nav: prevents compass saturation |
| Circular padding in CNN | Panorama is angularly continuous: right edge is adjacent to left edge — standard padding adds seam artifact |
| No compass/LiDAR in student obs | Forces true visual navigation — any privileged spatial info would invalidate the cross-modal claim |
| Scope: stop at Stage 3 | Random coins + no obstacles sufficient to validate the distillation research question |
