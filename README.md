# Cross-Modal Drone RL

PyBullet-based quadrotor simulation for curriculum reinforcement learning. A PPO teacher policy is trained through a 7-stage curriculum (hover → full autonomy), then distilled into a CNN-based student that navigates from camera pixels only.

## Current Status

**Stage 3 (Hunter)** — in progress. Collecting 10–18 random coins in a coin-dense room.

| Stage | Name | Status | Result |
|---|---|---|---|
| 0 | Hover | ✓ Solved | v4, ~10M steps, mean 25.6K, 20/20 full episodes |
| 1 | Scout | ✓ Solved | 120K steps, mean 1299, 20/20 — instant transfer from Stage 0 |
| 2 | Navigator | ✓ Solved | v4 280K best model, 16/20 all-4-coin collection |
| 3 | Hunter | ◉ Training | 10–18 random coins at random heights (Z: 1–6m) |
| 4 | Pathfinder | — | 20 fixed obstacles + 4 fixed coins |
| 5 | Pioneer | — | 20 fixed obstacles + random coins |
| 6 | Apex | — | 20 random obstacles + random coins |

## Project Structure

```
configs/teacher_ppo.yaml      — curriculum stage configs + reward weights
drone_env/drone_sim.py        — Gymnasium env (PyBullet 240Hz, PD controller, 50-D obs)
drone_env/reward_functions.py — hover + navigation reward functions
scripts/train_teacher.py      — PPO training entry point
scripts/distill_policy.py     — teacher → student distillation (future)
models/stage_N/<run_name>/    — working weights (.zip), VecNormalize stats (.pkl), monitor.csv
models/best/<run_name>/       — auto-updated canonical best checkpoint per stage
logs/teacher_ppo/stage_N/     — TensorBoard events + evaluations.npz per stage
logs/legacy/                  — pre-curriculum and deprecated runs (historical only)
notebooks/                    — PyBullet GUI watchers (live training + eval)
docs/reward_evo.md            — complete training history: every bug, fix, and lesson learned
```

## Architecture

**Action space (4-D):** `[Target Pitch, Target Roll, Target Yaw Rate, Target Thrust]`
- Pitch/Roll: ±30°, Yaw Rate: ±2 rad/s, Thrust: 0–19.62 N
- A low-level PD controller (Kp_ang=5, Kd_ang=3, Kp_yaw=2) handles motor mixing at 240 Hz
- The RL agent controls high-level attitude intent; it does NOT touch raw motor forces

**Observation space (50-D ego-centric):**
```
1D  Z altitude
2D  Body-frame roll, pitch  (nose-down positive; sign matches PD convention)
2D  sin(yaw), cos(yaw)      (continuous encoding — no gimbal-lock singularity)
3D  Local linear velocity   (body frame)
3D  Local angular velocity  (body frame)
3D  Ego-centric compass     (body-frame vector to nearest coin / hover target)
36D Gimbal-stabilized LiDAR (yaw-only rotation — immune to pitch/roll projection shrinkage)
```

All observations are ego-centric (body frame) so the teacher's action labels are directly interpretable by a future body-mounted camera student.

## Reward Design

**Stage 0 (Hover):**
```
R = max(0, 2 − dist²)
  − tilt_penalty        × (pitch² + roll²)
  − ang_vel_penalty     × |ω|
  − velocity_penalty    × √(vx² + vy²)
  − collision_penalty   [if crash]
```
Non-negative by design: staying alive is always ≥ crashing. The dist² gradient is 50× stronger near the target than the earlier dist⁴ formulation (which was too flat near origin to enforce tight hovering).

**Stage 1+ (Navigation):**
```
R = progress_weight × (prev_dist − curr_dist)   ← metres closed toward nearest coin × 50
  − smoothness_penalty × mean(Δaction²)
  − lidar_penalty      × max(0, 0.1 − min_lidar)
  + coin_reward                                  [+300 per coin collected]
  + success_bonus                                [+1000 if all coins collected]
  − collision_penalty                            [−300 if crash]
```
The progress reward replaced `alive_bonus − dist_penalty × dist` after Stage 2 collapsed. The old structure gave a negative per-step gradient at distances > 1m (alive=0.02, penalty=0.02×d → breakeven at 1m), making suicide economically optimal for coins 3–4 at 5.66m. Literature: Kaufmann et al. 2023 (Swift, Nature) uses identical pure-progress reward for champion-level drone racing.

## Key Hyperparameters

```python
gamma        = 0.9995   # Extended for 240Hz: 1-step horizon ≈ 8.3s (vs 0.42s with 0.99)
n_steps      = 4096     # Per env per rollout
batch_size   = 1792     # 4096 × 14 envs = 57,344 total / 32 mini-batches
N_ENVS       = 14       # SubprocVecEnv (i7 WSL2, 16 threads: 14 envs + 2 for OS/main)
ent_coef     = 0.005    # Minimal entropy (0.05 → explosion in 120K steps; 0.01 → collapse)
log_std_init = -1.2     # Initial std ≈ 0.3 → ±9° tilt — survives random policy at start
LR           = linear_schedule(3e-4)   # Decays to 0 — shrinks updates as good policy is found
net_arch     = [256, 256]              # pi and vf (needed for cross-correlation at 240Hz)
```

## Training

```bash
python scripts/train_teacher.py --stage 3   # Stage 3 (Hunter) — current
python scripts/train_teacher.py --stage 4   # Stage 4 (Pathfinder)
```

Auto-resume logic (in priority order):
1. `models/<run_name>/best_model.zip` — resume current stage if a best model exists
2. `models/best/<prev_run>/best_model.zip` — load previous stage weights for fresh start
3. Scratch — if nothing found

```bash
tensorboard --logdir logs/teacher_ppo/stage_3   # current stage only
tensorboard --logdir logs/teacher_ppo            # all stages
```

EvalCallback: every 10,000 policy steps, 20 deterministic episodes. `StopTrainingOnRewardThreshold` fires when the stage's `reward_threshold` is crossed.

## Notebooks

| Notebook | Purpose |
|---|---|
| `03_watch_agent_frozen.ipynb` | Load a model once, watch indefinitely. Set `STAGE_TO_WATCH` manually. |
| `04_watch_agent_best.ipynb` | Hot-reload `best_model.zip` after each episode — tracks improving policy during training. |
| `05_watch_agent_live.ipynb` | Watch the live model file as it updates during active training. |

**WSL2 GUI speed note:** Each `addUserDebugLine()` call is an IPC round-trip (~10–15ms). Effective rate ≈ 20–25 Hz vs 240 Hz sim = ~10x slower than real-time. Set `RENDER_STRIDE = 10` for ~real-time display, `1` for slow-motion inspection.

## Key Design Decisions

| Decision | Why |
|---|---|
| Hierarchical PD control | Raw motor control = cognitive overload; agent must learn flight physics AND navigation simultaneously |
| gamma=0.9995 | At 240Hz with gamma=0.99, a coin 5s away has discounted value ≈ 0. Horizon must match time domain |
| Body-frame observations | World-frame compass changes meaning as drone rotates — breaks ego-centric CNN distillation |
| Gimbal-stabilized LiDAR | Full rotation matrix on 2D LiDAR causes projection shrinkage on pitch: effective range 5m→3.5m at 45° |
| Progress reward | `alive_bonus − dist_penalty×dist` is negative at dist>1m → Suicide Policy at far coins |
| Virtual hover target | Compass always active in Stage 0 (points to [0,0,2]) — policy learns one generalizable skill: "minimize compass vector" |
| `models/best/` canonical | AutoArchiveBestCallback keeps it current; safe cross-stage weight loading without brittle path coupling |

## Distillation (future)

Once the Stage 6 (Apex) teacher is trained:
- Teacher MLP (50-D privileged state → 4-D action) generates expert demonstrations
- Student CNN (RGB/depth camera frames → 4-D action) via behavioral cloning / DAgger
- Teacher's ego-centric obs design ensures action labels are camera-frame compatible
- Teacher's smoothness penalty ensures low-jerk trajectories → clean CNN training labels
- Key open question: teacher doesn't always face the coin during approach (flies sideways). Options: yaw-alignment reward, wider camera FOV, or DAgger to handle distribution shift
