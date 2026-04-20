# Cross-Modal Drone RL

PyBullet-based 3D quadrotor simulation for curriculum reinforcement learning. A PPO teacher policy is trained through a 7-stage curriculum (hover → full autonomy), then distilled into a CNN-based student that flies from depth camera input only.

## Project Structure

```
configs/teacher_ppo.yaml   — curriculum stage configs + reward weights
drone_env/drone_sim.py     — Gymnasium env (PyBullet 240Hz, PD controller, 50-D obs)
drone_env/reward_functions.py — hover + navigation reward functions
scripts/train_teacher.py   — PPO training entry point
scripts/distill_policy.py  — teacher → student distillation (future)
models/                    — saved weights (.zip) + VecNormalize stats (.pkl)
notebooks/                 — PyBullet GUI watchers (live training + frozen eval)
docs/reward_evo.md         — full training history: every bug, local optimum, and fix
```

## Architecture

**Action space (4-D):** `[Target Pitch, Target Roll, Target Yaw Rate, Target Thrust]` mapped to ±30°, ±30°, ±2 rad/s, 0–19.62 N. A low-level PD controller handles motor mixing at 240Hz.

**Observation space (50-D ego-centric):**
- 1D Z altitude
- 2D body-frame roll, pitch
- 2D sin/cos yaw
- 3D local linear velocity (body frame)
- 3D local angular velocity (body frame)
- 3D ego-centric compass to nearest target (body frame)
- 36D gimbal-stabilized LiDAR (yaw-only rotation)

## Curriculum Stages

| Stage | Name | Obstacles | Coins | Key skill |
|---|---|---|---|---|
| 0 | Hover | 0 | 0 (virtual target) | Stable hover at [0,0,2] |
| 1 | Scout | 0 | 1 fixed (1m away) | Compass following |
| 2 | Navigator | 0 | 4 fixed | Sequential navigation |
| 3 | Hunter | 0 | 10-18 random | Search + collection |
| 4 | Pathfinder | 20 fixed | 4 fixed | Obstacle avoidance |
| 5 | Pioneer | 20 fixed | random | Avoidance + search |
| 6 | Apex | 20 random | random | Full autonomy |

Training loads weights from the previous stage automatically:

```bash
python scripts/train_teacher.py --stage 2
```

## Reward Design

**Stage 0 (Hover):**
```
R = max(0, 2 − dist²) − tilt_penalty×(pitch²+roll²) − ang_vel_penalty×|ω| − vel_penalty×lateral_vel
```

**Stage 1+ (Navigation):**
```
R = progress_weight × (prev_dist − curr_dist) + coin_reward + success_bonus − collision_penalty
```

Progress reward (metres closed toward nearest coin × weight) replaced the original `alive_bonus − distance_penalty×dist` structure after Stage 2 collapsed into alive-bonus farming. See `docs/reward_evo.md` for the full diagnosis.

## Monitoring

```bash
tensorboard --logdir logs/teacher_ppo
```

Notebooks 03–05 watch the agent live in PyBullet GUI. Set `RENDER_STRIDE = 10` for real-time speed, `1` for slow-motion.

## Distillation (future)

Once the Stage 6 teacher is complete: teacher MLP (50-D privileged state) → student CNN (depth camera frames) via behavioral cloning / DAgger. The teacher's ego-centric observation design ensures action labels are camera-frame compatible.
