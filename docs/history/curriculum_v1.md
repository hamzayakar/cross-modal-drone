## Stage 1 — Scout: Transfer Result

**Date:** 2026-04-19

### Setup

- 1 fixed coin at world position [1.0, 0.0, 2.0] — exactly 1m from room center
- Drone spawns with ±0.5m XY offset and random yaw (full symmetry breaking)
- Nav reward: alive_bonus=0.02, distance_penalty=0.02×dist (Golden Ratio: net=0/step at 1m), coin_reward=300, success_bonus=1000, collision_penalty=300
- Weights transferred from Stage_0_Hover_v4 best model

### Result: Trivially Solved at First Eval

Training stopped after **120,000 steps** (1 eval). Mean reward: **1,299**. All 20 eval episodes successful.

```
Eval step 120,000 | Mean R: 1299 | Ep Len: 523 | Full: 0/20
```

Monitor episodes showed collection in 171–1,326 simulation steps (0.7–5.5 simulation seconds). No failures across the entire training run.

### Why

The hover policy's core learned skill is "follow the compass vector to minimise it." In Stage 0, the compass pointed to the virtual hover target [0,0,2.0]. In Stage 1, the compass points to the coin — same structure, different destination. The policy transferred instantly with zero additional learning required.

This validates the unified compass architecture introduced in Stage 0.23: by making the hover target a real compass anchor (rather than zeroing the compass in hover), the policy learned a skill that generalised directly to navigation.

**Model saved:** `models/Stage_1_Scout/best_model.zip`

---

## Infrastructure Note: PyBullet GUI Simulation Speed (WSL2)

**Date:** 2026-04-20

### Finding

The PyBullet GUI notebook (notebook 05) runs at approximately **10x slower than real-time** on WSL2. Measured directly:

| Episodes | Steps | Sim time (steps/240Hz) | Real time | Effective Hz | Slowdown |
|---|---|---|---|---|---|
| Ep 1 | 702 | 2.92s | 29.64s | 23.7 Hz | 10.1× |
| Ep 2 | 922 | 3.84s | 36.64s | 25.2 Hz | 9.5× |
| Ep 3 | 873 | 3.64s | 34.61s | 25.2 Hz | 9.5× |
| Ep 8 | 1098 | 4.58s | 46.49s | 23.6 Hz | 10.2× |
| Ep 9 | 946 | 3.94s | 37.60s | 25.2 Hz | 9.5× |

Consistent factor: **~10×**. Short episodes show higher apparent slowdown due to fixed startup overhead per episode being diluted less.

### Root Cause

Each `addUserDebugLine()` and `addUserDebugText()` call is an IPC round-trip to the PyBullet GUI process (~10–15ms each). The main loop calls 3 arrow lines + 1 text label every step, plus trail lines every 12 steps. Total per-step GUI overhead: ~40–50ms. With `time.sleep(1/240)` = 4.17ms sleep, the effective step rate is ~20–25 Hz instead of 240 Hz.

This is a **WSL2 GUI overhead issue**, not a physics or reward issue. Training (headless, no GUI, no sleep, 12 parallel envs) runs thousands of steps per second.

### Consequence

14,400 simulation steps = **60 simulation seconds** (physics time, always correct).
14,400 steps in the GUI notebook ≈ **600 real seconds** (~10 minutes) to watch.

Episode lengths in training eval (e.g., 523 steps for Stage 1 coin collection) represent **2.18 simulation seconds** of drone flight, displayed as ~30 real seconds in the notebook.

### Fix

`RENDER_STRIDE = 10` added to notebook 05. Advances 10 physics steps per GUI frame, making wall-clock time match simulation time. GUI overlays update every 10th step instead of every step. Set `RENDER_STRIDE = 1` to revert to slow-motion for detailed observation.

---

## Stage 2 — Navigator: Policy Collapse & Reward Redesign

**Date:** 2026-04-20

### Training Run (Stage_2_Navigator, 7.92M steps)

Loaded from Stage_1_Scout best model. Ran overnight. Results:

| Phase | Steps | Mean R | Full eps | Diagnosis |
|---|---|---|---|---|
| Rise | 0–1.08M | 221→462 | 0→7/20 | Learning, occasionally collects all 4 coins |
| Farming | 1.2M–3.12M | 180–347 | 17–20/20 | Alive-bonus farming local optimum |
| Onset | 3.36M–4.8M | 0→−56 | 10–18/20 | Farming strategy destabilizing |
| Collapse | 5M–7.92M | −50→−230 | 0–2/20 | Full crash, ep_len 4000–6000 |

Training stopped at 7.92M. Threshold 1500 never approached.

### Root Cause: Suicide Policy at Far Coins

Coins 3 and 4 at `[4,4,2]` and `[-4,-4,2]` — **5.66m from center**. With the existing nav reward structure:

```
alive_bonus:       +0.02/step
distance_penalty:  −0.02 × 5.66 = −0.113/step
net per-step:      −0.093/step  ← NEGATIVE
```

At −0.093/step, the collision penalty (−300) is recouped in only **3,226 steps (13.4 sim seconds)**. Dying near a far coin is economically optimal. This is Stage 0.2 (Suicide Policy) reappearing for targets beyond the 1m Golden Ratio breakeven.

The policy correctly identified alive-bonus farming (~288/episode) as the best achievable strategy, then collapsed from that local optimum through entropy reduction and gradient destruction.

Note: Stage 1 was not a "bad policy." Its coin was at 1m — exactly the Golden Ratio breakeven — so the economics were neutral. The policy genuinely learned compass-following and coin collection. The reward structure was broken only for Stage 2's far coins.

### Fix: Progress Reward (Stage_2_Navigator_v2)

Removed `alive_bonus` and `distance_penalty_multiplier` from nav_rewards entirely. Replaced with:

```python
reward += progress_reward_weight * (prev_dist - curr_dist)
```

`coin_progress` computed per step as distance closed toward nearest coin. Resets cleanly on coin collection (no snap penalty). Initialised from spawn distance in `reset()`.

```yaml
nav_rewards:
  progress_reward_weight: 50.0   # metres closed × 50
  coin_collection_reward: 300.0
  success_bonus: 1000.0
  collision_penalty: 300.0
  smoothness_penalty_multiplier: 0.02
```

**Why this works:**
- Distance-agnostic: same shaped gradient whether coin is 1m or 5.66m away
- Hovering: 0 reward per step (not negative → farming is no longer a local optimum)
- Dying: −300 (always worse than forward movement)
- Literature: Kaufmann et al. 2023 (Swift, Nature) uses pure progress reward for champion-level drone navigation

**Expected reward for full 4-coin run:** ~3,200 (progress ~1000 + coins 1200 + success 1000). Threshold set at 2,000.

Corrupted Stage_2_Navigator model folder deleted. Stage_2_Navigator_v2 starts from Stage_1_Scout weights.

**Code changes:** `drone_sim.py` (prev_coin_distance tracking), `reward_functions.py` (progress reward replaces alive+distance), `configs/teacher_ppo.yaml` (nav_rewards restructured, run_name updated).

---

## Stage 2 — Navigator v3: Episode Length + N_ENVS Fix

**Date:** 2026-04-20

### Why v2 Failed (Full Analysis)

v2 peaked at **2.04M steps** (mean 1121, 11/20 episodes with r>1200 = multi-coin collection). Then regressed continuously through 3.72M steps, ending with r>1200=0/20 for 12 consecutive evals and max reward ~850 (1-2 coins only).

This is **policy instability at long episodes**, not a reward design problem. At 14400 steps (60 sim-seconds), the policy must maintain correct multi-coin behavior for a very long time. Any variance in the rollout causes forgetting. The best strategy was confirmed possible (2.04M proved it), the policy just couldn't hold it consistently.

This is distinct from v1's alive-bonus farming collapse (mean went to -200). v2's mean stayed at 400-700 — the policy was not broken, just inconsistent.

### Changes for v3

**1. max_steps: 14400 → 7200 (60s → 30s)**

Literature range: Swift 6s, DPRL 25s, gym-pybullet-drones 8s. Physical minimum to collect 4 coins at 0.5 m/s average is ~20s — 30s gives 1.5x margin. Shorter episodes mean more resets per hour, more gradient updates, and less opportunity for variance to compound into forgetting.

**2. N_ENVS: 12 → 14**

WSL2 has 16 processors. 14 envs + 2 for main process + OS headroom. ~15% more data per unit time.

**3. batch_size: 1536 → 1792**

Maintains 32 mini-batches: 4096×14=57344 rollout / 32 = 1792.

**4. reward_threshold: 2000 → 1500**

Scaled for shorter episodes. Absolute reward is lower with fewer steps available for progress accumulation, but coin/success bonuses unchanged.

**5. run_name: Stage_2_Navigator_v3**

Starts from Stage_1_Scout weights. Stage_2_Navigator_v2 best model (2.04M, mean 1121) preserved as fallback but not used as starting point to avoid any instability baked into its weights.

---

## Stage 2 — Navigator v4: Coin Geometry Redesign + Episode Length Correction

**Date:** 2026-04-21

### Why v3 Failed (Root Cause)

v3 ran the full 10M step budget. r>1200 = 0/20 across all 71 evals. The policy reliably collected 1-2 coins but never reached coins 3-4.

Root cause: **the coin geometry made the task physically impossible within the 30s (7200 step) episode budget.**

The old fixed positions:
```
Coin 1: [1,  0, 2]   →  1.0m from center
Coin 2: [0,  1.5, 2] →  1.5m from center
Coin 3: [4,  4, 2]   →  5.66m from center
Coin 4: [-4,-4, 2]   →  5.66m from center (OPPOSITE corner from coin 3)
```

Path from coin 3 → coin 4: `[4,4] → [-4,-4]` = 8√2 = **11.3m**. At 0.5 m/s that single leg takes 22.6 sim-seconds. Total full-collection path: ~20m = ~40 sim-seconds = 9600 steps. This physically exceeds the 7200-step budget. The policy was not failing to learn — it was mathematically prevented from completing the task within the time limit.

The v2 peak (11/20 success at 2.04M steps with 14400 step budget) was real capability; v3 cut the budget below the physical minimum.

### Changes for v4

**1. Coin positions redesigned**

New layout — clockwise ring, each coin ~2-3m from center, each in a different quadrant:
```python
[ 1.0,  0.0, 2.0],   # coin 1: 1m, easy entry
[ 0.0,  2.0, 2.0],   # coin 2: 2m, 90° heading change  
[-2.5,  1.5, 2.0],   # coin 3: ~2.9m, 135° heading change
[-1.5, -2.5, 2.0],   # coin 4: ~2.9m, opposite quadrant
```

Total path: ~13m. At 0.5 m/s: ~26 sim-seconds = 6240 steps. Fits in 45s with 1.7x margin.
Each coin forces a genuine heading change; no two consecutive coins are in opposite corners.

**2. max_steps: 7200 → 10800 (30s → 45s)**

26s physical minimum with new geometry. 45s gives comfortable margin for slow/non-optimal paths.

**3. run_name: Stage_2_Navigator_v4**

Starts from Stage_1_Scout weights.

---

## Stage 2 — Navigator v4: First Eval Result & Threshold Correction

**Date:** 2026-04-21

### Result at 140K Steps

Training stopped at the very first eval. New coin geometry worked immediately:

```
Step 140,000 | mean 1569.9 | max 2552.6 | min -24.2 | avg_len 4881 | r>1200: 12/20
```

12/20 episodes collected 2-3 coins. Max 2552 indicates some episodes reaching 3-4 coins. The Stage 1 policy transferred directly — same pattern as Stage 1 solving instantly from Stage 0 weights.

**Root cause of immediate stop:** reward_threshold was 1500, mean was 1569. `StopTrainingOnRewardThreshold` fired after one eval, same mistake as Stage 1 (threshold 600, mean 1299).

### Threshold Correction

1500 → **2000**. Requires more consistent multi-coin collection across all 20 eval episodes. The 12/20 success rate at 60% is not solid enough for Stage 3 transfer.

Training resumes from `Stage_2_Navigator_v4/best_model.zip` (the 1569-mean checkpoint). No restart from Stage 1 weights needed.

### Lesson

Thresholds for navigation stages need to account for the fact that progress reward + coin rewards can easily exceed a low bar even with inconsistent performance. Rule of thumb going forward: set threshold at ~70-80% of the theoretical maximum for the stage rather than a fixed value.

For Stage 2 with 4 coins: theoretical max ≈ 4×300 + 1000 success + ~730 progress = ~2930. Threshold 2000 ≈ 68% of max — requires consistent 3-coin collection or occasional 4-coin completion.

---

## Stage 3 — Hunter v1: Structural Failure (Z-Height Jump)

**Date:** 2026-04-21 → 2026-04-22

**Run:** `Stage_3_Hunter_v1` — 3.92M steps, 28 evals. Model deleted (failed run). Eval data preserved at `logs/teacher_ppo/stage_3/Stage_3_Hunter_v1/evaluations.npz`.

### Config

- 10–18 random coins, random Z ∈ [1.0, 6.0]m
- max_steps: 10800 (45s)
- N_ENVS: 14, started from Stage_2_Navigator_v4 best_model (280K, mean 1999)
- reward_threshold: 2000

### Results

| Quarter | Evals | Avg mean R | Avg max R |
|---|---|---|---|
| Q1 (0–1M) | 7 | 6.9 | 674 |
| Q2 (1–2M) | 7 | 5.6 | 720 |
| Q3 (2–3M) | 7 | 36.8 | 665 |
| Q4 (3–4M) | 7 | 45.3 | 779 |

Coin collection across all 560 eval episodes: 46% crashed/collected nothing, 48% collected 0–1 coins, 4 episodes (0.7%) collected 2+ coins. Zero upward trend over 4M steps. Threshold 2000 never approached. Run stopped and declared a structural failure.

### Root Cause: Two Simultaneous Hard Jumps

**1. Z-height gap (primary):** Stage 2 coins were all at Z=2.0m — the drone learned to navigate in a flat XY plane at constant altitude. Stage 3 coins spawn at Z=1.0–6.0m. Coins at Z=4–6m require 2–4m of altitude gain while navigating — a skill the Stage 2 policy never needed and the compass Z component was effectively ignored. The LiDAR is also horizontal-only (gimbal-stabilized, yaw-only rotation), so it provides no information about coin altitude.

**2. Fixed → random coin positions:** Stage 2 had 4 coins at known fixed positions. Stage 3 distributes 10–18 coins randomly across 16×16×5m = 1280m³. Even with 14 random coins that's ~91m³ per coin on average — a large 3D search space with no prior knowledge.

The policy correctly generalized from Stage 1 to Stage 2 because the coin geometry was similar (familiar XY plane, incremental distances). The Stage 2→3 jump broke both the altitude invariant and the positional familiarity simultaneously.

### Decision: Redesign Stage 3 as Separate Z-generalization Step

Stage 3 v2 will constrain coins to Z ≈ 2.0m (same as Stage 2), forcing XY generalization only. Z navigation will be introduced at a later stage once XY search is mastered.

---

## Distillation Readiness Note: Teacher Yaw Alignment Problem

**Date:** 2026-04-21

### Observation

The teacher drone does not consistently face coins when approaching them. As a quadrotor, it can fly sideways, backward, or at arbitrary angles relative to its nose direction. The red arrow in the GUI (body +Y forward) frequently points in a different direction from the movement vector.

### Why This Matters for Distillation

The student uses a camera fixed to the body frame. If the teacher approaches a coin from an angle where the coin is off-camera, the student receives unlearnable training data: no visual information about the coin, but a teacher action that implies "go here." Behavioral cloning requires that the teacher's reasoning is recoverable from the student's observations.

### Options (to evaluate before distillation begins)

1. **Yaw alignment reward**: Add a small bonus for having the compass vector aligned with body forward (+Y). Incentivizes the teacher to face coins before approaching — makes behavior camera-explainable by construction. Something like `cos(angle_between_compass_and_body_Y) × small_weight`.

2. **Wider camera FOV**: Use 120-150° wide-angle lens for the student. Coin more likely to be in frame even with misaligned yaw. No teacher retraining needed.

3. **DAgger instead of pure behavioral cloning**: Collects new teacher demonstrations at states the student actually reaches, including recovery from "coin off-screen" situations. Handles distribution shift directly.

4. **Accept and measure**: If the drone mostly faces coins when close (body-frame compass naturally incentivizes yaw toward target), the problem may be smaller than it looks. Watch several episodes and measure what fraction of approach steps have the coin significantly off-axis.

**Decision point**: Watch the trained Stage 2/3 teacher carefully. If coin is frequently behind the camera during approach → Option 1 or 3. If rarely → Option 4 or 2.

---

