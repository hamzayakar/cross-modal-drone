## Full Curriculum Restart — v5 Redesign

**Date:** 2026-04-22

### Diagnosis of v1-v4 Curriculum Failures

After reaching Stage 2 v4 (280K best model, 16/20 success, 4 immediate crashes) and observing the trained policy in the GUI, three root-cause issues were identified that justified a full restart rather than continued patching:

**1. Stage 0 velocity_penalty caused deceleration transfer (primary)**

Stage 0 v4 used `velocity_penalty=0.08 × |lateral_vel|` which penalises speed near the hover target. Over 10M training steps this baked a "near target = slow down" prior into the value function. This transferred through Stage 1 (120K steps, insufficient to override) and Stage 2 (280K steps before collapse). Observed in GUI: drone decelerates to near-zero at ~1m from each coin, then hovers extremely slowly for collection.

Confirmed by literature (arXiv 2501.18490): keep reward structure consistent across stages; use action smoothness (‖Δa‖²) not velocity magnitude penalties in hover.

**2. Stage 0 episode length (60s) allowed imprecise hover to pass threshold**

With `max(0, 2-dist²)` reward: hovering at 0.5m yields 1.75/step × 14400 = 25,200 pts — near the 25,000 declared threshold. Policy correctly identified 0.4-0.5m as the effort/reward optimum and never learned tight position control. The transition to navigation stages carried this imprecision.

Fix: 15-second episodes + `max(0, 2-4·dist²)` which zeros at 0.71m and gives 4× stronger gradient near origin.

**3. No yaw alignment = CNN distillation failure guaranteed**

Quadrotor omnidirectionality means the drone never needs to face the coin to collect it. Without explicit incentive, ~60-80% of near-target approach frames have the coin off-camera (estimated from FOV geometry). Since CNN student requires teacher actions to be recoverable from camera frames, an omnidirectional teacher generates unlearnable training data for a forward-camera student.

Fix: `r_yaw = 0.15 × cos(θ_error)` at dist < 2.5m.

**4. Trajectory squiggly — no velocity direction incentive**

Pure progress reward (`50 × Δdist`) rewards any movement that closes distance, including sideways and backward approaches. No incentive for straight-line efficient paths.

Fix: `r_dir = 0.20 × dot(v̂, û_target)` — trajectory constraint compatible with omnidirectionality.

### New Design Summary

| Component | Old (v4) | New (v5) |
|---|---|---|
| Hover reward | `max(0, 2-dist²)` | `max(0, 2-4·dist²)` |
| Hover velocity term | `velocity_penalty=0.08` | `smoothness_penalty=0.05` |
| Stage 0 max_steps | 14400 (60s) | 3600 (15s) |
| Stage 0 threshold | 25000 | 6000 (~0.25m avg dist) |
| Nav: approach zone | none | `+150 × progress` at dist < 1.5m |
| Nav: yaw alignment | none | `+0.15 × cos(θ)` at dist < 2.5m |
| Nav: trajectory | none | `+0.20 × dot(v̂, û_target)` |
| Stage 0 run_name | Stage_0_Hover_v4 | Stage_0_Hover_v5 |
| Stage 1 run_name | Stage_1_Scout_v1 | Stage_1_Scout_v2 |
| Stage 2 run_name | Stage_2_Navigator_v4 | Stage_2_Navigator_v5 |

Stage advancement criteria (manual enforcement):
- Threshold exceeded in **3 consecutive evals** (not just one peak)
- Max **1 early crash** per eval (episode < 1000 steps, negative reward)

---

## Stage 0 v5 — Declared Solved

**Date:** 2026-04-22

### Result

| Eval | Steps (session 2) | Mean Reward | Ep Len | Full Episodes |
|---|---|---|---|---|
| 1 | 140,000 | 6,233.5 | 3600 | 20/20 |
| 2 | 280,000 | 6,244.6 | 3600 | 20/20 |
| 3 | 420,000 | 6,111.7 | 3600 | 20/20 |

3 consecutive evals above threshold (6000) → `ConsecutiveThresholdCallback` fired. **Stage 0 v5 declared solved.**

Total training: ~4.48M steps across 2 sessions (29 evals in session 1 + 3 in session 2).

### What the numbers mean

- **ep_len=3600 in all 60 eval episodes** (20/20 × 3 evals): drone held hover for the full 15 seconds every single time, zero early terminations.
- **Mean ~6100–6250**: back-calculating via `2 − 4d² ≈ mean/3600 + penalty_offset` → average hover distance ~0.23–0.25m from target. Significantly tighter than v4's ~0.5m sweet spot.

### What changed vs v4 (why this worked)

The `4·dist²` scaling with 15s episodes removed the v4 perverse equilibrium where hovering at 0.5m was near-optimal. The `smoothness_penalty` replacing `velocity_penalty` removed the "decelerate near target" prior that had transferred destructively through Stage 1 and Stage 2 in the previous curriculum.

### Decision: proceed to Stage 1

Session 2 plateau was ~6100–6250 with LR decaying to 0 — no further improvement expected. 20/20 full episodes is the cleaner signal than the reward number. Stage 1 will validate whether hover quality is sufficient for navigation transfer.

---

## Stage 1 v2 — Design (pre-training)

**Date:** 2026-04-22

### What changed vs Stage 1 v1

| Component | v1 | v2 |
|---|---|---|
| Coin position | Fixed `[1.0, 0.0, 2.0]` (always +X, 1m) | Random angle, fixed 2m radius from origin |
| Effective nav challenge | Hover-drift to memorized position | Must use compass to navigate to unknown direction |
| Nav rewards | Progress only (from Stage 0.23) | Progress + approach_bonus + yaw_alignment + velocity_direction |
| Episode length | 30s (7200 steps) | 30s (7200 steps, unchanged) |
| Threshold | 800 | 800 |

### Reward math (2m coin, clean collect)

- Long-range progress (2m → 1.5m): 50 × 0.5 = **25 pts**
- Approach zone (1.5m → 0.6m): 200 × 0.9 = **180 pts**
- Coin collection: **300 pts**
- Success bonus: **1000 pts**
- Yaw alignment (~600 steps at <2.5m, cos≈0.8): **~72 pts**
- Velocity direction (~600 steps, dot≈0.8): **~96 pts**
- **Total clean collect: ~1673 pts**

Threshold 800 requires ~8/20 episode collections. Stage 0 v5 compass skill should transfer directly; expect 10-18/20 from the first eval.

### Expectation

Near-zero-shot transfer is likely but not guaranteed. v1 was fully zero-shot because the coin was within hover-drift range (1m). v2 coin at 2m genuinely requires departure from hover zone — the drone must commit to directional flight. The approach_bonus (+150×progress at <1.5m) counters any residual deceleration prior. The yaw alignment reward starts training here for the first time.

Expected outcome: solved in 1–4 evals (140–560K steps). If threshold is not hit by eval 3, something structural is wrong with the hover→nav transfer.

---

## Stage 1 v2 — Run 1 Post-Mortem (arc-trajectory failure)

**Date:** 2026-04-23

### Result

| Eval | Steps | Mean Reward | Ep Len | Notes |
|---|---|---|---|---|
| 1 | 140K | 1508.7 | 1270 | Collecting, but arcing |
| 2 | 280K | 1400.0 | 1340 | Same |
| 3 | 420K | 1439.4 | 1500 | Same — threshold 800 passed, training stopped |

Threshold 800 was passed (3 consecutive) → training stopped. But observed behavior in the GUI revealed the drone consistently **collected the coin via arc trajectories**: curved paths, circles around collection radius, stops and reaccelerations, sideways/backward approaches. The yaw alignment metric was effectively ignored.

### Root Cause

The yaw alignment reward (`0.15 × cos(θ)`) was **~3× weaker** than the approach bonus signal (`200 × Δdist/step ≈ 0.42/step` at 0.5m/s). The drone maximized distance-closure by flying sideways (omnidirectional quadrotor) — no yaw required. The arcs are locally optimal paths for approach bonus exploitation. The threshold of 800 was also too low: arc behavior still collected 20/20 and hit 1500+ mean.

### What Changed for v3

Two changes together enforce "face-first, then fly":

**1. Conditional approach_bonus** (`approach_bonus_requires_yaw: true`, `threshold: 0.5`):
The 3× approach multiplier only fires when `cos(θ) > 0.5` (coin within 60° of drone nose). Arc trajectories lose the multiplier: reward drops from ~205 pts progress to ~70 pts. Arcing behavior cannot pass threshold 1900 even with 20/20 collections (~1622 pts/episode < 1900).

**2. Yaw alignment weight**: `0.15 → 0.5` — now comparable in magnitude to approach signal. Each step of good yaw alignment (~0.5 pts) is meaningful relative to approach bonus (~0.42 pts).

Literature anchor: Penicka et al. ICRA 2023 (arXiv 2210.01841) — perception-aware reward shaping for camera-compatible teacher training.

### Virtual FOV / PD-override proposal (evaluated and rejected)

The idea of suppressing the compass target when no coin is in FOV and having a PD controller forcibly spin the drone was evaluated via wiki literature search. Rejected because:
1. PPO is on-policy — steps where PD overrides yaw produce invalid importance ratios in the rollout buffer
2. Restricted state distribution: RL never learns to turn on its own
3. Mode-transition jitter at PD→RL handoff creates out-of-distribution states

The soft reward approach (conditional approach_bonus + higher yaw weight) achieves the same behavioral goal without hybrid control.

---

## Stage 1 v3 — Design (pre-training)

**Date:** 2026-04-23

Same environment as v2 (coin at random angle, 2m radius, Z=2m) with strengthened rewards:
- `yaw_alignment_weight`: 0.15 → **0.5**
- `approach_bonus_requires_yaw`: **true** (new — unlocks 3× only when cos > 0.5)
- `reward_threshold`: 800 → **1900**
- `run_name`: Stage_1_Scout_v3 (fresh start from Stage_0_Hover_v5 weights)

### Expected reward structure

| Behavior | Progress | Yaw | Vel | Coin+Success | Total/ep |
|---|---|---|---|---|---|
| Good (yaw-aligned, 0.3m/s) | 205 | ~515 | ~197 | 1300 | **~2216** |
| Arc (not facing coin) | 70 | ~180 | ~72 | 1300 | **~1622** |
| No collect (crash) | ~50 | ~50 | ~20 | -300 | **~-180** |

Threshold 1900 requires arcing to be eliminated. 19/20 good collects → mean ~2100 (passes). 18/20 → ~1984 (passes). Arc 20/20 → ~1622 (fails).

---

## Stage 1 v3 — Run Post-Mortem (stall-and-face exploit)

**Date:** 2026-04-23

### Result

| Eval | Steps | Mean | Ep Len | 7200-step episodes |
|---|---|---|---|---|
| 1 | 140K | 1519 | 1274 | 0/20 |
| 4 | 560K | 1812 | 1762 | — |
| 6 | 840K | 2233 | 2716 | — |
| 8 | 1120K | 2781 | **4156** | **7/20 full** |

Correlation(ep_len, reward) = **0.817**. The drone learned that NOT collecting is more profitable than collecting: stalling while facing the coin earns `0.3 × 7200 = 2160` pts vs coin+success = 1300 pts.

### Root Cause

The per-step yaw reward is unbounded over the episode. When `yaw_weight × max_steps > collection_reward` → `0.3 × 7200 = 2160 > 1300`, stalling is a valid local optimum. The drone discovered: hover near coin at 0.7m (outside 0.6m collection radius), face it, accumulate yaw reward for 30 seconds, never collect.

This is a fundamental design flaw of any per-step reward with fixed episode length: the total per-step reward scales with episode duration, while the terminal reward is fixed.

### Fix for v4: `yaw_on_progress_only: true`

Yaw reward fires **only when `coin_progress > 0`** (drone is actively closing distance). Hovering while facing → `coin_progress = 0` → yaw reward = 0. Stalling earns exactly 0 pts. The exploit disappears entirely.

With stalling eliminated, `yaw_alignment_dist` is also extended from 2.5m to **5.0m**: the drone now faces the coin throughout the entire approach (not just the last 2.5m), which is required for the CNN student to see the coin in-camera during the full approach trajectory.

---

## Stage 1 v4 — Design (pre-training)

**Date:** 2026-04-23

All v3 changes retained, with:
- `yaw_on_progress_only: true` — yaw fires only on positive coin_progress
- `yaw_alignment_dist: 5.0` — extended from 2.5m (safe with progress gate)
- `run_name: Stage_1_Scout_v4`
- `reward_threshold: 1800` (unchanged — arc 1538 < 1800, good collect 2011 > 1800)

### Stall-free reward math
| Behavior | Yaw fires? | Total/ep |
|---|---|---|
| Good yaw-aligned collect | Yes (1120 approach steps) | ~2011 pts |
| Arc collect | Yes but cos≈0.3 (1120 steps) | ~1538 pts |
| Stall 7200s, no collect | **No** (no progress) | **0 pts** |

Threshold 1800 definitively rejects both arc behavior and stalling.

---

## FaceIt Stage — Design, 3 Attempts, and Abandonment

**Date:** 2026-04-24

### Motivation

Stage 1 Scout v1–v3 all showed the drone approaching coins laterally (action[2]=yaw_rate never used). Root cause diagnosis: after 4.48M hover-only steps, action[2] weights had 54M gradient updates saying "yaw=0 is always optimal." No reward engineering in Stage 1 could overcome this prior — the drone uses pitch/roll translation to satisfy any yaw-alignment metric geometrically.

Proposed solution: a dedicated FaceIt stage between Hover and Scout — drone already at target, must rotate in place to face a virtual target. Approach gradient removed so only yaw rotation can reduce yaw error.

### FaceIt v1/v2/v3 — Repeated Failure

All three runs failed with the same signature: policy oscillates, drone never reliably rotates to face target. Post-mortem identified four root causes (wiki literature search confirmed none have been addressed in published drone RL work):

1. **VecNormalize stats mismatch** — hover VecNorm calibrated to position-variance distributions; FaceIt yaw-cycling distributions are completely different → normalized obs garbage for yaw learning.
2. **Hover prior too strong** — loaded from 4.48M-step checkpoint; the approach gradient reactivates via position penalty terms despite reward engineering.
3. **Termination too aggressive at 1m** — any yaw rotation causes gyroscopic drift; episode terminates before alignment reward is ever seen.
4. **FaceIt as a separate stage is unvalidated in literature** — no published drone RL paper trains yaw-alignment as a discrete pre-navigation stage. All perception-aware approaches (Penicka et al. ICRA 2023) add yaw as a continuous reward during joint navigation training.

### Decision: Abandon FaceIt

FaceIt removed from curriculum. Stage numbering restored: Stage 0 Hover → Stage 1 Scout → Stage 2 Navigator → ... → Stage 6 Apex.

---

## Stage 0 v6 — Yaw Activation Attempt (weak weight)

**Date:** 2026-04-24

### Design

Instead of a separate FaceIt stage, add a yaw facing reward directly to Stage 0 hover training. Per episode: randomize `hover_yaw_target ∈ [−π, π]`. Reward += `hover_yaw_weight × cos(yaw − hover_yaw_target)`. Compass observation (when yaw weight active): changed from position offset to hover target to a unit direction vector toward `hover_yaw_target` in body frame — same compass structure as nav stage pointing to coin, so the "compass left → turn left" mapping transfers directly.

Parameters: `hover_yaw_weight = 0.15`, `reward_threshold = 6500`, run from scratch.

### Result

Hit threshold (6603, 6654, 6574) but drone did not visibly face the blue arrow target. **Root cause: threshold 6500 was reachable by hover alone.** Perfect hover yields 2.0/step × 3600 = 7200/episode; v5-quality hover yields ~6200. With yaw_weight=0.15, max yaw contribution is 0.15 × 3600 = 540/episode. Drone can hit 6500 at v5-level hover (6200) with only 300 pts of random yaw luck — no genuine alignment required.

---

## Stage 0 v7 — Yaw Activation (correct weight + threshold)

**Date:** 2026-04-25

### Design

Same structure as v6 but with corrected weight and threshold:
- `hover_yaw_weight: 0.5` — 3.3× increase; yaw max = 0.5 × 3600 = 1800/episode
- `reward_threshold: 7500` — hover max alone = 7200 < 7500 → **hover alone cannot pass**
- Trained from scratch (run_name: Stage_0_Hover_v7)

### Threshold analysis

| Behavior | Hover contrib | Yaw contrib | Total | Passes 7500? |
|---|---|---|---|---|
| Perfect hover, random yaw | 7200 | 0 | 7200 | **No** |
| Good hover (1.8/step), 43° alignment | 6480 | 1020 | 7500 | Borderline |
| Good hover, 25° alignment | 6480 | 1632 | 8112 | Yes |

Threshold forces genuine yaw alignment. Drone must consistently face within ~40–55° of target to pass 3 consecutive evals.

### Expected outcome

Drone learns hover + yaw simultaneously from scratch. action[2] gets real gradient from episode 0. After training, action[2] should be active and responsive to yaw error — enabling Stage 1 cos²(θ) gate to work.

---

## Face-First Abandoned — 360° Camera Decision

**Date:** 2026-04-25

### Summary of all failed face-first attempts

After 15+ attempts across different approaches, all failed for the same root reason: **you cannot retrain a converged action dimension with reward engineering.** The teacher's action[2] (yaw_rate) was trained to zero by 4.48M hover steps, and every downstream fix found an alternate path to satisfy the reward proxy without actually rotating.

Failed approaches in order:
1. **FaceIt stage (×3):** Separate pre-nav yaw stage. Failed — hover prior overwhelmed yaw gradient.
2. **cos²(θ) gate on progress:** Made sideways earn 0. Failed — action[2] dormant, no gradient to push it.
3. **FOV masking:** Zeroed compass when coin outside FOV. Failed — same dormant action[2] problem.
4. **Stage 0 v6 (scratch, yaw_w=0.15):** Threshold 6500 reachable by hover alone — no real yaw needed.
5. **Stage 0 v7 (scratch, yaw_w=0.5):** Hover learning phase (0–2.5M steps) killed action[2] again. Plateau at 6400, never reached 7500.
6. **Stage 0 v8 (v5 weights, yaw_w=1.0):** Drone flew toward the blue arrow tip, not rotated to face it — policy's baked-in "compass = position to fly to" semantic was preserved through weight loading.

### The actual fix: 360° camera for student

Proposed and validated by wiki research (Fly360, arXiv:2603.06573, 2026):

**Teacher:** Keep as-is. Approaches coins however it wants. Remove all yaw constraints from nav rewards.

**Student distillation:** 3 × 120° cameras at 0°/120°/240° body-frame yaw, concatenated into 768×64 panoramic strip. Coin always visible at some bearing in panorama. BC label = teacher action. Direction mapping is deterministic (bearing φ → action toward φ). No blank frames. No ambiguity.

**Why this is genuinely different from all previous fixes:** Every previous fix tried to change the teacher. This changes the student's sensors to match what the teacher already does. Zero teacher retraining required.

**Architecture for student:** circular-padding CNN (horizontal) + GRU(256) + action history (12D) + IMU (6D).

### Curriculum plan from here

- Stage 0: v5 ✓ (solved, no changes)
- Stage 1: Scout (1 coin, random angle, 2m) — no yaw constraints
- Stage 2: Navigator (4 fixed coins) — no yaw constraints  
- Stage 3: Hunter (10-18 random coins, Z=[1.5,2.5]m)
- Stage 4-6: obstacles introduced progressively
- Distillation: teacher MLP → student CNN with 360° panoramic camera
