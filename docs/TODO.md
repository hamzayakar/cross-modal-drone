# Known Issues & Future Considerations

## 1. Catastrophic Forgetting at Stage 3 → Stage 4 Transition

**Issue:** Stages 0–3 have zero obstacles. The drone never sees sub-1.0 LiDAR readings during training. When Stage 4 (Pathfinder) introduces 20 fixed obstacles, the LiDAR suddenly becomes meaningful — but the policy may have learned to ignore it entirely (weights near zero for those 36 inputs). Result: drone may fly into obstacles as if they don't exist.

**Options:**
- Introduce 1–2 ghost obstacles in Stage 3 to keep LiDAR weights active
- Gradually ramp obstacle count (0 → 2 → 5 → 20) over sub-stages
- Accept the regression and let Stage 4 relearn from scratch (slow but may work given strong transferred flight skills)

## 2. Curriculum Reward Scaling

**Issue:** `coin_collection_reward=300` is constant across all stages. Collecting a coin in an empty room (Stage 1) and in a 20-obstacle randomized room (Stage 6) give identical reward. The relative value of a coin arguably should decrease as task difficulty increases.

**Consideration:** Only worth changing if later stages show coin-farming local optima. Don't touch until Stage 4+.

## 3. Evaluation Overhead at Long Episodes

**Issue:** `eval_freq=10000` + `n_eval_episodes=20` with episodes up to 14400 steps. If the policy learns to survive the full 60 sim-seconds consistently, each eval round takes significant real time.

**Fix if needed:** Drop `n_eval_episodes` to 10, or parallelize the eval env with `SubprocVecEnv`.

---

## 4. Stage 1 Role — Accept Trivial Solve

**Design note (from literature synthesis, 2026-04-22):** Stage 1's purpose is to validate the hover→nav transfer, not to train new skills. Müller et al. (arXiv 2501.18490) skip the 1m step entirely. The v5 implementation uses a coin at random angle, 2m radius — this forces genuine compass-driven flight rather than a 1m drift. Expect solve in ~140–280K steps; advance immediately.

**If Stage 1 takes unexpectedly long:** The approach_bonus (<1.5m) and yaw_alignment (<2.5m) rewards are both active at 2m, so the policy has all the gradient it needs. Slow convergence would indicate a hover→nav transfer problem, not a Stage 1 design problem.

---

## 5. Stage 2.5 Buffer — Consider Before Stage 3

**Design note (from literature synthesis, 2026-04-22):** The Stage 2→3 jump is the highest-risk curriculum transition:
- Stage 2: 4 fixed coins, known positions, Z=2m
- Stage 3: 10–18 random coins, random angles, Z∈[1.5,2.5]m

If Stage 3 fails again despite the v2 Z-constraint fix, add a **Stage 2.5**: 6–8 coins at random angles, fixed 3m radius from origin, fixed Z=2m. This isolates XY generalization before introducing scale (10–18 coins) and mild Z variation.

---

## 6. Collection Radius Progression (Stage 4+)

**Design note (from physical scale analysis, 2026-04-22):** Current 0.6m collection radius is well-calibrated — matches Swift's implicit gate-passing tolerance (~2.7× arm span, consistent with ~2–2.4× in Kaufmann 2023). Leave at 0.6m through Stage 3.

**At Stage 4–5:** Consider tightening to 0.4m then 0.3m as obstacle avoidance demands more precise flight. Tightening collection radius adds difficulty without changing reward structure.

Reference: `wiki/synthesis/physical-scale-comparison.md`

---

## 7. LiDAR Activation Before Stage 4

Cross-reference with item 1 (Catastrophic Forgetting). Consider adding 1–2 dummy obstacles in Stage 3 to keep LiDAR weights non-zero before Stage 4's 20-obstacle environment. The 36 LiDAR inputs have seen no sub-1.0 readings for all of Stages 0–3.
