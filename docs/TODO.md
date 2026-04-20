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
