# Future Optimizations & Known Technical Debts

## 1. Transition Reward (Observation Shock)
- **Issue:** The moment the agent collects a coin, it suddenly receives a huge distance penalty because the next coin is far away (e.g., 6 meters).
- **Proposed Fix:** Soften this shock by granting a transition bonus on the step the coin is collected, e.g., `reward += max(0, 1.0 - velocity_magnitude) * 50`.

## 2. Catastrophic Forgetting (Stage 1 → Stage 2 Transition)
- **Issue:** Stage 1 has no obstacles in the room, so the agent may unlearn how to use the Lidar. When Stage 2 suddenly introduces 20 obstacles, the agent can experience catastrophic forgetting and crash into everything.
- **Proposed Fix:** Place 2–3 "virtual/ghost obstacles" inside Stage 1, or ramp obstacle count up gradually (Curriculum Ramping) instead of jumping from 0 to 20.

## 3. Curriculum Reward Scaling (ETH Zurich Style)
- **Issue:** The agent gets +300 for collecting a coin in an empty room (Stage 0) and also +300 for collecting one in a 20-obstacle randomized room (Stage 4).
- **Proposed Fix:** Shrink rewards as the agent becomes more capable. By Stage 4, drop `coin_collection_reward` to 100 so the agent is optimized to perform harder tasks for less reward.

## 4. Evaluation Overhead (Test Duration)
- **Issue:** The combination of `eval_freq=10000` and `n_eval_episodes=20` can stretch evaluation times significantly as the agent learns to survive longer.
- **Proposed Fix:** Monitor training logs. If Eval starts taking 10–15 minutes, drop `n_eval_episodes` to 10 or parallelize the Eval environment with `SubprocVecEnv`.