# Reward Evolution — Index

Full training history split into three files. Every line preserved.

---

## [1. Pre-Curriculum](history/pre_curriculum.md)

Architectural manifesto (physics/math justification), Phase 0–3 chaos (possum, breakdance, suicide, ceiling-hugging), and the long Stage 0.0–0.29 evolution before the curriculum existed.

**Key milestones:**
- Stage 0.5: Hierarchical PD control (end-to-end motor → attitude commands)
- Stage 0.8: gamma 0.99 → 0.9995 (myopic agent at 240Hz)
- Stage 0.9: Full body-frame transformation (ETH Zurich physics overhaul)
- Stage 0.13: Three hidden monsters (frame conflict, gamma mismatch, state desync)
- Stage 0.15: Gimbal-stabilized LiDAR (projection shrinkage fix)
- Stage 0.26: First successful stage completion ✓ (quartic reward)
- Stage 0.29 v4: 10M steps, final pre-v5 hover result

---

## [2. Curriculum v1](history/curriculum_v1.md)

First curriculum attempt (v1–v4): Hover → Scout (trivially solved 120K) → Navigator (v1–v4, collapse + fixes) → Hunter v1 (Z-height failure). Full restart diagnosis.

**Key milestones:**
- Stage 1 Scout: trivially solved at 120K steps
- Stage 2 v4: 280K best model, 16/20 all-4-coin, regression after peak
- Stage 3 Hunter v1: structural failure — Z-height jump
- v5 restart decision: deceleration prior, arc trajectories, obs normalization

---

## [3. Curriculum v5 — current](history/curriculum_v5.md)

v5 full restart to present. Stage 0 v5 solved, Stage 1 failures, FaceIt abandoned, Stage 0 yaw activation v6–v7.

**Key milestones:**
- Stage 0 v5: solved at 4.48M steps, mean 6200, 20/20 × 3 evals
- Stage 1 v2: arc-trajectory failure (approach_bonus dominated yaw)
- Stage 1 v3: stall-and-face exploit (per-step yaw > collection reward)
- FaceIt: 3 failures — VecNorm mismatch, hover prior, unvalidated in literature
- Stage 0 v6: yaw_weight=0.15 too weak (hover alone passed threshold)
- Stage 0 v7: yaw_weight=0.5, threshold=7500 — hover alone cannot pass ← **current**
