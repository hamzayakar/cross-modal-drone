# Cross-Modal Drone RL: Curriculum Learning

This repository contains a PyBullet-based 3D drone simulation designed to train an autonomous agent using Reinforcement Learning (PPO). The project employs a strict **Curriculum Learning** approach, scaling from a simple empty room up to a highly complex, randomized obstacle-dense environment.

## 🏗️ Project Architecture
* `configs/`: YAML configuration files containing curriculum stages and reward shaping weights (`teacher_ppo.yaml`).
* `drone_env/`: Custom Gymnasium environment integrated with PyBullet physics, including a mathematically accurate 32-D state space (16-D Kinematics + 16-D LiDAR raycasting).
* `models/`: Saved weights (`.zip`) of the best performing PPO agents.
* `scripts/`: Training scripts and CLI tools.
* `notebooks/`: Jupyter Notebooks for live-tracking the agent's progress and debugging.
* `docs/`: Markdown documents logging the evolution of the reward function and "Reward Hacking" behaviors.

## 🚀 Curriculum Learning Stages (5-Step Plan)
To prevent policy instability and catastrophic forgetting, the agent is trained sequentially. The environment parameters (obstacles, coin randomness) are fully managed via `configs/teacher_ppo.yaml`. You can control the training stage via the CLI parameter `--stage`.

### Stage 0: The Baby Step (Kinematics Focus)
Empty room, 4 fixed coins. Goal: Learn basic hover stability and flight towards static coordinates.
```bash
python scripts/train_teacher.py --stage 0
```

### Stage 1: The Toddler (Navigation Focus)
Empty room, random coins. Goal: Generalize flight paths based on dynamic relative distance vectors.
```bash
python scripts/train_teacher.py --stage 1
```

### Stage 2: The Explorer (Memory Focus)
20 fixed obstacles (Seed: 42), fixed coins. Goal: Introduce LiDAR penalties without overwhelming the agent.
```bash
python scripts/train_teacher.py --stage 2
```

### Stage 3: The Navigator (Dynamic Routing)
20 fixed obstacles (Seed: 42), random coins. Goal: Plan paths around known obstacles to reach dynamic targets.
```bash
python scripts/train_teacher.py --stage 3
```

### Stage 4: The Hunter (Full Autonomy)
20 random obstacles, random coins. Goal: True zero-shot obstacle avoidance and dynamic pathfinding in a completely randomized room.
```bash
python scripts/train_teacher.py --stage 4
```

*(Note: The training script is designed for sequential Curriculum Learning. Starting a higher stage will automatically load the `best_model.zip` from the previous stage to build upon the existing policy.)*

## 📊 Live Monitoring & Configuration
**Configuration:** Modify reward coefficients and stage parameters on the fly without changing Python code by editing `configs/teacher_ppo.yaml`.

**Evaluation:** Track the training metrics (Ep Length, Ep Reward, etc.) in real-time using TensorBoard:
```bash
tensorboard --logdir logs/
```

To watch the agent's behavior live in the PyBullet GUI without interrupting the training process, run the Live Tracker Notebook:
* `notebooks/04_watch_agent_live.ipynb`

## ⚖️ Custom Reward Shaping (Hunter Model)
The environment features a dense reward function specifically tuned to prevent "Ceiling Hugging" and "Suicide Policies." It mathematically balances a tight `Alive Bonus` with a `Distance Penalty`, forcing the agent to move toward the target to yield a positive net reward, heavily incentivized by a massive `Coin Collection` spike. (See `docs/reward_evolution.md` for the full history).