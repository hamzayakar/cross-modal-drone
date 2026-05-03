"""
Evaluate and compare Student A (BC) vs Student B (RL) vs Teacher.

Usage:
  python scripts/evaluate_student.py --agent a    # Student A (BC)
  python scripts/evaluate_student.py --agent b    # Student B (RL)
  python scripts/evaluate_student.py --agent both # side-by-side table
"""
import os, sys, argparse
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drone_env.visual_drone_env import VisualDroneEnv

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'teacher_ppo.yaml')


def make_stage3_visual_env():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    sc = config['stages']['stage_3']
    rw = config['nav_rewards']
    return VisualDroneEnv(
        gui=False,
        num_obstacles=sc['num_obstacles'],
        randomize_obstacles=sc['randomize_obstacles'],
        randomize_coins=sc['randomize_coins'],
        reward_weights=rw,
        hover_only=sc['hover_only'],
        num_fixed_coins=sc['num_fixed_coins'],
        max_steps=sc['max_steps'],
        coin_count_range=tuple(sc['coin_count_range']),
        coin_z_range=tuple(sc['coin_z_range']),
        coin_spawn_area=sc['coin_spawn_area'],
    )


def eval_student_a(n_episodes=20):
    import torch
    from student.student_cnn import StudentNet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(BASE_DIR, 'models', 'student_a', 'best_model.pt')
    model = StudentNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = make_stage3_visual_env()
    successes, rewards, ep_lengths = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r, ep_len = 0.0, 0
        while True:
            action = model.predict(obs['image'], obs['vector'], device=device)
            obs, r, terminated, truncated, info = env.step(action)
            ep_r += r; ep_len += 1
            if terminated or truncated:
                successes.append(info.get('is_success', False))
                rewards.append(ep_r)
                ep_lengths.append(ep_len)
                break

    env.close()
    return successes, rewards, ep_lengths


def eval_student_b(n_episodes=20):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from student.student_cnn import StudentFeatureExtractor

    model_dir = os.path.join(BASE_DIR, 'models', 'student_b')
    env = make_stage3_visual_env()
    vec_env = DummyVecEnv([lambda e=Monitor(env): e])
    vec_norm = VecNormalize(vec_env, norm_obs=False, norm_reward=False,
                            clip_obs=10., gamma=0.9995, training=False)

    model = PPO.load(os.path.join(model_dir, 'best_model.zip'), env=vec_norm)

    successes, rewards, ep_lengths = [], [], []
    obs = vec_norm.reset()

    for _ in range(n_episodes):
        ep_r, ep_len = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = vec_norm.step(action)
            ep_r += r[0]; ep_len += 1
            if done[0]:
                successes.append(info[0].get('is_success', False))
                rewards.append(ep_r)
                ep_lengths.append(ep_len)
                obs = vec_norm.reset()
                break

    vec_norm.close()
    return successes, rewards, ep_lengths


def print_results(name, successes, rewards, ep_lengths):
    sr   = sum(successes) / len(successes)
    mr   = np.mean(rewards)
    ml   = np.mean(ep_lengths)
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  SR          : {sum(successes)}/{len(successes)} ({sr*100:.0f}%)")
    print(f"  Mean reward : {mr:.1f}")
    print(f"  Mean ep len : {ml:.0f} steps ({ml/240:.1f}s)")
    print(f"{'─'*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent',    choices=['a', 'b', 'both'], default='both')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()

    if args.agent in ('a', 'both'):
        s, r, l = eval_student_a(args.episodes)
        print_results('Student A — Behavioral Cloning (with Teacher)', s, r, l)

    if args.agent in ('b', 'both'):
        s, r, l = eval_student_b(args.episodes)
        print_results('Student B — RL from Pixels (without Teacher)', s, r, l)
