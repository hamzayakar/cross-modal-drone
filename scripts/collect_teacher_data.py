"""
Collect teacher demonstrations for Student A (BC distillation).

Deploys Stage 3 best model, runs N episodes in CollectionDroneEnv,
records (panorama, proprioception, action_history, teacher_action) at every step.
Only successful (no-crash, within step budget) episodes are kept.

Writes directly to disk in chunks to avoid RAM overflow.
Chunks are never merged — train_student_a.py reads them directly.

Output: data/distill/<chunk_dir>/chunk_NNNN.npz
  panoramas — (N_steps, 3, PANO_H, PANO_W)  RGB
  vectors   — (N_steps, VECTOR_DIM)
  actions   — (N_steps, 4)
"""
import os, sys, argparse
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drone_env.visual_drone_env import CollectionDroneEnv, VECTOR_DIM, PANO_H, PANO_W

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH  = os.path.join(BASE_DIR, 'configs', 'teacher_ppo.yaml')
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'best', 'Stage_3_Hunter_v4', 'best_model.zip')
VECNORM_PATH = os.path.join(BASE_DIR, 'models', 'best', 'Stage_3_Hunter_v4', 'best_model_vecnormalize.pkl')

# p90 of Stage 3 v4 successful episode lengths = 4067 → 4200 trims slowest ~10%
MAX_EP_STEPS = 4200
# Flush to disk every N kept episodes
CHUNK_SIZE   = 20


def flush_chunk(chunk_dir, chunk_id, pano_buf, vec_buf, act_buf):
    os.makedirs(chunk_dir, exist_ok=True)
    path = os.path.join(chunk_dir, f'chunk_{chunk_id:04d}.npz')
    np.savez_compressed(path,
                        panoramas=np.stack(pano_buf),
                        vectors=np.stack(vec_buf),
                        actions=np.stack(act_buf))
    return path


def main(n_episodes=600, chunk_dir=None):
    """
    Deploy Stage 3 teacher and record demonstration data.

    Runs n_episodes episodes and keeps only those that are both successful
    (all coins collected) and short enough (≤ MAX_EP_STEPS = 4200), which
    correspond to the p90 of Stage 3 v4 episode lengths. Slow successful
    episodes are excluded to avoid injecting suboptimal trajectories.

    Data is written to disk in CHUNK_SIZE-episode batches to avoid RAM
    overflow from accumulating 1 M+ steps of panorama frames.

    Args:
        n_episodes: Total episodes to attempt (not all will be kept).
        chunk_dir: Output directory for chunk_*.npz files.
    """
    if chunk_dir is None:
        chunk_dir = os.path.join(BASE_DIR, 'data', 'distill', 'chunks_v2_rgb')

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    sc = config['stages']['stage_3']
    rw = config['nav_rewards']

    def make_env():
        env = CollectionDroneEnv(
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
        return Monitor(env)

    vec_env  = DummyVecEnv([make_env])
    env_norm = VecNormalize.load(VECNORM_PATH, vec_env)
    env_norm.training    = False
    env_norm.norm_reward = False
    model = PPO.load(MODEL_PATH, env=env_norm, device='cpu')

    raw_env: CollectionDroneEnv = vec_env.envs[0].env

    n_success  = 0
    n_crash    = 0
    n_too_slow = 0
    chunk_id   = 0
    chunk_paths = []

    buf_pano, buf_vec, buf_act = [], [], []

    print(f"Collecting {n_episodes} episodes → {chunk_dir}")
    print(f"(keep: success + steps<={MAX_EP_STEPS}, flush every {CHUNK_SIZE} kept)")

    for ep in range(n_episodes):
        obs = env_norm.reset()
        ep_pano, ep_vec, ep_act = [], [], []

        while True:
            pano = raw_env.last_panorama.copy()
            vec  = np.concatenate([raw_env.last_proprioception, raw_env.last_act_hist])

            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env_norm.step(action)

            ep_pano.append(pano)
            ep_vec.append(vec)
            ep_act.append(action[0].copy())

            if done[0]:
                success = info[0].get('is_success', False)
                ep_len  = len(ep_act)

                if success and ep_len <= MAX_EP_STEPS:
                    buf_pano.extend(ep_pano)
                    buf_vec.extend(ep_vec)
                    buf_act.extend(ep_act)
                    n_success += 1

                    if n_success % CHUNK_SIZE == 0:
                        path = flush_chunk(chunk_dir, chunk_id, buf_pano, buf_vec, buf_act)
                        chunk_paths.append(path)
                        buf_pano, buf_vec, buf_act = [], [], []
                        chunk_id += 1
                        print(f"  ep {ep+1}/{n_episodes}  kept={n_success}  "
                              f"crash={n_crash}  slow={n_too_slow}  "
                              f"→ chunk_{chunk_id-1:04d}.npz saved", flush=True)
                elif success:
                    n_too_slow += 1
                else:
                    n_crash += 1

                if (ep + 1) % 50 == 0:
                    saved_steps = sum(np.load(cp)['actions'].shape[0] for cp in chunk_paths)
                    buf_steps   = len(buf_act)
                    print(f"  ep {ep+1}/{n_episodes}  kept={n_success}  "
                          f"crash={n_crash}  slow={n_too_slow}  "
                          f"steps={saved_steps+buf_steps:,}", flush=True)
                break

    vec_env.close()

    # Flush remaining buffer
    if buf_pano:
        path = flush_chunk(chunk_dir, chunk_id, buf_pano, buf_vec, buf_act)
        chunk_paths.append(path)

    if not chunk_paths:
        print("No successful episodes collected.")
        return

    total_steps = sum(np.load(cp)['actions'].shape[0] for cp in chunk_paths)
    print(f"\nDone. {n_success}/{n_episodes} kept "
          f"({n_crash} crashed, {n_too_slow} too slow).")
    print(f"Total: {total_steps:,} steps in {len(chunk_paths)} chunks → {chunk_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',  type=int, default=600)
    parser.add_argument('--chunk_dir', type=str, default=None,
                        help='Output directory (default: data/distill/chunks_v2_rgb)')
    args = parser.parse_args()
    main(args.episodes, args.chunk_dir)
