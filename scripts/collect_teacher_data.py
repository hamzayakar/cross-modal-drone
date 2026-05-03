"""
Collect teacher demonstrations for Student A (BC distillation).

Deploys Stage 3 best model, runs N episodes in CollectionDroneEnv,
records (panorama, proprioception, action_history, teacher_action) at every step.
Only successful (no-crash, within step budget) episodes are kept.

Writes directly to disk in chunks to avoid RAM overflow.
(In-memory accumulation of 625K+ panoramas = ~11 GB — not viable.)

Output: data/distill/teacher_stage3.npz
  panoramas — (N_steps, 1, PANO_H, PANO_W)
  vectors   — (N_steps, VECTOR_DIM)
  actions   — (N_steps, 4)
"""
import os, sys, argparse, shutil
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drone_env.visual_drone_env import CollectionDroneEnv, VECTOR_DIM, PANO_H, PANO_W, CAM_W, N_CAMS

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH  = os.path.join(BASE_DIR, 'configs', 'teacher_ppo.yaml')
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'best', 'Stage_3_Hunter_v4', 'best_model.zip')
VECNORM_PATH = os.path.join(BASE_DIR, 'models', 'best', 'Stage_3_Hunter_v4', 'best_model_vecnormalize.pkl')
OUT_PATH     = os.path.join(BASE_DIR, 'data', 'distill', 'teacher_stage3.npz')
CHUNK_DIR    = os.path.join(BASE_DIR, 'data', 'distill', 'chunks')

# p90 of Stage 3 v4 successful episode lengths = 4067 → 4200 trims slowest ~10%
MAX_EP_STEPS = 4200
# Flush to disk every N kept episodes (limits RAM to ~N × 4200 × 18KB ≈ 1.5 GB at N=20)
CHUNK_SIZE   = 20


def flush_chunk(chunk_id, pano_buf, vec_buf, act_buf):
    os.makedirs(CHUNK_DIR, exist_ok=True)
    path = os.path.join(CHUNK_DIR, f'chunk_{chunk_id:04d}.npz')
    np.savez_compressed(path,
                        panoramas=np.stack(pano_buf),
                        vectors=np.stack(vec_buf),
                        actions=np.stack(act_buf))
    return path


def merge_chunks(chunk_paths, out_path):
    all_p, all_v, all_a = [], [], []
    for cp in chunk_paths:
        d = np.load(cp)
        all_p.append(d['panoramas'])
        all_v.append(d['vectors'])
        all_a.append(d['actions'])
    np.savez_compressed(out_path,
                        panoramas=np.concatenate(all_p),
                        vectors=np.concatenate(all_v),
                        actions=np.concatenate(all_a))


def main(n_episodes=50):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    sc = config['stages']['stage_3']
    rw = config['nav_rewards']

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

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
    env_norm.training   = False
    env_norm.norm_reward = False
    model = PPO.load(MODEL_PATH, env=env_norm, device='cpu')

    raw_env: CollectionDroneEnv = vec_env.envs[0].env

    n_success = 0
    n_crash   = 0
    n_too_slow = 0
    chunk_id   = 0
    chunk_paths = []

    # In-memory buffer for current chunk only
    buf_pano, buf_vec, buf_act = [], [], []

    print(f"Collecting {n_episodes} episodes "
          f"(keep: success + steps<={MAX_EP_STEPS}, flush every {CHUNK_SIZE} kept)...")

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
                        path = flush_chunk(chunk_id, buf_pano, buf_vec, buf_act)
                        chunk_paths.append(path)
                        buf_pano, buf_vec, buf_act = [], [], []
                        chunk_id += 1
                        print(f"  ep {ep+1}/{n_episodes}  kept={n_success}  "
                              f"crash={n_crash}  too_slow={n_too_slow}  "
                              f"chunk {chunk_id} saved", flush=True)
                elif success:
                    n_too_slow += 1
                else:
                    n_crash += 1

                if (ep + 1) % 50 == 0:
                    total_steps = sum(np.load(cp)['actions'].shape[0] for cp in chunk_paths) + len(buf_act)
                    print(f"  ep {ep+1}/{n_episodes}  kept={n_success}  "
                          f"crash={n_crash}  too_slow={n_too_slow}  "
                          f"steps~{total_steps}", flush=True)
                break

    vec_env.close()

    # Flush remaining buffer
    if buf_pano:
        path = flush_chunk(chunk_id, buf_pano, buf_vec, buf_act)
        chunk_paths.append(path)

    if not chunk_paths:
        print("No successful episodes collected — nothing saved.")
        return

    print(f"\nMerging {len(chunk_paths)} chunks → {OUT_PATH}")
    merge_chunks(chunk_paths, OUT_PATH)
    shutil.rmtree(CHUNK_DIR)

    d = np.load(OUT_PATH)
    print(f"Done. {n_success}/{n_episodes} kept ({n_crash} crashed, {n_too_slow} too slow).")
    print(f"Dataset: {d['actions'].shape[0]} steps → {OUT_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50,
                        help='Collection episodes (50=debug, 1000=full run)')
    args = parser.parse_args()
    main(args.episodes)
