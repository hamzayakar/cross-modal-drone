"""
Student B — RL from pixels (no teacher).

Trains the same StudentFeatureExtractor architecture directly with PPO
on visual observations. Baseline comparison for Student A.

Uses fewer envs (4) than teacher training because camera rendering is expensive.
"""
import os, sys
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drone_env.visual_drone_env import VisualDroneEnv
from student.student_cnn import StudentFeatureExtractor

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'teacher_ppo.yaml')
MODEL_DIR   = os.path.join(BASE_DIR, 'models', 'student_b')
LOG_DIR     = os.path.join(BASE_DIR, 'logs', 'student_b')
RUN_NAME    = 'Student_B_RL_v1'

# Fewer envs: camera rendering is ~5-10× slower than MLP obs
N_ENVS = 4
TOTAL_TIMESTEPS = 10_000_000


def linear_schedule(v):
    return lambda p: p * v


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    sc = config['stages']['stage_3']
    rw = config['nav_rewards']

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    def make_env(rank):
        def _init():
            env = VisualDroneEnv(
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
            return Monitor(env, os.path.join(MODEL_DIR, 'monitor.csv') if rank == 0 else None)
        return _init

    env_vec  = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env_norm = VecNormalize(env_vec, norm_obs=False, norm_reward=True,
                            clip_obs=10., gamma=0.9995)
    # norm_obs=False: image pixels already in [0,1]; vector is small-range proprioception.
    # Normalizing images would break [0,1] constraint and add overhead.

    eval_env_raw = VisualDroneEnv(
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
    eval_env = VecNormalize(DummyVecEnv([lambda e=Monitor(eval_env_raw): e]),
                            norm_obs=False, norm_reward=False,
                            clip_obs=10., gamma=0.9995, training=False)

    policy_kwargs = dict(
        features_extractor_class=StudentFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256], vf=[256]),
    )

    # Adjust n_steps for fewer envs: keep similar total rollout size
    # Teacher: n_steps=4096, N_ENVS=14 → 57344 per update
    # Student B: n_steps=4096, N_ENVS=4 → 16384 per update (acceptable)
    ppo_kwargs = dict(
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=linear_schedule(3e-4),
        n_steps=4096,
        batch_size=512,   # 4096×4=16384 / 32 mini-batches
        gamma=0.9995,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
    )

    current_best = os.path.join(MODEL_DIR, 'best_model.zip')
    if os.path.exists(current_best):
        print(f"Resuming from {current_best}")
        model = PPO.load(current_best, env=env_norm, **{k: v for k, v in ppo_kwargs.items()
                                                        if k not in ('policy_kwargs',)})
    else:
        model = PPO('MultiInputPolicy', env_norm, **ppo_kwargs)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=MODEL_DIR,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    print(f"Training Student B (RL from pixels): {TOTAL_TIMESTEPS:,} steps")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=RUN_NAME,
                callback=eval_cb, reset_num_timesteps=True)

    model.save(os.path.join(MODEL_DIR, 'final_model'))
    env_norm.save(os.path.join(MODEL_DIR, 'final_vecnormalize.pkl'))
    print(f"Done. Model → {MODEL_DIR}")


if __name__ == '__main__':
    main()
