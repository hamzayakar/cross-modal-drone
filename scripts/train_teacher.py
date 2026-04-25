import os
import sys
import shutil
import argparse
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drone_env.drone_sim import RoomDroneEnv

class SaveLatestCallback(BaseCallback):
    def __init__(self, save_path, vec_env, save_freq, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.vec_env = vec_env
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            self.vec_env.save(f"{self.save_path}_vecnormalize.pkl")
        return True

class SaveVecNormOnBestCallback(BaseCallback):
    def __init__(self, save_path: str, vec_env: VecNormalize):
        super().__init__()
        self.save_path = save_path
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        self.vec_env.save(os.path.join(self.save_path, "best_model_vecnormalize.pkl"))
        return True

class SyncEvalEnvCallback(BaseCallback):
    """
    CRITICAL FIX: Synchronization between Train and Eval environments.
    Copies running normalization statistics from training env to eval env at every step.
    Without this, eval_env stays frozen at zero stats and the model never saves best_model.zip.
    """
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        return True

class AutoArchiveBestCallback(BaseCallback):
    """
    Mirrors best_model to models/best/<run_name>/ whenever EvalCallback saves a new best.
    models/best/ is the canonical cross-stage reference: train_teacher.py always loads
    the previous stage weights from here, decoupled from working directory layout.
    """
    def __init__(self, stage_model_dir: str, best_dir: str, run_name: str):
        super().__init__()
        self.stage_model_dir = stage_model_dir
        self.best_dir = os.path.join(best_dir, run_name)

    def _on_step(self) -> bool:
        os.makedirs(self.best_dir, exist_ok=True)
        src_model   = os.path.join(self.stage_model_dir, "best_model.zip")
        src_vecnorm = os.path.join(self.stage_model_dir, "best_model_vecnormalize.pkl")
        if os.path.exists(src_model):
            shutil.copy2(src_model,   os.path.join(self.best_dir, "best_model.zip"))
        if os.path.exists(src_vecnorm):
            shutil.copy2(src_vecnorm, os.path.join(self.best_dir, "best_model_vecnormalize.pkl"))
        return True

class ConsecutiveThresholdCallback(BaseCallback):
    """
    Stops training when n_required consecutive evals all exceed reward_threshold.
    Checks evaluations.npz each eval interval rather than only on new-best events,
    so it correctly counts even when reward oscillates below a new peak.
    """
    def __init__(self, reward_threshold: float, evaluations_path: str,
                 n_required: int = 3, eval_freq: int = 10000, n_envs: int = 14,
                 verbose: int = 1):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.evaluations_path = evaluations_path
        self.n_required = n_required
        self.check_interval = eval_freq * n_envs
        self._last_checked = -1

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_checked < self.check_interval:
            return True
        self._last_checked = self.num_timesteps

        npz_path = os.path.join(self.evaluations_path, "evaluations.npz")
        if not os.path.exists(npz_path):
            return True

        means = np.load(npz_path)['results'].mean(axis=1)
        if len(means) < self.n_required:
            return True

        last_n = means[-self.n_required:]
        if all(m >= self.reward_threshold for m in last_n):
            if self.verbose:
                vals = ", ".join(f"{m:.1f}" for m in last_n)
                print(f"\n[ConsecutiveThreshold] {self.n_required} consecutive evals "
                      f">= {self.reward_threshold}: [{vals}] — stopping.")
            return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drone PPO with Curriculum Stages")
    parser.add_argument("--stage", type=int, default=0, help="Training stage level (0-6)")
    args = parser.parse_args()

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'teacher_ppo.yaml'))
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    stage_key = f"stage_{args.stage}"
    if stage_key not in config['stages']:
        print(f"Error: {stage_key} not found in the config file!")
        sys.exit()

    stage_config = config['stages'][stage_key]
    HOVER_ONLY       = stage_config.get('hover_only', False)
    NUM_FIXED_COINS  = stage_config.get('num_fixed_coins', 4)
    FIXED_SPAWN      = stage_config.get('fixed_spawn', False)
    REWARD_THRESHOLD = stage_config.get('reward_threshold', 1600.0)
    MAX_STEPS        = stage_config.get('max_steps', 10800)
    TOTAL_TIMESTEPS  = stage_config.get('total_timesteps', 10_000_000)
    COIN_COUNT_RANGE      = tuple(stage_config.get('coin_count_range', [10, 18]))
    COIN_Z_RANGE          = tuple(stage_config.get('coin_z_range', [1.5, 2.5]))
    COIN_SPAWN_RADIUS     = stage_config.get('coin_spawn_radius', None)
    SEED_WEIGHTS_FROM     = stage_config.get('seed_weights_from', None)
    if HOVER_ONLY:
        reward_weights = config['hover_rewards']
    else:
        reward_weights = config['nav_rewards']
    NUM_OBS    = stage_config['num_obstacles']
    RAND_OBS   = stage_config['randomize_obstacles']
    RAND_COINS = stage_config['randomize_coins']
    RUN_NAME   = stage_config['run_name']

    print(f"[{RUN_NAME}] Training Initialized (GUI Disabled for speed)...")

    # ── Directory layout ────────────────────────────────────────────────────────
    # logs/teacher_ppo/stage_N/          ← TensorBoard events only (pure TB)
    #   <RUN_NAME>_1/                    ← SB3 auto-creates session subdirs
    # models/stage_N/<RUN_NAME>/         ← all run artifacts:
    #   best_model.zip/pkl               ← peak checkpoint (EvalCallback)
    #   latest_model.zip/pkl             ← live snapshot (SaveLatestCallback, deleted at end)
    #   final_model.zip/pkl              ← exact end-of-training state
    #   evaluations.npz                  ← per-eval reward/length arrays (EvalCallback)
    #   monitor.csv                      ← per-episode training log (Monitor)
    # models/best/<RUN_NAME>/            ← canonical cross-stage reference (AutoArchiveBestCallback)
    # ────────────────────────────────────────────────────────────────────────────
    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir   = os.path.join(base_dir, 'logs', 'teacher_ppo', f'stage_{args.stage}')
    model_dir = os.path.join(base_dir, 'models')
    best_dir  = os.path.join(model_dir, 'best')

    stage_model_dir = os.path.join(model_dir, f'stage_{args.stage}', RUN_NAME)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(stage_model_dir, exist_ok=True)

    N_ENVS = 14  # 16 WSL2 processors; 14 envs + 2 for main + OS

    def make_env(rank):
        def _init():
            env = RoomDroneEnv(
                gui=False,
                num_obstacles=NUM_OBS,
                randomize_obstacles=RAND_OBS,
                randomize_coins=RAND_COINS,
                reward_weights=reward_weights,
                hover_only=HOVER_ONLY,
                num_fixed_coins=NUM_FIXED_COINS,
                fixed_spawn=FIXED_SPAWN,
                max_steps=MAX_STEPS,
                coin_count_range=COIN_COUNT_RANGE,
                coin_z_range=COIN_Z_RANGE,
                coin_spawn_radius=COIN_SPAWN_RADIUS,
            )
            monitor_path = os.path.join(stage_model_dir, "monitor.csv") if rank == 0 else None
            return Monitor(env, monitor_path)
        return _init

    env_vec = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    eval_env_raw = RoomDroneEnv(
        gui=False,
        num_obstacles=NUM_OBS,
        randomize_obstacles=RAND_OBS,
        randomize_coins=RAND_COINS,
        reward_weights=reward_weights,
        hover_only=HOVER_ONLY,
        num_fixed_coins=NUM_FIXED_COINS,
        fixed_spawn=FIXED_SPAWN,
        max_steps=MAX_STEPS,
        coin_count_range=COIN_COUNT_RANGE,
        coin_z_range=COIN_Z_RANGE,
        coin_spawn_radius=COIN_SPAWN_RADIUS,
    )
    eval_env_mon = Monitor(eval_env_raw)
    eval_env_vec = DummyVecEnv([lambda e=eval_env_mon: e])

    # Previous stage: always load from models/best/ (canonical, layout-agnostic)
    prev_model_path   = ""
    prev_vecnorm_path = ""
    if args.stage > 0:
        prev_stage_key = f"stage_{args.stage - 1}"
        prev_run_name  = config['stages'][prev_stage_key]['run_name']
        prev_model_path   = os.path.join(best_dir, prev_run_name, "best_model.zip")
        prev_vecnorm_path = os.path.join(best_dir, prev_run_name, "best_model_vecnormalize.pkl")

    def linear_schedule(initial_value: float):
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-1.2
    )
    ppo_kwargs = dict(
        verbose=1, tensorboard_log=log_dir,
        learning_rate=linear_schedule(3e-4),
        n_steps=4096,
        batch_size=1792,  # 4096×14=57344 rollout / 32 mini-batches
        gamma=0.9995,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs
    )

    # ── Model / env setup ───────────────────────────────────────────────────────
    current_best_path   = os.path.join(stage_model_dir, "best_model.zip")
    current_best_vecnorm = os.path.join(stage_model_dir, "best_model_vecnormalize.pkl")

    if os.path.exists(current_best_path):
        print(f"Resuming from current stage best model: {current_best_path}")
        if os.path.exists(current_best_vecnorm):
            env = VecNormalize.load(current_best_vecnorm, env_vec)
            env.training = True
            env.norm_reward = True
            env.gamma = 0.9995
        else:
            env = VecNormalize(env_vec, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.9995)
        model = PPO.load(current_best_path, env=env, tensorboard_log=log_dir, ent_coef=0.005,
                         learning_rate=linear_schedule(3e-4))

    elif SEED_WEIGHTS_FROM:
        # Load hover weights + VecNorm from a named run, but reset compass dims (11-13).
        # v5 VecNorm has compass mean≈0, std≈0.01 (drone always near hover target).
        # New compass is a unit direction vector (std≈0.5) — stale stats would saturate
        # clip_obs=10 and destroy the direction signal. Resetting just those 3 dims gives
        # the policy correct compass normalization from step 0 while keeping all other
        # calibrated stats (altitude, vel, LiDAR) so hover skill works immediately.
        seed_model_path  = os.path.join(best_dir, SEED_WEIGHTS_FROM, "best_model.zip")
        seed_vecnorm_path = os.path.join(best_dir, SEED_WEIGHTS_FROM, "best_model_vecnormalize.pkl")
        print(f"Seeding weights from {SEED_WEIGHTS_FROM} (compass dims reset)...")
        env = VecNormalize.load(seed_vecnorm_path, env_vec)
        env.training = True
        env.norm_reward = True
        env.gamma = 0.9995
        env.obs_rms.mean[11:14] = 0.0
        env.obs_rms.var[11:14]  = 0.25  # std=0.5 matches unit-vector distribution
        model = PPO.load(seed_model_path, env=env, tensorboard_log=log_dir, ent_coef=0.005,
                         learning_rate=linear_schedule(3e-4))

    elif args.stage > 0 and os.path.exists(prev_vecnorm_path):
        print(f"Loading previous stage weights from: {prev_vecnorm_path}")
        env = VecNormalize.load(prev_vecnorm_path, env_vec)
        env.training = True
        env.norm_reward = True
        env.gamma = 0.9995
        if os.path.exists(prev_model_path):
            print(f"Found previous brain ({prev_run_name})! Loading weights...")
            model = PPO.load(prev_model_path, env=env, tensorboard_log=log_dir, ent_coef=0.005,
                             learning_rate=linear_schedule(3e-4))
        else:
            print("WARNING: Previous model not found! Starting from scratch.")
            model = PPO("MlpPolicy", env, **ppo_kwargs)

    else:
        print("Starting from scratch...")
        env = VecNormalize(env_vec, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.9995)
        model = PPO("MlpPolicy", env, **ppo_kwargs)

    eval_env = VecNormalize(eval_env_vec, norm_obs=True, norm_reward=False, clip_obs=10., gamma=0.9995, training=False)

    # ── Callbacks ───────────────────────────────────────────────────────────────
    sync_callback = SyncEvalEnvCallback(train_env=env, eval_env=eval_env)
    auto_archive  = AutoArchiveBestCallback(stage_model_dir=stage_model_dir,
                                            best_dir=best_dir, run_name=RUN_NAME)

    callback_on_best = CallbackList([
        SaveVecNormOnBestCallback(save_path=stage_model_dir, vec_env=env),
        auto_archive,
    ])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=stage_model_dir,
        log_path=stage_model_dir,   # evaluations.npz → models/stage_N/RUN_NAME/
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best
    )

    consecutive_threshold = ConsecutiveThresholdCallback(
        reward_threshold=REWARD_THRESHOLD,
        evaluations_path=stage_model_dir,
        n_required=3,
        eval_freq=10000,
        n_envs=N_ENVS,
    )

    latest_model_path = os.path.join(stage_model_dir, "latest_model")
    save_latest_callback = SaveLatestCallback(save_path=latest_model_path, vec_env=env, save_freq=10000)

    callback_list = CallbackList([sync_callback, eval_callback, consecutive_threshold, save_latest_callback])

    print(f"Training is live! {TOTAL_TIMESTEPS:,} steps, threshold={REWARD_THRESHOLD}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=RUN_NAME,
        callback=callback_list,
        reset_num_timesteps=True
    )

    # Save exact end-of-training snapshot, then remove latest_model.
    # latest_model served nb05 during training; final_model is its permanent form.
    # For threshold runs final ≈ best. For regressed runs it captures the degraded state.
    final_model_path = os.path.join(stage_model_dir, "final_model")
    model.save(final_model_path)
    env.save(f"{final_model_path}_vecnormalize.pkl")
    for ext in ['.zip', '_vecnormalize.pkl']:
        fpath = latest_model_path + ext
        if os.path.exists(fpath):
            os.remove(fpath)
    print(f"Training complete! Best: {stage_model_dir}/best_model.zip  Final: {final_model_path}.zip")

    env.close()
    eval_env.close()
