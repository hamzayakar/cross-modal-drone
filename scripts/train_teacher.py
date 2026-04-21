import os
import sys
import shutil
import argparse
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList
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
    Automatically copies best_model to models/best/ whenever a new best is saved.
    models/best/ always holds the peak checkpoint for each completed stage.
    No manual copy ever needed.
    """
    def __init__(self, stage_model_dir: str, best_dir: str, run_name: str):
        super().__init__()
        self.stage_model_dir = stage_model_dir
        self.best_dir = os.path.join(best_dir, run_name)

    def _on_step(self) -> bool:
        os.makedirs(self.best_dir, exist_ok=True)
        src_model  = os.path.join(self.stage_model_dir, "best_model.zip")
        src_vecnorm = os.path.join(self.stage_model_dir, "best_model_vecnormalize.pkl")
        if os.path.exists(src_model):
            shutil.copy2(src_model,   os.path.join(self.best_dir, "best_model.zip"))
        if os.path.exists(src_vecnorm):
            shutil.copy2(src_vecnorm, os.path.join(self.best_dir, "best_model_vecnormalize.pkl"))
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
    reward_weights   = config['hover_rewards'] if HOVER_ONLY else config['nav_rewards']
    NUM_OBS  = stage_config['num_obstacles']
    RAND_OBS = stage_config['randomize_obstacles']
    RAND_COINS = stage_config['randomize_coins']
    RUN_NAME = stage_config['run_name']

    print(f"[{RUN_NAME}] Training Initialized (GUI Disabled for speed)...")

    # ── Directory layout ────────────────────────────────────────────────────────
    # logs/teacher_ppo/stage_N/   ← TensorBoard events + evaluations.npz
    # models/<RUN_NAME>/          ← best / latest / final weights + monitor.csv
    # models/best/<RUN_NAME>/     ← auto-updated copy of peak checkpoint
    # ────────────────────────────────────────────────────────────────────────────
    base_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_dir     = os.path.join(base_dir, 'logs', 'teacher_ppo', f'stage_{args.stage}')
    model_dir   = os.path.join(base_dir, 'models')
    best_dir    = os.path.join(model_dir, 'best')

    stage_model_dir = os.path.join(model_dir, RUN_NAME)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(stage_model_dir, exist_ok=True)

    N_ENVS = 14  # 16 WSL2 processors; 14 envs + 2 for main + OS

    def make_env(rank):
        def _init():
            env = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS,
                               randomize_obstacles=RAND_OBS,
                               randomize_coins=RAND_COINS,
                               reward_weights=reward_weights,
                               hover_only=HOVER_ONLY,
                               num_fixed_coins=NUM_FIXED_COINS,
                               fixed_spawn=FIXED_SPAWN)
            # monitor.csv lives with the model, not the log dir
            monitor_path = os.path.join(stage_model_dir, "monitor.csv") if rank == 0 else None
            return Monitor(env, monitor_path)
        return _init

    env_vec = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    eval_env_raw = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS, randomize_obstacles=RAND_OBS,
                                randomize_coins=RAND_COINS, reward_weights=reward_weights,
                                hover_only=HOVER_ONLY, num_fixed_coins=NUM_FIXED_COINS,
                                fixed_spawn=FIXED_SPAWN)
    eval_env_mon = Monitor(eval_env_raw)
    eval_env_vec = DummyVecEnv([lambda e=eval_env_mon: e])

    prev_vecnorm_path = ""
    if args.stage > 0:
        prev_stage_key  = f"stage_{args.stage - 1}"
        prev_run_name   = config['stages'][prev_stage_key]['run_name']
        prev_vecnorm_path = os.path.join(model_dir, prev_run_name, "best_model_vecnormalize.pkl")

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

    elif args.stage > 0 and os.path.exists(prev_vecnorm_path):
        print("Loading previous stage normalization statistics...")
        env = VecNormalize.load(prev_vecnorm_path, env_vec)
        env.training = True
        env.norm_reward = True
        env.gamma = 0.9995
        prev_model_path = os.path.join(model_dir, prev_run_name, "best_model.zip")
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
        StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    ])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=stage_model_dir,
        log_path=log_dir,           # evaluations.npz → logs/teacher_ppo/stage_N/
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best
    )

    latest_model_path = os.path.join(stage_model_dir, "latest_model")
    save_latest_callback = SaveLatestCallback(save_path=latest_model_path, vec_env=env, save_freq=10000)

    callback_list = CallbackList([sync_callback, eval_callback, save_latest_callback])

    print("Training is live! Monitor progress via TensorBoard.")
    model.learn(
        total_timesteps=10_000_000,
        tb_log_name=RUN_NAME,
        callback=callback_list,
        reset_num_timesteps=True
    )

    final_model_path = os.path.join(stage_model_dir, f"teacher_ppo_{RUN_NAME}_final")
    model.save(final_model_path)
    env.save(f"{final_model_path}_vecnormalize.pkl")
    print(f"Training complete! Final model and stats saved.")

    env.close()
    eval_env.close()
