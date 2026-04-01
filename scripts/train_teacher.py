import os
import sys
import argparse
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    CRITICAL FIX 3: Synchronization between Train and Eval environments.
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drone PPO with Curriculum Stages")
    parser.add_argument("--stage", type=int, default=0, help="Training stage level (0-4)")
    args = parser.parse_args()

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'teacher_ppo.yaml'))
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    stage_key = f"stage_{args.stage}"
    if stage_key not in config['stages']:
        print(f"Error: {stage_key} not found in the config file!")
        sys.exit()

    stage_config = config['stages'][stage_key]
    reward_weights = config['rewards'] 
    
    NUM_OBS = stage_config['num_obstacles']
    RAND_OBS = stage_config['randomize_obstacles']
    RAND_COINS = stage_config['randomize_coins']
    RUN_NAME = stage_config['run_name']

    print(f"[{RUN_NAME}] Training Initialized (GUI Disabled for speed)...")
    
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'teacher_ppo'))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    stage_model_dir = os.path.join(model_dir, RUN_NAME) 
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(stage_model_dir, exist_ok=True)
    
    # ========================================================================
    # ENVIRONMENT SETUP
    # FIX: Lambda closure — capture variable explicitly to prevent all lambdas
    # referencing the same object if n_envs is ever increased.
    # ========================================================================
    env_raw = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS, randomize_obstacles=RAND_OBS, randomize_coins=RAND_COINS, reward_weights=reward_weights)
    env_mon = Monitor(env_raw, log_dir)
    env_vec = DummyVecEnv([lambda e=env_mon: e])
    
    eval_env_raw = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS, randomize_obstacles=RAND_OBS, randomize_coins=RAND_COINS, reward_weights=reward_weights)
    eval_env_mon = Monitor(eval_env_raw)
    eval_env_vec = DummyVecEnv([lambda e=eval_env_mon: e])
    
    # ========================================================================
    # VECNORMALIZE — CRITICAL FIX 2: Align gamma with PPO to prevent horizon mismatch
    # ========================================================================
    prev_vecnorm_path = ""
    if args.stage > 0:
        prev_stage_key = f"stage_{args.stage - 1}"
        prev_run_name = config['stages'][prev_stage_key]['run_name']
        prev_vecnorm_path = os.path.join(model_dir, prev_run_name, "best_model_vecnormalize.pkl")
        
    if args.stage > 0 and os.path.exists(prev_vecnorm_path):
        print("Loading previous normalization statistics...")
        env = VecNormalize.load(prev_vecnorm_path, env_vec)
        env.training = True
        env.norm_reward = True
        env.gamma = 0.9995
        
        eval_env = VecNormalize.load(prev_vecnorm_path, eval_env_vec)
        eval_env.training = False
        eval_env.norm_reward = False
        eval_env.gamma = 0.9995
    else:
        env = VecNormalize(env_vec, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.9995)
        eval_env = VecNormalize(eval_env_vec, norm_obs=True, norm_reward=False, clip_obs=10., gamma=0.9995, training=False)
    # ========================================================================

    sync_callback = SyncEvalEnvCallback(train_env=env, eval_env=eval_env)
    
    callback_on_best = CallbackList([
        SaveVecNormOnBestCallback(save_path=stage_model_dir, vec_env=env),
        StopTrainingOnRewardThreshold(reward_threshold=1600.0, verbose=1)
    ])
    
    # FIX: n_eval_episodes raised from default 5 to 20 for reliable evaluation signal.
    # 5 episodes is too noisy for stochastic coin collection tasks.
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=stage_model_dir, 
        log_path=log_dir, 
        eval_freq=10000, 
        n_eval_episodes=20,
        deterministic=True, 
        render=False,
        callback_on_new_best=callback_on_best
    )
                                 
    latest_model_path = os.path.join(stage_model_dir, "latest_model")
    save_latest_callback = SaveLatestCallback(save_path=latest_model_path, vec_env=env, save_freq=10000)
    
    callback_list = CallbackList([sync_callback, eval_callback, save_latest_callback])
    
    # ========================================================================
    # PPO HYPERPARAMETERS
    # FIX: n_steps raised from default 2048 to 4096 for better advantage estimation
    # at 240Hz. Longer rollouts = smoother gradient signal for high-frequency control.
    # batch_size raised proportionally to maintain ~16 mini-batches per update.
    # ========================================================================
    if args.stage > 0:
        prev_model_path = os.path.join(model_dir, prev_run_name, "best_model.zip")
        if os.path.exists(prev_model_path):
            print(f"Found previous brain ({prev_run_name})! Loading weights...")
            model = PPO.load(prev_model_path, env=env, tensorboard_log=log_dir)
        else:
            print("WARNING: Previous model not found! Starting from scratch.")
            policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
            model = PPO(
                "MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                learning_rate=0.0003,
                n_steps=4096,
                batch_size=256,
                gamma=0.9995,
                policy_kwargs=policy_kwargs
            )
    else:
        print("Stage 0: Creating a fresh, high-capacity brain from scratch...")
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
            learning_rate=0.0003,
            n_steps=4096,
            batch_size=256,
            gamma=0.9995,
            policy_kwargs=policy_kwargs
        )
    
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