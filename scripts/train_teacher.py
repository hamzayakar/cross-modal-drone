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

# Special callback to save VecNormalize stats whenever a new best model is found during evaluation. This ensures that we always have the correct normalization stats corresponding to the best model, which is crucial for consistent performance when loading and evaluating later.
class SaveVecNormOnBestCallback(BaseCallback):
    def __init__(self, save_path: str, vec_env: VecNormalize):
        super().__init__()
        self.save_path = save_path
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        self.vec_env.save(os.path.join(self.save_path, "best_model_vecnormalize.pkl"))
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
    # VECNORMALIZE INTEGRATION (WRAPPER INCEPTION AVOIDANCE)
    # ========================================================================
    # First create the raw environments, then wrap with Monitor, then wrap with DummyVecEnv, and only after that apply VecNormalize. This way we avoid double-wrapping VecNormalize and ensure that the same normalization stats are used for both training and evaluation environments.
    env_raw = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS, randomize_obstacles=RAND_OBS, randomize_coins=RAND_COINS, reward_weights=reward_weights)
    env_mon = Monitor(env_raw, log_dir)
    env_vec = DummyVecEnv([lambda: env_mon])
    
    eval_env_raw = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS, randomize_obstacles=RAND_OBS, randomize_coins=RAND_COINS, reward_weights=reward_weights)
    eval_env_mon = Monitor(eval_env_raw)
    eval_env_vec = DummyVecEnv([lambda: eval_env_mon])
    
    # Curriculum: Do we have a previous stage's VecNormalize stats to load? If so, load them into the new VecNormalize wrappers. If not, create fresh VecNormalize wrappers.
    prev_vecnorm_path = ""
    if args.stage > 0:
        prev_stage_key = f"stage_{args.stage - 1}"
        prev_run_name = config['stages'][prev_stage_key]['run_name']
        prev_vecnorm_path = os.path.join(model_dir, prev_run_name, "best_model_vecnormalize.pkl")
        
    if args.stage > 0 and os.path.exists(prev_vecnorm_path):
        print("Loading previous normalization statistics...")
        # Temiz (Zırhsız) ortama yüklüyoruz ki çift sarmalama olmasın!
        env = VecNormalize.load(prev_vecnorm_path, env_vec)
        env.training = True
        env.norm_reward = True
        
        eval_env = VecNormalize.load(prev_vecnorm_path, eval_env_vec)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        # If starting fresh, wrap with new VecNormalize
        env = VecNormalize(env_vec, norm_obs=True, norm_reward=True, clip_obs=10.)
        eval_env = VecNormalize(eval_env_vec, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
    # ========================================================================

    # Save both model and VecNormalize stats when a new best is found during evaluation
    callback_on_best = CallbackList([
        SaveVecNormOnBestCallback(save_path=stage_model_dir, vec_env=env),
        StopTrainingOnRewardThreshold(reward_threshold=1600.0, verbose=1)
    ])
    
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=stage_model_dir, 
                                 log_path=log_dir, 
                                 eval_freq=10000, 
                                 deterministic=True, 
                                 render=False,
                                 callback_on_new_best=callback_on_best)
                                 
    latest_model_path = os.path.join(stage_model_dir, "latest_model")
    save_latest_callback = SaveLatestCallback(save_path=latest_model_path, vec_env=env, save_freq=10000)
    
    callback_list = CallbackList([eval_callback, save_latest_callback])
    
    if args.stage > 0:
        prev_model_path = os.path.join(model_dir, prev_run_name, "best_model.zip")
        if os.path.exists(prev_model_path):
            print(f"Found previous brain ({prev_run_name})! Loading weights...")
            model = PPO.load(prev_model_path, env=env, tensorboard_log=log_dir)
        else:
            print("WARNING: Previous model not found! Starting from scratch.")
            policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0003, batch_size=128, gamma=0.9995, policy_kwargs=policy_kwargs)
    else:
        print("Stage 0: Creating a fresh, high-capacity brain from scratch...")
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0003, batch_size=128, gamma=0.9995, policy_kwargs=policy_kwargs)
    
    print("Training is live! Monitor progress via TensorBoard.")
    model.learn(total_timesteps=10_000_000, 
                tb_log_name=RUN_NAME,
                callback=callback_list,
                reset_num_timesteps=True)
    
    final_model_path = os.path.join(stage_model_dir, f"teacher_ppo_{RUN_NAME}_final")
    model.save(final_model_path)
    env.save(f"{final_model_path}_vecnormalize.pkl")
    print(f"Training complete! Final model and stats saved.")
    
    env.close()
    eval_env.close()