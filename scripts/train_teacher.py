import os
import sys
import argparse
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drone_env.drone_sim import RoomDroneEnv

class SaveLatestCallback(BaseCallback):
    def __init__(self, save_path, save_freq, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
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
    
    # lock_z parametresi tamamen kaldırıldı
    env = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS, randomize_obstacles=RAND_OBS, randomize_coins=RAND_COINS, reward_weights=reward_weights)
    env = Monitor(env, log_dir)
    
    eval_env = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS, randomize_obstacles=RAND_OBS, randomize_coins=RAND_COINS, reward_weights=reward_weights)
    eval_env = Monitor(eval_env)
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1600.0, verbose=1)
    
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=stage_model_dir, 
                                 log_path=log_dir, 
                                 eval_freq=10000, 
                                 deterministic=True, 
                                 render=False,
                                 callback_on_new_best=callback_on_best)
                                 
    latest_model_path = os.path.join(stage_model_dir, "latest_model")
    save_latest_callback = SaveLatestCallback(save_path=latest_model_path, save_freq=10000)
    
    callback_list = CallbackList([eval_callback, save_latest_callback])
    
    if args.stage > 0:
        prev_stage_key = f"stage_{args.stage - 1}"
        prev_run_name = config['stages'][prev_stage_key]['run_name']
        prev_model_path = os.path.join(model_dir, prev_run_name, "best_model.zip")
        
        if os.path.exists(prev_model_path):
            print(f"Found previous brain ({prev_run_name})! Loading weights for Transfer Learning...")
            model = PPO.load(prev_model_path, env=env, tensorboard_log=log_dir)
        else:
            print(f"WARNING: {prev_model_path} not found! Starting from scratch.")
            policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
            # gamma=0.9995
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0003, batch_size=128, gamma=0.9995, policy_kwargs=policy_kwargs)
    else:
        print("Stage 0: Creating a fresh, high-capacity brain from scratch...")
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        # gamma=0.9995
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0003, batch_size=128, gamma=0.9995, policy_kwargs=policy_kwargs)
    
    print("Training is live! Monitor progress via TensorBoard.")
    model.learn(total_timesteps=10_000_000, 
                tb_log_name=RUN_NAME,
                callback=callback_list,
                reset_num_timesteps=True)
    
    final_model_path = os.path.join(stage_model_dir, f"teacher_ppo_{RUN_NAME}_final")
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}.zip")
    
    env.close()
    eval_env.close()