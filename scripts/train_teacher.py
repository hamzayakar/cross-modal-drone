import os
import sys
import argparse
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drone_env.drone_sim import RoomDroneEnv

if __name__ == "__main__":
    # 1. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train Drone PPO with Curriculum Stages")
    parser.add_argument("--stage", type=int, default=0, help="Training stage level (0-4)")
    args = parser.parse_args()

    # 2. Load configurations from YAML
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
    
    # 3. Setup Directories
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'teacher_ppo'))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    # CRITICAL: Each curriculum stage gets its own dedicated model folder
    stage_model_dir = os.path.join(model_dir, RUN_NAME) 
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(stage_model_dir, exist_ok=True)
    
    # 4. Initialize Environments
    env = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS, randomize_obstacles=RAND_OBS, randomize_coins=RAND_COINS, reward_weights=reward_weights)
    env = Monitor(env, log_dir)
    
    eval_env = RoomDroneEnv(gui=False, num_obstacles=NUM_OBS, randomize_obstacles=RAND_OBS, randomize_coins=RAND_COINS, reward_weights=reward_weights)
    eval_env = Monitor(eval_env)
    
    # 5. Setup Callbacks
    # Stop training early if the agent masters the room (reaches 2000 reward)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=2000.0, verbose=1)
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=stage_model_dir, # Saves to the specific stage folder!
                                 log_path=log_dir, 
                                 eval_freq=10000, 
                                 deterministic=True, 
                                 render=False,
                                 callback_on_new_best=callback_on_best)
    
    # 6. Transfer Learning (Curriculum) Logic
    # If we are on Stage > 0, find the best brain from the previous stage and load it
    if args.stage > 0:
        prev_stage_key = f"stage_{args.stage - 1}"
        prev_run_name = config['stages'][prev_stage_key]['run_name']
        prev_model_path = os.path.join(model_dir, prev_run_name, "best_model.zip")
        
        if os.path.exists(prev_model_path):
            print(f"Found previous brain ({prev_run_name})! Loading weights for Transfer Learning...")
            model = PPO.load(prev_model_path, env=env, tensorboard_log=log_dir)
        else:
            print(f"WARNING: {prev_model_path} not found! Starting from scratch.")
            policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0003, batch_size=64, policy_kwargs=policy_kwargs)
    else:
        print("Stage 0: Creating a fresh brain from scratch...")
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0003, batch_size=64, policy_kwargs=policy_kwargs)
    
    # 7. Start Training
    print("Training is live! Monitor progress via TensorBoard.")
    model.learn(total_timesteps=3_000_000, 
                tb_log_name=RUN_NAME,
                callback=eval_callback,
                reset_num_timesteps=True)
    
    # Save the final model just in case it didn't hit the threshold early
    final_model_path = os.path.join(stage_model_dir, f"teacher_ppo_{RUN_NAME}_final")
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}.zip")
    
    env.close()
    eval_env.close()