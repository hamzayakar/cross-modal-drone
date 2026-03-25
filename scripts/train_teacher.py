import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Add the project root to the system path to import our custom environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drone_env.drone_sim import RoomDroneEnv

if __name__ == "__main__":
    print("Initializing environment for training (GUI Disabled for maximum speed)...")
    
    # Define paths for logs and saved models
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'teacher_ppo'))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create the environment and wrap it with Monitor to log episode rewards/lengths
    env = RoomDroneEnv(gui=False)
    env = Monitor(env, log_dir)
    
    # Setup Evaluation Environment
    # We use a separate evaluation environment to test the agent periodically
    eval_env = RoomDroneEnv(gui=False)
    eval_env = Monitor(eval_env)
    
    # Optional: Stop training early if the agent reaches a consistently high reward
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=2000.0, verbose=1)
    
    # Save the best model automatically based on evaluation results
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=model_dir,
                                 log_path=log_dir, 
                                 eval_freq=10000, # Test the agent every 10,000 steps
                                 deterministic=True, 
                                 render=False,
                                 callback_on_new_best=callback_on_best)
    
    print("Building PPO Agent...")
    # Initialize PPO with a Multi-Layer Perceptron (MLP) architecture
    # Two hidden layers of 64 neurons each for both Policy (pi) and Value (vf) networks
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    
    model = PPO("MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=log_dir,
                learning_rate=0.0003,
                batch_size=64,
                policy_kwargs=policy_kwargs)
    
    print("Starting Training! This might take a few hours...")
    # Train for 3 Million steps. You can stop it manually anytime (Ctrl+C).
    # The best model is automatically saved along the way.
    model.learn(total_timesteps=3_000_000, 
                tb_log_name="PPO_Teacher_Run_1",
                callback=eval_callback)
    
    # Save the final model just in case it didn't hit the threshold
    final_model_path = os.path.join(model_dir, "teacher_ppo_final")
    model.save(final_model_path)
    print(f"Training finished. Final model saved to {final_model_path}.zip")
    
    env.close()
    eval_env.close()