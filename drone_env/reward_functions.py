import numpy as np

def compute_dense_reward(drone_pos, drone_vel, action, current_distance, is_collision, is_success, lidar_data, coin_collected, action_diff, reward_weights=None):
    """
    Hunter Model — Dense reward shaping for goal-oriented flight behavior.
    Weights are loaded from YAML config. Default values here are kept in sync
    with teacher_ppo.yaml to prevent silent reward economy mismatch in tests.
    """
    # FIX: Default weights now match teacher_ppo.yaml exactly.
    # Previously the defaults were different (e.g. distance_penalty was 0.001 here
    # vs 0.02 in YAML), which caused the "Golden Ratio" spawn balance to break
    # whenever reward_weights=None (e.g. test notebooks, manual env instantiation).
    if reward_weights is None:
        reward_weights = {
            'alive_bonus': 0.02,
            'distance_penalty_multiplier': 0.02,
            'velocity_penalty_multiplier': 0.003,
            'smoothness_penalty_multiplier': 0.05,
            'lidar_penalty_multiplier': 0.5,
            'coin_collection_reward': 300.0,
            'success_bonus': 1000.0,
            'collision_penalty': 300.0
        }

    reward = 0.0
    
    # 1. Alive Bonus — small constant reward for surviving each step
    reward += reward_weights['alive_bonus']
    
    # 2. Distance Penalty — penalizes being far from the nearest coin
    # Balanced against alive_bonus: at exactly 1.0m distance, net reward = 0.0 (Golden Ratio)
    reward -= reward_weights['distance_penalty_multiplier'] * current_distance
    
    # 3. Velocity Penalty — discourages reckless high-speed dives
    velocity_magnitude = np.linalg.norm(drone_vel)
    reward -= reward_weights['velocity_penalty_multiplier'] * velocity_magnitude
    
    # 4. Smoothness Penalty — penalizes erratic action changes (jerk).
    # Produces butter-smooth trajectories for clean CNN distillation datasets.
    reward -= reward_weights.get('smoothness_penalty_multiplier', 0.05) * action_diff
    
    # 5. LiDAR Proximity Penalty — activates only when very close to obstacles
    min_lidar = np.min(lidar_data)
    if min_lidar < 0.1:
        reward -= reward_weights['lidar_penalty_multiplier'] * (0.1 - min_lidar)
    
    # 6. Coin Collection — the primary learning signal
    if coin_collected:
        reward += reward_weights['coin_collection_reward']
    
    # 7. Terminal Conditions
    if is_success:
        reward += reward_weights['success_bonus']
    elif is_collision:
        reward -= reward_weights['collision_penalty']
        
    return reward