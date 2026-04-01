import numpy as np

def compute_dense_reward(drone_pos, drone_vel, action, current_distance, is_collision, is_success, lidar_data, coin_collected, action_diff, reward_weights=None):
    """
    Hunter Model (Reward Shaping for Goal-Oriented Behavior)
    Dynamic reward weights based on YAML config.
    """
    # Default reward weights if not provided
    if reward_weights is None:
        reward_weights = {
            'alive_bonus': 0.02,
            'distance_penalty_multiplier': 0.001,
            'velocity_penalty_multiplier': 0.001,
            'smoothness_penalty_multiplier': 0.05,
            'lidar_penalty_multiplier': 0.5,
            'coin_collection_reward': 300.0,
            'success_bonus': 1000.0,
            'collision_penalty': 50.0
        }

    reward = 0.0
    
    # 1. Alive Bonus
    reward += reward_weights['alive_bonus']
    
    # 2. Smart Distance Penalty
    reward -= reward_weights['distance_penalty_multiplier'] * current_distance
    
    # 3. Stability Penalties (Velocity and Smoothness)
    velocity_magnitude = np.linalg.norm(drone_vel)
    reward -= reward_weights['velocity_penalty_multiplier'] * velocity_magnitude
    
    # Agent gets penalized for large action changes to encourage smoother trajectories, not just for high velocities. This promotes more natural and efficient flight patterns.
    reward -= reward_weights.get('smoothness_penalty_multiplier', 0.05) * action_diff
    
    # 4. Proximity Penalty (LiDAR-based Obstacle Avoidance)
    min_lidar = np.min(lidar_data)
    if min_lidar < 0.1:
        reward -= reward_weights['lidar_penalty_multiplier'] * (0.1 - min_lidar)
    
    # 5. THE ULTIMATE PRIZE: Single Coin Collection Reward
    if coin_collected:
        reward += reward_weights['coin_collection_reward']
    
    # 6. Terminal Conditions
    if is_success:
        reward += reward_weights['success_bonus']
    elif is_collision:
        reward -= reward_weights['collision_penalty']
        
    return reward