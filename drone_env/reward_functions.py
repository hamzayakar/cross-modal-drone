import numpy as np

def compute_dense_reward(drone_pos, drone_vel, action, current_distance, is_collision, is_success, lidar_data, coin_collected):
    """
    Hunter Model (Reward Shaping for Goal-Oriented Behavior):
    Prioritizes reaching the target (coin) over merely surviving.
    Prevents 'ceiling hugging' and 'suicide' by balancing survival incentives with target proximity.
    """
    reward = 0.0
    
    # 1. Alive Bonus (Significantly Reduced)
    # Provides just enough incentive to stay in the air, acting as a baseline.
    # It is kept low so the agent doesn't accumulate high scores just by hovering aimlessly.
    reward += 0.02
    
    # 2. Smart Distance Penalty (The "Gravity" towards the Target)
    # At maximum room distance (~20m), the penalty is -0.02.
    # This perfectly cancels out the Alive Bonus, making distant hovering yield zero net reward.
    # As the drone approaches the target, the penalty decreases, creating a positive reward gradient.
    reward -= 0.001 * current_distance
    
    # 3. Stability Penalties (Velocity and Control Effort)
    # Penalizes erratic movements and excessive motor usage to encourage smooth flight.
    velocity_magnitude = np.linalg.norm(drone_vel)
    reward -= 0.001 * velocity_magnitude
    
    effort = np.sum(np.square(action))
    reward -= 0.001 * effort
    
    # 4. Proximity Penalty (LiDAR-based Obstacle Avoidance)
    # Applies a penalty if the drone gets dangerously close (< 0.1 normalized distance) to walls/obstacles.
    min_lidar = np.min(lidar_data)
    if min_lidar < 0.1:
        reward -= 0.5 * (0.1 - min_lidar)
    
    # 5. THE ULTIMATE PRIZE: Single Coin Collection Reward
    # A massive reward spike for successfully navigating to and consuming a coin.
    # This solves the 'reward overshadowing' problem by making the goal highly profitable.
    if coin_collected:
        reward += 300.0
    
    # 6. Terminal Conditions
    if is_success:
        reward += 1000.0  # Mission complete bonus for clearing the entire room.
    elif is_collision:
        reward -= 50.0  # Severe penalty for crashing (floor, walls, ceiling, or obstacles).
        
    return reward