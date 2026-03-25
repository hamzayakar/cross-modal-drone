import numpy as np

def compute_dense_reward(drone_pos, drone_vel, action, current_distance, is_collision, is_success, lidar_data, coin_collected):
    """
    Computes a comprehensive and balanced reward signal to prevent local optima (suicide policy)
    while encouraging exploration and coin collection.
    """
    reward = 0.0
    
    # 1. Distance Penalty (Significantly reduced to prevent instant suicide)
    # The drone won't bleed points too fast, giving it time to explore.
    reward -= 0.05 * current_distance
    
    # 2. Velocity Penalty (Softened)
    # Penalizes excessive speeds to ensure stable flight, but allows movement.
    velocity_magnitude = np.linalg.norm(drone_vel)
    reward -= 0.01 * velocity_magnitude
    
    # 3. Control Effort Penalty (Softened)
    # Encourages smooth motor usage.
    effort = np.sum(np.square(action))
    reward -= 0.005 * effort
    
    # 4. Proximity Penalty (LiDAR-based Obstacle Avoidance)
    # LiDAR values range from 0.0 to 1.0 (max range 5.0m).
    # If the minimum LiDAR reading is below 0.1, the drone is dangerously close (< 0.5m) to a wall.
    min_lidar = np.min(lidar_data)
    if min_lidar < 0.1:
        # Exponential penalty the closer it gets to the wall
        reward -= 1.0 * (0.1 - min_lidar)
    
    # 5. Alive Bonus (Increased!)
    # Makes hovering and staying alive mathematically better than crashing.
    reward += 0.2
    
    # 6. CRITICAL: Single Coin Collection Reward
    # The "Aha!" moment trigger. The drone gets a massive boost for eating a single coin.
    if coin_collected:
        reward += 100.0
    
    # 7. Terminal Conditions
    if is_success:
        reward += 500.0  # Mission complete bonus
    elif is_collision:
        reward -= 50.0   # Crash penalty (lowered slightly so it isn't terrified of exploring)
        
    return reward