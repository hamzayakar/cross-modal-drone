import numpy as np


def compute_hover_reward(drone_pos, target_pos, drone_vel, body_pitch, body_roll, local_ang_vel, is_collision, reward_weights):
    """
    Stage 0 (Hover) reward. No coins, no navigation.
    Goal: reach and hold the virtual hover target [0, 0, 2.0].

    Design principle: compass in obs always points to target_pos (same structure as
    nav stages where target = nearest coin). Policy learns one skill: minimize the
    compass vector. In hover the target is fixed; in nav it moves as coins are collected.

    Reward = alive_bonus - distance_penalty * dist_3D - tilt_penalty * (pitch²+roll²)
             - angular_velocity_penalty * |ang_vel| - collision_penalty (if any)

    No velocity_penalty: physics drag (-0.5*v) and tilt penalty provide natural
    damping. Explicit velocity penalty would bias the transferred policy toward
    slow movement, hurting coin-collection speed in Stage 1+.
    """
    reward = 0.0

    reward += reward_weights['alive_bonus']

    dist = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
    reward -= reward_weights['distance_penalty'] * dist

    tilt = body_pitch ** 2 + body_roll ** 2
    reward -= reward_weights['tilt_penalty'] * tilt

    ang_vel_magnitude = np.linalg.norm(local_ang_vel)
    reward -= reward_weights['angular_velocity_penalty'] * ang_vel_magnitude

    if is_collision:
        reward -= reward_weights['collision_penalty']

    return reward


def compute_dense_reward(drone_pos, drone_vel, action, current_distance, is_collision, is_success, lidar_data, coin_collected, action_diff, reward_weights=None):
    """
    Stage 1+ (Navigation) reward. Dense shaping for coin-collection flight.
    Weights are loaded from YAML nav_rewards section. Default values here
    are kept in sync with teacher_ppo.yaml to prevent silent reward mismatch
    in tests/notebooks that instantiate the env without passing reward_weights.
    """
    if reward_weights is None:
        reward_weights = {
            'alive_bonus': 0.02,
            'distance_penalty_multiplier': 0.02,
            'velocity_penalty_multiplier': 0.003,
            'smoothness_penalty_multiplier': 0.02,
            'lidar_penalty_multiplier': 0.5,
            'coin_collection_reward': 300.0,
            'success_bonus': 1000.0,
            'collision_penalty': 300.0
        }

    reward = 0.0

    reward += reward_weights['alive_bonus']

    reward -= reward_weights['distance_penalty_multiplier'] * current_distance

    velocity_magnitude = np.linalg.norm(drone_vel)
    reward -= reward_weights['velocity_penalty_multiplier'] * velocity_magnitude

    reward -= reward_weights.get('smoothness_penalty_multiplier', 0.02) * action_diff

    min_lidar = np.min(lidar_data)
    if min_lidar < 0.1:
        reward -= reward_weights['lidar_penalty_multiplier'] * (0.1 - min_lidar)

    if coin_collected:
        reward += reward_weights['coin_collection_reward']

    if is_success:
        reward += reward_weights['success_bonus']
    elif is_collision:
        reward -= reward_weights['collision_penalty']

    return reward
