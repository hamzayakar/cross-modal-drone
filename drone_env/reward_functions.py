import numpy as np


def compute_hover_reward(drone_pos, target_pos, drone_vel, body_pitch, body_roll, local_ang_vel, is_collision, reward_weights):
    """
    Stage 0 (Hover) reward. No coins, no navigation.
    Goal: reach and hold the virtual hover target [0, 0, 2.0].

    Reward = max(0, 2 - dist^4) - tilt_penalty*(pitch²+roll²)
             - angular_velocity_penalty*|ang_vel| - collision_penalty (if any)

    The quartic distance term is always >= 0, so per-step reward is non-negative
    for any state that doesn't trigger a collision. This eliminates the "die fast"
    local optimum: the policy gains nothing by crashing early, and loses the
    accumulated positive reward from staying alive near the target.

    Breakeven distance: dist = 2^(1/4) ≈ 1.19m. Beyond that the distance term
    clamps to 0 (neutral), so even a badly drifting drone has no incentive to
    crash. Staying alive is always >= crashing.

    Tilt and angular velocity penalties remain: they teach stable posture that
    transfers directly to Stage 1+ navigation maneuvers.

    No alive_bonus or linear distance_penalty: these caused "die fast" when the
    drone drifted beyond the breakeven distance (0.2m with old alive_bonus=0.1).
    """
    dist = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
    reward = max(0.0, 2.0 - dist ** 4)

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
