import numpy as np


def compute_hover_reward(drone_pos, target_pos, drone_vel, body_pitch, body_roll, local_ang_vel, is_collision, reward_weights):
    """
    Stage 0 (Hover) reward. No coins, no navigation.
    Goal: reach and hold the virtual hover target [0, 0, 2.0].

    Reward = max(0, 2 - dist^2) - tilt_penalty*(pitch²+roll²)
             - angular_velocity_penalty*|ang_vel|
             - velocity_penalty*sqrt(vx²+vy²)
             - collision_penalty (if any)

    Stage 0.29: added lateral velocity penalty.
    v3 (tilt_penalty=0.15) revealed that without a velocity penalty the policy
    learns to slowly drift rather than hold position — lateral motion carries no
    cost, so the optimal policy accepts perpetual drift at ~0.5m instead of
    converging to center. velocity_penalty directly penalises any horizontal
    movement, making "close AND still" strictly better than "close AND moving".
    This eliminates both the slow-drift failure mode and any circular-orbit
    strategy: both involve non-zero lateral velocity and are now penalised.
    tilt_penalty raised 0.15 → 0.27 to restore moderate damping without
    recreating the v2 overdamped 0.28m ceiling.
    """
    dist = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
    reward = max(0.0, 2.0 - dist ** 2)

    tilt = body_pitch ** 2 + body_roll ** 2
    reward -= reward_weights['tilt_penalty'] * tilt

    ang_vel_magnitude = np.linalg.norm(local_ang_vel)
    reward -= reward_weights['angular_velocity_penalty'] * ang_vel_magnitude

    lateral_vel = np.sqrt(drone_vel[0] ** 2 + drone_vel[1] ** 2)
    reward -= reward_weights.get('velocity_penalty', 0.0) * lateral_vel

    if is_collision:
        reward -= reward_weights['collision_penalty']

    return reward


def compute_dense_reward(drone_pos, drone_vel, action, current_distance, is_collision, is_success, lidar_data, coin_collected, action_diff, coin_progress=0.0, reward_weights=None):
    """
    Stage 1+ (Navigation) reward. Dense shaping for coin-collection flight.
    Weights are loaded from YAML nav_rewards section. Default values here
    are kept in sync with teacher_ppo.yaml to prevent silent reward mismatch
    in tests/notebooks that instantiate the env without passing reward_weights.

    Stage 2 redesign: replaced alive_bonus + distance_penalty with a progress
    reward (distance closed toward nearest coin per step). The old structure
    made per-step reward negative at distances > 1m (alive=0.02, penalty=0.02*d),
    causing the Suicide Policy / alive-bonus farming collapse at Stage 2.
    Progress reward is distance-agnostic: same shaped signal whether the coin
    is 1m or 5.66m away. Literature: Kaufmann et al. 2023 (Swift, Nature).
    """
    if reward_weights is None:
        reward_weights = {
            'progress_reward_weight': 50.0,
            'velocity_penalty_multiplier': 0.0,
            'smoothness_penalty_multiplier': 0.02,
            'lidar_penalty_multiplier': 0.5,
            'coin_collection_reward': 300.0,
            'success_bonus': 1000.0,
            'collision_penalty': 300.0
        }

    reward = 0.0

    # Progress reward: metres closed toward nearest coin × weight.
    # Positive when approaching, negative when retreating, zero when hovering.
    # Dying is always dominated: -300 collision < 0 progress any finite path.
    reward += reward_weights.get('progress_reward_weight', 50.0) * coin_progress

    velocity_magnitude = np.linalg.norm(drone_vel)
    reward -= reward_weights.get('velocity_penalty_multiplier', 0.0) * velocity_magnitude

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
