import numpy as np


def compute_hover_reward(drone_pos, target_pos, drone_vel, body_pitch, body_roll,
                         local_ang_vel, action_diff, is_collision, reward_weights):
    """
    Stage 0 (Hover) reward.

    Reward = max(0, 2 - 4·dist²)
           - tilt_penalty*(pitch²+roll²)
           - angular_velocity_penalty*|ω|
           - smoothness_penalty*mean(Δa²)
           - collision_penalty [if crash]

    v5 redesign:
    - 4·dist² instead of dist²: 4× stronger gradient near origin, zeros at 0.71m
      (old dist² had near-zero gradient at 0.1m, policy settled at 0.4-0.5m sweet spot)
    - smoothness_penalty replaces velocity_penalty: penalises jerky action changes,
      NOT speed magnitude. velocity_penalty baked "near target = slow" into the
      value function over 10M steps, which transferred as a deceleration prior to
      all navigation stages. smoothness has no such semantic — it only penalises
      abrupt control changes, which is equally valid in navigation.
    """
    dist = np.linalg.norm(np.array(drone_pos) - np.array(target_pos))
    reward = max(0.0, 2.0 - 4.0 * dist ** 2)

    tilt = body_pitch ** 2 + body_roll ** 2
    reward -= reward_weights['tilt_penalty'] * tilt

    ang_vel_magnitude = np.linalg.norm(local_ang_vel)
    reward -= reward_weights['angular_velocity_penalty'] * ang_vel_magnitude

    reward -= reward_weights.get('smoothness_penalty', 0.05) * action_diff

    if is_collision:
        reward -= reward_weights['collision_penalty']

    return reward


def compute_dense_reward(drone_pos, drone_vel, action, current_distance, is_collision,
                         is_success, lidar_data, coin_collected, action_diff,
                         coin_progress=0.0, local_vel=None, local_relative_pos=None,
                         reward_weights=None):
    """
    Stage 1+ (Navigation) reward.

    v5 additions (all distance-gated to avoid long-range interference):
    - approach_bonus: multiplies progress reward near coin (dist < approach_bonus_dist).
      Counters the hover-stage "slow near target" prior that transfers through
      curriculum weights — gives 3× signal in the exact zone where deceleration occurs.
    - yaw_alignment: rewards drone for facing the coin (dist < yaw_alignment_dist).
      cos(θ) = local_relative_pos[1] / dist (body +Y is forward).
      Prepares teacher for CNN distillation: ~60-80% of near-target frames have
      coin off-camera when the teacher never faces its target.
    - velocity_direction: rewards moving TOWARD the coin regardless of yaw.
      dot(v̂, û_target) in body frame — trajectory constraint, not heading constraint.
      Addresses squiggly paths from greedy progress reward without breaking
      quadrotor omnidirectionality.
    """
    if reward_weights is None:
        reward_weights = {
            'progress_reward_weight': 50.0,
            'approach_bonus_weight': 150.0,
            'approach_bonus_dist': 1.5,
            'yaw_alignment_weight': 0.15,
            'yaw_alignment_dist': 2.5,
            'velocity_direction_weight': 0.20,
            'smoothness_penalty_multiplier': 0.03,
            'lidar_penalty_multiplier': 0.5,
            'coin_collection_reward': 300.0,
            'success_bonus': 1000.0,
            'collision_penalty': 300.0,
        }

    reward = 0.0

    # ── Progress reward ──────────────────────────────────────────────────────
    progress_weight = reward_weights.get('progress_reward_weight', 50.0)
    reward += progress_weight * coin_progress

    # ── Approach bonus (hover-transfer fix) ─────────────────────────────────
    # Activates only in the final approach zone where the Stage 0 deceleration
    # prior is strongest. Multiplies the progress signal to overpower it.
    approach_bonus_weight = reward_weights.get('approach_bonus_weight', 0.0)
    approach_bonus_dist   = reward_weights.get('approach_bonus_dist', 1.5)
    if approach_bonus_weight > 0 and current_distance < approach_bonus_dist:
        reward += approach_bonus_weight * coin_progress

    # ── Yaw alignment (CNN distillation readiness) ───────────────────────────
    # cos(θ) between body forward (+Y) and compass direction.
    # local_relative_pos[1] / dist = cos(θ_error).
    # Only active near coin — no heading pressure during long-range transit.
    yaw_weight = reward_weights.get('yaw_alignment_weight', 0.0)
    yaw_dist   = reward_weights.get('yaw_alignment_dist', 2.5)
    if yaw_weight > 0 and local_relative_pos is not None and current_distance < yaw_dist and current_distance > 0.05:
        cos_yaw = local_relative_pos[1] / current_distance
        reward += yaw_weight * cos_yaw

    # ── Velocity direction alignment (trajectory straightness) ───────────────
    # Rewards moving toward the coin, independent of heading.
    # dot(v̂, û_target) in body frame.
    vel_dir_weight = reward_weights.get('velocity_direction_weight', 0.0)
    if vel_dir_weight > 0 and local_vel is not None and local_relative_pos is not None:
        v_norm = np.linalg.norm(local_vel)
        c_norm = current_distance
        if v_norm > 0.05 and c_norm > 0.05:
            v_hat = local_vel / v_norm
            c_hat = local_relative_pos / c_norm
            reward += vel_dir_weight * float(np.dot(v_hat, c_hat))

    # ── Smoothness penalty ───────────────────────────────────────────────────
    reward -= reward_weights.get('smoothness_penalty_multiplier', 0.02) * action_diff

    # ── LiDAR proximity penalty ──────────────────────────────────────────────
    min_lidar = np.min(lidar_data)
    if min_lidar < 0.1:
        reward -= reward_weights['lidar_penalty_multiplier'] * (0.1 - min_lidar)

    # ── Coin / success / collision ───────────────────────────────────────────
    if coin_collected:
        reward += reward_weights['coin_collection_reward']

    if is_success:
        reward += reward_weights['success_bonus']
    elif is_collision:
        reward -= reward_weights['collision_penalty']

    return reward
