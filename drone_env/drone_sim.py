import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
import os

from .reward_functions import compute_dense_reward, compute_hover_reward

class RoomDroneEnv(gym.Env):
    def __init__(self, gui=False, num_obstacles=0, randomize_obstacles=False, randomize_coins=False,
                 reward_weights=None, hover_only=False, num_fixed_coins=4, fixed_spawn=False,
                 max_steps=10800, coin_count_range=(10, 18), coin_z_range=(1.5, 2.5)):
        super().__init__()

        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Action space: 4-D [Target Pitch, Target Roll, Target Yaw Rate, Target Thrust]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: 50-D Ego-Centric State
        # 1 (Z-Altitude) + 2 (Roll, Pitch) + 2 (Sin/Cos Yaw) + 3 (Local Vel) +
        # 3 (Local Ang Vel) + 3 (Local Relative Pos) + 36 (LiDAR) = 50D
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)

        self.room_bounds = [8.0, 8.0, 4.0]
        self.max_steps = max_steps          # per-stage episode length, set via YAML
        self.coin_count_range = coin_count_range  # (min, max) coins for randomize_coins=True
        self.coin_z_range = coin_z_range    # (z_min, z_max) for random coin Z placement

        self.num_obstacles = num_obstacles
        self.randomize_obstacles = randomize_obstacles
        self.randomize_coins = randomize_coins
        self.reward_weights = reward_weights
        self.hover_only = hover_only
        self.num_fixed_coins = num_fixed_coins
        self.fixed_spawn = fixed_spawn
        # Virtual hover target: used as compass anchor + reward target in Stage 0.
        # Ego-centric compass always points here; policy learns "minimize this vector."
        # Same policy structure as nav stages (where target = nearest coin).
        if hover_only and reward_weights and 'hover_target' in reward_weights:
            self.hover_target = list(reward_weights['hover_target'])
        else:
            self.hover_target = [0.0, 0.0, 2.0]
        
        self.drone_id = None
        self.wall_ids = []
        self.obstacle_ids = []
        self.obstacle_positions = []
        self.gold_data = []
        
        self.num_rays = 36
        self.lidar_range = 5.0

        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_coin_distance = 0.0  # for progress reward: distance to nearest coin last step

    def _build_closed_room(self):
        """Builds the 16x16m room with glass walls."""
        wall_half_thickness = 0.1
        h_x, h_y, h_z = self.room_bounds
        
        wall_configs = [
            ([0, h_y, h_z], [h_x, wall_half_thickness, h_z]),
            ([0, -h_y, h_z], [h_x, wall_half_thickness, h_z]),
            ([-h_x, 0, h_z], [wall_half_thickness, h_y, h_z]),
            ([h_x, 0, h_z], [wall_half_thickness, h_y, h_z]),
            ([0, 0, h_z * 2], [h_x, h_y, wall_half_thickness])
        ]
        
        glass_color = [0.8, 0.9, 1.0, 0.25]
        self.wall_ids = []
        
        for pos, extents in wall_configs:
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=glass_color)
            w_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id,
                                     baseVisualShapeIndex=vis_id, basePosition=pos)
            self.wall_ids.append(w_id)

    def _spawn_obstacles(self):
        """Spawns procedural obstacles (Stage 2-4)."""
        self.obstacle_ids = []
        self.obstacle_positions = []
        
        if self.num_obstacles == 0:
            return
            
        rng = np.random.default_rng(seed=42) if not self.randomize_obstacles else np.random.default_rng()
            
        for _ in range(self.num_obstacles):
            ox = rng.uniform(-7.0, 7.0)
            oy = rng.uniform(-7.0, 7.0)
            
            if math.sqrt(ox**2 + oy**2) < 1.0:
                continue
                
            oz_half = rng.uniform(1.0, 3.0)
            is_cylinder = rng.choice([True, False])
            
            if is_cylinder:
                radius = rng.uniform(0.2, 0.6)
                col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=oz_half*2)
                vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=oz_half*2,
                                             rgbaColor=[0.4, 0.4, 0.5, 1])
                safe_radius = radius + 0.2
            else:
                ext_x = rng.uniform(0.2, 0.6)
                ext_y = rng.uniform(0.2, 0.6)
                col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ext_x, ext_y, oz_half])
                vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[ext_x, ext_y, oz_half],
                                             rgbaColor=[0.5, 0.4, 0.4, 1])
                safe_radius = max(ext_x, ext_y) + 0.2
                
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id,
                                       baseVisualShapeIndex=vis_id, basePosition=[ox, oy, oz_half])
            self.obstacle_ids.append(obs_id)
            self.obstacle_positions.append({"pos": [ox, oy], "safe_radius": safe_radius})

    def _spawn_coins_safely(self):
        """Spawns coins. Stage 0 uses fixed positions, others use random safe positions."""
        self.gold_data = []
        
        if not self.randomize_coins:
            # First coin 1m from spawn (World [1,0,2]).
            # NOTE: Drone is Y-forward. At Yaw=0 this coin is to the drone's RIGHT (+X),
            # not ahead (+Y). Force-feeding benefit is proximity (~1m away), not direction.
            # The ego-centric compass vector guides the agent regardless of heading.
            # Stage 2 coin geometry redesign (v4):
            # Old positions: coins 3-4 at [4,4] and [-4,-4] (opposite corners, 11.3m apart).
            # Total path ~20m @ 0.5m/s = 40s → exceeded 30s episode budget.
            # New positions: form a clockwise ring, each ~3m from center, ~4m between consecutive.
            # Total path ~13m @ 0.5m/s = 26s → fits 30s with margin at 7200 steps.
            # Graduated difficulty: 1m → 2m → 3m → 3m, each in a different quadrant to
            # force real heading changes. No two consecutive coins are opposite each other.
            fixed_positions = [
                [ 1.0,  0.0, 2.0],   # coin 1: 1m, easy entry
                [ 0.0,  2.0, 2.0],   # coin 2: 2m, 90° heading change
                [-2.5,  1.5, 2.0],   # coin 3: ~2.9m, 135° heading change
                [-1.5, -2.5, 2.0],   # coin 4: ~2.9m, 180° heading change
            ]
            for pos in fixed_positions[:self.num_fixed_coins]:
                vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.12, rgbaColor=[1, 0.84, 0, 1])
                gid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=pos)
                self.gold_data.append({"id": gid, "pos": pos})
            return

        num_coins = np.random.randint(self.coin_count_range[0], self.coin_count_range[1])
        attempts = 0

        while len(self.gold_data) < num_coins and attempts < 200:
            attempts += 1
            pos = [np.random.uniform(-7.0, 7.0),
                   np.random.uniform(-7.0, 7.0),
                   np.random.uniform(self.coin_z_range[0], self.coin_z_range[1])]
            
            if math.sqrt(pos[0]**2 + pos[1]**2 + (pos[2]-1.0)**2) < 1.0:
                continue
                
            hit_obstacle = False
            for obs in self.obstacle_positions:
                dist_to_obs = math.sqrt((pos[0] - obs["pos"][0])**2 + (pos[1] - obs["pos"][1])**2)
                if dist_to_obs < obs["safe_radius"]:
                    hit_obstacle = True
                    break
                    
            if hit_obstacle:
                continue
                
            vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.12, rgbaColor=[1, 0.84, 0, 1])
            gid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=pos)
            self.gold_data.append({"id": gid, "pos": pos})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        # Explicit timestep — never depend on PyBullet version defaults
        p.setTimeStep(1.0 / 240.0)
        self.current_step = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self._build_closed_room()
        self._spawn_obstacles()
        
        # Hover stage has no coins. For navigation stages, spawn coins first
        # so drone spawn position can be validated against them.
        if not self.hover_only:
            self._spawn_coins_safely()

        start_yaw = self.np_random.uniform(-math.pi, math.pi)
        if self.fixed_spawn:
            # Stage 0 (pure hover): spawn near room center with ±0.25m XY offset.
            # Small offset ensures compass is always non-zero at spawn, giving the
            # policy a recovery signal from step 1. Spawning exactly at the target
            # (Stage 0.27) caused a bimodal failure mode: compass=[0,0,0] → policy
            # output a memorized "nominal hover" action that worked for some physics
            # worker states but not others, with no gradient to fix it.
            # ±0.25m keeps the drone within the reward's positive zone (breakeven
            # sqrt(2)≈1.41m) and forces the policy to learn "navigate to hover point
            # and hold" rather than "stay where you spawned."
            start_x = self.np_random.uniform(-0.25, 0.25)
            start_y = self.np_random.uniform(-0.25, 0.25)
        else:
            # Symmetry Breaking: self.np_random is seeded by Gymnasium's reset(seed=)
            # FIX: Resample drone spawn if it lands inside any coin's collection radius.
            # Monte Carlo analysis showed ~4.5% of spawns place the drone within 0.6m
            # of coin 1 at [1.0, 0.0, 2.0], causing an instant free collection on step 1
            # with no learned behavior — a noisy false-positive reward signal.
            for _ in range(10):
                start_x = self.np_random.uniform(-0.5, 0.5)
                start_y = self.np_random.uniform(-0.5, 0.5)
                too_close = any(
                    math.sqrt((start_x - g["pos"][0])**2 + (start_y - g["pos"][1])**2) < 0.6
                    for g in self.gold_data
                )
                if not too_close:
                    break

        start_pos = [start_x, start_y, 2.0]
        start_ori = p.getQuaternionFromEuler([0, 0, start_yaw])

        urdf_path = os.path.join(os.path.dirname(__file__), "cf2x.urdf")
        self.drone_id = p.loadURDF(urdf_path, start_pos, baseOrientation=start_ori, globalScaling=4.0)
        p.changeDynamics(self.drone_id, -1, mass=1.0)

        # Initialise prev_coin_distance so first step doesn't spike a negative progress reward.
        if not self.hover_only and self.gold_data:
            self.prev_coin_distance = min(
                math.sqrt(sum((start_pos[i] - g["pos"][i])**2 for i in range(3)))
                for g in self.gold_data
            )
        else:
            self.prev_coin_distance = 0.0

        return self._get_obs(), {}

    def _compute_lidar(self, drone_pos, ori):
        """
        Casts 36 rays in a gimbal-stabilized horizontal plane.
        Rays rotate only with Yaw — not Pitch/Roll — to prevent projection shrinkage.
        This matches real-world 2D LiDAR behavior.
        """
        ray_starts = []
        ray_ends = []
        offset = 0.25

        euler = p.getEulerFromQuaternion(ori)
        yaw = euler[2]

        for i in range(self.num_rays):
            angle = (2 * math.pi * i) / self.num_rays + yaw
            dx = math.cos(angle)
            dy = math.sin(angle)

            start = [drone_pos[0] + dx*offset, drone_pos[1] + dy*offset, drone_pos[2]]
            end = [start[0] + dx*self.lidar_range, start[1] + dy*self.lidar_range, start[2]]

            ray_starts.append(start)
            ray_ends.append(end)

        results = p.rayTestBatch(ray_starts, ray_ends)
        return np.array([res[2] for res in results], dtype=np.float32)

    def _get_obs(self):
        drone_pos, ori = p.getBasePositionAndOrientation(self.drone_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.drone_id)
        
        # World -> Body frame rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        
        # Velocities in Body Frame
        local_vel = rot_matrix.T.dot(linear_vel)
        local_ang_vel = rot_matrix.T.dot(angular_vel)
        
        # Compass: target relative position in Body Frame
        if self.hover_only:
            # Compass points to virtual hover target [0, 0, 2.0] in body frame.
            # Policy sees "target is X meters left, Y meters ahead, Z meters up" —
            # same ego-centric structure as nav stages where target = nearest coin.
            global_relative_pos = np.array(self.hover_target) - np.array(drone_pos)
            local_relative_pos = rot_matrix.T.dot(global_relative_pos)
        elif len(self.gold_data) > 0:
            distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"])))
                         for g in self.gold_data]
            closest_idx = np.argmin(distances)
            closest_pos = self.gold_data[closest_idx]["pos"]
            global_relative_pos = np.array([g - d for g, d in zip(closest_pos, drone_pos)])
            local_relative_pos = rot_matrix.T.dot(global_relative_pos)
        else:
            local_relative_pos = np.array([0, 0, 0])
            
        lidar_data = self._compute_lidar(drone_pos, ori)
        
        euler = p.getEulerFromQuaternion(ori)
        world_pitch_raw = euler[0]
        world_roll_raw  = euler[1]
        current_yaw     = euler[2]

        # FIX (Stage 0.19): Convert World Euler to Body Euler for observation.
        # Step() corrects PD controller to Body Frame (Stage 0.17), but _get_obs()
        # was still sending raw World euler. With random yaw (Stage 0.14), at Yaw=90°
        # world pitch ≈ body roll and vice-versa. The agent was seeing axes swap randomly
        # every episode, making pitch/roll correlation impossible to learn.
        # Mirror the exact same 2D rotation used in step() + sign convention.
        body_pitch_raw = world_pitch_raw * math.cos(current_yaw) + world_roll_raw * math.sin(current_yaw)
        body_roll_raw  = -world_pitch_raw * math.sin(current_yaw) + world_roll_raw * math.cos(current_yaw)
        current_pitch = -body_pitch_raw  # Nose DOWN = positive, matches PD convention
        current_roll  =  body_roll_raw   # Roll RIGHT = positive, matches PD convention

        yaw_sin = math.sin(current_yaw)
        yaw_cos = math.cos(current_yaw)

        z_altitude = np.array([drone_pos[2]], dtype=np.float32)

        obs = np.concatenate([
            z_altitude,                    # 1D
            [current_roll, current_pitch], # 2D — Body-Frame roll & pitch (nose-down positive)
            [yaw_sin, yaw_cos],            # 2D — continuous yaw
            local_vel,                     # 3D
            local_ang_vel,                 # 3D
            local_relative_pos,            # 3D
            lidar_data                     # 36D
        ]).astype(np.float32)
        
        noise = np.random.normal(loc=0.0, scale=0.01, size=obs.shape)
        return (obs + noise).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Calculate action smoothness penalty (jerk)
        action_diff = np.mean(np.square(action - self.prev_action))
        self.prev_action = action.copy()
        
        # ==============================================================
        # PD CONTROLLER — ATTITUDE CONTROL MODE, BODY FRAME STABILIZED
        # ==============================================================
        # Map normalized actions to physical targets
        # action[0]: Pitch (Nose DOWN is positive)
        # action[1]: Roll (Right roll is positive)
        # action[2]: Yaw Rate (CCW is positive)
        target_pitch    = action[0] * (math.pi / 6)          # max ±30 deg
        target_roll     = action[1] * (math.pi / 6)          # max ±30 deg
        target_yaw_rate = action[2] * 2.0                    # max ±2 rad/s
        target_thrust   = 9.81 * (1.0 + action[3])             # action=0→9.81N (hover), action=±1→[0, 19.62]N

        # Fetch physical state from PyBullet
        drone_pos_pre, ori = p.getBasePositionAndOrientation(self.drone_id)
        euler = p.getEulerFromQuaternion(ori)
        _, ang_vel = p.getBaseVelocity(self.drone_id)

        # Transform world angular velocity to body frame angular velocity
        rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        local_ang_vel = rot_matrix.T.dot(ang_vel)

        # 1. RAW WORLD EULER ANGLES
        # PyBullet standard: euler[0] is rotation around World X. euler[1] is around World Y.
        world_pitch_raw = euler[0]
        world_roll_raw  = euler[1]
        current_yaw     = euler[2]

        # 2. TRANSFORM WORLD TILT TO BODY TILT (Crucial to prevent gimbal lock / blindness)
        body_pitch_raw = world_pitch_raw * math.cos(current_yaw) + world_roll_raw * math.sin(current_yaw)
        body_roll_raw  = -world_pitch_raw * math.sin(current_yaw) + world_roll_raw * math.cos(current_yaw)

        # 3. ALIGN SENSORS TO STANDARD FLIGHT CONTROLLER LOGIC
        # By PyBullet's right-hand rule, Body Pitch Raw (rot around X) positive means NOSE UP.
        # We define positive pitch as NOSE DOWN to match our motor matrix logic.
        current_pitch = -body_pitch_raw

        # Body Roll Raw (rot around Y) positive means ROLL RIGHT. This already matches our logic.
        current_roll = body_roll_raw

        # local_ang_vel[0] is pitch rate (Nose UP positive). We invert it to match NOSE DOWN positive.
        pitch_rate = -local_ang_vel[0]
        
        # local_ang_vel[1] is roll rate (Roll RIGHT positive). Matches.
        roll_rate  = local_ang_vel[1]
        
        # local_ang_vel[2] is yaw rate (CCW positive). Matches.
        current_yaw_rate = local_ang_vel[2]

        # 4. COMPUTE ERRORS
        pitch_error = target_pitch - current_pitch
        roll_error  = target_roll  - current_roll
        yaw_rate_error = target_yaw_rate - current_yaw_rate

        # 5. COMPUTE TORQUES (Proportional - Derivative Control)
        Kp_ang = 5.0
        Kd_ang = 3.0
        Kp_yaw = 2.0

        # Now that sensors and damping rates perfectly align, PD logic is safe.
        tau_pitch = (Kp_ang * pitch_error) - (Kd_ang * pitch_rate)
        tau_roll  = (Kp_ang * roll_error)  - (Kd_ang * roll_rate)
        tau_yaw   = Kp_yaw * yaw_rate_error

        base_f = target_thrust / 4.0

        # 6. MOTOR MIXING MATRIX (Strictly matched to URDF layout)
        # Motor 0: Front-Right (+X, +Y) -> CCW
        # Motor 1: Front-Left (-X, +Y) -> CW
        # Motor 2: Rear-Left (-X, -Y) -> CCW
        # Motor 3: Rear-Right (+X, -Y) -> CW
        f0 = base_f - tau_pitch - tau_roll - tau_yaw
        f1 = base_f - tau_pitch + tau_roll + tau_yaw
        f2 = base_f + tau_pitch + tau_roll - tau_yaw
        f3 = base_f + tau_pitch - tau_roll + tau_yaw

        forces = np.clip([f0, f1, f2, f3], 0.0, 7.5)
        # ==============================================================

        # Apply individual rotor forces in Body Frame (LINK_FRAME)
        # URDF prop offsets are ±0.028m × globalScaling=4.0 → ±0.112m
        rotor_offsets = [[0.112, 0.112, 0], [-0.112, 0.112, 0], [-0.112, -0.112, 0], [0.112, -0.112, 0]]
        for i in range(4):
            p.applyExternalForce(self.drone_id, -1,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=rotor_offsets[i],
                                 flags=p.LINK_FRAME)

        # Apply differential yaw torque
        # FIX (Stage 0.19): Sign was inverted. Rotor forces [0,0,F] at [x,y,0] in LINK_FRAME
        # produce only X/Y torque (r×F = [y·F, -x·F, 0]), so Z torque = 0 from forces alone.
        # torque_mag is the ONLY yaw mechanism. Physics: CCW rotors (0,2) create CW (-Z)
        # reaction; CW rotors (1,3) create CCW (+Z) reaction.
        # Correct formula: ((F1+F3) - (F0+F2)) * k
        torque_mag = ((forces[1] + forces[3]) - (forces[0] + forces[2])) * 0.01
        p.applyExternalTorque(self.drone_id, -1,
                              torqueObj=[0, 0, torque_mag],
                              flags=p.LINK_FRAME)

        # Apply Aerodynamic Drag
        # posObj must be the drone's actual world position so PyBullet applies the force
        # at the COM with zero lever arm. Using [0,0,0] with WORLD_FRAME would place the
        # application point at the world origin, creating a phantom torque proportional to
        # the drone's distance from origin and growing with altitude/speed.
        drone_vel_pre, angular_vel_pre = p.getBaseVelocity(self.drone_id)
        drag_force  = [v * -0.5  for v in drone_vel_pre]
        drag_torque = [w * -0.05 for w in angular_vel_pre]
        p.applyExternalForce(self.drone_id, -1,
                             forceObj=drag_force, posObj=list(drone_pos_pre),
                             flags=p.WORLD_FRAME)
        p.applyExternalTorque(self.drone_id, -1,
                              torqueObj=drag_torque,
                              flags=p.WORLD_FRAME)

        p.stepSimulation()

        # Re-fetch state AFTER physics step for accurate reward computation
        drone_pos, ori = p.getBasePositionAndOrientation(self.drone_id)
        drone_vel, _   = p.getBaseVelocity(self.drone_id)

        # Body-frame velocity and compass — used by nav reward (yaw alignment,
        # velocity direction). Compute once here rather than inside reward function.
        rot_post = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        local_vel_post = rot_post.T.dot(drone_vel)

        terminated = False
        truncated  = False
        info = {}
        is_success   = False
        is_collision = False
        coin_collected  = False
        current_distance = 0.0

        # Coin Collection Logic (navigation stages only)
        coin_progress = 0.0
        if not self.hover_only:
            if len(self.gold_data) > 0:
                distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"])))
                             for g in self.gold_data]
                closest_idx = np.argmin(distances)
                current_distance = distances[closest_idx]

                if current_distance < 0.6:
                    p.removeBody(self.gold_data[closest_idx]["id"])
                    self.gold_data.pop(closest_idx)
                    coin_collected = True
                    # Reset prev_distance to next coin; don't count snap as progress.
                    if self.gold_data:
                        next_distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"])))
                                          for g in self.gold_data]
                        self.prev_coin_distance = min(next_distances)
                    else:
                        self.prev_coin_distance = 0.0
                    coin_progress = 0.0
                else:
                    # Progress = distance closed toward nearest coin since last step.
                    coin_progress = self.prev_coin_distance - current_distance
                    self.prev_coin_distance = current_distance

            if len(self.gold_data) == 0:
                is_success = True
                terminated = True

        # Death Checks
        hx, hy, hz = self.room_bounds
        euler = p.getEulerFromQuaternion(ori)

        # Unrecoverable tilt death check
        if abs(euler[0]) > 1.3 or abs(euler[1]) > 1.3:
            is_collision = True
            terminated   = True

        # Room boundary death check
        if (abs(drone_pos[0]) > hx - 0.2 or
            abs(drone_pos[1]) > hy - 0.2 or
            drone_pos[2] < 0.3 or
            drone_pos[2] > hz * 2 - 0.2):
            is_collision = True
            terminated   = True

        # Obstacle/Wall collision check
        for entity_id in self.obstacle_ids + self.wall_ids:
            if p.getContactPoints(bodyA=self.drone_id, bodyB=entity_id):
                is_collision = True
                terminated   = True
                break

        # Hover-only: terminate if drone drifts too far from hover target.
        # New reward max(0, 2-4·dist²) zeros at 0.71m. Terminating at 1.0m gives
        # a small buffer beyond the zero-reward zone and triggers a fast reset when
        # the episode is clearly failing. No collision penalty — clean reset.
        if self.hover_only and not terminated:
            hover_dist = math.sqrt(sum((drone_pos[i] - self.hover_target[i])**2 for i in range(3)))
            if hover_dist > 1.0:
                terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        # Construct Observation
        obs = self._get_obs()
        info['is_success'] = is_success

        # Compute Reward
        if self.hover_only:
            reward = compute_hover_reward(
                drone_pos, self.hover_target, drone_vel, current_pitch, current_roll,
                local_ang_vel, action_diff, is_collision, self.reward_weights
            )
        else:
            # FIX (Stage 0.20): Use clean (pre-noise) LiDAR for reward computation.
            # _get_obs() injects Gaussian noise into the full observation vector.
            # Extracting LiDAR from that noisy array passes hallucinated proximity
            # readings to the reward judge. Ground truth LiDAR is recomputed here.
            clean_lidar = self._compute_lidar(drone_pos, ori)

            # Body-frame compass for yaw alignment and velocity direction rewards.
            # Reuse rot_post already computed from post-physics ori.
            if self.gold_data:
                distances_r = [math.sqrt(sum((d - v)**2 for d, v in zip(drone_pos, g["pos"])))
                                for g in self.gold_data]
                nearest_pos = self.gold_data[int(np.argmin(distances_r))]["pos"]
                global_rel  = np.array([g - d for g, d in zip(nearest_pos, drone_pos)])
                local_rel_post = rot_post.T.dot(global_rel)
            else:
                local_rel_post = np.zeros(3)

            reward = compute_dense_reward(
                drone_pos, drone_vel, action, current_distance,
                is_collision, is_success, clean_lidar, coin_collected,
                action_diff, coin_progress,
                local_vel=local_vel_post,
                local_relative_pos=local_rel_post,
                reward_weights=self.reward_weights
            )

        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)