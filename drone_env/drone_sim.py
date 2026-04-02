import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
import os

from .reward_functions import compute_dense_reward

class RoomDroneEnv(gym.Env):
    def __init__(self, gui=False, num_obstacles=0, randomize_obstacles=False, randomize_coins=False, reward_weights=None):
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
        self.max_steps = 14400
        
        self.num_obstacles = num_obstacles
        self.randomize_obstacles = randomize_obstacles
        self.randomize_coins = randomize_coins
        self.reward_weights = reward_weights
        
        self.drone_id = None
        self.wall_ids = []
        self.obstacle_ids = []
        self.obstacle_positions = []
        self.gold_data = []
        
        self.num_rays = 36
        self.lidar_range = 5.0
        
        self.prev_action = np.zeros(4, dtype=np.float32)

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
            # First coin 1m ahead: Golden Ratio spawn for curriculum force-feeding
            fixed_positions = [
                [1.0, 0.0, 2.0],
                [0.0, 1.5, 2.0],
                [4.0, 4.0, 2.0],
                [-4.0, -4.0, 2.0]
            ]
            for pos in fixed_positions:
                vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.12, rgbaColor=[1, 0.84, 0, 1])
                gid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=pos)
                self.gold_data.append({"id": gid, "pos": pos})
            return

        num_coins = np.random.randint(10, 18)
        attempts = 0
        
        while len(self.gold_data) < num_coins and attempts < 200:
            attempts += 1
            # Z minimum 1.0 to prevent coins near the z<0.3 death zone
            pos = [np.random.uniform(-7.0, 7.0),
                   np.random.uniform(-7.0, 7.0),
                   np.random.uniform(1.0, 6.0)]
            
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
        
        # Symmetry Breaking: self.np_random is seeded by Gymnasium's reset(seed=)
        start_x = self.np_random.uniform(-0.5, 0.5)
        start_y = self.np_random.uniform(-0.5, 0.5)
        start_yaw = self.np_random.uniform(-math.pi, math.pi)
        start_pos = [start_x, start_y, 2.0]
        start_ori = p.getQuaternionFromEuler([0, 0, start_yaw])
        
        urdf_path = os.path.join(os.path.dirname(__file__), "cf2x.urdf")
        self.drone_id = p.loadURDF(urdf_path, start_pos, baseOrientation=start_ori, globalScaling=4.0)
        p.changeDynamics(self.drone_id, -1, mass=1.0)
        
        self._spawn_coins_safely()
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
        if len(self.gold_data) > 0:
            distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"])))
                         for g in self.gold_data]
            closest_idx = np.argmin(distances)
            closest_pos = self.gold_data[closest_idx]["pos"]
            global_relative_pos = np.array([g - d for g, d in zip(closest_pos, drone_pos)])
            local_relative_pos = rot_matrix.T.dot(global_relative_pos)
        else:
            local_relative_pos = np.array([0, 0, 0])
            
        lidar_data = self._compute_lidar(drone_pos, ori)
        
        # PyBullet getEulerFromQuaternion returns (roll, pitch, yaw)
        euler = p.getEulerFromQuaternion(ori)
        current_roll  = euler[0]
        current_pitch = euler[1]
        current_yaw   = euler[2]
        yaw_sin = math.sin(current_yaw)
        yaw_cos = math.cos(current_yaw)
        
        z_altitude = np.array([drone_pos[2]], dtype=np.float32)
        
        obs = np.concatenate([
            z_altitude,                    # 1D
            [current_roll, current_pitch], # 2D
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
        target_thrust   = ((action[3] + 1.0) / 2.0) * 20.0   # 0–20 N total

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
        rotor_offsets = [[0.12, 0.12, 0], [-0.12, 0.12, 0], [-0.12, -0.12, 0], [0.12, -0.12, 0]]
        for i in range(4):
            p.applyExternalForce(self.drone_id, -1,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=rotor_offsets[i],
                                 flags=p.LINK_FRAME)

        # Apply differential yaw torque
        torque_mag = ((forces[0] + forces[2]) - (forces[1] + forces[3])) * 0.01
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
        drone_vel, _ = p.getBaseVelocity(self.drone_id)

        terminated = False
        truncated  = False
        info = {}
        is_success   = False
        is_collision = False
        coin_collected  = False
        current_distance = 0.0

        # Coin Collection Logic
        if len(self.gold_data) > 0:
            distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"])))
                         for g in self.gold_data]
            closest_idx = np.argmin(distances)
            current_distance = distances[closest_idx]
            
            # Expanded collection radius to 0.6m
            if current_distance < 0.6:
                p.removeBody(self.gold_data[closest_idx]["id"])
                self.gold_data.pop(closest_idx)
                coin_collected = True

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

        if self.current_step >= self.max_steps:
            truncated = True

        # Construct Observation
        obs = self._get_obs()
        lidar_data = obs[-36:]
        info['is_success'] = is_success

        # Compute Reward
        reward = compute_dense_reward(
            drone_pos, drone_vel, action, current_distance,
            is_collision, is_success, lidar_data, coin_collected,
            action_diff,
            reward_weights=self.reward_weights
        )

        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)