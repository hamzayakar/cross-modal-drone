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
        
        # Observation space: 50-D Ego-Centric State (Cross-Modal Distillation Ready)
        # 1 (Z-Altitude) + 2 (Roll, Pitch) + 2 (Sin/Cos Yaw) + 3 (Local Vel) + 3 (Local Ang Vel) + 3 (Local Relative Pos) + 36 (LiDAR) = 50D
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
        
        # 36 rays for 10-degree resolution (Prevents Modality Mismatch with CNN FOV)
        self.num_rays = 36
        self.lidar_range = 5.0 
        
        # Holds the previous action to calculate the smoothness (jerk) penalty
        self.prev_action = np.zeros(4, dtype=np.float32)

    def _build_closed_room(self):
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
            w_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=pos)
            self.wall_ids.append(w_id)

    def _spawn_obstacles(self):
        self.obstacle_ids = []
        self.obstacle_positions = []
        
        if self.num_obstacles == 0:
            return
            
        rng = np.random.default_rng(seed=42) if not self.randomize_obstacles else np.random.default_rng()
            
        for _ in range(self.num_obstacles):
            ox = rng.uniform(-7.0, 7.0)
            oy = rng.uniform(-7.0, 7.0)
            
            # Keep a clear spawn zone for the drone to prevent instant collisions
            if math.sqrt(ox**2 + oy**2) < 1.0:
                continue
                
            oz_half = rng.uniform(1.0, 3.0) 
            is_cylinder = rng.choice([True, False])
            
            if is_cylinder:
                radius = rng.uniform(0.2, 0.6)
                col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=oz_half*2)
                vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=oz_half*2, rgbaColor=[0.4, 0.4, 0.5, 1])
                safe_radius = radius + 0.2
            else:
                ext_x = rng.uniform(0.2, 0.6)
                ext_y = rng.uniform(0.2, 0.6)
                col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ext_x, ext_y, oz_half])
                vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[ext_x, ext_y, oz_half], rgbaColor=[0.5, 0.4, 0.4, 1])
                safe_radius = max(ext_x, ext_y) + 0.2
                
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[ox, oy, oz_half])
            self.obstacle_ids.append(obs_id)
            self.obstacle_positions.append({"pos": [ox, oy], "safe_radius": safe_radius})

    def _spawn_coins_safely(self):
        self.gold_data = []
        
        # Static Curriculum (Stage 0 & 2)
        if not self.randomize_coins:
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

        # Dynamic Curriculum (Stage 1, 3, 4)
        num_coins = np.random.randint(10, 18)
        attempts = 0
        
        while len(self.gold_data) < num_coins and attempts < 200:
            attempts += 1
            pos = [np.random.uniform(-7.0, 7.0), np.random.uniform(-7.0, 7.0), np.random.uniform(0.5, 6.0)]
            
            # Prevent spawning inside the drone's starting area
            if math.sqrt(pos[0]**2 + pos[1]**2 + (pos[2]-0.5)**2) < 1.0:
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
        self.current_step = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        self.plane_id = p.loadURDF("plane.urdf")
        self._build_closed_room()
        self._spawn_obstacles()
        
        urdf_path = os.path.join(os.path.dirname(__file__), "cf2x.urdf")
        # Initialize at Z=2.0m to avoid ground-effect turbulence
        self.drone_id = p.loadURDF(urdf_path, [0, 0, 2.0], globalScaling=4.0)
        p.changeDynamics(self.drone_id, -1, mass=1.0)
                                          
        self._spawn_coins_safely()
        
        return self._get_obs(), {}

    def _compute_lidar(self, drone_pos, rot_matrix):
        ray_starts = []
        ray_ends = []
        offset = 0.25 
        
        for i in range(self.num_rays):
            angle = (2 * math.pi * i) / self.num_rays
            
            # Define rays in the drone's local Body Frame
            local_ray = np.array([math.cos(angle), math.sin(angle), 0])
            
            # Rotate rays to the Global Frame (Ego-centric alignment)
            global_ray = rot_matrix.dot(local_ray)
            
            start = [drone_pos[0] + global_ray[0]*offset, drone_pos[1] + global_ray[1]*offset, drone_pos[2]]
            end = [start[0] + global_ray[0]*self.lidar_range, start[1] + global_ray[1]*self.lidar_range, start[2]]
            
            ray_starts.append(start)
            ray_ends.append(end)
            
        results = p.rayTestBatch(ray_starts, ray_ends)
        lidar_readings = np.array([res[2] for res in results], dtype=np.float32)
        return lidar_readings

    def _get_obs(self):
        drone_pos, ori = p.getBasePositionAndOrientation(self.drone_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.drone_id)
        
        # Drone's rotation matrix for World -> Body frame transformation
        rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        
        # 1. Body Frame Velocity
        local_vel = rot_matrix.T.dot(linear_vel)
        local_ang_vel = rot_matrix.T.dot(angular_vel)
        
        # 2. Body Frame Relative Target Vector (Compass)
        if len(self.gold_data) > 0:
            distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"]))) for g in self.gold_data]
            closest_idx = np.argmin(distances)
            closest_pos = self.gold_data[closest_idx]["pos"]
            
            global_relative_pos = np.array([g_val - d_val for g_val, d_val in zip(closest_pos, drone_pos)])
            local_relative_pos = rot_matrix.T.dot(global_relative_pos)
        else:
            local_relative_pos = np.array([0, 0, 0])
            
        # 3. Ego-centric LiDAR Raycasting
        lidar_data = self._compute_lidar(drone_pos, rot_matrix)
        
        # 4. Kinematics & Singularity Prevention
        euler = p.getEulerFromQuaternion(ori)
        current_roll = euler[0]
        current_pitch = euler[1]
        current_yaw = euler[2]
        
        # Decompose Yaw into Sin/Cos to prevent the -PI to +PI wrap-around singularity
        yaw_sin = math.sin(current_yaw)
        yaw_cos = math.cos(current_yaw)
            
        # Altitude is the only absolute global coordinate retained
        z_altitude = np.array([drone_pos[2]], dtype=np.float32)
        
        # Assemble 50-D State Space
        obs = np.concatenate([
            z_altitude, 
            [current_roll, current_pitch], 
            [yaw_sin, yaw_cos], 
            local_vel, 
            local_ang_vel, 
            local_relative_pos, 
            lidar_data
        ]).astype(np.float32)
        
        # Inject minimal Gaussian noise for Distillation Robustness
        noise = np.random.normal(loc=0.0, scale=0.01, size=obs.shape)
        obs_noisy = obs + noise
        
        return obs_noisy.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Calculate Action Smoothness (Jerk Penalty) to ensure clean CNN datasets
        action_diff = np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()
        
        # ==============================================================
        # PD CONTROLLER (ATTITUDE CONTROL MODE)
        # ==============================================================
        target_pitch = action[0] * (math.pi / 6)  
        target_roll = action[1] * (math.pi / 6)   
        target_yaw_rate = action[2] * 2.0         
        target_thrust = ((action[3] + 1.0) / 2.0) * 20.0  
        
        _, ori = p.getBasePositionAndOrientation(self.drone_id)
        euler = p.getEulerFromQuaternion(ori)
        _, ang_vel = p.getBaseVelocity(self.drone_id)
        
        current_roll, current_pitch, current_yaw = euler
        current_yaw_rate = ang_vel[2]
        
        Kp_ang = 8.0  
        Kd_ang = 4.0  
        Kp_yaw = 3.0  
        
        pitch_error = target_pitch - current_pitch
        roll_error = target_roll - current_roll
        yaw_rate_error = target_yaw_rate - current_yaw_rate
        
        tau_pitch = (Kp_ang * pitch_error) - (Kd_ang * ang_vel[1])
        tau_roll = (Kp_ang * roll_error) - (Kd_ang * ang_vel[0])
        tau_yaw = Kp_yaw * yaw_rate_error
        
        base_f = target_thrust / 4.0
        
        # X-Configuration Motor Mixing
        f0 = base_f - tau_pitch + tau_roll - tau_yaw  
        f1 = base_f + tau_pitch + tau_roll + tau_yaw  
        f2 = base_f + tau_pitch - tau_roll - tau_yaw  
        f3 = base_f - tau_pitch - tau_roll + tau_yaw  
        
        forces = np.clip([f0, f1, f2, f3], 0.0, 7.5)
        # ==============================================================
        
        rotor_offsets = [[0.12, 0.12, 0], [-0.12, 0.12, 0], [-0.12, -0.12, 0], [0.12, -0.12, 0]]
        
        for i in range(4):
            p.applyExternalForce(self.drone_id, -1, forceObj=[0, 0, forces[i]], posObj=rotor_offsets[i], flags=p.LINK_FRAME)

        # Differential Yaw Torque
        torque_mag = ((forces[0] + forces[2]) - (forces[1] + forces[3])) * 0.01
        p.applyExternalTorque(self.drone_id, -1, [0, 0, torque_mag], flags=p.LINK_FRAME)

        # Aerodynamic Damping (Linear & Angular Rotor Drag)
        drone_vel, angular_vel = p.getBaseVelocity(self.drone_id)
        drag_coeff_linear = -0.5  
        drag_coeff_angular = -0.05
        
        drag_force = [v * drag_coeff_linear for v in drone_vel]
        drag_torque = [w * drag_coeff_angular for w in angular_vel]
        
        p.applyExternalForce(self.drone_id, -1, forceObj=drag_force, posObj=[0,0,0], flags=p.WORLD_FRAME)
        p.applyExternalTorque(self.drone_id, -1, torqueObj=drag_torque, flags=p.WORLD_FRAME)
        
        p.stepSimulation()
        
        drone_pos, ori = p.getBasePositionAndOrientation(self.drone_id)
        drone_vel, _ = p.getBaseVelocity(self.drone_id)
        
        terminated = False
        truncated = False
        info = {}
        is_success = False
        is_collision = False
        coin_collected = False 
        current_distance = 0.0
        
        if len(self.gold_data) > 0:
            distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"]))) for g in self.gold_data]
            closest_idx = np.argmin(distances)
            current_distance = distances[closest_idx]
            
            if current_distance < 0.4: 
                p.removeBody(self.gold_data[closest_idx]["id"])
                self.gold_data.pop(closest_idx)
                coin_collected = True 
                
        if len(self.gold_data) == 0:
            is_success = True
            terminated = True
            info['is_success'] = True
            
        hx, hy, hz = self.room_bounds

        euler = p.getEulerFromQuaternion(ori)
        # Fatal collision limits (approx 75 degrees)
        if abs(euler[0]) > 1.3 or abs(euler[1]) > 1.3:
            is_collision = True
            terminated = True
            info['is_success'] = False

        if (abs(drone_pos[0]) > hx - 0.2 or 
            abs(drone_pos[1]) > hy - 0.2 or 
            drone_pos[2] < 0.3 or 
            drone_pos[2] > hz * 2 - 0.2):
            is_collision = True
            terminated = True
            info['is_success'] = False
            
        for entity_id in self.obstacle_ids + self.wall_ids:
            contact_points = p.getContactPoints(bodyA=self.drone_id, bodyB=entity_id)
            if len(contact_points) > 0:
                is_collision = True
                terminated = True
                info['is_success'] = False
                break
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        obs = self._get_obs()
        lidar_data = obs[-36:] 
        info['is_success'] = is_success

        reward = compute_dense_reward(
            drone_pos, drone_vel, action, current_distance, 
            is_collision, is_success, lidar_data, coin_collected, 
            action_diff, 
            reward_weights=self.reward_weights
        )
            
        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)