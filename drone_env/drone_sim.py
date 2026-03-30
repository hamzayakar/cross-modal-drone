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
        
        # Action space: 4 rotors, continuous values [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # Observation space: 32-D (Kinematics + Lidar)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        
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
        
        self.num_rays = 16
        self.lidar_range = 5.0 

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
            
        # Pin the random seed for reproducibility (Stage 2 & 3) if not fully randomized
        rng = np.random.default_rng(seed=42) if not self.randomize_obstacles else np.random.default_rng()
            
        for _ in range(self.num_obstacles):
            ox = rng.uniform(-7.0, 7.0)
            oy = rng.uniform(-7.0, 7.0)
            
            # Keep a clear spawn zone for the drone
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
        
        # CURRICULUM STAGE 0: The "Force-Feeding" Trick
        # If coins are fixed, spawn one extremely close to teach the reward association.
        if not self.randomize_coins:
            fixed_positions = [
                [1.0, 0.0, 2.0],  # Right in front of the drone's nose!
                [0.0, 1.5, 2.0],  # Slightly to the side
                [4.0, 4.0, 2.0],  # Far corner
                [-4.0, -4.0, 2.0] # Far corner
            ]
            for pos in fixed_positions:
                vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.12, rgbaColor=[1, 0.84, 0, 1])
                gid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=pos)
                self.gold_data.append({"id": gid, "pos": pos})
            return

        # STAGE 1+: Randomize coin positions safely
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
        
        self.plane_id = p.loadURDF("plane.urdf")
        self._build_closed_room()
        self._spawn_obstacles()
        
        urdf_path = os.path.join(os.path.dirname(__file__), "cf2x.urdf")
        # Drone starts at Z=2.0 meters (mid-air) to prevent immediate ground collisions
        self.drone_id = p.loadURDF(urdf_path, [0, 0, 2.0], globalScaling=4.0)
        p.changeDynamics(self.drone_id, -1, mass=1.0)
                                          
        self._spawn_coins_safely()
        
        return self._get_obs(), {}

    def _compute_lidar(self, drone_pos):
        ray_starts = []
        ray_ends = []
        offset = 0.25 
        
        for i in range(self.num_rays):
            angle = (2 * math.pi * i) / self.num_rays
            dx = math.cos(angle)
            dy = math.sin(angle)
            
            start = [drone_pos[0] + dx * offset, drone_pos[1] + dy * offset, drone_pos[2]]
            end = [start[0] + dx * self.lidar_range, start[1] + dy * self.lidar_range, start[2]]
            
            ray_starts.append(start)
            ray_ends.append(end)
            
        results = p.rayTestBatch(ray_starts, ray_ends)
        lidar_readings = np.array([res[2] for res in results], dtype=np.float32)
        return lidar_readings

    def _get_obs(self):
        drone_pos, ori = p.getBasePositionAndOrientation(self.drone_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.drone_id)
        
        # Calculate relative vector to the closest coin
        if len(self.gold_data) > 0:
            distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"]))) for g in self.gold_data]
            closest_idx = np.argmin(distances)
            closest_pos = self.gold_data[closest_idx]["pos"]
            relative_pos = [g_val - d_val for g_val, d_val in zip(closest_pos, drone_pos)]
        else:
            relative_pos = [0, 0, 0]
            
        lidar_data = self._compute_lidar(drone_pos)
            
        obs = np.concatenate([drone_pos, ori, linear_vel, angular_vel, relative_pos, lidar_data]).astype(np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        
        # --- HOVER BIAS & ACTION MAPPING ---
        # A 1kg drone needs 9.81N to hover (2.4525N per motor)
        hover_force = 9.81 / 4.0 
        
        # Agent outputs [-1.0, 1.0]. 0.0 maps directly to perfect hover.
        # Agent can add or subtract up to 5.0N from the baseline.
        raw_forces = hover_force + (action * 5.0)
        
        # CRITICAL FIX: Prevent negative thrust. Motors can only push up [0.0, 10.0].
        forces = np.clip(raw_forces, 0.0, 10.0) 
        
        rotor_offsets = [[0.12, 0.12, 0], [-0.12, 0.12, 0], [-0.12, -0.12, 0], [0.12, -0.12, 0]]
        
        # --- CRITICAL FIX: ACTION REPEAT REMOVED (Back to 240Hz) ---
        # Direct rotor control requires extremely high frequency. 
        # Holding actions for 4 frames caused unrecoverable aerodynamic flips.
        for i in range(4):
            p.applyExternalForce(self.drone_id, -1, forceObj=[0, 0, forces[i]], posObj=rotor_offsets[i], flags=p.LINK_FRAME)
            
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
        
        # Check coin collection
        if len(self.gold_data) > 0:
            distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"]))) for g in self.gold_data]
            closest_idx = np.argmin(distances)
            current_distance = distances[closest_idx]
            
            # 40cm collection radius
            if current_distance < 0.4: 
                p.removeBody(self.gold_data[closest_idx]["id"])
                self.gold_data.pop(closest_idx)
                coin_collected = True 
                
        # Room Cleared
        if len(self.gold_data) == 0:
            is_success = True
            terminated = True
            info['is_success'] = True
            
        hx, hy, hz = self.room_bounds

        # 75 Degree Tilt Rule (1.3 radians) -> Fatal Crash
        euler = p.getEulerFromQuaternion(ori)
        if abs(euler[0]) > 1.3 or abs(euler[1]) > 1.3:
            is_collision = True
            terminated = True
            info['is_success'] = False

        # Room Boundaries Constraints (Walls, Floor, Ceiling)
        if (abs(drone_pos[0]) > hx - 0.2 or 
            abs(drone_pos[1]) > hy - 0.2 or 
            drone_pos[2] < 0.3 or 
            drone_pos[2] > hz * 2 - 0.2):
            is_collision = True
            terminated = True
            info['is_success'] = False
            
        # Obstacle Collisions
        for obs_id in self.obstacle_ids:
            contact_points = p.getContactPoints(bodyA=self.drone_id, bodyB=obs_id)
            if len(contact_points) > 0:
                is_collision = True
                terminated = True
                info['is_success'] = False
                break
            
        # Timeout
        if self.current_step >= self.max_steps:
            truncated = True
            
        obs = self._get_obs()
        lidar_data = obs[-16:]
        info['is_success'] = is_success

        # Compute highly tuned dense reward from the YAML parameters
        reward = compute_dense_reward(
            drone_pos, drone_vel, action, current_distance, 
            is_collision, is_success, lidar_data, coin_collected, 
            reward_weights=self.reward_weights
        )
            
        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)