import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
import os

# Import the custom reward logic
from .reward_functions import compute_dense_reward

class RoomDroneEnv(gym.Env):
    """
    A 16x16m complex closed room environment with random obstacles and coins.
    State Space: 32-D (16 Kinematics + 16 LiDAR Raycasts)
    Action Space: 4-D continuous vector (motor thrusts).
    """
    def __init__(self, gui=False):
        super().__init__()
        
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # UPGRADED STATE SPACE: 16 Base + 16 Lidar = 32-D
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        
        self.room_bounds = [8.0, 8.0, 4.0]
        self.max_steps = 14400 # 1 minute
        
        self.drone_id = None
        self.wall_ids = []
        self.obstacle_ids = []
        self.obstacle_positions = [] 
        self.gold_data = []
        
        # LiDAR Settings
        self.num_rays = 16
        self.lidar_range = 5.0 # Max sensing distance in meters

    def _build_closed_room(self):
        """Constructs the room boundaries using semi-transparent glass-like walls."""
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
        """Spawns 20-30 random cylindrical and cubic obstacles."""
        self.obstacle_ids = []
        self.obstacle_positions = []
        
        num_obstacles = np.random.randint(20, 31)
        
        for _ in range(num_obstacles):
            ox = np.random.uniform(-7.0, 7.0)
            oy = np.random.uniform(-7.0, 7.0)
            
            if math.sqrt(ox**2 + oy**2) < 1.0:
                continue
                
            oz_half = np.random.uniform(1.0, 3.0) 
            is_cylinder = np.random.choice([True, False])
            
            if is_cylinder:
                radius = np.random.uniform(0.2, 0.6)
                col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=oz_half*2)
                vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=oz_half*2, rgbaColor=[0.4, 0.4, 0.5, 1])
                safe_radius = radius + 0.2
            else:
                ext_x = np.random.uniform(0.2, 0.6)
                ext_y = np.random.uniform(0.2, 0.6)
                col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ext_x, ext_y, oz_half])
                vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[ext_x, ext_y, oz_half], rgbaColor=[0.5, 0.4, 0.4, 1])
                safe_radius = max(ext_x, ext_y) + 0.2
                
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[ox, oy, oz_half])
            self.obstacle_ids.append(obs_id)
            self.obstacle_positions.append({"pos": [ox, oy], "safe_radius": safe_radius})

    def _spawn_coins_safely(self):
        """Spawns coins using Rejection Sampling."""
        self.gold_data = []
        num_coins = np.random.randint(10, 18)
        attempts = 0
        
        while len(self.gold_data) < num_coins and attempts < 200:
            attempts += 1
            pos = [np.random.uniform(-7.0, 7.0), np.random.uniform(-7.0, 7.0), np.random.uniform(0.5, 6.0)]
            
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
        # UPDATED: Drone starts at 2.0m height to avoid instant floor collisions
        self.drone_id = p.loadURDF(urdf_path, [0, 0, 2.0], globalScaling=4.0)
        p.changeDynamics(self.drone_id, -1, mass=1.0)
                                          
        self._spawn_coins_safely()
        
        return self._get_obs(), {}

    def _compute_lidar(self, drone_pos):
        """
        Casts 16 mathematical rays around the drone to detect obstacles.
        Returns a normalized array [0.0 - 1.0] where 1.0 means no obstacle in range.
        """
        ray_starts = []
        ray_ends = []
        
        # Offset to prevent the ray from hitting the drone's own body
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
        
        forces = np.clip(action, -1.0, 1.0) * 15.0 
        rotor_offsets = [[0.12, 0.12, 0], [-0.12, 0.12, 0], [-0.12, -0.12, 0], [0.12, -0.12, 0]]
        
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
        coin_collected = False # NEW: Track if a coin was eaten this step
        current_distance = 0.0
        
        if len(self.gold_data) > 0:
            distances = [math.sqrt(sum((d - val)**2 for d, val in zip(drone_pos, g["pos"]))) for g in self.gold_data]
            closest_idx = np.argmin(distances)
            current_distance = distances[closest_idx]
            
            if current_distance < 0.4: 
                p.removeBody(self.gold_data[closest_idx]["id"])
                self.gold_data.pop(closest_idx)
                coin_collected = True # NEW: Coin successfully collected!
                
        if len(self.gold_data) == 0:
            is_success = True
            terminated = True
            info['is_success'] = True
            
        hx, hy, hz = self.room_bounds

        euler = p.getEulerFromQuaternion(ori)
        # Roll (X ekseninde yatma) veya Pitch (Y ekseninde yatma) 1.3 radyanı (~75 derece) geçerse drone devrilmiştir!
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
            
        for obs_id in self.obstacle_ids:
            contact_points = p.getContactPoints(bodyA=self.drone_id, bodyB=obs_id)
            if len(contact_points) > 0:
                is_collision = True
                terminated = True
                info['is_success'] = False
                break
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        obs = self._get_obs()
        lidar_data = obs[-16:]

        info['is_success'] = is_success

        # NEW: Pass coin_collected to the reward function
        reward = compute_dense_reward(drone_pos, drone_vel, action, current_distance, is_collision, is_success, lidar_data, coin_collected)
            
        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)