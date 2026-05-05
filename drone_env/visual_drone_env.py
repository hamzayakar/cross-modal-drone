import math
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from .drone_sim import RoomDroneEnv

# Default per-camera image dimensions
CAM_H = 24
CAM_W = 64
N_CAMS = 3
PANO_H = CAM_H
PANO_W = CAM_W * N_CAMS  # 192
CAM_C = 3               # RGB channels

# Student non-visual observation dimensions
PROP_DIM = 11          # alt(1) + roll,pitch(2) + sin_yaw,cos_yaw(2) + lin_vel(3) + ang_vel(3)
ACT_HIST_LEN = 3       # how many past actions to append
ACT_DIM = 4
VECTOR_DIM = PROP_DIM + ACT_HIST_LEN * ACT_DIM  # 23


def render_cameras(client, drone_id, cam_h, cam_w):
    """
    Renders 3 cameras (0°, 120°, 240° relative to drone yaw).
    Returns float32 grayscale panorama (1, cam_h, cam_w*3) in [0, 1].
    Uses ER_TINY_RENDERER — works in DIRECT mode, no OpenGL required.
    """
    drone_pos, ori = p.getBasePositionAndOrientation(drone_id, physicsClientId=client)
    yaw = p.getEulerFromQuaternion(ori)[2]

    frames = []
    for offset_deg in [0, 120, 240]:
        cam_yaw = yaw + math.radians(offset_deg)
        target = [
            drone_pos[0] + math.cos(cam_yaw),
            drone_pos[1] + math.sin(cam_yaw),
            drone_pos[2],
        ]
        view_mat = p.computeViewMatrix(
            cameraEyePosition=list(drone_pos),
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
        )
        proj_mat = p.computeProjectionMatrixFOV(
            fov=120.0,
            aspect=float(cam_w) / cam_h,
            nearVal=0.3,  # clips drone body (~0.25m) without clipping coins (nearest face ≥0.35m at collection)
            farVal=15.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=cam_w,
            height=cam_h,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_TINY_RENDERER,
            shadow=1,
            lightDirection=[1, 1, 1],
            physicsClientId=client,
        )
        rgb = np.array(rgba, dtype=np.float32).reshape(cam_h, cam_w, 4)[:, :, :3]
        frames.append(rgb)  # (H, W, 3)

    panorama = np.concatenate(frames, axis=1)              # (H, 3W, 3)
    return (panorama / 255.0).transpose(2, 0, 1).astype(np.float32)  # (3, H, 3W)


def get_proprioception(drone_id, client):
    """
    Returns 11D ego-centric proprioception: no compass, no LiDAR.
    Same coordinate conventions as drone_sim._get_obs() for consistency.
    """
    drone_pos, ori = p.getBasePositionAndOrientation(drone_id, physicsClientId=client)
    linear_vel, angular_vel = p.getBaseVelocity(drone_id, physicsClientId=client)
    rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    local_vel = rot_matrix.T.dot(linear_vel)
    local_ang_vel = rot_matrix.T.dot(angular_vel)

    euler = p.getEulerFromQuaternion(ori)
    yaw = euler[2]
    wp, wr = euler[0], euler[1]
    body_pitch = -(wp * math.cos(yaw) + wr * math.sin(yaw))
    body_roll  = -wp * math.sin(yaw) + wr * math.cos(yaw)

    return np.array([
        drone_pos[2],
        body_roll, body_pitch,
        math.sin(yaw), math.cos(yaw),
        *local_vel,
        *local_ang_vel,
    ], dtype=np.float32)


class CollectionDroneEnv(RoomDroneEnv):
    """
    Drop-in replacement for RoomDroneEnv during teacher data collection.
    Returns the same 50D state obs (so VecNormalize / teacher policy work unchanged)
    but also renders and stores the current panorama + proprioception as attributes
    that the collection script reads after each step.
    """
    def __init__(self, cam_h=CAM_H, cam_w=CAM_W, **kwargs):
        super().__init__(**kwargs)
        self.cam_h = cam_h
        self.cam_w = cam_w
        self._act_hist = np.zeros(ACT_HIST_LEN * ACT_DIM, dtype=np.float32)
        # Side-channel outputs (updated every step/reset)
        self.last_panorama       = np.zeros((CAM_C, cam_h, cam_w * N_CAMS), dtype=np.float32)
        self.last_proprioception = np.zeros(PROP_DIM, dtype=np.float32)
        self.last_act_hist       = np.zeros(ACT_HIST_LEN * ACT_DIM, dtype=np.float32)

    def _update_visual(self):
        self.last_panorama       = render_cameras(self.client, self.drone_id, self.cam_h, self.cam_w)
        self.last_proprioception = get_proprioception(self.drone_id, self.client)
        self.last_act_hist       = self._act_hist.copy()

    def reset(self, **kwargs):
        self._act_hist[:] = 0.0
        obs, info = super().reset(**kwargs)
        self._update_visual()
        return obs, info  # 50D state obs — unchanged for teacher

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._act_hist = np.roll(self._act_hist, -ACT_DIM)
        self._act_hist[-ACT_DIM:] = action
        self._update_visual()
        return obs, reward, terminated, truncated, info  # 50D state obs — unchanged


class VisualDroneEnv(RoomDroneEnv):
    """
    Student B RL training environment.
    Returns Dict obs {'image': (1,H,3W), 'vector': (VECTOR_DIM,)} instead of 50D state.
    Identical reward function and physics as teacher.
    """
    def __init__(self, cam_h=CAM_H, cam_w=CAM_W, **kwargs):
        super().__init__(**kwargs)
        self.cam_h = cam_h
        self.cam_w = cam_w
        self._act_hist = np.zeros(ACT_HIST_LEN * ACT_DIM, dtype=np.float32)

        self.observation_space = spaces.Dict({
            'image':  spaces.Box(0.0, 1.0,
                                 shape=(CAM_C, cam_h, cam_w * N_CAMS), dtype=np.float32),
            'vector': spaces.Box(-np.inf, np.inf,
                                 shape=(VECTOR_DIM,), dtype=np.float32),
        })

    def _get_visual_obs(self):
        panorama = render_cameras(self.client, self.drone_id, self.cam_h, self.cam_w)
        prop = get_proprioception(self.drone_id, self.client)
        vector = np.concatenate([prop, self._act_hist])
        return {'image': panorama, 'vector': vector}

    def reset(self, **kwargs):
        self._act_hist[:] = 0.0
        _, info = super().reset(**kwargs)      # discard 50D state obs
        return self._get_visual_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        self._act_hist = np.roll(self._act_hist, -ACT_DIM)
        self._act_hist[-ACT_DIM:] = action
        return self._get_visual_obs(), reward, terminated, truncated, info
