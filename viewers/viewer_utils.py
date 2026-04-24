"""
viewer_utils.py — shared utilities for all drone viewer scripts.

Used by: watch_any.py, watch_best.py, watch_live.py
"""
import os
import sys
import math
import time
import numpy as np
import yaml
import pybullet as p

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drone_env.drone_sim import RoomDroneEnv

COLLECTION_RADIUS = 0.6  # must match drone_sim.py


# ── Config & environment ───────────────────────────────────────────────────────

def load_stage(stage_n):
    """Load YAML config for a given stage. Returns (config, stage_cfg, reward_weights)."""
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'teacher_ppo.yaml'))
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    sc = config['stages'][f'stage_{stage_n}']
    face_only  = sc.get('face_only', False)
    hover_only = sc.get('hover_only', False)
    if face_only:    rw = config['face_rewards']
    elif hover_only: rw = config['hover_rewards']
    else:            rw = config['nav_rewards']
    return config, sc, rw


def make_env(sc, rw):
    """Create RoomDroneEnv with GUI from a stage config dict."""
    return RoomDroneEnv(
        gui=True,
        num_obstacles=sc['num_obstacles'],
        randomize_obstacles=sc['randomize_obstacles'],
        randomize_coins=sc['randomize_coins'],
        reward_weights=rw,
        hover_only=sc.get('hover_only', False),
        num_fixed_coins=sc.get('num_fixed_coins', 4),
        fixed_spawn=sc.get('fixed_spawn', False),
        coin_spawn_radius=sc.get('coin_spawn_radius'),
        face_only=sc.get('face_only', False),
        face_spawn_radius=sc.get('face_spawn_radius', 3.0),
        face_threshold=sc.get('face_threshold', 0.95),
        face_consecutive_steps=sc.get('face_consecutive_steps', 10),
    )


# ── Visual helpers ─────────────────────────────────────────────────────────────

def _sphere(pos, radius, color):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=list(color))
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=list(pos))


def draw_coin(pos):
    """Solid gold coin sphere (visual only, matches PyBullet sim coin)."""
    _sphere(pos, 0.12, [1, 0.84, 0, 1])


def draw_collection_zone(pos):
    """Very transparent yellow sphere showing the 0.6m collection radius."""
    _sphere(pos, COLLECTION_RADIUS, [1, 0.84, 0, 0.07])


def draw_ghost_coin(pos):
    """Yellow wireframe sphere (3 great-circle rings) marking a collected coin."""
    cx, cy, cz = pos
    color = [1, 0.84, 0]
    n = 24
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        # XY equatorial
        p.addUserDebugLine(
            [cx + COLLECTION_RADIUS * math.cos(a0), cy + COLLECTION_RADIUS * math.sin(a0), cz],
            [cx + COLLECTION_RADIUS * math.cos(a1), cy + COLLECTION_RADIUS * math.sin(a1), cz],
            color, 1, lifeTime=0)
        # XZ meridian
        p.addUserDebugLine(
            [cx + COLLECTION_RADIUS * math.cos(a0), cy, cz + COLLECTION_RADIUS * math.sin(a0)],
            [cx + COLLECTION_RADIUS * math.cos(a1), cy, cz + COLLECTION_RADIUS * math.sin(a1)],
            color, 1, lifeTime=0)
        # YZ meridian
        p.addUserDebugLine(
            [cx, cy + COLLECTION_RADIUS * math.cos(a0), cz + COLLECTION_RADIUS * math.sin(a0)],
            [cx, cy + COLLECTION_RADIUS * math.cos(a1), cz + COLLECTION_RADIUS * math.sin(a1)],
            color, 1, lifeTime=0)


def draw_arrow(base, nose_dir, shaft_id=None, head_l_id=None, head_r_id=None):
    """Draw/update a directional arrow. Returns (shaft_id, head_l_id, head_r_id)."""
    RED = [1, 0, 0]
    W = 2
    tip  = np.array(base) + 0.5 * nose_dir
    up   = np.array([0, 0, 1])
    perp = np.cross(nose_dir, up)
    mg   = np.linalg.norm(perp)
    perp = perp / mg if mg > 0.01 else np.array([1, 0, 0])
    hb   = tip - 0.12 * nose_dir
    lp   = (hb + 0.07 * perp).tolist()
    rp   = (hb - 0.07 * perp).tolist()
    tip, base = tip.tolist(), list(base)
    kw = lambda uid: {'replaceItemUniqueId': uid} if uid is not None else {}
    shaft_id  = p.addUserDebugLine(base, tip, RED, W, lifeTime=0, **kw(shaft_id))
    head_l_id = p.addUserDebugLine(tip,  lp,  RED, W, lifeTime=0, **kw(head_l_id))
    head_r_id = p.addUserDebugLine(tip,  rp,  RED, W, lifeTime=0, **kw(head_r_id))
    return shaft_id, head_l_id, head_r_id


# ── Scene management ──────────────────────────────────────────────────────────

def draw_scene(env_raw):
    """Draw spawn marker, target marker, coins + collection zones. Call after reset."""
    if env_raw.hover_only or env_raw.face_only:
        _sphere(env_raw.hover_target, 0.15, [1, 1, 0, 0.8])
    spawn_pos, _ = p.getBasePositionAndOrientation(env_raw.drone_id)
    _sphere(list(spawn_pos), 0.1, [0.2, 1.0, 0.1, 1.0])
    for g in env_raw.gold_data:
        draw_collection_zone(g['pos'])


def redraw_scene(env_raw):
    """Remove all debug items and redraw the scene. Call on episode end."""
    p.removeAllUserDebugItems()
    draw_scene(env_raw)


# ── HUD ───────────────────────────────────────────────────────────────────────

def _dist_label(env_raw, drone_pos, rot):
    if env_raw.face_only:
        fv = np.array(env_raw.hover_target) - np.array(drone_pos)
        fd = np.linalg.norm(fv)
        cos_t = float((rot.T @ fv)[1] / fd) if fd > 0.05 else 0.0
        return f"cos={cos_t:.3f} stk={env_raw._face_streak}"
    elif env_raw.hover_only:
        d = np.linalg.norm(np.array(drone_pos) - np.array(env_raw.hover_target))
        return f"d={d:.2f}m"
    elif env_raw.gold_data:
        d = min(math.sqrt(sum((a - b) ** 2 for a, b in zip(drone_pos, g['pos'])))
                for g in env_raw.gold_data)
        return f"d={d:.2f}m"
    return "done"


def update_hud(env_raw, drone_pos, rot, stage_n, run_name, mode_label,
               step_r, ep_r, hud_id=None, stage_id=None):
    """
    Two floating text overlays above the drone:
      - White (upper):  stage / run / mode  — static info
      - Red   (lower):  distance + tilt + per-step reward + episode reward
    Returns (hud_id, stage_id) for replaceItemUniqueId on next call.
    """
    ori = p.getBasePositionAndOrientation(env_raw.drone_id)[1]
    eu  = p.getEulerFromQuaternion(ori)
    tilt = math.degrees(math.sqrt(eu[0] ** 2 + eu[1] ** 2))

    metrics   = f"{_dist_label(env_raw, drone_pos, rot)}  tilt:{tilt:.0f}°  r:{step_r:.2f}  R:{ep_r:.0f}"
    stage_lbl = f"S{stage_n} | {run_name} | {mode_label}"

    tp_m = [drone_pos[0], drone_pos[1], drone_pos[2] + 1.1]
    tp_s = [drone_pos[0], drone_pos[1], drone_pos[2] + 1.7]

    kw = lambda uid: {'replaceItemUniqueId': uid} if uid is not None else {}
    hud_id   = p.addUserDebugText(metrics,   tp_m, textColorRGB=[1, 0, 0], textSize=1.0, lifeTime=0, **kw(hud_id))
    stage_id = p.addUserDebugText(stage_lbl, tp_s, textColorRGB=[1, 1, 1], textSize=1.0, lifeTime=0, **kw(stage_id))
    return hud_id, stage_id
