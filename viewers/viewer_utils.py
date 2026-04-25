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

COLLECTION_RADIUS = 0.35  # must match drone_sim.py — wing tip (~0.30m) + coin radius (0.12m) ≈ 0.35m


# ── Config & environment ───────────────────────────────────────────────────────

def load_stage(stage_n):
    """Load YAML config for a given stage. Returns (config, stage_cfg, reward_weights)."""
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'teacher_ppo.yaml'))
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    sc = config['stages'][f'stage_{stage_n}']
    hover_only = sc.get('hover_only', False)
    rw = config['hover_rewards'] if hover_only else config['nav_rewards']
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
    """Small grey sphere marking a collected coin. Single IPC call — no freeze."""
    _sphere(pos, 0.12, [0.6, 0.6, 0.6, 0.8])


def draw_trail(prev_pos, curr_pos):
    """Green trajectory trail segment. Call each render frame; resets to None on episode end."""
    if prev_pos is not None:
        p.addUserDebugLine(prev_pos, curr_pos, [0.2, 1.0, 0.1], 2, lifeTime=0)


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
    if env_raw.hover_only:
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
    if env_raw.hover_only:
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

    text = (f"S{stage_n} | {run_name} | {mode_label}"
            f"    {_dist_label(env_raw, drone_pos, rot)}  tilt:{tilt:.0f}deg  r:{step_r:.2f}  R:{ep_r:.0f}")

    tp = [drone_pos[0], drone_pos[1], drone_pos[2] + 1.3]
    kw = {'replaceItemUniqueId': hud_id} if hud_id is not None else {}
    hud_id = p.addUserDebugText(text, tp, textColorRGB=[0, 0, 0], textSize=1.0, lifeTime=0, **kw)
    return hud_id, hud_id
