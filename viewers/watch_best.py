"""
watch_best.py — hot-reload best_model after each episode during active training.

Use this while training is running. The model reloads whenever EvalCallback
saves a new best checkpoint, so you see the policy improve in real time.

Usage:
  python notebooks/watch_best.py
"""
import os, sys, time
import numpy as np
import cloudpickle
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from viewers.viewer_utils import load_stage, make_env, draw_scene, redraw_scene, draw_arrow, draw_trail, update_hud, draw_ghost_coin

# ── Config ────────────────────────────────────────────────────────────────────
STAGE_TO_WATCH = 0          # 0=Hover 1=Scout 2=Navigator 3=Hunter 4=Pathfinder ...
RENDER_STRIDE  = 10
# ─────────────────────────────────────────────────────────────────────────────

config, sc, rw = load_stage(STAGE_TO_WATCH)
RUN_NAME = sc['run_name']

env_raw = make_env(sc, rw)
env_vec = DummyVecEnv([lambda e=env_raw: e])

model_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', f'stage_{STAGE_TO_WATCH}', RUN_NAME))
model_path = os.path.join(model_dir, 'best_model')
vecn_path  = f"{model_path}_vecnormalize.pkl"

print(f"[S{STAGE_TO_WATCH}|{RUN_NAME}|BEST] Waiting for {model_path}.zip ...")
while not (os.path.exists(f"{model_path}.zip") and os.path.exists(vecn_path)):
    print("  waiting for first EvalCallback checkpoint..."); time.sleep(5)

env = VecNormalize.load(vecn_path, env_vec)
env.training = False; env.norm_reward = False
model = PPO.load(model_path, env=env)
obs = env.reset()
p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,1])
print(f"[{RUN_NAME}] Running. Reloads best_model on each episode end.")

draw_scene(env_raw)
sid = hli = hri = hud_id = stage_id = None
ep_r = ep_s = 0; ep_t = time.time(); step_r = 0.
_tracked = [g['pos'][:] for g in env_raw.gold_data]
_trail = None

while True:
    done = False
    for _ in range(RENDER_STRIDE):
        act, _ = model.predict(obs, deterministic=True)
        obs, rews, dones, _ = env.step(act)
        step_r = rews[0]; ep_r += step_r; ep_s += 1
        if dones[0]: done = True; break
        cur = [g['pos'][:] for g in env_raw.gold_data]
        for pos in list(_tracked):
            if pos not in cur:
                draw_ghost_coin(pos); _tracked.remove(pos)

    if done:
        print(f"Episode | steps={ep_s} sim={ep_s/240:.1f}s R={ep_r:.1f}")
        ep_r = ep_s = 0; step_r = 0.; sid = hli = hri = hud_id = stage_id = None
        _trail = None
        p.removeAllUserDebugItems()
        time.sleep(1.0)
        redraw_scene(env_raw); ep_t = time.time()
        _tracked = [g['pos'][:] for g in env_raw.gold_data]
        try:
            with open(vecn_path, 'rb') as f: saved = cloudpickle.load(f)
            env.obs_rms = saved.obs_rms; env.ret_rms = saved.ret_rms
            model = PPO.load(model_path, env=env)
        except Exception: pass

    dp, ori = p.getBasePositionAndOrientation(env_raw.drone_id)
    rot = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    nd  = rot @ np.array([0, 1, 0])
    sid, hli, hri = draw_arrow(list(dp), nd, sid, hli, hri)
    draw_trail(_trail, list(dp)); _trail = list(dp)
    if ep_s % max(12, RENDER_STRIDE) == 0:
        hud_id, stage_id = update_hud(env_raw, dp, rot, STAGE_TO_WATCH, RUN_NAME,
                                       'BEST', step_r, ep_r, hud_id, stage_id)
