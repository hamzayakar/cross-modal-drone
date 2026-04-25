"""
watch_live.py — watch the raw training state in real time.

Follows model priority: latest (during training) → final (just ended) → best (peak).
The model reloads every episode so you see the policy as it's being learned,
including degradation and recovery phases.

Usage:
  python viewers/watch_live.py --stage 0
  python viewers/watch_live.py --stage 1 --stride 1   # slow-motion
"""
import os, sys, time, argparse
import numpy as np
import cloudpickle
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from viewers.viewer_utils import load_stage, make_env, draw_scene, redraw_scene, draw_arrow, draw_trail, update_hud, draw_ghost_coin

parser = argparse.ArgumentParser()
parser.add_argument('--stage',  type=int, default=0,  help='Curriculum stage (0=Hover, 1=Scout, ...)')
parser.add_argument('--stride', type=int, default=10, help='Physics steps per GUI frame (10≈real-time, 1=slow-motion)')
args = parser.parse_args()

config, sc, rw = load_stage(args.stage)
RUN_NAME = sc['run_name']

env_raw = make_env(sc, rw)
env_vec = DummyVecEnv([lambda e=env_raw: e])

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', f'stage_{args.stage}', RUN_NAME))

def _find_model():
    for name in ['latest_model', 'final_model', 'best_model']:
        mp = os.path.join(model_dir, name)
        vp = f"{mp}_vecnormalize.pkl"
        if os.path.exists(f"{mp}.zip") and os.path.exists(vp):
            return mp, vp, name
    return None, None, None

print(f"[S{args.stage}|{RUN_NAME}|LIVE] Waiting for model files ...")
model_path, vecn_path, model_name = _find_model()
while model_path is None:
    print("  waiting..."); time.sleep(5)
    model_path, vecn_path, model_name = _find_model()

env = VecNormalize.load(vecn_path, env_vec)
env.training = False; env.norm_reward = False
model = PPO.load(model_path, env=env)
obs = env.reset()
p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,1])
print(f"[{RUN_NAME}] Loaded {model_name}. Watching live training state.")

draw_scene(env_raw)
sid = hli = hri = hud_id = stage_id = None
ep_r = ep_s = 0; ep_t = time.time(); step_r = 0.
_tracked = [g['pos'][:] for g in env_raw.gold_data]
_trail = None

while True:
    done = False
    for _ in range(args.stride):
        act, _ = model.predict(obs, deterministic=True)
        obs, rews, dones, _ = env.step(act)
        step_r = rews[0]; ep_r += step_r; ep_s += 1
        if dones[0]: done = True; break
        cur = [g['pos'][:] for g in env_raw.gold_data]
        for pos in list(_tracked):
            if pos not in cur:
                draw_ghost_coin(pos); _tracked.remove(pos)

    if done:
        print(f"Episode | steps={ep_s} sim={ep_s/240:.1f}s R={ep_r:.1f} [{model_name}]")
        ep_r = ep_s = 0; step_r = 0.; sid = hli = hri = hud_id = stage_id = None
        _trail = None
        p.removeAllUserDebugItems()
        time.sleep(1.0)
        redraw_scene(env_raw); ep_t = time.time()
        _tracked = [g['pos'][:] for g in env_raw.gold_data]
        new_path, new_vn, new_name = _find_model()
        if new_path:
            model_path, vecn_path, model_name = new_path, new_vn, new_name
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
    if ep_s % max(12, args.stride) == 0:
        hud_id, stage_id = update_hud(env_raw, dp, rot, args.stage, RUN_NAME,
                                       f'LIVE({model_name})', step_r, ep_r, hud_id, stage_id)
