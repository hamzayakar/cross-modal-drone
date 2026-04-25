"""
watch_any.py — load and watch any model from any completed stage (never reloads).

Use cases:
  - Compare best_model vs final_model after training ends
  - Review any past stage's peak performance
  - Post-training analysis

Usage:
  python viewers/watch_any.py --stage 0
  python viewers/watch_any.py --stage 1 --model final
  python viewers/watch_any.py --stage 0 --stride 1    # slow-motion
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
parser.add_argument('--stage',  type=int,   default=0,      help='Curriculum stage (0=Hover, 1=Scout, ...)')
parser.add_argument('--model',  type=str,   default='best', help="Model type: 'best' or 'final'")
parser.add_argument('--stride', type=int,   default=10,     help='Physics steps per GUI frame (10≈real-time, 1=slow-motion)')
args = parser.parse_args()

config, sc, rw = load_stage(args.stage)
RUN_NAME = sc['run_name']

env_raw = make_env(sc, rw)
env_vec = DummyVecEnv([lambda e=env_raw: e])

model_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', f'stage_{args.stage}', RUN_NAME))
model_path = os.path.join(model_dir, f'{args.model}_model')
vecn_path  = f"{model_path}_vecnormalize.pkl"

if not os.path.exists(f"{model_path}.zip"):
    print(f"Model not found: {model_path}.zip"); sys.exit(1)

env = VecNormalize.load(vecn_path, env_vec)
env.training = False; env.norm_reward = False
model = PPO.load(model_path, env=env)
obs = env.reset()
p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,1])
print(f"[S{args.stage}|{RUN_NAME}|{args.model.upper()}] Watching. Model will NOT reload.")

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
        print(f"Episode | steps={ep_s} sim={ep_s/240:.1f}s R={ep_r:.1f}")
        ep_r = ep_s = 0; step_r = 0.; sid = hli = hri = hud_id = stage_id = None
        _trail = None
        p.removeAllUserDebugItems()
        time.sleep(1.0)
        redraw_scene(env_raw); ep_t = time.time()
        _tracked = [g['pos'][:] for g in env_raw.gold_data]

    dp, ori = p.getBasePositionAndOrientation(env_raw.drone_id)
    rot = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    nd  = rot @ np.array([0, 1, 0])
    sid, hli, hri = draw_arrow(list(dp), nd, sid, hli, hri)
    draw_trail(_trail, list(dp)); _trail = list(dp)
    if ep_s % max(12, args.stride) == 0:
        hud_id, stage_id = update_hud(env_raw, dp, rot, args.stage, RUN_NAME,
                                       args.model.upper(), step_r, ep_r, hud_id, stage_id)
