"""
watch_student.py — watch the trained student (CNN) fly with dual view.

Left window:  PyBullet 3D GUI — drone flying in the room
Right window: Student's own panoramic camera view (what the CNN sees)

Usage:
  python viewers/watch_student.py
  python viewers/watch_student.py --model models/student_a/best_model.pt
  python viewers/watch_student.py --stride 4   # slow-motion
"""
import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from student.student_cnn import StudentNet
from drone_env.visual_drone_env import VisualDroneEnv, CAM_C
import yaml

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')
    except ImportError:
        print("Neither cv2 nor matplotlib found — camera view disabled.")

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'teacher_ppo.yaml')
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'student_a', 'best_model.pt')

parser = argparse.ArgumentParser()
parser.add_argument('--model',  type=str, default=MODEL_PATH)
parser.add_argument('--stride', type=int, default=6,
                    help='Physics steps per GUI frame (6≈real-time, 1=slow-motion)')
parser.add_argument('--episodes', type=int, default=0, help='0 = run forever')
args = parser.parse_args()

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
sc = config['stages']['stage_3']
rw = config['nav_rewards']

# ── Load student model ─────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = StudentNet().to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()
print(f"Loaded student: {args.model}  (device={device})")

# ── Environment ────────────────────────────────────────────────────────────────
env = VisualDroneEnv(
    gui=True,
    num_obstacles=sc['num_obstacles'],
    randomize_obstacles=sc['randomize_obstacles'],
    randomize_coins=sc['randomize_coins'],
    reward_weights=rw,
    hover_only=sc['hover_only'],
    num_fixed_coins=sc['num_fixed_coins'],
    max_steps=sc['max_steps'],
    coin_count_range=tuple(sc['coin_count_range']),
    coin_z_range=tuple(sc['coin_z_range']),
    coin_spawn_area=sc['coin_spawn_area'],
)


def pano_to_display(pano_chw):
    """Convert (C,H,W) float32 [0,1] → uint8 BGR for cv2, or RGB for matplotlib."""
    if pano_chw.shape[0] == 3:
        img = (pano_chw.transpose(1, 2, 0) * 255).astype(np.uint8)  # (H,W,3) RGB
    else:
        gray = (pano_chw[0] * 255).astype(np.uint8)
        img = np.stack([gray, gray, gray], axis=2)                   # (H,W,3) gray→RGB
    # 8× upscale so it's actually visible
    img = np.repeat(np.repeat(img, 8, axis=0), 8, axis=1)
    return img


def show_camera(img_rgb, ep, step, coins_left, sr, episodes_done):
    label = f"Ep {ep}  step {step}  coins_left={coins_left}  SR={sr:.0%}"
    if HAS_CV2:
        img_bgr = img_rgb[:, :, ::-1].copy()
        cv2.putText(img_bgr, label, (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Student Camera (panoramic 360°)", img_bgr)
        cv2.waitKey(1)
    else:
        plt.clf()
        plt.imshow(img_rgb)
        plt.title(label)
        plt.axis('off')
        plt.pause(0.001)


# ── Main loop ──────────────────────────────────────────────────────────────────
ep = 0
successes = 0
total_coins = 0

if not HAS_CV2:
    plt.figure(figsize=(12, 2))
    plt.ion()
    plt.show()

print("Running student. Close the PyBullet window or press Ctrl-C to stop.\n")

try:
    while args.episodes == 0 or ep < args.episodes:
        obs, _ = env.reset()
        ep += 1
        step = 0
        ep_reward = 0.0

        while True:
            pano = obs['image']       # (C, H, W)
            vec  = obs['vector']      # (VECTOR_DIM,)

            action = model.predict(pano, vec, device=device)

            # Show camera every stride steps
            if step % args.stride == 0:
                img = pano_to_display(pano)
                coins_left = len(env.gold_data)
                sr = successes / ep if ep > 0 else 0.0
                show_camera(img, ep, step, coins_left, sr, ep)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1

            if terminated or truncated:
                success = info.get('is_success', False)
                coins   = info.get('coins_collected', 0)
                if success:
                    successes += 1
                total_coins += coins

                print(f"Ep {ep:3d} | steps={step:4d} | reward={ep_reward:7.1f} | "
                      f"coins={coins}/4 | {'✓ SUCCESS' if success else '✗ fail'} | "
                      f"SR={successes/ep:.0%}")
                break

        time.sleep(0.3)  # brief pause between episodes so you can see the reset

except KeyboardInterrupt:
    pass
finally:
    env.close()
    if HAS_CV2:
        cv2.destroyAllWindows()
    print(f"\nFinal: {successes}/{ep} episodes succeeded  "
          f"avg_coins={total_coins/max(ep,1):.2f}/4")
