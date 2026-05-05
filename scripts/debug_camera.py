"""
Quick sanity check: renders one frame from CollectionDroneEnv and saves it.
Run before training to verify coins are visible in camera images.

Usage:
    python scripts/debug_camera.py [--tag v2_rgb_darken]
Saves to: debug_images/debug_<tag>_{original,zoom}.png
"""
import os, sys, argparse, datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from drone_env.visual_drone_env import CollectionDroneEnv, PANO_H, PANO_W, CAM_H, CAM_W

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'teacher_ppo.yaml')
OUT_DIR     = os.path.join(os.path.dirname(__file__), '..', 'debug_images')

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default=None,
                    help='Version tag for output filenames (default: timestamp)')
args = parser.parse_args()

tag = args.tag if args.tag else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(OUT_DIR, exist_ok=True)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
sc = config['stages']['stage_3']
rw = config['nav_rewards']

env = CollectionDroneEnv(
    gui=False,
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

env.reset()

print("Coin positions:")
for i, g in enumerate(env.gold_data):
    print(f"  coin {i}: {g['pos']}")

pano_chw = env.last_panorama  # (C, H, W)
C, H, W = pano_chw.shape
print(f"\nPanorama shape (C,H,W): {pano_chw.shape}  "
      f"min={pano_chw.min():.3f}  max={pano_chw.max():.3f}  mean={pano_chw.mean():.3f}")
print(f"Per-camera resolution: {CAM_W}x{CAM_H}  |  Panorama: {W}x{H}  |  Channels: {C}")

try:
    from PIL import Image

    if C == 3:
        img = (pano_chw.transpose(1, 2, 0) * 255).astype(np.uint8)  # (H,W,3)
    else:
        img = (pano_chw[0] * 255).astype(np.uint8)  # (H,W)

    orig_path = os.path.join(OUT_DIR, f'debug_{tag}_original.png')
    zoom_path = os.path.join(OUT_DIR, f'debug_{tag}_zoom.png')

    Image.fromarray(img).save(orig_path)
    zoom = np.repeat(np.repeat(img, 8, axis=0), 8, axis=1)
    Image.fromarray(zoom).save(zoom_path)

    print(f"\nSaved:")
    print(f"  {orig_path}")
    print(f"  {zoom_path}")
    print("White/bright blobs = coins | Dark = floor | Check arms are gone from frame")

except ImportError:
    print("\nPIL not available — ASCII preview:")
    gray = pano_chw.mean(axis=0) if C == 3 else pano_chw[0]
    scaled = (gray * 9).astype(int)
    chars = ' .:-=+*#%@'
    for row in scaled:
        print(''.join(chars[v] for v in row))

env.close()
