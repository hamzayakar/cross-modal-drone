"""
Quick sanity check: renders one frame from CollectionDroneEnv and saves it.
Run before training to verify coins are visible in camera images.

Usage:
    python scripts/debug_camera.py
Saves: debug_camera.png (panorama) and debug_camera_zoom.png (8x upscaled)
"""
import os, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from drone_env.visual_drone_env import CollectionDroneEnv, PANO_H, PANO_W

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'teacher_ppo.yaml')

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

# Print coin positions
print("Coin positions:")
for i, g in enumerate(env.gold_data):
    print(f"  coin {i}: {g['pos']}")

pano = env.last_panorama[0]  # (H, W) grayscale [0,1]
print(f"\nPanorama shape: {pano.shape}  min={pano.min():.3f}  max={pano.max():.3f}  mean={pano.mean():.3f}")

# Save as PNG using only numpy (no matplotlib dependency required)
try:
    from PIL import Image
    img = (pano * 255).astype(np.uint8)
    Image.fromarray(img).save('debug_camera.png')
    # 8x upscale so we can actually see pixels
    import numpy as np
    zoom = np.repeat(np.repeat(img, 8, axis=0), 8, axis=1)
    Image.fromarray(zoom).save('debug_camera_zoom.png')
    print("Saved: debug_camera.png (original) + debug_camera_zoom.png (8x zoom)")
    print("Open debug_camera_zoom.png — gold pixels = coins visible in camera")
except ImportError:
    # Fallback: print ASCII art of the panorama
    print("\nPIL not available — ASCII preview (@ = bright pixel, likely coin):")
    scaled = (pano * 9).astype(int)
    chars = ' .:-=+*#%@'
    for row in scaled:
        print(''.join(chars[v] for v in row))

env.close()
