"""
Student A — Behavioral Cloning from teacher demonstrations.

Reads data/distill/chunks/chunk_*.npz sequentially (never loads full dataset
into RAM). Trains StudentNet with MSE loss, evaluates on live Stage 3 episodes,
saves to models/student_a/.

Run after: python scripts/collect_teacher_data.py
"""
import os, sys, argparse, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from student.student_cnn import StudentNet
from student.loss_functions import bc_loss
from drone_env.visual_drone_env import VisualDroneEnv

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHUNK_DIR   = os.path.join(BASE_DIR, 'data', 'distill', 'chunks')
OUT_DIR     = os.path.join(BASE_DIR, 'models', 'student_a')
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'teacher_ppo.yaml')


# ── Dataset ────────────────────────────────────────────────────────────────────

class ChunkDataset(Dataset):
    """Wraps a single loaded chunk as a PyTorch Dataset."""
    def __init__(self, chunk_path):
        d = np.load(chunk_path)
        self.panos   = torch.from_numpy(d['panoramas']).float()  # (N, 1, H, W)
        self.vectors = torch.from_numpy(d['vectors']).float()    # (N, 23)
        self.actions = torch.from_numpy(d['actions']).float()    # (N, 4)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return self.panos[i], self.vectors[i], self.actions[i]


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, n_episodes=20, device='cpu'):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    sc = config['stages']['stage_3']
    rw = config['nav_rewards']

    env = VisualDroneEnv(
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

    model.eval()
    successes, total_coins, total_reward = 0, 0, 0.0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        while True:
            action = model.predict(obs['image'], obs['vector'], device=device)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                if info.get('is_success', False):
                    successes += 1
                total_coins  += info.get('coins_collected', 0)
                total_reward += ep_reward
                break

    env.close()
    sr = successes / n_episodes
    avg_coins = total_coins / n_episodes
    mean_r    = total_reward / n_episodes
    print(f"  Eval: SR={successes}/{n_episodes} ({sr*100:.0f}%)  "
          f"avg_coins={avg_coins:.2f}/4  mean_reward={mean_r:.1f}")
    return sr, avg_coins


# ── Training loop ──────────────────────────────────────────────────────────────

def main(epochs=50, batch_size=256, lr=3e-4, eval_every=10, chunk_dir=None):
    if chunk_dir is None:
        chunk_dir = CHUNK_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    chunks = sorted(glob.glob(os.path.join(chunk_dir, 'chunk_*.npz')))
    if not chunks:
        print(f"No chunks found in {CHUNK_DIR}")
        return
    print(f"Found {len(chunks)} chunks")

    # Count total steps for reporting
    total_steps = sum(np.load(c)['actions'].shape[0] for c in chunks)
    print(f"Total steps: {total_steps:,}  |  chunks: {len(chunks)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    model     = StudentNet().to(device)
    optim     = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_sr = -1.0
    print(f"\nTraining Student A (BC) for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        # Shuffle chunk order each epoch so different orderings are seen
        rng = np.random.default_rng(epoch)
        epoch_chunks = rng.permutation(chunks).tolist()

        # Reserve last chunk for validation
        val_chunk   = epoch_chunks[-1]
        train_chunks = epoch_chunks[:-1]

        for cp in train_chunks:
            ds     = ChunkDataset(cp)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                drop_last=True, num_workers=0)
            for pano, vec, act in loader:
                pano, vec, act = pano.to(device), vec.to(device), act.to(device)
                pred = model(pano, vec)
                loss = bc_loss(pred, act)
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                train_losses.append(loss.item())

        # Validation on held-out chunk
        model.eval()
        val_losses = []
        with torch.no_grad():
            ds     = ChunkDataset(val_chunk)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
            for pano, vec, act in loader:
                pano, vec, act = pano.to(device), vec.to(device), act.to(device)
                val_losses.append(bc_loss(model(pano, vec), act).item())

        scheduler.step()
        tr_l = np.mean(train_losses)
        va_l = np.mean(val_losses)
        print(f"Epoch {epoch:3d}/{epochs}  train={tr_l:.5f}  val={va_l:.5f}", flush=True)

        if epoch % eval_every == 0:
            sr, avg_coins = evaluate(model, n_episodes=20, device=device)
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, f'student_a_ep{epoch}.pt'))
            if sr > best_sr:
                best_sr = sr
                torch.save(model.state_dict(),
                           os.path.join(OUT_DIR, 'best_model.pt'))
                print(f"  New best SR={best_sr*100:.0f}% → saved")

    print(f"\nDone. Best SR={best_sr*100:.0f}%  → {OUT_DIR}/best_model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--eval_every', type=int,   default=10)
    parser.add_argument('--chunk_dir',  type=str,   default=None,
                        help='Chunk directory (default: data/distill/chunks)')
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.eval_every, args.chunk_dir)
