"""
Student A — Behavioral Cloning from teacher demonstrations.

Loads data/distill/teacher_stage3.npz, trains StudentNet with MSE loss,
evaluates on live Stage 3 episodes, saves to models/student_a/.

Run after: python scripts/collect_teacher_data.py
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from student.student_cnn import StudentNet
from student.loss_functions import bc_loss
from drone_env.visual_drone_env import VisualDroneEnv, VECTOR_DIM

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'distill', 'teacher_stage3.npz')
OUT_DIR   = os.path.join(BASE_DIR, 'models', 'student_a')
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'teacher_ppo.yaml')


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_dataset(path, val_frac=0.1, device='cpu'):
    data = np.load(path)
    panos   = torch.from_numpy(data['panoramas']).float()   # (N, 1, H, W)
    vectors = torch.from_numpy(data['vectors']).float()     # (N, 23)
    actions = torch.from_numpy(data['actions']).float()     # (N, 4)

    N = len(actions)
    n_val = max(1, int(N * val_frac))
    idx = torch.randperm(N)
    tr, va = idx[n_val:], idx[:n_val]

    train_ds = TensorDataset(panos[tr], vectors[tr], actions[tr])
    val_ds   = TensorDataset(panos[va], vectors[va], actions[va])
    print(f"Dataset: {N} steps  |  train={len(tr)}  val={len(va)}")
    return train_ds, val_ds


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
    successes = 0
    total_reward = 0.0

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
                total_reward += ep_reward
                break

    env.close()
    sr = successes / n_episodes
    mean_r = total_reward / n_episodes
    print(f"  Eval: SR={successes}/{n_episodes} ({sr*100:.0f}%)  mean_reward={mean_r:.1f}")
    return sr


# ── Training loop ──────────────────────────────────────────────────────────────

def main(epochs=50, batch_size=256, lr=3e-4, eval_every=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds, val_ds = load_dataset(DATA_PATH, device=device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = StudentNet().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_sr = -1.0
    print(f"Training Student A (BC) for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for pano, vec, act in train_loader:
            pano, vec, act = pano.to(device), vec.to(device), act.to(device)
            pred = model(pano, vec)
            loss = bc_loss(pred, act)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_losses.append(loss.item())

        # Validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for pano, vec, act in val_loader:
                pano, vec, act = pano.to(device), vec.to(device), act.to(device)
                val_losses.append(bc_loss(model(pano, vec), act).item())

        scheduler.step()
        tr_l = np.mean(train_losses)
        va_l = np.mean(val_losses)
        print(f"Epoch {epoch:3d}/{epochs}  train_loss={tr_l:.5f}  val_loss={va_l:.5f}")

        if epoch % eval_every == 0:
            sr = evaluate(model, n_episodes=20, device=device)
            save_path = os.path.join(OUT_DIR, f'student_a_ep{epoch}.pt')
            torch.save(model.state_dict(), save_path)
            if sr > best_sr:
                best_sr = sr
                torch.save(model.state_dict(), os.path.join(OUT_DIR, 'best_model.pt'))
                print(f"  New best SR={best_sr*100:.0f}% → saved")

    print(f"\nDone. Best SR={best_sr*100:.0f}%  model → {OUT_DIR}/best_model.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--eval_every', type=int,   default=10,
                        help='Evaluate on live env every N epochs')
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr, args.eval_every)
