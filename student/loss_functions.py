import torch
import torch.nn.functional as F


def bc_loss(pred_actions, teacher_actions):
    """MSE between predicted and teacher actions. Both (B, 4) in [-1, 1]."""
    return F.mse_loss(pred_actions, teacher_actions)
