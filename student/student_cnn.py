"""
Student policy network: CNN (panoramic, circular-padded) + MLP (proprioception).
Used for both Student A (offline BC) and Student B (RL via SB3 feature extractor).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from drone_env.visual_drone_env import PANO_H, PANO_W, VECTOR_DIM, CAM_C


class _CircPadConv2d(nn.Module):
    """Conv2d with circular padding on W and zero padding on H."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.pad_w = kw // 2
        self.pad_h = kh // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (kh, kw), stride=stride, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode='circular')
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode='constant', value=0.0)
        return self.conv(x)


class StudentNet(nn.Module):
    """
    Standalone student policy for BC training (Student A).
    Input:
      image  — (B, 3, PANO_H, PANO_W) float32 in [0,1]  RGB
      vector — (B, VECTOR_DIM) float32
    Output:
      action — (B, 4) float32 in [-1, 1] via tanh
    """
    def __init__(self, pano_h=PANO_H, pano_w=PANO_W, vector_dim=VECTOR_DIM):
        super().__init__()

        self.cnn = nn.Sequential(
            _CircPadConv2d(CAM_C, 32, kernel_size=(3, 8), stride=(1, 4)),
            nn.ReLU(),
            _CircPadConv2d(32, 64, kernel_size=(3, 4), stride=(2, 2)),
            nn.ReLU(),
            _CircPadConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 16)),
            nn.Flatten(),
        )
        cnn_out = 64 * 4 * 16  # 4096

        self.vec_fc = nn.Sequential(
            nn.Linear(vector_dim, 64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(cnn_out + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

    def forward(self, image, vector):
        vis = self.cnn(image)
        vec = self.vec_fc(vector)
        return self.head(torch.cat([vis, vec], dim=1))

    @torch.no_grad()
    def predict(self, image_np, vector_np, device='cpu'):
        """Single-step inference. image_np: (1,H,W), vector_np: (VECTOR_DIM,)."""
        img = torch.from_numpy(image_np[np.newaxis]).to(device)
        vec = torch.from_numpy(vector_np[np.newaxis]).to(device)
        return self(img, vec).squeeze(0).cpu().numpy()


# ── SB3 feature extractor (Student B) ─────────────────────────────────────────
try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium

    class StudentFeatureExtractor(BaseFeaturesExtractor):
        """
        Custom feature extractor for SB3 MultiInputPolicy.
        Expects Dict obs: {'image': (1,H,W), 'vector': (VECTOR_DIM,)}.
        """
        def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 256):
            super().__init__(observation_space, features_dim)

            img_shape = observation_space['image'].shape   # (1, H, W)
            vec_dim   = observation_space['vector'].shape[0]

            self.cnn = nn.Sequential(
                _CircPadConv2d(img_shape[0], 32, kernel_size=(3, 8), stride=(1, 4)),
                nn.ReLU(),
                _CircPadConv2d(32, 64, kernel_size=(3, 4), stride=(2, 2)),
                nn.ReLU(),
                _CircPadConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 16)),
                nn.Flatten(),
            )
            cnn_out = 64 * 4 * 16  # 4096

            self.vec_fc = nn.Sequential(
                nn.Linear(vec_dim, 64),
                nn.ReLU(),
            )

            self.merge = nn.Sequential(
                nn.Linear(cnn_out + 64, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations):
            vis = self.cnn(observations['image'])
            vec = self.vec_fc(observations['vector'])
            return self.merge(torch.cat([vis, vec], dim=1))

except ImportError:
    pass  # SB3 not installed — StudentFeatureExtractor unavailable, BC path still works
