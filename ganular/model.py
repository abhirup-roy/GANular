import numpy as np
import torch
import torch.nn as nn
import pytorch3d as p3d
from . import params


def get_noise(size):
    return torch.Tensor(np.random.normal(0.0, 0.5, size).astype(np.float32))


class PGenerator(nn.Module):
    """
    3D Generator model, adapted from diaGAN (Coiffiere et al. 2020)

    """

    def __init__(self) -> None:
        super().__init__()

        self.noise_size = params.noise_size
        self.embed_shape = (1, params.embed_size, params.embed_size, params.embed_size)

        self.upscale = lambda x: nn.functional.interpolate(
            x, scale_factor=2, mode="trilinear", align_corners=True
        )

        self.lin_embed = nn.Sequential(
            nn.Linear(self.noise_size, self.embed_size),
            nn.ReLU6(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU6(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU6(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU6(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU6(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_embed(x)
        x = x.view((x.size(0),) + self.embed_shape)
        # 16x16x16

        x = self.conv1(x)
        x = self.upscale(x)
        # 32x32x32

        x = self.conv2(x)
        x = self.upscale(x)
        # 64x64x64
        # Stop upscaling here for 64^3 outputs

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # ReLU6 -> [0,6], rescale -> [0,1]
        return x / 6


class PDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.n_proj = params.n_proj

        self.conv1 = nn.Sequential(
            nn.Con
