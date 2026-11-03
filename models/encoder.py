"""
Dual-Branch Texture Encoder (E):
Extracts:
  1. Spatial residual map Z_res in R^{B x C_res x H' x W'} (palette, lighting, fine stochastic detail)
  2. K semantic attribute vectors Z_attr in R^{B x K x D}
"""

from typing import Tuple
import torch
import torch.nn as nn


class ConvResBlock(nn.Module):
    def __init__(self, channels: int, norm_type: str = "instance"):
        super().__init__()
        norm_layer = nn.InstanceNorm2d if norm_type == "instance" else nn.BatchNorm2d
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            norm_layer(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            norm_layer(channels, affine=True),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TextureEncoder(nn.Module):
    """
    Dual-branch convolutional encoder splitting input textures [B, 3, 256, 256] into:
      - z_attr: [B, K, latent_dim]
      - z_res:  [B, res_channels, 32, 32]
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        res_channels: int = 128,
        latent_dim: int = 128,
        num_clusters: int = 6,
        res_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.res_channels = res_channels

        # Shared Stem: 256x256 -> 128x128 -> 64x64 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 -> 128
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            ConvResBlock(base_channels * 4, norm_type="instance"),
        )

        # Branch 1: Spatial Residual Z_res [B, res_channels, 32, 32]
        # Uses channel bottleneck + InstanceNorm + optional spatial dropout to limit semantic shape leakage
        self.res_branch = nn.Sequential(
            ConvResBlock(base_channels * 4, norm_type="instance"),
            nn.Conv2d(base_channels * 4, res_channels, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(res_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=res_dropout) if res_dropout > 0 else nn.Identity(),
        )

        # Branch 2: Semantic Attribute Extractor Z_attr [B, K, latent_dim]
        # Further downsamples 32 -> 16 -> 8 and pools global texture statistics
        attr_feat_dim = base_channels * 8
        self.attr_trunk = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 6, attr_feat_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(attr_feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            ConvResBlock(attr_feat_dim, norm_type="batch"),
        )

        # Combine Global Avg Pool + Global Std/Max Pool for texture descriptor richness
        self.attr_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(attr_feat_dim * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(latent_dim, latent_dim),
            )
            for _ in range(num_clusters)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        z_res = self.res_branch(feat)

        attr_map = self.attr_trunk(feat)
        avg_pool = attr_map.mean(dim=(2, 3))
        max_pool = attr_map.amax(dim=(2, 3))
        global_feat = torch.cat([avg_pool, max_pool], dim=1)

        z_attr_list = [head(global_feat) for head in self.attr_heads]
        z_attr = torch.stack(z_attr_list, dim=1)  # [B, K, latent_dim]
        return z_attr, z_res

