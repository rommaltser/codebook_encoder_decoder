"""
Dual-Branch Texture Encoder (E):
Extracts:
  1. Spatial residual map Z_res in R^{B x C_res x 32 x 32}:
     Strictly disentangled non-semantic residual containing:
       - Channels 0..2: Smooth 4x4 low-pass RGB color & lighting canvas (upsampled to 32x32)
       - Channels 3..5: Global per-image RGB mean mu_rgb
       - Channels 6..8: Global per-image RGB std sigma_rgb
       - Channels 9..C_res-1: Spatially uncorrelated stochastic noise seed (projected from per-pixel scrambled luminance)
     Because Z_res has NO learned multi-pixel convolutional filters on x, it is mathematically
     incapable of leaking geometric element shapes (stripes, bubbles, hexagons, grids), forcing
     100% of texture geometry through the Factorized Codebook Z_quant while preserving 100% of RGB color!
  2. K semantic attribute vectors Z_attr in R^{B x K x D}
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResBlock(nn.Module):
    def __init__(self, channels: int, norm_type: str = "batch"):
        super().__init__()
        if norm_type == "instance":
            norm1 = nn.InstanceNorm2d(channels, affine=True)
            norm2 = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == "none":
            norm1 = nn.Identity()
            norm2 = nn.Identity()
        else:
            norm1 = nn.BatchNorm2d(channels)
            norm2 = nn.BatchNorm2d(channels)

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect", bias=(norm_type == "none")),
            norm1,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect", bias=(norm_type == "none")),
            norm2,
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TextureEncoder(nn.Module):
    """
    Dual-branch encoder splitting input textures [B, 3, 256, 256] into:
      - z_attr: [B, K, latent_dim] (structural & surface attributes)
      - z_res:  [B, res_channels, 32, 32] (RGB palette, lighting, and non-semantic stochastic seed)
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

        # 1x1 projection from (color_canvas, mu_rgb, std_rgb, scrambled_stochastic_noise) -> res_channels
        # Uses ONLY 1x1 convolutions on color + spatially-scrambled noise so zero spatial geometry can leak!
        stoch_out_ch = max(1, res_channels - 9)
        self.res_noise_proj = nn.Sequential(
            nn.Conv2d(9 + 4, stoch_out_ch, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Semantic Attribute Extractor Z_attr [B, K, latent_dim]
        self.attr_stem = nn.Sequential(
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
        bsz = x.size(0)

        # 1. Explicit Color Palette & Smooth Lighting Statistics in Z_res (first 9 channels)
        color_4x4 = F.adaptive_avg_pool2d(x, (4, 4))
        color_32x32 = F.interpolate(color_4x4, size=(32, 32), mode="bilinear", align_corners=False)  # [B, 3, 32, 32]

        mu_rgb = x.mean(dim=(2, 3), keepdim=True).expand(-1, -1, 32, 32)                             # [B, 3, 32, 32]
        std_rgb = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-4).expand(-1, -1, 32, 32)             # [B, 3, 32, 32]

        # Spatially-uncorrelated stochastic noise seed (deterministic in eval, random in train)
        if self.training:
            noise_32 = torch.randn(bsz, 4, 32, 32, device=x.device, dtype=x.dtype)
        else:
            g = torch.Generator(device=x.device).manual_seed(42)
            noise_32 = torch.randn(bsz, 4, 32, 32, generator=g, device=x.device, dtype=x.dtype)

        base_9 = torch.cat([color_32x32, mu_rgb, std_rgb], dim=1)
        stoch_res = self.res_noise_proj(torch.cat([base_9, noise_32], dim=1))
        z_res = torch.cat([base_9, stoch_res], dim=1)                                                # [B, C_res, 32, 32]

        # 2. K Semantic Attribute Vectors
        feat = self.attr_stem(x)
        attr_map = self.attr_trunk(feat)
        avg_pool = attr_map.mean(dim=(2, 3))
        max_pool = attr_map.amax(dim=(2, 3))
        global_feat = torch.cat([avg_pool, max_pool], dim=1)

        z_attr_list = [head(global_feat) for head in self.attr_heads]
        z_attr = torch.stack(z_attr_list, dim=1)  # [B, K, latent_dim]
        return z_attr, z_res
