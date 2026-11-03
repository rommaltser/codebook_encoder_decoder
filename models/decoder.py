"""
Texture Decoder (G):
Synthesizes RGB texture images [B, 3, 256, 256] from the spatial residual map Z_res
modulated by the K quantized attribute vectors Z_quant via AdaIN + SPADE-style conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaINLayer(nn.Module):
    """
    Adaptive Instance Normalization modulated by a global style vector w_style.
    """

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_proj = nn.Linear(style_dim, channels * 2)
        # Initialize scale near 1 and bias near 0
        nn.init.zeros_(self.style_proj.weight)
        nn.init.zeros_(self.style_proj.bias)

    def forward(self, x: torch.Tensor, w_style: torch.Tensor) -> torch.Tensor:
        style = self.style_proj(w_style).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        return (1.0 + gamma) * self.norm(x) + beta


class SPADEResidualModulation(nn.Module):
    """
    Spatial modulation conditioned on the resized spatial residual map Z_res
    fused with the attribute style vector w_style.
    """

    def __init__(self, channels: int, res_channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_gate = nn.Linear(style_dim, res_channels)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(res_channels, channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gamma_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.beta_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect")

    def forward(self, x: torch.Tensor, z_res: torch.Tensor, w_style: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        if z_res.shape[2:] != x.shape[2:]:
            z_res_resized = F.interpolate(z_res, size=x.shape[2:], mode="bilinear", align_corners=False)
        else:
            z_res_resized = z_res

        gate = torch.sigmoid(self.style_gate(w_style)).unsqueeze(-1).unsqueeze(-1)
        actv = self.shared_conv(z_res_resized * gate)
        gamma = self.gamma_conv(actv)
        beta = self.beta_conv(actv)
        return normed * (1.0 + gamma) + beta


class ModulatedDecodeBlock(nn.Module):
    """
    Residual block combining AdaIN attribute modulation and SPADE residual conditioning,
    optionally preceded by 2x spatial upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_channels: int,
        style_dim: int,
        upsample: bool = True,
    ):
        super().__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.adain1 = AdaINLayer(out_channels, style_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.spade2 = SPADEResidualModulation(out_channels, res_channels, style_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, z_res: torch.Tensor, w_style: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        residual = self.skip(x)
        out = self.conv1(x)
        out = self.act(self.adain1(out, w_style))
        out = self.conv2(out)
        out = self.act(self.spade2(out, z_res, w_style))
        return out + residual


class TextureDecoder(nn.Module):
    """
    Generates [B, 3, 256, 256] textures from:
      - z_quant: [B, K, latent_dim]
      - z_res:   [B, res_channels, 32, 32]
    """

    def __init__(
        self,
        num_clusters: int = 6,
        latent_dim: int = 128,
        res_channels: int = 128,
        style_dim: int = 256,
        out_channels: int = 3,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.res_channels = res_channels

        # Project concatenated K*D attribute tokens into a unified style embedding
        self.style_mlp = nn.Sequential(
            nn.Linear(num_clusters * latent_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.in_proj = nn.Conv2d(res_channels, 256, kernel_size=3, padding=1, padding_mode="reflect")

        # 32x32 refinement -> 64x64 -> 128x128 -> 256x256
        self.block_32 = ModulatedDecodeBlock(256, 256, res_channels, style_dim, upsample=False)
        self.block_64 = ModulatedDecodeBlock(256, 256, res_channels, style_dim, upsample=True)
        self.block_128 = ModulatedDecodeBlock(256, 128, res_channels, style_dim, upsample=True)
        self.block_256 = ModulatedDecodeBlock(128, 64, res_channels, style_dim, upsample=True)

        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=7, padding=3, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, z_quant: torch.Tensor, z_res: torch.Tensor) -> torch.Tensor:
        bsz = z_quant.shape[0]
        flat_attr = z_quant.reshape(bsz, -1)
        w_style = self.style_mlp(flat_attr)

        h = self.in_proj(z_res)
        h = self.block_32(h, z_res, w_style)
        h = self.block_64(h, z_res, w_style)
        h = self.block_128(h, z_res, w_style)
        h = self.block_256(h, z_res, w_style)
        return self.to_rgb(h)

