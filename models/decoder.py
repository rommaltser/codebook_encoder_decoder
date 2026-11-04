"""
Texture Decoder (G):
Synthesizes RGB texture images [B, 3, 256, 256] from the spatial residual map Z_res
modulated by the K quantized attribute vectors Z_quant via AdaIN + SPADE-style conditioning,
learnable 2D Fourier spatial basis + stochastic noise modulation, and explicit RGB palette restoration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaINLayer(nn.Module):
    """
    Adaptive Instance Normalization modulated by the quantized attribute style vector w_style.
    """

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_proj = nn.Linear(style_dim, channels * 2)
        nn.init.normal_(self.style_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.style_proj.bias)

    def forward(self, x: torch.Tensor, w_style: torch.Tensor) -> torch.Tensor:
        style = self.style_proj(w_style).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        return (1.0 + gamma) * self.norm(x) + beta


class SPADEResidualModulation(nn.Module):
    """
    Spatial modulation conditioned on the residual map Z_res fused with w_style.
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


class SpatialFourierBasis(nn.Module):
    """
    Learnable multi-frequency 2D spatial Fourier basis modulated by w_style so
    the decoder can synthesize periodic & geometric patterns (stripes, grids, hexagons, bubbles)
    directly from the quantized codebook tokens Z_quant without needing edge leakage from Z_res.
    """

    def __init__(self, style_dim: int, out_channels: int, num_freqs: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        # Learnable bank of 2D frequencies initialized across low and medium-high cycles
        init_freqs = torch.randn(num_freqs, 2) * 4.0
        self.freqs = nn.Parameter(init_freqs)
        self.style_to_phase_amp = nn.Linear(style_dim, num_freqs * 3)
        self.proj = nn.Conv2d(num_freqs * 2, out_channels, kernel_size=1)

    def forward(self, bsz: int, height: int, width: int, w_style: torch.Tensor) -> torch.Tensor:
        device = w_style.device
        dtype = w_style.dtype
        ys = torch.linspace(-math.pi, math.pi, height, device=device, dtype=dtype)
        xs = torch.linspace(-math.pi, math.pi, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

        # [H, W, 2] @ [2, F] -> [1, F, H, W]
        angles = torch.matmul(coords, self.freqs.to(dtype=dtype).t()).permute(2, 0, 1).unsqueeze(0)

        params = self.style_to_phase_amp(w_style).view(bsz, 3, self.num_freqs, 1, 1)
        amp_sin = torch.tanh(params[:, 0])
        amp_cos = torch.tanh(params[:, 1])
        phase = params[:, 2] * math.pi

        sin_map = amp_sin * torch.sin(angles + phase)
        cos_map = amp_cos * torch.cos(angles + phase)
        basis = torch.cat([sin_map, cos_map], dim=1)  # [B, 2*F, H, W]
        return self.proj(basis)


class ModulatedDecodeBlock(nn.Module):
    """
    Upsampling block combining:
      - Style-modulated stochastic spatial noise (for fine organic surface relief / texels)
      - AdaIN attribute modulation (for texture geometry)
      - SPADE residual conditioning (for regional color palette & lighting)
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
        self.noise_scale = nn.Linear(style_dim, out_channels)
        nn.init.zeros_(self.noise_scale.weight)
        nn.init.zeros_(self.noise_scale.bias)

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
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

        residual = self.skip(x)
        out = self.conv1(x)

        # Deterministic-seed spatial noise during eval, stochastic during training
        bsz, _, h, w = out.shape
        if self.training:
            noise = torch.randn(bsz, 1, h, w, device=out.device, dtype=out.dtype)
        else:
            g = torch.Generator(device=out.device).manual_seed(12345 + h)
            noise = torch.randn(bsz, 1, h, w, generator=g, device=out.device, dtype=out.dtype)
        n_scale = self.noise_scale(w_style).unsqueeze(-1).unsqueeze(-1) * 0.1
        out = out + noise * n_scale

        out = self.act(self.adain1(out, w_style))
        out = self.conv2(out)
        out = self.act(self.spade2(out, z_res, w_style))
        return out + residual


class TextureDecoder(nn.Module):
    """
    Generates [B, 3, 256, 256] RGB textures from:
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

        # Project each cluster token into its own dedicated subspace and concatenate
        # so every cluster k has independent, non-canceling control over modulation channels & frequencies
        per_cluster_dim = max(32, style_dim // num_clusters)
        concat_dim = per_cluster_dim * num_clusters
        self.cluster_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, per_cluster_dim),
                nn.LayerNorm(per_cluster_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for _ in range(num_clusters)
        ])
        self.style_mlp = nn.Sequential(
            nn.Linear(concat_dim, style_dim),
            nn.LayerNorm(style_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.in_proj = nn.Conv2d(res_channels, 256, kernel_size=3, padding=1, padding_mode="reflect")
        self.fourier_32 = SpatialFourierBasis(style_dim, 256, num_freqs=32)
        self.fourier_64 = SpatialFourierBasis(style_dim, 256, num_freqs=32)

        # 32x32 refinement -> 64x64 -> 128x128 -> 256x256
        self.block_32 = ModulatedDecodeBlock(256, 256, res_channels, style_dim, upsample=False)
        self.block_64 = ModulatedDecodeBlock(256, 256, res_channels, style_dim, upsample=True)
        self.block_128 = ModulatedDecodeBlock(256, 128, res_channels, style_dim, upsample=True)
        self.block_256 = ModulatedDecodeBlock(128, 64, res_channels, style_dim, upsample=True)

        # Synthesizes normalized structural texture pattern, which is then combined with
        # the explicit RGB color canvas and (mu_rgb, std_rgb) extracted in z_res[:, :9].
        self.pattern_head = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, padding_mode="reflect"),
        )
        self.color_residual_head = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, z_quant: torch.Tensor, z_res: torch.Tensor) -> torch.Tensor:
        bsz = z_quant.shape[0]
        # Concatenate per-cluster projections so each cluster k controls a dedicated slice of style space
        cluster_cat = torch.cat([proj(z_quant[:, k, :]) for k, proj in enumerate(self.cluster_projs)], dim=-1)
        w_style = self.style_mlp(cluster_cat)

        # Extract explicit RGB color canvas & statistics packed in z_res[:, :9]
        color_32x32 = z_res[:, 0:3, :, :]
        mu_rgb = z_res[:, 3:6, :1, :1]
        std_rgb = z_res[:, 6:9, :1, :1]
        color_canvas = F.interpolate(color_32x32, size=(256, 256), mode="bilinear", align_corners=False)

        h = self.in_proj(z_res) + self.fourier_32(bsz, 32, 32, w_style)
        h = self.block_32(h, z_res, w_style)
        h = self.block_64(h, z_res, w_style) + self.fourier_64(bsz, 64, 64, w_style)
        h = self.block_128(h, z_res, w_style)
        h = self.block_256(h, z_res, w_style)

        # Normalize spatial pattern to zero-mean unit-variance per channel and scale by target image's std_rgb,
        # anchored on the smooth RGB color_canvas + learned fine residual.
        raw_pattern = self.pattern_head(h)
        pat_mean = raw_pattern.mean(dim=(2, 3), keepdim=True)
        pat_std = raw_pattern.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
        norm_pattern = (raw_pattern - pat_mean) / pat_std

        fine_res = self.color_residual_head(h) * 0.25
        out_rgb = color_canvas + norm_pattern * std_rgb + fine_res * std_rgb + 0.0 * mu_rgb
        return torch.tanh(out_rgb * 1.15)
