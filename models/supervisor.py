"""
Discriminative Supervisor (D):
Combines:
  1. A shared convolutional trunk (ResNet-50 or custom multi-scale texture trunk)
     followed by K linear classification heads (each with `words_per_cluster` output logits).
  2. An auxiliary 70x70 PatchGAN discriminator head for surface realism.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as tv_models


class PatchGANDiscriminator(nn.Module):
    """
    Standard 70x70 PatchGAN discriminator (pix2pix / CycleGAN architecture).
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.extend([
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.extend([
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DiscriminativeSupervisor(nn.Module):
    """
    Dual-role supervisor:
      - Multi-head attribute classifier D_attr: returns list of K logit tensors [B, words_per_cluster]
      - PatchGAN realism discriminator D_patch: returns patch map [B, 1, H_p, W_p]
    """

    def __init__(
        self,
        num_clusters: int = 6,
        words_per_cluster: int = 24,
        in_channels: int = 3,
        backbone: str = "resnet50",
        pretrained_backbone: bool = False,
        ndf: int = 64,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.words_per_cluster = words_per_cluster

        # Build ResNet convolutional trunk for texture classification
        if backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
            resnet = tv_models.resnet18(weights=weights)
            feat_dim = 512
        else:
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
            resnet = tv_models.resnet50(weights=weights)
            feat_dim = 2048

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.attr_trunk = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(1),
        )

        # K independent classification heads (24 classes each)
        self.attr_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(feat_dim, words_per_cluster),
            )
            for _ in range(num_clusters)
        ])

        # Auxiliary 70x70 PatchGAN head
        self.patch_disc = PatchGANDiscriminator(in_channels=in_channels, ndf=ndf, n_layers=3)

    def classify_attributes(self, x: torch.Tensor) -> List[torch.Tensor]:
        feat = self.attr_trunk(x)
        return [head(feat) for head in self.attr_heads]

    def discriminate_patch(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_disc(x)

    def forward(
        self,
        x: torch.Tensor,
        return_attr: bool = True,
        return_patch: bool = True,
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        logits = self.classify_attributes(x) if return_attr else None
        patch_pred = self.discriminate_patch(x) if return_patch else None
        return logits, patch_pred
