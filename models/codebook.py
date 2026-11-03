"""
Factorized Codebook (C):
Maintains K independent embedding lookup tables of shape [words_per_cluster, latent_dim].
Supports:
  - Straight-Through Estimation (STE) continuous quantization during reconstruction
  - Optional supervised alignment loss binding codebook row w to lexicon word index w
  - Direct manual token substitution via lookup_index(cluster_idx, word_idx)
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedCodebook(nn.Module):
    def __init__(
        self,
        num_clusters: int = 6,
        words_per_cluster: int = 24,
        latent_dim: int = 128,
        beta: float = 0.25,
        align_weight: float = 0.5,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.words_per_cluster = words_per_cluster
        self.latent_dim = latent_dim
        self.beta = beta
        self.align_weight = align_weight

        # K separate embedding tables, each of shape [24, D]
        self.tables = nn.ModuleList([
            nn.Embedding(words_per_cluster, latent_dim)
            for _ in range(num_clusters)
        ])
        for table in self.tables:
            nn.init.uniform_(table.weight, -1.0 / words_per_cluster, 1.0 / words_per_cluster)

    def lookup_index(
        self,
        cluster_idx: int,
        word_idx: Union[int, torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Direct manual lookup of a lexicon token embedding from cluster `cluster_idx`.
        Returns tensor of shape [D] (or [B, D] if batch_size is given or word_idx is a 1D tensor).
        """
        if cluster_idx < 0 or cluster_idx >= self.num_clusters:
            raise IndexError(f"cluster_idx {cluster_idx} out of range [0, {self.num_clusters - 1}]")

        table = self.tables[cluster_idx]
        device = table.weight.device

        if isinstance(word_idx, int):
            idx_tensor = torch.tensor(word_idx, dtype=torch.long, device=device)
            emb = table(idx_tensor)  # [D]
            if batch_size is not None:
                emb = emb.unsqueeze(0).expand(batch_size, -1)
            return emb

        idx_tensor = word_idx.to(device=device, dtype=torch.long)
        return table(idx_tensor)

    def forward(
        self,
        z_attr: torch.Tensor,
        target_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_attr: [B, K, D] continuous attribute vectors from TextureEncoder
            target_indices: Optional [B, K] ground-truth lexicon indices. When provided and
                            self.align_weight > 0, adds a cross-entropy alignment loss on negative
                            distances so codebook entry w corresponds 1-to-1 with lexicon word w.
        Returns:
            z_quant: [B, K, D] quantized vectors with Straight-Through Estimator gradients
            indices: [B, K] nearest-neighbor codebook indices
            vq_loss: scalar VQ commitment + codebook (+ optional alignment) loss
        """
        bsz, num_k, dim = z_attr.shape
        if num_k != self.num_clusters or dim != self.latent_dim:
            raise ValueError(
                f"Expected z_attr of shape [B, {self.num_clusters}, {self.latent_dim}], got {tuple(z_attr.shape)}"
            )

        z_quant_list = []
        indices_list = []
        total_vq_loss = z_attr.new_tensor(0.0)

        for k in range(self.num_clusters):
            z_k = z_attr[:, k, :]               # [B, D]
            emb_k = self.tables[k].weight       # [W, D]

            # Compute squared L2 distance: ||z_k - emb_k||^2 = ||z_k||^2 + ||emb_k||^2 - 2 z_k @ emb_k^T
            dists = (
                torch.sum(z_k ** 2, dim=1, keepdim=True)
                + torch.sum(emb_k ** 2, dim=1)
                - 2.0 * torch.matmul(z_k, emb_k.t())
            )  # [B, W]

            idx_k = torch.argmin(dists, dim=1)  # [B]
            z_q_k = self.tables[k](idx_k)       # [B, D]

            # Standard VQ-VAE loss: codebook loss + beta * commitment loss
            codebook_loss = F.mse_loss(z_q_k, z_k.detach())
            commit_loss = F.mse_loss(z_k, z_q_k.detach())
            cluster_loss = codebook_loss + self.beta * commit_loss

            # Supervised codebook alignment: encourages -dists to peak at ground-truth word index y_A[:, k]
            # and pulls emb_k[y_A[:, k]] toward z_k
            if target_indices is not None and self.align_weight > 0.0:
                gt_idx_k = target_indices[:, k]
                align_ce = F.cross_entropy(-dists, gt_idx_k)
                gt_emb_k = self.tables[k](gt_idx_k)
                supervised_pull = F.mse_loss(gt_emb_k, z_k.detach()) + self.beta * F.mse_loss(z_k, gt_emb_k.detach())
                cluster_loss = cluster_loss + self.align_weight * (0.2 * align_ce + supervised_pull)

            total_vq_loss = total_vq_loss + cluster_loss

            # Straight-Through Estimator (STE)
            z_q_k_ste = z_k + (z_q_k - z_k).detach()
            z_quant_list.append(z_q_k_ste)
            indices_list.append(idx_k)

        z_quant = torch.stack(z_quant_list, dim=1)      # [B, K, D]
        indices = torch.stack(indices_list, dim=1)      # [B, K]
        total_vq_loss = total_vq_loss / float(self.num_clusters)

        return z_quant, indices, total_vq_loss

