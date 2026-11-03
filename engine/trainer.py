"""
Multi-Stage Swapping Trainer for CoTyle Dictionary-Controlled Texture Autoencoder.
Implements:
  - Phase 1 (Epochs 1..pretrain_epochs): Discriminative Supervisor multi-head CE pre-training
  - Phase 2 (Epochs pretrain_epochs+1..max_epochs): Full Autoencoder + Random Cluster Swapping + PatchGAN
"""

import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.encoder import TextureEncoder
from models.codebook import FactorizedCodebook
from models.decoder import TextureDecoder
from models.supervisor import DiscriminativeSupervisor


class PerceptualLoss(nn.Module):
    """
    Perceptual similarity loss.
    Uses `lpips.LPIPS(net='vgg')` if installed and available; otherwise falls back
    to multi-scale spatial gradient + structural Pyramid L1 loss that works offline.
    """

    def __init__(self):
        super().__init__()
        self.lpips_fn = None
        try:
            import lpips  # type: ignore
            self.lpips_fn = lpips.LPIPS(net="vgg")
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        except Exception:
            self.lpips_fn = None

        # Register Sobel filters for high-frequency texture structure preservation when offline
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1))

    def _offline_multiscale_texture_loss(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        loss = x_pred.new_tensor(0.0)
        curr_p, curr_t = x_pred, x_target
        for scale in range(3):
            gx_p = F.conv2d(curr_p, self.sobel_x, padding=1, groups=3)
            gy_p = F.conv2d(curr_p, self.sobel_y, padding=1, groups=3)
            gx_t = F.conv2d(curr_t, self.sobel_x, padding=1, groups=3)
            gy_t = F.conv2d(curr_t, self.sobel_y, padding=1, groups=3)

            grad_loss = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
            # Gram matrix style correlation on spatial gradients for texture fidelity
            b, c, h, w = gx_p.shape
            norm_factor = float(c * 3 * h * w) ** 0.5
            feat_p = torch.cat([curr_p, gx_p, gy_p], dim=1).view(b, c * 3, h * w) / norm_factor
            feat_t = torch.cat([curr_t, gx_t, gy_t], dim=1).view(b, c * 3, h * w) / norm_factor
            gram_p = torch.bmm(feat_p, feat_p.transpose(1, 2))
            gram_t = torch.bmm(feat_t, feat_t.transpose(1, 2))
            gram_loss = F.l1_loss(gram_p, gram_t)

            loss = loss + (grad_loss + 2.0 * gram_loss) / (2 ** scale)
            if scale < 2:
                curr_p = F.avg_pool2d(curr_p, kernel_size=2, stride=2)
                curr_t = F.avg_pool2d(curr_t, kernel_size=2, stride=2)
        return loss

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        if self.lpips_fn is not None:
            # LPIPS VGG16 runs safely and with 2x lower VRAM under AMP autocast
            return self.lpips_fn(x_pred, x_target).mean()
        with torch.amp.autocast("cuda", enabled=False):
            return self._offline_multiscale_texture_loss(x_pred.float(), x_target.float())


class CoTyleTrainer:
    def __init__(
        self,
        encoder: TextureEncoder,
        codebook: FactorizedCodebook,
        decoder: TextureDecoder,
        supervisor: DiscriminativeSupervisor,
        cfg: dict,
        device: torch.device,
        cluster_order: Optional[List[str]] = None,
        cluster_lexicon: Optional[Dict[str, List[str]]] = None,
    ):
        self.device = device
        self.cfg = cfg
        self.cluster_order = cluster_order or []
        self.cluster_lexicon = cluster_lexicon or {}

        self.encoder = encoder.to(device)
        self.codebook = codebook.to(device)
        self.decoder = decoder.to(device)
        self.supervisor = supervisor.to(device)
        self.perceptual_loss = PerceptualLoss().to(device)

        model_cfg = cfg.get("model", {})
        self.use_amp = bool(model_cfg.get("use_amp", True)) and (device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=self.use_amp)

        loss_cfg = cfg.get("loss_weights", {})
        self.w_recon_l1 = float(loss_cfg.get("recon_l1", 1.0))
        self.w_recon_lpips = float(loss_cfg.get("recon_lpips", 0.8))
        self.w_vq = float(loss_cfg.get("vq", 0.25))
        self.w_swap = float(loss_cfg.get("swap_attr", 0.5))
        self.w_adv = float(loss_cfg.get("adv_patch", 0.1))

        optim_cfg = cfg.get("optim", {})
        self.lr_g = float(optim_cfg.get("lr_g", 2e-4))
        self.lr_d = float(optim_cfg.get("lr_d", 2e-4))
        self.sup_finetune_scale = float(optim_cfg.get("sup_finetune_lr_scale", 0.1))
        beta1 = float(optim_cfg.get("beta1", 0.5))
        beta2 = float(optim_cfg.get("beta2", 0.999))
        self.save_dir = optim_cfg.get("save_dir", "checkpoints")
        self.log_interval = int(optim_cfg.get("log_interval", 20))
        self.vis_interval = int(optim_cfg.get("vis_interval", 5))
        os.makedirs(self.save_dir, exist_ok=True)

        # Phase 1 Optimizer: Supervisor Attribute Classifier (trunk + K heads)
        sup_attr_params = list(self.supervisor.attr_trunk.parameters()) + list(self.supervisor.attr_heads.parameters())
        self.opt_sup_pretrain = torch.optim.Adam(
            sup_attr_params, lr=self.lr_d, betas=(beta1, beta2)
        )

        # Phase 2 Optimizers:
        # 1) Generator (Encoder + Codebook + Decoder)
        g_params = (
            list(self.encoder.parameters())
            + list(self.codebook.parameters())
            + list(self.decoder.parameters())
        )
        self.opt_g = torch.optim.Adam(g_params, lr=self.lr_g, betas=(beta1, beta2))

        # 2) Discriminator (PatchGAN at full lr_d + optional Supervisor attr fine-tuning at 0.1 * lr_d)
        d_param_groups = [
            {"params": list(self.supervisor.patch_disc.parameters()), "lr": self.lr_d},
        ]
        if self.sup_finetune_scale > 0.0:
            d_param_groups.append(
                {"params": sup_attr_params, "lr": self.lr_d * self.sup_finetune_scale}
            )
        self.opt_d = torch.optim.Adam(d_param_groups, betas=(beta1, beta2))

    @staticmethod
    def _gan_loss(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)

    def compute_supervisor_ce(self, logits: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Multi-head cross-entropy across all K attribute heads:
            sum_{k=1}^K CrossEntropy(logits[k], targets[:, k])
        """
        total_ce = targets.new_tensor(0.0, dtype=torch.float32)
        for k, head_logits in enumerate(logits):
            total_ce = total_ce + F.cross_entropy(head_logits, targets[:, k])
        return total_ce / float(len(logits))

    def train_supervisor_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Phase 1: Train Discriminative Supervisor alone on real images and pseudo-labels.
        """
        self.supervisor.train()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        num_batches = 0

        for step, (x_real, y_real) in enumerate(loader, start=1):
            x_real = x_real.to(self.device)
            y_real = y_real.to(self.device)

            self.opt_sup_pretrain.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, _ = self.supervisor(x_real, return_attr=True, return_patch=False)
                assert logits is not None
                loss_sup = self.compute_supervisor_ce(logits, y_real)

            self.scaler.scale(loss_sup).backward()
            self.scaler.step(self.opt_sup_pretrain)
            self.scaler.update()

            # Compute mean top-1 accuracy across all K heads
            with torch.no_grad():
                for k, head_logits in enumerate(logits):
                    preds = head_logits.argmax(dim=-1)
                    total_correct += int((preds == y_real[:, k]).sum().item())
                    total_predictions += int(y_real.size(0))

            total_loss += float(loss_sup.item())
            num_batches += 1

            if step % self.log_interval == 0 or step == len(loader):
                acc = 100.0 * total_correct / max(1, total_predictions)
                print(
                    f"[Phase 1 | Epoch {epoch:03d} | Step {step:04d}/{len(loader):04d}] "
                    f"sup_ce: {loss_sup.item():.4f} | mean_attr_acc: {acc:.2f}%"
                )

        mean_loss = total_loss / max(1, num_batches)
        mean_acc = 100.0 * total_correct / max(1, total_predictions)
        return {"sup_ce": mean_loss, "attr_acc": mean_acc}

    def train_joint_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Phase 2: Jointly train Encoder, Codebook, Decoder, and Discriminator with Attribute Swapping.
        """
        self.encoder.train()
        self.codebook.train()
        self.decoder.train()
        self.supervisor.train()

        if self.sup_finetune_scale <= 0.0:
            self.supervisor.attr_trunk.eval()
            self.supervisor.attr_heads.eval()

        metrics = {
            "loss_g": 0.0,
            "loss_recon_l1": 0.0,
            "loss_recon_lpips": 0.0,
            "loss_vq": 0.0,
            "loss_swap_attr": 0.0,
            "loss_adv_g": 0.0,
            "loss_d": 0.0,
            "swap_target_acc": 0.0,
        }
        num_batches = 0
        swap_correct = 0
        swap_total = 0

        last_batch_vis = None

        # Free Phase 1 pretrain optimizer states if still allocated to reclaim VRAM for Phase 2
        if len(self.opt_sup_pretrain.state) > 0:
            self.opt_sup_pretrain.state.clear()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        for step, (x_a, y_a) in enumerate(loader, start=1):
            x_a = x_a.to(self.device)
            y_a = y_a.to(self.device)
            bsz, _, _, _ = x_a.shape
            num_clusters = self.codebook.num_clusters

            # Freeze supervisor parameters during Generator step to save VRAM
            self.supervisor.requires_grad_(False)
            self.opt_g.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # =================================================================
                # 1. Forward Pass & Quantization
                # =================================================================
                z_attr_a, z_res_a = self.encoder(x_a)
                z_quant_a, indices_a, vq_loss = self.codebook(z_attr_a, target_indices=y_a)

                # =================================================================
                # 2. Reconstruction Path
                # =================================================================
                x_rec = self.decoder(z_quant_a, z_res_a)
                loss_l1 = F.l1_loss(x_rec, x_a)
                loss_lpips = self.perceptual_loss(x_rec, x_a)
                loss_recon = self.w_recon_l1 * loss_l1 + self.w_recon_lpips * loss_lpips

                # =================================================================
                # 3. Attribute Swapping Path
                # =================================================================
                if bsz > 1:
                    perm = torch.roll(torch.arange(bsz, device=self.device), shifts=1)
                else:
                    perm = torch.arange(bsz, device=self.device)

                y_b = y_a[perm]
                z_quant_b = z_quant_a[perm]

                k_star = random.randint(0, num_clusters - 1)

                z_swapped = z_quant_a.clone()
                if random.random() < 0.5:
                    z_swapped[:, k_star, :] = self.codebook.lookup_index(k_star, y_b[:, k_star])
                else:
                    z_swapped[:, k_star, :] = z_quant_b[:, k_star, :]

                x_swap = self.decoder(z_swapped, z_res_a)

                # =================================================================
                # 4. Supervision & Generator Losses
                # =================================================================
                logits_swap, d_pred_fake_swap = self.supervisor(x_swap, return_attr=True, return_patch=True)
                _, d_pred_fake_rec = self.supervisor(x_rec, return_attr=False, return_patch=True)
                assert logits_swap is not None and d_pred_fake_swap is not None and d_pred_fake_rec is not None

                loss_swap_attr = x_a.new_tensor(0.0)
                for k in range(num_clusters):
                    target_k = y_b[:, k] if k == k_star else y_a[:, k]
                    weight_k = 2.0 if k == k_star else 1.0
                    loss_swap_attr = loss_swap_attr + weight_k * F.cross_entropy(logits_swap[k], target_k)
                loss_swap_attr = loss_swap_attr / float(num_clusters + 1)

                loss_adv_g = 0.5 * (
                    self._gan_loss(d_pred_fake_swap, target_is_real=True)
                    + self._gan_loss(d_pred_fake_rec, target_is_real=True)
                )

                # =================================================================
                # 5. Total Generator Step
                # =================================================================
                loss_g = (
                    loss_recon
                    + self.w_vq * vq_loss
                    + self.w_swap * loss_swap_attr
                    + self.w_adv * loss_adv_g
                )

            self.scaler.scale(loss_g).backward()
            self.scaler.step(self.opt_g)

            # =================================================================
            # 6. Discriminator & Supervisor Fine-Tuning Step
            # =================================================================
            self.supervisor.patch_disc.requires_grad_(True)
            if self.sup_finetune_scale > 0.0:
                self.supervisor.attr_trunk.requires_grad_(True)
                self.supervisor.attr_heads.requires_grad_(True)

            self.opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits_real, d_pred_real = self.supervisor(
                    x_a, return_attr=(self.sup_finetune_scale > 0.0), return_patch=True
                )
                _, d_pred_fake_rec_det = self.supervisor(x_rec.detach(), return_attr=False, return_patch=True)
                _, d_pred_fake_swap_det = self.supervisor(x_swap.detach(), return_attr=False, return_patch=True)
                assert d_pred_real is not None and d_pred_fake_rec_det is not None and d_pred_fake_swap_det is not None

                loss_d_patch = 0.5 * (
                    self._gan_loss(d_pred_real, target_is_real=True)
                    + 0.5 * (
                        self._gan_loss(d_pred_fake_rec_det, target_is_real=False)
                        + self._gan_loss(d_pred_fake_swap_det, target_is_real=False)
                    )
                )

                loss_d_total = loss_d_patch
                if logits_real is not None and self.sup_finetune_scale > 0.0:
                    loss_sup_real = self.compute_supervisor_ce(logits_real, y_a)
                    loss_d_total = loss_d_total + loss_sup_real

            self.scaler.scale(loss_d_total).backward()
            self.scaler.step(self.opt_d)
            self.scaler.update()

            # Track metrics
            with torch.no_grad():
                pred_k_star = logits_swap[k_star].argmax(dim=-1)
                swap_correct += int((pred_k_star == y_b[:, k_star]).sum().item())
                swap_total += int(bsz)

            metrics["loss_g"] += float(loss_g.item())
            metrics["loss_recon_l1"] += float(loss_l1.item())
            metrics["loss_recon_lpips"] += float(loss_lpips.item())
            metrics["loss_vq"] += float(vq_loss.item())
            metrics["loss_swap_attr"] += float(loss_swap_attr.item())
            metrics["loss_adv_g"] += float(loss_adv_g.item())
            metrics["loss_d"] += float(loss_d_total.item())
            num_batches += 1

            last_batch_vis = (x_a.detach(), x_a[perm].detach(), x_rec.detach(), x_swap.detach())

            if step % self.log_interval == 0 or step == len(loader):
                swap_acc = 100.0 * swap_correct / max(1, swap_total)
                print(
                    f"[Phase 2 | Epoch {epoch:03d} | Step {step:04d}/{len(loader):04d}] "
                    f"G: {loss_g.item():.4f} (L1: {loss_l1.item():.4f}, P: {loss_lpips.item():.4f}, "
                    f"VQ: {vq_loss.item():.4f}, Swap: {loss_swap_attr.item():.4f}) | "
                    f"D: {loss_d_total.item():.4f} | SwapAcc: {swap_acc:.1f}%"
                )

        for k in metrics:
            if k != "swap_target_acc":
                metrics[k] /= max(1, num_batches)
        metrics["swap_target_acc"] = 100.0 * swap_correct / max(1, swap_total)

        if last_batch_vis is not None and (epoch % self.vis_interval == 0 or epoch == 1):
            self.save_visual_grid(*last_batch_vis, epoch=epoch)

        return metrics

    def save_visual_grid(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        x_rec: torch.Tensor,
        x_swap: torch.Tensor,
        epoch: int,
    ) -> str:
        """
        Saves a visual comparison grid: Row 1 = x_A, Row 2 = x_rec, Row 3 = x_B (donor), Row 4 = x_swap.
        """
        n_show = min(4, x_a.size(0))
        grid = torch.cat([x_a[:n_show], x_rec[:n_show], x_b[:n_show], x_swap[:n_show]], dim=0)
        grid = (grid.clamp(-1.0, 1.0) + 1.0) * 0.5
        out_path = os.path.join(self.save_dir, f"recon_vs_swap_epoch_{epoch:03d}.png")
        save_image(grid, out_path, nrow=n_show)
        return out_path

    def save_checkpoint(self, filename: str, epoch: int, extra: Optional[dict] = None) -> str:
        path = os.path.join(self.save_dir, filename)
        state = {
            "epoch": epoch,
            "cfg": self.cfg,
            "cluster_order": self.cluster_order,
            "cluster_lexicon": self.cluster_lexicon,
            "encoder": self.encoder.state_dict(),
            "codebook": self.codebook.state_dict(),
            "decoder": self.decoder.state_dict(),
            "supervisor": self.supervisor.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "opt_sup_pretrain": self.opt_sup_pretrain.state_dict(),
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.codebook.load_state_dict(ckpt["codebook"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.supervisor.load_state_dict(ckpt["supervisor"])
        if "opt_g" in ckpt:
            self.opt_g.load_state_dict(ckpt["opt_g"])
        if "opt_d" in ckpt:
            self.opt_d.load_state_dict(ckpt["opt_d"])
        epoch = int(ckpt.get("epoch", 0))
        pretrain_epochs = int(self.cfg.get("optim", {}).get("pretrain_epochs", 15))
        if "opt_sup_pretrain" in ckpt and epoch < pretrain_epochs:
            self.opt_sup_pretrain.load_state_dict(ckpt["opt_sup_pretrain"])
        del ckpt
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return epoch
