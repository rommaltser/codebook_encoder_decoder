"""
Main training entry point for CoTyle Dictionary-Controlled Texture Autoencoder.
Supports two-phase training:
  - Phase 1 (Epochs 1..pretrain_epochs): Supervisor pre-training on real DTD images + pseudo-labels
  - Phase 2 (Epochs pretrain_epochs+1..max_epochs): Joint Autoencoder + Random Cluster Swapping + PatchGAN
"""

import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

from data.dataset import DTDTextureDataset, load_pseudo_labels_file
from data.lexicon import get_cluster_config
from engine.trainer import CoTyleTrainer
from models.codebook import FactorizedCodebook
from models.decoder import TextureDecoder
from models.encoder import TextureEncoder
from models.supervisor import DiscriminativeSupervisor


class SyntheticTextureDataset(Dataset):
    """
    Synthetic procedural texture dataset for fast end-to-end sanity checking
    when the full DTD dataset is not yet mounted.
    """

    def __init__(self, num_samples: int = 32, img_size: int = 256, num_clusters: int = 6, words_per_cluster: int = 24):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_clusters = num_clusters
        self.words_per_cluster = words_per_cluster
        g = torch.Generator().manual_seed(42)
        self.labels = torch.randint(0, words_per_cluster, (num_samples, num_clusters), generator=g, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.labels[idx]
        coords = torch.linspace(-3.14159, 3.14159, self.img_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        freq_x = float((y[0].item() % 6) + 2)
        freq_y = float((y[1].item() % 6) + 2)
        phase = float(y[2].item()) * 0.25

        r = torch.sin(freq_x * xx + phase) * torch.cos(freq_y * yy)
        g = torch.cos(freq_x * yy - phase)
        b = torch.sin((freq_x + freq_y) * 0.5 * (xx + yy))
        img = torch.stack([r, g, b], dim=0).clamp(-1.0, 1.0)
        return img, y


def build_models(cfg: dict) -> Tuple[TextureEncoder, FactorizedCodebook, TextureDecoder, DiscriminativeSupervisor]:
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss_weights", {})

    latent_dim = int(model_cfg.get("latent_dim", 128))
    res_channels = int(model_cfg.get("res_channels", 128))
    num_clusters = int(model_cfg.get("num_clusters", 6))
    words_per_cluster = int(model_cfg.get("words_per_cluster", 24))
    vq_beta = float(model_cfg.get("vq_beta", 0.25))
    align_codebook = bool(model_cfg.get("align_codebook", True))
    align_weight = float(loss_cfg.get("vq_align", 0.5)) if align_codebook else 0.0
    res_dropout = float(model_cfg.get("res_dropout", 0.1))
    sup_backbone = str(model_cfg.get("supervisor_backbone", "resnet50"))
    pretrained_sup = bool(model_cfg.get("pretrained_supervisor", False))

    encoder = TextureEncoder(
        in_channels=3,
        base_channels=64,
        res_channels=res_channels,
        latent_dim=latent_dim,
        num_clusters=num_clusters,
        res_dropout=res_dropout,
    )
    codebook = FactorizedCodebook(
        num_clusters=num_clusters,
        words_per_cluster=words_per_cluster,
        latent_dim=latent_dim,
        beta=vq_beta,
        align_weight=align_weight,
    )
    decoder = TextureDecoder(
        num_clusters=num_clusters,
        latent_dim=latent_dim,
        res_channels=res_channels,
        style_dim=latent_dim * 2,
        out_channels=3,
    )
    supervisor = DiscriminativeSupervisor(
        num_clusters=num_clusters,
        words_per_cluster=words_per_cluster,
        in_channels=3,
        backbone=sup_backbone,
        pretrained_backbone=pretrained_sup,
    )
    return encoder, codebook, decoder, supervisor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CoTyle Dictionary-Controlled Texture Autoencoder")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--img-dir", type=str, default=None, help="Override DTD image directory")
    parser.add_argument("--labels-json", type=str, default=None, help="Override pseudo-labels JSON/directory/zip path")
    parser.add_argument("--num-clusters", type=int, default=None, help="Override number of clusters (6 or 8)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--pretrain-epochs", type=int, default=None, help="Override Phase 1 supervisor pretrain epochs")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override total training epochs")
    parser.add_argument("--save-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable automatic resume from save_dir")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument(
        "--synthetic-sanity-check",
        action="store_true",
        help="Run a fast end-to-end sanity check on synthetic data (Phase 1 + Phase 2 + checkpointing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.img_dir is not None:
        cfg["data"]["img_dir"] = args.img_dir
    if args.labels_json is not None:
        cfg["data"]["labels_json"] = args.labels_json
    if args.num_clusters is not None:
        cfg["model"]["num_clusters"] = args.num_clusters
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.pretrain_epochs is not None:
        cfg["optim"]["pretrain_epochs"] = args.pretrain_epochs
    if args.max_epochs is not None:
        cfg["optim"]["max_epochs"] = args.max_epochs
    if args.save_dir is not None:
        cfg["optim"]["save_dir"] = args.save_dir

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda":
        # Enable TF32 and cuDNN autotuner for fast training on A100 / Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    num_clusters = int(cfg["model"].get("num_clusters", 6))
    words_per_cluster = int(cfg["model"].get("words_per_cluster", 24))
    img_size = int(cfg["data"].get("img_size", 256))
    batch_size = int(cfg["data"].get("batch_size", 16))
    num_workers = int(cfg["data"].get("num_workers", 4))

    if args.synthetic_sanity_check:
        print("[Sanity Check] Initializing synthetic procedural texture dataset...")
        cluster_order, cluster_lexicon = get_cluster_config(
            num_clusters=num_clusters, labels_dict=None, words_per_cluster=words_per_cluster
        )
        dataset: Dataset = SyntheticTextureDataset(
            num_samples=max(batch_size * 2, 16),
            img_size=img_size,
            num_clusters=num_clusters,
            words_per_cluster=words_per_cluster,
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        labels_dict = load_pseudo_labels_file(cfg["data"]["labels_json"])
        cluster_order, cluster_lexicon = get_cluster_config(
            num_clusters=num_clusters, labels_dict=labels_dict, words_per_cluster=words_per_cluster
        )
        full_dataset = DTDTextureDataset(
            img_dir=cfg["data"]["img_dir"],
            labels_json=cfg["data"]["labels_json"],
            cluster_lexicon=cluster_lexicon,
            cluster_order=cluster_order,
            img_size=img_size,
            is_train=True,
            preloaded_labels=labels_dict,
        )
        if len(full_dataset) == 0:
            raise RuntimeError(
                f"No valid DTD images matched between img_dir='{cfg['data']['img_dir']}' "
                f"and labels_json='{cfg['data']['labels_json']}'."
            )
        print(f"Loaded DTDTextureDataset with {len(full_dataset)} valid annotated textures across {num_clusters} clusters.")
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=(len(full_dataset) >= batch_size),
        )

    encoder, codebook, decoder, supervisor = build_models(cfg)
    trainer = CoTyleTrainer(
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        supervisor=supervisor,
        cfg=cfg,
        device=device,
        cluster_order=cluster_order,
        cluster_lexicon=cluster_lexicon,
    )

    start_epoch = 1
    save_dir = cfg["optim"].get("save_dir", "checkpoints")
    resume_path = args.resume

    # Automatically check save_dir for an existing checkpoint if --resume was not explicitly provided
    if not resume_path and not args.no_auto_resume:
        latest_cand = os.path.join(save_dir, "checkpoint_latest.pt")
        phase1_cand = os.path.join(save_dir, "checkpoint_phase1_supervisor.pt")
        if os.path.isfile(latest_cand):
            resume_path = latest_cand
        elif os.path.isfile(phase1_cand):
            resume_path = phase1_cand

    if resume_path and os.path.isfile(resume_path):
        start_epoch = trainer.load_checkpoint(resume_path) + 1
        print(f"Automatically resumed from checkpoint '{resume_path}' at epoch {start_epoch}")

    pretrain_epochs = int(cfg["optim"].get("pretrain_epochs", 15))
    max_epochs = int(cfg["optim"].get("max_epochs", 100))

    print(
        f"Starting training on {device}: "
        f"Phase 1 = Epochs 1..{pretrain_epochs}, Phase 2 = Epochs {pretrain_epochs + 1}..{max_epochs}"
    )

    for epoch in range(start_epoch, max_epochs + 1):
        if epoch <= pretrain_epochs:
            metrics = trainer.train_supervisor_epoch(train_loader, epoch)
            if epoch == pretrain_epochs:
                ckpt_path = trainer.save_checkpoint("checkpoint_phase1_supervisor.pt", epoch, extra={"metrics": metrics})
                print(f"Saved Phase 1 Supervisor checkpoint to: {ckpt_path}")
        else:
            metrics = trainer.train_joint_epoch(train_loader, epoch)
            ckpt_path = trainer.save_checkpoint("checkpoint_latest.pt", epoch, extra={"metrics": metrics})
            if epoch % 10 == 0 or epoch == max_epochs:
                trainer.save_checkpoint(f"checkpoint_epoch_{epoch:03d}.pt", epoch, extra={"metrics": metrics})
                print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
