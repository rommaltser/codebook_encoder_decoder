"""
Verification & Inference Script for CoTyle Dictionary-Controlled Texture Editing.
Allows swapping a single semantic attribute cluster (e.g., element_shape -> hexagonal)
using `codebook.lookup_index(target_cluster_idx, target_word_idx)` and verifying the edit
with the Discriminative Supervisor.
"""

import argparse
import os
from typing import Tuple

from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import yaml

from data.lexicon import CLUSTER_LEXICON, CLUSTER_ORDER_6, CLUSTER_ORDER_8
from train import build_models


def load_and_preprocess(img_path: str, img_size: int = 256) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)


def run_texture_edit(
    encoder: torch.nn.Module,
    codebook: torch.nn.Module,
    decoder: torch.nn.Module,
    supervisor: torch.nn.Module,
    img_path: str,
    target_cluster_idx: int,
    target_word_idx: int,
    img_size: int = 256,
) -> Tuple[torch.Tensor, int]:
    """
    Encodes `img_path`, replaces `target_cluster_idx` with the codebook embedding for `target_word_idx`,
    decodes the edited image, and returns `(edited_img, pred_idx)` evaluated by the supervisor.
    """
    encoder.eval()
    codebook.eval()
    decoder.eval()
    supervisor.eval()

    device = next(encoder.parameters()).device

    with torch.no_grad():
        img_tensor = load_and_preprocess(img_path, img_size=img_size).unsqueeze(0).to(device)  # [1, 3, 256, 256]

        # 1. Encode base texture
        z_attr, z_res = encoder(img_tensor)
        z_quant, _, _ = codebook(z_attr)

        # 2. Swap target attribute (e.g. element_shape -> hexagonal)
        z_edited = z_quant.clone()
        z_edited[:, target_cluster_idx, :] = codebook.lookup_index(target_cluster_idx, target_word_idx)

        # 3. Decode
        edited_img = decoder(z_edited, z_res)

        # 4. Supervisor check
        logits, _ = supervisor(edited_img)
        assert logits is not None
        pred_idx = int(torch.argmax(logits[target_cluster_idx], dim=-1).item())

    return edited_img, pred_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Test single-attribute dictionary swap on a texture image")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input texture image")
    parser.add_argument("--cluster", type=str, default="element_shape", help="Cluster name or integer index (0..K-1)")
    parser.add_argument("--word", type=str, default="hexagonal", help="Target lexicon word or integer index (0..23)")
    parser.add_argument("--output", type=str, default="edited_comparison.png", help="Output PNG path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]

    encoder, codebook, decoder, supervisor = build_models(cfg)
    encoder.load_state_dict(ckpt["encoder"])
    codebook.load_state_dict(ckpt["codebook"])
    decoder.load_state_dict(ckpt["decoder"])
    supervisor.load_state_dict(ckpt["supervisor"])

    encoder.to(device)
    codebook.to(device)
    decoder.to(device)
    supervisor.to(device)

    num_clusters = int(cfg["model"].get("num_clusters", 6))
    cluster_order = ckpt.get("cluster_order") or (CLUSTER_ORDER_8 if num_clusters == 8 else CLUSTER_ORDER_6[:num_clusters])
    cluster_lexicon = ckpt.get("cluster_lexicon") or CLUSTER_LEXICON

    # Resolve cluster index
    if args.cluster.isdigit():
        cluster_idx = int(args.cluster)
        cluster_name = cluster_order[cluster_idx]
    else:
        cluster_name = args.cluster
        if cluster_name not in cluster_order:
            raise ValueError(f"Unknown cluster '{cluster_name}'. Available clusters: {cluster_order}")
        cluster_idx = cluster_order.index(cluster_name)

    # Resolve word index
    words = cluster_lexicon[cluster_name]
    if args.word.isdigit():
        word_idx = int(args.word)
        word_name = words[word_idx]
    else:
        word_name = args.word
        if word_name not in words:
            raise ValueError(f"Unknown word '{word_name}' for cluster '{cluster_name}'. Available words: {words}")
        word_idx = words.index(word_name)

    img_size = int(cfg["data"].get("img_size", 256))
    edited_img, pred_idx = run_texture_edit(
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        supervisor=supervisor,
        img_path=args.image,
        target_cluster_idx=cluster_idx,
        target_word_idx=word_idx,
        img_size=img_size,
    )

    with torch.no_grad():
        orig_img = load_and_preprocess(args.image, img_size=img_size).unsqueeze(0).to(device)
        z_attr, z_res = encoder(orig_img)
        z_quant, _, _ = codebook(z_attr)
        rec_img = decoder(z_quant, z_res)

    comparison = torch.cat([orig_img, rec_img, edited_img], dim=0)
    comparison = (comparison.clamp(-1.0, 1.0) + 1.0) * 0.5
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_image(comparison, args.output, nrow=3)

    pred_word = words[pred_idx] if 0 <= pred_idx < len(words) else str(pred_idx)
    print(
        f"Edited [{cluster_name} -> {word_name} (idx={word_idx})] | "
        f"Supervisor predicted: {pred_word} (idx={pred_idx}) | "
        f"Saved side-by-side (Original | Reconstructed | Edited) to: {args.output}"
    )


if __name__ == "__main__":
    main()

