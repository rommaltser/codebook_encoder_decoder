"""
Verification & Inference Script for CoTyle Dictionary-Controlled Texture Editing.
Allows swapping a single semantic attribute cluster (e.g., element_shape -> hexagonal / striped / swirled)
using `codebook.lookup_index(target_cluster_idx, target_word_idx)`, verifying the edit
with the Discriminative Supervisor, and rendering the attribute string labels onto the output image.
"""

import argparse
import difflib
import glob
import os
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
import yaml

from data.lexicon import CLUSTER_LEXICON, CLUSTER_ORDER_6, CLUSTER_ORDER_8
from train import build_models


def resolve_image_path(image_arg: str, default_root: str = "/content/dtd/images") -> str:
    """
    Flexibly resolves an image path such as 'bubbly_0038.png', 'bubbly/bubbly_0038.jpg',
    or '/content/dtd/images/bubbly/bubbly_0038.png' even if the extension on disk is .jpg vs .png.
    """
    if os.path.isfile(image_arg):
        return image_arg

    base_no_ext, _ = os.path.splitext(image_arg)
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        if os.path.isfile(base_no_ext + ext):
            return base_no_ext + ext

    # Search by filename stem inside default_root and local dtd/images
    stem = os.path.splitext(os.path.basename(image_arg))[0]
    category_prefix = stem.split("_")[0] if "_" in stem else ""
    search_roots = [default_root, "dtd/images", "/content/dtd/images"]

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        if category_prefix:
            for ext in (".jpg", ".jpeg", ".png"):
                cand = os.path.join(root, category_prefix, stem + ext)
                if os.path.isfile(cand):
                    return cand
        matches = sorted(glob.glob(os.path.join(root, "**", stem + ".*"), recursive=True))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not find image matching '{image_arg}' (searched .jpg/.png and '{default_root}')")


def load_and_preprocess(img_path: str, img_size: int = 256) -> torch.Tensor:
    resolved = resolve_image_path(img_path)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = Image.open(resolved).convert("RGB")
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


def save_annotated_comparison(
    tensors: List[torch.Tensor],
    titles: List[str],
    out_path: str,
) -> None:
    """
    Saves a horizontal comparison strip with readable text labels drawn above each panel.
    """
    panels = []
    for t in tensors:
        img_t = ((t.squeeze(0).detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
        pil_img = Image.fromarray(img_t.permute(1, 2, 0).numpy(), mode="RGB")
        panels.append(pil_img)

    w, h = panels[0].size
    header_h = 46
    pad = 6
    total_w = len(panels) * w + (len(panels) + 1) * pad
    total_h = h + header_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (18, 24, 38))
    draw = ImageDraw.Draw(canvas)

    for i, (panel, title) in enumerate(zip(panels, titles)):
        x0 = pad + i * (w + pad)
        y0 = header_h + pad
        canvas.paste(panel, (x0, y0))
        lines = title.split("\n")
        for li, line in enumerate(lines):
            draw.text((x0 + 6, 6 + li * 18), line, fill=(240, 246, 252) if li == 0 else (56, 189, 248))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test single-attribute dictionary swap on a texture image")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path or filename of input texture image (e.g. bubbly_0038.png)")
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

    encoder.to(device).eval()
    codebook.to(device).eval()
    decoder.to(device).eval()
    supervisor.to(device).eval()

    num_clusters = int(cfg["model"].get("num_clusters", 6))
    cluster_order = ckpt.get("cluster_order") or (CLUSTER_ORDER_8 if num_clusters == 8 else CLUSTER_ORDER_6[:num_clusters])
    cluster_lexicon = ckpt.get("cluster_lexicon") or CLUSTER_LEXICON

    # Resolve cluster index (with fuzzy fallback)
    if args.cluster.isdigit():
        cluster_idx = int(args.cluster)
        cluster_name = cluster_order[cluster_idx]
    else:
        cluster_name = args.cluster.strip()
        if cluster_name not in cluster_order:
            close_c = difflib.get_close_matches(cluster_name, cluster_order, n=1, cutoff=0.4)
            if close_c:
                print(f"[Info] Cluster '{cluster_name}' matched to closest cluster '{close_c[0]}'")
                cluster_name = close_c[0]
            else:
                raise ValueError(f"Unknown cluster '{cluster_name}'. Available clusters: {cluster_order}")
        cluster_idx = cluster_order.index(cluster_name)

    # Resolve word index (with fuzzy fallback so typos like 'stiring' -> 'striped'/'swirled' work cleanly)
    words = cluster_lexicon[cluster_name]
    print(f"Available lexicon words for [{cluster_name}]: {words}")
    if args.word.isdigit():
        word_idx = int(args.word)
        word_name = words[word_idx]
    else:
        word_name = args.word.strip()
        if word_name not in words:
            close_w = difflib.get_close_matches(word_name, words, n=1, cutoff=0.35)
            if close_w:
                print(f"[Info] Word '{word_name}' not in lexicon; using closest match '{close_w[0]}'")
                word_name = close_w[0]
            else:
                raise ValueError(f"Unknown word '{word_name}' for cluster '{cluster_name}'. Available words: {words}")
        word_idx = words.index(word_name)

    resolved_img_path = resolve_image_path(args.image, default_root=cfg["data"].get("img_dir", "/content/dtd/images"))
    img_size = int(cfg["data"].get("img_size", 256))

    edited_img, pred_idx = run_texture_edit(
        encoder=encoder,
        codebook=codebook,
        decoder=decoder,
        supervisor=supervisor,
        img_path=resolved_img_path,
        target_cluster_idx=cluster_idx,
        target_word_idx=word_idx,
        img_size=img_size,
    )

    with torch.no_grad():
        orig_img = load_and_preprocess(resolved_img_path, img_size=img_size).unsqueeze(0).to(device)
        z_attr, z_res = encoder(orig_img)
        z_quant, orig_vq_indices, _ = codebook(z_attr)
        rec_img = decoder(z_quant, z_res)

        orig_logits, _ = supervisor(orig_img)
        rec_logits, _ = supervisor(rec_img)
        assert orig_logits is not None and rec_logits is not None
        orig_sup_idx = int(torch.argmax(orig_logits[cluster_idx], dim=-1).item())
        rec_sup_idx = int(torch.argmax(rec_logits[cluster_idx], dim=-1).item())
        orig_vq_idx = int(orig_vq_indices[0, cluster_idx].item())

    orig_word = words[orig_sup_idx] if 0 <= orig_sup_idx < len(words) else str(orig_sup_idx)
    orig_vq_word = words[orig_vq_idx] if 0 <= orig_vq_idx < len(words) else str(orig_vq_idx)
    rec_word = words[rec_sup_idx] if 0 <= rec_sup_idx < len(words) else str(rec_sup_idx)
    pred_word = words[pred_idx] if 0 <= pred_idx < len(words) else str(pred_idx)

    short_name = os.path.basename(resolved_img_path)
    titles = [
        f"Original ({short_name})\n{cluster_name}: {orig_word} (vq: {orig_vq_word})",
        f"Reconstructed\n{cluster_name}: {rec_word}",
        f"Edited -> {word_name}\nsup pred: {pred_word}",
    ]
    save_annotated_comparison([orig_img, rec_img, edited_img], titles, args.output)

    print(
        f"Image: {resolved_img_path}\n"
        f"Original [{cluster_name}]: sup='{orig_word}', codebook='{orig_vq_word}'\n"
        f"Edited   [{cluster_name}] -> '{word_name}' (idx={word_idx}) | Supervisor predicted: '{pred_word}' (idx={pred_idx})\n"
        f"Saved annotated comparison to: {args.output}"
    )


if __name__ == "__main__":
    main()
