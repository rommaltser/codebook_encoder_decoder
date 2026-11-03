import glob
import json
import os
import zipfile
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _normalize_labels_payload(raw_data) -> Dict[str, Dict[str, str]]:
    """
    Normalizes different JSON schemas from VLM pseudo-labeling into:
        { "rel_path.jpg": { "cluster_1": "word", ... } }
    """
    if isinstance(raw_data, dict):
        # Check if nested under a top-level key like "annotations", "labels", "results", "data"
        for wrapper_key in ("annotations", "labels", "results", "samples", "data"):
            if wrapper_key in raw_data and isinstance(raw_data[wrapper_key], (dict, list)):
                return _normalize_labels_payload(raw_data[wrapper_key])

        normalized: Dict[str, Dict[str, str]] = {}
        for k, v in raw_data.items():
            if isinstance(v, dict):
                # Check if attributes are nested under "attributes" or "clusters"
                attrs = v.get("attributes") or v.get("clusters") or v.get("labels") or v
                if isinstance(attrs, dict):
                    normalized[str(k)] = {
                        str(ck): str(cv).strip()
                        for ck, cv in attrs.items()
                        if isinstance(cv, str)
                    }
        return normalized

    if isinstance(raw_data, list):
        normalized = {}
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            rel_path = (
                item.get("image_path")
                or item.get("rel_path")
                or item.get("path")
                or item.get("file_name")
                or item.get("image")
            )
            if not rel_path:
                continue
            attrs = item.get("attributes") or item.get("clusters") or item.get("labels") or item
            if isinstance(attrs, dict):
                normalized[str(rel_path)] = {
                    str(ck): str(cv).strip()
                    for ck, cv in attrs.items()
                    if isinstance(cv, str) and ck not in ("image_path", "rel_path", "path", "file_name", "image")
                }
        return normalized

    raise ValueError(f"Unsupported pseudo-label JSON structure: {type(raw_data)}")


def load_pseudo_labels_file(labels_path: str) -> Dict[str, Dict[str, str]]:
    """
    Loads pseudo-labels from a .json file, .jsonl file, directory containing .json/.jsonl,
    or a downloaded .zip archive (e.g. dtd-pseudo-labled-Qwen2.5-VL-32B-AWQ-*.zip).
    """
    labels_path = os.path.expanduser(labels_path)

    if os.path.isdir(labels_path):
        candidates = sorted(
            glob.glob(os.path.join(labels_path, "**", "*.json"), recursive=True)
            + glob.glob(os.path.join(labels_path, "**", "*.jsonl"), recursive=True)
        )
        if not candidates:
            raise FileNotFoundError(f"No .json or .jsonl files found inside directory: {labels_path}")
        labels_path = candidates[0]

    elif not os.path.exists(labels_path):
        # Check if a matching .zip exists
        zip_candidates = sorted(glob.glob(labels_path + "*.zip"))
        if zip_candidates:
            labels_path = zip_candidates[0]
        else:
            raise FileNotFoundError(f"Pseudo-labels path not found: {labels_path}")

    if zipfile.is_zipfile(labels_path):
        with zipfile.ZipFile(labels_path, "r") as zf:
            json_members = [m for m in zf.namelist() if m.endswith(".json") or m.endswith(".jsonl")]
            if not json_members:
                raise FileNotFoundError(f"No .json/.jsonl member found inside zip archive: {labels_path}")
            member = json_members[0]
            with zf.open(member) as f:
                text = f.read().decode("utf-8")
                if member.endswith(".jsonl"):
                    raw_data = [json.loads(line) for line in text.splitlines() if line.strip()]
                else:
                    raw_data = json.loads(text)
        return _normalize_labels_payload(raw_data)

    with open(labels_path, "r", encoding="utf-8") as f:
        if labels_path.endswith(".jsonl"):
            raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            raw_data = json.load(f)

    return _normalize_labels_payload(raw_data)


class DTDTextureDataset(Dataset):
    """
    Loads DTD image patches and maps the pseudo-labeled text strings from the JSON file
    into integer tensor class indices of shape [K].
    """

    def __init__(
        self,
        img_dir: str,
        labels_json: str,
        cluster_lexicon: Dict[str, List[str]],
        cluster_order: List[str],
        img_size: int = 256,
        is_train: bool = True,
        preloaded_labels: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.img_dir = os.path.expanduser(img_dir)
        self.labels = preloaded_labels if preloaded_labels is not None else load_pseudo_labels_file(labels_json)

        self.cluster_order = list(cluster_order)
        self.cluster_lexicon = cluster_lexicon
        self.w2i: Dict[str, Dict[str, int]] = {
            c: {w: i for i, w in enumerate(cluster_lexicon[c])} for c in self.cluster_order
        }

        # Filter valid samples that have complete annotations in the lexicon and exist on disk
        self.samples: List[str] = []
        for rel_path, attr_dict in self.labels.items():
            if not all(attr_dict.get(c) in self.w2i[c] for c in self.cluster_order):
                continue
            resolved = self._resolve_img_path(rel_path)
            if resolved is not None:
                self.samples.append(rel_path)

        if is_train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def _resolve_img_path(self, rel_path: str) -> Optional[str]:
        candidates = [
            os.path.join(self.img_dir, rel_path),
            rel_path,
        ]
        if rel_path.startswith("images/"):
            candidates.append(os.path.join(self.img_dir, rel_path[len("images/"):]))
        else:
            candidates.append(os.path.join(self.img_dir, "images", rel_path))

        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rel_path = self.samples[idx]
        img_path = self._resolve_img_path(rel_path)
        if img_path is None:
            raise FileNotFoundError(f"Could not find image file for sample '{rel_path}' under '{self.img_dir}'")

        img = Image.open(img_path).convert("RGB")
        attr_indices = [
            self.w2i[c][self.labels[rel_path][c]]
            for c in self.cluster_order
        ]
        return self.transform(img), torch.tensor(attr_indices, dtype=torch.long)

