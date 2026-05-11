"""
Age-augmented kinship pair dataset for Model 10.

Wraps the shared `KinshipPairDataset` so that, for each face in a pair, it
also loads SAM-generated age variants (default: ages 8, 25, 70 — child /
young adult / elderly). The returned `img1` / `img2` tensors have shape
`(1 + N_ages, 3, H, W)`: the first slice is the original face, the rest are
aged variants in the order given by `target_ages`.

The M10 model (`AdaFaceFaCoRKinship.extract_tokens`) detects the extra leading
dimension and performs a weighted ensemble at the token level, so the rest
of the training/eval pipeline stays untouched.

Age variants are expected at `<age_augment_root>/<rel_path>/<stem>__age_<N>.jpg`,
mirroring the layout produced by `tools/sam_age_augment.py`. When a variant
is missing the original is used as a stand-in so the tensor shape is preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms as T

# Reuse the shared dataset
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from dataset import KinshipPairDataset  # noqa: E402


DEFAULT_TARGET_AGES: List[int] = [8, 25, 70]


class AgeAugmentedKinshipPairDataset(KinshipPairDataset):
    """KinshipPairDataset that also loads SAM age-augmented variants per face."""

    def __init__(
        self,
        *args,
        age_augment_root: Optional[str] = None,
        target_ages: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.age_augment_root = Path(age_augment_root) if age_augment_root else None
        self.target_ages = list(target_ages) if target_ages else list(DEFAULT_TARGET_AGES)

    def _aged_paths(self, img_path: str) -> List[Optional[Path]]:
        """Return list of aged variant paths for img_path (None if missing)."""
        if self.age_augment_root is None:
            return [None] * len(self.target_ages)

        # SAM was run against the aligned dataset, so we anchor on aligned_root
        # when set; otherwise fall back to root_dir.
        anchor = self.aligned_root if self.aligned_root else self.root_dir
        try:
            rel = Path(img_path).relative_to(anchor)
        except ValueError:
            return [None] * len(self.target_ages)

        stem = rel.stem
        suffix = rel.suffix or ".jpg"
        parent_rel = rel.parent
        out: List[Optional[Path]] = []
        for age in self.target_ages:
            candidate = (
                self.age_augment_root / parent_rel / f"{stem}__age_{age}{suffix}"
            )
            out.append(candidate if candidate.exists() else None)
        return out

    def _load_face_stack(self, img_path: str) -> torch.Tensor:
        """Load (original + N_ages aged) → (1+N, 3, H, W) after self.transform."""
        orig = Image.open(img_path).convert("RGB")
        imgs: List[Image.Image] = [orig]
        for aged_path in self._aged_paths(img_path):
            if aged_path is not None:
                try:
                    imgs.append(Image.open(aged_path).convert("RGB"))
                    continue
                except Exception:
                    pass
            # Missing variant — duplicate the original to preserve tensor shape.
            imgs.append(orig)

        if self.transform:
            tensors = [self.transform(i) for i in imgs]
        else:
            tensors = [T.ToTensor()(i) for i in imgs]
        return torch.stack(tensors, dim=0)

    def __getitem__(self, idx: int):
        img1_path, img2_path, relation = self.pairs[idx]
        label = self.labels[idx]

        img1_path = self._maybe_remap_aligned(img1_path)
        img2_path = self._maybe_remap_aligned(img2_path)

        img1 = self._load_face_stack(img1_path)
        img2 = self._load_face_stack(img2_path)

        return {
            "img1": img1,
            "img2": img2,
            "label": torch.tensor(label, dtype=torch.float32),
            "relation": relation,
        }


def parse_target_ages(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]
