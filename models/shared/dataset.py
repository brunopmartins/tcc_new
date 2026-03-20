"""
Shared dataset loaders for kinship classification.

The shared protocol uses deterministic, disjoint splits:
- KinFaceW: pair-disjoint splits by pair id
- FIW: family-disjoint splits by family id
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import DataConfig


SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2}


def _split_offset(split: str) -> int:
    return SPLIT_OFFSETS.get(split, 0)


def _shuffled_ids(ids: List[str], split_seed: int) -> List[str]:
    result = sorted(ids)
    random.Random(split_seed).shuffle(result)
    return result


def _split_id_sets(ids: List[str], split_seed: int) -> Dict[str, Set[str]]:
    ordered = _shuffled_ids(ids, split_seed)
    n = len(ordered)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {
        "train": set(ordered[:n_train]),
        "val": set(ordered[n_train:n_train + n_val]),
        "test": set(ordered[n_train + n_val:]),
    }


def _split_train_val_ids(ids: List[str], split_seed: int, val_ratio: float = 0.15) -> Tuple[Set[str], Set[str]]:
    ordered = _shuffled_ids(ids, split_seed)
    n_val = max(1, int(len(ordered) * val_ratio)) if len(ordered) > 1 else 0
    val_ids = set(ordered[:n_val])
    train_ids = set(ordered[n_val:])
    if not train_ids and val_ids:
        train_ids = set(list(val_ids)[1:])
        val_ids = {list(val_ids)[0]}
    return train_ids, val_ids


class KinshipPairDataset(Dataset):
    """
    Dataset for kinship verification that returns pairs of images.
    Supports FIW and KinFaceW.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_type: str = "kinface",
        split: str = "train",
        relation_types: Optional[List[str]] = None,
        transform: Optional[T.Compose] = None,
        negative_ratio: float = 1.0,
        negative_sampling_strategy: str = "random",
        split_seed: int = 42,
        explicit_pair_ids: Optional[set] = None,
        explicit_group_ids: Optional[set] = None,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform
        self.negative_ratio = negative_ratio
        self.negative_sampling_strategy = negative_sampling_strategy
        self.split_seed = split_seed
        self.explicit_group_ids = explicit_group_ids if explicit_group_ids is not None else explicit_pair_ids

        self.relation_types = relation_types or ["fd", "fs", "md", "ms"]
        self.pairs: List[Tuple[str, str, str]] = []
        self.labels: List[int] = []

        self._load_pairs()

    def _load_pairs(self):
        if self.dataset_type == "kinface":
            self._load_kinface_pairs()
        elif self.dataset_type == "fiw":
            self._load_fiw_pairs()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _load_kinface_pairs(self):
        """Load KinFaceW pairs using deterministic pair-disjoint splits."""
        relation_map = {
            "fd": "father-dau",
            "fs": "father-son",
            "md": "mother-dau",
            "ms": "mother-son",
        }

        positive_pairs = []
        all_images_by_pair: Dict[str, List[str]] = {}

        for rel in self.relation_types:
            if rel not in relation_map:
                continue

            images_dir = self.root_dir / "images" / relation_map[rel]
            if not images_dir.exists():
                continue

            pair_dict: Dict[str, Dict[str, str]] = {}
            for img_path in sorted(images_dir.glob("*.jpg")):
                if img_path.name.startswith("Thumb"):
                    continue
                parts = img_path.stem.split("_")
                if len(parts) < 3:
                    continue
                pair_id = f"{rel}_{parts[1]}"
                person_id = parts[2]
                pair_dict.setdefault(pair_id, {})[person_id] = str(img_path)
                all_images_by_pair.setdefault(pair_id, []).append(str(img_path))

            for pair_id, persons in pair_dict.items():
                if "1" in persons and "2" in persons:
                    positive_pairs.append((persons["1"], persons["2"], rel, pair_id))

        if self.explicit_group_ids is not None:
            active_ids = set(self.explicit_group_ids)
        else:
            split_ids = _split_id_sets(sorted({pair_id for *_, pair_id in positive_pairs}), self.split_seed)
            active_ids = split_ids.get(self.split, split_ids["train"])

        split_positive_records = [
            (img1, img2, rel, pair_id)
            for img1, img2, rel, pair_id in positive_pairs
            if pair_id in active_ids
        ]
        split_positive = [(img1, img2, rel) for img1, img2, rel, _ in split_positive_records]

        split_images = []
        for pair_id in active_ids:
            split_images.extend(all_images_by_pair.get(pair_id, []))

        if self.negative_sampling_strategy == "relation_matched":
            negative_pairs = self._sample_kinface_relation_matched_negatives(split_positive_records)
        else:
            negative_pairs = self._sample_kinface_negatives(split_images, len(split_positive))

        for img1, img2, rel in split_positive:
            self.pairs.append((img1, img2, rel))
            self.labels.append(1)

        for img1, img2, rel in negative_pairs:
            self.pairs.append((img1, img2, rel))
            self.labels.append(0)

        self._shuffle_pairs()

    def _sample_kinface_negatives(
        self,
        split_images: List[str],
        num_positive_pairs: int,
    ) -> List[Tuple[str, str, str]]:
        negative_pairs = []
        num_negatives = int(num_positive_pairs * self.negative_ratio)
        rng = random.Random(self.split_seed + 100 + _split_offset(self.split))

        attempts = 0
        seen = set()
        while len(negative_pairs) < num_negatives and attempts < max(num_negatives * 10, 100):
            attempts += 1
            if len(split_images) < 2:
                break
            img1, img2 = rng.sample(split_images, 2)
            pair1 = "_".join(Path(img1).stem.split("_")[:2])
            pair2 = "_".join(Path(img2).stem.split("_")[:2])
            key = tuple(sorted((img1, img2)))
            if pair1 != pair2 and key not in seen:
                seen.add(key)
                negative_pairs.append((img1, img2, "negative"))

        return negative_pairs

    def _sample_kinface_relation_matched_negatives(
        self,
        split_positive_records: List[Tuple[str, str, str, str]],
    ) -> List[Tuple[str, str, str]]:
        """
        Sample harder KinFace negatives by preserving relation structure.

        For example, an `fd` negative is built from a father image and a daughter
        image drawn from different `fd` pairs, rather than from arbitrary random
        identities across the split.
        """
        negative_pairs = []
        num_negatives = int(len(split_positive_records) * self.negative_ratio)
        rng = random.Random(self.split_seed + 150 + _split_offset(self.split))

        relation_role_pools: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        relation_choices: List[str] = []
        for img1, img2, rel, pair_id in split_positive_records:
            pools = relation_role_pools.setdefault(rel, {"1": [], "2": []})
            pools["1"].append((pair_id, img1))
            pools["2"].append((pair_id, img2))
            relation_choices.append(rel)

        attempts = 0
        seen = set()
        while len(negative_pairs) < num_negatives and attempts < max(num_negatives * 20, 100):
            attempts += 1
            if not relation_choices:
                break

            rel = rng.choice(relation_choices)
            pools = relation_role_pools.get(rel)
            if not pools:
                continue

            role1_pool = pools["1"]
            role2_pool = pools["2"]
            if len(role1_pool) < 2 or len(role2_pool) < 2:
                continue

            pair_id1, img1 = rng.choice(role1_pool)
            pair_id2, img2 = rng.choice(role2_pool)
            key = tuple(sorted((img1, img2)))
            if pair_id1 == pair_id2 or key in seen:
                continue

            seen.add(key)
            negative_pairs.append((img1, img2, "negative"))

        return negative_pairs

    def _load_fiw_pairs(self):
        """
        Load FIW pairs using family-disjoint splits.

        This fixes the previous protocol bug where train/val/test all saw the
        same families.
        """
        family_ids = get_fiw_family_ids(str(self.root_dir), split_seed=self.split_seed)
        if self.explicit_group_ids is not None:
            active_family_ids = set(self.explicit_group_ids)
        else:
            split_ids = _split_id_sets(family_ids, self.split_seed)
            active_family_ids = split_ids.get(self.split, split_ids["train"])

        positive_pairs = []
        images_by_family: Dict[str, List[str]] = {}

        for fid in sorted(active_family_ids):
            family_dir = self.root_dir / "FIDs" / fid
            if not family_dir.exists():
                continue

            family_images = sorted(str(img) for img in family_dir.rglob("*.jpg"))
            if len(family_images) < 2:
                continue

            images_by_family[fid] = family_images

            for i, img1 in enumerate(family_images):
                for img2 in family_images[i + 1:]:
                    positive_pairs.append((img1, img2, "kin"))

        negative_pairs = self._sample_fiw_negatives(images_by_family, len(positive_pairs))

        for img1, img2, rel in positive_pairs:
            self.pairs.append((img1, img2, rel))
            self.labels.append(1)

        for img1, img2, rel in negative_pairs:
            self.pairs.append((img1, img2, rel))
            self.labels.append(0)

        self._shuffle_pairs()

    def _sample_fiw_negatives(
        self,
        images_by_family: Dict[str, List[str]],
        num_positive_pairs: int,
    ) -> List[Tuple[str, str, str]]:
        negative_pairs = []
        num_negatives = int(num_positive_pairs * self.negative_ratio)
        rng = random.Random(self.split_seed + 200 + _split_offset(self.split))

        families = [fid for fid, images in images_by_family.items() if images]
        if len(families) < 2:
            return negative_pairs

        attempts = 0
        seen = set()
        while len(negative_pairs) < num_negatives and attempts < max(num_negatives * 10, 100):
            attempts += 1
            fid1, fid2 = rng.sample(families, 2)
            img1 = rng.choice(images_by_family[fid1])
            img2 = rng.choice(images_by_family[fid2])
            key = tuple(sorted((img1, img2)))
            if key not in seen:
                seen.add(key)
                negative_pairs.append((img1, img2, "non-kin"))

        return negative_pairs

    def _shuffle_pairs(self):
        combined = list(zip(self.pairs, self.labels))
        random.Random(self.split_seed + 300 + _split_offset(self.split)).shuffle(combined)
        self.pairs, self.labels = zip(*combined) if combined else ([], [])
        self.pairs = list(self.pairs)
        self.labels = list(self.labels)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img1_path, img2_path, relation = self.pairs[idx]
        label = self.labels[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            "img1": img1,
            "img2": img2,
            "label": torch.tensor(label, dtype=torch.float32),
            "relation": relation,
        }


def get_transforms(config: DataConfig, train: bool = True) -> T.Compose:
    """Get image transforms for training or evaluation."""
    if train:
        return T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ToTensor(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ])
    return T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


def create_dataloaders(
    config: DataConfig,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    dataset_type: str = "kinface",
    negative_ratio: Optional[float] = None,
    train_negative_ratio: Optional[float] = None,
    eval_negative_ratio: Optional[float] = None,
    train_negative_sampling_strategy: str = "random",
    eval_negative_sampling_strategy: str = "random",
    split_seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders using the shared protocol."""
    root_dir = config.kinface_i_root if dataset_type == "kinface" else config.fiw_root
    num_workers = config.num_workers if num_workers is None else num_workers
    negative_ratio = config.negative_ratio if negative_ratio is None else negative_ratio
    train_negative_ratio = negative_ratio if train_negative_ratio is None else train_negative_ratio
    eval_negative_ratio = negative_ratio if eval_negative_ratio is None else eval_negative_ratio
    split_seed = config.split_seed if split_seed is None else split_seed

    train_transform = get_transforms(config, train=True)
    eval_transform = get_transforms(config, train=False)

    train_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        split="train",
        transform=train_transform,
        negative_ratio=train_negative_ratio,
        negative_sampling_strategy=train_negative_sampling_strategy,
        split_seed=split_seed,
    )
    val_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        split="val",
        transform=eval_transform,
        negative_ratio=eval_negative_ratio,
        negative_sampling_strategy=eval_negative_sampling_strategy,
        split_seed=split_seed,
    )
    test_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        split="test",
        transform=eval_transform,
        negative_ratio=eval_negative_ratio,
        negative_sampling_strategy=eval_negative_sampling_strategy,
        split_seed=split_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_kinface_pair_ids(
    root_dir: str,
    relation_types: Optional[List[str]] = None,
    split_seed: int = 42,
) -> List[str]:
    """Return all KinFaceW pair ids in deterministic shuffled order."""
    relation_map = {
        "fd": "father-dau",
        "fs": "father-son",
        "md": "mother-dau",
        "ms": "mother-son",
    }
    relation_types = relation_types or ["fd", "fs", "md", "ms"]
    root = Path(root_dir)

    pair_ids = set()
    for rel in relation_types:
        if rel not in relation_map:
            continue
        images_dir = root / "images" / relation_map[rel]
        if not images_dir.exists():
            continue
        for img_path in sorted(images_dir.glob("*.jpg")):
            if img_path.name.startswith("Thumb"):
                continue
            parts = img_path.stem.split("_")
            if len(parts) >= 3:
                pair_ids.add(f"{rel}_{parts[1]}")

    return _shuffled_ids(list(pair_ids), split_seed)


def get_fiw_family_ids(
    root_dir: str,
    split_seed: int = 42,
) -> List[str]:
    """Return FIW family ids in deterministic shuffled order."""
    root = Path(root_dir)
    fids_dir = root / "FIDs"
    family_ids = set()

    if fids_dir.exists():
        family_ids.update(path.name for path in fids_dir.iterdir() if path.is_dir())

    csv_path = root / "FIW_PIDs_v2.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, sep="\t")
            if "FID" in df.columns:
                family_ids.update(str(fid) for fid in df["FID"].dropna().astype(str).tolist())
        except Exception:
            pass

    return _shuffled_ids(list(family_ids), split_seed)


def create_cv_fold_loaders(
    config: DataConfig,
    fold_k: int,
    n_folds: int = 5,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    dataset_type: str = "kinface",
    split_seed: int = 42,
    negative_ratio: Optional[float] = None,
    train_negative_ratio: Optional[float] = None,
    eval_negative_ratio: Optional[float] = None,
    train_negative_sampling_strategy: str = "random",
    eval_negative_sampling_strategy: str = "random",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) for one fold of k-fold CV.

    KinFaceW folds are pair-disjoint.
    FIW folds are family-disjoint.
    """
    root_dir = config.kinface_i_root if dataset_type == "kinface" else config.fiw_root
    num_workers = config.num_workers if num_workers is None else num_workers
    negative_ratio = config.negative_ratio if negative_ratio is None else negative_ratio
    train_negative_ratio = negative_ratio if train_negative_ratio is None else train_negative_ratio
    eval_negative_ratio = negative_ratio if eval_negative_ratio is None else eval_negative_ratio

    if dataset_type == "kinface":
        all_group_ids = get_kinface_pair_ids(root_dir, split_seed=split_seed)
    elif dataset_type == "fiw":
        all_group_ids = get_fiw_family_ids(root_dir, split_seed=split_seed)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    test_ids = {group_id for i, group_id in enumerate(all_group_ids) if i % n_folds == fold_k}
    fold_train_ids = [group_id for i, group_id in enumerate(all_group_ids) if i % n_folds != fold_k]
    train_ids, val_ids = _split_train_val_ids(fold_train_ids, split_seed + 500 + fold_k)

    train_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        split="train",
        transform=get_transforms(config, train=True),
        negative_ratio=train_negative_ratio,
        negative_sampling_strategy=train_negative_sampling_strategy,
        split_seed=split_seed + fold_k,
        explicit_group_ids=train_ids,
    )
    test_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        split="test",
        transform=get_transforms(config, train=False),
        negative_ratio=eval_negative_ratio,
        negative_sampling_strategy=eval_negative_sampling_strategy,
        split_seed=split_seed + fold_k,
        explicit_group_ids=test_ids,
    )
    val_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        split="val",
        transform=get_transforms(config, train=False),
        negative_ratio=eval_negative_ratio,
        negative_sampling_strategy=eval_negative_sampling_strategy,
        split_seed=split_seed + fold_k,
        explicit_group_ids=val_ids,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config = DataConfig()

    print("Testing KinFaceW-I dataset...")
    dataset = KinshipPairDataset(
        root_dir=config.kinface_i_root,
        dataset_type="kinface",
        transform=get_transforms(config, train=False),
    )

    print(f"Total pairs: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image 1 shape: {sample['img1'].shape}")
        print(f"Image 2 shape: {sample['img2'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Relation: {sample['relation']}")
