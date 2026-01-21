"""
Shared dataset loaders for kinship classification.
Supports FIW, KinFaceW-I, and KinFaceW-II datasets.
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import DataConfig


class KinshipPairDataset(Dataset):
    """
    Dataset for kinship verification that returns pairs of images.
    Supports multiple dataset formats: FIW, KinFaceW-I, KinFaceW-II.
    """
    
    def __init__(
        self,
        root_dir: str,
        dataset_type: str = "kinface",  # "fiw", "kinface"
        split: str = "train",
        relation_types: Optional[List[str]] = None,
        transform: Optional[T.Compose] = None,
        negative_ratio: float = 1.0,
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            dataset_type: Type of dataset ("fiw" or "kinface")
            split: Data split ("train", "val", "test")
            relation_types: List of relation types to include
            transform: Image transforms
            negative_ratio: Ratio of negative to positive pairs
        """
        self.root_dir = Path(root_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform
        self.negative_ratio = negative_ratio
        
        # Default relation types
        self.relation_types = relation_types or ["fd", "fs", "md", "ms"]
        
        # Load pairs
        self.pairs = []
        self.labels = []
        self._load_pairs()
    
    def _load_pairs(self):
        """Load image pairs based on dataset type."""
        if self.dataset_type == "kinface":
            self._load_kinface_pairs()
        elif self.dataset_type == "fiw":
            self._load_fiw_pairs()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_kinface_pairs(self):
        """Load KinFaceW format pairs."""
        relation_map = {
            "fd": "father-dau",
            "fs": "father-son",
            "md": "mother-dau",
            "ms": "mother-son",
        }
        
        positive_pairs = []
        all_images = []
        
        for rel in self.relation_types:
            if rel not in relation_map:
                continue
                
            rel_folder = relation_map[rel]
            images_dir = self.root_dir / "images" / rel_folder
            
            if not images_dir.exists():
                continue
            
            # Get all image pairs (format: XX_NNN_1.jpg, XX_NNN_2.jpg)
            images = sorted([f for f in images_dir.glob("*.jpg") if not f.name.startswith("Thumb")])
            
            # Group by pair ID
            pair_dict = {}
            for img_path in images:
                # Parse filename: fs_001_1.jpg -> pair_id=001, person=1
                parts = img_path.stem.split("_")
                if len(parts) >= 3:
                    pair_id = parts[1]
                    person_id = parts[2]
                    if pair_id not in pair_dict:
                        pair_dict[pair_id] = {}
                    pair_dict[pair_id][person_id] = str(img_path)
                    all_images.append(str(img_path))
            
            # Create positive pairs
            for pair_id, persons in pair_dict.items():
                if "1" in persons and "2" in persons:
                    positive_pairs.append((persons["1"], persons["2"], rel))
        
        # Create negative pairs (random pairing from different pairs)
        negative_pairs = []
        num_negatives = int(len(positive_pairs) * self.negative_ratio)
        
        for _ in range(num_negatives):
            img1, img2 = random.sample(all_images, 2)
            # Ensure they're not from the same pair
            if img1.split("_")[1] != img2.split("_")[1]:
                negative_pairs.append((img1, img2, "negative"))
        
        # Combine and create labels
        for img1, img2, rel in positive_pairs:
            self.pairs.append((img1, img2, rel))
            self.labels.append(1)
        
        for img1, img2, rel in negative_pairs:
            self.pairs.append((img1, img2, rel))
            self.labels.append(0)
        
        # Shuffle
        combined = list(zip(self.pairs, self.labels))
        random.shuffle(combined)
        self.pairs, self.labels = zip(*combined) if combined else ([], [])
        self.pairs = list(self.pairs)
        self.labels = list(self.labels)
    
    def _load_fiw_pairs(self):
        """Load FIW format pairs from CSV."""
        csv_path = self.root_dir / "FIW_PIDs_v2.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"FIW CSV not found: {csv_path}")
        
        # Load metadata
        df = pd.read_csv(csv_path, sep='\t')
        
        # Group by family
        families = df.groupby("FID")
        
        positive_pairs = []
        all_images = []
        
        for fid, family_df in families:
            # Get image paths for this family
            family_dir = self.root_dir / "FIDs" / fid
            if not family_dir.exists():
                continue
            
            # Find all images in family
            family_images = list(family_dir.rglob("*.jpg"))
            all_images.extend([str(img) for img in family_images])
            
            # Create positive pairs within family
            for i, img1 in enumerate(family_images):
                for img2 in family_images[i+1:]:
                    positive_pairs.append((str(img1), str(img2), "kin"))
        
        # Create negative pairs
        negative_pairs = []
        num_negatives = int(len(positive_pairs) * self.negative_ratio)
        
        if len(all_images) >= 2:
            for _ in range(num_negatives):
                img1, img2 = random.sample(all_images, 2)
                # Check they're from different families
                fid1 = Path(img1).parent.parent.name
                fid2 = Path(img2).parent.parent.name
                if fid1 != fid2:
                    negative_pairs.append((img1, img2, "non-kin"))
        
        # Combine
        for img1, img2, rel in positive_pairs:
            self.pairs.append((img1, img2, rel))
            self.labels.append(1)
        
        for img1, img2, rel in negative_pairs:
            self.pairs.append((img1, img2, rel))
            self.labels.append(0)
        
        # Shuffle
        combined = list(zip(self.pairs, self.labels))
        random.shuffle(combined)
        if combined:
            self.pairs, self.labels = zip(*combined)
            self.pairs = list(self.pairs)
            self.labels = list(self.labels)
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img1_path, img2_path, relation = self.pairs[idx]
        label = self.labels[idx]
        
        # Load images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        
        # Apply transforms
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
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ])
    else:
        return T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ])


def create_dataloaders(
    config: DataConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    dataset_type: str = "kinface",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    # Determine root directory based on dataset type
    if dataset_type == "kinface":
        root_dir = config.kinface_i_root
    else:
        root_dir = config.fiw_root
    
    # Get transforms
    train_transform = get_transforms(config, train=True)
    eval_transform = get_transforms(config, train=False)
    
    # Create datasets
    train_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        split="train",
        transform=train_transform,
    )
    
    val_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        split="val",
        transform=eval_transform,
    )
    
    test_dataset = KinshipPairDataset(
        root_dir=root_dir,
        dataset_type=dataset_type,
        split="test",
        transform=eval_transform,
    )
    
    # Create dataloaders
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
    # Test dataset loading
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
