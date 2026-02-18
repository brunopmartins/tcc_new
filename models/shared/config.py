"""
Shared configuration for all kinship classification models.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Dataset paths (absolute paths)
    fiw_root: str = "/home/bruno/Desktop/tcc_new/datasets/FIW"
    kinface_i_root: str = "/home/bruno/Desktop/tcc_new/datasets/KinFaceW-I"
    kinface_ii_root: str = "/home/bruno/Desktop/tcc_new/datasets/KinFaceW-II"
    fids_root: str = "/home/bruno/Desktop/tcc_new/datasets/FIW/FIDs"
    
    # Image settings
    image_size: int = 224
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Relation types
    relation_types: List[str] = field(default_factory=lambda: [
        "fd",  # father-daughter
        "fs",  # father-son
        "md",  # mother-daughter
        "ms",  # mother-son
        "bb",  # brother-brother
        "ss",  # sister-sister
        "sibs", # siblings (mixed)
        "gfgd", # grandfather-granddaughter
        "gfgs", # grandfather-grandson
        "gmgd", # grandmother-granddaughter
        "gmgs", # grandmother-grandson
    ])


@dataclass
class TrainConfig:
    """Training configuration."""
    # Basic training params
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter: float = 0.2
    random_rotation: int = 10
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "kinship-classification"


@dataclass
class ModelConfig:
    """Base model configuration."""
    # Backbone settings
    backbone: str = "arcface"  # "arcface", "facenet", "vit", "convnext"
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Embedding dimensions
    embedding_dim: int = 512
    projection_dim: int = 256
    
    # Dropout
    dropout: float = 0.1
    
    # Loss function
    loss_type: str = "contrastive"  # "bce", "contrastive", "triplet", "supcon"
    temperature: float = 0.07
    margin: float = 0.5


@dataclass
class AgeSynthesisConfig(ModelConfig):
    """Configuration for age synthesis model."""
    # Age synthesis settings
    use_age_synthesis: bool = True
    age_model: str = "sam"  # "sam", "hrfae", "agetransgan"
    num_age_variants: int = 3  # young, original, old
    target_ages: List[int] = field(default_factory=lambda: [20, 40, 60])
    
    # Aggregation
    aggregation: str = "attention"  # "max", "mean", "attention"


@dataclass
class ViTFaCoRConfig(ModelConfig):
    """Configuration for ViT + FaCoR model."""
    backbone: str = "vit"
    vit_model: str = "vit_base_patch16_224"
    
    # Cross-attention settings
    cross_attn_heads: int = 8
    cross_attn_layers: int = 2
    
    # Channel attention
    use_channel_attention: bool = True
    channel_reduction: int = 16


@dataclass
class ConvNeXtViTConfig(ModelConfig):
    """Configuration for ConvNeXt + ViT hybrid model."""
    # Dual backbone
    convnext_model: str = "convnext_base"
    vit_model: str = "vit_base_patch16_224"
    
    # Fusion settings
    fusion_type: str = "concat"  # "concat", "attention", "gated"
    convnext_dim: int = 1024
    vit_dim: int = 768


@dataclass
class UnifiedConfig(ModelConfig):
    """Configuration for unified model (all techniques combined)."""
    # Enable/disable components
    use_age_synthesis: bool = True
    use_hybrid_backbone: bool = True
    use_cross_attention: bool = True
    
    # Age synthesis
    age_model: str = "sam"
    num_age_variants: int = 3
    target_ages: List[int] = field(default_factory=lambda: [20, 40, 60])
    
    # Hybrid backbone
    convnext_model: str = "convnext_base"
    vit_model: str = "vit_base_patch16_224"
    fusion_type: str = "concat"
    
    # Cross-attention
    cross_attn_heads: int = 8
    cross_attn_layers: int = 2
    
    # Final aggregation
    age_aggregation: str = "attention"


def get_config(model_type: str = "base"):
    """Get configuration for specified model type."""
    configs = {
        "base": (DataConfig(), TrainConfig(), ModelConfig()),
        "age_synthesis": (DataConfig(), TrainConfig(), AgeSynthesisConfig()),
        "vit_facor": (DataConfig(), TrainConfig(), ViTFaCoRConfig()),
        "convnext_vit": (DataConfig(), TrainConfig(), ConvNeXtViTConfig()),
        "unified": (DataConfig(), TrainConfig(), UnifiedConfig()),
    }
    return configs.get(model_type, configs["base"])
