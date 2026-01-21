"""
Shared utilities for kinship classification models.
"""
from .config import (
    DataConfig,
    TrainConfig,
    ModelConfig,
    AgeSynthesisConfig,
    ViTFaCoRConfig,
    ConvNeXtViTConfig,
    UnifiedConfig,
    get_config,
)
from .dataset import (
    KinshipPairDataset,
    get_transforms,
    create_dataloaders,
)
from .losses import (
    ContrastiveLoss,
    CosineContrastiveLoss,
    TripletLoss,
    FairContrastiveLoss,
    RelationGuidedContrastiveLoss,
    CombinedLoss,
    get_loss,
)
from .evaluation import (
    KinshipMetrics,
    evaluate_model,
    find_optimal_threshold,
    print_metrics,
)
from .trainer import (
    Trainer,
    train_model,
)

__all__ = [
    # Config
    "DataConfig",
    "TrainConfig", 
    "ModelConfig",
    "AgeSynthesisConfig",
    "ViTFaCoRConfig",
    "ConvNeXtViTConfig",
    "UnifiedConfig",
    "get_config",
    # Dataset
    "KinshipPairDataset",
    "get_transforms",
    "create_dataloaders",
    # Losses
    "ContrastiveLoss",
    "CosineContrastiveLoss",
    "TripletLoss",
    "FairContrastiveLoss",
    "RelationGuidedContrastiveLoss",
    "CombinedLoss",
    "get_loss",
    # Evaluation
    "KinshipMetrics",
    "evaluate_model",
    "find_optimal_threshold",
    "print_metrics",
    # Training
    "Trainer",
    "train_model",
]
