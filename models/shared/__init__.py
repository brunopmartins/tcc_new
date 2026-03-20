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
    create_cv_fold_loaders,
    get_fiw_family_ids,
    get_kinface_pair_ids,
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
    collect_predictions,
    compute_metrics_from_predictions,
    evaluate_model,
    find_optimal_threshold,
    print_metrics,
)
from .trainer import (
    Trainer,
    train_model,
)
from .protocol import (
    PROTOCOL_VERSION,
    aggregate_numeric_metrics,
    apply_data_root_override,
    build_protocol_metadata,
    evaluate_with_validation_threshold,
    get_checkpoint_threshold,
    load_best_checkpoint,
    resolve_dataset_root,
    save_json,
    set_global_seed,
    update_checkpoint_payload,
    update_checkpoint_metadata,
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
    "create_cv_fold_loaders",
    "get_fiw_family_ids",
    "get_kinface_pair_ids",
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
    "collect_predictions",
    "compute_metrics_from_predictions",
    "evaluate_model",
    "find_optimal_threshold",
    "print_metrics",
    # Training
    "Trainer",
    "train_model",
    # Protocol
    "PROTOCOL_VERSION",
    "aggregate_numeric_metrics",
    "apply_data_root_override",
    "build_protocol_metadata",
    "evaluate_with_validation_threshold",
    "get_checkpoint_threshold",
    "load_best_checkpoint",
    "resolve_dataset_root",
    "save_json",
    "set_global_seed",
    "update_checkpoint_payload",
    "update_checkpoint_metadata",
]
