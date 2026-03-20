"""
Age Synthesis + All-vs-All Comparison Model for Kinship Verification.

Key Innovation:
- Generate multiple age variants (young, middle, old) of both input images
- Compare all age-matched pairs between the two individuals  
- Aggregate comparisons using attention weighting to reduce age-gap effects

Architecture:
    Input Pair → Age Synthesis → Multiple Age Variants
                      ↓
              All-vs-All Comparison (3×3 = 9 pairs)
                      ↓
              Attention-Weighted Aggregation
                      ↓
                 Kinship Score
"""
import os
import sys
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import timm

# Add SAM to path
SAM_PATH = Path(__file__).parent / "SAM"
if SAM_PATH.exists():
    sys.path.insert(0, str(SAM_PATH))


class AgeEncoder(nn.Module):
    """
    Age synthesis model using SAM (Style-based Age Manipulation).
    
    SAM uses StyleGAN2 to generate age-transformed faces while preserving identity.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self._initialized = False
        self.sam_model = None
        
        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = str(SAM_PATH / "pretrained_models" / "sam_ffhq_aging.pt")
        
        self.checkpoint_path = checkpoint_path
        
        # Try to load SAM if available
        if os.path.exists(checkpoint_path):
            self.load_pretrained(checkpoint_path)
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained SAM model."""
        try:
            from models.psp import pSp
            
            # Load checkpoint to get options
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            opts = ckpt['opts']
            
            # Update options for inference - FORCE CPU if no CUDA
            opts['checkpoint_path'] = checkpoint_path
            if not torch.cuda.is_available():
                opts['device'] = 'cpu'
            else:
                opts['device'] = self.device
            opts = Namespace(**opts)
            
            # Create SAM model
            self.sam_model = pSp(opts)
            self.sam_model.eval()

            # Register latent_avg as a buffer so model.to(device) moves it
            # (pSp stores it as a plain tensor that .to() would skip)
            if hasattr(self.sam_model, 'latent_avg') and isinstance(self.sam_model.latent_avg, torch.Tensor):
                lat = self.sam_model.latent_avg.data
                delattr(self.sam_model, 'latent_avg')
                self.sam_model.register_buffer('latent_avg', lat)

            self.sam_model.to(self.device)

            # Freeze all parameters
            for param in self.sam_model.parameters():
                param.requires_grad = False

            self._initialized = True
            print(f"SAM model loaded from {checkpoint_path}")
            
        except Exception as e:
            print(f"Warning: Could not load SAM model: {e}")
            import traceback
            traceback.print_exc()
            print("Age synthesis will use identity mapping (no aging effect)")
            self._initialized = False
    
    def _preprocess_for_sam(self, x: torch.Tensor, target_age: int) -> torch.Tensor:
        """
        Preprocess image for SAM input.
        SAM expects 4-channel input: RGB + age channel
        
        Args:
            x: Input image [B, 3, H, W] normalized to [-1, 1] or [0, 1]
            target_age: Target age (0-100)
        
        Returns:
            Preprocessed tensor [B, 4, 256, 256]
        """
        B = x.size(0)
        
        # Resize to 256x256 (SAM's expected input size)
        if x.size(-1) != 256 or x.size(-2) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1] if needed
        if x.min() >= 0:
            x = x * 2 - 1
        
        # Add age channel (normalized to [-1, 1])
        age_normalized = (target_age / 100.0) * 2 - 1  # Map 0-100 to -1 to 1
        age_channel = torch.full((B, 1, 256, 256), age_normalized, device=x.device, dtype=x.dtype)
        
        # Concatenate: [B, 4, 256, 256]
        x_with_age = torch.cat([x, age_channel], dim=1)
        
        return x_with_age
    
    def _postprocess_from_sam(self, x: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
        """
        Postprocess SAM output back to original format.
        
        Args:
            x: SAM output [B, 3, H, W]
            original_size: (H, W) of original input
        
        Returns:
            Processed tensor [B, 3, H, W] normalized to [0, 1]
        """
        # SAM outputs in [-1, 1], convert to [0, 1]
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        
        # Resize back to original size
        if x.size(-1) != original_size[1] or x.size(-2) != original_size[0]:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        target_age: int,
    ) -> torch.Tensor:
        """
        Generate face at target age using SAM.
        
        Args:
            x: Input face image [B, 3, H, W]
            target_age: Target age to generate (0-100)
        
        Returns:
            Age-transformed image [B, 3, H, W]
        """
        if not self._initialized or self.sam_model is None:
            # Fallback: return input unchanged
            return x

        original_size = (x.size(-2), x.size(-1))

        # Process one image at a time to avoid OOM — StyleGAN2 at 1024×1024
        # allocates ~4.5 GB intermediate buffers per batch, which OOMs on
        # 12 GB GPUs when combined with training optimizer states.
        results = []
        for i in range(x.size(0)):
            x_i = self._preprocess_for_sam(x[i:i+1], target_age)
            with torch.no_grad():
                out_i = self.sam_model(x_i, randomize_noise=False, resize=True)
            out_i = self._postprocess_from_sam(out_i, original_size)
            results.append(out_i)

        return torch.cat(results, dim=0)


class FeatureExtractor(nn.Module):
    """
    Face feature extractor backbone.
    Uses pretrained face recognition models.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == "arcface" or backbone == "resnet50":
            # Use timm ResNet as backbone
            self.backbone = timm.create_model(
                "resnet50",
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
            )
            backbone_dim = 2048
        elif backbone == "efficientnet":
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                num_classes=0,
            )
            backbone_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings."""
        features = self.backbone(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=1)


class PairComparator(nn.Module):
    """
    Compare a pair of face embeddings and produce similarity score.
    """
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        
        # Multiple comparison methods combined
        self.comparator = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),  # concat, diff, product, sum
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compare two embeddings.
        
        Args:
            emb1: First embedding [B, D]
            emb2: Second embedding [B, D]
        
        Returns:
            Similarity logits [B, 1]
        """
        # Multiple interaction features
        diff = emb1 - emb2
        product = emb1 * emb2
        combined = torch.cat([emb1, emb2, diff, product], dim=1)
        
        return self.comparator(combined)


class AgeAggregator(nn.Module):
    """
    Aggregate multiple age-comparison scores using attention.
    """
    
    def __init__(
        self,
        num_comparisons: int = 9,  # 3 ages × 3 ages
        hidden_dim: int = 64,
        aggregation: str = "attention",  # "attention", "max", "mean"
    ):
        super().__init__()
        self.aggregation = aggregation
        self.num_comparisons = num_comparisons
        
        if aggregation == "attention":
            # Learnable attention over age comparisons
            self.attention = nn.Sequential(
                nn.Linear(num_comparisons, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_comparisons),
            )
    
    def forward(self, comparison_scores: torch.Tensor) -> torch.Tensor:
        """
        Aggregate age comparison scores.
        
        Args:
            comparison_scores: Scores from all pairs [B, num_comparisons]
        
        Returns:
            Aggregated kinship score [B, 1]
        """
        if self.aggregation == "max":
            return comparison_scores.max(dim=1, keepdim=True)[0]
        elif self.aggregation == "mean":
            return comparison_scores.mean(dim=1, keepdim=True)
        else:  # attention
            weights = F.softmax(self.attention(comparison_scores), dim=1)
            return (comparison_scores * weights).sum(dim=1, keepdim=True)


class AgeSynthesisComparisonModel(nn.Module):
    """
    Full Age Synthesis + All-vs-All Comparison model.
    
    Pipeline:
    1. Generate age variants for both input faces
    2. Extract embeddings for all variants
    3. Compare all age-matched pairs
    4. Aggregate comparisons with learned attention
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dim: int = 512,
        target_ages: List[int] = [20, 40, 60],
        aggregation: str = "attention",
        use_age_synthesis: bool = True,
    ):
        super().__init__()
        
        self.target_ages = target_ages
        self.num_ages = len(target_ages)
        self.use_age_synthesis = use_age_synthesis
        
        # Age synthesis model (frozen during training)
        if use_age_synthesis:
            self.age_encoder = AgeEncoder()
            for param in self.age_encoder.parameters():
                param.requires_grad = False
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            embedding_dim=embedding_dim,
        )
        
        # Pair comparator
        self.pair_comparator = PairComparator(embedding_dim=embedding_dim)
        
        # Age aggregator (only needed when age synthesis is enabled)
        if use_age_synthesis:
            num_comparisons = self.num_ages * self.num_ages
            self.age_aggregator = AgeAggregator(
                num_comparisons=num_comparisons,
                aggregation=aggregation,
            )
        else:
            self.age_aggregator = None
    
    @torch.no_grad()
    def generate_age_variants(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Generate multiple age variants of input face."""
        if not self.use_age_synthesis:
            return [x]
        
        variants = []
        for age in self.target_ages:
            variant = self.age_encoder(x, age)
            variants.append(variant)
        return variants
    
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with age synthesis and all-vs-all comparison.
        
        Args:
            img1: First face image [B, 3, H, W]
            img2: Second face image [B, 3, H, W]
        
        Returns:
            kinship_score: Final kinship prediction [B, 1]
            comparison_matrix: Individual comparison scores [B, num_ages^2]
        """
        batch_size = img1.size(0)
        
        # Generate age variants
        variants1 = self.generate_age_variants(img1)
        variants2 = self.generate_age_variants(img2)
        
        # Extract embeddings for all variants
        embeddings1 = [self.feature_extractor(v) for v in variants1]
        embeddings2 = [self.feature_extractor(v) for v in variants2]
        
        # All-vs-all comparison
        comparison_scores = []
        for emb1 in embeddings1:
            for emb2 in embeddings2:
                score = self.pair_comparator(emb1, emb2)
                comparison_scores.append(score)
        
        # Stack comparison scores
        comparison_matrix = torch.cat(comparison_scores, dim=1)  # [B, num_ages^2]
        
        # Aggregate with attention (or return directly if no age synthesis)
        if self.age_aggregator is not None:
            kinship_score = self.age_aggregator(comparison_matrix)
        else:
            kinship_score = comparison_matrix  # Already [B, 1] when no age synthesis
        
        return kinship_score, comparison_matrix
    
    def get_embeddings(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get raw embeddings (for contrastive losses)."""
        emb1 = self.feature_extractor(img1)
        emb2 = self.feature_extractor(img2)
        return emb1, emb2


def create_model(config=None) -> AgeSynthesisComparisonModel:
    """Factory function to create model from config."""
    if config is None:
        return AgeSynthesisComparisonModel()
    
    return AgeSynthesisComparisonModel(
        backbone=getattr(config, "backbone", "resnet50"),
        embedding_dim=getattr(config, "embedding_dim", 512),
        target_ages=getattr(config, "target_ages", [20, 40, 60]),
        aggregation=getattr(config, "aggregation", "attention"),
        use_age_synthesis=getattr(config, "use_age_synthesis", True),
    )


if __name__ == "__main__":
    # Test model
    model = AgeSynthesisComparisonModel(use_age_synthesis=False)  # Disable for testing
    
    # Dummy input
    img1 = torch.randn(4, 3, 224, 224)
    img2 = torch.randn(4, 3, 224, 224)
    
    # Forward pass
    score, comparisons = model(img1, img2)
    
    print(f"Kinship score shape: {score.shape}")
    print(f"Comparison matrix shape: {comparisons.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
