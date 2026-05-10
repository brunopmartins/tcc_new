"""
Model 07 — AdaFace + Stacking Architecture for Kinship Verification.

STATUS: SCAFFOLD. The classes below are interface stubs marking what needs
to be implemented. See README.md and IMPLEMENTATION_PLAN.md for the plan.

Design intent (when fully implemented):
- AdaFaceBackbone: wraps a pretrained AdaFace face recognition model.
  Loads weights via insightface or facenet-pytorch. Optionally adds LoRA
  on attention projections (rank=16) so the model can adapt without
  destroying pretrained features.
- ArcFacePairLoss (in losses.py): margin-based angular loss on cos(emb1, emb2).
- AdaFaceStackedKinship: combines AdaFace + LoRA + differential cross-attention
  (ported from M05) + auxiliary heads + ArcFace-pair output.

The differential cross-attention class is reused from M05 (import path
in __init__).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Reused components from M05
# ----------------------------------------------------------------------------
# Import the differential cross-attention block from M05 so M07 builds on the
# architectural contribution we already validated.
_M05_PATH = Path(__file__).resolve().parents[1] / "05_dinov2_lora_diffattn"
if str(_M05_PATH) not in sys.path:
    sys.path.insert(0, str(_M05_PATH))
try:
    from model import DifferentialCrossAttention, CrossAttnBlock  # type: ignore  # noqa: F401
except ImportError:
    # Allow the scaffold to import even before M05 is on the path (e.g.
    # during code-search by linters). Real training will fail loudly until
    # the path is correct.
    DifferentialCrossAttention = None
    CrossAttnBlock = None


# ----------------------------------------------------------------------------
# AdaFace backbone wrapper (TO IMPLEMENT)
# ----------------------------------------------------------------------------
class AdaFaceBackbone(nn.Module):
    """
    Wraps a pretrained AdaFace face recognition model as a feature extractor.

    Expected usage:
        backbone = AdaFaceBackbone(arch="ir50", pretrained=True, freeze=True)
        tokens = backbone(img)  # → (B, N_tokens, embed_dim)

    AdaFace typical setup:
        - Input: 112×112 face crop (aligned)
        - ResNet-based (IR50 ≈ 24M params, IR100 ≈ 65M params)
        - Output: 512-d global embedding

    For our use we also need patch tokens for cross-attention. Two options:
      A) Take the intermediate feature map before global pooling
         (e.g. for IR50 at 112×112: ~7×7 = 49 tokens of dim 512)
      B) Add a small upstream module that produces patch-grid tokens

    Option A is simpler — pick the final convolutional block's output
    before pooling.

    TODO:
        [ ] Load AdaFace via insightface.app or facenet-pytorch
        [ ] Hook the intermediate feature map for token output
        [ ] Add LoRA wrappers on attention projections (no attention in
            ResNet — adapt to applying LoRA on convolutional 1×1 layers
            in the bottleneck blocks instead)
        [ ] Add freeze_blocks(n) method for partial unfreeze
    """

    def __init__(
        self,
        arch: str = "ir50",
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
        lora_rank: int = 0,  # 0 = no LoRA
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.arch = arch
        self.freeze = freeze
        self.lora_rank = lora_rank
        # TODO: actually load the model.
        self._loaded = False
        self.embed_dim = 512  # AdaFace standard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 112, 112) → patch tokens (B, N, embed_dim)."""
        if not self._loaded:
            raise NotImplementedError(
                "AdaFaceBackbone is a scaffold. Implement loading via "
                "insightface or facenet-pytorch before training."
            )
        # TODO: forward through AdaFace ResNet, return intermediate features.
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Stacked model (TO IMPLEMENT)
# ----------------------------------------------------------------------------
RELATION_SETS: Dict[str, Tuple[str, ...]] = {
    "fiw": ("bb", "ss", "sibs", "fd", "fs", "md", "ms", "gfgd", "gfgs", "gmgd", "gmgs"),
    "kinface": ("fd", "fs", "md", "ms"),
}


class AdaFaceStackedKinship(nn.Module):
    """
    Full M07 architecture.

    Components:
        1. AdaFaceBackbone (frozen or LoRA)
        2. Token projection to common embedding dim
        3. Differential cross-attention (2 layers, from M05)
        4. Pair pooling → emb1, emb2
        5. Binary head (BCE-aux) + ArcFace-pair head (main loss)
        6. Optional: relation head for auxiliary supervision

    Args mirror M05's DINOv2LoRAKinship for code reuse downstream.

    TODO:
        [ ] Wire AdaFaceBackbone once it's implemented
        [ ] Implement forward (mostly copy from M05's DINOv2LoRAKinship,
            adapted for AdaFace's token shape and the new loss head)
        [ ] Add @torch.no_grad() decorators on backbone calls when frozen
        [ ] Save model_config in training checkpoint for clean test/eval load
    """

    def __init__(
        self,
        # Backbone
        adaface_arch: str = "ir50",
        adaface_pretrained_path: Optional[str] = None,
        freeze_backbone: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        # Architecture
        embedding_dim: int = 512,
        cross_attn_layers: int = 2,
        cross_attn_heads: int = 8,
        dropout: float = 0.1,
        # Heads
        relation_set: str = "fiw",
        relation_loss_weight: float = 0.0,  # default off
    ):
        super().__init__()
        raise NotImplementedError(
            "AdaFaceStackedKinship is a scaffold. See IMPLEMENTATION_PLAN.md "
            "Phase 2.3 — train R002. Should mostly mirror M05's "
            "DINOv2LoRAKinship architecture with the AdaFace backbone swap "
            "and ArcFace-pair head."
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


def create_model(config=None):
    raise NotImplementedError("See implementation plan.")
