"""
Model 05 — DINOv2 + LoRA + Differential Cross-Attention + Relation Head.

Designed to fit in 12 GB VRAM (AMD RX 6750 XT) while training:
- DINOv2 ViT-B/14 backbone stays FROZEN (86M params, no gradients).
- LoRA adapters (rank 8-16) are injected into the backbone's qkv + proj
  projections. Only LoRA matrices + heads are trainable (~3-8M params).
- Cross-attention uses the Differential Attention formulation
  (Ye et al., 2024, arXiv:2410.05258) to suppress spurious correspondences
  between unrelated face regions.
- Auxiliary relation head produces a multiclass logit (11 for FIW, 4 for
  KinFaceW) from a learnable relation query token, providing extra
  supervision alongside the primary binary verification task.

Forward:
    (face A, face B) ─► DINOv2 (frozen) + LoRA ─► patch tokens
                    ─► Differential bidirectional cross-attention (×2)
                    ─► pooling (mean + [REL] token)
                    ─► binary head  [e1, e2, |e1-e2|, e1*e2] → 1 logit
                    ─► relation head [REL] → K logits  (auxiliary)

Memory tricks:
- torch.utils.checkpoint on the backbone forward.
- ViT backbone held in fp16 at inference, fp32 weights with autocast at train.
- Cross-attention layers also gradient-checkpointed.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required for Model 05 (DINOv2 backbone). "
        "Install via `pip install 'timm>=0.9.0'`."
    ) from exc


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """
    Wraps an nn.Linear with a low-rank additive adapter.

    Output:  y = W x + (B A x) * (alpha / r)
    where W is frozen and A, B are the trainable low-rank factors.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert r > 0, "LoRA rank must be positive"
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features

        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.drop(x))) * self.scaling


def _inject_lora(
    module: nn.Module,
    target_names: Tuple[str, ...],
    r: int,
    alpha: int,
    dropout: float,
) -> int:
    """
    Recursively replace nn.Linear layers whose attribute name is in
    `target_names` with LoRALinear wrappers. Returns the count injected.
    """
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            count += 1
        else:
            count += _inject_lora(child, target_names, r, alpha, dropout)
    return count


# ---------------------------------------------------------------------------
# DINOv2 backbone with LoRA
# ---------------------------------------------------------------------------
class DINOv2LoRABackbone(nn.Module):
    """
    DINOv2 ViT loaded via timm, with LoRA adapters on attention projections.

    Returns patch tokens (B, N, D). Supports gradient checkpointing.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        img_size: int = 224,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_targets: Tuple[str, ...] = ("qkv", "proj"),
        pretrained: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.model_name = model_name

        # timm will pick the correct patch size and pretrained weights. We
        # override img_size so attention matrices stay small enough for 12 GB.
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
        )

        # Freeze the whole backbone first.
        for p in self.vit.parameters():
            p.requires_grad = False

        # Inject LoRA into attention projections on every transformer block.
        n_injected = _inject_lora(
            self.vit,
            target_names=lora_targets,
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if n_injected == 0:
            raise RuntimeError(
                f"No LoRA adapters injected — targets {lora_targets} not found in "
                f"{model_name}. Check timm's internal attribute names."
            )

        self.embed_dim = self.vit.embed_dim
        self.n_injected = n_injected

        if use_gradient_checkpointing and hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing(enable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            patch_tokens: (B, N, D) — excluding CLS / register tokens
        """
        feats = self.vit.forward_features(x)  # (B, N_total, D)
        # timm returns CLS (and possibly register tokens) prepended.
        # Peel them off using the model's own num_prefix_tokens attribute.
        num_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        return feats[:, num_prefix:]


# ---------------------------------------------------------------------------
# Differential cross-attention
# ---------------------------------------------------------------------------
class DifferentialCrossAttention(nn.Module):
    """
    Bidirectional differential cross-attention between two token sequences.

    For each direction (x1 attends to x2 and vice-versa), attention is computed
    as the difference of two softmax maps, each over half of the head channels:

        DiffAttn(Q, K, V) = (softmax(Q1 K1^T / √d) - λ · softmax(Q2 K2^T / √d)) V

    λ is a learnable per-head scalar, initialised following
    Ye et al., 2024 (arXiv:2410.05258):
        lambda_init(layer) = 0.8 - 0.6 * exp(-0.3 * (layer - 1))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        layer_idx: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert (dim // num_heads) % 2 == 0, "per-head dim must be even for diff-attn"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.half_head = self.head_dim // 2
        self.scale = self.half_head ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * max(layer_idx - 1, 0))
        self.lambda_init = lambda_init
        # One λ per head, parameterised as two low-rank vectors (as in the paper).
        self.lambda_q1 = nn.Parameter(torch.zeros(num_heads, self.half_head))
        self.lambda_k1 = nn.Parameter(torch.zeros(num_heads, self.half_head))
        self.lambda_q2 = nn.Parameter(torch.zeros(num_heads, self.half_head))
        self.lambda_k2 = nn.Parameter(torch.zeros(num_heads, self.half_head))
        nn.init.normal_(self.lambda_q1, std=0.1)
        nn.init.normal_(self.lambda_k1, std=0.1)
        nn.init.normal_(self.lambda_q2, std=0.1)
        nn.init.normal_(self.lambda_k2, std=0.1)

        self.attn_drop = nn.Dropout(dropout)
        self.group_norm = nn.GroupNorm(num_heads, dim)

    def _lambda(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        l1 = (self.lambda_q1 * self.lambda_k1).sum(dim=-1)
        l2 = (self.lambda_q2 * self.lambda_k2).sum(dim=-1)
        return (torch.exp(l1) - torch.exp(l2) + self.lambda_init).to(dtype).view(1, -1, 1, 1)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q, k, v: (B, H, N, head_dim). Returns (B, N, dim) after group-norm.
        """
        B, H, N, D = q.shape
        q1, q2 = q.split(self.half_head, dim=-1)
        k1, k2 = k.split(self.half_head, dim=-1)

        a1 = F.softmax((q1 @ k1.transpose(-2, -1)) * self.scale, dim=-1)
        a2 = F.softmax((q2 @ k2.transpose(-2, -1)) * self.scale, dim=-1)
        lam = self._lambda(q.device, q.dtype)
        attn = a1 - lam * a2
        attn = self.attn_drop(attn)

        out = attn @ v                                     # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, H * D)     # (B, N, dim)
        # Per-head group norm, as in the paper, to stabilise the difference.
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        return (1 - self.lambda_init) * out

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, dim) → (B, H, N, head_dim) for Q/K/V."""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self._project(self.q_proj(x1))
        k1 = self._project(self.k_proj(x1))
        v1 = self._project(self.v_proj(x1))

        q2 = self._project(self.q_proj(x2))
        k2 = self._project(self.k_proj(x2))
        v2 = self._project(self.v_proj(x2))

        # x1 attends to x2
        out1 = self._attend(q1, k2, v2)
        out1 = self.out_proj(out1)

        # x2 attends to x1
        out2 = self._attend(q2, k1, v1)
        out2 = self.out_proj(out2)

        return out1, out2


class CrossAttnBlock(nn.Module):
    """Differential cross-attention block with residual, LayerNorm and FFN."""

    def __init__(self, dim: int, num_heads: int, layer_idx: int, dropout: float = 0.1):
        super().__init__()
        self.norm1a = nn.LayerNorm(dim)
        self.norm1b = nn.LayerNorm(dim)
        self.attn = DifferentialCrossAttention(dim, num_heads, layer_idx, dropout)

        self.norm2a = nn.LayerNorm(dim)
        self.norm2b = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, a2 = self.attn(self.norm1a(x1), self.norm1b(x2))
        x1 = x1 + a1
        x2 = x2 + a2
        x1 = x1 + self.ffn(self.norm2a(x1))
        x2 = x2 + self.ffn(self.norm2b(x2))
        return x1, x2


# ---------------------------------------------------------------------------
# Kinship model
# ---------------------------------------------------------------------------
RELATION_SETS: Dict[str, Tuple[str, ...]] = {
    "fiw": ("bb", "ss", "sibs", "fd", "fs", "md", "ms", "gfgd", "gfgs", "gmgd", "gmgs"),
    "kinface": ("fd", "fs", "md", "ms"),
}


class DINOv2LoRAKinship(nn.Module):
    """
    Full Model 05 architecture — see module docstring.
    """

    def __init__(
        self,
        # Backbone
        backbone_name: str = "vit_base_patch14_dinov2.lvd142m",
        img_size: int = 224,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        backbone_pretrained: bool = True,
        use_gradient_checkpointing: bool = True,
        # Cross-attention
        embedding_dim: int = 512,
        cross_attn_layers: int = 2,
        cross_attn_heads: int = 8,
        dropout: float = 0.1,
        # Relation head
        relation_set: str = "fiw",
        relation_loss_weight: float = 0.2,
    ):
        super().__init__()

        self.backbone = DINOv2LoRABackbone(
            model_name=backbone_name,
            img_size=img_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            pretrained=backbone_pretrained,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.token_proj = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttnBlock(embedding_dim, cross_attn_heads, layer_idx=i + 1, dropout=dropout)
            for i in range(cross_attn_layers)
        ])

        # Learnable [REL] query token, one shared for both faces.
        self.rel_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.rel_token, std=0.02)

        self.pool_norm = nn.LayerNorm(embedding_dim)

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.binary_head = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        if relation_set not in RELATION_SETS:
            raise ValueError(f"unknown relation_set '{relation_set}'")
        self.relation_classes = RELATION_SETS[relation_set]
        self.relation_set = relation_set
        self.relation_loss_weight = relation_loss_weight

        self.relation_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.relation_classes)),
        )

        self.embedding_dim = embedding_dim

    # -- internals ----------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image into projected token sequence (B, 1 + N, D).

        A [REL] token is prepended so the relation head can read a learnt
        summary that depends on this face's tokens.
        """
        if self.use_gradient_checkpointing and self.training:
            tokens = grad_checkpoint(self.backbone, x, use_reentrant=False)
        else:
            tokens = self.backbone(x)
        tokens = self.token_proj(tokens)  # (B, N, D)

        B = tokens.size(0)
        rel = self.rel_token.expand(B, -1, -1)
        return torch.cat([rel, tokens], dim=1)  # (B, 1 + N, D)

    def _cross_attend(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.cross_attn_blocks:
            if self.use_gradient_checkpointing and self.training:
                z1, z2 = grad_checkpoint(block, z1, z2, use_reentrant=False)
            else:
                z1, z2 = block(z1, z2)
        return z1, z2

    # -- public API ---------------------------------------------------------
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, torch.Tensor]:
        z1 = self._encode(img1)  # (B, 1 + N, D)
        z2 = self._encode(img2)
        z1, z2 = self._cross_attend(z1, z2)

        z1 = self.pool_norm(z1)
        z2 = self.pool_norm(z2)

        # Patch tokens pooled by mean for the verification embedding.
        emb1 = F.normalize(self.projection(z1[:, 1:].mean(dim=1)), dim=-1)
        emb2 = F.normalize(self.projection(z2[:, 1:].mean(dim=1)), dim=-1)

        # Relation head reads the [REL] token from both faces.
        rel_in = torch.cat([z1[:, 0], z2[:, 0]], dim=-1)
        rel_logits = self.relation_head(rel_in)

        diff = (emb1 - emb2).abs()
        prod = emb1 * emb2
        combined = torch.cat([emb1, emb2, diff, prod], dim=-1)
        logits = self.binary_head(combined)

        return {
            "logits": logits,
            "emb1": emb1,
            "emb2": emb2,
            "relation_logits": rel_logits,
        }

    # -- utilities ----------------------------------------------------------
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def relation_to_index(self, relation_name: str) -> int:
        """Map a relation string to an index; returns -1 for unknown / non-kin."""
        try:
            return self.relation_classes.index(relation_name)
        except ValueError:
            return -1

    def relations_to_indices(self, relations: List[str]) -> torch.Tensor:
        return torch.tensor(
            [self.relation_to_index(r) for r in relations],
            dtype=torch.long,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_model(config=None) -> DINOv2LoRAKinship:
    if config is None:
        return DINOv2LoRAKinship()
    return DINOv2LoRAKinship(
        backbone_name=getattr(config, "backbone_name", "vit_base_patch14_dinov2.lvd142m"),
        img_size=getattr(config, "img_size", 224),
        lora_rank=getattr(config, "lora_rank", 8),
        lora_alpha=getattr(config, "lora_alpha", 16),
        lora_dropout=getattr(config, "lora_dropout", 0.0),
        backbone_pretrained=getattr(config, "backbone_pretrained", True),
        use_gradient_checkpointing=getattr(config, "use_gradient_checkpointing", True),
        embedding_dim=getattr(config, "embedding_dim", 512),
        cross_attn_layers=getattr(config, "cross_attn_layers", 2),
        cross_attn_heads=getattr(config, "cross_attn_heads", 8),
        dropout=getattr(config, "dropout", 0.1),
        relation_set=getattr(config, "relation_set", "fiw"),
        relation_loss_weight=getattr(config, "relation_loss_weight", 0.2),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = DINOv2LoRAKinship(
        backbone_pretrained=False,  # skip download when smoke-testing
        use_gradient_checkpointing=False,
    )
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    out = model(x1, x2)
    total = sum(p.numel() for p in model.parameters())
    train = model.count_trainable()
    print(f"logits:          {out['logits'].shape}")
    print(f"emb1:            {out['emb1'].shape}")
    print(f"relation_logits: {out['relation_logits'].shape}")
    print(f"total params:    {total:,}")
    print(f"trainable:       {train:,}  ({100 * train / total:.2f}%)")
