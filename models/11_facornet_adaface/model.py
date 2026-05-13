"""
Model 11 — AdaFace + FaCoR Cross-Attention for Kinship Verification.

Replicates Model 02's recipe (best-performing model so far: Test ROC AUC =
0.850) but swaps the ImageNet ViT-B/16 backbone for **AdaFace IResNet-101**
pretrained on WebFace4M (face-discriminative features, robust to low-quality
inputs).

Key design choices vs. M02:

| Aspect          | M02 (ViT-B/16)                      | M10 (AdaFace IR-101)               |
|-----------------|-------------------------------------|------------------------------------|
| Backbone        | timm vit_base_patch16_224 (86M)     | AdaFaceIR101 (65M)                 |
| Pretrain        | ImageNet-21k                        | WebFace4M (AdaFace loss)           |
| Input           | 224×224, ImageNet norm              | 112×112, [-1,1] norm               |
| Tokens          | 196 patch tokens (16×16) of 768-d    | 49 spatial tokens (7×7) of 512-d   |
| Extra context   | CLS token                            | Pooled embedding (output_layer)    |
| Pos embed       | Built into ViT                       | Learnable 49-token PE (this model) |
| Fine-tune       | End-to-end                           | End-to-end (NOT frozen — key vs M08)|
| Cross-attn      | FaCoR, 2 layers, 8 heads             | FaCoR, 2 layers, 8 heads (port)    |
| Loss            | Supervised cosine contrastive m=0.3  | Same                               |

The hypothesis is that face-aware features + fully trainable backbone will
let the FaCoR cross-attention focus on kinship-relevant spatial patterns
without the "anti-kinship" trap that frozen ArcFace exhibits in M08.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored AdaFace IR-101 lives next to this file.
sys.path.insert(0, str(Path(__file__).parent))
from adaface_iresnet import AdaFaceIR101, load_adaface_state_dict  # noqa: E402


# ---------------------------------------------------------------------------
# FaCoR cross-attention — verbatim port from models/02_vit_facor_crossattn/model.py
# ---------------------------------------------------------------------------
class CrossAttentionModule(nn.Module):
    """
    FaCoR-style bidirectional cross-attention between two token sets.

    Each face attends to relevant regions of the other face (and vice versa).
    Identical to M02's `CrossAttentionModule`; only `dim` defaults change
    (512 here vs. 768 in M02).
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x1.shape

        # x1 attends to x2
        q1 = self.q_proj(x1).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k_proj(x2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = self.v_proj(x2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out1 = (attn @ v2).transpose(1, 2).reshape(B, N, D)
        out1 = self.out_proj(out1)
        out1 = self.norm1(x1 + out1)
        out1 = self.norm2(out1 + self.ffn(out1))

        # x2 attends to x1
        q2 = self.q_proj(x2).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k_proj(x1).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = self.v_proj(x1).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.dropout(attn2)

        out2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, D)
        out2 = self.out_proj(out2)
        out2 = self.norm1(x2 + out2)
        out2 = self.norm2(out2 + self.ffn(out2))

        attn_map = (attn + attn2.transpose(-2, -1)) / 2
        return out1, out2, attn_map


class ChannelAttention(nn.Module):
    """SE-style channel reweighting — identical to M02."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            avg = x.mean(dim=1)
        else:
            avg = x
        weights = self.fc(avg)
        if x.dim() == 3:
            weights = weights.unsqueeze(1)
        return x * weights


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class AdaFaceFaCoRKinship(nn.Module):
    """
    AdaFace IR-101 backbone (end-to-end fine-tunable) + FaCoR cross-attention
    + supervised contrastive embedding head. Mirrors M02's architecture.

    Forward returns (emb1, emb2, attn_map). The 512-dim AdaFace pooled
    embedding (from the original output_layer FC) is used as a "global"
    feature analogous to ViT's CLS token, summed into the per-face feature
    before projection.

    Input is (B, 3, 112, 112), normalised to [-1, 1] (AdaFace convention).
    """

    SPATIAL_DIM = 512  # AdaFace IR-101 final spatial channel dim
    NUM_TOKENS = 49    # 7x7

    def __init__(
        self,
        adaface_weights: Optional[str] = None,
        embedding_dim: int = 512,
        num_cross_attn_layers: int = 2,
        cross_attn_heads: int = 8,
        channel_reduction: int = 16,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        use_positional_embedding: bool = True,
        use_global_embedding: bool = True,
        original_weight: float = 0.5,
    ):
        super().__init__()
        # Weight applied to the original face in the SAM age-ensemble path.
        # When inputs are 5D (B, V, 3, H, W) with V = 1 + N_ages variants, the
        # original gets `original_weight` and the remaining (1 - original_weight)
        # is split equally across the aged variants. Has no effect on 4D inputs.
        self.original_weight = float(original_weight)

        # Backbone
        self.backbone = AdaFaceIR101(output_dim=self.SPATIAL_DIM)
        if adaface_weights is not None:
            state = torch.load(adaface_weights, map_location="cpu", weights_only=False)
            missing, unexpected = load_adaface_state_dict(self.backbone, state)
            if missing:
                print(f"  [M10] AdaFace missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"  [M10] AdaFace unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            else:
                print(f"  [M10] AdaFace IR-101 weights loaded cleanly from {adaface_weights}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        token_dim = self.SPATIAL_DIM

        # Optional learnable positional embedding for the 7x7 layout.
        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.NUM_TOKENS, token_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # FaCoR cross-attention stack
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionModule(
                dim=token_dim,
                num_heads=cross_attn_heads,
                dropout=dropout,
            )
            for _ in range(num_cross_attn_layers)
        ])

        # Channel attention on pooled feature
        self.channel_attn = ChannelAttention(token_dim, reduction=channel_reduction)

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(token_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.use_global_embedding = use_global_embedding
        self.token_dim = token_dim
        self.embedding_dim = embedding_dim

    # ----- token extraction -------------------------------------------------
    def _age_ensemble_weights(self, num_variants: int, device, dtype) -> torch.Tensor:
        """Return (V,) weights summing to 1: [original_weight, aged...]."""
        if num_variants <= 1:
            return torch.ones(1, device=device, dtype=dtype)
        w0 = self.original_weight
        aged_each = (1.0 - w0) / (num_variants - 1)
        w = torch.full((num_variants,), aged_each, device=device, dtype=dtype)
        w[0] = w0
        return w

    def extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the AdaFace backbone and return spatial tokens + global embedding.

        Accepts either:
        - 4D input (B, 3, H, W) — standard path.
        - 5D input (B, V, 3, H, W) — age-ensemble path. V = 1 + N_ages, where
          the first slice is the original face and the rest are SAM age
          variants. Tokens and global embeddings are weighted-averaged across
          V using `original_weight` (original) and uniform `(1-original_weight)/N_ages`
          (each aged variant) before any further processing.

        Returns
        -------
        tokens : Tensor (B, 49, 512)
            7x7 spatial features, optionally with positional embedding added.
        global_emb : Tensor (B, 512)
            AdaFace's pooled 512-d feature (un-normalised). Acts like a CLS
            token; summed into the per-face feature before projection when
            `use_global_embedding=True`.
        """
        if x.dim() == 5:
            B, V, C, H, W = x.shape
            flat = x.reshape(B * V, C, H, W)
            spatial = self.backbone.forward_spatial(flat)  # (B*V, 512, 7, 7)

            weights = self._age_ensemble_weights(V, spatial.device, spatial.dtype)

            tokens_flat = spatial.flatten(2).transpose(1, 2).contiguous()
            tokens_flat = tokens_flat.reshape(B, V, self.NUM_TOKENS, self.SPATIAL_DIM)
            tokens = (tokens_flat * weights.view(1, V, 1, 1)).sum(dim=1)

            if self.pos_embed is not None:
                tokens = tokens + self.pos_embed

            if self.use_global_embedding:
                global_flat = self.backbone.output_layer(spatial)  # (B*V, 512)
                global_flat = global_flat.reshape(B, V, self.SPATIAL_DIM)
                global_emb = (global_flat * weights.view(1, V, 1)).sum(dim=1)
            else:
                global_emb = tokens.mean(dim=1)
            return tokens, global_emb

        spatial = self.backbone.forward_spatial(x)  # (B, 512, 7, 7)
        b, c, h, w = spatial.shape
        tokens = spatial.flatten(2).transpose(1, 2).contiguous()  # (B, 49, 512)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed

        if self.use_global_embedding:
            global_emb = self.backbone.output_layer(spatial)  # (B, 512)
        else:
            global_emb = tokens.mean(dim=1)
        return tokens, global_emb

    # ----- main forward -----------------------------------------------------
    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        img1, img2 : Tensor (B, 3, 112, 112), normalised to [-1, 1].

        Returns
        -------
        emb1, emb2 : Tensor (B, embedding_dim) — L2-normalised.
        attn_map   : Tensor (B, num_heads, 49, 49) — mean of cross-attn layers.
        """
        tokens1, global1 = self.extract_tokens(img1)
        tokens2, global2 = self.extract_tokens(img2)

        attn_maps = []
        for layer in self.cross_attn_layers:
            tokens1, tokens2, attn = layer(tokens1, tokens2)
            attn_maps.append(attn)
        attn_map = torch.stack(attn_maps, dim=0).mean(dim=0)

        # Pool attended tokens
        feat1 = tokens1.mean(dim=1)
        feat2 = tokens2.mean(dim=1)

        # Channel attention
        feat1 = self.channel_attn(feat1)
        feat2 = self.channel_attn(feat2)

        # Combine with global embedding (CLS-token analogue)
        if self.use_global_embedding:
            feat1 = feat1 + global1
            feat2 = feat2 + global2

        # Project + L2 normalise
        emb1 = self.projection(feat1)
        emb2 = self.projection(feat2)
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        return emb1, emb2, attn_map

    def get_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        emb1, emb2, _ = self.forward(img1, img2)
        return F.cosine_similarity(emb1, emb2, dim=1)


# ---------------------------------------------------------------------------
# Classifier wrapper — identical role to M02's ViTFaCoRClassifier.
# ---------------------------------------------------------------------------
class AdaFaceFaCoRClassifier(nn.Module):
    def __init__(self, base_model: AdaFaceFaCoRKinship):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(base_model.embedding_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        emb1, emb2, attn_map = self.base_model(img1, img2)
        diff = emb1 - emb2
        product = emb1 * emb2
        combined = torch.cat([emb1, emb2, diff, product], dim=1)
        logits = self.classifier(combined)
        return logits, emb1, emb2, attn_map


# ---------------------------------------------------------------------------
# Output parsing — mirrors M02's `parse_model_outputs`.
# ---------------------------------------------------------------------------
def parse_model_outputs(outputs):
    """Normalise model outputs into embeddings, attention, and scalar scores."""
    if not isinstance(outputs, tuple):
        raise ValueError("AdaFace-FaCoR outputs are expected to be a tuple.")

    if len(outputs) >= 4:
        logits, emb1, emb2, attn_map = outputs[:4]
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        scores = torch.sigmoid(logits)
        return {
            "logits": logits,
            "emb1": emb1,
            "emb2": emb2,
            "attn_map": attn_map,
            "scores": scores,
        }

    if len(outputs) >= 3:
        emb1, emb2, attn_map = outputs[:3]
        scores = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2
        return {
            "emb1": emb1,
            "emb2": emb2,
            "attn_map": attn_map,
            "scores": scores,
        }

    if len(outputs) >= 2:
        emb1, emb2 = outputs[:2]
        scores = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2
        return {
            "emb1": emb1,
            "emb2": emb2,
            "attn_map": None,
            "scores": scores,
        }

    raise ValueError("Unsupported AdaFace-FaCoR output format.")


def build_adaface_facor_model(
    *,
    adaface_weights: Optional[str] = None,
    embedding_dim: int = 512,
    num_cross_attn_layers: int = 2,
    cross_attn_heads: int = 8,
    channel_reduction: int = 16,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
    use_positional_embedding: bool = True,
    use_global_embedding: bool = True,
    use_classifier_head: bool = False,
    original_weight: float = 0.5,
):
    """Create either the embedding model or the BCE classifier wrapper."""
    base = AdaFaceFaCoRKinship(
        adaface_weights=adaface_weights,
        embedding_dim=embedding_dim,
        num_cross_attn_layers=num_cross_attn_layers,
        cross_attn_heads=cross_attn_heads,
        channel_reduction=channel_reduction,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        use_positional_embedding=use_positional_embedding,
        use_global_embedding=use_global_embedding,
        original_weight=original_weight,
    )
    if use_classifier_head:
        return AdaFaceFaCoRClassifier(base)
    return base


def create_model(config=None) -> AdaFaceFaCoRKinship:
    if config is None:
        return AdaFaceFaCoRKinship()
    return build_adaface_facor_model(
        adaface_weights=getattr(config, "adaface_weights", None),
        embedding_dim=getattr(config, "embedding_dim", 512),
        num_cross_attn_layers=getattr(config, "cross_attn_layers", 2),
        cross_attn_heads=getattr(config, "cross_attn_heads", 8),
        channel_reduction=getattr(config, "channel_reduction", 16),
        dropout=getattr(config, "dropout", 0.2),
        freeze_backbone=getattr(config, "freeze_backbone", False),
        use_positional_embedding=getattr(config, "use_positional_embedding", True),
        use_global_embedding=getattr(config, "use_global_embedding", True),
        use_classifier_head=getattr(config, "use_classifier_head", False),
        original_weight=getattr(config, "original_weight", 0.5),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Smoke test: AdaFaceFaCoRKinship\n" + "=" * 50)

    model = AdaFaceFaCoRKinship()
    img1 = torch.randn(2, 3, 112, 112)
    img2 = torch.randn(2, 3, 112, 112)

    emb1, emb2, attn = model(img1, img2)
    print(f"emb1 shape:     {tuple(emb1.shape)}")
    print(f"emb2 shape:     {tuple(emb2.shape)}")
    print(f"attn_map shape: {tuple(attn.shape)}")
    print(f"cos sim:        {F.cosine_similarity(emb1, emb2, dim=1).detach().tolist()}")
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total/trainable: {total:,} / {train:,} ({100*train/total:.2f}%)")

    # Classifier wrapper
    cls = AdaFaceFaCoRClassifier(model)
    logits, _, _, _ = cls(img1, img2)
    print(f"\nclassifier logits shape: {tuple(logits.shape)}")
