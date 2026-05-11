"""
Model 09 — AdaFace IR-101 + Multi-Stage Cross-Attention (SAI-inspired).

Same backbone as Model 10 (AdaFace IR-101, WebFace4M) but moves the pairwise
cross-attention **inside** the backbone rather than only at the top. The
design is inspired by CI³Former (Zhang et al., IEEE TCSVT 2025), where the
Self-Attention based Interaction (SAI) module is inserted within each Swin
block so the two faces of a pair can mutually query each other during
feature extraction, not only after.

Differences vs Model 10:

| Aspect            | M10 (FaCoR top-only)             | M09 (SAI-inspired multi-stage) |
|-------------------|----------------------------------|--------------------------------|
| Backbone          | AdaFace IR-101                   | AdaFace IR-101 (same)          |
| Where cross-attn  | Only after `body` (7×7)          | After stage 3 (14×14) AND stage 4 (7×7) |
| Tokens (stage)    | 49 × 512                         | 196 × 256  →  49 × 512         |
| Cross-attn layers | 2 (single stage)                 | 1 per stage by default         |
| Input             | 112×112, [-1, 1]                 | 112×112, [-1, 1] (same)        |
| Loss              | Cosine contrastive (M02 recipe)  | Same                            |
| Age augmentation  | Optional SAM ensemble (M10 only) | None — direct comparison       |

The intent is to test whether *injecting* pair information mid-backbone
helps more than only adding it on top, isolating that architectural change
from any data-augmentation effect (which is what M10's SAM ensemble adds).

Stages tapped: after block 45 (end of stage 3, 14×14×256) and after block 48
(end of stage 4, 7×7×512). Stage 1 (56×56) and stage 2 (28×28) are too token-
heavy for paired cross-attention on a 12 GB GPU (stage 2 alone gives a
784² attention matrix per head per pair).

Output of `forward(img1, img2)` matches M02/M10:
    (emb1, emb2, attn_map_stage4)

`attn_map_stage4` is returned for interface compatibility with the existing
visualisers; the stage 3 attention map is also accessible via
`forward_with_attention_maps(img1, img2)`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored backbone — same as M10. We import from M10's directory so the
# IR-101 implementation has a single source of truth.
sys.path.insert(0, str(Path(__file__).parent.parent / "10_adaface_facor"))
from adaface_iresnet import AdaFaceIR101, load_adaface_state_dict  # noqa: E402


# ---------------------------------------------------------------------------
# FaCoR-style cross-attention — verbatim from M02/M10 except for generic dim.
# ---------------------------------------------------------------------------
class CrossAttentionModule(nn.Module):
    """Bidirectional cross-attention block (FaCoR convention)."""

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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
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


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class AdaFaceMultiStageKinship(nn.Module):
    """
    AdaFace IR-101 + cross-attention injected after stage 3 (14×14, 256-d
    tokens) and stage 4 (7×7, 512-d tokens) of the IR-101 body.

    Block index boundaries in IR-101 `body`:
        stage 1: 0..2   (56×56, 64ch)
        stage 2: 3..15  (28×28, 128ch)
        stage 3: 16..45 (14×14, 256ch)
        stage 4: 46..48 (7×7,  512ch)

    By default cross-attention is applied at the end of stages 3 and 4. Set
    `cross_attn_stages` to override (subset of {3, 4}).
    """

    STAGE_INFO = {
        # stage_id : (block_start, block_stop, spatial, channels)
        1: (0,  3,  56, 64),
        2: (3,  16, 28, 128),
        3: (16, 46, 14, 256),
        4: (46, 49, 7,  512),
    }

    def __init__(
        self,
        adaface_weights: Optional[str] = None,
        embedding_dim: int = 512,
        cross_attn_stages: Optional[List[int]] = None,
        num_cross_attn_layers_per_stage: int = 1,
        cross_attn_heads: int = 8,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        use_positional_embedding: bool = True,
        use_global_embedding: bool = True,
    ):
        super().__init__()

        if cross_attn_stages is None:
            cross_attn_stages = [3, 4]
        cross_attn_stages = sorted(set(int(s) for s in cross_attn_stages))
        for s in cross_attn_stages:
            if s not in self.STAGE_INFO:
                raise ValueError(f"cross_attn_stages must be in {{1,2,3,4}}, got {s}")
        self.cross_attn_stages = cross_attn_stages

        # Backbone
        self.backbone = AdaFaceIR101(output_dim=512)
        if adaface_weights is not None:
            state = torch.load(adaface_weights, map_location="cpu", weights_only=False)
            missing, unexpected = load_adaface_state_dict(self.backbone, state)
            if missing:
                print(f"  [M09] AdaFace missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"  [M09] AdaFace unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            else:
                print(f"  [M09] AdaFace IR-101 weights loaded cleanly from {adaface_weights}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Split body into chunks per stage so we can tap intermediate features.
        # `body_chunks[i]` runs all blocks from stage (i+1)'s start up to the
        # cumulative end of that stage. We rebuild as nn.Sequential views over
        # the *same* nn.Module instances (no parameter duplication).
        body_blocks = list(self.backbone.body.children())
        self.body_chunk_stage1 = nn.Sequential(*body_blocks[0:3])    # 0..2
        self.body_chunk_stage2 = nn.Sequential(*body_blocks[3:16])   # 3..15
        self.body_chunk_stage3 = nn.Sequential(*body_blocks[16:46])  # 16..45
        self.body_chunk_stage4 = nn.Sequential(*body_blocks[46:49])  # 46..48

        # Per-stage cross-attention + positional embedding + per-stage projection
        # to the shared `embedding_dim`. Created only for tapped stages.
        self.cross_attn = nn.ModuleDict()
        self.pos_embeds = nn.ParameterDict()
        self.stage_proj = nn.ModuleDict()
        for stage_id in self.cross_attn_stages:
            _, _, spatial, channels = self.STAGE_INFO[stage_id]
            num_tokens = spatial * spatial
            self.cross_attn[str(stage_id)] = nn.ModuleList([
                CrossAttentionModule(
                    dim=channels,
                    num_heads=cross_attn_heads,
                    dropout=dropout,
                )
                for _ in range(num_cross_attn_layers_per_stage)
            ])
            if use_positional_embedding:
                pe = nn.Parameter(torch.zeros(1, num_tokens, channels))
                nn.init.trunc_normal_(pe, std=0.02)
                self.pos_embeds[str(stage_id)] = pe
            self.stage_proj[str(stage_id)] = nn.Linear(channels, embedding_dim)

        # Optional global embedding (AdaFace's pooled output_layer, like M10).
        self.use_global_embedding = use_global_embedding
        if use_global_embedding:
            self.global_proj = nn.Linear(512, embedding_dim)

        # Final projection head — sums per-stage projections (+ global) into
        # a single `embedding_dim`-vector through an MLP, mirroring M02/M10.
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.use_positional_embedding = use_positional_embedding
        self.embedding_dim = embedding_dim
        self.num_cross_attn_layers_per_stage = num_cross_attn_layers_per_stage
        self.cross_attn_heads = cross_attn_heads

    # ----- internal helpers -------------------------------------------------
    def _apply_stage_cross_attn(
        self,
        stage_id: int,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        feat1, feat2 : (B, C, S, S) spatial features at stage `stage_id`.
        Returns (feat1_out, feat2_out, attn_map) with the spatial layout
        preserved. attn_map is the mean of all per-layer maps for the stage
        (B, num_heads, S², S²).
        """
        B, C, S, _ = feat1.shape
        tokens1 = feat1.flatten(2).transpose(1, 2).contiguous()  # (B, S², C)
        tokens2 = feat2.flatten(2).transpose(1, 2).contiguous()
        key = str(stage_id)

        if key in self.pos_embeds:
            tokens1 = tokens1 + self.pos_embeds[key]
            tokens2 = tokens2 + self.pos_embeds[key]

        attn_maps = []
        for layer in self.cross_attn[key]:
            tokens1, tokens2, attn = layer(tokens1, tokens2)
            attn_maps.append(attn)
        attn_map = torch.stack(attn_maps, dim=0).mean(dim=0)

        feat1_out = tokens1.transpose(1, 2).reshape(B, C, S, S)
        feat2_out = tokens2.transpose(1, 2).reshape(B, C, S, S)
        return feat1_out, feat2_out, attn_map

    # ----- main forward -----------------------------------------------------
    def forward_with_attention_maps(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Like `forward` but returns the *full* per-stage attention dict instead
        of only the last-stage map. Useful for analysis/visualisation.
        """
        feat1 = self.backbone.input_layer(img1)
        feat2 = self.backbone.input_layer(img2)

        attn_maps: Dict[int, torch.Tensor] = {}
        pooled_per_stage: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for stage_id, chunk in (
            (1, self.body_chunk_stage1),
            (2, self.body_chunk_stage2),
            (3, self.body_chunk_stage3),
            (4, self.body_chunk_stage4),
        ):
            feat1 = chunk(feat1)
            feat2 = chunk(feat2)

            if stage_id in self.cross_attn_stages:
                feat1, feat2, attn_map = self._apply_stage_cross_attn(stage_id, feat1, feat2)
                attn_maps[stage_id] = attn_map
                pooled_per_stage[stage_id] = (
                    feat1.flatten(2).mean(dim=2),  # (B, C)
                    feat2.flatten(2).mean(dim=2),
                )

        # Per-stage projections summed into a shared embedding space.
        feat1_sum = None
        feat2_sum = None
        for stage_id in self.cross_attn_stages:
            p1, p2 = pooled_per_stage[stage_id]
            p1 = self.stage_proj[str(stage_id)](p1)
            p2 = self.stage_proj[str(stage_id)](p2)
            feat1_sum = p1 if feat1_sum is None else feat1_sum + p1
            feat2_sum = p2 if feat2_sum is None else feat2_sum + p2

        if self.use_global_embedding:
            # After the reshape inside `_apply_stage_cross_attn`, `feat1`/`feat2`
            # may be non-contiguous; AdaFace's output_layer uses `view` which
            # requires contiguity, so force it here.
            global1 = self.backbone.output_layer(feat1.contiguous())  # (B, 512)
            global2 = self.backbone.output_layer(feat2.contiguous())
            feat1_sum = feat1_sum + self.global_proj(global1)
            feat2_sum = feat2_sum + self.global_proj(global2)

        emb1 = F.normalize(self.projection(feat1_sum), dim=1)
        emb2 = F.normalize(self.projection(feat2_sum), dim=1)
        return emb1, emb2, attn_maps

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
        emb1, emb2 : Tensor (B, embedding_dim) L2-normalised.
        attn_map   : Tensor (B, num_heads, N, N) — the *deepest tapped stage*'s
                     attention map (so the visualisers can keep using a 7×7
                     grid when stage 4 is included).
        """
        emb1, emb2, attn_maps = self.forward_with_attention_maps(img1, img2)
        # Pick the deepest tapped stage for the legacy attn_map output.
        deepest = max(self.cross_attn_stages)
        return emb1, emb2, attn_maps[deepest]

    def get_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        emb1, emb2, _ = self.forward(img1, img2)
        return F.cosine_similarity(emb1, emb2, dim=1)


# ---------------------------------------------------------------------------
# BCE classifier wrapper — mirrors M02/M10.
# ---------------------------------------------------------------------------
class AdaFaceMultiStageClassifier(nn.Module):
    def __init__(self, base_model: AdaFaceMultiStageKinship):
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
# Output parsing — same shape as M02/M10.
# ---------------------------------------------------------------------------
def parse_model_outputs(outputs):
    if not isinstance(outputs, tuple):
        raise ValueError("M09 outputs are expected to be a tuple.")

    if len(outputs) >= 4:
        logits, emb1, emb2, attn_map = outputs[:4]
        if logits.ndim > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        scores = torch.sigmoid(logits)
        return {"logits": logits, "emb1": emb1, "emb2": emb2,
                "attn_map": attn_map, "scores": scores}

    if len(outputs) >= 3:
        emb1, emb2, attn_map = outputs[:3]
        scores = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2
        return {"emb1": emb1, "emb2": emb2, "attn_map": attn_map, "scores": scores}

    if len(outputs) >= 2:
        emb1, emb2 = outputs[:2]
        scores = (F.cosine_similarity(emb1, emb2, dim=1) + 1) / 2
        return {"emb1": emb1, "emb2": emb2, "attn_map": None, "scores": scores}

    raise ValueError("Unsupported M09 output format.")


def _parse_stages(stages) -> List[int]:
    """Accept None, list, tuple, or comma-separated string."""
    if stages is None:
        return [3, 4]
    if isinstance(stages, str):
        return [int(x.strip()) for x in stages.split(",") if x.strip()]
    return [int(x) for x in stages]


def build_adaface_multistage_model(
    *,
    adaface_weights: Optional[str] = None,
    embedding_dim: int = 512,
    cross_attn_stages=None,
    num_cross_attn_layers_per_stage: int = 1,
    cross_attn_heads: int = 8,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
    use_positional_embedding: bool = True,
    use_global_embedding: bool = True,
    use_classifier_head: bool = False,
):
    base = AdaFaceMultiStageKinship(
        adaface_weights=adaface_weights,
        embedding_dim=embedding_dim,
        cross_attn_stages=_parse_stages(cross_attn_stages),
        num_cross_attn_layers_per_stage=num_cross_attn_layers_per_stage,
        cross_attn_heads=cross_attn_heads,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        use_positional_embedding=use_positional_embedding,
        use_global_embedding=use_global_embedding,
    )
    if use_classifier_head:
        return AdaFaceMultiStageClassifier(base)
    return base


def create_model(config=None):
    if config is None:
        return AdaFaceMultiStageKinship()
    return build_adaface_multistage_model(
        adaface_weights=getattr(config, "adaface_weights", None),
        embedding_dim=getattr(config, "embedding_dim", 512),
        cross_attn_stages=getattr(config, "cross_attn_stages", None),
        num_cross_attn_layers_per_stage=getattr(config, "cross_attn_layers_per_stage", 1),
        cross_attn_heads=getattr(config, "cross_attn_heads", 8),
        dropout=getattr(config, "dropout", 0.2),
        freeze_backbone=getattr(config, "freeze_backbone", False),
        use_positional_embedding=getattr(config, "use_positional_embedding", True),
        use_global_embedding=getattr(config, "use_global_embedding", True),
        use_classifier_head=getattr(config, "use_classifier_head", False),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Smoke test: AdaFaceMultiStageKinship\n" + "=" * 50)

    model = AdaFaceMultiStageKinship()
    img1 = torch.randn(2, 3, 112, 112)
    img2 = torch.randn(2, 3, 112, 112)

    emb1, emb2, attn = model(img1, img2)
    print(f"emb1 shape:     {tuple(emb1.shape)}")
    print(f"emb2 shape:     {tuple(emb2.shape)}")
    print(f"attn_map (s4):  {tuple(attn.shape)}")
    print(f"cos sim:        {F.cosine_similarity(emb1, emb2, dim=1).detach().tolist()}")

    _, _, attn_dict = model.forward_with_attention_maps(img1, img2)
    print(f"attn stages:    {[(s, tuple(a.shape)) for s, a in attn_dict.items()]}")

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total/trainable: {total:,} / {train:,} ({100*train/total:.2f}%)")

    cls = AdaFaceMultiStageClassifier(model)
    logits, _, _, _ = cls(img1, img2)
    print(f"\nclassifier logits shape: {tuple(logits.shape)}")
