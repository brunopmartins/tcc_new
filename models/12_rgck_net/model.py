"""
Model 12 — RGCK-Net (Region-Guided Cross Kinship Network).

Implements the proposal in `proposta_rgck_net_kinship.md`. Phase 1
recipe: AdaFace IR-101 frozen, fixed-partition region tokens, bidirectional
cross-region attention, sigmoid regional gating, BCE classifier head.

The architecture stack:

    aligned 224×224 face
        │
        ├─► resize whole → 112×112 → AdaFace ──► global token (1×512)
        │
        └─► crop 4 anatomical boxes (eyes/nose/mouth/jaw)
              │
              └─► each region resized → 112×112 → AdaFace ──► region tokens (4×512)

    [global, eyes, nose, mouth, jaw] × Face_A  →  K=5 tokens per face

        │
        ▼

    Cross-Region Adapter (bidirectional)
      tokens_A ←→ tokens_B  attend to each other

        │
        ▼

    Regional Gate (per-region sigmoid weights)
        ↓
    Weighted regional cosine similarity

        │
        ▼

    Fusion ([gA, gB, |diff|, prod, regional_sims, weights, regional_score])
        ↓
    MLP classifier → kinship logit
"""

from __future__ import annotations

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


# ---------------------------------------------------------------------------
# Region boxes in 224×224 aligned-face coordinates (y0, y1, x0, x1)
# ---------------------------------------------------------------------------

DEFAULT_REGIONS_224: List[Tuple[str, Tuple[int, int, int, int]]] = [
    ("global", (0, 224, 0, 224)),
    ("eyes",   (40, 100, 20, 204)),
    ("nose",   (80, 150, 70, 154)),
    ("mouth",  (140, 185, 50, 174)),
    ("jaw",    (170, 220, 20, 204)),
]


# ---------------------------------------------------------------------------
# Region tokenizer: crop fixed boxes, resize each to 112, run AdaFace
# ---------------------------------------------------------------------------

class FixedPartitionRegionTokenizer(nn.Module):
    """
    Crop fixed anatomical regions from a 224×224 aligned face, resize each to
    112×112, run them through a shared AdaFace backbone, return (K, 512) tokens.

    The backbone is shared across regions to keep the embedding space aligned.
    """

    def __init__(
        self,
        backbone: nn.Module,
        regions: List[Tuple[str, Tuple[int, int, int, int]]] = None,
        target_size: int = 112,
    ):
        super().__init__()
        self.backbone = backbone
        self.regions = regions if regions is not None else DEFAULT_REGIONS_224
        self.region_names = [name for name, _ in self.regions]
        self.target_size = target_size

    @property
    def num_regions(self) -> int:
        return len(self.regions)

    def _crop_regions(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, 224, 224) → (B, K, 3, 112, 112)."""
        B = img.shape[0]
        K = self.num_regions
        crops = []
        for _, (y0, y1, x0, x1) in self.regions:
            region = img[:, :, y0:y1, x0:x1]
            region = F.interpolate(
                region,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
            crops.append(region)
        # Stack: list of (B, 3, 112, 112) → (B, K, 3, 112, 112)
        return torch.stack(crops, dim=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, 224, 224) → (B, K, 512)."""
        B = img.shape[0]
        K = self.num_regions

        crops = self._crop_regions(img)  # (B, K, 3, 112, 112)
        crops_flat = crops.view(B * K, 3, self.target_size, self.target_size)

        # AdaFace returns L2-normalised embeddings (B, 512)
        emb_flat = self.backbone(crops_flat)
        if isinstance(emb_flat, tuple):
            emb_flat = emb_flat[0]

        return emb_flat.view(B, K, -1)


# ---------------------------------------------------------------------------
# ROI-Align region tokenizer (R013): one backbone pass + ROI-Align pooling
# ---------------------------------------------------------------------------

class ROIAlignRegionTokenizer(nn.Module):
    """
    R013 tokenizer. Instead of cropping each anatomical box, squashing it to
    112×112 and re-running AdaFace (the FixedPartition approach — which
    distorts thin strips like the eye box and feeds AdaFace out-of-distribution
    crops), this runs the *whole* aligned face through AdaFace's conv body ONCE
    and pools each region from the shared spatial feature map with ROI-Align.

    - global token  : ``output_layer(forward_spatial(face))`` — identical to the
      standard AdaFace embedding, so the strong global signal is preserved
      exactly.
    - region tokens : ``output_layer(roi_align(F, box, 7×7))`` — undistorted
      (ROI-Align respects the box aspect ratio), in-distribution (same conv
      features as the global), and in AdaFace's embedding space (shared
      output_layer), so per-region cosines remain comparable.

    Cost: 1 conv-body forward per face instead of 5.
    """

    HEAD_GRID = 7  # AdaFace output_layer expects a 7×7 spatial map (112 input)

    def __init__(
        self,
        backbone: nn.Module,
        regions: List[Tuple[str, Tuple[int, int, int, int]]] = None,
        src_coord_size: int = 224,
        backbone_input_size: int = 112,
    ):
        super().__init__()
        self.backbone = backbone
        self.regions = regions if regions is not None else DEFAULT_REGIONS_224
        self.region_names = [name for name, _ in self.regions]
        # Size the face is fed to the conv body. 112 -> 7×7 feature map (the
        # AdaFace-native grid; global token is then the exact embedding). 224 ->
        # 14×14 feature map (a finer grid for region ROI-Align); the global is
        # adaptive-pooled back to 7×7 for the fixed output_layer head. The source
        # FIW faces are ~110px, so 224 is upsampled — the gain is the finer
        # feature grid for region localisation, not new pixel detail.
        self.backbone_input_size = backbone_input_size

        # Index of the whole-face region (kept as the AdaFace embedding).
        self.global_idx = (
            self.region_names.index("global") if "global" in self.region_names else 0
        )

        # Pre-build ROI boxes (xyxy) for the non-global regions, scaled from the
        # 224-coord box list into the backbone-input coordinate frame.
        s = backbone_input_size / float(src_coord_size)
        anat_idx, boxes = [], []
        for i, (_, (y0, y1, x0, x1)) in enumerate(self.regions):
            if i == self.global_idx:
                continue
            anat_idx.append(i)
            boxes.append([x0 * s, y0 * s, x1 * s, y1 * s])
        self.anat_idx = anat_idx
        self.register_buffer("_boxes_xyxy", torch.tensor(boxes, dtype=torch.float32))

    @property
    def num_regions(self) -> int:
        return len(self.regions)

    def _head(self, fmap: torch.Tensor) -> torch.Tensor:
        """Run AdaFace's output_layer, adaptive-pooling the feature map to the
        head's expected 7×7 first (identity when the map is already 7×7)."""
        if fmap.shape[-1] != self.HEAD_GRID or fmap.shape[-2] != self.HEAD_GRID:
            fmap = F.adaptive_avg_pool2d(fmap, self.HEAD_GRID)
        return self.backbone.output_layer(fmap)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) aligned face → (B, K, 512) region tokens."""
        B = img.shape[0]
        K = self.num_regions
        S = self.backbone_input_size

        if img.shape[-1] != S or img.shape[-2] != S:
            face = F.interpolate(img, size=(S, S), mode="bilinear", align_corners=False)
        else:
            face = img

        fmap = self.backbone.forward_spatial(face)  # (B, 512, S/16, S/16)
        # spatial_scale maps box coords (in the S-input frame) to the feature map.
        spatial_scale = fmap.shape[-1] / float(S)

        out = [None] * K
        # Global token (adaptive-pooled to 7×7 head grid; exact embedding at S=112).
        out[self.global_idx] = self._head(fmap)  # (B, 512)

        # Anatomical regions via ROI-Align on the shared feature map -> 7×7 -> head.
        if self.anat_idx:
            boxes = self._boxes_xyxy.to(img.dtype)
            roi = roi_align(
                fmap, [boxes for _ in range(B)],
                output_size=self.HEAD_GRID,
                spatial_scale=spatial_scale,
                aligned=True,
            )  # (B * n_anat, 512, 7, 7)
            emb = self.backbone.output_layer(roi)  # (B * n_anat, 512)
            emb = emb.view(B, len(self.anat_idx), -1)
            for j, i in enumerate(self.anat_idx):
                out[i] = emb[:, j]

        return torch.stack(out, dim=1)  # (B, K, 512)


# ---------------------------------------------------------------------------
# Cross-Region Adapter (bidirectional cross-attention between region tokens)
# ---------------------------------------------------------------------------

class CrossRegionAdapter(nn.Module):
    """
    Bidirectional cross-attention between region tokens of two faces.

    Implements the proposal's section 19. One block by default; configurable
    layers if needed.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "attn_ab": nn.MultiheadAttention(
                        embed_dim=dim, num_heads=num_heads,
                        dropout=dropout, batch_first=True,
                    ),
                    "attn_ba": nn.MultiheadAttention(
                        embed_dim=dim, num_heads=num_heads,
                        dropout=dropout, batch_first=True,
                    ),
                    "norm_a1": nn.LayerNorm(dim),
                    "norm_b1": nn.LayerNorm(dim),
                    "ffn_a": nn.Sequential(
                        nn.Linear(dim, dim * ffn_expansion),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim * ffn_expansion, dim),
                    ),
                    "ffn_b": nn.Sequential(
                        nn.Linear(dim, dim * ffn_expansion),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim * ffn_expansion, dim),
                    ),
                    "norm_a2": nn.LayerNorm(dim),
                    "norm_b2": nn.LayerNorm(dim),
                })
            )

    def forward(
        self, tokens_a: torch.Tensor, tokens_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        tokens_a, tokens_b: (B, K, D)
        Returns (a_ctx, b_ctx, attn_map) where attn_map is from the last layer
        of A→B attention, shape (B, num_heads, K, K).
        """
        a, b = tokens_a, tokens_b
        last_attn_ab = None

        for layer in self.layers:
            # A attends to B
            a_ctx, attn_ab = layer["attn_ab"](
                query=a, key=b, value=b, average_attn_weights=False,
            )
            # B attends to A
            b_ctx, _ = layer["attn_ba"](query=b, key=a, value=a)

            a = layer["norm_a1"](a + a_ctx)
            b = layer["norm_b1"](b + b_ctx)

            a = layer["norm_a2"](a + layer["ffn_a"](a))
            b = layer["norm_b2"](b + layer["ffn_b"](b))

            last_attn_ab = attn_ab

        return a, b, last_attn_ab


# ---------------------------------------------------------------------------
# Regional Gate (per-region sigmoid weights from comparison features)
# ---------------------------------------------------------------------------

class RegionalGate(nn.Module):
    """
    Learns a per-region weight from `[rA_i, rB_i, |rA_i - rB_i|, rA_i * rB_i]`
    via an MLP with sigmoid output. Per the proposal's section 22, sigmoid is
    preferred over softmax so multiple regions can be salient at once.
    """

    def __init__(self, dim: int = 512, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, tokens_a: torch.Tensor, tokens_b: torch.Tensor
    ) -> torch.Tensor:
        """
        tokens_a, tokens_b: (B, K, D)
        Returns weights: (B, K) sigmoid.
        """
        comparison = torch.cat(
            [
                tokens_a,
                tokens_b,
                (tokens_a - tokens_b).abs(),
                tokens_a * tokens_b,
            ],
            dim=-1,
        )  # (B, K, 4D)
        logits = self.mlp(comparison).squeeze(-1)  # (B, K)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Full RGCK-Net model
# ---------------------------------------------------------------------------

class RGCKNet(nn.Module):
    """
    Region-Guided Cross Kinship Network — full model.

    Forward signature: `(img_a, img_b) -> kinship_logit, region_weights, attn_map`
    """

    def __init__(
        self,
        adaface_backbone: nn.Module,
        embedding_dim: int = 512,
        regions: List[Tuple[str, Tuple[int, int, int, int]]] = None,
        cross_attn_heads: int = 4,
        cross_attn_layers: int = 1,
        gate_hidden: int = 128,
        classifier_hidden: int = 512,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        unfreeze_last_stage: bool = False,
        unfreeze_extra_stage3_tail: bool = False,
        aux_relation_head: bool = False,
        num_relation_classes: int = 11,
        symmetric_forward: bool = False,
        comparison_only_fusion: bool = False,
        roi_align_tokenizer: bool = False,
        backbone_input_size: int = 112,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_stage = unfreeze_last_stage
        self.unfreeze_extra_stage3_tail = unfreeze_extra_stage3_tail
        self.aux_relation_head = aux_relation_head
        self.num_relation_classes = num_relation_classes
        self.symmetric_forward = symmetric_forward
        self.comparison_only_fusion = comparison_only_fusion
        self.roi_align_tokenizer = roi_align_tokenizer
        self.backbone_input_size = backbone_input_size

        # Region tokenizer (shared backbone across regions). R013 swaps the
        # crop-and-rerun FixedPartition tokenizer for ROI-Align pooling on a
        # single shared feature map (undistorted, in-distribution regions).
        # backbone_input_size controls the conv-body input resolution (112 ->
        # 7×7 map / exact global; 224 -> 14×14 map / finer regions). ROI-Align only.
        if roi_align_tokenizer:
            self.tokenizer = ROIAlignRegionTokenizer(
                backbone=adaface_backbone,
                regions=regions,
                backbone_input_size=backbone_input_size,
            )
        else:
            self.tokenizer = FixedPartitionRegionTokenizer(
                backbone=adaface_backbone,
                regions=regions,
            )
        self.region_names = self.tokenizer.region_names
        self.num_regions = self.tokenizer.num_regions

        if freeze_backbone:
            for p in adaface_backbone.parameters():
                p.requires_grad = False

            if unfreeze_last_stage:
                # Phase 2 of `proposta_rgck_net_kinship.md` §38: unfreeze the
                # last IR-101 block. For AdaFace IR-101 the deepest stage is
                # body[46:49] (3 BasicBlockIR units, 512-channel, 7×7 spatial)
                # plus the output_layer (FC head producing the 512-d embedding).
                stage4_modules = adaface_backbone.body[46:49]
                for p in stage4_modules.parameters():
                    p.requires_grad = True
                for p in adaface_backbone.output_layer.parameters():
                    p.requires_grad = True

                # R012: optionally extend the unfreeze into the tail of stage 3
                # (body[43:46]) to give the backbone more capacity for the
                # consistency-loss objective. Off by default — only enabled
                # via the R012 recipe.
                if unfreeze_extra_stage3_tail:
                    for p in adaface_backbone.body[43:46].parameters():
                        p.requires_grad = True

        # Cross-region adapter
        self.cross_region = CrossRegionAdapter(
            dim=embedding_dim,
            num_heads=cross_attn_heads,
            num_layers=cross_attn_layers,
            dropout=dropout,
        )

        # Regional gate
        self.regional_gate = RegionalGate(
            dim=embedding_dim, hidden=gate_hidden, dropout=dropout,
        )

        # Classifier head
        # Default fusion: [gA(512), gB(512), |diff|(512), prod(512), per-rel cosines(K), weights(K), regional_score(1)]
        # Comparison-only fusion (R009): drop gA and gB to remove identity-as-
        # feature leakage. Fusion becomes [|diff|(512), prod(512), per-rel
        # cosines(K), weights(K), regional_score(1)].
        K = self.num_regions
        if comparison_only_fusion:
            classifier_input_dim = 2 * embedding_dim + 2 * K + 1
        else:
            classifier_input_dim = 4 * embedding_dim + 2 * K + 1

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 4),
            nn.BatchNorm1d(classifier_hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(classifier_hidden // 4, 1),
        )

        # Phase 5 (proposta_rgck_net_kinship.md §38): auxiliary relation-type
        # classifier on the average of the contextualised global tokens. Used
        # only for positive pairs during training (mask handled in the loss).
        if aux_relation_head:
            self.relation_head = nn.Linear(embedding_dim, num_relation_classes)
        else:
            self.relation_head = None

    def _global_index(self) -> int:
        if "global" in self.region_names:
            return self.region_names.index("global")
        return 0  # Fallback

    def _forward_head(
        self, tokens_a: torch.Tensor, tokens_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "Optional[torch.Tensor]", torch.Tensor]:
        """Run the post-tokenizer stack on a single (tokens_a, tokens_b) order:
        cross-region adapter → regional gate → fusion + classifier (and
        relation_head when enabled). Used twice in the symmetric forward.

        Returns (logit, weights, attn_map, gA_norm, gB_norm, rel_logits, fusion).
        ``fusion`` is the pre-classifier feature vector exposed for the R012
        cosine-consistency loss between the AB and BA passes.
        """
        # Cross-region adapter (direction-dependent: attn_ab / attn_ba etc.)
        ctx_a, ctx_b, attn_map = self.cross_region(tokens_a, tokens_b)

        # Per-region cosine similarities (in contextualised space)
        ctx_a_n = F.normalize(ctx_a, dim=-1)
        ctx_b_n = F.normalize(ctx_b, dim=-1)
        sims = (ctx_a_n * ctx_b_n).sum(dim=-1)  # (B, K)

        # Regional gating
        weights = self.regional_gate(ctx_a, ctx_b)  # (B, K)
        regional_score = (weights * sims).sum(dim=-1, keepdim=True)  # (B, 1)

        # Global features (pulled from the "global" region's contextualised token)
        gA = ctx_a[:, self._global_index()]
        gB = ctx_b[:, self._global_index()]
        diff_abs = (gA - gB).abs()
        prod = gA * gB

        # Final fusion. R009 (comparison_only_fusion=True) drops gA and gB
        # from the classifier input to remove identity-as-feature signal.
        if self.comparison_only_fusion:
            fusion = torch.cat([diff_abs, prod, sims, weights, regional_score], dim=-1)
        else:
            fusion = torch.cat([gA, gB, diff_abs, prod, sims, weights, regional_score], dim=-1)
        logit = self.classifier(fusion).squeeze(-1)  # (B,)

        # L2-normalised global tokens for the SupCon aux loss
        gA_norm = ctx_a_n[:, self._global_index()]
        gB_norm = ctx_b_n[:, self._global_index()]

        # Phase 5 aux head — averages the two contextualised globals so it is
        # symmetric within a single forward direction (still direction-dependent
        # across the AB/BA pair because the contextualisation differs).
        if self.relation_head is not None:
            rel_logits = self.relation_head(0.5 * (gA + gB))
        else:
            rel_logits = None

        return logit, weights, attn_map, gA_norm, gB_norm, rel_logits, fusion

    def forward(
        self, img_a: torch.Tensor, img_b: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        img_a, img_b: (B, 3, 224, 224)  — aligned FIW faces

        Returns a 6-tuple (or 7-tuple when symmetric_forward=True):

            kinship_logit: (B,) raw logit. Symmetric mode returns the average
                of the AB and BA forward logits.
            region_weights: (B, K) sigmoid gating weights per region (AB direction).
            attn_map: (B, num_heads, K, K) last-layer A→B cross-attention (AB direction).
            gA_norm: (B, embedding_dim) L2-normalised contextualised global token A
                (AB direction; in symmetric mode this is what SupCon uses as the
                "A" partner).
            gB_norm: (B, embedding_dim) L2-normalised contextualised global token B
                (AB direction).
            rel_logits: (B, num_relation_classes) auxiliary relation logits, or
                None when aux_relation_head=False. Symmetric mode returns the
                average of the AB and BA logits.
            [optional 7th] sym_extras: dict with the per-direction outputs the
                training loss needs for the Option-B symmetric BCE/CE_rel terms.
                Present only when symmetric_forward=True.
        """
        # Region tokens (B, K, 512). Tokenizer is the expensive part — run once
        # per face even in symmetric mode.
        tokens_a = self.tokenizer(img_a)
        tokens_b = self.tokenizer(img_b)

        logit_ab, weights_ab, attn_ab, gA_norm_ab, gB_norm_ab, rel_ab, fusion_ab = (
            self._forward_head(tokens_a, tokens_b)
        )

        if not self.symmetric_forward:
            return logit_ab, weights_ab, attn_ab, gA_norm_ab, gB_norm_ab, rel_ab

        logit_ba, weights_ba, attn_ba, gA_norm_ba, gB_norm_ba, rel_ba, fusion_ba = (
            self._forward_head(tokens_b, tokens_a)
        )

        # Symmetric outputs: averaged for inference; per-direction kept for the
        # Option-B training loss.
        logit = 0.5 * (logit_ab + logit_ba)
        if rel_ab is not None and rel_ba is not None:
            rel_logits = 0.5 * (rel_ab + rel_ba)
        else:
            rel_logits = None

        sym_extras = {
            "logit_ab": logit_ab,
            "logit_ba": logit_ba,
            "rel_logits_ab": rel_ab,
            "rel_logits_ba": rel_ba,
            "gA_norm_ba": gA_norm_ba,
            "gB_norm_ba": gB_norm_ba,
            # R012: fusion features per direction for the consistency loss.
            "fusion_ab": fusion_ab,
            "fusion_ba": fusion_ba,
        }

        return logit, weights_ab, attn_ab, gA_norm_ab, gB_norm_ab, rel_logits, sym_extras


def build_rgck_net(
    adaface_weights: Optional[str] = None,
    embedding_dim: int = 512,
    regions: List[Tuple[str, Tuple[int, int, int, int]]] = None,
    cross_attn_heads: int = 4,
    cross_attn_layers: int = 1,
    gate_hidden: int = 128,
    classifier_hidden: int = 512,
    dropout: float = 0.2,
    freeze_backbone: bool = True,
    unfreeze_last_stage: bool = False,
    unfreeze_extra_stage3_tail: bool = False,
    aux_relation_head: bool = False,
    num_relation_classes: int = 11,
    symmetric_forward: bool = False,
    comparison_only_fusion: bool = False,
    roi_align_tokenizer: bool = False,
    backbone_input_size: int = 112,
) -> RGCKNet:
    """
    Build RGCK-Net with an AdaFace IR-101 backbone (shared by all regions).
    """
    # Import here to avoid circular dependencies in test harnesses
    from adaface_iresnet import adaface_ir101  # type: ignore

    backbone = adaface_ir101(weights_path=adaface_weights)
    if adaface_weights is not None:
        print(f"  [M12] AdaFace IR-101 weights loaded from {adaface_weights}")

    return RGCKNet(
        adaface_backbone=backbone,
        embedding_dim=embedding_dim,
        regions=regions,
        cross_attn_heads=cross_attn_heads,
        cross_attn_layers=cross_attn_layers,
        gate_hidden=gate_hidden,
        classifier_hidden=classifier_hidden,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        unfreeze_last_stage=unfreeze_last_stage,
        unfreeze_extra_stage3_tail=unfreeze_extra_stage3_tail,
        aux_relation_head=aux_relation_head,
        num_relation_classes=num_relation_classes,
        symmetric_forward=symmetric_forward,
        comparison_only_fusion=comparison_only_fusion,
        roi_align_tokenizer=roi_align_tokenizer,
        backbone_input_size=backbone_input_size,
    )


if __name__ == "__main__":
    # Smoke test (random init)
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    m = build_rgck_net(adaface_weights=None, freeze_backbone=True, aux_relation_head=True)
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"M12 RGCK-Net params total/trainable: {total:,}/{trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Regions: {m.region_names}")
    x = torch.randn(2, 3, 224, 224)
    out = m(x, x)
    logit, weights, attn, gA, gB, rel = out
    print(f"[asym] logit: {logit.shape}, weights: {weights.shape}, attn: {attn.shape}")
    print(f"[asym] gA: {gA.shape}, gB: {gB.shape}, rel: {rel.shape if rel is not None else None}")

    # Symmetric forward smoke test
    m_sym = build_rgck_net(adaface_weights=None, freeze_backbone=True,
                            aux_relation_head=True, symmetric_forward=True)
    out_sym = m_sym(x, x)
    logit_s, w_s, attn_s, gA_s, gB_s, rel_s, sym_extras = out_sym
    print(f"[sym]  logit: {logit_s.shape}, rel: {rel_s.shape if rel_s is not None else None}")
    print(f"[sym]  sym_extras keys: {sorted(sym_extras.keys())}")
