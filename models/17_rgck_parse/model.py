"""Model 17 — PG-RGCK: Parsing-Guided Region-Guided Cross Kinship Network.

M17 keeps the **M15 hi-res ROI-Align head verbatim** (region tokens via one
backbone pass + ROI-Align→FC, bidirectional cross-region attention, sigmoid
gate, fusion classifier, symmetric forward, comparison-only fusion, relation
aux) and the proven M12 R011 recipe. The single change: the **four anatomical
region boxes are PER-IMAGE**, derived offline by a face-parsing segmentation
network (BiSeNet / CelebAMask-HQ) run on each aligned 224 face, instead of the
fixed ``DEFAULT_REGIONS_224`` rectangles.

Why this, and why it is different from M13
------------------------------------------
The methodology already names the limitation: the regional boxes are fixed
coordinates on an aligned face — a static anatomical prior, not dynamic part
detection; if alignment shifts, the crops drift. M13 (Landmark Graph) tried to
fix this but used **canonical** template landmarks, which on pre-aligned FIW
collapse back to fixed positions ("the canonical-landmark design *is* already a
fixed-box model"). M17 uses a real **per-image** parser: each face's eyes /
nose / mouth / jaw boxes are the tight bounding boxes of *that face's* parsed
masks, so the geometry the 5-point similarity alignment leaves un-normalised
(face shape, jaw width, expression, residual pose) is actually tracked.

R001 mechanism (this file): **mask → tight per-face bbox → ROI-Align → FC.**
Boxes change per image, but tokens still pass through AdaFace's ``output_layer``,
so they stay in the AdaFace embedding space and the per-region cosine
similarities the M12 head relies on remain valid. A later R002 can weight the
ROI by the soft mask (contour-aware) — that variant lives behind the same
tokenizer.

The parser is an **offline preprocessing step** (``tools/parse_faces_boxes.py``)
that caches one (n_anat, 4) box array per aligned face to an ``.npz``; training
loads the cache (no parser in the training loop, no extra VRAM). When a region's
mask is empty/too small the cache stores the **fixed DEFAULT box** for that
region (per-image fixed-box fallback), so every image always has valid boxes and
M17-with-no-cache is byte-for-byte M15.

``build_pg_rgck_net`` has the same knobs as M15's ``build_rgck_hires_net``; it
builds the M15 model and swaps in the parsing-guided tokenizer.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

# Reuse the M15 hi-res tokenizer + the M12 head/recipe verbatim. M15's file is
# also ``model.py``; load it under a distinct module name.
_M15 = Path(__file__).resolve().parent.parent / "15_rgck_hires"
if str(_M15) not in sys.path:
    sys.path.insert(0, str(_M15))

_spec = importlib.util.spec_from_file_location("m15_model", str(_M15 / "model.py"))
_m15 = importlib.util.module_from_spec(_spec)
sys.modules["m15_model"] = _m15
_spec.loader.exec_module(_m15)
build_rgck_hires_net = _m15.build_rgck_hires_net
HiResROIAlignRegionTokenizer = _m15.HiResROIAlignRegionTokenizer
RGCKNet = _m15.RGCKNet
DEFAULT_REGIONS_224 = _m15.DEFAULT_REGIONS_224

# Anatomical region order for the per-image box cache (must match the order of
# the non-global regions in DEFAULT_REGIONS_224 → the tokenizer's anat_idx).
# DEFAULT_REGIONS_224 = [global, eyes, nose, mouth, jaw] → anat order below.
ANAT_REGION_ORDER: List[str] = [name for name, _ in DEFAULT_REGIONS_224 if name != "global"]


# ---------------------------------------------------------------------------
# Parsing-guided high-res ROI-Align tokenizer
# ---------------------------------------------------------------------------

class ParseGuidedHiResTokenizer(HiResROIAlignRegionTokenizer):
    """M15 hi-res tokenizer, but the anatomical ROIs are supplied per-image.

    ``forward(img, boxes)``:
      - ``boxes`` is (B, n_anat, 4) xyxy in the ``src_coord_size`` (224) frame,
        ordered like ``self.anat_idx`` (= ANAT_REGION_ORDER: eyes, nose, mouth,
        jaw). These are the per-face boxes from the parsing cache.
      - ``boxes=None`` → falls back to the parent's static boxes (≡ M15). This is
        only the "cache disabled" path; with the cache on, every image always
        gets a full (real-or-fixed-fallback) box tensor.

    The global token is unchanged (whole face — never parsed).
    """

    def forward(self, img: torch.Tensor, boxes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if boxes is None:
            # Static boxes — identical to M15 (and the per-image fixed fallback).
            return super().forward(img)

        B = img.shape[0]
        K = self.num_regions

        # One conv-body pass at native resolution → high-res feature map.
        fmap = self.backbone.forward_spatial(img)  # (B, 512, Hf, Wf)
        feat_size = fmap.shape[-1]
        spatial_scale = feat_size / float(self.src_coord_size)

        out: List[Optional[torch.Tensor]] = [None] * K

        if self.anat_idx:
            b = boxes.to(fmap.dtype)
            if b.shape[1] != len(self.anat_idx):
                raise ValueError(
                    f"PG-RGCK expects {len(self.anat_idx)} per-image boxes "
                    f"(order {ANAT_REGION_ORDER}); got {tuple(b.shape)}"
                )
            # Defensive: guarantee non-degenerate boxes (the parsing fallback
            # should already ensure this, but a zero-area ROI would NaN).
            x0, y0, x1, y1 = b.unbind(-1)
            x1 = torch.maximum(x1, x0 + 1.0)
            y1 = torch.maximum(y1, y0 + 1.0)
            b = torch.stack([x0, y0, x1, y1], dim=-1)
            # roi_align takes a list of (n_anat, 4) box tensors, one per image.
            roi = roi_align(
                fmap, list(b.unbind(0)),
                output_size=self.fc_grid,
                spatial_scale=spatial_scale,
                aligned=True,
            )  # (B * n_anat, 512, fc_grid, fc_grid)
            emb = self._fc(roi).view(B, len(self.anat_idx), -1)
            for j, i in enumerate(self.anat_idx):
                out[i] = emb[:, j]

        # Global token: same as the parent (exact = AdaFace 112 embedding; roi =
        # full-face box pooled from the high-res map).
        if self.global_token_mode == "exact":
            if img.shape[-1] != self.fc_input_size or img.shape[-2] != self.fc_input_size:
                face = F.interpolate(
                    img, size=(self.fc_input_size, self.fc_input_size),
                    mode="bilinear", align_corners=False,
                )
            else:
                face = img
            g = self.backbone(face)
            if isinstance(g, tuple):
                g = g[0]
            out[self.global_idx] = g
        else:  # "roi"
            gbox = self._global_box_xyxy.to(fmap.dtype)
            groi = roi_align(
                fmap, [gbox for _ in range(B)],
                output_size=self.fc_grid,
                spatial_scale=spatial_scale,
                aligned=True,
            )
            out[self.global_idx] = self._fc(groi)

        return torch.stack(out, dim=1)  # (B, K, 512)


# ---------------------------------------------------------------------------
# PG-RGCK net: M15/M12 head with per-image boxes threaded into the tokenizer
# ---------------------------------------------------------------------------

class PGRGCKNet(RGCKNet):
    """RGCK-Net whose forward threads per-image boxes into the tokenizer.

    This subclass adds **no new parameters or buffers** — it only overrides
    ``forward`` to pass ``boxes_a`` / ``boxes_b`` to the (parsing-guided)
    tokenizer. ``build_pg_rgck_net`` re-classes an M15-built instance to this
    type, so checkpoint keys are identical to M15. The body below mirrors
    ``RGCKNet.forward`` (12_rgck_net/model.py) verbatim except the two tokenizer
    calls now take boxes.
    """

    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        boxes_a: Optional[torch.Tensor] = None,
        boxes_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        # Per-image region tokens. boxes_* = (B, n_anat, 4); None ⇒ static (M15).
        tokens_a = self.tokenizer(img_a, boxes_a)
        tokens_b = self.tokenizer(img_b, boxes_b)

        logit_ab, weights_ab, attn_ab, gA_norm_ab, gB_norm_ab, rel_ab, fusion_ab = (
            self._forward_head(tokens_a, tokens_b)
        )

        if not self.symmetric_forward:
            return logit_ab, weights_ab, attn_ab, gA_norm_ab, gB_norm_ab, rel_ab

        logit_ba, weights_ba, attn_ba, gA_norm_ba, gB_norm_ba, rel_ba, fusion_ba = (
            self._forward_head(tokens_b, tokens_a)
        )

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
            "fusion_ab": fusion_ab,
            "fusion_ba": fusion_ba,
        }
        return logit, weights_ab, attn_ab, gA_norm_ab, gB_norm_ab, rel_logits, sym_extras


def build_pg_rgck_net(
    adaface_weights: Optional[str] = None,
    embedding_dim: int = 512,
    regions: List[Tuple[str, Tuple[int, int, int, int]]] = None,
    cross_attn_heads: int = 4,
    cross_attn_layers: int = 1,
    gate_hidden: int = 128,
    classifier_hidden: int = 512,
    dropout: float = 0.2,
    freeze_backbone: bool = True,
    unfreeze_last_stage: bool = True,
    unfreeze_extra_stage3_tail: bool = False,
    aux_relation_head: bool = False,
    num_relation_classes: int = 11,
    symmetric_forward: bool = False,
    comparison_only_fusion: bool = False,
    roi_align_tokenizer: bool = False,  # harness compat; M17 always parse-guided hi-res
    # M15 hi-res controls (passed through)
    global_token_mode: str = "exact",
    fc_grid: int = 7,
) -> RGCKNet:
    """Build the M15 hi-res RGCK-Net, then swap in the parsing-guided tokenizer.

    Same knobs as ``build_rgck_hires_net``. The per-image boxes are supplied at
    runtime by the trainer/eval from the parsing cache (``--region_box_cache``);
    nothing about the box *source* is baked into the weights, so a checkpoint is
    interchangeable with M15 if loaded without a cache.
    """
    model = build_rgck_hires_net(
        adaface_weights=adaface_weights,
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
        global_token_mode=global_token_mode,
        fc_grid=fc_grid,
    )

    # Swap the M15 tokenizer for the parsing-guided one, reusing the SAME
    # backbone instance (freeze/unfreeze already applied).
    old = model.tokenizer
    model.tokenizer = ParseGuidedHiResTokenizer(
        backbone=old.backbone,
        regions=old.regions,
        src_coord_size=old.src_coord_size,
        fc_grid=old.fc_grid,
        global_token_mode=old.global_token_mode,
        fc_input_size=old.fc_input_size,
    )
    model.region_names = model.tokenizer.region_names
    model.num_regions = model.tokenizer.num_regions
    # Add the box-threading forward without rebuilding (no new params/buffers).
    model.__class__ = PGRGCKNet
    model.pg_config = {
        "global_token_mode": global_token_mode,
        "fc_grid": fc_grid,
        "anat_region_order": ANAT_REGION_ORDER,
    }
    print(
        f"  [M17] parsing-guided tokenizer (per-image boxes for "
        f"{ANAT_REGION_ORDER}; global_token_mode={global_token_mode}, fc_grid={fc_grid})"
    )
    return model


if __name__ == "__main__":
    # Smoke test (random init unless weights are present).
    W = _M15.parent / "12_rgck_net" / "weights" / "adaface_ir101_webface4m.pth"
    weights = str(W) if W.exists() else None
    m = build_pg_rgck_net(
        adaface_weights=weights, aux_relation_head=True,
        symmetric_forward=True, comparison_only_fusion=True,
        unfreeze_last_stage=True, global_token_mode="exact",
    )
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"M17 params total/trainable: {tot:,}/{tr:,} ({100*tr/tot:.2f}%)")

    x = torch.randn(2, 3, 224, 224)
    n_anat = len(ANAT_REGION_ORDER)
    # Per-image boxes (xyxy, 224 frame). Fake but valid.
    boxes = torch.tensor(
        [[[20, 40, 200, 100], [70, 80, 150, 154], [50, 140, 170, 185], [20, 170, 200, 220]]],
        dtype=torch.float32,
    ).repeat(2, 1, 1)
    out = m(x, x, boxes, boxes)
    logit = out[0]
    print(f"forward items {len(out)}; logit {tuple(logit.shape)}; "
          f"sym_extras@6 dict: {isinstance(out[6], dict)}")
    feat = m.tokenizer(x, boxes)
    print(f"tokens {tuple(feat.shape)} (expect (2, {1 + n_anat}, 512))")
    # boxes=None path == M15 static fallback
    feat0 = m.tokenizer(x, None)
    print(f"static-fallback tokens {tuple(feat0.shape)} (expect (2, {1 + n_anat}, 512))")
