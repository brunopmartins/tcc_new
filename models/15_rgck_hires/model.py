"""Model 15 — RGCK-Net with a high-resolution ROI-Align region tokenizer.

M15 keeps the M12 RGCK-Net concept verbatim (region tokens, bidirectional
cross-region attention, sigmoid gating, fusion classifier, symmetric forward,
comparison-only fusion, relation-aux head) and the proven M12 R011/R012 recipe,
but **removes the 112×112 limitation** of the region tokenizer.

Why it matters
--------------
- M12 FixedPartition (R001-R012): crops each anatomical box, **squashes it to
  112×112** and re-runs AdaFace 5× per face. The crops are out-of-distribution
  (AdaFace was trained on whole aligned faces, not stretched eye/mouth strips).
- M12 R013 ROIAlign: one backbone pass + ROI-Align pooling — fixes the OOD-crop
  problem — but still runs the body at **112 → a 7×7 feature map**, so every
  region is pooled from at most 1-3 cells of a 7×7 grid (the same coarse-ROI
  problem M13 documented).

The 112 constraint actually comes from AdaFace's FC head
(`output_layer = ... Flatten → Linear(512*7*7, 512)`), which only accepts a 7×7
spatial map. The **conv body is fully convolutional and resolution-agnostic**.

M15 exploits that:
- Run the conv body on the face at its **native resolution** (224 → a 14×14
  feature map, 4× the spatial detail of R013's 7×7; ``--img_size`` can push it
  higher).
- ROI-Align each region from the high-res map to a **fixed 7×7 grid**, *then*
  apply the FC head. Because the ROI output is always 7×7, the FC works at any
  input resolution — the 112 limit is gone, and each region token is sampled
  from real per-region spatial detail.

All tokens still pass through ``output_layer`` (the AdaFace FC), so they live in
the same embedding space and the per-region cosine similarities the M12 head
relies on stay comparable.

``build_rgck_hires_net`` has the same knobs as M12's ``build_rgck_net`` plus the
high-res tokenizer controls; it builds the M12 model and swaps the tokenizer.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

# Reuse the M12 architecture (RGCKNet, build_rgck_net, region defs) verbatim.
# M12's file is also called ``model.py``; load it under a distinct module name.
_M12 = Path(__file__).resolve().parent.parent / "12_rgck_net"
if str(_M12) not in sys.path:
    sys.path.insert(0, str(_M12))

_spec = importlib.util.spec_from_file_location("m12_model", str(_M12 / "model.py"))
_m12 = importlib.util.module_from_spec(_spec)
sys.modules["m12_model"] = _m12
_spec.loader.exec_module(_m12)
build_rgck_net = _m12.build_rgck_net
RGCKNet = _m12.RGCKNet
DEFAULT_REGIONS_224 = _m12.DEFAULT_REGIONS_224


class HiResROIAlignRegionTokenizer(nn.Module):
    """High-resolution ROI-Align region tokenizer (M15).

    Runs AdaFace's conv body ONCE on the full aligned face at its native
    resolution, then pools each anatomical region from the resulting feature map
    with ROI-Align to a fixed ``fc_grid`` (7×7) before the AdaFace FC head.

    Decoupling the feature-map size (e.g. 14×14 at 224 input) from the FC grid
    (always 7×7) is what removes the 112 limitation: the body can run at any
    resolution while the FC still receives the 7×7 it was trained on.

    Parameters
    ----------
    backbone
        AdaFace IR-101 (must expose ``forward_spatial`` and ``output_layer``).
        Shared by the global token and every region; freeze/unfreeze state is
        set by the caller (M15 unfreezes stage 4 + output_layer like M12).
    regions
        ``[(name, (y0, y1, x0, x1)), ...]`` in ``src_coord_size`` coordinates.
        Defaults to M12's 5 boxes (global, eyes, nose, mouth, jaw).
    src_coord_size
        Coordinate frame the region boxes are expressed in (224 — the FIW_aligned
        canonical size). The feature map always covers the whole face, so a box
        maps to feature coords by ``box * feat_size / src_coord_size`` regardless
        of the actual input resolution.
    fc_grid
        ROI-Align output side fed to the FC head. Must be 7 for AdaFace IR-101's
        ``Linear(512*7*7, 512)``.
    global_token_mode
        ``"exact"`` (default): the global token is the genuine AdaFace embedding
        of the whole face resized to 112 (identical signal to M12 / the B0
        baseline; one extra cheap 112 pass). ``"roi"``: the global token is the
        full-face box pooled from the high-res map (single pass, slightly
        cheaper, marginally weaker global signal).
    fc_input_size
        Resolution used for the ``"exact"`` global pass (112 — AdaFace native).
    """

    def __init__(
        self,
        backbone: nn.Module,
        regions: List[Tuple[str, Tuple[int, int, int, int]]] = None,
        src_coord_size: int = 224,
        fc_grid: int = 7,
        global_token_mode: str = "exact",
        fc_input_size: int = 112,
    ):
        super().__init__()
        if global_token_mode not in ("exact", "roi"):
            raise ValueError(f"global_token_mode must be 'exact' or 'roi', got {global_token_mode!r}")
        self.backbone = backbone
        self.regions = regions if regions is not None else DEFAULT_REGIONS_224
        self.region_names = [name for name, _ in self.regions]
        self.src_coord_size = src_coord_size
        self.fc_grid = fc_grid
        self.global_token_mode = global_token_mode
        self.fc_input_size = fc_input_size

        self.global_idx = (
            self.region_names.index("global") if "global" in self.region_names else 0
        )

        # Anatomical boxes (everything except the global region) as xyxy in the
        # src_coord_size (224) frame. spatial_scale at forward time rescales them
        # to feature-map coordinates.
        anat_idx, boxes = [], []
        for i, (_, (y0, y1, x0, x1)) in enumerate(self.regions):
            if i == self.global_idx:
                continue
            anat_idx.append(i)
            boxes.append([float(x0), float(y0), float(x1), float(y1)])  # xyxy, 224-frame
        self.anat_idx = anat_idx
        self.register_buffer("_boxes_xyxy", torch.tensor(boxes, dtype=torch.float32))
        # Full-face box for the "roi" global mode.
        self.register_buffer(
            "_global_box_xyxy",
            torch.tensor([[0.0, 0.0, float(src_coord_size), float(src_coord_size)]], dtype=torch.float32),
        )

    @property
    def num_regions(self) -> int:
        return len(self.regions)

    def _fc(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply AdaFace's FC head to a (N, 512, fc_grid, fc_grid) tensor → (N, 512)."""
        return self.backbone.output_layer(grid)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) aligned face → (B, K, 512) region tokens.

        H/W can be any multiple of 16 (224 by default → 14×14 map). The region
        ROIs are sampled from that map; the global token is either the exact
        112-face AdaFace embedding ("exact") or the full-box ROI ("roi").
        """
        B = img.shape[0]
        K = self.num_regions

        # One conv-body pass at the native resolution → high-res feature map.
        fmap = self.backbone.forward_spatial(img)  # (B, 512, Hf, Wf)
        feat_size = fmap.shape[-1]
        spatial_scale = feat_size / float(self.src_coord_size)

        out: List[Optional[torch.Tensor]] = [None] * K

        # Anatomical region tokens: ROI-Align (high-res map → 7×7) → FC.
        if self.anat_idx:
            boxes = self._boxes_xyxy.to(fmap.dtype)
            roi = roi_align(
                fmap, [boxes for _ in range(B)],
                output_size=self.fc_grid,
                spatial_scale=spatial_scale,
                aligned=True,
            )  # (B * n_anat, 512, 7, 7)
            emb = self._fc(roi).view(B, len(self.anat_idx), -1)
            for j, i in enumerate(self.anat_idx):
                out[i] = emb[:, j]

        # Global token.
        if self.global_token_mode == "exact":
            # Genuine AdaFace embedding of the whole face at its native 112.
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
        else:  # "roi": full-face box pooled from the high-res map → FC.
            gbox = self._global_box_xyxy.to(fmap.dtype)
            groi = roi_align(
                fmap, [gbox for _ in range(B)],
                output_size=self.fc_grid,
                spatial_scale=spatial_scale,
                aligned=True,
            )  # (B, 512, 7, 7)
            out[self.global_idx] = self._fc(groi)

        return torch.stack(out, dim=1)  # (B, K, 512)


def build_rgck_hires_net(
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
    roi_align_tokenizer: bool = False,  # accepted for harness compat; ignored (M15 always hi-res ROI)
    # M15 high-res tokenizer controls
    global_token_mode: str = "exact",
    fc_grid: int = 7,
) -> RGCKNet:
    """Build the M12 RGCK-Net, then swap in the high-resolution ROI-Align tokenizer.

    The backbone freeze/unfreeze recipe matches M12 (stage 4 + output_layer
    trainable by default). ``roi_align_tokenizer`` is accepted so the M12/M14
    harness can pass it, but M15 always uses the high-res tokenizer.
    """
    model = build_rgck_net(
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
        roi_align_tokenizer=False,  # build with FixedPartition, then replace
    )

    # Swap the tokenizer for the high-res ROI-Align one, reusing the SAME
    # backbone instance (its freeze/unfreeze state is already applied).
    model.tokenizer = HiResROIAlignRegionTokenizer(
        backbone=model.tokenizer.backbone,
        regions=model.tokenizer.regions,
        global_token_mode=global_token_mode,
        fc_grid=fc_grid,
    )
    model.region_names = model.tokenizer.region_names
    model.num_regions = model.tokenizer.num_regions
    # Tag for checkpoint/debug + so test.py can rebuild identically.
    model.hires_config = {
        "global_token_mode": global_token_mode,
        "fc_grid": fc_grid,
    }
    print(f"  [M15] hi-res ROI-Align tokenizer (global_token_mode={global_token_mode}, fc_grid={fc_grid})")
    return model


if __name__ == "__main__":
    # Smoke test (random init unless weights are present).
    W = _M12 / "weights" / "adaface_ir101_webface4m.pth"
    weights = str(W) if W.exists() else None
    for mode in ("exact", "roi"):
        m = build_rgck_hires_net(
            adaface_weights=weights, aux_relation_head=True,
            symmetric_forward=True, comparison_only_fusion=True,
            unfreeze_last_stage=True, global_token_mode=mode,
        )
        tot = sum(p.numel() for p in m.parameters())
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"M15 [{mode}] params total/trainable: {tot:,}/{tr:,} ({100*tr/tot:.2f}%)")
        # 224 input → 14×14 feature map → region ROIs to 7×7
        x = torch.randn(2, 3, 224, 224)
        out = m(x, x)
        logit = out[0]
        feat = m.tokenizer(x)
        print(f"M15 [{mode}] tokens {tuple(feat.shape)} (expect (2, 5, 512)); logit {tuple(logit.shape)}; forward items {len(out)}")
