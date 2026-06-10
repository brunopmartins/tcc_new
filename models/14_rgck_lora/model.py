"""Model 14 — RGCK-Net + LoRA backbone adaptation.

M14 reuses the M12 RGCK-Net architecture verbatim (region tokens, bidirectional
cross-region attention, sigmoid gating, fusion classifier, symmetric forward,
comparison-only fusion, relation-aux head) but adapts the AdaFace IR-101 backbone
with LoRA adapters instead of full-unfreezing stage 4.

Rationale: see ``lora.py``. The M12 line memorises training families by ~epoch 3;
LoRA's low-rank, regularised adaptation surface (~1-2 M trainable params vs ~31 M
for the stage-4 unfreeze) aims to adapt the kinship representation with much less
memorisation, raising the generalisable ceiling.

``build_rgck_lora_net`` has the same signature knobs as M12's ``build_rgck_net``
plus the LoRA controls; it builds the M12 model with a FROZEN backbone
(unfreeze_last_stage=False) and then injects LoRA.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Reuse the M12 architecture (RGCKNet, tokenizers, build_rgck_net, region defs).
# M12's file is also called ``model.py``; load it under a distinct module name
# to avoid a circular import with this file.
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

from lora import inject_lora_into_adaface  # noqa: E402  (local M14)


def build_rgck_lora_net(
    adaface_weights: Optional[str] = None,
    embedding_dim: int = 512,
    regions: List[Tuple[str, Tuple[int, int, int, int]]] = None,
    cross_attn_heads: int = 4,
    cross_attn_layers: int = 1,
    gate_hidden: int = 128,
    classifier_hidden: int = 512,
    dropout: float = 0.2,
    aux_relation_head: bool = False,
    num_relation_classes: int = 11,
    symmetric_forward: bool = False,
    comparison_only_fusion: bool = False,
    roi_align_tokenizer: bool = False,
    # LoRA controls
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_stage4: bool = True,
    lora_stage3_tail: bool = False,
    lora_output_layer: bool = True,
) -> RGCKNet:
    """Build the M12 RGCK-Net with a frozen AdaFace backbone, then inject LoRA.

    The backbone is built frozen (freeze_backbone=True, unfreeze_last_stage=False)
    so the only trainable backbone params are the injected LoRA adapters; the
    RGCK head (cross-region adapter, gate, classifier, relation head) trains as
    usual.
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
        freeze_backbone=True,
        unfreeze_last_stage=False,
        unfreeze_extra_stage3_tail=False,
        aux_relation_head=aux_relation_head,
        num_relation_classes=num_relation_classes,
        symmetric_forward=symmetric_forward,
        comparison_only_fusion=comparison_only_fusion,
        roi_align_tokenizer=roi_align_tokenizer,
    )

    n = inject_lora_into_adaface(
        model.tokenizer.backbone,
        rank=lora_rank,
        alpha=lora_alpha,
        stage4=lora_stage4,
        output_layer=lora_output_layer,
        stage3_tail=lora_stage3_tail,
    )
    # Tag for checkpoint/debug.
    model.lora_config = {
        "lora_rank": lora_rank, "lora_alpha": lora_alpha,
        "lora_stage4": lora_stage4, "lora_stage3_tail": lora_stage3_tail,
        "lora_output_layer": lora_output_layer, "lora_layers_wrapped": n,
    }
    return model


if __name__ == "__main__":
    import torch
    W = str(_M12 / "weights/adaface_ir101_webface4m.pth")
    m = build_rgck_lora_net(adaface_weights=W, aux_relation_head=True,
                            symmetric_forward=True, comparison_only_fusion=True,
                            lora_rank=16, lora_alpha=16)
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"M14 params total/trainable: {tot:,}/{tr:,} ({100*tr/tot:.2f}%)")
    print("lora_config:", m.lora_config)
    x = torch.randn(2, 3, 224, 224)
    out = m(x, x)
    print("forward returns", len(out), "items; logit", tuple(out[0].shape))
