"""Model 16 — FA-RGCK: Family-Adversarial Region-Guided Cross Kinship Network.

The single binding constraint across all M00-M15 models is **generalisation to
unseen families** — the val→test gap. R012's diagnostic showed the M12 stack
memorises training families by ~epoch 3 (Val AUC peaks then declines while train
loss keeps falling). Every prior generalisation gain was *indirect*: shortcut
removal (R006 symmetric forward, R009 comparison-only fusion), LoRA
regularisation (M14), in-distribution region tokens (M15).

M16 attacks the cause **directly** with domain-adversarial training, treating
each FIW family as a domain (Ganin & Lempitsky, DANN, 2015):

    - Reuse the proven M12 R011 head verbatim (region tokens, cross-region
      attention, gate, fusion classifier, symmetric forward, comparison-only
      fusion, relation aux) — produces the per-face contextualised global
      tokens gA, gB.
    - A **family discriminator** tries to predict which training family each
      face belongs to from gA / gB.
    - A **Gradient Reversal Layer (GRL)** sits between the features and the
      discriminator: forward is identity, backward multiplies the gradient by
      −λ. So the discriminator learns to predict family, while the backbone +
      cross-region adapter are pushed to make family **unrecoverable** — i.e.
      to keep only the *pairwise kinship* signal, not the family identity that
      doesn't transfer across the train/test split.

    L = BCE_kin + 0.05·CE_rel  +  family_weight · CE_family
                                   └─ GRL reverses its gradient into the backbone

λ follows the standard DANN schedule (0 → 1 over training), so the model starts
exactly at the M12 R011 solution (λ=0 ⇒ no family pressure ⇒ floor = R011) and
ramps the invariance pressure as training proceeds. The hypothesis: removing
family-identifiable features closes more of the val→test gap than the indirect
shortcut-removal tricks, breaking the 0.876-0.884 ceiling.

``build_fa_rgck_net`` builds the M12 model and wraps it; the kinship output is
the **unchanged M12 tuple** (so eval/metrics/base-loss are untouched) with a
family-logits dict appended as the last element for the training loss.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# Reuse the M12 architecture (RGCKNet, build_rgck_net) verbatim.
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


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (DANN)
# ---------------------------------------------------------------------------

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse + scale the gradient flowing back into the feature extractor.
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


# ---------------------------------------------------------------------------
# Family discriminator (domain classifier)
# ---------------------------------------------------------------------------

class FamilyAdversary(nn.Module):
    """MLP that predicts the FIW family id from a per-face global token.

    No BatchNorm (avoids train/eval-stat coupling with the adversarial signal);
    LayerNorm + GELU + dropout, then a linear to ``num_families`` logits.
    """

    def __init__(self, dim: int = 512, hidden: int = 256, num_families: int = 600, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_families),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# FA-RGCK: M12 head + family-adversarial branch
# ---------------------------------------------------------------------------

class FARGCKNet(nn.Module):
    """M12 RGCK-Net + a family-adversarial branch on the per-face global tokens.

    Forward returns the **unchanged M12 output tuple** with a family-logits dict
    appended as the last element (only in symmetric mode, which M16 requires):

        (logit, weights, attn, gA_norm, gB_norm, rel_logits, sym_extras,
         {"family_logits_a": ..., "family_logits_b": ...})

    The kinship logit stays at index 0 and sym_extras at index 6, so
    RGCKBCELoss and the eval/metrics paths are unaffected. The training loss
    reads ``outputs[-1]`` for the family logits.

    ``grl_lambda`` (set by the trainer per the DANN schedule) controls the
    gradient-reversal strength: 0 ⇒ no pressure on the backbone ⇒ identical to
    the wrapped M12 model.
    """

    def __init__(self, rgck: RGCKNet, num_families: int, adv_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        if not getattr(rgck, "symmetric_forward", False):
            raise ValueError(
                "FA-RGCK requires the wrapped RGCKNet to use symmetric_forward "
                "(M16 R001 recipe). Build with symmetric_forward=True."
            )
        self.rgck = rgck
        self.num_families = int(num_families)
        self.embedding_dim = rgck.embedding_dim
        self.symmetric_forward = rgck.symmetric_forward
        self.comparison_only_fusion = getattr(rgck, "comparison_only_fusion", False)
        self.family_adv = FamilyAdversary(
            dim=rgck.embedding_dim, hidden=adv_hidden,
            num_families=self.num_families, dropout=dropout,
        )
        # DANN reversal strength; set by the trainer each epoch (0 → 1).
        self.grl_lambda = 0.0

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> Tuple:
        outputs = self.rgck(img_a, img_b)  # M12 6/7-tuple (7 in symmetric mode)

        # Per-face contextualised global tokens (AB direction), L2-normalised.
        gA = outputs[3]
        gB = outputs[4]
        fam_logits_a = self.family_adv(grad_reverse(gA, self.grl_lambda))
        fam_logits_b = self.family_adv(grad_reverse(gB, self.grl_lambda))

        family_extras = {
            "family_logits_a": fam_logits_a,
            "family_logits_b": fam_logits_b,
        }
        return (*outputs, family_extras)


def build_fa_rgck_net(
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
    symmetric_forward: bool = True,
    comparison_only_fusion: bool = False,
    roi_align_tokenizer: bool = False,
    # M16 family-adversarial controls
    num_families: int = 600,
    adv_hidden: int = 256,
) -> FARGCKNet:
    """Build the M12 RGCK-Net (R011 stack) and wrap it with the family adversary.

    ``num_families`` must match the training family vocabulary size (set by the
    trainer from the actual train split). ``symmetric_forward`` defaults to True
    — required by FA-RGCK.
    """
    rgck = build_rgck_net(
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
        roi_align_tokenizer=roi_align_tokenizer,
    )
    model = FARGCKNet(rgck, num_families=num_families, adv_hidden=adv_hidden, dropout=dropout)
    model.fa_config = {"num_families": int(num_families), "adv_hidden": int(adv_hidden)}
    print(f"  [M16] family adversary: {num_families} families, hidden={adv_hidden} (GRL, λ set by DANN schedule)")
    return model


if __name__ == "__main__":
    # Smoke test (random init unless weights present).
    W = _M12 / "weights" / "adaface_ir101_webface4m.pth"
    weights = str(W) if W.exists() else None
    m = build_fa_rgck_net(
        adaface_weights=weights, aux_relation_head=True,
        symmetric_forward=True, comparison_only_fusion=True,
        unfreeze_last_stage=True, num_families=571,
    )
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    adv = sum(p.numel() for p in m.family_adv.parameters())
    print(f"M16 params total/trainable: {tot:,}/{tr:,} ({100*tr/tot:.2f}%); adversary {adv:,}")

    x = torch.randn(2, 3, 224, 224)
    # λ=0 (start of DANN schedule): identical kinship path to M12 R011.
    m.grl_lambda = 0.0
    out = m(x, x)
    logit = out[0]
    fam = out[-1]
    print(f"forward items {len(out)} (8 = M12 7-tuple + family dict)")
    print(f"  logit {tuple(logit.shape)}; sym_extras@6 is dict: {isinstance(out[6], dict)}")
    print(f"  family_logits_a {tuple(fam['family_logits_a'].shape)} (expect (2, 571))")

    # λ>0: gradient reversal active. Check a backward step runs.
    m.grl_lambda = 0.5
    out = m(x, x)
    loss = out[0].sum() + out[-1]["family_logits_a"].sum()
    loss.backward()
    print("  backward with GRL λ=0.5 OK")
