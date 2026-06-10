"""LoRA adapters for the AdaFace IR-101 backbone (Model 14).

M14 = M12 RGCK-Net with the backbone adapted by **low-rank adapters** instead
of full unfreeze of stage 4. Motivation: the M12 diagnosis showed Val AUC peaks
at epoch ~3 and then memorises (train loss keeps falling while Val AUC drops) —
the binding constraint is generalisation, not head capacity. Full unfreeze of
stage 4 (~31 M trainable params) gives the model lots of room to memorise
training families. LoRA adds a small, low-rank, regularised adaptation surface
(~1-2 M params) to the same layers, which should adapt the kinship
representation with far less memorisation.

LoRA delta is zero at init (B / up-conv initialised to zero), so training starts
exactly at the frozen-AdaFace solution and departs gradually.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a low-rank additive update B@A."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B stays zero -> initial delta is exactly zero.
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.drop(x) @ self.A.t()) @ self.B.t()
        return self.base(x) + delta * self.scaling


class LoRAConv2d(nn.Module):
    """Wrap a frozen nn.Conv2d with a low-rank update: a rank-channel
    down-conv (same kernel/stride/padding as the base) followed by a 1x1
    up-conv. Up-conv is zero-initialised so the initial delta is zero."""

    def __init__(self, base: nn.Conv2d, rank: int = 16, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.down = nn.Conv2d(
            base.in_channels, rank, kernel_size=base.kernel_size,
            stride=base.stride, padding=base.padding, bias=False,
        )
        self.up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.up(self.down(x)) * self.scaling


def _replace_in(module: nn.Module, rank: int, alpha: int) -> int:
    """Recursively replace Conv2d/Linear children of ``module`` (in place) with
    their LoRA-wrapped versions. Returns the count replaced."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear) or isinstance(child, LoRAConv2d):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank, alpha))
            n += 1
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            n += 1
        else:
            n += _replace_in(child, rank, alpha)
    return n


def inject_lora_into_adaface(
    backbone: nn.Module,
    rank: int = 16,
    alpha: int = 16,
    stage4: bool = True,
    output_layer: bool = True,
    stage3_tail: bool = False,
) -> int:
    """Freeze the whole AdaFace backbone, then inject LoRA into the chosen
    regions (mutating ``backbone.body[i]`` / ``backbone.output_layer`` in place).

    Targets mirror the M12 unfreeze surface:
      - stage4       -> body[46:49] (the deepest 3 BasicBlockIR blocks)
      - stage3_tail  -> body[43:46]
      - output_layer -> the FC head Linear(512*7*7 -> 512)

    Returns the number of base layers wrapped.
    """
    for p in backbone.parameters():
        p.requires_grad = False

    n = 0
    if stage4:
        for i in range(46, 49):
            n += _replace_in(backbone.body[i], rank, alpha)
    if stage3_tail:
        for i in range(43, 46):
            n += _replace_in(backbone.body[i], rank, alpha)
    if output_layer:
        n += _replace_in(backbone.output_layer, rank, alpha)
    return n
