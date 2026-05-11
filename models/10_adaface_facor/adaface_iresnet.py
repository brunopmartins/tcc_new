"""
AdaFace IR-101 backbone (cvlface variant).

Vendored, minimally trimmed from
    https://huggingface.co/minchul/cvlface_adaface_ir101_webface4m
    models/iresnet/model.py
which itself derives from
    https://github.com/mk-minchul/AdaFace
    net.py
(MIT licensed, Minchul Kim).

Differences vs. the InsightFace iresnet100 used by M08:
- `BasicBlockIR` has `res_layer = Seq(BN, Conv, BN, PReLU, Conv, BN)` and
  `shortcut_layer = MaxPool2d(1, stride)` when in_channel == depth (identity
  with downsampling). InsightFace uses BN+conv+BN+PReLU+conv+BN inside the
  block but `Conv2d` shortcut whenever stride != 1.
- The body is a flat `Sequential` of 49 `BasicBlockIR` units indexed
  `net.body.0..48`, NOT split into layer1/2/3/4.
- State dict prefix is `net.input_layer.*`, `net.body.*`, `net.output_layer.*`
  (the published checkpoints prepend `net.` because they are wrapped by a
  `Wrapper` module on cvlface; we strip that on load).

For M10 we need the *pre-pool* spatial features (B, 512, 7, 7) so that the
M02-style FaCoR cross-attention can operate on 49 tokens. The standard
forward instead flattens and FCs to a 512 embedding. We add
`forward_spatial` and `forward_features` for this purpose.

The IR-101 unit ordering is:
    layer1 = blocks 0..2   (depth=64,  3 units, stride=2 then 1,1)
    layer2 = blocks 3..15  (depth=128, 13 units)
    layer3 = blocks 16..45 (depth=256, 30 units)
    layer4 = blocks 46..48 (depth=512, 3 units)
The 7x7 spatial map is the output of block 48 (i.e. the full `self.body`)
before the BN/Flatten/FC head in `output_layer`.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    Sequential,
)


# ---------------------------------------------------------------------------
# Block definitions (verbatim from cvlface model.py, IR variant only).
# ---------------------------------------------------------------------------
class Flatten(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class BasicBlockIR(Module):
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class _Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    pass


def _get_block(in_channel: int, depth: int, num_units: int, stride: int = 2):
    return [_Bottleneck(in_channel, depth, stride)] + [
        _Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


def _get_blocks_ir101():
    return [
        _get_block(in_channel=64, depth=64, num_units=3),
        _get_block(in_channel=64, depth=128, num_units=13),
        _get_block(in_channel=128, depth=256, num_units=30),
        _get_block(in_channel=256, depth=512, num_units=3),
    ]


# ---------------------------------------------------------------------------
# AdaFace IR-101 backbone with spatial-feature access.
# ---------------------------------------------------------------------------
class AdaFaceIR101(Module):
    """
    AdaFace IR-101 backbone.

    Standard `forward(x)` reproduces the original AdaFace inference: returns
    a 512-dim feature (un-normalised; AdaFace's L2 normalisation is done by
    the loss head, not the backbone). Use `forward_spatial(x)` to obtain the
    7x7x512 feature map *before* the output BN/FC.

    Input: (B, 3, 112, 112) in [-1, 1] (i.e. (img/255 - 0.5) / 0.5).
    """

    SPATIAL_CHANNELS = 512
    SPATIAL_SIZE = 7
    NUM_TOKENS = SPATIAL_SIZE * SPATIAL_SIZE  # 49

    def __init__(self, output_dim: int = 512):
        super().__init__()

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )

        blocks = _get_blocks_ir101()
        modules = []
        for block in blocks:
            for bn in block:
                modules.append(BasicBlockIR(bn.in_channel, bn.depth, bn.stride))
        self.body = Sequential(*modules)

        # Standard AdaFace head — kept so weights load 1:1 and the original
        # 512-dim embedding remains accessible if ever needed.
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(0.4),
            Flatten(),
            Linear(512 * 7 * 7, output_dim),
            BatchNorm1d(output_dim, affine=False),
        )

        self.output_dim = output_dim

    # ----- forward variants -------------------------------------------------
    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 512, 7, 7) feature map — pre-pool, pre-FC."""
        x = self.input_layer(x)
        x = self.body(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return (B, 49, 512) tokenised features for cross-attention.

        Spatial layout is row-major (top-left → bottom-right).
        """
        spatial = self.forward_spatial(x)  # (B, 512, 7, 7)
        b, c, h, w = spatial.shape
        # (B, C, H, W) -> (B, H*W, C)
        return spatial.flatten(2).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard 512-dim AdaFace embedding (un-normalised)."""
        x = self.forward_spatial(x)
        x = self.output_layer(x)
        return x


def load_adaface_state_dict(model: AdaFaceIR101, state_dict: dict) -> tuple[list, list]:
    """
    Load a cvlface-format AdaFace checkpoint into AdaFaceIR101.

    The published checkpoints (e.g. minchul/cvlface_adaface_ir101_webface4m
    pretrained_model/model.pt) wrap the network as `Wrapper.net` so all keys
    are prefixed `net.`. We strip that prefix and any `module.` left from DDP.

    Returns (missing, unexpected) as from nn.Module.load_state_dict(strict=False).
    """
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("net."):
            nk = nk[len("net."):]
        cleaned[nk] = v

    return model.load_state_dict(cleaned, strict=False)


def adaface_ir101(weights_path: str | None = None) -> AdaFaceIR101:
    """
    Factory: build AdaFaceIR101 and optionally load AdaFace checkpoint.

    The checkpoint format must match the cvlface AdaFace release (state dict
    with `net.` prefix). If `weights_path` is None the backbone is randomly
    initialised — useful only for smoke tests.
    """
    model = AdaFaceIR101()
    if weights_path is None:
        return model

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    missing, unexpected = load_adaface_state_dict(model, state)
    if missing:
        print(f"  [AdaFaceIR101] missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  [AdaFaceIR101] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    return model


if __name__ == "__main__":
    m = AdaFaceIR101()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"AdaFaceIR101: {n_params:,} parameters")
    x = torch.randn(2, 3, 112, 112)
    emb = m(x)
    spatial = m.forward_spatial(x)
    tokens = m.forward_features(x)
    print(f"forward(x)         shape: {tuple(emb.shape)}")
    print(f"forward_spatial(x) shape: {tuple(spatial.shape)}")
    print(f"forward_features(x) shape: {tuple(tokens.shape)}")
