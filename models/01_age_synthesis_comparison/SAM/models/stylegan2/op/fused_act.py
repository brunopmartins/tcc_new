"""
Pure PyTorch implementation of fused leaky ReLU (no CUDA compilation needed).
"""
import torch
from torch import nn
import torch.nn.functional as F


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    """Pure PyTorch fused leaky ReLU with bias."""
    if bias is not None:
        # Reshape bias for broadcasting
        rest_dim = [1] * (input.ndim - 2)
        out = F.leaky_relu(
            input + bias.view(1, bias.shape[0], *rest_dim),
            negative_slope=negative_slope
        )
    else:
        out = F.leaky_relu(input, negative_slope=negative_slope)
    
    return out * scale
