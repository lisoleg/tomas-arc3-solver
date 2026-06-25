"""NAR-Conv encoder module for TOMAS ARC-AGI-3 Solver.

Provides octonion-valued convolutional neural network components
for encoding grid states into structured feature representations.

The NAR-Conv (Non-Associative Residual Convolution) framework uses
Cayley-Dickson octonion algebra to maintain non-associative residuals
(Asym ≠ 0), distinguishing physical AI from statistical proxy AI.

Key classes:
    OctonionConv2d: Octonion-valued 2D convolution layer
    NARConvBlock: Convolutional block with BN + ReLU + Pool
    NARGridEncoder: Full grid encoder producing feature vectors + topo maps
"""

from .nar_conv import (
    OctonionConv2d,
    NARConvBlock,
    NARGridEncoder,
    color_index_to_octonion,
    OCT_MUL_TABLE,
    build_octonion_multiplication_tensors,
)

__all__ = [
    "OctonionConv2d",
    "NARConvBlock",
    "NARGridEncoder",
    "color_index_to_octonion",
    "OCT_MUL_TABLE",
    "build_octonion_multiplication_tensors",
]
