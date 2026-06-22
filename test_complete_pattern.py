#!/usr/bin/env python3
"""Test complete-pattern primitive."""
import sys
import numpy as np

sys.path.insert(0, '.')

from src.core.dsl_primitives import DSLElement, ProgramNode

# Test case: task 05269061
# Input: grid with seed pattern [2, 8, 3]
# Output: periodic tiling of [2, 8, 3]

# Create input grid (seed pattern in top-left)
inp = np.array([
    [2, 8, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=np.int8)

print("Input grid:")
print(inp)

# Test complete-pattern with different rotations
for rotation in range(3):
    prog = ProgramNode(DSLElement("complete-pattern", {"rotation": rotation}))
    out = prog.apply(inp)
    print(f"\nOutput (rotation={rotation}):")
    print(out)

# Expected output (for training pair 0):
# [2, 8, 3, 2, 8, 3]
# [2, 8, 3, 2, 8, 3]
# [2, 8, 3, 2, 8, 3]
expected = np.array([
    [2, 8, 3, 2, 8, 3],
    [2, 8, 3, 2, 8, 3],
    [2, 8, 3, 2, 8, 3],
], dtype=np.int8)

print("\nExpected output (rotation=0):")
print(expected)

# Check which rotation matches
for rotation in range(3):
    prog = ProgramNode(DSLElement("complete-pattern", {"rotation": rotation}))
    out = prog.apply(inp)
    if np.array_equal(out, expected):
        print(f"\n✓ Rotation {rotation} matches expected output!")
    else:
        print(f"\n✗ Rotation {rotation} does NOT match")
