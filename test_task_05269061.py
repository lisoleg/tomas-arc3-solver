#!/usr/bin/env python3
"""Test complete-pattern on task 05269061."""
import sys
import json
import numpy as np

sys.path.insert(0, '.')

from src.core.dsl_primitives import DSLElement, ProgramNode

# Load task data
with open('data/real_arc_converted/05269061.json') as f:
    task = json.load(f)

train = task.get('train', [])

# Test on each training pair
for i, pair in enumerate(train):
    inp = np.array(pair['input'][0], dtype=np.int8)
    expected = np.array(pair['output'][0], dtype=np.int8)

    print(f'Pair {i}:')
    print(f'  Input:')
    print(inp)
    print(f'  Expected Output:')
    print(expected)

    # Extract sequence from input
    seq = []
    seen = set()
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            color = inp[r, c]
            if color != 0 and color not in seen:
                seq.append(color)
                seen.add(color)

    print(f'  Extracted sequence: {seq}')

    # Try all rotations
    seq_len = len(seq)
    for rotation in range(seq_len):
        prog = ProgramNode(DSLElement("complete-pattern", {
            "strategy": "sequence",
            "rotation": rotation
        }))
        out = prog.apply(inp)

        if np.array_equal(out, expected):
            print(f'  ✓ Rotation {rotation} MATCHES!')
        else:
            print(f'  ✗ Rotation {rotation} does not match')

    print()
