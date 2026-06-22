#!/usr/bin/env python3
"""Debug param inference for task 05269061."""
import sys
import json
import numpy as np

sys.path.insert(0, '.')

from src.solver.param_inference import ParamInference
from src.core.dsl_primitives import DSLElement, ProgramNode

# Load task data
with open('data/real_arc_converted/05269061.json') as f:
    task = json.load(f)

train = task.get('train', [])

# Build demo pairs
demo_pairs = []
for pair in train:
    inp = [np.array(f, dtype=np.int8) for f in pair.get('input', [])]
    out = [np.array(f, dtype=np.int8) for f in pair.get('output', [])]
    demo_pairs.append({'input': inp, 'output': out})

# Run param inference
pi = ParamInference()
candidates = pi.infer_candidates(demo_pairs)

print(f'Total candidates: {len(candidates)}')
print()

# Filter complete-pattern candidates
cp_candidates = [c for c in candidates if c.element and c.element.name == 'complete-pattern']
print(f'complete-pattern candidates: {len(cp_candidates)}')
for c in cp_candidates:
    print(f'  {c.element.name}: {c.element.params}')

print()

# Check if any candidate matches all training pairs
def verify_program(prog, demo_pairs):
    for pair in demo_pairs:
        for i, inp in enumerate(pair['input']):
            if i < len(pair['output']):
                try:
                    pred = prog.apply(inp)
                    if not np.array_equal(pred, pair['output'][i]):
                        return False
                except Exception as e:
                    return False
    return True

print('Checking which candidates match all training pairs...')
for c in cp_candidates:
    if verify_program(c, demo_pairs):
        print(f'  ✓ {c.element.name}: {c.element.params} MATCHES!')
    else:
        # Check which pair fails
        for i, pair in enumerate(demo_pairs):
            inp = pair['input'][0]
            expected = pair['output'][0]
            pred = c.apply(inp)
            if not np.array_equal(pred, expected):
                print(f'  ✗ {c.element.name}: {c.element.params} fails on pair {i}')
                break
