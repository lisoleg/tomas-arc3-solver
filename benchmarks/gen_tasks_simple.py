#!/usr/bin/env python
"""Quick task generator - 50 simple ARC-like tasks."""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def gen_task(task_id: int) -> dict:
    """Generate a single ARC-like task."""
    rng = np.random.RandomState(task_id * 31 + 7)
    size = rng.randint(4, 9) * 2  # Even size to avoid symmetry issues
    inp = rng.randint(0, 5, size=(size, size))
    
    # Simple transform: rotate 90 if task_id even, flip if odd
    if task_id % 2 == 0:
        out = np.rot90(inp, 1)
    else:
        out = np.fliplr(inp)
    
    return {
        "task_id": f"task_{task_id:03d}",
        "train": [{"input": [inp.tolist()], "output": [out.tolist()]}],
        "test": [{"input": [inp.tolist()]}],
    }

def main():
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Generate 50 tasks (skip existing 001-015)
    count = 0
    for i in range(16, 66):
        task = gen_task(i)
        path = data_dir / f"task_{i:03d}.json"
        with open(path, "w") as f:
            json.dump(task, f, indent=2)
        count += 1
    
    print(f"Generated {count} tasks (task_016 to task_{i:03d})")

if __name__ == "__main__":
    main()
