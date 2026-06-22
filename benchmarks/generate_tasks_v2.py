#!/usr/bin/env python
"""Generate solvable synthetic ARC tasks using actual DSL primitives.

Each task is created by applying a single DSL primitive to random input grids.
This guarantees the solver can find the transformation program.

Task format matches the internal VideoARCTask parser:
  {"input": [grid], "output": [grid]}  (list of 2D grids)
"""

import json
import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dsl_primitives import get_all_primitives, ProgramNode, DSLElement


def make_random_grid(rows: int, cols: int, max_color: int = 4, density: float = 0.6) -> np.ndarray:
    """Generate a random grid with given parameters."""
    grid = np.zeros((rows, cols), dtype=np.int8)
    mask = np.random.random((rows, cols)) < density
    grid[mask] = np.random.randint(1, max_color + 1, size=mask.sum())
    return grid


def make_task_with_primitive(
    task_id: str,
    prim: DSLElement,
    num_train: int = 3,
    grid_sizes: list[tuple[int, int]] | None = None,
) -> dict | None:
    """Create a task using a single DSL primitive.

    Args:
        task_id: Task identifier.
        prim: DSL primitive element.
        num_train: Number of training pairs.
        grid_sizes: List of (rows, cols) for training grids.

    Returns:
        Task dict or None if primitive fails.
    """
    if grid_sizes is None:
        grid_sizes = [(4, 4), (5, 5), (4, 6), (6, 4), (5, 6)]

    node = ProgramNode(prim)
    train_pairs = []

    for i in range(num_train):
        rows, cols = grid_sizes[i % len(grid_sizes)]
        input_grid = make_random_grid(rows, cols, max_color=4, density=0.6)

        try:
            output_grid = node.apply(input_grid)
            if output_grid is None or output_grid.size == 0:
                return None
            # Ensure output is int8 2D array
            output_grid = np.asarray(output_grid, dtype=np.int8)
            if output_grid.ndim != 2:
                return None
        except Exception:
            return None

        train_pairs.append({
            "input": [input_grid.tolist()],
            "output": [output_grid.tolist()],
        })

    # Create test pair (different grid size if possible)
    test_rows, test_cols = (3, 3) if (4, 4) in grid_sizes else grid_sizes[0]
    test_input = make_random_grid(test_rows, test_cols, max_color=4, density=0.6)
    try:
        test_output = node.apply(test_input)
        test_output = np.asarray(test_output, dtype=np.int8)
        if test_output.ndim != 2:
            return None
    except Exception:
        return None

    test_pairs = [{
        "input": [test_input.tolist()],
        "output": [test_output.tolist()],
    }]

    return {
        "task_id": task_id,
        "train": train_pairs,
        "test": test_pairs,
    }


def generate_all_tasks() -> dict[str, dict]:
    """Generate tasks for all viable DSL primitives."""
    prims = get_all_primitives()
    tasks = {}

    # Primitives that are known to work well for task generation
    good_primitives = [
        "color-swap", "mirror", "rotate", "fill-region",
        "boundary-detect", "scale", "copy", "map-color",
        "complete-shape", "crop",
    ]

    task_num = 100
    for prim in prims:
        if prim.name not in good_primitives:
            continue

        # Try to generate a task with this primitive
        for attempt in range(3):
            np.random.seed(42 + task_num + attempt)
            task = make_task_with_primitive(
                f"task_{task_num:03d}",
                prim,
                num_train=3,
            )
            if task is not None:
                tasks[f"task_{task_num:03d}"] = task
                break
        task_num += 1

    return tasks


def main():
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(exist_ok=True)

    tasks = generate_all_tasks()
    for task_id, task_data in tasks.items():
        filepath = output_dir / f"{task_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)
        print(f"  Generated: {filepath.name} (primitive: {task_data.get('task_id', 'unknown')})")

    print(f"\nTotal: {len(tasks)} tasks generated in {output_dir}")


if __name__ == "__main__":
    main()
