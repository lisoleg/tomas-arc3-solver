#!/usr/bin/env python
"""Generate 50+ diverse ARC tasks for reliable benchmark evaluation.

Creates tasks with varying complexity:
- Simple transformations (10 tasks)
- Medium complexity (20 tasks)  
- Hard problems (20 tasks)
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_simple_task(task_id: int, transform_type: str) -> dict:
    """Create a simple ARC task with basic transformations."""
    rng = np.random.RandomState(task_id * 31 + 7)
    
    # Generate input grid (4-8 size)
    size = rng.randint(4, 9)
    input_grid = rng.randint(0, 5, size=(size, size)).tolist()
    
    # Apply transformation to get output
    input_arr = np.array(input_grid, dtype=np.int8)
    
    if transform_type == "rotate90":
        output = np.rot90(input_arr, 1).tolist()
    elif transform_type == "rotate180":
        output = np.rot90(input_arr, 2).tolist()
    elif transform_type == "flip_h":
        output = np.fliplr(input_arr).tolist()
    elif transform_type == "flip_v":
        output = np.flipud(input_arr).tolist()
    elif transform_type == "invert":
        output = (9 - input_arr).tolist()
    elif transform_type == "scale2x":
        output = input_arr.repeat(2, axis=0).repeat(2, axis=1).tolist()
    elif transform_type == "crop_center":
        h, w = input_arr.shape
        crop_h, crop_w = max(h // 2, 1), max(w // 2, 1)
        start_h, start_w = (h - crop_h)//2, (w - crop_w)//2
        output = input_arr[start_h:start_h+crop_h, start_w:start_w+crop_w].tolist()
    elif transform_type == "color_swap":
        mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
        output = np.vectorize(lambda x: mapping.get(x, x))(input_arr).tolist()
    elif transform_type == "edge_detect":
        output = input_arr.copy()
        output[1:-1, 1:-1] = 0  # Keep only edges
        output = output.tolist()
    else:  # identity
        output = input_grid
    
    return {
        "task_id": f"simple_{task_id:03d}",
        "transform": transform_type,
        "train": [{"input": [input_grid], "output": [output]}],
        "test": [{"input": [input_grid]}],
    }


def create_medium_task(task_id: int, pattern_type: str) -> dict:
    """Create medium complexity ARC task."""
    rng = np.random.RandomState(task_id * 37 + 13)
    
    size = rng.randint(6, 11)
    
    if pattern_type == "symmetry_completion":
        # Left half given, complete right half symmetrically
        half_size = size // 2
        half = rng.randint(0, 6, size=(size, half_size))
        input_grid = np.zeros((size, size), dtype=np.int8)
        input_grid[:, :half_size] = half
        output = input_grid.copy()
        # Complete right half symmetrically
        if half_size > 0:
            output[:, half_size:] = np.fliplr(half)
        
    elif pattern_type == "object_movement":
        # Move colored object from one position to another
        input_grid = np.zeros((size, size), dtype=np.int8)
        obj_size = rng.randint(2, 4)
        color = rng.randint(1, 6)
        # Place object at random position
        x1, y1 = rng.randint(0, max(1, size - obj_size), size=2)
        input_grid[x1:x1+obj_size, y1:y1+obj_size] = color
        # Output: object moved to different position
        output = input_grid.copy()
        x2, y2 = rng.randint(0, max(1, size - obj_size), size=2)
        output[x1:x1+obj_size, y1:y1+obj_size] = 0
        output[x2:x2+obj_size, y2:y2+obj_size] = color
        
    elif pattern_type == "pattern_repeat":
        # Repeat a small pattern across the grid
        pattern_size = rng.randint(2, 4)
        pattern = rng.randint(1, 6, size=(pattern_size, pattern_size))
        repeats_h = max(1, size // pattern_size)
        repeats_v = max(1, size // pattern_size)
        output = np.tile(pattern, (repeats_v, repeats_h))[:size, :size]
        # Input: show only some positions
        input_grid = output.copy()
        mask = rng.random((size, size)) > 0.5
        input_grid[mask] = 0
        
    elif pattern_type == "count_and_draw":
        # Count objects of each color, draw representation
        input_grid = rng.randint(0, 4, size=(size, size))
        # Output: modified based on counts
        output = input_grid.copy()
        for color in range(1, 4):
            count = np.sum(input_grid == color)
            if count > size // 2:
                output[input_grid == color] = 0  # Remove if too many
        
    else:  # default: color gradient
        input_grid = np.zeros((size, size), dtype=np.int8)
        for i in range(size):
            input_grid[i, :] = (i * 2) % 10
        output = input_grid.copy()
    
    return {
        "task_id": f"medium_{task_id:03d}",
        "pattern": pattern_type,
        "train": [{"input": [input_grid.tolist()], "output": [output.tolist()]}],
        "test": [{"input": [input_grid.tolist()]}],
    }


def create_hard_task(task_id: int, logic_type: str) -> dict:
    """Create hard ARC task requiring multi-step reasoning."""
    rng = np.random.RandomState(task_id * 41 + 23)
    
    size = rng.randint(8, 13)
    
    if logic_type == "recursive_pattern":
        # Fractal-like recursive pattern
        input_grid = np.zeros((size, size), dtype=np.int8)
        # Fill with recursive squares
        def fill_recursive(x, y, s, depth):
            if depth == 0 or s < 2:
                return
            color = (depth * 2) % 10
            input_grid[x:x+s, y:y+s] = color
            input_grid[x+1:x+s-1, y+1:y+s-1] = 0
            half = s // 2
            if half >= 2:
                fill_recursive(x, y, half, depth - 1)
                fill_recursive(x, y + half, half, depth - 1)
                fill_recursive(x + half, y, half, depth - 1)
                fill_recursive(x + half, y + half, half, depth - 1)
        
        fill_recursive(0, 0, size, 4)
        output = input_grid.copy()
        
    elif logic_type == "multi_step_transform":
        # Requires 3+ steps to solve
        input_grid = rng.randint(0, 7, size=(size, size))
        output = input_grid.copy()
        # Step 1: Rotate 90
        output = np.rot90(output, 1)
        # Step 2: Invert colors
        output = (9 - output) % 10
        # Step 3: Remove center cross
        center = size // 2
        output[center, :] = 0
        output[:, center] = 0
        output = output.tolist()
        input_grid = input_grid.tolist()
        
    elif logic_type == "conditional_fill":
        # Fill regions based on complex conditions
        input_grid = rng.randint(0, 3, size=(size, size))
        output = input_grid.copy()
        # Fill connected regions of color 1 with color 5
        visited = np.zeros_like(output, dtype=bool)
        for i in range(size):
            for j in range(size):
                if output[i, j] == 1 and not visited[i, j]:
                    # BFS to find connected region
                    queue = [(i, j)]
                    region = []
                    while queue:
                        x, y = queue.pop(0)
                        if x < 0 or x >= size or y < 0 or y >= size or visited[x, y]:
                            continue
                        if output[x, y] != 1:
                            continue
                        visited[x, y] = True
                        region.append((x, y))
                        queue.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
                    if len(region) >= 4:  # Only fill larger regions
                        for x, y in region:
                            output[x, y] = 5
        output = output.tolist()
        input_grid = input_grid.tolist()
        
    else:  # default: bitwise operations
        input_grid = rng.randint(0, 16, size=(size, size))
        output = (input_grid & 7).tolist()  # Keep only lower 3 bits
        input_grid = input_grid.tolist()
    
    return {
        "task_id": f"hard_{task_id:03d}",
        "logic": logic_type,
        "train": [{"input": [input_grid], "output": [output]}],
        "test": [{"input": [input_grid]}],
    }


def main():
    """Generate 50+ tasks with diverse complexity."""
    print("Generating 50+ ARC tasks for benchmark...")
    
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    
    tasks = []
    task_counter = 16  # Start from 016 (001-015 already exist)
    
    # 10 simple tasks
    simple_transforms = [
        "rotate90", "rotate180", "flip_h", "flip_v", "invert",
        "scale2x", "crop_center", "color_swap", "edge_detect", "identity"
    ]
    for i, transform in enumerate(simple_transforms):
        task = create_simple_task(task_counter, transform)
        task_path = data_dir / f"task_{task_counter:03d}.json"
        with open(task_path, "w") as f:
            json.dump(task, f, indent=2)
        tasks.append(task_path.name)
        task_counter += 1
    
    print(f"  Generated {len(simple_transforms)} simple tasks")
    
    # 20 medium tasks
    medium_patterns = [
        "symmetry_completion", "object_movement", "pattern_repeat",
        "count_and_draw", "symmetry_completion", "object_movement",
        "pattern_repeat", "count_and_draw", "symmetry_completion",
        "object_movement", "pattern_repeat", "count_and_draw",
        "symmetry_completion", "object_movement", "pattern_repeat",
        "count_and_draw", "symmetry_completion", "object_movement",
        "pattern_repeat", "count_and_draw"
    ]
    for i, pattern in enumerate(medium_patterns):
        task = create_medium_task(task_counter, pattern)
        task_path = data_dir / f"task_{task_counter:03d}.json"
        with open(task_path, "w") as f:
            json.dump(task, f, indent=2)
        tasks.append(task_path.name)
        task_counter += 1
    
    print(f"  Generated 20 medium tasks")
    
    # 20 hard tasks
    hard_logics = [
        "recursive_pattern", "multi_step_transform", "conditional_fill",
        "recursive_pattern", "multi_step_transform", "conditional_fill",
        "recursive_pattern", "multi_step_transform", "conditional_fill",
        "recursive_pattern", "multi_step_transform", "conditional_fill",
        "recursive_pattern", "multi_step_transform", "conditional_fill",
        "recursive_pattern", "multi_step_transform", "conditional_fill",
        "recursive_pattern", "multi_step_transform"
    ]
    for i, logic in enumerate(hard_logics):
        task = create_hard_task(task_counter, logic)
        task_path = data_dir / f"task_{task_counter:03d}.json"
        with open(task_path, "w") as f:
            json.dump(task, f, indent=2)
        tasks.append(task_path.name)
        task_counter += 1
    
    print(f"  Generated 20 hard tasks")
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Task files: {tasks[:5]} ... {tasks[-5:]}")


if __name__ == "__main__":
    main()
