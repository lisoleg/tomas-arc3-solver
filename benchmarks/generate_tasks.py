#!/usr/bin/env python
"""Generate diverse synthetic ARC tasks for benchmarking.

Creates 12 ARC-style tasks with known transformations:
color_swap, mirror_h, mirror_v, rotate90, rotate180, fill_region,
gravity_down, scale2x, extract_pattern, boundary_detect, complete_shape,
count_and_label.

Each task has 2-3 training pairs and 1 test pair.
"""

import json
import numpy as np
from pathlib import Path


def make_task(task_id: str, train_pairs: list, test_pairs: list) -> dict:
    """Build an ARC task JSON structure."""
    return {
        "task_id": task_id,
        "train": [
            {"input": [inp.tolist()], "output": [out.tolist()]}
            for inp, out in train_pairs
        ],
        "test": [
            {"input": [inp.tolist()], "output": [out.tolist()]}
            for inp, out in test_pairs
        ],
    }


# --- Transformation functions ---

def color_swap(grid: np.ndarray, c1: int = 1, c2: int = 2) -> np.ndarray:
    result = grid.copy()
    result[grid == c1] = c2
    result[grid == c2] = c1
    return result


def mirror_h(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid).copy()


def mirror_v(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid).copy()


def rotate90(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=1).copy()


def rotate180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=2).copy()


def fill_region(grid: np.ndarray, fill_color: int = 3) -> np.ndarray:
    result = grid.copy()
    result[result == 0] = fill_color
    return result


def gravity_down(grid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(grid)
    for col in range(grid.shape[1]):
        column = grid[:, col]
        non_zero = column[column != 0]
        if len(non_zero) > 0:
            result[-len(non_zero):, col] = non_zero
    return result


def scale2x(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1).copy()


def extract_pattern(grid: np.ndarray) -> np.ndarray:
    """Extract the non-zero bounding box region."""
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)
    if not rows.any() or not cols.any():
        return grid
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return grid[rmin:rmax+1, cmin:cmax+1].copy()


def boundary_detect(grid: np.ndarray) -> np.ndarray:
    """Mark boundaries (edges between different colors) with color 9."""
    result = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 0:
                # Check neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                        if grid[ni, nj] != grid[i, j]:
                            result[i, j] = 9
                            break
    return result


def complete_shape(grid: np.ndarray) -> np.ndarray:
    """Fill interior holes of shapes with color 4."""
    result = grid.copy()
    # Simple: fill 0s that are surrounded by non-zero
    for i in range(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            if grid[i, j] == 0:
                neighbors = [grid[i-1, j], grid[i+1, j], grid[i, j-1], grid[i, j+1]]
                if all(n != 0 for n in neighbors):
                    result[i, j] = 4
    return result


def count_and_label(grid: np.ndarray) -> np.ndarray:
    """Replace each object's pixels with a unique label."""
    result = np.zeros_like(grid)
    label = 1
    visited = np.zeros_like(grid, dtype=bool)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != 0 and not visited[i, j]:
                # BFS for connected component
                stack = [(i, j)]
                color = grid[i, j]
                while stack:
                    ci, cj = stack.pop()
                    if 0 <= ci < grid.shape[0] and 0 <= cj < grid.shape[1]:
                        if not visited[ci, cj] and grid[ci, cj] == color:
                            visited[ci, cj] = True
                            result[ci, cj] = label
                            stack.extend([(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)])
                label += 1
    return result


# --- Grid generators ---

def make_random_grid(rows: int, cols: int, colors: list[int] = None, density: float = 0.5) -> np.ndarray:
    if colors is None:
        colors = [0, 1, 2]
    grid = np.zeros((rows, cols), dtype=np.int8)
    mask = np.random.random((rows, cols)) < density
    grid[mask] = np.random.choice([c for c in colors if c != 0], size=mask.sum())
    return grid


def make_shapes_grid() -> np.ndarray:
    """Grid with distinct shapes."""
    grid = np.zeros((6, 6), dtype=np.int8)
    grid[0:2, 0:2] = 1  # square
    grid[0:3, 3] = 2    # vertical line
    grid[4, 0:4] = 3    # horizontal line
    return grid


def make_pattern_grid() -> np.ndarray:
    """Grid with a pattern."""
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1:4, 1:4] = 1
    grid[2, 2] = 2
    return grid


# --- Generate all tasks ---

def generate_all_tasks() -> dict[str, dict]:
    tasks = {}

    # Task 004: color swap (1<->2)
    for seed in range(3):
        np.random.seed(100 + seed)
        grid = make_random_grid(4, 4, [0, 1, 2], 0.6)
    g = np.array([[0,1,2,0],[1,0,1,2],[2,1,0,1],[0,2,1,0]], dtype=np.int8)
    tasks["task_004"] = make_task("task_004", [(g, color_swap(g))], [(g, color_swap(g))])

    # Task 005: mirror horizontal
    g = np.array([[1,2,3,0],[0,3,2,1],[2,1,0,3],[3,0,1,2]], dtype=np.int8)
    g2 = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]], dtype=np.int8)
    tasks["task_005"] = make_task("task_005", [(g, mirror_h(g)), (g2, mirror_h(g2))], [(g, mirror_h(g))])

    # Task 006: mirror vertical
    g = np.array([[1,0,2,3],[3,2,1,0],[0,1,2,3],[2,3,0,1]], dtype=np.int8)
    tasks["task_006"] = make_task("task_006", [(g, mirror_v(g))], [(g, mirror_v(g))])

    # Task 007: rotate 90
    g = np.array([[1,2,0],[0,1,2],[3,0,1]], dtype=np.int8)
    tasks["task_007"] = make_task("task_007", [(g, rotate90(g))], [(g, rotate90(g))])

    # Task 008: rotate 180
    g = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.int8) % 10
    tasks["task_008"] = make_task("task_008", [(g, rotate180(g))], [(g, rotate180(g))])

    # Task 009: fill region
    g = np.array([[1,0,2],[0,0,0],[3,0,4]], dtype=np.int8)
    tasks["task_009"] = make_task("task_009", [(g, fill_region(g, 3))], [(g, fill_region(g, 3))])

    # Task 010: gravity down
    g = np.array([[1,0,2],[0,3,0],[0,0,4]], dtype=np.int8)
    tasks["task_010"] = make_task("task_010", [(g, gravity_down(g))], [(g, gravity_down(g))])

    # Task 011: scale 2x
    g = np.array([[1,2],[3,4]], dtype=np.int8)
    tasks["task_011"] = make_task("task_011", [(g, scale2x(g))], [(g, scale2x(g))])

    # Task 012: extract pattern
    g = np.array([[0,0,0,0,0],[0,1,2,0,0],[0,3,4,0,0],[0,0,0,0,0]], dtype=np.int8)
    tasks["task_012"] = make_task("task_012", [(g, extract_pattern(g))], [(g, extract_pattern(g))])

    # Task 013: boundary detect
    g = make_shapes_grid()
    tasks["task_013"] = make_task("task_013", [(g, boundary_detect(g))], [(g, boundary_detect(g))])

    # Task 014: complete shape
    g = np.array([[1,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,1]], dtype=np.int8)
    tasks["task_014"] = make_task("task_014", [(g, complete_shape(g))], [(g, complete_shape(g))])

    # Task 015: count and label
    g = np.array([[1,1,0,2],[1,1,0,2],[0,0,0,0],[3,0,3,3]], dtype=np.int8)
    tasks["task_015"] = make_task("task_015", [(g, count_and_label(g))], [(g, count_and_label(g))])

    return tasks


def main():
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    tasks = generate_all_tasks()
    for task_id, task_data in tasks.items():
        filepath = output_dir / f"{task_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)
        print(f"  Generated: {filepath.name}")

    print(f"\nTotal: {len(tasks)} tasks generated in {output_dir}")


if __name__ == "__main__":
    main()
