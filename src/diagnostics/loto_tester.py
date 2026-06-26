"""
LOTO (Leave-One-Topology-Out) Generalization Tester: Standalone
"""

import numpy as np
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

def classify_topology(grid: np.ndarray) -> str:
    h, w = grid.shape
    try:
        from scipy.ndimage import label
        _, num = label(grid > 0)
    except:
        num = 1
    period_x = 0
    for p in range(1, w // 2 + 1):
        if np.array_equal(grid[:, :-p], grid[:, p:]):
            period_x = p
            break
    period_y = 0
    for p in range(1, h // 2 + 1):
        if np.array_equal(grid[:-p], grid[p:]):
            period_y = p
            break
    sym_h = np.array_equal(grid, np.fliplr(grid))
    sym_v = np.array_equal(grid, np.flipud(grid))
    parts = [f"ec{num}"]
    if period_x > 0 or period_y > 0:
        parts.append(f"px{period_x}_py{period_y}")
    if sym_h:
        parts.append("sh")
    if sym_v:
        parts.append("sv")
    return "_".join(parts)

def run_loto(tasks: List[Dict]) -> Dict:
    groups = {}
    for task in tasks:
        inp = np.array(task['demos'][0]['input'], dtype=np.int32)
        topo = classify_topology(inp)
        groups.setdefault(topo, []).append(task)
    print(f"  [LOTO] Topology groups: {list(groups.keys())}")
    results = []
    for held_out, test_tasks in groups.items():
        train_tasks = [t for topo, tlist in groups.items() for t in tlist if topo != held_out]
        hits = 0
        for task in test_tasks:
            train_tags = set()
            for t in train_tasks:
                for tag in t.get('tags', []):
                    train_tags.add(tag)
            task_tags = set(task.get('tags', []))
            if train_tags & task_tags:
                hits += 1
        hit_rate = hits / max(len(test_tasks), 1)
        results.append({
            'held_out': held_out,
            'train_size': len(train_tasks),
            'test_size': len(test_tasks),
            'hits': hits,
            'misses': len(test_tasks) - hits,
            'hit_rate': hit_rate,
        })
        print(f"  [LOTO] Hold out '{held_out}': hit_rate={hit_rate:.1%}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, required=True)
    parser.add_argument('--output', type=str, default='loto_report.json')
    args = parser.parse_args()
    with open(args.tasks) as f:
        tasks = json.load(f)
    results = run_loto(tasks)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nLOTO report saved to {args.output}")
