"""
GaussEx Pre-Check: Standalone Demo Consistency Verifier
"""

import numpy as np
import json
import argparse
import sys
from typing import List, Tuple, Dict, Any
from pathlib import Path

class GaussExPreChecker:
    def __init__(self):
        pass

    def execute_dsl(self, dsl_seq: List[str], grid: np.ndarray) -> np.ndarray:
        output = grid.copy().astype(np.int32)
        for cmd in dsl_seq:
            output = self._eval_primitive(cmd, output)
        return output

    def _eval_primitive(self, cmd: str, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        if cmd.startswith("repeat("):
            inner = cmd.split('(')[1].split(')')[0]
            a_type = inner.split(',')[0].strip().strip("'")
            count = int(inner.split(',')[1].strip())
            for _ in range(count):
                grid = self._apply_move(a_type, grid)
        elif cmd.startswith("action('"):
            a_type = cmd.split("'")[1]
            grid = self._apply_move(a_type, grid)
        elif cmd == "flip_horizontal()":
            grid = np.fliplr(grid)
        elif cmd == "flip_vertical()":
            grid = np.flipud(grid)
        elif cmd.startswith("fill_color("):
            parts = cmd.split('(')[1].split(')')[0].split(',')
            color = int(parts[0].strip())
            grid[:, :] = color
        return grid

    def _apply_move(self, direction: str, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        dx, dy = {'LEFT': (-1,0), 'RIGHT': (1,0), 'UP': (0,-1), 'DOWN': (0,1)}.get(direction, (0,0))
        nonzero = np.argwhere(grid > 0)
        if len(nonzero) == 0:
            return grid
        new_grid = grid.copy()
        ys, xs = nonzero[:, 0], nonzero[:, 1]
        new_ys = np.clip(ys + dy, 0, h - 1)
        new_xs = np.clip(xs + dx, 0, w - 1)
        new_grid[ys, xs] = 0
        new_grid[new_ys, new_xs] = grid[ys, xs]
        return new_grid

    def check(self, candidates: List[Dict], demos: List[Dict]) -> Dict:
        results = {}
        for cand in candidates:
            name = cand['name']
            dsl = cand['dsl']
            passed_all = True
            failures = []
            for idx, demo in enumerate(demos):
                inp = np.array(demo['input'], dtype=np.int32)
                expected = np.array(demo['output'], dtype=np.int32)
                actual = self.execute_dsl(dsl, inp)
                if not np.array_equal(actual, expected):
                    passed_all = False
                    failures.append({'demo_idx': idx, 'mismatches': int((actual != expected).sum())})
            results[name] = {'passed': passed_all, 'failures': failures}
            status = "PASS" if passed_all else f"FAIL ({len(failures)} demos)"
            print(f"  [GaussEx] {name}: {status}")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=str, required=True)
    parser.add_argument('--demos', type=str, required=True)
    parser.add_argument('--output', type=str, default='gaussex_result.json')
    args = parser.parse_args()
    with open(args.candidates) as f:
        candidates = json.load(f)
    with open(args.demos) as f:
        demos = json.load(f)
    checker = GaussExPreChecker()
    results = checker.check(candidates, demos)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nGaussEx results saved to {args.output}")
