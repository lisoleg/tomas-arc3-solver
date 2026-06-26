"""
Topo-Map Sensitivity Analyzer: Standalone Version
"""

import numpy as np
import json
import argparse
from typing import Dict, Any, List, Callable

def extract_topo_features(grid: np.ndarray) -> Dict[str, Any]:
    h, w = grid.shape
    unique = len(np.unique(grid))
    try:
        from scipy.ndimage import label
        _, num = label(grid > 0)
    except:
        num = 1

    def detect_period(arr, axis):
        if axis == 0:
            for p in range(1, arr.shape[0] // 2 + 1):
                if np.array_equal(arr[:-p], arr[p:]):
                    return p
        else:
            for p in range(1, arr.shape[1] // 2 + 1):
                if np.array_equal(arr[:, :-p], arr[:, p:]):
                    return p
        return 0

    return {
        'euler_char': int(num),
        'unique_colors': unique,
        'period_x': detect_period(grid, axis=1),
        'period_y': detect_period(grid, axis=0),
        'sym_h': float(np.array_equal(grid, np.fliplr(grid))),
        'sym_v': float(np.array_equal(grid, np.flipud(grid))),
        'grid_size': (h, w),
    }

def run_sensitivity(grid: np.ndarray, extractor: Callable = extract_topo_features) -> Dict:
    base = extractor(grid)
    results = {'base': base, 'tests': {}}

    # Test 1: Color permutation
    non_zero = [c for c in np.unique(grid) if c != 0]
    rng = np.random.RandomState(42)
    shuffled = non_zero.copy()
    rng.shuffle(shuffled)
    mapping = {z: s for z, s in zip(non_zero, shuffled)}
    mapping[0] = 0
    perturbed = np.vectorize(mapping.get)(grid)
    pert = extractor(perturbed)
    color_diff = {}
    for k in base:
        if isinstance(base[k], (int, float)):
            color_diff[k] = abs(base[k] - pert.get(k, 0))
    results['tests']['color_permutation'] = {'passed': max(color_diff.values()) == 0, 'diffs': color_diff}

    # Test 2: Noise injection
    rng2 = np.random.RandomState(123)
    noisy = grid.copy()
    h, w = grid.shape
    for _ in range(max(1, h * w // 20)):
        y, x = rng2.randint(0, h), rng2.randint(0, w)
        noisy[y, x] = rng2.randint(1, 10)
    pert2 = extractor(noisy)
    noise_diff = {}
    for k in base:
        if isinstance(base[k], (int, float)):
            noise_diff[k] = abs(base[k] - pert2.get(k, 0))
    results['tests']['noise_injection'] = {'passed': max(noise_diff.values()) <= 1, 'diffs': noise_diff}

    # Test 3: Periodicity break
    ptest = {'note': 'No period to test'}
    if base.get('period_x', 0) > 0 or base.get('period_y', 0) > 0:
        perturbed3 = grid.copy()
        mid_y, mid_x = h // 2, w // 2
        perturbed3[mid_y, mid_x] = (perturbed3[mid_y, mid_x] + 1) % 10
        pert3 = extractor(perturbed3)
        degraded = (pert3.get('period_x', 0) < base['period_x']) or \
                   (pert3.get('period_y', 0) < base['period_y'])
        ptest = {
            'passed': degraded,
            'base_period': (base.get('period_x', 0), base.get('period_y', 0)),
            'perturbed_period': (pert3.get('period_x', 0), pert3.get('period_y', 0)),
        }
    results['tests']['periodicity'] = ptest
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grids', type=str, required=True)
    parser.add_argument('--output', type=str, default='topo_sensitivity_report.json')
    args = parser.parse_args()
    with open(args.grids) as f:
        grids_data = json.load(f)
    all_results = []
    for item in grids_data:
        grid = np.array(item['grid'], dtype=np.int32)
        result = run_sensitivity(grid)
        result['grid_id'] = item.get('id', 'unknown')
        all_results.append(result)
        n_pass = sum(1 for t in result['tests'].values() if t.get('passed', False))
        print(f"  [Topo] {result['grid_id']}: {n_pass}/{len(result['tests'])} tests passed")
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTopo sensitivity report saved to {args.output}")
