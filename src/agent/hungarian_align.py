# -*- coding: utf-8 -*-
"""Hungarian alignment — Greedy approximate bipartite matching for JinlingSphere.

Implements greedy approximate Hungarian matching for object alignment
in GaussEx verification. Uses default cost function combining OctonionPhase
distance, color penalty, and position distance.

No scipy.optimize.linear_sum_assignment dependency — pure numpy greedy
approximation that runs in O(n² log n) for n = min(|source|, |target|).

The greedy algorithm:
  1. Compute full cost matrix between source and target spheres
  2. Sort all (i, j) pairs by ascending cost
  3. Greedily assign the cheapest available pair at each step
  4. Each source/target can only be assigned once

Fast rejection: if minimum cost > 2ε×n, return empty alignment
(the objects are too dissimilar for meaningful matching).

Default cost: sqrt(phase_dist² + color_penalty² + pos_dist²)
  phase_dist   = |Φ₁ − Φ₂|₂  (OctonionPhase vector L2 distance)
  color_penalty = 0 if same color, 1 if different
  pos_dist      = |centroid₁ − centroid₂|₂ / max_dim (normalized position)

Version: v4.1
TOMAS Correspondence: TOMAS Phase III → GaussEx Fast → Hungarian Alignment
IDO Correspondence: Hungarian alignment = object-correspondence delta
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from src.agent.eml_perceiver import JinlingSphere


# =============================================================================
# §1. Default Cost Function
# =============================================================================

def _default_cost(src: JinlingSphere, tgt: JinlingSphere) -> float:
    """Compute default alignment cost between two JinlingSphere objects.

    Cost formula: sqrt(phase_dist² + color_penalty² + pos_dist²)

    Components:
      phase_dist   = |Φ_src − Φ_tgt|₂  (L2 distance of phase vectors)
      color_penalty = 0.0 if same color, 1.0 if different
      pos_dist      = |centroid_src − centroid_tgt|₂ / max_dim
                     (normalized by the larger bbox dimension)

    Args:
        src: Source JinlingSphere.
        tgt: Target JinlingSphere.

    Returns:
        Non-negative alignment cost. Zero means perfect match.
    """
    # OctonionPhase vector L2 distance
    phase_dist: float = float(np.linalg.norm(
        src.oct_phase.phase_vector - tgt.oct_phase.phase_vector
    ))

    # Color penalty: 0 if same, 1 if different
    color_penalty: float = 0.0 if src.color == tgt.color else 1.0

    # Normalized position distance
    src_bbox_dims: Tuple[int, int] = (
        src.bbox[2] - src.bbox[0] + 1,
        src.bbox[3] - src.bbox[1] + 1,
    )
    tgt_bbox_dims: Tuple[int, int] = (
        tgt.bbox[2] - tgt.bbox[0] + 1,
        tgt.bbox[3] - tgt.bbox[1] + 1,
    )
    max_dim: float = max(
        float(src_bbox_dims[0]), float(src_bbox_dims[1]),
        float(tgt_bbox_dims[0]), float(tgt_bbox_dims[1]),
        1.0,  # Floor to prevent division by zero
    )
    centroid_diff: np.ndarray = np.array(src.centroid) - np.array(tgt.centroid)
    pos_dist: float = float(np.linalg.norm(centroid_diff)) / max_dim

    return float(np.sqrt(phase_dist ** 2 + color_penalty ** 2 + pos_dist ** 2))


# =============================================================================
# §2. Hungarian Alignment — Greedy Approximate Bipartite Matching
# =============================================================================

def hungarian_align(
    source_spheres: List[JinlingSphere],
    target_spheres: List[JinlingSphere],
    cost_fn: Optional[Callable[[JinlingSphere, JinlingSphere], float]] = None,
    epsilon: float = 0.1,
) -> Tuple[List[Tuple[int, int]], float]:
    """Greedy approximate bipartite matching for JinlingSphere alignment.

    Computes a cost matrix between all source-target pairs, then greedily
    assigns pairs in order of ascending cost. Each source and target can
    only appear in one pair.

    Fast rejection: if the minimum cost in the matrix exceeds 2ε×n
    (where n = max(|source|, |target|)), the alignment is deemed
    impossible and an empty result is returned.

    This is NOT scipy.optimize.linear_sum_assignment — it's a greedy
    approximation that runs in O(n² log n) and produces near-optimal
    results for typical ARC grid sizes (n ≤ 20 objects).

    Args:
        source_spheres: List of JinlingSphere objects from source grid.
        target_spheres: List of JinlingSphere objects from target grid.
        cost_fn: Optional custom cost function. If None, uses _default_cost
            which combines phase_dist, color_penalty, and pos_dist.
        epsilon: Error tolerance for fast rejection. Default 0.1.

    Returns:
        Tuple of (alignment, total_cost):
          alignment: List of (source_idx, target_idx) pairs.
          total_cost: Sum of costs for all aligned pairs.
          Empty alignment if fast rejection triggers or no spheres.
    """
    n_src: int = len(source_spheres)
    n_tgt: int = len(target_spheres)

    if n_src == 0 or n_tgt == 0:
        return [], 0.0

    # Use default cost function if none provided
    if cost_fn is None:
        cost_fn = _default_cost

    # ------------------------------------------------------------------
    # Compute cost matrix
    # ------------------------------------------------------------------
    cost_matrix: np.ndarray = np.full((n_src, n_tgt), np.inf, dtype=np.float64)
    for i in range(n_src):
        for j in range(n_tgt):
            cost_matrix[i, j] = cost_fn(source_spheres[i], target_spheres[j])

    # ------------------------------------------------------------------
    # Fast rejection: if min cost > 2ε×n, alignment is impossible
    # ------------------------------------------------------------------
    n_max: int = max(n_src, n_tgt)
    min_cost: float = float(np.min(cost_matrix))
    if min_cost > 2.0 * epsilon * n_max:
        return [], float(np.min(cost_matrix))

    # ------------------------------------------------------------------
    # Greedy matching: sort all pairs by cost, assign cheapest first
    # ------------------------------------------------------------------
    alignment: List[Tuple[int, int]] = []
    total_cost: float = 0.0
    used_src: set = set()
    used_tgt: set = set()

    # Flatten and sort cost matrix by ascending cost
    flat_costs: np.ndarray = cost_matrix.flatten()
    sorted_indices: np.ndarray = np.argsort(flat_costs)

    for flat_idx in sorted_indices:
        i: int = int(flat_idx // n_tgt)
        j: int = int(flat_idx % n_tgt)

        if i in used_src or j in used_tgt:
            continue
        if cost_matrix[i, j] == np.inf:
            break  # No more valid pairs

        alignment.append((i, j))
        total_cost += float(cost_matrix[i, j])
        used_src.add(i)
        used_tgt.add(j)

        # Stop when one side is fully assigned
        if len(used_src) == n_src or len(used_tgt) == n_tgt:
            break

    return alignment, total_cost


# =============================================================================
# §3. Self-Test
# =============================================================================

def _self_test() -> None:
    """Run self-tests for hungarian_align module."""
    print("hungarian_align — Self-Test")
    print("=" * 50)

    from src.agent.octonion_phase import OctonionPhase

    # Test 1: Perfect match — identical spheres
    src1 = [
        JinlingSphere(id=0, centroid=(2.0, 2.0), bbox=(1, 1, 3, 3),
                       color=1, oct_phase=OctonionPhase(
                           phase_vector=np.array([1.0, 0.0, 0.0, 1.0, 0.5])),
                       coupling=2.0),
    ]
    tgt1 = [
        JinlingSphere(id=1, centroid=(2.0, 2.0), bbox=(1, 1, 3, 3),
                       color=1, oct_phase=OctonionPhase(
                           phase_vector=np.array([1.0, 0.0, 0.0, 1.0, 0.5])),
                       coupling=2.0),
    ]
    align1, cost1 = hungarian_align(src1, tgt1)
    print(f"\n1. Perfect match: alignment={align1}, cost={cost1:.4f}")
    assert align1 == [(0, 0)], f"Expected [(0,0)], got {align1}"
    assert cost1 < 0.01, f"Expected near-zero cost, got {cost1}"
    print("   PASSED")

    # Test 2: Two spheres with different colors
    src2 = [
        JinlingSphere(id=0, centroid=(1.0, 1.0), bbox=(0, 0, 2, 2),
                       color=1, oct_phase=OctonionPhase(
                           phase_vector=np.array([0.5, 0.5, 0.0, 1.0, 0.25])),
                       coupling=2.0),
        JinlingSphere(id=1, centroid=(5.0, 5.0), bbox=(4, 4, 6, 6),
                       color=2, oct_phase=OctonionPhase(
                           phase_vector=np.array([0.5, 0.5, 0.0, 1.0, 0.25])),
                       coupling=2.0),
    ]
    tgt2 = [
        JinlingSphere(id=2, centroid=(1.5, 1.5), bbox=(0, 0, 3, 3),
                       color=1, oct_phase=OctonionPhase(
                           phase_vector=np.array([0.5, 0.5, 0.0, 1.0, 0.25])),
                       coupling=2.0),
        JinlingSphere(id=3, centroid=(5.5, 5.5), bbox=(4, 4, 7, 7),
                       color=2, oct_phase=OctonionPhase(
                           phase_vector=np.array([0.5, 0.5, 0.0, 1.0, 0.25])),
                       coupling=2.0),
    ]
    align2, cost2 = hungarian_align(src2, tgt2)
    print(f"\n2. Two spheres: alignment={align2}, cost={cost2:.4f}")
    assert len(align2) == 2, f"Expected 2 pairs, got {len(align2)}"
    # Color-match pairs should have lower cost than cross-color
    print("   PASSED")

    # Test 3: Cross-color penalty — matching red to blue should be costlier
    src3 = [
        JinlingSphere(id=0, centroid=(2.0, 2.0), bbox=(1, 1, 3, 3),
                       color=1, oct_phase=OctonionPhase(
                           phase_vector=np.zeros(5)),
                       coupling=1.0),
    ]
    tgt3_red = [
        JinlingSphere(id=1, centroid=(2.0, 2.0), bbox=(1, 1, 3, 3),
                       color=1, oct_phase=OctonionPhase(
                           phase_vector=np.zeros(5)),
                       coupling=1.0),
    ]
    tgt3_blue = [
        JinlingSphere(id=2, centroid=(2.0, 2.0), bbox=(1, 1, 3, 3),
                       color=5, oct_phase=OctonionPhase(
                           phase_vector=np.zeros(5)),
                       coupling=1.0),
    ]
    _, cost_same = hungarian_align(src3, tgt3_red)
    _, cost_diff = hungarian_align(src3, tgt3_blue)
    print(f"\n3. Color penalty: same-color={cost_same:.4f}, "
          f"diff-color={cost_diff:.4f}")
    assert cost_diff > cost_same, "Different color should have higher cost"
    print("   PASSED")

    # Test 4: Empty inputs
    align4, cost4 = hungarian_align([], [])
    print(f"\n4. Empty inputs: alignment={align4}, cost={cost4}")
    assert align4 == [] and cost4 == 0.0
    print("   PASSED")

    # Test 5: Asymmetric sizes (3 source, 2 target)
    src5 = [
        JinlingSphere(id=i, centroid=(float(i), float(i)),
                       bbox=(i, i, i+2, i+2), color=i+1,
                       oct_phase=OctonionPhase(
                           phase_vector=np.array([float(i)*0.1]*5)),
                       coupling=1.0)
        for i in range(3)
    ]
    tgt5 = [
        JinlingSphere(id=i+10, centroid=(float(i)+0.5, float(i)+0.5),
                       bbox=(i, i, i+2, i+2), color=i+1,
                       oct_phase=OctonionPhase(
                           phase_vector=np.array([float(i)*0.1]*5)),
                       coupling=1.0)
        for i in range(2)
    ]
    align5, cost5 = hungarian_align(src5, tgt5)
    print(f"\n5. Asymmetric (3→2): alignment={align5}, cost={cost5:.4f}")
    assert len(align5) == 2, f"Expected 2 pairs (min side), got {len(align5)}"
    print("   PASSED")

    # Test 6: Custom cost function
    def unit_cost(s: JinlingSphere, t: JinlingSphere) -> float:
        """Simple unit cost: 0 if same color, 1 otherwise."""
        return 0.0 if s.color == t.color else 1.0

    src6 = [
        JinlingSphere(id=0, centroid=(0, 0), bbox=(0,0,1,1),
                       color=1, oct_phase=OctonionPhase(), coupling=1.0),
        JinlingSphere(id=1, centroid=(0, 0), bbox=(0,0,1,1),
                       color=2, oct_phase=OctonionPhase(), coupling=1.0),
    ]
    tgt6 = [
        JinlingSphere(id=2, centroid=(0, 0), bbox=(0,0,1,1),
                       color=1, oct_phase=OctonionPhase(), coupling=1.0),
        JinlingSphere(id=3, centroid=(0, 0), bbox=(0,0,1,1),
                       color=2, oct_phase=OctonionPhase(), coupling=1.0),
    ]
    align6, cost6 = hungarian_align(src6, tgt6, cost_fn=unit_cost)
    print(f"\n6. Custom cost fn: alignment={align6}, cost={cost6:.4f}")
    assert cost6 == 0.0, f"Expected 0 cost (same colors match), got {cost6}"
    print("   PASSED")

    # Test 7: Default cost formula verification
    s7 = JinlingSphere(id=0, centroid=(0.0, 0.0), bbox=(0, 0, 3, 3),
                        color=1, oct_phase=OctonionPhase(
                            phase_vector=np.array([1.0, 0.0, 0.0, 1.0, 0.125])),
                        coupling=2.0)
    t7 = JinlingSphere(id=1, centroid=(1.0, 1.0), bbox=(0, 0, 4, 4),
                        color=2, oct_phase=OctonionPhase(
                            phase_vector=np.array([0.0, 1.0, 0.5, 0.5, 0.25])),
                        coupling=1.5)
    c7 = _default_cost(s7, t7)
    # Manual: phase_dist = |[1,0,0,1,0.125]-[0,1,0.5,0.5,0.25]| = sqrt(1+1+0.25+0.25+0.015625)
    phase_diff = np.array([1.0, 0.0, 0.0, 1.0, 0.125]) - np.array([0.0, 1.0, 0.5, 0.5, 0.25])
    expected_phase = float(np.linalg.norm(phase_diff))
    expected_color = 1.0  # different color
    # max_dim = max(src_dims, tgt_dims) = max(4,4,5,5,1) = 5.0
    # tgt bbox (0,0,4,4) → dims = (4-0+1, 4-0+1) = (5, 5)
    expected_pos = float(np.linalg.norm([0.0-1.0, 0.0-1.0])) / 5.0  # max_dim=5
    expected_cost = float(np.sqrt(expected_phase**2 + expected_color**2 + expected_pos**2))
    print(f"\n7. Default cost: computed={c7:.6f}, expected={expected_cost:.6f}")
    assert abs(c7 - expected_cost) < 1e-6, f"Cost mismatch: {c7} vs {expected_cost}"
    print("   PASSED")

    print("\n" + "=" * 50)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()
