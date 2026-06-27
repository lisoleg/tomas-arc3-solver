# -*- coding: utf-8 -*-
"""Semantic Constants Extractor — Extract semantic constants from EML perception.

Extracts 3 categories of semantic constants from JinlingSphere lists and
ARC grids, serving as input for the Ramanujan Conjecture Producer:

  1. Symmetry constants: symmetry orders {1,2,4,8} and mean
  2. Combinatorial sequences: arithmetic, geometric, Fibonacci patterns
  3. Information-theoretic constants: κ_UV, κ_IR, compression_ratio, H_topo

These constants are analogous to the mathematical constants that Ramanujan
discovered relationships for (π, e, golden ratio, etc.) — here, the
"constants" emerge from the structure of ARC grids themselves.

Version: v4.1
TOMAS Correspondence: TOMAS Phase III → SemanticConstants → RamanujanConjectureProducer
IDO Correspondence: SemanticConstants = κ-extracted deltas from EML perception
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agent.eml_perceiver import JinlingSphere


# =============================================================================
# §1. SemanticConstants Dataclass — 3-Category Semantic Constant Vector
# =============================================================================

@dataclass
class SemanticConstants:
    """Semantic constants extracted from EML perception results.

    Three categories of constants, analogous to mathematical constants
    that Ramanujan discovered integer relations for:

    Category 1 — Symmetry Constants:
      - symmetry_orders: per-object symmetry orders ∈ {1,2,4,8}
      - symmetry_mean: average symmetry order (structural regularity metric)

    Category 2 — Combinatorial Sequences:
      - combinatorial_sequences: detected arithmetic/geometric/Fibonacci
        patterns in color values or position indices

    Category 3 — Information-Theoretic Constants:
      - topological_entropy: H_topo = -Σ(p_i × log(p_i)), color frequency entropy
      - info_cardinality_uv: κ_UV = log2(unique_colors × unique_shapes × grid_area)
      - info_cardinality_ir: κ_IR = log2(dominant_structure_count)
      - compression_ratio: κ_UV / κ_IR (akin to fine-structure constant α ≈ 1/137)

    Additional:
      - dominant_colors: most frequent color values
      - grid_shape: (H, W) dimensions of the input grid

    Attributes:
        symmetry_orders: Per-object symmetry order list.
        symmetry_mean: Mean symmetry order across all objects.
        combinatorial_sequences: Detected integer sequence patterns.
        topological_entropy: Color frequency Shannon entropy.
        info_cardinality_uv: UV-state information cardinality.
        info_cardinality_ir: IR-state information cardinality.
        compression_ratio: κ_UV / κ_IR ratio.
        dominant_colors: List of dominant (most frequent) color values.
        grid_shape: Grid dimensions (H, W).
    """

    symmetry_orders: List[int] = field(default_factory=list)
    symmetry_mean: float = 0.0
    combinatorial_sequences: List[List[int]] = field(default_factory=list)
    topological_entropy: float = 0.0
    info_cardinality_uv: float = 0.0
    info_cardinality_ir: float = 0.0
    compression_ratio: float = 0.0
    dominant_colors: List[int] = field(default_factory=list)
    grid_shape: Tuple[int, int] = (0, 0)


# =============================================================================
# §2. Combinatorial Sequence Detection — Arithmetic/Geometric/Fibonacci
# =============================================================================

def _detect_arithmetic_sequence(values: List[int]) -> Optional[List[int]]:
    """Detect if a sequence of integers forms an arithmetic progression.

    Arithmetic progression: a, a+d, a+2d, ... (constant difference d).
    Only detects sequences of length ≥ 3.

    Args:
        values: List of integer values to test.

    Returns:
        The original sequence if it forms an arithmetic progression
        with length ≥ 3, None otherwise.
    """
    if len(values) < 3:
        return None

    # Check constant difference
    diffs: List[int] = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if all(d == diffs[0] for d in diffs) and diffs[0] != 0:
        return values

    return None


def _detect_geometric_sequence(values: List[int]) -> Optional[List[int]]:
    """Detect if a sequence of integers forms a geometric progression.

    Geometric progression: a, a×r, a×r², ... (constant ratio r).
    Only detects sequences of length ≥ 3 with integer ratio.

    Args:
        values: List of integer values to test.

    Returns:
        The original sequence if it forms a geometric progression
        with length ≥ 3, None otherwise.
    """
    if len(values) < 3:
        return None

    # Check constant ratio (integer ratios only for ARC context)
    # Avoid division by zero or zero values
    if any(v == 0 for v in values):
        return None

    # Check if ratio is consistent (allow small rounding tolerance for integers)
    ratios: List[float] = [values[i + 1] / values[i] for i in range(len(values) - 1)]
    if all(abs(r - ratios[0]) < 0.01 for r in ratios) and ratios[0] != 1.0:
        # Verify it's approximately an integer ratio
        if abs(ratios[0] - round(ratios[0])) < 0.01:
            return values

    return None


def _detect_fibonacci_sequence(values: List[int]) -> Optional[List[int]]:
    """Detect if a sequence of integers follows Fibonacci recurrence.

    Fibonacci recurrence: f(n+2) = f(n+1) + f(n).
    Only detects sequences of length ≥ 3.

    Args:
        values: List of integer values to test.

    Returns:
        The original sequence if it follows Fibonacci recurrence
        with length ≥ 3, None otherwise.
    """
    if len(values) < 3:
        return None

    # Check Fibonacci recurrence relation
    for i in range(len(values) - 2):
        if values[i + 2] != values[i + 1] + values[i]:
            return None

    return values


def _detect_sequences_in_values(values: List[int]) -> List[List[int]]:
    """Detect all combinatorial sequences in a list of integer values.

    Tests the full sequence and all contiguous subsequences of length ≥ 3
    for arithmetic, geometric, and Fibonacci patterns.

    Args:
        values: Sorted list of integer values (colors, positions, etc.).

    Returns:
        List of detected sequences (each is a list of integers).
    """
    detected: List[List[int]] = []
    n: int = len(values)

    if n < 3:
        return detected

    # Test the full sequence first
    sorted_vals: List[int] = sorted(values)

    # Check arithmetic
    arith = _detect_arithmetic_sequence(sorted_vals)
    if arith is not None:
        detected.append(arith)

    # Check geometric
    geom = _detect_geometric_sequence(sorted_vals)
    if geom is not None:
        detected.append(geom)

    # Check Fibonacci
    fib = _detect_fibonacci_sequence(sorted_vals)
    if fib is not None:
        detected.append(fib)

    # Check contiguous subsequences of length 3..n-1
    for length in range(3, n):
        for start in range(n - length + 1):
            subseq: List[int] = sorted_vals[start:start + length]

            arith_sub = _detect_arithmetic_sequence(subseq)
            if arith_sub is not None and arith_sub not in detected:
                detected.append(arith_sub)

            geom_sub = _detect_geometric_sequence(subseq)
            if geom_sub is not None and geom_sub not in detected:
                detected.append(geom_sub)

            fib_sub = _detect_fibonacci_sequence(subseq)
            if fib_sub is not None and fib_sub not in detected:
                detected.append(fib_sub)

    return detected


# =============================================================================
# §3. extract_semantic_constants — Main Extraction Pipeline
# =============================================================================

def extract_semantic_constants(
    spheres: List[JinlingSphere],
    grid: np.ndarray,
) -> SemanticConstants:
    """Extract semantic constants from EML perception results.

    Pipeline:
      1. Collect symmetry orders from all JinlingSphere objects → compute mean
      2. Extract color values → detect combinatorial sequences
      3. Compute topological entropy from color frequency distribution
      4. Compute κ_UV (UV-state information cardinality)
      5. Compute κ_IR (IR-state information cardinality, dominant structures)
      6. Compute compression ratio κ_UV / κ_IR

    Args:
        spheres: List of JinlingSphere objects from EMLPerceiver.perceive().
        grid: Original ARC grid (2D integer array, H×W).

    Returns:
        SemanticConstants dataclass with all fields populated.
    """
    grid = np.asarray(grid, dtype=np.int32)
    H, W = grid.shape

    # ------------------------------------------------------------------
    # Step 1: Symmetry constants — per-object symmetry orders
    # ------------------------------------------------------------------
    symmetry_orders: List[int] = []
    for s in spheres:
        symmetry_orders.append(s.oct_phase.symmetry_order)

    symmetry_mean: float = 0.0
    if symmetry_orders:
        symmetry_mean = float(sum(symmetry_orders)) / len(symmetry_orders)

    # ------------------------------------------------------------------
    # Step 2: Combinatorial sequence detection — colors and positions
    # ------------------------------------------------------------------
    # Collect color values from spheres
    color_values: List[int] = [s.color for s in spheres]

    # Also extract grid-wide color values for broader sequence detection
    all_colors: List[int] = sorted(set(int(c) for c in grid.flatten() if c != 0))

    combinatorial_sequences: List[List[int]] = []

    # Detect sequences in sphere colors
    if len(color_values) >= 3:
        seqs = _detect_sequences_in_values(color_values)
        combinatorial_sequences.extend(seqs)

    # Detect sequences in grid-wide colors (if different from sphere colors)
    if len(all_colors) >= 3:
        seqs = _detect_sequences_in_values(all_colors)
        for seq in seqs:
            if seq not in combinatorial_sequences:
                combinatorial_sequences.extend(seqs)

    # ------------------------------------------------------------------
    # Step 3: Topological entropy — color frequency Shannon entropy
    # ------------------------------------------------------------------
    # Color frequency distribution (excluding background color 0)
    non_zero_pixels: np.ndarray = grid[grid != 0]
    total_non_zero: int = len(non_zero_pixels)

    topological_entropy: float = 0.0
    if total_non_zero > 0:
        unique_colors_grid: np.ndarray = np.unique(non_zero_pixels)
        for color in unique_colors_grid:
            count: int = int(np.sum(non_zero_pixels == color))
            p_i: float = count / total_non_zero
            if p_i > 0:
                topological_entropy -= p_i * math.log(p_i)

    # ------------------------------------------------------------------
    # Step 4: κ_UV — UV-state information cardinality
    # ------------------------------------------------------------------
    # κ_UV = log2(unique_colors × unique_shapes × grid_area)
    unique_colors_count: int = len(set(int(c) for c in grid.flatten() if c != 0))
    unique_shapes_count: int = len(set(s.oct_phase.symmetry_order for s in spheres))
    grid_area: int = H * W

    info_cardinality_uv: float = math.log2(
        max(unique_colors_count, 1) * max(unique_shapes_count, 1) * max(grid_area, 1)
    )

    # ------------------------------------------------------------------
    # Step 5: κ_IR — IR-state information cardinality (dominant structures)
    # ------------------------------------------------------------------
    # Dominant structures: spheres with coupling > 1/6 (Dead-Zero threshold)
    # In ARC context, dominant = spheres that survived pruning
    dominant_count: int = len(spheres)  # Already pruned by Dead-Zero

    # Also count "strongly dominant" spheres (coupling > 1/3)
    strongly_dominant: int = sum(1 for s in spheres if s.coupling > 1.0 / 3.0)

    info_cardinality_ir: float = math.log2(max(strongly_dominant, 1))

    # ------------------------------------------------------------------
    # Step 6: Compression ratio — κ_UV / κ_IR
    # ------------------------------------------------------------------
    compression_ratio: float = info_cardinality_uv / max(info_cardinality_ir, 1e-6)

    # ------------------------------------------------------------------
    # Additional: Dominant colors and grid shape
    # ------------------------------------------------------------------
    # Dominant colors: most frequent non-zero colors (top 5)
    color_counts: Dict[int, int] = {}
    for c in non_zero_pixels:
        color_val: int = int(c)
        color_counts[color_val] = color_counts.get(color_val, 0) + 1

    # Sort by frequency (descending), take top 5
    sorted_colors: List[Tuple[int, int]] = sorted(
        color_counts.items(), key=lambda x: x[1], reverse=True
    )
    dominant_colors: List[int] = [c for c, _ in sorted_colors[:5]]

    grid_shape: Tuple[int, int] = (H, W)

    return SemanticConstants(
        symmetry_orders=symmetry_orders,
        symmetry_mean=symmetry_mean,
        combinatorial_sequences=combinatorial_sequences,
        topological_entropy=topological_entropy,
        info_cardinality_uv=info_cardinality_uv,
        info_cardinality_ir=info_cardinality_ir,
        compression_ratio=compression_ratio,
        dominant_colors=dominant_colors,
        grid_shape=grid_shape,
    )


# =============================================================================
# §4. Self-Test
# =============================================================================

def _self_test() -> None:
    """Run self-tests for SemanticConstants module."""
    print("SemanticConstants — Self-Test")
    print("=" * 50)

    from src.agent.eml_perceiver import EMLPerceiver

    # Test 1: Sequence detection — arithmetic
    arith_seq = _detect_arithmetic_sequence([2, 4, 6, 8])
    print(f"\n1. Arithmetic sequence: {arith_seq}")
    assert arith_seq is not None, "Should detect arithmetic sequence"
    assert arith_seq == [2, 4, 6, 8], f"Expected [2,4,6,8], got {arith_seq}"
    print("   PASSED")

    # Test 2: Sequence detection — geometric
    geom_seq = _detect_geometric_sequence([2, 4, 8, 16])
    print(f"\n2. Geometric sequence: {geom_seq}")
    assert geom_seq is not None, "Should detect geometric sequence"
    print("   PASSED")

    # Test 3: Sequence detection — Fibonacci
    fib_seq = _detect_fibonacci_sequence([1, 1, 2, 3, 5])
    print(f"\n3. Fibonacci sequence: {fib_seq}")
    assert fib_seq is not None, "Should detect Fibonacci sequence"
    assert fib_seq == [1, 1, 2, 3, 5], f"Expected [1,1,2,3,5], got {fib_seq}"
    print("   PASSED")

    # Test 4: Sequence detection — non-sequence
    random_vals = _detect_arithmetic_sequence([3, 7, 11, 20])
    print(f"\n4. Non-arithmetic: {random_vals}")
    # 3,7,11 is arithmetic (d=4), but 7,11,20 is not. Full sequence fails.
    assert random_vals is None, "Should not detect arithmetic in [3,7,11,20]"
    print("   PASSED")

    # Test 5: SemanticConstants extraction from a grid
    grid5 = np.zeros((8, 8), dtype=np.int32)
    grid5[1:3, 1:3] = 1  # 2×2 square (color 1)
    grid5[4:6, 4:8] = 2  # 2×4 rectangle (color 2)

    perceiver = EMLPerceiver()
    spheres5 = perceiver.perceive(grid5)
    constants5 = extract_semantic_constants(spheres5, grid5)

    print(f"\n5. Grid extraction:")
    print(f"   symmetry_orders={constants5.symmetry_orders}")
    print(f"   symmetry_mean={constants5.symmetry_mean:.4f}")
    print(f"   H_topo={constants5.topological_entropy:.4f}")
    print(f"   κ_UV={constants5.info_cardinality_uv:.4f}")
    print(f"   κ_IR={constants5.info_cardinality_ir:.4f}")
    print(f"   compression_ratio={constants5.compression_ratio:.4f}")
    print(f"   dominant_colors={constants5.dominant_colors}")
    print(f"   grid_shape={constants5.grid_shape}")
    assert constants5.grid_shape == (8, 8)
    assert len(constants5.symmetry_orders) == len(spheres5)
    assert constants5.topological_entropy >= 0
    assert constants5.info_cardinality_uv > 0
    print("   PASSED")

    # Test 6: Empty grid
    grid6 = np.zeros((5, 5), dtype=np.int32)
    spheres6 = perceiver.perceive(grid6)
    constants6 = extract_semantic_constants(spheres6, grid6)
    print(f"\n6. Empty grid: κ_UV={constants6.info_cardinality_uv:.4f}, "
          f"κ_IR={constants6.info_cardinality_ir:.4f}")
    assert constants6.grid_shape == (5, 5)
    assert len(constants6.symmetry_orders) == 0
    print("   PASSED")

    # Test 7: Subsequence detection
    subseqs = _detect_sequences_in_values([2, 4, 6, 3, 9, 27])
    print(f"\n7. Subsequence detection in [2,4,6,3,9,27]:")
    for seq in subseqs:
        print(f"   {seq}")
    # Should find: [2,4,6] (arithmetic d=2), [3,9,27] (geometric r=3)
    assert len(subseqs) >= 2, f"Expected at least 2 sequences, got {len(subseqs)}"
    print("   PASSED")

    print("\n" + "=" * 50)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()
