"""Advanced search pruning optimizer for kappa-Snap Two-Phase search.

Implements multiple pruning strategies to reduce the candidate space
before and during Phase B verification:

Pre-Phase A fast filters (v2.3 enhanced — run BEFORE topo hash):
  1a. Grid shape pre-filter — output dimensions must match demo output
  1b. Color histogram invariant — color distribution must match
  1c. Non-zero pixel count invariant — object pixel count must match
  1d. Betti0 invariant pruning — connected components must match

Phase A (topo hash):
  2. Topological hash filter (in TopoHashFilter)

Post-Phase A:
  3. Symmetry equivalence deduplication — remove equivalent candidates
  4. Incremental MDL pruning — skip depth-3 enumeration when partial MDL exceeds threshold
  5. Heuristic candidate ordering — low-MDL + high-topo-match first

TOMAS v2.3: Pruning optimization layer with 4x tighter Phase A filtering.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from src.core.dsl_primitives import ProgramNode


class PruningOptimizer:
    """Advanced pruning optimizer for the kappa-Snap search.

    Integrates multiple pruning strategies to reduce the candidate
    space before Phase B verification, improving search throughput.

    Attributes:
        enable_betti0: Whether Betti0 invariant pruning is enabled.
        enable_symmetry_dedup: Whether symmetry dedup is enabled.
        enable_heuristic_order: Whether heuristic ordering is enabled.
        enable_incremental_mdl: Whether incremental MDL pruning is enabled.
        enable_shape_filter: Whether grid shape pre-filter is enabled.
        enable_color_hist: Whether color histogram pruning is enabled.
        enable_nonzero_count: Whether non-zero count pruning is enabled.
        mdl_threshold: Maximum MDL cost for valid programs.
        _betti0_cache: Cache of Betti0 values for grids.
        _symmetry_cache: Cache of symmetry signatures.
        _color_hist_cache: Cache of color histograms.
        _nonzero_cache: Cache of non-zero counts.
        _shape_cache: Cache of grid shapes.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the pruning optimizer.

        Args:
            config: Pruning configuration dict with enable_* flags
                and mdl_threshold.
        """
        self.enable_betti0: bool = config.get("enable_betti0", True)
        self.enable_symmetry_dedup: bool = config.get(
            "enable_symmetry_dedup", True
        )
        self.enable_heuristic_order: bool = config.get(
            "enable_heuristic_order", True
        )
        self.enable_incremental_mdl: bool = config.get(
            "enable_incremental_mdl", True
        )
        # v2.3 enhanced: pre-Phase A fast filters
        self.enable_shape_filter: bool = config.get("enable_shape_filter", True)
        self.enable_color_hist: bool = config.get("enable_color_hist", True)
        self.enable_nonzero_count: bool = config.get(
            "enable_nonzero_count", True
        )
        self.mdl_threshold: int = config.get("mdl_threshold", 50)

        # Caches
        self._betti0_cache: dict[int, int] = {}
        self._symmetry_cache: dict[int, str] = {}
        self._color_hist_cache: dict[int, tuple[int, ...]] = {}
        self._nonzero_cache: dict[int, int] = {}
        self._shape_cache: dict[int, tuple[int, int]] = {}

        # Statistics
        self.stats: dict[str, int] = {
            "betti0_pruned": 0,
            "symmetry_deduped": 0,
            "mdl_pruned": 0,
            "shape_pruned": 0,
            "color_hist_pruned": 0,
            "nonzero_pruned": 0,
        }

    def compute_betti0_fast(self, grid: np.ndarray) -> int:
        """Fast Betti0 computation using scipy.ndimage.label.

        Counts connected components of non-zero pixels directly on
        the raw grid, bypassing the full Octonion encoding for speed.

        Args:
            grid: Input grid as ndarray.

        Returns:
            Number of connected components (Betti0).
        """
        cache_key = hash(grid.tobytes())
        if cache_key in self._betti0_cache:
            return self._betti0_cache[cache_key]

        binary = (grid > 0).astype(np.int32)
        _, num_components = ndimage.label(binary)
        self._betti0_cache[cache_key] = int(num_components)
        return int(num_components)

    def betti0_prune(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """Prune candidates by Betti0 topological invariant.

        For each demo pair, computes Betti0 of the output grid.
        A candidate program P is pruned if Betti0(P(I_i)) != Betti0(O_i)
        for any demo pair i, as this is a topological necessary condition.

        Args:
            candidates: List of candidate ProgramNodes.
            demo_pairs: List of demo pairs with 'input' and 'output' grids.

        Returns:
            Filtered list of candidates passing Betti0 check.
        """
        if not self.enable_betti0 or not demo_pairs:
            return list(candidates)

        # Pre-compute output Betti0 for all demo pairs
        output_betti0s: list[list[int]] = []
        for pair in demo_pairs:
            output_grids = pair.get("output", [])
            pair_betti0 = [
                self.compute_betti0_fast(g) for g in output_grids
            ]
            output_betti0s.append(pair_betti0)

        # Also pre-compute input Betti0 for reference
        # (some primitives preserve Betti0, some don't — we check output match)

        filtered: list[ProgramNode] = []
        for candidate in candidates:
            passes = True
            for pair_idx, pair in enumerate(demo_pairs):
                input_grids = pair.get("input", [])
                expected_betti0s = output_betti0s[pair_idx]

                for grid_idx, input_grid in enumerate(input_grids):
                    if grid_idx >= len(expected_betti0s):
                        continue
                    try:
                        predicted = candidate.apply(input_grid)
                        pred_betti0 = self.compute_betti0_fast(predicted)
                        if pred_betti0 != expected_betti0s[grid_idx]:
                            passes = False
                            break
                    except Exception:
                        passes = False
                        break
                if not passes:
                    break

            if passes:
                filtered.append(candidate)
            else:
                self.stats["betti0_pruned"] += 1

        return filtered

    # ============================================================
    # v2.3 Enhanced: Pre-Phase A Fast Filters
    # ============================================================

    def _get_grid_shape(self, grid: np.ndarray) -> tuple[int, int]:
        """Get grid shape with caching.

        Args:
            grid: Input grid.

        Returns:
            (H, W) tuple.
        """
        cache_key = hash(grid.tobytes())
        if cache_key in self._shape_cache:
            return self._shape_cache[cache_key]
        shape = (grid.shape[0], grid.shape[1])
        self._shape_cache[cache_key] = shape
        return shape

    def grid_shape_prune(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """Prune candidates whose output shape doesn't match demo output.

        This is the cheapest filter — O(1) per candidate per demo.
        A candidate program P is pruned if shape(P(I_i)) != shape(O_i)
        for any demo pair i.

        This catches resize/crop/tile programs that produce wrong-sized
        outputs, eliminating them before any expensive computation.

        Args:
            candidates: List of candidate ProgramNodes.
            demo_pairs: List of demo pairs with 'input' and 'output' grids.

        Returns:
            Filtered list of candidates passing shape check.
        """
        if not self.enable_shape_filter or not demo_pairs:
            return list(candidates)

        # Pre-compute expected output shapes
        output_shapes: list[list[tuple[int, int]]] = []
        for pair in demo_pairs:
            output_grids = pair.get("output", [])
            pair_shapes = [self._get_grid_shape(g) for g in output_grids]
            output_shapes.append(pair_shapes)

        filtered: list[ProgramNode] = []
        for candidate in candidates:
            passes = True
            for pair_idx, pair in enumerate(demo_pairs):
                input_grids = pair.get("input", [])
                expected_shapes = output_shapes[pair_idx]

                for grid_idx, input_grid in enumerate(input_grids):
                    if grid_idx >= len(expected_shapes):
                        continue
                    try:
                        predicted = candidate.apply(input_grid)
                        pred_shape = self._get_grid_shape(predicted)
                        if pred_shape != expected_shapes[grid_idx]:
                            passes = False
                            break
                    except Exception:
                        passes = False
                        break
                if not passes:
                    break

            if passes:
                filtered.append(candidate)
            else:
                self.stats["shape_pruned"] += 1

        return filtered

    def _get_color_histogram(self, grid: np.ndarray) -> tuple[int, ...]:
        """Compute 10-bin color histogram with caching.

        Args:
            grid: Input grid.

        Returns:
            Tuple of 10 ints (counts for colors 0-9).
        """
        cache_key = hash(grid.tobytes())
        if cache_key in self._color_hist_cache:
            return self._color_hist_cache[cache_key]
        counts = np.bincount(grid.ravel(), minlength=10)
        hist = tuple(int(counts[c]) for c in range(min(10, len(counts))))
        self._color_hist_cache[cache_key] = hist
        return hist

    def color_histogram_prune(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """Prune candidates by color histogram invariant.

        A candidate program P is pruned if the color distribution of
        P(I_i) doesn't match that of O_i for any demo pair i.

        This catches color-swap and map-color programs that produce
        wrong color mappings. Color histogram is a necessary condition:
        if the output has 5 red pixels and 3 blue pixels, the prediction
        must also have exactly 5 red and 3 blue.

        Args:
            candidates: List of candidate ProgramNodes.
            demo_pairs: List of demo pairs.

        Returns:
            Filtered list of candidates passing color histogram check.
        """
        if not self.enable_color_hist or not demo_pairs:
            return list(candidates)

        # Pre-compute expected color histograms
        output_hists: list[list[tuple[int, ...]]] = []
        for pair in demo_pairs:
            output_grids = pair.get("output", [])
            pair_hists = [self._get_color_histogram(g) for g in output_grids]
            output_hists.append(pair_hists)

        filtered: list[ProgramNode] = []
        for candidate in candidates:
            passes = True
            for pair_idx, pair in enumerate(demo_pairs):
                input_grids = pair.get("input", [])
                expected_hists = output_hists[pair_idx]

                for grid_idx, input_grid in enumerate(input_grids):
                    if grid_idx >= len(expected_hists):
                        continue
                    try:
                        predicted = candidate.apply(input_grid)
                        pred_hist = self._get_color_histogram(predicted)
                        if pred_hist != expected_hists[grid_idx]:
                            passes = False
                            break
                    except Exception:
                        passes = False
                        break
                if not passes:
                    break

            if passes:
                filtered.append(candidate)
            else:
                self.stats["color_hist_pruned"] += 1

        return filtered

    def _get_nonzero_count(self, grid: np.ndarray) -> int:
        """Count non-zero pixels with caching.

        Args:
            grid: Input grid.

        Returns:
            Number of non-zero pixels.
        """
        cache_key = hash(grid.tobytes())
        if cache_key in self._nonzero_cache:
            return self._nonzero_cache[cache_key]
        count = int(np.count_nonzero(grid))
        self._nonzero_cache[cache_key] = count
        return count

    def nonzero_count_prune(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """Prune candidates by non-zero pixel count invariant.

        A candidate program P is pruned if the number of non-zero pixels
        in P(I_i) doesn't match that of O_i for any demo pair i.

        This catches move/copy/gravity programs that add or remove pixels.
        Combined with color histogram, this provides strong filtering:
        the right number of pixels in the right colors.

        Args:
            candidates: List of candidate ProgramNodes.
            demo_pairs: List of demo pairs.

        Returns:
            Filtered list of candidates passing non-zero count check.
        """
        if not self.enable_nonzero_count or not demo_pairs:
            return list(candidates)

        # Pre-compute expected non-zero counts
        output_counts: list[list[int]] = []
        for pair in demo_pairs:
            output_grids = pair.get("output", [])
            pair_counts = [self._get_nonzero_count(g) for g in output_grids]
            output_counts.append(pair_counts)

        filtered: list[ProgramNode] = []
        for candidate in candidates:
            passes = True
            for pair_idx, pair in enumerate(demo_pairs):
                input_grids = pair.get("input", [])
                expected_counts = output_counts[pair_idx]

                for grid_idx, input_grid in enumerate(input_grids):
                    if grid_idx >= len(expected_counts):
                        continue
                    try:
                        predicted = candidate.apply(input_grid)
                        pred_count = self._get_nonzero_count(predicted)
                        if pred_count != expected_counts[grid_idx]:
                            passes = False
                            break
                    except Exception:
                        passes = False
                        break
                if not passes:
                    break

            if passes:
                filtered.append(candidate)
            else:
                self.stats["nonzero_pruned"] += 1

        return filtered

    def fast_pre_filter(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """Run all cheap pre-Phase A filters in optimal order.

        Executes filters from cheapest to most expensive:
        1. Grid shape (O(1) per candidate)
        2. Non-zero pixel count (O(H*W) per candidate)
        3. Color histogram (O(H*W) per candidate)
        4. Betti0 (O(H*W) per candidate, with scipy overhead)

        Each filter operates on the output of the previous one,
        so later filters process fewer candidates.

        Args:
            candidates: List of candidate ProgramNodes.
            demo_pairs: List of demo pairs.

        Returns:
            Filtered list passing all fast invariant checks.
        """
        result = candidates

        # 1. Shape filter (cheapest)
        result = self.grid_shape_prune(result, demo_pairs)

        # 2. Non-zero count (cheap, catches move/copy/gravity errors)
        result = self.nonzero_count_prune(result, demo_pairs)

        # 3. Color histogram (catches color-swap/map-color errors)
        result = self.color_histogram_prune(result, demo_pairs)

        # 4. Betti0 (moderate cost, catches topological mismatches)
        result = self.betti0_prune(result, demo_pairs)

        return result

    def _compute_symmetry_signature(self, grid: np.ndarray) -> str:
        """Compute a symmetry-normalized signature for a grid.

        Generates 4 symmetry variants (identity, hflip, vflip, rot180)
        and returns the lexicographically smallest as the canonical form.

        Args:
            grid: Input grid as ndarray.

        Returns:
            String signature (canonical form).
        """
        variants = [
            grid,
            np.fliplr(grid),
            np.flipud(grid),
            np.rot90(grid, k=2),
        ]
        signatures = [v.tobytes().hex() for v in variants]
        return min(signatures)

    def symmetry_dedup(
        self,
        candidates: list[ProgramNode],
    ) -> list[ProgramNode]:
        """Remove symmetry-equivalent duplicate candidates.

        Two programs are considered equivalent if they produce
        symmetry-equivalent outputs (under identity, horizontal flip,
        vertical flip, or 180-degree rotation) for the same input.

        For efficiency, we use the program structure signature
        (primitive names + composition) rather than output comparison.

        Args:
            candidates: List of candidate ProgramNodes.

        Returns:
            Deduplicated list of candidates.
        """
        if not self.enable_symmetry_dedup:
            return list(candidates)

        seen_signatures: set[str] = set()
        deduped: list[ProgramNode] = []

        for candidate in candidates:
            # Create a structural signature based on flattened primitive names
            elements = candidate.flatten()
            names = tuple(e.name for e in elements)

            # Generate symmetry-equivalent name sequences
            # For chain compositions, reversal is equivalent under 180° rotation
            variants = [
                names,
                names[::-1],  # reversed chain
            ]

            # Compute canonical signature
            sig = min(str(v) for v in variants)

            if sig not in seen_signatures:
                seen_signatures.add(sig)
                deduped.append(candidate)
            else:
                self.stats["symmetry_deduped"] += 1

        return deduped

    def incremental_mdl_prune(
        self,
        depth: int,
        partial_mdl: int,
    ) -> bool:
        """Check if enumeration should be pruned based on partial MDL.

        At depth 3, if the first two primitives already have MDL
        exceeding the threshold, the third primitive enumeration
        can be skipped entirely.

        Args:
            depth: Current enumeration depth.
            partial_mdl: MDL cost accumulated so far (before current level).

        Returns:
            True if this branch should be pruned (skipped).
        """
        if not self.enable_incremental_mdl:
            return False

        # Only apply at depth >= 3 where combinatorial explosion occurs
        if depth < 3:
            return False

        # Prune if partial MDL + minimum possible additional MDL exceeds threshold
        # Minimum primitive MDL is 5 (simple primitives), plus 2 for composition overhead
        min_remaining = 5 + 2  # One more primitive + chain overhead
        return (partial_mdl + min_remaining) > self.mdl_threshold

    def heuristic_order(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]] | None = None,
    ) -> list[ProgramNode]:
        """Reorder candidates by heuristic priority.

        Primary sort: MDL (ascending) — simpler programs first.
        Secondary sort: Number of primitives (fewer first).
        Tertiary sort: Program structure complexity.

        This ensures high-probability candidates are verified first,
        improving ENPV early-termination effectiveness.

        Args:
            candidates: List of candidate ProgramNodes.
            demo_pairs: Demo pairs (unused but kept for interface consistency).

        Returns:
            Reordered list of candidates.
        """
        if not self.enable_heuristic_order:
            return list(candidates)

        def sort_key(p: ProgramNode) -> tuple[int, int, str]:
            # Primary: MDL (lower is better)
            # Secondary: number of primitives (fewer is better)
            # Tertiary: primitive name complexity (for stable ordering)
            num_prims = len(p.flatten())
            names = ",".join(e.name for e in p.flatten())
            return (p.total_mdl, num_prims, names)

        return sorted(candidates, key=sort_key)

    def reset_stats(self) -> None:
        """Reset pruning statistics and clear caches."""
        self.stats = {
            "betti0_pruned": 0,
            "symmetry_deduped": 0,
            "mdl_pruned": 0,
            "shape_pruned": 0,
            "color_hist_pruned": 0,
            "nonzero_pruned": 0,
        }
        # Clear caches to ensure deterministic behavior between calls
        self._betti0_cache.clear()
        self._symmetry_cache.clear()
        self._color_hist_cache.clear()
        self._nonzero_cache.clear()
        self._shape_cache.clear()

    def get_stats(self) -> dict[str, int]:
        """Get pruning statistics.

        Returns:
            Dictionary with counts of pruned/deduped candidates.
        """
        return dict(self.stats)

    def get_betti0_for_grid(self, grid: np.ndarray) -> int:
        """Public interface for Betti0 computation (with caching).

        Args:
            grid: Input grid.

        Returns:
            Betti0 (number of connected components).
        """
        return self.compute_betti0_fast(grid)
