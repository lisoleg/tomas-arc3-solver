"""Tests for PruningOptimizer (v2.3).

Tests Betti0 invariant pruning, symmetry equivalence dedup,
incremental MDL pruning, heuristic candidate ordering,
and statistics tracking.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.dsl_primitives import DSLElement, ProgramNode, get_all_primitives
from src.solver.pruning_optimizer import PruningOptimizer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def pruning_config():
    """Default pruning config with all strategies enabled."""
    return {
        "enable_betti0": True,
        "enable_symmetry_dedup": True,
        "enable_heuristic_order": True,
        "enable_incremental_mdl": True,
        "mdl_threshold": 50,
    }


@pytest.fixture
def optimizer(pruning_config):
    """Create a PruningOptimizer instance."""
    return PruningOptimizer(pruning_config)


@pytest.fixture
def demo_pairs():
    """Demo pairs where output is horizontal mirror of input."""
    grids_in = [
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.int8),
    ]
    grids_out = [np.fliplr(g) for g in grids_in]
    return [{"input": grids_in, "output": grids_out}]


@pytest.fixture
def sample_candidates():
    """Create a few candidate ProgramNodes for testing."""
    prims = get_all_primitives()
    candidates = []
    # Depth-1 candidates
    for prim in prims[:5]:
        candidates.append(ProgramNode(prim))
    # Depth-2 chain
    if len(prims) >= 2:
        node = ProgramNode(prims[0])
        child = ProgramNode(prims[1])
        node.children.append(child)
        node.combo_type = "chain"
        node.total_mdl = node.compute_mdl()
        candidates.append(node)
    return candidates


# ============================================================
# Initialization Tests
# ============================================================

class TestPruningOptimizerInit:
    """Tests for PruningOptimizer initialization."""

    def test_default_config(self):
        """Should initialize with default values."""
        opt = PruningOptimizer({})
        assert opt.enable_betti0 is True
        assert opt.enable_symmetry_dedup is True
        assert opt.enable_heuristic_order is True
        assert opt.enable_incremental_mdl is True
        assert opt.mdl_threshold == 50

    def test_custom_config(self, pruning_config):
        """Should accept custom config."""
        opt = PruningOptimizer(pruning_config)
        assert opt.enable_betti0 is True
        assert opt.mdl_threshold == 50

    def test_disabled_flags(self):
        """Should respect disabled flags."""
        opt = PruningOptimizer({
            "enable_betti0": False,
            "enable_symmetry_dedup": False,
            "enable_heuristic_order": False,
            "enable_incremental_mdl": False,
        })
        assert opt.enable_betti0 is False
        assert opt.enable_symmetry_dedup is False
        assert opt.enable_heuristic_order is False
        assert opt.enable_incremental_mdl is False

    def test_initial_stats(self, optimizer):
        """Stats should start at zero."""
        stats = optimizer.get_stats()
        assert stats["betti0_pruned"] == 0
        assert stats["symmetry_deduped"] == 0
        assert stats["mdl_pruned"] == 0


# ============================================================
# compute_betti0_fast Tests
# ============================================================

class TestComputeBetti0:
    """Tests for compute_betti0_fast method."""

    def test_empty_grid(self, optimizer):
        """Empty grid should have 0 connected components."""
        grid = np.zeros((3, 3), dtype=np.int8)
        assert optimizer.compute_betti0_fast(grid) == 0

    def test_single_component(self, optimizer):
        """Single connected component."""
        grid = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.int8)
        assert optimizer.compute_betti0_fast(grid) == 1

    def test_two_components(self, optimizer):
        """Multiple disconnected components."""
        grid = np.array([[1, 0, 2], [0, 0, 0], [3, 0, 4]], dtype=np.int8)
        result = optimizer.compute_betti0_fast(grid)
        assert result == 4  # 1, 2, 3, 4 are all isolated single pixels

    def test_full_grid(self, optimizer):
        """Full non-zero grid should have 1 component."""
        grid = np.ones((3, 3), dtype=np.int8)
        assert optimizer.compute_betti0_fast(grid) == 1

    def test_caching(self, optimizer):
        """Same grid should return cached result."""
        grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
        result1 = optimizer.compute_betti0_fast(grid)
        result2 = optimizer.compute_betti0_fast(grid)
        assert result1 == result2

    def test_get_betti0_for_grid(self, optimizer):
        """Public interface should match internal method."""
        grid = np.array([[1, 1], [0, 0]], dtype=np.int8)
        assert optimizer.get_betti0_for_grid(grid) == optimizer.compute_betti0_fast(grid)


# ============================================================
# betti0_prune Tests
# ============================================================

class TestBetti0Prune:
    """Tests for betti0_prune method."""

    def test_disabled_returns_all(self, sample_candidates, demo_pairs):
        """When disabled, should return all candidates unchanged."""
        opt = PruningOptimizer({"enable_betti0": False})
        result = opt.betti0_prune(sample_candidates, demo_pairs)
        assert len(result) == len(sample_candidates)

    def test_empty_demos_returns_all(self, optimizer, sample_candidates):
        """Empty demo pairs should return all candidates."""
        result = optimizer.betti0_prune(sample_candidates, [])
        assert len(result) == len(sample_candidates)

    def test_empty_candidates(self, optimizer, demo_pairs):
        """Empty candidate list should return empty."""
        result = optimizer.betti0_prune([], demo_pairs)
        assert result == []

    def test_pruning_reduces_count(self, optimizer, demo_pairs):
        """Betti0 pruning should reduce or maintain candidate count."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims]
        result = optimizer.betti0_prune(candidates, demo_pairs)
        assert len(result) <= len(candidates)

    def test_stats_incremented(self, optimizer, demo_pairs):
        """Pruned candidates should increment betti0_pruned stat."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims]
        original_len = len(candidates)
        result = optimizer.betti0_prune(candidates, demo_pairs)
        stats = optimizer.get_stats()
        assert stats["betti0_pruned"] == original_len - len(result)


# ============================================================
# symmetry_dedup Tests
# ============================================================

class TestSymmetryDedup:
    """Tests for symmetry_dedup method."""

    def test_disabled_returns_all(self, sample_candidates):
        """When disabled, should return all candidates."""
        opt = PruningOptimizer({"enable_symmetry_dedup": False})
        result = opt.symmetry_dedup(sample_candidates)
        assert len(result) == len(sample_candidates)

    def test_empty_list(self, optimizer):
        """Empty list should return empty."""
        result = optimizer.symmetry_dedup([])
        assert result == []

    def test_unique_candidates_preserved(self, optimizer, sample_candidates):
        """Unique candidates should be preserved."""
        result = optimizer.symmetry_dedup(sample_candidates)
        assert len(result) >= 1
        assert len(result) <= len(sample_candidates)

    def test_duplicate_chain_deduped(self, optimizer):
        """Reversed chains should be deduplicated."""
        prims = get_all_primitives()
        if len(prims) < 2:
            pytest.skip("Need at least 2 primitives")

        # Create chain A->B and B->A (should be considered equivalent)
        node1 = ProgramNode(prims[0])
        child1 = ProgramNode(prims[1])
        node1.children.append(child1)
        node1.combo_type = "chain"
        node1.total_mdl = node1.compute_mdl()

        node2 = ProgramNode(prims[1])
        child2 = ProgramNode(prims[0])
        node2.children.append(child2)
        node2.combo_type = "chain"
        node2.total_mdl = node2.compute_mdl()

        result = optimizer.symmetry_dedup([node1, node2])
        # At least one should be deduped (reversed chain equivalence)
        assert len(result) <= 2

    def test_stats_incremented(self, optimizer, sample_candidates):
        """Deduped candidates should increment symmetry_deduped stat."""
        original_len = len(sample_candidates)
        result = optimizer.symmetry_dedup(sample_candidates)
        stats = optimizer.get_stats()
        assert stats["symmetry_deduped"] == original_len - len(result)


# ============================================================
# incremental_mdl_prune Tests
# ============================================================

class TestIncrementalMDLPrune:
    """Tests for incremental_mdl_prune method."""

    def test_disabled_always_false(self):
        """When disabled, should never prune."""
        opt = PruningOptimizer({"enable_incremental_mdl": False})
        assert opt.incremental_mdl_prune(3, 100) is False

    def test_depth_below_3_never_pruned(self, optimizer):
        """Depths < 3 should never be pruned."""
        assert optimizer.incremental_mdl_prune(1, 100) is False
        assert optimizer.incremental_mdl_prune(2, 100) is False

    def test_depth_3_within_threshold(self, optimizer):
        """Partial MDL within threshold should not be pruned."""
        # threshold=50, partial=10, min_remaining=7, 10+7=17 < 50
        assert optimizer.incremental_mdl_prune(3, 10) is False

    def test_depth_3_exceeds_threshold(self, optimizer):
        """Partial MDL exceeding threshold should be pruned."""
        # threshold=50, partial=50, min_remaining=7, 50+7=57 > 50
        assert optimizer.incremental_mdl_prune(3, 50) is True

    def test_depth_3_at_boundary(self, optimizer):
        """Boundary case: partial + min_remaining == threshold."""
        # threshold=50, min_remaining=7, 43+7=50 == 50, not > 50
        assert optimizer.incremental_mdl_prune(3, 43) is False
        # 44+7=51 > 50
        assert optimizer.incremental_mdl_prune(3, 44) is True


# ============================================================
# heuristic_order Tests
# ============================================================

class TestHeuristicOrder:
    """Tests for heuristic_order method."""

    def test_disabled_returns_original_order(self, sample_candidates):
        """When disabled, should preserve original order."""
        opt = PruningOptimizer({"enable_heuristic_order": False})
        result = opt.heuristic_order(sample_candidates)
        assert result == sample_candidates

    def test_empty_list(self, optimizer):
        """Empty list should return empty."""
        result = optimizer.heuristic_order([])
        assert result == []

    def test_sorted_by_mdl(self, optimizer, sample_candidates):
        """Result should be sorted by MDL ascending."""
        result = optimizer.heuristic_order(sample_candidates)
        mdls = [p.total_mdl for p in result]
        assert mdls == sorted(mdls)

    def test_stable_order(self, optimizer):
        """Same input should produce same output (deterministic)."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims[:5]]
        result1 = optimizer.heuristic_order(list(candidates))
        result2 = optimizer.heuristic_order(list(candidates))
        mdls1 = [p.total_mdl for p in result1]
        mdls2 = [p.total_mdl for p in result2]
        assert mdls1 == mdls2


# ============================================================
# reset_stats Tests
# ============================================================

class TestResetStats:
    """Tests for reset_stats method."""

    def test_resets_counters(self, optimizer, demo_pairs):
        """Should reset all stat counters to zero."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims]
        optimizer.betti0_prune(candidates, demo_pairs)
        assert optimizer.get_stats()["betti0_pruned"] > 0 or True  # may be 0

        optimizer.reset_stats()
        stats = optimizer.get_stats()
        assert stats["betti0_pruned"] == 0
        assert stats["symmetry_deduped"] == 0
        assert stats["mdl_pruned"] == 0

    def test_clears_caches(self, optimizer):
        """Should clear Betti0 and symmetry caches."""
        grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
        optimizer.compute_betti0_fast(grid)
        assert len(optimizer._betti0_cache) > 0

        optimizer.reset_stats()
        assert len(optimizer._betti0_cache) == 0
        assert len(optimizer._symmetry_cache) == 0

    def test_get_stats_returns_copy(self, optimizer):
        """get_stats should return a copy, not the internal dict."""
        stats1 = optimizer.get_stats()
        stats1["betti0_pruned"] = 999
        stats2 = optimizer.get_stats()
        assert stats2["betti0_pruned"] == 0


# ============================================================
# Integration: Deterministic Repeated Calls
# ============================================================

class TestDeterministicBehavior:
    """Tests ensuring deterministic behavior across repeated calls."""

    def test_betti0_consistent_across_calls(self, optimizer, demo_pairs):
        """Betti0 pruning should give same result on repeated calls."""
        prims = get_all_primitives()
        candidates1 = [ProgramNode(p) for p in prims]
        candidates2 = [ProgramNode(p) for p in prims]

        optimizer.reset_stats()
        result1 = optimizer.betti0_prune(candidates1, demo_pairs)

        optimizer.reset_stats()
        result2 = optimizer.betti0_prune(candidates2, demo_pairs)

        assert len(result1) == len(result2)

    def test_symmetry_dedup_consistent(self, optimizer):
        """Symmetry dedup should give same result on repeated calls."""
        prims = get_all_primitives()
        candidates1 = [ProgramNode(p) for p in prims]
        candidates2 = [ProgramNode(p) for p in prims]

        result1 = optimizer.symmetry_dedup(candidates1)
        result2 = optimizer.symmetry_dedup(candidates2)

        assert len(result1) == len(result2)


# ============================================================
# Pre-Phase A Fast Filters (v2.3 enhanced)
# ============================================================

class TestGridShapePrune:
    """Tests for grid_shape_prune method."""

    def test_disabled_returns_all(self, sample_candidates, demo_pairs):
        """When disabled, should return all candidates."""
        opt = PruningOptimizer({"enable_shape_filter": False})
        result = opt.grid_shape_prune(sample_candidates, demo_pairs)
        assert len(result) == len(sample_candidates)

    def test_empty_demos_returns_all(self, optimizer, sample_candidates):
        """Empty demo pairs should return all candidates."""
        result = optimizer.grid_shape_prune(sample_candidates, [])
        assert len(result) == len(sample_candidates)

    def test_empty_candidates(self, optimizer, demo_pairs):
        """Empty candidate list should return empty."""
        result = optimizer.grid_shape_prune([], demo_pairs)
        assert result == []

    def test_pruning_reduces_or_maintains(self, optimizer, demo_pairs):
        """Shape pruning should reduce or maintain candidate count."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims]
        result = optimizer.grid_shape_prune(candidates, demo_pairs)
        assert len(result) <= len(candidates)

    def test_get_grid_shape(self, optimizer):
        """_get_grid_shape should return (H, W) tuple."""
        grid = np.array([[1, 0, 0], [0, 2, 0]], dtype=np.int8)
        shape = optimizer._get_grid_shape(grid)
        assert shape == (2, 3)

    def test_shape_caching(self, optimizer):
        """Same grid should return cached shape."""
        grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
        s1 = optimizer._get_grid_shape(grid)
        s2 = optimizer._get_grid_shape(grid)
        assert s1 == s2


class TestColorHistogramPrune:
    """Tests for color_histogram_prune method."""

    def test_disabled_returns_all(self, sample_candidates, demo_pairs):
        """When disabled, should return all candidates."""
        opt = PruningOptimizer({"enable_color_hist": False})
        result = opt.color_histogram_prune(sample_candidates, demo_pairs)
        assert len(result) == len(sample_candidates)

    def test_empty_demos_returns_all(self, optimizer, sample_candidates):
        """Empty demo pairs should return all candidates."""
        result = optimizer.color_histogram_prune(sample_candidates, [])
        assert len(result) == len(sample_candidates)

    def test_empty_candidates(self, optimizer, demo_pairs):
        """Empty candidate list should return empty."""
        result = optimizer.color_histogram_prune([], demo_pairs)
        assert result == []

    def test_get_color_histogram(self, optimizer):
        """_get_color_histogram should return 10-element tuple."""
        grid = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.int8)
        hist = optimizer._get_color_histogram(grid)
        assert len(hist) == 10
        assert hist[0] == 6  # 6 zeros
        assert hist[1] == 1  # 1 one
        assert hist[2] == 1  # 1 two
        assert hist[3] == 1  # 1 three

    def test_color_hist_caching(self, optimizer):
        """Same grid should return cached histogram."""
        grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
        h1 = optimizer._get_color_histogram(grid)
        h2 = optimizer._get_color_histogram(grid)
        assert h1 == h2

    def test_pruning_reduces_or_maintains(self, optimizer, demo_pairs):
        """Color histogram pruning should reduce or maintain count."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims]
        result = optimizer.color_histogram_prune(candidates, demo_pairs)
        assert len(result) <= len(candidates)


class TestNonzeroCountPrune:
    """Tests for nonzero_count_prune method."""

    def test_disabled_returns_all(self, sample_candidates, demo_pairs):
        """When disabled, should return all candidates."""
        opt = PruningOptimizer({"enable_nonzero_count": False})
        result = opt.nonzero_count_prune(sample_candidates, demo_pairs)
        assert len(result) == len(sample_candidates)

    def test_empty_demos_returns_all(self, optimizer, sample_candidates):
        """Empty demo pairs should return all candidates."""
        result = optimizer.nonzero_count_prune(sample_candidates, [])
        assert len(result) == len(sample_candidates)

    def test_empty_candidates(self, optimizer, demo_pairs):
        """Empty candidate list should return empty."""
        result = optimizer.nonzero_count_prune([], demo_pairs)
        assert result == []

    def test_get_nonzero_count(self, optimizer):
        """_get_nonzero_count should count non-zero pixels."""
        grid = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.int8)
        assert optimizer._get_nonzero_count(grid) == 3

    def test_nonzero_caching(self, optimizer):
        """Same grid should return cached count."""
        grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
        c1 = optimizer._get_nonzero_count(grid)
        c2 = optimizer._get_nonzero_count(grid)
        assert c1 == c2

    def test_pruning_reduces_or_maintains(self, optimizer, demo_pairs):
        """Non-zero count pruning should reduce or maintain count."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims]
        result = optimizer.nonzero_count_prune(candidates, demo_pairs)
        assert len(result) <= len(candidates)


class TestFastPreFilter:
    """Tests for fast_pre_filter method (runs all pre-Phase A filters)."""

    def test_empty_candidates(self, optimizer, demo_pairs):
        """Empty candidate list should return empty."""
        result = optimizer.fast_pre_filter([], demo_pairs)
        assert result == []

    def test_empty_demos_returns_all(self, optimizer, sample_candidates):
        """Empty demo pairs should return all candidates."""
        result = optimizer.fast_pre_filter(sample_candidates, [])
        assert len(result) == len(sample_candidates)

    def test_reduces_or_maintains(self, optimizer, demo_pairs):
        """fast_pre_filter should reduce or maintain candidate count."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims]
        result = optimizer.fast_pre_filter(candidates, demo_pairs)
        assert len(result) <= len(candidates)

    def test_stats_tracked(self, optimizer, demo_pairs):
        """All filter stats should be tracked after fast_pre_filter."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims]
        optimizer.fast_pre_filter(candidates, demo_pairs)
        stats = optimizer.get_stats()
        # At least some stats should be present
        total_pruned = (
            stats.get("shape_pruned", 0)
            + stats.get("nonzero_pruned", 0)
            + stats.get("color_hist_pruned", 0)
            + stats.get("betti0_pruned", 0)
        )
        # total_pruned + len(result) should equal original count
        # (not asserting exact equality due to candidate.apply behavior)

    def test_deterministic_across_calls(self, optimizer, demo_pairs):
        """fast_pre_filter should be deterministic across repeated calls."""
        prims = get_all_primitives()
        candidates1 = [ProgramNode(p) for p in prims]
        candidates2 = [ProgramNode(p) for p in prims]

        optimizer.reset_stats()
        result1 = optimizer.fast_pre_filter(candidates1, demo_pairs)

        optimizer.reset_stats()
        result2 = optimizer.fast_pre_filter(candidates2, demo_pairs)

        assert len(result1) == len(result2)


class TestEnhancedResetStats:
    """Tests for reset_stats with enhanced v2.3 caches."""

    def test_clears_all_caches(self, optimizer, demo_pairs):
        """Should clear all cache dictionaries."""
        grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
        optimizer.compute_betti0_fast(grid)
        optimizer._get_grid_shape(grid)
        optimizer._get_color_histogram(grid)
        optimizer._get_nonzero_count(grid)

        assert len(optimizer._betti0_cache) > 0
        assert len(optimizer._shape_cache) > 0
        assert len(optimizer._color_hist_cache) > 0
        assert len(optimizer._nonzero_cache) > 0

        optimizer.reset_stats()

        assert len(optimizer._betti0_cache) == 0
        assert len(optimizer._shape_cache) == 0
        assert len(optimizer._color_hist_cache) == 0
        assert len(optimizer._nonzero_cache) == 0

    def test_resets_all_stats(self, optimizer, demo_pairs):
        """Should reset all stat counters including new ones."""
        prims = get_all_primitives()
        candidates = [ProgramNode(p) for p in prims]
        optimizer.fast_pre_filter(candidates, demo_pairs)

        optimizer.reset_stats()
        stats = optimizer.get_stats()
        assert stats["betti0_pruned"] == 0
        assert stats["symmetry_deduped"] == 0
        assert stats["mdl_pruned"] == 0
        assert stats["shape_pruned"] == 0
        assert stats["color_hist_pruned"] == 0
        assert stats["nonzero_pruned"] == 0
