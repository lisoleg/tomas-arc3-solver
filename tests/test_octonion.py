"""Octonion hyperedge encoding reversibility tests.

Tests that encode_grid -> decode_to_grid is reversible for various
grid sizes and color values. Also tests normalization stability,
EncodingReversibilityError, and HyperGraph topology operations.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.octonion_hyperedge import (
    EncodingReversibilityError,
    OctonionHyperEdge,
)
from src.core.hypergraph import HyperGraph


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def grid_3x3() -> np.ndarray:
    """A 3x3 grid with multiple colors."""
    return np.array([
        [1, 0, 2],
        [0, 3, 0],
        [4, 0, 5],
    ], dtype=np.int8)


@pytest.fixture
def grid_all_colors() -> np.ndarray:
    """A grid containing all colors 0-9."""
    return np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
    ], dtype=np.int8)


# ============================================================
# Encoding Reversibility Tests
# ============================================================

class TestEncodeDecodeReversibility:
    """Tests that encoding is reversible (encode -> decode == original)."""

    def test_reversibility_3x3(self, grid_3x3):
        """3x3 grid should be reversible."""
        edge = OctonionHyperEdge(grid_3x3)
        decoded = edge.decode_to_grid()
        np.testing.assert_array_equal(decoded, grid_3x3)

    def test_reversibility_5x5(self, grid_all_colors):
        """5x5 grid with all colors should be reversible."""
        edge = OctonionHyperEdge(grid_all_colors)
        decoded = edge.decode_to_grid()
        np.testing.assert_array_equal(decoded, grid_all_colors)

    def test_reversibility_10x10(self):
        """10x10 grid should be reversible."""
        rng = np.random.RandomState(42)
        grid = rng.randint(0, 10, size=(10, 10)).astype(np.int8)
        edge = OctonionHyperEdge(grid)
        decoded = edge.decode_to_grid()
        np.testing.assert_array_equal(decoded, grid)

    def test_reversibility_30x30(self):
        """30x30 grid should be reversible."""
        rng = np.random.RandomState(123)
        grid = rng.randint(0, 10, size=(30, 30)).astype(np.int8)
        edge = OctonionHyperEdge(grid)
        decoded = edge.decode_to_grid()
        np.testing.assert_array_equal(decoded, grid)

    def test_reversibility_all_zeros(self):
        """All-zero grid should be reversible."""
        grid = np.zeros((5, 5), dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        decoded = edge.decode_to_grid()
        np.testing.assert_array_equal(decoded, grid)

    def test_reversibility_single_color(self):
        """Single-color grid should be reversible."""
        grid = np.full((4, 4), 7, dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        decoded = edge.decode_to_grid()
        np.testing.assert_array_equal(decoded, grid)

    def test_reversibility_1x1(self):
        """1x1 grid edge case should be reversible."""
        grid = np.array([[5]], dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        decoded = edge.decode_to_grid()
        np.testing.assert_array_equal(decoded, grid)


# ============================================================
# Color Value Tests
# ============================================================

class TestColorValues:
    """Tests for different color values 0-9."""

    @pytest.mark.parametrize("color", range(10))
    def test_single_color_reversibility(self, color):
        """Each color value 0-9 should encode/decode correctly."""
        grid = np.full((3, 3), color, dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        decoded = edge.decode_to_grid()
        np.testing.assert_array_equal(decoded, grid)

    def test_color_phase_encoding(self):
        """Color should be encoded as phase angle in e3_color."""
        grid = np.array([[9]], dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        arr = edge.to_array()
        # e3_color = color/9 * 2*pi for color=9 -> 2*pi
        phase = arr[0, 3]
        expected_phase = (9 / 9) * 2.0 * np.pi
        assert abs(phase - expected_phase) < 0.01


# ============================================================
# Normalization Stability Tests
# ============================================================

class TestNormalization:
    """Tests for normalization stability with long sequences."""

    def test_normalization_clips_to_valid_ranges(self):
        """All components should be in valid ranges after normalization."""
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        arr = edge.to_array()
        # real_mask in [0, 1]
        assert np.all(arr[:, 0] >= 0) and np.all(arr[:, 0] <= 1)
        # e1_x in [-1, 1]
        assert np.all(arr[:, 1] >= -1) and np.all(arr[:, 1] <= 1)
        # e2_y in [-1, 1]
        assert np.all(arr[:, 2] >= -1) and np.all(arr[:, 2] <= 1)
        # e3_color in [0, 2*pi]
        assert np.all(arr[:, 3] >= 0) and np.all(arr[:, 3] <= 2 * np.pi + 0.01)
        # e4-e7 in [0, 1]
        for i in range(4, 8):
            assert np.all(arr[:, i] >= 0) and np.all(arr[:, i] <= 1)

    def test_long_sequence_no_overflow(self):
        """Large grid (simulating long sequence) should not overflow."""
        rng = np.random.RandomState(99)
        grid = rng.randint(0, 10, size=(50, 50)).astype(np.int8)
        edge = OctonionHyperEdge(grid)
        arr = edge.to_array()
        # Check no NaN or Inf
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))
        # All values should be finite and in valid ranges
        assert np.all(np.isfinite(arr))

    def test_repeated_encoding_stable(self):
        """Repeated encoding should produce stable results."""
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=np.int8)
        edge1 = OctonionHyperEdge(grid)
        edge2 = OctonionHyperEdge(grid)
        np.testing.assert_array_equal(edge1.to_array(), edge2.to_array())


# ============================================================
# EncodingReversibilityError Tests
# ============================================================

class TestEncodingError:
    """Tests for EncodingReversibilityError."""

    def test_decode_without_encoding_raises(self):
        """Decoding without encoding should raise EncodingReversibilityError."""
        edge = OctonionHyperEdge()  # No grid provided
        with pytest.raises(EncodingReversibilityError):
            edge.decode_to_grid()

    def test_error_is_exception(self):
        """EncodingReversibilityError should be an Exception subclass."""
        assert issubclass(EncodingReversibilityError, Exception)


# ============================================================
# Topological Invariants Tests
# ============================================================

class TestTopoInvariants:
    """Tests for topological invariant computation."""

    def test_compute_betti0_single_component(self):
        """Betti0 should be 1 for a single connected component."""
        grid = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        assert edge.compute_betti0() == 1

    def test_compute_betti0_multiple_components(self):
        """Betti0 should count connected components."""
        grid = np.array([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ], dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        assert edge.compute_betti0() == 4

    def test_compute_betti0_empty_grid(self):
        """Betti0 should be 0 for all-zero grid."""
        grid = np.zeros((3, 3), dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        assert edge.compute_betti0() == 0

    def test_compute_topo_invariants_returns_dict(self, grid_3x3):
        """compute_topo_invariants should return a dict with expected keys."""
        edge = OctonionHyperEdge(grid_3x3)
        inv = edge.compute_topo_invariants()
        assert "betti0" in inv
        assert "symmetry_residual" in inv
        assert "color_permutation_group" in inv
        assert isinstance(inv["betti0"], int)
        assert isinstance(inv["symmetry_residual"], float)
        assert isinstance(inv["color_permutation_group"], list)

    def test_color_permutation_group(self, grid_3x3):
        """Color permutation group should list unique non-zero colors."""
        edge = OctonionHyperEdge(grid_3x3)
        inv = edge.compute_topo_invariants()
        colors = inv["color_permutation_group"]
        assert 1 in colors
        assert 2 in colors
        assert 3 in colors
        assert 4 in colors
        assert 5 in colors

    def test_symmetry_residual_range(self, grid_3x3):
        """Symmetry residual should be in [0, 1]."""
        edge = OctonionHyperEdge(grid_3x3)
        sym = edge.compute_symmetry_residual()
        assert 0.0 <= sym <= 1.0


# ============================================================
# Grid Shape and Array Tests
# ============================================================

class TestGridShapeAndArray:
    """Tests for grid shape and array access."""

    def test_get_grid_shape(self, grid_3x3):
        """get_grid_shape should return original grid dimensions."""
        edge = OctonionHyperEdge(grid_3x3)
        assert edge.get_grid_shape() == (3, 3)

    def test_to_array_shape(self, grid_3x3):
        """to_array should return (H*W, 8) array."""
        edge = OctonionHyperEdge(grid_3x3)
        arr = edge.to_array()
        assert arr.shape == (9, 8)  # 3*3 pixels, 8 components

    def test_to_array_dtype(self, grid_3x3):
        """to_array should return float32."""
        edge = OctonionHyperEdge(grid_3x3)
        arr = edge.to_array()
        assert arr.dtype == np.float32

    def test_empty_edge_to_array(self):
        """Empty edge should return empty array."""
        edge = OctonionHyperEdge()
        arr = edge.to_array()
        assert arr.shape == (0, 8)


# ============================================================
# HyperGraph Tests
# ============================================================

class TestHyperGraph:
    """Tests for HyperGraph topology operations."""

    def test_topo_hash_deterministic(self, grid_3x3):
        """Topo hash should be deterministic for same grid."""
        edge1 = OctonionHyperEdge(grid_3x3)
        edge2 = OctonionHyperEdge(grid_3x3)
        hg1 = HyperGraph([edge1])
        hg2 = HyperGraph([edge2])
        assert hg1.get_topo_hash() == hg2.get_topo_hash()

    def test_topo_hash_different_for_different_grids(self):
        """Different grids should have different topo hashes."""
        grid1 = np.array([[1, 0], [0, 0]], dtype=np.int8)
        grid2 = np.array([[2, 0], [0, 0]], dtype=np.int8)
        edge1 = OctonionHyperEdge(grid1)
        edge2 = OctonionHyperEdge(grid2)
        hg1 = HyperGraph([edge1])
        hg2 = HyperGraph([edge2])
        assert hg1.get_topo_hash() != hg2.get_topo_hash()

    def test_is_isomorphic_same_grid(self, grid_3x3):
        """Same grid should be isomorphic."""
        edge1 = OctonionHyperEdge(grid_3x3)
        edge2 = OctonionHyperEdge(grid_3x3)
        hg1 = HyperGraph([edge1])
        hg2 = HyperGraph([edge2])
        assert hg1.is_isomorphic(hg2)

    def test_is_not_isomorphic_different_grids(self):
        """Different grids should not be isomorphic."""
        grid1 = np.array([[1, 0], [0, 0]], dtype=np.int8)
        grid2 = np.array([[0, 0], [0, 0]], dtype=np.int8)
        edge1 = OctonionHyperEdge(grid1)
        edge2 = OctonionHyperEdge(grid2)
        hg1 = HyperGraph([edge1])
        hg2 = HyperGraph([edge2])
        assert not hg1.is_isomorphic(hg2)

    def test_compute_fiber(self, grid_3x3):
        """compute_fiber should return a set of state hashes."""
        from src.core.dsl_primitives import DSLElement, ProgramNode

        edge = OctonionHyperEdge(grid_3x3)
        hg = HyperGraph([edge])
        program = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        fiber = hg.compute_fiber(program)
        assert isinstance(fiber, set)
        assert len(fiber) >= 1  # At least one state

    def test_compute_betti0_hypergraph(self):
        """HyperGraph Betti0 should sum across edges."""
        grid = np.array([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ], dtype=np.int8)
        edge = OctonionHyperEdge(grid)
        hg = HyperGraph([edge])
        assert hg.compute_betti0() == 4

    def test_temporal_sequence_single_frame(self, grid_3x3):
        """Non-temporal hypergraph should return itself as single element."""
        edge = OctonionHyperEdge(grid_3x3, frame_idx=0)
        hg = HyperGraph([edge])
        seq = hg.get_temporal_sequence()
        assert len(seq) == 1

    def test_temporal_sequence_multi_frame(self):
        """Temporal hypergraph should split into per-frame subgraphs."""
        grid1 = np.array([[1, 0], [0, 0]], dtype=np.int8)
        grid2 = np.array([[0, 1], [0, 0]], dtype=np.int8)
        edge1 = OctonionHyperEdge(grid1, frame_idx=0)
        edge2 = OctonionHyperEdge(grid2, frame_idx=1)
        hg = HyperGraph([edge1, edge2])
        assert hg.is_temporal
        seq = hg.get_temporal_sequence()
        assert len(seq) == 2

    def test_empty_hypergraph_hash(self):
        """Empty hypergraph should have a valid hash."""
        hg = HyperGraph([])
        h = hg.get_topo_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_color_permutation_group(self, grid_3x3):
        """HyperGraph color permutation group should list unique colors."""
        edge = OctonionHyperEdge(grid_3x3)
        hg = HyperGraph([edge])
        colors = hg.compute_color_permutation_group()
        assert isinstance(colors, list)
        assert colors == sorted(colors)

    def test_symmetry_axis_residual(self, grid_3x3):
        """HyperGraph symmetry residual should be in [0, 1]."""
        edge = OctonionHyperEdge(grid_3x3)
        hg = HyperGraph([edge])
        sym = hg.compute_symmetry_axis_residual()
        assert 0.0 <= sym <= 1.0
