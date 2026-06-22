"""Tests for CUDA GPU batch verification kernels (v2.3).

Tests CudaBatchVerifier with CPU fallback (no GPU required).
Covers batch_grid_equal, batch_grid_distance, batch_apply_mirror,
batch_apply_rotate, verify_candidates_batch, auto_batch_size,
and get_memory_info.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.cuda_kernels import CudaBatchVerifier, HAS_CUDA


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def verifier():
    """Create a CudaBatchVerifier instance (CPU fallback mode)."""
    return CudaBatchVerifier(batch_size=8, device=0)


@pytest.fixture
def sample_grids():
    """Sample 3x3 int8 grids for testing."""
    return [
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.int8),
        np.array([[0, 0, 1], [0, 2, 0], [3, 0, 0]], dtype=np.int8),  # hflip
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.int8),  # same as 0
    ]


@pytest.fixture
def sample_batch():
    """Batch of 4 prediction grids (4, 3, 3) int8."""
    return np.array([
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        [[0, 0, 1], [0, 2, 0], [3, 0, 0]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ], dtype=np.int8)


# ============================================================
# Initialization Tests
# ============================================================

class TestCudaBatchVerifierInit:
    """Tests for CudaBatchVerifier initialization."""

    def test_default_init(self):
        """Verifier should initialize with default parameters."""
        v = CudaBatchVerifier()
        assert v.batch_size == 256
        assert v.device == 0
        assert isinstance(v.is_available, bool)

    def test_custom_init(self):
        """Verifier should accept custom parameters."""
        v = CudaBatchVerifier(batch_size=128, device=1)
        assert v.batch_size == 128
        assert v.device == 1

    def test_is_available_matches_flag(self, verifier):
        """is_available should match HAS_CUDA flag."""
        assert verifier.is_available == HAS_CUDA


# ============================================================
# batch_grid_equal Tests
# ============================================================

class TestBatchGridEqual:
    """Tests for batch_grid_equal method."""

    def test_identical_grids(self, verifier, sample_batch):
        """Identical grids should return True."""
        expected = np.expand_dims(sample_batch[0], axis=0)
        preds = np.expand_dims(sample_batch[0], axis=0)
        result = verifier.batch_grid_equal(preds, expected)
        assert result.shape == (1, 1)
        assert result[0, 0] == True  # noqa: E712

    def test_different_grids(self, verifier, sample_batch):
        """Different grids should return False."""
        # Same grid -> True
        preds = np.expand_dims(sample_batch[0], axis=0)
        expected_same = np.expand_dims(sample_batch[0], axis=0)
        result = verifier.batch_grid_equal(preds, expected_same)
        assert result[0, 0] == True  # noqa: E712

        # Different grid -> False
        expected_diff = np.expand_dims(sample_batch[1], axis=0)  # hflip
        result_diff = verifier.batch_grid_equal(preds, expected_diff)
        assert result_diff[0, 0] == False  # noqa: E712

    def test_n_vs_m_matrix(self, verifier, sample_batch):
        """Should return (N, M) boolean matrix."""
        preds = sample_batch[:3]  # (3, 3, 3)
        expected = sample_batch[:2]  # (2, 3, 3)
        result = verifier.batch_grid_equal(preds, expected)
        assert result.shape == (3, 2)
        assert result.dtype == bool

    def test_all_pairs_correctness(self, verifier, sample_batch):
        """Verify all N*M pair comparisons are correct."""
        preds = sample_batch  # (4, 3, 3)
        expected = np.array([sample_batch[0], sample_batch[1]])  # (2, 3, 3)
        result = verifier.batch_grid_equal(preds, expected)

        for i in range(4):
            for j in range(2):
                expected_val = np.array_equal(preds[i], expected[j])
                assert result[i, j] == expected_val

    def test_shape_mismatch_returns_false(self, verifier):
        """Mismatched spatial dimensions should return all False."""
        preds = np.zeros((2, 3, 3), dtype=np.int8)
        expected = np.zeros((1, 4, 4), dtype=np.int8)
        result = verifier.batch_grid_equal(preds, expected)
        assert result.shape == (2, 1)
        assert not np.any(result)

    def test_invalid_ndim_raises(self, verifier):
        """Non-3D arrays should raise ValueError."""
        with pytest.raises(ValueError):
            verifier.batch_grid_equal(np.zeros((2, 3), dtype=np.int8),
                                       np.zeros((1, 3, 3), dtype=np.int8))
        with pytest.raises(ValueError):
            verifier.batch_grid_equal(np.zeros((2, 3, 3), dtype=np.int8),
                                       np.zeros((3,), dtype=np.int8))


# ============================================================
# batch_grid_distance Tests
# ============================================================

class TestBatchGridDistance:
    """Tests for batch_grid_distance method."""

    def test_identical_grids_zero_distance(self, verifier, sample_batch):
        """Identical grids should have zero pixel distance."""
        preds = np.expand_dims(sample_batch[0], axis=0)
        expected = np.expand_dims(sample_batch[0], axis=0)
        result = verifier.batch_grid_distance(preds, expected)
        assert result[0, 0] == 0

    def test_known_distance(self, verifier):
        """Verify exact pixel distance for known grids."""
        pred = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        exp = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        result = verifier.batch_grid_distance(
            np.expand_dims(pred, 0), np.expand_dims(exp, 0)
        )
        assert result[0, 0] == 1  # 1 pixel differs

    def test_n_vs_m_matrix(self, verifier, sample_batch):
        """Should return (N, M) int32 matrix."""
        preds = sample_batch[:3]
        expected = sample_batch[:2]
        result = verifier.batch_grid_distance(preds, expected)
        assert result.shape == (3, 2)
        assert result.dtype == np.int32

    def test_shape_mismatch_returns_large(self, verifier):
        """Shape mismatch should return large distance values."""
        preds = np.zeros((2, 3, 3), dtype=np.int8)
        expected = np.zeros((1, 4, 4), dtype=np.int8)
        result = verifier.batch_grid_distance(preds, expected)
        assert result[0, 0] == 999999

    def test_invalid_ndim_raises(self, verifier):
        """Non-3D arrays should raise ValueError."""
        with pytest.raises(ValueError):
            verifier.batch_grid_distance(
                np.zeros((2, 3), dtype=np.int8),
                np.zeros((1, 3, 3), dtype=np.int8),
            )


# ============================================================
# batch_apply_mirror Tests
# ============================================================

class TestBatchApplyMirror:
    """Tests for batch_apply_mirror method."""

    def test_horizontal_mirror(self, verifier, sample_batch):
        """Horizontal mirror should match np.fliplr."""
        result = verifier.batch_apply_mirror(sample_batch, axis="horizontal")
        expected = np.array([np.fliplr(g) for g in sample_batch])
        np.testing.assert_array_equal(result, expected)

    def test_vertical_mirror(self, verifier, sample_batch):
        """Vertical mirror should match np.flipud."""
        result = verifier.batch_apply_mirror(sample_batch, axis="vertical")
        expected = np.array([np.flipud(g) for g in sample_batch])
        np.testing.assert_array_equal(result, expected)

    def test_single_grid_2d(self, verifier):
        """Should handle 2D single grid."""
        grid = np.array([[1, 0], [0, 2]], dtype=np.int8)
        result = verifier.batch_apply_mirror(grid, axis="horizontal")
        expected = np.fliplr(grid)
        np.testing.assert_array_equal(result, expected)

    def test_preserves_dtype(self, verifier, sample_batch):
        """Output should be int8."""
        result = verifier.batch_apply_mirror(sample_batch, axis="horizontal")
        assert result.dtype == np.int8


# ============================================================
# batch_apply_rotate Tests
# ============================================================

class TestBatchApplyRotate:
    """Tests for batch_apply_rotate method."""

    def test_rotate_90(self, verifier, sample_batch):
        """90-degree rotation should match np.rot90."""
        result = verifier.batch_apply_rotate(sample_batch, k=1)
        expected = np.array([np.rot90(g, k=1) for g in sample_batch])
        np.testing.assert_array_equal(result, expected)

    def test_rotate_180(self, verifier, sample_batch):
        """180-degree rotation should match np.rot90 k=2."""
        result = verifier.batch_apply_rotate(sample_batch, k=2)
        expected = np.array([np.rot90(g, k=2) for g in sample_batch])
        np.testing.assert_array_equal(result, expected)

    def test_rotate_360_identity(self, verifier, sample_batch):
        """360-degree rotation should be identity."""
        result = verifier.batch_apply_rotate(sample_batch, k=4)
        np.testing.assert_array_equal(result, sample_batch)

    def test_preserves_dtype(self, verifier, sample_batch):
        """Output should be int8."""
        result = verifier.batch_apply_rotate(sample_batch, k=1)
        assert result.dtype == np.int8


# ============================================================
# verify_candidates_batch Tests
# ============================================================

class TestVerifyCandidatesBatch:
    """Tests for verify_candidates_batch method."""

    def test_all_match(self, verifier, sample_batch):
        """All candidates matching should return all True."""
        preds = np.array([sample_batch[0], sample_batch[0]])
        expected = np.expand_dims(sample_batch[0], 0)
        result = verifier.verify_candidates_batch(preds, expected)
        assert result.shape == (2,)
        assert np.all(result)

    def test_none_match(self, verifier, sample_batch):
        """No candidates matching should return all False."""
        preds = np.array([sample_batch[0], sample_batch[0]])
        expected = np.expand_dims(sample_batch[1], 0)  # hflip
        result = verifier.verify_candidates_batch(preds, expected)
        assert not np.any(result)

    def test_partial_match(self, verifier, sample_batch):
        """Some candidates matching should return mixed results."""
        preds = np.array([sample_batch[0], sample_batch[1]])
        expected = np.expand_dims(sample_batch[0], 0)
        result = verifier.verify_candidates_batch(preds, expected)
        assert result[0] == True  # noqa: E712
        assert result[1] == False  # noqa: E712

    def test_multi_expected_all_must_match(self, verifier, sample_batch):
        """Candidate must match ALL expected outputs."""
        preds = np.array([sample_batch[0]])  # (1, 3, 3)
        expected = np.array([sample_batch[0], sample_batch[0]])  # (2, 3, 3)
        result = verifier.verify_candidates_batch(preds, expected)
        assert result[0] == True  # noqa: E712

        # Now with one mismatch
        expected_mixed = np.array([sample_batch[0], sample_batch[1]])
        result_mixed = verifier.verify_candidates_batch(preds, expected_mixed)
        assert result_mixed[0] == False  # noqa: E712


# ============================================================
# auto_batch_size and get_memory_info Tests
# ============================================================

class TestUtilityMethods:
    """Tests for auto_batch_size and get_memory_info."""

    def test_auto_batch_size_no_gpu(self, verifier):
        """Without GPU, should return configured batch_size."""
        if not HAS_CUDA:
            result = verifier.auto_batch_size((3, 3))
            assert result == verifier.batch_size

    def test_get_memory_info_structure(self, verifier):
        """get_memory_info should return dict with required keys."""
        info = verifier.get_memory_info()
        assert "free_mb" in info
        assert "total_mb" in info
        assert "used_mb" in info
        if not HAS_CUDA:
            assert info["free_mb"] == 0.0
            assert info["total_mb"] == 0.0

    def test_auto_batch_size_positive(self, verifier):
        """auto_batch_size should always return >= 1."""
        result = verifier.auto_batch_size((100, 100))
        assert result >= 1
