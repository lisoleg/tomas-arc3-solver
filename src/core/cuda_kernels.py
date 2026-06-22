"""CUDA GPU kernels for batch grid operations via CuPy.

Provides GPU-accelerated batch verification for the kappa-Snap search
Phase B. Falls back gracefully to CPU (Numba/NumPy) when CuPy or CUDA
is unavailable.

TOMAS v2.3: GPU batch acceleration layer.
"""
from __future__ import annotations

import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    _HAS_CUPY = True
    # Verify CUDA is actually available
    try:
        cp.cuda.Device(0).compute_capability
        HAS_CUDA = True
    except Exception:
        HAS_CUDA = False
        cp = None  # type: ignore[assignment]
except ImportError:
    _HAS_CUPY = False
    HAS_CUDA = False
    cp = None  # type: ignore[assignment]

# Import Numba kernels for fallback
try:
    from src.core.numba_kernels import (
        HAS_NUMBA,
        grid_equal_kernel,
        grid_distance_kernel,
    )
except ImportError:
    HAS_NUMBA = False


class CudaBatchVerifier:
    """GPU batch verifier for parallel candidate verification.

    Enables verification of N candidate programs against M demo pairs
    in a single GPU kernel launch, replacing the serial ThreadPoolExecutor
    approach in v2.2.

    Attributes:
        batch_size: Maximum number of candidates per GPU batch.
        device: CuPy device string.
    """

    def __init__(self, batch_size: int = 256, device: int = 0) -> None:
        """Initialize the CUDA batch verifier.

        Args:
            batch_size: Maximum candidates per GPU batch.
            device: CUDA device index.
        """
        self.batch_size = batch_size
        self.device = device
        self._available = HAS_CUDA

    @property
    def is_available(self) -> bool:
        """Check if GPU acceleration is available.

        Returns:
            True if CuPy + CUDA GPU is available.
        """
        return self._available

    def batch_grid_equal(
        self,
        predictions: np.ndarray,
        expected: np.ndarray,
    ) -> np.ndarray:
        """Batch grid equality check on GPU.

        Compares N prediction grids against M expected grids, returning
        an (N, M) boolean matrix. On GPU, this is a single parallel
        operation; on CPU fallback, uses Numba serial comparison.

        Args:
            predictions: Array of shape (N, H, W) int8 — N predicted grids.
            expected: Array of shape (M, H, W) int8 — M expected grids.

        Returns:
            Boolean array of shape (N, M). result[i, j] = True if
            predictions[i] equals expected[j].

        Raises:
            RuntimeError: If predictions and expected have incompatible shapes.
        """
        if predictions.ndim != 3 or expected.ndim != 3:
            raise ValueError(
                f"Expected 3D arrays, got predictions.ndim={predictions.ndim}, "
                f"expected.ndim={expected.ndim}"
            )

        n_preds = predictions.shape[0]
        n_expected = expected.shape[0]

        # Shape mismatch check (H, W must match)
        if predictions.shape[1:] != expected.shape[1:]:
            # Return all False — shapes don't match
            return np.zeros((n_preds, n_expected), dtype=bool)

        if self._available:
            return self._batch_grid_equal_gpu(predictions, expected)
        return self._batch_grid_equal_cpu(predictions, expected)

    def _batch_grid_equal_gpu(
        self,
        predictions: np.ndarray,
        expected: np.ndarray,
    ) -> np.ndarray:
        """GPU implementation using CuPy.

        Uploads both arrays to GPU, performs element-wise comparison
        across all pairs, and reduces to boolean result.

        Args:
            predictions: (N, H, W) int8 array.
            expected: (M, H, W) int8 array.

        Returns:
            (N, M) boolean array.
        """
        n_preds = predictions.shape[0]
        n_expected = expected.shape[0]

        # Upload to GPU
        preds_gpu = cp.asarray(predictions)
        exp_gpu = cp.asarray(expected)

        # Reshape for broadcasting: (N, 1, H, W) vs (1, M, H, W)
        preds_gpu = preds_gpu[:, cp.newaxis, :, :]
        exp_gpu = exp_gpu[cp.newaxis, :, :, :]

        # Element-wise equality, then all-reduce over spatial dims
        equal = (preds_gpu == exp_gpu)
        result = cp.all(equal, axis=(2, 3))

        # Download result
        return cp.asnumpy(result).astype(bool)

    def _batch_grid_equal_cpu(
        self,
        predictions: np.ndarray,
        expected: np.ndarray,
    ) -> np.ndarray:
        """CPU fallback using Numba or NumPy.

        Args:
            predictions: (N, H, W) int8 array.
            expected: (M, H, W) int8 array.

        Returns:
            (N, M) boolean array.
        """
        n_preds = predictions.shape[0]
        n_expected = expected.shape[0]
        result = np.zeros((n_preds, n_expected), dtype=bool)

        if HAS_NUMBA:
            for i in range(n_preds):
                for j in range(n_expected):
                    result[i, j] = grid_equal_kernel(
                        predictions[i], expected[j]
                    )
        else:
            for i in range(n_preds):
                for j in range(n_expected):
                    result[i, j] = np.array_equal(
                        predictions[i], expected[j]
                    )

        return result

    def batch_grid_distance(
        self,
        predictions: np.ndarray,
        expected: np.ndarray,
    ) -> np.ndarray:
        """Batch grid pixel-distance computation on GPU.

        Computes the number of differing pixels between each prediction
        and each expected grid.

        Args:
            predictions: Array of shape (N, H, W) int8.
            expected: Array of shape (M, H, W) int8.

        Returns:
            Int32 array of shape (N, M) with pixel difference counts.
        """
        if predictions.ndim != 3 or expected.ndim != 3:
            raise ValueError("Expected 3D arrays for batch_grid_distance")

        if predictions.shape[1:] != expected.shape[1:]:
            return np.full(
                (predictions.shape[0], expected.shape[0]),
                999999,
                dtype=np.int32,
            )

        if self._available:
            return self._batch_grid_distance_gpu(predictions, expected)
        return self._batch_grid_distance_cpu(predictions, expected)

    def _batch_grid_distance_gpu(
        self,
        predictions: np.ndarray,
        expected: np.ndarray,
    ) -> np.ndarray:
        """GPU implementation of batch grid distance.

        Args:
            predictions: (N, H, W) int8 array.
            expected: (M, H, W) int8 array.

        Returns:
            (N, M) int32 array of pixel difference counts.
        """
        preds_gpu = cp.asarray(predictions)
        exp_gpu = cp.asarray(expected)

        # Broadcast: (N, 1, H, W) vs (1, M, H, W)
        preds_gpu = preds_gpu[:, cp.newaxis, :, :]
        exp_gpu = exp_gpu[cp.newaxis, :, :, :]

        # Count differing pixels
        diff = (preds_gpu != exp_gpu).astype(cp.int32)
        result = cp.sum(diff, axis=(2, 3))

        return cp.asnumpy(result).astype(np.int32)

    def _batch_grid_distance_cpu(
        self,
        predictions: np.ndarray,
        expected: np.ndarray,
    ) -> np.ndarray:
        """CPU fallback for batch grid distance.

        Args:
            predictions: (N, H, W) int8 array.
            expected: (M, H, W) int8 array.

        Returns:
            (N, M) int32 array.
        """
        n_preds = predictions.shape[0]
        n_expected = expected.shape[0]
        result = np.zeros((n_preds, n_expected), dtype=np.int32)

        if HAS_NUMBA:
            for i in range(n_preds):
                for j in range(n_expected):
                    result[i, j] = grid_distance_kernel(
                        predictions[i], expected[j]
                    )
        else:
            for i in range(n_preds):
                for j in range(n_expected):
                    result[i, j] = int(
                        np.sum(predictions[i] != expected[j])
                    )

        return result

    def batch_apply_mirror(
        self,
        grids: np.ndarray,
        axis: str = "horizontal",
    ) -> np.ndarray:
        """Batch horizontal/vertical mirror on GPU.

        Args:
            grids: Array of shape (N, H, W) int8.
            axis: 'horizontal' or 'vertical'.

        Returns:
            Mirrored grids array of same shape.
        """
        if self._available:
            grids_gpu = cp.asarray(grids)
            if axis == "horizontal":
                result = cp.fliplr(grids_gpu)
            else:
                result = cp.flipud(grids_gpu)
            return cp.asnumpy(result).astype(np.int8)

        # CPU fallback
        if axis == "horizontal":
            return np.fliplr(grids).copy() if grids.ndim == 2 else \
                np.array([np.fliplr(g).copy() for g in grids])
        return np.flipud(grids).copy() if grids.ndim == 2 else \
            np.array([np.flipud(g).copy() for g in grids])

    def batch_apply_rotate(
        self,
        grids: np.ndarray,
        k: int = 1,
    ) -> np.ndarray:
        """Batch rotation by 90*k degrees on GPU.

        Args:
            grids: Array of shape (N, H, W) int8.
            k: Number of 90-degree rotations.

        Returns:
            Rotated grids array.
        """
        if self._available:
            grids_gpu = cp.asarray(grids)
            result = cp.rot90(grids_gpu, k=k, axes=(1, 2))
            return cp.asnumpy(result).astype(np.int8)

        # CPU fallback
        return np.array([np.rot90(g, k=k).copy() for g in grids]).astype(np.int8)

    def verify_candidates_batch(
        self,
        candidates_predictions: np.ndarray,
        demo_outputs: np.ndarray,
    ) -> np.ndarray:
        """Verify candidates against demo outputs in batch.

        Given N candidates' predicted outputs for a single demo pair,
        returns a boolean mask of which candidates match.

        Args:
            candidates_predictions: (N, H, W) int8 — predicted outputs.
            demo_outputs: (M, H, W) int8 — expected outputs (usually M=1
                for a single demo pair, but supports multi-frame).

        Returns:
            Boolean array of shape (N,) — True if candidate matches
            ALL expected outputs for this demo pair.
        """
        # batch_grid_equal returns (N, M) — candidate must match all M
        equal_matrix = self.batch_grid_equal(
            candidates_predictions, demo_outputs
        )
        # A candidate passes if it matches ALL expected outputs
        return np.all(equal_matrix, axis=1)

    def get_memory_info(self) -> dict[str, float]:
        """Get GPU memory information.

        Returns:
            Dictionary with free_mb and total_mb, or zeros if unavailable.
        """
        if not self._available:
            return {"free_mb": 0.0, "total_mb": 0.0, "used_mb": 0.0}

        try:
            mem = cp.cuda.Device(self.device).mem_info
            free = mem[0] / (1024 ** 2)
            total = mem[1] / (1024 ** 2)
            return {
                "free_mb": round(free, 1),
                "total_mb": round(total, 1),
                "used_mb": round(total - free, 1),
            }
        except Exception:
            return {"free_mb": 0.0, "total_mb": 0.0, "used_mb": 0.0}

    def auto_batch_size(self, grid_shape: tuple[int, int]) -> int:
        """Determine optimal batch size based on available VRAM.

        Args:
            grid_shape: (H, W) of grids to be batched.

        Returns:
            Optimal batch size that fits in available VRAM.
        """
        if not self._available:
            return self.batch_size

        mem_info = self.get_memory_info()
        free_bytes = mem_info["free_mb"] * (1024 ** 2)

        # Each grid is H*W bytes (int8), plus comparison intermediate
        # needs N*M*H*W bytes. Assume M=3 (demo pairs).
        h, w = grid_shape
        bytes_per_grid = h * w
        # Need space for predictions (N*H*W) + expected (M*H*W) +
        # comparison result (N*M*H*W)
        # Solve: N * H * W * (1 + M) <= free_bytes * 0.7 (leave 30% headroom)
        m = 3  # Assume 3 demo pairs
        usable = free_bytes * 0.7
        n_max = int(usable / (bytes_per_grid * (1 + m)))

        return max(1, min(n_max, self.batch_size))
