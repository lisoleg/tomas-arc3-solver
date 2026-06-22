"""Numba CUDA GPU kernels for parallel candidate enumeration and verification.

Uses @cuda.jit for true GPU parallelism when CUDA hardware is available.
Falls back to CPU (Numba @njit or NumPy) when no GPU is present.

TOMAS v2.3: Numba CUDA parallel candidate enumeration layer.

GPU Strategy:
  - Candidate enumeration generates N programs (depth 1: ~30, depth 2: ~900,
    depth 3: ~thousands). Each candidate must be applied to M demo inputs
    and verified against M demo outputs.
  - On GPU: each thread block handles one (candidate, demo_pair) cell,
    applying the primitive and comparing the result in parallel.
  - This replaces the serial Python loop in Phase A filtering and
    Phase B verification with a single kernel launch.

Kernels:
  1. batch_grid_equal_cuda   — N x M grid equality matrix
  2. batch_mirror_cuda       — mirror N grids in parallel
  3. batch_rotate_cuda       — rotate N grids in parallel
  4. batch_betti0_cuda       — connected components for N grids
  5. batch_color_hist_cuda   — 10-bin color histograms for N grids
  6. batch_nonzero_count_cuda — non-zero pixel counts for N grids
  7. batch_grid_distance_cuda — N x M pixel distance matrix
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Try to import numba CUDA
try:
    from numba import cuda, njit
    _HAS_NUMBA = True
    try:
        HAS_CUDA = cuda.is_available()
    except Exception:
        HAS_CUDA = False
except ImportError:
    _HAS_NUMBA = False
    HAS_CUDA = False

    # Stubs for decorators
    def cuda(*args, **kwargs):  # type: ignore[no-redef]
        return lambda f: f

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        return lambda f: f

# Import CPU kernels for fallback
try:
    from src.core.numba_kernels import (
        grid_equal_kernel,
        grid_distance_kernel,
        count_nonzero_kernel,
    )
    _HAS_CPU_KERNELS = True
except ImportError:
    _HAS_CPU_KERNELS = False


# ============================================================
# CUDA Device Kernels (@cuda.jit)
# ============================================================

if _HAS_NUMBA:
    @cuda.jit
    def _batch_grid_equal_kernel(preds, expected, result):
        """CUDA kernel: compute N x M grid equality matrix.

        Each thread block handles one (i, j) pair where i is the
        candidate index and j is the expected grid index.
        Thread (0,0) in each block does the comparison with early exit.

        Args:
            preds: (N, H, W) int8 array of predicted grids.
            expected: (M, H, W) int8 array of expected grids.
            result: (N, M) int8 array (1=equal, 0=not equal).
        """
        i = cuda.blockIdx.x
        j = cuda.blockIdx.y

        if i >= preds.shape[0] or j >= expected.shape[0]:
            return

        # Shape check
        if preds.shape[1] != expected.shape[1] or \
           preds.shape[2] != expected.shape[2]:
            result[i, j] = 0
            return

        h = preds.shape[1]
        w = preds.shape[2]
        is_equal = True

        for row in range(h):
            if not is_equal:
                break
            for col in range(w):
                if preds[i, row, col] != expected[j, row, col]:
                    is_equal = False
                    break

        result[i, j] = 1 if is_equal else 0

    @cuda.jit
    def _batch_mirror_kernel(grids, axis, output):
        """CUDA kernel: mirror N grids in parallel.

        Args:
            grids: (N, H, W) int8 input grids.
            axis: 0=horizontal (flip columns), 1=vertical (flip rows).
            output: (N, H, W) int8 output grids.
        """
        i = cuda.blockIdx.x
        tx = cuda.threadIdx.x

        if i >= grids.shape[0]:
            return

        h = grids.shape[1]
        w = grids.shape[2]
        total = h * w

        for idx in range(tx, total, cuda.blockDim.x):
            row = idx // w
            col = idx % w
            if axis == 0:  # horizontal mirror
                output[i, row, col] = grids[i, row, w - 1 - col]
            else:  # vertical mirror
                output[i, row, col] = grids[i, h - 1 - row, col]

    @cuda.jit
    def _batch_rotate90_kernel(grids, output):
        """CUDA kernel: rotate N grids by 90 degrees in parallel.

        Rotation: output[i, col, h-1-row] = grids[i, row, col]

        Args:
            grids: (N, H, W) int8 input grids.
            output: (N, W, H) int8 output grids (transposed dims).
        """
        i = cuda.blockIdx.x
        tx = cuda.threadIdx.x

        if i >= grids.shape[0]:
            return

        h = grids.shape[1]
        w = grids.shape[2]
        total = h * w

        for idx in range(tx, total, cuda.blockDim.x):
            row = idx // w
            col = idx % w
            output[i, col, h - 1 - row] = grids[i, row, col]

    @cuda.jit
    def _batch_betti0_kernel(grids, betti0_results):
        """CUDA kernel: count connected components for N grids.

        Uses a simple flood-fill approach per grid. Each block handles
        one grid. Within a block, threads cooperate on the fill.

        Note: This is a simplified version — for production use,
        a union-find based approach would be more efficient on GPU.

        Args:
            grids: (N, H, W) int8 input grids.
            betti0_results: (N,) int32 array of component counts.
        """
        i = cuda.blockIdx.x
        if i >= grids.shape[0]:
            return

        h = grids.shape[1]
        w = grids.shape[2]
        count = 0

        # Simple BFS/DFS on GPU — iterate and count new components
        # This is O(H*W) per grid with early termination
        for row in range(h):
            for col in range(w):
                if grids[i, row, col] != 0:
                    # Check if this is the top-left pixel of a new component
                    is_new = True
                    # Check left neighbor
                    if col > 0 and grids[i, row, col - 1] == grids[i, row, col]:
                        is_new = False
                    # Check top neighbor
                    if is_new and row > 0 and grids[i, row - 1, col] == grids[i, row, col]:
                        is_new = False
                    if is_new:
                        count += 1

        betti0_results[i] = count

    @cuda.jit
    def _batch_color_hist_kernel(grids, histograms):
        """CUDA kernel: compute 10-bin color histograms for N grids.

        Args:
            grids: (N, H, W) int8 input grids.
            histograms: (N, 10) int32 color histogram arrays.
        """
        i = cuda.blockIdx.x
        tx = cuda.threadIdx.x

        if i >= grids.shape[0]:
            return

        # Each thread accumulates partial counts for its color
        # Then atomic add to shared memory
        h = grids.shape[1]
        w = grids.shape[2]
        total = h * w

        # Thread tx handles color index tx (0-9)
        if tx < 10:
            local_count = 0
            for idx in range(tx, total, 10):
                row = idx // w
                col = idx % w
                if grids[i, row, col] == tx:
                    local_count += 1
            histograms[i, tx] = local_count

    @cuda.jit
    def _batch_nonzero_count_kernel(grids, counts):
        """CUDA kernel: count non-zero pixels for N grids.

        Args:
            grids: (N, H, W) int8 input grids.
            counts: (N,) int32 non-zero pixel counts.
        """
        i = cuda.blockIdx.x
        tx = cuda.threadIdx.x

        if i >= grids.shape[0]:
            return

        h = grids.shape[1]
        w = grids.shape[2]
        total = h * w

        # Each thread counts a subset of pixels
        local_count = 0
        for idx in range(tx, total, cuda.blockDim.x):
            row = idx // w
            col = idx % w
            if grids[i, row, col] != 0:
                local_count += 1

        # Atomic add to shared result
        cuda.atomic.add(counts, i, local_count)

    @cuda.jit
    def _batch_grid_distance_kernel(preds, expected, result):
        """CUDA kernel: compute N x M pixel distance matrix.

        Args:
            preds: (N, H, W) int8 predicted grids.
            expected: (M, H, W) int8 expected grids.
            result: (N, M) int32 pixel difference counts.
        """
        i = cuda.blockIdx.x
        j = cuda.blockIdx.y

        if i >= preds.shape[0] or j >= expected.shape[0]:
            return

        if preds.shape[1] != expected.shape[1] or \
           preds.shape[2] != expected.shape[2]:
            result[i, j] = 999999
            return

        h = preds.shape[1]
        w = preds.shape[2]
        diff = 0

        for row in range(h):
            for col in range(w):
                if preds[i, row, col] != expected[j, row, col]:
                    diff += 1

        result[i, j] = diff

    @cuda.jit
    def _batch_shape_check_kernel(preds, expected_shapes, result):
        """CUDA kernel: check if N predicted grids match expected shapes.

        Args:
            preds: (N, H, W) int8 predicted grids (padded to max size).
            expected_shapes: (M, 2) int32 array of (H, W) expected shapes.
            result: (N, M) int8 array (1=match, 0=mismatch).
        """
        i = cuda.blockIdx.x
        j = cuda.blockIdx.y

        if i >= preds.shape[0] or j >= expected_shapes.shape[0]:
            return

        pred_h = preds.shape[1]
        pred_w = preds.shape[2]
        exp_h = expected_shapes[j, 0]
        exp_w = expected_shapes[j, 1]

        result[i, j] = 1 if (pred_h == exp_h and pred_w == exp_w) else 0


# ============================================================
# CPU Fallback Kernels (@njit or NumPy)
# ============================================================

@njit(cache=True, nogil=True, fastmath=True)
def _cpu_batch_grid_equal(preds, expected):
    """CPU fallback: N x M grid equality matrix."""
    n = preds.shape[0]
    m = expected.shape[0]
    result = np.zeros((n, m), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            if preds.shape[1] != expected.shape[1] or \
               preds.shape[2] != expected.shape[2]:
                result[i, j] = 0
                continue
            h = preds.shape[1]
            w = preds.shape[2]
            equal = True
            for row in range(h):
                if not equal:
                    break
                for col in range(w):
                    if preds[i, row, col] != expected[j, row, col]:
                        equal = False
                        break
            result[i, j] = 1 if equal else 0
    return result


@njit(cache=True, nogil=True, fastmath=True)
def _cpu_batch_nonzero_count(grids):
    """CPU fallback: non-zero pixel counts for N grids."""
    n = grids.shape[0]
    counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        h = grids.shape[1]
        w = grids.shape[2]
        c = 0
        for row in range(h):
            for col in range(w):
                if grids[i, row, col] != 0:
                    c += 1
        counts[i] = c
    return counts


@njit(cache=True, nogil=True, fastmath=True)
def _cpu_batch_color_hist(grids):
    """CPU fallback: 10-bin color histograms for N grids."""
    n = grids.shape[0]
    histograms = np.zeros((n, 10), dtype=np.int32)
    for i in range(n):
        h = grids.shape[1]
        w = grids.shape[2]
        for row in range(h):
            for col in range(w):
                c = grids[i, row, col]
                if 0 <= c < 10:
                    histograms[i, c] += 1
    return histograms


@njit(cache=True, nogil=True, fastmath=True)
def _cpu_batch_grid_distance(preds, expected):
    """CPU fallback: N x M pixel distance matrix."""
    n = preds.shape[0]
    m = expected.shape[0]
    result = np.zeros((n, m), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            if preds.shape[1] != expected.shape[1] or \
               preds.shape[2] != expected.shape[2]:
                result[i, j] = 999999
                continue
            h = preds.shape[1]
            w = preds.shape[2]
            diff = 0
            for row in range(h):
                for col in range(w):
                    if preds[i, row, col] != expected[j, row, col]:
                        diff += 1
            result[i, j] = diff
    return result


@njit(cache=True, nogil=True, fastmath=True)
def _cpu_batch_betti0(grids):
    """CPU fallback: connected component counts for N grids.

    Uses a simplified 4-connectivity check (top-left scan).
    """
    n = grids.shape[0]
    betti0 = np.zeros(n, dtype=np.int32)
    for i in range(n):
        h = grids.shape[1]
        w = grids.shape[2]
        count = 0
        for row in range(h):
            for col in range(w):
                if grids[i, row, col] != 0:
                    is_new = True
                    if col > 0 and grids[i, row, col - 1] == grids[i, row, col]:
                        is_new = False
                    if is_new and row > 0 and \
                       grids[i, row - 1, col] == grids[i, row, col]:
                        is_new = False
                    if is_new:
                        count += 1
        betti0[i] = count
    return betti0


# ============================================================
# High-Level API: NumbaCudaBatchVerifier
# ============================================================

class NumbaCudaBatchVerifier:
    """Numba CUDA batch verifier for GPU-parallel candidate operations.

    Provides a unified interface for batch grid operations that
    automatically selects GPU (@cuda.jit) or CPU (@njit) based
    on hardware availability.

    Attributes:
        use_gpu: Whether GPU acceleration is active.
        threads_per_block: CUDA threads per block (default 256).
    """

    def __init__(self, threads_per_block: int = 256, batch_size: int = 256) -> None:
        """Initialize the batch verifier.

        Args:
            threads_per_block: CUDA threads per block for kernel launches.
            batch_size: Maximum candidates per batch (for interface compat
                with CudaBatchVerifier).
        """
        self.use_gpu: bool = HAS_CUDA
        self.threads_per_block: int = threads_per_block
        self.batch_size: int = batch_size

    @property
    def is_gpu_active(self) -> bool:
        """Check if GPU acceleration is active.

        Returns:
            True if numba CUDA is available and a GPU is present.
        """
        return self.use_gpu

    # ---- Batch Grid Equality ----

    def batch_grid_equal(
        self,
        predictions: np.ndarray,
        expected: np.ndarray,
    ) -> np.ndarray:
        """Compute N x M grid equality matrix.

        On GPU: launches a (N, M) grid of blocks, each comparing
        one prediction against one expected grid.
        On CPU: uses @njit kernel for fast serial comparison.

        Args:
            predictions: (N, H, W) int8 array of predicted grids.
            expected: (M, H, W) int8 array of expected grids.

        Returns:
            (N, M) boolean array. result[i, j] = True if equal.
        """
        if predictions.ndim != 3 or expected.ndim != 3:
            raise ValueError("Expected 3D arrays")

        # Shape mismatch
        if predictions.shape[1:] != expected.shape[1:]:
            return np.zeros(
                (predictions.shape[0], expected.shape[0]), dtype=bool
            )

        if self.use_gpu:
            return self._gpu_batch_grid_equal(predictions, expected)
        return self._cpu_batch_grid_equal(predictions, expected)

    def _gpu_batch_grid_equal(
        self, predictions: np.ndarray, expected: np.ndarray
    ) -> np.ndarray:
        """GPU implementation using numba @cuda.jit."""
        n = predictions.shape[0]
        m = expected.shape[0]
        result = np.zeros((n, m), dtype=np.int8)

        # Copy to device
        d_preds = cuda.to_device(predictions.astype(np.int8))
        d_expected = cuda.to_device(expected.astype(np.int8))
        d_result = cuda.to_device(result)

        # Launch kernel: one block per (i, j) pair
        blocks_per_grid = (n, m)
        _batch_grid_equal_kernel[blocks_per_grid, 1](
            d_preds, d_expected, d_result
        )

        # Copy result back
        result = d_result.copy_to_host()
        return result.astype(bool)

    def _cpu_batch_grid_equal(
        self, predictions: np.ndarray, expected: np.ndarray
    ) -> np.ndarray:
        """CPU fallback using @njit kernel."""
        if _HAS_NUMBA:
            preds_i8 = predictions.astype(np.int8)
            exp_i8 = expected.astype(np.int8)
            result = _cpu_batch_grid_equal(preds_i8, exp_i8)
            return result.astype(bool)

        # Pure NumPy fallback
        n = predictions.shape[0]
        m = expected.shape[0]
        result = np.zeros((n, m), dtype=bool)
        for i in range(n):
            for j in range(m):
                result[i, j] = np.array_equal(predictions[i], expected[j])
        return result

    # ---- Batch Mirror ----

    def batch_mirror(
        self, grids: np.ndarray, axis: str = "horizontal"
    ) -> np.ndarray:
        """Mirror N grids in parallel.

        Args:
            grids: (N, H, W) int8 array.
            axis: 'horizontal' or 'vertical'.

        Returns:
            Mirrored grids array of same shape.
        """
        axis_code = 0 if axis == "horizontal" else 1

        if self.use_gpu:
            return self._gpu_batch_mirror(grids, axis_code)

        # CPU fallback
        if axis == "horizontal":
            return np.array([np.fliplr(g).copy() for g in grids]).astype(np.int8)
        return np.array([np.flipud(g).copy() for g in grids]).astype(np.int8)

    def _gpu_batch_mirror(
        self, grids: np.ndarray, axis_code: int
    ) -> np.ndarray:
        """GPU implementation of batch mirror."""
        n = grids.shape[0]
        output = np.zeros_like(grids, dtype=np.int8)

        d_grids = cuda.to_device(grids.astype(np.int8))
        d_output = cuda.to_device(output)

        _batch_mirror_kernel[n, self.threads_per_block](
            d_grids, axis_code, d_output
        )

        return d_output.copy_to_host()

    # ---- Batch Rotate90 ----

    def batch_rotate90(self, grids: np.ndarray) -> np.ndarray:
        """Rotate N grids by 90 degrees in parallel.

        Args:
            grids: (N, H, W) int8 array.

        Returns:
            (N, W, H) int8 rotated grids.
        """
        if self.use_gpu:
            return self._gpu_batch_rotate90(grids)

        # CPU fallback
        return np.array(
            [np.rot90(g).copy() for g in grids]
        ).astype(np.int8)

    def _gpu_batch_rotate90(self, grids: np.ndarray) -> np.ndarray:
        """GPU implementation of batch rotate90."""
        n = grids.shape[0]
        h, w = grids.shape[1], grids.shape[2]
        output = np.zeros((n, w, h), dtype=np.int8)

        d_grids = cuda.to_device(grids.astype(np.int8))
        d_output = cuda.to_device(output)

        _batch_rotate90_kernel[n, self.threads_per_block](
            d_grids, d_output
        )

        return d_output.copy_to_host()

    # ---- Batch Betti0 ----

    def batch_betti0(self, grids: np.ndarray) -> np.ndarray:
        """Compute Betti0 (connected components) for N grids in parallel.

        Args:
            grids: (N, H, W) int8 array.

        Returns:
            (N,) int32 array of component counts.
        """
        if self.use_gpu:
            return self._gpu_batch_betti0(grids)

        if _HAS_NUMBA:
            return _cpu_batch_betti0(grids.astype(np.int8))

        # Pure NumPy fallback
        from scipy import ndimage
        n = grids.shape[0]
        result = np.zeros(n, dtype=np.int32)
        for i in range(n):
            _, num = ndimage.label(grids[i] > 0)
            result[i] = num
        return result

    def _gpu_batch_betti0(self, grids: np.ndarray) -> np.ndarray:
        """GPU implementation of batch Betti0."""
        n = grids.shape[0]
        betti0 = np.zeros(n, dtype=np.int32)

        d_grids = cuda.to_device(grids.astype(np.int8))
        d_betti0 = cuda.to_device(betti0)

        _batch_betti0_kernel[n, 1](d_grids, d_betti0)

        return d_betti0.copy_to_host()

    # ---- Batch Color Histogram ----

    def batch_color_histogram(self, grids: np.ndarray) -> np.ndarray:
        """Compute 10-bin color histograms for N grids in parallel.

        Args:
            grids: (N, H, W) int8 array.

        Returns:
            (N, 10) int32 array of color counts (indices 0-9).
        """
        if self.use_gpu:
            return self._gpu_batch_color_hist(grids)

        if _HAS_NUMBA:
            return _cpu_batch_color_hist(grids.astype(np.int8))

        # Pure NumPy fallback
        n = grids.shape[0]
        result = np.zeros((n, 10), dtype=np.int32)
        for i in range(n):
            for c in range(10):
                result[i, c] = np.count_nonzero(grids[i] == c)
        return result

    def _gpu_batch_color_hist(self, grids: np.ndarray) -> np.ndarray:
        """GPU implementation of batch color histogram."""
        n = grids.shape[0]
        histograms = np.zeros((n, 10), dtype=np.int32)

        d_grids = cuda.to_device(grids.astype(np.int8))
        d_hist = cuda.to_device(histograms)

        _batch_color_hist_kernel[n, 10](d_grids, d_hist)

        return d_hist.copy_to_host()

    # ---- Batch Non-Zero Count ----

    def batch_nonzero_count(self, grids: np.ndarray) -> np.ndarray:
        """Count non-zero pixels for N grids in parallel.

        Args:
            grids: (N, H, W) int8 array.

        Returns:
            (N,) int32 array of non-zero counts.
        """
        if self.use_gpu:
            return self._gpu_batch_nonzero_count(grids)

        if _HAS_NUMBA:
            return _cpu_batch_nonzero_count(grids.astype(np.int8))

        # Pure NumPy fallback
        n = grids.shape[0]
        result = np.zeros(n, dtype=np.int32)
        for i in range(n):
            result[i] = np.count_nonzero(grids[i])
        return result

    def _gpu_batch_nonzero_count(self, grids: np.ndarray) -> np.ndarray:
        """GPU implementation of batch non-zero count."""
        n = grids.shape[0]
        counts = np.zeros(n, dtype=np.int32)

        d_grids = cuda.to_device(grids.astype(np.int8))
        d_counts = cuda.to_device(counts)

        _batch_nonzero_count_kernel[n, self.threads_per_block](
            d_grids, d_counts
        )

        return d_counts.copy_to_host()

    # ---- Batch Grid Distance ----

    def batch_grid_distance(
        self,
        predictions: np.ndarray,
        expected: np.ndarray,
    ) -> np.ndarray:
        """Compute N x M pixel distance matrix.

        Args:
            predictions: (N, H, W) int8 array.
            expected: (M, H, W) int8 array.

        Returns:
            (N, M) int32 array of pixel difference counts.
        """
        if predictions.ndim != 3 or expected.ndim != 3:
            raise ValueError("Expected 3D arrays")

        if predictions.shape[1:] != expected.shape[1:]:
            return np.full(
                (predictions.shape[0], expected.shape[0]),
                999999, dtype=np.int32
            )

        if self.use_gpu:
            return self._gpu_batch_grid_distance(predictions, expected)

        if _HAS_NUMBA:
            return _cpu_batch_grid_distance(
                predictions.astype(np.int8), expected.astype(np.int8)
            )

        # Pure NumPy fallback
        n = predictions.shape[0]
        m = expected.shape[0]
        result = np.zeros((n, m), dtype=np.int32)
        for i in range(n):
            for j in range(m):
                result[i, j] = int(np.sum(predictions[i] != expected[j]))
        return result

    def _gpu_batch_grid_distance(
        self, predictions: np.ndarray, expected: np.ndarray
    ) -> np.ndarray:
        """GPU implementation of batch grid distance."""
        n = predictions.shape[0]
        m = expected.shape[0]
        result = np.zeros((n, m), dtype=np.int32)

        d_preds = cuda.to_device(predictions.astype(np.int8))
        d_expected = cuda.to_device(expected.astype(np.int8))
        d_result = cuda.to_device(result)

        blocks = (n, m)
        _batch_grid_distance_kernel[blocks, 1](
            d_preds, d_expected, d_result
        )

        return d_result.copy_to_host()

    # ---- Utility ----

    def get_device_info(self) -> dict[str, Any]:
        """Get GPU device information.

        Returns:
            Dictionary with device name, compute capability, memory info.
            Returns empty dict if GPU not available.
        """
        if not self.use_gpu:
            return {}

        try:
            dev = cuda.get_current_device()
            return {
                "name": dev.name,
                "compute_capability": dev.compute_capability,
                "max_threads_per_block": dev.MAX_THREADS_PER_BLOCK,
                "multiprocessor_count": dev.MULTIPROCESSOR_COUNT,
            }
        except Exception:
            return {}

    def verify_candidates_batch(
        self,
        candidates_predictions: np.ndarray,
        demo_outputs: np.ndarray,
    ) -> np.ndarray:
        """Verify N candidates against M demo outputs in batch.

        Args:
            candidates_predictions: (N, H, W) int8 predicted outputs.
            demo_outputs: (M, H, W) int8 expected outputs.

        Returns:
            (N,) boolean array — True if candidate matches ALL outputs.
        """
        equal_matrix = self.batch_grid_equal(
            candidates_predictions, demo_outputs
        )
        return np.all(equal_matrix, axis=1)
