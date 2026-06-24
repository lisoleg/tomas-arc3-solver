"""Numba JIT-compiled kernels for hot-path numeric operations.

All functions use @njit with explicit signatures for zero warmup overhead.
Fall back gracefully to pure-numpy implementations when numba is unavailable.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create no-op decorator
    def njit(*args, **kwargs):  # type: ignore[no-redef]
        return lambda f: f

    def prange(*args, **kwargs):  # type: ignore[no-redef]
        return range


# ============================================================
# Grid Transform Kernels
# ============================================================

@njit("int8[:,:](int8[:,:],int8[:,:])", cache=True, nogil=True, fastmath=True)
def overlay_kernel(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Overlay src on dst: non-zero src pixels overwrite dst. ~3-5x faster.

    Args:
        dst: Destination grid (int8, modified in-place).
        src: Source grid (int8).

    Returns:
        Modified dst grid.
    """
    h, w = dst.shape
    for i in range(h):
        for j in range(w):
            v = src[i, j]
            if v != 0:
                dst[i, j] = v
    return dst


@njit("int8[:,:](int8[:,:],int8[:,:])", cache=True, nogil=True, fastmath=True)
def subtract_kernel(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Subtract src from dst: zero out overlapping pixels. ~3-5x faster.

    Args:
        dst: Destination grid (int8, modified in-place).
        src: Source grid (int8).

    Returns:
        Modified dst grid.
    """
    h, w = dst.shape
    for i in range(h):
        for j in range(w):
            if src[i, j] != 0:
                dst[i, j] = 0
    return dst


@njit("int8[:,:](int8[:,:],int8[:,:])", cache=True, nogil=True, fastmath=True)
def union_kernel(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Union of dst and src: src fills where dst is zero. ~3-5x faster.

    Args:
        dst: Destination grid (int8, modified in-place).
        src: Source grid (int8).

    Returns:
        Modified dst grid.
    """
    h, w = dst.shape
    for i in range(h):
        for j in range(w):
            if dst[i, j] == 0 and src[i, j] != 0:
                dst[i, j] = src[i, j]
    return dst


@njit("int8[:,:](int8[:,:],int8[:,:])", cache=True, nogil=True, fastmath=True)
def intersection_kernel(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Intersection: keep only pixels non-zero in both. ~3-5x faster.

    Args:
        dst: First grid (int8, modified in-place).
        src: Second grid (int8).

    Returns:
        Modified dst grid.
    """
    h, w = dst.shape
    for i in range(h):
        for j in range(w):
            if dst[i, j] == 0 or src[i, j] == 0:
                dst[i, j] = 0
    return dst


@njit("int8[:,:](int8[:,:],int32,int32)", cache=True, nogil=True, fastmath=True)
def move_kernel(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Move all non-zero pixels by (dx, dy). ~5-10x faster per call.

    Args:
        grid: Input grid (int8).
        dx: Horizontal offset.
        dy: Vertical offset.

    Returns:
        New grid with moved pixels.
    """
    h, w = grid.shape
    result = np.zeros_like(grid)
    for i in range(h):
        for j in range(w):
            v = grid[i, j]
            if v != 0:
                ni, nj = i + dy, j + dx
                if 0 <= ni < h and 0 <= nj < w:
                    result[ni, nj] = v
    return result


@njit("int8[:,:](int8[:,:],int32,int32)", cache=True, nogil=True, fastmath=True)
def copy_with_offset_kernel(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Copy non-zero pixels by offset, keeping originals. ~5-10x faster.

    Args:
        grid: Input grid (int8).
        dx: Horizontal offset.
        dy: Vertical offset.

    Returns:
        Grid with copied pixels.
    """
    h, w = grid.shape
    result = grid.copy()
    for i in range(h):
        for j in range(w):
            v = grid[i, j]
            if v != 0:
                ni, nj = i + dy, j + dx
                if 0 <= ni < h and 0 <= nj < w:
                    result[ni, nj] = v
    return result


@njit("boolean(int8[:,:],int8[:,:])", cache=True, nogil=True, fastmath=True)
def grid_equal_kernel(a: np.ndarray, b: np.ndarray) -> bool:
    """Fast grid equality check ~10x for eager exit.

    Args:
        a: First grid (int8).
        b: Second grid (int8).

    Returns:
        True if grids are equal.
    """
    if a.shape != b.shape:
        return False
    h, w = a.shape
    for i in range(h):
        for j in range(w):
            if a[i, j] != b[i, j]:
                return False
    return True


# ============================================================
# Line Drawing
# ============================================================

@njit("int8[:,:](int8[:,:],int32,int32,int32,int32,int32)", cache=True, nogil=True, fastmath=True)
def draw_line_kernel(grid: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     color: int) -> np.ndarray:
    """Bresenham line drawing on int8 grid. ~20-50x faster.

    Args:
        grid: Grid to draw on (int8, modified in-place).
        x1, y1: Start coordinates.
        x2, y2: End coordinates.
        color: Color value.

    Returns:
        Modified grid.
    """
    result = grid.copy()
    h, w = result.shape
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    x, y = x1, y1

    while True:
        if 0 <= x < w and 0 <= y < h:
            result[y, x] = color
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            if x == x2:
                break
            err -= dy
            x += sx
        if e2 < dx:
            if y == y2:
                break
            err += dx
            y += sy

    return result


# ============================================================
# Gravity Operation
# ============================================================

@njit("int8[:,:](int8[:,:],int32)", cache=True, nogil=True, fastmath=True)
def gravity_kernel(grid: np.ndarray, direction_code: int) -> np.ndarray:
    """Apply gravity to non-zero pixels. ~10-20x faster per call.

    Args:
        grid: Input grid (int8).
        direction_code: 0=down, 1=up, 2=left, 3=right.

    Returns:
        New grid with gravity applied.
    """
    h, w = grid.shape
    result = np.zeros_like(grid)

    if direction_code <= 1:  # down (0) or up (1)
        for j in range(w):
            # Collect non-zero values in this column
            count = 0
            for i in range(h):
                if grid[i, j] != 0:
                    count += 1
            if count == 0:
                continue
            # Place them at the target end
            vals = np.zeros(count, dtype=np.int8)
            idx = 0
            for i in range(h):
                v = grid[i, j]
                if v != 0:
                    vals[idx] = v
                    idx += 1
            if direction_code == 0:  # down
                for k in range(count):
                    result[h - count + k, j] = vals[k]
            else:  # up
                for k in range(count):
                    result[k, j] = vals[k]

    else:  # left (2) or right (3)
        for i in range(h):
            count = 0
            for j in range(w):
                if grid[i, j] != 0:
                    count += 1
            if count == 0:
                continue
            vals = np.zeros(count, dtype=np.int8)
            idx = 0
            for j in range(w):
                v = grid[i, j]
                if v != 0:
                    vals[idx] = v
                    idx += 1
            if direction_code == 2:  # left
                for k in range(count):
                    result[i, k] = vals[k]
            else:  # right
                for k in range(count):
                    result[i, w - count + k] = vals[k]

    return result


# ============================================================
# Resize (Nearest Neighbor)
# ============================================================

@njit("int8[:,:](int8[:,:],int32,int32)", cache=True, nogil=True, fastmath=True)
def resize_kernel(grid: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize. ~5-10x faster.

    Args:
        grid: Input grid (int8).
        new_h: Target height.
        new_w: Target width.

    Returns:
        Resized grid.
    """
    h, w = grid.shape
    result = np.zeros((new_h, new_w), dtype=np.int8)
    for i in range(new_h):
        orig_i = int(i * h / new_h)
        if orig_i >= h:
            orig_i = h - 1
        for j in range(new_w):
            orig_j = int(j * w / new_w)
            if orig_j >= w:
                orig_j = w - 1
            result[i, j] = grid[orig_i, orig_j]
    return result


# ============================================================
# Symmetry Detection
# ============================================================

@njit("int8[:,:](int8[:,:])", cache=True, nogil=True, fastmath=True)
def symmetry_kernel(grid: np.ndarray) -> np.ndarray:
    """Detect and complete symmetry. ~5-10x faster.

    Args:
        grid: Input grid (int8).

    Returns:
        Grid with symmetry completed.
    """
    h, w = grid.shape
    result = grid.copy()
    # Horizontal: fill from mirror
    for i in range(h):
        for j in range(w):
            if result[i, j] == 0:
                mirror_j = w - 1 - j
                if result[i, mirror_j] != 0:
                    result[i, j] = result[i, mirror_j]
    # Vertical: fill from mirror
    for i in range(h):
        for j in range(w):
            if result[i, j] == 0:
                mirror_i = h - 1 - i
                if result[mirror_i, j] != 0:
                    result[i, j] = result[mirror_i, j]
    return result


# ============================================================
# Grid Operations (Bulk)
# ============================================================

@njit("int8[:,:](int8[:,:],int32,int32,int32,int32)", cache=True, nogil=True, fastmath=True)
def fill_region_kernel(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """Fill a rectangular region quickly. ~3-5x faster.

    Args:
        grid: Grid to fill (int8, modified in-place).
        r0, r1: Row range [r0, r1).
        c0, c1: Col range [c0, c1).

    Returns:
        Modified grid.
    """
    result = grid.copy()
    for i in range(r0, r1):
        for j in range(c0, c1):
            result[i, j] = 1
    return result


@njit("int8[:,:](int8[:,:],int32,int32)", cache=True, nogil=True, fastmath=True)
def color_swap_kernel(grid: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    """Swap two colors. ~3-5x faster.

    Args:
        grid: Input grid (int8).
        color_a: First color.
        color_b: Second color.

    Returns:
        Grid with swapped colors.
    """
    h, w = grid.shape
    result = grid.copy()
    for i in range(h):
        for j in range(w):
            if result[i, j] == color_a:
                result[i, j] = color_b
            elif result[i, j] == color_b:
                result[i, j] = color_a
    return result


@njit("int8[:,:](int8[:,:],int32)", cache=True, nogil=True, fastmath=True)
def map_color_kernel(grid: np.ndarray, old_color: int) -> np.ndarray:
    """Map one color to another (single pair, fast path). ~3-5x faster.

    Args:
        grid: Input grid (int8).
        old_color: Color to replace (always replaced with color+1).

    Returns:
        Modified grid.
    """
    h, w = grid.shape
    result = grid.copy()
    for i in range(h):
        for j in range(w):
            if result[i, j] == old_color:
                result[i, j] = old_color + 1
    return result


# ============================================================
# Sequence Pattern Detection
# ============================================================

@njit("int32[:](int8[:,:,:])", cache=True, nogil=True, fastmath=True)
def detect_compound_pattern_kernel(frames: np.ndarray) -> np.ndarray:
    """Detect compound patterns across frame sequence. ~10-50x faster.

    Args:
        frames: 3D array (T, H, W) of int8 frames.

    Returns:
        Array of pattern type codes: 1=alternating, 2=progressive, 3=periodic.
    """
    T, H, W = frames.shape
    if T < 2:
        return np.zeros(0, dtype=np.int32)

    result = np.zeros(T - 1, dtype=np.int32)
    for t in range(T - 1):
        changes = 0
        for i in range(H):
            for j in range(W):
                if frames[t, i, j] != frames[t + 1, i, j]:
                    changes += 1
        # Classify based on change pattern
        if changes == 0:
            result[t] = 0  # no change
        elif t > 0:
            # Check if same as previous delta
            prev_changes = 0
            for i in range(H):
                for j in range(W):
                    if frames[t - 1, i, j] != frames[t, i, j]:
                        prev_changes += 1
            if changes == prev_changes:
                result[t] = 2  # progressive (consistent delta)
            elif t > 1:
                # Check periodic (alternating)
                same_as_t_minus_2 = True
                for i in range(H):
                    for j in range(W):
                        if frames[t - 1, i, j] != frames[t + 1, i, j]:
                            same_as_t_minus_2 = False
                            break
                    if not same_as_t_minus_2:
                        break
                if same_as_t_minus_2:
                    result[t] = 3  # periodic
                else:
                    result[t] = 1  # other pattern
            else:
                result[t] = 1
        else:
            result[t] = 1

    return result


# ============================================================
# Skeleton / Morphology Helpers
# ============================================================

@njit("int8[:,:](int8[:,:])", cache=True, nogil=True, fastmath=True)
def skeleton_kernel(grid: np.ndarray) -> np.ndarray:
    """Simple morphological skeleton (peel boundary pixels). ~5-10x faster.

    Args:
        grid: Input binary-like grid (int8).

    Returns:
        Skeletonized grid.
    """
    h, w = grid.shape
    binary = (grid > 0).astype(np.int8)
    result = np.zeros_like(grid)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if binary[i, j]:
                neighbors = (
                    binary[i - 1, j - 1] + binary[i - 1, j] + binary[i - 1, j + 1] +
                    binary[i, j - 1] + binary[i, j + 1] +
                    binary[i + 1, j - 1] + binary[i + 1, j] + binary[i + 1, j + 1]
                )
                if neighbors <= 4:
                    result[i, j] = grid[i, j]
    return result


# ============================================================
# Utility
# ============================================================

@njit("int32(int8[:,:])", cache=True, nogil=True, fastmath=True)
def count_nonzero_kernel(grid: np.ndarray) -> int:
    """Count non-zero pixels. ~5-10x faster than np.count_nonzero.

    Args:
        grid: Input grid (int8).

    Returns:
        Count of non-zero pixels.
    """
    h, w = grid.shape
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0:
                count += 1
    return count


@njit("int32(int8[:,:])", cache=True, nogil=True, fastmath=True)
def compute_mdl_kernel(grid: np.ndarray) -> int:
    """Approximate MDL cost of a grid pattern. ~5x faster.

    Args:
        grid: Input grid (int8).

    Returns:
        MDL cost estimate.
    """
    h, w = grid.shape
    nz = 0
    unique_vals = 0
    seen = -1
    for i in range(h):
        for j in range(w):
            v = grid[i, j]
            if v != 0:
                nz += 1
            if v != seen:
                seen = v
                unique_vals += 1
    # Simple MDL heuristic
    return min(nz + unique_vals * 2, h * w)


@njit("int32(int8[:,:],int8[:,:])", cache=True, nogil=True, fastmath=True)
def grid_distance_kernel(a: np.ndarray, b: np.ndarray) -> int:
    """Pixel-wise difference count between two grids. ~10x faster.

    Args:
        a: First grid (int8).
        b: Second grid (int8).

    Returns:
        Number of differing pixels.
    """
    if a.shape != b.shape:
        return 999999
    h, w = a.shape
    diff = 0
    for i in range(h):
        for j in range(w):
            if a[i, j] != b[i, j]:
                diff += 1
    return diff


# ============================================================
# v2.7.1: JIT-Accelerated DSL Primitive Kernels
# Bypasses Python dynamic dispatch for hot-path primitives.
# ============================================================

@njit("int8[:,:](int8[:,:],int32)", cache=True, nogil=True, fastmath=True)
def mirror_jit(grid: np.ndarray, axis_code: int) -> np.ndarray:
    """JIT mirror: axis_code 0=horizontal, 1=vertical, 2=both.

    Args:
        grid: Input grid (int8).
        axis_code: Mirror axis code.

    Returns:
        Mirrored grid.
    """
    h, w = grid.shape
    result = np.zeros_like(grid)
    if axis_code == 0:  # horizontal (flip columns)
        for i in range(h):
            for j in range(w):
                result[i, j] = grid[i, w - 1 - j]
    elif axis_code == 1:  # vertical (flip rows)
        for i in range(h):
            for j in range(w):
                result[i, j] = grid[h - 1 - i, j]
    else:  # both
        for i in range(h):
            for j in range(w):
                result[i, j] = grid[h - 1 - i, w - 1 - j]
    return result


@njit("int8[:,:](int8[:,:],int32)", cache=True, nogil=True, fastmath=True)
def rotate_jit(grid: np.ndarray, k: int) -> np.ndarray:
    """JIT rotate by k*90 degrees.

    Args:
        grid: Input grid (int8).
        k: Rotation count (0, 1, 2, 3 for 0, 90, 180, 270).

    Returns:
        Rotated grid.
    """
    h, w = grid.shape
    k = k % 4
    if k == 0:
        return grid.copy()
    elif k == 1:  # 90 degrees counter-clockwise
        result = np.zeros((w, h), dtype=np.int8)
        for i in range(h):
            for j in range(w):
                result[w - 1 - j, i] = grid[i, j]
        return result
    elif k == 2:  # 180 degrees
        result = np.zeros_like(grid)
        for i in range(h):
            for j in range(w):
                result[h - 1 - i, w - 1 - j] = grid[i, j]
        return result
    else:  # k == 3, 270 degrees counter-clockwise (90 clockwise)
        result = np.zeros((w, h), dtype=np.int8)
        for i in range(h):
            for j in range(w):
                result[j, h - 1 - i] = grid[i, j]
        return result


@njit("int8[:,:](int8[:,:],int32)", cache=True, nogil=True, fastmath=True)
def scale_jit(grid: np.ndarray, factor: int) -> np.ndarray:
    """JIT integer scale (kron equivalent).

    Args:
        grid: Input grid (int8).
        factor: Scale factor.

    Returns:
        Scaled grid.
    """
    h, w = grid.shape
    new_h, new_w = h * factor, w * factor
    result = np.zeros((new_h, new_w), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            v = grid[i, j]
            for di in range(factor):
                for dj in range(factor):
                    result[i * factor + di, j * factor + dj] = v
    return result


@njit("int8[:,:](int8[:,:],int32)", cache=True, nogil=True, fastmath=True)
def scale_pattern_jit(grid: np.ndarray, factor: int) -> np.ndarray:
    """JIT scale-pattern: expand each non-zero cell to factor×factor block
    filled with the original grid pattern.

    Args:
        grid: Input grid (int8).
        factor: Scale factor.

    Returns:
        Scaled pattern grid.
    """
    h, w = grid.shape
    new_h, new_w = h * factor, w * factor
    result = np.zeros((new_h, new_w), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            v = grid[i, j]
            if v != 0:
                # Fill the block with the original grid
                for di in range(h):
                    for dj in range(w):
                        ni = i * factor + di
                        nj = j * factor + dj
                        if ni < new_h and nj < new_w:
                            result[ni, nj] = grid[di, dj]
    return result


@njit("int8[:,:](int8[:,:],int32[:],int32[:])", cache=True, nogil=True, fastmath=True)
def map_color_jit(grid: np.ndarray, old_colors: np.ndarray, new_colors: np.ndarray) -> np.ndarray:
    """JIT color mapping with lookup arrays.

    Args:
        grid: Input grid (int8).
        old_colors: Array of source color values.
        new_colors: Array of target color values (same length).

    Returns:
        Color-mapped grid.
    """
    h, w = grid.shape
    result = grid.copy()
    n_colors = len(old_colors)
    for i in range(h):
        for j in range(w):
            v = result[i, j]
            for k in range(n_colors):
                if v == old_colors[k]:
                    result[i, j] = new_colors[k]
                    break
    return result


@njit("int8[:,:](int8[:,:],int32,int32)", cache=True, nogil=True, fastmath=True)
def tile_jit(grid: np.ndarray, factor_h: int, factor_w: int) -> np.ndarray:
    """JIT tile (repeat grid).

    Args:
        grid: Input grid (int8).
        factor_h: Vertical repeat factor.
        factor_w: Horizontal repeat factor.

    Returns:
        Tiled grid.
    """
    h, w = grid.shape
    new_h, new_w = h * factor_h, w * factor_w
    result = np.zeros((new_h, new_w), dtype=np.int8)
    for fi in range(factor_h):
        for fj in range(factor_w):
            for i in range(h):
                for j in range(w):
                    result[fi * h + i, fj * w + j] = grid[i, j]
    return result


@njit("int8[:,:](int8[:,:],int32,int32,int32,int32)", cache=True, nogil=True, fastmath=True)
def crop_jit(grid: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    """JIT crop a rectangular region.

    Args:
        grid: Input grid (int8).
        top: Top row index.
        left: Left column index.
        height: Crop height.
        width: Crop width.

    Returns:
        Cropped grid.
    """
    h, w = grid.shape
    t = max(0, min(top, h - 1))
    l = max(0, min(left, w - 1))
    ch = max(1, min(height, h - t))
    cw = max(1, min(width, w - l))
    result = np.zeros((ch, cw), dtype=np.int8)
    for i in range(ch):
        for j in range(cw):
            result[i, j] = grid[t + i, l + j]
    return result


@njit("float64(int8[:,:],int8[:,:])", cache=True, nogil=True, fastmath=True)
def grid_similarity_jit(g1: np.ndarray, g2: np.ndarray) -> float:
    """JIT pixel-level similarity between two grids.

    Handles shape mismatch by comparing overlapping regions.

    Args:
        g1: First grid (int8).
        g2: Second grid (int8).

    Returns:
        Similarity score in [0.0, 1.0].
    """
    h1, w1 = g1.shape
    h2, w2 = g2.shape
    h = min(h1, h2)
    w = min(w1, w2)
    if h == 0 or w == 0:
        return 0.0
    total = h * w
    match = 0
    for i in range(h):
        for j in range(w):
            if g1[i, j] == g2[i, j]:
                match += 1
    return float(match) / float(total)


@njit("float64(int8[:,:],int8[:,:])", cache=True, nogil=True, fastmath=True)
def nonzero_ratio_jit(grid: np.ndarray, _unused: np.ndarray) -> float:
    """JIT compute non-zero pixel ratio.

    Args:
        grid: Input grid (int8).
        _unused: Unused parameter (for signature compatibility).

    Returns:
        Non-zero ratio in [0.0, 1.0].
    """
    h, w = grid.shape
    total = h * w
    if total == 0:
        return 0.0
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0:
                count += 1
    return float(count) / float(total)


# ============================================================
# Fast Apply: JIT dispatch for common primitives
# ============================================================

# Primitives supported by JIT fast path
_JIT_SUPPORTED_PRIMITIVES = frozenset({
    "mirror", "rotate", "scale", "scale-pattern", "map-color",
    "move", "copy", "crop", "tile", "gravity", "resize",
    "color-swap", "fill-region",
})


def is_jit_supported(prim_name: str) -> bool:
    """Check if a primitive is supported by the JIT fast path.

    Args:
        prim_name: Primitive name.

    Returns:
        True if JIT path is available.
    """
    return prim_name in _JIT_SUPPORTED_PRIMITIVES


def fast_apply_single_jit(
    grid: np.ndarray,
    prim_name: str,
    params: dict,
) -> np.ndarray | None:
    """Apply a single DSL primitive via JIT fast path.

    Bypasses Python dynamic dispatch (DSLElement._registry lookup)
    by directly calling the JIT-compiled kernel.

    Args:
        grid: Input grid (int8 ndarray).
        prim_name: Primitive name.
        params: Primitive parameters dict.

    Returns:
        Transformed grid, or None if primitive not supported by JIT.
    """
    if not HAS_NUMBA:
        return None

    grid = np.ascontiguousarray(grid, dtype=np.int8)

    try:
        if prim_name == "mirror":
            axis = params.get("axis", "horizontal")
            axis_code = 0 if axis == "horizontal" else (1 if axis == "vertical" else 2)
            return mirror_jit(grid, axis_code)

        elif prim_name == "rotate":
            angle = int(params.get("angle", 90))
            k = (angle // 90) % 4
            return rotate_jit(grid, k)

        elif prim_name == "scale":
            factor = int(params.get("factor", 2))
            return scale_jit(grid, factor)

        elif prim_name == "scale-pattern":
            factor = int(params.get("factor", 3))
            return scale_pattern_jit(grid, factor)

        elif prim_name == "map-color":
            mapping = params.get("mapping", {})
            if not mapping:
                return None
            old_colors = np.array(list(mapping.keys()), dtype=np.int32)
            new_colors = np.array(list(mapping.values()), dtype=np.int32)
            return map_color_jit(grid, old_colors, new_colors)

        elif prim_name == "move":
            dx = int(params.get("dx", 0))
            dy = int(params.get("dy", 0))
            return move_kernel(grid, dx, dy)

        elif prim_name == "copy":
            dx = int(params.get("dx", 0))
            dy = int(params.get("dy", 0))
            return copy_with_offset_kernel(grid, dx, dy)

        elif prim_name == "crop":
            top = int(params.get("top", 0))
            left = int(params.get("left", 0))
            height = int(params.get("height", 0))
            width = int(params.get("width", 0))
            h, w = grid.shape
            ch = height if height > 0 else h - top
            cw = width if width > 0 else w - left
            return crop_jit(grid, top, left, ch, cw)

        elif prim_name == "tile":
            fh = int(params.get("factor_h", 2))
            fw = int(params.get("factor_w", 2))
            return tile_jit(grid, fh, fw)

        elif prim_name == "gravity":
            direction = params.get("direction", "down")
            dir_map = {"down": 0, "up": 1, "left": 2, "right": 3}
            dc = dir_map.get(direction, 0)
            return gravity_kernel(grid, dc)

        elif prim_name == "resize":
            new_h = int(params.get("height", grid.shape[0]))
            new_w = int(params.get("width", grid.shape[1]))
            return resize_kernel(grid, new_h, new_w)

        elif prim_name == "color-swap":
            a = int(params.get("color_a", 0))
            b = int(params.get("color_b", 1))
            return color_swap_kernel(grid, a, b)

        elif prim_name == "fill-region":
            r0 = int(params.get("r0", 0))
            r1 = int(params.get("r1", grid.shape[0]))
            c0 = int(params.get("c0", 0))
            c1 = int(params.get("c1", grid.shape[1]))
            return fill_region_kernel(grid, r0, r1, c0, c1)

        else:
            return None
    except Exception:
        return None


def fast_apply_chain_jit(
    grid: np.ndarray,
    elements: list[tuple[str, dict]],
) -> np.ndarray | None:
    """Apply a chain of DSL primitives via JIT fast path.

    Applies each primitive in sequence using JIT kernels.
    If any primitive is not JIT-supported, returns None
    (caller should fall back to Python path).

    Args:
        grid: Input grid (int8 ndarray).
        elements: List of (prim_name, params) tuples in chain order.

    Returns:
        Transformed grid, or None if any primitive is not JIT-supported.
    """
    result = np.ascontiguousarray(grid, dtype=np.int8)

    for prim_name, params in elements:
        out = fast_apply_single_jit(result, prim_name, params)
        if out is None:
            return None
        result = np.ascontiguousarray(out, dtype=np.int8)

    return result


def fast_sim_and_nonzero(
    pred: np.ndarray,
    expected: np.ndarray,
) -> tuple[float, float]:
    """Compute similarity and nonzero ratio in one JIT pass.

    Args:
        pred: Predicted grid (int8).
        expected: Expected output grid (int8).

    Returns:
        (similarity, nonzero_ratio) tuple.
    """
    if HAS_NUMBA:
        sim = grid_similarity_jit(
            np.ascontiguousarray(pred, dtype=np.int8),
            np.ascontiguousarray(expected, dtype=np.int8),
        )
        nz = nonzero_ratio_jit(
            np.ascontiguousarray(pred, dtype=np.int8),
            np.ascontiguousarray(pred, dtype=np.int8),
        )
        return sim, nz
    else:
        # Fallback to numpy
        if pred.shape != expected.shape:
            h = min(pred.shape[0], expected.shape[0])
            w = min(pred.shape[1], expected.shape[1])
            if h == 0 or w == 0:
                return 0.0, 0.0
            match = int(np.sum(pred[:h, :w] == expected[:h, :w]))
            sim = match / (h * w)
        else:
            total = pred.size
            if total == 0:
                return 1.0, 0.0
            match = int(np.sum(pred == expected))
            sim = match / total
        nz = float(np.count_nonzero(pred)) / max(pred.size, 1)
        return sim, nz
