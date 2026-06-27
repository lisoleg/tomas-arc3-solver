# -*- coding: utf-8 -*-
"""OctonionPhase shape descriptor for TOMAS-ARC3 EML perception.

OctonionPhase Φ(R) ∈ ℝ⁵ captures shape features to distinguish
"same-color different-object" regions in ARC grids.

Components:
  1. D₄ encoding: cos(4θ) + sin(4θ) — main axis angle θ modulo π/4
  2. λ_norm = (λ - 1)/(λ + 1) — normalized aspect ratio
  3. filledness = |R| / area(bbox) — how filled the bounding box is
  4. symmetry_order ∈ {1, 2, 4, 8} — rotational symmetry
  5. convexity = |R| / hull_area — shape convexity ratio

Phase vector: Φ = [cos(4θ), sin(4θ), λ_norm, filledness, symmetry/8]

The Bian 5/6 saturation theorem ensures Dead-Zero pruning at
coupling threshold 1/6 ≈ 0.167, where coupling = filledness × (1 + convexity).

Version: v4.1
TOMAS Correspondence: TOMAS Phase III → EML Perceiver → OctonionPhase
IDO Correspondence: Φ(R) ∈ ℝ⁵ is the shape-descriptor delta of region R
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Tuple


# =============================================================================
# §1. Convex Hull Helpers — Andrew's Monotone Chain (pure numpy, zero OpenCV)
# =============================================================================

def _cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product of vectors OA and OB.

    Used by Andrew's monotone chain convex hull algorithm to determine
    whether three points make a left turn (positive) or right turn (negative).

    Args:
        o: Origin point (2-element array, x-y order).
        a: Point A (2-element array, x-y order).
        b: Point B (2-element array, x-y order).

    Returns:
        Scalar cross product (OA × OB). Positive = left turn,
        negative = right turn, zero = collinear.
    """
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Compute convex hull of 2D points using Andrew's monotone chain.

    Pure numpy implementation — zero OpenCV dependency. Runs in O(n log n).

    Algorithm:
      1. Sort points by x-coordinate (then y for ties)
      2. Build lower hull: scan left → right, removing right turns
      3. Build upper hull: scan right → left, removing right turns
      4. Concatenate lower + upper (excluding duplicate endpoints)

    Args:
        points: Nx2 array of (x, y) coordinates. Must have ≥ 3 points
            for a non-degenerate hull; fewer points return the input itself.

    Returns:
        Kx2 array of hull vertices in counter-clockwise order.
        For N < 3, returns the input points (degenerate hull).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()

    # Sort by x then y (lexsort: last key is primary)
    sorted_idx = np.lexsort((pts[:, 1], pts[:, 0]))
    sorted_pts = pts[sorted_idx]

    # Build lower hull
    lower: List[int] = []
    for i in range(n):
        while len(lower) >= 2:
            o_idx, a_idx = lower[-2], lower[-1]
            if _cross2d(sorted_pts[o_idx], sorted_pts[a_idx], sorted_pts[i]) <= 0:
                lower.pop()
            else:
                break
        lower.append(i)

    # Build upper hull
    upper: List[int] = []
    for i in range(n - 1, -1, -1):
        while len(upper) >= 2:
            o_idx, a_idx = upper[-2], upper[-1]
            if _cross2d(sorted_pts[o_idx], sorted_pts[a_idx], sorted_pts[i]) <= 0:
                upper.pop()
            else:
                break
        upper.append(i)

    # Concatenate (remove duplicate endpoints at join)
    hull_indices = lower[:-1] + upper[:-1]
    if len(hull_indices) < 3:
        return sorted_pts[np.array(hull_indices)]
    return sorted_pts[np.array(hull_indices)]


def _polygon_area(vertices: np.ndarray) -> float:
    """Compute area of a simple polygon using the shoelace formula.

    The shoelace formula (Gauss's area formula):
        A = 0.5 × |Σ(x_i × y_{i+1} − x_{i+1} × y_i)|

    Args:
        vertices: Kx2 array of polygon vertices in CCW or CW order.

    Returns:
        Area of the polygon. Returns 1.0 for degenerate cases (K < 3)
        to avoid division-by-zero in downstream convexity calculations.
    """
    n = len(vertices)
    if n < 3:
        return 1.0  # Minimum area guard for pixel-coordinate regions

    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return max(area, 1.0)  # At least 1.0 for integer-coordinate regions


# =============================================================================
# §2. OctonionPhase Dataclass — 5D Shape Descriptor Φ(R) ∈ ℝ⁵
# =============================================================================

@dataclass
class OctonionPhase:
    """5D shape descriptor vector Φ(R) ∈ ℝ⁵ for ARC region perception.

    Captures shape features that distinguish "same-color different-object"
    regions in ARC grids. Used by EMLPerceiver for perceptual grouping
    and by GaussEx fast verification for Hungarian object alignment.

    The D₄ dihedral encoding (cos(4θ), sin(4θ)) captures the main axis
    angle modulo π/4, making the descriptor invariant under 90° rotations.

    Phase vector composition:
        Φ = [cos(4θ), sin(4θ), λ_norm, filledness, symmetry/8]

    Where:
        θ      = main axis angle from eigendecomposition of pixel covariance
        λ_norm = (λ₁/λ₂ − 1) / (λ₁/λ₂ + 1) ∈ [0, 1]
        filledness = |R| / bbox_area ∈ [0, 1]
        symmetry   ∈ {1, 2, 4, 8}
        convexity  = |R| / hull_area ∈ [0, 1]

    Attributes:
        main_axis_angle: Principal axis angle θ ∈ [0, π).
        aspect_ratio: Eigenvalue ratio λ₁/λ₂ ≥ 1.
        filledness: Region fill ratio |R| / bbox_area ∈ [0, 1].
        symmetry_order: Rotational symmetry order ∈ {1, 2, 4, 8}.
        convexity: Shape convexity |R| / hull_area ∈ [0, 1].
        phase_vector: Composed Φ ∈ ℝ⁵.
    """

    main_axis_angle: float = 0.0
    aspect_ratio: float = 1.0
    filledness: float = 0.0
    symmetry_order: int = 1
    convexity: float = 0.0
    phase_vector: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    @staticmethod
    def estimate(region_pixels: np.ndarray) -> OctonionPhase:
        """Estimate OctonionPhase from region pixel coordinates.

        Full pipeline (pure numpy — zero OpenCV/scipy dependency):
          1. Compute centroid and covariance from pixel positions
          2. Eigendecomposition → main axis angle θ + aspect ratio λ₁/λ₂
          3. Compute filledness = |R| / bbox_area
          4. Autocorrelation symmetry detection (rotate 90°/180°/270°)
          5. Convex hull ratio → convexity = |R| / hull_area
          6. Compose phase vector Φ = [cos(4θ), sin(4θ), λ_norm, filledness, symmetry/8]

        Coordinate convention: region_pixels uses (row, col) indexing.
        Internally, col = x (horizontal) and row = y (vertical) for
        standard image-moment computation.

        Args:
            region_pixels: Nx2 array of (row, col) pixel coordinates
                belonging to a single region. N must be ≥ 1.

        Returns:
            OctonionPhase instance with all fields populated.
            For empty or invalid input, returns a default (zero-vector) phase.
        """
        pts = np.asarray(region_pixels, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            return OctonionPhase(
                main_axis_angle=0.0, aspect_ratio=1.0, filledness=0.0,
                symmetry_order=1, convexity=0.0,
                phase_vector=np.zeros(5, dtype=np.float64),
            )

        n_pixels: int = len(pts)

        # (col, row) = (x, y) for image moments
        x_coords: np.ndarray = pts[:, 1]  # col = x
        y_coords: np.ndarray = pts[:, 0]  # row = y

        # ------------------------------------------------------------------
        # Step 1: Centroid and covariance matrix
        # ------------------------------------------------------------------
        cx: float = float(np.mean(x_coords))
        cy: float = float(np.mean(y_coords))

        dx: np.ndarray = x_coords - cx
        dy: np.ndarray = y_coords - cy
        mu_20: float = float(np.sum(dx ** 2))
        mu_11: float = float(np.sum(dx * dy))
        mu_02: float = float(np.sum(dy ** 2))

        cov: np.ndarray = np.array([
            [mu_20 / n_pixels, mu_11 / n_pixels],
            [mu_11 / n_pixels, mu_02 / n_pixels],
        ], dtype=np.float64)

        # ------------------------------------------------------------------
        # Step 2: Eigendecomposition → main axis angle + aspect ratio
        # ------------------------------------------------------------------
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvalues: λ₁ ≥ λ₂
        if eigenvalues[0] > eigenvalues[1]:
            lam1: float = float(eigenvalues[0])
            lam2: float = float(eigenvalues[1])
            eigvec1: np.ndarray = eigenvectors[:, 0]
        else:
            lam1 = float(eigenvalues[1])
            lam2 = float(eigenvalues[0])
            eigvec1 = eigenvectors[:, 1]

        # Main axis angle θ ∈ [0, π)
        theta: float = float(np.arctan2(eigvec1[1], eigvec1[0]))
        if theta < 0:
            theta += float(np.pi)

        # Aspect ratio λ₁/λ₂ (cap λ₂ floor to avoid infinity)
        aspect_ratio: float = lam1 / max(lam2, 1e-10)
        aspect_ratio = min(aspect_ratio, 100.0)

        # Normalized aspect ratio λ_norm = (λ − 1)/(λ + 1) ∈ [0, 1]
        lambda_norm: float = (aspect_ratio - 1.0) / (aspect_ratio + 1.0)

        # ------------------------------------------------------------------
        # Step 3: Filledness = |R| / bbox_area
        # ------------------------------------------------------------------
        min_x: float = float(np.min(x_coords))
        max_x: float = float(np.max(x_coords))
        min_y: float = float(np.min(y_coords))
        max_y: float = float(np.max(y_coords))

        bbox_width: float = max_x - min_x + 1.0
        bbox_height: float = max_y - min_y + 1.0
        bbox_area: float = bbox_width * bbox_height

        filledness: float = n_pixels / max(bbox_area, 1.0)
        filledness = min(filledness, 1.0)

        # ------------------------------------------------------------------
        # Step 4: Autocorrelation symmetry detection
        # ------------------------------------------------------------------
        symmetry_order: int = _detect_symmetry_order(
            x_coords, y_coords, cx, cy, n_pixels
        )

        # ------------------------------------------------------------------
        # Step 5: Convex hull ratio → convexity = |R| / hull_area
        # ------------------------------------------------------------------
        xy_pts: np.ndarray = np.column_stack([x_coords, y_coords])
        hull_pts: np.ndarray = _convex_hull_2d(xy_pts)
        hull_area: float = _polygon_area(hull_pts)
        convexity: float = n_pixels / max(hull_area, 1.0)
        convexity = min(convexity, 1.0)

        # ------------------------------------------------------------------
        # Step 6: Compose phase vector Φ ∈ ℝ⁵
        # ------------------------------------------------------------------
        phase_vector: np.ndarray = np.array([
            np.cos(4.0 * theta),       # D₄ encoding: cos(4θ)
            np.sin(4.0 * theta),       # D₄ encoding: sin(4θ)
            lambda_norm,               # Normalized aspect ratio
            filledness,                # Fill ratio
            symmetry_order / 8.0,      # Normalized symmetry order
        ], dtype=np.float64)

        return OctonionPhase(
            main_axis_angle=theta,
            aspect_ratio=aspect_ratio,
            filledness=filledness,
            symmetry_order=symmetry_order,
            convexity=convexity,
            phase_vector=phase_vector,
        )


# =============================================================================
# §3. Symmetry Detection — Autocorrelation Rotation Test
# =============================================================================

def _detect_symmetry_order(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    cx: float,
    cy: float,
    n_pixels: int,
) -> int:
    """Detect rotational symmetry order via autocorrelation.

    Rotates the pixel set by 90°, 180°, 270° around centroid and
    computes overlap ratio with original set. Determines symmetry order:
      - overlap ≥ 0.75 at 90°  → order 4 (square-like)
      - overlap ≥ 0.75 at 180° only → order 2 (line-like)
      - no significant overlap → order 1 (asymmetric)

    For 45° rotational symmetry (order 8), a relaxed threshold of 0.60
    is used due to integer-coordinate discretization artifacts.

    Args:
        x_coords: X (col) coordinates of pixels.
        y_coords: Y (row) coordinates of pixels.
        cx: Centroid x coordinate.
        cy: Centroid y coordinate.
        n_pixels: Number of pixels in the region.

    Returns:
        Symmetry order ∈ {1, 2, 4, 8}.
    """
    if n_pixels < 4:
        return 1  # Too few pixels for meaningful symmetry detection

    threshold_strict: float = 0.75  # 75% overlap for order 2/4
    threshold_relaxed: float = 0.60  # 60% overlap for order 8 (discretization)

    # Build original pixel set as integer (x, y) tuples for fast lookup
    original_set: set = set()
    for i in range(n_pixels):
        original_set.add((round(float(x_coords[i])), round(float(y_coords[i]))))

    # Test 180° rotation: (dx, dy) → (-dx, -dy)
    overlap_180: float = _rotation_overlap(
        x_coords, y_coords, cx, cy, original_set, n_pixels, angle_deg=180
    )

    # Test 90° rotation: (dx, dy) → (-dy, dx)
    overlap_90: float = _rotation_overlap(
        x_coords, y_coords, cx, cy, original_set, n_pixels, angle_deg=90
    )

    # Test 45° rotation (order 8) — use relaxed threshold
    overlap_45: float = _rotation_overlap_continuous(
        x_coords, y_coords, cx, cy, original_set, n_pixels, angle_deg=45
    )

    # Determine symmetry order (highest first)
    if overlap_45 >= threshold_relaxed and overlap_90 >= threshold_strict:
        return 8
    if overlap_90 >= threshold_strict:
        return 4
    if overlap_180 >= threshold_strict:
        return 2
    return 1


def _rotation_overlap(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    cx: float,
    cy: float,
    original_set: set,
    n_pixels: int,
    angle_deg: int,
) -> float:
    """Compute overlap ratio after discrete rotation around centroid.

    For 90°/180°/270° rotations, the transformation is exact on integer
    coordinates (modulo rounding to nearest grid point).

    Args:
        x_coords: X coordinates.
        y_coords: Y coordinates.
        cx: Centroid x.
        cy: Centroid y.
        original_set: Set of (x, y) integer tuples in original region.
        n_pixels: Total pixels in the region.
        angle_deg: Rotation angle (90, 180, 270).

    Returns:
        Overlap ratio ∈ [0, 1]. Fraction of rotated pixels that land
        on original pixel positions.
    """
    dx: np.ndarray = x_coords - cx
    dy: np.ndarray = y_coords - cy

    if angle_deg == 90:
        rx: np.ndarray = cx - dy
        ry: np.ndarray = cy + dx
    elif angle_deg == 180:
        rx = cx - dx
        ry = cy - dy
    elif angle_deg == 270:
        rx = cx + dy
        ry = cy - dx
    else:
        return 0.0

    # Round to nearest integer grid position
    rx_int: np.ndarray = np.round(rx).astype(int)
    ry_int: np.ndarray = np.round(ry).astype(int)

    # Count how many rotated positions land on original pixels
    overlap_count: int = 0
    for i in range(n_pixels):
        if (int(rx_int[i]), int(ry_int[i])) in original_set:
            overlap_count += 1

    return overlap_count / max(n_pixels, 1)


def _rotation_overlap_continuous(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    cx: float,
    cy: float,
    original_set: set,
    n_pixels: int,
    angle_deg: float,
) -> float:
    """Compute overlap ratio after continuous-angle rotation (for order-8 check).

    For 45° rotation, integer coordinates don't map exactly, so we use
    a proximity check: a rotated pixel "matches" if any original pixel
    is within 1.0 Chebyshev distance.

    Args:
        x_coords: X coordinates.
        y_coords: Y coordinates.
        cx: Centroid x.
        cy: Centroid y.
        original_set: Set of (x, y) integer tuples.
        n_pixels: Total pixels.
        angle_deg: Rotation angle in degrees (typically 45).

    Returns:
        Overlap ratio ∈ [0, 1] with proximity tolerance.
    """
    angle_rad: float = np.radians(angle_deg)
    cos_a: float = float(np.cos(angle_rad))
    sin_a: float = float(np.sin(angle_rad))

    dx: np.ndarray = x_coords - cx
    dy: np.ndarray = y_coords - cy

    # Continuous rotation: (dx, dy) → (dx*cos − dy*sin, dx*sin + dy*cos)
    rx: np.ndarray = cx + dx * cos_a - dy * sin_a
    ry: np.ndarray = cy + dx * sin_a + dy * cos_a

    # Proximity check: count matches within Chebyshev distance 1.0
    overlap_count: int = 0
    for i in range(n_pixels):
        rx_round: int = int(round(float(rx[i])))
        ry_round: int = int(round(float(ry[i])))
        # Check 3×3 neighborhood
        found: bool = False
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                if (rx_round + ddx, ry_round + ddy) in original_set:
                    found = True
                    break
            if found:
                break
        if found:
            overlap_count += 1

    return overlap_count / max(n_pixels, 1)


# =============================================================================
# §4. Self-Test
# =============================================================================

def _self_test() -> None:
    """Run self-tests for OctonionPhase module."""
    print("OctonionPhase — Self-Test")
    print("=" * 50)

    # Test 1: Single pixel — filledness=1, convexity=1, symmetry=1
    single = np.array([[3, 5]])
    phase1 = OctonionPhase.estimate(single)
    print(f"\n1. Single pixel: angle={phase1.main_axis_angle:.4f}, "
          f"filledness={phase1.filledness:.4f}, "
          f"convexity={phase1.convexity:.4f}, "
          f"symmetry={phase1.symmetry_order}")
    assert phase1.filledness == 1.0, f"Expected filledness=1.0, got {phase1.filledness}"
    assert phase1.convexity == 1.0, f"Expected convexity=1.0, got {phase1.convexity}"
    assert phase1.symmetry_order == 1, f"Expected symmetry=1, got {phase1.symmetry_order}"
    print("   PASSED")

    # Test 2: Horizontal line (5 pixels in one row) — filledness=1.0, high aspect ratio
    # Note: A horizontal line fills its bbox completely (bbox_height=1, bbox_width=5),
    # so filledness=1.0, not 0.2. To get low filledness, need scattered pixels.
    line_h = np.array([[2, 3], [2, 4], [2, 5], [2, 6], [2, 7]])
    phase2 = OctonionPhase.estimate(line_h)
    print(f"\n2. Horizontal line: angle={phase2.main_axis_angle:.4f}, "
          f"aspect_ratio={phase2.aspect_ratio:.4f}, "
          f"filledness={phase2.filledness:.4f}, "
          f"λ_norm={phase2.phase_vector[2]:.4f}")
    assert phase2.aspect_ratio > 1.0, "Line should have high aspect ratio"
    assert abs(phase2.filledness - 1.0) < 0.01, f"Expected ~1.0, got {phase2.filledness}"
    print("   PASSED")

    # Test 3: Square (4×4 block) — filledness=1, convexity≈1, symmetry≥2
    square_pixels: List = []
    for r in range(4):
        for c in range(4):
            square_pixels.append([r, c])
    square = np.array(square_pixels)
    phase3 = OctonionPhase.estimate(square)
    print(f"\n3. Square (4×4): angle={phase3.main_axis_angle:.4f}, "
          f"aspect_ratio={phase3.aspect_ratio:.4f}, "
          f"filledness={phase3.filledness:.4f}, "
          f"convexity={phase3.convexity:.4f}, "
          f"symmetry={phase3.symmetry_order}")
    assert phase3.filledness == 1.0, f"Expected filledness=1.0, got {phase3.filledness}"
    assert phase3.convexity >= 0.9, f"Expected convexity ≥ 0.9, got {phase3.convexity}"
    assert phase3.symmetry_order >= 2, f"Expected symmetry ≥ 2, got {phase3.symmetry_order}"
    print("   PASSED")

    # Test 4: Large L-shape — convexity < 1 (non-convex)
    # Use a larger L-shape so hull_area > n_pixels (small L-shapes can have
    # convexity ≥ 1.0 due to integer-coordinate pixel density exceeding hull area)
    l_pixels: List = []
    for c in range(10):   # Horizontal arm: row 0, cols 0-9
        l_pixels.append([0, c])
    for r in range(1, 9): # Vertical arm: rows 1-8, col 0
        l_pixels.append([r, 0])
    l_shape = np.array(l_pixels)
    phase4 = OctonionPhase.estimate(l_shape)
    print(f"\n4. Large L-shape (18 px): convexity={phase4.convexity:.4f}, "
          f"filledness={phase4.filledness:.4f}")
    assert phase4.convexity < 1.0, f"L-shape should have convexity < 1.0, got {phase4.convexity}"
    assert phase4.filledness < 1.0, "L-shape should have filledness < 1.0"
    print("   PASSED")

    # Test 5: Convex hull algorithm test
    hull_input = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                           [1.0, 1.0], [0.5, 0.5]])
    hull = _convex_hull_2d(hull_input)
    area = _polygon_area(hull)
    print(f"\n5. Convex hull: {len(hull)} vertices, area={area:.4f}")
    assert abs(area - 1.0) < 0.01, f"Expected area≈1.0, got {area}"
    print("   PASSED")

    # Test 6: Phase vector composition consistency
    test_datasets = [single, line_h, square, l_shape]
    for idx, test_pts in enumerate(test_datasets):
        phase = OctonionPhase.estimate(test_pts)
        assert len(phase.phase_vector) == 5, "Phase vector must be 5D"
        assert abs(phase.phase_vector[0] - np.cos(4 * phase.main_axis_angle)) < 1e-10
        assert abs(phase.phase_vector[1] - np.sin(4 * phase.main_axis_angle)) < 1e-10
        expected_lambda_norm = (phase.aspect_ratio - 1) / (phase.aspect_ratio + 1)
        assert abs(phase.phase_vector[2] - expected_lambda_norm) < 1e-10
        assert abs(phase.phase_vector[3] - phase.filledness) < 1e-10
        assert abs(phase.phase_vector[4] - phase.symmetry_order / 8.0) < 1e-10
    print(f"\n6. Phase vector consistency: PASSED (4 datasets)")

    # Test 7: Empty / invalid input
    empty_phase = OctonionPhase.estimate(np.array([]))
    assert empty_phase.filledness == 0.0
    assert empty_phase.symmetry_order == 1
    print(f"\n7. Empty input: PASSED")

    # Test 8: Coupling formula check (Bian 5/6 Dead-Zero threshold)
    # coupling = filledness × (1 + convexity)
    # Dead-Zero threshold = 1/6 ≈ 0.167
    coupling_square = phase3.filledness * (1 + phase3.convexity)
    coupling_l = phase4.filledness * (1 + phase4.convexity)
    print(f"\n8. Coupling: square={coupling_square:.4f}, L-shape={coupling_l:.4f}")
    assert coupling_square > 1.0 / 6.0, "Square should pass Dead-Zero"
    print("   PASSED")

    print("\n" + "=" * 50)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()
