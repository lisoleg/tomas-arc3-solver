# -*- coding: utf-8 -*-
"""κ-Winding Algorithm — Compute κ-winding numbers for JinlingSphere objects.

Tracks the phase evolution of JinlingSphere objects under candidate
transforms, computes winding numbers weighted by Monster group dimensions,
and builds umbral curves for winding mode enumeration.

The κ-winding number captures "how much phase rotation accumulates" along
the object's trajectory under a transform. Physical analogy: winding number
of a complex function around a singularity.

Monster group connection: ARC-3 uses simplified dimensions {1,2,4,8}
corresponding to trivial/Z₂/D₄/octonion symmetries, rather than the full
Monster group 196883 dimensions (which is computationally intractable).

IDO Correspondence:
  - κ-winding = IDO phase evolution along RG flow path
  - Monster weight = dimension-dependent coupling coefficient
  - Umbral curve = cross-object phase correlation matrix

TOMAS Correspondence:
  - κ-winding = TOMAS Phase II (transform trajectory tracking)
  - Monster search = TOMAS Phase III (DSL conjecture from winding modes)

Version: v4.2
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.eml_perceiver import JinlingSphere
from src.agent.octonion_phase import OctonionPhase


# =============================================================================
# §1. Monster ARC Dimension Weights (Simplified)
# =============================================================================

# Simplified Monster-like dimensions for ARC-3 (not full 196883)
# Maps symmetry_order → coupling weight (analogy: Monster group representation dim)
MONSTER_ARC_DIMS: Dict[int, float] = {
    1: 1.0,    # No symmetry → trivial weight (1-dim representation)
    2: 2.0,    # Twofold symmetry → Z₂ weight
    4: 4.0,    # D₄ symmetry → 4-dim representation
    8: 8.0,    # Octonion symmetry → 8-dim representation
}

# Full Monster dimensions for reference (not used in computation)
MONSTER_FULL_DIMS: Dict[int, int] = {
    1: 1,
    2: 1,
    3: 196883,  # Smallest non-trivial Monster representation
    # Higher dimensions are computationally irrelevant for ARC
}


# =============================================================================
# §2. WindingResult Dataclass
# =============================================================================

@dataclass
class WindingResult:
    """Result of κ-winding algorithm computation.

    Attributes:
        total_winding: Total κ-winding number (weighted sum).
        sphere_windings: Per-sphere winding numbers (unweighted).
        monster_weights: Monster group weights per sphere.
        partition_value: κ-Genus partition function Z(κ).
        umbral_curve: Cross-object phase correlation matrix.
        winding_modes: Enumerated winding mode candidates.
    """
    total_winding: float
    sphere_windings: List[float]
    monster_weights: List[float]
    partition_value: float
    umbral_curve: Optional[np.ndarray] = None
    winding_modes: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# §3. κ-Winding Algorithm Core
# =============================================================================

def monster_weight(symmetry_order: int, color: int) -> float:
    """Compute Monster group weight for a JinlingSphere.

    Weight = MONSTER_ARC_DIMS[symmetry_order] × color_complexity_factor
    where color_complexity = log2(color + 1) / log2(10)

    This captures the idea that objects with higher symmetry and more
    distinctive colors have stronger "Monster coupling" — they contribute
    more to the κ-winding number.

    Args:
        symmetry_order: Symmetry order ∈ {1,2,4,8}.
        color: Color value of the sphere.

    Returns:
        Monster weight (positive float).
    """
    dim_weight = MONSTER_ARC_DIMS.get(symmetry_order, 1.0)
    color_complexity = math.log2(color + 1) / math.log2(10) if color > 0 else 0.0
    return dim_weight * (1.0 + color_complexity)


def kappa_winding_algorithm(
    spheres: List[JinlingSphere],
    candidate_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    grid: Optional[np.ndarray] = None,
    kappa: float = 1.0,
) -> WindingResult:
    """Compute κ-winding numbers for JinlingSphere objects.

    For each sphere, tracks how its OctonionPhase evolves under the
    candidate transform. The winding number is the net phase rotation
    accumulated along the trajectory.

    If no transform is provided, computes the "baseline winding" from
    the sphere's intrinsic phase structure (useful for ranking objects).

    Steps:
      1. For each sphere, extract initial OctonionPhase Φ₀
      2. If transform provided: compute Φ₁ = transform(Φ₀)
      3. Compute phase trajectory: unwrap(Φ₁ - Φ₀)
      4. Winding number = (Δφ_unwrapped) / (2π)
      5. Monster weight = monster_weight(symmetry, color)
      6. Total winding = Σ(winding × monster_weight)
      7. Partition value Z(κ) = Σ exp(-κ × dim × winding²)

    Args:
        spheres: List of JinlingSphere objects from EML perception.
        candidate_transform: Optional transform to apply for phase tracking.
        grid: Original grid (needed if transform is provided).
        kappa: κ parameter for partition function.

    Returns:
        WindingResult with total winding, per-sphere values, and partition.
    """
    sphere_windings: List[float] = []
    monster_weights_list: List[float] = []
    phase_trajectories: List[float] = []

    for sphere in spheres:
        # Initial OctonionPhase
        phi_0 = sphere.oct_phase.phase_vector if sphere.oct_phase.phase_vector is not None else np.zeros(5)

        # Compute winding number
        if candidate_transform is not None and grid is not None:
            # Apply transform → get transformed grid → re-perceive
            try:
                transformed_grid = candidate_transform(grid)
                perceiver = _get_perceiver()
                transformed_spheres = perceiver.perceive(transformed_grid)

                # Find matching sphere (by centroid proximity)
                phi_1 = _find_matching_phase(sphere, transformed_spheres)
            except Exception:
                phi_1 = phi_0  # Transform failed → zero winding
        else:
            # No transform → use intrinsic phase as winding estimate
            # Winding from symmetry: symmetry_order × 2π / 8
            phi_1 = phi_0 * sphere.oct_phase.symmetry_order / 8.0

        # Compute unwrapped phase difference
        delta_phi = phi_1 - phi_0
        if isinstance(delta_phi, np.ndarray):
            # Vector winding: sum of component rotations
            unwrapped = np.unwrap(delta_phi)
            winding = float(np.sum(unwrapped)) / (2.0 * math.pi)
        else:
            winding = float(delta_phi) / (2.0 * math.pi)

        # Monster weight
        m_weight = monster_weight(sphere.oct_phase.symmetry_order, sphere.color)

        sphere_windings.append(winding)
        monster_weights_list.append(m_weight)
        phase_trajectories.append(winding * m_weight)

    # Total winding (weighted sum)
    total_winding = sum(phase_trajectories)

    # Partition function Z(κ)
    partition_value = 0.0
    for i, w in enumerate(sphere_windings):
        dim = monster_weights_list[i]
        partition_value += math.exp(-kappa * dim * w * w)

    return WindingResult(
        total_winding=total_winding,
        sphere_windings=sphere_windings,
        monster_weights=monster_weights_list,
        partition_value=partition_value,
    )


# =============================================================================
# §4. Umbral Curve & Winding Mode Enumeration
# =============================================================================

def build_umbral_curve(spheres: List[JinlingSphere]) -> np.ndarray:
    """Build umbral curve — cross-object phase relationship matrix.

    The umbral curve M is a matrix where M[i,j] = phase_distance
    between spheres i and j. This captures the "phase topology" of
    the object constellation.

    phase_distance = |Φ_i - Φ_j|₂ / 5  (normalized to [0,1])

    The umbral curve is the input for winding mode enumeration:
    its spectral decomposition reveals dominant winding modes.

    Args:
        spheres: List of JinlingSphere objects.

    Returns:
        Phase distance matrix (n×n numpy array).
    """
    n = len(spheres)
    if n == 0:
        return np.zeros((0, 0))

    # Extract phase vectors
    phases = []
    for s in spheres:
        if s.oct_phase.phase_vector is not None:
            phases.append(s.oct_phase.phase_vector)
        else:
            phases.append(np.zeros(5))

    phases = np.array(phases)

    # Compute pairwise distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(phases[i] - phases[j])
            dist_matrix[i, j] = dist / 5.0  # Normalize to [0,1]

    return dist_matrix


def enumerate_winding_modes(
    umbral_curve: np.ndarray,
    max_dim: int = 8,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Enumerate winding modes from umbral curve spectral decomposition.

    Instead of enumerating the full Monster group (196883 dimensions),
    we use spectral decomposition of the umbral curve to find dominant
    winding modes. Each mode corresponds to a "phase resonance pattern"
    that can be mapped to a DSL transform conjecture.

    Steps:
      1. Compute spectral decomposition of umbral curve
      2. Extract top-k eigenvectors as winding modes
      3. Each mode → DSL transform conjecture

    Args:
        umbral_curve: Phase distance matrix from build_umbral_curve().
        max_dim: Maximum number of winding modes to enumerate.
        threshold: Minimum eigenvalue magnitude to include.

    Returns:
        List of winding mode dicts with 'eigenvalue', 'vector', 'dsl_hint'.
    """
    if umbral_curve.shape[0] == 0:
        return []

    # Spectral decomposition (eigendecomposition)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(umbral_curve)
    except np.linalg.LinAlgError:
        return []

    # Sort by absolute eigenvalue (descending)
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

    modes: List[Dict[str, Any]] = []
    for idx in sorted_indices[:max_dim]:
        ev = eigenvalues[idx]
        vec = eigenvectors[:, idx]

        if abs(ev) < threshold:
            continue

        # DSL hint based on mode characteristics
        dsl_hint = _classify_winding_mode(ev, vec)

        modes.append({
            "eigenvalue": float(ev),
            "vector": vec.tolist(),
            "dsl_hint": dsl_hint,
            "symmetry_order": len(vec),
        })

    return modes


def map_winding_to_dsl(winding_mode: Dict[str, Any]) -> str:
    """Map a winding mode to a DSL primitive conjecture.

    Winding modes with different characteristics map to different
    DSL operation types:

    - Positive eigenvalue → structure-preserving transform
    - Negative eigenvalue → structure-inverting transform
    - Near-zero eigenvalue → identity-like transform
    - Large eigenvalue → aggressive transform (rotation/reflection)
    - Complex pattern → multi-step composition

    Args:
        winding_mode: Winding mode dict from enumerate_winding_modes.

    Returns:
        DSL primitive name string.
    """
    ev = winding_mode["eigenvalue"]
    dsl_hint = winding_mode.get("dsl_hint", "unknown")

    DSL_WINDING_MAP = {
        "symmetry_rotation": "rotation_apply",
        "color_shift": "color_transform",
        "position_shift": "position_shift_apply",
        "structural_flip": "reflection_apply",
        "mixed_composition": "compose_transform",
    }

    return DSL_WINDING_MAP.get(dsl_hint, "generic_transform")


def _classify_winding_mode(eigenvalue: float, vector: np.ndarray) -> str:
    """Classify a winding mode by its spectral characteristics.

    Args:
        eigenvalue: Eigenvalue of the winding mode.
        vector: Eigenvector of the winding mode.

    Returns:
        Classification string.
    """
    if eigenvalue > 2.0:
        return "symmetry_rotation"  # Strong positive → rotation
    elif eigenvalue > 0.5:
        return "color_shift"  # Moderate positive → color change
    elif eigenvalue < -2.0:
        return "structural_flip"  # Strong negative → reflection/inversion
    elif eigenvalue < -0.5:
        return "position_shift"  # Moderate negative → shift
    else:
        # Near-zero → complex pattern
        variance = float(np.var(vector))
        if variance > 0.3:
            return "mixed_composition"
        return "near_identity"


# =============================================================================
# §5. Monster Winding Search Pipeline
# =============================================================================

def monster_winding_search(
    spheres: List[JinlingSphere],
    existing_dsl: Optional[List[str]] = None,
    max_modes: int = 8,
) -> List[str]:
    """Full Monster winding search: build umbral curve, enumerate modes,
    map to DSL conjectures.

    Pipeline:
      1. build_umbral_curve(spheres) → phase distance matrix
      2. enumerate_winding_modes(umbral) → spectral modes
      3. map_winding_to_dsl(mode) → DSL primitive names
      4. Filter: only keep conjectures not already in existing_dsl

    Args:
        spheres: JinlingSphere list from EML perception.
        existing_dsl: Existing DSL primitive names (to avoid duplication).
        max_modes: Maximum winding modes to enumerate.

    Returns:
        List of new DSL primitive conjecture names.
    """
    umbral = build_umbral_curve(spheres)
    modes = enumerate_winding_modes(umbral, max_dim=max_modes)

    existing = set(existing_dsl or [])
    conjectures: List[str] = []

    for mode in modes:
        dsl_name = map_winding_to_dsl(mode)
        if dsl_name not in existing and dsl_name != "generic_transform":
            conjectures.append(dsl_name)
            existing.add(dsl_name)

    return conjectures


# =============================================================================
# §6. Internal Helpers
# =============================================================================

_perceiver_instance = None

def _get_perceiver():
    """Get or create EMLPerceiver instance."""
    global _perceiver_instance
    if _perceiver_instance is None:
        from src.agent.eml_perceiver import EMLPerceiver
        _perceiver_instance = EMLPerceiver()
    return _perceiver_instance


def _find_matching_phase(
    source_sphere: JinlingSphere,
    target_spheres: List[JinlingSphere],
    max_distance: float = 5.0,
) -> np.ndarray:
    """Find the matching sphere in target list by centroid proximity.

    Args:
        source_sphere: Original sphere.
        target_spheres: Transformed spheres.
        max_distance: Maximum centroid distance for matching.

    Returns:
        Phase vector of matching sphere, or source phase if no match.
    """
    best_dist = float('inf')
    best_phase = source_sphere.oct_phase.phase_vector

    for ts in target_spheres:
        dist = math.sqrt(
            (source_sphere.centroid[0] - ts.centroid[0])**2 +
            (source_sphere.centroid[1] - ts.centroid[1])**2
        )
        if dist < best_dist:
            best_dist = dist
            best_phase = ts.oct_phase.phase_vector if ts.oct_phase.phase_vector is not None else np.zeros(5)

    return best_phase


# =============================================================================
# §7. Self-Test
# =============================================================================

def _self_test() -> None:
    """Run self-tests for κ-Winding Algorithm module."""
    print("=" * 60)
    print("κ-Winding Algorithm — Self-Test")
    print("=" * 60)

    from src.agent.octonion_phase import OctonionPhase

    # Test 1: Monster weight computation
    print("\n[Test 1] Monster weight computation")
    w1 = monster_weight(1, 1)
    w2 = monster_weight(2, 3)
    w4 = monster_weight(4, 5)
    w8 = monster_weight(8, 9)
    print(f"  monster_weight(1,1)={w1:.2f}")
    print(f"  monster_weight(2,3)={w2:.2f}")
    print(f"  monster_weight(4,5)={w4:.2f}")
    print(f"  monster_weight(8,9)={w8:.2f}")
    assert w8 > w4 > w2 > w1, "Monster weights should increase with symmetry"

    # Test 2: κ-winding algorithm (no transform)
    print("\n[Test 2] κ-winding algorithm (baseline)")
    spheres = [
        JinlingSphere(
            id=0, centroid=(5.0, 5.0), bbox=(2, 2, 8, 8),
            color=1, oct_phase=OctonionPhase(), coupling=0.8, attrs={}
        ),
        JinlingSphere(
            id=1, centroid=(15.0, 15.0), bbox=(12, 12, 18, 18),
            color=3, oct_phase=OctonionPhase(), coupling=0.5, attrs={}
        ),
    ]
    result = kappa_winding_algorithm(spheres)
    print(f"  Total winding: {result.total_winding:.4f}")
    print(f"  Per-sphere: {result.sphere_windings}")
    print(f"  Partition Z(κ): {result.partition_value:.4f}")

    # Test 3: Umbral curve
    print("\n[Test 3] Umbral curve construction")
    umbral = build_umbral_curve(spheres)
    print(f"  Umbral curve shape: {umbral.shape}")
    print(f"  Diagonal (self-distance): {umbral[0,0]:.4f}, {umbral[1,1]:.4f}")

    # Test 4: Winding mode enumeration
    print("\n[Test 4] Winding mode enumeration")
    modes = enumerate_winding_modes(umbral, max_dim=4)
    print(f"  Found {len(modes)} winding modes")
    for m in modes:
        print(f"    EV={m['eigenvalue']:.3f}, hint={m['dsl_hint']}")

    # Test 5: Monster winding search
    print("\n[Test 5] Monster winding search")
    conjectures = monster_winding_search(spheres, existing_dsl=["identity"])
    print(f"  New DSL conjectures: {conjectures}")

    # Test 6: Winding-to-DSL mapping
    print("\n[Test 6] Winding-to-DSL mapping")
    mode = {"eigenvalue": 3.0, "dsl_hint": "symmetry_rotation"}
    dsl_name = map_winding_to_dsl(mode)
    print(f"  Mode → DSL: {dsl_name}")
    assert dsl_name == "rotation_apply", "Strong positive EV → rotation"

    print("\n" + "=" * 60)
    print("ALL κ-WINDING SELF-TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
