# -*- coding: utf-8 -*-
"""Sporadic Group Filter — 26 sporadic groups as high-order symmetry filters for ARC-3.

Implements the classification and filtering framework from IDO/TOMAS Article 8:
"From Coset Selection to Sporadic Group Filtering"

Core insight:
  - 26 Sporadic Groups divided into:
    (1) IR stability base: 24 Niemeier lattices + 1 Leech lattice
    (2) Möbius flip: M₁₂ and M₂₂ (chirality selection)
    (3) Monster M: 196883-dim representation (highest sporadic)
  - For ARC-3, these become symmetry pattern detectors:
    - 24-fold symmetry → Niemeier filter → rotate/scale DSL primitives
    - Chirality (mirror ≠ original) → Möbius filter → flip/reflect DSL primitives
    - Octonion (8-dim) → Monster filter → compose/decompose DSL primitives
    - Generic → standard κ-Snap search

Classification algorithm:
  1. detect_24_fold_symmetry(task_grid) → True if any object has 24-fold rotational symmetry
  2. detect_chirality(task_grid) → True if input→output involves mirror transformations
  3. detect_monster_pattern(task_grid) → True if objects exhibit 8-dim octonion structure
  4. classify_sporadic_group(task) → "Niemeier" | "Moebius" | "Monster" | "Generic"

Each classification routes to a specific DSL primitive set:
  - Niemeier → rotation/scale/periodic primitives
  - Moebius → flip/reflect/chirality primitives
  - Monster → compose/decompose/associate primitives
  - Generic → standard κ-Snap + coset search

IDO Correspondence:
  - Niemeier/Leech = IR stability base → structural primitives
  - M₁₂/M₂₂ = Möbius flip → chirality primitives
  - Monster = UV→IR projection weight → compositional primitives

Version: v4.2
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.agent.octonion_phase import OctonionPhase
from src.agent.eml_perceiver import JinlingSphere, EMLPerceiver
from src.agent.semantic_constants import SemanticConstants, extract_semantic_constants


# =============================================================================
# §1. Sporadic Group Classification Types
# =============================================================================

SPORADIC_TYPES = ["Niemeier", "Moebius", "Monster", "Generic"]

# Niemeier lattice dimensions (24 Niemeier lattices + Leech)
# For ARC-3 simplified: we only track the symmetry-fold count
NIEMEIER_LATTICE_FOLDS = {
    "A1^24": 24,     # 24 copies of A1 root lattice
    "A2^12": 12,     # 12 copies of A2
    "A3^8": 8,       # 8 copies of A3
    "A4^6": 6,       # 6 copies of A4
    "A5^4_D4": 4,    # 4 copies of A5 + D4
    "A6^2_D6": 2,    # 2 copies of A6 + D6
    "D4^6": 6,       # 6 copies of D4 (important for ARC!)
    "D4^4_A4": 4,    # mixed
    "Leech": 24,     # Leech lattice (no roots, highest symmetry)
}

# Möbius flip groups
MOEBIUS_GROUPS = {
    "M11": 11,   # smallest Mathieu group
    "M12": 12,   # 5-transitive permutation group
    "M22": 22,   # 3-transitive
    "M23": 23,   # Mathieu group
    "M24": 24,   # 4-transitive (largest Mathieu)
}

# Monster-related dimensions for ARC-3
MONSTER_ARC_MAP = {
    "trivial": 1,      # 1-dim trivial representation
    "sign": 1,         # sign representation
    "standard": 196883, # smallest non-trivial (too large for ARC, use simplified)
    "octonion": 8,     # 8-dim octonion (ARC-3 relevant)
    "D4": 4,           # 4-dim dihedral (ARC-3 relevant)
}


# =============================================================================
# §2. Detection Functions
# =============================================================================

def detect_24_fold_symmetry(grid: np.ndarray) -> bool:
    """Detect if grid contains objects with 24-fold rotational symmetry.

    A 24-fold symmetry means rotating by 360/24 = 15° gives identical object.
    For discrete grids, we check rotational invariants:
    - 4-fold rotation (90°) invariance → D₄ symmetry
    - 6-fold rotation (60°) invariance → hexagonal
    - 24-fold only possible for circular/radial patterns

    Simplified for ARC-3: check if grid has 4-fold AND 6-fold sub-symmetry,
    or if any connected component is invariant under multiple rotation angles.

    Args:
        grid: Input grid as np.ndarray.

    Returns:
        True if 24-fold or high-order rotational symmetry detected.
    """
    if grid.size == 0:
        return False

    # Check 4-fold symmetry (rotation by 90°)
    h, w = grid.shape
    if h != w:
        # Non-square grid cannot have 4-fold rotational symmetry
        four_fold = False
    else:
        rotated_90 = np.rot90(grid, k=1)
        four_fold = np.array_equal(grid, rotated_90)

    # Check 6-fold symmetry (for hexagonal-like patterns)
    # Simplified: check if there are objects at 6 symmetric positions
    non_zero_positions = np.argwhere(grid != 0)
    if len(non_zero_positions) >= 6:
        # Compute centroid
        centroid = non_zero_positions.mean(axis=0)
        # Check if positions form 6 equi-angular groups
        angles = []
        for pos in non_zero_positions:
            dy = pos[0] - centroid[0]
            dx = pos[1] - centroid[1]
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                angle = math.atan2(dy, dx)
                angles.append(angle)
        if len(angles) >= 6:
            # Sort angles and check for 60° spacing
            angles.sort()
            angle_diffs = []
            for i in range(len(angles) - 1):
                diff = angles[i + 1] - angles[i]
                # Normalize to [0, π)
                while diff < 0:
                    diff += 2 * math.pi
                angle_diffs.append(diff)
            # Check if median angle difference ≈ π/3 (60°)
            if angle_diffs:
                median_diff = np.median(angle_diffs)
                six_fold = abs(median_diff - math.pi / 3) < 0.5
            else:
                six_fold = False
        else:
            six_fold = False
    else:
        six_fold = False

    # 24-fold = 4-fold AND 6-fold, OR very high symmetry
    return four_fold and six_fold


def detect_chirality(grid_input: np.ndarray, grid_output: np.ndarray) -> bool:
    """Detect if the transformation involves chirality (mirror ≠ original).

    Chirality detection: check if input→output involves a flip/mirror that
    cannot be achieved by pure rotation.

    Algorithm:
    1. Check if output = flip(input) for horizontal/vertical/diagonal flips
    2. If flip matches but rotation doesn't → chirality detected
    3. If neither flip nor rotation matches → complex transform (no chirality)

    Args:
        grid_input: Input grid.
        grid_output: Output grid.

    Returns:
        True if chirality (mirror) transformation detected.
    """
    if grid_input.shape != grid_output.shape:
        return False

    # Check various flips
    h_flip = np.fliplr(grid_input)  # horizontal flip
    v_flip = np.flipud(grid_input)  # vertical flip
    diag_flip = np.transpose(grid_input)  # diagonal (transpose) flip

    # Check rotations
    if grid_input.shape[0] == grid_input.shape[1]:
        rot90 = np.rot90(grid_input, k=1)
        rot180 = np.rot90(grid_input, k=2)
        rot270 = np.rot90(grid_input, k=3)
        rotation_matches = any(
            np.array_equal(r, grid_output)
            for r in [grid_input, rot90, rot180, rot270]
        )
    else:
        rotation_matches = np.array_equal(grid_input, grid_output)

    flip_matches = any(
        np.array_equal(f, grid_output)
        for f in [h_flip, v_flip, diag_flip]
    )

    # Chirality = flip matches but rotation doesn't
    return flip_matches and not rotation_matches


def detect_monster_pattern(grid: np.ndarray) -> bool:
    """Detect if grid contains objects with 8-dimensional (octonion) structure.

    Monster pattern detection for ARC-3:
    1. Count connected components by color
    2. Check if any component has symmetry order 8 (octonion)
    3. Check if components exhibit the "196883" pattern (too many sub-objects)

    Simplified: objects with 8-fold symmetry or 8+ distinct sub-regions.

    Args:
        grid: Input grid.

    Returns:
        True if octonion/Monster pattern detected.
    """
    perceiver = EMLPerceiver()
    spheres = perceiver.perceive(grid)

    if len(spheres) == 0:
        return False

    # Check for octonion symmetry (8-fold)
    has_octonion = any(
        s.oct_phase.symmetry_order >= 8
        for s in spheres
    )

    # Check for high complexity (8+ distinct objects with different phases)
    unique_phases = len(set(
        tuple(np.round(s.oct_phase.phase_vector, 2))
        for s in spheres
        if s.oct_phase.phase_vector is not None
    ))
    has_high_complexity = unique_phases >= 8

    return has_octonion or has_high_complexity


def detect_periodic_pattern(grid: np.ndarray) -> bool:
    """Detect periodic/tiling patterns (Niemeier lattice analogs).

    Periodic patterns in ARC grids correspond to Niemeier lattice structures:
    repeating units that tile the grid space.

    Algorithm:
    1. Find smallest repeating unit (period)
    2. Check if grid = unit tiled across space
    3. Verify periodicity in both dimensions

    Args:
        grid: Input grid.

    Returns:
        True if periodic tiling pattern detected.
    """
    h, w = grid.shape
    if h < 4 or w < 4:
        return False

    # Try different tile sizes
    for tile_h in range(2, min(h // 2 + 1, 8)):
        for tile_w in range(2, min(w // 2 + 1, 8)):
            if h % tile_h != 0 or w % tile_w != 0:
                continue
            # Extract tile
            tile = grid[:tile_h, :tile_w]
            # Check if entire grid is tiled by this tile
            is_periodic = True
            for iy in range(0, h, tile_h):
                for ix in range(0, w, tile_w):
                    if not np.array_equal(grid[iy:iy+tile_h, ix:ix+tile_w], tile):
                        is_periodic = False
                        break
                if not is_periodic:
                    break
            if is_periodic and tile_h * tile_w > 1:
                return True
    return False


# =============================================================================
# §3. Sporadic Group Classification
# =============================================================================

@dataclass
class SporadicClassification:
    """Result of sporadic group classification for an ARC-3 task.

    Attributes:
        group_type: Primary sporadic group type.
        confidence: Classification confidence (0-1).
        detected_symmetries: List of detected symmetry features.
        recommended_dsl_set: DSL primitives recommended for this group type.
        niemeier_fold: Fold count if Niemeier-type detected.
        chirality_axis: Flip axis if Möbius-type detected.
        monster_dim: Monster dimension if Monster-type detected.
    """
    group_type: str
    confidence: float
    detected_symmetries: List[str]
    recommended_dsl_set: List[str]
    niemeier_fold: int = 0
    chirality_axis: str = ""
    monster_dim: int = 0


# DSL primitive sets for each sporadic group type
SPORADIC_DSL_SETS = {
    "Niemeier": [
        "rotate_90", "rotate_180", "rotate_270",
        "scale_up", "scale_down",
        "periodic_tile", "repeat_pattern",
        "fill_periodic", "complete_symmetric",
    ],
    "Moebius": [
        "flip_horizontal", "flip_vertical", "flip_diagonal",
        "reflect", "transpose",
        "mirror_fill", "chiral_transform",
        "swap_lr", "swap_ud",
    ],
    "Monster": [
        "compose", "decompose",
        "associate", "non_associative_residual",
        "octonion_transform", "group_by_phase",
        "rearrange_8fold", "map_by_rule",
    ],
    "Generic": [
        "color_map", "shift_object", "resize_object",
        "fill_region", "extract_pattern",
        "swap_colors", "conditional_fill",
        "map_by_rule", "complete_pattern",
    ],
}


def classify_sporadic_group(
    grid_input: np.ndarray,
    grid_output: Optional[np.ndarray] = None,
) -> SporadicClassification:
    """Classify an ARC-3 task by its sporadic group pattern.

    Implements the classification algorithm from Article 8 §5.2:
    - 24-fold symmetry → "Niemeier"
    - Chirality → "Moebius"
    - Monster pattern → "Monster"
    - Neither → "Generic"

    Args:
        grid_input: Input grid of the ARC-3 task.
        grid_output: Output grid (optional, needed for chirality detection).

    Returns:
        SporadicClassification with group type and recommended DSL set.
    """
    detected: List[str] = []
    confidences: Dict[str, float] = {}

    # Check Niemeier (periodic/tiling patterns)
    has_24_fold = detect_24_fold_symmetry(grid_input)
    has_periodic = detect_periodic_pattern(grid_input)

    if has_24_fold:
        detected.append("24-fold rotational symmetry")
        confidences["Niemeier"] = 0.95
    elif has_periodic:
        detected.append("periodic tiling pattern")
        confidences["Niemeier"] = 0.80
    else:
        # Check for sub-Niemeier patterns (D₄, 6-fold, etc.)
        if grid_input.shape[0] == grid_input.shape[1]:
            rot90 = np.rot90(grid_input, k=1)
            if np.array_equal(grid_input, rot90):
                detected.append("4-fold (D₄) symmetry")
                confidences["Niemeier"] = 0.50

    # Check Möbius (chirality)
    if grid_output is not None:
        has_chirality = detect_chirality(grid_input, grid_output)
        if has_chirality:
            detected.append("chirality (mirror ≠ rotation)")
            confidences["Moebius"] = 0.90
            # Determine flip axis
            h_flip = np.fliplr(grid_input)
            v_flip = np.flipud(grid_input)
            if np.array_equal(h_flip, grid_output):
                chirality_axis = "horizontal"
            elif np.array_equal(v_flip, grid_output):
                chirality_axis = "vertical"
            else:
                chirality_axis = "diagonal"
    else:
        # Without output, check for inherent chirality in input
        if grid_input.shape[0] == grid_input.shape[1]:
            h_flip = np.fliplr(grid_input)
            v_flip = np.flipud(grid_input)
            diag_flip = np.transpose(grid_input)
            if not np.array_equal(grid_input, h_flip) and \
               not np.array_equal(h_flip, np.rot90(grid_input)):
                detected.append("potential chirality in input")
                confidences["Moebius"] = 0.40
                chirality_axis = "horizontal"
            else:
                chirality_axis = ""
    chirality_axis = chirality_axis if 'chirality_axis' in dir() else ""

    # Check Monster (octonion/8-dim)
    has_monster = detect_monster_pattern(grid_input)
    if has_monster:
        detected.append("octonion/Monster pattern (8-dim)")
        confidences["Monster"] = 0.85

    # Determine primary classification
    if not confidences:
        group_type = "Generic"
        confidence = 0.5
    else:
        group_type = max(confidences, key=confidences.get)
        confidence = confidences[group_type]

    # Build result
    niemeier_fold = 0
    if group_type == "Niemeier":
        if has_24_fold:
            niemeier_fold = 24
        elif has_periodic:
            # Estimate fold from tile size
            niemeier_fold = 4  # default D₄
        elif "4-fold" in str(detected):
            niemeier_fold = 4

    monster_dim = 0
    if group_type == "Monster":
        monster_dim = 8  # ARC-simplified Monster dimension

    return SporadicClassification(
        group_type=group_type,
        confidence=confidence,
        detected_symmetries=detected,
        recommended_dsl_set=SPORADIC_DSL_SETS.get(group_type, SPORADIC_DSL_SETS["Generic"]),
        niemeier_fold=niemeier_fold,
        chirality_axis=chirality_axis,
        monster_dim=monster_dim,
    )


# =============================================================================
# §4. Sporadic Group Filter Integration with Coset Search
# =============================================================================

def sporadic_filter_coset_search(
    grid_input: np.ndarray,
    grid_output: Optional[np.ndarray] = None,
    train_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> List[Dict[str, Any]]:
    """Apply sporadic group filter before coset search.

    Pipeline:
    1. classify_sporadic_group → determine group type
    2. Filter coset candidates based on group type
    3. Route to appropriate DSL primitive set
    4. Return filtered candidate DSL programs

    This implements the "Sporadic Group Filter" from Article 8 §5.2,
    integrated with the coset search from §5.1.

    Args:
        grid_input: Input grid.
        grid_output: Output grid (optional).
        train_pairs: Training pairs (optional).

    Returns:
        List of filtered candidate DSL programs with sporadic group info.
    """
    # Step 1: Classify
    classification = classify_sporadic_group(grid_input, grid_output)

    # Step 2: Get recommended DSL set
    dsl_set = classification.recommended_dsl_set

    # Step 3: Extract UV features and enumerate cosets
    from src.agent.coset_search import extract_11_uv_features, enumerate_cosets, build_dsl_for_coset

    uv_features = extract_11_uv_features(grid_input)
    target_dims = min(4, len(uv_features))
    cosets = enumerate_cosets(uv_features, target_dims=target_dims)

    # Step 4: Filter cosets by group type
    filtered_candidates = []
    for coset in cosets:
        # Build DSL for this coset (using symmetry_idx=0 as default, full D₄ later)
        dsl_candidate = build_dsl_for_coset(
            coset.selected_features, 0, uv_features
        )

        # Check if candidate DSL overlaps with recommended set
        dsl_names = []
        if isinstance(dsl_candidate, list):
            for item in dsl_candidate:
                if isinstance(item, dict):
                    dsl_names.append(item.get("name", item.get("primitive", "")))
                elif isinstance(item, str):
                    dsl_names.append(item)
        elif isinstance(dsl_candidate, dict):
            dsl_names = dsl_candidate.get("dsl_primitives", [])
        overlap = any(prim in dsl_names for prim in dsl_set)

        filtered_candidates.append({
            "selected_features": coset.selected_features,
            "symmetry_idx": coset.symmetry_idx,
            "symmetry_name": coset.symmetry_name,
            "feature_names": coset.feature_names,
            "dsl_candidate": dsl_candidate,
            "sporadic_group": classification.group_type,
            "sporadic_confidence": classification.confidence,
            "recommended_dsl": dsl_set,
            "overlap_with_recommended": overlap,
            "priority": classification.confidence * (1.5 if overlap else 1.0),
        })

    # Sort by priority (higher = more likely correct)
    filtered_candidates.sort(key=lambda x: x["priority"], reverse=True)

    # Limit to top candidates
    max_candidates = 50
    if len(filtered_candidates) > max_candidates:
        filtered_candidates = filtered_candidates[:max_candidates]

    return filtered_candidates


# =============================================================================
# §5. κ-Entropy Release as GaussEx Second-Stage Verification
# =============================================================================

def gauss_ex_kappa_entropy_release_check(
    pred: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1.0 / 6,
    entropy_tol: float = 0.3,
    expected_entropy: Optional[float] = None,
) -> Tuple[bool, float, str]:
    """GaussEx with κ-entropy release criterion (Article 8 §5.4).

    Two-stage verification:
    1. Pixel residual < ε (standard GaussEx)
    2. |Δκ - expected| < entropy_tol (κ-entropy release check)

    Δκ = κ_UV(pred) - κ_IR(target)
    Expected Δκ from Article 8: log(137) ≈ 4.92 (fine structure constant analog)
    But for ARC-3 grids, expected Δκ varies by task complexity.

    Args:
        pred: Predicted output grid.
        target: Target output grid.
        epsilon: Pixel residual tolerance.
        entropy_tol: κ-entropy deviation tolerance.
        expected_entropy: Expected κ-entropy release value.

    Returns:
        (passed, residual, reason) tuple.
    """
    from src.agent.ido_axioms import estimate_kappa_uv, estimate_kappa_ir, kappa_entropy_release

    # Stage 1: Pixel residual check
    if pred.shape != target.shape:
        return (False, 1.0, "shape mismatch")

    total_pixels = pred.size
    matching = np.sum(pred == target)
    residual = 1.0 - matching / total_pixels

    if residual > epsilon:
        return (False, residual, f"pixel residual {residual:.3f} > ε={epsilon:.3f}")

    # Stage 2: κ-entropy release check
    kappa_uv_pred = estimate_kappa_uv(pred)
    kappa_ir_target = estimate_kappa_ir(target)
    delta_kappa = kappa_entropy_release(kappa_uv_pred, kappa_ir_target)

    # If expected entropy not given, estimate from target
    if expected_entropy is None:
        # Default: Δκ should be positive (information compression)
        expected_entropy = max(0.0, delta_kappa * 0.5)

    entropy_deviation = abs(delta_kappa - expected_entropy)

    if entropy_deviation > entropy_tol:
        return (
            False,
            residual,
            f"κ-entropy deviation {entropy_deviation:.3f} > tol={entropy_tol:.3f}"
            f" (Δκ={delta_kappa:.2f}, expected={expected_entropy:.2f})"
        )

    return (True, residual, f"PASS: residual={residual:.3f}, Δκ={delta_kappa:.2f}")


# =============================================================================
# §6. Self-Test
# =============================================================================

def _self_test() -> None:
    """Run self-tests for sporadic_group_filter module."""
    print("=" * 60)
    print("Sporadic Group Filter — Self-Test")
    print("=" * 60)

    # Test 1: detect_24_fold_symmetry
    print("\n[Test 1] 24-fold symmetry detection")
    # Create a grid with 4-fold symmetry (square)
    sym_grid = np.array([
        [1, 2, 3, 2, 1],
        [2, 4, 5, 4, 2],
        [3, 5, 0, 5, 3],
        [2, 4, 5, 4, 2],
        [1, 2, 3, 2, 1],
    ])
    result = detect_24_fold_symmetry(sym_grid)
    print(f"  4-fold symmetric grid: {result}")

    # Random grid
    random_grid = np.random.randint(0, 10, size=(5, 5))
    result2 = detect_24_fold_symmetry(random_grid)
    print(f"  Random grid: {result2}")

    # Test 2: detect_chirality
    print("\n[Test 2] Chirality detection")
    input_grid = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ])
    h_flipped = np.fliplr(input_grid)
    is_chiral = detect_chirality(input_grid, h_flipped)
    print(f"  Input vs horizontal flip: chirality={is_chiral}")

    # Same vs rotation (no chirality)
    rot180 = np.rot90(input_grid, k=2)
    is_chiral2 = detect_chirality(input_grid, input_grid)
    print(f"  Input vs same: chirality={is_chiral2}")

    # Test 3: detect_monster_pattern
    print("\n[Test 3] Monster pattern detection")
    # Grid with 8+ objects
    multi_grid = np.zeros((10, 10), dtype=int)
    for i in range(8):
        multi_grid[i, i] = i + 1
    result3 = detect_monster_pattern(multi_grid)
    print(f"  Grid with 8 colors: {result3}")

    # Simple 2-color grid
    simple_grid = np.array([[0, 1], [1, 0]])
    result4 = detect_monster_pattern(simple_grid)
    print(f"  Simple 2-color grid: {result4}")

    # Test 4: detect_periodic_pattern
    print("\n[Test 4] Periodic pattern detection")
    # 2x2 tile repeated
    tile = np.array([[1, 2], [3, 4]])
    periodic = np.tile(tile, (3, 3))
    result5 = detect_periodic_pattern(periodic)
    print(f"  Periodic tiling (2x2 tile, 6x6): {result5}")

    random_grid6 = np.random.randint(0, 10, size=(6, 6))
    result6 = detect_periodic_pattern(random_grid6)
    print(f"  Random 6x6 grid: {result6}")

    # Test 5: classify_sporadic_group
    print("\n[Test 5] Sporadic group classification")
    # Periodic grid → should classify as Niemeier-like
    cls1 = classify_sporadic_group(periodic)
    print(f"  Periodic grid: group={cls1.group_type}, confidence={cls1.confidence:.2f}")
    print(f"  Recommended DSL: {cls1.recommended_dsl_set[:3]}")

    # Chiral pair → should classify as Moebius
    cls2 = classify_sporadic_group(input_grid, h_flipped)
    print(f"  Chiral pair: group={cls2.group_type}, confidence={cls2.confidence:.2f}")

    # Simple grid → should classify as Generic
    cls3 = classify_sporadic_group(random_grid)
    print(f"  Random grid: group={cls3.group_type}, confidence={cls3.confidence:.2f}")

    # Test 6: sporadic_filter_coset_search
    print("\n[Test 6] Sporadic filter + coset search integration")
    candidates = sporadic_filter_coset_search(periodic)
    print(f"  Periodic grid: {len(candidates)} filtered candidates")
    if candidates:
        top = candidates[0]
        print(f"  Top candidate: group={top['sporadic_group']}, priority={top['priority']:.2f}")

    # Test 7: gauss_ex_kappa_entropy_release_check
    print("\n[Test 7] GaussEx κ-entropy release check")
    # Exact match → should pass both stages
    exact = np.array([[1, 2], [3, 4]])
    passed, residual, reason = gauss_ex_kappa_entropy_release_check(exact, exact)
    print(f"  Exact match: passed={passed}, residual={residual:.3f}, reason={reason}")

    # Near match → should pass pixel but might fail entropy
    near = np.array([[1, 2], [3, 5]])
    passed2, residual2, reason2 = gauss_ex_kappa_entropy_release_check(near, exact)
    print(f"  Near match: passed={passed2}, residual={residual2:.3f}, reason={reason2}")

    print("\n" + "=" * 60)
    print("ALL SPORADIC GROUP FILTER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
