# -*- coding: utf-8 -*-
"""RG Flow Simulator — UV→IR information collapse simulation.

Simulates the RG (Renormalization Group) flow from UV state (high entropy,
many features) to IR state (low entropy, high structure), analogous to
the dimension selection in M-theory (11→4 dimensions).

Core physics analogy:
  - UV state = input grid (high information cardinality κ_UV)
  - IR state = output grid (low information cardinality κ_IR)
  - RG flow = κ-Snap search (information collapse)
  - β function = flow rate (κ decreases monotonically)
  - Fixed point = IR state (κ stops decreasing → solution found)

The β function β(κ) = -κ × log(κ/κ_IR) / κ_UV describes the rate of
information compression. It vanishes at κ = κ_IR (the IR fixed point),
which is the "mass gap" — the solution has been found.

IDO Correspondence:
  - RG flow = IDO Δκ-constraint enforcement
  - β function = IDO information compression rate
  - Fixed point = IDO κ_IR target (4 macro dimensions)

TOMAS Correspondence:
  - RG collapse = TOMAS Phase I→III pipeline
  - β(κ) = TOMAS κ-gradient descent rate
  - IR candidate = TOMAS ProgramNode solution

Version: v4.2
"""

from __future__ import annotations

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.eml_perceiver import JinlingSphere, EMLPerceiver
from src.agent.ido_axioms import (
    IDOConfig,
    estimate_kappa_uv,
    estimate_kappa_ir,
    kappa_entropy_release,
    verify_anti_monotonicity,
    estimate_coset_count,
)
from src.agent.semantic_constants import SemanticConstants, extract_semantic_constants


# =============================================================================
# §1. RGFlowState Dataclass
# =============================================================================

@dataclass
class RGFlowState:
    """State of the RG flow at a particular step.

    Attributes:
        uv_state: UV (input) feature representation.
        ir_state: IR (output) collapsed feature representation.
        kappa_uv: Information cardinality at UV.
        kappa_ir: Information cardinality at IR.
        compression_ratio: κ_UV / κ_IR (analogy: fine structure 137).
        phase_trajectory: List of κ values along the RG flow path.
        beta_function: β(κ) = flow rate at current step.
        current_step: Current RG flow step (0 = UV, N = IR).
    """
    uv_state: np.ndarray
    ir_state: np.ndarray
    kappa_uv: float
    kappa_ir: float
    compression_ratio: float
    phase_trajectory: List[float]
    beta_function: float
    current_step: int


# =============================================================================
# §2. RGFlowSimulator Core
# =============================================================================

class RGFlowSimulator:
    """RG Flow Simulator — collapse UV state to IR state.

    Simulates the renormalization group flow from a high-dimensional UV
    state (input grid) to a low-dimensional IR state (output grid). The
    flow follows the β function β(κ) which ensures κ decreases
    monotonically (IDO anti-monotonicity axiom).

    The simulation proceeds by iteratively merging the most similar
    feature clusters until κ reaches the IR target (κ_IR). Each merge
    step corresponds to one unit of RG flow time.

    Attributes:
        target_dim: IR dimensionality (default 4, analogy: macro spacetime).
        max_steps: Maximum RG flow steps (default 10).
    """

    def __init__(self, target_dim: int = 4, max_steps: int = 10) -> None:
        """Initialize RG Flow Simulator.

        Args:
            target_dim: Target IR dimensionality (analogy: 4 macro dims).
            max_steps: Maximum number of RG flow collapse steps.
        """
        self.target_dim = target_dim
        self.max_steps = max_steps

    def collapse(
        self,
        uv_features: np.ndarray,
        steps: Optional[int] = None,
    ) -> RGFlowState:
        """RG flow collapse: UV → IR.

        Simulates information cardinality from κ_UV to κ_IR:
          1. Initial: κ_uv = log2(n_unique_features)
          2. Each step: β(κ) = -dκ/dt, κ decreases
          3. Target: κ_ir ≈ log2(target_dim²)
          4. Collapse path: merge closest features iteratively
          5. Final: IR state = minimal κ description

        Args:
            uv_features: UV feature array (n_features × feature_dim).
            steps: Number of collapse steps (default: max_steps).

        Returns:
            RGFlowState describing the collapse trajectory.
        """
        if steps is None:
            steps = self.max_steps

        # Initial κ values
        kappa_uv = estimate_kappa_uv_from_features(uv_features)
        kappa_ir = math.log2(self.target_dim ** 2) if self.target_dim > 0 else 0.0

        # Phase trajectory (κ at each step)
        trajectory: List[float] = []
        current_features = uv_features.copy() if isinstance(uv_features, np.ndarray) else np.array(uv_features)
        current_kappa = kappa_uv

        for step in range(steps):
            trajectory.append(current_kappa)

            # β function: flow rate
            beta = self._beta_function(current_kappa, kappa_uv, kappa_ir)

            # κ update: κ_new = κ_old + β × dt (dt=1)
            current_kappa = current_kappa + beta
            if current_kappa < kappa_ir:
                current_kappa = kappa_ir  # Fixed point reached

            # Feature merge: combine closest clusters
            if current_features.shape[0] > self.target_dim:
                current_features = self._merge_features(current_features, current_kappa)

            # Stop at fixed point
            if abs(current_kappa - kappa_ir) < 0.01:
                trajectory.append(current_kappa)
                break

        trajectory.append(current_kappa)

        # Compression ratio
        ratio = kappa_uv / kappa_ir if kappa_ir > 0 else float('inf')

        return RGFlowState(
            uv_state=uv_features,
            ir_state=current_features,
            kappa_uv=kappa_uv,
            kappa_ir=current_kappa,
            compression_ratio=ratio,
            phase_trajectory=trajectory,
            beta_function=self._beta_function(current_kappa, kappa_uv, kappa_ir),
            current_step=len(trajectory) - 1,
        )

    def _beta_function(
        self,
        kappa: float,
        kappa_uv: float,
        kappa_ir: float,
    ) -> float:
        """β function: RG flow rate.

        β(κ) = -κ × log(κ/κ_IR) / κ_UV

        This describes the rate of information compression:
          - β > 0 when κ > κ_IR → κ decreases (flow toward IR)
          - β ≈ 0 when κ ≈ κ_IR → fixed point reached
          - β < 0 when κ < κ_IR → flow reversal (violates IDO!)

        The anti-monotonicity axiom ensures β > 0 always (κ must decrease).

        Args:
            kappa: Current κ value.
            kappa_uv: UV κ (initial value).
            kappa_ir: IR κ (target value).

        Returns:
            β(κ) value (negative = κ decreases per step).
        """
        if kappa_uv <= 0 or kappa <= 0 or kappa_ir <= 0:
            return 0.0

        ratio = kappa / kappa_ir
        if ratio <= 1.0:
            return 0.0  # Already at fixed point

        # β = -κ × log(κ/κ_IR) / κ_UV
        beta = -kappa * math.log(ratio) / kappa_uv
        return beta

    def _merge_features(
        self,
        features: np.ndarray,
        kappa: float,
    ) -> np.ndarray:
        """Merge the closest pair of feature clusters.

        Finds the two most similar features (by Euclidean distance)
        and merges them into a single cluster (average of the two).

        This simulates the RG flow "renormalization" step: at each
        energy scale, features that are indistinguishable at that scale
        get merged together.

        Args:
            features: Current feature array (n × dim).
            kappa: Current κ value (controls merge aggressiveness).

        Returns:
            Merged feature array (n-1 × dim).
        """
        n = features.shape[0]
        if n <= 1:
            return features

        # Find closest pair
        min_dist = float('inf')
        min_pair = (0, 1)

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(features[i] - features[j])
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)

        # Merge pair → new cluster = average
        i, j = min_pair
        merged = (features[i] + features[j]) / 2.0

        # Remove old, add new
        new_features = []
        for k in range(n):
            if k != i and k != j:
                new_features.append(features[k])
        new_features.append(merged)

        return np.array(new_features)

    def solve_with_rg_flow(
        self,
        input_grid: np.ndarray,
        spheres: Optional[List[JinlingSphere]] = None,
        constants: Optional[SemanticConstants] = None,
        n_paths: int = 3,
    ) -> List[RGFlowState]:
        """RG flow solving: generate multiple IR candidate states.

        1. UV state = EML perceived features
        2. Multiple collapse paths → generate n_paths IR candidates
        3. Each candidate corresponds to a possible output structure
        4. Ranked by compression_ratio (closer to 137 = optimal)

        Args:
            input_grid: Input ARC grid.
            spheres: EML perceived spheres (optional, auto-compute).
            constants: Semantic constants (optional, auto-compute).
            n_paths: Number of collapse paths to generate.

        Returns:
            List of RGFlowState candidates, ranked by compression_ratio.
        """
        # Perceive if needed
        if spheres is None:
            perceiver = EMLPerceiver()
            spheres = perceiver.perceive(input_grid)

        if constants is None:
            constants = extract_semantic_constants(spheres, input_grid)

        # Extract UV features
        uv_features = self._extract_uv_from_spheres(spheres, input_grid)

        # Generate multiple collapse paths (vary target_dim)
        candidates: List[RGFlowState] = []
        for target_dim in range(2, min(6, uv_features.shape[0] + 1)):
            sim = RGFlowSimulator(target_dim=target_dim, max_steps=self.max_steps)
            state = sim.collapse(uv_features)
            candidates.append(state)

        # Sort by compression_ratio (closer to theoretical optimum ≈ 137/100)
        candidates.sort(
            key=lambda s: abs(s.compression_ratio - 1.37),  # 1.37 = α⁻¹/100
            reverse=False,  # Closer to optimum = better
        )

        return candidates[:n_paths]

    def _extract_uv_from_spheres(
        self,
        spheres: List[JinlingSphere],
        grid: np.ndarray,
    ) -> np.ndarray:
        """Extract UV feature matrix from JinlingSphere list and grid.

        Each sphere contributes a feature row with:
          [centroid_x, centroid_y, color, symmetry_order, coupling,
           filledness, convexity, main_axis_angle, aspect_ratio, area]

        Plus global grid features appended:
          [n_colors, grid_height, grid_width, entropy]

        Total: n_spheres × 10 + 1 × 4 global

        Args:
            spheres: JinlingSphere list.
            grid: Original grid.

        Returns:
            Feature matrix (n_features × feature_dim).
        """
        rows: List[np.ndarray] = []

        for s in spheres:
            row = np.array([
                s.centroid[0], s.centroid[1], float(s.color),
                float(s.oct_phase.symmetry_order), s.coupling,
                s.oct_phase.filledness, s.oct_phase.convexity,
                s.oct_phase.main_axis_angle, s.oct_phase.aspect_ratio,
                float(s.bbox[2] - s.bbox[0] + 1) * float(s.bbox[3] - s.bbox[1] + 1),  # area
            ])
            rows.append(row)

        # Global features
        n_colors = len(np.unique(grid))
        H, W = grid.shape
        entropy = _compute_entropy(grid)
        global_row = np.array([float(n_colors), float(H), float(W), entropy])
        rows.append(global_row)

        if not rows:
            return np.zeros((1, 10))

        return np.array(rows)


# =============================================================================
# §3. Helper Functions
# =============================================================================

def estimate_kappa_uv_from_features(features: np.ndarray) -> float:
    """Estimate κ_UV from feature array.

    κ_UV = log2(n_features × n_unique_values)

    Args:
        features: Feature matrix (n × dim).

    Returns:
        κ_UV estimate.
    """
    n_features = features.shape[0]
    if n_features <= 0:
        return 0.0

    # Count unique feature values (per-dimension)
    n_unique = 0
    for dim in range(features.shape[1]):
        n_unique += len(np.unique(features[:, dim]))

    kappa = math.log2(max(n_features * n_unique, 1))
    return kappa


def _compute_entropy(grid: np.ndarray) -> float:
    """Compute Shannon entropy of grid color distribution."""
    total = grid.size
    if total == 0:
        return 0.0

    entropy = 0.0
    for c in np.unique(grid):
        p = np.sum(grid == c) / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# =============================================================================
# §4. Self-Test
# =============================================================================

def _self_test() -> None:
    """Run self-tests for RG Flow Simulator module."""
    print("=" * 60)
    print("RG Flow Simulator — Self-Test")
    print("=" * 60)

    # Test 1: β function
    print("\n[Test 1] β function behavior")
    sim = RGFlowSimulator(target_dim=4)
    beta_at_uv = sim._beta_function(10.0, 10.0, 2.0)
    beta_at_fixed = sim._beta_function(2.0, 10.0, 2.0)
    print(f"  β(κ=10, κ_UV=10, κ_IR=2) = {beta_at_uv:.4f} (should be negative)")
    print(f"  β(κ=2, κ_UV=10, κ_IR=2) = {beta_at_fixed:.4f} (should be ≈0 at fixed point)")
    assert beta_at_uv < 0, "β should be negative when κ > κ_IR"
    assert abs(beta_at_fixed) < 0.01, "β ≈ 0 at fixed point"

    # Test 2: RG flow collapse
    print("\n[Test 2] RG flow collapse (UV → IR)")
    uv_features = np.array([
        [1.0, 2.0, 3.0],  # Feature cluster 1
        [1.1, 2.1, 3.1],  # Close to cluster 1 → will merge
        [10.0, 20.0, 30.0],  # Feature cluster 2
        [10.5, 20.5, 30.5],  # Close to cluster 2 → will merge
        [100.0, 200.0, 300.0],  # Feature cluster 3 (isolated)
    ])
    state = sim.collapse(uv_features, steps=5)
    print(f"  κ_UV = {state.kappa_uv:.2f}")
    print(f"  κ_IR = {state.kappa_ir:.2f}")
    print(f"  Compression ratio = {state.compression_ratio:.2f}")
    print(f"  Phase trajectory: {state.phase_trajectory[:5]}...")
    print(f"  Steps = {state.current_step}")
    assert state.kappa_uv > state.kappa_ir, "κ must decrease (anti-monotonicity)"

    # Test 3: Feature merging
    print("\n[Test 3] Feature merging")
    merged = sim._merge_features(uv_features, kappa=8.0)
    print(f"  Before: {uv_features.shape[0]} features")
    print(f"  After:  {merged.shape[0]} features (merged closest pair)")
    assert merged.shape[0] == uv_features.shape[0] - 1, "Should merge one pair"

    # Test 4: RG flow with grid
    print("\n[Test 4] RG flow solving with grid")
    grid = np.array([
        [1, 2, 3, 0],
        [4, 5, 6, 0],
        [7, 8, 9, 0],
        [0, 0, 0, 0],
    ])
    candidates = sim.solve_with_rg_flow(grid, n_paths=2)
    print(f"  Generated {len(candidates)} IR candidates:")
    for c in candidates:
        print(f"    κ_UV={c.kappa_uv:.2f}, κ_IR={c.kappa_ir:.2f}, ratio={c.compression_ratio:.2f}")

    # Test 5: Anti-monotonicity verification along trajectory
    print("\n[Test 5] Anti-monotonicity along trajectory")
    trajectory = state.phase_trajectory
    violations = 0
    for i in range(1, len(trajectory)):
        if trajectory[i] > trajectory[i-1]:
            violations += 1
    print(f"  Trajectory: {trajectory}")
    print(f"  Anti-monotonicity violations: {violations}")
    assert violations == 0, "No violations of anti-monotonicity"

    print("\n" + "=" * 60)
    print("ALL RG FLOW SELF-TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
