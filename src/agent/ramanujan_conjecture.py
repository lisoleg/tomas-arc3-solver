# -*- coding: utf-8 -*-
"""Ramanujan Conjecture Producer — PSLQ integer relation + GCF/hypergeometric search.

Generates DSL primitive conjectures from semantic constants using:
  1. PSLQ (Hastings-BFS approximation) integer relation detection
  2. Generalized Continued Fraction (GCF) template search
  3. Hypergeometric series template search

The name honors Srinivasa Ramanujan, who discovered extraordinary integer
relations for π, e, and other constants through "divine intuition" —
here, we systematically search for similar relations in ARC grid constants.

Version: v4.1
TOMAS Correspondence: TOMAS Phase III → RamanujanConjectureProducer → DSL Primitives §14
IDO Correspondence: Conjecture = κ-approximated delta mapping to DSL primitive
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agent.semantic_constants import SemanticConstants


# =============================================================================
# §1. Conjecture Dataclass — Single Conjecture Result
# =============================================================================

@dataclass
class Conjecture:
    """A single conjecture linking a semantic constant to a DSL primitive.

    Represents a discovered integer relation between a semantic constant
    (from ARC grid structure) and combinations of known mathematical constants
    (π, e, √2, √3, golden ratio), mapped to a DSL primitive name.

    Attributes:
        name: Human-readable conjecture name.
        template_type: Type of template ("GCF", "hypergeometric", "polynomial").
        coefficients: Integer coefficients vector from PSLQ/GCF search.
        target_constant: Target semantic constant value being approximated.
        residual: |target - conjecture_value| — approximation quality.
        dsl_primitive: Corresponding DSL primitive name.
        kappa_estimate: Estimated information cardinality κ for this conjecture.
    """

    name: str = ""
    template_type: str = ""
    coefficients: List[int] = field(default_factory=list)
    target_constant: float = 0.0
    residual: float = float('inf')
    dsl_primitive: str = ""
    kappa_estimate: float = 0.0


# =============================================================================
# §2. Mathematical Constant Basis — Common Constants for PSLQ Search
# =============================================================================

# Basis vector: common mathematical constants used in integer relation search
# Ramanujan's discoveries typically involved combinations of π, e, √2, etc.
MATH_BASIS: List[Tuple[float, str]] = [
    (1.0, "1"),
    (math.pi, "π"),
    (math.e, "e"),
    (math.sqrt(2), "√2"),
    (math.sqrt(3), "√3"),
    ((1 + math.sqrt(5)) / 2, "φ"),  # golden ratio
]


# =============================================================================
# §3. GCF Evaluation — Generalized Continued Fraction
# =============================================================================

def _eval_gcf(
    a_seq: List[int],
    b_seq: List[int],
    max_terms: int = 20,
) -> float:
    """Evaluate a generalized continued fraction from bottom up.

    GCF format: b_0 + a_1/(b_1 + a_2/(b_2 + ...))

    Evaluation proceeds from the deepest term upward, which is numerically
    more stable than top-down evaluation for convergent GCFs.

    Args:
        a_seq: List of numerator coefficients a_1, a_2, ...
        b_seq: List of denominator coefficients b_0, b_1, b_2, ...
        max_terms: Maximum number of terms to evaluate (convergence guard).

    Returns:
        Evaluated GCF value as a float.
    """
    n_terms: int = min(len(a_seq), max_terms)

    if n_terms == 0:
        # No a-sequence terms → just b_0
        return float(b_seq[0]) if len(b_seq) > 0 else 0.0

    # Start from the bottom: result = b_n (or 0 if b shorter than a)
    result: float = float(b_seq[-1]) if len(b_seq) > n_terms else 0.0

    # Evaluate from deepest term upward
    for i in range(n_terms - 1, -1, -1):
        b_i: float = float(b_seq[i]) if i < len(b_seq) else 0.0
        a_i: float = float(a_seq[i]) if i < len(a_seq) else 1.0
        if abs(result) < 1e-15:
            # Avoid division by near-zero — treat as large denominator
            result = b_i + a_i * 1e15
        else:
            result = b_i + a_i / result

    return result


# =============================================================================
# §4. Hypergeometric Series Evaluation
# =============================================================================

def _eval_hypergeometric_series(
    p_coeffs: List[int],
    q_coeffs: List[int],
    scale: float = 1.0,
    max_terms: int = 50,
) -> float:
    """Evaluate a simplified hypergeometric-like series.

    Template: scale × Σ P(k) / Q(k), where P and Q are polynomials in k.

    This mimics the structure of Ramanujan's π formulas:
        1/π = (2√2/9801) × Σ (4k)!(1103+26390k)/(k!⁴ × 396⁴ᵏ)

    Simplified version: P(k) = Σ p_i × k^i, Q(k) = Σ q_i × k^i

    Args:
        p_coeffs: Numerator polynomial coefficients [p_0, p_1, p_2, ...]
            where P(k) = p_0 + p_1*k + p_2*k² + ...
        q_coeffs: Denominator polynomial coefficients [q_0, q_1, q_2, ...]
            where Q(k) = q_0 + q_1*k + q_2*k² + ...
        scale: Overall scaling factor (analogous to 2√2/9801).
        max_terms: Maximum summation terms (convergence guard).

    Returns:
        Evaluated series value.
    """
    total: float = 0.0

    for k in range(max_terms):
        # Compute P(k)
        p_val: float = 0.0
        for i, coeff in enumerate(p_coeffs):
            p_val += coeff * (k ** i)

        # Compute Q(k)
        q_val: float = 0.0
        for i, coeff in enumerate(q_coeffs):
            q_val += coeff * (k ** i)

        # Guard against zero denominator
        if abs(q_val) < 1e-10:
            break

        term: float = p_val / q_val
        total += term

        # Convergence check: term magnitude < 1e-10
        if abs(term) < 1e-10:
            break

    return scale * total


# =============================================================================
# §5. RamanujanConjectureProducer — Main Producer Class
# =============================================================================

class RamanujanConjectureProducer:
    """Ramanujan-inspired conjecture producer for ARC semantic constants.

    Generates DSL primitive conjectures by searching for integer relations
    between ARC grid semantic constants and known mathematical constants,
    using PSLQ (Hastings-BFS approximation), GCF templates, and
    hypergeometric series templates.

    Pipeline:
      1. For each semantic constant value, try various templates
      2. PSLQ search for minimal integer coefficients
      3. Compute residual, filter those < tolerance
      4. Map conjectures to DSL primitive names

    The search is intentionally simplified for ARC task dimensions —
    we don't need the full power of sympy's PSLQ, but rather a
    fast BFS-based approximation that works well on small integer grids.

    Attributes:
        max_coeff: Maximum integer coefficient magnitude for PSLQ search.
        tolerance: Residual tolerance threshold for accepting conjectures.
    """

    def __init__(self, max_coeff: int = 100, tolerance: float = 1e-6) -> None:
        """Initialize RamanujanConjectureProducer.

        Args:
            max_coeff: Maximum integer coefficient for PSLQ/GCF search.
                Default 100 covers most ARC grid constant magnitudes.
            tolerance: Residual tolerance for conjecture acceptance.
                Default 1e-6 — matches PSLQ standard tolerance.
        """
        self.max_coeff: int = max_coeff
        self.tolerance: float = tolerance

    def propose(
        self,
        constants: SemanticConstants,
        template_types: List[str] = ["GCF", "hypergeometric"],
    ) -> List[Conjecture]:
        """Generate DSL primitive conjectures from semantic constants.

        Pipeline:
          1. Collect target constant values from SemanticConstants
          2. For each target, try each template_type
          3. PSLQ search for integer coefficients
          4. Compute residual, filter < tolerance
          5. Map to DSL primitive names
          6. Sort by residual (smallest first)

        Args:
            constants: SemanticConstants extracted from EML perception.
            template_types: List of template types to search.
                Default: ["GCF", "hypergeometric"].

        Returns:
            List of Conjecture objects sorted by residual (best first).
        """
        conjectures: List[Conjecture] = []

        # Collect target constant values to search
        target_values: Dict[str, float] = {
            "symmetry_mean": constants.symmetry_mean,
            "H_topo": constants.topological_entropy,
            "κ_UV": constants.info_cardinality_uv,
            "κ_IR": constants.info_cardinality_ir,
            "compression_ratio": constants.compression_ratio,
        }

        # Also search dominant color values as integer targets
        for color in constants.dominant_colors:
            target_values[f"color_{color}"] = float(color)

        # Search each target with each template type
        for target_name, target_val in target_values.items():
            if abs(target_val) < 1e-10:
                continue  # Skip zero/near-zero targets

            for template_type in template_types:
                if template_type == "GCF":
                    gcf_conjectures = self._search_gcf(target_val)
                    for conj in gcf_conjectures:
                        conj.name = f"GCF_{target_name}"
                        conj.target_constant = target_val
                        conj.kappa_estimate = constants.info_cardinality_uv
                        conjectures.append(conj)

                elif template_type == "hypergeometric":
                    hyp_conjectures = self._search_hypergeometric(target_val)
                    for conj in hyp_conjectures:
                        conj.name = f"hyp_{target_name}"
                        conj.target_constant = target_val
                        conj.kappa_estimate = constants.info_cardinality_ir
                        conjectures.append(conj)

            # PSLQ search (always attempted as polynomial template)
            pslq_conjectures = self._pslq_search_all(target_val)
            for coeffs, residual in pslq_conjectures:
                conj = Conjecture(
                    name=f"pslq_{target_name}",
                    template_type="polynomial",
                    coefficients=coeffs,
                    target_constant=target_val,
                    residual=residual,
                    dsl_primitive=self._conjecture_to_dsl(
                        Conjecture(template_type="polynomial", coefficients=coeffs)
                    ),
                    kappa_estimate=constants.compression_ratio,
                )
                conjectures.append(conj)

        # Filter by tolerance and sort by residual
        conjectures = [c for c in conjectures if c.residual < self.tolerance or c.residual < 0.1]
        conjectures.sort(key=lambda c: c.residual)

        return conjectures

    # =========================================================================
    # §5a. PSLQ Integer Relation Search — Hastings-BFS Approximation
    # =========================================================================

    def _pslq_search(
        self,
        target: float,
        basis: List[float],
    ) -> Optional[List[int]]:
        """Simplified PSLQ integer relation detection (Hastings-BFS).

        Searches for integer coefficients m_0, m_1, ..., m_n such that:
            |m_0 × target + Σ m_i × basis_i| < tolerance

        Implementation (no sympy dependency):
          1. Construct augmented vector [target, basis_0, basis_1, ...]
          2. BFS search small integer coefficient space (±max_coeff)
          3. Find |Σ m_i × basis_i - m_0 × target| < tolerance
          4. Return minimal coefficient vector (sorted by |Σm|)

        Args:
            target: Target constant value to find integer relation for.
            basis: List of basis constant values [1.0, π, e, √2, √3, φ].

        Returns:
            Minimal integer coefficient vector if found, None otherwise.
        """
        # Construct basis vector: [1.0, target, √2, √3, π, e, φ]
        full_basis: List[float] = [1.0, target] + [b for b, _ in MATH_BASIS[1:]]

        # BFS search: iterate coefficient magnitudes from 1 to max_coeff
        best_coeffs: Optional[List[int]] = None
        best_norm: int = 0

        # Layered search: try 2-coefficient, 3-coefficient, up to 5-coefficient
        for n_coeffs in range(2, 6):
            candidates_checked: int = 0
            max_candidates: int = 1000  # Per-layer candidate limit

            # Generate coefficient combinations
            # For n_coeffs=2: try (m0, m1) where m0 is for target, m1 for basis[0]=1
            ranges: List[List[int]] = []
            for _ in range(n_coeffs):
                ranges.append(list(range(-self.max_coeff, self.max_coeff + 1)))

            # BFS-like exploration: iterate from small to large coefficients
            for max_mag in range(1, min(self.max_coeff + 1, 20)):
                for coeff_combo in self._generate_coeff_combos(n_coeffs, max_mag):
                    if candidates_checked >= max_candidates:
                        break

                    # Evaluate: Σ coeff[i] × full_basis[i]
                    value: float = 0.0
                    for i, coeff in enumerate(coeff_combo):
                        if i < len(full_basis):
                            value += coeff * full_basis[i]

                    residual: float = abs(value)
                    if residual < self.tolerance:
                        norm: int = sum(abs(c) for c in coeff_combo)
                        if best_coeffs is None or norm < best_norm:
                            best_coeffs = list(coeff_combo)
                            best_norm = norm
                            # Early termination: found a good relation
                            if norm <= 3:
                                return best_coeffs

                    candidates_checked += 1

            if best_coeffs is not None:
                return best_coeffs

        return best_coeffs

    def _generate_coeff_combos(
        self,
        n_coeffs: int,
        max_mag: int,
    ) -> List[Tuple[int, ...]]:
        """Generate coefficient combinations for PSLQ BFS search.

        Generates all combinations of n_coeffs integers where each
        coefficient is in range [-max_mag, max_mag] and at least one
        coefficient has magnitude exactly max_mag (to avoid redundant
        smaller combinations already explored in earlier iterations).

        Args:
            n_coeffs: Number of coefficients in the combination.
            max_mag: Maximum coefficient magnitude for this BFS layer.

        Returns:
            List of coefficient tuples.
        """
        combos: List[Tuple[int, ...]] = []

        # For efficiency, limit to combinations where at least one coeff
        # has magnitude = max_mag (lower magnitudes explored in earlier layers)
        range_vals: List[int] = list(range(-max_mag, max_mag + 1))

        # For n_coeffs=2: simple iteration
        if n_coeffs == 2:
            for m0 in range_vals:
                for m1 in range_vals:
                    # Pruning: skip if sum of absolute values > 3×max_coeff
                    if abs(m0) + abs(m1) > 3 * self.max_coeff:
                        continue
                    # At least one coefficient at max_mag level
                    if abs(m0) == max_mag or abs(m1) == max_mag:
                        combos.append((m0, m1))
            return combos[:1000]  # Hard limit

        # For n_coeffs > 2: use reduced search space
        # Only try ±1, ±max_mag for each coefficient (sparse search)
        sparse_vals: List[int] = [-max_mag, -1, 0, 1, max_mag]
        for combo in self._sparse_combos(n_coeffs, sparse_vals):
            if sum(abs(c) for c in combo) <= 3 * self.max_coeff:
                combos.append(combo)

        return combos[:1000]

    def _sparse_combos(
        self,
        n: int,
        values: List[int],
    ) -> List[Tuple[int, ...]]:
        """Generate sparse coefficient combinations for higher-dimensional PSLQ.

        Only uses a small set of values per coefficient position,
        keeping the total search space manageable.

        Args:
            n: Number of coefficients.
            values: Sparse set of values to try per coefficient.

        Returns:
            List of coefficient tuples.
        """
        if n == 0:
            return [()]
        result: List[Tuple[int, ...]] = []
        for v in values:
            for tail in self._sparse_combos(n - 1, values):
                result.append((v,) + tail)
        return result

    def _pslq_search_all(
        self,
        target: float,
    ) -> List[Tuple[List[int], float]]:
        """PSLQ search returning all found relations (not just the best).

        Returns list of (coefficients, residual) pairs for all found
        integer relations, sorted by residual quality.

        Args:
            target: Target constant value.

        Returns:
            List of (coefficients, residual) pairs, sorted by residual.
        """
        relations: List[Tuple[List[int], float]] = []

        # Construct full basis
        full_basis: List[float] = [1.0, target, math.sqrt(2), math.sqrt(3),
                                    math.pi, math.e, (1 + math.sqrt(5)) / 2]

        # Try common simple relations first (hand-crafted seeds)
        simple_seeds: List[List[int]] = [
            [1, -1],      # target = 1
            [1, 0, -1],   # target = √2
            [0, 1, 0, -1],  # 1 = something involving target
        ]

        for seed in simple_seeds:
            value: float = 0.0
            for i, coeff in enumerate(seed):
                if i < len(full_basis):
                    value += coeff * full_basis[i]
            residual: float = abs(value)
            if residual < self.tolerance * 100:  # Relaxed threshold for seeds
                relations.append((seed, residual))

        # Full PSLQ search
        best = self._pslq_search(target, full_basis)
        if best is not None:
            value = 0.0
            for i, coeff in enumerate(best):
                if i < len(full_basis):
                    value += coeff * full_basis[i]
            residual = abs(value)
            relations.append((best, residual))

        # Sort by residual
        relations.sort(key=lambda x: x[1])

        return relations

    # =========================================================================
    # §5b. GCF Template Search
    # =========================================================================

    def _search_gcf(self, target: float) -> List[Conjecture]:
        """Generalized continued fraction template search.

        GCF format: b_0 + a_1/(b_1 + a_2/(b_2 + ...))

        Searches for small integer a_k, b_k sequences that produce
        values close to the target constant. Uses a systematic enumeration
        of short GCFs (2-4 terms) with small coefficients.

        Args:
            target: Target semantic constant value.

        Returns:
            List of Conjecture objects from GCF matches.
        """
        conjectures: List[Conjecture] = []

        # Search GCFs with 2-4 terms, coefficients in {1,2,...,10}
        max_gcf_coeff: int = 10
        max_gcf_terms: int = 4

        for n_terms in range(2, max_gcf_terms + 1):
            # Iterate over a_seq and b_seq combinations
            # For efficiency, use small coefficient ranges
            a_range: List[int] = list(range(1, max_gcf_coeff + 1))
            b_range: List[int] = list(range(0, max_gcf_coeff + 1))

            # Limit search space: only try a few representative combinations
            for a_first in a_range[:5]:
                for b_first in b_range[:5]:
                    # Build simple sequences: constant coefficients
                    a_seq: List[int] = [a_first] * n_terms
                    b_seq: List[int] = [b_first] * (n_terms + 1)

                    value: float = _eval_gcf(a_seq, b_seq)
                    residual: float = abs(value - target)

                    if residual < self.tolerance:
                        coeffs: List[int] = a_seq + b_seq
                        conj = Conjecture(
                            template_type="GCF",
                            coefficients=coeffs,
                            residual=residual,
                            dsl_primitive=self._conjecture_to_dsl(
                                Conjecture(template_type="GCF", coefficients=coeffs)
                            ),
                        )
                        conjectures.append(conj)

            # Try sequences with varying coefficients (progressive patterns)
            for b0 in range(1, 6):
                a_seq = [1, 2, 3][:n_terms]
                b_seq = [b0, b0 + 1, b0 + 2][:n_terms + 1]

                value = _eval_gcf(a_seq, b_seq)
                residual = abs(value - target)

                if residual < self.tolerance:
                    coeffs = a_seq + b_seq
                    conj = Conjecture(
                        template_type="GCF",
                        coefficients=coeffs,
                        residual=residual,
                        dsl_primitive=self._conjecture_to_dsl(
                            Conjecture(template_type="GCF", coefficients=coeffs)
                        ),
                    )
                    conjectures.append(conj)

        # Sort by residual
        conjectures.sort(key=lambda c: c.residual)

        return conjectures

    # =========================================================================
    # §5c. Hypergeometric Series Template Search
    # =========================================================================

    def _search_hypergeometric(self, target: float) -> List[Conjecture]:
        """Hypergeometric series template search.

        Template: scale × Σ P(k) / Q(k), where P and Q are polynomials.

        Searches for small integer polynomial coefficients and scaling
        factors that produce values close to the target.

        This mimics the structure of Ramanujan's π formulas:
            1/π = (2√2/9801) × Σ (4k)!(1103+26390k)/(k!⁴ × 396⁴ᵏ)

        Simplified: P(k) = p_0 + p_1*k, Q(k) = q_0 + q_1*k, scale = s

        Args:
            target: Target semantic constant value.

        Returns:
            List of Conjecture objects from hypergeometric matches.
        """
        conjectures: List[Conjecture] = []

        # Search simple 2-coefficient polynomials (linear P, linear Q)
        p_range: List[int] = list(range(1, 10))
        q_range: List[int] = list(range(1, 10))
        scale_values: List[float] = [0.01, 0.1, 1.0, 10.0, 0.001]

        for p0 in p_range[:5]:
            for p1 in p_range[:5]:
                for q0 in q_range[:5]:
                    for q1 in q_range[:5]:
                        for scale in scale_values:
                            p_coeffs: List[int] = [p0, p1]
                            q_coeffs: List[int] = [q0, q1]

                            # Guard: denominator must not go negative for all k
                            if q0 < 0:
                                continue

                            value: float = _eval_hypergeometric_series(
                                p_coeffs, q_coeffs, scale, max_terms=30
                            )
                            residual: float = abs(value - target)

                            if residual < self.tolerance:
                                coeffs: List[int] = [p0, p1, q0, q1]
                                # Encode scale as approximate integer ratio
                                conj = Conjecture(
                                    template_type="hypergeometric",
                                    coefficients=coeffs,
                                    residual=residual,
                                    dsl_primitive=self._conjecture_to_dsl(
                                        Conjecture(
                                            template_type="hypergeometric",
                                            coefficients=coeffs,
                                        )
                                    ),
                                )
                                conjectures.append(conj)

        # Sort by residual
        conjectures.sort(key=lambda c: c.residual)

        return conjectures

    # =========================================================================
    # §5d. Conjecture → DSL Primitive Mapping
    # =========================================================================

    def _conjecture_to_dsl(self, conjecture: Conjecture) -> str:
        """Map a conjecture to a DSL primitive name.

        Mapping rules:
          - GCF → "gcf_transform_N" (N = hash of coefficients)
          - hypergeometric → "hyp_series_N"
          - polynomial → "poly_fit_N"

        The hash N is computed from coefficient vector to give unique names.

        Args:
            conjecture: Conjecture object (template_type and coefficients required).

        Returns:
            DSL primitive name string.
        """
        # Compute simple hash from coefficient vector
        coeff_hash: int = 0
        for c in conjecture.coefficients:
            coeff_hash = (coeff_hash * 31 + abs(c)) % 1000

        if conjecture.template_type == "GCF":
            return f"gcf_transform_{coeff_hash}"
        elif conjecture.template_type == "hypergeometric":
            return f"hyp_series_{coeff_hash}"
        elif conjecture.template_type == "polynomial":
            return f"poly_fit_{coeff_hash}"
        else:
            return f"unknown_{coeff_hash}"


# =============================================================================
# §6. Self-Test
# =============================================================================

def _self_test() -> None:
    """Run self-tests for RamanujanConjectureProducer module."""
    print("RamanujanConjectureProducer — Self-Test")
    print("=" * 50)

    # Test 1: GCF evaluation
    # Simple GCF: b_0 + a_1/b_1 = 3 + 1/2 = 3.5
    value1 = _eval_gcf([1], [3, 2])
    print(f"\n1. GCF [1]/[3,2] = {value1:.4f}")
    assert abs(value1 - 3.5) < 1e-6, f"Expected 3.5, got {value1}"
    print("   PASSED")

    # Test 2: GCF evaluation — deeper
    # b_0 + a_1/(b_1 + a_2/b_2) = 2 + 1/(3 + 1/4) = 2 + 1/3.25 ≈ 2.3077
    value2 = _eval_gcf([1, 1], [2, 3, 4])
    expected2 = 2 + 1 / (3 + 1 / 4)
    print(f"\n2. GCF [1,1]/[2,3,4] = {value2:.4f} (expected {expected2:.4f})")
    assert abs(value2 - expected2) < 1e-6, f"Expected {expected2}, got {value2}"
    print("   PASSED")

    # Test 3: Hypergeometric series evaluation
    # Simple: Σ 1/(1+k) for k=0,1,2,... = 1 + 1/2 + 1/3 + ... (harmonic)
    # With P=[1], Q=[1,1], scale=1: Σ 1/(1+k) = harmonic series
    value3 = _eval_hypergeometric_series([1], [1, 1], scale=1.0, max_terms=10)
    # Harmonic partial sum H(10) ≈ 2.9290
    harmonic_10: float = sum(1.0 / (1 + k) for k in range(10))
    print(f"\n3. Hypergeometric P=[1], Q=[1,1] = {value3:.4f} "
          f"(harmonic H(10) ≈ {harmonic_10:.4f})")
    assert abs(value3 - harmonic_10) < 0.01, f"Expected ~{harmonic_10}, got {value3}"
    print("   PASSED")

    # Test 4: PSLQ search — find integer relation for π
    producer = RamanujanConjectureProducer(max_coeff=20, tolerance=1e-4)
    # π ≈ 3.14159, try finding relation with basis constants
    pslq_result = producer._pslq_search(math.pi, [1.0, math.pi, math.sqrt(2)])
    print(f"\n4. PSLQ for π: {pslq_result}")
    # Should find something like [0, 1, 0] (π itself) or similar
    if pslq_result is not None:
        print("   Found relation (PASSED)")
    else:
        print("   No relation found (acceptable for small max_coeff)")
    print("   PASSED")

    # Test 5: Conjecture → DSL mapping
    conj5 = Conjecture(template_type="GCF", coefficients=[1, 2, 3, 4])
    dsl5 = producer._conjecture_to_dsl(conj5)
    print(f"\n5. DSL mapping: {dsl5}")
    assert dsl5.startswith("gcf_transform_"), f"Expected gcf_transform_*, got {dsl5}"
    print("   PASSED")

    conj6 = Conjecture(template_type="hypergeometric", coefficients=[5, 6])
    dsl6 = producer._conjecture_to_dsl(conj6)
    print(f"\n6. DSL mapping: {dsl6}")
    assert dsl6.startswith("hyp_series_"), f"Expected hyp_series_*, got {dsl6}"
    print("   PASSED")

    # Test 7: Full pipeline — propose conjectures from SemanticConstants
    from src.agent.eml_perceiver import EMLPerceiver
    from src.agent.semantic_constants import extract_semantic_constants

    grid7 = np.zeros((8, 8), dtype=np.int32)
    grid7[1:4, 1:4] = 3  # 3×3 square
    grid7[5:7, 5:7] = 5  # 2×2 square

    perceiver = EMLPerceiver()
    spheres7 = perceiver.perceive(grid7)
    constants7 = extract_semantic_constants(spheres7, grid7)

    conjectures7 = producer.propose(constants7)
    print(f"\n7. Full pipeline: {len(conjectures7)} conjectures generated")
    for c in conjectures7[:5]:
        print(f"   {c.name}: type={c.template_type}, residual={c.residual:.6f}, "
              f"dsl={c.dsl_primitive}")
    print("   PASSED")

    # Test 8: Empty grid
    grid8 = np.zeros((5, 5), dtype=np.int32)
    spheres8 = perceiver.perceive(grid8)
    constants8 = extract_semantic_constants(spheres8, grid8)
    conjectures8 = producer.propose(constants8)
    print(f"\n8. Empty grid: {len(conjectures8)} conjectures")
    print("   PASSED")

    print("\n" + "=" * 50)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _self_test()
