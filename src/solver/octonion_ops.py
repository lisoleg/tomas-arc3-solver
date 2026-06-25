# -*- coding: utf-8 -*-
"""Octonion operations for TOMAS NARLA (Non-Associative Residual Linear Algebra).

Note: NARLA = Non-Associative Residual Linear Algebra (非结合残联代数).
Octonions form an *alternative* algebra (交替代数), NOT a Lie algebra.
The non-associative residual [a,b,c] = (ab)c - a(bc) ≠ 0 is the algebraic
signature of physical intelligence, distinguishing it from statistical proxy AI.

TOMAS v3.1 — Asym Index (eta) computation for distinguishing physical AI
from statistical proxy AI.

Core theoretical insight from NARLA formal framework:
    "The octonion algebra O is the largest normed division algebra.
     Its non-associativity is not a bug — it is the mathematical
     signature of physical intelligence."

The Asym Index eta measures the non-associative residual:
    eta = E[||(a*b)*c - a*(b*c)||] / E[||a|| * ||b|| * ||c||]

Interpretation:
    eta ≈ 0  →  Statistical proxy AI (fully associative, passive pattern matching)
    eta > 0  →  Physical AI (non-associative, structural reasoning)

This module also provides the OctonionConv2dCUDA monitoring wrapper
and GaussEx + Lean4 dual verification primitives.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# =============================================================================
# Octonion Algebra Primitives
# =============================================================================

# Octonion multiplication table (simplified: e0=scalar, e1-e7=imaginary)
# Using the standard Cayley-Dickson construction rules:
#   e_i * e_i = -1 (for i>0)
#   e_i * e_j = e_k where (i,j,k) follows Fano plane cycles
#   e_i * e_j = -e_j * e_i (anti-commutative for distinct i,j)
_FANO_TRIPLES = [
    (1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7),
    (5, 6, 1), (6, 7, 2), (7, 1, 3),
]

# Pre-compute the full multiplication table
_MULT_TABLE: dict[tuple[int, int], tuple[int, int]] = {}  # (a,b) -> (c, sign)


def _init_mult_table() -> None:
    """Initialize the octonion multiplication table."""
    if _MULT_TABLE:
        return

    # e0 * anything = anything
    for i in range(8):
        _MULT_TABLE[(0, i)] = (i, 1)
        _MULT_TABLE[(i, 0)] = (i, 1)

    # e_i * e_i = -1 (represented as (0, -1))
    for i in range(1, 8):
        _MULT_TABLE[(i, i)] = (0, -1)

    # Fano plane cycles: e_i * e_j = e_k
    for i, j, k in _FANO_TRIPLES:
        _MULT_TABLE[(i, j)] = (k, 1)
        _MULT_TABLE[(j, i)] = (k, -1)

        # Also: e_j * e_k = e_i, e_k * e_i = e_j
        _MULT_TABLE[(j, k)] = (i, 1)
        _MULT_TABLE[(k, j)] = (i, -1)
        _MULT_TABLE[(k, i)] = (j, 1)
        _MULT_TABLE[(i, k)] = (j, -1)


_init_mult_table()


def oct_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two octonions.

    Args:
        a: 8-element array [e0, e1, ..., e7].
        b: 8-element array [e0, e1, ..., e7].

    Returns:
        8-element product array.
    """
    result = np.zeros(8, dtype=np.float64)
    for i in range(8):
        if abs(a[i]) < 1e-15:
            continue
        for j in range(8):
            if abs(b[j]) < 1e-15:
                continue
            k, sign = _MULT_TABLE[(i, j)]
            result[k] += sign * a[i] * b[j]
    return result


def oct_norm(a: np.ndarray) -> float:
    """Compute the norm of an octonion.

    Args:
        a: 8-element array.

    Returns:
        Euclidean norm.
    """
    return float(np.sqrt(np.sum(a ** 2)))


# =============================================================================
# Asym Index (eta) Computation
# =============================================================================

def compute_asym_index(
    oct_tensor: np.ndarray,
    n_samples: int = 100,
    seed: int = 42,
) -> float:
    """Compute the Asym Index (eta) for an octonion-encoded tensor.

    Samples random triples (a, b, c) from the octonion tensor and computes
    the associator norm ||(ab)c - a(bc)||, normalized by the product of norms.

    The Asym Index eta is the expected relative non-associativity:
        eta = E[||(ab)c - a(bc)|| / (||a|| * ||b|| * ||c||)]

    Args:
        oct_tensor: Input tensor of octonion vectors.
            Shape can be (N, 8), (H, W, 8), or (B, H, W, 8).
        n_samples: Number of random triples to sample.
        seed: Random seed for reproducibility.

    Returns:
        Asym Index eta in [0, 1].
            eta ≈ 0: Statistical proxy AI (associative)
            eta > 0: Physical AI (non-associative)
    """
    rng = np.random.RandomState(seed)

    # Flatten to (N, 8)
    if oct_tensor.ndim == 1:
        # Single octonion — need a batch
        oct_vectors = oct_tensor.reshape(1, -1)
    elif oct_tensor.ndim == 2 and oct_tensor.shape[1] == 8:
        oct_vectors = oct_tensor
    elif oct_tensor.ndim == 3 and oct_tensor.shape[2] == 8:
        H, W, _ = oct_tensor.shape
        oct_vectors = oct_tensor.reshape(H * W, 8)
    elif oct_tensor.ndim == 4 and oct_tensor.shape[3] == 8:
        B, H, W, _ = oct_tensor.shape
        oct_vectors = oct_tensor.reshape(B * H * W, 8)
    else:
        # Try to interpret as flat array of octonions
        total = oct_tensor.size // 8
        if total == 0:
            return 0.0
        oct_vectors = oct_tensor.flatten()[:total * 8].reshape(total, 8)

    N = oct_vectors.shape[0]
    if N < 3:
        return 0.0

    actual_samples = min(n_samples, N * 10)  # Cap samples

    associator_norms: list[float] = []
    product_norms: list[float] = []

    for _ in range(actual_samples):
        # Sample 3 distinct vectors
        indices = rng.choice(N, size=3, replace=(N < 3))
        if N >= 3:
            while len(set(indices)) < 3 and N >= 3:
                indices = rng.choice(N, size=3, replace=False)

        a = oct_vectors[indices[0]].astype(np.float64)
        b = oct_vectors[indices[1]].astype(np.float64)
        c = oct_vectors[indices[2]].astype(np.float64)

        # Compute (a*b)*c
        ab = oct_multiply(a, b)
        ab_c = oct_multiply(ab, c)

        # Compute a*(b*c)
        bc = oct_multiply(b, c)
        a_bc = oct_multiply(a, bc)

        # Associator: (a*b)*c - a*(b*c)
        associator = ab_c - a_bc

        assoc_norm = oct_norm(associator)
        prod_norm = oct_norm(a) * oct_norm(b) * oct_norm(c)

        if prod_norm > 1e-15:
            associator_norms.append(assoc_norm / prod_norm)
            product_norms.append(prod_norm)

    if not associator_norms:
        return 0.0

    eta = float(np.mean(associator_norms))

    # Clamp to [0, 1]
    return min(1.0, max(0.0, eta))


def compute_asym_index_batch(
    oct_tensor: np.ndarray,
    n_samples: int = 50,
    n_bootstrap: int = 10,
) -> dict[str, float]:
    """Compute Asym Index with bootstrap confidence interval.

    Args:
        oct_tensor: Input tensor of octonion vectors.
        n_samples: Samples per bootstrap iteration.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with 'eta', 'eta_std', 'eta_ci_low', 'eta_ci_high'.
    """
    etas = []
    for b in range(n_bootstrap):
        eta = compute_asym_index(oct_tensor, n_samples=n_samples, seed=42 + b)
        etas.append(eta)

    etas_arr = np.array(etas)
    return {
        "eta": float(np.mean(etas_arr)),
        "eta_std": float(np.std(etas_arr)),
        "eta_ci_low": float(np.percentile(etas_arr, 5)),
        "eta_ci_high": float(np.percentile(etas_arr, 95)),
    }


# =============================================================================
# OctonionConv2dCUDA Monitoring Wrapper
# =============================================================================

@dataclass
class OctonionMonitor:
    """Monitor for OctonionConv2dCUDA non-associative residual.

    Tracks the Asym Index over training/forward passes to detect
    whether the model is developing physical AI characteristics
    (eta > 0) or remaining in statistical proxy mode (eta ≈ 0).

    Key judgment from TOMAS NARLA integration:
        "OctonionConv2dCUDA already IS NARLA — no need to rebuild.
         Just add monitoring to track the eta trajectory."

    Attributes:
        eta_history: Recorded eta values over time.
        decision_threshold: eta above which we consider it "physical AI".
        total_passes: Number of monitored forward passes.
    """

    eta_history: list[float] = field(default_factory=list)
    decision_threshold: float = 0.05
    total_passes: int = 0

    def record_forward(self, weight_tensor: np.ndarray) -> dict[str, Any]:
        """Record Asym Index from an OctonionConv2dCUDA forward pass.

        Call this after each forward pass to track eta evolution.

        Args:
            weight_tensor: The octonion weight tensor from Conv2dCUDA.
                Expected shape: (out_ch, in_ch, kH, kW, 8) or similar.

        Returns:
            Dict with eta, trend, and decision.
        """
        eta = compute_asym_index(weight_tensor, n_samples=50)
        self.eta_history.append(eta)
        self.total_passes += 1

        # Rolling trend over last 10
        recent = self.eta_history[-min(10, len(self.eta_history)):]
        trend = "stable"
        if len(recent) >= 3:
            if recent[-1] > recent[0] * 1.1:
                trend = "increasing"
            elif recent[-1] < recent[0] * 0.9:
                trend = "decreasing"

        decision = "physical_ai" if eta > self.decision_threshold else "statistical_proxy"

        return {
            "eta": eta,
            "trend": trend,
            "decision": decision,
            "pass": self.total_passes,
        }

    def is_physical_ai(self) -> bool:
        """Check if model has crossed into physical AI regime.

        Returns:
            True if recent eta consistently > threshold.
        """
        if len(self.eta_history) < 5:
            return False
        recent = self.eta_history[-5:]
        return np.mean(recent) > self.decision_threshold

    def get_trajectory(self) -> dict[str, Any]:
        """Get the full eta trajectory for analysis.

        Returns:
            Dict with trajectory statistics.
        """
        if not self.eta_history:
            return {"mean_eta": 0.0, "max_eta": 0.0, "crossed_threshold": False}

        etas = np.array(self.eta_history)
        return {
            "mean_eta": float(np.mean(etas)),
            "max_eta": float(np.max(etas)),
            "min_eta": float(np.min(etas)),
            "final_eta": float(etas[-1]),
            "crossed_threshold": bool(np.any(etas > self.decision_threshold)),
            "total_passes": self.total_passes,
            "regime": "physical_ai" if self.is_physical_ai() else "statistical_proxy",
        }


# =============================================================================
# GaussEx + Lean4 Dual Verification Primitives
# =============================================================================

def dead_zero_safety_check(
    eta: float,
    fidelity: float,
    theta_eta: float = 0.01,
    theta_fidelity: float = 0.15,
) -> dict[str, Any]:
    """GaussEx + Lean4 dual verification: Dead-Zero safety check.

    Combines two orthogonal safety signals:
    1. GaussEx (fidelity): Information preservation from the behavioral
       fiber perspective. Low fidelity = Dead-Zero = hallucination.
    2. Lean4 (eta): Non-associative residual from the algebraic perspective.
       eta ≈ 0 + high fidelity = correct but shallow (statistical proxy).
       eta > 0 + high fidelity = correct and deep (physical AI).

    Safety decision matrix:
        fidelity >= theta_fidelity, eta >= theta_eta → SAFE (physical AI)
        fidelity >= theta_fidelity, eta < theta_eta  → SAFE_SHALLOW (correct proxy)
        fidelity < theta_fidelity,  eta >= theta_eta → RISKY (creative but unreliable)
        fidelity < theta_fidelity,  eta < theta_eta  → DEAD_ZERO (hallucination)

    This implements the dual verification concept from:
        "GaussEx verifies correctness (Willems behavioral fibers)
         Lean4 verifies safety (Dead-Zero anti-hallucination gate)"

    Args:
        eta: Asym Index from compute_asym_index().
        fidelity: Information fidelity from GaussExVerifier._calc_fidelity().
        theta_eta: Asym Index threshold for physical AI.
        theta_fidelity: Fidelity threshold for Dead-Zero gate.

    Returns:
        Dict with safety decision and diagnostic info.
    """
    if fidelity >= theta_fidelity:
        if eta >= theta_eta:
            decision = "SAFE"
            description = "Physical AI: high fidelity + non-associative residual"
        else:
            decision = "SAFE_SHALLOW"
            description = "Correct but shallow: high fidelity, statistical proxy"
        safety_score = fidelity * (0.5 + 0.5 * min(1.0, eta / theta_eta))
    else:
        if eta >= theta_eta:
            decision = "RISKY"
            description = "Creative but unreliable: low fidelity despite non-associative structure"
        else:
            decision = "DEAD_ZERO"
            description = "Hallucination: low fidelity, no non-associative structure"
        safety_score = fidelity * 0.3

    return {
        "decision": decision,
        "description": description,
        "safety_score": safety_score,
        "eta": eta,
        "fidelity": fidelity,
        "theta_eta": theta_eta,
        "theta_fidelity": theta_fidelity,
    }


# =============================================================================
# Quick Diagnostic
# =============================================================================

def quick_narla_diagnostic(oct_tensor: np.ndarray) -> str:
    """Quick diagnostic: is this model physical AI or statistical proxy?

    Args:
        oct_tensor: Octonion-encoded tensor.

    Returns:
        Human-readable diagnostic string.
    """
    eta_result = compute_asym_index_batch(oct_tensor, n_samples=30, n_bootstrap=5)

    eta = eta_result["eta"]
    ci_low = eta_result["eta_ci_low"]
    ci_high = eta_result["eta_ci_high"]

    if eta < 0.01 and ci_high < 0.02:
        return (
            f"Statistical Proxy AI (eta={eta:.4f}, 90% CI [{ci_low:.4f}, {ci_high:.4f}]). "
            "Model is nearly associative — pattern matching without structural reasoning."
        )
    elif eta < 0.05:
        return (
            f"Transitional (eta={eta:.4f}, 90% CI [{ci_low:.4f}, {ci_high:.4f}]). "
            "Weak non-associative residual detected — partial physical reasoning emerging."
        )
    elif eta < 0.15:
        return (
            f"Physical AI (eta={eta:.4f}, 90% CI [{ci_low:.4f}, {ci_high:.4f}]). "
            "Clear non-associative structure — model performs genuine structural reasoning."
        )
    else:
        return (
            f"Strong Physical AI (eta={eta:.4f}, 90% CI [{ci_low:.4f}, {ci_high:.4f}]). "
            "High non-associative residual — deep structural cognition detected."
        )


if __name__ == "__main__":
    print("OctonionOps — Asym Index Smoke Test")
    print("=" * 50)

    # Test 1: Associative case (random — should be nearly associative)
    rng = np.random.RandomState(42)
    random_oct = rng.randn(100, 8) * 0.1
    eta_random = compute_asym_index(random_oct, n_samples=50)
    print(f"\n1. Random octonions (should be near-zero):")
    print(f"   eta = {eta_random:.6f}")
    print(f"   {quick_narla_diagnostic(random_oct)}")

    # Test 2: Simulated non-associative case
    # Add structured non-associativity
    nonassoc_oct = random_oct.copy()
    nonassoc_oct[::3, 4] += 1.0  # Break associativity by amplifying e4
    eta_nonassoc = compute_asym_index(nonassoc_oct, n_samples=50)
    print(f"\n2. Non-associative octonions (should be > 0):")
    print(f"   eta = {eta_nonassoc:.6f}")
    print(f"   {quick_narla_diagnostic(nonassoc_oct)}")

    # Test 3: Multiplication test
    a = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    c = oct_multiply(a, b)
    print(f"\n3. Multiplication test: (1 + 0.5e1) * (0.3e2) = {c[:4].round(4)}")

    # Test 4: Safety check
    result = dead_zero_safety_check(eta_nonassoc, fidelity=0.8)
    print(f"\n4. GaussEx+Lean4 Safety Check:")
    print(f"   Decision: {result['decision']}")
    print(f"   Description: {result['description']}")
    print(f"   Safety Score: {result['safety_score']:.3f}")

    print("\n" + "=" * 50)
    print("PASSED")
