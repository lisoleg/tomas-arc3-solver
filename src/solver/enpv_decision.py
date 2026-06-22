"""ENPV decision module: Expected Net Positive Value for search termination.

Computes the expected net positive value of continuing kappa-Snap search
vs. stopping early. Integrates with kappa_snap_searcher.py Phase B
termination conditions. ENPV < 0 → abandon search, ENPV > 0 → continue.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode


class ENPVDecision:
    """Expected Net Positive Value calculator for search termination.

    ENPV = Sum_i P(H_i|E) * (1 - MDL_i / max_MDL) - cost(compute)

    where cost(compute) estimates the computational resources needed
    to continue enumerating and verifying remaining candidates.

    Attributes:
        cost_per_evaluation: Estimated cost (seconds) per candidate evaluation.
        min_enpv_threshold: ENPV below which search should be abandoned.
        historical_enpv: Running history of ENPV values for trend tracking.
    """

    def __init__(self, cost_per_evaluation: float = 0.5,
                 min_enpv_threshold: float = 0.0) -> None:
        """Initialize the ENPV decision module.

        Args:
            cost_per_evaluation: Cost per candidate evaluation in seconds.
            min_enpv_threshold: Minimum ENPV value to continue search.
        """
        self.cost_per_evaluation = cost_per_evaluation
        self.min_enpv_threshold = min_enpv_threshold
        self.historical_enpv: list[float] = []

    def compute_enpv(self, candidates: list[tuple[ProgramNode, float]],
                     max_mdl: float | None = None) -> float:
        """Compute ENPV from ranked candidates with posteriors.

        ENPV = Sum P(H_i|E) * benefit_i - cost(remaining_evaluations)

        Benefit_i = 1 - MDL_i / max_MDL (higher benefit for simpler programs).

        Args:
            candidates: List of (program, posterior) tuples, sorted by posterior.
            max_mdl: Maximum MDL among candidates (computed if None).

        Returns:
            ENPV value. Positive → continue, negative → abandon.
        """
        if not candidates:
            return -float('inf')  # No candidates → abandon

        # Compute max MDL if not provided
        if max_mdl is None:
            max_mdl = float(max(
                (c[0].total_mdl for c in candidates if c[0].total_mdl > 0),
                default=1.0
            ))

        # Expected benefit: weighted sum of (1 - MDL_i/max_MDL)
        expected_benefit = 0.0
        for program, posterior in candidates:
            benefit = 1.0 - (program.total_mdl / max(max_mdl, 1))
            expected_benefit += posterior * benefit

        # Cost: remaining evaluations
        remaining_cost = len(candidates) * self.cost_per_evaluation / 3600.0  # in hours

        enpv = expected_benefit - remaining_cost
        self.historical_enpv.append(enpv)
        return enpv

    def should_continue(self, candidates: list[tuple[ProgramNode, float]],
                        max_mdl: float | None = None) -> bool:
        """Decide whether to continue search.

        Returns True if ENPV > min_enpv_threshold AND candidates remain.

        Args:
            candidates: Ranked candidate list.
            max_mdl: Maximum MDL (computed if None).

        Returns:
            True to continue search, False to stop.
        """
        if not candidates:
            return False

        enpv = self.compute_enpv(candidates, max_mdl)
        return enpv > self.min_enpv_threshold

    def select_best(self, candidates: list[tuple[ProgramNode, float]],
                    min_posterior: float = 0.01) -> ProgramNode | None:
        """Select the best candidate that exceeds minimum posterior.

        Args:
            candidates: Ranked candidate list.
            min_posterior: Minimum posterior threshold.

        Returns:
            Best ProgramNode or None.
        """
        if not candidates:
            return None

        best_program, best_posterior = candidates[0]
        if best_posterior >= min_posterior:
            return best_program
        return None

    def get_enpv_trend(self) -> str:
        """Analyze ENPV trend from history.

        Returns:
            'improving', 'declining', 'stable', or 'insufficient_data'.
        """
        if len(self.historical_enpv) < 3:
            return "insufficient_data"

        recent = self.historical_enpv[-3:]
        if recent[-1] > recent[0] * 1.1:
            return "improving"
        elif recent[-1] < recent[0] * 0.9:
            return "declining"
        else:
            return "stable"

    def reset(self) -> None:
        """Reset ENPV history."""
        self.historical_enpv.clear()
