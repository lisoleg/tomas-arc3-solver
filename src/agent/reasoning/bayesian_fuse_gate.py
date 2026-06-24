"""BayesianFuseGate - L4 Observer for the Taiyi mutual-combat agent architecture.

This module implements the ``BayesianFuseGate`` class, which acts as a
meta-reasoning layer that:

1. **RHAE efficiency prediction** — predicts whether the current plan
   is efficient enough to succeed within a baseline budget of steps.
   If the predicted efficiency falls below a threshold, the gate *fuses*
   (triggers a circuit-breaker), signalling the planner to backtrack or
   re-plan.

2. **Dynamic weight balancing** — maintains a Bayesian-style balance
   between a *logic* weight (``w_logic``) and a *statistical* weight
   (``w_stat``).  When plans succeed, ``w_logic`` increases (the agent
   trusts its logic more); when plans fail, ``w_stat`` increases (the
   agent falls back to statistical priors).

3. **Fusion score** — combines a logic term (inversely proportional to
   remaining plan length) with a statistical term (placeholder for LLM
   confidence) using the current weights, then applies the RHAE
   circuit-breaker on the efficiency dimension.

The gate is intentionally lightweight (pure-Python, no external deps
beyond ``math``) so it can be called on every step of the game loop.
"""

from __future__ import annotations

import math
from typing import Optional


class BayesianFuseGate:
    """L4 observer: RHAE efficiency prediction + dynamic weight balance + fuse.

    The gate sits above the planner and evaluates every candidate plan
    before execution.  When the fused score or the efficiency ratio
    drops below acceptable levels, ``should_fuse`` returns ``True``,
    instructing the caller to backtrack or re-plan.

    Attributes:
        w_logic: Weight of the logic term (0.0–1.0).  Starts at
            ``w_logic_init`` and is adapted via :meth:`adapt_weights`.
        w_stat: Weight of the statistical term (``1 - w_logic``).
        success_count: Number of successful plan completions (Laplace
            smoothed, starts at 1).
        fail_count: Number of failed plan completions (Laplace smoothed,
            starts at 1).
        fuse_threshold: Minimum acceptable fused score.  Scores below
            this threshold trigger a fuse event.
        last_score: The most recently computed fused score (or ``None``
            if :meth:`score` has not been called yet).
        last_efficiency: The most recently computed efficiency ratio.
    """

    def __init__(
        self,
        w_logic_init: float = 0.5,
        fuse_threshold: float = 0.3,
    ) -> None:
        """Initialize the Bayesian fuse gate.

        Args:
            w_logic_init: Initial weight for the logic term.  Must be
                in [0.0, 1.0].  The statistical weight is set to
                ``1 - w_logic_init``.
            fuse_threshold: Minimum acceptable fused score.  When the
                score falls below this value (or efficiency < 0.5),
                :meth:`should_fuse` returns ``True``.

        Raises:
            ValueError: If ``w_logic_init`` is outside [0, 1].
        """
        if not 0.0 <= w_logic_init <= 1.0:
            raise ValueError(
                f"w_logic_init must be in [0, 1], got {w_logic_init}"
            )
        self._w_logic_init: float = w_logic_init
        self.w_logic: float = w_logic_init
        self.w_stat: float = 1.0 - w_logic_init
        self.success_count: int = 1
        self.fail_count: int = 1
        self.fuse_threshold: float = fuse_threshold
        self.last_score: Optional[float] = None
        self.last_efficiency: Optional[float] = None

    # ------------------------------------------------------------------
    # Weight adaptation
    # ------------------------------------------------------------------

    def adapt_weights(self) -> None:
        """Dynamically adjust logic/statistical weights.

        The logic weight is computed as a log-ratio of success and fail
        counts (both Laplace-smoothed by starting at 1)::

            w_logic = log(1 + success) / (log(1 + success) + log(1 + fail))

        When successes dominate, ``w_logic`` approaches 1 (trust logic).
        When failures dominate, ``w_logic`` approaches 0 (rely on stats).
        The statistical weight is always ``1 - w_logic``.

        This method is called automatically by :meth:`update_result`,
        but can also be called manually to force re-computation.
        """
        log_s: float = math.log(1 + self.success_count)
        log_f: float = math.log(1 + self.fail_count)
        denom: float = log_s + log_f
        if denom < 1e-12:
            # Both counts are effectively zero — keep balanced weights.
            self.w_logic = 0.5
        else:
            self.w_logic = log_s / denom
        self.w_stat = 1.0 - self.w_logic

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        plan_len: int,
        baseline: int,
        steps_used: int,
    ) -> float:
        """Compute the fused score for a candidate plan.

        The fused score combines two terms:

        * **Logic term** — ``log(1 / (plan_len + 1e-3))``.  Shorter
          remaining plans yield higher logic scores, reflecting
          confidence that the goal is within reach.

        * **Statistical term** — ``0.5`` (placeholder for future LLM
          confidence integration).

        The two terms are combined using the current ``w_logic`` /
        ``w_stat`` weights.  An RHAE efficiency circuit-breaker is then
        applied: if the predicted efficiency (``baseline / predicted_total``)
        is below 0.5, the score is forced to 0.0 regardless of the
        weighted combination.

        Args:
            plan_len: Remaining length of the candidate plan (number of
                actions still to execute).
            baseline: RHAE baseline step budget for the current level.
            steps_used: Steps already consumed on the current level.

        Returns:
            The fused score in [0.0, ~0.7].  A return value of ``0.0``
            indicates an RHAE circuit-breaker trip (efficiency too low).
        """
        predicted_total: int = steps_used + plan_len
        efficiency: float = baseline / max(predicted_total, 1)

        # Logic term: normalized to [0, 1].
        # 1.0 when plan_len=0 (goal reached), decreasing as plan grows.
        # Uses 1/(1+plan_len) which gives 1.0, 0.5, 0.33, 0.25, ...
        logic_term: float = 1.0 / (1.0 + plan_len)

        # Statistical term: placeholder for future LLM confidence.
        stat_term: float = 0.5

        fused: float = self.w_logic * logic_term + self.w_stat * stat_term

        # RHAE circuit-breaker: if predicted efficiency < 0.5, fuse.
        if efficiency < 0.5:
            fused = 0.0

        self.last_score = fused
        self.last_efficiency = efficiency
        return fused

    def should_fuse(
        self,
        plan_len: int,
        baseline: int,
        steps_used: int,
    ) -> bool:
        """Determine whether the gate should fuse (trigger re-planning).

        The gate fuses when **either** condition is met:

        1. RHAE efficiency circuit-breaker: predicted efficiency < 0.5.
        2. Fused score < ``fuse_threshold``.

        Args:
            plan_len: Remaining length of the candidate plan.
            baseline: RHAE baseline step budget for the current level.
            steps_used: Steps already consumed on the current level.

        Returns:
            ``True`` if the gate fuses (caller should backtrack / re-plan),
            ``False`` otherwise.
        """
        fused_score: float = self.score(plan_len, baseline, steps_used)
        # Condition 1: RHAE circuit-breaker (score was zeroed by efficiency).
        if fused_score <= 0.0:
            return True
        # Condition 2: score below configured threshold.
        if fused_score < self.fuse_threshold:
            return True
        return False

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def update_result(self, success: bool) -> None:
        """Record the outcome of a plan and adapt weights accordingly.

        Args:
            success: ``True`` if the plan led to a level completion,
                ``False`` if it led to a game-over or dead end.
        """
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
        self.adapt_weights()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a diagnostic snapshot of the gate's internal state.

        Returns:
            Dictionary with ``w_logic``, ``w_stat``, ``success_count``,
            ``fail_count``, ``fuse_threshold``, ``last_score``, and
            ``last_efficiency``.
        """
        return {
            "w_logic": self.w_logic,
            "w_stat": self.w_stat,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "fuse_threshold": self.fuse_threshold,
            "last_score": self.last_score,
            "last_efficiency": self.last_efficiency,
        }

    def reset(self) -> None:
        """Reset the gate to its initial state.

        Weights return to the initial split (``_w_logic_init``) and
        counts are Laplace-smoothed back to 1.
        """
        self.w_logic = self._w_logic_init
        self.w_stat = 1.0 - self._w_logic_init
        self.success_count = 1
        self.fail_count = 1
        self.last_score = None
        self.last_efficiency = None
