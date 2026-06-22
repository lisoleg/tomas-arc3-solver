"""Bayesian confidence quantification: P(H|E) = P(E|H) * P(H) / P(E).

TOMAS v2.0 upgrade: Jitter variance estimation — adaptive noise_sigma from
multi-frame residual autocorrelation; residual trend detection penalizes
programs with monotonically increasing prediction errors.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode


class BayesianConfidence:
    """Bayesian posterior estimation for program candidate ranking.

    Computes prior P(H) proportional to exp(-lambda * MDL), likelihood
    P(E|H) using Gaussian fit with adaptive sigma (Jitter estimation),
    evidence P(E) as marginal likelihood, and posterior P(H|E).

    Attributes:
        candidates: List of candidate ProgramNodes.
        prior_lambda: Prior temperature parameter.
        noise_sigma: Base observation noise standard deviation.
        adaptive_sigma: Whether to use Jitter-adaptive sigma estimation.
        residual_trend_window: Window for residual trend detection.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the Bayesian confidence module.

        Args:
            config: Bayesian config with prior_lambda, noise_sigma,
                adaptive_sigma, residual_trend_window, max_candidates.
        """
        self.candidates: list[ProgramNode] = []
        self.prior_lambda: float = config.get("prior_lambda", 0.1)
        self.noise_sigma: float = config.get("noise_sigma", 0.5)
        self.max_candidates: int = config.get("max_candidates", 100)
        self.adaptive_sigma: bool = config.get("adaptive_sigma", True)
        self.residual_trend_window: int = config.get("residual_trend_window", 3)
        self._evidence_cache: float | None = None

        # Jitter tracking
        self._residual_history: list[float] = []

    def compute_prior(self, program: ProgramNode) -> float:
        """Compute prior probability P(H) proportional to exp(-lambda * MDL).

        Lower MDL programs have higher prior probability.

        Args:
            program: ProgramNode to evaluate.

        Returns:
            Prior probability (unnormalized).
        """
        mdl = max(program.total_mdl, 1)
        return math.exp(-self.prior_lambda * mdl)

    def compute_likelihood(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> float:
        """Compute likelihood P(E|H) using Gaussian distribution.

        Measures how well the program's predictions match observed
        demo outputs. Uses adaptive sigma (Jitter estimation) when enabled:
        sigma = std(residuals) across all demo predictions.
        Uses Gaussian: exp(-||predicted - observed||^2 / (2*sigma^2)).

        Args:
            program: ProgramNode to evaluate.
            demo_pairs: List of demo pairs with input/output grids.

        Returns:
            Likelihood value in (0, 1].
        """
        total_match = 0.0
        total_count = 0
        residuals: list[float] = []

        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, input_grid in enumerate(input_grids):
                if i >= len(output_grids):
                    continue
                try:
                    predicted = program.apply(input_grid)
                    expected = output_grids[i]

                    if predicted.shape != expected.shape:
                        total_match += 0.0
                        total_count += 1
                        residuals.append(1.0)
                        continue

                    diff = np.abs(
                        predicted.astype(np.float32) - expected.astype(np.float32)
                    )
                    match_ratio = 1.0 - (np.mean(diff) / 9.0)
                    residuals.append(1.0 - match_ratio)

                    # Use adaptive sigma if enabled
                    sigma = self.estimate_jitter(residuals) if self.adaptive_sigma else self.noise_sigma
                    sigma_sq = max(sigma ** 2, 1e-8)

                    gaussian = math.exp(-((1.0 - match_ratio) ** 2) / (2 * sigma_sq))
                    total_match += gaussian
                    total_count += 1
                except Exception:
                    total_match += 0.0
                    total_count += 1
                    residuals.append(1.0)

        if total_count == 0:
            return 0.0

        # Penalize residual trends
        trend_penalty = 1.0
        if len(residuals) >= self.residual_trend_window:
            if self.detect_residual_trend(residuals):
                trend_penalty = 0.5  # 50% penalty for increasing residuals

        return (total_match / total_count) * trend_penalty

    def estimate_jitter(self, residuals: list[float]) -> float:
        """Estimate adaptive noise sigma from residuals (Jitter estimation).

        Uses the standard deviation of recent prediction residuals
        as the adaptive sigma. This captures the predictive uncertainty
        of the program: programs with erratic predictions get higher sigma
        (lower likelihood), while consistent programs get lower sigma.

        Args:
            residuals: List of prediction error residuals [0, 1].

        Returns:
            Adaptive sigma value.
        """
        if len(residuals) < 2:
            return self.noise_sigma  # Fallback to base sigma

        arr = np.array(residuals, dtype=np.float32)
        sigma = float(np.std(arr))
        # Blend with base sigma for stability
        sigma = 0.7 * sigma + 0.3 * self.noise_sigma
        return max(sigma, 0.01)  # Minimum sigma to avoid division by zero

    def detect_residual_trend(self, residuals: list[float]) -> bool:
        """Detect if residuals show a monotonically increasing trend.

        A growing residual trend indicates the program is getting worse
        over time — penalize its confidence.

        Args:
            residuals: List of residuals to check.

        Returns:
            True if increasing trend detected.
        """
        if len(residuals) < self.residual_trend_window:
            return False

        recent = residuals[-self.residual_trend_window:]
        # Check if residuals are monotonically increasing
        increasing = all(
            recent[i] <= recent[i + 1] * 1.05  # Allow 5% noise tolerance
            for i in range(len(recent) - 1)
        )
        # Also check mean trend
        if len(recent) >= 3:
            first_half = np.mean(recent[: len(recent) // 2])
            second_half = np.mean(recent[len(recent) // 2:])
            if second_half > first_half * 1.2:
                return True

        return increasing and recent[-1] > 0.3  # Only if residuals are significant

    def compute_evidence(self) -> float:
        """Compute evidence P(E) as marginal likelihood.

        P(E) = sum over all candidates of P(E|H_i) * P(H_i).

        Returns:
            Evidence value.
        """
        if self._evidence_cache is not None:
            return self._evidence_cache

        if not self.candidates:
            self._evidence_cache = 0.0
            return 0.0

        # Evidence is computed in rank_candidates and cached
        self._evidence_cache = 1.0  # Placeholder; updated in rank_candidates
        return self._evidence_cache

    def compute_posterior(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> float:
        """Compute posterior probability P(H|E) = P(E|H) * P(H) / P(E).

        Args:
            program: ProgramNode to evaluate.
            demo_pairs: List of demo pairs.

        Returns:
            Posterior probability.
        """
        prior = self.compute_prior(program)
        likelihood = self.compute_likelihood(program, demo_pairs)
        evidence = self.compute_evidence()

        if evidence == 0:
            return 0.0

        posterior = (likelihood * prior) / evidence
        return posterior

    def rank_candidates(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[tuple[ProgramNode, float]]:
        """Rank candidates by posterior probability.

        Args:
            candidates: List of ProgramNode candidates.
            demo_pairs: List of demo pairs.

        Returns:
            List of (ProgramNode, posterior) tuples, sorted by descending posterior.
        """
        self.candidates = candidates[: self.max_candidates]

        # Compute prior and likelihood for all candidates
        posteriors: list[tuple[ProgramNode, float]] = []
        prior_likelihoods: list[float] = []

        for program in self.candidates:
            prior = self.compute_prior(program)
            likelihood = self.compute_likelihood(program, demo_pairs)
            pl = prior * likelihood
            prior_likelihoods.append(pl)

        # Compute evidence (marginal likelihood)
        evidence = sum(prior_likelihoods)
        self._evidence_cache = evidence

        if evidence == 0:
            # All candidates have zero likelihood; return uniform
            uniform = 1.0 / max(len(self.candidates), 1)
            for program in self.candidates:
                posteriors.append((program, uniform))
        else:
            for i, program in enumerate(self.candidates):
                posterior = prior_likelihoods[i] / evidence
                posteriors.append((program, posterior))

        # Sort by descending posterior
        posteriors.sort(key=lambda x: x[1], reverse=True)
        return posteriors
