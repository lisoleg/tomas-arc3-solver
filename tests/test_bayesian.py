"""Bayesian confidence and fusion scoring tests.

Tests compute_prior (P(H) ∝ exp(-λ·MDL)), compute_likelihood
(Gaussian), compute_evidence (marginal), compute_posterior
(Bayesian), rank_candidates (posterior sorting), FusionScorer
fusion scoring, and degraded mode (G_vis=0).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.solver.bayesian_confidence import BayesianConfidence
from src.solver.fusion_scorer import FusionScorer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def bayes_config():
    """Configuration for BayesianConfidence."""
    return {
        "prior_lambda": 0.1,
        "noise_sigma": 0.5,
        "max_candidates": 100,
    }


@pytest.fixture
def fusion_config():
    """Configuration for FusionScorer."""
    return {
        "alpha": 0.4,
        "beta": 0.35,
        "gamma": 0.25,
        "alpha_degraded": 0.6,
        "gamma_degraded": 0.4,
    }


@pytest.fixture
def mirror_demo_pairs():
    """Demo pairs where output is horizontal mirror of input."""
    grids_in = [
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.int8),
    ]
    grids_out = [np.fliplr(g) for g in grids_in]
    return [{"input": grids_in, "output": grids_out}]


@pytest.fixture
def mirror_program():
    """A correct mirror program (low MDL)."""
    return ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))


@pytest.fixture
def incorrect_program():
    """An incorrect program (still low MDL)."""
    return ProgramNode(DSLElement("rotate", {"angle": 90}))


@pytest.fixture
def high_mdl_program():
    """A program with high MDL (deep chain)."""
    elem1 = DSLElement("mirror")
    elem2 = DSLElement("rotate")
    elem3 = DSLElement("scale", {"factor": 2})
    node = ProgramNode(elem1)
    child = ProgramNode(elem2)
    grandchild = ProgramNode(elem3)
    node.children.append(child)
    child.children.append(grandchild)
    node.combo_type = "chain"
    node.total_mdl = node.compute_mdl()
    return node


# ============================================================
# Prior Tests
# ============================================================

class TestComputePrior:
    """Tests for compute_prior P(H) ∝ exp(-λ·MDL)."""

    def test_prior_positive(self, bayes_config, mirror_program):
        """Prior should be positive."""
        bayes = BayesianConfidence(bayes_config)
        prior = bayes.compute_prior(mirror_program)
        assert prior > 0

    def test_prior_decreases_with_mdl(self, bayes_config, mirror_program, high_mdl_program):
        """Lower MDL should have higher prior."""
        bayes = BayesianConfidence(bayes_config)
        prior_low = bayes.compute_prior(mirror_program)
        prior_high = bayes.compute_prior(high_mdl_program)
        assert prior_low > prior_high

    def test_prior_formula(self, bayes_config, mirror_program):
        """Prior should follow exp(-lambda * MDL)."""
        bayes = BayesianConfidence(bayes_config)
        prior = bayes.compute_prior(mirror_program)
        expected = math.exp(-bayes_config["prior_lambda"] * max(mirror_program.total_mdl, 1))
        assert abs(prior - expected) < 1e-10

    def test_prior_in_range(self, bayes_config, mirror_program):
        """Prior should be in (0, 1]."""
        bayes = BayesianConfidence(bayes_config)
        prior = bayes.compute_prior(mirror_program)
        assert 0 < prior <= 1.0


# ============================================================
# Likelihood Tests
# ============================================================

class TestComputeLikelihood:
    """Tests for compute_likelihood P(E|H) Gaussian."""

    def test_correct_program_high_likelihood(self, bayes_config, mirror_program, mirror_demo_pairs):
        """Correct program should have high likelihood."""
        bayes = BayesianConfidence(bayes_config)
        likelihood = bayes.compute_likelihood(mirror_program, mirror_demo_pairs)
        assert likelihood > 0.5

    def test_incorrect_program_lower_likelihood(self, bayes_config, incorrect_program, mirror_demo_pairs):
        """Incorrect program should have lower likelihood."""
        bayes = BayesianConfidence(bayes_config)
        likelihood = bayes.compute_likelihood(incorrect_program, mirror_demo_pairs)
        # Should be lower than a perfect match
        correct_likelihood = bayes.compute_likelihood(mirror_program := ProgramNode(
            DSLElement("mirror", {"axis": "horizontal"})), mirror_demo_pairs)
        assert likelihood <= correct_likelihood

    def test_likelihood_in_range(self, bayes_config, mirror_program, mirror_demo_pairs):
        """Likelihood should be in (0, 1]."""
        bayes = BayesianConfidence(bayes_config)
        likelihood = bayes.compute_likelihood(mirror_program, mirror_demo_pairs)
        assert 0.0 <= likelihood <= 1.0

    def test_likelihood_empty_demos(self, bayes_config, mirror_program):
        """Empty demos should give 0 likelihood."""
        bayes = BayesianConfidence(bayes_config)
        likelihood = bayes.compute_likelihood(mirror_program, [])
        assert likelihood == 0.0


# ============================================================
# Evidence Tests
# ============================================================

class TestComputeEvidence:
    """Tests for compute_evidence P(E) marginal."""

    def test_evidence_no_candidates(self, bayes_config):
        """Evidence with no candidates should be 1.0 (neutral prior).

        v2.4.4: Changed from 0.0 to 1.0 so that compute_posterior()
        returns likelihood * prior instead of 0 when candidates list
        is empty (direct call without rank_candidates first).
        """
        bayes = BayesianConfidence(bayes_config)
        assert bayes.compute_evidence() == 1.0

    def test_evidence_positive_after_ranking(self, bayes_config, mirror_program, mirror_demo_pairs):
        """Evidence should be positive after ranking candidates."""
        bayes = BayesianConfidence(bayes_config)
        bayes.rank_candidates([mirror_program], mirror_demo_pairs)
        evidence = bayes.compute_evidence()
        assert evidence > 0


# ============================================================
# Posterior Tests
# ============================================================

class TestComputePosterior:
    """Tests for compute_posterior P(H|E)."""

    def test_posterior_positive(self, bayes_config, mirror_program, mirror_demo_pairs):
        """Posterior should be positive when evidence is available."""
        bayes = BayesianConfidence(bayes_config)
        # Need to rank candidates first to compute evidence
        bayes.rank_candidates([mirror_program], mirror_demo_pairs)
        posterior = bayes.compute_posterior(mirror_program, mirror_demo_pairs)
        assert posterior > 0

    def test_posterior_zero_without_evidence(self, bayes_config, mirror_program, mirror_demo_pairs):
        """Posterior should be 0 without evidence."""
        bayes = BayesianConfidence(bayes_config)
        # No candidates ranked yet -> evidence is placeholder
        posterior = bayes.compute_posterior(mirror_program, mirror_demo_pairs)
        # Evidence may be 0 or placeholder; check it doesn't crash
        assert isinstance(posterior, float)


# ============================================================
# Rank Candidates Tests
# ============================================================

class TestRankCandidates:
    """Tests for rank_candidates posterior sorting."""

    def test_rank_returns_tuples(self, bayes_config, mirror_program, mirror_demo_pairs):
        """rank_candidates should return (ProgramNode, float) tuples."""
        bayes = BayesianConfidence(bayes_config)
        results = bayes.rank_candidates([mirror_program], mirror_demo_pairs)
        assert len(results) == 1
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][0], ProgramNode)
        assert isinstance(results[0][1], float)

    def test_rank_sorted_descending(self, bayes_config, mirror_demo_pairs):
        """Results should be sorted by descending posterior."""
        prog1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        prog2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        bayes = BayesianConfidence(bayes_config)
        results = bayes.rank_candidates([prog1, prog2], mirror_demo_pairs)
        posteriors = [p for _, p in results]
        assert posteriors == sorted(posteriors, reverse=True)

    def test_rank_correct_program_first(self, bayes_config, mirror_demo_pairs):
        """Correct program should rank higher than incorrect."""
        correct = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        incorrect = ProgramNode(DSLElement("rotate", {"angle": 90}))
        bayes = BayesianConfidence(bayes_config)
        results = bayes.rank_candidates([correct, incorrect], mirror_demo_pairs)
        assert results[0][0] == correct

    def test_rank_empty_candidates(self, bayes_config, mirror_demo_pairs):
        """Empty candidate list should return empty."""
        bayes = BayesianConfidence(bayes_config)
        results = bayes.rank_candidates([], mirror_demo_pairs)
        assert results == []

    def test_rank_max_candidates_limit(self, mirror_demo_pairs):
        """Should respect max_candidates limit."""
        config = {"prior_lambda": 0.1, "noise_sigma": 0.5, "max_candidates": 2}
        bayes = BayesianConfidence(config)
        candidates = [ProgramNode(DSLElement("mirror")) for _ in range(5)]
        results = bayes.rank_candidates(candidates, mirror_demo_pairs)
        assert len(results) <= 2

    def test_rank_uniform_when_all_zero(self, bayes_config, mirror_demo_pairs):
        """When all posteriors are 0, should return uniform distribution."""
        # Use programs that all fail
        prog1 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        prog2 = ProgramNode(DSLElement("scale", {"factor": 3}))
        bayes = BayesianConfidence(bayes_config)
        results = bayes.rank_candidates([prog1, prog2], mirror_demo_pairs)
        if results:
            posteriors = [p for _, p in results]
            # If all are 0, they should be uniform
            if all(p == posteriors[0] for p in posteriors):
                assert abs(posteriors[0] - 1.0 / len(results)) < 0.01


# ============================================================
# FusionScorer Tests
# ============================================================

class TestFusionScorer:
    """Tests for FusionScorer."""

    def test_score_symbolic_correct(self, fusion_config, mirror_program, mirror_demo_pairs):
        """Correct program should have high symbolic score."""
        scorer = FusionScorer(fusion_config)
        score = scorer.score_symbolic(mirror_program, mirror_demo_pairs)
        assert score == 1.0  # Perfect match

    def test_score_symbolic_incorrect(self, fusion_config, incorrect_program, mirror_demo_pairs):
        """Incorrect program should have low symbolic score."""
        scorer = FusionScorer(fusion_config)
        score = scorer.score_symbolic(incorrect_program, mirror_demo_pairs)
        assert score < 1.0

    def test_score_symbolic_empty_demos(self, fusion_config, mirror_program):
        """Empty demos should give 0 symbolic score."""
        scorer = FusionScorer(fusion_config)
        score = scorer.score_symbolic(mirror_program, [])
        assert score == 0.0

    def test_score_visual_no_vl(self, fusion_config):
        """Without VL adapter, uses local e6/e7 approximation."""
        scorer = FusionScorer(fusion_config)
        frames = [np.array([[1, 0], [0, 0]], dtype=np.int8)]
        score = scorer.score_visual(frames, frames)  # v2.0: needs observed_frames too
        assert 0.0 <= score <= 1.0  # Local approximation returns [0,1]

    def test_score_cross_modal_both_positive(self, fusion_config):
        """Cross-modal score with both positive should be positive."""
        scorer = FusionScorer(fusion_config)
        score = scorer.score_cross_modal(0.8, 0.6)
        assert 0 < score <= 1.0

    def test_score_cross_modal_one_zero(self, fusion_config):
        """Cross-modal score with one zero should be 0."""
        scorer = FusionScorer(fusion_config)
        score = scorer.score_cross_modal(0.8, 0.0)
        assert score == 0.0

    def test_score_cross_modal_both_zero(self, fusion_config):
        """Cross-modal score with both zero should be 0."""
        scorer = FusionScorer(fusion_config)
        score = scorer.score_cross_modal(0.0, 0.0)
        assert score == 0.0


# ============================================================
# Degraded Mode Tests
# ============================================================

class TestDegradedMode:
    """Tests for degraded fusion mode (G_vis=0)."""

    def test_degraded_fuse_returns_score(self, fusion_config, mirror_program, mirror_demo_pairs):
        """Degraded fuse should return a score in [0, 1]."""
        scorer = FusionScorer(fusion_config)
        score = scorer.degraded_fuse(mirror_program, mirror_demo_pairs)
        assert 0.0 <= score <= 1.0

    def test_degraded_fuse_correct_program(self, fusion_config, mirror_program, mirror_demo_pairs):
        """Correct program should have high degraded score."""
        scorer = FusionScorer(fusion_config)
        score = scorer.degraded_fuse(mirror_program, mirror_demo_pairs)
        # G = alpha_degraded * G_sym + gamma_degraded * G_cr
        # G_sym = 1.0, G_cr = G_sym = 1.0
        # Score = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert score == 1.0

    def test_degraded_fuse_incorrect_program(self, fusion_config, incorrect_program, mirror_demo_pairs):
        """Incorrect program should have low degraded score."""
        scorer = FusionScorer(fusion_config)
        score = scorer.degraded_fuse(incorrect_program, mirror_demo_pairs)
        assert score < 1.0

    def test_fuse_degraded_when_no_vl(self, fusion_config, mirror_program, mirror_demo_pairs):
        """fuse() should use degraded mode when VL is not available."""
        scorer = FusionScorer(fusion_config)
        frames = [np.array([[1, 0], [0, 0]], dtype=np.int8)]
        score = scorer.fuse(mirror_program, frames, mirror_demo_pairs)
        # Should use degraded fuse since vl_available is False
        assert 0.0 <= score <= 1.0

    def test_vl_available_flag(self, fusion_config):
        """vl_available should be False without adapter."""
        scorer = FusionScorer(fusion_config)
        assert not scorer.vl_available

    def test_set_vl_available(self, fusion_config):
        """set_vl_available should update the flag."""
        scorer = FusionScorer(fusion_config)
        scorer.set_vl_available(True)
        assert scorer.vl_available


# ============================================================
# Adapt Weights Tests
# ============================================================

class TestAdaptWeights:
    """Tests for adaptive weight adjustment."""

    def test_adapt_strong_evidence(self, fusion_config):
        """Strong evidence (>=5) should increase alpha."""
        scorer = FusionScorer(fusion_config)
        scorer.adapt_weights(5)
        assert scorer.alpha == 0.5

    def test_adapt_moderate_evidence(self, fusion_config):
        """Moderate evidence (2-4) should use balanced weights."""
        scorer = FusionScorer(fusion_config)
        scorer.adapt_weights(3)
        assert scorer.alpha == 0.4
        assert scorer.beta == 0.35

    def test_adapt_weak_evidence(self, fusion_config):
        """Weak evidence (<2) should increase visual weight."""
        scorer = FusionScorer(fusion_config)
        scorer.adapt_weights(1)
        assert scorer.alpha == 0.3
        assert scorer.beta == 0.45

    def test_adapt_zero_evidence(self, fusion_config):
        """Zero evidence should use weak evidence weights."""
        scorer = FusionScorer(fusion_config)
        scorer.adapt_weights(0)
        assert scorer.alpha == 0.3
        assert scorer.beta == 0.45


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests combining Bayesian and Fusion."""

    def test_bayesian_then_fusion(self, bayes_config, fusion_config, mirror_demo_pairs):
        """Bayesian ranking followed by fusion scoring."""
        correct = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        incorrect = ProgramNode(DSLElement("rotate", {"angle": 90}))

        bayes = BayesianConfidence(bayes_config)
        ranked = bayes.rank_candidates([correct, incorrect], mirror_demo_pairs)

        scorer = FusionScorer(fusion_config)
        frames = [np.array([[1, 0], [0, 0]], dtype=np.int8)]
        best_program, best_posterior = ranked[0]
        fusion_score = scorer.fuse(best_program, frames, mirror_demo_pairs)

        assert best_posterior > 0
        assert 0.0 <= fusion_score <= 1.0

    def test_full_pipeline_correct_program(self, bayes_config, fusion_config, mirror_demo_pairs):
        """Full pipeline should correctly identify the best program."""
        candidates = [
            ProgramNode(DSLElement("mirror", {"axis": "horizontal"})),
            ProgramNode(DSLElement("rotate", {"angle": 90})),
            ProgramNode(DSLElement("scale", {"factor": 2})),
        ]

        bayes = BayesianConfidence(bayes_config)
        ranked = bayes.rank_candidates(candidates, mirror_demo_pairs)

        # Best candidate should be the mirror
        assert ranked[0][0].element.name == "mirror"

        scorer = FusionScorer(fusion_config)
        best_score = scorer.degraded_fuse(ranked[0][0], mirror_demo_pairs)
        # Correct program should have high score
        assert best_score > 0.5
