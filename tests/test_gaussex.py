"""GaussEx interconnected verification tests.

Tests demo constraint -> Willems behavioral system fiber mapping,
candidate program verification against all demo pairs (fiber
intersection minimization), fiber intersection computation, and
cross-video transfer.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.solver.gaussex_verifier import GaussExVerifier


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mirror_demo_pairs():
    """Demo pairs where output is horizontal mirror of input."""
    grids_in = [
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.int8),
        np.array([[5, 0], [0, 7]], dtype=np.int8),
    ]
    grids_out = [np.fliplr(g) for g in grids_in]
    return [{"input": grids_in, "output": grids_out}]


@pytest.fixture
def verifier():
    """Create a GaussExVerifier instance."""
    return GaussExVerifier()


@pytest.fixture
def mirror_program():
    """A correct mirror program."""
    return ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))


@pytest.fixture
def rotate_program():
    """An incorrect rotate program."""
    return ProgramNode(DSLElement("rotate", {"angle": 90}))


# ============================================================
# Verify Program Tests
# ============================================================

class TestVerifyProgram:
    """Tests for program verification against demo pairs."""

    def test_correct_program_passes(self, verifier, mirror_program, mirror_demo_pairs):
        """A correct program should pass verification."""
        assert verifier.verify_program(mirror_program, mirror_demo_pairs)

    def test_incorrect_program_fails(self, verifier, rotate_program, mirror_demo_pairs):
        """An incorrect program should fail verification."""
        assert not verifier.verify_program(rotate_program, mirror_demo_pairs)

    def test_verify_empty_demos(self, verifier, mirror_program):
        """Verification with empty demos should pass (vacuously true)."""
        assert verifier.verify_program(mirror_program, [])

    def test_verify_multiple_demos(self, verifier):
        """Verification should check all demo pairs."""
        demos = [
            {"input": [np.array([[1, 0], [0, 0]], dtype=np.int8)],
             "output": [np.array([[0, 1], [0, 0]], dtype=np.int8)]},
            {"input": [np.array([[0, 2], [0, 0]], dtype=np.int8)],
             "output": [np.array([[2, 0], [0, 0]], dtype=np.int8)]},
        ]
        program = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        assert verifier.verify_program(program, demos)

    def test_verify_partial_match_fails(self, verifier):
        """A program matching only some demos should fail."""
        demos = [
            {"input": [np.array([[1, 0], [0, 0]], dtype=np.int8)],
             "output": [np.array([[0, 1], [0, 0]], dtype=np.int8)]},  # mirror
            {"input": [np.array([[1, 0], [0, 0]], dtype=np.int8)],
             "output": [np.array([[1, 0], [0, 0]], dtype=np.int8)]},  # identity (not mirror)
        ]
        program = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        assert not verifier.verify_program(program, demos)


# ============================================================
# Fiber Intersection Tests
# ============================================================

class TestFiberIntersection:
    """Tests for fiber intersection computation."""

    def test_empty_programs_returns_empty(self, verifier):
        """Empty program list should return empty intersection."""
        result = verifier.compute_fiber_intersection([])
        assert result == set()

    def test_single_program_fiber(self, verifier, mirror_demo_pairs):
        """Single program should produce a non-empty fiber."""
        verifier.set_demo_pairs(mirror_demo_pairs)
        program = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        fiber = verifier.compute_fiber_intersection([program])
        assert isinstance(fiber, set)
        assert len(fiber) >= 1

    def test_intersection_of_same_programs(self, verifier, mirror_demo_pairs):
        """Intersection of identical programs should be non-empty."""
        verifier.set_demo_pairs(mirror_demo_pairs)
        program = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        fiber = verifier.compute_fiber_intersection([program, program])
        assert len(fiber) >= 1

    def test_intersection_of_different_programs(self, verifier, mirror_demo_pairs):
        """Intersection of different programs may be empty or not."""
        verifier.set_demo_pairs(mirror_demo_pairs)
        prog1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        prog2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        fiber = verifier.compute_fiber_intersection([prog1, prog2])
        assert isinstance(fiber, set)

    def test_fiber_is_set_of_ints(self, verifier, mirror_demo_pairs):
        """Fiber should be a set of integers (state hashes)."""
        verifier.set_demo_pairs(mirror_demo_pairs)
        program = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        fiber = verifier.compute_fiber_intersection([program])
        for state_hash in fiber:
            assert isinstance(state_hash, int)


# ============================================================
# Cross-Video Transfer Tests
# ============================================================

class TestCrossVideoTransfer:
    """Tests for cross-video program transfer."""

    def test_transferable_programs(self, verifier, mirror_program):
        """Programs valid across videos should transfer."""
        # Both videos use mirror transformation
        video1 = [
            {"input": [np.array([[1, 0], [0, 0]], dtype=np.int8)],
             "output": [np.array([[0, 1], [0, 0]], dtype=np.int8)]},
        ]
        video2 = [
            {"input": [np.array([[0, 2], [0, 0]], dtype=np.int8)],
             "output": [np.array([[2, 0], [0, 0]], dtype=np.int8)]},
        ]
        transferable = verifier.cross_video_transfer(
            [mirror_program], [video2]
        )
        assert mirror_program in transferable

    def test_non_transferable_programs(self, verifier, rotate_program):
        """Programs invalid for other videos should not transfer."""
        video2 = [
            {"input": [np.array([[1, 0], [0, 0]], dtype=np.int8)],
             "output": [np.array([[0, 1], [0, 0]], dtype=np.int8)]},  # mirror, not rotate
        ]
        transferable = verifier.cross_video_transfer(
            [rotate_program], [video2]
        )
        assert rotate_program not in transferable

    def test_transfer_empty_programs(self, verifier):
        """Empty program list should return empty."""
        result = verifier.cross_video_transfer([], [])
        assert result == []

    def test_transfer_multiple_videos(self, verifier, mirror_program):
        """Transfer should check against all provided videos."""
        video1 = [
            {"input": [np.array([[1, 0], [0, 0]], dtype=np.int8)],
             "output": [np.array([[0, 1], [0, 0]], dtype=np.int8)]},
        ]
        video2 = [
            {"input": [np.array([[3, 0], [0, 0]], dtype=np.int8)],
             "output": [np.array([[0, 3], [0, 0]], dtype=np.int8)]},
        ]
        video3 = [
            {"input": [np.array([[0, 0], [0, 5]], dtype=np.int8)],
             "output": [np.array([[0, 0], [5, 0]], dtype=np.int8)]},
        ]
        transferable = verifier.cross_video_transfer(
            [mirror_program], [video2, video3]
        )
        assert mirror_program in transferable


# ============================================================
# Minimize Fiber Intersection Tests
# ============================================================

class TestMinimizeFiberIntersection:
    """Tests for minimize_fiber_intersection."""

    def test_empty_candidates_returns_none(self, verifier):
        """Empty candidate list should return None."""
        assert verifier.minimize_fiber_intersection([]) is None

    def test_returns_a_program(self, verifier, mirror_demo_pairs):
        """Should return one of the candidates."""
        verifier.set_demo_pairs(mirror_demo_pairs)
        prog1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        prog2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        result = verifier.minimize_fiber_intersection([prog1, prog2])
        assert result is not None
        assert result in [prog1, prog2]

    def test_single_candidate(self, verifier, mirror_demo_pairs):
        """Single candidate should be returned."""
        verifier.set_demo_pairs(mirror_demo_pairs)
        prog = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        result = verifier.minimize_fiber_intersection([prog])
        assert result == prog


# ============================================================
# Set Demo Pairs Tests
# ============================================================

class TestSetDemoPairs:
    """Tests for set_demo_pairs method."""

    def test_set_demo_pairs(self, verifier, mirror_demo_pairs):
        """set_demo_pairs should store the pairs."""
        verifier.set_demo_pairs(mirror_demo_pairs)
        assert verifier.demo_pairs == mirror_demo_pairs

    def test_set_demo_pairs_clears_fibers(self, verifier, mirror_demo_pairs):
        """set_demo_pairs should clear the fiber cache."""
        verifier.fibers = {0: set([1, 2, 3])}
        verifier.set_demo_pairs(mirror_demo_pairs)
        assert verifier.fibers == {}

    def test_compute_fiber_for_pair(self, verifier, mirror_demo_pairs):
        """compute_fiber_for_pair should return a set."""
        verifier.set_demo_pairs(mirror_demo_pairs)
        program = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        fiber = verifier.compute_fiber_for_pair(program, 0)
        assert isinstance(fiber, set)
        assert len(fiber) >= 1

    def test_compute_fiber_invalid_pair_idx(self, verifier, mirror_demo_pairs):
        """Invalid pair index should return empty set."""
        verifier.set_demo_pairs(mirror_demo_pairs)
        program = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        fiber = verifier.compute_fiber_for_pair(program, 999)
        assert fiber == set()
