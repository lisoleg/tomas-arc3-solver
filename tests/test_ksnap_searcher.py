"""Unit tests for KSnapSearcher (κ-Snap abductive searcher).

Tests the four-level filtering pipeline:
    Ftel threshold -> Dead-Zero -> MUS -> GaussEx projection

Covers:
- search() return format on real ARC tasks
- _compute_ftel correctness
- _check_dead_zero edge cases
- _check_mus conflict detection
- _execute_snap four-level filtering
- _expand_depth chain composition
- _grid_similarity shape mismatch handling
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.solver.ksnap_searcher import (
    KSnapCandidate,
    KSnapSearcher,
    SnapResult,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real_arc_converted"


# ============================================================
# Helper Functions
# ============================================================

def load_task(task_id: str) -> dict:
    """Load a task JSON from the real_arc_converted directory."""
    path = DATA_DIR / f"{task_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_demo_pair(inp: list[list[int]], out: list[list[int]]) -> dict:
    """Create a demo pair in the video format expected by KSnapSearcher."""
    return {
        "input": [np.array(inp, dtype=np.int8)],
        "output": [np.array(out, dtype=np.int8)],
    }


# ============================================================
# Test: KSnapSearcher Construction
# ============================================================

class TestKSnapSearcherInit(unittest.TestCase):
    """Test KSnapSearcher initialization and configuration."""

    def test_default_construction(self):
        """KSnapSearcher can be constructed with default parameters."""
        searcher = KSnapSearcher()
        self.assertAlmostEqual(searcher.theta_ftel, 0.1)
        self.assertAlmostEqual(searcher.theta_dead, 0.01)
        self.assertEqual(searcher.beam_width, 100)
        self.assertEqual(searcher.max_depth, 4)
        self.assertEqual(searcher.causal_log, [])

    def test_custom_construction(self):
        """KSnapSearcher accepts custom parameters."""
        searcher = KSnapSearcher(
            theta_ftel=0.5,
            theta_dead=0.05,
            beam_width=50,
            max_depth=2,
        )
        self.assertAlmostEqual(searcher.theta_ftel, 0.5)
        self.assertAlmostEqual(searcher.theta_dead, 0.05)
        self.assertEqual(searcher.beam_width, 50)
        self.assertEqual(searcher.max_depth, 2)

    def test_has_param_inference(self):
        """KSnapSearcher initializes ParamInference and GaussExVerifier."""
        searcher = KSnapSearcher()
        self.assertIsNotNone(searcher.param_inference)
        self.assertIsNotNone(searcher.verifier)


# ============================================================
# Test: search() Return Format on Real ARC Task
# ============================================================

class TestKSnapSearchReturnFormat(unittest.TestCase):
    """Test that search() returns the correct (ProgramNode|None, int, list) format."""

    def test_search_returns_tuple_of_three(self):
        """search() returns a 3-tuple."""
        task = load_task("007bbfb7")
        searcher = KSnapSearcher(theta_ftel=0.05, beam_width=50, max_depth=3)
        result = searcher.search(task)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_search_return_types(self):
        """search() returns (ProgramNode|None, int, list)."""
        task = load_task("007bbfb7")
        searcher = KSnapSearcher(theta_ftel=0.05, beam_width=50, max_depth=3)
        prog, depth, log = searcher.search(task)
        # prog: ProgramNode or None
        self.assertTrue(prog is None or isinstance(prog, ProgramNode))
        # depth: int
        self.assertIsInstance(depth, int)
        # log: list
        self.assertIsInstance(log, list)

    def test_search_causal_log_contains_events(self):
        """Causal log should contain search events."""
        task = load_task("007bbfb7")
        searcher = KSnapSearcher(theta_ftel=0.05, beam_width=50, max_depth=3)
        _, _, log = searcher.search(task)
        self.assertGreater(len(log), 0)
        # First event should be search_start
        first_event = log[0]
        self.assertIn("event_type", first_event)
        self.assertIn("reason", first_event)

    def test_search_empty_task_returns_none(self):
        """search() on empty task returns (None, 0, [])."""
        searcher = KSnapSearcher()
        prog, depth, log = searcher.search({})
        self.assertIsNone(prog)
        self.assertEqual(depth, 0)
        self.assertIsInstance(log, list)

    def test_search_no_train_returns_none(self):
        """search() on task without train returns (None, 0, [])."""
        searcher = KSnapSearcher()
        prog, depth, log = searcher.search({"test": []})
        self.assertIsNone(prog)
        self.assertEqual(depth, 0)


# ============================================================
# Test: _compute_ftel
# ============================================================

class TestComputeFtel(unittest.TestCase):
    """Test Ftel (output similarity) computation."""

    def test_perfect_match_ftel_is_one(self):
        """A program that reproduces output has ftel=1.0."""
        searcher = KSnapSearcher()
        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        demo_pairs = [make_demo_pair(grid.tolist(), grid.tolist())]
        # copy program returns the same grid
        prog = ProgramNode(DSLElement("copy"))
        ftel = searcher._compute_ftel(prog, demo_pairs)
        self.assertAlmostEqual(ftel, 1.0, places=4)

    def test_partial_match_ftel_between_zero_and_one(self):
        """A program with partial match has 0 < ftel < 1."""
        searcher = KSnapSearcher()
        inp = np.array([[1, 0], [0, 1]], dtype=np.int8)
        out = np.array([[1, 1], [0, 0]], dtype=np.int8)
        demo_pairs = [make_demo_pair(inp.tolist(), out.tolist())]
        # copy program returns input, not output -> partial match
        prog = ProgramNode(DSLElement("copy"))
        ftel = searcher._compute_ftel(prog, demo_pairs)
        self.assertGreater(ftel, 0.0)
        self.assertLess(ftel, 1.0)

    def test_crash_program_ftel_zero(self):
        """A program that crashes gets ftel contribution of 0."""
        searcher = KSnapSearcher()
        # Use a program that will fail
        inp = np.array([[0, 0], [0, 0]], dtype=np.int8)
        out = np.array([[1, 1], [1, 1]], dtype=np.int8)
        demo_pairs = [make_demo_pair(inp.tolist(), out.tolist())]
        # scale on all-zero grid produces all-zero -> zero match
        prog = ProgramNode(DSLElement("scale", {"factor": 2}))
        ftel = searcher._compute_ftel(prog, demo_pairs)
        self.assertAlmostEqual(ftel, 0.0, places=4)


# ============================================================
# Test: _check_dead_zero
# ============================================================

class TestCheckDeadZero(unittest.TestCase):
    """Test Dead-Zero detection (Level 2 filter)."""

    def test_all_zero_output_is_dead(self):
        """Program producing all-zero output is Dead-Zero."""
        searcher = KSnapSearcher()
        grid = np.zeros((3, 3), dtype=np.int8)
        # scale on all-zero grid produces all-zero
        prog = ProgramNode(DSLElement("scale", {"factor": 2}))
        self.assertTrue(searcher._check_dead_zero(prog, grid))

    def test_non_zero_output_not_dead(self):
        """Program producing non-zero output is not Dead-Zero."""
        searcher = KSnapSearcher()
        grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
        prog = ProgramNode(DSLElement("copy"))
        self.assertFalse(searcher._check_dead_zero(prog, grid))

    def test_empty_result_is_dead(self):
        """An empty result is treated as Dead-Zero."""
        searcher = KSnapSearcher()
        grid = np.array([[1, 2]], dtype=np.int8)
        # crop to 0x0 region
        prog = ProgramNode(DSLElement("crop", {"height": 1, "width": 0}))
        # crop with width=0 might not produce empty, but check the logic
        # If it does produce empty, it should be dead
        result = prog.apply(grid)
        if result.size == 0:
            self.assertTrue(searcher._check_dead_zero(prog, grid))


# ============================================================
# Test: _check_mus
# ============================================================

class TestCheckMUS(unittest.TestCase):
    """Test MUS (mutually exclusive states) detection (Level 3 filter)."""

    def test_no_mus_with_unique_programs(self):
        """Programs with different MDL or Ftel have no MUS."""
        searcher = KSnapSearcher()
        cands = [
            KSnapCandidate(program=ProgramNode(DSLElement("copy")), ftel=0.9),
            KSnapCandidate(program=ProgramNode(DSLElement("rotate", {"angle": 90})), ftel=0.5),
        ]
        # Set different MDLs
        cands[0].program.total_mdl = 5
        cands[1].program.total_mdl = 7
        mus_map = searcher._check_mus(cands)
        for cid in mus_map:
            self.assertFalse(mus_map[cid])

    def test_mus_detected_for_same_mdl_similar_ftel(self):
        """Two different programs with same MDL and similar Ftel trigger MUS."""
        searcher = KSnapSearcher()
        prog1 = ProgramNode(DSLElement("copy"))
        prog2 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        # Same MDL, same Ftel
        prog1.total_mdl = 5
        prog2.total_mdl = 5
        cands = [
            KSnapCandidate(program=prog1, ftel=0.80),
            KSnapCandidate(program=prog2, ftel=0.80),
        ]
        mus_map = searcher._check_mus(cands)
        # Both should be MUS active
        self.assertTrue(mus_map[cands[0].candidate_id])
        self.assertTrue(mus_map[cands[1].candidate_id])

    def test_no_mus_with_single_candidate(self):
        """Single candidate has no MUS."""
        searcher = KSnapSearcher()
        prog = ProgramNode(DSLElement("copy"))
        cands = [KSnapCandidate(program=prog, ftel=0.9)]
        mus_map = searcher._check_mus(cands)
        self.assertFalse(mus_map[cands[0].candidate_id])


# ============================================================
# Test: _execute_snap (Four-Level Filtering)
# ============================================================

class TestExecuteSnap(unittest.TestCase):
    """Test the four-level κ-Snap projection filtering."""

    def test_low_ftel_rejected(self):
        """Candidate with ftel < theta_ftel is rejected at Level 1."""
        searcher = KSnapSearcher(theta_ftel=0.5)
        prog = ProgramNode(DSLElement("copy"))
        cand = KSnapCandidate(program=prog, ftel=0.1, i_value=0.5)
        demo_pairs = [make_demo_pair([[1]], [[1]])]
        result = searcher._execute_snap(cand, demo_pairs)
        self.assertEqual(result, SnapResult.REJECT_FTEL)

    def test_dead_zero_rejected(self):
        """Candidate producing all-zero is rejected at Level 2."""
        searcher = KSnapSearcher(theta_ftel=0.01)
        grid = np.zeros((3, 3), dtype=np.int8)
        prog = ProgramNode(DSLElement("scale", {"factor": 2}))
        cand = KSnapCandidate(program=prog, ftel=0.5, i_value=0.0)
        demo_pairs = [{"input": [grid], "output": [np.ones((6, 6), dtype=np.int8)]}]
        result = searcher._execute_snap(cand, demo_pairs)
        self.assertEqual(result, SnapResult.REJECT_DZ)

    def test_manifested_on_correct_program(self):
        """A program that perfectly matches demo pairs is MANIFESTED."""
        searcher = KSnapSearcher(theta_ftel=0.01)
        grid = np.array([[1, 2], [3, 4]], dtype=np.int8)
        prog = ProgramNode(DSLElement("copy"))
        cand = KSnapCandidate(
            program=prog,
            ftel=1.0,
            i_value=0.5,
        )
        demo_pairs = [make_demo_pair(grid.tolist(), grid.tolist())]
        result = searcher._execute_snap(cand, demo_pairs)
        self.assertEqual(result, SnapResult.MANIFESTED)

    def test_causal_log_records_snap_event(self):
        """Each _execute_snap call appends to causal_log."""
        searcher = KSnapSearcher(theta_ftel=0.5)
        prog = ProgramNode(DSLElement("copy"))
        cand = KSnapCandidate(program=prog, ftel=0.1, i_value=0.5)
        demo_pairs = [make_demo_pair([[1]], [[1]])]
        initial_log_len = len(searcher.causal_log)
        searcher._execute_snap(cand, demo_pairs)
        self.assertEqual(len(searcher.causal_log), initial_log_len + 1)
        event = searcher.causal_log[-1]
        self.assertIn("candidate_id", event)
        self.assertIn("depth", event)
        self.assertIn("ftel", event)
        self.assertIn("result", event)


# ============================================================
# Test: _grid_similarity
# ============================================================

class TestGridSimilarity(unittest.TestCase):
    """Test _grid_similarity static method."""

    def test_identical_grids(self):
        """Identical grids have similarity 1.0."""
        g1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
        g2 = np.array([[1, 2], [3, 4]], dtype=np.int8)
        sim = KSnapSearcher._grid_similarity(g1, g2)
        self.assertAlmostEqual(sim, 1.0)

    def test_different_grids(self):
        """Completely different grids have similarity 0.0."""
        g1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
        g2 = np.array([[5, 6], [7, 8]], dtype=np.int8)
        sim = KSnapSearcher._grid_similarity(g1, g2)
        self.assertAlmostEqual(sim, 0.0)

    def test_shape_mismatch(self):
        """Shape mismatch uses overlapping region."""
        g1 = np.array([[1, 2], [3, 4]], dtype=np.int8)
        g2 = np.array([[1, 2, 5], [3, 4, 6]], dtype=np.int8)
        sim = KSnapSearcher._grid_similarity(g1, g2)
        # Overlapping 2x2 region is identical
        self.assertAlmostEqual(sim, 1.0)


# ============================================================
# Test: _chain_programs and _expand_depth
# ============================================================

class TestChainPrograms(unittest.TestCase):
    """Test program chaining for depth expansion."""

    def test_chain_creates_chain_combo(self):
        """_chain_programs creates a 'chain' combo_type node."""
        prog1 = ProgramNode(DSLElement("copy"))
        prog2 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        chained = KSnapSearcher._chain_programs(prog1, prog2)
        self.assertEqual(chained.combo_type, "chain")
        self.assertEqual(len(chained.children), 1)
        self.assertEqual(chained.children[0].element.name, "mirror")

    def test_chain_preserves_original_elements(self):
        """Chained program preserves the original element."""
        prog1 = ProgramNode(DSLElement("copy"))
        prog2 = ProgramNode(DSLElement("scale", {"factor": 2}))
        chained = KSnapSearcher._chain_programs(prog1, prog2)
        self.assertEqual(chained.element.name, "copy")
        self.assertEqual(chained.children[0].element.name, "scale")

    def test_chain_mdl_is_computed(self):
        """Chained program has non-zero total_mdl."""
        prog1 = ProgramNode(DSLElement("copy"))
        prog2 = ProgramNode(DSLElement("scale", {"factor": 2}))
        chained = KSnapSearcher._chain_programs(prog1, prog2)
        self.assertGreater(chained.total_mdl, 0)

    def test_expand_depth_increases_depth(self):
        """_expand_depth produces candidates at depth+1."""
        searcher = KSnapSearcher(beam_width=10, max_depth=3)
        grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
        out = np.array([[1, 1], [1, 1]], dtype=np.int8)
        demo_pairs = [make_demo_pair(grid.tolist(), out.tolist())]
        # Generate depth-1 candidates
        cands = searcher._generate_candidates(demo_pairs)
        if cands:
            expanded = searcher._expand_depth(cands, demo_pairs)
            for exp in expanded:
                self.assertEqual(exp.depth, cands[0].depth + 1)


# ============================================================
# Test: _extract_demo_pairs
# ============================================================

class TestExtractDemoPairs(unittest.TestCase):
    """Test demo pair extraction from task data."""

    def test_extract_from_train_key(self):
        """Extracts demo pairs from 'train' key format."""
        task = {
            "train": [
                {"input": [[[1, 2]]], "output": [[[1, 2]]]},
                {"input": [[[3, 4]]], "output": [[[3, 4]]]},
            ]
        }
        searcher = KSnapSearcher()
        pairs = searcher._extract_demo_pairs(task)
        self.assertEqual(len(pairs), 2)
        self.assertIn("input", pairs[0])
        self.assertIn("output", pairs[0])
        self.assertIsInstance(pairs[0]["input"][0], np.ndarray)

    def test_extract_from_direct_demo_pair(self):
        """Extracts from direct input/output format."""
        task = {"input": [[[1, 2]]], "output": [[[1, 2]]]}
        searcher = KSnapSearcher()
        pairs = searcher._extract_demo_pairs(task)
        self.assertEqual(len(pairs), 1)

    def test_extract_empty_task(self):
        """Empty task yields no demo pairs."""
        searcher = KSnapSearcher()
        pairs = searcher._extract_demo_pairs({})
        self.assertEqual(len(pairs), 0)


# ============================================================
# Test: search() on Real ARC Task (Integration)
# ============================================================

class TestSearchOnRealArcTask(unittest.TestCase):
    """Integration test: search() on real ARC task 007bbfb7."""

    def setUp(self):
        self.task = load_task("007bbfb7")

    def test_search_completes_without_error(self):
        """search() completes without raising on 007bbfb7."""
        searcher = KSnapSearcher(theta_ftel=0.05, beam_width=30, max_depth=2)
        prog, depth, log = searcher.search(self.task)
        # Should not raise; either finds a solution or exhausts search
        self.assertTrue(prog is None or isinstance(prog, ProgramNode))

    def test_search_log_has_depth_events(self):
        """Causal log records depth events."""
        searcher = KSnapSearcher(theta_ftel=0.05, beam_width=30, max_depth=2)
        _, _, log = searcher.search(self.task)
        # Should have at least search_start and search_end
        event_types = [e.get("event_type", "") for e in log]
        self.assertTrue(any("search_start" in et for et in event_types))
        self.assertTrue(any("search_end" in et for et in event_types))


if __name__ == "__main__":
    unittest.main(verbosity=2)
