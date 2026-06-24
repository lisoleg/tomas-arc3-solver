"""Integration tests for TOMASSolver.solve_with_taiyi and solve_with_continual_learning.

Tests the integration entry points that connect:
    - κ-Snap abductive search (太一理论)
    - Continual learning pipeline (Wake-Sleep cycles)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.dsl_primitives import ProgramNode
from src.solver.tomas_solver import TOMASSolver

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real_arc_converted"


# ============================================================
# Helper Functions
# ============================================================

def load_task(task_id: str) -> dict:
    """Load a task JSON from the real_arc_converted directory."""
    path = DATA_DIR / f"{task_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tasks(task_ids: list[str]) -> list[dict]:
    """Load multiple tasks."""
    return [load_task(tid) for tid in task_ids]


def make_solver_config(tmpdir: Path) -> dict:
    """Create a minimal TOMASSolver config."""
    return {
        "library": {
            "persistence_path": str(tmpdir / "taiyi_library.json"),
            "frequency_threshold": 1,
            "max_abstractions": 200,
        },
        "search": {
            "time_limit_seconds": 30.0,
        },
        "gpu": {"enabled": False},
        "bayesian": {},
        "fusion": {},
        "mode_switch": {
            "video_time_threshold": 100,
            "bayesian_time_threshold": 200,
        },
    }


# ============================================================
# Test: TOMASSolver Construction
# ============================================================

class TestTOMASSolverConstruction(unittest.TestCase):
    """Test TOMASSolver can be constructed with minimal config."""

    def test_construction(self):
        """TOMASSolver constructs without error."""
        tmpdir = Path(tempfile.mkdtemp())
        config = make_solver_config(tmpdir)
        solver = TOMASSolver(config)
        self.assertIsNotNone(solver)
        self.assertIsNotNone(solver.searcher)
        self.assertIsNotNone(solver.library)


# ============================================================
# Test: solve_with_taiyi (κ-Snap Integration)
# ============================================================

class TestSolveWithTaiyi(unittest.TestCase):
    """Test solve_with_taiyi on single ARC tasks."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config = make_solver_config(self.tmpdir)
        self.solver = TOMASSolver(self.config)

    def test_returns_dict(self):
        """solve_with_taiyi returns a result dictionary."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        self.assertIsInstance(result, dict)

    def test_result_has_required_keys(self):
        """Result dict has required keys."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        self.assertIn("predictions", result)
        self.assertIn("best_program", result)
        self.assertIn("depth", result)
        self.assertIn("causal_log", result)
        self.assertIn("solved", result)
        self.assertIn("mode", result)

    def test_mode_is_taiyi_ksnap(self):
        """Mode is 'taiyi_ksnap'."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        self.assertEqual(result["mode"], "taiyi_ksnap")

    def test_solved_is_boolean(self):
        """solved is a boolean."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        self.assertIsInstance(result["solved"], bool)

    def test_depth_is_int(self):
        """depth is an integer."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        self.assertIsInstance(result["depth"], int)

    def test_causal_log_is_list(self):
        """causal_log is a list."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        self.assertIsInstance(result["causal_log"], list)

    def test_predictions_is_list(self):
        """predictions is a list."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        self.assertIsInstance(result["predictions"], list)

    def test_best_program_type(self):
        """best_program is ProgramNode or None."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        prog = result["best_program"]
        self.assertTrue(prog is None or isinstance(prog, ProgramNode))

    def test_solved_implies_program_not_none(self):
        """If solved=True, best_program should not be None."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        if result["solved"]:
            self.assertIsNotNone(result["best_program"])

    def test_not_solved_implies_program_none(self):
        """If solved=False, best_program should be None."""
        task = load_task("007bbfb7")
        result = self.solver.solve_with_taiyi(task)
        if not result["solved"]:
            self.assertIsNone(result["best_program"])

    def test_on_map_color_task(self):
        """solve_with_taiyi works on 0d3d703e (map-color)."""
        task = load_task("0d3d703e")
        result = self.solver.solve_with_taiyi(task)
        self.assertIsInstance(result, dict)
        self.assertIn("solved", result)

    def test_empty_task(self):
        """solve_with_taiyi on empty task returns solved=False."""
        result = self.solver.solve_with_taiyi({})
        self.assertFalse(result["solved"])
        self.assertEqual(result["depth"], 0)


# ============================================================
# Test: solve_with_continual_learning (Pipeline Integration)
# ============================================================

class TestSolveWithContinualLearning(unittest.TestCase):
    """Test solve_with_continual_learning on multiple ARC tasks."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config = make_solver_config(self.tmpdir)
        self.solver = TOMASSolver(self.config)

    def test_returns_dict(self):
        """solve_with_continual_learning returns a result dictionary."""
        tasks = load_tasks(["007bbfb7", "0d3d703e"])
        result = self.solver.solve_with_continual_learning(tasks, epochs=2)
        self.assertIsInstance(result, dict)

    def test_result_has_required_keys(self):
        """Result dict has required keys."""
        tasks = load_tasks(["007bbfb7", "0d3d703e"])
        result = self.solver.solve_with_continual_learning(tasks, epochs=1)
        self.assertIn("epoch_results", result)
        self.assertIn("final_accuracy", result)
        self.assertIn("total_new_primitives", result)
        self.assertIn("accuracy_history", result)
        self.assertIn("library_size", result)
        self.assertIn("mode", result)

    def test_mode_is_continual_learning(self):
        """Mode is 'continual_learning'."""
        tasks = load_tasks(["007bbfb7"])
        result = self.solver.solve_with_continual_learning(tasks, epochs=1)
        self.assertEqual(result["mode"], "continual_learning")

    def test_epoch_results_length_matches_epochs(self):
        """epoch_results has one entry per epoch."""
        tasks = load_tasks(["007bbfb7", "0d3d703e"])
        result = self.solver.solve_with_continual_learning(tasks, epochs=2)
        self.assertEqual(len(result["epoch_results"]), 2)

    def test_accuracy_history_length_matches_epochs(self):
        """accuracy_history has one entry per epoch."""
        tasks = load_tasks(["007bbfb7"])
        result = self.solver.solve_with_continual_learning(tasks, epochs=2)
        self.assertEqual(len(result["accuracy_history"]), 2)

    def test_final_accuracy_matches_last_epoch(self):
        """final_accuracy matches the last epoch's accuracy."""
        tasks = load_tasks(["007bbfb7", "0d3d703e"])
        result = self.solver.solve_with_continual_learning(tasks, epochs=2)
        last_epoch = result["epoch_results"][-1]
        self.assertAlmostEqual(
            result["final_accuracy"], last_epoch["accuracy"], places=4
        )

    def test_total_new_primitives_non_negative(self):
        """total_new_primitives is non-negative."""
        tasks = load_tasks(["007bbfb7"])
        result = self.solver.solve_with_continual_learning(tasks, epochs=1)
        self.assertGreaterEqual(result["total_new_primitives"], 0)

    def test_accuracy_in_valid_range(self):
        """All accuracy values in [0, 1]."""
        tasks = load_tasks(["007bbfb7", "0d3d703e"])
        result = self.solver.solve_with_continual_learning(tasks, epochs=2)
        for acc in result["accuracy_history"]:
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
        self.assertGreaterEqual(result["final_accuracy"], 0.0)
        self.assertLessEqual(result["final_accuracy"], 1.0)

    def test_empty_tasks(self):
        """Empty task list yields 0 final_accuracy."""
        result = self.solver.solve_with_continual_learning([], epochs=1)
        self.assertAlmostEqual(result["final_accuracy"], 0.0)

    def test_library_size_non_negative(self):
        """library_size is non-negative."""
        tasks = load_tasks(["007bbfb7"])
        result = self.solver.solve_with_continual_learning(tasks, epochs=1)
        self.assertGreaterEqual(result["library_size"], 0)


# ============================================================
# Test: Consistency Between solve_with_taiyi and Continual Learning
# ============================================================

class TestIntegrationConsistency(unittest.TestCase):
    """Verify solve_with_taiyi and continual learning are consistent."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.config = make_solver_config(self.tmpdir)
        self.solver = TOMASSolver(self.config)

    def test_continual_learning_epoch_0_matches_single_solve(self):
        """Epoch 0 of continual learning matches solve_with_taiyi on same task."""
        task = load_task("007bbfb7")
        # Single solve
        single_result = self.solver.solve_with_taiyi(task)
        # Continual learning with 1 epoch
        cl_result = self.solver.solve_with_continual_learning([task], epochs=1)
        epoch_0 = cl_result["epoch_results"][0]
        # The solved status should match
        self.assertEqual(epoch_0["solved"] > 0, single_result["solved"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
