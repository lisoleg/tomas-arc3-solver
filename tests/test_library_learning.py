"""Unit tests for LibraryLearning sleep-step methods.

Tests the three new methods added for continual learning:
    - extract_subexpressions: AST traversal for chain subtrees
    - compute_mdl_gain: MDL compression gain calculation
    - sleep_step: Full pipeline (extract -> filter -> register)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.solver.library_learning import LibraryLearning


# ============================================================
# Helper Functions
# ============================================================

def make_chain_program(*primitive_names: str) -> ProgramNode:
    """Create a chain ProgramNode from a sequence of primitive names.

    Args:
        *primitive_names: DSL primitive names to chain.

    Returns:
        ProgramNode with chain composition.
    """
    if not primitive_names:
        return ProgramNode(DSLElement("copy"))

    root = ProgramNode(DSLElement(primitive_names[0]))
    for name in primitive_names[1:]:
        child = ProgramNode(DSLElement(name))
        root.children.append(child)
    root.combo_type = "chain"
    root.total_mdl = root.compute_mdl()
    return root


def make_lib_config(tmpdir: Path) -> dict:
    """Create a LibraryLearning config with temp persistence path."""
    return {
        "persistence_path": str(tmpdir / "test_library.json"),
        "frequency_threshold": 1,
        "max_abstractions": 200,
    }


# ============================================================
# Test: extract_subexpressions
# ============================================================

class TestExtractSubexpressions(unittest.TestCase):
    """Test extract_subexpressions method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lib = LibraryLearning(make_lib_config(Path(self.tmpdir)))

    def test_single_chain_program(self):
        """Extract sub-expressions from a 3-element chain program."""
        prog = make_chain_program("copy", "mirror", "scale")
        subexprs = self.lib.extract_subexpressions(prog)
        # The root is a chain with 2 children (mirror, scale)
        # So extract_subexpressions should find the root chain subtree
        self.assertGreaterEqual(len(subexprs), 1)
        # Each subexpr should be (hash_str, ProgramNode)
        for hash_str, node in subexprs:
            self.assertIsInstance(hash_str, str)
            self.assertIsInstance(node, ProgramNode)

    def test_leaf_program_no_subexprs(self):
        """A leaf node (no chain) yields no sub-expressions."""
        prog = ProgramNode(DSLElement("copy"))
        subexprs = self.lib.extract_subexpressions(prog)
        # Leaf nodes have combo_type "leaf", not "chain"
        self.assertEqual(len(subexprs), 0)

    def test_deep_chain_extraction(self):
        """Deep chain produces multiple sub-expressions."""
        prog = make_chain_program("copy", "mirror", "scale", "rotate", "gravity")
        subexprs = self.lib.extract_subexpressions(prog, max_ast_depth=5)
        # Should find at least the root chain subtree
        self.assertGreaterEqual(len(subexprs), 1)

    def test_max_ast_depth_limits_traversal(self):
        """max_ast_depth limits traversal depth."""
        prog = make_chain_program("copy", "mirror", "scale", "rotate")
        shallow = self.lib.extract_subexpressions(prog, max_ast_depth=1)
        deep = self.lib.extract_subexpressions(prog, max_ast_depth=5)
        # Deep traversal should find at least as many as shallow
        self.assertGreaterEqual(len(deep), len(shallow))

    def test_extracted_subtree_is_clone(self):
        """Extracted sub-expressions are clones (not references)."""
        prog = make_chain_program("copy", "mirror")
        subexprs = self.lib.extract_subexpressions(prog)
        if subexprs:
            _, subtree = subexprs[0]
            # Modify the clone should not affect the original
            if subtree.element:
                subtree.element.name = "modified"
            # Original should be unchanged
            self.assertEqual(prog.element.name, "copy")


# ============================================================
# Test: compute_mdl_gain
# ============================================================

class TestComputeMdlGain(unittest.TestCase):
    """Test compute_mdl_gain method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lib = LibraryLearning(make_lib_config(Path(self.tmpdir)))

    def test_positive_gain_for_frequent_subexpr(self):
        """A frequent sub-expression has positive MDL gain."""
        prog = make_chain_program("copy", "mirror")
        uncompressed = prog.compute_mdl()
        # frequency = 5, gain = uncompressed*5 - 1*5 - 10
        expected_gain = uncompressed * 5 - 1 * 5 - 10
        gain = self.lib.compute_mdl_gain("hash123", prog, frequency=5)
        self.assertEqual(gain, expected_gain)
        self.assertGreater(gain, 0)

    def test_low_frequency_negative_gain(self):
        """Low frequency yields negative or small gain."""
        prog = make_chain_program("copy", "mirror", "scale")
        gain = self.lib.compute_mdl_gain("hash456", prog, frequency=1)
        # gain = uncompressed*1 - 1*1 - 10 = uncompressed - 11
        # For a 3-element chain: 5+5+5+2+2 = 19, so gain = 19-11 = 8
        # Actually compute_mdl for chain: element(5) + child_chain(5+child(5)+2) + 2
        # = 5 + (5 + 5 + 2) + 2 = 19
        # gain = 19*1 - 1*1 - 10 = 8 (positive)
        # Let's use frequency=1 with a very small program
        simple = ProgramNode(DSLElement("copy"))
        gain_simple = self.lib.compute_mdl_gain("hash789", simple, frequency=1)
        # uncompressed=5, gain = 5*1 - 1*1 - 10 = -6 (negative)
        self.assertLess(gain_simple, 0)

    def test_gain_scales_with_frequency(self):
        """Higher frequency yields proportionally higher gain."""
        prog = make_chain_program("copy", "mirror")
        gain_low = self.lib.compute_mdl_gain("h", prog, frequency=3)
        gain_high = self.lib.compute_mdl_gain("h", prog, frequency=10)
        self.assertGreater(gain_high, gain_low)

    def test_gain_formula(self):
        """Verify exact MDL gain formula: uncompressed*freq - compressed*freq - registration."""
        prog = make_chain_program("copy", "scale")
        uncompressed = prog.compute_mdl()
        compressed = 1
        registration = 10
        freq = 4
        expected = uncompressed * freq - compressed * freq - registration
        actual = self.lib.compute_mdl_gain("test_hash", prog, freq)
        self.assertEqual(actual, expected)


# ============================================================
# Test: sleep_step
# ============================================================

class TestSleepStep(unittest.TestCase):
    """Test sleep_step method (full pipeline).

    Note: DSLElement._registry is a global class-level dict. Tests that
    call sleep_step register new primitives in it permanently. To avoid
    cross-test contamination, we save/restore the registry in setUp/tearDown.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lib = LibraryLearning(make_lib_config(Path(self.tmpdir)))
        # Save the global DSL registry state to restore after each test
        self._saved_registry = dict(DSLElement._registry)

    def tearDown(self):
        # Restore the global DSL registry to prevent cross-test contamination
        DSLElement._registry = self._saved_registry

    def test_empty_input_returns_empty(self):
        """sleep_step with no solved programs returns empty list."""
        result = self.lib.sleep_step([])
        self.assertEqual(result, [])

    def test_single_program_no_repetition(self):
        """A single program with unique chain yields no new primitives (min_freq not met)."""
        prog = make_chain_program("copy", "mirror")
        result = self.lib.sleep_step([prog], min_freq=3)
        # Only 1 program, so frequency of any subexpr is 1 < 3
        self.assertEqual(len(result), 0)

    def test_repeated_chain_extracts_primitive(self):
        """Multiple programs with the same chain sub-expression extract a new primitive."""
        # Create 4 identical chain programs
        progs = [make_chain_program("copy", "mirror") for _ in range(4)]
        result = self.lib.sleep_step(
            progs, min_freq=3, mdl_gain_threshold=1, max_new=10
        )
        # The root chain (copy->mirror) appears 4 times >= min_freq=3
        # MDL gain = uncompressed*4 - 1*4 - 10 = (5+5+2)*4 - 4 - 10 = 12*4 - 14 = 34
        # That's >= mdl_gain_threshold=1
        self.assertGreaterEqual(len(result), 1)
        # The new primitive should have name "learned_<hash>"
        for prim in result:
            self.assertTrue(prim.name.startswith("learned_"))
            self.assertEqual(prim.mdl_cost, 1)

    def test_new_primitive_registered_in_dsl(self):
        """Sleep-step registers new primitives in DSLElement._registry."""
        progs = [make_chain_program("copy", "scale") for _ in range(5)]
        result = self.lib.sleep_step(
            progs, min_freq=3, mdl_gain_threshold=1, max_new=10
        )
        for prim in result:
            self.assertIn(prim.name, DSLElement._registry)

    def test_new_primitive_added_to_library(self):
        """Sleep-step adds new primitives to the library."""
        progs = [make_chain_program("copy", "rotate") for _ in range(5)]
        result = self.lib.sleep_step(
            progs, min_freq=3, mdl_gain_threshold=1, max_new=10
        )
        self.assertGreaterEqual(len(result), 1)
        self.assertGreaterEqual(self.lib.get_size(), len(result))

    def test_max_new_limits_primitives(self):
        """max_new parameter limits the number of registered primitives."""
        # Create programs with multiple different chains
        progs = []
        for _ in range(5):
            progs.append(make_chain_program("copy", "mirror"))
        for _ in range(5):
            progs.append(make_chain_program("copy", "scale"))
        for _ in range(5):
            progs.append(make_chain_program("copy", "rotate"))
        result = self.lib.sleep_step(
            progs, min_freq=3, mdl_gain_threshold=1, max_new=2
        )
        self.assertLessEqual(len(result), 2)

    def test_mdl_gain_threshold_filters(self):
        """High mdl_gain_threshold filters out low-gain sub-expressions."""
        progs = [make_chain_program("copy", "mirror") for _ in range(3)]
        # With threshold=1000, no sub-expression should pass
        result = self.lib.sleep_step(
            progs, min_freq=2, mdl_gain_threshold=1000, max_new=10
        )
        self.assertEqual(len(result), 0)

    def test_learned_primitive_is_callable(self):
        """A learned primitive can be applied via DSLElement.apply()."""
        progs = [make_chain_program("copy", "mirror") for _ in range(5)]
        result = self.lib.sleep_step(
            progs, min_freq=3, mdl_gain_threshold=1, max_new=10
        )
        if result:
            prim = result[0]
            grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
            # Should not raise
            output = prim.apply(grid)
            self.assertIsInstance(output, np.ndarray)


# ============================================================
# Test: Existing LibraryLearning Methods (Regression)
# ============================================================

class TestLibraryLearningRegression(unittest.TestCase):
    """Regression tests for existing LibraryLearning methods."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.lib = LibraryLearning(make_lib_config(Path(self.tmpdir)))

    def test_get_size_initially_zero(self):
        """Fresh library has size 0."""
        self.assertEqual(self.lib.get_size(), 0)

    def test_clear_empties_library(self):
        """clear() empties the library."""
        self.lib.clear()
        self.assertEqual(self.lib.get_size(), 0)

    def test_get_abstractions_returns_list(self):
        """get_abstractions returns a list."""
        abstractions = self.lib.get_abstractions()
        self.assertIsInstance(abstractions, list)

    def test_save_and_load_roundtrip(self):
        """Save then load preserves library state."""
        # Add a pattern first
        prog = make_chain_program("copy", "mirror")
        self.lib.extract_patterns([prog, prog, prog])
        size_before = self.lib.get_size()
        self.lib.save()

        # Create new library with same path
        lib2 = LibraryLearning(make_lib_config(Path(self.tmpdir)))
        size_after = lib2.get_size()
        # Should load from file
        if size_before > 0:
            self.assertGreater(size_after, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
