"""DSL primitive correctness tests.

Tests each of the ~30 DSL primitives for apply() correctness using
known input -> expected output pairs. Also tests ProgramNode
composition (chain, additive, conditional) and fail-safe behavior.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.dsl_primitives import (
    DSLElement,
    ProgramNode,
    get_all_primitives,
    get_primitive_names,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def simple_grid() -> np.ndarray:
    """A simple 3x3 grid for testing."""
    return np.array([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
    ], dtype=np.int8)


@pytest.fixture
def uniform_grid() -> np.ndarray:
    """A 3x3 grid filled with a single color."""
    return np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ], dtype=np.int8)


@pytest.fixture
def two_color_grid() -> np.ndarray:
    """A grid with two distinct colors."""
    return np.array([
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 1],
    ], dtype=np.int8)


# ============================================================
# Registry Tests
# ============================================================

class TestDSLRegistry:
    """Tests for DSL primitive registry."""

    def test_all_primitives_registered(self):
        """All ~30 primitives should be registered."""
        names = get_primitive_names()
        assert len(names) >= 28, f"Expected >= 28 primitives, got {len(names)}"

    def test_get_all_primitives_returns_elements(self):
        """get_all_primitives should return DSLElement instances."""
        prims = get_all_primitives()
        assert all(isinstance(p, DSLElement) for p in prims)

    def test_expected_primitives_present(self):
        """Key primitives should be in the registry."""
        names = set(get_primitive_names())
        expected = {
            "resize", "fill-region", "color-swap", "mirror", "rotate",
            "scale", "map-color", "move", "copy", "gravity", "flood-fill",
            "tile", "crop", "overlay", "subtract", "union", "intersection",
            "draw-line", "dilate", "erode", "label-connected", "histogram",
            "find-objects", "symmetry-detect", "count", "extract-pattern",
            "complete-shape", "objects-complete", "boundary-detect", "skeleton",
        }
        missing = expected - names
        assert not missing, f"Missing primitives: {missing}"


# ============================================================
# Simple Primitive Tests
# ============================================================

class TestMirror:
    """Tests for mirror primitive."""

    def test_horizontal_mirror(self, simple_grid):
        """Horizontal mirror should flip left-right."""
        elem = DSLElement("mirror", {"axis": "horizontal"})
        result = elem.apply(simple_grid)
        expected = np.fliplr(simple_grid)
        np.testing.assert_array_equal(result, expected)

    def test_vertical_mirror(self, simple_grid):
        """Vertical mirror should flip up-down."""
        elem = DSLElement("mirror", {"axis": "vertical"})
        result = elem.apply(simple_grid)
        expected = np.flipud(simple_grid)
        np.testing.assert_array_equal(result, expected)

    def test_both_mirror(self, simple_grid):
        """Both-axis mirror should flip both ways."""
        elem = DSLElement("mirror", {"axis": "both"})
        result = elem.apply(simple_grid)
        expected = np.flipud(np.fliplr(simple_grid))
        np.testing.assert_array_equal(result, expected)


class TestRotate:
    """Tests for rotate primitive."""

    def test_rotate_90(self, simple_grid):
        """Rotate 90 degrees."""
        elem = DSLElement("rotate", {"angle": 90})
        result = elem.apply(simple_grid)
        expected = np.rot90(simple_grid, k=1)
        np.testing.assert_array_equal(result, expected)

    def test_rotate_180(self, simple_grid):
        """Rotate 180 degrees."""
        elem = DSLElement("rotate", {"angle": 180})
        result = elem.apply(simple_grid)
        expected = np.rot90(simple_grid, k=2)
        np.testing.assert_array_equal(result, expected)

    def test_rotate_360(self, simple_grid):
        """Rotate 360 degrees should be identity."""
        elem = DSLElement("rotate", {"angle": 360})
        result = elem.apply(simple_grid)
        np.testing.assert_array_equal(result, simple_grid)


class TestScale:
    """Tests for scale primitive."""

    def test_scale_2x(self, simple_grid):
        """Scale by factor 2 should double dimensions."""
        elem = DSLElement("scale", {"factor": 2})
        result = elem.apply(simple_grid)
        assert result.shape == (6, 6)
        # Top-left pixel should be repeated
        assert result[0, 0] == simple_grid[0, 0]
        assert result[1, 1] == simple_grid[0, 0]

    def test_scale_3x(self, simple_grid):
        """Scale by factor 3 should triple dimensions."""
        elem = DSLElement("scale", {"factor": 3})
        result = elem.apply(simple_grid)
        assert result.shape == (9, 9)


class TestColorSwap:
    """Tests for color-swap primitive."""

    def test_swap_colors(self, two_color_grid):
        """Swap color 0 and 1."""
        elem = DSLElement("color-swap", {"color_a": 0, "color_b": 1})
        result = elem.apply(two_color_grid)
        # 0s become 1s, 1s become 0s
        assert np.all(result[:1, :] == 1)  # Was 0
        assert np.all(result[1:, :] == 0)  # Was 1


class TestMapColor:
    """Tests for map-color primitive."""

    def test_map_single_color(self, simple_grid):
        """Map color 1 to 5."""
        elem = DSLElement("map-color", {"mapping": {1: 5}})
        result = elem.apply(simple_grid)
        assert result[0, 0] == 5
        assert result[1, 1] == 2  # Unchanged
        assert result[2, 2] == 3  # Unchanged

    def test_map_multiple_colors(self, simple_grid):
        """Map multiple colors."""
        elem = DSLElement("map-color", {"mapping": {1: 7, 2: 8, 3: 9}})
        result = elem.apply(simple_grid)
        assert result[0, 0] == 7
        assert result[1, 1] == 8
        assert result[2, 2] == 9


class TestFillRegion:
    """Tests for fill-region primitive."""

    def test_fill_all(self, simple_grid):
        """Fill entire grid with a color."""
        elem = DSLElement("fill-region", {"color": 5, "region": "all"})
        result = elem.apply(simple_grid)
        assert np.all(result == 5)

    def test_fill_border(self, simple_grid):
        """Fill only border pixels."""
        elem = DSLElement("fill-region", {"color": 9, "region": "border"})
        result = elem.apply(simple_grid)
        assert result[0, 0] == 9
        assert result[0, 2] == 9
        assert result[2, 0] == 9
        assert result[2, 2] == 9
        # Center should be unchanged
        assert result[1, 1] == 2


class TestMove:
    """Tests for move primitive."""

    def test_move_right(self):
        """Move all pixels right by 1."""
        grid = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        elem = DSLElement("move", {"dx": 1, "dy": 0})
        result = elem.apply(grid)
        assert result[1, 1] == 1
        assert result[1, 0] == 0  # Original position cleared

    def test_move_down(self):
        """Move all pixels down by 1."""
        grid = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        elem = DSLElement("move", {"dx": 0, "dy": 1})
        result = elem.apply(grid)
        assert result[1, 1] == 1
        assert result[0, 1] == 0


class TestCopy:
    """Tests for copy primitive."""

    def test_copy_preserves_original(self):
        """Copy should keep originals and add copies."""
        grid = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        elem = DSLElement("copy", {"dx": 1, "dy": 0})
        result = elem.apply(grid)
        assert result[0, 0] == 1  # Original preserved
        assert result[0, 1] == 1  # Copy added


class TestGravity:
    """Tests for gravity primitive."""

    def test_gravity_down(self):
        """Gravity down should move pixels to bottom."""
        grid = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        elem = DSLElement("gravity", {"direction": "down"})
        result = elem.apply(grid)
        assert result[2, 0] == 1
        assert result[0, 0] == 0

    def test_gravity_up(self):
        """Gravity up should move pixels to top."""
        grid = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ], dtype=np.int8)
        elem = DSLElement("gravity", {"direction": "up"})
        result = elem.apply(grid)
        assert result[0, 0] == 1
        assert result[2, 0] == 0


class TestCrop:
    """Tests for crop primitive."""

    def test_crop_subregion(self, simple_grid):
        """Crop a sub-region."""
        elem = DSLElement("crop", {"top": 0, "left": 0, "height": 2, "width": 2})
        result = elem.apply(simple_grid)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, simple_grid[:2, :2])


class TestTile:
    """Tests for tile primitive."""

    def test_tile_2x2(self, simple_grid):
        """Tile grid 2x2."""
        elem = DSLElement("tile", {"factor_h": 2, "factor_w": 2})
        result = elem.apply(simple_grid)
        assert result.shape == (6, 6)
        # Top-left block should equal original
        np.testing.assert_array_equal(result[:3, :3], simple_grid)


class TestResize:
    """Tests for resize primitive."""

    def test_resize_upscale(self, simple_grid):
        """Resize to larger dimensions."""
        elem = DSLElement("resize", {"height": 6, "width": 6})
        result = elem.apply(simple_grid)
        assert result.shape == (6, 6)

    def test_resize_downscale(self, simple_grid):
        """Resize to smaller dimensions."""
        elem = DSLElement("resize", {"height": 2, "width": 2})
        result = elem.apply(simple_grid)
        assert result.shape == (2, 2)


class TestDrawLine:
    """Tests for draw-line primitive."""

    def test_draw_horizontal_line(self):
        """Draw a horizontal line."""
        grid = np.zeros((3, 3), dtype=np.int8)
        elem = DSLElement("draw-line", {"x1": 0, "y1": 1, "x2": 2, "y2": 1, "color": 5})
        result = elem.apply(grid)
        assert result[1, 0] == 5
        assert result[1, 1] == 5
        assert result[1, 2] == 5

    def test_draw_diagonal_line(self):
        """Draw a diagonal line."""
        grid = np.zeros((3, 3), dtype=np.int8)
        elem = DSLElement("draw-line", {"x1": 0, "y1": 0, "x2": 2, "y2": 2, "color": 7})
        result = elem.apply(grid)
        assert result[0, 0] == 7
        assert result[1, 1] == 7
        assert result[2, 2] == 7


class TestSetOperations:
    """Tests for overlay, subtract, union, intersection."""

    def test_overlay(self):
        """Overlay grid_b on grid_a."""
        grid_a = np.array([[1, 0], [0, 0]], dtype=np.int8)
        grid_b = np.array([[0, 2], [0, 0]], dtype=np.int8)
        elem = DSLElement("overlay", {"grid_b": grid_b})
        result = elem.apply(grid_a)
        assert result[0, 0] == 1  # From a (b is 0)
        assert result[0, 1] == 2  # From b (non-zero overrides)

    def test_subtract(self):
        """Subtract grid_b from grid_a."""
        grid_a = np.array([[1, 2], [3, 4]], dtype=np.int8)
        grid_b = np.array([[1, 0], [0, 4]], dtype=np.int8)
        elem = DSLElement("subtract", {"grid_b": grid_b})
        result = elem.apply(grid_a)
        assert result[0, 0] == 0  # Subtracted
        assert result[0, 1] == 2  # Unchanged (b is 0)
        assert result[1, 1] == 0  # Subtracted

    def test_union(self):
        """Union of two grids."""
        grid_a = np.array([[1, 0], [0, 0]], dtype=np.int8)
        grid_b = np.array([[0, 2], [0, 0]], dtype=np.int8)
        elem = DSLElement("union", {"grid_b": grid_b})
        result = elem.apply(grid_a)
        assert result[0, 0] == 1
        assert result[0, 1] == 2

    def test_intersection(self):
        """Intersection of two grids."""
        grid_a = np.array([[1, 2], [3, 4]], dtype=np.int8)
        grid_b = np.array([[1, 0], [0, 4]], dtype=np.int8)
        elem = DSLElement("intersection", {"grid_b": grid_b})
        result = elem.apply(grid_a)
        assert result[0, 0] == 1  # Both non-zero
        assert result[0, 1] == 0  # b is 0
        assert result[1, 1] == 4  # Both non-zero


class TestMorphology:
    """Tests for dilate, erode, boundary-detect, skeleton."""

    def test_dilate(self):
        """Dilate should expand non-zero regions."""
        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.int8)
        elem = DSLElement("dilate", {"iterations": 1})
        result = elem.apply(grid)
        # Center should still be 1
        assert result[1, 2] == 1
        # Dilated region should have at least some non-zero neighbors
        non_zero_count = np.sum(result > 0)
        assert non_zero_count >= 1

    def test_erode(self):
        """Erode should shrink non-zero regions."""
        grid = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=np.int8)
        elem = DSLElement("erode", {"iterations": 1})
        result = elem.apply(grid)
        # Corners should be eroded
        assert result[0, 0] == 0
        # Center should survive
        assert result[1, 1] == 1

    def test_boundary_detect(self):
        """Boundary detection should find edges."""
        grid = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        elem = DSLElement("boundary-detect", {})
        result = elem.apply(grid)
        # Boundary pixels should be non-zero
        assert result[1, 1] == 1 or result[1, 2] == 1
        # Non-boundary interior (if any) should be 0


class TestLabelConnected:
    """Tests for label-connected and find-objects."""

    def test_label_connected_two_objects(self):
        """Label two separate connected components."""
        grid = np.array([
            [1, 0, 2],
            [1, 0, 2],
            [0, 0, 0],
        ], dtype=np.int8)
        elem = DSLElement("label-connected", {})
        result = elem.apply(grid)
        # Two distinct labels (1 and 2), background is 0
        unique_labels = set(np.unique(result).tolist())
        assert 0 in unique_labels
        assert len(unique_labels) >= 3  # 0 + at least 2 components

    def test_find_objects(self):
        """Find objects should label distinct objects."""
        grid = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
        ], dtype=np.int8)
        elem = DSLElement("find-objects", {})
        result = elem.apply(grid)
        # Two objects should be labeled differently
        assert result[0, 0] != 0
        assert result[2, 2] != 0
        assert result[0, 0] != result[2, 2]


class TestHistogram:
    """Tests for histogram primitive."""

    def test_histogram_counts(self):
        """Histogram should count color frequencies."""
        grid = np.array([
            [0, 1, 2],
            [1, 1, 0],
        ], dtype=np.int8)
        elem = DSLElement("histogram", {})
        result = elem.apply(grid)
        assert result.shape == (1, 10)
        assert result[0, 0] == 2  # Two 0s
        assert result[0, 1] == 3  # Three 1s
        assert result[0, 2] == 1  # One 2


class TestCount:
    """Tests for count primitive."""

    def test_count_objects(self):
        """Count should return number of connected components."""
        grid = np.array([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ], dtype=np.int8)
        elem = DSLElement("count", {})
        result = elem.apply(grid)
        assert result.shape == (1, 1)
        assert result[0, 0] == 4  # Four separate objects


class TestFloodFill:
    """Tests for flood-fill primitive."""

    def test_flood_fill_enclosed(self):
        """Flood fill should fill enclosed regions."""
        grid = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ], dtype=np.int8)
        elem = DSLElement("flood-fill", {"color": 5})
        result = elem.apply(grid)
        # Interior should be filled
        assert result[1, 1] == 5
        assert result[1, 2] == 5
        assert result[2, 1] == 5
        assert result[2, 2] == 5
        # Border should be unchanged
        assert result[0, 0] == 1


class TestExtractPattern:
    """Tests for extract-pattern primitive."""

    def test_extract_repeating_pattern(self):
        """Extract the smallest repeating tile."""
        tile = np.array([[1, 2]], dtype=np.int8)
        grid = np.tile(tile, (1, 3))  # [1, 2, 1, 2, 1, 2]
        elem = DSLElement("extract-pattern", {})
        result = elem.apply(grid)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, tile)


class TestSymmetryDetect:
    """Tests for symmetry-detect primitive."""

    def test_complete_horizontal_symmetry(self):
        """Symmetry detect should complete missing symmetric pixels."""
        grid = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        elem = DSLElement("symmetry-detect", {})
        result = elem.apply(grid)
        # Should complete the horizontal mirror
        assert result[0, 0] == 1
        assert result[0, 2] == 1  # Completed by symmetry


# ============================================================
# ProgramNode Tests
# ============================================================

class TestProgramNode:
    """Tests for ProgramNode composition."""

    def test_leaf_node_apply(self, simple_grid):
        """Leaf node should apply its element."""
        elem = DSLElement("mirror", {"axis": "horizontal"})
        node = ProgramNode(elem)
        result = node.apply(simple_grid)
        np.testing.assert_array_equal(result, np.fliplr(simple_grid))

    def test_chain_composition(self, simple_grid):
        """Chain composition should apply elements sequentially."""
        elem1 = DSLElement("mirror", {"axis": "horizontal"})
        elem2 = DSLElement("mirror", {"axis": "vertical"})
        node = ProgramNode(elem1)
        child = ProgramNode(elem2)
        node.children.append(child)
        node.combo_type = "chain"
        node.total_mdl = node.compute_mdl()

        result = node.apply(simple_grid)
        expected = np.flipud(np.fliplr(simple_grid))
        np.testing.assert_array_equal(result, expected)

    def test_additive_composition(self):
        """Additive composition should overlay results."""
        grid = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        elem1 = DSLElement("copy", {"dx": 1, "dy": 0})
        elem2 = DSLElement("copy", {"dx": 0, "dy": 1})
        node = ProgramNode(elem1)
        child = ProgramNode(elem2)
        node.children.append(child)
        node.combo_type = "additive"
        node.total_mdl = node.compute_mdl()

        result = node.apply(grid)
        # Original + right copy + down copy
        assert result[0, 0] == 1
        assert result[0, 1] == 1  # From right copy
        assert result[1, 0] == 1  # From down copy

    def test_conditional_composition(self, simple_grid):
        """Conditional composition should try elements until one changes grid."""
        # First element that changes grid wins
        elem1 = DSLElement("mirror", {"axis": "horizontal"})
        node = ProgramNode(elem1)
        node.combo_type = "conditional"
        node.total_mdl = node.compute_mdl()

        result = node.apply(simple_grid)
        # Mirror changes the grid, so it should be used
        assert not np.array_equal(result, simple_grid)

    def test_conditional_fallthrough(self):
        """Conditional should fall through if first element doesn't change grid."""
        grid = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        # 'copy' with dx=0,dy=0 doesn't change the grid
        elem1 = DSLElement("copy", {"dx": 0, "dy": 0})
        elem2 = DSLElement("mirror", {"axis": "horizontal"})
        node = ProgramNode(elem1)
        child = ProgramNode(elem2)
        node.children.append(child)
        node.combo_type = "conditional"
        node.total_mdl = node.compute_mdl()

        result = node.apply(grid)
        # copy doesn't change, so mirror should be used
        expected = np.fliplr(grid)
        np.testing.assert_array_equal(result, expected)

    def test_compute_mdl(self):
        """MDL should sum element costs plus composition overhead."""
        elem1 = DSLElement("mirror")  # mdl=5
        elem2 = DSLElement("rotate")  # mdl=5
        node = ProgramNode(elem1)
        child = ProgramNode(elem2)
        node.children.append(child)
        node.combo_type = "chain"
        mdl = node.compute_mdl()
        # 5 + 5 + 2 (composition overhead) = 12
        assert mdl == 12

    def test_flatten(self):
        """Flatten should return all elements in traversal order."""
        elem1 = DSLElement("mirror")
        elem2 = DSLElement("rotate")
        node = ProgramNode(elem1)
        child = ProgramNode(elem2)
        node.children.append(child)
        elements = node.flatten()
        assert len(elements) == 2
        assert elements[0].name == "mirror"
        assert elements[1].name == "rotate"


# ============================================================
# Fail-Safe Tests
# ============================================================

class TestFailSafe:
    """Tests for fail-safe behavior."""

    def test_unknown_primitive_returns_original(self, simple_grid):
        """Unknown primitive should return original grid (fail-safe)."""
        elem = DSLElement("nonexistent-primitive", {})
        result = elem.apply(simple_grid)
        np.testing.assert_array_equal(result, simple_grid)

    def test_invalid_params_returns_original(self, simple_grid):
        """Invalid parameters should return original grid (fail-safe)."""
        # overlay with grid_b=None should return original
        elem = DSLElement("overlay", {"grid_b": None})
        result = elem.apply(simple_grid)
        np.testing.assert_array_equal(result, simple_grid)

    def test_apply_does_not_modify_input(self, simple_grid):
        """apply should not modify the input grid."""
        original = simple_grid.copy()
        elem = DSLElement("mirror", {"axis": "horizontal"})
        _ = elem.apply(simple_grid)
        np.testing.assert_array_equal(simple_grid, original)


# ============================================================
# DSLElement Utility Tests
# ============================================================

class TestDSLElementUtils:
    """Tests for DSLElement utility methods."""

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization roundtrip should preserve element."""
        elem = DSLElement("mirror", {"axis": "vertical"})
        d = elem.to_dict()
        restored = DSLElement.from_dict(d)
        assert restored.name == elem.name
        assert restored.params == elem.params
        assert restored.mdl_cost == elem.mdl_cost

    def test_compose_returns_program_node(self):
        """compose should return a chain ProgramNode."""
        elem1 = DSLElement("mirror")
        elem2 = DSLElement("rotate")
        node = elem1.compose(elem2)
        assert isinstance(node, ProgramNode)
        assert node.combo_type == "chain"
        assert len(node.children) == 1

    def test_mdl_cost_categories(self):
        """MDL costs should follow simple/medium/complex categories."""
        simple = DSLElement("mirror")
        medium = DSLElement("gravity")
        assert simple.mdl_cost == 5
        assert medium.mdl_cost == 10

    def test_octonion_transform_desc(self):
        """Each element should have an octonion transform description."""
        elem = DSLElement("mirror")
        assert isinstance(elem.octonion_transform, str)
        assert len(elem.octonion_transform) > 0
