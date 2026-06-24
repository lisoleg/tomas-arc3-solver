"""ARC DSL primitives (~30 operations) and ProgramNode composition tree.

Each primitive implements ``apply(grid)`` with fail-safe behavior (returns
original grid on error), carries an MDL cost, and an octonion transform
description. ProgramNode supports chain, additive, and conditional composition.

TOMAS v2.2: Numba JIT-compiled kernels for hot-path operations.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import ndimage

# Numba-accelerated kernels (graceful fallback if numba unavailable)
try:
    from src.core.numba_kernels import (
        HAS_NUMBA,
        overlay_kernel,
        subtract_kernel,
        union_kernel,
        intersection_kernel,
        move_kernel,
        copy_with_offset_kernel,
        draw_line_kernel,
        gravity_kernel,
        resize_kernel,
        symmetry_kernel,
        color_swap_kernel,
        skeleton_kernel,
    )
except ImportError:
    HAS_NUMBA = False


# ============================================================
# DSL Element
# ============================================================

class DSLElement:
    """A single ARC transformation primitive.

    TOMAS v2.0: GATlab axiom interfaces — verify_compositionality() and
    verify_reversibility() prepare for formal GATlab verification.

    Attributes:
        name: Primitive name (e.g. 'mirror', 'rotate').
        params: Parameters dict (e.g. {'axis': 'horizontal'}).
        mdl_cost: Minimum Description Length cost.
        octonion_transform: Description of octonion-space transformation.
    """

    # Registry of primitive implementations
    _registry: dict[str, Callable[..., np.ndarray]] = {}

    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        """Initialize a DSL element.

        Args:
            name: Name of the primitive.
            params: Parameters for the primitive.
        """
        self.name = name
        self.params = params or {}
        self.mdl_cost = self._compute_default_mdl(name)
        self.octonion_transform = self._get_octonion_transform_desc(name)

    def _compute_default_mdl(self, name: str) -> int:
        """Compute default MDL cost based on primitive complexity.

        Args:
            name: Primitive name.

        Returns:
            MDL cost integer.
        """
        SIMPLE_PRIMITIVES = {
            "resize", "fill-region", "color-swap", "mirror", "rotate", "scale",
            "map-color", "move", "copy", "crop", "draw-line", "boundary-detect",
            "universal-color-map",
        }
        MEDIUM_PRIMITIVES = {
            "complete-shape", "objects-complete", "gravity", "flood-fill",
            "extract-pattern", "symmetry-detect", "tile", "overlay", "subtract",
            "union", "intersection", "skeleton", "dilate", "erode",
            "label-connected", "histogram", "find-objects", "count",
            "smart-flood-fill", "grid-diff-apply", "extract-and-recolor",
            "conditional-map", "nearest-neighbor-transform",
            "replace-marker-with-border", "gravity-stack",
            "draw-frame-around-objects", "fill-enclosed-regions",
            "sort-objects-by-size", "move-object-to-pos",
            "count-objects-mark", "propagate-color", "connect-dots",
        }
        if name in SIMPLE_PRIMITIVES:
            return 5
        if name in MEDIUM_PRIMITIVES:
            return 10
        return 15

    def _get_octonion_transform_desc(self, name: str) -> str:
        """Get a description of how this primitive transforms octonion space.

        Args:
            name: Primitive name.

        Returns:
            String description of octonion transformation.
        """
        descs = {
            "mirror": "negate e1_x (horizontal) or e2_y (vertical)",
            "rotate": "rotate (e1_x, e2_y) plane by angle",
            "scale": "scale e1_x, e2_y coordinates",
            "resize": "resample e1_x, e2_y grid",
            "color-swap": "permute e3_color phase values",
            "map-color": "remap e3_color phase values",
            "fill-region": "set e3_color for masked region",
            "move": "translate e1_x, e2_y by delta",
            "copy": "duplicate region preserving all components",
            "gravity": "shift e2_y towards maximum (gravity direction)",
            "flood-fill": "propagate e3_color through connected e4_component_id",
            "complete-shape": "infer missing pixels from e5_symmetry pattern",
            "rotate-symmetry-detect": "analyze e5_symmetry invariant",
        }
        return descs.get(name, f"transform {name}")

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply this primitive to a grid (fail-safe).

        Args:
            grid: Input grid as int8 ndarray.

        Returns:
            Transformed grid, or original grid on failure.
        """
        try:
            func = DSLElement._registry.get(self.name)
            if func is None:
                return grid.copy()
            result = func(grid, **self.params)
            return np.asarray(result, dtype=np.int8)
        except Exception:
            return grid.copy()

    def apply_to_hyperedge(self, edge: Any) -> Any:
        """Apply this primitive to an octonion hyperedge.

        Args:
            edge: OctonionHyperEdge instance.

        Returns:
            Transformed OctonionHyperEdge.
        """
        from src.core.octonion_hyperedge import OctonionHyperEdge

        grid = edge.decode_to_grid()
        transformed = self.apply(grid)
        return OctonionHyperEdge(transformed, frame_idx=edge.frame_idx)

    def verify_compositionality(self) -> bool:
        """GATlab axiom: verify this primitive is compositional.

        A compositional primitive satisfies:
        f(g(x)) = (f ∘ g)(x) for all valid inputs x.
        Returns True by default; override for primitives that need
        formal GATlab verification.

        Returns:
            True if compositionality holds.
        """
        return True

    def verify_reversibility(self) -> bool:
        """GATlab axiom: verify this primitive is reversible.

        A reversible primitive has an inverse f⁻¹ such that:
        f⁻¹(f(x)) = x for all valid inputs x.
        Returns True by default; override for primitives where
        reversibility is not guaranteed (e.g., crop, resize).

        Returns:
            True if reversibility holds.
        """
        # Common non-reversible primitives
        NON_REVERSIBLE = {"crop", "resize", "tile"}
        if self.name in NON_REVERSIBLE:
            return False
        return True

    def clone(self) -> DSLElement:
        """Fast clone — shallow copy, safe for ProgramNode tree operations.

        Returns:
            New DSLElement with same name and params.
        """
        return DSLElement(self.name, dict(self.params))

    def compose(self, other: DSLElement) -> ProgramNode:
        """Compose with another element into a chain ProgramNode.

        Args:
            other: The next DSL element to apply.

        Returns:
            A ProgramNode representing self ⊙ other.
        """
        node_self = ProgramNode(self)
        node_other = ProgramNode(other)
        node_self.children.append(node_other)
        node_self.combo_type = "chain"
        node_self.total_mdl = node_self.compute_mdl()
        return node_self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "params": self.params,
            "mdl_cost": self.mdl_cost,
            "octonion_transform": self.octonion_transform,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DSLElement:
        """Deserialize from dictionary.

        Args:
            d: Dictionary representation.

        Returns:
            DSLElement instance.
        """
        elem = cls(d["name"], d.get("params", {}))
        elem.mdl_cost = d.get("mdl_cost", elem.mdl_cost)
        elem.octonion_transform = d.get("octonion_transform", elem.octonion_transform)
        return elem

    def __repr__(self) -> str:
        return f"DSLElement({self.name}, params={self.params}, mdl={self.mdl_cost})"


# ============================================================
# Program Node
# ============================================================

class ProgramNode:
    """A node in a program composition tree.

    Attributes:
        element: The DSL element at this node.
        children: Child ProgramNodes.
        combo_type: 'chain' | 'additive' | 'conditional'.
        total_mdl: Cumulative MDL cost.
    """

    def __init__(
        self,
        element: DSLElement | None = None,
        children: list[ProgramNode] | None = None,
    ) -> None:
        """Initialize a program node.

        Args:
            element: DSL element for this node.
            children: List of child nodes.
        """
        self.element = element
        self.children = children or []
        self.combo_type: str = "leaf"
        self.total_mdl: int = 0
        self.total_mdl = self.compute_mdl()

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Recursively apply the program tree to a grid.

        For chain: apply self, then apply children in sequence.
        For additive: apply self and all children, combine by overlay.
        For conditional: apply self if condition met, else apply children.

        Args:
            grid: Input grid.

        Returns:
            Transformed grid.
        """
        if self.element is None and not self.children:
            return grid.copy()

        if self.combo_type == "chain":
            result = grid.copy()
            if self.element is not None:
                result = self.element.apply(result)
            for child in self.children:
                result = child.apply(result)
            return result
        elif self.combo_type == "additive":
            results = []
            if self.element is not None:
                results.append(self.element.apply(grid.copy()))
            for child in self.children:
                results.append(child.apply(grid.copy()))
            if not results:
                return grid.copy()
            # Combine by taking non-zero pixels from each result
            combined = results[0]
            for r in results[1:]:
                mask = r != 0
                combined[mask] = r[mask]
            return combined
        elif self.combo_type == "conditional":
            # Conditional: if element succeeds (changes grid), use it;
            # otherwise fall through to children
            if self.element is not None:
                result = self.element.apply(grid.copy())
                if not np.array_equal(result, grid):
                    return result
            for child in self.children:
                result = child.apply(grid.copy())
                if not np.array_equal(result, grid):
                    return result
            return grid.copy()
        else:
            # Leaf node
            if self.element is not None:
                return self.element.apply(grid)
            return grid.copy()

    def compute_mdl(self) -> int:
        """Compute total MDL cost of the program tree.

        Returns:
            Total MDL cost.
        """
        mdl = 0
        if self.element is not None:
            mdl += self.element.mdl_cost
        for child in self.children:
            mdl += child.compute_mdl()
        # Add composition overhead
        if self.combo_type in ("chain", "additive", "conditional"):
            mdl += 2
        self.total_mdl = mdl
        return mdl

    def flatten(self) -> list[DSLElement]:
        """Flatten the program tree into a list of DSL elements.

        Returns:
            List of DSLElement in traversal order.
        """
        elements: list[DSLElement] = []
        if self.element is not None:
            elements.append(self.element)
        for child in self.children:
            elements.extend(child.flatten())
        return elements

    def clone(self) -> ProgramNode:
        """Fast clone — recursively copies element and children.

        Much faster than copy.deepcopy for program trees (~3x speedup).

        Returns:
            Deep-cloned ProgramNode.
        """
        new_elem = self.element.clone() if self.element is not None else None
        new_node = ProgramNode(new_elem)
        new_node.children = [c.clone() for c in self.children]
        new_node.combo_type = self.combo_type
        new_node.total_mdl = self.total_mdl
        return new_node

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "element": self.element.to_dict() if self.element else None,
            "children": [c.to_dict() for c in self.children],
            "combo_type": self.combo_type,
            "total_mdl": self.total_mdl,
        }

    def __repr__(self) -> str:
        return (
            f"ProgramNode(element={self.element}, children={len(self.children)}, "
            f"combo={self.combo_type}, mdl={self.total_mdl})"
        )


# ============================================================
# Primitive Implementations
# ============================================================

def _resize(grid: np.ndarray, height: int = 0, width: int = 0, **_: Any) -> np.ndarray:
    """Resize grid to specified dimensions — JIT kernel for small grids."""
    h, w = grid.shape
    new_h = height if height > 0 else h
    new_w = width if width > 0 else w
    if HAS_NUMBA and h * w <= 10000:
        return resize_kernel(grid, new_h, new_w)
    # Vectorized fallback for large grids
    row_idx = np.clip((np.arange(new_h) * h / new_h).astype(np.int32), 0, h - 1)
    col_idx = np.clip((np.arange(new_w) * w / new_w).astype(np.int32), 0, w - 1)
    return grid[np.ix_(row_idx, col_idx)]


def _fill_region(grid: np.ndarray, color: int = 1, region: str = "all", **_: Any) -> np.ndarray:
    """Fill a region with a color."""
    result = grid.copy()
    if region == "all":
        result[:] = color
    elif region == "border":
        result[0, :] = color
        result[-1, :] = color
        result[:, 0] = color
        result[:, -1] = color
    elif region == "interior":
        result[1:-1, 1:-1] = color
    return result


def _color_swap(grid: np.ndarray, color_a: int = 0, color_b: int = 1, **_: Any) -> np.ndarray:
    """Swap two colors in the grid — JIT kernel."""
    if HAS_NUMBA:
        return color_swap_kernel(grid, color_a, color_b)
    result = grid.copy()
    mask_a = result == color_a
    mask_b = result == color_b
    result[mask_a] = color_b
    result[mask_b] = color_a
    return result


def _multi_swap(grid: np.ndarray, swap_pairs: list[list[int]] | None = None, **_: Any) -> np.ndarray:
    """Swap multiple color pairs at once.
    
    Args:
        swap_pairs: List of [color_a, color_b] pairs to swap.
                    If None, tries to infer from grid.
    """
    result = grid.copy()
    if swap_pairs is None:
        # Try to infer: find all non-zero colors and pair them
        # This is a heuristic: pair (1,5), (2,6), (3,7), (4,8), (9,0)
        non_zero = sorted(set(grid.flatten()) - {0})
        swap_pairs = []
        for c in non_zero:
            pair = (c + 4) % 10
            if pair in non_zero and [pair, c] not in swap_pairs:
                swap_pairs.append([c, pair])
    
    if swap_pairs:
        for a, b in swap_pairs:
            mask_a = result == a
            mask_b = result == b
            result[mask_a] = b
            result[mask_b] = a
    
    return result


def _mirror(grid: np.ndarray, axis: str = "horizontal", **_: Any) -> np.ndarray:
    """Mirror the grid along an axis."""
    if axis == "horizontal":
        return np.fliplr(grid).copy()
    elif axis == "vertical":
        return np.flipud(grid).copy()
    else:
        return np.flipud(np.fliplr(grid)).copy()


def _rotate(grid: np.ndarray, angle: int = 90, **_: Any) -> np.ndarray:
    """Rotate the grid by a given angle (90, 180, 270)."""
    k = (angle // 90) % 4
    return np.rot90(grid, k=k).copy()


def _scale(grid: np.ndarray, factor: int = 2, **_: Any) -> np.ndarray:
    """Scale the grid by an integer factor."""
    return np.kron(grid, np.ones((factor, factor), dtype=np.int8))


def _map_color(grid: np.ndarray, mapping: dict[int, int] | None = None, **_: Any) -> np.ndarray:
    """Map colors according to a mapping dictionary."""
    result = grid.copy()
    if mapping is None:
        mapping = {}
    for old_color, new_color in mapping.items():
        result[grid == old_color] = new_color
    return result


def _complete_shape(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Complete partially-drawn shapes by symmetry."""
    result = grid.copy()
    h, w = grid.shape
    # Try to complete horizontal symmetry
    left_half = result[:, :w // 2]
    right_mirror = np.fliplr(left_half)
    target_start = w - w // 2
    result[:, target_start:] = right_mirror[:, :w - target_start]
    return result


def _objects_complete(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Complete missing objects based on detected patterns."""
    result = grid.copy()
    binary = (grid > 0).astype(np.int32)
    labeled, num = ndimage.label(binary)
    if num >= 2:
        # Find bounding boxes of objects
        objs = ndimage.find_objects(labeled)
        if len(objs) >= 2:
            # Duplicate first object pattern to fill gaps
            first_slice = objs[0]
            obj_pattern = grid[first_slice]
            for sl in objs[1:]:
                obj_h = sl[0].stop - sl[0].start
                obj_w = sl[1].stop - sl[1].start
                if obj_h == obj_pattern.shape[0] and obj_w == obj_pattern.shape[1]:
                    result[sl] = obj_pattern
    return result


def _move(grid: np.ndarray, dx: int = 0, dy: int = 0, **_: Any) -> np.ndarray:
    """Move all non-zero pixels by (dx, dy) — JIT kernel."""
    if HAS_NUMBA:
        return move_kernel(grid, dx, dy)
    result = np.zeros_like(grid)
    h, w = grid.shape
    src_r0, src_r1 = max(0, -dy), min(h, h - dy)
    src_c0, src_c1 = max(0, -dx), min(w, w - dx)
    dst_r0, dst_c0 = max(0, dy), max(0, dx)
    if src_r1 > src_r0 and src_c1 > src_c0:
        result[dst_r0:dst_r0 + (src_r1 - src_r0),
               dst_c0:dst_c0 + (src_c1 - src_c0)] = grid[src_r0:src_r1, src_c0:src_c1]
    return result


def _copy(grid: np.ndarray, dx: int = 0, dy: int = 0, **_: Any) -> np.ndarray:
    """Copy all non-zero pixels by offset, keeping originals — JIT kernel."""
    if HAS_NUMBA:
        return copy_with_offset_kernel(grid, dx, dy)
    result = grid.copy()
    h, w = grid.shape
    mask = grid != 0
    if not np.any(mask):
        return result
    yy, xx = np.where(mask)
    ny, nx = yy + dy, xx + dx
    valid = (0 <= ny) & (ny < h) & (0 <= nx) & (nx < w)
    if np.any(valid):
        result[ny[valid], nx[valid]] = grid[yy[valid], xx[valid]]
    return result


def _gravity(grid: np.ndarray, direction: str = "down", **_: Any) -> np.ndarray:
    """Apply gravity to non-zero pixels — JIT kernel."""
    dir_map = {"down": 0, "up": 1, "left": 2, "right": 3}
    dc = dir_map.get(direction, 0)
    if HAS_NUMBA:
        return gravity_kernel(grid, dc)
    # Pure-numpy fallback
    result = np.zeros_like(grid)
    h, w = grid.shape
    if direction in ("down", "up"):
        for j in range(w):
            col = grid[:, j]
            nz = col[col != 0]
            if len(nz):
                dst = slice(-len(nz), None) if direction == "down" else slice(0, len(nz))
                result[dst, j] = nz
    elif direction in ("left", "right"):
        for i in range(h):
            row = grid[i, :]
            nz = row[row != 0]
            if len(nz):
                dst = slice(0, len(nz)) if direction == "left" else slice(-len(nz), None)
                result[i, dst] = nz
    return result


def _flood_fill(grid: np.ndarray, color: int = 1, **_: Any) -> np.ndarray:
    """Flood fill enclosed regions with a color."""
    result = grid.copy()
    binary = (grid == 0).astype(np.int32)
    # Label background regions
    labeled, num = ndimage.label(binary)
    if num == 0:
        return result
    # Find the outer background (touches border)
    h, w = grid.shape
    outer_labels = set()
    outer_labels.update(labeled[0, :].tolist())
    outer_labels.update(labeled[-1, :].tolist())
    outer_labels.update(labeled[:, 0].tolist())
    outer_labels.update(labeled[:, -1].tolist())
    outer_labels.discard(0)
    # Fill inner regions
    for label in range(1, num + 1):
        if label not in outer_labels:
            result[labeled == label] = color
    return result


def _extract_pattern(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Extract the repeating pattern from the grid — vectorized via np.tile."""
    h, w = grid.shape
    for tile_h in range(1, h + 1):
        if h % tile_h != 0:
            continue
        for tile_w in range(1, w + 1):
            if w % tile_w != 0:
                continue
            tile = grid[:tile_h, :tile_w]
            # Use np.tile for vectorized full-grid comparison
            if np.array_equal(grid, np.tile(tile, (h // tile_h, w // tile_w))):
                if tile_h < h or tile_w < w:
                    return tile.copy()
    return grid.copy()


def _symmetry_detect(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Detect and complete symmetry in the grid — JIT kernel."""
    if HAS_NUMBA:
        return symmetry_kernel(grid)
    # Vectorized fallback
    result = grid.copy()
    h_flipped = np.fliplr(result)
    h_mask = (result == 0) & (h_flipped != 0)
    result[h_mask] = h_flipped[h_mask]
    v_flipped = np.flipud(result)
    v_mask = (result == 0) & (v_flipped != 0)
    result[v_mask] = v_flipped[v_mask]
    return result


def _count(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Count objects and return count as a grid representation."""
    binary = (grid > 0).astype(np.int32)
    _, num = ndimage.label(binary)
    # Return count as a 1xN grid
    result = np.array([[num]], dtype=np.int8)
    return result


def _tile(grid: np.ndarray, factor_h: int = 2, factor_w: int = 2, **_: Any) -> np.ndarray:
    """Tile the grid by repeating it."""
    return np.tile(grid, (factor_h, factor_w)).astype(np.int8)


def _crop(grid: np.ndarray, top: int = 0, left: int = 0, height: int = 0, width: int = 0, **_: Any) -> np.ndarray:
    """Crop a region from the grid."""
    h, w = grid.shape
    crop_h = height if height > 0 else h - top
    crop_w = width if width > 0 else w - left
    top = max(0, min(top, h - 1))
    left = max(0, min(left, w - 1))
    crop_h = max(1, min(crop_h, h - top))
    crop_w = max(1, min(crop_w, w - left))
    return grid[top:top + crop_h, left:left + crop_w].copy()


def _overlay(grid_a: np.ndarray, grid_b: np.ndarray | None = None, **_: Any) -> np.ndarray:
    """Overlay grid_b on grid_a (non-zero pixels from b override a) — JIT kernel."""
    if grid_b is None:
        return grid_a.copy()
    if HAS_NUMBA:
        return overlay_kernel(grid_a.copy(), grid_b)
    result = grid_a.copy()
    mask = grid_b != 0
    result[mask] = grid_b[mask]
    return result


def _subtract(grid_a: np.ndarray, grid_b: np.ndarray | None = None, **_: Any) -> np.ndarray:
    """Subtract grid_b from grid_a (remove overlapping pixels) — JIT kernel."""
    if grid_b is None:
        return grid_a.copy()
    if HAS_NUMBA:
        return subtract_kernel(grid_a.copy(), grid_b)
    result = grid_a.copy()
    mask = grid_b != 0
    result[mask] = 0
    return result


def _union(grid_a: np.ndarray, grid_b: np.ndarray | None = None, **_: Any) -> np.ndarray:
    """Union of two grids (non-zero pixels from either) — JIT kernel."""
    if grid_b is None:
        return grid_a.copy()
    if HAS_NUMBA:
        return union_kernel(grid_a.copy(), grid_b)
    result = grid_a.copy()
    mask = (grid_b != 0) & (grid_a == 0)
    result[mask] = grid_b[mask]
    return result


def _intersection(grid_a: np.ndarray, grid_b: np.ndarray | None = None, **_: Any) -> np.ndarray:
    """Intersection of two grids (pixels non-zero in both) — JIT kernel."""
    if grid_b is None:
        return grid_a.copy()
    if HAS_NUMBA:
        return intersection_kernel(grid_a.copy(), grid_b)
    result = np.zeros_like(grid_a)
    mask = (grid_a != 0) & (grid_b != 0)
    result[mask] = grid_a[mask]
    return result


def _boundary_detect(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Detect boundaries of objects (edge pixels)."""
    binary = (grid > 0).astype(np.int32)
    eroded = ndimage.binary_erosion(binary)
    boundary = binary & ~eroded
    result = np.zeros_like(grid)
    result[boundary] = grid[boundary]
    return result


def _skeleton(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Compute skeleton using numba kernel or ndimage morphological thinning."""
    if HAS_NUMBA:
        return skeleton_kernel(grid)
    # scipy.ndimage fallback
    binary = (grid > 0).astype(np.uint8)
    try:
        skeleton = ndimage.skeletonize(binary)
    except AttributeError:
        from scipy.ndimage import binary_erosion, binary_dilation
        skeleton = binary & ~binary_dilation(binary_erosion(binary))
    result = np.zeros_like(grid)
    result[skeleton] = grid[skeleton]
    return result


def _dilate(grid: np.ndarray, iterations: int = 1, **_: Any) -> np.ndarray:
    """Dilate non-zero regions."""
    binary = (grid > 0).astype(np.int32)
    dilated = ndimage.binary_dilation(binary, iterations=iterations)
    result = np.zeros_like(grid)
    result[dilated] = 1
    # Preserve original colors
    result[binary] = grid[binary]
    return result


def _erode(grid: np.ndarray, iterations: int = 1, **_: Any) -> np.ndarray:
    """Erode non-zero regions."""
    binary = (grid > 0).astype(np.int32)
    eroded = ndimage.binary_erosion(binary, iterations=iterations)
    result = np.zeros_like(grid)
    result[eroded] = grid[eroded]
    return result


def _label_connected(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Label connected components with sequential IDs."""
    binary = (grid > 0).astype(np.int32)
    labeled, num = ndimage.label(binary)
    return labeled.astype(np.int8)


def _histogram(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Compute color histogram as a 1D grid."""
    counts = np.bincount(grid.flatten(), minlength=10)
    return counts.reshape(1, -1).astype(np.int8)


def _find_objects(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Find and label distinct objects."""
    binary = (grid > 0).astype(np.int32)
    labeled, _ = ndimage.label(binary)
    return labeled.astype(np.int8)


def _draw_line(grid: np.ndarray, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0,
               color: int = 1, **_: Any) -> np.ndarray:
    """Draw a Bresenham line — JIT kernel (~20-50x speedup)."""
    if HAS_NUMBA:
        return draw_line_kernel(grid, x1, y1, x2, y2, color)
    # Pure-Python fallback
    result = grid.copy()
    h, w = result.shape
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steps = max(dx, dy, 1)
    x_inc = (x2 - x1) / steps
    y_inc = (y2 - y1) / steps
    x, y = float(x1), float(y1)
    for _ in range(steps + 1):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            result[yi, xi] = color
        x += x_inc
        y += y_inc
    return result


def _shift_object(grid: np.ndarray, color: int = 1, dx: int = 0, dy: int = 0, **_: Any) -> np.ndarray:
    """Shift an object (connected component of given color) by (dx, dy)."""
    result = grid.copy()
    binary = (grid == color).astype(np.int8)
    if not np.any(binary):
        return result
    # Find bounding box of the object
    rows = np.where(np.any(binary, axis=1))[0]
    cols = np.where(np.any(binary, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return result
    min_r, max_r = rows[0], rows[-1]
    min_c, max_c = cols[0], cols[-1]
    # Extract object
    obj = result[min_r:max_r+1, min_c:max_c+1].copy()
    # Clear original position
    result[min_r:max_r+1, min_c:max_c+1][binary[min_r:max_r+1, min_c:max_c+1] == 1] = 0
    # Place at new position
    new_r = min_r + dy
    new_c = min_c + dx
    if 0 <= new_r and 0 <= new_c:
        end_r = min(new_r + obj.shape[0], grid.shape[0])
        end_c = min(new_c + obj.shape[1], grid.shape[1])
        if end_r > new_r and end_c > new_c:
            result[new_r:end_r, new_c:end_c] = obj[:end_r-new_r, :end_c-new_c]
    return result


def _complete_pattern(grid: np.ndarray, rotation: int = 0, strategy: str = "auto", **_: Any) -> np.ndarray:
    """Detect periodic pattern in input and complete it to fill the grid.
    
    Improved v3.1: Hybrid strategy (sequence tiling + bounding box tiling).
    
    Args:
        grid: Input grid.
        rotation: Rotation offset for sequence tiling.
        strategy: 'sequence', 'bounding_box', or 'auto' (try both).
    """
    h, w = grid.shape
    
    if strategy == "auto":
        # Try sequence tiling first (if input has a clear color sequence)
        non_zero = grid[grid != 0]
        unique_colors = len(set(non_zero.flatten()))
        
        # Heuristic: if the input has few unique colors and they appear in a sequence,
        # use sequence tiling
        if unique_colors <= 10 and _has_clear_sequence(grid):
            return _complete_pattern_sequence(grid, rotation)
        else:
            return _complete_pattern_bbox(grid)
    elif strategy == "sequence":
        return _complete_pattern_sequence(grid, rotation)
    else:  # bounding_box
        return _complete_pattern_bbox(grid)


def _has_clear_sequence(grid: np.ndarray) -> bool:
    """Check if the grid has a clear color sequence (non-zero colors in a row)."""
    non_zero = grid[grid != 0]
    # Check if non-zero colors form a sequence (not scattered)
    if len(non_zero) < 2:
        return False
    
    # Check if non-zero colors are concentrated in a small region
    rows = np.where(np.any(grid != 0, axis=1))[0]
    cols = np.where(np.any(grid != 0, axis=0))[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return False
    
    # If the non-zero region is small compared to grid size, it's likely a seed pattern
    region_size = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
    grid_size = grid.shape[0] * grid.shape[1]
    
    return region_size < grid_size * 0.5


def _complete_pattern_sequence(grid: np.ndarray, rotation: int = 0) -> np.ndarray:
    """Complete pattern by extracting and tiling color sequence."""
    h, w = grid.shape
    result = np.zeros_like(grid)
    
    # Extract color sequence (non-zero colors in row-major order, first occurrence)
    seq = []
    seen = set()
    for r in range(h):
        for c in range(w):
            color = grid[r, c]
            if color != 0 and color not in seen:
                seq.append(color)
                seen.add(color)
    
    if not seq:
        return result
    
    # Apply rotation
    seq_len = len(seq)
    if seq_len > 0 and rotation > 0:
        rotation = rotation % seq_len
        seq = seq[rotation:] + seq[:rotation]
    
    # Tile sequence to fill grid
    idx = 0
    for r in range(h):
        for c in range(w):
            result[r, c] = seq[idx % seq_len]
            idx += 1
    
    return result


def _complete_pattern_bbox(grid: np.ndarray) -> np.ndarray:
    """Complete pattern by tiling the bounding box of non-zero regions."""
    result = grid.copy()
    h, w = grid.shape
    
    non_zero = grid != 0
    if not np.any(non_zero):
        return result
    
    # Find bounding box of non-zero regions
    rows = np.where(np.any(non_zero, axis=1))[0]
    cols = np.where(np.any(non_zero, axis=0))[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return result
    
    min_r, max_r = rows[0], rows[-1]
    min_c, max_c = cols[0], cols[-1]
    
    # Extract pattern (bounding box of non-zero)
    pattern = grid[min_r:max_r+1, min_c:max_c+1].copy()
    ph, pw = pattern.shape
    
    # Tile the pattern across the grid
    for r in range(0, h, ph):
        for c in range(0, w, pw):
            end_r = min(r + ph, h)
            end_c = min(c + pw, w)
            result[r:end_r, c:end_c] = pattern[:end_r-r, :end_c-c]
    
    return result


def _map_by_function(grid: np.ndarray, func_type: str = "add", value: int = 1, modulo: int = 10, **_: Any) -> np.ndarray:
    """Map colors by a function: new = f(old).
    
    Args:
        func_type: 'add', 'sub', 'mul', 'mod', 'inv_mod'
        value: parameter for the function
        modulo: modulo base (usualy 10 for ARC)
    """
    result = grid.copy()
    mask = grid != 0  # Only map non-background colors
    
    if func_type == "add":
        result[mask] = (result[mask] + value) % modulo
    elif func_type == "sub":
        result[mask] = (result[mask] - value) % modulo
    elif func_type == "mul":
        result[mask] = (result[mask] * value) % modulo
    elif func_type == "mod":
        result[mask] = result[mask] % value
    elif func_type == "inv_mod":
        # Map X to Y such that X + Y = modulo - 1 (swap pairs)
        result[mask] = (modulo - 1 - result[mask]) % modulo
    
    return result


def _extract_largest_object(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Extract the largest object (by pixel count) from the grid.

    Returns a grid with only the largest object (other pixels set to 0).
    """
    result = np.zeros_like(grid)
    binary = (grid > 0).astype(np.int32)
    labeled, num = ndimage.label(binary)

    if num == 0:
        return result

    # Find the largest object (by pixel count)
    max_size = 0
    max_label = 0
    for label in range(1, num + 1):
        size = np.sum(labeled == label)
        if size > max_size:
            max_size = size
            max_label = label

    # Extract the largest object
    result[labeled == max_label] = grid[labeled == max_label]

    return result


def _extract_smallest_object(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Extract the smallest object (by pixel count) from the grid.

    Returns a grid with only the smallest object (other pixels set to 0).
    """
    result = np.zeros_like(grid)
    binary = (grid > 0).astype(np.int32)
    labeled, num = ndimage.label(binary)

    if num == 0:
        return result

    # Find the smallest object (by pixel count, excluding size 0)
    min_size = float('inf')
    min_label = 0
    for label in range(1, num + 1):
        size = np.sum(labeled == label)
        if size > 0 and size < min_size:
            min_size = size
            min_label = label

    if min_label == 0:
        return result

    # Extract the smallest object
    result[labeled == min_label] = grid[labeled == min_label]

    return result


def _invert_colors(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Invert colors in the grid (0↔1, 2↔3, ...).
    
    Args:
        grid: Input grid.
        
    Returns:
        Grid with colors inverted (color i → color i^1).
    """
    result = grid.copy()
    # XOR with 1: 0↔1, 2↔3, 4↔5, ...
    result = result ^ 1
    return result


def _fill_connected(grid: np.ndarray, color: int = 1, **_: Any) -> np.ndarray:
    """Fill connected regions of a specific color.
    
    Args:
        grid: Input grid.
        color: Color to fill (default: 1).
        
    Returns:
        Grid with connected regions of specified color filled.
    """
    from scipy import ndimage
    
    result = grid.copy()
    
    # Create binary mask for the specified color
    mask = (grid == color)
    
    if not np.any(mask):
        return result
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return result
    
    # Fill each connected component with the same color
    for i in range(1, num_features + 1):
        component_mask = (labeled == i)
        # Keep original color for the component
        result[component_mask] = color
    
    return result


def _recolor_by_cc(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Recolor each connected component with a different color.
    
    Args:
        grid: Input grid.
        
    Returns:
        Grid with each connected component recolored.
    """
    from scipy import ndimage
    
    result = np.zeros_like(grid)
    
    # Create binary mask for non-zero pixels
    mask = (grid != 0)
    
    if not np.any(mask):
        return result
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    # Recolor each component with a different color
    for i in range(1, num_features + 1):
        result[labeled == i] = i
    
    return result


def _tile_repeat(grid: np.ndarray, repeats: int = 2, **_: Any) -> np.ndarray:
    """Repeat the grid pattern horizontally and vertically.
    
    Handles both 2D (H,W) and 3D (1,H,W) inputs.
    
    Args:
        grid: Input grid.
        repeats: Number of times to repeat (default: 2).
        
    Returns:
        Grid with pattern repeated.
    """
    # Handle 3D input (batch, H, W) → treat as (H, W)
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    
    h, w = g.shape
    result = np.zeros((h * repeats, w * repeats), dtype=grid.dtype)
    
    for r in range(repeats):
        for c in range(repeats):
            result[r*h:(r+1)*h, c*w:(c+1)*w] = g
    
    if is_3d:
        result = result[np.newaxis, :, :]  # (1, H*repeats, W*repeats)
    
    return result




def _select_obj_by(grid: np.ndarray, prop: str = 'size', **_: Any) -> np.ndarray:
    from scipy import ndimage
    result = np.zeros_like(grid)
    binary = (grid != 0).astype(np.int8)
    labeled, num = ndimage.label(binary)
    if num == 0:
        return result
    best_label = 0
    if prop == 'size':
        max_size = 0
        for i in range(1, num+1):
            s = np.sum(labeled == i)
            if s > max_size:
                max_size = s
                best_label = i
    elif prop == 'color':
        best_count = 0
        for i in range(1, num+1):
            pixels = grid[labeled == i]
            if len(pixels) == 0: continue
            vals, cnts = np.unique(pixels, return_counts=True)
            mc = np.max(cnts)
            if mc > best_count:
                best_count = mc
                best_label = i
    else:  # leftmost/rightmost/topmost/bottommost
        best_pos = None
        for i in range(1, num+1):
            rows, cols = np.where(labeled == i)
            if len(rows) == 0: continue
            if prop == 'leftmost': pos = np.min(cols)
            elif prop == 'rightmost': pos = np.max(cols)
            elif prop == 'topmost': pos = np.min(rows)
            else: pos = np.max(rows)
            if best_pos is None:
                best_pos = pos
                best_label = i
            else:
                if (prop in ['leftmost','topmost'] and pos < best_pos) or                    (prop in ['rightmost','bottommost'] and pos > best_pos):
                    best_pos = pos
                    best_label = i
    if best_label > 0:
        result[labeled == best_label] = grid[labeled == best_label]
    return result


def _sort_obj_by(grid: np.ndarray, prop: str = 'size', **_: Any) -> np.ndarray:
    from scipy import ndimage
    binary = (grid != 0).astype(np.int8)
    labeled, num = ndimage.label(binary)
    if num == 0:
        return grid.copy()
    objs = []
    for i in range(1, num+1):
        rows, cols = np.where(labeled == i)
        if len(rows) == 0: continue
        pixels = grid[labeled == i]
        if prop == 'size': val = len(pixels)
        elif prop == 'color':
            vals, cnts = np.unique(pixels, return_counts=True)
            val = vals[0] if len(vals) > 0 else 0
        elif prop == 'row': val = np.min(rows)
        else: val = np.min(cols)
        objs.append({'label': i, 'val': val, 'mask': labeled == i})
    objs.sort(key=lambda x: x['val'])
    result = np.zeros_like(grid)
    # Place sorted objects at original positions (simplified)
    for obj in objs:
        result[obj['mask']] = grid[obj['mask']]
    return result


# ============================================================
# New primitives v2.4.7
# ============================================================

def _scale_pattern(grid: np.ndarray, factor: int = 3, **_: Any) -> np.ndarray:
    """Scale grid by factor, fill each block with pattern if source cell non-zero.
    
    Needed for tasks like 007bbfb7:
    - Input (H,W): each cell is 0 or color
    - Output (H*factor, W*factor): each input cell expands to factor×factor block
      - If input[i,j] == 0 → block of 0s
      - Else → block filled with the original input pattern
    
    Args:
        grid: Input grid (2D or 3D).
        factor: Scale factor (default: 3).
    
    Returns:
        Scaled grid.
    """
    # Handle 3D
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    
    h, w = g.shape
    out_h, out_w = h * factor, w * factor
    result = np.zeros((out_h, out_w), dtype=grid.dtype)
    
    for i in range(h):
        for j in range(w):
            r0, r1 = i * factor, (i+1) * factor
            c0, c1 = j * factor, (j+1) * factor
            if g[i,j] != 0:
                # Fill block with the original pattern (clipped to block size)
                for pi in range(factor):
                    for pj in range(factor):
                        si, sj = pi % h, pj % w
                        result[r0+pi, c0+pj] = g[si, sj]
    
    if is_3d:
        result = result[np.newaxis, :, :]
    
    return result


# ============================================================
# Register all primitives
# ============================================================


# ======== v2.4.8 New Primitives ========

def _crop_to_obj(grid: np.ndarray, color: int = 0, **_: Any) -> np.ndarray:
    "Crop grid to bounding box of a specific color object."
    g = grid[0] if grid.ndim == 3 else grid
    mask = (g != 0) if color == 0 else (g == color)
    if not np.any(mask):
        return grid
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    result = g[r0:r1+1, c0:c1+1]
    return result[np.newaxis, :, :] if grid.ndim == 3 else result


def _crop_to_largest_cc(grid: np.ndarray, bg_color: int = 0, **_: Any) -> np.ndarray:
    """Crop to the bounding box of the largest connected component (color-independent).
    
    Finds all connected components of non-background cells, selects the largest
    one (by area), and crops to its bounding box. This is color-independent,
    making it robust to tasks where the target object has different colors
    in different demo pairs.
    
    Args:
        grid: Input grid.
        bg_color: Background color (default 0).
    """
    from scipy import ndimage
    g = grid[0] if grid.ndim == 3 else grid
    
    # Find connected components of non-background cells
    mask = g != bg_color
    if not np.any(mask):
        return grid
    
    labeled, num = ndimage.label(mask)
    if num == 0:
        return grid
    
    # Find the largest CC by area
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    
    # Get bounding box of largest CC
    objs = ndimage.find_objects(labeled)
    if largest_idx - 1 >= len(objs) or objs[largest_idx - 1] is None:
        return grid
    
    sl = objs[largest_idx - 1]
    result = g[sl[0], sl[1]]
    return result[np.newaxis, :, :] if grid.ndim == 3 else result


def _replicate_obj(grid: np.ndarray, direction: str = 'right', **_: Any) -> np.ndarray:
    "Replicate the largest object in a direction."
    from scipy import ndimage
    g = grid[0] if grid.ndim == 3 else grid
    binary = (g != 0).astype(np.int8)
    labeled, num = ndimage.label(binary)
    if num == 0:
        return grid
    best = max(range(1, num+1), key=lambda i: np.sum(labeled == i))
    mask = (labeled == best).astype(g.dtype)
    if direction == 'right':
        result = np.hstack([g, mask])
    elif direction == 'left':
        result = np.hstack([mask, g])
    elif direction == 'down':
        result = np.vstack([g, mask])
    elif direction == 'up':
        result = np.vstack([mask, g])
    else:
        result = g.copy()
    return result[np.newaxis, :, :] if grid.ndim == 3 else result


def _pad_row(grid: np.ndarray, num: int = 1, top: bool = False, **_: Any) -> np.ndarray:
    "Pad rows to grid."
    g = grid[0] if grid.ndim == 3 else grid
    pad = np.zeros((num, g.shape[1]), dtype=g.dtype)
    result = np.vstack([pad, g]) if top else np.vstack([g, pad])
    return result[np.newaxis, :, :] if grid.ndim == 3 else result



# ======== v2.4.9 New Primitives (batch add) ========

def _label_cc(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Label connected components. Each component gets a unique integer ID."""
    if grid.ndim == 3:
        g = grid[0]
        result = np.zeros_like(g)
        labeled, num = ndimage.label(g != 0)
        result = labeled.astype(grid.dtype)
        return result[np.newaxis, :, :]
    else:
        result = np.zeros_like(grid)
        labeled, num = ndimage.label(grid != 0)
        return labeled.astype(grid.dtype)


def _count_cc(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Count connected components. Return grid filled with count value."""
    if grid.ndim == 3:
        g = grid[0]
    else:
        g = grid
    labeled, num = ndimage.label(g != 0)
    result = np.full_like(g, num)
    if grid.ndim == 3:
        return result[np.newaxis, :, :]
    return result


def _symmetric_fill(grid: np.ndarray, axis: str = 'horizontal', **_: Any) -> np.ndarray:
    """Complete symmetric pattern by mirroring across axis."""
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    h, w = g.shape
    result = g.copy()
    if axis == 'horizontal':
        # Mirror top half to bottom half
        for r in range(h // 2):
            result[h - 1 - r, :] = g[r, :]
    elif axis == 'vertical':
        for c in range(w // 2):
            result[:, w - 1 - c] = g[:, c]
    elif axis == 'diag_main':
        # Mirror across main diagonal
        result = np.maximum(result, result.T)
    elif axis == 'diag_anti':
        # Mirror across anti-diagonal  
        flipped = np.flipud(np.fliplr(g))
        result = np.maximum(g, flipped)
    
    if is_3d:
        return result[np.newaxis, :, :]
    return result


def _apply_if(grid: np.ndarray, cond: str = 'has_color', 
             color: int = 1, op: str = 'invert', **_: Any) -> np.ndarray:
    """Apply operation only if condition is met."""
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    
    result = g.copy()
    meet = False
    if cond == 'has_color':
        meet = np.any(g == color)
    elif cond == 'all_same':
        meet = len(np.unique(g[g != 0])) <= 1
    elif cond == 'is_uniform':
        meet = np.all(g[g != 0] == g[g != 0][0]) if np.any(g != 0) else True
        
    if meet:
        if op == 'invert':
            result = (g.max() - g) * (g != 0)  # simplified invert
        elif op == 'clear':
            result = np.zeros_like(g)
        elif op == 'fill':
            result = np.full_like(g, color)
            
    if is_3d:
        return result[np.newaxis, :, :]
    return result


def _remove_obj(grid: np.ndarray, prop: str = 'smallest', **_: Any) -> np.ndarray:
    """Remove object by property (smallest/largest/leftmost/etc)."""
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    
    binary = (g != 0).astype(np.int8)
    labeled, num = ndimage.label(binary)
    if num == 0:
        if is_3d:
            return grid
        return g
        
    # Find target object
    target = 0
    if prop == 'smallest':
        sizes = [np.sum(labeled == i) for i in range(1, num+1)]
        target = np.argmin(sizes) + 1
    elif prop == 'largest':
        sizes = [np.sum(labeled == i) for i in range(1, num+1)]
        target = np.argmax(sizes) + 1
    elif prop == 'leftmost':
        for i in range(1, num+1):
            cols = np.where(labeled == i)[1]
            if target == 0 or cols.min() < np.where(labeled == target)[1].min():
                target = i
                
    result = g.copy()
    if target > 0:
        result[labeled == target] = 0
            
    if is_3d:
        return result[np.newaxis, :, :]
    return result


def _object_count(grid: np.ndarray, **_: Any) -> np.ndarray:
    """Replace grid with object count (each cell = count)."""
    if grid.ndim == 3:
        g = grid[0]
    else:
        g = grid
    labeled, num = ndimage.label(g != 0)
    result = np.full_like(g, num)
    if grid.ndim == 3:
        return result[np.newaxis, :, :]
    return result


def _fill_gradient(grid: np.ndarray, direction: str = 'left_to_right', **_: Any) -> np.ndarray:
    """Fill with gradient pattern."""
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    h, w = g.shape
    result = np.zeros_like(g)
    if direction == 'left_to_right':
        for c in range(w):
            result[:, c] = c + 1
    elif direction == 'top_to_bottom':
        for r in range(h):
            result[r, :] = r + 1
    elif direction == 'diag':
        for r in range(h):
            for c in range(w):
                result[r, c] = (r + c) % 10 + 1
                
    if is_3d:
        return result[np.newaxis, :, :]
    return result



def _fill_if(grid: np.ndarray, color: int = 1, cond: str = 'has_color', 
             cond_color: int = 1, **_: Any) -> np.ndarray:
    """Fill with color only if condition is met.
    
    Args:
        color: Fill color
        cond: Condition type ('has_color', 'is_uniform', 'count_eq')
        cond_color: Condition parameter (color to check, or count threshold)
    """
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    
    result = g.copy()
    meet = False
    
    if cond == 'has_color':
        meet = np.any(g == cond_color)
    elif cond == 'is_uniform':
        non_zero = g[g != 0]
        meet = len(np.unique(non_zero)) <= 1 if len(non_zero) > 0 else True
    elif cond == 'count_eq':
        meet = np.sum(g != 0) == cond_color
        
    if meet:
        result[g != 0] = color
        
    if is_3d:
        return result[np.newaxis, :, :]
    return result


def _rotate_object(grid: np.ndarray, k: int = 1, **_: Any) -> np.ndarray:
    """Rotate the largest object by k*90 degrees.
    
    Args:
        k: Number of 90-degree rotations (1, 2, 3)
    """
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    
    # Find largest object
    binary = (g != 0).astype(np.int8)
    labeled, num = ndimage.label(binary)
    if num == 0:
        if is_3d:
            return grid
        return g
    
    sizes = [np.sum(labeled == i) for i in range(1, num+1)]
    largest = np.argmax(sizes) + 1
    
    # Extract object mask and bounding box
    mask = labeled == largest
    rows, cols = np.where(mask)
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    
    # Extract object region
    obj_region = g[r_min:r_max+1, c_min:c_max+1].copy()
    obj_mask = mask[r_min:r_max+1, c_min:c_max+1]
    
    # Rotate region
    rotated = np.rot90(obj_region, k=k)
    rotated_mask = np.rot90(obj_mask, k=k)
    
    # Create result with same shape as original
    result = g.copy()
    r_new, c_new = rotated.shape
    
    # Place rotated object at same location
    r_place = r_min
    c_place = c_min
    
    # Make sure it fits
    if r_place + r_new <= g.shape[0] and c_place + c_new <= g.shape[1]:
        result[r_place:r_place+r_new, c_place:c_place+c_new][rotated_mask] = rotated[rotated_mask]
    
    if is_3d:
        return result[np.newaxis, :, :]
    return result


def _scale_object(grid: np.ndarray, factor: int = 2, **_: Any) -> np.ndarray:
    """Scale the largest object by factor.
    
    Args:
        factor: Scaling factor (2, 3, 4)
    """
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    
    # Find largest object
    binary = (g != 0).astype(np.int8)
    labeled, num = ndimage.label(binary)
    if num == 0:
        if is_3d:
            return grid
        return g
    
    sizes = [np.sum(labeled == i) for i in range(1, num+1)]
    largest = np.argmax(sizes) + 1
    
    # Extract object mask
    mask = labeled == largest
    color = g[mask][0] if np.any(mask) else 1
    
    # Create scaled object
    h, w = g.shape
    new_h, new_w = h * factor, w * factor
    result = np.zeros((new_h, new_w), dtype=g.dtype)
    
    # Scale: each pixel becomes factor x factor block
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                result[r*factor:(r+1)*factor, c*factor:(c+1)*factor] = g[r, c]
    
    if is_3d:
        return result[np.newaxis, :, :]
    return result


def _diagonal_fill(grid: np.ndarray, color: int = 1, diag: str = 'main', **_: Any) -> np.ndarray:
    """Fill diagonal lines with color.
    
    Args:
        color: Fill color
        diag: Diagonal type ('main', 'anti', 'all')
    """
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    
    result = g.copy()
    h, w = g.shape
    
    if diag == 'main':
        for i in range(min(h, w)):
            result[i, i] = color
    elif diag == 'anti':
        for i in range(min(h, w)):
            result[i, w-1-i] = color
    elif diag == 'all':
        # Fill both diagonals
        for i in range(min(h, w)):
            result[i, i] = color
            result[i, w-1-i] = color
            
    if is_3d:
        return result[np.newaxis, :, :]
    return result


def _pattern_extend(grid: np.ndarray, pattern: str = 'checkerboard', 
                   color: int = 1, **_: Any) -> np.ndarray:
    """Extend pattern to fill entire grid.
    
    Args:
        pattern: Pattern type ('checkerboard', 'striped_h', 'striped_v', 'dot')
        color: Pattern color
    """
    if grid.ndim == 3:
        g = grid[0]
        is_3d = True
    else:
        g = grid
        is_3d = False
    
    h, w = g.shape
    result = np.zeros((h, w), dtype=g.dtype)
    
    if pattern == 'checkerboard':
        for r in range(h):
            for c in range(w):
                if (r + c) % 2 == 0:
                    result[r, c] = color
    elif pattern == 'striped_h':
        for r in range(h):
            if r % 2 == 0:
                result[r, :] = color
    elif pattern == 'striped_v':
        for c in range(w):
            if c % 2 == 0:
                result[:, c] = color
    elif pattern == 'dot':
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                result[r, c] = color
                
    if is_3d:
        return result[np.newaxis, :, :]
    return result


# ============================================================
# v2.9: High-impact primitives for 68% accuracy target
# ============================================================

def _tile_seed(grid: np.ndarray, bg_color: int = 0, **_: Any) -> np.ndarray:
    """Detect the seed pattern (bounding box of non-bg) and tile it across the entire grid.
    
    This handles tasks where the input has a small pattern in one corner
    and the output tiles that pattern across the whole grid.
    
    Args:
        grid: Input grid.
        bg_color: Background color to detect seed against (default 0).
    """
    h, w = grid.shape
    result = np.zeros_like(grid)
    
    # Find bounding box of non-background region
    non_bg = grid != bg_color
    if not np.any(non_bg):
        return grid
    
    rows = np.where(np.any(non_bg, axis=1))[0]
    cols = np.where(np.any(non_bg, axis=0))[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return grid
    
    min_r, max_r = rows[0], rows[-1]
    min_c, max_c = cols[0], cols[-1]
    
    # Extract seed pattern
    seed = grid[min_r:max_r+1, min_c:max_c+1].copy()
    ph, pw = seed.shape
    
    if ph == 0 or pw == 0:
        return grid
    
    # Tile seed across entire grid (starting from 0,0)
    for r in range(0, h, ph):
        for c in range(0, w, pw):
            end_r = min(r + ph, h)
            end_c = min(c + pw, w)
            result[r:end_r, c:end_c] = seed[:end_r-r, :end_c-c]
    
    return result


def _fill_by_period(grid: np.ndarray, period_h: int = 0, period_w: int = 0, **_: Any) -> np.ndarray:
    """Fill empty (0) cells using a detected periodic pattern.
    
    v2.9.1: Improved period detection — also considers output pattern
    from all cells (not just non-zero). For background fill tasks,
    detects the period from the non-zero cells and fills 0s accordingly.
    
    Args:
        grid: Input grid.
        period_h: Pre-detected horizontal period (0 = auto-detect).
        period_w: Pre-detected vertical period (0 = auto-detect).
    """
    h, w = grid.shape
    result = grid.copy()
    
    non_zero = grid != 0
    if not np.any(non_zero):
        return result
    
    # Auto-detect period if not provided
    if period_h <= 0 or period_w <= 0:
        # Strategy: find the smallest period (ph, pw) such that all non-zero
        # cells are consistent with a repeating pattern of that size
        best_ph, best_pw = 1, 1
        best_score = -1
        best_pattern = {}
        
        max_ph = min(h, 12)
        max_pw = min(w, 12)
        
        for ph in range(1, max_ph + 1):
            for pw in range(1, max_pw + 1):
                pattern = {}
                consistent = True
                filled_count = 0
                
                for r in range(h):
                    for c in range(w):
                        if grid[r, c] != 0:
                            key = (r % ph, c % pw)
                            if key in pattern:
                                if pattern[key] != grid[r, c]:
                                    consistent = False
                                    break
                            else:
                                pattern[key] = grid[r, c]
                                filled_count += 1
                    if not consistent:
                        break
                
                if consistent and filled_count > 0:
                    # Prefer periods that explain more cells with fewer pattern entries
                    score = filled_count / (ph * pw + 1)
                    if score > best_score:
                        best_score = score
                        best_ph, best_pw = ph, pw
                        best_pattern = pattern
        
        if best_score <= 0:
            return result
        
        period_h = best_ph
        period_w = best_pw
        pattern = best_pattern
    else:
        # Build pattern from provided period
        pattern = {}
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    key = (r % period_h, c % period_w)
                    pattern[key] = grid[r, c]
    
    # Fill empty cells using the detected pattern
    for r in range(h):
        for c in range(w):
            if result[r, c] == 0:
                key = (r % period_h, c % period_w)
                if key in pattern:
                    result[r, c] = pattern[key]
    
    return result


def _crop_to_bbox(grid: np.ndarray, bg_color: int = 0, **_: Any) -> np.ndarray:
    """Crop grid to the bounding box of non-background region.
    
    Handles size-decrease tasks where the output is a sub-region of the input.
    
    Args:
        grid: Input grid.
        bg_color: Background color (default 0).
    """
    non_bg = grid != bg_color
    if not np.any(non_bg):
        return grid
    
    rows = np.where(np.any(non_bg, axis=1))[0]
    cols = np.where(np.any(non_bg, axis=0))[0]
    
    min_r, max_r = rows[0], rows[-1]
    min_c, max_c = cols[0], cols[-1]
    
    return grid[min_r:max_r+1, min_c:max_c+1].copy()


def _fill_empty_neighbor(grid: np.ndarray, bg_color: int = 0, **_: Any) -> np.ndarray:
    """Fill empty (background) cells with the color of the nearest non-empty neighbor.
    
    Uses BFS from non-empty cells to fill empty cells.
    Handles tasks where empty regions are filled based on surrounding colors.
    
    Args:
        grid: Input grid.
        bg_color: Background color to fill (default 0).
    """
    h, w = grid.shape
    result = grid.copy()
    
    # BFS from non-background cells
    from collections import deque
    queue = deque()
    visited = np.zeros((h, w), dtype=bool)
    
    # Initialize queue with all non-background cells
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg_color:
                queue.append((r, c, grid[r, c]))
                visited[r, c] = True
    
    # BFS to fill empty cells
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c, color = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if result[nr, nc] == bg_color:
                    result[nr, nc] = color
                    visited[nr, nc] = True
                    queue.append((nr, nc, color))
    
    return result


def _direct_map(grid: np.ndarray, mapping: dict | None = None, 
                period_h: int = 1, period_w: int = 1, **_: Any) -> np.ndarray:
    """Apply a pre-computed pixel mapping from demo pairs.
    
    The mapping uses (color, r % period_h, c % period_w) as key.
    This is the most powerful primitive — it directly learns the
    transformation from examples.
    
    Args:
        grid: Input grid.
        mapping: Dict mapping "color,row_mod,col_mod" -> new_color (string keys for JSON compat).
        period_h: Horizontal period for position-dependent mapping.
        period_w: Vertical period for position-dependent mapping.
    """
    if mapping is None:
        return grid
    
    h, w = grid.shape
    result = grid.copy()
    
    for r in range(h):
        for c in range(w):
            key = f"{int(grid[r, c])},{r % period_h},{c % period_w}"
            if key in mapping:
                result[r, c] = mapping[key]
    
    return result


def _repeat_grid(grid: np.ndarray, factor_h: int = 2, factor_w: int = 2, **_: Any) -> np.ndarray:
    """Repeat the entire grid factor_h × factor_w times.
    
    Handles size-increase tasks where the output is a tiled version of input.
    Different from 'tile' in that it repeats the ENTIRE grid (including padding).
    
    Args:
        grid: Input grid.
        factor_h: Vertical repeat factor.
        factor_w: Horizontal repeat factor.
    """
    if factor_h <= 1 and factor_w <= 1:
        return grid
    
    result = np.tile(grid, (factor_h, factor_w))
    return result


# ============================================================
# v3.0: High-impact primitives for 68% accuracy target
# ============================================================

def _universal_color_map(grid: np.ndarray, mapping: dict | None = None, **_: Any) -> np.ndarray:
    """Apply universal color mapping learned from all demo pairs.

    Handles multi-to-one mappings (multiple input colors -> same output color).
    Only applies to colors present in the mapping; unmapped colors are unchanged.

    Args:
        grid: Input grid as int8 ndarray.
        mapping: Dict mapping old_color (int or str key) -> new_color (int).

    Returns:
        Grid with colors remapped.
    """
    result = grid.copy()
    if not mapping:
        return result
    for old_color, new_color in mapping.items():
        old_c = int(old_color)
        new_c = int(new_color)
        if old_c != new_c:
            result[grid == old_c] = new_c
    return result


def _smart_flood_fill(grid: np.ndarray, color: int = 1, connectivity: int = 4,
                      fill_target: str = "enclosed", **_: Any) -> np.ndarray:
    """Smart flood-fill with configurable connectivity and target region.

    Args:
        grid: Input grid.
        color: Fill color.
        connectivity: 4 (orthogonal) or 8 (orthogonal + diagonal).
        fill_target: 'enclosed' (fill bg regions not touching border),
                     'border' (fill bg regions touching border),
                     'all_bg' (fill all background),
                     'all_nonzero' (fill all non-zero regions with color).

    Returns:
        Grid with flood-fill applied.
    """
    result = grid.copy()

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.int32)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)

    if fill_target == "all_bg":
        result[grid == 0] = color
        return result
    elif fill_target == "all_nonzero":
        result[grid != 0] = color
        return result

    # For enclosed/border: label background regions
    binary = (grid == 0).astype(np.int32)
    labeled, num = ndimage.label(binary, structure=structure)
    if num == 0:
        return result

    # Find border-touching labels
    border_labels: set[int] = set()
    border_labels.update(labeled[0, :].tolist())
    border_labels.update(labeled[-1, :].tolist())
    border_labels.update(labeled[:, 0].tolist())
    border_labels.update(labeled[:, -1].tolist())
    border_labels.discard(0)

    for label_id in range(1, num + 1):
        if fill_target == "enclosed" and label_id not in border_labels:
            result[labeled == label_id] = color
        elif fill_target == "border" and label_id in border_labels:
            result[labeled == label_id] = color

    return result


def _grid_diff_apply(grid: np.ndarray, diff_type: str = "border",
                     color: int = 1, **_: Any) -> np.ndarray:
    """Apply a pattern-based difference to the grid.

    Args:
        grid: Input grid.
        diff_type: 'border' (add border), 'frame' (add frame around objects),
                   'corners' (fill corners), 'extend_lines' (extend lines),
                   'fill_adjacent' (fill cells adjacent to non-zero).
        color: Color to use for the difference.

    Returns:
        Grid with the difference pattern applied.
    """
    result = grid.copy()
    h, w = grid.shape

    if diff_type == "border":
        result[0, :] = color
        result[-1, :] = color
        result[:, 0] = color
        result[:, -1] = color
    elif diff_type == "frame":
        binary = (grid != 0).astype(np.int32)
        dilated = ndimage.binary_dilation(binary, iterations=1)
        frame = dilated & ~binary.astype(bool)
        result[frame] = color
    elif diff_type == "corners":
        if h > 0 and w > 0:
            result[0, 0] = color
            result[0, w - 1] = color
            result[h - 1, 0] = color
            result[h - 1, w - 1] = color
    elif diff_type == "extend_lines":
        for r in range(h):
            non_zero_cols = np.where(grid[r, :] != 0)[0]
            if len(non_zero_cols) >= 2:
                c_val = int(grid[r, non_zero_cols[0]])
                result[r, non_zero_cols[0]:non_zero_cols[-1] + 1] = c_val
        for c in range(w):
            non_zero_rows = np.where(grid[:, c] != 0)[0]
            if len(non_zero_rows) >= 2:
                r_val = int(grid[non_zero_rows[0], c])
                result[non_zero_rows[0]:non_zero_rows[-1] + 1, c] = r_val
    elif diff_type == "fill_adjacent":
        binary = (grid != 0).astype(np.int32)
        dilated = ndimage.binary_dilation(binary, iterations=1)
        adjacent = dilated & (grid == 0)
        result[adjacent] = color

    return result


def _extract_and_recolor(grid: np.ndarray, source_color: int = 1,
                         target_color: int = 2, keep_others: bool = True,
                         **_: Any) -> np.ndarray:
    """Extract objects of a specific color and recolor them.

    Args:
        grid: Input grid.
        source_color: Color of objects to recolor.
        target_color: New color for the objects.
        keep_others: If True, keep other objects unchanged. If False, remove them.

    Returns:
        Grid with objects recolored.
    """
    result = grid.copy()
    if not keep_others:
        mask_other = (grid != source_color) & (grid != 0)
        result[mask_other] = 0
    result[grid == source_color] = target_color
    return result


def _conditional_map(grid: np.ndarray, rules: list | None = None,
                     **_: Any) -> np.ndarray:
    """Apply conditional pixel mapping based on neighbor context.

    Args:
        grid: Input grid.
        rules: List of rule dicts, each with:
            'center': center pixel color to match,
            'neighbor': neighbor color to check,
            'direction': 'up', 'down', 'left', 'right', 'any',
            'output': output color for matching pixels.

    Returns:
        Grid with conditional mapping applied.
    """
    if not rules:
        return grid.copy()

    result = grid.copy()
    h, w = grid.shape

    for rule in rules:
        center_c = int(rule.get("center", -1))
        neighbor_c = int(rule.get("neighbor", -1))
        direction = rule.get("direction", "any")
        output_c = int(rule.get("output", 0))

        for r in range(h):
            for c in range(w):
                if int(grid[r, c]) != center_c:
                    continue

                match = False
                neighbors: list[tuple[int, int]] = []
                if direction == "any":
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            neighbors.append((nr, nc))
                elif direction == "up" and r > 0:
                    neighbors = [(r - 1, c)]
                elif direction == "down" and r < h - 1:
                    neighbors = [(r + 1, c)]
                elif direction == "left" and c > 0:
                    neighbors = [(r, c - 1)]
                elif direction == "right" and c < w - 1:
                    neighbors = [(r, c + 1)]

                for nr, nc in neighbors:
                    if int(grid[nr, nc]) == neighbor_c:
                        match = True
                        break

                if match:
                    result[r, c] = output_c

    return result


def _nearest_neighbor_transform(grid: np.ndarray, train_data: list | None = None,
                                **_: Any) -> np.ndarray:
    """Transform input using the transformation from the most similar training pair.

    Finds the training input most similar to the test input, then applies
    the same color mapping that transforms that training input to its output.

    Args:
        grid: Input grid.
        train_data: List of (input_grid, output_grid) tuples from training pairs.

    Returns:
        Transformed grid.
    """
    if not train_data:
        return grid.copy()

    g = grid[0] if grid.ndim == 3 else grid

    best_sim = -1.0
    best_idx = 0

    for idx, (train_in, train_out) in enumerate(train_data):
        t_in = np.array(train_in, dtype=np.int8)
        if t_in.ndim == 3:
            t_in = t_in[0]

        if t_in.shape == g.shape:
            sim = float(np.sum(t_in == g)) / max(g.size, 1)
        else:
            h = min(t_in.shape[0], g.shape[0])
            w = min(t_in.shape[1], g.shape[1])
            if h > 0 and w > 0:
                sim = float(np.sum(t_in[:h, :w] == g[:h, :w])) / (h * w)
            else:
                sim = 0.0

        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    train_in, train_out = train_data[best_idx]
    t_in = np.array(train_in, dtype=np.int8)
    t_out = np.array(train_out, dtype=np.int8)
    if t_in.ndim == 3:
        t_in = t_in[0]
    if t_out.ndim == 3:
        t_out = t_out[0]

    # Apply same color mapping
    if t_in.shape == g.shape:
        color_map: dict[int, int] = {}
        for r in range(t_in.shape[0]):
            for c in range(t_in.shape[1]):
                if t_in[r, c] != t_out[r, c]:
                    color_map[int(t_in[r, c])] = int(t_out[r, c])

        result = g.copy()
        for old_c, new_c in color_map.items():
            result[g == old_c] = new_c
        return result

    return g.copy()


def _replace_marker_with_border(grid: np.ndarray, marker_color: int = 3,
                                **_: Any) -> np.ndarray:
    """Replace marker pixels with the nearest border color.

    Detects border colors (non-zero colors on grid edges) and replaces
    all marker_color pixels with the color of the nearest border.
    Only considers borders with a single consistent non-zero color.

    Args:
        grid: Input grid.
        marker_color: Color of pixels to replace.

    Returns:
        Grid with marker pixels replaced by nearest border color.
    """
    result = grid.copy()
    h, w = grid.shape

    border_colors: dict[str, int] = {}

    # Only use borders with a single consistent non-zero color
    def _check_border(colors_1d: np.ndarray, name: str) -> None:
        """Register a border color if all non-zero pixels share one color."""
        nz = colors_1d[colors_1d != 0]
        if len(nz) > 0:
            unique_nz = set(nz.tolist())
            if len(unique_nz) == 1:
                border_colors[name] = int(unique_nz.pop())

    _check_border(grid[0, :], "top")
    _check_border(grid[-1, :], "bottom")
    _check_border(grid[:, 0], "left")
    _check_border(grid[:, -1], "right")

    if not border_colors:
        return result

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == marker_color:
                min_dist = float("inf")
                nearest_c = int(grid[r, c])

                for bname, bc in border_colors.items():
                    if bname == "top":
                        dist = r
                    elif bname == "bottom":
                        dist = h - 1 - r
                    elif bname == "left":
                        dist = c
                    elif bname == "right":
                        dist = w - 1 - c
                    else:
                        continue

                    if dist < min_dist:
                        min_dist = dist
                        nearest_c = bc

                result[r, c] = nearest_c

    return result


def _gravity_stack(grid: np.ndarray, direction: str = "down",
                   **_: Any) -> np.ndarray:
    """Apply gravity to stack all non-zero pixels in a direction.

    Preserves the relative order of pixels within each row/column.

    Args:
        grid: Input grid.
        direction: 'down', 'up', 'left', or 'right'.

    Returns:
        Grid with pixels stacked in the specified direction.
    """
    result = np.zeros_like(grid)
    h, w = grid.shape

    if direction == "down":
        for c in range(w):
            col = grid[:, c]
            nz = col[col != 0]
            if len(nz) > 0:
                result[h - len(nz):, c] = nz
    elif direction == "up":
        for c in range(w):
            col = grid[:, c]
            nz = col[col != 0]
            if len(nz) > 0:
                result[:len(nz), c] = nz
    elif direction == "left":
        for r in range(h):
            row = grid[r, :]
            nz = row[row != 0]
            if len(nz) > 0:
                result[r, :len(nz)] = nz
    elif direction == "right":
        for r in range(h):
            row = grid[r, :]
            nz = row[row != 0]
            if len(nz) > 0:
                result[r, w - len(nz):] = nz

    return result


def _draw_frame_around_objects(grid: np.ndarray, frame_color: int = 1,
                               **_: Any) -> np.ndarray:
    """Draw a 1-pixel frame around each connected object.

    Args:
        grid: Input grid.
        frame_color: Color of the frame.

    Returns:
        Grid with frames drawn around objects.
    """
    result = grid.copy()
    binary = (grid != 0).astype(np.int32)
    labeled, num = ndimage.label(binary)

    if num == 0:
        return result

    for label_id in range(1, num + 1):
        mask = labeled == label_id
        dilated = ndimage.binary_dilation(mask, iterations=1)
        frame = dilated & ~mask
        frame = frame & (grid == 0)
        result[frame] = frame_color

    return result


def _fill_enclosed_regions(grid: np.ndarray, color: int = 1,
                           bg_color: int = 0, **_: Any) -> np.ndarray:
    """Fill enclosed background regions with the specified color.

    Args:
        grid: Input grid.
        color: Fill color.
        bg_color: Background color to fill.

    Returns:
        Grid with enclosed regions filled.
    """
    result = grid.copy()
    binary = (grid == bg_color).astype(np.int32)
    labeled, num = ndimage.label(binary)

    if num == 0:
        return result

    border_labels: set[int] = set()
    border_labels.update(labeled[0, :].tolist())
    border_labels.update(labeled[-1, :].tolist())
    border_labels.update(labeled[:, 0].tolist())
    border_labels.update(labeled[:, -1].tolist())
    border_labels.discard(0)

    for label_id in range(1, num + 1):
        if label_id not in border_labels:
            result[labeled == label_id] = color

    return result


def _sort_objects_by_size(grid: np.ndarray, ascending: bool = True,
                          start_color: int = 1, **_: Any) -> np.ndarray:
    """Sort objects by size and recolor them sequentially.

    Args:
        grid: Input grid.
        ascending: If True, smallest gets color start_color.
        start_color: Starting color for the smallest/largest object.

    Returns:
        Grid with objects recolored by size rank.
    """
    result = np.zeros_like(grid)
    binary = (grid != 0).astype(np.int32)
    labeled, num = ndimage.label(binary)

    if num == 0:
        return grid.copy()

    obj_info: list[tuple[int, int]] = []
    for i in range(1, num + 1):
        size = int(np.sum(labeled == i))
        obj_info.append((size, i))

    obj_info.sort(reverse=not ascending)

    for rank, (_size, label_id) in enumerate(obj_info):
        c = start_color + rank
        if c > 9:
            c = 9
        result[labeled == label_id] = c

    return result


def _move_object_to_pos(grid: np.ndarray, color: int = 1,
                        target_row: int = 0, target_col: int = 0,
                        **_: Any) -> np.ndarray:
    """Move all pixels of a specific color to a target position.

    Args:
        grid: Input grid.
        color: Color of pixels to move.
        target_row: Target row for top-left of the moved object.
        target_col: Target column for top-left of the moved object.

    Returns:
        Grid with the object moved to the target position.
    """
    result = grid.copy()
    mask = grid == color
    if not np.any(mask):
        return result

    rows, cols = np.where(mask)
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())

    obj = grid[min_r:max_r + 1, min_c:max_c + 1].copy()
    obj_mask = mask[min_r:max_r + 1, min_c:max_c + 1]

    result[mask] = 0

    h, w = result.shape
    for r in range(obj.shape[0]):
        for c in range(obj.shape[1]):
            nr, nc = target_row + r, target_col + c
            if 0 <= nr < h and 0 <= nc < w and obj_mask[r, c]:
                result[nr, nc] = obj[r, c]

    return result


def _count_objects_mark(grid: np.ndarray, bg_color: int = 0,
                        mark_color: int = 1, **_: Any) -> np.ndarray:
    """Count objects and represent the count as a grid.

    Creates a grid where each cell contains the count of objects.

    Args:
        grid: Input grid.
        bg_color: Background color.
        mark_color: Unused (kept for API compatibility).

    Returns:
        Grid filled with object count.
    """
    binary = (grid != bg_color).astype(np.int32)
    _, num = ndimage.label(binary)
    result = np.full_like(grid, num)
    return result


def _propagate_color(grid: np.ndarray, source_color: int = 1,
                     target_color: int = 0, **_: Any) -> np.ndarray:
    """Propagate source color to adjacent target color cells via BFS.

    Args:
        grid: Input grid.
        source_color: Color to propagate.
        target_color: Color to replace (default 0 = background).

    Returns:
        Grid with color propagated.
    """
    from collections import deque

    result = grid.copy()
    h, w = grid.shape

    queue: deque = deque()
    visited = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if grid[r, c] == source_color:
                queue.append((r, c))
                visited[r, c] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if result[nr, nc] == target_color:
                    result[nr, nc] = source_color
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    return result


def _connect_dots(grid: np.ndarray, color: int = 1, **_: Any) -> np.ndarray:
    """Connect dots (single pixels) with lines in row-major order.

    Args:
        grid: Input grid.
        color: Unused (lines use original pixel colors).

    Returns:
        Grid with dots connected by lines.
    """
    result = grid.copy()
    h, w = grid.shape

    points: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                points.append((r, c, int(grid[r, c])))

    for i in range(len(points) - 1):
        r1, c1, col1 = points[i]
        r2, c2, col2 = points[i + 1]
        if col1 == col2:
            dr = abs(r2 - r1)
            dc = abs(c2 - c1)
            steps = max(dr, dc, 1)
            for s in range(steps + 1):
                r = int(round(r1 + (r2 - r1) * s / steps))
                c = int(round(c1 + (c2 - c1) * s / steps))
                if 0 <= r < h and 0 <= c < w:
                    result[r, c] = col1

    return result


def _register_primitives() -> None:
    """Register all DSL primitives in the registry."""
    DSLElement._registry = {
        "resize": _resize,
        "fill-region": _fill_region,
        "color-swap": _color_swap,
        "mirror": _mirror,
        "rotate": _rotate,
        "scale": _scale,
        "map-color": _map_color,
        "complete-shape": _complete_shape,
        "objects-complete": _objects_complete,
        "move": _move,
        "copy": _copy,
        "gravity": _gravity,
        "flood-fill": _flood_fill,
        "extract-pattern": _extract_pattern,
        "symmetry-detect": _symmetry_detect,
        "count": _count,
        "tile": _tile,
        "crop": _crop,
        "overlay": _overlay,
        "subtract": _subtract,
        "union": _union,
        "intersection": _intersection,
        "boundary-detect": _boundary_detect,
        "skeleton": _skeleton,
        "dilate": _dilate,
        "erode": _erode,
        "label-connected": _label_connected,
        "histogram": _histogram,
        "find-objects": _find_objects,
        "draw-line": _draw_line,
        # New primitives v2.6
        "multi-swap": _multi_swap,
        "shift-object": _shift_object,
        "complete-pattern": _complete_pattern,
        "map-by-function": _map_by_function,
        # New primitives v2.4.5
        "extract-largest-object": _extract_largest_object,
        "extract-smallest-object": _extract_smallest_object,
        # New primitives v2.4.6
        "invert-colors": _invert_colors,
        "fill-connected": _fill_connected,
        "recolor-by-cc": _recolor_by_cc,
        "tile-repeat": _tile_repeat,
        "select-obj-by": _select_obj_by,
        "sort-obj-by": _sort_obj_by,
        # New primitives v2.4.7
        "scale-pattern": _scale_pattern,
        # New primitives v2.4.8
        "crop-to-obj": _crop_to_obj,
        "replicate-obj": _replicate_obj,
        "pad-row": _pad_row,
        # New primitives v2.4.9
        "label-cc": _label_cc,
        "count-cc": _count_cc,
        "symmetric-fill": _symmetric_fill,
        "apply-if": _apply_if,
        "remove-obj": _remove_obj,
        "object-count": _object_count,
        "fill-gradient": _fill_gradient,
        # New primitives v2.5.0 - High-value primitives
        "fill-if": _fill_if,
        "rotate-object": _rotate_object,
        "scale-object": _scale_object,
        "diagonal-fill": _diagonal_fill,
        "pattern-extend": _pattern_extend,
        # v2.9: High-impact primitives for 68% accuracy target
        "tile-seed": _tile_seed,
        "fill-by-period": _fill_by_period,
        "crop-to-bbox": _crop_to_bbox,
        "fill-empty-neighbor": _fill_empty_neighbor,
        "direct-map": _direct_map,
        "repeat-grid": _repeat_grid,
        "crop-to-largest-cc": _crop_to_largest_cc,
        # v3.0: High-impact primitives for 68% accuracy target
        "universal-color-map": _universal_color_map,
        "smart-flood-fill": _smart_flood_fill,
        "grid-diff-apply": _grid_diff_apply,
        "extract-and-recolor": _extract_and_recolor,
        "conditional-map": _conditional_map,
        "nearest-neighbor-transform": _nearest_neighbor_transform,
        "replace-marker-with-border": _replace_marker_with_border,
        "gravity-stack": _gravity_stack,
        "draw-frame-around-objects": _draw_frame_around_objects,
        "fill-enclosed-regions": _fill_enclosed_regions,
        "sort-objects-by-size": _sort_objects_by_size,
        "move-object-to-pos": _move_object_to_pos,
        "count-objects-mark": _count_objects_mark,
        "propagate-color": _propagate_color,
        "connect-dots": _connect_dots,
    }


# Register on import
_register_primitives()


def get_all_primitives() -> list[DSLElement]:
    """Get all registered DSL primitives as DSLElement instances.

    Returns:
        List of DSLElement instances for all primitives.
    """
    return [DSLElement(name) for name in DSLElement._registry.keys()]


def get_primitive_names() -> list[str]:
    """Get names of all registered primitives.

    Returns:
        List of primitive names.
    """
    return list(DSLElement._registry.keys())
