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
        }
        MEDIUM_PRIMITIVES = {
            "complete-shape", "objects-complete", "gravity", "flood-fill",
            "extract-pattern", "symmetry-detect", "tile", "overlay", "subtract",
            "union", "intersection", "skeleton", "dilate", "erode",
            "label-connected", "histogram", "find-objects", "count",
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
