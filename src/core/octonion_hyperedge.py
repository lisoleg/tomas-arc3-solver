"""Octonion hyperedge encoding: reversible grid-to-octonion transformation.

Encodes ARC grid pixels into 8-component octonion vectors with cognitive
semantics: real_mask, e1_x, e2_y, e3_color, e4_component_id, e5_symmetry,
e6_boundary, e7_area_ratio. Encoding is fully reversible — decode_to_grid
reconstructs the original grid.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


class EncodingReversibilityError(Exception):
    """Raised when octonion encoding fails to reversibly decode."""


class OctonionHyperEdge:
    """Encodes an ARC grid as a set of octonion hyperedges.

    Each non-zero pixel in the grid is encoded as one octonion vector with
    8 float32 components, each carrying explicit cognitive semantics
    (TOMAS v2.0 GAT axiom system):

    - e0 (real_mask): Object existence — 1.0 if pixel is non-zero, else 0.0.
    - e1 (e1_x): Object constancy / x-coordinate normalized to [-1, 1].
    - e2 (e2_y): Translation invariance / y-coordinate normalized to [-1, 1].
    - e3 (e3_color): Rotation / color phase angle (color/9 * 2*pi).
    - e4 (e4_component_id): Scale / connected component ID (normalized).
    - e5 (e5_symmetry): Boundary touch / symmetry residual.
    - e6 (e6_boundary): Color mapping / boundary mask.
    - e7 (e7_area_ratio): Time phase / area ratio (temporal frame context).

    The encoding is reversible: ``decode_to_grid()`` reconstructs the
    original grid from the encoded octonion array.

    Neural analogy (Nature single-neuron study):
    The orthogonality of octonion imaginary components mirrors the
    feature selectivity of single neurons in visual cortex.
    """

    #: Human-readable cognitive semantics for each component.
    COMPONENT_NAMES: dict[int, str] = {
        0: "real_mask (Object Existence)",
        1: "e1_x (Object Constancy)",
        2: "e2_y (Translation Invariance)",
        3: "e3_color (Rotation / Color Phase)",
        4: "e4_component_id (Scale / Gestalt Grouping)",
        5: "e5_symmetry (Boundary Touch / Symmetry)",
        6: "e6_boundary (Color Mapping / Edge)",
        7: "e7_area_ratio (Time Phase / Frame Context)",
    }

    #: Cognitive semantic mapping from component index to description.
    COGNITIVE_SEMANTIC_MAP: dict[int, str] = {
        0: "object_existence",
        1: "object_constancy",
        2: "translation_invariance",
        3: "rotation_phase",
        4: "scale_gestalt",
        5: "boundary_touch",
        6: "color_mapping",
        7: "time_phase",
    }

    NUM_COMPONENTS: int = 8
    MAX_COLOR: int = 9

    def __init__(self, grid: np.ndarray | None = None, frame_idx: int = 0) -> None:
        """Initialize from a grid or as empty.

        Args:
            grid: Input grid as int8 ndarray of shape (H, W).
            frame_idx: Temporal frame index.
        """
        self.real_mask: float = 0.0
        self.e1_x: float = 0.0
        self.e2_y: float = 0.0
        self.e3_color: float = 0.0
        self.e4_component_id: float = 0.0
        self.e5_symmetry: float = 0.0
        self.e6_boundary: float = 0.0
        self.e7_area_ratio: float = 0.0
        self.frame_idx: int = frame_idx

        # Store encoded data for reversibility
        self._encoded_array: np.ndarray | None = None
        self._grid_shape: tuple[int, int] = (0, 0)

        if grid is not None:
            self._encoded_array = self.encode_grid(grid)
            self._grid_shape = grid.shape

    def encode_grid(self, grid: np.ndarray) -> np.ndarray:
        """Encode a grid into octonion hyperedge array.

        Each non-zero pixel becomes one 8-component octonion vector.
        Zero pixels are included with real_mask=0 for reversibility.

        Args:
            grid: Input grid as int8 ndarray of shape (H, W).

        Returns:
            float32 ndarray of shape (H*W, 8) — one octonion per pixel.
        """
        grid = np.asarray(grid, dtype=np.int8)
        h, w = grid.shape
        self._grid_shape = (h, w)

        total_pixels = h * w
        encoded = np.zeros((total_pixels, self.NUM_COMPONENTS), dtype=np.float32)

        # Generate coordinate grids
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        xs_flat = xs.flatten().astype(np.float32)
        ys_flat = ys.flatten().astype(np.float32)
        colors_flat = grid.flatten().astype(np.float32)

        # Component 0 (real): mask — 1.0 if non-zero pixel
        encoded[:, 0] = (colors_flat > 0).astype(np.float32)

        # Component 1 (e1_x): x normalized to [-1, 1]
        encoded[:, 1] = (xs_flat / max(w - 1, 1)) * 2.0 - 1.0

        # Component 2 (e2_y): y normalized to [-1, 1]
        encoded[:, 2] = (ys_flat / max(h - 1, 1)) * 2.0 - 1.0

        # Component 3 (e3_color): color as phase angle
        encoded[:, 3] = (colors_flat / self.MAX_COLOR) * 2.0 * np.pi

        # Component 4 (e4_component_id): connected component ID
        comp_labels = self._compute_connected_components(grid)
        max_comp = max(int(comp_labels.max()), 1)
        encoded[:, 4] = comp_labels.flatten().astype(np.float32) / max_comp

        # Component 5 (e5_symmetry): symmetry residual
        sym_residual = self._compute_symmetry_residual(grid)
        encoded[:, 5] = sym_residual.flatten().astype(np.float32)

        # Component 6 (e6_boundary): boundary touching
        boundary = self._compute_boundary_mask(grid)
        encoded[:, 6] = boundary.flatten().astype(np.float32)

        # Component 7 (e7_area_ratio): area ratio of connected component
        area_ratios = self._compute_area_ratios(comp_labels)
        encoded[:, 7] = area_ratios.flatten().astype(np.float32)

        self.normalize()
        self._encoded_array = encoded
        return encoded

    def _compute_connected_components(self, grid: np.ndarray) -> np.ndarray:
        """Compute connected component labels for non-zero pixels.

        Args:
            grid: Input grid.

        Returns:
            Label array of same shape as grid.
        """
        binary = (grid > 0).astype(np.int32)
        labeled, _ = ndimage.label(binary)
        return labeled

    def _compute_symmetry_residual(self, grid: np.ndarray) -> np.ndarray:
        """Compute symmetry residual for each pixel.

        Measures how well the grid's local region around each pixel
        matches the dominant symmetry axis.

        Args:
            grid: Input grid.

        Returns:
            Residual array of same shape, values in [0, 1].
        """
        h, w = grid.shape
        residual = np.zeros((h, w), dtype=np.float32)

        # Horizontal symmetry
        h_mirror = np.fliplr(grid)
        h_diff = np.abs(grid.astype(np.float32) - h_mirror.astype(np.float32))
        h_sym = 1.0 - np.mean(h_diff) / self.MAX_COLOR

        # Vertical symmetry
        v_mirror = np.flipud(grid)
        v_diff = np.abs(grid.astype(np.float32) - v_mirror.astype(np.float32))
        v_sym = 1.0 - np.mean(v_diff) / self.MAX_COLOR

        # Assign the better symmetry score as residual
        best_sym = max(h_sym, v_sym)
        residual[:, :] = best_sym

        return residual

    def _compute_boundary_mask(self, grid: np.ndarray) -> np.ndarray:
        """Compute boundary mask (pixels touching grid edge).

        Args:
            grid: Input grid.

        Returns:
            Binary mask of same shape, 1.0 for boundary pixels.
        """
        h, w = grid.shape
        boundary = np.zeros((h, w), dtype=np.float32)
        boundary[0, :] = 1.0
        boundary[-1, :] = 1.0
        boundary[:, 0] = 1.0
        boundary[:, -1] = 1.0
        # Only mark non-zero boundary pixels
        boundary *= (grid > 0).astype(np.float32)
        return boundary

    def _compute_area_ratios(self, comp_labels: np.ndarray) -> np.ndarray:
        """Compute area ratio for each pixel based on its component.

        Args:
            comp_labels: Connected component label array.

        Returns:
            Array of same shape with area ratio per pixel.
        """
        h, w = comp_labels.shape
        total = h * w
        ratios = np.zeros((h, w), dtype=np.float32)
        unique_labels = np.unique(comp_labels[comp_labels > 0])
        for label in unique_labels:
            count = np.sum(comp_labels == label)
            ratio = float(count) / max(total, 1)
            ratios[comp_labels == label] = ratio
        return ratios

    def decode_to_grid(self) -> np.ndarray:
        """Decode the octonion array back to the original grid.

        Reverses the encoding: extracts color from e3_color phase angle
        and reconstructs the grid using stored coordinates.

        Returns:
            Reconstructed grid as int8 ndarray.

        Raises:
            EncodingReversibilityError: If no encoded data is available.
        """
        if self._encoded_array is None:
            raise EncodingReversibilityError("No encoded data available for decoding")

        encoded = self._encoded_array
        h, w = self._grid_shape
        grid = np.zeros((h, w), dtype=np.int8)

        # Reconstruct from the flat array (pixels are in row-major order)
        for i in range(h * w):
            row, col = divmod(i, w)
            mask = encoded[i, 0]
            if mask > 0.5:
                # Decode color from phase angle
                phase = encoded[i, 3]
                color = int(round(phase / (2.0 * np.pi) * self.MAX_COLOR))
                color = max(0, min(self.MAX_COLOR, color))
                grid[row, col] = color

        return grid

    def compute_topo_invariants(self) -> dict[str, Any]:
        """Compute topological invariants from the encoded data.

        Returns:
            Dictionary with betti0, symmetry_residual, color_permutation_group.
        """
        if self._encoded_array is None:
            return {"betti0": 0, "symmetry_residual": 0.0, "color_permutation_group": []}

        encoded = self._encoded_array
        # Betti0: number of unique connected components (from e4)
        comp_ids = encoded[encoded[:, 0] > 0.5, 4] if np.any(encoded[:, 0] > 0.5) else np.array([])
        betti0 = len(np.unique(comp_ids)) if len(comp_ids) > 0 else 0

        # Symmetry residual: mean of e5
        sym_residual = float(np.mean(encoded[:, 5])) if len(encoded) > 0 else 0.0

        # Color permutation group: unique colors present
        mask_pixels = encoded[encoded[:, 0] > 0.5]
        if len(mask_pixels) > 0:
            phases = mask_pixels[:, 3]
            colors = np.round(phases / (2.0 * np.pi) * self.MAX_COLOR).astype(int)
            color_perm = sorted(set(colors.tolist()))
        else:
            color_perm = []

        return {
            "betti0": int(betti0),
            "symmetry_residual": round(sym_residual, 4),
            "color_permutation_group": color_perm,
        }

    def compute_betti0(self) -> int:
        """Compute Betti-0: number of connected components.

        Returns:
            Number of connected components.
        """
        return int(self.compute_topo_invariants()["betti0"])

    def compute_symmetry_residual(self) -> float:
        """Compute the mean symmetry residual.

        Returns:
            Symmetry residual value.
        """
        return float(self.compute_topo_invariants()["symmetry_residual"])

    def normalize(self) -> None:
        """Normalize octonion components to prevent overflow in long sequences.

        Ensures all components are in valid ranges:
        - real_mask, e6_boundary: [0, 1]
        - e1_x, e2_y: [-1, 1]
        - e3_color: [0, 2*pi]
        - e4_component_id: [0, 1]
        - e5_symmetry: [0, 1]
        - e7_area_ratio: [0, 1]
        """
        if self._encoded_array is None:
            return

        arr = self._encoded_array
        arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
        arr[:, 1] = np.clip(arr[:, 1], -1.0, 1.0)
        arr[:, 2] = np.clip(arr[:, 2], -1.0, 1.0)
        arr[:, 3] = np.clip(arr[:, 3], 0.0, 2.0 * np.pi)
        arr[:, 4] = np.clip(arr[:, 4], 0.0, 1.0)
        arr[:, 5] = np.clip(arr[:, 5], 0.0, 1.0)
        arr[:, 6] = np.clip(arr[:, 6], 0.0, 1.0)
        arr[:, 7] = np.clip(arr[:, 7], 0.0, 1.0)

    def to_array(self) -> np.ndarray:
        """Return the encoded octonion array.

        Returns:
            float32 ndarray of shape (N, 8).
        """
        if self._encoded_array is None:
            return np.zeros((0, self.NUM_COMPONENTS), dtype=np.float32)
        return self._encoded_array

    def get_grid_shape(self) -> tuple[int, int]:
        """Return the original grid shape.

        Returns:
            Tuple of (height, width).
        """
        return self._grid_shape
