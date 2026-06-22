"""Hypergraph structure with topological invariants and Willems behavioral fibers."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy import ndimage

from src.core.octonion_hyperedge import OctonionHyperEdge


class HyperGraph:
    """A hypergraph of octonion hyperedges supporting temporal dimensions.

    Manages a collection of OctonionHyperEdge objects and provides
    topological invariant computation, hash fingerprinting, isomorphism
    checking, and Willems behavioral fiber computation.

    Attributes:
        edges: List of OctonionHyperEdge objects.
        nodes: Node dictionary for graph structure.
        is_temporal: Whether this is a temporal (video) hypergraph.
        temporal_length: Number of temporal frames.
    """

    def __init__(self, edges: list[OctonionHyperEdge] | None = None) -> None:
        """Initialize the hypergraph.

        Args:
            edges: Initial list of octonion hyperedges.
        """
        self.edges: list[OctonionHyperEdge] = edges or []
        self.nodes: dict[int, dict[str, Any]] = {}
        self.is_temporal: bool = False
        self.temporal_length: int = 0
        self._topo_hash: str | None = None

        # Detect temporal dimension
        if self.edges:
            frame_indices = {e.frame_idx for e in self.edges}
            if len(frame_indices) > 1:
                self.is_temporal = True
                self.temporal_length = max(frame_indices) + 1

    def add_edge(self, edge: OctonionHyperEdge) -> None:
        """Add an octonion hyperedge to the hypergraph.

        Args:
            edge: The OctonionHyperEdge to add.
        """
        self.edges.append(edge)
        self._topo_hash = None  # Invalidate cache
        if edge.frame_idx > 0:
            self.is_temporal = True
            self.temporal_length = max(self.temporal_length, edge.frame_idx + 1)

    def get_topo_hash(self) -> str:
        """Compute a topological hash fingerprint of the hypergraph.

        The hash encodes Betti-0, symmetry residual, color permutation group,
        grid dimensions, and pixel count for quick isomorphism pre-checking.

        Returns:
            Hex string hash fingerprint.
        """
        if self._topo_hash is not None:
            return self._topo_hash

        if not self.edges:
            self._topo_hash = hashlib.md5(b"empty").hexdigest()
            return self._topo_hash

        # Aggregate invariants across all edges
        total_betti0 = 0
        total_sym = 0.0
        all_colors: set[int] = set()
        total_pixels = 0
        grid_shapes: list[tuple[int, int]] = []

        for edge in self.edges:
            invariants = edge.compute_topo_invariants()
            total_betti0 += invariants["betti0"]
            total_sym += invariants["symmetry_residual"]
            all_colors.update(invariants["color_permutation_group"])
            total_pixels += int(np.sum(edge.to_array()[:, 0] > 0.5))
            grid_shapes.append(edge.get_grid_shape())

        hash_input = (
            f"betti0={total_betti0}|"
            f"sym={total_sym:.4f}|"
            f"colors={sorted(all_colors)}|"
            f"pixels={total_pixels}|"
            f"shapes={grid_shapes}|"
            f"temporal={self.temporal_length}"
        )
        self._topo_hash = hashlib.md5(hash_input.encode()).hexdigest()
        return self._topo_hash

    def is_isomorphic(self, other: HyperGraph) -> bool:
        """Check if this hypergraph is topologically isomorphic to another.

        Uses hash fingerprint as a fast necessary condition, then checks
        structural invariants for a stronger match.

        Args:
            other: Another HyperGraph to compare.

        Returns:
            True if the hypergraphs are likely isomorphic.
        """
        if self.get_topo_hash() != other.get_topo_hash():
            return False

        # Deeper check: compare invariant signatures
        self_inv = self._compute_full_invariants()
        other_inv = other._compute_full_invariants()
        return self_inv == other_inv

    def _compute_full_invariants(self) -> dict[str, Any]:
        """Compute full invariant signature for isomorphism checking.

        Returns:
            Dictionary of invariant values.
        """
        total_betti0 = sum(e.compute_betti0() for e in self.edges)
        total_sym = sum(e.compute_symmetry_residual() for e in self.edges) / max(len(self.edges), 1)
        all_colors: set[int] = set()
        for e in self.edges:
            inv = e.compute_topo_invariants()
            all_colors.update(inv["color_permutation_group"])
        return {
            "betti0": total_betti0,
            "symmetry": round(total_sym, 4),
            "colors": tuple(sorted(all_colors)),
            "edge_count": len(self.edges),
            "temporal_length": self.temporal_length,
        }

    def get_temporal_sequence(self) -> list[HyperGraph]:
        """Split a temporal hypergraph into per-frame sub-hypergraphs.

        Returns:
            List of HyperGraph objects, one per frame.
        """
        if not self.is_temporal:
            return [self]

        frame_groups: dict[int, list[OctonionHyperEdge]] = {}
        for edge in self.edges:
            frame_groups.setdefault(edge.frame_idx, []).append(edge)

        result: list[HyperGraph] = []
        for frame_idx in sorted(frame_groups.keys()):
            sub_hg = HyperGraph(frame_groups[frame_idx])
            sub_hg.is_temporal = False
            sub_hg.temporal_length = 1
            result.append(sub_hg)
        return result

    def compute_fiber(self, program: Any) -> set[int]:
        """Compute the Willems behavioral fiber for a given program.

        The fiber represents the set of grid states consistent with the
        program's behavior. Each element is a hash of a valid output state.

        Args:
            program: A DSLElement or ProgramNode to compute fiber for.

        Returns:
            Set of state hashes in the behavioral fiber.
        """
        fiber: set[int] = set()
        for edge in self.edges:
            try:
                grid = edge.decode_to_grid()
                transformed = program.apply(grid)
                state_hash = hash(transformed.tobytes())
                fiber.add(state_hash)
            except Exception:
                continue
        return fiber

    def compute_betti0(self) -> int:
        """Compute total Betti-0 (connected components) across all edges.

        Returns:
            Total number of connected components.
        """
        return sum(e.compute_betti0() for e in self.edges)

    def compute_color_permutation_group(self) -> list[int]:
        """Compute the color permutation group across all edges.

        Returns:
            Sorted list of unique color values.
        """
        all_colors: set[int] = set()
        for e in self.edges:
            inv = e.compute_topo_invariants()
            all_colors.update(inv["color_permutation_group"])
        return sorted(all_colors)

    def compute_symmetry_axis_residual(self) -> float:
        """Compute the mean symmetry axis residual.

        Returns:
            Mean symmetry residual value in [0, 1].
        """
        if not self.edges:
            return 0.0
        return sum(e.compute_symmetry_residual() for e in self.edges) / len(self.edges)
