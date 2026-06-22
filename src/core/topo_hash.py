"""Topological hash quick-filter for Phase A of Two-Phase search.

TOMAS v2.0 upgrade: integrated Luzhao DNA (Fibonacci/Lucas/Bagua) hash
for enhanced filtering precision.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.hypergraph import HyperGraph
from src.core.octonion_hyperedge import OctonionHyperEdge
from src.core.luzhao_dna import LuzhaoDNA


class TopoHashFilter:
    """Phase A quick filter using topological invariant hashing.

    Filters candidate programs by checking the necessary condition:
    ``topo_hash(P(I_i)) == topo_hash(O_i)`` for each demo pair.
    Optionally integrates Luzhao DNA hash for enhanced precision.
    This provides O(1) lookup per candidate after hash computation.
    """

    def __init__(self, cache_size: int = 10000,
                 use_luzhao: bool = True) -> None:
        """Initialize the filter.

        Args:
            cache_size: Maximum number of cached hash entries.
            use_luzhao: Enable Luzhao DNA hash integration.
        """
        self.hash_cache: dict[str, str] = {}
        self.luzhao_cache: dict[str, str] = {}
        self.cache_size = cache_size
        self.use_luzhao = use_luzhao
        self.luzhao_dna = LuzhaoDNA() if use_luzhao else None

    def compute_hash(self, graph: HyperGraph,
                     use_luzhao: bool | None = None) -> str:
        """Compute the topological hash of a hypergraph.

        Optionally integrates Luzhao DNA hash (Fibonacci/Lucas/Bagua)
        for enhanced topological discrimination.

        Args:
            graph: HyperGraph to hash.
            use_luzhao: Override use_luzhao flag (None uses instance default).

        Returns:
            Hex string hash.
        """
        should_use_luzhao = use_luzhao if use_luzhao is not None else self.use_luzhao

        # Try cache first using object id
        cache_key = f"hg_{id(graph)}_{should_use_luzhao}"
        if cache_key in self.hash_cache:
            return self.hash_cache[cache_key]

        base_hash = graph.get_topo_hash()

        if should_use_luzhao and self.luzhao_dna is not None:
            luzhao_hash = self.luzhao_dna.compute_dna_hash(graph)
            result = f"{base_hash}:{luzhao_hash[:16]}"
        else:
            result = base_hash

        # Cache management
        if len(self.hash_cache) >= self.cache_size:
            keys = list(self.hash_cache.keys())
            for k in keys[: len(keys) // 2]:
                del self.hash_cache[k]

        self.hash_cache[cache_key] = result
        return result

    def compute_hash_for_grid(self, grid: np.ndarray) -> str:
        """Compute topological hash directly from a grid.

        Args:
            grid: Input grid as ndarray.

        Returns:
            Hex string hash.
        """
        edge = OctonionHyperEdge(grid)
        hg = HyperGraph([edge])
        return self.compute_hash(hg)

    def quick_filter(
        self,
        candidates: list[Any],
        demo_pairs: list[dict[str, Any]],
    ) -> list[Any]:
        """Filter candidates by topological hash necessary condition.

        For each candidate program P, checks if topo_hash(P(I_i)) matches
        topo_hash(O_i) for all demo pairs. Candidates that fail this
        necessary condition are filtered out.

        Args:
            candidates: List of ProgramNode candidates.
            demo_pairs: List of demo pairs with 'input' and 'output' grids.

        Returns:
            Filtered list of candidates that pass the hash check.
        """
        if not demo_pairs:
            return list(candidates)

        # Pre-compute output hashes
        output_hashes: list[str] = []
        for pair in demo_pairs:
            output_grids = pair.get("output", [])
            if output_grids:
                output_hashes.append(self.compute_hash_for_grid(output_grids[0]))

        if not output_hashes:
            return list(candidates)

        filtered: list[Any] = []
        for candidate in candidates:
            passes = True
            for i, pair in enumerate(demo_pairs):
                input_grids = pair.get("input", [])
                if not input_grids:
                    continue
                input_grid = input_grids[0]
                try:
                    predicted = candidate.apply(input_grid)
                    pred_hash = self.compute_hash_for_grid(predicted)
                    if pred_hash != output_hashes[i]:
                        passes = False
                        break
                except Exception:
                    passes = False
                    break
            if passes:
                filtered.append(candidate)

        return filtered

    def batch_hash(self, graphs: list[HyperGraph]) -> list[str]:
        """Compute hashes for a batch of hypergraphs.

        Args:
            graphs: List of HyperGraph objects.

        Returns:
            List of hash strings.
        """
        return [self.compute_hash(g) for g in graphs]

    def batch_hash_grids(self, grids: list[np.ndarray]) -> list[str]:
        """Compute hashes for a batch of grids.

        Args:
            grids: List of grid ndarrays.

        Returns:
            List of hash strings.
        """
        return [self.compute_hash_for_grid(g) for g in grids]

    def clear_cache(self) -> None:
        """Clear the hash cache."""
        self.hash_cache.clear()
