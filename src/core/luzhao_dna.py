"""Luzhao DNA topological invariants: Fibonacci, Lucas, Bagua sequences.

Encodes hypergraph structure into Luzhao DNA triplets (Fibonacci/Lucas/Bagua)
as topological invariants for Phase A hash filtering enhancement.

Reference: TOMAS v2.0 GAT axiom system — Luzhao quantification theory.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LuzhaoDNA:
    """Computes Luzhao DNA topological invariants for hypergraphs.

    Three invariant sequences:
    - Fibonacci: F_0=0, F_1=1, F_2=1, F_3=2, F_4=3, F_5=5, ...
    - Lucas: L_0=2, L_1=1, L_2=3, L_3=4, L_4=7, L_5=11, ...
    - Bagua: 八卦常数映射 (8 trigram constants mapped to structural features)

    Attributes:
        fibonacci_offset: Starting offset for Fibonacci sequence.
        bagua_mapping: Dict mapping feature values to Bagua constants.
    """

    def __init__(self, fibonacci_offset: int = 0,
                 bagua_mapping: dict[int, int] | None = None) -> None:
        """Initialize Luzhao DNA calculator.

        Args:
            fibonacci_offset: Fibonacci sequence start offset.
            bagua_mapping: Custom Bagua constant mapping.
        """
        self.fibonacci_offset = fibonacci_offset
        self.bagua_mapping = bagua_mapping or {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
        }

        # Cache sequences
        self._fib_cache: list[int] = []
        self._lucas_cache: list[int] = []

    def fibonacci_sequence(self, n: int) -> list[int]:
        """Generate Fibonacci sequence up to F_n (0-indexed).

        F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}

        Args:
            n: Number of terms to generate.

        Returns:
            List of n Fibonacci numbers.
        """
        if n <= 0:
            return []

        if len(self._fib_cache) >= n:
            return self._fib_cache[:n]

        # Generate new terms
        if not self._fib_cache:
            self._fib_cache = [0, 1]

        while len(self._fib_cache) < n:
            self._fib_cache.append(
                self._fib_cache[-1] + self._fib_cache[-2]
            )
        return self._fib_cache[:n]

    def lucas_sequence(self, n: int) -> list[int]:
        """Generate Lucas sequence up to L_n (0-indexed).

        L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}

        Args:
            n: Number of terms to generate.

        Returns:
            List of n Lucas numbers.
        """
        if n <= 0:
            return []

        if len(self._lucas_cache) >= n:
            return self._lucas_cache[:n]

        if not self._lucas_cache:
            self._lucas_cache = [2, 1]

        while len(self._lucas_cache) < n:
            self._lucas_cache.append(
                self._lucas_cache[-1] + self._lucas_cache[-2]
            )
        return self._lucas_cache[:n]

    def bagua_map(self, values: list[int]) -> list[int]:
        """Map feature values to Bagua constants.

        Uses the Bagua (八卦) mapping to transform structural feature values
        into invariant constants. Values outside the mapping range are wrapped.

        Args:
            values: List of feature values.

        Returns:
            List of Bagua-mapped constants.
        """
        mapped: list[int] = []
        for v in values:
            key = v % len(self.bagua_mapping)
            mapped.append(self.bagua_mapping.get(key, v))
        return mapped

    def seq_dna_replication(self, hypergraph: Any) -> tuple[str, list[int], list[int], list[int]]:
        """Compute Luzhao DNA triplets for a hypergraph.

        Encodes the hypergraph's structural invariants into three Luzhao
        DNA sequences: Fibonacci encoding of Betti numbers, Lucas encoding
        of symmetry residuals, and Bagua encoding of color permutation groups.

        Args:
            hypergraph: HyperGraph object with compute_betti0(),
                compute_symmetry_axis_residual(), and
                compute_color_permutation_group() methods.

        Returns:
            Tuple of (dna_hash, fib_triplet, lucas_triplet, bagua_triplet).
        """
        betti0 = hypergraph.compute_betti0()
        sym_residual = hypergraph.compute_symmetry_axis_residual()
        colors = hypergraph.compute_color_permutation_group()

        # Fibonacci encoding: Betti-0 → Fibonacci terms
        fib_n = min(betti0 + self.fibonacci_offset + 3, 20)
        fib = self.fibonacci_sequence(fib_n)
        fib_triplet = fib[-3:] if len(fib) >= 3 else fib + [0] * (3 - len(fib))

        # Lucas encoding: symmetry residual → Lucas terms
        sym_scaled = max(1, min(20, int(sym_residual * 10) + 3))
        lucas = self.lucas_sequence(sym_scaled)
        lucas_triplet = lucas[-3:] if len(lucas) >= 3 else lucas + [0] * (3 - len(lucas))

        # Bagua encoding: color permutation group → Bagua constants
        bagua_triplet = self.bagua_map(colors[:3])
        while len(bagua_triplet) < 3:
            bagua_triplet.append(0)
        bagua_triplet = bagua_triplet[:3]

        # Combine into DNA hash
        dna_hash = f"LUZHAO:{fib_triplet}|{lucas_triplet}|{bagua_triplet}"
        return dna_hash, fib_triplet, lucas_triplet, bagua_triplet

    def compute_dna_hash(self, hypergraph: Any) -> str:
        """Compute Luzhao DNA hash string for Phase A filtering.

        Args:
            hypergraph: HyperGraph object.

        Returns:
            Hex-formatted DNA hash string.
        """
        dna_hash, _, _, _ = self.seq_dna_replication(hypergraph)
        import hashlib
        return hashlib.md5(dna_hash.encode()).hexdigest()

    def is_dna_match(self, hg1: Any, hg2: Any) -> bool:
        """Check if two hypergraphs have matching Luzhao DNA fingerprints.

        Args:
            hg1: First HyperGraph.
            hg2: Second HyperGraph.

        Returns:
            True if DNA fingerprints match.
        """
        _, f1, l1, b1 = self.seq_dna_replication(hg1)
        _, f2, l2, b2 = self.seq_dna_replication(hg2)
        return (tuple(f1), tuple(l1), tuple(b1)) == (tuple(f2), tuple(l2), tuple(b2))
