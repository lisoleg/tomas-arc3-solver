"""Slip cost model: phase alignment fee for program → hypergraph application.

Models the "slippage" (滑点) cost when applying a program to a hypergraph,
analogous to financial market order book slippage. The cost is computed from
the e7 (time phase) component offset induced by the program's transformation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SlipCostCalculator:
    """Calculates phase alignment (slippage) cost for program applications.

    In the TOMAS framework, each program application shifts the octonion
    hyperedge's e7 time-phase component. This shift represents a "slippage"
    analogous to financial order book impact. Programs with larger phase
    shifts incur higher costs, penalizing overly complex transformations.

    Attributes:
        base_slip_rate: Base slippage cost per unit phase shift.
        max_slip_cost: Maximum slippage cost cap.
        use_phase_cost: Whether to include phase alignment in MDL.
    """

    def __init__(self, base_slip_rate: float = 0.5,
                 max_slip_cost: float = 20.0,
                 use_phase_cost: bool = True) -> None:
        """Initialize the slip cost calculator.

        Args:
            base_slip_rate: Cost multiplier per unit phase shift.
            max_slip_cost: Upper bound on slip cost.
            use_phase_cost: Enable/disable phase alignment cost.
        """
        self.base_slip_rate = base_slip_rate
        self.max_slip_cost = max_slip_cost
        self.use_phase_cost = use_phase_cost

    def phase_alignment_cost(self, program: Any,
                             hypergraph: Any) -> float:
        """Compute the phase alignment cost for a program on a hypergraph.

        Cost = base_slip_rate * Sum(|e7_before - e7_after| for all edges).

        The e7 component encodes temporal phase. Programs that cause large
        phase shifts (complex transformations) incur higher costs.

        Args:
            program: ProgramNode or DSLElement to apply.
            hypergraph: HyperGraph containing octonion hyperedges.

        Returns:
            Phase alignment cost (non-negative float).
        """
        if not self.use_phase_cost:
            return 0.0

        if hypergraph is None or not hypergraph.edges:
            return 0.0

        total_shift = 0.0
        edge_count = len(hypergraph.edges)

        for edge in hypergraph.edges:
            try:
                # Get e7 before transformation
                e7_before = edge.e7_area_ratio

                # Apply program to get transformed edge
                grid = edge.decode_to_grid()
                transformed_grid = program.apply(grid)

                # Encode transformed grid to get new e7
                from src.core.octonion_hyperedge import OctonionHyperEdge
                transformed_edge = OctonionHyperEdge(transformed_grid,
                                                     frame_idx=edge.frame_idx)
                e7_after = transformed_edge.e7_area_ratio

                # Phase shift
                shift = abs(e7_after - e7_before)
                total_shift += shift
            except Exception:
                # On error, assign maximum penalty
                total_shift += 1.0

        cost = self.base_slip_rate * total_shift / max(edge_count, 1)
        return min(cost, self.max_slip_cost)

    def estimate_batch_cost(self, programs: list[Any],
                            hypergraphs: list[Any]) -> list[float]:
        """Estimate phase alignment costs for a batch of programs.

        Args:
            programs: List of program candidates.
            hypergraphs: List of hypergraphs (demo inputs).

        Returns:
            List of cost values, one per program.
        """
        costs: list[float] = []
        for program in programs:
            total_cost = 0.0
            for hg in hypergraphs:
                total_cost += self.phase_alignment_cost(program, hg)
            avg_cost = total_cost / max(len(hypergraphs), 1)
            costs.append(avg_cost)
        return costs
