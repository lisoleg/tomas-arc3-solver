"""Delta-T composition tree: chain, additive, and conditional composition."""
from __future__ import annotations

import copy
from typing import Any

from src.core.dsl_primitives import DSLElement, ProgramNode


class DeltaTCombinator:
    """Composes delta-T transformations into program trees.

    Supports three composition strategies:
    - Chain (⊙): Sequential application of transformations.
    - Additive (+): Parallel application with overlay combination.
    - Conditional (if-then): Branch based on frame state.

    Attributes:
        delta_list: List of delta-T ProgramNodes to compose.
        combo_type: Current composition type.
    """

    def __init__(self, deltas: list[ProgramNode] | None = None) -> None:
        """Initialize the combinator.

        Args:
            deltas: List of delta-T ProgramNodes.
        """
        self.delta_list: list[ProgramNode] = deltas or []
        self.combo_type: str = "chain"

    def chain_compose(self) -> ProgramNode:
        """Chain-compose all deltas: delta_1 ⊙ delta_2 ⊙ ... ⊙ delta_n.

        Each delta is applied sequentially to the output of the previous.
        Uses deepcopy to avoid mutating the original delta objects.

        Returns:
            A ProgramNode representing the chained composition.
        """
        if not self.delta_list:
            return ProgramNode(DSLElement("copy"))

        # Deep copy all deltas to avoid in-place mutation of originals
        deltas = [copy.deepcopy(d) for d in self.delta_list]

        # Build a chain: each node's first child is the next delta
        root = deltas[0]
        for delta in deltas[1:]:
            current = root
            while current.children:
                current = current.children[0]
            current.children.append(delta)
            current.combo_type = "chain"
        root.combo_type = "chain"
        root.total_mdl = root.compute_mdl()
        return root

    def additive_compose(self) -> ProgramNode:
        """Additive-compose all deltas: delta_1 + delta_2 + ... + delta_n.

        All deltas are applied to the same input and results are overlaid.
        Uses deepcopy to avoid mutating the original delta objects.

        Returns:
            A ProgramNode representing the additive composition.
        """
        if not self.delta_list:
            return ProgramNode(DSLElement("copy"))

        # Deep copy to avoid mutating originals
        root = ProgramNode(copy.deepcopy(self.delta_list[0].element))
        root.combo_type = "additive"
        for delta in self.delta_list[1:]:
            root.children.append(copy.deepcopy(delta))
        root.total_mdl = root.compute_mdl()
        return root

    def conditional_compose(self, tree: Any) -> ProgramNode:
        """Conditionally compose deltas using a decision tree.

        Uses a ConditionalTreeInducer to determine which delta to apply
        based on frame state.
        Uses deepcopy to avoid mutating the original delta objects.

        Args:
            tree: ConditionalTreeInducer instance.

        Returns:
            A ProgramNode representing the conditional composition.
        """
        if not self.delta_list:
            return ProgramNode(DSLElement("copy"))

        # Deep copy to avoid mutating originals
        root = ProgramNode(copy.deepcopy(self.delta_list[0].element))
        root.combo_type = "conditional"

        # Add all deltas as branches (deep copies)
        for delta in self.delta_list[1:]:
            root.children.append(copy.deepcopy(delta))

        root.total_mdl = root.compute_mdl()
        return root

    def search_combinations(self, max_depth: int = 3) -> list[ProgramNode]:
        """Search over all composition combinations up to max_depth.

        Generates chain, additive, and conditional compositions and
        ranks them by MDL.

        Args:
            max_depth: Maximum number of deltas to combine.

        Returns:
            List of composed ProgramNodes ranked by MDL.
        """
        results: list[ProgramNode] = []

        if not self.delta_list:
            return results

        # Limit to max_depth deltas
        deltas = self.delta_list[:max_depth]

        # Chain composition
        self.delta_list = deltas
        self.combo_type = "chain"
        chain_result = self.chain_compose()
        results.append(chain_result)

        # Additive composition
        self.combo_type = "additive"
        additive_result = self.additive_compose()
        results.append(additive_result)

        # Pairwise compositions
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                pair_combinator = DeltaTCombinator([deltas[i], deltas[j]])
                results.append(pair_combinator.chain_compose())

        # Sort by MDL
        results.sort(key=lambda p: p.total_mdl)
        return results

    def add_delta(self, delta: ProgramNode) -> None:
        """Add a delta-T to the combinator.

        Args:
            delta: ProgramNode delta to add.
        """
        self.delta_list.append(delta)

    def clear(self) -> None:
        """Clear all deltas."""
        self.delta_list = []
