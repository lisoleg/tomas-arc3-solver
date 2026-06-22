"""Frame-to-frame pattern induction: uniform motion, rotation, color cycles."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.core.delta_history_buffer import DeltaHistoryBuffer


class SequenceDSL:
    """Induces frame-to-frame temporal patterns from delta-T history.

    Detects composite temporal patterns such as:
    - Uniform motion (constant velocity translation)
    - Rotation cycles (periodic rotation)
    - Color cycles (periodic color changes)
    - Compound patterns (e.g., uniform 3 frames → rotation → uniform)

    Attributes:
        history_buffer: DeltaHistoryBuffer for pattern context.
    """

    def __init__(self, history_buffer: DeltaHistoryBuffer | None = None) -> None:
        """Initialize the sequence DSL inducer.

        Args:
            history_buffer: Optional DeltaHistoryBuffer for context.
        """
        self.history_buffer = history_buffer or DeltaHistoryBuffer(window_size=20)

    def induce_pattern(self, deltaT_list: list[ProgramNode]) -> ProgramNode:
        """Induce the dominant temporal pattern from a delta-T sequence.

        Args:
            deltaT_list: List of delta-T ProgramNodes.

        Returns:
            A ProgramNode representing the induced temporal pattern.
        """
        if not deltaT_list:
            return ProgramNode(DSLElement("copy"))

        # Push all deltas to history buffer
        for delta in deltaT_list:
            self.history_buffer.push(delta)

        pattern = self.history_buffer.detect_pattern()

        if pattern == "arithmetic":
            return self._induce_uniform_motion(deltaT_list)
        elif pattern == "periodic":
            return self._induce_periodic_pattern(deltaT_list)
        elif pattern == "mutation":
            return self._induce_compound_pattern(deltaT_list)
        else:
            # Default: chain all deltas
            if len(deltaT_list) == 1:
                return deltaT_list[0]
            root = deltaT_list[0]
            for delta in deltaT_list[1:]:
                current = root
                while current.children:
                    current = current.children[0]
                current.children.append(delta)
                current.combo_type = "chain"
            root.combo_type = "chain"
            root.total_mdl = root.compute_mdl()
            return root

    def _induce_uniform_motion(self, deltaT_list: list[ProgramNode]) -> ProgramNode:
        """Induce a uniform motion pattern.

        All deltas are the same, so we return a single instance.

        Args:
            deltaT_list: List of uniform delta-T ProgramNodes.

        Returns:
            ProgramNode representing the uniform motion.
        """
        # Return the first delta (they're all the same)
        return deltaT_list[0]

    def _induce_periodic_pattern(self, deltaT_list: list[ProgramNode]) -> ProgramNode:
        """Induce a periodic pattern (e.g., rotation cycle).

        Args:
            deltaT_list: List of periodic delta-T ProgramNodes.

        Returns:
            ProgramNode representing one period of the pattern.
        """
        period = self.history_buffer.get_period()
        if period == 0:
            period = 2  # Default period

        # Return the deltas for one period as a chain
        period_deltas = deltaT_list[:period]
        if len(period_deltas) == 1:
            return period_deltas[0]

        root = period_deltas[0]
        for delta in period_deltas[1:]:
            current = root
            while current.children:
                current = current.children[0]
            current.children.append(delta)
            current.combo_type = "chain"
        root.combo_type = "chain"
        root.total_mdl = root.compute_mdl()
        return root

    def _induce_compound_pattern(self, deltaT_list: list[ProgramNode]) -> ProgramNode:
        """Induce a compound pattern (e.g., uniform → rotation → uniform).

        Args:
            deltaT_list: List of delta-T ProgramNodes with mutation.

        Returns:
            ProgramNode representing the compound pattern.
        """
        # Find the mutation point
        mutation_idx = len(deltaT_list) - 1
        for i in range(len(deltaT_list) - 1, 0, -1):
            elements_i = deltaT_list[i].flatten()
            elements_prev = deltaT_list[i - 1].flatten()
            if len(elements_i) != len(elements_prev):
                mutation_idx = i
                break
            for e1, e2 in zip(elements_i, elements_prev):
                if e1.name != e2.name or e1.params != e2.params:
                    mutation_idx = i
                    break
            else:
                continue
            break

        # Build conditional composition
        root = ProgramNode(deltaT_list[0].element)
        root.combo_type = "conditional"
        for delta in deltaT_list[1:]:
            root.children.append(delta)
        root.total_mdl = root.compute_mdl()
        return root

    def detect_motion_type(self, deltaT_list: list[ProgramNode]) -> str:
        """Detect the type of motion from delta-T sequence.

        Args:
            deltaT_list: List of delta-T ProgramNodes.

        Returns:
            Motion type string: 'translation', 'rotation', 'color_cycle',
            'scale', 'compound', or 'unknown'.
        """
        if not deltaT_list:
            return "unknown"

        # Analyze first delta
        elements = deltaT_list[0].flatten()
        element_names = [e.name for e in elements]

        if "move" in element_names:
            return "translation"
        if "rotate" in element_names:
            return "rotation"
        if "color-swap" in element_names or "map-color" in element_names:
            return "color_cycle"
        if "scale" in element_names:
            return "scale"
        if len(deltaT_list) > 1:
            pattern = self.history_buffer.detect_pattern()
            if pattern == "mutation":
                return "compound"

        return "unknown"

    def predict_next_delta(self, deltaT_list: list[ProgramNode]) -> ProgramNode:
        """Predict the next delta-T based on detected patterns.

        Args:
            deltaT_list: History of delta-T ProgramNodes.

        Returns:
            Predicted next ProgramNode.
        """
        if not deltaT_list:
            return ProgramNode(DSLElement("copy"))

        pattern = self.history_buffer.detect_pattern()

        if pattern == "periodic":
            period = self.history_buffer.get_period()
            if period > 0 and len(deltaT_list) >= period:
                next_idx = len(deltaT_list) % period
                return deltaT_list[next_idx]

        if pattern == "arithmetic":
            return deltaT_list[-1]

        # Default: return last delta
        return deltaT_list[-1]
