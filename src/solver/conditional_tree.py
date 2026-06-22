"""Conditional delta-T decision tree induction."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.core.hypergraph import HyperGraph
from src.core.delta_history_buffer import DeltaHistoryBuffer


class ConditionalTree:
    """A decision tree for conditional delta-T application.

    Attributes:
        conditions: List of condition descriptors.
        branches: Mapping from condition to ProgramNode.
    """

    def __init__(self) -> None:
        """Initialize the conditional tree."""
        self.conditions: list[dict[str, Any]] = []
        self.branches: dict[str, ProgramNode] = {}


class ConditionalTreeInducer:
    """Induces conditional if-then rules from delta-T history.

    Automatically discovers conditions such as boundary hitting,
    color appearance, and frame index thresholds that determine
    which delta-T to apply.

    Attributes:
        conditions: List of discovered conditions.
        branches: Mapping from condition to ProgramNode.
        condition_types: Supported condition type names.
    """

    def __init__(self) -> None:
        """Initialize the inducer."""
        self.conditions: list[dict[str, Any]] = []
        self.branches: dict[str, ProgramNode] = {}
        self.condition_types: list[str] = [
            "boundary_hit",
            "color_appeared",
            "frame_idx_threshold",
        ]

    def induce_conditions(
        self,
        deltaT_history: list[ProgramNode] | DeltaHistoryBuffer,
    ) -> ConditionalTree:
        """Induce if-then conditions from delta-T history.

        Analyzes the history to find conditions that explain when
        different delta-T transformations are applied.

        Args:
            deltaT_history: List of ProgramNodes or DeltaHistoryBuffer.

        Returns:
            A ConditionalTree with induced rules.
        """
        tree = ConditionalTree()

        if isinstance(deltaT_history, DeltaHistoryBuffer):
            history = deltaT_history.get_context()
        else:
            history = list(deltaT_history)

        if len(history) < 2:
            return tree

        # Group deltas by their structure
        delta_groups: dict[str, list[tuple[int, ProgramNode]]] = {}
        for idx, delta in enumerate(history):
            elements = delta.flatten()
            key = "|".join(e.name for e in elements)
            delta_groups.setdefault(key, []).append((idx, delta))

        # If there are multiple groups, induce conditions
        if len(delta_groups) > 1:
            for group_key, items in delta_groups.items():
                # Determine condition based on frame indices
                indices = [idx for idx, _ in items]
                if indices:
                    # Check if this group occurs at specific frame intervals
                    if len(indices) >= 2:
                        diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
                        if all(d == diffs[0] for d in diffs):
                            # Periodic condition
                            condition = {
                                "type": "frame_idx_threshold",
                                "period": diffs[0],
                                "offset": indices[0],
                            }
                        else:
                            condition = {
                                "type": "frame_idx_threshold",
                                "indices": indices,
                            }
                    else:
                        condition = {
                            "type": "frame_idx_threshold",
                            "indices": indices,
                        }
                    tree.conditions.append(condition)
                    tree.branches[group_key] = items[0][1]
                    self.conditions.append(condition)
                    self.branches[group_key] = items[0][1]

        return tree

    def apply(
        self,
        frame: HyperGraph | np.ndarray,
        frame_idx: int,
    ) -> ProgramNode:
        """Apply the conditional tree to select a delta-T for a frame.

        Args:
            frame: Current frame as HyperGraph or ndarray.
            frame_idx: Current frame index.

        Returns:
            The selected ProgramNode for this frame.
        """
        # Check conditions in order
        for condition in self.conditions:
            cond_type = condition.get("type", "")

            if cond_type == "frame_idx_threshold":
                period = condition.get("period", 0)
                offset = condition.get("offset", 0)
                if period > 0 and (frame_idx - offset) % period == 0:
                    # Find matching branch
                    for key, delta in self.branches.items():
                        return delta

            if cond_type == "boundary_hit":
                if self.detect_boundary_hit(frame):
                    for key, delta in self.branches.items():
                        return delta

            if cond_type == "color_appeared":
                color = condition.get("color", 0)
                if self.detect_color_appeared(frame, color):
                    for key, delta in self.branches.items():
                        return delta

        # Default: return first branch or identity
        if self.branches:
            return next(iter(self.branches.values()))
        return ProgramNode(DSLElement("copy"))

    def detect_boundary_hit(self, frame: HyperGraph | np.ndarray) -> bool:
        """Detect if any non-zero pixel touches the grid boundary.

        Args:
            frame: Frame as HyperGraph or ndarray.

        Returns:
            True if boundary is hit by non-zero pixels.
        """
        if isinstance(frame, HyperGraph):
            for edge in frame.edges:
                arr = edge.to_array()
                boundary_pixels = arr[arr[:, 6] > 0.5]
                if len(boundary_pixels) > 0:
                    return True
            return False

        grid = np.asarray(frame)
        if grid.ndim != 2:
            return False
        # Check border pixels
        if np.any(grid[0, :] > 0) or np.any(grid[-1, :] > 0):
            return True
        if np.any(grid[:, 0] > 0) or np.any(grid[:, -1] > 0):
            return True
        return False

    def detect_color_appeared(self, frame: HyperGraph | np.ndarray, color: int) -> bool:
        """Detect if a specific color appears in the frame.

        Args:
            frame: Frame as HyperGraph or ndarray.
            color: Color value to check for.

        Returns:
            True if the color is present.
        """
        if isinstance(frame, HyperGraph):
            colors = frame.compute_color_permutation_group()
            return color in colors

        grid = np.asarray(frame)
        return bool(np.any(grid == color))
