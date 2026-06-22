"""Delta-T history buffer with pattern detection (predictive coding)."""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode


class DeltaHistoryBuffer:
    """Buffer for delta-T history with pattern detection.

    Maintains a sliding window of delta-T transformations and detects
    patterns: arithmetic (uniform), periodic (repeating), or mutation
    (sudden change). Inspired by neural context predictive coding.

    Attributes:
        buffer: Deque of recent ProgramNode deltas.
        window_size: Maximum number of deltas to retain.
    """

    def __init__(self, window_size: int = 10) -> None:
        """Initialize the buffer.

        Args:
            window_size: Maximum number of delta-T entries to keep.
        """
        self.window_size = window_size
        self.buffer: deque[ProgramNode] = deque(maxlen=window_size)

    def push(self, deltaT: ProgramNode) -> None:
        """Push a new delta-T onto the buffer.

        Args:
            deltaT: The delta-T ProgramNode to add.
        """
        self.buffer.append(deltaT)

    def detect_pattern(self) -> str:
        """Detect the dominant pattern in the delta-T history.

        Returns:
            One of 'arithmetic', 'periodic', 'mutation', or 'none'.
        """
        if len(self.buffer) < 2:
            return "none"

        if self.detect_arithmetic():
            return "arithmetic"
        if self.detect_periodic():
            return "periodic"
        if self.detect_mutation():
            return "mutation"
        return "none"

    def detect_arithmetic(self) -> bool:
        """Detect if deltas follow an arithmetic (uniform) pattern.

        Checks if all deltas in the buffer are the same transformation,
        indicating uniform/constant motion.

        Returns:
            True if arithmetic pattern detected.
        """
        if len(self.buffer) < 2:
            return False

        # Check if all deltas have the same structure
        first = self.buffer[0]
        first_elements = first.flatten()

        for delta in list(self.buffer)[1:]:
            elements = delta.flatten()
            if len(elements) != len(first_elements):
                return False
            # Compare element names and params
            for e1, e2 in zip(first_elements, elements):
                if e1.name != e2.name or e1.params != e2.params:
                    return False
        return True

    def detect_periodic(self) -> bool:
        """Detect if deltas follow a periodic (repeating) pattern.

        Checks for repeating sub-sequences of deltas.

        Returns:
            True if periodic pattern detected.
        """
        if len(self.buffer) < 4:
            return False

        deltas = list(self.buffer)
        # Check for period lengths from 2 to len/2
        for period in range(2, len(deltas) // 2 + 1):
            is_periodic = True
            for i in range(len(deltas)):
                ref_idx = i % period
                if i >= period:
                    elem_i = deltas[i].flatten()
                    elem_ref = deltas[ref_idx].flatten()
                    if len(elem_i) != len(elem_ref):
                        is_periodic = False
                        break
                    for e1, e2 in zip(elem_i, elem_ref):
                        if e1.name != e2.name or e1.params != e2.params:
                            is_periodic = False
                            break
                    if not is_periodic:
                        break
            if is_periodic:
                return True
        return False

    def detect_mutation(self) -> bool:
        """Detect if there's a sudden mutation (pattern break) in history.

        Checks if the most recent delta differs significantly from
        the preceding pattern.

        Returns:
            True if a mutation is detected.
        """
        if len(self.buffer) < 3:
            return False

        deltas = list(self.buffer)
        # Check if last delta differs from the pattern of previous deltas
        last = deltas[-1]
        last_elements = last.flatten()

        # Check if previous deltas were uniform
        prev_uniform = True
        first_prev = deltas[0].flatten()
        for d in deltas[1:-1]:
            elements = d.flatten()
            if len(elements) != len(first_prev):
                prev_uniform = False
                break
            for e1, e2 in zip(first_prev, elements):
                if e1.name != e2.name or e1.params != e2.params:
                    prev_uniform = False
                    break
            if not prev_uniform:
                break

        if not prev_uniform:
            return False

        # Check if last delta breaks the pattern
        if len(last_elements) != len(first_prev):
            return True
        for e1, e2 in zip(first_prev, last_elements):
            if e1.name != e2.name or e1.params != e2.params:
                return True

        return False

    def get_context(self) -> list[ProgramNode]:
        """Get the current context (all buffered deltas).

        Returns:
            List of ProgramNode objects in the buffer.
        """
        return list(self.buffer)

    def get_period(self) -> int:
        """Get the detected period length if periodic.

        Returns:
            Period length, or 0 if not periodic.
        """
        if len(self.buffer) < 4:
            return 0

        deltas = list(self.buffer)
        for period in range(2, len(deltas) // 2 + 1):
            is_periodic = True
            for i in range(len(deltas)):
                ref_idx = i % period
                if i >= period:
                    elem_i = deltas[i].flatten()
                    elem_ref = deltas[ref_idx].flatten()
                    if len(elem_i) != len(elem_ref):
                        is_periodic = False
                        break
                    for e1, e2 in zip(elem_i, elem_ref):
                        if e1.name != e2.name or e1.params != e2.params:
                            is_periodic = False
                            break
                    if not is_periodic:
                        break
            if is_periodic:
                return period
        return 0

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
