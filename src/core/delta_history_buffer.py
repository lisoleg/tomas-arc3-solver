"""Delta-T history buffer with pattern detection (predictive coding).

TOMAS v2.0 upgrade: Window=5, linear prediction, DeltaPattern output,
mutation threshold control. Inspired by Nature single-neuron research —
context predictive coding with 5-frame temporal receptive field.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode


@dataclass
class DeltaPattern:
    """Detected delta-T pattern with confidence score.

    Attributes:
        pattern_type: 'arithmetic', 'periodic', 'mutation', or 'none'.
        confidence: Detection confidence in [0, 1].
        period: Detected period (0 if not periodic).
        predicted_next: Predicted next ProgramNode (None if unavailable).
        prediction_weight: Confidence weight for prediction (neural analogy).
    """
    pattern_type: str = "none"
    confidence: float = 0.0
    period: int = 0
    predicted_next: ProgramNode | None = None
    prediction_weight: float = 0.6
    metadata: dict[str, Any] = field(default_factory=dict)


class DeltaHistoryBuffer:
    """Buffer for delta-T history with pattern detection and prediction.

    Maintains a sliding window of delta-T transformations and detects
    patterns: arithmetic (uniform), periodic (repeating), or mutation
    (sudden Betti₀ drop). Neural analogy: 5-frame temporal receptive field
    with context predictive coding (Nature single-neuron study).

    Attributes:
        buffer: Deque of recent ProgramNode deltas.
        window_size: Maximum number of deltas to retain (default 5).
        mutation_threshold: Betti₀ drop ratio to trigger mutation detection.
    """

    def __init__(self, window_size: int = 5,
                 mutation_threshold: float = 0.5,
                 prediction_weight: float = 0.6) -> None:
        """Initialize the buffer.

        Args:
            window_size: Maximum number of delta-T entries to keep (default 5).
            mutation_threshold: Betti₀ drop ratio threshold for mutation.
            prediction_weight: Confidence weight for linear prediction.
        """
        self.window_size = window_size
        self.mutation_threshold = mutation_threshold
        self.prediction_weight = prediction_weight
        self.buffer: deque[ProgramNode] = deque(maxlen=window_size)
        # Track frame indices for Betti₀ comparison
        self._frame_indices: deque[int] = deque(maxlen=window_size)
        self._betti_history: deque[int] = deque(maxlen=window_size)

    def push(self, deltaT: ProgramNode) -> None:
        """Push a new delta-T onto the buffer.

        Args:
            deltaT: The delta-T ProgramNode to add.
        """
        self.buffer.append(deltaT)

    def push_with_frame(self, deltaT: ProgramNode, frame_idx: int,
                        betti0: int = 0) -> None:
        """Push a delta-T with frame index and Betti-0 for mutation detection.

        Args:
            deltaT: The delta-T ProgramNode.
            frame_idx: Temporal frame index.
            betti0: Betti-0 (connected component count) for this frame.
        """
        self.buffer.append(deltaT)
        self._frame_indices.append(frame_idx)
        self._betti_history.append(betti0)

    def detect_pattern(self) -> str:
        """Detect the dominant pattern in the delta-T history.

        Returns:
            One of 'arithmetic', 'periodic', 'mutation', or 'none'.
        """
        if len(self.buffer) < 2:
            return "none"

        # Check Betti₀-based mutation first (higher priority)
        if self._detect_betti_mutation():
            return "mutation"

        if self.detect_arithmetic():
            return "arithmetic"
        if self.detect_periodic():
            return "periodic"
        if self.detect_mutation():
            return "mutation"
        return "none"

    def detect_pattern_full(self) -> DeltaPattern:
        """Detect pattern with full confidence scoring and prediction.

        Returns:
            DeltaPattern with type, confidence, period, and prediction.
        """
        pattern_type = self.detect_pattern()

        if pattern_type == "arithmetic":
            confidence = self._arithmetic_confidence()
            period = 1
        elif pattern_type == "periodic":
            period = self.get_period()
            confidence = self._periodic_confidence(period)
        elif pattern_type == "mutation":
            confidence = self._mutation_confidence()
            period = 0
        else:
            confidence = 0.0
            period = 0

        predicted = self.predict_next_delta() if pattern_type != "none" else None

        return DeltaPattern(
            pattern_type=pattern_type,
            confidence=confidence,
            period=period,
            predicted_next=predicted,
            prediction_weight=self.prediction_weight,
            metadata={"buffer_length": len(self.buffer)},
        )

    def predict_next_delta(self) -> ProgramNode | None:
        """Predict the next delta-T using linear extrapolation.

        Neural analogy: context predictive coding — the expected next
        delta is the linear extrapolation from the previous 5-frame
        temporal receptive field.

        Returns:
            Predicted ProgramNode, or None if insufficient history.
        """
        if len(self.buffer) < 2:
            return None

        deltas = list(self.buffer)

        # For arithmetic (uniform) pattern: return the last delta
        if self.detect_arithmetic():
            return deltas[-1]

        # For periodic: predict next in cycle
        if self.detect_periodic():
            period = self.get_period()
            if period > 0:
                next_idx = len(deltas) % period
                return deltas[next_idx]
            return deltas[-1]

        # For mutation or unknown: return last delta (conservative)
        return deltas[-1]

    def _detect_betti_mutation(self) -> bool:
        """Detect Betti₀ sudden drop (singularity) across frames.

        Returns True if the latest Betti₀ is below threshold * previous Betti₀.
        """
        if len(self._betti_history) < 2:
            return False
        current = self._betti_history[-1]
        previous = self._betti_history[-2]
        if previous == 0:
            return False
        return float(current) / float(previous) < self.mutation_threshold

    def _arithmetic_confidence(self) -> float:
        """Compute confidence for arithmetic pattern.

        Returns:
            Confidence in [0, 1].
        """
        if len(self.buffer) < 2:
            return 0.0
        deltas = list(self.buffer)
        first_elements = deltas[0].flatten()
        matches = 0
        for d in deltas[1:]:
            elements = d.flatten()
            if len(elements) == len(first_elements):
                if all(e1.name == e2.name and e1.params == e2.params
                       for e1, e2 in zip(first_elements, elements)):
                    matches += 1
        return matches / max(len(deltas) - 1, 1)

    def _periodic_confidence(self, period: int) -> float:
        """Compute confidence for periodic pattern.

        Args:
            period: Detected period.

        Returns:
            Confidence in [0, 1].
        """
        if period == 0 or len(self.buffer) < period * 2:
            return 0.0
        deltas = list(self.buffer)
        matches = 0
        total = 0
        for i in range(period, len(deltas)):
            ref_idx = i % period
            ref_elems = deltas[ref_idx].flatten()
            cur_elems = deltas[i].flatten()
            total += 1
            if len(ref_elems) == len(cur_elems):
                if all(e1.name == e2.name and e1.params == e2.params
                       for e1, e2 in zip(ref_elems, cur_elems)):
                    matches += 1
        return matches / max(total, 1)

    def _mutation_confidence(self) -> float:
        """Compute confidence that the last delta is a true mutation.

        Returns:
            Confidence in [0, 1].
        """
        if len(self.buffer) < 3:
            return 0.5
        # Higher confidence if previous pattern was strongly established
        prev_pattern = self.detect_arithmetic()
        if prev_pattern:
            return 0.8
        return 0.4

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
