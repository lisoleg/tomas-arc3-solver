"""Video frame tensor operations: encoding, delta-T extraction, prediction."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.hypergraph import HyperGraph
from src.core.octonion_hyperedge import OctonionHyperEdge
from src.core.dsl_primitives import DSLElement, ProgramNode


class VideoTemporalEncoder:
    """Encodes video frame sequences into temporal hypergraph sequences.

    Converts frame sequences to temporal hypergraphs, extracts inter-frame
    delta-T transformations, and predicts next frames.

    Attributes:
        frame_sequence: List of frame arrays (T, H, W).
        temporal_hypergraphs: List of per-frame HyperGraph objects.
    """

    def __init__(self, frames: list[np.ndarray] | None = None) -> None:
        """Initialize the encoder.

        Args:
            frames: List of frame arrays.
        """
        self.frame_sequence: list[np.ndarray] = frames or []
        self.temporal_hypergraphs: list[HyperGraph] = []

    def encode_frames(self) -> list[HyperGraph]:
        """Encode all frames into temporal hypergraphs.

        Each frame is encoded as an OctonionHyperEdge, and all edges for
        a frame are grouped into a HyperGraph.

        Returns:
            List of HyperGraph objects, one per frame.
        """
        self.temporal_hypergraphs = []
        for frame_idx, frame in enumerate(self.frame_sequence):
            if frame.ndim == 2:
                edge = OctonionHyperEdge(frame, frame_idx=frame_idx)
                hg = HyperGraph([edge])
                hg.is_temporal = len(self.frame_sequence) > 1
                hg.temporal_length = len(self.frame_sequence)
                self.temporal_hypergraphs.append(hg)
            elif frame.ndim == 3:
                # Multi-channel frame — encode each channel
                edges = []
                for c in range(frame.shape[0]):
                    edge = OctonionHyperEdge(frame[c], frame_idx=frame_idx)
                    edges.append(edge)
                hg = HyperGraph(edges)
                hg.is_temporal = len(self.frame_sequence) > 1
                hg.temporal_length = len(self.frame_sequence)
                self.temporal_hypergraphs.append(hg)
        return self.temporal_hypergraphs

    def extract_deltaT(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
    ) -> ProgramNode:
        """Extract the delta-T transformation between two frames.

        Analyzes the difference between frame_a and frame_b to determine
        the most likely transformation (move, color-swap, mirror, etc.).

        Args:
            frame_a: First frame grid.
            frame_b: Second frame grid.

        Returns:
            ProgramNode representing the inferred transformation.
        """
        if frame_a.shape != frame_b.shape:
            # Resize transformation
            element = DSLElement("resize", {
                "height": frame_b.shape[0],
                "width": frame_b.shape[1],
            })
            return ProgramNode(element)

        # Check for identity
        if np.array_equal(frame_a, frame_b):
            element = DSLElement("copy", {})
            return ProgramNode(element)

        # Check for color swap
        colors_a = set(np.unique(frame_a[frame_a > 0]).tolist())
        colors_b = set(np.unique(frame_b[frame_b > 0]).tolist())
        if colors_a != colors_b and np.array_equal(frame_a > 0, frame_b > 0):
            mapping = {}
            for ca in colors_a:
                for cb in colors_b:
                    if cb not in mapping.values():
                        mapping[ca] = cb
                        break
            element = DSLElement("map-color", {"mapping": mapping})
            return ProgramNode(element)

        # Check for mirror
        if np.array_equal(np.fliplr(frame_a), frame_b):
            element = DSLElement("mirror", {"axis": "horizontal"})
            return ProgramNode(element)
        if np.array_equal(np.flipud(frame_a), frame_b):
            element = DSLElement("mirror", {"axis": "vertical"})
            return ProgramNode(element)

        # Check for rotation
        for angle in [90, 180, 270]:
            if np.array_equal(np.rot90(frame_a, k=angle // 90), frame_b):
                element = DSLElement("rotate", {"angle": angle})
                return ProgramNode(element)

        # Check for translation (move)
        diff = frame_b.astype(np.int16) - frame_a.astype(np.int16)
        if np.sum(diff != 0) > 0:
            # Try to detect movement direction
            nonzero_diff = np.argwhere(diff != 0)
            if len(nonzero_diff) > 0:
                dy = int(np.mean(nonzero_diff[:, 0]))
                dx = int(np.mean(nonzero_diff[:, 1]))
                h, w = frame_a.shape
                # Check if it's a simple move
                moved = np.zeros_like(frame_a)
                for i in range(h):
                    for j in range(w):
                        ni, nj = i + dy, j + dx
                        if 0 <= ni < h and 0 <= nj < w:
                            moved[ni, nj] = frame_a[i, j]
                if np.array_equal(moved, frame_b):
                    element = DSLElement("move", {"dx": dx, "dy": dy})
                    return ProgramNode(element)

        # Default: overlay (best-effort)
        element = DSLElement("overlay", {})
        node = ProgramNode(element)
        # Add the actual target as a child for prediction
        return node

    def extract_all_deltaT(self) -> list[ProgramNode]:
        """Extract delta-T for all consecutive frame pairs.

        Returns:
            List of ProgramNode objects, one per frame transition.
        """
        deltas: list[ProgramNode] = []
        for i in range(len(self.frame_sequence) - 1):
            delta = self.extract_deltaT(
                self.frame_sequence[i], self.frame_sequence[i + 1]
            )
            deltas.append(delta)
        return deltas

    def predict_next_frame(
        self,
        deltaT: ProgramNode,
        last_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict the next frame by applying deltaT to the last frame.

        Args:
            deltaT: The delta-T transformation program.
            last_frame: The last known frame (defaults to last in sequence).

        Returns:
            Predicted next frame as ndarray.
        """
        if last_frame is None:
            if not self.frame_sequence:
                return np.zeros((1, 1), dtype=np.int8)
            last_frame = self.frame_sequence[-1]

        try:
            predicted = deltaT.apply(last_frame)
            return np.asarray(predicted, dtype=np.int8)
        except Exception:
            return last_frame.copy()

    def predict_frame_sequence(
        self,
        deltaT_list: list[ProgramNode],
        start_frame: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        """Predict a sequence of future frames.

        Args:
            deltaT_list: List of delta-T transformations to apply.
            start_frame: The starting frame.
            num_frames: Number of frames to predict.

        Returns:
            List of predicted frames.
        """
        frames = [start_frame]
        current = start_frame
        for i in range(num_frames):
            if i < len(deltaT_list):
                delta = deltaT_list[i]
            else:
                # Reuse last delta
                delta = deltaT_list[-1] if deltaT_list else ProgramNode(DSLElement("copy"))
            current = self.predict_next_frame(delta, current)
            frames.append(current)
        return frames[1:]  # Exclude the start frame
