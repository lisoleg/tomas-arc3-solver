"""Keyframe extraction via inter-frame delta-T mutation detection."""
from __future__ import annotations

import numpy as np


class KeyframeExtractor:
    """Extracts keyframes by detecting inter-frame mutation events.

    A mutation is a significant change between consecutive frames that
    indicates a transition point. Keyframes are the frames immediately
    before and after each mutation.

    Attributes:
        diff_threshold: Normalized difference threshold for mutation detection.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        """Initialize the extractor.

        Args:
            threshold: Normalized difference threshold (0-1). Frames with
                difference above this are considered mutations.
        """
        self.diff_threshold = threshold

    def detect_mutation(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
    ) -> bool:
        """Detect if there is a mutation between two frames.

        A mutation occurs when the normalized pixel difference exceeds
        the threshold.

        Args:
            frame_a: First frame.
            frame_b: Second frame.

        Returns:
            True if a mutation is detected.
        """
        if frame_a.shape != frame_b.shape:
            return True

        diff = np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))
        total_pixels = frame_a.size
        changed_pixels = np.sum(diff > 0)
        normalized_diff = changed_pixels / max(total_pixels, 1)
        return normalized_diff >= self.diff_threshold

    def extract(self, frames: list[np.ndarray]) -> list[int]:
        """Extract keyframe indices from a frame sequence.

        Always includes frame 0. Additional frames are included when
        a mutation is detected from the previous keyframe.

        Args:
            frames: List of frame arrays.

        Returns:
            List of keyframe indices.
        """
        return self.get_keyframe_indices(frames)

    def get_keyframe_indices(self, frames: list[np.ndarray]) -> list[int]:
        """Get indices of keyframes in a frame sequence.

        Args:
            frames: List of frame arrays.

        Returns:
            Sorted list of keyframe indices.
        """
        if not frames:
            return []
        if len(frames) == 1:
            return [0]

        keyframes = [0]
        for i in range(1, len(frames)):
            if self.detect_mutation(frames[i - 1], frames[i]):
                keyframes.append(i)

        # Always include the last frame
        if keyframes[-1] != len(frames) - 1:
            keyframes.append(len(frames) - 1)

        return keyframes

    def extract_keyframe_pairs(
        self, frames: list[np.ndarray]
    ) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
        """Extract consecutive keyframe pairs with their frames.

        Args:
            frames: List of frame arrays.

        Returns:
            List of (idx_a, idx_b, frame_a, frame_b) tuples.
        """
        indices = self.get_keyframe_indices(frames)
        pairs = []
        for i in range(len(indices) - 1):
            idx_a = indices[i]
            idx_b = indices[i + 1]
            pairs.append((idx_a, idx_b, frames[idx_a], frames[idx_b]))
        return pairs

    def compute_frame_differences(self, frames: list[np.ndarray]) -> list[float]:
        """Compute normalized differences between consecutive frames.

        Args:
            frames: List of frame arrays.

        Returns:
            List of normalized difference values.
        """
        diffs = []
        for i in range(1, len(frames)):
            if frames[i - 1].shape == frames[i].shape:
                diff = np.abs(
                    frames[i - 1].astype(np.float32) - frames[i].astype(np.float32)
                )
                changed = np.sum(diff > 0)
                normalized = changed / max(frames[i].size, 1)
                diffs.append(float(normalized))
            else:
                diffs.append(1.0)
        return diffs
