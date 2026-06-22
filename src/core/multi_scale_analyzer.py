"""Multi-scale frame analysis engine with VL adapter interface."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.keyframe_extractor import KeyframeExtractor


class MultiScaleAnalyzer:
    """Multi-scale frame analysis using keyframe extraction and VL embeddings.

    Performs hierarchical analysis of video frames at multiple scales:
    global (full frame), object (connected component), and pixel-level.
    Uses a VL adapter for visual feature extraction when available.

    Attributes:
        keyframe_extractor: KeyframeExtractor for mutation-based segmentation.
        vl_adapter: Optional VL API adapter for visual analysis.
    """

    def __init__(
        self,
        extractor: KeyframeExtractor | None = None,
        vl_adapter: Any = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            extractor: KeyframeExtractor instance (creates default if None).
            vl_adapter: Optional DeepSeekVLAdapter for visual analysis.
        """
        self.keyframe_extractor = extractor or KeyframeExtractor()
        self.vl_adapter = vl_adapter

    def analyze(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """Analyze a frame sequence at multiple scales.

        Args:
            frames: List of frame arrays.

        Returns:
            Dictionary with keyframe_indices, motion_features, global_features,
            and optionally VL analysis results.
        """
        result: dict[str, Any] = {}

        # Keyframe extraction
        keyframe_indices = self.keyframe_extractor.get_keyframe_indices(frames)
        result["keyframe_indices"] = keyframe_indices

        # Motion features
        motion_features = self.extract_motion_features(frames)
        result["motion_features"] = motion_features
        result["motion_direction"] = self._estimate_motion_direction(motion_features)

        # Global features
        result["global_features"] = self._compute_global_features(frames)

        # Hierarchical analysis
        result["hierarchical"] = self.hierarchical_analysis(
            [frames[i] for i in keyframe_indices]
        )

        # VL analysis if adapter available
        if self.vl_adapter is not None:
            try:
                vl_result = self.vl_adapter.analyze_frames(
                    [frames[i] for i in keyframe_indices]
                )
                result["vl_analysis"] = vl_result
            except Exception:
                result["vl_analysis"] = None

        return result

    def hierarchical_analysis(self, keyframes: list[np.ndarray]) -> dict[str, Any]:
        """Perform hierarchical analysis on keyframes.

        Analyzes at three levels: global, object, and pixel.

        Args:
            keyframes: List of keyframe arrays.

        Returns:
            Dictionary with per-level analysis results.
        """
        result: dict[str, Any] = {
            "global": [],
            "object": [],
            "pixel": [],
        }

        for kf in keyframes:
            # Global level: overall statistics
            global_feat = {
                "shape": kf.shape,
                "num_colors": len(np.unique(kf[kf > 0])),
                "density": float(np.sum(kf > 0) / max(kf.size, 1)),
                "mean_color": float(np.mean(kf[kf > 0])) if np.any(kf > 0) else 0.0,
            }
            result["global"].append(global_feat)

            # Object level: connected components
            from scipy import ndimage
            binary = (kf > 0).astype(np.int32)
            labeled, num_objects = ndimage.label(binary)
            objects = []
            for obj_id in range(1, num_objects + 1):
                obj_mask = labeled == obj_id
                obj_pixels = np.argwhere(obj_mask)
                if len(obj_pixels) > 0:
                    objects.append({
                        "id": obj_id,
                        "size": int(np.sum(obj_mask)),
                        "bbox": (
                            int(obj_pixels[:, 0].min()),
                            int(obj_pixels[:, 0].max()),
                            int(obj_pixels[:, 1].min()),
                            int(obj_pixels[:, 1].max()),
                        ),
                        "color": int(kf[obj_mask][0]) if len(kf[obj_mask]) > 0 else 0,
                    })
            result["object"].append(objects)

            # Pixel level: color histogram
            hist = np.bincount(kf.flatten(), minlength=10)
            result["pixel"].append(hist.tolist())

        return result

    def extract_motion_features(self, frames: list[np.ndarray]) -> np.ndarray:
        """Extract motion features from frame sequence.

        Computes per-frame optical flow approximation using pixel differences.

        Args:
            frames: List of frame arrays.

        Returns:
            ndarray of motion feature vectors, shape (T-1, 4) where
            columns are [dx_mean, dy_mean, magnitude, direction].
        """
        if len(frames) < 2:
            return np.zeros((0, 4), dtype=np.float32)

        features = []
        for i in range(1, len(frames)):
            if frames[i - 1].shape != frames[i].shape:
                features.append([0.0, 0.0, 0.0, 0.0])
                continue

            prev = frames[i - 1].astype(np.float32)
            curr = frames[i].astype(np.float32)
            diff = curr - prev

            # Compute centroid shift of non-zero pixels
            prev_nz = np.argwhere(prev > 0)
            curr_nz = np.argwhere(curr > 0)

            if len(prev_nz) > 0 and len(curr_nz) > 0:
                prev_centroid = prev_nz.mean(axis=0)
                curr_centroid = curr_nz.mean(axis=0)
                dy = float(curr_centroid[0] - prev_centroid[0])
                dx = float(curr_centroid[1] - prev_centroid[1])
            else:
                dy, dx = 0.0, 0.0

            magnitude = float(np.sqrt(dx ** 2 + dy ** 2))
            direction = float(np.arctan2(dy, dx)) if magnitude > 0 else 0.0

            features.append([dx, dy, magnitude, direction])

        return np.array(features, dtype=np.float32)

    def _estimate_motion_direction(self, motion_features: np.ndarray) -> dict[str, float]:
        """Estimate overall motion direction from features.

        Args:
            motion_features: Array of motion feature vectors.

        Returns:
            Dictionary with mean_dx, mean_dy, mean_magnitude, mean_direction.
        """
        if len(motion_features) == 0:
            return {"mean_dx": 0.0, "mean_dy": 0.0, "mean_magnitude": 0.0, "mean_direction": 0.0}

        return {
            "mean_dx": float(np.mean(motion_features[:, 0])),
            "mean_dy": float(np.mean(motion_features[:, 1])),
            "mean_magnitude": float(np.mean(motion_features[:, 2])),
            "mean_direction": float(np.mean(motion_features[:, 3])),
        }

    def _compute_global_features(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """Compute global features across all frames.

        Args:
            frames: List of frame arrays.

        Returns:
            Dictionary of global features.
        """
        if not frames:
            return {}

        all_colors: set[int] = set()
        total_pixels = 0
        non_zero_pixels = 0

        for f in frames:
            colors = np.unique(f[f > 0])
            all_colors.update(colors.tolist())
            total_pixels += f.size
            non_zero_pixels += int(np.sum(f > 0))

        return {
            "num_frames": len(frames),
            "unique_colors": sorted(all_colors),
            "avg_density": float(non_zero_pixels / max(total_pixels, 1)),
            "frame_shapes": [f.shape for f in frames],
        }
