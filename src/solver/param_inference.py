"""Parameter inference from demo pairs for ARC DSL primitives.

Analyzes input-output example pairs to infer likely parameters for each
DSL primitive, enabling the searcher to try parameterized variants
instead of only default-parameter versions.

Key inference strategies:
- Size change detection → scale, tile, resize, crop parameters
- Color mapping inference → map-color, color-swap parameters
- Symmetry detection → mirror axis, rotate angle
- Object shift detection → move, copy dx/dy
- Gravity direction → gravity direction
- Identity detection → copy (no-op)

TOMAS v2.5: Parameter inference layer for real ARC task support.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import DSLElement, ProgramNode


class ParamInference:
    """Infer DSL primitive parameters from demo pairs.

    Given a list of demo pairs (input→output grids), this module:
    1. Extracts task-level features (size ratio, color changes, symmetry)
    2. Generates parameterized DSLElement candidates
    3. Returns both single-primitive and chain candidates

    All inference is fail-safe — returns empty list on any error.
    """

    def __init__(self) -> None:
        """Initialize the parameter inference module."""
        pass

    def infer_candidates(
        self, demo_pairs: list[dict[str, Any]]
    ) -> list[ProgramNode]:
        """Generate parameterized candidates from demo pairs.

        Args:
            demo_pairs: List of demo pairs with 'input' and 'output' grids.
                Each grid is a list of 2D arrays (video format).

        Returns:
            List of ProgramNode candidates with inferred parameters,
            sorted by likelihood (most likely first).
        """
        if not demo_pairs:
            return []

        candidates: list[ProgramNode] = []

        # Extract features from all demo pairs
        features = self._extract_features(demo_pairs)
        if features is None:
            return []

        # Generate single-primitive candidates (depth 1)
        candidates.extend(self._gen_identity(features))
        candidates.extend(self._gen_scale_tile(features))
        candidates.extend(self._gen_color_ops(features))
        candidates.extend(self._gen_symmetry(features))
        candidates.extend(self._gen_gravity(features))
        candidates.extend(self._gen_move_copy(features))
        candidates.extend(self._gen_crop_resize(features))
        candidates.extend(self._gen_fill_ops(features))
        # v2.6: New inference methods
        candidates.extend(self._gen_multi_swap(features))
        candidates.extend(self._gen_map_by_function(features))
        candidates.extend(self._gen_complete_pattern(features))
        candidates.extend(self._gen_shift_object(features))
        # v2.4.5: Object extraction primitives (no parameters needed)
        candidates.append(ProgramNode(DSLElement("extract-largest-object")))
        candidates.append(ProgramNode(DSLElement("extract-smallest-object")))
        # v2.4.6: New primitives
        candidates.extend(self._gen_invert_colors(features))
        candidates.extend(self._gen_fill_connected(features))
        candidates.extend(self._gen_recolor_by_cc(features))
        candidates.extend(self._gen_tile_repeat(features))
        candidates.extend(self._gen_scale_pattern(features))

        # v2.8: Removed broken v2.4.9 map-color block that used {"from":..,"to":..}
        # params instead of {"mapping":{..}}. The _gen_color_ops method
        # already generates correct map-color candidates.
        pairs = features.get("pairs", [])

        candidates.extend(self._gen_crop_to_obj(features))
        candidates.extend(self._gen_replicate_obj(features))
        candidates.extend(self._gen_pad_row(features))
        
        # v2.5.0: Add new high-value primitives (optimized - only use colors from output)
        # Get output colors to reduce candidates
        out_colors = set()
        if len(pairs) >= 1:
            out0 = pairs[0]["output"][0]
            out_colors = set(out0.flatten().tolist())
            if 0 in out_colors:
                out_colors.remove(0)
        out_colors_list = sorted(out_colors) if out_colors else [1, 2, 3]  # default if no color info
        
        # apply-if: condition-based operation (use output colors + common conditions)
        for op in ['invert', 'clear', 'fill']:
            for cond in ['has_color', 'all_same', 'is_uniform']:
                for color in out_colors_list[:2]:  # limit to top 2 colors
                    elem = DSLElement("apply-if", {
                        "cond": cond, "color": color, "op": op
                    })
                    candidates.append(ProgramNode(elem))
        
        # fill-if: condition-based fill (only use output colors)
        for color in out_colors_list[:3]:  # limit to top 3 colors
            for cond in ['has_color', 'is_uniform']:
                for cond_color in out_colors_list[:2]:
                    elem = DSLElement("fill-if", {"color": color, "cond": cond, "cond_color": cond_color})
                    candidates.append(ProgramNode(elem))
        
        # rotate-object: rotate largest object (keep all 3 candidates)
        for k in [1, 2, 3]:
            elem = DSLElement("rotate-object", {"k": k})
            candidates.append(ProgramNode(elem))
        
        # scale-object: scale largest object (keep all 3 candidates)
        for factor in [2, 3, 4]:
            elem = DSLElement("scale-object", {"factor": factor})
            candidates.append(ProgramNode(elem))
        
        # diagonal-fill: fill diagonal lines (only use output colors)
        for color in out_colors_list[:2]:  # limit to top 2 colors
            for diag in ['main', 'anti']:  # skip 'all' to reduce candidates
                elem = DSLElement("diagonal-fill", {"color": color, "diag": diag})
                candidates.append(ProgramNode(elem))
        
        # pattern-extend: extend pattern (only use output colors)
        for color in out_colors_list[:2]:  # limit to top 2 colors
            for pattern in ['checkerboard', 'striped_h']:  # skip 'dot' to reduce candidates
                elem = DSLElement("pattern-extend", {"color": color, "pattern": pattern})
                candidates.append(ProgramNode(elem))
        
        # Generate two-primitive chains (depth 2) for common combos
        candidates.extend(self._gen_chain_candidates(features))

        # v2.9: High-impact new primitives
        candidates.extend(self._gen_tile_seed(features))
        candidates.extend(self._gen_fill_by_period(features, demo_pairs))
        candidates.extend(self._gen_crop_to_bbox(features))
        candidates.extend(self._gen_fill_empty_neighbor(features))
        candidates.extend(self._gen_direct_map(features, demo_pairs))
        candidates.extend(self._gen_repeat_grid(features))

        # v3.0: High-impact inference methods for 68% accuracy target
        candidates.extend(self._gen_universal_color_map(features, demo_pairs))
        candidates.extend(self._gen_smart_flood_fill(features, demo_pairs))
        candidates.extend(self._gen_grid_diff_apply(features, demo_pairs))
        candidates.extend(self._gen_object_transform(features, demo_pairs))
        candidates.extend(self._gen_conditional_map(features, demo_pairs))
        candidates.extend(self._gen_nearest_neighbor_transform(features, demo_pairs))
        candidates.extend(self._gen_replace_marker_with_border(features, demo_pairs))
        candidates.extend(self._gen_gravity_stack(features))
        candidates.extend(self._gen_draw_frame(features))
        candidates.extend(self._gen_sort_objects(features))
        candidates.extend(self._gen_propagate_color(features))

        # Deduplicate by program signature
        seen: set[str] = set()
        unique: list[ProgramNode] = []
        for node in candidates:
            sig = self._node_signature(node)
            if sig not in seen:
                seen.add(sig)
                unique.append(node)

        # v2.9: Score and truncate candidates to reduce search space.
        # Use ALL demo pairs for scoring. Keep top 200 candidates
        # (increased from 80 to give more chances for complex tasks).
        unique = self._score_and_truncate(unique, demo_pairs, max_candidates=200)

        return unique

    def _score_and_truncate(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
        max_candidates: int = 60,
    ) -> list[ProgramNode]:
        """Score candidates by demo pair similarity and truncate to top-N.

        v2.8: Use ALL demo pairs for scoring (not just the first).
        A candidate that matches pair 1 but not pair 2 is likely
        overfitting — we want candidates that generalize across
        all pairs.

        Scoring: apply each candidate to ALL demo inputs, compute
        average pixel similarity to expected outputs. Candidates
        that crash or produce wrong-sized output get score 0.

        Args:
            candidates: List of ProgramNode candidates.
            demo_pairs: Demo pairs for scoring.
            max_candidates: Maximum candidates to keep.

        Returns:
            Truncated and sorted list of ProgramNode candidates.
        """
        if len(candidates) <= max_candidates:
            return candidates

        if not demo_pairs:
            return candidates[:max_candidates]

        # v2.8: Collect all (input, expected) pairs for scoring
        eval_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, inp in enumerate(input_grids):
                if i < len(output_grids):
                    eval_pairs.append(
                        (np.asarray(inp, dtype=np.int8),
                         np.asarray(output_grids[i], dtype=np.int8))
                    )

        if not eval_pairs:
            return candidates[:max_candidates]

        scored: list[tuple[float, int, ProgramNode]] = []
        for idx, node in enumerate(candidates):
            total_score = 0.0
            valid_pairs = 0
            for inp, expected in eval_pairs:
                score = 0.0
                try:
                    pred = node.apply(inp)
                    pred_arr = np.asarray(pred, dtype=np.int8)
                    if pred_arr.shape == expected.shape:
                        total_pixels = expected.size
                        if total_pixels > 0:
                            match = int(np.sum(pred_arr == expected))
                            score = match / total_pixels
                    else:
                        # Shape mismatch — partial credit for overlapping region
                        h = min(pred_arr.shape[0], expected.shape[0])
                        w = min(pred_arr.shape[1], expected.shape[1])
                        if h > 0 and w > 0:
                            total = h * w
                            match = int(np.sum(pred_arr[:h, :w] == expected[:h, :w]))
                            score = match / total * 0.5
                except Exception:
                    score = 0.0

                total_score += score
                valid_pairs += 1

            avg_score = total_score / max(valid_pairs, 1)
            scored.append((avg_score, idx, node))

        # Sort by score descending, keep top max_candidates
        scored.sort(key=lambda x: (-x[0], x[1]))
        truncated = [node for _, _, node in scored[:max_candidates]]

        return truncated

    def _extract_features(
        self, demo_pairs: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Extract task-level features from demo pairs.

        Args:
            demo_pairs: List of demo pairs.

        Returns:
            Feature dictionary or None if extraction fails.
        """
        try:
            pair_features = []
            for pair in demo_pairs:
                input_grids = pair.get("input", [])
                output_grids = pair.get("output", [])
                if not input_grids or not output_grids:
                    continue

                inp = np.asarray(input_grids[0], dtype=np.int8)
                out = np.asarray(output_grids[0], dtype=np.int8)

                pf = {
                    "input": inp,
                    "output": out,
                    "in_h": inp.shape[0],
                    "in_w": inp.shape[1],
                    "out_h": out.shape[0],
                    "out_w": out.shape[1],
                    "in_colors": set(np.unique(inp).tolist()),
                    "out_colors": set(np.unique(out).tolist()),
                    "is_identity": np.array_equal(inp, out),
                }
                pair_features.append(pf)

            if not pair_features:
                return None

            # Aggregate features across pairs
            first = pair_features[0]
            features = {
                "pairs": pair_features,
                "num_pairs": len(pair_features),
                "all_identity": all(p["is_identity"] for p in pair_features),
                "in_h": first["in_h"],
                "in_w": first["in_w"],
                "out_h": first["out_h"],
                "out_w": first["out_w"],
                "size_same": all(
                    p["in_h"] == p["out_h"] and p["in_w"] == p["out_w"]
                    for p in pair_features
                ),
                "h_ratio": first["out_h"] / max(first["in_h"], 1),
                "w_ratio": first["out_w"] / max(first["in_w"], 1),
                "all_colors": set(),
                "color_mappings": [],
            }
            for p in pair_features:
                features["all_colors"].update(p["in_colors"])
                features["all_colors"].update(p["out_colors"])

                # Infer color mapping
                if p["in_h"] == p["out_h"] and p["in_w"] == p["out_w"]:
                    mapping = self._infer_color_mapping(p["input"], p["output"])
                    if mapping:
                        features["color_mappings"].append(mapping)

            # Consistent color mapping across all pairs?
            # Note: mappings may differ in key set (different pairs may have
            # different colors), but should be compatible (no contradictions).
            if features["color_mappings"]:
                # Merge all mappings, checking for contradictions
                merged_map: dict[int, int] = {}
                compatible = True
                for m in features["color_mappings"]:
                    for k, v in m.items():
                        if k in merged_map:
                            if merged_map[k] != v:
                                compatible = False
                                break
                        else:
                            merged_map[k] = v
                    if not compatible:
                        break

                features["consistent_color_map"] = compatible
                features["color_map"] = merged_map if compatible else features["color_mappings"][0]
            else:
                features["consistent_color_map"] = False
                features["color_map"] = {}

            # Check symmetry
            features["is_mirror_h"] = all(
                np.array_equal(p["output"], np.fliplr(p["input"]))
                for p in pair_features
            )
            features["is_mirror_v"] = all(
                np.array_equal(p["output"], np.flipud(p["input"]))
                for p in pair_features
            )
            features["is_rotate_180"] = all(
                np.array_equal(p["output"], np.rot90(p["input"], k=2))
                for p in pair_features
            )
            features["is_rotate_90"] = all(
                p["output"].shape == (p["in_w"], p["in_h"])
                and np.array_equal(p["output"], np.rot90(p["input"], k=1))
                for p in pair_features
            )
            features["is_rotate_270"] = all(
                p["output"].shape == (p["in_w"], p["in_h"])
                and np.array_equal(p["output"], np.rot90(p["input"], k=3))
                for p in pair_features
            )

            return features
        except Exception:
            return None

    def _infer_color_mapping(
        self, inp: np.ndarray, out: np.ndarray
    ) -> dict[int, int]:
        """Infer color mapping from input to output (same-size grids).

        Args:
            inp: Input grid.
            out: Output grid.

        Returns:
            Mapping dict {old_color: new_color}.
        """
        if inp.shape != out.shape:
            return {}

        mapping: dict[int, int] = {}
        inp_flat = inp.flatten()
        out_flat = out.flatten()

        for i in range(len(inp_flat)):
            old_c = int(inp_flat[i])
            new_c = int(out_flat[i])
            if old_c != new_c:
                if old_c in mapping:
                    # Inconsistent mapping
                    if mapping[old_c] != new_c:
                        return {}
                else:
                    mapping[old_c] = new_c

        return mapping

    def _gen_identity(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate identity (copy) candidate if all pairs are identity.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        if features.get("all_identity", False):
            return [ProgramNode(DSLElement("copy"))]
        return []

    def _gen_scale_tile(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate scale/tile candidates based on size ratios.

        v2.5.1: Support non-uniform scaling (different h/w ratios),
                 and try multiple common scale factors (2, 3, 4).

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []
        h_ratio = features.get("h_ratio", 1.0)
        w_ratio = features.get("w_ratio", 1.0)

        # Integer scaling (same ratio for h and w)
        if h_ratio == w_ratio and h_ratio > 1.0:
            factor = int(round(h_ratio))
            if abs(factor - h_ratio) < 0.01 and factor > 1:
                # Try scale
                candidates.append(
                    ProgramNode(DSLElement("scale", {"factor": factor}))
                )
                # Try tile
                candidates.append(
                    ProgramNode(DSLElement("tile", {"factor_h": factor, "factor_w": factor}))
                )

        # Non-uniform scaling
        if h_ratio > 1.0 and w_ratio > 1.0:
            fh = int(round(h_ratio))
            fw = int(round(w_ratio))
            if abs(fh - h_ratio) < 0.01 and abs(fw - w_ratio) < 0.01:
                if fh != fw:
                    candidates.append(
                        ProgramNode(DSLElement("tile", {"factor_h": fh, "factor_w": fw}))
                    )
                # Also try scale with the larger factor + crop
                # (e.g., scale by 2 then crop to 12x10 from 12x12)

        # v2.5.1: Also try common scale factors even if ratio doesn't match exactly
        # (the ratio might differ between train pairs, but the factor is consistent)
        for factor in [2, 3, 4]:
            if factor > 1:
                # Only add if not already added
                sig = f"scale:factor={factor}"
                existing = any(
                    c.element and c.element.name == "scale" and c.element.params.get("factor") == factor
                    for c in candidates
                )
                if not existing and h_ratio > 1.5:
                    candidates.append(
                        ProgramNode(DSLElement("scale", {"factor": factor}))
                    )

        return candidates

    def _gen_color_ops(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate color-swap and map-color candidates.

        v2.8: Multi-color mapping inference with cross-pair consistency.
        Instead of trying ALL single-color X→Y pairs (which generates
        O(N²) noise candidates), we:
        1. Infer the complete color mapping from ALL demo pairs
        2. Only generate candidates for mappings that are consistent
        3. For single-color swaps, only try colors that actually change

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        # v2.8: Use the consistent color map from ALL pairs
        # This is the most important candidate — the full mapping
        if features.get("consistent_color_map", False):
            mapping = features.get("color_map", {})
            if mapping and len(mapping) > 0:
                # Generate the full multi-color mapping as top candidate
                candidates.append(
                    ProgramNode(DSLElement("map-color", {"mapping": dict(mapping)}))
                )

                # Also try as color-swap if exactly one pair swapped
                if len(mapping) == 2:
                    keys = list(mapping.keys())
                    vals = list(mapping.values())
                    if keys[0] == vals[1] and keys[1] == vals[0]:
                        candidates.append(
                            ProgramNode(
                                DSLElement("color-swap", {
                                    "color_a": keys[0],
                                    "color_b": keys[1],
                                })
                            )
                        )

                # Also generate individual single-color mappings
                # from the consistent map (for tasks where only some
                # colors change)
                for old_c, new_c in mapping.items():
                    if old_c != new_c:
                        candidates.append(
                            ProgramNode(
                                DSLElement("map-color", {"mapping": {old_c: new_c}})
                            )
                        )
        else:
            # Inconsistent mapping — try the first pair's mapping anyway
            if features.get("color_mappings"):
                first_map = features["color_mappings"][0]
                if first_map:
                    candidates.append(
                        ProgramNode(DSLElement("map-color", {"mapping": dict(first_map)}))
                    )

        # v2.8: For single-color swaps, only try colors that ACTUALLY
        # differ between input and output (not all N² pairs)
        pairs = features.get("pairs", [])
        if pairs:
            # Find colors that change between input and output
            changed_colors: set[int] = set()
            for p in pairs:
                inp = p["input"]
                out = p["output"]
                if inp.shape == out.shape:
                    diff_mask = inp != out
                    changed_colors.update(inp[diff_mask].tolist())
                    changed_colors.update(out[diff_mask].tolist())

            # Only try color-swap between colors that actually change
            changed_list = sorted(changed_colors)
            for i in range(len(changed_list)):
                for j in range(i + 1, len(changed_list)):
                    ca, cb = changed_list[i], changed_list[j]
                    candidates.append(
                        ProgramNode(
                            DSLElement("color-swap", {"color_a": ca, "color_b": cb})
                        )
                    )

            # Try single-color mappings only for changed colors
            all_colors = sorted(features.get("all_colors", set()))
            for old_c in changed_list:
                for new_c in all_colors:
                    if old_c != new_c:
                        candidates.append(
                            ProgramNode(
                                DSLElement("map-color", {"mapping": {old_c: new_c}})
                            )
                        )

        return candidates

    def _gen_symmetry(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate mirror/rotate candidates based on symmetry detection.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        if features.get("is_mirror_h", False):
            candidates.append(
                ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
            )
        if features.get("is_mirror_v", False):
            candidates.append(
                ProgramNode(DSLElement("mirror", {"axis": "vertical"}))
            )
        if features.get("is_rotate_180", False):
            candidates.append(
                ProgramNode(DSLElement("rotate", {"angle": 180}))
            )
        if features.get("is_rotate_90", False):
            candidates.append(
                ProgramNode(DSLElement("rotate", {"angle": 90}))
            )
        if features.get("is_rotate_270", False):
            candidates.append(
                ProgramNode(DSLElement("rotate", {"angle": 270}))
            )

        # Also generate all symmetry variants as candidates
        # (low priority — they'll be filtered by verification)
        candidates.append(ProgramNode(DSLElement("mirror", {"axis": "horizontal"})))
        candidates.append(ProgramNode(DSLElement("mirror", {"axis": "vertical"})))
        candidates.append(ProgramNode(DSLElement("rotate", {"angle": 90})))
        candidates.append(ProgramNode(DSLElement("rotate", {"angle": 180})))
        candidates.append(ProgramNode(DSLElement("rotate", {"angle": 270})))

        return candidates

    def _gen_gravity(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate gravity candidates.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        # Check each direction
        for direction in ["down", "up", "left", "right"]:
            candidates.append(
                ProgramNode(DSLElement("gravity", {"direction": direction}))
            )

        return candidates

    def _gen_move_copy(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate move/copy candidates with inferred offsets.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        # Try to infer move offset from first pair
        pairs = features.get("pairs", [])
        if pairs:
            inp = pairs[0]["input"]
            out = pairs[0]["output"]

            # Find the offset that best explains the transformation
            offset = self._find_move_offset(inp, out)
            if offset is not None:
                dx, dy = offset
                candidates.append(
                    ProgramNode(DSLElement("move", {"dx": dx, "dy": dy}))
                )
                candidates.append(
                    ProgramNode(DSLElement("copy", {"dx": dx, "dy": dy}))
                )

        # Also try common small offsets
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                candidates.append(
                    ProgramNode(DSLElement("move", {"dx": dx, "dy": dy}))
                )

        return candidates

    def _find_move_offset(
        self, inp: np.ndarray, out: np.ndarray
    ) -> tuple[int, int] | None:
        """Find the (dx, dy) offset that transforms input to output.

        Args:
            inp: Input grid.
            out: Output grid.

        Returns:
            (dx, dy) tuple or None if no single offset works.
        """
        if inp.shape != out.shape:
            return None

        h, w = inp.shape
        # Find non-zero region in input
        inp_mask = inp != 0
        if not np.any(inp_mask):
            return None

        # Find non-zero region in output
        out_mask = out != 0
        if not np.any(out_mask):
            return None

        # Try all possible offsets
        for dy in range(-h + 1, h):
            for dx in range(-w + 1, w):
                # Shift input by (dx, dy) and compare
                shifted = np.zeros_like(inp)
                src_r0 = max(0, -dy)
                src_c0 = max(0, -dx)
                dst_r0 = max(0, dy)
                dst_c0 = max(0, dx)
                copy_h = min(h - src_r0, h - dst_r0)
                copy_w = min(w - src_c0, w - dst_c0)
                if copy_h > 0 and copy_w > 0:
                    shifted[dst_r0:dst_r0 + copy_h, dst_c0:dst_c0 + copy_w] = \
                        inp[src_r0:src_r0 + copy_h, src_c0:src_c0 + copy_w]
                    if np.array_equal(shifted, out):
                        return (dx, dy)

        return None

    def _gen_crop_resize(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate crop and resize candidates.

        v2.5.1: Multi-position crop — try extracting objects at different
                 starting positions, not just (0,0). Also try extract-pattern
                 for size-reduction tasks.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        out_h = features.get("out_h", 0)
        out_w = features.get("out_w", 0)
        in_h = features.get("in_h", 0)
        in_w = features.get("in_w", 0)

        # If output is smaller, try crop at multiple positions
        if out_h < in_h or out_w < in_w:
            # Try crop at (0,0)
            candidates.append(
                ProgramNode(DSLElement("crop", {"height": out_h, "width": out_w}))
            )

            # Try crop at every position where a non-zero region starts
            pairs = features.get("pairs", [])
            if pairs:
                inp = pairs[0]["input"]
                # Find bounding boxes of non-zero regions
                mask = inp != 0
                if np.any(mask):
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]

                    # Try crop at the start of the first object
                    if rmin + out_h <= in_h and cmin + out_w <= in_w:
                        candidates.append(
                            ProgramNode(DSLElement("crop", {
                                "top": int(rmin), "left": int(cmin),
                                "height": out_h, "width": out_w,
                            }))
                        )

                    # Try crop at center
                    center_top = max(0, (in_h - out_h) // 2)
                    center_left = max(0, (in_w - out_w) // 2)
                    candidates.append(
                        ProgramNode(DSLElement("crop", {
                            "top": center_top, "left": center_left,
                            "height": out_h, "width": out_w,
                        }))
                    )

                    # Try several positions along the grid
                    for top in range(0, max(1, in_h - out_h + 1), max(1, (in_h - out_h) // 3)):
                        for left in range(0, max(1, in_w - out_w + 1), max(1, (in_w - out_w) // 3)):
                            if top + out_h <= in_h and left + out_w <= in_w:
                                candidates.append(
                                    ProgramNode(DSLElement("crop", {
                                        "top": top, "left": left,
                                        "height": out_h, "width": out_w,
                                    }))
                                )

            # Try extract-pattern
            candidates.append(ProgramNode(DSLElement("extract-pattern")))

        # If output size differs, try resize
        if not features.get("size_same", False) and out_h > 0 and out_w > 0:
            candidates.append(
                ProgramNode(DSLElement("resize", {"height": out_h, "width": out_w}))
            )

        return candidates

    def _gen_fill_ops(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate flood-fill and fill-region candidates.

        v2.5.1: Try flood-fill with ALL colors 1-9 (not just new colors),
                 since the fill color may already appear in the input as
                 a boundary color. Also try fill-region with border/interior.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        # Try flood-fill with each color that appears in output or is common
        pairs = features.get("pairs", [])
        if pairs:
            out_colors = pairs[0]["out_colors"]
            inp_colors = pairs[0]["in_colors"]
            # Colors in output but not input (new colors = fill candidates)
            new_colors = out_colors - inp_colors
            # Also try all colors present in output (fill color may be a boundary)
            all_try_colors = out_colors | {1, 2, 3, 4, 5, 6, 7, 8, 9}
            for c in sorted(all_try_colors):
                candidates.append(
                    ProgramNode(DSLElement("flood-fill", {"color": c}))
                )

        # Try boundary-detect and symmetry-detect
        candidates.append(ProgramNode(DSLElement("boundary-detect")))
        candidates.append(ProgramNode(DSLElement("symmetry-detect")))
        candidates.append(ProgramNode(DSLElement("complete-shape")))

        # v2.5.1: Try fill-region with different colors and regions
        for c in range(1, 10):
            for region in ["border", "interior"]:
                candidates.append(
                    ProgramNode(DSLElement("fill-region", {"color": c, "region": region}))
                )

        return candidates

    def _gen_chain_candidates(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate two-primitive chain candidates for common combos.

        Common ARC patterns:
        - rotate + map-color
        - scale + map-color
        - mirror + color-swap
        - crop + map-color

        Args:
            features: Extracted features.

        Returns:
            List of chain ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []
        color_map = features.get("color_map", {})

        # Helper to create a chain of two primitives
        def make_chain(
            elem1: DSLElement, elem2: DSLElement
        ) -> ProgramNode:
            node = ProgramNode(elem1)
            child = ProgramNode(elem2)
            node.children.append(child)
            node.combo_type = "chain"
            node.total_mdl = node.compute_mdl()
            return node

        # If there's a color mapping, combine with symmetry ops
        if color_map:
            map_elem = DSLElement("map-color", {"mapping": color_map})
            for angle in [90, 180, 270]:
                candidates.append(
                    make_chain(DSLElement("rotate", {"angle": angle}), map_elem)
                )
            for axis in ["horizontal", "vertical"]:
                candidates.append(
                    make_chain(DSLElement("mirror", {"axis": axis}), map_elem)
                )

            # scale + map-color
            h_ratio = features.get("h_ratio", 1.0)
            if h_ratio > 1.0 and h_ratio == features.get("w_ratio", 1.0):
                factor = int(round(h_ratio))
                if abs(factor - h_ratio) < 0.01 and factor > 1:
                    candidates.append(
                        make_chain(
                            DSLElement("scale", {"factor": factor}), map_elem
                        )
                    )

        # mirror + color-swap combos
        all_colors = sorted(features.get("all_colors", set()))
        if len(all_colors) >= 2:
            ca, cb = all_colors[0], all_colors[1]
            swap_elem = DSLElement("color-swap", {"color_a": ca, "color_b": cb})
            for axis in ["horizontal", "vertical"]:
                candidates.append(
                    make_chain(DSLElement("mirror", {"axis": axis}), swap_elem)
                )

        return candidates

    def _node_signature(self, node: ProgramNode) -> str:
        """Generate a unique signature for a ProgramNode.

        Args:
            node: ProgramNode to sign.

        Returns:
            String signature.
        """
        parts: list[str] = []
        if node.element is not None:
            parts.append(
                f"{node.element.name}:{sorted(node.element.params.items())}"
            )
        for child in node.children:
            parts.append(self._node_signature(child))
        return f"{node.combo_type}|{'->'.join(parts)}"

    # ============================================================
    # v2.6: New inference methods for enhanced primitives
    # ============================================================

    def _infer_swap_pairs(
        self, demo_pairs: list[dict[str, Any]]
    ) -> list[list[int]] | None:
        """Infer multiple color swap pairs from demo pairs.

        Args:
            demo_pairs: List of demo pairs.

        Returns:
            List of [color_a, color_b] pairs, or None if not applicable.
        """
        try:
            # Collect all color mappings across pairs
            all_mappings = []
            for pair in demo_pairs:
                inp = pair["input"][0]
                out = pair["output"][0]
                if inp.shape != out.shape:
                    return None
                mapping = {}
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        if inp[i,j] != out[i,j]:
                            mapping[int(inp[i,j])] = int(out[i,j])
                all_mappings.append(mapping)
            
            if not all_mappings:
                return None
            
            # Check if all mappings are consistent swaps
            # A swap means: if a->b, then b->a in some other pair
            swap_pairs = []
            processed = set()
            
            # Collect all mapped colors
            all_colors = set()
            for m in all_mappings:
                all_colors.update(m.keys())
                all_colors.update(m.values())
            
            # Try to find swap pairs
            for c in sorted(all_colors):
                if c in processed:
                    continue
                # Find what c maps to
                targets = set()
                for m in all_mappings:
                    if c in m:
                        targets.add(m[c])
                
                if len(targets) == 1:
                    target = targets.pop()
                    # Check if target maps back to c
                    target_maps_to_c = False
                    for m in all_mappings:
                        if target in m and m[target] == c:
                            target_maps_to_c = True
                            break
                    
                    if target_maps_to_c:
                        swap_pairs.append([c, target])
                        processed.add(c)
                        processed.add(target)
            
            return swap_pairs if swap_pairs else None
        except Exception:
            return None

    def _gen_multi_swap(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate multi-swap candidates.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        # Try to infer swap pairs from pairs
        pairs = features.get("pairs", [])
        if not pairs:
            return candidates

        demo_pairs = []
        for p in pairs:
            inp = p.get("input", None)
            out = p.get("output", None)
            if inp is not None and out is not None:
                demo_pairs.append({"input": [inp], "output": [out]})

        swap_pairs = self._infer_swap_pairs(demo_pairs)
        if swap_pairs:
            candidates.append(
                ProgramNode(DSLElement("multi-swap", {"swap_pairs": swap_pairs}))
            )
        
        # Also try heuristic: pair (1,5), (2,6), (3,7), (4,8), (9,0)
        heuristic_pairs = []
        for c in range(10):
            pair = (c + 4) % 10
            if [c, pair] not in heuristic_pairs and [pair, c] not in heuristic_pairs:
                heuristic_pairs.append([c, pair])
        
        candidates.append(
            ProgramNode(DSLElement("multi-swap", {"swap_pairs": heuristic_pairs}))
        )

        return candidates

    def _gen_map_by_function(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate map-by-function candidates.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        pairs = features.get("pairs", [])
        if not pairs:
            return candidates

        # Try to infer a mathematical mapping
        # Check if output = (input + const) % 10
        for const in range(1, 10):
            all_match = True
            for p in pairs:
                inp = p["input"]
                out = p["output"]
                if inp.shape != out.shape:
                    all_match = False
                    break
                # Check if out = (inp + const) % 10 for all non-zero
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        if inp[i,j] != 0:
                            expected = (inp[i,j] + const) % 10
                            if out[i,j] != expected:
                                all_match = False
                                break
                    if not all_match:
                        break
                if not all_match:
                    break
            
            if all_match:
                candidates.append(
                    ProgramNode(DSLElement("map-by-function", {
                        "func_type": "add",
                        "value": const,
                        "modulo": 10,
                    }))
                )
        
        # Check if output = (input - const) % 10
        for const in range(1, 10):
            all_match = True
            for p in pairs:
                inp = p["input"]
                out = p["output"]
                if inp.shape != out.shape:
                    all_match = False
                    break
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        if inp[i,j] != 0:
                            expected = (inp[i,j] - const) % 10
                            if out[i,j] != expected:
                                all_match = False
                                break
                    if not all_match:
                        break
                if not all_match:
                    break
            
            if all_match:
                candidates.append(
                    ProgramNode(DSLElement("map-by-function", {
                        "func_type": "sub",
                        "value": const,
                        "modulo": 10,
                    }))
                )
        
        # Also add "inv_mod" (swap pairs) as candidate
        candidates.append(
            ProgramNode(DSLElement("map-by-function", {
                "func_type": "inv_mod",
                "value": 0,
                "modulo": 10,
            }))
        )

        return candidates

    def _gen_complete_pattern(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate complete-pattern candidates.

        Improved v3.1: Try multiple strategies (sequence tiling + bounding box tiling).

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        pairs = features.get("pairs", [])
        if len(pairs) < 1:
            return candidates

        # Strategy 1: Bounding box tiling (works for many pattern completion tasks)
        candidates.append(
            ProgramNode(DSLElement("complete-pattern", {"strategy": "bounding_box"}))
        )

        # Strategy 2: Sequence tiling (for color sequence tasks)
        # Extract color sequence from first training pair's input
        inp = np.array(pairs[0]["input"])  # Convert to numpy array
        # Ensure 2D array
        if inp.ndim == 3:
            inp = inp[0]
        seq = []
        seen = set()
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                color = int(inp[r, c])  # Convert to Python int
                if color != 0 and color not in seen:
                    seq.append(color)
                    seen.add(color)

        if seq:
            # Try all rotations
            seq_len = len(seq)
            for rotation in range(seq_len):
                candidates.append(
                    ProgramNode(DSLElement("complete-pattern", {
                        "strategy": "sequence",
                        "rotation": rotation
                    }))
                )

        # Strategy 3: Auto (let the primitive decide)
        candidates.append(
            ProgramNode(DSLElement("complete-pattern", {"strategy": "auto"}))
        )

        return candidates

    def _gen_shift_object(self, features: dict[str, Any]) -> list[ProgramNode]:
        """Generate shift-object candidates.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        pairs = features.get("pairs", [])
        if len(pairs) < 1:
            return candidates

        # Try to find a consistent object shift
        inp = np.array(pairs[0]["input"])  # Convert to numpy array
        out = pairs[0]["output"]

        # Find move offset for the entire grid
        offset = self._find_move_offset(inp, out)
        if offset:
            dx, dy = offset
            # Use move primitive (which shifts all non-zero pixels)
            candidates.append(
                ProgramNode(DSLElement("move", {"dx": dx, "dy": dy}))
            )

        return candidates

    # ======== v2.4.6 New inference methods ========

    def _gen_invert_colors(self, features: dict) -> list:
        """Generate invert-colors candidates (no parameters)."""
        return [ProgramNode(DSLElement("invert-colors"))]

    def _gen_fill_connected(self, features: dict) -> list:
        """Generate fill-connected candidates (try all non-zero colors)."""
        candidates = []
        pairs = features.get("pairs", [])
        if not pairs:
            return candidates
        inp = pairs[0]["input"]
        colors = np.unique(inp)
        for c in colors:
            if c == 0:
                continue
            candidates.append(ProgramNode(DSLElement("fill-connected", {"color": int(c)})))
        return candidates

    def _gen_recolor_by_cc(self, features: dict) -> list:
        """Generate recolor-by-cc candidates (no parameters)."""
        return [ProgramNode(DSLElement("recolor-by-cc"))]

    def _gen_tile_repeat(self, features: dict) -> list:
        """Generate tile-repeat candidates (try repeats=2,3)."""
        candidates = []
        for r in [2, 3]:
            candidates.append(ProgramNode(DSLElement("tile-repeat", {"repeats": r})))
        return candidates


    def _gen_select_obj_by(self, features: dict) -> list:
        """Generate select-obj-by candidates."""
        candidates = []
        for prop in ['size', 'color', 'leftmost', 'rightmost', 'topmost', 'bottommost']:
            candidates.append(ProgramNode(DSLElement('select-obj-by', {'prop': prop})))
        return candidates

    def _gen_sort_obj_by(self, features: dict) -> list:
        """Generate sort-obj-by candidates."""
        candidates = []
        for prop in ['size', 'color', 'row', 'col']:
            candidates.append(ProgramNode(DSLElement('sort-obj-by', {'prop': prop})))
        return candidates


    def _gen_scale_pattern(self, features: dict) -> list:
        """Generate scale-pattern candidates (try factor=2,3,4)."""
        candidates = []
        for factor in [2, 3, 4]:
            candidates.append(ProgramNode(DSLElement('scale-pattern', {'factor': factor})))
        return candidates


    # ========= v2.4.8 New inference methods =========

    def _gen_crop_to_obj(self, features: dict) -> list:
        """Generate crop-to-obj candidates."""
        candidates = []
        pairs = features.get('pairs', [])
        colors = set()
        for p in pairs:
            for f in p.get('input', []):
                colors.update(np.unique(f))
        for c in colors:
            candidates.append(ProgramNode(DSLElement('crop-to-obj', {'color': int(c)})))
        return candidates

    def _gen_replicate_obj(self, features: dict) -> list:
        """Generate replicate-obj candidates."""
        candidates = []
        for d in ['right', 'left', 'down', 'up']:
            candidates.append(ProgramNode(DSLElement('replicate-obj', {'direction': d})))
        return candidates

    def _gen_pad_row(self, features: dict) -> list:
        """Generate pad-row candidates (improved v2.5.1: infer num from shape diff)."""
        candidates = []
        
        # v2.5.1: Infer num from output/input height difference
        out_h = features.get("out_h", 0)
        in_h = features.get("in_h", 0)
        if out_h > in_h and out_h - in_h <= 5:
            # Likely needs to pad rows at bottom (or top)
            num = out_h - in_h
            # Try both top=True and top=False
            candidates.append(ProgramNode(DSLElement('pad-row', {'num': num, 'top': True})))
            candidates.append(ProgramNode(DSLElement('pad-row', {'num': num, 'top': False})))
        else:
            # Fallback: try num=1,2,3
            for num in [1, 2, 3]:
                for top in [True, False]:
                    candidates.append(ProgramNode(DSLElement('pad-row', {'num': num, 'top': top})))
        
        return candidates

    # ============================================================
    # v2.9: High-impact candidate generators for 68% accuracy
    # ============================================================

    def _gen_tile_seed(self, features: dict) -> list[ProgramNode]:
        """Generate tile-seed candidates.
        
        Detects a seed pattern (bounding box of non-zero) and tiles it.
        Effective for pattern completion/tiling tasks.
        """
        candidates: list[ProgramNode] = []
        
        if not features.get("size_same", False):
            return candidates
        
        pairs = features.get("pairs", [])
        if not pairs:
            return candidates
        
        inp = pairs[0]["input"]
        non_zero = inp != 0
        if not np.any(non_zero):
            return candidates
        
        rows = np.where(np.any(non_zero, axis=1))[0]
        cols = np.where(np.any(non_zero, axis=0))[0]
        
        seed_h = rows[-1] - rows[0] + 1
        seed_w = cols[-1] - cols[0] + 1
        grid_h, grid_w = inp.shape
        
        if seed_h < grid_h or seed_w < grid_w:
            candidates.append(ProgramNode(DSLElement("tile-seed", {"bg_color": 0})))
        
        return candidates

    def _gen_fill_by_period(self, features: dict, demo_pairs: list) -> list[ProgramNode]:
        """Generate fill-by-period candidates with auto-detected periods."""
        candidates: list[ProgramNode] = []
        
        if not features.get("size_same", False):
            return candidates
        
        candidates.append(ProgramNode(DSLElement("fill-by-period", {
            "period_h": 0, "period_w": 0
        })))
        
        pairs = features.get("pairs", [])
        if pairs:
            inp = pairs[0]["input"]
            h, w = inp.shape
            
            for ph in range(1, min(h, 8) + 1):
                if h % ph == 0 and ph > 1:
                    for pw in range(1, min(w, 8) + 1):
                        if w % pw == 0 and pw > 1:
                            candidates.append(ProgramNode(DSLElement("fill-by-period", {
                                "period_h": ph, "period_w": pw
                            })))
        
        return candidates

    def _gen_crop_to_bbox(self, features: dict) -> list[ProgramNode]:
        """Generate crop-to-bbox and crop-to-largest-cc candidates for size-decrease tasks."""
        candidates: list[ProgramNode] = []
        
        out_h = features.get("out_h", 0)
        out_w = features.get("out_w", 0)
        in_h = features.get("in_h", 0)
        in_w = features.get("in_w", 0)
        
        if out_h < in_h or out_w < in_w:
            # crop-to-bbox: crop to bounding box of all non-zero cells
            candidates.append(ProgramNode(DSLElement("crop-to-bbox", {"bg_color": 0})))
            # crop-to-largest-cc: crop to largest connected component (color-independent)
            candidates.append(ProgramNode(DSLElement("crop-to-largest-cc", {"bg_color": 0})))
        
        return candidates

    def _gen_fill_empty_neighbor(self, features: dict) -> list[ProgramNode]:
        """Generate fill-empty-neighbor candidates."""
        candidates: list[ProgramNode] = []
        
        if not features.get("size_same", False):
            return candidates
        
        pairs = features.get("pairs", [])
        if pairs:
            inp = pairs[0]["input"]
            out = pairs[0]["output"]
            inp_zeros = int(np.sum(inp == 0))
            out_zeros = int(np.sum(out == 0))
            
            if out_zeros < inp_zeros:
                candidates.append(ProgramNode(DSLElement("fill-empty-neighbor", {"bg_color": 0})))
        
        return candidates

    def _gen_direct_map(self, features: dict, demo_pairs: list) -> list[ProgramNode]:
        """Generate direct-map candidates by learning transformation from demo pairs.
        
        Finds a period (ph, pw) such that (color, r%ph, c%pw) -> new_color
        is consistent across all demo pairs. This is the most powerful generator.
        
        v2.9.1: Also generates fill-by-position-map candidates that map
        (r%ph, c%pw) -> new_color for background cells only, handling
        tasks where different pairs have different color schemes.
        """
        candidates: list[ProgramNode] = []
        
        if not features.get("size_same", False):
            return candidates
        
        pairs = features.get("pairs", [])
        if len(pairs) < 1:
            return candidates
        
        first_pair = pairs[0]
        inp0 = first_pair["input"]
        h, w = inp0.shape
        
        max_period = min(h, w, 12)
        
        # Strategy 1: Full (color, r%ph, c%pw) -> new_color mapping
        for ph in range(1, max_period + 1):
            for pw in range(1, max_period + 1):
                mapping: dict[tuple, int] = {}
                consistent = True
                
                for p in pairs:
                    inp = p["input"]
                    out = p["output"]
                    if inp.shape != out.shape:
                        consistent = False
                        break
                    
                    for r in range(inp.shape[0]):
                        for c in range(inp.shape[1]):
                            key = (int(inp[r, c]), r % ph, c % pw)
                            expected = int(out[r, c])
                            if key in mapping:
                                if mapping[key] != expected:
                                    consistent = False
                                    break
                            else:
                                mapping[key] = expected
                        if not consistent:
                            break
                    if not consistent:
                        break
                
                if consistent and len(mapping) > 0:
                    has_change = False
                    for p in pairs:
                        inp = p["input"]
                        for r in range(inp.shape[0]):
                            for c in range(inp.shape[1]):
                                key = (int(inp[r, c]), r % ph, c % pw)
                                if key in mapping and mapping[key] != int(inp[r, c]):
                                    has_change = True
                                    break
                            if has_change:
                                break
                        if has_change:
                            break
                    
                    if has_change:
                        str_mapping = {f"{k[0]},{k[1]},{k[2]}": v for k, v in mapping.items()}
                        elem = DSLElement("direct-map", {
                            "mapping": str_mapping,
                            "period_h": ph,
                            "period_w": pw,
                        })
                        candidates.append(ProgramNode(elem))
        
        # Strategy 2: Position-only mapping for background (0) cells
        # (r%ph, c%pw) -> new_color, only for cells where input=0
        # This handles tasks where different pairs have different color schemes
        for ph in range(1, max_period + 1):
            for pw in range(1, max_period + 1):
                pos_mapping: dict[tuple, int] = {}
                consistent = True
                
                for p in pairs:
                    inp = p["input"]
                    out = p["output"]
                    if inp.shape != out.shape:
                        consistent = False
                        break
                    
                    for r in range(inp.shape[0]):
                        for c in range(inp.shape[1]):
                            if inp[r, c] == 0:  # Only background cells
                                key = (r % ph, c % pw)
                                expected = int(out[r, c])
                                if key in pos_mapping:
                                    if pos_mapping[key] != expected:
                                        consistent = False
                                        break
                                else:
                                    pos_mapping[key] = expected
                        if not consistent:
                            break
                    if not consistent:
                        break
                
                if consistent and len(pos_mapping) > 0:
                    # Check that this mapping actually changes something
                    has_change = any(v != 0 for v in pos_mapping.values())
                    if has_change:
                        # Create a fill-by-period candidate with this period
                        # The fill-by-period primitive will use the detected pattern
                        elem = DSLElement("fill-by-period", {
                            "period_h": ph,
                            "period_w": pw,
                        })
                        candidates.append(ProgramNode(elem))
                        
                        # Also create a direct-map variant
                        # Build full mapping: non-zero cells keep their color,
                        # zero cells get filled by position mapping
                        full_mapping = {}
                        for p in pairs:
                            inp = p["input"]
                            out = p["output"]
                            for r in range(inp.shape[0]):
                                for c in range(inp.shape[1]):
                                    if inp[r, c] == 0:
                                        key_str = f"0,{r % ph},{c % pw}"
                                        full_mapping[key_str] = int(out[r, c])
                        
                        if full_mapping:
                            elem2 = DSLElement("direct-map", {
                                "mapping": full_mapping,
                                "period_h": ph,
                                "period_w": pw,
                            })
                            candidates.append(ProgramNode(elem2))
        
        return candidates

    def _gen_repeat_grid(self, features: dict) -> list[ProgramNode]:
        """Generate repeat-grid candidates for size-increase tasks."""
        candidates: list[ProgramNode] = []
        
        out_h = features.get("out_h", 0)
        out_w = features.get("out_w", 0)
        in_h = features.get("in_h", 0)
        in_w = features.get("in_w", 0)
        
        if out_h > in_h and out_w > in_w:
            fh = out_h / in_h if in_h > 0 else 0
            fw = out_w / in_w if in_w > 0 else 0
            
            if abs(fh - round(fh)) < 0.01 and abs(fw - round(fw)) < 0.01:
                fh_int = int(round(fh))
                fw_int = int(round(fw))
                if fh_int > 1 or fw_int > 1:
                    candidates.append(ProgramNode(DSLElement("repeat-grid", {
                        "factor_h": fh_int, "factor_w": fw_int
                    })))
        
        return candidates

    # ============================================================
    # v3.0: High-impact inference methods for 68% accuracy target
    # ============================================================

    def _gen_universal_color_map(
        self, features: dict, demo_pairs: list
    ) -> list[ProgramNode]:
        """Generate universal-color-map candidates with cross-pair consistency.

        Learns input_color -> output_color mapping from ALL demo pairs.
        Only generates a candidate if the mapping is consistent across all pairs.
        Handles multi-to-one mappings (multiple input colors -> same output color).
        """
        candidates: list[ProgramNode] = []
        pairs = features.get("pairs", [])
        if len(pairs) < 1:
            return candidates

        if not features.get("size_same", False):
            return candidates

        # Build universal color mapping from all pairs
        merged_map: dict[int, int] = {}
        compatible = True

        for p in pairs:
            inp = p["input"]
            out = p["output"]
            if inp.shape != out.shape:
                compatible = False
                break

            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    old_c = int(inp[r, c])
                    new_c = int(out[r, c])
                    if old_c != new_c:
                        if old_c in merged_map:
                            if merged_map[old_c] != new_c:
                                compatible = False
                                break
                        else:
                            merged_map[old_c] = new_c
                if not compatible:
                    break
            if not compatible:
                break

        if compatible and len(merged_map) > 0:
            # Generate the full universal color map
            mapping_str = {str(k): v for k, v in merged_map.items()}
            candidates.append(
                ProgramNode(DSLElement("universal-color-map", {"mapping": mapping_str}))
            )

            # Also try map-color with the same mapping (as fallback)
            candidates.append(
                ProgramNode(DSLElement("map-color", {"mapping": dict(merged_map)}))
            )

        # Also try per-pair color mapping (for tasks where colors differ between pairs)
        # but the mapping RULE is the same (e.g., "all non-zero -> color X")
        if not compatible:
            # Try to find a consistent rule: "all non-zero -> most common output color"
            for p in pairs:
                inp = p["input"]
                out = p["output"]
                if inp.shape != out.shape:
                    continue

                inp_nz = inp[inp != 0]
                out_nz = out[out != 0]
                if len(inp_nz) == 0 or len(out_nz) == 0:
                    continue

                # Check if all non-zero input maps to a single output color
                out_colors_for_nz = set()
                for r in range(inp.shape[0]):
                    for c in range(inp.shape[1]):
                        if inp[r, c] != 0:
                            out_colors_for_nz.add(int(out[r, c]))

                if len(out_colors_for_nz) == 1:
                    target = out_colors_for_nz.pop()
                    # Check consistency across all pairs
                    all_match = True
                    for p2 in pairs:
                        inp2 = p2["input"]
                        out2 = p2["output"]
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        for r in range(inp2.shape[0]):
                            for c in range(inp2.shape[1]):
                                if inp2[r, c] != 0 and int(out2[r, c]) != target:
                                    all_match = False
                                    break
                            if not all_match:
                                break
                        if not all_match:
                            break

                    if all_match:
                        # Generate mapping: all input non-zero colors -> target
                        all_in_colors = set()
                        for p2 in pairs:
                            inp2 = p2["input"]
                            all_in_colors.update(int(x) for x in np.unique(inp2) if x != 0)
                        
                        rule_map = {str(c): target for c in all_in_colors}
                        candidates.append(
                            ProgramNode(DSLElement("universal-color-map", {"mapping": rule_map}))
                        )
                        break

        return candidates

    def _gen_smart_flood_fill(
        self, features: dict, demo_pairs: list
    ) -> list[ProgramNode]:
        """Generate smart-flood-fill candidates with multiple variants.

        Tries different connectivity (4/8), fill targets (enclosed/border/all_bg),
        and auto-infers fill color from output.
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        pairs = features.get("pairs", [])
        if not pairs:
            return candidates

        # Infer fill colors from output
        fill_colors: set[int] = set()
        for p in pairs:
            inp = p["input"]
            out = p["output"]
            if inp.shape != out.shape:
                continue
            # Colors that appear in output but not input (new colors = fill candidates)
            new_colors = set(out.flatten()) - set(inp.flatten())
            fill_colors.update(int(c) for c in new_colors if c != 0)
            # Also try colors that increase in count
            for c in range(1, 10):
                inp_count = int(np.sum(inp == c))
                out_count = int(np.sum(out == c))
                if out_count > inp_count:
                    fill_colors.add(c)

        if not fill_colors:
            fill_colors = {1, 2, 3}

        # Generate variants
        for color in sorted(fill_colors)[:5]:  # limit to top 5 colors
            for connectivity in [4, 8]:
                for fill_target in ["enclosed", "border", "all_bg"]:
                    candidates.append(
                        ProgramNode(DSLElement("smart-flood-fill", {
                            "color": color,
                            "connectivity": connectivity,
                            "fill_target": fill_target,
                        }))
                    )

        # Also generate fill-enclosed-regions candidates
        for color in sorted(fill_colors)[:5]:
            candidates.append(
                ProgramNode(DSLElement("fill-enclosed-regions", {
                    "color": color, "bg_color": 0
                }))
            )

        return candidates

    def _gen_grid_diff_apply(
        self, features: dict, demo_pairs: list
    ) -> list[ProgramNode]:
        """Generate grid-diff-apply candidates by learning diff patterns.

        For same-size tasks, computes the difference between output and input
        and tries to find a consistent diff pattern (border, frame, etc.).
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        pairs = features.get("pairs", [])
        if not pairs:
            return candidates

        # Try all diff types with colors from output
        out_colors: set[int] = set()
        for p in pairs:
            out_colors.update(int(c) for c in np.unique(p["output"]) if c != 0)

        if not out_colors:
            out_colors = {1, 2}

        for color in sorted(out_colors)[:4]:
            for diff_type in ["border", "frame", "corners", "extend_lines", "fill_adjacent"]:
                candidates.append(
                    ProgramNode(DSLElement("grid-diff-apply", {
                        "diff_type": diff_type, "color": color
                    }))
                )

        return candidates

    def _gen_object_transform(
        self, features: dict, demo_pairs: list
    ) -> list[ProgramNode]:
        """Generate object-level transform candidates.

        Includes extract-and-recolor, move-object-to-pos, count-objects-mark.
        """
        candidates: list[ProgramNode] = []

        pairs = features.get("pairs", [])
        if not pairs:
            return candidates

        # Extract and recolor: try all color pairs that change
        if features.get("size_same", False):
            changed_colors: set[int] = set()
            for p in pairs:
                inp = p["input"]
                out = p["output"]
                if inp.shape != out.shape:
                    continue
                diff_mask = inp != out
                changed_colors.update(int(c) for c in inp[diff_mask])

            all_colors = sorted(features.get("all_colors", set()))
            for src_c in sorted(changed_colors)[:5]:
                for tgt_c in all_colors:
                    if src_c != tgt_c:
                        candidates.append(
                            ProgramNode(DSLElement("extract-and-recolor", {
                                "source_color": src_c,
                                "target_color": tgt_c,
                                "keep_others": True,
                            }))
                        )

        # Move object to position: try moving objects to corners
        if features.get("size_same", False):
            for color in sorted(features.get("all_colors", {1}))[:3]:
                if color == 0:
                    continue
                for tr in [0, 1]:
                    for tc in [0, 1]:
                        candidates.append(
                            ProgramNode(DSLElement("move-object-to-pos", {
                                "color": color,
                                "target_row": tr,
                                "target_col": tc,
                            }))
                        )

        return candidates

    def _gen_conditional_map(
        self, features: dict, demo_pairs: list
    ) -> list[ProgramNode]:
        """Generate conditional-map candidates by mining context rules.

        Learns rules like: if input[r,c]==A and neighbor==B: output[r,c]=C.
        """
        candidates: list[ProgramNode] = []
        pairs = features.get("pairs", [])

        if not features.get("size_same", False) or len(pairs) < 1:
            return candidates

        # Mine conditional rules from first pair
        inp = pairs[0]["input"]
        out = pairs[0]["output"]
        h, w = inp.shape

        # Find pixels that change
        changed = inp != out
        if not np.any(changed):
            return candidates

        # For each changed pixel, check neighbor context
        rules: set[tuple] = set()
        for r in range(h):
            for c in range(w):
                if not changed[r, c]:
                    continue
                center_c = int(inp[r, c])
                output_c = int(out[r, c])
                if center_c == output_c:
                    continue

                # Check all 4 neighbors
                for direction, (dr, dc) in [("up", (-1, 0)), ("down", (1, 0)),
                                             ("left", (0, -1)), ("right", (0, 1))]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbor_c = int(inp[nr, nc])
                        if neighbor_c != center_c:
                            rules.add((center_c, neighbor_c, direction, output_c))

                # Also check "any" direction
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbor_c = int(inp[nr, nc])
                        if neighbor_c != center_c:
                            rules.add((center_c, neighbor_c, "any", output_c))
                            break

        # Generate candidates from mined rules (limit to top 10)
        rule_list = sorted(rules)[:10]
        if rule_list:
            rule_dicts = [
                {"center": r[0], "neighbor": r[1], "direction": r[2], "output": r[3]}
                for r in rule_list
            ]
            # Single rule
            for rd in rule_dicts:
                candidates.append(
                    ProgramNode(DSLElement("conditional-map", {"rules": [rd]}))
                )
            # All rules combined
            candidates.append(
                ProgramNode(DSLElement("conditional-map", {"rules": rule_dicts}))
            )

        return candidates

    def _gen_nearest_neighbor_transform(
        self, features: dict, demo_pairs: list
    ) -> list[ProgramNode]:
        """Generate nearest-neighbor-transform candidates.

        For test input, find most similar training input and apply same transform.
        """
        candidates: list[ProgramNode] = []

        pairs = features.get("pairs", [])
        if len(pairs) < 1:
            return candidates

        # Build train_data for the primitive
        train_data = []
        for p in pairs:
            train_data.append([p["input"].tolist(), p["output"].tolist()])

        candidates.append(
            ProgramNode(DSLElement("nearest-neighbor-transform", {
                "train_data": train_data
            }))
        )

        return candidates

    def _gen_replace_marker_with_border(
        self, features: dict, demo_pairs: list
    ) -> list[ProgramNode]:
        """Generate replace-marker-with-border candidates.

        Detects a marker color (color that appears in input but is replaced
        in output) and generates candidates to replace it with border colors.
        """
        candidates: list[ProgramNode] = []

        pairs = features.get("pairs", [])
        if not features.get("size_same", False) or not pairs:
            return candidates

        # Find potential marker colors: colors that are replaced by different colors
        # in different positions (indicating position-dependent replacement)
        for p in pairs:
            inp = p["input"]
            out = p["output"]
            if inp.shape != out.shape:
                continue

            for marker_c in range(1, 10):
                marker_mask = inp == marker_c
                if not np.any(marker_mask):
                    continue

                # Check if marker pixels are replaced by different colors
                out_at_marker = out[marker_mask]
                unique_out = set(out_at_marker.tolist())
                if len(unique_out) > 1:
                    # This is a marker color that gets replaced differently
                    candidates.append(
                        ProgramNode(DSLElement("replace-marker-with-border", {
                            "marker_color": marker_c
                        }))
                    )

        # Also try common marker colors (3 is common in ARC)
        for mc in [3, 5, 2, 4, 6]:
            candidates.append(
                ProgramNode(DSLElement("replace-marker-with-border", {
                    "marker_color": mc
                }))
            )

        return candidates

    def _gen_gravity_stack(self, features: dict) -> list[ProgramNode]:
        """Generate gravity-stack candidates for all directions."""
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        for direction in ["down", "up", "left", "right"]:
            candidates.append(
                ProgramNode(DSLElement("gravity-stack", {"direction": direction}))
            )

        return candidates

    def _gen_draw_frame(self, features: dict) -> list[ProgramNode]:
        """Generate draw-frame-around-objects candidates."""
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        pairs = features.get("pairs", [])
        if not pairs:
            return candidates

        # Try frame colors from output
        out_colors: set[int] = set()
        for p in pairs:
            out_colors.update(int(c) for c in np.unique(p["output"]) if c != 0)

        for color in sorted(out_colors)[:4]:
            candidates.append(
                ProgramNode(DSLElement("draw-frame-around-objects", {
                    "frame_color": color
                }))
            )

        # Default frame color
        candidates.append(
            ProgramNode(DSLElement("draw-frame-around-objects", {"frame_color": 1}))
        )

        return candidates

    def _gen_sort_objects(self, features: dict) -> list[ProgramNode]:
        """Generate sort-objects-by-size candidates."""
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        candidates.append(
            ProgramNode(DSLElement("sort-objects-by-size", {
                "ascending": True, "start_color": 1
            }))
        )
        candidates.append(
            ProgramNode(DSLElement("sort-objects-by-size", {
                "ascending": False, "start_color": 1
            }))
        )

        return candidates

    def _gen_propagate_color(self, features: dict) -> list[ProgramNode]:
        """Generate propagate-color candidates."""
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        pairs = features.get("pairs", [])
        if not pairs:
            return candidates

        # Try propagating each non-zero color from input
        inp_colors: set[int] = set()
        for p in pairs:
            inp_colors.update(int(c) for c in np.unique(p["input"]) if c != 0)

        for src_c in sorted(inp_colors)[:5]:
            candidates.append(
                ProgramNode(DSLElement("propagate-color", {
                    "source_color": src_c,
                    "target_color": 0
                }))
            )

        return candidates
