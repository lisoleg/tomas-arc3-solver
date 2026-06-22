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

        # Generate two-primitive chains (depth 2) for common combos
        candidates.extend(self._gen_chain_candidates(features))

        # Deduplicate by program signature
        seen: set[str] = set()
        unique: list[ProgramNode] = []
        for node in candidates:
            sig = self._node_signature(node)
            if sig not in seen:
                seen.add(sig)
                unique.append(node)

        return unique

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

        v2.5.1: Try ALL single-color mappings (color X → color Y) for
                 every pair of colors present in the grids, not just the
                 inferred consistent mapping. This covers tasks where a
                 single color is replaced.

        Args:
            features: Extracted features.

        Returns:
            List of ProgramNode candidates.
        """
        candidates: list[ProgramNode] = []

        if not features.get("size_same", False):
            return candidates

        # Consistent color mapping across all pairs
        if features.get("color_map", {}):
            mapping = features.get("color_map", {})
            if mapping:
                # Always generate the inferred mapping as a candidate
                # (even if not fully consistent — let Phase B decide)
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

        # Try all pairwise color swaps from present colors
        all_colors = sorted(features.get("all_colors", set()))
        for i in range(len(all_colors)):
            for j in range(i + 1, min(len(all_colors), 10)):
                ca, cb = all_colors[i], all_colors[j]
                candidates.append(
                    ProgramNode(
                        DSLElement("color-swap", {"color_a": ca, "color_b": cb})
                    )
                )

        # v2.5.1: Try ALL single-color mappings (X → Y)
        # This covers tasks like "replace all red with blue"
        for old_c in all_colors:
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
