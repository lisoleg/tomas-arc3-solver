"""Direct Solver for ARC tasks — bypass candidate search with targeted analysis.

Instead of generating hundreds of candidates and hoping one wins,
this module analyzes each task's transformation directly from demo pairs
and applies the exact transformation to the test input.

This implements the L4 Bayesian fusion approach: each sub-solver provides
a prediction with a confidence score, and the meta-solver picks the best.

Solving strategies:
1. Exact color map (same-size, consistent color mapping)
2. Pattern tile fill (background filled with repeating pattern)
3. Symmetry completion (complete symmetric patterns)
4. Marker replacement (replace markers with border/region colors)
5. Object shift (shift/move objects by inferred offset)
6. Plus/cross marker (draw plus patterns around markers)
7. Size decrease (crop/extract region or object)
8. Size increase (tile/repeat grid)
9. Grid diff apply (apply consistent pixel-level diff)
10. Nearest-neighbor transform (find most similar training pair)
11. Draw frame/border around objects
12. Fill enclosed regions

Performance target: 68%+ accuracy on 50 real ARC tasks.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy import ndimage


class DirectSolver:
    """Direct task solver that analyzes transformations from demo pairs.

    For each task, tries multiple solving strategies and returns the
    prediction with the highest confidence score.
    """

    def __init__(self) -> None:
        """Initialize the direct solver."""
        pass

    def solve(self, task: dict[str, Any]) -> tuple[np.ndarray, float, str]:
        """Solve an ARC task by analyzing demo pairs.

        Args:
            task: Task dict with 'train' and 'test' keys.

        Returns:
            Tuple of (prediction, confidence, solver_name).
        """
        train = task.get("train", [])
        test = task.get("test", [])
        if not train or not test:
            return np.array([[0]], dtype=np.int8), 0.0, "none"

        # Extract demo pairs
        demo_pairs = []
        for pair in train:
            inp = np.array(pair["input"], dtype=np.int8)
            out = np.array(pair["output"], dtype=np.int8)
            if inp.ndim == 3:
                inp = inp[0]
            if out.ndim == 3:
                out = out[0]
            demo_pairs.append((inp, out))

        test_input = np.array(test[0]["input"], dtype=np.int8)
        if test_input.ndim == 3:
            test_input = test_input[0]

        # Try each solver, collect results
        results: list[tuple[np.ndarray, float, str]] = []

        for solver_fn in [
            self._solve_color_map,
            self._solve_pattern_tile,
            self._solve_symmetry_complete,
            self._solve_marker_replace,
            self._solve_object_shift,
            self._solve_plus_marker,
            self._solve_size_decrease,
            self._solve_size_increase,
            self._solve_grid_diff,
            self._solve_nearest_neighbor,
            self._solve_draw_frame,
            self._solve_fill_enclosed,
            self._solve_complete_shape_symmetry,
            self._solve_color_count_remap,
        ]:
            try:
                pred, conf, name = solver_fn(demo_pairs, test_input)
                if pred is not None and conf > 0:
                    # Verify: does this prediction match all demo pairs?
                    verify_score = self._verify(demo_pairs, solver_fn, test_input)
                    if verify_score > 0:
                        results.append((pred, verify_score * conf, name))
            except Exception:
                pass

        if not results:
            # Fallback: return test input unchanged
            return test_input, 0.0, "fallback"

        # Pick best by confidence
        results.sort(key=lambda x: -x[1])
        return results[0]

    def _verify(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        solver_fn: Any,
        test_input: np.ndarray,
    ) -> float:
        """Verify a solver by applying it to each demo pair's input.

        Returns average similarity score (0-1).
        """
        scores = []
        for inp, expected in demo_pairs:
            try:
                pred, _, _ = solver_fn(demo_pairs, inp)
                if pred is not None and pred.shape == expected.shape:
                    score = np.sum(pred == expected) / expected.size
                    scores.append(score)
                elif pred is not None:
                    # Partial credit for overlapping region
                    h = min(pred.shape[0], expected.shape[0])
                    w = min(pred.shape[1], expected.shape[1])
                    if h > 0 and w > 0:
                        score = np.sum(pred[:h, :w] == expected[:h, :w]) / (h * w)
                        scores.append(score * 0.5)
            except Exception:
                pass
        return np.mean(scores) if scores else 0.0

    # ========== Solver 1: Exact Color Map ==========

    def _solve_color_map(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Solve tasks where colors change but positions don't.

        Computes exact color mapping from demo pairs. If mapping is
        consistent across all pairs, applies it to test input.
        """
        # Check all pairs are same-size
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "color_map"

        # Build color map from all pairs
        color_map: dict[int, int] = {}
        consistent = True

        for inp, out in demo_pairs:
            for c in range(10):
                mask = inp == c
                if mask.any():
                    out_vals = out[mask]
                    unique_out = set(out_vals.tolist())
                    if len(unique_out) == 1:
                        target = int(unique_out.pop())
                        if c in color_map:
                            if color_map[c] != target:
                                consistent = False
                        else:
                            color_map[c] = target
                    # If maps to multiple values, skip (position-dependent)

        if not consistent or not color_map:
            return None, 0.0, "color_map"

        # Apply to test
        result = test_input.copy()
        for c, target in color_map.items():
            result[test_input == c] = target

        # Confidence: fraction of pixels that have a mapping
        mapped_pixels = sum(
            np.sum(test_input == c) for c in color_map
        )
        confidence = mapped_pixels / test_input.size

        return result, confidence, "color_map"

    # ========== Solver 2: Pattern Tile Fill ==========

    def _solve_pattern_tile(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Solve tasks where background is filled with a repeating pattern.

        Detects the repeating pattern from demo pairs and fills
        background (0) pixels in the test input.
        """
        best_pred = None
        best_conf = 0.0

        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue

            # Find the non-zero region in the input (seed pattern)
            nonzero_mask = inp != 0
            if not nonzero_mask.any():
                continue

            rows = np.any(nonzero_mask, axis=1)
            cols = np.any(nonzero_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            seed = inp[rmin : rmax + 1, cmin : cmax + 1]
            sh, sw = seed.shape

            if sh == 0 or sw == 0:
                continue

            # Try to tile the seed to fill the entire grid
            result = np.zeros_like(out)
            for r in range(out.shape[0]):
                for c in range(out.shape[1]):
                    result[r, c] = seed[r % sh, c % sw]

            # Check if this matches the output
            if np.array_equal(result, out):
                # Apply to test
                test_result = np.zeros_like(test_input)
                for r in range(test_input.shape[0]):
                    for c in range(test_input.shape[1]):
                        test_result[r, c] = seed[r % sh, c % sw]

                return test_result, 1.0, "pattern_tile"

            # Also try: fill only background (0) pixels with tiled pattern
            result_bg = inp.copy()
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if inp[r, c] == 0:
                        result_bg[r, c] = seed[r % sh, c % sw]

            if np.array_equal(result_bg, out):
                test_result = test_input.copy()
                for r in range(test_input.shape[0]):
                    for c in range(test_input.shape[1]):
                        if test_input[r, c] == 0:
                            test_result[r, c] = seed[r % sh, c % sw]

                return test_result, 1.0, "pattern_tile_bg"

        return best_pred, best_conf, "pattern_tile"

    # ========== Solver 3: Symmetry Completion ==========

    def _solve_symmetry_complete(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Complete symmetric patterns.

        Detects symmetry axes and fills missing pixels to complete symmetry.
        Supports horizontal, vertical, and diagonal symmetry.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue

            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue

            # Try horizontal symmetry (mirror left-right)
            result_h = inp.copy()
            h, w = inp.shape
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == 0 and inp[r, w - 1 - c] != 0:
                        result_h[r, c] = inp[r, w - 1 - c]

            if np.array_equal(result_h, out):
                test_result = test_input.copy()
                th, tw = test_input.shape
                for r in range(th):
                    for c in range(tw):
                        if test_input[r, c] == 0 and test_input[r, tw - 1 - c] != 0:
                            test_result[r, c] = test_input[r, tw - 1 - c]
                return test_result, 1.0, "symmetry_h"

            # Try vertical symmetry (mirror top-bottom)
            result_v = inp.copy()
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == 0 and inp[h - 1 - r, c] != 0:
                        result_v[r, c] = inp[h - 1 - r, c]

            if np.array_equal(result_v, out):
                test_result = test_input.copy()
                th, tw = test_input.shape
                for r in range(th):
                    for c in range(tw):
                        if test_input[r, c] == 0 and test_input[th - 1 - r, c] != 0:
                            test_result[r, c] = test_input[th - 1 - r, c]
                return test_result, 1.0, "symmetry_v"

            # Try point symmetry (180 degree rotation)
            result_p = inp.copy()
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == 0 and inp[h - 1 - r, w - 1 - c] != 0:
                        result_p[r, c] = inp[h - 1 - r, w - 1 - c]

            if np.array_equal(result_p, out):
                test_result = test_input.copy()
                th, tw = test_input.shape
                for r in range(th):
                    for c in range(tw):
                        if test_input[r, c] == 0 and test_input[th - 1 - r, tw - 1 - c] != 0:
                            test_result[r, c] = test_input[th - 1 - r, tw - 1 - c]
                return test_result, 1.0, "symmetry_point"

            # Try block symmetry: complete a pattern block
            # Find the bbox of non-zero in input
            nonzero = np.argwhere(inp != 0)
            if len(nonzero) > 0:
                rmin, cmin = nonzero.min(axis=0)
                rmax, cmax = nonzero.max(axis=0)
                bh = rmax - rmin + 1
                bw = cmax - cmin + 1

                # Check if output is a tiling of this block
                block = inp[rmin : rmax + 1, cmin : cmax + 1]
                tiled = np.zeros_like(out)
                for r in range(out.shape[0]):
                    for c in range(out.shape[1]):
                        tiled[r, c] = block[(r - rmin) % bh, (c - cmin) % bw]

                if np.array_equal(tiled, out):
                    test_result = np.zeros_like(test_input)
                    for r in range(test_input.shape[0]):
                        for c in range(test_input.shape[1]):
                            test_result[r, c] = block[(r - rmin) % bh, (c - cmin) % bw]
                    return test_result, 1.0, "symmetry_block"

        return None, 0.0, "symmetry"

    # ========== Solver 4: Marker Replace with Border ==========

    def _solve_marker_replace(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Replace marker pixels with nearest border color.

        Detects border colors and marker color (the color that changes
        in demo pairs), then replaces markers with nearest border color.
        """
        # Find the marker color: the color that disappears in output
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            in_colors = set(np.unique(inp).tolist())
            out_colors = set(np.unique(out).tolist())
            # Marker is a color that exists in input but not output
            # (or significantly decreases)
            removed = in_colors - out_colors
            if not removed:
                # Also check: color that changes to different colors
                for c in in_colors:
                    if c == 0:
                        continue
                    mask = inp == c
                    if mask.any():
                        out_vals = set(out[mask].tolist())
                        if len(out_vals) > 1:
                            removed.add(c)

            for marker_color in removed:
                if marker_color == 0:
                    continue

                # Check if this is a border-replace pattern
                result = inp.copy()
                h, w = inp.shape

                # Detect border colors
                borders: dict[str, int] = {}
                top_nz = inp[0, :][inp[0, :] != 0]
                bot_nz = inp[-1, :][inp[-1, :] != 0]
                left_nz = inp[:, 0][inp[:, 0] != 0]
                right_nz = inp[:, -1][inp[:, -1] != 0]

                if len(set(top_nz.tolist())) == 1 and len(top_nz) > 0:
                    borders["top"] = int(top_nz[0])
                if len(set(bot_nz.tolist())) == 1 and len(bot_nz) > 0:
                    borders["bottom"] = int(bot_nz[0])
                if len(set(left_nz.tolist())) == 1 and len(left_nz) > 0:
                    borders["left"] = int(left_nz[0])
                if len(set(right_nz.tolist())) == 1 and len(right_nz) > 0:
                    borders["right"] = int(right_nz[0])

                if not borders:
                    continue

                # Apply: replace markers with nearest border
                for r in range(h):
                    for c in range(w):
                        if inp[r, c] == marker_color:
                            min_dist = float("inf")
                            nearest = marker_color
                            for bname, bc in borders.items():
                                if bname == "top":
                                    dist = r
                                elif bname == "bottom":
                                    dist = h - 1 - r
                                elif bname == "left":
                                    dist = c
                                elif bname == "right":
                                    dist = w - 1 - c
                                else:
                                    continue
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest = bc
                            result[r, c] = nearest

                if np.array_equal(result, out):
                    # Apply to test
                    test_result = test_input.copy()
                    th, tw = test_input.shape

                    test_borders: dict[str, int] = {}
                    top_nz = test_input[0, :][test_input[0, :] != 0]
                    bot_nz = test_input[-1, :][test_input[-1, :] != 0]
                    left_nz = test_input[:, 0][test_input[:, 0] != 0]
                    right_nz = test_input[:, -1][test_input[:, -1] != 0]

                    if len(set(top_nz.tolist())) == 1 and len(top_nz) > 0:
                        test_borders["top"] = int(top_nz[0])
                    if len(set(bot_nz.tolist())) == 1 and len(bot_nz) > 0:
                        test_borders["bottom"] = int(bot_nz[0])
                    if len(set(left_nz.tolist())) == 1 and len(left_nz) > 0:
                        test_borders["left"] = int(left_nz[0])
                    if len(set(right_nz.tolist())) == 1 and len(right_nz) > 0:
                        test_borders["right"] = int(right_nz[0])

                    for r in range(th):
                        for c in range(tw):
                            if test_input[r, c] == marker_color:
                                min_dist = float("inf")
                                nearest = marker_color
                                for bname, bc in test_borders.items():
                                    if bname == "top":
                                        dist = r
                                    elif bname == "bottom":
                                        dist = th - 1 - r
                                    elif bname == "left":
                                        dist = c
                                    elif bname == "right":
                                        dist = tw - 1 - c
                                    if dist < min_dist:
                                        min_dist = dist
                                        nearest = bc
                                test_result[r, c] = nearest

                    return test_result, 1.0, "marker_replace"

        return None, 0.0, "marker_replace"

    # ========== Solver 5: Object Shift ==========

    def _solve_object_shift(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Shift/move objects by inferred offset.

        Detects the shift offset from demo pairs and applies it to test.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue

            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue

            # Try to find a consistent shift
            # For each changed pixel, see if the output value came from
            # a different position in the input
            best_shift = None
            best_match = 0

            for dr in range(-inp.shape[0] + 1, inp.shape[0]):
                for dc in range(-inp.shape[1] + 1, inp.shape[1]):
                    if dr == 0 and dc == 0:
                        continue
                    # Shift input by (dr, dc)
                    shifted = np.zeros_like(inp)
                    for r in range(inp.shape[0]):
                        for c in range(inp.shape[1]):
                            sr, sc = r - dr, c - dc
                            if 0 <= sr < inp.shape[0] and 0 <= sc < inp.shape[1]:
                                shifted[r, c] = inp[sr, sc]

                    match = np.sum(shifted == out)
                    if match > best_match:
                        best_match = match
                        best_shift = (dr, dc)

            if best_shift and best_match / out.size > 0.9:
                dr, dc = best_shift
                # Verify with all pairs
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    shifted = np.zeros_like(inp2)
                    for r in range(inp2.shape[0]):
                        for c in range(inp2.shape[1]):
                            sr, sc = r - dr, c - dc
                            if 0 <= sr < inp2.shape[0] and 0 <= sc < inp2.shape[1]:
                                shifted[r, c] = inp2[sr, sc]
                    if not np.array_equal(shifted, out2):
                        all_match = False
                        break

                if all_match:
                    test_result = np.zeros_like(test_input)
                    for r in range(test_input.shape[0]):
                        for c in range(test_input.shape[1]):
                            sr, sc = r - dr, c - dc
                            if 0 <= sr < test_input.shape[0] and 0 <= sc < test_input.shape[1]:
                                test_result[r, c] = test_input[sr, sc]
                    return test_result, 1.0, "object_shift"

        return None, 0.0, "object_shift"

    # ========== Solver 6: Plus/Cross Marker ==========

    def _solve_plus_marker(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Draw plus/cross patterns around marker pixels.

        For each marker, draws a plus (up/down/left/right by 1) with
        an inferred color.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue

            diff = out.astype(int) - inp.astype(int)
            new_positions = np.argwhere((diff != 0) & (inp == 0))

            if len(new_positions) == 0:
                continue

            # Find marker positions (non-zero pixels that don't change)
            markers = []
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if inp[r, c] != 0 and out[r, c] == inp[r, c]:
                        markers.append((r, c, int(inp[r, c])))

            # For each new position, find the nearest marker
            # and check if it forms a plus pattern
            marker_plus_map: dict[int, int] = {}  # marker_color -> plus_color
            consistent = True

            for nr, nc in new_positions:
                plus_color = int(out[nr, nc])
                # Find nearest marker
                found = False
                for mr, mc, mc_color in markers:
                    # Check if (nr,nc) is adjacent to (mr,mc) in a plus pattern
                    if abs(nr - mr) + abs(nc - mc) == 1:
                        if mc_color not in marker_plus_map:
                            marker_plus_map[mc_color] = plus_color
                        elif marker_plus_map[mc_color] != plus_color:
                            consistent = False
                        found = True
                        break

                if not found:
                    # Maybe it's a diagonal/corner pattern
                    consistent = False

            if not consistent or not marker_plus_map:
                continue

            # Verify: apply plus pattern to all pairs
            all_match = True
            for inp2, out2 in demo_pairs:
                if inp2.shape != out2.shape:
                    all_match = False
                    break
                result = inp2.copy()
                for r in range(inp2.shape[0]):
                    for c in range(inp2.shape[1]):
                        pixel_color = int(inp2[r, c])
                        if pixel_color in marker_plus_map:
                            plus_c = marker_plus_map[pixel_color]
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < inp2.shape[0] and 0 <= nc < inp2.shape[1]:
                                    if inp2[nr, nc] == 0:
                                        result[nr, nc] = plus_c
                if not np.array_equal(result, out2):
                    all_match = False
                    break

            if all_match:
                # Apply to test
                test_result = test_input.copy()
                for r in range(test_input.shape[0]):
                    for c in range(test_input.shape[1]):
                        pixel_color = int(test_input[r, c])
                        if pixel_color in marker_plus_map:
                            plus_c = marker_plus_map[pixel_color]
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
                                    if test_input[nr, nc] == 0:
                                        test_result[nr, nc] = plus_c
                return test_result, 1.0, "plus_marker"

        return None, 0.0, "plus_marker"

    # ========== Solver 7: Size Decrease (Crop/Extract) ==========

    def _solve_size_decrease(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Solve tasks where output is smaller than input.

        Strategies:
        - Extract largest connected component
        - Crop to bounding box of non-zero
        - Extract specific pattern/quadrant
        """
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] and inp.shape[1] <= out.shape[1]:
                continue  # Not size decrease

            # Strategy 1: Extract largest CC
            binary = (inp != 0).astype(int)
            labeled, num = ndimage.label(binary)
            if num > 0:
                sizes = ndimage.sum(binary, labeled, range(1, num + 1))
                largest = np.argmax(sizes) + 1
                mask = labeled == largest
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                extracted = inp[rmin : rmax + 1, cmin : cmax + 1]

                if extracted.shape == out.shape and np.array_equal(extracted, out):
                    # Apply to test
                    test_binary = (test_input != 0).astype(int)
                    test_labeled, test_num = ndimage.label(test_binary)
                    if test_num > 0:
                        test_sizes = ndimage.sum(test_binary, test_labeled, range(1, test_num + 1))
                        test_largest = np.argmax(test_sizes) + 1
                        test_mask = test_labeled == test_largest
                        test_rows = np.any(test_mask, axis=1)
                        test_cols = np.any(test_mask, axis=0)
                        trmin, trmax = np.where(test_rows)[0][[0, -1]]
                        tcmin, tcmax = np.where(test_cols)[0][[0, -1]]
                        test_result = test_input[trmin : trmax + 1, tcmin : tcmax + 1]
                        return test_result, 1.0, "crop_largest_cc"

            # Strategy 2: Crop to non-zero bbox (all non-zero)
            nonzero = np.argwhere(inp != 0)
            if len(nonzero) > 0:
                rmin, cmin = nonzero.min(axis=0)
                rmax, cmax = nonzero.max(axis=0)
                cropped = inp[rmin : rmax + 1, cmin : cmax + 1]
                if cropped.shape == out.shape and np.array_equal(cropped, out):
                    test_nonzero = np.argwhere(test_input != 0)
                    if len(test_nonzero) > 0:
                        trmin, tcmin = test_nonzero.min(axis=0)
                        trmax, tcmax = test_nonzero.max(axis=0)
                        test_result = test_input[trmin : trmax + 1, tcmin : tcmax + 1]
                        return test_result, 1.0, "crop_bbox"

            # Strategy 3: Extract by specific color
            for color in range(1, 10):
                mask = inp == color
                if mask.any():
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=1)
                    if rows.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        extracted = inp[rmin : rmax + 1, cmin : cmax + 1]
                        # Check: output only has this color (or this color + background)
                        out_colors = set(np.unique(out).tolist())
                        if extracted.shape == out.shape and np.array_equal(extracted, out):
                            # Apply to test
                            test_mask = test_input == color
                            if test_mask.any():
                                test_rows = np.any(test_mask, axis=1)
                                test_cols = np.any(test_mask, axis=0)
                                trmin, trmax = np.where(test_rows)[0][[0, -1]]
                                tcmin, tcmax = np.where(test_cols)[0][[0, -1]]
                                test_result = test_input[trmin : trmax + 1, tcmin : tcmax + 1]
                                return test_result, 1.0, f"crop_color_{color}"

        return None, 0.0, "size_decrease"

    # ========== Solver 8: Size Increase (Tile/Repeat) ==========

    def _solve_size_increase(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Solve tasks where output is larger than input.

        Strategies:
        - Tile/repeat the input
        - Scale up by integer factor
        """
        for inp, out in demo_pairs:
            if inp.shape[0] >= out.shape[0] and inp.shape[1] >= out.shape[1]:
                continue  # Not size increase

            # Check integer scale factor
            rh = out.shape[0] / inp.shape[0]
            rw = out.shape[1] / inp.shape[1]

            if rh == rw and rh == int(rh) and rh > 1:
                factor = int(rh)
                # Scale by repetition
                scaled = np.repeat(np.repeat(inp, factor, axis=0), factor, axis=1)
                if scaled.shape == out.shape and np.array_equal(scaled, out):
                    test_scaled = np.repeat(np.repeat(test_input, factor, axis=0), factor, axis=1)
                    return test_scaled, 1.0, "scale_repeat"

            # Check tiling
            if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                th = out.shape[0] // inp.shape[0]
                tw = out.shape[1] // inp.shape[1]
                tiled = np.tile(inp, (th, tw))
                if tiled.shape == out.shape and np.array_equal(tiled, out):
                    test_tiled = np.tile(test_input, (th, tw))
                    return test_tiled, 1.0, "tile_repeat"

        return None, 0.0, "size_increase"

    # ========== Solver 9: Grid Diff Apply ==========

    def _solve_grid_diff(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Apply consistent pixel-level diff to test input.

        For same-size tasks, computes the diff (output - input) and
        checks if it's consistent across pairs. If so, applies it to test.
        """
        # All pairs must be same-size
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "grid_diff"

        # Check if diff is the same across all pairs
        diffs = []
        for inp, out in demo_pairs:
            diff = out.astype(int) - inp.astype(int)
            diffs.append(diff)

        # Check if all diffs are identical
        consistent = True
        for i in range(1, len(diffs)):
            if not np.array_equal(diffs[0], diffs[i]):
                consistent = False
                break

        if consistent and len(diffs) > 0:
            result = test_input.astype(int) + diffs[0]
            result = np.clip(result, 0, 9).astype(np.int8)
            return result, 1.0, "grid_diff"

        # Check if diff is position-dependent but follows a pattern
        # E.g., "add 1 to all non-zero pixels" or "set background to max color"
        # Strategy: for each position, if input[r,c] == X, output[r,c] = Y
        # Build a position-independent rule: input_value -> output_value
        # (Already handled by color_map solver)

        # Strategy: "fill 0 pixels with the most common non-zero color"
        for inp, out in demo_pairs:
            diff = out.astype(int) - inp.astype(int)
            changed_to_zero = np.argwhere((inp != 0) & (out == 0))
            changed_from_zero = np.argwhere((inp == 0) & (out != 0))

            if len(changed_to_zero) == 0 and len(changed_from_zero) > 0:
                # All changes are 0 -> something
                # Check if all changed pixels get the same color
                fill_colors = set(out[inp == 0].tolist()) - {0}
                if len(fill_colors) == 1:
                    fill_color = fill_colors.pop()
                    # Check consistency
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        result = inp2.copy()
                        result[inp2 == 0] = fill_color
                        if not np.array_equal(result, out2):
                            all_match = False
                            break
                    if all_match:
                        test_result = test_input.copy()
                        test_result[test_input == 0] = fill_color
                        return test_result, 1.0, "fill_bg"

        return None, 0.0, "grid_diff"

    # ========== Solver 10: Nearest Neighbor Transform ==========

    def _solve_nearest_neighbor(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Find most similar training input and apply same transformation.

        For test input, find the training input that's most similar
        (by pixel match), and apply the same transformation.
        """
        if not demo_pairs:
            return None, 0.0, "nearest_neighbor"

        # For same-size tasks: find the training pair with most similar input
        best_sim = 0
        best_pair = None

        for inp, out in demo_pairs:
            if inp.shape == test_input.shape:
                sim = np.sum(inp == test_input) / test_input.size
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (inp, out)

        if best_pair and best_sim > 0.5:
            # Apply the same color mapping
            inp, out = best_pair
            color_map: dict[int, int] = {}
            for c in range(10):
                mask = inp == c
                if mask.any():
                    out_vals = set(out[mask].tolist())
                    if len(out_vals) == 1:
                        color_map[c] = int(out_vals.pop())

            result = test_input.copy()
            for c, target in color_map.items():
                result[test_input == c] = target

            return result, best_sim * 0.8, "nearest_neighbor"

        return None, 0.0, "nearest_neighbor"

    # ========== Solver 11: Draw Frame Around Objects ==========

    def _solve_draw_frame(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Draw a frame/border around objects.

        Detects objects and draws a 1-pixel border around them.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue

            diff = out.astype(int) - inp.astype(int)
            new_pixels = np.argwhere((diff != 0) & (inp == 0))
            if len(new_pixels) == 0:
                continue

            # Check if new pixels form a border around existing objects
            # Dilate the non-zero mask and subtract to get border
            binary = (inp != 0).astype(int)
            dilated = ndimage.binary_dilation(binary, iterations=1).astype(int)
            border = dilated - binary

            if border.sum() == len(new_pixels):
                # Check if border pixels match output
                border_mask = border.astype(bool)
                if np.array_equal((out != 0) & border_mask, border_mask & (out != 0)):
                    # Find border color
                    border_colors = set(out[border_mask].tolist())
                    if len(border_colors) == 1:
                        bc = border_colors.pop()
                        # Verify all pairs
                        all_match = True
                        for inp2, out2 in demo_pairs:
                            if inp2.shape != out2.shape:
                                all_match = False
                                break
                            b2 = (inp2 != 0).astype(int)
                            d2 = ndimage.binary_dilation(b2, iterations=1).astype(int)
                            border2 = d2 - b2
                            result = inp2.copy()
                            result[border2.astype(bool)] = bc
                            if not np.array_equal(result, out2):
                                all_match = False
                                break

                        if all_match:
                            test_binary = (test_input != 0).astype(int)
                            test_dilated = ndimage.binary_dilation(test_binary, iterations=1).astype(int)
                            test_border = test_dilated - test_binary
                            test_result = test_input.copy()
                            test_result[test_border.astype(bool)] = bc
                            return test_result, 1.0, "draw_frame"

        return None, 0.0, "draw_frame"

    # ========== Solver 12: Fill Enclosed Regions ==========

    def _solve_fill_enclosed(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Fill enclosed (surrounded) background regions.

        Detects background regions that are fully enclosed by non-zero
        pixels and fills them with a specific color.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue

            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue

            # Check if changed pixels are all 0 -> X
            from_zero = np.argwhere((inp == 0) & (out != 0))
            if len(from_zero) != len(changed):
                continue

            fill_colors = set(out[from_zero[:, 0], from_zero[:, 1]].tolist())
            if len(fill_colors) != 1:
                continue

            fill_color = fill_colors.pop()

            # Check if these are enclosed regions
            binary = (inp != 0).astype(int)
            # Flood fill from borders
            from scipy.ndimage import binary_fill_holes
            filled = binary_fill_holes(binary)
            enclosed = filled & ~binary.astype(bool)

            if np.array_equal(enclosed[np.where(inp == 0)] != 0, (out[np.where(inp == 0)] != 0)):
                # Verify all pairs
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    b2 = (inp2 != 0).astype(int)
                    filled2 = binary_fill_holes(b2)
                    enclosed2 = filled2 & ~b2.astype(bool)
                    result = inp2.copy()
                    result[enclosed2] = fill_color
                    if not np.array_equal(result, out2):
                        all_match = False
                        break

                if all_match:
                    test_binary = (test_input != 0).astype(int)
                    test_filled = binary_fill_holes(test_binary)
                    test_enclosed = test_filled & ~test_binary.astype(bool)
                    test_result = test_input.copy()
                    test_result[test_enclosed] = fill_color
                    return test_result, 1.0, "fill_enclosed"

        return None, 0.0, "fill_enclosed"

    # ========== Solver 13: Complete Shape by Symmetry ==========

    def _solve_complete_shape_symmetry(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Complete shapes using various symmetry operations.

        More advanced than basic symmetry - tries multiple symmetry
        axes and combinations.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue

            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue

            # Try: copy from a reference position
            # The pattern might be: "for each 0 pixel, copy from the
            # nearest non-zero pixel in a specific direction"

            # Try 8 directions
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                result = inp.copy()
                for r in range(inp.shape[0]):
                    for c in range(inp.shape[1]):
                        if inp[r, c] == 0:
                            sr, sc = r + dr, c + dc
                            if 0 <= sr < inp.shape[0] and 0 <= sc < inp.shape[1]:
                                if inp[sr, sc] != 0:
                                    result[r, c] = inp[sr, sc]

                if np.array_equal(result, out):
                    # Verify all pairs
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        r2 = inp2.copy()
                        for r in range(inp2.shape[0]):
                            for c in range(inp2.shape[1]):
                                if inp2[r, c] == 0:
                                    sr, sc = r + dr, c + dc
                                    if 0 <= sr < inp2.shape[0] and 0 <= sc < inp2.shape[1]:
                                        if inp2[sr, sc] != 0:
                                            r2[r, c] = inp2[sr, sc]
                        if not np.array_equal(r2, out2):
                            all_match = False
                            break

                    if all_match:
                        test_result = test_input.copy()
                        for r in range(test_input.shape[0]):
                            for c in range(test_input.shape[1]):
                                if test_input[r, c] == 0:
                                    sr, sc = r + dr, c + dc
                                    if 0 <= sr < test_input.shape[0] and 0 <= sc < test_input.shape[1]:
                                        if test_input[sr, sc] != 0:
                                            test_result[r, c] = test_input[sr, sc]
                        return test_result, 1.0, f"shape_dir_{dr}_{dc}"

        return None, 0.0, "shape_symmetry"

    # ========== Solver 14: Color Count Remap ==========

    def _solve_color_count_remap(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> tuple[np.ndarray | None, float, str]:
        """Remap colors based on count/frequency.

        If the number of pixels of each color changes consistently,
        remap accordingly.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue

            # Check: does each color map to exactly one other color?
            color_map: dict[int, int] = {}
            consistent = True
            for c in range(10):
                mask = inp == c
                if mask.any():
                    out_vals = out[mask]
                    # Allow multiple output values if they're all the same
                    unique = np.unique(out_vals)
                    if len(unique) == 1:
                        color_map[c] = int(unique[0])
                    else:
                        # Try: color maps to the output color that appears
                        # most frequently at this input color's positions
                        # Actually, this is position-dependent, skip
                        pass

            if not consistent or not color_map:
                continue

            # Check: is there a "swap" pattern?
            # E.g., color A -> B and color B -> A
            swaps = {}
            for c, target in color_map.items():
                if target in color_map and color_map[target] == c and c != target:
                    swaps[c] = target

            if len(swaps) >= 2:
                # Verify swap pattern
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    result = inp2.copy()
                    for c, target in swaps.items():
                        result[inp2 == c] = target
                    if not np.array_equal(result, out2):
                        all_match = False
                        break

                if all_match:
                    test_result = test_input.copy()
                    for c, target in swaps.items():
                        test_result[test_input == c] = target
                    return test_result, 1.0, "color_swap"

        return None, 0.0, "color_count_remap"
