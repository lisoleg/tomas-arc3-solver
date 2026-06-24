"""Direct Solver v2 for ARC tasks — targeted analysis + Bayesian fusion.

Analyzes each task's transformation directly from demo pairs and applies
the exact transformation to the test input. This is the L4 Bayesian
fusion layer: each sub-solver provides a prediction with confidence,
and the meta-solver picks the best.

Key improvements over v1:
- Pattern tile: tries all period sizes from OUTPUT (handles diagonal wrap)
- Object shift: handles edge cases (off-grid pixels)
- Added gravity, scale-pattern, recolor-cc, flood-fill solvers
- Combined with DSL-based approach for maximum coverage
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy import ndimage


class DirectSolverV2:
    """Direct task solver v2 with improved analysis and Bayesian fusion."""

    def __init__(self) -> None:
        pass

    def solve(self, task: dict[str, Any]) -> tuple[np.ndarray, float, str]:
        """Solve an ARC task by analyzing demo pairs.

        Returns:
            Tuple of (prediction, confidence, solver_name).
        """
        train = task.get("train", [])
        test = task.get("test", [])
        if not train or not test:
            return np.array([[0]], dtype=np.int8), 0.0, "none"

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

        # Try each solver, collect verified results
        results: list[tuple[np.ndarray, float, str]] = []

        solvers = [
            self._solve_color_map,
            self._solve_pattern_tile,
            self._solve_symmetry_complete,
            self._solve_marker_replace,
            self._solve_object_shift,
            self._solve_plus_marker,
            self._solve_size_decrease,
            self._solve_size_increase,
            self._solve_grid_diff,
            self._solve_fill_enclosed,
            self._solve_shape_dir,
            self._solve_color_swap,
            self._solve_gravity,
            self._solve_scale_pattern,
            self._solve_recolor_cc,
            self._solve_flood_fill_smart,
            self._solve_nearest_neighbor,
            self._solve_draw_frame,
            self._solve_count_and_mark,
        ]

        for solver_fn in solvers:
            try:
                pred, conf, name = solver_fn(demo_pairs, test_input)
                if pred is not None and conf > 0:
                    # Verify on all demo pairs
                    verify = self._verify(demo_pairs, solver_fn)
                    if verify > 0.5:
                        results.append((pred, verify * conf, name))
            except Exception:
                pass

        if not results:
            return test_input.copy(), 0.0, "fallback"

        results.sort(key=lambda x: -x[1])
        return results[0]

    def _verify(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        solver_fn: Any,
    ) -> float:
        """Verify a solver by applying it to each demo pair's input."""
        scores = []
        for inp, expected in demo_pairs:
            try:
                pred, _, _ = solver_fn(demo_pairs, inp)
                if pred is not None and pred.shape == expected.shape:
                    score = float(np.sum(pred == expected) / expected.size)
                    scores.append(score)
            except Exception:
                pass
        return float(np.mean(scores)) if scores else 0.0

    def _solve_color_map(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Exact color mapping (same-size, consistent mapping)."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "color_map"

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
                        if c in color_map and color_map[c] != target:
                            consistent = False
                        else:
                            color_map[c] = target

        if not consistent or not color_map:
            return None, 0.0, "color_map"

        # Only use if at least one non-trivial mapping exists
        has_change = any(k != v for k, v in color_map.items())
        if not has_change:
            return None, 0.0, "color_map"

        result = test_input.copy()
        for c, target in color_map.items():
            result[test_input == c] = target

        mapped = sum(int(np.sum(test_input == c)) for c in color_map)
        return result, mapped / test_input.size, "color_map"

    def _solve_pattern_tile(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Pattern tile fill — tries all period sizes from OUTPUT."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            h, w = out.shape

            # Strategy 1: Find period from output
            for ph in range(1, min(h, 10) + 1):
                if h % ph != 0:
                    continue
                for pw in range(1, min(w, 10) + 1):
                    if w % pw != 0:
                        continue
                    tile = out[:ph, :pw]
                    tiled = np.tile(tile, (h // ph, w // pw))
                    if np.array_equal(tiled, out):
                        # Check input consistency
                        tiled_check = np.tile(tile, (h // ph, w // pw))
                        input_mask = inp != 0
                        if np.array_equal(inp[input_mask], tiled_check[input_mask]):
                            th, tw = test_input.shape
                            test_result = np.tile(
                                tile, (th // ph + 1, tw // pw + 1)
                            )[:th, :tw]
                            test_mask = test_input != 0
                            test_result[test_mask] = test_input[test_mask]
                            return test_result, 1.0, "pattern_tile"

            # Strategy 2: Seed from input non-zero region
            nonzero_mask = inp != 0
            if nonzero_mask.any():
                rows = np.any(nonzero_mask, axis=1)
                cols = np.any(nonzero_mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                seed = inp[rmin : rmax + 1, cmin : cmax + 1]
                sh, sw = seed.shape
                if sh > 0 and sw > 0:
                    result = np.zeros_like(out)
                    for r in range(h):
                        for c in range(w):
                            result[r, c] = seed[r % sh, c % sw]
                    if np.array_equal(result, out):
                        th, tw = test_input.shape
                        test_result = np.zeros_like(test_input)
                        for r in range(th):
                            for c in range(tw):
                                test_result[r, c] = seed[r % sh, c % sw]
                        return test_result, 1.0, "pattern_seed_tile"
                    # Fill only background
                    result_bg = inp.copy()
                    for r in range(h):
                        for c in range(w):
                            if inp[r, c] == 0:
                                result_bg[r, c] = seed[r % sh, c % sw]
                    if np.array_equal(result_bg, out):
                        test_result = test_input.copy()
                        for r in range(test_input.shape[0]):
                            for c in range(test_input.shape[1]):
                                if test_input[r, c] == 0:
                                    test_result[r, c] = seed[r % sh, c % sw]
                        return test_result, 1.0, "pattern_seed_fill"

            # Strategy 3: Diagonal wrap (circulant)
            for ph in range(2, min(h, 8) + 1):
                for pw in range(2, min(w, 8) + 1):
                    if h % ph != 0 or w % pw != 0:
                        continue
                    pattern = out[:ph, :pw].copy()
                    match = True
                    for r in range(h):
                        for c in range(w):
                            if out[r, c] != pattern[r % ph, (c + r) % pw]:
                                match = False
                                break
                        if not match:
                            break
                    if match:
                        input_ok = True
                        for r in range(h):
                            for c in range(w):
                                if inp[r, c] != 0:
                                    if inp[r, c] != pattern[r % ph, (c + r) % pw]:
                                        input_ok = False
                                        break
                            if not input_ok:
                                break
                        if input_ok:
                            th, tw = test_input.shape
                            test_result = np.zeros_like(test_input)
                            for r in range(th):
                                for c in range(tw):
                                    test_result[r, c] = pattern[r % ph, (c + r) % pw]
                            test_mask = test_input != 0
                            test_result[test_mask] = test_input[test_mask]
                            return test_result, 1.0, "pattern_diag_wrap"

            # Strategy 4: Anti-diagonal wrap (each row shifts right)
            for ph in range(2, min(h, 8) + 1):
                for pw in range(2, min(w, 8) + 1):
                    if h % ph != 0 or w % pw != 0:
                        continue
                    pattern = out[:ph, :pw].copy()
                    match = True
                    for r in range(h):
                        for c in range(w):
                            if out[r, c] != pattern[r % ph, (c - r) % pw]:
                                match = False
                                break
                        if not match:
                            break
                    if match:
                        input_ok = True
                        for r in range(h):
                            for c in range(w):
                                if inp[r, c] != 0:
                                    if inp[r, c] != pattern[r % ph, (c - r) % pw]:
                                        input_ok = False
                                        break
                            if not input_ok:
                                break
                        if input_ok:
                            th, tw = test_input.shape
                            test_result = np.zeros_like(test_input)
                            for r in range(th):
                                for c in range(tw):
                                    test_result[r, c] = pattern[r % ph, (c - r) % pw]
                            test_mask = test_input != 0
                            test_result[test_mask] = test_input[test_mask]
                            return test_result, 1.0, "pattern_anti_diag_wrap"

        return None, 0.0, "pattern_tile"

    def _solve_symmetry_complete(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Complete symmetric patterns (horizontal, vertical, point, block)."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            h, w = inp.shape

            # Horizontal symmetry
            result_h = inp.copy()
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == 0 and inp[r, w - 1 - c] != 0:
                        result_h[r, c] = inp[r, w - 1 - c]
            if np.array_equal(result_h, out):
                tr = test_input.copy()
                th, tw = test_input.shape
                for r in range(th):
                    for c in range(tw):
                        if test_input[r, c] == 0 and test_input[r, tw - 1 - c] != 0:
                            tr[r, c] = test_input[r, tw - 1 - c]
                return tr, 1.0, "symmetry_h"

            # Vertical symmetry
            result_v = inp.copy()
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == 0 and inp[h - 1 - r, c] != 0:
                        result_v[r, c] = inp[h - 1 - r, c]
            if np.array_equal(result_v, out):
                tr = test_input.copy()
                th, tw = test_input.shape
                for r in range(th):
                    for c in range(tw):
                        if test_input[r, c] == 0 and test_input[th - 1 - r, c] != 0:
                            tr[r, c] = test_input[th - 1 - r, c]
                return tr, 1.0, "symmetry_v"

            # Point symmetry (180 rotation)
            result_p = inp.copy()
            for r in range(h):
                for c in range(w):
                    if inp[r, c] == 0 and inp[h - 1 - r, w - 1 - c] != 0:
                        result_p[r, c] = inp[h - 1 - r, w - 1 - c]
            if np.array_equal(result_p, out):
                tr = test_input.copy()
                th, tw = test_input.shape
                for r in range(th):
                    for c in range(tw):
                        if test_input[r, c] == 0 and test_input[th - 1 - r, tw - 1 - c] != 0:
                            tr[r, c] = test_input[th - 1 - r, tw - 1 - c]
                return tr, 1.0, "symmetry_point"

            # Block tile symmetry: complete a repeating block
            nonzero = np.argwhere(inp != 0)
            if len(nonzero) > 0:
                rmin, cmin = nonzero.min(axis=0)
                rmax, cmax = nonzero.max(axis=0)
                bh = rmax - rmin + 1
                bw = cmax - cmin + 1
                if bh > 0 and bw > 0:
                    block = inp[rmin : rmax + 1, cmin : cmax + 1]
                    tiled = np.zeros_like(out)
                    for r in range(out.shape[0]):
                        for c in range(out.shape[1]):
                            tiled[r, c] = block[(r - rmin) % bh, (c - cmin) % bw]
                    if np.array_equal(tiled, out):
                        tr = np.zeros_like(test_input)
                        for r in range(test_input.shape[0]):
                            for c in range(test_input.shape[1]):
                                tr[r, c] = block[(r - rmin) % bh, (c - cmin) % bw]
                        return tr, 1.0, "symmetry_block"

        return None, 0.0, "symmetry"

    def _solve_marker_replace(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Replace marker pixels with nearest border color."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            in_colors = set(np.unique(inp).tolist())
            out_colors = set(np.unique(out).tolist())

            # Find marker color: disappears or changes to multiple colors
            for marker_color in in_colors:
                if marker_color == 0:
                    continue
                mask = inp == marker_color
                if not mask.any():
                    continue
                out_vals = set(out[mask].tolist())
                if len(out_vals) <= 1 and marker_color in out_colors:
                    continue  # Color didn't change

                # Detect borders
                borders: dict[str, int] = {}
                for name, arr in [("top", inp[0, :]), ("bottom", inp[-1, :]),
                                  ("left", inp[:, 0]), ("right", inp[:, -1])]:
                    nz = arr[arr != 0]
                    if len(nz) > 0 and len(set(nz.tolist())) == 1:
                        borders[name] = int(nz[0])

                if not borders:
                    continue

                # Apply
                result = inp.copy()
                h, w = inp.shape
                for r in range(h):
                    for c in range(w):
                        if inp[r, c] == marker_color:
                            min_dist = float("inf")
                            nearest = marker_color
                            for bname, bc in borders.items():
                                if bname == "top": dist = r
                                elif bname == "bottom": dist = h - 1 - r
                                elif bname == "left": dist = c
                                elif bname == "right": dist = w - 1 - c
                                else: continue
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest = bc
                            result[r, c] = nearest

                if np.array_equal(result, out):
                    # Apply to test
                    tr = test_input.copy()
                    th, tw = test_input.shape
                    tborders: dict[str, int] = {}
                    for name, arr in [("top", test_input[0, :]), ("bottom", test_input[-1, :]),
                                      ("left", test_input[:, 0]), ("right", test_input[:, -1])]:
                        nz = arr[arr != 0]
                        if len(nz) > 0 and len(set(nz.tolist())) == 1:
                            tborders[name] = int(nz[0])
                    for r in range(th):
                        for c in range(tw):
                            if test_input[r, c] == marker_color:
                                min_dist = float("inf")
                                nearest = marker_color
                                for bname, bc in tborders.items():
                                    if bname == "top": dist = r
                                    elif bname == "bottom": dist = th - 1 - r
                                    elif bname == "left": dist = c
                                    elif bname == "right": dist = tw - 1 - c
                                    if dist < min_dist:
                                        min_dist = dist
                                        nearest = bc
                                tr[r, c] = nearest
                    return tr, 1.0, "marker_replace"

        return None, 0.0, "marker_replace"

    def _solve_object_shift(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Shift/move objects by inferred offset."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            h, w = inp.shape

            # Find best shift offset
            best_shift = None
            best_match = 0
            for dr in range(-h + 1, h):
                for dc in range(-w + 1, w):
                    if dr == 0 and dc == 0:
                        continue
                    shifted = np.zeros_like(inp)
                    for r in range(h):
                        for c in range(w):
                            sr, sc = r - dr, c - dc
                            if 0 <= sr < h and 0 <= sc < w:
                                shifted[r, c] = inp[sr, sc]
                    match = int(np.sum(shifted == out))
                    if match > best_match:
                        best_match = match
                        best_shift = (dr, dc)

            if best_shift and best_match / out.size > 0.95:
                dr, dc = best_shift
                # Verify all pairs
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    shifted = np.zeros_like(inp2)
                    h2, w2 = inp2.shape
                    for r in range(h2):
                        for c in range(w2):
                            sr, sc = r - dr, c - dc
                            if 0 <= sr < h2 and 0 <= sc < w2:
                                shifted[r, c] = inp2[sr, sc]
                    if not np.array_equal(shifted, out2):
                        all_match = False
                        break
                if all_match:
                    tr = np.zeros_like(test_input)
                    th, tw = test_input.shape
                    for r in range(th):
                        for c in range(tw):
                            sr, sc = r - dr, c - dc
                            if 0 <= sr < th and 0 <= sc < tw:
                                tr[r, c] = test_input[sr, sc]
                    return tr, 1.0, "object_shift"

        return None, 0.0, "object_shift"

    def _solve_plus_marker(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Draw plus/cross patterns around marker pixels."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            new_pos = np.argwhere((diff != 0) & (inp == 0))
            if len(new_pos) == 0:
                continue

            # Find markers: non-zero pixels that stay unchanged
            markers = []
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if inp[r, c] != 0 and out[r, c] == inp[r, c]:
                        markers.append((r, c, int(inp[r, c])))

            # Map: marker_color -> plus_color
            mc_map: dict[int, int] = {}
            consistent = True
            for nr, nc in new_pos:
                plus_c = int(out[nr, nc])
                found = False
                for mr, mc, mcolor in markers:
                    if abs(nr - mr) + abs(nc - mc) == 1:
                        if mcolor not in mc_map:
                            mc_map[mcolor] = plus_c
                        elif mc_map[mcolor] != plus_c:
                            consistent = False
                        found = True
                        break
                if not found:
                    consistent = False

            if not consistent or not mc_map:
                continue

            # Verify all pairs
            all_match = True
            for inp2, out2 in demo_pairs:
                if inp2.shape != out2.shape:
                    all_match = False
                    break
                result = inp2.copy()
                for r in range(inp2.shape[0]):
                    for c in range(inp2.shape[1]):
                        pc = int(inp2[r, c])
                        if pc in mc_map:
                            plus_c = mc_map[pc]
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < inp2.shape[0] and 0 <= nc < inp2.shape[1]:
                                    if inp2[nr, nc] == 0:
                                        result[nr, nc] = plus_c
                if not np.array_equal(result, out2):
                    all_match = False
                    break

            if all_match:
                tr = test_input.copy()
                for r in range(test_input.shape[0]):
                    for c in range(test_input.shape[1]):
                        pc = int(test_input[r, c])
                        if pc in mc_map:
                            plus_c = mc_map[pc]
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
                                    if test_input[nr, nc] == 0:
                                        tr[nr, nc] = plus_c
                return tr, 1.0, "plus_marker"

        return None, 0.0, "plus_marker"

    def _solve_size_decrease(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Crop/extract region for size-decrease tasks."""
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] and inp.shape[1] <= out.shape[1]:
                continue

            # Strategy 1: Largest CC
            binary = (inp != 0).astype(int)
            labeled, num = ndimage.label(binary)
            if num > 0:
                sizes = ndimage.sum(binary, labeled, range(1, num + 1))
                largest = int(np.argmax(sizes)) + 1
                mask = labeled == largest
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                extracted = inp[rmin : rmax + 1, cmin : cmax + 1]
                if extracted.shape == out.shape and np.array_equal(extracted, out):
                    tb = (test_input != 0).astype(int)
                    tl, tn = ndimage.label(tb)
                    if tn > 0:
                        ts = ndimage.sum(tb, tl, range(1, tn + 1))
                        tla = int(np.argmax(ts)) + 1
                        tm = tl == tla
                        tr = np.any(tm, axis=1)
                        tc = np.any(tm, axis=0)
                        trmin, trmax = np.where(tr)[0][[0, -1]]
                        tcmin, tcmax = np.where(tc)[0][[0, -1]]
                        return test_input[trmin : trmax + 1, tcmin : tcmax + 1], 1.0, "crop_largest_cc"

            # Strategy 2: Crop to non-zero bbox
            nz = np.argwhere(inp != 0)
            if len(nz) > 0:
                rmin, cmin = nz.min(axis=0)
                rmax, cmax = nz.max(axis=0)
                cropped = inp[rmin : rmax + 1, cmin : cmax + 1]
                if cropped.shape == out.shape and np.array_equal(cropped, out):
                    tnz = np.argwhere(test_input != 0)
                    if len(tnz) > 0:
                        trmin, tcmin = tnz.min(axis=0)
                        trmax, tcmax = tnz.max(axis=0)
                        return test_input[trmin : trmax + 1, tcmin : tcmax + 1], 1.0, "crop_bbox"

            # Strategy 3: Crop by specific color
            for color in range(1, 10):
                mask = inp == color
                if mask.any():
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        extracted = inp[rmin : rmax + 1, cmin : cmax + 1]
                        if extracted.shape == out.shape and np.array_equal(extracted, out):
                            tm = test_input == color
                            if tm.any():
                                tr = np.any(tm, axis=1)
                                tcc = np.any(tm, axis=0)
                                trmin, trmax = np.where(tr)[0][[0, -1]]
                                tcmin, tcmax = np.where(tcc)[0][[0, -1]]
                                return test_input[trmin : trmax + 1, tcmin : tcmax + 1], 1.0, f"crop_color_{color}"

        return None, 0.0, "size_decrease"

    def _solve_size_increase(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Tile/repeat for size-increase tasks."""
        for inp, out in demo_pairs:
            if inp.shape[0] >= out.shape[0] and inp.shape[1] >= out.shape[1]:
                continue
            rh = out.shape[0] / inp.shape[0]
            rw = out.shape[1] / inp.shape[1]
            if rh == rw and rh == int(rh) and rh > 1:
                factor = int(rh)
                scaled = np.repeat(np.repeat(inp, factor, axis=0), factor, axis=1)
                if scaled.shape == out.shape and np.array_equal(scaled, out):
                    return np.repeat(np.repeat(test_input, factor, axis=0), factor, axis=1), 1.0, "scale_repeat"
            if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                th = out.shape[0] // inp.shape[0]
                tw = out.shape[1] // inp.shape[1]
                tiled = np.tile(inp, (th, tw))
                if tiled.shape == out.shape and np.array_equal(tiled, out):
                    return np.tile(test_input, (th, tw)), 1.0, "tile_repeat"
        return None, 0.0, "size_increase"

    def _solve_grid_diff(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Apply consistent pixel-level diff."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "grid_diff"

        # Check if diff is identical across all pairs
        diffs = [out.astype(int) - inp.astype(int) for inp, out in demo_pairs]
        consistent = all(np.array_equal(diffs[0], d) for d in diffs[1:])
        if consistent:
            result = np.clip(test_input.astype(int) + diffs[0], 0, 9).astype(np.int8)
            return result, 1.0, "grid_diff"

        # Fill all 0 pixels with a single color
        for inp, out in demo_pairs:
            from_zero = np.argwhere((inp == 0) & (out != 0))
            if len(from_zero) > 0:
                fill_colors = set(out[inp == 0].tolist()) - {0}
                if len(fill_colors) == 1:
                    fc = fill_colors.pop()
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        r = inp2.copy()
                        r[inp2 == 0] = fc
                        if not np.array_equal(r, out2):
                            all_match = False
                            break
                    if all_match:
                        tr = test_input.copy()
                        tr[test_input == 0] = fc
                        return tr, 1.0, "fill_bg"
        return None, 0.0, "grid_diff"

    def _solve_fill_enclosed(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Fill enclosed background regions."""
        from scipy.ndimage import binary_fill_holes

        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue
            from_zero = np.argwhere((inp == 0) & (out != 0))
            if len(from_zero) != len(changed):
                continue
            fill_colors = set(out[from_zero[:, 0], from_zero[:, 1]].tolist())
            if len(fill_colors) != 1:
                continue
            fc = fill_colors.pop()
            binary = (inp != 0).astype(int)
            filled = binary_fill_holes(binary)
            enclosed = filled & ~binary.astype(bool)

            all_match = True
            for inp2, out2 in demo_pairs:
                if inp2.shape != out2.shape:
                    all_match = False
                    break
                b2 = (inp2 != 0).astype(int)
                f2 = binary_fill_holes(b2)
                e2 = f2 & ~b2.astype(bool)
                r = inp2.copy()
                r[e2] = fc
                if not np.array_equal(r, out2):
                    all_match = False
                    break
            if all_match:
                tb = (test_input != 0).astype(int)
                tf = binary_fill_holes(tb)
                te = tf & ~tb.astype(bool)
                tr = test_input.copy()
                tr[te] = fc
                return tr, 1.0, "fill_enclosed"
        return None, 0.0, "fill_enclosed"

    def _solve_shape_dir(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Complete shape by copying from a direction."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
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
                        tr = test_input.copy()
                        for r in range(test_input.shape[0]):
                            for c in range(test_input.shape[1]):
                                if test_input[r, c] == 0:
                                    sr, sc = r + dr, c + dc
                                    if 0 <= sr < test_input.shape[0] and 0 <= sc < test_input.shape[1]:
                                        if test_input[sr, sc] != 0:
                                            tr[r, c] = test_input[sr, sc]
                        return tr, 1.0, f"shape_dir_{dr}_{dc}"
        return None, 0.0, "shape_dir"

    def _solve_color_swap(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Swap two or more colors."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            # Build color map
            cm: dict[int, int] = {}
            for c in range(10):
                mask = inp == c
                if mask.any():
                    unique = np.unique(out[mask])
                    if len(unique) == 1:
                        cm[c] = int(unique[0])
            # Check for swaps
            swaps = {c: t for c, t in cm.items() if c != t and t in cm and cm[t] == c}
            if len(swaps) >= 2:
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    r = inp2.copy()
                    for c, t in swaps.items():
                        r[inp2 == c] = t
                    if not np.array_equal(r, out2):
                        all_match = False
                        break
                if all_match:
                    tr = test_input.copy()
                    for c, t in swaps.items():
                        tr[test_input == c] = t
                    return tr, 1.0, "color_swap"
        return None, 0.0, "color_swap"

    def _solve_gravity(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Gravity: drop non-zero pixels to bottom of each column."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            # Try gravity down
            result = np.zeros_like(inp)
            for c in range(inp.shape[1]):
                col = inp[:, c]
                nz = col[col != 0]
                result[-len(nz):, c] = nz
            if np.array_equal(result, out):
                tr = np.zeros_like(test_input)
                for c in range(test_input.shape[1]):
                    col = test_input[:, c]
                    nz = col[col != 0]
                    tr[-len(nz):, c] = nz
                return tr, 1.0, "gravity_down"

            # Try gravity up
            result = np.zeros_like(inp)
            for c in range(inp.shape[1]):
                col = inp[:, c]
                nz = col[col != 0]
                result[:len(nz), c] = nz
            if np.array_equal(result, out):
                tr = np.zeros_like(test_input)
                for c in range(test_input.shape[1]):
                    col = test_input[:, c]
                    nz = col[col != 0]
                    tr[:len(nz), c] = nz
                return tr, 1.0, "gravity_up"

            # Try gravity left
            result = np.zeros_like(inp)
            for r in range(inp.shape[0]):
                row = inp[r, :]
                nz = row[row != 0]
                result[r, :len(nz)] = nz
            if np.array_equal(result, out):
                tr = np.zeros_like(test_input)
                for r in range(test_input.shape[0]):
                    row = test_input[r, :]
                    nz = row[row != 0]
                    tr[r, :len(nz)] = nz
                return tr, 1.0, "gravity_left"

            # Try gravity right
            result = np.zeros_like(inp)
            for r in range(inp.shape[0]):
                row = inp[r, :]
                nz = row[row != 0]
                result[r, -len(nz):] = nz
            if np.array_equal(result, out):
                tr = np.zeros_like(test_input)
                for r in range(test_input.shape[0]):
                    row = test_input[r, :]
                    nz = row[row != 0]
                    tr[r, -len(nz):] = nz
                return tr, 1.0, "gravity_right"
        return None, 0.0, "gravity"

    def _solve_scale_pattern(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Scale pattern: each pixel becomes a block of size factor x factor."""
        for inp, out in demo_pairs:
            if inp.shape[0] == 0:
                continue
            rh = out.shape[0] / inp.shape[0]
            rw = out.shape[1] / inp.shape[1]
            if rh == rw and rh == int(rh) and rh > 1:
                factor = int(rh)
                scaled = np.repeat(np.repeat(inp, factor, axis=0), factor, axis=1)
                if np.array_equal(scaled, out):
                    return np.repeat(np.repeat(test_input, factor, axis=0), factor, axis=1), 1.0, "scale_pattern"
            # Also try: each non-zero pixel becomes a pattern
            if rh == int(rh) and rw == int(rw):
                fhr = int(rh)
                fwr = int(rw)
                # Check if output is input scaled with pattern fill
                # E.g., each pixel -> block of same color
                scaled = np.kron(inp, np.ones((fhr, fwr), dtype=np.int8))
                if np.array_equal(scaled, out):
                    return np.kron(test_input, np.ones((fhr, fwr), dtype=np.int8)), 1.0, "scale_kron"
        return None, 0.0, "scale_pattern"

    def _solve_recolor_cc(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Recolor connected components."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            # Label connected components in input
            binary = (inp != 0).astype(int)
            labeled, num = ndimage.label(binary)
            if num == 0:
                continue

            # Check: each CC gets a single color in output
            cc_colors: dict[int, int] = {}
            consistent = True
            for label in range(1, num + 1):
                mask = labeled == label
                out_vals = set(out[mask].tolist())
                if len(out_vals) == 1:
                    cc_colors[label] = int(out_vals.pop())
                else:
                    consistent = False

            if not consistent:
                continue

            # Check: CC color depends on CC size or position
            # Try: color by size rank
            sizes = {l: int(np.sum(labeled == l)) for l in range(1, num + 1)}
            sorted_labels = sorted(sizes, key=lambda x: sizes[x])
            color_by_rank = {}
            for rank, label in enumerate(sorted_labels):
                color_by_rank[rank] = cc_colors.get(label, 0)

            # Verify: does color_by_rank work for all pairs?
            all_match = True
            for inp2, out2 in demo_pairs:
                if inp2.shape != out2.shape:
                    all_match = False
                    break
                b2 = (inp2 != 0).astype(int)
                l2, n2 = ndimage.label(b2)
                if n2 == 0:
                    continue
                sizes2 = {l: int(np.sum(l2 == l)) for l in range(1, n2 + 1)}
                sorted2 = sorted(sizes2, key=lambda x: sizes2[x])
                r2 = inp2.copy()
                for rank, label in enumerate(sorted2):
                    if rank in color_by_rank:
                        r2[l2 == label] = color_by_rank[rank]
                if not np.array_equal(r2, out2):
                    all_match = False
                    break

            if all_match and color_by_rank:
                # Apply to test
                tb = (test_input != 0).astype(int)
                tl, tn = ndimage.label(tb)
                if tn > 0:
                    tsizes = {l: int(np.sum(tl == l)) for l in range(1, tn + 1)}
                    tsorted = sorted(tsizes, key=lambda x: tsizes[x])
                    tr = test_input.copy()
                    for rank, label in enumerate(tsorted):
                        if rank in color_by_rank:
                            tr[tl == label] = color_by_rank[rank]
                    return tr, 1.0, "recolor_cc_by_size"

        return None, 0.0, "recolor_cc"

    def _solve_flood_fill_smart(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Smart flood fill: detect fill color and region from demo pairs."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue

            # Check: all changed pixels are 0 -> X
            from_zero = np.argwhere((inp == 0) & (out != 0))
            if len(from_zero) == 0:
                continue
            fill_colors = set(out[from_zero[:, 0], from_zero[:, 1]].tolist())
            if len(fill_colors) != 1:
                continue
            fc = fill_colors.pop()

            # Check: are the filled regions enclosed?
            from scipy.ndimage import binary_fill_holes
            binary = (inp != 0).astype(int)
            filled = binary_fill_holes(binary)
            enclosed = filled & ~binary.astype(bool)
            enclosed_match = np.array_equal(enclosed & (inp == 0), (inp == 0) & (out == fc))

            if enclosed_match:
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    b2 = (inp2 != 0).astype(int)
                    f2 = binary_fill_holes(b2)
                    e2 = f2 & ~b2.astype(bool)
                    r = inp2.copy()
                    r[e2] = fc
                    if not np.array_equal(r, out2):
                        all_match = False
                        break
                if all_match:
                    tb = (test_input != 0).astype(int)
                    tf = binary_fill_holes(tb)
                    te = tf & ~tb.astype(bool)
                    tr = test_input.copy()
                    tr[te] = fc
                    return tr, 1.0, "flood_enclosed"

            # Check: are the filled regions connected to border?
            # (flood fill from border 0s with fill color)
            # This is the opposite: fill non-enclosed 0s
            border_filled = ~enclosed & (inp == 0)
            border_match = np.array_equal(border_filled, (inp == 0) & (out == fc))
            if border_match:
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    b2 = (inp2 != 0).astype(int)
                    f2 = binary_fill_holes(b2)
                    e2 = f2 & ~b2.astype(bool)
                    bf2 = ~e2 & (inp2 == 0)
                    r = inp2.copy()
                    r[bf2] = fc
                    if not np.array_equal(r, out2):
                        all_match = False
                        break
                if all_match:
                    tb = (test_input != 0).astype(int)
                    tf = binary_fill_holes(tb)
                    te = tf & ~tb.astype(bool)
                    tbf = ~te & (test_input == 0)
                    tr = test_input.copy()
                    tr[tbf] = fc
                    return tr, 1.0, "flood_border"

        return None, 0.0, "flood_fill"

    def _solve_nearest_neighbor(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Apply transformation from most similar training pair."""
        if not demo_pairs:
            return None, 0.0, "nearest_neighbor"
        best_sim = 0.0
        best_pair = None
        for inp, out in demo_pairs:
            if inp.shape == test_input.shape:
                sim = float(np.sum(inp == test_input) / test_input.size)
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (inp, out)
        if best_pair and best_sim > 0.5:
            inp, out = best_pair
            cm: dict[int, int] = {}
            for c in range(10):
                mask = inp == c
                if mask.any():
                    unique = np.unique(out[mask])
                    if len(unique) == 1:
                        cm[c] = int(unique[0])
            result = test_input.copy()
            for c, t in cm.items():
                result[test_input == c] = t
            return result, best_sim * 0.7, "nearest_neighbor"
        return None, 0.0, "nearest_neighbor"

    def _solve_draw_frame(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Draw a 1-pixel frame around objects."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            new_pixels = np.argwhere((diff != 0) & (inp == 0))
            if len(new_pixels) == 0:
                continue
            binary = (inp != 0).astype(int)
            dilated = ndimage.binary_dilation(binary, iterations=1).astype(int)
            border = dilated - binary
            if border.sum() == len(new_pixels):
                border_colors = set(out[border.astype(bool)].tolist())
                if len(border_colors) == 1:
                    bc = border_colors.pop()
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        b2 = (inp2 != 0).astype(int)
                        d2 = ndimage.binary_dilation(b2, iterations=1).astype(int)
                        bd2 = d2 - b2
                        r = inp2.copy()
                        r[bd2.astype(bool)] = bc
                        if not np.array_equal(r, out2):
                            all_match = False
                            break
                    if all_match:
                        tb = (test_input != 0).astype(int)
                        td = ndimage.binary_dilation(tb, iterations=1).astype(int)
                        tbd = td - tb
                        tr = test_input.copy()
                        tr[tbd.astype(bool)] = bc
                        return tr, 1.0, "draw_frame"
        return None, 0.0, "draw_frame"

    def _solve_count_and_mark(
        self, demo_pairs, test_input
    ) -> tuple[np.ndarray | None, float, str]:
        """Count objects and mark count in output."""
        for inp, out in demo_pairs:
            if inp.shape == out.shape:
                continue  # Size must change
            # Output is always 1xN or Nx1 (count array)
            if out.shape[0] == 1 or out.shape[1] == 1:
                binary = (inp != 0).astype(int)
                _, num = ndimage.label(binary)
                flat_out = out.flatten()
                # Check if output represents count
                if len(flat_out) == num:
                    # Each position in output represents one object
                    # Color might indicate object size or position
                    pass
        return None, 0.0, "count_mark"
