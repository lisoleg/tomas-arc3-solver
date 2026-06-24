"""Direct Solver v3 — comprehensive pattern coverage with Bayesian fusion.

Adds 15+ new sub-solvers based on deep analysis of 43 failed tasks:
- Diagonal cyclic pattern fill
- Voronoi nearest-color fill
- Marker-to-nearest-color replacement
- Line drawing between marker pairs
- X-marker (diagonal cross)
- Object expand/contract
- Row/column insertion
- Smart crop variants
- Color-count based output
- Partial tile completion
- Pair symmetry completion
- Connected component shape matching
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy import ndimage


class DirectSolverV3:
    """Comprehensive direct solver with 35+ sub-solvers."""

    def __init__(self) -> None:
        pass

    def solve(self, task: dict[str, Any]) -> tuple[np.ndarray, float, str]:
        """Solve an ARC task. Returns (prediction, confidence, solver_name)."""
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

        solvers = [
            # V2 solvers
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
            # V3 new solvers
            self._solve_diag_cyclic_fill,
            self._solve_voronoi_fill,
            self._solve_marker_to_nearest,
            self._solve_line_between_markers,
            self._solve_x_marker,
            self._solve_object_expand,
            self._solve_object_contract,
            self._solve_row_insert,
            self._solve_col_insert,
            self._solve_smart_crop_color,
            self._solve_color_count_output,
            self._solve_partial_tile,
            self._solve_pair_symmetry,
            self._solve_cc_shape_match,
            self._solve_bg_fill_by_proximity,
            self._solve_color_replace_by_context,
            self._solve_extract_pattern_cell,
            self._solve_majority_color_crop,
            self._solve_flip_and_merge,
            self._solve_remove_color_rows,
            self._solve_grid_line_fill,
            self._solve_cross_marker_combined,
            self._solve_bg_fill_adjacent,
            self._solve_remove_border,
            self._solve_extract_inner,
            # V3.2 new high-impact solvers
            self._solve_repeat_extend,
            self._solve_extract_grid_content,
            self._solve_extract_largest_obj,
            self._solve_extract_separator_col,
            self._solve_extract_row_pattern,
            self._solve_count_output,
            self._solve_position_color_map,
            self._solve_draw_outline,
            self._solve_color_propagate,
            self._solve_crop_to_content,
            self._solve_delta_color,
            self._solve_sort_objects,
            self._solve_fill_between_seps,
            self._solve_mirror_extend,
            self._solve_combine_objects_v2,
            self._solve_replace_with_shape,
            self._solve_color_border_fill,
            self._solve_max_filter,
            self._solve_min_filter,
            self._solve_tile_repeat,
            self._solve_stretch_pattern,
            # V3.3 targeted solvers
            self._solve_grid_cell_count,
            self._solve_recolor_and_fill,
            self._solve_fill_enclosed_nearest,
            self._solve_shift_objects_v2,
            self._solve_extract_smallest_obj,
            self._solve_tile_with_checker,
            self._solve_recolor_inplace,
            self._solve_copy_half,
            self._solve_move_to_edge,
            # V3.4 high-impact solvers
            self._solve_center_pixel,
            self._solve_split_combine,
            self._solve_repeat_recolor,
            self._solve_diagonal_connect,
            self._solve_triangle_fill,
            self._solve_left_boundary_fill,
            self._solve_marker_row_copy,
            self._solve_recolor_fill_enclosed,
            self._solve_compact_grid,
            self._solve_symmetric_complete,
            self._solve_recolor_border_nearest,
            self._solve_scale_2x_pattern,
            self._solve_fill_zeros_rule,
            self._solve_shift_objects_v3,
            self._solve_fill_enclosed_v3,
            self._solve_copy_pattern_down,
            # V3.5 high-impact solvers — targeting specific failing patterns
            self._solve_voronoi_fill,
            self._solve_draw_lines_markers,
            self._solve_fill_row_between_markers,
            self._solve_object_shift,
            self._solve_recolor_boundary_interior,
            self._solve_diagonal_extend,
            self._solve_connect_grid_markers,
            self._solve_compact_colored_pixels,
            # V3.6 solvers — targeting remaining failing patterns
            self._solve_align_objects_row,
            self._solve_complete_tile_pattern,
            self._solve_knight_move_diagonal,
            self._solve_per_color_shift,
            self._solve_fill_between_objects,
            self._solve_recolor_by_boundary,
            # V3.7 solvers — batch targeting remaining failed patterns
            self._solve_replicate_col_pattern,
            self._solve_extend_to_lines,
            self._solve_grid_row_replicate,
            self._solve_expand_plus_diamond,
            self._solve_draw_diagonal_line,
            self._solve_draw_rect_borders,
            self._solve_mirror_grid_fill,
            self._solve_traffic_jam_shift,
            self._solve_fill_enclosed_by_marker,
            self._solve_replicate_adjacent_pattern,
            self._solve_zone_dominant_fill,
        ]

        # Two-tier verification: perfect (verify=1.0) gets priority,
        # imperfect (verify>0.5) gets heavily discounted
        perfect_results: list[tuple[np.ndarray, float, str]] = []
        imperfect_results: list[tuple[np.ndarray, float, str]] = []

        for solver_fn in solvers:
            try:
                pred, conf, name = solver_fn(demo_pairs, test_input)
                if pred is not None and conf > 0:
                    verify = self._verify(demo_pairs, solver_fn)
                    if verify >= 0.99:
                        # Perfect verification — high confidence
                        perfect_results.append((pred, verify * conf, name))
                    elif verify > 0.5:
                        # Imperfect — heavily discounted (0.3x penalty)
                        imperfect_results.append((pred, verify * conf * 0.3, name))
            except Exception:
                pass

        # Prefer perfect verification results
        if perfect_results:
            perfect_results.sort(key=lambda x: -x[1])
            # Tie-breaking: when multiple solvers have same confidence,
            # prefer the LAST one (more specific V3 solvers run later)
            max_conf = perfect_results[0][1]
            for i in range(len(perfect_results) - 1, -1, -1):
                if perfect_results[i][1] == max_conf:
                    return perfect_results[i]
            return perfect_results[0]

        # Fall back to best imperfect result
        if imperfect_results:
            imperfect_results.sort(key=lambda x: -x[1])
            return imperfect_results[0]

        return test_input.copy(), 0.0, "fallback"

    def _verify(self, demo_pairs, solver_fn) -> float:
        """Verify a solver on all demo pairs."""
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

    # ==================== V2 Solvers (copied) ====================

    def _solve_color_map(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "color_map"
        color_map = {}
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
        has_change = any(k != v for k, v in color_map.items())
        if not has_change:
            return None, 0.0, "color_map"
        result = test_input.copy()
        for c, target in color_map.items():
            result[test_input == c] = target
        mapped = sum(int(np.sum(test_input == c)) for c in color_map)
        return result, mapped / test_input.size, "color_map"

    def _solve_pattern_tile(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            h, w = out.shape
            for ph in range(1, min(h, 10) + 1):
                if h % ph != 0:
                    continue
                for pw in range(1, min(w, 10) + 1):
                    if w % pw != 0:
                        continue
                    tile = out[:ph, :pw]
                    tiled = np.tile(tile, (h // ph, w // pw))
                    if np.array_equal(tiled, out):
                        tiled_check = np.tile(tile, (h // ph, w // pw))
                        input_mask = inp != 0
                        if np.array_equal(inp[input_mask], tiled_check[input_mask]):
                            th, tw = test_input.shape
                            test_result = np.tile(tile, (th // ph + 1, tw // pw + 1))[:th, :tw]
                            test_mask = test_input != 0
                            test_result[test_mask] = test_input[test_mask]
                            return test_result, 1.0, "pattern_tile"
            nonzero_mask = inp != 0
            if nonzero_mask.any():
                rows = np.any(nonzero_mask, axis=1)
                cols = np.any(nonzero_mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                seed = inp[rmin:rmax + 1, cmin:cmax + 1]
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
        return None, 0.0, "pattern_tile"

    def _solve_symmetry_complete(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            h, w = inp.shape
            for flip_fn, name in [(lambda x: np.fliplr(x), "sym_h"),
                                   (lambda x: np.flipud(x), "sym_v"),
                                   (lambda x: np.rot90(x, 2), "sym_p")]:
                result = inp.copy()
                flipped = flip_fn(inp)
                mask = (inp == 0) & (flipped != 0)
                result[mask] = flipped[mask]
                if np.array_equal(result, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        r2 = inp2.copy()
                        f2 = flip_fn(inp2)
                        m2 = (inp2 == 0) & (f2 != 0)
                        r2[m2] = f2[m2]
                        if not np.array_equal(r2, out2):
                            all_match = False
                            break
                    if all_match:
                        tr = test_input.copy()
                        tf = flip_fn(test_input)
                        tm = (test_input == 0) & (tf != 0)
                        tr[tm] = tf[tm]
                        return tr, 1.0, name
        return None, 0.0, "symmetry"

    def _solve_marker_replace(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            in_colors = set(np.unique(inp).tolist())
            for marker_color in in_colors:
                if marker_color == 0:
                    continue
                mask = inp == marker_color
                if not mask.any():
                    continue
                out_vals = set(out[mask].tolist())
                if len(out_vals) <= 1 and marker_color in set(out.tolist()):
                    continue
                borders = {}
                for name, arr in [("top", inp[0, :]), ("bottom", inp[-1, :]),
                                  ("left", inp[:, 0]), ("right", inp[:, -1])]:
                    nz = arr[arr != 0]
                    if len(nz) > 0 and len(set(nz.tolist())) == 1:
                        borders[name] = int(nz[0])
                if not borders:
                    continue
                result = inp.copy()
                h, w = inp.shape
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
                    tr = test_input.copy()
                    th, tw = test_input.shape
                    tborders = {}
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
                                tr[r, c] = nearest
                    return tr, 1.0, "marker_replace"
        return None, 0.0, "marker_replace"

    def _solve_object_shift(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            h, w = inp.shape
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

    def _solve_plus_marker(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            new_pos = np.argwhere((diff != 0) & (inp == 0))
            if len(new_pos) == 0:
                continue
            markers = []
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if inp[r, c] != 0 and out[r, c] == inp[r, c]:
                        markers.append((r, c, int(inp[r, c])))
            mc_map = {}
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

    def _solve_size_decrease(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] and inp.shape[1] <= out.shape[1]:
                continue
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
                extracted = inp[rmin:rmax + 1, cmin:cmax + 1]
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
                        return test_input[trmin:trmax + 1, tcmin:tcmax + 1], 1.0, "crop_largest_cc"
            nz = np.argwhere(inp != 0)
            if len(nz) > 0:
                rmin, cmin = nz.min(axis=0)
                rmax, cmax = nz.max(axis=0)
                cropped = inp[rmin:rmax + 1, cmin:cmax + 1]
                if cropped.shape == out.shape and np.array_equal(cropped, out):
                    tnz = np.argwhere(test_input != 0)
                    if len(tnz) > 0:
                        trmin, tcmin = tnz.min(axis=0)
                        trmax, tcmax = tnz.max(axis=0)
                        return test_input[trmin:trmax + 1, tcmin:tcmax + 1], 1.0, "crop_bbox"
            for color in range(1, 10):
                mask = inp == color
                if mask.any():
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if rows.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        extracted = inp[rmin:rmax + 1, cmin:cmax + 1]
                        if extracted.shape == out.shape and np.array_equal(extracted, out):
                            tm = test_input == color
                            if tm.any():
                                tr = np.any(tm, axis=1)
                                tcc = np.any(tm, axis=0)
                                trmin, trmax = np.where(tr)[0][[0, -1]]
                                tcmin, tcmax = np.where(tcc)[0][[0, -1]]
                                return test_input[trmin:trmax + 1, tcmin:tcmax + 1], 1.0, f"crop_color_{color}"
        return None, 0.0, "size_decrease"

    def _solve_size_increase(self, demo_pairs, test_input):
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

    def _solve_grid_diff(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "grid_diff"
        diffs = [out.astype(int) - inp.astype(int) for inp, out in demo_pairs]
        consistent = all(np.array_equal(diffs[0], d) for d in diffs[1:])
        if consistent:
            result = np.clip(test_input.astype(int) + diffs[0], 0, 9).astype(np.int8)
            return result, 1.0, "grid_diff"
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

    def _solve_fill_enclosed(self, demo_pairs, test_input):
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

    def _solve_shape_dir(self, demo_pairs, test_input):
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

    def _solve_color_swap(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            cm = {}
            for c in range(10):
                mask = inp == c
                if mask.any():
                    unique = np.unique(out[mask])
                    if len(unique) == 1:
                        cm[c] = int(unique[0])
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

    def _solve_gravity(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            for direction, name in [("down", "gravity_down"), ("up", "gravity_up"),
                                     ("left", "gravity_left"), ("right", "gravity_right")]:
                result = np.zeros_like(inp)
                if direction == "down":
                    for c in range(inp.shape[1]):
                        col = inp[:, c]
                        nz = col[col != 0]
                        result[-len(nz):, c] = nz
                elif direction == "up":
                    for c in range(inp.shape[1]):
                        col = inp[:, c]
                        nz = col[col != 0]
                        result[:len(nz), c] = nz
                elif direction == "left":
                    for r in range(inp.shape[0]):
                        row = inp[r, :]
                        nz = row[row != 0]
                        result[r, :len(nz)] = nz
                elif direction == "right":
                    for r in range(inp.shape[0]):
                        row = inp[r, :]
                        nz = row[row != 0]
                        result[r, -len(nz):] = nz
                if np.array_equal(result, out):
                    tr = np.zeros_like(test_input)
                    if direction == "down":
                        for c in range(test_input.shape[1]):
                            col = test_input[:, c]
                            nz = col[col != 0]
                            tr[-len(nz):, c] = nz
                    elif direction == "up":
                        for c in range(test_input.shape[1]):
                            col = test_input[:, c]
                            nz = col[col != 0]
                            tr[:len(nz), c] = nz
                    elif direction == "left":
                        for r in range(test_input.shape[0]):
                            row = test_input[r, :]
                            nz = row[row != 0]
                            tr[r, :len(nz)] = nz
                    elif direction == "right":
                        for r in range(test_input.shape[0]):
                            row = test_input[r, :]
                            nz = row[row != 0]
                            tr[r, -len(nz):] = nz
                    return tr, 1.0, name
        return None, 0.0, "gravity"

    def _solve_scale_pattern(self, demo_pairs, test_input):
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
            if rh == int(rh) and rw == int(rw):
                fhr = int(rh)
                fwr = int(rw)
                scaled = np.kron(inp, np.ones((fhr, fwr), dtype=np.int8))
                if np.array_equal(scaled, out):
                    return np.kron(test_input, np.ones((fhr, fwr), dtype=np.int8)), 1.0, "scale_kron"
        return None, 0.0, "scale_pattern"

    def _solve_recolor_cc(self, demo_pairs, test_input):
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            binary = (inp != 0).astype(int)
            labeled, num = ndimage.label(binary)
            if num == 0:
                continue
            cc_colors = {}
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
            sizes = {l: int(np.sum(labeled == l)) for l in range(1, num + 1)}
            sorted_labels = sorted(sizes, key=lambda x: sizes[x])
            color_by_rank = {}
            for rank, label in enumerate(sorted_labels):
                color_by_rank[rank] = cc_colors.get(label, 0)
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

    def _solve_flood_fill_smart(self, demo_pairs, test_input):
        from scipy.ndimage import binary_fill_holes
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue
            from_zero = np.argwhere((inp == 0) & (out != 0))
            if len(from_zero) == 0:
                continue
            fill_colors = set(out[from_zero[:, 0], from_zero[:, 1]].tolist())
            if len(fill_colors) != 1:
                continue
            fc = fill_colors.pop()
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
        return None, 0.0, "flood_fill"

    def _solve_nearest_neighbor(self, demo_pairs, test_input):
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
            cm = {}
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

    def _solve_draw_frame(self, demo_pairs, test_input):
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

    def _solve_count_and_mark(self, demo_pairs, test_input):
        return None, 0.0, "count_mark"

    # ==================== V3 NEW SOLVERS ====================

    def _solve_diag_cyclic_fill(self, demo_pairs, test_input):
        """Fill background (0) with a diagonal cyclic pattern.

        Color-agnostic: extracts the pattern from each pair's non-zero pixels
        independently, then applies the same structural pattern to the test input.

        Pattern: out[r,c] = color_at_position[(r+c) % cycle_len]
        The cycle is extracted from the input's non-zero pixels.
        """
        # For each pair, extract the diagonal cycle from the output
        cycles = []
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "diag_cyclic_fill"

            # Try cycle lengths 2-5
            best_cycle = None
            best_clen = 0
            for clen in range(2, 6):
                # Build cycle from output: for each (r+c)%clen, find the color
                cycle = {}
                consistent = True
                for r in range(out.shape[0]):
                    for c in range(out.shape[1]):
                        idx = (r + c) % clen
                        expected = out[r, c]
                        if idx not in cycle:
                            cycle[idx] = expected
                        elif cycle[idx] != expected:
                            consistent = False
                            break
                    if not consistent:
                        break
                if consistent and len(cycle) == clen:
                    # Verify: input non-zero pixels match output
                    input_ok = True
                    for r in range(inp.shape[0]):
                        for c in range(inp.shape[1]):
                            if inp[r, c] != 0 and out[r, c] != inp[r, c]:
                                input_ok = False
                                break
                        if not input_ok:
                            break
                    if input_ok:
                        best_cycle = cycle
                        best_clen = clen
                        break

            if best_cycle is None:
                return None, 0.0, "diag_cyclic_fill"
            cycles.append((best_cycle, best_clen))

        # Check all pairs have the same cycle length
        clen = cycles[0][1]
        if not all(c[1] == clen for c in cycles):
            return None, 0.0, "diag_cyclic_fill"

        # Extract cycle from test input's non-zero pixels
        test_cycle = {}
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                if test_input[r, c] != 0:
                    idx = (r + c) % clen
                    if idx not in test_cycle:
                        test_cycle[idx] = int(test_input[r, c])
                    elif test_cycle[idx] != int(test_input[r, c]):
                        return None, 0.0, "diag_cyclic_fill"

        # If we don't have all cycle positions from test input, try to infer
        # from the pattern: use the first pair's cycle structure
        if len(test_cycle) < clen:
            # Try to infer missing positions from demo pairs
            # The cycle positions should follow a consistent ordering
            for pair_idx, (cycle, _) in enumerate(cycles):
                for idx in range(clen):
                    if idx not in test_cycle:
                        # Find what color this position maps to in this pair
                        # and check if the test input has this color at any position
                        pair_color = cycle[idx]
                        # Check if test input has this color
                        if (test_input == pair_color).any():
                            # Check if this color appears at a position with this idx
                            for r in range(test_input.shape[0]):
                                for c in range(test_input.shape[1]):
                                    if test_input[r, c] == pair_color and (r + c) % clen == idx:
                                        test_cycle[idx] = int(test_input[r, c])
                                        break
                                if idx in test_cycle:
                                    break

        if len(test_cycle) < clen:
            return None, 0.0, "diag_cyclic_fill"

        # Fill test input
        tr = test_input.copy()
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                if test_input[r, c] == 0:
                    idx = (r + c) % clen
                    tr[r, c] = test_cycle.get(idx, 0)

        return tr, 1.0, "diag_cyclic_fill"

    def _solve_voronoi_fill(self, demo_pairs, test_input):
        """Fill background (0) with nearest non-zero color (Voronoi-like).

        Uses BFS distance transform to assign each 0 pixel to its nearest
        non-zero neighbor's color.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue
            # Check: only 0 pixels changed
            from_zero = np.argwhere((inp == 0) & (out != 0))
            to_zero = np.argwhere((inp != 0) & (out == 0))
            if len(to_zero) > 0:
                continue
            if len(from_zero) != len(changed):
                continue

            # Compute Voronoi fill
            filled = self._voronoi_fill(inp)
            if filled is not None and np.array_equal(filled, out):
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    f2 = self._voronoi_fill(inp2)
                    if f2 is None or not np.array_equal(f2, out2):
                        all_match = False
                        break
                if all_match:
                    tr = self._voronoi_fill(test_input)
                    if tr is not None:
                        return tr, 1.0, "voronoi_fill"
        return None, 0.0, "voronoi_fill"

    def _voronoi_fill(self, grid):
        """Fill 0 pixels with nearest non-zero color using distance transform."""
        nonzero = grid != 0
        if not nonzero.any():
            return None
        # Use scipy distance transform with indices
        dist, indices = ndimage.distance_transform_edt(~nonzero, return_indices=True)
        result = grid.copy()
        zero_mask = grid == 0
        if zero_mask.any():
            nearest_r = indices[0][zero_mask]
            nearest_c = indices[1][zero_mask]
            result[zero_mask] = grid[nearest_r, nearest_c]
        return result

    def _solve_marker_to_nearest(self, demo_pairs, test_input):
        """Replace a specific marker color with the nearest other non-zero color.

        Detects which color is the 'marker' (gets replaced) and which colors
        it maps to based on proximity.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            in_colors = set(np.unique(inp).tolist())
            out_colors = set(np.unique(out).tolist())

            # Find marker color: present in input, changed in output
            for marker in in_colors:
                if marker == 0:
                    continue
                mask = inp == marker
                if not mask.any():
                    continue
                out_at_marker = set(out[mask].tolist())
                # Marker is replaced by other colors
                if marker in out_at_marker:
                    continue
                if len(out_at_marker) < 1:
                    continue

                # Check: marker replaced by nearest non-marker, non-zero color
                result = inp.copy()
                h, w = inp.shape
                non_marker_nonzero = (inp != 0) & (inp != marker)

                if not non_marker_nonzero.any():
                    continue

                dist, indices = ndimage.distance_transform_edt(
                    ~non_marker_nonzero, return_indices=True
                )
                marker_mask = inp == marker
                nearest_r = indices[0][marker_mask]
                nearest_c = indices[1][marker_mask]
                result[marker_mask] = inp[nearest_r, nearest_c]

                if np.array_equal(result, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        r2 = inp2.copy()
                        nm2 = (inp2 != 0) & (inp2 != marker)
                        if not nm2.any():
                            all_match = False
                            break
                        d2, i2 = ndimage.distance_transform_edt(~nm2, return_indices=True)
                        mm2 = inp2 == marker
                        r2[mm2] = inp2[i2[0][mm2], i2[1][mm2]]
                        if not np.array_equal(r2, out2):
                            all_match = False
                            break
                    if all_match:
                        tr = test_input.copy()
                        tm = (test_input != 0) & (test_input != marker)
                        if tm.any():
                            td, ti = ndimage.distance_transform_edt(~tm, return_indices=True)
                            tmm = test_input == marker
                            tr[tmm] = test_input[ti[0][tmm], ti[1][tmm]]
                        return tr, 1.0, "marker_to_nearest"
        return None, 0.0, "marker_to_nearest"

    def _solve_line_between_markers(self, demo_pairs, test_input):
        """Draw lines between pairs of same-colored markers with a fill color.

        Detects pairs of same-colored pixels and draws a line (horizontal,
        vertical, or diagonal) between them, filling with a specific color.
        """
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

            # Find fill color
            fill_colors = set(out[from_zero[:, 0], from_zero[:, 1]].tolist())
            if len(fill_colors) != 1:
                continue
            fc = fill_colors.pop()

            # Find marker pairs (same color, form lines)
            marker_colors = set(np.unique(inp).tolist()) - {0}
            for mc in marker_colors:
                positions = np.argwhere(inp == mc)
                if len(positions) < 2:
                    continue

                # Try drawing lines between all pairs
                line_pixels = set()
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        r1, c1 = positions[i]
                        r2, c2 = positions[j]
                        # Check if horizontal, vertical, or diagonal
                        if r1 == r2:
                            for c in range(min(c1, c2), max(c1, c2) + 1):
                                line_pixels.add((r1, c))
                        elif c1 == c2:
                            for r in range(min(r1, r2), max(r1, r2) + 1):
                                line_pixels.add((r, c1))
                        elif abs(r1 - r2) == abs(c1 - c2):
                            dr = 1 if r2 > r1 else -1
                            dc = 1 if c2 > c1 else -1
                            steps = abs(r1 - r2)
                            for s in range(steps + 1):
                                line_pixels.add((r1 + s * dr, c1 + s * dc))

                # Check if line pixels match changed pixels
                result = inp.copy()
                for r, c in line_pixels:
                    if 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]:
                        if inp[r, c] == 0:
                            result[r, c] = fc

                if np.array_equal(result, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        r2 = inp2.copy()
                        pos2 = np.argwhere(inp2 == mc)
                        lp2 = set()
                        for i in range(len(pos2)):
                            for j in range(i + 1, len(pos2)):
                                r1, c1 = pos2[i]
                                r2_, c2_ = pos2[j]
                                if r1 == r2_:
                                    for c in range(min(c1, c2_), max(c1, c2_) + 1):
                                        lp2.add((r1, c))
                                elif c1 == c2_:
                                    for r in range(min(r1, r2_), max(r1, r2_) + 1):
                                        lp2.add((r, c1))
                                elif abs(r1 - r2_) == abs(c1 - c2_):
                                    dr = 1 if r2_ > r1 else -1
                                    dc = 1 if c2_ > c1 else -1
                                    steps = abs(r1 - r2_)
                                    for s in range(steps + 1):
                                        lp2.add((r1 + s * dr, c1 + s * dc))
                        for r, c in lp2:
                            if 0 <= r < inp2.shape[0] and 0 <= c < inp2.shape[1]:
                                if inp2[r, c] == 0:
                                    r2[r, c] = fc
                        if not np.array_equal(r2, out2):
                            all_match = False
                            break
                    if all_match:
                        tr = test_input.copy()
                        tpos = np.argwhere(test_input == mc)
                        tlp = set()
                        for i in range(len(tpos)):
                            for j in range(i + 1, len(tpos)):
                                r1, c1 = tpos[i]
                                r2_, c2_ = tpos[j]
                                if r1 == r2_:
                                    for c in range(min(c1, c2_), max(c1, c2_) + 1):
                                        tlp.add((r1, c))
                                elif c1 == c2_:
                                    for r in range(min(r1, r2_), max(r1, r2_) + 1):
                                        tlp.add((r, c1))
                                elif abs(r1 - r2_) == abs(c1 - c2_):
                                    dr = 1 if r2_ > r1 else -1
                                    dc = 1 if c2_ > c1 else -1
                                    steps = abs(r1 - r2_)
                                    for s in range(steps + 1):
                                        tlp.add((r1 + s * dr, c1 + s * dc))
                        for r, c in tlp:
                            if 0 <= r < test_input.shape[0] and 0 <= c < test_input.shape[1]:
                                if test_input[r, c] == 0:
                                    tr[r, c] = fc
                        return tr, 1.0, "line_between_markers"
        return None, 0.0, "line_between_markers"

    def _solve_x_marker(self, demo_pairs, test_input):
        """Draw X-shaped (diagonal) cross around marker pixels.

        Like plus_marker but with diagonal neighbors instead of orthogonal.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            new_pos = np.argwhere((diff != 0) & (inp == 0))
            if len(new_pos) == 0:
                continue
            markers = []
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if inp[r, c] != 0 and out[r, c] == inp[r, c]:
                        markers.append((r, c, int(inp[r, c])))
            mc_map = {}
            consistent = True
            for nr, nc in new_pos:
                plus_c = int(out[nr, nc])
                found = False
                for mr, mc, mcolor in markers:
                    if abs(nr - mr) == 1 and abs(nc - mc) == 1:
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
                            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
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
                            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
                                    if test_input[nr, nc] == 0:
                                        tr[nr, nc] = plus_c
                return tr, 1.0, "x_marker"
        return None, 0.0, "x_marker"

    def _solve_object_expand(self, demo_pairs, test_input):
        """Expand objects by 1 pixel in all directions (dilation with same color).

        Each non-zero pixel expands to fill its 0-valued neighbors with the
        same color. Uses morphological dilation.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            # Check: only 0 pixels changed to non-zero, and non-zero preserved
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue
            from_zero = np.argwhere((inp == 0) & (out != 0))
            to_zero = np.argwhere((inp != 0) & (out == 0))
            if len(to_zero) > 0:
                continue
            if len(from_zero) != len(changed):
                continue

            # Try dilation with color preservation
            result = inp.copy()
            binary = inp != 0
            dilated = ndimage.binary_dilation(binary, iterations=1)
            new_pixels = dilated & ~binary

            # Assign color from nearest non-zero
            dist, indices = ndimage.distance_transform_edt(~binary, return_indices=True)
            result[new_pixels] = inp[indices[0][new_pixels], indices[1][new_pixels]]

            if np.array_equal(result, out):
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    r2 = inp2.copy()
                    b2 = inp2 != 0
                    d2 = ndimage.binary_dilation(b2, iterations=1)
                    np2 = d2 & ~b2
                    _, i2 = ndimage.distance_transform_edt(~b2, return_indices=True)
                    r2[np2] = inp2[i2[0][np2], i2[1][np2]]
                    if not np.array_equal(r2, out2):
                        all_match = False
                        break
                if all_match:
                    tr = test_input.copy()
                    tb = test_input != 0
                    td = ndimage.binary_dilation(tb, iterations=1)
                    tnp = td & ~tb
                    _, ti = ndimage.distance_transform_edt(~tb, return_indices=True)
                    tr[tnp] = test_input[ti[0][tnp], ti[1][tnp]]
                    return tr, 1.0, "object_expand"
        return None, 0.0, "object_expand"

    def _solve_object_contract(self, demo_pairs, test_input):
        """Contract objects by 1 pixel (erosion). Non-zero pixels at the
        border become 0."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue
            to_zero = np.argwhere((inp != 0) & (out == 0))
            from_zero = np.argwhere((inp == 0) & (out != 0))
            if len(from_zero) > 0:
                continue
            if len(to_zero) != len(changed):
                continue

            binary = inp != 0
            eroded = ndimage.binary_erosion(binary, iterations=1)
            border = binary & ~eroded
            result = inp.copy()
            result[border] = 0

            if np.array_equal(result, out):
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    b2 = inp2 != 0
                    e2 = ndimage.binary_erosion(b2, iterations=1)
                    bd2 = b2 & ~e2
                    r2 = inp2.copy()
                    r2[bd2] = 0
                    if not np.array_equal(r2, out2):
                        all_match = False
                        break
                if all_match:
                    tr = test_input.copy()
                    tb = test_input != 0
                    te = ndimage.binary_erosion(tb, iterations=1)
                    tbd = tb & ~te
                    tr[tbd] = 0
                    return tr, 1.0, "object_contract"
        return None, 0.0, "object_contract"

    def _solve_row_insert(self, demo_pairs, test_input):
        """Insert rows to expand the grid. Detects which rows are inserted
        and what pattern they follow."""
        for inp, out in demo_pairs:
            if inp.shape[0] >= out.shape[0]:
                continue
            if inp.shape[1] != out.shape[1]:
                continue

            # Find which rows of output correspond to input rows
            h_in, w = inp.shape
            h_out = out.shape[0]

            # Try: every k-th row is a copy of the previous
            for insert_every in range(2, h_in + 2):
                result = []
                in_idx = 0
                for out_idx in range(h_out):
                    if (out_idx + 1) % insert_every == 0 and in_idx < h_in:
                        # Insert a copy of the previous row
                        if result:
                            result.append(result[-1])
                        else:
                            result.append(inp[0])
                    else:
                        if in_idx < h_in:
                            result.append(inp[in_idx])
                            in_idx += 1
                        elif result:
                            result.append(result[-1])
                if len(result) == h_out:
                    result_arr = np.array(result, dtype=np.int8)
                    if np.array_equal(result_arr, out):
                        all_match = True
                        for inp2, out2 in demo_pairs:
                            if inp2.shape[0] >= out2.shape[0]:
                                all_match = False
                                break
                            r2 = []
                            idx2 = 0
                            for o2 in range(out2.shape[0]):
                                if (o2 + 1) % insert_every == 0 and idx2 < inp2.shape[0]:
                                    r2.append(r2[-1] if r2 else inp2[0])
                                else:
                                    if idx2 < inp2.shape[0]:
                                        r2.append(inp2[idx2])
                                        idx2 += 1
                                    elif r2:
                                        r2.append(r2[-1])
                            if len(r2) != out2.shape[0] or not np.array_equal(np.array(r2, dtype=np.int8), out2):
                                all_match = False
                                break
                        if all_match:
                            tr_rows = []
                            t_idx = 0
                            for o in range(h_out if h_out > test_input.shape[0] else test_input.shape[0] + test_input.shape[0] // (insert_every - 1)):
                                if (o + 1) % insert_every == 0 and t_idx < test_input.shape[0]:
                                    tr_rows.append(tr_rows[-1] if tr_rows else test_input[0])
                                else:
                                    if t_idx < test_input.shape[0]:
                                        tr_rows.append(test_input[t_idx])
                                        t_idx += 1
                                    elif tr_rows:
                                        tr_rows.append(tr_rows[-1])
                            tr = np.array(tr_rows[:test_input.shape[0] + test_input.shape[0] // (insert_every - 1)], dtype=np.int8)
                            return tr, 1.0, "row_insert"
        return None, 0.0, "row_insert"

    def _solve_col_insert(self, demo_pairs, test_input):
        """Insert columns to expand the grid."""
        for inp, out in demo_pairs:
            if inp.shape[1] >= out.shape[1]:
                continue
            if inp.shape[0] != out.shape[0]:
                continue
            w_in, h = inp.shape[1], inp.shape[0]
            w_out = out.shape[1]
            for insert_every in range(2, w_in + 2):
                result = []
                in_idx = 0
                for out_idx in range(w_out):
                    if (out_idx + 1) % insert_every == 0 and in_idx < w_in:
                        result.append(result[-1] if result else inp[:, 0])
                    else:
                        if in_idx < w_in:
                            result.append(inp[:, in_idx])
                            in_idx += 1
                        elif result:
                            result.append(result[-1])
                if len(result) == w_out:
                    result_arr = np.column_stack(result).astype(np.int8)
                    if np.array_equal(result_arr, out):
                        all_match = True
                        for inp2, out2 in demo_pairs:
                            if inp2.shape[1] >= out2.shape[1]:
                                all_match = False
                                break
                            r2 = []
                            idx2 = 0
                            for o2 in range(out2.shape[1]):
                                if (o2 + 1) % insert_every == 0 and idx2 < inp2.shape[1]:
                                    r2.append(r2[-1] if r2 else inp2[:, 0])
                                else:
                                    if idx2 < inp2.shape[1]:
                                        r2.append(inp2[:, idx2])
                                        idx2 += 1
                                    elif r2:
                                        r2.append(r2[-1])
                            if len(r2) != out2.shape[1] or not np.array_equal(np.column_stack(r2).astype(np.int8), out2):
                                all_match = False
                                break
                        if all_match:
                            tr_cols = []
                            t_idx = 0
                            target_w = test_input.shape[1] + test_input.shape[1] // (insert_every - 1)
                            for o in range(target_w):
                                if (o + 1) % insert_every == 0 and t_idx < test_input.shape[1]:
                                    tr_cols.append(tr_cols[-1] if tr_cols else test_input[:, 0])
                                else:
                                    if t_idx < test_input.shape[1]:
                                        tr_cols.append(test_input[:, t_idx])
                                        t_idx += 1
                                    elif tr_cols:
                                        tr_cols.append(tr_cols[-1])
                            tr = np.column_stack(tr_cols[:target_w]).astype(np.int8)
                            return tr, 1.0, "col_insert"
        return None, 0.0, "col_insert"

    def _solve_smart_crop_color(self, demo_pairs, test_input):
        """Crop to the bounding box of a specific color, trying all colors."""
        for inp, out in demo_pairs:
            if inp.shape[0] < out.shape[0] or inp.shape[1] < out.shape[1]:
                continue
            # Try each color as the crop target
            for color in range(10):
                mask = inp == color
                if not mask.any():
                    continue
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if not rows.any() or not cols.any():
                    continue
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                cropped = inp[rmin:rmax + 1, cmin:cmax + 1]
                if cropped.shape == out.shape and np.array_equal(cropped, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape[0] < out2.shape[0] or inp2.shape[1] < out2.shape[1]:
                            all_match = False
                            break
                        m2 = inp2 == color
                        if not m2.any():
                            all_match = False
                            break
                        r2 = np.any(m2, axis=1)
                        c2 = np.any(m2, axis=0)
                        rmin2, rmax2 = np.where(r2)[0][[0, -1]]
                        cmin2, cmax2 = np.where(c2)[0][[0, -1]]
                        cr2 = inp2[rmin2:rmax2 + 1, cmin2:cmax2 + 1]
                        if cr2.shape != out2.shape or not np.array_equal(cr2, out2):
                            all_match = False
                            break
                    if all_match:
                        tm = test_input == color
                        if tm.any():
                            tr = np.any(tm, axis=1)
                            tc = np.any(tm, axis=0)
                            trmin, trmax = np.where(tr)[0][[0, -1]]
                            tcmin, tcmax = np.where(tc)[0][[0, -1]]
                            return test_input[trmin:trmax + 1, tcmin:tcmax + 1], 1.0, f"crop_color_{color}"
        return None, 0.0, "smart_crop_color"

    def _solve_color_count_output(self, demo_pairs, test_input):
        """Output is a small grid representing counts of objects.

        E.g., count the number of objects of each color and output as a row.
        """
        for inp, out in demo_pairs:
            if out.size > 20:
                continue
            # Count objects by color in input
            binary = (inp != 0).astype(int)
            labeled, num = ndimage.label(binary)
            if num == 0:
                continue

            # Count objects per color
            color_counts = {}
            for label in range(1, num + 1):
                mask = labeled == label
                colors = set(inp[mask].tolist())
                if len(colors) == 1:
                    c = colors.pop()
                    color_counts[c] = color_counts.get(c, 0) + 1

            # Check if output matches some count representation
            if out.size == sum(color_counts.values()):
                # Output is 1xN with each cell = color of an object
                all_match = True
                for inp2, out2 in demo_pairs:
                    b2 = (inp2 != 0).astype(int)
                    l2, n2 = ndimage.label(b2)
                    if n2 == 0:
                        all_match = False
                        break
                    cc2 = {}
                    for lb in range(1, n2 + 1):
                        m2 = l2 == lb
                        cs = set(inp2[m2].tolist())
                        if len(cs) == 1:
                            c = cs.pop()
                            cc2[c] = cc2.get(c, 0) + 1
                    if out2.size != sum(cc2.values()):
                        all_match = False
                        break
                if all_match:
                    # Apply to test
                    tb = (test_input != 0).astype(int)
                    tl, tn = ndimage.label(tb)
                    if tn > 0:
                        tcc = {}
                        for lb in range(1, tn + 1):
                            tm = tl == lb
                            cs = set(test_input[tm].tolist())
                            if len(cs) == 1:
                                c = cs.pop()
                                tcc[c] = tcc.get(c, 0) + 1
                        result = []
                        for c in sorted(tcc.keys()):
                            result.extend([c] * tcc[c])
                        return np.array(result, dtype=np.int8).reshape(1, -1), 0.8, "color_count"
        return None, 0.0, "color_count"

    def _solve_partial_tile(self, demo_pairs, test_input):
        """Complete a partial tile: output has a repeating pattern that's
        partially present in input. Fill in the missing parts.

        Handles non-divisible periods by using modulo.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            h, w = out.shape
            # Try all small periods (not requiring divisibility)
            for ph in range(1, min(h, 8) + 1):
                for pw in range(1, min(w, 8) + 1):
                    if ph == h and pw == w:
                        continue
                    # Extract pattern from output
                    pattern = np.zeros((ph, pw), dtype=np.int8)
                    consistent = True
                    for r in range(h):
                        for c in range(w):
                            pr, pc = r % ph, c % pw
                            expected = out[r, c]
                            if pattern[pr, pc] == 0:
                                pattern[pr, pc] = expected
                            elif pattern[pr, pc] != expected:
                                consistent = False
                                break
                        if not consistent:
                            break
                    if not consistent:
                        continue
                    # Check: input non-zero matches pattern
                    input_ok = True
                    for r in range(h):
                        for c in range(w):
                            if inp[r, c] != 0 and inp[r, c] != pattern[r % ph, c % pw]:
                                input_ok = False
                                break
                        if not input_ok:
                            break
                    if input_ok:
                        # Check output = pattern tiled
                        output_ok = True
                        for r in range(h):
                            for c in range(w):
                                if out[r, c] != pattern[r % ph, c % pw]:
                                    output_ok = False
                                    break
                            if not output_ok:
                                break
                        if output_ok:
                            all_match = True
                            for inp2, out2 in demo_pairs:
                                if inp2.shape != out2.shape:
                                    all_match = False
                                    break
                                h2, w2 = out2.shape
                                for r in range(h2):
                                    for c in range(w2):
                                        if out2[r, c] != pattern[r % ph, c % pw]:
                                            all_match = False
                                            break
                                    if not all_match:
                                        break
                                if not all_match:
                                    break
                            if all_match:
                                th, tw = test_input.shape
                                tr = np.zeros_like(test_input)
                                for r in range(th):
                                    for c in range(tw):
                                        tr[r, c] = pattern[r % ph, c % pw]
                                # Preserve test input non-zero pixels
                                tm = test_input != 0
                                tr[tm] = test_input[tm]
                                return tr, 1.0, "partial_tile"
        return None, 0.0, "partial_tile"

    def _solve_pair_symmetry(self, demo_pairs, test_input):
        """Complete patterns using pairwise symmetry between regions.

        Splits the grid into halves and mirrors one half to complete the other.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            h, w = inp.shape

            # Try splitting horizontally and mirroring
            if h >= 2:
                mid = h // 2
                # Top half -> bottom half
                result = inp.copy()
                for r in range(mid, h):
                    sr = h - 1 - r
                    if sr < mid:
                        for c in range(w):
                            if inp[r, c] == 0 and inp[sr, c] != 0:
                                result[r, c] = inp[sr, c]
                if np.array_equal(result, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        h2 = inp2.shape[0]
                        mid2 = h2 // 2
                        r2 = inp2.copy()
                        for r in range(mid2, h2):
                            sr = h2 - 1 - r
                            if sr < mid2:
                                for c in range(inp2.shape[1]):
                                    if inp2[r, c] == 0 and inp2[sr, c] != 0:
                                        r2[r, c] = inp2[sr, c]
                        if not np.array_equal(r2, out2):
                            all_match = False
                            break
                    if all_match:
                        tr = test_input.copy()
                        th = test_input.shape[0]
                        tmid = th // 2
                        for r in range(tmid, th):
                            sr = th - 1 - r
                            if sr < tmid:
                                for c in range(test_input.shape[1]):
                                    if test_input[r, c] == 0 and test_input[sr, c] != 0:
                                        tr[r, c] = test_input[sr, c]
                        return tr, 1.0, "pair_sym_h"

            # Try splitting vertically and mirroring
            if w >= 2:
                mid = w // 2
                result = inp.copy()
                for c in range(mid, w):
                    sc = w - 1 - c
                    if sc < mid:
                        for r in range(h):
                            if inp[r, c] == 0 and inp[r, sc] != 0:
                                result[r, c] = inp[r, sc]
                if np.array_equal(result, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        w2 = inp2.shape[1]
                        mid2 = w2 // 2
                        r2 = inp2.copy()
                        for c in range(mid2, w2):
                            sc = w2 - 1 - c
                            if sc < mid2:
                                for r in range(inp2.shape[0]):
                                    if inp2[r, c] == 0 and inp2[r, sc] != 0:
                                        r2[r, c] = inp2[r, sc]
                        if not np.array_equal(r2, out2):
                            all_match = False
                            break
                    if all_match:
                        tr = test_input.copy()
                        tw = test_input.shape[1]
                        tmid = tw // 2
                        for c in range(tmid, tw):
                            sc = tw - 1 - c
                            if sc < tmid:
                                for r in range(test_input.shape[0]):
                                    if test_input[r, c] == 0 and test_input[r, sc] != 0:
                                        tr[r, c] = test_input[r, sc]
                        return tr, 1.0, "pair_sym_v"
        return None, 0.0, "pair_symmetry"

    def _solve_cc_shape_match(self, demo_pairs, test_input):
        """Match connected components by shape and apply consistent transformation.

        Identifies CCs in input, finds corresponding CCs in output, and
        applies the same transformation to test input CCs.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            # Label CCs in input and output
            binary_in = inp != 0
            labeled_in, num_in = ndimage.label(binary_in)
            binary_out = out != 0
            labeled_out, num_out = ndimage.label(binary_out)

            if num_in == 0 or num_out != num_in:
                continue

            # For each CC, check if it's just recolored
            transform_map = {}  # (shape_signature) -> (color_change)
            consistent = True
            for label in range(1, num_in + 1):
                mask_in = labeled_in == label
                mask_out = labeled_out == label
                if not np.array_equal(mask_in, mask_out):
                    # CCs don't align position-wise, try shape matching
                    consistent = False
                    break
                in_color = int(np.unique(inp[mask_in]).tolist()[0]) if len(np.unique(inp[mask_in])) == 1 else -1
                out_color = int(np.unique(out[mask_out]).tolist()[0]) if len(np.unique(out[mask_out])) == 1 else -1
                if in_color >= 0 and out_color >= 0:
                    transform_map[in_color] = out_color

            if consistent and transform_map:
                has_change = any(k != v for k, v in transform_map.items())
                if has_change:
                    result = test_input.copy()
                    for c, t in transform_map.items():
                        result[test_input == c] = t
                    # Verify
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        r2 = inp2.copy()
                        for c, t in transform_map.items():
                            r2[inp2 == c] = t
                        if not np.array_equal(r2, out2):
                            all_match = False
                            break
                    if all_match:
                        return result, 1.0, "cc_shape_match"
        return None, 0.0, "cc_shape_match"

    def _solve_bg_fill_by_proximity(self, demo_pairs, test_input):
        """Fill background (0) pixels based on proximity to specific colored objects.

        More sophisticated than Voronoi: detects which color 'owns' each
        background pixel based on distance to nearest object of each color.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue
            from_zero = np.argwhere((inp == 0) & (out != 0))
            to_zero = np.argwhere((inp != 0) & (out == 0))
            if len(to_zero) > 0:
                continue
            if len(from_zero) != len(changed):
                continue

            # Check: each background pixel gets the color of the nearest non-zero pixel
            # But only considering pixels of the same "object"
            nonzero = inp != 0
            if not nonzero.any():
                continue

            dist, indices = ndimage.distance_transform_edt(~nonzero, return_indices=True)
            result = inp.copy()
            zero_mask = inp == 0
            result[zero_mask] = inp[indices[0][zero_mask], indices[1][zero_mask]]

            if np.array_equal(result, out):
                all_match = True
                for inp2, out2 in demo_pairs:
                    if inp2.shape != out2.shape:
                        all_match = False
                        break
                    nz2 = inp2 != 0
                    if not nz2.any():
                        all_match = False
                        break
                    _, i2 = ndimage.distance_transform_edt(~nz2, return_indices=True)
                    r2 = inp2.copy()
                    zm2 = inp2 == 0
                    r2[zm2] = inp2[i2[0][zm2], i2[1][zm2]]
                    if not np.array_equal(r2, out2):
                        all_match = False
                        break
                if all_match:
                    tnz = test_input != 0
                    if tnz.any():
                        _, ti = ndimage.distance_transform_edt(~tnz, return_indices=True)
                        tr = test_input.copy()
                        tzm = test_input == 0
                        tr[tzm] = test_input[ti[0][tzm], ti[1][tzm]]
                        return tr, 1.0, "bg_fill_proximity"
        return None, 0.0, "bg_fill_proximity"

    def _solve_color_replace_by_context(self, demo_pairs, test_input):
        """Replace a specific color based on surrounding context.

        Detects which color gets replaced and what it maps to based on
        its local neighborhood.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue

            # Find the color that changes
            changing_colors = set()
            for r, c in changed:
                changing_colors.add(int(inp[r, c]))

            for cc in changing_colors:
                if cc == 0:
                    continue
                mask = inp == cc
                out_vals = out[mask]
                # Check if all changed pixels of this color map to the same target
                changed_mask = mask & (inp != out)
                if not changed_mask.any():
                    continue
                targets = set(out[changed_mask].tolist())
                if len(targets) != 1:
                    continue
                target = targets.pop()

                # Check: unchanged pixels of this color stay the same
                unchanged_mask = mask & (inp == out)
                if unchanged_mask.any():
                    # Some pixels of this color change, others don't
                    # Need to find the distinguishing factor
                    # Try: based on neighbor colors
                    changed_positions = np.argwhere(changed_mask)
                    unchanged_positions = np.argwhere(unchanged_mask)

                    # Check neighbors of changed vs unchanged
                    # If all changed pixels have a specific neighbor color, use that
                    neighbor_colors_changed = set()
                    for r, c in changed_positions:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                                nval = int(inp[nr, nc])
                                if nval != cc and nval != 0:
                                    neighbor_colors_changed.add(nval)

                    neighbor_colors_unchanged = set()
                    for r, c in unchanged_positions:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                                nval = int(inp[nr, nc])
                                if nval != cc and nval != 0:
                                    neighbor_colors_unchanged.add(nval)

                    # If changed pixels have a unique neighbor color
                    unique_changed = neighbor_colors_changed - neighbor_colors_unchanged
                    if unique_changed:
                        trigger_color = unique_changed.pop()
                        result = inp.copy()
                        for r in range(inp.shape[0]):
                            for c in range(inp.shape[1]):
                                if inp[r, c] == cc:
                                    has_trigger = False
                                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                        nr, nc = r + dr, c + dc
                                        if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                                            if inp[nr, nc] == trigger_color:
                                                has_trigger = True
                                                break
                                    if has_trigger:
                                        result[r, c] = target
                        if np.array_equal(result, out):
                            all_match = True
                            for inp2, out2 in demo_pairs:
                                if inp2.shape != out2.shape:
                                    all_match = False
                                    break
                                r2 = inp2.copy()
                                for r in range(inp2.shape[0]):
                                    for c in range(inp2.shape[1]):
                                        if inp2[r, c] == cc:
                                            ht = False
                                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                                nr, nc = r + dr, c + dc
                                                if 0 <= nr < inp2.shape[0] and 0 <= nc < inp2.shape[1]:
                                                    if inp2[nr, nc] == trigger_color:
                                                        ht = True
                                                        break
                                            if ht:
                                                r2[r, c] = target
                                if not np.array_equal(r2, out2):
                                    all_match = False
                                    break
                            if all_match:
                                tr = test_input.copy()
                                for r in range(test_input.shape[0]):
                                    for c in range(test_input.shape[1]):
                                        if test_input[r, c] == cc:
                                            ht = False
                                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                                nr, nc = r + dr, c + dc
                                                if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
                                                    if test_input[nr, nc] == trigger_color:
                                                        ht = True
                                                        break
                                            if ht:
                                                tr[r, c] = target
                                return tr, 1.0, "color_replace_context"
                else:
                    # All pixels of this color change
                    result = inp.copy()
                    result[mask] = target
                    if np.array_equal(result, out):
                        all_match = True
                        for inp2, out2 in demo_pairs:
                            if inp2.shape != out2.shape:
                                all_match = False
                                break
                            r2 = inp2.copy()
                            r2[inp2 == cc] = target
                            if not np.array_equal(r2, out2):
                                all_match = False
                                break
                        if all_match:
                            tr = test_input.copy()
                            tr[test_input == cc] = target
                            return tr, 1.0, "color_replace_all"
        return None, 0.0, "color_replace_context"

    def _solve_extract_pattern_cell(self, demo_pairs, test_input):
        """Extract a pattern cell from the grid based on separator lines.

        Detects grid structure (separator rows/cols) and extracts a specific cell.
        """
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] and inp.shape[1] <= out.shape[1]:
                continue

            # Find separator rows (all same color)
            sep_color = None
            sep_rows = []
            for r in range(inp.shape[0]):
                row = inp[r, :]
                unique = set(row.tolist())
                if len(unique) == 1 and list(unique)[0] != 0:
                    if sep_color is None:
                        sep_color = list(unique)[0]
                    if list(unique)[0] == sep_color:
                        sep_rows.append(r)

            sep_cols = []
            for c in range(inp.shape[1]):
                col = inp[:, c]
                unique = set(col.tolist())
                if len(unique) == 1 and list(unique)[0] == sep_color:
                    sep_cols.append(c)

            if not sep_rows and not sep_cols:
                continue

            # Split into cells
            row_bounds = [0] + [r + 1 for r in sep_rows] + [inp.shape[0]]
            col_bounds = [0] + [c + 1 for c in sep_cols] + [inp.shape[1]]

            cells = []
            for i in range(len(row_bounds) - 1):
                for j in range(len(col_bounds) - 1):
                    r_start = row_bounds[i]
                    r_end = row_bounds[i + 1] - 1 if i + 1 < len(row_bounds) - 1 else row_bounds[i + 1]
                    if i + 1 < len(row_bounds) - 1:
                        r_end = sep_rows[i]  # exclusive
                    else:
                        r_end = inp.shape[0]
                    c_start = col_bounds[j]
                    if j + 1 < len(col_bounds) - 1:
                        c_end = sep_cols[j]
                    else:
                        c_end = inp.shape[1]
                    cell = inp[r_start:r_end, c_start:c_end]
                    cells.append(cell)

            # Check if any cell matches the output
            for cell in cells:
                if cell.shape == out.shape and np.array_equal(cell, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape[0] <= out2.shape[0] and inp2.shape[1] <= out2.shape[1]:
                            all_match = False
                            break
                        # Find same sep structure
                        sr2 = []
                        sc2 = []
                        for r in range(inp2.shape[0]):
                            row = inp2[r, :]
                            if len(set(row.tolist())) == 1 and list(set(row.tolist()))[0] == sep_color:
                                sr2.append(r)
                        for c in range(inp2.shape[1]):
                            col = inp2[:, c]
                            if len(set(col.tolist())) == 1 and list(set(col.tolist()))[0] == sep_color:
                                sc2.append(c)
                        rb2 = [0] + [r + 1 for r in sr2] + [inp2.shape[0]]
                        cb2 = [0] + [c + 1 for c in sc2] + [inp2.shape[1]]
                        found = False
                        for i in range(len(rb2) - 1):
                            for j in range(len(cb2) - 1):
                                rs = rb2[i]
                                re_ = sr2[i] if i < len(sr2) else inp2.shape[0]
                                cs = cb2[j]
                                ce_ = sc2[j] if j < len(sc2) else inp2.shape[1]
                                cell2 = inp2[rs:re_, cs:ce_]
                                if cell2.shape == out2.shape and np.array_equal(cell2, out2):
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            all_match = False
                            break
                    if all_match:
                        # Apply to test - extract first cell
                        tsr = []
                        tsc = []
                        for r in range(test_input.shape[0]):
                            row = test_input[r, :]
                            if len(set(row.tolist())) == 1 and list(set(row.tolist()))[0] == sep_color:
                                tsr.append(r)
                        for c in range(test_input.shape[1]):
                            col = test_input[:, c]
                            if len(set(col.tolist())) == 1 and list(set(col.tolist()))[0] == sep_color:
                                tsc.append(c)
                        if tsr and tsc:
                            return test_input[0:tsr[0], 0:tsc[0]], 1.0, "extract_cell"
        return None, 0.0, "extract_pattern_cell"

    def _solve_majority_color_crop(self, demo_pairs, test_input):
        """Crop to region where a specific color is the majority.

        Finds the dominant non-zero color and crops to its bounding box.
        """
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] and inp.shape[1] <= out.shape[1]:
                continue

            # Find the most common non-zero color
            colors, counts = np.unique(inp[inp != 0], return_counts=True)
            if len(colors) == 0:
                continue

            # Try each color as the crop target
            for ci, color in enumerate(colors):
                mask = inp == color
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if not rows.any():
                    continue
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                cropped = inp[rmin:rmax + 1, cmin:cmax + 1]
                if cropped.shape == out.shape and np.array_equal(cropped, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape[0] <= out2.shape[0] and inp2.shape[1] <= out2.shape[1]:
                            all_match = False
                            break
                        m2 = inp2 == color
                        if not m2.any():
                            all_match = False
                            break
                        r2 = np.any(m2, axis=1)
                        c2 = np.any(m2, axis=0)
                        rmin2, rmax2 = np.where(r2)[0][[0, -1]]
                        cmin2, cmax2 = np.where(c2)[0][[0, -1]]
                        cr2 = inp2[rmin2:rmax2 + 1, cmin2:cmax2 + 1]
                        if cr2.shape != out2.shape or not np.array_equal(cr2, out2):
                            all_match = False
                            break
                    if all_match:
                        tm = test_input == color
                        if tm.any():
                            tr = np.any(tm, axis=1)
                            tc = np.any(tm, axis=0)
                            trmin, trmax = np.where(tr)[0][[0, -1]]
                            tcmin, tcmax = np.where(tc)[0][[0, -1]]
                            return test_input[trmin:trmax + 1, tcmin:tcmax + 1], 1.0, f"majority_crop_{color}"
        return None, 0.0, "majority_color_crop"

    def _solve_flip_and_merge(self, demo_pairs, test_input):
        """Merge input with its flip to complete the pattern.

        Combines input with its horizontal/vertical/rotational flip,
        filling in 0 pixels from the flipped version.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue

            for flip_fn, name in [
                (np.fliplr, "flip_h_merge"),
                (np.flipud, "flip_v_merge"),
                (lambda x: np.rot90(x, 2), "rot180_merge"),
                (lambda x: np.rot90(x, 1), "rot90_merge"),
                (lambda x: np.rot90(x, 3), "rot270_merge"),
                (np.transpose, "transpose_merge"),
            ]:
                flipped = flip_fn(inp)
                result = inp.copy()
                mask = (inp == 0) & (flipped != 0)
                result[mask] = flipped[mask]
                if np.array_equal(result, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        f2 = flip_fn(inp2)
                        r2 = inp2.copy()
                        m2 = (inp2 == 0) & (f2 != 0)
                        r2[m2] = f2[m2]
                        if not np.array_equal(r2, out2):
                            all_match = False
                            break
                    if all_match:
                        tr = test_input.copy()
                        tf = flip_fn(test_input)
                        tm = (test_input == 0) & (tf != 0)
                        tr[tm] = tf[tm]
                        return tr, 1.0, name
        return None, 0.0, "flip_merge"

    def _solve_remove_color_rows(self, demo_pairs, test_input):
        """Remove rows or columns that are all a specific color (separator).

        Detects and removes separator rows/columns to decrease grid size.
        """
        for inp, out in demo_pairs:
            if inp.shape[0] < out.shape[0] and inp.shape[1] < out.shape[1]:
                continue
            if inp.shape == out.shape:
                continue

            # Find separator rows (all same non-zero color)
            for sep_color in range(1, 10):
                sep_rows = [r for r in range(inp.shape[0]) if len(set(inp[r, :].tolist())) == 1 and inp[r, 0] == sep_color]
                sep_cols = [c for c in range(inp.shape[1]) if len(set(inp[:, c].tolist())) == 1 and inp[0, c] == sep_color]

                if sep_rows or sep_cols:
                    keep_rows = [r for r in range(inp.shape[0]) if r not in sep_rows]
                    keep_cols = [c for c in range(inp.shape[1]) if c not in sep_cols]
                    result = inp[np.ix_(keep_rows, keep_cols)]
                    if result.shape == out.shape and np.array_equal(result, out):
                        all_match = True
                        for inp2, out2 in demo_pairs:
                            sr2 = [r for r in range(inp2.shape[0]) if len(set(inp2[r, :].tolist())) == 1 and inp2[r, 0] == sep_color]
                            sc2 = [c for c in range(inp2.shape[1]) if len(set(inp2[:, c].tolist())) == 1 and inp2[0, c] == sep_color]
                            kr2 = [r for r in range(inp2.shape[0]) if r not in sr2]
                            kc2 = [c for c in range(inp2.shape[1]) if c not in sc2]
                            r2 = inp2[np.ix_(kr2, kc2)]
                            if r2.shape != out2.shape or not np.array_equal(r2, out2):
                                all_match = False
                                break
                        if all_match:
                            tsr = [r for r in range(test_input.shape[0]) if len(set(test_input[r, :].tolist())) == 1 and test_input[r, 0] == sep_color]
                            tsc = [c for c in range(test_input.shape[1]) if len(set(test_input[:, c].tolist())) == 1 and test_input[0, c] == sep_color]
                            tkr = [r for r in range(test_input.shape[0]) if r not in tsr]
                            tkc = [c for c in range(test_input.shape[1]) if c not in tsc]
                            return test_input[np.ix_(tkr, tkc)], 1.0, f"remove_sep_{sep_color}"
        return None, 0.0, "remove_sep"

    def _solve_grid_line_fill(self, demo_pairs, test_input):
        """Fill grid cells with a color based on grid structure.

        Detects a grid structure (separated by lines) and fills cells
        with a specific color based on their position or content.
        """
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

            # Find separator lines
            for sep_color in range(1, 10):
                sep_rows = [r for r in range(inp.shape[0]) if len(set(inp[r, :].tolist())) == 1 and inp[r, 0] == sep_color]
                sep_cols = [c for c in range(inp.shape[1]) if len(set(inp[:, c].tolist())) == 1 and inp[0, c] == sep_color]

                if not sep_rows and not sep_cols:
                    continue

                # Get cell boundaries
                row_bounds = [0] + [r for r in sep_rows] + [inp.shape[0]]
                col_bounds = [0] + [c for c in sep_cols] + [inp.shape[1]]

                # For each cell, check if it gets filled
                fill_colors = set(out[from_zero[:, 0], from_zero[:, 1]].tolist())
                if len(fill_colors) != 1:
                    continue
                fc = fill_colors.pop()

                # Check: cells with any non-zero content get filled with fc
                result = inp.copy()
                for i in range(len(row_bounds) - 1):
                    for j in range(len(col_bounds) - 1):
                        r_start = row_bounds[i] + (1 if i > 0 else 0)
                        r_end = row_bounds[i + 1] if i + 1 < len(row_bounds) - 1 else row_bounds[i + 1]
                        c_start = col_bounds[j] + (1 if j > 0 else 0)
                        c_end = col_bounds[j + 1] if j + 1 < len(col_bounds) - 1 else col_bounds[j + 1]

                        cell = inp[r_start:r_end, c_start:c_end]
                        if cell.size > 0 and (cell != 0).any():
                            result[r_start:r_end, c_start:c_end] = np.where(
                                cell == 0, fc, cell
                            )

                if np.array_equal(result, out):
                    all_match = True
                    for inp2, out2 in demo_pairs:
                        if inp2.shape != out2.shape:
                            all_match = False
                            break
                        sr2 = [r for r in range(inp2.shape[0]) if len(set(inp2[r, :].tolist())) == 1 and inp2[r, 0] == sep_color]
                        sc2 = [c for c in range(inp2.shape[1]) if len(set(inp2[:, c].tolist())) == 1 and inp2[0, c] == sep_color]
                        rb2 = [0] + sr2 + [inp2.shape[0]]
                        cb2 = [0] + sc2 + [inp2.shape[1]]
                        r2 = inp2.copy()
                        for i in range(len(rb2) - 1):
                            for j in range(len(cb2) - 1):
                                rs = rb2[i] + (1 if i > 0 else 0)
                                re_ = rb2[i + 1] if i + 1 < len(rb2) - 1 else rb2[i + 1]
                                cs = cb2[j] + (1 if j > 0 else 0)
                                ce_ = cb2[j + 1] if j + 1 < len(cb2) - 1 else cb2[j + 1]
                                cell2 = inp2[rs:re_, cs:ce_]
                                if cell2.size > 0 and (cell2 != 0).any():
                                    r2[rs:re_, cs:ce_] = np.where(cell2 == 0, fc, cell2)
                        if not np.array_equal(r2, out2):
                            all_match = False
                            break
                    if all_match:
                        tr = test_input.copy()
                        tsr = [r for r in range(test_input.shape[0]) if len(set(test_input[r, :].tolist())) == 1 and test_input[r, 0] == sep_color]
                        tsc = [c for c in range(test_input.shape[1]) if len(set(test_input[:, c].tolist())) == 1 and test_input[0, c] == sep_color]
                        trb = [0] + tsr + [test_input.shape[0]]
                        tcb = [0] + tsc + [test_input.shape[1]]
                        for i in range(len(trb) - 1):
                            for j in range(len(tcb) - 1):
                                rs = trb[i] + (1 if i > 0 else 0)
                                re_ = trb[i + 1] if i + 1 < len(trb) - 1 else trb[i + 1]
                                cs = tcb[j] + (1 if j > 0 else 0)
                                ce_ = tcb[j + 1] if j + 1 < len(tcb) - 1 else tcb[j + 1]
                                cell_t = test_input[rs:re_, cs:ce_]
                                if cell_t.size > 0 and (cell_t != 0).any():
                                    tr[rs:re_, cs:ce_] = np.where(cell_t == 0, fc, cell_t)
                        return tr, 1.0, "grid_line_fill"
        return None, 0.0, "grid_line_fill"

    # ==================== V3.2 Solvers ====================

    def _solve_cross_marker_combined(self, demo_pairs, test_input):
        """Different colored markers get different cross patterns (X vs +).
        E.g. color 2 -> diagonal X with fill color A, color 1 -> orthogonal + with fill color B.
        """
        # Build marker->cross_type->fill_color mapping from demos
        marker_rules = {}  # {marker_color: (cross_type, fill_color)}
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "cross_marker"
            diff = out.astype(int) - inp.astype(int)
            new_pixels = np.argwhere(diff != 0)
            if len(new_pixels) == 0:
                continue
            # Find markers (non-zero in input that are surrounded by new pixels)
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if inp[r, c] == 0:
                        continue
                    mc = int(inp[r, c])
                    # Check what new pixels appeared around this marker
                    neighbors_ortho = []  # up/down/left/right
                    neighbors_diag = []   # diagonal
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                            if inp[nr, nc] == 0 and out[nr, nc] != 0:
                                neighbors_ortho.append(int(out[nr, nc]))
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                            if inp[nr, nc] == 0 and out[nr, nc] != 0:
                                neighbors_diag.append(int(out[nr, nc]))
                    if neighbors_ortho and not neighbors_diag:
                        fc = neighbors_ortho[0]
                        if not all(v == fc for v in neighbors_ortho):
                            continue
                        if mc in marker_rules and marker_rules[mc] != ("ortho", fc):
                            return None, 0.0, "cross_marker"
                        marker_rules[mc] = ("ortho", fc)
                    elif neighbors_diag and not neighbors_ortho:
                        fc = neighbors_diag[0]
                        if not all(v == fc for v in neighbors_diag):
                            continue
                        if mc in marker_rules and marker_rules[mc] != ("diag", fc):
                            return None, 0.0, "cross_marker"
                        marker_rules[mc] = ("diag", fc)
                    elif neighbors_ortho and neighbors_diag:
                        # Both - pick the dominant one
                        if len(neighbors_ortho) > len(neighbors_diag):
                            fc = neighbors_ortho[0]
                            marker_rules[mc] = ("ortho", fc)
                        else:
                            fc = neighbors_diag[0]
                            marker_rules[mc] = ("diag", fc)

        if not marker_rules:
            return None, 0.0, "cross_marker"

        # Apply to test input
        result = test_input.copy()
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                mc = int(test_input[r, c])
                if mc in marker_rules:
                    cross_type, fc = marker_rules[mc]
                    if cross_type == "ortho":
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                                if result[nr, nc] == 0:
                                    result[nr, nc] = fc
                    elif cross_type == "diag":
                        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                                if result[nr, nc] == 0:
                                    result[nr, nc] = fc
        return result, 0.9, "cross_marker"

    def _solve_bg_fill_adjacent(self, demo_pairs, test_input):
        """Fill background (0) pixels with the color of adjacent non-zero pixels."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "bg_fill_adj"
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                continue
            # Check: all changed pixels are 0->X
            all_zero_to_nonzero = all(inp[r, c] == 0 and out[r, c] != 0 for r, c in changed)
            if not all_zero_to_nonzero:
                continue

        # Determine fill rule: each 0 pixel gets filled with a neighbor's color
        # Try: fill with the most common adjacent non-zero color
        def apply_fill(grid):
            result = grid.copy()
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if grid[r, c] == 0:
                        neighbors = []
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                                if grid[nr, nc] != 0:
                                    neighbors.append(int(grid[nr, nc]))
                        if neighbors:
                            # Most common
                            from collections import Counter
                            result[r, c] = Counter(neighbors).most_common(1)[0][0]
            return result

        # Verify on demos
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                continue
            pred = apply_fill(inp)
            if not np.array_equal(pred, out):
                return None, 0.0, "bg_fill_adj"

        return apply_fill(test_input), 1.0, "bg_fill_adj"

    def _solve_remove_border(self, demo_pairs, test_input):
        """Remove uniform-colored border rows/columns from edges."""
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] or inp.shape[1] <= out.shape[1]:
                return None, 0.0, "remove_border"

        # Determine how many border rows/cols to remove
        for inp, out in demo_pairs:
            top_remove = 0
            while top_remove < inp.shape[0] - out.shape[0]:
                row = inp[top_remove, :]
                if len(set(row.tolist())) == 1:
                    top_remove += 1
                else:
                    break
            bottom_remove = 0
            while bottom_remove < inp.shape[0] - out.shape[0] - top_remove:
                row = inp[inp.shape[0] - 1 - bottom_remove, :]
                if len(set(row.tolist())) == 1:
                    bottom_remove += 1
                else:
                    break
            left_remove = 0
            while left_remove < inp.shape[1] - out.shape[1]:
                col = inp[:, left_remove]
                if len(set(col.tolist())) == 1:
                    left_remove += 1
                else:
                    break
            right_remove = 0
            while right_remove < inp.shape[1] - out.shape[1] - left_remove:
                col = inp[:, inp.shape[1] - 1 - right_remove]
                if len(set(col.tolist())) == 1:
                    right_remove += 1
                else:
                    break
            expected_out = inp[top_remove:inp.shape[0] - bottom_remove, left_remove:inp.shape[1] - right_remove]
            if not np.array_equal(expected_out, out):
                return None, 0.0, "remove_border"

        # Apply to test
        inp0 = demo_pairs[0][0]
        top_remove = 0
        while top_remove < inp0.shape[0]:
            if len(set(inp0[top_remove, :].tolist())) == 1:
                top_remove += 1
            else:
                break
        bottom_remove = 0
        while bottom_remove < inp0.shape[0] - top_remove:
            if len(set(inp0[inp0.shape[0] - 1 - bottom_remove, :].tolist())) == 1:
                bottom_remove += 1
            else:
                break
        left_remove = 0
        while left_remove < inp0.shape[1]:
            if len(set(inp0[:, left_remove].tolist())) == 1:
                left_remove += 1
            else:
                break
        right_remove = 0
        while right_remove < inp0.shape[1] - left_remove:
            if len(set(inp0[:, inp0.shape[1] - 1 - right_remove].tolist())) == 1:
                right_remove += 1
            else:
                break
        result = test_input[top_remove:test_input.shape[0] - bottom_remove,
                            left_remove:test_input.shape[1] - right_remove]
        if result.size == 0:
            return None, 0.0, "remove_border"
        return result, 1.0, "remove_border"

    def _solve_extract_inner(self, demo_pairs, test_input):
        """Extract inner region by removing uniform border (1 layer)."""
        for inp, out in demo_pairs:
            if inp.shape[0] <= 2 or inp.shape[1] <= 2:
                return None, 0.0, "extract_inner"
            expected = inp[1:-1, 1:-1]
            if not np.array_equal(expected, out):
                return None, 0.0, "extract_inner"
        return test_input[1:-1, 1:-1], 1.0, "extract_inner"

    def _solve_repeat_extend(self, demo_pairs, test_input):
        """Repeat/extend pattern to make output larger (e.g. 6->9 rows = 1.5x)."""
        for inp, out in demo_pairs:
            if inp.shape[0] >= out.shape[0] or inp.shape[1] >= out.shape[1]:
                return None, 0.0, "repeat_extend"
            # Check if output is a vertical repeat of input
            ratio_r = out.shape[0] / inp.shape[0]
            ratio_c = out.shape[1] / inp.shape[1]
            if ratio_r == int(ratio_r) and ratio_c == 1:
                n = int(ratio_r)
                pred = np.tile(inp, (n, 1))
                if np.array_equal(pred, out):
                    return np.tile(test_input, (n, 1)), 1.0, "repeat_v"
            if ratio_c == int(ratio_c) and ratio_r == 1:
                n = int(ratio_c)
                pred = np.tile(inp, (1, n))
                if np.array_equal(pred, out):
                    return np.tile(test_input, (1, n)), 1.0, "repeat_h"
            # Check partial repeat (e.g. 6->9 = repeat 6 + first 3)
            if ratio_r > 1 and ratio_c == 1:
                extra = out.shape[0] - inp.shape[0]
                if extra < inp.shape[0]:
                    pred = np.vstack([inp, inp[:extra, :]])
                    if np.array_equal(pred, out):
                        return np.vstack([test_input, test_input[:extra, :]]), 1.0, "repeat_partial_v"
            if ratio_c > 1 and ratio_r == 1:
                extra = out.shape[1] - inp.shape[1]
                if extra < inp.shape[1]:
                    pred = np.hstack([inp, inp[:, :extra]])
                    if np.array_equal(pred, out):
                        return np.hstack([test_input, test_input[:, :extra]]), 1.0, "repeat_partial_h"
            # Check repeat with color change
            if ratio_r == int(ratio_r) and ratio_c == 1:
                n = int(ratio_r)
                # Check if each repeat block has a different color
                color_map = {}
                consistent = True
                for rep in range(n):
                    block = out[rep * inp.shape[0]:(rep + 1) * inp.shape[0], :]
                    for c in range(10):
                        mask = inp == c
                        if mask.any():
                            out_vals = set(block[mask].tolist())
                            if len(out_vals) == 1:
                                target = int(out_vals.pop())
                                if c in color_map and color_map[c] != target:
                                    consistent = False
                                color_map[c] = target
                if consistent and color_map:
                    result = np.tile(test_input, (n, 1))
                    for c, target in color_map.items():
                        result[test_input == c] = target
                    # Verify on demos
                    all_ok = True
                    for inp2, out2 in demo_pairs:
                        r2 = np.tile(inp2, (n, 1))
                        for c, t in color_map.items():
                            r2[inp2 == c] = t
                        if not np.array_equal(r2, out2):
                            all_ok = False
                            break
                    if all_ok:
                        return result, 1.0, "repeat_recolor"
        return None, 0.0, "repeat_extend"

    def _solve_extract_grid_content(self, demo_pairs, test_input):
        """Extract content from grid cells (grid divided by separator lines).
        The output is the content of one cell or a combination of cells.
        """
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] or inp.shape[1] <= out.shape[1]:
                return None, 0.0, "extract_grid"

        for inp, out in demo_pairs:
            # Find separator rows/cols (uniform color)
            for sep_color in range(1, 10):
                sep_rows = [r for r in range(inp.shape[0])
                           if len(set(inp[r, :].tolist())) == 1 and inp[r, 0] == sep_color]
                sep_cols = [c for c in range(inp.shape[1])
                           if len(set(inp[:, c].tolist())) == 1 and inp[0, c] == sep_color]
                if not sep_rows and not sep_cols:
                    continue
                row_bounds = [0] + sep_rows + [inp.shape[0]]
                col_bounds = [0] + sep_cols + [inp.shape[1]]
                # Extract each cell
                cells = []
                for i in range(len(row_bounds) - 1):
                    for j in range(len(col_bounds) - 1):
                        r_start = row_bounds[i] + (1 if i > 0 else 0)
                        r_end = row_bounds[i + 1]
                        c_start = col_bounds[j] + (1 if j > 0 else 0)
                        c_end = col_bounds[j + 1]
                        cell = inp[r_start:r_end, c_start:c_end]
                        cells.append(cell)
                # Check if output matches one of the cells
                for cell in cells:
                    if cell.shape == out.shape and np.array_equal(cell, out):
                        # Found it - extract same cell from test
                        sep_rows_t = [r for r in range(test_input.shape[0])
                                     if len(set(test_input[r, :].tolist())) == 1 and test_input[r, 0] == sep_color]
                        sep_cols_t = [c for c in range(test_input.shape[1])
                                     if len(set(test_input[:, c].tolist())) == 1 and test_input[0, c] == sep_color]
                        row_bounds_t = [0] + sep_rows_t + [test_input.shape[0]]
                        col_bounds_t = [0] + sep_cols_t + [test_input.shape[1]]
                        cells_t = []
                        for i in range(len(row_bounds_t) - 1):
                            for j in range(len(col_bounds_t) - 1):
                                r_start = row_bounds_t[i] + (1 if i > 0 else 0)
                                r_end = row_bounds_t[i + 1]
                                c_start = col_bounds_t[j] + (1 if j > 0 else 0)
                                c_end = col_bounds_t[j + 1]
                                cells_t.append(test_input[r_start:r_end, c_start:c_end])
                        # Find the cell that has the same position pattern
                        cell_idx = None
                        for idx, ct in enumerate(cells_t):
                            if ct.shape == out.shape:
                                cell_idx = idx
                                break
                        if cell_idx is not None:
                            return cells_t[cell_idx], 0.9, "extract_grid"
        return None, 0.0, "extract_grid"

    def _solve_extract_largest_obj(self, demo_pairs, test_input):
        """Extract the largest connected component (non-background)."""
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] or inp.shape[1] <= out.shape[1]:
                return None, 0.0, "extract_largest"

        for inp, out in demo_pairs:
            # Find all non-zero connected components
            best_match = None
            for color in range(1, 10):
                mask = (inp == color) | (inp == 0)  # Include background in component
                labeled, num = ndimage.label(inp != 0)
                for label_id in range(1, num + 1):
                    comp_mask = labeled == label_id
                    rows = np.any(comp_mask, axis=1)
                    cols = np.any(comp_mask, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    sub = inp[rmin:rmax + 1, cmin:cmax + 1]
                    if sub.shape == out.shape and np.array_equal(sub, out):
                        best_match = (label_id, color)
                        break
                if best_match:
                    break
            if not best_match:
                return None, 0.0, "extract_largest"

        # Apply to test - find largest component
        labeled, num = ndimage.label(test_input != 0)
        if num == 0:
            return None, 0.0, "extract_largest"
        sizes = ndimage.sum(np.ones_like(test_input), labeled, range(1, num + 1))
        largest_label = np.argmax(sizes) + 1
        comp_mask = labeled == largest_label
        rows = np.any(comp_mask, axis=1)
        cols = np.any(comp_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        result = test_input[rmin:rmax + 1, cmin:cmax + 1]
        return result, 0.9, "extract_largest"

    def _solve_extract_separator_col(self, demo_pairs, test_input):
        """Extract content between/around separator columns.
        E.g. 5x7 -> 5x3 where column 3 is a separator (color 1).
        """
        for inp, out in demo_pairs:
            if inp.shape[0] != out.shape[0] or inp.shape[1] <= out.shape[1]:
                return None, 0.0, "extract_sep_col"

        # Find which columns are removed
        for inp, out in demo_pairs:
            removed_cols = []
            out_col_idx = 0
            for c in range(inp.shape[1]):
                if out_col_idx < out.shape[1] and np.array_equal(inp[:, c], out[:, out_col_idx]):
                    out_col_idx += 1
                else:
                    removed_cols.append(c)
            if not removed_cols:
                return None, 0.0, "extract_sep_col"
            # Check if removed columns are uniform (separators)
            all_sep = True
            for c in removed_cols:
                col = inp[:, c]
                if len(set(col.tolist())) > 1:
                    all_sep = False
                    break
            if not all_sep:
                return None, 0.0, "extract_sep_col"

        # Apply to test: remove columns that are uniform (same separator color)
        sep_colors = set()
        for inp, out in demo_pairs:
            removed_cols = []
            out_col_idx = 0
            for c in range(inp.shape[1]):
                if out_col_idx < out.shape[1] and np.array_equal(inp[:, c], out[:, out_col_idx]):
                    out_col_idx += 1
                else:
                    removed_cols.append(c)
            for c in removed_cols:
                if len(set(inp[:, c].tolist())) == 1:
                    sep_colors.add(int(inp[0, c]))

        keep_cols = [c for c in range(test_input.shape[1])
                     if not (len(set(test_input[:, c].tolist())) == 1 and int(test_input[0, c]) in sep_colors)]
        if not keep_cols:
            return None, 0.0, "extract_sep_col"
        result = test_input[:, keep_cols]
        return result, 0.9, "extract_sep_col"

    def _solve_extract_row_pattern(self, demo_pairs, test_input):
        """Extract a specific row or column from the input as the output."""
        for inp, out in demo_pairs:
            if out.shape[0] != 1 and out.shape[1] != 1:
                return None, 0.0, "extract_row"

        # Try extracting a row
        for inp, out in demo_pairs:
            if out.shape[0] == 1:
                found = False
                for r in range(inp.shape[0]):
                    if np.array_equal(inp[r:r + 1, :], out):
                        found = True
                        break
                if not found:
                    return None, 0.0, "extract_row"
            elif out.shape[1] == 1:
                found = False
                for c in range(inp.shape[1]):
                    if np.array_equal(inp[:, c:c + 1], out):
                        found = True
                        break
                if not found:
                    return None, 0.0, "extract_row"

        # Find which row/col index to extract
        for inp, out in demo_pairs:
            if out.shape[0] == 1:
                for r in range(inp.shape[0]):
                    if np.array_equal(inp[r:r + 1, :], out):
                        if r < test_input.shape[0]:
                            return test_input[r:r + 1, :], 0.9, "extract_row"
            elif out.shape[1] == 1:
                for c in range(inp.shape[1]):
                    if np.array_equal(inp[:, c:c + 1], out):
                        if c < test_input.shape[1]:
                            return test_input[:, c:c + 1], 0.9, "extract_col"
        return None, 0.0, "extract_row"

    def _solve_count_output(self, demo_pairs, test_input):
        """Output is a small grid representing a count (e.g. 1x1 with a number)."""
        for inp, out in demo_pairs:
            if out.size > 9:
                return None, 0.0, "count_output"

        # Try: count of each color
        for inp, out in demo_pairs:
            if out.shape == (1, 1):
                val = int(out[0, 0])
                # Count non-zero pixels
                count = int(np.sum(inp != 0))
                if count != val:
                    # Count specific colors
                    for c in range(1, 10):
                        if int(np.sum(inp == c)) == val:
                            break
                    else:
                        return None, 0.0, "count_output"
            else:
                return None, 0.0, "count_output"

        # Determine what to count
        for inp, out in demo_pairs:
            if out.shape == (1, 1):
                val = int(out[0, 0])
                count_nonzero = int(np.sum(inp != 0))
                if count_nonzero == val:
                    return np.array([[int(np.sum(test_input != 0))]], dtype=np.int8), 0.9, "count_nonzero"
                for c in range(1, 10):
                    if int(np.sum(inp == c)) == val:
                        return np.array([[int(np.sum(test_input == c))]], dtype=np.int8), 0.9, f"count_color_{c}"
        return None, 0.0, "count_output"

    def _solve_position_color_map(self, demo_pairs, test_input):
        """Position-dependent color mapping: out[r,c] = f(inp[r,c], r, c)."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "pos_color_map"

        # Build position-color mapping
        # For each (r, c) position, check if inp[r,c] -> out[r,c] is consistent
        pos_map = {}  # {(r, c): (in_color, out_color)}
        consistent = True
        for inp, out in demo_pairs:
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    key = (r, c)
                    ic = int(inp[r, c])
                    oc = int(out[r, c])
                    if ic == oc:
                        continue
                    if key in pos_map:
                        if pos_map[key] != (ic, oc):
                            consistent = False
                            break
                    pos_map[key] = (ic, oc)
            if not consistent:
                break

        if not consistent or not pos_map:
            return None, 0.0, "pos_color_map"

        # Check if test_input has same shape
        inp0 = demo_pairs[0][0]
        if test_input.shape != inp0.shape:
            return None, 0.0, "pos_color_map"

        result = test_input.copy()
        for (r, c), (ic, oc) in pos_map.items():
            if r < result.shape[0] and c < result.shape[1]:
                if result[r, c] == ic:
                    result[r, c] = oc
        return result, 0.9, "pos_color_map"

    def _solve_draw_outline(self, demo_pairs, test_input):
        """Draw outline/border around non-zero objects."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "draw_outline"

        # Check if output adds border pixels around objects
        for inp, out in demo_pairs:
            diff = out.astype(int) - inp.astype(int)
            new_pixels = np.argwhere(diff != 0)
            if len(new_pixels) == 0:
                return None, 0.0, "draw_outline"
            # Check: new pixels should be adjacent to existing non-zero pixels
            for r, c in new_pixels:
                if inp[r, c] != 0:
                    return None, 0.0, "draw_outline"  # Changed existing pixel
            # Find the fill color
            fill_colors = set(int(out[r, c]) for r, c in new_pixels)
            if len(fill_colors) != 1:
                return None, 0.0, "draw_outline"
            fc = fill_colors.pop()

        # Determine outline rule: for each 0 pixel adjacent to non-zero, fill with fc
        fc = None
        for inp, out in demo_pairs:
            diff = out.astype(int) - inp.astype(int)
            new_pixels = np.argwhere(diff != 0)
            if len(new_pixels) > 0:
                fc = int(out[new_pixels[0][0], new_pixels[0][1]])
                break

        if fc is None:
            return None, 0.0, "draw_outline"

        def apply_outline(grid, fill_color):
            result = grid.copy()
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if grid[r, c] == 0:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                                if grid[nr, nc] != 0:
                                    result[r, c] = fill_color
                                    break
            return result

        # Verify
        for inp, out in demo_pairs:
            if not np.array_equal(apply_outline(inp, fc), out):
                return None, 0.0, "draw_outline"
        return apply_outline(test_input, fc), 1.0, "draw_outline"

    def _solve_color_propagate(self, demo_pairs, test_input):
        """Propagate colors along rows or columns.
        E.g. fill 0s in a row with the nearest non-zero color.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "color_prop"

        # Try row propagation: fill 0s with nearest non-zero in same row
        def apply_row_prop(grid):
            result = grid.copy()
            for r in range(grid.shape[0]):
                row = result[r, :]
                # Forward fill
                last = 0
                for c in range(grid.shape[1]):
                    if row[c] != 0:
                        last = row[c]
                    elif last != 0:
                        row[c] = last
                # Backward fill
                last = 0
                for c in range(grid.shape[1] - 1, -1, -1):
                    if result[r, c] != 0:
                        last = result[r, c]
                    elif last != 0:
                        result[r, c] = last
            return result

        def apply_col_prop(grid):
            result = grid.copy()
            for c in range(grid.shape[1]):
                col = result[:, c]
                last = 0
                for r in range(grid.shape[0]):
                    if col[r] != 0:
                        last = col[r]
                    elif last != 0:
                        col[r] = last
                last = 0
                for r in range(grid.shape[0] - 1, -1, -1):
                    if result[r, c] != 0:
                        last = result[r, c]
                    elif last != 0:
                        result[r, c] = last
            return result

        for inp, out in demo_pairs:
            if np.array_equal(apply_row_prop(inp), out):
                return apply_row_prop(test_input), 1.0, "color_prop_row"
            if np.array_equal(apply_col_prop(inp), out):
                return apply_col_prop(test_input), 1.0, "color_prop_col"
        return None, 0.0, "color_prop"

    def _solve_crop_to_content(self, demo_pairs, test_input):
        """Crop to bounding box of non-zero content."""
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] or inp.shape[1] <= out.shape[1]:
                return None, 0.0, "crop_content"

        for inp, out in demo_pairs:
            rows = np.any(inp != 0, axis=1)
            cols = np.any(inp != 0, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            cropped = inp[rmin:rmax + 1, cmin:cmax + 1]
            if not np.array_equal(cropped, out):
                return None, 0.0, "crop_content"

        rows = np.any(test_input != 0, axis=1)
        cols = np.any(test_input != 0, axis=0)
        if not rows.any():
            return None, 0.0, "crop_content"
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return test_input[rmin:rmax + 1, cmin:cmax + 1], 1.0, "crop_content"

    def _solve_delta_color(self, demo_pairs, test_input):
        """Delta-based color change: out = inp + delta (mod 10) or out = (inp * k) % 10."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "delta_color"

        # Try additive delta
        delta = None
        consistent = True
        for inp, out in demo_pairs:
            diff = (out.astype(int) - inp.astype(int)) % 10
            if delta is None:
                # Check if delta is uniform
                unique_deltas = set(diff.flatten().tolist())
                if len(unique_deltas) == 1:
                    delta = unique_deltas.pop()
                else:
                    consistent = False
                    break
            else:
                if not np.all(diff == delta):
                    consistent = False
                    break

        if consistent and delta is not None and delta != 0:
            result = (test_input.astype(int) + delta) % 10
            return result.astype(np.int8), 1.0, f"delta_{delta}"

        # Try multiplicative
        for k in range(2, 10):
            all_match = True
            for inp, out in demo_pairs:
                pred = (inp.astype(int) * k) % 10
                if not np.array_equal(pred.astype(np.int8), out):
                    all_match = False
                    break
            if all_match:
                result = (test_input.astype(int) * k) % 10
                return result.astype(np.int8), 1.0, f"mult_{k}"
        return None, 0.0, "delta_color"

    def _solve_sort_objects(self, demo_pairs, test_input):
        """Sort objects/rows by some criterion and arrange in output."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "sort_obj"

        # Try sorting rows
        for inp, out in demo_pairs:
            sorted_rows = np.sort(inp, axis=0)
            if np.array_equal(sorted_rows, out):
                return np.sort(test_input, axis=0), 1.0, "sort_rows"
            # Try reverse sort
            sorted_rows_desc = np.sort(inp, axis=0)[::-1]
            if np.array_equal(sorted_rows_desc, out):
                return np.sort(test_input, axis=0)[::-1], 1.0, "sort_rows_desc"

        # Try sorting columns
        for inp, out in demo_pairs:
            sorted_cols = np.sort(inp, axis=1)
            if np.array_equal(sorted_cols, out):
                return np.sort(test_input, axis=1), 1.0, "sort_cols"

        return None, 0.0, "sort_obj"

    def _solve_fill_between_seps(self, demo_pairs, test_input):
        """Fill regions between separator lines with a specific color."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "fill_between"

        for sep_color in range(1, 10):
            all_match = True
            fill_color = None
            for inp, out in demo_pairs:
                sep_rows = [r for r in range(inp.shape[0])
                           if len(set(inp[r, :].tolist())) == 1 and inp[r, 0] == sep_color]
                if not sep_rows:
                    all_match = False
                    break
                # Fill between separators
                pred = inp.copy()
                for i in range(len(sep_rows) - 1):
                    r1, r2 = sep_rows[i], sep_rows[i + 1]
                    region = inp[r1 + 1:r2, :]
                    out_region = out[r1 + 1:r2, :]
                    diff = out_region.astype(int) - region.astype(int)
                    changed = np.argwhere(diff != 0)
                    if len(changed) > 0:
                        fc = int(out_region[changed[0][0], changed[0][1]])
                        if fill_color is None:
                            fill_color = fc
                        elif fill_color != fc:
                            all_match = False
                            break
                        pred[r1 + 1:r2, :] = np.where(region == 0, fc, region)
                if not np.array_equal(pred, out):
                    all_match = False
                    break
            if all_match and fill_color is not None:
                # Apply to test
                sep_rows_t = [r for r in range(test_input.shape[0])
                             if len(set(test_input[r, :].tolist())) == 1 and test_input[r, 0] == sep_color]
                result = test_input.copy()
                for i in range(len(sep_rows_t) - 1):
                    r1, r2 = sep_rows_t[i], sep_rows_t[i + 1]
                    result[r1 + 1:r2, :] = np.where(test_input[r1 + 1:r2, :] == 0, fill_color,
                                                     test_input[r1 + 1:r2, :])
                return result, 1.0, f"fill_between_{sep_color}"
        return None, 0.0, "fill_between"

    def _solve_mirror_extend(self, demo_pairs, test_input):
        """Mirror/extend the grid to create output (e.g. mirror left half to right)."""
        for inp, out in demo_pairs:
            if inp.shape[0] >= out.shape[0] and inp.shape[1] >= out.shape[1]:
                return None, 0.0, "mirror_ext"

        # Try horizontal mirror extend: inp -> [inp, inp[:, ::-1]]
        for inp, out in demo_pairs:
            if out.shape[1] == inp.shape[1] * 2:
                pred = np.hstack([inp, inp[:, ::-1]])
                if np.array_equal(pred, out):
                    return np.hstack([test_input, test_input[:, ::-1]]), 1.0, "mirror_h"
            if out.shape[0] == inp.shape[0] * 2:
                pred = np.vstack([inp, inp[::-1, :]])
                if np.array_equal(pred, out):
                    return np.vstack([test_input, test_input[::-1, :]]), 1.0, "mirror_v"
            # Try with overlap (remove last row/col before mirror)
            if out.shape[1] == inp.shape[1] * 2 - 1:
                pred = np.hstack([inp, inp[:, :-1][:, ::-1]])
                if np.array_equal(pred, out):
                    return np.hstack([test_input, test_input[:, :-1][:, ::-1]]), 1.0, "mirror_h_overlap"
            if out.shape[0] == inp.shape[0] * 2 - 1:
                pred = np.vstack([inp, inp[:-1, :][::-1, :]])
                if np.array_equal(pred, out):
                    return np.vstack([test_input, test_input[:-1, :][::-1, :]]), 1.0, "mirror_v_overlap"
        return None, 0.0, "mirror_ext"

    def _solve_combine_objects_v2(self, demo_pairs, test_input):
        """Combine multiple objects into a compact output grid."""
        for inp, out in demo_pairs:
            if inp.shape[0] < out.shape[0] or inp.shape[1] < out.shape[1]:
                return None, 0.0, "combine_v2"

        # Try: find all objects, extract their bounding boxes, arrange in output
        for inp, out in demo_pairs:
            labeled, num = ndimage.label(inp != 0)
            if num < 2:
                return None, 0.0, "combine_v2"
            # Get bounding boxes
            boxes = []
            for label_id in range(1, num + 1):
                mask = labeled == label_id
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                boxes.append((rmin, rmax, cmin, cmax, inp[rmin:rmax + 1, cmin:cmax + 1]))
            # Sort by position (top to bottom, left to right)
            boxes.sort(key=lambda b: (b[0], b[2]))
            # Try stacking vertically
            if all(b[4].shape[1] == boxes[0][4].shape[1] for b in boxes):
                stacked = np.vstack([b[4] for b in boxes])
                if stacked.shape == out.shape and np.array_equal(stacked, out):
                    # Apply to test
                    labeled_t, num_t = ndimage.label(test_input != 0)
                    boxes_t = []
                    for lid in range(1, num_t + 1):
                        mask = labeled_t == lid
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        boxes_t.append((rmin, rmax, cmin, cmax, test_input[rmin:rmax + 1, cmin:cmax + 1]))
                    boxes_t.sort(key=lambda b: (b[0], b[2]))
                    if all(b[4].shape[1] == boxes_t[0][4].shape[1] for b in boxes_t):
                        return np.vstack([b[4] for b in boxes_t]), 0.9, "combine_v_stack"
            # Try stacking horizontally
            if all(b[4].shape[0] == boxes[0][4].shape[0] for b in boxes):
                stacked = np.hstack([b[4] for b in boxes])
                if stacked.shape == out.shape and np.array_equal(stacked, out):
                    labeled_t, num_t = ndimage.label(test_input != 0)
                    boxes_t = []
                    for lid in range(1, num_t + 1):
                        mask = labeled_t == lid
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        boxes_t.append((rmin, rmax, cmin, cmax, test_input[rmin:rmax + 1, cmin:cmax + 1]))
                    boxes_t.sort(key=lambda b: (b[0], b[2]))
                    if all(b[4].shape[0] == boxes_t[0][4].shape[0] for b in boxes_t):
                        return np.hstack([b[4] for b in boxes_t]), 0.9, "combine_h_stack"
        return None, 0.0, "combine_v2"

    def _solve_replace_with_shape(self, demo_pairs, test_input):
        """Replace markers with a specific shape pattern."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "replace_shape"

        # Find marker colors and their replacement patterns
        marker_shapes = {}  # {marker_color: shape_pattern (3x3 or similar)}
        for inp, out in demo_pairs:
            diff = out.astype(int) - inp.astype(int)
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if inp[r, c] != 0 and inp[r, c] != out[r, c]:
                        mc = int(inp[r, c])
                        # Extract a small region around the marker
                        rmin = max(0, r - 1)
                        rmax = min(inp.shape[0], r + 2)
                        cmin = max(0, c - 1)
                        cmax = min(inp.shape[1], c + 2)
                        in_region = inp[rmin:rmax, cmin:cmax].copy()
                        out_region = out[rmin:rmax, cmin:cmax].copy()
                        # Normalize: set marker to 1, everything else to 0
                        in_norm = (in_region == mc).astype(int)
                        out_norm = out_region.copy()
                        if mc in marker_shapes:
                            if not np.array_equal(marker_shapes[mc], out_norm):
                                return None, 0.0, "replace_shape"
                        else:
                            marker_shapes[mc] = out_norm

        if not marker_shapes:
            return None, 0.0, "replace_shape"

        # Apply: for each marker in test, replace with the corresponding shape
        result = test_input.copy()
        for mc, shape in marker_shapes.items():
            positions = np.argwhere(test_input == mc)
            for r, c in positions:
                # Place the shape centered on (r, c)
                sh, sw = shape.shape
                offset_r = sh // 2
                offset_c = sw // 2
                for dr in range(sh):
                    for dc in range(sw):
                        nr, nc = r - offset_r + dr, c - offset_c + dc
                        if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                            result[nr, nc] = shape[dr, dc]
        return result, 0.9, "replace_shape"

    def _solve_color_border_fill(self, demo_pairs, test_input):
        """Fill enclosed regions with a color, where the border color defines the fill."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "border_fill"

        # Check: some 0 pixels become non-zero, and they're enclosed by non-zero
        for inp, out in demo_pairs:
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                return None, 0.0, "border_fill"
            # All changed pixels should be 0 -> X
            for r, c in changed:
                if inp[r, c] != 0:
                    return None, 0.0, "border_fill"
            # Fill color
            fill_colors = set(int(out[r, c]) for r, c in changed)
            if len(fill_colors) != 1:
                return None, 0.0, "border_fill"

        fc = None
        for inp, out in demo_pairs:
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) > 0:
                fc = int(out[changed[0][0], changed[0][1]])
                break
        if fc is None:
            return None, 0.0, "border_fill"

        # Apply: fill enclosed 0 regions with fc
        def apply_fill(grid, fill_color):
            result = grid.copy()
            # Find enclosed 0 regions (not connected to border)
            labeled, num = ndimage.label(grid == 0)
            if num == 0:
                return result
            # Check which labels touch the border
            border_labels = set()
            for c in range(grid.shape[1]):
                if grid[0, c] == 0 and labeled[0, c] > 0:
                    border_labels.add(labeled[0, c])
                if grid[grid.shape[0] - 1, c] == 0 and labeled[grid.shape[0] - 1, c] > 0:
                    border_labels.add(labeled[grid.shape[0] - 1, c])
            for r in range(grid.shape[0]):
                if grid[r, 0] == 0 and labeled[r, 0] > 0:
                    border_labels.add(labeled[r, 0])
                if grid[r, grid.shape[1] - 1] == 0 and labeled[r, grid.shape[1] - 1] > 0:
                    border_labels.add(labeled[r, grid.shape[1] - 1])
            # Fill non-border 0 regions
            for label_id in range(1, num + 1):
                if label_id not in border_labels:
                    result[labeled == label_id] = fill_color
            return result

        for inp, out in demo_pairs:
            if not np.array_equal(apply_fill(inp, fc), out):
                return None, 0.0, "border_fill"
        return apply_fill(test_input, fc), 1.0, "border_fill"

    def _solve_max_filter(self, demo_pairs, test_input):
        """Apply max filter: each pixel becomes the max in its neighborhood."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "max_filter"

        for size in [3, 5]:
            all_match = True
            for inp, out in demo_pairs:
                pred = ndimage.maximum_filter(inp, size=size, mode='constant', cval=0)
                if not np.array_equal(pred.astype(np.int8), out):
                    all_match = False
                    break
            if all_match:
                pred = ndimage.maximum_filter(test_input, size=size, mode='constant', cval=0)
                return pred.astype(np.int8), 1.0, f"max_filter_{size}"
        return None, 0.0, "max_filter"

    def _solve_min_filter(self, demo_pairs, test_input):
        """Apply min filter: each pixel becomes the min in its neighborhood."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "min_filter"

        for size in [3, 5]:
            all_match = True
            for inp, out in demo_pairs:
                pred = ndimage.minimum_filter(inp, size=size, mode='constant', cval=0)
                if not np.array_equal(pred.astype(np.int8), out):
                    all_match = False
                    break
            if all_match:
                pred = ndimage.minimum_filter(test_input, size=size, mode='constant', cval=0)
                return pred.astype(np.int8), 1.0, f"min_filter_{size}"
        return None, 0.0, "min_filter"

    def _solve_tile_repeat(self, demo_pairs, test_input):
        """Tile/repeat the input to create a larger output."""
        for inp, out in demo_pairs:
            if inp.shape[0] >= out.shape[0] or inp.shape[1] >= out.shape[1]:
                return None, 0.0, "tile_repeat"

        for inp, out in demo_pairs:
            ratio_r = out.shape[0] / inp.shape[0]
            ratio_c = out.shape[1] / inp.shape[1]
            if ratio_r == int(ratio_r) and ratio_c == int(ratio_c):
                nr, nc = int(ratio_r), int(ratio_c)
                pred = np.tile(inp, (nr, nc))
                if np.array_equal(pred, out):
                    return np.tile(test_input, (nr, nc)), 1.0, f"tile_{nr}x{nc}"
        return None, 0.0, "tile_repeat"

    def _solve_stretch_pattern(self, demo_pairs, test_input):
        """Stretch/interpolate the input to create a larger output."""
        for inp, out in demo_pairs:
            if inp.shape[0] >= out.shape[0] or inp.shape[1] >= out.shape[1]:
                return None, 0.0, "stretch"

        for inp, out in demo_pairs:
            ratio_r = out.shape[0] / inp.shape[0]
            ratio_c = out.shape[1] / inp.shape[1]
            if ratio_r == int(ratio_r) and ratio_c == int(ratio_c):
                nr, nc = int(ratio_r), int(ratio_c)
                # Stretch by repeating each pixel nr x nc times
                pred = np.repeat(np.repeat(inp, nr, axis=0), nc, axis=1)
                if np.array_equal(pred, out):
                    return np.repeat(np.repeat(test_input, nr, axis=0), nc, axis=1), 1.0, f"stretch_{nr}x{nc}"
        return None, 0.0, "stretch"

    # ==================== V3.3 Targeted Solvers ====================

    def _solve_grid_cell_count(self, demo_pairs, test_input):
        """Output dimensions = (num_row_cells, num_col_cells) in grid, filled with cell color.
        Each pair can have a different separator color.
        """
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] or inp.shape[1] <= out.shape[1]:
                return None, 0.0, "grid_cell_count"

        # For each pair, independently find sep_color and verify
        for inp, out in demo_pairs:
            found = False
            for sep_color in range(1, 10):
                sep_rows = [r for r in range(inp.shape[0])
                           if len(set(inp[r, :].tolist())) == 1 and inp[r, 0] == sep_color]
                sep_cols = [c for c in range(inp.shape[1])
                           if len(set(inp[:, c].tolist())) == 1 and inp[0, c] == sep_color]
                if not sep_rows and not sep_cols:
                    continue
                n_rows = len(sep_rows) + 1
                n_cols = len(sep_cols) + 1
                if out.shape != (n_rows, n_cols):
                    continue
                non_sep_colors = set(inp.flatten().tolist()) - {0, sep_color}
                if len(non_sep_colors) != 1:
                    continue
                fc = non_sep_colors.pop()
                if np.all(out == fc):
                    found = True
                    break
            if not found:
                return None, 0.0, "grid_cell_count"

        # Apply to test: find sep_color in test input
        test_sep = None
        for sep_color in range(1, 10):
            sep_rows_t = [r for r in range(test_input.shape[0])
                         if len(set(test_input[r, :].tolist())) == 1 and test_input[r, 0] == sep_color]
            sep_cols_t = [c for c in range(test_input.shape[1])
                         if len(set(test_input[:, c].tolist())) == 1 and test_input[0, c] == sep_color]
            if sep_rows_t or sep_cols_t:
                test_sep = sep_color
                n_rows_t = len(sep_rows_t) + 1
                n_cols_t = len(sep_cols_t) + 1
                non_sep = set(test_input.flatten().tolist()) - {0, sep_color}
                if len(non_sep) == 1:
                    fc = non_sep.pop()
                    return np.full((n_rows_t, n_cols_t), fc, dtype=np.int8), 1.0, f"grid_count_{sep_color}"
        return None, 0.0, "grid_cell_count"

    def _solve_recolor_and_fill(self, demo_pairs, test_input):
        """Two-step: recolor one color to another, then fill enclosed 0s with a third color."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "recolor_fill"

        # Find the recolor mapping and fill color
        recolor_map = {}
        fill_color = None
        for inp, out in demo_pairs:
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            for r, c in changed:
                ic = int(inp[r, c])
                oc = int(out[r, c])
                if ic != 0 and oc != 0:
                    # Recolor: non-zero to different non-zero
                    if ic in recolor_map and recolor_map[ic] != oc:
                        return None, 0.0, "recolor_fill"
                    recolor_map[ic] = oc
                elif ic == 0 and oc != 0:
                    # Fill: 0 to non-zero
                    if fill_color is None:
                        fill_color = oc
                    elif fill_color != oc:
                        return None, 0.0, "recolor_fill"

        if not recolor_map or fill_color is None:
            return None, 0.0, "recolor_fill"

        # Apply: first recolor, then fill enclosed
        def apply_recolor_fill(grid):
            result = grid.copy()
            # Step 1: recolor
            for ic, oc in recolor_map.items():
                result[grid == ic] = oc
            # Step 2: fill enclosed 0s
            labeled, num = ndimage.label(result == 0)
            if num > 0:
                border_labels = set()
                for c in range(result.shape[1]):
                    if result[0, c] == 0 and labeled[0, c] > 0:
                        border_labels.add(labeled[0, c])
                    if result[result.shape[0] - 1, c] == 0 and labeled[result.shape[0] - 1, c] > 0:
                        border_labels.add(labeled[result.shape[0] - 1, c])
                for r in range(result.shape[0]):
                    if result[r, 0] == 0 and labeled[r, 0] > 0:
                        border_labels.add(labeled[r, 0])
                    if result[r, result.shape[1] - 1] == 0 and labeled[r, result.shape[1] - 1] > 0:
                        border_labels.add(labeled[r, result.shape[1] - 1])
                for lid in range(1, num + 1):
                    if lid not in border_labels:
                        result[labeled == lid] = fill_color
            return result

        # Verify
        for inp, out in demo_pairs:
            if not np.array_equal(apply_recolor_fill(inp), out):
                return None, 0.0, "recolor_fill"
        return apply_recolor_fill(test_input), 1.0, "recolor_fill"

    def _solve_fill_enclosed_nearest(self, demo_pairs, test_input):
        """Fill enclosed 0 regions with the nearest external non-zero color."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "fill_enclosed_nn"

        # Check pattern: only 0s change, and they're enclosed
        for inp, out in demo_pairs:
            diff = out.astype(int) - inp.astype(int)
            changed = np.argwhere(diff != 0)
            if len(changed) == 0:
                return None, 0.0, "fill_enclosed_nn"
            for r, c in changed:
                if inp[r, c] != 0:
                    return None, 0.0, "fill_enclosed_nn"

        def apply_fill(grid):
            result = grid.copy()
            labeled, num = ndimage.label(grid == 0)
            if num == 0:
                return result
            # Find border-connected 0 regions (don't fill these)
            border_labels = set()
            for c in range(grid.shape[1]):
                if grid[0, c] == 0 and labeled[0, c] > 0:
                    border_labels.add(labeled[0, c])
                if grid[grid.shape[0] - 1, c] == 0 and labeled[grid.shape[0] - 1, c] > 0:
                    border_labels.add(labeled[grid.shape[0] - 1, c])
            for r in range(grid.shape[0]):
                if grid[r, 0] == 0 and labeled[r, 0] > 0:
                    border_labels.add(labeled[r, 0])
                if grid[r, grid.shape[1] - 1] == 0 and labeled[r, grid.shape[1] - 1] > 0:
                    border_labels.add(labeled[r, grid.shape[1] - 1])
            # For each enclosed 0 region, find nearest non-zero color
            for lid in range(1, num + 1):
                if lid in border_labels:
                    continue
                mask = labeled == lid
                # Find the border pixels of this region
                dilated = ndimage.binary_dilation(mask)
                border = dilated & ~mask & (grid != 0)
                if border.any():
                    # Most common border color
                    from collections import Counter
                    border_colors = Counter(int(v) for v in grid[border])
                    fc = border_colors.most_common(1)[0][0]
                    result[mask] = fc
            return result

        for inp, out in demo_pairs:
            if not np.array_equal(apply_fill(inp), out):
                return None, 0.0, "fill_enclosed_nn"
        return apply_fill(test_input), 1.0, "fill_enclosed_nn"

    def _solve_shift_objects_v2(self, demo_pairs, test_input):
        """Shift all objects by a detected (dr, dc) offset."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "shift_obj_v2"

        # Detect shift from first demo pair
        inp0, out0 = demo_pairs[0]
        # Find non-zero positions
        in_pos = set(zip(*np.where(inp0 != 0)))
        out_pos = set(zip(*np.where(out0 != 0)))
        if not in_pos or not out_pos:
            return None, 0.0, "shift_obj_v2"

        # Try all possible shifts
        best_shift = None
        for dr in range(-inp0.shape[0], inp0.shape[0]):
            for dc in range(-inp0.shape[1], inp0.shape[1]):
                shifted = set((r + dr, c + dc) for r, c in in_pos)
                if shifted == out_pos:
                    best_shift = (dr, dc)
                    break
            if best_shift:
                break

        if best_shift is None:
            return None, 0.0, "shift_obj_v2"

        dr, dc = best_shift
        # Verify on all demos
        for inp, out in demo_pairs:
            result = np.zeros_like(inp)
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if inp[r, c] != 0:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                            result[nr, nc] = inp[r, c]
            if not np.array_equal(result, out):
                return None, 0.0, "shift_obj_v2"

        # Apply to test
        result = np.zeros_like(test_input)
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                if test_input[r, c] != 0:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]:
                        result[nr, nc] = test_input[r, c]
        return result, 1.0, f"shift_{dr}_{dc}"

    def _solve_extract_smallest_obj(self, demo_pairs, test_input):
        """Extract the smallest non-zero connected component (by color-specific labeling)."""
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] or inp.shape[1] <= out.shape[1]:
                return None, 0.0, "extract_smallest"

        for inp, out in demo_pairs:
            # Find objects by each color separately
            found_match = False
            for color in range(1, 10):
                mask = inp == color
                if not mask.any():
                    continue
                labeled, num = ndimage.label(mask)
                for lid in range(1, num + 1):
                    comp_mask = labeled == lid
                    rows = np.any(comp_mask, axis=1)
                    cols = np.any(comp_mask, axis=0)
                    if not rows.any():
                        continue
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    sub = inp[rmin:rmax + 1, cmin:cmax + 1]
                    if sub.shape == out.shape and np.array_equal(sub, out):
                        found_match = True
                        break
                if found_match:
                    break
            if not found_match:
                return None, 0.0, "extract_smallest"

        # Apply to test: find smallest color-specific object
        best_sub = None
        best_size = float('inf')
        for color in range(1, 10):
            mask = test_input == color
            if not mask.any():
                continue
            labeled, num = ndimage.label(mask)
            for lid in range(1, num + 1):
                comp_mask = labeled == lid
                size = int(comp_mask.sum())
                rows = np.any(comp_mask, axis=1)
                cols = np.any(comp_mask, axis=0)
                if not rows.any():
                    continue
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                sub = test_input[rmin:rmax + 1, cmin:cmax + 1]
                if size < best_size:
                    best_size = size
                    best_sub = sub
        if best_sub is not None:
            return best_sub, 1.0, "extract_smallest"
        return None, 0.0, "extract_smallest"

    def _solve_tile_with_checker(self, demo_pairs, test_input):
        """Tile input + fill zero rows with checkerboard pattern."""
        for inp, out in demo_pairs:
            if inp.shape[0] >= out.shape[0] or inp.shape[1] >= out.shape[1]:
                return None, 0.0, "tile_checker"

        for inp, out in demo_pairs:
            ratio_r = out.shape[0] / inp.shape[0]
            ratio_c = out.shape[1] / inp.shape[1]
            if ratio_r != int(ratio_r) or ratio_c != int(ratio_c):
                return None, 0.0, "tile_checker"
            nr, nc = int(ratio_r), int(ratio_c)
            tiled = np.tile(inp, (nr, nc))
            # Find fill color (appears in output but not in input)
            in_colors = set(inp.flatten().tolist())
            out_colors = set(out.flatten().tolist())
            new_colors = out_colors - in_colors
            if len(new_colors) != 1:
                return None, 0.0, "tile_checker"
            fc = new_colors.pop()
            # Fill 0s in tiled with checkerboard of fc
            result = tiled.copy()
            for r in range(result.shape[0]):
                for c in range(result.shape[1]):
                    if result[r, c] == 0:
                        if (r + c) % 2 == 0:
                            result[r, c] = fc
            if not np.array_equal(result, out):
                return None, 0.0, "tile_checker"

        # Apply to test
        inp0 = demo_pairs[0][0]
        ratio_r = test_input.shape[0] / inp0.shape[0] if inp0.shape[0] > 0 else 1
        ratio_c = test_input.shape[1] / inp0.shape[1] if inp0.shape[1] > 0 else 1
        # Actually we need to use the same ratio from demos
        for inp, out in demo_pairs:
            nr = out.shape[0] // inp.shape[0]
            nc = out.shape[1] // inp.shape[1]
            break
        tiled = np.tile(test_input, (nr, nc))
        # Find fill color from demos
        in_colors = set(demo_pairs[0][0].flatten().tolist())
        out_colors = set(demo_pairs[0][1].flatten().tolist())
        fc = (out_colors - in_colors).pop()
        result = tiled.copy()
        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                if result[r, c] == 0:
                    if (r + c) % 2 == 0:
                        result[r, c] = fc
        return result, 1.0, "tile_checker"

    def _solve_recolor_inplace(self, demo_pairs, test_input):
        """Simple color recoloring: each color maps to a different color (no position dependency)."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "recolor_inplace"

        # Build color map
        color_map = {}
        consistent = True
        for inp, out in demo_pairs:
            for c in range(10):
                mask = inp == c
                if mask.any():
                    out_vals = set(out[mask].tolist())
                    if len(out_vals) > 1:
                        consistent = False
                        break
                    target = out_vals.pop()
                    if c in color_map and color_map[c] != target:
                        consistent = False
                        break
                    color_map[c] = target
            if not consistent:
                break

        if not consistent:
            return None, 0.0, "recolor_inplace"

        # Only apply if at least one color changes
        changes = {c: t for c, t in color_map.items() if c != t}
        if not changes:
            return None, 0.0, "recolor_inplace"

        result = test_input.copy()
        for c, t in changes.items():
            result[test_input == c] = t
        return result, 1.0, "recolor_inplace"

    def _solve_copy_half(self, demo_pairs, test_input):
        """Copy one half of the grid to the other half (with optional color change)."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "copy_half"

        # Try: left half -> right half
        for inp, out in demo_pairs:
            mid = inp.shape[1] // 2
            left = inp[:, :mid]
            right = out[:, mid:]
            if np.array_equal(left, right):
                # Check all demos
                all_ok = True
                for inp2, out2 in demo_pairs:
                    mid2 = inp2.shape[1] // 2
                    if not np.array_equal(inp2[:, :mid2], out2[:, mid2:]):
                        all_ok = False
                        break
                if all_ok:
                    result = test_input.copy()
                    mid_t = test_input.shape[1] // 2
                    result[:, mid_t:] = test_input[:, :mid_t]
                    return result, 1.0, "copy_lr"

        # Try: right half -> left half
        for inp, out in demo_pairs:
            mid = inp.shape[1] // 2
            right = inp[:, mid:]
            left = out[:, :mid]
            if np.array_equal(right, left):
                all_ok = True
                for inp2, out2 in demo_pairs:
                    mid2 = inp2.shape[1] // 2
                    if not np.array_equal(inp2[:, mid2:], out2[:, :mid2]):
                        all_ok = False
                        break
                if all_ok:
                    result = test_input.copy()
                    mid_t = test_input.shape[1] // 2
                    result[:, :mid_t] = test_input[:, mid_t:]
                    return result, 1.0, "copy_rl"

        # Try: top half -> bottom half
        for inp, out in demo_pairs:
            mid = inp.shape[0] // 2
            top = inp[:mid, :]
            bottom = out[mid:, :]
            if np.array_equal(top, bottom):
                all_ok = True
                for inp2, out2 in demo_pairs:
                    mid2 = inp2.shape[0] // 2
                    if not np.array_equal(inp2[:mid2, :], out2[mid2:, :]):
                        all_ok = False
                        break
                if all_ok:
                    result = test_input.copy()
                    mid_t = test_input.shape[0] // 2
                    result[mid_t:, :] = test_input[:mid_t, :]
                    return result, 1.0, "copy_tb"

        # Try: bottom half -> top half
        for inp, out in demo_pairs:
            mid = inp.shape[0] // 2
            bottom = inp[mid:, :]
            top = out[:mid, :]
            if np.array_equal(bottom, top):
                all_ok = True
                for inp2, out2 in demo_pairs:
                    mid2 = inp2.shape[0] // 2
                    if not np.array_equal(inp2[mid2:, :], out2[:mid2, :]):
                        all_ok = False
                        break
                if all_ok:
                    result = test_input.copy()
                    mid_t = test_input.shape[0] // 2
                    result[:mid_t, :] = test_input[mid_t:, :]
                    return result, 1.0, "copy_bt"
        return None, 0.0, "copy_half"

    def _solve_move_to_edge(self, demo_pairs, test_input):
        """Move objects to the nearest edge of the grid."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "move_edge"

        # Detect: objects in input move to a specific edge in output
        for direction in ['up', 'down', 'left', 'right']:
            all_match = True
            for inp, out in demo_pairs:
                pred = np.zeros_like(inp)
                if direction == 'up':
                    for c in range(inp.shape[1]):
                        col = inp[:, c]
                        nonzero = col[col != 0]
                        if len(nonzero) > 0:
                            pred[:len(nonzero), c] = nonzero
                elif direction == 'down':
                    for c in range(inp.shape[1]):
                        col = inp[:, c]
                        nonzero = col[col != 0]
                        if len(nonzero) > 0:
                            pred[inp.shape[0] - len(nonzero):, c] = nonzero
                elif direction == 'left':
                    for r in range(inp.shape[0]):
                        row = inp[r, :]
                        nonzero = row[row != 0]
                        if len(nonzero) > 0:
                            pred[r, :len(nonzero)] = nonzero
                elif direction == 'right':
                    for r in range(inp.shape[0]):
                        row = inp[r, :]
                        nonzero = row[row != 0]
                        if len(nonzero) > 0:
                            pred[r, inp.shape[1] - len(nonzero):] = nonzero
                if not np.array_equal(pred, out):
                    all_match = False
                    break
            if all_match:
                # Apply to test
                pred = np.zeros_like(test_input)
                if direction == 'up':
                    for c in range(test_input.shape[1]):
                        col = test_input[:, c]
                        nonzero = col[col != 0]
                        if len(nonzero) > 0:
                            pred[:len(nonzero), c] = nonzero
                elif direction == 'down':
                    for c in range(test_input.shape[1]):
                        col = test_input[:, c]
                        nonzero = col[col != 0]
                        if len(nonzero) > 0:
                            pred[test_input.shape[0] - len(nonzero):, c] = nonzero
                elif direction == 'left':
                    for r in range(test_input.shape[0]):
                        row = test_input[r, :]
                        nonzero = row[row != 0]
                        if len(nonzero) > 0:
                            pred[r, :len(nonzero)] = nonzero
                elif direction == 'right':
                    for r in range(test_input.shape[0]):
                        row = test_input[r, :]
                        nonzero = row[row != 0]
                        if len(nonzero) > 0:
                            pred[r, test_input.shape[1] - len(nonzero):] = nonzero
                return pred, 1.0, f"move_{direction}"
        return None, 0.0, "move_edge"

    # ==================== V3.4 High-Impact Solvers ====================

    def _solve_center_pixel(self, demo_pairs, test_input):
        """Output is 1x1, value = pixel at a consistent position in input."""
        for inp, out in demo_pairs:
            o = out[0] if out.ndim == 3 else out
            if o.shape != (1, 1):
                return None, 0.0, "center_pixel"
        # Try all positions to find one that works for ALL demo pairs
        ref_inp = demo_pairs[0][0]
        candidates = []
        for r in range(ref_inp.shape[0]):
            for c in range(ref_inp.shape[1]):
                all_match = True
                for inp, out in demo_pairs:
                    o = out[0] if out.ndim == 3 else out
                    if r >= inp.shape[0] or c >= inp.shape[1]:
                        all_match = False
                        break
                    if inp[r, c] != o[0, 0]:
                        all_match = False
                        break
                if all_match:
                    candidates.append((r, c))
        if len(candidates) == 1:
            r, c = candidates[0]
            return np.array([[test_input[r, c]]], dtype=np.int8), 1.0, f"pixel_pos_{r}_{c}"
        # Try majority color
        for inp, out in demo_pairs:
            o = out[0] if out.ndim == 3 else out
            colors, counts = np.unique(inp, return_counts=True)
            majority = colors[np.argmax(counts)]
            if majority != o[0, 0]:
                break
        else:
            colors, counts = np.unique(test_input, return_counts=True)
            majority = colors[np.argmax(counts)]
            return np.array([[majority]], dtype=np.int8), 1.0, "majority_color"
        # Try minority color (least common non-zero)
        for inp, out in demo_pairs:
            o = out[0] if out.ndim == 3 else out
            colors, counts = np.unique(inp, return_counts=True)
            non_zero = [(c, n) for c, n in zip(colors, counts) if c != 0]
            if non_zero:
                non_zero.sort(key=lambda x: x[1])
                minority = non_zero[0][0]
                if minority != o[0, 0]:
                    break
        else:
            colors, counts = np.unique(test_input, return_counts=True)
            non_zero = [(c, n) for c, n in zip(colors, counts) if c != 0]
            if non_zero:
                non_zero.sort(key=lambda x: x[1])
                return np.array([[non_zero[0][0]]], dtype=np.int8), 1.0, "minority_color"
        return None, 0.0, "center_pixel"

    def _solve_split_combine(self, demo_pairs, test_input):
        """Split at separator column, combine two halves (AND/XOR/both-zero)."""
        # Find separator column (all same non-zero color)
        sep_col = None
        sep_color = None
        for inp, out in demo_pairs:
            found = False
            for c in range(inp.shape[1]):
                col = inp[:, c]
                if len(set(col.tolist())) == 1 and col[0] != 0:
                    if sep_col is None:
                        sep_col = c
                        sep_color = int(col[0])
                    elif sep_col != c or sep_color != int(col[0]):
                        return None, 0.0, "split_combine"
                    found = True
                    break
            if not found:
                return None, 0.0, "split_combine"
        if sep_col is None or sep_col == 0 or sep_col >= test_input.shape[1] - 1:
            return None, 0.0, "split_combine"
        # Left and right halves
        left_t = test_input[:, :sep_col]
        right_t = test_input[:, sep_col + 1:]
        if left_t.shape != right_t.shape:
            return None, 0.0, "split_combine"
        # Try different combine rules
        for rule_name, rule_fn in [
            ("and_recolor", lambda l, r, fc: np.where((l != 0) & (r != 0), fc, 0)),
            ("both_zero", lambda l, r, fc: np.where((l == 0) & (r == 0), fc, 0)),
            ("xor", lambda l, r, fc: np.where((l != 0) != (r != 0), fc, 0)),
            ("left_only", lambda l, r, fc: np.where((l != 0) & (r == 0), fc, 0)),
            ("right_only", lambda l, r, fc: np.where((l == 0) & (r != 0), fc, 0)),
        ]:
            all_match = True
            fill_color = None
            for inp, out in demo_pairs:
                l = inp[:, :sep_col]
                r = inp[:, sep_col + 1:]
                if l.shape != r.shape:
                    all_match = False
                    break
                o = out[0] if out.ndim == 3 else out
                found_fc = False
                for fc in range(1, 10):
                    pred = rule_fn(l, r, fc)
                    if pred.shape == o.shape and np.array_equal(pred, o):
                        if fill_color is None:
                            fill_color = fc
                        elif fill_color != fc:
                            all_match = False
                        found_fc = True
                        break
                if not found_fc:
                    all_match = False
            if all_match and fill_color is not None:
                pred = rule_fn(left_t, right_t, fill_color)
                return pred.astype(np.int8), 1.0, f"split_{rule_name}_{fill_color}"
        return None, 0.0, "split_combine"

    def _solve_repeat_recolor(self, demo_pairs, test_input):
        """Output = input repeated (partially) + recolor."""
        for inp, out in demo_pairs:
            if out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
                return None, 0.0, "repeat_recolor"
            if out.shape == inp.shape:
                return None, 0.0, "repeat_recolor"
        for inp, out in demo_pairs:
            # Check vertical repeat
            if out.shape[1] == inp.shape[1] and out.shape[0] > inp.shape[0]:
                ratio = out.shape[0] / inp.shape[0]
                if ratio > 1:
                    # Find recolor map
                    recolor = {}
                    consistent = True
                    for r in range(inp.shape[0]):
                        for c in range(inp.shape[1]):
                            ic = int(inp[r, c])
                            oc = int(out[r, c])
                            if ic != oc:
                                if ic in recolor and recolor[ic] != oc:
                                    consistent = False
                                    break
                                recolor[ic] = oc
                    if not consistent:
                        return None, 0.0, "repeat_recolor"
                    # Check: output = input repeated + recolored
                    n_full = out.shape[0] // inp.shape[0]
                    extra = out.shape[0] - n_full * inp.shape[0]
                    pred = np.vstack([inp] * n_full + [inp[:extra, :]])
                    for ic, oc in recolor.items():
                        pred[pred == ic] = oc
                    if np.array_equal(pred, out):
                        # Apply to test
                        n_full_t = test_input.shape[0] // inp.shape[0]
                        # Actually, we need to figure out the output size from the pattern
                        # Use the ratio from demo pairs
                        ratio_r = out.shape[0] / inp.shape[0]
                        out_rows = int(test_input.shape[0] * ratio_r)
                        n_full_t = out_rows // test_input.shape[0]
                        extra_t = out_rows - n_full_t * test_input.shape[0]
                        if n_full_t < 1:
                            return None, 0.0, "repeat_recolor"
                        pred_t = np.vstack([test_input] * n_full_t + [test_input[:extra_t, :]] if extra_t > 0 else [test_input] * n_full_t)
                        for ic, oc in recolor.items():
                            pred_t[pred_t == ic] = oc
                        return pred_t.astype(np.int8), 1.0, f"repeat_v_recolor"
            # Check horizontal repeat
            if out.shape[0] == inp.shape[0] and out.shape[1] > inp.shape[1]:
                ratio = out.shape[1] / inp.shape[1]
                if ratio > 1:
                    recolor = {}
                    consistent = True
                    for r in range(inp.shape[0]):
                        for c in range(inp.shape[1]):
                            ic = int(inp[r, c])
                            oc = int(out[r, c])
                            if ic != oc:
                                if ic in recolor and recolor[ic] != oc:
                                    consistent = False
                                    break
                                recolor[ic] = oc
                    if not consistent:
                        return None, 0.0, "repeat_recolor"
                    n_full = out.shape[1] // inp.shape[1]
                    extra = out.shape[1] - n_full * inp.shape[1]
                    pred = np.hstack([inp] * n_full + ([inp[:, :extra]] if extra > 0 else []))
                    for ic, oc in recolor.items():
                        pred[pred == ic] = oc
                    if np.array_equal(pred, out):
                        ratio_c = out.shape[1] / inp.shape[1]
                        out_cols = int(test_input.shape[1] * ratio_c)
                        n_full_t = out_cols // test_input.shape[1]
                        extra_t = out_cols - n_full_t * test_input.shape[1]
                        if n_full_t < 1:
                            return None, 0.0, "repeat_recolor"
                        pred_t = np.hstack([test_input] * n_full_t + ([test_input[:, :extra_t]] if extra_t > 0 else []))
                        for ic, oc in recolor.items():
                            pred_t[pred_t == ic] = oc
                        return pred_t.astype(np.int8), 1.0, f"repeat_h_recolor"
        return None, 0.0, "repeat_recolor"

    def _solve_diagonal_connect(self, demo_pairs, test_input):
        """Connect same-color markers with diagonal lines."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "diag_connect"
        for inp, out in demo_pairs:
            # Check: output adds pixels on diagonal lines between same-color pairs
            added = out.astype(int) - inp.astype(int)
            added_mask = added != 0
            if not added_mask.any():
                return None, 0.0, "diag_connect"
            # All added pixels must be same color as one of the markers
            added_colors = set(out[added_mask].tolist())
            if not added_colors:
                return None, 0.0, "diag_connect"
            # For each color with added pixels, check diagonal connection
            for color in added_colors:
                markers = np.argwhere(inp == color)
                if len(markers) < 2:
                    continue
                # Check if added pixels form diagonal lines between marker pairs
                added_for_color = np.argwhere((out == color) & (inp != color))
                if len(added_for_color) == 0:
                    continue
                ok = True
                for r, c in added_for_color:
                    # Must be on a diagonal between two markers of same color
                    found = False
                    for i in range(len(markers)):
                        for j in range(i + 1, len(markers)):
                            r1, c1 = markers[i]
                            r2, c2 = markers[j]
                            dr = r2 - r1
                            dc = c2 - c1
                            if abs(dr) == abs(dc) and dr != 0:
                                # Check if (r,c) is on this diagonal
                                t = (r - r1) / dr
                                if 0 < t < 1 and abs(c - c1 - t * dc) < 0.01:
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        ok = False
                        break
                if not ok:
                    return None, 0.0, "diag_connect"
        # Apply to test
        pred = test_input.copy()
        for color in range(1, 10):
            markers = np.argwhere(test_input == color)
            if len(markers) < 2:
                continue
            for i in range(len(markers)):
                for j in range(i + 1, len(markers)):
                    r1, c1 = markers[i]
                    r2, c2 = markers[j]
                    dr = r2 - r1
                    dc = c2 - c1
                    if abs(dr) == abs(dc) and dr != 0:
                        steps = abs(dr)
                        dr_s = 1 if dr > 0 else -1
                        dc_s = 1 if dc > 0 else -1
                        for s in range(1, steps):
                            nr, nc = int(r1) + s * dr_s, int(c1) + s * dc_s
                            if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
                                if pred[nr, nc] == 0:
                                    pred[nr, nc] = color
        if np.array_equal(pred, test_input):
            return None, 0.0, "diag_connect"
        return pred, 1.0, "diag_connect"

    def _solve_triangle_fill(self, demo_pairs, test_input):
        """Fill interior of shapes outlined by same-color markers."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "triangle_fill"
        for inp, out in demo_pairs:
            added = out.astype(int) - inp.astype(int)
            added_mask = added != 0
            if not added_mask.any():
                return None, 0.0, "triangle_fill"
            # All added pixels should be same color as existing markers
            added_colors = set(out[added_mask].tolist()) - set(inp[~added_mask].tolist()) if (~added_mask).any() else set(out[added_mask].tolist())
            # Check: added pixels fill between existing same-color pixels
            for color in set(out[added_mask].tolist()):
                markers = np.argwhere(inp == color)
                added_for_color = np.argwhere((out == color) & (inp != color))
                if len(markers) < 2 or len(added_for_color) == 0:
                    continue
                # Check if added pixels are between markers (convex hull fill)
                for r, c in added_for_color:
                    # Must be between two markers in same row or column
                    row_markers = [m for m in markers if m[0] == r]
                    col_markers = [m for m in markers if m[1] == c]
                    if not row_markers and not col_markers:
                        return None, 0.0, "triangle_fill"
        # Apply: fill between same-color markers in each row and column
        pred = test_input.copy()
        for color in range(1, 10):
            markers = np.argwhere(test_input == color)
            if len(markers) < 2:
                continue
            # Fill rows
            for r in range(test_input.shape[0]):
                row_markers = sorted([int(m[1]) for m in markers if m[0] == r])
                if len(row_markers) >= 2:
                    for c in range(row_markers[0], row_markers[-1] + 1):
                        if pred[r, c] == 0:
                            pred[r, c] = color
            # Fill columns
            for c in range(test_input.shape[1]):
                col_markers = sorted([int(m[0]) for m in markers if m[1] == c])
                if len(col_markers) >= 2:
                    for r in range(col_markers[0], col_markers[-1] + 1):
                        if pred[r, c] == 0:
                            pred[r, c] = color
        if np.array_equal(pred, test_input):
            return None, 0.0, "triangle_fill"
        return pred, 1.0, "triangle_fill"

    def _solve_left_boundary_fill(self, demo_pairs, test_input):
        """Fill 0s to the left of the leftmost non-zero pixel in each row with a specific color."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "left_fill"
        for inp, out in demo_pairs:
            added = out.astype(int) - inp.astype(int)
            added_mask = added != 0
            if not added_mask.any():
                return None, 0.0, "left_fill"
            added_colors = set(out[added_mask].tolist())
            if len(added_colors) != 1:
                return None, 0.0, "left_fill"
            fill_color = added_colors.pop()
            # Verify: all added pixels are 0->fill_color, and are to the left of leftmost non-zero in their row
            for r, c in np.argwhere(added_mask):
                if inp[r, c] != 0:
                    return None, 0.0, "left_fill"
                row_nonzero = np.where(inp[r] != 0)[0]
                if len(row_nonzero) == 0:
                    return None, 0.0, "left_fill"
                leftmost = row_nonzero[0]
                if c >= leftmost:
                    return None, 0.0, "left_fill"
                if out[r, c] != fill_color:
                    return None, 0.0, "left_fill"
        # Apply
        pred = test_input.copy()
        for r in range(test_input.shape[0]):
            row_nonzero = np.where(test_input[r] != 0)[0]
            if len(row_nonzero) > 0:
                leftmost = row_nonzero[0]
                for c in range(leftmost):
                    if pred[r, c] == 0:
                        pred[r, c] = fill_color
        if np.array_equal(pred, test_input):
            return None, 0.0, "left_fill"
        return pred, 1.0, f"left_fill_{fill_color}"

    def _solve_marker_row_copy(self, demo_pairs, test_input):
        """Copy column markers from first row to other rows with different color."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "marker_copy"
        for inp, out in demo_pairs:
            added = out.astype(int) - inp.astype(int)
            added_mask = added != 0
            if not added_mask.any():
                return None, 0.0, "marker_copy"
            added_colors = set(out[added_mask].tolist())
            if len(added_colors) != 1:
                return None, 0.0, "marker_copy"
            new_color = added_colors.pop()
            # Find source row (row 0 markers)
            row0_markers = set(np.where(inp[0] != 0)[0].tolist())
            if not row0_markers:
                return None, 0.0, "marker_copy"
            source_color = int(inp[0, list(row0_markers)[0]])
            # Check: added pixels are at row0 marker columns, in rows that have a non-zero pixel not in row0 columns
            for r, c in np.argwhere(added_mask):
                if c not in row0_markers:
                    return None, 0.0, "marker_copy"
                if r == 0:
                    return None, 0.0, "marker_copy"
                if out[r, c] != new_color:
                    return None, 0.0, "marker_copy"
        # Apply
        row0_markers = set(np.where(test_input[0] != 0)[0].tolist())
        if not row0_markers:
            return None, 0.0, "marker_copy"
        pred = test_input.copy()
        for r in range(1, test_input.shape[0]):
            # Check if this row has a marker not in row0 columns
            row_nonzero = set(np.where(test_input[r] != 0)[0].tolist())
            external_markers = row_nonzero - row0_markers
            if external_markers:
                for c in row0_markers:
                    if pred[r, c] == 0:
                        pred[r, c] = new_color
        if np.array_equal(pred, test_input):
            return None, 0.0, "marker_copy"
        return pred, 1.0, f"marker_copy_{new_color}"

    def _solve_recolor_fill_enclosed(self, demo_pairs, test_input):
        """Recolor one color to another, then fill enclosed 0s with a third color."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "recolor_fill"
        # Find recolor map (non-zero to non-zero)
        recolor = {}
        for inp, out in demo_pairs:
            for c in range(1, 10):
                mask = inp == c
                if mask.any():
                    out_vals = set(out[mask].tolist())
                    if len(out_vals) == 1:
                        target = out_vals.pop()
                        if target != c:
                            if c in recolor and recolor[c] != target:
                                return None, 0.0, "recolor_fill"
                            recolor[c] = target
        if not recolor:
            return None, 0.0, "recolor_fill"
        # Find fill color (0 -> non-zero)
        fill_color = None
        for inp, out in demo_pairs:
            zero_mask = inp == 0
            out_at_zero = out[zero_mask]
            nonzero_fills = set(out_at_zero[out_at_zero != 0].tolist())
            if len(nonzero_fills) == 1:
                fc = nonzero_fills.pop()
                if fill_color is None:
                    fill_color = fc
                elif fill_color != fc:
                    return None, 0.0, "recolor_fill"
        if fill_color is None:
            return None, 0.0, "recolor_fill"
        # Verify: recolor + fill enclosed
        for inp, out in demo_pairs:
            pred = inp.copy()
            for old_c, new_c in recolor.items():
                pred[pred == old_c] = new_c
            # Fill enclosed 0s
            from scipy import ndimage
            labeled, num = ndimage.label(pred == 0)
            if num > 0:
                h, w = pred.shape
                border_labels = set()
                for i in range(h):
                    if labeled[i, 0] > 0: border_labels.add(labeled[i, 0])
                    if labeled[i, w-1] > 0: border_labels.add(labeled[i, w-1])
                for j in range(w):
                    if labeled[0, j] > 0: border_labels.add(labeled[0, j])
                    if labeled[h-1, j] > 0: border_labels.add(labeled[h-1, j])
                for lid in range(1, num + 1):
                    if lid not in border_labels:
                        pred[labeled == lid] = fill_color
            if not np.array_equal(pred, out):
                return None, 0.0, "recolor_fill"
        # Apply to test
        pred = test_input.copy()
        for old_c, new_c in recolor.items():
            pred[pred == old_c] = new_c
        from scipy import ndimage
        labeled, num = ndimage.label(pred == 0)
        if num > 0:
            h, w = pred.shape
            border_labels = set()
            for i in range(h):
                if labeled[i, 0] > 0: border_labels.add(labeled[i, 0])
                if labeled[i, w-1] > 0: border_labels.add(labeled[i, w-1])
            for j in range(w):
                if labeled[0, j] > 0: border_labels.add(labeled[0, j])
                if labeled[h-1, j] > 0: border_labels.add(labeled[h-1, j])
            for lid in range(1, num + 1):
                if lid not in border_labels:
                    pred[labeled == lid] = fill_color
        return pred, 1.0, f"recolor_fill_{fill_color}"

    def _solve_compact_grid(self, demo_pairs, test_input):
        """Compact sparse objects into a grid based on their positions."""
        from scipy import ndimage
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] or inp.shape[1] <= out.shape[1]:
                return None, 0.0, "compact_grid"
        for inp, out in demo_pairs:
            # Find all non-zero objects
            labeled, num = ndimage.label(inp != 0)
            if num < 2:
                return None, 0.0, "compact_grid"
            # Get object centers
            centers = []
            colors = []
            for lid in range(1, num + 1):
                mask = labeled == lid
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                cr = (rmin + rmax) / 2
                cc = (cmin + cmax) / 2
                obj_colors = set(inp[mask].tolist())
                centers.append((cr, cc))
                colors.append(obj_colors)
            # Check if objects form a grid
            row_centers = sorted(set(c[0] for c in centers))
            col_centers = sorted(set(c[1] for c in centers))
            if len(row_centers) < 1 or len(col_centers) < 1:
                return None, 0.0, "compact_grid"
            if out.shape != (len(row_centers), len(col_centers)):
                return None, 0.0, "compact_grid"
            # Build grid: for each cell, find the object color
            pred_out = np.zeros((len(row_centers), len(col_centers)), dtype=np.int8)
            for lid in range(1, num + 1):
                mask = labeled == lid
                obj_color = set(inp[mask].tolist())
                if len(obj_color) == 1:
                    oc = obj_color.pop()
                    cr = centers[lid-1][0]
                    cc = centers[lid-1][1]
                    ri = row_centers.index(cr) if cr in row_centers else min(range(len(row_centers)), key=lambda i: abs(row_centers[i]-cr))
                    ci = col_centers.index(cc) if cc in col_centers else min(range(len(col_centers)), key=lambda i: abs(col_centers[i]-cc))
                    pred_out[ri, ci] = oc
            if not np.array_equal(pred_out, out):
                return None, 0.0, "compact_grid"
        # Apply to test
        labeled_t, num_t = ndimage.label(test_input != 0)
        if num_t < 2:
            return None, 0.0, "compact_grid"
        centers_t = []
        for lid in range(1, num_t + 1):
            mask = labeled_t == lid
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            cr = (rmin + rmax) / 2
            cc = (cmin + cmax) / 2
            centers_t.append((cr, cc))
        row_centers_t = sorted(set(c[0] for c in centers_t))
        col_centers_t = sorted(set(c[1] for c in centers_t))
        pred_t = np.zeros((len(row_centers_t), len(col_centers_t)), dtype=np.int8)
        for lid in range(1, num_t + 1):
            mask = labeled_t == lid
            obj_color = set(test_input[mask].tolist())
            if len(obj_color) == 1:
                oc = obj_color.pop()
                cr = centers_t[lid-1][0]
                cc = centers_t[lid-1][1]
                ri = min(range(len(row_centers_t)), key=lambda i: abs(row_centers_t[i]-cr))
                ci = min(range(len(col_centers_t)), key=lambda i: abs(col_centers_t[i]-cc))
                pred_t[ri, ci] = oc
        return pred_t, 1.0, "compact_grid"

    def _solve_symmetric_complete(self, demo_pairs, test_input):
        """Complete a symmetric pattern by filling missing pixels (tile-based)."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "symmetric_fill"
        for inp, out in demo_pairs:
            added = out.astype(int) - inp.astype(int)
            added_mask = added != 0
            if not added_mask.any():
                return None, 0.0, "symmetric_fill"
            if not np.all(inp[added_mask] == 0):
                return None, 0.0, "symmetric_fill"
        # Try tile-based completion (non-divisible heights allowed)
        h, w = test_input.shape
        # Vertical tiles
        for ph in range(2, h):
            # Check if a tile of height ph can explain the pattern
            tile = np.zeros((ph, w), dtype=test_input.dtype)
            valid = True
            for r in range(h):
                src_row = r % ph
                for c in range(w):
                    if tile[src_row, c] == 0 and test_input[r, c] != 0:
                        tile[src_row, c] = test_input[r, c]
                    elif tile[src_row, c] != 0 and test_input[r, c] != 0 and tile[src_row, c] != test_input[r, c]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                # Verify on demo pairs
                all_verify = True
                for inp, out in demo_pairs:
                    demo_tile = np.zeros((ph, w), dtype=inp.dtype)
                    demo_valid = True
                    for r in range(inp.shape[0]):
                        src_row = r % ph
                        for c in range(w):
                            if demo_tile[src_row, c] == 0 and inp[r, c] != 0:
                                demo_tile[src_row, c] = inp[r, c]
                            elif demo_tile[src_row, c] != 0 and inp[r, c] != 0 and demo_tile[src_row, c] != inp[r, c]:
                                demo_valid = False
                                break
                        if not demo_valid:
                            break
                    if demo_valid:
                        pred = np.tile(demo_tile, (inp.shape[0] // ph + 1, 1))[:inp.shape[0], :]
                        if not np.array_equal(pred, out):
                            all_verify = False
                            break
                    else:
                        all_verify = False
                        break
                if all_verify:
                    pred = np.tile(tile, (h // ph + 1, 1))[:h, :]
                    if not np.array_equal(pred, test_input):
                        return pred, 1.0, f"sym_fill_h{ph}"
        # Horizontal tiles
        for pw in range(2, w):
            tile = np.zeros((h, pw), dtype=test_input.dtype)
            valid = True
            for c in range(w):
                src_col = c % pw
                for r in range(h):
                    if tile[r, src_col] == 0 and test_input[r, c] != 0:
                        tile[r, src_col] = test_input[r, c]
                    elif tile[r, src_col] != 0 and test_input[r, c] != 0 and tile[r, src_col] != test_input[r, c]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                all_verify = True
                for inp, out in demo_pairs:
                    demo_tile = np.zeros((h, pw), dtype=inp.dtype)
                    demo_valid = True
                    for c in range(inp.shape[1]):
                        src_col = c % pw
                        for r in range(inp.shape[0]):
                            if demo_tile[r, src_col] == 0 and inp[r, c] != 0:
                                demo_tile[r, src_col] = inp[r, c]
                            elif demo_tile[r, src_col] != 0 and inp[r, c] != 0 and demo_tile[r, src_col] != inp[r, c]:
                                demo_valid = False
                                break
                        if not demo_valid:
                            break
                    if demo_valid:
                        pred = np.tile(demo_tile, (1, inp.shape[1] // pw + 1))[:, :inp.shape[1]]
                        if not np.array_equal(pred, out):
                            all_verify = False
                            break
                    else:
                        all_verify = False
                        break
                if all_verify:
                    pred = np.tile(tile, (1, w // pw + 1))[:, :w]
                    if not np.array_equal(pred, test_input):
                        return pred, 1.0, f"sym_fill_w{pw}"
        return None, 0.0, "symmetric_fill"

    def _solve_recolor_border_nearest(self, demo_pairs, test_input):
        """For each marker pixel, recolor the single nearest border pixel of the object.
        Each individual marker pixel (not each color) recolors exactly one border pixel.
        No two markers can claim the same border pixel.
        """
        from scipy import ndimage
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "recolor_border"
        for inp, out in demo_pairs:
            changed_mask = out.astype(int) != inp.astype(int)
            if not changed_mask.any():
                return None, 0.0, "recolor_border"
            changed_from = set(inp[changed_mask].tolist())
            if len(changed_from) != 1:
                return None, 0.0, "recolor_border"
            obj_color = changed_from.pop()
            if obj_color == 0:
                return None, 0.0, "recolor_border"
            marker_colors = set(inp.flatten().tolist()) - {0, obj_color}
            if not marker_colors:
                return None, 0.0, "recolor_border"
            # Get border pixels of the object
            obj_mask = inp == obj_color
            eroded = ndimage.binary_erosion(obj_mask)
            border_mask = obj_mask & ~eroded
            border_pixels = np.argwhere(border_mask)
            if len(border_pixels) == 0:
                return None, 0.0, "recolor_border"
            # Get all marker pixels (individual positions, not grouped by color)
            all_markers = []
            for mc in sorted(marker_colors):
                for r, c in np.argwhere(inp == mc):
                    all_markers.append((int(r), int(c), mc))
            if len(all_markers) == 0:
                return None, 0.0, "recolor_border"
            # Number of changed pixels should match number of markers
            n_changed = int((changed_mask & (inp == obj_color)).sum())
            if n_changed != len(all_markers):
                return None, 0.0, "recolor_border"
            # Assign each marker to its nearest unclaimed border pixel
            claimed = {}  # border_pixel -> marker_color
            used_border = set()
            # Sort markers by distance to nearest available border pixel (greedy)
            marker_dists = []
            for mr, mc, mcol in all_markers:
                best_d = float('inf')
                for i, (br, bc) in enumerate(border_pixels):
                    d = abs(mr - int(br)) + abs(mc - int(bc))
                    if d < best_d:
                        best_d = d
                marker_dists.append((best_d, mr, mc, mcol))
            marker_dists.sort()
            for _, mr, mc, mcol in marker_dists:
                best_d = float('inf')
                best_bp = None
                for i, (br, bc) in enumerate(border_pixels):
                    if i in used_border:
                        continue
                    d = abs(mr - int(br)) + abs(mc - int(bc))
                    if d < best_d:
                        best_d = d
                        best_bp = i
                if best_bp is None:
                    return None, 0.0, "recolor_border"
                used_border.add(best_bp)
                br, bc = border_pixels[best_bp]
                # Verify: this border pixel should be changed to marker color in output
                if out[int(br), int(bc)] != mcol:
                    return None, 0.0, "recolor_border"
                claimed[best_bp] = mcol
        # Apply to test
        obj_color_set = set()
        for inp, out in demo_pairs:
            changed_mask = out.astype(int) != inp.astype(int)
            if changed_mask.any():
                obj_color_set.update(inp[changed_mask].tolist())
        obj_color_set.discard(0)
        if not obj_color_set:
            return None, 0.0, "recolor_border"
        obj_color = obj_color_set.pop()
        marker_colors = set(test_input.flatten().tolist()) - {0, obj_color}
        if not marker_colors:
            return None, 0.0, "recolor_border"
        pred = test_input.copy()
        obj_mask = test_input == obj_color
        eroded = ndimage.binary_erosion(obj_mask)
        border_mask = obj_mask & ~eroded
        border_pixels = np.argwhere(border_mask)
        if len(border_pixels) == 0:
            return None, 0.0, "recolor_border"
        # Get all marker pixels
        all_markers = []
        for mc in sorted(marker_colors):
            for r, c in np.argwhere(test_input == mc):
                all_markers.append((int(r), int(c), mc))
        if len(all_markers) == 0:
            return None, 0.0, "recolor_border"
        # Assign each marker to nearest unclaimed border pixel (greedy by distance)
        used_border = set()
        marker_dists = []
        for mr, mc, mcol in all_markers:
            best_d = float('inf')
            for i, (br, bc) in enumerate(border_pixels):
                d = abs(mr - int(br)) + abs(mc - int(bc))
                if d < best_d:
                    best_d = d
            marker_dists.append((best_d, mr, mc, mcol))
        marker_dists.sort()
        for _, mr, mc, mcol in marker_dists:
            best_d = float('inf')
            best_bp = None
            for i, (br, bc) in enumerate(border_pixels):
                if i in used_border:
                    continue
                d = abs(mr - int(br)) + abs(mc - int(bc))
                if d < best_d:
                    best_d = d
                    best_bp = i
            if best_bp is not None:
                used_border.add(best_bp)
                br, bc = border_pixels[best_bp]
                pred[int(br), int(bc)] = mcol
        if np.array_equal(pred, test_input):
            return None, 0.0, "recolor_border"
        return pred, 1.0, "recolor_border"

    def _solve_scale_2x_pattern(self, demo_pairs, test_input):
        """Scale input 2x, each pixel becomes a 2x2 block."""
        for inp, out in demo_pairs:
            if out.shape[0] != inp.shape[0] * 2 or out.shape[1] != inp.shape[1] * 2:
                return None, 0.0, "scale_2x"
        # Find the 2x2 pattern for each color
        color_patterns = {}
        for inp, out in demo_pairs:
            for c in range(10):
                mask = inp == c
                if not mask.any():
                    continue
                # Get a 2x2 block from output corresponding to this color
                for r in range(inp.shape[0]):
                    for cc in range(inp.shape[1]):
                        if inp[r, cc] == c:
                            block = out[r*2:r*2+2, cc*2:cc*2+2]
                            if c in color_patterns:
                                if not np.array_equal(block, color_patterns[c]):
                                    return None, 0.0, "scale_2x"
                            else:
                                color_patterns[c] = block
        if not color_patterns:
            return None, 0.0, "scale_2x"
        # Apply
        h, w = test_input.shape
        pred = np.zeros((h * 2, w * 2), dtype=np.int8)
        for r in range(h):
            for c in range(w):
                color = int(test_input[r, c])
                if color in color_patterns:
                    pred[r*2:r*2+2, c*2:c*2+2] = color_patterns[color]
        return pred, 1.0, "scale_2x"

    def _solve_fill_zeros_rule(self, demo_pairs, test_input):
        """Universal fill-zeros solver: learn fill rule from demo pairs."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "fill_zeros"
        # Check: all changes are 0 -> non-zero
        all_fill_color = None
        for inp, out in demo_pairs:
            changed = out.astype(int) - inp.astype(int)
            changed_mask = changed != 0
            if not changed_mask.any():
                return None, 0.0, "fill_zeros"
            from_set = set(inp[changed_mask].tolist())
            if from_set != {0}:
                return None, 0.0, "fill_zeros"
            to_set = set(out[changed_mask].tolist())
            if len(to_set) != 1:
                return None, 0.0, "fill_zeros"
            fc = to_set.pop()
            if all_fill_color is None:
                all_fill_color = fc
            elif all_fill_color != fc:
                return None, 0.0, "fill_zeros"
        # Learn which 0s to fill: try position-based rules
        # Rule 1: fill 0s that are in rows/cols with specific pattern
        # Rule 2: fill 0s adjacent to specific colors
        # Rule 3: fill 0s at positions that are non-zero in other demo pairs
        # Try: fill positions where ANY demo pair has non-zero
        fill_positions = None
        for inp, out in demo_pairs:
            changed_mask = (inp == 0) & (out != 0)
            if fill_positions is None:
                fill_positions = changed_mask.copy()
            else:
                fill_positions = fill_positions | changed_mask
        # Check: are the fill positions the same across all demo pairs?
        consistent = True
        for inp, out in demo_pairs:
            changed_mask = (inp == 0) & (out != 0)
            if not np.array_equal(changed_mask, fill_positions & (inp == 0)):
                consistent = False
                break
        if consistent and fill_positions is not None:
            # Apply: fill 0s at fill_positions with fill_color
            pred = test_input.copy()
            pred[fill_positions & (test_input == 0)] = all_fill_color
            if not np.array_equal(pred, test_input):
                return pred, 1.0, f"fill_zeros_pos_{all_fill_color}"
        # Try: fill 0s based on row pattern (same column positions as non-zero in same row)
        for inp, out in demo_pairs:
            pred = inp.copy()
            for r in range(inp.shape[0]):
                nonzero_cols = set(np.where(inp[r] != 0)[0].tolist())
                if nonzero_cols:
                    # Find the period
                    sorted_cols = sorted(nonzero_cols)
                    if len(sorted_cols) >= 2:
                        period = sorted_cols[1] - sorted_cols[0]
                        for c in range(inp.shape[1]):
                            if inp[r, c] == 0 and any(abs(c - sc) % period == 0 for sc in sorted_cols):
                                pred[r, c] = all_fill_color
            if np.array_equal(pred, out):
                # Apply to test
                pred_t = test_input.copy()
                for r in range(test_input.shape[0]):
                    nonzero_cols = set(np.where(test_input[r] != 0)[0].tolist())
                    if nonzero_cols:
                        sorted_cols = sorted(nonzero_cols)
                        if len(sorted_cols) >= 2:
                            period = sorted_cols[1] - sorted_cols[0]
                            for c in range(test_input.shape[1]):
                                if test_input[r, c] == 0 and any(abs(c - sc) % period == 0 for sc in sorted_cols):
                                    pred_t[r, c] = all_fill_color
                if not np.array_equal(pred_t, test_input):
                    return pred_t, 1.0, f"fill_zeros_period_{all_fill_color}"
        return None, 0.0, "fill_zeros"

    def _solve_shift_objects_v3(self, demo_pairs, test_input):
        """Shift all non-zero objects by a fixed offset."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "shift_v3"
        # Find shift offset
        shift = None
        for inp, out in demo_pairs:
            in_pos = set(zip(*np.where(inp != 0)))
            out_pos = set(zip(*np.where(out != 0)))
            if len(in_pos) != len(out_pos):
                return None, 0.0, "shift_v3"
            # Find common shift
            for r1, c1 in in_pos:
                for r2, c2 in out_pos:
                    if inp[r1, c1] == out[r2, c2]:
                        dr, dc = r2 - r1, c2 - c1
                        if dr == 0 and dc == 0:
                            continue
                        # Check if this shift works for all
                        ok = True
                        for r, c in in_pos:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                                if out[nr, nc] != inp[r, c]:
                                    ok = False
                                    break
                            else:
                                ok = False
                                break
                        if ok:
                            if shift is None:
                                shift = (dr, dc)
                            elif shift != (dr, dc):
                                return None, 0.0, "shift_v3"
                        break
                break
        if shift is None:
            return None, 0.0, "shift_v3"
        dr, dc = shift
        # Apply
        pred = np.zeros_like(test_input)
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                if test_input[r, c] != 0:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
                        pred[nr, nc] = test_input[r, c]
        if np.array_equal(pred, test_input):
            return None, 0.0, "shift_v3"
        return pred, 1.0, f"shift_{dr}_{dc}"

    def _solve_fill_enclosed_v3(self, demo_pairs, test_input):
        """Fill enclosed 0 regions. Uses brute-force color mapping from demos.
        Handles 'swap' pattern: non-enclosed pixels of fill colors are removed.
        """
        from scipy import ndimage
        from itertools import permutations
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "fill_enclosed_v3"
        for inp, out in demo_pairs:
            changed_fill = (inp == 0) & (out != 0)
            if not changed_fill.any():
                return None, 0.0, "fill_enclosed_v3"
        # Learn fill color mapping from demos
        # For each demo, find enclosed 0 regions and their fill colors
        demo_mappings = []
        has_removals = False
        for inp, out in demo_pairs:
            changed_fill = (inp == 0) & (out != 0)
            changed_remove = (inp != 0) & (out == 0)
            if changed_remove.any():
                has_removals = True
            # Find enclosed 0 regions in INPUT (not output)
            zero_mask = inp == 0
            labeled, num = ndimage.label(zero_mask)
            if num == 0:
                demo_mappings.append({})
                continue
            h, w = inp.shape
            border_labels = set()
            for i in range(h):
                if labeled[i, 0] > 0: border_labels.add(labeled[i, 0])
                if labeled[i, w-1] > 0: border_labels.add(labeled[i, w-1])
            for j in range(w):
                if labeled[0, j] > 0: border_labels.add(labeled[0, j])
                if labeled[h-1, j] > 0: border_labels.add(labeled[h-1, j])
            # Find enclosed regions (not border-connected) that have fills
            regions = []
            for lid in range(1, num + 1):
                if lid in border_labels:
                    continue
                region_mask = labeled == lid
                # Check if this region is filled in output
                fill_vals = out[region_mask]
                if (fill_vals == 0).all():
                    continue  # Not filled
                fill_colors_in_region = set(fill_vals[fill_vals != 0].tolist())
                if len(fill_colors_in_region) != 1:
                    return None, 0.0, "fill_enclosed_v3"
                fc = fill_colors_in_region.pop()
                # Check region is enclosed (not touching grid border)
                rows = np.any(region_mask, axis=1)
                cols = np.any(region_mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                if rmin == 0 or rmax == h-1 or cmin == 0 or cmax == w-1:
                    return None, 0.0, "fill_enclosed_v3"
                # Get centroid for sorting
                ys, xs = np.where(region_mask)
                centroid = (float(ys.mean()), float(xs.mean()))
                regions.append((centroid, fc, lid))
            # Sort regions by position (row, then col)
            regions.sort()
            demo_mappings.append([fc for _, fc, _ in regions])
        # Check consistency using reverse-sorted assignment rule:
        # Sort regions by row, sort removed colors by avg row, assign in REVERSE
        # If no removals, use nearest non-zero color (distance transform)
        if not demo_mappings:
            return None, 0.0, "fill_enclosed_v3"
        # For demos with removals, verify reverse-sorted rule
        use_reverse_rule = has_removals
        if use_reverse_rule:
            for inp, out in demo_pairs:
                changed_remove = (inp != 0) & (out == 0)
                if not changed_remove.any():
                    continue
                # Find enclosed regions sorted by row
                zero_mask = inp == 0
                labeled_d, num_d = ndimage.label(zero_mask)
                if num_d == 0:
                    return None, 0.0, "fill_enclosed_v3"
                h, w = inp.shape
                border_labels_d = set()
                for i in range(h):
                    if labeled_d[i, 0] > 0: border_labels_d.add(labeled_d[i, 0])
                    if labeled_d[i, w-1] > 0: border_labels_d.add(labeled_d[i, w-1])
                for j in range(w):
                    if labeled_d[0, j] > 0: border_labels_d.add(labeled_d[0, j])
                    if labeled_d[h-1, j] > 0: border_labels_d.add(labeled_d[h-1, j])
                regions_d = []
                for lid in range(1, num_d + 1):
                    if lid in border_labels_d:
                        continue
                    region_mask = labeled_d == lid
                    fill_vals = out[region_mask]
                    if (fill_vals == 0).all():
                        continue
                    fc = set(fill_vals[fill_vals != 0].tolist())
                    if len(fc) != 1:
                        return None, 0.0, "fill_enclosed_v3"
                    ys, xs = np.where(region_mask)
                    regions_d.append((float(ys.mean()), float(xs.mean()), fc.pop()))
                regions_d.sort()
                # Find removed colors sorted by avg row
                removed_mask = changed_remove
                removed_colors_info = []
                for rc in sorted(set(inp[removed_mask].tolist())):
                    rc_pixels = np.argwhere((inp == rc) & removed_mask)
                    avg_row = float(rc_pixels[:, 0].mean())
                    removed_colors_info.append((avg_row, rc))
                removed_colors_info.sort()
                # Verify reverse assignment
                if len(regions_d) != len(removed_colors_info):
                    return None, 0.0, "fill_enclosed_v3"
                for i, (_, _, fc) in enumerate(regions_d):
                    expected_fc = removed_colors_info[len(removed_colors_info) - 1 - i][1]
                    if fc != expected_fc:
                        return None, 0.0, "fill_enclosed_v3"
        else:
            # No removals — use nearest non-zero color
            # Verify with distance transform
            for inp, out in demo_pairs:
                changed_fill = (inp == 0) & (out != 0)
                zero_mask = inp == 0
                labeled_d, num_d = ndimage.label(zero_mask)
                if num_d == 0:
                    return None, 0.0, "fill_enclosed_v3"
                h, w = inp.shape
                border_labels_d = set()
                for i in range(h):
                    if labeled_d[i, 0] > 0: border_labels_d.add(labeled_d[i, 0])
                    if labeled_d[i, w-1] > 0: border_labels_d.add(labeled_d[i, w-1])
                for j in range(w):
                    if labeled_d[0, j] > 0: border_labels_d.add(labeled_d[0, j])
                    if labeled_d[h-1, j] > 0: border_labels_d.add(labeled_d[h-1, j])
                # Check enclosed regions are filled with nearest non-zero color
                colors = sorted(set(inp.flatten()) - {0})
                if not colors:
                    return None, 0.0, "fill_enclosed_v3"
                min_dist = np.full(inp.shape, np.inf)
                nearest_col = np.zeros(inp.shape, dtype=int)
                for c in colors:
                    mask = inp == c
                    if not mask.any():
                        continue
                    dist = ndimage.distance_transform_edt(~mask)
                    closer = dist < min_dist
                    min_dist[closer] = dist[closer]
                    nearest_col[closer] = c
                for lid in range(1, num_d + 1):
                    if lid in border_labels_d:
                        continue
                    region_mask = labeled_d == lid
                    for r, c in np.argwhere(region_mask):
                        if changed_fill[r, c]:
                            if nearest_col[r, c] == 0 or out[r, c] != nearest_col[r, c]:
                                return None, 0.0, "fill_enclosed_v3"
        # Determine fill colors for test
        if use_reverse_rule:
            # Find removed colors in test, sort by avg row
            # We need to find which colors will be removed — use demo's removed colors pattern
            # For now, use colors that are NOT the enclosing color and are outside enclosed regions
            pass  # Will handle in apply section
        fill_colors_set = set()
        for dm in demo_mappings:
            fill_colors_set.update(dm)
        if not fill_colors_set:
            return None, 0.0, "fill_enclosed_v3"
        # Verify removals: removed colors should be fill colors
        if has_removals:
            for inp, out in demo_pairs:
                changed_remove = (inp != 0) & (out == 0)
                if changed_remove.any():
                    removed_colors = set(inp[changed_remove].tolist())
                    if not removed_colors.issubset(fill_colors_set):
                        return None, 0.0, "fill_enclosed_v3"
        # Apply to test
        pred = test_input.copy()
        zero_mask = test_input == 0
        labeled, num = ndimage.label(zero_mask)
        fill_colors_to_remove = set()
        if num > 0:
            h, w = test_input.shape
            border_labels = set()
            for i in range(h):
                if labeled[i, 0] > 0: border_labels.add(labeled[i, 0])
                if labeled[i, w-1] > 0: border_labels.add(labeled[i, w-1])
            for j in range(w):
                if labeled[0, j] > 0: border_labels.add(labeled[0, j])
                if labeled[h-1, j] > 0: border_labels.add(labeled[h-1, j])
            # Find enclosed regions and sort by position
            test_regions = []
            for lid in range(1, num + 1):
                if lid in border_labels:
                    continue
                region_mask = labeled == lid
                ys, xs = np.where(region_mask)
                centroid = (float(ys.mean()), float(xs.mean()))
                test_regions.append((centroid, lid, region_mask))
            test_regions.sort()
            if use_reverse_rule:
                # Find "removable" colors in test: colors that are NOT the enclosing color
                # The enclosing color is the one forming borders around enclosed 0s
                # Non-enclosing, non-zero colors that are outside enclosed regions are removable
                enclosing_colors = set()
                for centroid, lid, region_mask in test_regions:
                    # Find bordering color
                    border_pixels = []
                    for r, c in np.argwhere(region_mask):
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if not region_mask[nr, nc] and test_input[nr, nc] != 0:
                                    border_pixels.append(int(test_input[nr, nc]))
                    if border_pixels:
                        enclosing_colors.add(max(set(border_pixels), key=border_pixels.count))
                # Removable colors: all non-zero colors except enclosing colors
                all_colors = set(test_input.flatten().tolist()) - {0}
                removable_colors = all_colors - enclosing_colors
                # Sort removable colors by average row
                removable_info = []
                for rc in sorted(removable_colors):
                    rc_pixels = np.argwhere(test_input == rc)
                    if len(rc_pixels) == 0:
                        continue
                    # Only consider pixels outside enclosed regions
                    outside_pixels = [(r, c) for r, c in rc_pixels if labeled[r, c] in border_labels or labeled[r, c] == 0]
                    if not outside_pixels:
                        continue
                    avg_row = float(np.mean([r for r, _ in outside_pixels]))
                    removable_info.append((avg_row, rc))
                removable_info.sort()
                # Assign in reverse: region[0] gets removable[-1], etc.
                for i, (centroid, lid, region_mask) in enumerate(test_regions):
                    if len(removable_info) > 0:
                        idx = len(removable_info) - 1 - i
                        if idx < 0:
                            idx = len(removable_info) - 1
                        fc = removable_info[idx][1]
                        pred[region_mask] = fc
                        fill_colors_to_remove.add(fc)
            else:
                # Use nearest non-zero color (distance transform)
                colors = sorted(set(test_input.flatten()) - {0})
                if colors:
                    min_dist = np.full(test_input.shape, np.inf)
                    nearest_col = np.zeros(test_input.shape, dtype=int)
                    for c in colors:
                        mask = test_input == c
                        if not mask.any():
                            continue
                        dist = ndimage.distance_transform_edt(~mask)
                        closer = dist < min_dist
                        min_dist[closer] = dist[closer]
                        nearest_col[closer] = c
                    for centroid, lid, region_mask in test_regions:
                        for r, c in np.argwhere(region_mask):
                            nc = nearest_col[r, c]
                            if nc != 0:
                                pred[r, c] = nc
                                fill_colors_to_remove.add(nc)
        # Remove non-enclosed pixels of fill colors (swap pattern)
        if fill_colors_to_remove and has_removals:
            for r in range(test_input.shape[0]):
                for c in range(test_input.shape[1]):
                    if test_input[r, c] in fill_colors_to_remove:
                        # Check if this pixel is inside an enclosed region (keep it)
                        # or outside (remove it)
                        adjacent_enclosed = False
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
                                if zero_mask[nr, nc] and labeled[nr, nc] not in border_labels and labeled[nr, nc] > 0:
                                    adjacent_enclosed = True
                                    break
                        if not adjacent_enclosed:
                            # Check if it's part of the enclosing border (not a fill color pixel)
                            # Only remove if it's NOT adjacent to enclosed 0
                            # and IS a fill color
                            pred[r, c] = 0
        if np.array_equal(pred, test_input):
            return None, 0.0, "fill_enclosed_v3"
        return pred, 1.0, "fill_enclosed_v3"

    def _solve_copy_pattern_down(self, demo_pairs, test_input):
        """Copy a pattern from one part of the grid to another (downward)."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "copy_down"
        for inp, out in demo_pairs:
            changed = out.astype(int) - inp.astype(int)
            changed_mask = changed != 0
            if not changed_mask.any():
                return None, 0.0, "copy_down"
            # All changes must be 0 -> non-zero
            if not np.all(inp[changed_mask] == 0):
                return None, 0.0, "copy_down"
            # Find the source: a row/region that matches the changed region
            added_rows = set(r for r, c in np.argwhere(changed_mask))
            if not added_rows:
                return None, 0.0, "copy_down"
            # Check: the added pattern matches a source row/region
            min_added_row = min(added_rows)
            max_added_row = max(added_rows)
            pattern_height = max_added_row - min_added_row + 1
            # Try copying from above
            source_start = None
            for sr in range(0, min_added_row):
                if sr + pattern_height <= min_added_row:
                    match = True
                    for r in range(pattern_height):
                        for c in range(inp.shape[1]):
                            if changed_mask[min_added_row + r, c]:
                                if inp[sr + r, c] != out[min_added_row + r, c]:
                                    match = False
                                    break
                    if match:
                        source_start = sr
                        break
            if source_start is None:
                return None, 0.0, "copy_down"
        # Apply: find source and copy down
        changed_demo = None
        for inp, out in demo_pairs:
            changed = (inp == 0) & (out != 0)
            if changed.any():
                changed_demo = changed
                break
        if changed_demo is None:
            return None, 0.0, "copy_down"
        added_rows = sorted(set(r for r, c in np.argwhere(changed_demo)))
        pattern_height = added_rows[-1] - added_rows[0] + 1
        source_start = None
        for sr in range(0, added_rows[0]):
            match = True
            for r in range(pattern_height):
                for c in range(demo_pairs[0][0].shape[1]):
                    if changed_demo[added_rows[0] + r, c]:
                        if demo_pairs[0][0][sr + r, c] != demo_pairs[0][1][added_rows[0] + r, c]:
                            match = False
                            break
            if match:
                source_start = sr
                break
        if source_start is None:
            return None, 0.0, "copy_down"
        # Apply to test: find where to fill
        # Use the same offset
        offset = added_rows[0] - source_start
        pred = test_input.copy()
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                sr = r - offset
                if 0 <= sr < test_input.shape[0] and pred[r, c] == 0 and test_input[sr, c] != 0:
                    pred[r, c] = test_input[sr, c]
        if np.array_equal(pred, test_input):
            return None, 0.0, "copy_down"
        return pred, 1.0, "copy_down"

    # ===== V3.5 solvers — targeting specific failing patterns =====

    def _solve_voronoi_fill(self, demo_pairs, test_input):
        """Fill ALL 0 pixels with nearest non-zero neighbor color (Voronoi fill).
        Unlike fill_enclosed_v3, this fills ALL zeros including border-touching regions.
        Rule: every 0→non-zero change must match nearest non-zero neighbor in input.
        No non-zero→0 changes allowed (pure additive fill).
        """
        from scipy import ndimage
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "voronoi_fill"
            # Must have changes
            changed = out.astype(int) != inp.astype(int)
            if not changed.any():
                return None, 0.0, "voronoi_fill"
            # No removals (non-zero→0)
            removed = (inp != 0) & (out == 0)
            if removed.any():
                return None, 0.0, "voronoi_fill"
            # All additions (0→non-zero) must match nearest non-zero
            added = (inp == 0) & (out != 0)
            if not added.any():
                return None, 0.0, "voronoi_fill"
            # Vectorized nearest-color check using distance transforms
            colors = sorted(set(inp.flatten()) - {0})
            if not colors:
                return None, 0.0, "voronoi_fill"
            min_dist = np.full(inp.shape, np.inf)
            nearest_col = np.zeros(inp.shape, dtype=int)
            for c in colors:
                mask = inp == c
                if not mask.any():
                    continue
                dist = ndimage.distance_transform_edt(~mask)
                closer = dist < min_dist
                min_dist[closer] = dist[closer]
                nearest_col[closer] = c
            # Check: every added pixel matches nearest non-zero color
            for r, c in np.argwhere(added):
                if nearest_col[r, c] == 0 or out[r, c] != nearest_col[r, c]:
                    return None, 0.0, "voronoi_fill"
        # Apply to test
        pred = test_input.copy()
        colors = sorted(set(test_input.flatten()) - {0})
        if not colors:
            return None, 0.0, "voronoi_fill"
        min_dist = np.full(test_input.shape, np.inf)
        nearest_col = np.zeros(test_input.shape, dtype=int)
        for c in colors:
            mask = test_input == c
            if not mask.any():
                continue
            dist = ndimage.distance_transform_edt(~mask)
            closer = dist < min_dist
            min_dist[closer] = dist[closer]
            nearest_col[closer] = c
        zero_mask = test_input == 0
        pred[zero_mask] = nearest_col[zero_mask]
        # Check: must have actually filled something
        if np.array_equal(pred, test_input):
            return None, 0.0, "voronoi_fill"
        # Check: 0s that remain 0 (no nearest neighbor) should stay 0
        # Already handled since nearest_col=0 for unreachable pixels
        return pred, 1.0, "voronoi_fill"

    def _solve_draw_lines_markers(self, demo_pairs, test_input):
        """Extend single-pixel markers to full rows/columns/crosses.
        Supports: h-only, v-only, and cross (both h+v) with intersection color.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "draw_lines"
        # Determine marker colors and their line directions
        marker_info = {}  # color -> set of directions ('h', 'v', or both)
        intersection_color = None
        for inp, out in demo_pairs:
            changed = out.astype(int) != inp.astype(int)
            if not changed.any():
                continue
            added = (inp == 0) & (out != 0)
            for r, c in np.argwhere(added):
                color = int(out[r, c])
                if color not in marker_info:
                    positions = np.argwhere(inp == color)
                    if len(positions) == 0:
                        # This might be an intersection color
                        if intersection_color is None:
                            intersection_color = color
                        continue
                    if len(positions) == 1:
                        mr, mc = int(positions[0][0]), int(positions[0][1])
                        row_count = np.sum(out[mr, :] == color)
                        col_count = np.sum(out[:, mc] == color)
                        new_in_row = np.sum(added[mr, :])
                        new_in_col = np.sum(added[:, mc])
                        dirs = set()
                        if row_count > 1 or new_in_row > 0:
                            dirs.add('h')
                        if col_count > 1 or new_in_col > 0:
                            dirs.add('v')
                        if dirs:
                            marker_info[color] = dirs
        if not marker_info:
            return None, 0.0, "draw_lines"
        # Verify on all demo pairs
        for inp, out in demo_pairs:
            pred = inp.copy()
            # First pass: draw all lines
            for color, dirs in marker_info.items():
                positions = np.argwhere(inp == color)
                for mr, mc in positions:
                    mr, mc = int(mr), int(mc)
                    if 'h' in dirs:
                        pred[mr, :] = color
                    if 'v' in dirs:
                        pred[:, mc] = color
            # Second pass: handle intersections
            if intersection_color is not None:
                # Find intersection points (where two different marker lines cross)
                marker_rows = {}  # row -> color
                marker_cols = {}  # col -> color
                for color, dirs in marker_info.items():
                    positions = np.argwhere(inp == color)
                    for mr, mc in positions:
                        mr, mc = int(mr), int(mc)
                        if 'h' in dirs:
                            marker_rows[mr] = color
                        if 'v' in dirs:
                            marker_cols[mc] = color
                for r, rc in marker_rows.items():
                    for c, cc in marker_cols.items():
                        if rc != cc:
                            pred[r, c] = intersection_color
            if not np.array_equal(pred, out):
                return None, 0.0, "draw_lines"
        # Apply to test
        pred = test_input.copy()
        for color, dirs in marker_info.items():
            positions = np.argwhere(test_input == color)
            for mr, mc in positions:
                mr, mc = int(mr), int(mc)
                if 'h' in dirs:
                    pred[mr, :] = color
                if 'v' in dirs:
                    pred[:, mc] = color
        if intersection_color is not None:
            marker_rows = {}
            marker_cols = {}
            for color, dirs in marker_info.items():
                positions = np.argwhere(test_input == color)
                for mr, mc in positions:
                    mr, mc = int(mr), int(mc)
                    if 'h' in dirs:
                        marker_rows[mr] = color
                    if 'v' in dirs:
                        marker_cols[mc] = color
            for r, rc in marker_rows.items():
                for c, cc in marker_cols.items():
                    if rc != cc:
                        pred[r, c] = intersection_color
        if np.array_equal(pred, test_input):
            return None, 0.0, "draw_lines"
        return pred, 1.0, "draw_lines"

    def _solve_fill_row_between_markers(self, demo_pairs, test_input):
        """Fill 0s between same-color markers on the same row OR column.
        Supports multiple fill colors — each row/col fills with its own marker color.
        Marker color and fill color can be different (marker_color defines which
        pixels to fill between, fill_color defines what to fill with).
        Also supports: fill_color == marker_color (simplest case).
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "fill_row_markers"
        any_changed = False
        for inp, out in demo_pairs:
            if (out.astype(int) != inp.astype(int)).any():
                any_changed = True
                break
        if not any_changed:
            return None, 0.0, "fill_row_markers"
        # Detect fill direction and fill colors from demos
        # Strategy: for each row with added pixels, the fill color is the color
        # of the added pixels, and markers are non-zero pixels at the edges
        has_horizontal = False
        has_vertical = False
        fill_colors_set = set()
        for inp, out in demo_pairs:
            added = (inp == 0) & (out != 0)
            if not added.any():
                continue
            fill_colors_set.update(out[added].tolist())
            # Check horizontal: rows with markers flanking added region
            for r in range(inp.shape[0]):
                if not added[r].any():
                    continue
                added_cols = np.where(added[r])[0]
                left_col = int(added_cols[0]) - 1
                right_col = int(added_cols[-1]) + 1
                has_left = left_col >= 0 and inp[r, left_col] != 0
                has_right = right_col < inp.shape[1] and inp[r, right_col] != 0
                if has_left or has_right:
                    has_horizontal = True
            # Check vertical: columns with markers flanking added region
            for c in range(inp.shape[1]):
                if not added[:, c].any():
                    continue
                added_rows = np.where(added[:, c])[0]
                # All added pixels in this column must be same color to be vertical
                col_fill = out[added_rows[0], c]
                if not np.all(out[added_rows, c] == col_fill):
                    continue  # Different colors, not vertical fill
                top_row = int(added_rows[0]) - 1
                bot_row = int(added_rows[-1]) + 1
                has_top = top_row >= 0 and inp[top_row, c] != 0
                has_bot = bot_row < inp.shape[0] and inp[bot_row, c] != 0
                if has_top or has_bot:
                    has_vertical = True
        if not has_horizontal and not has_vertical:
            return None, 0.0, "fill_row_markers"
        # Verify: for each demo, the pattern is consistent
        # Rule: every added pixel must be flanked by non-zero markers on its row (horizontal)
        # or column (vertical), and all added pixels between the same pair of markers
        # must have the same color
        for inp, out in demo_pairs:
            added = (inp == 0) & (out != 0)
            if not added.any():
                continue
            if has_horizontal:
                for r in range(inp.shape[0]):
                    if not added[r].any():
                        continue
                    added_cols = np.where(added[r])[0]
                    # All added pixels on this row must be same color
                    row_fill = out[r, added_cols[0]]
                    if not np.all(out[r, added_cols] == row_fill):
                        return None, 0.0, "fill_row_markers"
                    # Check markers flank the added region
                    left_col = int(added_cols[0]) - 1
                    right_col = int(added_cols[-1]) + 1
                    # At least one side should have a non-zero marker
                    left_ok = left_col >= 0 and inp[r, left_col] != 0
                    right_ok = right_col < inp.shape[1] and inp[r, right_col] != 0
                    if not (left_ok or right_ok):
                        # Maybe it's a vertical fill on this row
                        if not has_vertical:
                            return None, 0.0, "fill_row_markers"
            if has_vertical:
                for c in range(inp.shape[1]):
                    if not added[:, c].any():
                        continue
                    added_rows = np.where(added[:, c])[0]
                    col_fill = out[added_rows[0], c]
                    if not np.all(out[added_rows, c] == col_fill):
                        return None, 0.0, "fill_row_markers"
                    top_row = int(added_rows[0]) - 1
                    bot_row = int(added_rows[-1]) + 1
                    top_ok = top_row >= 0 and inp[top_row, c] != 0
                    bot_ok = bot_row < inp.shape[0] and inp[bot_row, c] != 0
                    if not (top_ok or bot_ok):
                        if not has_horizontal:
                            return None, 0.0, "fill_row_markers"
        # Detect marker colors from demos (non-zero pixels flanking added regions)
        marker_colors_set = set()
        for inp, out in demo_pairs:
            added = (inp == 0) & (out != 0)
            if not added.any():
                continue
            for r in range(inp.shape[0]):
                if not added[r].any():
                    continue
                added_cols = np.where(added[r])[0]
                left_col = int(added_cols[0]) - 1
                right_col = int(added_cols[-1]) + 1
                if left_col >= 0 and inp[r, left_col] != 0:
                    marker_colors_set.add(int(inp[r, left_col]))
                if right_col < inp.shape[1] and inp[r, right_col] != 0:
                    marker_colors_set.add(int(inp[r, right_col]))
            for c in range(inp.shape[1]):
                if not added[:, c].any():
                    continue
                added_rows = np.where(added[:, c])[0]
                top_row = int(added_rows[0]) - 1
                bot_row = int(added_rows[-1]) + 1
                if top_row >= 0 and inp[top_row, c] != 0:
                    marker_colors_set.add(int(inp[top_row, c]))
                if bot_row < inp.shape[0] and inp[bot_row, c] != 0:
                    marker_colors_set.add(int(inp[bot_row, c]))
        # Determine universal fill color (when fill color != marker color)
        universal_fill = None
        fill_only = fill_colors_set - marker_colors_set
        if len(fill_only) == 1:
            universal_fill = fill_only.pop()
        # Apply to test
        pred = test_input.copy()
        if has_horizontal:
            for r in range(test_input.shape[0]):
                row = test_input[r]
                nz_positions = np.argwhere(row != 0).flatten()
                if len(nz_positions) < 2:
                    continue
                first_pos = int(nz_positions[0])
                last_pos = int(nz_positions[-1])
                first_color = int(row[first_pos])
                last_color = int(row[last_pos])
                if first_color != last_color:
                    continue
                fill_c = universal_fill if universal_fill is not None else first_color
                # Only fill 0 pixels, don't overwrite existing non-zero values
                for c in range(first_pos, last_pos + 1):
                    if pred[r, c] == 0:
                        pred[r, c] = fill_c
        if has_vertical:
            for c in range(test_input.shape[1]):
                col = test_input[:, c]
                nz_positions = np.argwhere(col != 0).flatten()
                if len(nz_positions) < 2:
                    continue
                first_pos = int(nz_positions[0])
                last_pos = int(nz_positions[-1])
                first_color = int(col[first_pos])
                last_color = int(col[last_pos])
                if first_color != last_color:
                    continue
                fill_c = universal_fill if universal_fill is not None else first_color
                # Only fill 0 pixels, don't overwrite existing non-zero values
                for r in range(first_pos, last_pos + 1):
                    if pred[r, c] == 0:
                        pred[r, c] = fill_c
        if np.array_equal(pred, test_input):
            return None, 0.0, "fill_row_markers"
        return pred, 1.0, "fill_row_markers"

    def _solve_object_shift(self, demo_pairs, test_input):
        """Shift all non-zero objects by a consistent (dr, dc) offset.
        Objects that would shift out of bounds are clipped.
        Original positions become 0.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "obj_shift"
        # Find the shift offset from first demo pair
        offsets = []
        for inp, out in demo_pairs:
            inp_nz = np.argwhere(inp != 0)
            out_nz = np.argwhere(out != 0)
            if len(inp_nz) == 0 or len(out_nz) == 0:
                return None, 0.0, "obj_shift"
            if len(inp_nz) != len(out_nz):
                return None, 0.0, "obj_shift"
            # Try to find consistent offset
            best_offset = None
            best_matches = 0
            for dr in range(-inp.shape[0], inp.shape[0]):
                for dc in range(-inp.shape[1], inp.shape[1]):
                    shifted = inp_nz + [dr, dc]
                    valid = (shifted[:, 0] >= 0) & (shifted[:, 0] < inp.shape[0]) & \
                            (shifted[:, 1] >= 0) & (shifted[:, 1] < inp.shape[1])
                    if not valid.all():
                        continue
                    # Check colors match
                    matches = 0
                    for i, (r, c) in enumerate(inp_nz):
                        sr, sc = r + dr, c + dc
                        if 0 <= sr < inp.shape[0] and 0 <= sc < inp.shape[1]:
                            if out[sr, sc] == inp[r, c]:
                                matches += 1
                    if matches > best_matches:
                        best_matches = matches
                        best_offset = (dr, dc)
            if best_offset is None or best_matches < len(inp_nz):
                return None, 0.0, "obj_shift"
            offsets.append(best_offset)
        # All offsets must be the same
        if len(set(offsets)) != 1:
            return None, 0.0, "obj_shift"
        dr, dc = offsets[0]
        if dr == 0 and dc == 0:
            return None, 0.0, "obj_shift"
        # Apply
        pred = np.zeros_like(test_input)
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                if test_input[r, c] != 0:
                    sr, sc = r + dr, c + dc
                    if 0 <= sr < test_input.shape[0] and 0 <= sc < test_input.shape[1]:
                        pred[sr, sc] = test_input[r, c]
        if np.array_equal(pred, test_input):
            return None, 0.0, "obj_shift"
        return pred, 1.0, "obj_shift"

    def _solve_recolor_boundary_interior(self, demo_pairs, test_input):
        """Recolor object pixels: boundary (adjacent to 0) gets one color,
        interior (not adjacent to 0) gets another color.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "recolor_bi"
        # Find the object color and boundary/interior colors
        for inp, out in demo_pairs:
            changed = out.astype(int) != inp.astype(int)
            if not changed.any():
                return None, 0.0, "recolor_bi"
            changed_from = set(inp[changed].tolist())
            if len(changed_from) != 1:
                return None, 0.0, "recolor_bi"
            obj_color = changed_from.pop()
            if obj_color == 0:
                return None, 0.0, "recolor_bi"
            # Determine boundary and interior colors
            boundary_mask = np.zeros_like(inp, dtype=bool)
            interior_mask = np.zeros_like(inp, dtype=bool)
            obj_mask = inp == obj_color
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if obj_mask[r, c]:
                        # Check 4-neighbors for 0
                        has_zero_neighbor = False
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if nr < 0 or nr >= inp.shape[0] or nc < 0 or nc >= inp.shape[1]:
                                has_zero_neighbor = True
                                break
                            if inp[nr, nc] == 0:
                                has_zero_neighbor = True
                                break
                        if has_zero_neighbor:
                            boundary_mask[r, c] = True
                        else:
                            interior_mask[r, c] = True
            # Check: boundary pixels get one color, interior gets another
            boundary_colors = set(out[boundary_mask].tolist()) if boundary_mask.any() else set()
            interior_colors = set(out[interior_mask].tolist()) if interior_mask.any() else set()
            if len(boundary_colors) != 1 or len(interior_colors) != 1:
                return None, 0.0, "recolor_bi"
            b_color = boundary_colors.pop()
            i_color = interior_colors.pop()
            if b_color == i_color:
                return None, 0.0, "recolor_bi"
            # Verify: unchanged pixels stay same
            unchanged = ~changed
            if not np.all(out[unchanged] == inp[unchanged]):
                return None, 0.0, "recolor_bi"
            # Store for this pair
            break
        # Verify on all pairs
        for inp, out in demo_pairs:
            changed = out.astype(int) != inp.astype(int)
            changed_from = set(inp[changed].tolist())
            if len(changed_from) != 1:
                return None, 0.0, "recolor_bi"
            oc = changed_from.pop()
            obj_mask = inp == oc
            pred = inp.copy()
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    if obj_mask[r, c]:
                        has_zero = False
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if nr < 0 or nr >= inp.shape[0] or nc < 0 or nc >= inp.shape[1]:
                                has_zero = True
                                break
                            if inp[nr, nc] == 0:
                                has_zero = True
                                break
                        pred[r, c] = b_color if has_zero else i_color
            if not np.array_equal(pred, out):
                return None, 0.0, "recolor_bi"
        # Apply to test
        # Find obj_color from test
        test_colors = set(test_input.flatten()) - {0}
        # Use the color that was the obj_color in demos
        changed_colors = set()
        for inp, out in demo_pairs:
            changed = out.astype(int) != inp.astype(int)
            changed_colors.update(inp[changed].tolist())
        changed_colors.discard(0)
        if len(changed_colors) != 1:
            return None, 0.0, "recolor_bi"
        obj_color = changed_colors.pop()
        obj_mask = test_input == obj_color
        pred = test_input.copy()
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                if obj_mask[r, c]:
                    has_zero = False
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if nr < 0 or nr >= test_input.shape[0] or nc < 0 or nc >= test_input.shape[1]:
                            has_zero = True
                            break
                        if test_input[nr, nc] == 0:
                            has_zero = True
                            break
                    pred[r, c] = b_color if has_zero else i_color
        if np.array_equal(pred, test_input):
            return None, 0.0, "recolor_bi"
        return pred, 1.0, "recolor_bi"

    def _solve_diagonal_extend(self, demo_pairs, test_input):
        """Extend a diagonal pattern from a seed.
        The seed is a small colored block. The output extends it diagonally
        (down-right or down-left), removing a 'marker' color (color 2).
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "diag_extend"
        for inp, out in demo_pairs:
            changed = out.astype(int) != inp.astype(int)
            if not changed.any():
                return None, 0.0, "diag_extend"
            added = (inp == 0) & (out != 0)
            removed = (inp != 0) & (out == 0)
            # Must have both additions and removals
            if not added.any() or not removed.any():
                return None, 0.0, "diag_extend"
            # The removed color is the marker (usually color 2)
            removed_colors = set(inp[removed].tolist())
            if len(removed_colors) != 1:
                return None, 0.0, "diag_extend"
            marker_color = removed_colors.pop()
            # The added color is the extension color
            added_colors = set(out[added].tolist())
            if len(added_colors) != 1:
                return None, 0.0, "diag_extend"
            ext_color = added_colors.pop()
            # Find the seed position (where ext_color exists in input)
            seed_positions = np.argwhere(inp == ext_color)
            if len(seed_positions) == 0:
                return None, 0.0, "diag_extend"
            # Determine diagonal direction
            seed_r, seed_c = seed_positions[0]
            # Check if extension goes down-right or down-left
            added_positions = np.argwhere(added)
            if len(added_positions) == 0:
                return None, 0.0, "diag_extend"
            # Check direction
            first_added = added_positions[0]
            dr = int(first_added[0]) - int(seed_r)
            dc = int(first_added[1]) - int(seed_c)
            if dr <= 0:
                return None, 0.0, "diag_extend"
            direction = 1 if dc > 0 else (-1 if dc < 0 else 0)
            if direction == 0:
                return None, 0.0, "diag_extend"
            # Verify: extension follows diagonal pattern
            # Each step adds a row of ext_color at the diagonal position
            for r, c in added_positions:
                step = r - seed_r
                expected_c = seed_c + direction * step
                # The extension forms a band of width = seed_width
                if abs(c - expected_c) > 2:
                    return None, 0.0, "diag_extend"
            break
        # Apply to test
        # Remove marker color, extend diagonal
        seed_positions = np.argwhere(test_input == ext_color)
        if len(seed_positions) == 0:
            return None, 0.0, "diag_extend"
        seed_r = int(seed_positions[0][0])
        seed_c = int(seed_positions[0][1])
        # Get the seed block dimensions
        seed_rows = seed_positions[:, 0]
        seed_cols = seed_positions[:, 1]
        seed_h = int(seed_rows.max() - seed_rows.min() + 1)
        seed_w = int(seed_cols.max() - seed_cols.min() + 1)
        seed_rmin = int(seed_rows.min())
        seed_cmin = int(seed_cols.min())
        pred = test_input.copy()
        # Remove marker color
        pred[test_input == marker_color] = 0
        # Extend diagonally
        h, w = test_input.shape
        for step in range(1, max(h, w)):
            for dr in range(seed_h):
                for dc in range(seed_w):
                    r = seed_rmin + step + dr
                    c = seed_cmin + direction * step + dc
                    if 0 <= r < h and 0 <= c < w and pred[r, c] == 0:
                        pred[r, c] = ext_color
        if np.array_equal(pred, test_input):
            return None, 0.0, "diag_extend"
        return pred, 1.0, "diag_extend"

    def _solve_connect_grid_markers(self, demo_pairs, test_input):
        """Two markers define a dividing line. The grid is split into regions,
        each region's border is filled with the corresponding marker's color.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "grid_markers"
        for inp, out in demo_pairs:
            changed = out.astype(int) != inp.astype(int)
            if not changed.any():
                return None, 0.0, "grid_markers"
            added = (inp == 0) & (out != 0)
            # Find marker colors (single pixels in input that become borders)
            colors_in = set(inp.flatten()) - {0}
            if len(colors_in) != 2:
                return None, 0.0, "grid_markers"
            c1, c2 = sorted(colors_in)
            # Find marker positions
            pos1 = np.argwhere(inp == c1)
            pos2 = np.argwhere(inp == c2)
            if len(pos1) != 1 or len(pos2) != 1:
                return None, 0.0, "grid_markers"
            r1, c1_pos = int(pos1[0][0]), int(pos1[0][1])
            r2, c2_pos = int(pos2[0][0]), int(pos2[0][1])
            # Dividing line is between r1 and r2
            midpoint = (r1 + r2) / 2
            # Top region uses c1, bottom uses c2
            # Verify: added pixels in top region have color c1, bottom have c2
            for r, c in np.argwhere(added):
                if r < midpoint and out[r, c] != c1:
                    return None, 0.0, "grid_markers"
                if r >= midpoint and out[r, c] != c2:
                    return None, 0.0, "grid_markers"
            # Verify pattern: borders of each region are filled
            # Top region: rows 0 to floor(midpoint)
            # Border: first row, last row of region, left/right edges
            top_end = int(np.floor(midpoint))
            bottom_start = int(np.ceil(midpoint))
            # Check top region borders
            for r in range(0, top_end + 1):
                for c in range(inp.shape[1]):
                    if r == 0 or r == top_end or c == 0 or c == inp.shape[1] - 1:
                        if out[r, c] != c1 and out[r, c] != 0 and inp[r, c] == 0:
                            return None, 0.0, "grid_markers"
            break
        # Apply to test
        pos1 = np.argwhere(test_input == c1)
        pos2 = np.argwhere(test_input == c2)
        if len(pos1) != 1 or len(pos2) != 1:
            return None, 0.0, "grid_markers"
        r1 = int(pos1[0][0])
        r2 = int(pos2[0][0])
        midpoint = (r1 + r2) / 2
        top_end = int(np.floor(midpoint))
        bottom_start = int(np.ceil(midpoint))
        h, w = test_input.shape
        pred = test_input.copy()
        # Fill top region borders with c1
        if top_end >= 0:
            pred[0, :] = np.where(pred[0, :] == 0, c1, pred[0, :])
            if top_end < h:
                pred[top_end, :] = np.where(pred[top_end, :] == 0, c1, pred[top_end, :])
            for r in range(0, top_end + 1):
                if pred[r, 0] == 0: pred[r, 0] = c1
                if pred[r, w-1] == 0: pred[r, w-1] = c1
        # Fill bottom region borders with c2
        if bottom_start < h:
            pred[bottom_start, :] = np.where(pred[bottom_start, :] == 0, c2, pred[bottom_start, :])
            if h - 1 >= 0:
                pred[h-1, :] = np.where(pred[h-1, :] == 0, c2, pred[h-1, :])
            for r in range(bottom_start, h):
                if pred[r, 0] == 0: pred[r, 0] = c2
                if pred[r, w-1] == 0: pred[r, w-1] = c2
        # Keep markers
        pred[test_input == c1] = c1
        pred[test_input == c2] = c2
        if np.array_equal(pred, test_input):
            return None, 0.0, "grid_markers"
        return pred, 1.0, "grid_markers"

    def _solve_compact_colored_pixels(self, demo_pairs, test_input):
        """Extract scattered colored pixels into a compact grid.
        The input has a few colored pixels scattered in a grid.
        The output is a small grid preserving relative positions.
        """
        for inp, out in demo_pairs:
            if inp.ndim == 3: inp = inp[0]
            if out.ndim == 3: out = out[0]
        # Check: output is much smaller than input
        for inp, out in demo_pairs:
            if inp.shape[0] <= out.shape[0] or inp.shape[1] <= out.shape[1]:
                return None, 0.0, "compact_pixels"
        # Determine the output grid size
        out_h = demo_pairs[0][1].shape[0] if demo_pairs[0][1].ndim == 2 else demo_pairs[0][1].shape[1]
        out_w = demo_pairs[0][1].shape[1] if demo_pairs[0][1].ndim == 2 else demo_pairs[0][1].shape[2]
        # Check all pairs have same output size
        for inp, out in demo_pairs:
            o = out[0] if out.ndim == 3 else out
            if o.shape[0] != out_h or o.shape[1] != out_w:
                return None, 0.0, "compact_pixels"
        # For each pair, map colored pixels to grid positions
        for inp, out in demo_pairs:
            i = inp[0] if inp.ndim == 3 else inp
            o = out[0] if out.ndim == 3 else out
            colored = np.argwhere(i != 0)
            if len(colored) == 0:
                return None, 0.0, "compact_pixels"
            # Divide input into out_h x out_w zones
            in_h, in_w = i.shape
            # Map each colored pixel to its zone
            for r, c in colored:
                zone_r = min(int(r * out_h / in_h), out_h - 1)
                zone_c = min(int(c * out_w / in_w), out_w - 1)
                if o[zone_r, zone_c] != i[r, c]:
                    # Maybe the mapping is different
                    # Try: zone = r // (in_h // out_h), c // (in_w // out_w)
                    if in_h % out_h == 0 and in_w % out_w == 0:
                        zone_r2 = r // (in_h // out_h)
                        zone_c2 = c // (in_w // out_w)
                        if o[zone_r2, zone_c2] != i[r, c]:
                            return None, 0.0, "compact_pixels"
                    else:
                        return None, 0.0, "compact_pixels"
        # Apply to test
        t = test_input[0] if test_input.ndim == 3 else test_input
        in_h, in_w = t.shape
        pred = np.zeros((out_h, out_w), dtype=t.dtype)
        colored = np.argwhere(t != 0)
        for r, c in colored:
            if in_h % out_h == 0 and in_w % out_w == 0:
                zone_r = r // (in_h // out_h)
                zone_c = c // (in_w // out_w)
            else:
                zone_r = min(int(r * out_h / in_h), out_h - 1)
                zone_c = min(int(c * out_w / in_w), out_w - 1)
            pred[zone_r, zone_c] = t[r, c]
        return pred, 1.0, "compact_pixels"

    # ============ V3.6 Solvers ============

    def _solve_align_objects_row(self, demo_pairs, test_input):
        """Align colored objects vertically to the same row as an anchor color.
        The anchor color is the one whose row position doesn't change.
        All other colored objects shift to match the anchor's row range.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "align_row"
        # Find the anchor color: the color whose row range doesn't change
        anchor_color = None
        for inp, out in demo_pairs:
            colors = sorted(set(inp.flatten()) - {0})
            for c in colors:
                inp_rows = set(np.argwhere(inp == c)[:, 0].tolist())
                out_rows = set(np.argwhere(out == c)[:, 0].tolist())
                if inp_rows == out_rows:
                    if anchor_color is None:
                        anchor_color = c
                    elif anchor_color != c:
                        # Multiple non-moving colors — pick the one with most pixels
                        pass
        if anchor_color is None:
            return None, 0.0, "align_row"
        # Verify: anchor doesn't move, all others align to anchor's row range
        for inp, out in demo_pairs:
            anchor_rows = np.argwhere(inp == anchor_color)[:, 0]
            if len(anchor_rows) == 0:
                return None, 0.0, "align_row"
            anchor_rmin, anchor_rmax = int(anchor_rows.min()), int(anchor_rows.max())
            anchor_height = anchor_rmax - anchor_rmin + 1
            colors = sorted(set(inp.flatten()) - {0})
            for c in colors:
                if c == anchor_color:
                    continue
                inp_pixels = np.argwhere(inp == c)
                out_pixels = np.argwhere(out == c)
                if len(inp_pixels) == 0 or len(out_pixels) == 0:
                    return None, 0.0, "align_row"
                inp_rmin = int(inp_pixels[:, 0].min())
                inp_rmax = int(inp_pixels[:, 0].max())
                out_rmin = int(out_pixels[:, 0].min())
                out_rmax = int(out_pixels[:, 0].max())
                # Check output row range matches anchor's row range
                if out_rmin != anchor_rmin or out_rmax != anchor_rmax:
                    return None, 0.0, "align_row"
                # Check columns are preserved
                if not np.array_equal(np.sort(inp_pixels[:, 1]), np.sort(out_pixels[:, 1])):
                    return None, 0.0, "align_row"
                # Check object height is preserved
                if (inp_rmax - inp_rmin) != (out_rmax - out_rmin):
                    return None, 0.0, "align_row"
        # Apply to test
        anchor_rows = np.argwhere(test_input == anchor_color)[:, 0]
        if len(anchor_rows) == 0:
            return None, 0.0, "align_row"
        anchor_rmin, anchor_rmax = int(anchor_rows.min()), int(anchor_rows.max())
        pred = np.zeros_like(test_input)
        # Copy anchor color
        pred[test_input == anchor_color] = anchor_color
        # Shift each other color to align with anchor
        colors = sorted(set(test_input.flatten()) - {0})
        for c in colors:
            if c == anchor_color:
                continue
            pixels = np.argwhere(test_input == c)
            if len(pixels) == 0:
                continue
            rmin = int(pixels[:, 0].min())
            rmax = int(pixels[:, 0].max())
            obj_height = rmax - rmin + 1
            # Shift to align with anchor
            shift = anchor_rmin - rmin
            for r, col in pixels:
                new_r = int(r) + shift
                if 0 <= new_r < test_input.shape[0]:
                    pred[new_r, int(col)] = c
        if np.array_equal(pred, test_input):
            return None, 0.0, "align_row"
        return pred, 1.0, "align_row"

    def _solve_complete_tile_pattern(self, demo_pairs, test_input):
        """Complete a tile/diamond pattern by finding the repeating unit
        and filling in missing pixels.
        The pattern repeats with a specific period, and missing pixels
        are filled by copying from the same relative position in the pattern.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "tile_complete"
        # Find the pattern period from demos
        # The added pixels complete a repeating pattern
        for inp, out in demo_pairs:
            added = (inp == 0) & (out != 0)
            if not added.any():
                return None, 0.0, "tile_complete"
            # Check: added pixels should match existing pattern when shifted
            # Try different periods
            best_period = None
            best_matches = 0
            h, w = inp.shape
            for ph in range(1, h):
                for pw in range(1, w):
                    matches = 0
                    total = 0
                    for r, c in np.argwhere(added):
                        # Check if the pattern at (r, c) matches (r-ph, c) or (r+ph, c) etc.
                        for dr in [-ph, ph]:
                            nr = r + dr
                            if 0 <= nr < h and inp[nr, c] == out[r, c]:
                                matches += 1
                                break
                        else:
                            for dc in [-pw, pw]:
                                nc = c + dc
                                if 0 <= nc < w and inp[r, nc] == out[r, c]:
                                    matches += 1
                                    break
                        total += 1
                    if total > 0 and matches == total:
                        if matches > best_matches:
                            best_matches = matches
                            best_period = (ph, pw)
            if best_period is None:
                return None, 0.0, "tile_complete"
            ph, pw = best_period
            # Verify: the pattern is consistent
            for r, c in np.argwhere(added):
                found = False
                for dr in [-ph, ph]:
                    nr = r + dr
                    if 0 <= nr < h and inp[nr, c] == out[r, c]:
                        found = True
                        break
                if not found:
                    for dc in [-pw, pw]:
                        nc = c + dc
                        if 0 <= nc < w and inp[r, nc] == out[r, c]:
                            found = True
                            break
                if not found:
                    return None, 0.0, "tile_complete"
        # Find period from first demo
        inp0, out0 = demo_pairs[0]
        added0 = (inp0 == 0) & (out0 != 0)
        h, w = inp0.shape
        best_period = None
        for ph in range(1, h):
            for pw in range(1, w):
                all_match = True
                for r, c in np.argwhere(added0):
                    found = False
                    for dr in [-ph, ph]:
                        nr = r + dr
                        if 0 <= nr < h and inp0[nr, c] == out0[r, c]:
                            found = True
                            break
                    if not found:
                        for dc in [-pw, pw]:
                            nc = c + dc
                            if 0 <= nc < w and inp0[r, nc] == out0[r, c]:
                                found = True
                                break
                    if not found:
                        all_match = False
                        break
                if all_match and len(np.argwhere(added0)) > 0:
                    best_period = (ph, pw)
                    break
            if best_period:
                break
        if best_period is None:
            return None, 0.0, "tile_complete"
        ph, pw = best_period
        # Apply to test
        pred = test_input.copy()
        h, w = test_input.shape
        changed = True
        while changed:
            changed = False
            for r in range(h):
                for c in range(w):
                    if pred[r, c] == 0:
                        # Try to fill from pattern neighbors
                        for dr in [-ph, ph]:
                            nr = r + dr
                            if 0 <= nr < h and pred[nr, c] != 0:
                                pred[r, c] = pred[nr, c]
                                changed = True
                                break
                        if pred[r, c] == 0:
                            for dc in [-pw, pw]:
                                nc = c + dc
                                if 0 <= nc < w and pred[r, nc] != 0:
                                    pred[r, c] = pred[r, nc]
                                    changed = True
                                    break
        # Only keep changes that complete the pattern (fill 0s)
        result = test_input.copy()
        result[pred != test_input] = pred[pred != test_input]
        if np.array_equal(result, test_input):
            return None, 0.0, "tile_complete"
        return result, 1.0, "tile_complete"

    def _solve_knight_move_diagonal(self, demo_pairs, test_input):
        """For diagonal pairs of a marker color, place a fill color at
        knight's move positions perpendicular to the diagonal.
        Direction (1,1): knight moves (-1,+2) and (+1,-2)
        Direction (1,-1): knight moves (-1,-2) and (+1,+2)
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "knight_diag"
        # Find the marker color and fill color from first demo
        inp0, out0 = demo_pairs[0]
        added = (inp0 == 0) & (out0 != 0)
        if not added.any():
            return None, 0.0, "knight_diag"
        fill_color = int(out0[np.where(added)[0][0], np.where(added)[1][0]])
        # Marker color: the color that forms diagonal pairs
        colors = sorted(set(inp0.flatten()) - {0, fill_color})
        if not colors:
            return None, 0.0, "knight_diag"
        marker_color = None
        for c in colors:
            pixels = np.argwhere(inp0 == c)
            if len(pixels) >= 2:
                marker_color = c
                break
        if marker_color is None:
            return None, 0.0, "knight_diag"
        # Verify pattern for all demos
        for inp, out in demo_pairs:
            marker_pixels = np.argwhere(inp == marker_color)
            if len(marker_pixels) < 2:
                return None, 0.0, "knight_diag"
            added = (inp == 0) & (out == fill_color)
            # For each pair of adjacent marker pixels forming a diagonal
            for i in range(len(marker_pixels)):
                for j in range(i+1, len(marker_pixels)):
                    r1, c1 = marker_pixels[i]
                    r2, c2 = marker_pixels[j]
                    dr, dc = int(r2-r1), int(c2-c1)
                    # Check if diagonal (|dr| == |dc| and dr != 0)
                    if abs(dr) != abs(dc) or dr == 0:
                        continue
                    # Normalize direction
                    sdr = 1 if dr > 0 else -1
                    sdc = 1 if dc > 0 else -1
                    # Knight moves perpendicular to diagonal
                    if sdc > 0:  # going right
                        offsets = [(-1, 2), (1, -2)]
                    else:  # going left
                        offsets = [(-1, -2), (1, 2)]
                    for odr, odc in offsets:
                        for pr, pc in [(r1, c1), (r2, c2)]:
                            nr, nc = int(pr)+odr, int(pc)+odc
                            if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                                if inp[nr, nc] == 0:
                                    if out[nr, nc] != fill_color:
                                        return None, 0.0, "knight_diag"
            # Check no extra additions
            for r, c in np.argwhere(added):
                if out[r, c] != fill_color:
                    return None, 0.0, "knight_diag"
        # Apply to test
        pred = test_input.copy()
        marker_pixels = np.argwhere(test_input == marker_color)
        for i in range(len(marker_pixels)):
            for j in range(i+1, len(marker_pixels)):
                r1, c1 = marker_pixels[i]
                r2, c2 = marker_pixels[j]
                dr, dc = int(r2-r1), int(c2-c1)
                if abs(dr) != abs(dc) or dr == 0:
                    continue
                sdr = 1 if dr > 0 else -1
                sdc = 1 if dc > 0 else -1
                if sdc > 0:
                    offsets = [(-1, 2), (1, -2)]
                else:
                    offsets = [(-1, -2), (1, 2)]
                for odr, odc in offsets:
                    for pr, pc in [(r1, c1), (r2, c2)]:
                        nr, nc = int(pr)+odr, int(pc)+odc
                        if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
                            if pred[nr, nc] == 0:
                                pred[nr, nc] = fill_color
        if np.array_equal(pred, test_input):
            return None, 0.0, "knight_diag"
        return pred, 1.0, "knight_diag"

    def _solve_per_color_shift(self, demo_pairs, test_input):
        """Shift each color independently by its own (dr, dc) offset.
        Original positions become 0, new positions get the color.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "per_color_shift"
        # Find per-color shifts from first demo
        inp0, out0 = demo_pairs[0]
        colors = sorted(set(inp0.flatten()) - {0})
        color_shifts = {}
        for c in colors:
            inp_pixels = set(map(tuple, np.argwhere(inp0 == c)))
            out_pixels = set(map(tuple, np.argwhere(out0 == c)))
            if not inp_pixels or not out_pixels:
                continue
            # Find the shift that maps inp_pixels to out_pixels
            best_shift = None
            for inp_pos in inp_pixels:
                for out_pos in out_pixels:
                    shift = (out_pos[0] - inp_pos[0], out_pos[1] - inp_pos[1])
                    # Check if this shift works for all pixels of this color
                    shifted = set((r+shift[0], c+shift[1]) for r, c in inp_pixels)
                    if shifted == out_pixels:
                        best_shift = shift
                        break
                if best_shift:
                    break
            if best_shift is None:
                return None, 0.0, "per_color_shift"
            color_shifts[c] = best_shift
        if not color_shifts:
            return None, 0.0, "per_color_shift"
        # Verify across all demos
        for inp, out in demo_pairs:
            for c, (dr, dc) in color_shifts.items():
                inp_pixels = np.argwhere(inp == c)
                if len(inp_pixels) == 0:
                    continue
                for r, col in inp_pixels:
                    nr, nc = int(r)+dr, int(col)+dc
                    if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                        if out[nr, nc] != c:
                            return None, 0.0, "per_color_shift"
                    # Original position should be 0 (unless another color fills it)
                    if out[int(r), int(col)] == c:
                        # Check if another pixel of same color moved here
                        pass
        # Apply to test
        pred = np.zeros_like(test_input)
        for c, (dr, dc) in color_shifts.items():
            pixels = np.argwhere(test_input == c)
            for r, col in pixels:
                nr, nc = int(r)+dr, int(col)+dc
                if 0 <= nr < test_input.shape[0] and 0 <= nc < test_input.shape[1]:
                    pred[nr, nc] = c
        # Copy any colors that don't have shifts
        for c in colors:
            if c not in color_shifts:
                pred[test_input == c] = c
        if np.array_equal(pred, test_input):
            return None, 0.0, "per_color_shift"
        return pred, 1.0, "per_color_shift"

    def _solve_fill_between_objects(self, demo_pairs, test_input):
        """Fill 0s between same-colored objects with a specific color.
        Detects regions of 0 that are between two objects of the same color
        and fills them with a fill color learned from demos.
        """
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "fill_between_obj"
        # Learn fill color from first demo
        inp0, out0 = demo_pairs[0]
        added = (inp0 == 0) & (out0 != 0)
        if not added.any():
            return None, 0.0, "fill_between_obj"
        fill_colors = set(out0[added].tolist())
        if len(fill_colors) != 1:
            return None, 0.0, "fill_between_obj"
        fill_color = fill_colors.pop()
        # Verify: all added pixels are fill_color
        for inp, out in demo_pairs:
            added = (inp == 0) & (out != 0)
            if not added.any():
                return None, 0.0, "fill_between_obj"
            if not np.all(out[added] == fill_color):
                return None, 0.0, "fill_between_obj"
            # Check: no removals
            removed = (inp != 0) & (out == 0)
            if removed.any():
                return None, 0.0, "fill_between_obj"
        # Apply: fill all 0s with fill_color
        # But only if the pattern is "fill all 0s"
        # Check if demos fill ALL 0s or just some
        all_zeros_filled = True
        for inp, out in demo_pairs:
            zeros = inp == 0
            added = (inp == 0) & (out != 0)
            if not np.array_equal(zeros, added):
                all_zeros_filled = False
                break
        if all_zeros_filled:
            pred = test_input.copy()
            pred[test_input == 0] = fill_color
        else:
            # Try: fill 0s that are between same-color objects
            # Simple heuristic: fill 0s adjacent to non-zero pixels
            pred = test_input.copy()
            from scipy import ndimage
            for c in range(10):
                if c == 0 or c == fill_color:
                    continue
                mask = test_input == c
                if not mask.any():
                    continue
                # Dilate and fill adjacent 0s
                dilated = ndimage.binary_dilation(mask)
                fill_mask = dilated & (test_input == 0)
                pred[fill_mask] = fill_color
        if np.array_equal(pred, test_input):
            return None, 0.0, "fill_between_obj"
        return pred, 1.0, "fill_between_obj"

    def _solve_recolor_by_boundary(self, demo_pairs, test_input):
        """Recolor pixels of an object based on whether they are on the boundary
        or interior. Boundary pixels get one color, interior get another.
        Also handles: recolor based on adjacency to specific colors.
        """
        from scipy import ndimage
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "recolor_boundary"
        # Find the object color and recolor targets from first demo
        inp0, out0 = demo_pairs[0]
        changed = (inp0 != out0) & (inp0 != 0) & (out0 != 0)
        if not changed.any():
            return None, 0.0, "recolor_boundary"
        changed_from = set(inp0[changed].tolist())
        changed_to = set(out0[changed].tolist())
        if len(changed_from) != 1:
            return None, 0.0, "recolor_boundary"
        obj_color = changed_from.pop()
        # Find target colors
        target_colors = sorted(changed_to)
        if len(target_colors) < 1:
            return None, 0.0, "recolor_boundary"
        # Try: boundary → color1, interior → color2 (or vice versa)
        for inp, out in demo_pairs:
            obj_mask = inp == obj_color
            if not obj_mask.any():
                return None, 0.0, "recolor_boundary"
            eroded = ndimage.binary_erosion(obj_mask)
            boundary = obj_mask & ~eroded
            interior = eroded
            changed_mask = (inp != out) & (inp == obj_color)
            if not changed_mask.any():
                return None, 0.0, "recolor_boundary"
            # Check if boundary and interior get different colors
            boundary_changed = changed_mask & boundary
            interior_changed = changed_mask & interior
            if boundary_changed.any() and interior_changed.any():
                b_colors = set(out[boundary_changed].tolist())
                i_colors = set(out[interior_changed].tolist())
                if len(b_colors) == 1 and len(i_colors) == 1:
                    b_color = b_colors.pop()
                    i_color = i_colors.pop()
                    if b_color == i_color:
                        return None, 0.0, "recolor_boundary"
                else:
                    return None, 0.0, "recolor_boundary"
            elif boundary_changed.any() and not interior_changed.any():
                # Only boundary changes
                b_colors = set(out[boundary_changed].tolist())
                if len(b_colors) != 1:
                    return None, 0.0, "recolor_boundary"
            else:
                return None, 0.0, "recolor_boundary"
        # Determine the pattern
        inp0, out0 = demo_pairs[0]
        obj_mask0 = inp0 == obj_color
        eroded0 = ndimage.binary_erosion(obj_mask0)
        boundary0 = obj_mask0 & ~eroded0
        interior0 = eroded0
        changed0 = (inp0 != out0) & (inp0 == obj_color)
        b_changed0 = changed0 & boundary0
        i_changed0 = changed0 & interior0
        b_color = i_color = None
        if b_changed0.any():
            b_color = int(set(out0[b_changed0].tolist()).pop())
        if i_changed0.any():
            i_color = int(set(out0[i_changed0].tolist()).pop())
        # Apply to test
        pred = test_input.copy()
        obj_mask = test_input == obj_color
        if not obj_mask.any():
            return None, 0.0, "recolor_boundary"
        eroded = ndimage.binary_erosion(obj_mask)
        boundary = obj_mask & ~eroded
        interior = eroded
        if b_color is not None:
            pred[boundary] = b_color
        if i_color is not None:
            pred[interior] = i_color
        if np.array_equal(pred, test_input):
            return None, 0.0, "recolor_boundary"
        return pred, 1.0, "recolor_boundary"

    # ==================== V3.7 Solvers ====================

    def _solve_replicate_col_pattern(self, demo_pairs, test_input):
        """When input has very few non-zero pixels that define a column/row pattern,
        replicate that pattern across all rows/columns. e.g. 0a938d79."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "replicate_col"
        inp0 = demo_pairs[0][0]
        # Check sparsity
        nonzero_ratio = np.count_nonzero(inp0) / inp0.size
        if nonzero_ratio > 0.05 or np.count_nonzero(inp0) < 2:
            return None, 0.0, "replicate_col"
        if len(marker_cols) < 2 and len(marker_rows) < 2:
            return None, 0.0, "replicate_col"
        # Learn column vs row mode from demo 0's output
        out0 = demo_pairs[0][1]
        # If all rows of output are identical → column-based
        # If all columns of output are identical → row-based
        row_mode = True  # default: column-based (replicate across rows)
        if out0.shape[0] > 1:
            first_row = out0[0]
            all_same = all(np.array_equal(out0[r], first_row) for r in range(out0.shape[0]))
            if all_same:
                row_mode = True  # column-based
        if out0.shape[1] > 1 and not row_mode:
            first_col = out0[:, 0]
            all_same = all(np.array_equal(out0[:, c], first_col) for c in range(out0.shape[1]))
            if all_same:
                row_mode = False  # row-based
        # Collect markers from TEST INPUT
        markers = []
        for r in range(test_input.shape[0]):
            for c in range(test_input.shape[1]):
                if test_input[r, c] != 0:
                    markers.append((r, c, int(test_input[r, c])))
        if len(markers) < 2:
            return None, 0.0, "replicate_col"
        marker_cols = sorted(set(m[1] for m in markers))
        marker_rows = sorted(set(m[0] for m in markers))
        if row_mode and len(marker_cols) > 1:
            # Column-based replication
            min_col = marker_cols[0]
            max_col = marker_cols[-1]
            span = max_col - min_col
            period = span * 2
            if period < 2:
                period = span + 1
            col_pattern = {}
            for _, c, v in markers:
                col_pattern[(c - min_col) % period] = v
            pred = test_input.copy()
            H, W = pred.shape
            for r in range(H):
                for c in range(min_col, W):
                    offset = (c - min_col) % period
                    if offset in col_pattern and pred[r, c] == 0:
                        pred[r, c] = col_pattern[offset]
            if np.array_equal(pred, test_input):
                return None, 0.0, "replicate_col"
            return pred, 1.0, "replicate_col"
        elif row_span > 0:
            # Row-based replication
            min_row = marker_rows[0]
            max_row = marker_rows[-1]
            span = max_row - min_row
            period = span * 2
            if period < 2:
                period = span + 1
            row_pattern = {}
            for r, _, v in markers:
                row_pattern[(r - min_row) % period] = v
            pred = test_input.copy()
            H, W = pred.shape
            for r in range(min_row, H):
                for c in range(W):
                    offset = (r - min_row) % period
                    if offset in row_pattern and pred[r, c] == 0:
                        pred[r, c] = row_pattern[offset]
            if np.array_equal(pred, test_input):
                return None, 0.0, "replicate_col"
            return pred, 1.0, "replicate_col"
        return None, 0.0, "replicate_col"

    def _solve_extend_to_lines(self, demo_pairs, test_input):
        """Extend isolated markers to full lines (vertical or horizontal).
        e.g. 178fcbfb: 2→vertical line, 3→horizontal line, 1→horizontal line."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "extend_lines"
        # Learn directions from demos
        color_direction = {}  # color -> 'V' or 'H'
        for inp, out in demo_pairs:
            added = (out != 0) & (inp == 0)
            if not added.any():
                return None, 0.0, "extend_lines"
            for color in range(1, 10):
                color_added = added & (out == color)
                if not color_added.any():
                    continue
                rows_with = set(np.where(color_added.any(axis=1))[0].tolist())
                cols_with = set(np.where(color_added.any(axis=0))[0].tolist())
                if len(rows_with) > len(cols_with):
                    color_direction[color] = 'V'
                elif len(cols_with) > len(rows_with):
                    color_direction[color] = 'H'
                elif len(rows_with) == 1:
                    color_direction[color] = 'H'
                elif len(cols_with) == 1:
                    color_direction[color] = 'V'
        if not color_direction:
            return None, 0.0, "extend_lines"
        # Apply to test — process H first, then V only fills 0s
        pred = test_input.copy()
        H, W = pred.shape
        # First pass: horizontal extensions
        for color, direction in color_direction.items():
            if direction != 'H':
                continue
            mask = test_input == color
            if not mask.any():
                continue
            rows = np.where(mask.any(axis=1))[0]
            for r in rows:
                pred[r, :] = color
        # Second pass: vertical extensions (only fill 0s to avoid overwriting H lines)
        for color, direction in color_direction.items():
            if direction != 'V':
                continue
            mask = test_input == color
            if not mask.any():
                continue
            cols = np.where(mask.any(axis=0))[0]
            for c in cols:
                for r in range(H):
                    if pred[r, c] == 0:
                        pred[r, c] = color
        if np.array_equal(pred, test_input):
            return None, 0.0, "extend_lines"
        return pred, 1.0, "extend_lines"

    def _solve_grid_row_replicate(self, demo_pairs, test_input):
        """In a grid separated by lines of a separator color, replicate colored
        blocks to fill all cells in the same row. e.g. 06df4c85."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "grid_row_replicate"
        inp0 = demo_pairs[0][0]
        # Find separator color (most common non-zero color forming full lines)
        sep_color = None
        for c in range(1, 10):
            lines = 0
            for r in range(inp0.shape[0]):
                if np.all(inp0[r, :] == c) or np.all(inp0[r, :] == 0):
                    if np.all(inp0[r, :] == c):
                        lines += 1
            for col in range(inp0.shape[1]):
                if np.all(inp0[:, col] == c):
                    lines += 1
            if lines >= 2:
                sep_color = c
                break
        if sep_color is None:
            return None, 0.0, "grid_row_replicate"
        # Find grid boundaries
        h_sep_rows = [r for r in range(inp0.shape[0]) if np.all(inp0[r, :] == sep_color)]
        v_sep_cols = [c for c in range(inp0.shape[1]) if np.all(inp0[:, c] == sep_color)]
        if not h_sep_rows or not v_sep_cols:
            return None, 0.0, "grid_row_replicate"
        # Define grid cells
        row_bounds = [0] + [r + 1 for r in h_sep_rows] + [inp0.shape[0]]
        col_bounds = [0] + [c + 1 for c in v_sep_cols] + [inp0.shape[1]]
        # For each row of cells, find the pattern and replicate
        pred = test_input.copy()
        H, W = pred.shape
        h_sep_test = [r for r in range(H) if np.all(pred[r, :] == sep_color)]
        v_sep_test = [c for c in range(W) if np.all(pred[:, c] == sep_color)]
        row_b = [0] + [r + 1 for r in h_sep_test] + [H]
        col_b = [0] + [c + 1 for c in v_sep_test] + [W]
        for ri in range(len(row_b) - 1):
            r_start, r_end = row_b[ri], row_b[ri + 1] - 1
            if r_start >= r_end:
                continue
            # Find non-separator, non-zero pattern in this row of cells
            template = None
            for ci in range(len(col_b) - 1):
                c_start, c_end = col_b[ci], col_b[ci + 1] - 1
                if c_start >= c_end:
                    continue
                cell = pred[r_start:r_end + 1, c_start:c_end + 1].copy()
                cell[cell == sep_color] = 0
                if np.any(cell):
                    template = cell
                    template_cols = (c_start, c_end)
                    break
            if template is not None:
                for ci in range(len(col_b) - 1):
                    c_start, c_end = col_b[ci], col_b[ci + 1] - 1
                    if c_start >= c_end:
                        continue
                    cell = pred[r_start:r_end + 1, c_start:c_end + 1]
                    cell_no_sep = cell.copy()
                    cell_no_sep[cell == sep_color] = 0
                    if not np.any(cell_no_sep):
                        th, tw = template.shape
                        ch, cw = cell.shape
                        for tr in range(min(th, ch)):
                            for tc in range(min(tw, cw)):
                                if template[tr, tc] != 0 and cell[tr, tc] == 0:
                                    pred[r_start + tr, c_start + tc] = template[tr, tc]
        if np.array_equal(pred, test_input):
            return None, 0.0, "grid_row_replicate"
        return pred, 1.0, "grid_row_replicate"

    def _solve_expand_plus_diamond(self, demo_pairs, test_input):
        """Expand a plus/cross pattern of markers around a center into a
        diamond pattern. e.g. 0962bcdd."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "expand_diamond"
        inp0, out0 = demo_pairs[0]
        added = (out0 != 0) & (inp0 == 0)
        if not added.any():
            return None, 0.0, "expand_diamond"
        # Find ALL plus patterns in test_input and expand them all
        pred = test_input.copy()
        H, W = pred.shape
        found_any = False
        for center_color in range(1, 10):
            for arm_color in range(1, 10):
                if center_color == arm_color:
                    continue
                centers = []
                for r in range(1, H - 1):
                    for c in range(1, W - 1):
                        if (test_input[r, c] == center_color and
                            test_input[r - 1, c] == arm_color and
                            test_input[r + 1, c] == arm_color and
                            test_input[r, c - 1] == arm_color and
                            test_input[r, c + 1] == arm_color):
                            centers.append((r, c))
                if not centers:
                    continue
                found_any = True
                for cr, cc in centers:
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            r, c = cr + dr, cc + dc
                            if not (0 <= r < H and 0 <= c < W):
                                continue
                            if dr == 0 and dc == 0:
                                if pred[r, c] == 0 or pred[r, c] == center_color:
                                    pred[r, c] = center_color
                            elif (dr == 0 or dc == 0) and max(abs(dr), abs(dc)) <= 2:
                                if pred[r, c] == 0 or pred[r, c] == arm_color:
                                    pred[r, c] = arm_color
                            elif abs(dr) == abs(dc) and abs(dr) <= 2:
                                if pred[r, c] == 0:
                                    pred[r, c] = center_color
        if not found_any or np.array_equal(pred, test_input):
            return None, 0.0, "expand_diamond"
        return pred, 1.0, "expand_diamond"

    def _solve_draw_diagonal_line(self, demo_pairs, test_input):
        """Draw diagonal lines from a marker. e.g. 1f0c79e5."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "draw_diagonal"
        inp0, out0 = demo_pairs[0]
        added = (out0 != 0) & (inp0 == 0)
        if not added.any():
            return None, 0.0, "draw_diagonal"
        # Find the main object and the direction marker
        inp_colors = set(int(v) for v in inp0[inp0 != 0])
        if len(inp_colors) < 2:
            return None, 0.0, "draw_diagonal"
        # The most common color is the main object, the other is the direction marker
        color_counts = {c: np.count_nonzero(inp0 == c) for c in inp_colors}
        main_color = max(color_counts, key=color_counts.get)
        marker_colors = inp_colors - {main_color}
        if not marker_colors:
            return None, 0.0, "draw_diagonal"
        marker_color = marker_colors.pop()
        # Find marker position
        marker_pos = np.argwhere(inp0 == marker_color)
        if len(marker_pos) == 0:
            return None, 0.0, "draw_diagonal"
        mr, mc = marker_pos[0]
        # Find main object center
        main_pos = np.argwhere(inp0 == main_color)
        if len(main_pos) == 0:
            return None, 0.0, "draw_diagonal"
        obj_r, obj_c = main_pos[0]
        # Direction from object to marker
        dr = 1 if mr > obj_r else -1
        dc = 1 if mc > obj_c else -1
        # Draw diagonal with width 3 (horizontal triple)
        pred = test_input.copy()
        H, W = pred.shape
        # Find test marker and object
        test_marker = np.argwhere(test_input == marker_color)
        test_main = np.argwhere(test_input == main_color)
        if len(test_marker) == 0 or len(test_main) == 0:
            return None, 0.0, "draw_diagonal"
        tr, tc = test_main[0]
        # Draw diagonal line
        r, c = tr, tc
        while 0 <= r < H and 0 <= c < W:
            for dc_w in [-1, 0, 1]:
                cc = c + dc_w
                if 0 <= cc < W and pred[r, cc] == 0:
                    pred[r, cc] = main_color
            r += dr
            c += dc
        if np.array_equal(pred, test_input):
            return None, 0.0, "draw_diagonal"
        return pred, 1.0, "draw_diagonal"

    def _solve_draw_rect_borders(self, demo_pairs, test_input):
        """Draw rectangular borders around markers. e.g. 1bfc4729."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "rect_borders"
        inp0, out0 = demo_pairs[0]
        added = (out0 != 0) & (inp0 == 0)
        if not added.any():
            return None, 0.0, "rect_borders"
        # Find markers (isolated non-zero pixels)
        markers = []
        for r in range(inp0.shape[0]):
            for c in range(inp0.shape[1]):
                if inp0[r, c] != 0:
                    markers.append((r, c, int(inp0[r, c])))
        if not markers:
            return None, 0.0, "rect_borders"
        # Determine radius from demo — only check small neighborhood
        radius = 2
        for mr, mc, mv in markers:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    r, c = mr + dr, mc + dc
                    if 0 <= r < inp0.shape[0] and 0 <= c < inp0.shape[1]:
                        if out0[r, c] == mv and inp0[r, c] == 0:
                            radius = max(radius, max(abs(dr), abs(dc)))
        # Apply to test
        pred = test_input.copy()
        H, W = pred.shape
        test_markers = []
        for r in range(H):
            for c in range(W):
                if test_input[r, c] != 0:
                    test_markers.append((r, c, int(test_input[r, c])))
        for mr, mc, mv in test_markers:
            top = max(0, mr - radius)
            bot = min(H - 1, mr + radius)
            # Check which edge is closer to grid boundary
            top_closer = mr < H - 1 - mr
            # Draw filled top or bottom row, marker row, and sides
            for r in range(top, bot + 1):
                for c in range(W):
                    if r == mr or (top_closer and r == top) or (not top_closer and r == bot):
                        if pred[r, c] == 0:
                            pred[r, c] = mv
                    elif c == 0 or c == W - 1:
                        if pred[r, c] == 0:
                            pred[r, c] = mv
        if np.array_equal(pred, test_input):
            return None, 0.0, "rect_borders"
        return pred, 1.0, "rect_borders"

    def _solve_mirror_grid_fill(self, demo_pairs, test_input):
        """In a grid, fill empty cells with a mirror/copy of patterns from
        other cells in the same row, using the separator color. e.g. 1e32b0e9."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "mirror_grid"
        inp0 = demo_pairs[0][0]
        # Find separator color
        sep_color = None
        for c in range(1, 10):
            h_lines = sum(1 for r in range(inp0.shape[0]) if np.all(inp0[r, :] == c))
            v_lines = sum(1 for c_col in range(inp0.shape[1]) if np.all(inp0[:, c_col] == c))
            if h_lines + v_lines >= 4:
                sep_color = c
                break
        if sep_color is None:
            return None, 0.0, "mirror_grid"
        # Find grid cells
        h_seps = [r for r in range(inp0.shape[0]) if np.all(inp0[r, :] == sep_color)]
        v_seps = [c for c in range(inp0.shape[1]) if np.all(inp0[:, c] == sep_color)]
        if len(h_seps) < 2 or len(v_seps) < 2:
            return None, 0.0, "mirror_grid"
        row_bounds = [0] + [r + 1 for r in h_seps] + [inp0.shape[0]]
        col_bounds = [0] + [c + 1 for c in v_seps] + [inp0.shape[1]]
        # For each row of cells, find the pattern and fill empty cells
        pred = test_input.copy()
        H, W = pred.shape
        h_s = [r for r in range(H) if np.all(pred[r, :] == sep_color)]
        v_s = [c for c in range(W) if np.all(pred[:, c] == sep_color)]
        rb = [0] + [r + 1 for r in h_s] + [H]
        cb = [0] + [c + 1 for c in v_s] + [W]
        for ri in range(len(rb) - 1):
            r_start, r_end = rb[ri], rb[ri + 1] - 1
            if r_start >= r_end:
                continue
            # Find pattern cell
            template = None
            for ci in range(len(cb) - 1):
                c_start, c_end = cb[ci], cb[ci + 1] - 1
                if c_start >= c_end:
                    continue
                cell = pred[r_start:r_end + 1, c_start:c_end + 1].copy()
                cell[cell == sep_color] = 0
                if np.any(cell):
                    template = cell.copy()
                    break
            if template is not None:
                for ci in range(len(cb) - 1):
                    c_start, c_end = cb[ci], cb[ci + 1] - 1
                    if c_start >= c_end:
                        continue
                    cell = pred[r_start:r_end + 1, c_start:c_end + 1].copy()
                    cell_no_sep = cell.copy()
                    cell_no_sep[cell == sep_color] = 0
                    if not np.any(cell_no_sep):
                        th, tw = template.shape
                        ch, cw = cell.shape
                        for tr in range(min(th, ch)):
                            for tc in range(min(tw, cw)):
                                if template[tr, tc] != 0 and template[tr, tc] != sep_color:
                                    if pred[r_start + tr, c_start + tc] == 0:
                                        pred[r_start + tr, c_start + tc] = sep_color
        if np.array_equal(pred, test_input):
            return None, 0.0, "mirror_grid"
        return pred, 1.0, "mirror_grid"

    def _solve_traffic_jam_shift(self, demo_pairs, test_input):
        """Shift objects right by 1, with rightmost pixels acting as walls.
        e.g. 025d127b."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "traffic_shift"
        from scipy import ndimage
        inp0, out0 = demo_pairs[0]
        # Check that it's a same-size transform with shifts
        added = np.sum((out0 > 0) & (inp0 == 0))
        removed = np.sum((out0 == 0) & (inp0 > 0))
        if added == 0 or removed == 0:
            return None, 0.0, "traffic_shift"
        # Find unique colors
        colors = set(int(v) for v in inp0[inp0 != 0])
        if not colors:
            return None, 0.0, "traffic_shift"
        pred = test_input.copy()
        H, W = pred.shape
        for color in colors:
            mask = test_input == color
            if not mask.any():
                continue
            # Label connected components
            labeled, n_objs = ndimage.label(mask)
            for obj_id in range(1, n_objs + 1):
                obj_mask = labeled == obj_id
                obj_cols = np.where(obj_mask.any(axis=0))[0]
                if len(obj_cols) == 0:
                    continue
                max_col = obj_cols[-1]
                # Process each row from right to left
                for r in range(H):
                    row_cols = np.where(obj_mask[r])[0]
                    if len(row_cols) == 0:
                        continue
                    # Sort descending (right to left)
                    row_cols = sorted(row_cols, reverse=True)
                    for c in row_cols:
                        if c == max_col:
                            continue  # Wall pixel, stays
                        target = c + 1
                        if target <= max_col and pred[r, target] == 0:
                            pred[r, target] = color
                            pred[r, c] = 0
        if np.array_equal(pred, test_input):
            return None, 0.0, "traffic_shift"
        return pred, 1.0, "traffic_shift"

    def _solve_fill_enclosed_by_marker(self, demo_pairs, test_input):
        """Fill enclosed regions inside border objects with a color indicated by
        scattered markers outside. e.g. 228f6490."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "fill_enclosed_marker"
        from scipy import ndimage
        inp0, out0 = demo_pairs[0]
        # Find border color (forms enclosed shapes)
        for border_color in range(1, 10):
            border_mask = inp0 == border_color
            if not border_mask.any():
                continue
            # Find enclosed regions
            inverted = border_mask == False
            labeled, n_regions = ndimage.label(inverted)
            enclosed = []
            for reg_id in range(1, n_regions + 1):
                reg_mask = labeled == reg_id
                # Check if region touches the border of the grid
                reg_rows = np.where(reg_mask.any(axis=1))[0]
                reg_cols = np.where(reg_mask.any(axis=0))[0]
                if len(reg_rows) == 0:
                    continue
                touches_edge = (reg_rows[0] == 0 or reg_rows[-1] == inp0.shape[0] - 1 or
                               reg_cols[0] == 0 or reg_cols[-1] == inp0.shape[1] - 1)
                if not touches_edge:
                    enclosed.append(reg_mask)
            if not enclosed:
                continue
            # Find markers (non-zero, non-border colors outside the border)
            marker_pixels = []
            for r in range(inp0.shape[0]):
                for c in range(inp0.shape[1]):
                    v = inp0[r, c]
                    if v != 0 and v != border_color:
                        # Check if it's outside the border object
                        in_enclosed = any(em[r, c] for em in enclosed)
                        if not in_enclosed:
                            marker_pixels.append((r, c, int(v)))
            if not marker_pixels:
                continue
            # Match markers to enclosed regions by proximity
            pred = test_input.copy()
            H, W = pred.shape
            test_border = pred == border_color
            if not test_border.any():
                continue
            test_inv = test_border == False
            test_labeled, test_n = ndimage.label(test_inv)
            test_enclosed = []
            for reg_id in range(1, test_n + 1):
                reg_mask = test_labeled == reg_id
                reg_rows = np.where(reg_mask.any(axis=1))[0]
                reg_cols = np.where(reg_mask.any(axis=0))[0]
                if len(reg_rows) == 0:
                    continue
                touches_edge = (reg_rows[0] == 0 or reg_rows[-1] == H - 1 or
                               reg_cols[0] == 0 or reg_cols[-1] == W - 1)
                if not touches_edge:
                    test_enclosed.append(reg_mask)
            # Find test markers
            test_markers = []
            for r in range(H):
                for c in range(W):
                    v = pred[r, c]
                    if v != 0 and v != border_color:
                        in_enc = any(em[r, c] for em in test_enclosed)
                        if not in_enc:
                            test_markers.append((r, c, int(v)))
            # Match each enclosed region to nearest marker
            for em in test_enclosed:
                em_rows = np.where(em.any(axis=1))[0]
                em_cols = np.where(em.any(axis=0))[0]
                if len(em_rows) == 0:
                    continue
                center_r = (em_rows[0] + em_rows[-1]) / 2
                center_c = (em_cols[0] + em_cols[-1]) / 2
                best_dist = float('inf')
                best_color = 0
                for mr, mc, mv in test_markers:
                    d = abs(mr - center_r) + abs(mc - center_c)
                    if d < best_dist:
                        best_dist = d
                        best_color = mv
                if best_color != 0:
                    pred[em] = best_color
            if not np.array_equal(pred, test_input):
                return pred, 1.0, "fill_enclosed_marker"
        return None, 0.0, "fill_enclosed_marker"

    def _solve_replicate_adjacent_pattern(self, demo_pairs, test_input):
        """Replicate patterns adjacent to a template object with a period
        defined by the template size. e.g. 045e512c."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "replicate_adjacent"
        inp0, out0 = demo_pairs[0]
        added = (out0 != 0) & (inp0 == 0)
        if not added.any():
            return None, 0.0, "replicate_adjacent"
        # Find the largest object (template)
        from scipy import ndimage
        colors = set(int(v) for v in inp0[inp0 != 0])
        best_template = None
        best_size = 0
        for color in colors:
            mask = inp0 == color
            labeled, n = ndimage.label(mask)
            for i in range(1, n + 1):
                size = np.sum(labeled == i)
                if size > best_size:
                    best_size = size
                    best_template = (color, labeled == i)
        if best_template is None or best_size < 4:
            return None, 0.0, "replicate_adjacent"
        template_color, template_mask = best_template
        # Get template bounding box
        rows = np.where(template_mask.any(axis=1))[0]
        cols = np.where(template_mask.any(axis=0))[0]
        t_r0, t_r1 = rows[0], rows[-1]
        t_c0, t_c1 = cols[0], cols[-1]
        t_h = t_r1 - t_r0 + 1
        t_w = t_c1 - t_c0 + 1
        period_h = t_h + 1
        period_w = t_w + 1
        # Find adjacent patterns (non-template colors near the template)
        pred = test_input.copy()
        H, W = pred.shape
        # Find template in test input
        test_template_mask = test_input == template_color
        if not test_template_mask.any():
            return None, 0.0, "replicate_adjacent"
        test_rows = np.where(test_template_mask.any(axis=1))[0]
        test_cols = np.where(test_template_mask.any(axis=0))[0]
        tt_r0, tt_r1 = test_rows[0], test_rows[-1]
        tt_c0, tt_c1 = test_cols[0], test_cols[-1]
        # Find adjacent patterns in test input
        # Right side: replicate horizontally with period_w
        for color in colors:
            if color == template_color:
                continue
            color_mask = test_input == color
            if not color_mask.any():
                continue
            color_cols = np.where(color_mask.any(axis=0))[0]
            color_rows = np.where(color_mask.any(axis=1))[0]
            if len(color_cols) == 0:
                continue
            # Check if it's to the right of template (horizontal replication)
            if color_cols[0] > tt_c1:
                # Replicate horizontally
                c_start = color_cols[0]
                pattern_cols = color_cols.tolist()
                for pc in pattern_cols:
                    offset = pc - c_start
                    for r in color_rows:
                        v = test_input[r, pc]
                        if v == color:
                            # Replicate
                            target_c = c_start + offset + period_w
                            while target_c < W:
                                if pred[r, target_c] == 0:
                                    pred[r, target_c] = color
                                target_c += period_w
            # Check if it's below template (vertical replication)
            elif color_rows[0] > tt_r1:
                r_start = color_rows[0]
                pattern_rows = color_rows.tolist()
                for pr in pattern_rows:
                    offset = pr - r_start
                    for c in color_cols:
                        v = test_input[pr, c]
                        if v == color:
                            target_r = r_start + offset + period_h
                            while target_r < H:
                                if pred[target_r, c] == 0:
                                    pred[target_r, c] = color
                                target_r += period_h
        if np.array_equal(pred, test_input):
            return None, 0.0, "replicate_adjacent"
        return pred, 1.0, "replicate_adjacent"

    def _solve_zone_dominant_fill(self, demo_pairs, test_input):
        """Fill zones between separator lines with the dominant color of each zone.
        e.g. 09629e4f."""
        for inp, out in demo_pairs:
            if inp.shape != out.shape:
                return None, 0.0, "zone_fill"
        inp0, out0 = demo_pairs[0]
        # Find separator lines (full rows or columns of a single color)
        sep_color = None
        sep_rows = []
        sep_cols = []
        for c in range(1, 10):
            rows = [r for r in range(inp0.shape[0]) if np.all(inp0[r, :] == c)]
            cols = [col for col in range(inp0.shape[1]) if np.all(inp0[:, col] == c)]
            if len(rows) + len(cols) >= 2:
                sep_color = c
                sep_rows = rows
                sep_cols = cols
                break
        if sep_color is None:
            return None, 0.0, "zone_fill"
        # Define zones (regions between separator lines)
        row_bounds = [0] + [r + 1 for r in sep_rows] + [inp0.shape[0]]
        col_bounds = [0] + [c + 1 for c in sep_cols] + [inp0.shape[1]]
        # For each zone, find the dominant color and fill
        pred = test_input.copy()
        H, W = pred.shape
        test_sep_rows = [r for r in range(H) if np.all(pred[r, :] == sep_color)]
        test_sep_cols = [c for c in range(W) if np.all(pred[:, c] == sep_color)]
        test_rb = [0] + [r + 1 for r in test_sep_rows] + [H]
        test_cb = [0] + [c + 1 for c in test_sep_cols] + [W]
        for ri in range(len(test_rb) - 1):
            for ci in range(len(test_cb) - 1):
                r0, r1 = test_rb[ri], test_rb[ri + 1]
                c0, c1 = test_cb[ci], test_cb[ci + 1]
                if r0 >= r1 or c0 >= c1:
                    continue
                zone = pred[r0:r1, c0:c1]
                # Find dominant non-zero, non-sep color
                zone_colors = zone[(zone != 0) & (zone != sep_color)]
                if len(zone_colors) == 0:
                    continue
                # Find the most common color
                vals, counts = np.unique(zone_colors, return_counts=True)
                dominant = int(vals[np.argmax(counts)])
                # Fill the entire zone with dominant color, preserving separators
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        if pred[r, c] != sep_color:
                            pred[r, c] = dominant
        if np.array_equal(pred, test_input):
            return None, 0.0, "zone_fill"
        return pred, 1.0, "zone_fill"
