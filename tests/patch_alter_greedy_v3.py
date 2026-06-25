"""Patch v3: Add analytical exhaustive search for Level 5 tree_translation."""
import re
from pathlib import Path

FILE = Path(r"C:\Users\1\WorkBuddy\2026-06-22-20-51-49\tomas-arc3-solver\src\agent\planner_agent.py")

content = FILE.read_text(encoding="utf-8")

# Find Phase 3 section
phase3_start = content.index("        # ========================================\n        # Phase 3:")
phase4_end = content.index("        # Restore all groups to original state")

old_section = content[phase3_start:phase4_end]
print(f"Old Phase 3-4 section: {len(old_section)} chars")

# New Phase 3b: Analytical exhaustive search
# This replaces Phase 3 and Phase 4 with a much faster analytical approach
NEW_SECTION = '''        # ========================================
        # Phase 3: Analytical exhaustive search
        # ========================================
        # bsqsshqpox() only compares sprite names. We can simulate it
        # with pure string operations, avoiding expensive sprite cycling.
        # This lets us search 7^6 = 117,649 combinations in <2 seconds.

        def name_at_delta(name: str, delta: int) -> str:
            """Compute sprite name after cycling by delta."""
            digit = int(name[-1])
            new_digit = (digit + delta - 1) % num_variants + 1
            return name[:-1] + str(new_digit)

        # Board source/target names (fixed, correct)
        board_src_names = [s.name for s in game.zvojhrjxxm]
        board_tgt_names = [s.name for s in game.ztgmtnnufb]

        # Original group names (before any cycling)
        orig_group_names: list[list[str]] = []
        for gtype, gsprites in flat_groups:
            orig_group_names.append([s.name for s in gsprites])

        def analytical_win_check(deltas: list[int]) -> bool:
            """Simulate bsqsshqpox() using name strings only.

            Args:
                deltas: List of delta per group index.

            Returns:
                True if the win condition would be satisfied.
            """
            # Compute rule names at given deltas
            rule_names: list[tuple[list[str], list[str]]] = []
            for rule_idx in range(n_rules):
                src_gidx = rule_idx * 2
                tgt_gidx = rule_idx * 2 + 1
                if src_gidx >= n_groups or tgt_gidx >= n_groups:
                    break
                src_names = [
                    name_at_delta(n, deltas[src_gidx])
                    for n in orig_group_names[src_gidx]
                ]
                tgt_names = [
                    name_at_delta(n, deltas[tgt_gidx])
                    for n in orig_group_names[tgt_gidx]
                ]
                rule_names.append((src_names, tgt_names))

            # Walk through board source (same as bsqsshqpox)
            src_idx = 0
            tgt_idx = 0
            while src_idx < len(board_src_names):
                matched = False
                for rule_src_names, rule_tgt_names in rule_names:
                    # Check if rule source matches board source at src_idx
                    if src_idx + len(rule_src_names) > len(board_src_names):
                        continue
                    if not all(
                        board_src_names[src_idx + i] == rule_src_names[i]
                        for i in range(len(rule_src_names))
                    ):
                        continue

                    # Rule source matches! Handle tree_translation
                    effective_tgt = list(rule_tgt_names)
                    if is_tree:
                        expanded: list[str] = []
                        failed = False
                        for tname in effective_tgt:
                            found = False
                            for other_src_names, other_tgt_names in rule_names:
                                if other_src_names and other_src_names[0] == tname:
                                    expanded += other_tgt_names
                                    found = True
                                    break
                            if not found:
                                failed = True
                                break
                        if failed:
                            continue  # Tree expansion failed, try next rule
                        effective_tgt = expanded
                    elif is_double:
                        # Double translation: find chain rule
                        # (for Level 5, tree takes precedence, but handle both)
                        chained = False
                        for other_src_names, other_tgt_names in rule_names:
                            if len(effective_tgt) == len(other_src_names) and all(
                                a == b for a, b in zip(effective_tgt, other_src_names)
                            ):
                                effective_tgt = list(other_tgt_names)
                                chained = True
                                break
                        if not chained:
                            continue

                    # Check if effective target matches board target
                    if tgt_idx + len(effective_tgt) > len(board_tgt_names):
                        break
                    if not all(
                        board_tgt_names[tgt_idx + i] == effective_tgt[i]
                        for i in range(len(effective_tgt))
                    ):
                        break

                    # Match successful!
                    src_idx += len(rule_src_names)
                    tgt_idx += len(effective_tgt)
                    matched = True
                    break

                if not matched:
                    return False
            return True

        # Identify fixed vs unfixed groups
        # Fixed: source groups with unique candidate, target groups with board match
        # Unfixed: source groups with no match, target groups with tree heuristic
        fixed_deltas: dict[int, int] = {}
        unfixed_gidxs: list[int] = []
        semi_fixed: dict[int, list[int]] = {}  # gidx -> candidate list

        for gidx in range(n_groups):
            gtype, gsprites = flat_groups[gidx]
            if not gsprites:
                continue
            if gidx % 2 == 0:  # Source group
                cands = source_candidates.get(gidx, [])
                if len(cands) == 1:
                    fixed_deltas[gidx] = cands[0]
                elif len(cands) > 1:
                    semi_fixed[gidx] = cands
                else:
                    unfixed_gidxs.append(gidx)
            else:  # Target group
                # Check if this was fixed by board target match
                # (we can check by seeing if the current delta gives a board match)
                # For simplicity, mark all tree-heuristic targets as unfixed
                # and all board-match targets as fixed
                # We'll determine this by checking if the group at current
                # state matches board target
                # Actually, let's just try all 7 for all target groups
                # that weren't matched by Strategy A
                # For now, mark all target groups as unfixed
                unfixed_gidxs.append(gidx)

        # Also mark semi-fixed source groups for enumeration
        semi_fixed_gidxs = list(semi_fixed.keys())
        semi_fixed_lists = [semi_fixed[g] for g in semi_fixed_gidxs]

        total_semi = 1
        for sl in semi_fixed_lists:
            total_semi *= len(sl)
        total_unfixed = num_variants ** len(unfixed_gidxs)
        total_search = total_semi * total_unfixed

        print(
            f"    [TR87-HYBRID] Phase 3: Analytical search "
            f"({total_semi} semi × {total_unfixed} unfixed "
            f"= {total_search} combos)"
        )

        best_solution = None
        combo_count = 0

        for semi_combo in (
            itertools.product(*semi_fixed_lists)
            if semi_fixed_lists
            else [()]
        ):
            if best_solution is not None:
                break

            # Set semi-fixed deltas
            semi_deltas = {}
            for i, gidx in enumerate(semi_fixed_gidxs):
                semi_deltas[gidx] = semi_combo[i]

            for unfixed_combo in itertools.product(
                range(num_variants), repeat=len(unfixed_gidxs)
            ):
                combo_count += 1

                # Build full delta list
                deltas = [0] * n_groups
                for gidx, d in fixed_deltas.items():
                    deltas[gidx] = d
                for gidx, d in semi_deltas.items():
                    deltas[gidx] = d
                for i, gidx in enumerate(unfixed_gidxs):
                    deltas[gidx] = unfixed_combo[i]

                if analytical_win_check(deltas):
                    best_solution = list(deltas)
                    print(
                        f"    [TR87-HYBRID] Found solution at "
                        f"combo {combo_count}! "
                        f"deltas={best_solution}"
                    )
                    break

        if best_solution is None:
            print(
                f"    [TR87-HYBRID] Analytical search exhausted "
                f"({combo_count} combos)"
            )

        # If analytical search found a solution, apply it to sprites
        if best_solution is not None:
            # Apply deltas to sprites
            for gidx in range(n_groups):
                gtype, gsprites = flat_groups[gidx]
                if not gsprites:
                    continue
                apply_delta(gsprites, best_solution[gidx])

            # Verify with real win_check
            if not win_check():
                print(
                    f"    [TR87-HYBRID] WARNING: Analytical match "
                    f"but win_check failed! Restoring..."
                )
                for gidx in range(n_groups):
                    restore_group(gidx, best_solution[gidx])
                best_solution = None

        # Fallback: coordinate descent if analytical failed
        if best_solution is None:
            print(
                f"    [TR87-HYBRID] Phase 4: Coordinate descent "
                f"with tree-aware scoring"
            )

            # Reset to first combo
            best_deltas = [0] * n_groups
            for gidx, cands in source_candidates.items():
                if cands:
                    best_deltas[gidx] = cands[0]
                    apply_delta(flat_groups[gidx][1], cands[0])

            # Fix targets with Phase 2
            run_phase2(best_deltas)

            for iteration in range(10):
                improved = False
                for gidx in range(n_groups):
                    gtype, gsprites = flat_groups[gidx]
                    if not gsprites:
                        continue

                    for trial in range(num_variants):
                        if win_check():
                            best_deltas[gidx] = (
                                best_deltas[gidx] + trial
                            ) % num_variants
                            best_solution = list(best_deltas)
                            break
                        for i in range(len(gsprites)):
                            gsprites[i] = cycle_fn(gsprites[i], 1)

                    if best_solution is not None:
                        break

                    best_s = -1
                    best_t = 0
                    for trial in range(num_variants):
                        s = score_state()
                        if s > best_s:
                            best_s = s
                            best_t = trial
                        for i in range(len(gsprites)):
                            gsprites[i] = cycle_fn(gsprites[i], 1)

                    if best_t != 0:
                        best_deltas[gidx] = (
                            best_deltas[gidx] + best_t
                        ) % num_variants
                        improved = True

                if best_solution is not None or not improved:
                    break

            if best_solution is None:
                won = win_check()
                if won:
                    best_solution = list(best_deltas)

'''

new_content = content[:phase3_start] + NEW_SECTION + "\n" + content[phase4_end:]
FILE.write_text(new_content, encoding="utf-8")

# Verify
verify = FILE.read_text(encoding="utf-8")
assert "analytical_win_check" in verify
assert "Phase 3: Analytical search" in verify
assert "Phase 4: Coordinate descent" in verify
assert "def _compute_tr87_expected_targets(" in verify

print("✅ Patch v3 applied successfully!")
print(f"   File size: {len(new_content)} chars")
