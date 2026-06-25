"""Patch script v2: replace _solve_tr87_alter_greedy with comprehensive search."""
import re
from pathlib import Path

FILE = Path(r"C:\Users\1\WorkBuddy\2026-06-22-20-51-49\tomas-arc3-solver\src\agent\planner_agent.py")

content = FILE.read_text(encoding="utf-8")

# Find the method boundaries
start_marker = "    def _solve_tr87_alter_greedy("
end_marker = "    def _compute_tr87_expected_targets("

start_idx = content.index(start_marker)
end_idx = content.index(end_marker)

old_method = content[start_idx:end_idx]
print(f"Old method length: {len(old_method)} chars, {old_method.count(chr(10))} lines")

NEW_METHOD = '''    def _solve_tr87_alter_greedy(
        self,
        game: Any,
        flat_groups: list,
        source_patterns: list,
        target_patterns: list,
        current_index: int,
        num_variants: int,
        win_check: Any,
        cycle_fn: Any,
    ) -> Optional[list]:
        """Solve tr87 alter_rules with chain-aware matching for complex modes.

        Multi-phase algorithm:
        Phase 1: Fix source groups by matching against board source patterns.
                 Records ALL candidate deltas for groups with multiple matches.
        Phase 2: Fix target groups by matching against board target or
                 chain source groups.
        Phase 3: If win_check fails, try all source candidate combinations.
                 For each, re-run Phase 2 and check win.
        Phase 4: If still failing, coordinate descent with tree-aware scoring.

        Args:
            game: The env._game object.
            flat_groups: Flattened [(type, sprites), ...] list.
            source_patterns: Board source sprites (correct).
            target_patterns: Board target sprites (correct).
            current_index: Current selector position.
            num_variants: Number of variants per pattern.
            win_check: Callable game.bsqsshqpox.
            cycle_fn: Callable game.wpbnovjwkv.

        Returns:
            List of (GameAction, name) tuples, or None.
        """
        from arcengine import GameAction
        import itertools

        n_groups = len(flat_groups)
        rules = game.cifzvbcuwqe
        n_rules = len(rules)
        is_double = game.current_level.get_data("double_translation")
        is_tree = game.current_level.get_data("tree_translation")

        # Save original names for restoration
        original_names: list[list[str]] = []
        for gtype, gsprites in flat_groups:
            original_names.append([s.name for s in gsprites])

        def apply_delta(gsprites: list, delta: int) -> None:
            """Cycle gsprites by delta positions."""
            for i in range(len(gsprites)):
                for _ in range(delta):
                    gsprites[i] = cycle_fn(gsprites[i], 1)

        def restore_group(gidx: int, delta: int) -> None:
            """Restore group gidx from delta back to original."""
            gtype, gsprites = flat_groups[gidx]
            restore = (num_variants - delta) % num_variants
            apply_delta(gsprites, restore)

        def find_deltas_vs_reference(
            gsprites: list, ref_sprites: list
        ) -> list[int]:
            """Try all variants, return deltas where gsprites match ref.

            After calling, gsprites is back to original state.
            """
            deltas: list[int] = []
            for delta in range(num_variants):
                if len(gsprites) == len(ref_sprites):
                    if all(
                        a.name == b.name
                        for a, b in zip(gsprites, ref_sprites)
                    ):
                        deltas.append(delta)
                elif len(gsprites) < len(ref_sprites):
                    for start in range(
                        len(ref_sprites) - len(gsprites) + 1
                    ):
                        if all(
                            gsprites[i].name
                            == ref_sprites[start + i].name
                            for i in range(len(gsprites))
                        ):
                            deltas.append(delta)
                            break
                for i in range(len(gsprites)):
                    gsprites[i] = cycle_fn(gsprites[i], 1)
            return deltas

        def score_state() -> int:
            """Comprehensive scoring: board source + tree readiness + board target."""
            score = 0
            # 1. Count matched board source segments
            src_idx = 0
            board_src = game.zvojhrjxxm
            while src_idx < len(board_src):
                matched = False
                for rule_src, rule_tgt in rules:
                    if game.iwbhnvdaao(board_src, src_idx, rule_src):
                        score += len(rule_src) * 100
                        # 2. Tree expansion readiness for this rule
                        if is_tree:
                            for sprite in rule_tgt:
                                for other_src, _ in rules:
                                    if other_src and other_src[0].name == sprite.name:
                                        score += 50
                                        break
                        src_idx += len(rule_src)
                        matched = True
                        break
                if not matched:
                    src_idx += 1
            # 3. Count matched board target segments
            tgt_idx = 0
            board_tgt = game.ztgmtnnufb
            while tgt_idx < len(board_tgt):
                matched = False
                for _, rule_tgt in rules:
                    if game.iwbhnvdaao(board_tgt, tgt_idx, rule_tgt):
                        score += len(rule_tgt) * 10
                        tgt_idx += len(rule_tgt)
                        matched = True
                        break
                if not matched:
                    tgt_idx += 1
            return score

        # ========================================
        # Phase 1: Fix source groups against board source
        # ========================================
        print(
            f"    [TR87-HYBRID] Phase 1: Matching {n_rules} source groups "
            f"against {len(source_patterns)} board source patterns"
        )

        # Record ALL candidate deltas for each source group
        source_candidates: dict[int, list[int]] = {}

        for rule_idx in range(n_rules):
            src_gidx = rule_idx * 2
            if src_gidx >= n_groups:
                break
            gtype, gsprites = flat_groups[src_gidx]
            if not gsprites:
                continue

            deltas = find_deltas_vs_reference(gsprites, source_patterns)
            source_candidates[src_gidx] = deltas if deltas else [0]

            if deltas:
                print(
                    f"      Rule {rule_idx} source: {len(deltas)} candidates "
                    f"={deltas}"
                )
            else:
                print(
                    f"      Rule {rule_idx} source: NO board match "
                    f"(chain/sub rule)"
                )

        def run_phase2(best_deltas: list[int]) -> bool:
            """Run Phase 2 target matching. Returns True if win_check passes."""
            # Fix target groups
            for rule_idx in range(n_rules):
                tgt_gidx = rule_idx * 2 + 1
                if tgt_gidx >= n_groups:
                    break
                gtype, gsprites = flat_groups[tgt_gidx]
                if not gsprites:
                    continue

                # Strategy A: Match against board target
                deltas = find_deltas_vs_reference(
                    gsprites, target_patterns
                )
                if deltas:
                    best_deltas[tgt_gidx] = deltas[0]
                    apply_delta(gsprites, deltas[0])
                    continue

                # Strategy B: Match against other rules' source groups
                if is_double or is_tree:
                    found = False
                    for other_idx in range(n_rules):
                        if other_idx == rule_idx:
                            continue
                        other_src_gidx = other_idx * 2
                        if other_src_gidx >= n_groups:
                            continue
                        _, other_gsprites = flat_groups[other_src_gidx]
                        if not other_gsprites:
                            continue
                        chain_deltas = find_deltas_vs_reference(
                            gsprites, other_gsprites
                        )
                        if chain_deltas:
                            best_deltas[tgt_gidx] = chain_deltas[0]
                            apply_delta(gsprites, chain_deltas[0])
                            found = True
                            break
                    if found:
                        continue

                # Strategy C: Tree heuristic
                if is_tree:
                    best_td = 0
                    best_ts = -1
                    for delta in range(num_variants):
                        s = sum(
                            1 for s in gsprites
                            if any(
                                other_src and other_src[0].name == s.name
                                for other_src, _ in rules
                            )
                        )
                        if s > best_ts:
                            best_ts = s
                            best_td = delta
                        for i in range(len(gsprites)):
                            gsprites[i] = cycle_fn(gsprites[i], 1)
                    best_deltas[tgt_gidx] = best_td
                    apply_delta(gsprites, best_td)

            return win_check()

        # ========================================
        # Phase 3: Try all source candidate combinations
        # ========================================
        # Build list of source groups with multiple candidates
        multi_candidate_gidxs = [
            gidx for gidx, cands in source_candidates.items()
            if len(cands) > 1
        ]
        multi_candidate_lists = [
            source_candidates[gidx] for gidx in multi_candidate_gidxs
        ]

        total_combos = 1
        for cl in multi_candidate_lists:
            total_combos *= len(cl)

        print(
            f"    [TR87-HYBRID] Phase 3: Trying {total_combos} source "
            f"candidate combinations"
        )

        best_solution: Optional[list[int]] = None

        for combo in itertools.product(*multi_candidate_lists) if multi_candidate_lists else [()]:
            # Apply source group deltas
            best_deltas = [0] * n_groups
            for i, gidx in enumerate(multi_candidate_gidxs):
                best_deltas[gidx] = combo[i]
                apply_delta(flat_groups[gidx][1], combo[i])
            # Apply single-candidate source deltas
            for gidx, cands in source_candidates.items():
                if gidx not in multi_candidate_gidxs and cands:
                    best_deltas[gidx] = cands[0]
                    apply_delta(flat_groups[gidx][1], cands[0])

            # Run Phase 2
            if run_phase2(best_deltas):
                best_solution = list(best_deltas)
                print(
                    f"    [TR87-HYBRID] Found solution with "
                    f"source combo={combo}"
                )
                break

            # Restore all groups for next combo
            for gidx in range(n_groups):
                restore_group(gidx, best_deltas[gidx])

        if best_solution is None:
            # ========================================
            # Phase 4: Coordinate descent with tree-aware scoring
            # ========================================
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
            run_phase2(best_deltas)

            for iteration in range(10):
                improved = False
                for gidx in range(n_groups):
                    gtype, gsprites = flat_groups[gidx]
                    if not gsprites:
                        continue

                    # Try all variants with win_check
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

                    # Tree-aware scoring fallback
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

        # Restore all groups to original state
        if best_solution is not None:
            for gidx in range(n_groups):
                restore_group(gidx, best_solution[gidx])
        else:
            for gidx in range(n_groups):
                restore_group(gidx, best_deltas[gidx])

        if best_solution is None:
            print(f"    [TR87-HYBRID] Failed to find solution")
            return None

        # Generate actions from original state to correct state
        actions: list = []
        cur_idx = current_index
        has_cycling = False

        for gidx in range(n_groups):
            delta = best_solution[gidx]
            if delta == 0:
                continue

            gtype, gsprites = flat_groups[gidx]
            if not gsprites:
                continue

            # Navigate to group
            nav_right = (gidx - cur_idx) % n_groups
            nav_left = (cur_idx - gidx) % n_groups
            if nav_right <= nav_left:
                for _ in range(nav_right):
                    actions.append((GameAction.ACTION4, "RIGHT"))
            else:
                for _ in range(nav_left):
                    actions.append((GameAction.ACTION3, "LEFT"))
            cur_idx = gidx

            # Cycle variant
            if delta <= num_variants // 2:
                for _ in range(delta):
                    actions.append((GameAction.ACTION2, "DOWN"))
            else:
                for _ in range(num_variants - delta):
                    actions.append((GameAction.ACTION1, "UP"))
            has_cycling = True

        # Trigger win check if needed
        if not has_cycling:
            actions.append((GameAction.ACTION2, "DOWN"))
            actions.append((GameAction.ACTION1, "UP"))
        elif actions[-1][0] not in (
            GameAction.ACTION1,
            GameAction.ACTION2,
        ):
            actions.append((GameAction.ACTION2, "DOWN"))
            actions.append((GameAction.ACTION1, "UP"))

        print(
            f"    [TR87-HYBRID] Generated {len(actions)} actions "
            f"for {n_groups} groups (deltas={best_solution})"
        )
        return actions if actions else []

'''

new_content = content[:start_idx] + NEW_METHOD + "\n" + content[end_idx:]
FILE.write_text(new_content, encoding="utf-8")

# Verify
verify = FILE.read_text(encoding="utf-8")
assert "def _solve_tr87_alter_greedy(" in verify
assert "def _compute_tr87_expected_targets(" in verify
assert "Phase 3: Trying" in verify
assert "Phase 4: Coordinate descent" in verify

print("✅ Patch v2 applied successfully!")
print(f"   File size: {len(new_content)} chars")
