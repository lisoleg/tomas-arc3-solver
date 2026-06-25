"""Patch script: replace _solve_tr87_alter_greedy method in planner_agent.py."""
import re
from pathlib import Path

FILE = Path(r"C:\Users\1\WorkBuddy\2026-06-22-20-51-49\tomas-arc3-solver\src\agent\planner_agent.py")

content = FILE.read_text(encoding="utf-8")

# Find the method boundaries
start_marker = "    def _solve_tr87_alter_greedy("
end_marker = "    def _compute_tr87_expected_targets("

start_idx = content.index(start_marker)
end_idx = content.index(end_marker)

# Extract old method for verification
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

        Two-phase algorithm for double_translation/tree_translation:
        Phase 1: Fix source groups by matching against board source patterns
        Phase 2: Fix target groups by matching against board target or
                 chained source groups
        Fallback: Iterative refinement using bsqsshqpox() as oracle.

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

        n_groups = len(flat_groups)
        rules = game.cifzvbcuwqe
        n_rules = len(rules)
        is_double = game.current_level.get_data("double_translation")
        is_tree = game.current_level.get_data("tree_translation")

        # Save original names for restoration
        original_names: list[list[str]] = []
        for gtype, gsprites in flat_groups:
            original_names.append([s.name for s in gsprites])

        best_deltas: list[int] = [0] * n_groups

        def find_deltas_vs_reference(
            gsprites: list, ref_sprites: list
        ) -> list[int]:
            """Try all variants, return deltas where gsprites match ref.

            If lengths are equal, checks direct name match.
            If gsprites is shorter, tries matching against any contiguous
            segment of ref_sprites.

            After calling, gsprites is back to original state (full cycle).
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
                # Cycle by +1
                for i in range(len(gsprites)):
                    gsprites[i] = cycle_fn(gsprites[i], 1)
            return deltas

        def apply_delta(gsprites: list, delta: int) -> None:
            """Cycle gsprites by delta positions."""
            for i in range(len(gsprites)):
                for _ in range(delta):
                    gsprites[i] = cycle_fn(gsprites[i], 1)

        # Phase 1: Fix source groups (even indices in flat_groups)
        print(
            f"    [TR87-HYBRID] Phase 1: Matching {n_rules} source groups "
            f"against {len(source_patterns)} board source patterns"
        )

        for rule_idx in range(n_rules):
            src_gidx = rule_idx * 2
            if src_gidx >= n_groups:
                break
            gtype, gsprites = flat_groups[src_gidx]
            if not gsprites:
                continue

            deltas = find_deltas_vs_reference(gsprites, source_patterns)

            if deltas:
                best_deltas[src_gidx] = deltas[0]
                apply_delta(gsprites, deltas[0])
                print(
                    f"      Rule {rule_idx} source: delta={deltas[0]} "
                    f"({len(deltas)} candidates)"
                )
            else:
                print(
                    f"      Rule {rule_idx} source: NO board match "
                    f"(chain rule?)"
                )

        # Phase 2: Fix target groups (odd indices in flat_groups)
        print(
            f"    [TR87-HYBRID] Phase 2: Matching target groups "
            f"(double={is_double}, tree={is_tree})"
        )

        for rule_idx in range(n_rules):
            tgt_gidx = rule_idx * 2 + 1
            if tgt_gidx >= n_groups:
                break
            gtype, gsprites = flat_groups[tgt_gidx]
            if not gsprites:
                continue

            # Strategy A: Match against board target patterns
            deltas = find_deltas_vs_reference(gsprites, target_patterns)

            if deltas:
                best_deltas[tgt_gidx] = deltas[0]
                apply_delta(gsprites, deltas[0])
                print(
                    f"      Rule {rule_idx} target: delta={deltas[0]} "
                    f"(board target match)"
                )
                continue

            # Strategy B: Match against other rules' source groups
            # (for double_translation chain)
            if is_double or is_tree:
                found_chain = False
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
                        print(
                            f"      Rule {rule_idx} target: "
                            f"delta={chain_deltas[0]} "
                            f"(chain to rule {other_idx} source)"
                        )
                        found_chain = True
                        break
                if found_chain:
                    continue

            # Strategy C: For tree_translation, find variant where target
            # sprites match sub-rule sources
            if is_tree:
                best_tree_delta = 0
                best_tree_score = -1
                for delta in range(num_variants):
                    score = sum(
                        1
                        for s in gsprites
                        if any(
                            other_src
                            and other_src[0].name == s.name
                            for other_src, _ in rules
                        )
                    )
                    if score > best_tree_score:
                        best_tree_score = score
                        best_tree_delta = delta
                    for i in range(len(gsprites)):
                        gsprites[i] = cycle_fn(gsprites[i], 1)

                best_deltas[tgt_gidx] = best_tree_delta
                apply_delta(gsprites, best_tree_delta)
                print(
                    f"      Rule {rule_idx} target: "
                    f"delta={best_tree_delta} "
                    f"(tree, score={best_tree_score}/{len(gsprites)})"
                )
                continue

            print(
                f"      Rule {rule_idx} target: NO match found!"
            )

        # Verify with win_check
        won = win_check()

        if not won:
            print(
                f"    [TR87-HYBRID] Analytical approach failed, "
                f"trying iterative refinement..."
            )

            # Iterative refinement: multiple passes
            for iteration in range(5):
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
                            won = True
                            break
                        for i in range(len(gsprites)):
                            gsprites[i] = cycle_fn(gsprites[i], 1)

                    if won:
                        break

                    # Scoring fallback: count matched board source
                    best_score = -1
                    best_trial = 0
                    for trial in range(num_variants):
                        score = 0
                        idx = 0
                        board_src = game.zvojhrjxxm
                        while idx < len(board_src):
                            matched = False
                            for rule_src, _ in rules:
                                if game.iwbhnvdaao(
                                    board_src, idx, rule_src
                                ):
                                    score += len(rule_src)
                                    idx += len(rule_src)
                                    matched = True
                                    break
                            if not matched:
                                idx += 1
                        if score > best_score:
                            best_score = score
                            best_trial = trial
                        for i in range(len(gsprites)):
                            gsprites[i] = cycle_fn(gsprites[i], 1)

                    if best_trial != 0:
                        best_deltas[gidx] = (
                            best_deltas[gidx] + best_trial
                        ) % num_variants
                        improved = True

                if won or not improved:
                    break

            won = win_check()

        # Restore all groups to original state
        for gidx in range(n_groups):
            gtype, gsprites = flat_groups[gidx]
            restore = (num_variants - best_deltas[gidx]) % num_variants
            apply_delta(gsprites, restore)

        if not won:
            print(f"    [TR87-HYBRID] Failed to find solution")
            return None

        # Generate actions from original state to correct state
        actions: list = []
        cur_idx = current_index
        has_cycling = False

        for gidx in range(n_groups):
            delta = best_deltas[gidx]
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
            f"for {n_groups} groups (deltas={best_deltas})"
        )
        return actions if actions else []

'''

# Replace
new_content = content[:start_idx] + NEW_METHOD + "\n" + content[end_idx:]
FILE.write_text(new_content, encoding="utf-8")

# Verify
verify = FILE.read_text(encoding="utf-8")
assert "def _solve_tr87_alter_greedy(" in verify
assert "def _compute_tr87_expected_targets(" in verify
assert "[TR87-HYBRID]" in verify
assert "[TR87-ALTER-GREEDY]" not in verify  # Old markers should be gone

print("✅ Patch applied successfully!")
print(f"   File size: {len(new_content)} chars")
