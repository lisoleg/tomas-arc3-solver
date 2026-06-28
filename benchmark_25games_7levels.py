"""Full 7-level benchmark for all 25 games.

Tests all 25 games × 7 levels = 175 levels, measuring RHAE performance.
"""
import sys
sys.path.insert(0, '.')

import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput, GameState
from src.agent.game_solvers import solve_game
from src.agent.game_profiles import ALL_GAME_BASELINES
import copy, time

# v3.32.0: Games whose lambda closures break after deepcopy (okllwtboml dict).
# For these games, replay actions on original game instead of deepcopy verification.
_DEEPCOPY_UNSAFE_GAMES = frozenset({"tn36"})

def check_solved(game, original_level):
    """Check if game has advanced past original_level."""
    return (hasattr(game, '_current_level_index') and game._current_level_index > original_level) \
        or (hasattr(game, '_state') and game._state == GameState.WIN)

arc = Arcade(operation_mode=OperationMode.OFFLINE)

# All 25 games
ALL_GAMES = [
    'ls20', 'vc33', 'tr87', 'tu93', 're86', 'g50t', 'wa30',
    'ft09', 's5i5', 'tn36', 'su15', 'r11l', 'lp85',
    'bp35', 'dc22', 'sk48', 'lf52', 'sc25', 'm0r0', 'cn04',
    'cd82', 'sp80', 'ka59', 'ar25', 'sb26',
]

baselines = ALL_GAME_BASELINES
N_LEVELS_TO_TEST = 7

results = {}
total_solved = 0
total_rhae = 0.0
total_levels = 0
game_rhae_totals = {}  # gid → sum of RHAE across all levels
game_max_rhae = {}     # gid → max possible RHAE (115.0 × n_levels)

for gid in ALL_GAMES:
    bl_list = baselines.get(gid, [])
    level_scores = []
    game_rhae_sum = 0.0
    n_valid_levels = min(N_LEVELS_TO_TEST, len(bl_list))

    for li in range(n_valid_levels):
        baseline = bl_list[li]

        try:
            env = arc.make(gid)
            obs = env.step(GameAction.RESET)
            g2 = env._game

            if li > 0:
                g2._current_level_index = li
                if hasattr(g2, 'on_set_level'):
                    g2.on_set_level(g2._levels[li])
        except Exception as e:
            print(f'{gid} L{li}: LOAD ERROR {e}')
            level_scores.append((li, False, 999, baseline, 0.0, 0.0))
            continue

        t0 = time.time()
        plan = solve_game(g2, gid, level_idx=li)
        elapsed = time.time() - t0

        solved = False
        steps = len(plan) if plan else 999
        if plan:
            # v3.32.0: Δ-State Replay for deepcopy-unsafe games
            if gid in _DEEPCOPY_UNSAFE_GAMES:
                # Replay on original game (solver was called on original too)
                for step in plan[:300]:
                    aid, data = step
                    ai = ActionInput(id=aid, data=data if data else {})
                    try:
                        g2.perform_action(ai)
                    except Exception:
                        pass
                    for _ in range(2):
                        try:
                            g2.complete_action()
                        except Exception:
                            break
                    if check_solved(g2, li):
                        solved = True
                        break
            else:
                # Deepcopy-safe games: verify on fresh copy
                g3 = copy.deepcopy(g2)
                for step in plan[:300]:
                    aid, data = step
                    ai = ActionInput(id=aid, data=data if data else {})
                    try:
                        g3.perform_action(ai)
                    except Exception:
                        pass
                    for _ in range(2):
                        try:
                            g3.complete_action()
                        except Exception:
                            break
                    if check_solved(g3, li):
                        solved = True
                        break

        rhae = min(115.0, (baseline / max(1, steps))**2 * 100) if solved else 0
        level_scores.append((li, solved, steps, baseline, round(rhae, 1), round(elapsed, 1)))
        total_levels += 1
        game_rhae_sum += rhae
        if solved:
            total_solved += 1
            total_rhae += rhae

    results[gid] = level_scores
    game_rhae_totals[gid] = game_rhae_sum
    game_max_rhae[gid] = n_valid_levels * 115.0

    # Print per-game summary
    solved_n = sum(1 for s in level_scores if s[1])
    scores_str = " | ".join(f"L{s[0]}:{s[4]}" for s in level_scores)
    print(f'{gid}: {solved_n}/{n_valid_levels} solved, RHAE={game_rhae_sum:.1f}/{n_valid_levels*115.0:.0f} | {scores_str}')

print()
print('=== OVERALL SUMMARY ===')
print(f'Total: {total_solved}/{total_levels} levels solved ({total_solved/max(1,total_levels)*100:.1f}%)')
if total_solved > 0:
    print(f'Average RHAE (solved levels): {total_rhae/total_solved:.1f}')
print(f'Effective RHAE (all levels): {total_rhae/max(1,total_levels):.1f}')

# Print game-by-game RHAE table
print()
print('=== RHAE SCORE TABLE ===')
print(f'{"Game":<8} {"RHAE":<10} {"Max":<10} {"Pct":<8} {"Status"}')
for gid in ALL_GAMES:
    rhae_sum = game_rhae_totals.get(gid, 0.0)
    max_sum = game_max_rhae.get(gid, 0.0)
    pct = rhae_sum / max(1, max_sum) * 100
    n_valid = int(max_sum / 115.0)
    solved_n = sum(1 for s in results.get(gid, []) if s[1])
    status = "✅" if solved_n == n_valid else "⚠️" if solved_n > 0 else "❌"
    print(f'{gid:<8} {rhae_sum:<10.1f} {max_sum:<10.0f} {pct:<8.1f}% {status} {solved_n}/{n_valid}')

total_sum = sum(game_rhae_totals.values())
total_max = sum(game_max_rhae.values())
print(f'{"TOTAL":<8} {total_sum:<10.1f} {total_max:<10.0f} {total_sum/max(1,total_max)*100:<8.1f}%')
