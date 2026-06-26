"""Full 25-game RHAE benchmark for TOMAS ARC-AGI-3 Solver.

Tests all 25 games across available levels, computing RHAE scores.
Uses solve_game() from game_solvers.py as the entry point.

RHAE formula: min(115.0, (baseline / max(1, steps))^2 * 100) if solved, else 0.
"""
import sys
sys.path.insert(0, '.')

import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_game
from src.agent.game_profiles import ALL_GAME_BASELINES
import copy
import time
import json

def check_solved(game, original_level):
    """Check if game has advanced past original_level."""
    return hasattr(game, '_current_level_index') and game._current_level_index > original_level

arc = Arcade(operation_mode=OperationMode.OFFLINE)

ALL_GAMES = [
    'ls20', 'vc33', 'tr87', 'tu93', 're86', 'g50t', 'wa30',
    'ft09', 's5i5', 'tn36', 'su15', 'r11l', 'lp85',
    'bp35', 'dc22', 'sk48', 'lf52', 'sc25', 'm0r0', 'cn04',
    'cd82', 'sp80', 'ka59', 'ar25', 'sb26',
]

baselines = ALL_GAME_BASELINES

results = {}
total_solved = 0
total_levels = 0
total_rhae = 0.0
game_scores = {}

for gid in ALL_GAMES:
    bl_list = baselines.get(gid, [])
    level_scores = []
    game_total_rhae = 0.0
    game_solved = 0
    
    for li in range(min(7, len(bl_list))):
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
            g3 = copy.deepcopy(g2)
            for step in plan[:300]:
                aid, data = step
                ai = ActionInput(id=aid, data=data if data else {})
                try:
                    g3.perform_action(ai)
                except Exception:
                    pass
                if check_solved(g3, li):
                    solved = True
                    break
        
        rhae = 0.0
        if solved:
            rhae = min(115.0, (baseline / max(1, steps))**2 * 100)
            game_solved += 1
        
        game_total_rhae += rhae
        total_rhae += rhae
        total_levels += 1
        
        if solved:
            total_solved += 1
        
        level_scores.append((li, solved, steps, baseline, round(rhae, 2), round(elapsed, 1)))
        status = "✅" if solved else "❌"
        print(f'{gid} L{li}: {status} steps={steps}/{baseline} RHAE={rhae:.1f} time={elapsed:.1f}s')
    
    game_scores[gid] = {
        'levels': level_scores,
        'total_rhae': game_total_rhae,
        'levels_solved': game_solved,
        'levels_tested': len(level_scores),
        'avg_rhae': round(game_total_rhae / max(1, len(level_scores)), 2),
    }

# Summary
print("\n" + "="*70)
print("25-GAME RHAE BENCHMARK SUMMARY")
print("="*70)
print(f"Total levels tested: {total_levels}")
print(f"Total levels solved: {total_solved}")
print(f"Overall avg RHAE: {total_rhae / max(1, total_levels):.2f}")
print()

# Sort by avg_rhae descending
sorted_games = sorted(game_scores.items(), key=lambda x: x[1]['avg_rhae'], reverse=True)
print(f"{'Game':>6} {'Solved':>6} {'Tested':>6} {'Avg RHAE':>8} {'Total RHAE':>10}")
print("-"*40)
for gid, info in sorted_games:
    print(f"{gid:>6} {info['levels_solved']:>6} {info['levels_tested']:>6} {info['avg_rhae']:>8.2f} {info['total_rhae']:>10.2f}")

# Identify zero-score games
zero_games = [gid for gid, info in game_scores.items() if info['avg_rhae'] == 0.0]
print(f"\n❌ Zero-score games ({len(zero_games)}): {', '.join(zero_games)}")

# Save results
with open('benchmark_results.json', 'w') as f:
    json.dump(game_scores, f, indent=2)
print("\nResults saved to benchmark_results.json")
