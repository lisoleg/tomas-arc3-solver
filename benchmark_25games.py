"""Quick benchmark for UniversalSolverPipeline Phase A/B optimization.

Tests 25 games across first 3 levels, measuring RHAE performance.
"""
import sys
sys.path.insert(0, '.')

import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_game
from src.agent.game_profiles import ALL_GAME_BASELINES
import copy, time

def check_solved(game, original_level):
    """Check if game has advanced past original_level."""
    return hasattr(game, '_current_level_index') and game._current_level_index > original_level

arc = Arcade(operation_mode=OperationMode.OFFLINE)

# All 25 games
ALL_GAMES = [
    'ls20', 'vc33', 'tr87', 'tu93', 're86', 'g50t', 'wa30',
    'ft09', 's5i5', 'tn36', 'su15', 'r11l', 'lp85',
    'bp35', 'dc22', 'sk48', 'lf52', 'sc25', 'm0r0', 'cn04',
    'cd82', 'sp80', 'ka59', 'ar25', 'sb26',
]

baselines = ALL_GAME_BASELINES
N_LEVELS_TO_TEST = 3

results = {}
total_solved = 0
total_rhae = 0.0
total_levels = 0

for gid in ALL_GAMES:
    bl_list = baselines.get(gid, [])
    level_scores = []
    
    for li in range(N_LEVELS_TO_TEST):
        if li >= len(bl_list):
            break
        
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
            continue
        
        baseline = bl_list[li]
        
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
        
        rhae = min(115.0, (baseline / max(1, steps))**2 * 100) if solved else 0
        level_scores.append((li, solved, steps, baseline, rhae, round(elapsed, 1)))
        total_levels += 1
        if solved:
            total_solved += 1
            total_rhae += rhae
    
    results[gid] = level_scores
    solved_n = sum(1 for s in level_scores if s[1])
    if solved_n > 0:
        avg = sum(s[4] for s in level_scores if s[1]) / solved_n
        print(f'{gid}: {solved_n}/{len(level_scores)} solved, avg RHAE={avg:.1f} | {level_scores}')
    else:
        print(f'{gid}: 0/{len(level_scores)} solved | {level_scores}')

print()
print('=== OVERALL SUMMARY ===')
overall_avg = total_rhae / max(1, total_solved)
print(f'Total: {total_solved}/{total_levels} levels solved ({total_solved/total_levels*100:.1f}%)')
print(f'Average RHAE (solved levels): {overall_avg:.1f}')
print(f'Effective RHAE (all levels): {total_rhae/total_levels:.1f}')
