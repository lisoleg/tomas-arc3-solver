"""Full 25-game benchmark for Phase A/B optimization validation."""
import sys
sys.path.insert(0, '.')
import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_game
from src.agent.game_profiles import ALL_GAME_BASELINES
import copy, time

def check_solved(game, original_level):
    return hasattr(game, '_current_level_index') and game._current_level_index > original_level

arc = Arcade(operation_mode=OperationMode.OFFLINE)

all_games = list(ALL_GAME_BASELINES.keys())
baselines = ALL_GAME_BASELINES

# Only test first 3 levels per game for speed
MAX_LEVELS = 3

results = {}
total_solved = 0
total_levels = 0
total_rhae = 0.0

for gid in all_games:
    bl_list = baselines[gid]
    env = arc.make(gid)
    obs = env.step(GameAction.RESET)
    game = env._game
    n_levels = len(game._levels)
    n_test = min(n_levels, MAX_LEVELS)
    
    level_scores = []
    for li in range(n_test):
        env2 = arc.make(gid)
        obs2 = env2.step(GameAction.RESET)
        g2 = env2._game
        if li > 0:
            g2._current_level_index = li
            if hasattr(g2, 'on_set_level'):
                g2.on_set_level(g2._levels[li])
        
        baseline = bl_list[li] if li < len(bl_list) else 50
        
        t0 = time.time()
        plan = solve_game(g2, gid)
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
                except:
                    pass
                if check_solved(g3, li):
                    solved = True
                    break
        
        rhae = min(115.0, (baseline / max(1, steps))**2 * 100) if solved else 0
        level_scores.append((li, solved, steps, baseline, round(rhae, 1), round(elapsed, 1)))
        
        if solved:
            total_solved += 1
            total_rhae += rhae
        total_levels += 1
    
    results[gid] = level_scores
    solved_n = sum(1 for s in level_scores if s[1])
    avg_rhae = sum(s[4] for s in level_scores if s[1]) / max(1, solved_n)
    print(f'{gid}: {solved_n}/{n_test} solved, avg RHAE={avg_rhae:.1f}  |  {level_scores}')

print()
print('=' * 60)
print(f'TOTAL: {total_solved}/{total_levels} levels solved ({total_solved/total_levels*100:.0f}%)')
print(f'Overall avg RHAE (solved): {total_rhae/max(1,total_solved):.1f}')
print('=' * 60)
