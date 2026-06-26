"""Quick benchmark for Phase A/B optimization verification."""
import sys
sys.path.insert(0, '.')
import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_game, _is_level_solved
from src.agent.game_profiles import ALL_GAME_BASELINES
import copy, time

arc = Arcade(operation_mode=OperationMode.OFFLINE)

games = ['vc33', 'lp85', 's5i5', 'tu93', 'ls20']
baselines = ALL_GAME_BASELINES

results = {}
for gid in games:
    bl_list = baselines.get(gid, [])
    level_scores = []
    
    for li in range(2):
        env = arc.make(gid)
        obs = env.step(GameAction.RESET)
        g2 = env._game
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
            original_level = g3._current_level_index
            for step in plan[:200]:
                aid, data = step
                ai = ActionInput(id=aid, data=data if data else {})
                try:
                    g3.perform_action(ai)
                except:
                    pass
                if _is_level_solved(g3, original_level):
                    solved = True
                    break
        
        rhae = min(115.0, (baseline / max(1, steps))**2 * 100) if solved else 0
        level_scores.append((li, solved, steps, baseline, rhae, round(elapsed, 1)))
    
    results[gid] = level_scores
    print(f'{gid}: {level_scores}')

print()
print('--- Summary ---')
for gid, scores in results.items():
    solved_n = sum(1 for s in scores if s[1])
    total_rhae = sum(s[4] for s in scores if s[1])
    avg = total_rhae / max(1, solved_n)
    print(f'{gid}: {solved_n}/{len(scores)} solved, avg RHAE={avg:.1f}')
