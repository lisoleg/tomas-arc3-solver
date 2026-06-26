"""Targeted benchmark for zero-score games: ka59, ar25, tn36.

Quick test to verify v3.14.0 specialized strategies are working.
Tests L0 only, 30s time limit per game.
"""
import sys
sys.path.insert(0, '.')

import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_game
from src.agent.game_profiles import ALL_GAME_BASELINES
import copy, time

arc = Arcade(operation_mode=OperationMode.OFFLINE)

ZERO_SCORE_GAMES = ['ka59', 'ar25', 'tn36']
baselines = ALL_GAME_BASELINES

print("=" * 60)
print("ZERO-SCORE GAME TARGETED BENCHMARK (v3.14.0)")
print("=" * 60)
print()

for gid in ZERO_SCORE_GAMES:
    bl_list = baselines.get(gid, [])
    if not bl_list:
        print(f'{gid}: NO BASELINE DATA')
        continue
    
    print(f"--- {gid} (baselines: {bl_list[:3]}) ---")
    
    for li in range(min(3, len(bl_list))):
        try:
            env = arc.make(gid)
            obs = env.step(GameAction.RESET)
            g2 = env._game
            
            if li > 0:
                g2._current_level_index = li
                if hasattr(g2, 'on_set_level'):
                    g2.on_set_level(g2._levels[li])
        except Exception as e:
            print(f'  L{li}: LOAD ERROR - {e}')
            continue
        
        baseline = bl_list[li]
        original_level = g2._current_level_index
        
        t0 = time.time()
        try:
            plan = solve_game(g2, gid, level_idx=li)
        except Exception as e:
            print(f'  L{li}: SOLVER ERROR - {e}')
            plan = None
        elapsed = time.time() - t0
        
        solved = False
        steps = len(plan) if plan else 999
        if plan:
            g3 = copy.deepcopy(g2)
            for i, step in enumerate(plan[:300]):
                aid, data = step
                ai_data = data if data else {}
                if isinstance(data, (tuple, list)):
                    ai_data = {"x": int(data[0]), "y": int(data[1])}
                ai = ActionInput(id=aid, data=ai_data)
                try:
                    g3.perform_action(ai)
                except Exception:
                    pass
                if hasattr(g3, '_current_level_index') and g3._current_level_index > original_level:
                    solved = True
                    steps = i + 1
                    break
        
        rhae = min(115.0, (baseline / max(1, steps))**2 * 100) if solved else 0
        status = "✅" if solved else "❌"
        print(f'  L{li}: {status} rhae={rhae:.1f} steps={steps}/{baseline} time={elapsed:.1f}s')
    
    print()

print("=" * 60)
print("DONE")
print("=" * 60)
