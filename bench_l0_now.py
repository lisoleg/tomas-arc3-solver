"""Quick L0 benchmark for all 25 games - fast status assessment."""
import sys
sys.path.insert(0, '.')

import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput, GameState
from src.agent.game_solvers import solve_game, SOLVERS
from src.agent.game_profiles import ALL_GAME_BASELINES
import copy, time

arc = Arcade(operation_mode=OperationMode.OFFLINE)

ALL_GAMES = [
    'ls20', 'vc33', 'tr87', 'tu93', 're86', 'g50t', 'wa30',
    'ft09', 's5i5', 'tn36', 'su15', 'r11l', 'lp85',
    'bp35', 'dc22', 'sk48', 'lf52', 'sc25', 'm0r0', 'cn04',
    'cd82', 'sp80', 'ka59', 'ar25', 'sb26',
]

baselines = ALL_GAME_BASELINES
results = {}
total_rhae = 0.0
total_solved = 0

for gid in ALL_GAMES:
    bl_list = baselines.get(gid, [])
    if not bl_list:
        print(f'{gid}: NO BASELINE')
        continue

    baseline = bl_list[0]  # L0 only
    try:
        env = arc.make(gid)
        obs = env.step(GameAction.RESET)
        g2 = env._game
    except Exception as e:
        print(f'{gid} L0: LOAD ERROR {e}')
        continue

    original_level = g2._current_level_index
    t0 = time.time()
    plan = solve_game(g2, gid, level_idx=0)
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
            for _ in range(2):
                try:
                    g3.complete_action()
                except Exception:
                    break
            if hasattr(g3, '_current_level_index') and g3._current_level_index > original_level:
                solved = True
                steps = plan.index(step) + 1
                break
            if hasattr(g3, '_state') and g3._state == GameState.WIN:
                solved = True
                steps = plan.index(step) + 1
                break

    rhae = min(115.0, (baseline / max(1, steps))**2 * 100) if solved else 0
    total_rhae += rhae
    if solved:
        total_solved += 1
    phase = "Adapter" if SOLVERS.get(gid) else "Pipeline"
    status = "✅" if solved else "❌"
    results[gid] = (solved, steps, baseline, rhae, elapsed, phase)
    print(f'{gid} L0: {status} steps={steps} bl={baseline} RHAE={rhae:.1f} time={elapsed:.1f}s [{phase}]')

print()
print(f'=== L0 SUMMARY: {total_solved}/25 solved, Total RHAE={total_rhae:.1f}, Avg={total_rhae/25:.1f} ===')

# Categorize
failing = [gid for gid, r in results.items() if not r[0]]
low_rhae = [gid for gid, r in results.items() if r[0] and r[3] < 100]
good = [gid for gid, r in results.items() if r[0] and r[3] >= 100]
print(f'Failing (RHAE=0): {failing}')
print(f'Low RHAE (<100): {low_rhae}')
print(f'Good (RHAE>=100): {good}')
