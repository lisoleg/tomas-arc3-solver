"""Quick L0-only benchmark: tests each game at Level 0 only, 10s limit.
Used for rapid status assessment of 25-game baseline RHAE scores.
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "environment_files"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput, GameState
from agent.game_solvers import solve_game, SOLVERS
from agent.game_profiles import ALL_GAME_BASELINES

# v3.32.0: Games whose lambda closures break after deepcopy (okllwtboml dict).
# For these games, solver computes plans from game internals — verification
# must use Δ-State Replay (replay actions on original game) instead of deepcopy.
_DEEPCOPY_UNSAFE_GAMES = frozenset({"tn36"})

arc = Arcade(operation_mode=OperationMode.OFFLINE)

ALL_GAMES = [
    'ls20', 'vc33', 'tr87', 'tu93', 're86', 'g50t', 'wa30',
    'ft09', 's5i5', 'tn36', 'su15', 'r11l', 'lp85',
    'bp35', 'dc22', 'sk48', 'lf52', 'sc25', 'm0r0', 'cn04',
    'cd82', 'sp80', 'ka59', 'ar25', 'sb26',
]

baselines = ALL_GAME_BASELINES

print("=" * 70)
print("25-Game L0 Quick Benchmark (debug OFF)")
print("=" * 70)

results = {}
for gid in ALL_GAMES:
    bl_list = baselines.get(gid, [])
    if not bl_list:
        print(f'{gid}: NO BASELINE → skip')
        continue
    
    baseline = bl_list[0]  # L0 baseline only
    
    try:
        env = arc.make(gid)
        obs = env.step(GameAction.RESET)
        g2 = env._game
    except Exception as e:
        print(f'{gid}: LOAD_ERR: {e}')
        continue
    
    original_level = g2._current_level_index
    
    t0 = time.time()
    plan = solve_game(g2, gid, level_idx=0)
    elapsed = time.time() - t0
    
    solved = False
    steps = len(plan) if plan else 999
    if plan:
        # v3.32.0: Δ-State Replay verification for deepcopy-unsafe games
        if gid in _DEEPCOPY_UNSAFE_GAMES:
            # Replay on original game (solver was called on original too)
            # Save/restore level index to detect level transitions
            orig_level = g2._current_level_index
            for step in plan[:300]:
                aid, data = step
                ai = ActionInput(id=aid, data=data if data else {})
                try:
                    g2.perform_action(ai)
                except Exception:
                    pass
                for _ in range(5):
                    if hasattr(g2, '_current_level_index') and g2._current_level_index > orig_level:
                        break
                    if hasattr(g2, '_state') and g2._state == GameState.WIN:
                        break
                    try:
                        g2.complete_action()
                    except Exception:
                        break
                if hasattr(g2, '_current_level_index') and g2._current_level_index > orig_level:
                    solved = True
                    steps = plan.index(step) + 1
                    break
                if hasattr(g2, '_state') and g2._state == GameState.WIN:
                    solved = True
                    steps = plan.index(step) + 1
                    break
        else:
            # Deepcopy-safe games: verify on fresh deepcopy
            g3 = copy.deepcopy(g2)
            for step in plan[:300]:
                aid, data = step
                ai = ActionInput(id=aid, data=data if data else {})
                try:
                    g3.perform_action(ai)
                except Exception:
                    pass
                # CRITICAL: call complete_action() to settle animation frames
                for _ in range(5):
                    if hasattr(g3, '_current_level_index') and g3._current_level_index > original_level:
                        break
                    if hasattr(g3, '_state') and g3._state == GameState.WIN:
                        break
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
    phase = "Adapter" if SOLVERS.get(gid) else "Pipeline"
    status = "✅" if solved else "❌"
    
    results[gid] = {
        'solved': solved,
        'steps': steps,
        'baseline': baseline,
        'rhae': round(rhae, 1),
        'time': round(elapsed, 1),
        'phase': phase if solved else '?',
    }
    print(f'{gid:>6} L0: {status} rhae={rhae:.1f} steps={steps} bl={baseline} time={elapsed:.1f}s [{phase if solved else "fail"}]')

# Summary
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

passed_games = [gid for gid in ALL_GAMES if results.get(gid, {}).get('solved', False)]
failed_games = [gid for gid in ALL_GAMES if not results.get(gid, {}).get('solved', False)]
total_rhae = sum(results[gid]['rhae'] for gid in passed_games)

print(f'Passed: {len(passed_games)}/{len(results)} games')
print(f'Failed: {len(failed_games)} games: {failed_games}')
print(f'Total RHAE: {total_rhae:.1f}')
print(f'Avg RHAE (passed): {total_rhae/max(1,len(passed_games)):1f}')

print()
print("Per-Game RHAE Table")
print("-" * 50)
print(f'{"Game":<6} {"RHAE":>8} {"Steps":>6} {"BL":>6} {"Time":>6} {"Status":>4}')
print("-" * 50)
for gid in ALL_GAMES:
    r = results.get(gid, {})
    if r:
        s = "✅" if r['solved'] else "❌"
        print(f'{gid:<6} {r["rhae"]:>8.1f} {r["steps"]:>6} {r["baseline"]:>6} {r["time"]:>6.1f} {s:>4}')
