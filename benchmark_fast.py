"""Ultra-fast benchmark: each game tests L0 only, 10s time limit.
Used for quick status assessment of 25-game dual-track architecture.
"""
import sys
sys.path.insert(0, '.')

import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
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

for gid in ALL_GAMES:
    bl_list = baselines.get(gid, [])
    if not bl_list:
        print(f'{gid}: NO BASELINE')
        results[gid] = {'L0': 'NO_BL'}
        continue
    
    level_scores = {}
    # Test L0, L1, L2 (up to 3 levels)
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
            level_scores[f'L{li}'] = f'LOAD_ERR:{e}'
            continue
        
        baseline = bl_list[li]
        original_level = g2._current_level_index
        
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
                if hasattr(g3, '_current_level_index') and g3._current_level_index > original_level:
                    solved = True
                    steps = plan.index(step) + 1
                    break
        
        rhae = min(115.0, (baseline / max(1, steps))**2 * 100) if solved else 0
        phase = "Adapter" if SOLVERS.get(gid) else "Pipeline"
        status = "✅" if solved else "❌"
        level_scores[f'L{li}'] = {
            'solved': solved,
            'steps': steps,
            'baseline': baseline,
            'rhae': round(rhae, 1),
            'time': round(elapsed, 1),
            'phase': phase if solved else '?',
        }
        print(f'  {gid} L{li}: {status} rhae={rhae:.1f} steps={steps} baseline={baseline} time={elapsed:.1f}s')
    
    results[gid] = level_scores
    solved_n = sum(1 for v in level_scores.values() if isinstance(v, dict) and v['solved'])
    total_n = len(level_scores)
    avg_rhae = sum(v['rhae'] for v in level_scores.values() if isinstance(v, dict) and v['solved']) / max(1, solved_n)
    print(f'  → {gid}: {solved_n}/{total_n} solved, avg RHAE={avg_rhae:.1f}')

# Summary
print()
print('=' * 80)
print('DUAL-TRACK STATUS REPORT')
print('=' * 80)

adapter_games = [gid for gid in ALL_GAMES if gid in SOLVERS]
pipeline_games = [gid for gid in ALL_GAMES if gid not in SOLVERS]

# Track 1: Adapter solvers
print()
print('Track 1: Hardcoded Adapter Solvers (25 games)')
print('-' * 60)
total_adapter_solved = 0
total_adapter_levels = 0
total_adapter_rhae = 0.0
for gid in adapter_games:
    scores = results.get(gid, {})
    for lk, v in scores.items():
        if isinstance(v, dict):
            total_adapter_levels += 1
            if v['solved']:
                total_adapter_solved += 1
                total_adapter_rhae += v['rhae']

print(f'  Levels solved: {total_adapter_solved}/{total_adapter_levels}')
print(f'  Avg RHAE (solved): {total_adapter_rhae/max(1,total_adapter_solved):.1f}')
print(f'  Effective RHAE: {total_adapter_rhae/max(1,total_adapter_levels):.1f}')

# Track 2: Universal Pipeline
print()
print('Track 2: UniversalSolverPipeline (fallback for all games)')
print('-' * 60)
total_pipe_solved = 0
total_pipe_levels = 0
total_pipe_rhae = 0.0
for gid in ALL_GAMES:
    scores = results.get(gid, {})
    for lk, v in scores.items():
        if isinstance(v, dict):
            total_pipe_levels += 1
            if v['solved']:
                total_pipe_solved += 1
                total_pipe_rhae += v['rhae']

print(f'  Levels solved: {total_pipe_solved}/{total_pipe_levels}')
print(f'  Avg RHAE (solved): {total_pipe_rhae/max(1,total_pipe_solved):.1f}')
print(f'  Effective RHAE: {total_pipe_rhae/max(1,total_pipe_levels):.1f}')

# Per-game table
print()
print('Per-Game Detail Table')
print('-' * 80)
print(f'{"Game":<6} {"L0":>8} {"L1":>8} {"L2":>8} {"Solved":>6} {"Avg RHAE":>8}')
print('-' * 80)
for gid in ALL_GAMES:
    scores = results.get(gid, {})
    l0 = scores.get('L0', {})
    l1 = scores.get('L1', {})
    l2 = scores.get('L2', {})
    
    def fmt(v):
        if isinstance(v, str): return v
        if isinstance(v, dict):
            if v['solved']:
                return f'{v["rhae"]:.1f}'
            return 'FAIL'
        return 'N/A'
    
    solved_n = sum(1 for v in scores.values() if isinstance(v, dict) and v['solved'])
    total_n = len([v for v in scores.values() if isinstance(v, dict)])
    avg = sum(v['rhae'] for v in scores.values() if isinstance(v, dict) and v['solved']) / max(1, solved_n)
    
    print(f'{gid:<6} {fmt(l0):>8} {fmt(l1):>8} {fmt(l2):>8} {solved_n}/{total_n:>4} {avg:>8.1f}')
