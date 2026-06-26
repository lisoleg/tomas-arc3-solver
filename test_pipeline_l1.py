"""Test if UniversalSolverPipeline (with ActionInput fix) solves s5i5/lp85 L1."""
import sys
sys.path.insert(0, '.')
import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import _is_level_solved
from src.agent.universal_solver_pipeline import UniversalSolverPipeline
import copy, time

arc = Arcade(operation_mode=OperationMode.OFFLINE)

def _normalize(plan):
    if not plan:
        return None
    out = []
    for action, data in plan:
        if isinstance(data, (tuple, list)):
            out.append((action, {"x": int(data[0]), "y": int(data[1])}))
        else:
            out.append((action, data if data else {}))
    return out

def _verify(plan, pristine, orig_level):
    if not plan:
        return False
    sim = copy.deepcopy(pristine)
    for aid, data in plan[:300]:
        ai = ActionInput(id=aid, data=data)
        try:
            sim.perform_action(ai)
        except:
            pass
        if _is_level_solved(sim, orig_level):
            return True
    return _is_level_solved(sim, orig_level)

# Test s5i5 L1 via Pipeline
print("Testing UniversalSolverPipeline for s5i5 L1...")
env = arc.make('s5i5')
obs = env.step(GameAction.RESET)
g = env._game
g._current_level_index = 1
if hasattr(g, 'on_set_level'):
    g.on_set_level(g._levels[1])

t0 = time.time()
pipeline = UniversalSolverPipeline(g, 's5i5', max_time=30.0)
plan = pipeline.solve()
elapsed = time.time() - t0
steps = len(plan) if plan else 0
plan_n = _normalize(plan)
verified = _verify(plan_n, copy.deepcopy(g), g._current_level_index) if plan_n else False
print(f"  s5i5 L1 Pipeline: {steps} steps, {elapsed:.1f}s, verified={verified}")

# Test lp85 L1 via Pipeline
print("\nTesting UniversalSolverPipeline for lp85 L1...")
env = arc.make('lp85')
obs = env.step(GameAction.RESET)
g = env._game
g._current_level_index = 1
if hasattr(g, 'on_set_level'):
    g.on_set_level(g._levels[1])

t0 = time.time()
pipeline = UniversalSolverPipeline(g, 'lp85', max_time=30.0)
plan = pipeline.solve()
elapsed = time.time() - t0
steps = len(plan) if plan else 0
plan_n = _normalize(plan)
verified = _verify(plan_n, copy.deepcopy(g), g._current_level_index) if plan_n else False
print(f"  lp85 L1 Pipeline: {steps} steps, {elapsed:.1f}s, verified={verified}")
