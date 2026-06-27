"""Quick L2 debug test — load game locally, no network needed."""
import sys, os, time, copy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files", "ls20", "9607627b"))

from ls20 import Ls20
from arcengine import ActionInput
from src.agent.game_solvers import solve_ls20

# Create game and advance to L2
game = Ls20()
game._choose_level(0)

# Solve L0
print("[L0] Starting solve...")
t0 = time.time()
plan = solve_ls20(game, 0)
elapsed = time.time() - t0
print(f"  L0 plan: {len(plan) if plan else 'None'} steps, {elapsed:.1f}s")
if plan:
    for aid, data in plan:
        game.perform_action(ActionInput(id=aid, data=data if data else {}))
    print(f"  L0 level now: {game._current_level_index}")
else:
    print("  L0 FAILED — cannot reach L2")
    sys.exit(1)

# Solve L1
print("[L1] Starting solve...")
t0 = time.time()
plan = solve_ls20(game, 1)
elapsed = time.time() - t0
print(f"  L1 plan: {len(plan) if plan else 'None'} steps, {elapsed:.1f}s")
if plan:
    for aid, data in plan:
        game.perform_action(ActionInput(id=aid, data=data if data else {}))
    print(f"  L1 level now: {game._current_level_index}")
else:
    print("  L1 FAILED — cannot reach L2")
    sys.exit(1)

# Solve L2
print("\n[L2] Starting solve...")
t0 = time.time()
plan = solve_ls20(game, 2)
elapsed = time.time() - t0
if plan is not None and len(plan) > 0:
    steps = 0
    for aid, data in plan:
        game.perform_action(ActionInput(id=aid, data=data if data else {}))
        steps += 1
    level_now = game._current_level_index
    status = "PASS" if level_now > 2 else "FAIL"
    print(f"  L2: {steps} steps, {elapsed:.1f}s, level_now={level_now}, status={status}")
else:
    print(f"  L2 FAILED: plan=None, {elapsed:.1f}s")
