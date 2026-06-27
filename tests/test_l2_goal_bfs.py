"""Test BFS from rotation changer (49,10) to goal (54,50) on L2.
 
We need to know: how many steps does it actually take?
And what is the exact path?
"""
import sys, os, time, copy
from collections import deque

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20, _is_level_solved
from src.agent.oracle_adapters import get_oracle_adapter
from src.agent.game_solvers import (
    _detect_direction_mapping, _detect_game_step,
    _lightweight_bfs, _lightweight_bfs_replay_engine,
)

# Use OFFLINE mode
arc = arc_agi.Arcade(
    operation_mode=arc_agi.OperationMode.OFFLINE,
    environments_dir=os.path.join(project_root, "environment_files"),
)
env = arc.make("ls20")
env.reset()
obs = env.step(GameAction.ACTION1)
game = env._game

# Solve L0 and L1
for lvl in range(2):
    plan = solve_ls20(game, lvl)
    if plan:
        for aid, data in plan:
            game.perform_action(ActionInput(id=aid, data=data if data else {}))
    else:
        print(f"L{lvl} FAILED")
        sys.exit(1)

print(f"Reached L2 (level={game._current_level_index})")

# Now replay the 40 actions to reach the state at iter=6
# From debug output:
# L0: 14 steps, L1: 45 steps
# Then L2 solver does 40 collected actions before iter=6 state
# Actually, the solver modifies game directly...
# We need to play the 40 actions to reach (49,10), rotation=2, remaining=9

# But we don't know the exact 40 actions. Let's use solve_ls20 debug.
# Instead, let's run a special test: after solving L0, L1, 
# we replay some specific actions to reach iteration 6 state.

# Simplest approach: solve L2 up to getting stuck, then inspect.
# The game is already at L2 after L0+L1.
# Run the first 40 moves of the L2 solver by patching it.

# Actually the cleanest test: just check BFS from (49,10) to (54,50)
# using a fresh L2 state manually navigated.

# First, let's look at game walls around (49,10) and (54,50)
adapter = get_oracle_adapter("ls20", game)
print(f"\nL2 initial state:")
print(f"  Player: ({adapter.player.x}, {adapter.player.y}), remaining={adapter.steps_remaining}")

step_size = _detect_game_step(game)
dir_map = _detect_direction_mapping(game)
print(f"  step_size={step_size}, dir_map={dir_map}")

# Test BFS from current pos to (54,50) — this won't work since we're at (9,45)
# Let's try BFS with max_steps=100 to see if (54,50) is reachable at all
print(f"\nTesting BFS from (9,45) to (54,50) with max_steps=100...")
t0 = time.time()
path = _lightweight_bfs(game, 54, 50, max_steps=100, avoid_positions=set())
elapsed = time.time() - t0
if path:
    print(f"  Found path: {len(path)} steps in {elapsed:.2f}s")
else:
    print(f"  No path found in {elapsed:.2f}s")
    # Try ReplayEngine
    print(f"  Trying ReplayEngine BFS...")
    t0 = time.time()
    path = _lightweight_bfs_replay_engine(game, 54, 50, max_steps=100)
    elapsed = time.time() - t0
    if path:
        print(f"  ReplayEngine path: {len(path)} steps in {elapsed:.2f}s")
    else:
        print(f"  ReplayEngine: No path in {elapsed:.2f}s")

# Now manually navigate to (49,10) to test BFS from there
print(f"\nManually navigating to (49,10)...")
# Note: we need to do this on a COPY of the game
game_copy = copy.deepcopy(game)

# From (9,45) to (49,10):
# Need to go RIGHT (x: 9→49 = +40, +8 steps) and UP (y: 45→10 = -35, +7 steps = 15 steps)
# But we don't know the exact moves without BFS. Let's just do BFS.
path_to_rot = _lightweight_bfs_replay_engine(game_copy, 49, 10, max_steps=50)
if path_to_rot:
    print(f"  Path to (49,10): {len(path_to_rot)} steps")
    for aid, data in path_to_rot:
        game_copy.perform_action(ActionInput(id=aid, data=data if data else {}))
    
    adapter2 = get_oracle_adapter("ls20", game_copy)
    print(f"  After navigation: pos=({adapter2.player.x}, {adapter2.player.y}), remaining={adapter2.steps_remaining}")
    
    # Now test BFS from (49,10) to (54,50) 
    print(f"\nBFS from ({adapter2.player.x},{adapter2.player.y}) to (54,50):")
    for max_s in [9, 11, 15, 20, 30, 50]:
        t0 = time.time()
        p = _lightweight_bfs(game_copy, 54, 50, max_steps=max_s, avoid_positions=set())
        elapsed = time.time() - t0
        print(f"  max_steps={max_s}: {'None' if p is None else len(p)} steps ({elapsed:.2f}s)")
        if p is not None:
            print(f"  Path found with max_steps={max_s}!")
            # Also test with ReplayEngine
            t0 = time.time()
            p2 = _lightweight_bfs_replay_engine(game_copy, 54, 50, max_steps=50)
            elapsed = time.time() - t0
            print(f"  ReplayEngine path: {'None' if p2 is None else len(p2)} steps ({elapsed:.2f}s)")
            break
    else:
        print(f"  Not reachable within 50 steps! Trying ReplayEngine...")
        t0 = time.time()
        p2 = _lightweight_bfs_replay_engine(game_copy, 54, 50, max_steps=50)
        elapsed = time.time() - t0
        print(f"  ReplayEngine path: {'None' if p2 is None else len(p2)} steps ({elapsed:.2f}s)")
else:
    print(f"  Cannot reach (49,10) from start!")
