"""Test: can we reach goal (54,50) from initial L2 position using ReplayEngine?"""
import sys, os, time, copy
from collections import deque

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20
from src.agent.oracle_adapters import get_oracle_adapter
from src.agent.game_solvers import _detect_direction_mapping, _detect_game_step

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
        print(f"L{lvl} FAILED"); sys.exit(1)

print(f"At L2 (level={game._current_level_index})")

# Get L2 initial state
adapter = get_oracle_adapter("ls20", game)
print(f"Start: pos=({adapter.player.x},{adapter.player.y}), remaining={adapter.steps_remaining}")
step_size = _detect_game_step(game)
dir_map = _detect_direction_mapping(game)
print(f"step_size={step_size}, dir_map={dir_map}")

# Test: raw ReplayEngine BFS from start to goal (54,50)
goal_x, goal_y = 54, 50
print(f"\nReplayEngine BFS from ({adapter.player.x},{adapter.player.y}) to goal ({goal_x},{goal_y})...")

# Manual ReplayEngine BFS (no time limit except max_steps)
root_game = copy.deepcopy(game)
root_adapter = get_oracle_adapter("ls20", root_game)
start_x = int(root_adapter.player.x)
start_y = int(root_adapter.player.y)

visited = {(start_x, start_y)}
queue = deque()
queue.append(([], start_x, start_y))

t0 = time.time()
max_steps = 30  # L2 goal should be reachable in 30 steps from start
found_path = None

while queue and (time.time() - t0) < 30.0:  # 30 sec timeout
    cur_actions, cur_px, cur_py = queue.popleft()
    if len(cur_actions) >= max_steps:
        continue
    if cur_px == goal_x and cur_py == goal_y:
        found_path = cur_actions
        break
    for action_id in [1, 2, 3, 4]:
        # Replay from root
        child = copy.deepcopy(root_game)
        for a_id, a_data in cur_actions:
            child.perform_action(ActionInput(id=a_id, data=a_data if a_data else {}))
        child.perform_action(ActionInput(id=action_id, data={}))
        child_adapter = get_oracle_adapter("ls20", child)
        if child_adapter is None or child_adapter.player is None:
            continue
        new_x = int(child_adapter.player.x)
        new_y = int(child_adapter.player.y)
        if (new_x, new_y) == (cur_px, cur_py):
            continue  # Blocked
        if (new_x, new_y) in visited:
            continue
        visited.add((new_x, new_y))
        queue.append((cur_actions + [(action_id, None)], new_x, new_y))

elapsed = time.time() - t0
if found_path:
    print(f"  FOUND! {len(found_path)} steps, {elapsed:.2f}s")
    print(f"  Path: {found_path}")
else:
    print(f"  NOT FOUND in {elapsed:.2f}s (timeout or not reachable)")
    print(f"  Visited {len(visited)} positions")
    # Check if (54,50) is even in the reachable set
    print(f"  Goal (54,50) in visited? {(54,50) in visited}")
    # Print some visited positions near goal
    near_goal = [(x,y) for x,y in visited if abs(x-54)<=10 and abs(y-50)<=10]
    print(f"  Visited near goal: {sorted(near_goal)[:20]}")
