"""Replay L2 baseline solution to understand how it reaches goal (54,50)."""
import sys, os, time, copy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20
from src.agent.oracle_adapters import get_oracle_adapter

# Use OFFLINE mode
arc = arc_agi.Arcade(
    operation_mode=arc_agi.OperationMode.OFFLINE,
    environments_dir=os.path.join(project_root, "environment_files"),
)

# Get baseline for L2
baselines = None
for e in arc.available_environments:
    if "ls20" in e.game_id:
        baselines = e.baseline_actions
        print(f"Baselines: {baselines}")
        break

if baselines is None:
    baselines = [22, 123, 73, 84, 96, 192, 186]
    print(f"Using default baselines: {baselines}")

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

print(f"\nAt L2 (level={game._current_level_index})")
adapter = get_oracle_adapter("ls20", game)
print(f"Start: ({adapter.player.x},{adapter.player.y}), remaining={adapter.steps_remaining}")
print(f"State: {adapter.player_state}")
print(f"Goal reqs: {adapter.goal_requirements}")
print(f"Goals: {[(int(g.x),int(g.y)) for g in adapter.goals]}")
print(f"Changers: {adapter.state_changers}")
print(f"Coins: {[(int(c.x),int(c.y)) for c in adapter.coins] if adapter.coins else []}")

# Check if there's a baseline for L2
# baseline_actions usually contains a sequence for each level
# The game might provide access to baseline
try:
    baseline_l2 = env._game_info.baseline_actions[2] if hasattr(env, '_game_info') else None
    print(f"\nBaseline L2 actions: {baseline_l2}")
except Exception as e:
    print(f"No baseline access: {e}")

# Try to get baseline from arc
try:
    env_infos = arc.available_environments
    for ei in env_infos:
        if "ls20" in ei.game_id:
            print(f"\nGame info: {ei}")
            break
except Exception as e:
    print(f"Error: {e}")

# Let's try to manually step through L2 and trace what happens
# when we move from (54,45) → DOWN → to see if we reach (54,50)
print(f"\n=== Testing movement from (54,45) to see if (54,50) reachable ===")
game_copy = copy.deepcopy(game)
adapter_c = get_oracle_adapter("ls20", game_copy)
print(f"Current pos: ({adapter_c.player.x},{adapter_c.player.y})")

# Navigate to (54,45) first using the solver path through iter=2 
# We know from debug that iter=1 gets us to (19,30) after 16 actions
# Let's manually try a partial plan
# Actually, let's use the solver's first 39 steps (actions before iter=6)
# to get the game to state before STATE MATCH

# Run solver to get the 40 collected actions  
plan = solve_ls20(game_copy, 2)
print(f"\nSolver result: {plan}")
# The solver returns None, so we can't get the plan easily
# But we know the game state got modified (solver modifies in-place)
adapter_after = get_oracle_adapter("ls20", game_copy)
print(f"After solver attempt: pos=({adapter_after.player.x},{adapter_after.player.y}), remaining={adapter_after.steps_remaining}")
print(f"State: {adapter_after.player_state}")

# At this point game_copy is at (49,10), rotation=2, color=1, shape=5, remaining=9
# Let's test what happens when we move from here
for action_id in [1, 2, 3, 4]:
    test_game = copy.deepcopy(game_copy)
    test_game.perform_action(ActionInput(id=action_id, data={}))
    test_adapter = get_oracle_adapter("ls20", test_game)
    if test_adapter and test_adapter.player:
        print(f"  Action {action_id} → ({test_adapter.player.x},{test_adapter.player.y}), remaining={test_adapter.steps_remaining}, solved={test_adapter.steps_remaining}")
    # Check if level is solved
    level_after = test_game._current_level_index
    print(f"    level_index after action {action_id}: {level_after}")
