"""Quick L2-only debug test — offline mode, no network needed."""
import sys, os, time, copy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20

# Use OFFLINE mode — no network needed
arc = arc_agi.Arcade(
    operation_mode=arc_agi.OperationMode.OFFLINE,
    environments_dir=os.path.join(project_root, "environment_files"),
)
env = arc.make("ls20")
baselines = [22, 123, 73, 84, 96, 192, 186]

# Solve L0 and L1 first, then test L2
env.reset()
obs = env.step(GameAction.ACTION1)

for level_idx in range(3):  # L0, L1, L2
    game = env._game
    original_level = game._current_level_index
    print(f"\n[LEVEL {level_idx}] Starting solve...")
    t0 = time.time()
    plan = solve_ls20(game, level_idx)
    solve_time = time.time() - t0

    if plan is not None and len(plan) > 0:
        steps_this_level = 0
        for action_id, action_data in plan:
            ai = ActionInput(id=action_id, data=action_data if action_data else {})
            try:
                game.perform_action(ai)
                steps_this_level += 1
            except Exception as e:
                print(f"  Action {action_id} failed: {e}")
                break

        rhae = min(115, int(((baselines[level_idx] / max(1, steps_this_level)) ** 2) * 100))
        status = "PASS" if game._current_level_index > original_level else "FAIL (not advanced)"
        print(f"  Steps={steps_this_level}, RHAE={rhae}, Time={solve_time:.1f}s, Status={status}")
    else:
        print(f"  FAILED: plan=None, Time={solve_time:.1f}s")

    if level_idx == 2:
        break  # Only care about L2
