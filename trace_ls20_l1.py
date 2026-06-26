"""Detailed step-by-step trace of κ-gradient solver on ls20 L1."""
import sys
import os
import copy
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20, _solve_ls20_kappa_gradient_direct, _is_level_solved
from src.agent.oracle_adapters import get_oracle_adapter


def trace_solver_l1():
    """Step-by-step trace of solver behavior on L1."""
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    env.step(GameAction.ACTION1)

    # Advance to L1
    plan_l0 = solve_ls20(env._game, 0)
    ACTION_MAP = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}
    for aid, data in plan_l0:
        if aid in ACTION_MAP:
            env.step(ACTION_MAP[aid])

    print(f"Env at level: {env._game._current_level_index}")
    game_l1 = copy.deepcopy(env._game)

    # Now simulate the solver step-by-step
    adapter = get_oracle_adapter("ls20", game_l1)
    print(f"\n=== Initial L1 State ===")
    print(f"Player: ({adapter.player.x}, {adapter.player.y})")
    print(f"Player state: {adapter.player_state}")
    print(f"Goal: ({adapter.goals[0].x}, {adapter.goals[0].y})")
    print(f"Goal reqs: {adapter.goal_requirements}")
    print(f"State dim sizes: {adapter.state_dimension_sizes}")
    print(f"State changers: {adapter.state_changers}")
    print(f"Step size: {game_l1.gisrhqpee}")

    # Manually simulate the solver's first 30 steps
    step_size = game_l1.gisrhqpee
    sim = copy.deepcopy(game_l1)
    original_level = sim._current_level_index

    for step in range(40):
        adapter = get_oracle_adapter("ls20", sim)
        if adapter is None or adapter.player is None:
            print(f"  Step {step}: adapter broken")
            break

        px = int(adapter.player.x)
        py = int(adapter.player.y)
        ps = adapter.player_state
        sc = adapter.state_changers

        # Check if on a state changer
        on_changer = False
        changer_dim = None
        for dim, changers in sc.items():
            for ch in changers:
                if int(ch.x) == px and int(ch.y) == py:
                    on_changer = True
                    changer_dim = dim
                    break

        # Check if solved
        if _is_level_solved(sim, original_level):
            print(f"  Step {step}: SOLVED! Player at ({px},{py}) state={ps}")
            break

        # Print state
        state_str = f"rot={ps['rotation']} col={ps['color']} shp={ps['shape']}"
        changer_str = f"ON {changer_dim} changer!" if on_changer else ""
        print(f"  Step {step}: ({px},{py}) {state_str} {changer_str}")

        # Try all 4 directions
        best_dir = None
        best_score = -10000
        best_new_state = None
        for d in [1, 2, 3, 4]:
            test_sim = copy.deepcopy(sim)
            ai = ActionInput(id=d, data={})
            try:
                test_sim.perform_action(ai)
            except Exception:
                continue
            test_adapter = get_oracle_adapter("ls20", test_sim)
            if test_adapter and test_adapter.player:
                npx = int(test_adapter.player.x)
                npy = int(test_adapter.player.y)
                if (npx, npy) != (px, py):
                    new_ps = test_adapter.player_state
                    # Simple score: reduce state gradient + reduce distance to changer/goal
                    # Goal at (14,40), changer at (49,45)
                    goal_dist = abs(npx - 14) + abs(npy - 40)
                    changer_dist = abs(npx - 49) + abs(npy - 45)

                    # If state doesn't match, prioritize changer; if matches, prioritize goal
                    rot_triggers = (3 - new_ps['rotation']) % 4  # target=3
                    if rot_triggers > 0:
                        target_dist = changer_dist
                    else:
                        target_dist = goal_dist

                    score = -target_dist + (3 - rot_triggers) * 10  # higher = better
                    dir_name = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}[d]
                    if score > best_score:
                        best_score = score
                        best_dir = d
                        best_new_state = new_ps

        if best_dir is None:
            print(f"  Step {step}: STUCK - no valid direction")
            break

        dir_name = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}[best_dir]
        ai = ActionInput(id=best_dir, data={})
        sim.perform_action(ai)

    print("\n=== Final state ===")
    adapter = get_oracle_adapter("ls20", sim)
    if adapter and adapter.player:
        print(f"Player: ({adapter.player.x}, {adapter.player.y})")
        print(f"State: {adapter.player_state}")


if __name__ == "__main__":
    trace_solver_l1()
