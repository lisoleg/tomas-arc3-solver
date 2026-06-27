"""Benchmark: Run solve_ls20() from game_solvers.py on LS20 with per-level RHAE scoring.

This benchmark directly calls the solver function, bypassing TomasAgent.
Validates the Wall-Map BFS + 属性置换原语 + Ψ-Cut fixes.

RHAE formula per level: ((baseline / agent_steps) ** 2) * 100, capped at 115.
"""

import sys
import os
import time
import copy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, GameState, ActionInput
from src.agent.game_solvers import solve_ls20


def run_solver_benchmark(verbose: bool = True):
    """Run solve_ls20() directly on LS20 and report RHAE scores."""
    # v3.18.6: Use OFFLINE mode to avoid network dependency on three.arcprize.org
    arc = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=os.path.join(project_root, "environment_files"),
    )
    env = arc.make("ls20")

    # Get baselines
    baselines = None
    for e in arc.available_environments:
        if e.game_id.startswith("ls20"):
            baselines = e.baseline_actions
            break
    if baselines is None:
        baselines = [22, 123, 73, 84, 96, 192, 186]

    if verbose:
        print(f"LS20 baselines: {baselines}")
        print(f"Total baseline: {sum(baselines)}")
        print(f"Levels: {len(baselines)}")

    level_steps = []
    total_actions = 0
    start_time = time.time()

    for level_idx in range(len(baselines)):
        # Reset environment to start of this level
        env.reset()
        obs = env.step(GameAction.ACTION1)  # Initial step to activate level

        # Get game object
        game = copy.deepcopy(env._game)
        original_level = game._current_level_index

        # Try to advance to the target level
        # For level > 0, we need to solve previous levels first
        # Use the solver for each level sequentially
        pass  # We'll handle this below

    # Sequential approach: solve each level one at a time
    env.reset()
    obs = env.step(GameAction.ACTION1)

    for level_idx in range(len(baselines)):
        game = env._game
        original_level = game._current_level_index

        if verbose:
            print(f"\n  [LEVEL {level_idx}] Starting solve...")

        t0 = time.time()
        plan = solve_ls20(game, level_idx)
        solve_time = time.time() - t0

        if plan is not None and len(plan) > 0:
            # Execute the plan
            steps_this_level = 0
            for action_id, action_data in plan:
                ai = ActionInput(id=action_id, data=action_data if action_data else {})
                try:
                    game.perform_action(ai)
                    steps_this_level += 1
                    total_actions += 1
                except Exception as e:
                    if verbose:
                        print(f"    Action {action_id} failed: {e}")
                    break

                # Check if level solved
                if game._current_level_index > original_level:
                    break

            # Verify level completion
            if game._current_level_index > original_level:
                level_steps.append(steps_this_level)
                baseline = baselines[level_idx]
                rhae = min(115, ((baseline / max(steps_this_level, 1)) ** 2) * 100)
                if verbose:
                    print(f"  Level {level_idx}: {steps_this_level} steps "
                          f"(baseline={baseline}, RHAE={rhae:.1f}) "
                          f"[PASS] solve_time={solve_time:.1f}s")
            else:
                if verbose:
                    print(f"  Level {level_idx}: Plan executed but level NOT solved "
                          f"({steps_this_level} steps) [FAIL] solve_time={solve_time:.1f}s")
                break  # Can't proceed to next level
        else:
            if verbose:
                print(f"  Level {level_idx}: No plan found [FAIL] solve_time={solve_time:.1f}s")
            break  # Can't proceed to next level

    elapsed = time.time() - start_time

    # Report
    print("\n" + "=" * 70)
    print("LS20 SOLVER BENCHMARK RESULTS (solve_ls20 direct)")
    print("=" * 70)

    total_rhae = 0
    completed_levels = len(level_steps)
    total_levels = len(baselines)

    for i, steps in enumerate(level_steps):
        baseline = baselines[i]
        rhae = min(115, ((baseline / max(steps, 1)) ** 2) * 100)
        total_rhae += rhae
        status = "PASS" if rhae >= 80 else "FAIL"
        print(f"  Level {i}: {steps} steps (baseline={baseline}) -> RHAE={rhae:.1f} [{status}]")

    for i in range(completed_levels, total_levels):
        print(f"  Level {i}: NOT COMPLETED (baseline={baselines[i]}) -> RHAE=0.0 [FAIL]")

    avg_rhae = total_rhae / total_levels if total_levels > 0 else 0
    print(f"\n  Completed: {completed_levels}/{total_levels} levels")
    print(f"  Average RHAE: {avg_rhae:.1f}")
    print(f"  Total steps: {sum(level_steps)}")
    print(f"  Time: {elapsed:.1f}s")

    return {"avg_rhae": avg_rhae, "completed": completed_levels, "total": total_levels,
            "level_steps": level_steps, "baselines": baselines}


if __name__ == "__main__":
    print("TOMAS Solver LS20 Benchmark (solve_ls20 direct)")
    print("=" * 70)
    run_solver_benchmark(verbose=True)
