"""Direct test of the solve_ls20 solver function on LS20 levels.

Tests the game_solvers.py solve_ls20 function directly,
not the TomasAgent pipeline.
"""
import sys
import os
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, GameState, ActionInput
from src.agent.game_solvers import solve_ls20, _is_level_solved

LS20_BASELINES = [22, 123, 73, 84, 96, 192, 186]


def replay_plan(game, plan):
    """Replay an action plan on a game copy and check if level solved."""
    sim = copy.deepcopy(game)
    original_level = sim._current_level_index
    for aid, data in plan:
        ai = ActionInput(id=aid, data=data if data else {})
        try:
            sim.perform_action(ai)
        except Exception as e:
            print(f"    Replay error at action {aid}: {e}")
            return False
    return _is_level_solved(sim, original_level)


import copy


def test_solver_on_levels():
    """Test solve_ls20 on each LS20 level individually."""
    print("=" * 70)
    print("solve_ls20 Direct Solver Test")
    print("=" * 70)

    arc = arc_agi.Arcade()
    env = arc.make("ls20")

    # Start the game
    obs = env.step(GameAction.ACTION1)

    results = {}

    for level_idx in range(7):  # LS20 has 7 levels
        game = copy.deepcopy(env._game)
        original_level = game._current_level_index

        print(f"\nLevel {level_idx}: (baseline={LS20_BASELINES[level_idx]})")

        t0 = time.time()
        plan = solve_ls20(game, level_idx)
        elapsed = time.time() - t0

        if plan is None:
            print(f"  FAILED: plan=None (time={elapsed:.1f}s)")
            results[level_idx] = {"status": "FAIL", "plan_len": 0, "rhae": 0.0}
            continue

        plan_len = len(plan)
        print(f"  Plan length: {plan_len} (time={elapsed:.1f}s)")

        # Verify plan on pristine copy
        game_verify = copy.deepcopy(env._game)
        verified = replay_plan(game_verify, plan)

        if verified:
            rhae = min(115, ((LS20_BASELINES[level_idx] / max(plan_len, 1)) ** 2) * 100)
            status = "PASS" if rhae >= 80 else "FAIL"
            print(f"  VERIFIED: plan works! RHAE={rhae:.1f} [{status}]")
            results[level_idx] = {"status": status, "plan_len": plan_len, "rhae": rhae}
        else:
            print(f"  FAILED: plan does not solve level")
            results[level_idx] = {"status": "FAIL", "plan_len": plan_len, "rhae": 0.0}

        # Advance to next level for testing
        if level_idx < 6:
            # Need to solve current level in the env to reach next level
            try:
                env_game = env._game
                for aid, data in plan[:50]:
                    ai = ActionInput(id=aid, data=data if data else {})
                    env.step(ai)
                if env._game._current_level_index > original_level:
                    print(f"  Environment advanced to next level")
                else:
                    print(f"  Could not advance environment — breaking")
                    break
            except Exception as e:
                print(f"  Error advancing: {e}")
                break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_rhae = sum(r["rhae"] for r in results.values())
    avg_rhae = total_rhae / 7
    passed = sum(1 for r in results.values() if r["status"] == "PASS")

    for i, r in results.items():
        print(f"  Level {i}: {r['status']} | plan_len={r['plan_len']} | RHAE={r['rhae']:.1f}")

    print(f"\n  Passed: {passed}/7")
    print(f"  Average RHAE: {avg_rhae:.1f}")

    overall_pass = avg_rhae >= 80
    print(f"\n  IS_PASS: {'YES' if overall_pass else 'NO'}")

    return results


if __name__ == "__main__":
    test_solver_on_levels()
