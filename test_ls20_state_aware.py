"""State-aware κ-gradient ls20 solver test — L0/L1/L2 benchmark.

Properly handles level transitions via GameAction replay,
then tests solve_ls20 on each level's game state.
"""
import sys
import os
import time
import copy

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, GameState, ActionInput
from src.agent.game_solvers import solve_ls20, _is_level_solved

LS20_BASELINES = [22, 123, 73, 84, 96, 192, 186]


def replay_plan_on_env(env, plan, max_actions=50):
    """Replay a plan of (action_id, data) tuples on the environment.

    Converts action_id to GameAction for env.step().
    Returns True if the env advanced to the next level.
    """
    ACTION_MAP = {
        1: GameAction.ACTION1,  # UP
        2: GameAction.ACTION2,  # DOWN
        3: GameAction.ACTION3,  # LEFT
        4: GameAction.ACTION4,  # RIGHT
    }

    for i, (aid, data) in enumerate(plan[:max_actions]):
        if aid in ACTION_MAP:
            env.step(ACTION_MAP[aid])
        elif aid == 6:
            # ACTION6 clicks — skip (invalid in ls20 anyway)
            continue
        else:
            # Unknown action — try as GameAction
            try:
                env.step(GameAction.ACTION1)
            except Exception:
                pass

    return env._game._current_level_index


def verify_plan(game, plan):
    """Verify a plan on a pristine game copy."""
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


def test_ls20_levels():
    """Test solve_ls20 on L0, L1, L2 with proper level transitions."""
    print("=" * 70)
    print("ls20 State-Aware κ-gradient Solver — L0/L1/L2 Benchmark")
    print("=" * 70)

    arc = arc_agi.Arcade()
    env = arc.make("ls20")

    # Initialize game
    obs = env.step(GameAction.ACTION1)

    results = {}

    for target_level in range(3):  # L0, L1, L2
        print(f"\n{'─' * 70}")
        print(f"Level {target_level}: (baseline={LS20_BASELINES[target_level]})")

        # Get current game state
        game = copy.deepcopy(env._game)
        current_level = game._current_level_index
        print(f"  Current env level: {current_level}")

        if current_level != target_level:
            print(f"  ERROR: env at level {current_level}, expected {target_level}")
            break

        # Solve
        t0 = time.time()
        plan = solve_ls20(game, target_level)
        elapsed = time.time() - t0

        if plan is None:
            print(f"  FAILED: plan=None (time={elapsed:.1f}s)")
            results[target_level] = {"status": "FAIL", "plan_len": 0, "rhae": 0.0}
            break  # Can't advance to next level

        plan_len = len(plan)
        print(f"  Plan: {plan_len} actions (time={elapsed:.1f}s)")

        # Verify plan on pristine copy
        verified = verify_plan(copy.deepcopy(env._game), plan)

        if verified:
            rhae = min(115, ((LS20_BASELINES[target_level] / max(plan_len, 1)) ** 2) * 100)
            status = "PASS" if rhae >= 80 else "FAIL"
            print(f"  VERIFIED ✓ RHAE={rhae:.1f} [{status}]")
            results[target_level] = {"status": status, "plan_len": plan_len, "rhae": rhae}

            # Advance env to next level using the plan
            if target_level < 2:
                new_level = replay_plan_on_env(env, plan)
                if new_level > target_level:
                    print(f"  Env advanced to level {new_level}")
                else:
                    print(f"  Env failed to advance — stopping")
                    break
        else:
            print(f"  FAILED: plan does not solve level {target_level}")
            results[target_level] = {"status": "FAIL", "plan_len": plan_len, "rhae": 0.0}
            break

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    total_rhae = sum(r["rhae"] for r in results.values())
    n_tested = len(results)
    avg_rhae = total_rhae / max(1, n_tested)
    passed = sum(1 for r in results.values() if r["status"] == "PASS")

    for i, r in results.items():
        print(f"  Level {i}: {r['status']} | plan_len={r['plan_len']} | RHAE={r['rhae']:.1f}")

    print(f"\n  Passed: {passed}/{n_tested}")
    print(f"  Average RHAE: {avg_rhae:.1f}")

    overall_pass = passed >= 3 and avg_rhae >= 80
    print(f"\n  IS_PASS: {'YES' if overall_pass else 'NO'}")

    return results


if __name__ == "__main__":
    test_ls20_levels()
