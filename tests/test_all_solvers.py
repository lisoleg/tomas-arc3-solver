#!/usr/bin/env python
"""Comprehensive test script for all 22 game solvers.

Tests each game by:
1. Creating the environment (OFFLINE mode)
2. Getting the game object
3. Calling the solver
4. Executing the plan
5. Reporting results

Usage:
    python tests/test_all_solvers.py [--games tu93,su15] [--max-steps 300]
"""

import sys
import os
import time
import traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import arc_agi
from arcengine import GameAction

# Game solver
from src.agent.game_solvers import solve_game

# All 22 games (excluding ls20, tr87, ft09 which have dedicated adapters)
ALL_GAMES = [
    "tu93", "g50t", "wa30", "re86",  # keyboard_only
    "vc33", "s5i5", "tn36", "su15", "r11l", "lp85",  # click_only
    "bp35", "dc22", "sk48", "lf52", "sc25", "m0r0",  # keyboard+click
    "cn04", "cd82", "sp80", "ka59", "ar25", "sb26",  # keyboard+click
]

# Game baselines (step counts for RHAE calculation)
BASELINES = {
    "tu93": 50, "g50t": 50, "wa30": 50, "re86": 50,
    "vc33": 50, "s5i5": 50, "tn36": 50, "su15": 50,
    "r11l": 60, "lp85": 50, "bp35": 50, "dc22": 50,
    "sk48": 196, "lf52": 50, "sc25": 50, "m0r0": 50,
    "cn04": 50, "cd82": 100, "sp80": 50, "ka59": 50,
    "ar25": 50, "sb26": 50,
}


def test_game(game_id, max_steps=300):
    """Test a single game solver.

    Returns:
        dict with keys: game_id, status, levels_completed, total_levels,
                        steps_used, plan_length, error, rhae
    """
    result = {
        "game_id": game_id,
        "status": "unknown",
        "levels_completed": 0,
        "total_levels": 0,
        "steps_used": 0,
        "plan_length": 0,
        "error": None,
        "rhae": 0.0,
    }

    try:
        arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
        env = arc.make(game_id)

        # Reset to start
        obs = env.step(GameAction.RESET)
        result["total_levels"] = len(getattr(env._game, "_levels", []))

        # Get game object
        game = env._game

        # Try to solve level 0
        plan = solve_game(game, game_id, 0)

        if plan is None:
            result["status"] = "no_plan"
            result["error"] = "Solver returned None"
            return result

        if len(plan) == 0:
            result["status"] = "empty_plan"
            result["error"] = "Solver returned empty plan"
            return result

        result["plan_length"] = len(plan)

        # Execute the plan
        steps_used = 0
        for action, click_data in plan:
            if steps_used >= max_steps:
                result["status"] = "max_steps"
                result["error"] = f"Exceeded {max_steps} steps"
                break

            try:
                if click_data is not None:
                    # Convert tuple (x, y) to dict {'x': x, 'y': y}
                    if isinstance(click_data, (tuple, list)):
                        data_dict = {"x": int(click_data[0]), "y": int(click_data[1])}
                    elif isinstance(click_data, dict):
                        data_dict = click_data
                    else:
                        data_dict = {"x": 0, "y": 0}
                    obs = env.step(action, data=data_dict)
                else:
                    obs = env.step(action)
                steps_used += 1

                # Check if level completed
                if hasattr(obs, "levels_completed") and obs.levels_completed > 0:
                    result["levels_completed"] = obs.levels_completed
                    result["status"] = "level_passed"
                    break

                # Check if game over
                if hasattr(obs, "state") and obs.state == "game_over":
                    result["status"] = "game_over"
                    result["error"] = "Game over during plan execution"
                    break

            except Exception as e:
                result["status"] = "step_error"
                result["error"] = f"Step error: {e}"
                break

        result["steps_used"] = steps_used

        # Calculate RHAE if level passed
        if result["levels_completed"] > 0:
            baseline = BASELINES.get(game_id, 50)
            rhae = min(115.0, (baseline / max(1, steps_used)) ** 2 * 100)
            result["rhae"] = rhae
            result["status"] = "passed"

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test all game solvers")
    parser.add_argument("--games", type=str, default="", help="Comma-separated game IDs to test")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per game")
    args = parser.parse_args()

    if args.games:
        games = [g.strip() for g in args.games.split(",")]
    else:
        games = ALL_GAMES

    print(f"\n{'='*80}")
    print(f"Testing {len(games)} game solvers")
    print(f"{'='*80}\n")

    results = []
    passed = 0
    for game_id in games:
        print(f"  Testing {game_id}...", end=" ", flush=True)
        t0 = time.time()
        result = test_game(game_id, max_steps=args.max_steps)
        dt = time.time() - t0

        status_icon = {
            "passed": "PASS",
            "no_plan": "NOPLAN",
            "empty_plan": "EMPTY",
            "level_passed": "PASS",
            "max_steps": "MAXSTEPS",
            "game_over": "GAMEOVER",
            "step_error": "STEPERR",
            "error": "ERROR",
            "unknown": "???",
        }.get(result["status"], result["status"])

        if result["status"] in ("passed", "level_passed"):
            passed += 1
            print(f"PASS  levels={result['levels_completed']}/{result['total_levels']}  "
                  f"steps={result['steps_used']}  plan={result['plan_length']}  "
                  f"RHAE={result['rhae']:.1f}  ({dt:.1f}s)")
        else:
            err = result["error"] or ""
            print(f"{status_icon:8s} plan={result['plan_length']}  ({dt:.1f}s)  {err[:80]}")

        results.append(result)

    print(f"\n{'='*80}")
    print(f"Summary: {passed}/{len(games)} games passed")
    print(f"{'='*80}")

    # Print detailed results
    print(f"\n{'Game':8s} {'Status':10s} {'Levels':8s} {'Steps':6s} {'Plan':6s} {'RHAE':8s} Error")
    print("-" * 80)
    for r in results:
        print(f"{r['game_id']:8s} {r['status']:10s} "
              f"{r['levels_completed']}/{r['total_levels']:<5d} "
              f"{r['steps_used']:6d} {r['plan_length']:6d} "
              f"{r['rhae']:8.1f} {r['error'] or ''}")

    return passed


if __name__ == "__main__":
    n = main()
    sys.exit(0 if n > 0 else 1)
