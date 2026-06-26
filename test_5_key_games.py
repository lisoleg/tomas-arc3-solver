"""Test 5 key games (s5i5, tn36, tu93, vc33, lp85) with the optimized pipeline.

Runs solve_game for each game's first level only, measures plan quality,
verification, and RHAE.

Expected outcomes (from task spec):
  - s5i5: ❌ → ≥30 RHAE (plan found)
  - tn36: ❌ → ≥20 RHAE (plan found)
  - tu93: 12.0 → ≥20 RHAE (shorter path)
  - vc33: 115.0 (maintained)
  - lp85: 115.0 (maintained)
"""
import sys
import os
import time
import copy
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import arc_agi
from arcengine import GameAction, ActionInput, GameState
from agent.game_solvers import solve_game

BASELINES = {
    "s5i5": [30, 50, 70, 90, 110, 130, 150],
    "tn36": [30, 50, 70, 90, 110, 130, 150],
    "tu93": [30, 50, 70, 90, 110, 130, 150],
    "vc33": [30, 50, 70, 90, 110, 130, 150],
    "lp85": [30, 50, 70, 90, 110, 130, 150],
}

KEY_GAMES = ["s5i5", "tn36", "tu93", "vc33", "lp85"]


def load_game(game_id: str, level: int = 0):
    """Load a game instance."""
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make(game_id)
    obs = env.step(GameAction.RESET)
    game = env._game

    if level > 0 and hasattr(game, '_levels') and level < len(game._levels):
        game._current_level_index = level
        if hasattr(game, 'on_set_level'):
            game.on_set_level(game._levels[level])

    return env, game


def calc_rhae(baseline: int, steps: int) -> float:
    """Calculate RHAE for a single level."""
    if steps <= 0:
        return 0.0
    return min(115.0, (baseline / max(1, steps)) ** 2 * 100)


def verify_plan(game, plan, original_level):
    """Verify a plan solves the level."""
    if not plan:
        return False, 0
    try:
        sim = copy.deepcopy(game)
        for i, (aid, data) in enumerate(plan[:300]):
            ai = ActionInput(id=aid, data=data if data else {})
            sim.perform_action(ai)
            if sim._state == GameState.WIN or sim._current_level_index > original_level:
                return True, i + 1
        return sim._state == GameState.WIN or sim._current_level_index > original_level, len(plan)
    except Exception:
        return False, 0


def main():
    print("=" * 80)
    print("5 Key Games Validation Test")
    print("=" * 80)

    for game_id in KEY_GAMES:
        baselines = BASELINES[game_id]
        baseline = baselines[0]

        print(f"\n--- {game_id} (baseline={baseline}) ---")
        try:
            env, game = load_game(game_id, 0)
            original_level = game._current_level_index

            t0 = time.time()
            plan = solve_game(game, game_id, 0)
            elapsed = time.time() - t0

            solved, actual_steps = verify_plan(game, plan, original_level)
            steps = actual_steps if solved else (len(plan) if plan else 0)
            rhae = calc_rhae(baseline, steps) if solved else 0.0

            print(f"  Plan: {len(plan) if plan else 0} actions")
            print(f"  Verified: {solved}")
            print(f"  Steps: {steps}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  RHAE: {rhae:.1f}")
            if plan:
                # Show first 10 actions
                for i, (aid, data) in enumerate(plan[:10]):
                    print(f"    [{i}] action={aid}, data={data}")
                if len(plan) > 10:
                    print(f"    ... ({len(plan) - 10} more actions)")

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == "__main__":
    main()
