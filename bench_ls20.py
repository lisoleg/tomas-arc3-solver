"""LS20 full benchmark — test all 7 levels sequentially.
Uses deep-copy of game state for each level test.
Advances through levels by replaying solved plans on the live env.
"""
import sys, os, time, copy

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, GameState, ActionInput
from agent.game_solvers import solve_ls20, _is_level_solved

LS20_BASELINES = [22, 123, 73, 84, 96, 192, 186]

# GameAction mapping for env.step() — Arcade wrapper uses GameAction enum
GA_MAP = {
    0: GameAction.RESET,
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
    6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}


def main():
    print("=" * 70)
    print("LS20 Full Benchmark (solve_ls20)")
    print("=" * 70)

    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    obs = env.step(GameAction.ACTION1)  # Start game

    results = {}
    for level_idx in range(7):
        game = copy.deepcopy(env._game)
        original_level = game._current_level_index

        print(f"\n--- Level {level_idx} (baseline={LS20_BASELINES[level_idx]}) ---")

        t0 = time.time()
        plan = solve_ls20(game, level_idx)
        elapsed = time.time() - t0

        if plan is None:
            print(f"  FAIL: plan=None ({elapsed:.1f}s)")
            results[level_idx] = {"status": "FAIL", "plan_len": 0, "rhae": 0.0}
            print("  Cannot advance env — stopping benchmark")
            break

        # Verify plan on fresh copy
        game_verify = copy.deepcopy(env._game)
        orig_lv = game_verify._current_level_index
        for aid, data in plan:
            ai = ActionInput(id=aid, data=data if data else {})
            try:
                game_verify.perform_action(ai)
            except Exception as e:
                print(f"  Replay error: {e}")
                break

        solved = _is_level_solved(game_verify, orig_lv)
        plan_len = len(plan)

        if solved:
            rhae = min(115, ((LS20_BASELINES[level_idx] / max(plan_len, 1)) ** 2) * 100)
            status = "PASS" if rhae >= 80 else "LOW-RHAE"
            print(f"  {status}: plan_len={plan_len} RHAE={rhae:.1f} ({elapsed:.1f}s)")
            results[level_idx] = {"status": status, "plan_len": plan_len, "rhae": rhae}
        else:
            print(f"  FAIL: plan_len={plan_len} verified=False ({elapsed:.1f}s)")
            results[level_idx] = {"status": "FAIL", "plan_len": plan_len, "rhae": 0.0}

        # Advance env to next level by replaying solved plan
        if level_idx < 6 and solved:
            for aid, data in plan:
                ga = GA_MAP.get(aid, GameAction.ACTION1)
                obs = env.step(ga)
                if obs.state == GameState.WIN:
                    print(f"  Env: L{env._game._current_level_index} SOLVED!")
                    break
                if obs.state not in (GameState.NOT_FINISHED,):
                    print(f"  Env state: {obs.state}")
                    break
            if env._game._current_level_index <= original_level:
                print(f"  Env stuck at L{env._game._current_level_index}")
                break
        elif not solved and level_idx < 6:
            # Can't advance — need to force advance or break
            print(f"  Cannot advance env — stopping at L{level_idx}")
            break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, r in results.items():
        print(f"  L{i}: {r['status']} | plan={r['plan_len']} | RHAE={r['rhae']:.1f}")
    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    avg_rhae = sum(r["rhae"] for r in results.values()) / max(len(results), 1)
    print(f"\n  Passed: {passed}/7 | Avg RHAE: {avg_rhae:.1f}")
    overall = "YES" if passed >= 7 else "NO"
    print(f"  ALL_PASS: {overall}")


if __name__ == "__main__":
    main()
