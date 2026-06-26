"""Direct test of updated κ-PS BFS (v2) on ls20 L1."""
import sys, os, copy, time
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20, _solve_ls20_kappa_ps_bfs, _is_level_solved
from src.agent.oracle_adapters import get_oracle_adapter


def test_kappa_ps_bfs_l1():
    """Test only the κ-PS BFS (Stage 3) on L1."""
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

    # Check state
    adapter = get_oracle_adapter("ls20", game_l1)
    print(f"Player: ({adapter.player.x}, {adapter.player.y})")
    print(f"State: {adapter.player_state}")
    print(f"Goal: ({adapter.goals[0].x}, {adapter.goals[0].y})")
    print(f"Goal reqs: {adapter.goal_requirements}")
    print(f"State changers: {adapter.state_changers}")

    # Run ONLY κ-PS BFS (skip Stage 1 and 2)
    print("\n=== Running κ-PS BFS v2 (dynamic target) on L1 ===")
    t0 = time.time()
    result = _solve_ls20_kappa_ps_bfs(game_l1, 1)
    elapsed = time.time() - t0

    if result is not None:
        print(f"SUCCESS! Plan: {len(result)} actions (time={elapsed:.1f}s)")
        print(f"Plan: {result[:20]}{'...' if len(result) > 20 else ''}")

        # Verify on pristine copy
        verify_game = copy.deepcopy(env._game)
        for aid, data in result:
            ai = ActionInput(id=aid, data=data if data else {})
            try:
                verify_game.perform_action(ai)
            except Exception:
                print(f"Error at action {aid}")
                break
        if _is_level_solved(verify_game, verify_game._current_level_index):
            baseline = 123
            rhae = min(115, ((baseline / max(len(result), 1)) ** 2) * 100)
            print(f"VERIFIED! RHAE={rhae:.1f}")
        else:
            print(f"NOT verified — plan doesn't solve level")
    else:
        print(f"FAILED: None (time={elapsed:.1f}s)")


if __name__ == "__main__":
    test_kappa_ps_bfs_l1()
