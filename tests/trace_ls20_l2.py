"""L2 Solver Trace: Step-by-step execution trace of _solve_ls20_delta_state_bfs on Level 2.

Injects logging into the solver to see exactly which branch it enters,
how many steps it executes before failing, and whether it collects coins.
"""

import sys
import os
import time
import copy
import traceback

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, GameState, ActionInput


def main():
    print("LS20 L2 Solver Step-by-Step Trace")
    print("=" * 70)

    arc = arc_agi.Arcade()
    env = arc.make("ls20")

    # Advance to L2
    env.reset()
    obs = env.step(GameAction.ACTION1)
    game = env._game

    from src.agent.game_solvers import solve_ls20

    # Solve L0
    plan0 = solve_ls20(game, 0)
    if plan0:
        for aid, data in plan0:
            game.perform_action(ActionInput(id=aid, data=data if data else {}))
    else:
        print("L0 failed")
        return

    # Solve L1
    plan1 = solve_ls20(game, 1)
    if plan1:
        for aid, data in plan1:
            game.perform_action(ActionInput(id=aid, data=data if data else {}))
    else:
        print("L1 failed")
        return

    print(f"Current level: {game._current_level_index}")

    # Now trace L2 solver manually (simulating the main loop)
    from src.agent.game_solvers import (
        _detect_direction_mapping, _detect_game_step,
        _is_level_solved, _compute_coin_proximity_position,
    )
    from src.agent.oracle_adapters import get_oracle_adapter

    sim = copy.deepcopy(game)
    original_level = sim._current_level_index
    t0 = time.time()
    MAX_TOTAL_TIME = 45.0
    MIN_RESERVE = 8

    step_size = _detect_game_step(sim)
    dir_offsets_cached = _detect_direction_mapping(game)
    collected = []

    print(f"\nInitial state:")
    adapter = get_oracle_adapter("ls20", sim)
    if adapter and adapter.player:
        print(f"  Player: ({int(adapter.player.x)}, {int(adapter.player.y)})")
        print(f"  Remaining: {adapter.steps_remaining}")
        print(f"  State: {adapter.player_state}")
        print(f"  Dir offsets: {dir_offsets_cached}")
        print(f"  Step size: {step_size}")

    for iteration in range(30):
        elapsed = time.time() - t0
        if elapsed > MAX_TOTAL_TIME:
            print(f"\n⏰ TIME OUT at iteration {iteration}, {elapsed:.1f}s")
            break

        if _is_level_solved(sim, original_level):
            print(f"\n✅ LEVEL SOLVED at iteration {iteration}, {len(collected)} steps")
            break

        if len(collected) >= 200:
            print(f"\n❌ MAX STEPS reached at iteration {iteration}")
            break

        adapter = get_oracle_adapter("ls20", sim)
        if adapter is None or adapter.player is None:
            print(f"\n❌ No adapter/player at iteration {iteration}")
            break

        px = int(adapter.player.x)
        py = int(adapter.player.y)
        player_state = adapter.player_state
        goal_reqs = adapter.goal_requirements
        dim_sizes = adapter.state_dimension_sizes
        state_changers = adapter.state_changers
        goals = adapter.goals
        actions_remaining = adapter.steps_remaining
        coins = adapter.coins

        print(f"\n  [ITER {iteration}] pos=({px},{py}), remaining={actions_remaining}, "
              f"state={player_state}, collected_so_far={len(collected)}")

        if not goal_reqs or not goals:
            print(f"  ❌ No goal_reqs or goals")
            break

        # Step budget check
        if actions_remaining < MIN_RESERVE and coins:
            print(f"  ⚠️ Remaining={actions_remaining} < MIN_RESERVE={MIN_RESERVE}, "
                  f"trying coin collection...")
            # We'll handle this below

        # Compute mismatches
        best_goal_idx = 0
        best_mismatch = 999
        for i, req in enumerate(goal_reqs):
            mismatch = (
                (req['rotation'] != player_state['rotation'])
                + (req['color'] != player_state['color'])
                + (req['shape'] != player_state['shape'])
            )
            if mismatch < best_mismatch:
                best_mismatch = mismatch
                best_goal_idx = i

        best_goal_req = goal_reqs[best_goal_idx]
        best_goal = goals[best_goal_idx]
        print(f"  Best goal: idx={best_goal_idx}, mismatch={best_mismatch}, "
              f"req={best_goal_req}, pos=({int(best_goal.x)}, {int(best_goal.y)})")

        if best_mismatch == 0:
            print(f"  ✅ State MATCHED — navigating to goal")
            # This would BFS to goal
        else:
            # Compute triggers needed per dimension
            mismatches = []
            for dim_name in ['rotation', 'color', 'shape']:
                current_val = player_state.get(dim_name, 0)
                target_val = best_goal_req.get(dim_name, 0)
                dim_size = dim_sizes.get(dim_name, 4)
                triggers_needed = (target_val - current_val) % dim_size
                if triggers_needed > 0:
                    mismatches.append((dim_name, triggers_needed, dim_size))
                    print(f"  {dim_name}: current={current_val}, target={target_val}, "
                          f"triggers={triggers_needed}, dim_size={dim_size}")

            mismatches.sort(key=lambda x: -x[1])

            # Choose target changer
            target_changer_x, target_changer_y = px, py
            target_changer_dim = mismatches[0][0] if mismatches else None

            for dim_name, _, _ in mismatches:
                changers = state_changers.get(dim_name, [])
                if changers:
                    nearest_ch = min(changers,
                        key=lambda c: abs(px - int(c.x)) + abs(py - int(c.y)))
                    target_changer_x = int(nearest_ch.x)
                    target_changer_y = int(nearest_ch.y)
                    target_changer_dim = dim_name
                    break

            triggers_needed = mismatches[0][1] if mismatches else 0
            print(f"  Target changer: {target_changer_dim} at ({target_changer_x},{target_changer_y}), "
                  f"triggers_needed={triggers_needed}")

            # Calculate BFS path length to changer
            from collections import deque
            wall_positions = set()
            for w in (adapter.walls or []):
                wall_positions.add((int(w.x), int(w.y)))

            other_changer_positions = set()
            for dim, chs in state_changers.items():
                for ch in chs:
                    ch_pos = (int(ch.x), int(ch.y))
                    if ch_pos != (target_changer_x, target_changer_y):
                        other_changer_positions.add(ch_pos)

            blocked = set(wall_positions)
            blocked.update(other_changer_positions)
            blocked.discard((target_changer_x, target_changer_y))

            visited_bfs = {(px, py)}
            visited_bfs.update(blocked)
            queue_bfs = deque([([], px, py)])
            path_to_changer = None

            while queue_bfs:
                ca, cx, cy = queue_bfs.popleft()
                if len(ca) >= 200:
                    continue
                if cx == target_changer_x and cy == target_changer_y:
                    path_to_changer = ca
                    break
                for d, (dx, dy) in dir_offsets_cached.items():
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in visited_bfs:
                        continue
                    visited_bfs.add((nx, ny))
                    queue_bfs.append((ca + [(d, None)], nx, ny))

            if path_to_changer:
                print(f"  BFS to changer: {len(path_to_changer)} steps, "
                      f"remaining={actions_remaining}, "
                      f"budget_exceed={len(path_to_changer) > actions_remaining}")
            else:
                print(f"  BFS to changer: NO PATH FOUND")

            # Also calculate BFS path to nearest coin
            if coins:
                for c in coins:
                    prox_pos = _compute_coin_proximity_position(
                        px, py, step_size, int(c.x), int(c.y))
                    # BFS to coin
                    coin_blocked = set(wall_positions)
                    coin_visited = {(px, py)}
                    coin_visited.update(coin_blocked)
                    coin_queue = deque([([], px, py)])
                    coin_path = None
                    while coin_queue:
                        ca, cx, cy = coin_queue.popleft()
                        if len(ca) >= 200:
                            continue
                        if cx == prox_pos[0] and cy == prox_pos[1]:
                            coin_path = ca
                            break
                        for d, (dx, dy) in dir_offsets_cached.items():
                            nx, ny = cx + dx, cy + dy
                            if (nx, ny) in coin_visited:
                                continue
                            coin_visited.add((nx, ny))
                            coin_queue.append((ca + [(d, None)], nx, ny))
                    if coin_path:
                        print(f"  BFS to coin ({int(c.x)},{int(c.y)}): {len(coin_path)} steps, "
                              f"prox_pos=({prox_pos[0]},{prox_pos[1]})")
                    else:
                        print(f"  BFS to coin ({int(c.x)},{int(c.y)}): NO PATH")

        # Now actually run the solver to see what happens
        if iteration >= 5:
            print(f"\n  Stopping trace at iteration 5 to avoid timeout")
            break

    # Finally, try the actual solver
    print(f"\n{'='*60}")
    print(f"  ACTUAL SOLVER RUN ON L2")
    print(f"{'='*60}")
    game_copy = copy.deepcopy(game)
    t_solver = time.time()
    plan2 = solve_ls20(game_copy, 2)
    solver_elapsed = time.time() - t_solver
    if plan2:
        print(f"  ✅ Solver found plan: {len(plan2)} steps, {solver_elapsed:.1f}s")
    else:
        print(f"  ❌ Solver returned None, {solver_elapsed:.1f}s")


if __name__ == "__main__":
    main()
