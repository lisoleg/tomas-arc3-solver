"""L2 solver integration diagnostic — trace _solve_ls20_delta_state_bfs decisions.

This script traces why the L2 solver returns None despite ReplayEngine
BFS succeeding for individual paths. Key question: where does the
integration break?
"""

import sys
import os
import time
import copy
from collections import deque

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import (
    _solve_ls20_delta_state_bfs,
    _is_level_solved,
    _detect_game_step,
    _detect_direction_mapping,
)
from src.agent.oracle_adapters import get_oracle_adapter


def _compute_coin_proximity_position(
    px: int, py: int, step_size: int, coin_x: int, coin_y: int,
) -> tuple[int, int]:
    """Compute grid position where player bounding box overlaps coin anchor."""
    half = step_size // 2
    for dx in [-step_size, 0, step_size]:
        for dy in [-step_size, 0, step_size]:
            cand_x = coin_x + dx
            cand_y = coin_y + dy
            if (cand_x <= coin_x + half < cand_x + step_size and
                cand_y <= coin_y + half < cand_y + step_size):
                return (cand_x, cand_y)
    grid_x = round((coin_x + half) / step_size) * step_size
    grid_y = round((coin_y + half) / step_size) * step_size
    return (grid_x, grid_y)


def trace_l2_solver_integration():
    """Trace _solve_ls20_delta_state_bfs decisions on L2."""
    print("=" * 80)
    print("L2 Solver Integration Diagnostic")
    print("=" * 80)

    # Load LS20 game environment
    arc = arc_agi.Arcade()
    env = arc.make("ls20")

    # Solve L0 first to advance to L1, then L1 to advance to L2
    baselines = [22, 123, 73, 84, 96, 192, 186]

    env.reset()
    obs = env.step(GameAction.ACTION1)

    # Solve L0
    from src.agent.game_solvers import solve_ls20
    game = env._game
    plan0 = solve_ls20(game, 0)
    if plan0:
        for aid, data in plan0:
            ai = ActionInput(id=aid, data=data if data else {})
            game.perform_action(ai)
        print(f"✅ L0 solved: {len(plan0)} steps")
    else:
        print("❌ L0 failed — cannot test L2")
        return

    # Solve L1
    game = env._game
    plan1 = solve_ls20(game, 1)
    if plan1:
        for aid, data in plan1:
            ai = ActionInput(id=aid, data=data if data else {})
            game.perform_action(ai)
        print(f"✅ L1 solved: {len(plan1)} steps")
    else:
        print("❌ L1 failed — cannot test L2")
        return

    # Now at L2
    game = env._game
    original_level = game._current_level_index
    print(f"\n── L2 State ── level_index={original_level}")

    adapter = get_oracle_adapter("ls20", game)
    if adapter is None:
        print("❌ No LS20 adapter for L2")
        return

    print(f"  Player: ({int(adapter.player.x)}, {int(adapter.player.y)})")
    print(f"  Player state: {adapter.player_state}")
    print(f"  Steps remaining: {adapter.steps_remaining}")
    print(f"  Step size: {adapter.step}")
    print(f"  Goal requirements: {adapter.goal_requirements}")
    print(f"  Goals: {[(int(g.x), int(g.y)) for g in adapter.goals]}")
    print(f"  Coins: {[(int(c.x), int(c.y)) for c in adapter.coins]}")
    print(f"  State changers:")
    for dim, chs in adapter.state_changers.items():
        for ch in chs:
            print(f"    {dim}: ({int(ch.x)}, {int(ch.y)})")

    # Detect direction mapping
    dir_offsets = _detect_direction_mapping(game)
    print(f"\n  Direction offsets: {dir_offsets}")

    # ── Mismatch analysis ──
    px = int(adapter.player.x)
    py = int(adapter.player.y)
    player_state = adapter.player_state
    goal_reqs = adapter.goal_requirements
    dim_sizes = adapter.state_dimension_sizes
    state_changers = adapter.state_changers
    remaining = adapter.steps_remaining

    print(f"\n── Mismatch Analysis ──")
    mismatches = []
    for dim_name in ['rotation', 'color', 'shape']:
        current_val = player_state.get(dim_name, 0)
        target_val = goal_reqs[0].get(dim_name, 0)
        dim_size = dim_sizes.get(dim_name, 4)
        triggers_needed = (target_val - current_val) % dim_size
        if triggers_needed > 0:
            mismatches.append((dim_name, triggers_needed, dim_size))
            print(f"  {dim_name}: current={current_val}, target={target_val}, "
                  f"triggers_needed={triggers_needed}, dim_size={dim_size}")

    # ── Changer reachability: wall-map BFS ──
    print(f"\n── Changer Reachability (Wall-Map BFS) ──")

    for dim_name, trig, dim_sz in mismatches:
        changers = state_changers.get(dim_name, [])
        if changers:
            nearest_ch = min(changers, key=lambda c: abs(px - int(c.x)) + abs(py - int(c.y)))
            ch_x, ch_y = int(nearest_ch.x), int(nearest_ch.y)
            manhattan = abs(px - ch_x) + abs(py - ch_y)
            print(f"\n  {dim_name} changer at ({ch_x}, {ch_y}):")
            print(f"    Manhattan: {manhattan}, remaining: {remaining}")

            # Build blocked positions (walls + other changers)
            wall_positions = set()
            for w in (adapter.walls or []):
                wall_positions.add((int(w.x), int(w.y)))

            other_ch_pos = set()
            for d2, chs2 in state_changers.items():
                for ch2 in chs2:
                    cp2 = (int(ch2.x), int(ch2.y))
                    if cp2 != (ch_x, ch_y):
                        other_ch_pos.add(cp2)

            blocked = set(wall_positions) | other_ch_pos
            blocked.discard((ch_x, ch_y))

            visited = {(px, py)} | blocked
            queue = deque([([], px, py)])

            t0 = time.time()
            wall_map_result = None
            wall_map_steps = None

            while queue and time.time() - t0 < 10.0:
                cur_actions, cur_px, cur_py = queue.popleft()
                if len(cur_actions) >= remaining:
                    continue
                if cur_px == ch_x and cur_py == ch_y:
                    wall_map_result = cur_actions
                    wall_map_steps = len(cur_actions)
                    break
                for d, (dx, dy) in dir_offsets.items():
                    new_px = cur_px + dx
                    new_py = cur_py + dy
                    if (new_px, new_py) in visited:
                        continue
                    visited.add((new_px, new_py))
                    queue.append((cur_actions + [(d, None)], new_px, new_py))

            if wall_map_result:
                print(f"    Wall-map BFS: ✅ FOUND in {wall_map_steps} steps")

                # Verify on real game
                sim_verify = copy.deepcopy(game)
                verified_ok = True
                for aid, data in wall_map_result:
                    ai = ActionInput(id=aid, data=data if data else {})
                    try:
                        sim_verify.perform_action(ai)
                    except Exception:
                        verified_ok = False
                        break

                if verified_ok:
                    v_adapter = get_oracle_adapter("ls20", sim_verify)
                    if v_adapter and v_adapter.player:
                        v_px = int(v_adapter.player.x)
                        v_py = int(v_adapter.player.y)
                        if v_px == ch_x and v_py == ch_y:
                            print(f"    Verification: ✅ PASSED at ({v_px}, {v_py})")
                        else:
                            print(f"    Verification: ❌ FAILED at ({v_px}, {v_py}) "
                                  f"≠ ({ch_x}, {ch_y}) → ReplayEngine needed")
                            wall_map_result = None
                    else:
                        print(f"    Verification: ❌ no adapter → ReplayEngine needed")
                        wall_map_result = None
                else:
                    print(f"    Verification: ❌ action error → ReplayEngine needed")
                    wall_map_result = None
            else:
                search_time = time.time() - t0
                print(f"    Wall-map BFS: ❌ NOT FOUND (searched {search_time:.1f}s, "
                      f"max_steps={remaining})")

            # ── ReplayEngine BFS test ──
            print(f"\n    Testing ReplayEngine BFS...")
            root_game = copy.deepcopy(game)
            root_adapter = get_oracle_adapter("ls20", root_game)
            if root_adapter is None or root_adapter.player is None:
                print(f"    ReplayEngine: ❌ no adapter")
                continue

            replay_visited = {(int(root_adapter.player.x), int(root_adapter.player.y))}
            if other_ch_pos:
                replay_visited.update(other_ch_pos)
            replay_visited.discard((ch_x, ch_y))

            replay_queue = deque([([], int(root_adapter.player.x), int(root_adapter.player.y))])
            t0_replay = time.time()
            replay_result = None

            expansions = 0
            while replay_queue and time.time() - t0_replay < 20.0:
                cur_actions, cur_px, cur_py = replay_queue.popleft()
                if len(cur_actions) >= remaining:
                    continue
                if cur_px == ch_x and cur_py == ch_y:
                    replay_result = cur_actions
                    break

                cur_game = copy.deepcopy(root_game)
                for aid, data in cur_actions:
                    ai = ActionInput(id=aid, data=data if data else {})
                    try:
                        cur_game.perform_action(ai)
                    except Exception:
                        continue

                cur_adapter = get_oracle_adapter("ls20", cur_game)
                if cur_adapter is None or cur_adapter.player is None:
                    continue

                for d in [1, 2, 3, 4]:
                    child_game = copy.deepcopy(cur_game)
                    ai = ActionInput(id=d, data={})
                    try:
                        child_game.perform_action(ai)
                    except Exception:
                        continue
                    test_adapter = get_oracle_adapter("ls20", child_game)
                    if test_adapter is None or test_adapter.player is None:
                        continue
                    new_px = int(test_adapter.player.x)
                    new_py = int(test_adapter.player.y)
                    if (new_px, new_py) == (cur_px, cur_py):
                        continue
                    if (new_px, new_py) in replay_visited:
                        continue
                    replay_visited.add((new_px, new_py))
                    replay_queue.append((cur_actions + [(d, None)], new_px, new_py))
                    expansions += 1

                if expansions % 100 == 0 and time.time() - t0_replay > 5.0:
                    print(f"    ReplayEngine: {expansions} expansions, "
                          f"{time.time() - t0_replay:.1f}s elapsed...")

            if replay_result:
                replay_time = time.time() - t0_replay
                print(f"    ReplayEngine: ✅ FOUND in {len(replay_result)} steps, "
                      f"{replay_time:.1f}s, {expansions} expansions")

                # Verify
                sim_verify2 = copy.deepcopy(game)
                for aid, data in replay_result:
                    ai = ActionInput(id=aid, data=data if data else {})
                    try:
                        sim_verify2.perform_action(ai)
                    except Exception:
                        break
                v2 = get_oracle_adapter("ls20", sim_verify2)
                if v2 and v2.player:
                    print(f"    Replay verification: player at ({int(v2.player.x)}, "
                          f"{int(v2.player.y)}), remaining={v2.steps_remaining}")
                    # Check if state changed accidentally
                    print(f"    State after path: {v2.player_state}")
            else:
                replay_time = time.time() - t0_replay
                print(f"    ReplayEngine: ❌ NOT FOUND ({replay_time:.1f}s, "
                      f"{expansions} expansions)")

    # ── Coin reachability ──
    print(f"\n── Coin Reachability ──")
    coins = adapter.coins
    if coins:
        for c in coins:
            coin_x, coin_y = int(c.x), int(c.y)
            prox = _compute_coin_proximity_position(px, py, adapter.step, coin_x, coin_y)
            print(f"\n  Coin ({coin_x}, {coin_y}) → prox={prox}")
            manhattan = abs(px - prox[0]) + abs(py - prox[1])
            print(f"    Manhattan: {manhattan}")

            # Wall-map BFS
            wall_pos = set()
            for w in (adapter.walls or []):
                wall_pos.add((int(w.x), int(w.y)))
            all_ch = set()
            for dim, chs in state_changers.items():
                for ch in chs:
                    all_ch.add((int(ch.x), int(ch.y)))
            blocked = wall_pos | all_ch
            blocked.discard(prox)

            visited = {(px, py)} | blocked
            queue = deque([([], px, py)])
            t0 = time.time()
            coin_path = None

            while queue and time.time() - t0 < 10.0:
                cur_actions, cur_px, cur_py = queue.popleft()
                if len(cur_actions) >= remaining:
                    continue
                if cur_px == prox[0] and cur_py == prox[1]:
                    coin_path = cur_actions
                    break
                for d, (dx, dy) in dir_offsets.items():
                    new_px = cur_px + dx
                    new_py = cur_py + dy
                    if (new_px, new_py) in visited:
                        continue
                    visited.add((new_px, new_py))
                    queue.append((cur_actions + [(d, None)], new_px, new_py))

            if coin_path:
                print(f"    Wall-map BFS: FOUND in {len(coin_path)} steps")
                # Verify
                sim_verify = copy.deepcopy(game)
                for aid, data in coin_path:
                    ai = ActionInput(id=aid, data=data if data else {})
                    try:
                        sim_verify.perform_action(ai)
                    except Exception:
                        break
                v = get_oracle_adapter("ls20", sim_verify)
                if v and v.player:
                    v_px, v_py = int(v.player.x), int(v.player.y)
                    print(f"    Verification: player at ({v_px}, {v_py}), "
                          f"remaining={v.steps_remaining}")
                    if (v_px, v_py) == prox:
                        print(f"    ✅ Coin reachable via wall-map BFS")
                    else:
                        print(f"    ❌ Divergence: ({v_px}, {v_py}) ≠ {prox}")
            else:
                print(f"    Wall-map BFS: NOT FOUND within {remaining} steps")

    # ── Run the actual solver ──
    print(f"\n── Running _solve_ls20_delta_state_bfs on L2 ──")
    game_copy = copy.deepcopy(game)
    t_start = time.time()
    result = _solve_ls20_delta_state_bfs(game_copy, original_level)
    t_end = time.time()

    if result is not None:
        print(f"\n✅ Solver PASSED: {len(result)} actions in {t_end - t_start:.1f}s")
    else:
        print(f"\n❌ Solver FAILED: None in {t_end - t_start:.1f}s")

    # ── Integration analysis ──
    print(f"\n── Integration Analysis ──")
    print(f"  Key timing concern:")
    print(f"  - _lightweight_bfs → verification fails → ReplayEngine fallback")
    print(f"  - Each ReplayEngine call: ~3-10s (deepcopy per expansion)")
    print(f"  - Changer reachability loop: 2 mismatches × BFS = 2 fallbacks")
    print(f"  - Total per iteration: 6-20s just for BFS calls")
    print(f"  - MAX_TOTAL_TIME = 45s → 2-3 iterations max")
    print(f"  ")
    print(f"  SOLUTION: Use Manhattan distance for changer reachability")
    print(f"  pre-filter, only use BFS (and ReplayEngine fallback) for")
    print(f"  the final path execution, not for reachability testing.")
    print(f"  ")
    print(f"  Or: Cache BFS results — if wall-map BFS found a path to a")
    print(f"  changer and verification succeeded, trust it. Only use")
    print(f"  ReplayEngine for actual path execution when wall-map fails.")


if __name__ == "__main__":
    trace_l2_solver_integration()
