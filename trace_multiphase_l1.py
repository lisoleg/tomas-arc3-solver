"""Trace multiphase BFS on L1 with detailed logging."""
import sys, os, copy, time
from collections import deque
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20, _is_level_solved, _detect_game_step
from src.agent.oracle_adapters import get_oracle_adapter


def trace_multiphase_l1():
    """Trace multiphase BFS on L1."""
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    env.step(GameAction.ACTION1)

    # Advance to L1
    plan_l0 = solve_ls20(env._game, 0)
    ACTION_MAP = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}
    for aid, data in plan_l0:
        if aid in ACTION_MAP:
            env.step(ACTION_MAP[aid])

    game_l1 = copy.deepcopy(env._game)
    step_size = _detect_game_step(game_l1)
    original_level = game_l1._current_level_index

    sim = copy.deepcopy(game_l1)
    collected = []
    t0 = time.time()

    # Lightweight BFS
    def lightweight_bfs(start_game, target_x, target_y, max_steps=50):
        start_adapter = get_oracle_adapter("ls20", start_game)
        if not start_adapter or not start_adapter.player:
            return None
        start_px = int(start_adapter.player.x)
        start_py = int(start_adapter.player.y)
        visited = {(start_px, start_py)}
        queue = deque()
        queue.append((copy.deepcopy(start_game), []))

        nodes = 0
        while queue and time.time() - t0 < 40:
            cur_game, cur_actions = queue.popleft()
            nodes += 1

            if len(cur_actions) >= max_steps:
                continue

            cur_adapter = get_oracle_adapter("ls20", cur_game)
            if not cur_adapter or not cur_adapter.player:
                continue

            cur_px = int(cur_adapter.player.x)
            cur_py = int(cur_adapter.player.y)

            if cur_px == target_x and cur_py == target_y:
                print(f"  BFS found path: {len(cur_actions)} steps, {nodes} nodes explored")
                return cur_actions

            if _is_level_solved(cur_game, original_level):
                print(f"  BFS solved level: {len(cur_actions)} steps")
                return cur_actions

            for d in [1, 2, 3, 4]:
                test_sim = copy.deepcopy(cur_game)
                ai = ActionInput(id=d, data={})
                try:
                    test_sim.perform_action(ai)
                except Exception:
                    continue
                test_adapter = get_oracle_adapter("ls20", test_sim)
                if not test_adapter or not test_adapter.player:
                    continue
                new_px = int(test_adapter.player.x)
                new_py = int(test_adapter.player.y)
                if (new_px, new_py) == (cur_px, cur_py):
                    continue
                if (new_px, new_py) in visited:
                    continue
                visited.add((new_px, new_py))
                queue.append((test_sim, cur_actions + [(d, None)]))

        print(f"  BFS failed: {nodes} nodes, visited={len(visited)} positions")
        return None

    # Main loop
    for iteration in range(20):
        if time.time() - t0 > 40:
            break
        if _is_level_solved(sim, original_level):
            print(f"\nSOLVED! Total actions: {len(collected)}")
            return collected
        if len(collected) >= 80:
            break

        adapter = get_oracle_adapter("ls20", sim)
        if not adapter or not adapter.player:
            break

        px = int(adapter.player.x)
        py = int(adapter.player.y)
        ps = adapter.player_state
        goal_reqs = adapter.goal_requirements
        dim_sizes = adapter.state_dimension_sizes
        state_changers = adapter.state_changers
        goals = adapter.goals

        if not goal_reqs or not goals:
            break

        # Find best goal
        best_i = 0
        best_mismatch = 999
        for i, req in enumerate(goal_reqs):
            mismatch = (req['rotation'] != ps['rotation']) + (req['color'] != ps['color']) + (req['shape'] != ps['shape'])
            if mismatch < best_mismatch:
                best_mismatch = mismatch
                best_i = i

        best_req = goal_reqs[best_i]
        best_goal = goals[best_i] if best_i < len(goals) else None

        print(f"\nIteration {iteration}: pos=({px},{py}) state={ps} mismatch={best_mismatch}")

        if best_mismatch == 0 and best_goal:
            goal_x = int(best_goal.x)
            goal_y = int(best_goal.y)
            print(f"  State MATCHES goal! BFS to goal ({goal_x},{goal_y})")
            path = lightweight_bfs(sim, goal_x, goal_y)
            if path:
                for aid, data in path:
                    sim.perform_action(ActionInput(id=aid, data=data if data else {}))
                    collected.append((aid, data))
                if _is_level_solved(sim, original_level):
                    print(f"\nSOLVED! Total: {len(collected)}")
                    return collected
            break

        # Find needed changers
        mismatches = []
        for dim in ['rotation', 'color', 'shape']:
            cur = ps.get(dim, 0)
            tgt = best_req.get(dim, 0)
            dsize = dim_sizes.get(dim, 4)
            triggers = (tgt - cur) % dsize
            if triggers > 0:
                mismatches.append((dim, triggers, dsize))
        mismatches.sort(key=lambda x: -x[1])
        print(f"  Mismatches: {mismatches}")

        target_x, target_y = px, py
        target_dim = mismatches[0][0] if mismatches else 'rotation'
        for dim_name, _, _ in mismatches:
            changers = state_changers.get(dim_name, [])
            if changers:
                nearest_ch = min(changers, key=lambda c: abs(px - int(c.x)) + abs(py - int(c.y)))
                target_x = int(nearest_ch.x)
                target_y = int(nearest_ch.y)
                target_dim = dim_name
                break

        print(f"  Target changer: ({target_x},{target_y}) dim={target_dim}")

        # If already on changer → retreat
        if target_x == px and target_y == py:
            print(f"  Already on changer! Retreat 1 step...")
            for d in [1, 2, 3, 4]:
                test = copy.deepcopy(sim)
                ai = ActionInput(id=d, data={})
                try:
                    test.perform_action(ai)
                except Exception:
                    continue
                ta = get_oracle_adapter("ls20", test)
                if ta and ta.player:
                    nx = int(ta.player.x)
                    ny = int(ta.player.y)
                    if (nx, ny) != (px, py):
                        sim.perform_action(ai)
                        collected.append((d, None))
                        print(f"  Retreat: dir {d}, now at ({nx},{ny})")
                        # Check state after retreat (may have stepped on another changer)
                        new_ps = get_oracle_adapter("ls20", sim).player_state
                        print(f"  State after retreat: {new_ps}")
                        break
            continue

        # BFS to changer
        print(f"  BFS to changer ({target_x},{target_y})...")
        path = lightweight_bfs(sim, target_x, target_y)
        if path:
            for aid, data in path:
                sim.perform_action(ActionInput(id=aid, data=data if data else {}))
                collected.append((aid, data))

            new_adapter = get_oracle_adapter("ls20", sim)
            if new_adapter and new_adapter.player:
                new_ps = new_adapter.player_state
                new_px = int(new_adapter.player.x)
                new_py = int(new_adapter.player.y)
                print(f"  After BFS: pos=({new_px},{new_py}) state={new_ps}")

            if _is_level_solved(sim, original_level):
                print(f"\nSOLVED! Total: {len(collected)}")
                return collected
        else:
            print(f"  BFS to changer FAILED — stopping")
            break

    print(f"\nFailed. Total actions attempted: {len(collected)}")
    return None


if __name__ == "__main__":
    result = trace_multiphase_l1()
    if result:
        print(f"\nFinal result: {len(result)} actions")
