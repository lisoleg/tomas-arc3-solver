"""Trace κ-PS BFS solver on L1 to understand why it fails."""
import sys, os, copy, time, heapq
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20, _is_level_solved, _detect_game_step
from src.agent.oracle_adapters import get_oracle_adapter


def trace_kappa_ps_bfs_l1():
    """Trace the κ-PS BFS solver on L1."""
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    env.step(GameAction.ACTION1)

    # Advance to L1
    plan_l0 = solve_ls20(env._game, 0)
    ACTION_MAP = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}
    for aid, data in plan_l0:
        if aid in ACTION_MAP:
            env.step(ACTION_MAP[aid])

    game = copy.deepcopy(env._game)
    original_level = game._current_level_index

    # Skip stages 1 and 2, go directly to κ-PS BFS
    step_size = _detect_game_step(game)
    adapter = get_oracle_adapter("ls20", game)

    px0 = int(adapter.player.x)
    py0 = int(adapter.player.y)
    init_state = adapter.player_state

    print(f"Start: ({px0},{py0}) state={init_state}")
    print(f"Step size: {step_size}")

    goal_requirements = adapter.goal_requirements
    dim_sizes = adapter.state_dimension_sizes
    goals = adapter.goals

    # Build initial targets
    initial_targets = []
    if goals and goal_requirements:
        for i, g in enumerate(goals):
            if i < len(goal_requirements):
                req = goal_requirements[i]
                initial_targets.append((int(g.x), int(g.y), req['rotation'], req['color'], req['shape']))

    print(f"Targets: {initial_targets}")

    init_rot = init_state.get('rotation', 0)
    init_color = init_state.get('color', 0)
    init_shape = init_state.get('shape', 0)

    # Composite distance function
    def composite_dist(pos_x, pos_y, rot, color, shape, target, dim_sizes):
        gx, gy, g_rot, g_color, g_shape = target
        pos_dist = abs(pos_x - gx) + abs(pos_y - gy)
        rot_triggers = (g_rot - rot) % dim_sizes.get('rotation', 4)
        color_triggers = (g_color - color) % dim_sizes.get('color', 4)
        shape_triggers = (g_shape - shape) % dim_sizes.get('shape', 3)
        state_dist = rot_triggers + color_triggers + shape_triggers
        return state_dist * 5.0 * step_size + pos_dist

    nearest_target = min(
        initial_targets,
        key=lambda t: composite_dist(px0, py0, init_rot, init_color, init_shape, t, dim_sizes),
    )
    print(f"Nearest target: {nearest_target}")
    print(f"Composite dist to target: {composite_dist(px0, py0, init_rot, init_color, init_shape, nearest_target, dim_sizes)}")

    # Run κ-PS BFS with increased limits
    LIU_EPSILON = 0.01
    MAX_NODES = 5000  # Increased
    MAX_BFS_TIME = 60.0  # Increased
    MAX_PATH_LEN = 80  # Increased

    visited_composite = {(px0, py0, init_rot, init_color, init_shape)}
    pq = []
    counter = 0

    # Initialize
    for d in [1, 2, 3, 4]:
        test_sim = copy.deepcopy(game)
        ai = ActionInput(id=d, data={})
        try:
            test_sim.perform_action(ai)
        except Exception:
            continue
        test_adapter = get_oracle_adapter("ls20", test_sim)
        if test_adapter is None or test_adapter.player is None:
            continue
        new_px = int(test_adapter.player.x)
        new_py = int(test_adapter.player.y)
        if (new_px, new_py) == (px0, py0):
            continue
        new_pstate = test_adapter.player_state
        new_rot = new_pstate.get('rotation', init_rot)
        new_color = new_pstate.get('color', init_color)
        new_shape = new_pstate.get('shape', init_shape)
        composite_key = (new_px, new_py, new_rot, new_color, new_shape)
        if composite_key in visited_composite:
            continue
        visited_composite.add(composite_key)

        comp_dist = composite_dist(new_px, new_py, new_rot, new_color, new_shape, nearest_target, dim_sizes)
        ic = comp_dist / max(1, step_size)
        s_rel = 0.1 - 0.5 * ic
        kappa_pri = 1.0 / (s_rel + LIU_EPSILON)

        heapq.heappush(pq, (-kappa_pri, counter, test_sim, [(d, None)]))
        counter += 1

    print(f"\nInitial queue size: {len(pq)}")

    # κ-PS BFS exploration
    t0 = time.time()
    nodes = 0
    best_state_grad = 999

    while pq and time.time() - t0 < MAX_BFS_TIME and nodes < MAX_NODES:
        neg_pri, _, cur_sim, cur_actions = heapq.heappop(pq)
        nodes += 1

        # Periodic logging
        if nodes % 500 == 0:
            elapsed = time.time() - t0
            cur_adapter = get_oracle_adapter("ls20", cur_sim)
            if cur_adapter and cur_adapter.player:
                cp = (int(cur_adapter.player.x), int(cur_adapter.player.y))
                cs = cur_adapter.player_state
                print(f"  Node {nodes}: pos={cp} state={cs} actions={len(cur_actions)} time={elapsed:.1f}s")

        # Check solved
        if _is_level_solved(cur_sim, original_level):
            print(f"\nSOLVED at node {nodes}! Actions: {len(cur_actions)}")
            print(f"Plan: {cur_actions}")
            return cur_actions

        cur_adapter = get_oracle_adapter("ls20", cur_sim)
        if cur_adapter is None or cur_adapter.player is None:
            continue

        cur_px = int(cur_adapter.player.x)
        cur_py = int(cur_adapter.player.y)

        if len(cur_actions) >= MAX_PATH_LEN:
            continue

        cur_pstate = cur_adapter.player_state
        cur_rot = cur_pstate.get('rotation', 0)
        cur_color = cur_pstate.get('color', 0)
        cur_shape = cur_pstate.get('shape', 0)

        # Dynamic target
        cur_goals = cur_adapter.goals
        cur_goal_reqs = cur_adapter.goal_requirements
        cur_dim_sizes = cur_adapter.state_dimension_sizes

        cur_targets = []
        if cur_goals and cur_goal_reqs:
            for i, g in enumerate(cur_goals):
                if i < len(cur_goal_reqs):
                    req = cur_goal_reqs[i]
                    cur_targets.append((int(g.x), int(g.y), req['rotation'], req['color'], req['shape']))

        if cur_targets:
            cur_nearest = min(
                cur_targets,
                key=lambda t: composite_dist(cur_px, cur_py, cur_rot, cur_color, cur_shape, t, cur_dim_sizes),
            )
        else:
            cur_nearest = nearest_target

        # Expand
        for d in [1, 2, 3, 4]:
            test_sim = copy.deepcopy(cur_sim)
            ai = ActionInput(id=d, data={})
            try:
                test_sim.perform_action(ai)
            except Exception:
                continue
            test_adapter = get_oracle_adapter("ls20", test_sim)
            if test_adapter is None or test_adapter.player is None:
                continue
            new_px = int(test_adapter.player.x)
            new_py = int(test_adapter.player.y)
            if (new_px, new_py) == (cur_px, cur_py):
                continue
            new_pstate = test_adapter.player_state
            new_rot = new_pstate.get('rotation', cur_rot)
            new_color = new_pstate.get('color', cur_color)
            new_shape = new_pstate.get('shape', cur_shape)
            composite_key = (new_px, new_py, new_rot, new_color, new_shape)
            if composite_key in visited_composite:
                continue
            visited_composite.add(composite_key)

            # Check solved
            if _is_level_solved(test_sim, original_level):
                print(f"\nSOLVED at expansion! Actions: {len(cur_actions) + 1}")
                return cur_actions + [(d, None)]

            comp_dist = composite_dist(new_px, new_py, new_rot, new_color, new_shape, cur_nearest, cur_dim_sizes)
            ic = comp_dist / max(1, step_size)
            s_rel = 0.1 - 0.5 * ic
            kappa_pri = 1.0 / (s_rel + LIU_EPSILON)

            # State matching bonus
            _, _, g_rot, g_color, g_shape = cur_nearest
            if new_rot == g_rot:
                kappa_pri *= 0.8
            if new_color == g_color:
                kappa_pri *= 0.8
            if new_shape == g_shape:
                kappa_pri *= 0.8
            if new_rot == g_rot and new_color == g_color and new_shape == g_shape:
                kappa_pri *= 0.5

            heapq.heappush(pq, (-kappa_pri, counter, test_sim, cur_actions + [(d, None)]))
            counter += 1

    elapsed = time.time() - t0
    print(f"\nκ-PS BFS exhausted: nodes={nodes}, time={elapsed:.1f}s")
    print(f"Visited composite states: {len(visited_composite)}")
    return None


if __name__ == "__main__":
    trace_kappa_ps_bfs_l1()
