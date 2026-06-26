"""BFS path finder for ls20 L1 — find path from player to rotation changer."""
import sys, os, copy, time
from collections import deque
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20, _is_level_solved
from src.agent.oracle_adapters import get_oracle_adapter


def bfs_path_to_changer():
    """BFS to find path from player start to rotation changer in L1."""
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
    adapter = get_oracle_adapter("ls20", game)

    print(f"Start: ({adapter.player.x}, {adapter.player.y})")
    print(f"Goal: ({adapter.goals[0].x}, {adapter.goals[0].y})")
    print(f"Changer: ({adapter.state_changers['rotation'][0].x}, {adapter.state_changers['rotation'][0].y})")

    # BFS from start position, finding path to changer
    # Use game simulation to check valid moves
    start_x = int(adapter.player.x)
    start_y = int(adapter.player.y)
    changer_x = int(adapter.state_changers['rotation'][0].x)
    changer_y = int(adapter.state_changers['rotation'][0].y)

    print(f"\n=== BFS Path from ({start_x},{start_y}) to changer ({changer_x},{changer_y}) ===")

    # BFS with game state tracking
    # State = (game_copy, actions_list)
    visited_positions: set[tuple[int, int]] = {(start_x, start_y)}
    queue: deque = deque()
    queue.append((copy.deepcopy(game), []))

    found_path = None
    max_steps = 50

    while queue:
        cur_game, cur_actions = queue.popleft()

        if len(cur_actions) >= max_steps:
            continue

        cur_adapter = get_oracle_adapter("ls20", cur_game)
        if cur_adapter is None or cur_adapter.player is None:
            continue

        cur_px = int(cur_adapter.player.x)
        cur_py = int(cur_adapter.player.y)

        # Check if reached changer
        if cur_px == changer_x and cur_py == changer_y:
            found_path = cur_actions
            print(f"Found path to changer! Length: {len(cur_actions)}")
            print(f"Path: {cur_actions}")
            # Check state after reaching changer
            new_adapter = get_oracle_adapter("ls20", cur_game)
            print(f"State after reaching changer: {new_adapter.player_state}")
            break

        # Expand directions
        for d in [1, 2, 3, 4]:
            test_sim = copy.deepcopy(cur_game)
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
                continue  # Blocked
            if (new_px, new_py) in visited_positions:
                continue
            visited_positions.add((new_px, new_py))
            dir_name = {1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}[d]
            queue.append((test_sim, cur_actions + [dir_name]))

    if found_path is None:
        print("No path found to changer!")
        # Print all reachable positions
        print(f"Reachable positions: {sorted(visited_positions)}")

    # Now find path from changer to goal (after triggering)
    if found_path:
        print(f"\n=== BFS Path from changer ({changer_x},{changer_y}) to goal ===")
        # After 3 triggers, rotation should be 3 (= goal requirement)
        # Need to trigger 3 times: leave changer, come back, repeat

        # Simulate 3 trigger cycles
        trigger_game = copy.deepcopy(game)
        # Navigate to changer
        for d_name in found_path:
            d = {'UP': 1, 'DOWN': 2, 'LEFT': 3, 'RIGHT': 4}[d_name]
            trigger_game.perform_action(ActionInput(id=d, data={}))

        t_adapter = get_oracle_adapter("ls20", trigger_game)
        print(f"After 1st trigger: state = {t_adapter.player_state}")

        # Need 2 more triggers: leave changer and come back
        # Retreat: move away from changer
        for retreat_dir in [3, 1]:  # LEFT then UP
            test = copy.deepcopy(trigger_game)
            test.perform_action(ActionInput(id=retreat_dir, data={}))
            t2 = get_oracle_adapter("ls20", test)
            if t2 and t2.player:
                new_x = int(t2.player.x)
                new_y = int(t2.player.y)
                if (new_x, new_y) != (changer_x, changer_y):
                    trigger_game.perform_action(ActionInput(id=retreat_dir, data={}))
                    print(f"  Retreated {retreat_dir}: now at ({new_x},{new_y})")

        # Come back to changer (2nd trigger)
        # Need BFS from current position back to changer
        t_adapter2 = get_oracle_adapter("ls20", trigger_game)
        cur_x = int(t_adapter2.player.x)
        cur_y = int(t_adapter2.player.y)
        print(f"Before 2nd return: at ({cur_x},{cur_y})")

        # Quick BFS back to changer
        visited2: set[tuple[int, int]] = {(cur_x, cur_y)}
        queue2: deque = deque()
        queue2.append((copy.deepcopy(trigger_game), []))
        path2 = None
        while queue2:
            g2, a2 = queue2.popleft()
            if len(a2) >= 20:
                continue
            a2_adapter = get_oracle_adapter("ls20", g2)
            if a2_adapter and a2_adapter.player:
                a2x = int(a2_adapter.player.x)
                a2y = int(a2_adapter.player.y)
                if a2x == changer_x and a2y == changer_y:
                    path2 = a2
                    break
                for d in [1, 2, 3, 4]:
                    t2 = copy.deepcopy(g2)
                    t2.perform_action(ActionInput(id=d, data={}))
                    t2a = get_oracle_adapter("ls20", t2)
                    if t2a and t2a.player:
                        nx2 = int(t2a.player.x)
                        ny2 = int(t2a.player.y)
                        if (nx2, ny2) != (a2x, a2y) and (nx2, ny2) not in visited2:
                            visited2.add((nx2, ny2))
                            queue2.append((t2, a2 + [d]))

        if path2:
            for d in path2:
                trigger_game.perform_action(ActionInput(id=d, data={}))
            t3 = get_oracle_adapter("ls20", trigger_game)
            print(f"After 2nd trigger: state = {t3.player_state}")
        else:
            print("Can't find path back to changer for 2nd trigger!")


if __name__ == "__main__":
    bfs_path_to_changer()
