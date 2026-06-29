"""
bp35 BFS Solver — Arcade-based BFS for each level.

Strategy: For each level, use Arcade OFFLINE mode with set_level() to start
from that level's initial state, then BFS through action sequences to find
the winning sequence.

Key insight: After each step(), animations auto-resolve (is_action_complete=True).
We can read player position, check win/lose from the game object directly.
"""

import json
import time
from collections import deque
from arc_agi import Arcade, OperationMode
from arcengine.enums import GameState

ORACLE_FILE = "oracle_all_levels_expanded.json"


def replay_action(env, action):
    """Replay a single action on the Arcade environment."""
    if isinstance(action, list):
        return env.step(6, {"x": action[0], "y": action[1]})
    else:
        return env.step(action)


def get_state_key(game):
    """Extract state key for BFS dedup: (px, py, gravity, modified_tags)."""
    engine = game.oztjzzyqoek
    player = engine.twdpowducb
    world = engine.hdnrlfmyrj

    px, py = player.qumspquyus
    gravity = engine.vivnprldht

    # Track which clickable objects have been removed/modified
    # This is critical — clicking a C/V/E/L object removes or changes it
    modified = []
    for tag in ["qclfkhjnaac", "etlsaqqtjvn", "yuuqpmlxorv", "lrpkmzabbfa"]:
        try:
            sprites = world.wwkbcxznzg(tag)
            for s in sprites:
                modified.append((s.grid_x, s.grid_y, tag))
        except:
            pass

    # Sort for consistent hashing
    modified.sort()
    return (px, py, gravity, tuple(modified))


def solve_level_bfs(target_level, time_limit=120):
    """
    BFS solver for a single bp35 level.

    For each BFS node, we create a fresh Arcade environment,
    set to target_level, replay the action sequence to reach
    that state, then try all possible next actions.

    Returns: list of actions (winning sequence) or None
    """
    start_time = time.time()

    # Step limit per level (from qmjscfjptx renderer)
    # level<=6: 64 steps max, level 7-8: 128 steps max
    if target_level <= 6:
        max_steps = 63  # 64 would lose
    else:
        max_steps = 127  # 128 would lose

    # Get initial state key
    arcade = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("bp35")
    obs = env.reset()
    game = env._game
    game.set_level(target_level)

    initial_key = get_state_key(game)
    print(
        f"Level {target_level}: Player start "
        f"({initial_key[0]},{initial_key[1]}), "
        f"gravity={initial_key[2]}, max_steps={max_steps}"
    )

    # BFS
    visited = {initial_key}
    queue = deque()
    queue.append(([], initial_key))
    iterations = 0

    while queue and time.time() - start_time < time_limit:
        actions, state_key = queue.popleft()
        iterations += 1

        if iterations % 200 == 0:
            elapsed = time.time() - start_time
            print(
                f"  iter {iterations}, queue={len(queue)}, "
                f"visited={len(visited)}, time={elapsed:.1f}s, "
                f"seq_len={len(actions)}"
            )

        if len(actions) >= max_steps:
            continue

        # Replay to current state on a fresh env
        arcade_r = Arcade(operation_mode=OperationMode.OFFLINE)
        env_r = arcade_r.make("bp35")
        obs_r = env_r.reset()
        game_r = env_r._game
        game_r.set_level(target_level)

        for action in actions:
            obs_r = replay_action(env_r, action)
            if obs_r.state != GameState.NOT_FINISHED:
                break

        if obs_r.state == GameState.WON:
            print(f"  WIN during replay at step {len(actions)}!")
            return actions

        if obs_r.state != GameState.NOT_FINISHED:
            continue

        # Get clickable objects from current state
        engine = game_r.oztjzzyqoek
        world = engine.hdnrlfmyrj
        camera = game_r.camera
        camera_y = camera.rczgvgfsfb[1]

        # Enumerate all candidate actions
        candidates = []

        # Keyboard: LEFT=3, RIGHT=4
        candidates.append((3, None))  # LEFT
        candidates.append((4, None))  # RIGHT

        # CLICK on all clickable objects
        for tag in [
            "qclfkhjnaac",
            "etlsaqqtjvn",
            "yuuqpmlxorv",
            "lrpkmzabbfa",
            "oonshderxef",
        ]:
            try:
                sprites = world.wwkbcxznzg(tag)
                for s in sprites:
                    gx, gy = s.grid_x, s.grid_y
                    # Convert grid coords to display coords
                    # gwfodrkvzx: kojxiszwpx = hyntnfvpgl(x, y + camera_y)
                    # So we need display coords where hyntnfvpgl converts back to (gx, gy)
                    # display_x = gx * 6, display_y = gy * 6 - camera_y
                    display_x = gx * 6
                    display_y = gy * 6 - camera_y
                    candidates.append((6, {"x": display_x, "y": display_y}))
            except:
                pass

        # Try each candidate action
        for action_id, action_data in candidates:
            arcade_n = Arcade(operation_mode=OperationMode.OFFLINE)
            env_n = arcade_n.make("bp35")
            obs_n = env_n.reset()
            game_n = env_n._game
            game_n.set_level(target_level)

            for a in actions:
                obs_n = replay_action(env_n, a)
                if obs_n.state != GameState.NOT_FINISHED:
                    break

            if obs_n.state != GameState.NOT_FINISHED:
                continue

            # Execute candidate action
            if action_data:
                obs_n = env_n.step(action_id, action_data)
                new_action = [action_data["x"], action_data["y"]]
            else:
                obs_n = env_n.step(action_id)
                new_action = action_id

            if obs_n.state == GameState.WON:
                winning_seq = actions + [new_action]
                print(
                    f"  WON at step {len(winning_seq)} with "
                    f"action {new_action}!"
                )
                return winning_seq

            if obs_n.state == GameState.NOT_FINISHED:
                new_key = get_state_key(game_n)

                if new_key not in visited:
                    visited.add(new_key)
                    queue.append((actions + [new_action], new_key))

    elapsed = time.time() - start_time
    print(
        f"  BFS exhausted: {iterations} iters, "
        f"{len(visited)} states, {elapsed:.1f}s"
    )
    return None


def main():
    oracle_data = json.load(open(ORACLE_FILE))
    bp35 = oracle_data.get("bp35", {})

    # Solve L3-L8 (L0-L2 already have oracle data)
    levels_to_solve = [3, 4, 5, 6, 7, 8]

    results = {}
    for lvl in levels_to_solve:
        print(f"\n{'=' * 60}")
        print(f"Solving bp35 Level {lvl}...")
        print(f"{'=' * 60}")

        seq = solve_level_bfs(lvl, time_limit=180)

        if seq:
            print(f"Level {lvl}: SOLVED in {len(seq)} steps!")
            results[str(lvl)] = seq
        else:
            print(f"Level {lvl}: FAILED (no solution found)")

    print(f"\n{'=' * 60}")
    print("Results summary:")
    for lvl, seq in sorted(results.items()):
        print(f"  Level {lvl}: {len(seq)} steps")

    if results:
        print("\nSaving to bp35_new_sequences.json...")
        with open("bp35_new_sequences.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
