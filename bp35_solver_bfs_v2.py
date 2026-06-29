"""
bp35 BFS Solver v2 — Uses deepcopy for fast state save/restore.

Instead of replaying from scratch for each BFS node, we deepcopy
the engine state before branching. This is much faster.

Strategy: 
1. Start each level with set_level()
2. BFS through (keyboard + click) actions
3. Use deepcopy(engine) to save state before exploring branches
4. Restore engine state after exploring a branch
"""

import json
import time
import copy
from collections import deque
from arc_agi import Arcade, OperationMode
from arcengine.enums import GameState

ORACLE_FILE = "oracle_all_levels_expanded.json"


def replay_action(env, action):
    """Replay a single action."""
    if isinstance(action, list):
        return env.step(6, {"x": action[0], "y": action[1]})
    else:
        return env.step(action)


def get_state_key(game):
    """State key for dedup: (px, py, gravity, remaining_clickables)."""
    engine = game.oztjzzyqoek
    player = engine.twdpowducb
    world = engine.hdnrlfmyrj

    px, py = player.qumspquyus
    gravity = engine.vivnprldht

    # Track remaining clickable objects (clicking removes/changes them)
    clickables = []
    for tag in ["qclfkhjnaac", "etlsaqqtjvn", "yuuqpmlxorv", "lrpkmzabbfa"]:
        try:
            sprites = world.wwkbcxznzg(tag)
            for s in sprites:
                clickables.append((s.grid_x, s.grid_y, tag))
        except:
            pass
    clickables.sort()
    return (px, py, gravity, tuple(clickables))


def get_click_targets(game):
    """Get all clickable objects and compute display coords for clicks."""
    engine = game.oztjzzyqoek
    world = engine.hdnrlfmyrj
    # Use engine.camera (obfuscated type) which has rczgvgfsfb
    camera_y = engine.camera.rczgvgfsfb[1]

    targets = []
    for tag in ["qclfkhjnaac", "etlsaqqtjvn", "yuuqpmlxorv", "lrpkmzabbfa", "oonshderxef"]:
        try:
            sprites = world.wwkbcxznzg(tag)
            for s in sprites:
                gx, gy = s.grid_x, s.grid_y
                # display coords for click: grid * 6 - camera_y
                display_x = gx * 6
                display_y = gy * 6 - camera_y
                targets.append((tag, gx, gy, display_x, display_y))
        except:
            pass
    return targets


def solve_level(target_level, time_limit=180):
    """BFS solver for a single bp35 level using deepcopy."""
    start_time = time.time()

    # Step limits (from qmjscfjptx renderer)
    if target_level <= 6:
        max_steps = 63
    else:
        max_steps = 127

    # Initialize
    arcade = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("bp35")
    obs = env.reset()
    game = env._game
    game.set_level(target_level)

    initial_key = get_state_key(game)
    print(
        f"L{target_level}: start ({initial_key[0]},{initial_key[1]}), "
        f"grav={initial_key[2]}, max_steps={max_steps}"
    )

    visited = {initial_key}
    # Queue: (action_sequence, engine_deepcopy, steps_count)
    queue = deque()
    queue.append((
        [],
        copy.deepcopy(game.oztjzzyqoek),
        game.hbqwwgceeqp,
    ))

    iterations = 0

    while queue and time.time() - start_time < time_limit:
        actions, saved_engine, saved_steps = queue.popleft()
        iterations += 1

        if iterations % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"  iter {iterations}, queue={len(queue)}, "
                f"visited={len(visited)}, time={elapsed:.1f}s"
            )

        if len(actions) >= max_steps:
            continue

        # Restore state
        game.oztjzzyqoek = saved_engine
        game.hbqwwgceeqp = saved_steps

        # Verify state matches
        current_key = get_state_key(game)

        # Get click targets at current state
        click_targets = get_click_targets(game)

        # Enumerate all candidate actions
        candidates = [(3, None), (4, None)]  # LEFT, RIGHT
        for tag, gx, gy, dx, dy in click_targets:
            candidates.append((6, {"x": dx, "y": dy}))

        for action_id, action_data in candidates:
            # Save state before trying this action
            pre_engine = copy.deepcopy(game.oztjzzyqoek)
            pre_steps = game.hbqwwgceeqp

            # Execute action
            if action_data:
                obs = env.step(action_id, action_data)
                new_action = [action_data["x"], action_data["y"]]
            else:
                obs = env.step(action_id)
                new_action = action_id

            if obs.state == GameState.WIN:
                winning_seq = actions + [new_action]
                print(
                    f"  WON at step {len(winning_seq)}! "
                    f"Last action: {new_action}"
                )
                return winning_seq

            if obs.state == GameState.NOT_FINISHED:
                new_key = get_state_key(game)

                if new_key not in visited:
                    visited.add(new_key)
                    # Save current state for future BFS exploration
                    queue.append((
                        actions + [new_action],
                        copy.deepcopy(game.oztjzzyqoek),
                        game.hbqwwgceeqp,
                    ))

            # Restore state for next candidate
            game.oztjzzyqoek = pre_engine
            game.hbqwwgceeqp = pre_steps

    elapsed = time.time() - start_time
    print(f"  BFS exhausted: {iterations} iters, {len(visited)} states, {elapsed:.1f}s")
    return None


def main():
    oracle_data = json.load(open(ORACLE_FILE))

    levels_to_solve = [3, 4, 5, 6, 7, 8]

    results = {}
    for lvl in levels_to_solve:
        print(f"\n{'=' * 60}")
        print(f"Solving bp35 Level {lvl}...")
        print(f"{'=' * 60}")

        seq = solve_level(lvl, time_limit=180)

        if seq:
            print(f"Level {lvl}: SOLVED in {len(seq)} steps!")
            results[str(lvl)] = seq
        else:
            print(f"Level {lvl}: FAILED")

    print(f"\n{'=' * 60}")
    print("Summary:")
    for lvl, seq in sorted(results.items()):
        print(f"  L{lvl}: {len(seq)} steps")

    if results:
        with open("bp35_new_sequences.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved to bp35_new_sequences.json")

    return results


if __name__ == "__main__":
    main()
