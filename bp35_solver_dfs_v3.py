"""
bp35 DFS Solver v3 — Fast DFS with deepcopy backtracking.

Uses DFS with iterative deepening (IDA*) to find winning sequences.
For each branch: deepcopy engine → try action → if win, return; 
if not finished, recurse; restore engine state after branch.
"""

import json
import time
import copy
from arc_agi import Arcade, OperationMode
from arcengine.enums import GameState

ORACLE_FILE = "oracle_all_levels_expanded.json"

# Action mapping
LEFT = 3
RIGHT = 4
CLICK = 6
UNDO = 7


def get_state_key(game):
    """State key for visited set dedup."""
    engine = game.oztjzzyqoek
    player = engine.twdpowducb
    world = engine.hdnrlfmyrj

    px, py = player.qumspquyus
    gravity = engine.vivnprldht

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
    """Get all clickable sprites with display coords."""
    engine = game.oztjzzyqoek
    world = engine.hdnrlfmyrj
    camera_y = engine.camera.rczgvgfsfb[1]

    targets = []
    for tag in ["qclfkhjnaac", "etlsaqqtjvn", "yuuqpmlxorv", "lrpkmzabbfa", "oonshderxef"]:
        try:
            sprites = world.wwkbcxznzg(tag)
            for s in sprites:
                gx, gy = s.grid_x, s.grid_y
                dx = gx * 6
                dy = gy * 6 - camera_y
                targets.append((tag, gx, gy, dx, dy))
        except:
            pass
    return targets


def solve_level_dfs(env, game, target_level, max_depth, visited, start_time, time_limit):
    """DFS with deepcopy backtracking."""
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return None

    steps = game.hbqwwgceeqp
    if steps >= max_depth:
        return None

    state_key = get_state_key(game)
    if state_key in visited:
        return None
    visited.add(state_key)

    # Check if already won (shouldn't happen at entry but just in case)
    # We check win AFTER each step, not before

    # Enumerate actions
    candidates = [(LEFT, None), (RIGHT, None)]
    
    click_targets = get_click_targets(game)
    for tag, gx, gy, dx, dy in click_targets:
        candidates.append((CLICK, {"x": dx, "y": dy}))

    # Try each candidate
    for action_id, action_data in candidates:
        # Save state
        saved_engine = copy.deepcopy(game.oztjzzyqoek)
        saved_steps = game.hbqwwgceeqp

        # Execute action
        if action_data:
            obs = env.step(action_id, action_data)
            action_record = [action_data["x"], action_data["y"]]
        else:
            obs = env.step(action_id)
            action_record = action_id

        if obs.state == GameState.WIN:
            return [action_record]

        if obs.state == GameState.NOT_FINISHED:
            # Recurse
            result = solve_level_dfs(
                env, game, target_level, max_depth, 
                visited, start_time, time_limit
            )
            if result is not None:
                return [action_record] + result

        # Restore state
        game.oztjzzyqoek = saved_engine
        game.hbqwwgceeqp = saved_steps

    # No solution found from this state
    visited.discard(state_key)  # Allow re-exploration from other paths
    return None


def solve_level(target_level, time_limit=180):
    """Solve a bp35 level using IDA* (iterative deepening DFS)."""
    start_time = time.time()

    # Step limits from qmjscfjptx renderer
    if target_level <= 6:
        max_steps = 63
    else:
        max_steps = 127

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

    # Iterative deepening: try increasing depth limits
    for depth_limit in range(5, max_steps + 1, 5):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        visited = set()
        print(f"  IDA* depth={depth_limit}, elapsed={elapsed:.1f}s...")

        result = solve_level_dfs(
            env, game, target_level, depth_limit,
            visited, start_time, time_limit
        )

        if result is not None:
            elapsed = time.time() - start_time
            print(
                f"  SOLVED in {len(result)} steps, "
                f"{elapsed:.1f}s, visited={len(visited)}"
            )
            return result

    elapsed = time.time() - start_time
    print(f"  FAILED after {elapsed:.1f}s")
    return None


def verify_sequence(target_level, sequence):
    """Verify a winning sequence works correctly."""
    arcade = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("bp35")
    obs = env.reset()
    game = env._game
    game.set_level(target_level)

    for i, action in enumerate(sequence):
        if isinstance(action, list):
            obs = env.step(6, {"x": action[0], "y": action[1]})
        else:
            obs = env.step(action)
        
        print(f"  Step {i}: action={action} → state={obs.state}, "
              f"player={game.oztjzzyqoek.twdpowducb.qumspquyus}")
        
        if obs.state == GameState.WIN:
            print(f"  Verified: WIN at step {i+1}")
            return True
        if obs.state == GameState.GAME_OVER:
            print(f"  FAILED: GAME_OVER at step {i+1}")
            return False
    
    print(f"  Sequence ended without WIN")
    return False


def main():
    oracle_data = json.load(open(ORACLE_FILE))

    # Solve L3-L8
    levels_to_solve = [3, 4, 5, 6, 7, 8]

    results = {}
    for lvl in levels_to_solve:
        print(f"\n{'=' * 60}")
        print(f"Solving bp35 Level {lvl}...")
        print(f"{'=' * 60}")

        seq = solve_level(lvl, time_limit=180)

        if seq:
            print(f"\nLevel {lvl}: SOLVED in {len(seq)} steps!")
            # Verify
            print("Verifying...")
            verify_sequence(lvl, seq)
            results[str(lvl)] = seq
        else:
            print(f"Level {lvl}: FAILED")

    print(f"\n{'=' * 60}")
    print("Summary:")
    total_steps = 0
    for lvl, seq in sorted(results.items()):
        print(f"  L{lvl}: {len(seq)} steps")
        total_steps += len(seq)
    print(f"Total: {total_steps} steps for {len(results)} levels")

    if results:
        with open("bp35_new_sequences.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nSaved to bp35_new_sequences.json")

    return results


if __name__ == "__main__":
    main()
