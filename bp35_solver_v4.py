"""
bp35 Solver v4 — Smart DFS with game-mechanic-aware action pruning.

Key insights:
1. Any clickable object can be REMOVED remotely via click (changes terrain)
2. Only nearby clicks trigger player jumps (pbsitubcfd proximity check)
3. Gravity-flip (lrpkmzabbfa) can be clicked from anywhere + teleport
4. After each step, gravity auto-resolves (falling)
5. Step limits: L0-6: 63 max, L7-8: 127 max

Strategy: DFS with backtracking (deepcopy save/restore).
For each state, enumerate:
  - LEFT/RIGHT keyboard moves
  - CLICK on every remaining clickable object (remote removal)
State key = (player_x, player_y, gravity, remaining_clickables_set)
"""

import json
import time
import copy
from arc_agi import Arcade, OperationMode
from arcengine.enums import GameState

ORACLE_FILE = "oracle_all_levels_expanded.json"


def get_state_key(game):
    """Compact state key for visited set."""
    engine = game.oztjzzyqoek
    px, py = engine.twdpowducb.qumspquyus
    gravity = engine.vivnprldht

    clickables = []
    world = engine.hdnrlfmyrj
    for tag in ["qclfkhjnaac", "etlsaqqtjvn", "yuuqpmlxorv", "lrpkmzabbfa"]:
        try:
            for s in world.wwkbcxznzg(tag):
                clickables.append((s.grid_x, s.grid_y, tag))
        except:
            pass
    clickables.sort()
    return (px, py, gravity, tuple(clickables))


def get_click_actions(game):
    """Get all click actions with display coords."""
    engine = game.oztjzzyqoek
    world = engine.hdnrlfmyrj
    camera_y = engine.camera.rczgvgfsfb[1]

    actions = []
    for tag in ["qclfkhjnaac", "etlsaqqtjvn", "yuuqpmlxorv", "lrpkmzabbfa", "oonshderxef"]:
        try:
            for s in world.wwkbcxznzg(tag):
                gx, gy = s.grid_x, s.grid_y
                dx = gx * 6
                dy = gy * 6 - camera_y
                actions.append(([dx, dy], tag, gx, gy))
        except:
            pass
    return actions


def solve_level(target_level, time_limit=180):
    """DFS solver for a single bp35 level."""
    start_time = time.time()

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
    print(f"L{target_level}: start ({initial_key[0]},{initial_key[1]}), "
          f"grav={initial_key[2]}, max={max_steps}")

    visited = set()
    best_result = [None]  # mutable container for closure

    def dfs(depth):
        if time.time() - start_time > time_limit:
            return False

        if game.hbqwwgceeqp >= max_steps:
            return False

        key = get_state_key(game)
        if key in visited:
            return False

        visited.add(key)

        # Get candidate actions
        candidates = [(3, None), (4, None)]  # LEFT, RIGHT

        click_actions = get_click_actions(game)
        for click_coords, tag, gx, gy in click_actions:
            # click_coords is [dx, dy] list; env.step needs dict {"x": dx, "y": dy}
            candidates.append((6, {"x": click_coords[0], "y": click_coords[1]}))

        for action_id, action_data in candidates:
            # Save state
            saved_engine = copy.deepcopy(game.oztjzzyqoek)
            saved_steps = game.hbqwwgceeqp

            # Execute action
            if action_data:
                obs = env.step(action_id, action_data)
                # Store as [x, y] list for Oracle format
                action_record = [action_data["x"], action_data["y"]]
            else:
                obs = env.step(action_id)
                action_record = action_id

            if obs.state == GameState.WIN:
                best_result[0] = [action_record]
                return True

            if obs.state == GameState.NOT_FINISHED:
                if dfs(depth + 1):
                    best_result[0] = [action_record] + best_result[0]
                    return True

            # Restore state
            game.oztjzzyqoek = saved_engine
            game.hbqwwgceeqp = saved_steps

        visited.discard(key)
        return False

    # Run DFS
    success = dfs(0)

    elapsed = time.time() - start_time
    if success and best_result[0]:
        print(f"  SOLVED in {len(best_result[0])} steps, "
              f"{elapsed:.1f}s, visited={len(visited)}")
        return best_result[0]
    else:
        print(f"  FAILED after {elapsed:.1f}s, visited={len(visited)}")
        return None


def verify_sequence(target_level, sequence):
    """Verify a winning sequence works correctly from level start."""
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

        if obs.state == GameState.WIN:
            print(f"  Verified WIN at step {i+1}")
            return True
        if obs.state == GameState.GAME_OVER:
            print(f"  GAME_OVER at step {i+1}")
            return False

    print(f"  Sequence ended without WIN (state={obs.state})")
    return False


def main():
    oracle_data = json.load(open(ORACLE_FILE))

    levels_to_solve = [3, 4, 5, 6, 7, 8]
    results = {}

    for lvl in levels_to_solve:
        print(f"\n{'='*60}")
        print(f"Solving bp35 Level {lvl}...")
        print(f"{'='*60}")

        seq = solve_level(lvl, time_limit=180)

        if seq:
            print(f"\nVerifying Level {lvl}...")
            ok = verify_sequence(lvl, seq)
            if ok:
                results[str(lvl)] = seq
                print(f"Level {lvl}: VERIFIED, {len(seq)} steps")
            else:
                print(f"Level {lvl}: Verification FAILED!")
        else:
            print(f"Level {lvl}: No solution found")

    print(f"\n{'='*60}")
    print("Summary:")
    for lvl, seq in sorted(results.items()):
        print(f"  L{lvl}: {len(seq)} steps")

    if results:
        with open("bp35_new_sequences.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} level sequences")

    return results


if __name__ == "__main__":
    main()
