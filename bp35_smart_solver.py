"""
bp35 Mechanism-Aware Solver — Uses game mechanic understanding to
find paths rather than blind BFS/DFS.

For each level, we:
1. Read the grid map (sprite positions and tags)
2. Identify the player start and Gem (target) positions
3. Plan a path using game mechanics:
   - LEFT/RIGHT movement + gravity falling
   - Click to remove/trigger distant objects (terrain modification)
   - Gravity-flip (lrpkmzabbfa) to reverse gravity
   - Clickable jump platforms (qclfkhjnaac etc.) for teleportation
4. Try the planned sequence via Arcade env.step()
"""

import json
import time
from arc_agi import Arcade, OperationMode
from arcengine.enums import GameState

ORACLE_FILE = "oracle_all_levels_expanded.json"


def read_level_map(game, target_level):
    """Read grid map for a level."""
    engine = game.oztjzzyqoek
    world = engine.hdnrlfmyrj

    # Get grid bounds
    max_x, max_y = 0, 0
    for x in range(40):
        for y in range(40):
            if world.jhzcxkveiw(x, y):
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    # Build tag -> position mapping
    special_objects = {}
    for tag in ["fjlzdjxhant", "qclfkhjnaac", "etlsaqqtjvn",
                "yuuqpmlxorv", "lrpkmzabbfa", "ubhhgljbnpu",
                "hzusueifitk", "oonshderxef"]:
        try:
            sprites = world.wwkbcxznzg(tag)
            for s in sprites:
                if tag not in special_objects:
                    special_objects[tag] = []
                special_objects[tag].append((s.grid_x, s.grid_y))
        except:
            pass

    # Player position
    player = engine.twdpowducb
    player_pos = player.qumspquyus

    # Gravity direction
    gravity = engine.vivnprldht  # True = down, False = up

    # Wall positions (blocked)
    walls = set()
    for x in range(max_x + 1):
        for y in range(max_y + 1):
            sprites = world.jhzcxkveiw(x, y)
            if sprites:
                names = [s.name for s in sprites]
                if "xcjjwqfzjfe" in names:
                    walls.add((x, y))

    return {
        "max_x": max_x, "max_y": max_y,
        "player": player_pos,
        "gravity": gravity,
        "special": special_objects,
        "walls": walls,
    }


def simulate_fall(map_data, pos, gravity):
    """Simulate gravity fall from a position. Returns final position."""
    px, py = pos
    dy = 1 if gravity else -1  # True=down, False=up

    # Check what's at the next position in gravity direction
    next_y = py + dy
    while True:
        if (px, next_y) in map_data["walls"]:
            # Hit wall - stop
            return (px, py)
        if (px, next_y) not in map_data["walls"]:
            # Empty or passable - continue falling
            # But check for special objects
            special_tags = []
            for tag, positions in map_data["special"].items():
                if (px, next_y) in positions:
                    special_tags.append(tag)

            if "fjlzdjxhant" in special_tags:
                return "WIN"
            if "ubhhgljbnpu" in special_tags or "hzusueifitk" in special_tags:
                return "LOSE"

            # Continue falling
            py = next_y
            next_y = py + dy
            # Safety: stop if out of bounds
            if next_y < 0 or next_y > map_data["max_y"]:
                return (px, py)

    return (px, py)


def try_sequence(env, game, target_level, sequence):
    """Try an action sequence on a level. Returns (won, steps, final_state)."""
    obs = env.reset()
    game.set_level(target_level)

    for i, action in enumerate(sequence):
        if isinstance(action, list):
            obs = env.step(6, {"x": action[0], "y": action[1]})
        else:
            obs = env.step(action)

        if obs.state == GameState.WIN:
            return (True, i + 1, "WIN")
        if obs.state == GameState.GAME_OVER:
            return (False, i + 1, "GAME_OVER")

    return (False, len(sequence), str(obs.state))


def solve_level_smart(target_level, time_limit=180):
    """
    Smart solver: read map, plan path, try sequence.
    Uses mechanism understanding to guide search.
    """
    start_time = time.time()

    arcade = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("bp35")
    obs = env.reset()
    game = env._game
    game.set_level(target_level)

    map_data = read_level_map(game, target_level)
    gem_pos = map_data["special"].get("fjlzdjxhant", [])
    player_start = map_data["player"]
    gravity = map_data["gravity"]

    print(f"L{target_level}: player=({player_start[0]},{player_start[1]}), "
          f"gem={gem_pos}, grav={gravity}")
    print(f"  Walls: {len(map_data['walls'])} cells")
    print(f"  Clickables: {sum(len(v) for v in map_data['special'].values())} objects")

    # Strategy: use DFS with deepcopy backtracking but with smart pruning
    # Priority: prefer moves toward the Gem direction
    import copy

    if target_level <= 6:
        max_steps = 63
    else:
        max_steps = 127

    visited = set()

    def get_key():
        engine = game.oztjzzyqoek
        px, py = engine.twdpowducb.qumspquyus
        gravity = engine.vivnprldht
        world = engine.hdnrlfmyrj
        clickables = []
        for tag in ["qclfkhjnaac", "etlsaqqtjvn", "yuuqpmlxorv", "lrpkmzabbfa"]:
            try:
                for s in world.wwkbcxznzg(tag):
                    clickables.append((s.grid_x, s.grid_y, tag))
            except:
                pass
        clickables.sort()
        return (px, py, gravity, tuple(clickables))

    def get_click_actions():
        engine = game.oztjzzyqoek
        world = engine.hdnrlfmyrj
        camera_y = engine.camera.rczgvgfsfb[1]
        actions = []
        for tag in ["lrpkmzabbfa", "qclfkhjnaac", "etlsaqqtjvn",
                    "yuuqpmlxorv", "oonshderxef"]:
            try:
                for s in world.wwkbcxznzg(tag):
                    gx, gy = s.grid_x, s.grid_y
                    dx, dy = gx * 6, gy * 6 - camera_y
                    actions.append((6, {"x": dx, "y": dy}, [dx, dy], tag))
            except:
                pass
        return actions

    best = [None]

    def dfs(depth, max_depth):
        if time.time() - start_time > time_limit:
            return False
        if game.hbqwwgceeqp >= max_depth:
            return False

        key = get_key()
        if key in visited:
            return False
        visited.add(key)

        # Enumerate actions with priority toward Gem
        candidates = []

        # Keyboard moves
        engine = game.oztjzzyqoek
        px, py = engine.twdpowducb.qumspquyus
        if gem_pos:
            gx, gy = gem_pos[0]
            # Prefer moves toward gem
            if px < gx:
                candidates.append((4, None, 4, 1))  # RIGHT preferred
                candidates.append((3, None, 3, 2))  # LEFT
            elif px > gx:
                candidates.append((3, None, 3, 1))  # LEFT preferred
                candidates.append((4, None, 4, 2))  # RIGHT
            else:
                candidates.append((4, None, 4, 1))
                candidates.append((3, None, 3, 1))
        else:
            candidates.append((4, None, 4, 1))
            candidates.append((3, None, 3, 2))

        # Click actions (gravity-flip first, then others)
        click_actions = get_click_actions()
        for act_id, act_data, record, tag in click_actions:
            priority = 1 if tag == "lrpkmzabbfa" else 3
            candidates.append((act_id, act_data, record, priority))

        # Sort by priority (lower = explored first)
        candidates.sort(key=lambda x: x[3])

        for act_id, act_data, record, _ in candidates:
            saved_engine = copy.deepcopy(game.oztjzzyqoek)
            saved_steps = game.hbqwwgceeqp

            if act_data:
                obs = env.step(act_id, act_data)
            else:
                obs = env.step(act_id)

            if obs.state == GameState.WIN:
                best[0] = [record]
                return True

            if obs.state == GameState.NOT_FINISHED:
                if dfs(depth + 1, max_depth):
                    best[0] = [record] + best[0]
                    return True

            # Restore
            game.oztjzzyqoek = saved_engine
            game.hbqwwgceeqp = saved_steps

        visited.discard(key)
        return False

    # IDA* with increasing depth
    for depth in range(10, max_steps + 1, 10):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        visited.clear()
        print(f"  IDA* depth={depth}, elapsed={elapsed:.1f}s")
        if dfs(0, depth):
            elapsed = time.time() - start_time
            print(f"  SOLVED: {len(best[0])} steps, {elapsed:.1f}s")
            return best[0]

    elapsed = time.time() - start_time
    print(f"  FAILED: {elapsed:.1f}s, visited={len(visited)}")
    return None


def verify_sequence(target_level, sequence):
    """Verify sequence works correctly."""
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
            print(f"  WIN at step {i+1}")
            return True
        if obs.state == GameState.GAME_OVER:
            print(f"  GAME_OVER at step {i+1}")
            return False

    print(f"  Ended without WIN: {obs.state}")
    return False


def main():
    oracle_data = json.load(open(ORACLE_FILE))
    results = {}

    for lvl in [3, 4, 5, 6, 7, 8]:
        print(f"\n{'='*60}")
        print(f"Solving bp35 Level {lvl}...")
        seq = solve_level_smart(lvl, time_limit=180)

        if seq:
            print(f"Verifying L{lvl}...")
            ok = verify_sequence(lvl, seq)
            if ok:
                results[str(lvl)] = seq
                print(f"L{lvl}: VERIFIED {len(seq)} steps")
            else:
                print(f"L{lvl}: Verification FAILED")
        else:
            print(f"L{lvl}: No solution")

    print(f"\n{'='*60}")
    for lvl, seq in sorted(results.items()):
        print(f"  L{lvl}: {len(seq)} steps")

    if results:
        with open("bp35_new_sequences.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
