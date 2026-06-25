"""Batch dump all 22 unsolved games' structure.

Outputs a compact summary of each game's:
- Sprite layout (tags, positions, sizes)
- Key game attributes
- Available actions
- Win condition hints
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
env_files = os.path.join(project_root, "environment_files")
if os.path.exists(env_files):
    sys.path.insert(0, env_files)


def dump_game_compact(game_id: str):
    """Dump a single game's structure in compact format."""
    import arc_agi
    from arcengine import GameAction

    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    obs = env.step(GameAction.RESET)

    game = getattr(env, "_game", None)
    if game is None:
        print(f"\n{'='*60}")
        print(f"[{game_id}] NO _game found!")
        return

    print(f"\n{'='*60}")
    print(f"[{game_id}] type={type(game).__name__}")

    # Get current level
    cl = getattr(game, "current_level", None)
    if cl is None:
        print(f"  NO current_level!")
        return

    # Dump sprites
    sprites = getattr(cl, "_sprites", [])
    print(f"  _sprites: {len(sprites)}")
    for i, s in enumerate(sprites):
        x = getattr(s, "x", "?")
        y = getattr(s, "y", "?")
        w = getattr(s, "width", "?")
        h = getattr(s, "height", "?")
        name = getattr(s, "name", "?")
        tags = getattr(s, "tags", [])
        # Try to get pixels summary
        pixels = getattr(s, "pixels", None)
        px_info = ""
        if pixels is not None:
            try:
                import numpy as np
                arr = np.array(pixels)
                unique_colors = sorted(set(arr.flatten().tolist()))[:8]
                px_info = f" colors={unique_colors}"
            except Exception:
                px_info = f" pixels_shape={getattr(pixels, 'shape', '?')}"
        print(f"    [{i}] ({x},{y}) {w}x{h} name={name} tags={tags}{px_info}")

    # Dump sprite_lists
    sl = getattr(game, "sprite_lists", None)
    if isinstance(sl, dict) and sl:
        print(f"  sprite_lists: {len(sl)} keys")
        for key, val in sl.items():
            if isinstance(val, list):
                print(f"    '{key}': list[{len(val)}]")

    # Dump key non-private, non-callable attributes of game
    print(f"  game attrs:")
    for attr in sorted(dir(game)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(game, attr)
            if callable(val) and not isinstance(val, (list, dict, set, tuple)):
                continue
            if isinstance(val, (list, tuple)):
                print(f"    {attr}: {type(val).__name__}[{len(val)}]")
            elif isinstance(val, dict):
                print(f"    {attr}: dict[{len(val)}] keys={list(val.keys())[:5]}")
            elif isinstance(val, set):
                print(f"    {attr}: set[{len(val)}]")
            elif isinstance(val, (int, float, str, bool, type(None))):
                print(f"    {attr}: {type(val).__name__} = {val}")
            else:
                print(f"    {attr}: {type(val).__name__}")
        except Exception:
            pass

    # Dump key attributes of current_level
    print(f"  level attrs:")
    for attr in sorted(dir(cl)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(cl, attr)
            if callable(val) and not isinstance(val, (list, dict, set, tuple)):
                continue
            if isinstance(val, (list, tuple)):
                print(f"    {attr}: {type(val).__name__}[{len(val)}]")
            elif isinstance(val, dict):
                print(f"    {attr}: dict[{len(val)}] keys={list(val.keys())[:5]}")
            elif isinstance(val, set):
                print(f"    {attr}: set[{len(val)}]")
            elif isinstance(val, (int, float, str, bool, type(None))):
                print(f"    {attr}: {type(val).__name__} = {val}")
            else:
                print(f"    {attr}: {type(val).__name__}")
        except Exception:
            pass

    # Available actions
    print(f"  available_actions: {sorted(obs.available_actions)}")
    print(f"  levels_completed: {obs.levels_completed}")


GAMES = [
    "tu93", "re86", "g50t", "wa30",  # keyboard_only
    "vc33", "s5i5", "tn36", "su15", "r11l", "lp85",  # click_only
    "bp35", "dc22", "sk48", "lf52", "sc25",  # kb+click
    "m0r0", "cn04", "cd82", "sp80", "ka59", "ar25", "sb26",  # kb+click
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="all", help="Comma-separated game IDs or 'all'")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--count", type=int, default=22, help="Number of games to dump")
    args = parser.parse_args()

    if args.games == "all":
        games = GAMES[args.start:args.start + args.count]
    else:
        games = args.games.split(",")

    for gid in games:
        try:
            dump_game_compact(gid.strip())
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[{gid.strip()}] ERROR: {e}")
            import traceback
            traceback.print_exc()
