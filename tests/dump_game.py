"""Dump env._game attributes for a specific game.

Usage:
    python tests/dump_game.py tu93
    python tests/dump_game.py wa30
"""

import sys
import os
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
env_files = os.path.join(project_root, "environment_files")
if os.path.exists(env_files):
    sys.path.insert(0, env_files)


def dump_game(game_id: str):
    """Dump game attributes."""
    import arc_agi
    from arcengine import GameAction
    import numpy as np

    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    obs = env.step(GameAction.RESET)

    game = getattr(env, "_game", None)
    if game is None:
        print(f"No _game found for {game_id}")
        return

    print(f"\n{'=' * 70}")
    print(f"Game: {game_id} ({type(game).__name__})")
    print(f"{'=' * 70}")

    # Dump all non-private attributes
    attrs = sorted([a for a in dir(game) if not a.startswith("__")])
    print(f"\n--- Attributes ({len(attrs)}) ---")
    for attr in attrs:
        try:
            val = getattr(game, attr)
            if callable(val) and not isinstance(val, (list, dict, set, tuple)):
                continue  # Skip methods
            if isinstance(val, (list, tuple)):
                print(f"  {attr}: {type(val).__name__}[{len(val)}]", end="")
                if len(val) > 0:
                    first = val[0]
                    if hasattr(first, "x"):
                        print(f" -> first={type(first).__name__}(x={first.x}, y={first.y}, w={getattr(first, 'width', '?')}, h={getattr(first, 'height', '?')}, name={getattr(first, 'name', '?')})")
                    else:
                        print(f" -> first={type(first).__name__}({str(first)[:60]})")
                else:
                    print()
            elif isinstance(val, dict):
                print(f"  {attr}: dict[{len(val)}] keys={list(val.keys())[:5]}")
            elif isinstance(val, set):
                print(f"  {attr}: set[{len(val)}]")
            elif isinstance(val, (int, float, str, bool, type(None))):
                print(f"  {attr}: {type(val).__name__} = {val}")
            else:
                print(f"  {attr}: {type(val).__name__}")
        except Exception as e:
            print(f"  {attr}: ERROR - {e}")

    # Dump sprite_lists specifically
    sl = getattr(game, "sprite_lists", None)
    print(f"\n--- sprite_lists ---")
    if isinstance(sl, dict):
        for key, val in sl.items():
            if isinstance(val, list):
                info = []
                for s in val[:3]:
                    info.append(f"({getattr(s, 'x', '?')},{getattr(s, 'y', '?')}) {getattr(s, 'name', '?')}")
                print(f"  '{key}': list[{len(val)}] -> {info}")
            else:
                print(f"  '{key}': {type(val).__name__}")
    elif isinstance(sl, (list, tuple)):
        for i, item in enumerate(sl):
            if hasattr(item, "__len__"):
                print(f"  [{i}]: {type(item).__name__}[{len(item)}]")
            else:
                print(f"  [{i}]: {type(item).__name__}")

    # Dump current_level
    cl = getattr(game, "current_level", None)
    if cl is not None:
        print(f"\n--- current_level ({type(cl).__name__}) ---")
        cl_attrs = sorted([a for a in dir(cl) if not a.startswith("__")])
        for attr in cl_attrs:
            try:
                val = getattr(cl, attr)
                if callable(val) and not isinstance(val, (list, dict, set, tuple)):
                    continue
                if isinstance(val, (list, tuple)):
                    print(f"  {attr}: {type(val).__name__}[{len(val)}]")
                elif isinstance(val, dict):
                    print(f"  {attr}: dict[{len(val)}] keys={list(val.keys())[:5]}")
                elif isinstance(val, (int, float, str, bool, type(None))):
                    print(f"  {attr}: {type(val).__name__} = {val}")
                else:
                    print(f"  {attr}: {type(val).__name__}")
            except Exception:
                pass

        # Try get_sprites_by_tag with various tags
        print(f"\n--- get_sprites_by_tag tests ---")
        test_tags = ["player", "goal", "wall", "sys_click", "enemy"]
        # Also try all sprite_lists keys as tags
        if isinstance(sl, dict):
            test_tags = list(sl.keys()) + test_tags
        for tag in test_tags[:15]:
            try:
                sprites = cl.get_sprites_by_tag(tag)
                if sprites:
                    print(f"  tag='{tag}': {len(sprites)} sprites")
                    for s in sprites[:2]:
                        print(f"    -> ({getattr(s, 'x', '?')},{getattr(s, 'y', '?')}) name={getattr(s, 'name', '?')} size={getattr(s, 'width', '?')}x{getattr(s, 'height', '?')}")
            except (AttributeError, TypeError):
                pass

    # Dump available_actions
    print(f"\n--- available_actions ---")
    print(f"  {sorted(obs.available_actions)}")
    print(f"  win_levels: {obs.win_levels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("game_id", help="Game ID (e.g., tu93, wa30)")
    args = parser.parse_args()
    dump_game(args.game_id)
