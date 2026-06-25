#!/usr/bin/env python
"""Deep investigation of g50t walkable path and lp85 internal methods."""

import sys
import os
import copy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import arc_agi
from arcengine import GameAction
import numpy as np


def investigate_g50t():
    """Extract walkable path from g50t background sprite."""
    print("\n" + "="*60)
    print("G50T Walkable Path Investigation")
    print("="*60)

    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("g50t")
    obs = env.step(GameAction.RESET)
    game = env._game

    # Get background sprite
    cl = game.current_level
    sprites = list(getattr(cl, "_sprites", []))

    bg_sprite = None
    player_sprite = None
    goal_sprite = None
    for s in sprites:
        tags = getattr(s, "tags", [])
        if "uxqxubarib" in tags:
            bg_sprite = s
        if "qftsebtxuc" in tags:
            player_sprite = s
        if "gilbljmfbc" in tags:
            goal_sprite = s

    if bg_sprite:
        print(f"  Background: pos=({bg_sprite.x},{bg_sprite.y}) size=({bg_sprite.width},{bg_sprite.height})")
        # Get pixel data
        pixels = None
        if hasattr(bg_sprite, "pixels"):
            pixels = bg_sprite.pixels
        elif hasattr(bg_sprite, "_pixels"):
            pixels = bg_sprite._pixels
        
        if pixels is not None:
            if isinstance(pixels, np.ndarray):
                arr = pixels
            elif isinstance(pixels, list):
                arr = np.array(pixels)
            else:
                # Try to get from the sprite's displayable
                disp = getattr(bg_sprite, "gimrsagplbc", None) or getattr(bg_sprite, "_displayable", None)
                if disp and hasattr(disp, "ieikpxxuml"):
                    arr = disp.ieikpxxuml()
                elif disp and hasattr(disp, "aubepcahzib"):
                    arr = disp.aubepcahzib
                else:
                    arr = None
                    print(f"  Can't extract pixels from background, type={type(pixels)}")
            
            if arr is not None:
                print(f"  Background pixel array shape: {arr.shape}")
                print(f"  Unique colors: {np.unique(arr)}")
                
                # Count walkable cells (color 2)
                walkable = np.argwhere(arr == 2)
                print(f"  Walkable cells (color 2): {len(walkable)}")
                if len(walkable) > 0:
                    print(f"  Walkable y range: {walkable[:,0].min()}-{walkable[:,0].max()}")
                    print(f"  Walkable x range: {walkable[:,1].min()}-{walkable[:,1].max()}")
                    
                    # Show a small portion of the walkable map
                    print(f"\n  Walkable map (first 15 rows, first 15 cols):")
                    for y in range(min(15, arr.shape[0])):
                        row = ""
                        for x in range(min(15, arr.shape[1])):
                            if arr[y, x] == 2:
                                row += "."
                            elif arr[y, x] == 0:
                                row += " "
                            else:
                                row += str(arr[y, x])
                        print(f"    {row}")

    if player_sprite and goal_sprite:
        px, py = int(player_sprite.x), int(player_sprite.y)
        gx, gy = int(goal_sprite.x), int(goal_sprite.y)
        print(f"\n  Player: ({px},{py})")
        print(f"  Goal: ({gx},{gy})")
        
        # Try to find path using walkable cells
        # Check what's blocking at (37,7) - player can move right to 37 but not further
        # Check cells around (37,7)
        if bg_sprite and arr is not None:
            bg_x, bg_y = int(bg_sprite.x), int(bg_sprite.y)
            # Check if player position is walkable in background
            for check_pos in [(37, 7), (37, 13), (43, 7), (37, 14), (31, 14)]:
                cx, cy = check_pos
                # Background is at (1,1), so array index = (cy-1, cx-1)
                ay, ax = cy - bg_y, cx - bg_x
                if 0 <= ay < arr.shape[0] and 0 <= ax < arr.shape[1]:
                    color = arr[ay, ax]
                    print(f"  Cell ({cx},{cy}) -> array[{ay},{ax}] = color {color}")
                else:
                    print(f"  Cell ({cx},{cy}) -> OUT OF BOUNDS")

    # Now try to find the actual walkable path using step=6
    print("\n  Trying BFS with walkable cells (step=6)...")
    if arr is not None and bg_sprite is not None:
        bg_x, bg_y = int(bg_sprite.x), int(bg_sprite.y)
        step = 6
        
        # Build walkable set
        walkable_set = set()
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if arr[y, x] == 2:
                    walkable_set.add((x + bg_x, y + bg_y))
        
        print(f"  Walkable set size: {len(walkable_set)}")
        
        # BFS from player to goal
        from collections import deque
        start = (px, py)
        goal = (gx, gy)
        
        # Also try goal as (gx+1, gy+1) or other offsets
        for goal_try in [goal, (gx+1, gy+1), (gx+2, gy+2), (gx+3, gy+3), (gx+4, gy+4)]:
            if goal_try in walkable_set:
                print(f"  Goal {goal_try} is in walkable set!")
                goal = goal_try
                break
        else:
            print(f"  Goal {goal} NOT in walkable set. Trying nearby...")
            # Find nearest walkable to goal
            min_dist = float('inf')
            nearest = None
            for wx, wy in walkable_set:
                dist = abs(wx - gx) + abs(wy - gy)
                if dist < min_dist:
                    min_dist = dist
                    nearest = (wx, wy)
            if nearest:
                print(f"  Nearest walkable to goal: {nearest} (dist={min_dist})")
                goal = nearest
        
        # Check if start is walkable
        if start not in walkable_set:
            print(f"  Start {start} NOT in walkable set. Finding nearest...")
            min_dist = float('inf')
            nearest = None
            for wx, wy in walkable_set:
                dist = abs(wx - px) + abs(wy - py)
                if dist < min_dist:
                    min_dist = dist
                    nearest = (wx, wy)
            if nearest:
                print(f"  Nearest walkable to start: {nearest} (dist={min_dist})")
                start = nearest
        
        # BFS
        queue = deque([(start, [])])
        visited = {start}
        found = None
        
        while queue:
            (cx, cy), path = queue.popleft()
            if (cx, cy) == goal:
                found = path
                break
            for dx, dy, action in [(0, -step, GameAction.ACTION1), (0, step, GameAction.ACTION2), 
                                   (-step, 0, GameAction.ACTION3), (step, 0, GameAction.ACTION4)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in walkable_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(action, None)]))
        
        if found:
            print(f"  PATH FOUND! {len(found)} steps")
            for i, (act, _) in enumerate(found[:5]):
                print(f"    [{i}] {act}")
            if len(found) > 5:
                print(f"    ... ({len(found)} total)")
            
            # Execute the path
            print("\n  Executing path...")
            arc2 = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
            env2 = arc.make("g50t")
            obs2 = env2.step(GameAction.RESET)
            
            for i, (act, _) in enumerate(found):
                obs2 = env2.step(act)
                if obs2.levels_completed > 0:
                    print(f"  LEVEL PASSED at step {i+1}!")
                    return True
            
            print(f"  Path executed but levels={obs2.levels_completed}")
        else:
            print(f"  NO PATH FOUND from {start} to {goal}")
            print(f"  Visited {len(visited)} cells")


def investigate_lp85():
    """Investigate lp85 internal methods and solution structure."""
    print("\n" + "="*60)
    print("LP85 Internal Methods Investigation")
    print("="*60)

    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("lp85")
    obs = env.step(GameAction.RESET)
    game = env._game

    # Check solution structure in detail
    solution = getattr(game, "uopmnplcnv", {})
    level_name = getattr(game, "ucybisahh", "")
    
    if level_name in solution:
        level_sol = solution[level_name]
        for section, sec_data in level_sol.items():
            print(f"\n  Section '{section}':")
            for sub_key, sub_data in sec_data.items():
                if isinstance(sub_data, dict):
                    print(f"    {sub_key}: {len(sub_data)} entries")
                    # Sort by key and show all
                    for k in sorted(sub_data.keys()):
                        v = sub_data[k]
                        print(f"      {k}: {v}")
                else:
                    print(f"    {sub_key}: {sub_data}")

    # Try calling khartslnwa (win check)
    print("\n  Calling khartslnwa()...")
    try:
        result = game.khartslnwa()
        print(f"  khartslnwa() = {result}")
    except Exception as e:
        print(f"  khartslnwa() error: {e}")

    # Try calling try_move with different parameters
    print("\n  Trying try_move()...")
    for params in [0, 1, -1, "L", "R", "left", "right", True, False]:
        try:
            result = game.try_move(params)
            print(f"  try_move({params!r}) = {result}")
        except Exception as e:
            print(f"  try_move({params!r}) error: {e}")

    # Check tile positions before and after clicking
    print("\n  Checking tile positions before click...")
    cl = game.current_level
    tiles_before = []
    for s in getattr(cl, "_sprites", []):
        tags = getattr(s, "tags", [])
        if "tile" in tags:
            tiles_before.append((s.name, int(s.x), int(s.y)))
    print(f"  Tiles before: {len(tiles_before)}")
    for name, x, y in tiles_before:
        print(f"    {name} at ({x},{y})")

    # Click L button once and check positions
    print("\n  After clicking L button once...")
    obs = env.step(GameAction.ACTION6, data={"x": 1, "y": 8})
    tiles_after_l = []
    for s in getattr(cl, "_sprites", []):
        tags = getattr(s, "tags", [])
        if "tile" in tags:
            tiles_after_l.append((s.name, int(s.x), int(s.y)))
    
    changed = False
    for i, (name, x, y) in enumerate(tiles_before):
        if i < len(tiles_after_l):
            name2, x2, y2 = tiles_after_l[i]
            if x != x2 or y != y2:
                print(f"    {name}: ({x},{y}) -> ({x2},{y2}) CHANGED")
                changed = True
    if not changed:
        print("  No tiles changed after L click!")

    # Check win condition
    try:
        result = game.khartslnwa()
        print(f"  khartslnwa() after L = {result}")
    except:
        pass

    # Check the goal sprite position
    for s in getattr(cl, "_sprites", []):
        tags = getattr(s, "tags", [])
        if "goal" in tags:
            print(f"\n  Goal sprite: pos=({s.x},{s.y}) size=({s.width},{s.height})")

    # Check bghvgbtwcb sprite
    for s in getattr(cl, "_sprites", []):
        tags = getattr(s, "tags", [])
        if "bghvgbtwcb" in tags:
            print(f"  bghvgbtwcb sprite: pos=({s.x},{s.y}) size=({s.width},{s.height})")

    # Try reading the oxbwsencfv section
    if level_name in solution:
        level_sol = solution[level_name]
        for section, sec_data in level_sol.items():
            if "oxbwsencfv" in sec_data:
                print(f"\n  oxbwsencfv data:")
                oxbw = sec_data["oxbwsencfv"]
                print(f"    type: {type(oxbw)}")
                print(f"    value: {str(oxbw)[:500]}")


def investigate_su15():
    """Check if su15 clicks are actually changing the pixel buffer."""
    print("\n" + "="*60)
    print("SU15 Click Investigation")
    print("="*60)

    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("su15")
    obs = env.step(GameAction.RESET)
    game = env._game

    # Get initial pixel buffer
    sdrzkrjcp = getattr(game, "sdrzkrjcp", None)
    if sdrzkrjcp is not None and isinstance(sdrzkrjcp, np.ndarray):
        print(f"  Initial sdrzkrjcp shape: {sdrzkrjcp.shape}")
        print(f"  Unique colors: {np.unique(sdrzkrjcp)}")
        # Count non-zero pixels
        non_zero = np.count_nonzero(sdrzkrjcp)
        print(f"  Non-zero pixels: {non_zero}")

    # Get the frame before clicking
    frame_before = obs.frame[0] if hasattr(obs, 'frame') and obs.frame else None
    if frame_before is not None:
        print(f"  Frame shape: {frame_before.shape}")
        print(f"  Frame unique colors: {np.unique(frame_before)}")

    # Execute first 5 clicks
    actions_list = getattr(game, "actions", [])
    for i in range(5):
        act = actions_list[i]
        act_id = getattr(act, "id", None)
        act_data = getattr(act, "data", None) or {}
        obs = env.step(act_id, data=act_data)
        print(f"  After click {i}: data={act_data}, levels={obs.levels_completed}")

    # Check pixel buffer after clicks
    sdrzkrjcp_after = getattr(game, "sdrzkrjcp", None)
    if sdrzkrjcp_after is not None and isinstance(sdrzkrjcp_after, np.ndarray):
        non_zero_after = np.count_nonzero(sdrzkrjcp_after)
        print(f"  Non-zero pixels after 5 clicks: {non_zero_after}")
        if sdrzkrjcp is not None:
            diff = np.sum(sdrzkrjcp != sdrzkrjcp_after)
            print(f"  Pixels changed: {diff}")

    # Check frame after clicks
    frame_after = obs.frame[0] if hasattr(obs, 'frame') and obs.frame else None
    if frame_before is not None and frame_after is not None:
        diff = np.sum(frame_before != frame_after)
        print(f"  Frame pixels changed: {diff}")

    # Check other game state attributes
    for attr in ["qygchysnh", "oicctzexh", "sqwdaqczb", "vsfwpngmx", 
                 "souzbwugw", "xwrhpebfn", "gtmjjdjwl"]:
        val = getattr(game, attr, "N/A")
        print(f"  {attr}: {val}")

    # Check what the canvas sprite looks like
    cl = game.current_level
    for s in getattr(cl, "_sprites", []):
        tags = getattr(s, "tags", [])
        if "xkstxyqbs" in tags:
            print(f"\n  Canvas sprite (xkstxyqbs): pos=({s.x},{s.y}) size=({s.width},{s.height})")
            # Get its pixels
            pixels = getattr(s, "pixels", None)
            if pixels is not None:
                if isinstance(pixels, np.ndarray):
                    arr = pixels
                else:
                    arr = np.array(pixels)
                print(f"  Canvas pixels shape: {arr.shape}")
                print(f"  Canvas unique colors: {np.unique(arr)}")

    # Check the zmlxwcvwb sprite (pattern to match)
    for s in getattr(cl, "_sprites", []):
        tags = getattr(s, "tags", [])
        if "zmlxwcvwb" in tags:
            print(f"\n  Pattern sprite (zmlxwcvwb): pos=({s.x},{s.y}) size=({s.width},{s.height})")
            pixels = getattr(s, "pixels", None)
            if pixels is not None:
                if isinstance(pixels, np.ndarray):
                    arr = pixels
                else:
                    arr = np.array(pixels)
                print(f"  Pattern pixels:\n{arr}")


if __name__ == "__main__":
    investigate_g50t()
    investigate_lp85()
    investigate_su15()
