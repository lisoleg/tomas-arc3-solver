"""Solver for G50T: Navigate player to goal.

Uses midpoint pixel check (same as tu93 solver):
- Player sprite: tag='qftsebtxuc', 7x7, step=6
- Background map: compute walkable cells (color==2 pixels)
- BFS: for each potential move, check midpoint pixel (half_step=3 ahead)
  If midpoint pixel color==2, the move is valid.
"""
from __future__ import annotations

from typing import Any, Optional
from collections import deque

import numpy as np

from arcengine import GameAction


# ------------------------------------------------------------
# Helpers (self-contained, no dependencies on game_solvers module)
# ------------------------------------------------------------

def _get_sprites_by_tag(game: Any, tag: str) -> list:
    cl = getattr(game, "current_level", None)
    if cl is None:
        return []
    try:
        sprites = cl.get_sprites_by_tag(tag)
        if sprites:
            return list(sprites)
    except (AttributeError, TypeError):
        pass
    sprites = getattr(cl, "_sprites", [])
    result = []
    for s in sprites:
        s_tags = getattr(s, "tags", [])
        if isinstance(s_tags, (list, tuple)) and tag in s_tags:
            result.append(s)
    return result


def _get_all_sprites(game: Any) -> list:
    cl = getattr(game, "current_level", None)
    if cl is None:
        return []
    return list(getattr(cl, "_sprites", []))


def _sprite_center(s: Any) -> tuple[int, int]:
    x = int(getattr(s, "x", 0))
    y = int(getattr(s, "y", 0))
    w = int(getattr(s, "width", 1))
    h = int(getattr(s, "height", 1))
    return (x + w // 2, y + h // 2)


def _get_background_pixels(game: Any) -> Optional[np.ndarray]:
    """Get the background map sprite's pixel data.
    
    Returns (bg_sprite, pixels_array, bg_x, bg_y) or None.
    """
    cl = getattr(game, "current_level", None)
    if cl is None:
        return None
    for s in getattr(cl, "_sprites", []):
        tags = getattr(s, "tags", [])
        name = getattr(s, "name", "")
        # Background map: large sprite with color 2 pixels
        pixels = getattr(s, "pixels", None)
        if pixels is None:
            continue
        pixels = np.asarray(pixels)
        if pixels.size < 100:  # too small to be background
            continue
        # Check if has color 2
        if 2 in pixels:
            bg_x = int(getattr(s, "x", 0))
            bg_y = int(getattr(s, "y", 0))
            return (s, pixels, bg_x, bg_y)
    return None


def solve_g50t(game: Any, level_idx: int) -> list | None:
    """Solve G50T: Navigate player to goal position.
    
    Strategy:
    1. Get player center and goal center
    2. Build walkable set from background pixels (color==2)
    3. BFS with midpoint check: for each move (dx,dy)*step,
       check the midpoint pixel (current + (dx,dy)*half_step).
       If midpoint is walkable (color==2), allow the move.
    4. Convert BFS path to actions.
    """
    players = _get_sprites_by_tag(game, "qftsebtxuc")
    goals = _get_sprites_by_tag(game, "gilbljmfbc")

    if not players or not goals:
        return None

    step = 6
    half_step = 3
    player_pos = _sprite_center(players[0])
    goal_pos = _sprite_center(goals[0])

    # Get background pixels for walkable check
    bg_result = _get_background_pixels(game)
    if bg_result is None:
        return None  # can't compute path without background
    bg_sprite, bg_pixels, bg_x, bg_y = bg_result

    def is_walkable(px: int, py: int) -> bool:
        """Check if (px,py) is on a walkable pixel (color==2)."""
        # Convert player-centered grid position to background local pixel
        # Background is at (bg_x, bg_y), pixels is (height, width)
        local_x = px - bg_x
        local_y = py - bg_y
        if 0 <= local_y < bg_pixels.shape[0] and 0 <= local_x < bg_pixels.shape[1]:
            return int(bg_pixels[local_y, local_x]) == 2
        return False

    def is_valid_move(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> bool:
        """Check if moving from from_pos to to_pos is valid.
        
        Uses midpoint check: the pixel at (from + (dx,dy)*half_step) must be walkable.
        """
        fx, fy = from_pos
        tx, ty = to_pos
        # Midpoint = from + direction * half_step
        mx = fx + (tx - fx) // 2  # = fx + dx*3 (since full step=6)
        my = fy + (ty - fy) // 2
        return is_walkable(mx, my)

    # BFS on grid with step=6
    directions = [(0, -step), (0, step), (-step, 0), (step, 0)]

    visited = {player_pos}
    queue = deque([(player_pos, [])])

    for _ in range(10000):
        if not queue:
            return None  # no path found
        (cx, cy), path = queue.popleft()

        if (cx, cy) == goal_pos:
            # Found path!
            actions = []
            full_path = [player_pos] + path
            for i in range(1, len(full_path)):
                px, py = full_path[i - 1]
                nx, ny = full_path[i]
                dx, dy = nx - px, ny - py
                if dy < 0:
                    actions.append((GameAction.ACTION1, None))  # UP
                elif dy > 0:
                    actions.append((GameAction.ACTION2, None))  # DOWN
                elif dx < 0:
                    actions.append((GameAction.ACTION3, None))  # LEFT
                elif dx > 0:
                    actions.append((GameAction.ACTION4, None))  # RIGHT
            return actions

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if nx < 0 or ny < 0 or nx >= 64 or ny >= 64:
                continue
            if not is_valid_move((cx, cy), (nx, ny)):
                continue
            visited.add((nx, ny))
            new_path = path + [(nx, ny)]
            queue.append(((nx, ny), new_path))

    return None  # BFS timeout
