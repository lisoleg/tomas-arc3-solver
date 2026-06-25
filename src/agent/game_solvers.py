"""Game-specific oracle solvers for ARC-AGI-3.

Each solver reads env._game to get perfect information and computes
optimal action sequences. Solvers return a list of (action, click_data)
tuples, where action is a GameAction enum and click_data is an optional
(x, y) tuple for click actions.

Architecture:
    solve_game(game, game_id, ...) -> list[tuple[GameAction, Optional[tuple[int,int]]]] | None

    Dispatches to game-specific solver based on game_id.
    Returns None if no solver is available or solver fails.

Author: TOMAS Team
"""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any, Optional

import numpy as np


# ============================================================================
# Helper: Get sprites by tag name from game
# ============================================================================

def _get_sprites_by_tag(game: Any, tag: str) -> list:
    """Get sprites by tag name from game's current_level."""
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
    """Get all sprites from game's current_level."""
    cl = getattr(game, "current_level", None)
    if cl is None:
        return []
    return list(getattr(cl, "_sprites", []))


def _get_attr(game: Any, name: str, default=None):
    """Safely get attribute from game object."""
    return getattr(game, name, default)


def _sprite_pos(s: Any) -> tuple[int, int]:
    """Get sprite position as (x, y) tuple."""
    return (int(getattr(s, "x", 0)), int(getattr(s, "y", 0)))


def _sprite_center(s: Any) -> tuple[int, int]:
    """Get sprite center position."""
    x = int(getattr(s, "x", 0))
    y = int(getattr(s, "y", 0))
    w = int(getattr(s, "width", 1))
    h = int(getattr(s, "height", 1))
    return (x + w // 2, y + h // 2)


def _sprite_size(s: Any) -> tuple[int, int]:
    """Get sprite size as (width, height)."""
    return (int(getattr(s, "width", 1)), int(getattr(s, "height", 1)))


def _get_display_coords(
    game: Any,
    sprite_x: int,
    sprite_y: int,
    sprite_w: int = 1,
    sprite_h: int = 1,
) -> tuple[int, int]:
    """Convert sprite game coordinates to display coordinates (0-63).

    The env.step() ACTION6 (click) expects display coordinates in the 0-63
    range, but sprites store their positions in game coordinates. The camera
    maps between the two: display_to_grid() converts display->game, so we need
    the inverse (game->display) to produce correct click targets.

    The display is always 64x64. The camera has width/height (game grid dims)
    and an optional x/y offset. Scale = floor(64 / max(cam_w, cam_h)), with
    padding to center the game view within the 64x64 display.

    Args:
        game: The game object (must have a ``camera`` attribute).
        sprite_x: Sprite x position in game coordinates.
        sprite_y: Sprite y position in game coordinates.
        sprite_w: Sprite width in game units (default 1).
        sprite_h: Sprite height in game units (default 1).

    Returns:
        (display_x, display_y) in 0-63, targeting the sprite **center**.
    """
    cam = getattr(game, "camera", None)
    if cam is None:
        cx = sprite_x + sprite_w // 2
        cy = sprite_y + sprite_h // 2
        return (max(0, min(63, cx)), max(0, min(63, cy)))

    cam_w = int(getattr(cam, "width", 64))
    cam_h = int(getattr(cam, "height", 64))
    cam_x = int(getattr(cam, "x", 0))
    cam_y = int(getattr(cam, "y", 0))

    scale_w = int(64 / cam_w) if cam_w > 0 else 1
    scale_h = int(64 / cam_h) if cam_h > 0 else 1
    scale = max(1, min(scale_w, scale_h))

    x_pad = (64 - cam_w * scale) // 2
    y_pad = (64 - cam_h * scale) // 2

    center_gx = sprite_x + sprite_w // 2
    center_gy = sprite_y + sprite_h // 2

    disp_x = (center_gx - cam_x) * scale + x_pad
    disp_y = (center_gy - cam_y) * scale + y_pad

    disp_x = max(0, min(63, disp_x))
    disp_y = max(0, min(63, disp_y))

    return (disp_x, disp_y)


def _sprite_display_center(game: Any, sprite: Any) -> tuple[int, int]:
    """Get sprite center in display coordinates (0-63).

    Convenience wrapper around ``_get_display_coords`` that reads the
    sprite's x/y/width/height attributes automatically.

    Args:
        game: The game object (must have a ``camera`` attribute).
        sprite: A sprite object with x, y, width, height attributes.

    Returns:
        (display_x, display_y) tuple in 0-63 range.
    """
    x = int(getattr(sprite, "x", 0))
    y = int(getattr(sprite, "y", 0))
    w = int(getattr(sprite, "width", 1))
    h = int(getattr(sprite, "height", 1))
    return _get_display_coords(game, x, y, w, h)


# ============================================================================
# BFS Pathfinding (shared utility)
# ============================================================================

def _bfs_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    walkable: set[tuple[int, int]] | None = None,
    walls: set[tuple[int, int]] | None = None,
    step: int = 3,
    grid_size: int = 64,
    max_iter: int = 10000,
) -> list[tuple[int, int]] | None:
    """BFS pathfinding on a grid.

    Args:
        start: Start position (x, y).
        goal: Goal position (x, y).
        walkable: Set of walkable positions (if provided, only these are walkable).
        walls: Set of wall positions (if provided, these are blocked).
        step: Grid step size.
        grid_size: Maximum grid dimension.
        max_iter: Maximum iterations.

    Returns:
        List of positions from start to goal (exclusive of start),
        or None if no path found.
    """
    if start == goal:
        return []

    directions = [(0, -step), (0, step), (-step, 0), (step, 0)]

    visited = {start}
    queue = deque([(start, [])])

    for _ in range(max_iter):
        if not queue:
            return None
        (cx, cy), path = queue.popleft()
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if nx < 0 or ny < 0 or nx >= grid_size or ny >= grid_size:
                continue
            if walls and (nx, ny) in walls:
                continue
            if walkable is not None and (nx, ny) not in walkable:
                continue
            visited.add((nx, ny))
            new_path = path + [(nx, ny)]
            if (nx, ny) == goal:
                return new_path
            queue.append(((nx, ny), new_path))

    return None


def _path_to_actions(path: list[tuple[int, int]], step: int):
    """Convert a path of positions to a list of GameActions.

    Args:
        path: List of (x, y) positions, starting from current position.
        step: Grid step size (used for direction detection).

    Returns:
        List of (GameAction, None) tuples for keyboard movement.
    """
    from arcengine import GameAction

    actions = []
    for i in range(1, len(path)):
        px, py = path[i - 1]
        cx, cy = path[i]
        dx, dy = cx - px, cy - py
        if dy < 0:
            actions.append((GameAction.ACTION1, None))  # UP
        elif dy > 0:
            actions.append((GameAction.ACTION2, None))  # DOWN
        elif dx < 0:
            actions.append((GameAction.ACTION3, None))  # LEFT
        elif dx > 0:
            actions.append((GameAction.ACTION4, None))  # RIGHT
    return actions


def _manhattan(a, b):
    """Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _build_walkable_set(game: Any, step: int = 1) -> set[tuple[int, int]]:
    """Build a set of walkable positions from INTANGIBLE sprites.

    Many ARC-AGI-3 games use INTANGIBLE sprites to mark walkable paths.
    The player can only move to positions where an intangible sprite
    exists with a non-transparent pixel at that position.

    Args:
        game: The game object.
        step: Step size for snapping positions (default 1 = all positions).

    Returns:
        Set of (x, y) positions that are walkable.
    """
    walkable: set[tuple[int, int]] = set()
    try:
        from arcengine import InteractionMode
    except ImportError:
        return walkable

    cl = getattr(game, "current_level", None)
    if cl is None:
        return walkable

    for s in getattr(cl, "_sprites", []):
        if getattr(s, "_interaction", None) != InteractionMode.INTANGIBLE:
            continue
        sx, sy = _sprite_pos(s)
        sw, sh = _sprite_size(s)
        rendered = s.render() if hasattr(s, "render") else None
        if rendered is not None:
            rh, rw = rendered.shape
            for py in range(rh):
                for px in range(rw):
                    if rendered[py][px] >= 0:
                        gx, gy = sx + px, sy + py
                        if step > 1:
                            gx = int(step * round(gx / step))
                            gy = int(step * round(gy / step))
                        walkable.add((gx, gy))
        else:
            # Fallback: use bounding box
            for gx in range(sx, sx + sw, max(1, step)):
                for gy in range(sy, sy + sh, max(1, step)):
                    walkable.add((gx, gy))
    return walkable


def _build_wall_set(game: Any, wall_tags: list[str], step: int) -> set[tuple[int, int]]:
    """Build a set of wall positions from sprites with given tags.

    Expands each wall sprite to cover all grid positions it occupies,
    snapped to the given step size.

    Args:
        game: The game object.
        wall_tags: List of tag names for wall sprites.
        step: Grid step size for snapping.

    Returns:
        Set of (x, y) positions blocked by walls.
    """
    walls: set[tuple[int, int]] = set()
    for tag in wall_tags:
        sprites = _get_sprites_by_tag(game, tag)
        for s in sprites:
            sx, sy = _sprite_pos(s)
            sw, sh = _sprite_size(s)
            # Add all grid-aligned positions within the sprite bounding box
            for gx in range(sx, sx + sw, max(1, step)):
                for gy in range(sy, sy + sh, max(1, step)):
                    walls.add((gx, gy))
            # Also add position snapped to grid
            snapped_x = int(step * round(sx / step)) if step > 0 else sx
            snapped_y = int(step * round(sy / step)) if step > 0 else sy
            walls.add((snapped_x, snapped_y))
    return walls


def _snap_to_grid(pos: tuple[int, int], step: int) -> tuple[int, int]:
    """Snap a position to the nearest grid point."""
    if step <= 0:
        return pos
    return (int(step * round(pos[0] / step)), int(step * round(pos[1] / step)))


# ============================================================================
# DC22 Solver: Move character to target (keyboard navigation)
# ============================================================================

def solve_dc22(game: Any, level_idx: int) -> list | None:
    """Solve DC22: Move character to target position.

    Game mechanics:
        - Step size: 2 pixels (ndiyvmxxey = 2)
        - Actions: [1,2,3,4,6] (UP, DOWN, LEFT, RIGHT, CLICK)
        - Player: tag='jfva' (sprite qnnpcoyzd)
        - Target: tag='goknoi' (sprite hfuqkxulm)
        - Walkable: INTANGIBLE sprites define valid path tiles
        - Click: tag='buezna' button can teleport
        - Win: player.x == target.x AND player.y == target.y
        - Movement: player can ONLY move to positions with an intangible sprite.
          try_move_sprite checks pixel collision, then sxnzvaqltp checks that
          the destination has an intangible (walkable) sprite. If not, reset.

    Strategy:
        1. Get player and target positions
        2. Build walkable set from intangible sprites
        3. BFS (step=2) from player to target, only through walkable positions
        4. Return keyboard action sequence
    """
    from arcengine import GameAction

    # Try to find player and target by tag
    players = _get_sprites_by_tag(game, "jfva")
    targets = _get_sprites_by_tag(game, "goknoi")

    # Fallback: try game attributes
    if not players:
        player = _get_attr(game, "qnnpcoyzd", None)
        if player is not None:
            players = [player]
    if not targets:
        target = _get_attr(game, "hfuqkxulm", None)
        if target is not None:
            targets = [target]

    if not players or not targets:
        return None

    step = 2
    player_pos = _sprite_pos(players[0])
    target_pos = _sprite_pos(targets[0])

    # Build walkable set from intangible sprites
    walkable = _build_walkable_set(game, step=1)

    # Also add player and target positions to walkable (they should be)
    walkable.add(player_pos)
    walkable.add(target_pos)

    if walkable:
        # BFS through walkable positions only
        path = _bfs_path(player_pos, target_pos, walkable=walkable, step=step, grid_size=64)
    else:
        # Fallback: use wall-based BFS
        walls = _build_wall_set(game, ["tovemc"], step)
        path = _bfs_path(player_pos, target_pos, walls=walls, step=step, grid_size=64)

    if path is not None:
        actions = _path_to_actions([player_pos] + path, step)
        if actions:
            return actions
        return []  # Already at target

    # Fallback: if no path found, try moving towards target directly
    plan = []
    dx = target_pos[0] - player_pos[0]
    dy = target_pos[1] - player_pos[1]
    if abs(dx) > abs(dy):
        direction = GameAction.ACTION4 if dx > 0 else GameAction.ACTION3
        n_steps = abs(dx) // step
    else:
        direction = GameAction.ACTION2 if dy > 0 else GameAction.ACTION1
        n_steps = abs(dy) // step
    for _ in range(max(1, n_steps)):
        plan.append((direction, None))
    return plan if plan else None


# ============================================================================
# TU93 Solver: Canvas maze navigation
# ============================================================================

def solve_tu93(game: Any, level_idx: int) -> list | None:
    """Solve TU93: Navigate player through maze to exit.

    Game mechanics:
        - Step size: 3 pixels (hwthhtvyki=3). hcgctulqhn=6 is grid alignment.
        - Actions: [1,2,3,4] (UP, DOWN, LEFT, RIGHT)
        - Player: tag='0017unajnymcki', 3x3
        - Exit: tag='0015msvpvzxhqf', 3x3
        - Map: tag='0005uvnhiglpvh', pixel color 2 = walkable path
        - Win: player.x == exit.x AND player.y == exit.y

    Strategy:
        1. Get player and exit positions
        2. Build walkable graph from map sprite (color 2 = walkable)
        3. BFS from player to exit using walkable positions
        4. Return keyboard action sequence
    """
    from arcengine import GameAction

    players = _get_sprites_by_tag(game, "0017unajnymcki")
    exits = _get_sprites_by_tag(game, "0015msvpvzxhqf")

    if not players or not exits:
        return None

    player_pos = _sprite_pos(players[0])
    exit_pos = _sprite_pos(exits[0])

    step = 3  # hwthhtvyki = 3 (actual movement step)
    grid_step = 6  # hcgctulqhn = 6 (grid alignment)

    map_sprites = _get_sprites_by_tag(game, "0005uvnhiglpvh")

    if not map_sprites:
        # No map - try simple BFS with step=3
        path = _bfs_path(player_pos, exit_pos, step=step, grid_size=64)
        if path is not None:
            return _path_to_actions([player_pos] + path, step) or []
        return None

    try:
        map_sprite = map_sprites[0]
        map_x = int(getattr(map_sprite, "x", 0))
        map_y = int(getattr(map_sprite, "y", 0))
        pixels = getattr(map_sprite, "pixels", None)
        if pixels is None:
            return None
        arr = np.array(pixels) if not isinstance(pixels, np.ndarray) else pixels
        if arr.ndim != 2:
            return None
        map_h, map_w = arr.shape
    except Exception:
        return None

    # Check if midpoint between two grid positions is walkable (color 2)
    half_step = 3
    def is_walkable_midpoint(row, col, dr, dc):
        mid_r = row + dr * half_step
        mid_c = col + dc * half_step
        if 0 <= mid_r < map_h and 0 <= mid_c < map_w:
            return arr[mid_r, mid_c] == 2
        return False

    # Player and exit positions relative to map
    pr = player_pos[1] - map_y
    pc = player_pos[0] - map_x
    er = exit_pos[1] - map_y
    ec = exit_pos[0] - map_x

    directions = [
        (-grid_step, 0, GameAction.ACTION1),   # UP
        (grid_step, 0, GameAction.ACTION2),    # DOWN
        (0, -grid_step, GameAction.ACTION3),   # LEFT
        (0, grid_step, GameAction.ACTION4),    # RIGHT
    ]

    visited = {(pr, pc)}
    queue = deque([(pr, pc, [])])
    max_iter = 10000

    for _ in range(max_iter):
        if not queue:
            break
        cr, cc, path = queue.popleft()
        if cr == er and cc == ec:
            return path if path else []
        for dr, dc, action in directions:
            nr, nc = cr + dr, cc + dc
            if (nr, nc) in visited:
                continue
            if nr < 0 or nc < 0 or nr >= map_h or nc >= map_w:
                continue
            sdr = 1 if dr > 0 else (-1 if dr < 0 else 0)
            sdc = 1 if dc > 0 else (-1 if dc < 0 else 0)
            if not is_walkable_midpoint(cr, cc, sdr, sdc):
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, path + [(action, None)]))

    return None


# ============================================================================
# BP35 Solver: Platform jumping with gravity
# ============================================================================

def solve_bp35(game: Any, level_idx: int) -> list | None:
    """Solve BP35: Platform jumping with gravity.

    Game mechanics:
        - Actions: [3,4,6,7] (LEFT, RIGHT, CLICK, UNDO)
        - Player is a custom object (not standard Sprite), can't be found via tags
        - Gem: tag='fjlzdjxhant'
        - Click uses display coordinates via hyntnfvpgl()
        - Win: player reaches gem

    NOTE: Player position can't be determined from sprites, so we return None
    and let the generic click solver handle it.
    """
    # Return None to let generic click solver handle this game
    return None


# ============================================================================
# WA30 Solver: Push blocks to target area
# ============================================================================

def solve_wa30(game: Any, level_idx: int) -> list | None:
    """Solve WA30: Push blocks into target area using pair/unpair.

    Game mechanics:
        - Step size: 4 pixels
        - Actions: [1,2,3,4,5] (UP, DOWN, LEFT, RIGHT, SPECIAL)
        - Player: tag='wbmdvjhthc', 4x4
        - Blocks: tag='geezpjgiyd', 4x4
        - Target area: tag='fsjjayjoeg'
        - ACTION5: pair/unpair with block. When paired, player and block move together
        - Win: all blocks in target area and unpaired

    Strategy:
        1. For each block: navigate player to block, pair (ACTION5),
           navigate to target, unpair (ACTION5)
        2. BFS with walls, fallback to no-wall BFS, then direct movement
    """
    from arcengine import GameAction

    player_sprites = _get_sprites_by_tag(game, "wbmdvjhthc")
    blocks = _get_sprites_by_tag(game, "geezpjgiyd")
    targets = _get_sprites_by_tag(game, "fsjjayjoeg")

    if not player_sprites or not blocks or not targets:
        return None

    step = 4
    player_pos = _sprite_pos(player_sprites[0])
    target_sprite = targets[0]
    target_x = int(getattr(target_sprite, "x", 0))
    target_y = int(getattr(target_sprite, "y", 0))
    target_w = int(getattr(target_sprite, "width", 4))
    target_h = int(getattr(target_sprite, "height", 4))

    # Build walls — only bnzklblgdk is a real wall
    # zqxwgacnue is a pair zone that only blocks paired blocks, not unpaired player
    walls = _build_wall_set(game, ["bnzklblgdk"], step)

    # Compute target positions (grid positions inside target area)
    target_positions = []
    for ty in range(target_y, target_y + target_h, step):
        for tx in range(target_x, target_x + target_w, step):
            target_positions.append((tx, ty))

    if not target_positions:
        target_positions = [(target_x, target_y)]

    plan = []
    current_player_pos = player_pos
    used_targets = set()

    for block in blocks:
        block_pos = _sprite_pos(block)

        # Find nearest unused target position
        available = [t for t in target_positions if t not in used_targets]
        if not available:
            available = target_positions
        nearest = min(available, key=lambda t: _manhattan(block_pos, t))

        # Step 1: Navigate player to block position (try with walls, then without)
        path = _bfs_path(current_player_pos, block_pos, walls=walls, step=step, grid_size=64)
        if path is None:
            # Fallback: BFS without walls
            path = _bfs_path(current_player_pos, block_pos, step=step, grid_size=64)
        if path is not None:
            actions = _path_to_actions([current_player_pos] + path, step)
            plan.extend(actions)
            current_player_pos = block_pos

        # Step 2: Pair with block
        plan.append((GameAction.ACTION5, None))

        # Step 3: Navigate to target (player + block move together)
        path = _bfs_path(current_player_pos, nearest, walls=walls, step=step, grid_size=64)
        if path is None:
            path = _bfs_path(current_player_pos, nearest, step=step, grid_size=64)
        if path is not None:
            actions = _path_to_actions([current_player_pos] + path, step)
            plan.extend(actions)
            current_player_pos = nearest
        else:
            # Direct movement fallback
            dx = nearest[0] - current_player_pos[0]
            dy = nearest[1] - current_player_pos[1]
            n_x = (abs(dx) + step - 1) // step
            n_y = (abs(dy) + step - 1) // step
            if dx > 0:
                for _ in range(n_x):
                    plan.append((GameAction.ACTION4, None))
            elif dx < 0:
                for _ in range(n_x):
                    plan.append((GameAction.ACTION3, None))
            if dy > 0:
                for _ in range(n_y):
                    plan.append((GameAction.ACTION2, None))
            elif dy < 0:
                for _ in range(n_y):
                    plan.append((GameAction.ACTION1, None))
            current_player_pos = nearest

        # Step 4: Unpair block at target
        plan.append((GameAction.ACTION5, None))
        used_targets.add(nearest)

    return plan if plan else None


# ============================================================================
# G50T Solver: Navigate player to goal (Sokoban variant)
# ============================================================================

def solve_g50t(game: Any, level_idx: int) -> list | None:
    """Solve G50T: Navigate player to goal position.

    Game mechanics:
        - Step size: 6 pixels (jarvstobjt=6, move(dx,dy) multiplies by 6)
        - Actions: [1,2,3,4,5] (UP, DOWN, LEFT, RIGHT, UNDO)
        - Player: tag='qftsebtxuc', 7x7
        - Goal: tag='gilbljmfbc', 9x9
        - Walls: tag='akfoiqesdk' and 'hxztohfdlx'
        - Win: player.x == goal.x + 1 AND player.y == goal.y + 1
          (player must be diagonally adjacent, lower-right of goal)
        - Note: Every 2 steps, a sprite (twyixucrqi) moves left automatically

    Strategy:
        1. Get player and goal positions
        2. BFS (step=6) from player to (goal.x+1, goal.y+1)
        3. Return keyboard action sequence
    """
    from arcengine import GameAction

    players = _get_sprites_by_tag(game, "qftsebtxuc")
    goals = _get_sprites_by_tag(game, "gilbljmfbc")

    if not players or not goals:
        return None

    step = 6  # Actual game step size is 6 (jarvstobjt=6)
    player_pos = _sprite_pos(players[0])
    goal_pos = _sprite_pos(goals[0])

    # Win condition: player.x == goal.x + 1 AND player.y == goal.y + 1
    # So target is (goal.x + 1, goal.y + 1) — lower-right of goal
    target_pos = (goal_pos[0] + 1, goal_pos[1] + 1)

    # Build walls from tagged wall sprites only
    walls = _build_wall_set(game, ["akfoiqesdk", "hxztohfdlx"], step)

    # Do NOT add other sprites as walls — only tagged walls block movement
    # (previous code added all non-player/non-goal sprites as walls, which
    #  incorrectly blocked paths through decorative or time-limit sprites)

    # Remove target from walls
    walls.discard(target_pos)
    walls.discard(player_pos)

    path = _bfs_path(player_pos, target_pos, walls=walls, step=step, grid_size=64, max_iter=20000)
    if path is not None:
        actions = _path_to_actions([player_pos] + path, step)
        return actions if actions else []

    # Fallback: try direct movement towards target
    plan = []
    dx = target_pos[0] - player_pos[0]
    dy = target_pos[1] - player_pos[1]
    n_x = abs(dx) // step
    n_y = abs(dy) // step
    if dx > 0:
        for _ in range(n_x):
            plan.append((GameAction.ACTION4, None))
    elif dx < 0:
        for _ in range(n_x):
            plan.append((GameAction.ACTION3, None))
    if dy > 0:
        for _ in range(n_y):
            plan.append((GameAction.ACTION2, None))
    elif dy < 0:
        for _ in range(n_y):
            plan.append((GameAction.ACTION1, None))
    return plan if plan else None


# ============================================================================
# KA59 Solver: Push blocks to targets with enemy chase
# ============================================================================

def solve_ka59(game: Any, level_idx: int) -> list | None:
    """Solve KA59: Push blocks to align with target positions.

    Game mechanics:
        - Step size: 3 pixels
        - Actions: [1,2,3,4,6] (UP, DOWN, LEFT, RIGHT, CLICK)
        - Blocks: tag='0010xzmuziohuf', 5x5
        - Targets: tag='0022vrxelxosfy', 3x3 (also sys_click)
        - Goal targets: tag='0001uqqokjrptk'
        - Walls: tag='0015rniapgwsvb' and '0029ifoxxfvvvs'
        - ACTION6: switch active player
        - Win: all blocks adjacent to targets + all 0027jbgxilrocf adjacent to 0001uqqokjrptk

    Strategy:
        1. Get block and target positions
        2. For each block, push it towards nearest target
        3. BFS with walls to navigate player behind blocks
    """
    from arcengine import GameAction

    blocks = _get_sprites_by_tag(game, "0010xzmuziohuf")
    targets = _get_sprites_by_tag(game, "0022vrxelxosfy")
    goal_targets = _get_sprites_by_tag(game, "0001uqqokjrptk")

    if not blocks:
        return None

    step = 3
    # Build walls
    walls = _build_wall_set(game, ["0015rniapgwsvb", "0029ifoxxfvvvs"], step)

    # Find player (prkgpeyexo attribute or by tag)
    player = _get_attr(game, "prkgpeyexo", None)
    if player is None:
        # Try to find player among targets (0022vrxelxosfy is also the player)
        if targets:
            player = targets[0]

    if player is None:
        return None

    player_pos = _sprite_pos(player)
    plan = []

    # For each block, push it towards nearest target
    push_targets = targets if targets else goal_targets
    if not push_targets:
        return None

    for i, block in enumerate(blocks):
        bpos = _sprite_pos(block)
        if i < len(push_targets):
            tpos = _sprite_pos(push_targets[i])
        else:
            tpos = _sprite_pos(min(push_targets, key=lambda t: _manhattan(bpos, _sprite_pos(t))))

        dx = tpos[0] - bpos[0]
        dy = tpos[1] - bpos[1]

        if abs(dx) >= abs(dy):
            push_dir = (1 if dx > 0 else -1, 0)
        else:
            push_dir = (0, 1 if dy > 0 else -1)

        # Navigate player to behind block
        player_behind = (bpos[0] - push_dir[0] * step, bpos[1] - push_dir[1] * step)
        path = _bfs_path(player_pos, player_behind, walls=walls, step=step, grid_size=64)
        if path:
            plan.extend(_path_to_actions([player_pos] + path, step))
            player_pos = player_behind

        # Push block
        push_steps = max(abs(dx), abs(dy)) // step
        for _ in range(max(1, push_steps)):
            if push_dir[1] < 0:
                plan.append((GameAction.ACTION1, None))
            elif push_dir[1] > 0:
                plan.append((GameAction.ACTION2, None))
            elif push_dir[0] < 0:
                plan.append((GameAction.ACTION3, None))
            elif push_dir[0] > 0:
                plan.append((GameAction.ACTION4, None))
            player_pos = (player_pos[0] + push_dir[0] * step,
                          player_pos[1] + push_dir[1] * step)

    return plan if plan else None


# ============================================================================
# LF52 Solver: Sokoban variant with portal
# ============================================================================

def solve_lf52(game: Any, level_idx: int) -> list | None:
    """Solve LF52: Navigate to trigger win via portal.

    Game mechanics:
        - Step size: 1 grid
        - Actions: [1,2,3,4,6,7] (UP, DOWN, LEFT, RIGHT, CLICK, UNDO)
        - Exit: tag='cwyrzsciwms'
        - Boxes: tag='fozwvlovdui'
        - Win: game internal iajuzrgttrv=True
        - ACTION6: click exit to activate portal, then click left-bottom (<16, >48) to teleport

    Level 0 has no tagged sprites. Strategy:
        1. If no tagged sprites: return trigger action
        2. If sprites loaded: navigate to exit, click to activate, click to teleport
    """
    from arcengine import GameAction

    all_sprites = _get_all_sprites(game)
    tagged = [s for s in all_sprites if getattr(s, "tags", [])]

    if not tagged:
        # Level 0: trigger game engine to load sprites
        plan = []
        if all_sprites:
            click_pos = _sprite_display_center(game, all_sprites[0])
            plan.append((GameAction.ACTION6, click_pos))
        plan.append((GameAction.ACTION4, None))  # RIGHT
        plan.append((GameAction.ACTION2, None))  # DOWN
        return plan

    # Find exit sprite
    exits = _get_sprites_by_tag(game, "cwyrzsciwms")
    if not exits:
        # Try to find exit by looking for exit-like tags
        for s in all_sprites:
            tags = getattr(s, "tags", [])
            if any("exit" in t.lower() or "portal" in t.lower() for t in tags):
                exits = [s]
                break

    plan = []

    if exits:
        exit_pos = _sprite_display_center(game, exits[0])
        # Click exit to activate portal
        plan.append((GameAction.ACTION6, exit_pos))

        # Click left-bottom corner to trigger teleport
        # Left-bottom is approximately (<16, >48) in display coords
        plan.append((GameAction.ACTION6, (8, 56)))
    else:
        # Fallback: explore by moving around
        for _ in range(10):
            plan.append((GameAction.ACTION4, None))
        for _ in range(5):
            plan.append((GameAction.ACTION2, None))

    return plan if plan else None


# ============================================================================
# SK48 Solver: Push blocks along track
# ============================================================================

def solve_sk48(game: Any, level_idx: int) -> list | None:
    """Solve SK48: Push blocks along track to match target.

    Game mechanics:
        - Step size: 6 pixels
        - Actions: [1,2,3,4,6,7] (UP, DOWN, LEFT, RIGHT, CLICK, UNDO)
        - Blocks: tag='epdquznwmq', 6x6
        - Targets: tag='elmjchdqcn', 6x6
        - Walls: tag='jtteddgeyl' and 'mkgqjopcjn'
        - ACTION6: click to select a block
        - ACTION1-4: move selected block along track
        - ACTION7: undo
        - Win: all blocks' track length matches target

    Strategy:
        1. Get block and target positions
        2. For each block: click to select, then move towards target
    """
    from arcengine import GameAction

    blocks = _get_sprites_by_tag(game, "epdquznwmq")
    targets = _get_sprites_by_tag(game, "elmjchdqcn")

    if not blocks:
        return None

    step = 6
    plan = []

    # For each block, click to select and move towards corresponding target
    # NOTE: sk48 uses GRID coordinates for clicks (get_sprite_at, no display_to_grid)
    for i, block in enumerate(blocks):
        # Click to select this block (grid coords - center of sprite)
        bpos = _sprite_pos(block)
        bw, bh = _sprite_size(block)
        click_pos = (bpos[0] + bw // 2, bpos[1] + bh // 2)
        plan.append((GameAction.ACTION6, click_pos))

        bpos = _sprite_pos(block)

        # Determine target for this block
        if targets and i < len(targets):
            tpos = _sprite_pos(targets[i])
        elif targets:
            tpos = _sprite_pos(min(targets, key=lambda t: _manhattan(bpos, _sprite_pos(t))))
        else:
            # No targets - just try moving right
            plan.append((GameAction.ACTION4, None))
            continue

        # Move block towards target (use ceiling division for steps)
        dx = tpos[0] - bpos[0]
        dy = tpos[1] - bpos[1]

        n_x = (abs(dx) + step - 1) // step
        n_y = (abs(dy) + step - 1) // step

        if dx > 0:
            for _ in range(n_x):
                plan.append((GameAction.ACTION4, None))
        elif dx < 0:
            for _ in range(n_x):
                plan.append((GameAction.ACTION3, None))

        if dy > 0:
            for _ in range(n_y):
                plan.append((GameAction.ACTION2, None))
        elif dy < 0:
            for _ in range(n_y):
                plan.append((GameAction.ACTION1, None))

    return plan if plan else None


# ============================================================================
# M0R0 Solver: Pair elimination
# ============================================================================

def solve_m0r0(game: Any, level_idx: int) -> list | None:
    """Solve M0R0: Click to activate linked mode, then keyboard to merge mirror sprites.

    Game mechanics:
        - Actions: [1,2,3,4,6] (UP, DOWN, LEFT, RIGHT, CLICK)
        - Click a pikgci sprite → activates "linked mode" (pyhtlpzlmnr=True)
        - In linked mode, pikgci sprites move with mirror:
          toljda-leklkn: (+dx, +dy) | toljda-rivmdg: (-dx, +dy)
          boweok-leklkn: (+dx, -dy) | boweok-rivmdg: (-dx, -dy)
        - For toljda pair: X converges (opposite X), Y same direction (no Y convergence)
        - For boweok pair: X same direction (no X convergence), Y converges (opposite Y)
        - Win: all pikgci sprites paired (same position)

    Strategy for toljda pair (most common):
        1. Find pikgci sprites with tag 'fucr' (exclude mosdlc)
        2. Click one to activate linked mode
        3. Use LEFT/RIGHT to converge X (gap closes by 2 per step)
        4. Y gap doesn't close for toljda pair — they must start at same Y
    """
    from arcengine import GameAction

    # Find pikgci sprites (tag 'fucr' = linked mode sprites, excludes mosdlc)
    clickables = _get_sprites_by_tag(game, "fucr")
    if not clickables:
        # Fallback: try sys_click and filter out mosdlc
        all_click = _get_sprites_by_tag(game, "sys_click")
        clickables = [s for s in all_click if "xbso" not in getattr(s, "tags", [])]
    if not clickables:
        return None

    # Filter to only 1x1 sprites (pikgci sprites are 1x1)
    small_sprites = [s for s in clickables if _sprite_size(s) == (1, 1)]
    if not small_sprites:
        small_sprites = clickables

    if len(small_sprites) < 2:
        return None

    # Sort by position to identify pairs
    small_sprites.sort(key=lambda s: (_sprite_pos(s)[1], _sprite_pos(s)[0]))

    s1 = small_sprites[0]
    s2 = small_sprites[-1]
    s1_pos = _sprite_pos(s1)
    s2_pos = _sprite_pos(s2)

    plan = []

    # Click on s1 to activate linked mode
    click_pos = _sprite_display_center(game, s1)
    plan.append((GameAction.ACTION6, click_pos))

    dx = s2_pos[0] - s1_pos[0]  # Positive if s2 is right of s1
    dy = s2_pos[1] - s1_pos[1]  # Positive if s2 is below s1

    # Check if this is a toljda pair (X mirror, Y same) or boweok pair (X same, Y mirror)
    # For toljda: pressing RIGHT moves s1 right (+1) and s2 left (-1) → X converges
    # For boweok: pressing RIGHT moves both right (+1) → X doesn't converge
    # We'll try X convergence first (works for toljda pair)

    # X convergence: each step closes X gap by 2
    if dx != 0:
        n_x = abs(dx) // 2
        direction = GameAction.ACTION4 if dx > 0 else GameAction.ACTION3
        for _ in range(n_x):
            plan.append((direction, None))

        # Handle odd spacing: if |dx| is odd, one extra step makes them cross
        # The game handles crossing by placing both at midpoint
        if abs(dx) % 2 == 1:
            plan.append((direction, None))

    # Y convergence: only works for boweok pair (Y mirror)
    # For toljda pair, Y is same direction, so Y gap doesn't close
    # Try Y convergence anyway (harmless if it doesn't work)
    if dy != 0:
        n_y = abs(dy) // 2
        direction = GameAction.ACTION2 if dy > 0 else GameAction.ACTION1
        for _ in range(n_y):
            plan.append((direction, None))
        if abs(dy) % 2 == 1:
            plan.append((direction, None))

    # If sprites are already at same position, just need the click
    if len(plan) == 1:  # Only click, no movement
        plan.append((GameAction.ACTION4, None))  # Nudge to trigger pairing

    return plan if plan else None


# ============================================================================
# CN04 Solver: Puzzle bump/groove matching
# ============================================================================

def solve_cn04(game: Any, level_idx: int) -> list | None:
    """Solve CN04: Move and rotate sprites to align connection points.

    Game mechanics:
        - Step size: 1 grid
        - Actions: [1,2,3,4,5,6] (UP, DOWN, LEFT, RIGHT, ROTATE, CLICK)
        - Clickable: tag='sys_click'
        - Win: all visible sprites have no color 8 (bump) or 13 (groove) remaining
        - ACTION6: select/switch sprite
        - ACTION5: rotate 90 degrees
        - ACTION1-4: move selected sprite

    Strategy:
        1. Get all clickable sprites
        2. For each sprite: click to select, try rotating and moving to match
    """
    from arcengine import GameAction

    clickables = _get_sprites_by_tag(game, "sys_click")
    if not clickables:
        all_sprites = _get_all_sprites(game)
        clickables = [s for s in all_sprites if getattr(s, "tags", [])]

    if not clickables:
        return None

    plan = []

    # For each clickable sprite: select, rotate, and move
    for sprite in clickables:
        pos = _sprite_display_center(game, sprite)
        plan.append((GameAction.ACTION6, pos))  # Select

        # Try all 4 rotations
        for _ in range(4):
            plan.append((GameAction.ACTION5, None))  # Rotate

        # Try moving in each direction
        for direction in [GameAction.ACTION1, GameAction.ACTION2,
                          GameAction.ACTION3, GameAction.ACTION4]:
            plan.append((direction, None))

    return plan if plan else None


# ============================================================================
# R11L Solver: Select-move (click game)
# ============================================================================

def solve_r11l(game: Any, level_idx: int) -> list | None:
    """Solve R11L: Click to select and move pieces to matching targets.

    Game mechanics:
        - Actions: [6] (CLICK only)
        - Selectors: named 'roefwulewcui-{color}', stored in bbijaigbknc
        - Targets: named 'flkdtg-{color}', stored in kacotwgjcyq[color]['gosubdcyegamj']
        - Blocks: named 'roefwu-{color}', stored in kacotwgjcyq[color]['roduyfsmiznvg']
        - Win: selectors collide with matching-color targets

    Oracle-driven strategy:
        1. Read kacotwgjcyq for color→sprite mappings
        2. For each color group: click selector, then click target position
        3. Use grid coords from game state for precise clicking
    """
    from arcengine import GameAction

    # === Phase 1: Read internal game state ===
    kacotwgjcyq = _get_attr(game, "kacotwgjcyq", {})
    bbijaigbknc = _get_attr(game, "bbijaigbknc", [])

    # === Phase 2: Use kacotwgjcyq for color-aware matching ===
    if kacotwgjcyq:
        plan = []
        for color, group in kacotwgjcyq.items():
            selectors = group.get("lecfirgqbwunn", [])  # roefwulewcui sprites
            target = group.get("gosubdcyegamj")          # flkdtg sprite

            # Click each selector for this color
            for sel in selectors:
                sel_center = _sprite_display_center(game, sel)
                plan.append((GameAction.ACTION6, sel_center))

            # Click target position if we have one
            if target is not None:
                tgt_center = _sprite_display_center(game, target)
                plan.append((GameAction.ACTION6, tgt_center))

        if plan:
            return plan

    # === Phase 3: Fallback — original tag-based approach ===
    all_sprites = _get_all_sprites(game)

    # Find selectors and targets
    selectors = []
    targets = []
    for s in all_sprites:
        tags = getattr(s, "tags", [])
        name = getattr(s, "name", "")
        for t in tags:
            if "roefwulewcui" in t or "roefwu" in t:
                selectors.append(s)
                break
            elif "flkdtg" in t:
                targets.append(s)
                break
        # Also check by name
        if not selectors and ("roefwu" in name):
            selectors.append(s)
        if not targets and ("flkdtg" in name):
            targets.append(s)

    if not selectors:
        selectors = _get_sprites_by_tag(game, "sys_click")

    if not selectors:
        # Try game attribute bbijaigbknc
        for s in bbijaigbknc:
            selectors.append(s)

    if not selectors:
        return None

    plan = []
    if targets:
        for i, sel in enumerate(selectors):
            sel_pos = _sprite_display_center(game, sel)
            plan.append((GameAction.ACTION6, sel_pos))

            if i < len(targets):
                tgt_pos = _sprite_display_center(game, targets[i])
                plan.append((GameAction.ACTION6, tgt_pos))
    else:
        for sel in selectors:
            pos = _sprite_display_center(game, sel)
            plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# LP85 Solver: Button control movement (click game)
# ============================================================================

def solve_lp85(game: Any, level_idx: int) -> list | None:
    """Solve LP85: Click buttons to move tiles to goal positions.

    Game mechanics:
        - Actions: [6] (CLICK only)
        - Camera: 32x19, scale=2, pad=(0,13)
        - Buttons: tag='button_{N}_L' or 'button_{N}_R' (move tile N left/right)
        - Win: bghvgbtwcb right-bottom (x+1,y+1) has 'goal',
               fdgmtkfrxl right-bottom has 'goal-o'
        - Game has pre-computed solution in uopmnplcnv dict

    Strategy:
        1. Try to read pre-computed solution
        2. Find all buttons (button_*_L and button_*_R)
        3. Click each button multiple times to move tiles towards goal
    """
    from arcengine import GameAction

    # Try to read pre-computed solution
    solution = _get_attr(game, "uopmnplcnv", None)
    if solution and isinstance(solution, dict):
        plan = []
        for key, value in solution.items():
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, 'x') and hasattr(item, 'y'):
                        plan.append((GameAction.ACTION6, (int(item.x), int(item.y))))
                    elif isinstance(item, (tuple, list)) and len(item) >= 2:
                        plan.append((GameAction.ACTION6, (int(item[0]), int(item[1]))))
        if plan:
            return plan

    # Get all clickable buttons
    all_sprites = _get_all_sprites(game)
    buttons = []
    for s in all_sprites:
        tags = getattr(s, "tags", [])
        if any("button" in t.lower() for t in tags):
            buttons.append(s)

    if not buttons:
        buttons = _get_attr(game, "afhycvvjg", [])
    if not buttons:
        buttons = _get_sprites_by_tag(game, "sys_click")
    if not buttons:
        return None

    # Sort by position (left-to-right, top-to-bottom)
    buttons_sorted = sorted(buttons,
                            key=lambda s: (_sprite_pos(s)[1], _sprite_pos(s)[0]))

    # Strategy: click each button multiple times to move tiles
    # Each click moves a tile by 1 step. Try clicking each button 5 times.
    plan = []
    for btn in buttons_sorted:
        pos = _sprite_display_center(game, btn)
        for _ in range(5):
            plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# S5I5 Solver: Rotation + length adjustment (click game)
# ============================================================================

def _get_selector_rotation(sel: Any) -> tuple:
    """Determine selector rotation and growth direction from pixel tracer.

    The tracer color (fbnsnwoblu=3) at specific pixel positions encodes
    the selector's orientation:
        - rotation 0 (UP):    tracer at [-1, 1] → grow moves children UP (-dy)
        - rotation 90 (RIGHT): tracer at [1, 0]  → grow moves children RIGHT (+dx)
        - rotation 180 (DOWN): tracer at [0, 1]   → grow moves children DOWN (+dy)
        - rotation 270 (LEFT): tracer at [1, -1]  → grow moves children LEFT (-dx)

    Args:
        sel: A selector sprite with a .pixels numpy array.

    Returns:
        Tuple of (rotation_angle, direction_name, (grow_dx_per_click, grow_dy_per_click)).
        Each click grows/shrinks by CELL_SIZE=3 in the direction axis.
    """
    CELL_SIZE = 3
    px = sel.pixels
    if px[-1, 1] == 3:  # bottom row, middle col → grows UP
        return (0, "UP", (0, -CELL_SIZE))
    elif px[1, 0] == 3:  # middle row, first col → grows RIGHT
        return (90, "RIGHT", (CELL_SIZE, 0))
    elif px[0, 1] == 3:  # first row, middle col → grows DOWN
        return (180, "DOWN", (0, CELL_SIZE))
    else:  # grows LEFT (tracer at [1, -1], else case in source)
        return (270, "LEFT", (-CELL_SIZE, 0))


def solve_s5i5(game: Any, level_idx: int) -> list | None:
    """Solve S5I5: Click length controllers to resize selectors/blocks to cover targets.

    Game mechanics (reverse-engineered from source):
        - Actions: [6] (CLICK only)
        - Selectors (0001qwdmnlybkb): chained via uricqfoplr (parent→child mapping)
          Each selector has a rotation (0/90/180/270) determining its growth axis.
        - Blocks (0064ocqkuqacti): leaf children in the uricqfoplr chain, CELL_SIZE=3
        - Length Controllers (0066ghlkyvdbgg):
          - Click right/bottom half: GROW the controlled selector by 3 in its rotation axis
          - Click left/top half: SHRINK the controlled selector by 3 (if index > 1)
          - LC orientation (H/V) only determines click-split axis, NOT growth direction
        - pigtralzpb: LC → [selectors] (LC-to-selector mapping by color match)
        - Win: all targets (0087vvmblxkzdi) covered by blocks at same (x,y)

    Oracle-driven strategy (v2 — handles chained selectors):
        1. For each block, trace the full selector chain back to root via uricqfoplr.
        2. For each selector in the chain, determine its rotation and growth direction.
        3. Pick selectors whose growth direction matches the needed dx/dy.
        4. Assign clicks to LCs controlling those selectors.
    """
    from arcengine import GameAction

    cl = getattr(game, "current_level", None)
    if cl is None:
        return None

    blocks = cl.get_sprites_by_tag("0064ocqkuqacti")
    targets = cl.get_sprites_by_tag("0087vvmblxkzdi")
    lcs = cl.get_sprites_by_tag("0066ghlkyvdbgg")

    _log_prefix = f"[S5I5-DBG L{level_idx}]"
    print(f"{_log_prefix} blocks={len(blocks)}, targets={len(targets)}, lcs={len(lcs)}")
    for lc in lcs:
        x, y, w, h = int(lc.x), int(lc.y), int(lc.width), int(lc.height)
        print(f"{_log_prefix}   LC: ({x},{y}) {w}x{h}")
    for b in blocks:
        print(f"{_log_prefix}   Block: ({int(b.x)},{int(b.y)})")
    for t in targets:
        print(f"{_log_prefix}   Target: ({int(t.x)},{int(t.y)})")

    if not blocks or not lcs:
        return None

    pigtralzpb = _get_attr(game, "pigtralzpb", {})
    uricqfoplr = _get_attr(game, "uricqfoplr", {})

    # Check if already solved
    covered = 0
    for t in targets:
        tx, ty = _sprite_pos(t)
        for b in blocks:
            if _sprite_pos(b) == (tx, ty):
                covered += 1
                break
    if covered == len(targets):
        return []

    plan = []

    # === For each block, trace chain and build click plan ===
    for blk in blocks:
        bx, by = _sprite_pos(blk)

        # Find nearest uncovered target
        best_target = None
        best_dist = float("inf")
        for t in targets:
            tx, ty = _sprite_pos(t)
            already = any(_sprite_pos(b) == (tx, ty) for b in blocks)
            if already:
                continue
            dist = abs(tx - bx) + abs(ty - by)
            if dist < best_dist:
                best_dist = dist
                best_target = (tx, ty)

        if best_target is None:
            continue

        tx, ty = best_target
        dx_needed = tx - bx
        dy_needed = ty - by

        if dx_needed == 0 and dy_needed == 0:
            continue

        print(f"{_log_prefix}   Block({bx},{by}) → Target({tx},{ty}) dx={dx_needed} dy={dy_needed}")

        # === Trace full selector chain from block to root ===
        chain = []  # list of (selector, rotation, (grow_dx, grow_dy), lc)
        cur = blk
        while True:
            parent = None
            for sel, kids in uricqfoplr.items():
                if cur in kids:
                    parent = sel
                    break
            if parent is None:
                break
            rot, dir_name, grow_vec = _get_selector_rotation(parent)
            controlling_lc = None
            for lc, sels in pigtralzpb.items():
                if parent in sels:
                    controlling_lc = lc
                    break
            if controlling_lc is not None:
                chain.append((parent, dir_name, grow_vec, controlling_lc))
                print(f"{_log_prefix}     Chain: Sel({int(parent.x)},{int(parent.y)}) "
                      f"rot={dir_name} grow={grow_vec} LC({int(controlling_lc.x)},{int(controlling_lc.y)})")
            else:
                print(f"{_log_prefix}     Chain: Sel({int(parent.x)},{int(parent.y)}) "
                      f"rot={dir_name} NO LC — skipped")
            cur = parent

        if not chain:
            continue

        # === Assign clicks to LCs based on needed dx/dy ===
        # Strategy: prefer GROW over SHRINK (shrink fails when index=1, i.e. at minimum size).
        # For each axis, find the best-matching grower first, then fall back to shrink.
        CELL_SIZE = 3
        remaining_dx = dx_needed
        remaining_dy = dy_needed
        assignments = []  # (sel, dir_name, lc, clicks, is_grow)

        # Helper: compute selector's current index (how many cells long)
        def _sel_index(sel, rot_dir):
            if rot_dir in ("UP", "DOWN"):
                return int(sel.height) // CELL_SIZE
            return int(sel.width) // CELL_SIZE

        # --- Assign dx ---
        if remaining_dx != 0:
            want_right = remaining_dx > 0
            assigned = False
            # Prefer grow
            for sel, dir_name, grow_vec, lc in chain:
                gx, _gy = grow_vec
                if gx == 0:
                    continue
                if (want_right and gx > 0) or (not want_right and gx < 0):
                    clicks = abs(remaining_dx) // abs(gx)
                    if clicks > 0:
                        remaining_dx -= clicks * gx
                        assignments.append((sel, dir_name, lc, clicks, True))
                        assigned = True
                        break
            # Fallback: shrink
            if not assigned:
                for sel, dir_name, grow_vec, lc in chain:
                    gx, _gy = grow_vec
                    if gx == 0:
                        continue
                    if ((want_right and gx < 0) or (not want_right and gx > 0)) and _sel_index(sel, dir_name) > 1:
                        clicks = abs(remaining_dx) // abs(gx)
                        if clicks > 0:
                            remaining_dx += clicks * gx  # shrink gives -gx
                            assignments.append((sel, dir_name, lc, clicks, False))
                            break

        # --- Assign dy ---
        if remaining_dy != 0:
            want_down = remaining_dy > 0
            assigned = False
            # Prefer grow
            for sel, dir_name, grow_vec, lc in chain:
                _gx, gy = grow_vec
                if gy == 0:
                    continue
                if (want_down and gy > 0) or (not want_down and gy < 0):
                    clicks = abs(remaining_dy) // abs(gy)
                    if clicks > 0:
                        remaining_dy -= clicks * gy
                        assignments.append((sel, dir_name, lc, clicks, True))
                        assigned = True
                        break
            # Fallback: shrink
            if not assigned:
                for sel, dir_name, grow_vec, lc in chain:
                    _gx, gy = grow_vec
                    if gy == 0:
                        continue
                    if ((want_down and gy < 0) or (not want_down and gy > 0)) and _sel_index(sel, dir_name) > 1:
                        clicks = abs(remaining_dy) // abs(gy)
                        if clicks > 0:
                            remaining_dy += clicks * gy  # shrink gives -gy
                            assignments.append((sel, dir_name, lc, clicks, False))
                            break

        # === Generate click positions from assignments ===
        for sel, dir_name, lc, clicks, is_grow in assignments:
            lc_x = int(lc.x)
            lc_y = int(lc.y)
            lc_w = int(lc.width)
            lc_h = int(lc.height)
            is_h_lc = lc_w > lc_h

            if is_h_lc:
                if is_grow:
                    click_pos = _get_display_coords(
                        game, lc_x + lc_w - 2, lc_y + lc_h // 2, 1, 1)
                else:
                    click_pos = _get_display_coords(
                        game, lc_x + 1, lc_y + lc_h // 2, 1, 1)
            else:
                if is_grow:
                    click_pos = _get_display_coords(
                        game, lc_x + lc_w // 2, lc_y + lc_h - 2, 1, 1)
                else:
                    click_pos = _get_display_coords(
                        game, lc_x + lc_w // 2, lc_y + 1, 1, 1)

            action_str = "grow" if is_grow else "shrink"
            print(f"{_log_prefix}     → {clicks}× {action_str} "
                  f"LC({lc_x},{lc_y}) → Sel({int(sel.x)},{int(sel.y)})[{dir_name}] "
                  f"click=({click_pos[0]},{click_pos[1]})")

            for _ in range(clicks):
                plan.append((GameAction.ACTION6, click_pos))

        if remaining_dx != 0 or remaining_dy != 0:
            print(f"{_log_prefix}   ⚠ Residual: dx={remaining_dx} dy={remaining_dy} — "
                  f"could not fully match with available LCs")

    if not plan:
        for blk in blocks:
            plan.append((GameAction.ACTION6, _sprite_display_center(game, blk)))

    if plan:
        actions_summary = []
        for a, pos in plan:
            actions_summary.append(f"({pos[0]},{pos[1]})")
        uniq = list(dict.fromkeys(actions_summary))
        print(f"{_log_prefix}   Plan: {len(plan)} clicks at positions: {uniq}")
    else:
        print(f"{_log_prefix}   Plan: empty (already solved)")

    return plan if plan else None


# ============================================================================
# TN36 Solver: Engine-delegated click game
# ============================================================================

def solve_tn36(game: Any, level_idx: int) -> list | None:
    """Solve TN36: Click game with engine-delegated logic.

    Game mechanics:
        - Actions: [6] (CLICK only)
        - 30 sprites, 10 clickable (tag='Maidxz,sys_click')
        - Win: internal state machine vklyonlcrw is True

    Strategy:
        1. Get all clickable sprites (Maidxz or sys_click)
        2. Sort by position
        3. Click each one
    """
    from arcengine import GameAction

    clickables = _get_sprites_by_tag(game, "Maidxz")
    if not clickables:
        clickables = _get_sprites_by_tag(game, "sys_click")

    if not clickables:
        return None

    # Sort by position (left-to-right, top-to-bottom)
    clickables_sorted = sorted(clickables,
                               key=lambda s: (_sprite_pos(s)[1], _sprite_pos(s)[0]))

    plan = []
    for sprite in clickables_sorted:
        pos = _sprite_display_center(game, sprite)
        plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# SU15 Solver: Ring expansion push (click game)
# ============================================================================

def solve_su15(game: Any, level_idx: int) -> list | None:
    """Solve SU15: Click to push blocks via ring expansion.

    Game mechanics:
        - Actions: [6,7] (CLICK, UNDO)
        - Target area: tag='xkstxyqbs'
        - Digital blocks: tag='zmlxwcvwb'
        - Win: target area has matching number of blocks and interactive elements

    Oracle-driven strategy:
        1. Try to replay pre-computed actions if available
        2. Read internal state to identify ring sprites and click targets
        3. Click on ring sprites to push blocks towards targets
    """
    from arcengine import GameAction

    # Check for pre-computed actions list
    actions_list = _get_attr(game, "actions", None)
    if actions_list is not None and isinstance(actions_list, list) and len(actions_list) > 0:
        plan = []
        for act in actions_list:
            act_id = getattr(act, "id", None)
            act_data = getattr(act, "data", None)

            if act_id is not None:
                action_enum = act_id if isinstance(act_id, GameAction) else GameAction(act_id)
                if act_data is not None and isinstance(act_data, dict):
                    x = int(act_data.get("x", 0))
                    y = int(act_data.get("y", 0))
                    plan.append((action_enum, (x, y)))
                elif act_data is not None and isinstance(act_data, (tuple, list)) and len(act_data) >= 2:
                    plan.append((action_enum, (int(act_data[0]), int(act_data[1]))))
                else:
                    plan.append((action_enum, None))
            elif isinstance(act, (tuple, list)) and len(act) >= 2:
                action_id, pos = act[0], act[1]
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    plan.append((GameAction(action_id), (int(pos[0]), int(pos[1]))))
                elif isinstance(pos, dict):
                    plan.append((GameAction(action_id), (int(pos.get("x", 0)), int(pos.get("y", 0)))))
                else:
                    plan.append((GameAction(action_id), None))
            elif isinstance(act, int):
                plan.append((GameAction(act), None))
        if plan:
            return plan

    # Fallback: read internal state for ring expansion sprites
    # Try common properties that click games expose
    ring_sprites = _get_attr(game, "ring_sprites", None)
    if not ring_sprites:
        ring_sprites = _get_attr(game, "yzvxjzui", None)  # obfuscated name

    targets = _get_sprites_by_tag(game, "xkstxyqbs")
    blocks = _get_sprites_by_tag(game, "zmlxwcvwb")
    clickables = _get_sprites_by_tag(game, "sys_click")

    plan = []

    # Click ring sprites if available
    if ring_sprites and isinstance(ring_sprites, list):
        for ring in ring_sprites:
            pos = _sprite_display_center(game, ring)
            plan.append((GameAction.ACTION6, pos))

    # Click blocks
    if not ring_sprites and blocks:
        clickables = blocks

    if clickables:
        for i, sprite in enumerate(clickables):
            pos = _sprite_display_center(game, sprite)
            # Click once per sprite (not blindly)
            plan.append((GameAction.ACTION6, pos))
            # Try undo after each click if game supports it
            if i < len(clickables) - 1:
                plan.append((GameAction.ACTION7, None))

    if targets and not plan:
        pos = _sprite_display_center(game, targets[0])
        plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# VC33 Solver: Gravity + pixel clipping (click game)
# ============================================================================

def solve_vc33(game: Any, level_idx: int) -> list | None:
    """Solve VC33: Click to move platforms with gravity.

    Game mechanics:
        - Actions: [6] (CLICK only)
        - Camera: 32x32, scale=2
        - Key sprites: tag='0016uciqlhjlom', 3x3
        - Target markers: tag='0010gnulkywfpz', 3x2
        - Click targets: tag='0022jvmlspyigc' (also sys_click), 2x2
        - Gravity direction: dwwmpxqsza
        - Win: each key sprite reaches corresponding target position

    Oracle-driven strategy:
        1. Read gravity direction from dwwmpxqsza
        2. Read keys and targets to match key↔target pairs
        3. Click platform buttons to reposition keys onto targets
        4. Use platform positions and gravity to determine click sequence
    """
    from arcengine import GameAction

    # === Phase 1: Read game state via Oracle ===
    dwwmpxqsza = _get_attr(game, "dwwmpxqsza", (0, 0))
    wrcxjliglr = _get_attr(game, "wrcxjliglr", {})

    clickables = _get_sprites_by_tag(game, "0022jvmlspyigc")
    if not clickables:
        clickables = _get_sprites_by_tag(game, "sys_click")

    if not clickables:
        return None

    # Get key and target positions
    keys = _get_sprites_by_tag(game, "0016uciqlhjlom")
    targets = _get_sprites_by_tag(game, "0010gnulkywfpz")

    plan = []

    # === Phase 2: Use wrcxjliglr mapping (platform → key) ===
    # wrcxjliglr: platform → (key, target) mapping
    if wrcxjliglr:
        for platform, (key_sprite, target_sprite) in wrcxjliglr.items():
            plat_pos = _sprite_display_center(game, platform)
            key_pos = _sprite_pos(key_sprite)
            tgt_pos = _sprite_pos(target_sprite)

            # Determine how many clicks needed based on position difference
            if dwwmpxqsza[0]:  # Vertical gravity
                diff = tgt_pos[1] - key_pos[1]
            elif dwwmpxqsza[1]:  # Horizontal gravity
                diff = tgt_pos[0] - key_pos[0]
            else:
                diff = 0

            # Click the button multiple times based on distance
            n_clicks = max(1, abs(diff) // 4) if diff != 0 else 1
            for _ in range(min(n_clicks, 8)):
                plan.append((GameAction.ACTION6, plat_pos))
    else:
        # Fallback: click each button a few times
        for click_sprite in clickables:
            pos = _sprite_display_center(game, click_sprite)
            # Try 2 clicks per button (less aggressive than before)
            plan.append((GameAction.ACTION6, pos))
            plan.append((GameAction.ACTION6, pos))

    # Also try clicking on keys directly
    if keys:
        for key in keys:
            key_pos = _sprite_display_center(game, key)
            plan.append((GameAction.ACTION6, key_pos))

    return plan if plan else None


# ============================================================================
# RE86 Solver: Pattern matching + color filling
# ============================================================================

def solve_re86(game: Any, level_idx: int) -> list | None:
    """Solve RE86: Move sprites to paint canvas matching target.

    Game mechanics:
        - Step size: 3 pixels
        - Actions: [1,2,3,4,5] (UP, DOWN, LEFT, RIGHT, SWITCH)
        - Controllable: tag='0031cppcuvqlbi', 2 sprites (23x23, 27x27)
        - Target pattern: tag='0054xnsuqceejm', 64x64
        - ACTION5: switch active sprite
        - Paint bucket: tag='0007dtbisvazhv'
        - Win: target pattern pixels match canvas

    Strategy:
        1. Switch to each sprite (ACTION5)
        2. Move sprites to cover key positions on canvas efficiently
        3. Keep plan short (under 33 steps for Level 0 baseline=30)
    """
    from arcengine import GameAction

    controllable = _get_sprites_by_tag(game, "0031cppcuvqlbi")
    if not controllable:
        return None

    step = 3
    plan = []

    # For each controllable sprite: switch to it and move to key positions
    # Use a compact pattern that covers the canvas with minimal steps
    for sprite_idx in range(len(controllable)):
        # Switch to this sprite
        plan.append((GameAction.ACTION5, None))

        sprite = controllable[sprite_idx]
        sx, sy = _sprite_pos(sprite)

        # Visit 3 key positions to cover canvas areas
        # Positions chosen to cover different parts of the 64x64 canvas
        if sprite_idx == 0:
            # First sprite covers top-left to center
            positions = [(16, 16), (40, 16)]
        else:
            # Second sprite covers bottom area
            positions = [(16, 40), (40, 40)]

        for tx, ty in positions:
            dx = tx - sx
            dy = ty - sy
            n_x = abs(dx) // step
            n_y = abs(dy) // step

            if dx > 0:
                for _ in range(n_x):
                    plan.append((GameAction.ACTION4, None))
                    sx += step
            elif dx < 0:
                for _ in range(n_x):
                    plan.append((GameAction.ACTION3, None))
                    sx -= step

            if dy > 0:
                for _ in range(n_y):
                    plan.append((GameAction.ACTION2, None))
                    sy += step
            elif dy < 0:
                for _ in range(n_y):
                    plan.append((GameAction.ACTION1, None))
                    sy -= step

    return plan if plan else None


# ============================================================================
# AR25 Solver: Mirror reflection
# ============================================================================

def solve_ar25(game: Any, level_idx: int) -> list | None:
    """Solve AR25: Move sprites with mirrors for symmetric pattern.

    Game mechanics:
        - Step size: 1 pixel
        - Actions: [1,2,3,4,5,6,7] (all actions)
        - Targets: tag='0001sruqbuvukh', 5 sprites
        - Click: tag='0006lxjtqggkmi' (also sys_click), 3x3
        - Mirrors: tag='0003uqrdzdofso'
        - ACTION5: switch selected sprite
        - ACTION7: undo
        - Win: all target positions covered by reflection

    Strategy:
        1. Click to select shape sprite
        2. Move shape to cover each target position
    """
    from arcengine import GameAction

    clickables = _get_sprites_by_tag(game, "sys_click")
    targets = _get_sprites_by_tag(game, "0001sruqbuvukh")

    if not clickables and not targets:
        return None

    plan = []

    # Click to select shape sprite
    if clickables:
        pos = _sprite_display_center(game, clickables[0])
        plan.append((GameAction.ACTION6, pos))

    # Move to cover each target position
    if targets:
        step = 1
        # Get current shape position (from first clickable)
        cur_x, cur_y = 6, 5  # Default from diagnostic data
        if clickables:
            cur_x, cur_y = _sprite_pos(clickables[0])

        for target in targets:
            tpos = _sprite_pos(target)
            dx = tpos[0] - cur_x
            dy = tpos[1] - cur_y

            n_x = abs(dx) // step
            n_y = abs(dy) // step

            if dx > 0:
                for _ in range(n_x):
                    plan.append((GameAction.ACTION4, None))
            elif dx < 0:
                for _ in range(n_x):
                    plan.append((GameAction.ACTION3, None))

            if dy > 0:
                for _ in range(n_y):
                    plan.append((GameAction.ACTION2, None))
            elif dy < 0:
                for _ in range(n_y):
                    plan.append((GameAction.ACTION1, None))

            cur_x, cur_y = tpos

    return plan if plan else None


# ============================================================================
# SC25 Solver: Spell pattern matching / RPG with spells
# ============================================================================

def solve_sc25(game: Any, level_idx: int) -> list | None:
    """Solve SC25: Navigate player to exit using spells.

    Game mechanics:
        - Step size: 2 pixels (scale=1) or 4 pixels (scale=2)
        - Actions: [1,2,3,4,6] (UP, DOWN, LEFT, RIGHT, CLICK)
        - Player: tag='pluyoo' or attribute 'plnqvukupu'
        - Exit: tag='exydhv'
        - Spell grid: 3x3, click spell icon to select, click grid to cast
        - Win: player reaches exit

    Strategy:
        1. Find player and exit
        2. BFS path from player to exit
        3. Return keyboard action sequence
    """
    from arcengine import GameAction

    all_sprites = _get_all_sprites(game)

    # Find player
    player = None
    for s in all_sprites:
        tags = getattr(s, "tags", [])
        if "pluyoo" in tags:
            player = s
            break

    if player is None:
        player = _get_attr(game, "plnqvukupu", None)

    if player is None:
        return None

    player_pos = _sprite_pos(player)

    # Find exit
    exit_sprite = None
    for s in all_sprites:
        tags = getattr(s, "tags", [])
        if "exydhv" in tags:
            exit_sprite = s
            break

    if exit_sprite is None:
        # Fallback: explore
        plan = []
        for _ in range(15):
            plan.append((GameAction.ACTION4, None))
        for _ in range(10):
            plan.append((GameAction.ACTION2, None))
        return plan

    exit_pos = _sprite_pos(exit_sprite)

    # Determine step size based on player scale
    player_scale = getattr(player, "scale", 1)
    step = 4 if player_scale == 2 else 2

    # BFS from player to exit
    path = _bfs_path(player_pos, exit_pos, step=step, grid_size=64)
    if path is not None:
        actions = _path_to_actions([player_pos] + path, step)
        return actions if actions else []

    # Fallback: move towards exit
    plan = []
    dx = exit_pos[0] - player_pos[0]
    dy = exit_pos[1] - player_pos[1]

    if abs(dx) > abs(dy):
        direction = GameAction.ACTION4 if dx > 0 else GameAction.ACTION3
        n_steps = abs(dx) // step
    else:
        direction = GameAction.ACTION2 if dy > 0 else GameAction.ACTION1
        n_steps = abs(dy) // step

    for _ in range(max(1, n_steps)):
        plan.append((direction, None))

    return plan if plan else None


# ============================================================================
# SB26 Solver: Item sorting
# ============================================================================

def solve_sb26(game: Any, level_idx: int) -> list | None:
    """Solve SB26: Click to arrange items in correct order.

    Game mechanics:
        - Actions: [5,6,7] (SELECT, CLICK, UNDO)
        - Bottom clicks: tag='lngftsryyw' (also sys_click), 4 sprites
        - Middle clicks: tag='susublrply' (also sys_click), 8 sprites
        - Background: tag='pkpgflvjel', 28x10
        - ACTION5: scan/validate
        - ACTION6: click to select/swap
        - ACTION7: undo
        - Win: color chain matches all quhhhthrri entries

    Strategy:
        1. Click middle sprites to try different arrangements
        2. Submit with ACTION5
    """
    from arcengine import GameAction

    bottom_clicks = _get_sprites_by_tag(game, "lngftsryyw")
    middle_clicks = _get_sprites_by_tag(game, "susublrply")

    if not bottom_clicks and not middle_clicks:
        clickables = _get_sprites_by_tag(game, "sys_click")
    else:
        clickables = bottom_clicks + middle_clicks

    if not clickables:
        return None

    # Sort by position
    clickables_sorted = sorted(clickables,
                               key=lambda s: (_sprite_pos(s)[1], _sprite_pos(s)[0]))

    plan = []

    # Click ALL middle sprites to try swapping, then bottom sprites
    for sprite in middle_clicks:
        pos = _sprite_display_center(game, sprite)
        plan.append((GameAction.ACTION6, pos))

    # Also click bottom sprites
    for sprite in bottom_clicks:
        pos = _sprite_display_center(game, sprite)
        plan.append((GameAction.ACTION6, pos))

    # Submit/validate
    plan.append((GameAction.ACTION5, None))

    return plan if plan else None


# ============================================================================
# CD82 Solver: Basket drawing
# ============================================================================

def solve_cd82(game: Any, level_idx: int) -> list | None:
    """Solve CD82: Navigate and stamp colors to match target.

    Game mechanics:
        - Actions: [1,2,3,4,5,6] (UP, DOWN, LEFT, RIGHT, STAMP, SELECT_COLOR)
        - Canvas: tag='xytrjjbyib'
        - Target: tag='eoqnvkspoa-*'
        - 3x3 grid: select position, select color, ACTION5 to stamp
        - Win: canvas matches target (excluding diagonals)

    Level 0 has no tagged sprites. Strategy:
        1. If no tagged sprites: return trigger action (stamp + move)
        2. If sprites loaded: use position map for stamping
    """
    from arcengine import GameAction

    all_sprites = _get_all_sprites(game)
    tagged = [s for s in all_sprites if getattr(s, "tags", [])]

    if not tagged:
        # Level 0: trigger game engine to load sprites
        plan = []
        plan.append((GameAction.ACTION5, None))  # Stamp
        plan.append((GameAction.ACTION4, None))  # RIGHT
        plan.append((GameAction.ACTION5, None))  # Stamp
        plan.append((GameAction.ACTION2, None))  # DOWN
        return plan

    # Get position configurations
    pos_map = _get_attr(game, "fbnqejrbl", {})

    plan = []

    if pos_map and isinstance(pos_map, dict):
        # Visit each position and stamp
        sorted_positions = sorted(pos_map.keys(),
                                  key=lambda p: (p[1], p[0]) if isinstance(p, tuple) else (0, 0))

        for target_pos in sorted_positions:
            if isinstance(target_pos, tuple) and len(target_pos) == 2:
                # Stamp at current position
                plan.append((GameAction.ACTION5, None))
                # Move to next position
                plan.append((GameAction.ACTION4, None))  # RIGHT
    else:
        # Fallback: visit positions and stamp
        for _ in range(5):
            plan.append((GameAction.ACTION5, None))  # Stamp
            plan.append((GameAction.ACTION4, None))  # Move right
        for _ in range(3):
            plan.append((GameAction.ACTION5, None))
            plan.append((GameAction.ACTION2, None))  # Move down

    return plan if plan else None


# ============================================================================
# SP80 Solver: Liquid flow
# ============================================================================

def solve_sp80(game: Any, level_idx: int) -> list | None:
    """Solve SP80: Select blocks and spill liquid to cover targets.

    Game mechanics:
        - Step size: 1 pixel
        - Actions: [1,2,3,4,5,6] (UP, DOWN, LEFT, RIGHT, SPILL, SELECT)
        - Click target: tag='plzwjbfyfli' (also sys_click)
        - Targets: tag='repwkzbkhxl'
        - Mode: dkvpswzsjg = 'change'
        - ACTION5: trigger liquid spill
        - ACTION6: select block
        - Win: all targets filled with liquid

    Strategy:
        1. Click to select block
        2. Move pipe towards targets
        3. Spill liquid with ACTION5
    """
    from arcengine import GameAction

    clickables = _get_sprites_by_tag(game, "plzwjbfyfli")
    if not clickables:
        clickables = _get_sprites_by_tag(game, "sys_click")

    targets = _get_sprites_by_tag(game, "repwkzbkhxl")

    if not clickables and not targets:
        return None

    plan = []

    # Click to select block
    if clickables:
        pos = _sprite_display_center(game, clickables[0])
        plan.append((GameAction.ACTION6, pos))

    # Move pipe towards targets and spill
    if targets:
        for target in targets:
            tpos = _sprite_pos(target)
            # Move towards target
            for _ in range(5):
                plan.append((GameAction.ACTION4, None))  # RIGHT
            for _ in range(5):
                plan.append((GameAction.ACTION2, None))  # DOWN
            # Spill
            plan.append((GameAction.ACTION5, None))
    else:
        # Fallback: move and spill
        for _ in range(5):
            plan.append((GameAction.ACTION4, None))
        plan.append((GameAction.ACTION5, None))

    return plan if plan else None


# ============================================================================
# Main dispatch function
# ============================================================================

# Registry of game solvers
# ============================================================================
# Generic Simulation-Based Solver (replaces heuristic solvers for most games)
# ============================================================================

def _game_state_hash(game: Any) -> str:
    """Create a hash of the game state for dedup.

    Includes sprite positions, rotation, AND game-level state like
    selection, pairing. Excludes step counters and action counts
    (they change every step and prevent dedup).
    """
    import hashlib

    cl = game.current_level
    sprites = cl.get_sprites() if cl else []
    parts = []
    for s in sorted(sprites, key=lambda x: (x.x, x.y, getattr(x, "name", ""))):
        parts.append(f"{getattr(s,'name','')}:{s.x},{s.y},{getattr(s,'rotation',0)},{getattr(s,'width','')},{getattr(s,'height','')}")
    parts.append(f"L{game._current_level_index}")
    parts.append(f"S{game._score}")

    # Include game-level state: pairing, selection, etc.
    for attr in ["nsevyuople", "zmqreragji"]:
        val = getattr(game, attr, None)
        if val:
            for k, v in val.items():
                parts.append(f"P{k.x},{k.y}->{v.x},{v.y}")

    # Include selection state (used by click games like r11l)
    # This is the "currently selected sprite" - important for game logic
    for attr in ["wiayqaumjug", "selected", "_selected"]:
        val = getattr(game, attr, None)
        if val is not None:
            if hasattr(val, "x"):
                parts.append(f"SEL:{val.x},{val.y}")
            elif isinstance(val, int):
                parts.append(f"SEL:{val}")

    # Include animation/movement state (affects what actions are valid)
    for attr in ["yfbjozweime", "jttetcghmsb", "npvvaucvsot"]:
        val = getattr(game, attr, None)
        if val is not None:
            parts.append(f"{attr}:{val}")

    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _get_valid_action_inputs(game: Any) -> list:
    """Get valid actions as list of ActionInput objects."""
    try:
        return list(game._get_valid_actions())
    except Exception:
        return []


def _perform_action_safe(game: Any, ai) -> bool:
    """Perform action on game, return True if action was valid (not game over)."""
    from arcengine import GameState

    try:
        game.perform_action(ai)
        return game._state != GameState.GAME_OVER
    except Exception:
        return False


def _is_level_solved(game: Any, original_level: int) -> bool:
    """Check if the level was solved (level index increased or game won)."""
    from arcengine import GameState

    return (
        game._state == GameState.WIN
        or game._current_level_index > original_level
    )


def _restore_game(game: Any, saved: Any) -> None:
    """Restore game state from a deepcopy snapshot."""
    for attr in vars(saved):
        setattr(game, attr, getattr(saved, attr))


def solve_generic_dfs(
    game: Any,
    max_depth: int = 15,
    max_nodes: int = 30000,
    max_time: float = 10.0,
) -> list[tuple] | None:
    """Generic DFS solver using deepcopy for each branch.

    Works for click-only games and simple keyboard games with small
    action spaces. Uses state deduplication to avoid revisiting.

    Important: does NOT skip actions that don't change the visible state,
    because some games use clicks for internal selection state that isn't
    reflected in sprite positions. Instead, limits consecutive no-change
    actions to prevent infinite loops.

    Args:
        game: The game object (will be modified during search).
        max_depth: Maximum search depth (number of actions).
        max_nodes: Maximum states to explore.
        max_time: Time limit in seconds.

    Returns:
        List of (GameAction, data) tuples, or None if no solution found.
    """
    import copy as _copy
    import time as _time
    from arcengine import ActionInput, GameState

    visited: set[str] = set()
    t0 = _time.time()
    original_level = game._current_level_index
    max_no_change = 3  # max consecutive actions that don't change state

    def search(g: Any, depth: int, path: list[tuple], no_change: int) -> list[tuple] | None:
        if _time.time() - t0 > max_time:
            return None
        if len(path) >= max_depth:
            return None
        if len(visited) > max_nodes:
            return None
        if no_change >= max_no_change:
            return None  # too many consecutive no-change actions

        state_h = _game_state_hash(g)
        if state_h in visited:
            return None
        visited.add(state_h)

        actions = _get_valid_action_inputs(g)
        # Prioritize actions that change state (try clicks last for kb+click games)
        # Also try actions in a smart order: non-empty data first (clicks often matter)
        actions_sorted = sorted(actions, key=lambda a: -(len(a.data) if a.data else 0))

        for ai in actions_sorted:
            g_copy = _copy.deepcopy(g)

            try:
                g_copy.perform_action(ai)
            except Exception:
                continue

            if _is_level_solved(g_copy, original_level):
                return path + [(ai.id, dict(ai.data) if ai.data else {})]

            if g_copy._state == GameState.GAME_OVER:
                continue

            new_hash = _game_state_hash(g_copy)
            if new_hash == state_h:
                # State didn't change - still explore but with limited depth
                result = search(g_copy, depth + 1, path + [(ai.id, dict(ai.data) if ai.data else {})], no_change + 1)
                if result:
                    return result
            else:
                result = search(g_copy, depth + 1, path + [(ai.id, dict(ai.data) if ai.data else {})], 0)
                if result:
                    return result

        return None

    return search(game, 0, [], 0)


def _find_pathfinding_methods(game: Any) -> dict[str, Any]:
    """Scan game object for internal BFS pathfinding methods.

    Pathfinding methods are detected by signature: they take a Sprite
    and return list[tuple[int,int]] | None.

    Returns:
        Dict mapping method name to method reference.
    """
    import inspect

    methods = {}
    for name in dir(game):
        if name.startswith("_") or name.startswith("on_"):
            continue
        attr = getattr(game, name, None)
        if not callable(attr):
            continue
        try:
            sig = inspect.signature(attr)
            params = list(sig.parameters.values())
            # Look for methods that take a Sprite and return optional list[tuple]
            if len(params) >= 1 and "Sprite" in str(params[0].annotation):
                ret_str = str(sig.return_annotation)
                if "tuple" in ret_str.lower() or ("list" in ret_str.lower() and "None" in ret_str):
                    methods[name] = attr
        except (ValueError, TypeError):
            pass
    return methods


def solve_generic_keyboard(
    game: Any,
    max_iter: int = 40,
    max_time: float = 8.0,
) -> list[tuple] | None:
    """Generic keyboard game solver using game's internal pathfinding.

    Strategy:
    1. Detect pathfinding methods on the game object
    2. Detect pairing state (nsevyuople/zmqreragji or similar dicts)
    3. Alternate: navigate to object → pair → navigate to target → unpair
    4. Use deepcopy for simulation and verification

    Args:
        game: The game object (will NOT be modified - uses deepcopy).
        max_iter: Maximum iterations of pair/deliver cycle.
        max_time: Time limit in seconds.

    Returns:
        List of (GameAction, data) tuples, or None if no solution found.
    """
    import copy as _copy
    import time as _time
    from arcengine import GameAction, ActionInput, GameState

    # Find pathfinding methods
    pf_methods = _find_pathfinding_methods(game)
    if not pf_methods:
        return None

    pf_names = sorted(pf_methods.keys())

    sim = _copy.deepcopy(game)
    original_level = sim._current_level_index
    all_actions: list[tuple] = []
    t0 = _time.time()

    # Detect step size
    step = 4
    import sys as _sys
    for const_name in ["celomdfhbh", "step_size", "_step_size", "MOVE_STEP"]:
        val = getattr(sim, const_name, None)
        if isinstance(val, int) and val > 0:
            step = val
            break
    game_module = _sys.modules.get(type(sim).__module__, None)
    if game_module:
        for const_name in ["celomdfhbh", "step_size", "MOVE_STEP"]:
            val = getattr(game_module, const_name, None)
            if isinstance(val, int) and val > 0:
                step = val
                break

    # Detect pairing state attributes
    pair_attr = None
    for attr in ["nsevyuople", "paired", "_paired", "carrying"]:
        if hasattr(sim, attr) and isinstance(getattr(sim, attr), dict):
            pair_attr = attr
            break

    # Detect special action (pair/unpair button)
    # Try ACTION5 first, then ACTION6, etc.
    special_actions = [GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]

    def path_to_actions(path: list[tuple[int, int]]) -> list:
        actions = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            if dx > 0:
                actions.append(ActionInput(id=GameAction.ACTION4, data={}))
            elif dx < 0:
                actions.append(ActionInput(id=GameAction.ACTION3, data={}))
            elif dy > 0:
                actions.append(ActionInput(id=GameAction.ACTION2, data={}))
            elif dy < 0:
                actions.append(ActionInput(id=GameAction.ACTION1, data={}))
        return actions

    def get_player(sim_game):
        """Find player sprite."""
        sprites = sim_game.current_level.get_sprites() if sim_game.current_level else []
        for s in sprites:
            tags = getattr(s, "tags", [])
            if any(t in tags for t in ["wbmdvjhthc", "player", "sys_player"]):
                return s
        return sprites[0] if sprites else None

    def is_paired(sim_game, player):
        """Check if player is currently paired with something."""
        if pair_attr and player:
            return player in getattr(sim_game, pair_attr, {})
        return False

    def try_special(sim_game, action_id):
        """Try a special action and return True if it changed pairing state."""
        before = is_paired(sim_game, get_player(sim_game))
        sim_game.perform_action(ActionInput(id=action_id, data={}))
        after = is_paired(sim_game, get_player(sim_game))
        return before != after

    # Try each pathfinding method to see which ones work
    # Method that returns a path when not paired = "go to block"
    # Method that returns a path when paired = "go to target"
    go_to_block_methods = []
    go_to_target_methods = []

    player = get_player(sim)
    if player is None:
        return None

    for name in pf_names:
        method = pf_methods[name]
        try:
            path = method(player)
            if path is not None and isinstance(path, list) and len(path) > 1:
                go_to_block_methods.append(name)
        except Exception:
            pass

    if not go_to_block_methods:
        return None

    # If we can pair, test which methods work when paired
    if pair_attr:
        # Try pairing first to discover target-finding methods
        test_sim = _copy.deepcopy(sim)
        test_player = get_player(test_sim)
        # Navigate to block first
        for name in go_to_block_methods:
            method = getattr(test_sim, name)
            try:
                path = method(test_player)
                if path and len(path) > 1:
                    acts = path_to_actions(path)
                    for a in acts:
                        test_sim.perform_action(a)
                    break
            except Exception:
                continue

        # Try to pair
        for sa in special_actions:
            saved = _copy.deepcopy(test_sim)
            if try_special(test_sim, sa):
                # Found the pair action! Now test pathfinding methods
                test_player2 = get_player(test_sim)
                for name in pf_names:
                    if name in go_to_block_methods:
                        continue
                    method = getattr(test_sim, name)
                    try:
                        path = method(test_player2)
                        if path is not None and isinstance(path, list) and len(path) > 1:
                            go_to_target_methods.append(name)
                    except Exception:
                        pass
                # Also check if block methods work when paired
                for name in go_to_block_methods:
                    method = getattr(test_sim, name)
                    try:
                        path = method(test_player2)
                        if path is not None and isinstance(path, list) and len(path) > 1:
                            go_to_target_methods.append(name)
                    except Exception:
                        pass
                break
            else:
                _restore_game(test_sim, saved)

    # Main solve loop
    pair_action = None
    unpair_action = None

    # Detect pair/unpair actions by testing
    if pair_attr:
        for sa in special_actions:
            test_sim = _copy.deepcopy(sim)
            # First navigate to a block
            for name in go_to_block_methods:
                method = getattr(test_sim, name)
                tp = get_player(test_sim)
                try:
                    path = method(tp)
                    if path and len(path) > 1:
                        acts = path_to_actions(path)
                        for a in acts:
                            test_sim.perform_action(a)
                        break
                except Exception:
                    continue

            tp = get_player(test_sim)
            saved = _copy.deepcopy(test_sim)
            test_sim.perform_action(ActionInput(id=sa, data={}))
            tp2 = get_player(test_sim)
            if is_paired(test_sim, tp2) and not is_paired(saved, tp):
                pair_action = sa
                # Test unpair
                test_sim.perform_action(ActionInput(id=sa, data={}))
                tp3 = get_player(test_sim)
                if not is_paired(test_sim, tp3):
                    unpair_action = sa
                break
                _restore_game(test_sim, saved)

    # Execute solve loop
    for iteration in range(max_iter):
        if _time.time() - t0 > max_time:
            break

        if _is_level_solved(sim, original_level):
            return all_actions

        player = get_player(sim)
        if player is None:
            break

        paired = is_paired(sim, player)

        if paired and go_to_target_methods:
            # Navigate to target
            for name in go_to_target_methods:
                method = getattr(sim, name)
                try:
                    path = method(player)
                except Exception:
                    continue
                if path is not None and isinstance(path, list) and len(path) > 1:
                    acts = path_to_actions(path)
                    for a in acts:
                        sim.perform_action(a)
                        all_actions.append((a.id, {}))
                    break
            # Unpair
            if unpair_action:
                sim.perform_action(ActionInput(id=unpair_action, data={}))
                all_actions.append((unpair_action, {}))
        elif not paired and go_to_block_methods:
            # Navigate to block
            path_found = False
            for name in go_to_block_methods:
                method = getattr(sim, name)
                try:
                    path = method(player)
                except Exception:
                    continue
                if path is not None and isinstance(path, list) and len(path) > 1:
                    acts = path_to_actions(path)
                    for a in acts:
                        sim.perform_action(a)
                        all_actions.append((a.id, {}))
                    path_found = True
                    break

            if not path_found:
                break

            # Refresh player
            player = get_player(sim)

            # Fix rotation if needed
            if player and hasattr(player, "rotation"):
                all_sprites = sim.current_level.get_sprites() if sim.current_level else []
                for s in all_sprites:
                    if s is player:
                        continue
                    dist = abs(s.x - player.x) + abs(s.y - player.y)
                    if dist == step:
                        dx = s.x - player.x
                        dy = s.y - player.y
                        needed = 0
                        if dy < 0: needed = 0
                        elif dy > 0: needed = 180
                        elif dx > 0: needed = 90
                        elif dx < 0: needed = 270

                        if player.rotation != needed:
                            # Back-and-forth to fix rotation
                            if dx > 0:
                                away, back = GameAction.ACTION3, GameAction.ACTION4
                            elif dx < 0:
                                away, back = GameAction.ACTION4, GameAction.ACTION3
                            elif dy > 0:
                                away, back = GameAction.ACTION1, GameAction.ACTION2
                            elif dy < 0:
                                away, back = GameAction.ACTION2, GameAction.ACTION1
                            else:
                                break

                            saved = _copy.deepcopy(sim)
                            sim.perform_action(ActionInput(id=away, data={}))
                            all_actions.append((away, {}))
                            sim.perform_action(ActionInput(id=back, data={}))
                            all_actions.append((back, {}))
                        break

            # Pair
            if pair_action:
                sim.perform_action(ActionInput(id=pair_action, data={}))
                all_actions.append((pair_action, {}))
        else:
            # No pathfinding worked - try all special actions as last resort
            for sa in special_actions:
                sim.perform_action(ActionInput(id=sa, data={}))
                all_actions.append((sa, {}))
                if _is_level_solved(sim, original_level):
                    return all_actions
            break

    if _is_level_solved(sim, original_level):
        return all_actions

    return None


SOLVERS: dict[str, callable] = {
    "tu93": solve_tu93,
    "wa30": solve_wa30,
    "dc22": solve_dc22,
    "m0r0": solve_m0r0,
    "re86": solve_re86,
    "lp85": solve_lp85,
    "cd82": solve_cd82,
    "g50t": solve_g50t,
    "sb26": solve_sb26,
    "r11l": solve_r11l,
    "cn04": solve_cn04,
    "sp80": solve_sp80,
    "ar25": solve_ar25,
    "ka59": solve_ka59,
    "sc25": solve_sc25,
    "sk48": solve_sk48,
    "su15": solve_su15,
    "vc33": solve_vc33,
    "s5i5": solve_s5i5,
    "tn36": solve_tn36,
    "bp35": solve_bp35,
    "lf52": solve_lf52,
}


def _snap_click_coordinates(
    plan: list[tuple],
    game: Any,
) -> list[tuple]:
    """Snap click coordinates to nearest valid clickable position.

    Game solvers compute click positions from sprite coordinates, but the
    actual valid click positions are determined by the game's camera
    transform (scale, offset) and sprite rendering. This function corrects
    solver-computed coordinates to the nearest valid click position.

    Args:
        plan: Action plan with (GameAction, click_data|None) tuples.
        game: The env._game object.

    Returns:
        Corrected action plan with snapped click coordinates.
    """
    from arcengine import GameAction

    # Collect all click actions that need correction
    has_clicks = any(
        a in (GameAction.ACTION6, GameAction.ACTION7) and d is not None
        for a, d in plan
    )
    if not has_clicks:
        return plan

    # Get valid clickable actions from the game
    valid_clicks: list[dict] = []
    try:
        if hasattr(game, '_get_valid_clickable_actions'):
            for ai in game._get_valid_clickable_actions():
                if ai.data:
                    valid_clicks.append(ai.data)
    except (AttributeError, Exception):
        pass

    if not valid_clicks:
        return plan

    # Build set of valid click positions
    valid_positions: list[tuple[int, int]] = []
    for vc in valid_clicks:
        vx = vc.get('x', vc.get('X', 0))
        vy = vc.get('y', vc.get('Y', 0))
        valid_positions.append((int(vx), int(vy)))

    if not valid_positions:
        return plan

    # Snap each click to nearest valid position
    corrected_plan: list[tuple] = []
    for action, click_data in plan:
        if (action in (GameAction.ACTION6, GameAction.ACTION7)
                and click_data is not None):
            # Get solver-computed position
            if isinstance(click_data, (tuple, list)):
                sx, sy = int(click_data[0]), int(click_data[1])
            elif isinstance(click_data, dict):
                sx = int(click_data.get('x', click_data.get('X', 0)))
                sy = int(click_data.get('y', click_data.get('Y', 0)))
            else:
                corrected_plan.append((action, click_data))
                continue

            # Find nearest valid position
            best_dist = float('inf')
            best_pos = (sx, sy)
            for vx, vy in valid_positions:
                dist = (sx - vx) ** 2 + (sy - vy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (vx, vy)

            # Only snap if within reasonable distance (max 10 pixels)
            if best_dist <= 100:  # 10 pixel radius
                corrected_plan.append((action, (best_pos[0], best_pos[1])))
            else:
                corrected_plan.append((action, click_data))
        else:
            corrected_plan.append((action, click_data))

    return corrected_plan


def solve_game(
    game: Any,
    game_id: str,
    level_idx: int = 0,
) -> list[tuple] | None:
    """Dispatch to game-specific solver.

    Tries generic simulation-based solvers first (DFS search for small
    action spaces, game's internal pathfinding for keyboard games),
    then falls back to heuristic solvers.

    Args:
        game: The env._game object.
        game_id: Game identifier (e.g., "tu93", "wa30").
        level_idx: Current level index.

    Returns:
        Action plan as list of (GameAction, click_data|None) tuples,
        or None if no solver is available or solver fails.
    """
    base_id = game_id.split("-")[0] if game_id else ""

    valid_actions = _get_valid_action_inputs(game)
    n_actions = len(valid_actions)
    original_level = game._current_level_index

    def _normalize_plan(plan: list[tuple] | None) -> list[tuple] | None:
        """Normalize click data to dict format."""
        if not plan:
            return None
        normalized = []
        for action, click_data in plan:
            if click_data is not None and isinstance(click_data, (tuple, list)):
                normalized.append((action, {"x": int(click_data[0]), "y": int(click_data[1])}))
            else:
                normalized.append((action, click_data))
        return normalized

    def _verify_plan(plan: list[tuple] | None) -> bool:
        """Verify plan solves the level by replaying on a fresh deepcopy."""
        if not plan:
            return False
        import copy as _vc
        from arcengine import ActionInput
        try:
            sim = _vc.deepcopy(game)
            for aid, data in plan[:300]:
                ai = ActionInput(id=aid, data=data if data else {})
                sim.perform_action(ai)
                if _is_level_solved(sim, original_level):
                    return True
            return _is_level_solved(sim, original_level)
        except Exception:
            return False

    # Phase 1: Try generic DFS solver for ALL games
    try:
        import copy as _copy
        sim = _copy.deepcopy(game)
        dfs_depth = 30 if n_actions <= 4 else 20
        dfs_nodes = 100000 if n_actions <= 4 else 50000
        plan = solve_generic_dfs(sim, max_depth=dfs_depth, max_nodes=dfs_nodes, max_time=12.0)
        plan = _normalize_plan(plan)
        if plan is not None and _verify_plan(plan):
            return plan
    except Exception:
        pass

    # Phase 2: Try generic keyboard solver (uses game's internal pathfinding)
    if n_actions > 2:
        try:
            plan = solve_generic_keyboard(game, max_iter=50, max_time=8.0)
            plan = _normalize_plan(plan)
            if plan is not None and _verify_plan(plan):
                return plan
        except Exception:
            pass

    # Phase 3: Fall back to heuristic solver
    solver = SOLVERS.get(base_id)
    if solver is not None:
        try:
            plan = solver(game, level_idx)
            if plan is not None:
                plan = _snap_click_coordinates(plan, game)
                plan = _normalize_plan(plan)
                if plan is not None and _verify_plan(plan):
                    return plan
        except Exception as e:
            print(f"    [GAME-SOLVER] {base_id} error: {e}")

    # Phase 4: Last resort - return unverified DFS/keyboard plan if we have one
    # (better than nothing - the planner agent might still work with it)
    if plan:
        return plan

    return None
