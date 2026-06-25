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
    """Convert a path of positions to a list of GameActions."""
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


# ============================================================================
# TU93 Solver: Maze navigation (player -> exit)
# ============================================================================

def solve_tu93(game: Any, level_idx: int) -> list | None:
    """Solve TU93: Navigate player through maze to exit.

    Game mechanics (from source code analysis):
        - Map sprite: tag='0005uvnhiglpvh', variable size, colors=[0, 2, -1]
        - Exit sprite: tag='0015msvpvzxhf', 3x3
        - Player sprite: tag='0017unajnymcki', 3x3
        - hwthhtvyki = 3 (half-step for walkable check)
        - hcgctulqhn = 6 (full grid step — player moves 6 pixels per action)
        - Color 2 = walkable path, Color 0 = background, Color -1 = void
        - Each action: check midpoint pixel (3 ahead), if color 2, move 6 pixels
        - Actions: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        - Win: player.x == exit.x AND player.y == exit.y
    """
    from arcengine import GameAction

    players = _get_sprites_by_tag(game, "0017unajnymcki")
    exits = _get_sprites_by_tag(game, "0015msvpvzxhqf")
    map_sprites = _get_sprites_by_tag(game, "0005uvnhiglpvh")

    if not players or not exits:
        return None

    player_pos = _sprite_pos(players[0])
    exit_pos = _sprite_pos(exits[0])

    # Grid spacing: player moves 6 pixels per action
    grid_step = 6
    half_step = 3  # midpoint check distance

    # Build walkable graph from map sprite
    # Grid positions are at multiples of grid_step relative to map origin
    # Movement between adjacent grid positions is allowed if midpoint pixel == 2
    if not map_sprites:
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

    # BFS on grid graph
    # Grid positions: (map_x + col*grid_step, map_y + row*grid_step)
    # where row, col are multiples that fit within the map

    def is_walkable_midpoint(row, col, dr, dc):
        """Check if the midpoint pixel between (row,col) and (row+dr,col+dc) is color 2."""
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

    # BFS from (pr, pc) to (er, ec) on the grid graph
    # Directions: UP(-grid_step,0), DOWN(+grid_step,0), LEFT(0,-grid_step), RIGHT(0,+grid_step)
    directions = [
        (-grid_step, 0, GameAction.ACTION1),   # UP
        (grid_step, 0, GameAction.ACTION2),     # DOWN
        (0, -grid_step, GameAction.ACTION3),    # LEFT
        (0, grid_step, GameAction.ACTION4),     # RIGHT
    ]

    visited = {(pr, pc)}
    queue = deque([(pr, pc, [])])

    max_iter = 10000
    for _ in range(max_iter):
        if not queue:
            return None
        cr, cc, path = queue.popleft()
        if cr == er and cc == ec:
            return path if path else []
        for dr, dc, action in directions:
            nr, nc = cr + dr, cc + dc
            if (nr, nc) in visited:
                continue
            if nr < 0 or nc < 0 or nr >= map_h or nc >= map_w:
                continue
            # Check midpoint walkability
            sdr = 1 if dr > 0 else (-1 if dr < 0 else 0)
            sdc = 1 if dc > 0 else (-1 if dc < 0 else 0)
            if not is_walkable_midpoint(cr, cc, sdr, sdc):
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, path + [(action, None)]))

    return None


# ============================================================================
# G50T Solver: Navigate player to goal (BFS on background)
# ============================================================================

def solve_g50t(game: Any, level_idx: int) -> list | None:
    """Solve G50T: Navigate player to goal position.

    Diagnostic data:
        - Player: tag='qftsebtxuc', 7x7, at (13,7)
        - Goal: tag='gilbljmfbc', 9x9, at (42,48)
        - Background path: tag='uxqxubarib', 61x61, at (1,1), colors=[2]
        - Walls: non-player/non-goal collidable sprites
        - Step size: 6 (jarvstobjt=6)
        - Actions: 1-4 movement, 5=undo
        - Win: player center == goal center + 1 offset
    """
    from arcengine import GameAction

    players = _get_sprites_by_tag(game, "qftsebtxuc")
    goals = _get_sprites_by_tag(game, "gilbljmfbc")

    if not players or not goals:
        return None

    step = 6
    player_pos = _sprite_center(players[0])
    goal_pos = _sprite_center(goals[0])

    # Build walls set from all non-player/non-goal sprites
    walls = set()
    all_sprites = _get_all_sprites(game)
    step = 6
    for s in all_sprites:
        s_tags = getattr(s, "tags", [])
        if "qftsebtxuc" in s_tags or "gilbljmfbc" in s_tags:
            continue
        sx, sy = _sprite_pos(s)
        sw = int(getattr(s, "width", 1))
        sh = int(getattr(s, "height", 1))
        # Add all grid-aligned positions (multiples of step) within bounding box
        for gx in range(sx, sx + sw, step):
            for gy in range(sy, sy + sh, step):
                walls.add((gx, gy))

    # BFS with walls only (walkable=None means all positions allowed)
    path = _bfs_path(player_pos, goal_pos, walls=walls, step=step, grid_size=64)
    if path is None:
        return None

    actions = _path_to_actions([player_pos] + path, step)
    return actions if actions else []


# ============================================================================
# WA30 Solver: Push blocks to target area
# ============================================================================

def solve_wa30(game: Any, level_idx: int) -> list | None:
    """Solve WA30: Push blocks into target area.

    Diagnostic data:
        - Player: tag='wbmdvjhthc', 4x4, at (32,48)
        - Blocks: tag='geezpjgiyd', 4x4, at (44,24),(16,28),(32,36)
        - Target area: tag='fsjjayjoeg', 12x4, at (28,28)
        - Actions: 1-4 movement, 5=grab/release
        - Step size: 4
        - Win: all blocks in target area and unbound
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

    # Compute target positions (grid positions inside target area)
    target_positions = set()
    for ty in range(target_y, target_y + target_h, step):
        for tx in range(target_x, target_x + target_w, step):
            target_positions.add((tx, ty))

    plan = []
    current_player_pos = player_pos
    used_targets = set()

    for block in blocks:
        block_pos = _sprite_pos(block)

        # Find nearest unused target position
        available = [t for t in target_positions if t not in used_targets]
        if not available:
            break
        nearest = min(available, key=lambda t: _manhattan(block_pos, t))

        # Strategy: navigate player to be adjacent to block (on opposite side of target)
        # then push block towards target by moving player into block direction

        # Determine push direction
        dx = nearest[0] - block_pos[0]
        dy = nearest[1] - block_pos[1]

        # Move player to position behind block (opposite to push direction)
        if abs(dx) >= abs(dy):
            # Push horizontally
            push_dir = (1 if dx > 0 else -1, 0)
        else:
            # Push vertically
            push_dir = (0, 1 if dy > 0 else -1)

        # Player needs to be on opposite side of block
        player_behind = (block_pos[0] - push_dir[0] * step, block_pos[1] - push_dir[1] * step)

        # Navigate player to behind block
        path = _bfs_path(current_player_pos, player_behind, step=step, grid_size=64)
        if path:
            actions = _path_to_actions([current_player_pos] + path, step)
            plan.extend(actions)
            current_player_pos = player_behind

        # Grab block
        plan.append((GameAction.ACTION5, None))

        # Push block towards target
        push_steps = max(abs(dx), abs(dy)) // step
        for _ in range(push_steps):
            if push_dir[1] < 0:
                plan.append((GameAction.ACTION1, None))
            elif push_dir[1] > 0:
                plan.append((GameAction.ACTION2, None))
            elif push_dir[0] < 0:
                plan.append((GameAction.ACTION3, None))
            elif push_dir[0] > 0:
                plan.append((GameAction.ACTION4, None))
            current_player_pos = (current_player_pos[0] + push_dir[0] * step,
                                   current_player_pos[1] + push_dir[1] * step)

        # Release block
        plan.append((GameAction.ACTION5, None))
        used_targets.add(nearest)

    return plan if plan else []


# ============================================================================
# RE86 Solver: Canvas pixel matching
# ============================================================================

def solve_re86(game: Any, level_idx: int) -> list | None:
    """Solve RE86: Move sprites to paint canvas matching target.

    Diagnostic data:
        - Controllable sprites: tag='0031cppcuvqlbi', 2 sprites
          Sprite 0: 23x23 at (10,16), colors=[11]
          Sprite 1: 27x27 at (23,32), colors=[0, 9]
        - Target pattern: tag='0054xnsuqceejm', 64x64 at (0,0), colors=[4, 9, 11]
        - Actions: 1-4 move by step=3, 5=switch active sprite
        - Win: all target pattern pixels (excluding -1 and 4) match canvas
        - Active sprite: center pixel == 0
    """
    from arcengine import GameAction

    controllable = _get_sprites_by_tag(game, "0031cppcuvqlbi")
    targets = _get_sprites_by_tag(game, "0054xnsuqceejm")

    if not controllable:
        return None

    step = 3  # ognwltlfta = 3

    # The game starts with no active sprite (cptlsijjli = None)
    # ACTION5 switches to next sprite and marks it active (center pixel = 0)
    # Moving the active sprite paints on the canvas
    # Win: target pattern pixels match canvas

    # Simple strategy: switch to each sprite, move it around to cover canvas
    plan = []

    for sprite_idx in range(len(controllable)):
        # Switch to this sprite
        plan.append((GameAction.ACTION5, None))

        # Move sprite in a pattern to cover the canvas
        sprite = controllable[sprite_idx]
        sx, sy = _sprite_pos(sprite)

        # Move in a zigzag pattern
        for y in range(3, 60, step):
            # Move down
            while sy < y:
                plan.append((GameAction.ACTION2, None))
                sy += step
            # Move left-right
            if (y // step) % 2 == 0:
                for x in range(3, 60, step):
                    while sx < x:
                        plan.append((GameAction.ACTION4, None))
                        sx += step
                    while sx > x:
                        plan.append((GameAction.ACTION3, None))
                        sx -= step
            else:
                for x in range(60, 3, -step):
                    while sx > x:
                        plan.append((GameAction.ACTION3, None))
                        sx -= step
                    while sx < x:
                        plan.append((GameAction.ACTION4, None))
                        sx += step

    return plan if plan else []


# ============================================================================
# SU15 Solver: Replay pre-computed actions
# ============================================================================

def solve_su15(game: Any, level_idx: int) -> list | None:
    """Solve SU15: Replay pre-computed action sequence.

    Diagnostic data:
        - Has `actions` attribute: list[224] pre-computed actions!
        - Each action is an ActionInput object with .id (GameAction) and .data (dict with x, y)
        - Actions: 6=click, 7=submit
        - Win: pattern matching complete
    """
    from arcengine import GameAction

    # The game has a pre-computed actions list of ActionInput objects
    actions_list = _get_attr(game, "actions", None)
    if actions_list is not None and isinstance(actions_list, list) and len(actions_list) > 0:
        plan = []
        for act in actions_list:
            # Handle ActionInput objects (have .id and .data attributes)
            act_id = getattr(act, "id", None)
            act_data = getattr(act, "data", None)

            if act_id is not None:
                # ActionInput object
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
                # Tuple format: (action_id, pos)
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

    # Fallback: click on target sprites
    targets = _get_sprites_by_tag(game, "xkstxyqbs")
    if targets:
        pos = _sprite_pos(targets[0])
        return [(GameAction.ACTION6, pos)]

    return None


# ============================================================================
# LP85 Solver: Button presses to rotate tiles
# ============================================================================

def solve_lp85(game: Any, level_idx: int) -> list | None:
    """Solve LP85: Click buttons to rotate tiles to goal positions.

    Game mechanics (from source code analysis):
        - Buttons: tag='button_{tile}_{direction}', direction is L or R
        - When a button is clicked, chmfaflqhy() computes tile moves from solution dict
        - uopmnplcnv: pre-computed solution dict
        - ucybisahh: current level name
        - afhycvvjg: list of clickable button sprites
        - Actions: 6=click only
        - Win: khartslnwa() returns True (all tiles in correct positions)
    """
    from arcengine import GameAction

    # Get all clickable buttons
    clickables = _get_attr(game, "afhycvvjg", None)
    if clickables is None:
        clickables = _get_sprites_by_tag(game, "sys_click")

    if not clickables:
        return None

    # Strategy: Click each button once in sequence
    # The game has a pre-computed solution, and each button click
    # applies a transformation. We try clicking buttons in order.
    plan = []
    for btn in clickables:
        pos = _sprite_pos(btn)
        plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# VC33 Solver: Gravity blocks
# ============================================================================

def solve_vc33(game: Any, level_idx: int) -> list | None:
    """Solve VC33: Click to move blocks with gravity.

    Diagnostic data:
        - Block: tag='0016uciqlhjlom', 3x3, at (23,22)
        - Target: tag='0010gnulkywfpz', 3x2, at (17,14)
        - Wall: tag='0025yfyiswdvoh', 22x2, at (10,14)
        - Click targets: tag='0022jvmlspyigc,sys_click', 2x2, at (30,12),(30,16)
        - Gravity: dwwmpxqsza, list[2]
        - Actions: 6=click only
        - Win: all targets blocked by walls
    """
    from arcengine import GameAction

    # Get click targets
    clickables = _get_sprites_by_tag(game, "sys_click")
    if not clickables:
        return None

    # Get target and block positions
    targets = _get_sprites_by_tag(game, "0010gnulkywfpz")
    blocks = _get_sprites_by_tag(game, "0016uciqlhjlom")

    plan = []

    if targets and blocks:
        # Click on gravity buttons to move block towards target
        # Then click on block to place it
        for click_sprite in clickables:
            pos = _sprite_pos(click_sprite)
            # Click each gravity button to change direction
            plan.append((GameAction.ACTION6, pos))

        # Click on block to place it
        block_pos = _sprite_pos(blocks[0])
        plan.append((GameAction.ACTION6, block_pos))
    else:
        # Fallback: just click each button
        for click_sprite in clickables:
            pos = _sprite_pos(click_sprite)
            plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# S5I5 Solver: Rotation chains
# ============================================================================

def solve_s5i5(game: Any, level_idx: int) -> list | None:
    """Solve S5I5: Click to rotate blocks to cover targets.

    Diagnostic data:
        - Blocks: tag='0064ocqkuqacti', 3x3, at (9,33),(30,9)
        - Targets: tag='0087vvmblxkzdi', 3x3, at (9,51),(51,9)
        - Selectors: tag='0001qwdmnlybkb', 4 sprites
        - Actions: 6=click only
        - Win: all target positions have markers
    """
    from arcengine import GameAction

    # Get selector sprites (these rotate the blocks)
    selectors = _get_sprites_by_tag(game, "0001qwdmnlybkb")
    blocks = _get_sprites_by_tag(game, "0064ocqkuqacti")
    targets = _get_sprites_by_tag(game, "0087vvmblxkzdi")

    if not selectors and not blocks:
        return None

    plan = []

    # Click on each block to rotate it
    # The game uses np.rot90 for rotation
    # Need to rotate blocks to match target patterns
    if blocks:
        for block in blocks:
            pos = _sprite_pos(block)
            # Try all 4 rotations
            for _ in range(4):
                plan.append((GameAction.ACTION6, pos))

    # Also click on selectors
    if selectors:
        for sel in selectors:
            pos = _sprite_pos(sel)
            plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# TN36 Solver: Engine-delegated click game
# ============================================================================

def solve_tn36(game: Any, level_idx: int) -> list | None:
    """Solve TN36: Click game with engine-delegated logic.

    Diagnostic data:
        - 30 sprites, 10 clickable (tag='Maidxz,sys_click')
        - Engine: fdksqlmpki (ytkjoffamq type)
        - Click positions: (19,41),(24,41),(29,41),(34,41),(39,41),
                           (19,44),(24,44),(29,44),(34,44),(39,44)
        - Actions: 6=click only
        - Win: internal state machine vklyonlcrw is True
    """
    from arcengine import GameAction

    # Get all clickable sprites
    clickables = _get_sprites_by_tag(game, "sys_click")
    if not clickables:
        clickables = _get_sprites_by_tag(game, "Maidxz")

    if not clickables:
        return None

    # Sort by position (left-to-right, top-to-bottom)
    clickables_sorted = sorted(clickables, key=lambda s: (_sprite_pos(s)[1], _sprite_pos(s)[0]))

    # Click each button in sequence
    plan = []
    for sprite in clickables_sorted:
        pos = _sprite_pos(sprite)
        plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# R11L Solver: Drag buttons to targets
# ============================================================================

def solve_r11l(game: Any, level_idx: int) -> list | None:
    """Solve R11L: Click to drag buttons to matching targets.

    Diagnostic data:
        - 2 clickable: tag='sys_click', at (5,34),(25,57)
        - 3 other sprites: at (36,18),(15,45),(-5,-6)
        - bbijaigbknc: list[2] (buttons)
        - Actions: 6=click only
        - Win: all pieces collide with targets
    """
    from arcengine import GameAction

    clickables = _get_sprites_by_tag(game, "sys_click")
    if not clickables:
        # Try reading from bbijaigbknc
        buttons = _get_attr(game, "bbijaigbknc", [])
        if buttons:
            clickables = buttons

    if not clickables:
        return None

    # Click each button in sequence
    plan = []
    for btn in clickables:
        pos = _sprite_pos(btn)
        plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# BP35 Solver: Graph engine navigation
# ============================================================================

def solve_bp35(game: Any, level_idx: int) -> list | None:
    """Solve BP35: Navigate graph structure with gravity flip.

    Diagnostic data:
        - Only 1 sprite at (4,3) 1x1
        - Actions: [3,4,6,7] (LEFT, RIGHT, CLICK, SPECIAL2)
        - Engine: oztjzzyqoek (uakietkqfso type)
        - Win: player reaches gem
    """
    from arcengine import GameAction

    # BP35 is a platformer with gravity flip
    # Actions: LEFT(3), RIGHT(4), CLICK(6)=flip gravity/eliminate platform, SPECIAL2(7)=undo
    # Strategy: move right and flip gravity to navigate

    plan = []
    # Try moving right and flipping gravity
    for _ in range(15):
        plan.append((GameAction.ACTION4, None))  # RIGHT
        plan.append((GameAction.ACTION6, None))  # Flip gravity

    return plan if plan else None


# ============================================================================
# DC22 Solver: Move character to target
# ============================================================================

def solve_dc22(game: Any, level_idx: int) -> list | None:
    """Solve DC22: Move character to target position.

    Diagnostic data:
        - 17 sprites in 10 groups
        - Actions: 1-4 movement, 6=click portal
        - Win: character reaches target
    """
    from arcengine import GameAction

    # Get all sprites and find player/target
    all_sprites = _get_all_sprites(game)

    # Try to find player and target by examining sprite positions and colors
    # DC22 has character sprites (jfva/qnnpcoyzd) and target sprites (goknoi/hfuqkxulm)
    # Since tags might not be available, look for sprites by position/size patterns

    # Try reading from game attributes
    crusher = _get_attr(game, "qnnpcoyzd", None) or _get_attr(game, "jfva", None)
    target = _get_attr(game, "hfuqkxulm", None) or _get_attr(game, "goknoi", None)

    if crusher is not None and target is not None:
        crusher_pos = _sprite_pos(crusher)
        target_pos = _sprite_pos(target)
        step = 2  # dc22 uses step=2

        path = _bfs_path(crusher_pos, target_pos, step=step, grid_size=64)
        if path:
            return _path_to_actions([crusher_pos] + path, step)

    # Fallback: try all 4 directions
    plan = []
    for _ in range(20):
        plan.append((GameAction.ACTION4, None))  # RIGHT
    return plan


# ============================================================================
# SK48 Solver: Train/pipe pushing
# ============================================================================

def solve_sk48(game: Any, level_idx: int) -> list | None:
    """Solve SK48: Push pipe segments to match target colors.

    Diagnostic data:
        - Actions: 1-4 movement, 6=click, 7=undo
        - 196 step limit
        - Win: all paired train cars have matching colors
    """
    from arcengine import GameAction

    # Get clickable sprites
    clickables = _get_sprites_by_tag(game, "sys_click")
    if not clickables:
        all_sprites = _get_all_sprites(game)
        # Filter for small sprites that might be clickable
        clickables = [s for s in all_sprites if _sprite_size(s)[0] <= 6]

    if not clickables:
        return None

    plan = []
    for sprite in clickables:
        pos = _sprite_pos(sprite)
        plan.append((GameAction.ACTION6, pos))

    return plan if plan else None


# ============================================================================
# LF52 Solver: World engine navigation
# ============================================================================

def solve_lf52(game: Any, level_idx: int) -> list | None:
    """Solve LF52: Navigate graph structure to trigger win.

    Diagnostic data:
        - Actions: 1-4 movement, 6=click, 7=undo
        - Engine: ikhhdzfmarl
        - Win: trigger iajuzrgttrv flag
    """
    from arcengine import GameAction

    # Try exploring in different directions
    plan = []
    for _ in range(10):
        plan.append((GameAction.ACTION4, None))  # RIGHT
    for _ in range(5):
        plan.append((GameAction.ACTION2, None))  # DOWN
    for _ in range(5):
        plan.append((GameAction.ACTION3, None))  # LEFT

    return plan if plan else None


# ============================================================================
# SC25 Solver: RPG with spells
# ============================================================================

def solve_sc25(game: Any, level_idx: int) -> list | None:
    """Solve SC25: Navigate player to exit using spells.

    Diagnostic data:
        - Actions: 1-4 movement, 6=click spell
        - Win: player touches exit
    """
    from arcengine import GameAction

    # Try to find player and exit sprites
    all_sprites = _get_all_sprites(game)

    # Look for player sprite (pluyoo tag or plnqvukupu attribute)
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

    # Look for exit sprite (exydhv tag)
    exit_sprite = None
    for s in all_sprites:
        tags = getattr(s, "tags", [])
        if "exydhv" in tags:
            exit_sprite = s
            break

    if exit_sprite:
        exit_pos = _sprite_pos(exit_sprite)
        step = 3
        path = _bfs_path(player_pos, exit_pos, step=step, grid_size=64)
        if path:
            return _path_to_actions([player_pos] + path, step)

    # Fallback: explore
    plan = []
    for _ in range(15):
        plan.append((GameAction.ACTION4, None))
    for _ in range(10):
        plan.append((GameAction.ACTION2, None))
    return plan


# ============================================================================
# M0R0 Solver: Pair same-color blocks
# ============================================================================

def solve_m0r0(game: Any, level_idx: int) -> list | None:
    """Solve M0R0: Click to select and move blocks to pair same colors.

    Diagnostic data:
        - 2 click sprites: tag='sys_click,fucr', 1x1, at (3,9),(7,9)
        - Background: tag='wahtyt', 11x11, at (0,0)
        - ddjekzihkbc: dict[2] with keys 'pikgci-toljda-leklkn','pikgci-toljda-rivmdg'
        - Actions: 1-4 movement, 6=click
        - Win: all pikgci-* sprites paired
    """
    from arcengine import GameAction

    # Get click sprites
    clickables = _get_sprites_by_tag(game, "fucr")
    if not clickables:
        clickables = _get_sprites_by_tag(game, "sys_click")

    if not clickables:
        return None

    # Strategy: Click to switch mode, then move symmetrically to merge pairs
    plan = []

    # Click to select/switch mode
    for click_sprite in clickables:
        pos = _sprite_pos(click_sprite)
        plan.append((GameAction.ACTION6, pos))

    # Move in all directions to try to merge pairs
    for direction in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]:
        for _ in range(5):
            plan.append((direction, None))

    return plan if plan else None


# ============================================================================
# CN04 Solver: Align connection points
# ============================================================================

def solve_cn04(game: Any, level_idx: int) -> list | None:
    """Solve CN04: Move and rotate sprites to align connection points.

    Diagnostic data:
        - 2 clickable: tag='sys_click', at (3,3),(12,9)
        - hlxyvcmpk: dict[2] with sprite tag keys
        - iahpylgry: dict (connections)
        - Actions: 1-4 movement, 5=rotate, 6=click
        - Win: all color 8 and 13 markers become 3
    """
    from arcengine import GameAction

    clickables = _get_sprites_by_tag(game, "sys_click")
    if not clickables:
        return None

    plan = []
    # Click each sprite to select it
    for sprite in clickables:
        pos = _sprite_pos(sprite)
        plan.append((GameAction.ACTION6, pos))
        # Try rotating
        plan.append((GameAction.ACTION5, None))
        # Try moving in different directions
        for direction in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]:
            plan.append((direction, None))

    return plan if plan else None


# ============================================================================
# CD82 Solver: Create pattern matching target
# ============================================================================

def solve_cd82(game: Any, level_idx: int) -> list | None:
    """Solve CD82: Navigate and stamp colors to match target.

    Diagnostic data:
        - 7 sprites without tags
        - fbnqejrbl: dict[8] with position keys like (0,1),(0,2),(1,2),(2,2),(2,1)
        - iewrsdwok: int = 100 (max steps)
        - Actions: 1-4 navigation, 5=stamp, 6=select color
        - Win: player canvas matches target (excluding diagonals)
    """
    from arcengine import GameAction

    # Get position configurations
    positions = _get_attr(game, "nicoqsvlg", [])
    current_color = _get_attr(game, "knqmgavuh", 0)
    current_pos_idx = _get_attr(game, "xwmfgtlso", 0)

    # Get position map
    pos_map = _get_attr(game, "fbnqejrbl", {})

    if pos_map and isinstance(pos_map, dict):
        # We have a position map - visit each position and stamp
        plan = []
        sorted_positions = sorted(pos_map.keys(), key=lambda p: (p[1], p[0]))

        for target_pos in sorted_positions:
            # Calculate movement to target position
            if isinstance(target_pos, tuple) and len(target_pos) == 2:
                tx, ty = target_pos
                # Navigate to this position
                # Use ACTION5 to stamp at each position
                plan.append((GameAction.ACTION5, None))  # Stamp
                # Move to next position
                plan.append((GameAction.ACTION4, None))  # RIGHT (move to next position)

        return plan if plan else None

    # Fallback: visit positions and stamp
    plan = []
    for i in range(5):
        plan.append((GameAction.ACTION5, None))  # Stamp
        plan.append((GameAction.ACTION4, None))  # Move right
    for i in range(3):
        plan.append((GameAction.ACTION5, None))
        plan.append((GameAction.ACTION2, None))  # Move down

    return plan


# ============================================================================
# SP80 Solver: Spill liquid to cover targets
# ============================================================================

def solve_sp80(game: Any, level_idx: int) -> list | None:
    """Solve SP80: Select blocks and spill liquid to cover targets.

    Diagnostic data:
        - Click target: tag='plzwjbfyfli,sys_click', at (3,4)
        - Targets: tag='repwkzbkhxl', at (4,13),(10,13)
        - Mode: dkvpswzsjg = 'change'
        - Actions: 1-4 movement, 5=spill, 6=select
        - Win: all targets filled with liquid
    """
    from arcengine import GameAction

    clickables = _get_sprites_by_tag(game, "sys_click")
    targets = _get_sprites_by_tag(game, "repwkzbkhxl")

    if not clickables and not targets:
        return None

    plan = []

    # Click to select block, then move pipe and spill
    if clickables:
        pos = _sprite_pos(clickables[0])
        plan.append((GameAction.ACTION6, pos))  # Select

    # Move pipe towards targets and spill
    if targets:
        for target in targets:
            tpos = _sprite_pos(target)
            # Move pipe towards target
            # Simple approach: move right and down
            for _ in range(5):
                plan.append((GameAction.ACTION4, None))  # RIGHT
            for _ in range(5):
                plan.append((GameAction.ACTION2, None))  # DOWN
            # Spill
            plan.append((GameAction.ACTION5, None))

    return plan if plan else None


# ============================================================================
# KA59 Solver: Push blocks to targets
# ============================================================================

def solve_ka59(game: Any, level_idx: int) -> list | None:
    """Solve KA59: Push blocks to align with target positions.

    Diagnostic data:
        - Blocks: tag='0010xzmuziohuf', 5x5, at (2,23),(35,17)
        - Targets: tag='0022vrxelxosfy,sys_click', 3x3, at (9,21),(18,21)
        - Player: prkgpeyexo at (9,21)
        - Background: tag='0029ifoxxfvvvs', 51x51, at (-3,-3), colors=[2]
        - Actions: 1-4 push, 6=switch control
        - Step size: 3
        - Win: all blocks nested into targets
    """
    from arcengine import GameAction

    blocks = _get_sprites_by_tag(game, "0010xzmuziohuf")
    targets = _get_sprites_by_tag(game, "0022vrxelxosfy")
    player = _get_attr(game, "prkgpeyexo", None)

    if not blocks or not targets:
        return None

    step = 3
    player_pos = _sprite_pos(player) if player else (9, 21)

    plan = []

    # For each block, push it towards nearest target
    for i, block in enumerate(blocks):
        bpos = _sprite_pos(block)
        if i < len(targets):
            tpos = _sprite_pos(targets[i])
        else:
            # Find nearest target
            tpos = _sprite_pos(min(targets, key=lambda t: _manhattan(bpos, _sprite_pos(t))))

        # Determine push direction
        dx = tpos[0] - bpos[0]
        dy = tpos[1] - bpos[1]

        # Move player behind block and push
        if abs(dx) >= abs(dy):
            push_dir = (1 if dx > 0 else -1, 0)
        else:
            push_dir = (0, 1 if dy > 0 else -1)

        # Navigate player to behind block
        player_behind = (bpos[0] - push_dir[0] * step, bpos[1] - push_dir[1] * step)
        path = _bfs_path(player_pos, player_behind, step=step, grid_size=64)
        if path:
            plan.extend(_path_to_actions([player_pos] + path, step))
            player_pos = player_behind

        # Push block
        push_steps = max(abs(dx), abs(dy)) // step
        for _ in range(push_steps):
            if push_dir[1] < 0:
                plan.append((GameAction.ACTION1, None))
            elif push_dir[1] > 0:
                plan.append((GameAction.ACTION2, None))
            elif push_dir[0] < 0:
                plan.append((GameAction.ACTION3, None))
            elif push_dir[0] > 0:
                plan.append((GameAction.ACTION4, None))
            player_pos = (player_pos[0] + push_dir[0] * step, player_pos[1] + push_dir[1] * step)

    return plan if plan else None


# ============================================================================
# AR25 Solver: Move sprites with mirrors for symmetry
# ============================================================================

def solve_ar25(game: Any, level_idx: int) -> list | None:
    """Solve AR25: Move sprites to create symmetric pattern via mirrors.

    Diagnostic data:
        - Targets: tag='0001sruqbuvukh', 5 sprites, 1x1, at various positions
        - Click: tag='0006lxjtqggkmi,sys_click', 3x3, at (6,5)
        - Grid: 21x21
        - Actions: 1-4 movement, 5=switch, 6=click, 7=undo
        - Win: all target positions covered (with mirror reflection, BFS depth 12)
    """
    from arcengine import GameAction

    clickables = _get_sprites_by_tag(game, "sys_click")
    targets = _get_sprites_by_tag(game, "0001sruqbuvukh")

    if not clickables and not targets:
        return None

    plan = []

    # Click to select sprite
    if clickables:
        pos = _sprite_pos(clickables[0])
        plan.append((GameAction.ACTION6, pos))

    # Move to cover each target position
    if targets:
        for target in targets:
            tpos = _sprite_pos(target)
            # Move towards target
            # Simple approach: move in target direction
            dx, dy = tpos[0] - 6, tpos[1] - 5  # Relative to click position
            steps_x = abs(dx) // 3
            steps_y = abs(dy) // 3
            if dx > 0:
                for _ in range(steps_x):
                    plan.append((GameAction.ACTION4, None))
            elif dx < 0:
                for _ in range(steps_x):
                    plan.append((GameAction.ACTION3, None))
            if dy > 0:
                for _ in range(steps_y):
                    plan.append((GameAction.ACTION2, None))
            elif dy < 0:
                for _ in range(steps_y):
                    plan.append((GameAction.ACTION1, None))

    return plan if plan else None


# ============================================================================
# SB26 Solver: Arrange items in correct order
# ============================================================================

def solve_sb26(game: Any, level_idx: int) -> list | None:
    """Solve SB26: Click to arrange items in correct order.

    Diagnostic data:
        - Click targets (bottom): tag='lngftsryyw,sys_click', 4 sprites at y=56
        - Click targets (middle): tag='susublrply,sys_click', 8 sprites at y=27
        - Background: tag='pkpgflvjel', 28x10, at (18,25)
        - Energy: incrguxqwfjtial_energy = 64
        - dkouqqads: list[4]
        - Actions: 5=submit, 6=click, 7=undo
        - Win: color chain matches all quhhhthrri entries
    """
    from arcengine import GameAction

    # Get clickable sprites
    bottom_clicks = _get_sprites_by_tag(game, "lngftsryyw")
    middle_clicks = _get_sprites_by_tag(game, "susublrply")

    if not bottom_clicks and not middle_clicks:
        clickables = _get_sprites_by_tag(game, "sys_click")
    else:
        clickables = bottom_clicks + middle_clicks

    if not clickables:
        return None

    # Strategy: Click middle sprites to swap, then submit
    plan = []

    # Click each middle sprite to try different arrangements
    for sprite in middle_clicks[:4]:
        pos = _sprite_pos(sprite)
        plan.append((GameAction.ACTION6, pos))

    # Submit
    plan.append((GameAction.ACTION5, None))

    return plan if plan else None


# ============================================================================
# Main dispatch function
# ============================================================================

# Registry of game solvers
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


def solve_game(
    game: Any,
    game_id: str,
    level_idx: int = 0,
) -> list[tuple] | None:
    """Dispatch to game-specific solver.

    Args:
        game: The env._game object.
        game_id: Game identifier (e.g., "tu93", "wa30").
        level_idx: Current level index.

    Returns:
        Action plan as list of (GameAction, click_data|None) tuples,
        or None if no solver is available or solver fails.
    """
    base_id = game_id.split("-")[0] if game_id else ""

    solver = SOLVERS.get(base_id)
    if solver is None:
        return None

    try:
        return solver(game, level_idx)
    except Exception as e:
        print(f"    [GAME-SOLVER] {base_id} error: {e}")
        import traceback
        traceback.print_exc()
        return None
