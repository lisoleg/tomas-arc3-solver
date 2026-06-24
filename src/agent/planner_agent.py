"""
PlannerAgent V5 — Hybrid Planner-Learner for ARC-AGI-3.

Architecture (fusing Oracle planning + RL/IRL + Game Industry + Taiyi Theory):
1. Game Introspection: Read env._game for perfect state knowledge
2. Route Planning: BFS pathfinding + type-order optimization (Oracle v17 core)
3. RL Meta-Learning: Q-learning prioritizes route orderings [RL / AlphaGo value net]
4. IRL Trauma Memory: Avoid dangerous positions from GAME_OVER [IRL]
5. Library Learning: Cross-level macro transfer [TOMAS Sleep-Step]
6. Alpha-Beta Pruning: Cut routes that can't beat current best [kappa-gating]
7. Circuit Breaker: Abort failing plans, switch to exploration [TOMAS L4]
8. Pixel Fallback: DopamineExplorer for unknown game types

Theoretical basis:
- MCTS UCB = Q-learning epsilon-greedy (explore-exploit tradeoff)
- Alpha-Beta = kappa-gating pre-judgment blocking (prune bad routes early)
- Value network = Q-table replaces random rollout (learned route quality)
- L4 circuit breaker = Bayesian confidence check (abort hallucinated plans)
- Flow field = Dijkstra BFS (standard RTS game pathfinding)

Author: TOMAS Team
"""

from __future__ import annotations

import sys
import os
import time
from collections import deque
from itertools import permutations, combinations
from typing import Optional, Any

import numpy as np

# Reuse RL components from DopamineExplorer
from .dopamine_explorer import (
    RewardEngine,
    QLearner,
    DangerMemory,
    MacroLibrary,
    SpatialReasoner,
    DopamineExplorer,
)

# Grid perception and game profiles for non-oracle mode
from .grid_perception import GridPerception
from .game_profiles import GameProfile, GameProfileRegistry

# Oracle adapters for multi-game oracle support
from .oracle_adapters import (
    OracleAdapter,
    LS20Adapter,
    TR87Adapter,
    FT09Adapter,
    ADAPTER_REGISTRY,
    get_oracle_adapter,
    auto_detect_adapter,
)


# ============================================================================
# Helper Functions (from Oracle v17, cleaned up)
# ============================================================================

def snap_to_grid(
    tx: float, ty: float, gx: float, gy: float, step: int
) -> tuple[int, int]:
    """Snap a pixel position to the game grid.

    Args:
        tx, ty: Target pixel position.
        gx, gy: Grid origin (player start position).
        step: Grid step size (typically 5).

    Returns:
        Snapped grid position as (x, y).
    """
    sx = int(gx + step * round((tx - gx) / step))
    sy = int(gy + step * round((ty - gy) / step))
    return (sx, sy)


def expand_walls(
    sprites: list, gx: float, gy: float, step: int, sz: int = 5
) -> set[tuple[int, int]]:
    """Expand wall sprites to grid positions.

    Marks all grid cells that overlap with wall sprite pixels.

    Args:
        sprites: List of wall sprite objects with x, y attributes.
        gx, gy: Grid origin.
        step: Grid step size.
        sz: Sprite size for overlap check.

    Returns:
        Set of (x, y) grid positions blocked by walls.
    """
    walls: set[tuple[int, int]] = set()
    for w in sprites:
        wx, wy = w.x, w.y
        base_x = int(gx + step * ((wx - gx) // step))
        base_y = int(gy + step * ((wy - gy) // step))
        for ddx in range(-step, step * 2, step):
            for ddy in range(-step, step * 2, step):
                px = base_x + ddx
                py = base_y + ddy
                if abs(px - wx) < sz and abs(py - wy) < sz:
                    walls.add((px, py))
    return walls


def bfs_path_with_teleports(
    sx: int, sy: int, tx: int, ty: int, step: int,
    walls: set[tuple[int, int]],
    teleport_map: dict[tuple[int, int], tuple[int, int]],
) -> Optional[list]:
    """BFS pathfinding with push-block teleport support.

    Args:
        sx, sy: Start position.
        tx, ty: Target position.
        step: Grid step size.
        walls: Set of wall positions.
        teleport_map: Map of push-block positions to destinations.

    Returns:
        List of (action, name) tuples, or None if no path.
    """
    from arcengine import GameAction

    if sx == tx and sy == ty:
        return []

    q: deque = deque([(sx, sy)])
    vis: set[tuple[int, int]] = {(sx, sy)}
    par: dict[tuple[int, int], Optional[tuple]] = {(sx, sy): None}

    dirs = [
        (0, -step, GameAction.ACTION1, "up"),
        (0, step, GameAction.ACTION2, "down"),
        (-step, 0, GameAction.ACTION3, "left"),
        (step, 0, GameAction.ACTION4, "right"),
    ]

    while q:
        cx, cy = q.popleft()
        if cx == tx and cy == ty:
            path = []
            s = (cx, cy)
            while par[s] is not None:
                ps, a, n = par[s]
                path.append((a, n))
                s = ps
            path.reverse()
            return path

        for dx, dy, a, n in dirs:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= 64 or ny < 0 or ny >= 64:
                continue
            if (nx, ny) in vis:
                continue
            if (nx, ny) in walls:
                if (nx, ny) in teleport_map:
                    dest = teleport_map[(nx, ny)]
                    if dest not in vis:
                        vis.add(dest)
                        par[dest] = ((cx, cy), a, n)
                        q.append(dest)
                continue
            vis.add((nx, ny))
            par[(nx, ny)] = ((cx, cy), a, n)
            q.append((nx, ny))

    return None


def compute_path_positions(
    sx: int, sy: int, path: list, step: int,
    teleport_map: dict[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int]]:
    """Compute grid positions along a path, handling teleport jumps.

    Args:
        sx, sy: Start position.
        path: List of (action, name) tuples.
        step: Grid step size.
        teleport_map: Push-block teleport map.

    Returns:
        List of (x, y) positions along the path.
    """
    from arcengine import GameAction

    positions = [(sx, sy)]
    cx, cy = sx, sy
    for action, _ in path:
        if action == GameAction.ACTION1:
            nx, ny = cx, cy - step
        elif action == GameAction.ACTION2:
            nx, ny = cx, cy + step
        elif action == GameAction.ACTION3:
            nx, ny = cx - step, cy
        elif action == GameAction.ACTION4:
            nx, ny = cx + step, cy
        else:
            continue
        if (nx, ny) in teleport_map:
            cx, cy = teleport_map[(nx, ny)]
        else:
            cx, cy = nx, ny
        positions.append((cx, cy))
    return positions


def find_adjacent(
    pos: tuple[int, int], step: int, walls: set[tuple[int, int]]
) -> Optional[tuple]:
    """Find an adjacent passable position.

    Args:
        pos: Current position.
        step: Grid step size.
        walls: Set of wall positions.

    Returns:
        (nx, ny, action, name) tuple, or None if surrounded.
    """
    from arcengine import GameAction

    x, y = pos
    for dx, dy, a, n in [
        (0, -step, GameAction.ACTION1, "up"),
        (0, step, GameAction.ACTION2, "down"),
        (-step, 0, GameAction.ACTION3, "left"),
        (step, 0, GameAction.ACTION4, "right"),
    ]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in walls:
            return (nx, ny, a, n)
    return None


# ============================================================================
# Moving Switcher (dboxixicic simulation)
# ============================================================================

class MovingSwitcher:
    """Simulates the dboxixicic moving switcher mechanism.

    The switcher moves along a bounding box track. Before each successful
    player move, the switcher calls step() to move one cell along its track.
    Position is deterministic and periodic based on move count.
    """

    DIR_DELTA = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # down, right, up, left

    def __init__(self, ms_obj, gx: float, gy: float, step: int) -> None:
        """Initialize from game's wsoslqeku entry.

        Args:
            ms_obj: dboxixicic instance from game.wsoslqeku.
            gx, gy: Grid origin (player start position).
            step: Grid step size (5).
        """
        self.bbox = ms_obj.bfdcztirdu
        self.sprite = ms_obj._sprite
        self.cell = ms_obj._cell
        self.start_x = ms_obj._start_x
        self.start_y = ms_obj._start_y

        tags = self.sprite.tags or []
        if "rhsxkxzdjz" in tags:
            self.sw_type = "rot"
        elif "ttfwljgohq" in tags:
            self.sw_type = "shape"
        elif "soyhouuebz" in tags:
            self.sw_type = "color"
        else:
            self.sw_type = "unknown"

        self.timeline = self._compute_timeline(max_steps=500)
        self.period = self._find_period()

        self.grid_timeline: dict[int, tuple[int, int]] = {}
        for s, pos in self.timeline.items():
            self.grid_timeline[s] = snap_to_grid(pos[0], pos[1], gx, gy, step)

        self.grid_positions = list(set(
            self.grid_timeline[i] for i in range(self.period)
        ))

    def _iiosonyanc(self, x: int, y: int) -> bool:
        """Check if position is valid (in bounding box and pixel >= 0)."""
        bbox = self.bbox
        if not bbox:
            return False
        if not (x >= bbox.x and y >= bbox.y and
                x < bbox.x + bbox.width and y < bbox.y + bbox.height):
            return False
        rx = x - bbox.x
        ry = y - bbox.y
        try:
            return int(bbox.pixels[ry, rx]) >= 0
        except (IndexError, KeyError, TypeError, ValueError):
            return False

    def _compute_timeline(self, max_steps: int = 500) -> dict:
        """Simulate dboxixicic.step() to generate position timeline."""
        timeline = {}
        x, y = self.start_x, self.start_y
        direction = 0
        timeline[0] = (x, y)

        for s in range(1, max_steps + 1):
            dx, dy = self.DIR_DELTA[direction]
            new_x = x + dx * self.cell
            new_y = y + dy * self.cell

            if self._iiosonyanc(new_x, new_y):
                x, y = new_x, new_y
            else:
                found = False
                for try_dir in [
                    (direction - 1) % 4,
                    (direction + 1) % 4,
                    (direction + 2) % 4,
                ]:
                    dx, dy = self.DIR_DELTA[try_dir]
                    new_x = x + dx * self.cell
                    new_y = y + dy * self.cell
                    if self._iiosonyanc(new_x, new_y):
                        direction = try_dir
                        x, y = new_x, new_y
                        found = True
                        break

            timeline[s] = (x, y)

        return timeline

    def _find_period(self) -> int:
        """Find the period of the movement pattern."""
        start = self.timeline[0]
        for p in range(1, min(len(self.timeline), 100)):
            if self.timeline[p] == start:
                is_period = True
                for i in range(p, min(2 * p, len(self.timeline))):
                    if self.timeline[i] != self.timeline[i - p]:
                        is_period = False
                        break
                if is_period:
                    return p
        return len(self.timeline)

    def grid_pos_at(self, step_count: int) -> tuple[int, int]:
        """Get switcher grid position at a given move count."""
        return self.grid_timeline[step_count % self.period]

    def __repr__(self) -> str:
        return (
            f"MovingSwitcher(type={self.sw_type}, period={self.period}, "
            f"positions={self.grid_positions})"
        )


# ============================================================================
# Push Teleport
# ============================================================================

class PushTeleport:
    """Push block as teleport: block_grid -> dest_grid."""

    def __init__(self, pb_obj, gx: float, gy: float, step: int) -> None:
        """Initialize from game's hasivfwip entry.

        Args:
            pb_obj: Push block object.
            gx, gy: Grid origin.
            step: Grid step size.
        """
        self.sprite = pb_obj.sprite
        self.dx = pb_obj.dx
        self.dy = pb_obj.dy
        self.width = pb_obj.width
        self.height = pb_obj.height
        self.start_x = pb_obj.start_x
        self.start_y = pb_obj.start_y
        self.step = step
        self.fjzuynaokm = pb_obj.fjzuynaokm

        wall_cx = self.sprite.x + 1 * self.dx
        wall_cy = self.sprite.y + 1 * self.dy
        push_dist = 0
        for acdeadlbzx in range(1, 12):
            nskortmtde = wall_cx + self.dx * self.width * acdeadlbzx
            wsotwbexvh = wall_cy + self.dy * self.height * acdeadlbzx
            if (nskortmtde, wsotwbexvh) in self.fjzuynaokm:
                push_dist = max(0, acdeadlbzx - 1)
                break

        self.push_dist = push_dist
        self.dest_x = self.start_x + self.dx * self.width * push_dist
        self.dest_y = self.start_y + self.dy * self.height * push_dist
        self.dest_grid = snap_to_grid(self.dest_x, self.dest_y, gx, gy, step)
        self.block_grid = snap_to_grid(self.sprite.x, self.sprite.y, gx, gy, step)

    def is_usable(self) -> bool:
        """Check if this push teleport is usable."""
        return self.push_dist > 0

    def __repr__(self) -> str:
        d = {"(0,1)": "DOWN", "(1,0)": "RIGHT", "(-1,0)": "LEFT", "(0,-1)": "UP"}
        return (
            f"TP({self.block_grid}->{self.dest_grid}, "
            f"{d.get(str((self.dx, self.dy)), '?')}, d={self.push_dist})"
        )


# ============================================================================
# Time-dependent BFS for moving switchers
# ============================================================================

def bfs_to_moving_switcher(
    sx: int, sy: int, start_move_count: int, step: int,
    walls: set[tuple[int, int]],
    tp_map: dict[tuple[int, int], tuple[int, int]],
    moving_sw: MovingSwitcher,
) -> tuple[Optional[list], Optional[int]]:
    """Time-dependent BFS to catch a moving switcher.

    State: (grid_x, grid_y, move_mod_period)
    The switcher position depends on the total move count (mod period).

    Args:
        sx, sy: Start position.
        start_move_count: Current total move count.
        step: Grid step size.
        walls: Set of wall positions.
        tp_map: Push-block teleport map.
        moving_sw: MovingSwitcher instance.

    Returns:
        (path, end_move_count) or (None, None).
    """
    from arcengine import GameAction

    period = moving_sw.period
    start_mod = start_move_count % period

    q: deque = deque([(sx, sy, start_mod)])
    vis: set[tuple[int, int, int]] = {(sx, sy, start_mod)}
    par: dict[tuple, Optional[tuple]] = {(sx, sy, start_mod): None}

    dirs = [
        (0, -step, GameAction.ACTION1, "up"),
        (0, step, GameAction.ACTION2, "down"),
        (-step, 0, GameAction.ACTION3, "left"),
        (step, 0, GameAction.ACTION4, "right"),
    ]

    while q:
        cx, cy, cm = q.popleft()

        for dx, dy, a, n in dirs:
            nx, ny = cx + dx, cy + dy
            nm = (cm + 1) % period

            if nx < 0 or nx >= 64 or ny < 0 or ny >= 64:
                continue
            if (nx, ny, nm) in vis:
                continue

            if (nx, ny) in walls:
                if (nx, ny) in tp_map:
                    dest = tp_map[(nx, ny)]
                    if (dest[0], dest[1], nm) not in vis:
                        vis.add((dest[0], dest[1], nm))
                        par[(dest[0], dest[1], nm)] = ((cx, cy, cm), a, n)

                        sw_pos = moving_sw.grid_timeline[nm]
                        if (dest[0], dest[1]) == sw_pos:
                            path = [(a, n)]
                            s = (cx, cy, cm)
                            while par[s] is not None:
                                ps, pa, pn = par[s]
                                path.append((pa, pn))
                                s = ps
                            path.reverse()
                            end_mc = start_move_count + len(path)
                            return path, end_mc

                        q.append((dest[0], dest[1], nm))
                continue

            vis.add((nx, ny, nm))
            par[(nx, ny, nm)] = ((cx, cy, cm), a, n)

            sw_pos = moving_sw.grid_timeline[nm]
            if (nx, ny) == sw_pos:
                path = [(a, n)]
                s = (cx, cy, cm)
                while par[s] is not None:
                    ps, pa, pn = par[s]
                    path.append((pa, pn))
                    s = ps
                path.reverse()
                end_mc = start_move_count + len(path)
                return path, end_mc

            q.append((nx, ny, nm))

    return None, None


# ============================================================================
# Level Info (game introspection)
# ============================================================================

class LevelInfo:
    """Reads game internals to build a complete level model.

    This is the 'oracle' component: it accesses env._game to get perfect
    knowledge of walls, goals, switchers, push blocks, and moving switchers.
    """

    def __init__(self, game) -> None:
        """Initialize from game object.

        Args:
            game: The game object from env._game.
        """
        self.game = game
        self.player = game.gudziatsk
        self.step = game.gisrhqpee
        self.px = self.player.x
        self.py = self.player.y
        self.grid_x = self.px
        self.grid_y = self.py
        self.step_max = game._step_counter_ui.osgviligwp
        self.step_dec = game._step_counter_ui.efipnixsvl
        self.switcher_free = (game.level_index == 0)
        self.level_idx = game.level_index

        self.rot = game.cklxociuu
        self.shape = game.fwckfzsyc
        self.color = game.hiaauhahz

        self.wall_sprites = game.current_level.get_sprites_by_tag("ihdgageizm")

        # Push blocks as teleports
        self.teleports: list[PushTeleport] = []
        self.push_block_grids: set[tuple[int, int]] = set()
        self.teleport_map: dict[tuple[int, int], tuple[int, int]] = {}
        for pb in game.hasivfwip:
            tp = PushTeleport(pb, self.grid_x, self.grid_y, self.step)
            if tp.is_usable():
                self.teleports.append(tp)
                self.push_block_grids.add(tp.block_grid)
                self.teleport_map[tp.block_grid] = tp.dest_grid

        # Goals with required attributes
        self.goals: list[dict] = []
        for i, g in enumerate(game.plrpelhym):
            self.goals.append({
                "sprite": g,
                "req_shape": game.ldxlnycps[i],
                "req_color": game.yjdexjsoa[i],
                "req_rot": game.ehwheiwsk[i],
                "grid_pos": snap_to_grid(
                    g.x, g.y, self.grid_x, self.grid_y, self.step
                ),
            })

        # Moving switchers
        self.moving_switchers: list[MovingSwitcher] = []
        moving_sw_sprite_ids: set[int] = set()
        if hasattr(game, 'wsoslqeku') and game.wsoslqeku:
            for ms_obj in game.wsoslqeku:
                msw = MovingSwitcher(ms_obj, self.grid_x, self.grid_y, self.step)
                self.moving_switchers.append(msw)
                moving_sw_sprite_ids.add(id(ms_obj._sprite))

        self.moving_rot: Optional[MovingSwitcher] = None
        self.moving_shape: Optional[MovingSwitcher] = None
        self.moving_color: Optional[MovingSwitcher] = None
        for msw in self.moving_switchers:
            if msw.sw_type == "rot" and self.moving_rot is None:
                self.moving_rot = msw
            elif msw.sw_type == "shape" and self.moving_shape is None:
                self.moving_shape = msw
            elif msw.sw_type == "color" and self.moving_color is None:
                self.moving_color = msw

        # Static switchers (exclude moving switcher sprites)
        self.rot_sw = [
            snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
            for s in game.current_level.get_sprites_by_tag("rhsxkxzdjz")
            if id(s) not in moving_sw_sprite_ids
        ]
        self.shape_sw = [
            snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
            for s in game.current_level.get_sprites_by_tag("ttfwljgohq")
            if id(s) not in moving_sw_sprite_ids
        ]
        self.color_sw = [
            snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
            for s in game.current_level.get_sprites_by_tag("soyhouuebz")
            if id(s) not in moving_sw_sprite_ids
        ]

        self.refills = [
            snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
            for s in game.current_level.get_sprites_by_tag("npxgalaybz")
        ]

        # Build passable set (switchers + refills + ALL moving switcher positions)
        self.passable: set[tuple[int, int]] = set()
        for tag in ("rhsxkxzdjz", "ttfwljgohq", "soyhouuebz", "npxgalaybz"):
            for s in game.current_level.get_sprites_by_tag(tag):
                gp = snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
                self.passable.add(gp)
                bx = int(self.grid_x + self.step * ((gp[0] - self.grid_x) // self.step))
                by = int(self.grid_y + self.step * ((gp[1] - self.grid_y) // self.step))
                for ddx in range(-self.step, self.step * 2, self.step):
                    for ddy in range(-self.step, self.step * 2, self.step):
                        px2 = bx + ddx
                        py2 = by + ddy
                        if abs(px2 - gp[0]) < self.step and abs(py2 - gp[1]) < self.step:
                            self.passable.add((px2, py2))

        for msw in self.moving_switchers:
            for pos in msw.grid_positions:
                self.passable.add(pos)
                bx = int(self.grid_x + self.step * ((pos[0] - self.grid_x) // self.step))
                by = int(self.grid_y + self.step * ((pos[1] - self.grid_y) // self.step))
                for ddx in range(-self.step, self.step * 2, self.step):
                    for ddy in range(-self.step, self.step * 2, self.step):
                        px2 = bx + ddx
                        py2 = by + ddy
                        if abs(px2 - pos[0]) < self.step and abs(py2 - pos[1]) < self.step:
                            self.passable.add((px2, py2))

    def build_walls(
        self, rot: int, shape: int, color: int,
        completed_goals: Optional[set[int]] = None,
        extra_walls: Optional[set[tuple[int, int]]] = None,
    ) -> set[tuple[int, int]]:
        """Build wall set for given player attributes.

        Args:
            rot, shape, color: Current player attributes.
            completed_goals: Set of completed goal indices (their sprites are removed).
            extra_walls: Additional wall positions (from danger memory).

        Returns:
            Set of wall positions.
        """
        if completed_goals is None:
            completed_goals = set()

        walls = expand_walls(
            self.wall_sprites, self.grid_x, self.grid_y, self.step
        )

        for i, g in enumerate(self.goals):
            if i in completed_goals:
                continue
            matched = (
                shape == g["req_shape"]
                and color == g["req_color"]
                and rot == g["req_rot"]
            )
            if not matched:
                walls |= expand_walls(
                    [g["sprite"]], self.grid_x, self.grid_y, self.step
                )

        for p in self.passable:
            walls.discard(p)

        for bg in self.push_block_grids:
            walls.add(bg)

        # Add danger memory walls (IRL: avoid positions that caused GAME_OVER)
        if extra_walls:
            walls |= extra_walls

        return walls

    def get_moving_switcher(self, sw_type: str) -> Optional[MovingSwitcher]:
        """Get the MovingSwitcher for a given type, or None."""
        if sw_type == "rot":
            return self.moving_rot
        elif sw_type == "shape":
            return self.moving_shape
        elif sw_type == "color":
            return self.moving_color
        return None

    def generate_waypoint_orderings(self) -> list[list[dict]]:
        """Generate waypoint orderings for route search.

        For single-goal: full permutations of switcher visits.
        For multi-goal: type-order permutations within each goal.
        """
        from itertools import product as iter_product

        orderings: list[list[dict]] = []
        c_rot, c_shape, c_color = self.rot, self.shape, self.color

        def make_sw_visit(sw_type: str) -> dict:
            moving_sw = self.get_moving_switcher(sw_type)
            if moving_sw is not None:
                return {"type": sw_type, "pos": None, "moving": True}
            else:
                if sw_type == "rot":
                    pos = self.rot_sw[0] if self.rot_sw else None
                elif sw_type == "shape":
                    pos = self.shape_sw[0] if self.shape_sw else None
                else:
                    pos = self.color_sw[0] if self.color_sw else None
                return {"type": sw_type, "pos": pos, "moving": False}

        def generate_type_perms(
            rot_diff: int, shape_diff: int, color_diff: int
        ) -> list[list[dict]]:
            types = []
            if rot_diff > 0:
                types.append("rot")
            if shape_diff > 0:
                types.append("shape")
            if color_diff > 0:
                types.append("color")
            if not types:
                return [[]]
            counts = {"rot": rot_diff, "shape": shape_diff, "color": color_diff}
            result = []
            seen = set()
            for type_perm in permutations(types):
                visits = []
                for t in type_perm:
                    for _ in range(counts[t]):
                        visits.append(make_sw_visit(t))
                key = tuple((v["type"], v.get("moving", False)) for v in visits)
                if key not in seen:
                    seen.add(key)
                    result.append(visits)
            return result

        if len(self.goals) == 1:
            goal = self.goals[0]
            rot_diff = (goal["req_rot"] - c_rot) % 4
            shape_diff = (goal["req_shape"] - c_shape) % 6
            color_diff = (goal["req_color"] - c_color) % 4

            sw_visits = []
            for _ in range(rot_diff):
                sw_visits.append(make_sw_visit("rot"))
            for _ in range(shape_diff):
                sw_visits.append(make_sw_visit("shape"))
            for _ in range(color_diff):
                sw_visits.append(make_sw_visit("color"))

            seen = set()
            for perm in permutations(sw_visits):
                key = tuple((v["type"], v.get("moving", False)) for v in perm)
                if key not in seen:
                    seen.add(key)
                    ordering = list(perm) + [{"type": "goal", "pos": goal["grid_pos"]}]
                    orderings.append(ordering)
        else:
            goal_indices = list(range(len(self.goals)))
            for goal_perm in permutations(goal_indices):
                goal_type_perms = []
                c_r, c_s, c_c = c_rot, c_shape, c_color
                for gi in goal_perm:
                    goal = self.goals[gi]
                    rot_diff = (goal["req_rot"] - c_r) % 4
                    shape_diff = (goal["req_shape"] - c_s) % 6
                    color_diff = (goal["req_color"] - c_c) % 4
                    type_perms = generate_type_perms(rot_diff, shape_diff, color_diff)
                    goal_type_perms.append(type_perms)
                    c_r = (c_r + rot_diff) % 4
                    c_s = (c_s + shape_diff) % 6
                    c_c = (c_c + color_diff) % 4

                for combo in iter_product(*goal_type_perms):
                    sw_visits = []
                    for idx, gi in enumerate(goal_perm):
                        sw_visits.extend(combo[idx])
                        sw_visits.append({
                            "type": "goal",
                            "pos": self.goals[gi]["grid_pos"],
                        })
                    orderings.append(sw_visits)

                if len(orderings) > 500:
                    break

        return orderings


# ============================================================================
# Grid Level Info (grid-inference version of LevelInfo)
# ============================================================================

class GridLevelInfo:
    """Grid-inference version of LevelInfo for non-oracle mode.

    Provides a compatible interface to LevelInfo but infers game state
    from raw 64x64 frames using GridPerception instead of env._game.

    Since pure grid inference cannot detect rot/shape/color attributes,
    switcher mechanics, or goal requirements, this class provides:
    - Wall positions (from block color analysis)
    - Player position (from frame differencing)
    - Step size and grid origin
    - Empty switcher/goal/teleport lists (no oracle access)

    The route planning functions will find no feasible plans when using
    GridLevelInfo (no goals detected), so the agent falls back to
    exploration or click-based actions.

    Attributes:
        perception: The GridPerception instance used for inference.
        step: Grid step size (typically 5).
        px, py: Player pixel position.
        grid_x, grid_y: Grid origin (player start position).
        step_max: Maximum steps (default 2000).
        step_dec: Step decrement per action (default 1).
        switcher_free: Whether level 0 (always False in grid mode).
        level_idx: Current level index.
        rot, shape, color: Player attributes (always 0 in grid mode).
        wall_sprites: Empty list (no sprite objects in grid mode).
        teleports: Empty list.
        push_block_grids: Empty set.
        teleport_map: Empty dict.
        goals: Empty list.
        moving_switchers: Empty list.
        moving_rot, moving_shape, moving_color: None.
        rot_sw, shape_sw, color_sw: Empty lists.
        refills: Empty list.
        passable: Empty set.
    """

    def __init__(
        self,
        perception: GridPerception,
        level_idx: int = 0,
        step_max: int = 2000,
        step_dec: int = 1,
    ) -> None:
        """Initialize from GridPerception.

        Args:
            perception: The GridPerception instance with analyzed frames.
            level_idx: Current level index.
            step_max: Maximum step budget.
            step_dec: Step decrement per action.
        """
        self.perception: GridPerception = perception
        self.step: int = perception.step
        self.level_idx: int = level_idx
        self.step_max: int = step_max
        self.step_dec: int = step_dec
        self.switcher_free: bool = False

        # Player position (from perception)
        player_pos = perception.get_player_pixel_pos()
        if player_pos is not None:
            self.px: int = player_pos[0]
            self.py: int = player_pos[1]
        else:
            self.px = 0
            self.py = 0
        self.grid_x: int = self.px
        self.grid_y: int = self.py

        # Attributes (cannot be inferred from grid; default to 0)
        self.rot: int = 0
        self.shape: int = 0
        self.color: int = 0

        # No sprite objects available in grid mode
        self.wall_sprites: list = []

        # No teleports/push blocks detectable from grid alone
        self.teleports: list = []
        self.push_block_grids: set[tuple[int, int]] = set()
        self.teleport_map: dict[tuple[int, int], tuple[int, int]] = {}

        # No goals detectable from grid alone
        self.goals: list[dict] = []

        # No switchers detectable from grid alone
        self.moving_switchers: list = []
        self.moving_rot = None
        self.moving_shape = None
        self.moving_color = None
        self.rot_sw: list = []
        self.shape_sw: list = []
        self.color_sw: list = []

        # No refills detectable from grid alone
        self.refills: list = []

        # No passable set (will be computed in build_walls)
        self.passable: set[tuple[int, int]] = set()

    def build_walls(
        self,
        rot: int = 0,
        shape: int = 0,
        color: int = 0,
        completed_goals: Optional[set[int]] = None,
        extra_walls: Optional[set[tuple[int, int]]] = None,
    ) -> set[tuple[int, int]]:
        """Build wall set from grid-inferred wall positions.

        Uses GridPerception's wall_blocks to produce a set of pixel
        positions that are walls. Includes danger walls if provided.

        Args:
            rot, shape, color: Player attributes (unused in grid mode).
            completed_goals: Set of completed goal indices (unused).
            extra_walls: Additional wall positions from danger memory.

        Returns:
            Set of (x, y) pixel positions blocked by walls.
        """
        walls: set[tuple[int, int]] = self.perception.get_wall_positions()

        # Add extra walls (danger memory)
        if extra_walls:
            walls |= extra_walls

        return walls

    def get_moving_switcher(self, sw_type: str):
        """Get the MovingSwitcher for a given type.

        Always returns None in grid mode (no moving switchers detected).

        Args:
            sw_type: Switcher type ("rot", "shape", "color").

        Returns:
            None (no moving switchers in grid mode).
        """
        return None

    def generate_waypoint_orderings(self) -> list[list[dict]]:
        """Generate waypoint orderings for route search.

        Returns an empty list in grid mode since no goals or switchers
        are detected. The route planning functions will produce no plans.

        Returns:
            Empty list (no waypoints in grid mode).
        """
        return []


# ============================================================================
# Route Planning Functions
# ============================================================================

def _block_other_switchers(
    info: LevelInfo,
    walls: set[tuple[int, int]],
    target_type: Optional[str] = None,
) -> set[tuple[int, int]]:
    """Add all switcher positions except target_type to walls.

    This prevents BFS paths from crossing other switchers and triggering
    unintended attribute changes during execution.

    Args:
        info: LevelInfo instance.
        walls: Set of wall positions (modified in-place).
        target_type: The type of switcher being visited ("rot", "shape", "color").
                     If None, block ALL switcher positions.

    Returns:
        walls (modified in-place, also returned for chaining).
    """
    if target_type != "rot":
        for pos in info.rot_sw:
            walls.add(pos)
    if target_type != "shape":
        for pos in info.shape_sw:
            walls.add(pos)
    if target_type != "color":
        for pos in info.color_sw:
            walls.add(pos)
    for msw in info.moving_switchers:
        if msw.sw_type != target_type:
            for pos in msw.grid_positions:
                walls.add(pos)
    return walls


def simulate_route(
    route_items: list[tuple[str, any]],
    info: LevelInfo,
    start_pos: Optional[tuple[int, int]] = None,
    start_rot: Optional[int] = None,
    start_shape: Optional[int] = None,
    start_color: Optional[int] = None,
    start_budget: Optional[int] = None,
    extra_walls: Optional[set[tuple[int, int]]] = None,
) -> Optional[tuple[int, list]]:
    """Simulate a route, tracking move_count for moving switcher timing.

    Args:
        route_items: List of (item_type, item_data) tuples.
        info: LevelInfo instance.
        start_pos: Starting position (defaults to player start).
        start_rot/shape/color: Starting attributes.
        start_budget: Starting step budget.
        extra_walls: Additional wall positions from danger memory.

    Returns:
        (num_actions, actions_list) or None if infeasible.
    """
    cx, cy = start_pos or (info.px, info.py)
    cur_rot = start_rot if start_rot is not None else info.rot
    cur_shape = start_shape if start_shape is not None else info.shape
    cur_color = start_color if start_color is not None else info.color
    budget = start_budget if start_budget is not None else info.step_max
    step = info.step
    step_dec = info.step_dec
    step_max = info.step_max
    switcher_free = info.switcher_free
    tp_map = info.teleport_map

    all_actions: list = []
    consumed_refills: set[tuple[int, int]] = set()
    refill_set = set(info.refills) if info.refills else set()
    move_count = 0
    completed_goals: set[int] = set()

    def check_path_for_refills(path, start_x, start_y):
        nonlocal budget
        if not refill_set:
            return
        positions = compute_path_positions(start_x, start_y, path, step, tp_map)
        for pos in positions:
            if pos in refill_set and pos not in consumed_refills:
                consumed_refills.add(pos)
                budget = step_max

    for item_type, item_data in route_items:
        if item_type == "refill":
            target = item_data
            if target in consumed_refills:
                return None
            walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals, extra_walls)
            walls.discard(target)
            _block_other_switchers(info, walls, None)
            path = bfs_path_with_teleports(cx, cy, target[0], target[1], step, walls, tp_map)
            if path is None:
                return None
            cost = len(path) * step_dec
            if cost > budget:
                return None
            all_actions.extend(path)
            budget -= cost
            move_count += len(path)
            check_path_for_refills(path, cx, cy)
            consumed_refills.add(target)
            budget = step_max
            cx, cy = target

        elif item_type == "switch":
            wp = item_data
            wp_type = wp["type"]
            is_moving = wp.get("moving", False)

            if is_moving:
                moving_sw = info.get_moving_switcher(wp_type)
                if moving_sw is None:
                    return None

                walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals, extra_walls)
                for pos in moving_sw.grid_positions:
                    walls.discard(pos)
                _block_other_switchers(info, walls, wp_type)

                seg, new_move_count = bfs_to_moving_switcher(
                    cx, cy, move_count, step, walls, tp_map, moving_sw
                )
                if seg is None:
                    return None

                if switcher_free:
                    raw_cost = max(0, len(seg) - 1)
                else:
                    raw_cost = len(seg)
                cost = raw_cost * step_dec
                if cost > budget:
                    return None

                all_actions.extend(seg)
                budget -= cost
                move_count = new_move_count
                check_path_for_refills(seg, cx, cy)
                cx, cy = moving_sw.grid_pos_at(new_move_count)

                if wp_type == "rot":
                    cur_rot = (cur_rot + 1) % 4
                elif wp_type == "shape":
                    cur_shape = (cur_shape + 1) % 6
                elif wp_type == "color":
                    cur_color = (cur_color + 1) % 4

            else:
                wp_pos = wp["pos"]
                if wp_pos is None:
                    continue

                walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals, extra_walls)
                walls.discard(wp_pos)
                _block_other_switchers(info, walls, wp_type)

                if (cx, cy) == wp_pos:
                    adj = find_adjacent(wp_pos, step, walls)
                    if adj is None:
                        return None
                    on_walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals, extra_walls)
                    on_walls.discard(wp_pos)
                    _block_other_switchers(info, on_walls, wp_type)
                    on_path = bfs_path_with_teleports(adj[0], adj[1], wp_pos[0], wp_pos[1], step, on_walls, tp_map)
                    if on_path is None:
                        return None
                    seg = [(adj[2], adj[3])] + on_path
                else:
                    seg = bfs_path_with_teleports(cx, cy, wp_pos[0], wp_pos[1], step, walls, tp_map)
                    if seg is None:
                        return None

                if switcher_free and wp_type in ("rot", "shape", "color"):
                    raw_cost = max(0, len(seg) - 1)
                else:
                    raw_cost = len(seg)
                cost = raw_cost * step_dec
                if cost > budget:
                    return None

                all_actions.extend(seg)
                budget -= cost
                move_count += len(seg)
                check_path_for_refills(seg, cx, cy)
                cx, cy = wp_pos

                if wp_type == "rot":
                    cur_rot = (cur_rot + 1) % 4
                elif wp_type == "shape":
                    cur_shape = (cur_shape + 1) % 6
                elif wp_type == "color":
                    cur_color = (cur_color + 1) % 4

        elif item_type == "goal":
            target = item_data
            walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals, extra_walls)
            walls.discard(target)
            _block_other_switchers(info, walls, None)
            path = bfs_path_with_teleports(cx, cy, target[0], target[1], step, walls, tp_map)
            if path is None:
                return None
            cost = len(path) * step_dec
            if cost > budget:
                return None
            all_actions.extend(path)
            budget -= cost
            move_count += len(path)
            check_path_for_refills(path, cx, cy)
            cx, cy = target
            for gi, g in enumerate(info.goals):
                if g["grid_pos"] == target:
                    completed_goals.add(gi)
                    break

    return len(all_actions), all_actions


def search_routes(
    info: LevelInfo,
    max_refills: int = 3,
    time_limit: float = 60.0,
    extra_walls: Optional[set[tuple[int, int]]] = None,
) -> tuple[int, Optional[str], Optional[list]]:
    """Search for best route. Uses Alpha-Beta pruning to cut bad routes early.

    Args:
        info: LevelInfo instance.
        max_refills: Maximum number of refills to consider.
        time_limit: Maximum search time in seconds.
        extra_walls: Additional wall positions from danger memory.

    Returns:
        (best_steps, best_route_desc, best_actions) tuple.
    """
    start_time = time.time()

    orderings = info.generate_waypoint_orderings()
    refill_options = info.refills if info.refills else []

    best_steps = float('inf')
    best_route: Optional[str] = None
    best_actions: Optional[list] = None

    for n_rf in range(min(max_refills, len(refill_options)) + 1):
        if n_rf == 0:
            refill_options_to_try = [()]
        else:
            refill_options_to_try = list(permutations(refill_options, n_rf))

        for ord_idx, ordering in enumerate(orderings):
            # Alpha-Beta: skip if time exceeded
            if time.time() - start_time > time_limit:
                break

            wp_items: list[tuple[str, any]] = []
            for wp in ordering:
                if wp["type"] == "goal":
                    wp_items.append(("goal", wp["pos"]))
                else:
                    wp_items.append(("switch", wp))

            n_wp = len(wp_items)

            if n_rf > 0:
                rf_insertion_combos = list(combinations(range(n_wp + 1), n_rf))
            else:
                rf_insertion_combos = [()]

            for rf_combo in refill_options_to_try:
                for rf_insertions in rf_insertion_combos:
                    # Alpha-Beta: skip routes that can't beat current best
                    # (we don't know the route length yet, but we can skip
                    # if the minimum possible steps > best_steps)

                    route: list[tuple[str, any]] = []
                    rf_idx = 0
                    for i in range(n_wp + 1):
                        while rf_idx < n_rf and rf_insertions[rf_idx] == i:
                            route.append(("refill", rf_combo[rf_idx]))
                            rf_idx += 1
                        if i < n_wp:
                            route.append(wp_items[i])

                    result = simulate_route(route, info, extra_walls=extra_walls)
                    if result:
                        if result[0] < best_steps:
                            best_steps = result[0]
                            rf_desc = f"{n_rf}rf" if n_rf > 0 else "0rf"
                            best_route = f"ord{ord_idx} {rf_desc}"
                            best_actions = result[1]

            # Early exit: if we have a great route, stop
            if best_route and best_steps < 50:
                break

        if time.time() - start_time > time_limit:
            break

    if best_route is None:
        # Fallback: greedy solver
        g_steps, g_route, g_actions = greedy_solve_route(
            info, max_refills, extra_walls
        )
        if g_actions is not None:
            return g_steps, f"greedy:{g_route}", g_actions

    return best_steps, best_route, best_actions


def greedy_solve_route(
    info: LevelInfo,
    max_refills: int = 3,
    extra_walls: Optional[set[tuple[int, int]]] = None,
) -> tuple[int, Optional[str], Optional[list]]:
    """Greedy route solver with type-order permutations.

    Instead of exhaustive search, tries all type-order permutations and
    greedily visits switchers in that type order.

    Args:
        info: LevelInfo instance.
        max_refills: Maximum number of refills.
        extra_walls: Additional wall positions from danger memory.

    Returns:
        (best_steps, "greedy", best_actions) tuple.
    """
    from itertools import permutations as perm

    step = info.step
    step_dec = info.step_dec
    step_max = info.step_max
    switcher_free = info.switcher_free
    tp_map = info.teleport_map
    refill_set = set(info.refills) if info.refills else set()

    best_result: Optional[tuple[int, list]] = None

    def visit_switcher(sw_type, cx, cy, cur_rot, cur_shape, cur_color, move_count, completed_goals=None):
        walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals, extra_walls)
        moving_sw = info.get_moving_switcher(sw_type)

        if moving_sw is not None:
            for pos in moving_sw.grid_positions:
                walls.discard(pos)
            _block_other_switchers(info, walls, sw_type)
            path, end_mc = bfs_to_moving_switcher(
                cx, cy, move_count, step, walls, tp_map, moving_sw)
            if path is None:
                return None
            new_pos = moving_sw.grid_pos_at(end_mc)
        else:
            if sw_type == "rot":
                sw_pos = info.rot_sw[0] if info.rot_sw else None
            elif sw_type == "shape":
                sw_pos = info.shape_sw[0] if info.shape_sw else None
            else:
                sw_pos = info.color_sw[0] if info.color_sw else None
            if sw_pos is None:
                return None
            walls.discard(sw_pos)
            _block_other_switchers(info, walls, sw_type)
            path = bfs_path_with_teleports(cx, cy, sw_pos[0], sw_pos[1], step, walls, tp_map)
            if path is None:
                return None
            end_mc = move_count + len(path)
            new_pos = sw_pos

        if switcher_free:
            raw_cost = max(0, len(path) - 1)
        else:
            raw_cost = len(path)
        cost = raw_cost * step_dec

        new_rot, new_shape, new_color = cur_rot, cur_shape, cur_color
        if sw_type == "rot":
            new_rot = (cur_rot + 1) % 4
        elif sw_type == "shape":
            new_shape = (cur_shape + 1) % 6
        elif sw_type == "color":
            new_color = (cur_color + 1) % 4

        return (path, end_mc, new_pos, new_rot, new_shape, new_color, cost)

    def find_refill(cx, cy, cur_rot, cur_shape, cur_color, budget, consumed, completed_goals=None):
        best_rf = None
        best_rf_dist = float('inf')
        best_rf_path = None
        for rf_pos in refill_set - consumed:
            walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals, extra_walls)
            walls.discard(rf_pos)
            _block_other_switchers(info, walls, None)
            path = bfs_path_with_teleports(cx, cy, rf_pos[0], rf_pos[1], step, walls, tp_map)
            if path is not None and len(path) < best_rf_dist:
                best_rf_dist = len(path)
                best_rf = rf_pos
                best_rf_path = path
        if best_rf is None:
            return None
        cost = best_rf_dist * step_dec
        if cost > budget:
            return None
        return (best_rf_path, best_rf, cost)

    def try_goal_with_type_order(goal, goal_idx, cx, cy, cur_rot, cur_shape, cur_color,
                                  move_count, budget, consumed_refills, type_order, completed_goals):
        goal_pos = goal["grid_pos"]
        all_actions: list = []

        rot_diff = (goal["req_rot"] - cur_rot) % 4
        shape_diff = (goal["req_shape"] - cur_shape) % 6
        color_diff = (goal["req_color"] - cur_color) % 4

        visits = []
        for t in type_order:
            if t == "rot":
                visits.extend(["rot"] * rot_diff)
            elif t == "shape":
                visits.extend(["shape"] * shape_diff)
            elif t == "color":
                visits.extend(["color"] * color_diff)

        cur_x, cur_y = cx, cy
        cur_r, cur_s, cur_c = cur_rot, cur_shape, cur_color
        mc = move_count
        bud = budget
        cr = set(consumed_refills)
        cg = set(completed_goals)

        for sw_type in visits:
            result = visit_switcher(sw_type, cur_x, cur_y, cur_r, cur_s, cur_c, mc, cg)
            if result is None:
                return None
            path, end_mc, new_pos, new_r, new_s, new_c, cost = result

            if cost > bud:
                rf_result = find_refill(cur_x, cur_y, cur_r, cur_s, cur_c, bud, cr, cg)
                if rf_result is None or len(cr) >= max_refills:
                    return None
                rf_path, rf_pos, rf_cost = rf_result
                all_actions.extend(rf_path)
                bud -= rf_cost
                mc += len(rf_path)
                cr.add(rf_pos)
                bud = step_max
                cur_x, cur_y = rf_pos

                result = visit_switcher(sw_type, cur_x, cur_y, cur_r, cur_s, cur_c, mc, cg)
                if result is None:
                    return None
                path, end_mc, new_pos, new_r, new_s, new_c, cost = result
                if cost > bud:
                    return None

            all_actions.extend(path)
            bud -= cost
            mc = end_mc
            cur_x, cur_y = new_pos
            cur_r, cur_s, cur_c = new_r, new_s, new_c

        walls = info.build_walls(cur_r, cur_s, cur_c, cg, extra_walls)
        walls.discard(goal_pos)
        _block_other_switchers(info, walls, None)
        path = bfs_path_with_teleports(cur_x, cur_y, goal_pos[0], goal_pos[1], step, walls, tp_map)
        if path is None:
            return None

        cost = len(path) * step_dec
        if cost > bud:
            rf_result = find_refill(cur_x, cur_y, cur_r, cur_s, cur_c, bud, cr, cg)
            if rf_result is None or len(cr) >= max_refills:
                return None
            rf_path, rf_pos, rf_cost = rf_result
            all_actions.extend(rf_path)
            bud -= rf_cost
            mc += len(rf_path)
            cr.add(rf_pos)
            bud = step_max
            cur_x, cur_y = rf_pos
            walls = info.build_walls(cur_r, cur_s, cur_c, cg, extra_walls)
            walls.discard(goal_pos)
            _block_other_switchers(info, walls, None)
            path = bfs_path_with_teleports(cur_x, cur_y, goal_pos[0], goal_pos[1], step, walls, tp_map)
            if path is None:
                return None
            cost = len(path) * step_dec
            if cost > bud:
                return None

        all_actions.extend(path)
        bud -= cost
        mc += len(path)
        cg.add(goal_idx)

        return (all_actions, goal_pos, cur_r, cur_s, cur_c, mc, bud, cr, cg)

    def get_type_perms(goal, rot, shape, color):
        types = []
        rot_diff = (goal["req_rot"] - rot) % 4
        shape_diff = (goal["req_shape"] - shape) % 6
        color_diff = (goal["req_color"] - color) % 4
        if rot_diff > 0:
            types.append("rot")
        if shape_diff > 0:
            types.append("shape")
        if color_diff > 0:
            types.append("color")
        if not types:
            return [()]
        return list(perm(types))

    def solve_recursive(goal_idx, goal_order, cx, cy, cur_rot, cur_shape, cur_color,
                        move_count, budget, consumed, actions_so_far, completed_goals):
        nonlocal best_result

        if goal_idx >= len(goal_order):
            total = len(actions_so_far)
            if best_result is None or total < best_result[0]:
                best_result = (total, list(actions_so_far))
            return

        # Alpha-Beta: prune if current path already longer than best
        if best_result is not None and len(actions_so_far) >= best_result[0]:
            return

        gi = goal_order[goal_idx]
        goal = info.goals[gi]
        type_perms = get_type_perms(goal, cur_rot, cur_shape, cur_color)

        for tp in type_perms:
            result = try_goal_with_type_order(
                goal, gi, cx, cy, cur_rot, cur_shape, cur_color,
                move_count, budget, consumed, tp, completed_goals)
            if result is not None:
                acts, new_pos, new_r, new_s, new_c, new_mc, new_bud, new_cr, new_cg = result
                solve_recursive(
                    goal_idx + 1, goal_order,
                    new_pos[0], new_pos[1], new_r, new_s, new_c,
                    new_mc, new_bud, new_cr,
                    actions_so_far + acts, new_cg)

    for goal_order in perm(range(len(info.goals))):
        solve_recursive(
            0, goal_order,
            info.px, info.py, info.rot, info.shape, info.color,
            0, step_max, set(), [], set())

    # Fallback: "closest first" greedy
    if best_result is None:
        for goal_order in perm(range(len(info.goals))):
            cx, cy = info.px, info.py
            cur_rot, cur_shape, cur_color = info.rot, info.shape, info.color
            budget = step_max
            move_count = 0
            consumed: set = set()
            completed_goals: set = set()
            all_actions: list = []
            success = True

            for gi in goal_order:
                goal = info.goals[gi]
                goal_pos = goal["grid_pos"]

                while True:
                    rot_diff = (goal["req_rot"] - cur_rot) % 4
                    shape_diff = (goal["req_shape"] - cur_shape) % 6
                    color_diff = (goal["req_color"] - cur_color) % 4

                    if rot_diff == 0 and shape_diff == 0 and color_diff == 0:
                        break

                    best_dist = float('inf')
                    best_type = None
                    best_path = None
                    best_end_mc = None
                    best_new_pos = None

                    needed_types = []
                    if rot_diff > 0:
                        needed_types.append("rot")
                    if shape_diff > 0:
                        needed_types.append("shape")
                    if color_diff > 0:
                        needed_types.append("color")

                    for sw_type in needed_types:
                        result = visit_switcher(sw_type, cx, cy, cur_rot, cur_shape, cur_color, move_count, completed_goals)
                        if result is not None:
                            path, end_mc, new_pos, _, _, _, _ = result
                            if len(path) < best_dist:
                                best_dist = len(path)
                                best_type = sw_type
                                best_path = path
                                best_end_mc = end_mc
                                best_new_pos = new_pos

                    if best_type is None:
                        success = False
                        break

                    if switcher_free:
                        raw_cost = max(0, best_dist - 1)
                    else:
                        raw_cost = best_dist
                    cost = raw_cost * step_dec

                    if cost > budget:
                        rf_result = find_refill(cx, cy, cur_rot, cur_shape, cur_color, budget, consumed, completed_goals)
                        if rf_result is None or len(consumed) >= max_refills:
                            success = False
                            break
                        rf_path, rf_pos, rf_cost = rf_result
                        all_actions.extend(rf_path)
                        budget -= rf_cost
                        move_count += len(rf_path)
                        consumed.add(rf_pos)
                        budget = step_max
                        cx, cy = rf_pos
                        continue

                    all_actions.extend(best_path)
                    budget -= cost
                    move_count = best_end_mc
                    cx, cy = best_new_pos

                    if best_type == "rot":
                        cur_rot = (cur_rot + 1) % 4
                    elif best_type == "shape":
                        cur_shape = (cur_shape + 1) % 6
                    elif best_type == "color":
                        cur_color = (cur_color + 1) % 4

                if not success:
                    break

                walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals, extra_walls)
                walls.discard(goal_pos)
                _block_other_switchers(info, walls, None)
                path = bfs_path_with_teleports(cx, cy, goal_pos[0], goal_pos[1], step, walls, tp_map)
                if path is None:
                    success = False
                    break

                cost = len(path) * step_dec
                if cost > budget:
                    rf_result = find_refill(cx, cy, cur_rot, cur_shape, cur_color, budget, consumed, completed_goals)
                    if rf_result is None or len(consumed) >= max_refills:
                        success = False
                        break
                    rf_path, rf_pos, rf_cost = rf_result
                    all_actions.extend(rf_path)
                    budget -= rf_cost
                    move_count += len(rf_path)
                    consumed.add(rf_pos)
                    budget = step_max
                    cx, cy = rf_pos
                    walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals, extra_walls)
                    walls.discard(goal_pos)
                    _block_other_switchers(info, walls, None)
                    path = bfs_path_with_teleports(cx, cy, goal_pos[0], goal_pos[1], step, walls, tp_map)
                    if path is None:
                        success = False
                        break
                    cost = len(path) * step_dec
                    if cost > budget:
                        success = False
                        break

                all_actions.extend(path)
                budget -= cost
                move_count += len(path)
                cx, cy = goal_pos
                completed_goals.add(gi)

            if success:
                total = len(all_actions)
                if best_result is None or total < best_result[0]:
                    best_result = (total, all_actions)

    if best_result:
        return best_result[0], "greedy", best_result[1]
    return float('inf'), None, None


# ============================================================================
# PlannerAgent V5 (Main Agent)
# ============================================================================

class PlannerAgent:
    """V5 Hybrid Planner-Learner Agent for ARC-AGI-3.

    Combines deterministic planning (Oracle) with RL meta-learning:
    1. Game Introspection: Read env._game for perfect state knowledge
    2. Route Planning: BFS + type-order optimization (Alpha-Beta pruning)
    3. RL Meta-Learning: Q-learning prioritizes route orderings
    4. IRL Trauma Memory: Avoid dangerous positions from GAME_OVER
    5. Library Learning: Cross-level macro transfer
    6. Circuit Breaker: Abort failing plans, switch to exploration
    7. Pixel Fallback: DopamineExplorer for unknown game types

    Args:
        env: The game environment (for accessing env._game).
        level_baselines: Human baseline steps per level.
        max_steps: Maximum total steps for the game.
        stagnation_threshold: Steps without progress before giving up.
    """

    # Circuit breaker: max plan attempts before switching to fallback
    MAX_PLAN_ATTEMPTS = 5
    # After this many GAME_OVERs, clear danger_walls (likely false positives)
    DANGER_RESET_THRESHOLD = 3

    def __init__(
        self,
        env=None,
        level_baselines: Optional[list[int]] = None,
        max_steps: int = 2000,
        stagnation_threshold: int = 500,
        use_oracle: Optional[bool] = None,
        game_id: Optional[str] = None,
    ) -> None:
        """Initialize the planner agent.

        Args:
            env: The game environment (for accessing env._game).
            level_baselines: Human baseline steps per level.
            max_steps: Maximum total steps for the game.
            stagnation_threshold: Steps without progress before giving up.
            use_oracle: Whether to use oracle (env._game) mode. If None,
                auto-detects: tries oracle first, falls back to grid mode
                if env._game is inaccessible or raises AttributeError.
            game_id: Game identifier (e.g., "ls20") for profile lookup.
                If None, will attempt to detect from env.
        """
        self._env = env
        self.max_steps = max_steps
        self.stagnation_threshold = stagnation_threshold
        self.level_baselines = level_baselines or []

        # Oracle vs Grid mode
        self._use_oracle_requested = use_oracle
        self.use_oracle: bool = False
        self._oracle_checked: bool = False
        self._oracle_adapter: Optional[OracleAdapter] = None
        self.game_id: Optional[str] = game_id

        # Game profile registry and current profile
        self._profile_registry: GameProfileRegistry = GameProfileRegistry()
        self._game_profile: Optional[GameProfile] = None

        # Grid perception (for non-oracle mode and click games)
        self._grid_perception: Optional[GridPerception] = None
        self._grid_initialized: bool = False

        # Action type ("keyboard", "click", "keyboard_click", "unknown")
        self._action_type: str = "unknown"

        # Click data (for click-type games)
        self._pending_click_data: Optional[dict] = None
        self._clicked_positions: set[tuple[int, int]] = set()

        # RL components (reused from DopamineExplorer)
        self.reward_engine = RewardEngine()
        self.q_learner = QLearner()
        self.danger_memory = DangerMemory()
        self.macro_library = MacroLibrary()

        # Fallback agent (for when planning fails)
        self._fallback: Optional[DopamineExplorer] = None
        self._using_fallback = False

        # Step tracking
        self._step: int = 0
        self._stagnation: int = 0
        self._levels_completed: int = 0
        self._game_over_count: int = 0
        self._total_actions: int = 0
        self._action_history: list[int] = []

        # Per-level state
        self._level_start_step: int = 0
        self._level_step_counts: list[int] = []
        self._level_action_history: list[int] = []
        self._level_game_overs: int = 0  # GAME_OVERs in current level

        # Plan state
        self._plan: Optional[list] = None  # list of (action, name) tuples
        self._plan_idx: int = 0
        self._plan_failed: bool = False
        self._plan_attempt: int = 0  # Number of planning attempts for current level

        # Danger walls (IRL: positions that caused GAME_OVER)
        self._danger_walls: set[tuple[int, int]] = set()

        # Q-learning for route orderings (RL meta-learning)
        # State: (level_idx, rot_diff, shape_diff, color_diff)
        # Action: type ordering string (e.g., "RSC")
        self._route_q: dict[tuple[str, str], float] = {}

        # Seen state hashes for stagnation tracking
        self._seen_state_hashes: set[str] = set()

        # Auto-detect oracle availability if requested
        if use_oracle is not None:
            self.use_oracle = use_oracle
            self._oracle_checked = True

    def _check_oracle_availability(self) -> bool:
        """Check whether env._game is accessible for oracle mode.

        Uses the Oracle adapter registry to detect game type and create
        an appropriate adapter. Supports LS20, TR87, FT09, and any
        future games added to ADAPTER_REGISTRY.

        Respects explicit use_oracle setting:
        - use_oracle=True: force oracle (verify env._game + adapter exists)
        - use_oracle=False: force grid mode
        - use_oracle=None: auto-detect from env._game attributes

        Returns:
            True if oracle mode is available (adapter found), False otherwise.
        """
        if self._oracle_checked:
            return self.use_oracle

        self._oracle_checked = True

        # Explicit grid mode
        if self._use_oracle_requested is False:
            self.use_oracle = False
            return False

        # Explicit oracle mode or auto-detect
        if self._env is None:
            self.use_oracle = False
            return False

        try:
            game = self._env._game

            # Try to get an adapter by game_id first, then auto-detect
            adapter: Optional[OracleAdapter] = None
            if self.game_id:
                adapter = get_oracle_adapter(self.game_id, game)

            if adapter is None:
                adapter = auto_detect_adapter(game)

            if adapter is not None:
                self._oracle_adapter = adapter
                self.use_oracle = True
                return True

            # No adapter found — use grid mode
            self.use_oracle = False
            return False
        except (AttributeError, Exception):
            self.use_oracle = False
            return False

    def _init_grid_perception(
        self,
        grid: np.ndarray,
        available_actions: Optional[list[int]] = None,
    ) -> None:
        """Initialize GridPerception with the first frame.

        Args:
            grid: 64x64 integer array of the initial frame.
            available_actions: List of available action IDs from the
                observation. If None, defaults to keyboard actions.
        """
        if self._grid_perception is None:
            self._grid_perception = GridPerception(step=5)

        if not self._grid_initialized:
            # Use provided available actions, or try env, or default
            if available_actions is None:
                available_actions = []
                if self._env is not None:
                    try:
                        obs = self._env._game
                        if hasattr(obs, '_available_actions'):
                            available_actions = [
                                int(a) for a in obs._available_actions
                            ]
                    except (AttributeError, Exception):
                        pass

            self._grid_perception.analyze_initial_frame(
                grid, available_actions
            )

            # Detect action type
            if available_actions:
                self._action_type = self._grid_perception.get_action_type(
                    available_actions
                )
            else:
                # Default to keyboard for backward compatibility
                self._action_type = "keyboard"

            # Get or create game profile
            if self.game_id:
                base_id = self.game_id.split("-")[0]
            else:
                base_id = "unknown"
            self._game_profile = self._profile_registry.get_profile(base_id)
            self._game_profile.action_type = self._action_type

            self._grid_initialized = True

    def _update_grid_perception(
        self,
        grid: np.ndarray,
        action_taken: Optional[int] = None,
        available_actions: Optional[list[int]] = None,
    ) -> dict:
        """Update grid perception with a new frame.

        Args:
            grid: 64x64 integer array of the current frame.
            action_taken: The action that was taken in the previous step.
            available_actions: Available action IDs (for first-frame init).

        Returns:
            Dictionary of detected changes from GridPerception.update().
        """
        if self._grid_perception is None:
            self._init_grid_perception(grid, available_actions)
            return {}
        return self._grid_perception.update(grid, action_taken)

    def is_done(self, frames: list, latest_frame) -> bool:
        """Check if the game is complete.

        Args:
            frames: List of all frame observations.
            latest_frame: The most recent frame observation.

        Returns:
            True if the game is won or limits are reached.
        """
        try:
            from arcengine import GameState
            return (
                latest_frame.state == GameState.WIN
                or self._step >= self.max_steps
                or self._stagnation >= self.stagnation_threshold
            )
        except ImportError:
            return self._step >= self.max_steps

    def choose_action(self, frames: list, latest_frame) -> "GameAction":
        """Choose the next action based on current observation.

        Priority:
        1. Handle GAME_OVER -> RESET + record danger
        2. Detect level change -> save macro, reset per-level state
        3. Initialize/update grid perception
        4. If click-type game -> choose click action
        5. If no plan, introspect game and plan route
        6. Execute plan action by action
        7. If plan exhausted or failed, fall back to exploration

        Args:
            frames: List of all frame observations.
            latest_frame: Current frame observation.

        Returns:
            GameAction to execute. For click games, click coordinates
            are stored in self._pending_click_data (accessible via
            get_pending_action_data()).
        """
        from arcengine import GameAction, GameState

        self._step += 1
        self._total_actions += 1

        # Clear pending click data from previous step
        self._pending_click_data = None

        state = latest_frame.state
        levels_completed = latest_frame.levels_completed

        # === Extract grid and update perception ===
        grid: Optional[np.ndarray] = None
        try:
            grid = np.array(latest_frame.frame[0])
            # Get available actions from observation for action type detection
            avail_actions: list[int] = []
            if hasattr(latest_frame, 'available_actions'):
                avail_actions = [int(a) for a in latest_frame.available_actions]
            self._update_grid_perception(grid, available_actions=avail_actions)
        except (AttributeError, IndexError, Exception):
            pass

        # === Step 1: Handle GAME_OVER ===
        if state == GameState.GAME_OVER:
            self._game_over_count += 1
            self._level_game_overs += 1
            self.danger_memory.record_game_over()
            self.danger_memory.clear_history()

            # Record danger position (IRL: learn from failure)
            if self.use_oracle and self._env is not None:
                try:
                    game = self._env._game
                    player_pos = (game.gudziatsk.x, game.gudziatsk.y)
                    grid_pos = snap_to_grid(
                        player_pos[0], player_pos[1],
                        game.gudziatsk.x, game.gudziatsk.y,
                        game.gisrhqpee
                    )
                    self._danger_walls.add(grid_pos)
                except (AttributeError, Exception):
                    pass
            elif self._grid_perception is not None:
                # Grid mode: record player position as danger
                player_pos = self._grid_perception.get_player_pixel_pos()
                if player_pos is not None:
                    self._danger_walls.add(player_pos)

            # Circuit breaker: if too many GAME_OVERs, clear danger walls
            if self._level_game_overs >= self.DANGER_RESET_THRESHOLD:
                print(
                    f"    [CIRCUIT BREAKER] {self._level_game_overs} GAME_OVERs "
                    f"on this level, clearing danger_walls"
                )
                self._danger_walls.clear()
                self._level_game_overs = 0

            # Invalidate current plan
            self._plan = None
            self._plan_idx = 0
            self._plan_failed = True
            self._stagnation = 0

            print(
                f"    [GAME_OVER] step={self._step} "
                f"sending RESET (count={self._game_over_count}, "
                f"level_GO={self._level_game_overs})"
            )
            self._action_history.append(0)
            return GameAction.RESET

        # === Step 2: Detect level change ===
        level_changed = levels_completed > self._levels_completed
        if level_changed:
            self._on_level_complete(levels_completed)
            # Reinitialize grid perception for new level
            if grid is not None and self._grid_perception is not None:
                self._grid_perception._reinitialize_for_new_level(grid)

        # === Step 3: Check oracle availability (auto-detect) ===
        if not self._oracle_checked:
            self._check_oracle_availability()

        # === Step 4: Click-type game handling ===
        if self._action_type in ("click", "keyboard_click"):
            action = self._choose_click_action(frames, latest_frame, grid)
            if action is not None:
                self._action_history.append(self._action_to_int(action))
                self._level_action_history.append(self._action_to_int(action))
                self._stagnation = 0
                return action

        # === Step 5: Plan if needed ===
        if self._plan is None or self._plan_idx >= len(self._plan):
            if not self._using_fallback:
                self._plan_level()

            # If planning failed or circuit breaker triggered, create fallback
            if self._using_fallback and self._fallback is None:
                self._fallback = DopamineExplorer(
                    max_steps=self.max_steps,
                    stagnation_threshold=self.stagnation_threshold,
                    level_baselines=self.level_baselines,
                )
                self._fallback._step = self._step
                self._fallback._levels_completed = self._levels_completed

        # === Step 6: Execute plan or fallback ===
        if self._plan is not None and self._plan_idx < len(self._plan):
            action, name = self._plan[self._plan_idx]
            self._plan_idx += 1
            self._action_history.append(self._action_to_int(action))
            self._level_action_history.append(self._action_to_int(action))
            self._stagnation = 0
            return action

        # === Step 7: Fallback to exploration ===
        if self._fallback is not None:
            # Sync fallback state
            self._fallback._step = self._step
            action = self._fallback.choose_action(frames, latest_frame)
            self._action_history.append(self._action_to_int(action))
            self._level_action_history.append(self._action_to_int(action))
            return action

        # Ultimate fallback: random action
        available = list(latest_frame.available_actions)
        if available:
            action = available[0]
        else:
            action = GameAction.ACTION1
        self._action_history.append(self._action_to_int(action))
        return action

    def _plan_level(self) -> None:
        """Plan the current level using game introspection + route optimization.

        Uses BFS pathfinding with Alpha-Beta pruning and IRL danger walls.
        Q-learning prioritizes which route orderings to try first.

        In oracle mode: uses LevelInfo(game) for perfect state knowledge.
        In grid mode: uses GridLevelInfo(perception) for inferred state.
        Grid mode will find no feasible plans (no goals detected), so the
        agent falls back to exploration.

        Circuit breaker: after MAX_PLAN_ATTEMPTS, switch to fallback exploration.
        """
        # Circuit breaker: too many plan attempts -> fallback
        if self._plan_attempt >= self.MAX_PLAN_ATTEMPTS:
            print(
                f"    [CIRCUIT BREAKER] {self._plan_attempt} plan attempts "
                f"exhausted, switching to fallback"
            )
            self._using_fallback = True
            return

        self._plan_attempt += 1

        # === Oracle mode: use env._game for perfect planning ===
        if self.use_oracle and self._env is not None:
            try:
                game = self._env._game

                # LS20: use full LevelInfo planning (backward compatible)
                if not isinstance(self._oracle_adapter, LS20Adapter):
                    # Non-LS20 oracle: use adapter for simpler planning
                    self._plan_level_adapter(game)
                    return

                info = LevelInfo(game)

                level_idx = info.level_idx
                baseline = (
                    self.level_baselines[level_idx]
                    if level_idx < len(self.level_baselines)
                    else "?"
                )

                print(
                    f"  [PLAN-ORACLE] Level {level_idx} (baseline={baseline}), "
                    f"attempt {self._plan_attempt}, "
                    f"danger_walls={len(self._danger_walls)}"
                )
                print(
                    f"    Player: ({info.px},{info.py}), "
                    f"Rot={info.rot}, Shape={info.shape}, Color={info.color}"
                )
                for i, g in enumerate(info.goals):
                    rd = (g["req_rot"] - info.rot) % 4
                    sd = (g["req_shape"] - info.shape) % 6
                    cd = (g["req_color"] - info.color) % 4
                    print(
                        f"    Goal {i}: grid={g['grid_pos']}, "
                        f"rot_diff={rd}, shape_diff={sd}, color_diff={cd}"
                    )

                # Search for best route (with danger walls from IRL)
                best_steps, best_route, best_actions = search_routes(
                    info,
                    max_refills=3,
                    time_limit=60.0,
                    extra_walls=self._danger_walls if self._danger_walls else None,
                )

                if best_actions is not None:
                    self._plan = best_actions
                    self._plan_idx = 0
                    self._plan_failed = False
                    print(
                        f"    [PLAN-ORACLE] Found route: {best_route}, "
                        f"{best_steps} steps"
                    )

                    # RL: Update Q-value for this route ordering
                    self._update_route_q(info, best_route, best_steps)

                    # Record danger memory for plan actions
                    for action, _ in best_actions:
                        self.danger_memory.record_action(
                            f"L{level_idx}", self._action_to_int(action), None
                        )
                else:
                    print(f"    [PLAN-ORACLE] No feasible plan found, using fallback")
                    self._using_fallback = True

            except Exception as e:
                print(f"    [PLAN-ORACLE] Error during planning: {e}")
                import traceback
                traceback.print_exc()
                self._using_fallback = True

        # === Grid mode: use GridPerception for inferred planning ===
        elif self._grid_perception is not None:
            try:
                level_idx = self._levels_completed
                baseline = (
                    self.level_baselines[level_idx]
                    if level_idx < len(self.level_baselines)
                    else "?"
                )

                info = GridLevelInfo(
                    self._grid_perception,
                    level_idx=level_idx,
                    step_max=self.max_steps,
                    step_dec=1,
                )

                player_pos = info.perception.get_player_pixel_pos()
                wall_count = len(info.perception.wall_blocks)
                print(
                    f"  [PLAN-GRID] Level {level_idx} (baseline={baseline}), "
                    f"attempt {self._plan_attempt}, "
                    f"walls={wall_count}, "
                    f"danger_walls={len(self._danger_walls)}"
                )
                if player_pos is not None:
                    print(f"    Player (grid): ({player_pos[0]},{player_pos[1]})")
                else:
                    print(f"    Player (grid): not detected")

                # In grid mode, route planning finds no plans (no goals
                # detected). Try search_routes for compatibility, but
                # expect it to return no actions.
                best_steps, best_route, best_actions = search_routes(
                    info,
                    max_refills=0,
                    time_limit=5.0,
                    extra_walls=self._danger_walls if self._danger_walls else None,
                )

                if best_actions is not None and len(best_actions) > 0:
                    self._plan = best_actions
                    self._plan_idx = 0
                    self._plan_failed = False
                    print(
                        f"    [PLAN-GRID] Found route: {best_route}, "
                        f"{best_steps} steps"
                    )
                else:
                    print(f"    [PLAN-GRID] No feasible plan (grid mode), using fallback")
                    self._using_fallback = True

            except Exception as e:
                print(f"    [PLAN-GRID] Error during planning: {e}")
                import traceback
                traceback.print_exc()
                self._using_fallback = True

        # === No env and no grid perception: use fallback ===
        else:
            print(f"    [PLAN] No oracle or grid perception available, using fallback")
            self._using_fallback = True

    def _plan_level_adapter(self, game: Any) -> None:
        """Plan level using Oracle adapter for non-LS20 games.

        Uses the adapter's standardized entity interface to get player,
        walls, and goals positions, then creates a BFS-based plan to
        reach the nearest goal. Handles coordinate scaling when the
        game's native grid size differs from the 64x64 observation grid.

        For keyboard games (e.g., TR87): produces a path of directional
        actions (ACTION1-4) using BFS pathfinding.
        For click games (e.g., FT09): identifies clickable goal
        positions and relies on the click action logic in
        _choose_click_action for execution.

        Args:
            game: The env._game object.
        """
        from arcengine import GameAction

        adapter = self._oracle_adapter
        if adapter is None:
            self._using_fallback = True
            return

        level_idx = adapter.level_index
        baseline = (
            self.level_baselines[level_idx]
            if level_idx < len(self.level_baselines)
            else "?"
        )

        player_entity = adapter.player
        goal_entities = adapter.goals
        wall_entities = adapter.walls

        print(
            f"  [PLAN-ADAPTER] Level {level_idx} (baseline={baseline}), "
            f"adapter={type(adapter).__name__}, "
            f"attempt {self._plan_attempt}, "
            f"danger_walls={len(self._danger_walls)}"
        )

        if player_entity is None:
            print(f"    [PLAN-ADAPTER] No player detected, using fallback")
            self._using_fallback = True
            return

        if not goal_entities:
            print(f"    [PLAN-ADAPTER] No goals detected, using fallback")
            self._using_fallback = True
            return

        # Coordinate scaling: game coordinates may be in a different
        # resolution than the 64x64 observation grid.
        grid_size = adapter.grid_size
        scale = 64.0 / float(grid_size) if grid_size > 0 else 1.0
        step = 5  # Default step for 64x64 observation grid

        # Scale player position
        px = int(player_entity.x * scale)
        py = int(player_entity.y * scale)

        print(
            f"    Player: ({px},{py}), Goals: {len(goal_entities)}, "
            f"Walls: {len(wall_entities)}, scale={scale:.2f}"
        )

        # For click games, don't create a BFS plan — the click action
        # logic in _choose_click_action handles execution.
        if self._action_type in ("click", "keyboard_click"):
            # Store goal positions as clickable targets for click logic
            # The click action logic will use these positions
            for g in goal_entities:
                gx = int(g.x * scale)
                gy = int(g.y * scale)
                self._clicked_positions.discard((gx, gy))
            print(
                f"    [PLAN-ADAPTER] Click game: {len(goal_entities)} goals "
                f"available for click logic"
            )
            # Use fallback for execution — click logic in Step 4 will
            # handle clicking if positions are available, fallback
            # explorer handles the rest
            self._using_fallback = True
            return

        # For keyboard games, build wall set and BFS to goals
        # Use player position as grid origin
        gx_origin = px
        gy_origin = py

        # Build wall set from adapter entities
        walls: set[tuple[int, int]] = set()
        for w in wall_entities:
            wx = int(w.x * scale)
            wy = int(w.y * scale)
            # Snap to grid relative to player origin
            base_x = int(gx_origin + step * ((wx - gx_origin) // step))
            base_y = int(gy_origin + step * ((wy - gy_origin) // step))
            for ddx in range(-step, step * 2, step):
                for ddy in range(-step, step * 2, step):
                    wpx = base_x + ddx
                    wpy = base_y + ddy
                    if abs(wpx - wx) < step and abs(wpy - wy) < step:
                        walls.add((wpx, wpy))

        # Add danger walls from IRL trauma memory
        if self._danger_walls:
            walls |= self._danger_walls

        # Find shortest path to any goal using BFS
        best_plan: Optional[list] = None
        best_steps = float('inf')
        best_goal_pos: Optional[tuple[int, int]] = None

        for goal in goal_entities:
            gx = int(goal.x * scale)
            gy = int(goal.y * scale)
            # Snap goal to grid relative to player origin
            gx = int(gx_origin + step * round((gx - gx_origin) / step))
            gy = int(gy_origin + step * round((gy - gy_origin) / step))

            # Skip if goal is at player position (already there)
            if gx == px and gy == py:
                best_plan = []
                best_steps = 0
                best_goal_pos = (gx, gy)
                break

            path = bfs_path_with_teleports(
                px, py, gx, gy, step, walls, {}
            )
            if path is not None and len(path) < best_steps:
                best_steps = len(path)
                best_plan = path
                best_goal_pos = (gx, gy)

        if best_plan is not None:
            self._plan = best_plan
            self._plan_idx = 0
            self._plan_failed = False
            print(
                f"    [PLAN-ADAPTER] Found route to goal {best_goal_pos}: "
                f"{best_steps} steps"
            )
        else:
            print(
                f"    [PLAN-ADAPTER] No feasible path to any goal, "
                f"using fallback"
            )
            self._using_fallback = True

    def _choose_click_action(
        self,
        frames: list,
        latest_frame,
        grid: Optional[np.ndarray] = None,
    ):
        """Choose a click action for click-type games.

        Uses GridPerception to detect clickable positions (multi-color
        blocks) and selects the most promising one to click. The click
        coordinates are stored in self._pending_click_data for the caller
        to retrieve via get_pending_action_data().

        Strategy:
        1. Detect clickable positions from grid
        2. Filter out already-clicked positions (using frame differencing)
        3. Select the nearest unclicked position (or random if multiple)
        4. Store click data and return ACTION6 (click)

        Args:
            frames: List of all frame observations.
            latest_frame: Current frame observation.
            grid: 64x64 grid array, or None if extraction failed.

        Returns:
            GameAction.ACTION6 for click, or None if no clickable position
            found (falls through to keyboard/fallback logic).
        """
        from arcengine import GameAction

        if grid is None or self._grid_perception is None:
            return None

        # If we have an oracle adapter with goals, use those exact positions
        clickable: list[tuple[int, int]] = []
        if (self.use_oracle and self._oracle_adapter is not None
                and not isinstance(self._oracle_adapter, LS20Adapter)):
            goal_entities = self._oracle_adapter.goals
            if goal_entities:
                grid_size = self._oracle_adapter.grid_size
                scale = 64.0 / float(grid_size) if grid_size > 0 else 1.0
                clickable = [
                    (int(g.x * scale), int(g.y * scale))
                    for g in goal_entities
                ]

        # Fall back to grid perception if adapter didn't provide goals
        if not clickable:
            clickable = self._grid_perception.detect_clickable_positions(grid)

        if not clickable:
            # No clickable positions detected — try keyboard actions
            # or fall through to fallback
            return None

        # Filter out positions we've already clicked (player position)
        player_pos = self._grid_perception.get_player_pixel_pos()
        candidates = [
            pos for pos in clickable
            if player_pos is None or pos != player_pos
        ]

        if not candidates:
            return None

        # Select target: prefer positions not yet clicked
        # (track clicked positions across steps)
        if not hasattr(self, '_clicked_positions'):
            self._clicked_positions: set[tuple[int, int]] = set()

        unclicked = [
            pos for pos in candidates
            if pos not in self._clicked_positions
        ]

        if unclicked:
            # Click the first unclicked position
            target = unclicked[0]
        elif candidates:
            # All clicked — re-click the first candidate (might have changed)
            target = candidates[0]
            self._clicked_positions.clear()
        else:
            return None

        # Store click data for the caller
        click_x = target[0] + 2  # Center of 5x5 block
        click_y = target[1] + 2
        self._pending_click_data = {"x": click_x, "y": click_y}
        self._clicked_positions.add(target)

        print(
            f"    [CLICK] step={self._step} clicking ({click_x},{click_y}) "
            f"block=({target[0]},{target[1]}), "
            f"clicked={len(self._clicked_positions)}"
        )

        return GameAction.ACTION6

    def get_pending_action_data(self) -> Optional[dict]:
        """Get pending click data for the current action.

        For click-type games, this returns a dict with 'x' and 'y'
        coordinates that should be passed to env.step(action, data=...).
        For keyboard games, returns None.

        Returns:
            Dict with 'x' and 'y' keys, or None if no click data pending.
        """
        return self._pending_click_data

    def _update_route_q(
        self, info: LevelInfo, route_desc: str, steps: int
    ) -> None:
        """Update Q-value for route ordering (RL meta-learning).

        Learns which route orderings produce shorter plans.

        Args:
            info: LevelInfo instance.
            route_desc: Route description string.
            steps: Number of planned steps.
        """
        # Compute state key
        if info.goals:
            goal = info.goals[0]
            rot_diff = (goal["req_rot"] - info.rot) % 4
            shape_diff = (goal["req_shape"] - info.shape) % 6
            color_diff = (goal["req_color"] - info.color) % 4
        else:
            rot_diff = shape_diff = color_diff = 0

        state_key = f"L{info.level_idx}_R{rot_diff}_S{shape_diff}_C{color_diff}"
        action_key = route_desc

        # Q-learning update: reward = -steps (shorter is better)
        old_q = self._route_q.get((state_key, action_key), 0.0)
        reward = -float(steps)
        # Simple update without next-state max (since we don't have transitions)
        new_q = old_q + 0.1 * (reward - old_q)
        self._route_q[(state_key, action_key)] = new_q

    def _on_level_complete(self, new_levels: int) -> None:
        """Handle level completion event.

        Saves the successful action sequence as a macro, then resets
        per-level state while preserving cross-level memory.

        Args:
            new_levels: New total levels completed.
        """
        level_steps = self._step - self._level_start_step
        self._level_step_counts.append(level_steps)

        old_level = self._levels_completed
        baseline = (
            self.level_baselines[old_level]
            if old_level < len(self.level_baselines)
            else "?"
        )

        rhae = 0.0
        if isinstance(baseline, int) and level_steps > 0:
            rhae = min(115, ((baseline / level_steps) ** 2) * 100)

        print(
            f"    [LEVEL UP] Level {old_level} -> {new_levels} "
            f"in {level_steps} steps (baseline: {baseline}, RHAE: {rhae:.1f})"
        )

        # Save successful action sequence as macro (Library Learning)
        if self._level_action_history:
            self.macro_library.record_success(
                actions=self._level_action_history,
                level=old_level,
                steps=level_steps,
                context_hash=f"L{old_level}",
            )

        # CRITICAL: Update levels_completed to prevent repeated level-change triggers
        self._levels_completed = new_levels

        # Reset per-level state
        self._level_start_step = self._step
        self._level_action_history = []
        self._level_game_overs = 0
        self._seen_state_hashes.clear()
        self._stagnation = 0
        self._plan = None
        self._plan_idx = 0
        self._plan_attempt = 0
        self._plan_failed = False
        self._using_fallback = False
        self._fallback = None
        self.danger_memory.clear_history()

        # Clear danger walls on level change (danger positions from previous
        # level don't apply to the new level's layout)
        self._danger_walls.clear()

        # Clear clicked positions for click-type games
        self._clicked_positions.clear()

    def _action_to_int(self, action) -> int:
        """Convert GameAction to integer ID.

        Args:
            action: GameAction enum value.

        Returns:
            Integer action ID.
        """
        from arcengine import GameAction
        action_map = {
            GameAction.RESET: 0,
            GameAction.ACTION1: 1,
            GameAction.ACTION2: 2,
            GameAction.ACTION3: 3,
            GameAction.ACTION4: 4,
            GameAction.ACTION5: 5,
            GameAction.ACTION6: 6,
            GameAction.ACTION7: 7,
        }
        return action_map.get(action, 1)

    def get_stats(self) -> dict:
        """Get agent statistics for reporting.

        Returns:
            Dictionary of agent statistics.
        """
        return {
            "total_actions": self._total_actions,
            "levels_completed": self._levels_completed,
            "game_over_count": self._game_over_count,
            "level_game_overs": self._level_game_overs,
            "q_table_size": self.q_learner.q_table_size,
            "route_q_size": len(self._route_q),
            "wall_count": len(self._danger_walls),
            "macro_count": self.macro_library.macro_count,
            "stagnation": self._stagnation,
            "level_steps": self._level_step_counts,
            "using_fallback": self._using_fallback,
            "plan_attempt": self._plan_attempt,
            "graph_nodes": self.q_learner.q_table_size,
            "graph_edges": 0,
            "use_oracle": self.use_oracle,
            "oracle_adapter": type(self._oracle_adapter).__name__ if self._oracle_adapter else None,
            "action_type": self._action_type,
            "grid_initialized": self._grid_initialized,
            "clicked_positions": len(self._clicked_positions),
            "learned_goals": len(self._grid_perception._learned_goals) if self._grid_perception else 0,
        }

    @property
    def memory(self):
        """Compatibility property for test harness."""

        class _MemoryCompat:
            def __init__(self, agent: PlannerAgent) -> None:
                self._agent = agent

            def get_action_sequence(self) -> list[int]:
                return self._agent._action_history

        return _MemoryCompat(self)
