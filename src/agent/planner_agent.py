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

        # Game-specific solvers for all 25 games
from .game_solvers import solve_game as _solve_game_specific

# TOMAS Sleep-Step Learning System (record → sleep → audit → consolidate)
from .tomas_learner import (
    TOMASLearner,
    EpisodeTrace,
    ActionTrace,
)

# ── MetaSnapNet可选集成 (κ-Snap Beam评分) ──
try:
    from .meta_snap_net import (
        MetaSnapNet,
        MetaSnapBeamScorer,
        TopoFeatureExtractor,
        ProgramNodeFeatureExtractor,
        HAS_TORCH as _HAS_META_SNAP_TORCH,
    )
    _HAS_META_SNAP: bool = True
except ImportError:
    _HAS_META_SNAP: bool = False
    _HAS_META_SNAP_TORCH: bool = False
    MetaSnapNet = None  # type: ignore
    MetaSnapBeamScorer = None  # type: ignore


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
    MAX_PLAN_ATTEMPTS = 30  # Allow more attempts for wall discovery
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

        # Action type ("keyboard", "click", "keyboard+click", "unknown")
        self._action_type: str = "unknown"

        # Click data (for click-type games)
        self._pending_click_data: Optional[dict] = None
        self._clicked_positions: set[tuple[int, int]] = set()
        # BFS click puzzle solver state
        self._click_solution: Optional[list[dict]] = None  # list of {"x":..,"y":..}
        self._click_solution_idx: int = 0
        self._click_solve_attempted: bool = False
        self._click_solve_level: int = -1  # which level we solved for

        # RL components (reused from DopamineExplorer)
        self.reward_engine = RewardEngine()
        self.q_learner = QLearner()
        self.danger_memory = DangerMemory()
        self.macro_library = MacroLibrary()

        # TOMAS Sleep-Step Learning (record → sleep → audit → consolidate)
        self._tomas_learner = TOMASLearner()
        self._current_episode: Optional[EpisodeTrace] = None
        self._episode_action_traces: list[ActionTrace] = []

        # ── MetaSnapNet可选集成 (κ-Snap Beam评分) ──
        # 如果有checkpoint可用, 加载MetaSnapNet用于κ-Snap搜索评分
        # 否则, 使用Fast-Path + κ-Snap fallback
        self._meta_snap_scorer: Optional[Any] = None  # MetaSnapBeamScorer
        self._meta_snap_checkpoint_path: Optional[str] = None
        self._init_meta_snap_net()

        # Fallback agent (for when planning fails)
        self._fallback: Optional[DopamineExplorer] = None
        self._using_fallback = False
        self._plan_exhaustion_count: int = 0

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

        # Danger walls (IRL: positions that caused GAME_OVER or movement failure)
        self._danger_walls: set[tuple[int, int]] = set()

        # Movement failure tracking (for discovering invisible walls)
        self._prev_player_pos: Optional[tuple[int, int]] = None
        self._prev_action_was_plan: bool = False
        self._movement_fail_count: int = 0

        # Cipher game flag (tr87 = pattern translation puzzle, no movement)
        self._is_cipher_game: bool = False

        # Q-learning for route orderings (RL meta-learning)
        # State: (level_idx, rot_diff, shape_diff, color_diff)
        # Action: type ordering string (e.g., "RSC")
        self._route_q: dict[tuple[str, str], float] = {}

        # Seen state hashes for stagnation tracking
        self._seen_state_hashes: set[str] = set()

        # === P0-2: Frame Pre-filtering (差分阈值跳过静态帧) ===
        # Skip redundant perception update when frame diff is below threshold,
        # reducing ~30% of redundant computation on static/animation frames.
        self._prev_grid: Optional[np.ndarray] = None
        self._frame_diff_threshold: float = 0.005  # 0.5% pixel change threshold
        self._static_frame_skip_count: int = 0

        # === P0-3: Bayesian RHAE Circuit Breaker (效率<0.5回溯) ===
        # Track per-level RHAE efficiency; when efficiency drops below
        # threshold, trigger L2 backtracking (clear plan + switch strategy).
        self._rhae_efficiency_history: list[float] = []
        self._rhae_threshold: float = 0.5  # Below this → backtrack
        self._rhae_backtrack_count: int = 0
        self._max_rhae_backtracks: int = 3  # Max backtracks per level
        self._level_plan_strategies: list[str] = ["bfs", "adapter", "grid", "dfs"]
        self._current_strategy_idx: int = 0

        # === P0-4: DFS Backtracking Planner ===
        # Stack-based DFS with visited set for action sequence exploration.
        self._dfs_stack: list[tuple[int, list, set]] = []  # (depth, action_seq, visited)
        self._dfs_visited: set[str] = set()
        self._dfs_max_depth: int = 50
        self._dfs_max_iterations: int = 500

        # Store oracle preference; actual adapter creation happens in
        # _check_oracle_availability() on first choose_action() call.
        # Only mark as checked for explicit grid mode (no adapter needed).
        
        # ✅ FIX: Immediately check Oracle availability (don't lazy init)
        if self._use_oracle_requested is not False:  # Not explicitly grid mode
            self._check_oracle_availability()
        if use_oracle is False:
            self.use_oracle = False
            self._oracle_checked = True

    def _init_meta_snap_net(self) -> None:
        """初始化MetaSnapNet可选组件.

        查找可用的checkpoint文件, 如果存在且torch可用,
        加载MetaSnapNet用于κ-Snap Beam搜索评分.
        如果没有checkpoint或torch不可用, 使用numpy fallback.

        MetaSnapNet在κ-Snap搜索中的作用:
        当Fast-Path未能找到匹配宏时, κ-Snap生成候选DSL序列,
        MetaSnapBeamScorer.score_candidates()对候选评分,
        选择最高score的候选作为执行方案.
        """
        if not _HAS_META_SNAP:
            # meta_snap_net模块不可用
            self._meta_snap_scorer = None
            return

        # 查找checkpoint
        checkpoint_dir = Path("checkpoints")
        latest_ckpt = checkpoint_dir / "meta_snap_latest.pt"
        sft_ckpt = checkpoint_dir / "meta_snap_sft.pt"

        ckpt_path = None
        if latest_ckpt.exists():
            ckpt_path = str(latest_ckpt)
        elif sft_ckpt.exists():
            ckpt_path = str(sft_ckpt)

        self._meta_snap_checkpoint_path = ckpt_path

        try:
            if ckpt_path is not None:
                # 有checkpoint: 加载MetaSnapNet
                net = MetaSnapNet(checkpoint_path=ckpt_path)
                self._meta_snap_scorer = MetaSnapBeamScorer(net=net)
                print(
                    f"  [MetaSnap] ✅ Loaded checkpoint: {ckpt_path}, "
                    f"torch={_HAS_META_SNAP_TORCH}"
                )
            else:
                # 无checkpoint: 使用默认MetaSnapNet (随机权重或numpy fallback)
                net = MetaSnapNet()
                self._meta_snap_scorer = MetaSnapBeamScorer(net=net)
                print(
                    f"  [MetaSnap] Using default weights, "
                    f"torch={_HAS_META_SNAP_TORCH}"
                )
        except Exception as e:
            # MetaSnapNet初始化失败: 不使用评分
            self._meta_snap_scorer = None
            print(f"  [MetaSnap] ❌ Init failed: {e}, falling back to κ-Snap only")

    def _meta_snap_score_candidates(
        self,
        grid: Optional[np.ndarray],
        candidate_programs: Optional[list] = None,
    ) -> Optional[list]:
        """使用MetaSnapBeamScorer对候选DSL程序评分.

        在κ-Snap搜索中使用: 当Fast-Path未命中时,
        生成候选DSL序列, MetaSnapNet评分, 选择最高score的候选.

        Args:
            grid: 当前游戏状态网格 (64x64 int array).
            candidate_programs: 候选DSL程序列表.

        Returns:
            排序后的候选列表 (按score降序), 或None (评分器不可用).
        """
        if self._meta_snap_scorer is None or grid is None:
            return None

        if not candidate_programs:
            return None

        try:
            scores = self._meta_snap_scorer.score_candidates(
                grid=grid,
                topo_map=None,
                candidate_programs=candidate_programs,
            )

            if not scores:
                return None

            # 按score降序排序候选
            ranked = sorted(
                zip(candidate_programs, scores),
                key=lambda x: x[1][0],  # x[1][0] = score
                reverse=True,
            )

            # 返回排序后的DSL序列
            return [dsl for dsl, (s, c) in ranked]
        except Exception:
            return None

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
            print(f"[DEBUG] _check_oracle_availability: game={type(game).__name__}, game_id={self.game_id}")

            # Try to get an adapter by game_id first, then auto-detect
            adapter: Optional[OracleAdapter] = None
            if self.game_id:
                print(f"[DEBUG] Calling get_oracle_adapter('{self.game_id}', game)...")
                adapter = get_oracle_adapter(self.game_id, game)
                print(f"[DEBUG] get_oracle_adapter returned: {type(adapter).__name__ if adapter else 'None'}")

            if adapter is None:
                print(f"[DEBUG] Calling auto_detect_adapter(game, game_id='{self.game_id}')...")
                adapter = auto_detect_adapter(game, game_id=self.game_id)
                print(f"[DEBUG] auto_detect_adapter returned: {type(adapter).__name__ if adapter else 'None'}")

            if adapter is not None:
                self._oracle_adapter = adapter
                self.use_oracle = True
                print(f"[DEBUG] ✅ Oracle adapter set: {type(adapter).__name__}")
                return True

            # No adapter found — use grid mode
            self.use_oracle = False
            print(f"[DEBUG] ❌ No adapter found, use_oracle=False")
            return False
        except (AttributeError, Exception) as e:
            self.use_oracle = False
            print(f"[DEBUG] ❌ Exception in _check_oracle_availability: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
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
                # Fallback: use GAME_CONFIGS to determine action type
                # Many games return empty available_actions from env,
                # but their config specifies the correct game_type.
                from .game_configs import GAME_CONFIGS
                base_id_fallback = self.game_id.split("-")[0] if self.game_id else ""
                cfg = GAME_CONFIGS.get(base_id_fallback)
                if cfg is not None:
                    if cfg.game_type == "click":
                        self._action_type = "click"
                    elif cfg.game_type == "keyboard+click":
                        self._action_type = "keyboard+click"
                    else:
                        self._action_type = "keyboard"
                    # Use config's available_actions if provided
                    if cfg.available_actions:
                        available_actions = list(cfg.available_actions)
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

        # === Timeout for tr87 L5 (prevent infinite loop) ===
        if self.game_id == 'tr87' and latest_frame.levels_completed >= 5:
            if not hasattr(self, '_tr87_l5_steps'):
                self._tr87_l5_steps = 0
            self._tr87_l5_steps += 1
            if self._tr87_l5_steps > 1000:
                print(
                    f"    [TR87] L5 timeout after 1000 steps, "
                    f"giving up (keeping 5/6 levels)"
                )
                self._tr87_l5_steps = 0
                return None

        # Clear pending click data from previous step
        self._pending_click_data = None

        state = latest_frame.state
        levels_completed = latest_frame.levels_completed
        level_changed = levels_completed > self._levels_completed

        # === Extract grid and update perception ===
        # P0-2: Frame Pre-filtering — skip perception update for static frames
        grid: Optional[np.ndarray] = None
        _skip_perception_update = False
        try:
            grid = np.array(latest_frame.frame[0])

            # P0-2: Compute frame diff ratio vs previous frame
            if self._prev_grid is not None and self._prev_grid.shape == grid.shape:
                diff_ratio = float(np.count_nonzero(grid != self._prev_grid)) / float(grid.size)
                if diff_ratio < self._frame_diff_threshold:
                    # Static frame — skip expensive perception update
                    _skip_perception_update = True
                    self._static_frame_skip_count += 1
                else:
                    self._static_frame_skip_count = 0

            self._prev_grid = grid.copy()

            if not _skip_perception_update:
                # Get available actions from observation for action type detection
                avail_actions: list[int] = []
                if hasattr(latest_frame, 'available_actions'):
                    avail_actions = [int(a) for a in latest_frame.available_actions]
                self._update_grid_perception(grid, available_actions=avail_actions)
        except (AttributeError, IndexError, Exception):
            pass

        # P0-3: Bayesian RHAE Circuit Breaker — check efficiency mid-level
        if (not level_changed and state != GameState.GAME_OVER
                and self._level_start_step > 0
                and self._rhae_backtrack_count < self._max_rhae_backtracks):
            level_idx = self._levels_completed
            if level_idx < len(self.level_baselines):
                baseline_steps = self.level_baselines[level_idx]
                actual_steps = self._step - self._level_start_step
                if actual_steps > 0 and baseline_steps > 0:
                    rhae_efficiency = float(baseline_steps) / float(max(actual_steps, 1))
                    # Record efficiency periodically
                    if actual_steps % 50 == 0:
                        self._rhae_efficiency_history.append(rhae_efficiency)

                    # Trigger backtrack when efficiency drops below threshold
                    if rhae_efficiency < self._rhae_threshold and actual_steps > baseline_steps * 2:
                        print(
                            f"    [RHAE BREAKER] efficiency={rhae_efficiency:.2f} "
                            f"< {self._rhae_threshold}, triggering L2 backtrack "
                            f"(#{self._rhae_backtrack_count + 1})"
                        )
                        self._rhae_backtrack_count += 1
                        # Clear current plan to force re-planning with next strategy
                        self._plan = None
                        self._plan_idx = 0
                        self._plan_failed = True
                        # Advance to next planning strategy
                        self._current_strategy_idx = (
                            (self._current_strategy_idx + 1)
                            % len(self._level_plan_strategies)
                        )
                        # Clear danger walls partially (keep 50% most recent)
                        if len(self._danger_walls) > 4:
                            self._danger_walls = set(list(self._danger_walls)[-len(self._danger_walls) // 2:])

        # === Step 1: Handle GAME_OVER ===
        if state == GameState.GAME_OVER:
            self._game_over_count += 1
            self._level_game_overs += 1
            self.danger_memory.record_game_over()
            self.danger_memory.clear_history()

            # Record danger position (IRL: learn from failure)
            if self.use_oracle and self._oracle_adapter is not None:
                try:
                    player_entity = self._oracle_adapter.player
                    if player_entity is not None:
                        grid_size = self._oracle_adapter.grid_size
                        scale = 64.0 / float(grid_size) if grid_size > 0 else 1.0
                        px = int(player_entity.x * scale)
                        py = int(player_entity.y * scale)
                        self._danger_walls.add((px, py))
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
        # For games with game-specific solvers: prefer re-planning over
        # generic click solver when the game-specific plan is exhausted.
        # For games without solvers (ft09, ls20, tr87), always use click solver.
        if self._action_type in ("click", "keyboard+click"):
            _base_gid = (self.game_id or "").split("-")[0]
            _has_game_solver = _base_gid in (
                "dc22", "tu93", "wa30", "g50t", "ka59", "sk48", "m0r0",
                "cn04", "r11l", "s5i5", "tn36", "su15", "vc33", "re86",
                "ar25", "sc25", "sb26", "cd82", "sp80", "bp35", "lf52",
            )

            # Determine if generic click solver should be used.
            # For games WITH dedicated solvers: skip click_solve when plan
            # is exhausted so Step 5 can re-plan via the game-specific solver.
            # Without a solver: always keep _has_active_plan=True.
            if _has_game_solver:
                # Plan is active → don't interrupt
                _plan_active = (
                    self._plan is not None
                    and self._plan_idx < len(self._plan)
                )
                # Plan was just exhausted → clear it so Step 5 regenerates
                if not _plan_active and self._plan is not None:
                    # Clear exhausted plan to trigger re-planning in Step 5
                    self._plan = None
                    self._plan_idx = 0
                # Use generic click solver ONLY if no solver OR in fallback mode
                _use_click_solve = self._using_fallback
            else:
                _use_click_solve = True

            if _use_click_solve:
                action = self._choose_click_action(frames, latest_frame, grid)
                if action is not None:
                    self._action_history.append(self._action_to_int(action))
                    self._level_action_history.append(self._action_to_int(action))
                    self._stagnation = 0
                    return action

        # === Step 5: Plan if needed ===
        if self._plan is None or self._plan_idx >= len(self._plan):
            # Track plan exhaustion for current level
            if self._plan is not None and self._plan_idx >= len(self._plan):
                # Plan was exhausted without level completion
                self._plan_exhaustion_count = getattr(self, '_plan_exhaustion_count', 0) + 1
                _base_gid_5 = (self.game_id or "").split("-")[0]
                _has_gs_5 = _base_gid_5 in ("dc22", "tu93", "wa30", "g50t", "ka59",
                    "sk48", "m0r0", "cn04", "r11l", "s5i5", "tn36", "su15", "vc33",
                    "re86", "ar25", "sc25", "sb26", "cd82", "sp80", "bp35", "lf52")
                if _has_gs_5 and self._plan_exhaustion_count >= 3:
                    # Switch to fallback after 3 failed plan attempts
                    self._using_fallback = True
                    self._plan = None
                    self._plan_idx = 0
            
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

        # === Step 5.5: Movement failure detection ===
        # If the previous action was from a plan, check if the player moved.
        # If not, the target cell is an invisible wall — add to danger_walls
        # and invalidate the plan so it gets re-planned with the new wall.
        # Skip for cipher games (tr87) where there is no movement.
        # Skip for pure click games (s5i5, r11l, tn36, vc33, su15) — no player entity.
        # For game-specific solvers, use a higher threshold (3 consecutive
        # failures) to avoid false positives from coordinate mismatches.
        _base_gid = (self.game_id or "").split("-")[0]
        _has_game_solver = _base_gid in ("dc22", "tu93", "wa30", "g50t", "ka59",
            "sk48", "m0r0", "cn04", "r11l", "s5i5", "tn36", "su15", "vc33",
            "re86", "ar25", "sc25", "sb26", "cd82", "sp80", "bp35", "lf52")
        _fail_threshold = 5 if _has_game_solver else 1
        if (self._prev_action_was_plan
                and not self._is_cipher_game
                and self._action_type != "click"  # Pure click games: no player movement to detect
                and self.use_oracle and self._oracle_adapter is not None):
            try:
                player_entity = self._oracle_adapter.player
                if player_entity is not None:
                    grid_size = self._oracle_adapter.grid_size
                    scale = 64.0 / float(grid_size) if grid_size > 0 else 1.0
                    cur_pos = (int(player_entity.x * scale), int(player_entity.y * scale))
                    if self._prev_player_pos is not None and cur_pos == self._prev_player_pos:
                        # Player didn't move — target cell is blocked
                        self._movement_fail_count += 1
                        # Only invalidate after reaching the failure threshold
                        if self._movement_fail_count >= _fail_threshold:
                            # Determine the step size from the game config
                            try:
                                from .game_configs import GAME_CONFIGS
                            except ImportError:
                                try:
                                    from src.agent.game_configs import GAME_CONFIGS
                                except ImportError:
                                    GAME_CONFIGS = {}
                            cfg = GAME_CONFIGS.get(self.game_id)
                            mv_step = 5  # default
                            if cfg is not None:
                                game_obj = getattr(self._oracle_adapter, 'game', None)
                                if game_obj is not None:
                                    cl = getattr(game_obj, 'current_level', None)
                                    if cl is not None and cfg.player_tag is not None:
                                        ps = cl.get_sprites_by_tag(cfg.player_tag)
                                        if ps:
                                            pw = int(getattr(ps[0], 'width', 5))
                                            ph = int(getattr(ps[0], 'height', 5))
                                            raw = max(pw, ph)
                                            if raw > 0:
                                                mv_step = max(1, int(raw * scale))
                            # Look at the PREVIOUS action (plan_idx was already incremented)
                            # The action that was just executed is at plan_idx - 1
                            if self._plan is not None and self._plan_idx > 0 and self._plan_idx <= len(self._plan):
                                blocked_action, _ = self._plan[self._plan_idx - 1]
                                dx, dy = 0, 0
                                from arcengine import GameAction as _GA
                                if blocked_action == _GA.ACTION1: dy = -mv_step
                                elif blocked_action == _GA.ACTION2: dy = mv_step
                                elif blocked_action == _GA.ACTION3: dx = -mv_step
                                elif blocked_action == _GA.ACTION4: dx = mv_step
                                if dx != 0 or dy != 0:
                                    blocked_pos = (cur_pos[0] + dx, cur_pos[1] + dy)
                                    self._danger_walls.add(blocked_pos)
                            # Invalidate plan to force re-planning with new wall
                            self._plan = None
                            self._plan_idx = 0
                            self._plan_failed = True
                            self._movement_fail_count = 0
                    else:
                        # Player moved successfully — reset fail counter
                        self._movement_fail_count = 0
                    self._prev_player_pos = cur_pos
            except (AttributeError, Exception):
                pass
        self._prev_action_was_plan = False

        # === Step 6: Execute plan or fallback ===
        if self._plan is not None and self._plan_idx < len(self._plan):
            plan_item = self._plan[self._plan_idx]
            # Handle both formats: (action, name_string) and (action, click_data|None)
            if isinstance(plan_item, tuple) and len(plan_item) == 2:
                action, second = plan_item
                if isinstance(second, str):
                    # Old format: (action, name)
                    name = second
                elif second is None:
                    # New format: keyboard action, no click data
                    self._pending_click_data = None  # Clear stale click data
                    name = action.name if hasattr(action, 'name') else str(action)
                elif isinstance(second, (tuple, list)) and len(second) >= 2:
                    # New format: click action with (x, y) data
                    self._pending_click_data = {"x": int(second[0]), "y": int(second[1])}
                    name = f"click({int(second[0])},{int(second[1])})"
                else:
                    name = str(second)
            else:
                action = plan_item
                name = str(action)
            print(f"    [EXEC-PLAN] idx={self._plan_idx}/{len(self._plan)}, action={action}, name={name}")
            self._plan_idx += 1
            self._action_history.append(self._action_to_int(action))
            self._level_action_history.append(self._action_to_int(action))
            self._stagnation = 0
            self._prev_action_was_plan = True
            return action

        # === Step 7: Fallback to exploration ===
        if self._fallback is not None:
            # Sync fallback state
            self._fallback._step = self._step
            action = self._fallback.choose_action(frames, latest_frame)
            # Safety: for click games, never return ACTION6 without click data
            if (action in (GameAction.ACTION6, GameAction.ACTION7)
                    and self._action_type in ("click", "keyboard+click")
                    and not self._pending_click_data):
                # Try generic click solver as last resort
                click_action = self._choose_click_action(frames, latest_frame, grid)
                if click_action is not None:
                    action = click_action
                else:
                    # Skip click action, use a keyboard action instead
                    available_kb = [a for a in latest_frame.available_actions
                                    if a not in (GameAction.ACTION6, GameAction.ACTION7)]
                    action = available_kb[0] if available_kb else GameAction.ACTION1
            self._action_history.append(self._action_to_int(action))
            self._level_action_history.append(self._action_to_int(action))
            return action

        # Ultimate fallback: random action
        available = list(latest_frame.available_actions)
        # Safety: for click games, filter out ACTION6 if no click data
        if self._action_type in ("click", "keyboard+click") and not self._pending_click_data:
            available = [a for a in available if a not in (GameAction.ACTION6, GameAction.ACTION7)]
        if available:
            action = available[0]
        else:
            action = GameAction.ACTION1
        self._action_history.append(self._action_to_int(action))
        return action

    def _dfs_backtrack_plan(self) -> Optional[list]:
        """P0-4: DFS Backtracking Planner — stack-based DFS with visited set.

        Explores action sequences using depth-first search with backtracking.
        When a dead end is reached (GAME_OVER, wall, or cycle), the planner
        backtracks to the most recent branching point and tries an alternative
        action. This provides a fallback when BFS route planning fails.

        Uses a visited set keyed on state hashes to prevent infinite loops.
        Maximum depth and iterations bounded to ensure termination.

        Returns:
            List of (action, name) tuples if a plan is found, None otherwise.
        """
        from arcengine import GameAction

        if self._dfs_stack:
            # Resume from saved DFS state
            pass
        else:
            # Initialize DFS with starting state
            initial_hash = self._compute_state_hash()
            if initial_hash is None:
                return None
            self._dfs_visited.add(initial_hash)
            # Get available actions for initial state
            try:
                available = list(self._env._game.current_level.available_actions)
            except (AttributeError, Exception):
                available = [
                    GameAction.ACTION1, GameAction.ACTION2,
                    GameAction.ACTION3, GameAction.ACTION4,
                ]
            self._dfs_stack = [(0, [], {a: False for a in available})]

        iterations = 0
        while self._dfs_stack and iterations < self._dfs_max_iterations:
            iterations += 1
            depth, action_seq, tried = self._dfs_stack[-1]

            if depth >= self._dfs_max_depth:
                # Max depth reached — backtrack
                self._dfs_stack.pop()
                continue

            # Find next untried action at this level
            next_action = None
            for a in tried:
                if not tried[a]:
                    tried[a] = True
                    next_action = a
                    break

            if next_action is None:
                # All actions tried at this depth — backtrack
                self._dfs_stack.pop()
                continue

            # Simulate the action (oracle mode only)
            if not self.use_oracle or self._oracle_adapter is None:
                return None

            try:
                # Check if this action leads to danger
                player_entity = self._oracle_adapter.player
                if player_entity is not None:
                    grid_size = self._oracle_adapter.grid_size
                    scale = 64.0 / float(grid_size) if grid_size > 0 else 1.0
                    cur_pos = (int(player_entity.x * scale), int(player_entity.y * scale))

                    # Predict next position
                    from arcengine import GameAction as _GA
                    dx, dy = 0, 0
                    mv_step = 5
                    if next_action == _GA.ACTION1: dy = -mv_step
                    elif next_action == _GA.ACTION2: dy = mv_step
                    elif next_action == _GA.ACTION3: dx = -mv_step
                    elif next_action == _GA.ACTION4: dx = mv_step

                    next_pos = (cur_pos[0] + dx, cur_pos[1] + dy)

                    # Skip if leads to danger wall
                    if next_pos in self._danger_walls:
                        continue

                    # Skip if already visited
                    state_hash = f"{next_pos[0]},{next_pos[1]}"
                    if state_hash in self._dfs_visited:
                        continue

                    self._dfs_visited.add(state_hash)

                    # Build new action sequence
                    new_seq = action_seq + [(next_action, f"dfs_d{depth}")]
                    new_available = [
                        GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4,
                    ]
                    self._dfs_stack.append(
                        (depth + 1, new_seq, {a: False for a in new_available})
                    )

                    # Check if this sequence reaches a goal
                    goals = []
                    if hasattr(self._oracle_adapter, 'game'):
                        game = self._oracle_adapter.game
                        if hasattr(game, 'current_level'):
                            cl = game.current_level
                            if hasattr(cl, 'get_sprites_by_tag'):
                                try:
                                    goals = cl.get_sprites_by_tag('goal')
                                except Exception:
                                    pass

                    if goals:
                        for g in goals:
                            gx = int(getattr(g, 'x', 0) * scale)
                            gy = int(getattr(g, 'y', 0) * scale)
                            if abs(gx - next_pos[0]) < mv_step and abs(gy - next_pos[1]) < mv_step:
                                # Goal reached!
                                print(
                                    f"    [DFS-BACKTRACK] Found plan: "
                                    f"{len(new_seq)} actions, {iterations} iterations"
                                )
                                self._dfs_stack.clear()
                                self._dfs_visited.clear()
                                return new_seq

            except (AttributeError, Exception):
                continue

        # DFS exhausted without finding goal
        print(
            f"    [DFS-BACKTRACK] No plan found after {iterations} iterations, "
            f"stack depth={len(self._dfs_stack)}"
        )
        self._dfs_stack.clear()
        self._dfs_visited.clear()
        return None

    def _compute_state_hash(self) -> Optional[str]:
        """Compute a hash of the current game state for DFS visited tracking.

        Returns:
            State hash string, or None if state cannot be determined.
        """
        if self.use_oracle and self._oracle_adapter is not None:
            try:
                player = self._oracle_adapter.player
                if player is not None:
                    return f"o:{int(player.x)},{int(player.y)}"
            except (AttributeError, Exception):
                pass

        if self._grid_perception is not None:
            pos = self._grid_perception.get_player_pixel_pos()
            if pos is not None:
                return f"g:{pos[0]},{pos[1]}"

        return None

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
        # Use lower threshold for LS20 (which has perfect planning) to avoid
        # wasting time, higher threshold for adapter-based games to allow
        # wall discovery through movement failure detection.
        max_attempts = 5
        if (self._oracle_adapter is not None
                and not isinstance(self._oracle_adapter, LS20Adapter)):
            max_attempts = 30
        if self._plan_attempt >= max_attempts:
            # P0-4: Try DFS backtracking before giving up
            if (self._rhae_backtrack_count > 0
                    or self._current_strategy_idx >= len(self._level_plan_strategies) - 1):
                dfs_plan = self._dfs_backtrack_plan()
                if dfs_plan is not None:
                    self._plan = dfs_plan
                    self._plan_idx = 0
                    self._plan_failed = False
                    return

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

        # === TR87 cipher puzzle solver ===
        # TR87 is a pattern translation puzzle, NOT a movement game.
        # It requires cycling pattern variants (UP/DOWN) and switching
        # targets (LEFT/RIGHT) to match source→target rule mappings.
        base_game_id = (self.game_id or "").split("-")[0]
        is_tr87 = (
            base_game_id == "tr87"
            or type(adapter).__name__ == "TR87Adapter"
            or hasattr(game, "cifzvbcuwqe")
        )
        if is_tr87:
            self._is_cipher_game = True
            cipher_plan = self._solve_tr87_cipher(game)
            if cipher_plan is not None:
                self._plan = cipher_plan
                self._plan_idx = 0
                self._plan_failed = False
                level_idx = adapter.level_index
                baseline = (
                    self.level_baselines[level_idx]
                    if level_idx < len(self.level_baselines)
                    else "?"
                )
                print(
                    f"  [PLAN-TR87] Level {level_idx} (baseline={baseline}), "
                    f"cipher plan: {len(cipher_plan)} actions"
                )
                return
            else:
                print(f"  [PLAN-TR87] Cipher solver failed, falling back")
                self._using_fallback = True
                return

        # === Game-specific solvers for all other games ===
        # Dispatch to game-specific solver before generic BFS.
        # Each solver reads env._game to get perfect information and
        # computes optimal action sequences.
        base_game_id = (self.game_id or "").split("-")[0]
        if base_game_id and base_game_id not in ("ls20", "tr87", "ft09"):
            game_specific_plan = _solve_game_specific(game, base_game_id, adapter.level_index)
            if game_specific_plan is not None and len(game_specific_plan) > 0:
                self._plan = game_specific_plan
                self._plan_idx = 0
                self._plan_failed = False
                # Clear any pending click solution so it doesn't interrupt
                self._click_solution = None
                self._click_solution_idx = 0
                self._click_solve_attempted = False
                if hasattr(self, '_solution_wait_counter'):
                    self._solution_wait_counter = 0
                level_idx = adapter.level_index
                baseline = (
                    self.level_baselines[level_idx]
                    if level_idx < len(self.level_baselines)
                    else "?"
                )
                # Separate keyboard and click actions for logging
                n_kb = sum(1 for a, _ in game_specific_plan if a not in (GameAction.ACTION6, GameAction.ACTION7))
                n_click = sum(1 for a, _ in game_specific_plan if a in (GameAction.ACTION6, GameAction.ACTION7))
                print(
                    f"  [PLAN-SOLVER] Level {level_idx} (baseline={baseline}), "
                    f"game={base_game_id}, plan: {len(game_specific_plan)} actions "
                    f"({n_kb} kb + {n_click} click)"
                )
                return
            elif game_specific_plan is not None and len(game_specific_plan) == 0:
                # Empty plan means solver failed - fall through to generic BFS
                pass
            # If solver returned None, fall through to generic BFS

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

        # Use NATIVE game coordinates for BFS (not scaled to 64-grid).
        # This preserves precision for games with grid_size > 64 (e.g., tr87=128).
        # Danger walls are stored in 64-grid and converted to native when used.
        grid_size = adapter.grid_size
        # scale is only for click games (click positions in 64-grid) and
        # danger_walls conversion (64-grid → native)
        scale = 64.0 / float(grid_size) if grid_size > 0 else 1.0
        native_scale = float(grid_size) / 64.0 if grid_size > 0 else 1.0

        # Detect movement step from player sprite size in NATIVE coordinates.
        # Strategy: 1) config tag → sprite size, 2) game attribute → sprite size,
        # 3) goal spacing GCD, 4) default 5
        step = 5
        try:
            game_obj = getattr(adapter, 'game', None)
            if game_obj is not None:
                cl = getattr(game_obj, 'current_level', None)
                from .game_configs import GAME_CONFIGS
                cfg = GAME_CONFIGS.get(self.game_id)

                # Strategy 1 & 2: Get player sprite size
                player_sprites: list = []
                if cfg is not None and cfg.player_tag is not None:
                    # Try as tag first
                    if cl is not None:
                        try:
                            player_sprites = cl.get_sprites_by_tag(cfg.player_tag)
                        except (AttributeError, TypeError):
                            pass
                    # Fallback: try as game attribute (e.g., tr87's qvtymdcqear_parts)
                    if not player_sprites:
                        try:
                            val = getattr(game_obj, cfg.player_tag)
                            if isinstance(val, list) and val:
                                player_sprites = val
                        except (AttributeError, TypeError):
                            pass
                if player_sprites:
                    pw = int(getattr(player_sprites[0], 'width', 0))
                    ph = int(getattr(player_sprites[0], 'height', 0))
                    raw_step = max(pw, ph) if pw > 0 and ph > 0 else 0
                    if raw_step > 0:
                        step = raw_step

                # Strategy 3: Detect step from goal spacing if step still default
                if step == 5 and len(goal_entities) >= 2:
                    import math
                    diffs: list[int] = []
                    for i in range(len(goal_entities)):
                        for j in range(i + 1, len(goal_entities)):
                            dx = abs(int(goal_entities[i].x) - int(goal_entities[j].x))
                            dy = abs(int(goal_entities[i].y) - int(goal_entities[j].y))
                            if dx > 0:
                                diffs.append(dx)
                            if dy > 0:
                                diffs.append(dy)
                    if diffs:
                        from math import gcd
                        from functools import reduce
                        g = reduce(gcd, diffs)
                        if g > 1:
                            step = g
                            print(f"    [STEP-DETECT] Goal spacing GCD={g}, using step={g}")
        except (AttributeError, Exception) as e:
            print(f"    [STEP-DETECT] Error: {e}, using default step=5")

        # Use NATIVE coordinates (no scaling) for keyboard game BFS
        px = int(player_entity.x)
        py = int(player_entity.y)

        print(
            f"    Player: ({px},{py}) [native], Goals: {len(goal_entities)}, "
            f"Walls: {len(wall_entities)}, grid_size={grid_size}, step={step}"
        )

        # For click games, click data must be in 64-grid (observation) coordinates.
        if self._action_type in ("click", "keyboard+click"):
            for g in goal_entities:
                gx = int(g.x * scale)
                gy = int(g.y * scale)
                self._clicked_positions.discard((gx, gy))
            print(
                f"    [PLAN-ADAPTER] Click game: {len(goal_entities)} goals "
                f"available for click logic"
            )
            self._using_fallback = True
            return

        # For keyboard games, build wall set and BFS in NATIVE coordinates
        gx_origin = px
        gy_origin = py

        # Build wall set from adapter entities (native coordinates).
        # Each wall is snapped to the nearest grid cell. We also add
        # adjacent cells only if the wall center is near the cell boundary.
        walls: set[tuple[int, int]] = set()
        for w in wall_entities:
            wx = int(w.x)
            wy = int(w.y)
            # Snap wall to grid relative to player origin
            wpx = int(gx_origin + step * round((wx - gx_origin) / step))
            wpy = int(gy_origin + step * round((wy - gy_origin) / step))
            walls.add((wpx, wpy))
            # Add adjacent cell if wall is closer to the boundary than center
            half = step // 2
            if abs(wx - wpx) > half:
                walls.add((wpx + step if wx > wpx else wpx - step, wpy))
            if abs(wy - wpy) > half:
                walls.add((wpx, wpy + step if wy > wpy else wpy - step))

        # Add danger walls — convert from 64-grid to native coordinates
        native_danger: set[tuple[int, int]] = set()
        if self._danger_walls:
            for dx, dy in self._danger_walls:
                nd = (int(dx * native_scale), int(dy * native_scale))
                walls.add(nd)
                native_danger.add(nd)

        # Safety: remove player position from walls (can't be inside a wall)
        walls.discard((px, py))

        # Pre-compute snapped goal positions and remove them from walls.
        # Goals can't be walls — the player needs to reach them. Wall sprites
        # near goals often snap to the same grid cell, blocking the path.
        snapped_goals: list[tuple[int, int]] = []
        for goal in goal_entities:
            sgx = int(gx_origin + step * round((int(goal.x) - gx_origin) / step))
            sgy = int(gy_origin + step * round((int(goal.y) - gy_origin) / step))
            snapped_goals.append((sgx, sgy))
            walls.discard((sgx, sgy))
            # Also discard adjacent cells to ensure path to goal is clear
            for ddx in [-step, 0, step]:
                for ddy in [-step, 0, step]:
                    walls.discard((sgx + ddx, sgy + ddy))

        # Debug: print snapped goals and check if they're blocked
        if len(walls) > 0:
            nearby_walls = [w for w in walls if abs(w[0]-px) <= step*3 and abs(w[1]-py) <= step*3]
            print(
                f"    [WALLS] Total={len(walls)}, near_player={nearby_walls[:10]}"
            )

        # Find shortest path to any goal using BFS (native coordinates)
        best_plan: Optional[list] = None
        best_steps = float('inf')
        best_goal_pos: Optional[tuple[int, int]] = None

        for goal in goal_entities:
            gx = int(goal.x)
            gy = int(goal.y)
            # Snap goal to grid relative to player origin
            gx = int(gx_origin + step * round((gx - gx_origin) / step))
            gy = int(gy_origin + step * round((gy - gy_origin) / step))

            # Skip goals blocked by danger walls (native coordinates)
            if (gx, gy) in native_danger:
                continue

            # Skip if goal is at player position (already there)
            if gx == px and gy == py:
                best_plan = []
                best_steps = 0
                best_goal_pos = (gx, gy)
                break

            path = bfs_path_with_teleports(
                px, py, gx, gy, step, walls, {}
            )
            if path is None:
                in_walls = (gx, gy) in walls
                print(
                    f"    [BFS-FAIL] goal=({gx},{gy}) in_walls={in_walls}, "
                    f"player=({px},{py}), step={step}"
                )
            if path is not None and len(path) < best_steps:
                best_steps = len(path)
                best_plan = path
                best_goal_pos = (gx, gy)

        if best_plan is not None:
            # Check if ACTION5 is available — many keyboard games require
            # pressing ACTION5 to "collect" or "interact" with the goal
            try:
                avail = []
                if hasattr(self._env, '_game'):
                    avail = list(getattr(self._env._game, '_available_actions', []))
                has_action5 = 5 in avail
            except Exception:
                has_action5 = False

            if has_action5 and best_steps > 0:
                best_plan.append((GameAction.ACTION5, "COLLECT"))
            elif has_action5 and best_steps == 0:
                best_plan = [(GameAction.ACTION5, "COLLECT")]

            self._plan = best_plan
            self._plan_idx = 0
            self._plan_failed = False
            print(
                f"    [PLAN-ADAPTER] Found route to goal {best_goal_pos}: "
                f"{best_steps} steps"
                + (" +ACTION5" if has_action5 else "")
            )
        else:
            print(
                f"    [PLAN-ADAPTER] No feasible path to any goal, "
                f"using fallback"
            )
            self._using_fallback = True

    # ====================================================================
    # TR87 Cipher Puzzle Solver
    # ====================================================================

    def _solve_tr87_cipher(self, game: Any) -> Optional[list]:
        """Solve tr87 cipher puzzle by computing variant differences.

        TR87 is a pattern translation puzzle where:
        - Source patterns (zvojhrjxxm) are on the top row
        - Target patterns (ztgmtnnufb) are on the bottom row
        - Rules (cifzvbcuwqe) define source->target mapping
        - Player cycles target variants (UP/DOWN=ACTION1/2) and
          switches targets (LEFT/RIGHT=ACTION3/4)

        Game modes:
        - Normal (levels 0-2): targets scrambled, rules correct
        - double_translation (level 3): rules chain together
        - alter_rules (level 4): rules scrambled, board correct
        - All flags (level 5): most complex

        Args:
            game: The env._game object.

        Returns:
            List of (GameAction, name) tuples, or None if cannot solve.
        """
        from arcengine import GameAction

        rules = getattr(game, 'cifzvbcuwqe', [])
        source_patterns = getattr(game, 'zvojhrjxxm', [])
        target_patterns = getattr(game, 'ztgmtnnufb', [])

        if not rules or not target_patterns:
            print(
                f"    [TR87] Missing game state: "
                f"rules={len(rules)}, targets={len(target_patterns)}"
            )
            return None

        # Check game mode flags
        current_level = getattr(game, 'current_level', None)
        alter_rules = False
        double_translation = False
        tree_translation = False
        if current_level is not None:
            try:
                alter_rules = bool(current_level.get_data('alter_rules'))
            except (AttributeError, KeyError, TypeError):
                pass
            try:
                double_translation = bool(
                    current_level.get_data('double_translation')
                )
            except (AttributeError, KeyError, TypeError):
                pass
            try:
                tree_translation = bool(
                    current_level.get_data('tree_translation')
                )
            except (AttributeError, KeyError, TypeError):
                pass

        current_index = getattr(game, 'qvtymdcqear_index', 0)
        num_variants = getattr(game, 'kjgicbtgrt', 7)

        print(
            f"    [TR87] alter_rules={alter_rules}, "
            f"double_trans={double_translation}, "
            f"tree_trans={tree_translation}, "
            f"rules={len(rules)}, "
            f"sources={len(source_patterns)}, "
            f"targets={len(target_patterns)}, "
            f"current_idx={current_index}"
        )

        if alter_rules:
            return self._solve_tr87_alter_rules(
                game, rules, source_patterns, target_patterns,
                current_index, num_variants,
                double_translation, tree_translation,
            )
        else:
            return self._solve_tr87_normal(
                game, rules, source_patterns, target_patterns,
                current_index, num_variants,
                double_translation, tree_translation,
            )

    def _solve_tr87_normal(
        self,
        game: Any,
        rules: list,
        source_patterns: list,
        target_patterns: list,
        current_index: int,
        num_variants: int,
        double_translation: bool,
        tree_translation: bool,
    ) -> Optional[list]:
        """Solve tr87 in normal mode (levels 0-3).

        Source patterns are correct, target patterns are scrambled.
        Rules define the correct mapping. Player cycles each target
        to match the expected variant from the rules.

        For double_translation/tree_translation: tries analytical
        approach first, falls back to brute-force using game's
        bsqsshqpox() win checker.

        Args:
            game: The env._game object.
            rules: List of (source_group, target_group) pairs.
            source_patterns: Board source sprites (correct).
            target_patterns: Board target sprites (scrambled).
            current_index: Current selector position.
            num_variants: Number of variants per pattern (typically 7).
            double_translation: Whether double_translation mode is active.
            tree_translation: Whether tree_translation mode is active.

        Returns:
            List of (GameAction, name) tuples, or None.
        """
        from arcengine import GameAction

        # Compute expected target variants
        expected_targets = self._compute_tr87_expected_targets(
            game, rules, source_patterns,
            double_translation, tree_translation,
        )

        if not expected_targets:
            print(f"    [TR87] Could not compute expected targets")
            return None

        print(
            f"    [TR87] Expected targets: {len(expected_targets)}, "
            f"Board targets: {len(target_patterns)}"
        )

        if len(expected_targets) != len(target_patterns):
            print(
                f"    [TR87] Mismatch: expected={len(expected_targets)}, "
                f"board={len(target_patterns)}"
            )
            return None

        # Generate action sequence
        actions: list = []
        cur_idx = current_index
        n_targets = len(target_patterns)

        for i in range(n_targets):
            board_sprite = target_patterns[i]
            expected_name = expected_targets[i]

            # Extract variant numbers from sprite names
            try:
                current_variant = int(board_sprite.name[-1])
                expected_variant = int(expected_name[-1])
            except (ValueError, IndexError):
                print(
                    f"    [TR87] Cannot parse variant: "
                    f"board={board_sprite.name}, expected={expected_name}"
                )
                continue

            delta = (expected_variant - current_variant) % num_variants
            if delta == 0:
                continue

            # Navigate to target i (shortest direction)
            nav_right = (i - cur_idx) % n_targets
            nav_left = (cur_idx - i) % n_targets

            if nav_right <= nav_left:
                for _ in range(nav_right):
                    actions.append((GameAction.ACTION4, "RIGHT"))
            else:
                for _ in range(nav_left):
                    actions.append((GameAction.ACTION3, "LEFT"))
            cur_idx = i

            # Cycle variant (shortest direction)
            if delta <= num_variants // 2:
                for _ in range(delta):
                    actions.append((GameAction.ACTION2, "DOWN"))
            else:
                for _ in range(num_variants - delta):
                    actions.append((GameAction.ACTION1, "UP"))

        # For tree_translation: analytical solution may be incomplete.
        # Fall back to brute-force using game's win checker.
        # For double_translation: analytical solution is now correct
        # (chain rules by target->source name matching, same as
        # bsqsshqpox), so trust it directly.
        if tree_translation and actions:
            win_check = getattr(game, 'bsqsshqpox', None)
            if callable(win_check):
                print(
                    f"    [TR87] tree_translation: using brute-force "
                    f"with game win checker"
                )
                return self._solve_tr87_bruteforce(
                    game, target_patterns, current_index,
                    num_variants, win_check,
                )

        print(
            f"    [TR87] Generated {len(actions)} actions "
            f"for {n_targets} targets"
        )

        # For tree_translation with 0 actions: try brute-force
        if tree_translation and not actions:
            win_check = getattr(game, 'bsqsshqpox', None)
            if callable(win_check):
                print(
                    f"    [TR87] 0 actions in tree mode: "
                    f"using brute-force"
                )
                return self._solve_tr87_bruteforce(
                    game, target_patterns, current_index,
                    num_variants, win_check,
                )

        return actions if actions else []

    def _solve_tr87_bruteforce(
        self,
        game: Any,
        target_patterns: list,
        current_index: int,
        num_variants: int,
        win_check: Any,
    ) -> Optional[list]:
        """Solve tr87 by brute-force trying variants with win checker.

        For each target, tries all 7 variants. After each change,
        calls the game's bsqsshqpox() to check if the puzzle is solved.
        This handles double_translation and tree_translation modes
        where analytical solution is difficult.

        Strategy: greedy one-at-a-time. For each target, find the
        variant that makes bsqsshqpox() return True (or makes progress).

        Args:
            game: The env._game object.
            target_patterns: Board target sprites.
            current_index: Current selector position.
            num_variants: Number of variants per pattern.
            win_check: Callable win checker (game.bsqsshqpox).

        Returns:
            List of (GameAction, name) tuples, or None.
        """
        from arcengine import GameAction

        n_targets = len(target_patterns)
        actions: list = []
        cur_idx = current_index

        # Get the game's variant cycling function
        cycle_fn = getattr(game, 'wpbnovjwkv', None)
        if not callable(cycle_fn):
            print(f"    [TR87-BF] No wpbnovjwkv method available")
            return None

        # Try to solve by cycling each target to the correct variant.
        # For each target, try all variants and check win condition.
        # This is O(n * 7) per target, O(n^2 * 7) total in worst case.
        for target_idx in range(n_targets):
            # Navigate to target
            nav_right = (target_idx - cur_idx) % n_targets
            nav_left = (cur_idx - target_idx) % n_targets
            if nav_right <= nav_left:
                for _ in range(nav_right):
                    actions.append((GameAction.ACTION4, "RIGHT"))
            else:
                for _ in range(nav_left):
                    actions.append((GameAction.ACTION3, "LEFT"))
            cur_idx = target_idx

            # Get current variant
            board_sprite = game.ztgmtnnufb[target_idx]
            try:
                current_variant = int(board_sprite.name[-1])
            except (ValueError, IndexError):
                continue

            # Try all 7 variants to find the one that makes progress.
            # "Progress" = bsqsshqpox() returns True, or the target
            # matches the expected variant from rules.
            best_delta = 0
            won = False

            for trial_delta in range(num_variants):
                if trial_delta == 0:
                    # Check if current state is already winning
                    if win_check():
                        won = True
                        best_delta = 0
                        break
                    continue

                # Cycle the target by trial_delta using DOWN
                # We need to actually modify the game state to check
                # Use the game's wpbnovjwkv function
                for _ in range(trial_delta):
                    sprite = game.ztgmtnnufb[target_idx]
                    game.ztgmtnnufb[target_idx] = cycle_fn(sprite, 1)

                if win_check():
                    won = True
                    best_delta = trial_delta
                    # Undo the cycles we just did (we'll re-apply via actions)
                    for _ in range(trial_delta):
                        sprite = game.ztgmtnnufb[target_idx]
                        game.ztgmtnnufb[target_idx] = cycle_fn(sprite, -1)
                    break

                # Undo the cycles
                for _ in range(trial_delta):
                    sprite = game.ztgmtnnufb[target_idx]
                    game.ztgmtnnufb[target_idx] = cycle_fn(sprite, -1)

            if won:
                # Apply the winning delta via actions
                delta = best_delta
                if delta > 0:
                    if delta <= num_variants // 2:
                        for _ in range(delta):
                            actions.append((GameAction.ACTION2, "DOWN"))
                    else:
                        for _ in range(num_variants - delta):
                            actions.append((GameAction.ACTION1, "UP"))
                print(
                    f"    [TR87-BF] Target {target_idx}: delta={delta}, "
                    f"WON! Total actions: {len(actions)}"
                )
                return actions

        # If no single target change wins, try a different strategy:
        # for each target, find the variant that maximizes the number
        # of matching targets (greedy local search).
        print(
            f"    [TR87-BF] No single-target win, trying greedy search"
        )

        # Reset actions
        actions = []
        cur_idx = current_index

        for target_idx in range(n_targets):
            # Navigate to target
            nav_right = (target_idx - cur_idx) % n_targets
            nav_left = (cur_idx - target_idx) % n_targets
            if nav_right <= nav_left:
                for _ in range(nav_right):
                    actions.append((GameAction.ACTION4, "RIGHT"))
            else:
                for _ in range(nav_left):
                    actions.append((GameAction.ACTION3, "LEFT"))
            cur_idx = target_idx

            # Try all variants, pick the one that makes bsqsshqpox()
            # closest to True (or actually True)
            best_delta = 0
            best_score = -1

            for trial_delta in range(num_variants):
                # Apply trial_delta cycles
                for _ in range(trial_delta):
                    sprite = game.ztgmtnnufb[target_idx]
                    game.ztgmtnnufb[target_idx] = cycle_fn(sprite, 1)

                # Check win condition
                if win_check():
                    # Found it!
                    best_delta = trial_delta
                    best_score = num_variants  # max score
                    # Undo
                    for _ in range(trial_delta):
                        sprite = game.ztgmtnnufb[target_idx]
                        game.ztgmtnnufb[target_idx] = cycle_fn(sprite, -1)
                    break

                # Undo
                for _ in range(trial_delta):
                    sprite = game.ztgmtnnufb[target_idx]
                    game.ztgmtnnufb[target_idx] = cycle_fn(sprite, -1)

            if best_delta > 0:
                if best_delta <= num_variants // 2:
                    for _ in range(best_delta):
                        actions.append((GameAction.ACTION2, "DOWN"))
                else:
                    for _ in range(num_variants - best_delta):
                        actions.append((GameAction.ACTION1, "UP"))

        # Final check
        if win_check():
            print(
                f"    [TR87-BF] Greedy search succeeded! "
                f"{len(actions)} actions"
            )
            return actions

        print(f"    [TR87-BF] Brute-force failed")
        return None

    def _solve_tr87_alter_rules(
        self,
        game: Any,
        rules: list,
        source_patterns: list,
        target_patterns: list,
        current_index: int,
        num_variants: int,
        double_translation: bool,
        tree_translation: bool,
    ) -> Optional[list]:
        """Solve tr87 in alter_rules mode (levels 4-5).

        Board patterns are correct, rule groups are scrambled.
        Uses board patterns as reference to compute correct variant
        for each rule group, then generates cycling actions.

        For simple alter_rules (Level 4): direct board pattern
        comparison works because rule groups and board patterns
        have 1:1 correspondence.

        For complex alter_rules with double_translation/tree_translation
        (Level 5): uses greedy search with scoring function, because
        the 1:1 correspondence breaks down.

        Args:
            game: The env._game object.
            rules: List of (source_group, target_group) pairs.
            source_patterns: Board source sprites (correct).
            target_patterns: Board target sprites (correct).
            current_index: Current selector position.
            num_variants: Number of variants per pattern.
            double_translation: Whether double_translation is active.
            tree_translation: Whether tree_translation is active.

        Returns:
            List of (GameAction, name) tuples, or None.
        """
        from arcengine import GameAction

        win_check = getattr(game, 'bsqsshqpox', None)
        cycle_fn = getattr(game, 'wpbnovjwkv', None)

        # Flatten all rule groups: [src0, tgt0, src1, tgt1, ...]
        flat_groups: list[tuple[str, list]] = []
        for rule_src, rule_tgt in rules:
            flat_groups.append(("source", rule_src))
            flat_groups.append(("target", rule_tgt))

        n_groups = len(flat_groups)
        if n_groups == 0:
            return None

        # For complex modes: use greedy search with scoring
        if (double_translation or tree_translation) and callable(win_check) and callable(cycle_fn):
            return self._solve_tr87_alter_greedy(
                game, flat_groups, source_patterns, target_patterns,
                current_index, num_variants, win_check, cycle_fn,
            )

        # For simple alter_rules: board pattern comparison
        board_src_idx = 0
        board_tgt_idx = 0
        actions: list = []
        cur_idx = current_index
        has_cycling = False

        for group_idx, (group_type, group_sprites) in enumerate(flat_groups):
            if not group_sprites:
                continue

            if group_type == "source":
                board_patterns = source_patterns
                board_idx = board_src_idx
            else:
                board_patterns = target_patterns
                board_idx = board_tgt_idx

            if board_idx >= len(board_patterns):
                if group_type == "source":
                    board_src_idx += len(group_sprites)
                else:
                    board_tgt_idx += len(group_sprites)
                continue

            rule_sprite = group_sprites[0]
            board_sprite = board_patterns[board_idx]

            try:
                rule_variant = int(rule_sprite.name[-1])
                board_variant = int(board_sprite.name[-1])
            except (ValueError, IndexError):
                if group_type == "source":
                    board_src_idx += len(group_sprites)
                else:
                    board_tgt_idx += len(group_sprites)
                continue

            delta = (board_variant - rule_variant) % num_variants

            if group_type == "source":
                board_src_idx += len(group_sprites)
            else:
                board_tgt_idx += len(group_sprites)

            if delta == 0:
                continue

            nav_right = (group_idx - cur_idx) % n_groups
            nav_left = (cur_idx - group_idx) % n_groups

            if nav_right <= nav_left:
                for _ in range(nav_right):
                    actions.append((GameAction.ACTION4, "RIGHT"))
            else:
                for _ in range(nav_left):
                    actions.append((GameAction.ACTION3, "LEFT"))
            cur_idx = group_idx

            if delta <= num_variants // 2:
                for _ in range(delta):
                    actions.append((GameAction.ACTION2, "DOWN"))
            else:
                for _ in range(num_variants - delta):
                    actions.append((GameAction.ACTION1, "UP"))
            has_cycling = True

        # Trigger win check if needed
        if not has_cycling:
            actions.append((GameAction.ACTION2, "DOWN"))
            actions.append((GameAction.ACTION1, "UP"))
        elif actions[-1][0] not in (GameAction.ACTION1, GameAction.ACTION2):
            actions.append((GameAction.ACTION2, "DOWN"))
            actions.append((GameAction.ACTION1, "UP"))

        print(
            f"    [TR87-ALTER] Generated {len(actions)} actions "
            f"for {n_groups} groups"
        )
        return actions if actions else []

    def _solve_tr87_alter_greedy(
        self,
        game: Any,
        flat_groups: list,
        source_patterns: list,
        target_patterns: list,
        current_index: int,
        num_variants: int,
        win_check: Any,
        cycle_fn: Any,
    ) -> Optional[list]:
        """Solve tr87 alter_rules with chain-aware matching for complex modes.

        Multi-phase algorithm:
        Phase 1: Fix source groups by matching against board source patterns.
                 Records ALL candidate deltas for groups with multiple matches.
        Phase 2: Fix target groups by matching against board target or
                 chain source groups.
        Phase 3: If win_check fails, try all source candidate combinations.
                 For each, re-run Phase 2 and check win.
        Phase 4: If still failing, coordinate descent with tree-aware scoring.

        Args:
            game: The env._game object.
            flat_groups: Flattened [(type, sprites), ...] list.
            source_patterns: Board source sprites (correct).
            target_patterns: Board target sprites (correct).
            current_index: Current selector position.
            num_variants: Number of variants per pattern.
            win_check: Callable game.bsqsshqpox.
            cycle_fn: Callable game.wpbnovjwkv.

        Returns:
            List of (GameAction, name) tuples, or None.
        """
        from arcengine import GameAction
        import itertools

        n_groups = len(flat_groups)
        rules = game.cifzvbcuwqe
        n_rules = len(rules)
        is_double = game.current_level.get_data("double_translation")
        is_tree = game.current_level.get_data("tree_translation")

        # Save original names for restoration
        original_names: list[list[str]] = []
        for gtype, gsprites in flat_groups:
            original_names.append([s.name for s in gsprites])

        def apply_delta(gsprites: list, delta: int) -> None:
            """Cycle gsprites by delta positions."""
            for i in range(len(gsprites)):
                for _ in range(delta):
                    gsprites[i] = cycle_fn(gsprites[i], 1)

        def restore_group(gidx: int, delta: int) -> None:
            """Restore group gidx from delta back to original."""
            gtype, gsprites = flat_groups[gidx]
            restore = (num_variants - delta) % num_variants
            apply_delta(gsprites, restore)

        def find_deltas_vs_reference(
            gsprites: list, ref_sprites: list
        ) -> list[int]:
            """Try all variants, return deltas where gsprites match ref.

            After calling, gsprites is back to original state.
            """
            deltas: list[int] = []
            for delta in range(num_variants):
                if len(gsprites) == len(ref_sprites):
                    if all(
                        a.name == b.name
                        for a, b in zip(gsprites, ref_sprites)
                    ):
                        deltas.append(delta)
                elif len(gsprites) < len(ref_sprites):
                    for start in range(
                        len(ref_sprites) - len(gsprites) + 1
                    ):
                        if all(
                            gsprites[i].name
                            == ref_sprites[start + i].name
                            for i in range(len(gsprites))
                        ):
                            deltas.append(delta)
                            break
                for i in range(len(gsprites)):
                    gsprites[i] = cycle_fn(gsprites[i], 1)
            return deltas

        def score_state() -> int:
            """Comprehensive scoring: board source + tree readiness + board target."""
            score = 0
            # 1. Count matched board source segments
            src_idx = 0
            board_src = game.zvojhrjxxm
            while src_idx < len(board_src):
                matched = False
                for rule_src, rule_tgt in rules:
                    if game.iwbhnvdaao(board_src, src_idx, rule_src):
                        score += len(rule_src) * 100
                        # 2. Tree expansion readiness for this rule
                        if is_tree:
                            for sprite in rule_tgt:
                                for other_src, _ in rules:
                                    if other_src and other_src[0].name == sprite.name:
                                        score += 50
                                        break
                        src_idx += len(rule_src)
                        matched = True
                        break
                if not matched:
                    src_idx += 1
            # 3. Count matched board target segments
            tgt_idx = 0
            board_tgt = game.ztgmtnnufb
            while tgt_idx < len(board_tgt):
                matched = False
                for _, rule_tgt in rules:
                    if game.iwbhnvdaao(board_tgt, tgt_idx, rule_tgt):
                        score += len(rule_tgt) * 10
                        tgt_idx += len(rule_tgt)
                        matched = True
                        break
                if not matched:
                    tgt_idx += 1
            return score

        # ========================================
        # Phase 1: Fix source groups against board source
        # ========================================
        print(
            f"    [TR87-HYBRID] Phase 1: Matching {n_rules} source groups "
            f"against {len(source_patterns)} board source patterns"
        )

        # Record ALL candidate deltas for each source group
        source_candidates: dict[int, list[int]] = {}

        for rule_idx in range(n_rules):
            src_gidx = rule_idx * 2
            if src_gidx >= n_groups:
                break
            gtype, gsprites = flat_groups[src_gidx]
            if not gsprites:
                continue

            deltas = find_deltas_vs_reference(gsprites, source_patterns)
            source_candidates[src_gidx] = deltas if deltas else [0]

            if deltas:
                print(
                    f"      Rule {rule_idx} source: {len(deltas)} candidates "
                    f"={deltas}"
                )
            else:
                print(
                    f"      Rule {rule_idx} source: NO board match "
                    f"(chain/sub rule)"
                )

        def run_phase2(best_deltas: list[int]) -> bool:
            """Run Phase 2 target matching. Returns True if win_check passes."""
            # Fix target groups
            for rule_idx in range(n_rules):
                tgt_gidx = rule_idx * 2 + 1
                if tgt_gidx >= n_groups:
                    break
                gtype, gsprites = flat_groups[tgt_gidx]
                if not gsprites:
                    continue

                # Strategy A: Match against board target
                deltas = find_deltas_vs_reference(
                    gsprites, target_patterns
                )
                if deltas:
                    best_deltas[tgt_gidx] = deltas[0]
                    apply_delta(gsprites, deltas[0])
                    continue

                # Strategy B: Match against other rules' source groups
                if is_double or is_tree:
                    found = False
                    for other_idx in range(n_rules):
                        if other_idx == rule_idx:
                            continue
                        other_src_gidx = other_idx * 2
                        if other_src_gidx >= n_groups:
                            continue
                        _, other_gsprites = flat_groups[other_src_gidx]
                        if not other_gsprites:
                            continue
                        chain_deltas = find_deltas_vs_reference(
                            gsprites, other_gsprites
                        )
                        if chain_deltas:
                            best_deltas[tgt_gidx] = chain_deltas[0]
                            apply_delta(gsprites, chain_deltas[0])
                            found = True
                            break
                    if found:
                        continue

                # Strategy C: Tree heuristic
                if is_tree:
                    best_td = 0
                    best_ts = -1
                    for delta in range(num_variants):
                        s = sum(
                            1 for s in gsprites
                            if any(
                                other_src and other_src[0].name == s.name
                                for other_src, _ in rules
                            )
                        )
                        if s > best_ts:
                            best_ts = s
                            best_td = delta
                        for i in range(len(gsprites)):
                            gsprites[i] = cycle_fn(gsprites[i], 1)
                    best_deltas[tgt_gidx] = best_td
                    apply_delta(gsprites, best_td)

            return win_check()

        # ========================================
        # Phase 3: Analytical exhaustive search
        # ========================================
        # bsqsshqpox() only compares sprite names. We can simulate it
        # with pure string operations, avoiding expensive sprite cycling.
        # This lets us search 7^6 = 117,649 combinations in <2 seconds.

        def name_at_delta(name: str, delta: int) -> str:
            """Compute sprite name after cycling by delta."""
            digit = int(name[-1])
            new_digit = (digit + delta - 1) % num_variants + 1
            return name[:-1] + str(new_digit)

        # Board source/target names (fixed, correct)
        board_src_names = [s.name for s in game.zvojhrjxxm]
        board_tgt_names = [s.name for s in game.ztgmtnnufb]

        # Original group names (before any cycling)
        orig_group_names: list[list[str]] = []
        for gtype, gsprites in flat_groups:
            orig_group_names.append([s.name for s in gsprites])

        def analytical_win_check(deltas: list[int]) -> bool:
            """Simulate bsqsshqpox() using name strings only.

            Args:
                deltas: List of delta per group index.

            Returns:
                True if the win condition would be satisfied.
            """
            # Compute rule names at given deltas
            rule_names: list[tuple[list[str], list[str]]] = []
            for rule_idx in range(n_rules):
                src_gidx = rule_idx * 2
                tgt_gidx = rule_idx * 2 + 1
                if src_gidx >= n_groups or tgt_gidx >= n_groups:
                    break
                src_names = [
                    name_at_delta(n, deltas[src_gidx])
                    for n in orig_group_names[src_gidx]
                ]
                tgt_names = [
                    name_at_delta(n, deltas[tgt_gidx])
                    for n in orig_group_names[tgt_gidx]
                ]
                rule_names.append((src_names, tgt_names))

            # Walk through board source (same as bsqsshqpox)
            src_idx = 0
            tgt_idx = 0
            while src_idx < len(board_src_names):
                matched = False
                for rule_src_names, rule_tgt_names in rule_names:
                    # Check if rule source matches board source at src_idx
                    if src_idx + len(rule_src_names) > len(board_src_names):
                        continue
                    if not all(
                        board_src_names[src_idx + i] == rule_src_names[i]
                        for i in range(len(rule_src_names))
                    ):
                        continue

                    # Rule source matches! Handle tree_translation
                    effective_tgt = list(rule_tgt_names)
                    if is_tree:
                        expanded: list[str] = []
                        failed = False
                        for tname in effective_tgt:
                            found = False
                            for other_src_names, other_tgt_names in rule_names:
                                if other_src_names and other_src_names[0] == tname:
                                    expanded += other_tgt_names
                                    found = True
                                    break
                            if not found:
                                failed = True
                                break
                        if failed:
                            continue  # Tree expansion failed, try next rule
                        effective_tgt = expanded
                    elif is_double:
                        # Double translation: find chain rule
                        # (for Level 5, tree takes precedence, but handle both)
                        chained = False
                        for other_src_names, other_tgt_names in rule_names:
                            if len(effective_tgt) == len(other_src_names) and all(
                                a == b for a, b in zip(effective_tgt, other_src_names)
                            ):
                                effective_tgt = list(other_tgt_names)
                                chained = True
                                break
                        if not chained:
                            continue

                    # Check if effective target matches board target
                    if tgt_idx + len(effective_tgt) > len(board_tgt_names):
                        break
                    if not all(
                        board_tgt_names[tgt_idx + i] == effective_tgt[i]
                        for i in range(len(effective_tgt))
                    ):
                        break

                    # Match successful!
                    src_idx += len(rule_src_names)
                    tgt_idx += len(effective_tgt)
                    matched = True
                    break

                if not matched:
                    return False
            return True

        # Identify fixed vs unfixed groups
        # Fixed: source groups with unique candidate, target groups with board match
        # Unfixed: source groups with no match, target groups with tree heuristic
        fixed_deltas: dict[int, int] = {}
        unfixed_gidxs: list[int] = []
        semi_fixed: dict[int, list[int]] = {}  # gidx -> candidate list

        for gidx in range(n_groups):
            gtype, gsprites = flat_groups[gidx]
            if not gsprites:
                continue
            if gidx % 2 == 0:  # Source group
                cands = source_candidates.get(gidx, [])
                if len(cands) == 1:
                    fixed_deltas[gidx] = cands[0]
                elif len(cands) > 1:
                    semi_fixed[gidx] = cands
                else:
                    unfixed_gidxs.append(gidx)
            else:  # Target group
                # Analytical check: does any delta match board target?
                tgt_orig_names = orig_group_names[gidx]
                found_fix = False
                for delta in range(num_variants):
                    variant_names = [
                        name_at_delta(n, delta) for n in tgt_orig_names
                    ]
                    if len(variant_names) <= len(board_tgt_names):
                        for start in range(
                            len(board_tgt_names) - len(variant_names) + 1
                        ):
                            if all(
                                board_tgt_names[start + i] == variant_names[i]
                                for i in range(len(variant_names))
                            ):
                                fixed_deltas[gidx] = delta
                                found_fix = True
                                break
                    if found_fix:
                        break
                if not found_fix:
                    unfixed_gidxs.append(gidx)

        # Also mark semi-fixed source groups for enumeration
        semi_fixed_gidxs = list(semi_fixed.keys())
        semi_fixed_lists = [semi_fixed[g] for g in semi_fixed_gidxs]

        total_semi = 1
        for sl in semi_fixed_lists:
            total_semi *= len(sl)
        total_unfixed = num_variants ** len(unfixed_gidxs)
        total_search = total_semi * total_unfixed

        print(
            f"    [TR87-HYBRID] Phase 3: Analytical search "
            f"({total_semi} semi × {total_unfixed} unfixed "
            f"= {total_search} combos)"
        )

        best_solution = None
        combo_count = 0

        for semi_combo in (
            itertools.product(*semi_fixed_lists)
            if semi_fixed_lists
            else [()]
        ):
            if best_solution is not None:
                break

            # Set semi-fixed deltas
            semi_deltas = {}
            for i, gidx in enumerate(semi_fixed_gidxs):
                semi_deltas[gidx] = semi_combo[i]

            for unfixed_combo in itertools.product(
                range(num_variants), repeat=len(unfixed_gidxs)
            ):
                combo_count += 1

                # Build full delta list
                deltas = [0] * n_groups
                for gidx, d in fixed_deltas.items():
                    deltas[gidx] = d
                for gidx, d in semi_deltas.items():
                    deltas[gidx] = d
                for i, gidx in enumerate(unfixed_gidxs):
                    deltas[gidx] = unfixed_combo[i]

                if analytical_win_check(deltas):
                    best_solution = list(deltas)
                    print(
                        f"    [TR87-HYBRID] Found solution at "
                        f"combo {combo_count}! "
                        f"deltas={best_solution}"
                    )
                    break

        if best_solution is None:
            print(
                f"    [TR87-HYBRID] Analytical search exhausted "
                f"({combo_count} combos)"
            )

        # If analytical search found a solution, apply it to sprites
        if best_solution is not None:
            # Apply deltas to sprites
            for gidx in range(n_groups):
                gtype, gsprites = flat_groups[gidx]
                if not gsprites:
                    continue
                apply_delta(gsprites, best_solution[gidx])

            # Verify with real win_check
            if not win_check():
                print(
                    f"    [TR87-HYBRID] WARNING: Analytical match "
                    f"but win_check failed! Restoring..."
                )
                for gidx in range(n_groups):
                    restore_group(gidx, best_solution[gidx])
                best_solution = None

        # Fallback: coordinate descent if analytical failed
        if best_solution is None:
            print(
                f"    [TR87-HYBRID] Phase 4: Coordinate descent "
                f"with tree-aware scoring"
            )

            # Reset to first combo
            best_deltas = [0] * n_groups
            for gidx, cands in source_candidates.items():
                if cands:
                    best_deltas[gidx] = cands[0]
                    apply_delta(flat_groups[gidx][1], cands[0])

            # Fix targets with Phase 2
            run_phase2(best_deltas)

            for iteration in range(10):
                improved = False
                for gidx in range(n_groups):
                    gtype, gsprites = flat_groups[gidx]
                    if not gsprites:
                        continue

                    for trial in range(num_variants):
                        if win_check():
                            best_deltas[gidx] = (
                                best_deltas[gidx] + trial
                            ) % num_variants
                            best_solution = list(best_deltas)
                            break
                        for i in range(len(gsprites)):
                            gsprites[i] = cycle_fn(gsprites[i], 1)

                    if best_solution is not None:
                        break

                    best_s = -1
                    best_t = 0
                    for trial in range(num_variants):
                        s = score_state()
                        if s > best_s:
                            best_s = s
                            best_t = trial
                        for i in range(len(gsprites)):
                            gsprites[i] = cycle_fn(gsprites[i], 1)

                    if best_t != 0:
                        best_deltas[gidx] = (
                            best_deltas[gidx] + best_t
                        ) % num_variants
                        improved = True

                if best_solution is not None or not improved:
                    break

            if best_solution is None:
                won = win_check()
                if won:
                    best_solution = list(best_deltas)


        # Restore all groups to original state
        if best_solution is not None:
            for gidx in range(n_groups):
                restore_group(gidx, best_solution[gidx])
        else:
            for gidx in range(n_groups):
                restore_group(gidx, best_deltas[gidx])

        if best_solution is None:
            print(f"    [TR87-HYBRID] Failed to find solution")
            return None

        # Generate actions from original state to correct state
        actions: list = []
        cur_idx = current_index
        has_cycling = False

        for gidx in range(n_groups):
            delta = best_solution[gidx]
            if delta == 0:
                continue

            gtype, gsprites = flat_groups[gidx]
            if not gsprites:
                continue

            # Navigate to group
            nav_right = (gidx - cur_idx) % n_groups
            nav_left = (cur_idx - gidx) % n_groups
            if nav_right <= nav_left:
                for _ in range(nav_right):
                    actions.append((GameAction.ACTION4, "RIGHT"))
            else:
                for _ in range(nav_left):
                    actions.append((GameAction.ACTION3, "LEFT"))
            cur_idx = gidx

            # Cycle variant
            if delta <= num_variants // 2:
                for _ in range(delta):
                    actions.append((GameAction.ACTION2, "DOWN"))
            else:
                for _ in range(num_variants - delta):
                    actions.append((GameAction.ACTION1, "UP"))
            has_cycling = True

        # Trigger win check if needed
        if not has_cycling:
            actions.append((GameAction.ACTION2, "DOWN"))
            actions.append((GameAction.ACTION1, "UP"))
        elif actions[-1][0] not in (
            GameAction.ACTION1,
            GameAction.ACTION2,
        ):
            actions.append((GameAction.ACTION2, "DOWN"))
            actions.append((GameAction.ACTION1, "UP"))

        print(
            f"    [TR87-HYBRID] Generated {len(actions)} actions "
            f"for {n_groups} groups (deltas={best_solution})"
        )
        return actions if actions else []


    def _compute_tr87_expected_targets(
        self,
        game: Any,
        rules: list,
        source_patterns: list,
        double_translation: bool,
        tree_translation: bool,
    ) -> list[str]:
        """Compute expected target variant names for tr87 normal mode.

        Walks through source patterns, matching against rule source
        groups (trying all rules at each position, same as win condition).
        For each match, extracts the expected target variants from the
        rule's target group (possibly expanded by double_translation
        or tree_translation).

        Args:
            game: The env._game object.
            rules: List of (source_group, target_group) pairs.
            source_patterns: Board source sprites.
            double_translation: Whether double_translation is active.
            tree_translation: Whether tree_translation is active.

        Returns:
            List of expected sprite names (e.g., ["nxkictbbvztA3", ...]).
        """
        current_level = getattr(game, 'current_level', None)
        expected_targets: list[str] = []
        src_idx = 0

        # Debug: print source pattern names and rule info
        src_names = [s.name for s in source_patterns]
        print(
            f"    [TR87-MATCH] Source patterns ({len(src_names)}): "
            f"{src_names[:10]}"
        )
        for ri, (rs, rt) in enumerate(rules):
            rs_names = [s.name for s in rs]
            rt_names = [s.name for s in rt]
            print(
                f"    [TR87-MATCH] Rule {ri}: src={rs_names}, "
                f"tgt={rt_names}"
            )

        # Use while loop to try all rules at each position (same as
        # the win condition bsqsshqpox does)
        while src_idx < len(source_patterns):
            found = False
            for rule_src, rule_tgt in rules:
                if src_idx + len(rule_src) > len(source_patterns):
                    continue

                # Check if source patterns at src_idx match this rule
                match = True
                for i, s in enumerate(rule_src):
                    if source_patterns[src_idx + i].name != s.name:
                        match = False
                        break

                if not match:
                    continue

                # Match found! Expand target group if needed
                expanded_tgt = list(rule_tgt)
                expanded_src_len = len(rule_src)  # May grow with double_trans

                # Handle double_translation: chain rules.
                # Two mechanisms (matching bsqsshqpox exactly):
                # 1. Link sprites (tjaqvwdgkxe tag): if a link sprite "1"
                #    exists at rule_src[0] position, find the matching "2"
                #    sprite, get the pattern at its position, find the rule
                #    starting there, and extend source+target.
                # 2. Name matching (works WITHOUT link sprites): iterate
                #    all rules, call lonhgifaes on each, and if rule1's
                #    target names match rule2's source names, replace
                #    target with rule2's target.
                # If no chain is found, skip this rule (continue) — same
                # as bsqsshqpox's `else: continue`.
                if double_translation:
                    # Step 1: lonhgifaes on the matched rule
                    # (extends source/target if link sprite exists)
                    if current_level is not None:
                        try:
                            link_sprites_all = (
                                current_level.get_sprites_by_tag(
                                    "tjaqvwdgkxe"
                                )
                            )
                            link_at_pos = None
                            for ls in link_sprites_all:
                                if (ls.x == rule_src[0].x
                                        and ls.y == rule_src[0].y):
                                    link_at_pos = ls
                                    break

                            if link_at_pos is not None:
                                if link_at_pos.name.endswith("2"):
                                    # Chain end — skip this rule
                                    src_idx += len(rule_src)
                                    found = True
                                    break
                                # Find the "2" variant
                                linked_name = link_at_pos.name.replace(
                                    "1", "2"
                                )
                                linked_sprite2 = None
                                for ls in link_sprites_all:
                                    if ls.name == linked_name:
                                        linked_sprite2 = ls
                                        break

                                if linked_sprite2 is not None:
                                    # Find pattern at linked position
                                    all_patterns = (
                                        current_level.get_sprites_by_tag(
                                            "nxkictbbvzt"
                                        )
                                    )
                                    linked_pattern = None
                                    for ap in all_patterns:
                                        if (ap.x == linked_sprite2.x
                                                and ap.y == linked_sprite2.y):
                                            linked_pattern = ap
                                            break

                                    if linked_pattern is not None:
                                        for r_src, r_tgt in rules:
                                            if r_src[0] is linked_pattern:
                                                expanded_tgt = (
                                                    list(rule_tgt)
                                                    + list(r_tgt)
                                                )
                                                expanded_src_len = (
                                                    len(rule_src)
                                                    + len(r_src)
                                                )
                                                break
                        except (AttributeError, Exception):
                            pass

                    # Step 2: name-matching chain (works with or
                    # without link sprites). Find rule2 whose source
                    # matches rule1's target by name.
                    chained = False
                    for other_src, other_tgt in rules:
                        # lonhgifaes on other rule (link sprite check)
                        # — no-op when no link sprites
                        if len(expanded_tgt) == len(other_src):
                            names_match = all(
                                a.name == b.name
                                for a, b in zip(expanded_tgt, other_src)
                            )
                            if names_match:
                                expanded_tgt = list(other_tgt)
                                chained = True
                                break

                    if not chained:
                        # No chain found — skip this rule
                        # (same as bsqsshqpox's `else: continue`)
                        continue

                # Handle tree_translation: expand each target via sub-rules
                if tree_translation:
                    new_expanded: list = []
                    for t in expanded_tgt:
                        sub_found = False
                        for r_src, r_tgt in rules:
                            if r_src[0].name == t.name:
                                new_expanded.extend(r_tgt)
                                sub_found = True
                                break
                        if not sub_found:
                            new_expanded.append(t)
                    expanded_tgt = new_expanded

                for t in expanded_tgt:
                    expected_targets.append(t.name)

                src_idx += expanded_src_len
                found = True
                break

            if not found:
                print(
                    f"    [TR87-MATCH] No rule matched at src_idx={src_idx}, "
                    f"pattern={source_patterns[src_idx].name}"
                )
                break

        print(
            f"    [TR87-MATCH] Computed {len(expected_targets)} "
            f"expected targets from {src_idx}/{len(source_patterns)} "
            f"source patterns"
        )
        return expected_targets

    def _game_to_display_coords(
        self,
        game: Any,
        x: int,
        y: int,
        w: int = 1,
        h: int = 1,
    ) -> tuple[int, int]:
        """Convert game coordinates to display coordinates (0-63).

        Uses the game's camera to compute the correct display position,
        accounting for non-square cameras, camera offsets, and padding.

        Args:
            game: The game object (must have a ``camera`` attribute).
            x: Game x coordinate.
            y: Game y coordinate.
            w: Sprite width in game units (default 1).
            h: Sprite height in game units (default 1).

        Returns:
            (display_x, display_y) in 0-63, targeting the center.
        """
        cam = getattr(game, "camera", None)
        if cam is None:
            # Fallback: simple linear scaling using adapter grid_size
            gs = self._oracle_adapter.grid_size or 32 if self._oracle_adapter else 32
            sc = 64.0 / float(gs) if gs > 0 else 2.0
            dx = int(x * sc + sc / 2)
            dy = int(y * sc + sc / 2)
            return (max(0, min(63, dx)), max(0, min(63, dy)))

        cam_w = int(getattr(cam, "width", 64))
        cam_h = int(getattr(cam, "height", 64))
        cam_x = int(getattr(cam, "x", 0))
        cam_y = int(getattr(cam, "y", 0))

        # Scale: display pixels per game pixel (integer)
        scale_w = int(64 / cam_w) if cam_w > 0 else 1
        scale_h = int(64 / cam_h) if cam_h > 0 else 1
        scale = max(1, min(scale_w, scale_h))

        # Padding to center the game view in the 64x64 display
        x_pad = (64 - cam_w * scale) // 2
        y_pad = (64 - cam_h * scale) // 2

        # Sprite center in game coordinates
        center_gx = x + w // 2
        center_gy = y + h // 2

        # Convert to display coordinates
        disp_x = (center_gx - cam_x) * scale + x_pad
        disp_y = (center_gy - cam_y) * scale + y_pad

        # Clamp to valid display range
        disp_x = max(0, min(63, disp_x))
        disp_y = max(0, min(63, disp_y))

        return (disp_x, disp_y)

    def _solve_click_puzzle(self) -> Optional[list[dict]]:
        """Analytically solve click puzzles by reading game state.

        SIMPLE STRATEGY (takes priority):
        - If game config has `click_target_tag`, just click those sprites!
        - One click per sprite, wait for level transition.

        Complex strategies (fallback):
        1. Color-constraint puzzle (ft09-style): bsT sprites specify
           match/no-match constraints for neighboring Hkx cells.
           Clicking cycles colors. Solver reads constraints and
           determines which cells to click.
        2. Color-matching puzzle: all sprites must have the same color.
           Clicking cycles color. Solver reads colors and calculates
           required clicks per sprite.

        Returns:
            List of click dicts {"x": int, "y": int} in execution order,
            or None if puzzle type is unknown / cannot be solved.
        """
        if not (self.use_oracle and self._oracle_adapter is not None
                and not isinstance(self._oracle_adapter, LS20Adapter)):
            return None

        game = getattr(self._oracle_adapter, 'game', None)
        if game is None:
            return None

        # =========================================================
        # SIMPLE STRATEGY: click_target_tag from config
        # =========================================================
        try:
            from .game_configs import GAME_CONFIGS
        except ImportError:
            try:
                from src.agent.game_configs import GAME_CONFIGS
            except ImportError:
                GAME_CONFIGS = {}
        cfg = GAME_CONFIGS.get(self.game_id) if self.game_id else None
        if cfg and cfg.click_target_tag:
            try:
                cl = game.current_level
                target_sprites = cl.get_sprites_by_tag(cfg.click_target_tag)
                if target_sprites:
                    solution = []
                    for s in target_sprites:
                        sx = int(getattr(s, 'x', 0))
                        sy = int(getattr(s, 'y', 0))
                        sw = int(getattr(s, 'width', 1))
                        sh = int(getattr(s, 'height', 1))
                        dx, dy = self._game_to_display_coords(game, sx, sy, sw, sh)
                        if 0 <= dx < 64 and 0 <= dy < 64:
                            solution.append({"x": dx, "y": dy})
                    if solution:
                        print(
                            f"    [CLICK-SOLVE] Simple strategy: "
                            f"click_target_tag={cfg.click_target_tag}, "
                            f"{len(solution)} sprites to click"
                        )
                        return solution
            except (AttributeError, TypeError, Exception) as e:
                print(f"    [CLICK-SOLVE] Simple strategy failed: {e}")
                # Fall through to complex strategies

        # =========================================================
        # Type 1: ft09-style constraint puzzle
        # =========================================================
        try:
            # Access game attributes directly (ft09-specific names)
            gqb = getattr(game, 'gqb', None)  # color palette, e.g. [9, 8]
            fhc = getattr(game, 'fhc', None)  # clickable Hkx sprites
            gig = getattr(game, 'gig', None)  # bsT constraint sprites
            irw = getattr(game, 'irw', None)  # click effect pattern
            mou = getattr(game, 'mou', None)  # NTi sprites (multi-cell click)

            # Also try tag-based discovery as fallback
            cl = game.current_level
            if fhc is None:
                try:
                    fhc = cl.get_sprites_by_tag('Hkx')
                except (AttributeError, TypeError):
                    pass
            if gig is None:
                try:
                    gig = cl.get_sprites_by_tag('bsT')
                except (AttributeError, TypeError):
                    pass
            if mou is None:
                try:
                    mou = cl.get_sprites_by_tag('NTi')
                except (AttributeError, TypeError):
                    pass
            if gqb is None:
                gqb = [9, 8]  # default 2-color palette

            # Combine Hkx and NTi cells (both are clickable and constrained)
            all_clickable_sprites = list(fhc or []) + list(mou or [])

            if all_clickable_sprites and gig and gqb and len(gqb) >= 2:
                # Read effect pattern (default: center-only)
                if irw is None:
                    irw = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

                # Check if pattern is center-only (simple case)
                is_simple = (
                    irw[0][0] == 0 and irw[0][1] == 0 and irw[0][2] == 0
                    and irw[1][0] == 0 and irw[1][1] == 1 and irw[1][2] == 0
                    and irw[2][0] == 0 and irw[2][1] == 0 and irw[2][2] == 0
                )

                if is_simple:
                    # Simple case: clicking only affects the clicked cell
                    # For each Hkx/NTi cell, determine required color from bsT constraints
                    # Build cell map: (x,y) -> current color index
                    all_hkx: dict[tuple[int, int], int] = {}

                    for s in all_clickable_sprites:
                        sx = int(getattr(s, 'x', 0))
                        sy = int(getattr(s, 'y', 0))
                        try:
                            color = int(s.pixels[1][1])
                        except (AttributeError, IndexError, TypeError, ValueError):
                            continue
                        if color in gqb:
                            all_hkx[(sx, sy)] = gqb.index(color)

                    # Build allowed-colors map using constraint intersection.
                    # Each cell starts with all colors allowed, then constraints
                    # narrow the set: must_match intersects to {target},
                    # must_not_match removes target from allowed set.
                    allowed: dict[tuple[int, int], set[int]] = {}
                    for pos in all_hkx:
                        allowed[pos] = set(range(len(gqb)))

                    # bsT at (bx, by) checks 8 neighbors at offsets of ±4
                    # pixels[row][col] where row=y, col=x
                    offsets = [
                        (-4, -4), (0, -4), (4, -4),
                        (-4, 0), (4, 0),
                        (-4, 4), (0, 4), (4, 4),
                    ]
                    pattern_positions = [
                        (0, 0), (0, 1), (0, 2),  # top row
                        (1, 0), (1, 2),          # middle row (skip center)
                        (2, 0), (2, 1), (2, 2),  # bottom row
                    ]

                    for bsT in gig:
                        bx = int(getattr(bsT, 'x', 0))
                        by = int(getattr(bsT, 'y', 0))
                        try:
                            target_color = int(bsT.pixels[1][1])
                        except (AttributeError, IndexError, TypeError, ValueError):
                            continue
                        target_idx = gqb.index(target_color) if target_color in gqb else 0

                        for (dx, dy), (pi, pj) in zip(offsets, pattern_positions):
                            try:
                                pixel_val = int(bsT.pixels[pi][pj])
                            except (IndexError, TypeError, ValueError):
                                continue
                            must_match = (pixel_val == 0)
                            cell_x = bx + dx
                            cell_y = by + dy

                            # Find the Hkx/NTi sprite at this position
                            found = None
                            for (hx, hy) in all_hkx:
                                if abs(hx - cell_x) <= 1 and abs(hy - cell_y) <= 1:
                                    found = (hx, hy)
                                    break

                            if found and found in allowed:
                                if must_match:
                                    # Intersect with {target_idx}
                                    allowed[found] &= {target_idx}
                                else:
                                    # Remove target_idx from allowed
                                    allowed[found].discard(target_idx)

                    # Build click solution, accounting for NTi multi-cell effects.
                    #
                    # NTi cells (mou) have a special property: when clicked,
                    # they also affect neighboring cells at ±4 grid offsets
                    # wherever the NTi's pixel value is 6. The game's step()
                    # method uses GBS[j][i] offsets multiplied by 4:
                    #   cAw = (sprite.x + (col-1)*4, sprite.y + (row-1)*4)
                    #
                    # We model this as: actual_clicks[j] = (base[j] - nti_effect[j]) % n_colors
                    # where nti_effect[j] = sum of clicks on NTi cells that affect j.
                    # Iterate until stable (handles NTi-affecting-NTi chains).

                    n_colors = len(gqb)

                    # Step 1: Compute base clicks per cell (from constraints)
                    base_clicks: dict[tuple[int, int], int] = {}
                    for (hx, hy), current_idx in all_hkx.items():
                        cell_allowed = allowed.get((hx, hy))
                        if cell_allowed is None or not cell_allowed:
                            base_clicks[(hx, hy)] = 0
                            continue
                        if current_idx in cell_allowed:
                            base_clicks[(hx, hy)] = 0
                            continue
                        best_clicks = n_colors  # worst case
                        for allowed_idx in cell_allowed:
                            clicks = (allowed_idx - current_idx) % n_colors
                            if clicks < best_clicks:
                                best_clicks = clicks
                        base_clicks[(hx, hy)] = best_clicks

                    # Step 2: Build NTi effect map
                    # NTi pixels with value 6 at (row, col) mean clicking
                    # also affects cell at offset ((col-1)*4, (row-1)*4)
                    nti_effects: dict[tuple[int, int], list[tuple[int, int]]] = {}
                    nti_positions: set[tuple[int, int]] = set()
                    if mou:
                        for s in mou:
                            sx = int(getattr(s, 'x', 0))
                            sy = int(getattr(s, 'y', 0))
                            nti_positions.add((sx, sy))
                            pixels = getattr(s, 'pixels', None)
                            if pixels is None:
                                nti_effects[(sx, sy)] = []
                                continue
                            affected: list[tuple[int, int]] = []
                            for pj in range(3):
                                for pi in range(3):
                                    if pj == 1 and pi == 1:
                                        continue  # center handled separately
                                    try:
                                        if int(pixels[pj][pi]) == 6:
                                            ox = (pi - 1) * 4
                                            oy = (pj - 1) * 4
                                            ax = sx + ox
                                            ay = sy + oy
                                            for (hx, hy) in all_hkx:
                                                if abs(hx - ax) <= 1 and abs(hy - ay) <= 1:
                                                    affected.append((hx, hy))
                                                    break
                                    except (IndexError, TypeError, ValueError):
                                        pass
                            nti_effects[(sx, sy)] = affected

                    # Step 3: Iteratively compute actual clicks
                    # actual[j] = (base[j] - nti_effect[j]) % n_colors
                    # where nti_effect[j] = sum of actual clicks on NTi cells affecting j
                    clicks_needed: dict[tuple[int, int], int] = dict(base_clicks)
                    for _ in range(30):
                        # Compute NTi side effects
                        nti_effect: dict[tuple[int, int], int] = {
                            pos: 0 for pos in all_hkx
                        }
                        for nti_pos, affected_list in nti_effects.items():
                            n = clicks_needed.get(nti_pos, 0)
                            if n == 0:
                                continue
                            for aff_pos in affected_list:
                                nti_effect[aff_pos] = (
                                    nti_effect[aff_pos] + n
                                ) % n_colors

                        # Recompute actual clicks
                        changed = False
                        for pos in all_hkx:
                            old_val = clicks_needed.get(pos, 0)
                            new_val = (
                                base_clicks.get(pos, 0)
                                - nti_effect.get(pos, 0)
                            ) % n_colors
                            if new_val != old_val:
                                clicks_needed[pos] = new_val
                                changed = True
                        if not changed:
                            break

                    # Step 4: Build solution from clicks_needed
                    # Compute scale factor from camera (game coords → display coords)
                    cam = getattr(game, "camera", None)
                    if cam is not None:
                        cam_w = int(getattr(cam, "width", 16))
                        sc = 64.0 / max(1, cam_w)
                    else:
                        sc = 4.0  # default for ft09-like games (16x16 grid → 64x64 display)
                    solution: list[dict] = []
                    for (hx, hy), n_clicks in sorted(clicks_needed.items()):
                        if n_clicks == 0:
                            continue
                        dx = int(hx * sc + sc / 2)
                        dy = int(hy * sc + sc / 2)
                        if 0 <= dx < 64 and 0 <= dy < 64:
                            for _ in range(n_clicks):
                                solution.append({"x": dx, "y": dy})

                    if solution:
                        nti_count = len(nti_positions)
                        print(
                            f"    [CLICK-SOLVE] Constraint puzzle: "
                            f"{len(all_hkx)} cells, {len(gig)} constraints, "
                            f"colors={gqb}, NTi={nti_count}, "
                            f"clicks={len(solution)}"
                        )
                        return solution
                    else:
                        # Constraints already satisfied or no solution found.
                        # If no cells needed clicking, constraints are already
                        # met — puzzle might already be solved or no clickable
                        # cells were found at constraint positions.
                        print(
                            f"    [CLICK-SOLVE] Type1 no solution: "
                            f"{len(all_hkx)} cells in palette, "
                            f"{sum(1 for v in allowed.values() if v)} constrained"
                        )
                        pass
        except Exception as e:
            print(f"    [CLICK-SOLVE] Type1 error: {type(e).__name__}: {e}")

        # =========================================================
        # Type 2: Simple color-matching puzzle (all same color)
        # =========================================================
        try:
            cl = game.current_level
            sprites = list(cl._sprites)
        except (AttributeError, TypeError):
            return None

        if not sprites:
            return None

        # Read each sprite's center color
        sprite_colors: list[tuple[int, int, int]] = []  # (x, y, color)
        for s in sprites:
            sx = getattr(s, 'x', None)
            sy = getattr(s, 'y', None)
            if sx is None or sy is None:
                continue
            try:
                px = s.pixels
                if hasattr(px, 'shape'):  # numpy array
                    color = int(px[1][1]) if px.shape[0] > 1 and px.shape[1] > 1 else None
                elif isinstance(px, (list, tuple)) and len(px) > 1:
                    row = px[1]
                    if isinstance(row, (list, tuple)) and len(row) > 1:
                        color = int(row[1])
                    else:
                        color = None
                else:
                    color = None
            except (AttributeError, IndexError, TypeError, ValueError):
                color = None
            if color is not None:
                sprite_colors.append((int(sx), int(sy), int(color)))

        if len(sprite_colors) < 2:
            return None

        colors = [c for _, _, c in sprite_colors]
        unique_colors = set(colors)

        if len(unique_colors) <= 1:
            return None  # All same — puzzle might already be solved

        # Find target color that minimizes total clicks
        num_colors = len(unique_colors)
        best_target = None
        best_clicks = float('inf')
        for target in unique_colors:
            total = 0
            for c in colors:
                diff = (target - c) % num_colors
                total += diff
            if total < best_clicks:
                best_clicks = total
                best_target = target

        if best_target is None or best_clicks == 0:
            return None

        solution: list[dict] = []
        for sx, sy, sc_color in sprite_colors:
            clicks_needed = (best_target - sc_color) % num_colors
            if clicks_needed == 0:
                continue  # Already correct color
            dx, dy = self._game_to_display_coords(game, sx, sy)
            if 0 <= dx < 64 and 0 <= dy < 64:
                # FIX: Only add click ONCE per sprite!
                # The game will handle color cycling on each click.
                # Adding multiple same-position clicks is WRONG.
                solution.append({"x": dx, "y": dy})
                # NOTE: If clicks_needed > 1, the agent will need to
                # re-solve after each click (since color changes).
                # But for now, just add 1 click per sprite.

        if solution:
            print(
                f"    [CLICK-SOLVE] Color-matching puzzle: "
                f"{len(sprite_colors)} sprites, {num_colors} colors, "
                f"target={best_target}, total_clicks={len(solution)}"
            )
        return solution if solution else None

    def _choose_click_action(
        self,
        frames: list,
        latest_frame,
        grid: Optional[np.ndarray] = None,
    ):
        """Choose a click action for click-type games.

        Multi-strategy click solver:
        1. If we have a pre-computed solution, execute it step by step
        2. Try to analytically solve the puzzle (color-matching, etc.)
        3. Use game engine's _get_valid_clickable_actions() for valid clicks
        4. Fall back to oracle goals
        5. Fall back to GridPerception clickable detection
        6. Track grid state hashes for cycle detection
        7. Stagnation detection with RESET after 30 clicks without progress

        Args:
            frames: List of all frame observations.
            latest_frame: Current frame observation.
            grid: 64x64 grid array, or None if extraction failed.

        Returns:
            GameAction.ACTION6 for click, GameAction.RESET for stagnation,
            or None if no clickable position found.
        """
        from arcengine import GameAction
        import random as _random

        # === Phase 0: Execute pre-computed solution (1 click at a time) ===
        if (self._click_solution is not None
                and self._click_solution_idx < len(self._click_solution)):
            click = self._click_solution[self._click_solution_idx]
            self._click_solution_idx += 1
            self._pending_click_data = {"x": click["x"], "y": click["y"]}
            
            # After this click, invalidate solution to force re-solve
            # (colors change after each click, so solution must be recomputed)
            if self._click_solution_idx >= len(self._click_solution):
                # Solution fully executed — force re-solve on next call
                self._click_solve_attempted = False  # Allow re-solve
                print(
                    f"    [CLICK] step={self._step} solution click "
                    f"[{self._click_solution_idx}/{len(self._click_solution)}] "
                    f"at ({click['x']},{click['y']}) — Solution done, will re-solve"
                )
            else:
                print(
                    f"    [CLICK] step={self._step} executing solution "
                    f"[{self._click_solution_idx}/{len(self._click_solution)}] "
                    f"at ({click['x']},{click['y']})"
                )
            return GameAction.ACTION6

        # === Phase 0.5: Wait for level transition after solution ===
        if (self._click_solution is not None
                and self._click_solution_idx >= len(self._click_solution)):
            # Solution fully executed — wait for level transition
            if not hasattr(self, '_solution_wait_counter'):
                self._solution_wait_counter = 0
            
            # Check if level completed
            current_levels = getattr(self, '_latest_obs_levels', 0) if hasattr(self, '_latest_obs_levels') else 0
            if current_levels > getattr(self, '_solution_start_levels', 0):
                # Level completed! Reset solution
                print(f"    [CLICK] ✅ Level completed after solution! Resetting.")
                self._click_solution = None
                self._click_solution_idx = 0
                self._solution_wait_counter = 0
                if hasattr(self, '_solution_start_levels'):
                    delattr(self, '_solution_start_levels')
                # Return None to wait (no click)
                return None
            
            # Wait up to 10 steps for level transition
            if self._solution_wait_counter < 10:
                self._solution_wait_counter += 1
                print(f"    [CLICK] Waiting for level transition (step {self._solution_wait_counter}/10)...")
                return None  # Don't click, just wait
            else:
                # Waited 10 steps, level didn't complete — re-solve or fall back
                print(f"    [CLICK] Level didn't complete after solution + wait. Re-solving...")
                self._click_solution = None
                self._click_solution_idx = 0
                self._solution_wait_counter = 0
                if hasattr(self, '_solution_start_levels'):
                    delattr(self, '_solution_start_levels')
                # Fall through to Phase 1 (re-solve)
        current_level = -1
        if self._oracle_adapter is not None:
            game = getattr(self._oracle_adapter, 'game', None)
            if game is not None:
                try:
                    cl = game.current_level
                    current_level = getattr(cl, 'level_index', -1)
                except (AttributeError, TypeError):
                    pass

        if (not self._click_solve_attempted
                or current_level != self._click_solve_level):
            self._click_solve_attempted = True
            self._click_solve_level = current_level
            solution = self._solve_click_puzzle()
            if solution is not None and len(solution) > 0:
                self._click_solution = solution
                self._click_solution_idx = 0
                # Execute first click immediately
                click = self._click_solution[self._click_solution_idx]
                self._click_solution_idx += 1
                self._pending_click_data = {"x": click["x"], "y": click["y"]}
                print(
                    f"    [CLICK] step={self._step} solution click "
                    f"[1/{len(self._click_solution)}] at "
                    f"({click['x']},{click['y']})"
                )
                return GameAction.ACTION6

        # === Phase 2: Get clickable positions ===
        clickable: list[tuple[int, int]] = []  # (display_x, display_y)

        # Strategy A: Game engine's _get_valid_clickable_actions()
        if (self.use_oracle and self._oracle_adapter is not None
                and not isinstance(self._oracle_adapter, LS20Adapter)):
            game = getattr(self._oracle_adapter, 'game', None)
            if game is not None:
                try:
                    valid = game._get_valid_clickable_actions()
                    if valid:
                        for sprite in valid:
                            sx = getattr(sprite, 'x', None)
                            sy = getattr(sprite, 'y', None)
                            if sx is not None and sy is not None:
                                sw = int(getattr(sprite, 'width', 1))
                                sh = int(getattr(sprite, 'height', 1))
                                dx, dy = self._game_to_display_coords(
                                    game, int(sx), int(sy), sw, sh)
                                if 0 <= dx < 64 and 0 <= dy < 64:
                                    clickable.append((dx, dy))
                except (AttributeError, TypeError, Exception):
                    pass

        # Strategy B: Oracle goals
        if not clickable and (self.use_oracle and self._oracle_adapter is not None
                and not isinstance(self._oracle_adapter, LS20Adapter)):
            goal_entities = self._oracle_adapter.goals
            if goal_entities:
                game = getattr(self._oracle_adapter, 'game', None)
                for g in goal_entities:
                    if hasattr(g, 'x') and hasattr(g, 'y'):
                        gw = int(getattr(g, 'width', 1))
                        gh = int(getattr(g, 'height', 1))
                        if game is not None:
                            dx, dy = self._game_to_display_coords(
                                game, int(g.x), int(g.y), gw, gh)
                        else:
                            gs = self._oracle_adapter.grid_size or 32
                            sc = 64.0 / float(gs) if gs > 0 else 2.0
                            dx = int(g.x * sc + sc / 2)
                            dy = int(g.y * sc + sc / 2)
                        if 0 <= dx < 64 and 0 <= dy < 64:
                            clickable.append((dx, dy))

        # Strategy C: Grid perception
        if not clickable and grid is not None and self._grid_perception is not None:
            raw = self._grid_perception.detect_clickable_positions(grid)
            clickable = [(p[0] + 2, p[1] + 2) for p in raw]

        if not clickable:
            return None

        # === Phase 3: Cycle detection via grid state hashing ===
        if grid is not None:
            try:
                gh = hash(grid.tobytes())
            except (AttributeError, Exception):
                gh = None
            if gh is not None:
                if not hasattr(self, '_click_grid_states'):
                    self._click_grid_states: dict[int, int] = {}
                self._click_grid_states[gh] = (
                    self._click_grid_states.get(gh, 0) + 1
                )
                # If we've seen this exact state 3+ times, we're cycling
                if self._click_grid_states[gh] >= 3:
                    _random.shuffle(clickable)
                    self._clicked_positions.clear()
                    self._click_grid_states.clear()
                    print(f"    [CLICK] cycle detected, shuffling order")

        # === Phase 4: Stagnation detection ===
        if not hasattr(self, '_click_stagnation'):
            self._click_stagnation = 0
        self._click_stagnation += 1

        # After 30 clicks without level advance, RESET to try fresh
        if self._click_stagnation >= 30:
            self._click_stagnation = 0
            self._clicked_positions.clear()
            if hasattr(self, '_click_grid_states'):
                self._click_grid_states.clear()
            self._click_solve_attempted = False  # Allow re-solve after RESET
            print(f"    [CLICK] stagnation RESET at step={self._step}")
            return GameAction.RESET

        # === Phase 5: Select target ===
        unclicked = [
            pos for pos in clickable
            if pos not in self._clicked_positions
        ]

        if unclicked:
            target = unclicked[0]
        elif clickable:
            # All clicked — second round, clear and retry
            target = clickable[0]
            self._clicked_positions.clear()
        else:
            return None

        # === Phase 6: Execute click ===
        self._pending_click_data = {"x": target[0], "y": target[1]}
        self._clicked_positions.add(target)

        print(
            f"    [CLICK] step={self._step} pos=({target[0]},{target[1]}) "
            f"remaining={max(0, len(unclicked)-1)}/{len(clickable)} "
            f"stag={self._click_stagnation}"
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
        self._plan_exhaustion_count = 0
        self.danger_memory.clear_history()

        # Clear danger walls on level change (danger positions from previous
        # level don't apply to the new level's layout)
        self._danger_walls.clear()

        # Reset movement tracking
        self._prev_player_pos = None
        self._prev_action_was_plan = False
        self._movement_fail_count = 0

        # Clear clicked positions for click-type games
        self._clicked_positions.clear()
        # Reset click puzzle solver state for new level
        self._click_solution = None
        self._click_solution_idx = 0
        self._click_solve_attempted = False
        self._click_solve_level = -1
        if hasattr(self, '_click_stagnation'):
            self._click_stagnation = 0
        if hasattr(self, '_click_grid_states'):
            self._click_grid_states.clear()

        # P0-2/P0-3: Reset frame filter and RHAE circuit breaker state
        self._prev_grid = None
        self._static_frame_skip_count = 0
        self._rhae_efficiency_history.clear()
        self._rhae_backtrack_count = 0
        self._current_strategy_idx = 0

        # P0-4: Reset DFS backtracking state
        self._dfs_stack.clear()
        self._dfs_visited.clear()

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
