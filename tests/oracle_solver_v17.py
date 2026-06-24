"""
Oracle solver v17 — moving switcher support via time-dependent BFS.

Key architectural change from v16:
- Detects moving switchers (dboxixicic / wsoslqeku mechanism)
- Time-dependent BFS: state = (position, step_mod_period) to catch moving switchers
- Position timeline simulation replicates dboxixicic.step() movement pattern

Moving switcher mechanism:
- dboxixicic wraps a switcher sprite + bounding box (track)
- Before each player move, wsoslqeku.step() moves the switcher along its track
- If player is blocked (wall), fwtnsrvkrz() undoes the switcher step
- Switcher position = f(successful_player_moves) — deterministic, periodic

Level 0-3: Same as v16 (RHAE=115)
Level 4: Moving rotation switcher (NEW — time-dependent BFS)
"""

import sys
import os
from collections import deque
from itertools import permutations, combinations

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, GameState


# ============================================================================
# Helper functions (same as v16)
# ============================================================================

def snap_to_grid(tx, ty, gx, gy, step):
    sx = gx + step * round((tx - gx) / step)
    sy = gy + step * round((ty - gy) / step)
    return (sx, sy)


def expand_walls(sprites, gx, gy, step, sz=5):
    walls = set()
    for w in sprites:
        wx, wy = w.x, w.y
        base_x = gx + step * ((wx - gx) // step)
        base_y = gy + step * ((wy - gy) // step)
        for ddx in range(-step, step * 2, step):
            for ddy in range(-step, step * 2, step):
                px = base_x + ddx
                py = base_y + ddy
                if abs(px - wx) < sz and abs(py - wy) < sz:
                    walls.add((px, py))
    return walls


def bfs_path_with_teleports(sx, sy, tx, ty, step, walls, teleport_map):
    """Standard BFS with push block teleports (same as v16)."""
    if sx == tx and sy == ty:
        return []

    q = deque([(sx, sy)])
    vis = {(sx, sy)}
    par = {(sx, sy): None}

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


def compute_path_positions(sx, sy, path, step, teleport_map):
    """Compute grid positions along a path, handling teleport jumps."""
    positions = [(sx, sy)]
    cx, cy = sx, sy
    for action, name in path:
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


def find_adjacent(pos, step, walls):
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

    # Direction index -> (dx, dy): 0=down, 1=right, 2=up, 3=left
    DIR_DELTA = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, ms_obj, gx, gy, step):
        """Initialize from game's wsoslqeku entry.

        Args:
            ms_obj: dboxixicic instance from game.wsoslqeku
            gx, gy: grid origin (player start position)
            step: grid step size (5)
        """
        self.bbox = ms_obj.bfdcztirdu  # Bounding box sprite (track)
        self.sprite = ms_obj._sprite   # Switcher sprite
        self.cell = ms_obj._cell       # Cell size (5)
        self.start_x = ms_obj._start_x
        self.start_y = ms_obj._start_y
        self.grid_x = gx
        self.grid_y = gy
        self.step_size = step

        # Determine switcher type from sprite tags
        tags = self.sprite.tags or []
        if "rhsxkxzdjz" in tags:
            self.sw_type = "rot"
        elif "ttfwljgohq" in tags:
            self.sw_type = "shape"
        elif "soyhouuebz" in tags:
            self.sw_type = "color"
        else:
            self.sw_type = "unknown"

        # Precompute position timeline (pixel coordinates)
        self.timeline = self._compute_timeline(max_steps=500)
        self.period = self._find_period()

        # Convert to grid coordinates
        self.grid_timeline = {}
        for s, pos in self.timeline.items():
            self.grid_timeline[s] = snap_to_grid(pos[0], pos[1], gx, gy, step)

        # All distinct grid positions in cycle
        self.grid_positions = list(set(
            self.grid_timeline[i] for i in range(self.period)
        ))

    def _iiosonyanc(self, x, y):
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

    def _compute_timeline(self, max_steps=500):
        """Simulate dboxixicic.step() to generate position timeline.

        Replicates the npdjlrkhsg() direction selection logic:
        1. Try current direction
        2. Try (dir-1)%4, (dir+1)%4, (dir+2)%4
        """
        timeline = {}
        x, y = self.start_x, self.start_y
        direction = 0  # Start going down
        timeline[0] = (x, y)

        for s in range(1, max_steps + 1):
            # Try current direction first
            dx, dy = self.DIR_DELTA[direction]
            new_x = x + dx * self.cell
            new_y = y + dy * self.cell

            if self._iiosonyanc(new_x, new_y):
                # Keep current direction, move
                x, y = new_x, new_y
            else:
                # Try turning: (dir-1)%4, (dir+1)%4, (dir+2)%4
                found = False
                for try_dir in [(direction - 1) % 4, (direction + 1) % 4, (direction + 2) % 4]:
                    dx, dy = self.DIR_DELTA[try_dir]
                    new_x = x + dx * self.cell
                    new_y = y + dy * self.cell
                    if self._iiosonyanc(new_x, new_y):
                        direction = try_dir
                        x, y = new_x, new_y
                        found = True
                        break
                if not found:
                    # Can't move, stay in place
                    pass

            timeline[s] = (x, y)

        return timeline

    def _find_period(self):
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

    def grid_pos_at(self, step_count):
        """Get switcher grid position at a given move count."""
        return self.grid_timeline[step_count % self.period]

    def __repr__(self):
        return (f"MovingSwitcher(type={self.sw_type}, period={self.period}, "
                f"positions={self.grid_positions})")


# ============================================================================
# Time-dependent BFS for moving switchers
# ============================================================================

def bfs_to_moving_switcher(sx, sy, start_move_count, step, walls, tp_map, moving_sw):
    """Time-dependent BFS to catch a moving switcher.

    State: (grid_x, grid_y, move_mod_period)
    The switcher position depends on the total move count (mod period).

    Goal: player position == switcher position at that time step.
    The trigger happens when the player MOVES to the switcher's position
    (after the switcher has stepped).

    Returns: (path, end_move_count) or (None, None)
        path: list of (action, name) tuples
        end_move_count: start_move_count + len(path)
    """
    period = moving_sw.period
    start_mod = start_move_count % period

    q = deque([(sx, sy, start_mod)])
    vis = {(sx, sy, start_mod)}
    par = {(sx, sy, start_mod): None}

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
                # Check if it's a push block (teleport)
                if (nx, ny) in tp_map:
                    dest = tp_map[(nx, ny)]
                    if (dest[0], dest[1], nm) not in vis:
                        vis.add((dest[0], dest[1], nm))
                        par[(dest[0], dest[1], nm)] = ((cx, cy, cm), a, n)

                        # Check goal: player at dest, switcher at timeline[nm]
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

            # Open position
            vis.add((nx, ny, nm))
            par[(nx, ny, nm)] = ((cx, cy, cm), a, n)

            # Check goal: player at (nx, ny), switcher at timeline[nm]
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
# PushTeleport (same as v16)
# ============================================================================

class PushTeleport:
    """Push block as teleport: block_grid -> dest_grid."""
    def __init__(self, pb_obj, gx, gy, step):
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

    def is_usable(self):
        return self.push_dist > 0

    def __repr__(self):
        d = {"(0,1)": "DOWN", "(1,0)": "RIGHT", "(-1,0)": "LEFT", "(0,-1)": "UP"}
        return f"TP({self.block_grid}->{self.dest_grid}, {d.get(str((self.dx,self.dy)),'?')}, d={self.push_dist})"


# ============================================================================
# LevelInfo (modified for moving switchers)
# ============================================================================

class LevelInfo:
    def __init__(self, game):
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
        self.teleports = []
        self.push_block_grids = set()
        self.teleport_map = {}
        for pb in game.hasivfwip:
            tp = PushTeleport(pb, self.grid_x, self.grid_y, self.step)
            if tp.is_usable():
                self.teleports.append(tp)
                self.push_block_grids.add(tp.block_grid)
                self.teleport_map[tp.block_grid] = tp.dest_grid

        self.goals = []
        for i, g in enumerate(game.plrpelhym):
            self.goals.append({
                "sprite": g,
                "req_shape": game.ldxlnycps[i],
                "req_color": game.yjdexjsoa[i],
                "req_rot": game.ehwheiwsk[i],
                "grid_pos": snap_to_grid(g.x, g.y, self.grid_x, self.grid_y, self.step),
            })

        # Detect moving switchers from game.wsoslqeku
        self.moving_switchers = []
        moving_sw_sprite_ids = set()
        if hasattr(game, 'wsoslqeku') and game.wsoslqeku:
            for ms_obj in game.wsoslqeku:
                msw = MovingSwitcher(ms_obj, self.grid_x, self.grid_y, self.step)
                self.moving_switchers.append(msw)
                moving_sw_sprite_ids.add(id(ms_obj._sprite))

        # Map moving switcher types
        self.moving_rot = None
        self.moving_shape = None
        self.moving_color = None
        for msw in self.moving_switchers:
            if msw.sw_type == "rot" and self.moving_rot is None:
                self.moving_rot = msw
            elif msw.sw_type == "shape" and self.moving_shape is None:
                self.moving_shape = msw
            elif msw.sw_type == "color" and self.moving_color is None:
                self.moving_color = msw

        # Static switchers (exclude moving switcher sprites)
        self.rot_sw = [snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
                       for s in game.current_level.get_sprites_by_tag("rhsxkxzdjz")
                       if id(s) not in moving_sw_sprite_ids]
        self.shape_sw = [snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
                         for s in game.current_level.get_sprites_by_tag("ttfwljgohq")
                         if id(s) not in moving_sw_sprite_ids]
        self.color_sw = [snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
                         for s in game.current_level.get_sprites_by_tag("soyhouuebz")
                         if id(s) not in moving_sw_sprite_ids]

        self.refills = [snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
                        for s in game.current_level.get_sprites_by_tag("npxgalaybz")]

        # Build passable set (switchers + refills + ALL moving switcher positions)
        self.passable = set()
        for tag in ("rhsxkxzdjz", "ttfwljgohq", "soyhouuebz", "npxgalaybz"):
            for s in game.current_level.get_sprites_by_tag(tag):
                gp = snap_to_grid(s.x, s.y, self.grid_x, self.grid_y, self.step)
                self.passable.add(gp)
                bx = self.grid_x + self.step * ((gp[0] - self.grid_x) // self.step)
                by = self.grid_y + self.step * ((gp[1] - self.grid_y) // self.step)
                for ddx in range(-self.step, self.step * 2, self.step):
                    for ddy in range(-self.step, self.step * 2, self.step):
                        px2 = bx + ddx
                        py2 = by + ddy
                        if abs(px2 - gp[0]) < self.step and abs(py2 - gp[1]) < self.step:
                            self.passable.add((px2, py2))

        # Add all moving switcher cycle positions to passable
        for msw in self.moving_switchers:
            for pos in msw.grid_positions:
                self.passable.add(pos)
                # Also add expanded positions (like static switchers)
                bx = self.grid_x + self.step * ((pos[0] - self.grid_x) // self.step)
                by = self.grid_y + self.step * ((pos[1] - self.grid_y) // self.step)
                for ddx in range(-self.step, self.step * 2, self.step):
                    for ddy in range(-self.step, self.step * 2, self.step):
                        px2 = bx + ddx
                        py2 = by + ddy
                        if abs(px2 - pos[0]) < self.step and abs(py2 - pos[1]) < self.step:
                            self.passable.add((px2, py2))

    def build_walls(self, rot, shape, color, completed_goals=None):
        """Build walls INCLUDING push block positions.

        Args:
            completed_goals: set of goal indices that are already completed.
                Completed goals' sprites are removed from the game, so they
                should NOT be treated as walls.
        """
        if completed_goals is None:
            completed_goals = set()

        walls = expand_walls(self.wall_sprites, self.grid_x, self.grid_y, self.step)

        for i, g in enumerate(self.goals):
            if i in completed_goals:
                continue  # Completed goal sprite is removed from game
            matched = (shape == g["req_shape"] and
                       color == g["req_color"] and
                       rot == g["req_rot"])
            if not matched:
                walls |= expand_walls([g["sprite"]], self.grid_x, self.grid_y, self.step)

        for p in self.passable:
            walls.discard(p)

        for bg in self.push_block_grids:
            walls.add(bg)

        return walls

    def get_moving_switcher(self, sw_type):
        """Get the MovingSwitcher for a given type, or None."""
        if sw_type == "rot":
            return self.moving_rot
        elif sw_type == "shape":
            return self.moving_shape
        elif sw_type == "color":
            return self.moving_color
        return None

    def generate_waypoint_orderings(self):
        """Generate waypoint orderings, marking moving switchers.

        For single-goal: full permutations of switcher visits.
        For multi-goal: type-order permutations within each goal
        (e.g., rot→shape→color vs shape→rot→color etc.)
        """
        from itertools import product as iter_product

        orderings = []
        c_rot, c_shape, c_color = self.rot, self.shape, self.color

        def make_sw_visit(sw_type):
            """Create a switcher waypoint, marking moving ones."""
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

        def generate_type_perms(rot_diff, shape_diff, color_diff):
            """Generate type-order permutations for switcher visits.

            Instead of full permutations of individual visits (N! explosion),
            we permute the ORDER of types: e.g., all rot visits first,
            then all shape, then all color — vs other type orders.
            """
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
                # Generate type-order permutations for each goal
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

                # Combine type perms across goals
                for combo in iter_product(*goal_type_perms):
                    sw_visits = []
                    for idx, gi in enumerate(goal_perm):
                        sw_visits.extend(combo[idx])
                        sw_visits.append({"type": "goal", "pos": self.goals[gi]["grid_pos"]})
                    orderings.append(sw_visits)

                # Limit explosion
                if len(orderings) > 500:
                    break

        return orderings


# ============================================================================
# Route simulation (modified for moving switchers)
# ============================================================================

def _block_other_switchers(info, walls, target_type=None):
    """Add all switcher positions except target_type to walls.

    This prevents BFS paths from crossing other switchers and triggering
    unintended attribute changes during execution.

    Args:
        info: LevelInfo instance
        walls: set of wall positions (modified in-place)
        target_type: The type of switcher being visited ("rot", "shape", "color").
                     If None, block ALL switcher positions (use for goal/refill navigation).
    Returns:
        walls (modified in-place, also returned for chaining)
    """
    # Static switchers
    if target_type != "rot":
        for pos in info.rot_sw:
            walls.add(pos)
    if target_type != "shape":
        for pos in info.shape_sw:
            walls.add(pos)
    if target_type != "color":
        for pos in info.color_sw:
            walls.add(pos)
    # Moving switchers — block ALL positions in their cycle
    for msw in info.moving_switchers:
        if msw.sw_type != target_type:
            for pos in msw.grid_positions:
                walls.add(pos)
    return walls


def simulate_route(route_items, info, start_pos=None, start_rot=None, start_shape=None,
                   start_color=None, start_budget=None):
    """Simulate a route, tracking move_count for moving switcher timing.

    Key changes from v16:
    - Tracks move_count (total successful player moves)
    - For moving switcher visits, uses time-dependent BFS
    - move_count = len(all_actions) at any point
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

    all_actions = []
    consumed_refills = set()
    refill_set = set(info.refills) if info.refills else set()
    move_count = 0  # Total successful player moves
    completed_goals = set()  # Track completed goals for wall calculation

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
            walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals)
            walls.discard(target)
            _block_other_switchers(info, walls, None)  # Block all switchers
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
                # === Moving switcher: use time-dependent BFS ===
                moving_sw = info.get_moving_switcher(wp_type)
                if moving_sw is None:
                    return None

                walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals)
                # Make all moving switcher positions passable
                for pos in moving_sw.grid_positions:
                    walls.discard(pos)
                # Block other switchers to prevent unintended triggers
                _block_other_switchers(info, walls, wp_type)

                seg, new_move_count = bfs_to_moving_switcher(
                    cx, cy, move_count, step, walls, tp_map, moving_sw
                )

                if seg is None:
                    return None

                # Calculate cost (Level 1+ switchers consume steps)
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

                # Player ends up at the switcher's position at new_move_count
                cx, cy = moving_sw.grid_pos_at(new_move_count)

                # Update attribute
                if wp_type == "rot":
                    cur_rot = (cur_rot + 1) % 4
                elif wp_type == "shape":
                    cur_shape = (cur_shape + 1) % 6
                elif wp_type == "color":
                    cur_color = (cur_color + 1) % 4

            else:
                # === Static switcher: same as v16 ===
                wp_pos = wp["pos"]
                if wp_pos is None:
                    continue

                walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals)
                walls.discard(wp_pos)
                _block_other_switchers(info, walls, wp_type)  # Block other switchers

                if (cx, cy) == wp_pos:
                    adj = find_adjacent(wp_pos, step, walls)
                    if adj is None:
                        return None
                    on_walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals)
                    on_walls.discard(wp_pos)
                    _block_other_switchers(info, on_walls, wp_type)  # Block other switchers
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
            walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals)
            walls.discard(target)
            _block_other_switchers(info, walls, None)  # Block all switchers
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
            # Mark this goal as completed (sprite removed from game)
            for gi, g in enumerate(info.goals):
                if g["grid_pos"] == target:
                    completed_goals.add(gi)
                    break

    return len(all_actions), all_actions


# ============================================================================
# Greedy route solver (fallback for complex multi-switcher levels)
# ============================================================================

def greedy_solve_route(info, max_refills=3):
    """Greedy route solver with type-order permutations.

    Instead of always picking the closest switcher (which can lead to dead-end
    positions), this solver tries all type-order permutations (e.g. rot→shape→color
    vs shape→color→rot) and for each, greedily visits switchers in that type order.
    This explores different final positions, some of which may have a valid path
    to the goal.

    Also tries the "closest first" heuristic as one of the strategies.
    """
    from itertools import permutations as perm

    step = info.step
    step_dec = info.step_dec
    step_max = info.step_max
    switcher_free = info.switcher_free
    tp_map = info.teleport_map
    refill_set = set(info.refills) if info.refills else set()

    best_result = None

    def visit_switcher(sw_type, cx, cy, cur_rot, cur_shape, cur_color, move_count, completed_goals=None):
        """Visit one switcher. Returns (path, end_mc, new_pos, new_rot, new_shape, new_color, cost) or None."""
        walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals)
        moving_sw = info.get_moving_switcher(sw_type)

        if moving_sw is not None:
            for pos in moving_sw.grid_positions:
                walls.discard(pos)
            _block_other_switchers(info, walls, sw_type)  # Block other switchers
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
            _block_other_switchers(info, walls, sw_type)  # Block other switchers
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

        new_rot = cur_rot
        new_shape = cur_shape
        new_color = cur_color
        if sw_type == "rot":
            new_rot = (cur_rot + 1) % 4
        elif sw_type == "shape":
            new_shape = (cur_shape + 1) % 6
        elif sw_type == "color":
            new_color = (cur_color + 1) % 4

        return (path, end_mc, new_pos, new_rot, new_shape, new_color, cost)

    def find_refill(cx, cy, cur_rot, cur_shape, cur_color, budget, consumed, completed_goals=None):
        """Find closest reachable refill. Returns (path, pos, cost) or None."""
        best_rf = None
        best_rf_dist = float('inf')
        best_rf_path = None
        for rf_pos in refill_set - consumed:
            walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals)
            walls.discard(rf_pos)
            _block_other_switchers(info, walls, None)  # Block all switchers
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
        """Try to complete one goal using a specific type visit order.
        Returns (actions, new_pos, new_rot, new_shape, new_color, end_mc, budget, consumed, new_completed) or None.
        """
        goal_pos = goal["grid_pos"]
        all_actions = []

        rot_diff = (goal["req_rot"] - cur_rot) % 4
        shape_diff = (goal["req_shape"] - cur_shape) % 6
        color_diff = (goal["req_color"] - cur_color) % 4

        # Build visit sequence
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
                # Refill
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

                # Retry switcher
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

        # Navigate to goal
        walls = info.build_walls(cur_r, cur_s, cur_c, cg)
        walls.discard(goal_pos)
        _block_other_switchers(info, walls, None)  # Block all switchers
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
            walls = info.build_walls(cur_r, cur_s, cur_c, cg)
            walls.discard(goal_pos)
            _block_other_switchers(info, walls, None)  # Block all switchers
            path = bfs_path_with_teleports(cur_x, cur_y, goal_pos[0], goal_pos[1], step, walls, tp_map)
            if path is None:
                return None
            cost = len(path) * step_dec
            if cost > bud:
                return None

        all_actions.extend(path)
        bud -= cost
        mc += len(path)

        # Mark goal as completed
        cg.add(goal_idx)

        return (all_actions, goal_pos, cur_r, cur_s, cur_c, mc, bud, cr, cg)

    def get_type_perms(goal, rot, shape, color):
        """Get all type-order permutations for a goal."""
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
            return [()]  # No switchers needed
        return list(perm(types))

    def solve_recursive(goal_idx, goal_order, cx, cy, cur_rot, cur_shape, cur_color,
                        move_count, budget, consumed, actions_so_far, completed_goals):
        nonlocal best_result

        if goal_idx >= len(goal_order):
            total = len(actions_so_far)
            if best_result is None or total < best_result[0]:
                best_result = (total, list(actions_so_far))
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

    # Try all goal orders with all type-order permutations
    for goal_order in perm(range(len(info.goals))):
        solve_recursive(
            0, goal_order,
            info.px, info.py, info.rot, info.shape, info.color,
            0, step_max, set(), [], set())

    # Also try "closest first" greedy as a fallback
    if best_result is None:
        for goal_order in perm(range(len(info.goals))):
            cx, cy = info.px, info.py
            cur_rot, cur_shape, cur_color = info.rot, info.shape, info.color
            budget = step_max
            move_count = 0
            consumed = set()
            completed_goals = set()
            all_actions = []
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

                walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals)
                walls.discard(goal_pos)
                _block_other_switchers(info, walls, None)  # Block all switchers
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
                    walls = info.build_walls(cur_rot, cur_shape, cur_color, completed_goals)
                    walls.discard(goal_pos)
                    _block_other_switchers(info, walls, None)  # Block all switchers
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
# Route search (same as v16)
# ============================================================================

def search_routes(info, max_refills=3, time_limit=60):
    """Search for best route. Teleports are automatic.

    Optimization: try refill counts in increasing order. If a valid route
    is found with N refills, still try N+1 for other orderings but skip
    routes that can't possibly beat the current best.
    """
    import time
    start_time = time.time()

    orderings = info.generate_waypoint_orderings()
    refill_options = info.refills if info.refills else []

    best_steps = float('inf')
    best_route = None
    best_actions = None

    for n_rf in range(min(max_refills, len(refill_options)) + 1):
        if n_rf == 0:
            refill_options_to_try = [()]
        else:
            refill_options_to_try = list(permutations(refill_options, n_rf))

        found_at_this_level = False

        for ord_idx, ordering in enumerate(orderings):
            if time.time() - start_time > time_limit:
                print(f"  [search] Time limit reached at n_rf={n_rf}, ord={ord_idx}")
                break

            wp_items = []
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
                    route = []
                    rf_idx = 0
                    for i in range(n_wp + 1):
                        while rf_idx < n_rf and rf_insertions[rf_idx] == i:
                            route.append(("refill", rf_combo[rf_idx]))
                            rf_idx += 1
                        if i < n_wp:
                            route.append(wp_items[i])

                    result = simulate_route(route, info)
                    if result:
                        if result[0] < best_steps:
                            best_steps = result[0]
                            rf_desc = f"{n_rf}rf" if n_rf > 0 else "0rf"
                            best_route = f"ord{ord_idx} {rf_desc}"
                            best_actions = result[1]
                            found_at_this_level = True

            # Early exit: if we already have a great route, stop
            if best_route and best_steps < 50:
                break

        if time.time() - start_time > time_limit:
            break

        # If we found valid routes at this refill level, we can still try
        # higher levels for potentially better routes, but only if time allows
        if found_at_this_level and n_rf >= 1:
            # We have a valid route with n_rf refills. Try n_rf+1 only if
            # there's a chance of significant improvement.
            pass

    if best_route is None:
        print("  [search] Exhaustive search failed, trying greedy solver...")
        g_steps, g_route, g_actions = greedy_solve_route(info, max_refills)
        if g_actions is not None:
            print(f"  [search] Greedy solver found route: {g_steps} steps")
            return g_steps, g_route, g_actions

    return best_steps, best_route, best_actions


# ============================================================================
# Level solver (same as v16, with moving switcher debug output)
# ============================================================================

def solve_level(env, game, level_idx, baseline):
    info = LevelInfo(game)

    print(f"\n=== Level {level_idx} (baseline={baseline}) ===")
    print(f"  Player: ({info.px},{info.py}), Rot={info.rot}, Shape={info.shape}, Color={info.color}")
    print(f"  StepMax: {info.step_max}, Dec: {info.step_dec}, Switcher free: {info.switcher_free}")
    for i, g in enumerate(info.goals):
        rd = (g["req_rot"] - info.rot) % 4
        sd = (g["req_shape"] - info.shape) % 6
        cd = (g["req_color"] - info.color) % 4
        print(f"  Goal {i}: grid={g['grid_pos']}, req_rot={g['req_rot']}(diff={rd}), "
              f"req_shape={g['req_shape']}(diff={sd}), req_color={g['req_color']}(diff={cd})")
    print(f"  Rot SW: {info.rot_sw}, Shape SW: {info.shape_sw}, Color SW: {info.color_sw}")
    print(f"  Refills: {info.refills}")
    print(f"  Teleports: {len(info.teleports)}")
    for tp in info.teleports:
        print(f"    {tp}")

    if info.moving_switchers:
        print(f"  Moving switchers: {len(info.moving_switchers)}")
        for msw in info.moving_switchers:
            print(f"    {msw}")
            print(f"    Timeline (first {min(msw.period * 2, 20)} steps):")
            for s in range(min(msw.period * 2, 20)):
                print(f"      Step {s}: {msw.grid_pos_at(s)}")

    orderings = info.generate_waypoint_orderings()
    print(f"  Waypoint orderings: {len(orderings)}")

    best_steps, best_route, best_actions = search_routes(info, max_refills=3)

    if best_route is None:
        print(f"  FAIL: No feasible plan found")
        return 0, 0

    print(f"  Best plan: {best_route}")
    print(f"  Planned steps: {best_steps}")

    # Debug: print the planned route actions
    if level_idx >= 5:
        print(f"  Planned actions ({len(best_actions)}):")
        for i, (a, n) in enumerate(best_actions):
            print(f"    {i}: {n}")

    # Execute
    actual_steps = 0
    obs = None
    prev_pos = (info.px, info.py)
    prev_rot, prev_shape, prev_color = info.rot, info.shape, info.color

    for i, (action, name) in enumerate(best_actions):
        obs = env.step(action)
        actual_steps += 1
        game = env._game
        cur_pos = (game.gudziatsk.x, game.gudziatsk.y)
        cur_rot = game.cklxociuu
        cur_shape = game.fwckfzsyc
        cur_color = game.hiaauhahz

        dx = abs(cur_pos[0] - prev_pos[0])
        dy = abs(cur_pos[1] - prev_pos[1])
        attr_changed = (cur_rot != prev_rot or cur_shape != prev_shape or cur_color != prev_color)

        if level_idx >= 5:
            if dx > info.step or dy > info.step:
                print(f"    Step {actual_steps}: JUMP {prev_pos} -> {cur_pos} ({name})")
            if attr_changed:
                print(f"    Step {actual_steps}: ATTR CHANGE at {cur_pos} "
                      f"rot={prev_rot}->{cur_rot}, shape={prev_shape}->{cur_shape}, "
                      f"color={prev_color}->{cur_color} ({name})")
            if actual_steps <= 5 or actual_steps % 10 == 0:
                print(f"    Step {actual_steps}: pos={cur_pos}, rot={cur_rot}, "
                      f"shape={cur_shape}, color={cur_color}, "
                      f"steps_left={game._step_counter_ui.current_steps}")

        if obs.levels_completed > level_idx:
            rhae = min(115, ((baseline / max(actual_steps, 1)) ** 2) * 100)
            print(f"  \u2713 Level {level_idx} completed at step {actual_steps}")
            print(f"    Steps: {actual_steps}, Baseline: {baseline}, RHAE: {rhae:.1f}")
            return actual_steps, rhae

        if obs.state == GameState.GAME_OVER:
            print(f"  \u2717 GAME OVER at step {actual_steps}")
            print(f"    Player at ({game.gudziatsk.x},{game.gudziatsk.y}), lives={game.aqygnziho}")
            return 0, 0

        prev_pos = cur_pos
        prev_rot, prev_shape, prev_color = cur_rot, cur_shape, cur_color

    if obs and obs.levels_completed > level_idx:
        rhae = min(115, ((baseline / max(actual_steps, 1)) ** 2) * 100)
        print(f"  \u2713 Level {level_idx} completed at step {actual_steps}")
        return actual_steps, rhae

    print(f"  ? Not completed after {actual_steps} steps")
    print(f"    Player at ({game.gudziatsk.x},{game.gudziatsk.y})")
    print(f"    Rot={game.cklxociuu}, Shape={game.fwckfzsyc}, Color={game.hiaauhahz}")
    print(f"    Lives={game.aqygnziho}, LC={obs.levels_completed if obs else 'N/A'}")
    return 0, 0


# ============================================================================
# Main
# ============================================================================

def run_oracle():
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    baselines = None
    for e in arc.available_environments:
        if e.game_id.startswith("ls20"):
            baselines = e.baseline_actions
            break

    print(f"LS20 baselines: {baselines}, Total: {sum(baselines)}")

    env = arc.make("ls20")
    obs = env.step(GameAction.RESET)
    game = env._game

    total_steps = 0
    total_rhae = 0
    results = []

    for level_idx in range(len(baselines)):
        bl = baselines[level_idx]
        steps, rhae = solve_level(env, game, level_idx, bl)

        if steps > 0:
            total_steps += steps
            total_rhae += rhae
            results.append((level_idx, steps, bl, rhae))
            game = env._game
        else:
            results.append((level_idx, 0, bl, 0))
            break

    print("\n" + "=" * 60)
    print("ORACLE SOLVER V17 RESULTS")
    print("=" * 60)
    print(f"{'Level':<8} {'Steps':<10} {'Baseline':<10} {'RHAE':<10}")
    print("-" * 40)
    completed = 0
    for li, s, b, r in results:
        if s > 0:
            print(f"{li:<8} {s:<10} {b:<10} {r:.1f}")
            completed += 1
        else:
            print(f"{li:<8} {'FAIL':<10} {b:<10} 0.0")
    print("-" * 40)
    print(f"Completed: {completed}/{len(baselines)}")
    print(f"Total steps: {total_steps}, Total baseline: {sum(baselines)}")
    avg_rhae = total_rhae / len(baselines) if results else 0
    overall = min(115, ((sum(baselines) / max(total_steps, 1)) ** 2) * 100) if total_steps > 0 else 0
    print(f"Average RHAE: {avg_rhae:.1f}, Overall RHAE: {overall:.1f}")


if __name__ == "__main__":
    run_oracle()
