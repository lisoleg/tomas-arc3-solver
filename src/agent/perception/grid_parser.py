"""
GridParser - Parse 64x64 int8 grids into structured game objects.

LS20-specific player detection:
- Player is a 5x5 sprite: top 2 rows = color 12, bottom 3 rows = color 9.
- Player position = center of the 5x5 area: (x+2, y+2).
- Goals = color 9 pixels NOT inside the player's 5x5 area.

Uses color-aware connected component labeling (scipy.ndimage.label) for
detecting non-player objects (walls, goals, etc.) in the grid.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, List

import numpy as np
from scipy import ndimage


@dataclass
class GameObject:
    """A detected object in the grid.

    Attributes:
        label: Connected component label assigned by scipy.ndimage.label.
        color: Dominant color value (int) of this object.
        bbox: (min_x, min_y, max_x, max_y) bounding box.
        centroid: (cx, cy) center of mass in (x, y) = (col, row) order.
        area: Pixel count of the object.
        is_player: Whether this object is the player.
        is_wall: Whether this object is a wall/obstacle.
        is_goal: Whether this object is a goal/target.
        tags: Optional tags for classification.
    """

    label: int
    color: int
    bbox: tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    centroid: tuple[float, float]  # (cx, cy) = (col, row)
    area: int
    is_player: bool = False
    is_wall: bool = False
    is_goal: bool = False
    is_switcher: bool = False  # Whether this object is a switcher
    switcher_type: str = "unknown"  # "rot", "shape", "color", "unknown"
    tags: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def center(self) -> tuple[int, int]:
        """Integer centroid position (cx, cy) = (col, row)."""
        return (int(self.centroid[0]), int(self.centroid[1]))

    @property
    def position(self) -> tuple[int, int]:
        """Return position as (row, col) for compatibility.

        This is (y, x) order to match numpy array indexing.
        """
        return (int(self.centroid[1]), int(self.centroid[0]))

    @property
    def size(self) -> int:
        """Alias for area, for planner compatibility."""
        return self.area


@dataclass
class ParsedGrid:
    """Structured representation of a parsed grid.

    Wraps the raw grid and all detected objects into a single
    container that the planner can easily consume.

    Attributes:
        grid: The raw 64x64 numpy array.
        objects: All detected GameObjects.
        player_object: The detected player object (or None).
        player_position: Player position as (col, row) (or None).
        goal_objects: List of goal/target objects.
        wall_objects: List of wall/obstacle objects.
        background_color: The detected background color.
        dimensions: (height, width) of the grid.
    """

    grid: np.ndarray
    objects: list[GameObject]
    player_object: Optional[GameObject] = None
    player_position: Optional[tuple[int, int]] = None
    goal_objects: list[GameObject] = field(default_factory=list)
    wall_objects: list[GameObject] = field(default_factory=list)
    switcher_objects: list[GameObject] = field(default_factory=list)
    background_color: int = 0
    dimensions: tuple[int, int] = (64, 64)


class GridParser:
    """Parse a 64x64 int8 grid into structured objects.

    LS20-specific: Detects the player as a 5x5 sprite where the top 2 rows
    are color 12 and the bottom 3 rows are color 9. Goals are color 9
    pixels that fall outside the player's 5x5 area.

    Color-aware connected component labeling is used for detecting other
    objects (walls, goals, triggers) in the grid. The player area is
    excluded from this labeling to prevent the player sprite from being
    misclassified as a goal or wall.
    """

    # Player sprite colors and detection parameters
    PLAYER_TOP_COLOR: int = 12       # Top 2 rows of the 5x5 player sprite
    PLAYER_BOTTOM_COLOR: int = 9     # Bottom 3 rows of the 5x5 player sprite
    PLAYER_SIZE: int = 5             # Player sprite is 5x5 pixels
    PLAYER_MATCH_RATIO: float = 0.6  # Min fraction of matching pixels (15/25)

    # Delta-detection threshold: connected components with area above this
    # value are considered wall/level changes, not player movement.
    WALL_THRESHOLD: int = 800

    # Common background/padding colors seen in ARC-AGI-3
    BACKGROUND_COLORS: set[int] = {0, 4, 5, -2}
    # Wall/obstacle colors (solid blocks)
    WALL_COLORS: set[int] = {3, 4}
    # Goal/trigger colors (bright colors)
    GOAL_COLORS: set[int] = {1, 2, 6, 7, 8, 9}
    # UI/overlay colors that should never be the player or goal
    UI_COLORS: set[int] = {11, 12, 14}
    # 8-connectivity structure for diagonal connections
    _STRUCTURE_8: np.ndarray = np.array(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool
    )

    # Switcher detection parameters
    SWITCHER_SIZE: int = 5        # Switcher sprite is also 5x5 pixels
    SWITCHER_MATCH_RATIO: float = 0.5  # Min fraction of non-bg pixels

    # Player property tracking for switcher type inference
    _prev_player_color: Optional[int] = None
    _prev_player_shape: Optional[str] = None
    _prev_player_orientation: Optional[str] = None
    _current_player_color: Optional[int] = None
    _current_player_shape: Optional[str] = None
    _current_player_orientation: Optional[str] = None
    _switcher_visits: dict[tuple[int, int], dict[str, Any]] = {}
    # switcher_visits[(x, y)] = {
    #   "visits": count,
    #   "color_changes": count,
    #   "shape_changes": count,
    #   "rot_changes": count,
    #   "inferred_type": "rot"/"shape"/"color"/"unknown"
    # }

    def __init__(self) -> None:
        """Initialize the grid parser."""
        self._prev_player_pos: Optional[tuple[int, int]] = None
        self._prev_grid: Optional[np.ndarray] = None
        self._grid_offset_x: Optional[int] = None
        self._grid_offset_y: Optional[int] = None

    def parse(
        self,
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray] = None,
        available_actions: Optional[list[int]] = None,
    ) -> ParsedGrid:
        """Parse a 64x64 grid into a ParsedGrid using grid-aligned sprite extraction.

        The grid is composed of 5x5 sprites aligned to the player's starting
        position. The grid offset is computed as (player_x % 5, player_y % 5)
        and all sprites are extracted at positions (offset_x + i*5, offset_y + j*5).

        Classification:
        - Player: 5x5 sprite with color 12 (top 2 rows) + color 9 (bottom 3 rows)
        - Background: pure bg_color 5x5 block
        - Wall: pure color 3 5x5 block
        - Boundary: pure color 5 5x5 block (also treated as wall)
        - Goal: any non-background, non-wall, non-boundary, non-player sprite

        Args:
            grid: 64x64 int8 numpy array (or list of 2D arrays).
            prev_grid: Previous frame grid (for delta detection fallback).
            available_actions: List of available action IDs (unused).

        Returns:
            ParsedGrid containing all detected objects and metadata.
        """
        # Ensure 2D numpy array
        grid = np.asarray(grid, dtype=np.int8)
        if grid.ndim > 2:
            grid = grid[0]

        # Determine background color (most common color)
        colors, counts = np.unique(grid, return_counts=True)
        color_hist = dict(zip(colors.tolist(), counts.tolist()))
        bg_color: int = max(color_hist, key=color_hist.get) if color_hist else 0

        # --- Step 1: Detect player via 5x5 sprite pattern ---
        player_info: Optional[tuple[int, int, int, int]] = (
            self._detect_player_sprite(grid)
        )

        # Fallback: frame-difference detection if sprite detection fails
        delta_player_pos: Optional[tuple[int, int]] = None
        if player_info is None:
            delta_player_pos = self.detect_player_delta(grid)

        # --- Step 2: Compute grid offset from player position ---
        if player_info is not None:
            px, py = player_info[0], player_info[1]
            self._grid_offset_x = px % self.PLAYER_SIZE
            self._grid_offset_y = py % self.PLAYER_SIZE
        # If player not detected this frame, use stored offset from previous frame

        offset_x: int = self._grid_offset_x if self._grid_offset_x is not None else 0
        offset_y: int = self._grid_offset_y if self._grid_offset_y is not None else 0

        # --- Step 3: Extract all sprites at grid-aligned positions ---
        objects: list[GameObject] = self._extract_sprites_aligned(
            grid, bg_color, offset_x, offset_y, player_info
        )

        # --- Step 4: Create player GameObject ---
        player_obj: Optional[GameObject] = None
        player_pos: Optional[Tuple[int, int]] = None

        if player_info is not None:
            px, py, cx, cy = player_info
            player_pos = (cx, cy)
            player_obj = GameObject(
                label=-1,
                color=self.PLAYER_BOTTOM_COLOR,
                bbox=(px, py, px + self.PLAYER_SIZE - 1, py + self.PLAYER_SIZE - 1),
                centroid=(float(cx), float(cy)),
                area=self.PLAYER_SIZE * self.PLAYER_SIZE,
                is_player=True,
            )
            # Ensure player is first in objects list, remove any duplicate
            objects = [o for o in objects if not o.is_player]
            objects.insert(0, player_obj)
            self._prev_player_pos = (cx, cy)
        elif delta_player_pos is not None:
            cx_d, cy_d = delta_player_pos
            px_d = max(0, cx_d - self.PLAYER_SIZE // 2)
            py_d = max(0, cy_d - self.PLAYER_SIZE // 2)
            player_obj = GameObject(
                label=-1,
                color=self.PLAYER_BOTTOM_COLOR,
                bbox=(
                    px_d, py_d,
                    px_d + self.PLAYER_SIZE - 1,
                    py_d + self.PLAYER_SIZE - 1,
                ),
                centroid=(float(cx_d), float(cy_d)),
                area=self.PLAYER_SIZE * self.PLAYER_SIZE,
                is_player=True,
            )
            objects.insert(0, player_obj)
            player_pos = (cx_d, cy_d)
            self._prev_player_pos = (cx_d, cy_d)
        elif self._prev_player_pos is not None:
            player_pos = self._prev_player_pos

        # --- Step 5: Update player property tracking for switcher detection ---
        if player_pos is not None:
            self._update_player_properties(grid, player_pos)

        # --- Step 6: Add detected switchers from property change tracking ---
        # Only add switchers that weren't already detected by _extract_sprites_aligned
        existing_switcher_positions: set[tuple[int, int]] = {
            (int(o.centroid[0]), int(o.centroid[1]))
            for o in objects if o.is_switcher
        }
        for (sx, sy), info in self._switcher_visits.items():
            if (sx, sy) in existing_switcher_positions:
                continue  # Skip duplicates
            switcher_obj = GameObject(
                label=-2,
                color=0,
                bbox=(sx - 2, sy - 2, sx + 2, sy + 2),
                centroid=(float(sx), float(sy)),
                area=25,
                is_switcher=True,
                switcher_type=info["inferred_type"],
            )
            objects.append(switcher_obj)
            existing_switcher_positions.add((sx, sy))

        # --- Step 7: Build ParsedGrid ---
        goal_objects: list[GameObject] = [o for o in objects if o.is_goal]
        wall_objects: list[GameObject] = [o for o in objects if o.is_wall]
        switcher_objects_final: list[GameObject] = [
            o for o in objects if o.is_switcher
        ]

        result = ParsedGrid(
            grid=grid,
            objects=objects,
            player_object=player_obj,
            player_position=player_pos,
            goal_objects=goal_objects,
            wall_objects=wall_objects,
            switcher_objects=switcher_objects_final,
            background_color=bg_color,
            dimensions=(64, 64),
        )

        # Store current grid for next frame's delta detection
        self._prev_grid = grid.copy()

        return result

    # LS20 sprite color constants (from game source code analysis):
    #   ihdgageizm (wall)    = solid color 4, 5x5, collidable, layer=-5
    #   rjlbuycveu (goal)    = solid color 5, 5x5, NOT collidable, layer=-3
    #   rhsxkxzdjz (rot sw)  = colors 0,1 on transparent bg, 5x5, layer=-1
    #   mkjdaccuuf (shape sw)= color 0 on transparent bg, 5x5, NOT collidable
    #   soyhouuebz (color sw)= colors 9,14,0,8,12 on transparent bg, 5x5
    #   npxgalaybz (refill)  = color 11, 3x3, NOT collidable
    #   nszegiawib (decor)   = color 3, 9x9, NOT a wall (passable by Oracle)
    WALL_COLOR: int = 4          # ihdgageizm sprite
    GOAL_COLOR: int = 5          # rjlbuycveu sprite
    ROT_SW_COLOR: int = 1        # Distinctive color in rhsxkxzdjz
    COLOR_SW_COLORS: set[int] = {8, 14}  # Distinctive colors in soyhouuebz
    REFILL_COLOR: int = 11       # npxgalaybz sprite

    def _extract_sprites_aligned(
        self,
        grid: np.ndarray,
        bg_color: int,
        offset_x: int,
        offset_y: int,
        player_info: Optional[tuple[int, int, int, int]],
    ) -> list[GameObject]:
        """Extract all 5x5 sprites at grid-aligned positions.

        Iterates over the grid in steps of 5 starting from (offset_x, offset_y),
        extracting and classifying each 5x5 sprite.

        Classification rules (in priority order):
        1. Skip player sprite (detected separately).
        2. Skip sprites matching player pattern (color 12 top + color 9 bottom).
        3. Skip pure-background sprites (all pixels == bg_color).
        4. Wall: dominant color is 4 (>80%) → is_wall=True.
           (ihdgageizm sprite = solid color 4)
        5. Goal: dominant color is 5 (>80%) → is_goal=True.
           (rjlbuycveu sprite = solid color 5)
        6. Rotation switcher: contains color 1 → is_switcher=True, type="rot".
           (rhsxkxzdjz sprite has colors 0,1 on transparent bg)
        7. Color switcher: contains color 8 or 14 → is_switcher=True, type="color".
           (soyhouuebz sprite has colors 9,14,0,8,12)
        8. Refill: contains color 11 → potential target (not wall/goal/switcher).
        9. Everything else (color 3 decorative, etc.): skip.

        Args:
            grid: 64x64 int8 numpy array.
            bg_color: Detected background color.
            offset_x: Grid x offset (player_x % 5).
            offset_y: Grid y offset (player_y % 5).
            player_info: (px, py, cx, cy) of detected player, or None.

        Returns:
            List of detected GameObjects (excluding player and background).
        """
        objects: list[GameObject] = []
        height, width = grid.shape
        sz: int = self.PLAYER_SIZE
        label_counter: int = 0

        player_tl: Optional[tuple[int, int]] = None
        if player_info is not None:
            player_tl = (player_info[0], player_info[1])

        for y in range(offset_y, height - sz + 1, sz):
            for x in range(offset_x, width - sz + 1, sz):
                sprite = grid[y:y + sz, x:x + sz]
                if sprite.shape != (sz, sz):
                    continue

                cx: int = x + 2
                cy: int = y + 2
                unique_colors = np.unique(sprite)
                color_set = set(int(c) for c in unique_colors)

                # Skip player sprite (handled separately in parse())
                if player_tl is not None and x == player_tl[0] and y == player_tl[1]:
                    continue

                # Fallback: skip sprites matching the player pattern
                top_match = int(np.sum(sprite[:2, :] == self.PLAYER_TOP_COLOR))
                bottom_match = int(
                    np.sum(sprite[2:, :] == self.PLAYER_BOTTOM_COLOR)
                )
                if top_match + bottom_match >= 15:
                    continue

                # Skip pure-background sprites (all pixels == bg_color)
                if len(color_set) == 1 and bg_color in color_set:
                    continue

                # Compute dominant color and ratio
                color_counts = np.bincount(
                    sprite.flatten().astype(np.int64), minlength=20
                )
                dominant_color = int(np.argmax(color_counts))
                dominant_ratio = float(color_counts[dominant_color]) / float(sz * sz)

                # --- Wall detection: dominant color is 4 (>80%) ---
                # ihdgageizm sprite = solid color 4, collidable, always impassable
                if dominant_color == self.WALL_COLOR and dominant_ratio > 0.8:
                    obj = GameObject(
                        label=label_counter,
                        color=self.WALL_COLOR,
                        bbox=(x, y, x + sz - 1, y + sz - 1),
                        centroid=(float(cx), float(cy)),
                        area=sz * sz,
                        is_wall=True,
                    )
                    objects.append(obj)
                    label_counter += 1
                    continue

                # --- Goal detection: contains color 5 AND 9 ---
                # rjlbuycveu sprite has a color 5 border with color 9 interior.
                # The mix is roughly 50-50, so dominant_ratio check won't work.
                # Must NOT be the player (player has 12+9, already skipped above).
                if (
                    self.GOAL_COLOR in color_set
                    and self.PLAYER_BOTTOM_COLOR in color_set
                ):
                    obj = GameObject(
                        label=label_counter,
                        color=self.GOAL_COLOR,
                        bbox=(x, y, x + sz - 1, y + sz - 1),
                        centroid=(float(cx), float(cy)),
                        area=sz * sz,
                        is_goal=True,
                    )
                    objects.append(obj)
                    label_counter += 1
                    continue

                # --- Rotation switcher: contains color 1 ---
                # rhsxkxzdjz sprite has colors 0,1 on transparent background.
                # Color 1 is the distinctive marker (background is 0).
                if self.ROT_SW_COLOR in color_set:
                    obj = GameObject(
                        label=label_counter,
                        color=self.ROT_SW_COLOR,
                        bbox=(x, y, x + sz - 1, y + sz - 1),
                        centroid=(float(cx), float(cy)),
                        area=sz * sz,
                        is_switcher=True,
                        switcher_type="rot",
                    )
                    objects.append(obj)
                    label_counter += 1
                    continue

                # --- Color switcher: contains color 8 or 14 ---
                # soyhouuebz sprite has colors 9,14,0,8,12 on transparent bg.
                # Colors 8 and 14 are distinctive (not in player or goal).
                if self.COLOR_SW_COLORS & color_set:
                    obj = GameObject(
                        label=label_counter,
                        color=8,
                        bbox=(x, y, x + sz - 1, y + sz - 1),
                        centroid=(float(cx), float(cy)),
                        area=sz * sz,
                        is_switcher=True,
                        switcher_type="color",
                    )
                    objects.append(obj)
                    label_counter += 1
                    continue

                # --- Refill: contains color 11 ---
                # npxgalaybz sprite = color 11, 3x3. Acts as a potential target
                # when the agent is low on steps.
                if self.REFILL_COLOR in color_set:
                    obj = GameObject(
                        label=label_counter,
                        color=self.REFILL_COLOR,
                        bbox=(x, y, x + sz - 1, y + sz - 1),
                        centroid=(float(cx), float(cy)),
                        area=sz * sz,
                    )
                    objects.append(obj)
                    label_counter += 1
                    continue

                # --- Shape switcher: contains color 0 on non-bg background ---
                # mkjdaccuuf sprite = color 0 on transparent (-2) background.
                # In rendered grid, transparent shows as bg color 4 or decor 3.
                # The sprite has color 0 pixels on a mix of 3/4 background.
                # Must NOT match any other rule (not wall, goal, rot, color, refill).
                if 0 in color_set and color_set != {0, bg_color}:
                    obj = GameObject(
                        label=label_counter,
                        color=0,
                        bbox=(x, y, x + sz - 1, y + sz - 1),
                        centroid=(float(cx), float(cy)),
                        area=sz * sz,
                        is_switcher=True,
                        switcher_type="shape",
                    )
                    objects.append(obj)
                    label_counter += 1
                    continue

                # Everything else (color 3 decorative nszegiawib, mixed
                # sprites, etc.): skip.  Dynamic wall learning in the
                # planner handles actual wall collisions at runtime.

        return objects

    def _detect_player_sprite(
        self, grid: np.ndarray
    ) -> Optional[tuple[int, int, int, int]]:
        """Scan the entire grid for the 5x5 player sprite pattern.

        The LS20 player sprite is a 5x5 block where:
        - Top 2 rows (y, y+1) contain color 12.
        - Bottom 3 rows (y+2, y+3, y+4) contain color 9.

        We scan every possible 5x5 window and compute a match score
        (fraction of pixels matching the expected pattern). The window
        with the highest score above PLAYER_MATCH_RATIO is the player.

        Args:
            grid: 64x64 int8 numpy array.

        Returns:
            Tuple (x, y, cx, cy) where (x, y) is the top-left corner
            and (cx, cy) is the center of the 5x5 sprite, or None if
            no match was found.
        """
        height, width = grid.shape
        best_match: Optional[tuple[int, int, int, int]] = None
        best_score: float = 0.0

        for y in range(height - self.PLAYER_SIZE + 1):
            for x in range(width - self.PLAYER_SIZE + 1):
                window = grid[y:y + self.PLAYER_SIZE, x:x + self.PLAYER_SIZE]

                # Top 2 rows should be color 12 (10 pixels)
                top_rows = window[:2, :]
                top_match = int(np.sum(top_rows == self.PLAYER_TOP_COLOR))

                # Bottom 3 rows should be color 9 (15 pixels)
                bottom_rows = window[2:, :]
                bottom_match = int(
                    np.sum(bottom_rows == self.PLAYER_BOTTOM_COLOR)
                )

                total_match = top_match + bottom_match
                score: float = total_match / 25.0

                if score >= self.PLAYER_MATCH_RATIO and score > best_score:
                    best_score = score
                    cx = x + 2  # Center column
                    cy = y + 2  # Center row
                    best_match = (x, y, cx, cy)

        return best_match

    def detect_player_delta(
        self, grid: np.ndarray
    ) -> Optional[tuple[int, int]]:
        """Detect the player position via frame-to-frame difference.

        Computes the pixel-wise difference between the current grid and
        the previous grid, then performs connected-component labeling on
        the difference mask.  Large components (area > ``WALL_THRESHOLD``)
        are treated as wall/level changes and filtered out.  The smallest
        remaining component is assumed to be the player, and its centroid
        is returned.

        This method serves as a fallback when the 5×5 sprite pattern
        matcher fails (e.g., partial occlusion, color shifts).

        Args:
            grid: 64×64 int8 numpy array (current frame).

        Returns:
            ``(cx, cy)`` centroid of the smallest difference component
            in (col, row) order, or ``None`` if there is no previous
            frame or no valid component is found.
        """
        if self._prev_grid is None:
            return None

        # Ensure same shape and dtype for comparison.
        prev = self._prev_grid
        if prev.shape != grid.shape:
            return None

        # Pixel-wise difference mask.
        delta_mask: np.ndarray = grid != prev
        if not delta_mask.any():
            return None

        # Connected-component labeling (8-connectivity).
        labeled, num_features = ndimage.label(
            delta_mask, structure=self._STRUCTURE_8
        )
        if num_features == 0:
            return None

        # Collect component areas and centroids.
        # ndimage.find_objects returns a list of slices, one per label.
        best_centroid: Optional[tuple[int, int]] = None
        best_area: int = -1  # We want the *smallest* non-wall component.

        for label_idx in range(1, num_features + 1):
            component_mask = labeled == label_idx
            area: int = int(component_mask.sum())

            # Filter out large regions (walls, level transitions).
            if area > self.WALL_THRESHOLD:
                continue

            # Skip tiny noise (1-2 pixels).
            if area < 3:
                continue

            # Compute centroid via ndimage.center_of_mass.
            cy_float, cx_float = ndimage.center_of_mass(
                component_mask
            )
            cx: int = int(cx_float)
            cy: int = int(cy_float)

            # Prefer the smallest valid component (player is small).
            if best_area == -1 or area < best_area:
                best_area = area
                best_centroid = (cx, cy)

        return best_centroid

    def _detect_objects(
        self,
        grid: np.ndarray,
        bg_color: int,
        player_mask: np.ndarray,
    ) -> list[GameObject]:
        """Detect non-player objects via color-aware connected component labeling.

        Labels connected components WITHIN each color separately to prevent
        small sprites from being merged with adjacent large walls. The player
        area is excluded via player_mask.

        Classification rules:
        - Walls: color in WALL_COLORS and area > 20.
        - Goals: color == PLAYER_BOTTOM_COLOR (9), or other bright colors
          with area <= 50 and not in UI_COLORS.

        Args:
            grid: 64x64 int8 numpy array.
            bg_color: The detected background color.
            player_mask: Boolean mask of player pixels (excluded from detection).

        Returns:
            List of detected GameObjects (without the player).
        """
        all_bg = self.BACKGROUND_COLORS | {bg_color}
        colors = np.unique(grid)
        non_bg_colors = [int(c) for c in colors if int(c) not in all_bg]

        objects: list[GameObject] = []
        label_offset: int = 0

        for color_val in non_bg_colors:
            # Create mask for this specific color, excluding player area
            color_mask = (grid == color_val) & ~player_mask
            if not color_mask.any():
                continue

            # Label connected components within this color (8-connectivity)
            color_labeled, color_num = ndimage.label(
                color_mask, structure=self._STRUCTURE_8
            )

            for i in range(1, color_num + 1):
                obj_mask = color_labeled == i
                ys, xs = np.where(obj_mask)

                if len(xs) == 0:
                    continue

                min_x, max_x = int(xs.min()), int(xs.max())
                min_y, max_y = int(ys.min()), int(ys.max())
                cx = float(xs.mean())
                cy = float(ys.mean())

                obj = GameObject(
                    label=label_offset + i,
                    color=color_val,
                    bbox=(min_x, min_y, max_x, max_y),
                    centroid=(cx, cy),
                    area=int(obj_mask.sum()),
                )

                # Classify walls: large solid-color blocks
                obj.is_wall = color_val in self.WALL_COLORS and obj.area > 20

                # Classify goals:
                # - Color 9 (PLAYER_BOTTOM_COLOR) outside player area = always goal
                # - Other bright colors = goal if small and not UI
                if color_val == self.PLAYER_BOTTOM_COLOR:
                    obj.is_goal = True
                elif (
                    color_val in self.GOAL_COLORS
                    and color_val not in self.UI_COLORS
                    and obj.area <= 50
                ):
                    obj.is_goal = True

                objects.append(obj)

            label_offset += color_num

        return objects

    def get_player(self, objects: list[GameObject]) -> Optional[GameObject]:
        """Get the player object from a list of detected objects.

        Args:
            objects: List of GameObjects.

        Returns:
            The player GameObject, or None if not found.
        """
        for obj in objects:
            if obj.is_player:
                return obj
        return None

    def get_goals(self, objects: list[GameObject]) -> list[GameObject]:
        """Get all goal objects.

        Args:
            objects: List of GameObjects.

        Returns:
            List of goal GameObjects.
        """
        return [obj for obj in objects if obj.is_goal]

    def get_walls(self, objects: list[GameObject]) -> list[GameObject]:
        """Get all wall/obstacle objects.

        Args:
            objects: List of GameObjects.

        Returns:
            List of wall GameObjects.
        """
        return [obj for obj in objects if obj.is_wall]

    def get_non_background_objects(
        self, objects: list[GameObject]
    ) -> list[GameObject]:
        """Get all non-background, non-wall objects (potential targets).

        Args:
            objects: List of GameObjects.

        Returns:
            List of non-wall GameObjects with area > 2.
        """
        return [obj for obj in objects if not obj.is_wall and obj.area > 2]

    # ------------------------------------------------------------------
    # Switcher detection (TOSAS Theory - Instruction 1: MNQ Gate hardening)
    # ------------------------------------------------------------------

    def _detect_switchers(
        self, grid: np.ndarray
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Detect potential switchers from frame differences.

        This method implements the TOSAS theory requirement:
        "从grid推断切换器/目标/墙壁" (infer switchers/goals/walls from grid)

        Approach:
        1. Compute frame difference (current grid vs previous grid)
        2. Detect regions where pixels changed (potential switcher triggers)
        3. Check if player was at that position (visit correlation)
        4. Infer switcher type from player property changes

        Args:
            grid: 64x64 int8 numpy array (current frame).

        Returns:
            Dictionary mapping (x, y) -> info dict:
                - "visits": number of times player stopped here
                - "color_changes": number of color changes after visit
                - "shape_changes": number of shape changes after visit
                - "rot_changes": number of orientation changes after visit
                - "inferred_type": "rot"/"shape"/"color"/"unknown"
        """
        if self._prev_grid is None or self._prev_player_pos is None:
            return {}

        # This is a simplified implementation
        # Full implementation requires multi-frame tracking
        # For now, return empty dict (will be enhanced in next iteration)

        # TODO: Implement full multi-frame tracking:
        # 1. Track player position history
        # 2. When player stops (same position for 1-2 frames), record visit
        # 3. Check if player properties changed after visit
        # 4. Correlate visits with property changes to infer switcher type

        return self._switcher_visits

    def get_switchers(self) -> dict[tuple[int, int], dict[str, Any]]:
        """Get detected switcher positions and inferred types.

        Returns:
            Dictionary mapping (x, y) -> info dict with inferred type.
        """
        return self._switcher_visits

    def infer_walls(self, grid: np.ndarray, player_pos: tuple[int, int]) -> set[tuple[int, int]]:
        """Infer wall positions from connectivity analysis.

        This method analyzes the grid to find impassable regions:
        1. Perform flood fill from player position
        2. Positions not reached by flood fill are likely walls/obstacles
        3. This implements the "灵体几何" (Lingti Geometry)
           concept from the theory analysis report.

        Args:
            grid: 64x64 int8 numpy array.
            player_pos: (cx, cy) = (col, row) player position.

        Returns:
            Set of (x, y) tuples representing inferred wall positions.
        """
        # Simplified wall inference:
        # Use connected component labeling to find large solid-color regions
        # Regions with color in WALL_COLORS and area > 20 are likely walls

        walls = set()
        bg_color = np.bincount(grid.flatten()).argmax()

        # Label connected components
        labeled, num_features = ndimage.label(grid != bg_color, structure=self._STRUCTURE_8)

        for i in range(1, num_features + 1):
            component_mask = labeled == i
            area = int(component_mask.sum())
            ys, xs = np.where(component_mask)

            if len(xs) == 0:
                continue

            color = int(grid[ys[0], xs[0]])

            # Wall criteria:
            # 1. Color in WALL_COLORS
            # 2. Large area (impassable region)
            # 3. Not the player sprite
            if color in self.WALL_COLORS and area > 20:
                # Add all pixels in this component to walls
                for idx in range(len(xs)):
                    walls.add((int(xs[idx]), int(ys[idx])))

        return walls

    # ------------------------------------------------------------------
    # Switcher detection helper methods
    # ------------------------------------------------------------------

    def _detect_all_sprites(self, grid: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect ALL 5x5 sprites in the grid.

        Scans the entire grid for 5x5 contiguous color regions.
        Returns list of (x, y, cx, cy) where (x, y) is top-left corner
        and (cx, cy) is center.

        Args:
            grid: 64x64 int8 numpy array.

        Returns:
            List of (x, y, cx, cy) tuples for each detected 5x5 sprite.
        """
        height, width = grid.shape
        sprites = []

        # Get background color
        bg_color = np.bincount(grid.flatten()).argmax()

        # Scan grid in steps of 5 (grid cell size)
        for y in range(0, height - self.SWITCHER_SIZE + 1, self.SWITCHER_SIZE):
            for x in range(0, width - self.SWITCHER_SIZE + 1, self.SWITCHER_SIZE):
                window = grid[y:y + self.SWITCHER_SIZE, x:x + self.SWITCHER_SIZE]

                # Check if this window contains a sprite (non-bg pixels)
                non_bg = np.sum(window != bg_color)
                if non_bg >= 10:  # At least 10 non-bg pixels
                    cx = x + 2  # Center column
                    cy = y + 2  # Center row
                    sprites.append((x, y, cx, cy))

        return sprites

    def _correlate_visits_with_changes(
        self, player_pos: tuple[int, int]
    ) -> None:
        """Corelate player visits with property changes to infer switcher type.

        Call this method after updating player properties. If:
        - Player stopped at a position (same as previous frame)
        - Player properties changed
        - Then that position is likely a switcher

        Args:
            player_pos: (cx, cy) = (col, row) current player position.
        """
        if self._prev_player_pos is None:
            return

        # Check if player stopped at this position
        if player_pos == self._prev_player_pos:
            # Player didn't move → might be visiting a switcher
            # Check for property changes
            color_changed = (self._prev_player_color is not None and
                             self._current_player_color != self._prev_player_color)
            shape_changed = (self._prev_player_shape is not None and
                             self._current_player_shape != self._prev_player_shape)
            orientation_changed = (self._prev_player_orientation is not None and
                                   self._current_player_orientation != self._prev_player_orientation)

            if color_changed or shape_changed or orientation_changed:
                # This position likely has a switcher
                if player_pos not in self._switcher_visits:
                    self._switcher_visits[player_pos] = {
                        "visits": 0,
                        "color_changes": 0,
                        "shape_changes": 0,
                        "rot_changes": 0,
                        "inferred_type": "unknown"
                    }

                info = self._switcher_visits[player_pos]
                info["visits"] += 1

                if color_changed:
                    info["color_changes"] += 1
                if shape_changed:
                    info["shape_changes"] += 1
                if orientation_changed:
                    info["rot_changes"] += 1

                # Infer type from dominant change type
                changes = [
                    (info["color_changes"], "color"),
                    (info["shape_changes"], "shape"),
                    (info["rot_changes"], "rot")
                ]
                dominant = max(changes, key=lambda x: x[0])
                if dominant[0] > 0:
                    info["inferred_type"] = dominant[1]

    def _update_player_properties(
        self, grid: np.ndarray, player_pos: tuple[int, int]
    ) -> dict[str, Any]:
        """Extract player properties (color, shape, orientation) from grid.

        This method tracks player property changes across frames to infer
        switcher types. When the player visits a position and their
        properties change, that position is likely a switcher.

        Args:
            grid: 64x64 int8 numpy array.
            player_pos: (cx, cy) = (col, row) player position.

        Returns:
            Dictionary with player properties:
                - "color": dominant color of player sprite
                - "shape": shape descriptor (bbox aspect ratio)
                - "orientation": orientation descriptor (top color pattern)
        """
        px, py = player_pos  # (col, row)
        # Player sprite is 5x5: top-left = (px-2, py-2) to (px+2, py+2)
        # But need to clamp to grid boundaries
        x_min = max(0, px - 2)
        x_max = min(64, px + 3)
        y_min = max(0, py - 2)
        y_max = min(64, py + 3)

        sprite = grid[y_min:y_max, x_min:x_max]

        if sprite.size == 0:
            return {"color": None, "shape": None, "orientation": None}

        # Color: dominant color in sprite
        colors, counts = np.unique(sprite, return_counts=True)
        color = int(colors[np.argmax(counts)])

        # Shape: bounding box aspect ratio
        ys, xs = np.where(sprite > 0)
        if len(xs) == 0:
            shape = "empty"
        else:
            width = xs.max() - xs.min() + 1
            height = ys.max() - ys.min() + 1
            aspect = width / max(height, 1)
            if aspect > 1.5:
                shape = "wide"
            elif aspect < 0.67:
                shape = "tall"
            else:
                shape = "square"

        # Orientation: top row color pattern
        if sprite.shape[0] >= 2:
            top_row = sprite[0, :]
            top_colors = np.unique(top_row)
            if len(top_colors) == 1:
                orientation = f"uniform_{int(top_colors[0])}"
            else:
                orientation = "mixed"
        else:
            orientation = "unknown"

        props = {"color": color, "shape": shape, "orientation": orientation}

        # Store current properties for next frame comparison
        self._current_player_color = color
        self._current_player_shape = shape
        self._current_player_orientation = orientation

        # Correlate with previous properties to detect switcher visits
        self._correlate_visits_with_changes(player_pos)

        # Update previous properties for next frame
        self._prev_player_color = color
        self._prev_player_shape = shape
        self._prev_player_orientation = orientation

        return props
