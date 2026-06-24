"""
GridPerception — Pure grid-based game state inference for ARC-AGI-3.

Infers game state (player position, walls, goals, switchers) from raw 64x64
frame data WITHOUT accessing env._game internals. This enables the agent to
work on all 25 games, including the 10 that cannot be accessed via env._game,
and in competition (Kaggle) environments where env._game may be restricted.

Key design principles:
1. Frame differencing for player movement detection (O(n) per step)
2. Block-based analysis (5x5 sprites aligned to grid)
3. State bar detection (UI elements at top/bottom of frame)
4. Level transition detection (major grid changes)
5. Clickable position detection for click-based games

Author: TOMAS Team
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================================
# SpriteBlock: A 5x5 sprite block extracted from the grid
# ============================================================================

@dataclass
class SpriteBlock:
    """A 5x5 sprite block extracted from the grid.

    Represents a single grid cell's pixel content, used for
    classifying walls, goals, switchers, and the player.

    Attributes:
        grid_x: Pixel x coordinate of the top-left corner.
        grid_y: Pixel y coordinate of the top-left corner.
        pixels: 5x5 integer array of color values.
        colors: Sorted tuple of unique color values in the block.
        is_multicolor: Whether the block contains more than one color.
    """

    grid_x: int
    grid_y: int
    pixels: np.ndarray
    colors: tuple
    is_multicolor: bool

    @property
    def block_x(self) -> int:
        """Block index in x direction (grid_x // step)."""
        return self.grid_x // 5

    @property
    def block_y(self) -> int:
        """Block index in y direction (grid_y // step)."""
        return self.grid_y // 5


# ============================================================================
# GridPerception: Pure grid-based game state inference
# ============================================================================

class GridPerception:
    """Pure grid-based game state inference.

    Works without env._game access. Infers game state from raw 64x64 frames
    using block analysis and frame differencing.

    The perception maintains:
    - Background and wall color detection
    - Wall block positions (impassable cells)
    - Player position tracking via frame differencing
    - Static block classification (goals, switchers)
    - State bar detection (UI elements at top/bottom)
    - Level transition detection

    Attributes:
        GRID_SIZE: The grid dimension (64x64).
        STEP: Default sprite step size (5 pixels).
        BLOCKS_PER_AXIS: Number of blocks per axis (64 // 5 = 12).
    """

    GRID_SIZE: int = 64
    STEP: int = 5
    BLOCKS_PER_AXIS: int = 64 // 5  # = 12 (with 4 pixels leftover)

    # Threshold for level transition detection (number of changed blocks)
    LEVEL_CHANGE_THRESHOLD: int = 30

    def __init__(self, step: int = 5) -> None:
        """Initialize the grid perception module.

        Args:
            step: Sprite step size in pixels (typically 5).
        """
        self.step: int = step

        # Detected colors
        self.background_color: Optional[int] = None
        self.wall_color: Optional[int] = None

        # Player tracking
        self.player_sprite: Optional[np.ndarray] = None
        self.player_pos: Optional[tuple[int, int]] = None  # pixel (x, y)
        self.player_grid: Optional[tuple[int, int]] = None  # block (bx, by)

        # Frame history
        self.prev_grid: Optional[np.ndarray] = None
        self.frame_count: int = 0

        # Learned state (block coordinates)
        self.wall_blocks: set[tuple[int, int]] = set()
        self.goal_blocks: set[tuple[int, int]] = set()
        self.switcher_blocks: set[tuple[int, int]] = set()
        self.static_blocks: dict[tuple[int, int], np.ndarray] = {}

        # State bar detection
        self.state_bar_top: int = 0  # rows to skip at top
        self.state_bar_bottom: int = 0  # rows to skip at bottom

        # Movement history for player tracking
        self._player_history: list[tuple[int, int]] = []

        # Interactive learning state (persists across levels)
        # position (pixel x, y) -> confidence score
        self._learned_goals: dict[tuple[int, int], int] = {}
        # action -> list of ((block_x, block_y), effect_type)
        self._action_effects: dict[int, list[tuple[tuple[int, int], int]]] = {}
        # Level transition records for cross-level learning
        self._transition_history: list[dict] = []

    # ------------------------------------------------------------------
    # Sprite extraction
    # ------------------------------------------------------------------

    def extract_sprites(self, grid: np.ndarray) -> list[SpriteBlock]:
        """Extract all 5x5 sprite blocks from the grid.

        Scans the grid in step-sized blocks, skipping state bar regions
        at the top and bottom.

        Args:
            grid: 64x64 integer array of color values.

        Returns:
            List of SpriteBlock objects for each grid cell.
        """
        sprites: list[SpriteBlock] = []
        y_start = self.state_bar_top
        y_end = self.GRID_SIZE - self.state_bar_bottom

        for y in range(y_start, y_end, self.step):
            for x in range(0, self.GRID_SIZE, self.step):
                block = grid[y:y + self.step, x:x + self.step]
                if block.shape != (self.step, self.step):
                    continue
                colors = tuple(sorted(np.unique(block).tolist()))
                sprites.append(SpriteBlock(
                    grid_x=x,
                    grid_y=y,
                    pixels=block.copy(),
                    colors=colors,
                    is_multicolor=len(colors) > 1,
                ))
        return sprites

    # ------------------------------------------------------------------
    # Initial frame analysis
    # ------------------------------------------------------------------

    def analyze_initial_frame(
        self,
        grid: np.ndarray,
        available_actions: list[int],
    ) -> None:
        """Analyze the first frame to set up baseline knowledge.

        Detects:
        - Background color (most common color in the grid)
        - Wall color (most common single-color non-background block)
        - State bar regions (UI elements at top/bottom)
        - Initial block classification (walls, static entities)

        Args:
            grid: 64x64 integer array of the initial frame.
            available_actions: List of available action IDs.
        """
        # 1. Background = most common color
        colors, counts = np.unique(grid, return_counts=True)
        self.background_color = int(colors[np.argmax(counts)])

        # 2. Detect state bar regions
        self._detect_state_bar(grid)

        # 3. Extract sprites from playable area
        sprites = self.extract_sprites(grid)

        # 4. Find wall color: most common single-color non-background block
        single_color_blocks: defaultdict[int, int] = defaultdict(int)
        for s in sprites:
            if not s.is_multicolor and s.colors[0] != self.background_color:
                single_color_blocks[s.colors[0]] += 1

        if single_color_blocks:
            self.wall_color = max(
                single_color_blocks, key=single_color_blocks.get
            )

        # 5. Classify blocks
        for s in sprites:
            pos = (s.block_x, s.block_y)
            if s.is_multicolor:
                # Multi-color = potential player/goal/switcher
                # Don't classify yet; will be refined during updates
                self.static_blocks[pos] = s.pixels.copy()
            elif self.wall_color is not None and s.colors[0] == self.wall_color:
                self.wall_blocks.add(pos)
            elif s.colors[0] != self.background_color:
                # Other single-color = potential goal/switcher
                self.static_blocks[pos] = s.pixels.copy()

        # 6. Try to detect player on initial frame
        self._detect_player_initial(grid, sprites)

        self.prev_grid = grid.copy()
        self.frame_count = 1

    def _detect_player_initial(
        self,
        grid: np.ndarray,
        sprites: list[SpriteBlock],
    ) -> None:
        """Detect the player on the initial frame.

        The player is typically a multi-color block that is not at the
        edge of the grid and has a distinctive color pattern.

        Args:
            grid: 64x64 integer array.
            sprites: List of SpriteBlock objects from the initial frame.
        """
        # Heuristic: player is a multi-color block near the center
        # or the first multi-color block found
        center_x = self.GRID_SIZE // 2
        center_y = self.GRID_SIZE // 2

        best_sprite: Optional[SpriteBlock] = None
        best_dist = float("inf")

        for s in sprites:
            if not s.is_multicolor:
                continue
            # Skip blocks at the very edge
            if s.grid_x <= 0 or s.grid_x >= self.GRID_SIZE - self.step:
                continue
            if s.grid_y <= self.state_bar_top or s.grid_y >= self.GRID_SIZE - self.step - self.state_bar_bottom:
                continue

            # Prefer blocks closer to center
            cx = s.grid_x + self.step // 2
            cy = s.grid_y + self.step // 2
            dist = abs(cx - center_x) + abs(cy - center_y)
            if dist < best_dist:
                best_dist = dist
                best_sprite = s

        if best_sprite is not None:
            self.player_sprite = best_sprite.pixels.copy()
            self.player_pos = (best_sprite.grid_x, best_sprite.grid_y)
            self.player_grid = (best_sprite.block_x, best_sprite.block_y)
            self._player_history.append(self.player_grid)

    # ------------------------------------------------------------------
    # State bar detection
    # ------------------------------------------------------------------

    def _detect_state_bar(self, grid: np.ndarray) -> None:
        """Detect state bar regions (UI elements at top/bottom).

        State bars often have distinct colors or patterns that differ
        from the main playable area. This method checks if the top
        and/or bottom rows have colors not present in the middle section.

        Args:
            grid: 64x64 integer array.
        """
        # Sample regions: top 5 rows, middle 10 rows, bottom 5 rows
        top_colors = set(np.unique(grid[:5, :]).tolist())
        bottom_colors = set(np.unique(grid[-5:, :]).tolist())
        mid_colors = set(np.unique(grid[25:35, :]).tolist())

        # If top has colors not in middle, it's likely a state bar
        top_unique = top_colors - mid_colors
        bot_unique = bottom_colors - mid_colors

        if top_unique and len(top_colors) < 8:
            self.state_bar_top = 5
        if bot_unique and len(bottom_colors) < 8:
            self.state_bar_bottom = 5

    # ------------------------------------------------------------------
    # Frame update (called every step)
    # ------------------------------------------------------------------

    def update(
        self,
        grid: np.ndarray,
        action_taken: Optional[int] = None,
    ) -> dict:
        """Update perception with a new frame.

        This method is called every step and must be efficient.
        It uses frame differencing to detect changes and track the player.

        Args:
            grid: 64x64 integer array of the current frame.
            action_taken: The action that was taken (for movement tracking).

        Returns:
            Dictionary with detected changes:
                - player_moved: bool
                - old_pos: tuple or None
                - new_pos: tuple or None
                - blocks_changed: list of (pos, old_pixels, new_pixels)
                - level_completed: bool
        """
        changes: dict = {
            "player_moved": False,
            "old_pos": None,
            "new_pos": None,
            "blocks_changed": [],
            "level_completed": False,
        }

        # First frame: initialize
        if self.prev_grid is None:
            self.analyze_initial_frame(grid, [])
            return changes

        # Frame differencing (vectorized for efficiency)
        diff = grid != self.prev_grid
        if not diff.any():
            self.prev_grid = grid.copy()
            self.frame_count += 1
            return changes

        changed_rows, changed_cols = np.where(diff)

        # Find changed blocks
        changed_block_positions: set[tuple[int, int]] = set()
        for r, c in zip(changed_rows, changed_cols):
            bx = c // self.step
            by = r // self.step
            changed_block_positions.add((bx, by))

        # Major change = level transition
        if len(changed_block_positions) > self.LEVEL_CHANGE_THRESHOLD:
            changes["level_completed"] = True
            self._reinitialize_for_new_level(grid)
            return changes

        # Find player: the multi-color block that changed position
        old_player_pos = self.player_grid
        new_player_pos: Optional[tuple[int, int]] = None

        for bx, by in changed_block_positions:
            x = bx * self.step
            y = by * self.step

            # Skip if block is outside playable area
            if y < self.state_bar_top or y >= self.GRID_SIZE - self.state_bar_bottom:
                continue

            block = grid[y:y + self.step, x:x + self.step]
            if block.shape != (self.step, self.step):
                continue

            colors = np.unique(block)
            if len(colors) > 1:
                # Multi-color = player or entity
                if old_player_pos is None or (bx, by) != old_player_pos:
                    # Check if this could be the player's new position
                    # (player moves one block at a time)
                    if old_player_pos is not None:
                        dist = abs(bx - old_player_pos[0]) + abs(by - old_player_pos[1])
                        if dist <= 2:  # Allow diagonal-ish movement
                            new_player_pos = (bx, by)
                            self.player_sprite = block.copy()
                            break
                    else:
                        new_player_pos = (bx, by)
                        self.player_sprite = block.copy()
                        break

        # Also check if old player position changed (player moved away)
        if new_player_pos is None and old_player_pos is not None:
            ox, oy = old_player_pos
            x, y = ox * self.step, oy * self.step
            if 0 <= y < self.GRID_SIZE and 0 <= x < self.GRID_SIZE:
                old_block = grid[y:y + self.step, x:x + self.step]
                if old_block.shape == (self.step, self.step):
                    old_colors = np.unique(old_block)
                    # If old position is now single-color (background/wall),
                    # player moved away — find where
                    if len(old_colors) <= 1:
                        for bx, by in changed_block_positions:
                            if (bx, by) == old_player_pos:
                                continue
                            x2, y2 = bx * self.step, by * self.step
                            block = grid[y2:y2 + self.step, x2:x2 + self.step]
                            if block.shape != (self.step, self.step):
                                continue
                            colors = np.unique(block)
                            if len(colors) > 1:
                                new_player_pos = (bx, by)
                                self.player_sprite = block.copy()
                                break

        if new_player_pos is not None and new_player_pos != old_player_pos:
            changes["player_moved"] = True
            changes["old_pos"] = old_player_pos
            changes["new_pos"] = new_player_pos
            self.player_grid = new_player_pos
            self.player_pos = (
                new_player_pos[0] * self.step,
                new_player_pos[1] * self.step,
            )
            self._player_history.append(self.player_grid)

        # Track changed blocks for mechanic learning
        for bx, by in changed_block_positions:
            x = bx * self.step
            y = by * self.step
            if y < self.state_bar_top or y >= self.GRID_SIZE - self.state_bar_bottom:
                continue
            old_block = self.prev_grid[y:y + self.step, x:x + self.step]
            new_block = grid[y:y + self.step, x:x + self.step]
            if (old_block.shape == (self.step, self.step)
                    and new_block.shape == (self.step, self.step)):
                changes["blocks_changed"].append((
                    (bx, by),
                    old_block.copy(),
                    new_block.copy(),
                ))

        # Update static blocks (remove changed ones)
        for pos, _, _ in changes["blocks_changed"]:
            self.static_blocks.pop(pos, None)

        # === Interactive learning: detect potential goals from block changes ===
        # If a block goes from multi-color to single-color (background),
        # it may be a completed goal. If player moved to a static_block
        # position and that block disappeared, mark as goal.
        for pos, old_pixels, new_pixels in changes["blocks_changed"]:
            old_colors = set(np.unique(old_pixels).tolist())
            new_colors = set(np.unique(new_pixels).tolist())

            # Block disappeared (became background) — potential goal
            if (self.background_color is not None
                    and new_colors == {self.background_color}
                    and old_colors != {self.background_color}
                    and len(old_colors) > 1):
                # Multi-color block became background = likely goal
                pixel_pos = (pos[0] * self.step, pos[1] * self.step)
                # Only mark as goal if player was nearby
                if self.player_grid is not None:
                    dist = (abs(pos[0] - self.player_grid[0])
                            + abs(pos[1] - self.player_grid[1]))
                    if dist <= 2:
                        confidence = max(1, 3 - dist)
                        self._learned_goals[pixel_pos] = (
                            self._learned_goals.get(pixel_pos, 0) + confidence
                        )

            # Player moved to a static block position and it changed
            if (changes["player_moved"]
                    and self.player_grid is not None
                    and pos == self.player_grid
                    and old_colors != new_colors):
                pixel_pos = (pos[0] * self.step, pos[1] * self.step)
                self._learned_goals[pixel_pos] = (
                    self._learned_goals.get(pixel_pos, 0) + 1
                )

        # Learn from action effects (action-impact tracking)
        if action_taken is not None and self.prev_grid is not None:
            self.learn_from_action_effect(action_taken, self.prev_grid, grid)

        self.prev_grid = grid.copy()
        self.frame_count += 1
        return changes

    def _reinitialize_for_new_level(self, grid: np.ndarray) -> None:
        """Reinitialize perception for a new level, preserving learned knowledge.

        Saves the old grid for transition learning, then clears all
        per-level state (walls, goals, switchers, player position) while
        keeping cross-level learned knowledge (_learned_goals,
        _action_effects, _transition_history). Finally re-analyzes
        the new level's initial frame.

        Args:
            grid: 64x64 integer array of the new level's initial frame.
        """
        # Save old grid for transition learning
        old_grid = self.prev_grid.copy() if self.prev_grid is not None else None

        # Clear per-level state but keep learned goals and action effects
        self.wall_blocks.clear()
        self.goal_blocks.clear()
        self.switcher_blocks.clear()
        self.static_blocks.clear()
        self.player_grid = None
        self.player_pos = None
        self.player_sprite = None
        self._player_history.clear()

        # Learn from transition if we have old grid
        if old_grid is not None:
            self.learn_from_level_transition(old_grid, grid, [])

        self.analyze_initial_frame(grid, [])

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_wall_positions(self) -> set[tuple[int, int]]:
        """Get wall positions as pixel coordinates (x, y).

        Returns:
            Set of (x, y) pixel positions where walls are located.
        """
        return {
            (bx * self.step, by * self.step)
            for bx, by in self.wall_blocks
        }

    def get_player_grid_pos(self) -> Optional[tuple[int, int]]:
        """Get player position in block coordinates.

        Returns:
            (bx, by) block position, or None if player not detected.
        """
        return self.player_grid

    def get_player_pixel_pos(self) -> Optional[tuple[int, int]]:
        """Get player position in pixel coordinates.

        Returns:
            (x, y) pixel position, or None if player not detected.
        """
        return self.player_pos

    def compute_state_hash(self, grid: np.ndarray) -> str:
        """Compute a hash of the grid state for cycle detection.

        Hashes only the playable area (skipping state bars) for
        efficient cycle detection during exploration.

        Args:
            grid: 64x64 integer array.

        Returns:
            String hash of the grid state.
        """
        y_end = self.GRID_SIZE - self.state_bar_bottom
        if self.state_bar_bottom == 0:
            playable = grid[self.state_bar_top:, :]
        else:
            playable = grid[self.state_bar_top:y_end, :]
        return hash(playable.tobytes()).__str__()

    def detect_clickable_positions(
        self, grid: np.ndarray
    ) -> list[tuple[int, int]]:
        """Detect clickable positions for click-based games.

        Returns pixel positions of multi-color blocks in the playable
        area, which are potential clickable entities.

        Args:
            grid: 64x64 integer array.

        Returns:
            List of (x, y) pixel positions that might be clickable.
        """
        clickable: list[tuple[int, int]] = []
        sprites = self.extract_sprites(grid)
        for s in sprites:
            if s.is_multicolor:
                clickable.append((s.grid_x, s.grid_y))
        return clickable

    def get_action_type(self, available_actions: list[int]) -> str:
        """Determine action type from available actions.

        Maps action IDs to interaction types:
        - Keyboard: ACTION1-4 (directional movement)
        - Click: ACTION6-7 (click/right-click)

        Args:
            available_actions: List of available action IDs.

        Returns:
            One of: 'keyboard', 'click', 'keyboard_click', 'unknown'.
        """
        keyboard_actions = {1, 2, 3, 4}
        click_actions = {6, 7}

        has_keyboard = any(a in keyboard_actions for a in available_actions)
        has_click = any(a in click_actions for a in available_actions)

        if has_keyboard and has_click:
            return "keyboard_click"
        elif has_keyboard:
            return "keyboard"
        elif has_click:
            return "click"
        return "unknown"

    def get_static_entity_positions(self) -> list[tuple[int, int]]:
        """Get positions of static entities (goals, switchers, etc.).

        Returns:
            List of (x, y) pixel positions of multi-color static blocks.
        """
        return [
            (bx * self.step, by * self.step)
            for bx, by in self.static_blocks
        ]

    def is_wall(self, pixel_x: int, pixel_y: int) -> bool:
        """Check if a pixel position is a wall.

        Args:
            pixel_x: X coordinate in pixels.
            pixel_y: Y coordinate in pixels.

        Returns:
            True if the position is a wall block.
        """
        bx = pixel_x // self.step
        by = pixel_y // self.step
        return (bx, by) in self.wall_blocks

    def reset(self) -> None:
        """Reset all perception state.

        Called when starting a new game or when the agent is reset.
        Clears all state including cross-level learned knowledge.
        """
        self.background_color = None
        self.wall_color = None
        self.player_sprite = None
        self.player_pos = None
        self.player_grid = None
        self.prev_grid = None
        self.frame_count = 0
        self.wall_blocks.clear()
        self.goal_blocks.clear()
        self.switcher_blocks.clear()
        self.static_blocks.clear()
        self.state_bar_top = 0
        self.state_bar_bottom = 0
        self._player_history.clear()
        self._learned_goals.clear()
        self._action_effects.clear()
        self._transition_history.clear()

    # ------------------------------------------------------------------
    # Interactive learning methods
    # ------------------------------------------------------------------

    def learn_from_level_transition(
        self,
        old_grid: np.ndarray,
        new_grid: np.ndarray,
        action_history: list[int],
    ) -> None:
        """Learn goal positions from level transition observations.

        When a level transition occurs, compare the old and new grids to
        identify which sprites disappeared (completed goals) and which
        new sprites appeared (new level elements). Disappeared multi-color
        or non-wall blocks near the player's last position are likely
        completed goals.

        Args:
            old_grid: Grid state before transition.
            new_grid: Grid state after transition (new level).
            action_history: Actions taken in the completed level.
        """
        if old_grid is None or new_grid is None:
            return

        # Find blocks that disappeared or changed significantly
        old_sprites = self.extract_sprites(old_grid)
        new_sprites_map: dict[tuple[int, int], SpriteBlock] = {}
        for s in self.extract_sprites(new_grid):
            new_sprites_map[(s.block_x, s.block_y)] = s

        disappeared: list[SpriteBlock] = []
        for s in old_sprites:
            pos = (s.block_x, s.block_y)
            is_significant = s.is_multicolor or (
                self.wall_color is not None
                and s.colors[0] != self.wall_color
                and s.colors[0] != self.background_color
            )
            if not is_significant:
                continue

            if pos not in new_sprites_map:
                # Block position doesn't exist in new grid — disappeared
                disappeared.append(s)
            else:
                # Block position exists but content may have changed.
                # If old was multi-color and new is background, the
                # entity at this position "disappeared"
                new_s = new_sprites_map[pos]
                if (s.is_multicolor and not new_s.is_multicolor
                        and self.background_color is not None
                        and len(new_s.colors) > 0
                        and new_s.colors[0] == self.background_color):
                    disappeared.append(s)

        # Disappeared multi-color or non-wall blocks near player's last
        # position are likely goals
        if self.player_grid is not None:
            player_bx, player_by = self.player_grid
            player_px = player_bx * self.step
            player_py = player_by * self.step
            for s in disappeared:
                pos = (s.grid_x, s.grid_y)
                dist = abs(s.block_x - player_bx) + abs(s.block_y - player_by)
                if dist <= 3:  # Within reach of player
                    confidence = max(1, 3 - dist)  # Closer = higher confidence
                    self._learned_goals[pos] = (
                        self._learned_goals.get(pos, 0) + confidence
                    )

        # Record transition
        self._transition_history.append({
            'frame_count': self.frame_count,
            'disappeared_count': len(disappeared),
            'action_count': len(action_history),
            'learned_goals': len(self._learned_goals),
        })

    def learn_from_action_effect(
        self,
        action: int,
        old_grid: np.ndarray,
        new_grid: np.ndarray,
    ) -> None:
        """Learn game mechanics from action effects.

        Tracks how each action type affects the grid, building a model
        of which blocks respond to player interaction. Blocks that
        disappear (become background) near the player are likely goals.
        Blocks that change colors may be switchers.

        Args:
            action: The action that was taken.
            old_grid: Grid state before action.
            new_grid: Grid state after action.
        """
        if old_grid is None or new_grid is None:
            return

        diff = old_grid != new_grid
        if not diff.any():
            return

        changed_blocks: set[tuple[int, int]] = set()
        changed_rows, changed_cols = np.where(diff)
        for r, c in zip(changed_rows, changed_cols):
            bx = c // self.step
            by = r // self.step
            changed_blocks.add((bx, by))

        for bx, by in changed_blocks:
            x = bx * self.step
            y = by * self.step
            old_block = old_grid[y:y + self.step, x:x + self.step]
            new_block = new_grid[y:y + self.step, x:x + self.step]
            if (old_block.shape != (self.step, self.step)
                    or new_block.shape != (self.step, self.step)):
                continue

            old_colors = set(np.unique(old_block).tolist())
            new_colors = set(np.unique(new_block).tolist())

            # Block disappeared (became background)
            if (self.background_color is not None
                    and new_colors == {self.background_color}
                    and old_colors != {self.background_color}):
                effect_type = 1  # DISAPPEARED
                # If player was nearby, this might be a goal
                if self.player_grid is not None:
                    dist = (abs(bx - self.player_grid[0])
                            + abs(by - self.player_grid[1]))
                    if dist <= 2:
                        pos = (x, y)
                        self._learned_goals[pos] = (
                            self._learned_goals.get(pos, 0) + 1
                        )
            # Block changed colors
            elif old_colors != new_colors:
                effect_type = 2  # CHANGED
            else:
                effect_type = 3  # OTHER

            if action not in self._action_effects:
                self._action_effects[action] = []
            self._action_effects[action].append(((bx, by), effect_type))

    def detect_goals_interactive(
        self, grid: np.ndarray
    ) -> list[tuple[int, int]]:
        """Detect goal positions based on learned knowledge.

        Returns positions of blocks that match learned goal patterns
        (same color/pattern as previously disappeared goals). If no
        goals have been learned yet, falls back to a heuristic of
        returning multi-color static blocks.

        Args:
            grid: Current grid state.

        Returns:
            List of (x, y) pixel positions of likely goals.
        """
        goals: list[tuple[int, int]] = []

        # If we have no learned goals, use heuristic: multi-color static blocks
        if not self._learned_goals:
            for pos, pixels in self.static_blocks.items():
                colors = np.unique(pixels)
                if len(colors) > 1:
                    goals.append((pos[0] * self.step, pos[1] * self.step))
            return goals

        # Find blocks similar to learned goals
        # Collect colors from learned goal positions
        learned_colors: set[int] = set()
        for pos in self._learned_goals:
            bx = pos[0] // self.step
            by = pos[1] // self.step
            if (bx, by) in self.static_blocks:
                learned_colors.update(
                    np.unique(self.static_blocks[(bx, by)]).tolist()
                )

        # Also check current grid for matching patterns
        sprites = self.extract_sprites(grid)
        for s in sprites:
            if s.is_multicolor:
                block_colors = set(s.colors)
                overlap = block_colors & learned_colors
                if overlap:
                    goals.append((s.grid_x, s.grid_y))

        return goals

    def get_learned_goal_positions(self) -> list[tuple[int, int]]:
        """Get goal positions learned through interaction.

        Returns positions of goals that were identified through level
        transitions and action effects, sorted by confidence (highest
        first).

        Returns:
            List of (x, y) pixel positions of learned goals, sorted
            by descending confidence.
        """
        sorted_goals = sorted(
            self._learned_goals.items(),
            key=lambda x: -x[1],
        )
        return [pos for pos, _ in sorted_goals]
