"""
UniversalOracleAdapter — Generic Oracle adapter for any ARC-AGI-3 game.

Auto-discovers game entities by scanning env._game attributes, enabling
Oracle mode for all 25 games without requiring game-specific adapters.

The adapter uses heuristic rules to classify sprite lists and single
sprites into the standard entity categories (player, walls, goals,
switchers) by analyzing their properties:

- Player: sprite that moves between frames, or a single sprite with
  a unique tag at a non-origin position.
- Walls: large sprite lists (>5 elements), sprites with collidable=True,
  or sprites retrieved via current_level.get_sprites_by_tag.
- Goals: small sprite lists (1-10 elements) not classified as walls,
  or sprites with click-related tags (e.g., 'sys_click').
- Switchers: sprites with special tags (rot, shape, color, switch).

The adapter supports both keyboard games (move player to goal) and
click games (click on target sprites).

Usage:
    from .universal_oracle_adapter import UniversalOracleAdapter

    adapter = UniversalOracleAdapter(env._game)
    player = adapter.player
    walls = adapter.walls
    goals = adapter.goals

Author: TOMAS Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np

from .oracle_adapters import OracleAdapter, GameEntity


# ============================================================================
# Discovery result dataclasses
# ============================================================================

@dataclass
class DiscoveredSpriteList:
    """A discovered sprite list from game introspection.

    Attributes:
        attr_name: The attribute name on the game object.
        sprites: The list of sprite objects.
        count: Number of sprites in the list.
        sample_tags: Tags from the first sprite (for classification).
    """

    attr_name: str
    sprites: list
    count: int
    sample_tags: list[str] = field(default_factory=list)


@dataclass
class DiscoveredSprite:
    """A discovered single sprite from game introspection.

    Attributes:
        attr_name: The attribute name on the game object.
        sprite: The sprite object.
        tags: Tags associated with the sprite.
        x: Sprite x position.
        y: Sprite y position.
    """

    attr_name: str
    sprite: Any
    tags: list[str] = field(default_factory=list)
    x: int = 0
    y: int = 0


# ============================================================================
# UniversalOracleAdapter
# ============================================================================

class UniversalOracleAdapter(OracleAdapter):
    """Universal Oracle adapter that auto-discovers game entities.

    Scans env._game attributes to find sprite lists and single sprites,
    then classifies them using heuristic rules. This enables Oracle mode
    for any game that exposes env._game, without requiring a
    game-specific adapter.

    Classification heuristics:
        - Player: Moving sprite (frame tracking) or single sprite with
          unique tag at non-origin position.
        - Walls: Large sprite lists (>5), collidable sprites, or
          sprites from current_level.get_sprites_by_tag.
        - Goals: Small lists (1-10) not classified as walls, or sprites
          with click-related tags ('sys_click', 'click', 'target').
        - Switchers: Sprites with special tags ('rot', 'shape', 'color',
          'switch').

    The adapter caches discovery results (which attributes hold sprites)
    but reads fresh positions on each property access, ensuring
    up-to-date entity positions.

    Attributes:
        game: The env._game object.
        step: Grid step size (typically 5).
    """

    # Classification thresholds
    WALL_LIST_MIN_SIZE: int = 5
    GOAL_LIST_MAX_SIZE: int = 10
    SWITCHER_LIST_MAX_SIZE: int = 20

    # Tag keywords for classification
    WALL_TAG_KEYWORDS: set[str] = {
        'wall', 'block', 'barrier', 'obstacle', 'solid', 'static',
    }
    GOAL_TAG_KEYWORDS: set[str] = {
        'goal', 'target', 'click', 'win', 'exit', 'end', 'objective',
        'sys_click', 'prize', 'reward', 'finish',
    }
    SWITCHER_TAG_KEYWORDS: set[str] = {
        'switch', 'rot', 'shape', 'color', 'change', 'toggle', 'lever',
        'rhsxkxzdjz', 'ttfwljgohq', 'soyhouuebz',
    }
    CAMERA_KEYWORDS: set[str] = {
        'camera', 'view', 'viewport', 'screen', 'canvas', 'render',
    }
    CLICK_TAG_KEYWORDS: set[str] = {
        'sys_click', 'click', 'tap', 'press', 'button',
    }

    def __init__(self, game: Any, step: int = 5) -> None:
        """Initialize the universal adapter.

        Args:
            game: The env._game object.
            step: Grid step size for block alignment.
        """
        super().__init__(game, step)

        # Discovery cache (which attributes hold sprites)
        self._sprite_lists: list[DiscoveredSpriteList] = []
        self._single_sprites: list[DiscoveredSprite] = []
        self._discovered: bool = False

        # Classification cache (which attrs are player/walls/goals/switchers)
        self._player_attr: Optional[str] = None
        self._player_is_list: bool = False
        self._player_list_idx: int = 0
        self._wall_attrs: list[str] = []
        self._wall_tag_sprites: list[str] = []  # Tags for get_sprites_by_tag
        self._goal_attrs: list[str] = []
        self._switcher_attrs: list[str] = []
        self._classified: bool = False

        # Frame tracking state
        self._prev_positions: dict[str, tuple[int, int]] = {}
        self._frame_initialized: bool = False
        self._player_found_via_tracking: bool = False

        # Game type detection
        self._is_click_game: bool = False
        self._grid_size_value: int = 64

        # All discovered tags (for supplementary wall detection)
        self._all_tags: set[str] = set()

    # ------------------------------------------------------------------
    # Discovery: scan game attributes for sprite lists and single sprites
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        """Run entity discovery if not yet done.

        Scans the game object's attributes to find sprite lists and
        single sprites. Results are cached for subsequent calls.
        """
        if self._discovered:
            return
        self._discover_sprite_lists()
        self._discover_single_sprites()
        self._collect_all_tags()
        self._detect_grid_size()
        self._detect_game_type()
        self._discovered = True

    def _discover_sprite_lists(self) -> None:
        """Scan game object attributes to find sprite lists.

        Iterates over all non-callable attributes of the game object,
        checking each for being a list of objects with x/y attributes.
        Found sprite lists are stored in self._sprite_lists.
        """
        try:
            attr_names = dir(self.game)
        except Exception:
            return

        for attr_name in attr_names:
            # Skip private/dunder attributes (but allow _game, _level etc.
            # that might hold sprites in some games)
            if attr_name.startswith('__'):
                continue

            try:
                value = getattr(self.game, attr_name)
            except Exception:
                continue

            # Must be a list
            if not isinstance(value, list):
                continue

            # Must be non-empty
            if len(value) == 0:
                continue

            # Check if elements have x/y attributes
            if not self._list_has_xy(value):
                continue

            # Extract sample tags from first sprite
            sample_tags: list[str] = []
            try:
                first = value[0]
                if hasattr(first, 'tags'):
                    tags = first.tags
                    if isinstance(tags, list):
                        sample_tags = [str(t) for t in tags]
                    elif isinstance(tags, (set, tuple)):
                        sample_tags = [str(t) for t in tags]
            except Exception:
                pass

            self._sprite_lists.append(DiscoveredSpriteList(
                attr_name=attr_name,
                sprites=value,
                count=len(value),
                sample_tags=sample_tags,
            ))

    def _discover_single_sprites(self) -> None:
        """Find non-camera single sprites with x/y attributes.

        Iterates over all non-callable, non-list attributes of the game
        object, checking each for being an object with x/y attributes.
        Camera-like objects are excluded.
        """
        try:
            attr_names = dir(self.game)
        except Exception:
            return

        for attr_name in attr_names:
            if attr_name.startswith('__'):
                continue

            # Skip attributes already found as sprite lists
            if any(sl.attr_name == attr_name for sl in self._sprite_lists):
                continue

            try:
                value = getattr(self.game, attr_name)
            except Exception:
                continue

            # Skip lists and basic types
            if isinstance(value, (list, dict, set, tuple, str, int, float, bool, type(None))):
                continue

            # Skip numpy arrays
            if isinstance(value, np.ndarray):
                continue

            # Skip modules and functions
            if callable(value):
                continue

            # Check for x/y attributes
            if not self._has_xy(value):
                continue

            # Skip camera-like objects
            if self._is_camera(attr_name, value):
                continue

            # Extract tags
            tags: list[str] = []
            try:
                if hasattr(value, 'tags'):
                    raw_tags = value.tags
                    if isinstance(raw_tags, (list, set, tuple)):
                        tags = [str(t) for t in raw_tags]
            except Exception:
                pass

            # Get position
            x_val = 0
            y_val = 0
            try:
                x_val = int(value.x)
                y_val = int(value.y)
            except (TypeError, ValueError, AttributeError):
                pass

            self._single_sprites.append(DiscoveredSprite(
                attr_name=attr_name,
                sprite=value,
                tags=tags,
                x=x_val,
                y=y_val,
            ))

    def _collect_all_tags(self) -> None:
        """Collect all unique tags from discovered sprites.

        This is used for supplementary wall detection via
        current_level.get_sprites_by_tag.
        """
        all_tags: set[str] = set()

        for sl in self._sprite_lists:
            all_tags.update(sl.sample_tags)
            # Also check other sprites in the list for tags
            for sprite in sl.sprites[:10]:  # Limit for performance
                try:
                    if hasattr(sprite, 'tags'):
                        raw_tags = sprite.tags
                        if isinstance(raw_tags, (list, set, tuple)):
                            all_tags.update(str(t) for t in raw_tags)
                except Exception:
                    pass

        for ss in self._single_sprites:
            all_tags.update(ss.tags)

        self._all_tags = all_tags

    # ------------------------------------------------------------------
    # Classification: assign entity roles using heuristics
    # ------------------------------------------------------------------

    def _classify_entities(self) -> None:
        """Classify discovered entities using heuristic rules.

        Assigns each discovered sprite list and single sprite to one of:
        player, walls, goals, or switchers. Classification is cached
        in self._player_attr, self._wall_attrs, self._goal_attrs,
        self._switcher_attrs.
        """
        if self._classified:
            return

        self._discover()

        # Classify walls first (largest lists, collidable sprites)
        self._classify_walls()

        # Classify player (single sprites, frame tracking)
        self._classify_player()

        # Classify switchers BEFORE goals so that switcher-tagged lists
        # are not misclassified as goals (small lists with special tags
        # like 'rot', 'shape', 'color' should be switchers, not goals).
        self._classify_switchers()

        # Classify goals (remaining small lists, click tags)
        self._classify_goals()

        self._classified = True

    def _classify_walls(self) -> None:
        """Classify wall entities using heuristics.

        Wall detection priority:
        1. current_level.get_sprites_by_tag for wall-like tags
        2. Large sprite lists (>WALL_LIST_MIN_SIZE elements)
        3. Sprites with collidable=True attribute
        4. Lists with wall-related tag keywords
        """
        used_attrs: set[str] = set()

        # 1. Try current_level.get_sprites_by_tag for wall-like tags
        self._wall_tag_sprites = []
        try:
            current_level = getattr(self.game, 'current_level', None)
            if current_level is not None and hasattr(current_level, 'get_sprites_by_tag'):
                for tag in self._all_tags:
                    try:
                        sprites = current_level.get_sprites_by_tag(tag)
                        if sprites and len(sprites) >= self.WALL_LIST_MIN_SIZE:
                            # Large tag group -> likely walls
                            self._wall_tag_sprites.append(tag)
                    except Exception:
                        pass
        except Exception:
            pass

        # 2. Large sprite lists are walls
        for sl in self._sprite_lists:
            if sl.count >= self.WALL_LIST_MIN_SIZE:
                if sl.attr_name not in used_attrs:
                    self._wall_attrs.append(sl.attr_name)
                    used_attrs.add(sl.attr_name)

        # 3. Lists with collidable sprites
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            collidable_count = 0
            for sprite in sl.sprites[:5]:  # Check first 5
                try:
                    if getattr(sprite, 'collidable', False):
                        collidable_count += 1
                except Exception:
                    pass
            if collidable_count >= 2:  # At least 2 of 5 are collidable
                self._wall_attrs.append(sl.attr_name)
                used_attrs.add(sl.attr_name)

        # 4. Lists with wall-related tag keywords
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            for tag in sl.sample_tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.WALL_TAG_KEYWORDS):
                    self._wall_attrs.append(sl.attr_name)
                    used_attrs.add(sl.attr_name)
                    break

    def _classify_player(self) -> None:
        """Classify the player entity using heuristics.

        Player detection priority:
        1. Frame tracking: the sprite that moved between frames
        2. Single sprite not at (0,0) with unique tags
        3. First element of a list with '_parts' in attr name
        4. The only single sprite (if exactly one exists)
        5. First element of the smallest non-wall list
        """
        # 1. Frame tracking (if already initialized)
        if self._player_found_via_tracking and self._player_attr is not None:
            return

        # 2. Single sprite not at (0,0) with unique tags
        for ss in self._single_sprites:
            if ss.x != 0 or ss.y != 0:
                # Check if tags are unique (not shared with wall lists)
                if ss.tags:
                    tag_set = set(ss.tags)
                    wall_tag_overlap = any(
                        tag_set & set(sl.sample_tags)
                        for sl in self._sprite_lists
                        if sl.attr_name in self._wall_attrs
                    )
                    if not wall_tag_overlap:
                        self._player_attr = ss.attr_name
                        self._player_is_list = False
                        return

        # 3. List with '_parts' in attr name (like TR87's qvtymdcqear_parts)
        for sl in self._sprite_lists:
            if '_parts' in sl.attr_name.lower() and sl.attr_name not in self._wall_attrs:
                self._player_attr = sl.attr_name
                self._player_is_list = True
                self._player_list_idx = 0
                return

        # 4. Only single sprite
        if len(self._single_sprites) == 1:
            ss = self._single_sprites[0]
            self._player_attr = ss.attr_name
            self._player_is_list = False
            return

        # 5. Smallest non-wall list (likely player parts)
        non_wall_lists = [
            sl for sl in self._sprite_lists
            if sl.attr_name not in self._wall_attrs and sl.count <= 3
        ]
        if non_wall_lists:
            # Sort by count (smallest first)
            non_wall_lists.sort(key=lambda x: x.count)
            self._player_attr = non_wall_lists[0].attr_name
            self._player_is_list = True
            self._player_list_idx = 0
            return

        # 6. Any single sprite (even at origin)
        if self._single_sprites:
            self._player_attr = self._single_sprites[0].attr_name
            self._player_is_list = False

    def _classify_goals(self) -> None:
        """Classify goal entities using heuristics.

        Goal detection priority:
        1. Small sprite lists (1-GOAL_LIST_MAX_SIZE) not walls
        2. Lists with click-related tags ('sys_click', 'click')
        3. Lists with goal-related tag keywords
        4. Single sprites with click tags (excluding player)
        """
        used_attrs: set[str] = set(self._wall_attrs)
        if self._player_attr is not None:
            used_attrs.add(self._player_attr)
        # Exclude switchers (classified before goals)
        used_attrs.update(self._switcher_attrs)

        # 1. Small non-wall, non-switcher lists
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            if 1 <= sl.count <= self.GOAL_LIST_MAX_SIZE:
                self._goal_attrs.append(sl.attr_name)
                used_attrs.add(sl.attr_name)

        # 2. Lists with click-related tags
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            for tag in sl.sample_tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.CLICK_TAG_KEYWORDS):
                    self._goal_attrs.append(sl.attr_name)
                    used_attrs.add(sl.attr_name)
                    break

        # 3. Lists with goal-related tag keywords
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            for tag in sl.sample_tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.GOAL_TAG_KEYWORDS):
                    self._goal_attrs.append(sl.attr_name)
                    used_attrs.add(sl.attr_name)
                    break

        # 4. Single sprites with click tags (excluding player)
        for ss in self._single_sprites:
            if ss.attr_name == self._player_attr:
                continue
            for tag in ss.tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.CLICK_TAG_KEYWORDS):
                    # Store as a goal attr (read from single sprite)
                    self._goal_attrs.append(ss.attr_name)
                    break

    def _classify_switchers(self) -> None:
        """Classify switcher entities using heuristics.

        Switcher detection: lists or sprites with special tags
        ('rot', 'shape', 'color', 'switch') that aren't already
        classified as walls, goals, or player.
        """
        used_attrs: set[str] = set(self._wall_attrs + self._goal_attrs)
        if self._player_attr is not None:
            used_attrs.add(self._player_attr)

        # Lists with switcher-related tags
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            for tag in sl.sample_tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.SWITCHER_TAG_KEYWORDS):
                    self._switcher_attrs.append(sl.attr_name)
                    used_attrs.add(sl.attr_name)
                    break

        # Single sprites with switcher tags
        for ss in self._single_sprites:
            if ss.attr_name in used_attrs:
                continue
            for tag in ss.tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.SWITCHER_TAG_KEYWORDS):
                    self._switcher_attrs.append(ss.attr_name)
                    used_attrs.add(ss.attr_name)
                    break

        # Medium-sized lists not yet classified (possible switchers)
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            if self.GOAL_LIST_MAX_SIZE < sl.count <= self.SWITCHER_LIST_MAX_SIZE:
                self._switcher_attrs.append(sl.attr_name)
                used_attrs.add(sl.attr_name)

    # ------------------------------------------------------------------
    # Frame tracking: identify player by movement
    # ------------------------------------------------------------------

    def _track_frame_changes(self) -> None:
        """Track sprite positions between frames to identify the player.

        Compares current sprite positions with previously stored
        positions. The sprite that moved is identified as the player.
        This is called automatically when the player property is
        accessed and no player has been identified yet.
        """
        # Build current position map: attr_name -> (x, y)
        current_positions: dict[str, tuple[int, int]] = {}

        for sl in self._sprite_lists:
            for i, sprite in enumerate(sl.sprites):
                try:
                    pos = (int(sprite.x), int(sprite.y))
                    current_positions[f"{sl.attr_name}[{i}]"] = pos
                except (TypeError, ValueError, AttributeError):
                    pass

        for ss in self._single_sprites:
            try:
                pos = (int(ss.sprite.x), int(ss.sprite.y))
                current_positions[ss.attr_name] = pos
            except (TypeError, ValueError, AttributeError):
                pass

        if not self._frame_initialized:
            # First call: store positions and return
            self._prev_positions = current_positions
            self._frame_initialized = True
            return

        # Compare positions to find moving sprite
        moved_attrs: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
        for key, curr_pos in current_positions.items():
            prev_pos = self._prev_positions.get(key)
            if prev_pos is not None and prev_pos != curr_pos:
                moved_attrs.append((key, prev_pos, curr_pos))

        if moved_attrs:
            # The first moving sprite is likely the player
            moved_key = moved_attrs[0][0]

            # Extract attr name (handle list index format)
            if '[' in moved_key:
                base_attr = moved_key.split('[')[0]
                idx_str = moved_key.split('[')[1].rstrip(']')
                try:
                    idx = int(idx_str)
                except ValueError:
                    idx = 0
                self._player_attr = base_attr
                self._player_is_list = True
                self._player_list_idx = idx
            else:
                self._player_attr = moved_key
                self._player_is_list = False

            self._player_found_via_tracking = True

        # Update stored positions
        self._prev_positions = current_positions

    # ------------------------------------------------------------------
    # Game type and grid size detection
    # ------------------------------------------------------------------

    def _detect_grid_size(self) -> None:
        """Detect grid size from game attributes.

        Looks for integer attributes that are common grid dimensions
        (32, 64, 128, 256). Falls back to 64 if none found.
        """
        common_sizes = {32, 64, 128, 256, 512}

        try:
            attr_names = dir(self.game)
        except Exception:
            self._grid_size_value = 64
            return

        found_sizes: list[int] = []
        for attr_name in attr_names:
            if attr_name.startswith('__'):
                continue
            try:
                value = getattr(self.game, attr_name)
            except Exception:
                continue
            if isinstance(value, int) and value in common_sizes:
                found_sizes.append(value)

        if found_sizes:
            # Use the most common found size, or the largest
            from collections import Counter
            size_counts = Counter(found_sizes)
            self._grid_size_value = size_counts.most_common(1)[0][0]
        else:
            self._grid_size_value = 64

    def _detect_game_type(self) -> None:
        """Detect if this is a click-based or keyboard-based game.

        A game is classified as click-based if:
        1. No wall lists are found, AND
        2. Goal sprites have click-related tags, OR
        3. There are very few walls and many clickable-looking sprites.
        """
        has_walls = len(self._wall_attrs) > 0 or len(self._wall_tag_sprites) > 0
        has_click_tags = False

        for sl in self._sprite_lists:
            for tag in sl.sample_tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.CLICK_TAG_KEYWORDS):
                    has_click_tags = True
                    break
            if has_click_tags:
                break

        if not has_click_tags:
            for ss in self._single_sprites:
                for tag in ss.tags:
                    tag_lower = tag.lower()
                    if any(kw in tag_lower for kw in self.CLICK_TAG_KEYWORDS):
                        has_click_tags = True
                        break
                if has_click_tags:
                    break

        # Click game: no walls + click tags, or very few walls + click tags
        if has_click_tags and not has_walls:
            self._is_click_game = True
        elif has_click_tags and len(self._wall_attrs) <= 1:
            self._is_click_game = True
        else:
            self._is_click_game = False

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _has_xy(self, obj: Any) -> bool:
        """Check if an object has x and y numeric attributes.

        Args:
            obj: The object to check.

        Returns:
            True if the object has x and y attributes that can be
            converted to int.
        """
        try:
            _ = int(obj.x)
            _ = int(obj.y)
            return True
        except (AttributeError, TypeError, ValueError):
            return False

    def _list_has_xy(self, lst: list) -> bool:
        """Check if a list contains objects with x/y attributes.

        Args:
            lst: The list to check.

        Returns:
            True if at least the first element has x and y attributes.
        """
        if not lst:
            return False
        return self._has_xy(lst[0])

    def _is_camera(self, attr_name: str, obj: Any) -> bool:
        """Check if an object is likely a camera.

        A camera is identified by:
        - Attribute name containing camera keywords
        - Object having camera-like attributes (zoom, follow, viewport)

        Args:
            attr_name: The attribute name on the game object.
            obj: The object to check.

        Returns:
            True if the object appears to be a camera.
        """
        # Check attribute name
        name_lower = attr_name.lower()
        if any(kw in name_lower for kw in self.CAMERA_KEYWORDS):
            return True

        # Check for camera-like attributes
        camera_attrs = {'zoom', 'follow', 'viewport', 'scale', 'offset_x', 'offset_y'}
        try:
            obj_attrs = set(dir(obj))
        except Exception:
            return False

        camera_attr_count = len(camera_attrs & obj_attrs)
        if camera_attr_count >= 2:
            return True

        # Check if it's a level/container object (has sprites attribute)
        if hasattr(obj, 'sprites') and isinstance(getattr(obj, 'sprites'), list):
            return True

        # Check if it's the current_level object
        if attr_name == 'current_level':
            return True

        return False

    def _get_tags(self, sprite: Any) -> list[str]:
        """Safely extract tags from a sprite.

        Args:
            sprite: The sprite object.

        Returns:
            List of tag strings, or empty list if no tags.
        """
        try:
            tags = sprite.tags
            if isinstance(tags, list):
                return [str(t) for t in tags]
            elif isinstance(tags, (set, tuple)):
                return [str(t) for t in tags]
        except Exception:
            pass
        return []

    def _to_entity_safe(self, sprite: Any) -> Optional[GameEntity]:
        """Safely convert a sprite to GameEntity.

        Args:
            sprite: The sprite object with x, y attributes.

        Returns:
            GameEntity, or None if conversion fails.
        """
        try:
            return GameEntity(
                x=int(sprite.x),
                y=int(sprite.y),
                tags=self._get_tags(sprite),
                name=getattr(sprite, 'name', ''),
                grid_x=int(sprite.x),
                grid_y=int(sprite.y),
            )
        except (AttributeError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Public API: property overrides
    # ------------------------------------------------------------------

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player entity.

        Reads the current player position from the game object using
        the cached player attribute. If no player has been identified
        yet, attempts frame tracking.

        Returns:
            GameEntity for the player, or None if not found.
        """
        self._discover()
        if not self._classified:
            self._classify_entities()

        if self._player_attr is None:
            # Try frame tracking if no player found via heuristics
            if not self._player_found_via_tracking:
                self._track_frame_changes()
            if self._player_attr is None:
                return None

        return self._read_player()

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities.

        Reads current wall positions from the game object using cached
        wall attributes. Also retrieves walls via
        current_level.get_sprites_by_tag if available.

        Returns:
            List of wall GameEntity objects.
        """
        self._discover()
        if not self._classified:
            self._classify_entities()
        return self._read_walls()

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal entities.

        Reads current goal positions from the game object using cached
        goal attributes.

        Returns:
            List of goal GameEntity objects.
        """
        self._discover()
        if not self._classified:
            self._classify_entities()
        return self._read_goals()

    @property
    def switchers(self) -> list[GameEntity]:
        """Get switcher entities.

        Reads current switcher positions from the game object using
        cached switcher attributes.

        Returns:
            List of switcher GameEntity objects.
        """
        self._discover()
        if not self._classified:
            self._classify_entities()
        return self._read_switchers()

    @property
    def grid_size(self) -> int:
        """Get the detected grid size.

        Returns:
            Grid dimension (typically 32, 64, 128, or 256).
        """
        self._discover()
        return self._grid_size_value

    @property
    def is_click_game(self) -> bool:
        """Whether this game uses click-based interaction.

        Returns:
            True if the game appears to be click-based.
        """
        self._discover()
        return self._is_click_game

    # ------------------------------------------------------------------
    # Internal readers: read fresh positions from game object
    # ------------------------------------------------------------------

    def _read_player(self) -> Optional[GameEntity]:
        """Read current player position from the game object.

        Uses the cached player attribute to access the sprite and
        convert it to a GameEntity.

        Returns:
            GameEntity for the player, or None if not accessible.
        """
        if self._player_attr is None:
            return None

        try:
            if self._player_is_list:
                lst = getattr(self.game, self._player_attr, None)
                if isinstance(lst, list) and len(lst) > self._player_list_idx:
                    sprite = lst[self._player_list_idx]
                    return self._to_entity_safe(sprite)
            else:
                sprite = getattr(self.game, self._player_attr, None)
                if sprite is not None:
                    return self._to_entity_safe(sprite)
        except Exception:
            pass
        return None

    def _read_walls(self) -> list[GameEntity]:
        """Read current wall positions from the game object.

        Combines walls from:
        1. Sprite lists classified as walls
        2. current_level.get_sprites_by_tag results

        Returns:
            List of wall GameEntity objects.
        """
        walls: list[GameEntity] = []
        seen_positions: set[tuple[int, int]] = set()

        # 1. Wall sprite lists
        for attr_name in self._wall_attrs:
            try:
                sprites = getattr(self.game, attr_name, None)
                if isinstance(sprites, list):
                    for sprite in sprites:
                        entity = self._to_entity_safe(sprite)
                        if entity is not None:
                            pos = (entity.x, entity.y)
                            if pos not in seen_positions:
                                seen_positions.add(pos)
                                walls.append(entity)
            except Exception:
                pass

        # 2. current_level.get_sprites_by_tag
        if self._wall_tag_sprites:
            try:
                current_level = getattr(self.game, 'current_level', None)
                if current_level is not None and hasattr(current_level, 'get_sprites_by_tag'):
                    for tag in self._wall_tag_sprites:
                        try:
                            sprites = current_level.get_sprites_by_tag(tag)
                            if sprites:
                                for sprite in sprites:
                                    entity = self._to_entity_safe(sprite)
                                    if entity is not None:
                                        pos = (entity.x, entity.y)
                                        if pos not in seen_positions:
                                            seen_positions.add(pos)
                                            walls.append(entity)
                        except Exception:
                            pass
            except Exception:
                pass

        return walls

    def _read_goals(self) -> list[GameEntity]:
        """Read current goal positions from the game object.

        Uses cached goal attributes to access sprites. Handles both
        sprite lists and single sprites.

        Returns:
            List of goal GameEntity objects.
        """
        goals: list[GameEntity] = []
        seen_positions: set[tuple[int, int]] = set()

        for attr_name in self._goal_attrs:
            try:
                value = getattr(self.game, attr_name, None)
                if isinstance(value, list):
                    for sprite in value:
                        entity = self._to_entity_safe(sprite)
                        if entity is not None:
                            pos = (entity.x, entity.y)
                            if pos not in seen_positions:
                                seen_positions.add(pos)
                                goals.append(entity)
                elif value is not None and self._has_xy(value):
                    entity = self._to_entity_safe(value)
                    if entity is not None:
                        pos = (entity.x, entity.y)
                        if pos not in seen_positions:
                            seen_positions.add(pos)
                            goals.append(entity)
            except Exception:
                pass

        return goals

    def _read_switchers(self) -> list[GameEntity]:
        """Read current switcher positions from the game object.

        Uses cached switcher attributes to access sprites. Handles both
        sprite lists and single sprites.

        Returns:
            List of switcher GameEntity objects.
        """
        switchers: list[GameEntity] = []
        seen_positions: set[tuple[int, int]] = set()

        for attr_name in self._switcher_attrs:
            try:
                value = getattr(self.game, attr_name, None)
                if isinstance(value, list):
                    for sprite in value:
                        entity = self._to_entity_safe(sprite)
                        if entity is not None:
                            pos = (entity.x, entity.y)
                            if pos not in seen_positions:
                                seen_positions.add(pos)
                                switchers.append(entity)
                elif value is not None and self._has_xy(value):
                    entity = self._to_entity_safe(value)
                    if entity is not None:
                        pos = (entity.x, entity.y)
                        if pos not in seen_positions:
                            seen_positions.add(pos)
                            switchers.append(entity)
            except Exception:
                pass

        return switchers

    # ------------------------------------------------------------------
    # Public utility methods
    # ------------------------------------------------------------------

    def update(self) -> None:
        """Update frame tracking state.

        Should be called between game steps to enable player detection
        via frame differencing. This is optional — the player property
        will also attempt frame tracking on first access.
        """
        self._discover()
        self._track_frame_changes()

    def get_stats(self) -> dict:
        """Get adapter statistics for debugging.

        Returns:
            Dictionary with discovery and classification statistics.
        """
        self._discover()
        if not self._classified:
            self._classify_entities()

        return {
            'adapter_type': 'UniversalOracleAdapter',
            'sprite_lists_found': len(self._sprite_lists),
            'single_sprites_found': len(self._single_sprites),
            'wall_attrs': list(self._wall_attrs),
            'wall_tag_sprites': list(self._wall_tag_sprites),
            'goal_attrs': list(self._goal_attrs),
            'switcher_attrs': list(self._switcher_attrs),
            'player_attr': self._player_attr,
            'player_is_list': self._player_is_list,
            'player_found_via_tracking': self._player_found_via_tracking,
            'is_click_game': self._is_click_game,
            'grid_size': self._grid_size_value,
            'all_tags': sorted(self._all_tags),
            'sprite_list_details': [
                {
                    'attr': sl.attr_name,
                    'count': sl.count,
                    'tags': sl.sample_tags,
                }
                for sl in self._sprite_lists
            ],
            'single_sprite_details': [
                {
                    'attr': ss.attr_name,
                    'x': ss.x,
                    'y': ss.y,
                    'tags': ss.tags,
                }
                for ss in self._single_sprites
            ],
        }

    def __repr__(self) -> str:
        """String representation of the adapter.

        Returns:
            Summary string with key statistics.
        """
        self._discover()
        if not self._classified:
            self._classify_entities()
        return (
            f"UniversalOracleAdapter("
            f"sprite_lists={len(self._sprite_lists)}, "
            f"single_sprites={len(self._single_sprites)}, "
            f"walls={len(self._wall_attrs)}, "
            f"goals={len(self._goal_attrs)}, "
            f"switchers={len(self._switcher_attrs)}, "
            f"player={'found' if self._player_attr else 'none'}, "
            f"click_game={self._is_click_game}, "
            f"grid_size={self._grid_size_value})"
        )
