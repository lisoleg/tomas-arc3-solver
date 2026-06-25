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

    def __init__(self, game: Any, step: int = 5, game_id: Optional[str] = None) -> None:
        """Initialize the universal adapter.

        Args:
            game: The env._game object.
            step: Grid step size for block alignment.
            game_id: Optional game identifier for config-based entity detection.
                When provided, uses game_configs.py to identify player/goal/wall
                tags instead of heuristic classification.
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

        # Config-based entity detection
        self._game_id = game_id
        self._game_config = None
        self._config_loaded: bool = False
        if game_id is not None:
            self._load_game_config(game_id)

        # Game type detection
        self._is_click_game: bool = False
        self._grid_size_value: int = 64

        # Action detection cache
        self._available_action_ids: list[int] = []
        self._step_size_value: int = step  # Override with _detect_step_size()

        # All discovered tags (for supplementary wall detection)
        self._all_tags: set[str] = set()

    def _load_game_config(self, game_id: str) -> None:
        """Load game configuration from game_configs.py.

        Sets up config-based tag mappings for entity detection,
        bypassing heuristic classification when config is available.

        Args:
            game_id: Game identifier (e.g. "tu93", "vc33").
        """
        try:
            from .game_configs import GAME_CONFIGS, GameConfig
            cfg = GAME_CONFIGS.get(game_id)
            if cfg is not None:
                self._game_config = cfg
                self._grid_size_value = cfg.grid_size
                self._is_click_game = cfg.game_type in ("click", "keyboard+click")
        except ImportError:
            pass
        self._config_loaded = True

    def _get_sprites_by_tag(self, tag: str) -> list:
        """Get sprites from current_level by tag name.

        Args:
            tag: Sprite tag name.

        Returns:
            List of sprite objects, empty if not found.
        """
        try:
            cl = getattr(self.game, 'current_level', None)
            if cl is not None and hasattr(cl, 'get_sprites_by_tag'):
                return cl.get_sprites_by_tag(tag) or []
        except Exception:
            pass
        return []

    def _read_player_from_config(self) -> Optional[GameEntity]:
        """Read player using config-based tag detection.

        Returns:
            GameEntity for the player, or None if not found.
        """
        if self._game_config is None or self._game_config.player_tag is None:
            return None

        tag = self._game_config.player_tag
        sprites = self._get_sprites_by_tag(tag)

        # Also try game attribute (some games store player as attribute)
        if not sprites:
            try:
                attr_val = getattr(self.game, tag, None)
                if attr_val is not None:
                    if isinstance(attr_val, list) and len(attr_val) > 0:
                        sprites = [attr_val[0]]
                    else:
                        sprites = [attr_val]
            except Exception:
                pass

        if sprites:
            return self._to_entity_safe(sprites[0])
        return None

    def _read_goals_from_config(self) -> list[GameEntity]:
        """Read goals using config-based tag detection.

        Returns:
            List of goal GameEntity objects.
        """
        if self._game_config is None or self._game_config.goal_tag is None:
            return []

        tag = self._game_config.goal_tag
        sprites = self._get_sprites_by_tag(tag)

        goals: list[GameEntity] = []
        seen: set[tuple[int, int]] = set()

        # Get player position for filtering
        player_pos: Optional[tuple[int, int]] = None
        try:
            p = self._read_player_from_config()
            if p is not None:
                player_pos = (p.x, p.y)
        except Exception:
            pass

        for sprite in sprites:
            entity = self._to_entity_safe(sprite)
            if entity is None:
                continue
            pos = (entity.x, entity.y)
            if pos in seen:
                continue
            if pos[0] == 0 and pos[1] == 0:
                continue
            if player_pos is not None and pos == player_pos:
                continue
            seen.add(pos)
            goals.append(entity)

        return goals

    def _read_walls_from_config(self) -> list[GameEntity]:
        """Read walls using config-based tag detection.

        Returns:
            List of wall GameEntity objects.
        """
        if self._game_config is None:
            return []

        walls: list[GameEntity] = []
        seen: set[tuple[int, int]] = set()

        # Primary wall tag
        if self._game_config.wall_tag is not None:
            sprites = self._get_sprites_by_tag(self._game_config.wall_tag)
            for sprite in sprites:
                entity = self._to_entity_safe(sprite)
                if entity is not None:
                    pos = (entity.x, entity.y)
                    if pos not in seen:
                        seen.add(pos)
                        walls.append(entity)

        # Map tag (for map-based collision games like tu93)
        if self._game_config.map_tag is not None:
            sprites = self._get_sprites_by_tag(self._game_config.map_tag)
            for sprite in sprites:
                entity = self._to_entity_safe(sprite)
                if entity is not None:
                    pos = (entity.x, entity.y)
                    if pos not in seen:
                        seen.add(pos)
                        walls.append(entity)

        return walls

    def _read_switchers_from_config(self) -> list[GameEntity]:
        """Read switchers using config-based tag detection.

        Returns:
            List of switcher GameEntity objects.
        """
        if self._game_config is None or self._game_config.switcher_tag is None:
            return []

        sprites = self._get_sprites_by_tag(self._game_config.switcher_tag)
        switchers: list[GameEntity] = []
        seen: set[tuple[int, int]] = set()

        for sprite in sprites:
            entity = self._to_entity_safe(sprite)
            if entity is not None:
                pos = (entity.x, entity.y)
                if pos not in seen:
                    seen.add(pos)
                    switchers.append(entity)

        return switchers

    # ------------------------------------------------------------------
    # Discovery: scan game attributes for sprite lists and single sprites
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        """Run entity discovery if not yet done.

        Scans the game object's attributes to find sprite lists and
        single sprites. Always also checks current_level._sprites as
        a supplementary source. If the initial scan finds very little,
        also checks nested objects and dicts. Results are cached.
        """
        if self._discovered:
            return
        # Phase 1: scan direct game attributes
        self._discover_sprite_lists()
        self._discover_single_sprites()

        # Phase 2: always check current_level._sprites (supplementary)
        # This catches sprites not exposed as direct game attributes
        self._discover_from_current_level()

        # Phase 3: if still very few sprites, try nested objects
        total_found = len(self._sprite_lists) + len(self._single_sprites)
        if total_found < 3:
            self._discover_from_nested_objects()

        # Phase 4: if still very few, try dict values
        total_found = len(self._sprite_lists) + len(self._single_sprites)
        if total_found < 3:
            self._discover_from_dicts()

        self._collect_all_tags()
        self._detect_grid_size()
        self._available_action_ids = self._detect_available_actions()
        self._step_size_value = self._detect_step_size()
        self._detect_game_type()
        self._discovered = True

    # ------------------------------------------------------------------
    # Fallback discovery: current_level, nested objects, dicts
    # ------------------------------------------------------------------

    def _discover_from_current_level(self) -> None:
        """Discover sprites from current_level._sprites.

        Many games (tu93, vc33, bp35, lf52, m0r0, etc.) store all
        sprites inside current_level._sprites rather than as direct
        game attributes. This method extracts that flat sprite list
        and splits it by tags into separate groups for classification.

        Sprites already discovered via game-level attributes are
        skipped (by object identity) to avoid duplicates.
        """
        try:
            current_level = getattr(self.game, 'current_level', None)
            if current_level is None:
                return

            # Build set of already-discovered sprite object IDs
            known_ids: set[int] = set()
            for sl in self._sprite_lists:
                for s in sl.sprites:
                    known_ids.add(id(s))
            for ss in self._single_sprites:
                known_ids.add(id(ss.sprite))

            # Try _sprites first, then _sorted_sprites
            for attr in ('_sprites', '_sorted_sprites', 'sprites'):
                sprites = getattr(current_level, attr, None)
                if sprites and isinstance(sprites, list) and len(sprites) > 0:
                    if self._list_has_xy(sprites):
                        # Filter out already-discovered sprites
                        new_sprites = [s for s in sprites if id(s) not in known_ids]
                        if not new_sprites:
                            break  # All sprites already known

                        # Split sprites by their tags into groups
                        tag_groups: dict[str, list] = {}
                        no_tag_sprites: list = []

                        for s in new_sprites:
                            tags_str = ""
                            try:
                                if hasattr(s, 'tags'):
                                    raw = s.tags
                                    if isinstance(raw, (list, set, tuple)):
                                        tags_str = "|".join(sorted(str(t) for t in raw))
                            except Exception:
                                pass

                            if tags_str:
                                if tags_str not in tag_groups:
                                    tag_groups[tags_str] = []
                                tag_groups[tags_str].append(s)
                            else:
                                no_tag_sprites.append(s)

                        # Add each tag group as a separate sprite list
                        for tags_key, group_sprites in tag_groups.items():
                            sample_tags = tags_key.split("|") if tags_key else []
                            self._sprite_lists.append(DiscoveredSpriteList(
                                attr_name=f"current_level.{attr}[{tags_key[:20]}]",
                                sprites=group_sprites,
                                count=len(group_sprites),
                                sample_tags=sample_tags,
                            ))

                        # Add no-tag sprites as a separate group
                        if no_tag_sprites:
                            self._sprite_lists.append(DiscoveredSpriteList(
                                attr_name=f"current_level.{attr}[no_tag]",
                                sprites=no_tag_sprites,
                                count=len(no_tag_sprites),
                                sample_tags=[],
                            ))

                        break  # Found sprites, stop searching
        except Exception:
            pass

    def _discover_from_nested_objects(self) -> None:
        """Discover sprites from nested complex objects on the game.

        Some games (bp35, lf52) store sprites inside nested objects
        like game.oztjzzyqoek or game.ikhhdzfmarl. This method
        scans one level deeper into non-trivial objects.
        """
        try:
            attr_names = dir(self.game)
        except Exception:
            return

        for attr_name in attr_names:
            if attr_name.startswith('__'):
                continue
            # Skip already-found attrs
            if any(sl.attr_name == attr_name for sl in self._sprite_lists):
                continue
            if any(ss.attr_name == attr_name for ss in self._single_sprites):
                continue

            try:
                value = getattr(self.game, attr_name)
            except Exception:
                continue

            # Skip basic types
            if isinstance(value, (list, dict, set, tuple, str, int, float, bool, type(None))):
                continue
            if isinstance(value, np.ndarray):
                continue
            if callable(value):
                continue
            # Skip if it has x/y (would have been found as single sprite)
            if self._has_xy(value) and not self._is_camera(attr_name, value):
                continue

            # Scan this object's attributes for sprite lists/singles
            try:
                child_attrs = dir(value)
            except Exception:
                continue

            for child_name in child_attrs:
                if child_name.startswith('__'):
                    continue
                try:
                    child_val = getattr(value, child_name)
                except Exception:
                    continue
                if callable(child_val):
                    continue

                # Check for sprite list
                if isinstance(child_val, list) and len(child_val) > 0:
                    if self._list_has_xy(child_val):
                        # Avoid duplicates
                        full_name = f"{attr_name}.{child_name}"
                        if not any(sl.attr_name == full_name for sl in self._sprite_lists):
                            sample_tags: list[str] = []
                            try:
                                first = child_val[0]
                                if hasattr(first, 'tags'):
                                    raw = first.tags
                                    if isinstance(raw, (list, set, tuple)):
                                        sample_tags = [str(t) for t in raw]
                            except Exception:
                                pass

                            self._sprite_lists.append(DiscoveredSpriteList(
                                attr_name=full_name,
                                sprites=child_val,
                                count=len(child_val),
                                sample_tags=sample_tags,
                            ))

                # Check for single sprite
                elif not isinstance(child_val, (list, dict, set, tuple, str, int, float, bool, type(None), np.ndarray)):
                    if self._has_xy(child_val) and not self._is_camera(child_name, child_val):
                        full_name = f"{attr_name}.{child_name}"
                        if not any(ss.attr_name == full_name for ss in self._single_sprites):
                            tags: list[str] = []
                            try:
                                if hasattr(child_val, 'tags'):
                                    raw = child_val.tags
                                    if isinstance(raw, (list, set, tuple)):
                                        tags = [str(t) for t in raw]
                            except Exception:
                                pass

                            x_val, y_val = 0, 0
                            try:
                                x_val = int(child_val.x)
                                y_val = int(child_val.y)
                            except (TypeError, ValueError, AttributeError):
                                pass

                            self._single_sprites.append(DiscoveredSprite(
                                attr_name=full_name,
                                sprite=child_val,
                                tags=tags,
                                x=x_val,
                                y=y_val,
                            ))

    def _discover_from_dicts(self) -> None:
        """Discover sprites from dict values on the game object.

        Some games (vc33) store sprites inside dicts. This method
        checks dict values for sprite-like objects.
        """
        try:
            attr_names = dir(self.game)
        except Exception:
            return

        for attr_name in attr_names:
            if attr_name.startswith('__'):
                continue
            try:
                value = getattr(self.game, attr_name)
            except Exception:
                continue

            if not isinstance(value, dict) or len(value) == 0:
                continue

            # Collect sprites from dict values
            sprites: list = []
            for v in value.values():
                if self._has_xy(v) and not self._is_camera(attr_name, v):
                    sprites.append(v)

            if sprites:
                sample_tags: list[str] = []
                try:
                    first = sprites[0]
                    if hasattr(first, 'tags'):
                        raw = first.tags
                        if isinstance(raw, (list, set, tuple)):
                            sample_tags = [str(t) for t in raw]
                except Exception:
                    pass

                self._sprite_lists.append(DiscoveredSpriteList(
                    attr_name=attr_name,
                    sprites=sprites,
                    count=len(sprites),
                    sample_tags=sample_tags,
                ))

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
        2. Common player attribute names (_player, _controlled, etc.)
        3. Single sprite not at (0,0) with unique tags
        4. First element of a list with '_parts' in attr name
        5. The only single sprite (if exactly one exists)
        6. First element of the smallest non-wall list
        7. For click games: player may not exist (no movement needed)
        """
        # 1. Frame tracking (if already initialized)
        if self._player_found_via_tracking and self._player_attr is not None:
            return

        # 2. Common player attribute names
        common_player_attrs = [
            '_player', '_controlled', '_active_sprite',
            'player', 'controlled_sprite', 'active_sprite',
            'plnqvukupu',  # sc25 player
            'qnnpcoyzd',   # dc22 player
            'gudziatsk',   # ls20 player
            'wbmdvjhthc',  # wa30 player (tag-based, also an attr)
            'qftsebtxuc',  # g50t player (tag-based)
        ]
        for attr_name in common_player_attrs:
            val = getattr(self.game, attr_name, None)
            if val is not None and not isinstance(val, (list, dict, set, tuple)):
                if self._has_xy(val):
                    self._player_attr = attr_name
                    self._player_is_list = False
                    return
            # Also check if it's a list with 1-3 sprites
            if isinstance(val, list) and len(val) > 0 and len(val) <= 3:
                if self._has_xy(val[0]):
                    self._player_attr = attr_name
                    self._player_is_list = True
                    self._player_list_idx = 0
                    return

        # 2b. Check for click games: player may not exist
        if self._is_click_game:
            # In click-only games (actions = [6]), there may be no player
            # sprite at all. Return without setting player_attr (None).
            return

        # 3. Single sprite not at (0,0) with unique tags
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

        # 4. List with '_parts' in attr name (like TR87's qvtymdcqear_parts)
        for sl in self._sprite_lists:
            if '_parts' in sl.attr_name.lower() and sl.attr_name not in self._wall_attrs:
                self._player_attr = sl.attr_name
                self._player_is_list = True
                self._player_list_idx = 0
                return

        # 5. Only single sprite
        if len(self._single_sprites) == 1:
            ss = self._single_sprites[0]
            self._player_attr = ss.attr_name
            self._player_is_list = False
            return

        # 6. Smallest non-wall list (likely player parts)
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

        # 7. Any single sprite (even at origin)
        if self._single_sprites:
            self._player_attr = self._single_sprites[0].attr_name
            self._player_is_list = False

    def _classify_goals(self) -> None:
        """Classify goal entities using heuristics.

        Goal detection priority:
        1. sys_click tagged sprites (in click games, these are the targets)
        2. Common goal attribute names (_exit, _goal, _target, etc.)
        3. Small sprite lists (1-GOAL_LIST_MAX_SIZE) not walls
        4. Lists with click-related tags ('sys_click', 'click')
        5. Lists with goal-related tag keywords
        6. Single sprites with click tags (excluding player)
        7. In keyboard games: small sprites far from player are likely goals
        """
        used_attrs: set[str] = set(self._wall_attrs)
        if self._player_attr is not None:
            used_attrs.add(self._player_attr)
        # Exclude switchers (classified before goals)
        used_attrs.update(self._switcher_attrs)

        # 1. sys_click tagged sprites (click games — these ARE the targets)
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            for tag in sl.sample_tags:
                if tag.lower() == 'sys_click' or tag == 'sys_click':
                    self._goal_attrs.append(sl.attr_name)
                    used_attrs.add(sl.attr_name)
                    break

        # 2. Common goal attribute names
        common_goal_attrs = [
            '_exit', '_goal', '_target', '_destination',
            'exit', 'goal', 'target', 'destination',
            'hfuqkxulm',   # dc22 target
            '0015msvpvzxhqf',  # tu93 exit
            'gilbljmfbc',  # g50t goal
        ]
        for attr_name in common_goal_attrs:
            val = getattr(self.game, attr_name, None)
            if val is not None:
                if isinstance(val, list) and len(val) > 0:
                    if attr_name not in used_attrs:
                        self._goal_attrs.append(attr_name)
                        used_attrs.add(attr_name)
                elif self._has_xy(val):
                    if attr_name not in used_attrs:
                        self._goal_attrs.append(attr_name)
                        used_attrs.add(attr_name)

        # 3. Small non-wall, non-switcher lists
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            if 1 <= sl.count <= self.GOAL_LIST_MAX_SIZE:
                self._goal_attrs.append(sl.attr_name)
                used_attrs.add(sl.attr_name)

        # 4. Lists with click-related tags
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            for tag in sl.sample_tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.CLICK_TAG_KEYWORDS):
                    self._goal_attrs.append(sl.attr_name)
                    used_attrs.add(sl.attr_name)
                    break

        # 5. Lists with goal-related tag keywords
        for sl in self._sprite_lists:
            if sl.attr_name in used_attrs:
                continue
            for tag in sl.sample_tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.GOAL_TAG_KEYWORDS):
                    self._goal_attrs.append(sl.attr_name)
                    used_attrs.add(sl.attr_name)
                    break

        # 6. Single sprites with click tags (excluding player)
        for ss in self._single_sprites:
            if ss.attr_name == self._player_attr:
                continue
            for tag in ss.tags:
                tag_lower = tag.lower()
                if any(kw in tag_lower for kw in self.CLICK_TAG_KEYWORDS):
                    # Store as a goal attr (read from single sprite)
                    self._goal_attrs.append(ss.attr_name)
                    break

        # 7. In keyboard games: sprites far from player that are small
        if not self._is_click_game and self._player_attr is not None:
            try:
                player_entity = self._read_player()
                if player_entity is not None:
                    player_pos = (player_entity.x, player_entity.y)
                    for ss in self._single_sprites:
                        if ss.attr_name in used_attrs:
                            continue
                        if ss.attr_name == self._player_attr:
                            continue
                        dist = abs(ss.x - player_pos[0]) + abs(ss.y - player_pos[1])
                        # Small sprites far from player could be goals
                        if dist > 6 and 1 <= ss.x <= 60 and 1 <= ss.y <= 60:
                            self._goal_attrs.append(ss.attr_name)
                            used_attrs.add(ss.attr_name)
            except Exception:
                pass

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

    def _detect_available_actions(self) -> list[int]:
        """Detect available action IDs from game introspection.

        Extracts action IDs by calling multiple detection methods:
        1. _get_valid_actions() — primary, works for most games
        2. _get_valid_clickable_actions() — fallback for multi-step click games
        3. game.actions attribute — fallback for grid-based click games

        This is critical for game type detection: click-only games
        typically only have ACTION6 (id=6), while keyboard games have
        ACTION1-4 (id=1-4). Some games (s5i5/ft09) return None from
        _get_valid_actions but have clickable_actions or actions attrs.

        Returns:
            List of unique available action IDs (ints), or empty list if
            all detection methods fail.
        """
        action_ids: list[int] = []
        seen_ids: set[int] = set()

        def _extract_action_id(ai: Any) -> Optional[int]:
            """Extract int action ID from an ActionInput, handling GameAction enum."""
            aid = getattr(ai, 'id', None)
            if aid is None:
                return None
            # GameAction is a regular Enum (not IntEnum).
            # int(GameAction.ACTION6) raises TypeError;
            # use .value attribute instead.
            if hasattr(aid, 'value'):
                try:
                    return int(aid.value)
                except (TypeError, ValueError):
                    return None
            else:
                try:
                    return int(aid)
                except (TypeError, ValueError):
                    return None

        # Method 1: _get_valid_actions (primary)
        try:
            actions = self.game._get_valid_actions()
            if actions is not None:
                for ai in actions:
                    aid_val = _extract_action_id(ai)
                    if aid_val is not None and aid_val not in seen_ids:
                        seen_ids.add(aid_val)
                        action_ids.append(aid_val)
        except (AttributeError, TypeError, Exception):
            pass

        # Method 2: _get_valid_clickable_actions (fallback for multi-step click games)
        if not action_ids:
            try:
                click_actions = self.game._get_valid_clickable_actions()
                if click_actions is not None and len(click_actions) > 0:
                    for ai in click_actions:
                        aid_val = _extract_action_id(ai)
                        if aid_val is not None and aid_val not in seen_ids:
                            seen_ids.add(aid_val)
                            action_ids.append(aid_val)
            except (AttributeError, TypeError, Exception):
                pass

        # Method 3: game.actions attribute (fallback for grid-based click games)
        if not action_ids:
            try:
                actions_attr = self.game.actions
                if actions_attr is not None and len(actions_attr) > 0:
                    for ai in actions_attr:
                        aid_val = _extract_action_id(ai)
                        if aid_val is not None and aid_val not in seen_ids:
                            seen_ids.add(aid_val)
                            action_ids.append(aid_val)
            except (AttributeError, TypeError, Exception):
                pass

        return sorted(action_ids)

    def _detect_step_size(self) -> int:
        """Detect the game's step size from game attributes.

        Scans game attributes and its module for common step-size
        variable names used in ARC-AGI-3 games. The step size determines
        how many pixels a keyboard action moves the player per step.

        Known obfuscated step-size variable names:
            - celomdfhbh (used by several games)
            - MOVE_STEP
            - step_size, _step_size
            - hwthhtvyki (used by tu93)
            - jarvstobjt (used by g50t)
            - ndiyvmxxey (used by dc22)

        Returns:
            Detected step size, or 5 (default most common).
        """
        import sys as _sys

        # Priority list of known step-size attribute names (obfuscated + common)
        step_attr_names = [
            'celomdfhbh', 'MOVE_STEP', 'step_size', '_step_size',
            'hwthhtvyki', 'jarvstobjt', 'ndiyvmxxey',
            'gisrhqpee',  # LS20 step
        ]

        # 1. Check game instance attributes
        for attr_name in step_attr_names:
            val = getattr(self.game, attr_name, None)
            # bool is subclass of int in Python; filter it out.
            if isinstance(val, int) and not isinstance(val, bool) and val > 0 and val <= 64:
                return val

        # 2. Check game module-level constants
        try:
            game_module = _sys.modules.get(type(self.game).__module__, None)
            if game_module is not None:
                for attr_name in step_attr_names:
                    val = getattr(game_module, attr_name, None)
                    # bool is subclass of int in Python; filter it out.
                    if isinstance(val, int) and not isinstance(val, bool) and val > 0 and val <= 64:
                        return val
        except Exception:
            pass

        # 3. Heuristic: scan all game int attrs for likely step values
        common_steps = {1, 2, 3, 4, 5, 6, 8, 10, 16, 32}
        try:
            for attr_name in dir(self.game):
                if attr_name.startswith('__'):
                    continue
                try:
                    val = getattr(self.game, attr_name)
                except Exception:
                    continue
                if isinstance(val, int) and not isinstance(val, bool) and val in common_steps:
                    # Lower values are more likely step sizes than grid dims
                    if val <= 6:
                        return val
        except Exception:
            pass

        # Default: 5 is the most empirically common step size
        return 5

    def _detect_game_type(self) -> None:
        """Detect if this is a click-based or keyboard-based game.

        Uses multiple signals:
        1. Available action IDs from _get_valid_actions():
           - Only ACTION6 (id=6) → click-only game
           - ACTION1-4 (ids 1-4) → keyboard game
           - Both → keyboard+click hybrid
        2. Heuristic signals (wall presence, click tags) as fallback
        """
        # Signal 1: Action-based detection (most reliable)
        action_ids = self._detect_available_actions()
        has_keyboard = any(aid in {1, 2, 3, 4} for aid in action_ids)
        has_click = 6 in action_ids

        if action_ids:
            if has_click and not has_keyboard:
                # Only ACTION6 → click-only game
                self._is_click_game = True
                return
            elif has_keyboard:
                # Has keyboard actions → keyboard game (may also have click)
                self._is_click_game = False
                return

        # Signal 2: Check if game has _get_valid_clickable_actions() method
        # Games like s5i5/ft09/cn04 have this method even when
        # _get_valid_actions() returns None, indicating click-based games
        if not action_ids:
            has_clickable_method = hasattr(self.game, '_get_valid_clickable_actions')
            has_placeble_method = hasattr(self.game, '_get_valid_placeble_actions')
            # If both clickable & placeble methods exist → click game
            if has_clickable_method or has_placeble_method:
                self._is_click_game = True
                return

        # Signal 3: Heuristic fallback (original logic)
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
            GameEntity with width/height, or None if conversion fails.
        """
        try:
            w = int(getattr(sprite, 'width', 1))
            h = int(getattr(sprite, 'height', 1))
            return GameEntity(
                x=int(sprite.x),
                y=int(sprite.y),
                tags=self._get_tags(sprite),
                name=getattr(sprite, 'name', ''),
                grid_x=int(sprite.x),
                grid_y=int(sprite.y),
                width=w,
                height=h,
            )
        except (AttributeError, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Public API: property overrides
    # ------------------------------------------------------------------

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player entity.

        When a game config is available, uses config-based tag detection
        first. Falls back to heuristic classification and frame tracking.

        Returns:
            GameEntity for the player, or None if not found.
        """
        # Config-based detection (priority)
        if self._game_config is not None:
            entity = self._read_player_from_config()
            if entity is not None:
                return entity

        # Heuristic fallback
        self._discover()
        if not self._classified:
            self._classify_entities()

        if self._player_attr is None:
            if not self._player_found_via_tracking:
                self._track_frame_changes()
            if self._player_attr is None:
                return None

        return self._read_player()

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities.

        When a game config is available, uses config-based tag detection
        first. Falls back to heuristic classification.

        Returns:
            List of wall GameEntity objects.
        """
        # Config-based detection (priority)
        if self._game_config is not None:
            walls = self._read_walls_from_config()
            if walls:
                return walls

        # Heuristic fallback
        self._discover()
        if not self._classified:
            self._classify_entities()
        return self._read_walls()

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal entities.

        When a game config is available, uses config-based tag detection
        first. Falls back to heuristic classification.

        Returns:
            List of goal GameEntity objects.
        """
        # Config-based detection (priority)
        if self._game_config is not None:
            goals = self._read_goals_from_config()
            if goals:
                return goals

        # Heuristic fallback
        self._discover()
        if not self._classified:
            self._classify_entities()
        return self._read_goals()

    @property
    def switchers(self) -> list[GameEntity]:
        """Get switcher entities.

        When a game config is available, uses config-based tag detection
        first. Falls back to heuristic classification.

        Returns:
            List of switcher GameEntity objects.
        """
        # Config-based detection (priority)
        if self._game_config is not None:
            switchers = self._read_switchers_from_config()
            if switchers:
                return switchers

        # Heuristic fallback
        self._discover()
        if not self._classified:
            self._classify_entities()
        return self._read_switchers()

    @property
    def grid_size(self) -> int:
        """Get the detected grid size.

        When a game config is available, returns the config's grid_size.
        Otherwise, uses heuristic discovery.

        Returns:
            Grid dimension (typically 32, 64, 128, or 256).
        """
        if self._game_config is not None:
            return self._game_config.grid_size
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

    def get_all_sprites(self) -> list[Any]:
        """Get all discovered sprites from the game.

        This method returns all sprites discovered by the adapter,
        including both sprite lists and single sprites.

        Returns:
            List of all sprite objects.
        """
        self._discover()

        all_sprites: list[Any] = []

        # Add sprites from sprite lists
        for sprite_list in self._sprite_lists:
            all_sprites.extend(sprite_list.sprites)

        # Add single sprites
        for discovered_sprite in self._single_sprites:
            all_sprites.append(discovered_sprite.sprite)

        return all_sprites

    # ------------------------------------------------------------------
    # Internal readers: read fresh positions from game object
    # ------------------------------------------------------------------

    def _resolve_attr(self, attr_name: str) -> Any:
        """Resolve a (possibly virtual) attribute name to its value.

        Handles three cases:
        1. Simple name: getattr(self.game, name)
        2. Dotted path: Navigate through dot-separated attributes
           (e.g., 'oztjzzyqoek.uzkgdtbjsr' -> game.oztjzzyqoek.uzkgdtbjsr)
        3. Virtual current_level path with tag filter:
           'current_level._sprites[tag]' -> filter current_level._sprites
           by tag, returning a list of matching sprites.

        Args:
            attr_name: The attribute name to resolve.

        Returns:
            The resolved value (sprite, list of sprites, or None).
        """
        # Case 3: Virtual current_level path with tag filter
        if attr_name.startswith('current_level.') and '[' in attr_name:
            try:
                # Parse: current_level._sprites[tag] or current_level._sorted_sprites[tag]
                base, tag_part = attr_name.split('[', 1)
                tag = tag_part.rstrip(']')

                # Navigate to the base (e.g., current_level._sprites)
                parts = base.split('.')
                obj = self.game
                for part in parts:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        return None

                if not isinstance(obj, list):
                    return None

                # Filter by tag
                if tag == 'no_tag':
                    # Return sprites with no tags
                    result = []
                    for s in obj:
                        try:
                            raw_tags = getattr(s, 'tags', None)
                            if raw_tags is None or (
                                isinstance(raw_tags, (list, set, tuple)) and len(raw_tags) == 0
                            ):
                                result.append(s)
                        except Exception:
                            result.append(s)
                    return result
                elif tag:
                    # Return sprites whose tags match
                    result = []
                    for s in obj:
                        try:
                            raw_tags = getattr(s, 'tags', None)
                            if raw_tags and isinstance(raw_tags, (list, set, tuple)):
                                tag_set = set(str(t) for t in raw_tags)
                                if tag in tag_set:
                                    result.append(s)
                        except Exception:
                            pass
                    return result
                else:
                    return obj
            except Exception:
                return None

        # Case 2: Dotted path (nested object)
        if '.' in attr_name:
            try:
                parts = attr_name.split('.')
                obj = self.game
                for part in parts:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        return None
                return obj
            except Exception:
                return None

        # Case 1: Simple name
        try:
            return getattr(self.game, attr_name, None)
        except Exception:
            return None

    def _read_player(self) -> Optional[GameEntity]:
        """Read current player position from the game object.

        Uses the cached player attribute to access the sprite and
        convert it to a GameEntity. Supports virtual attribute names
        from current_level._sprites fallback discovery.

        Returns:
            GameEntity for the player, or None if not accessible.
        """
        if self._player_attr is None:
            return None

        try:
            if self._player_is_list:
                lst = self._resolve_attr(self._player_attr)
                if isinstance(lst, list) and len(lst) > self._player_list_idx:
                    sprite = lst[self._player_list_idx]
                    return self._to_entity_safe(sprite)
            else:
                sprite = self._resolve_attr(self._player_attr)
                if sprite is not None:
                    return self._to_entity_safe(sprite)
        except Exception:
            pass
        return None

    def _read_walls(self) -> list[GameEntity]:
        """Read current wall positions from the game object.

        Combines walls from:
        1. Sprite lists classified as walls (via _resolve_attr)
        2. current_level.get_sprites_by_tag results

        Returns:
            List of wall GameEntity objects.
        """
        walls: list[GameEntity] = []
        seen_positions: set[tuple[int, int]] = set()

        # 1. Wall sprite lists
        for attr_name in self._wall_attrs:
            try:
                sprites = self._resolve_attr(attr_name)
                if isinstance(sprites, list):
                    for sprite in sprites:
                        entity = self._to_entity_safe(sprite)
                        if entity is not None:
                            pos = (entity.x, entity.y)
                            if pos not in seen_positions:
                                seen_positions.add(pos)
                                walls.append(entity)
                elif sprites is not None and self._has_xy(sprites):
                    entity = self._to_entity_safe(sprites)
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

        Uses cached goal attributes to access sprites via _resolve_attr.
        Handles both sprite lists and single sprites, including
        virtual attribute names from current_level._sprites.

        Filtering rules:
        - Exclude goals at (0, 0) — typically level/screen origin.
        - Exclude goals at the player's exact position — avoids
          misclassifying the player or player-adjacent sprites as goals.
        - Exclude goals with very large pixel arrays (>100 pixels per
          side) — these are usually background/terrain, not goals.

        Returns:
            List of goal GameEntity objects.
        """
        goals: list[GameEntity] = []
        seen_positions: set[tuple[int, int]] = set()

        # Get player position for filtering
        player_pos: Optional[tuple[int, int]] = None
        try:
            p = self._read_player()
            if p is not None:
                player_pos = (p.x, p.y)
        except Exception:
            pass

        for attr_name in self._goal_attrs:
            try:
                value = self._resolve_attr(attr_name)
                if isinstance(value, list):
                    sprites = value
                elif value is not None and self._has_xy(value):
                    sprites = [value]
                else:
                    continue

                for sprite in sprites:
                    entity = self._to_entity_safe(sprite)
                    if entity is None:
                        continue
                    pos = (entity.x, entity.y)
                    if pos in seen_positions:
                        continue

                    # Filter: exclude (0, 0) — usually level origin
                    if pos[0] == 0 and pos[1] == 0:
                        continue

                    # Filter: exclude player position
                    if player_pos is not None and pos == player_pos:
                        continue

                    # Filter: exclude very large sprites (background/terrain)
                    try:
                        px = getattr(sprite, 'pixels', None)
                        if px is not None:
                            px_arr = np.asarray(px)
                            if px_arr.ndim >= 2 and max(px_arr.shape) > 100:
                                continue
                    except Exception:
                        pass

                    seen_positions.add(pos)
                    goals.append(entity)
            except Exception:
                pass

        return goals

    def _read_switchers(self) -> list[GameEntity]:
        """Read current switcher positions from the game object.

        Uses cached switcher attributes to access sprites via
        _resolve_attr. Handles both sprite lists and single sprites.

        Returns:
            List of switcher GameEntity objects.
        """
        switchers: list[GameEntity] = []
        seen_positions: set[tuple[int, int]] = set()

        for attr_name in self._switcher_attrs:
            try:
                value = self._resolve_attr(attr_name)
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
