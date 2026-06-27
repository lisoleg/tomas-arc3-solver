"""
OracleAdapters — Game-specific Oracle adapters for ARC-AGI-3.

Provides adapter classes that translate obfuscated game attribute names
to a unified interface. Each adapter wraps env._game and exposes
standardized properties: player, walls, goals, switchers, etc.

This enables Oracle mode to work beyond LS20, supporting games like
tr87 (keyboard) and ft09 (click) that also expose env._game but with
different obfuscated attribute names.

Adapter Registry:
    - LS20Adapter: LS20 game (player=gudziatsk, walls=plrpelhym, etc.)
    - TR87Adapter: TR87 game (walls=zdwrfusvmx, goals=ztgmtnnufb, etc.)
    - FT09Adapter: FT09 game (goals=fhc, player=zth, etc.)
    - KA59Adapter: KA59 game (player=0022vrxelxosfy tag, blocks=0010xzmuziohuf tag, etc.)
    - AR25Adapter: AR25 game (player=0006lxjtqggkmi tag, goals=0001sruqbuvukh tag, etc.)
    - TN36Adapter: TN36 game (goals=kntfjgchzd tag, state_machines=mkfavqnwxy tag, etc.)
    - SB26Adapter: SB26 game (goals=lngftsryyw tag, slots=susublrply tag, etc.)

Usage:
    from .oracle_adapters import auto_detect_adapter, get_oracle_adapter

    # Auto-detect from game object
    adapter = auto_detect_adapter(env._game)

    # Or by game ID
    adapter = get_oracle_adapter("ls20", env._game)

    if adapter:
        player = adapter.player
        walls = adapter.walls
        goals = adapter.goals

Author: TOMAS Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np


# ============================================================================
# GameEntity: Standardized game entity representation
# ============================================================================

@dataclass
class GameEntity:
    """Standardized game entity representation.

    Provides a unified interface for game entities (player, walls, goals,
    switchers) regardless of the underlying game's obfuscated attribute
    names.

    Attributes:
        x: Pixel x position.
        y: Pixel y position.
        tags: Sprite tags for identification.
        name: Sprite name.
        grid_x: Grid-aligned x position (may differ from x if grid
            origin is not 0).
        grid_y: Grid-aligned y position.
        width: Sprite width in pixels (default 1 for minimal sprites).
        height: Sprite height in pixels (default 1 for minimal sprites).
    """

    x: int
    y: int
    tags: list[str]
    name: str = ""
    grid_x: int = 0
    grid_y: int = 0
    width: int = 1
    height: int = 1


# ============================================================================
# OracleAdapter: Base class for game-specific adapters
# ============================================================================

class OracleAdapter:
    """Base class for game-specific Oracle adapters.

    Subclasses implement game-specific attribute mapping by overriding
    the abstract properties. The base class provides common utilities
    for sprite extraction and safe attribute access.

    The adapter wraps env._game and exposes a unified interface:
        - player: GameEntity or None
        - walls: list[GameEntity]
        - goals: list[GameEntity]
        - switchers: list[GameEntity]
        - grid_size: int
        - level_index: int
        - win_score: int

    Attributes:
        game: The env._game object.
        step: Grid step size (typically 5).
    """

    def __init__(self, game: Any, step: int = 5) -> None:
        """Initialize the adapter.

        Args:
            game: The env._game object.
            step: Grid step size for block alignment.
        """
        self.game = game
        self.step = step

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player entity.

        Returns:
            GameEntity for the player, or None if not found.
        """
        raise NotImplementedError

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities.

        Returns:
            List of wall GameEntity objects.
        """
        raise NotImplementedError

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal entities.

        Returns:
            List of goal GameEntity objects.
        """
        raise NotImplementedError

    @property
    def switchers(self) -> list[GameEntity]:
        """Get switcher entities.

        Returns:
            List of switcher GameEntity objects (empty by default).
        """
        return []

    @property
    def grid_size(self) -> int:
        """Get grid size.

        Returns:
            Grid dimension (typically 64).
        """
        return 64

    @property
    def level_index(self) -> int:
        """Get current level index.

        Returns:
            Current level index (0-based), or 0 if not available.
        """
        if hasattr(self.game, 'level_index'):
            return self.game.level_index
        return 0

    @property
    def win_score(self) -> int:
        """Get number of levels to win.

        Returns:
            Win score (total levels), or 0 if not available.
        """
        if hasattr(self.game, 'win_score'):
            return self.game.win_score
        return 0

    def _to_entity(self, sprite: Any, step: Optional[int] = None) -> GameEntity:
        """Convert a sprite to GameEntity.

        Args:
            sprite: Sprite object with x, y attributes.
            step: Override step size for grid alignment.

        Returns:
            GameEntity with position, tag, and size information.
        """
        s = step if step is not None else self.step
        w = int(getattr(sprite, 'width', 1))
        h = int(getattr(sprite, 'height', 1))
        return GameEntity(
            x=sprite.x,
            y=sprite.y,
            tags=list(sprite.tags) if hasattr(sprite, 'tags') else [],
            name=sprite.name if hasattr(sprite, 'name') else "",
            grid_x=sprite.x,
            grid_y=sprite.y,
            width=w,
            height=h,
        )

    def _get_sprite_list(self, attr_name: str) -> list:
        """Safely get a list of sprites from a game attribute.

        This method should only be used when attr_name is a real game
        attribute (e.g., 'fhc', 'mou', 'gig', 'zdwrfusvmx') — NOT for
        sprite tag identifiers. For tag-based lookup, use _get_sprites_by_tag().

        Args:
            attr_name: Name of the game attribute to access (must be a
                real attribute on the game object, not a sprite tag).

        Returns:
            List of sprite objects, or empty list if attribute
            doesn't exist or isn't a list.
        """
        try:
            val = getattr(self.game, attr_name)
            if isinstance(val, list):
                return val
        except AttributeError:
            pass
        return []

    def _get_sprites_by_tag(self, tag_name: str) -> list:
        """Find sprites by their tag name (searching through sprite .tags).

        Sprite tags (e.g., '0022vrxelxosfy', '0010xzmuziohuf') are stored
        in each sprite object's .tags property, NOT as direct attributes on
        the game object. Therefore getattr(game, '0022vrxelxosfy') would
        return None/AttributeError, which is why _get_sprite_list() cannot
        be used for tag-based identifiers.

        This method first tries the game's current_level.get_sprites_by_tag()
        (the canonical ARC-AGI-3 API for tag-based sprite lookup), then
        falls back to scanning all game attributes for sprites whose .tags
        contain the given tag_name.

        Use this method instead of _get_sprite_list() whenever the identifier
        looks like a sprite tag (random alphanumeric string, often starting
        with digits like '0022vrxelxosfy') rather than a normal game attribute
        name (short alphabetic strings like 'fhc', 'mou', 'gig').

        Args:
            tag_name: Tag name to search for (e.g., '0022vrxelxosfy',
                '0010xzmuziohuf', 'sys_click', 'lngftsryyw').

        Returns:
            List of sprite objects that have this tag, or empty list if
            none found.
        """
        # Primary approach: use current_level.get_sprites_by_tag() if
        # available. This is the canonical API for tag-based sprite lookup
        # in ARC-AGI-3 games and is used successfully in LS20Adapter.
        try:
            current_level = getattr(self.game, 'current_level', None)
            if current_level is not None and hasattr(current_level, 'get_sprites_by_tag'):
                sprites = current_level.get_sprites_by_tag(tag_name)
                if sprites:
                    return sprites
        except (AttributeError, TypeError, Exception):
            pass

        # Fallback: brute-force scan through all game attributes to find
        # sprites whose .tags property contains the given tag_name.
        # This handles cases where current_level.get_sprites_by_tag() is
        # not available or doesn't find the tag.
        results: list = []
        for attr_name in dir(self.game):
            try:
                val = getattr(self.game, attr_name)
                if isinstance(val, list):
                    for item in val:
                        if hasattr(item, 'tags') and tag_name in item.tags:
                            results.append(item)
                elif hasattr(val, 'tags') and tag_name in val.tags:
                    results.append(val)
            except Exception:
                continue
        return results


# ============================================================================
# LS20Adapter: Oracle adapter for LS20 game
# ============================================================================

class LS20Adapter(OracleAdapter):
    """Oracle adapter for LS20 game.

    LS20 uses heavily obfuscated attribute names:
    - gudziatsk: player sprite
    - gisrhqpee: grid step size
    - plrpelhym: wall sprites (also serves as goals container)
    - gexqgvxkbm: goal sprites (alternative attribute name)
    - fzhmwzexaj: switcher sprites

    This adapter provides standardized access to these obfuscated
    attributes, enabling the PlannerAgent to use Oracle mode for LS20.
    """

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player entity from LS20's gudziatsk attribute.

        Returns:
            GameEntity for the player, or None if not found.
        """
        if hasattr(self.game, 'gudziatsk'):
            p = self.game.gudziatsk
            step = getattr(self.game, 'gisrhqpee', 5)
            w = int(getattr(p, 'width', 1))
            h = int(getattr(p, 'height', 1))
            return GameEntity(
                x=p.x,
                y=p.y,
                tags=list(p.tags) if hasattr(p, 'tags') else [],
                name=getattr(p, 'name', ''),
                grid_x=p.x,
                grid_y=p.y,
                width=w,
                height=h,
            )
        return None

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities from LS20's wall attributes.

        LS20 stores walls via get_sprites_by_tag('ihdgageizm') on
        the current level object. v4.0: Also includes push blocks
        (gbvqrjtaqo) as impassable obstacles.

        Returns:
            List of wall GameEntity objects (walls + push blocks).
        """
        # LS20 walls are accessed via current_level.get_sprites_by_tag
        result_entities: list[GameEntity] = []
        try:
            current_level = self.game.current_level
            if hasattr(current_level, 'get_sprites_by_tag'):
                # Static walls (ihdgageizm)
                wall_sprites = current_level.get_sprites_by_tag("ihdgageizm")
                if wall_sprites:
                    result_entities.extend(self._to_entity(s) for s in wall_sprites)
                # v4.0: Push blocks (gbvqrjtaqo) — impassable obstacles in L3+
                push_block_sprites = current_level.get_sprites_by_tag("gbvqrjtaqo")
                if push_block_sprites:
                    result_entities.extend(self._to_entity(s) for s in push_block_sprites)
                if result_entities:
                    return result_entities
        except (AttributeError, Exception):
            pass
        # Fallback: try common attribute names
        for attr in ['walls', 'wall_sprites']:
            sprites = self._get_sprite_list(attr)
            if sprites:
                return [self._to_entity(s) for s in sprites]
        return []

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal entities from LS20's plrpelhym attribute.

        LS20 stores goals in the plrpelhym list.

        Returns:
            List of goal GameEntity objects.
        """
        # plrpelhym is the goals list in LS20
        sprites = self._get_sprite_list('plrpelhym')
        if sprites:
            return [self._to_entity(s) for s in sprites]
        # Fallback: try common attribute names
        for attr in ['gexqgvxkbm', 'goals', 'goal_sprites']:
            sprites = self._get_sprite_list(attr)
            if sprites:
                return [self._to_entity(s) for s in sprites]
        return []

    @property
    def switchers(self) -> list[GameEntity]:
        """Get all switcher/state-changer entities from LS20.

        Detects all 3 types of state changers:
        - rhsxkxzdjz: rotation changers (cklxociuu cycling)
        - soyhouuebz: color changers (hiaauhahz cycling)
        - ttfwljgohq: shape changers (fwckfzsyc cycling)

        Also checks game-level attributes for backward compatibility.

        Returns:
            List of all switcher GameEntity objects (deduplicated by position).
        """
        result: list[GameEntity] = []
        # Check game-level attributes
        for attr in ['fzhmwzexaj', 'switchers']:
            sprites = self._get_sprite_list(attr)
            if sprites:
                result.extend([self._to_entity(s) for s in sprites])
        # Check level sprite tags — all 3 state changer types
        changer_tags = ['rhsxkxzdjz', 'soyhouuebz', 'ttfwljgohq']
        try:
            current_level = self.game.current_level
            if hasattr(current_level, 'get_sprites_by_tag'):
                for tag in changer_tags:
                    tag_sprites = current_level.get_sprites_by_tag(tag)
                    if tag_sprites:
                        result.extend([self._to_entity(s) for s in tag_sprites])
        except (AttributeError, Exception):
            pass
        # Deduplicate by position
        seen: set[tuple[int, int]] = set()
        unique: list[GameEntity] = []
        for e in result:
            pos = (int(e.x), int(e.y))
            if pos not in seen:
                seen.add(pos)
                unique.append(e)
        return unique

    @property
    def state_changers(self) -> dict[str, list[GameEntity]]:
        """Get categorized state changer entities for LS20.

        Returns a dict with 3 keys mapping to the corresponding
        sprite type. Each changer is walked over by the player to
        trigger automatic state cycling (no ACTION6 needed).

        Returns:
            Dict with keys 'rotation', 'color', 'shape', each containing
            a list of GameEntity objects for that changer type.
        """
        result: dict[str, list[GameEntity]] = {
            'rotation': [],
            'color': [],
            'shape': [],
        }
        tag_map: dict[str, str] = {
            'rhsxkxzdjz': 'rotation',
            'soyhouuebz': 'color',
            'ttfwljgohq': 'shape',
        }
        try:
            current_level = self.game.current_level
            if hasattr(current_level, 'get_sprites_by_tag'):
                for tag, category in tag_map.items():
                    tag_sprites = current_level.get_sprites_by_tag(tag)
                    if tag_sprites:
                        # Deduplicate by position within each category
                        seen: set[tuple[int, int]] = set()
                        for s in tag_sprites:
                            pos = (int(s.x), int(s.y))
                            if pos not in seen:
                                seen.add(pos)
                                entity = self._to_entity(s)
                                # Store the tag in entity.tags for identification
                                if tag not in entity.tags:
                                    entity.tags.append(tag)
                                result[category].append(entity)
        except (AttributeError, Exception):
            pass
        return result

    @property
    def player_state(self) -> dict[str, int]:
        """Get the player's current state indices (rotation/color/shape).

        These indices cycle through their respective value lists when
        the player walks over the corresponding state changer sprite.
        The cycling is automatic (no ACTION needed).

        Returns:
            Dict with keys 'rotation', 'color', 'shape' and their current
            index values. Rotation: 0-3 (0°/90°/180°/270°),
            Color: 0-3 (mapping to color values), Shape: 0-2.
        """
        return {
            'rotation': getattr(self.game, 'cklxociuu', 0),
            'color': getattr(self.game, 'hiaauhahz', 0),
            'shape': getattr(self.game, 'fwckfzsyc', 0),
        }

    @property
    def goal_requirements(self) -> list[dict[str, int]]:
        """Get state requirements for each goal sprite.

        Each goal requires the player to have specific rotation, color,
        and shape values to pass through and be collected. When the
        player's state matches all requirements, the goal sprite allows
        passage and is collected.

        Returns:
            List of dicts, one per goal, with keys 'rotation', 'color',
            'shape' containing the required index values.
        """
        ehwheiwsk = getattr(self.game, 'ehwheiwsk', [])  # rotation requirements
        yjdexjsoa = getattr(self.game, 'yjdexjsoa', [])  # color requirements
        ldxlnycps = getattr(self.game, 'ldxlnycps', [])  # shape requirements

        n_goals = max(len(ehwheiwsk), len(yjdexjsoa), len(ldxlnycps))
        requirements: list[dict[str, int]] = []
        for i in range(n_goals):
            requirements.append({
                'rotation': ehwheiwsk[i] if i < len(ehwheiwsk) else 0,
                'color': yjdexjsoa[i] if i < len(yjdexjsoa) else 0,
                'shape': ldxlnycps[i] if i < len(ldxlnycps) else 0,
            })
        return requirements

    @property
    def coins(self) -> list[GameEntity]:
        """Get coin/step-reset entities from LS20.

        Coins (npxgalaybz tag) reset the step counter when collected,
        giving the player more steps within the level budget.

        Returns:
            List of coin GameEntity objects.
        """
        result: list[GameEntity] = []
        try:
            current_level = self.game.current_level
            if hasattr(current_level, 'get_sprites_by_tag'):
                coin_sprites = current_level.get_sprites_by_tag('npxgalaybz')
                if coin_sprites:
                    seen: set[tuple[int, int]] = set()
                    for s in coin_sprites:
                        pos = (int(s.x), int(s.y))
                        if pos not in seen:
                            seen.add(pos)
                            result.append(self._to_entity(s))
        except (AttributeError, Exception):
            pass
        return result

    @property
    def step_decrement(self) -> int:
        """Get the step decrement value for the current level.

        Each action consumes `step_decrement` units from the step counter.
        Default is 2 (when StepsDecrement is not set in level data).

        Returns:
            Step decrement value (default 2).
        """
        try:
            hgkhqetaxy = self.game.current_level.get_data("StepsDecrement")
            return 2 if hgkhqetaxy is None else hgkhqetaxy
        except (AttributeError, Exception):
            return 2

    @property
    def steps_remaining(self) -> int:
        """Get the number of remaining actions the player can take.

        Calculates: (current_steps in step counter) // step_decrement
        This gives the number of ACTIONS (not step units) remaining.

        Returns:
            Number of remaining actions, or 0 if step counter unavailable.
        """
        try:
            step_counter_ui = getattr(self.game, '_step_counter_ui', None)
            if step_counter_ui is not None:
                current_steps = getattr(step_counter_ui, 'current_steps', 0)
                decrement = self.step_decrement
                return max(0, current_steps // decrement)
            return 0
        except (AttributeError, Exception):
            return 0

    @property
    def step_budget(self) -> int:
        """Get the step budget for the current level.

        LS20 allows osgviligwp steps per level attempt (default 42).
        Coins reset this counter.

        Returns:
            Maximum steps per level attempt (default 42 if not found).
        """
        return getattr(self.game, 'osgviligwp', 42)

    @property
    def state_dimension_sizes(self) -> dict[str, int]:
        """Get the cycling dimension sizes for each state variable.

        Rotation cycles through 4 values (cklxociuu), color through
        len(tnkekoeuk) values, shape through len(ijessuuig) values.

        Returns:
            Dict with keys 'rotation', 'color', 'shape' containing
            the number of distinct values each state cycles through.
        """
        # Rotation: always 4 (0°/90°/180°/270°)
        rot_size = 4
        # Color: len(tnkekoeuk) — color palette length
        color_palette = getattr(self.game, 'tnkekoeuk', None)
        if color_palette is not None:
            try:
                color_size = len(color_palette)
            except (TypeError, Exception):
                color_size = 4
        else:
            color_size = 4
        # Shape: len(ijessuuig) — shape pattern count
        # v4.0: Default changed from 3 to 6 (LS20 actual ijessuuig has 6 indices)
        shape_patterns = getattr(self.game, 'ijessuuig', None)
        if shape_patterns is not None:
            try:
                shape_size = len(shape_patterns)
            except (TypeError, Exception):
                shape_size = 6
        else:
            shape_size = 6
        return {
            'rotation': rot_size,
            'color': color_size,
            'shape': shape_size,
        }

    @property
    def doors(self) -> list[GameEntity]:
        """Get door entities from LS20 that block path until switchers are visited.

        Doors are tagged 'hoswmpiqkw' on current_level sprites.
        These sprites block movement until the player visits a switcher
        (rhsxkxzdjz sprites).

        Returns:
            List of door GameEntity objects.
        """
        result: list[GameEntity] = []
        try:
            current_level = self.game.current_level
            if hasattr(current_level, 'get_sprites_by_tag'):
                door_sprites = current_level.get_sprites_by_tag('hoswmpiqkw')
                if door_sprites:
                    result.extend([self._to_entity(s) for s in door_sprites])
        except (AttributeError, Exception):
            pass
        return result


# ============================================================================
# TR87Adapter: Oracle adapter for TR87 game
# ============================================================================

class TR87Adapter(OracleAdapter):
    """Oracle adapter for TR87 game.

    TR87 is a keyboard game with 6 levels. Its structure:
    - zdwrfusvmx: 22 sprites (likely walls)
    - ztgmtnnufb: 5 sprites (likely goals)
    - zvojhrjxxm: 5 sprites (likely switchers)
    - qvtymdcqear_parts: 2 sprites (likely player parts)
    - upmkivwyrxz / vfpimnmtnta: 128 (grid dimensions)
    - win_score: 6

    The adapter maps these obfuscated names to the unified interface.
    """

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player entity from TR87's qvtymdcqear_parts.

        The player is likely the first element of qvtymdcqear_parts.

        Returns:
            GameEntity for the player, or None if not found.
        """
        # Player is likely one of qvtymdcqear_parts
        parts = self._get_sprite_list('qvtymdcqear_parts')
        if parts:
            p = parts[0]
            return self._to_entity(p)
        return None

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities from TR87's zdwrfusvmx.

        zdwrfusvmx contains 22 sprites that are likely walls.

        Returns:
            List of wall GameEntity objects.
        """
        # zdwrfusvmx: 22 sprites, likely walls
        sprites = self._get_sprite_list('zdwrfusvmx')
        return [self._to_entity(s) for s in sprites]

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal entities from TR87's ztgmtnnufb.

        ztgmtnnufb contains 5 sprites that are likely goals.

        Returns:
            List of goal GameEntity objects.
        """
        # ztgmtnnufb: 5 sprites, likely goals
        sprites = self._get_sprite_list('ztgmtnnufb')
        return [self._to_entity(s) for s in sprites]

    @property
    def switchers(self) -> list[GameEntity]:
        """Get switcher entities from TR87's zvojhrjxxm.

        zvojhrjxxm contains 5 sprites that are likely switchers.

        Returns:
            List of switcher GameEntity objects.
        """
        # zvojhrjxxm: 5 sprites, likely switchers
        sprites = self._get_sprite_list('zvojhrjxxm')
        return [self._to_entity(s) for s in sprites]

    @property
    def grid_size(self) -> int:
        """Get grid size from TR87's upmkivwyrxz attribute.

        Returns:
            Grid dimension (typically 128 for TR87).
        """
        return getattr(self.game, 'upmkivwyrxz', 128)


# ============================================================================
# FT09Adapter: Oracle adapter for FT09 game
# ============================================================================

class FT09Adapter(OracleAdapter):
    """Oracle adapter for FT09 game.

    FT09 is a click-based game with 6 levels. Its structure:
    - fhc: 8 sprites with tags ['Hkx', 'gOi'] (clickable objects)
    - gig: 1 sprite with tag ['bsT'] (special entity)
    - zth: single sprite at (16, 16) with tag 'Ycb' (player/cursor)
    - pdw / zbh: 32 (half-grid dimensions, step=32?)
    - our: 0 (score)

    The adapter maps these to the unified interface. FT09 is a click
    game, so walls are typically empty and goals are the clickable
    objects.
    """

    def __init__(self, game: Any, step: int = 5) -> None:
        """Initialize the FT09 adapter.

        Args:
            game: The env._game object.
            step: Grid step size (overridden by FT09's pdw attribute).
        """
        super().__init__(game, step)
        # FT09 might use different step size based on pdw attribute
        raw_pdw = getattr(game, 'pdw', 32)
        # pdw is typically 32, which means step might be 16 (half)
        # but we use it cautiously
        self._ft09_step: int = max(1, raw_pdw // 2)

    @property
    def grid_size(self) -> int:
        """Get grid size from FT09's pdw attribute.

        FT09 uses a 32x32 game grid with a 64x64 display, so the
        scale factor for coordinate conversion is 2.0. The base
        class returns 64 which would give scale=1.0 (incorrect).

        Returns:
            Grid dimension (32 for FT09, from game.pdw).
        """
        return getattr(self.game, 'pdw', 32)

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player/cursor entity from FT09's zth.

        zth is the single sprite at (16, 16) representing the
        player or cursor in this click-based game.

        Returns:
            GameEntity for the player/cursor, or None if not found.
        """
        # zth is the player/cursor
        if hasattr(self.game, 'zth'):
            p = self.game.zth
            return self._to_entity(p, step=self._ft09_step)
        return None

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities for FT09.

        Click games typically don't have walls, so this returns
        an empty list.

        Returns:
            Empty list (no walls in click games).
        """
        return []

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal entities from FT09's fhc and mou.

        fhc contains Hkx clickable objects and mou contains NTi
        clickable objects. Both are valid click targets.

        Returns:
            List of goal GameEntity objects.
        """
        # Combine Hkx (fhc) and NTi (mou) clickable objects
        sprites = self._get_sprite_list('fhc')
        sprites += self._get_sprite_list('mou')
        return [self._to_entity(s, step=self._ft09_step) for s in sprites]

    @property
    def switchers(self) -> list[GameEntity]:
        """Get switcher entities from FT09's gig.

        gig contains 1 special entity that may act as a switcher.

        Returns:
            List of switcher GameEntity objects.
        """
        # gig: 1 special entity
        sprites = self._get_sprite_list('gig')
        return [self._to_entity(s, step=self._ft09_step) for s in sprites]


# ============================================================================
# KA59Adapter: Oracle adapter for KA59 game (Sokoban push-block game)
# ============================================================================

class KA59Adapter(OracleAdapter):
    """Oracle adapter for KA59 game (v3.18.0).

    KA59 is a Sokoban (push-block) game with:
    - Player: first sprite from 0022vrxelxosfy (target sprites, 3x3)
    - Walls: sprites from 0015qniapgwsvb (inner walls, NOT boundary 0029ifoxxfvvvs)
    - Goals: sprites from 0001uqqokjrptk (goal targets for special blocks)
    - Blocks: sprites from 0010xzmuziohuf (pushable blocks, 5x5) — Sokoban blocks
    - Switchers: 0022vrxelxosfy (ACTION6 switches active target) — all target sprites
    - Enemy: sprite from omeizjufss (chases goal target)
    - Step: 3 (pixel step size for KA59)

    Win condition: all 0010xzmuziohuf overlap with 0022vrxelxosfy AND
                   all 0027jbgxilrocf overlap with 0001uqqokjrptk.
    """

    def __init__(self, game: Any, step: int = 3) -> None:
        """Initialize KA59Adapter with step=3 for Sokoban grid."""
        super().__init__(game, step)

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player entity from KA59's 0022vrxelxosfy (first target sprite).

        Note: '0022vrxelxosfy' is a sprite tag (stored in sprite.tags),
        not a game attribute. Uses _get_sprites_by_tag() for proper lookup.

        Returns:
            GameEntity for the player, or None if not found.
        """
        # 0022vrxelxosfy is a sprite tag, not a game attribute — must use
        # _get_sprites_by_tag() instead of _get_sprite_list()
        sprites = self._get_sprites_by_tag('0022vrxelxosfy')
        if sprites:
            p = sprites[0]
            return self._to_entity(p)
        return None

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities from KA59's 0015qniapgwsvb (inner walls only).

        Note: '0015qniapgwsvb' is a sprite tag, not a game attribute.
        Uses _get_sprites_by_tag() for proper lookup.

        NOTE: Does NOT include 0029ifoxxfvvvs (51×51 boundary walls).
        Inner walls are the ones that affect pathfinding.

        Returns:
            List of wall GameEntity objects (inner walls only).
        """
        # 0015qniapgwsvb is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('0015qniapgwsvb')
        return [self._to_entity(s) for s in sprites]

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal entities from KA59's 0001uqqokjrptk.

        Note: '0001uqqokjrptk' is a sprite tag, not a game attribute.
        Uses _get_sprites_by_tag() for proper lookup.

        Returns:
            List of goal GameEntity objects.
        """
        # 0001uqqokjrptk is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('0001uqqokjrptk')
        return [self._to_entity(s) for s in sprites]

    @property
    def blocks(self) -> list[GameEntity]:
        """Get pushable block entities from KA59's 0010xzmuziohuf (Sokoban blocks).

        Note: '0010xzmuziohuf' is a sprite tag, not a game attribute.
        Uses _get_sprites_by_tag() for proper lookup.

        Returns:
            List of block GameEntity objects (5x5 pushable blocks).
        """
        # 0010xzmuziohuf is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('0010xzmuziohuf')
        return [self._to_entity(s) for s in sprites]

    @property
    def switchers(self) -> list[GameEntity]:
        """Get all switcher entities from KA59's 0022vrxelxosfy.

        ACTION6 click on any 0022vrxelxosfy sprite switches active player.

        Note: '0022vrxelxosfy' is a sprite tag — uses _get_sprites_by_tag().

        Returns:
            List of all target switcher GameEntity objects.
        """
        # 0022vrxelxosfy is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('0022vrxelxosfy')
        return [self._to_entity(s) for s in sprites]

    @property
    def enemy(self) -> Optional[GameEntity]:
        """Get the enemy entity from KA59's omeizjufss.

        The enemy sprite chases the goal target. If enemy reaches goal,
        the game is lost.

        Note: 'omeizjufss' is a sprite tag — uses _get_sprites_by_tag().

        Returns:
            GameEntity for the enemy, or None if not found.
        """
        # omeizjufss is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('omeizjufss')
        if sprites:
            return self._to_entity(sprites[0])
        return None

    @property
    def grid_size(self) -> int:
        """Get grid size for KA59 (typically 153 = 51×3 step_size).

        Returns:
            Grid dimension from game attribute, or 153 as default.
        """
        return getattr(self.game, 'upmkivwyrxz', 153)


# ============================================================================
# AR25Adapter: Oracle adapter for AR25 game (mirror-reflection coverage)
# ============================================================================

class AR25Adapter(OracleAdapter):
    """Oracle adapter for AR25 game (v3.18.0).

    AR25 is a mirror-reflection coverage game:
    - Player: first sprite from 0006lxjtqggkmi (click/select, 3x3)
    - Walls: empty (no walls in coverage game)
    - Goals: sprites from 0001sruqbuvukh (coins to cover, 5 sprites)
    - Mirrors: sprites from 0003uqrdzdofso
    - pieces_vertical: 0054kgxrvfihgm — only moves UP/DOWN
    - pieces_horizontal: 0002nuguepuujf — only moves LEFT/RIGHT
    - Switchers: sprites with ACTION5/ACTION6/ACTION7 capability
    - Step: 1

    Win condition: all 0001sruqbuvukh (coins) have non-negative value
    in merged grid naxbskjmlg().
    """

    def __init__(self, game: Any, step: int = 1) -> None:
        """Initialize AR25Adapter with step=1 for fine-grained movement."""
        super().__init__(game, step)

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player/cursor entity from AR25's 0006lxjtqggkmi.

        Note: '0006lxjtqggkmi' is a sprite tag, not a game attribute.
        Uses _get_sprites_by_tag() for proper lookup.

        Returns:
            GameEntity for the player/cursor, or None if not found.
        """
        # 0006lxjtqggkmi is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('0006lxjtqggkmi')
        if sprites:
            p = sprites[0]
            return self._to_entity(p)
        return None

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities for AR25 (coverage game has no walls).

        Returns:
            Empty list (no walls in coverage game).
        """
        return []

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal entities from AR25's 0001sruqbuvukh (coins to cover).

        Note: '0001sruqbuvukh' is a sprite tag, not a game attribute.
        Uses _get_sprites_by_tag() for proper lookup.

        Returns:
            List of coin GameEntity objects.
        """
        # 0001sruqbuvukh is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('0001sruqbuvukh')
        return [self._to_entity(s) for s in sprites]

    @property
    def mirrors(self) -> list[GameEntity]:
        """Get mirror entities from AR25's 0003uqrdzdofso.

        Pieces with this tag get reflected through mirror axes when moved.

        Note: '0003uqrdzdofso' is a sprite tag — uses _get_sprites_by_tag().

        Returns:
            List of mirror GameEntity objects.
        """
        # 0003uqrdzdofso is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('0003uqrdzdofso')
        return [self._to_entity(s) for s in sprites]

    @property
    def pieces_vertical(self) -> list[GameEntity]:
        """Get vertical-only piece entities from AR25's 0054kgxrvfihgm.

        These pieces can only move UP/DOWN (constrained by vertical mirror tag).

        Note: '0054kgxrvfihgm' is a sprite tag — uses _get_sprites_by_tag().

        Returns:
            List of vertical-only piece GameEntity objects.
        """
        # 0054kgxrvfihgm is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('0054kgxrvfihgm')
        return [self._to_entity(s) for s in sprites]

    @property
    def pieces_horizontal(self) -> list[GameEntity]:
        """Get horizontal-only piece entities from AR25's 0002nuguepuujf.

        These pieces can only move LEFT/RIGHT (constrained by horizontal mirror tag).

        Note: '0002nuguepuujf' is a sprite tag — uses _get_sprites_by_tag().

        Returns:
            List of horizontal-only piece GameEntity objects.
        """
        # 0002nuguepuujf is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('0002nuguepuujf')
        return [self._to_entity(s) for s in sprites]

    @property
    def switchers(self) -> list[GameEntity]:
        """Get switcher entities from AR25 (sprites with ACTION5/6/7 capability).

        ACTION5: cycle through movable pieces.
        ACTION6: click to select piece at grid position.
        ACTION7: undo last move.

        Note: Both '0006lxjtqggkmi' and 'sys_click' are sprite tags,
        not game attributes. Uses _get_sprites_by_tag() for proper lookup.

        Returns:
            List of switcher GameEntity objects.
        """
        result: list[GameEntity] = []
        # 0006lxjtqggkmi and sys_click are sprite tags — use
        # _get_sprites_by_tag() instead of _get_sprite_list()
        for tag in ['0006lxjtqggkmi', 'sys_click']:
            sprites = self._get_sprites_by_tag(tag)
            if sprites:
                result.extend([self._to_entity(s) for s in sprites])
        # Deduplicate by position
        seen: set[tuple[int, int]] = set()
        unique: list[GameEntity] = []
        for e in result:
            pos = (int(e.x), int(e.y))
            if pos not in seen:
                seen.add(pos)
                unique.append(e)
        return unique

    @property
    def grid_size(self) -> int:
        """Get grid size for AR25 (typically 64).

        Returns:
            Grid dimension from game attribute, or 64 as default.
        """
        return getattr(self.game, 'upmkivwyrxz', 64)


# ============================================================================
# TN36Adapter: Oracle adapter for TN36 game (click-programming state machine)
# ============================================================================

class TN36Adapter(OracleAdapter):
    """Oracle adapter for TN36 game (v3.18.0).

    TN36 is a click-programming game with state machines:
    - Player: None (click-only game, no player entity)
    - Walls: None (no walls in click game)
    - Goals: sprites from kntfjgchzd (clickable sprites) — also has sys_click tag
    - Switchers: empty (no switcher entities)
    - Step: 1
    - state_machines: mvqheosngn (left) and bzirenxmrg (right)
    - Win check: bzirenxmrg.vklyonlcrw == True

    When you click a kntfjgchzd sprite → egjahxmvrj(sprite) sets first
    sprite to that position/rotation/program, then animates it moving.
    Win: bzirenxmrg.vklyonlcrw becomes True (second sprite reaches state).
    """

    def __init__(self, game: Any, step: int = 1) -> None:
        """Initialize TN36Adapter with step=1 for click positions."""
        super().__init__(game, step)

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player entity for TN36 (None — click-only game).

        Returns:
            None (TN36 has no player entity — only click actions).
        """
        return None

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities for TN36 (None — click-only game).

        Returns:
            Empty list (no walls in click game).
        """
        return []

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal/clickable entities from TN36's kntfjgchzd.

        These are the clickable sprites that trigger state machine transitions.

        Note: 'kntfjgchzd', 'Maidxz', 'qqifsatqdo', and 'sys_click' are
        all sprite tags, not game attributes. Uses _get_sprites_by_tag().

        Returns:
            List of clickable goal GameEntity objects.
        """
        # kntfjgchzd is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('kntfjgchzd')
        if sprites:
            return [self._to_entity(s) for s in sprites]
        # Fallback tag names — all are sprite tags, not game attributes
        for tag_name in ('Maidxz', 'qqifsatqdo', 'sys_click'):
            sprites = self._get_sprites_by_tag(tag_name)
            if sprites:
                return [self._to_entity(s) for s in sprites]
        return []

    @property
    def switchers(self) -> list[GameEntity]:
        """Get switcher entities for TN36 (empty — no switchers).

        Returns:
            Empty list (TN36 uses only click actions, no switchers).
        """
        return []

    @property
    def state_machines(self) -> dict[str, Optional[GameEntity]]:
        """Get state machine entities for TN36.

        mvqheosngn: left state machine (first sprite, mkfavqnwxy tag).
        bzirenxmrg: right state machine (goal-check sprite, second sprite).

        Returns:
            Dict with keys 'left' and 'right' mapping to GameEntity or None.
        """
        left_sm = None
        right_sm = None

        # Try to get state machines from game attributes
        # mkfavqnwxy is a sprite tag — use _get_sprites_by_tag()
        mkfav_sprites = self._get_sprites_by_tag('mkfavqnwxy')
        if mkfav_sprites and len(mkfav_sprites) >= 1:
            left_sm = self._to_entity(mkfav_sprites[0])
        if mkfav_sprites and len(mkfav_sprites) >= 2:
            right_sm = self._to_entity(mkfav_sprites[1])

        return {'left': left_sm, 'right': right_sm}

    def is_won(self) -> bool:
        """Check TN36-specific win condition: bzirenxmrg.vklyonlcrw == True.

        Accesses the game's internal state machine to check if the
        right sprite has reached its target state.

        Returns:
            True if bzirenxmrg.vklyonlcrw is True, False otherwise.
        """
        try:
            fdksqlmpki = getattr(self.game, 'fdksqlmpki', None)
            if fdksqlmpki is not None:
                bzirenxmrg = getattr(fdksqlmpki, 'bzirenxmrg', None)
                if bzirenxmrg is not None:
                    vklyonlcrw = getattr(bzirenxmrg, 'vklyonlcrw', None)
                    if vklyonlcrw is True or vklyonlcrw == 1:
                        return True
        except (AttributeError, TypeError):
            pass
        return False

    @property
    def grid_size(self) -> int:
        """Get grid size for TN36 (typically 64).

        Returns:
            Grid dimension from game attribute, or 64 as default.
        """
        return getattr(self.game, 'upmkivwyrxz', 64)


# ============================================================================
# SB26Adapter: Oracle adapter for SB26 game (click-to-match pattern)
# ============================================================================

class SB26Adapter(OracleAdapter):
    """Oracle adapter for SB26 game.

    SB26 is a click-based matching/pattern game:
    - Player: None (click-only game, no player entity)
    - Walls: None (no walls in click game)
    - Goals/click targets: sprites with tag 'lngftsryyw' (4 blocks, 6x6)
    - Slots: sprites with tag 'susublrply' (8 slots to fill)
    - sys_click sprites: UI interaction sprites
    - Available actions: ACTION5 (select), ACTION6/7 (click)
    - Step: 1

    Win condition: match lngftsryyw blocks to susublrply slots correctly.
    """

    def __init__(self, game: Any, step: int = 1) -> None:
        """Initialize SB26Adapter with step=1 for click positions."""
        super().__init__(game, step)

    @property
    def player(self) -> Optional[GameEntity]:
        """Get the player entity for SB26 (None — click-only game).

        Returns:
            None (SB26 has no player entity — only click actions).
        """
        return None

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities for SB26 (None — click-only game).

        Returns:
            Empty list (no walls in click game).
        """
        return []

    @property
    def goals(self) -> list[GameEntity]:
        """Get goal/clickable entities from SB26's lngftsryyw (4 blocks).

        Note: 'lngftsryyw' is a sprite tag, not a game attribute.
        Uses _get_sprites_by_tag() for proper lookup.

        Returns:
            List of clickable block GameEntity objects (4 blocks).
        """
        # lngftsryyw is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('lngftsryyw')
        return [self._to_entity(s) for s in sprites]

    @property
    def slots(self) -> list[GameEntity]:
        """Get slot entities from SB26's susublrply (8 slots).

        These are the 8 slot positions where blocks need to be placed.

        Note: 'susublrply' is a sprite tag — uses _get_sprites_by_tag().

        Returns:
            List of slot GameEntity objects (8 slots).
        """
        # susublrply is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('susublrply')
        return [self._to_entity(s) for s in sprites]

    @property
    def switchers(self) -> list[GameEntity]:
        """Get sys_click switcher entities from SB26.

        Note: 'sys_click' is a sprite tag — uses _get_sprites_by_tag().

        Returns:
            List of sys_click GameEntity objects.
        """
        # sys_click is a sprite tag — use _get_sprites_by_tag()
        sprites = self._get_sprites_by_tag('sys_click')
        return [self._to_entity(s) for s in sprites]

    @property
    def grid_size(self) -> int:
        """Get grid size for SB26 (typically 64).

        Returns:
            Grid dimension from game attribute, or 64 as default.
        """
        return getattr(self.game, 'upmkivwyrxz', 64)


# ============================================================================
# Registry and factory functions
# ============================================================================

# Registry of game adapters
ADAPTER_REGISTRY: dict[str, type[OracleAdapter]] = {
    'ls20': LS20Adapter,
    'tr87': TR87Adapter,
    'ft09': FT09Adapter,
    'ka59': KA59Adapter,
    'ar25': AR25Adapter,
    'tn36': TN36Adapter,
    'sb26': SB26Adapter,
}


def get_oracle_adapter(
    game_id: str,
    game: Any,
    step: int = 5,
) -> Optional[OracleAdapter]:
    """Get an Oracle adapter for a specific game.

    Looks up the game ID in the adapter registry and creates an
    instance if found. Game IDs may include version suffixes
    (e.g., "ls20-9607627b"), which are stripped before lookup.

    Args:
        game_id: Game identifier (e.g., "ls20", "tr87-abc123").
        game: The env._game object.
        step: Grid step size.

    Returns:
        OracleAdapter instance, or None if no adapter exists for
        the given game ID.
    """
    base_id = game_id.split('-')[0]
    adapter_class = ADAPTER_REGISTRY.get(base_id)
    if adapter_class is not None:
        return adapter_class(game, step)
    return None


def auto_detect_adapter(game: Any, game_id: Optional[str] = None) -> Optional[OracleAdapter]:
    """Auto-detect game type and return appropriate adapter.

    Tries each adapter's detection method to find the best match
    based on the game object's available attributes. This is used
    when the game ID is unknown or not in the registry.

    Detection order:
        1. LS20: has 'gudziatsk' attribute
        2. TR87: has 'zdwrfusvmx' and 'qvtymdcqear_parts' attributes
        3. FT09: has 'zth' and 'fhc' attributes
        4. KA59: has '0010xzmuziohuf' and '0022vrxelxosfy' sprite tags
        5. AR25: has '0001sruqbuvukh' and '0006lxjtqggkmi' sprite tags
        6. TN36: has 'kntfjgchzd' sprite tag
        7. SB26: has 'lngftsryyw' and 'susublrply' sprite tags
        8. Universal: fallback for any game with env._game access.
           Uses UniversalOracleAdapter to auto-discover entities.
           When game_id is provided, uses game_configs for entity detection.

    Args:
        game: The env._game object.
        game_id: Optional game identifier for config-based detection.

    Returns:
        OracleAdapter instance (never None if UniversalOracleAdapter
        is available), or None if the universal adapter cannot be
        imported.
    """
    # Check LS20
    if hasattr(game, 'gudziatsk'):
        return LS20Adapter(game)

    # Check TR87
    if hasattr(game, 'zdwrfusvmx') and hasattr(game, 'qvtymdcqear_parts'):
        return TR87Adapter(game)

    # Check FT09
    if hasattr(game, 'zth') and hasattr(game, 'fhc'):
        return FT09Adapter(game)

    # Check KA59 — Sokoban game has pushable blocks (0010xzmuziohuf tag)
    # and target sprites (0022vrxelxosfy tag)
    try:
        cl = getattr(game, 'current_level', None)
        if cl is not None and hasattr(cl, 'get_sprites_by_tag'):
            ka59_blocks = cl.get_sprites_by_tag('0010xzmuziohuf')
            ka59_targets = cl.get_sprites_by_tag('0022vrxelxosfy')
            if ka59_blocks and ka59_targets:
                return KA59Adapter(game)
    except (AttributeError, TypeError):
        pass

    # Check AR25 — Coverage game has coins (0001sruqbuvukh tag)
    # and selectable pieces (0006lxjtqggkmi tag)
    try:
        cl = getattr(game, 'current_level', None)
        if cl is not None and hasattr(cl, 'get_sprites_by_tag'):
            ar25_coins = cl.get_sprites_by_tag('0001sruqbuvukh')
            ar25_select = cl.get_sprites_by_tag('0006lxjtqggkmi')
            if ar25_coins and ar25_select:
                return AR25Adapter(game)
    except (AttributeError, TypeError):
        pass

    # Check TN36 — Click-programming game has kntfjgchzd tag sprites
    try:
        cl = getattr(game, 'current_level', None)
        if cl is not None and hasattr(cl, 'get_sprites_by_tag'):
            tn36_clickables = cl.get_sprites_by_tag('kntfjgchzd')
            if tn36_clickables:
                return TN36Adapter(game)
    except (AttributeError, TypeError):
        pass

    # Check SB26 — Click-to-match game has lngftsryyw blocks and
    # susublrply slots
    try:
        cl = getattr(game, 'current_level', None)
        if cl is not None and hasattr(cl, 'get_sprites_by_tag'):
            sb26_blocks = cl.get_sprites_by_tag('lngftsryyw')
            sb26_slots = cl.get_sprites_by_tag('susublrply')
            if sb26_blocks and sb26_slots:
                return SB26Adapter(game)
    except (AttributeError, TypeError):
        pass

    # Fallback: Universal adapter for any game with env._game access.
    # The UniversalOracleAdapter auto-discovers entities by scanning
    # game attributes, enabling Oracle mode for all 25 games.
    # When game_id is provided, uses game_configs.py for entity tags.
    try:
        from .universal_oracle_adapter import UniversalOracleAdapter
        return UniversalOracleAdapter(game, game_id=game_id)
    except ImportError:
        pass

    return None


def get_universal_adapter(
    game: Any,
    step: int = 5,
) -> Optional[OracleAdapter]:
    """Get a UniversalOracleAdapter for any game.

    The UniversalOracleAdapter auto-discovers game entities by scanning
    env._game attributes, enabling Oracle mode for all 25 games without
    requiring game-specific adapters.

    Args:
        game: The env._game object.
        step: Grid step size for block alignment.

    Returns:
        UniversalOracleAdapter instance, or None if the universal
        adapter module cannot be imported.
    """
    try:
        from .universal_oracle_adapter import UniversalOracleAdapter
        return UniversalOracleAdapter(game, step)
    except ImportError:
        return None


# Update ADAPTER_REGISTRY with UniversalOracleAdapter (lazy import to
# avoid circular dependency: universal_oracle_adapter imports from
# this module at the top level).
try:
    from .universal_oracle_adapter import UniversalOracleAdapter as _UniversalAdapter
    ADAPTER_REGISTRY['universal'] = _UniversalAdapter
except ImportError:
    pass
