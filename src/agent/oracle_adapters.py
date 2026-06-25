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
    """

    x: int
    y: int
    tags: list[str]
    name: str = ""
    grid_x: int = 0
    grid_y: int = 0


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
            GameEntity with position and tag information.
        """
        s = step if step is not None else self.step
        return GameEntity(
            x=sprite.x,
            y=sprite.y,
            tags=list(sprite.tags) if hasattr(sprite, 'tags') else [],
            name=sprite.name if hasattr(sprite, 'name') else "",
            grid_x=sprite.x,
            grid_y=sprite.y,
        )

    def _get_sprite_list(self, attr_name: str) -> list:
        """Safely get a list of sprites from a game attribute.

        Args:
            attr_name: Name of the game attribute to access.

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
            return GameEntity(
                x=p.x,
                y=p.y,
                tags=list(p.tags) if hasattr(p, 'tags') else [],
                name=getattr(p, 'name', ''),
                grid_x=p.x,
                grid_y=p.y,
            )
        return None

    @property
    def walls(self) -> list[GameEntity]:
        """Get wall entities from LS20's wall attributes.

        LS20 stores walls via get_sprites_by_tag('ihdgageizm') on
        the current level object.

        Returns:
            List of wall GameEntity objects.
        """
        # LS20 walls are accessed via current_level.get_sprites_by_tag
        try:
            current_level = self.game.current_level
            if hasattr(current_level, 'get_sprites_by_tag'):
                sprites = current_level.get_sprites_by_tag("ihdgageizm")
                if sprites:
                    return [self._to_entity(s) for s in sprites]
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
        """Get switcher entities from LS20's switcher attributes.

        Returns:
            List of switcher GameEntity objects.
        """
        result: list[GameEntity] = []
        for attr in ['fzhmwzexaj', 'switchers']:
            sprites = self._get_sprite_list(attr)
            if sprites:
                result.extend([self._to_entity(s) for s in sprites])
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
# Registry and factory functions
# ============================================================================

# Registry of game adapters
ADAPTER_REGISTRY: dict[str, type[OracleAdapter]] = {
    'ls20': LS20Adapter,
    'tr87': TR87Adapter,
    'ft09': FT09Adapter,
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
        4. Universal: fallback for any game with env._game access.
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
