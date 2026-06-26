"""Game-specific configuration for all 25 ARC-AGI-3 games.

This module provides entity tag mappings, action types, and game mechanics
metadata extracted from source code analysis. Used by UniversalOracleAdapter
to correctly identify player/goal/wall/click-target sprites per game.

Attributes:
    GAME_CONFIGS: Dict mapping game_id to GameConfig with entity tags and metadata.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GameConfig:
    """Configuration for a single ARC-AGI-3 game.

    Attributes:
        game_type: "keyboard", "click", or "keyboard+click"
        player_tag: Tag name for the player sprite(s).
        goal_tag: Tag name for goal/exit sprites.
        wall_tag: Tag name for wall/obstacle sprites.
        switcher_tag: Tag name for switcher/interaction sprites.
        click_target_tag: Tag name for clickable sprites (click games).
        click_target_tags: Multiple click target tags (if needed).
        map_tag: Tag name for the map/background sprite (map-based collision).
        available_actions: List of available GameAction IDs.
        action_mapping: Dict mapping action ID to direction/function.
        grid_size: Grid size (32 or 64).
        notes: Additional game-specific notes.
    """
    game_type: str = "unknown"
    player_tag: Optional[str] = None
    goal_tag: Optional[str] = None
    wall_tag: Optional[str] = None
    switcher_tag: Optional[str] = None
    click_target_tag: Optional[str] = None
    click_target_tags: Optional[list[str]] = None
    map_tag: Optional[str] = None
    available_actions: list[int] = field(default_factory=list)
    action_mapping: dict[int, str] = field(default_factory=dict)
    grid_size: int = 64
    notes: str = ""


# Standard action mapping for keyboard games
# ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT, ACTION5=special, ACTION6=click, ACTION7=special2
STD_KB_MAP = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPECIAL", 6: "CLICK", 7: "SPECIAL2"}


GAME_CONFIGS: dict[str, GameConfig] = {
    # === KEYBOARD GAMES (6) ===
    "ls20": GameConfig(
        game_type="keyboard",
        player_tag="sfqyzhzkij",
        goal_tag="eqatonpohu",
        wall_tag="ihdgageizm",
        switcher_tag="rhsxkxzdjz",
        available_actions=[1, 2, 3, 4],
        action_mapping=STD_KB_MAP,
        grid_size=64,
        notes="LS20: Visit all switchers to unlock goals, then reach goal. 116 wall sprites.",
    ),
    "tu93": GameConfig(
        game_type="keyboard",
        player_tag="0017unajnymcki",
        goal_tag="0015msvpvzxhqf",
        wall_tag=None,  # No wall tag - uses map-based collision
        map_tag="0005uvnhiglpvh",
        available_actions=[1, 2, 3, 4],
        action_mapping=STD_KB_MAP,
        grid_size=64,
        notes="TU93: Map-based collision. Player at (3,3), goal at (33,33). Map sprite 33x33 with colors 0,2 (2=wall).",
    ),
    "tr87": GameConfig(
        game_type="keyboard",
        player_tag="qvtymdcqear_parts",  # Game attribute (not a tag) — used for sprite size detection
        goal_tag="ztgmtnnufb",  # Game attribute for goals
        wall_tag="zdwrfusvmx",  # Game attribute for walls
        available_actions=[1, 2, 3, 4],
        action_mapping=STD_KB_MAP,
        grid_size=128,
        notes="TR87: Collect all ztgmtnnufb goals while avoiding zdwrfusvmx walls. 22 walls + 5 goals. Grid=128x128.",
    ),
    "re86": GameConfig(
        game_type="keyboard",
        player_tag="0031cppcuvqlbi",
        goal_tag=None,  # Need to find win condition - likely reaching specific position
        wall_tag="0001jdldomszsf",  # 64x64 background with colors 4,9,11
        map_tag="0054xnsuqceejm",  # Alternative map sprite
        available_actions=[1, 2, 3, 4, 5],
        action_mapping={**STD_KB_MAP, 5: "SWITCH_PLAYER"},
        grid_size=64,
        notes="RE86: ACTION5 switches active player. 2 player sprites (23x23). Win condition TBD.",
    ),
    "g50t": GameConfig(
        game_type="keyboard",
        player_tag="twyixucrqi",  # This is a game attribute referencing the player sprite
        goal_tag=None,  # Need to find win condition
        wall_tag="uxqxubarib",  # 61x61 background
        available_actions=[1, 2, 3, 4, 5],
        action_mapping={**STD_KB_MAP, 5: "SPECIAL"},
        grid_size=64,
        notes="G50T: Uses evgpfjbmvf constants class for tag names. Player=qftsebtxuc(moved). ACTION5=special.",
    ),
    "wa30": GameConfig(
        game_type="keyboard",
        player_tag="wbmdvjhthc",
        goal_tag="geezpjgiyd",
        wall_tag=None,  # No explicit wall tag
        available_actions=[1, 2, 3, 4, 5],
        action_mapping={**STD_KB_MAP, 5: "SPECIAL"},
        grid_size=64,
        notes="WA30: Player=wbmdvjhthc(4x4), goals=geezpjgiyd(4x4, 3 sprites). ACTION5=special.",
    ),

    # === CLICK GAMES (7) ===
    "ft09": GameConfig(
        game_type="click",
        click_target_tags=["Hkx", "NTi"],  # fhc + mou sprites
        available_actions=[6],
        action_mapping={6: "CLICK"},
        grid_size=32,
        notes="FT09: Color constraint puzzle. Hkx(clickable cells) + NTi(multi-effect cells) + bsT(constraints). Already has FT09Adapter.",
    ),
    "vc33": GameConfig(
        game_type="click",
        click_target_tag="0022jvmlspyigc",
        available_actions=[6],
        action_mapping={6: "CLICK"},
        grid_size=64,
        notes="VC33: Click on 0022jvmlspyigc sprites (2x2, 2 sprites at (30,12),(30,16)).",
    ),
    "s5i5": GameConfig(
        game_type="click",
        click_target_tag="0001qwdmnlybkb",
        available_actions=[6],
        action_mapping={6: "CLICK"},
        grid_size=64,
        notes="S5I5: Click on 0001qwdmnlybkb sprites (3x9, 4 sprites). Also has 0064ocqkuqacti and 0087vvmblxkzdi.",
    ),
    "su15": GameConfig(
        game_type="click",
        click_target_tag="rgjznrcin",
        available_actions=[6, 7],
        action_mapping={6: "CLICK", 7: "CLICK2"},
        grid_size=64,
        notes="SU15: Click game with ACTION6 and ACTION7. rgjznrcin(3x3) + ooutlqdaq(4x4) + xkstxyqbs(9x9).",
    ),
    "lp85": GameConfig(
        game_type="click",
        click_target_tag="sys_click",
        click_target_tags=["button_A_L", "button_A_R"],
        goal_tag="goal",
        available_actions=[6],
        action_mapping={6: "CLICK"},
        grid_size=64,
        notes="LP85: Button puzzle. Click left/right buttons to move tiles to goal. Has explicit 'goal' tag! 19 tiles.",
    ),
    "r11l": GameConfig(
        game_type="click",
        click_target_tag="sys_click",
        available_actions=[6],
        action_mapping={6: "CLICK"},
        grid_size=64,
        notes="R11L: Click on sys_click sprites (5x5, 2 sprites at (5,34),(25,57)).",
    ),
    "tn36": GameConfig(
        game_type="click",
        click_target_tag="Maidxz",
        click_target_tags=["Maidxz", "sys_click"],
        available_actions=[6],
        action_mapping={6: "CLICK"},
        grid_size=64,
        notes="TN36: Click on Maidxz sprites (5x3, 11 sprites). Complex game with 12 tag types.",
    ),

    # === KEYBOARD+CLICK GAMES (12) ===
    "ar25": GameConfig(
        game_type="keyboard+click",
        player_tag="0006lxjtqggkmi",
        click_target_tag="sys_click",
        available_actions=[1, 2, 3, 4, 5, 6, 7],
        action_mapping={1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SELECT", 6: "CLICK", 7: "REFLECT"},
        grid_size=64,
        notes="AR25: 0006lxjtqggkmi is both player and sys_click (3x3 sprite). Mirror/reflection game with ACTION5/6/7.",
    ),
    "cn04": GameConfig(
        game_type="click",
        click_target_tag="sys_click",
        available_actions=[],
        action_mapping={},
        grid_size=64,
        notes="CN04: Click on sys_click sprites (5x6, 2 sprites). Moved on ACTION2.",
    ),
    "dc22": GameConfig(
        game_type="click",
        click_target_tag="buezna",
        click_target_tags=["buezna", "sys_click"],
        available_actions=[],
        action_mapping={},
        grid_size=64,
        notes="DC22: Click on buezna sprites (13x5). 10 tag types. jfva moved on ACTION2.",
    ),
    "ka59": GameConfig(
        game_type="keyboard+click",
        player_tag="0022vrxelxosfy",
        click_target_tag="sys_click",
        available_actions=[1, 2, 3, 4, 6],
        action_mapping=STD_KB_MAP,
        grid_size=64,
        notes="KA59: Keyboard+click. Player=0022vrxelxosfy(3x3, 2 sprites). ACTION6=click. Also has Enemy tag.",
    ),
    "m0r0": GameConfig(
        game_type="keyboard+click",
        player_tag="wahtyt",  # 11x11 sprite
        click_target_tag="sys_click",
        click_target_tags=["sys_click", "fucr"],
        available_actions=[1, 2, 3, 4, 5, 6],
        action_mapping=STD_KB_MAP,
        grid_size=64,
        notes="M0R0: Keyboard+click. fucr/sys_click moved on ACTION2. wahtyt is 11x11 level background.",
    ),
    "sp80": GameConfig(
        game_type="click",
        click_target_tag="plzwjbfyfli",
        click_target_tags=["plzwjbfyfli", "sys_click"],
        available_actions=[],
        action_mapping={},
        grid_size=64,
        notes="SP80: Click on plzwjbfyfli sprites (5x1). 6 tag types. plzwjbfyfli moved on ACTION2.",
    ),
    "sb26": GameConfig(
        game_type="click",
        click_target_tag="lngftsryyw",
        click_target_tags=["lngftsryyw", "sys_click"],
        available_actions=[5, 6, 7],
        action_mapping={5: "SELECT", 6: "CLICK", 7: "CLICK2"},
        grid_size=64,
        notes="SB26: Click on lngftsryyw sprites (6x6, 4 sprites). ACTION5=select, ACTION6/7=click. 8 susublrply sprites.",
    ),
    "sc25": GameConfig(
        game_type="keyboard+click",
        click_target_tag="sys_click",
        available_actions=[1, 2, 3, 4, 6],
        action_mapping=STD_KB_MAP,
        grid_size=64,
        notes="SC25: Keyboard+click. sys_click sprites (3x3, 9 sprites). clcbko sprites (6x6, 9 sprites). lyhbotskgaq has 26 items.",
    ),
    "sk48": GameConfig(
        game_type="keyboard+click",
        click_target_tag="sys_click",
        click_target_tags=["epdquznwmq", "sys_click"],
        available_actions=[1, 2, 3, 4, 6, 7],
        action_mapping=STD_KB_MAP,
        grid_size=64,
        notes="SK48: Keyboard+click. epdquznwmq/sys_click (6x6, 2 sprites). elmjchdqcn(6x6, 6 sprites). irkeobngyh(2x8, 4 sprites).",
    ),
    "bp35": GameConfig(
        game_type="keyboard",
        player_tag=None,  # 0 tags at level 0 - needs dynamic detection
        available_actions=[3, 4, 6, 7],
        action_mapping={3: "LEFT", 4: "RIGHT", 6: "CLICK", 7: "SPECIAL2"},
        grid_size=64,
        notes="BP35: 0 tags at level 0. Actions [3,4,6,7]. Sprites may load after first action. Needs dynamic detection.",
    ),
    "cd82": GameConfig(
        game_type="keyboard",
        player_tag=None,  # 0 tags at level 0
        available_actions=[1, 2, 3, 4, 5, 6],
        action_mapping=STD_KB_MAP,
        grid_size=64,
        notes="CD82: 0 tags at level 0. Actions [1-6]. ACTION6 has click data. Needs dynamic detection.",
    ),
    "lf52": GameConfig(
        game_type="keyboard",
        player_tag=None,  # 0 tags at level 0
        available_actions=[1, 2, 3, 4, 6, 7],
        action_mapping={**STD_KB_MAP, 7: "UNDO"},
        grid_size=64,
        notes="LF52: 0 tags at level 0. Actions [1,2,3,4,6,7]. ACTION7=UNDO. Needs dynamic detection.",
    ),
}


def get_config(game_id: str) -> GameConfig | None:
    """Get game configuration by game ID.

    Args:
        game_id: The game identifier (e.g. "ls20", "ft09").

    Returns:
        GameConfig if found, None otherwise.
    """
    return GAME_CONFIGS.get(game_id)


def is_keyboard_game(game_id: str) -> bool:
    """Check if a game uses keyboard controls.

    Args:
        game_id: The game identifier.

    Returns:
        True if the game is keyboard or keyboard+click type.
    """
    cfg = GAME_CONFIGS.get(game_id)
    if cfg is None:
        return False
    return cfg.game_type in ("keyboard", "keyboard+click")


def is_click_game(game_id: str) -> bool:
    """Check if a game uses click controls.

    Args:
        game_id: The game identifier.

    Returns:
        True if the game is click or keyboard+click type.
    """
    cfg = GAME_CONFIGS.get(game_id)
    if cfg is None:
        return False
    return cfg.game_type in ("click", "keyboard+click")


def get_click_target_tags(game_id: str) -> list[str]:
    """Get the list of click target tags for a game.

    Args:
        game_id: The game identifier.

    Returns:
        List of tag names for clickable sprites.
    """
    cfg = GAME_CONFIGS.get(game_id)
    if cfg is None:
        return ["sys_click"]
    if cfg.click_target_tags:
        return cfg.click_target_tags
    if cfg.click_target_tag:
        return [cfg.click_target_tag]
    return ["sys_click"]
