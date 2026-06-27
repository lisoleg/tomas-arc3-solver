# -*- coding: utf-8 -*-
"""SBInjector — Per-game structural hint injector for the universal pipeline.

Each ARC-3 game registers a preamble function that returns an SBInjector
instance. The pipeline reads these parameters to drive strategy selection,
including coset search scope, symmetry priors, Ramanujan Machine toggle,
temporal window for EML perception, verification tolerance, sporadic
group routing, and κ-Phase physics primitives.

IDO Correspondence:
    - coset_filter = UV→IR dimension selection priority
    - symmetry_hint = D4/Monster spacetime symmetry prior
    - enable_rm = Ramanujan Machine integer relation detection
    - time_window = EML temporal edge preservation (interactive tasks)
    - eps_factor = GaussEx verification tolerance multiplier
    - sporadic_pref = Sporadic group DSL routing preference
    - physics_primitives = κ-Phase native operators for Dead-Zero pruning

TOMAS Correspondence:
    - SBInjector = Phase I→II→III parameter injection

κ-Phase物理原语 (IDO/TOMAS):
    物理直觉的本质是κ-相位感知 — ARC-AGI-3智能体必须识别网格变换中的κ-相位一致性
    四类原语: newton_push(推箱), mirror_geo(反射), dfa(因果状态机), poset(偏序排序)
    注入方式: SBInjector.physics_primitives → κ-Snap搜索调用原语做Dead-Zero剪枝

Version: v1.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# =============================================================================
# §1. SBInjector Dataclass — Per-game structural hint carrier
# =============================================================================

@dataclass
class SBInjector:
    """Per-game structural hint injector for the universal pipeline.

    Each game registers a preamble function that returns an SBInjector
    instance. The pipeline reads these parameters to drive strategy selection.

    Attributes:
        name: Human-readable injector name (e.g., "ls20_keyboard_maze").
        coset_filter: Optional list of coset indices to search.
            None = 搜全330陪集; [0, 3] = 只搜陪集0和3.
        symmetry_hint: Spacetime symmetry prior. None=auto, "D4", "Monster".
        enable_rm: Ramanujan Machine (PSLQ integer relation) 开关.
            True = 调用PSLQ检测; False = 跳过RM阶段.
        time_window: EML temporal edge preservation window.
            1 = 单帧(默认); 2 = 多帧因果边(交互任务).
        eps_factor: GaussEx verification tolerance multiplier.
            1.0 = 默认严格; 1.5/2.0 = 更宽松(交互任务).
        sporadic_pref: Sporadic group DSL routing preference.
            None=auto; "M11"/"M12"/"M24"/"Niemeier"/"Monster".
        physics_primitives: κ-Phase native operators for Dead-Zero pruning.
            [] = no physics prior; ["newton_push"/"mirror_geo"/"dfa"/"poset"].
    """

    name: str = "default"
    coset_filter: Optional[List[int]] = None
    symmetry_hint: Optional[str] = None
    enable_rm: bool = True
    time_window: int = 1
    eps_factor: float = 1.0
    sporadic_pref: Optional[str] = None
    physics_primitives: List[str] = field(default_factory=list)


# =============================================================================
# §2. 25 Preamble Functions — Per-game SBInjector factories
# =============================================================================

def ls20_preamble() -> SBInjector:
    """ls20: 键盘迷宫游戏 — D4对称先验."""
    return SBInjector(
        name="ls20_keyboard_maze",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def vc33_preamble() -> SBInjector:
    """vc33: D4对称先验."""
    return SBInjector(
        name="vc33_default",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def tr87_preamble() -> SBInjector:
    """tr87: 无对称先验，auto检测."""
    return SBInjector(
        name="tr87_default",
        symmetry_hint=None,
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def tu93_preamble() -> SBInjector:
    """tu93: 键盘迷宫 — D4对称先验."""
    return SBInjector(
        name="tu93_keyboard_maze",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def re86_preamble() -> SBInjector:
    """re86: D4对称先验."""
    return SBInjector(
        name="re86_default",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def g50t_preamble() -> SBInjector:
    """g50t: 无对称先验，auto检测."""
    return SBInjector(
        name="g50t_default",
        symmetry_hint=None,
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def wa30_preamble() -> SBInjector:
    """wa30: 更宽松的GaussEx容差(1.5)."""
    return SBInjector(
        name="wa30_default",
        symmetry_hint=None,
        enable_rm=True,
        time_window=1,
        eps_factor=1.5,
    )

def ft09_preamble() -> SBInjector:
    """ft09: D4对称先验."""
    return SBInjector(
        name="ft09_default",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def s5i5_preamble() -> SBInjector:
    """s5i5: D4对称先验(重复点击游戏)."""
    return SBInjector(
        name="s5i5_repeat_click",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def tn36_preamble() -> SBInjector:
    """tn36: 交互点击 — Monster先验 + M11散在群 + 多帧窗口 + dfa原语."""
    return SBInjector(
        name="tn36_click_interactive",
        symmetry_hint="Monster",
        enable_rm=True,
        time_window=2,
        eps_factor=1.5,
        sporadic_pref="M11",
        physics_primitives=['dfa'],
    )

def su15_preamble() -> SBInjector:
    """su15: 无对称先验，auto检测."""
    return SBInjector(
        name="su15_default",
        symmetry_hint=None,
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def r11l_preamble() -> SBInjector:
    """r11l: 无对称先验，auto检测."""
    return SBInjector(
        name="r11l_default",
        symmetry_hint=None,
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def lp85_preamble() -> SBInjector:
    """lp85: 无对称先验，auto检测."""
    return SBInjector(
        name="lp85_default",
        symmetry_hint=None,
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def bp35_preamble() -> SBInjector:
    """bp35: 无对称先验，auto检测."""
    return SBInjector(
        name="bp35_default",
        symmetry_hint=None,
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def dc22_preamble() -> SBInjector:
    """dc22: D4对称先验."""
    return SBInjector(
        name="dc22_default",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def sk48_preamble() -> SBInjector:
    """sk48: D4对称先验."""
    return SBInjector(
        name="sk48_default",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def lf52_preamble() -> SBInjector:
    """lf52: D4对称先验."""
    return SBInjector(
        name="lf52_default",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def sc25_preamble() -> SBInjector:
    """sc25: D4对称先验."""
    return SBInjector(
        name="sc25_default",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def m0r0_preamble() -> SBInjector:
    """m0r0: D4对称先验."""
    return SBInjector(
        name="m0r0_default",
        symmetry_hint="D4",
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def cn04_preamble() -> SBInjector:
    """cn04: 旋转+平移对齐 — D4陪集 + rotation_translation对称 + affine_transform原语."""
    return SBInjector(
        name='cn04',
        coset_filter=[3, 7, 11, 15],
        symmetry_hint='rotation_translation',
        enable_rm=True,
        time_window=2,
        eps_factor=1.5,
        sporadic_pref=False,
        physics_primitives=['affine_transform'],
    )

def cd82_preamble() -> SBInjector:
    """cd82: 无对称先验，auto检测."""
    return SBInjector(
        name="cd82_default",
        symmetry_hint=None,
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def sp80_preamble() -> SBInjector:
    """sp80: 无对称先验，auto检测."""
    return SBInjector(
        name="sp80_default",
        symmetry_hint=None,
        enable_rm=True,
        time_window=1,
        eps_factor=1.0,
    )

def ka59_preamble() -> SBInjector:
    """ka59: 交互推箱子 — Monster先验 + M12散在群 + RM关闭 + newton_push原语."""
    return SBInjector(
        name="ka59_sokoban_interactive",
        symmetry_hint="Monster",
        enable_rm=False,
        time_window=2,
        eps_factor=2.0,
        sporadic_pref="M12",
        physics_primitives=['newton_push'],
    )

def ar25_preamble() -> SBInjector:
    """ar25: 交互镜像 — Monster先验 + M24散在群 + RM关闭 + mirror_geo原语."""
    return SBInjector(
        name="ar25_mirror_interactive",
        symmetry_hint="Monster",
        enable_rm=False,
        time_window=2,
        eps_factor=2.0,
        sporadic_pref="M24",
        physics_primitives=['mirror_geo'],
    )

def sb26_preamble() -> SBInjector:
    """sb26: 交互交换 — D4先验 + RM关闭 + 多帧窗口 + poset原语."""
    return SBInjector(
        name="sb26_swap_interactive",
        symmetry_hint="D4",
        enable_rm=False,
        time_window=2,
        eps_factor=1.5,
        physics_primitives=['poset'],
    )


# =============================================================================
# §3. GAME_INJECTORS Registry — 25 game preamble dispatch
# =============================================================================

GAME_INJECTORS: Dict[str, Callable[[], SBInjector]] = {
    "ls20": ls20_preamble,
    "vc33": vc33_preamble,
    "tr87": tr87_preamble,
    "tu93": tu93_preamble,
    "re86": re86_preamble,
    "g50t": g50t_preamble,
    "wa30": wa30_preamble,
    "ft09": ft09_preamble,
    "s5i5": s5i5_preamble,
    "tn36": tn36_preamble,
    "su15": su15_preamble,
    "r11l": r11l_preamble,
    "lp85": lp85_preamble,
    "bp35": bp35_preamble,
    "dc22": dc22_preamble,
    "sk48": sk48_preamble,
    "lf52": lf52_preamble,
    "sc25": sc25_preamble,
    "m0r0": m0r0_preamble,
    "cn04": cn04_preamble,
    "cd82": cd82_preamble,
    "sp80": sp80_preamble,
    "ka59": ka59_preamble,
    "ar25": ar25_preamble,
    "sb26": sb26_preamble,
}


# =============================================================================
# §4. Default Injector + Lookup Functions
# =============================================================================

def get_default_injector(game_id: str = "") -> SBInjector:
    """Default injector for unknown/unregistered games.

    Returns a vanilla SBInjector with all defaults — no symmetry hint,
    full coset search, RM enabled, single frame, standard tolerance.

    Args:
        game_id: Base game identifier (for naming only).

    Returns:
        SBInjector with default parameters.
    """
    return SBInjector(name=f"default_{game_id}")


def get_injector(game_id: str) -> SBInjector:
    """Get SBInjector for a game. Returns default if no preamble registered.

    Normalizes game_id by stripping version suffix (e.g., "ls20-9607627b" → "ls20"),
    then looks up the preamble function in GAME_INJECTORS. If found, calls it
    to produce a game-specific SBInjector; otherwise returns a default injector.

    Args:
        game_id: Game identifier (may include version suffix).

    Returns:
        SBInjector instance for the specified game.
    """
    base_id: str = game_id.split("-")[0] if game_id else ""
    preamble_fn: Optional[Callable[[], SBInjector]] = GAME_INJECTORS.get(base_id)
    if preamble_fn is not None:
        return preamble_fn()
    return get_default_injector(base_id)
