"""
GameProfiles — Per-game configuration profiles for ARC-AGI-3.

Provides GameProfile dataclass and GameProfileRegistry for managing
per-game configuration. Includes pre-configured profiles for known games
and auto-detection for unknown games.

The registry maps game IDs (e.g., "ls20") to GameProfile instances that
contain action type, step size, color information, mechanism flags, and
baseline step counts. This enables the GridPerception module to use
game-specific knowledge when available.

v3.18.0: Added HybridGameProfile with l1/l2/l3/l4_strategy fields
         for four-layer hybrid search pipeline configuration.

Author: TOMAS Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================================
# GameProfile: Per-game configuration
# ============================================================================

@dataclass
class GameProfile:
    """Per-game configuration profile.

    Stores game-specific parameters learned from the environment or
    pre-configured from known games. Used by GridPerception and
    PlannerAgent to adapt behavior per game.

    Attributes:
        game_id: Game identifier (e.g., "ls20").
        action_type: Interaction type ("keyboard", "click", "keyboard_click").
        step_size: Sprite step size in pixels (typically 5).
        grid_size: Grid dimension (typically 64).
        background_color: Most common color in the grid.
        wall_colors: Set of colors that represent walls.
        player_colors: Set of colors associated with the player.
        goal_colors: Set of colors associated with goals.
        has_switchers: Whether the game has switcher mechanics.
        has_push_blocks: Whether the game has push block mechanics.
        has_moving_switchers: Whether the game has moving switchers.
        has_refills: Whether the game has refill mechanics.
        state_bar_top: Number of rows to skip at top (UI elements).
        state_bar_bottom: Number of rows to skip at bottom (UI elements).
        baselines: Per-level baseline step counts.
    """

    game_id: str = ""
    action_type: str = "unknown"
    step_size: int = 5
    grid_size: int = 64
    # Colors
    background_color: Optional[int] = None
    wall_colors: set[int] = field(default_factory=set)
    player_colors: set[int] = field(default_factory=set)
    goal_colors: set[int] = field(default_factory=set)
    # Mechanism flags
    has_switchers: bool = False
    has_push_blocks: bool = False
    has_moving_switchers: bool = False
    has_refills: bool = False
    # State bar
    state_bar_top: int = 0
    state_bar_bottom: int = 0
    # Baselines
    baselines: list[int] = field(default_factory=list)

    @property
    def is_keyboard(self) -> bool:
        """Whether this game uses keyboard (directional) actions."""
        return "keyboard" in self.action_type

    @property
    def is_click(self) -> bool:
        """Whether this game uses click actions."""
        return "click" in self.action_type

    @property
    def total_baseline(self) -> int:
        """Sum of all per-level baselines."""
        return sum(self.baselines)


# ============================================================================
# HybridGameProfile: Game profile with hybrid search pipeline config
# ============================================================================

@dataclass
class HybridGameProfile(GameProfile):
    """Game profile with four-layer hybrid search pipeline configuration.

    Extends GameProfile with l1/l2/l3/l4 strategy names for the
    HybridSearchPipeline. Each strategy name maps to a concrete
    strategy class in the pipeline's registry.

    Strategy names:
        L1 (Candidate Generation):
            - 'wall_bfs': KA59推箱Wall-BFS (物理原语约束)
            - 'bfs': 标准BFS (结构探索)
            - 'dfs': DFS枚举 (深度优先)
            - 'delta_replay': Δ-State Replay (IDO流贯)
            - 'direct': 直接计算 (简单游戏)

        L2 (Structure Pruning):
            - 'combo_symmetry': 组合对称性剪枝 (KA59)
            - 'prime_signature': 素签名剪枝 (CN04)
            - 'matroid_constraint': 拟阵约束剪枝
            - 'pass_through': 不剪枝(直接传递)

        L3 (Residual Evaluation):
            - 'kappa_snap': κ-Snap投影评估 (Octonion内积)
            - 'dead_zero_fuse': Dead-Zero熔断评估 (卞氏阈值)
            - 'gauss_ex': GaussEx残差评估 (像素匹配)
            - 'asym_index': 不对称指数评估
            - 'pass_through': 不评估(直接传递)

        L4 (Decision Fusion):
            - 'kappa_selector': κ-优选η升序 + Liu-Score双约束
            - 'liu_priority': 纯Liu-Score优先

    Attributes:
        l1_strategy: L1候选生成策略名。
        l2_strategy: L2结构剪枝策略名。
        l3_strategy: L3残差评估策略名。
        l4_strategy: L4决策融合策略名。
    """

    # 四层混合搜索策略配置
    l1_strategy: str = "bfs"
    l2_strategy: str = "pass_through"
    l3_strategy: str = "gauss_ex"
    l4_strategy: str = "kappa_selector"

    # 搜索参数 — 控制 HybridSearchPipeline 的搜索边界
    max_depth: int = 40          # 最大搜索深度（步数）
    max_nodes: int = 300_000     # 最大扩展节点数
    max_time: float = 25.0       # 最大搜索时间（秒）

    def to_pipeline_strategies(self) -> "PipelineStrategies":
        """转换为PipelineStrategies配置。

        Returns:
            PipelineStrategies实例，用于HybridSearchPipeline构建。
        """
        from .hybrid_search_engine import PipelineStrategies
        return PipelineStrategies(
            l1_strategy=self.l1_strategy,
            l2_strategy=self.l2_strategy,
            l3_strategy=self.l3_strategy,
            l4_strategy=self.l4_strategy,
        )

    @classmethod
    def from_game_profile(
        cls,
        profile: GameProfile,
        l1: str = "bfs",
        l2: str = "pass_through",
        l3: str = "gauss_ex",
        l4: str = "kappa_selector",
    ) -> "HybridGameProfile":
        """从GameProfile创建HybridGameProfile。

        Args:
            profile: 基础GameProfile实例。
            l1: L1策略名。
            l2: L2策略名。
            l3: L3策略名。
            l4: L4策略名。

        Returns:
            HybridGameProfile实例。
        """
        return cls(
            game_id=profile.game_id,
            action_type=profile.action_type,
            step_size=profile.step_size,
            grid_size=profile.grid_size,
            background_color=profile.background_color,
            wall_colors=set(profile.wall_colors),
            player_colors=set(profile.player_colors),
            goal_colors=set(profile.goal_colors),
            has_switchers=profile.has_switchers,
            has_push_blocks=profile.has_push_blocks,
            has_moving_switchers=profile.has_moving_switchers,
            has_refills=profile.has_refills,
            state_bar_top=profile.state_bar_top,
            state_bar_bottom=profile.state_bar_bottom,
            baselines=list(profile.baselines),
            l1_strategy=l1,
            l2_strategy=l2,
            l3_strategy=l3,
            l4_strategy=l4,
        )


# ============================================================================
# Pre-configured profiles for known games
# ============================================================================

KNOWN_PROFILES: dict[str, GameProfile] = {
    "ls20": GameProfile(
        game_id="ls20",
        action_type="keyboard",
        step_size=5,
        background_color=4,
        wall_colors={3},
        player_colors={12, 9},
        goal_colors={5},
        has_switchers=True,
        has_push_blocks=True,
        has_moving_switchers=True,
        has_refills=True,
        baselines=[22, 123, 73, 84, 96, 192, 186],
    ),
    "vc33": GameProfile(
        game_id="vc33",
        action_type="click",
        step_size=5,
        baselines=[7, 18, 44, 61, 131, 34, 152],
    ),
    "tr87": GameProfile(
        game_id="tr87",
        action_type="keyboard",
        step_size=5,
        baselines=[54, 58, 40, 45, 71, 146],
    ),
}


# ============================================================================
# All 25 game baselines (from API metadata)
# ============================================================================

ALL_GAME_BASELINES: dict[str, list[int]] = {
    "ls20": [22, 123, 73, 84, 96, 192, 186],
    "vc33": [7, 18, 44, 61, 131, 34, 152],
    "tr87": [54, 58, 40, 45, 71, 146],
    "tu93": [19, 16, 34, 42, 123, 80, 14, 23, 111],
    "bp35": [21, 48, 44, 38, 33, 87, 86, 131, 163],
    "dc22": [59, 102, 67, 98, 324, 578],
    "s5i5": [20, 89, 106, 54, 162, 38, 86, 83],
    "sk48": [61, 177, 101, 103, 230, 181, 125, 92],
    "tn36": [32, 72, 26, 40, 30, 55, 62],
    "ft09": [43, 12, 23, 28, 65, 37],
    "su15": [22, 42, 26, 115, 36, 31, 8, 40, 41],
    "lf52": [32, 81, 60, 71, 205, 148, 244, 109, 164, 225],
    "sc25": [36, 6, 32, 83, 143, 50],
    "m0r0": [30, 111, 203, 26, 500, 237],
    "re86": [26, 42, 86, 108, 189, 139, 424, 241],
    "r11l": [22, 33, 51, 26, 52, 49],
    "cn04": [29, 54, 85, 300, 208, 113],
    "lp85": [17, 38, 31, 16, 41, 60, 26, 159],
    "cd82": [55, 8, 41, 21, 23, 23],
    "g50t": [78, 175, 179, 230, 96, 54, 67],
    "sp80": [39, 58, 25, 148, 96, 152],
    "ka59": [28, 109, 51, 51, 33, 132, 326],
    "ar25": [32, 50, 75, 37, 89, 159, 233, 73],
    "wa30": [71, 119, 183, 98, 368, 68, 79, 442, 415],
    "sb26": [18, 28, 18, 19, 31, 23, 58, 18],
}


# ============================================================================
# 25-game HybridGameProfile configuration table
# ============================================================================

# Five zero-score game configurations (l1=l2=l3=l4='pass_through')
ZERO_SCORE_GAMES: dict[str, HybridGameProfile] = {
    "s5i5": HybridGameProfile(
        game_id="s5i5", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[20, 89, 106, 54, 162, 38, 86, 83],
    ),
    "sk48": HybridGameProfile(
        game_id="sk48", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[61, 177, 101, 103, 230, 181, 125, 92],
    ),
    "lf52": HybridGameProfile(
        game_id="lf52", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[32, 81, 60, 71, 205, 148, 244, 109, 164, 225],
    ),
    "bp35": HybridGameProfile(
        game_id="bp35", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[21, 48, 44, 38, 33, 87, 86, 131, 163],
    ),
    "m0r0": HybridGameProfile(
        game_id="m0r0", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[30, 111, 203, 26, 500, 237],
    ),
}

# Full 25-game HybridGameProfile configuration
HYBRID_GAME_PROFILES: dict[str, HybridGameProfile] = {
    # ── KA59: 推箱 — wall_bfs + combo_symmetry + dead_zero_fuse + kappa_selector ──
    "ka59": HybridGameProfile(
        game_id="ka59", action_type="keyboard",
        background_color=4, wall_colors={0}, player_colors={9}, goal_colors={2},
        has_push_blocks=True,
        l1_strategy="wall_bfs", l2_strategy="combo_symmetry",
        l3_strategy="dead_zero_fuse", l4_strategy="kappa_selector",
        baselines=[28, 109, 51, 51, 33, 132, 326],
    ),
    # ── TN36: DFA点击编程 — direct + pass_through + pass_through + kappa_selector ──
    "tn36": HybridGameProfile(
        game_id="tn36", action_type="click",
        l1_strategy="direct", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="kappa_selector",
        baselines=[32, 72, 26, 40, 30, 55, 62],
    ),
    # ── CN04: 仿射变换 — dfs + prime_signature + kappa_snap + kappa_selector ──
    "cn04": HybridGameProfile(
        game_id="cn04", action_type="keyboard",
        l1_strategy="dfs", l2_strategy="prime_signature",
        l3_strategy="kappa_snap", l4_strategy="kappa_selector",
        baselines=[29, 54, 85, 300, 208, 113],
    ),
    # ── AR25: 镜像覆盖 — bfs + prime_signature + kappa_snap + kappa_selector ──
    "ar25": HybridGameProfile(
        game_id="ar25", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="prime_signature",
        l3_strategy="kappa_snap", l4_strategy="kappa_selector",
        baselines=[32, 50, 75, 37, 89, 159, 233, 73],
    ),
    # ── SB26: 偏序排序 — direct + prime_signature + gauss_ex + kappa_selector ──
    "sb26": HybridGameProfile(
        game_id="sb26", action_type="keyboard",
        l1_strategy="direct", l2_strategy="prime_signature",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[18, 28, 18, 19, 31, 23, 58, 18],
    ),
    # ── LS20: 金色猎人 — bfs + pass_through + gauss_ex + kappa_selector ──
    "ls20": HybridGameProfile(
        game_id="ls20", action_type="keyboard",
        background_color=4, wall_colors={3}, player_colors={12, 9}, goal_colors={5},
        has_switchers=True, has_push_blocks=True,
        has_moving_switchers=True, has_refills=True,
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[22, 123, 73, 84, 96, 192, 186],
    ),
    # ── TR87: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "tr87": HybridGameProfile(
        game_id="tr87", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[54, 58, 40, 45, 71, 146],
    ),
    # ── FT09: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "ft09": HybridGameProfile(
        game_id="ft09", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[43, 12, 23, 28, 65, 37],
    ),
    # ── VC33: 点击游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "vc33": HybridGameProfile(
        game_id="vc33", action_type="click",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[7, 18, 44, 61, 131, 34, 152],
    ),
    # ── TU93: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "tu93": HybridGameProfile(
        game_id="tu93", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[19, 16, 34, 42, 123, 80, 14, 23, 111],
    ),
    # ── WA30: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "wa30": HybridGameProfile(
        game_id="wa30", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[71, 119, 183, 98, 368, 68, 79, 442, 415],
    ),
    # ── DC22: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "dc22": HybridGameProfile(
        game_id="dc22", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[59, 102, 67, 98, 324, 578],
    ),
    # ── RE86: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "re86": HybridGameProfile(
        game_id="re86", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[26, 42, 86, 108, 189, 139, 424, 241],
    ),
    # ── LP85: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "lp85": HybridGameProfile(
        game_id="lp85", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[17, 38, 31, 16, 41, 60, 26, 159],
    ),
    # ── CD82: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "cd82": HybridGameProfile(
        game_id="cd82", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[55, 8, 41, 21, 23, 23],
    ),
    # ── G50T: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "g50t": HybridGameProfile(
        game_id="g50t", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[78, 175, 179, 230, 96, 54, 67],
    ),
    # ── SP80: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "sp80": HybridGameProfile(
        game_id="sp80", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[39, 58, 25, 148, 96, 152],
    ),
    # ── R11L: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "r11l": HybridGameProfile(
        game_id="r11l", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[22, 33, 51, 26, 52, 49],
    ),
    # ── SU15: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "su15": HybridGameProfile(
        game_id="su15", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[22, 42, 26, 115, 36, 31, 8, 40, 41],
    ),
    # ── SC25: 键盘游戏 — bfs + pass_through + gauss_ex + kappa_selector ──
    "sc25": HybridGameProfile(
        game_id="sc25", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="gauss_ex", l4_strategy="kappa_selector",
        baselines=[36, 6, 32, 83, 143, 50],
    ),
    # ── Five zero-score games (pass_through all layers) ──
    "s5i5": HybridGameProfile(
        game_id="s5i5", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[20, 89, 106, 54, 162, 38, 86, 83],
    ),
    "sk48": HybridGameProfile(
        game_id="sk48", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[61, 177, 101, 103, 230, 181, 125, 92],
    ),
    "lf52": HybridGameProfile(
        game_id="lf52", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[32, 81, 60, 71, 205, 148, 244, 109, 164, 225],
    ),
    "bp35": HybridGameProfile(
        game_id="bp35", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[21, 48, 44, 38, 33, 87, 86, 131, 163],
    ),
    "m0r0": HybridGameProfile(
        game_id="m0r0", action_type="keyboard",
        l1_strategy="bfs", l2_strategy="pass_through",
        l3_strategy="pass_through", l4_strategy="pass_through",
        baselines=[30, 111, 203, 26, 500, 237],
    ),
}


# ============================================================================
# GameProfileRegistry: Registry with auto-detection
# ============================================================================

class GameProfileRegistry:
    """Registry for game profiles with auto-detection.

    Maintains a dictionary of known profiles and can auto-detect
    profiles for unknown games based on initial frame analysis.

    Usage:
        registry = GameProfileRegistry()
        profile = registry.get_profile("ls20")
        # Or with auto-detection from grid:
        profile = registry.auto_detect("newgame", grid, [1,2,3,4])
    """

    # Action ID sets for type detection
    KEYBOARD_ACTIONS: set[int] = {1, 2, 3, 4}
    CLICK_ACTIONS: set[int] = {6, 7}

    def __init__(self) -> None:
        """Initialize the registry with known profiles."""
        self.profiles: dict[str, GameProfile] = dict(KNOWN_PROFILES)

    def get_profile(self, game_id: str) -> GameProfile:
        """Get or create a profile for a game.

        Strips version suffixes (e.g., "ls20-9607627b" -> "ls20")
        and returns the known profile if available. Otherwise,
        creates a new profile with auto-detected parameters.

        Args:
            game_id: Game identifier, possibly with version suffix.

        Returns:
            GameProfile for the game.
        """
        # Strip version suffix (e.g., "ls20-9607627b" -> "ls20")
        base_id = game_id.split("-")[0]

        if base_id in self.profiles:
            return self.profiles[base_id]

        # Auto-detect with no grid (will use baselines only)
        profile = self.auto_detect(base_id, None, [])
        self.profiles[base_id] = profile
        return profile

    def auto_detect(
        self,
        game_id: str,
        grid: Optional[np.ndarray],
        available_actions: list[int],
    ) -> GameProfile:
        """Auto-detect game profile from initial frame and actions.

        Infers:
        - Action type from available_actions
        - Background and wall colors from grid color distribution
        - Baseline steps from ALL_GAME_BASELINES

        Args:
            game_id: Base game identifier (without version suffix).
            grid: 64x64 integer array of the initial frame, or None.
            available_actions: List of available action IDs.

        Returns:
            Auto-detected GameProfile.
        """
        baselines = ALL_GAME_BASELINES.get(game_id, [])

        # Determine action type
        has_keyboard = any(
            a in self.KEYBOARD_ACTIONS for a in available_actions
        )
        has_click = any(
            a in self.CLICK_ACTIONS for a in available_actions
        )

        if has_keyboard and has_click:
            action_type = "keyboard_click"
        elif has_keyboard:
            action_type = "keyboard"
        elif has_click:
            action_type = "click"
        else:
            action_type = "unknown"

        profile = GameProfile(
            game_id=game_id,
            action_type=action_type,
            step_size=5,
            grid_size=64,
            baselines=baselines,
        )

        if grid is not None:
            # Infer colors from grid
            colors, counts = np.unique(grid, return_counts=True)
            sorted_by_count = sorted(
                zip(colors, counts), key=lambda x: -x[1]
            )

            if sorted_by_count:
                profile.background_color = int(sorted_by_count[0][0])
            if len(sorted_by_count) > 1:
                profile.wall_colors = {int(sorted_by_count[1][0])}

        return profile

    def update_profile(
        self,
        game_id: str,
        **kwargs,
    ) -> GameProfile:
        """Update an existing profile with new information.

        Args:
            game_id: Game identifier.
            **kwargs: Profile attributes to update.

        Returns:
            Updated GameProfile.
        """
        base_id = game_id.split("-")[0]
        if base_id not in self.profiles:
            self.profiles[base_id] = self.get_profile(base_id)

        profile = self.profiles[base_id]
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        return profile

    def list_known_games(self) -> list[str]:
        """List all known game IDs.

        Returns:
            Sorted list of known game IDs.
        """
        return sorted(self.profiles.keys())

    def has_profile(self, game_id: str) -> bool:
        """Check if a profile exists for a game.

        Args:
            game_id: Game identifier.

        Returns:
            True if a profile exists (known or auto-detected).
        """
        base_id = game_id.split("-")[0]
        return base_id in self.profiles
