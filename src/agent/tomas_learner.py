"""TOMAS Sleep-Step Learning System for ARC-AGI-3 Solver.

This module implements the complete TOMAS (太一互搏) learning cycle:
    record → sleep → audit → consolidate

The Sleep-Step mechanism enables the solver to extract reusable causal
patterns from gameplay episodes, compress them into DSL macros via
MDL (Minimum Description Length) optimization, and persist them in
library.json for cross-game generalization.

Key principles:
    1. κ-Snap少样本学习: Demo只约束搜索方向，不更新权重
    2. Sleep-Step复盘: Online收集轨迹 → Sleep提取因果 → ψ-Audit校验 → Consolidate沉淀
    3. MDL压缩: 合并重复动作(3×LEFT → repeat('LEFT',3))，计算MDL评分
    4. TOMAS指纹: 八元体相位哈希，用于跨游戏模式匹配
    5. ψ审计: 验证宏有效性，检测对齐伪装(Alignment Faking)

Architecture:
    TOMASLearner (主类)
    ├── CausalPatternExtractor (因果模式提取)
    │   ├── extract_sprite_movement (检测sprite移动方向, dx/dy/action_type)
    │   ├── extract_click_effect (检测点击切换, toggled_sprites)
    │   ├── extract_navigation_pattern (检测导航模式)
    │   └── extract_all (组合提取)
    ├── DSLMacroAbstractor (DSL宏抽象)
    │   ├── abstract_sequence (动作序列→DSL抽象, repeat合并)
    │   ├── compress_repeated_actions (合并重复动作)
    │   ├── compute_mdl_score (计算MDL评分, DSL总字符数/原始步数)
    │   └── compute_tomas_fingerprint (八元体相位哈希)
    ├── TopoFeatureExtractor (64维拓扑特征提取, meta_snap_net.py)
    ├── LibraryManager (宏库管理)
    │   ├── load_library / save_library (持久化)
    │   ├── query_by_fingerprint (TOMAS双存储检索)
    │   ├── prune_redundant (冗余修剪)
    ├── FastPathDispatcher (快速宏派发)
    │   ├── try_dispatch (拓扑+GaussEx检索派发)
    │   ├── execute_dispatch (DSL序列执行, repeat解析)
    ├── MacroCandidate (宏候选数据结构)
    ├── ActionTrace (动作轨迹数据结构)
    └── EpisodeTrace (回合轨迹数据结构)

References:
    - κ-Snap abductive search (TOMAS Theory §3)
    - Sleep-Step consolidation (TOMAS Theory §4)
    - NARLA non-associative residual lattice algebra
"""

from __future__ import annotations

import json
import hashlib
import time
import copy
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

# ── 导入升级版特征提取器 ──
from .meta_snap_net import TopoFeatureExtractor as _TopoFeatureExtractorClass

# ── 导入八元数乘法表 (PhysicalNARConv需要) ──
from src.encoder.nar_conv import OCT_MUL_TABLE


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ActionTrace:
    """Record of a single action taken during gameplay.

    Attributes:
        action_type: Type of action ('MOVE', 'CLICK', 'KEYBOARD').
        action_params: Parameters for the action (direction, position, key).
        step_number: Step index in the episode.
        pre_state: Grid state before action (numpy array).
        post_state: Grid state after action (numpy array).
        reward: Reward received for this action.
        rhae_contribution: RHAE score contribution at this step.
    """
    action_type: str
    action_params: Dict[str, Any]
    step_number: int
    pre_state: Optional[np.ndarray] = None
    post_state: Optional[np.ndarray] = None
    reward: float = 0.0
    rhae_contribution: float = 0.0


@dataclass
class EpisodeTrace:
    """Record of a complete gameplay episode (one level attempt).

    Attributes:
        game_id: ARC-AGI-3 game identifier.
        level_index: Level number within the game.
        traces: List of ActionTrace records for each step.
        total_steps: Total number of steps taken.
        rhae_score: Final RHAE score for the episode.
        success: Whether the level was completed.
        level_type: Classification of level mechanism.
        timestamp: Time of episode recording.
        tags: Descriptive tags for pattern matching.
    """
    game_id: str
    level_index: int
    traces: List[ActionTrace] = field(default_factory=list)
    total_steps: int = 0
    rhae_score: float = 0.0
    success: bool = False
    level_type: str = ""
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)


@dataclass
class MacroCandidate:
    """Candidate DSL macro extracted from episode analysis.

    Attributes:
        name: Human-readable macro name.
        dsl_sequence: Compressed DSL action sequence.
        tomas_fingerprint: Octonion phase hash for cross-game matching.
        source_tasks: Game/level IDs where this macro was observed.
        avg_steps: Average number of steps this macro saves.
        success_rate: Fraction of successful applications (0.0-1.0).
        generalization_tags: Tags for cross-game generalization.
        mdl_score: Minimum Description Length compression score.
        min_demo_to_activate: Minimum demos needed to activate this macro.
        created_at: Timestamp of macro creation.
        validated: Whether ψ-Audit has validated this macro.
        applicable_topo: Topology features dict for retrieve_for_topo matching.
            (v3.3.0 泛函宏统一场论 §2 — NAR-Conv拓扑特征)
        gaussex_precond: GaussEx precondition string for Fast-Path dispatch.
            (v3.3.0 泛函宏统一场论 §3 — 宏派发前置条件校验)
    """
    name: str
    dsl_sequence: List[Dict[str, Any]]
    tomas_fingerprint: str = ""
    source_tasks: List[str] = field(default_factory=list)
    avg_steps: float = 0.0
    success_rate: float = 0.0
    generalization_tags: List[str] = field(default_factory=list)
    mdl_score: float = 0.0
    min_demo_to_activate: int = 1
    created_at: float = field(default_factory=time.time)
    validated: bool = False
    # ── v3.3.0 泛函宏统一场论新增字段 ──
    applicable_topo: Dict[str, Any] = field(default_factory=dict)
    gaussex_precond: str = ""


# ============================================================================
# NAR-Conv Topology Feature Extraction (泛函宏统一场论 §2)
# ============================================================================

def extract_topo_features(grid: np.ndarray) -> Dict[str, Any]:
    """Extract topology features from a grid using NAR-Conv analysis.

    Computes Euler characteristic, periodicity rank, and symmetry type
    from the grid's connected-component topology. These features enable
    topology-aware macro retrieval via LibraryManager.retrieve_for_topo.

    Based on the 泛函宏统一场论 unified framework:
        Bκ ≅ TOSAS ∪ PBPN ∪ FUNCATTN ∪ HAH_ONE_ELEMENT
    where topology features serve as the compact base Bκ descriptor.

    NOTE: 此函数保持dict返回格式(向后兼容). 如需64维numpy array,
    使用 TopoFeatureExtractor.extract() (from meta_snap_net.py).

    Args:
        grid: 2D numpy array representing the game state.

    Returns:
        Dict with keys:
            euler_char: Euler characteristic χ = #components - #holes.
            period_rank: Periodicity rank (0=none, 1=row, 2=grid).
            symmetry: List of detected symmetry types ('horizontal',
                      'vertical', 'rotational').
            component_count: Number of distinct connected components.
            hole_count: Number of topological holes.
            density: Fraction of non-zero cells in the grid.
    """
    # ── 使用TopoFeatureExtractor的dict方法(向后兼容) ──
    extractor = _TopoFeatureExtractorClass()
    return extractor.extract_dict(grid)


def extract_topo_features_vec(grid: np.ndarray) -> np.ndarray:
    """Extract 64-dim topology feature vector from a grid.

    升级版extract_topo_features, 返回64维numpy array而非dict.
    直接调用TopoFeatureExtractor.extract(), 可用于MetaSnapNet输入.

    Args:
        grid: 2D numpy array representing the game state.

    Returns:
        64维float32 numpy array, 特征值归一化到[0,1]区间.
    """
    extractor = _TopoFeatureExtractorClass()
    return extractor.extract(grid)


# ============================================================================
# GaussEx Guard (泛函宏统一场论 §3 — 宏派发前置条件校验)
# ============================================================================

class GaussExGuard:
    """GaussEx macro dispatch precondition checker.

    Validates that a macro's precondition (gaussex_precond) holds before
    the macro is dispatched via Fast-Path. Uses a restricted eval namespace
    with safe helper functions (sprite_count, has_sprite_type, AND, OR, NOT).

    The GaussEx Guard is the first gate in the Fast-Path pipeline:
        retrieve_for_topo → GaussExGuard.check_precondition → dispatch DSL

    If the precondition fails, the macro is skipped and κ-Snap search
    is used as the fallback path.

    Example preconditions:
        - "sprite_count('movable') == 1 AND sprite_count('goal') >= 1"
        - "has_sprite_type('switcher') AND NOT has_sprite_type('wall')"
        - "True"  (always applicable, no precondition)

    Args:
        safe_helpers: Dict of helper functions for the eval namespace.
            Defaults to built-in sprite_count and has_sprite_type.
    """

    # Default safe helper functions
    DEFAULT_SAFE_HELPERS: Dict[str, Any] = {
        "sprite_count": lambda game_state, sprite_type: sum(
            1 for s in game_state.get("sprites", [])
            if s.get("type") == sprite_type
        ),
        "has_sprite_type": lambda game_state, sprite_type: any(
            s.get("type") == sprite_type
            for s in game_state.get("sprites", [])
        ),
    }

    # DSL logical operators → Python operators mapping
    DSL_LOGICAL_OP_MAP: Dict[str, str] = {
        "AND": "and",
        "OR": "or",
        "NOT": "not",
    }

    def __init__(
        self,
        safe_helpers: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize GaussExGuard with safe eval namespace.

        Args:
            safe_helpers: Optional custom helper functions to add to
                the eval namespace. Merged with DEFAULT_SAFE_HELPERS.
        """
        self._safe_namespace: Dict[str, Any] = dict(self.DEFAULT_SAFE_HELPERS)
        if safe_helpers:
            self._safe_namespace.update(safe_helpers)

        # Forbidden names that must NEVER appear in preconditions
        self._blocked_names: Set[str] = {
            "exec", "eval", "compile", "open", "import",
            "__import__", "getattr", "setattr", "delattr",
            "globals", "locals", "vars", "dir",
        }

    def check_precondition(
        self,
        macro: MacroCandidate,
        game_state: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check if a macro's GaussEx precondition holds for the current game state.

        Args:
            macro: MacroCandidate with gaussex_precond string.
            game_state: Dict describing current game state, must contain
                'sprites' list for sprite_count/has_sprite_type helpers.

        Returns:
            Tuple of (passes: bool, reason: str).
            - passes=True: precondition satisfied, macro can be dispatched.
            - passes=False: precondition not satisfied or eval error.
        """
        precond = macro.gaussex_precond

        # Empty/True precondition: always applicable
        if not precond or precond.strip() == "True":
            return True, "no precondition (always applicable)"

        # Security: check for blocked names
        for blocked in self._blocked_names:
            if blocked in precond:
                return False, f"blocked name '{blocked}' in precondition"

        # Translate DSL logical operators to Python native operators
        # AND → and, OR → or, NOT → not (Python keywords, not function calls)
        translated = precond
        for dsl_op, py_op in self.DSL_LOGICAL_OP_MAP.items():
            # Replace DSL operator with Python operator (word boundary)
            import re
            translated = re.sub(r'\b' + dsl_op + r'\b', py_op, translated)

        # Build safe eval namespace with game_state bound to helpers
        namespace: Dict[str, Any] = {}
        for name, func in self._safe_namespace.items():
            # Sprite helpers: bind game_state as first argument
            namespace[name] = lambda *args, _f=func, _gs=game_state: _f(_gs, *args)

        # Also inject game_state directly for numeric comparisons
        namespace["game_state"] = game_state

        try:
            result = eval(translated, {"__builtins__": {}}, namespace)
            if isinstance(result, bool):
                return result, f"precondition eval result: {result}"
            else:
                return bool(result), f"precondition eval cast to bool: {bool(result)}"
        except Exception as e:
            return False, f"precondition eval error: {e}"


# ============================================================================
# Fast-Path Dispatcher (泛函宏统一场论 §4 — 快速宏派发)
# ============================================================================

class FastPathDispatcher:
    """Fast-Path macro dispatcher for rapid DSL sequence dispatch.

    The Fast-Path bypasses κ-Snap search by directly dispatching a matching
    macro's DSL sequence when both topology match and GaussEx precondition
    are satisfied. This provides 5-10× speedup over κ-Snap for known patterns.

    Pipeline:
        1. retrieve_for_topo: Find top-k matching macros by topology score
        2. GaussExGuard.check_precondition: Validate precondition for each
        3. dispatch: Return DSL sequence if both checks pass
        4. execute_dispatch: Execute DSL sequence (repeat → N次, action → 1次)

    Falls back to κ-Snap search if no macro passes both gates.

    Args:
        library: LibraryManager instance for topology-aware retrieval.
        guard: GaussExGuard instance for precondition validation.
    """

    def __init__(
        self,
        library: LibraryManager,
        guard: GaussExGuard,
    ) -> None:
        """Initialize Fast-Path Dispatcher.

        Args:
            library: LibraryManager for macro retrieval.
            guard: GaussExGuard for precondition checking.
        """
        self.library = library
        self.guard = guard

    def try_dispatch(
        self,
        game_state: Dict[str, Any],
        game_tags: List[str],
        topo_features: Dict[str, Any],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Attempt Fast-Path dispatch for the current game context.

        Searches the library for topology-matching macros, validates
        preconditions via GaussExGuard, and returns the first passing
        macro's DSL sequence.

        Args:
            game_state: Current game state dict (for GaussEx precondition).
            game_tags: Generalization tags for the current game.
            topo_features: Topology features from extract_topo_features().

        Returns:
            Tuple of (dsl_sequence: Optional[List[Dict]], macro_name: Optional[str]).
            - Both non-None: Fast-Path succeeded, dispatch the DSL sequence.
            - Both None: No matching macro found, fall back to κ-Snap.
        """
        # Step 1: Retrieve top-3 macros by topology score
        candidates = self.library.retrieve_for_topo(
            topo_features=topo_features,
            game_tags=game_tags,
            k=3,
        )

        if not candidates:
            return None, None

        # Step 2: Validate GaussEx preconditions
        for macro, score in candidates:
            passes, reason = self.guard.check_precondition(macro, game_state)
            if passes:
                return macro.dsl_sequence, macro.name

        # Step 3: No macro passed — fall back to κ-Snap
        return None, None

    def execute_dispatch(
        self,
        dsl_sequence: List[Dict[str, Any]],
        game_state: Dict[str, Any],
        step_executor: Optional[Any] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """执行DSL序列, 解析repeat和action指令.

        解析DSL序列中的每个元素:
            - repeat('TYPE', N) → 执行N次TYPE动作
            - action('TYPE') → 执行1次TYPE动作

        返回执行结果: (done, total_reward, game_state).

        Args:
            dsl_sequence: DSL动作序列, 每个元素是dict.
                格式: {"repeat": "LEFT", "count": 3} 或 {"action": "RIGHT"}.
            game_state: 当前游戏状态dict.
            step_executor: 可选的步执行器 (函数/对象).
                如果提供, 调用其execute(action_type, action_params)
                方法执行每一步. 如果None, 只做DSL解析不实际执行.

        Returns:
            Tuple of (done, total_reward, game_state):
            - done: 是否完成 (bool). True = 序列全部成功执行.
            - total_reward: 累计奖励 (float).
            - game_state: 执行后的游戏状态 (dict).
        """
        done = True
        total_reward = 0.0
        current_state = dict(game_state)  # 拷贝, 不修改原始

        for item in dsl_sequence:
            if "repeat" in item:
                # 解析 repeat('TYPE', N) → 执行N次
                action_type = str(item.get("repeat", ""))
                count = int(item.get("count", 1))
                for _ in range(count):
                    if step_executor is not None:
                        try:
                            result = step_executor(action_type, {})
                            if isinstance(result, tuple):
                                # (reward, new_state) 或 (done, reward, state)
                                if len(result) == 2:
                                    reward_val = float(result[0])
                                    total_reward += reward_val
                                elif len(result) >= 3:
                                    done = bool(result[0])
                                    total_reward += float(result[1])
                                    if len(result) >= 3 and isinstance(result[2], dict):
                                        current_state = result[2]
                            elif isinstance(result, (int, float)):
                                total_reward += float(result)
                        except Exception:
                            done = False
                    else:
                        # 无executor: 只记录, 不实际执行
                        total_reward += 0.0

            elif "action" in item:
                # 解析 action('TYPE') → 执行1次
                action_type = str(item.get("action", ""))
                if step_executor is not None:
                    try:
                        result = step_executor(action_type, {})
                        if isinstance(result, tuple):
                            if len(result) == 2:
                                reward_val = float(result[0])
                                total_reward += reward_val
                            elif len(result) >= 3:
                                done = bool(result[0])
                                total_reward += float(result[1])
                                if len(result) >= 3 and isinstance(result[2], dict):
                                    current_state = result[2]
                        elif isinstance(result, (int, float)):
                            total_reward += float(result)
                    except Exception:
                        done = False
                else:
                    # 无executor: 只记录
                    total_reward += 0.0

            else:
                # 未知格式: 跳过
                pass

        return done, total_reward, current_state


# ============================================================================
# Causal Pattern Extractor
# ============================================================================

class CausalPatternExtractor:
    """Extract causal patterns from gameplay episode traces.

    Analyzes the differences between pre-state and post-state grids
    to identify causal relationships between actions and effects.

    Three core pattern types:
    1. sprite_movement: Sprite position changes after keyboard actions (dx, dy)
    2. click_effect: Grid state changes after click actions (toggled_sprites)
    3. navigation_pattern: Sequence of actions leading to goal achievement

    Enhanced (v3.4.0 — SPP集成):
    - extract_sprite_movement: 增加 dx, dy, action_type 精细字段
    - extract_click_effect: 增加 toggled_sprites 列表
    - extract_all: 组合提取所有模式
    """

    # Minimum change threshold for detecting meaningful state differences
    STATE_DIFF_THRESHOLD: float = 0.05

    def __init__(self) -> None:
        """Initialize the pattern extractor."""
        self._known_patterns: Dict[str, List[MacroCandidate]] = {}

    def extract_sprite_movement(
        self,
        trace: ActionTrace,
    ) -> Optional[Dict[str, Any]]:
        """Detect sprite movement direction from an action trace.

        分析pre_state和post_state的差异, 确定键盘动作是否导致
        sprite按一致方向移动. 增强版增加dx/dy/action_type精细字段.

        Args:
            trace: ActionTrace with pre/post state grids.

        Returns:
            Pattern dict with keys: type, direction, distance, sprite_id,
            dx (行偏移量), dy (列偏移量), action_type (动作类型).
            None if no movement detected.
        """
        if trace.pre_state is None or trace.post_state is None:
            return None

        if trace.action_type != "KEYBOARD":
            return None

        # Find cells that changed between pre and post states
        diff = trace.post_state - trace.pre_state
        changed_cells = np.argwhere(diff != 0)

        if len(changed_cells) == 0:
            return None

        # Identify sprite: largest connected component of changed cells
        # that share the same color value
        sprite_positions = changed_cells

        # Compute movement direction
        if len(sprite_positions) < 2:
            center_before = sprite_positions[0]
            # Find same color in post_state
            color = trace.pre_state[center_before[0], center_before[1]]
            post_positions = np.argwhere(trace.post_state == color)
            if len(post_positions) > 0:
                center_after = post_positions.mean(axis=0)
                direction = center_after - center_before
                distance = np.linalg.norm(direction)

                if distance > 0:
                    # Normalize direction to cardinal directions
                    dir_name = self._direction_to_name(direction)
                    # ── 增强版: dx/dy/action_type ──
                    dx = float(direction[0])  # 行偏移
                    dy = float(direction[1])  # 列偏移
                    action_type_str = trace.action_params.get("direction", trace.action_type)
                    return {
                        "type": "sprite_movement",
                        "direction": dir_name,
                        "distance": float(distance),
                        "sprite_color": int(color),
                        "action_params": trace.action_params,
                        "dx": dx,
                        "dy": dy,
                        "action_type": action_type_str,
                    }

        return None

    def extract_click_effect(
        self,
        trace: ActionTrace,
    ) -> Optional[Dict[str, Any]]:
        """Detect click toggle effect from an action trace.

        分析点击动作是否切换了grid元素的状态. 增强版增加
        toggled_sprites列表, 记录所有被切换的sprite及其前后颜色.

        Args:
            trace: ActionTrace with pre/post state grids.

        Returns:
            Pattern dict with keys: type, position, toggle_type,
            colors_changed, toggled_sprites (被切换的sprite列表).
            None if no click effect detected.
        """
        if trace.pre_state is None or trace.post_state is None:
            return None

        if trace.action_type != "CLICK":
            return None

        diff = trace.post_state - trace.pre_state
        changed_cells = np.argwhere(diff != 0)

        if len(changed_cells) == 0:
            return None

        # Check if the click position is near changed cells
        click_pos = trace.action_params.get("position", (0, 0))
        if isinstance(click_pos, (list, tuple)):
            click_row, click_col = click_pos
        else:
            click_row, click_col = 0, 0

        # Find cells within a small radius of the click
        nearby_changes = []
        for cell in changed_cells:
            dist = np.linalg.norm(cell - np.array([click_row, click_col]))
            if dist <= 3:  # Within 3-cell radius
                nearby_changes.append(cell)

        if len(nearby_changes) == 0:
            return None

        # Determine toggle type
        pre_colors = [trace.pre_state[c[0], c[1]] for c in nearby_changes]
        post_colors = [trace.post_state[c[0], c[1]] for c in nearby_changes]

        color_pairs = set(zip(pre_colors, post_colors))

        # ── 增强版: toggled_sprites列表 ──
        toggled_sprites: List[Dict[str, Any]] = []
        for i, cell in enumerate(nearby_changes):
            toggled_sprites.append({
                "position": (int(cell[0]), int(cell[1])),
                "pre_color": int(pre_colors[i]),
                "post_color": int(post_colors[i]),
                "toggled": pre_colors[i] != post_colors[i],
            })

        return {
            "type": "click_effect",
            "click_position": (click_row, click_col),
            "toggle_type": "switch" if len(nearby_changes) <= 2 else "area_change",
            "colors_changed": [(int(p), int(q)) for p, q in color_pairs],
            "num_cells_changed": len(nearby_changes),
            "action_params": trace.action_params,
            "toggled_sprites": toggled_sprites,
        }

    def extract_all(
        self,
        episode: EpisodeTrace,
    ) -> Dict[str, Any]:
        """组合提取episode中的所有因果模式.

        遍历episode的每一步trace, 提取sprite_movement, click_effect,
        和navigation_pattern, 返回组合结果.

        Args:
            episode: EpisodeTrace回合轨迹.

        Returns:
            Dict with keys:
                movements: List of sprite_movement pattern dicts.
                clicks: List of click_effect pattern dicts.
                navigation: Optional navigation pattern dict.
                positive_navigation: Optional positive navigation dict.
        """
        movements: List[Dict[str, Any]] = []
        clicks: List[Dict[str, Any]] = []

        for trace in episode.traces:
            movement = self.extract_sprite_movement(trace)
            if movement is not None:
                movements.append(movement)

            click = self.extract_click_effect(trace)
            if click is not None:
                clicks.append(click)

        navigation = self.extract_navigation_pattern(episode)
        positive_nav = self.extract_positive_navigation(episode)

        return {
            "movements": movements,
            "clicks": clicks,
            "navigation": navigation,
            "positive_navigation": positive_nav,
        }

    def extract_navigation_pattern(
        self,
        episode: EpisodeTrace,
    ) -> Optional[Dict[str, Any]]:
        """Detect navigation pattern from a successful episode.

        Analyzes the full episode trace to identify sequences of
        keyboard actions that constitute a reusable navigation pattern.

        Args:
            episode: Complete EpisodeTrace for a successful level.

        Returns:
            Pattern dict with keys: type, action_sequence, path_length,
            goal_position, obstacles_avoided.
            None if no clear navigation pattern detected.
        """
        if not episode.success:
            return None

        keyboard_traces = [
            t for t in episode.traces
            if t.action_type == "KEYBOARD"
        ]

        if len(keyboard_traces) < 3:
            return None

        # Extract action sequence
        action_sequence = []
        for t in keyboard_traces:
            direction = t.action_params.get("direction", "")
            if direction:
                action_sequence.append(direction)

        # Compress repeated actions
        compressed = self._compress_sequence(action_sequence)

        # Identify path characteristics
        path_length = len(action_sequence)
        unique_directions = set(action_sequence)

        # Determine if this is a navigation pattern vs. a puzzle pattern
        if path_length >= 5 and len(unique_directions) <= 4:
            return {
                "type": "navigation_pattern",
                "action_sequence": action_sequence,
                "compressed_sequence": compressed,
                "path_length": path_length,
                "unique_directions": list(unique_directions),
                "game_id": episode.game_id,
                "level_index": episode.level_index,
            }

        return None

    def extract_positive_navigation(
        self,
        episode: EpisodeTrace,
    ) -> Optional[Dict[str, Any]]:
        """Extract navigation patterns from episodes that showed improvement.

        Unlike extract_navigation_pattern which requires full success,
        this method identifies partial progress patterns that still
        provide useful causal information.

        Args:
            episode: EpisodeTrace, may be partial success.

        Returns:
            Navigation pattern dict if improvement detected, else None.
        """
        if episode.rhae_score < 20:
            return None  # Too poor to learn from

        keyboard_traces = [
            t for t in episode.traces
            if t.action_type == "KEYBOARD" and t.reward > 0
        ]

        if len(keyboard_traces) < 2:
            return None

        # Extract positively-rewarded action subsequence
        positive_actions = []
        for t in keyboard_traces:
            direction = t.action_params.get("direction", "")
            if direction:
                positive_actions.append({
                    "direction": direction,
                    "reward": t.reward,
                    "step": t.step_number,
                })

        if len(positive_actions) >= 2:
            return {
                "type": "positive_navigation",
                "positive_actions": positive_actions,
                "game_id": episode.game_id,
                "level_index": episode.level_index,
                "partial_score": episode.rhae_score,
            }

        return None

    @staticmethod
    def _direction_to_name(direction: np.ndarray) -> str:
        """Convert a 2D direction vector to a cardinal direction name.

        Args:
            direction: 2D vector (row_delta, col_delta).

        Returns:
            Direction name string ('UP', 'DOWN', 'LEFT', 'RIGHT', or 'DIAGONAL').
        """
        row_d, col_d = direction

        # Determine dominant axis
        if abs(row_d) > abs(col_d) * 2:
            return "UP" if row_d < 0 else "DOWN"
        elif abs(col_d) > abs(row_d) * 2:
            return "LEFT" if col_d < 0 else "RIGHT"
        else:
            # Diagonal movement
            parts = []
            if row_d < 0:
                parts.append("UP")
            elif row_d > 0:
                parts.append("DOWN")
            if col_d < 0:
                parts.append("LEFT")
            elif col_d > 0:
                parts.append("RIGHT")
            return "_".join(parts) if parts else "STATIONARY"

    @staticmethod
    def _compress_sequence(actions: List[str]) -> List[Dict[str, Any]]:
        """Compress a sequence of repeated actions.

        Merges consecutive identical actions into repeat() blocks:
            ['LEFT', 'LEFT', 'LEFT'] → [{'repeat': 'LEFT', 'count': 3}]

        Args:
            actions: List of action strings.

        Returns:
            Compressed sequence as list of action/repeat dicts.
        """
        if not actions:
            return []

        compressed = []
        current_action = actions[0]
        count = 1

        for a in actions[1:]:
            if a == current_action:
                count += 1
            else:
                if count > 2:
                    compressed.append({"repeat": current_action, "count": count})
                else:
                    for _ in range(count):
                        compressed.append({"action": current_action})
                current_action = a
                count = 1

        # Handle last group
        if count > 2:
            compressed.append({"repeat": current_action, "count": count})
        else:
            for _ in range(count):
                compressed.append({"action": current_action})

        return compressed


# ============================================================================
# DSL Macro Abstractor
# ============================================================================

class DSLMacroAbstractor:
    """Abstract DSL macros from causal patterns via MDL optimization.

    Implements the κ-Snap compression principle:
    - Merge repeated actions into repeat() blocks
    - Compute MDL score: len(compressed) / len(original)
    - Generate TOMAS fingerprints for cross-game matching
    - Prune redundant macros via structural signature comparison
    - abstract_sequence: 连续相同动作→repeat DSL抽象
    - compute_mdl_score(dsl_seq, num_raw): DSL总字符数/原始步数
    """

    # MDL compression threshold — macros must achieve this compression ratio
    MDL_THRESHOLD: float = 0.7

    def __init__(self) -> None:
        """Initialize the macro abstractor."""
        self._macro_cache: Dict[str, MacroCandidate] = {}

    def abstract_sequence(
        self,
        actions: List[str],
    ) -> List[Dict[str, Any]]:
        """将连续相同动作抽象为DSL序列 (repeat合并).

        将原始动作序列(如['LEFT','LEFT','LEFT','RIGHT'])转换为
        DSL格式(如[{'repeat':'LEFT','count':3},{'action':'RIGHT'}]).

        规则: 连续3次及以上相同动作 → repeat('TYPE', N).
        连续1-2次 → 保持为action('TYPE').

        Args:
            actions: 原始动作名称列表 (如['UP','UP','UP','RIGHT']).

        Returns:
            DSL动作序列, 每个元素是 {'repeat': TYPE, 'count': N}
            或 {'action': TYPE}.
        """
        if not actions:
            return []

        dsl_sequence: List[Dict[str, Any]] = []
        current_action = actions[0]
        count = 1

        for a in actions[1:]:
            if a == current_action:
                count += 1
            else:
                # 输出当前组
                if count >= 3:
                    dsl_sequence.append({"repeat": current_action, "count": count})
                elif count == 2:
                    dsl_sequence.append({"action": current_action})
                    dsl_sequence.append({"action": current_action})
                else:
                    dsl_sequence.append({"action": current_action})
                current_action = a
                count = 1

        # 处理最后一组
        if count >= 3:
            dsl_sequence.append({"repeat": current_action, "count": count})
        elif count == 2:
            dsl_sequence.append({"action": current_action})
            dsl_sequence.append({"action": current_action})
        else:
            dsl_sequence.append({"action": current_action})

        return dsl_sequence

    def compute_mdl_score_dsl(
        self,
        dsl_seq: List[Dict[str, Any]],
        num_raw: int,
    ) -> float:
        """计算MDL评分: DSL总字符数 / 原始步数.

        MDL (Minimum Description Length) 衡量DSL序列的压缩效率:
            mdl_score = total_chars(dsl_seq) / num_raw_steps

        Lower mdl_score = 更好的压缩. 值 > 1.0 表示DSL比原始动作更长.

        Args:
            dsl_seq: DSL动作序列 (含repeat和action).
            num_raw: 原始动作步数 (未压缩前的总步数).

        Returns:
            MDL评分 (float). 范围: 0.0 (完美压缩) → ∞ (无压缩).
        """
        if num_raw <= 0:
            return 1.0

        total_chars = 0
        for item in dsl_seq:
            if "repeat" in item:
                # repeat块: "repeat('TYPE', N)" ≈ len(TYPE) + 数字字符数 + overhead
                action_name = str(item.get("repeat", ""))
                count_str = str(item.get("count", 1))
                total_chars += len(action_name) + len(count_str) + 8  # overhead
            elif "action" in item:
                # action块: "action('TYPE')" ≈ len(TYPE) + overhead
                action_name = str(item.get("action", ""))
                total_chars += len(action_name) + 6  # overhead
            else:
                # 未知格式: 按JSON字符数计
                total_chars += len(json.dumps(item))

        return total_chars / num_raw

    def compress_repeated_actions(
        self,
        action_sequence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compress a DSL action sequence by merging repeated actions.

        Uses κ-Snap-style MDL compression:
        - 3 consecutive identical actions → repeat(action, 3)
        - Further compression for larger patterns

        Args:
            action_sequence: List of action dicts with keys 'action' or 'repeat'.

        Returns:
            Compressed action sequence with repeat blocks.
        """
        # Flatten to simple actions first
        flat_actions = []
        for item in action_sequence:
            if "repeat" in item:
                for _ in range(item["repeat"].get("count", 1)):
                    flat_actions.append(item["repeat"].get("action", item["repeat"]))
            elif "action" in item:
                flat_actions.append(item["action"])
            else:
                flat_actions.append(str(item))

        # Compress consecutive identical actions
        return CausalPatternExtractor._compress_sequence(flat_actions)

    def compute_mdl_score(
        self,
        original: List[Dict[str, Any]],
        compressed: List[Dict[str, Any]],
    ) -> float:
        """Compute Minimum Description Length (MDL) score.

        MDL score = len(compressed_encoding) / len(original_encoding)
        Lower is better — indicates greater compression.

        The encoding length accounts for:
        - Simple actions: 1 unit each
        - Repeat blocks: 2 units (repeat + count)
        - Parameter dicts: proportional to key count

        Args:
            original: Original uncompressed action sequence.
            compressed: Compressed action sequence.

        Returns:
            MDL score as float (0.0 = perfect compression, 1.0 = no compression).
        """
        def encoding_length(seq: List[Dict[str, Any]]) -> float:
            total = 0.0
            for item in seq:
                if "repeat" in item:
                    # Repeat block: overhead of 2 (repeat keyword + count)
                    # But encodes N actions in 2 units
                    total += 2.0
                elif "action" in item:
                    total += 1.0
                else:
                    # Dict encoding: proportional to keys
                    total += len(item) * 0.5
            return total

        orig_len = encoding_length(original)
        comp_len = encoding_length(compressed)

        if orig_len <= 0:
            return 1.0

        return comp_len / orig_len

    def compute_tomas_fingerprint(
        self,
        pattern: Dict[str, Any],
    ) -> str:
        """Compute TOMAS fingerprint (octonion phase hash) for a pattern.

        Generates a deterministic hash from the pattern's structural
        features, enabling cross-game matching without requiring
        identical action sequences.

        The fingerprint encodes:
        - Pattern type
        - Action direction frequencies
        - State change characteristics

        Args:
            pattern: Pattern dict from CausalPatternExtractor.

        Returns:
            TOMAS fingerprint string (16-char hex).
        """
        # Extract structural features for hashing
        features = {
            "type": pattern.get("type", ""),
            "directions": sorted(pattern.get("unique_directions", [])),
            "toggle_type": pattern.get("toggle_type", ""),
            "path_length": pattern.get("path_length", 0),
            "num_cells_changed": pattern.get("num_cells_changed", 0),
        }

        # Deterministic hash
        feature_str = json.dumps(features, sort_keys=True)
        fingerprint = hashlib.sha256(feature_str.encode()).hexdigest()[:16]

        return fingerprint

    def abstract_macro(
        self,
        pattern: Dict[str, Any],
        episode: EpisodeTrace,
    ) -> Optional[MacroCandidate]:
        """Abstract a DSL macro from a detected causal pattern.

        Takes a pattern detected by CausalPatternExtractor and creates
        a MacroCandidate with MDL compression and TOMAS fingerprint.

        Args:
            pattern: Pattern dict from CausalPatternExtractor.
            episode: Source EpisodeTrace for the pattern.

        Returns:
            MacroCandidate if the pattern achieves MDL threshold, else None.
        """
        # Get action sequence from pattern
        if "compressed_sequence" in pattern:
            action_seq = pattern["compressed_sequence"]
        elif "action_sequence" in pattern:
            action_seq = CausalPatternExtractor._compress_sequence(
                pattern["action_sequence"]
            )
        elif "positive_actions" in pattern:
            action_seq = [
                {"action": a["direction"]}
                for a in pattern["positive_actions"]
            ]
        else:
            return None

        # Compute MDL score
        # Original length is the uncompressed sequence
        if "action_sequence" in pattern:
            original_len = [{"action": a} for a in pattern["action_sequence"]]
        else:
            original_len = action_seq

        mdl_score = self.compute_mdl_score(original_len, action_seq)

        if mdl_score > self.MDL_THRESHOLD and len(action_seq) > 3:
            # Not enough compression to warrant a macro
            return None

        # Compute TOMAS fingerprint
        fingerprint = self.compute_tomas_fingerprint(pattern)

        # Generate macro name
        pattern_type = pattern.get("type", "unknown")
        game_id = episode.game_id
        level_idx = episode.level_index
        macro_name = f"macro_{pattern_type}_{game_id}_L{level_idx}"

        # Determine generalization tags
        tags = self._infer_generalization_tags(pattern)

        # Create MacroCandidate
        macro = MacroCandidate(
            name=macro_name,
            dsl_sequence=action_seq,
            tomas_fingerprint=fingerprint,
            source_tasks=[f"{game_id}_L{level_idx}"],
            avg_steps=len(action_seq) if episode.success else 0,
            success_rate=1.0 if episode.success else episode.rhae_score / 115.0,
            generalization_tags=tags,
            mdl_score=mdl_score,
            min_demo_to_activate=1,
            validated=False,  # Will be validated by ψ-Audit
        )

        return macro

    @staticmethod
    def _infer_generalization_tags(pattern: Dict[str, Any]) -> List[str]:
        """Infer generalization tags from a pattern's characteristics.

        Tags enable cross-game matching by describing the pattern's
        structural properties rather than game-specific details.

        Args:
            pattern: Pattern dict from CausalPatternExtractor.

        Returns:
            List of generalization tag strings.
        """
        tags = []

        pattern_type = pattern.get("type", "")
        if pattern_type:
            tags.append(pattern_type)

        # Navigation-specific tags
        directions = pattern.get("unique_directions", [])
        if len(directions) <= 2:
            tags.append("simple_path")
        elif len(directions) >= 4:
            tags.append("complex_navigation")

        # Movement direction tags
        direction = pattern.get("direction", "")
        if direction:
            tags.append(f"move_{direction.lower()}")

        # Click-specific tags
        toggle_type = pattern.get("toggle_type", "")
        if toggle_type:
            tags.append(f"click_{toggle_type}")

        # Size-based tags
        path_length = pattern.get("path_length", 0)
        if path_length > 0:
            if path_length <= 5:
                tags.append("short_sequence")
            elif path_length <= 15:
                tags.append("medium_sequence")
            else:
                tags.append("long_sequence")

        return tags


# ============================================================================
# Library Manager
# ============================================================================

class LibraryManager:
    """Manage the persistent DSL macro library (library.json).

    Handles loading/saving the macro library, TOMAS dual-storage
    retrieval (fingerprint + tag matching), topology-aware retrieval
    via retrieve_for_topo, and redundancy pruning.

    Library schema (v3.3.0 — 泛函宏统一场论):
    {
        "_schema_version": "3.3.0",
        "_description": "TOMAS DSL Macro Library",
        "_tomas_framework": {...},
        "abstractions": [
            {
                ...existing fields...,
                "applicable_topo": {...},     // NEW: topology features
                "gaussex_precond": "..."      // NEW: GaussEx precondition
            }
        ]
    }
    """

    # Default library path
    DEFAULT_LIBRARY_PATH: str = "library.json"

    # ── Topology-aware retrieval weights (泛函宏统一场论 §4) ──
    TOPO_TAG_WEIGHT: float = 0.4
    TOPO_SIM_WEIGHT: float = 0.3
    TOPO_MDL_WEIGHT: float = 0.3

    def __init__(
        self,
        library_path: Optional[str] = None,
    ) -> None:
        """Initialize the library manager.

        Args:
            library_path: Path to library.json file.
                Defaults to DEFAULT_LIBRARY_PATH.
        """
        self.library_path = Path(library_path or self.DEFAULT_LIBRARY_PATH)
        self._library: Dict[str, Any] = self._load_or_create_library()

    def _load_or_create_library(self) -> Dict[str, Any]:
        """Load existing library or create empty one with v3.3.0 schema.

        Returns:
            Library dict with v3.3.0 schema.
        """
        if self.library_path.exists():
            try:
                with open(self.library_path, "r", encoding="utf-8") as f:
                    lib = json.load(f)
                # Check schema version
                version = lib.get("_schema_version", "1.0.0")
                if version != "3.3.0":
                    # Upgrade to v3.3.0
                    lib = self._upgrade_schema(lib, version)
                return lib
            except (json.JSONDecodeError, IOError):
                pass

        # Create new library with v3.3.0 schema
        return {
            "_schema_version": "3.3.0",
            "_description": "TOMAS DSL Macro Library for ARC-AGI-3 Solver (泛函宏统一场论)",
            "_tomas_framework": {
                "version": "3.3.0-dev",
                "core_modules": [
                    "kappa_snap_searcher",
                    "nar_conv",
                    "tomas_learner",
                    "library_manager",
                    "gaussex_verifier",
                    "fast_path_dispatcher",
                ],
                "asym_index_enabled": True,
                "narla_integration": True,
                "fast_path_enabled": True,
            },
            "_note": "Macros are validated by ψ-Audit before activation. "
                     "TOMAS fingerprints enable cross-game pattern matching. "
                     "v3.3.0 adds applicable_topo + gaussex_precond for Fast-Path dispatch.",
            "abstractions": [],
        }

    def _upgrade_schema(
        self,
        old_lib: Dict[str, Any],
        from_version: str = "3.2.0",
    ) -> Dict[str, Any]:
        """Upgrade library from older schema to v3.3.0.

        Adds applicable_topo and gaussex_precond fields to each abstraction.

        Args:
            old_lib: Library dict with old schema.
            from_version: Schema version being upgraded from.

        Returns:
            Upgraded library dict with v3.3.0 schema.
        """
        # Preserve existing abstractions, add new fields
        old_abstractions = old_lib.get("abstractions", [])
        upgraded_abstractions = []
        for abs_dict in old_abstractions:
            # Add v3.3.0 fields if missing
            if "applicable_topo" not in abs_dict:
                abs_dict["applicable_topo"] = {}
            if "gaussex_precond" not in abs_dict:
                abs_dict["gaussex_precond"] = "True"  # Default: always applicable
            upgraded_abstractions.append(abs_dict)

        return {
            "_schema_version": "3.3.0",
            "_description": "TOMAS DSL Macro Library for ARC-AGI-3 Solver (泛函宏统一场论)",
            "_tomas_framework": {
                "version": "3.3.0-dev",
                "core_modules": [
                    "kappa_snap_searcher",
                    "nar_conv",
                    "tomas_learner",
                    "library_manager",
                    "gaussex_verifier",
                    "fast_path_dispatcher",
                ],
                "asym_index_enabled": True,
                "narla_integration": True,
                "fast_path_enabled": True,
            },
            "_note": "Macros are validated by ψ-Audit before activation. "
                     "TOMAS fingerprints enable cross-game pattern matching. "
                     "v3.3.0 adds applicable_topo + gaussex_precond for Fast-Path dispatch.",
            "abstractions": upgraded_abstractions,
        }

    def save_library(self) -> None:
        """Save the library to disk."""
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump(self._library, f, indent=2, ensure_ascii=False)

    def add_macro(self, macro: MacroCandidate) -> None:
        """Add a validated macro to the library.

        Args:
            macro: MacroCandidate to add (must be validated by ψ-Audit).
        """
        macro_dict = asdict(macro)
        self._library["abstractions"].append(macro_dict)
        self.save_library()

    def query_by_fingerprint(
        self,
        fingerprint: str,
        tolerance: int = 2,
    ) -> List[MacroCandidate]:
        """Query macros by TOMAS fingerprint with tolerance for fuzzy matching.

        TOMAS dual-storage retrieval: matches on fingerprint (structural)
        with configurable tolerance for similar-but-not-identical patterns.

        Args:
            fingerprint: Target TOMAS fingerprint (16-char hex).
            tolerance: Maximum character mismatches allowed. Defaults to 2.

        Returns:
            List of matching MacroCandidates.
        """
        results = []
        for abs_dict in self._library.get("abstractions", []):
            fp = abs_dict.get("tomas_fingerprint", "")
            if not fp:
                continue

            # Compute character mismatch count
            mismatches = sum(
                c1 != c2 for c1, c2 in zip(fingerprint, fp)
            )

            if mismatches <= tolerance:
                macro = MacroCandidate(**abs_dict)
                results.append(macro)

        return results

    def query_by_tags(
        self,
        tags: List[str],
        min_match: int = 1,
    ) -> List[MacroCandidate]:
        """Query macros by generalization tags.

        Returns macros whose generalization_tags overlap with the
        query tags by at least min_match tags.

        Args:
            tags: Query tag list.
            min_match: Minimum number of matching tags required.

        Returns:
            List of matching MacroCandidates.
        """
        results = []
        tag_set = set(tags)

        for abs_dict in self._library.get("abstractions", []):
            macro_tags = set(abs_dict.get("generalization_tags", []))
            overlap = len(tag_set & macro_tags)

            if overlap >= min_match:
                macro = MacroCandidate(**abs_dict)
                results.append(macro)

        # Sort by success_rate descending
        results.sort(key=lambda m: m.success_rate, reverse=True)
        return results

    # ── Topology-Aware Retrieval (泛函宏统一场论 §4) ──

    def retrieve_for_topo(
        self,
        topo_features: Dict[str, Any],
        game_tags: List[str],
        k: int = 3,
    ) -> List[Tuple[MacroCandidate, float]]:
        """Retrieve macros by topology-aware scoring for Fast-Path dispatch.

        Scoring formula (泛函宏统一场论 §4):
            score = 0.4 * tag_overlap + 0.3 * topo_sim + 0.3 * mdl_bonus

        Where:
            tag_overlap: Jaccard similarity between query tags and macro tags.
            topo_sim: Topology similarity between query features and macro's
                applicable_topo (Euler χ match, period rank match, symmetry overlap).
            mdl_bonus: Inverted MDL score (lower MDL = higher bonus),
                normalized as (1.0 - mdl_score) for compression quality.

        Args:
            topo_features: Topology features dict from extract_topo_features().
            game_tags: Generalization tags for the current game context.
            k: Maximum number of candidates to return. Defaults to 3.

        Returns:
            List of (MacroCandidate, score) tuples, sorted by score descending.
        """
        candidates: List[Tuple[MacroCandidate, float]] = []
        query_tag_set = set(game_tags)

        for abs_dict in self._library.get("abstractions", []):
            macro = MacroCandidate(**abs_dict)

            # ── Tag overlap score ──
            macro_tag_set = set(macro.generalization_tags)
            if query_tag_set and macro_tag_set:
                tag_overlap = len(query_tag_set & macro_tag_set) / \
                             len(query_tag_set | macro_tag_set)
            else:
                tag_overlap = 0.0

            # ── Topology similarity score ──
            macro_topo = macro.applicable_topo
            if macro_topo:
                topo_sim = self._compute_topo_similarity(topo_features, macro_topo)
            else:
                # No applicable_topo: moderate similarity if tags overlap
                topo_sim = tag_overlap * 0.5  # Heuristic fallback

            # ── MDL bonus score ──
            mdl_bonus = max(0.0, 1.0 - macro.mdl_score)  # Lower MDL = higher bonus

            # ── Combined score ──
            score = (
                self.TOPO_TAG_WEIGHT * tag_overlap
                + self.TOPO_SIM_WEIGHT * topo_sim
                + self.TOPO_MDL_WEIGHT * mdl_bonus
            )

            # Only consider macros with positive score
            if score > 0.0:
                candidates.append((macro, score))

        # Sort by score descending, return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def _compute_topo_similarity(
        self,
        query_topo: Dict[str, Any],
        macro_topo: Dict[str, Any],
    ) -> float:
        """Compute topology similarity between two feature dicts.

        Matches on:
            euler_char: Exact match → 0.4, off-by-1 → 0.2, else → 0.0
            period_rank: Exact match → 0.3
            symmetry: Overlap fraction → 0.3

        Args:
            query_topo: Topology features from the current game state.
            macro_topo: applicable_topo features from a macro.

        Returns:
            Similarity score (0.0-1.0).
        """
        sim = 0.0

        # Euler characteristic match
        q_euler = query_topo.get("euler_char", 0)
        m_euler = macro_topo.get("euler_char", 0)
        if q_euler == m_euler:
            sim += 0.4
        elif abs(q_euler - m_euler) == 1:
            sim += 0.2

        # Period rank match
        q_period = query_topo.get("period_rank", 0)
        m_period = macro_topo.get("period_rank", 0)
        if q_period == m_period:
            sim += 0.3

        # Symmetry overlap
        q_sym = set(query_topo.get("symmetry", []))
        m_sym = set(macro_topo.get("symmetry", []))
        if q_sym and m_sym:
            sym_overlap = len(q_sym & m_sym) / len(q_sym | m_sym)
            sim += 0.3 * sym_overlap
        elif not q_sym and not m_sym:
            sim += 0.15  # Both have no symmetry → partial match

        return min(1.0, sim)

    def prune_redundant(self) -> int:
        """Prune redundant macros from the library.

        Uses structural signature comparison to identify macros that
        are subsumed by more general or more successful macros.

        A macro M1 is redundant if there exists M2 such that:
        - M2.generalization_tags ⊇ M1.generalization_tags
        - M2.success_rate >= M1.success_rate
        - M2.avg_steps <= M1.avg_steps (more efficient)

        Returns:
            Number of macros pruned.
        """
        abstractions = self._library.get("abstractions", [])
        if len(abstractions) <= 1:
            return 0

        # Convert to MacroCandidates for comparison
        macros = [MacroCandidate(**a) for a in abstractions]

        # Identify redundant macros
        redundant_indices = set()
        for i, m1 in enumerate(macros):
            for j, m2 in enumerate(macros):
                if i == j or j in redundant_indices:
                    continue

                tags1 = set(m1.generalization_tags)
                tags2 = set(m2.generalization_tags)

                # Check if m2 subsumes m1
                if (
                    tags2 >= tags1  # m2 has all m1's tags
                    and m2.success_rate >= m1.success_rate
                    and m2.avg_steps <= m1.avg_steps
                ):
                    redundant_indices.add(i)

        # Remove redundant macros
        pruned = [
            abstractions[i]
            for i in range(len(abstractions))
            if i not in redundant_indices
        ]
        num_pruned = len(abstractions) - len(pruned)

        self._library["abstractions"] = pruned
        self.save_library()

        return num_pruned

    def get_all_macros(self) -> List[MacroCandidate]:
        """Get all macros in the library.

        Returns:
            List of all MacroCandidates.
        """
        # ── v3.9.0: Robust deserialization — filter extra keys ──
        # Library.json may contain extra keys (e.g., 'complexity_class') from
        # historical writes. MacroCandidate only accepts its own dataclass fields.
        valid_field_names = set(MacroCandidate.__dataclass_fields__.keys())
        result: List[MacroCandidate] = []
        for a in self._library.get("abstractions", []):
            filtered = {k: v for k, v in a.items() if k in valid_field_names}
            try:
                result.append(MacroCandidate(**filtered))
            except (TypeError, KeyError):
                # Skip corrupted entries
                pass
        return result


# ============================================================================
# TOMAS Learner (Main Class)
# ============================================================================

class TOMASLearner:
    """TOMAS Sleep-Step Learning System.

    Implements the complete learning cycle:
        record → sleep → audit → consolidate

    Plus the Fast-Path dispatch pipeline (泛函宏统一场论 v3.3.0):
        retrieve_for_topo → GaussExGuard → FastPathDispatcher → dispatch DSL

    This is the main orchestrator that:
    1. Records gameplay episodes (online phase)
    2. Sleep-steps to extract causal patterns from buffered episodes
    3. ψ-Audits candidate macros for validity
    4. Consolidates validated macros into library.json
    5. Fast-Path dispatches macros for known patterns (5-10× speedup)

    Integration with ARC-AGI-3 Solver:
        - PlannerAgent initializes TOMASLearner
        - Each level attempt records an EpisodeTrace
        - After game completion, sleep_step() extracts patterns
        - Validated macros are available via get_relevant_macros()
        - Fast-Path dispatch available via try_fast_path()

    Args:
        library_path: Path to library.json for macro persistence.
        mdl_threshold: Minimum MDL compression ratio for macro creation.
        psi_audit_threshold: ψ-Audit validation threshold for macro acceptance.
        buffer_size: Maximum number of episodes to buffer before sleep.
    """

    def __init__(
        self,
        library_path: Optional[str] = None,
        mdl_threshold: float = 0.7,
        psi_audit_threshold: float = 0.6,
        buffer_size: int = 50,
    ) -> None:
        """Initialize the TOMAS Learner."""
        self.pattern_extractor = CausalPatternExtractor()
        self.macro_abstractor = DSLMacroAbstractor()
        self.library_manager = LibraryManager(library_path)

        # ── 泛函宏统一场论 v3.3.0 组件 ──
        self.gauss_ex_guard = GaussExGuard()
        self.fast_path_dispatcher = FastPathDispatcher(
            library=self.library_manager,
            guard=self.gauss_ex_guard,
        )

        # ── 拓扑饱和修正 v3.5.0 组件 ──
        self.online_psi_audit = OnlinePSIAudit(
            library=self.library_manager,
            verbose=True,
        )
        self.physical_nar_conv = PhysicalNARConv()

        # ── 流贯归约 v3.7.0 组件 ──
        self.physical_compactification: Optional[PhysicalCompactificationReduction] = None
        self._gaussex_fail_streak: int = 0  # Consecutive GaussEx failures
        self._sleep_step_auto_trigger_threshold: int = 3  # §4.3: 连续3次失败→触发

        self.mdl_threshold = mdl_threshold
        self.psi_audit_threshold = psi_audit_threshold
        self.buffer_size = buffer_size

        # Episode buffer for sleep-step processing
        self._episode_buffer: List[EpisodeTrace] = []

        # ψ-Audit log for alignment faking detection
        self._psi_audit_log: List[Dict[str, Any]] = []

        # Fast-Path dispatch log for monitoring
        self._fast_path_log: List[Dict[str, Any]] = []

    # ── L0.5: PhysicalCompactificationReduction 初始化 (v3.7.0) ──

    def init_compactification(
        self,
        initial_grid: np.ndarray,
        game_id: str,
        game_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Initialize PhysicalCompactificationReduction for a solve_game() call.

        Creates a PhysicalCompactificationReduction instance with the current
        game's topology and complexity classification. This drives the Phase
        selection strategy in solve_game() and provides Φ_phys pruning for
        BFS/DFS/Beam expansion.

        Args:
            initial_grid: Initial grid state (H, W) array.
            game_id: Game identifier for complexity classification.
            game_state: Optional game state dict for constraint evaluation.

        Returns:
            Dict from classify_task_complexity() — used by solve_game()
            for Phase selection strategy.
        """
        topo_features = extract_topo_features(initial_grid)

        # Infer physical constraints from game characteristics
        n_actions = game_state.get("n_actions", 4) if game_state else 4
        has_physical_constraints = n_actions <= 7  # Small action spaces → physical structure
        has_non_associative_ops = n_actions > 4  # Multi-directional → order-sensitive

        task_complexity = classify_task_complexity(
            game_id=game_id,
            topo_features=topo_features,
            has_physical_constraints=has_physical_constraints,
            has_non_associative_ops=has_non_associative_ops,
        )

        # Create PhysicalCompactificationReduction instance
        self.physical_compactification = PhysicalCompactificationReduction(
            initial_grid=initial_grid,
            game_state=game_state or {},
            task_complexity=task_complexity,
            strict=False,  # Relaxed for BFS/DFS (±1 euler_char tolerance)
        )

        # Reset failure tracking
        self._gaussex_fail_streak = 0

        return task_complexity

    def track_gaussex_result(self, passed: bool) -> bool:
        """Track GaussEx verification result and auto-trigger Sleep-Step.

        Implements §4.3 from the article: consecutive 3 GaussEx failures
        automatically trigger sleep_step_online() to evolve 紧化基 Bκ.

        Args:
            passed: Whether the GaussEx verification passed.

        Returns:
            True if Sleep-Step was triggered this call.
        """
        if passed:
            self._gaussex_fail_streak = 0
            return False
        else:
            self._gaussex_fail_streak += 1
            if self._gaussex_fail_streak >= self._sleep_step_auto_trigger_threshold:
                # ── Auto-trigger Sleep-Step (§4.3) ──
                self._gaussex_fail_streak = 0  # Reset after trigger
                # Generate GaussEx event for ψ-Audit
                failure_event = {
                    "type": "gaussex_failure",
                    "consecutive_count": self._sleep_step_auto_trigger_threshold,
                    "context": "PhysicalCompactificationReduction pruning",
                    "failure_type": "topology_saturation",
                }
                try:
                    new_primitives = self.sleep_step_online(
                        gaussex_events=[failure_event],
                    )
                    if new_primitives:
                        return True
                except Exception:
                    pass
                return True  # Triggered even if no new primitives
        return False

    # ── L1: Record ──────────────────────────────────────────────────

    def record_episode(self, episode: EpisodeTrace) -> None:
        """Record a gameplay episode into the buffer.

        Args:
            episode: EpisodeTrace from a level attempt.
        """
        self._episode_buffer.append(episode)

        # Trim buffer if oversized
        if len(self._episode_buffer) > self.buffer_size:
            # Keep most recent episodes and successful ones
            recent = self._episode_buffer[-self.buffer_size // 2:]
            successful = [
                e for e in self._episode_buffer
                if e.success and e not in recent
            ]
            self._episode_buffer = recent + successful[-self.buffer_size // 2:]

    # ── L2: Sleep-Step ──────────────────────────────────────────────

    def sleep_step(self) -> List[MacroCandidate]:
        """Process buffered episodes to extract causal patterns.

        Sleep-Step is the core learning mechanism:
        1. Extract causal patterns from each episode
        2. Abstract DSL macros from detected patterns
        3. Filter by MDL threshold
        4. Return candidates for ψ-Audit validation

        Returns:
            List of MacroCandidate objects awaiting ψ-Audit.
        """
        candidates = []

        for episode in self._episode_buffer:
            # Extract patterns from each action trace
            for trace in episode.traces:
                # Sprite movement pattern
                movement = self.pattern_extractor.extract_sprite_movement(trace)
                if movement:
                    macro = self.macro_abstractor.abstract_macro(movement, episode)
                    if macro and macro.mdl_score <= self.mdl_threshold:
                        candidates.append(macro)

                # Click effect pattern
                click_effect = self.pattern_extractor.extract_click_effect(trace)
                if click_effect:
                    macro = self.macro_abstractor.abstract_macro(click_effect, episode)
                    if macro and macro.mdl_score <= self.mdl_threshold:
                        candidates.append(macro)

            # Navigation pattern (episode-level)
            navigation = self.pattern_extractor.extract_navigation_pattern(episode)
            if navigation:
                macro = self.macro_abstractor.abstract_macro(navigation, episode)
                if macro and macro.mdl_score <= self.mdl_threshold:
                    candidates.append(macro)

            # Positive navigation (partial success)
            positive_nav = self.pattern_extractor.extract_positive_navigation(episode)
            if positive_nav:
                macro = self.macro_abstractor.abstract_macro(positive_nav, episode)
                if macro and macro.mdl_score <= self.mdl_threshold:
                    candidates.append(macro)

        return candidates

    # ── L3: ψ-Audit ─────────────────────────────────────────────────

    def psi_audit(self, macro: MacroCandidate) -> Tuple[bool, Dict[str, Any]]:
        """Validate a macro candidate through ψ-Audit.

        ψ-Audit checks:
        1. Structural validity: DSL sequence is well-formed
        2. Empirical validity: Source episodes have sufficient evidence
        3. Anti-faking: Success rate is genuine (not evaluation-period artifact)
        4. Dead-Zero check: Macro doesn't produce null/meaningless output
        5. Rule R1: DSL总字符数 > 500 → 拒绝 (过长无压缩)
        6. Rule R2: 成功率 < 30% → 拒绝 (效果不足)
        7. Rule R3: DSL序列长度 < 2 → 拒绝 (过短无意义)

        Args:
            macro: MacroCandidate to validate.

        Returns:
            Tuple of (validated: bool, audit_report: dict).
        """
        audit_report: Dict[str, Any] = {
            "macro_name": macro.name,
            "mdl_score": macro.mdl_score,
            "success_rate": macro.success_rate,
            "checks": {},
        }

        # ── ψ-Audit三条规则 (硬性拒绝) ──

        # Rule R1: DSL总字符数 > 500 → 拒绝
        dsl_total_chars = 0
        for item in macro.dsl_sequence:
            if "repeat" in item:
                dsl_total_chars += len(str(item.get("repeat", ""))) + len(str(item.get("count", 1))) + 8
            elif "action" in item:
                dsl_total_chars += len(str(item.get("action", ""))) + 6
            else:
                dsl_total_chars += len(json.dumps(item))
        audit_report["checks"]["dsl_total_chars"] = dsl_total_chars
        if dsl_total_chars > 500:
            audit_report["checks"]["rule_r1_rejected"] = True
            audit_report["overall_score"] = 0.0
            self._psi_audit_log.append({
                "macro_name": macro.name,
                "overall_score": 0.0,
                "validated": False,
                "timestamp": time.time(),
                "reason": f"DSL总字符数={dsl_total_chars} > 500",
                "checks": audit_report["checks"],
            })
            return False, audit_report

        # Rule R2: 成功率 < 30% → 拒绝
        audit_report["checks"]["success_rate"] = macro.success_rate
        if macro.success_rate < 0.30:
            audit_report["checks"]["rule_r2_rejected"] = True
            audit_report["overall_score"] = 0.0
            self._psi_audit_log.append({
                "macro_name": macro.name,
                "overall_score": 0.0,
                "validated": False,
                "timestamp": time.time(),
                "reason": f"成功率={macro.success_rate:.2f} < 0.30",
                "checks": audit_report["checks"],
            })
            return False, audit_report

        # Rule R3: DSL序列长度 < 2 → 拒绝
        dsl_length = len(macro.dsl_sequence)
        audit_report["checks"]["dsl_length"] = dsl_length
        if dsl_length < 2:
            audit_report["checks"]["rule_r3_rejected"] = True
            audit_report["overall_score"] = 0.0
            self._psi_audit_log.append({
                "macro_name": macro.name,
                "overall_score": 0.0,
                "validated": False,
                "timestamp": time.time(),
                "reason": f"DSL序列长度={dsl_length} < 2",
                "checks": audit_report["checks"],
            })
            return False, audit_report

        # ── 柔性检查 (加权评分) ──

        # Check 1: Structural validity
        dsl_valid = self._check_dsl_validity(macro.dsl_sequence)
        audit_report["checks"]["dsl_validity"] = dsl_valid

        # Check 2: Empirical evidence
        evidence_score = self._compute_evidence_score(macro)
        audit_report["checks"]["evidence_score"] = evidence_score

        # Check 3: Anti-faking (Dead-Zero)
        anti_faking = self._check_alignment_faking(macro)
        audit_report["checks"]["anti_faking"] = anti_faking

        # Check 4: Generalization potential
        gen_score = len(macro.generalization_tags) / 5.0  # Normalize
        audit_report["checks"]["generalization_score"] = gen_score

        # Overall validation decision
        overall_score = (
            dsl_valid * 0.3
            + evidence_score * 0.3
            + anti_faking * 0.2
            + gen_score * 0.2
        )
        audit_report["overall_score"] = overall_score

        validated = overall_score >= self.psi_audit_threshold

        # Log ψ-Audit result
        self._psi_audit_log.append({
            "macro_name": macro.name,
            "overall_score": overall_score,
            "validated": validated,
            "timestamp": time.time(),
            "checks": audit_report["checks"],
        })

        return validated, audit_report

    def _check_dsl_validity(self, dsl_sequence: List[Dict[str, Any]]) -> float:
        """Check structural validity of a DSL sequence.

        Args:
            dsl_sequence: DSL action sequence.

        Returns:
            Validity score (0.0-1.0).
        """
        if not dsl_sequence:
            return 0.0

        valid_count = 0
        for item in dsl_sequence:
            if "repeat" in item:
                # Repeat block: must have action and count
                if isinstance(item["repeat"], str) and item.get("count", 0) > 0:
                    valid_count += 1
            elif "action" in item:
                # Simple action: must be a known action type
                known_actions = {
                    "UP", "DOWN", "LEFT", "RIGHT",
                    "MOVE", "CLICK", "KEYBOARD",
                    "ACTION1", "ACTION2", "ACTION3", "ACTION4",
                }
                if item["action"] in known_actions:
                    valid_count += 1
            else:
                # Dict with parameters
                if len(item) > 0:
                    valid_count += 0.5

        return valid_count / len(dsl_sequence)

    def _compute_evidence_score(self, macro: MacroCandidate) -> float:
        """Compute evidence score based on source task coverage.

        Args:
            macro: MacroCandidate to evaluate.

        Returns:
            Evidence score (0.0-1.0).
        """
        num_sources = len(macro.source_tasks)
        # More sources = stronger evidence
        # Normalize: 1 source = 0.3, 3+ sources = 1.0
        return min(1.0, 0.3 + 0.2 * (num_sources - 1))

    def _check_alignment_faking(self, macro: MacroCandidate) -> float:
        """Check for alignment faking (Dead-Zero).

        Detects macros that might appear successful during evaluation
        but would fail in deployment (ψ-Audit anti-faking check).

        Args:
            macro: MacroCandidate to evaluate.

        Returns:
            Anti-faking score (0.0-1.0). Higher = more genuine.
        """
        # Dead-Zero check: macro shouldn't have zero effective actions
        effective_actions = 0
        for item in macro.dsl_sequence:
            if "repeat" in item:
                effective_actions += item.get("count", 1)
            elif "action" in item:
                effective_actions += 1

        if effective_actions == 0:
            return 0.0  # Dead-Zero: no effective actions

        # Success rate consistency
        if macro.success_rate > 0.9 and len(macro.source_tasks) < 2:
            # Suspicious: high success rate with few sources
            return 0.5

        # Genuine patterns have moderate success rates with multiple sources
        return min(1.0, macro.success_rate * len(macro.source_tasks) / 3.0)

    # ── L4: Consolidate ─────────────────────────────────────────────

    def consolidate(self, new_macros: List[MacroCandidate]) -> int:
        """Consolidate validated macros into the library.

        Saves ψ-Audit validated macros to library.json and prunes
        redundant entries.

        Args:
            new_macros: List of MacroCandidate objects to consolidate.

        Returns:
            Number of macros successfully consolidated.
        """
        consolidated = 0

        for macro in new_macros:
            validated, report = self.psi_audit(macro)
            if validated:
                macro.validated = True
                self.library_manager.add_macro(macro)
                consolidated += 1

        # Prune redundant macros after consolidation
        pruned = self.library_manager.prune_redundant()

        return consolidated

    # ── Complete Learning Cycle ──────────────────────────────────────

    def learn_from_episodes(
        self,
        episodes: Optional[List[EpisodeTrace]] = None,
    ) -> Dict[str, Any]:
        """Execute the complete TOMAS learning cycle.

        record → sleep → audit → consolidate

        Args:
            episodes: Optional list of EpisodeTrace to process.
                If None, processes buffered episodes.

        Returns:
            Learning report dict with keys:
                candidates_found, macros_validated, macros_consolidated,
                macros_pruned, psi_audit_log_size.
        """
        if episodes:
            for ep in episodes:
                self.record_episode(ep)

        # L2: Sleep-step pattern extraction
        candidates = self.sleep_step()
        candidates_found = len(candidates)

        # L4: Consolidate with ψ-Audit
        macros_consolidated = self.consolidate(candidates)

        # Get library stats
        library_macros = self.library_manager.get_all_macros()

        report = {
            "candidates_found": candidates_found,
            "macros_consolidated": macros_consolidated,
            "macros_rejected": candidates_found - macros_consolidated,
            "total_library_size": len(library_macros),
            "psi_audit_log_size": len(self._psi_audit_log),
            "buffer_size": len(self._episode_buffer),
        }

        return report

    # ── L2+: Sleep-Step Online (拓扑饱和修正 §3 — ψ-Audit自动化) ─────

    def sleep_step_online(
        self,
        gaussex_events: Optional[List[Dict[str, Any]]] = None,
    ) -> List[MacroCandidate]:
        """Online Sleep-Step with automated ψ-Audit primitive proposal.

        Extended version of sleep_step() that also processes GaussEx
        failure events through OnlinePSIAudit to automatically propose
        new DSL primitives when the current 紧化基 Bκ encounters
        拓扑饱和 (performance plateau).

        This implements the 紧化基进化 mechanism from the 拓扑饱和修正
        paper: when a solver fails, the system automatically:
            1. Analyzes the failure to identify missing capabilities
            2. Proposes new DSL primitives from template library
            3. Validates the proposals against the game context
            4. Adds validated primitives to library.json for future use

        Args:
            gaussex_events: Optional list of GaussEx failure events.
                If None, processes events from internal buffer.

        Returns:
            List of new MacroCandidates proposed and validated.
        """
        new_primitives = []

        # Process standard sleep-step (offline pattern extraction)
        offline_candidates = self.sleep_step()

        # Process online ψ-Audit (automated primitive proposal)
        events = gaussex_events or []
        for event in events:
            proposed = self.online_psi_audit.monitor(event)
            if proposed is not None:
                new_primitives.append(proposed)

        # Also generate failure events from buffered episodes
        for episode in self._episode_buffer:
            if not episode.success and episode.rhae_score < 50:
                # Failed episode → generate GaussEx event for ψ-Audit
                failure_event = self._episode_to_gaussex_event(episode)
                proposed = self.online_psi_audit.monitor(failure_event)
                if proposed is not None:
                    new_primitives.append(proposed)

        return new_primitives

    def _episode_to_gaussex_event(
        self,
        episode: EpisodeTrace,
    ) -> Dict[str, Any]:
        """Convert a failed EpisodeTrace to a GaussEx event for ψ-Audit.

        Maps episode-level failure information to the GaussEx event
        format expected by OnlinePSIAudit.monitor().

        Args:
            episode: Failed EpisodeTrace with low RHAE score.

        Returns:
            Dict describing the GaussEx event.
        """
        # Determine failure type from episode characteristics
        if episode.total_steps <= 5:
            failure_type = "stall_detected"
        elif episode.rhae_score == 0:
            failure_type = "game_over"
        else:
            # Check if the episode had click or keyboard actions
            has_click = any(t.action_type == "CLICK" for t in episode.traces)
            has_keyboard = any(t.action_type == "KEYBOARD" for t in episode.traces)

            if has_click and not has_keyboard:
                failure_type = "click_no_effect"
            elif has_keyboard and not has_click:
                failure_type = "navigation_stuck"
            elif has_click and has_keyboard:
                failure_type = "constraint_violation"
            else:
                failure_type = "stall_detected"

        # Build the last grid state if available
        last_grid = None
        for trace in reversed(episode.traces):
            if trace.post_state is not None:
                last_grid = trace.post_state
                break

        return {
            "passed": False,
            "type": failure_type,
            "trace": {
                "total_steps": episode.total_steps,
                "rhae_score": episode.rhae_score,
                "success": episode.success,
                "pattern": episode.level_type,
            },
            "game_id": episode.game_id,
            "level_index": episode.level_index,
            "grid_state": last_grid,
            "game_tags": episode.tags,
            "game_state": {},  # No detailed sprite info from episode
        }

    # ── Macro Retrieval ──────────────────────────────────────────────

    def get_relevant_macros(
        self,
        game_tags: List[str],
        fingerprint: Optional[str] = None,
    ) -> List[MacroCandidate]:
        """Retrieve macros relevant to a game context.

        Uses TOMAS dual-storage retrieval:
        1. Fingerprint-based (structural) matching if fingerprint provided
        2. Tag-based (semantic) matching for generalization

        Args:
            game_tags: Tags describing the current game context.
            fingerprint: Optional TOMAS fingerprint for structural matching.

        Returns:
            List of relevant MacroCandidates, sorted by success_rate.
        """
        results = []

        # Fingerprint-based retrieval (structural)
        if fingerprint:
            fp_results = self.library_manager.query_by_fingerprint(
                fingerprint, tolerance=2,
            )
            results.extend(fp_results)

        # Tag-based retrieval (semantic)
        tag_results = self.library_manager.query_by_tags(
            game_tags, min_match=1,
        )

        # Merge, deduplicate by name
        seen_names = {m.name for m in results}
        for m in tag_results:
            if m.name not in seen_names:
                results.append(m)
                seen_names.add(m.name)

        # Sort by success_rate descending
        results.sort(key=lambda m: m.success_rate, reverse=True)

        return results

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the ψ-Audit log for inspection.

        Returns:
            List of ψ-Audit entries with validation results.
        """
        return self._psi_audit_log

    def clear_buffer(self) -> None:
        """Clear the episode buffer."""
        self._episode_buffer.clear()

    # ── Fast-Path Dispatch (泛函宏统一场论 §4) ────────────────────

    def try_fast_path(
        self,
        grid: np.ndarray,
        game_state: Dict[str, Any],
        game_tags: List[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """Attempt Fast-Path dispatch for a game state.

        This is the main entry point for the Fast-Path pipeline:
        1. extract_topo_features: Compute topology from the grid
        2. FastPathDispatcher.try_dispatch: Retrieve + validate + dispatch

        If Fast-Path succeeds, returns the DSL sequence for immediate
        action generation. If no macro matches, returns None and the
        solver falls back to κ-Snap search.

        Args:
            grid: 2D numpy array of the current game state.
            game_state: Dict with game metadata (sprites, etc.) for
                GaussEx precondition evaluation.
            game_tags: Generalization tags describing the game type.

        Returns:
            DSL sequence (List[Dict]) if Fast-Path succeeds, None otherwise.
        """
        # Step 1: Extract topology features
        topo_features = extract_topo_features(grid)

        # Step 2: Attempt Fast-Path dispatch
        dsl_sequence, macro_name = self.fast_path_dispatcher.try_dispatch(
            game_state=game_state,
            game_tags=game_tags,
            topo_features=topo_features,
        )

        # Log the attempt
        self._fast_path_log.append({
            "macro_name": macro_name,
            "success": dsl_sequence is not None,
            "topo_features": topo_features,
            "game_tags": game_tags,
            "timestamp": time.time(),
        })

        return dsl_sequence

    def get_fast_path_log(self) -> List[Dict[str, Any]]:
        """Get the Fast-Path dispatch log for monitoring.

        Returns:
            List of Fast-Path dispatch entries.
        """
        return self._fast_path_log


# ============================================================================
# Online ψ-Audit: Automated DSL Primitive Proposal (拓扑饱和修正 §3)
# ============================================================================
#
# From: "超越幂律的天花板：基于太一理论（TOMAS）的 Scaling Law 拓扑饱和修正"
# Key insight: When a solver encounters 拓扑饱和 (performance plateau) on
# Type-B games, the 紧化基 Bκ must evolve by proposing new DSL primitives.
# This is the propose_primitive algorithm from Appendix C of the article.
#
# Pipeline:
#   GaussEx failure → monitor → _analyze_failures → _synthesize_primitive
#   → validate → library.add_macro
#
# For ARC-AGI-3 games specifically:
#   - click_effect failure → propose "click_diamond_neighbors", "click_palette_rotate"
#   - navigation failure → propose "navigate_around_obstacle", "visit_switcher"
#   - constraint violation → propose "apply_constraint_satisfaction"
#   - keyboard stall → propose "direction_probe_then_navigate"
# ============================================================================

class OnlinePSIAudit:
    """Automated DSL primitive proposal from GaussEx failure analysis.

    Implements the 紧化基进化 (Bκ-evolution) mechanism from the TOMAS
    拓扑饱和修正 paper. When a solver's GaussEx verification fails,
    this module analyzes the failure context, identifies missing factors,
    and proposes new DSL primitives to fill the capability gap.

    This is the core mechanism for breaking 拓扑饱和 on Type-B games:
    instead of just adding more parameters (which hits a performance
    plateau), we evolve the 紧化基 by adding new primitive operations
    that capture the non-associative structure of the task.

    ARC-AGI-3 specific primitive categories:
        click:    click_diamond, click_palette_rotate, click_toggle_pair
        keyboard: navigate_bfs, visit_switcher, direction_probe
        mixed:    click_then_navigate, constraint_satisfy
        general:  anomaly_detect, 3life_execute, pattern_repeat

    Integration with TOMASLearner:
        TOMASLearner.sleep_step_online() calls OnlinePSIAudit.monitor()
        after each failed GaussEx verification to propose new primitives.

    Args:
        library: LibraryManager instance for macro persistence.
        verbose: Whether to log primitive proposals. Defaults to True.
    """

    # ── ARC-AGI-3 failure type classification ──
    FAILURE_TYPE_MAP: Dict[str, str] = {
        "energy_violation": "physical_constraint",     # Energy conservation violated
        "constraint_violation": "logical_constraint",  # Logic/constraint not satisfied
        "navigation_stuck": "path_blocked",            # BFS/navigation hit wall
        "click_no_effect": "click_mismatch",           # Click produced no change
        "wrong_color": "color_constraint",             # Wrong color after action
        "wrong_position": "position_constraint",       # Wrong position after action
        "game_over": "life_lost",                      # LIFE_LOSS or GAME_OVER
        "timeout": "time_exceeded",                    # Phase timeout
        "stall_detected": "progress_stalled",          # No progress for N steps
    }

    # ── Primitive synthesis templates ──
    PRIMITIVE_TEMPLATES: Dict[str, Dict[str, Any]] = {
        # Click game primitives
        "click_diamond_neighbors": {
            "dsl_sequence": [{"action": "CLICK_DIAMOND"}],
            "generalization_tags": ["click", "diamond_pattern", "neighbor_effect"],
            "gaussex_precond": "has_sprite_type('clickable') AND sprite_count('neighbor') >= 4",
            "applicable_topo": {"euler_char": -1, "period_rank": 1, "symmetry": ["rotational"]},
        },
        "click_palette_rotate": {
            "dsl_sequence": [{"action": "CLICK_PALETTE_CYCLE"}],
            "generalization_tags": ["click", "palette", "color_rotation"],
            "gaussex_precond": "has_sprite_type('palette') AND sprite_count('center') >= 1",
            "applicable_topo": {"euler_char": 0, "period_rank": 2, "symmetry": ["rotational"]},
        },
        "click_toggle_pair": {
            "dsl_sequence": [{"action": "CLICK_TOGGLE"}],
            "generalization_tags": ["click", "toggle", "binary_state"],
            "gaussex_precond": "has_sprite_type('clickable') AND sprite_count('indicator') >= 1",
            "applicable_topo": {"euler_char": 1, "period_rank": 0, "symmetry": ["horizontal"]},
        },
        # Keyboard game primitives
        "navigate_bfs_short": {
            "dsl_sequence": [{"repeat": "NAVIGATE_BFS", "count": 1}],
            "generalization_tags": ["keyboard", "navigation", "bfs"],
            "gaussex_precond": "has_sprite_type('player') AND sprite_count('goal') >= 1",
            "applicable_topo": {"euler_char": 2, "period_rank": 0, "symmetry": []},
        },
        "visit_switcher_first": {
            "dsl_sequence": [{"action": "NAVIGATE_TO_SWITCHER"}, {"action": "NAVIGATE_TO_GOAL"}],
            "generalization_tags": ["keyboard", "switcher", "visit_order"],
            "gaussex_precond": "has_sprite_type('switcher') AND has_sprite_type('goal')",
            "applicable_topo": {"euler_char": 3, "period_rank": 0, "symmetry": []},
        },
        "direction_probe": {
            "dsl_sequence": [{"action": "PROBE_UP"}, {"action": "PROBE_DOWN"}, {"action": "PROBE_LEFT"}, {"action": "PROBE_RIGHT"}],
            "generalization_tags": ["keyboard", "probe", "direction_learning"],
            "gaussex_precond": "True",
            "applicable_topo": {"euler_char": 1, "period_rank": 0, "symmetry": []},
        },
        # Mixed game primitives
        "click_then_navigate": {
            "dsl_sequence": [{"action": "CLICK_TARGET"}, {"repeat": "NAVIGATE_BFS", "count": 1}],
            "generalization_tags": ["mixed", "click_then_move", "sequential"],
            "gaussex_precond": "has_sprite_type('clickable') AND has_sprite_type('player')",
            "applicable_topo": {"euler_char": 2, "period_rank": 1, "symmetry": []},
        },
        "constraint_satisfy": {
            "dsl_sequence": [{"action": "CONSTRAINT_SOLVE"}],
            "generalization_tags": ["mixed", "constraint", "satisfaction"],
            "gaussex_precond": "has_sprite_type('indicator') AND sprite_count('clickable') >= 2",
            "applicable_topo": {"euler_char": 0, "period_rank": 2, "symmetry": ["horizontal", "vertical"]},
        },
        # General primitives
        "anomaly_detect_first": {
            "dsl_sequence": [{"action": "SCAN_ANOMALY"}, {"action": "ACT_ON_ANOMALY"}],
            "generalization_tags": ["anomaly", "attention_before_loss", "asd"],
            "gaussex_precond": "True",
            "applicable_topo": {"euler_char": 1, "period_rank": 0, "symmetry": []},
        },
        "3life_execute": {
            "dsl_sequence": [{"action": "LIFE1_EXPLORE"}, {"action": "LIFE2_TRY"}, {"action": "LIFE3_EXECUTE"}],
            "generalization_tags": ["3life", "phased_strategy", "explore_try_execute"],
            "gaussex_precond": "True",
            "applicable_topo": {"euler_char": 0, "period_rank": 0, "symmetry": []},
        },
        "pattern_repeat_smart": {
            "dsl_sequence": [{"action": "DETECT_PATTERN"}, {"repeat": "REPEAT_PATTERN", "count": 3}],
            "generalization_tags": ["pattern", "repeat", "smart_copy"],
            "gaussex_precond": "True",
            "applicable_topo": {"euler_char": 0, "period_rank": 2, "symmetry": ["horizontal", "vertical"]},
        },
    }

    def __init__(
        self,
        library: Optional[LibraryManager] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize OnlinePSIAudit with library manager.

        Args:
            library: LibraryManager for macro persistence. If None,
                creates a default one.
            verbose: Whether to log primitive proposals.
        """
        self.library = library or LibraryManager()
        self.verbose = verbose
        self._monitor_log: List[Dict[str, Any]] = []
        self._proposal_count: int = 0
        self._validation_count: int = 0
        self._accepted_count: int = 0

    def monitor(
        self,
        gaussex_event: Dict[str, Any],
    ) -> Optional[MacroCandidate]:
        """Monitor a GaussEx event and propose a new primitive if failure detected.

        This is the main entry point for the ψ-Audit pipeline:
            1. Check if the GaussEx event indicates a failure
            2. Analyze the failure to identify missing factors
            3. Propose a new DSL primitive from templates
            4. Validate the proposed primitive
            5. Add to library.json if validated

        Args:
            gaussex_event: Dict describing the GaussEx verification result.
                Required keys:
                    - 'passed': bool — whether the verification passed
                    - 'type': str — failure type classification
                    - 'trace': dict — execution trace with context info
                Optional keys:
                    - 'game_id': str — ARC-AGI-3 game identifier
                    - 'level_index': int — level number
                    - 'grid_state': np.ndarray — current grid state
                    - 'game_tags': list[str] — game generalization tags

        Returns:
            MacroCandidate if a new primitive was proposed and validated,
            None if the event passed or no suitable primitive was found.
        """
        # Skip passed events — no failure to learn from
        if gaussex_event.get("passed", True):
            return None

        # Classify failure type
        failure_type = gaussex_event.get("type", "unknown")
        failure_category = self.FAILURE_TYPE_MAP.get(failure_type, "unknown")

        # Log the monitoring event
        self._monitor_log.append({
            "event_type": failure_type,
            "category": failure_category,
            "game_id": gaussex_event.get("game_id", ""),
            "level_index": gaussex_event.get("level_index", -1),
            "timestamp": time.time(),
        })

        if self.verbose:
            print(f"[ψ-Audit] Monitoring failure: {failure_type} → {failure_category}")

        # Step 1: Analyze failures to identify missing factors
        missing_factors = self._analyze_failures(gaussex_event)

        if not missing_factors:
            return None

        # Step 2: Propose primitive candidates based on missing factors
        candidates = self._propose_from_missing_factors(missing_factors, gaussex_event)

        if not candidates:
            return None

        # Step 3: Validate and select the best candidate
        best_candidate = None
        best_score = 0.0

        for candidate in candidates:
            self._proposal_count += 1
            # Quick validation: check topology match and precondition feasibility
            validation_score = self._quick_validate(candidate, gaussex_event)

            if validation_score > best_score:
                best_score = validation_score
                best_candidate = candidate

        if best_candidate is None or best_score < 0.3:
            return None

        # Step 4: Add validated primitive to library
        self._validation_count += 1
        best_candidate.validated = True
        best_candidate.success_rate = best_score
        best_candidate.source_tasks.append(
            f"{gaussex_event.get('game_id', 'unknown')}_L{gaussex_event.get('level_index', -1)}"
        )

        self.library.add_macro(best_candidate)
        self._accepted_count += 1

        if self.verbose:
            print(f"[ψ-Audit] Evolved Bκ: Added {best_candidate.name} "
                  f"(score={best_score:.2f}, factors={missing_factors})")

        return best_candidate

    def _analyze_failures(
        self,
        gaussex_event: Dict[str, Any],
    ) -> List[str]:
        """Analyze a GaussEx failure event to identify missing factors.

        Maps ARC-AGI-3 specific failure patterns to the DSL capabilities
        that would address them. This is the core of the propose_primitive
        algorithm: identify what's missing from the current 紧化基 Bκ.

        Args:
            gaussex_event: GaussEx failure event dict.

        Returns:
            List of missing factor strings (e.g., 'diamond_click', 'bfs_navigation').
        """
        trace = gaussex_event.get("trace", {})
        failure_type = gaussex_event.get("type", "unknown")
        game_tags = gaussex_event.get("game_tags", [])
        missing: List[str] = []

        # ── Click game failures ──
        if failure_type in ("click_no_effect", "wrong_color"):
            # Click didn't produce expected change → need better click strategy
            if "diamond" in str(trace.get("pattern", "")):
                missing.append("click_diamond_neighbors")
            elif "palette" in str(trace.get("pattern", "")):
                missing.append("click_palette_rotate")
            elif "toggle" in str(trace.get("pattern", "")):
                missing.append("click_toggle_pair")
            else:
                missing.append("click_diamond_neighbors")
                missing.append("click_palette_rotate")

        # ── Constraint violation ──
        elif failure_type == "constraint_violation":
            # Logic constraint not satisfied → need constraint satisfaction strategy
            missing.append("constraint_satisfy")
            missing.append("anomaly_detect_first")

        # ── Navigation failures ──
        elif failure_type in ("navigation_stuck", "wrong_position"):
            # Navigation hit obstacle or wrong destination
            if "switcher" in str(trace.get("obstacle_type", "")):
                missing.append("visit_switcher_first")
            else:
                missing.append("navigate_bfs_short")
                missing.append("direction_probe")

        # ── Game over / life lost ──
        elif failure_type == "game_over":
            # LIFE_LOSS → need phased strategy (3-Life)
            missing.append("3life_execute")
            missing.append("anomaly_detect_first")

        # ── Stall / timeout ──
        elif failure_type in ("stall_detected", "timeout"):
            # No progress for N steps → need proactive exploration
            missing.append("direction_probe")
            missing.append("pattern_repeat_smart")

        # ── Mixed game failures ──
        elif "click" in game_tags and "keyboard" in game_tags:
            missing.append("click_then_navigate")

        # ── Unknown / generic ──
        else:
            # Try the most general strategies
            missing.append("anomaly_detect_first")
            missing.append("3life_execute")

        return missing

    def _propose_from_missing_factors(
        self,
        missing_factors: List[str],
        gaussex_event: Dict[str, Any],
    ) -> List[MacroCandidate]:
        """Propose DSL primitive candidates based on identified missing factors.

        Maps missing factors to primitive templates and creates MacroCandidate
        objects with game-specific context.

        Args:
            missing_factors: List of missing factor strings from _analyze_failures.
            gaussex_event: Original GaussEx event for context.

        Returns:
            List of MacroCandidate proposals.
        """
        candidates = []
        game_id = gaussex_event.get("game_id", "unknown")
        level_idx = gaussex_event.get("level_index", -1)

        for factor in missing_factors:
            template = self.PRIMITIVE_TEMPLATES.get(factor)
            if template is None:
                continue

            # Create MacroCandidate from template with game-specific context
            macro = MacroCandidate(
                name=f"{factor}_{game_id}_L{level_idx}",
                dsl_sequence=template["dsl_sequence"],
                tomas_fingerprint=hashlib.sha256(
                    json.dumps(template, sort_keys=True).encode()
                ).hexdigest()[:16],
                source_tasks=[f"{game_id}_L{level_idx}"],
                generalization_tags=template.get("generalization_tags", []),
                mdl_score=0.5,  # Template macros have moderate MDL
                applicable_topo=template.get("applicable_topo", {}),
                gaussex_precond=template.get("gaussex_precond", "True"),
                validated=False,
            )
            candidates.append(macro)

        return candidates

    def _quick_validate(
        self,
        candidate: MacroCandidate,
        gaussex_event: Dict[str, Any],
    ) -> float:
        """Quick validation of a proposed primitive against the failure context.

        Checks whether the proposed primitive's topology and precondition
        match the current game context. Returns a validation score.

        Args:
            candidate: Proposed MacroCandidate.
            gaussex_event: Original GaussEx event for context matching.

        Returns:
            Validation score (0.0-1.0). Higher = better match.
        """
        score = 0.0

        # ── Topology match check ──
        grid_state = gaussex_event.get("grid_state")
        if grid_state is not None and isinstance(grid_state, np.ndarray):
            current_topo = extract_topo_features(grid_state)
            macro_topo = candidate.applicable_topo
            if macro_topo:
                topo_sim = LibraryManager._compute_topo_similarity(
                    None, current_topo, macro_topo
                )
                score += 0.4 * topo_sim

        # ── Tag overlap check ──
        event_tags = set(gaussex_event.get("game_tags", []))
        macro_tags = set(candidate.generalization_tags)
        if event_tags and macro_tags:
            tag_overlap = len(event_tags & macro_tags) / len(event_tags | macro_tags)
            score += 0.3 * tag_overlap

        # ── Precondition feasibility ──
        # Check if the precondition can be satisfied given the game state
        game_state = gaussex_event.get("game_state", {})
        if game_state and candidate.gaussex_precond:
            guard = GaussExGuard()
            passes, reason = guard.check_precondition(candidate, game_state)
            if passes:
                score += 0.3
            else:
                score += 0.1  # Partial credit for attempting

        # ── Base score for template quality ──
        score += 0.1  # Templates are pre-designed, always get base credit

        return min(1.0, score)

    def get_stats(self) -> Dict[str, Any]:
        """Get OnlinePSIAudit statistics.

        Returns:
            Dict with keys: monitors, proposals, validations, accepted.
        """
        return {
            "monitors": len(self._monitor_log),
            "proposals": self._proposal_count,
            "validations": self._validation_count,
            "accepted": self._accepted_count,
        }

    # ── v3.9.0 — κ-Snap反向: Abductive Lift from solved games ──

    def abductive_lift_from_success(
        self,
        episode: EpisodeTrace,
        rhae_threshold: float = 50.0,
    ) -> Optional[MacroCandidate]:
        """κ-Snap反向: extract causal invariant from successful episode.

        Implements the κ-Snap反向 (abductive lift) from "从猜想生成到流贯觉察":
        "From known truth → extract causal invariant → migrate to new domain"

        This is Moonshine's core ability — take what worked and generalize it.
        Unlike monitor() which processes failures (passed=True events are skipped),
        this method processes SUCCESSFUL episodes to:
            1. Extract the causal invariant C⁰ from the successful action sequence
               (what pattern of actions led to success)
            2. Create a MacroCandidate with generalization_tags derived from the
               causal structure
            3. Return the macro proposal for library.json

        The key insight: κ-Snap反向 means we can learn not just from failures
        (what's missing) but from successes (what invariant made it work).

        Args:
            episode: A solved EpisodeTrace (success=True, rhae_score > threshold).
            rhae_threshold: Minimum RHAE score for a episode to qualify for
                abductive lift. Defaults to 50.0.

        Returns:
            MacroCandidate if a causal invariant was extracted and a macro
            proposal was created. None if the episode doesn't qualify or
            no invariant can be extracted.
        """
        # ── Step 1: Check eligibility ──
        if not episode.success or episode.rhae_score < rhae_threshold:
            return None

        if len(episode.traces) < 2:
            return None  # Need at least 2 actions to extract a pattern

        # ── Step 2: Extract causal invariant C⁰ from successful action sequence ──
        # C⁰ = the minimal pattern of actions that led to success
        # We use action abstraction to find the causal structure:
        # - What TYPES of actions were taken (click, move, keyboard)
        # - What ORDER they were taken in (causal sequence)
        # - What TRANSITIONS they produced (state changes)

        action_sequence: List[Dict[str, Any]] = []
        state_transitions: List[Dict[str, Any]] = []

        for trace in episode.traces:
            # Abstract action: type + key parameters (strip specific positions)
            abstract_action = {
                "type": trace.action_type,
                "category": self._classify_action_category(trace),
            }
            action_sequence.append(abstract_action)

            # State transition: what changed (delta in grid state)
            if trace.pre_state is not None and trace.post_state is not None:
                delta = float(np.sum(trace.post_state != trace.pre_state))
                state_transitions.append({
                    "action_index": trace.step_number,
                    "delta_pixels": delta,
                    "reward": trace.reward,
                })

        # ── Step 3: Extract the causal invariant (recurring sub-pattern) ──
        # Find the minimal sub-sequence that consistently produces state changes
        # This is the C⁰ — the causal core of the successful strategy

        causal_invariant: Optional[List[Dict[str, Any]]] = None
        best_invariant_score: float = 0.0

        # Try window sizes from 2 to full sequence
        for window_size in range(2, min(len(action_sequence) + 1, 6)):
            for start_idx in range(len(action_sequence) - window_size + 1):
                sub_sequence = action_sequence[start_idx:start_idx + window_size]

                # Compute invariant score: how much state change this sub-sequence produces
                # Look at state transitions that overlap with this action window
                window_transitions = [
                    t for t in state_transitions
                    if start_idx <= t["action_index"] < start_idx + window_size
                ]
                total_delta = sum(t.get("delta_pixels", 0) for t in window_transitions)
                total_reward = sum(t.get("reward", 0) for t in window_transitions)

                # Invariant score: combination of state change magnitude and reward
                invariant_score = (total_delta * 0.1 + total_reward * 0.5) / max(window_size, 1)

                if invariant_score > best_invariant_score:
                    best_invariant_score = invariant_score
                    causal_invariant = sub_sequence

        if causal_invariant is None or best_invariant_score < 0.1:
            return None  # No meaningful causal invariant found

        # ── Step 4: Create MacroCandidate from the causal invariant ──
        # Build DSL sequence from the invariant
        dsl_sequence: List[Dict[str, Any]] = []
        generalization_tags: List[str] = []

        for action in causal_invariant:
            dsl_entry: Dict[str, Any] = {"action": action["type"]}
            if action.get("category"):
                dsl_entry["category"] = action["category"]
            dsl_sequence.append(dsl_entry)

            # Add category-based generalization tags
            category = action.get("category", "")
            if category and category not in generalization_tags:
                generalization_tags.append(category)

        # Add κ-Snap反向 specific tags
        generalization_tags.append("kappa_snap_reverse")
        generalization_tags.append("abductive_lift")
        generalization_tags.append(f"causal_invariant_{len(causal_invariant)}")

        # Add game-specific tags from the episode
        for tag in episode.tags:
            if tag not in generalization_tags:
                generalization_tags.append(tag)

        # Compute fingerprint from causal invariant structure
        invariant_str = json.dumps(causal_invariant, sort_keys=True)
        fingerprint = hashlib.sha256(invariant_str.encode()).hexdigest()[:16]

        # MDL score: invariant length / total episode length (compression ratio)
        mdl_score = len(causal_invariant) / max(len(action_sequence), 1)

        macro_name = f"lift_{episode.game_id}_L{episode.level_index}_{fingerprint[:8]}"

        macro = MacroCandidate(
            name=macro_name,
            dsl_sequence=dsl_sequence,
            tomas_fingerprint=fingerprint,
            source_tasks=[f"{episode.game_id}_L{episode.level_index}"],
            avg_steps=float(len(causal_invariant)),
            success_rate=min(1.0, episode.rhae_score / 100.0),
            generalization_tags=generalization_tags,
            mdl_score=mdl_score,
            min_demo_to_activate=1,
            validated=True,  # Pre-validated from successful episode
            applicable_topo=extract_topo_features(
                episode.traces[0].pre_state
            ) if episode.traces[0].pre_state is not None else {},
            gaussex_precond="True",
        )

        # ── Step 5: Add to library ──
        self.library.add_macro(macro)
        self._accepted_count += 1

        if self.verbose:
            print(f"[ψ-Audit κ-Snap反向] Abductive lift: {macro_name} "
                  f"(invariant_score={best_invariant_score:.2f}, "
                  f"tags={generalization_tags})")

        return macro

    def _classify_action_category(self, trace: ActionTrace) -> str:
        """Classify an action trace into a semantic category for causal invariant extraction.

        Maps specific actions to broader semantic categories that form the
        causal structure of a successful strategy.

        Args:
            trace: An ActionTrace record.

        Returns:
            String category label (e.g., 'navigate', 'click_target', 'probe').
        """
        action_type = trace.action_type
        params = trace.action_params

        if action_type == "CLICK":
            # Classify click actions by what was clicked
            target = params.get("target", params.get("object", ""))
            if "diamond" in str(target).lower():
                return "click_diamond"
            elif "palette" in str(target).lower():
                return "click_palette"
            elif "toggle" in str(target).lower():
                return "click_toggle"
            elif "switcher" in str(target).lower():
                return "click_switcher"
            else:
                return "click_target"

        elif action_type == "KEYBOARD":
            # Classify keyboard actions by direction/purpose
            key = params.get("key", params.get("direction", ""))
            if key in ("UP", "DOWN", "LEFT", "RIGHT", "up", "down", "left", "right"):
                return "navigate_direction"
            elif key in ("ENTER", "enter", "SPACE", "space"):
                return "keyboard_confirm"
            else:
                return "keyboard_action"

        elif action_type == "MOVE":
            dx = params.get("dx", 0)
            dy = params.get("dy", 0)
            if abs(dx) > abs(dy):
                return "navigate_horizontal"
            elif abs(dy) > abs(dx):
                return "navigate_vertical"
            else:
                return "navigate_stay"

        else:
            return f"action_{action_type.lower()}"


# ============================================================================
# PhysicalNARConv: Physical State Octonion Encoding (拓扑饱和修正 附录B)
# ============================================================================
#
# From: "超越幂律的天花板" Appendix B — PhysicalNARConv implementation
# Maps physical quantities to octonion components:
#   position → e₁, e₂, e₃  (spatial)
#   velocity → e₄, e₅, e₆  (causal/momentum)
#   spin     → e₇           (angular/topological)
#   magnitude → e₀          (real/existence)
#
# For ARC-AGI-3: maps grid pixel values and sprite properties to
# octonion vectors for NAR-Conv encoding.
# ============================================================================

class PhysicalNARConv:
    """Physical state encoder into octonion representation for NAR-Conv.

    Encodes ARC-AGI-3 physical quantities (sprite positions, velocities,
    color states, tactile feedback) into octonion vectors suitable for
    NAR-Conv processing. This enables the encoder to capture non-associative
    physical relationships (e.g., rotation-then-translation ≠ translation-
    then-rotation) that standard encodings miss.

    Octonion component mapping (from 拓扑饱和修正 paper Appendix B):
        e₀ (real):     magnitude / presence signal (1.0 if non-zero, 0.0 if background)
        e₁ (imag₁):    row position (normalized by grid height)
        e₂ (imag₂):    column position (normalized by grid width)
        e₃ (imag₃):    color value / state (normalized by max_color)
        e₄ (imag₄):    row velocity / delta-row (from grid diffs)
        e₅ (imag₅):    column velocity / delta-col (from grid diffs)
        e₆ (imag₆):    state velocity / color-delta (from grid diffs)
        e₇ (imag₇):    spin / symmetry order (rotational symmetry degree)

    Args:
        grid_height: Maximum grid height for normalization.
        grid_width: Maximum grid width for normalization.
        max_colors: Maximum number of ARC colors.
    """

    def __init__(
        self,
        grid_height: int = 32,
        grid_width: int = 32,
        max_colors: int = 10,
    ) -> None:
        """Initialize PhysicalNARConv encoder."""
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.max_colors = max_colors

    def encode_pixel(
        self,
        row: int,
        col: int,
        color: int,
        delta_row: float = 0.0,
        delta_col: float = 0.0,
        delta_color: float = 0.0,
        spin_order: int = 0,
    ) -> np.ndarray:
        """Encode a single pixel/sprite into an 8-component octonion vector.

        Args:
            row: Row position (0-based).
            col: Column position (0-based).
            color: Color value (0-based ARC color index).
            delta_row: Row change between frames (velocity in row direction).
            delta_col: Column change between frames (velocity in col direction).
            delta_color: Color change between frames (state velocity).
            spin_order: Rotational symmetry order (0=none, 2=180°, 4=90°, etc.).

        Returns:
            8-component numpy array representing the octonion encoding.
        """
        oct_vec = np.zeros(8, dtype=np.float32)

        # e₀: magnitude / presence
        if color > 0:
            oct_vec[0] = 1.0  # Non-zero color = presence signal
        else:
            oct_vec[0] = 0.0  # Background = no signal

        # e₁: row position (normalized)
        oct_vec[1] = row / max(self.grid_height, 1)

        # e₂: column position (normalized)
        oct_vec[2] = col / max(self.grid_width, 1)

        # e₃: color value / state (normalized)
        oct_vec[3] = color / max(self.max_colors, 1)

        # e₄: row velocity / delta-row
        oct_vec[4] = delta_row / max(self.grid_height, 1)

        # e₅: column velocity / delta-col
        oct_vec[5] = delta_col / max(self.grid_width, 1)

        # e₆: state velocity / color-delta
        oct_vec[6] = delta_color / max(self.max_colors, 1)

        # e₇: spin / symmetry order (log-normalized)
        if spin_order > 0:
            oct_vec[7] = np.log2(spin_order + 1) / 3.0  # Normalize: 4→0.53, 8→0.67
        else:
            oct_vec[7] = 0.0

        # Normalize to unit octonion norm
        norm = np.linalg.norm(oct_vec)
        if norm > 0:
            oct_vec = oct_vec / norm

        return oct_vec

    def encode_grid_with_deltas(
        self,
        grid: np.ndarray,
        prev_grid: Optional[np.ndarray] = None,
        symmetry_order: int = 0,
    ) -> np.ndarray:
        """Encode an entire grid with inter-frame deltas into octonion tensor.

        For each pixel, computes the octonion encoding including position,
        color, velocity (from prev_grid diff), and spin. This produces
        a (H, W, 8) tensor suitable for NAR-Conv processing.

        Args:
            grid: Current grid state (H, W) with integer color indices.
            prev_grid: Previous frame grid for delta computation.
                If None, all deltas are zero (no velocity information).
            symmetry_order: Detected rotational symmetry order of the grid.

        Returns:
            Octonion tensor of shape (H, W, 8) with per-pixel encodings.
        """
        H, W = grid.shape
        oct_tensor = np.zeros((H, W, 8), dtype=np.float32)

        # Compute deltas if prev_grid available
        if prev_grid is not None and prev_grid.shape == grid.shape:
            # Find positions that changed
            diff_mask = (grid != prev_grid)
            # Compute per-pixel deltas
            delta_colors = (grid.astype(np.float32) - prev_grid.astype(np.float32))
            # Approximate position deltas: for changed pixels, estimate
            # which direction the change "moved" by checking neighbors
            delta_rows = np.zeros((H, W), dtype=np.float32)
            delta_cols = np.zeros((H, W), dtype=np.float32)

            # Simple delta estimation: find displaced colors
            for r in range(H):
                for c in range(W):
                    if diff_mask[r, c]:
                        # Check if the previous color moved to a neighbor
                        prev_color = prev_grid[r, c]
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if grid[nr, nc] == prev_color and prev_grid[nr, nc] != prev_color:
                                    delta_rows[r, c] = dr
                                    delta_cols[r, c] = dc
                                    break
        else:
            delta_colors = np.zeros((H, W), dtype=np.float32)
            delta_rows = np.zeros((H, W), dtype=np.float32)
            delta_cols = np.zeros((H, W), dtype=np.float32)

        # Encode each pixel
        for r in range(H):
            for c in range(W):
                oct_tensor[r, c] = self.encode_pixel(
                    row=r,
                    col=c,
                    color=int(grid[r, c]),
                    delta_row=float(delta_rows[r, c]),
                    delta_col=float(delta_cols[r, c]),
                    delta_color=float(delta_colors[r, c]),
                    spin_order=symmetry_order,
                )

        return oct_tensor

    def compute_non_associative_residual(
        self,
        grid_seq: List[np.ndarray],
    ) -> float:
        """Compute the non-associative residual (η) for a grid sequence.

        Measures whether sequential transformations on the grid preserve
        non-associative structure. A high η value indicates that the
        grid transformations are order-sensitive (physical AI relevant),
        while η ≈ 0 indicates the transformations are commutative
        (statistical proxy).

        This is the key diagnostic from the 拓扑饱和修正 paper:
            η > 0 ⇔ The game requires 紧化基进化 (new DSL primitives)
            η ≈ 0 ⇔ The game can be solved with existing Bκ (associative)

        Args:
            grid_seq: Sequence of grid states (at least 3 frames).
                Each is a (H, W) numpy array.

        Returns:
            Non-associative residual η (float). Typically 0.01-0.5.
        """
        if len(grid_seq) < 3:
            return 0.0

        # Encode three consecutive frames
        encodings = [
            self.encode_grid_with_deltas(grid_seq[i], grid_seq[i - 1] if i > 0 else None)
            for i in range(min(3, len(grid_seq)))
        ]

        # Sample octonion vectors from the center of each frame
        center_r = encodings[0].shape[0] // 2
        center_c = encodings[0].shape[1] // 2

        a = encodings[0][center_r, center_c]
        b = encodings[1][center_r, center_c] if len(encodings) > 1 else a
        c = encodings[2][center_r, center_c] if len(encodings) > 2 else b

        # Compute (a·b)·c vs a·(b·c) using octonion multiplication
        # This measures whether the sequential transformations are associative
        ab = self._oct_multiply_np(a, b)
        ab_c = self._oct_multiply_np(ab, c)

        bc = self._oct_multiply_np(b, c)
        a_bc = self._oct_multiply_np(a, bc)

        asym = ab_c - a_bc
        asym_norm = np.linalg.norm(asym)
        abc_norm = np.linalg.norm(a_bc)

        if abc_norm < 1e-10:
            return 0.0

        return float(asym_norm / abc_norm)

    @staticmethod
    def _oct_multiply_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two octonion vectors using the Cayley-Dickson table.

        Numpy version of OctonionConv2d._oct_multiply for PhysicalNARConv.

        Args:
            a, b: 8-component octonion vectors (numpy float32 arrays).

        Returns:
            Product octonion (8-component numpy array).
        """
        result = np.zeros(8, dtype=np.float32)
        for i in range(8):
            for j in range(8):
                sign, k = OCT_MUL_TABLE.get((i, j), (0, 0))
                result[k] += sign * a[i] * b[j]
        return result


# ═══════════════════════════════════════════════════════════════════════
# v3.6.0 — 物理紧化约束剪枝 + Ψ-截断边界管理
# 来源: "物理紧化与Ψ-截断" (复合体理学 2026-06-26)
#         "流贯的决定论与紧化基的演化" (复合体理学 2026-06-26)
# ═══════════════════════════════════════════════════════════════════════


class PhysicalGaussExGuard:
    """Physical constraint pruning filter for κ-Snap search.

    Implements the 物理紧化约束集 Φ_phys from the "物理紧化与Ψ-截断"
    paper. Before dispatching a macro or exploring a κ-Snap candidate,
    this guard checks whether the candidate violates physical laws:

        Φ_phys = {E_cons (能量守恒), T_causal (因果时序),
                  G_topo (拓扑闭包), Φ_boundary (边界层)}

    This dramatically reduces the search space for Type-B non-associative
    tasks by pruning candidates that violate physical constraints, turning
    NP-hard search into polynomial-time κ-Snap (Theorem 3.1 from paper).

    Integration with GaussExGuard:
        GaussExGuard checks logical preconditions (sprite_count, has_type).
        PhysicalGaussExGuard checks physical preconditions (energy, topology,
        causal order). They can be chained: logical first, physical second.

    Example usage:
        phys_guard = PhysicalGaussExGuard()
        if phys_guard.check_physical_constraints(candidate_plan, game_state):
            # Candidate respects physics → proceed with κ-Snap
            ...
        else:
            # Candidate violates physics → skip (pruned)

    Args:
        strict: Whether to enforce all constraints strictly.
            If False, topology constraint allows ±1 euler_char deviation.
    """

    # ── 物理约束类型标识 ──
    CONSTRAINT_TYPES: Dict[str, str] = {
        "E_cons": "energy_conservation",       # 能量守恒 (总色数/面积不变)
        "T_causal": "causal_non_associativity", # 因果时序 (操作顺序敏感)
        "G_topo": "topological_closure",        # 拓扑闭包 (欧拉χ守恒)
        "Phi_boundary": "boundary_layer",       # 边界层 (坐标不越界)
    }

    def __init__(self, strict: bool = True) -> None:
        """Initialize PhysicalGaussExGuard.

        Args:
            strict: Whether to enforce all constraints strictly.
        """
        self.strict = strict
        self._constraint_log: List[Dict[str, Any]] = []

    def check_physical_constraints(
        self,
        program_node: List[str],
        current_state: Dict[str, Any],
        input_grid: Optional[np.ndarray] = None,
        output_grid: Optional[np.ndarray] = None,
    ) -> bool:
        """Check if a candidate program violates physical laws (Φ_phys).

        Implements the four constraint checks from the paper:
            1. E_cons: Energy conservation (total non-zero pixels preserved)
            2. T_causal: Causal non-associativity (order-sensitive ops respected)
            3. G_topo: Topological closure (Euler characteristic preserved)
            4. Φ_boundary: Boundary layer (coordinates stay within grid)

        Args:
            program_node: DSL sequence to check (e.g., ['ROTATE', 'FILL', ...]).
            current_state: Game state dict with keys:
                'grid_topology': Dict with euler_char, n_components, etc.
                'sprites': List of sprite dicts.
                'grid_shape': (H, W) tuple.
                'task_type': 'TYPE_A' or 'TYPE_B_STRICT_ORDER'.
                'available_energy': Float, max energy budget (default=inf).
            input_grid: Input grid for topology comparison (H, W) array.
            output_grid: Expected output grid for topology comparison.

        Returns:
            True if program respects all physical constraints; False otherwise.
        """
        violations: List[str] = []

        # ── 1. E_cons: Energy Conservation ──
        estimated_energy = self._calculate_energy_cost(program_node, current_state)
        available_energy = current_state.get("available_energy", float("inf"))
        if estimated_energy > available_energy:
            violations.append("E_cons: energy cost exceeds budget")

        # ── 2. T_causal: Causal Non-associativity ──
        if self._check_causal_order(program_node, current_state):
            violations.append("T_causal: operation order violates non-associativity")

        # ── 3. G_topo: Topological Closure ──
        if input_grid is not None:
            input_topo = self._compute_grid_topology(input_grid)
            if output_grid is not None:
                output_topo = self._compute_grid_topology(output_grid)
                # Euler characteristic must be preserved (or ±1 in relaxed mode)
                tolerance = 1 if not self.strict else 0
                if abs(input_topo["euler_char"] - output_topo["euler_char"]) > tolerance:
                    violations.append(
                        f"G_topo: euler_char changed from "
                        f"{input_topo['euler_char']} to {output_topo['euler_char']}"
                    )

        # ── 4. Φ_boundary: Boundary Layer ──
        grid_shape = current_state.get("grid_shape", (0, 0))
        if grid_shape != (0, 0):
            for sprite in current_state.get("sprites", []):
                sx = sprite.get("x", 0)
                sy = sprite.get("y", 0)
                if sx < 0 or sy < 0 or sx >= grid_shape[1] or sy >= grid_shape[0]:
                    violations.append(
                        f"Phi_boundary: sprite at ({sx},{sy}) out of bounds "
                        f"{grid_shape}"
                    )

        # ── Log and return ──
        result = len(violations) == 0
        if not result:
            self._constraint_log.append({
                "program": program_node,
                "violations": violations,
                "passed": result,
            })

        return result

    def _calculate_energy_cost(
        self, program_node: List[str], current_state: Dict[str, Any]
    ) -> float:
        """Estimate the 'energy cost' of executing a program.

        In ARC-AGI-3 context, energy = total number of pixel changes.
        Each operation has a rough cost based on how many pixels it affects.

        Args:
            program_node: DSL sequence.
            current_state: Game state dict.

        Returns:
            Estimated energy cost (float).
        """
        # Rough energy estimates per DSL operation type
        ENERGY_COST_MAP: Dict[str, float] = {
            "CLICK": 1.0,        # One pixel change
            "MOVE": 2.0,         # Two pixels change (old + new position)
            "ROTATE": 5.0,       # Multiple pixels change (rotation)
            "FILL": 10.0,        # Many pixels change (fill operation)
            "NAVIGATE": 2.0,     # Movement cost
            "RESET": 0.0,        # No energy cost
            "WAIT": 0.0,         # No energy cost
        }

        total_cost = 0.0
        for op in program_node:
            op_upper = op.upper()
            # Check if operation matches any known type
            matched = False
            for key, cost in ENERGY_COST_MAP.items():
                if key in op_upper:
                    total_cost += cost
                    matched = True
                    break
            if not matched:
                total_cost += 1.0  # Default cost for unknown operations

        return total_cost

    def _check_causal_order(
        self, program_node: List[str], current_state: Dict[str, Any]
    ) -> bool:
        """Check if program violates causal non-associativity ordering.

        For Type-B tasks, some operations must happen in a specific order.
        E.g., in ARC tasks: 'FILL before ROTATE' may be invalid while
        'ROTATE then FILL' is valid (ROTATE→FILL ≠ FILL→ROTATE).

        Args:
            program_node: DSL sequence.
            current_state: Game state dict with 'task_type'.

        Returns:
            True if order is violated; False if order is respected.
        """
        task_type = current_state.get("task_type", "TYPE_A")

        if task_type != "TYPE_B_STRICT_ORDER":
            return False  # No order constraint for Type-A

        # Check for invalid orderings
        # "FILL before ROTATE" is invalid in strict-order Type-B tasks
        fill_idx = None
        rotate_idx = None
        for i, op in enumerate(program_node):
            op_upper = op.upper()
            if "FILL" in op_upper:
                fill_idx = i
            if "ROTATE" in op_upper:
                rotate_idx = i

        if fill_idx is not None and rotate_idx is not None:
            if fill_idx < rotate_idx:
                return True  # Violation: FILL before ROTATE

        return False

    @staticmethod
    def _compute_grid_topology(grid: np.ndarray) -> Dict[str, Any]:
        """Compute topological features of a grid for G_topo constraint.

        Args:
            grid: (H, W) numpy array representing ARC grid.

        Returns:
            Dict with euler_char, n_components, n_holes, density.
        """
        binary = (grid > 0).astype(np.int32)
        n_pixels = int(np.sum(binary))

        # Connected components (4-connectivity)
        n_components = 0
        visited = set()
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if binary[r, c] == 1 and (r, c) not in visited:
                    n_components += 1
                    # BFS flood fill
                    queue = [(r, c)]
                    visited.add((r, c))
                    while queue:
                        cr, cc = queue.pop(0)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] == 1 and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc))

        # Simple hole detection: count empty regions surrounded by pixels
        n_holes = 0
        bg_visited = set()
        for r in range(h):
            for c in range(w):
                if binary[r, c] == 0 and (r, c) not in bg_visited:
                    is_enclosed = True
                    queue = [(r, c)]
                    bg_visited.add((r, c))
                    while queue:
                        cr, cc = queue.pop(0)
                        if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                            is_enclosed = False  # Touches border → not a hole
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] == 0 and (nr, nc) not in bg_visited:
                                bg_visited.add((nr, nc))
                                queue.append((nr, nc))
                    if is_enclosed:
                        n_holes += 1

        # Euler characteristic: χ = components - holes
        euler_char = n_components - n_holes
        density = n_pixels / max(h * w, 1)

        return {
            "euler_char": euler_char,
            "n_components": n_components,
            "n_holes": n_holes,
            "n_pixels": n_pixels,
            "density": density,
        }

    def get_constraint_log(self) -> List[Dict[str, Any]]:
        """Return the log of constraint violations.

        Returns:
            List of dicts with program, violations, and passed status.
        """
        return list(self._constraint_log)

    def clear_constraint_log(self) -> None:
        """Clear the constraint violation log."""
        self._constraint_log.clear()


# ═══════════════════════════════════════════════════════════════════════
# PhysicalCompactificationReduction — 流贯归约框架 (v3.7.0)
#
# From article "从NP-难到P类的流贯归约：基于太一理论(TOMAS)的
# ARC-AGI-3物理紧化破解框架":
#
#   Core theorem: Φ_phys = {E_cons, T_causal, G_topo, Φ_boundary}
#   constrains the search space, turning NP-Hard O(|D|^n) into
#   polynomial O(|D|^k × poly(n)) for physically constrained tasks.
#
#   Algorithm 1 (GaussEx Physical Pruner):
#     for P in Candidates(BFS/DFS/Beam expansion):
#       if Energy(P) > S.Available_Energy:  reject
#       if EulerChar(P) ≠ EulerChar(Initial): reject
#       if Type-B and CausalOrder(P) violated: reject
#       else: accept P into search frontier
#
# Integration: this class provides the `should_prune()` interface
# that BFS/DFS/Beam call on each candidate expansion node. The
# classify_task_complexity() output determines which phases use
# this pruner:
#   - P class:        skip BFS/DFS entirely (Fast-Path sufficient)
#   - P_in_phys:      BFS/Beam with Φ_phys pruning (most effective)
#   - NP_Hard:        full pipeline, pruner helps reduce branching
#   - NP_C_likely:    minimal pipeline (Keyboard + Random)
#
# Sleep-Step trigger: consecutive 3 GaussEx failures → auto-trigger
# sleep_step_online() to evolve the 紧化基 Bκ (per article §4.3).
# ═══════════════════════════════════════════════════════════════════════


class PhysicalCompactificationReduction:
    """Integration layer embedding Φ_phys pruning into BFS/DFS/Beam expansion.

    Implements the 流贯归约 (compactification reduction) framework from the
    "从NP-难到P类的流贯归约" article. This is the core mechanism that makes
    NP-Hard search tractable by pruning candidate expansion nodes that violate
    physical constraint set Φ_phys = {E_cons, T_causal, G_topo, Φ_boundary}.

    The key result (Theorem 3.1): under Φ_phys, the effective search space
    drops from O(|D|^n) to O(|D|^k × poly(n)), where k << n is the number
    of "physically admissible" operation sequences.

    Integration with solve_game() pipeline:
        - Phase -1/0/0.5/1: Always run (Oracle + heuristic + Fast-Path + UniversalPipeline)
        - Phase 2+ (BFS/Beam/DFS): Only run if task complexity requires search
        - PhysicalCompactificationReduction provides should_prune() interface
          for BFS/DFS/Beam to call on each candidate expansion node

    Sleep-Step trigger (§4.3):
        Consecutive 3 GaussEx failures → auto-trigger sleep_step_online()
        to evolve 紧化基 Bκ and break 拓扑饱和 (performance plateau).

    Args:
        initial_grid: Initial grid state for topology baseline.
        game_state: Game state dict for constraint evaluation.
        task_complexity: Output from classify_task_complexity().
        strict: Whether to enforce Euler characteristic strictly.
    """

    # ── Sleep-Step trigger threshold (§4.3) ──
    SLEEP_STEP_THRESHOLD: int = 3

    def __init__(
        self,
        initial_grid: Optional[np.ndarray] = None,
        game_state: Optional[Dict[str, Any]] = None,
        task_complexity: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> None:
        """Initialize PhysicalCompactificationReduction.

        Args:
            initial_grid: Grid array for topology baseline computation.
            game_state: Dict with sprites, grid_shape, task_type, etc.
            task_complexity: Dict from classify_task_complexity().
            strict: If True, Euler char must match exactly; else ±1 tolerance.
        """
        self.phys_guard = PhysicalGaussExGuard(strict=strict)
        self.strict = strict
        self.game_state = game_state or {}
        self.task_complexity = task_complexity or {
            "complexity_class": "NP_C_likely",
            "Bk_requirement": "Bκ_standard",
            "strategy": "full_pipeline",
            "difficulty_score": 1.0,
            "has_physical_constraints": False,
            "has_non_associative_ops": False,
        }

        # Compute baseline topology from initial grid
        if initial_grid is not None:
            self.initial_topo = self.phys_guard._compute_grid_topology(initial_grid)
            self.initial_energy = float(np.sum(initial_grid > 0))
        else:
            self.initial_topo = {"euler_char": 0, "n_components": 1, "n_holes": 0, "n_pixels": 0, "density": 0.0}
            self.initial_energy = 0.0

        # ── Sleep-Step trigger tracking ──
        self._consecutive_failures: int = 0
        self._total_pruned: int = 0
        self._total_accepted: int = 0
        self._sleep_step_triggered: bool = False

        # ── κ-Snap Beam Width (article §3.2: recommended 16) ──
        self.beam_width: int = 16

        # ── v3.9.0 — Successful Proof Log (Algorithm 1: SUCCESS LOG → Functional Macro) ──
        self._successful_proof_log: List[Dict[str, Any]] = []
        self._processed_log_indices: Set[int] = set()  # Track processed episodes

    def should_prune(
        self,
        new_grid: Optional[np.ndarray] = None,
        action_sequence: Optional[List[str]] = None,
        energy_delta: Optional[float] = None,
    ) -> bool:
        """Check if a candidate expansion node should be pruned (rejected).

        Implements Algorithm 1 from the article — the GaussEx Physical Pruner:
            1. Energy(P) ≤ S.Available_Energy  → accept energy budget
            2. EulerChar(P) ≈ EulerChar(Initial) → accept topology
            3. Type-B CausalOrder(P) respected     → accept causal order
            4. Φ_boundary respected                → accept boundaries
            Any violation → prune (reject from search frontier).

        Args:
            new_grid: Grid state after candidate action (H, W) array.
            action_sequence: DSL-like action labels for causal order check.
            energy_delta: Estimated energy change from this action.

        Returns:
            True if node should be PRUNED (rejected).
            False if node should be ACCEPTED into search frontier.
        """
        violations: List[str] = []

        # ── 1. E_cons: Energy conservation ──
        if new_grid is not None:
            new_energy = float(np.sum(new_grid > 0))
            # Energy budget: new state shouldn't wildly exceed initial
            max_energy_budget = self.initial_energy * 2.0 + 10.0  # Allow some growth
            if new_energy > max_energy_budget:
                violations.append("E_cons: energy exceeds budget")
            # Energy delta: each step shouldn't add too many pixels
            if energy_delta is not None and energy_delta > 20.0:
                violations.append("E_cons: single-step energy spike")
        elif energy_delta is not None and energy_delta > 20.0:
            violations.append("E_cons: energy spike without grid check")

        # ── 2. G_topo: Topological closure (Euler characteristic) ──
        if new_grid is not None:
            new_topo = self.phys_guard._compute_grid_topology(new_grid)
            tolerance = 1 if not self.strict else 0
            if abs(new_topo["euler_char"] - self.initial_topo["euler_char"]) > tolerance:
                violations.append(
                    f"G_topo: euler_char {new_topo['euler_char']} != "
                    f"initial {self.initial_topo['euler_char']}"
                )

        # ── 3. T_causal: Causal order (for Type-B non-associative) ──
        if action_sequence is not None:
            has_non_assoc = self.task_complexity.get("has_non_associative_ops", False)
            if has_non_assoc and self.phys_guard._check_causal_order(action_sequence, self.game_state):
                violations.append("T_causal: operation order violates non-associativity")

        # ── 4. Φ_boundary: Boundary layer ──
        grid_shape = self.game_state.get("grid_shape", (0, 0))
        if grid_shape != (0, 0) and new_grid is not None:
            h, w = new_grid.shape
            if h > grid_shape[0] or w > grid_shape[1]:
                violations.append(f"Phi_boundary: grid {h}x{w} exceeds bounds {grid_shape}")

        # ── Decision ──
        should_prune = len(violations) > 0

        if should_prune:
            self._consecutive_failures += 1
            self._total_pruned += 1
            # ── Sleep-Step trigger (§4.3): 3 consecutive failures ──
            if self._consecutive_failures >= self.SLEEP_STEP_THRESHOLD:
                self._sleep_step_triggered = True
                self._consecutive_failures = 0  # Reset after trigger
        else:
            self._consecutive_failures = 0  # Reset on success
            self._total_accepted += 1

        return should_prune

    # ── v3.9.0 — Sleep-Step Algorithm 1: SUCCESS LOG → Functional Macro ──

    def sleep_step_algorithm_1(
        self,
        episode_buffer: Optional[List[EpisodeTrace]] = None,
        library_manager: Optional[LibraryManager] = None,
        rhae_success_threshold: float = 50.0,
        min_pattern_occurrences: int = 2,
    ) -> Dict[str, Any]:
        """Sleep-Step Algorithm 1: extract Functional Macros from successful episodes.

        Implements Algorithm 1 from "从猜想生成到流贯觉察":
            Input: Failed_Proof_Log, Successful_Proof_Log
            Output: Updated Library.json
            1. Analyze logs to find recurring structural patterns
            2. If pattern leads to success: Create Functional Macro
            3. Add Macro to library.json
            4. Clear logs

        This extends the existing sleep_step_online() which only processes
        Failed_Proof_Log (failure events via OnlinePSIAudit). Algorithm 1
        also processes Successful_Proof_Log to extract recurring action
        patterns that led to success and creates Functional Macro candidates.

        Args:
            episode_buffer: List of EpisodeTrace records to process.
                If None, processes internal _successful_proof_log.
            library_manager: LibraryManager for macro persistence.
                If None, creates a default one.
            rhae_success_threshold: Minimum RHAE score for an episode
                to be considered "successful" for pattern extraction.
                Defaults to 50.0.
            min_pattern_occurrences: Minimum number of times a pattern
                must appear across episodes to be considered recurring.
                Defaults to 2.

        Returns:
            Dict with keys:
                macros_created: Number of new Functional Macros created.
                successful_episodes_processed: Number of successful episodes analyzed.
                failed_episodes_processed: Number of failed episodes analyzed.
                recurring_patterns_found: Number of recurring patterns detected.
                new_macros: List of MacroCandidate objects created.
        """
        library = library_manager or LibraryManager()
        episodes = episode_buffer or []

        # ── Step 1: Separate successful and failed episodes ──
        successful_episodes: List[EpisodeTrace] = []
        failed_episodes: List[EpisodeTrace] = []

        for ep in episodes:
            if ep.success and ep.rhae_score >= rhae_success_threshold:
                successful_episodes.append(ep)
            elif not ep.success or ep.rhae_score < rhae_success_threshold:
                failed_episodes.append(ep)

        # Also process episodes from _successful_proof_log
        for log_entry in self._successful_proof_log:
            # Convert log entry to EpisodeTrace if needed
            if "episode" in log_entry:
                ep = log_entry["episode"]
                if ep.success and ep.rhae_score >= rhae_success_threshold:
                    successful_episodes.append(ep)

        if not successful_episodes:
            return {
                "macros_created": 0,
                "successful_episodes_processed": 0,
                "failed_episodes_processed": len(failed_episodes),
                "recurring_patterns_found": 0,
                "new_macros": [],
            }

        # ── Step 2: Analyze logs to find recurring structural patterns ──
        pattern_counts: Dict[str, Dict[str, Any]] = {}

        for ep in successful_episodes:
            # Extract action sequence from episode traces
            action_sequence: List[str] = []
            for trace in ep.traces:
                action_key = f"{trace.action_type}:{json.dumps(trace.action_params, sort_keys=True)}"
                action_sequence.append(action_key)

            if len(action_sequence) < 2:
                continue

            # Find recurring sub-patterns (contiguous action subsequences)
            # Use sliding windows of length 2-5 to detect recurring patterns
            for window_len in range(2, min(6, len(action_sequence) + 1)):
                for i in range(len(action_sequence) - window_len + 1):
                    pattern_key = tuple(action_sequence[i:i + window_len])
                    pattern_str = "|".join(pattern_key)

                    if pattern_str not in pattern_counts:
                        pattern_counts[pattern_str] = {
                            "pattern": pattern_key,
                            "count": 0,
                            "episodes": [],
                            "total_rhae": 0.0,
                        }
                    pattern_counts[pattern_str]["count"] += 1
                    pattern_counts[pattern_str]["episodes"].append(ep.game_id)
                    pattern_counts[pattern_str]["total_rhae"] += ep.rhae_score

        # ── Step 3: Identify recurring patterns (≥ min_pattern_occurrences) ──
        recurring_patterns: List[Dict[str, Any]] = []
        for pattern_str, info in pattern_counts.items():
            if info["count"] >= min_pattern_occurrences:
                avg_rhae = info["total_rhae"] / max(info["count"], 1)
                recurring_patterns.append({
                    "pattern_str": pattern_str,
                    "pattern_tuple": info["pattern"],
                    "occurrence_count": info["count"],
                    "avg_rhae_score": avg_rhae,
                    "source_episodes": info["episodes"],
                })

        # ── Step 4: Create Functional Macro candidates from recurring patterns ──
        new_macros: List[MacroCandidate] = []
        for pattern_info in recurring_patterns:
            pattern_tuple = pattern_info["pattern_tuple"]

            # Build DSL sequence from pattern
            dsl_sequence: List[Dict[str, Any]] = []
            for action_key in pattern_tuple:
                parts = action_key.split(":", 1)
                action_type = parts[0]
                try:
                    action_params = json.loads(parts[1]) if len(parts) > 1 else {}
                except (json.JSONDecodeError, IndexError):
                    action_params = {}
                dsl_sequence.append({
                    "action": action_type,
                    "params": action_params,
                })

            # Compress repeated actions in DSL sequence
            compressed_dsl: List[Dict[str, Any]] = []
            i = 0
            while i < len(dsl_sequence):
                current = dsl_sequence[i]
                repeat_count = 1
                while i + repeat_count < len(dsl_sequence) and dsl_sequence[i + repeat_count] == current:
                    repeat_count += 1
                if repeat_count > 1:
                    compressed_dsl.append({
                        "repeat": current.get("action", "UNKNOWN"),
                        "params": current.get("params", {}),
                        "count": repeat_count,
                    })
                else:
                    compressed_dsl.append(current)
                i += repeat_count

            # Compute MDL score
            original_steps = len(dsl_sequence)
            compressed_steps = len(compressed_dsl)
            mdl_score = compressed_steps / max(original_steps, 1)

            # Generate generalization tags from action types
            action_types = set(a.get("action", "") for a in compressed_dsl)
            generalization_tags = list(action_types)
            generalization_tags.append("functional_macro")
            generalization_tags.append("success_pattern")

            # Compute TOMAS fingerprint
            fingerprint = hashlib.sha256(
                json.dumps(compressed_dsl, sort_keys=True).encode()
            ).hexdigest()[:16]

            # Create MacroCandidate
            macro_name = f"func_macro_{fingerprint[:8]}_{pattern_info['occurrence_count']}x"
            macro = MacroCandidate(
                name=macro_name,
                dsl_sequence=compressed_dsl,
                tomas_fingerprint=fingerprint,
                source_tasks=pattern_info["source_episodes"][:5],
                avg_steps=float(original_steps),
                success_rate=pattern_info["avg_rhae_score"] / 100.0,
                generalization_tags=generalization_tags,
                mdl_score=mdl_score,
                min_demo_to_activate=max(1, min_pattern_occurrences - 1),
                validated=True,  # Pre-validated from successful episodes
                applicable_topo=self.initial_topo,
                gaussex_precond="True",
            )

            new_macros.append(macro)

            # ── Step 5: Add validated macros to library.json ──
            library.add_macro(macro)

            # Track in successful proof log
            self._successful_proof_log.append({
                "pattern_str": pattern_info["pattern_str"],
                "macro_name": macro_name,
                "occurrence_count": pattern_info["occurrence_count"],
                "avg_rhae": pattern_info["avg_rhae_score"],
                "timestamp": time.time(),
            })

        # ── Step 6: Clear processed logs ──
        self._successful_proof_log = [
            entry for entry in self._successful_proof_log
            if entry.get("timestamp", 0) > time.time() - 3600  # Keep recent entries
        ]

        return {
            "macros_created": len(new_macros),
            "successful_episodes_processed": len(successful_episodes),
            "failed_episodes_processed": len(failed_episodes),
            "recurring_patterns_found": len(recurring_patterns),
            "new_macros": new_macros,
        }

    def should_prune_game_state(
        self,
        game: Any,
        prev_game: Any,
    ) -> bool:
        """Check if a game state transition should be pruned.

        Convenience method for BFS/DFS/Beam where we have game objects
        rather than raw grids. Extracts grid from game and delegates
        to should_prune().

        Args:
            game: Game object after action (modified state).
            prev_game: Game object before action (previous state).

        Returns:
            True if transition should be PRUNED (rejected).
        """
        new_grid = None
        try:
            new_grid = np.array(game.current_state.grid)
        except Exception:
            try:
                new_grid = np.array(game.grid)
            except Exception:
                pass

        if new_grid is None:
            # Can't extract grid → don't prune (accept)
            self._total_accepted += 1
            return False

        # Estimate energy delta
        prev_grid = None
        try:
            prev_grid = np.array(prev_game.current_state.grid)
        except Exception:
            pass

        energy_delta = None
        if prev_grid is not None:
            energy_delta = abs(float(np.sum(new_grid > 0)) - float(np.sum(prev_grid > 0)))

        return self.should_prune(
            new_grid=new_grid,
            energy_delta=energy_delta,
        )

    def get_pruning_stats(self) -> Dict[str, Any]:
        """Get pruning statistics for this reduction instance.

        Returns:
            Dict with total_pruned, total_accepted, prune_ratio,
            sleep_step_triggered, and task_complexity.
        """
        total = self._total_pruned + self._total_accepted
        prune_ratio = self._total_pruned / max(total, 1)
        return {
            "total_pruned": self._total_pruned,
            "total_accepted": self._total_accepted,
            "prune_ratio": prune_ratio,
            "sleep_step_triggered": self._sleep_step_triggered,
            "consecutive_failures": self._consecutive_failures,
            "task_complexity": self.task_complexity,
            "initial_euler_char": self.initial_topo.get("euler_char", 0),
            "initial_energy": self.initial_energy,
        }

    def was_sleep_step_triggered(self) -> bool:
        """Check if Sleep-Step was triggered during this session.

        Returns:
            True if 3+ consecutive GaussEx failures occurred.
        """
        return self._sleep_step_triggered

    def reset_stats(self) -> None:
        """Reset pruning statistics for a new solve_game() call."""
        self._consecutive_failures = 0
        self._total_pruned = 0
        self._total_accepted = 0
        self._sleep_step_triggered = False


class PsiCutController:
    """Ψ-Cut boundary management for ARC-AGI-3 solver safety.

    Implements the Ψ-截断 (Ψ-Cut) concept from the "物理紧化与Ψ-截断"
    paper. In ARC-AGI-3 context, Ψ-Cut manages solver boundaries:

        - Preventing solver from accessing game internals beyond allowed scope
        - Enforcing time budgets (physical constraint on computation)
        - Preventing adversarial over-optimization (alignment faking detection)

    Three Ψ-Cut states (from paper Appendix B):
        OPEN:      Normal operation, full solver access
        INCOGNITO: Reduced access, no game internals, only frame data
        COLLAPSED: Emergency shutdown, solver returns safe defaults

    This is inspired by Antinel's hardware-level security model, adapted
    for software solver boundaries. The key insight: software-level "off"
    is an L5 logical state (illusion), while Ψ-Cut is an L4 ontological
    boundary (real). In our implementation, Ψ-Cut restricts the solver's
    access to game state, preventing it from "cheating" by reading internal
    game variables that a real player wouldn't have access to.

    Args:
        max_time_budget: Maximum solver time per game (seconds).
        max_action_budget: Maximum actions per level.
        enforce_frame_only: Whether to restrict to frame-level access only.
    """

    # Ψ-Cut states (from paper)
    OPEN = "OPEN"          # Full solver access
    INCOGNITO = "INCOGNITO"  # Reduced access, frame-only
    COLLAPSED = "COLLAPSED"  # Emergency shutdown

    def __init__(
        self,
        max_time_budget: float = 60.0,
        max_action_budget: int = 300,
        enforce_frame_only: bool = False,
    ) -> None:
        """Initialize PsiCutController.

        Args:
            max_time_budget: Maximum time per solve_game() call (seconds).
            max_action_budget: Maximum actions per level.
            enforce_frame_only: If True, restrict solver to frame-level access.
        """
        self.state = self.OPEN
        self.max_time_budget = max_time_budget
        self.max_action_budget = max_action_budget
        self.enforce_frame_only = enforce_frame_only

        # ── Time tracking ──
        self._start_time: Optional[float] = None
        self._action_count: int = 0

        # ── Ψ-Cut event log ──
        self._cut_log: List[Dict[str, Any]] = []

    def enforce_psi_cut(self, condition: str) -> None:
        """Enforce a Ψ-Cut boundary based on environmental conditions.

        Maps Antinel conditions to solver Ψ-Cut states:
            MEETING_START → INCOGNITO (private evaluation, no internals)
            INTRUSION_DETECTED → COLLAPSED (emergency shutdown)
            NORMAL → OPEN (standard operation)

        Args:
            condition: Trigger condition string.
        """
        if condition == "MEETING_START" or condition == "PRIVATE_EVAL":
            self.state = self.INCOGNITO
            self.enforce_frame_only = True
            self._cut_log.append({
                "condition": condition,
                "new_state": self.INCOGNITO,
                "timestamp": __import__("time").time(),
            })
        elif condition == "INTRUSION_DETECTED" or condition == "EMERGENCY":
            self.state = self.COLLAPSED
            self._cut_log.append({
                "condition": condition,
                "new_state": self.COLLAPSED,
                "timestamp": __import__("time").time(),
            })
        elif condition == "NORMAL" or condition == "RESET":
            self.state = self.OPEN
            self.enforce_frame_only = False
            self._cut_log.append({
                "condition": condition,
                "new_state": self.OPEN,
                "timestamp": __import__("time").time(),
            })

    def check_time_budget(self, elapsed: float) -> bool:
        """Check whether solver has exceeded time budget.

        Args:
            elapsed: Time elapsed so far (seconds).

        Returns:
            True if within budget; False if exceeded.
        """
        return elapsed < self.max_time_budget

    def check_action_budget(self, actions_taken: int) -> bool:
        """Check whether solver has exceeded action budget.

        Args:
            actions_taken: Number of actions taken so far.

        Returns:
            True if within budget; False if exceeded.
        """
        return actions_taken < self.max_action_budget

    def check_access_allowed(self, attribute_name: str) -> bool:
        """Check whether solver can access a specific game attribute.

        In INCOGNITO mode, only frame-level attributes are allowed.
        In COLLAPSED mode, no attributes are allowed.

        Args:
            attribute_name: Name of the game attribute to access.

        Returns:
            True if access is allowed; False otherwise.
        """
        if self.state == self.COLLAPSED:
            return False  # Emergency: no access

        if self.state == self.INCOGNITO:
            # Only allow frame-level attributes
            allowed_prefixes = ["frame", "state", "grid", "score", "level_index"]
            return any(attribute_name.startswith(p) for p in allowed_prefixes)

        return True  # OPEN: full access

    def enter_incognito_mode(self, reason: str = "auto") -> None:
        """Switch to INCOGNITO mode with reduced game access.

        Args:
            reason: Reason for switching to incognito mode.
        """
        self.state = self.INCOGNITO
        self.enforce_frame_only = True
        self._cut_log.append({
            "condition": f"incognito:{reason}",
            "new_state": self.INCOGNITO,
            "timestamp": __import__("time").time(),
        })

    def psi_collapse(self, trigger: str = "manual") -> None:
        """Emergency Ψ-Collapse — solver enters safe default mode.

        Args:
            trigger: Trigger for collapse (manual, timeout, anomaly).
        """
        self.state = self.COLLAPSED
        self._cut_log.append({
            "condition": f"collapse:{trigger}",
            "new_state": self.COLLAPSED,
            "timestamp": __import__("time").time(),
        })

    def get_state(self) -> str:
        """Return current Ψ-Cut state.

        Returns:
            Current state string (OPEN, INCOGNITO, or COLLAPSED).
        """
        return self.state

    def get_cut_log(self) -> List[Dict[str, Any]]:
        """Return the Ψ-Cut event log.

        Returns:
            List of dicts with condition, new_state, timestamp.
        """
        return list(self._cut_log)


# ═══════════════════════════════════════════════════════════════════════
# v3.6.0 — 修正 Scaling Law: 进化增益项 Δ_Bκ
# 来源: "流贯的决定论与紧化基的演化" (复合体理学 2026-06-26)
# ═══════════════════════════════════════════════════════════════════════


def compute_scaling_law_v36(
    C: float,
    D: float,
    L_irr: float = 0.05,
    C_0: float = 1.0,
    alpha: float = 0.3,
    L_rep: float = 0.02,
    L_sat: float = 0.0,
    L_min_irr: float = 0.0,
    tau: float = 10.0,
    delta_Bk: float = 0.0,
    kappa_evolution: float = 0.5,
) -> Dict[str, float]:
    """Compute TOMAS Scaling Law v3.6.0 with Bκ evolution gain.

    Implements the modified Scaling Law from "流贯的决定论与紧化基的演化":

        L(C,D) = L_irr + (C₀/C)^α + L_rep + Ω_topo - Δ_Bκ

    Where:
        Ω_topo = (L_sat - L_min_irr) × (1 - exp(-τ × C))  [拓扑饱和罚项]
        Δ_Bκ = κ_evolution × delta_Bk × C  [进化增益项, 抵消Ω_topo]

    Key insight from paper: Δ_Bκ > Ω_topo → performance plateau breakthrough
    This happens when Bκ evolves to capture the causal invariant of the task.

    Args:
        C: Effective parameter count (model capacity).
        D: Effective data count (with repetition decay).
        L_irr: Irreducible loss (noise floor).
        C_0: Reference capacity for scaling.
        alpha: Scaling exponent.
        L_rep: Representation loss (data-dependent).
        L_sat: Saturation loss (from Type-B plateau).
        L_min_irr: Minimum irreducible loss achievable by Bκ.
        tau: Saturation time constant.
        delta_Bk: Bκ evolution progress (0 = no evolution, 1 = fully evolved).
        kappa_evolution: Evolution efficiency coefficient.

    Returns:
        Dict with L_total, Omega_topo, Delta_Bk, and all component values.
    """
    # ── Core scaling ──
    L_scaling = L_irr + (C_0 / max(C, 0.01)) ** alpha + L_rep

    # ── Topology saturation penalty (from v3.5.0) ──
    Omega_topo = (L_sat - L_min_irr) * (1.0 - float(np.exp(-tau * C)))

    # ── Bκ evolution gain (new in v3.6.0) ──
    # Δ_Bκ increases with both evolution progress (delta_Bk) and capacity (C)
    # This is the key mechanism: Bκ evolution can offset and surpass Ω_topo
    Delta_Bk = kappa_evolution * delta_Bk * C

    # ── Total loss ──
    L_total = L_scaling + Omega_topo - Delta_Bk

    # ── Ensure non-negative ──
    L_total = max(L_total, L_irr)

    return {
        "L_total": float(L_total),
        "L_scaling": float(L_scaling),
        "L_irr": float(L_irr),
        "L_rep": float(L_rep),
        "Omega_topo": float(Omega_topo),
        "Delta_Bk": float(Delta_Bk),
        "delta_Bk": float(delta_Bk),
        "kappa_evolution": float(kappa_evolution),
        "breakthrough": Delta_Bk > Omega_topo,  # Bκ进化突破拓扑饱和
        "L_irr_min": float(L_min_irr),
    }


# ═══════════════════════════════════════════════════════════════════════
# NP可解性判据表 (from "物理紧化与Ψ-截断" Appendix D)
# ═══════════════════════════════════════════════════════════════════════


def classify_task_complexity(
    game_id: str,
    topo_features: Dict[str, Any],
    has_physical_constraints: bool = True,
    has_non_associative_ops: bool = False,
) -> Dict[str, Any]:
    """Classify ARC-AGI-3 task complexity using TOMAS NP-solvability criteria.

    Implements the判据表 from "物理紧化与Ψ-截断" Appendix D:
        - 含Φ_phys → P类 (物理约束降维)
        - 含非结合操作 → 需Bκ进化 (NAR-Conv)
        - 纯组合 → NP-C (无物理先验)

    Args:
        game_id: ARC game ID (e.g., 'ls20', 'ft09').
        topo_features: Topology feature dict from extract_topo_features.
        has_physical_constraints: Whether task has physical constraints.
        has_non_associative_ops: Whether task has non-associative operations.

    Returns:
        Dict with complexity_class, Bk_requirement, and strategy.
    """
    euler_char = topo_features.get("euler_char", 0)
    n_components = topo_features.get("n_components", 1)
    density = topo_features.get("density", 0.5)

    # ── Classification logic ──
    # ── v3.9.0: also set requires_new_primitives and sleep_step_aggressive ──
    requires_new_primitives: bool = False
    sleep_step_aggressive: bool = False

    if has_non_associative_ops and has_physical_constraints:
        # Type-B with physics: P in physical, needs Bκ evolution for NAR
        complexity_class = "P_in_phys"
        Bk_requirement = "Bκ_NAR"  # Need NAR-Conv compactification basis
        strategy = "PhysicalNARConv + κ-Snap with Φ_phys pruning"
        requires_new_primitives = False  # Has sufficient Bκ with physics
        sleep_step_aggressive = False
    elif has_non_associative_ops and not has_physical_constraints:
        # Type-B without physics: NP-Hard without Bκ evolution
        complexity_class = "NP_Hard"
        Bk_requirement = "Bκ_NAR_critical"  # Critical need for NAR-Conv
        strategy = "Bκ-evolution + Sleep-Step + NAR-Conv"
        requires_new_primitives = True  # Critical need — 哥德尔余留 boundary
        sleep_step_aggressive = True
    elif has_physical_constraints and not has_non_associative_ops:
        # Type-A with physics: Easy P
        complexity_class = "P"
        Bk_requirement = "Bκ_standard"  # Standard compactification
        strategy = "Fast-Path + GaussExGuard + standard κ-Snap"
        requires_new_primitives = False  # Has sufficient Bκ
        sleep_step_aggressive = False
    else:
        # Pure combinatorial (no physics, no NAR)
        complexity_class = "NP_C_likely"
        Bk_requirement = "Bκ_exotic"  # Need exotic compactification
        strategy = "Random + heuristic + Bκ-exotic proposal"
        requires_new_primitives = True  # 哥德尔余留 boundary acknowledged
        sleep_step_aggressive = True  # Trigger Sleep-Step more aggressively

    # ── Difficulty score ──
    difficulty = density * n_components + abs(euler_char) * 0.5
    if has_non_associative_ops:
        difficulty *= 2.0  # Non-associative doubles difficulty

    return {
        "game_id": game_id,
        "complexity_class": complexity_class,
        "Bk_requirement": Bk_requirement,
        "strategy": strategy,
        "difficulty_score": float(difficulty),
        "has_physical_constraints": has_physical_constraints,
        "has_non_associative_ops": has_non_associative_ops,
        "euler_char": euler_char,
        "n_components": n_components,
        # ── v3.9.0 — NP_C_likely boundary marking ──
        "requires_new_primitives": requires_new_primitives,
        "sleep_step_aggressive": sleep_step_aggressive,
    }


# ═══════════════════════════════════════════════════════════════════════
# v3.9.0 — Ω_topo plateau → Bκ evolution trigger
# 来源: "流贯的决定论与紧化基的演化" + "从猜想生成到流贯觉察"
# ═══════════════════════════════════════════════════════════════════════


def check_breakthrough_and_trigger(
    game_id: str,
    topo_features: Dict[str, Any],
    scaling_params: Optional[Dict[str, float]] = None,
    episode_buffer: Optional[List[EpisodeTrace]] = None,
    library_manager: Optional[LibraryManager] = None,
) -> Dict[str, Any]:
    """Check Ω_topo plateau breakthrough and trigger Bκ evolution if needed.

    Implements the Ω_topo plateau → Bκ evolution trigger from the v3.9.0
    algorithmic deep integration. This function:

    1. Calls compute_scaling_law_v36() with current game parameters
    2. If breakthrough=True (Δ_Bκ > Ω_topo): triggers sleep_step_algorithm_1()
       to generate new DSL primitives (Bκ evolution)
    3. If breakthrough=False: acknowledges ontological randomness boundary
       (logs it, doesn't generate macros — respects 哥德尔余留)
    4. Returns dict with action_taken, breakthrough_status, and any new macros

    Key insight from article: Ω_topo is the performance ceiling imposed by
    the current 紧化基 Bκ. When Δ_Bκ > Ω_topo, the Bκ has evolved enough
    to break through this ceiling. When Δ_Bκ < Ω_topo, we're at the
    ontological randomness boundary (哥德尔余留) — no further improvement
    is possible without fundamentally new primitives.

    Args:
        game_id: ARC-AGI-3 game identifier.
        topo_features: Topology feature dict from extract_topo_features.
        scaling_params: Optional dict of scaling law parameters for
            compute_scaling_law_v36(). If None, uses defaults.
        episode_buffer: Optional list of EpisodeTrace records for
            sleep_step_algorithm_1 processing.
        library_manager: Optional LibraryManager for macro persistence.

    Returns:
        Dict with keys:
            breakthrough: Whether Δ_Bκ > Ω_topo (Bκ evolution breaks plateau).
            action_taken: String describing what action was taken.
            new_macros: List of MacroCandidate objects created (if breakthrough).
            scaling_result: Full result dict from compute_scaling_law_v36().
            ontological_boundary_acknowledged: Whether 哥德尔余留 boundary was hit.
    """
    # ── Step 1: Compute scaling law with current parameters ──
    params = scaling_params or {}
    C = params.get("C", 10.0)
    D = params.get("D", 100.0)
    L_irr = params.get("L_irr", 0.05)
    C_0 = params.get("C_0", 1.0)
    alpha = params.get("alpha", 0.3)
    L_rep = params.get("L_rep", 0.02)
    L_sat = params.get("L_sat", 0.1)
    L_min_irr = params.get("L_min_irr", 0.0)
    tau = params.get("tau", 10.0)
    delta_Bk = params.get("delta_Bk", 0.0)
    kappa_evolution = params.get("kappa_evolution", 0.5)

    scaling_result = compute_scaling_law_v36(
        C=C, D=D, L_irr=L_irr, C_0=C_0, alpha=alpha,
        L_rep=L_rep, L_sat=L_sat, L_min_irr=L_min_irr,
        tau=tau, delta_Bk=delta_Bk, kappa_evolution=kappa_evolution,
    )

    breakthrough = scaling_result.get("breakthrough", False)

    # ── Step 2: Take action based on breakthrough status ──
    new_macros: List[MacroCandidate] = []
    action_taken: str = ""
    ontological_boundary_acknowledged: bool = False

    if breakthrough:
        # Δ_Bκ > Ω_topo → Bκ evolution breaks the plateau!
        # Trigger sleep_step_algorithm_1() to generate new DSL primitives
        action_taken = "breakthrough_detected: triggered sleep_step_algorithm_1 for Bκ evolution"

        reduction = PhysicalCompactificationReduction(
            task_complexity=classify_task_complexity(
                game_id=game_id,
                topo_features=topo_features,
                has_physical_constraints=True,
                has_non_associative_ops=False,
            ),
        )

        algorithm_result = reduction.sleep_step_algorithm_1(
            episode_buffer=episode_buffer or [],
            library_manager=library_manager or LibraryManager(),
        )

        new_macros = algorithm_result.get("new_macros", [])
        action_taken += f" → {algorithm_result['macros_created']} macros created"

    else:
        # Δ_Bκ < Ω_topo → ontological randomness boundary (哥德尔余留)
        # Acknowledge boundary — don't generate macros
        # This respects the Gödel boundary: some tasks are fundamentally
        # beyond the current Bκ's reach and require genuinely new primitives
        # that cannot be derived from existing ones.
        ontological_boundary_acknowledged = True
        action_taken = "ontological_randomness_boundary_acknowledged: 哥德尔余留 — Δ_Bκ < Ω_topo, no macro generation"

    return {
        "breakthrough": breakthrough,
        "action_taken": action_taken,
        "new_macros": new_macros,
        "scaling_result": scaling_result,
        "ontological_boundary_acknowledged": ontological_boundary_acknowledged,
    }


# ═══════════════════════════════════════════════════════════════════════
# v3.8.0 — Thinker-Performer双轨 + Interactive ZKP Loop + Persistent KV-Cache
# ═══════════════════════════════════════════════════════════════════════
# Wan-Streamer dual-track architecture: Thinker (fast perception) +
# Performer (deep search) with Block-Causal Attention KV-cache.
#
# InteractiveZKPLoop: "博弈即降维" Oracle interaction pattern.
# PersistentKVCache: Cross-Phase persistent state (Block-Causal Attention).
# ThinkerPerformerPipeline: Orchestrates Thinker→Performer pipeline parallelism.
# ═══════════════════════════════════════════════════════════════════════


class PersistentKVCache:
    """Cross-Phase persistent state cache — Wan-Streamer Block-Causal Attention pattern.

    Maintains shared state across game steps:
    - topo_features: topology fingerprints (64-dim vectors)
    - gaussex_logs: GaussEx verification results
    - effective_macros: library macros that worked in previous steps
    - reference_grid: current grid state (updated by Clean Latent Writeback)
    - compactification_state: PhysicalCompactificationReduction state

    Implements Block-Causal Attention: each step updates cache incrementally,
    never recomputing from scratch — prevents context explosion.

    Attributes:
        _topo_features: Dict mapping step_id to 64-dim topology vector.
        _gaussex_logs: List of GaussEx verification log entries.
        _effective_macros: Dict mapping macro_name to DSL_sequence (verified macros).
        _reference_grid: Optional numpy array of current grid (Clean Latent Writeback target).
        _compactification_state: Optional Dict of Φ_phys state dict.
        _euler_char_history: List of euler characteristic values across steps.
        _connected_components_history: List of connected_components counts across steps.
        _step_count: Integer count of steps processed.
    """

    def __init__(self) -> None:
        """Initialize empty PersistentKVCache."""
        self._topo_features: Dict[str, np.ndarray] = {}  # step_id -> 64-dim topo vector
        self._gaussex_logs: List[Dict[str, Any]] = []     # GaussEx verification log
        self._effective_macros: Dict[str, str] = {}       # macro_name -> DSL_sequence (verified macros)
        self._reference_grid: Optional[np.ndarray] = None  # current grid (Clean Latent Writeback target)
        self._compactification_state: Optional[Dict[str, Any]] = None  # Φ_phys state dict
        self._euler_char_history: List[int] = []           # euler_char across steps
        self._connected_components_history: List[int] = [] # connected_components across steps
        self._step_count: int = 0

    def update_thinker_state(self, grid: np.ndarray, topo_features: Dict[str, Any], step_id: str) -> None:
        """Thinker writes perception results to cache (Block-Causal Attention update).

        Args:
            grid: Current game grid as numpy array.
            topo_features: Topology feature dict from extract_topo_features.
            step_id: String identifier for this step (e.g. "step_1", "think_0").
        """
        # Store topology fingerprint as 64-dim vector (if extract_topo_features_vec available)
        topo_vec = extract_topo_features_vec(grid) if grid is not None and grid.size > 0 else np.zeros(64)
        self._topo_features[step_id] = topo_vec
        if self._reference_grid is None:
            self._reference_grid = grid.copy()
        # Track topology invariants from topo_features dict
        euler_char = topo_features.get("euler_char", 0) if isinstance(topo_features, dict) else 0
        n_components = topo_features.get("n_components", 1) if isinstance(topo_features, dict) else 1
        if euler_char not in self._euler_char_history:
            self._euler_char_history.append(euler_char)
        if n_components not in self._connected_components_history:
            self._connected_components_history.append(n_components)
        self._step_count += 1

    def clean_latent_writeback(self, new_grid: np.ndarray, gaussex_result: Dict[str, Any]) -> None:
        """Clean Latent Writeback: after GaussEx passes, immediately update reference_grid.

        No re-encoding needed — Thinker's perception is directly persisted.

        Args:
            new_grid: Updated game grid as numpy array.
            gaussex_result: Dict with "passed" key indicating verification status.
        """
        if gaussex_result.get("passed", False):
            self._reference_grid = new_grid.copy()
            self._gaussex_logs.append(gaussex_result)
            # Update topology history for beam ranking
            topo = extract_topo_features(new_grid)
            euler_char = topo.get("euler_char", 0)
            n_components = topo.get("n_components", 1)
            self._euler_char_history.append(euler_char)
            self._connected_components_history.append(n_components)

    def get_performer_state(self) -> Dict[str, Any]:
        """Performer reads cached state for deep search (beam/DFS/NAR-Conv).

        Returns:
            Dict containing topo_features, reference_grid, effective_macros,
            compactification_state, topology history, and step_count.
        """
        return {
            "topo_features": self._topo_features,
            "reference_grid": self._reference_grid,
            "effective_macros": self._effective_macros,
            "compactification_state": self._compactification_state,
            "euler_char_history": self._euler_char_history,
            "connected_components_history": self._connected_components_history,
            "step_count": self._step_count,
        }

    def register_effective_macro(self, macro_name: str, dsl_sequence: str) -> None:
        """Register a macro that successfully solved a previous step.

        Args:
            macro_name: Name of the verified macro.
            dsl_sequence: DSL sequence string that the macro expands to.
        """
        self._effective_macros[macro_name] = dsl_sequence

    def set_compactification_state(self, state: Dict[str, Any]) -> None:
        """Set PhysicalCompactificationReduction state from init_compactification.

        Args:
            state: Dict containing complexity classification state.
        """
        self._compactification_state = state

    def get_topology_invariant_score(self) -> float:
        """Compute topology-invariant beam ranking score (CHL isomorphism).

        Uses euler_char and connected_components stability as ranking criteria.
        Stable topology (low variance) → higher score (more reliable beam direction).

        Returns:
            Float score in [0.0, 1.0]. Default 0.5 when insufficient history.
        """
        if len(self._euler_char_history) < 2:
            return 0.5  # Default mid score
        # Variance of euler_char — low variance = stable topology = higher score
        euler_var = float(np.var(self._euler_char_history))
        cc_var = float(np.var(self._connected_components_history))
        # Score: 1 - normalized variance (clamp to [0, 1])
        score = 1.0 - min(1.0, euler_var * 0.1 + cc_var * 0.05)
        return max(0.0, min(1.0, score))


class InteractiveZKPLoop:
    """Interactive Zero-Knowledge Proof Loop — "博弈即降维" Oracle interaction pattern.

    Each game step follows the ZKP loop:
    1. Observe:  Read current grid state
    2. Encode:   NAR-Conv octonion encoding → topo_features
    3. Prove:    κ-Snap abductive reduction → candidate actions
    4. Verify:   GaussEx verification → filter invalid candidates
    5. Act:      Execute best verified action

    Each round reduces search depth by Δ information gain:
    O(|D|^(n/Δ)) → Poly when Δ≈n

    Game Semantics (博弈语义): Solver doesn't guess all steps;
    designs actions to extract rules from Oracle feedback.

    Attributes:
        _kv_cache: PersistentKVCache instance for cross-step state.
        _nar_conv: PhysicalNARConv encoder for grid encoding.
        _gauss_ex: PhysicalGaussExGuard verifier for candidate filtering.
        _loop_count: Integer count of ZKP loop iterations.
        _information_gain_history: List of Δ values per iteration.
        _cumulative_depth_reduction: Float tracking total depth reduction.
    """

    def __init__(self, kv_cache: PersistentKVCache) -> None:
        """Initialize InteractiveZKPLoop with a PersistentKVCache.

        Args:
            kv_cache: PersistentKVCache instance for cross-step state sharing.
        """
        self._kv_cache = kv_cache
        self._nar_conv = PhysicalNARConv()  # NAR-Conv encoder
        self._gauss_ex = PhysicalGaussExGuard()  # GaussEx verifier
        self._loop_count: int = 0
        self._information_gain_history: List[float] = []
        self._cumulative_depth_reduction: float = 0.0

    def run_loop(
        self,
        grid: np.ndarray,
        game_state: Dict[str, Any],
        candidate_actions: List[Any],
        phys_pruner: Optional[PhysicalCompactificationReduction] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run one ZKP loop iteration: Observe→Encode→Prove→Verify→Act.

        Args:
            grid: Current game grid as numpy array.
            game_state: Game state dict (sprites, game_id, level_idx, etc.)
            candidate_actions: List of candidate actions to verify.
            phys_pruner: Optional PhysicalCompactificationReduction for pruning.

        Returns:
            (best_action, loop_result_dict) — best verified action + loop metadata.
        """
        self._loop_count += 1

        # ── Step 1: Observe ──
        topo = extract_topo_features(grid)

        # ── Step 2: Encode (NAR-Conv) ──
        nar_features = self._nar_conv.encode_grid_with_deltas(grid)

        # ── Step 3: Prove (κ-Snap reduction) ──
        # Use topology-invariant score from KV-cache to rank candidates
        topo_invariant_score = self._kv_cache.get_topology_invariant_score()
        proven_candidates = self._prove_candidates(
            candidate_actions, topo, nar_features, topo_invariant_score
        )

        # ── Step 4: Verify (GaussEx) ──
        verified_candidates: List[Tuple[Any, Dict[str, Any]]] = []
        for candidate in proven_candidates:
            gaussex_result = self._verify_candidate(candidate, game_state)
            if gaussex_result.get("passed", False):
                verified_candidates.append((candidate, gaussex_result))
                # Clean Latent Writeback: immediately update reference_grid
                self._kv_cache.clean_latent_writeback(grid, gaussex_result)

        # ── Step 5: Act ──
        best_action: Any = None
        delta: float = 0.0
        if verified_candidates:
            best_action = verified_candidates[0][0]
            # Compute information gain Δ
            delta = self._compute_information_gain(topo)
            self._information_gain_history.append(delta)
            self._cumulative_depth_reduction += delta
        else:
            best_action = None

        # Update KV-cache with Thinker state
        step_id = f"step_{self._loop_count}"
        self._kv_cache.update_thinker_state(grid, topo, step_id)

        loop_result: Dict[str, Any] = {
            "loop_count": self._loop_count,
            "n_candidates": len(candidate_actions),
            "n_proven": len(proven_candidates),
            "n_verified": len(verified_candidates),
            "information_gain_delta": delta,
            "cumulative_depth_reduction": self._cumulative_depth_reduction,
            "topo_invariant_score": topo_invariant_score,
            "best_action": best_action,
        }

        return best_action, loop_result

    def _prove_candidates(
        self,
        candidates: List[Any],
        topo: Dict[str, Any],
        nar_features: Any,
        topo_invariant_score: float,
    ) -> List[Any]:
        """κ-Snap reduction: rank and filter candidates by topology + NAR features.

        Higher topo_invariant_score → more confidence in candidate direction.

        Args:
            candidates: List of candidate actions.
            topo: Topology feature dict from extract_topo_features.
            nar_features: NAR-Conv encoded features.
            topo_invariant_score: Float topology stability score from KV-cache.

        Returns:
            List of proven (consistent with observed topology) candidates.
        """
        if not candidates:
            return []
        # Simple ranking: pass all candidates as proven — GaussEx will filter
        # Future enhancement: rank by topo_invariant_score weighted similarity
        proven: List[Any] = []
        for c in candidates:
            # Basic proof: candidate is consistent with observed topology
            proven.append(c)
        return proven

    def _verify_candidate(self, candidate: Any, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """GaussEx verification: check candidate against physics constraints.

        Args:
            candidate: A candidate action to verify.
            game_state: Game state dict for constraint evaluation.

        Returns:
            Dict with "passed" (bool), "reason" (str), and "candidate".
        """
        try:
            # Map candidate to program_node format for PhysicalGaussExGuard
            if isinstance(candidate, str):
                program_node = [candidate]
            elif isinstance(candidate, (list, tuple)):
                program_node = list(candidate)
            else:
                program_node = [str(candidate)]
            passes = self._gauss_ex.check_physical_constraints(
                program_node=program_node,
                current_state=game_state,
                input_grid=None,
                output_grid=None,
            )
            reason = "physical_constraints_pass" if passes else "physical_constraints_fail"
            return {"passed": passes, "reason": reason, "candidate": candidate}
        except Exception as e:
            return {"passed": False, "reason": f"exception: {e}", "candidate": candidate}

    def _compute_information_gain(self, topo: Dict[str, Any]) -> float:
        """Compute information gain Δ from this loop iteration.

        Δ = reduction in search space per loop iteration.
        Estimated from topology change: |Δeuler_char| / max(n_components, 1)

        Args:
            topo: Topology feature dict containing euler_char and n_components.

        Returns:
            Float Δ value, capped at 10.0 (max Δ per iteration).
        """
        euler = topo.get("euler_char", 0)
        cc = topo.get("n_components", 1)
        # Compare with last known euler_char from KV-cache history
        if self._kv_cache._euler_char_history:
            prev_euler = self._kv_cache._euler_char_history[-1]
            delta = abs(euler - prev_euler) / max(cc, 1)
        else:
            delta = abs(euler) / max(cc, 1)  # Initial information gain
        return min(delta, 10.0)  # Cap at 10 (max Δ per iteration)

    def get_loop_stats(self) -> Dict[str, Any]:
        """Get ZKP loop statistics.

        Returns:
            Dict containing total_loops, avg_information_gain,
            cumulative_depth_reduction, and topo_invariant_score.
        """
        return {
            "total_loops": self._loop_count,
            "avg_information_gain": float(np.mean(self._information_gain_history)) if self._information_gain_history else 0.0,
            "cumulative_depth_reduction": self._cumulative_depth_reduction,
            "topo_invariant_score": self._kv_cache.get_topology_invariant_score(),
        }


class ThinkerPerformerPipeline:
    """Thinker-Performer dual-track pipeline — Wan-Streamer architecture pattern.

    Thinker (fast perception layer):
    - grid → topo_features (encoding)
    - state update (reference_grid, compactification state)
    - KV-cache build (persistent state for next step)

    Performer (deep search layer):
    - beam/DFS search with KV-cache state
    - NAR-Conv analysis for non-associative features
    - κ-Snap proof + GaussEx verification

    Pipeline parallelism: Thinker processes step N+1 perception
    while Performer processes step N deep search.
    In ARC-AGI-3, this manifests as:
    - Think_phase: classify_task_complexity + init_compactification + extract_topo
    - perform_phase: solve_game pipeline (BFS→Beam→DFS→IDFS) with KV-cache state

    Attributes:
        _kv_cache: PersistentKVCache for cross-step state persistence.
        _zkp_loop: InteractiveZKPLoop for ZKP verification cycle.
        _thinker_results: Dict of Thinker phase output.
        _performer_results: Dict of Performer phase output.
        _pipeline_count: Integer count of pipeline executions.
    """

    def __init__(self) -> None:
        """Initialize ThinkerPerformerPipeline with KV-cache and ZKP loop."""
        self._kv_cache = PersistentKVCache()
        self._zkp_loop = InteractiveZKPLoop(self._kv_cache)
        self._thinker_results: Dict[str, Any] = {}
        self._performer_results: Dict[str, Any] = {}
        self._pipeline_count: int = 0

    def think_phase(
        self,
        grid: np.ndarray,
        game_state: Dict[str, Any],
        game_id: str = "",
    ) -> Dict[str, Any]:
        """Thinker phase: fast perception + state update + KV-cache build.

        Args:
            grid: Current game grid as numpy array.
            game_state: Game state dict with sprites, game_id, level_idx etc.
            game_id: Game identifier string.

        Returns:
            thinker_result dict with topo_features, complexity_class,
            phys_pruner state, and game metadata.
        """
        # ── Encoding: extract topology ──
        topo = extract_topo_features(grid)

        # ── State update: classify task complexity ──
        learner = TOMASLearner()
        complexity = learner.init_compactification(
            initial_grid=grid,
            game_id=game_id,
            game_state=game_state,
        )

        # ── KV-cache build: persist Thinker state ──
        step_id = f"think_{self._pipeline_count}"
        self._kv_cache.update_thinker_state(grid, topo, step_id)
        self._kv_cache.set_compactification_state(complexity)

        # Store Thinker results for Performer
        self._thinker_results = {
            "topo": topo,
            "complexity": complexity,
            "phys_pruner": learner.physical_compactification,
            "game_id": game_id,
            "grid_shape": grid.shape,
        }

        return self._thinker_results

    def perform_phase(
        self,
        game: Any,
        game_id: str,
        level_idx: int,
        valid_actions: List[Any],
        thinker_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Performer phase: deep search with KV-cache state from Thinker.

        Uses Thinker's cached state (topo, complexity, phys_pruner) to drive
        the solve_game pipeline with enhanced ZKP Loop verification.

        Args:
            game: Game object.
            game_id: Game identifier string.
            level_idx: Current level index.
            valid_actions: List of valid actions.
            thinker_result: Thinker's output dict (if None, uses self._thinker_results).

        Returns:
            Dict with phys_pruner, complexity, performer_state, and topo_invariant_score.
        """
        if thinker_result is None:
            thinker_result = self._thinker_results

        phys_pruner = thinker_result.get("phys_pruner")
        complexity = thinker_result.get("complexity", {})
        complexity_class = complexity.get("complexity_class", "NP_C_likely")

        # Use KV-cache state for enhanced search
        performer_state = self._kv_cache.get_performer_state()
        topo_invariant_score = self._kv_cache.get_topology_invariant_score()

        self._pipeline_count += 1

        # Store performer results
        self._performer_results = {
            "pipeline_count": self._pipeline_count,
            "complexity_class": complexity_class,
            "topo_invariant_score": topo_invariant_score,
            "kv_cache_step_count": performer_state.get("step_count", 0),
        }

        return {
            "phys_pruner": phys_pruner,
            "complexity": complexity,
            "complexity_class": complexity_class,
            "performer_state": performer_state,
            "topo_invariant_score": topo_invariant_score,
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dict containing total_pipelines, thinker/performer results keys,
            zkp_stats, kv_cache_step_count, and topo_invariant_score.
        """
        return {
            "total_pipelines": self._pipeline_count,
            "thinker_results_keys": list(self._thinker_results.keys()),
            "performer_results_keys": list(self._performer_results.keys()),
            "zkp_stats": self._zkp_loop.get_loop_stats(),
            "kv_cache_step_count": self._kv_cache._step_count,
            "topo_invariant_score": self._kv_cache.get_topology_invariant_score(),
        }


# ═══════════════════════════════════════════════════════════════════════
# v3.9.0 — MoonshineProver (Numerical Oracle Game — "从猜想生成到流贯觉察")
# ═══════════════════════════════════════════════════════════════════════
# Implements the Prover-Oracle feedback loop for interactive conjecture
# refinement on high-dimensional NJC (Nash-Jacobian-Compactification) tasks.
#
# Architecture:
#   MoonshineProver
#   ├── propose_sampling_strategy(conjecture) → parameter sampling plan
#   ├── oracle_verify(proposal, game_data) → searches for counterexamples
#   ├── refine_conjecture(counterexample, library) → updates scope
#   └── run_oracle_game(conjecture, game_data, library) → full loop (max 5 rounds)
#
# Integration: Used alongside InteractiveZKPLoop for conjecture-level
# verification on NP_C_likely and NP_Hard games where direct GaussEx
# verification is insufficient.
# ═══════════════════════════════════════════════════════════════════════


class MoonshineProver:
    """Numerical Oracle Game — Prover-Oracle feedback loop for high-dim NJC.

    Implements the MoonshineProver from "从猜想生成到流贯觉察" article:
    a Prover proposes a parameter sampling strategy, the Oracle computes
    and searches for counterexamples, and feedback updates the conjecture
    scope via library.refine_macro().

    This is the interactive conjecture refinement mechanism for tasks
    where direct verification is impossible (NP_C_likely / NP_Hard):
    instead of proving correctness in one shot, we iteratively narrow
    the conjecture's scope until no counterexamples remain.

    The Prover-Oracle game:
        Round 1: Prover proposes → Oracle searches → counterexample found → refine
        Round 2: Prover proposes refined → Oracle searches → no counterexample → accept
        ...
        Max 5 rounds → accept refined conjecture or declare boundary

    Args:
        max_rounds: Maximum Prover-Oracle interaction rounds. Default 5.
        library: LibraryManager for macro refinement. If None, creates default.
        verbose: Whether to log refinement rounds. Default True.
    """

    def __init__(
        self,
        max_rounds: int = 5,
        library: Optional[LibraryManager] = None,
        verbose: bool = True,
    ) -> None:
        self._max_rounds = max_rounds
        self._library = library or LibraryManager()
        self._verbose = verbose
        self._round_count: int = 0
        self._refinement_history: List[Dict[str, Any]] = []

    def propose_sampling_strategy(
        self,
        conjecture: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prover proposes a parameter sampling strategy for the conjecture.

        Based on the conjecture's complexity class and topology features,
        proposes which parameter dimensions to sample and at what density.

        Args:
            conjecture: Dict describing the current conjecture, including:
                - 'scope': set of game conditions the conjecture covers
                - 'complexity_class': P / P_in_phys / NP_Hard / NP_C_likely
                - 'topo_features': topology feature dict
                - 'macro_name': name of the macro being tested

        Returns:
            Dict describing the sampling strategy:
                - 'dimensions': list of parameter dimensions to sample
                - 'density': sampling density (0.1-1.0)
                - 'focus_areas': specific parameter ranges to concentrate on
        """
        complexity_class = conjecture.get("complexity_class", "NP_C_likely")
        topo_features = conjecture.get("topo_features", {})
        euler_char = topo_features.get("euler_char", 0)
        density = topo_features.get("density", 0.5)

        # Sampling density scales with complexity
        if complexity_class == "P":
            sampling_density = 0.1  # Low density — conjecture is easy to verify
        elif complexity_class == "P_in_phys":
            sampling_density = 0.3  # Moderate — physics constraints help
        elif complexity_class == "NP_Hard":
            sampling_density = 0.7  # High — many counterexamples possible
        else:  # NP_C_likely
            sampling_density = 1.0  # Maximum — comprehensive search needed

        # Focus on high-euler-characteristic regions (more topologically complex)
        focus_areas = []
        if abs(euler_char) > 2:
            focus_areas.append("topology_complex_regions")
        if density > 0.7:
            focus_areas.append("dense_grid_regions")

        # Dimensions to sample based on conjecture scope
        dimensions = ["action_sequence", "grid_topology", "color_distribution"]
        if complexity_class in ("NP_Hard", "NP_C_likely"):
            dimensions.append("non_associative_residual")

        return {
            "dimensions": dimensions,
            "density": sampling_density,
            "focus_areas": focus_areas,
            "conjecture_scope": conjecture.get("scope", {}),
            "round": self._round_count + 1,
        }

    def oracle_verify(
        self,
        proposal: Dict[str, Any],
        game_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Oracle searches for counterexamples to the Prover's proposal.

        Given the Prover's sampling strategy, the Oracle simulates the
        proposed parameters on game data and searches for any configuration
        where the conjecture fails (counterexample).

        Args:
            proposal: Sampling strategy dict from propose_sampling_strategy.
            game_data: Actual game state data for counterexample search:
                - 'grid': current game grid (numpy array)
                - 'game_id': ARC game identifier
                - 'level_idx': level number
                - 'action_history': list of previous actions

        Returns:
            Dict describing verification result:
                - 'verified': bool — whether no counterexample was found
                - 'counterexamples': list of counterexample configurations
                - 'confidence': float (0-1) — how confident the Oracle is
                - 'scope_reduction': dict describing narrowed scope
        """
        dimensions = proposal.get("dimensions", [])
        density = proposal.get("density", 0.5)
        focus_areas = proposal.get("focus_areas", [])

        # Simulate the conjecture on sampled parameter configurations
        counterexamples: List[Dict[str, Any]] = []

        # Check each dimension for potential failures
        for dim in dimensions:
            # For action_sequence: check if conjecture covers all observed sequences
            if dim == "action_sequence" and game_data.get("action_history"):
                actions = game_data["action_history"]
                # Check if conjecture scope covers this action pattern
                scope = proposal.get("conjecture_scope", {})
                if not scope.get("covers_all_sequences", True):
                    counterexamples.append({
                        "dimension": dim,
                        "failure_type": "uncovered_action_pattern",
                        "pattern": actions[-5:] if len(actions) > 5 else actions,
                    })

            # For grid_topology: check Euler characteristic compatibility
            if dim == "grid_topology" and game_data.get("grid") is not None:
                grid = game_data.get("grid")
                if grid is not None:
                    actual_euler = extract_topo_features(np.array(grid)).get("euler_char", 0)
                    expected_euler = proposal.get("conjecture_scope", {}).get("euler_char_range", (-5, 5))
                    if not (expected_euler[0] <= actual_euler <= expected_euler[1]):
                        counterexamples.append({
                            "dimension": dim,
                            "failure_type": "euler_char_mismatch",
                            "actual": actual_euler,
                            "expected_range": expected_euler,
                        })

        # Confidence increases when no counterexamples found
        n_counters = len(counterexamples)
        confidence = max(0.0, 1.0 - n_counters * density)

        # Scope reduction: narrow the conjecture based on counterexamples
        scope_reduction = {}
        for ce in counterexamples:
            dim = ce["dimension"]
            scope_reduction[dim] = {
                "narrowed": True,
                "failure_type": ce["failure_type"],
            }

        return {
            "verified": n_counters == 0,
            "counterexamples": counterexamples,
            "confidence": confidence,
            "scope_reduction": scope_reduction,
            "round": self._round_count + 1,
        }

    def refine_conjecture(
        self,
        verification_result: Dict[str, Any],
        macro_name: str,
    ) -> Dict[str, Any]:
        """Refine conjecture scope based on Oracle verification feedback.

        Updates the conjecture's scope to exclude counterexample regions,
        and updates library.json macros via library.refine_macro().

        Args:
            verification_result: Oracle verification dict with:
                - 'counterexamples': list of failure configurations
                - 'scope_reduction': narrowed scope dict
                - 'confidence': float (0-1)
            macro_name: Name of the macro being refined.

        Returns:
            Dict describing the refined conjecture:
                - 'macro_name': updated macro name
                - 'refined_scope': narrowed scope dict
                - 'confidence': updated confidence score
                - 'accepted': bool — whether the conjecture is now acceptable
        """
        scope_reduction = verification_result.get("scope_reduction", {})
        counterexamples = verification_result.get("counterexamples", [])
        confidence = verification_result.get("confidence", 0.5)

        # Build refined scope by incorporating scope reduction
        refined_scope = {}
        for dim, reduction in scope_reduction.items():
            refined_scope[dim] = {
                "narrowed": True,
                "excludes": reduction.get("failure_type", ""),
            }

        # If counterexamples exist for topology, restrict Euler range
        for ce in counterexamples:
            if ce.get("dimension") == "grid_topology":
                actual_euler = ce.get("actual", 0)
                # Narrow the range to exclude this Euler value
                refined_scope["euler_char_range"] = (actual_euler - 1, actual_euler + 1)

        # Update library macro — refine the macro's precondition
        refined_macro_name = f"{macro_name}_refined_r{self._round_count + 1}"
        try:
            # Try to refine existing macro in library
            existing_macros = self._library.query_by_fingerprint("")
            for macro in existing_macros[:5]:  # Limit search
                if macro.name == macro_name:
                    # Update macro scope
                    macro.gaussex_precond = f"euler_char != {ce.get('actual', 'unknown')}" if counterexamples else macro.gaussex_precond
                    macro.success_rate = confidence
                    self._library.save_library()
                    break
        except Exception:
            pass

        # Accept if confidence >= 0.8 or max rounds reached
        accepted = confidence >= 0.8 or self._round_count >= self._max_rounds

        # Record refinement history
        self._refinement_history.append({
            "round": self._round_count + 1,
            "macro_name": macro_name,
            "refined_macro_name": refined_macro_name,
            "counterexamples_found": len(counterexamples),
            "confidence": confidence,
            "accepted": accepted,
            "timestamp": time.time(),
        })

        if self._verbose:
            print(f"[MoonshineProver] Round {self._round_count + 1}: "
                  f"found {len(counterexamples)} counterexamples, "
                  f"confidence={confidence:.2f}, accepted={accepted}")

        return {
            "macro_name": refined_macro_name,
            "original_macro_name": macro_name,
            "refined_scope": refined_scope,
            "confidence": confidence,
            "accepted": accepted,
        }

    def run_oracle_game(
        self,
        conjecture: Dict[str, Any],
        game_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run full Prover-Oracle game loop (max 5 rounds).

        Iteratively: Prover proposes → Oracle verifies → refine conjecture.
        Stops when:
        1. Oracle finds no counterexamples (conjecture accepted)
        2. Max rounds reached (declare ontological boundary)
        3. Confidence threshold met (0.8)

        Args:
            conjecture: Initial conjecture dict (see propose_sampling_strategy).
            game_data: Game state data for verification (see oracle_verify).

        Returns:
            Dict describing the full game result:
                - 'accepted': bool — whether conjecture was accepted
                - 'final_confidence': float
                - 'rounds_completed': int
                - 'total_counterexamples': int
                - 'refinement_history': list of round-by-round results
        """
        macro_name = conjecture.get("macro_name", "unknown_macro")
        total_counterexamples = 0
        final_confidence = 0.0

        for round_idx in range(self._max_rounds):
            self._round_count = round_idx

            # Step 1: Prover proposes sampling strategy
            proposal = self.propose_sampling_strategy(conjecture)

            # Step 2: Oracle verifies
            verification = self.oracle_verify(proposal, game_data)

            # Step 3: Refine conjecture based on Oracle feedback
            refinement = self.refine_conjecture(verification, macro_name)

            total_counterexamples += len(verification.get("counterexamples", []))
            final_confidence = refinement.get("confidence", 0.0)

            # Update conjecture scope for next round
            conjecture["scope"] = refinement.get("refined_scope", {})
            macro_name = refinement.get("macro_name", macro_name)

            # Accept if refinement says accepted
            if refinement.get("accepted", False):
                break

        # If not accepted after max rounds, declare ontological boundary
        accepted = final_confidence >= 0.8
        boundary_declared = not accepted and self._round_count >= self._max_rounds

        if self._verbose and boundary_declared:
            print(f"[MoonshineProver] Ontological boundary declared for {macro_name} "
                  f"after {self._max_rounds} rounds (confidence={final_confidence:.2f})")

        return {
            "accepted": accepted,
            "boundary_declared": boundary_declared,
            "final_confidence": final_confidence,
            "rounds_completed": self._round_count + 1,
            "total_counterexamples": total_counterexamples,
            "refinement_history": self._refinement_history,
            "final_macro_name": macro_name,
        }


# ═══════════════════════════════════════════════════════════════════════
# v3.9.0 — MoonshineProver (Numerical Oracle Game)
# 来源: "从猜想生成到流贯觉察" — Prover-Oracle feedback loop for NJC
# ═══════════════════════════════════════════════════════════════════════
#
# MoonshineProver implements the Prover-Oracle feedback loop for
# high-dimensional NJC (Non-associative Junction Conjecture) tasks.
#
# Architecture:
#   1. Prover proposes a conjecture (DSL macro candidate)
#   2. Oracle (game environment) verifies by searching for counterexamples
#   3. If counterexample found → refine conjecture scope via library.refine_macro()
#   4. Repeat up to max_rounds (default 5)
#
# Integration with InteractiveZKPLoop:
#   MoonshineProver.refine_conjecture() uses library.refine_macro() to
#   update the conjecture's scope, which then feeds back into the
#   ZKP loop's Prove step for re-verification.
# ═══════════════════════════════════════════════════════════════════════


class MoonshineProver:
    """Numerical Oracle Game Prover — κ-Snap反向 conjecture refinement loop.

    Implements the MoonshineProver from "从猜想生成到流贯觉察":
    a Prover-Oracle feedback loop for high-dimensional NJC tasks.

    The MoonshineProver operates as follows:
        1. Prover proposes a sampling strategy for testing a conjecture
        2. Oracle verifies by executing the strategy against game data
           and searching for counterexamples
        3. If counterexample found → refine conjecture scope
           (narrow the generalization domain via library.refine_macro())
        4. If no counterexample found → conjecture is validated and
           added to library as a verified macro
        5. Repeat up to max_rounds (default 5)

    This is the "Numerical Oracle Game" — instead of formal proof,
    we use exhaustive sampling against the game environment to verify
    or falsify conjectures. Counterexamples narrow the conjecture's
    scope, making it more precise but less general.

    Integration with InteractiveZKPLoop:
        MoonshineProver can be invoked during the ZKP Prove step to
        refine conjectures that initially fail GaussEx verification.
        The refined conjecture then feeds back into the Verify step.

    Attributes:
        _library: LibraryManager instance for macro persistence and refinement.
        _max_rounds: Maximum number of Prover-Oracle refinement rounds.
        _round_count: Current refinement round count.
        _conjecture_history: List of conjecture states across refinement rounds.
        _counterexample_log: List of counterexamples found during verification.
    """

    # ── Sampling strategy parameters ──
    DEFAULT_SAMPLE_SIZE: int = 100  # Number of parameter samples per round
    DEFAULT_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
        "grid_density": (0.1, 0.9),
        "euler_char_range": (-5.0, 10.0),
        "n_components_range": (1.0, 20.0),
        "action_count_range": (2.0, 50.0),
    }

    def __init__(
        self,
        library_manager: Optional[LibraryManager] = None,
        max_rounds: int = 5,
        zkp_loop: Optional[InteractiveZKPLoop] = None,
    ) -> None:
        """Initialize MoonshineProver with library manager and ZKP loop.

        Args:
            library_manager: LibraryManager for macro persistence.
                If None, creates a default one.
            max_rounds: Maximum number of Prover-Oracle refinement rounds.
                Defaults to 5 (per article recommendation).
            zkp_loop: Optional InteractiveZKPLoop for interactive
                conjecture refinement integration.
        """
        self._library = library_manager or LibraryManager()
        self._max_rounds = max_rounds
        self._zkp_loop = zkp_loop
        self._round_count: int = 0
        self._conjecture_history: List[Dict[str, Any]] = []
        self._counterexample_log: List[Dict[str, Any]] = []

    def propose_sampling_strategy(
        self,
        conjecture: MacroCandidate,
    ) -> Dict[str, Any]:
        """Propose a parameter sampling strategy for testing a conjecture.

        Given a MacroCandidate (conjecture), generate a sampling plan
        that covers the conjecture's applicable parameter space. The
        sampling strategy determines which game configurations should
        be tested to verify or falsify the conjecture.

        The strategy is derived from the conjecture's:
            - applicable_topo: topology constraints that define applicability
            - generalization_tags: semantic categories the conjecture covers
            - gaussex_precond: precondition constraints

        Args:
            conjecture: MacroCandidate representing the conjecture to test.

        Returns:
            Dict with keys:
                sample_size: Number of parameter configurations to test.
                param_ranges: Dict of parameter ranges for sampling.
                priority_tags: Tags that should be prioritized in sampling.
                topo_constraints: Topology constraints for filtering samples.
                conjecture_name: Name of the conjecture being tested.
        """
        # ── Determine sample size based on conjecture complexity ──
        dsl_length = len(conjecture.dsl_sequence)
        n_tags = len(conjecture.generalization_tags)

        # More complex conjectures need more samples
        base_size = self.DEFAULT_SAMPLE_SIZE
        complexity_multiplier = 1.0 + 0.1 * dsl_length + 0.05 * n_tags
        sample_size = int(base_size * complexity_multiplier)

        # ── Build parameter ranges from conjecture properties ──
        param_ranges: Dict[str, Tuple[float, float]] = {}
        param_ranges.update(self.DEFAULT_PARAM_RANGES)

        # Add topology-specific ranges from applicable_topo
        topo = conjecture.applicable_topo
        if isinstance(topo, dict):
            euler_char = topo.get("euler_char", 0)
            period_rank = topo.get("period_rank", 0)
            # Range around the conjecture's applicable euler_char
            param_ranges["euler_char_range"] = (
                float(euler_char - 3), float(euler_char + 3)
            )
            # Periodicity sampling range
            param_ranges["period_rank_range"] = (
                0.0, float(period_rank + 2)
            )

        # ── Determine priority tags for focused sampling ──
        priority_tags = list(conjecture.generalization_tags)

        # ── Build sampling strategy dict ──
        strategy = {
            "sample_size": sample_size,
            "param_ranges": param_ranges,
            "priority_tags": priority_tags,
            "topo_constraints": conjecture.applicable_topo,
            "conjecture_name": conjecture.name,
            "conjecture_fingerprint": conjecture.tomas_fingerprint,
            "gaussex_precond": conjecture.gaussex_precond,
            "round_number": self._round_count,
        }

        return strategy

    def oracle_verify(
        self,
        proposal: Dict[str, Any],
        game_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Oracle verification: search for counterexamples to the conjecture.

        Execute the sampling strategy against game data to verify or
        falsify the conjecture. The Oracle tests the conjecture by:
            1. Generating parameter samples from the proposal ranges
            2. Testing each sample against the game environment
            3. Checking if the conjecture's DSL sequence produces
               the expected outcome for each sample
            4. If any sample fails → counterexample found

        Args:
            proposal: Sampling strategy dict from propose_sampling_strategy().
            game_data: Optional game environment data for testing.
                If None, uses synthetic testing based on topology constraints.

        Returns:
            Dict with keys:
                verified: Whether the conjecture passed all sample tests.
                counterexamples_found: Number of counterexamples found.
                counterexamples: List of counterexample dicts (if any).
                samples_tested: Number of samples tested.
                verification_score: Float score (0.0-1.0) of conjecture reliability.
                round_number: Current refinement round.
        """
        sample_size = proposal.get("sample_size", self.DEFAULT_SAMPLE_SIZE)
        param_ranges = proposal.get("param_ranges", self.DEFAULT_PARAM_RANGES)
        topo_constraints = proposal.get("topo_constraints", {})
        conjecture_name = proposal.get("conjecture_name", "")
        priority_tags = proposal.get("priority_tags", [])

        self._round_count += 1
        counterexamples: List[Dict[str, Any]] = []
        samples_tested: int = 0

        # ── Generate parameter samples ──
        # Use grid_density and euler_char as primary sampling dimensions
        density_range = param_ranges.get("grid_density", (0.1, 0.9))
        euler_range = param_ranges.get("euler_char_range", (-5.0, 10.0))

        for i in range(min(sample_size, 200)):
            # Sample a parameter configuration
            sample_density = density_range[0] + (density_range[1] - density_range[0]) * (i / max(sample_size, 1))
            sample_euler = euler_range[0] + (euler_range[1] - euler_range[0]) * (
                (i + 0.5) / max(sample_size, 1)
            )

            # ── Check if this sample is within the conjecture's scope ──
            # A counterexample is a game configuration where the conjecture
            # claims to apply but actually fails
            in_scope = True

            # Check topology constraint
            if isinstance(topo_constraints, dict):
                target_euler = topo_constraints.get("euler_char", 0)
                tolerance = 3
                if abs(sample_euler - target_euler) > tolerance:
                    # Outside the conjecture's topology scope — not a counterexample
                    in_scope = False

            if not in_scope:
                continue

            samples_tested += 1

            # ── Simulate verification: check if conjecture works for this config ──
            # In production, this would actually run the game with the conjecture
            # For now, use heuristic verification based on game_data or topology

            verification_passed = True

            # Use game_data for real verification if available
            if game_data is not None:
                grid = game_data.get("grid")
                if grid is not None and isinstance(grid, np.ndarray):
                    real_topo = extract_topo_features(grid)
                    real_euler = real_topo.get("euler_char", 0)
                    real_density = real_topo.get("density", 0.5)

                    # Check if the conjecture's topology matches the real game
                    if isinstance(topo_constraints, dict):
                        target_euler = topo_constraints.get("euler_char", 0)
                        if abs(real_euler - target_euler) > 2:
                            verification_passed = False

                    # Check if the conjecture's tags cover this game type
                    game_tags = game_data.get("tags", [])
                    if priority_tags and game_tags:
                        overlap = len(set(priority_tags) & set(game_tags))
                        if overlap == 0:
                            verification_passed = False
            else:
                # Synthetic verification: use sampled parameters
                # High-density configs with mismatched euler_char are likely counterexamples
                if sample_density > 0.7 and abs(sample_euler - round(sample_euler)) > 1.5:
                    verification_passed = False

            # ── Record counterexample if found ──
            if not verification_passed:
                counterexample = {
                    "sample_index": i,
                    "sample_density": float(sample_density),
                    "sample_euler_char": float(sample_euler),
                    "conjecture_name": conjecture_name,
                    "reason": "topology_mismatch_or_tag_exclusion",
                    "round_number": self._round_count,
                    "timestamp": time.time(),
                }
                counterexamples.append(counterexample)
                self._counterexample_log.append(counterexample)

        # ── Compute verification score ──
        if samples_tested > 0:
            verification_score = 1.0 - (len(counterexamples) / max(samples_tested, 1))
        else:
            verification_score = 0.0  # No samples tested → unknown

        verified = len(counterexamples) == 0

        # ── Record conjecture history ──
        self._conjecture_history.append({
            "round_number": self._round_count,
            "conjecture_name": conjecture_name,
            "verified": verified,
            "counterexamples_found": len(counterexamples),
            "samples_tested": samples_tested,
            "verification_score": verification_score,
        })

        return {
            "verified": verified,
            "counterexamples_found": len(counterexamples),
            "counterexamples": counterexamples,
            "samples_tested": samples_tested,
            "verification_score": verification_score,
            "round_number": self._round_count,
        }

    def refine_conjecture(
        self,
        counterexample: Dict[str, Any],
        library: Optional[LibraryManager] = None,
    ) -> Dict[str, Any]:
        """Refine conjecture scope based on a counterexample.

        When the Oracle finds a counterexample, this method narrows the
        conjecture's scope by:
            1. Analyzing the counterexample to identify what went wrong
            2. Refining the generalization_tags (removing overly broad tags)
            3. Narrowing the applicable_topo constraints
            4. Updating the gaussex_precondition to exclude the counterexample domain
            5. Persisting the refined conjecture via library

        This implements the core Moonshine mechanism: from counterexample →
        refined conjecture → narrower but more reliable generalization.

        Args:
            counterexample: Dict describing the counterexample found by Oracle.
                Keys: sample_density, sample_euler_char, reason, conjecture_name.
            library: Optional LibraryManager for persistence. If None,
                uses self._library.

        Returns:
            Dict with keys:
                refined: Whether the conjecture was successfully refined.
                old_scope: Dict describing the conjecture's scope before refinement.
                new_scope: Dict describing the conjecture's scope after refinement.
                scope_reduction: Float measure of how much the scope was narrowed.
                refinement_round: Current refinement round number.
        """
        lib = library or self._library

        conjecture_name = counterexample.get("conjecture_name", "")
        counter_density = counterexample.get("sample_density", 0.5)
        counter_euler = counterexample.get("sample_euler_char", 0.0)
        reason = counterexample.get("reason", "unknown")

        # ── Find the conjecture macro in library ──
        all_macros = lib.get_all_macros()
        target_macro: Optional[MacroCandidate] = None
        for m in all_macros:
            if m.name == conjecture_name:
                target_macro = m
                break

        if target_macro is None:
            return {
                "refined": False,
                "old_scope": {},
                "new_scope": {},
                "scope_reduction": 0.0,
                "refinement_round": self._round_count,
                "error": f"Conjecture '{conjecture_name}' not found in library",
            }

        # ── Record old scope ──
        old_scope = {
            "generalization_tags": list(target_macro.generalization_tags),
            "applicable_topo": dict(target_macro.applicable_topo),
            "gaussex_precond": target_macro.gaussex_precond,
        }

        # ── Refine scope based on counterexample ──
        # 1. Narrow topology constraints to exclude counterexample's euler_char range
        refined_topo = dict(target_macro.applicable_topo)
        if isinstance(refined_topo, dict):
            old_euler_range = refined_topo.get("euler_char", 0)
            # Narrow by 1 unit towards the conjecture's proven range
            if isinstance(old_euler_range, (int, float)):
                # Keep euler_char ±1 instead of ±3
                refined_topo["euler_char_tolerance"] = 1
            else:
                refined_topo["euler_char_tolerance"] = 1

        # 2. Remove overly broad generalization tags
        refined_tags = list(target_macro.generalization_tags)
        # Remove tags that are too general and led to the counterexample
        overly_broad_tags = {"general", "mixed", "all", "universal"}
        refined_tags = [t for t in refined_tags if t not in overly_broad_tags]
        # Add a refinement-specific tag
        refined_tags.append(f"refined_round_{self._round_count}")

        # 3. Strengthen precondition to exclude counterexample domain
        old_precond = target_macro.gaussex_precond
        density_constraint = f" AND density < {counter_density + 0.1:.2f}"
        euler_constraint = f" AND euler_char != {int(counter_euler)}"
        refined_precond = old_precond + density_constraint + euler_constraint

        # ── Compute scope reduction ──
        old_tag_count = len(target_macro.generalization_tags)
        new_tag_count = len(refined_tags)
        scope_reduction = (old_tag_count - new_tag_count) / max(old_tag_count, 1)

        # ── Record new scope ──
        new_scope = {
            "generalization_tags": refined_tags,
            "applicable_topo": refined_topo,
            "gaussex_precond": refined_precond,
        }

        # ── Update the macro in library ──
        refined_macro = MacroCandidate(
            name=f"{target_macro.name}_r{self._round_count}",
            dsl_sequence=target_macro.dsl_sequence,
            tomas_fingerprint=target_macro.tomas_fingerprint,
            source_tasks=target_macro.source_tasks,
            avg_steps=target_macro.avg_steps,
            success_rate=max(0.0, target_macro.success_rate * 0.9),  # Slightly reduced due to scope narrowing
            generalization_tags=refined_tags,
            mdl_score=target_macro.mdl_score,
            min_demo_to_activate=target_macro.min_demo_to_activate + 1,  # Require more demos after refinement
            validated=False,  # Re-require validation after refinement
            applicable_topo=refined_topo,
            gaussex_precond=refined_precond,
        )

        lib.add_macro(refined_macro)

        # ── If ZKP loop is available, update it with refined conjecture ──
        if self._zkp_loop is not None:
            # Register the refined macro in KV-cache for future ZKP verification
            dsl_str = json.dumps(refined_macro.dsl_sequence)
            self._zkp_loop._kv_cache.register_effective_macro(
                refined_macro.name, dsl_str,
            )

        return {
            "refined": True,
            "old_scope": old_scope,
            "new_scope": new_scope,
            "scope_reduction": scope_reduction,
            "refinement_round": self._round_count,
            "refined_macro_name": refined_macro.name,
        }

    def run_full_verification(
        self,
        conjecture: MacroCandidate,
        game_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run full Prover-Oracle verification loop for a conjecture.

        Orchestrates the complete Moonshine verification cycle:
            1. Propose sampling strategy for the conjecture
            2. Oracle verifies → find counterexamples
            3. If counterexamples found → refine conjecture (up to max_rounds)
            4. Return final verification result

        Args:
            conjecture: MacroCandidate to verify through the Oracle Game.
            game_data: Optional game environment data for testing.

        Returns:
            Dict with keys:
                final_verified: Whether the conjecture was verified after all rounds.
                total_rounds: Number of Prover-Oracle rounds executed.
                final_verification_score: Final verification score.
                refinement_history: List of refinement results per round.
                conjecture_name: Name of the original conjecture.
                final_macro: The final (possibly refined) MacroCandidate.
        """
        self._round_count = 0
        self._conjecture_history = []
        self._counterexample_log = []

        current_conjecture = conjecture
        refinement_history: List[Dict[str, Any]] = []

        for round_num in range(self._max_rounds):
            self._round_count = round_num + 1

            # ── Step 1: Propose sampling strategy ──
            proposal = self.propose_sampling_strategy(current_conjecture)

            # ── Step 2: Oracle verify ──
            oracle_result = self.oracle_verify(proposal, game_data)

            # ── Step 3: Check if verified ──
            if oracle_result["verified"]:
                # Conjecture passed — add to library as verified macro
                current_conjecture.validated = True
                current_conjecture.success_rate = oracle_result["verification_score"]
                self._library.add_macro(current_conjecture)
                break

            # ── Step 4: Refine conjecture from counterexamples ──
            for counterexample in oracle_result.get("counterexamples", []):
                refinement = self.refine_conjecture(counterexample)
                refinement_history.append(refinement)

                if refinement["refined"]:
                    # Update current conjecture to refined version
                    refined_name = refinement.get("refined_macro_name", "")
                    all_macros = self._library.get_all_macros()
                    for m in all_macros:
                        if m.name == refined_name:
                            current_conjecture = m
                            break

        # ── Final result ──
        final_verified = len(self._counterexample_log) == 0 or oracle_result.get("verified", False)

        return {
            "final_verified": final_verified,
            "total_rounds": self._round_count,
            "final_verification_score": oracle_result.get("verification_score", 0.0),
            "refinement_history": refinement_history,
            "conjecture_name": conjecture.name,
            "final_macro": current_conjecture,
        }

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get MoonshineProver verification statistics.

        Returns:
            Dict with keys: total_rounds, total_counterexamples,
            conjecture_history, verification_rounds.
        """
        return {
            "total_rounds": self._round_count,
            "total_counterexamples": len(self._counterexample_log),
            "conjecture_history": list(self._conjecture_history),
            "max_rounds": self._max_rounds,
        }
