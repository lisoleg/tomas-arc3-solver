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
    │   ├── extract_sprite_movement (检测sprite移动方向)
    │   ├── extract_click_effect (检测点击切换)
    │   └── extract_navigation_pattern (检测导航模式)
    ├── DSLMacroAbstractor (DSL宏抽象)
    │   ├── compress_repeated_actions (合并重复动作)
    │   ├── compute_mdl_score (计算MDL评分)
    │   └── compute_tomas_fingerprint (八元体相位哈希)
    ├── LibraryManager (宏库管理)
    │   ├── load_library / save_library (持久化)
    │   ├── query_by_fingerprint (TOMAS双存储检索)
    │   ├── prune_redundant (冗余修剪)
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
    if grid is None or grid.size == 0:
        return {
            "euler_char": 0,
            "period_rank": 0,
            "symmetry": [],
            "component_count": 0,
            "hole_count": 0,
            "density": 0.0,
        }

    # ── Density ──
    density = float(np.count_nonzero(grid)) / grid.size

    # ── Connected components (4-connectivity) ──
    binary = (grid != 0).astype(np.int32)
    from scipy.ndimage import label as ndlabel
    _, component_count = ndlabel(binary)
    # Estimate holes: in a padded version, components that don't touch border
    padded = np.pad(binary, 1, mode='constant', constant_values=1)
    padded_labels, _ = ndlabel(padded)
    border_labels = set(padded_labels[0, :]) | set(padded_labels[-1, :]) | \
                    set(padded_labels[:, 0]) | set(padded_labels[:, -1])
    hole_count = component_count - len(border_labels & set(range(1, component_count + 1)))
    # Clamp hole_count to non-negative
    hole_count = max(0, hole_count)

    euler_char = component_count - hole_count

    # ── Periodicity rank ──
    period_rank = 0
    if grid.shape[0] > 1:
        row_hashes = [hash(grid[i].tobytes()) for i in range(grid.shape[0])]
        if len(set(row_hashes)) <= 2:
            period_rank = 1  # Row-periodic
    if period_rank == 1 and grid.shape[1] > 1:
        col_hashes = [hash(grid[:, j].tobytes()) for j in range(grid.shape[1])]
        if len(set(col_hashes)) <= 2:
            period_rank = 2  # Grid-periodic

    # ── Symmetry ──
    symmetry = []
    if grid.shape[0] > 1 and np.allclose(grid, grid[::-1], atol=0):
        symmetry.append("horizontal")
    if grid.shape[1] > 1 and np.allclose(grid, grid[:, ::-1], atol=0):
        symmetry.append("vertical")
    if grid.shape[0] == grid.shape[1] and grid.shape[0] > 1:
        if np.allclose(grid, np.rot90(grid, 2), atol=0):
            symmetry.append("rotational_180")
        if np.allclose(grid, np.rot90(grid), atol=0):
            symmetry.append("rotational_90")

    return {
        "euler_char": euler_char,
        "period_rank": period_rank,
        "symmetry": symmetry,
        "component_count": component_count,
        "hole_count": hole_count,
        "density": density,
    }


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


# ============================================================================
# Causal Pattern Extractor
# ============================================================================

class CausalPatternExtractor:
    """Extract causal patterns from gameplay episode traces.

    Analyzes the differences between pre-state and post-state grids
    to identify causal relationships between actions and effects.

    Three core pattern types:
    1. sprite_movement: Sprite position changes after keyboard actions
    2. click_effect: Grid state changes after click actions (toggles, selections)
    3. navigation_pattern: Sequence of actions leading to goal achievement
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

        Analyzes the difference between pre_state and post_state to
        determine if a keyboard action caused a sprite to move in a
        consistent direction.

        Args:
            trace: ActionTrace with pre/post state grids.

        Returns:
            Pattern dict with keys: type, direction, distance, sprite_id.
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
                    return {
                        "type": "sprite_movement",
                        "direction": dir_name,
                        "distance": float(distance),
                        "sprite_color": int(color),
                        "action_params": trace.action_params,
                    }

        return None

    def extract_click_effect(
        self,
        trace: ActionTrace,
    ) -> Optional[Dict[str, Any]]:
        """Detect click toggle effect from an action trace.

        Analyzes whether a click action toggled or changed the state
        of a specific grid element (switch, button, selector).

        Args:
            trace: ActionTrace with pre/post state grids.

        Returns:
            Pattern dict with keys: type, position, toggle_type, colors_changed.
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

        return {
            "type": "click_effect",
            "click_position": (click_row, click_col),
            "toggle_type": "switch" if len(nearby_changes) <= 2 else "area_change",
            "colors_changed": [(int(p), int(q)) for p, q in color_pairs],
            "num_cells_changed": len(nearby_changes),
            "action_params": trace.action_params,
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
    """

    # MDL compression threshold — macros must achieve this compression ratio
    MDL_THRESHOLD: float = 0.7

    def __init__(self) -> None:
        """Initialize the macro abstractor."""
        self._macro_cache: Dict[str, MacroCandidate] = {}

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
        return [MacroCandidate(**a) for a in self._library.get("abstractions", [])]


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

        self.mdl_threshold = mdl_threshold
        self.psi_audit_threshold = psi_audit_threshold
        self.buffer_size = buffer_size

        # Episode buffer for sleep-step processing
        self._episode_buffer: List[EpisodeTrace] = []

        # ψ-Audit log for alignment faking detection
        self._psi_audit_log: List[Dict[str, Any]] = []

        # Fast-Path dispatch log for monitoring
        self._fast_path_log: List[Dict[str, Any]] = []

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
