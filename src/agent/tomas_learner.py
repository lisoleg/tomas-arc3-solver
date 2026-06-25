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
    retrieval (fingerprint + tag matching), and redundancy pruning.

    Library schema (v3.2.0):
    {
        "_schema_version": "3.2.0",
        "_description": "TOMAS DSL Macro Library",
        "_tomas_framework": {...},
        "abstractions": [...]
    }
    """

    # Default library path
    DEFAULT_LIBRARY_PATH: str = "library.json"

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
        """Load existing library or create empty one with v3.2.0 schema.

        Returns:
            Library dict with v3.2.0 schema.
        """
        if self.library_path.exists():
            try:
                with open(self.library_path, "r", encoding="utf-8") as f:
                    lib = json.load(f)
                # Check schema version
                if lib.get("_schema_version") != "3.2.0":
                    # Upgrade to v3.2.0
                    lib = self._upgrade_schema(lib)
                return lib
            except (json.JSONDecodeError, IOError):
                pass

        # Create new library with v3.2.0 schema
        return {
            "_schema_version": "3.2.0",
            "_description": "TOMAS DSL Macro Library for ARC-AGI-3 Solver",
            "_tomas_framework": {
                "version": "3.2.0-dev",
                "core_modules": [
                    "kappa_snap_searcher",
                    "nar_conv",
                    "tomas_learner",
                    "library_manager",
                    "gaussex_verifier",
                ],
                "asym_index_enabled": True,
                "narla_integration": True,
            },
            "_note": "Macros are validated by ψ-Audit before activation. "
                     "TOMAS fingerprints enable cross-game pattern matching.",
            "abstractions": [],
        }

    def _upgrade_schema(self, old_lib: Dict[str, Any]) -> Dict[str, Any]:
        """Upgrade library from older schema to v3.2.0.

        Args:
            old_lib: Library dict with old schema.

        Returns:
            Upgraded library dict with v3.2.0 schema.
        """
        # Preserve existing abstractions
        old_abstractions = old_lib.get("abstractions", [])

        return {
            "_schema_version": "3.2.0",
            "_description": "TOMAS DSL Macro Library for ARC-AGI-3 Solver",
            "_tomas_framework": {
                "version": "3.2.0-dev",
                "core_modules": [
                    "kappa_snap_searcher",
                    "nar_conv",
                    "tomas_learner",
                    "library_manager",
                    "gaussex_verifier",
                ],
                "asym_index_enabled": True,
                "narla_integration": True,
            },
            "_note": "Macros are validated by ψ-Audit before activation. "
                     "TOMAS fingerprints enable cross-game pattern matching.",
            "abstractions": old_abstractions,
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

    This is the main orchestrator that:
    1. Records gameplay episodes (online phase)
    2. Sleep-steps to extract causal patterns from buffered episodes
    3. ψ-Audits candidate macros for validity
    4. Consolidates validated macros into library.json

    Integration with ARC-AGI-3 Solver:
        - PlannerAgent initializes TOMASLearner
        - Each level attempt records an EpisodeTrace
        - After game completion, sleep_step() extracts patterns
        - Validated macros are available via get_relevant_macros()

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

        self.mdl_threshold = mdl_threshold
        self.psi_audit_threshold = psi_audit_threshold
        self.buffer_size = buffer_size

        # Episode buffer for sleep-step processing
        self._episode_buffer: List[EpisodeTrace] = []

        # ψ-Audit log for alignment faking detection
        self._psi_audit_log: List[Dict[str, Any]] = []

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
