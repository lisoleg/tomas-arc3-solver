"""
Graph-based Exploration Engine for ARC-AGI-3.

Based on the 3rd-place solution (arXiv:2512.24156) with key enhancements:
- Cross-level mechanism memory (missing in original)
- Reset action marking (bug fix from original)
- Strategic planning switch when mechanisms are understood
- CCG rotation tracking for rotation-based games

Architecture:
    FrameProcessor → StateGraph → ActionSelector → MechanismDetector

Author: TOMAS Team
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Component:
    """A single-color connected component in the frame.

    Attributes:
        color: The color value (0-15) of this component.
        pixels: Frozen set of (row, col) pixel coordinates.
        bbox: Bounding box as (min_row, min_col, max_row, max_col).
        area: Number of pixels in the component.
        centroid: Center of mass as (row, col).
    """
    color: int
    pixels: frozenset[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]

    @property
    def is_edge(self) -> bool:
        """Check if component touches the frame edge."""
        min_r, min_c, max_r, max_c = self.bbox
        return min_r == 0 or min_c == 0 or max_r == 63 or max_c == 63

    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def aspect_ratio(self) -> float:
        """Width / height ratio."""
        h = self.height
        return self.width / h if h > 0 else 0.0


@dataclass
class ActionInfo:
    """Information about an action from a specific state.

    Attributes:
        tested: Whether this action has been tried from this state.
        dest_hash: Hash of the destination state (None if untested or reset).
        causes_reset: Whether this action triggers a game/level reset.
        causes_level_change: Whether this action triggers a level transition.
        priority: Priority level (0=highest, 4=lowest).
    """
    tested: bool = False
    dest_hash: Optional[str] = None
    causes_reset: bool = False
    causes_level_change: bool = False
    priority: int = 0


@dataclass
class StateNode:
    """A node in the state graph.

    Attributes:
        hash: Unique hash of the masked frame.
        actions: Mapping of action_id -> ActionInfo.
        distance_to_frontier: BFS distance to nearest frontier (-1 if uncomputed).
        is_initial: Whether this is the initial state of a level.
    """
    hash: str
    actions: dict[int, ActionInfo] = field(default_factory=dict)
    distance_to_frontier: int = -1
    is_initial: bool = False

    def has_untested(self, max_priority: int = 4) -> bool:
        """Check if this node has untested actions within priority limit."""
        return any(
            not info.tested and info.priority <= max_priority
            for info in self.actions.values()
        )

    def get_untested(self, max_priority: int = 4) -> list[int]:
        """Get list of untested action IDs within priority limit."""
        return [
            aid for aid, info in self.actions.items()
            if not info.tested and info.priority <= max_priority
        ]


# ============================================================================
# Frame Processor
# ============================================================================

class FrameProcessor:
    """Processes 64x64 frames for the graph explorer.

    Responsibilities:
        - Segment frame into single-color connected components
        - Detect and mask status bars (step counters, level indicators)
        - Generate state hash from masked frame
        - Group components by exploration priority

    The status bar detection uses frame-to-frame change frequency:
    rows that change in >30% of frames are likely status bars and get masked
    to prevent state space explosion from counter changes.
    """

    MIN_COMPONENT_SIZE: int = 2
    STATUS_BAR_CHANGE_THRESHOLD: float = 0.3

    def __init__(self) -> None:
        """Initialize the frame processor."""
        self._frame_history: list[np.ndarray] = []
        self._status_bar_rows: set[int] = set()
        self._initial_frame: Optional[np.ndarray] = None

    def process(self, grid: np.ndarray) -> tuple[str, list[Component], np.ndarray]:
        """Process a frame and return state hash, components, and masked grid.

        Args:
            grid: 64x64 int8 numpy array representing the game frame.

        Returns:
            Tuple of (state_hash, components, masked_grid).
        """
        grid = np.asarray(grid, dtype=np.int8)

        # Track frame history for status bar detection
        if self._initial_frame is None:
            self._initial_frame = grid.copy()
        self._frame_history.append(grid.copy())
        if len(self._frame_history) > 30:
            self._frame_history = self._frame_history[-30:]

        # Detect and update status bar rows
        self._detect_status_bars()

        # Mask status bars
        masked = grid.copy()
        for r in self._status_bar_rows:
            masked[r, :] = 0

        # Generate state hash from masked grid
        state_hash = hashlib.md5(masked.tobytes()).hexdigest()

        # Find connected components in masked grid
        components = self._find_components(masked)

        return state_hash, components, masked

    def _detect_status_bars(self) -> None:
        """Detect rows that change frequently between frames (likely status bars)."""
        if len(self._frame_history) < 3:
            return

        n = len(self._frame_history)
        change_counts = np.zeros(64, dtype=np.int32)

        for i in range(1, n):
            diff = self._frame_history[i] != self._frame_history[i - 1]
            change_counts += diff.sum(axis=1)

        avg_changes = change_counts / max(n - 1, 1)
        threshold = 64 * self.STATUS_BAR_CHANGE_THRESHOLD

        for r in range(64):
            if avg_changes[r] > threshold:
                self._status_bar_rows.add(r)

    def _find_components(self, grid: np.ndarray) -> list[Component]:
        """Find single-color connected components using scipy.ndimage.label.

        Args:
            grid: 64x64 masked grid.

        Returns:
            List of Component objects sorted by area (descending).
        """
        components: list[Component] = []

        for color in np.unique(grid):
            if color == 0:
                continue

            mask = grid == color
            labeled, num = ndimage.label(mask)

            for i in range(1, num + 1):
                coords = np.argwhere(labeled == i)
                if len(coords) < self.MIN_COMPONENT_SIZE:
                    continue

                pixel_set = frozenset(map(tuple, coords))
                min_r, min_c = coords[:, 0].min(), coords[:, 1].min()
                max_r, max_c = coords[:, 0].max(), coords[:, 1].max()

                components.append(Component(
                    color=int(color),
                    pixels=pixel_set,
                    bbox=(int(min_r), int(min_c), int(max_r), int(max_c)),
                    area=len(coords),
                    centroid=(float(coords[:, 0].mean()), float(coords[:, 1].mean())),
                ))

        components.sort(key=lambda c: c.area, reverse=True)
        return components

    def reset_level(self) -> None:
        """Reset per-level state (frame history and status bar detection)."""
        self._frame_history.clear()
        self._initial_frame = None
        self._status_bar_rows.clear()


# ============================================================================
# State Graph
# ============================================================================

class StateGraph:
    """Directed graph of explored states and action transitions.

    Nodes are unique game states (identified by masked frame hash).
    Edges are action transitions between states.

    Key operations:
        - add_or_update_node: Register a state with its available actions
        - record_transition: Record the result of taking an action
        - bfs_to_nearest_frontier: Find shortest path to unexplored actions
    """

    def __init__(self) -> None:
        """Initialize an empty state graph."""
        self.nodes: dict[str, StateNode] = {}
        self.edges: dict[tuple[str, int], str] = {}
        self._initial_hash: Optional[str] = None

    def add_or_update_node(
        self, state_hash: str, available_actions: list[int], is_initial: bool = False
    ) -> StateNode:
        """Add a node if new, or update its available actions if existing.

        Args:
            state_hash: Hash of the masked frame.
            available_actions: List of action IDs available in this state.
            is_initial: Whether this is the initial state of a level.

        Returns:
            The StateNode for this state.
        """
        if state_hash not in self.nodes:
            node = StateNode(
                hash=state_hash,
                actions={a: ActionInfo() for a in available_actions},
                is_initial=is_initial,
            )
            self.nodes[state_hash] = node
            if is_initial and self._initial_hash is None:
                self._initial_hash = state_hash
        else:
            node = self.nodes[state_hash]
            for a in available_actions:
                if a not in node.actions:
                    node.actions[a] = ActionInfo()

        return node

    def record_transition(
        self,
        src_hash: str,
        action: int,
        dest_hash: str,
        causes_reset: bool = False,
        causes_level_change: bool = False,
    ) -> None:
        """Record the result of taking an action from a source state.

        Args:
            src_hash: Hash of the source state.
            action: Action ID that was taken.
            dest_hash: Hash of the destination state.
            causes_reset: Whether this action triggered a reset.
            causes_level_change: Whether this action triggered a level transition.
        """
        if src_hash in self.nodes:
            info = self.nodes[src_hash].actions.get(action)
            if info is None:
                info = ActionInfo()
                self.nodes[src_hash].actions[action] = info
            info.tested = True
            info.dest_hash = dest_hash
            info.causes_reset = causes_reset
            info.causes_level_change = causes_level_change

        # Don't add reset transitions as normal edges
        if not causes_reset:
            self.edges[(src_hash, action)] = dest_hash

    def get_frontier_nodes(self, max_priority: int = 4) -> list[StateNode]:
        """Get all nodes that have untested actions within priority limit.

        Args:
            max_priority: Maximum priority level to consider.

        Returns:
            List of StateNode objects with untested actions.
        """
        return [
            node for node in self.nodes.values()
            if node.has_untested(max_priority)
        ]

    def bfs_to_nearest_frontier(
        self, start_hash: str, max_priority: int = 4
    ) -> Optional[list[int]]:
        """BFS to find shortest action sequence to nearest frontier node.

        A frontier node is one with untested actions. This navigates the
        graph to reach unexplored parts of the state space.

        Args:
            start_hash: Hash of the current state.
            max_priority: Maximum priority for considering actions untested.

        Returns:
            List of action IDs to execute, or None if no frontier reachable.
            Empty list means current state is already a frontier.
        """
        if start_hash not in self.nodes:
            return None

        start_node = self.nodes[start_hash]

        # Already at a frontier
        if start_node.has_untested(max_priority):
            return []

        # BFS through known transitions
        queue: deque[tuple[str, list[int]]] = deque([(start_hash, [])])
        visited: set[str] = {start_hash}

        while queue:
            current_hash, path = queue.popleft()
            current_node = self.nodes.get(current_hash)
            if current_node is None:
                continue

            for action_id, info in current_node.actions.items():
                if info.dest_hash is None or info.causes_reset:
                    continue

                dest_hash = info.dest_hash
                if dest_hash in visited:
                    continue

                visited.add(dest_hash)
                new_path = path + [action_id]

                dest_node = self.nodes.get(dest_hash)
                if dest_node and dest_node.has_untested(max_priority):
                    return new_path

                queue.append((dest_hash, new_path))

        return None

    def reset(self) -> None:
        """Reset the graph for a new level."""
        self.nodes.clear()
        self.edges.clear()
        self._initial_hash = None

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)


# ============================================================================
# Mechanism Memory (Cross-Level)
# ============================================================================

@dataclass
class MechanismMemory:
    """Cross-level memory of discovered game mechanisms.

    This persists across levels within a game, allowing the agent to
    leverage knowledge from earlier levels in later ones.

    Attributes:
        goal_colors: Colors associated with goal objects.
        wall_colors: Colors associated with impassable walls.
        switcher_colors: Colors associated with property switchers.
        refill_colors: Colors associated with step refills.
        player_color: Color of the player sprite.
        levels_seen: Number of levels encountered.
        mechanisms_confident: Whether enough mechanisms are understood
            to switch from exploration to strategic planning.
        level_patterns: Discovered patterns per level index.
    """
    goal_colors: set[int] = field(default_factory=set)
    wall_colors: set[int] = field(default_factory=set)
    switcher_colors: set[int] = field(default_factory=set)
    refill_colors: set[int] = field(default_factory=set)
    player_color: Optional[int] = None
    levels_seen: int = 0
    mechanisms_confident: bool = False
    level_patterns: dict[int, str] = field(default_factory=dict)

    def update_from_transition(
        self,
        prev_components: list[Component],
        curr_components: list[Component],
        state_changed: bool,
        level_changed: bool,
    ) -> None:
        """Update mechanism memory from a state transition.

        Args:
            prev_components: Components in the previous frame.
            curr_components: Components in the current frame.
            state_changed: Whether the game state changed.
            level_changed: Whether a level transition occurred.
        """
        prev_colors = {c.color for c in prev_components}
        curr_colors = {c.color for c in curr_components}

        new_colors = curr_colors - prev_colors
        gone_colors = prev_colors - curr_colors

        if level_changed:
            self.mechanisms_confident = True

        # Heuristic: colors that disappear might be goals or refills
        if gone_colors:
            for c in gone_colors:
                if c not in self.wall_colors:
                    self.goal_colors.add(c)

    def reset_level(self) -> None:
        """Reset per-level state while keeping cross-level memory."""
        pass  # Cross-level memory persists


# ============================================================================
# Explorer Agent
# ============================================================================

class ExplorerAgent:
    """Graph-based exploration agent for ARC-AGI-3.

    Combines systematic state-space exploration with cross-level mechanism
    memory. Based on the 3rd-place solution (arXiv:2512.24156) with
    enhancements for cross-level knowledge transfer and reset handling.

    The agent operates in two modes:
        1. Exploration mode (default): Systematic graph-based exploration
        2. Strategic mode (when mechanisms understood): Directed planning

    Args:
        max_steps: Maximum total steps for the game.
        stagnation_threshold: Steps without progress before giving up.
        level_baselines: Human baseline steps per level (for reporting).
    """

    def __init__(
        self,
        max_steps: int = 800,
        stagnation_threshold: int = 200,
        level_baselines: Optional[list[int]] = None,
    ) -> None:
        """Initialize the explorer agent."""
        self.max_steps = max_steps
        self.stagnation_threshold = stagnation_threshold
        self.level_baselines = level_baselines or []

        # Core components
        self.frame_processor = FrameProcessor()
        self.graph = StateGraph()
        self.mechanism_memory = MechanismMemory()

        # Step tracking
        self._step: int = 0
        self._stagnation: int = 0
        self._levels_completed: int = 0
        self._game_over_count: int = 0
        self._resets: int = 0
        self._total_actions: int = 0
        self._action_history: list[int] = []

        # Per-level state
        self._level_initial_hash: Optional[str] = None
        self._level_start_step: int = 0
        self._level_step_counts: list[int] = []

        # Navigation state
        self._current_path: list[int] = []
        self._current_path_idx: int = 0

        # Previous frame state
        self._prev_state = None
        self._prev_hash: Optional[str] = None
        self._prev_components: list[Component] = []
        self._last_action: Optional[int] = None

        # Game state
        self._game_started: bool = False

        # Player tracking (goal-directed navigation)
        self._player_color: Optional[int] = None
        self._player_area: int = 0
        self._player_pos: Optional[tuple[float, float]] = None
        self._prev_player_pos: Optional[tuple[float, float]] = None
        self._action_directions: dict[int, tuple[int, int]] = {}  # action -> (dr, dc)
        self._action_dir_history: dict[int, list[tuple[int, int]]] = {}  # for voting

        # Target detection
        self._target_candidates: list[tuple[float, float, int]] = []  # (row, col, color)
        self._current_target: Optional[tuple[float, float]] = None
        self._visited_targets: set[tuple[float, float]] = set()
        self._failed_targets: set[tuple[float, float]] = set()

        # Game over learning
        self._dangerous_actions: dict[int, int] = {}  # action -> danger count

    def is_done(self, frames: list, latest_frame) -> bool:
        """Check if the game is complete.

        Args:
            frames: List of all frame observations so far.
            latest_frame: The most recent frame observation.

        Returns:
            True if the game is won or limits are reached.
        """
        try:
            from arcengine import GameState
            return (
                latest_frame.state == GameState.WIN
                or self._step >= self.max_steps
                or self._stagnation >= self.stagnation_threshold
            )
        except ImportError:
            return self._step >= self.max_steps

    def choose_action(self, frames: list, latest_frame) -> 'GameAction':
        """Choose the next action based on current observation.

        Implements the hierarchical action selection strategy:
            1. Handle game over (send RESET)
            2. Process frame (hash + components)
            3. Record transition in state graph
            4. Select action (untested > frontier navigation > random)

        Args:
            frames: List of all frame observations.
            latest_frame: Current frame observation.

        Returns:
            GameAction to execute.
        """
        from arcengine import GameAction, GameState

        self._step += 1
        self._total_actions += 1

        grid = np.array(latest_frame.frame[0])
        available_actions = list(latest_frame.available_actions)
        state = latest_frame.state
        levels_completed = latest_frame.levels_completed

        # Handle GAME_OVER: must send RESET
        if state == GameState.GAME_OVER:
            self._game_over_count += 1
            self._resets += 1
            self._action_history.append(0)
            self._last_action = 0
            self._prev_hash = None
            self._prev_components = []
            self._current_path = []
            self._current_path_idx = 0
            print(f"    [GAME_OVER] step={self._step} sending RESET "
                  f"(game_over_count={self._game_over_count})")
            return GameAction.RESET

        # Check for level change
        level_changed = levels_completed > self._levels_completed
        if level_changed:
            self._on_level_complete(levels_completed)

        # Process frame
        state_hash, components, masked = self.frame_processor.process(grid)

        # Detect reset (returned to initial state without level change)
        is_reset = self._detect_reset(state_hash, state, level_changed)

        # Record transition from previous state
        if self._prev_hash is not None and self._last_action is not None:
            self.graph.record_transition(
                self._prev_hash,
                self._last_action,
                state_hash,
                causes_reset=is_reset,
                causes_level_change=level_changed,
            )
            self.mechanism_memory.update_from_transition(
                self._prev_components, components,
                state != self._prev_state, level_changed,
            )

        # Add/update current node
        is_initial = (self._level_initial_hash is None) or is_reset
        if is_initial:
            self._level_initial_hash = state_hash
        is_new_state = state_hash not in self.graph.nodes
        self.graph.add_or_update_node(state_hash, available_actions, is_initial=is_initial)

        # Player detection and direction inference (goal-directed navigation)
        self._detect_player(components, is_reset)
        if self._last_action is not None and self._last_action != 0:
            self._infer_action_direction(self._last_action, components)
        self._detect_targets(components, is_reset)

        # Update stagnation counter
        # Key insight: discovering a NEW state is progress, not stagnation!
        if level_changed or is_reset:
            self._stagnation = 0
        elif is_new_state:
            self._stagnation = 0  # New state = progress
        else:
            self._stagnation += 1  # Revisited state = no progress

        # Invalidate cached path if state changed unexpectedly
        if level_changed or is_reset:
            self._current_path = []
            self._current_path_idx = 0

        # Select action using hierarchical strategy with goal-directed navigation
        action = self._select_action(state_hash, available_actions, components)

        # Record state for next iteration
        self._prev_hash = state_hash
        self._prev_components = components
        self._prev_state = state
        self._levels_completed = levels_completed
        self._last_action = action
        self._action_history.append(action)

        # Progress logging
        if self._step % 50 == 0 or level_changed:
            target_info = ""
            if self._player_pos:
                target_info = f" | Player: ({self._player_pos[0]:.0f},{self._player_pos[1]:.0f})"
                if self._action_directions:
                    target_info += f" | Dirs: {self._action_directions}"
                if self._target_candidates:
                    target_info += f" | Targets: {len(self._target_candidates)}"
                if self._current_target:
                    target_info += f" | Target: ({self._current_target[0]:.0f},{self._current_target[1]:.0f})"
            print(
                f"  Step {self._step:4d} | Action: {action} | "
                f"Levels: {levels_completed}/{latest_frame.win_levels} | "
                f"Graph: {self.graph.node_count}N/{self.graph.edge_count}E | "
                f"Stagnation: {self._stagnation}{target_info}"
            )

        return self._int_to_action(action)

    def _detect_reset(self, state_hash: str, state, level_changed: bool) -> bool:
        """Detect if a reset occurred (silent life loss).

        In some games, step counter exhaustion silently resets the player
        position without triggering GAME_OVER. This is detected by checking
        if we returned to the initial state hash without a level change.

        Args:
            state_hash: Current state hash.
            state: Current game state.
            level_changed: Whether a level transition occurred.

        Returns:
            True if a reset is detected.
        """
        if state_hash == self._level_initial_hash and not level_changed:
            if self._last_action is not None and self._last_action != 0:
                if self._prev_hash is not None and self._prev_hash != state_hash:
                    return True
        return False

    def _detect_player(self, components: list[Component], is_reset: bool) -> None:
        """Detect the player component by tracking frame-to-frame movement.

        Strategy:
            1. FIRST try frame-to-frame movement detection (most reliable)
            2. If movement found, update player color/area/position
            3. If no movement, track known player color
            4. Fallback: heuristic (medium-sized component)

        Args:
            components: Current frame components.
            is_reset: Whether a reset just occurred.
        """
        if is_reset:
            self._prev_player_pos = None
            self._player_pos = None
            return

        # Step 1: Try frame-to-frame movement detection (highest priority)
        if self._prev_components:
            best_movement = None
            best_move_dist = float('inf')

            for curr in components:
                if curr.area < 3 or curr.area > 100:
                    continue
                # Skip edge components (likely status bar)
                if curr.bbox[0] >= 59 or curr.bbox[2] <= 2:
                    continue
                for prev in self._prev_components:
                    if prev.color != curr.color:
                        continue
                    if abs(prev.area - curr.area) > 3:
                        continue
                    if prev.centroid == curr.centroid:
                        continue

                    dr = abs(curr.centroid[0] - prev.centroid[0])
                    dc = abs(curr.centroid[1] - prev.centroid[1])
                    total = dr + dc

                    # Only accept small movements (player moves 1-5 pixels/step)
                    if 0.5 < total <= 15:
                        # Prefer movements of known player color
                        if self._player_color is not None and curr.color == self._player_color:
                            self._player_area = curr.area
                            self._prev_player_pos = prev.centroid
                            self._player_pos = curr.centroid
                            return
                        # Track best alternative movement
                        if total < best_move_dist:
                            best_move_dist = total
                            best_movement = (curr, prev)

            # Use best movement if found (and either no known color or color changed)
            if best_movement:
                curr, prev = best_movement
                self._player_color = curr.color
                self._player_area = curr.area
                self._prev_player_pos = prev.centroid
                self._player_pos = curr.centroid
                return

        # Step 2: Track known player color (no movement detected)
        if self._player_color is not None:
            best_match = None
            best_area_diff = float('inf')
            for c in components:
                if c.color == self._player_color and c.area > 0:
                    ref_area = self._player_area if self._player_area > 0 else c.area
                    diff = abs(c.area - ref_area)
                    if diff < best_area_diff:
                        best_area_diff = diff
                        best_match = c
            if best_match:
                self._prev_player_pos = self._player_pos
                self._player_pos = best_match.centroid
                self._player_area = best_match.area
                return

        # Step 3: Heuristic fallback (first frame only)
        if self._player_color is None and components:
            candidates = [
                c for c in components
                if 3 <= c.area <= 50
                and c.bbox[0] < 59 and c.bbox[2] > 2  # Not status bar
            ]
            if candidates:
                candidates.sort(key=lambda c: c.area)
                mid = candidates[len(candidates) // 2]
                self._player_color = mid.color
                self._player_area = mid.area
                self._player_pos = mid.centroid

    def _infer_action_direction(self, action: int, components: list[Component]) -> None:
        """Infer the direction (dr, dc) caused by each action using voting.

        Records all observed displacements for each action and uses the
        most frequent (mode) as the inferred direction. This is robust
        to occasional mis-detections.

        Args:
            action: The action that was just taken.
            components: Current frame components (after the action).
        """
        if self._player_pos is None or self._prev_player_pos is None:
            return

        dr = round(self._player_pos[0] - self._prev_player_pos[0])
        dc = round(self._player_pos[1] - self._prev_player_pos[1])

        total = abs(dr) + abs(dc)
        if total == 0:
            return  # No movement

        # Filter out abnormally large displacements (likely detection errors)
        if total > 15:
            return

        # Record in history
        if action not in self._action_dir_history:
            self._action_dir_history[action] = []
        self._action_dir_history[action].append((dr, dc))

        # Use majority vote (mode) for stable direction inference
        history = self._action_dir_history[action]
        if len(history) >= 1:
            from collections import Counter
            counts = Counter(history)
            most_common_dir = counts.most_common(1)[0][0]
            self._action_directions[action] = most_common_dir

    def _detect_targets(self, components: list[Component], is_reset: bool) -> None:
        """Detect candidate target components (goals, switchers, refills).

        Targets are small, non-edge, non-player components that are likely
        interactive game objects.

        Args:
            components: Current frame components.
            is_reset: Whether a reset just occurred.
        """
        if is_reset:
            self._target_candidates.clear()

        if not components:
            return

        max_area = components[0].area  # Largest component (likely wall/floor)

        candidates: list[tuple[float, float, int]] = []
        for c in components:
            if c.color == self._player_color:
                continue
            if c.area > max_area * 0.5:
                continue  # Large = wall/floor
            if c.area < 2 or c.area > 50:
                continue
            if c.is_edge and c.area > 20:
                continue  # Edge status bar
            candidates.append((c.centroid[0], c.centroid[1], c.color))

        self._target_candidates = candidates

    def _select_best_target(self) -> Optional[tuple[float, float]]:
        """Select the best target to navigate toward.

        Prefers unvisited targets, then nearest target.

        Returns:
            Target position as (row, col), or None if no candidates.
        """
        if not self._target_candidates or self._player_pos is None:
            return None

        # Filter out visited and failed targets
        unvisited = [
            t for t in self._target_candidates
            if (round(t[0]), round(t[1])) not in self._visited_targets
            and (round(t[0]), round(t[1])) not in self._failed_targets
        ]

        pool = unvisited if unvisited else self._target_candidates

        # Select nearest target
        best = min(
            pool,
            key=lambda t: abs(t[0] - self._player_pos[0])
            + abs(t[1] - self._player_pos[1]),
        )
        return (best[0], best[1])

    def _direction_to_action(
        self, available_actions: list[int], target: tuple[float, float]
    ) -> Optional[int]:
        """Select the action that moves the player toward the target.

        Uses the learned action-direction mapping to pick the action
        whose direction best aligns with the target direction.

        Args:
            available_actions: Actions available in the current state.
            target: Target position as (row, col).

        Returns:
            Best action ID, or None if no suitable action found.
        """
        if not self._player_pos or not self._action_directions:
            return None

        dr = target[0] - self._player_pos[0]
        dc = target[1] - self._player_pos[1]

        # If very close, any direction is fine
        if abs(dr) < 1 and abs(dc) < 1:
            return None

        best_action = None
        best_score = 0.0

        for action in available_actions:
            if action not in self._action_directions:
                continue
            adr, adc = self._action_directions[action]
            # Dot product: direction alignment
            score = adr * dr + adc * dc
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _select_action(
        self, state_hash: str, available_actions: list[int],
        components: Optional[list[Component]] = None,
    ) -> int:
        """Hierarchical action selection with goal-directed navigation.

        Level 0: Untested actions (prefer those toward target).
        Level 1: Goal-directed navigation (if direction mapping known).
        Level 2: Navigate to nearest frontier via BFS.
        Level 3: Random safe action (avoid known reset actions).

        Args:
            state_hash: Hash of the current state.
            available_actions: Actions available in the current state.
            components: Current frame components (for target detection).

        Returns:
            Selected action ID.
        """
        node = self.graph.nodes.get(state_hash)
        if node is None:
            return random.choice(available_actions) if available_actions else 1

        # Find untested actions
        untested = [
            a for a in available_actions
            if a not in node.actions or not node.actions[a].tested
        ]

        # Level 0: Try untested actions, preferring those toward target
        if untested:
            self._current_path = []
            self._current_path_idx = 0
            # If we know directions and have a target, prefer goal-directed
            if self._action_directions and self._player_pos:
                target = self._select_best_target()
                if target:
                    best = self._direction_to_action(untested, target)
                    if best is not None:
                        return best
            return random.choice(untested)

        # Level 1: Goal-directed navigation (all actions tested but know target)
        if self._action_directions and self._player_pos:
            target = self._select_best_target()
            if target:
                # Mark current target as visited if we're close
                dist = abs(target[0] - self._player_pos[0]) + abs(target[1] - self._player_pos[1])
                if dist < 3:
                    self._visited_targets.add((round(target[0]), round(target[1])))
                    # Pick a new target
                    target = self._select_best_target()
                if target:
                    best = self._direction_to_action(available_actions, target)
                    if best is not None:
                        return best

        # Level 2: Navigate to nearest frontier via BFS
        if (
            not self._current_path
            or self._current_path_idx >= len(self._current_path)
        ):
            path = self.graph.bfs_to_nearest_frontier(state_hash)
            if path is not None:
                self._current_path = path
                self._current_path_idx = 0

        if self._current_path and self._current_path_idx < len(self._current_path):
            action = self._current_path[self._current_path_idx]
            self._current_path_idx += 1
            if action in available_actions:
                return action
            self._current_path = []
            self._current_path_idx = 0

        # Level 3: Random safe action (avoid known reset actions)
        safe_actions = [
            a for a in available_actions
            if a not in node.actions or not node.actions[a].causes_reset
        ]
        if safe_actions:
            return random.choice(safe_actions)

        return random.choice(available_actions) if available_actions else 1

    def _on_level_complete(self, new_levels: int) -> None:
        """Handle level completion event.

        Resets per-level state (graph, frame processor) while preserving
        cross-level mechanism memory.

        Args:
            new_levels: New total levels completed.
        """
        level_steps = self._step - self._level_start_step
        self._level_step_counts.append(level_steps)

        old_level = self._levels_completed
        baseline = (
            self.level_baselines[old_level]
            if old_level < len(self.level_baselines)
            else "?"
        )
        print(
            f"    [LEVEL UP] Level {old_level} -> {new_levels} "
            f"in {level_steps} steps (baseline: {baseline})"
        )

        # Reset per-level state
        self._level_start_step = self._step
        self._level_initial_hash = None
        self.graph.reset()
        self.frame_processor.reset_level()
        self.mechanism_memory.reset_level()
        self.mechanism_memory.levels_seen = new_levels
        self._current_path = []
        self._current_path_idx = 0
        self._stagnation = 0

        # Reset player tracking and targets for new level
        self._player_color = None
        self._player_pos = None
        self._prev_player_pos = None
        self._target_candidates.clear()
        self._visited_targets.clear()
        self._failed_targets.clear()
        # Keep action_directions (same game, same controls)

    def _int_to_action(self, action_id: int) -> 'GameAction':
        """Convert integer action ID to GameAction enum.

        Args:
            action_id: Integer action ID (0=RESET, 1-7=ACTION1-7).

        Returns:
            Corresponding GameAction.
        """
        from arcengine import GameAction

        action_map = {
            0: GameAction.RESET,
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }
        return action_map.get(action_id, GameAction.ACTION1)

    def get_stats(self) -> dict:
        """Get agent statistics for reporting.

        Returns:
            Dictionary of agent statistics.
        """
        return {
            "total_actions": self._total_actions,
            "resets": self._resets,
            "levels_completed": self._levels_completed,
            "stagnation": self._stagnation,
            "game_over_count": self._game_over_count,
            "graph_nodes": self.graph.node_count,
            "graph_edges": self.graph.edge_count,
            "level_steps": self._level_step_counts,
        }

    @property
    def memory(self):
        """Compatibility property for test harness that expects agent.memory."""

        class _MemoryCompat:
            def __init__(self, agent: ExplorerAgent) -> None:
                self._agent = agent

            def get_action_sequence(self) -> list[int]:
                return self._agent._action_history

        return _MemoryCompat(self)
