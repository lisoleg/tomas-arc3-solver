"""
DopamineExplorer V4 - RL-driven Agent for ARC-AGI-3.

Architecture: RewardEngine + QLearner + FlowField + DangerMemory +
              OccupancyGrid + MacroLibrary + CircuitBreaker

Key innovations (fusing RL/IRL + Game Industry + CG + Taiyi Theory):
1. Dopamine reward signals (curiosity + progress - danger - step_cost) [RL]
2. Q-learning with epsilon-greedy exploitation [RL]
3. Dijkstra flow field for spatial navigation [Game Industry / RTS pathfinding]
4. Trauma memory: penalize actions leading to GAME_OVER [IRL]
5. Occupancy grid: learn walls from collision feedback [CG/Robotics/SLAM]
6. Success macros: record & replay winning action sequences [Library Learning]
7. Bayesian circuit breaker: abort low-confidence moves [TOMAS-Play L4]

Theoretical basis:
- Alpha-Beta = kappa-gating pre-judgment blocking (DangerMemory filter)
- MCTS UCB = flow-exploration tradeoff (Q-learning epsilon-greedy)
- Value network = pre-solidified EML superedge (Q-table replaces random rollout)
- L4 circuit breaker = Bayesian confidence check before action execution
- Flow field = Dijkstra distance field (standard RTS game pathfinding)

Author: TOMAS Team
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage


# ============================================================================
# Module 1: RewardEngine (Dopamine Engine)
# ============================================================================

class RewardEngine:
    """Computes intrinsic dopamine rewards for each state transition.

    Implements the RL reward signal that drives learning. The reward
    combines curiosity (exploration), progress (goal-directed), danger
    (trauma avoidance), and efficiency (step cost pressure).

    Reward components:
        - Curiosity: +1.0 for discovering a new state
        - Progress: +2.0 * progress_score for moving toward goal
        - Danger: -5.0 * danger_level for dangerous actions
        - Step cost: -0.02 per step (efficiency pressure)
        - Level complete: +10.0
        - Reset penalty: -3.0
        - Discovery bonus: +0.5 for seeing a new object/color
    """

    CURIOSITY_WEIGHT: float = 1.0
    PROGRESS_WEIGHT: float = 2.0
    DANGER_WEIGHT: float = -5.0
    STEP_COST: float = -0.02
    LEVEL_COMPLETE_BONUS: float = 10.0
    RESET_PENALTY: float = -3.0
    DISCOVERY_BONUS: float = 0.5

    def compute_reward(
        self,
        is_new_state: bool,
        level_changed: bool,
        is_reset: bool,
        is_game_over: bool,
        progress_score: float,
        danger_level: float,
        new_discovery: bool,
    ) -> float:
        """Compute the total reward for a transition.

        Args:
            is_new_state: Whether this is a previously unseen state.
            level_changed: Whether a level was completed.
            is_reset: Whether a silent reset occurred.
            is_game_over: Whether GAME_OVER was triggered.
            progress_score: Normalized progress (0-1) based on flow field.
            danger_level: Normalized danger (0-1) from danger memory.
            new_discovery: Whether a new object/color was discovered.

        Returns:
            Total reward value.
        """
        reward = self.STEP_COST
        if is_new_state:
            reward += self.CURIOSITY_WEIGHT
        if level_changed:
            reward += self.LEVEL_COMPLETE_BONUS
        if is_reset:
            reward += self.RESET_PENALTY
        if is_game_over:
            reward += self.RESET_PENALTY * 2.0
        reward += self.PROGRESS_WEIGHT * max(0.0, min(1.0, progress_score))
        reward += self.DANGER_WEIGHT * max(0.0, min(1.0, danger_level))
        if new_discovery:
            reward += self.DISCOVERY_BONUS
        return reward


# ============================================================================
# Module 2: QLearner (Q-Learning with epsilon-greedy)
# ============================================================================

class QLearner:
    """Q-learning with temporal difference updates and epsilon-greedy policy.

    Uses a dictionary-based Q-table mapping (state_hash, action) to Q-value.
    Exploration rate epsilon decays over time to shift from exploration
    to exploitation.

    This replaces MCTS random rollout with a learned value function,
    analogous to AlphaGo's value network as pre-solidified EML superedge.
    """

    INITIAL_EPSILON: float = 1.0
    MIN_EPSILON: float = 0.1
    EPSILON_DECAY: float = 0.998
    LEARNING_RATE: float = 0.15
    DISCOUNT_FACTOR: float = 0.95

    def __init__(self) -> None:
        """Initialize the Q-learner."""
        self._q_table: dict[tuple[str, int], float] = {}
        self._epsilon: float = self.INITIAL_EPSILON
        self._step_count: int = 0

    def get_q(self, state_hash: str, action: int) -> float:
        """Get Q-value for a state-action pair.

        Args:
            state_hash: Hash of the state.
            action: Action ID.

        Returns:
            Q-value, defaulting to 0.0 for unseen pairs.
        """
        return self._q_table.get((state_hash, action), 0.0)

    def get_best_action(self, state_hash: str, available_actions: list[int]) -> int:
        """Get the action with highest Q-value.

        Args:
            state_hash: Hash of the state.
            available_actions: List of available action IDs.

        Returns:
            Best action ID.
        """
        if not available_actions:
            return 1
        return max(available_actions, key=lambda a: self.get_q(state_hash, a))

    def update(
        self,
        state_hash: str,
        action: int,
        reward: float,
        next_state_hash: str,
        next_actions: list[int],
        is_terminal: bool,
    ) -> None:
        """Update Q-value using temporal difference learning.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state_hash: Current state hash.
            action: Action taken.
            reward: Reward received.
            next_state_hash: Next state hash.
            next_actions: Available actions in next state.
            is_terminal: Whether the next state is terminal.
        """
        old_q = self.get_q(state_hash, action)
        if is_terminal:
            target = reward
        else:
            max_next_q = max(
                (self.get_q(next_state_hash, a) for a in next_actions),
                default=0.0,
            )
            target = reward + self.DISCOUNT_FACTOR * max_next_q
        new_q = old_q + self.LEARNING_RATE * (target - old_q)
        self._q_table[(state_hash, action)] = new_q

    def select_epsilon_greedy(
        self,
        state_hash: str,
        available_actions: list[int],
        untested: list[int],
    ) -> Optional[int]:
        """Select action using epsilon-greedy policy.

        Priority:
        1. If exploring AND untested actions exist -> try untested
        2. Else -> exploit best Q-value
        3. Returns None if no actions available

        Args:
            state_hash: Current state hash.
            available_actions: All available actions.
            untested: Untested actions in this state.

        Returns:
            Selected action ID, or None.
        """
        self._step_count += 1
        self._epsilon = max(
            self.MIN_EPSILON,
            self.INITIAL_EPSILON * (self.EPSILON_DECAY ** self._step_count),
        )

        if not available_actions:
            return None

        # Always try untested actions first if epsilon is high
        if untested and (random.random() < self._epsilon or self._step_count < 15):
            return random.choice(untested)

        # Exploit: pick best Q-value
        return max(available_actions, key=lambda a: self.get_q(state_hash, a))

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._epsilon

    @property
    def q_table_size(self) -> int:
        """Number of entries in Q-table."""
        return len(self._q_table)


# ============================================================================
# Module 3: DangerMemory (Trauma Memory - IRL)
# ============================================================================

class DangerMemory:
    """Remembers dangerous states and actions (trauma memory).

    When GAME_OVER occurs, penalizes recent actions with temporal discount.
    This implements the inverse RL idea: learn what NOT to do from failures.

    Also implements the TOMAS-Play Bayesian circuit breaker: if the
    confidence in the best action is too low, the agent should explore
    rather than exploit a potentially hallucinated optimal path.
    """

    MAX_HISTORY: int = 15
    TEMPORAL_DECAY: float = 0.7
    DANGER_THRESHOLD: float = 1.0
    CIRCUIT_BREAKER_THRESHOLD: float = 0.15

    def __init__(self) -> None:
        """Initialize danger memory."""
        self._dangerous_transitions: dict[tuple[str, int], float] = {}
        self._dangerous_positions: dict[tuple[int, int], int] = {}
        self._recent_history: list[tuple[str, int, tuple[float, float]]] = []

    def record_action(
        self,
        state_hash: str,
        action: int,
        player_pos: Optional[tuple[float, float]],
    ) -> None:
        """Record a recent action for potential blame assignment.

        Args:
            state_hash: State hash when action was taken.
            action: Action ID.
            player_pos: Player position when action was taken.
        """
        self._recent_history.append(
            (state_hash, action, player_pos or (0.0, 0.0))
        )
        if len(self._recent_history) > self.MAX_HISTORY:
            self._recent_history.pop(0)

    def record_game_over(self) -> None:
        """Penalize recent actions when GAME_OVER occurs.

        Uses temporal discount: most recent action gets the most blame,
        older actions get exponentially less.
        """
        decay = 1.0
        for s, a, pos in reversed(self._recent_history):
            key = (s, a)
            self._dangerous_transitions[key] = (
                self._dangerous_transitions.get(key, 0.0) + decay
            )
            r, c = int(pos[0]), int(pos[1])
            self._dangerous_positions[(r, c)] = (
                self._dangerous_positions.get((r, c), 0) + 1
            )
            decay *= self.TEMPORAL_DECAY

    def record_reset(self) -> None:
        """Lighter penalty for silent resets."""
        decay = 0.5
        for s, a, pos in reversed(self._recent_history[-5:]):
            key = (s, a)
            self._dangerous_transitions[key] = (
                self._dangerous_transitions.get(key, 0.0) + decay
            )
            decay *= self.TEMPORAL_DECAY

    def get_danger(self, state_hash: str, action: int) -> float:
        """Get danger level for a state-action pair.

        Args:
            state_hash: State hash.
            action: Action ID.

        Returns:
            Danger level (0.0 = safe, higher = more dangerous).
        """
        return self._dangerous_transitions.get((state_hash, action), 0.0)

    def get_position_danger(self, pos: tuple[float, float]) -> float:
        """Get spatial danger level for a position.

        Args:
            pos: Player position (row, col).

        Returns:
            Normalized danger level (0-1).
        """
        r, c = int(pos[0]), int(pos[1])
        count = self._dangerous_positions.get((r, c), 0)
        return min(1.0, count / 3.0)

    def is_dangerous(self, state_hash: str, action: int) -> bool:
        """Check if a state-action pair is dangerous.

        Args:
            state_hash: State hash.
            action: Action ID.

        Returns:
            True if danger level exceeds threshold.
        """
        return self.get_danger(state_hash, action) >= self.DANGER_THRESHOLD

    def filter_safe_actions(
        self, state_hash: str, actions: list[int]
    ) -> list[int]:
        """Filter out dangerous actions, keeping at least one.

        Args:
            state_hash: State hash.
            actions: List of action IDs to filter.

        Returns:
            List of safe action IDs.
        """
        safe = [a for a in actions if not self.is_dangerous(state_hash, a)]
        return safe if safe else actions

    def check_circuit_breaker(
        self,
        state_hash: str,
        available_actions: list[int],
        q_learner: QLearner,
    ) -> bool:
        """Bayesian circuit breaker: check if confidence is too low.

        Implements TOMAS-Play L4 mechanism: if the Q-value spread
        between best and worst actions is too small, the agent's
        confidence in its policy is low, and it should explore rather
        than exploit a potentially hallucinated optimal path.

        Args:
            state_hash: Current state hash.
            available_actions: Available actions.
            q_learner: Q-learner to query values from.

        Returns:
            True if circuit breaker triggers (should explore).
        """
        if len(available_actions) < 2:
            return False

        q_values = [q_learner.get_q(state_hash, a) for a in available_actions]
        spread = max(q_values) - min(q_values)

        # If all Q-values are nearly identical, confidence is low
        return spread < self.CIRCUIT_BREAKER_THRESHOLD

    def clear_history(self) -> None:
        """Clear recent action history (e.g., on level change)."""
        self._recent_history.clear()


# ============================================================================
# Module 4: MacroLibrary (Success Macros - Library Learning)
# ============================================================================

class MacroLibrary:
    """Library of successful action sequences for cross-level transfer.

    When a level is completed, the action sequence is recorded.
    In future levels (same game), the macro is tried first.
    If it works, it saves many exploration steps.

    This implements the Library Learning concept from TOMAS: extract
    reusable action patterns from successful trajectories.
    """

    MAX_MACROS: int = 20

    def __init__(self) -> None:
        """Initialize the macro library."""
        self._macros: list[dict] = []

    def record_success(
        self, actions: list[int], level: int, steps: int, context_hash: str
    ) -> None:
        """Record a successful action sequence.

        Args:
            actions: List of action IDs.
            level: Level index that was completed.
            steps: Number of steps taken.
            context_hash: Hash of the starting context.
        """
        self._macros.append({
            "actions": actions.copy(),
            "level": level,
            "steps": steps,
            "context": context_hash,
            "uses": 0,
            "successes": 0,
        })
        self._macros.sort(key=lambda m: (m["steps"], -m["successes"]))
        if len(self._macros) > self.MAX_MACROS:
            self._macros = self._macros[: self.MAX_MACROS]

    def get_best_macro(self) -> Optional[list[int]]:
        """Get the shortest successful action sequence.

        Returns:
            List of action IDs, or None if no macros.
        """
        if not self._macros:
            return None
        return self._macros[0]["actions"].copy()

    def record_macro_use(self, success: bool) -> None:
        """Record whether a macro replay was successful.

        Args:
            success: Whether the macro replay succeeded.
        """
        if self._macros:
            self._macros[0]["uses"] += 1
            if success:
                self._macros[0]["successes"] += 1

    def has_macros(self) -> bool:
        """Check if any macros are available."""
        return len(self._macros) > 0

    @property
    def macro_count(self) -> int:
        """Number of macros in library."""
        return len(self._macros)


# ============================================================================
# Module 5: SpatialReasoner (Occupancy Grid + Flow Field)
# ============================================================================

class SpatialReasoner:
    """Spatial reasoning module combining occupancy grid and flow field.

    Occupancy grid: learns which cells are passable/blocked from
    movement feedback and component analysis (robotics SLAM approach).

    Flow field: Dijkstra distance field from goals, used for gradient
    descent navigation (standard RTS game pathfinding like Starcraft).

    Works on a grid snapped to MOVE_STEP pixel intervals, matching
    the game's movement grid for efficient pathfinding.
    """

    MOVE_STEP: int = 5
    GRID_RADIUS: int = 15  # 31x31 grid centered on player start

    def __init__(self) -> None:
        """Initialize spatial reasoner."""
        self._origin: Optional[tuple[int, int]] = None
        self._walls: set[tuple[int, int]] = set()
        self._goals: set[tuple[int, int]] = set()
        self._visited: set[tuple[int, int]] = set()
        self._flow_field: dict[tuple[int, int], float] = {}
        self._flow_dirty: bool = True
        self._grid_step: int = self.MOVE_STEP

    def set_origin(self, pos: tuple[float, float]) -> None:
        """Set the grid origin (player's starting position).

        Args:
            pos: Player starting position in pixel coordinates.
        """
        self._origin = (int(pos[0]), int(pos[1]))

    def snap(self, pos: tuple[float, float]) -> tuple[int, int]:
        """Snap a pixel position to the grid.

        Args:
            pos: Position in pixel coordinates.

        Returns:
            Snapped grid position as (row, col).
        """
        if self._origin is None:
            return (int(pos[0]), int(pos[1]))
        dr = round((pos[0] - self._origin[0]) / self._grid_step)
        dc = round((pos[1] - self._origin[1]) / self._grid_step)
        return (
            self._origin[0] + dr * self._grid_step,
            self._origin[1] + dc * self._grid_step,
        )

    def add_wall(self, pos: tuple[int, int]) -> None:
        """Mark a grid cell as a wall.

        Args:
            pos: Grid position to mark as blocked.
        """
        self._walls.add(pos)
        self._flow_dirty = True

    def add_wall_from_pixels(self, pixels: frozenset[tuple[int, int]]) -> None:
        """Mark grid cells that contain wall pixels as blocked.

        Only marks cells that actually have wall pixels, not the entire
        bounding box. This prevents false wall detection that blocks
        valid paths.

        Args:
            pixels: Frozen set of (row, col) pixel coordinates of the wall.
        """
        snapped_cells: set[tuple[int, int]] = set()
        for r, c in pixels:
            snapped = self.snap((float(r), float(c)))
            snapped_cells.add(snapped)
        for cell in snapped_cells:
            self._walls.add(cell)
        self._flow_dirty = True

    def add_wall_region(self, bbox: tuple[int, int, int, int]) -> None:
        """Mark a rectangular region as walls.

        Deprecated: prefer add_wall_from_pixels for accuracy.
        Only used for quick approximate wall marking.

        Args:
            bbox: Bounding box (min_row, min_col, max_row, max_col) in pixels.
        """
        min_r, min_c, max_r, max_c = bbox
        # Snap corners to grid
        sr, sc = self.snap((float(min_r), float(min_c)))
        er, ec = self.snap((float(max_r), float(max_c)))
        step = self._grid_step
        r = sr
        while r <= er + step:
            c = sc
            while c <= ec + step:
                self._walls.add((r, c))
                c += step
            r += step
        self._flow_dirty = True

    def add_goal(self, pos: tuple[float, float]) -> None:
        """Add a goal position.

        Args:
            pos: Goal position in pixel coordinates.
        """
        snapped = self.snap(pos)
        if snapped not in self._goals:
            self._goals.add(snapped)
            self._flow_dirty = True

    def mark_visited(self, pos: tuple[int, int]) -> None:
        """Mark a position as visited.

        Args:
            pos: Grid position.
        """
        self._visited.add(pos)

    def compute_flow_field(self) -> None:
        """Compute Dijkstra distances from all goals.

        The flow field maps each reachable grid cell to its distance
        from the nearest goal. The agent navigates by following the
        gradient (decreasing distance).
        """
        self._flow_field.clear()
        if not self._goals:
            self._flow_dirty = False
            return

        # Multi-source BFS from all goals
        queue: deque[tuple[tuple[int, int], float]] = deque()
        for g in self._goals:
            if g not in self._walls:
                self._flow_field[g] = 0.0
                queue.append((g, 0.0))

        step = self._grid_step
        # 8-directional neighbors (4 orthogonal + 4 diagonal)
        directions = [
            (-step, 0, 1.0), (step, 0, 1.0),
            (0, -step, 1.0), (0, step, 1.0),
            (-step, -step, 1.414), (-step, step, 1.414),
            (step, -step, 1.414), (step, step, 1.414),
        ]

        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc, cost in directions:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self._walls:
                    continue
                if (nr, nc) in self._flow_field:
                    continue
                # Check bounds (within reasonable range of origin)
                if self._origin:
                    if abs(nr - self._origin[0]) > self.GRID_RADIUS * step:
                        continue
                    if abs(nc - self._origin[1]) > self.GRID_RADIUS * step:
                        continue
                nd = dist + cost
                self._flow_field[(nr, nc)] = nd
                queue.append(((nr, nc), nd))

        self._flow_dirty = False

    def get_distance(self, pos: tuple[int, int]) -> float:
        """Get flow field distance from pos to nearest goal.

        Args:
            pos: Grid position.

        Returns:
            Distance value, or inf if unreachable.
        """
        if self._flow_dirty:
            self.compute_flow_field()
        return self._flow_field.get(pos, float("inf"))

    def get_direction(self, pos: tuple[int, int]) -> tuple[float, float]:
        """Get the direction to move from pos toward the nearest goal.

        Follows the flow field gradient (steepest distance decrease).

        Args:
            pos: Current grid position.

        Returns:
            Normalized (dr, dc) direction vector.
        """
        if self._flow_dirty:
            self.compute_flow_field()

        if pos not in self._flow_field:
            return (0.0, 0.0)

        current_dist = self._flow_field[pos]
        if current_dist == 0.0:
            return (0.0, 0.0)  # At goal

        step = self._grid_step
        directions = [
            (-step, 0), (step, 0), (0, -step), (0, step),
            (-step, -step), (-step, step), (step, -step), (step, step),
        ]

        best_dr, best_dc = 0.0, 0.0
        best_dist = current_dist

        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            nd = self._flow_field.get((nr, nc), float("inf"))
            if nd < best_dist:
                best_dist = nd
                best_dr, best_dc = float(dr), float(dc)

        mag = (best_dr ** 2 + best_dc ** 2) ** 0.5
        if mag > 0:
            return (best_dr / mag, best_dc / mag)
        return (0.0, 0.0)

    def get_progress_score(
        self, prev_pos: tuple[int, int], curr_pos: tuple[int, int]
    ) -> float:
        """Compute normalized progress score based on distance reduction.

        Args:
            prev_pos: Previous grid position.
            curr_pos: Current grid position.

        Returns:
            Progress score (0-1), 1.0 = moved closer to goal.
        """
        if self._flow_dirty:
            self.compute_flow_field()

        prev_d = self.get_distance(prev_pos)
        curr_d = self.get_distance(curr_pos)

        if prev_d == float("inf"):
            return 0.0
        reduction = prev_d - curr_d
        if reduction <= 0:
            return 0.0
        return min(1.0, reduction / max(prev_d, 1.0))

    @property
    def has_goals(self) -> bool:
        """Whether any goals are known."""
        return len(self._goals) > 0

    @property
    def wall_count(self) -> int:
        """Number of known wall cells."""
        return len(self._walls)

    @property
    def goal_count(self) -> int:
        """Number of known goals."""
        return len(self._goals)

    def reset_level(self) -> None:
        """Reset per-level state (walls, goals, flow field)."""
        self._walls.clear()
        self._goals.clear()
        self._visited.clear()
        self._flow_field.clear()
        self._flow_dirty = True


# ============================================================================
# Module 6: DopamineExplorer (Main Agent)
# ============================================================================

# Reuse Component and FrameProcessor from graph_explorer
from .graph_explorer import Component, FrameProcessor


class DopamineExplorer:
    """RL-driven exploration agent for ARC-AGI-3.

    Combines 6 modules:
    1. RewardEngine - dopamine reward signals [RL]
    2. QLearner - Q-learning with epsilon-greedy [RL]
    3. DangerMemory - trauma memory + circuit breaker [IRL / TOMAS-Play]
    4. MacroLibrary - cross-level success transfer [Library Learning]
    5. SpatialReasoner - occupancy grid + flow field [CG / Game Industry]
    6. Behavior tree - hierarchical action selection [Game Industry]

    The agent operates in phases:
    - Calibration (first ~8 steps): Learn action-direction mapping
    - Navigation (main phase): Flow field guided + Q-learning
    - Exploitation (later levels): Macro replay when available

    Args:
        max_steps: Maximum total steps for the game.
        stagnation_threshold: Steps without progress before giving up.
        level_baselines: Human baseline steps per level.
    """

    # Calibration: try each action once to learn directions
    CALIBRATION_STEPS: int = 8

    # Component size thresholds for wall/goal inference
    WALL_AREA_THRESHOLD: int = 80
    GOAL_MIN_AREA: int = 2
    GOAL_MAX_AREA: int = 50

    # Player detection
    PLAYER_MIN_AREA: int = 3
    PLAYER_MAX_AREA: int = 100

    def __init__(
        self,
        max_steps: int = 2000,
        stagnation_threshold: int = 500,
        level_baselines: Optional[list[int]] = None,
    ) -> None:
        """Initialize the dopamine explorer.

        Args:
            max_steps: Maximum total steps for the game.
            stagnation_threshold: Steps without progress before giving up.
            level_baselines: Human baseline steps per level.
        """
        self.max_steps = max_steps
        self.stagnation_threshold = stagnation_threshold
        self.level_baselines = level_baselines or []

        # Core modules
        self.reward_engine = RewardEngine()
        self.q_learner = QLearner()
        self.danger_memory = DangerMemory()
        self.macro_library = MacroLibrary()
        self.spatial = SpatialReasoner()
        self.frame_processor = FrameProcessor()

        # Step tracking
        self._step: int = 0
        self._stagnation: int = 0
        self._levels_completed: int = 0
        self._game_over_count: int = 0
        self._resets: int = 0
        self._total_actions: int = 0
        self._action_history: list[int] = []

        # Per-level state
        self._level_start_step: int = 0
        self._level_step_counts: list[int] = []
        self._level_action_history: list[int] = []

        # Player tracking
        self._player_pos: Optional[tuple[float, float]] = None
        self._prev_player_pos: Optional[tuple[float, float]] = None
        self._player_color: Optional[int] = None
        self._player_area: int = 0
        self._grid_origin_set: bool = False

        # Action-direction mapping
        self._action_dirs: dict[int, tuple[int, int]] = {}
        self._action_votes: dict[int, Counter] = {}
        self._calibration_actions: list[int] = []
        self._calibration_idx: int = 0
        self._calibration_done: bool = False

        # Previous frame state
        self._prev_state_hash: Optional[str] = None
        self._prev_action: Optional[int] = None
        self._prev_available_actions: list[int] = []
        self._prev_components: list[Component] = []
        self._prev_grid: Optional[np.ndarray] = None

        # Goal tracking
        self._discovered_colors: set[int] = set()
        self._level_initial_hash: Optional[str] = None

        # Macro replay state
        self._macro_replay: Optional[list[int]] = None
        self._macro_replay_idx: int = 0

        # Known wall colors (inferred from large static components)
        self._wall_colors: set[int] = set()

        # Seen state hashes for stagnation tracking
        self._seen_state_hashes: set[str] = set()

        # Anti-stuck: track consecutive non-movement and action repetition
        self._stuck_counter: int = 0
        self._last_action: int = -1
        self._consecutive_same: int = 0
        self._round_robin_idx: int = 0

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

    def choose_action(self, frames: list, latest_frame) -> "GameAction":
        """Choose the next action based on current observation.

        Implements the main agent loop:
        1. Handle GAME_OVER -> RESET + record danger
        2. Detect level change -> save macro, reset per-level state
        3. Process frame -> components, player, walls, goals
        4. Update occupancy grid from movement feedback
        5. Compute reward and update Q-table
        6. Select action via behavior tree

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

        # === Step 1: Handle GAME_OVER ===
        if state == GameState.GAME_OVER:
            self._game_over_count += 1
            self._resets += 1
            self.danger_memory.record_game_over()
            self.danger_memory.clear_history()
            self._action_history.append(0)
            self._level_action_history.append(0)
            self._prev_action = 0
            self._prev_state_hash = None
            self._prev_components = []
            self._prev_player_pos = None
            self._player_pos = None
            self._macro_replay = None
            self._calibration_done = False
            self._calibration_idx = 0
            self._stuck_counter = 0
            self._consecutive_same = 0
            self._round_robin_idx = 0
            self._stagnation = 0
            print(
                f"    [GAME_OVER] step={self._step} sending RESET "
                f"(count={self._game_over_count})"
            )
            return GameAction.RESET

        # === Step 2: Detect level change ===
        level_changed = levels_completed > self._levels_completed
        if level_changed:
            self._on_level_complete(levels_completed)

        # === Step 3: Process frame ===
        state_hash, components, masked = self.frame_processor.process(grid)

        # Detect reset (returned to initial state)
        is_reset = self._detect_reset(state_hash, level_changed)
        if is_reset:
            self.danger_memory.record_reset()
            self._player_pos = None
            self._prev_player_pos = None
            self._macro_replay = None
            self._calibration_done = False
            self._calibration_idx = 0

        # Set grid origin on first frame
        if not self._grid_origin_set:
            self._detect_player_initial(components)
            if self._player_pos is not None:
                self.spatial.set_origin(self._player_pos)
                self._grid_origin_set = True
                self._level_initial_hash = state_hash
        else:
            # Normal player detection
            self._detect_player(components, is_reset)

        # Infer walls from large components
        self._infer_walls(components)

        # Infer goals from small components
        self._infer_goals(components)

        # === Step 4: Update occupancy grid from movement ===
        if self._prev_player_pos is not None and self._player_pos is not None:
            self._update_occupancy_from_movement()

        # === Step 5: Compute reward and update Q-table ===
        if self._prev_state_hash is not None and self._prev_action is not None:
            self._update_q_learning(
                state_hash, components, is_reset, level_changed,
                state == GameState.GAME_OVER,
            )

        # Record danger memory
        if self._player_pos is not None and self._prev_action is not None:
            self.danger_memory.record_action(
                state_hash, self._prev_action, self._player_pos
            )

        # Infer action direction from movement
        if self._prev_action is not None and self._prev_action != 0:
            self._infer_action_direction(self._prev_action)

        # === Step 6: Update stagnation ===
        is_new_state = state_hash not in self._seen_state_hashes
        self._seen_state_hashes.add(state_hash)
        if level_changed or is_reset:
            self._stagnation = 0
        elif self._player_pos and self._prev_player_pos:
            moved = (
                abs(self._player_pos[0] - self._prev_player_pos[0]) > 0.5
                or abs(self._player_pos[1] - self._prev_player_pos[1]) > 0.5
            )
            if moved:
                self._stagnation = 0
            else:
                self._stagnation += 1
        else:
            self._stagnation += 1

        # === Step 7: Select action via behavior tree ===
        action = self._select_action(state_hash, available_actions, components)

        # Record state for next iteration
        self._prev_state_hash = state_hash
        self._prev_action = action
        self._prev_available_actions = available_actions
        self._prev_components = components
        self._prev_grid = grid.copy()
        self._levels_completed = levels_completed
        self._action_history.append(action)
        self._level_action_history.append(action)

        # Progress logging
        if self._step % 50 == 0 or level_changed or self._step <= 10:
            self._log_progress(levels_completed, latest_frame, action)

        return self._int_to_action(action)

    def _detect_player_initial(self, components: list[Component]) -> None:
        """Detect player on the first frame using heuristics.

        Strategy: look for a medium-sized component near the center
        of the frame, not touching edges.

        Args:
            components: Components from the first frame.
        """
        if not components:
            return

        candidates = [
            c for c in components
            if self.PLAYER_MIN_AREA <= c.area <= self.PLAYER_MAX_AREA
            and c.bbox[0] < 59 and c.bbox[2] > 2
            and c.bbox[1] < 59 and c.bbox[3] > 2
        ]
        if candidates:
            # Prefer components closest to center
            candidates.sort(
                key=lambda c: abs(c.centroid[0] - 32) + abs(c.centroid[1] - 32)
            )
            best = candidates[0]
            self._player_color = best.color
            self._player_area = best.area
            self._player_pos = best.centroid
            self._prev_player_pos = best.centroid

    def _detect_player(self, components: list[Component], is_reset: bool) -> None:
        """Detect the player component using frame differencing as primary method.

        Strategy:
        1. Frame differencing (highest priority, most reliable)
           - Compute pixel-level diff between current and previous frame
           - The player is the component that moved (appears in diff)
        2. Known color tracking (fallback)
        3. Heuristic (first frame only)

        Args:
            components: Current frame components.
            is_reset: Whether a reset just occurred.
        """
        if is_reset:
            self._prev_player_pos = None
            self._player_pos = None
            return

        # Step 1: Frame differencing (highest priority)
        if self._prev_grid is not None and self._player_pos is not None:
            curr_grid = self.frame_processor._frame_history[-1] if self.frame_processor._frame_history else None
            if curr_grid is not None:
                new_pos = self._find_player_by_diff(curr_grid, self._prev_grid)
                if new_pos is not None:
                    self._prev_player_pos = self._player_pos
                    self._player_pos = new_pos
                    return

        # Step 2: Frame-to-frame movement detection via components
        if self._prev_components:
            best_movement = None
            best_move_dist = float("inf")

            for curr in components:
                if curr.area < self.PLAYER_MIN_AREA or curr.area > self.PLAYER_MAX_AREA:
                    continue
                if curr.bbox[0] >= 59 or curr.bbox[2] <= 2:
                    continue
                if curr.bbox[1] >= 59 or curr.bbox[3] <= 2:
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

                    if 0.5 < total <= 15:
                        if (
                            self._player_color is not None
                            and curr.color == self._player_color
                        ):
                            self._player_area = curr.area
                            self._prev_player_pos = prev.centroid
                            self._player_pos = curr.centroid
                            return
                        if total < best_move_dist:
                            best_move_dist = total
                            best_movement = (curr, prev)

            if best_movement:
                curr, prev = best_movement
                self._player_color = curr.color
                self._player_area = curr.area
                self._prev_player_pos = prev.centroid
                self._player_pos = curr.centroid
                return

        # Step 3: Track known player color
        if self._player_color is not None:
            best_match = None
            best_area_diff = float("inf")
            for c in components:
                if c.color == self._player_color and c.area > 0:
                    ref_area = (
                        self._player_area if self._player_area > 0 else c.area
                    )
                    diff = abs(c.area - ref_area)
                    if diff < best_area_diff:
                        best_area_diff = diff
                        best_match = c
            if best_match:
                self._prev_player_pos = self._player_pos
                self._player_pos = best_match.centroid
                self._player_area = best_match.area
                return

    def _find_player_by_diff(
        self, curr_grid: np.ndarray, prev_grid: np.ndarray
    ) -> Optional[tuple[float, float]]:
        """Find player position using frame differencing.

        Computes the pixel-level difference between consecutive frames.
        The player's new position is where new pixels appeared, and the
        old position is where pixels disappeared. This is more reliable
        than component matching for multi-color sprites.

        Args:
            curr_grid: Current frame grid (64x64).
            prev_grid: Previous frame grid (64x64).

        Returns:
            New player position as (row, col), or None if no movement detected.
        """
        # Compute pixel differences
        diff = curr_grid != prev_grid

        if not diff.any():
            return None  # No change at all

        # Find changed regions
        changed_rows = np.where(diff.any(axis=1))[0]
        changed_cols = np.where(diff.any(axis=0))[0]

        if len(changed_rows) == 0 or len(changed_cols) == 0:
            return None

        # Filter out status bar regions (top and bottom 3 rows change frequently)
        min_r = max(3, changed_rows.min())
        max_r = min(60, changed_rows.max())
        min_c = max(0, changed_cols.min())
        max_c = min(63, changed_cols.max())

        if min_r > max_r or min_c > max_c:
            return None

        # Extract the diff region (non-status-bar)
        diff_region = diff[min_r:max_r + 1, min_c:max_c + 1]
        if not diff_region.any():
            return None

        # Find connected components in the diff
        labeled, num_features = ndimage.label(diff_region)
        if num_features == 0:
            return None

        # Find the largest changed cluster (likely player movement)
        largest_label = 0
        largest_size = 0
        for i in range(1, num_features + 1):
            size = (labeled == i).sum()
            if size > largest_size:
                largest_size = size
                largest_label = i

        if largest_label == 0 or largest_size < 3:
            return None

        # Get centroid of the largest changed cluster
        coords = np.argwhere(labeled == largest_label)
        # Adjust coordinates back to full frame
        centroid_r = float(coords[:, 0].mean()) + min_r
        centroid_c = float(coords[:, 1].mean()) + min_c

        # Validate: the new position should be close to the previous position
        # (player moves at most ~5 pixels per step)
        if self._player_pos is not None:
            dr = abs(centroid_r - self._player_pos[0])
            dc = abs(centroid_c - self._player_pos[1])
            if dr + dc > 20:  # Too far, probably not the player
                return None

        return (centroid_r, centroid_c)

    def _infer_walls(self, components: list[Component]) -> None:
        """Infer wall positions from large static components.

        Large same-color connected components (>WALL_AREA_THRESHOLD)
        are likely walls. Only the actual pixels are marked as blocked
        in the occupancy grid (not the entire bounding box).

        Args:
            components: Current frame components.
        """
        for c in components:
            if c.area >= self.WALL_AREA_THRESHOLD:
                self._wall_colors.add(c.color)
                self.spatial.add_wall_from_pixels(c.pixels)

    def _infer_goals(self, components: list[Component]) -> None:
        """Infer goal positions from small components.

        More selective than before: only components that are:
        - Small (2-30 pixels, tighter filter)
        - Non-edge (not touching frame border)
        - Non-wall color
        - Non-player color
        - At least 10 pixels from player (not adjacent)

        are added as goals. This reduces false positives that confuse
        the flow field.

        Args:
            components: Current frame components.
        """
        if self._player_pos is None:
            return

        for c in components:
            # Skip player
            if c.color == self._player_color:
                continue
            # Skip walls
            if c.color in self._wall_colors:
                continue
            # Tighter size filter
            if c.area < self.GOAL_MIN_AREA or c.area > 30:
                continue
            # Skip edge components
            if c.is_edge:
                continue
            # Skip components too close to player (likely not a goal)
            dist = abs(c.centroid[0] - self._player_pos[0]) + abs(
                c.centroid[1] - self._player_pos[1]
            )
            if dist < 10:
                continue
            # Add as goal
            self.spatial.add_goal(c.centroid)

            # Track discovered colors
            if c.color not in self._discovered_colors:
                self._discovered_colors.add(c.color)

    def _update_occupancy_from_movement(self) -> None:
        """Update occupancy grid based on player movement feedback.

        If the player moved, both old and new positions are passable.
        If the player tried to move but stayed, there's a wall in
        that direction.
        """
        if self._player_pos is None or self._prev_player_pos is None:
            return

        moved = (
            abs(self._player_pos[0] - self._prev_player_pos[0]) > 0.5
            or abs(self._player_pos[1] - self._prev_player_pos[1]) > 0.5
        )

        if moved:
            self._stuck_counter = 0
        else:
            self._stuck_counter += 1

        if not moved and self._prev_action is not None and self._prev_action != 0:
            # Player didn't move despite taking a movement action
            # -> wall in the direction of the action
            action_dir = self._action_dirs.get(self._prev_action)
            if action_dir:
                dr, dc = action_dir
                # Snap wall position to grid
                prev_snapped = self.spatial.snap(self._prev_player_pos)
                wall_r = prev_snapped[0] + dr
                wall_c = prev_snapped[1] + dc
                self.spatial.add_wall((wall_r, wall_c))

    def _infer_action_direction(self, action: int) -> None:
        """Infer the direction (dr, dc) caused by each action using voting.

        Records all observed displacements for each action and uses
        the most frequent (mode) as the inferred direction.

        Args:
            action: The action that was just taken.
        """
        if self._player_pos is None or self._prev_player_pos is None:
            return

        dr = round(self._player_pos[0] - self._prev_player_pos[0])
        dc = round(self._player_pos[1] - self._prev_player_pos[1])

        total = abs(dr) + abs(dc)
        if total == 0 or total > 15:
            return  # No movement or abnormal displacement

        if action not in self._action_votes:
            self._action_votes[action] = Counter()
        self._action_votes[action][(dr, dc)] += 1

        # Use majority vote for stable direction
        most_common = self._action_votes[action].most_common(1)[0][0]
        self._action_dirs[action] = most_common

        # Check if calibration is complete
        if not self._calibration_done:
            move_actions = sum(
                1 for v in self._action_votes.values() if v.most_common(1)[0][0] != (0, 0)
            )
            if move_actions >= min(4, len(self._action_votes)):
                self._calibration_done = True

    def _update_q_learning(
        self,
        state_hash: str,
        components: list[Component],
        is_reset: bool,
        level_changed: bool,
        is_game_over: bool,
    ) -> None:
        """Compute reward and update Q-table.

        Args:
            state_hash: Current state hash.
            components: Current frame components.
            is_reset: Whether a reset occurred.
            level_changed: Whether a level was completed.
            is_game_over: Whether GAME_OVER occurred.
        """
        if self._prev_state_hash is None or self._prev_action is None:
            return

        # Compute state hash for Q-learning
        prev_q_state = self._get_q_state_hash()
        curr_q_state = self._get_q_state_hash_from_pos(self._player_pos)

        # Compute progress score from flow field
        progress_score = 0.0
        if self._player_pos and self._prev_player_pos:
            prev_snapped = self.spatial.snap(self._prev_player_pos)
            curr_snapped = self.spatial.snap(self._player_pos)
            progress_score = self.spatial.get_progress_score(
                prev_snapped, curr_snapped
            )

        # Compute danger level
        danger_level = 0.0
        if self._prev_action is not None:
            danger_level = min(
                1.0,
                self.danger_memory.get_danger(prev_q_state, self._prev_action)
                / 3.0,
            )

        # Check for new discoveries
        new_discovery = any(
            c.color not in self._discovered_colors for c in components
        )

        # Compute reward
        is_new_state = state_hash != self._prev_state_hash
        reward = self.reward_engine.compute_reward(
            is_new_state=is_new_state,
            level_changed=level_changed,
            is_reset=is_reset,
            is_game_over=is_game_over,
            progress_score=progress_score,
            danger_level=danger_level,
            new_discovery=new_discovery,
        )

        # Update Q-table
        next_actions = self._prev_available_actions
        self.q_learner.update(
            state_hash=prev_q_state,
            action=self._prev_action,
            reward=reward,
            next_state_hash=curr_q_state,
            next_actions=next_actions,
            is_terminal=is_game_over,
        )

    def _get_q_state_hash(self) -> str:
        """Get Q-learning state hash from previous player position.

        Returns:
            State hash string for Q-table lookup.
        """
        return self._get_q_state_hash_from_pos(self._prev_player_pos)

    def _get_q_state_hash_from_pos(
        self, pos: Optional[tuple[float, float]]
    ) -> str:
        """Get Q-learning state hash from a position.

        The state is (snapped_position, level) which gives a compact,
        meaningful representation for Q-learning.

        Args:
            pos: Player position.

        Returns:
            State hash string.
        """
        if pos is None:
            return "unknown"
        snapped = self.spatial.snap(pos)
        return f"{snapped[0]}_{snapped[1]}_L{self._levels_completed}"

    def _detect_reset(self, state_hash: str, level_changed: bool) -> bool:
        """Detect if a silent reset occurred.

        Args:
            state_hash: Current state hash.
            level_changed: Whether a level transition occurred.

        Returns:
            True if a reset is detected.
        """
        if (
            state_hash == self._level_initial_hash
            and not level_changed
            and self._prev_action is not None
            and self._prev_action != 0
            and self._prev_state_hash is not None
            and self._prev_state_hash != state_hash
        ):
            return True
        return False

    def _select_action(
        self,
        state_hash: str,
        available_actions: list[int],
        components: list[Component],
    ) -> int:
        """Select action via hierarchical behavior tree with anti-stuck.

        Behavior tree priority:
        0. GAME_OVER handled -> RESET
        1. Anti-stuck escape (if stuck for 3+ steps, force different action)
        2. Macro replay (if available and applicable)
        3. Calibration (if not done): try untested actions
        4. Flow field navigation (if goals known and directions calibrated)
        5. Q-learning epsilon-greedy (with danger filter)
        6. Random safe action (fallback)

        Args:
            state_hash: Current state hash.
            available_actions: Available action IDs.
            components: Current frame components.

        Returns:
            Selected action ID.
        """
        if not available_actions:
            return 1

        # Track action repetition
        if self._prev_action == self._last_action:
            self._consecutive_same += 1
        else:
            self._consecutive_same = 0
        self._last_action = self._prev_action or -1

        # === Level 1: Anti-stuck escape ===
        # If player hasn't moved for 3+ steps, force a different action
        if self._stuck_counter >= 3:
            # Exclude the action that's been failing
            escape_actions = [
                a for a in available_actions
                if a != 0 and a != self._prev_action
            ]
            if escape_actions:
                # Round-robin through escape actions
                action = escape_actions[self._round_robin_idx % len(escape_actions)]
                self._round_robin_idx += 1
                return action

        # Anti-repetition: if same action 5+ times, penalize by trying different
        if self._consecutive_same >= 5:
            different = [
                a for a in available_actions
                if a != 0 and a != self._prev_action
            ]
            if different:
                return random.choice(different)

        # === Level 2: Macro replay ===
        if self._macro_replay is not None:
            if self._macro_replay_idx < len(self._macro_replay):
                action = self._macro_replay[self._macro_replay_idx]
                self._macro_replay_idx += 1
                if action in available_actions:
                    return action
            self._macro_replay = None

        # Try macro from library if starting a new level
        if (
            self._step > 1
            and self.macro_library.has_macros()
            and self._macro_replay is None
            and self._stagnation == 0
            and self._step - self._level_start_step <= 1
        ):
            macro = self.macro_library.get_best_macro()
            if macro:
                self._macro_replay = macro
                self._macro_replay_idx = 0
                self.macro_library.record_macro_use(False)
                if macro[0] in available_actions:
                    self._macro_replay_idx = 1
                    return macro[0]

        # === Level 3: Calibration ===
        if not self._calibration_done and self._player_pos is not None:
            tested = set(self._action_votes.keys())
            untested = [a for a in available_actions if a not in tested and a != 0]
            if untested:
                return untested[0]

        # === Level 4: Flow field navigation ===
        if (
            self._calibration_done
            and self.spatial.has_goals
            and self._player_pos is not None
            and self._action_dirs
        ):
            action = self._flow_field_navigate(available_actions)
            if action is not None:
                # Apply danger filter
                q_state = self._get_q_state_hash_from_pos(self._player_pos)
                safe = self.danger_memory.filter_safe_actions(
                    q_state,
                    [action] + [a for a in available_actions if a != action],
                )
                if safe:
                    return safe[0]

        # === Level 5: Q-learning epsilon-greedy ===
        q_state = self._get_q_state_hash_from_pos(self._player_pos)
        untested = [
            a for a in available_actions
            if (q_state, a) not in self.q_learner._q_table
        ]

        # Apply danger filter first
        safe_actions = self.danger_memory.filter_safe_actions(
            q_state, available_actions
        )
        safe_untested = [a for a in untested if a in safe_actions]

        q_action = self.q_learner.select_epsilon_greedy(
            q_state, safe_actions, safe_untested
        )
        if q_action is not None:
            return q_action

        # === Level 6: Flow field fallback ===
        if self.spatial.has_goals and self._player_pos and self._action_dirs:
            action = self._flow_field_navigate(available_actions)
            if action is not None:
                return action

        # === Level 7: Random safe action ===
        return random.choice(safe_actions) if safe_actions else random.choice(available_actions)

    def _flow_field_navigate(self, available_actions: list[int]) -> Optional[int]:
        """Navigate using the flow field (Dijkstra gradient descent).

        Selects the action whose learned direction best aligns with
        the flow field gradient (steepest distance decrease toward goal).

        Args:
            available_actions: Available action IDs.

        Returns:
            Best action ID for navigation, or None.
        """
        if self._player_pos is None:
            return None

        snapped = self.spatial.snap(self._player_pos)
        direction = self.spatial.get_direction(snapped)

        if direction == (0.0, 0.0):
            return None

        # Find the action whose direction best matches the flow field gradient
        best_action = None
        best_score = -float("inf")

        for action in available_actions:
            if action == 0 or action not in self._action_dirs:
                continue
            adr, adc = self._action_dirs[action]
            # Normalize action direction
            mag = (adr ** 2 + adc ** 2) ** 0.5
            if mag == 0:
                continue
            nadr, nadc = adr / mag, adc / mag
            # Dot product: alignment between action direction and flow gradient
            # Note: flow direction points toward goal, so we want positive dot product
            score = nadr * direction[0] + nadc * direction[1]
            if score > best_score:
                best_score = score
                best_action = action

        # Only navigate if the best action has positive alignment
        if best_score > 0.1:
            return best_action

        return None

    def _on_level_complete(self, new_levels: int) -> None:
        """Handle level completion event.

        Saves the successful action sequence as a macro, then resets
        per-level state while preserving cross-level memory (Q-table,
        danger memory, macro library).

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

        # Save successful action sequence as macro
        if self._level_action_history:
            context_hash = self._level_initial_hash or "unknown"
            self.macro_library.record_success(
                actions=self._level_action_history,
                level=old_level,
                steps=level_steps,
                context_hash=context_hash,
            )

        # Reset per-level state
        self._level_start_step = self._step
        self._level_initial_hash = None
        self._level_action_history = []
        self._seen_state_hashes.clear()
        self._stuck_counter = 0
        self._consecutive_same = 0
        self._round_robin_idx = 0
        self.frame_processor.reset_level()
        self.spatial.reset_level()
        self.danger_memory.clear_history()
        self._stagnation = 0

        # Reset player tracking
        self._player_pos = None
        self._prev_player_pos = None
        self._player_color = None
        self._player_area = 0
        self._grid_origin_set = False

        # Reset calibration for new level
        self._calibration_done = False
        self._calibration_idx = 0

        # Reset macro replay
        self._macro_replay = None
        self._macro_replay_idx = 0

        # Keep action_dirs (same game, same controls)
        # Keep Q-table (cross-level knowledge)
        # Keep danger_memory (cross-level trauma)

    def _int_to_action(self, action_id: int) -> "GameAction":
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

    def _log_progress(self, levels_completed: int, latest_frame, action: int) -> None:
        """Log progress information.

        Args:
            levels_completed: Number of levels completed.
            latest_frame: Latest frame observation.
            action: Selected action.
        """
        player_info = ""
        if self._player_pos:
            player_info = f" | Player: ({self._player_pos[0]:.0f},{self._player_pos[1]:.0f})"
            if self._action_dirs:
                player_info += f" | Dirs: {len(self._action_dirs)}"
            if self.spatial.has_goals:
                player_info += f" | Goals: {self.spatial.goal_count}"
            if self.spatial.wall_count:
                player_info += f" | Walls: {self.spatial.wall_count}"
            snapped = self.spatial.snap(self._player_pos)
            dist = self.spatial.get_distance(snapped)
            if dist < float("inf"):
                player_info += f" | Dist: {dist:.1f}"

        print(
            f"  Step {self._step:4d} | Action: {action} | "
            f"Levels: {levels_completed}/{latest_frame.win_levels} | "
            f"Q: {self.q_learner.q_table_size} | "
            f"eps: {self.q_learner.epsilon:.2f} | "
            f"Stag: {self._stagnation}{player_info}"
        )

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
            "q_table_size": self.q_learner.q_table_size,
            "wall_count": self.spatial.wall_count,
            "goal_count": self.spatial.goal_count,
            "macro_count": self.macro_library.macro_count,
            "action_dirs": len(self._action_dirs),
            "level_steps": self._level_step_counts,
            # Compatibility with test harness expecting graph_nodes/edges
            "graph_nodes": self.q_learner.q_table_size,
            "graph_edges": 0,
        }

    @property
    def memory(self):
        """Compatibility property for test harness.

        Returns:
            Memory-like object with get_action_sequence method.
        """

        class _MemoryCompat:
            def __init__(self, agent: DopamineExplorer) -> None:
                self._agent = agent

            def get_action_sequence(self) -> list[int]:
                return self._agent._action_history

        return _MemoryCompat(self)
