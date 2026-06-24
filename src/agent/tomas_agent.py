"""
TomasAgent - Main ARC-AGI-3 interactive agent.

Implements the ARC-AGI-3 Agent interface:
- is_done(frames, latest_frame) -> bool
- choose_action(frames, latest_frame) -> GameAction

Architecture:
1. Perception: GridParser parses 64x64 grid into objects
   - LS20 player detection: 5x5 sprite (color 12 top, color 9 bottom)
   - Player position = center of 5x5 sprite: (x+2, y+2)
   - Delta-detection fallback: frame-to-frame difference for occluded sprites
   - Switcher detection: multi-frame tracking of player property changes
2. World Model: StateTracker tracks frame-to-frame changes
3. Planner: HeuristicPlanner selects actions via BFS path search
   - BFS with wall checking (from GridParser inferred walls)
   - _block_other_switchers() logic (MNQ Gate hardening)
   - Manhattan distance fallback when BFS fails
   - DFS backtracking mode (L4 fuse recovery)
4. Memory: EpisodeMemory records gameplay for learning
   - Library learning: macro extraction and failure-pattern analysis
5. Reasoning: BayesianFuseGate (L4 observer)
   - RHAE efficiency prediction + dynamic weight balance
   - Circuit-breaker mechanism triggers planner backtracking
"""

from typing import Optional, List
import random
import numpy as np
from collections import deque

from .perception.grid_parser import GridParser, GameObject
from .world_model.state_tracker import StateTracker
from .planner.heuristic_planner import HeuristicPlanner
from .memory.episode_memory import EpisodeMemory
from .reasoning.bayesian_fuse_gate import BayesianFuseGate
from .reasoning.sleep_step import SleepStep


class TomasAgent:
    """TOMAS Interactive Agent for ARC-AGI-3.

    The agent perceives the game grid, tracks state changes, plans actions
    using heuristic rules, and records gameplay for potential learning.

    Attributes:
        perception: GridParser for parsing 64x64 grids.
        world_model: StateTracker for tracking frame history.
        planner: HeuristicPlanner for selecting actions.
        memory: EpisodeMemory for recording gameplay.
        fuse_gate: BayesianFuseGate for L4 meta-reasoning.
    """

    def __init__(
        self,
        max_steps: int = 500,
        stagnation_threshold: int = 80,
        level_baselines: Optional[List[int]] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            max_steps: Maximum number of steps before the agent gives up.
            stagnation_threshold: Steps without progress before termination.
            level_baselines: Per-level baseline step counts. If provided,
                _level_baseline and _level_hard_cap are set from
                level_baselines[levels_completed] on each level change.
        """
        self.perception: GridParser = GridParser()
        self.world_model: StateTracker = StateTracker(history_size=max_steps)
        self.planner: HeuristicPlanner = HeuristicPlanner(seed=42)
        self.memory: EpisodeMemory = EpisodeMemory(max_steps=max_steps)

        # L4 observer: Bayesian fuse gate for mutual-combat loop.
        self.fuse_gate: BayesianFuseGate = BayesianFuseGate(
            w_logic_init=0.5, fuse_threshold=0.3
        )
        # RHAE baseline step budget per level.
        self._baseline_steps: int = 15

        # L4 Sleep-Step: defensive learning from failure trajectories.
        self.sleep_step: SleepStep = SleepStep()

        # Per-level step limit (ARC-AGI-3: 5 × human_baseline_actions).
        self._level_baselines: Optional[List[int]] = level_baselines
        self._level_baseline: int = 15   # Default; updated from level_baselines.
        self._level_hard_cap: int = 75   # 5 × 15
        if level_baselines and len(level_baselines) > 0:
            self._level_baseline = level_baselines[0]
            self._level_hard_cap = 5 * self._level_baseline
        self._level_steps: int = 0

        # RNG for Explore-then-Commit random safe actions.
        self._rng: random.Random = random.Random(42)

        self._step: int = 0
        self._max_steps: int = max_steps
        self._stagnation_threshold: int = stagnation_threshold
        self._prev_grid: Optional[np.ndarray] = None
        self._prev_player_pos: Optional[tuple[int, int]] = None
        self._prev_player_sprite: Optional[np.ndarray] = None
        self._prev_levels: int = 0
        self._last_action: Optional[int] = None
        self._game_over_count: int = 0
        self._max_game_over_retries: int = 20  # Allow cycling through switchers

        # Simple BFS navigation state (like minimal agent).
        # Local wall set — cleared on level change.
        self._simple_walls: set[tuple[int, int]] = set()
        self._simple_step: int = 5  # MOVE_STEP for LS20
        # Track visited switchers per level.
        self._simple_visited_sw: set[tuple[int, int]] = set()
        # Known switcher positions (accumulated across steps, since
        # switchers are not detected when player is on top of them).
        self._known_switchers: set[tuple[int, int]] = set()
        # Track last BFS action for wall detection.
        self._simple_last_bfs_action: Optional[int] = None
        # Previous player top-left position (for movement detection).
        self._prev_simple_player_tl: Optional[tuple[int, int]] = None
        # Goals blocked by attribute mismatch (wrong rot/shape/color).
        # Cleared on switcher visit or level change.
        self._goal_blocked: set[tuple[int, int]] = set()
        # Positions that caused GAME_OVER. Cleared on switcher visit.
        self._fatal_positions: set[tuple[int, int]] = set()
        # Track if previous frame was GAME_OVER (prevents wall clearing
        # on reset-induced position jump).
        self._was_game_over: bool = False
        # Persistent grid-detected walls (color 4 sprites). Updated
        # each step from the grid. Not cleared on switcher visit.
        self._grid_walls: set[tuple[int, int]] = set()
        # Force switcher visit after GAME_OVER. When True, the agent
        # must visit a switcher before trying the goal. Set on
        # GAME_OVER, cleared on switcher visit.
        self._force_switcher_visit: bool = False
        # Target switcher lock: when goal_blocked, the agent locks onto
        # a specific switcher. Only visiting THIS switcher clears
        # goal_blocked. Prevents false positive switcher positions
        # from prematurely clearing goal_blocked.
        self._target_switcher: Optional[tuple[int, int]] = None
        # Discovered walls (permanent): real walls found by collision
        # that are NOT goals. These are never cleared by fallback logic
        # or switcher visits — only on level change. This prevents the
        # agent from repeatedly trying to path through the same wall
        # after dynamic walls are cleared.
        self._discovered_walls: set[tuple[int, int]] = set()

        # Switcher visit counter: tracks how many times the agent has
        # visited each switcher position in the current level. Used to
        # detect excessive cycling (max 4 visits for 4-state rotation).
        # After max_visits, the agent tries a different switcher.
        self._switcher_visit_count: dict[tuple[int, int], int] = {}
        self._max_switcher_visits: int = 6  # Upper bound for bounce count
        # Track switcher visits per level for adaptive bounce.
        # After GAME_OVER, use bounce to do all visits in one trip.
        self._level_switcher_visits: int = 0  # visits in current level (across lives)
        self._level_had_game_over: bool = False  # True after first GAME_OVER in level
        # Track whether the PREVIOUS step completed a level. Used to
        # filter false LIFE_LOSS detections on level transitions.
        # When a level completes, the next step's position jumps to the
        # new level's start — this is NOT life loss.
        self._prev_progress: bool = False
        # Track whether goal was EVER unblocked (no GOAL_FROZEN) after
        # a switcher visit in the current level. Used to determine
        # bounce_target: if goal was never unblocked, try more triggers.
        self._goal_was_ever_unblocked: bool = False
        # CCG: track which rotation_k values were tried and FAILED
        # (GOAL_FROZEN). This allows precise bounce_target calculation
        # without needing to reach the goal to verify.
        # Example: tried_k = {1, 2} means rotation_k=1 and 2 both
        # triggered GOAL_FROZEN. bounce_target = 3 (try k=3 next).
        self._tried_failed_rotations: set[int] = set()

        # Bounce mode: after visiting a switcher, step off and back on
        # to trigger it again without traveling to the goal. This saves
        # ~35 steps per extra switcher visit (2 steps vs 35+ for a
        # round trip to the goal).
        # Uses triangular number sequence: trip N does (N-1) bounces.
        # Trip 1: 0 bounces (1 visit). Trip 2: 1 bounce (3 total).
        # _bounce_mode: True when agent is in step-off-and-on mode.
        # _bounce_phase: 0 = on switcher (step off), 1 = off switcher (step back on).
        # _bounce_dir: direction to step off (1=up, 2=down, 3=left, 4=right).
        # _bounce_switcher: the switcher position being bounced on.
        # _bounce_count: total bounces done (0 to _max_switcher_visits-1).
        self._bounce_mode: bool = False
        self._bounce_phase: int = 0
        self._bounce_dir: int = 0
        self._bounce_switcher: Optional[tuple[int, int]] = None
        self._bounce_count: int = 0
        # Last switcher position where visit counter was incremented.
        # Prevents double-counting when player stays on switcher.
        # Cleared when player leaves switcher, so re-entry counts.
        self._last_visit_pos: Optional[tuple[int, int]] = None
        # Switchers that have been bounced to max and still failed goal.
        # These are skipped in target_switcher selection to avoid loops.
        self._exhausted_switchers: set[tuple[int, int]] = set()

        # Freeze detection: track consecutive non-movement steps.
        # When the player steps on a goal with wrong attributes, the game
        # freezes for 6 steps (akoadfsur=5). During freeze, step() returns
        # early and the player doesn't move. The agent must NOT add walls
        # during this period — they are fake walls caused by the freeze,
        # not real collisions. Only add walls on the FIRST non-movement
        # step (the actual collision); skip on subsequent steps (freeze).
        self._consecutive_no_move: int = 0

        # Refill detection: refills are sprites with color 11 (unique
        # to refills in LS20). When the player steps on a refill, the
        # step counter is reset to max, preventing silent life loss.
        # This is CRITICAL for levels with step_dec > 1 where each life
        # has fewer moves (e.g., Level 1: step_dec=2, 21 moves/life).
        # Without refills, long journeys (switcher→goal) cause silent
        # life loss which resets rotation, making the goal unreachable.
        self._known_refills: set[tuple[int, int]] = set()
        self._consumed_refills: set[tuple[int, int]] = set()
        self._need_refill: bool = False  # Set after bounce_done

        # CCG (Computational Conformal Geometry) rotation tracking.
        # User prediction: using complex structure (i^k) as inductive
        # bias for rotation tasks gives higher RHAE than real-coordinate
        # counting. Each switcher visit: k = (k+1) % 4.
        # k=0: StartRotation, k=1: +90°, k=2: +180°, k=3: +270°.
        # After life loss: k reset to 0 (rotation reset to Start).
        # bounce_target = (goal_rotation_k - current_k) % 4.
        # This gives the EXACT number of switcher visits needed,
        # eliminating the need for GAME_OVER-based learning.
        self._rotation_k: int = 0  # CCG: i^k representation
        self._goal_rotation_k: Optional[int] = None  # Discovered via trial

        # Step counter tracking for refill navigation.
        # Tracks steps since last "reset" (level start, life loss,
        # or refill consumption). When this exceeds ~60% of the
        # estimated step budget, trigger refill navigation.
        # The step budget is estimated from observed life-loss patterns.
        self._steps_since_reset: int = 0
        self._estimated_step_budget: int = 20  # Conservative default
        self._known_goals: set[tuple[int, int]] = set()  # All goals seen

        # Dynamic import of GameAction/GameState (package name may vary)
        self._GameAction = None
        self._GameState = None
        try:
            from arcengine import GameAction, GameState
            self._GameAction = GameAction
            self._GameState = GameState
        except ImportError:
            try:
                from arc_agi import GameAction, GameState
                self._GameAction = GameAction
                self._GameState = GameState
            except ImportError:
                pass

    # ------------------------------------------------------------------
    # Simple BFS navigation (minimal agent strategy)
    # ------------------------------------------------------------------

    def _find_player_simple(self, grid: np.ndarray) -> Optional[tuple[int, int]]:
        """Find player's top-left position via 5x5 sprite detection.

        Detects the 5x5 sprite with color 12 (top rows) and color 9
        (bottom rows). Returns the (x, y) top-left coordinate.

        CRITICAL: Excludes sprites containing color 5 (goal color).
        The goal indicator (3x3 multi-color center of goal sprite)
        can contain colors 9 and 12, causing false positive player
        detection. Adding `5 not in colors` prevents this.

        Args:
            grid: 64x64 int8 numpy array.

        Returns:
            (x, y) top-left position of the player sprite, or None.
        """
        h, w = grid.shape
        best_pos = None
        best_score = 0
        for y in range(h - 4):
            for x in range(w - 4):
                block = grid[y:y + 5, x:x + 5]
                # Exclude sprites with color 5 (goal indicator)
                if 5 in block:
                    continue
                top = block[:2]
                bottom = block[2:]
                score_12 = int(np.sum(top == 12))
                score_9 = int(np.sum(bottom == 9))
                if score_12 == 0 or score_9 == 0:
                    continue
                total = score_12 + score_9
                if total > best_score:
                    best_score = total
                    best_pos = (x, y)
        return best_pos

    def _find_sprites_simple(
        self, grid: np.ndarray, player_pos: tuple[int, int]
    ) -> tuple[list, list, set, list]:
        """Extract goals, switchers, walls, and refills aligned to the player's grid.

        Classification rules (based on LS20 game internals):
        - Background: pure color 3 (dark green)
        - Wall: pure color 4 (yellow), 5x5 solid block
        - Player: top 2 rows predominantly color 12, bottom 3 rows
          predominantly color 9 (specific pattern, not just color presence)
        - Goal: predominantly color 5 (≥16 of 25 pixels), not player
        - Switcher: has color 0 or 1, not solid color 1 (push block),
          not predominantly color 5 (goal with indicator)
        - Refill: has color 11 (≥2 pixels), unique to refills in LS20.
          Refills reset the step counter when the player steps on them.
        - Color switcher has {9,14,0,8,12} — must NOT be confused
          with player (player has structured 12-top/9-bottom pattern)

        Args:
            grid: 64x64 int8 numpy array.
            player_pos: (x, y) top-left position of the player sprite.

        Returns:
            Tuple of (goals, switchers, walls, refills) where goals,
            switchers, and refills are lists of (x, y) positions,
            walls is a set of (x, y).
        """
        step = self._simple_step
        offset_x = player_pos[0] % step
        offset_y = player_pos[1] % step

        goals: list = []
        switchers: list = []
        walls: set = set()
        refills: list = []

        for y in range(offset_y, 64, step):
            for x in range(offset_x, 64, step):
                if y + 5 > 64 or x + 5 > 64:
                    continue
                sprite = grid[y:y + 5, x:x + 5]
                colors = set(int(c) for c in sprite.flatten())
                pos = (x, y)

                # Skip background (pure color 3 or {3,0})
                if colors <= {3, 0}:
                    continue

                # Compute color counts once for all classification.
                count_4 = int(np.sum(sprite == 4))
                count_5 = int(np.sum(sprite == 5))
                count_0 = int(np.sum(sprite == 0))
                count_1 = int(np.sum(sprite == 1))
                count_11 = int(np.sum(sprite == 11))

                # Refill: has color 11 (≥2 pixels). Color 11 is unique
                # to refills in LS20 — no other sprite type uses it.
                # Must check BEFORE wall/switcher classification because
                # refills have background color 3 which could leak into
                # wall detection edge cases.
                if count_11 >= 2:
                    refills.append(pos)
                    continue

                # Wall: predominantly color 4 (≥20 of 25 pixels).
                # Lowered from ==25 to >=20 to catch partial walls
                # (edges that have a few non-4 pixels from adjacent
                # sprites). Excludes goals (color 5 ≥10) and switchers
                # (color 0/1 ≥2 pixels) to prevent false positives.
                if count_4 >= 20 and count_5 < 10 and not (count_0 >= 2 or count_1 >= 2):
                    walls.add(pos)
                    continue

                # Skip player: use pattern-based detection (top 2 rows
                # predominantly 12, bottom 3 predominantly 9). This
                # prevents color switchers (which have 12 and 9 but in
                # a different pattern) from being misidentified.
                # CRITICAL: Also exclude sprites with color 5 (goal
                # indicator) — the 3x3 goal indicator can contain
                # colors 9 and 12, matching the player pattern.
                top_rows = sprite[:2]
                bottom_rows = sprite[2:]
                top_12 = int(np.sum(top_rows == 12))
                bottom_9 = int(np.sum(bottom_rows == 9))
                if top_12 >= 4 and bottom_9 >= 6 and 5 not in colors:
                    continue  # This is the player

                # Goal: predominantly color 5 (≥16 of 25 pixels).
                # count_5 already computed above.
                if count_5 >= 16:
                    goals.append(pos)
                    continue

                # Switcher: has color 0 (at least 2 pixels) or color 1
                # (at least 2 pixels), but not solid color 1 (push block).
                # count_0, count_1 already computed above.
                if (count_0 >= 2 or count_1 >= 2) and colors != {1}:
                    switchers.append(pos)
                    continue

                # Push block: solid color 1 → treat as wall
                if colors == {1}:
                    walls.add(pos)
                    continue

        return goals, switchers, walls, refills

    def _bfs_simple(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
        walls: set[tuple[int, int]],
        avoid: Optional[set[tuple[int, int]]] = None,
    ) -> Optional[list[int]]:
        """BFS from start to target, avoiding walls and avoid positions.

        Args:
            start: (x, y) start position (top-left aligned).
            target: (x, y) target position (top-left aligned).
            walls: Set of (x, y) wall positions.
            avoid: Optional set of positions to avoid routing through
                (e.g., other goals, fatal positions). The target itself
                is always reachable even if in avoid.

        Returns:
            List of action IDs (1=up, 2=down, 3=left, 4=right) or None.
        """
        if start == target:
            return []

        avoid = avoid or set()
        step = self._simple_step
        # Action ID -> (dx, dy) in pixels
        dirs = [
            (0, -step, 1),   # UP (ACTION1)
            (0, step, 2),    # DOWN (ACTION2)
            (-step, 0, 3),   # LEFT (ACTION3)
            (step, 0, 4),    # RIGHT (ACTION4)
        ]

        q: deque = deque([(start, [])])
        visited = {start}

        while q:
            (cx, cy), path = q.popleft()

            for dx, dy, action_id in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) not in visited:
                    is_target = (nx == target[0] and ny == target[1])
                    # Walls block unless target
                    if (nx, ny) in walls and not is_target:
                        continue
                    # Avoid positions block unless target
                    if (nx, ny) in avoid and not is_target:
                        continue
                    if is_target:
                        return path + [action_id]
                    visited.add((nx, ny))
                    q.append(((nx, ny), path + [action_id]))

        return None

    def _simple_navigate(
        self,
        grid: np.ndarray,
        available_actions: list[int],
    ) -> int:
        """Navigate using BFS with switcher-cycling strategy.

        Strategy:
        1. Find player, goals, switchers, and walls (from grid).
        2. Use grid-detected walls as base, dynamic walls as supplement.
        3. If player is on a switcher: clear goal_blocked + dynamic walls
           (attributes changed, goal may be reachable now).
        4. If goal is NOT blocked: navigate to goal.
        5. If goal IS blocked: navigate to nearest switcher (for cycling).
           The agent naturally alternates: switcher → goal → blocked →
           switcher → goal → success!
        6. BFS avoids routing through other goals and fatal positions.
        7. Fallback: clear dynamic walls and retry, then Manhattan.

        Args:
            grid: 64x64 int8 numpy array.
            available_actions: List of available action IDs.

        Returns:
            Action ID (int).
        """
        # Find player using simple detection
        player_tl = self._find_player_simple(grid)
        if player_tl is None:
            return self._rng.choice(available_actions) if available_actions else 1

        # Find goals, switchers, walls, and refills (from grid)
        goals, switchers, grid_walls, refills = self._find_sprites_simple(grid, player_tl)
        all_goals_set: set = set(goals)

        # Update persistent grid walls (detected from color 4 sprites)
        self._grid_walls = grid_walls

        # Accumulate known switcher positions
        for sw in switchers:
            self._known_switchers.add(sw)

        # Accumulate known goal positions (for LIFE_LOSS goal_blocked setting)
        for g in goals:
            self._known_goals.add(g)

        # Accumulate known refill positions and detect consumed refills.
        # Refills are consumed when the player steps on them (they
        # disappear from the grid). We detect consumption by checking
        # if a previously known refill is no longer detected.
        for r in refills:
            self._known_refills.add(r)
        # Check for consumed refills: if a known refill is not in the
        # current refills list AND the player is near it, it was consumed.
        newly_consumed = self._known_refills - set(refills) - self._consumed_refills
        for rc in newly_consumed:
            # Only mark as consumed if player is at or near the refill
            # position (within 1 step). Otherwise the refill might have
            # been consumed by a life-loss reset, not by the player.
            dist = abs(rc[0] - player_tl[0]) + abs(rc[1] - player_tl[1])
            if dist <= self._simple_step:
                self._consumed_refills.add(rc)
                self._need_refill = False  # Refill visited, clear flag
                self._steps_since_reset = 0  # Reset step counter
                print(f"    [REFILL_CONSUMED] step={self._step} pos={rc} "
                      f"step_counter reset -> steps_since_reset=0")

        # Unified switcher clearing: when player is on ANY known
        # switcher (regardless of target lock), clear goal_blocked +
        # simple_walls + force_visit. This handles two cases:
        # 1. Player reached the LOCKED target switcher → clear everything.
        # 2. Player is on a known switcher without a lock (e.g., after
        #    target was cleared by choose_action's movement check) →
        #    still clear goal_blocked so agent retries the goal.
        # Without this unified clear, the elif branch would NOT clear
        # goal_blocked, causing the agent to oscillate between
        # switcher and goal indefinitely.
        # Save goal_blocked state BEFORE clearing — needed for bounce
        # decision. If goal was blocked when we arrived at the switcher,
        # we need to bounce (cycle attributes) before trying the goal.
        goal_was_blocked = bool(self._goal_blocked)

        # Pre-compute values needed by both the switcher-clearing block
        # and the adaptive-bounce logic below. Previously these were
        # defined inside the if-block, causing UnboundLocalError when
        # the player was NOT on a switcher.
        is_new_visit = (player_tl != self._last_visit_pos)
        # CCG-based adaptive bounce target calculation.
        # i^k rotation tracking: each switcher visit increments k by 1.
        # GoalRotation = i^g for some unknown g ∈ {1,2,3}.
        # _tried_failed_rotations tracks which k values caused GOAL_FROZEN.
        # bounce_target = smallest k NOT in _tried_failed_rotations.
        # This gives the EXACT number of triggers needed, without
        # needing to reach the goal to verify.
        #
        # Example: Level 1 needs 3 triggers (rot_diff=3, g=3).
        #   First life: k=1 → GOAL_FROZEN, k=2 → GOAL_FROZEN
        #   _tried_failed_rotations = {1, 2}
        #   bounce_target = 3 (smallest k not in {1,2})
        #   Second life: 1 arrival + 2 bounces = 3 triggers → k=3 → goal!
        if self._level_had_game_over:
            # Find smallest k in {1,2,3} not in tried_failed_rotations.
            # This is the target rotation_k for the next life.
            target_k = 1
            while target_k in self._tried_failed_rotations and target_k <= 3:
                target_k += 1
            # bounce_target = target_k - 1 (because 1 arrival counts as 1 trigger)
            bounce_target = max(0, target_k - 1)
        else:
            bounce_target = 0

        on_target = (self._target_switcher is not None
                     and player_tl == self._target_switcher)
        on_any_sw = (self._target_switcher is None
                     and player_tl in self._known_switchers)
        if on_target or on_any_sw:
            # is_new_visit was pre-computed above (before this block)
            if is_new_visit:
                self._switcher_visit_count[player_tl] = \
                    self._switcher_visit_count.get(player_tl, 0) + 1
                self._last_visit_pos = player_tl
                # CCG rotation tracking: each new switcher visit
                # increments rotation by 90° (multiply by i).
                # i^k where k = (k+1) % 4.
                self._rotation_k = (self._rotation_k + 1) % 4
                print(f"    [CCG_ROT] step={self._step} rotation_k={self._rotation_k} "
                      f"(i^{self._rotation_k} = {['0°','90°','180°','270°'][self._rotation_k]})")
                # Reset bounce count when visiting a DIFFERENT switcher
                # (allows bouncing on each switcher independently)
                if (self._bounce_switcher is not None
                        and player_tl != self._bounce_switcher):
                    self._bounce_count = 0
                    self._bounce_switcher = None
            visit_n = self._switcher_visit_count.get(player_tl, 0)
            # DIAGNOSTIC: extract player sprite to detect rotation
            ps = grid[player_tl[1]:player_tl[1]+5, player_tl[0]:player_tl[0]+5]
            # Compact: top row colors and left col colors (rotation fingerprint)
            top_row = ps[0].tolist() if ps.shape[0] > 0 else []
            left_col = ps[:, 0].tolist() if ps.shape[1] > 0 else []
            if on_target:
                print(f"    [SW_CLEAR_T] step={self._step} player={player_tl} "
                      f"target_sw={self._target_switcher} visit#{visit_n} "
                      f"goal_was_blocked={goal_was_blocked} -> CLEARING (target)")
            else:
                print(f"    [SW_CLEAR_A] step={self._step} player={player_tl} "
                      f"visit#{visit_n} goal_was_blocked={goal_was_blocked} -> CLEARING (any) "
                      f"top={top_row} left={left_col}")
            self._simple_visited_sw.add(player_tl)
            self._simple_walls.clear()
            # Clear goal_blocked when:
            # - Not in bounce mode (first life: try goal after each visit)
            # - After GAME_OVER: current rotation_k is NOT in failed_rot
            #   (meaning this rotation might be the target — try the goal)
            # FIX 7: Replace bounce_count >= bounce_target with direct
            # rotation_k check. This is more precise because:
            # 1. After GOAL_FROZEN at k=2, agent returns to switcher
            #    and arrives at k=3. k=3 is NOT in failed_rot={0,1,2},
            #    so goal_blocked is cleared immediately — no need for
            #    4 extra wraparound bounces (k=0,1,2,3).
            # 2. The old check (bounce_count >= bounce_target) relied on
            #    a calculated bounce_target that could be wrong when
            #    failed_rot is incomplete.
            if (not self._level_had_game_over
                    or self._rotation_k not in self._tried_failed_rotations):
                # If goal was blocked and we're in bounce mode (after
                # life loss), set _need_refill for the long journey to
                # goal. Without this, the agent goes directly to goal
                # and may run out of steps (LIFE_LOSS).
                if self._level_had_game_over and self._goal_blocked:
                    self._need_refill = True
                self._goal_blocked.clear()
            self._fatal_positions.clear()
            self._force_switcher_visit = False
            self._target_switcher = None
        else:
            # Player NOT on switcher: clear last visit position so
            # next switcher arrival counts as a new visit.
            self._last_visit_pos = None

        # --- Bounce mode: continuous rapid switcher cycling ---
        # When goal is blocked, the agent bounces on the switcher
        # (step off + step on = 2 steps per bounce) to cycle through
        # all attribute states WITHOUT traveling to the goal between
        # bounces. This saves ~35 steps per avoided round trip.
        #
        # Strategy: bounce up to _max_switcher_visits-1 times (covering
        # all possible attribute states: 4 rot × 6 shape × 4 color = 96
        # max, but typically only 1 attribute type per switcher so 3-5
        # bounces suffice). Then go to goal ONCE.
        #
        # Phase 0: step OFF switcher (return bounce_dir action)
        # Phase 1: step BACK ON switcher (return reverse action)
        # After phase 1: increment bounce_count, check if more needed
        if self._bounce_mode and self._bounce_phase == 1:
            # Player just stepped back on switcher (clearing fired above)
            self._bounce_mode = False
            self._bounce_phase = 0
            self._bounce_count += 1
            # CRITICAL: When all bounces are done, clear goal_blocked so
            # the agent navigates to the goal instead of back to switcher.
            # The switcher-clearing block above runs BEFORE BOUNCE_DONE
            # increments the count, so it uses the OLD count and may not
            # clear goal_blocked on the final bounce.
            if self._bounce_count >= bounce_target:
                # CCG insight: check if current rotation_k is already
                # in _tried_failed_rotations. If so, going to goal at
                # this rotation would waste a round trip (it will
                # trigger GOAL_FROZEN again). Instead, extend
                # bounce_target to keep cycling until we reach a
                # rotation NOT in failed_rot.
                # Example: After 2 bounces, k=0 (wraparound). If 0 is
                # already in failed_rot (tried in a previous life),
                # extend bounce_target to 3 and bounce once more (k=1).
                # Continue until k is NOT in failed_rot.
                if self._rotation_k in self._tried_failed_rotations:
                    bounce_target = self._bounce_count + 1
                    print(f"    [BOUNCE_EXT] step={self._step} rot_k={self._rotation_k} "
                          f"in failed_rot={self._tried_failed_rotations} "
                          f"-> extending bounce_target to {bounce_target}")
                else:
                    self._goal_blocked.clear()
                    self._need_refill = True  # Visit refill before long journey to goal
                    print(f"    [BOUNCE_DONE] step={self._step} count={self._bounce_count}/{bounce_target} "
                          f"rot_k={self._rotation_k} -> GOAL UNBLOCKED")
            else:
                print(f"    [BOUNCE_DONE] step={self._step} count={self._bounce_count}/{bounce_target}")

        # Check if should bounce (player on switcher, not already bouncing,
        # and goal was blocked when we arrived — meaning we need more
        # attribute cycling before trying the goal).
        # ADAPTIVE BOUNCE: Only bounce after GAME_OVER in this level.
        # First life: natural cycling (1 trigger per visit, try goal each time).
        # After GAME_OVER: bounce _level_switcher_visits times to do all
        # triggers in one trip, then go to goal once.
        # This adapts to the required number of triggers without knowing
        # it in advance.
        if (on_target or on_any_sw) and is_new_visit and not self._level_had_game_over:
            self._level_switcher_visits += 1
        # bounce_target was pre-computed above (before switcher-clearing block)
        # FIX 7: Use bool(self._goal_blocked) instead of goal_was_blocked.
        # With Fix 7, the switcher clearing logic may have already cleared
        # goal_blocked (if rotation_k is not in failed_rot). Using the
        # current state ensures we don't bounce when the goal is already
        # unblocked.
        need_bounce = (bool(self._goal_blocked)
                       and self._level_had_game_over
                       and self._bounce_count < bounce_target)
        if (on_target or on_any_sw) and not self._bounce_mode and need_bounce:
            if self._bounce_count < self._max_switcher_visits - 1:
                # Need more bounces: find safe direction to step off
                step = self._simple_step
                known_walls = self._grid_walls | self._discovered_walls
                bounce_dirs = [
                    (0, -step, 1),   # UP
                    (0, step, 2),    # DOWN
                    (-step, 0, 3),   # LEFT
                    (step, 0, 4),    # RIGHT
                ]
                for dx, dy, act_id in bounce_dirs:
                    nx, ny = player_tl[0] + dx, player_tl[1] + dy
                    if not (0 <= nx < 64 and 0 <= ny < 64):
                        continue
                    if (nx, ny) in known_walls:
                        continue
                    if (nx, ny) in all_goals_set:
                        continue
                    # Safe direction found: enter bounce mode
                    self._bounce_mode = True
                    self._bounce_phase = 0
                    self._bounce_dir = act_id
                    self._bounce_switcher = player_tl
                    print(f"    [BOUNCE_OFF] step={self._step} dir={act_id} sw={player_tl} "
                          f"bounce={self._bounce_count + 1}/{bounce_target}")
                    if act_id in available_actions:
                        return act_id
                    break
                # No safe direction: skip remaining bounces, go to goal
                self._bounce_count = bounce_target  # force done
                self._goal_blocked.clear()
                self._need_refill = True  # Visit refill before long journey to goal
                print(f"    [BOUNCE_SKIP] step={self._step} no safe dir, going to goal")
            else:
                # All bounces done: clear goal_blocked and go to goal
                self._goal_blocked.clear()
                self._need_refill = True  # Visit refill before long journey to goal
                print(f"    [BOUNCE_END] step={self._step} bounces={self._bounce_count}/{bounce_target} -> goal")

        # Handle bounce phase: if in bounce mode, execute step off/on
        if self._bounce_mode:
            if self._bounce_phase == 0:
                # Player stepped off (or trying to). Advance to phase 1.
                self._bounce_phase = 1
                # Return reverse action to step back on
                reverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
                reverse_action = reverse_map.get(self._bounce_dir, 2)
                print(f"    [BOUNCE_ON] step={self._step} dir={reverse_action} sw={self._bounce_switcher}")
                if reverse_action in available_actions:
                    return reverse_action
                # Fallback: exit bounce mode
                self._bounce_mode = False

        # Prune dynamic walls if too many
        if len(self._simple_walls) > 15:
            self._simple_walls.clear()

        # Detect goal-blocked: any goal in dynamic walls means we tried
        # to step on it but couldn't (attribute mismatch — goal has
        # collision when player's rot/shape/color don't match).
        # Set _force_switcher_visit and lock target switcher to ensure
        # agent cycles to the NEAREST real switcher.
        for g in goals:
            if g in self._simple_walls and g not in self._goal_blocked:
                self._goal_blocked.add(g)
                self._force_switcher_visit = True
                # DIAGNOSTIC: print when goal_blocked is set
                print(f"    [GOAL_BLOCKED_SET] step={self._step} goal={g} "
                      f"simple_walls={self._simple_walls} "
                      f"player={player_tl}")
                # Lock onto nearest known switcher that is NOT a wall
                # and hasn't been visited too many times.
                # _known_switchers may contain false positives (positions
                # that were detected as switchers in one frame but are
                # actually walls). Filter them out using grid_walls.
                # Also filter out switchers that have been visited more
                # than _max_switcher_visits times (excessive cycling).
                # If we just bounced on a switcher and goal is STILL
                # blocked, mark that switcher as exhausted (wrong type).
                # Only check exhaustion after adaptive bounce (post-GAME_OVER)
                if (self._level_had_game_over
                        and self._bounce_count >= bounce_target
                        and bounce_target > 0):
                    # Find the switcher we were bouncing on
                    last_sw = self._bounce_switcher
                    if last_sw is None:
                        # Use the nearest known switcher as fallback
                        last_sw = player_tl if player_tl in self._known_switchers else None
                    if last_sw is not None:
                        self._exhausted_switchers.add(last_sw)
                        self._bounce_count = 0  # Reset for next switcher
                        # Try one more visit next time
                        self._level_switcher_visits += 1
                        print(f"    [SW_EXHAUSTED] step={self._step} sw={last_sw} "
                              f"-> trying different switcher (next target={self._level_switcher_visits})")
                valid_switchers = {
                    s for s in self._known_switchers
                    if s not in self._grid_walls
                    and s not in self._discovered_walls
                    and s not in self._exhausted_switchers
                    and self._switcher_visit_count.get(s, 0) < self._max_switcher_visits
                }
                if not valid_switchers:
                    # All switchers exhausted: reset everything and retry
                    self._switcher_visit_count.clear()
                    self._exhausted_switchers.clear()
                    valid_switchers = {
                        s for s in self._known_switchers
                        if s not in self._grid_walls and s not in self._discovered_walls
                    }
                if valid_switchers:
                    self._target_switcher = min(
                        valid_switchers,
                        key=lambda s: (
                            # Primary: fewer visits first (try each switcher
                            # in round-robin to cycle all attributes)
                            self._switcher_visit_count.get(s, 0),
                            # Secondary: nearest distance
                            abs(s[0] - player_tl[0]) + abs(s[1] - player_tl[1])
                        )
                    )

        # Also mark fatal positions as goal-blocked
        for fp in list(self._fatal_positions):
            if fp in all_goals_set:
                self._goal_blocked.add(fp)

        # Combined wall set for BFS: grid walls + dynamic walls + discovered walls
        # _discovered_walls are permanent (real walls found by collision,
        # not goals). They prevent BFS from routing through known walls.
        combined_walls = self._grid_walls | self._simple_walls | self._discovered_walls

        # Build waypoints based on goal-blocked state and force_visit:
        # - If _force_switcher_visit: MUST go to switcher first (after
        #   GAME_OVER, agent needs to cycle attributes before trying goal)
        # - If goal is blocked: go to nearest switcher (for cycling)
        # - If goal is not blocked: go to goal
        # - If on switcher and goal not blocked: go to goal (leave switcher)
        unblocked_goals = [g for g in goals if g not in self._goal_blocked]

        if self._target_switcher is not None:
            # Locked onto a specific switcher: navigate directly to it.
            # BUT skip if the target is actually a wall (false positive
            # in _known_switchers that wasn't filtered out).
            if self._target_switcher in combined_walls:
                self._target_switcher = None
                # Fall through to next branch
            else:
                waypoints = [self._target_switcher]
        if self._target_switcher is None:
            # Refill navigation: after bounce_done, visit a refill before
            # going to goal. This resets the step counter, preventing
            # silent life loss during the long journey from switcher to
            # goal. Without this, levels with step_dec > 1 (e.g., Level 1:
            # step_dec=2, 21 moves/life) cause the agent to run out of
            # steps mid-journey, silently resetting rotation and making
            # the goal unreachable.
            available_refills = [
                r for r in self._known_refills
                if r not in self._consumed_refills
                and r not in combined_walls
            ]
            # Step counter-based refill navigation (De Sitter exclusion
            # insight: prevent "logic singularity" from step exhaustion).
            # When steps_since_reset exceeds 60% of estimated budget,
            # trigger refill navigation — BUT ONLY after the first life
            # loss (when _level_had_game_over=True). During the first
            # life's natural circulation, urgent refill interrupts
            # goal attempts, preventing collection of failed rotations
            # (CCG insight: each GOAL_FROZEN records a rotation_k,
            # making bounce_target more precise for subsequent lives).
            step_budget_ratio = (
                self._steps_since_reset / max(1, self._estimated_step_budget)
            )
            need_refill_urgent = (
                step_budget_ratio > 0.6
                and available_refills
                and not self._force_switcher_visit
                and not self._bounce_mode
                and self._level_had_game_over  # FIX 3: only after first life loss
                and unblocked_goals  # FIX 5: only when goals are unblocked
                # When all goals are blocked, the agent is heading to a
                # switcher — don't divert to a refill. The refill detour
                # can waste 20+ steps, causing LIFE_LOSS before reaching
                # the switcher. After BOUNCE_DONE, goals are unblocked
                # and _need_refill handles the refill navigation.
            )
            if (self._need_refill or need_refill_urgent) and available_refills \
                    and not self._force_switcher_visit:
                # After bounce: go to the refill that minimizes the
                # TOTAL journey: player→refill + refill→goal.
                # Previously: only considered refill→goal distance,
                # which picked refills far from the switcher (e.g.,
                # Level 1: refill at (14,15) is close to goal but
                # 13 actions from switcher, causing step exhaustion).
                # Now: total distance ensures the agent can actually
                # REACH the refill before step counter exhausts.
                if unblocked_goals:
                    goal_ref = unblocked_goals[0]
                elif goals:
                    goal_ref = goals[0]
                else:
                    goal_ref = player_tl  # fallback
                best_refill = min(
                    available_refills,
                    key=lambda r: (
                        abs(r[0] - player_tl[0]) + abs(r[1] - player_tl[1])
                        + abs(r[0] - goal_ref[0]) + abs(r[1] - goal_ref[1])
                    )
                )
                waypoints = [best_refill]
                print(f"    [REFILL_NAV] step={self._step} target={best_refill} "
                      f"player={player_tl} goal_ref={goal_ref} "
                      f"need_refill={self._need_refill} urgent={need_refill_urgent} "
                      f"steps_since_reset={self._steps_since_reset}/{self._estimated_step_budget}")
            elif self._force_switcher_visit and switchers:
                # After GAME_OVER: must visit switcher before trying goal
                waypoints = sorted(
                    switchers,
                    key=lambda s: abs(s[0] - player_tl[0]) + abs(s[1] - player_tl[1])
                )
            elif unblocked_goals:
                # Goal might be reachable: try it first
                waypoints = unblocked_goals
            elif switchers:
                # Goal is blocked: cycle to nearest switcher
                waypoints = sorted(
                    switchers,
                    key=lambda s: abs(s[0] - player_tl[0]) + abs(s[1] - player_tl[1])
                )
            elif goals:
                # No switchers but goals exist (all blocked): try anyway
                waypoints = list(goals)
            else:
                return self._rng.choice(available_actions) if available_actions else 1

        # Build avoid set: other goals + fatal positions
        # (prevent routing through goals to avoid accidental triggering)
        base_avoid = (all_goals_set | self._fatal_positions)
        # When navigating to goal (not to switcher), also avoid all
        # known switcher positions. This prevents BFS from routing
        # through a switcher, which would trigger an unwanted rotation
        # change and re-block the goal. Critical for levels where the
        # player must bounce on a switcher then navigate to goal
        # without crossing any switcher en route.
        going_to_goal = (self._target_switcher is None
                         and not self._force_switcher_visit
                         and bool(unblocked_goals))
        if going_to_goal:
            base_avoid |= self._known_switchers
        # When NOT actively seeking a refill (_need_refill=False), avoid
        # all known refill positions. This prevents the agent from
        # accidentally stepping on a refill and consuming it while
        # navigating to the switcher or goal. Preserving refills is
        # critical: after bounce_done, the agent needs a refill to
        # reset the step counter for the journey to the goal. If the
        # refill was already consumed, the agent will run out of steps
        # and suffer silent life loss (rotation reset).
        if not self._need_refill:
            base_avoid |= (self._known_refills - self._consumed_refills)

        # Try BFS to each waypoint
        for wp in waypoints:
            avoid = base_avoid - {wp}
            path = self._bfs_simple(player_tl, wp, combined_walls, avoid)
            if path and len(path) > 0:
                action = path[0]
                if action in available_actions:
                    self._simple_last_bfs_action = action
                    # DIAGNOSTIC: check if first step goes through a wall
                    if self._step % 10 == 0:
                        step_deltas = {1: (0, -self._simple_step), 2: (0, self._simple_step), 3: (-self._simple_step, 0), 4: (self._simple_step, 0)}
                        dx2, dy2 = step_deltas[action]
                        next_pos = (player_tl[0] + dx2, player_tl[1] + dy2)
                        in_walls = next_pos in combined_walls
                        print(f"    [BFS_OK] step={self._step} player={player_tl} target={wp} "
                              f"action={action} next={next_pos} in_walls={in_walls} "
                              f"path_len={len(path)} path[:5]={path[:5]}")
                    return action
            # If path is empty (already at target), try next waypoint
            if path is not None and len(path) == 0:
                # Already at target but it didn't trigger level complete
                # → this target is blocked, try next
                continue

        # All BFS failed: go straight to direction exploration.
        # DO NOT clear dynamic walls and retry — that causes the agent
        # to repeatedly try to path through the same wall (wall cycling).
        # Discovered walls and goal-blocked walls stay in place. The
        # direction exploration fallback below will find an alternative
        # route around the blocked position.

        # DIAGNOSTIC: print BFS failure details
        if self._step % 10 == 0 and waypoints:
            wp = waypoints[0]
            # Check specific positions
            north = (player_tl[0], player_tl[1] - self._simple_step)
            south = (player_tl[0], player_tl[1] + self._simple_step)
            east = (player_tl[0] + self._simple_step, player_tl[1])
            west = (player_tl[0] - self._simple_step, player_tl[1])
            print(f"    [BFS_FAIL] step={self._step} player={player_tl} target={wp} "
                  f"N={north}{'W' if north in combined_walls else '.'} "
                  f"S={south}{'W' if south in combined_walls else '.'} "
                  f"E={east}{'W' if east in combined_walls else '.'} "
                  f"W={west}{'W' if west in combined_walls else '.'} "
                  f"combined={len(combined_walls)} "
                  f"avoid={base_avoid - {wp}}")

        # Still failed: direction exploration fallback
        # Try each direction that's not a known wall, preferring
        # directions toward the target. This allows the agent to
        # discover new paths when BFS fails (e.g., when there are
        # undetected partial walls blocking the path).
        target = waypoints[0]
        dx = target[0] - player_tl[0]
        dy = target[1] - player_tl[1]
        step = self._simple_step

        # Order directions by preference (toward target first)
        dirs_ordered = []
        if abs(dx) >= abs(dy):
            # Horizontal priority
            if dx > 0:
                dirs_ordered.append((step, 0, 4))   # RIGHT
                dirs_ordered.append((-step, 0, 3))  # LEFT
            else:
                dirs_ordered.append((-step, 0, 3))  # LEFT
                dirs_ordered.append((step, 0, 4))   # RIGHT
            if dy > 0:
                dirs_ordered.append((0, step, 2))   # DOWN
                dirs_ordered.append((0, -step, 1))  # UP
            else:
                dirs_ordered.append((0, -step, 1))  # UP
                dirs_ordered.append((0, step, 2))   # DOWN
        else:
            # Vertical priority
            if dy > 0:
                dirs_ordered.append((0, step, 2))   # DOWN
                dirs_ordered.append((0, -step, 1))  # UP
            else:
                dirs_ordered.append((0, -step, 1))  # UP
                dirs_ordered.append((0, step, 2))   # DOWN
            if dx > 0:
                dirs_ordered.append((step, 0, 4))   # RIGHT
                dirs_ordered.append((-step, 0, 3))  # LEFT
            else:
                dirs_ordered.append((-step, 0, 3))  # LEFT
                dirs_ordered.append((step, 0, 4))   # RIGHT

        for dx2, dy2, act_id in dirs_ordered:
            if act_id not in available_actions:
                continue
            nx, ny = player_tl[0] + dx2, player_tl[1] + dy2
            if not (0 <= nx < 64 and 0 <= ny < 64):
                continue
            # Skip known walls (grid + dynamic)
            if (nx, ny) in combined_walls:
                continue
            # Skip positions to avoid (other goals, fatal positions)
            if (nx, ny) in (base_avoid - {target}):
                continue
            # This direction is not blocked: try it
            return act_id

        # All directions blocked: clear temporary walls (goal blocks,
        # false positives) but keep discovered walls (real walls). This
        # allows the agent to try directions that were blocked by
        # conditional obstacles (like goals) while still avoiding
        # confirmed real walls.
        if self._simple_walls:
            self._simple_walls.clear()
            combined_walls = self._grid_walls | self._discovered_walls
            for dx2, dy2, act_id in dirs_ordered:
                if act_id not in available_actions:
                    continue
                nx, ny = player_tl[0] + dx2, player_tl[1] + dy2
                if not (0 <= nx < 64 and 0 <= ny < 64):
                    continue
                if (nx, ny) in combined_walls:
                    continue
                if (nx, ny) in (base_avoid - {target}):
                    continue
                return act_id

        return available_actions[0] if available_actions else 1

    def _get_grid(self, frame) -> np.ndarray:
        """Extract the 64x64 grid from a FrameDataRaw object.

        Args:
            frame: FrameDataRaw object with a 'frame' attribute.

        Returns:
            64x64 int8 numpy array.
        """
        if hasattr(frame, "frame") and frame.frame:
            return np.array(frame.frame[0], dtype=np.int8)
        return np.zeros((64, 64), dtype=np.int8)

    def _get_game_state(self, frame):
        """Extract the game state from a frame.

        Args:
            frame: FrameDataRaw object.

        Returns:
            GameState enum value or None.
        """
        return getattr(frame, "state", None)

    def _get_levels_completed(self, frame) -> int:
        """Extract the number of completed levels from a frame.

        Args:
            frame: FrameDataRaw object.

        Returns:
            Number of completed levels (int).
        """
        return getattr(frame, "levels_completed", 0)

    def _get_win_levels(self, frame) -> int:
        """Extract the number of win levels from a frame.

        Args:
            frame: FrameDataRaw object.

        Returns:
            Number of win levels (int).
        """
        return getattr(frame, "win_levels", 0)

    def _get_available_actions(self, frame) -> list[int]:
        """Extract available actions from a frame.

        Args:
            frame: FrameDataRaw object.

        Returns:
            List of available action IDs.
        """
        actions = getattr(frame, "available_actions", [])
        if isinstance(actions, list):
            return actions
        return list(actions) if actions is not None else []

    def is_done(self, frames: list, latest_frame) -> bool:
        """Determine if the agent should stop.

        The agent stops when:
        - The game state is WIN.
        - The game state is GAME_OVER and max retries exceeded.
        - The step count exceeds max_steps.
        - The agent is stuck (stagnation threshold exceeded).

        Args:
            frames: List of all frames so far.
            latest_frame: The most recent FrameDataRaw.

        Returns:
            True if the agent should stop, False otherwise.
        """
        game_state = self._get_game_state(latest_frame)

        # Win condition
        if self._GameState and game_state == self._GameState.WIN:
            return True

        # Game over: retry up to max retries
        if self._GameState and game_state == self._GameState.GAME_OVER:
            self._game_over_count += 1
            if self._game_over_count > self._max_game_over_retries:
                return True
            return False

        # Step limit
        if self._step >= self._max_steps:
            return True

        # Per-level hard cap (ARC-AGI-3: 5 × human_baseline_actions).
        if self._level_steps >= self._level_hard_cap:
            return True

        # Stagnation
        if self.world_model.is_stuck(self._stagnation_threshold):
            return True

        return False

    def choose_action(self, frames: list, latest_frame):
        """Choose the next action to take.

        Pipeline:
        1. Extract grid, game state, and available actions from the frame.
        2. Parse the grid to detect player and objects.
        3. Check if the player moved since last step (update planner).
        4. Update world model and episode memory.
        5. Handle GAME_OVER by resetting.
        6. Plan the next action using the heuristic planner.
        7. Convert to GameAction and return.

        Args:
            frames: List of all frames so far.
            latest_frame: The most recent FrameDataRaw.

        Returns:
            GameAction enum value (or int if GameAction unavailable).
        """
        grid: np.ndarray = self._get_grid(latest_frame)
        game_state = self._get_game_state(latest_frame)
        levels_completed: int = self._get_levels_completed(latest_frame)
        available_actions: list[int] = self._get_available_actions(latest_frame)

        if not available_actions:
            available_actions = [1, 2, 3, 4]

        # Parse grid with 5x5 sprite detection
        parsed_grid = self.perception.parse(
            grid, self._prev_grid, available_actions
        )
        player_obj: Optional[GameObject] = parsed_grid.player_object

        # Use player_position from ParsedGrid (includes fallback to prev pos)
        player_pos: Optional[tuple[int, int]] = parsed_grid.player_position
        objects: list = parsed_grid.objects

        # --- Perception correction (Sleep-Step integration) ---
        # Detect position jumps (穿墙误检): if the detected position jumps
        # more than JUMP_THRESHOLD from the previous frame, use the
        # previous frame's position instead.
        if player_pos and self._prev_player_pos:
            jump_dist = abs(player_pos[0] - self._prev_player_pos[0]) + abs(
                player_pos[1] - self._prev_player_pos[1]
            )
            if jump_dist > 10:  # JUMP_THRESHOLD = 10 (2 × MOVE_STEP)
                player_pos = self._prev_player_pos

        # Check if player moved since last step
        moved: bool = False
        grid_changed: bool = (
            self._prev_grid is not None
            and not np.array_equal(grid, self._prev_grid)
        )

        # Track GAME_OVER transition (to prevent wall clearing on reset)
        prev_was_game_over = self._was_game_over
        self._was_game_over = (
            self._GameState is not None
            and game_state == self._GameState.GAME_OVER
        )

        # Simple player position (top-left) for wall detection
        simple_player_tl = self._find_player_simple(grid)
        simple_prev_tl = self._prev_simple_player_tl  # Use stored position

        if simple_player_tl and simple_prev_tl:
            if simple_player_tl != simple_prev_tl:
                moved = True
                self._consecutive_no_move = 0  # Reset freeze counter
                # Detect silent life loss: when the step counter exhausts,
                # the game silently resets player position + rotation
                # (without changing GameState). This manifests as a
                # position jump > 2×MOVE_STEP. The agent MUST detect
                # this and reset rotation-dependent state, otherwise it
                # thinks the player still has the correct rotation from
                # previous switcher bounces — but the rotation was
                # actually reset to StartRotation by the game.
                if not prev_was_game_over and not self._was_game_over:
                    _jump = abs(simple_player_tl[0] - simple_prev_tl[0]) + abs(
                        simple_player_tl[1] - simple_prev_tl[1]
                    )
                    # FIX 6: Filter out level transitions. When a level
                    # is completed, the player position jumps to the new
                    # level's start position. This jump is NOT life loss
                    # — it's a level transition. We detect this by checking
                    # if the PREVIOUS step had level progress (completed
                    # a level). If so, the position jump is expected and
                    # should NOT trigger LIFE_LOSS handling.
                    # Without this filter, the false LIFE_LOSS sets
                    # _level_had_game_over=True prematurely, defeating
                    # Fix 3 (no urgent refill during first life) and
                    # corrupting rotation tracking.
                    if _jump > 10 and not self._prev_progress:
                        print(f"    [LIFE_LOSS] step={self._step} jump={_jump} "
                              f"prev={simple_prev_tl} new={simple_player_tl} — "
                              f"resetting rotation/bounce state")
                        # CRITICAL: After life loss, the game resets
                        # rotation to StartRotation. This means the goal
                        # is BLOCKED again (unless StartRotation ==
                        # GoalRotation, which we discover by trying).
                        # Set goal_blocked for ALL known goals so the
                        # agent goes back to switcher instead of
                        # wasting steps on an unreachable goal.
                        # (Previously: self._goal_blocked.clear() — WRONG!)
                        self._goal_blocked.clear()
                        for g in self._known_goals:
                            self._goal_blocked.add(g)
                        self._target_switcher = None
                        self._force_switcher_visit = False
                        self._simple_walls.clear()
                        self._bounce_mode = False
                        self._bounce_phase = 0
                        self._bounce_dir = 0
                        self._bounce_switcher = None
                        self._bounce_count = 0
                        self._switcher_visit_count.clear()
                        self._last_visit_pos = None
                        self._exhausted_switchers.clear()
                        # Reset consumed refills (restored on life loss).
                        # Keep _known_refills since positions don't change.
                        self._consumed_refills.clear()
                        self._need_refill = False
                        # CCG rotation tracking: reset rotation counter.
                        # i^k where k=0 = StartRotation. After life loss,
                        # rotation is reset, so k=0.
                        self._rotation_k = 0
                        # Reset step counter (life was refreshed).
                        # Also update estimated step budget: the number
                        # of steps since last reset tells us the actual
                        # step budget for this level.
                        if self._steps_since_reset > 5:
                            self._estimated_step_budget = max(
                                10,
                                int(self._steps_since_reset * 0.9)
                            )
                            print(f"    [STEP_BUDGET] Updated to "
                                  f"{self._estimated_step_budget} "
                                  f"(observed {self._steps_since_reset} steps)")
                        self._steps_since_reset = 0
                        # CRITICAL FIX: Set _level_had_game_over = True
                        # after LIFE_LOSS (not just GAME_OVER). This
                        # enables bounce mode in subsequent lives, using
                        # the CCG-based bounce_target calculated from
                        # _tried_failed_rotations collected during the
                        # first life's natural circulation.
                        # Without this, the agent keeps doing expensive
                        # natural circulation (switcher→goal→switcher)
                        # in every life, never entering efficient bounce
                        # mode until all 3 lives are lost (GAME_OVER).
                        self._level_had_game_over = True
                        print(f"    [LIFE_LOSS→BOUNCE] _level_had_game_over=True, "
                              f"failed_rot={self._tried_failed_rotations} "
                              f"-> bounce mode enabled for next life")
                # DIAGNOSTIC: print movement during bounce mode
                if self._bounce_mode or self._bounce_count > 0:
                    print(f"    [BOUNCE_MOVE] step={self._step} player={simple_player_tl} "
                          f"prev={simple_prev_tl} bounce_mode={self._bounce_mode} "
                          f"bounce_phase={self._bounce_phase} bounce_count={self._bounce_count}")
                # Check if player moved to a known switcher position
                # NOTE: Don't clear goal_blocked here — _simple_navigate
                # handles the bounce logic and decides when to clear it.
                # Only clear _target_switcher so _simple_navigate uses the
                # on_any_sw path (which has the bounce logic).
                if simple_player_tl == self._target_switcher:
                    print(f"    [SW_ARRIVE] step={self._step} player={simple_player_tl} "
                          f"target_sw={self._target_switcher}")
                    self._simple_visited_sw.add(simple_player_tl)
                    self._simple_walls.clear()
                    self._target_switcher = None
                # Player moved — check if sprite changed (switcher triggered)
                # BUT don't clear walls if this movement is from GAME_OVER
                # (either previous frame was GAME_OVER, or current frame IS
                # GAME_OVER — in both cases the position change is not from
                # normal switcher interaction).
                if not prev_was_game_over and not self._was_game_over:
                    step = self._simple_step
                    curr_sp = grid[simple_player_tl[1]:simple_player_tl[1]+5,
                                   simple_player_tl[0]:simple_player_tl[0]+5]
                    prev_sp = self._prev_grid[simple_prev_tl[1]:simple_prev_tl[1]+5,
                                              simple_prev_tl[0]:simple_prev_tl[0]+5]
                    if curr_sp.shape == prev_sp.shape and not np.array_equal(curr_sp, prev_sp):
                        # Sprite changed → switcher triggered → clear walls
                        self._simple_walls.clear()
            else:
                # Player didn't move — increment freeze counter.
                # Only the FIRST non-movement (consecutive_no_move==1) is
                # a real collision. Subsequent non-movements are freeze
                # period (akoadfsur>0) and must NOT add fake walls.
                self._consecutive_no_move += 1
                # DIAGNOSTIC: print when player didn't move during bounce mode
                if self._bounce_mode or self._bounce_count > 0:
                    print(f"    [BOUNCE_NOMOVE] step={self._step} player={simple_player_tl} "
                          f"prev={simple_prev_tl} last_act={self._last_action} "
                          f"bounce_mode={self._bounce_mode} bounce_count={self._bounce_count} "
                          f"no_move={self._consecutive_no_move}")
                if (not prev_was_game_over
                        and self._consecutive_no_move <= 1
                        and self._last_action is not None
                        and self._last_action in [1, 2, 3, 4]):
                    action_deltas = {
                        1: (0, -self._simple_step),   # UP
                        2: (0, self._simple_step),    # DOWN
                        3: (-self._simple_step, 0),   # LEFT
                        4: (self._simple_step, 0),    # RIGHT
                    }
                    dx, dy = action_deltas[self._last_action]
                    wall_pos = (simple_player_tl[0] + dx, simple_player_tl[1] + dy)
                    # Classify the blocking wall:
                    # - Goal block (sprite has color 5) → temporary wall
                    #   in _simple_walls, cleared on switcher visit.
                    # - Real wall (no color 5) → permanent wall in
                    #   _discovered_walls, never cleared by fallback.
                    #   This prevents BFS from re-routing through the
                    #   same wall after dynamic walls are cleared.
                    wx, wy = wall_pos
                    if 0 <= wx < 60 and 0 <= wy < 60:
                        wall_sprite = grid[wy:wy + 5, wx:wx + 5]
                        if 5 in wall_sprite:
                            # Goal block: the goal is blocked because
                            # the player's attributes (rotation/shape/
                            # color) don't match. Set goal_blocked so
                            # the agent goes back to switcher instead
                            # of repeatedly trying the goal.
                            # GPCT insight: this is a boundary layer
                            # state transition — the goal's reachability
                            # depends on the rotation state, which we
                            # can only change by visiting the switcher.
                            self._goal_blocked.add(wall_pos)
                            self._simple_walls.discard(wall_pos)
                            # CCG: record this rotation_k as FAILED.
                            # This allows precise bounce_target calculation:
                            # bounce_target = smallest k not in _tried_failed_rotations.
                            self._tried_failed_rotations.add(self._rotation_k)
                            self._goal_was_ever_unblocked = False
                            print(f"    [GOAL_FROZEN] step={self._step} "
                                  f"goal={wall_pos} player={simple_player_tl} "
                                  f"rot_k={self._rotation_k} "
                                  f"failed_rotations={self._tried_failed_rotations} "
                                  f"-> goal_blocked SET (attributes mismatch)")
                        else:
                            # Real wall: permanent (only cleared on level change)
                            self._discovered_walls.add(wall_pos)
                    else:
                        self._discovered_walls.add(wall_pos)
        else:
            # Player position unknown — reset freeze counter
            self._consecutive_no_move = 0

        # Store current player position for next step
        self._prev_simple_player_tl = simple_player_tl

        # Track level progress
        progress: bool = levels_completed > self._prev_levels

        # --- Per-level step limit + Sleep-Step trajectory management ---
        if progress:
            # Level completed: archive trajectory as success and learn.
            self.memory.end_trajectory(success=True, info="LEVEL_COMPLETE")
            rules = self.sleep_step.learn_from_episode(self.memory)
            self.planner.load_defensive_rules(rules)
            # Start fresh trajectory for the next level.
            self.memory.start_trajectory()
            # Reset per-level step counter.
            self._level_steps = 0
            # Update per-level baseline from level_baselines if available.
            if self._level_baselines:
                idx = min(levels_completed, len(self._level_baselines) - 1)
                self._level_baseline = self._level_baselines[idx]
                self._level_hard_cap = 5 * self._level_baseline
            # Reset planner per-level state.
            self.planner.notify_level_changed()
            # Reset simple BFS navigation state for new level.
            self._simple_walls.clear()
            self._simple_visited_sw.clear()
            self._known_switchers.clear()
            self._goal_blocked.clear()
            self._fatal_positions.clear()
            self._grid_walls.clear()
            self._force_switcher_visit = False
            self._target_switcher = None
            self._discovered_walls.clear()
            self._switcher_visit_count.clear()
            self._bounce_mode = False
            self._bounce_phase = 0
            self._bounce_dir = 0
            self._bounce_switcher = None
            self._bounce_count = 0
            self._last_visit_pos = None
            self._exhausted_switchers.clear()
            self._level_switcher_visits = 0
            self._level_had_game_over = False
            self._goal_was_ever_unblocked = False
            self._tried_failed_rotations.clear()
            self._consecutive_no_move = 0
            # Reset refill state for new level.
            self._known_refills.clear()
            self._consumed_refills.clear()
            self._need_refill = False
            # CCG rotation tracking: reset for new level.
            self._rotation_k = 0
            self._goal_rotation_k = None
            # Step counter tracking: reset for new level.
            self._steps_since_reset = 0
            self._known_goals.clear()
            # Reset GAME_OVER count per level (each level may need
            # multiple GAME_OVERs for attribute cycling).
            self._game_over_count = 0
        else:
            # Increment per-level step counter.
            self._level_steps += 1

        # Detect goal-blocked: player is AT goal but level didn't
        # advance. This means the player's attributes (rot, shape, color)
        # don't match the goal's requirements. The planner will seek
        # switchers to change attributes.
        # Only trigger when player is exactly at the goal position
        # (distance 0), not just nearby.
        if player_pos is not None and not progress:
            at_goal: bool = False
            for obj in objects:
                if getattr(obj, "is_goal", False) and not getattr(obj, "is_player", False):
                    goal_pos = (int(obj.centroid[0]), int(obj.centroid[1]))
                    dist_to_goal = abs(goal_pos[0] - player_pos[0]) + abs(goal_pos[1] - player_pos[1])
                    if dist_to_goal <= 2:  # At goal position (exact or near-exact)
                        self.planner.mark_goal_blocked(goal_pos)
                        at_goal = True
                        break
            if not at_goal:
                # Player is not at any goal: clear goal-blocked state
                # so the agent doesn't stay in switcher mode forever
                self.planner._goal_blocked_pos = None

        # Update world model
        self.world_model.update(
            step=self._step,
            grid=grid,
            levels_completed=levels_completed,
            win_levels=self._get_win_levels(latest_frame),
            game_state=game_state,
            available_actions=available_actions,
            action_taken=self._last_action,
            player_pos=player_pos,
            player_color=player_obj.color if player_obj else None,
        )

        # Record to episode memory
        self.memory.record(
            step=self._step,
            action=self._last_action if self._last_action is not None else 0,
            grid=grid,
            levels_completed=levels_completed,
            game_state=game_state,
            player_pos=player_pos,
            moved=moved,
            progress=progress,
        )

        # Handle GAME_OVER: reset and continue
        if self._GameState and game_state == self._GameState.GAME_OVER:
            # CCG insight: check which rotation_k values failed.
            # _tried_failed_rotations contains all k values that
            # triggered GOAL_FROZEN. bounce_target = smallest k not in set.
            target_k = 1
            while target_k in self._tried_failed_rotations and target_k <= 3:
                target_k += 1
            bt = max(0, target_k - 1)
            print(f"    [GAME_OVER_ANALYSIS] failed_rotations={self._tried_failed_rotations} "
                  f"visits={self._level_switcher_visits} "
                  f"target_k={target_k} bounce_target={bt}")
            # In LS20, GAME_OVER is from step exhaustion, NOT from
            # stepping on a dangerous position. So we do NOT record
            # fatal positions — they would be false positives that
            # block BFS from finding valid paths.
            # Clear any previously recorded false fatal positions.
            self._fatal_positions.clear()
            # Clear stale navigation state: after RESET, the player's
            # attributes (rot/shape/color) are reset to initial values.
            # goal_blocked and target_switcher are no longer valid.
            # simple_walls (temporary goal-block walls) are also stale.
            # KEEP: _discovered_walls (real walls), _known_switchers
            # (switcher positions don't change), _grid_walls (will be
            # refreshed next frame).
            # CRITICAL: SET goal_blocked for all known goals (rotation
            # was reset to StartRotation → goal IS blocked unless
            # StartRotation == GoalRotation, which we discover by trying).
            # Previously: self._goal_blocked.clear() — WRONG, caused
            # agent to waste steps on unreachable goal after GAME_OVER.
            self._goal_blocked.clear()
            for g in self._known_goals:
                self._goal_blocked.add(g)
            self._target_switcher = None
            self._force_switcher_visit = False
            self._simple_walls.clear()
            # Reset bounce state: player attributes reset to initial,
            # so bounce trip/count are no longer valid.
            self._bounce_mode = False
            self._bounce_phase = 0
            self._bounce_dir = 0
            self._bounce_switcher = None
            self._bounce_count = 0
            self._switcher_visit_count.clear()
            self._last_visit_pos = None
            self._exhausted_switchers.clear()
            self._consecutive_no_move = 0
            # CCG: reset rotation counter (GAME_OVER resets rotation).
            self._rotation_k = 0
            # Reset step counter (fresh life after GAME_OVER).
            self._steps_since_reset = 0
            # Reset consumed refills (they reappear after level restart).
            # Keep _known_refills since positions don't change.
            self._consumed_refills.clear()
            self._need_refill = False
            # Mark that we've had a GAME_OVER in this level — enables
            # adaptive bounce on the next life.
            self._level_had_game_over = True
            # CCG: calculate bounce_target from tried_failed_rotations
            tk = 1
            while tk in self._tried_failed_rotations and tk <= 3:
                tk += 1
            print(f"    [ADAPTIVE] GAME_OVER — failed_rot={self._tried_failed_rotations} "
                  f"bounce_target={max(0, tk-1)} for next life")
            # L4: record failure for Bayesian weight adaptation.
            self.fuse_gate.update_result(success=False)
            # Sleep-Step: archive trajectory as failure and learn rules.
            self.memory.end_trajectory(success=False, info="GAME_OVER")
            rules = self.sleep_step.learn_from_episode(self.memory)
            self.planner.load_defensive_rules(rules)
            # Start fresh trajectory for the next attempt.
            self.memory.start_trajectory()
            self._step += 1
            self._level_steps += 1
            self._prev_grid = grid.copy()
            self._prev_player_pos = player_pos
            self._prev_levels = levels_completed
            self._prev_progress = progress  # For Fix 6
            self._last_action = 0
            if self._GameAction:
                return self._GameAction.RESET
            return 0

        # L4: On level completion, extract macros and record success.
        if progress:
            self.memory.extract_macros()
            self.fuse_gate.update_result(success=True)

        # --- Direct planning (no exploration phase) ---
        # Plan next action directly from step 0 for maximum RHAE efficiency.
        steps_since_progress: int = self.world_model.get_steps_since_progress()

        # DEBUG: Print detection info every 10 steps
        if self._step % 10 == 0:
            goals_simple, sw_simple, _, _ = self._find_sprites_simple(
                grid, simple_player_tl
            ) if simple_player_tl else ([], [], set(), [])
            print(f"    [DBG] step={self._step} player_tl={simple_player_tl} "
                  f"goals={goals_simple} switchers={sw_simple} "
                  f"grid_walls={len(self._grid_walls)} "
                  f"dyn_walls={len(self._simple_walls)} "
                  f"disc_walls={len(self._discovered_walls)} "
                  f"goal_blocked={len(self._goal_blocked)} "
                  f"fatal={len(self._fatal_positions)} "
                  f"level_steps={self._level_steps}/{self._level_hard_cap}")

        # --- Simple BFS navigation (minimal agent strategy) ---
        action_val: int = self._simple_navigate(grid, available_actions)

        # DEBUG: Print when player doesn't move
        if simple_player_tl and simple_prev_tl and simple_player_tl == simple_prev_tl:
            if self._step % 5 == 0:
                print(f"    [STUCK] step={self._step} act={self._last_action} "
                      f"player_tl={simple_player_tl} "
                      f"walls={list(self._simple_walls)[:5]} "
                      f"disc={list(self._discovered_walls)[:5]}")

        # Convert to GameAction enum
        # NOTE: GameAction(value) does NOT work for this enum despite values
        # being ints. Must use name-based lookup or direct attribute access.
        if self._GameAction:
            _action_map = {
                0: self._GameAction.RESET,
                1: self._GameAction.ACTION1,
                2: self._GameAction.ACTION2,
                3: self._GameAction.ACTION3,
                4: self._GameAction.ACTION4,
                5: self._GameAction.ACTION5,
                6: self._GameAction.ACTION6,
                7: self._GameAction.ACTION7,
            }
            action = _action_map.get(action_val, self._GameAction.ACTION1)
        else:
            action = action_val

        # Update state
        self._step += 1
        self._steps_since_reset += 1  # Step counter for refill navigation
        self._prev_grid = grid.copy()
        self._prev_player_pos = player_pos
        self._prev_levels = levels_completed
        self._prev_progress = progress  # For Fix 6: level transition detection
        self._last_action = action_val

        return action

    def reset(self) -> None:
        """Reset agent state for a new game."""
        self.perception = GridParser()
        self.world_model.reset()
        self.planner.reset()
        self.memory.reset()
        self.fuse_gate.reset()
        self.sleep_step.reset()
        self._step = 0
        self._level_steps = 0
        self._prev_grid = None
        self._prev_player_pos = None
        self._prev_player_sprite = None
        self._prev_levels = 0
        self._last_action = None
        self._game_over_count = 0
        # Reset simple BFS navigation state.
        self._simple_walls.clear()
        self._simple_visited_sw.clear()
        self._known_switchers.clear()
        self._simple_last_bfs_action = None
        self._prev_simple_player_tl = None
        self._goal_blocked.clear()
        self._fatal_positions.clear()
        self._was_game_over = False
        self._grid_walls.clear()
        self._force_switcher_visit = False
        self._target_switcher = None
        self._discovered_walls.clear()
        self._switcher_visit_count.clear()
        self._bounce_mode = False
        self._bounce_phase = 0
        self._bounce_dir = 0
        self._bounce_switcher = None
        self._bounce_count = 0
        self._last_visit_pos = None
        self._exhausted_switchers.clear()
        self._level_switcher_visits = 0
        self._level_had_game_over = False
        self._known_refills.clear()
        self._consumed_refills.clear()
        self._need_refill = False
        self._rotation_k = 0
        self._goal_rotation_k = None
        self._steps_since_reset = 0
        self._known_goals.clear()
        self._tried_failed_rotations.clear()

    def get_stats(self) -> dict:
        """Get agent statistics.

        Returns:
            Dictionary with step count, stagnation, game over count,
            fuse gate state, and Sleep-Step defensive rules.
        """
        stats: dict = self.memory.get_stats()
        stats["steps"] = self._step
        stats["level_steps"] = self._level_steps
        stats["level_hard_cap"] = self._level_hard_cap
        stats["stagnation"] = self.world_model.get_steps_since_progress()
        stats["game_over_count"] = self._game_over_count
        stats["fuse_gate"] = self.fuse_gate.get_state()
        stats["sleep_step"] = self.sleep_step.get_rules()
        return stats
