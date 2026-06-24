"""
HeuristicPlanner - Rule-based action selection for ARC-AGI-3 games.

LS20-specific configuration:
- MOVE_STEP = 5: Each keyboard action moves the player 5 pixels.
- No BFS pathfinding: Grid colors do NOT reliably indicate wall positions,
  so BFS with wall checking is unreliable. Instead, we use simple Manhattan
  distance planning: move toward the nearest target along the axis with the
  greatest distance.
- Blocked-direction tracking: If a move doesn't result in player movement,
  that direction is marked as blocked and avoided in subsequent steps.

Key parameters:
- MOVE_STEP: Pixels per action (5 for LS20).
- BFS_MAX_DEPTH: Maximum exploration depth (kept for API compatibility).
- STUCK_THRESHOLD: Steps without progress before exploration mode.
"""

from __future__ import annotations

from typing import Optional, Any
import random

import numpy as np


class HeuristicPlanner:
    """Rule-based planner for ARC-AGI-3 games using Manhattan distance.

    The planner selects keyboard actions (ACTION1-4) that move the player
    toward the nearest target object. No BFS or wall checking is performed
    because grid colors in LS20 do not reliably indicate wall positions.

    Blocked-direction tracking:
    - When the agent takes an action and the player doesn't move, that
      action is added to _failed_directions.
    - The planner avoids _failed_directions when choosing the next action.
    - If all preferred directions are blocked, _failed_directions is cleared
      and the planner tries again.

    Stuck detection:
    - The planner hashes a downsampled version of the grid each step.
    - If the same hash is seen repeatedly, the stuck counter increments.
    - When the stuck counter exceeds STUCK_THRESHOLD, the planner enters
      exploration mode and rotates through available actions.
    """

    # Action ID -> (name, (dx, dy)) where dx=column delta, dy=row delta
    # These are unit directions; actual movement is MOVE_STEP pixels.
    KEYBOARD_ACTIONS: dict[int, tuple[str, tuple[int, int]]] = {
        1: ("up", (0, -1)),     # row decreases (ACTION1)
        2: ("down", (0, 1)),    # row increases (ACTION2)
        3: ("left", (-1, 0)),   # col decreases (ACTION3)
        4: ("right", (1, 0)),   # col increases (ACTION4)
    }

    # UI colors that should never be targets
    UI_COLORS: set[int] = {11, 12, 14}

    MOVE_STEP: int = 5         # LS20 moves 5 pixels per action
    BFS_MAX_DEPTH: int = 12    # Max exploration depth (not used for BFS)
    STUCK_THRESHOLD: int = 10  # Steps without progress before exploration
    DFS_MAX_DEPTH: int = 25    # Max depth for DFS backtracking search
    SWITCHER_MODE_THRESHOLD: int = 5  # Steps since switcher visit before entering switcher mode

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the planner.

        Args:
            seed: Random seed for reproducible exploration.
        """
        self._rng: random.Random = random.Random(seed)
        self._action_history: list[int] = []
        self._failed_directions: set[int] = set()
        self._blocked_positions: set[tuple[int, int]] = set()  # Dynamic wall learning
        self._exploration_steps: int = 0
        self._last_action: Optional[int] = None
        self._visited_states: set[str] = set()
        self._stuck_counter: int = 0
        self._exploration_rotation: int = 0

        # Switcher-visit mode: proactively visit switchers when stuck.
        # When the agent can't reach the goal (attribute mismatch or walls),
        # it enters switcher-visit mode to change player attributes.
        self._switcher_visit_mode: bool = False
        self._visited_switchers: set[tuple[int, int]] = set()
        self._steps_since_switcher: int = 0
        self._goal_blocked_pos: Optional[tuple[int, int]] = None

        # No-progress counter: tracks consecutive steps where the player
        # didn't move. When it exceeds NO_PROGRESS_THRESHOLD, all blocked
        # positions are cleared (they're likely false positives).
        self._no_progress_counter: int = 0

        # Current level index (0-based). Used to determine whether
        # switcher-visit mode should be active. On Level 0, the goal
        # is directly reachable (switcher_free=True), so switcher mode
        # is disabled.
        self._level_index: int = 0

        # DFS backtracking state (L4 fuse / stuck recovery).
        # Each stack entry is (position, action_path).
        self._dfs_stack: list[tuple[tuple[int, int], list[int]]] = []
        self._dfs_visited: set[tuple[int, int]] = set()
        self._dfs_active: bool = False

        # Defensive rules loaded from SleepStep analysis.
        self._forbidden_positions: set[tuple[int, int]] = set()
        self._forbidden_actions: set[int] = set()
        self._perception_overrides: list[dict] = []

    def plan(
        self,
        grid: np.ndarray,
        player_pos: Optional[tuple[int, int]],
        player_obj: Optional[Any],
        objects: list,
        available_actions: list[int],
        game_state: Optional[Any] = None,
        steps_since_progress: int = 0,
    ) -> int:
        """Choose the next action.

        Uses Manhattan distance planning: finds the nearest target object
        and moves toward it along the axis with the greatest distance.
        Falls back to exploration when no target is found or when stuck.

        Args:
            grid: 64x64 int8 numpy array.
            player_pos: Player position (cx, cy) = (col, row).
            player_obj: Player GameObject (or None).
            objects: List of detected GameObjects.
            available_actions: List of available action IDs.
            game_state: Current game state (unused, kept for API compat).
            steps_since_progress: Steps since last level completion.

        Returns:
            Action ID (int).
        """
        # Reset blocked directions on new level
        if steps_since_progress == 0:
            self._failed_directions.clear()
            self._stuck_counter = 0
            self._goal_blocked_pos = None
            self._visited_switchers.clear()
            self._switcher_visit_mode = False  # Try goal first
            self._steps_since_switcher = 0

        # Check if player reached a switcher (BEFORE stuck detection,
        # so switcher visits are detected even in stuck mode).
        if player_pos is not None:
            self._check_switcher_reached(player_pos, objects)

        # Detect revisited states (stuck detection)
        state_hash: str = self._hash_grid(grid)
        if state_hash in self._visited_states:
            self._stuck_counter += 1
        else:
            self._visited_states.add(state_hash)
            self._stuck_counter = 0

        # Before entering stuck handler, ensure ALL directions have been tried.
        # This prevents the agent from entering stuck mode with untried directions.
        if 0 < self._stuck_counter < self.STUCK_THRESHOLD and player_pos is not None:
            for d in [1, 2, 3, 4]:
                if d in available_actions and d not in self._failed_directions:
                    self._last_action = d
                    self._action_history.append(d)
                    self._exploration_steps += 1
                    return d

        # Stuck handling takes priority
        if self._stuck_counter >= self.STUCK_THRESHOLD:
            # Emergency reset: if stuck for too long, clear all dynamic
            # state to allow retrying all directions.
            if self._stuck_counter >= 8:
                self._blocked_positions.clear()
                self._failed_directions.clear()
                self._stuck_counter = 0
                self._visited_states.clear()
                # Fall through to normal planning after clearing
            else:
                # Try DFS backtracking, but only if the action isn't blocked.
                if player_pos is not None:
                    target = self._find_nearest_target(player_pos, objects)
                    if target is not None:
                        target_pos = (
                            int(target.centroid[0]),
                            int(target.centroid[1]),
                        )
                        keyboard_acts = [
                            a for a in available_actions if a in self.KEYBOARD_ACTIONS
                        ]
                        if keyboard_acts:
                            dfs_path = self._plan_dfs(
                                player_pos, target_pos, keyboard_acts
                            )
                            if dfs_path and len(dfs_path) > 0:
                                action = dfs_path[0]
                                # Only return if action isn't blocked
                                if action not in self._failed_directions:
                                    self._last_action = action
                                    self._action_history.append(action)
                                    self._exploration_steps += 1
                                    self._dfs_active = True
                                    return action

                # DFS failed or blocked: fall back to exploration rotation.
                # Try ALL directions systematically, not just rotation.
                for d in [1, 2, 3, 4]:
                    if d in available_actions and d not in self._failed_directions:
                        self._last_action = d
                        self._action_history.append(d)
                        self._exploration_steps += 1
                        self._stuck_counter = max(0, self._stuck_counter - 1)
                        return d

                # All directions blocked: force clear and try any action
                self._failed_directions.clear()
                self._blocked_positions.clear()
                action = self._handle_stuck(available_actions)
                self._last_action = action
                self._action_history.append(action)
                self._exploration_steps += 1
                return action

        # Classify available actions
        keyboard_actions: list[int] = [
            a for a in available_actions if a in self.KEYBOARD_ACTIONS
        ]
        is_keyboard: bool = len(keyboard_actions) > 0

        if is_keyboard and player_pos is not None:
            action = self._plan_manhattan(
                player_pos, objects, keyboard_actions, steps_since_progress,
            )
        elif is_keyboard:
            # No player position: random keyboard action
            action = self._rng.choice(keyboard_actions)
        else:
            # Non-keyboard game: use first available action
            action = available_actions[0] if available_actions else 1

        self._last_action = action
        self._action_history.append(action)
        self._exploration_steps += 1
        return action

    def _plan_manhattan(
        self,
        player_pos: tuple[int, int],
        objects: list,
        available_actions: list[int],
        steps_since_progress: int,
    ) -> int:
        """Plan action using Manhattan distance with switcher-visit mode.

        Strategy:
        1. Check if player reached a switcher (mark visited, clear blocks).
        2. Determine target based on switcher-visit mode:
           - In switcher mode: target nearest unvisited switcher.
           - Normal mode: target nearest goal. If goal blocked, enter switcher mode.
           - After SWITCHER_MODE_THRESHOLD steps without switcher visit, enter mode.
        3. Navigate to target using BFS then Manhattan fallback.
        4. Check _blocked_positions for all direction choices to prevent oscillation.
        5. Fallback to systematic exploration.

        Args:
            player_pos: (cx, cy) = (col, row) player position.
            objects: List of detected GameObjects.
            available_actions: Available keyboard action IDs.
            steps_since_progress: Steps since last level completion.

        Returns:
            Action ID (int).
        """
        px, py = player_pos

        # --- Step 1: Check if player reached a switcher ---
        self._check_switcher_reached(player_pos, objects)

        # --- Step 2: Detect switchers and goals ---
        all_switchers: list = [
            o for o in objects if o.is_switcher and not o.is_player
        ]
        unvisited_switchers: list = [
            s for s in all_switchers
            if (int(s.centroid[0]), int(s.centroid[1]))
            not in self._visited_switchers
        ]

        # Track steps since last switcher visit
        self._steps_since_switcher += 1

        # If all switchers visited but goal still not completed after
        # extended time, clear visited set to allow re-visiting
        # (multiple visits may be needed for multi-attribute goals)
        if (
            not unvisited_switchers
            and all_switchers
            and self._steps_since_switcher > self.SWITCHER_MODE_THRESHOLD * 5
        ):
            self._visited_switchers.clear()
            unvisited_switchers = list(all_switchers)
            self._steps_since_switcher = 0

        # --- Step 3: Determine target based on mode ---
        target: Optional[Any] = None

        if self._switcher_visit_mode:
            if unvisited_switchers:
                target = min(
                    unvisited_switchers,
                    key=lambda o: abs(o.centroid[0] - px)
                    + abs(o.centroid[1] - py),
                )
            else:
                # All switchers visited: exit mode, try goal
                self._switcher_visit_mode = False
                self._failed_directions.clear()
                target = self._find_nearest_target(player_pos, objects)
        elif (
            self._goal_blocked_pos is not None
            and unvisited_switchers
        ):
            # Enter switcher mode ONLY when goal is attribute-blocked
            # (player at goal but level didn't advance).
            # Wall-blocked goals are handled in the else branch below.
            self._switcher_visit_mode = True
            self._failed_directions.clear()
            target = min(
                unvisited_switchers,
                key=lambda o: abs(o.centroid[0] - px)
                + abs(o.centroid[1] - py),
            )
        else:
            # Normal mode: target nearest goal
            target = self._find_nearest_target(player_pos, objects)

            # Check if goal is blocked (in _blocked_positions)
            goal_blocked: bool = False
            if target is not None and target.is_goal:
                goal_pos = (int(target.centroid[0]), int(target.centroid[1]))
                if goal_pos in self._blocked_positions:
                    goal_blocked = True

            if goal_blocked:
                if unvisited_switchers:
                    self._switcher_visit_mode = True
                    self._failed_directions.clear()
                    target = min(
                        unvisited_switchers,
                        key=lambda o: abs(o.centroid[0] - px)
                        + abs(o.centroid[1] - py),
                    )
                else:
                    target = self._find_nearest_non_goal_target(
                        player_pos, objects
                    )

        # Periodically clear failed directions (every 10 steps)
        # but NOT _blocked_positions (walls don't change)
        if self._exploration_steps > 0 and self._exploration_steps % 10 == 0:
            self._failed_directions.clear()

        # --- Step 4: Navigate to target (multi-waypoint BFS) ---
        if target is not None:
            # Build waypoint list: primary target first, then alternatives.
            # Like the minimal agent: try goal first, if BFS fails try
            # switcher, if that fails too, clear walls and retry.
            waypoints: list = [target]
            if not target.is_switcher:
                # Primary is goal: add switchers as fallback waypoints
                for sw in unvisited_switchers:
                    sw_pos = (int(sw.centroid[0]), int(sw.centroid[1]))
                    if sw_pos not in self._visited_switchers:
                        waypoints.append(sw)
            else:
                # Primary is switcher: add goal as fallback
                goal_target = self._find_nearest_target(player_pos, objects)
                if goal_target is not None:
                    waypoints.append(goal_target)

            for wp in waypoints:
                bfs_action = self._plan_bfs(
                    player_pos, wp, objects, available_actions
                )
                if bfs_action is not None and bfs_action in available_actions:
                    return bfs_action

            # All BFS failed: clear blocked positions and retry BFS once
            self._blocked_positions.clear()
            self._failed_directions.clear()
            for wp in waypoints:
                bfs_action = self._plan_bfs(
                    player_pos, wp, objects, available_actions
                )
                if bfs_action is not None and bfs_action in available_actions:
                    return bfs_action

            # Manhattan fallback
            tx: int = int(target.centroid[0])
            ty: int = int(target.centroid[1])
            dx: int = tx - px
            dy: int = ty - py
            preferred: list[int] = self._build_preferred_actions(dx, dy)

            # Try preferred actions that aren't blocked
            for action_id in preferred:
                if (
                    action_id in available_actions
                    and action_id not in self._failed_directions
                    and not self._is_forbidden(player_pos, action_id)
                ):
                    return action_id

            # All preferred blocked: try any unblocked action
            for action_id in [1, 2, 3, 4]:
                if (
                    action_id in available_actions
                    and action_id not in self._failed_directions
                    and not self._is_forbidden(player_pos, action_id)
                ):
                    return action_id

            # Everything blocked: clear ALL state and try ALL directions
            self._failed_directions.clear()
            self._blocked_positions.clear()
            for action_id in [1, 2, 3, 4]:
                if (
                    action_id in available_actions
                    and not self._is_forbidden(player_pos, action_id)
                ):
                    return action_id

        # --- Step 5: Systematic exploration ---
        direction_order: list[int] = [4, 2, 3, 1]  # right, down, left, up
        for d in direction_order:
            if d in available_actions and d != self._last_action:
                return d

        # Last resort: clear everything and try random
        self._blocked_positions.clear()
        self._failed_directions.clear()
        return self._rng.choice(available_actions)

    def _get_destination(
        self, pos: tuple[int, int], action_id: int
    ) -> tuple[int, int]:
        """Compute the destination position after taking an action.

        Args:
            pos: (col, row) current position.
            action_id: Action ID (1=up, 2=down, 3=left, 4=right).

        Returns:
            (col, row) destination position after the action.
        """
        if action_id not in self.KEYBOARD_ACTIONS:
            return pos
        _, (dx, dy) = self.KEYBOARD_ACTIONS[action_id]
        return (
            pos[0] + dx * self.MOVE_STEP,
            pos[1] + dy * self.MOVE_STEP,
        )

    def _build_preferred_actions(self, dx: int, dy: int) -> list[int]:
        """Build ordered list of preferred actions based on delta to target.

        Prioritizes the axis with the greater absolute distance.
        ALWAYS includes all four directions to ensure the agent tries
        every possible move before giving up.

        Args:
            dx: Column delta (target_x - player_x).
            dy: Row delta (target_y - player_y).

        Returns:
            List of action IDs in priority order (all 4 included).
        """
        preferred: list[int] = []
        if abs(dy) >= abs(dx):
            if dy < 0:
                preferred.append(1)  # up
            if dy > 0:
                preferred.append(2)  # down
            if dx < 0:
                preferred.append(3)  # left
            if dx > 0:
                preferred.append(4)  # right
            if dx == 0:
                preferred.extend([3, 4])
        else:
            if dx < 0:
                preferred.append(3)  # left
            if dx > 0:
                preferred.append(4)  # right
            if dy < 0:
                preferred.append(1)  # up
            if dy > 0:
                preferred.append(2)  # down
            if dy == 0:
                preferred.extend([1, 2])
        # ALWAYS add any missing directions to ensure all 4 are tried
        for d in [1, 2, 3, 4]:
            if d not in preferred:
                preferred.append(d)
        return preferred

    def _check_switcher_reached(
        self, player_pos: tuple[int, int], objects: list
    ) -> bool:
        """Check if player has reached an unvisited switcher.

        When the player is within MOVE_STEP distance of an unvisited
        switcher, the switcher is marked as visited and blocked positions
        are cleared (attributes may have changed). The switcher-visit
        mode is NOT exited here — the caller (_plan_manhattan) decides
        when to exit based on whether all switchers have been visited.

        Args:
            player_pos: (col, row) player position.
            objects: List of detected GameObjects.

        Returns:
            True if a switcher was reached, False otherwise.
        """
        px, py = player_pos
        for obj in objects:
            if not obj.is_switcher or obj.is_player:
                continue
            sw_pos = (int(obj.centroid[0]), int(obj.centroid[1]))
            if sw_pos in self._visited_switchers:
                continue
            dist = abs(sw_pos[0] - px) + abs(sw_pos[1] - py)
            if dist <= self.MOVE_STEP:
                self._visited_switchers.add(sw_pos)
                # DON'T clear _blocked_positions — walls are permanent and
                # clearing them causes the agent to re-try known walls.
                # Only clear failed directions (attribute change may unblock).
                self._failed_directions.clear()
                self._steps_since_switcher = 0
                self._goal_blocked_pos = None
                self._stuck_counter = 0
                self._visited_states.clear()
                return True
        return False

    def notify_level_changed(self) -> None:
        """Reset per-level planner state.

        Called by the agent when a level change is detected.
        Clears all per-level state including switcher tracking,
        blocked positions, and stuck detection.
        Starts in switcher-visit mode to visit all switchers first,
        then go to the goal (like the minimal agent strategy).
        """
        self._switcher_visit_mode = False  # Try goal first (like minimal agent)
        self._visited_switchers.clear()
        self._steps_since_switcher = 0
        self._goal_blocked_pos = None
        self._no_progress_counter = 0
        self._blocked_positions.clear()
        self._failed_directions.clear()
        self._stuck_counter = 0
        self._visited_states.clear()
        self._exploration_steps = 0

    def _find_nearest_target(
        self,
        player_pos: tuple[int, int],
        objects: list,
    ) -> Optional[Any]:
        """Find the nearest target object to the player.

        Priority:
        1. Actual goal objects (is_goal=True, not player).
        2. Any non-wall, non-player, non-UI object with area >= 2.

        Distance is measured as Manhattan distance (|dx| + |dy|).

        Args:
            player_pos: (cx, cy) = (col, row) player position.
            objects: List of GameObjects.

        Returns:
            The nearest target GameObject, or None.
        """
        px, py = player_pos

        # Strategy 1: Find actual goal objects
        goals: list = [o for o in objects if o.is_goal and not o.is_player]
        if goals:
            best = min(
                goals,
                key=lambda o: abs(o.centroid[0] - px) + abs(o.centroid[1] - py),
            )
            return best

        # Strategy 2: Any non-wall, non-player, non-UI object
        best: Optional[Any] = None
        best_dist: float = float("inf")
        for obj in objects:
            if getattr(obj, "is_player", False) or getattr(obj, "is_wall", False):
                continue
            if obj.area < 2:
                continue
            if obj.color in self.UI_COLORS:
                continue
            ox, oy = obj.centroid
            dist: float = abs(ox - px) + abs(oy - py)
            if dist < best_dist:
                best_dist = dist
                best = obj
        return best

    def _handle_stuck(self, available_actions: list[int]) -> int:
        """Handle stuck state by rotating through available actions.

        Clears the stuck counter partially and returns the next action
        in the exploration rotation.

        Args:
            available_actions: Available action IDs.

        Returns:
            Action ID (int).
        """
        non_reset_actions: list[int] = [a for a in available_actions if a != 0]
        if not non_reset_actions:
            return 0
        idx: int = self._exploration_rotation % len(non_reset_actions)
        self._exploration_rotation += 1
        self._stuck_counter = max(0, self._stuck_counter - 2)
        return non_reset_actions[idx]

    # ------------------------------------------------------------------
    # DFS backtracking (L4 fuse recovery)
    # ------------------------------------------------------------------

    def _plan_dfs(
        self,
        player_pos: tuple[int, int],
        target_pos: tuple[int, int],
        available_actions: list[int],
    ) -> Optional[list[int]]:
        """Depth-first search for a path from player to target.

        Uses an explicit stack-based DFS with MOVE_STEP=5 pixel moves.
        Visited positions are tracked to avoid cycles.  The search is
        bounded by ``DFS_MAX_DEPTH``.

        Action→delta mapping (col, row):
            1 = up    → (0, -5)
            2 = down  → (0, +5)
            3 = left  → (-5, 0)
            4 = right → (+5, 0)

        Args:
            player_pos: (col, row) current player position.
            target_pos: (col, row) target position.
            available_actions: Available keyboard action IDs.

        Returns:
            List of action IDs forming a path to the target, or ``None``
            if no path is found within the depth limit.
        """
        # Action → (dcol, drow) with MOVE_STEP scaling.
        action_deltas: dict[int, tuple[int, int]] = {
            1: (0, -self.MOVE_STEP),   # up
            2: (0, self.MOVE_STEP),    # down
            3: (-self.MOVE_STEP, 0),   # left
            4: (self.MOVE_STEP, 0),    # right
        }

        # Filter to actions that are both available and mapped.
        valid_actions = [
            a for a in available_actions if a in action_deltas
        ]
        if not valid_actions:
            return None

        # Reset DFS state for a fresh search.
        self._dfs_stack.clear()
        self._dfs_visited.clear()

        # Manhattan-distance-based action ordering for efficiency.
        def _action_heuristic(action_id: int) -> int:
            """Return distance-to-target after taking this action."""
            dcol, drow = action_deltas[action_id]
            new_col = player_pos[0] + dcol
            new_row = player_pos[1] + drow
            return abs(target_pos[0] - new_col) + abs(target_pos[1] - new_row)

        ordered_actions = sorted(valid_actions, key=_action_heuristic)

        # Push initial state: (position, path_so_far).
        initial_entry: tuple[tuple[int, int], list[int]] = (player_pos, [])
        self._dfs_stack.append(initial_entry)
        self._dfs_visited.add(player_pos)

        while self._dfs_stack:
            current_pos, path = self._dfs_stack.pop()

            # Check if we've reached the target (within MOVE_STEP tolerance).
            dist_to_target = (
                abs(target_pos[0] - current_pos[0])
                + abs(target_pos[1] - current_pos[1])
            )
            if dist_to_target <= self.MOVE_STEP:
                return path

            # Depth limit check.
            if len(path) >= self.DFS_MAX_DEPTH:
                continue

            # Expand neighbours in heuristic order (closest first).
            # We push in reverse so the closest action is popped first.
            for action_id in reversed(ordered_actions):
                dcol, drow = action_deltas[action_id]
                new_col = current_pos[0] + dcol
                new_row = current_pos[1] + drow

                # Grid boundary check (64×64).
                if new_col < 0 or new_col >= 64:
                    continue
                if new_row < 0 or new_row >= 64:
                    continue

                new_pos = (new_col, new_row)
                if new_pos in self._dfs_visited:
                    continue

                # Defensive rule check: skip forbidden positions/actions.
                if self._is_forbidden(new_pos, action_id):
                    continue

                self._dfs_visited.add(new_pos)
                self._dfs_stack.append((new_pos, path + [action_id]))

        return None

    def backtrack(self) -> None:
        """Pop the DFS stack top for L4 fuse-triggered backtracking.

        When the BayesianFuseGate signals a fuse, the agent calls this
        method to discard the top of the DFS search stack, effectively
        abandoning the most recent exploration branch.  If the stack
        is empty, this is a no-op.
        """
        if self._dfs_stack:
            popped_pos, _ = self._dfs_stack.pop()
            # Optionally remove from visited so the position can be
            # re-explored from a different branch.
            self._dfs_visited.discard(popped_pos)
        self._dfs_active = False

    def _hash_grid(self, grid: np.ndarray) -> str:
        """Hash a downsampled grid for state comparison.

        Downsamples the 64x64 grid to 16x16 by taking the max of each
        4x4 block. This provides a coarse state hash that is robust to
        small pixel-level changes while detecting overall grid revisits.

        Args:
            grid: 64x64 int8 numpy array.

        Returns:
            String hash of the downsampled grid.
        """
        h, w = grid.shape
        ds_h: int = max(1, h // 4)
        ds_w: int = max(1, w // 4)
        if ds_h * 4 > h:
            ds_h = h // 4
        if ds_w * 4 > w:
            ds_w = w // 4
        if ds_h == 0 or ds_w == 0:
            return str(hash(grid.tobytes()))
        downsampled = grid[:ds_h * 4, :ds_w * 4].reshape(
            ds_h, 4, ds_w, 4
        ).max(axis=(1, 3))
        return str(hash(downsampled.tobytes()))

    def notify_no_progress(self) -> None:
        """Called when the player didn't move this step.

        Increments the no-progress counter. When it reaches 6, all
        blocked positions are cleared (they're likely false positives
        from previous failed moves) and the counter resets. The threshold
        of 6 ensures all 4 directions are tried before clearing.
        This prevents the agent from getting permanently stuck when
        _blocked_positions accumulates incorrect entries.
        """
        self._no_progress_counter += 1
        if self._no_progress_counter >= 4:
            # After 4 no-progress steps (enough to try all 4 directions),
            # clear blocked positions — they're likely false positives or
            # the agent is truly stuck and needs a fresh start.
            self._blocked_positions.clear()
            self._failed_directions.clear()
            self._no_progress_counter = 0

    def notify_moved(self) -> None:
        """Called when the player moved this step.

        Resets the no-progress counter since the player is making
        progress.
        """
        self._no_progress_counter = 0

    def mark_blocked(self, action: int, player_pos: Optional[tuple[int, int]] = None) -> None:
        """Mark an action as blocked (didn't result in movement).

        The action is added to _failed_directions so the planner avoids
        it in subsequent steps. This is how the planner learns about
        walls and obstacles without reading grid colors.

        Also, if player_pos is provided, the position the player tried to
        move to is added to _blocked_positions for BFS wall avoidance.

        Args:
            action: The action ID that was blocked.
            player_pos: (col, row) player position before the action.
        """
        self._failed_directions.add(action)

        # Dynamic wall learning: if player didn't move, the next position is likely a wall
        if player_pos is not None and action in self.KEYBOARD_ACTIONS:
            _, (dx, dy) = self.KEYBOARD_ACTIONS[action]
            # KEYBOARD_ACTIONS uses unit directions (0, -1, 1),
            # multiply by MOVE_STEP to get actual pixel delta.
            blocked_pos = (player_pos[0] + dx * self.MOVE_STEP, player_pos[1] + dy * self.MOVE_STEP)
            self._blocked_positions.add(blocked_pos)

    # ------------------------------------------------------------------
    # Defensive rules (Sleep-Step integration)
    # ------------------------------------------------------------------

    def load_defensive_rules(self, rules: dict) -> None:
        """Load defensive rules generated by SleepStep.

        Merges SleepStep-generated forbidden positions, forbidden actions,
        and perception overrides into the planner's internal rule sets.
        Existing rules are preserved (additive merge).

        Args:
            rules: Dictionary from ``SleepStep.learn_from_episode()`` or
                ``SleepStep.get_rules()``.  Recognised keys:
                    - ``"forbidden_positions"``: list of (col, row) tuples.
                    - ``"forbidden_actions"``: list of action IDs.
                    - ``"perception_fixes"``: list of override dicts.
        """
        for pos in rules.get("forbidden_positions", []):
            self._forbidden_positions.add(tuple(pos))
        for act in rules.get("forbidden_actions", []):
            self._forbidden_actions.add(int(act))
        fixes = rules.get("perception_fixes", [])
        if fixes:
            self._perception_overrides.extend(fixes)

    def _is_forbidden(self, pos: tuple[int, int], action: int) -> bool:
        """Check if a position/action is forbidden by defensive rules.

        Args:
            pos: (col, row) position to check.
            action: Action ID to check.

        Returns:
            ``True`` if the position or action is forbidden, ``False``
            otherwise.
        """
        if pos in self._forbidden_positions:
            return True
        if action in self._forbidden_actions:
            return True
        return False

    def mark_success(self, action: int, new_pos: Optional[tuple[int, int]] = None) -> None:
        """Mark an action as successful (resulted in movement).

        Clears failed directions (blockages are position-specific — moving
        to a new position means old direction-blockages no longer apply).
        Does NOT clear _blocked_positions — walls don't move and should
        be remembered. Blocked positions are only cleared when:
        - A switcher is visited (attributes may have changed)
        - Emergency reset (stuck for too long)
        - No-progress counter exceeds threshold

        Args:
            action: The action ID that was successful.
            new_pos: The new player position after the move (optional).
        """
        self._failed_directions.clear()

    def mark_goal_blocked(self, goal_pos: tuple[int, int]) -> None:
        """Mark the goal as attribute-blocked.

        Called when the player is at the goal position but the level
        doesn't advance. This means the player's attributes (rot, shape,
        color) don't match the goal's requirements. The planner will
        seek the nearest unvisited switcher to change attributes.

        Args:
            goal_pos: (col, row) position of the blocked goal.
        """
        self._goal_blocked_pos = goal_pos

    def mark_switcher_visited(self, switcher_pos: tuple[int, int]) -> None:
        """Mark a switcher as visited and clear goal-blocked state.

        Called when the player is at a switcher position. The switcher
        is added to the visited set (to avoid revisiting), and the
        goal-blocked state is cleared (to retry the goal after the
        attribute change).

        Args:
            switcher_pos: (col, row) position of the visited switcher.
        """
        self._visited_switchers.add(switcher_pos)
        self._goal_blocked_pos = None  # Retry goal after switcher visit

    def clear_goal_blocked(self) -> None:
        """Clear goal-blocked state and visited switchers.

        Called when a level is completed (level advance detected).
        Resets the goal-blocked flag and clears the visited switcher
        set for the next level.
        """
        self._goal_blocked_pos = None
        self._visited_switchers.clear()

    def reset(self) -> None:
        """Reset planner state for a new game."""
        self._action_history.clear()
        self._failed_directions.clear()
        self._blocked_positions.clear()
        self._exploration_steps = 0
        self._last_action = None
        self._visited_states.clear()
        self._stuck_counter = 0
        self._exploration_rotation = 0
        self._dfs_stack.clear()
        self._dfs_visited.clear()
        self._dfs_active = False
        self._forbidden_positions.clear()
        self._forbidden_actions.clear()
        self._perception_overrides.clear()
        self._switcher_visit_mode = False  # Try goal first on new game
        self._visited_switchers.clear()
        self._steps_since_switcher = 0
        self._goal_blocked_pos = None
        self._no_progress_counter = 0

    # ------------------------------------------------------------------
    # BFS path search (TOSAS Theory - Instruction 2 & 3: Full-RSB + Epiplexity)
    # ------------------------------------------------------------------

    def _bfs_path(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
        walls: set[tuple[int, int]],
    ) -> Optional[list[int]]:
        """BFS path search from start to target with wall checking.

        This method implements BFS pathfinding with wall avoidance.
        It returns a list of actions (ACTION1-4) from start to target.

        Args:
            start: (col, row) start position.
            target: (col, row) target position.
            walls: Set of (col, row) wall positions.

        Returns:
            List of action IDs (int) from start to target, or None if no path.
        """
        from collections import deque

        # Use ONLY the passed walls (from _plan_bfs, which passes empty set).
        # _blocked_positions is NOT merged here — it grows too large and
        # blocks valid paths. Instead, walls are learned at the _plan_bfs
        # level: when BFS fails, we try the next target before clearing.
        all_walls = walls | self._blocked_positions

        # Don't block the target position.
        all_walls.discard(target)
        # Also don't block positions adjacent to target (within MOVE_STEP)
        # so BFS can reach the target even if a wall was mistakenly added.
        tx, ty = target
        for dx in range(-self.MOVE_STEP, self.MOVE_STEP + 1, self.MOVE_STEP):
            for dy in range(-self.MOVE_STEP, self.MOVE_STEP + 1, self.MOVE_STEP):
                all_walls.discard((tx + dx, ty + dy))

        # BFS queue: (position, path)
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            (cx, cy), path = queue.popleft()

            # Check if reached target (within MOVE_STEP tolerance).
            # Using tolerance handles minor grid alignment differences
            # between player center and target centroid.
            if abs(cx - target[0]) <= self.MOVE_STEP and abs(cy - target[1]) <= self.MOVE_STEP:
                return path if path else None  # Empty path = already at target

            # Expand neighbors
            for action_id, (name, (dx, dy)) in self.KEYBOARD_ACTIONS.items():
                nx, ny = cx + dx * self.MOVE_STEP, cy + dy * self.MOVE_STEP

                # Boundary check
                if nx < 0 or nx >= 64 or ny < 0 or ny >= 64:
                    continue

                # Wall check (static + dynamic)
                if (nx, ny) in all_walls:
                    continue

                new_pos = (nx, ny)
                if new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, path + [action_id]))

        return None  # No path found

    def _block_other_switchers(
        self,
        walls: set[tuple[int, int]],
        switchers: list,
        target_type: Optional[str] = None,
    ) -> set[tuple[int, int]]:
        """Add non-target switcher positions to walls (MNQ Gate hardening).

        This method adapts Oracle v17's _block_other_switchers() logic.
        It prevents BFS paths from crossing non-target switchers, which
        could trigger unintended attribute changes during execution.

        Args:
            walls: Set of (col, row) wall positions (modified in-place).
            switchers: List of switcher GameObjects.
            target_type: "rot"/"shape"/"color"/None. If None, block ALL switchers.

        Returns:
            walls (modified in-place, also returned for chaining).
        """
        for sw in switchers:
            if not sw.is_switcher:
                continue

            sw_type = sw.switcher_type
            sw_pos = (int(sw.centroid[0]), int(sw.centroid[1]))

            # Block this switcher if it's not the target type
            if target_type is None or sw_type != target_type:
                walls.add(sw_pos)

        return walls

    def _plan_bfs(
        self,
        player_pos: tuple[int, int],
        target_obj: Any,
        objects: list,
        available_actions: list[int],
    ) -> Optional[int]:
        """Plan action using BFS path search.

        Uses BFS to find a path from player to target, considering walls
        (both static from GridParser and dynamic from _blocked_positions).
        Switchers are NOT blocked — the player can walk through them.
        The cycle strategy in _plan_manhattan handles any unintended
        attribute changes from walking through switchers.

        Wall positions are extracted from GameObjects' centroids (center
        coordinates), consistent with the player's center position. BFS
        moves in steps of MOVE_STEP (5) between grid-aligned center positions.

        Args:
            player_pos: (col, row) player center position.
            target_obj: Target GameObject.
            objects: List of GameObjects (including walls and switchers).
            available_actions: Available action IDs.

        Returns:
            Action ID (int), or None if BFS fails.
        """
        # NO static walls — rely solely on dynamic wall learning.
        # The GridParser's wall detection (color 4 sprites) is unreliable
        # and can block valid paths. Instead, walls are learned at runtime
        # when the player tries to move and fails (mark_blocked).
        walls: set[tuple[int, int]] = set()

        # NOTE: Switchers are NOT blocked. The player can walk through
        # switchers (they are not collidable in LS20). The cycle strategy
        # in _plan_manhattan handles unintended attribute changes.

        # BFS path search
        target_pos = (int(target_obj.centroid[0]), int(target_obj.centroid[1]))
        path = self._bfs_path(player_pos, target_pos, walls)

        if path and len(path) > 0:
            # Always return the BFS action — it's computed from the current
            # position, so _failed_directions (which is global) shouldn't
            # override it. If the player can't move in that direction,
            # mark_blocked will add the wall and BFS will recalculate.
            return path[0]
        return None

    def _find_nearest_non_goal_target(
        self,
        player_pos: tuple[int, int],
        objects: list,
    ) -> Optional[Any]:
        """Find the nearest non-goal, non-wall target (e.g., switcher, refill).

        This method is used when the goal is blocked (player attributes don't
        match goal requirements). It finds the nearest switcher or other
        interactive object that the player can visit to change its attributes.

        Priority:
        1. Switcher objects (is_switcher=True)
        2. Any non-wall, non-player, non-goal, non-UI object with area >= 2

        Distance is measured as Manhattan distance (|dx| + |dy|).

        Args:
            player_pos: (cx, cy) = (col, row) player position.
            objects: List of GameObjects.

        Returns:
            The nearest non-goal target GameObject, or None.
        """
        px, py = player_pos

        # Strategy 1: Find switcher objects
        switchers: list = [o for o in objects if o.is_switcher and not o.is_player]
        if switchers:
            best = min(
                switchers,
                key=lambda o: abs(o.centroid[0] - px) + abs(o.centroid[1] - py),
            )
            return best

        # Strategy 2: Any non-wall, non-player, non-goal, non-UI object
        best: Optional[Any] = None
        best_dist: float = float("inf")
        for obj in objects:
            if getattr(obj, "is_player", False):
                continue
            if getattr(obj, "is_wall", False):
                continue
            if getattr(obj, "is_goal", False):
                continue
            if obj.area < 2:
                continue
            if obj.color in self.UI_COLORS:
                continue
            ox, oy = obj.centroid
            dist: float = abs(ox - px) + abs(oy - py)
            if dist < best_dist:
                best_dist = dist
                best = obj
        return best
