"""
StateTracker - Track game state changes across frames.

Records (state, action, next_state) tuples and identifies:
- Player movement direction
- Whether levels were completed
- Whether game was won/lost
- Stagnation detection (no progress)
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from collections import deque

import numpy as np


@dataclass
class FrameSnapshot:
    """A snapshot of game state at a point in time.

    Attributes:
        step: Step index.
        grid: 64x64 grid copy.
        levels_completed: Number of levels completed.
        win_levels: Total levels to win.
        game_state: GameState enum value.
        available_actions: List of available action IDs.
        action_taken: The GameAction value taken, if any.
        player_pos: (x, y) player position, if detected.
        player_color: Player object color, if detected.
    """

    step: int
    grid: np.ndarray
    levels_completed: int
    win_levels: int
    game_state: Any
    available_actions: list[int]
    action_taken: Optional[int] = None
    player_pos: Optional[tuple[int, int]] = None
    player_color: Optional[int] = None


class StateTracker:
    """Track game state transitions across frames.

    Maintains a history of FrameSnapshots and provides utilities for
    stagnation detection and progress tracking.

    Attributes:
        history: Deque of FrameSnapshot objects.
    """

    def __init__(self, history_size: int = 500) -> None:
        """Initialize the state tracker.

        Args:
            history_size: Maximum number of snapshots to retain.
        """
        self.history: deque[FrameSnapshot] = deque(maxlen=history_size)
        self._prev_player_pos: Optional[tuple[int, int]] = None
        self._stagnation_count: int = 0
        self._last_progress_step: int = 0

    def update(
        self,
        step: int,
        grid: np.ndarray,
        levels_completed: int,
        win_levels: int,
        game_state: Any,
        available_actions: list[int],
        action_taken: Optional[int] = None,
        player_pos: Optional[tuple[int, int]] = None,
        player_color: Optional[int] = None,
    ) -> FrameSnapshot:
        """Record a new frame snapshot and return it.

        Args:
            step: Current step index.
            grid: 64x64 grid array.
            levels_completed: Number of levels completed.
            win_levels: Total levels to win.
            game_state: GameState enum value.
            available_actions: List of available action IDs.
            action_taken: The action ID that was taken, if any.
            player_pos: (x, y) player position, if detected.
            player_color: Player object color, if detected.

        Returns:
            The created FrameSnapshot.
        """
        snapshot = FrameSnapshot(
            step=step,
            grid=grid.copy(),
            levels_completed=levels_completed,
            win_levels=win_levels,
            game_state=game_state,
            available_actions=list(available_actions),
            action_taken=action_taken,
            player_pos=player_pos,
            player_color=player_color,
        )

        # Track progress
        if self.history:
            prev = self.history[-1]
            if levels_completed > prev.levels_completed:
                self._last_progress_step = step
                self._stagnation_count = 0
            else:
                self._stagnation_count += 1

        # Track player movement
        if player_pos is not None:
            self._prev_player_pos = player_pos

        self.history.append(snapshot)
        return snapshot

    def get_player_movement_direction(self) -> Optional[tuple[int, int]]:
        """Get the direction the player moved in the last step.

        Returns:
            (dx, dy) movement vector, or None if no movement or insufficient data.
        """
        if len(self.history) < 2:
            return None

        curr = self.history[-1]
        prev = self.history[-2]

        if curr.player_pos is None or prev.player_pos is None:
            return None

        dx = curr.player_pos[0] - prev.player_pos[0]
        dy = curr.player_pos[1] - prev.player_pos[1]

        if dx == 0 and dy == 0:
            return None

        return (dx, dy)

    def is_stuck(self, threshold: int = 50) -> bool:
        """Check if the agent has made no progress for threshold steps.

        Args:
            threshold: Number of steps without progress to be considered stuck.

        Returns:
            True if stuck, False otherwise.
        """
        return self._stagnation_count >= threshold

    def get_steps_since_progress(self) -> int:
        """Get number of steps since last level completion.

        Returns:
            Number of steps since last progress.
        """
        return self._stagnation_count

    def get_total_actions(self) -> int:
        """Get total number of actions taken.

        Returns:
            Total action count.
        """
        return len(self.history)

    def get_last_snapshot(self) -> Optional[FrameSnapshot]:
        """Get the most recent snapshot.

        Returns:
            The last FrameSnapshot, or None if empty.
        """
        return self.history[-1] if self.history else None

    def get_levels_progress(self) -> tuple[int, int]:
        """Get (levels_completed, win_levels) from latest snapshot.

        Returns:
            Tuple of (levels_completed, win_levels).
        """
        if not self.history:
            return (0, 0)
        snap = self.history[-1]
        return (snap.levels_completed, snap.win_levels)

    def get_color_history(self) -> list[int]:
        """Get the history of dominant player colors.

        Returns:
            List of player color values.
        """
        return [s.player_color for s in self.history if s.player_color is not None]

    def reset(self) -> None:
        """Reset the tracker for a new game."""
        self.history.clear()
        self._prev_player_pos = None
        self._stagnation_count = 0
        self._last_progress_step = 0
