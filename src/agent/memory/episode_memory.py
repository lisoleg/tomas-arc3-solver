"""
EpisodeMemory - Record and replay game episodes.

Stores frame-by-frame records of gameplay for:
- Pattern extraction (repeating action sequences)
- Strategy evaluation (which actions lead to progress)
- Stagnation detection (no progress for N steps)
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from collections import deque

import numpy as np


@dataclass
class EpisodeStep:
    """A single step in an episode.

    Attributes:
        step: Step index.
        action: GameAction value (int).
        grid: Grid before action (copy).
        levels_completed: Levels completed at this step.
        game_state: GameState enum value.
        player_pos: (x, y) player position, if detected.
        moved: Whether the action resulted in player movement.
        progress: Whether levels_completed increased.
    """

    step: int
    action: int
    grid: Optional[np.ndarray] = None
    levels_completed: int = 0
    game_state: Any = None
    player_pos: Optional[tuple[int, int]] = None
    moved: bool = False
    progress: bool = False


class EpisodeMemory:
    """Memory of a game episode for learning and adaptation.

    Stores EpisodeStep records and provides utilities for stuck detection,
    action history retrieval, and successful sequence extraction.

    Attributes:
        steps: Deque of EpisodeStep objects.
    """

    def __init__(self, max_steps: int = 500) -> None:
        """Initialize the episode memory.

        Args:
            max_steps: Maximum number of steps to store.
        """
        self.steps: deque[EpisodeStep] = deque(maxlen=max_steps)
        self._total_actions: int = 0
        self._resets: int = 0
        self._level_progress: list[int] = []
        self._macros: list[dict] = []

        # Trajectory recording for Sleep-Step defensive learning.
        # Each entry is a list of EpisodeStep objects for one level attempt.
        self.failures: list[list[EpisodeStep]] = []
        self.successes: list[list[EpisodeStep]] = []
        self._current_traj: list[EpisodeStep] = []

    def record(
        self,
        step: int,
        action: int,
        grid: Optional[np.ndarray] = None,
        levels_completed: int = 0,
        game_state: Any = None,
        player_pos: Optional[tuple[int, int]] = None,
        moved: bool = False,
        progress: bool = False,
    ) -> None:
        """Record a single episode step.

        Args:
            step: Step index.
            action: GameAction value (int).
            grid: Grid before action.
            levels_completed: Levels completed at this step.
            game_state: GameState enum value.
            player_pos: (x, y) player position, if detected.
            moved: Whether the action resulted in player movement.
            progress: Whether levels_completed increased.
        """
        ep_step = EpisodeStep(
            step=step,
            action=action,
            grid=grid.copy() if grid is not None else None,
            levels_completed=levels_completed,
            game_state=game_state,
            player_pos=player_pos,
            moved=moved,
            progress=progress,
        )
        self.steps.append(ep_step)
        self._total_actions += 1

        # Append to current trajectory for Sleep-Step analysis.
        self._current_traj.append(ep_step)

        if action == 0:  # GameAction.RESET
            self._resets += 1

        if progress:
            self._level_progress.append(step)

    def is_stuck(self, threshold: int = 50) -> bool:
        """Check if no progress has been made for threshold steps.

        Args:
            threshold: Number of steps without progress to be considered stuck.

        Returns:
            True if stuck, False otherwise.
        """
        if not self.steps:
            return False
        last = self.steps[-1]
        if not self._level_progress:
            return last.step >= threshold
        return (last.step - self._level_progress[-1]) >= threshold

    def get_action_sequence(self) -> list[int]:
        """Get the sequence of actions taken.

        Returns:
            List of action IDs in order.
        """
        return [s.action for s in self.steps]

    def get_successful_sequences(self) -> list[list[int]]:
        """Get action sequences that led to level completion.

        Returns:
            List of action ID sequences, one per level completion.
        """
        sequences = []
        start = 0
        for i, step in enumerate(self.steps):
            if step.progress:
                sequences.append([s.action for s in list(self.steps)[start : i + 1]])
                start = i + 1
        return sequences

    def get_stats(self) -> dict:
        """Get episode statistics.

        Returns:
            Dictionary with total_actions, resets, levels_completed, etc.
        """
        return {
            "total_actions": self._total_actions,
            "resets": self._resets,
            "levels_completed": self.steps[-1].levels_completed if self.steps else 0,
            "level_progress_steps": list(self._level_progress),
        }

    def reset(self) -> None:
        """Reset memory for a new episode."""
        self.steps.clear()
        self._total_actions = 0
        self._resets = 0
        self._level_progress.clear()
        self._macros.clear()
        self.failures.clear()
        self.successes.clear()
        self._current_traj.clear()

    # ------------------------------------------------------------------
    # Trajectory recording for Sleep-Step defensive learning
    # ------------------------------------------------------------------

    def start_trajectory(self) -> None:
        """Begin a new trajectory for the current level attempt.

        Clears the current trajectory buffer so that subsequent
        :meth:`record` calls append to a fresh trajectory.
        """
        self._current_traj = []

    def end_trajectory(self, success: bool, info: str = "") -> None:
        """End the current trajectory and archive it.

        The current trajectory is moved to either ``self.successes``
        or ``self.failures`` based on the ``success`` flag, then the
        buffer is cleared.

        Args:
            success: ``True`` if the trajectory ended in level
                completion (success), ``False`` if it ended in
                game-over or stagnation (failure).
            info: Optional description of the outcome (e.g.
                ``"LEVEL_COMPLETE"``, ``"GAME_OVER"``).
        """
        if not self._current_traj:
            return
        traj_copy = list(self._current_traj)
        if success:
            self.successes.append(traj_copy)
        else:
            self.failures.append(traj_copy)
        self._current_traj = []

    # ------------------------------------------------------------------
    # Library Learning: macro extraction and failure analysis
    # ------------------------------------------------------------------

    def extract_macros(
        self,
        min_length: int = 3,
        min_repeats: int = 2,
    ) -> list[dict]:
        """Extract repeating action subsequences as macro actions.

        Scans all successful action sequences (from
        :meth:`get_successful_sequences`) for subsequences of length
        >= ``min_length`` that appear at least ``min_repeats`` times.
        Each discovered macro is recorded internally via
        :meth:`record_macro` and returned.

        Args:
            min_length: Minimum length of a subsequence to qualify as a macro.
            min_repeats: Minimum number of times a subsequence must appear
                across all successful sequences.

        Returns:
            List of macro dicts, each with keys:
            ``"pattern"`` (list[int]), ``"count"`` (int), and
            ``"context"`` (str, always ``"level_completion"``).
        """
        sequences: list[list[int]] = self.get_successful_sequences()
        if not sequences:
            return []

        # Build a frequency map of all subsequences of length >= min_length.
        subseq_counts: dict[tuple[int, ...], int] = {}

        for seq in sequences:
            seq_len = len(seq)
            for length in range(min_length, seq_len + 1):
                for start in range(seq_len - length + 1):
                    subseq = tuple(seq[start:start + length])
                    subseq_counts[subseq] = (
                        subseq_counts.get(subseq, 0) + 1
                    )

        # Filter by minimum repeat count and extract macros.
        macros: list[dict] = []
        for pattern_tuple, count in subseq_counts.items():
            if count >= min_repeats:
                macro = {
                    "pattern": list(pattern_tuple),
                    "count": count,
                    "context": "level_completion",
                }
                macros.append(macro)
                # Auto-record discovered macros.
                self._macros.append(macro)

        # Sort by count descending, then by pattern length descending.
        macros.sort(key=lambda m: (-m["count"], -len(m["pattern"])))
        return macros

    def get_failure_patterns(self) -> list[dict]:
        """Identify positions where actions repeatedly fail (moved=False).

        Scans the episode history for steps where ``moved=False`` and
        groups them by player position.  Positions with multiple failed
        actions indicate walls or obstacles.

        Returns:
            List of failure-pattern dicts, each with keys:
            ``"position"`` (tuple[int, int]) and
            ``"blocked_actions"`` (list[int], sorted unique).
        """
        # Map position → set of blocked actions.
        position_blocks: dict[tuple[int, int], set[int]] = {}

        for step in self.steps:
            if step.moved:
                continue
            if step.player_pos is None:
                continue
            pos = step.player_pos
            if pos not in position_blocks:
                position_blocks[pos] = set()
            position_blocks[pos].add(step.action)

        # Convert to list of dicts.
        patterns: list[dict] = []
        for pos, actions in position_blocks.items():
            patterns.append({
                "position": pos,
                "blocked_actions": sorted(actions),
            })

        return patterns

    def record_macro(self, pattern: list[int], name: str = "") -> None:
        """Manually register a macro action.

        Args:
            pattern: List of action IDs forming the macro.
            name: Optional human-readable name for the macro.
        """
        macro: dict = {
            "pattern": list(pattern),
            "count": 1,
            "context": name if name else "manual",
        }
        self._macros.append(macro)

    def get_macros(self) -> list[dict]:
        """Return all discovered and manually recorded macros.

        Returns:
            List of macro dicts (shallow copies to prevent external
            mutation).
        """
        return [dict(m) for m in self._macros]
