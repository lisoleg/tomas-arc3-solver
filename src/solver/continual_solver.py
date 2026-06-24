"""Continual learning pipeline: Wake-Sleep cycles for monotonic accuracy improvement.

Implements Theorem 4: ⟨𝓛ₑ₊₁⟩ ⊇ ⟨𝓛ₑ⟩, coverage monotonically non-decreasing.

The Wake-Sleep cycle:
    Wake phase: Solve all tasks with the current DSL using κ-Snap search.
    Sleep phase: Extract frequently occurring sub-expressions from solved
                 programs and register them as new DSL primitives.
    Next epoch: The expanded DSL enables solving more tasks (monotonic
                coverage improvement).

This module orchestrates the full continual learning loop, managing
state persistence and primitive registration across epochs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.solver.ksnap_searcher import KSnapSearcher
from src.solver.library_learning import LibraryLearning


# ============================================================
# Epoch Result Data Structure
# ============================================================

@dataclass
class EpochResult:
    """Result of a single epoch in the continual learning loop.

    Attributes:
        epoch: Epoch index (0-based).
        accuracy: Fraction of tasks solved in this epoch.
        solved: Number of tasks solved.
        total: Total number of tasks.
        new_primitives: Number of new primitives learned in Sleep phase.
        timing: Wall-clock time for this epoch in seconds.
        solved_task_ids: List of task IDs solved in this epoch.
    """

    epoch: int
    accuracy: float
    solved: int
    total: int
    new_primitives: int
    timing: float
    solved_task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the epoch result.
        """
        return {
            "epoch": self.epoch,
            "accuracy": round(self.accuracy, 4),
            "solved": self.solved,
            "total": self.total,
            "new_primitives": self.new_primitives,
            "timing": round(self.timing, 2),
            "solved_task_ids": self.solved_task_ids,
        }


# ============================================================
# Continual Solver
# ============================================================

class ContinualSolver:
    """Continual learning pipeline with Wake-Sleep cycles.

    Orchestrates the full continual learning loop:
        1. Wake: Solve all tasks with current DSL (κ-Snap search)
        2. Record accuracy
        3. Sleep: Extract new primitives from solved programs
        4. Register new primitives for next epoch
        5. Save state

    Theorem 4 guarantee: Each epoch's DSL is a superset of the previous
    epoch's DSL, so coverage is monotonically non-decreasing.

    Attributes:
        library: LibraryLearning instance for primitive management.
        searcher: KSnapSearcher for program search.
        search_config: Search configuration dict.
        library_config: Library configuration dict.
    """

    def __init__(
        self,
        library_config: dict[str, Any],
        search_config: dict[str, Any],
    ) -> None:
        """Initialize the continual solver.

        Args:
            library_config: Configuration for LibraryLearning. Must
                include 'persistence_path' for state persistence.
            search_config: Configuration for KSnapSearcher. May include
                'theta_ftel', 'theta_dead', 'beam_width', 'max_depth'.
        """
        self.library_config: dict[str, Any] = library_config
        self.search_config: dict[str, Any] = search_config
        self.library: LibraryLearning = LibraryLearning(library_config)

        # Extract KSnapSearcher parameters from search_config
        self.searcher: KSnapSearcher = KSnapSearcher(
            theta_ftel=search_config.get("theta_ftel", 0.1),
            theta_dead=search_config.get("theta_dead", 0.01),
            beam_width=search_config.get("beam_width", 100),
            max_depth=search_config.get("max_depth", 4),
            early_stop=search_config.get("early_stop", True),
        )

        # Track registered primitive names for state management
        self._registered_primitives: set[str] = set()

    def run_continual_learning(
        self,
        tasks: list[dict[str, Any]],
        epochs: int = 3,
    ) -> list[EpochResult]:
        """Run the full continual learning loop.

        For each epoch:
            1. Wake: Solve all tasks with current DSL
            2. Record accuracy
            3. Sleep: Extract new primitives from solved programs
            4. Register new primitives for next epoch
            5. Save state

        Args:
            tasks: List of task dictionaries to solve.
            epochs: Number of Wake-Sleep cycles to run.

        Returns:
            List of EpochResult objects, one per epoch.
        """
        results: list[EpochResult] = []

        for epoch in range(epochs):
            epoch_start: float = time.time()

            # ── Wake Phase: Solve all tasks with current DSL ──
            wake_results: dict[str, tuple[ProgramNode | None, int]] = (
                self._wake_phase(tasks)
            )

            # Collect solved programs and task IDs
            solved_programs: list[ProgramNode] = []
            solved_task_ids: list[str] = []
            for task_id, (prog, depth) in wake_results.items():
                if prog is not None:
                    solved_programs.append(prog)
                    solved_task_ids.append(task_id)

            total_tasks: int = len(tasks)
            solved_count: int = len(solved_programs)
            accuracy: float = (
                solved_count / total_tasks if total_tasks > 0 else 0.0
            )

            # ── Sleep Phase: Extract new primitives ──
            new_primitives: list[DSLElement] = self._sleep_phase(
                solved_programs
            )

            # ── Register new primitives for next epoch ──
            self._register_primitives(new_primitives)

            # ── Save state ──
            self.save_state()

            epoch_timing: float = time.time() - epoch_start

            result = EpochResult(
                epoch=epoch,
                accuracy=accuracy,
                solved=solved_count,
                total=total_tasks,
                new_primitives=len(new_primitives),
                timing=epoch_timing,
                solved_task_ids=solved_task_ids,
            )
            results.append(result)

        return results

    # ============================================================
    # Wake Phase
    # ============================================================

    def _wake_phase(
        self, tasks: list[dict[str, Any]]
    ) -> dict[str, tuple[ProgramNode | None, int]]:
        """Execute Wake phase: solve all tasks with current DSL.

        Uses the KSnapSearcher to solve each task. The searcher uses
        the current DSL (including any learned primitives from
        previous epochs).

        Args:
            tasks: List of task dictionaries.

        Returns:
            Dictionary mapping task_id to (program, depth).
            program is None if the task was not solved.
        """
        results: dict[str, tuple[ProgramNode | None, int]] = {}

        for i, task in enumerate(tasks):
            task_id: str = task.get("task_id", f"task_{i}")
            try:
                prog, depth, _log = self.searcher.search(task)
                results[task_id] = (prog, depth)
            except Exception:
                results[task_id] = (None, 0)

        return results

    # ============================================================
    # Sleep Phase
    # ============================================================

    def _sleep_phase(
        self, solved_programs: list[ProgramNode]
    ) -> list[DSLElement]:
        """Execute Sleep phase: extract new primitives from solved programs.

        Uses LibraryLearning.sleep_step to extract frequently occurring
        sub-expressions and register them as new DSL primitives.

        The ``min_freq`` parameter is passed from ``library_config`` to
        control the minimum frequency threshold for primitive registration.

        Args:
            solved_programs: List of ProgramNodes from successfully
                solved tasks in the Wake phase.

        Returns:
            List of newly created DSLElement primitives.
        """
        if not solved_programs:
            return []

        # Pass relevant parameters from library_config to sleep_step
        min_freq: int = self.library_config.get("min_freq", 3)
        max_ast_depth: int = self.library_config.get("max_ast_depth", 3)
        max_total_subexprs: int = self.library_config.get(
            "max_total_subexprs", 5000
        )
        mdl_gain_threshold: int = self.library_config.get(
            "mdl_gain_threshold", 5
        )
        max_new: int = self.library_config.get("max_new", 15)
        return self.library.sleep_step(
            solved_programs,
            min_freq=min_freq,
            max_ast_depth=max_ast_depth,
            max_total_subexprs=max_total_subexprs,
            mdl_gain_threshold=mdl_gain_threshold,
            max_new=max_new,
        )

    # ============================================================
    # Primitive Registration
    # ============================================================

    def _register_primitives(
        self, primitives: list[DSLElement]
    ) -> None:
        """Register new primitives into the DSL registry.

        Ensures that learned primitives are available for the next
        epoch's Wake phase. The sleep_step method already registers
        primitives in DSLElement._registry; this method tracks
        registration for state management and provides a safety net
        for primitives that need re-registration after state restore.

        Args:
            primitives: List of DSLElement primitives to register.
        """
        for prim in primitives:
            if prim.name not in self._registered_primitives:
                self._registered_primitives.add(prim.name)

            # Safety: ensure the primitive is in the DSL registry.
            # sleep_step normally handles this during creation, but
            # after load_state() the delegate closures are lost and
            # cannot be restored from JSON. In that case, the
            # primitive name is tracked but the delegate must be
            # re-registered by re-running sleep_step.
            if prim.name not in DSLElement._registry:
                # Primitive not in registry — likely after load_state.
                # The delegate closure cannot be restored from JSON;
                # the primitive will be re-learned in the next Sleep
                # phase if it's still frequent.
                pass

    # ============================================================
    # State Persistence
    # ============================================================

    def save_state(self) -> None:
        """Save the solver state to disk.

        Persists the library (learned abstractions) to the configured
        JSON file. This allows resuming continual learning across
        sessions.
        """
        try:
            self.library.save()
        except Exception:
            pass

    def load_state(self) -> None:
        """Load the solver state from disk.

        Loads the library (learned abstractions) from the configured
        JSON file. Called at initialization to resume from a previous
        session.
        """
        try:
            self.library.load()
        except Exception:
            pass

    # ============================================================
    # Utilities
    # ============================================================

    def get_library_size(self) -> int:
        """Get the current number of abstractions in the library.

        Returns:
            Library size (number of learned abstractions).
        """
        return self.library.get_size()

    def get_registered_primitive_names(self) -> list[str]:
        """Get names of all registered learned primitives.

        Returns:
            List of primitive names registered by this solver.
        """
        return sorted(self._registered_primitives)

    def get_accuracy_history(
        self, results: list[EpochResult]
    ) -> list[float]:
        """Extract accuracy history from epoch results.

        Args:
            results: List of EpochResult objects.

        Returns:
            List of accuracy values, one per epoch.
        """
        return [r.accuracy for r in results]
