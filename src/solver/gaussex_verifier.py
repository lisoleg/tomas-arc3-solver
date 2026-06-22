"""GaussEx fiber verification: demo constraints as Willems behavioral fibers."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode


class GaussExVerifier:
    """GaussEx interconnected verification using Willems behavioral fibers.

    Each demo pair defines a behavioral fiber constraint. A candidate
    program must pass all demo pair fibers (fiber intersection minimization).

    Attributes:
        demo_pairs: List of demo pairs for verification.
        fibers: Computed fiber sets per demo pair.
    """

    def __init__(self) -> None:
        """Initialize the verifier."""
        self.demo_pairs: list[dict[str, Any]] = []
        self.fibers: dict[int, set[int]] = {}

    def verify_program(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> bool:
        """Verify a program against all demo pairs.

        A program is valid if it produces correct output for every
        demo pair (fiber intersection is non-empty for all constraints).

        Args:
            program: ProgramNode to verify.
            demo_pairs: List of demo pairs with 'input' and 'output'.

        Returns:
            True if program satisfies all demo constraints.
        """
        for pair_idx, pair in enumerate(demo_pairs):
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, input_grid in enumerate(input_grids):
                if i >= len(output_grids):
                    continue
                try:
                    predicted = program.apply(input_grid)
                    if not np.array_equal(predicted, output_grids[i]):
                        return False
                except Exception:
                    return False
        return True

    def compute_fiber_intersection(
        self,
        programs: list[ProgramNode],
    ) -> set[int]:
        """Compute the fiber intersection across multiple programs.

        The fiber intersection contains state hashes that are valid
        across all programs' behavioral fibers.

        Args:
            programs: List of ProgramNodes.

        Returns:
            Set of state hashes in the fiber intersection.
        """
        if not programs:
            return set()

        # Compute fiber for each program
        program_fibers: list[set[int]] = []
        for program in programs:
            fiber: set[int] = set()
            for pair in self.demo_pairs:
                input_grids = pair.get("input", [])
                for grid in input_grids:
                    try:
                        result = program.apply(grid)
                        state_hash = hash(result.tobytes())
                        fiber.add(state_hash)
                    except Exception:
                        continue
            program_fibers.append(fiber)

        # Intersection
        if not program_fibers:
            return set()

        intersection = program_fibers[0]
        for fiber in program_fibers[1:]:
            intersection = intersection & fiber

        return intersection

    def cross_video_transfer(
        self,
        programs: list[ProgramNode],
        other_videos: list[list[dict[str, Any]]],
    ) -> list[ProgramNode]:
        """Transfer programs across videos via fiber intersection.

        Only programs that satisfy constraints across multiple videos
        are considered transferable.

        Args:
            programs: Source video programs.
            other_videos: List of other video demo pairs.

        Returns:
            List of programs that transfer successfully.
        """
        transferable: list[ProgramNode] = []

        for program in programs:
            all_pass = True
            for video_demos in other_videos:
                if not self.verify_program(program, video_demos):
                    all_pass = False
                    break
            if all_pass:
                transferable.append(program)

        return transferable

    def minimize_fiber_intersection(
        self,
        candidates: list[ProgramNode],
    ) -> ProgramNode | None:
        """Select the candidate with minimal fiber intersection.

        The candidate whose behavioral fiber is most constrained
        (smallest intersection) is preferred, as it is most specific.

        Args:
            candidates: List of candidate ProgramNodes.

        Returns:
            The candidate with minimal fiber intersection, or None.
        """
        if not candidates:
            return None

        best_program: ProgramNode | None = None
        min_size = float("inf")

        for program in candidates:
            fiber = self.compute_fiber_intersection([program])
            if len(fiber) < min_size:
                min_size = len(fiber)
                best_program = program

        return best_program

    def set_demo_pairs(self, demo_pairs: list[dict[str, Any]]) -> None:
        """Set the demo pairs for verification.

        Args:
            demo_pairs: List of demo pairs.
        """
        self.demo_pairs = demo_pairs
        self.fibers = {}

    def compute_fiber_for_pair(
        self,
        program: ProgramNode,
        pair_idx: int,
    ) -> set[int]:
        """Compute the behavioral fiber for a specific demo pair.

        Args:
            program: ProgramNode to evaluate.
            pair_idx: Index of the demo pair.

        Returns:
            Set of state hashes in the fiber.
        """
        if pair_idx >= len(self.demo_pairs):
            return set()

        pair = self.demo_pairs[pair_idx]
        input_grids = pair.get("input", [])
        fiber: set[int] = set()

        for grid in input_grids:
            try:
                result = program.apply(grid)
                fiber.add(hash(result.tobytes()))
            except Exception:
                continue

        return fiber
