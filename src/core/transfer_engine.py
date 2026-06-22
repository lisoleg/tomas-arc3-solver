"""Cross-video transfer engine using GaussEx fiber intersection."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.core.hypergraph import HyperGraph


class TransferEngine:
    """Transfers general DSL subroutines across videos via fiber intersection.

    Only transfers universal DSL subroutines (via Library Learning),
    not task-specific patterns, to comply with competition rules.

    Attributes:
        learned_programs: List of programs learned from source videos.
        verifier: GaussExVerifier for fiber intersection computation.
    """

    def __init__(self, verifier: Any = None) -> None:
        """Initialize the transfer engine.

        Args:
            verifier: GaussExVerifier instance for fiber computation.
        """
        self.learned_programs: list[ProgramNode] = []
        self.verifier = verifier

    def transfer(
        self,
        programs: list[ProgramNode],
        target_video: list[np.ndarray],
    ) -> list[ProgramNode]:
        """Transfer learned programs to a target video.

        Filters source programs by checking if they produce valid
        transformations on the target video frames.

        Args:
            programs: Source video programs to transfer.
            target_video: Target video frames to apply transferred programs.

        Returns:
            List of programs that successfully transfer to the target.
        """
        transferred: list[ProgramNode] = []

        for program in programs:
            # Check if program is applicable to target video
            score = self.score_transfer(program, target_video)
            if score > 0.0:
                transferred.append(program)

        # Store successfully transferred programs
        self.learned_programs.extend(transferred)
        return transferred

    def compute_cross_video_fiber(
        self,
        source: list[ProgramNode],
        target: list[ProgramNode],
    ) -> set[int]:
        """Compute the fiber intersection between source and target programs.

        The fiber intersection represents transformations that are valid
        in both source and target contexts.

        Args:
            source: Source video program nodes.
            target: Target video program nodes.

        Returns:
            Set of fiber state hashes in the intersection.
        """
        source_fibers: set[int] = set()
        target_fibers: set[int] = set()

        # Compute fibers for source programs
        for program in source:
            elements = program.flatten()
            for elem in elements:
                # Use element name as a simple fiber identifier
                source_fibers.add(hash(elem.name))

        # Compute fibers for target programs
        for program in target:
            elements = program.flatten()
            for elem in elements:
                target_fibers.add(hash(elem.name))

        return source_fibers & target_fibers

    def score_transfer(
        self,
        program: ProgramNode,
        target_frames: list[np.ndarray],
    ) -> float:
        """Score how well a program transfers to target video frames.

        Applies the program to target frames and checks if the
        transformation produces a non-trivial result.

        Args:
            program: The program to evaluate.
            target_frames: Target video frames.

        Returns:
            Transfer score in [0, 1]. Higher is better.
        """
        if not target_frames:
            return 0.0

        scores = []
        for frame in target_frames:
            try:
                transformed = program.apply(frame)
                if transformed.shape != frame.shape:
                    scores.append(0.0)
                    continue
                # Score based on how much the program changes the frame
                diff = np.abs(
                    frame.astype(np.float32) - transformed.astype(np.float32)
                )
                change_ratio = float(np.sum(diff > 0) / max(frame.size, 1))
                # Moderate change is good; no change or complete change is bad
                if 0.01 <= change_ratio <= 0.99:
                    scores.append(1.0 - abs(0.5 - change_ratio) * 2)
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    def extract_general_subroutines(
        self,
        programs: list[ProgramNode],
    ) -> list[DSLElement]:
        """Extract general (transferable) DSL subroutines from programs.

        Identifies DSL elements that appear across multiple programs,
        indicating they are general-purpose rather than task-specific.

        Args:
            programs: List of programs to analyze.

        Returns:
            List of general DSLElement subroutines.
        """
        element_counts: dict[str, int] = {}
        element_map: dict[str, DSLElement] = {}

        for program in programs:
            elements = program.flatten()
            seen_names: set[str] = set()
            for elem in elements:
                if elem.name not in seen_names:
                    element_counts[elem.name] = element_counts.get(elem.name, 0) + 1
                    element_map[elem.name] = elem
                    seen_names.add(elem.name)

        # Elements appearing in multiple programs are general
        general: list[DSLElement] = []
        for name, count in element_counts.items():
            if count >= 2:
                general.append(element_map[name])

        return general

    def get_learned_programs(self) -> list[ProgramNode]:
        """Get all learned (transferred) programs.

        Returns:
            List of ProgramNode objects.
        """
        return self.learned_programs.copy()
