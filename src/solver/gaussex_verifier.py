"""GaussEx fiber verification: demo constraints as Willems behavioral fibers.

TOMAS v2.2: Numba JIT grid comparison in verify_program.
TOMAS v2.3: Batch verification interface for GPU acceleration.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode

# Numba-accelerated grid comparison
try:
    from src.core.numba_kernels import HAS_NUMBA, grid_equal_kernel
except ImportError:
    HAS_NUMBA = False

# CUDA batch verification (v2.3)
try:
    from src.core.cuda_kernels import HAS_CUDA, CudaBatchVerifier
except ImportError:
    HAS_CUDA = False
    CudaBatchVerifier = None  # type: ignore[misc, assignment]


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
        # Cache: (program_id, pair_idx) -> fiber set
        self._fiber_cache: dict[tuple[int, int], set[int]] = {}

    def verify_program(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> bool:
        """Verify a program against all demo pairs (JIT grid comparison).

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
                    if HAS_NUMBA:
                        if not grid_equal_kernel(predicted, output_grids[i]):
                            return False
                    elif not np.array_equal(predicted, output_grids[i]):
                        return False
                except Exception:
                    return False
        return True

    @staticmethod
    def _fast_array_hash(arr: np.ndarray) -> int:
        """Fast hash of numpy array using CRC32 (faster than Python hash for bytes).

        Args:
            arr: Input numpy array.

        Returns:
            Integer hash.
        """
        import zlib
        return zlib.crc32(arr.tobytes())

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

        # Compute fiber for each program using cached results where possible
        program_fibers: list[set[int]] = []
        for prog_idx, program in enumerate(programs):
            fiber: set[int] = set()
            for pair_idx, pair in enumerate(self.demo_pairs):
                cache_key = (id(program), pair_idx)
                if cache_key in self._fiber_cache:
                    fiber |= self._fiber_cache[cache_key]
                    continue
                sub_fiber: set[int] = set()
                input_grids = pair.get("input", [])
                for grid in input_grids:
                    try:
                        result = program.apply(grid)
                        state_hash = self._fast_array_hash(result)
                        sub_fiber.add(state_hash)
                    except Exception:
                        continue
                self._fiber_cache[cache_key] = sub_fiber
                fiber |= sub_fiber
            program_fibers.append(fiber)

        # Intersection
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
        """Set the demo pairs for verification and clear caches.

        Args:
            demo_pairs: List of demo pairs.
        """
        self.demo_pairs = demo_pairs
        self.fibers = {}
        self._fiber_cache = {}

    def verify_program_batch(
        self,
        programs: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
        cuda_verifier: Any = None,
    ) -> list[ProgramNode]:
        """Batch verify multiple programs against demo pairs.

        v2.3: When a CudaBatchVerifier is provided, uses GPU batch
        comparison for accelerated verification. Falls back to serial
        verification otherwise.

        Args:
            programs: List of ProgramNodes to verify.
            demo_pairs: List of demo pairs.
            cuda_verifier: Optional CudaBatchVerifier for GPU acceleration.

        Returns:
            List of programs that pass all demo verifications.
        """
        if not programs:
            return []

        # GPU batch path
        if cuda_verifier is not None and hasattr(
            cuda_verifier, "batch_grid_equal"
        ):
            return self._verify_batch_gpu(
                programs, demo_pairs, cuda_verifier
            )

        # CPU serial path
        valid: list[ProgramNode] = []
        for program in programs:
            if self.verify_program(program, demo_pairs):
                valid.append(program)
        return valid

    def _verify_batch_gpu(
        self,
        programs: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
        cuda_verifier: Any,
    ) -> list[ProgramNode]:
        """GPU-accelerated batch verification.

        Args:
            programs: List of ProgramNodes to verify.
            demo_pairs: List of demo pairs.
            cuda_verifier: CudaBatchVerifier instance.

        Returns:
            List of valid programs.
        """
        candidate_mask = np.ones(len(programs), dtype=bool)

        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])

            for grid_idx, input_grid in enumerate(input_grids):
                if grid_idx >= len(output_grids):
                    continue

                expected = output_grids[grid_idx]

                # Apply all programs to this input
                predictions: list[np.ndarray] = []
                for program in programs:
                    try:
                        pred = program.apply(input_grid)
                        predictions.append(
                            np.asarray(pred, dtype=np.int8)
                        )
                    except Exception:
                        predictions.append(
                            np.zeros_like(input_grid, dtype=np.int8)
                        )

                # Check shape consistency
                shapes = {p.shape for p in predictions}
                if (
                    len(shapes) > 1
                    or predictions[0].shape != expected.shape
                ):
                    for i, pred in enumerate(predictions):
                        if candidate_mask[i]:
                            if not np.array_equal(pred, expected):
                                candidate_mask[i] = False
                    continue

                pred_batch = np.stack(predictions)
                exp_batch = np.expand_dims(
                    expected.astype(np.int8), axis=0
                )

                equal_matrix = cuda_verifier.batch_grid_equal(
                    pred_batch, exp_batch
                )
                candidate_mask &= equal_matrix[:, 0]

                if not np.any(candidate_mask):
                    return []

        return [
            programs[i] for i in range(len(programs))
            if candidate_mask[i]
        ]

    def compute_fiber_for_pair(
        self,
        program: ProgramNode,
        pair_idx: int,
    ) -> set[int]:
        """Compute the behavioral fiber for a specific demo pair.

        Uses cached results when available.

        Args:
            program: ProgramNode to evaluate.
            pair_idx: Index of the demo pair.

        Returns:
            Set of state hashes in the fiber.
        """
        if pair_idx >= len(self.demo_pairs):
            return set()

        cache_key = (id(program), pair_idx)
        if cache_key in self._fiber_cache:
            return self._fiber_cache[cache_key]

        pair = self.demo_pairs[pair_idx]
        input_grids = pair.get("input", [])
        fiber: set[int] = set()

        for grid in input_grids:
            try:
                result = program.apply(grid)
                fiber.add(self._fast_array_hash(result))
            except Exception:
                continue

        self._fiber_cache[cache_key] = fiber
        return fiber
