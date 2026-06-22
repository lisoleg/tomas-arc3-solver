"""A100 batch verifier for static grid verification."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode
from src.utils.gpu_optimizer import GPUOptimizer


class A100Verifier:
    """Batch verifier for static grid program validation.

    Verifies candidate programs against demo pairs in batch,
    using GPU acceleration when available.

    Attributes:
        batch_size: Batch size for parallel verification.
        gpu_optimizer: GPUOptimizer for memory management.
        use_amp: Whether to use mixed precision.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the verifier.

        Args:
            config: Configuration with GPU parameters.
        """
        self.gpu_optimizer = GPUOptimizer(config.get("gpu", {}))
        self.batch_size: int = self.gpu_optimizer.auto_batch_size()
        self.use_amp: bool = self.gpu_optimizer.amp_enabled

    def batch_verify(
        self,
        programs: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[bool]:
        """Batch-verify programs against demo pairs.

        Each program is applied to all demo inputs and checked
        against expected outputs.

        Args:
            programs: List of ProgramNode candidates.
            demo_pairs: List of demo pairs with 'input' and 'output'.

        Returns:
            List of boolean pass/fail for each program.
        """
        results: list[bool] = []

        # Process in batches
        for batch_start in range(0, len(programs), self.batch_size):
            batch = programs[batch_start:batch_start + self.batch_size]
            for program in batch:
                passes = True
                for pair in demo_pairs:
                    input_grids = pair.get("input", [])
                    output_grids = pair.get("output", [])
                    for i, input_grid in enumerate(input_grids):
                        if i >= len(output_grids):
                            continue
                        try:
                            predicted = program.apply(input_grid)
                            if not np.array_equal(predicted, output_grids[i]):
                                passes = False
                                break
                        except Exception:
                            passes = False
                            break
                    if not passes:
                        break
                results.append(passes)

        return results

    def parallel_validate(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[tuple[ProgramNode, bool, float]]:
        """Validate candidates in parallel with match scoring.

        Args:
            candidates: List of ProgramNode candidates.
            demo_pairs: List of demo pairs.

        Returns:
            List of (program, passes, match_score) tuples.
        """
        results: list[tuple[ProgramNode, bool, float]] = []

        for program in candidates:
            total_match = 0.0
            total_count = 0
            passes = True

            for pair in demo_pairs:
                input_grids = pair.get("input", [])
                output_grids = pair.get("output", [])
                for i, input_grid in enumerate(input_grids):
                    if i >= len(output_grids):
                        continue
                    try:
                        predicted = program.apply(input_grid)
                        expected = output_grids[i]

                        if predicted.shape == expected.shape:
                            match_ratio = 1.0 - float(
                                np.sum(predicted != expected) / max(predicted.size, 1)
                            )
                            total_match += match_ratio
                            if match_ratio < 1.0:
                                passes = False
                        else:
                            total_match += 0.0
                            passes = False
                    except Exception:
                        total_match += 0.0
                        passes = False
                    total_count += 1

            avg_match = total_match / max(total_count, 1)
            results.append((program, passes, avg_match))

        return results

    def verify_single(
        self,
        program: ProgramNode,
        input_grid: np.ndarray,
        expected_output: np.ndarray,
    ) -> bool:
        """Verify a single program against a single input-output pair.

        Args:
            program: ProgramNode to verify.
            input_grid: Input grid.
            expected_output: Expected output grid.

        Returns:
            True if program produces expected output.
        """
        try:
            predicted = program.apply(input_grid)
            return np.array_equal(predicted, expected_output)
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get verifier statistics.

        Returns:
            Dictionary with batch size and GPU info.
        """
        return {
            "batch_size": self.batch_size,
            "use_amp": self.use_amp,
            "gpu_available": self.gpu_optimizer.is_gpu_available,
            "memory_usage": self.gpu_optimizer.get_memory_usage(),
        }
