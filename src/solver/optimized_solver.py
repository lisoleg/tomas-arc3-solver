"""GPU-optimized solver: adaptive batch, AMP, frame pre-filtering."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode, DSLElement
from src.solver.tomas_solver import TOMASSolver
from src.solver.kappa_snap_searcher import KappaSnapSearcher
from src.solver.gaussex_verifier import GaussExVerifier
from src.utils.gpu_optimizer import GPUOptimizer


class OptimizedSolver:
    """GPU-optimized wrapper around TOMASSolver.

    Integrates GPUOptimizer for adaptive batch sizing, AMP mixed
    precision, gradient checkpointing, and frame pre-filtering
    to maximize throughput on T4/A100 GPUs.

    Attributes:
        solver: Underlying TOMASSolver.
        gpu_optimizer: GPUOptimizer instance.
        batch_size: Current optimal batch size.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the optimized solver.

        Args:
            config: Full configuration dictionary.
        """
        self.solver = TOMASSolver(config)
        self.gpu_optimizer = self.solver.gpu_optimizer
        self.batch_size: int = self.gpu_optimizer.auto_batch_size()

        # Enable optimizations
        self.gpu_optimizer.enable_amp()
        self._amp_context = self.gpu_optimizer.get_amp_context()

    def solve(self, task: dict[str, Any], mode: str = "auto") -> dict[str, Any]:
        """Solve a task with GPU optimizations.

        Pre-filters static frames and uses adaptive batch verification.

        Args:
            task: Raw task dictionary.
            mode: Inference mode.

        Returns:
            Result dictionary with predictions.
        """
        # Pre-filter static frames from test input
        video_task = self.solver.parse_input(task)
        if video_task.test_frames:
            filtered_frames = self.gpu_optimizer.frame_prefilter(video_task.test_frames)
            # Update task with filtered frames
            for i, test_item in enumerate(task.get("test", [])):
                if i < len(filtered_frames):
                    test_item["input"] = [filtered_frames[i].tolist()]

        # Solve with underlying solver
        result = self.solver.solve(task, mode=mode)

        # Add optimization metadata
        result["gpu_optimized"] = True
        result["batch_size"] = self.batch_size
        result["gpu_available"] = self.gpu_optimizer.is_gpu_available
        mem = self.gpu_optimizer.get_memory_usage()
        result["gpu_memory_gb"] = mem.get("allocated_gb", 0.0)

        return result

    def batch_solve(
        self, tasks: list[dict[str, Any]], mode: str = "auto"
    ) -> list[dict[str, Any]]:
        """Solve multiple tasks in batch with GPU optimization.

        Args:
            tasks: List of task dictionaries.
            mode: Inference mode.

        Returns:
            List of result dictionaries.
        """
        results: list[dict[str, Any]] = []
        for task in tasks:
            try:
                result = self.solve(task, mode=mode)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "predictions": [],
                    "mode": mode,
                })

            # Check for OOM and adjust batch size
            if self.gpu_optimizer.is_gpu_available:
                try:
                    import torch
                    if torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                        self.batch_size = self.gpu_optimizer.handle_oom(self.batch_size)
                except Exception:
                    pass

        return results

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get current optimization statistics.

        Returns:
            Dictionary with GPU and optimization stats.
        """
        return {
            "batch_size": self.batch_size,
            "gpu_available": self.gpu_optimizer.is_gpu_available,
            "amp_enabled": self.gpu_optimizer.amp_enabled,
            "checkpoint_enabled": self.gpu_optimizer.checkpoint_enabled,
            "int8_enabled": self.gpu_optimizer.int8_enabled,
            "memory_usage": self.gpu_optimizer.get_memory_usage(),
        }
