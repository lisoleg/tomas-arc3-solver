"""TensorBoard monitoring metric recording."""
from __future__ import annotations

from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class TensorBoardMonitor:
    """Records solver metrics to TensorBoard for real-time monitoring.

    Tracks GPU memory, solve time, search filter counts, Bayesian posteriors,
    fusion component scores, library size, and batch verification success rate.
    """

    def __init__(self, log_dir: str = "runs/tomas_arc3", flush_secs: int = 10) -> None:
        """Initialize the TensorBoard monitor.

        Args:
            log_dir: Directory for TensorBoard event files.
            flush_secs: How often to flush data to disk (seconds).
        """
        self.log_dir = log_dir
        self.flush_secs = flush_secs
        self._writer: Any = None
        self._step = 0

    @property
    def writer(self) -> Any:
        """Lazily initialize and return the SummaryWriter.

        Returns:
            SummaryWriter instance, or None if TensorBoard unavailable.
        """
        if self._writer is None and _TB_AVAILABLE:
            self._writer = SummaryWriter(self.log_dir, flush_secs=self.flush_secs)
        return self._writer

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Log a scalar metric.

        Args:
            tag: Metric tag (e.g. "gpu/allocated_gb").
            value: Scalar value to log.
            step: Global step (defaults to internal counter).
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step if step is not None else self._step)

    def log_gpu_memory(self, allocated_gb: float, reserved_gb: float) -> None:
        """Log GPU memory usage metrics.

        Args:
            allocated_gb: Allocated GPU memory in GB.
            reserved_gb: Reserved GPU memory in GB.
        """
        self.log_scalar("gpu/allocated_gb", allocated_gb)
        self.log_scalar("gpu/reserved_gb", reserved_gb)

    def log_solve_time(self, total_time_s: float) -> None:
        """Log total solve time for a task.

        Args:
            total_time_s: Solve time in seconds.
        """
        self.log_scalar("solve/total_time_s", total_time_s)

    def log_search_stats(self, phase_a_filtered: int, phase_b_validated: int) -> None:
        """Log search phase statistics.

        Args:
            phase_a_filtered: Number of candidates filtered by Phase A.
            phase_b_validated: Number of candidates validated in Phase B.
        """
        self.log_scalar("search/phase_a_filtered", phase_a_filtered)
        self.log_scalar("search/phase_b_validated", phase_b_validated)

    def log_bayesian_posterior(self, posteriors: list[float]) -> None:
        """Log Bayesian posterior distribution as a histogram.

        Args:
            posteriors: List of posterior probability values.
        """
        if self.writer is not None and _TB_AVAILABLE:
            import numpy as np
            self.writer.add_histogram(
                "bayesian/posterior", np.array(posteriors), self._step
            )

    def log_fusion_scores(
        self, g_sym: float, g_vis: float, g_cr: float
    ) -> None:
        """Log fusion component scores.

        Args:
            g_sym: Symbolic consistency score.
            g_vis: Visual feature score.
            g_cr: Cross-modal consistency score.
        """
        self.log_scalar("fusion/G_sym", g_sym)
        self.log_scalar("fusion/G_vis", g_vis)
        self.log_scalar("fusion/G_cr", g_cr)

    def log_library_size(self, size: int) -> None:
        """Log the current Library Learning abstraction count.

        Args:
            size: Number of abstractions in the library.
        """
        self.log_scalar("library/size", size)

    def log_batch_success_rate(self, success_rate: float) -> None:
        """Log batch verification success rate.

        Args:
            success_rate: Fraction of candidates that passed verification.
        """
        self.log_scalar("batch/success_rate", success_rate)

    def step(self) -> None:
        """Increment the internal step counter."""
        self._step += 1

    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
