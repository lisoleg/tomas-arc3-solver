"""GPU memory optimization: adaptive batch size, AMP, gradient checkpointing, INT8."""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class GPUOptimizer:
    """GPU memory optimizer with adaptive batch sizing and mixed precision.

    Implements binary-search batch size probing, automatic mixed precision (AMP),
    gradient checkpointing, INT8 quantization, and static frame pre-filtering.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the GPU optimizer.

        Args:
            config: GPU configuration dict with device, max_vram_gb,
                initial_batch_size, amp_enabled, etc.
        """
        self.device = self._resolve_device(config.get("device", "auto"))
        self.max_vram_gb = config.get("max_vram_gb", 16)
        self.initial_batch_size = config.get("initial_batch_size", 32)
        self.min_batch_size = config.get("min_batch_size", 1)
        self.amp_enabled = config.get("amp_enabled", True)
        self.checkpoint_enabled = config.get("checkpoint_enabled", True)
        self.int8_enabled = config.get("int8_enabled", False)
        self.frame_diff_threshold = config.get("frame_diff_threshold", 0.01)
        self._optimal_batch_size: int | None = None
        self._amp_scaler: Any = None

    def _resolve_device(self, device_str: str) -> str:
        """Resolve the compute device string.

        Args:
            device_str: Device specification ('auto', 'cuda:0', 'cpu', etc.).

        Returns:
            Resolved device string.
        """
        if device_str == "auto":
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda:0"
            return "cpu"
        return device_str

    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available.

        Returns:
            True if CUDA GPU is available.
        """
        return "cuda" in self.device and _TORCH_AVAILABLE and torch.cuda.is_available()

    def auto_batch_size(self, model_size: int = 0) -> int:
        """Determine optimal batch size via binary search probing.

        Starts from initial_batch_size and halves on OOM until a stable
        batch size is found. Uses cached result for subsequent calls.

        Args:
            model_size: Estimated model parameter count (for memory estimation).

        Returns:
            Optimal batch size that fits in available VRAM.
        """
        if self._optimal_batch_size is not None:
            return self._optimal_batch_size

        if not self.is_gpu_available:
            self._optimal_batch_size = self.initial_batch_size
            return self._optimal_batch_size

        batch_size = self.initial_batch_size
        while batch_size > self.min_batch_size:
            try:
                # Probe by allocating a test tensor
                test_shape = (batch_size, 3, 224, 224)
                _ = torch.zeros(test_shape, device=self.device, dtype=torch.float16)
                if _TORCH_AVAILABLE:
                    torch.cuda.synchronize()
                _ = None
                self._optimal_batch_size = batch_size
                return batch_size
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                batch_size = batch_size // 2
                if _TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self._optimal_batch_size = self.min_batch_size
        return self.min_batch_size

    def enable_amp(self) -> Any:
        """Enable automatic mixed precision (AMP).

        Returns:
            AMP gradient scaler if available, None otherwise.
        """
        if not self.is_gpu_available or not self.amp_enabled:
            return None
        if _TORCH_AVAILABLE:
            self._amp_scaler = torch.cuda.amp.GradScaler()
            return self._amp_scaler
        return None

    def enable_checkpoint(self, model: Any = None) -> Any:
        """Enable gradient checkpointing for a model.

        Args:
            model: PyTorch model to wrap with checkpointing.

        Returns:
            Wrapped model or original if checkpointing unavailable.
        """
        if not self.checkpoint_enabled or not _TORCH_AVAILABLE:
            return model

        if model is not None and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return model

    def quantize_int8(self, model: Any) -> Any:
        """Apply INT8 dynamic quantization to a model.

        Args:
            model: PyTorch model to quantize.

        Returns:
            Quantized model, or original if quantization unavailable.
        """
        if not self.int8_enabled or not _TORCH_AVAILABLE:
            return model
        try:
            return torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception:
            return model

    def get_memory_usage(self) -> dict[str, float]:
        """Get current GPU memory usage statistics.

        Returns:
            Dictionary with allocated, reserved, and total memory in GB.
        """
        if not self.is_gpu_available:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}

        if not _TORCH_AVAILABLE:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return {
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "total_gb": round(total, 3),
        }

    def frame_prefilter(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Filter out static (near-identical) frames to reduce computation.

        Frames where ``‖frame_t - frame_{t-1}‖ < threshold`` are considered
        static and skipped (only the first occurrence is kept).

        Args:
            frames: List of frame arrays (T, H, W) or (H, W).

        Returns:
            Filtered list of frames with dynamic content only.
        """
        if len(frames) <= 1:
            return list(frames)

        filtered = [frames[0]]
        for i in range(1, len(frames)):
            prev = frames[i - 1].astype(np.float32)
            curr = frames[i].astype(np.float32)
            if prev.shape != curr.shape:
                filtered.append(frames[i])
                continue
            diff = np.mean(np.abs(curr - prev))
            if diff >= self.frame_diff_threshold:
                filtered.append(frames[i])
        return filtered

    def handle_oom(self, current_batch_size: int) -> int:
        """Handle OOM by halving batch size and clearing cache.

        Args:
            current_batch_size: The batch size that caused OOM.

        Returns:
            Reduced batch size (halved, minimum 1).
        """
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        new_size = max(current_batch_size // 2, self.min_batch_size)
        self._optimal_batch_size = new_size
        return new_size

    def get_amp_context(self) -> Any:
        """Get an AMP autocast context manager.

        Returns:
            Autocast context if GPU+AMP available, nullcontext otherwise.
        """
        import contextlib

        if self.is_gpu_available and self.amp_enabled and _TORCH_AVAILABLE:
            return torch.cuda.amp.autocast()
        return contextlib.nullcontext()
