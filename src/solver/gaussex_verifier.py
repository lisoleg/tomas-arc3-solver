"""GaussEx fiber verification: demo constraints as Willems behavioral fibers.

TOMAS v2.2: Numba JIT grid comparison in verify_program.
TOMAS v2.3: Batch verification interface for GPU acceleration.
TOMAS v3.1: Dead-Zero gating (information fidelity threshold) + MUS
(Mutual Exclusion System) dual-storage for conflicting hypotheses.
"""
from __future__ import annotations

import hashlib
import time
from collections import defaultdict
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

# CuPy batch verification (v2.9) — GPU-accelerated batch grid comparison
try:
    import cupy as _cp  # type: ignore[import-untyped]
    _HAS_CUPY: bool = True
except ImportError:
    _HAS_CUPY = False
    _cp = None  # type: ignore[assignment]


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

        # v3.1: Dead-Zero gating
        self.theta_dead: float = 0.15  # Information fidelity dead threshold
        self._fidelity_cache: dict[int, float] = {}  # id(program) -> fidelity
        self._dead_zero_rejects: int = 0  # Counter for rejected programs

        # v3.1: MUS (Mutual Exclusion System) dual-storage
        self.mus_log: list[dict[str, Any]] = []  # MUS candidate pairs
        self._mus_detection_threshold: float = 0.4  # Min divergence to flag MUS
        self._mus_max_retained: int = 20  # Max MUS pairs to keep

        # v3.1: Psi anchor audit
        self.psi_anchors: list[dict[str, Any]] = []  # ψ audit trail
        self._psi_anchor_counter: int = 0

    def verify_program(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
        pre_computed: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> bool:
        """Verify a program against all demo pairs (JIT grid comparison).

        A program is valid if it produces correct output for every
        demo pair (fiber intersection is non-empty for all constraints).

        Args:
            program: ProgramNode to verify.
            demo_pairs: List of demo pairs with 'input' and 'output'.
            pre_computed: Optional dict mapping (pair_idx, input_idx)
                to pre-computed predicted output. If provided,
                skips program.apply() and uses cached output directly.

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
                    if pre_computed is not None:
                        pred_key = (pair_idx, i)
                        if pred_key in pre_computed:
                            predicted = pre_computed[pred_key]
                        else:
                            predicted = program.apply(input_grid)
                    else:
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
        comparison for accelerated verification.
        v2.9: When CuPy is available (and no cuda_verifier), uses CuPy
        for GPU-accelerated batch grid comparison. Falls back to CPU
        serial verification otherwise.

        Priority: cuda_verifier (Numba CUDA) > CuPy > CPU serial.

        Args:
            programs: List of ProgramNodes to verify.
            demo_pairs: List of demo pairs.
            cuda_verifier: Optional CudaBatchVerifier for GPU acceleration.

        Returns:
            List of programs that pass all demo verifications.
        """
        if not programs:
            return []

        # GPU batch path — Numba CUDA (highest priority)
        if cuda_verifier is not None and hasattr(
            cuda_verifier, "batch_grid_equal"
        ):
            return self._verify_batch_gpu(
                programs, demo_pairs, cuda_verifier
            )

        # GPU batch path — CuPy (v2.9)
        if _HAS_CUPY:
            return self._verify_batch_cupy(
                programs, demo_pairs
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

    def _verify_batch_cupy(
        self,
        programs: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """CuPy-accelerated batch verification (v2.9).

        Uploads all predicted and expected grids to GPU memory as CuPy
        arrays and performs batch element-wise comparison. This is
        significantly faster than serial CPU verification when there are
        many programs with same-shaped outputs.

        Falls back to CPU serial path if shapes are inconsistent (CuPy
        requires same-shape arrays for stacking).

        Args:
            programs: List of ProgramNodes to verify.
            demo_pairs: List of demo pairs.

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

                expected = np.asarray(
                    output_grids[grid_idx], dtype=np.int8
                )

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
                            np.zeros_like(
                                np.asarray(input_grid, dtype=np.int8)
                            )
                        )

                # Check shape consistency — CuPy requires same shape
                # for stacking. If shapes differ, fall back to numpy.
                shapes = {p.shape for p in predictions}
                if (
                    len(shapes) > 1
                    or predictions[0].shape != expected.shape
                ):
                    # Shape mismatch — use numpy comparison
                    for i, pred in enumerate(predictions):
                        if candidate_mask[i]:
                            if not np.array_equal(pred, expected):
                                candidate_mask[i] = False
                    continue

                # CuPy batch comparison: upload to GPU, compare all at once
                try:
                    pred_batch_cp = _cp.asarray(
                        np.stack(predictions)
                    )
                    exp_batch_cp = _cp.asarray(
                        np.expand_dims(expected, axis=0)
                    )
                    # Broadcast comparison: (N, H, W) == (1, H, W) → (N, H, W)
                    equal_all = _cp.all(
                        pred_batch_cp == exp_batch_cp,
                        axis=(1, 2),
                    )
                    # Transfer result back to CPU
                    equal_np = _cp.asnumpy(equal_all).astype(bool)
                    candidate_mask &= equal_np
                except Exception:
                    # CuPy failure — fall back to numpy
                    for i, pred in enumerate(predictions):
                        if candidate_mask[i]:
                            if not np.array_equal(pred, expected):
                                candidate_mask[i] = False

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

    # =========================================================================
    # v3.1: Dead-Zero Information Fidelity Gating
    # =========================================================================

    def _calc_fidelity(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> float:
        """Compute information fidelity of a program against demo pairs.

        Fidelity measures how much information the program preserves
        about the input-output transformation, as opposed to hallucinating
        or producing degenerate outputs.

        The metric combines three signals:
        1. **Structural fidelity**: How well output structure matches input
           (shape preservation, color count ratio)
        2. **Predictive fidelity**: How consistent predictions are across
           demo pairs (Jitter-based sigma)
        3. **Topological fidelity**: Betti0 preservation ratio

        Fidelity ∈ [0, 1]. Programs with fidelity < theta_dead are
        considered "Dead-Zero" — information-destructive and rejected.

        This implements the Dead-Zero gate from the TOMAS theory:
            "Low-fidelity programs correspond to hallucinatory reasoning;
             they must be gated before consuming verification budget."
        — GaussEx + Lean4 Dual Verification (Dead-Zero safety)

        Args:
            program: ProgramNode to evaluate.
            demo_pairs: Demo pairs for fidelity computation.

        Returns:
            Fidelity score in [0, 1].
        """
        # Check cache
        prog_id = id(program)
        if prog_id in self._fidelity_cache:
            return self._fidelity_cache[prog_id]

        if not demo_pairs:
            return 1.0  # No demos to check against

        struct_scores: list[float] = []
        pred_scores: list[float] = []
        topo_scores: list[float] = []

        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, input_grid in enumerate(input_grids):
                if i >= len(output_grids):
                    continue
                try:
                    predicted = program.apply(input_grid)
                    expected = output_grids[i]
                    inp = np.asarray(input_grid)
                    pred = np.asarray(predicted)
                    exp = np.asarray(expected)

                    # 1. Structural fidelity: shape match
                    if pred.shape == exp.shape:
                        struct_scores.append(1.0)
                    else:
                        struct_scores.append(0.0)
                        pred_scores.append(0.0)
                        topo_scores.append(0.0)
                        continue

                    # 2. Predictive fidelity: pixel match ratio
                    match_ratio = float(np.mean(pred == exp))
                    # Degenerate output check: all zeros or all same color
                    is_degenerate = (
                        np.all(pred == 0)
                        or len(np.unique(pred)) == 1
                    )
                    if is_degenerate:
                        match_ratio *= 0.3  # Heavy penalty for degenerate output
                    pred_scores.append(match_ratio)

                    # 3. Topological fidelity: Betti0 ratio
                    inp_betti0 = _count_connected_components(inp)
                    pred_betti0 = _count_connected_components(pred)
                    exp_betti0 = _count_connected_components(exp)

                    # How close is predicted Betti0 to expected?
                    betti0_range = max(abs(inp_betti0 - exp_betti0), 1)
                    betti0_error = abs(pred_betti0 - exp_betti0) / betti0_range
                    topo_scores.append(max(0.0, 1.0 - betti0_error))

                except Exception:
                    struct_scores.append(0.0)
                    pred_scores.append(0.0)
                    topo_scores.append(0.0)

        if not struct_scores:
            return 0.0

        # Weighted fidelity score
        struct_fidelity = float(np.mean(struct_scores)) if struct_scores else 0.0
        pred_fidelity = float(np.mean(pred_scores)) if pred_scores else 0.0
        topo_fidelity = float(np.mean(topo_scores)) if topo_scores else 0.0

        # Fidelity = 0.3*structural + 0.5*predictive + 0.2*topological
        fidelity = 0.3 * struct_fidelity + 0.5 * pred_fidelity + 0.2 * topo_fidelity

        # Cache result
        self._fidelity_cache[prog_id] = fidelity
        return fidelity

    def _dead_zero_gate(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> tuple[bool, float]:
        """Dead-Zero gate: reject programs with critically low fidelity.

        This is the primary safety mechanism against hallucinatory programs.
        When fidelity < theta_dead, the program is considered information-
        destructive and must be rejected regardless of formal verification.

        Returns (passed, fidelity):
            - passed: True if fidelity >= theta_dead
            - fidelity: The computed fidelity score

        Args:
            program: ProgramNode to gate.
            demo_pairs: Demo pairs for fidelity computation.

        Returns:
            Tuple of (passed, fidelity_score).
        """
        fidelity = self._calc_fidelity(program, demo_pairs)
        passed = fidelity >= self.theta_dead

        if not passed:
            self._dead_zero_rejects += 1

        return passed, fidelity

    # =========================================================================
    # v3.1: MUS (Mutual Exclusion System) Dual-Storage
    # =========================================================================

    def _detect_mutual_exclusion(
        self,
        prog_a: ProgramNode,
        prog_b: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> bool:
        """Detect if two programs are mutually exclusive.

        Two programs are mutually exclusive if they produce conflicting
        outputs for the same input — i.e., both cannot be simultaneously
        correct. This triggers MUS dual-storage: keep both as alternative
        hypotheses instead of prematurely discarding one.

        Detection criteria:
        1. Both programs pass verification individually
        2. Outputs diverge on at least one demo pair
        3. Divergence exceeds mus_detection_threshold

        Args:
            prog_a: First program.
            prog_b: Second program.
            demo_pairs: Demo pairs for comparison.

        Returns:
            True if mutual exclusion is detected.
        """
        divergences: list[float] = []

        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            for grid in input_grids:
                try:
                    out_a = np.asarray(prog_a.apply(grid))
                    out_b = np.asarray(prog_b.apply(grid))

                    if out_a.shape != out_b.shape:
                        # Different output shapes — strong divergence
                        divergences.append(1.0)
                        continue

                    # Pixel-level divergence
                    divergence = float(np.mean(out_a != out_b))
                    divergences.append(divergence)
                except Exception:
                    # If either fails, they can't be directly compared
                    divergences.append(0.5)

        if not divergences:
            return False

        mean_divergence = float(np.mean(divergences))
        return mean_divergence >= self._mus_detection_threshold

    def record_mus_pair(
        self,
        prog_a: ProgramNode,
        prog_b: ProgramNode,
        confidence_a: float = 0.5,
        confidence_b: float = 0.5,
        tag: str = "",
    ) -> str:
        """Record a MUS (mutually exclusive) hypothesis pair.

        Both programs are preserved as alternative solutions. The MUS
        cell can be resolved later via Bayesian evidence accumulation
        (see psi_fusion_gate.MusCell).

        Args:
            prog_a: First mutually exclusive program.
            prog_b: Second mutually exclusive program.
            confidence_a: Confidence in program A.
            confidence_b: Confidence in program B.
            tag: Optional conflict description tag.

        Returns:
            MUS cell ID (hash).
        """
        mus_entry = {
            "cell_id": hashlib.md5(
                f"{id(prog_a)}{id(prog_b)}{time.time()}".encode()
            ).hexdigest()[:12],
            "prog_a_id": id(prog_a),
            "prog_b_id": id(prog_b),
            "confidence_a": confidence_a,
            "confidence_b": confidence_b,
            "tag": tag,
            "timestamp": time.time(),
            "mdl_a": getattr(prog_a, "total_mdl", 0),
            "mdl_b": getattr(prog_b, "total_mdl", 0),
        }

        self.mus_log.append(mus_entry)

        # Prune old entries if exceeding max
        if len(self.mus_log) > self._mus_max_retained:
            self.mus_log = self.mus_log[-self._mus_max_retained:]

        return mus_entry["cell_id"]

    def get_mus_statistics(self) -> dict[str, Any]:
        """Get MUS dual-storage statistics.

        Returns:
            Dict with mus_count, resolution_stats, etc.
        """
        return {
            "total_mus_pairs": len(self.mus_log),
            "mus_tags": list(set(e["tag"] for e in self.mus_log if e["tag"])),
            "avg_confidence_a": (
                float(np.mean([e["confidence_a"] for e in self.mus_log]))
                if self.mus_log else 0.0
            ),
            "avg_confidence_b": (
                float(np.mean([e["confidence_b"] for e in self.mus_log]))
                if self.mus_log else 0.0
            ),
        }

    # =========================================================================
    # v3.1: Psi Anchor Audit
    # =========================================================================

    def record_psi_anchor(
        self,
        program: ProgramNode,
        fidelity: float,
        decision: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Record a psi audit anchor for this verification step.

        Psi anchors provide an audit trail for every verification decision,
        enabling detection of alignment faking and confidence drift.

        Each anchor records:
            - What program was evaluated
            - Its computed fidelity
            - The decision made (pass/reject/mus)
            - Contextual metadata

        Args:
            program: The program being verified.
            fidelity: Computed fidelity score.
            decision: Decision string ("PASS", "DEAD_ZERO_REJECT",
                "MUS_STORED", "VERIFIED").
            metadata: Optional extra context.

        Returns:
            Anchor index.
        """
        anchor = {
            "anchor_id": self._psi_anchor_counter,
            "program_id": id(program),
            "fidelity": fidelity,
            "decision": decision,
            "mdl": getattr(program, "total_mdl", 0),
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self.psi_anchors.append(anchor)
        self._psi_anchor_counter += 1

        # Prune old anchors
        if len(self.psi_anchors) > 100:
            self.psi_anchors = self.psi_anchors[-100:]

        return anchor["anchor_id"]

    def get_psi_audit_trail(self) -> list[dict[str, Any]]:
        """Get the full psi audit trail.

        Returns:
            List of all recorded psi anchors.
        """
        return list(self.psi_anchors)

    # =========================================================================
    # Dead-Zero + MUS integrated verification
    # =========================================================================

    def verify_with_gating(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
        pre_computed: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> tuple[bool, float, str]:
        """Verify a program with Dead-Zero gating and MUS detection.

        Full pipeline:
        1. Compute information fidelity
        2. Dead-Zero gate: reject if fidelity < theta_dead
        3. Standard GaussEx verification (if passed gate)
        4. Record psi anchor for audit

        Args:
            program: ProgramNode to verify.
            demo_pairs: Demo pairs for verification.
            pre_computed: Optional pre-computed predictions.

        Returns:
            Tuple of (passed, fidelity, decision):
                - passed: True if program is valid
                - fidelity: Computed fidelity score
                - decision: "PASS", "DEAD_ZERO_REJECT", or "FAILED_VERIFY"
        """
        # Step 1: Dead-Zero gate
        passed_gate, fidelity = self._dead_zero_gate(program, demo_pairs)

        if not passed_gate:
            self.record_psi_anchor(
                program, fidelity, "DEAD_ZERO_REJECT",
                {"theta_dead": self.theta_dead},
            )
            return False, fidelity, "DEAD_ZERO_REJECT"

        # Step 2: Standard verification
        verified = self.verify_program(program, demo_pairs, pre_computed)

        if verified:
            self.record_psi_anchor(
                program, fidelity, "PASS",
                {"verified": True},
            )
            return True, fidelity, "PASS"
        else:
            self.record_psi_anchor(
                program, fidelity, "FAILED_VERIFY",
                {"verified": False},
            )
            return False, fidelity, "FAILED_VERIFY"

    def verify_batch_with_gating(
        self,
        programs: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
        cuda_verifier: Any = None,
    ) -> tuple[list[ProgramNode], list[dict[str, Any]]]:
        """Batch verify with Dead-Zero gating + MUS detection.

        Modified batch verification that:
        1. Pre-filters programs via Dead-Zero gate
        2. Standard batch verification for surviving programs
        3. Detects mutually exclusive (MUS) pairs among valid programs
        4. Records psi audit anchors throughout

        Args:
            programs: List of ProgramNodes to verify.
            demo_pairs: Demo pairs for verification.
            cuda_verifier: Optional CUDA batch verifier.

        Returns:
            Tuple of (valid_programs, mus_pairs):
                - valid_programs: Programs passing all checks
                - mus_pairs: Detected MUS hypothesis pairs
        """
        if not programs:
            return [], []

        # Step 1: Dead-Zero pre-filtering
        survivors: list[ProgramNode] = []
        for prog in programs:
            passed, fidelity = self._dead_zero_gate(prog, demo_pairs)
            if passed:
                survivors.append(prog)

        if not survivors:
            return [], []

        # Step 2: Standard batch verification
        valid = self.verify_program_batch(survivors, demo_pairs, cuda_verifier)

        # Step 3: MUS detection among valid programs
        mus_pairs: list[dict[str, Any]] = []
        if len(valid) >= 2:
            # Only check pairs with close MDL (likely competitors)
            sorted_valid = sorted(valid, key=lambda p: p.total_mdl)
            for i in range(len(sorted_valid)):
                for j in range(i + 1, min(i + 4, len(sorted_valid))):
                    if self._detect_mutual_exclusion(
                        sorted_valid[i], sorted_valid[j], demo_pairs
                    ):
                        cell_id = self.record_mus_pair(
                            sorted_valid[i], sorted_valid[j],
                            confidence_a=0.6, confidence_b=0.4,
                            tag="batch_verification_mus",
                        )
                        mus_pairs.append({
                            "cell_id": cell_id,
                            "prog_a_mdl": sorted_valid[i].total_mdl,
                            "prog_b_mdl": sorted_valid[j].total_mdl,
                        })

        return valid, mus_pairs

    def reset_gate_stats(self) -> None:
        """Reset Dead-Zero gate and MUS statistics."""
        self._dead_zero_rejects = 0
        self._fidelity_cache.clear()
        self.mus_log.clear()
        self.psi_anchors.clear()
        self._psi_anchor_counter = 0


# =============================================================================
# Helper: Connected Components (for Betti0)
# =============================================================================

def _count_connected_components(grid: np.ndarray) -> int:
    """Count 4-connected components (Betti0) in a grid.

    Args:
        grid: Input numpy array.

    Returns:
        Number of connected components.
    """
    from collections import deque

    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    count = 0

    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0 and not visited[r, c]:
                count += 1
                # BFS
                color = grid[r, c]
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and not visited[nr, nc]
                                and grid[nr, nc] == color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

    return count
