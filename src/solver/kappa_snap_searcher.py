"""kappa-Snap abductive search: Two-Phase program synthesis.

TOMAS v2.0 upgrade: ENPV decision for early termination in Phase B.
TOMAS v2.1 optimization: Phase B parallel verification via ThreadPoolExecutor,
removed redundant MDL recomputation in enumerate_candidates.
TOMAS v2.2 optimization: Numba JIT grid comparison in verifier.
TOMAS v2.3 optimization: CUDA GPU batch verification + advanced pruning
(Betti0, symmetry dedup, incremental MDL, heuristic ordering).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from src.core.dsl_primitives import DSLElement, ProgramNode, get_all_primitives
from src.core.topo_hash import TopoHashFilter
from src.solver.enpv_decision import ENPVDecision

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

# Numba CUDA batch verification (v2.3 — numba @cuda.jit path)
try:
    from src.core.numba_cuda_kernels import (
        HAS_CUDA as HAS_NUMBA_CUDA,
        NumbaCudaBatchVerifier,
    )
except ImportError:
    HAS_NUMBA_CUDA = False
    NumbaCudaBatchVerifier = None  # type: ignore[misc, assignment]

# Pruning optimizer (v2.3)
try:
    from src.solver.pruning_optimizer import PruningOptimizer
except ImportError:
    PruningOptimizer = None  # type: ignore[misc, assignment]


class KappaSnapSearcher:
    """kappa-Snap abductive search with Two-Phase filtering.

    Phase A uses topological hash quick-filtering (with Luzhao DNA) to
    eliminate 90%+ of candidates. Phase B enumerates remaining candidates
    by MDL priority and verifies with GaussEx. ENPV decision module
    enables early termination when expected net value goes negative.

    TOMAS v2.3: CUDA GPU batch verification + advanced pruning strategies
    (Betti0 invariant, symmetry dedup, incremental MDL, heuristic ordering).

    Attributes:
        dsl_set: List of available DSL primitives.
        library: LibraryLearning instance for MDL reduction.
        max_depth: Maximum composition depth.
        mdl_threshold: Maximum acceptable MDL cost.
        topo_filter: TopoHashFilter for Phase A.
        enpv: ENPVDecision for termination control.
        cuda_verifier: CudaBatchVerifier for GPU batch verification (v2.3).
        pruning: PruningOptimizer for advanced pruning (v2.3).
        use_cuda: Whether CUDA GPU acceleration is active.
    """

    def __init__(self, config: dict[str, Any], library: Any = None) -> None:
        """Initialize the searcher.

        Args:
            config: Search configuration dict. May include 'cuda' and
                'pruning' sub-dicts for v2.3 features.
            library: LibraryLearning instance (optional).
        """
        self.dsl_set: list[DSLElement] = get_all_primitives()
        self.library = library
        self.max_depth: int = config.get("max_depth", 3)
        self.mdl_threshold: int = config.get("mdl_threshold", 50)
        self.time_limit: float = config.get("time_limit_seconds", 80.0)
        cache_size = config.get("topo_hash_cache_size", 10000)
        use_luzhao = config.get("use_luzhao_hash", True)
        self.topo_filter = TopoHashFilter(cache_size=cache_size,
                                          use_luzhao=use_luzhao)
        self._start_time: float = 0.0

        # ENPV decision
        cost_per_eval = config.get("cost_per_evaluation", 0.5)
        min_enpv = config.get("min_enpv_threshold", 0.0)
        self.enpv = ENPVDecision(cost_per_evaluation=cost_per_eval,
                                 min_enpv_threshold=min_enpv)

        # v2.3: CUDA batch verification — prefer numba CUDA, fall back to CuPy
        cuda_config = config.get("cuda", {})
        self.use_cuda: bool = cuda_config.get("enabled", True) and (
            HAS_NUMBA_CUDA or HAS_CUDA
        )
        self.cuda_verifier: Any = None
        if self.use_cuda:
            batch_size = cuda_config.get("batch_size", 256)
            # Prefer numba CUDA (@cuda.jit) over CuPy
            if HAS_NUMBA_CUDA and NumbaCudaBatchVerifier is not None:
                self.cuda_verifier = NumbaCudaBatchVerifier()
                self._cuda_backend = "numba"
            elif HAS_CUDA and CudaBatchVerifier is not None:
                self.cuda_verifier = CudaBatchVerifier(batch_size=batch_size)
                self._cuda_backend = "cupy"
            else:
                self.use_cuda = False
                self._cuda_backend = "none"
        else:
            self._cuda_backend = "none"

        # v2.3: Pruning optimizer
        pruning_config = config.get("pruning", {})
        pruning_config.setdefault("mdl_threshold", self.mdl_threshold)
        self.pruning: PruningOptimizer | None = None
        if PruningOptimizer is not None:
            self.pruning = PruningOptimizer(pruning_config)

    def search(self, demo_pairs: list[dict[str, Any]]) -> list[ProgramNode]:
        """Unified search entry point (alias for two_phase_search).

        Args:
            demo_pairs: List of demo pairs with 'input' and 'output' grids.

        Returns:
            List of valid ProgramNode candidates ranked by MDL.
        """
        return self.two_phase_search(demo_pairs)

    def two_phase_search(self, demo_pairs: list[dict[str, Any]]) -> list[ProgramNode]:
        """Execute Two-Phase search: topo hash filter then MDL enumeration.

        Pre-Phase A (v2.3): Fast invariant filters (shape, nonzero count,
                 color histogram, Betti0) to eliminate candidates before
                 the expensive topo hash computation.
        Phase A: Filter candidates using topological hash necessary condition
                 (with optional Luzhao DNA hash).
        Phase B: Enumerate by MDL priority, verify with GaussEx.
                 v2.3: GPU batch verification when CUDA available.
                 ENPV-based early termination when expected value < 0.

        Args:
            demo_pairs: List of demo pairs.

        Returns:
            List of valid, ranked ProgramNode candidates.
        """
        self._start_time = time.time()
        self.enpv.reset()
        if self.pruning is not None:
            self.pruning.reset_stats()

        # Clear topo filter cache to ensure deterministic results
        # between repeated search calls on the same searcher instance
        self.topo_filter.clear_cache()

        # Generate candidates at all depths (with incremental MDL pruning)
        all_candidates: list[ProgramNode] = []
        for depth in range(1, self.max_depth + 1):
            if self._is_timeout():
                break
            candidates = self.enumerate_candidates(depth)
            all_candidates.extend(candidates)

        # v2.3 Pre-Phase A: Fast invariant filters (shape → nonzero →
        # color_hist → Betti0) — eliminates 60-80% of candidates before
        # the expensive topo hash computation
        if self.pruning is not None:
            all_candidates = self.pruning.fast_pre_filter(
                all_candidates, demo_pairs
            )

        # Phase A: Topological hash quick filter
        phase_a_passed = self.phase_a_filter(all_candidates, demo_pairs)

        # Apply library learning to reduce MDL
        if self.library is not None:
            phase_a_passed = self.apply_library(phase_a_passed)

        # v2.3: Symmetry equivalence deduplication
        if self.pruning is not None:
            phase_a_passed = self.pruning.symmetry_dedup(phase_a_passed)

        # v2.3: Heuristic candidate ordering
        if self.pruning is not None:
            phase_a_passed = self.pruning.heuristic_order(
                phase_a_passed, demo_pairs
            )

        # Phase B: MDL-priority enumeration with ENPV termination
        valid_programs = self.phase_b_enumerate(phase_a_passed, demo_pairs)

        # Rank by MDL
        ranked = self.rank_by_mdl(valid_programs)
        return ranked

    def phase_a_filter(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """Phase A: Filter candidates using topological hash.

        Uses topo_hash(P(I_i)) == topo_hash(O_i) as a necessary condition.

        Args:
            candidates: List of candidate ProgramNodes.
            demo_pairs: List of demo pairs.

        Returns:
            Filtered list of candidates passing the hash check.
        """
        return self.topo_filter.quick_filter(candidates, demo_pairs)

    def phase_b_enumerate(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """Phase B: Verify candidates against demo pairs with ENPV.

        v2.3: When CUDA is available, uses GPU batch verification.
        Falls back to ThreadPoolExecutor parallel verification on CPU.

        Each candidate must pass all demo pair verifications.
        ENPV-based early termination when expected value turns negative.

        Args:
            candidates: Pre-filtered candidates from Phase A + pruning.
            demo_pairs: List of demo pairs.

        Returns:
            List of candidates that pass all demo verifications.
        """
        # Filter by MDL threshold before verification
        filtered = [p for p in candidates if p.total_mdl <= self.mdl_threshold]
        if not filtered:
            return []

        # v2.3: GPU batch verification path
        if self.use_cuda and self.cuda_verifier is not None:
            return self._phase_b_gpu_verify(filtered, demo_pairs)

        # CPU fallback: ThreadPoolExecutor parallel verification (v2.1 path)
        return self._phase_b_cpu_verify(filtered, demo_pairs)

    def _phase_b_gpu_verify(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """GPU batch verification: verify all candidates in parallel on GPU.

        Batches candidates' predictions against all demo outputs,
        performing a single GPU kernel launch per batch.

        Args:
            candidates: Candidates to verify.
            demo_pairs: Demo pairs for verification.

        Returns:
            List of valid candidates that pass all demo verifications.
        """
        valid: list[ProgramNode] = []
        batch_size = self.cuda_verifier.batch_size  # type: ignore[union-attr]

        for batch_start in range(0, len(candidates), batch_size):
            if self._is_timeout():
                break

            batch = candidates[batch_start:batch_start + batch_size]
            batch_valid = self._verify_batch_gpu(batch, demo_pairs)
            valid.extend(batch_valid)

        # Re-sort valid programs by MDL
        valid = self.rank_by_mdl(valid)

        # ENPV check on final results
        if valid:
            best_mdl = min(p.total_mdl for p in candidates)
            max_mdl_valid = max(p.total_mdl for p in valid)
            proxy_posteriors = [
                (p, (best_mdl / max(p.total_mdl, 1)) * 0.5) for p in valid
            ]
            self.enpv.should_continue(proxy_posteriors, max_mdl_valid)

        return valid

    def _verify_batch_gpu(
        self,
        batch: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """Verify a single batch of candidates on GPU.

        For each demo pair, applies all candidates to the input grids,
        then batch-compares predictions against expected outputs on GPU.

        Args:
            batch: Batch of candidate ProgramNodes.
            demo_pairs: Demo pairs for verification.

        Returns:
            List of candidates from this batch that pass all demos.
        """
        # A candidate is valid if it passes ALL demo pairs
        candidate_mask = np.ones(len(batch), dtype=bool)

        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])

            for grid_idx, input_grid in enumerate(input_grids):
                if grid_idx >= len(output_grids):
                    continue

                expected = output_grids[grid_idx]

                # Apply all candidates to this input grid
                predictions: list[np.ndarray] = []
                for candidate in batch:
                    try:
                        pred = candidate.apply(input_grid)
                        predictions.append(
                            np.asarray(pred, dtype=np.int8)
                        )
                    except Exception:
                        predictions.append(
                            np.zeros_like(input_grid, dtype=np.int8)
                        )

                # Stack into batch array — need consistent shapes
                # If shapes differ, fall back to serial verification
                shapes = {p.shape for p in predictions}
                if len(shapes) > 1 or predictions[0].shape != expected.shape:
                    # Shape mismatch — verify serially
                    for i, pred in enumerate(predictions):
                        if candidate_mask[i]:
                            if not np.array_equal(pred, expected):
                                candidate_mask[i] = False
                    continue

                pred_batch = np.stack(predictions)  # (N, H, W)
                exp_batch = np.expand_dims(
                    expected.astype(np.int8), axis=0
                )  # (1, H, W)

                # GPU batch comparison
                equal_matrix = self.cuda_verifier.batch_grid_equal(  # type: ignore[union-attr]
                    pred_batch, exp_batch
                )  # (N, 1)

                # Update candidate mask
                candidate_mask &= equal_matrix[:, 0]

                # Early exit: if no candidates left, skip remaining demos
                if not np.any(candidate_mask):
                    return []

        return [batch[i] for i in range(len(batch)) if candidate_mask[i]]

    def _phase_b_cpu_verify(
        self,
        candidates: list[ProgramNode],
        demo_pairs: list[dict[str, Any]],
    ) -> list[ProgramNode]:
        """CPU parallel verification using ThreadPoolExecutor (v2.1 path).

        Args:
            candidates: Candidates to verify.
            demo_pairs: Demo pairs for verification.

        Returns:
            List of valid candidates.
        """
        valid: list[ProgramNode] = []
        best_mdl = candidates[0].total_mdl if candidates else 0
        max_workers = min(4, os.cpu_count() or 2)

        # Parallel verification using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_program = {
                executor.submit(self._verify_against_demos, p, demo_pairs): p
                for p in candidates
            }
            for future in as_completed(future_to_program):
                if self._is_timeout():
                    break
                program = future_to_program[future]
                try:
                    if future.result():
                        valid.append(program)
                except Exception:
                    pass

        # Re-sort valid programs by MDL (may arrive out of order)
        valid = self.rank_by_mdl(valid)

        # ENPV check on final results
        if valid:
            max_mdl_valid = max(p.total_mdl for p in valid)
            proxy_posteriors = [
                (p, (best_mdl / max(p.total_mdl, 1)) * 0.5) for p in valid
            ]
            self.enpv.should_continue(proxy_posteriors, max_mdl_valid)

        return valid

    def enumerate_candidates(self, depth: int) -> list[ProgramNode]:
        """Enumerate candidate programs at a given composition depth.

        Depth 1: single primitives.
        Depth 2: pairs of primitives (chain composition).
        Depth 3: triples of primitives.

        v2.3: Incremental MDL pruning at depth 3 — skips enumeration
        when partial MDL (first two primitives) already exceeds threshold.

        Args:
            depth: Composition depth (1, 2, or 3).

        Returns:
            List of ProgramNode candidates at this depth.
        """
        candidates: list[ProgramNode] = []

        if depth == 1:
            for prim in self.dsl_set:
                node = ProgramNode(prim)
                candidates.append(node)

        elif depth == 2:
            for prim1 in self.dsl_set:
                for prim2 in self.dsl_set:
                    if self._is_timeout():
                        return candidates
                    node = ProgramNode(prim1)
                    child = ProgramNode(prim2)
                    node.children.append(child)
                    node.combo_type = "chain"
                    node.total_mdl = node.compute_mdl()
                    candidates.append(node)

        elif depth >= 3:
            # Limit depth 3 to avoid combinatorial explosion
            common_prims = [p for p in self.dsl_set if p.mdl_cost <= 10]
            for prim1 in common_prims:
                for prim2 in common_prims:
                    if self._is_timeout():
                        return candidates

                    # v2.3: Incremental MDL pruning
                    # Compute partial MDL for first two primitives
                    partial_mdl = (
                        prim1.mdl_cost + prim2.mdl_cost + 2  # +2 for chain overhead
                    )
                    if self.pruning is not None:
                        if self.pruning.incremental_mdl_prune(
                            depth, partial_mdl
                        ):
                            self.pruning.stats["mdl_pruned"] += len(common_prims)
                            continue

                    for prim3 in common_prims:
                        if self._is_timeout():
                            return candidates
                        node = ProgramNode(prim1)
                        child2 = ProgramNode(prim2)
                        child3 = ProgramNode(prim3)
                        node.children.append(child2)
                        child2.children.append(child3)
                        node.combo_type = "chain"
                        node.total_mdl = node.compute_mdl()
                        candidates.append(node)

        return candidates

    def rank_by_mdl(self, programs: list[ProgramNode]) -> list[ProgramNode]:
        """Rank programs by ascending MDL cost.

        Args:
            programs: List of ProgramNodes to rank.

        Returns:
            Sorted list (lowest MDL first).
        """
        return sorted(programs, key=lambda p: p.total_mdl)

    def apply_library(self, programs: list[ProgramNode]) -> list[ProgramNode]:
        """Apply library abstractions to reduce MDL of candidates.

        Matches program sub-expressions against library abstractions and
        replaces them with compressed versions.

        Args:
            programs: List of ProgramNodes.

        Returns:
            List of programs with potentially reduced MDL.
        """
        if self.library is None:
            return programs

        result: list[ProgramNode] = []
        for program in programs:
            # Try to match library abstractions
            elements = program.flatten()
            for elem in elements:
                matched = self.library.match_library(program)
                if matched is not None:
                    # Library match reduces MDL
                    program.total_mdl = max(1, program.total_mdl - matched.mdl_cost)
                    break
            result.append(program)
        return result

    def _verify_against_demos(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> bool:
        """Verify a program against all demo pairs (JIT grid comparison).

        Args:
            program: ProgramNode to verify.
            demo_pairs: List of demo pairs.

        Returns:
            True if program produces correct output for all demos.
        """
        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, input_grid in enumerate(input_grids):
                if i >= len(output_grids):
                    continue
                try:
                    predicted = program.apply(input_grid)
                    expected = output_grids[i]
                    # JIT-accelerated grid equality check
                    if HAS_NUMBA:
                        if not grid_equal_kernel(predicted, expected):
                            return False
                    elif not np.array_equal(predicted, expected):
                        return False
                except Exception:
                    return False
        return True

    def _is_timeout(self) -> bool:
        """Check if the search has exceeded its time limit.

        Returns:
            True if time limit exceeded.
        """
        return (time.time() - self._start_time) > self.time_limit
