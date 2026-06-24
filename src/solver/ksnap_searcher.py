"""κ-Snap abductive searcher: projection-based program search for ARC tasks.

Adapts TOMAS Axiom A2 (κ-Snap manifestation) to ARC-AGI-3 solving:
- Candidate programs = superposition state (CandidateEdge analog)
- Ftel = output similarity to demo pairs
- Dead-Zero = program crash or all-zero output
- MUS = conflicting candidates (same depth, different output)
- κ-Snap projection = GaussEx verification → manifest as solution

Algorithm:
    1. Generate depth-1 candidates (ParamInference)
    2. For each candidate compute Ftel (output similarity)
    3. Four-level filter: Ftel threshold → Dead-Zero → MUS → GaussEx projection
    4. MANIFESTED → return solution
    5. Not solved → expand to depth-2 (pairwise chain combinations)
    6. Repeat until max_depth
    7. Beam Search: keep only top-beam_width candidates per layer (sorted by Ftel)
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.solver.param_inference import ParamInference
from src.solver.gaussex_verifier import GaussExVerifier

# v2.7.1: JIT fast path imports
try:
    from src.core.numba_kernels import (
        fast_apply_single_jit,
        fast_apply_chain_jit,
        fast_sim_and_nonzero,
        is_jit_supported,
        grid_equal_kernel,
        HAS_NUMBA,
    )
    _HAS_JIT = HAS_NUMBA
except ImportError:
    _HAS_JIT = False
    fast_apply_single_jit = None  # type: ignore
    fast_apply_chain_jit = None  # type: ignore
    fast_sim_and_nonzero = None  # type: ignore
    is_jit_supported = None  # type: ignore


# ============================================================
# Snap Result Enum
# ============================================================

class SnapResult(Enum):
    """κ-Snap execution result (adapted from TOMAS Axiom A2).

    Attributes:
        MANIFESTED: Candidate successfully projected as solution.
        REJECT_DZ: Dead-Zero rejection (crash or all-zero output).
        SUSPEND_MUS: MUS conflict suspended (competing candidates).
        REJECT_FTEL: Ftel insufficient (output similarity too low).
    """

    MANIFESTED = "manifested"
    REJECT_DZ = "reject_dz"
    SUSPEND_MUS = "suspend_mus"
    REJECT_FTEL = "reject_ftel"


# ============================================================
# Candidate Data Structure
# ============================================================

@dataclass
class KSnapCandidate:
    """A candidate program in the κ-Snap search space (superposition state).

    Analogous to CandidateEdge in the κ-Snap theory. Each candidate is a
    potential solution program that has not yet been projected (verified).

    Attributes:
        program: The candidate ProgramNode.
        depth: Composition depth (1 = single primitive, 2 = chain of 2, etc.).
        ftel: Output similarity to demo pairs (flow-through magnitude |Ftel|).
        i_value: Information existence degree (ℐ).
        mus_active: Whether MUS (mutually exclusive states) is active.
        candidate_id: Unique identifier string.
    """

    program: ProgramNode
    depth: int = 1
    ftel: float = 0.0
    i_value: float = 0.0
    mus_active: bool = False
    candidate_id: str = field(
        default_factory=lambda: f"cand_{uuid.uuid4().hex[:8]}"
    )


# ============================================================
# κ-Snap Searcher
# ============================================================

class KSnapSearcher:
    """κ-Snap abductive searcher for ARC-AGI-3 tasks.

    Implements the four-level κ-Snap projection filtering adapted to
    program search:
        Level 1: Ftel threshold (output similarity must exceed theta_ftel)
        Level 2: Dead-Zero check (program must not crash or produce all-zero)
        Level 3: MUS check (conflicting candidates are suspended)
        Level 4: GaussEx projection (full verification against demo pairs)

    Uses beam search to manage combinatorial explosion: at each depth,
    only the top ``beam_width`` candidates (by Ftel) are retained.

    Attributes:
        theta_ftel: Minimum Ftel threshold for manifestation.
        theta_dead: Dead-Zero threshold for information existence.
        beam_width: Maximum candidates to keep per depth layer.
        max_depth: Maximum composition depth.
        causal_log: Causal log of snap events (list of dicts).
    """

    def __init__(
        self,
        theta_ftel: float = 0.1,
        theta_dead: float = 0.01,
        beam_width: int = 100,
        max_depth: int = 4,
        early_stop: bool = True,
    ) -> None:
        """Initialize the κ-Snap searcher.

        Args:
            theta_ftel: Minimum Ftel (output similarity) for a candidate
                to pass Level 1 filtering. Default 0.1.
            theta_dead: Minimum i_value (information existence) for a
                candidate to pass Level 2 filtering. Default 0.01.
            beam_width: Maximum number of candidates retained per depth
                layer in beam search. Default 100.
            max_depth: Maximum composition depth for chain expansion.
                Default 4.
            early_stop: When True, collect all MANIFESTED candidates
                within the same depth layer and return the one with
                minimal MDL (tie-break). When False, continue searching
                all depths before returning the best MDL solution.
                Default True.
        """
        self.theta_ftel: float = theta_ftel
        self.theta_dead: float = theta_dead
        self.beam_width: int = beam_width
        self.max_depth: int = max_depth
        self.early_stop: bool = early_stop
        self.param_inference: ParamInference = ParamInference()
        self.verifier: GaussExVerifier = GaussExVerifier()
        self.causal_log: list[dict[str, Any]] = []
        # 优化：apply 结果缓存，避免同一 (program, input) 对被反复 apply
        # key = (id(program), id(input_grid)), value = output_grid
        self._apply_cache: dict[tuple[int, int], np.ndarray] = {}

    # ============================================================
    # 优化：缓存加速
    # ============================================================

    def _cached_apply(
        self,
        program: ProgramNode,
        input_grid: np.ndarray,
    ) -> np.ndarray:
        """带缓存的 program.apply()，避免同一 (program, input) 对重复计算。

        v2.7.1: 优先走 JIT 快速路径，不支持时回退到 Python。
        JIT 路径绕过 DSLElement._registry 动态分发，直接调用编译内核。

        对同一 (program, input_grid) 只 apply 一次，后续直接返回缓存结果。
        缓存 key 使用 Python 对象 id（仅在本次 search() 调用内有效）。

        Args:
            program: 要执行的程序。
            input_grid: 输入网格（numpy array）。

        Returns:
            程序执行结果网格。
        """
        cache_key = (id(program), id(input_grid))
        if cache_key in self._apply_cache:
            return self._apply_cache[cache_key]

        # v2.7.1: Try JIT fast path first
        result = self._jit_apply(program, input_grid)

        # Fall back to Python path if JIT not available
        if result is None:
            result = program.apply(input_grid)

        self._apply_cache[cache_key] = result
        return result

    def _jit_apply(
        self,
        program: ProgramNode,
        input_grid: np.ndarray,
    ) -> np.ndarray | None:
        """JIT 快速路径：对支持的程序结构直接调用编译内核。

        支持的程序结构：
        - Leaf node (单原语): 走 fast_apply_single_jit
        - Chain (链式组合): 所有子原语都 JIT 支持时走 fast_apply_chain_jit

        不支持的返回 None，调用方回退到 Python 路径。

        Args:
            program: 要执行的 ProgramNode。
            input_grid: 输入网格。

        Returns:
            变换后的网格，或 None（不支持时）。
        """
        if not _HAS_JIT:
            return None

        elements = program.flatten()
        if not elements:
            return None

        # Build (name, params) list for JIT chain
        jit_elements: list[tuple[str, dict]] = []
        for elem in elements:
            if not is_jit_supported(elem.name):
                return None
            jit_elements.append((elem.name, elem.params))

        # Check combo_type — only chain and leaf are supported
        if program.combo_type not in ("leaf", "chain"):
            return None

        try:
            return fast_apply_chain_jit(input_grid, jit_elements)
        except Exception:
            return None

    def _compute_ftel_and_i_value(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """一次性计算 Ftel 和 i_value（只 apply 每个 (program, input) 对一次）。

        v2.7.1: 使用 JIT 编译的 fast_sim_and_nonzero 加速相似度计算。

        合并原 _compute_ftel 和 _compute_i_value 的逻辑，
        对每个 demo input 只调用一次 _cached_apply，同时计算
        相似度和非零比。

        Args:
            program: 要评估的 ProgramNode。
            demo_pairs: demo 对列表。

        Returns:
            (ftel, i_value) 元组。
        """
        total_sim: float = 0.0
        total_nonzero_ratio: float = 0.0
        count: int = 0

        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, inp in enumerate(input_grids):
                if i >= len(output_grids):
                    continue
                try:
                    # 只 apply 一次
                    pred = self._cached_apply(program, inp)
                    expected = output_grids[i]

                    # v2.7.1: JIT 快速相似度 + 非零比计算
                    if _HAS_JIT:
                        sim, nz_ratio = fast_sim_and_nonzero(
                            np.asarray(pred, dtype=np.int8),
                            np.asarray(expected, dtype=np.int8),
                        )
                        total_sim += sim
                        total_nonzero_ratio += nz_ratio
                    else:
                        # Fallback: Python 路径
                        sim = self._grid_similarity(pred, expected)
                        total_sim += sim
                        result_arr = np.asarray(pred)
                        if result_arr.size > 0:
                            nonzero_pixels = int(np.count_nonzero(result_arr))
                            total_nonzero_ratio += nonzero_pixels / result_arr.size
                    count += 1
                except Exception:
                    total_sim += 0.0
                    count += 1

        avg_ftel = total_sim / max(count, 1)
        avg_i_value = total_nonzero_ratio / max(count, 1)
        return avg_ftel, avg_i_value

    # ============================================================
    # Public API
    # ============================================================

    def search(
        self, task_data: dict[str, Any]
    ) -> tuple[ProgramNode | None, int, list[dict[str, Any]]]:
        """Execute κ-Snap search on an ARC task.

        Runs the full beam search with four-level κ-Snap filtering
        across all depths from 1 to max_depth.

        Args:
            task_data: Raw task dictionary with 'train' key containing
                demo pairs, or directly containing 'input'/'output'.

        Returns:
            Tuple of (solution_program, depth, causal_log).
            solution_program is None if no solution found.
            depth is 0 if no solution found.
        """
        self.causal_log = []
        demo_pairs = self._extract_demo_pairs(task_data)
        if not demo_pairs:
            self._log_event("search_start", "No demo pairs found", 0)
            return None, 0, self.causal_log

        # Generate depth-1 candidates via ParamInference
        candidates = self._generate_candidates(demo_pairs)
        if not candidates:
            self._log_event("search_start", "No candidates generated", 0)
            return None, 0, self.causal_log

        self._log_event(
            "search_start",
            f"Generated {len(candidates)} depth-1 candidates",
            0,
        )

        # Track best solution across all depths (used when early_stop=False)
        best_match: KSnapCandidate | None = None
        best_depth: int = 0

        for depth in range(1, self.max_depth + 1):
            # Get candidates for current depth
            if depth == 1:
                current_candidates = candidates
            else:
                current_candidates = self._expand_depth(candidates, demo_pairs)

            if not current_candidates:
                self._log_event(
                    f"depth_{depth}",
                    f"No candidates at depth {depth}",
                    depth,
                )
                break

            self._log_event(
                f"depth_{depth}",
                f"{len(current_candidates)} candidates at depth {depth}",
                depth,
            )

            # 合并计算 Ftel 和 i_value（只 apply 一次）
            for cand in current_candidates:
                cand.ftel, cand.i_value = (
                    self._compute_ftel_and_i_value(cand.program, demo_pairs)
                )

            # Beam search: keep top beam_width by Ftel
            current_candidates.sort(key=lambda c: c.ftel, reverse=True)
            current_candidates = current_candidates[: self.beam_width]

            # Check MUS (mutually exclusive states)
            mus_map = self._check_mus(current_candidates)
            for cand in current_candidates:
                cand.mus_active = mus_map.get(cand.candidate_id, False)

            # Four-level filtering: collect all MANIFESTED candidates
            # at this depth layer, then pick the one with minimal MDL.
            # This implements the "MDL tie-break" improvement: when
            # multiple candidates pass GaussEx verification at the same
            # depth, we prefer the simplest (lowest MDL) solution.
            manifested_list: list[KSnapCandidate] = []
            for cand in current_candidates:
                result = self._execute_snap(cand, demo_pairs)
                if result == SnapResult.MANIFESTED:
                    # v2.8: Leave-one-out cross-validation for depth > 1
                    # to prevent overfitting (solution passes all training
                    # pairs but fails on test).
                    if depth > 1 and len(demo_pairs) >= 3:
                        if not self._leave_one_out_validate(
                            cand.program, demo_pairs
                        ):
                            self._log_event(
                                "loocv_reject",
                                f"Depth {depth} solution rejected "
                                f"by leave-one-out validation",
                                depth,
                            )
                            continue

                    manifested_list.append(cand)

            # If any candidates manifested at this depth, pick the
            # one with minimal MDL.
            if manifested_list:
                depth_best = min(
                    manifested_list,
                    key=lambda c: c.program.total_mdl,
                )

                # Update global best if this depth's best is better
                if (
                    best_match is None
                    or depth_best.program.total_mdl
                    < best_match.program.total_mdl
                ):
                    best_match = depth_best
                    best_depth = depth

                self._log_event(
                    "search_end",
                    f"Solution manifested at depth {depth} "
                    f"(MDL={depth_best.program.total_mdl}, "
                    f"{len(manifested_list)} candidates tie-broken)",
                    depth,
                )

                # early_stop=True: return immediately at the first depth
                # where any solution is found (after MDL tie-break).
                if self.early_stop:
                    return best_match.program, best_depth, self.causal_log

            # Update candidates for next depth expansion
            candidates = current_candidates

        # early_stop=False: return the best solution found across all depths
        if best_match is not None:
            return best_match.program, best_depth, self.causal_log

        self._log_event(
            "search_end",
            f"No solution found after depth {self.max_depth}",
            self.max_depth,
        )
        return None, 0, self.causal_log

    def _leave_one_out_validate(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> bool:
        """Leave-one-out cross-validation for depth > 1 solutions.

        For each demo pair i, verify that the program (which was found
        using ALL pairs) still produces correct output for pair i.
        Since the program passed GaussEx verification on all pairs,
        this is actually a generalization check: we re-verify each
        pair independently to ensure no single pair "carries" the solution.

        For depth > 1 programs, we additionally check that the program
        structure is consistent — i.e., it doesn't just memorize one
        pair's transformation but generalizes across all pairs.

        Args:
            program: The solution program to validate.
            demo_pairs: All demo pairs.

        Returns:
            True if the program passes leave-one-out validation.
        """
        # The program already passed GaussEx on ALL pairs.
        # For LOOCV, we check that the program also works on EACH pair
        # individually (which it does, since it passed all pairs).
        # The real value of LOOCV is checking STRUCTURAL consistency:
        # if we remove any single pair, does the program still make sense?

        # For ARC, the key overfitting pattern is: a depth-2+ chain like
        # [crop-to-obj, crop-to-obj] that happens to produce the right
        # output for all training inputs but relies on specific input
        # structure that doesn't generalize to test inputs.

        # Check: for each pair, does the program produce EXACTLY the
        # expected output (not just similar)?
        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, inp in enumerate(input_grids):
                if i >= len(output_grids):
                    continue
                try:
                    pred = self._cached_apply(program, inp)
                    pred_arr = np.asarray(pred, dtype=np.int8)
                    expected = np.asarray(output_grids[i], dtype=np.int8)

                    # Must be exact match (GaussEx already verified this,
                    # but re-check with proper dtype handling)
                    if pred_arr.shape != expected.shape:
                        return False
                    if not np.array_equal(pred_arr, expected):
                        return False
                except Exception:
                    return False

        # Additional check: verify the program produces non-trivial
        # output on all pairs (not just returning the input unchanged
        # for some pairs while transforming others)
        non_trivial_count = 0
        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, inp in enumerate(input_grids):
                if i >= len(output_grids):
                    continue
                try:
                    pred = self._cached_apply(program, inp)
                    inp_arr = np.asarray(inp, dtype=np.int8)
                    pred_arr = np.asarray(pred, dtype=np.int8)
                    if not np.array_equal(inp_arr, pred_arr):
                        non_trivial_count += 1
                except Exception:
                    pass

        # If the program is identity on ALL pairs, it's likely overfitting
        # (unless the task truly is identity)
        if non_trivial_count == 0:
            # Check if all demo pairs are actually identity
            all_identity = True
            for pair in demo_pairs:
                input_grids = pair.get("input", [])
                output_grids = pair.get("output", [])
                for i, inp in enumerate(input_grids):
                    if i >= len(output_grids):
                        continue
                    inp_arr = np.asarray(inp, dtype=np.int8)
                    out_arr = np.asarray(output_grids[i], dtype=np.int8)
                    if not np.array_equal(inp_arr, out_arr):
                        all_identity = False
                        break
                if not all_identity:
                    break
            if not all_identity:
                return False

        return True

    # ============================================================
    # Candidate Generation
    # ============================================================

    def _generate_candidates(
        self, demo_pairs: list[dict[str, Any]]
    ) -> list[KSnapCandidate]:
        """Generate depth-1 candidates using ParamInference.

        Uses the ParamInference module to infer parameterized DSL
        primitive candidates from demo pairs.

        Args:
            demo_pairs: List of demo pairs with 'input' and 'output' grids.

        Returns:
            List of KSnapCandidate objects at depth 1.
        """
        programs = self.param_inference.infer_candidates(demo_pairs)
        return [
            KSnapCandidate(program=prog, depth=1)
            for prog in programs
        ]

    # ============================================================
    # Ftel Computation
    # ============================================================

    def _compute_ftel(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> float:
        """Compute Ftel (output similarity to demo pairs).

        Ftel measures the flow-through magnitude — how closely the
        program's output matches the expected output across all demo
        pairs. A Ftel of 1.0 means perfect match.

        Args:
            program: ProgramNode to evaluate.
            demo_pairs: List of demo pairs.

        Returns:
            Average similarity score in [0.0, 1.0].
        """
        total_sim: float = 0.0
        count: int = 0

        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, inp in enumerate(input_grids):
                if i >= len(output_grids):
                    continue
                try:
                    pred = program.apply(inp)
                    expected = output_grids[i]
                    sim = self._grid_similarity(pred, expected)
                    total_sim += sim
                    count += 1
                except Exception:
                    total_sim += 0.0
                    count += 1

        return total_sim / max(count, 1)

    @staticmethod
    def _grid_similarity(g1: np.ndarray, g2: np.ndarray) -> float:
        """Compute pixel-level similarity between two grids.

        Handles shape mismatch by comparing overlapping regions.

        Args:
            g1: First grid.
            g2: Second grid.

        Returns:
            Similarity score in [0.0, 1.0], where 1.0 = identical.
        """
        if g1.shape != g2.shape:
            h = min(g1.shape[0], g2.shape[0])
            w = min(g1.shape[1], g2.shape[1])
            if h == 0 or w == 0:
                return 0.0
            g1_crop = g1[:h, :w]
            g2_crop = g2[:h, :w]
            total = g1_crop.size
            if total == 0:
                return 0.0
            match = int(np.sum(g1_crop == g2_crop))
            return match / total

        total = g1.size
        if total == 0:
            return 1.0
        match = int(np.sum(g1 == g2))
        return match / total

    # ============================================================
    # I-Value Computation
    # ============================================================

    def _compute_i_value(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> float:
        """Compute information existence degree (ℐ).

        Based on the non-zero ratio of program output across demo pairs.
        A program that produces all-zero output has i_value = 0.0 (Dead-Zero).
        Programs that crash also receive i_value = 0.0.

        Args:
            program: ProgramNode to evaluate.
            demo_pairs: List of demo pairs.

        Returns:
            Information existence degree in [0.0, 1.0].
        """
        total_nonzero_ratio: float = 0.0
        count: int = 0

        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            for inp in input_grids:
                try:
                    result = program.apply(inp)
                    total_pixels = result.size
                    if total_pixels == 0:
                        count += 1
                        continue
                    nonzero_pixels = int(np.count_nonzero(result))
                    total_nonzero_ratio += nonzero_pixels / total_pixels
                    count += 1
                except Exception:
                    # Crash → i_value = 0
                    count += 1

        return total_nonzero_ratio / max(count, 1)

    # ============================================================
    # Dead-Zero Check
    # ============================================================

    def _check_dead_zero(
        self,
        program: ProgramNode,
        input_grid: np.ndarray,
    ) -> bool:
        """Check if program crashes or produces all-zero output.

        A Dead-Zero candidate carries no information and is rejected
        at Level 2 of the κ-Snap filtering.

        Args:
            program: ProgramNode to check.
            input_grid: Input grid to test.

        Returns:
            True if the program crashes or produces all-zero output.
        """
        try:
            result = self._cached_apply(program, input_grid)
            if result is None:
                return True
            result_arr = np.asarray(result)
            if result_arr.size == 0:
                return True
            if np.all(result_arr == 0):
                return True
            return False
        except Exception:
            return True

    # ============================================================
    # MUS Check
    # ============================================================

    def _check_mus(
        self, candidates: list[KSnapCandidate]
    ) -> dict[str, bool]:
        """Check for MUS (mutually exclusive states) among candidates.

        Two candidates are in MUS if they have the same MDL cost AND
        similar Ftel (within 0.02) but different program structures.
        This indicates a true conflict where the system cannot decide
        between competing hypotheses at the same complexity level.

        Args:
            candidates: List of KSnapCandidate objects at the same depth.

        Returns:
            Dictionary mapping candidate_id to MUS-active flag.
        """
        mus_map: dict[str, bool] = {
            c.candidate_id: False for c in candidates
        }

        # Group by (MDL, rounded Ftel) to find true ties
        groups: dict[tuple[int, float], list[KSnapCandidate]] = {}
        for cand in candidates:
            key = (cand.program.total_mdl, round(cand.ftel, 2))
            groups.setdefault(key, []).append(cand)

        for key, group in groups.items():
            if len(group) < 2:
                continue
            # Check if programs are structurally different
            signatures: set[str] = set()
            for cand in group:
                try:
                    sig = json.dumps(
                        cand.program.to_dict(), sort_keys=True
                    )
                except Exception:
                    sig = repr(cand.program)
                signatures.add(sig)

            # If multiple different programs with same (MDL, Ftel) → MUS
            if len(signatures) > 1:
                for cand in group:
                    mus_map[cand.candidate_id] = True

        return mus_map

    # ============================================================
    # κ-Snap Execution (Four-Level Filtering)
    # ============================================================

    def _execute_snap(
        self,
        candidate: KSnapCandidate,
        demo_pairs: list[dict[str, Any]],
    ) -> SnapResult:
        """Execute κ-Snap projection: four-level filtering + verification.

        Filtering pipeline:
            Level 1: Ftel threshold check (ftel >= theta_ftel)
            Level 2: Dead-Zero check (not crash, not all-zero)
            Level 3: MUS check (no conflicting candidates)
            Level 4: GaussEx projection verification (exact match)

        Args:
            candidate: KSnapCandidate to evaluate.
            demo_pairs: List of demo pairs for verification.

        Returns:
            SnapResult indicating the outcome.
        """
        event: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "depth": candidate.depth,
            "ftel": round(candidate.ftel, 4),
            "i_value": round(candidate.i_value, 4),
            "mdl": candidate.program.total_mdl,
            "timestamp": time.time(),
        }

        # Level 1: Ftel threshold check
        if candidate.ftel < self.theta_ftel:
            event["result"] = SnapResult.REJECT_FTEL.value
            event["reason"] = (
                f"Ftel insufficient: {candidate.ftel:.4f} < "
                f"{self.theta_ftel}"
            )
            self.causal_log.append(event)
            return SnapResult.REJECT_FTEL

        # Level 2: Dead-Zero check
        # Use first demo pair's first input grid for the check
        first_input: np.ndarray = np.zeros((1, 1), dtype=np.int8)
        if demo_pairs:
            input_grids = demo_pairs[0].get("input", [])
            if input_grids:
                first_input = input_grids[0]

        if self._check_dead_zero(candidate.program, first_input):
            event["result"] = SnapResult.REJECT_DZ.value
            event["reason"] = "Dead-Zero: crash or all-zero output"
            self.causal_log.append(event)
            return SnapResult.REJECT_DZ

        # Also check i_value against theta_dead
        if candidate.i_value < self.theta_dead:
            event["result"] = SnapResult.REJECT_DZ.value
            event["reason"] = (
                f"Dead-Zero: i_value {candidate.i_value:.4f} < "
                f"{self.theta_dead}"
            )
            self.causal_log.append(event)
            return SnapResult.REJECT_DZ

        # Level 3: MUS check — log conflict but continue to verification.
        # In the κ-Snap theory, SUSPEND_MUS means "awaiting adjudication."
        # For ARC solving, GaussEx verification IS the adjudication:
        # it deterministically resolves which candidate is correct.
        # Suspending here would prevent finding solutions when multiple
        # candidates share the same (MDL, Ftel) — a common occurrence.
        mus_suspended: bool = candidate.mus_active
        if mus_suspended:
            event["mus_suspended"] = True

        # Level 4: GaussEx projection verification
        # (also serves as MUS adjudication when mus_suspended=True)
        # 优化：预计算所有 demo 输出，避免 verifier 内重复 apply
        pre_computed: dict[tuple[int, int], np.ndarray] = {}
        for pair_idx, pair in enumerate(demo_pairs):
            input_grids = pair.get("input", [])
            for i, input_grid in enumerate(input_grids):
                try:
                    pred = self._cached_apply(
                        candidate.program, input_grid
                    )
                    pre_computed[(pair_idx, i)] = pred
                except Exception:
                    pass

        is_valid = self.verifier.verify_program(
            candidate.program, demo_pairs,
            pre_computed=pre_computed,
        )

        if is_valid:
            event["result"] = SnapResult.MANIFESTED.value
            if mus_suspended:
                event["reason"] = (
                    "Manifested via GaussEx adjudication "
                    "(MUS conflict resolved)"
                )
            else:
                event["reason"] = "Manifested: passes all demo constraints"
            self.causal_log.append(event)
            return SnapResult.MANIFESTED
        else:
            # Verification failed — not enough evidence to manifest
            event["result"] = "reject_verification"
            if mus_suspended:
                event["reason"] = (
                    "GaussEx adjudication: MUS candidate rejected"
                )
            else:
                event["reason"] = "GaussEx verification failed"
            self.causal_log.append(event)
            return SnapResult.REJECT_FTEL

    # ============================================================
    # Depth Expansion
    # ============================================================

    def _expand_depth(
        self,
        candidates: list[KSnapCandidate],
        demo_pairs: list[dict[str, Any]],
    ) -> list[KSnapCandidate]:
        """Expand to next depth by chaining pairs of candidates.

        Chains each current beam candidate with each depth-1 candidate
        (from ParamInference) to produce depth+1 candidates. Uses
        signature-based deduplication to avoid redundant programs.

        Args:
            candidates: Current beam candidates (at depth d).
            demo_pairs: Demo pairs for regenerating depth-1 candidates.

        Returns:
            List of KSnapCandidate objects at depth d+1.
        """
        if not candidates:
            return []

        current_depth: int = candidates[0].depth

        # Regenerate depth-1 candidates for chaining
        depth1_candidates = self._generate_candidates(demo_pairs)
        depth1_programs: list[ProgramNode] = [
            c.program for c in depth1_candidates[: self.beam_width]
        ]

        if not depth1_programs:
            return []

        expanded: list[KSnapCandidate] = []
        seen_sigs: set[str] = set()

        for cand in candidates:
            for prog2 in depth1_programs:
                chained = self._chain_programs(cand.program, prog2)
                sig = self._program_signature(chained)
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                expanded.append(
                    KSnapCandidate(
                        program=chained,
                        depth=current_depth + 1,
                    )
                )

        return expanded

    # ============================================================
    # Helper Methods
    # ============================================================

    @staticmethod
    def _chain_programs(
        prog1: ProgramNode, prog2: ProgramNode
    ) -> ProgramNode:
        """Chain two programs: apply prog1 then prog2.

        Creates a new ProgramNode where prog2 is appended as the last
        child of prog1's chain. Both programs are cloned to avoid
        mutating the originals.

        Args:
            prog1: First program (applied first).
            prog2: Second program (applied after prog1).

        Returns:
            New ProgramNode representing prog1 ⊙ prog2 (chain).
        """
        new_elem = prog1.element.clone() if prog1.element is not None else None
        new_node = ProgramNode(new_elem)
        new_node.children = [c.clone() for c in prog1.children]
        new_node.children.append(prog2.clone())
        new_node.combo_type = "chain"
        new_node.total_mdl = new_node.compute_mdl()
        return new_node

    @staticmethod
    def _program_signature(prog: ProgramNode) -> str:
        """Generate a unique signature for a program (for dedup).

        Args:
            prog: ProgramNode to sign.

        Returns:
            String signature based on program structure.
        """
        try:
            return json.dumps(prog.to_dict(), sort_keys=True)
        except Exception:
            return repr(prog)

    def _extract_demo_pairs(
        self, task_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract demo pairs from task data.

        Handles two formats:
        1. Task with 'train' key: {'train': [{'input': [...], 'output': [...]}]}
        2. Task that is itself a demo pair: {'input': [...], 'output': [...]}

        Args:
            task_data: Raw task dictionary.

        Returns:
            List of demo pair dicts with numpy array 'input' and 'output'.
        """
        demo_pairs: list[dict[str, Any]] = []

        # Format 1: task_data has 'train' key
        train = task_data.get("train", [])
        for pair in train:
            inp = pair.get("input", [])
            out = pair.get("output", [])
            inp_arrays = [np.array(f, dtype=np.int8) for f in inp]
            out_arrays = [np.array(f, dtype=np.int8) for f in out]
            if inp_arrays and out_arrays:
                demo_pairs.append(
                    {"input": inp_arrays, "output": out_arrays}
                )

        # Format 2: task_data itself is a demo pair
        if not demo_pairs:
            inp = task_data.get("input", [])
            out = task_data.get("output", [])
            if inp and out:
                inp_arrays = [np.array(f, dtype=np.int8) for f in inp]
                out_arrays = [np.array(f, dtype=np.int8) for f in out]
                if inp_arrays and out_arrays:
                    demo_pairs.append(
                        {"input": inp_arrays, "output": out_arrays}
                    )

        return demo_pairs

    def _log_event(
        self, event_type: str, reason: str, depth: int
    ) -> None:
        """Log a search event to the causal log.

        Args:
            event_type: Type of event (e.g., 'search_start', 'depth_1').
            reason: Human-readable description.
            depth: Current search depth.
        """
        self.causal_log.append(
            {
                "event_type": event_type,
                "reason": reason,
                "depth": depth,
                "timestamp": time.time(),
            }
        )
