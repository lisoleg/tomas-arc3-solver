"""kappa-Snap abductive search: Two-Phase program synthesis.

TOMAS v2.0 upgrade: ENPV decision for early termination in Phase B.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.core.dsl_primitives import DSLElement, ProgramNode, get_all_primitives
from src.core.topo_hash import TopoHashFilter
from src.solver.enpv_decision import ENPVDecision


class KappaSnapSearcher:
    """kappa-Snap abductive search with Two-Phase filtering.

    Phase A uses topological hash quick-filtering (with Luzhao DNA) to
    eliminate 90%+ of candidates. Phase B enumerates remaining candidates
    by MDL priority and verifies with GaussEx. ENPV decision module
    enables early termination when expected net value goes negative.

    Attributes:
        dsl_set: List of available DSL primitives.
        library: LibraryLearning instance for MDL reduction.
        max_depth: Maximum composition depth.
        mdl_threshold: Maximum acceptable MDL cost.
        topo_filter: TopoHashFilter for Phase A.
        enpv: ENPVDecision for termination control.
    """

    def __init__(self, config: dict[str, Any], library: Any = None) -> None:
        """Initialize the searcher.

        Args:
            config: Search configuration dict.
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

        Phase A: Filter candidates using topological hash necessary condition
                 (with optional Luzhao DNA hash).
        Phase B: Enumerate by MDL priority, verify with GaussEx.
                 ENPV-based early termination when expected value < 0.

        Args:
            demo_pairs: List of demo pairs.

        Returns:
            List of valid, ranked ProgramNode candidates.
        """
        self._start_time = time.time()
        self.enpv.reset()

        # Generate candidates at all depths
        all_candidates: list[ProgramNode] = []
        for depth in range(1, self.max_depth + 1):
            if self._is_timeout():
                break
            candidates = self.enumerate_candidates(depth)
            all_candidates.extend(candidates)

        # Phase A: Topological hash quick filter
        phase_a_passed = self.phase_a_filter(all_candidates, demo_pairs)

        # Apply library learning to reduce MDL
        if self.library is not None:
            phase_a_passed = self.apply_library(phase_a_passed)

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
        """Phase B: Verify candidates against demo pairs with ENPV termination.

        Each candidate must pass all demo pair verifications. ENPV-based
        early termination: if continuing search has negative expected value,
        stop and return current best results.

        Args:
            candidates: Pre-filtered candidates from Phase A.
            demo_pairs: List of demo pairs.

        Returns:
            List of candidates that pass all demo verifications.
        """
        valid: list[ProgramNode] = []
        ranked = self.rank_by_mdl(candidates)

        # Build posterior-like scores for ENPV decision
        # Use inverse MDL ratio as proxy posterior
        best_mdl = ranked[0].total_mdl if ranked else 1

        for program in ranked:
            if self._is_timeout():
                break
            if program.total_mdl > self.mdl_threshold:
                continue

            # Verify against all demo pairs
            if self._verify_against_demos(program, demo_pairs):
                valid.append(program)

            # ENPV check every 10 candidates to avoid overhead
            if len(valid) > 0 and len(valid) % 10 == 0:
                # Build proxy posterior list for ENPV
                max_mdl_valid = max(p.total_mdl for p in valid) if valid else best_mdl
                proxy_posteriors = []
                for p in valid:
                    posterior = (best_mdl / max(p.total_mdl, 1)) * 0.5
                    proxy_posteriors.append((p, posterior))

                if not self.enpv.should_continue(proxy_posteriors, max_mdl_valid):
                    break

        return valid

    def enumerate_candidates(self, depth: int) -> list[ProgramNode]:
        """Enumerate candidate programs at a given composition depth.

        Depth 1: single primitives.
        Depth 2: pairs of primitives (chain composition).
        Depth 3: triples of primitives.

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
            # Only chain common primitives
            common_prims = [p for p in self.dsl_set if p.mdl_cost <= 10]
            for prim1 in common_prims:
                for prim2 in common_prims:
                    if self._is_timeout():
                        return candidates
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
        """Verify a program against all demo pairs.

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
                    if not np.array_equal(predicted, expected):
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
