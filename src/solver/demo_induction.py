"""Demo Induction Engine (Enhanced with Modifier Hints).

Implements κ-Snap + Modifier-Aware MDL Prior for ARC-AGI-3.
Adds modifier_hints as MDL search prior bias, aligning κ-Snap search
with hidden game mechanics without accessing private logic.

TOMAS Concepts:
  L2 Streamlined Reduction  → MDL search with prior bias
  Modifier Hints           → MDL prior bias (Prior Bias)
  lock_key_seq             → boosts unlock_*, sequence_* DSL prior probability
  fog_of_war              → boosts explore_*, delta_detect* DSL prior probability
  gravity                  → boosts fall_*, move_down* DSL prior probability
  Sleep-Step Macro         → high-frequency combinations in library.json prioritized
"""
from __future__ import annotations

import os
import json
from typing import Any, Optional
import numpy as np

from src.core.dsl_primitives import DSLElement, ProgramNode, get_all_primitives
from src.core.octonion_hyperedge import OctonionHyperEdge
from src.solver.kappa_snap_searcher import KappaSnapSearcher
from src.solver.gaussex_verifier import GaussExVerifier
from src.solver.enpv_decision import ENPVDecision
from src.core.topo_hash import TopoHashFilter
try:
    from src.core.luzhao_dna import LuzhaoDNA
    _HAVE_LUZHAO = True
except ImportError:
    _HAVE_LUZHAO = False
    LuzhaoDNA = None  # type: ignore

try:
    from src.solver.pruning_optimizer import PruningOptimizer
    _HAVE_PRUNING = True
except ImportError:
    _HAVE_PRUNING = False
    PruningOptimizer = None  # type: ignore

# Default library path
_DEFAULT_LIBRARY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "library.json"
)


class DemoInducer:
    """Induce a DSL program from demo pairs, biased by modifier_hints (MDL prior boost).

    The modifier_hints parameter provides domain knowledge about the game mechanics
    without accessing private logic. Each modifier hint maps to a set of DSL keyword
    weights, which boost the MDL score of matching programs during search.

    Attributes:
        dsl: DSL primitives set.
        library: Sleep-Step Macros from library.json.
        searcher: KappaSnapSearcher for program search.
        verifier: GaussExVerifier for validation.
        enpv: ENPVDecision for early termination.
        topo_filter: TopoHashFilter for Phase A fast filtering.
        luohao: LuzhaoDNA for DNA-level matching.
        modifier_prior_boost: Dict mapping modifier names to keyword-weight dicts.
    """

    def __init__(self, library: Any = None, max_depth: int = 3) -> None:
        """Initialize the DemoInducer.

        Args:
            library: LibraryLearning instance or list of Sleep-Step Macros.
            max_depth: Maximum program composition depth for search.
        """
        self.dsl = get_all_primitives()
        self.library = library if library is not None else self._load_library()

        self.searcher = KappaSnapSearcher(
            {"max_depth": max_depth, "mdl_threshold": 50, "time_limit_seconds": 80.0}
        )
        self.verifier = GaussExVerifier()
        self.enpv = ENPVDecision()
        self.topo_filter = TopoHashFilter()
        if _HAVE_LUZHAO:
            self.luzhao = LuzhaoDNA()
        else:
            self.luzhao = None

        # ---------- Modifier → Prior Boost ----------
        # Maps modifier hint names to keyword-weight dictionaries.
        # When a program's string representation contains a keyword,
        # the corresponding weight is added to the MDL prior boost.
        self.modifier_prior_boost: dict[str, dict[str, float]] = {
            "lock_key_seq": {
                "unlock": 2.0,
                "sequence": 1.8,
                "key": 1.5,
                "door": 1.5,
            },
            "fog_of_war": {
                "explore": 2.0,
                "delta": 1.8,
                "visible": 1.5,
            },
            "gravity": {
                "fall": 2.0,
                "down": 1.8,
                "move": 1.5,
            },
            "color_cycle": {
                "cycle": 2.0,
                "palette": 1.8,
            },
            "mirror_axis": {
                "mirror": 2.0,
                "flip": 1.8,
            },
        }

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def induce_from_demos(
        self,
        demo_pairs: list[tuple[np.ndarray, np.ndarray]],
        modifier_hints: Optional[list[str]] = None,
    ) -> Optional[ProgramNode]:
        """Induce a program from demo pairs with modifier-aware MDL prior.

        Args:
            demo_pairs: List of (input_grid, output_grid) pairs.
            modifier_hints: List of modifier hint strings (e.g., ["lock_key_seq"]).
                Each hint boosts the prior probability of matching DSL primitives.

        Returns:
            Best matching ProgramNode, or None if induction fails.
        """
        modifier_hints = modifier_hints or []
        perceived_demos = [
            (inp, out, self._encode_to_hyperedge(inp, out))
            for inp, out in demo_pairs
        ]

        # ---- Phase A: Fast filters ----
        candidates = self._phase_a_fast_filters(perceived_demos)
        if not candidates:
            return self._fallback_explore(demo_pairs, modifier_hints)

        # ---- Phase B: MDL Search with Modifier Prior ----
        best_prog = None
        best_score = -np.inf

        for prog in candidates:
            if not self._verify_on_demos(prog, perceived_demos):
                continue

            mdl = self._compute_mdl(prog)
            prior_boost = self._calc_modifier_prior(prog, modifier_hints)
            score = -mdl + prior_boost

            if not self.enpv.should_continue(score, mdl):
                break

            if score > best_score:
                best_score = score
                best_prog = prog

        return best_prog or self._fallback_explore(demo_pairs, modifier_hints)

    # ------------------------------------------------------------
    # Modifier Prior Calculation
    # ------------------------------------------------------------

    def _calc_modifier_prior(
        self, prog: ProgramNode, modifier_hints: list[str]
    ) -> float:
        """Boost MDL score if DSL name matches modifier hints.

        For each modifier hint, checks if the program's string representation
        contains any of the keywords associated with that modifier.
        The total boost is capped at 15.0 to prevent over-fitting.

        Args:
            prog: ProgramNode to evaluate.
            modifier_hints: List of active modifier hint strings.

        Returns:
            Prior boost value (float, non-negative, capped at 15.0).
        """
        boost = 0.0
        prog_str = str(prog).lower()

        for hint in modifier_hints:
            if hint not in self.modifier_prior_boost:
                continue
            for keyword, weight in self.modifier_prior_boost[hint].items():
                if keyword in prog_str:
                    boost += weight
        return min(boost, 15.0)  # cap

    # ------------------------------------------------------------
    # Phase A: Topo Filters
    # ------------------------------------------------------------

    def _phase_a_fast_filters(
        self, perceived_demos: list[tuple[np.ndarray, np.ndarray, Any]]
    ) -> list[ProgramNode]:
        """Apply fast topological filters to eliminate invalid candidates.

        Uses TopoHashFilter and LuzhaoDNA to quickly filter out programs
        that cannot explain the demo pairs.

        Args:
            perceived_demos: List of (input, output, hyperedge) tuples.

        Returns:
            List of candidate ProgramNodes that pass all filters.
        """
        keep: list[ProgramNode] = []
        base_programs = self.searcher.enumerate_base_programs(self.library)

        for prog in base_programs:
            ok = True
            for inp, out, edge in perceived_demos:
                if not self.topo_filter.matches(prog, inp, out):
                    ok = False
                    break
                if self.luzhao is not None and not self.luzhao.matches(prog, edge):
                    ok = False
                    break
            if ok:
                keep.append(prog)

        return keep

    # ------------------------------------------------------------
    # MDL Computation
    # ------------------------------------------------------------

    def _compute_mdl(self, prog: ProgramNode) -> float:
        """Compute the MDL cost of a program.

        Args:
            prog: ProgramNode to evaluate.

        Returns:
            MDL cost (float). Lower is better.
        """
        return float(prog.total_mdl) if hasattr(prog, "total_mdl") else 50.0

    # ------------------------------------------------------------
    # Verification & Fallback
    # ------------------------------------------------------------

    def _verify_on_demos(
        self, prog: ProgramNode, perceived_demos: list[tuple[np.ndarray, np.ndarray, Any]]
    ) -> bool:
        """Verify that a program correctly transforms all demo inputs to outputs.

        Args:
            prog: ProgramNode to verify.
            perceived_demos: List of (input, output, hyperedge) tuples.

        Returns:
            True if program passes verification on all demos.
        """
        for inp, out, _ in perceived_demos:
            try:
                predicted = prog.apply(inp.copy())
                if not self.verifier.verify(predicted, out):
                    return False
            except Exception:
                return False
        return True

    def _fallback_explore(
        self, demo_pairs: list[tuple[np.ndarray, np.ndarray]], modifier_hints: list[str]
    ) -> Optional[ProgramNode]:
        """Fallback: search primitives with non-zero prior boost.

        When Phase A+B fail to find a candidate, try individual DSL primitives
        that have a non-zero modifier prior boost.

        Args:
            demo_pairs: List of (input, output) pairs.
            modifier_hints: List of active modifier hints.

        Returns:
            A single DSL primitive that matches, or None.
        """
        if not demo_pairs:
            return None
        inp0, out0 = demo_pairs[0]
        for prim in self.dsl:
            if self._calc_modifier_prior(prim, modifier_hints) == 0:
                continue
            try:
                if self.verifier.verify(prim.apply(inp0.copy()), out0):
                    return prim
            except Exception:
                continue
        return None

    # ------------------------------------------------------------
    # HyperEdge Encoding
    # ------------------------------------------------------------

    def _encode_to_hyperedge(self, inp: np.ndarray, out: np.ndarray) -> Any:
        """Encode input-output pair as an OctonionHyperEdge.

        Args:
            inp: Input grid.
            out: Output grid.

        Returns:
            OctonionHyperEdge with topological invariants attached.
        """
        edge = OctonionHyperEdge.from_grids(inp, out)
        if hasattr(edge, "attach_topo_invariants"):
            edge.attach_topo_invariants()
        if hasattr(edge, "attach_luzhao_dna"):
            edge.attach_luzhao_dna()
        return edge

    # ------------------------------------------------------------
    # Library Loading
    # ------------------------------------------------------------

    def _load_library(self) -> list[Any]:
        """Load Sleep-Step Macros from library.json.

        Returns:
            List of library macros (programs), or empty list if not found.
        """
        library_path = _DEFAULT_LIBRARY_PATH
        if not os.path.exists(library_path):
            return []
        try:
            with open(library_path, "r") as f:
                data = json.load(f)
            return data.get("macros", [])
        except Exception:
            return []
