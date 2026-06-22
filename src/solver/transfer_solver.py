"""Transfer solver: cross-video pattern transfer + interactive VL debugging."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode
from src.core.transfer_engine import TransferEngine
from src.solver.kappa_snap_searcher import KappaSnapSearcher
from src.solver.gaussex_verifier import GaussExVerifier


class TransferSolver:
    """Solver with cross-video pattern transfer and VL dialog debugging.

    Transfers general DSL subroutines from previously solved videos
    to accelerate search on new tasks. Supports interactive VL dialog
    for debugging when visual analysis is available.

    Attributes:
        searcher: KappaSnapSearcher for program search.
        verifier: GaussExVerifier for validation.
        transfer_engine: TransferEngine for cross-video transfer.
        vl_adapter: Optional VL adapter for dialog debugging.
    """

    def __init__(
        self,
        searcher: KappaSnapSearcher,
        verifier: GaussExVerifier,
        transfer_engine: TransferEngine | None = None,
        vl_adapter: Any = None,
    ) -> None:
        """Initialize the transfer solver.

        Args:
            searcher: KappaSnapSearcher instance.
            verifier: GaussExVerifier instance.
            transfer_engine: TransferEngine for cross-video transfer.
            vl_adapter: Optional DeepSeekVLAdapter for debugging.
        """
        self.searcher = searcher
        self.verifier = verifier
        self.transfer_engine = transfer_engine or TransferEngine(verifier)
        self.vl_adapter = vl_adapter
        self._solved_history: list[tuple[str, list[ProgramNode]]] = []

    def solve(
        self,
        task: dict[str, Any],
        demo_pairs: list[dict[str, Any]],
        test_frames: list[np.ndarray],
    ) -> dict[str, Any]:
        """Solve a task using transferred patterns.

        Args:
            task: Raw task dictionary.
            demo_pairs: List of demo pairs.
            test_frames: Test frames to predict.

        Returns:
            Dictionary with predictions and transfer metadata.
        """
        # First, try transferred programs
        transferred = self.transfer_engine.get_learned_programs()

        # Verify transferred programs against current task
        self.verifier.set_demo_pairs(demo_pairs)
        valid_transferred: list[ProgramNode] = []
        for program in transferred:
            if self.verifier.verify_program(program, demo_pairs):
                valid_transferred.append(program)

        # Search for new programs
        new_programs = self.searcher.two_phase_search(demo_pairs)

        # Combine and select best
        all_programs = valid_transferred + new_programs
        if all_programs:
            best_program = min(all_programs, key=lambda p: p.total_mdl)
        else:
            from src.core.dsl_primitives import DSLElement
            best_program = ProgramNode(DSLElement("copy"))

        # Predict using best program
        predictions: list[np.ndarray] = []
        if test_frames:
            last_frame = test_frames[-1]
            for _ in range(len(test_frames)):
                try:
                    predicted = best_program.apply(last_frame)
                    predictions.append(np.asarray(predicted, dtype=np.int8))
                    last_frame = predicted
                except Exception:
                    predictions.append(last_frame.copy())
        elif demo_pairs:
            last_input = demo_pairs[-1].get("input", [np.zeros((1, 1), dtype=np.int8)])[-1]
            try:
                predicted = best_program.apply(last_input)
                predictions.append(np.asarray(predicted, dtype=np.int8))
            except Exception:
                predictions.append(last_input.copy())

        # Store solved programs for future transfer
        task_id = task.get("task_id", "unknown")
        self._solved_history.append((task_id, all_programs))

        # Extract general subroutines for transfer
        general = self.transfer_engine.extract_general_subroutines(all_programs)
        if general:
            self.transfer_engine.learned_programs.extend(
                [ProgramNode(g) for g in general]
            )

        pred_list = [p.tolist() for p in predictions] if predictions else []

        return {
            "predictions": pred_list,
            "best_program_mdl": best_program.total_mdl,
            "num_transferred": len(valid_transferred),
            "num_new_programs": len(new_programs),
            "num_general_subroutines": len(general),
            "mode": "transfer",
        }

    def interactive_debug(
        self,
        frames: list[np.ndarray],
        program: ProgramNode,
    ) -> dict[str, Any]:
        """Interactive VL dialog for debugging a program.

        Args:
            frames: Video frames to analyze.
            program: ProgramNode to debug.

        Returns:
            Dictionary with VL analysis results.
        """
        if self.vl_adapter is None or not self.vl_adapter.check_availability():
            return {"vl_debug": False, "reason": "VL adapter unavailable"}

        try:
            analysis = self.vl_adapter.multi_round_dialog(
                frames,
                history=[],
            )
            return {"vl_debug": True, "analysis": analysis}
        except Exception as e:
            return {"vl_debug": False, "reason": str(e)}

    def get_transfer_stats(self) -> dict[str, int]:
        """Get statistics about transfer learning.

        Returns:
            Dictionary with transfer statistics.
        """
        return {
            "total_solved": len(self._solved_history),
            "learned_programs": len(self.transfer_engine.get_learned_programs()),
        }
