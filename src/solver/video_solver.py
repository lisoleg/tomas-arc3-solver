"""Video mode main logic: encode -> search -> compose -> predict -> output."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode
from src.core.video_tensor import VideoTemporalEncoder
from src.core.delta_history_buffer import DeltaHistoryBuffer
from src.solver.kappa_snap_searcher import KappaSnapSearcher
from src.solver.gaussex_verifier import GaussExVerifier
from src.solver.delta_composer import DeltaTCombinator
from src.solver.library_learning import LibraryLearning


class VideoSolver:
    """Video mode solver: pure symbolic reasoning pipeline.

    Executes the video mode flow:
    1. Encode frames to temporal hypergraphs
    2. Extract delta-T transformations
    3. Two-Phase search for valid programs
    4. Compose delta-T combinations
    5. Predict next frame
    6. Generate output

    Attributes:
        searcher: KappaSnapSearcher for program search.
        verifier: GaussExVerifier for validation.
        library: LibraryLearning for MDL reduction.
    """

    def __init__(
        self,
        searcher: KappaSnapSearcher,
        verifier: GaussExVerifier,
        library: LibraryLearning,
    ) -> None:
        """Initialize the video solver.

        Args:
            searcher: KappaSnapSearcher instance.
            verifier: GaussExVerifier instance.
            library: LibraryLearning instance.
        """
        self.searcher = searcher
        self.verifier = verifier
        self.library = library

    def solve(
        self,
        task: dict[str, Any],
        demo_pairs: list[dict[str, Any]],
        test_frames: list[np.ndarray],
    ) -> dict[str, Any]:
        """Solve a video ARC task in video mode.

        Args:
            task: Raw task dictionary.
            demo_pairs: List of demo pairs.
            test_frames: Test frames to predict.

        Returns:
            Dictionary with predictions and metadata.
        """
        # Collect all demo frames
        demo_frames: list[np.ndarray] = []
        for pair in demo_pairs:
            demo_frames.extend(pair.get("input", []))
            demo_frames.extend(pair.get("output", []))

        # Step 1: Encode frames
        encoder = VideoTemporalEncoder(demo_frames)
        temporal_hgs = encoder.encode_frames()

        # Step 2: Extract delta-T
        deltaT_list = encoder.extract_all_deltaT()

        # Step 3: Two-Phase search
        self.verifier.set_demo_pairs(demo_pairs)
        valid_programs = self.searcher.two_phase_search(demo_pairs)

        # Step 4: Compose delta-T if available
        if deltaT_list:
            combinator = DeltaTCombinator(deltaT_list)
            composed = combinator.search_combinations(max_depth=3)
            valid_programs.extend(composed)

        # Step 5: Select best program (lowest MDL)
        if valid_programs:
            best_program = min(valid_programs, key=lambda p: p.total_mdl)
        else:
            from src.core.dsl_primitives import DSLElement
            best_program = ProgramNode(DSLElement("copy"))

        # Step 6: Predict next frames
        predictions: list[np.ndarray] = []
        if test_frames:
            last_frame = test_frames[-1]
            for _ in range(len(test_frames)):
                predicted = encoder.predict_next_frame(best_program, last_frame)
                predictions.append(predicted)
                last_frame = predicted
        elif demo_frames:
            last_frame = demo_frames[-1]
            predicted = encoder.predict_next_frame(best_program, last_frame)
            predictions.append(predicted)

        # Convert predictions to serializable format
        pred_list = [p.tolist() for p in predictions] if predictions else []

        return {
            "predictions": pred_list,
            "best_program_mdl": best_program.total_mdl,
            "num_valid_programs": len(valid_programs),
            "mode": "video",
        }
