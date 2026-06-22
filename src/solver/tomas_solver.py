"""Main TOMAS solver: mode dispatch, auto-switch, unified solve entry."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.core.video_tensor import VideoTemporalEncoder
from src.core.keyframe_extractor import KeyframeExtractor
from src.core.multi_scale_analyzer import MultiScaleAnalyzer
from src.solver.kappa_snap_searcher import KappaSnapSearcher
from src.solver.gaussex_verifier import GaussExVerifier
from src.solver.bayesian_confidence import BayesianConfidence
from src.solver.fusion_scorer import FusionScorer
from src.solver.library_learning import LibraryLearning
from src.solver.video_solver import VideoSolver
from src.solver.transfer_solver import TransferSolver
from src.utils.kaggle_format import KaggleFormatAdapter, VideoARCTask
from src.utils.gpu_optimizer import GPUOptimizer
from src.utils.logger import get_auditor


class TOMASSolver:
    """Main TOMAS solver with mode dispatch and auto-switching.

    Orchestrates the full solving pipeline across three modes:
    video (fast symbolic), bayesian (posterior ranking), fusion (multi-modal).

    Attributes:
        mode: Current inference mode.
        searcher: KappaSnapSearcher for program search.
        verifier: GaussExVerifier for validation.
        bayesian: BayesianConfidence for posterior ranking.
        fusion: FusionScorer for multi-modal scoring.
        library: LibraryLearning for MDL reduction.
        gpu_optimizer: GPUOptimizer for memory management.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the TOMAS solver with all components.

        Args:
            config: Full configuration dictionary.
        """
        self.config = config
        self.mode: str = "auto"

        # Initialize components
        self.gpu_optimizer = GPUOptimizer(config.get("gpu", {}))
        self.library = LibraryLearning(config.get("library", {}))
        self.searcher = KappaSnapSearcher(
            config.get("search", {}), library=self.library
        )
        self.verifier = GaussExVerifier()
        self.bayesian = BayesianConfidence(config.get("bayesian", {}))

        # Initialize VL adapter (lazy)
        self._vl_adapter: Any = None
        vl_config = config.get("vl_api", {})
        if vl_config.get("available", False) or vl_config.get("api_key", ""):
            try:
                from src.api.deepseek_vl import DeepSeekVLAdapter
                self._vl_adapter = DeepSeekVLAdapter(vl_config)
            except Exception:
                self._vl_adapter = None

        self.fusion = FusionScorer(config.get("fusion", {}), self._vl_adapter)

        # Solvers
        self.video_solver = VideoSolver(self.searcher, self.verifier, self.library)
        self.transfer_solver = TransferSolver(
            self.searcher, self.verifier, vl_adapter=self._vl_adapter
        )

        # Format adapter
        self.kaggle_adapter = KaggleFormatAdapter()

        # Mode switch thresholds
        ms_config = config.get("mode_switch", {})
        self.video_time_threshold: float = ms_config.get("video_time_threshold", 40)
        self.bayesian_time_threshold: float = ms_config.get("bayesian_time_threshold", 80)
        self.video_complexity_threshold: int = ms_config.get("video_complexity_threshold", 2)
        self.bayesian_complexity_threshold: int = ms_config.get("bayesian_complexity_threshold", 4)

    def solve(self, task: dict[str, Any], mode: str = "auto") -> dict[str, Any]:
        """Unified solve entry point.

        Args:
            task: Raw task dictionary.
            mode: Inference mode ('video', 'bayesian', 'fusion', 'auto').

        Returns:
            Dictionary with predictions and metadata.
        """
        self.mode = mode
        task_id = task.get("task_id", "unknown")

        # Parse input
        video_task = self.parse_input(task)
        demo_pairs = video_task.demo_pairs
        test_frames = video_task.test_frames

        # Auto-select mode if needed
        if mode == "auto":
            complexity = len(demo_pairs)
            mode = self.auto_select_mode(
                self.config.get("search", {}).get("time_limit_seconds", 80.0),
                complexity,
            )
            self.mode = mode

        # Record audit
        auditor = get_auditor()
        auditor.record(
            task_id=task_id,
            action=f"solve_start_{mode}",
            decision_path=[f"mode={mode}", f"demos={len(demo_pairs)}"],
        )

        # Dispatch to appropriate mode
        if mode == "video":
            result = self.solve_video(task)
        elif mode == "bayesian":
            result = self.solve_bayesian(task)
        elif mode == "fusion":
            result = self.solve_fusion(task)
        else:
            result = self.solve_video(task)

        # Record audit
        auditor.record(
            task_id=task_id,
            action="solve_end",
            program_mdl=result.get("best_program_mdl", 0),
            decision_path=[f"mode={mode}", f"predictions={len(result.get('predictions', []))}"],
        )

        return result

    def solve_video(self, task: dict[str, Any]) -> dict[str, Any]:
        """Solve in video mode (pure symbolic, fast).

        Args:
            task: Raw task dictionary.

        Returns:
            Result dictionary with predictions.
        """
        video_task = self.parse_input(task)
        return self.video_solver.solve(task, video_task.demo_pairs, video_task.test_frames)

    def solve_bayesian(self, task: dict[str, Any]) -> dict[str, Any]:
        """Solve in bayesian mode (posterior ranking).

        Args:
            task: Raw task dictionary.

        Returns:
            Result dictionary with predictions.
        """
        video_task = self.parse_input(task)
        demo_pairs = video_task.demo_pairs
        test_frames = video_task.test_frames

        # Search for valid programs
        self.verifier.set_demo_pairs(demo_pairs)
        valid_programs = self.searcher.two_phase_search(demo_pairs)

        # Bayesian ranking
        ranked = self.bayesian.rank_candidates(valid_programs, demo_pairs)

        # Select best by posterior
        if ranked:
            best_program, best_posterior = ranked[0]
        else:
            best_program = ProgramNode(DSLElement("copy"))
            best_posterior = 0.0

        # Predict
        predictions = self._predict_with_program(best_program, demo_pairs, test_frames)

        return {
            "predictions": predictions,
            "best_program_mdl": best_program.total_mdl,
            "best_posterior": best_posterior,
            "num_valid_programs": len(valid_programs),
            "mode": "bayesian",
        }

    def solve_fusion(self, task: dict[str, Any]) -> dict[str, Any]:
        """Solve in fusion mode (multi-modal, most accurate).

        Args:
            task: Raw task dictionary.

        Returns:
            Result dictionary with predictions.
        """
        video_task = self.parse_input(task)
        demo_pairs = video_task.demo_pairs
        test_frames = video_task.test_frames

        # Search for valid programs
        self.verifier.set_demo_pairs(demo_pairs)
        valid_programs = self.searcher.two_phase_search(demo_pairs)

        # Bayesian ranking for initial candidate selection
        ranked = self.bayesian.rank_candidates(valid_programs, demo_pairs)

        # Take top candidates
        top_candidates = [p for p, _ in ranked[:20]] if ranked else valid_programs[:20]

        # Multi-scale frame analysis
        all_frames: list[np.ndarray] = []
        for pair in demo_pairs:
            all_frames.extend(pair.get("input", []))
            all_frames.extend(pair.get("output", []))

        keyframe_extractor = KeyframeExtractor()
        multi_scale = MultiScaleAnalyzer(keyframe_extractor, self._vl_adapter)
        analysis = multi_scale.analyze(all_frames)

        # Fusion scoring
        self.fusion.adapt_weights(len(demo_pairs))
        fusion_ranked: list[tuple[ProgramNode, float]] = []
        for program in top_candidates:
            score = self.fusion.fuse(program, all_frames, demo_pairs)
            fusion_ranked.append((program, score))

        fusion_ranked.sort(key=lambda x: x[1], reverse=True)

        if fusion_ranked:
            best_program, best_score = fusion_ranked[0]
        else:
            best_program = ProgramNode(DSLElement("copy"))
            best_score = 0.0

        # Predict
        predictions = self._predict_with_program(best_program, demo_pairs, test_frames)

        return {
            "predictions": predictions,
            "best_program_mdl": best_program.total_mdl,
            "best_fusion_score": best_score,
            "vl_available": self.fusion.vl_available,
            "num_valid_programs": len(valid_programs),
            "mode": "fusion",
        }

    def auto_select_mode(self, time_budget: float, complexity: int) -> str:
        """Auto-select mode based on time budget and complexity.

        Args:
            time_budget: Available time in seconds.
            complexity: Number of demo pairs (complexity proxy).

        Returns:
            Selected mode string.
        """
        if time_budget < self.video_time_threshold or complexity < self.video_complexity_threshold:
            return "video"
        elif time_budget < self.bayesian_time_threshold or complexity < self.bayesian_complexity_threshold:
            return "bayesian"
        else:
            return "fusion"

    def parse_input(self, task: dict[str, Any]) -> VideoARCTask:
        """Parse raw task data into VideoARCTask.

        Args:
            task: Raw task dictionary.

        Returns:
            Parsed VideoARCTask.
        """
        return self.kaggle_adapter.parse_input(task)

    def generate_output(self, prediction: np.ndarray) -> dict[str, Any]:
        """Generate output dictionary from a prediction.

        Args:
            prediction: Predicted grid as ndarray.

        Returns:
            Output dictionary.
        """
        if isinstance(prediction, np.ndarray):
            return prediction.tolist()
        return prediction

    def _predict_with_program(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
        test_frames: list[np.ndarray],
    ) -> list[list[list[int]]]:
        """Generate predictions using a program.

        Args:
            program: ProgramNode to use for prediction.
            demo_pairs: Demo pairs for context.
            test_frames: Test frames to predict.

        Returns:
            List of predicted grids as nested lists.
        """
        predictions: list[list[list[int]]] = []

        if test_frames:
            last_frame = test_frames[-1]
            for _ in range(len(test_frames)):
                try:
                    predicted = program.apply(last_frame)
                    predictions.append(np.asarray(predicted, dtype=np.int8).tolist())
                    last_frame = np.asarray(predicted, dtype=np.int8)
                except Exception:
                    predictions.append(last_frame.tolist())
        elif demo_pairs:
            last_input = demo_pairs[-1].get("input", [np.zeros((1, 1), dtype=np.int8)])[-1]
            try:
                predicted = program.apply(last_input)
                predictions.append(np.asarray(predicted, dtype=np.int8).tolist())
            except Exception:
                predictions.append(last_input.tolist())

        return predictions

    def _post_solve_learning(self, result: dict[str, Any]) -> None:
        """Post-solve Library Learning feedback.

        Extracts patterns from solved programs and updates the library.

        Args:
            result: Solve result dictionary.
        """
        # This is called after each task to enable library learning
        # In a real implementation, we'd reconstruct the program from result
        # For now, we just save the library
        try:
            self.library.save()
        except Exception:
            pass
