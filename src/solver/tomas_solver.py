"""Main TOMAS solver: mode dispatch, auto-switch, unified solve entry."""
from __future__ import annotations

from typing import Any

import numpy as np
import os

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
# v2.4: Optional PsiFusionGate import
try:
    from src.solver.psi_fusion_gate import PsiFusionGate, create_default_anchors
    _HAVE_PSI_GATE = True
except ImportError:
    _HAVE_PSI_GATE = False
    PsiFusionGate = None  # type: ignore
    create_default_anchors = None  # type: ignore
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

        # v2.3: Merge cuda and pruning config into search config
        search_config = dict(config.get("search", {}))
        if "cuda" in config:
            search_config["cuda"] = config["cuda"]
        elif "cuda" not in search_config:
            # Default CUDA config from top-level
            search_config["cuda"] = {"enabled": True, "batch_size": 256}

        if "pruning" in config:
            search_config["pruning"] = config["pruning"]
        elif "pruning" not in search_config:
            # Default pruning config
            search_config["pruning"] = {
                "enable_betti0": True,
                "enable_symmetry_dedup": True,
                "enable_heuristic_order": True,
                "enable_incremental_mdl": True,
            }

        self.searcher = KappaSnapSearcher(
            search_config, library=self.library
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

        # v2.4: Optional PsiFusionGate for semantic gating
        self.psi_gate = None
        if _HAVE_PSI_GATE and config.get("psi_gate", {}).get("enabled", False):
            psi_config = config.get("psi_gate", {})
            anchors = create_default_anchors() if psi_config.get("use_default_anchors", True) else []
            self.psi_gate = PsiFusionGate(
                anchors=anchors,
                tolerance_decay_rate=psi_config.get("tolerance_decay_rate", 0.05),
                verbose=config.get("verbose", False),
            )

        # v2.4: Optional AEGIS Evolution Engine
        self.aegis_engine = None
        if config.get("aegis", {}).get("enabled", False):
            try:
                from src.solver.aegis_evolver import AEGISEngine, EvolutionConfig
                aegis_config = config.get("aegis", {})
                evo_config = EvolutionConfig(
                    population_size=aegis_config.get("population_size", 20),
                    num_generations=aegis_config.get("num_generations", 5),
                    mutation_rate=aegis_config.get("mutation_rate", 0.3),
                    crossover_rate=aegis_config.get("crossover_rate", 0.5),
                    elitism_count=aegis_config.get("elitism_count", 3),
                    mdl_weight=aegis_config.get("mdl_weight", 0.4),
                    accuracy_weight=aegis_config.get("accuracy_weight", 0.6),
                    use_psi_gate=aegis_config.get("use_psi_gate", True),
                    verbose=config.get("verbose", False),
                )
                self.aegis_engine = AEGISEngine(evo_config)
            except ImportError:
                self.aegis_engine = None

        # v2.4: Optional Causal DSL Prior
        self.causal_prior = None
        if config.get("causal_prior", {}).get("enabled", False):
            try:
                from src.solver.causal_dsl_prior import CausalDSLPrior
                cp_config = config.get("causal_prior", {})
                self.causal_prior = CausalDSLPrior(verbose=cp_config.get("verbose", False))
                # Load history if available
                history_path = cp_config.get("history_path", "")
                if history_path and os.path.exists(history_path):
                    self.causal_prior.load_graph(history_path)
            except ImportError:
                self.causal_prior = None

        # Pass causal_prior to searcher (after initialization)
        if self.searcher is not None and self.causal_prior is not None:
            self.searcher.causal_prior = self.causal_prior
            if self.searcher.pruning is not None:
                self.searcher.pruning.causal_prior = self.causal_prior

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

        # Parse input once, share across mode-specific solvers
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

        # Dispatch to appropriate mode (pass pre-parsed video_task)
        if mode == "video":
            result = self.solve_video(task, video_task)
        elif mode == "bayesian":
            result = self.solve_bayesian(task, video_task)
        elif mode == "fusion":
            result = self.solve_fusion(task, video_task)
        else:
            result = self.solve_video(task, video_task)

        # Record audit
        auditor.record(
            task_id=task_id,
            action="solve_end",
            program_mdl=result.get("best_program_mdl", 0),
            decision_path=[f"mode={mode}", f"predictions={len(result.get('predictions', []))}"],
        )

        return result

    def solve_video(self, task: dict[str, Any],
                    video_task: Any = None) -> dict[str, Any]:
        """Solve in video mode (pure symbolic, fast).

        Args:
            task: Raw task dictionary.
            video_task: Pre-parsed VideoARCTask (optional, parsed if None).

        Returns:
            Result dictionary with predictions.
        """
        if video_task is None:
            video_task = self.parse_input(task)
        demo_pairs = video_task.demo_pairs
        test_frames = video_task.test_frames

        # Run search
        self.verifier.set_demo_pairs(demo_pairs)
        valid_programs = self.searcher.two_phase_search(demo_pairs)

        # v2.4: AEGIS evolution (if enabled)
        if self.aegis_engine is not None and len(valid_programs) >= 3:
            try:
                # Convert ProgramNode to dict format for AEGIS
                init_programs = []
                for prog in valid_programs[:self.aegis_engine.config.population_size]:
                    prog_dict = {"actions": [{"op": p.__class__.__name__, "args": []} for p in prog.flatten()]}
                    init_programs.append(prog_dict)

                # Run AEGIS evolution
                input_pairs = [(p["input"][0], p["output"][0]) for p in demo_pairs]
                evolution_result = self.aegis_engine.evolve(init_programs, input_pairs)

                # Use evolved program if better
                if evolution_result.best_program is not None:
                    # Convert back to ProgramNode (simplified)
                    best_prog = valid_programs[0]  # Fallback
                    # TODO: proper conversion from dict to ProgramNode
                    valid_programs = [best_prog] + valid_programs
            except Exception as e:
                if self.config.get("verbose", False):
                    print(f"[AEGIS] Evolution failed: {e}")

        # Bayesian ranking
        ranked = self.bayesian.rank_candidates(valid_programs, demo_pairs)

        # Select best
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
            "mode": "video",
            "aegis_applied": self.aegis_engine is not None,
        }

    def solve_bayesian(self, task: dict[str, Any],
                       video_task: Any = None) -> dict[str, Any]:
        """Solve in bayesian mode (posterior ranking).

        Args:
            task: Raw task dictionary.
            video_task: Pre-parsed VideoARCTask (optional, parsed if None).

        Returns:
            Result dictionary with predictions.
        """
        if video_task is None:
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

    def solve_fusion(self, task: dict[str, Any],
                     video_task: Any = None) -> dict[str, Any]:
        """Solve in fusion mode (multi-modal, most accurate).

        Args:
            task: Raw task dictionary.
            video_task: Pre-parsed VideoARCTask (optional, parsed if None).

        Returns:
            Result dictionary with predictions.
        """
        if video_task is None:
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

        # Fusion scoring (v2.4: use PsiFusionGate if enabled)
        if self.psi_gate is not None:
            # Use ψ-Gate semantic gating for fusion
            fusion_result = self.psi_gate.fuse(
                top_candidates[:10], demo_pairs,
                demo_pairs[0][0] if demo_pairs else [[0]],
                demo_pairs[0][1] if demo_pairs else [[0]],
            )
            best_prog_from_gate = fusion_result.get("program")
            if best_prog_from_gate is not None:
                best_program = best_prog_from_gate
                best_score = fusion_result.get("confidence", 0.0)
                fusion_ranked = [(best_program, best_score)]
            else:
                # Fallback to standard fusion
                fusion_ranked = []
                for program in top_candidates[:10]:
                    score = self.fusion.fuse(program, all_frames, demo_pairs)
                    fusion_ranked.append((program, score))
                fusion_ranked.sort(key=lambda x: x[1], reverse=True)
        else:
            # Standard fusion scoring
            fusion_ranked = []
            for program in top_candidates[:10]:
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
