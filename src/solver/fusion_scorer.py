"""Multi-modal fusion scoring: G = alpha*G_sym + beta*G_vis + gamma*G_cr.

TOMAS v2.0 upgrade:
- VLM embedding differential fusion: G_vis = ||pred_emb - obs_emb||_L2
- Neural-inspired weight adaptation: prediction error anomaly → increase
  symbolic trust (α↑, β↓), mirroring cortical reweighting after surprise.
- Local embedding fallback using e6/e7 octonion components when VL unavailable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode


class FusionScorer:
    """Multi-modal Gan polarization fusion scorer.

    Combines symbolic consistency (G_sym), visual feature score (G_vis),
    and cross-modal consistency (G_cr) into a unified fusion score.
    Neural-inspired weight adaptation based on prediction error anomaly.

    Attributes:
        alpha: Symbolic reasoning weight.
        beta: Visual feature weight.
        gamma: Cross-modal consistency weight.
        vl_adapter: DeepSeekVLAdapter for visual analysis.
        vl_available: Whether VL API is available.
        prediction_error_history: Recent G_vis values for anomaly detection.
    """

    def __init__(self, config: dict[str, Any], vl_adapter: Any = None) -> None:
        """Initialize the fusion scorer.

        Args:
            config: Fusion config with alpha, beta, gamma, degraded weights,
                prediction_error_history_size, anomaly_sigma_threshold.
            vl_adapter: DeepSeekVLAdapter instance (optional).
        """
        self.alpha: float = config.get("alpha", 0.4)
        self.beta: float = config.get("beta", 0.35)
        self.gamma: float = config.get("gamma", 0.25)
        self.alpha_degraded: float = config.get("alpha_degraded", 0.6)
        self.gamma_degraded: float = config.get("gamma_degraded", 0.4)
        self.vl_adapter = vl_adapter
        self.vl_available: bool = False

        # Neural-inspired parameters
        self.error_history_size: int = config.get("prediction_error_history_size", 5)
        self.anomaly_sigma_threshold: float = config.get("anomaly_sigma_threshold", 2.0)
        self.prediction_error_history: list[float] = []

        # Check VL availability
        if vl_adapter is not None:
            try:
                self.vl_available = vl_adapter.check_availability()
            except Exception:
                self.vl_available = False

    def score_symbolic(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> float:
        """Compute symbolic consistency score G_sym.

        Measures how well the program's predictions match demo outputs.

        Args:
            program: ProgramNode to evaluate.
            demo_pairs: List of demo pairs.

        Returns:
            Symbolic score in [0, 1].
        """
        total_match = 0.0
        total_count = 0

        for pair in demo_pairs:
            input_grids = pair.get("input", [])
            output_grids = pair.get("output", [])
            for i, input_grid in enumerate(input_grids):
                if i >= len(output_grids):
                    continue
                try:
                    predicted = program.apply(input_grid)
                    expected = output_grids[i]

                    if predicted.shape != expected.shape:
                        total_match += 0.0
                    else:
                        match = np.array_equal(predicted, expected)
                        total_match += 1.0 if match else 0.0
                except Exception:
                    total_match += 0.0
                total_count += 1

        if total_count == 0:
            return 0.0
        return total_match / total_count

    def score_visual(self, predicted_frames: list[np.ndarray],
                     observed_frames: list[np.ndarray]) -> float:
        """Compute visual feature score G_vis using VLM embedding differential.

        VLM embedding diff: G_vis = 1 - ||pred_emb - obs_emb||_L2 / max_norm

        Neural analogy: the prediction error signal in visual cortex —
        larger embedding differential = larger surprise = lower score.
        When VL is unavailable, falls back to local octonion e6/e7
        component approximation.

        Args:
            predicted_frames: List of predicted frame arrays.
            observed_frames: List of observed frame arrays.

        Returns:
            Visual score in [0, 1].
        """
        if not predicted_frames or not observed_frames:
            return 0.5

        if self.vl_available and self.vl_adapter is not None:
            return self._score_visual_vl(predicted_frames, observed_frames)
        else:
            return self._score_visual_local(predicted_frames, observed_frames)

    def _score_visual_vl(self, predicted_frames: list[np.ndarray],
                         observed_frames: list[np.ndarray]) -> float:
        """VLM embedding differential scoring.

        Computes ||emb_pred - emb_obs||_L2 and maps to [0, 1] score.

        Args:
            predicted_frames: Predicted frame arrays.
            observed_frames: Observed frame arrays.

        Returns:
            Visual score in [0, 1].
        """
        try:
            pred_embs = self.vl_adapter.get_embeddings(predicted_frames)
            obs_embs = self.vl_adapter.get_embeddings(observed_frames)

            if pred_embs is None or obs_embs is None:
                return 0.5

            pred_arr = np.asarray(pred_embs).flatten()
            obs_arr = np.asarray(obs_embs).flatten()

            # L2 norm of embedding differential
            diff_norm = float(np.linalg.norm(pred_arr - obs_arr))
            max_norm = max(float(np.linalg.norm(pred_arr)),
                          float(np.linalg.norm(obs_arr)), 1.0)

            # Score: 1 = perfect match, 0 = completely different
            score = 1.0 - min(diff_norm / max_norm, 1.0)
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def _score_visual_local(self, predicted_frames: list[np.ndarray],
                            observed_frames: list[np.ndarray]) -> float:
        """Local embedding approximation using octonion e6/e7 components.

        When VL API is unavailable, uses e6_boundary (color mapping)
        and e7_area_ratio (time phase) as a local visual proxy.

        Args:
            predicted_frames: Predicted frame arrays.
            observed_frames: Observed frame arrays.

        Returns:
            Approximate visual score in [0, 1].
        """
        from src.core.octonion_hyperedge import OctonionHyperEdge

        total_score = 0.0
        count = 0

        for pred, obs in zip(predicted_frames, observed_frames):
            try:
                pred_edge = OctonionHyperEdge(pred)
                obs_edge = OctonionHyperEdge(obs)

                # Compare e6 (boundary/color mapping) and e7 (time phase)
                pred_e6 = float(np.mean(pred_edge.to_array()[:, 6]))
                obs_e6 = float(np.mean(obs_edge.to_array()[:, 6]))
                pred_e7 = float(np.mean(pred_edge.to_array()[:, 7]))
                obs_e7 = float(np.mean(obs_edge.to_array()[:, 7]))

                e6_diff = abs(pred_e6 - obs_e6)
                e7_diff = abs(pred_e7 - obs_e7)

                # Combine into local score
                local_score = 1.0 - 0.5 * (e6_diff + e7_diff)
                total_score += max(0.0, local_score)
                count += 1
            except Exception:
                total_score += 0.5
                count += 1

        if count == 0:
            return 0.5
        return max(0.0, min(1.0, total_score / count))

    def score_cross_modal(
        self,
        symbolic_score: float,
        visual_score: float,
    ) -> float:
        """Compute cross-modal consistency score G_cr.

        Measures the agreement between symbolic and visual scores
        using cosine similarity.

        Args:
            symbolic_score: Symbolic consistency score.
            visual_score: Visual feature score.

        Returns:
            Cross-modal score in [0, 1].
        """
        # Cosine similarity between two scalar scores treated as 1D vectors
        if symbolic_score == 0 and visual_score == 0:
            return 0.0

        sym_norm = symbolic_score
        vis_norm = visual_score

        sym_mag = abs(sym_norm)
        vis_mag = abs(vis_norm)

        if sym_mag == 0 or vis_mag == 0:
            return 0.0

        cosine = (sym_norm * vis_norm) / (sym_mag * vis_mag)
        return float(max(0.0, min(1.0, cosine)))

    def fuse(
        self,
        program: ProgramNode,
        frames: list[np.ndarray],
        demo_pairs: list[dict[str, Any]],
        observed_frames: list[np.ndarray] | None = None,
    ) -> float:
        """Compute fused score G = alpha*G_sym + beta*G_vis + gamma*G_cr.

        TOMAS v2.0: G_vis now uses predicted vs observed embedding differential.

        Args:
            program: ProgramNode to evaluate.
            frames: Predicted video frames for visual analysis.
            demo_pairs: Demo pairs for symbolic verification.
            observed_frames: Observed frames for differential scoring.

        Returns:
            Fused score in [0, 1].
        """
        g_sym = self.score_symbolic(program, demo_pairs)

        if observed_frames is None:
            observed_frames = []

        if self.vl_available:
            g_vis = self.score_visual(frames, observed_frames)
            g_cr = self.score_cross_modal(g_sym, g_vis)
            score = self.alpha * g_sym + self.beta * g_vis + self.gamma * g_cr
        else:
            # Degraded mode: use local embedding approximation
            g_vis_local = self._score_visual_local(frames, observed_frames) if observed_frames else 0.0
            g_cr = self.score_cross_modal(g_sym, g_vis_local) if g_vis_local > 0 else g_sym
            score = self.alpha_degraded * g_sym + self.gamma_degraded * g_cr

        return float(max(0.0, min(1.0, score)))

    def adapt_weights(self, evidence_count: int,
                      recent_g_vis: float | None = None) -> None:
        """Adaptively adjust fusion weights with neural-inspired anomaly detection.

        Standard adaptation: more evidence → more symbolic weight.
        Neural-inspired: if recent G_vis exceeds 2σ of history mean,
        reduce beta (visual trust) and increase alpha (symbolic trust).
        Mirrors cortex reweighting after prediction error.

        Args:
            evidence_count: Number of demo pairs available.
            recent_g_vis: Latest visual score for anomaly detection.
        """
        # Base adaptation from evidence count
        if evidence_count >= 5:
            self.alpha = 0.5
            self.beta = 0.25
            self.gamma = 0.25
        elif evidence_count >= 2:
            self.alpha = 0.4
            self.beta = 0.35
            self.gamma = 0.25
        else:
            self.alpha = 0.3
            self.beta = 0.45
            self.gamma = 0.25

        # Neural-inspired anomaly detection
        if recent_g_vis is not None:
            self.prediction_error_history.append(recent_g_vis)
            if len(self.prediction_error_history) > self.error_history_size:
                self.prediction_error_history.pop(0)

            if len(self.prediction_error_history) >= 3:
                mean_vis = float(np.mean(self.prediction_error_history))
                std_vis = float(np.std(self.prediction_error_history)) + 1e-8

                # If recent visual error > 2σ, trust symbolic more
                if recent_g_vis < mean_vis - self.anomaly_sigma_threshold * std_vis:
                    # Visual prediction failed → trust symbolic
                    self.alpha = min(0.7, self.alpha + 0.15)
                    self.beta = max(0.1, self.beta - 0.15)

    def degraded_fuse(
        self,
        program: ProgramNode,
        demo_pairs: list[dict[str, Any]],
    ) -> float:
        """Degraded fusion without VL API (Kaggle environment).

        G = alpha_degraded * G_sym + gamma_degraded * G_cr
        where G_cr is computed from symbolic score alone.

        Args:
            program: ProgramNode to evaluate.
            demo_pairs: Demo pairs.

        Returns:
            Degraded fusion score in [0, 1].
        """
        g_sym = self.score_symbolic(program, demo_pairs)
        # Without visual, G_cr is based on symbolic self-consistency
        g_cr = g_sym  # Trivially consistent with itself
        score = self.alpha_degraded * g_sym + self.gamma_degraded * g_cr
        return float(max(0.0, min(1.0, score)))

    def set_vl_available(self, available: bool) -> None:
        """Set VL API availability flag.

        Args:
            available: Whether VL API is available.
        """
        self.vl_available = available
