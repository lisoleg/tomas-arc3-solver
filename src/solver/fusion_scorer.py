"""Multi-modal fusion scoring: G = alpha*G_sym + beta*G_vis + gamma*G_cr."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.core.dsl_primitives import ProgramNode


class FusionScorer:
    """Multi-modal Gan polarization fusion scorer.

    Combines symbolic consistency (G_sym), visual feature score (G_vis),
    and cross-modal consistency (G_cr) into a unified fusion score.

    Attributes:
        alpha: Symbolic reasoning weight.
        beta: Visual feature weight.
        gamma: Cross-modal consistency weight.
        vl_adapter: DeepSeekVLAdapter for visual analysis.
        vl_available: Whether VL API is available.
    """

    def __init__(self, config: dict[str, Any], vl_adapter: Any = None) -> None:
        """Initialize the fusion scorer.

        Args:
            config: Fusion config with alpha, beta, gamma, degraded weights.
            vl_adapter: DeepSeekVLAdapter instance (optional).
        """
        self.alpha: float = config.get("alpha", 0.4)
        self.beta: float = config.get("beta", 0.35)
        self.gamma: float = config.get("gamma", 0.25)
        self.alpha_degraded: float = config.get("alpha_degraded", 0.6)
        self.gamma_degraded: float = config.get("gamma_degraded", 0.4)
        self.vl_adapter = vl_adapter
        self.vl_available: bool = False

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

    def score_visual(self, frames: list[np.ndarray]) -> float:
        """Compute visual feature score G_vis using VL embeddings.

        Uses VL adapter to extract embeddings and compute frame consistency.

        Args:
            frames: List of frame arrays.

        Returns:
            Visual score in [0, 1].
        """
        if not self.vl_available or self.vl_adapter is None:
            return 0.0

        try:
            embeddings = self.vl_adapter.get_embeddings(frames)
            if embeddings is None or len(embeddings) < 2:
                return 0.5  # Neutral score

            # Compute embedding consistency (lower variance = more consistent)
            embeddings = np.asarray(embeddings)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # Pairwise cosine similarity
            normalized = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            )
            sim_matrix = normalized @ normalized.T
            # Average off-diagonal similarity
            n = sim_matrix.shape[0]
            if n > 1:
                mask = ~np.eye(n, dtype=bool)
                avg_sim = float(np.mean(sim_matrix[mask]))
                return max(0.0, min(1.0, avg_sim))
            return 0.5
        except Exception:
            return 0.5

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
    ) -> float:
        """Compute fused score G = alpha*G_sym + beta*G_vis + gamma*G_cr.

        Args:
            program: ProgramNode to evaluate.
            frames: Video frames for visual analysis.
            demo_pairs: Demo pairs for symbolic verification.

        Returns:
            Fused score in [0, 1].
        """
        g_sym = self.score_symbolic(program, demo_pairs)

        if self.vl_available:
            g_vis = self.score_visual(frames)
            g_cr = self.score_cross_modal(g_sym, g_vis)
            score = self.alpha * g_sym + self.beta * g_vis + self.gamma * g_cr
        else:
            # Degraded mode: no visual score
            score = self.degraded_fuse(program, demo_pairs)

        return float(max(0.0, min(1.0, score)))

    def adapt_weights(self, evidence_count: int) -> None:
        """Adaptively adjust fusion weights based on evidence count.

        With more evidence (demo pairs), increase symbolic weight.
        With less evidence, increase visual weight for robustness.

        Args:
            evidence_count: Number of demo pairs available.
        """
        if evidence_count >= 5:
            # Strong symbolic evidence
            self.alpha = 0.5
            self.beta = 0.25
            self.gamma = 0.25
        elif evidence_count >= 2:
            # Balanced
            self.alpha = 0.4
            self.beta = 0.35
            self.gamma = 0.25
        else:
            # Weak evidence, rely more on visual
            self.alpha = 0.3
            self.beta = 0.45
            self.gamma = 0.25

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
