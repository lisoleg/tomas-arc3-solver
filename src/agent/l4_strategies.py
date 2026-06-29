"""
src/agent/l4_strategies.py
TOMAS 四层混合搜索 L4 决策融合策略

L4层负责从L3评估后的候选中做最终选择(决策融合)。

不同L4策略:
  - KappaSelector: κ-优选η升序 + Liu-Score双约束选择
  - LiuSelector: 纯Liu-Score优先选择
  - ★ BayesianKappaSelector: Bayesian-RHAE融合选择 (升级5新增)

★ 升级5新增:
  - needs_critique: 空候选时标记, 触发critique_self_loop (文章 §14.1)
  - Bayesian-RHAE融合: prior_confidence × w_prior + (1-η/δ_K) × w_kappa + rhae × w_rhae
  - ψ-Audit日志: 每次选择操作记录审计轨迹

所有策略类实现L4DecisionSelector Protocol:
  select(evaluated_set) → List[Dict[str, Any]]
  confidence(eta) → float

Version: v3.20.1 — κ-优选升级: critique_self_loop + Bayesian-RHAE + ψ-Audit
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .hybrid_search_engine import EvaluatedCandidateSet
from .kappa_selector import (
    KappaEtaAscendSelector,
    LiuPrioritySelector,
    BayesianRHAESSelector,
    PsiAuditLog,
    PsiAuditEntry,
    get_psi_audit_log,
    KAPPA_DELTA_K,
)


# ============================================================================
# §1. KappaSelector — κ-优选η升序 + Liu-Score双约束 (★ 升级5: needs_critique + ψ-Audit)
# ============================================================================

class KappaSelector:
    """κ-优选η升序选择器 — L4决策融合策略 + early-stop优化。

    使用kappa_selector.KappaEtaAscendSelector做最终选择:
      1. η升序排序 (κ-优选)
      2. 计算Liu-Score
      3. 背景剪枝 (Liu-Score < DEAD_ZERO_RATIO → 丢弃)
      4. Liu-Score降序排列
      5. 取前max_select个最优候选

    ★ v4.3 优化:
      - early-stop: η < DELTA_K_EARLY → 立即返回最优候选
      - 检查L3的early_stop标记 → 有则立即返回该候选

    ★ 升级5新增:
      - needs_critique: 空候选时标记为True, 触发critique_self_loop
      - critique_diagnosis: 诊断信息 (critique_self_loop触发时)
      - psi_audit: ψ-Audit日志, 记录每次选择操作的审计轨迹

    Attributes:
        selector: KappaEtaAscendSelector实例。
        needs_critique: 是否需要触发critique_self_loop。
        critique_diagnosis: 诊断信息。
        DELTA_K_EARLY: early-stop阈值。
    """

    DELTA_K_EARLY: float = 0.005  # early-stop threshold

    def __init__(
        self,
        delta_k: float = KAPPA_DELTA_K,
        max_select: int = 10,
    ) -> None:
        """初始化κ-优选η升序选择器。

        Args:
            delta_k: κ-Snap残差阈值δ_K，默认0.036。
            max_select: 最大选择数量，默认10。
        """
        self._selector: KappaEtaAscendSelector = KappaEtaAscendSelector(
            delta_k=delta_k,
            max_select=max_select,
        )

    @property
    def needs_critique(self) -> bool:
        """★ 是否需要触发critique_self_loop (空候选时True)。"""
        return self._selector.needs_critique

    @property
    def critique_diagnosis(self) -> str:
        """★ critique_self_loop触发时的诊断信息。"""
        return self._selector.critique_diagnosis

    def select(
        self,
        evaluated_set: EvaluatedCandidateSet,
    ) -> List[Dict[str, Any]]:
        """κ-优选η升序选择 + early-stop。

        ★ v4.3 优化:
          1. 检查L3的early_stop标记 → 有则立即返回该候选
          2. η升序排序候选
          3. early-stop: η < DELTA_K_EARLY → 立即返回该候选
          4. 无early-stop → 委托KappaEtaAscendSelector做完整选择

        ★ 升级5: 空候选时通过needs_critique属性标记触发critique_self_loop。
        ★ 升级5: 每次select通过底层KappaEtaAscendSelector记录ψ-Audit日志。

        Args:
            evaluated_set: L3评估后的候选集。

        Returns:
            最优候选列表(按Liu-Score降序)。
        """
        candidates: List[Dict[str, Any]] = evaluated_set.candidates

        # ★ v4.3: Check for early_stop flag from L3
        for c in candidates:
            if c.get('early_stop', False):
                return [c]  # L3 already found a great candidate (WIN state or grid-mode perfect match)

        # BUGFIX v4.3.1: REMOVED standalone eta < DELTA_K_EARLY early-stop.
        # For game-mode self-comparison examples, many non-solving candidates
        # have near-zero eta (grid similar to start) — this would return
        # a non-solving plan. Only trust L3's explicit early_stop flag.

        # No early-stop → delegate to KappaEtaAscendSelector for full selection
        return self._selector.select(evaluated_set.candidates)

    def confidence(self, eta: float) -> float:
        """计算κ-优选置信度。

        Args:
            eta: GaussEx残差η。

        Returns:
            置信度值 (0~1范围)。
        """
        return self._selector.confidence(eta)


# ============================================================================
# §2. LiuSelector — 纯Liu-Score优先选择 (★ 升级5: needs_critique + ψ-Audit)
# ============================================================================

class LiuSelector:
    """Liu-Score优先选择器 — L4决策融合策略(纯Liu-Score版)。

    使用kappa_selector.LiuPrioritySelector做最终选择，
    不使用η升序预处理，直接按Liu-Score排序选优。

    ★ 升级5新增:
      - needs_critique: 空候选时标记为True, 触发critique_self_loop
      - critique_diagnosis: 诊断信息

    Attributes:
        selector: LiuPrioritySelector实例。
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        max_select: int = 10,
    ) -> None:
        """初始化Liu-Score优先选择器。

        Args:
            epsilon: 防止除零ε，默认0.01。
            max_select: 最大选择数量，默认10。
        """
        self._selector: LiuPrioritySelector = LiuPrioritySelector(
            epsilon=epsilon,
            max_select=max_select,
        )

    @property
    def needs_critique(self) -> bool:
        """★ 是否需要触发critique_self_loop (空候选时True)。"""
        return self._selector.needs_critique

    @property
    def critique_diagnosis(self) -> str:
        """★ critique_self_loop触发时的诊断信息。"""
        return self._selector.critique_diagnosis

    def select(
        self,
        evaluated_set: EvaluatedCandidateSet,
    ) -> List[Dict[str, Any]]:
        """Liu-Score优先选择。

        Args:
            evaluated_set: L3评估后的候选集。

        Returns:
            最优候选列表(按Liu-Score降序)。
        """
        return self._selector.select(evaluated_set.candidates)

    def confidence(self, eta: float) -> float:
        """计算Liu优先置信度。

        Args:
            eta: GaussEx残差η。

        Returns:
            置信度值。
        """
        return self._selector.confidence(eta)


# ============================================================================
# §3. BayesianKappaSelector — Bayesian-RHAE融合选择 (★ 升级5新增)
# ============================================================================

class BayesianKappaSelector:
    """Bayesian-RHAE融合选择器 — L4决策融合策略 (升级5新增)。

    贝叶斯先验 + κ-优选置信度 + RHAE评分 三维度融合选择。

    融合公式:
      bayesian_rhae_score = prior_confidence × BAYESIAN_PRIOR_WEIGHT
                          + (1 - η/δ_K) × BAYESIAN_KAPPA_WEIGHT
                          + rhae_normalized × BAYESIAN_RHAE_WEIGHT

    ★ 升级5新增:
      - needs_critique: 空候选时标记, 触发critique_self_loop
      - critique_diagnosis: 诊断信息
      - psi_audit: ψ-Audit日志, 记录每次选择操作

    Attributes:
        selector: BayesianRHAESSelector实例。
    """

    def __init__(
        self,
        delta_k: float = KAPPA_DELTA_K,
        prior_weight: float = 0.6,
        kappa_weight: float = 0.3,
        rhae_weight: float = 0.1,
        rhae_max: float = 115.0,
        max_select: int = 10,
        prior_confidence: float = 0.5,
    ) -> None:
        """初始化Bayesian-RHAE融合选择器。

        Args:
            delta_k: κ-Snap残差阈值δ_K。
            prior_weight: 贝叶斯先验权重, 默认0.6。
            kappa_weight: κ-优选权重, 默认0.3。
            rhae_weight: RHAE评分权重, 默认0.1。
            rhae_max: RHAE满分值, 默认115.0。
            max_select: 最大选择数量, 默认10。
            prior_confidence: 贝叶斯先验置信度初始值, 默认0.5。
        """
        self._selector: BayesianRHAESSelector = BayesianRHAESSelector(
            delta_k=delta_k,
            prior_weight=prior_weight,
            kappa_weight=kappa_weight,
            rhae_weight=rhae_weight,
            rhae_max=rhae_max,
            max_select=max_select,
        )
        self._selector.set_prior_confidence(prior_confidence)

    @property
    def needs_critique(self) -> bool:
        """★ 是否需要触发critique_self_loop (空候选时True)。"""
        return self._selector.needs_critique

    @property
    def critique_diagnosis(self) -> str:
        """★ critique_self_loop触发时的诊断信息。"""
        return self._selector.critique_diagnosis

    def set_prior_confidence(self, prior: float) -> None:
        """★ 设置贝叶斯先验置信度。

        先验可以从历史轨迹或Inflow模型推断得出。

        Args:
            prior: 先验置信度值 (0~1范围)。
        """
        self._selector.set_prior_confidence(prior)

    def select(
        self,
        evaluated_set: EvaluatedCandidateSet,
    ) -> List[Dict[str, Any]]:
        """Bayesian-RHAE融合选择。

        Args:
            evaluated_set: L3评估后的候选集。

        Returns:
            最优候选列表(按Bayesian-RHAE分数降序)。
        """
        return self._selector.select(evaluated_set.candidates)

    def confidence(self, eta: float) -> float:
        """计算Bayesian-RHAE融合置信度。

        Args:
            eta: GaussEx残差η。

        Returns:
            融合置信度值 (0~1范围)。
        """
        return self._selector.confidence(eta)


# ============================================================================
# §4. L4策略注册到HybridSearchPipeline
# ============================================================================

def register_l4_strategies() -> None:
    """将所有L4策略注册到HybridSearchPipeline.L4_REGISTRY。"""
    from .hybrid_search_engine import HybridSearchPipeline

    HybridSearchPipeline.L4_REGISTRY['kappa_selector'] = KappaSelector
    HybridSearchPipeline.L4_REGISTRY['liu_priority'] = LiuSelector
    HybridSearchPipeline.L4_REGISTRY['bayesian_rhae'] = BayesianKappaSelector  # ★ 升级5新增


# 自动注册
register_l4_strategies()


# ============================================================================
# §5. 自测函数
# ============================================================================

def _self_test() -> bool:
    """L4策略自测: 验证KappaSelector/LiuSelector/BayesianKappaSelector + needs_critique + ψ-Audit。

    Returns:
        True if all tests pass, False otherwise.
    """
    from .hybrid_search_engine import HybridSearchPipeline

    # 验证注册 (★ 新增bayesian_rhae)
    assert 'kappa_selector' in HybridSearchPipeline.L4_REGISTRY, "kappa_selector should be registered"
    assert 'liu_priority' in HybridSearchPipeline.L4_REGISTRY, "liu_priority should be registered"
    assert 'bayesian_rhae' in HybridSearchPipeline.L4_REGISTRY, "★ bayesian_rhae should be registered"

    # 测试KappaSelector (★ 使用η < δ_K的候选，确保有有效选择)
    ecs: EvaluatedCandidateSet = EvaluatedCandidateSet(
        candidates=[
            {'node_id': 1, 'eta': 0.03, 'ic': 0.5, 'depth': 3},  # confidence ≈ 0.17 ≥ 1/6
            {'node_id': 2, 'eta': 0.02, 'ic': 0.3, 'depth': 2},  # confidence ≈ 0.44 ≥ 1/6
        ],
    )
    kappa_sel: KappaSelector = KappaSelector()
    kappa_result: List[Dict[str, Any]] = kappa_sel.select(ecs)
    assert isinstance(kappa_result, list), "KappaSelector should return list"
    # 置信度
    conf: float = kappa_sel.confidence(0.03)
    assert conf > 0, "Confidence should be positive"
    # ★ needs_critique应为False (有有效候选，η < δ_K)
    assert not kappa_sel.needs_critique, "★ Should not need critique with valid candidates (η < δ_K)"

    # 测试LiuSelector
    liu_sel: LiuSelector = LiuSelector()
    liu_result: List[Dict[str, Any]] = liu_sel.select(ecs)
    assert isinstance(liu_result, list), "LiuSelector should return list"

    # ★ 测试BayesianKappaSelector
    br_sel: BayesianKappaSelector = BayesianKappaSelector(prior_confidence=0.6)
    br_result: List[Dict[str, Any]] = br_sel.select(ecs)
    assert isinstance(br_result, list), "★ BayesianKappaSelector should return list"
    # 融合置信度
    br_conf: float = br_sel.confidence(0.03)
    assert br_conf > 0, "★ Bayesian-RHAE confidence should be positive"
    # 设置先验
    br_sel.set_prior_confidence(0.8)
    assert not br_sel.needs_critique, "★ Should not need critique with valid candidates"

    # ★ 测试空候选触发needs_critique
    empty_ecs: EvaluatedCandidateSet = EvaluatedCandidateSet(
        candidates=[
            {'node_id': 100, 'eta': 1.0, 'ic': 0.5, 'depth': 3},
        ],
    )
    empty_result: List[Dict[str, Any]] = kappa_sel.select(empty_ecs)
    assert len(empty_result) == 0, "★ All η > δ_K → should return empty"
    assert kappa_sel.needs_critique, "★ Should need critique when all η > δ_K"

    # ★ 测试ψ-Audit日志
    audit_log: PsiAuditLog = get_psi_audit_log()
    assert len(audit_log.entries) > 0, "★ ψ-Audit log should have entries"
    critique_entries = audit_log.query(needs_critique=True)
    assert len(critique_entries) > 0, "★ Should have critique entries in audit log"

    print("[PASS] l4_strategies _self_test passed (★ includes needs_critique + Bayesian-RHAE + ψ-Audit)")
    return True


if __name__ == "__main__":
    _self_test()
