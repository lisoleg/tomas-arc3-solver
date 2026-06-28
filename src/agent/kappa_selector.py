"""
src/agent/kappa_selector.py
TOMAS κ-优选η升序 — 四层混合搜索 L4 核心决策融合

κ-优选η升序选择器: 从L3评估后的候选中，按η(残差)升序选择最小残差候选。
置信度公式: confidence = 1 - η/δ_K (类比量子态纯度)

设计原则:
  1. η升序: 残差最小的候选优先选择 (最接近目标的候选)
  2. 置信度阈值: confidence ≥ 1/6 (卞氏5/6饱和) → 通过
  3. κ-优选: η升序 + 卞氏阈值双约束
  4. Liu-Score排序: 深度×0.1 - 0.5×IC + 2.0×(1-η) → Liu优先级
  5. ★ 升级5: 空候选触发critique_self_loop (文章 §14.1)
  6. ★ 升级5: Bayesian-RHAE融合 (贝叶斯先验+RHAE评分)
  7. ★ 升级5: ψ-Audit日志 (审计轨迹记录)

Version: v3.20.1 — κ-优选升级: critique_self_loop + Bayesian-RHAE + ψ-Audit
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .delta_state import (
    Node,
    ReplayEngine,
    GEX_PASS_THRESHOLD,
    DEAD_ZERO_RATIO,
)


# ============================================================================
# §0. ψ-Audit日志 (升级5: 审计轨迹记录)
# ============================================================================

@dataclass
class PsiAuditEntry:
    """ψ-Audit审计轨迹条目 — 每次L4选择操作的审计记录。

    κ-Phase: ψ审计 = κ-Snap自指残差校验的全链路日志
    记录每次选择操作的完整决策轨迹，用于事后分析和调试。

    Attributes:
        selector: 选择器名称 (e.g. 'kappa_selector', 'bayesian_rhae')。
        eta: GaussEx残差η。
        confidence: κ-优选置信度 = 1 - η/δ_K。
        liu_score: Liu-Score优先级。
        bayesian_rhae_score: Bayesian-RHAE融合分数 (升级5新增)。
        timestamp: 审计记录时间戳。
        node_id: 候选节点ID。
        needs_critique: 是否需要触发critique_self_loop。
        diagnosis: 诊断信息 (critique_self_loop触发时)。
    """

    selector: str = ""
    eta: float = 1.0
    confidence: float = 0.0
    liu_score: float = 0.0
    bayesian_rhae_score: float = 0.0
    timestamp: float = 0.0
    node_id: int = -1
    needs_critique: bool = False
    diagnosis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式(用于JSON序列化)。"""
        return {
            'selector': self.selector,
            'eta': self.eta,
            'confidence': self.confidence,
            'liu_score': self.liu_score,
            'bayesian_rhae_score': self.bayesian_rhae_score,
            'timestamp': self.timestamp,
            'node_id': self.node_id,
            'needs_critique': self.needs_critique,
            'diagnosis': self.diagnosis,
        }


class PsiAuditLog:
    """ψ-Audit审计日志管理器 — 记录L4决策融合的全链路轨迹。

    κ-Phase: ψ审计日志 = 每次select操作的审计轨迹
    支持追加、查询、导出功能，用于事后分析和调试。

    Attributes:
        entries: 审计轨迹条目列表。
        max_entries: 最大条目数 (防止内存溢出)。
    """

    def __init__(self, max_entries: int = 10000) -> None:
        """初始化ψ-Audit日志管理器。

        Args:
            max_entries: 最大条目数，默认10000。
        """
        self.entries: List[PsiAuditEntry] = []
        self.max_entries: int = max_entries

    def append(self, entry: PsiAuditEntry) -> None:
        """追加审计条目。

        Args:
            entry: PsiAuditEntry实例。
        """
        if len(self.entries) >= self.max_entries:
            # 保留最近80%的条目 (防止内存溢出)
            self.entries = self.entries[-int(self.max_entries * 0.8):]
        self.entries.append(entry)

    def append_batch(self, entries: List[PsiAuditEntry]) -> None:
        """批量追加审计条目。

        Args:
            entries: PsiAuditEntry列表。
        """
        for entry in entries:
            self.append(entry)

    def query(
        self,
        selector: Optional[str] = None,
        needs_critique: Optional[bool] = None,
        min_eta: Optional[float] = None,
        max_eta: Optional[float] = None,
    ) -> List[PsiAuditEntry]:
        """查询审计条目。

        Args:
            selector: 按选择器名过滤。
            needs_critique: 按critique_self_loop触发标志过滤。
            min_eta: 最小η阈值过滤。
            max_eta: 最大η阈值过滤。

        Returns:
            过滤后的审计条目列表。
        """
        result: List[PsiAuditEntry] = []
        for entry in self.entries:
            if selector is not None and entry.selector != selector:
                continue
            if needs_critique is not None and entry.needs_critique != needs_critique:
                continue
            if min_eta is not None and entry.eta < min_eta:
                continue
            if max_eta is not None and entry.eta > max_eta:
                continue
            result.append(entry)
        return result

    def export_json(self) -> str:
        """导出审计日志为JSON字符串。

        Returns:
            JSON格式的审计日志。
        """
        return json.dumps([e.to_dict() for e in self.entries], indent=2)

    def summary(self) -> Dict[str, Any]:
        """生成审计日志摘要统计。

        Returns:
            摘要字典: {total_entries, critique_count, avg_eta, avg_confidence, ...}
        """
        if len(self.entries) == 0:
            return {'total_entries': 0, 'critique_count': 0}

        critique_count: int = sum(1 for e in self.entries if e.needs_critique)
        etas: List[float] = [e.eta for e in self.entries]
        confs: List[float] = [e.confidence for e in self.entries]
        lius: List[float] = [e.liu_score for e in self.entries]

        return {
            'total_entries': len(self.entries),
            'critique_count': critique_count,
            'avg_eta': sum(etas) / len(etas),
            'avg_confidence': sum(confs) / len(confs),
            'avg_liu_score': sum(lius) / len(lius),
            'min_eta': min(etas),
            'max_eta': max(etas),
        }

    def clear(self) -> None:
        """清空审计日志。"""
        self.entries.clear()


# 全局ψ-Audit日志实例 (所有选择器共享)
_GLOBAL_PSI_AUDIT: PsiAuditLog = PsiAuditLog()


def get_psi_audit_log() -> PsiAuditLog:
    """获取全局ψ-Audit日志实例。

    Returns:
        全局PsiAuditLog实例。
    """
    return _GLOBAL_PSI_AUDIT


# ============================================================================
# §1. κ-优选常量
# ============================================================================

# δ_K: κ-Snap GaussEx残差阈值 (from KSnapEngine)
KAPPA_DELTA_K: float = 0.036

# 卞氏5/6饱和置信度阈值
# confidence ≥ KAPPA_MIN_CONFIDENCE → 通过
KAPPA_MIN_CONFIDENCE: float = 1.0 - 5.0 / 6.0  # = 1/6 ≈ 0.167

# Liu-Score ε (防止除零)
LIU_EPSILON: float = 0.01

# ★ 升级5: Bayesian-RHAE融合权重
BAYESIAN_PRIOR_WEIGHT: float = 0.6  # 贝叶斯先验权重 (Prior confidence)
BAYESIAN_KAPPA_WEIGHT: float = 0.3  # κ-优选权重 (1-η/δ_K)
BAYESIAN_RHAE_WEIGHT: float = 0.1  # RHAE评分权重

# ★ 升级5: RHAE评分基准 (满分115.0)
DEFAULT_RHAE_MAX: float = 115.0

# 背景剪枝: Liu-Score < DEAD_ZERO_RATIO → 丢弃


# ============================================================================
# §2. κ-优选η升序核心算法
# ============================================================================

def kappa_eta_ascend_sort(
    candidates: List[Dict[str, Any]],
    delta_k: float = KAPPA_DELTA_K,
) -> List[Dict[str, Any]]:
    """κ-优选η升序排序: 按残差η升序排列候选。

    核心思想: η越小 → 残差越小 → 越接近目标 → 优先选择。
    置信度 confidence = 1 - η/δ_K，η升序选择最小残差。

    Args:
        candidates: L3评估后的候选列表，每个候选包含:
            - 'node_id': 节点ID
            - 'eta': GaussEx残差η
            - 'confidence': 置信度 = 1 - η/δ_K
            - 'gex_result': GaussEx验证结果
            - 'ic': 信息基数IC
            - 'depth': 搜索深度
        delta_k: κ-Snap残差阈值δ_K，默认0.036。

    Returns:
        η升序排序后的候选列表 (η最小在前)。
    """
    if len(candidates) == 0:
        return []

    # 计算置信度并过滤
    valid_candidates: List[Dict[str, Any]] = []
    for cand in candidates:
        eta: float = cand.get('eta', 1.0)
        confidence: float = 1.0 - eta / delta_k if delta_k > 0 else 0.0
        cand['confidence'] = confidence

        # 卞氏5/6饱和阈值过滤: confidence ≥ 1/6 → 有效
        if confidence >= KAPPA_MIN_CONFIDENCE:
            valid_candidates.append(cand)

    # η升序排序 (残差最小在前)
    valid_candidates.sort(key=lambda x: x.get('eta', 1.0))

    return valid_candidates


def compute_liu_score(
    candidate: Dict[str, Any],
    epsilon: float = LIU_EPSILON,
) -> float:
    """计算Liu-Score优先级。

    Liu-Score = 1 / (S_rel + ε)
    S_rel = 0.1 × num_primitives - 0.5 × IC + 2.0 × (1 - η)
    其中 num_primitives ≈ depth, IC = 信息基数, η = 残差。

    Args:
        candidate: 候选字典，包含 'depth', 'ic', 'eta'。
        epsilon: 防止除零的ε，默认0.01。

    Returns:
        Liu-Score优先级值。
    """
    depth: int = candidate.get('depth', 1)
    ic: float = candidate.get('ic', 0.5)
    eta: float = candidate.get('eta', 1.0)
    gex_score: float = 1.0 - eta  # GEX = 1 - η (一致性)

    s_rel: float = 0.1 * depth - 0.5 * ic + 2.0 * gex_score
    liu_score: float = 1.0 / (s_rel + epsilon)

    return liu_score


def kappa_priority_select(
    candidates: List[Dict[str, Any]],
    delta_k: float = KAPPA_DELTA_K,
    max_select: int = 10,
) -> List[Dict[str, Any]]:
    """κ-优选η升序 + Liu-Score双约束选择。

    步骤:
      1. η升序排序 (κ-优选)
      2. 计算Liu-Score
      3. 背景剪枝: Liu-Score < DEAD_ZERO_RATIO → 丢弃
      4. 按Liu-Score降序排列 (最高优先级在前)
      5. 取前max_select个最优候选

    Args:
        candidates: L3评估后的候选列表。
        delta_k: κ-Snap残差阈值δ_K。
        max_select: 最大选择数量。

    Returns:
        κ-优选η升序 + Liu-Score双约束后的最优候选列表。
    """
    # Step 1: η升序排序
    eta_sorted: List[Dict[str, Any]] = kappa_eta_ascend_sort(candidates, delta_k)

    if len(eta_sorted) == 0:
        return []

    # Step 2: 计算Liu-Score
    scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
    for cand in eta_sorted:
        liu_score: float = compute_liu_score(cand)
        cand['liu_score'] = liu_score

        # Step 3: 背景剪枝
        if liu_score < DEAD_ZERO_RATIO:
            continue  # 低价值候选丢弃

        scored_candidates.append((liu_score, cand))

    # Step 4: Liu-Score降序排列
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Step 5: 取前max_select个
    result: List[Dict[str, Any]] = [
        cand for _, cand in scored_candidates[:max_select]
    ]

    return result


# ============================================================================
# §3. κ-优选η升序选择器类 (L4 Strategy Protocol实现)
# ============================================================================

class KappaEtaAscendSelector:
    """κ-优选η升序选择器 — L4决策融合策略。

    实现4层Protocol L4Selector:
      - select(candidates): κ-优选η升序 + Liu-Score双约束选择
      - confidence(node_id, eta): 计算置信度 = 1 - η/δ_K

    ★ 升级5新增:
      - needs_critique: 空候选时标记为True, 触发critique_self_loop
      - psi_audit: ψ-Audit日志, 记录每次select操作的审计轨迹
      - critique_diagnosis: critique_self_loop触发时的诊断信息

    Attributes:
        delta_k: κ-Snap残差阈值δ_K。
        max_select: 最大选择数量。
        needs_critique: 是否需要触发critique_self_loop (空候选时True)。
        critique_diagnosis: 诊断信息 (critique_self_loop触发时)。
        psi_audit: ψ-Audit日志实例。
    """

    def __init__(
        self,
        delta_k: float = KAPPA_DELTA_K,
        max_select: int = 10,
        psi_audit: Optional[PsiAuditLog] = None,
    ) -> None:
        """初始化κ-优选η升序选择器。

        Args:
            delta_k: κ-Snap残差阈值δ_K，默认0.036。
            max_select: 最大选择数量，默认10。
            psi_audit: ψ-Audit日志实例, 默认使用全局实例。
        """
        self.delta_k: float = delta_k
        self.max_select: int = max_select
        self.needs_critique: bool = False  # ★ 升级5: 空候选触发critique标志
        self.critique_diagnosis: str = ""  # ★ 升级5: 诊断信息
        self.psi_audit: PsiAuditLog = psi_audit or get_psi_audit_log()  # ★ 升级5: ψ-Audit日志

    def select(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """κ-优选η升序选择: 从L3评估后的候选中选最优。

        ★ 升级5: 空候选时设置needs_critique=True, 触发critique_self_loop。
        ★ 升级5: 每次select记录ψ-Audit日志。

        Args:
            candidates: L3评估后的候选列表。

        Returns:
            最优候选列表(按Liu-Score降序)。
            空候选时设置self.needs_critique=True。
        """
        result: List[Dict[str, Any]] = kappa_priority_select(
            candidates, self.delta_k, self.max_select,
        )

        # ★ 升级5: 空候选 → 触发critique_self_loop
        if len(result) == 0:
            self.needs_critique = True
            self.critique_diagnosis = (
                f"κ-优选选不出候选: all η > δ_K({self.delta_k:.4f}), "
                f"candidates={len(candidates)}, "
                f"best_eta={min(c.get('eta',1.0) for c in candidates) if candidates else 'N/A'}"
            )

            # ★ 记录空候选触发ψ-Audit
            self.psi_audit.append(PsiAuditEntry(
                selector='kappa_eta_ascend',
                eta=min(c.get('eta', 1.0) for c in candidates) if candidates else 1.0,
                confidence=0.0,
                liu_score=0.0,
                bayesian_rhae_score=0.0,
                timestamp=time.time(),
                node_id=-1,
                needs_critique=True,
                diagnosis=self.critique_diagnosis,
            ))

        else:
            self.needs_critique = False
            self.critique_diagnosis = ""

            # ★ 记录成功选择ψ-Audit日志
            audit_entries: List[PsiAuditEntry] = []
            for cand in result:
                audit_entries.append(PsiAuditEntry(
                    selector='kappa_eta_ascend',
                    eta=cand.get('eta', 1.0),
                    confidence=cand.get('confidence', 0.0),
                    liu_score=cand.get('liu_score', 0.0),
                    bayesian_rhae_score=cand.get('bayesian_rhae_score', 0.0),
                    timestamp=time.time(),
                    node_id=cand.get('node_id', -1),
                    needs_critique=False,
                    diagnosis="",
                ))
            self.psi_audit.append_batch(audit_entries)

        return result

    def confidence(self, eta: float) -> float:
        """计算κ-优选置信度。

        confidence = 1 - η/δ_K
        η越小 → confidence越大 → 越接近目标。

        Args:
            eta: GaussEx残差η。

        Returns:
            置信度值 (0~1范围)。
        """
        if self.delta_k <= 0:
            return 0.0
        return max(0.0, 1.0 - eta / self.delta_k)


class LiuPrioritySelector:
    """Liu-Score优先选择器 — L4决策融合策略(纯Liu-Score版)。

    不使用η升序预处理，直接按Liu-Score排序选优。
    Liu-Score = 1 / (S_rel + ε)，S_rel考虑深度、IC、GEX。

    ★ 升级5新增:
      - needs_critique: 空候选时标记为True, 触发critique_self_loop
      - psi_audit: ψ-Audit日志, 记录每次select操作的审计轨迹

    Attributes:
        epsilon: 防止除零ε。
        max_select: 最大选择数量。
        needs_critique: 是否需要触发critique_self_loop。
        psi_audit: ψ-Audit日志实例。
    """

    def __init__(
        self,
        epsilon: float = LIU_EPSILON,
        max_select: int = 10,
        psi_audit: Optional[PsiAuditLog] = None,
    ) -> None:
        """初始化Liu-Score优先选择器。

        Args:
            epsilon: 防止除零ε，默认0.01。
            max_select: 最大选择数量，默认10。
            psi_audit: ψ-Audit日志实例, 默认使用全局实例。
        """
        self.epsilon: float = epsilon
        self.max_select: int = max_select
        self.needs_critique: bool = False  # ★ 升级5
        self.critique_diagnosis: str = ""  # ★ 升级5
        self.psi_audit: PsiAuditLog = psi_audit or get_psi_audit_log()  # ★ 升级5

    def select(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Liu-Score优先选择: 按Liu-Score降序选优。

        ★ 升级5: 空候选时设置needs_critique=True。
        ★ 升级5: 每次select记录ψ-Audit日志。

        Args:
            candidates: L3评估后的候选列表。

        Returns:
            最优候选列表(按Liu-Score降序)。
        """
        if len(candidates) == 0:
            self.needs_critique = True
            self.critique_diagnosis = "Liu-Score选择器: 无候选"
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for cand in candidates:
            liu_score: float = compute_liu_score(cand, self.epsilon)
            cand['liu_score'] = liu_score

            # 背景剪枝
            if liu_score < DEAD_ZERO_RATIO:
                continue

            scored.append((liu_score, cand))

        result: List[Dict[str, Any]] = [cand for _, cand in sorted(scored, key=lambda x: x[0], reverse=True)[:self.max_select]]

        # ★ 升级5: 空候选 → 触发critique_self_loop
        if len(result) == 0:
            self.needs_critique = True
            self.critique_diagnosis = (
                f"Liu-Score选不出候选: all liu_score < DEAD_ZERO_RATIO({DEAD_ZERO_RATIO}), "
                f"candidates={len(candidates)}"
            )
            self.psi_audit.append(PsiAuditEntry(
                selector='liu_priority',
                eta=min(c.get('eta', 1.0) for c in candidates) if candidates else 1.0,
                confidence=0.0,
                liu_score=0.0,
                bayesian_rhae_score=0.0,
                timestamp=time.time(),
                node_id=-1,
                needs_critique=True,
                diagnosis=self.critique_diagnosis,
            ))
        else:
            self.needs_critique = False
            self.critique_diagnosis = ""
            audit_entries: List[PsiAuditEntry] = []
            for cand in result:
                audit_entries.append(PsiAuditEntry(
                    selector='liu_priority',
                    eta=cand.get('eta', 1.0),
                    confidence=cand.get('confidence', 0.0),
                    liu_score=cand.get('liu_score', 0.0),
                    bayesian_rhae_score=0.0,
                    timestamp=time.time(),
                    node_id=cand.get('node_id', -1),
                    needs_critique=False,
                    diagnosis="",
                ))
            self.psi_audit.append_batch(audit_entries)

        return result

    def confidence(self, eta: float) -> float:
        """计算Liu优先置信度(与KappaEtaAscendSelector相同公式)。

        Args:
            eta: GaussEx残差η。

        Returns:
            置信度值。
        """
        return max(0.0, 1.0 - eta / KAPPA_DELTA_K)


# ============================================================================
# §5. Bayesian-RHAE融合选择器 (升级5新增)
# ============================================================================

class BayesianRHAESSelector:
    """Bayesian-RHAE融合选择器 — L4决策融合策略 (升级5新增)。

    贝叶斯先验 + κ-优选置信度 + RHAE评分 三维度融合选择。

    融合公式:
      bayesian_rhae_score = prior_confidence × BAYESIAN_PRIOR_WEIGHT
                          + (1 - η/δ_K) × BAYESIAN_KAPPA_WEIGHT
                          + rhae_normalized × BAYESIAN_RHAE_WEIGHT

    其中:
      - prior_confidence: 贝叶斯先验置信度 (从历史轨迹或Inflow模型推断)
      - 1 - η/δ_K: κ-优选置信度 (GaussEx残差纯度)
      - rhae_normalized: RHAE评分归一化 = rhae_score / RHAE_MAX

    ★ 升级5: 空候选时触发critique_self_loop
    ★ 升级5: ψ-Audit日志记录每次选择操作

    Attributes:
        delta_k: κ-Snap残差阈值δ_K。
        prior_weight: 贝叶斯先验权重。
        kappa_weight: κ-优选权重。
        rhae_weight: RHAE评分权重。
        rhae_max: RHAE满分值。
        max_select: 最大选择数量。
        needs_critique: 是否需要触发critique_self_loop。
        psi_audit: ψ-Audit日志实例。
    """

    def __init__(
        self,
        delta_k: float = KAPPA_DELTA_K,
        prior_weight: float = BAYESIAN_PRIOR_WEIGHT,
        kappa_weight: float = BAYESIAN_KAPPA_WEIGHT,
        rhae_weight: float = BAYESIAN_RHAE_WEIGHT,
        rhae_max: float = DEFAULT_RHAE_MAX,
        max_select: int = 10,
        psi_audit: Optional[PsiAuditLog] = None,
    ) -> None:
        """初始化Bayesian-RHAE融合选择器。

        Args:
            delta_k: κ-Snap残差阈值δ_K。
            prior_weight: 贝叶斯先验权重, 默认0.6。
            kappa_weight: κ-优选权重, 默认0.3。
            rhae_weight: RHAE评分权重, 默认0.1。
            rhae_max: RHAE满分值, 默认115.0。
            max_select: 最大选择数量, 默认10。
            psi_audit: ψ-Audit日志实例, 默认使用全局实例。
        """
        self.delta_k: float = delta_k
        self.prior_weight: float = prior_weight
        self.kappa_weight: float = kappa_weight
        self.rhae_weight: float = rhae_weight
        self.rhae_max: float = rhae_max
        self.max_select: int = max_select
        self.needs_critique: bool = False
        self.critique_diagnosis: str = ""
        self.psi_audit: PsiAuditLog = psi_audit or get_psi_audit_log()
        self._prior_confidence: float = 0.5  # 默认先验置信度

    def set_prior_confidence(self, prior: float) -> None:
        """设置贝叶斯先验置信度。

        先验可以从历史轨迹或Inflow模型推断得出。

        Args:
            prior: 先验置信度值 (0~1范围)。
        """
        self._prior_confidence = max(0.0, min(1.0, prior))

    def compute_bayesian_rhae_score(self, candidate: Dict[str, Any]) -> float:
        """计算Bayesian-RHAE融合分数。

        融合公式:
          bayesian_rhae_score = prior_confidence × prior_weight
                              + (1 - η/δ_K) × kappa_weight
                              + rhae_normalized × rhae_weight

        Args:
            candidate: 候选字典, 包含 'eta', 'rhae_score', 'confidence' 等。

        Returns:
            Bayesian-RHAE融合分数 (0~1范围)。
        """
        eta: float = candidate.get('eta', 1.0)
        kappa_confidence: float = max(0.0, 1.0 - eta / self.delta_k) if self.delta_k > 0 else 0.0
        rhae_score: float = candidate.get('rhae_score', 0.0)
        rhae_normalized: float = rhae_score / self.rhae_max if self.rhae_max > 0 else 0.0

        bayesian_score: float = (
            self._prior_confidence * self.prior_weight
            + kappa_confidence * self.kappa_weight
            + rhae_normalized * self.rhae_weight
        )

        return max(0.0, min(1.0, bayesian_score))

    def select(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Bayesian-RHAE融合选择: 三维度融合评分选优。

        步骤:
          1. 计算Liu-Score
          2. 计算Bayesian-RHAE融合分数
          3. 背景剪枝: bayesian_rhae_score < KAPPA_MIN_CONFIDENCE → 丢弃
          4. 按Bayesian-RHAE分数降序排列
          5. 取前max_select个最优候选

        ★ 升级5: 空候选时设置needs_critique=True。
        ★ 升级5: 每次select记录ψ-Audit日志。

        Args:
            candidates: L3评估后的候选列表。

        Returns:
            最优候选列表(按Bayesian-RHAE分数降序)。
        """
        if len(candidates) == 0:
            self.needs_critique = True
            self.critique_diagnosis = "Bayesian-RHAE选择器: 无候选"
            return []

        # Step 1-2: 计算Liu-Score和Bayesian-RHAE融合分数
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
        for cand in candidates:
            liu_score: float = compute_liu_score(cand)
            cand['liu_score'] = liu_score
            bayesian_rhae_score: float = self.compute_bayesian_rhae_score(cand)
            cand['bayesian_rhae_score'] = bayesian_rhae_score

            # Step 3: 背景剪枝 (卞氏阈值)
            if bayesian_rhae_score < KAPPA_MIN_CONFIDENCE:
                continue

            scored_candidates.append((bayesian_rhae_score, cand))

        # Step 4: Bayesian-RHAE分数降序排列
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Step 5: 取前max_select个
        result: List[Dict[str, Any]] = [
            cand for _, cand in scored_candidates[:self.max_select]
        ]

        # ★ 升级5: 空候选 → 触发critique_self_loop
        if len(result) == 0:
            self.needs_critique = True
            self.critique_diagnosis = (
                f"Bayesian-RHAE选不出候选: "
                f"prior={self._prior_confidence:.3f}, "
                f"all bayesian_rhae_score < KAPPA_MIN_CONFIDENCE({KAPPA_MIN_CONFIDENCE:.4f}), "
                f"candidates={len(candidates)}"
            )
            self.psi_audit.append(PsiAuditEntry(
                selector='bayesian_rhae',
                eta=min(c.get('eta', 1.0) for c in candidates) if candidates else 1.0,
                confidence=0.0,
                liu_score=0.0,
                bayesian_rhae_score=0.0,
                timestamp=time.time(),
                node_id=-1,
                needs_critique=True,
                diagnosis=self.critique_diagnosis,
            ))
        else:
            self.needs_critique = False
            self.critique_diagnosis = ""
            audit_entries: List[PsiAuditEntry] = []
            for cand in result:
                audit_entries.append(PsiAuditEntry(
                    selector='bayesian_rhae',
                    eta=cand.get('eta', 1.0),
                    confidence=cand.get('confidence', 0.0),
                    liu_score=cand.get('liu_score', 0.0),
                    bayesian_rhae_score=cand.get('bayesian_rhae_score', 0.0),
                    timestamp=time.time(),
                    node_id=cand.get('node_id', -1),
                    needs_critique=False,
                    diagnosis="",
                ))
            self.psi_audit.append_batch(audit_entries)

        return result

    def confidence(self, eta: float) -> float:
        """计算Bayesian-RHAE融合置信度。

        融合置信度 = prior × prior_weight + (1 - η/δ_K) × kappa_weight + rhae_norm × rhae_weight

        Args:
            eta: GaussEx残差η。

        Returns:
            融合置信度值 (0~1范围)。
        """
        kappa_confidence: float = max(0.0, 1.0 - eta / self.delta_k) if self.delta_k > 0 else 0.0
        return max(0.0, min(1.0, self._prior_confidence * self.prior_weight + kappa_confidence * self.kappa_weight))


# ============================================================================
# §4. 自测函数
# ============================================================================

def _self_test() -> bool:
    """κ-优选η升序自测: 验证排序、选择、置信度计算 + ψ-Audit + Bayesian-RHAE。

    Returns:
        True if all tests pass, False otherwise.
    """
    # 测试1: η升序排序
    candidates: List[Dict[str, Any]] = [
        {'node_id': 1, 'eta': 0.8, 'ic': 0.5, 'depth': 3},
        {'node_id': 2, 'eta': 0.05, 'ic': 0.3, 'depth': 2},
        {'node_id': 3, 'eta': 0.2, 'ic': 0.7, 'depth': 4},
        {'node_id': 4, 'eta': 0.03, 'ic': 0.2, 'depth': 1},
    ]

    sorted_cands = kappa_eta_ascend_sort(candidates)
    # η升序: 0.03, 0.05, 0.2 (0.8过滤掉, confidence < 1/6)
    assert len(sorted_cands) > 0, "Should have valid candidates after filtering"
    # η最小应在最前
    if len(sorted_cands) >= 2:
        assert sorted_cands[0]['eta'] <= sorted_cands[1]['eta'], \
            "η升序: η should be ascending"

    # 测试2: Liu-Score计算
    liu_score: float = compute_liu_score({'depth': 2, 'ic': 0.3, 'eta': 0.05})
    assert liu_score > 0, "Liu-Score should be positive"
    assert isinstance(liu_score, float), "Liu-Score should be float"

    # 测试3: κ-优选η升序选择器 (★ 升级5: needs_critique)
    selector: KappaEtaAscendSelector = KappaEtaAscendSelector()
    result: List[Dict[str, Any]] = selector.select(candidates)
    assert isinstance(result, list), "Selector should return list"
    # 结果应按Liu-Score降序
    if len(result) >= 2:
        assert result[0]['liu_score'] >= result[1]['liu_score'], \
            "Results should be Liu-Score descending"
    # ★ needs_critique应为False (有有效候选)
    assert not selector.needs_critique, "Should not need critique with valid candidates"

    # 测试4: 置信度计算
    conf: float = selector.confidence(0.03)
    expected: float = 1.0 - 0.03 / 0.036  # ≈ 0.167
    assert abs(conf - expected) < 0.01, f"confidence ≈ {expected}, got {conf}"

    # 测试5: Liu-Score优先选择器
    liu_selector: LiuPrioritySelector = LiuPrioritySelector()
    liu_result: List[Dict[str, Any]] = liu_selector.select(candidates)
    assert isinstance(liu_result, list), "Liu selector should return list"

    # ★ 测试6: 空候选触发needs_critique
    empty_candidates: List[Dict[str, Any]] = [
        {'node_id': 100, 'eta': 1.0, 'ic': 0.5, 'depth': 3},  # confidence=0 → 过滤掉
    ]
    empty_result: List[Dict[str, Any]] = selector.select(empty_candidates)
    assert len(empty_result) == 0, "All η > δ_K → should return empty"
    assert selector.needs_critique, "★ Should need critique when all η > δ_K"
    assert len(selector.critique_diagnosis) > 0, "★ Should have diagnosis text"

    # ★ 测试7: Bayesian-RHAE融合选择器
    br_selector: BayesianRHAESSelector = BayesianRHAESSelector()
    br_result: List[Dict[str, Any]] = br_selector.select(candidates)
    assert isinstance(br_result, list), "Bayesian-RHAE selector should return list"
    # Bayesian-RHAE融合分数应已计算
    if len(br_result) > 0:
        assert 'bayesian_rhae_score' in br_result[0], "★ Should have bayesian_rhae_score"
        assert br_result[0]['bayesian_rhae_score'] > 0, "★ bayesian_rhae_score should be positive"

    # ★ 测试8: Bayesian-RHAE置信度
    br_conf: float = br_selector.confidence(0.03)
    assert br_conf > 0, "★ Bayesian-RHAE confidence should be positive"

    # ★ 测试9: Bayesian-RHAE compute_bayesian_rhae_score
    test_cand: Dict[str, Any] = {'eta': 0.03, 'rhae_score': 100.0}
    br_score: float = br_selector.compute_bayesian_rhae_score(test_cand)
    assert 0 <= br_score <= 1, f"★ Bayesian-RHAE score should be in [0,1], got {br_score}"

    # ★ 测试10: ψ-Audit日志
    audit_log: PsiAuditLog = get_psi_audit_log()
    assert len(audit_log.entries) > 0, "★ ψ-Audit log should have entries"
    summary: Dict[str, Any] = audit_log.summary()
    assert summary['total_entries'] > 0, "★ Audit summary should have total_entries"
    # 查询critique触发记录
    critique_entries: List[PsiAuditEntry] = audit_log.query(needs_critique=True)
    assert len(critique_entries) > 0, "★ Should have critique entries"

    # ★ 测试11: ψ-Audit JSON导出
    json_str: str = audit_log.export_json()
    assert len(json_str) > 0, "★ Audit JSON export should be non-empty"

    # ★ 测试12: PsiAuditEntry数据结构
    entry: PsiAuditEntry = PsiAuditEntry(
        selector='test', eta=0.05, confidence=0.9, liu_score=0.8,
        bayesian_rhae_score=0.75, timestamp=time.time(),
        node_id=42, needs_critique=False, diagnosis="test",
    )
    entry_dict: Dict[str, Any] = entry.to_dict()
    assert entry_dict['selector'] == 'test', "★ Entry dict should preserve selector"
    assert entry_dict['bayesian_rhae_score'] == 0.75, "★ Entry dict should preserve bayesian_rhae_score"

    print("[PASS] kappa_selector _self_test passed (★ includes ψ-Audit + Bayesian-RHAE)")
    return True


if __name__ == "__main__":
    _self_test()
