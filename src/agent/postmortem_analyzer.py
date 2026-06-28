"""src/agent/postmortem_analyzer.py
IDO/TOMAS复盘引擎 — 三问复盘算法+贝叶斯prior修正

复盘三问:
  Q1: 规则归纳是否正确? (κ-Snap perceive + GaussEx校验)
  Q2: κ-Snap排序是否合理? (η单调递增检查)
  Q3: 部分accept定位? (哪个候选被accept, 哪些被reject)

κ-Phase: 复盘 = κ-Snap对主动探测结果的系统性事后分析,
从三问中提取诊断信息, 调整贝叶斯prior以优化后续探测。

核心公式:
  prior_adjustment = η_est × sign(q_failed) × prior_weight
  confidence = 1 - η_est/δ_K

Version: v1.0.0 — ARC-AGI主动探测+IDO/TOMAS复盘框架
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .semi_private_prober import (
    ProbeResult,
    ProbeBatch,
    estimate_residual,
    DELTA_K,
    compute_confidence_from_residual,
)
from .kappa_selector import (
    BayesianRHAESSelector,
    KappaEtaAscendSelector,
    KAPPA_DELTA_K,
    KAPPA_MIN_CONFIDENCE,
)

__all__ = [
    'PostmortemDiagnosis',
    'PostmortemAnalyzer',
    'diagnose_three_questions',
    # v4.0 — 复盘框架常量和类 (与__init__.py兼容)
    'RuleInductionCheck',
    'RankingCheck',
    'PartialAcceptLocator',
    'QUESTION_RULE_INDUCTION',
    'QUESTION_RANKING',
    'QUESTION_PARTIAL_ACCEPT',
    'PRIOR_UPDATE_RATE',
    'RULE_CONFIDENCE_THRESHOLD',
    'RANKING_CONSISTENCY_THRESHOLD',
    'PARTIAL_ACCEPT_SEARCH_DEPTH',
    'MAX_POSTMORTEM_ROUNDS',
]

# v4.0 — 复盘框架常量 (与src/agent/__init__.py兼容)
QUESTION_RULE_INDUCTION: str = "Q1: 规则归纳是否正确?"
QUESTION_RANKING: str = "Q2: κ-Snap排序是否合理?"
QUESTION_PARTIAL_ACCEPT: str = "Q3: 部分accept定位?"
PRIOR_UPDATE_RATE: float = 0.1  # prior修正率
RULE_CONFIDENCE_THRESHOLD: float = DELTA_K / 3.0  # Q1判定阈值
RANKING_CONSISTENCY_THRESHOLD: float = 0.1  # Q2排序容差 (10%逆序允许)
PARTIAL_ACCEPT_SEARCH_DEPTH: int = 3  # Q3搜索深度
MAX_POSTMORTEM_ROUNDS: int = 3  # 最大复盘轮数


# ============================================================================
# §1. 数据结构
# ============================================================================

@dataclass
class PostmortemDiagnosis:
    """三问复盘诊断结果 — IDO/TOMAS对主动探测的事后分析。

    κ-Phase: PostmortemDiagnosis = κ-Snap三问复盘的诊断输出,
    包含Q1/Q2/Q3的回答和prior修正建议。

    Attributes:
        task_id: ARC-AGI任务ID.
        q1_rule_correct: Q1回答 — 规则归纳是否正确.
        q2_ranking_reasonable: Q2回答 — κ-Snap排序是否合理.
        q3_partial_accept: Q3回答 — 部分accept定位 (None=全部reject).
        suggested_prior_adjustment: prior修正建议值.
        confidence: 置信度 = 1 - η_est/δ_K.
        eta_estimate: 残差估计 η_est.
        q1_detail: Q1详细诊断文本.
        q2_detail: Q2详细诊断文本.
        q3_detail: Q3详细诊断文本.
        diagnosis_timestamp: 诊断时间戳.
        severity: 严重等级 ('low'/'medium'/'high'/'critical').
    """

    task_id: str = ""
    q1_rule_correct: bool = False
    q2_ranking_reasonable: bool = False
    q3_partial_accept: Optional[str] = None
    suggested_prior_adjustment: float = 0.0
    confidence: float = 0.0
    eta_estimate: float = DELTA_K
    q1_detail: str = ""
    q2_detail: str = ""
    q3_detail: str = ""
    diagnosis_timestamp: float = 0.0
    severity: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式.

        Returns:
            包含所有PostmortemDiagnosis字段的字典.
        """
        return {
            'task_id': self.task_id,
            'q1_rule_correct': self.q1_rule_correct,
            'q2_ranking_reasonable': self.q2_ranking_reasonable,
            'q3_partial_accept': self.q3_partial_accept,
            'suggested_prior_adjustment': self.suggested_prior_adjustment,
            'confidence': self.confidence,
            'eta_estimate': self.eta_estimate,
            'q1_detail': self.q1_detail,
            'q2_detail': self.q2_detail,
            'q3_detail': self.q3_detail,
            'diagnosis_timestamp': self.diagnosis_timestamp,
            'severity': self.severity,
        }

    def all_correct(self) -> bool:
        """判断三问是否全部通过.

        Returns:
            True if Q1=True AND Q2=True AND Q3=accepted, False otherwise.
        """
        return (
            self.q1_rule_correct
            and self.q2_ranking_reasonable
            and self.q3_partial_accept is not None
        )

    def get_severity_level(self) -> int:
        """获取严重等级数值 (用于排序和过滤).

        Returns:
            0=low, 1=medium, 2=high, 3=critical.
        """
        severity_map: Dict[str, int] = {
            'low': 0, 'medium': 1, 'high': 2, 'critical': 3,
        }
        return severity_map.get(self.severity, 0)


@dataclass
class PostmortemBatchResult:
    """批量复盘结果 — 多任务的PostmortemDiagnosis集合.

    Attributes:
        diagnoses: 各任务的PostmortemDiagnosis列表.
        n_tasks: 任务总数.
        n_q1_pass: Q1通过的任务数.
        n_q2_pass: Q2通过的任务数.
        n_q3_pass: Q3有accept的任务数.
        avg_prior_adjustment: 平均prior修正值.
        avg_confidence: 平均置信度.
        avg_eta_estimate: 平均η_est.
    """

    diagnoses: List[PostmortemDiagnosis] = field(default_factory=list)
    n_tasks: int = 0
    n_q1_pass: int = 0
    n_q2_pass: int = 0
    n_q3_pass: int = 0
    avg_prior_adjustment: float = 0.0
    avg_confidence: float = 0.0
    avg_eta_estimate: float = 0.0

    def compute_summary(self) -> None:
        """计算汇总统计."""
        self.n_tasks = len(self.diagnoses)
        self.n_q1_pass = sum(1 for d in self.diagnoses if d.q1_rule_correct)
        self.n_q2_pass = sum(1 for d in self.diagnoses if d.q2_ranking_reasonable)
        self.n_q3_pass = sum(1 for d in self.diagnoses if d.q3_partial_accept is not None)

        if self.n_tasks > 0:
            self.avg_prior_adjustment = sum(
                d.suggested_prior_adjustment for d in self.diagnoses
            ) / self.n_tasks
            self.avg_confidence = sum(
                d.confidence for d in self.diagnoses
            ) / self.n_tasks
            self.avg_eta_estimate = sum(
                d.eta_estimate for d in self.diagnoses
            ) / self.n_tasks


# ============================================================================
# §2. 三问复盘快捷函数
# ============================================================================

def diagnose_three_questions(
    probe_batch: ProbeBatch,
    delta_k: float = DELTA_K,
) -> PostmortemDiagnosis:
    """三问复盘快捷函数 — 对单个ProbeBatch执行完整三问分析。

    Args:
        probe_batch: 单任务的探测批次.
        delta_k: κ-Snap残差阈值δ_K.

    Returns:
        PostmortemDiagnosis诊断结果.
    """
    analyzer: PostmortemAnalyzer = PostmortemAnalyzer(delta_k=delta_k)
    return analyzer.analyze_one(None, probe_batch)


# ============================================================================
# §3. 复盘分析引擎
# ============================================================================

class PostmortemAnalyzer:
    """IDO/TOMAS复盘引擎 — 三问复盘算法+贝叶斯prior修正。

    κ-Phase: PostmortemAnalyzer = κ-Snap的复盘软件模拟,
    对主动探测结果执行三问分析, 调整贝叶斯prior以优化后续探测。

    三问算法:
      Q1: 规则归纳是否正确? → η_est < δ_K/3 → True
      Q2: κ-Snap排序是否合理? → η单调递增 → True
      Q3: 部分accept定位? → first_accept_idx → 定位信息

    Prior修正:
      - 全accept → prior↑ (规则归纳好, 排序好)
      - 部分accept → prior↓ (规则归纳或排序有问题)
      - 全reject → prior↓↓ (规则归纳或排序严重偏差)

    Attributes:
        delta_k: κ-Snap残差阈值δ_K.
        prior_weight: prior修正权重 (控制修正幅度).
        q1_threshold: Q1判定阈值 (η_est < threshold → True).
        bayesian_selector: BayesianRHAESSelector实例 (用于prior调整).
        history: 已完成的PostmortemDiagnosis列表.
    """

    def __init__(
        self,
        delta_k: float = DELTA_K,
        prior_weight: float = 0.1,
        q1_threshold: Optional[float] = None,
    ) -> None:
        """初始化复盘分析引擎.

        Args:
            delta_k: κ-Snap残差阈值δ_K, 默认0.036.
            prior_weight: prior修正权重, 默认0.1.
            q1_threshold: Q1判定阈值 (None=自动设为δ_K/3).
        """
        self.delta_k: float = delta_k
        self.prior_weight: float = prior_weight
        self.q1_threshold: float = (
            q1_threshold if q1_threshold is not None else delta_k / 3.0
        )
        self.bayesian_selector: BayesianRHAESSelector = BayesianRHAESSelector(
            delta_k=delta_k,
        )
        self.history: List[PostmortemDiagnosis] = []

    def analyze_one(
        self,
        probe_result: Optional[ProbeResult],
        probe_batch: ProbeBatch,
    ) -> PostmortemDiagnosis:
        """三问复盘 — 对单个探测批次执行完整三问分析。

        κ-Phase: analyze_one = κ-Snap对一次主动探测结果的系统性复盘,
        Q1→Q2→Q3→prior修正→severity评估。

        Args:
            probe_result: 最后一次探测结果 (可选, 用于辅助诊断).
            probe_batch: 单任务的探测批次 (包含完整候选序列).

        Returns:
            PostmortemDiagnosis诊断结果.
        """
        task_id: str = probe_batch.task_id
        eta_est: float = probe_batch.eta_estimate

        # Q1: 规则归纳是否正确?
        q1_correct: bool = self.q1_rule_induction_check_from_batch(probe_batch)
        q1_detail: str = self._generate_q1_detail(probe_batch, q1_correct)

        # Q2: κ-Snap排序是否合理?
        q2_reasonable: bool = self.q2_ranking_check(probe_batch.candidates)
        q2_detail: str = self._generate_q2_detail(probe_batch, q2_reasonable)

        # Q3: 部分accept定位?
        q3_partial: Optional[str] = self.q3_partial_accept_locator(probe_batch)
        q3_detail: str = self._generate_q3_detail(probe_batch, q3_partial)

        # prior修正建议
        prior_adj: float = self.suggest_prior_adjustment_from_batch(probe_batch)

        # 置信度
        confidence: float = compute_confidence_from_residual(eta_est, self.delta_k)

        # severity评估
        severity: str = self._compute_severity(q1_correct, q2_reasonable, q3_partial)

        diagnosis: PostmortemDiagnosis = PostmortemDiagnosis(
            task_id=task_id,
            q1_rule_correct=q1_correct,
            q2_ranking_reasonable=q2_reasonable,
            q3_partial_accept=q3_partial,
            suggested_prior_adjustment=prior_adj,
            confidence=confidence,
            eta_estimate=eta_est,
            q1_detail=q1_detail,
            q2_detail=q2_detail,
            q3_detail=q3_detail,
            diagnosis_timestamp=time.time(),
            severity=severity,
        )

        self.history.append(diagnosis)
        return diagnosis

    def analyze_batch(
        self,
        probe_batches: List[ProbeBatch],
    ) -> PostmortemBatchResult:
        """批量复盘 — 对多个探测批次执行三问分析。

        Args:
            probe_batches: 多任务的探测批次列表.

        Returns:
            PostmortemBatchResult批量诊断结果.
        """
        diagnoses: List[PostmortemDiagnosis] = []
        for batch in probe_batches:
            diag: PostmortemDiagnosis = self.analyze_one(None, batch)
            diagnoses.append(diag)

        batch_result: PostmortemBatchResult = PostmortemBatchResult(
            diagnoses=diagnoses,
        )
        batch_result.compute_summary()
        return batch_result

    def q1_rule_induction_check(
        self,
        task_id: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
    ) -> bool:
        """Q1: 规则归纳是否正确 — κ-Snap perceive + GaussEx校验。

        κ-Phase: Q1 = κ-Snap对规则归纳的校验,
        通过比较input→output的变换一致性来判断归纳是否正确。

        简化判定:
          - output与input结构一致 → True
          - output与input差异过大 → False

        Args:
            task_id: ARC-AGI任务ID.
            input_grid: 输入网格.
            output_grid: 输出网格.

        Returns:
            True if 规则归纳正确, False otherwise.
        """
        if input_grid is None or output_grid is None:
            return False

        # 基础一致性检查: 输出网格形状应与输入一致或合理变换
        input_shape: Tuple[int, ...] = input_grid.shape
        output_shape: Tuple[int, ...] = output_grid.shape

        # 形状一致性
        shape_consistent: bool = (
            input_shape == output_shape
            or (output_shape[0] == input_shape[0] * 2 and output_shape[1] == input_shape[1] * 2)
            or (output_shape[0] == input_shape[0] and output_shape[1] == input_shape[1] * 2)
            or (output_shape[0] == input_shape[0] * 2 and output_shape[1] == input_shape[1])
        )

        # 颜色变化合理性
        input_colors: np.ndarray = np.unique(input_grid)
        output_colors: np.ndarray = np.unique(output_grid)
        color_change_reasonable: bool = len(output_colors) <= len(input_colors) * 3

        # GaussEx一致性 (简化版: 输出颜色变化在合理范围)
        gex_pass: bool = shape_consistent and color_change_reasonable

        return gex_pass

    def q1_rule_induction_check_from_batch(
        self,
        probe_batch: ProbeBatch,
    ) -> bool:
        """Q1: 从ProbeBatch判断规则归纳是否正确。

        κ-Phase: Q1简化判定 — η_est < q1_threshold → 规则归纳正确。
        η_est反映规则归纳的错误率, η_est越小 → 规则越准。

        Args:
            probe_batch: 单任务的探测批次.

        Returns:
            True if η_est < q1_threshold, False otherwise.
        """
        eta_est: float = probe_batch.eta_estimate
        return eta_est < self.q1_threshold

    def q2_ranking_check(
        self,
        candidates: List[ProbeResult],
    ) -> bool:
        """Q2: κ-Snap排序是否合理 — η单调递增检查。

        κ-Phase: Q2 = κ-Snap排序合理性校验,
        η升序排列应严格单调递增 (或允许小波动)。
        如果η出现递减, 说明κ-Snap排序有偏差。

        判定标准:
          - η严格单调递增 → True (完美排序)
          - η允许10%容差的单调递增 → True (合理排序)
          - η出现明显逆序 → False (排序偏差)

        Args:
            candidates: 按η升序排列的ProbeResult列表.

        Returns:
            True if η排序合理, False otherwise.
        """
        if len(candidates) <= 1:
            return True

        etas: List[float] = [c.eta for c in candidates]

        # 检查η严格单调递增
        strictly_monotone: bool = all(
            etas[i] < etas[i + 1] for i in range(len(etas) - 1)
        )

        if strictly_monotone:
            return True

        # 允许10%容差: 逆序波动不超过10%
        n_reversals: int = 0
        for i in range(len(etas) - 1):
            if etas[i] >= etas[i + 1]:
                n_reversals += 1

        reversal_rate: float = n_reversals / (len(etas) - 1)
        return reversal_rate <= 0.1

    def q3_partial_accept_locator(
        self,
        probe_batch: ProbeBatch,
    ) -> Optional[str]:
        """Q3: 部分accept定位 — 哪个候选被accept, 哪些被reject。

        κ-Phase: Q3 = κ-Snap对accept/reject信号的定位分析,
        从部分accept中提取"哪个候选被接受"的信息。

        Args:
            probe_batch: 单任务的探测批次.

        Returns:
            accept定位描述 (None=全部reject).
            - "rank_0_eta_0.01": 第1候选(η=0.01)被accept
            - "rank_1_eta_0.02": 第2候选(η=0.02)被accept
            - None: 全部reject
        """
        accepted: List[ProbeResult] = probe_batch.get_accepted_results()

        if len(accepted) == 0:
            return None

        # 定位第一个accept
        first_accept: ProbeResult = accepted[0]
        return f"rank_{first_accept.rank}_eta_{first_accept.eta:.4f}"

    def suggest_prior_adjustment(
        self,
        diagnosis: PostmortemDiagnosis,
    ) -> float:
        """贝叶斯prior修正建议 — 根据三问结果调整prior。

        κ-Phase: prior修正 = κ-Snap的贝叶斯prior调整,
        根据三问结果调整后续探测的置信度prior。

        修正公式:
          - 全accept → prior↑0.05 (规则归纳好, 排序好)
          - 部分accept → prior↓η_est×weight (规则归纳或排序有问题)
          - 全reject → prior↓δ_K×weight (规则归纳或排序严重偏差)

        Args:
            diagnosis: 三问复盘诊断结果.

        Returns:
            prior修正建议值 (正=提升prior, 负=降低prior).
        """
        eta_est: float = diagnosis.eta_estimate

        if diagnosis.all_correct():
            # 全正确 → 提升prior
            adjustment: float = 0.05
        elif diagnosis.q3_partial_accept is not None:
            # 部分accept → 降低prior (η_est×weight)
            adjustment: float = -eta_est * self.prior_weight
        else:
            # 全reject → 大幅降低prior (δ_K×weight)
            adjustment: float = -self.delta_k * self.prior_weight

        return adjustment

    def suggest_prior_adjustment_from_batch(
        self,
        probe_batch: ProbeBatch,
    ) -> float:
        """从ProbeBatch直接计算prior修正建议 (不依赖diagnosis).

        κ-Phase: prior修正 = η_est × sign(accept_rate) × weight

        Args:
            probe_batch: 单任务的探测批次.

        Returns:
            prior修正建议值.
        """
        eta_est: float = probe_batch.eta_estimate
        accept_rate: float = (
            probe_batch.n_accept / probe_batch.n_total
            if probe_batch.n_total > 0
            else 0.0
        )

        # accept_rate高 → prior↑, accept_rate低 → prior↓
        if accept_rate >= 0.8:
            return 0.05  # 高accept率 → 轻微提升prior
        elif accept_rate >= 0.5:
            return -eta_est * self.prior_weight  # 部分accept → 降低prior
        elif accept_rate > 0.0:
            return -eta_est * self.prior_weight * 2.0  # 低accept → 较大降低
        else:
            return -self.delta_k * self.prior_weight * 3.0  # 全reject → 大幅降低

    def apply_prior_to_selector(
        self,
        diagnosis: PostmortemDiagnosis,
        selector: Optional[BayesianRHAESSelector] = None,
    ) -> float:
        """将prior修正应用到BayesianRHAESSelector。

        κ-Phase: prior应用 = κ-Snap将复盘结果反馈到贝叶斯选择器,
        调整prior_confidence以优化后续探测的η排序。

        Args:
            diagnosis: 三问复盘诊断结果.
            selector: BayesianRHAESSelector实例 (None=使用self.bayesian_selector).

        Returns:
            新的prior_confidence值.
        """
        target: BayesianRHAESSelector = selector or self.bayesian_selector
        current_prior: float = target._prior_confidence

        adjustment: float = self.suggest_prior_adjustment(diagnosis)
        new_prior: float = max(0.0, min(1.0, current_prior + adjustment))

        target.set_prior_confidence(new_prior)
        return new_prior

    # ========================================================================
    # 内部辅助函数
    # ========================================================================

    def _generate_q1_detail(
        self,
        probe_batch: ProbeBatch,
        q1_correct: bool,
    ) -> str:
        """生成Q1详细诊断文本.

        Args:
            probe_batch: 探测批次.
            q1_correct: Q1判定结果.

        Returns:
            Q1详细诊断字符串.
        """
        eta_est: float = probe_batch.eta_estimate
        n_accept: int = probe_batch.n_accept
        n_total: int = probe_batch.n_total
        accept_rate: float = n_accept / n_total if n_total > 0 else 0.0

        status: str = "PASS" if q1_correct else "FAIL"
        return (
            f"Q1[{status}] η_est={eta_est:.6f}, "
            f"accept_rate={accept_rate:.2f} ({n_accept}/{n_total}), "
            f"threshold={self.q1_threshold:.6f}"
        )

    def _generate_q2_detail(
        self,
        probe_batch: ProbeBatch,
        q2_reasonable: bool,
    ) -> str:
        """生成Q2详细诊断文本.

        Args:
            probe_batch: 探测批次.
            q2_reasonable: Q2判定结果.

        Returns:
            Q2详细诊断字符串.
        """
        candidates: List[ProbeResult] = probe_batch.candidates
        etas: List[float] = [c.eta for c in candidates]

        status: str = "PASS" if q2_reasonable else "FAIL"
        eta_sequence: str = ", ".join(f"{e:.4f}" for e in etas[:5])

        n_reversals: int = 0
        for i in range(len(etas) - 1):
            if etas[i] >= etas[i + 1]:
                n_reversals += 1

        return (
            f"Q2[{status}] η_sequence=[{eta_sequence}], "
            f"n_reversals={n_reversals}/{len(etas)-1 if len(etas)>1 else 0}"
        )

    def _generate_q3_detail(
        self,
        probe_batch: ProbeBatch,
        q3_partial: Optional[str],
    ) -> str:
        """生成Q3详细诊断文本.

        Args:
            probe_batch: 探测批次.
            q3_partial: Q3判定结果.

        Returns:
            Q3详细诊断字符串.
        """
        if q3_partial is None:
            return f"Q3[FAIL] 全部reject, 无accept候选 (n_total={probe_batch.n_total})"

        idx: Optional[int] = probe_batch.first_accept_idx
        accepted: List[ProbeResult] = probe_batch.get_accepted_results()

        return (
            f"Q3[PASS] 首次accept@idx={idx}, "
            f"定位={q3_partial}, "
            f"n_accept={len(accepted)}/{probe_batch.n_total}"
        )

    def _compute_severity(
        self,
        q1: bool,
        q2: bool,
        q3: Optional[str],
    ) -> str:
        """计算严重等级.

        Args:
            q1: Q1判定结果.
            q2: Q2判定结果.
            q3: Q3判定结果 (None=全reject).

        Returns:
            严重等级字符串.
        """
        n_fail: int = sum([
            not q1,
            not q2,
            q3 is None,
        ])

        if n_fail == 0:
            return "low"
        elif n_fail == 1:
            return "medium"
        elif n_fail == 2:
            return "high"
        else:
            return "critical"

    def get_statistics(self) -> Dict[str, Any]:
        """获取复盘统计信息.

        Returns:
            包含Q1/Q2/Q3通过率、平均prior修正等的统计字典.
        """
        n_diagnoses: int = len(self.history)
        if n_diagnoses == 0:
            return {'n_diagnoses': 0}

        n_q1_pass: int = sum(1 for d in self.history if d.q1_rule_correct)
        n_q2_pass: int = sum(1 for d in self.history if d.q2_ranking_reasonable)
        n_q3_pass: int = sum(1 for d in self.history if d.q3_partial_accept is not None)

        avg_prior_adj: float = sum(
            d.suggested_prior_adjustment for d in self.history
        ) / n_diagnoses
        avg_confidence: float = sum(
            d.confidence for d in self.history
        ) / n_diagnoses

        severity_counts: Dict[str, int] = {}
        for d in self.history:
            severity_counts[d.severity] = severity_counts.get(d.severity, 0) + 1

        return {
            'n_diagnoses': n_diagnoses,
            'q1_pass_rate': n_q1_pass / n_diagnoses,
            'q2_pass_rate': n_q2_pass / n_diagnoses,
            'q3_pass_rate': n_q3_pass / n_diagnoses,
            'avg_prior_adjustment': avg_prior_adj,
            'avg_confidence': avg_confidence,
            'severity_counts': severity_counts,
        }


# ============================================================================
# §4a. 复盘三问函数别名 (与src/agent/__init__.py兼容)
# ============================================================================

class RuleInductionCheck:
    """Q1规则归纳检查的便捷封装类 (与__init__.py兼容).

    κ-Phase: RuleInductionCheck = Q1三问复盘的类封装,
    简化规则归纳校验的调用方式.

    Attributes:
        analyzer: PostmortemAnalyzer实例.
    """

    def __init__(self, delta_k: float = DELTA_K) -> None:
        """初始化Q1规则归纳检查器.

        Args:
            delta_k: κ-Snap残差阈值δ_K.
        """
        self.analyzer: PostmortemAnalyzer = PostmortemAnalyzer(delta_k=delta_k)

    def check(
        self,
        task_id: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
    ) -> bool:
        """执行Q1规则归纳校验.

        Args:
            task_id: ARC-AGI任务ID.
            input_grid: 输入网格.
            output_grid: 输出网格.

        Returns:
            True if 规则归纳正确.
        """
        return self.analyzer.q1_rule_induction_check(task_id, input_grid, output_grid)

    def check_from_batch(
        self,
        probe_batch: ProbeBatch,
    ) -> bool:
        """从ProbeBatch执行Q1规则归纳校验.

        Args:
            probe_batch: 探测批次.

        Returns:
            True if η_est < threshold.
        """
        return self.analyzer.q1_rule_induction_check_from_batch(probe_batch)


class RankingCheck:
    """Q2排序合理性检查的便捷封装类 (与__init__.py兼容).

    κ-Phase: RankingCheck = Q2三问复盘的类封装,
    简化η单调递增校验的调用方式.

    Attributes:
        analyzer: PostmortemAnalyzer实例.
    """

    def __init__(self, delta_k: float = DELTA_K) -> None:
        """初始化Q2排序合理性检查器.

        Args:
            delta_k: κ-Snap残差阈值δ_K.
        """
        self.analyzer: PostmortemAnalyzer = PostmortemAnalyzer(delta_k=delta_k)

    def check(
        self,
        candidates: List[ProbeResult],
    ) -> bool:
        """执行Q2排序合理性校验.

        Args:
            candidates: 按η升序排列的ProbeResult列表.

        Returns:
            True if η排序合理.
        """
        return self.analyzer.q2_ranking_check(candidates)


class PartialAcceptLocator:
    """Q3部分accept定位的便捷封装类 (与__init__.py兼容).

    κ-Phase: PartialAcceptLocator = Q3三问复盘的类封装,
    简化部分accept定位的调用方式.

    Attributes:
        analyzer: PostmortemAnalyzer实例.
    """

    def __init__(self, delta_k: float = DELTA_K) -> None:
        """初始化Q3部分accept定位器.

        Args:
            delta_k: κ-Snap残差阈值δ_K.
        """
        self.analyzer: PostmortemAnalyzer = PostmortemAnalyzer(delta_k=delta_k)

    def locate(
        self,
        probe_batch: ProbeBatch,
    ) -> Optional[str]:
        """执行Q3部分accept定位.

        Args:
            probe_batch: 探测批次.

        Returns:
            accept定位描述 (None=全部reject).
        """
        return self.analyzer.q3_partial_accept_locator(probe_batch)


# ============================================================================
# §4. 自测函数
# ============================================================================

def _self_test() -> bool:
    """PostmortemAnalyzer自测: 验证三问复盘、prior修正、severity评估。

    Returns:
        True if all tests pass, False otherwise.
    """
    # 测试1: PostmortemDiagnosis数据结构
    diag: PostmortemDiagnosis = PostmortemDiagnosis(
        task_id="test_001",
        q1_rule_correct=True,
        q2_ranking_reasonable=True,
        q3_partial_accept="rank_0_eta_0.0100",
        suggested_prior_adjustment=0.05,
        confidence=0.72,
        eta_estimate=0.01,
    )
    assert diag.all_correct(), "All three questions should pass"
    assert diag.get_severity_level() == 0, f"Severity 'low' → level 0, got {diag.get_severity_level()}"

    diag_dict: Dict[str, Any] = diag.to_dict()
    assert diag_dict['task_id'] == "test_001", "to_dict should preserve task_id"
    assert diag_dict['q1_rule_correct'] is True, "to_dict should preserve q1"

    # 测试2: PostmortemAnalyzer — Q1: η_est < threshold → True
    analyzer: PostmortemAnalyzer = PostmortemAnalyzer(delta_k=DELTA_K)

    # 构造ProbeBatch: 全accept → η_est=0 → Q1=True
    batch_pass: ProbeBatch = ProbeBatch(
        task_id="test_pass",
        candidates=[
            ProbeResult(task_id="test_pass", eta=0.01, accepted=True, rank=0),
        ],
        n_accept=1,
        n_total=1,
        delta_k=DELTA_K,
    )
    batch_pass.compute_eta_estimate()

    diag_pass: PostmortemDiagnosis = analyzer.analyze_one(None, batch_pass)
    assert diag_pass.q1_rule_correct, "全accept → Q1 should pass"
    assert diag_pass.q3_partial_accept is not None, "全accept → Q3 should have accept"

    # 测试3: Q1: η_est > threshold → False (全reject)
    batch_fail: ProbeBatch = ProbeBatch(
        task_id="test_fail",
        candidates=[
            ProbeResult(task_id="test_fail", eta=0.8, accepted=False, rank=0),
            ProbeResult(task_id="test_fail", eta=0.9, accepted=False, rank=1),
            ProbeResult(task_id="test_fail", eta=0.95, accepted=False, rank=2),
        ],
        n_accept=0,
        n_total=3,
        delta_k=DELTA_K,
    )
    batch_fail.compute_eta_estimate()

    diag_fail: PostmortemDiagnosis = analyzer.analyze_one(None, batch_fail)
    assert not diag_fail.q1_rule_correct, "全reject → Q1 should fail"
    assert diag_fail.q3_partial_accept is None, "全reject → Q3 should be None"
    assert diag_fail.severity in ('high', 'critical'), f"全reject → severity=high/critical, got {diag_fail.severity}"

    # 测试4: Q2: η单调递增 → True
    candidates_mono: List[ProbeResult] = [
        ProbeResult(eta=0.01, rank=0),
        ProbeResult(eta=0.03, rank=1),
        ProbeResult(eta=0.05, rank=2),
    ]
    assert analyzer.q2_ranking_check(candidates_mono), "η单调递增 → Q2 should pass"

    # 测试5: Q2: η逆序 → False
    candidates_reverse: List[ProbeResult] = [
        ProbeResult(eta=0.05, rank=0),
        ProbeResult(eta=0.01, rank=1),
        ProbeResult(eta=0.03, rank=2),
    ]
    assert not analyzer.q2_ranking_check(candidates_reverse), "η逆序 → Q2 should fail"

    # 测试6: Q3: 部分accept定位
    batch_partial: ProbeBatch = ProbeBatch(
        task_id="test_partial",
        candidates=[
            ProbeResult(task_id="test_partial", eta=0.01, accepted=False, rank=0),
            ProbeResult(task_id="test_partial", eta=0.03, accepted=True, rank=1),
            ProbeResult(task_id="test_partial", eta=0.05, accepted=False, rank=2),
        ],
        n_accept=1,
        n_total=3,
        delta_k=DELTA_K,
        first_accept_idx=1,
    )
    batch_partial.compute_eta_estimate()

    q3_result: Optional[str] = analyzer.q3_partial_accept_locator(batch_partial)
    assert q3_result is not None, "部分accept → Q3 should have location"
    assert "rank_1" in q3_result, f"Q3 should locate rank_1, got {q3_result}"

    # 测试7: prior修正 — 全accept → prior↑
    prior_adj_pass: float = analyzer.suggest_prior_adjustment(diag_pass)
    assert prior_adj_pass > 0, f"全accept → prior↑, got {prior_adj_pass}"

    # 测试8: prior修正 — 全reject → prior↓↓
    prior_adj_fail: float = analyzer.suggest_prior_adjustment(diag_fail)
    assert prior_adj_fail < 0, f"全reject → prior↓↓, got {prior_adj_fail}"

    # 测试9: prior应用到BayesianRHAESSelector
    new_prior: float = analyzer.apply_prior_to_selector(diag_pass)
    assert 0 <= new_prior <= 1, f"new_prior should be in [0,1], got {new_prior}"

    # 测试10: diagnose_three_questions快捷函数
    quick_diag: PostmortemDiagnosis = diagnose_three_questions(batch_pass)
    assert quick_diag.task_id == "test_pass", "Quick diagnose should have correct task_id"

    # 测试11: analyze_batch批量复盘
    batch_result: PostmortemBatchResult = analyzer.analyze_batch(
        [batch_pass, batch_fail, batch_partial]
    )
    assert batch_result.n_tasks == 3, f"Batch should have 3 tasks, got {batch_result.n_tasks}"
    assert batch_result.n_q1_pass >= 1, "At least 1 task should pass Q1"

    # 测试12: get_statistics
    stats: Dict[str, Any] = analyzer.get_statistics()
    assert 'n_diagnoses' in stats, "Statistics should have n_diagnoses"
    assert 'q1_pass_rate' in stats, "Statistics should have q1_pass_rate"

    # 测试13: PostmortemDiagnosis.all_correct + severity计算
    diag_all_fail: PostmortemDiagnosis = PostmortemDiagnosis(
        task_id="test_all_fail",
        q1_rule_correct=False,
        q2_ranking_reasonable=False,
        q3_partial_accept=None,
        severity="critical",
    )
    assert not diag_all_fail.all_correct(), "All fail → all_correct=False"
    assert diag_all_fail.get_severity_level() == 3, f"critical → level 3, got {diag_all_fail.get_severity_level()}"

    # 测试14: _compute_severity方法
    assert analyzer._compute_severity(False, False, None) == "critical", \
        "3 failures → severity=critical"
    assert analyzer._compute_severity(True, True, "accepted") == "low", \
        "0 failures → severity=low"

    # 测试14: Q1 grid-based check
    input_grid: np.ndarray = np.array([[1, 2], [2, 1]])
    output_grid: np.ndarray = np.array([[2, 1], [1, 2]])
    q1_grid: bool = analyzer.q1_rule_induction_check("grid_test", input_grid, output_grid)
    assert isinstance(q1_grid, bool), "Q1 grid check should return bool"

    print("[PASS] postmortem_analyzer _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
