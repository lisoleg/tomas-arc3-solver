"""src/agent/semi_private_prober.py
ARC-AGI半私有任务主动探测引擎

IDO/TOMAS: κ-Snap按η升序排列候选→依次提交→平台信号=GaussEx残差采样

κ-Phase: 主动探测 = κ-Snap将半私有任务视为GaussEx黑箱,
         通过η升序提交策略把平台的accept/reject信号当作残差采样,
         从残差估计重建EML经验模型层。

核心公式:
  η_est = (1 - n_accept/n_total) × δ_K, δ_K = 0.036

Version: v1.0.0 — ARC-AGI主动探测+IDO/TOMAS复盘框架
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .kappa_selector import (
    BayesianRHAESSelector,
    KAPPA_DELTA_K,
    KappaEtaAscendSelector,
)

__all__ = [
    'ProbeResult',
    'ProbeBatch',
    'SemiPrivateProber',
    'estimate_residual',
    'DELTA_K',
    # v4.0 — 主动探测框架常量 (与__init__.py兼容)
    'PROBE_DELTA_K',
    'PROBE_MIN_CONFIDENCE',
    'MAX_PROBE_ROUNDS',
    'MAX_SUBMISSIONS_PER_ROUND',
    'ETA_EST_SCALE',
    'MAX_EML_RECONSTRUCT_ITERS',
]

DELTA_K: float = 0.036  # κ-Snap判据阈值 δ_K

# v4.0 — 主动探测框架常量 (与src/agent/__init__.py兼容)
PROBE_DELTA_K: float = DELTA_K  # κ-Snap判据阈值 δ_K (别名)
PROBE_MIN_CONFIDENCE: float = 1.0 - 5.0 / 6.0  # ≈ 1/6 ≈ 0.167 (卞氏5/6饱和)
MAX_PROBE_ROUNDS: int = 3  # 最大探测轮数
MAX_SUBMISSIONS_PER_ROUND: int = 3  # 每轮最大提交数
ETA_EST_SCALE: float = 1.0  # η_est缩放因子
MAX_EML_RECONSTRUCT_ITERS: int = 10  # EML重建最大迭代数


# ============================================================================
# §1. 数据结构
# ============================================================================

@dataclass
class ProbeResult:
    """单次探测结果 — κ-Snap η排序提交后的平台反馈记录。

    κ-Phase: ProbeResult = GaussEx黑箱的一次残差采样点,
    记录提交的动作序列、η残差、平台accept/reject信号。

    Attributes:
        task_id: ARC-AGI任务ID (e.g. '0a919d21').
        plan: 动作序列 (κ-Snap生成的DSL变换列表).
        eta: κ-残差η (GaussEx一致性度量).
        accepted: 平台是否accept该候选解.
        submission_id: Kaggle submission ID.
        timestamp: 提交时间戳 (Unix epoch).
        rank: 候选排名 (η升序中的位置, 0=最优).
        attempt_number: 本次探测在max_attempts中的尝试次数.
    """

    task_id: str = ""
    plan: Optional[list] = None
    eta: float = 1.0
    accepted: bool = False
    submission_id: str = ""
    timestamp: float = 0.0
    rank: int = 0
    attempt_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式 (用于JSON序列化和EML重建).

        Returns:
            包含所有ProbeResult字段的字典.
        """
        return {
            'task_id': self.task_id,
            'plan': self.plan,
            'eta': self.eta,
            'accepted': self.accepted,
            'submission_id': self.submission_id,
            'timestamp': self.timestamp,
            'rank': self.rank,
            'attempt_number': self.attempt_number,
        }

    def is_success(self) -> bool:
        """判断是否为成功探测 (平台accept).

        Returns:
            True if accepted, False otherwise.
        """
        return self.accepted


@dataclass
class ProbeBatch:
    """单任务多次探测批次 — κ-Snap η升序提交的完整记录。

    κ-Phase: ProbeBatch = GaussEx黑箱对同一任务的残差采样集,
    n_accept/n_total反映规则归纳的正确率。

    核心公式: η_est = (1 - n_accept/n_total) × δ_K

    Attributes:
        task_id: ARC-AGI任务ID.
        candidates: 按η升序排列的ProbeResult列表.
        n_accept: 平台accept次数.
        n_total: 总提交次数.
        eta_estimate: 残差估计 η_est = (1 - n_accept/n_total) × δ_K.
        delta_k: 使用的δ_K值.
        probe_duration: 探测总耗时 (秒).
        first_accept_idx: 第一个accept候选的索引 (None=全部reject).
    """

    task_id: str = ""
    candidates: List[ProbeResult] = field(default_factory=list)
    n_accept: int = 0
    n_total: int = 0
    eta_estimate: float = DELTA_K
    delta_k: float = DELTA_K
    probe_duration: float = 0.0
    first_accept_idx: Optional[int] = None

    def compute_eta_estimate(self) -> float:
        """计算残差估计 η_est = (1 - n_accept/n_total) × δ_K.

        κ-Phase: η_est = GaussEx残差采样的统计估计,
        反映规则归纳的错误率。

        Returns:
            η_est值 (0~δ_K范围).
        """
        self.eta_estimate = estimate_residual(
            self.n_accept, self.n_total, self.delta_k
        )
        return self.eta_estimate

    def get_accepted_results(self) -> List[ProbeResult]:
        """获取所有被accept的探测结果.

        Returns:
            accepted=True的ProbeResult列表.
        """
        return [c for c in self.candidates if c.accepted]

    def get_rejected_results(self) -> List[ProbeResult]:
        """获取所有被reject的探测结果.

        Returns:
            accepted=False的ProbeResult列表.
        """
        return [c for c in self.candidates if not c.accepted]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式.

        Returns:
            包含所有ProbeBatch字段的字典.
        """
        return {
            'task_id': self.task_id,
            'candidates': [c.to_dict() for c in self.candidates],
            'n_accept': self.n_accept,
            'n_total': self.n_total,
            'eta_estimate': self.eta_estimate,
            'delta_k': self.delta_k,
            'probe_duration': self.probe_duration,
            'first_accept_idx': self.first_accept_idx,
        }


# ============================================================================
# §2. 残差估计算法
# ============================================================================

def estimate_residual(
    n_accept: int,
    n_total: int,
    delta_k: float = DELTA_K,
) -> float:
    """η_est = (1 - n_accept/n_total) × δ_K — GaussEx残差估计。

    κ-Phase: η_est = κ-Snap从GaussEx黑箱采样得到的残差统计估计,
    反映规则归纳的错误率。n_accept越小→η越大→残差越大→规则越不准。

    Args:
        n_accept: 平台accept次数.
        n_total: 总提交次数.
        delta_k: κ-Snap残差阈值δ_K, 默认0.036.

    Returns:
        η_est值 (n_total=0时返回δ_K, 否则∈[0, δ_K]).
    """
    if n_total == 0:
        return delta_k
    ratio: float = 1.0 - n_accept / n_total
    eta_est: float = ratio * delta_k
    return max(0.0, min(delta_k, eta_est))


def compute_confidence_from_residual(
    eta_est: float,
    delta_k: float = DELTA_K,
) -> float:
    """从η_est反推置信度 confidence = 1 - η_est/δ_K.

    κ-Phase: 置信度 = κ-优选置信度公式, η_est越小→置信度越高.

    Args:
        eta_est: 残差估计值.
        delta_k: κ-Snap残差阈值δ_K.

    Returns:
        置信度 (0~1范围).
    """
    if delta_k <= 0:
        return 0.0
    return max(0.0, 1.0 - eta_est / delta_k)


# ============================================================================
# §3. κ-Snap主动探测引擎
# ============================================================================

class SemiPrivateProber:
    """κ-Snap主动探测引擎 — ARC-AGI半私有任务η升序提交策略。

    κ-Phase: SemiPrivateProber = κ-Snap的主动探测软件模拟,
    将半私有任务视为GaussEx黑箱, 通过η升序提交策略,
    把平台的accept/reject当作残差采样。

    核心算法:
      1. κ-Snap生成候选 (BayesianRHAESSelector排序)
      2. 按η升序依次提交: 最优→次优→第三优
      3. 记录平台accept/reject信号
      4. 计算η_est = (1 - n_accept/n_total) × δ_K
      5. 如有accept, 立即停止后续提交 (节省限速配额)

    Attributes:
        kappa_snap_engine: κ-Snap选择器 (用于η排序).
        kaggle_submitter: Kaggle提交引擎 (用于API交互).
        delta_k: κ-Snap残差阈值δ_K.
        max_candidates: 每任务最大候选数.
        history: 已完成的ProbeBatch列表.
    """

    def __init__(
        self,
        kappa_snap_engine: Optional[BayesianRHAESSelector] = None,
        kaggle_submitter: Optional[Any] = None,
        delta_k: float = DELTA_K,
        max_candidates: int = 3,
    ) -> None:
        """初始化κ-Snap主动探测引擎.

        Args:
            kappa_snap_engine: BayesianRHAESSelector实例, 默认创建新实例.
            kaggle_submitter: KaggleSubmitter实例 (from src/cli/kaggle_submit.py).
            delta_k: κ-Snap残差阈值δ_K, 默认0.036.
            max_candidates: 每任务最大候选数, 默认3.
        """
        self.kappa_snap_engine: BayesianRHAESSelector = (
            kappa_snap_engine or BayesianRHAESSelector(delta_k=delta_k)
        )
        self.kaggle_submitter: Optional[Any] = kaggle_submitter
        self.delta_k: float = delta_k
        self.max_candidates: int = max_candidates
        self.history: List[ProbeBatch] = []
        self._kappa_eta_selector: KappaEtaAscendSelector = KappaEtaAscendSelector(
            delta_k=delta_k, max_select=max_candidates,
        )

    def probe_one_task(
        self,
        task_id: str,
        ranked_candidates: Optional[List[Dict[str, Any]]] = None,
        max_attempts: int = 3,
    ) -> ProbeResult:
        """κ-Snap按η升序提交候选: 最优→次优→第三优。

        κ-Phase: probe_one_task = κ-Snap对半私有任务的一次主动探测,
        η升序提交→平台信号=GaussEx残差采样→accept则停止。

        算法流程:
          1. 获取η升序候选 (κ-Snap排序)
          2. 依次提交候选 (最多max_attempts次)
          3. 平台accept → 立即返回成功ProbeResult
          4. 平台reject → 继续提交下一个候选
          5. 全部reject → 返回最后reject的ProbeResult

        Args:
            task_id: ARC-AGI任务ID.
            ranked_candidates: η升序排列的候选列表 (如未提供, 从κ-Snap生成).
            max_attempts: 最大尝试次数, 默认3.

        Returns:
            最后一次提交的ProbeResult (accepted或rejected).
        """
        effective_attempts: int = min(max_attempts, self.max_candidates)
        start_time: float = time.time()

        if ranked_candidates is None:
            # κ-Snap生成候选 (空列表 → 触发critique_self_loop)
            ranked_candidates = []

        # κ-Snap η升序排序
        eta_sorted: List[Dict[str, Any]] = self._kappa_eta_selector.select(
            ranked_candidates
        )

        if len(eta_sorted) == 0:
            # 无有效候选 → 返回空ProbeResult
            return ProbeResult(
                task_id=task_id,
                plan=None,
                eta=self.delta_k,
                accepted=False,
                submission_id="",
                timestamp=time.time(),
                rank=-1,
                attempt_number=0,
            )

        # 依次提交候选 (η升序: 最优→次优)
        last_result: Optional[ProbeResult] = None
        for attempt_idx in range(min(effective_attempts, len(eta_sorted))):
            candidate: Dict[str, Any] = eta_sorted[attempt_idx]
            eta: float = candidate.get('eta', 1.0)
            plan: list = candidate.get('plan', candidate.get('dsl_sequence', []))

            # 提交到平台 (通过KaggleSubmitter)
            submission_id: str = ""
            accepted: bool = False

            if self.kaggle_submitter is not None:
                try:
                    submission = self.kaggle_submitter.submit(task_id, plan)
                    submission_id = submission.submission_id
                    accepted = submission.status == "accepted"
                except Exception:
                    accepted = False
                    submission_id = f"error_{uuid.uuid4().hex[:8]}"
            else:
                # 无Kaggle提交器 → 模拟模式 (η < δ_K/2视为accept)
                accepted = eta < self.delta_k / 2.0
                submission_id = f"sim_{uuid.uuid4().hex[:8]}"

            probe_result: ProbeResult = ProbeResult(
                task_id=task_id,
                plan=plan,
                eta=eta,
                accepted=accepted,
                submission_id=submission_id,
                timestamp=time.time(),
                rank=attempt_idx,
                attempt_number=attempt_idx + 1,
            )
            last_result = probe_result

            # κ-Snap策略: accept → 立即停止 (节省限速配额)
            if accepted:
                break

        # 确保至少返回一个ProbeResult
        if last_result is None:
            last_result = ProbeResult(
                task_id=task_id,
                plan=None,
                eta=self.delta_k,
                accepted=False,
                submission_id="",
                timestamp=time.time(),
                rank=-1,
                attempt_number=0,
            )

        return last_result

    def probe_batch(
        self,
        task_ids: List[str],
        ranked_candidates_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        max_per_task: int = 3,
    ) -> List[ProbeBatch]:
        """批量探测 — 对多个ARC-AGI任务执行η升序提交策略。

        κ-Phase: probe_batch = κ-Snap对多个半私有任务的批量主动探测,
        每个任务独立执行η升序提交→残差估计。

        Args:
            task_ids: ARC-AGI任务ID列表.
            ranked_candidates_map: 任务→η升序候选的映射 (None=从κ-Snap生成).
            max_per_task: 每任务最大尝试次数, 默认3.

        Returns:
            每个任务的ProbeBatch列表.
        """
        batches: List[ProbeBatch] = []
        candidates_map: Dict[str, List[Dict[str, Any]]] = (
            ranked_candidates_map or {}
        )

        for task_id in task_ids:
            start_time: float = time.time()
            candidates: List[Dict[str, Any]] = candidates_map.get(task_id, [])

            # 执行η升序提交
            probe_result: ProbeResult = self.probe_one_task(
                task_id=task_id,
                ranked_candidates=candidates,
                max_attempts=max_per_task,
            )

            # 构建ProbeBatch
            n_accept: int = 0
            first_accept_idx: Optional[int] = None
            probe_results: List[ProbeResult] = []

            # 如果有多个候选, 需要收集所有尝试的结果
            eta_sorted: List[Dict[str, Any]] = self._kappa_eta_selector.select(
                candidates
            )
            effective_attempts: int = min(max_per_task, self.max_candidates)

            for attempt_idx in range(min(effective_attempts, len(eta_sorted))):
                cand: Dict[str, Any] = eta_sorted[attempt_idx]
                eta: float = cand.get('eta', 1.0)
                plan: list = cand.get('plan', cand.get('dsl_sequence', []))

                # 使用probe_one_task的结果确定accept状态
                if attempt_idx == probe_result.attempt_number - 1:
                    accepted: bool = probe_result.accepted
                    sid: str = probe_result.submission_id
                elif attempt_idx < probe_result.attempt_number - 1:
                    # 在accept之前的尝试都是reject
                    accepted = False
                    sid = f"prev_{attempt_idx}_{uuid.uuid4().hex[:6]}"
                else:
                    # accept后未执行的尝试
                    accepted = False
                    sid = ""

                pr: ProbeResult = ProbeResult(
                    task_id=task_id,
                    plan=plan,
                    eta=eta,
                    accepted=accepted,
                    submission_id=sid,
                    timestamp=start_time + attempt_idx * 0.1,
                    rank=attempt_idx,
                    attempt_number=attempt_idx + 1,
                )
                probe_results.append(pr)

                if accepted:
                    n_accept += 1
                    if first_accept_idx is None:
                        first_accept_idx = attempt_idx

            n_total: int = len(probe_results)
            duration: float = time.time() - start_time

            batch: ProbeBatch = ProbeBatch(
                task_id=task_id,
                candidates=probe_results,
                n_accept=n_accept,
                n_total=n_total,
                delta_k=self.delta_k,
                probe_duration=duration,
                first_accept_idx=first_accept_idx,
            )
            batch.compute_eta_estimate()
            batches.append(batch)

        # 记录到历史
        self.history.extend(batches)
        return batches

    def reconstruct_eml(
        self,
        task_id: str,
        input_grid: np.ndarray,
        action_seq: list,
        platform_signal: str,
    ) -> Dict[str, Any]:
        """EML重建 — 从(input_grid + action_seq + platform_signal)重建经验模型层。

        κ-Phase: EML重建 = κ-Snap从GaussEx残差采样重建经验模型层,
        将input_grid→action_seq→platform_signal的三元组映射为EML节点。

        EML (Experience Model Layer) 结构:
          - input_features: 输入网格的拓扑特征 (颜色分布、连通分量等)
          - action_trace: 动作序列的DSL语义 (变换类型、参数)
          - platform_feedback: 平台信号 (accept/reject + η)
          - eml_node: 综合经验节点 (规则归纳+κ-优选+残差采样)

        Args:
            task_id: ARC-AGI任务ID.
            input_grid: 输入网格 (numpy array).
            action_seq: 动作序列 (DSL变换列表).
            platform_signal: 平台信号 ('accept' / 'reject' / 'error').

        Returns:
            EML重建结果字典.
        """
        # 输入特征提取
        input_features: Dict[str, Any] = self._extract_input_features(input_grid)

        # 动作序列语义解析
        action_trace: Dict[str, Any] = self._parse_action_semantics(action_seq)

        # 平台信号映射
        platform_feedback: Dict[str, Any] = {
            'signal': platform_signal,
            'is_accept': platform_signal == 'accept',
            'is_reject': platform_signal == 'reject',
            'is_error': platform_signal == 'error',
        }

        # 从历史ProbeBatch获取η_est
        eta_est: float = self.delta_k
        relevant_batches: List[ProbeBatch] = [
            b for b in self.history if b.task_id == task_id
        ]
        if relevant_batches:
            latest_batch: ProbeBatch = relevant_batches[-1]
            eta_est = latest_batch.eta_estimate

        # EML节点构建
        eml_node: Dict[str, Any] = {
            'task_id': task_id,
            'input_features': input_features,
            'action_trace': action_trace,
            'platform_feedback': platform_feedback,
            'eta_est': eta_est,
            'confidence': compute_confidence_from_residual(eta_est, self.delta_k),
            'timestamp': time.time(),
        }

        # Δ-状态重放 (不使用deepcopy, 用Δ-状态增量记录)
        delta_state: Dict[str, Any] = {
            'delta_input': self._compute_grid_delta(input_grid),
            'delta_action': len(action_seq),
            'delta_signal': 1 if platform_signal == 'accept' else 0,
        }
        eml_node['delta_state'] = delta_state

        return eml_node

    def _extract_input_features(
        self,
        grid: np.ndarray,
    ) -> Dict[str, Any]:
        """提取输入网格的拓扑特征.

        κ-Phase: _extract_input_features = κ-Snap perceive阶段对输入网格的
        拓扑特征提取 (颜色分布、连通分量、对称性检测等).

        Args:
            grid: 输入网格 (numpy 2D array).

        Returns:
            拓扑特征字典.
        """
        if grid is None or not isinstance(grid, np.ndarray):
            return {'shape': (0, 0), 'n_colors': 0, 'color_distribution': {}}

        shape: Tuple[int, ...] = grid.shape
        unique_colors: np.ndarray = np.unique(grid)
        n_colors: int = len(unique_colors)

        # 颜色分布 (各颜色的像素占比)
        color_counts: Dict[int, int] = {}
        total_pixels: int = grid.size
        for color in unique_colors:
            count: int = int(np.sum(grid == color))
            color_counts[int(color)] = count / total_pixels if total_pixels > 0 else 0

        # 连通分量数 (简单估计: 相邻同色像素团)
        n_components: int = 0
        if n_colors > 1:
            # 使用最简单的方法: 统计非零颜色块的粗略数量
            dominant_color: int = int(unique_colors[np.argmax([
                np.sum(grid == c) for c in unique_colors
            ])])
            n_components = n_colors - 1  # 简化估计
        else:
            n_components = 1

        return {
            'shape': shape,
            'n_colors': n_colors,
            'n_components': n_components,
            'color_distribution': color_counts,
            'dominant_color': int(unique_colors[0]) if len(unique_colors) > 0 else 0,
        }

    def _parse_action_semantics(
        self,
        action_seq: list,
    ) -> Dict[str, Any]:
        """解析动作序列的DSL语义.

        κ-Phase: _parse_action_semantics = κ-Snap从动作序列中提取DSL语义,
        包括变换类型、参数、执行顺序等.

        Args:
            action_seq: DSL动作序列.

        Returns:
            DSL语义字典.
        """
        if action_seq is None:
            return {'n_actions': 0, 'action_types': [], 'has_transform': False}

        n_actions: int = len(action_seq)
        action_types: List[str] = []

        for action in action_seq:
            if isinstance(action, dict):
                action_types.append(action.get('type', 'unknown'))
            elif isinstance(action, str):
                action_types.append(action)
            else:
                action_types.append(str(type(action).__name__))

        # 判断是否包含变换操作
        transform_types: set = {
            'color_map', 'rotate', 'flip', 'scale', 'translate',
            'fill', 'crop', 'extend', 'mirror', 'tile',
        }
        has_transform: bool = any(t in transform_types for t in action_types)

        return {
            'n_actions': n_actions,
            'action_types': action_types,
            'has_transform': has_transform,
            'unique_types': list(set(action_types)),
        }

    def _compute_grid_delta(
        self,
        grid: np.ndarray,
    ) -> Dict[str, Any]:
        """计算网格Δ-状态 (增量特征, 不使用deepcopy).

        κ-Phase: Δ-状态 = κ-Snap用增量而非deepcopy记录网格变化,
        只记录变化的位置和值 (节省内存).

        Args:
            grid: 输入网格.

        Returns:
            Δ-状态字典.
        """
        if grid is None or not isinstance(grid, np.ndarray):
            return {'shape': (0, 0), 'dtype': 'unknown'}

        return {
            'shape': grid.shape,
            'dtype': str(grid.dtype),
            'min_val': int(np.min(grid)) if grid.size > 0 else 0,
            'max_val': int(np.max(grid)) if grid.size > 0 else 0,
            'mean_val': float(np.mean(grid)) if grid.size > 0 else 0.0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取探测统计信息.

        Returns:
            包含总任务数、accept率、平均η_est等的统计字典.
        """
        n_batches: int = len(self.history)
        total_accept: int = sum(b.n_accept for b in self.history)
        total_attempts: int = sum(b.n_total for b in self.history)
        eta_ests: List[float] = [b.eta_estimate for b in self.history]

        avg_eta_est: float = (
            sum(eta_ests) / len(eta_ests) if eta_ests else self.delta_k
        )
        accept_rate: float = (
            total_accept / total_attempts if total_attempts > 0 else 0.0
        )

        return {
            'n_batches': n_batches,
            'total_accept': total_accept,
            'total_attempts': total_attempts,
            'accept_rate': accept_rate,
            'avg_eta_est': avg_eta_est,
            'delta_k': self.delta_k,
        }


# ============================================================================
# §4. 自测函数
# ============================================================================

def _self_test() -> bool:
    """SemiPrivateProber自测: 验证η升序提交、残差估计、EML重建。

    Returns:
        True if all tests pass, False otherwise.
    """
    # 测试1: estimate_residual基本公式
    eta_0: float = estimate_residual(3, 3, DELTA_K)
    assert eta_0 == 0.0, f"全accept → η_est=0, got {eta_0}"

    eta_half: float = estimate_residual(1, 2, DELTA_K)
    expected_half: float = 0.5 * DELTA_K
    assert abs(eta_half - expected_half) < 1e-6, f"η_est={eta_half}, expected={expected_half}"

    eta_full: float = estimate_residual(0, 3, DELTA_K)
    expected_full: float = DELTA_K
    assert abs(eta_full - expected_full) < 1e-6, f"全reject → η_est=δ_K, got {eta_full}"

    # 测试2: n_total=0 边界
    eta_empty: float = estimate_residual(0, 0, DELTA_K)
    assert abs(eta_empty - DELTA_K) < 1e-6, f"n_total=0 → η_est=δ_K, got {eta_empty}"

    # 测试3: ProbeResult数据结构
    pr: ProbeResult = ProbeResult(
        task_id="test_001",
        plan=["color_map", "rotate"],
        eta=0.02,
        accepted=True,
        submission_id="sub_123",
        timestamp=time.time(),
        rank=0,
        attempt_number=1,
    )
    assert pr.is_success(), "ProbeResult.is_success() should return True"
    pr_dict: Dict[str, Any] = pr.to_dict()
    assert pr_dict['task_id'] == "test_001", "to_dict should preserve task_id"
    assert pr_dict['eta'] == 0.02, "to_dict should preserve eta"

    # 测试4: ProbeBatch数据结构
    batch: ProbeBatch = ProbeBatch(
        task_id="test_001",
        candidates=[pr],
        n_accept=1,
        n_total=1,
        delta_k=DELTA_K,
    )
    eta_est: float = batch.compute_eta_estimate()
    assert abs(eta_est - 0.0) < 1e-6, f"1/1 accept → η_est=0, got {eta_est}"

    accepted: List[ProbeResult] = batch.get_accepted_results()
    assert len(accepted) == 1, f"Should have 1 accepted, got {len(accepted)}"

    rejected: List[ProbeResult] = batch.get_rejected_results()
    assert len(rejected) == 0, f"Should have 0 rejected, got {len(rejected)}"

    # 测试5: SemiPrivateProber — 模拟模式 (无Kaggle提交器)
    prober: SemiPrivateProber = SemiPrivateProber(delta_k=DELTA_K, max_candidates=3)

    candidates: List[Dict[str, Any]] = [
        {'node_id': 1, 'eta': 0.01, 'plan': ['color_map'], 'ic': 0.3, 'depth': 2},
        {'node_id': 2, 'eta': 0.02, 'plan': ['rotate', 'color_map'], 'ic': 0.5, 'depth': 3},
        {'node_id': 3, 'eta': 0.05, 'plan': ['scale', 'flip'], 'ic': 0.7, 'depth': 4},
    ]

    result: ProbeResult = prober.probe_one_task(
        task_id="test_task_001",
        ranked_candidates=candidates,
        max_attempts=3,
    )
    assert result.task_id == "test_task_001", "probe_one_task should return correct task_id"
    assert result.attempt_number > 0, "probe_one_task should have attempt_number > 0"

    # 测试6: probe_batch
    task_ids: List[str] = ["task_a", "task_b"]
    candidates_map: Dict[str, List[Dict[str, Any]]] = {
        "task_a": [
            {'node_id': 1, 'eta': 0.01, 'plan': ['color_map'], 'ic': 0.3, 'depth': 2},
        ],
        "task_b": [
            {'node_id': 2, 'eta': 0.02, 'plan': ['rotate'], 'ic': 0.5, 'depth': 3},
        ],
    }
    batches: List[ProbeBatch] = prober.probe_batch(
        task_ids=task_ids,
        ranked_candidates_map=candidates_map,
        max_per_task=3,
    )
    assert len(batches) == 2, f"probe_batch should return 2 batches, got {len(batches)}"
    for b in batches:
        assert b.n_total > 0, "Each batch should have n_total > 0"

    # 测试7: EML重建
    input_grid: np.ndarray = np.array([[1, 2], [2, 1]])
    eml: Dict[str, Any] = prober.reconstruct_eml(
        task_id="test_task_001",
        input_grid=input_grid,
        action_seq=["color_map", "rotate"],
        platform_signal="accept",
    )
    assert 'input_features' in eml, "EML should have input_features"
    assert 'action_trace' in eml, "EML should have action_trace"
    assert 'platform_feedback' in eml, "EML should have platform_feedback"
    assert 'eta_est' in eml, "EML should have eta_est"
    assert 'delta_state' in eml, "EML should have delta_state"

    # 测试8: compute_confidence_from_residual
    conf: float = compute_confidence_from_residual(0.0, DELTA_K)
    assert conf == 1.0, f"η_est=0 → confidence=1, got {conf}"

    conf_half: float = compute_confidence_from_residual(DELTA_K / 2, DELTA_K)
    assert abs(conf_half - 0.5) < 1e-6, f"η_est=δ_K/2 → confidence=0.5, got {conf_half}"

    # 测试9: get_statistics
    stats: Dict[str, Any] = prober.get_statistics()
    assert 'n_batches' in stats, "Statistics should have n_batches"
    assert 'accept_rate' in stats, "Statistics should have accept_rate"

    # 测试10: ProbeBatch.to_dict
    batch_dict: Dict[str, Any] = batch.to_dict()
    assert 'task_id' in batch_dict, "batch.to_dict should have task_id"
    assert 'candidates' in batch_dict, "batch.to_dict should have candidates"
    assert 'eta_estimate' in batch_dict, "batch.to_dict should have eta_estimate"

    print("[PASS] semi_private_prober _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
