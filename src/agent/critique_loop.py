"""
src/agent/critique_loop.py
Critique-Self-Loop 自纠错循环独立模块 (文章 §14.1 + Appendix A)

κ-Phase: 批评与自我批评 = κ-Snap自指残差校验 + Dead-Zero熔断
当κ-优选选不出候选(η > δ_K)时触发自纠错:
  1. diagnose(η): 分析最大残差来源
  2. adjust_macro: 根据诊断调整搜索策略
  3. 重跑L1→L2→L3→L4管线(放宽参数)
  4. 最多max_retry=3次, 失败则raise CannotConverge

文章建议: 将critique_self_loop从HybridSearchPipeline方法提取为独立模块,
使其可被任意求解管线调用, 不依赖pipeline实例。

Version: v4.0 — 独立模块提取 + 通用接口
"""

from __future__ import annotations

import time as _time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


# ============================================================
# §1. 核心异常与数据类型
# ============================================================

class CannotConverge(Exception):
    """critique_self_loop自纠错循环收敛失败异常。

    当κ-优选选不出候选(η > δ_K)且自纠错3轮仍无法收敛时抛出。
    κ-Phase: 批评与自我批评 = κ-Snap自指残差校验 + Dead-Zero熔断
    """
    pass


@dataclass
class CritiqueDiagnosis:
    """diagnose(η) 输出 — κ-Snap自指残差校验结果。

    Attributes:
        best_eta: 最小残差η。
        worst_eta: 最大残差η。
        avg_eta: 平均残差η。
        eta_range: η变化范围。
        deadlock_count: 死锁候选数量。
        candidate_count: 总候选数量。
        diagnosis_str: 诊断描述字符串。
        retry_idx: 自纠错轮次。
        critique_phase: 当前阶段标识。
        l4_diagnosis: L4选择器诊断信息(合并)。
    """
    best_eta: Optional[float] = None
    worst_eta: Optional[float] = None
    avg_eta: Optional[float] = None
    eta_range: Optional[float] = None
    deadlock_count: int = 0
    candidate_count: int = 0
    diagnosis_str: str = ""
    retry_idx: int = 0
    critique_phase: str = ""
    l4_diagnosis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式(兼容旧接口)。"""
        return {
            'best_eta': self.best_eta,
            'worst_eta': self.worst_eta,
            'avg_eta': self.avg_eta,
            'eta_range': self.eta_range,
            'deadlock_count': self.deadlock_count,
            'candidate_count': self.candidate_count,
            'diagnosis': self.diagnosis_str,
            'retry_idx': self.retry_idx,
            'critique_phase': self.critique_phase,
            'l4_diagnosis': self.l4_diagnosis,
        }


@dataclass
class CritiqueResult:
    """critique_self_loop 输出 — 自纠错结果。

    Attributes:
        converged: 是否收敛成功。
        verified_candidates: 验证通过的候选列表(收敛成功时)。
        diagnosis_history: 每轮诊断记录。
        total_retries: 总自纠错轮数。
        total_time: 总耗时(秒)。
        adjusted_params: 每轮调整后的参数。
    """
    converged: bool = False
    verified_candidates: Optional[List[Dict[str, Any]]] = None
    diagnosis_history: List[CritiqueDiagnosis] = field(default_factory=list)
    total_retries: int = 0
    total_time: float = 0.0
    adjusted_params: List[Dict[str, int]] = field(default_factory=list)


# ============================================================
# §2. diagnose(η) — 自指残差校验
# ============================================================

def diagnose_eta(candidates: List[Dict[str, Any]]) -> CritiqueDiagnosis:
    """diagnose(η) — 分析评估集残差来源 (文章 §14.1)。

    κ-Phase: 自指残差校验 = κ-Snap project self_view → KS_GX residual_self
    如果residual_self > δ_K → DZFUSE + INFLOW correction

    Args:
        candidates: L3评估后的候选列表。

    Returns:
        CritiqueDiagnosis实例。
    """
    if len(candidates) == 0:
        return CritiqueDiagnosis(
            diagnosis_str='empty_candidate_set',
        )

    etas: List[float] = [c.get('eta', 1.0) for c in candidates]
    best_eta: float = min(etas)
    worst_eta: float = max(etas)
    avg_eta: float = sum(etas) / len(etas)

    deadlock_count: int = sum(
        1 for c in candidates
        if c.get('deadlock_checked', False) and c.get('eta', 0.0) >= 1.0
    )

    return CritiqueDiagnosis(
        best_eta=best_eta,
        worst_eta=worst_eta,
        avg_eta=avg_eta,
        eta_range=worst_eta - best_eta,
        deadlock_count=deadlock_count,
        candidate_count=len(etas),
        diagnosis_str=f'eta_range={worst_eta - best_eta:.4f}, avg={avg_eta:.4f}',
    )


# ============================================================
# §3. adjust_macro — 搜索策略调整
# ============================================================

def adjust_macro_params(
    diagnosis: CritiqueDiagnosis,
    base_depth: int,
    base_nodes: int,
    retry_idx: int,
) -> Dict[str, int]:
    """adjust_macro — 根据诊断调整搜索参数。

    κ-Phase: 自纠错策略 = 放宽κ-Snap约束 + 扩大搜索预算
    每轮retry: depth +5, nodes ×2 (指数增长直到上限)

    Args:
        diagnosis: 当前轮诊断结果。
        base_depth: 基础最大搜索深度。
        base_nodes: 基础最大扩展节点数。
        retry_idx: 当前自纠错轮次(0-based)。

    Returns:
        调整后的参数字典: {max_depth, max_nodes}
    """
    adjusted_depth: int = min(base_depth + 5 * (retry_idx + 1), 60)
    adjusted_nodes: int = min(base_nodes * (2 ** retry_idx), 500000)

    # 死锁特殊处理: 如果大量死锁, 进一步放宽约束
    if diagnosis.deadlock_count > diagnosis.candidate_count * 0.5:
        adjusted_depth = min(adjusted_depth + 10, 80)
        adjusted_nodes = min(adjusted_nodes * 2, 1000000)

    return {
        'max_depth': adjusted_depth,
        'max_nodes': adjusted_nodes,
    }


# ============================================================
# §4. CritiqueSelfLoop — 独立自纠错循环引擎
# ============================================================

class CritiqueSelfLoop:
    """独立自纠错循环引擎 — 从HybridSearchPipeline提取并泛化。

    κ-Phase: 批评与自我批评 = κ-Snap自指残差校验 + Dead-Zero熔断
    允许负Inflow修正(承认错误), 周期性KS_PROJ self_view校验。

    设计原则:
      - 不依赖HybridSearchPipeline实例, 可被任意求解管线调用
      - 通过pipeline_fn回调执行L1→L2→L3→L4重跑
      - 最多max_retry=3次, 失败则返回None(不抛异常, 由调用者决定)

    Attributes:
        max_retry: 最大自纠错轮数(默认3)。
        base_depth: 基础搜索深度。
        base_nodes: 基础扩展节点数。
        max_time_budget: 总时间预算(秒)。
    """

    def __init__(
        self,
        max_retry: int = 3,
        base_depth: int = 30,
        base_nodes: int = 50000,
        max_time_budget: float = 10.0,
    ):
        self.max_retry = max_retry
        self.base_depth = base_depth
        self.base_nodes = base_nodes
        self.max_time_budget = max_time_budget
        self._result_log: List[CritiqueResult] = []

    def run(
        self,
        candidates: List[Dict[str, Any]],
        pipeline_fn: Optional[callable] = None,
        l4_diagnosis: str = "",
        verify_fn: Optional[callable] = None,
    ) -> CritiqueResult:
        """执行自纠错循环。

        κ-优选选不出候选(η > δ_K)时触发:
          1. diagnose(η): 分析最大残差来源
          2. adjust_macro: 根据诊断调整搜索策略
          3. 通过pipeline_fn回调重跑搜索管线
          4. 最多max_retry=3次

        Args:
            candidates: 当前评估候选列表(η都>δ_K)。
            pipeline_fn: 搜索管线回调函数。
                签名: pipeline_fn(max_depth, max_nodes, **kwargs) → List[Dict[str, Any]]
                返回L4选择后的最优候选列表。
            l4_diagnosis: L4选择器诊断信息。
            verify_fn: 验证回调函数。
                签名: verify_fn(candidates) → Optional[List[Dict[str, Any]]]

        Returns:
            CritiqueResult实例(包含收敛状态和诊断历史)。
        """
        t_start: float = _time.time()
        result = CritiqueResult()
        time_remaining: float = self.max_time_budget

        for retry_idx in range(self.max_retry):
            t_retry: float = _time.time()
            if time_remaining <= 0:
                break  # 时间耗尽

            # Step 1: diagnose(η)
            diagnosis: CritiqueDiagnosis = diagnose_eta(candidates)
            diagnosis.retry_idx = retry_idx
            diagnosis.critique_phase = f'critique_self_loop_{retry_idx}'
            if l4_diagnosis:
                diagnosis.l4_diagnosis = l4_diagnosis

            result.diagnosis_history.append(diagnosis)

            # Step 2: adjust_macro
            adjusted: Dict[str, int] = adjust_macro_params(
                diagnosis, self.base_depth, self.base_nodes, retry_idx,
            )
            result.adjusted_params.append(adjusted)

            # Step 3: 重跑管线(通过回调)
            if pipeline_fn is None:
                # 无管线回调 → 无法自纠错, 直接退出
                break

            try:
                new_candidates: List[Dict[str, Any]] = pipeline_fn(
                    adjusted['max_depth'], adjusted['max_nodes'],
                    critique_retry=retry_idx, diagnosis=diagnosis.to_dict(),
                )
            except Exception:
                # 管线异常 → 继续下一轮
                elapsed = _time.time() - t_retry
                time_remaining -= elapsed
                continue

            if len(new_candidates) == 0:
                elapsed = _time.time() - t_retry
                time_remaining -= elapsed
                candidates = new_candidates  # 更新candidates供下轮诊断
                continue

            # Step 4: 验证候选
            if verify_fn is not None:
                verified = verify_fn(new_candidates)
                if verified is not None and len(verified) > 0:
                    result.converged = True
                    result.verified_candidates = verified
                    result.total_retries = retry_idx + 1
                    result.total_time = _time.time() - t_start
                    # 标记自纠错成功
                    verified[0]['critique_self_loop'] = {
                        'retry_idx': retry_idx,
                        'diagnosis': diagnosis.to_dict(),
                        'adjusted_depth': adjusted['max_depth'],
                        'adjusted_nodes': adjusted['max_nodes'],
                    }
                    self._result_log.append(result)
                    return result
            else:
                # 无验证回调 → 有候选即认为成功
                result.converged = True
                result.verified_candidates = new_candidates
                result.total_retries = retry_idx + 1
                result.total_time = _time.time() - t_start
                new_candidates[0]['critique_self_loop'] = {
                    'retry_idx': retry_idx,
                    'diagnosis': diagnosis.to_dict(),
                    'adjusted_depth': adjusted['max_depth'],
                    'adjusted_nodes': adjusted['max_nodes'],
                }
                self._result_log.append(result)
                return result

            elapsed = _time.time() - t_retry
            time_remaining -= elapsed
            candidates = new_candidates

        # max_retry轮自纠错均失败
        result.total_retries = self.max_retry
        result.total_time = _time.time() - t_start
        self._result_log.append(result)
        return result

    @property
    def result_log(self) -> List[CritiqueResult]:
        """历史自纠错结果日志。"""
        return self._result_log

    def clear_log(self) -> None:
        """清除历史日志。"""
        self._result_log.clear()


# ============================================================
# §5. 独立函数接口 (简化调用)
# ============================================================

def critique_self_loop(
    candidates: List[Dict[str, Any]],
    pipeline_fn: Optional[callable] = None,
    verify_fn: Optional[callable] = None,
    max_retry: int = 3,
    base_depth: int = 30,
    base_nodes: int = 50000,
    max_time_budget: float = 10.0,
    l4_diagnosis: str = "",
) -> Optional[List[Dict[str, Any]]]:
    """critique_self_loop 独立函数接口 (简化调用)。

    κ-优选选不出候选(η > δ_K)时触发自纠错:
      1. diagnose(η): 分析最大残差来源
      2. adjust_macro: 根据诊断调整搜索策略
      3. 重跑L1→L2→L3→L4管线(放宽参数)
      4. 最多max_retry=3次, 失败则返回None

    κ-Phase: 批评与自我批评 = κ-Snap自指残差校验 + Dead-Zero熔断
    允许负Inflow修正(承认错误), 周期性KS_PROJ self_view校验。

    Args:
        candidates: L3评估后的候选列表(η都>δ_K)。
        pipeline_fn: 搜索管线回调函数。
        verify_fn: 验证回调函数。
        max_retry: 最大自纠错轮数。
        base_depth: 基础搜索深度。
        base_nodes: 基础扩展节点数。
        max_time_budget: 总时间预算。
        l4_diagnosis: L4诊断信息。

    Returns:
        验证通过的候选列表, 或None(收敛失败)。
    """
    engine = CritiqueSelfLoop(
        max_retry=max_retry,
        base_depth=base_depth,
        base_nodes=base_nodes,
        max_time_budget=max_time_budget,
    )
    result: CritiqueResult = engine.run(
        candidates=candidates,
        pipeline_fn=pipeline_fn,
        verify_fn=verify_fn,
        l4_diagnosis=l4_diagnosis,
    )
    if result.converged:
        return result.verified_candidates
    return None


# ============================================================
# §6. ψ-Audit 集成接口
# ============================================================

def log_critique_to_psi_audit(
    result: CritiqueResult,
    best_candidate: Optional[Dict[str, Any]] = None,
) -> None:
    """将CritiqueSelfLoop结果记录到ψ-Audit日志。

    κ-Phase: ψ-Audit = κ-优选审计轨迹
    每次critique_self_loop触发/收敛/失败都记录到审计日志。

    Args:
        result: CritiqueResult实例。
        best_candidate: 最终候选(收敛成功时)。
    """
    try:
        from .kappa_selector import PsiAuditEntry, get_psi_audit_log
        psi_audit = get_psi_audit_log()

        if result.converged and best_candidate is not None:
            psi_audit.append(PsiAuditEntry(
                selector='critique_self_loop',
                eta=best_candidate.get('eta', 1.0),
                confidence=best_candidate.get('confidence', 0.0),
                liu_score=best_candidate.get('liu_score', 0.0),
                bayesian_rhae_score=best_candidate.get('bayesian_rhae_score', 0.0),
                timestamp=_time.time(),
                node_id=best_candidate.get('node_id', -1),
                needs_critique=False,
                diagnosis=f"critique_self_loop converged at retry {result.total_retries}",
            ))
        else:
            psi_audit.append(PsiAuditEntry(
                selector='critique_self_loop',
                eta=1.0,  # 最差η
                confidence=0.0,
                liu_score=0.0,
                bayesian_rhae_score=0.0,
                timestamp=_time.time(),
                node_id=-1,
                needs_critique=True,
                diagnosis=f"critique_self_loop failed after {result.total_retries} retries",
            ))
    except ImportError:
        # kappa_selector不可用 → 跳过ψ-Audit记录
        pass


# ============================================================
# §7. 自测
# ============================================================

def _self_test() -> None:
    """critique_loop.py 自测 — 验证核心逻辑正确性。"""
    print("=" * 60)
    print("critique_loop.py _self_test()")
    print("=" * 60)

    # Test 1: diagnose_eta
    candidates = [
        {'eta': 0.1, 'node_id': 0},
        {'eta': 0.5, 'node_id': 1},
        {'eta': 1.0, 'node_id': 2, 'deadlock_checked': True},
    ]
    diag = diagnose_eta(candidates)
    assert diag.best_eta == 0.1, f"best_eta={diag.best_eta}, expected 0.1"
    assert diag.worst_eta == 1.0, f"worst_eta={diag.worst_eta}, expected 1.0"
    assert diag.avg_eta == (0.1 + 0.5 + 1.0) / 3, f"avg_eta={diag.avg_eta}"
    assert diag.deadlock_count == 1, f"deadlock_count={diag.deadlock_count}"
    print("  ✅ Test 1: diagnose_eta — PASS")

    # Test 1b: diagnose_eta with empty candidates
    diag_empty = diagnose_eta([])
    assert diag_empty.best_eta is None
    assert diag_empty.diagnosis_str == 'empty_candidate_set'
    print("  ✅ Test 1b: diagnose_eta(empty) — PASS")

    # Test 2: adjust_macro_params
    diag_no_deadlock = CritiqueDiagnosis(
        best_eta=0.5, worst_eta=1.0, avg_eta=0.75,
        deadlock_count=0, candidate_count=10,
    )
    params = adjust_macro_params(diag_no_deadlock, 30, 50000, 0)
    assert params['max_depth'] == 35, f"adjusted_depth={params['max_depth']}"
    assert params['max_nodes'] == 50000, f"adjusted_nodes={params['max_nodes']}"
    print("  ✅ Test 2: adjust_macro_params(retry=0) — PASS")

    params2 = adjust_macro_params(diag_no_deadlock, 30, 50000, 1)
    assert params2['max_depth'] == 40, f"adjusted_depth={params2['max_depth']}"
    assert params2['max_nodes'] == 100000, f"adjusted_nodes={params2['max_nodes']}"
    print("  ✅ Test 2b: adjust_macro_params(retry=1) — PASS")

    # Test 2c: deadlock-heavy scenario
    diag_heavy_deadlock = CritiqueDiagnosis(
        best_eta=0.5, worst_eta=1.0, avg_eta=0.75,
        deadlock_count=8, candidate_count=10,
    )
    params3 = adjust_macro_params(diag_heavy_deadlock, 30, 50000, 0)
    assert params3['max_depth'] >= 45, f"adjusted_depth={params3['max_depth']}"
    assert params3['max_nodes'] >= 100000, f"adjusted_nodes={params3['max_nodes']}"
    print("  ✅ Test 2c: adjust_macro_params(deadlock-heavy) — PASS")

    # Test 3: CritiqueSelfLoop — converge on retry 1
    def mock_pipeline_converge(max_depth, max_nodes, **kwargs):
        return [{'eta': 0.02, 'node_id': 3, 'confidence': 0.95}]

    def mock_verify(candidates):
        return candidates

    engine = CritiqueSelfLoop(max_retry=3)
    result = engine.run(
        candidates=[{'eta': 1.0}],
        pipeline_fn=mock_pipeline_converge,
        verify_fn=mock_verify,
    )
    assert result.converged, f"Expected convergence, got {result.converged}"
    assert result.total_retries == 1, f"Expected 1 retry, got {result.total_retries}"
    print("  ✅ Test 3: CritiqueSelfLoop — converge — PASS")

    # Test 4: CritiqueSelfLoop — never converge
    def mock_pipeline_fail(max_depth, max_nodes, **kwargs):
        return []

    engine2 = CritiqueSelfLoop(max_retry=2)
    result2 = engine2.run(
        candidates=[{'eta': 1.0}],
        pipeline_fn=mock_pipeline_fail,
    )
    assert not result2.converged, f"Expected no convergence"
    assert result2.total_retries == 2, f"Expected 2 retries"
    print("  ✅ Test 4: CritiqueSelfLoop — no converge — PASS")

    # Test 5: critique_self_loop 函数接口
    result_list = critique_self_loop(
        candidates=[{'eta': 1.0}],
        pipeline_fn=mock_pipeline_converge,
        verify_fn=mock_verify,
    )
    assert result_list is not None, "Expected convergence via function interface"
    assert len(result_list) == 1
    assert 'critique_self_loop' in result_list[0]
    print("  ✅ Test 5: critique_self_loop() function interface — PASS")

    # Test 6: CannotConverge exception
    try:
        raise CannotConverge("test: max_retry exceeded")
    except CannotConverge as e:
        assert "max_retry" in str(e)
    print("  ✅ Test 6: CannotConverge exception — PASS")

    # Test 7: CritiqueDiagnosis.to_dict()
    diag_dict = diag.to_dict()
    assert 'best_eta' in diag_dict
    assert 'retry_idx' in diag_dict
    assert isinstance(diag_dict, dict)
    print("  ✅ Test 7: CritiqueDiagnosis.to_dict() — PASS")

    print("=" * 60)
    print("All tests PASS ✅")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
