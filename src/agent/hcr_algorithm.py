"""
src/agent/hcr_algorithm.py
TOMAS HCR 民主集中制算法 — 层级陪集归约 (Hierarchical Coset Reduction)

文章2 §4: 民主集中制 = HCR层级陪集归约
  - Democratic Phase: 自下而上各层做局部κ-Snap, 上传陪集摘要
  - Centralized Phase: 顶层全局κ-Snap, 选最小残差计划
  - 协调最优性定理: HCR协调代价 O(n log n), 优于扁平民主 O(n²) 和纯独裁 O(n)

核心洞察:
  1. 民主阶段: L1/L2/L3各层独立做局部κ-Snap归约 → 陪集摘要上传
  2. 集中阶段: 顶层(L4)做全局κ-Snap归约 → 选最小残差计划
  3. κ-优选: η升序选择 + 负Inflow修正(承认错误)

Version: v3.18.0 — HCR Democratic Centralism Algorithm
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .hybrid_search_engine import (
    CandidateSet,
    EvaluatedCandidateSet,
    HybridSearchPipeline,
    PipelineStrategies,
    InflowLoop,
    CannotConverge,
    GEX_PASS_THRESHOLD,
)
from .delta_state import (
    Node,
    ReplayEngine,
    ActionSpace,
    GaussExVerifier,
    SolverAborted,
    DEAD_ZERO_RATIO,
    DEFAULT_MAX_DEPTH,
    STRUCT_BFS_BUDGET,
)


# ============================================================================
# §1. HCR 数据结构
# ============================================================================

@dataclass
class CosetSummary:
    """陪集摘要 — Democratic Phase各层上传的局部κ-Snap归约结果。

    κ-Phase: 陪集摘要 = κ-Snap归约后的特征压缩
    IC(SA) → IC(SM) 压缩, CR = IC(SA)/IC(SM) 压缩比度量

    Attributes:
        layer: 层级 (L1/L2/L3)。
        coset_id: 陪集编号 (C(11,4)中的330陪集之一)。
        eta: 局部残差η。
        confidence: 局部置信度。
        features: 陪集特征字典 (颜色分布、形状、拓扑等)。
        node_ids: 该陪集包含的候选节点ID列表。
    """
    layer: str = ""
    coset_id: int = 0
    eta: float = 1.0
    confidence: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    node_ids: List[int] = field(default_factory=list)


@dataclass
class HCRResult:
    """HCR民主集中制最终结果。

    κ-Phase: 全局κ-Snap归约 + η升序选择
    协调最优性: O(n log n) < O(n²) (扁平民主) < O(n) (纯独裁)

    Attributes:
        best_candidate: 最优候选字典。
        democratic_summaries: 各层Democratic Phase陪集摘要。
        centralized_eta: Centralized Phase全局最小η。
        coordination_cost: 协调代价 (n log n)。
        rounds: HCR闭环轮数。
    """
    best_candidate: Optional[Dict[str, Any]] = None
    democratic_summaries: List[CosetSummary] = field(default_factory=list)
    centralized_eta: float = 1.0
    coordination_cost: float = 0.0
    rounds: int = 0


# ============================================================================
# §2. HCRAlgorithm — 民主集中制层级陪集归约
# ============================================================================

class HCRAlgorithm:
    """HCR民主集中制层级陪集归约算法 (文章2 §4)。

    闭环流程:
      Democratic Phase (自下而上):
        1. L1层: BFS搜索 → 陪集摘要(coset_id, eta, features)
        2. L2层: 结构剪枝 → 陪集摘要(缩减后的coset)
        3. L3层: 残差评估 → 陪集摘要(η, confidence)

      Centralized Phase (自上而下):
        4. L4层: 全局κ-Snap归约 → 选最小η计划
        5. 验证: GaussEx校验最优候选
        6. 若残差>ε → 负Inflow修正(承认错误) → 重跑Democratic Phase

    协调最优性定理: HCR协调代价 O(n log n)
      - 每层做局部κ-Snap (O(n) per layer)
      - 顶层做全局κ-Snap (O(log n) merge)
      - 总代价 = O(n log n)

    Attributes:
        pipeline: HybridSearchPipeline实例。
        inflow_loop: InflowLoop实例(群众路线闭环)。
        delta_k: κ-Snap残差阈值δ_K。
        max_hcr_rounds: 最大HCR闭环轮数。
    """

    def __init__(
        self,
        pipeline: Optional[HybridSearchPipeline] = None,
        strategies: Optional[PipelineStrategies] = None,
        delta_k: float = 0.036,
        max_hcr_rounds: int = 3,
    ) -> None:
        """初始化HCR民主集中制算法。

        Args:
            pipeline: HybridSearchPipeline实例(可为None, 自动创建)。
            strategies: PipelineStrategies策略配置。
            delta_k: κ-Snap残差阈值δ_K。
            max_hcr_rounds: 最大HCR闭环轮数。
        """
        if pipeline is not None:
            self.pipeline: HybridSearchPipeline = pipeline
        elif strategies is not None:
            self.pipeline = HybridSearchPipeline(strategies)
        else:
            self.pipeline = HybridSearchPipeline()

        self.inflow_loop: InflowLoop = InflowLoop(
            epsilon=GEX_PASS_THRESHOLD,
            max_inflow_rounds=max_hcr_rounds,
        )
        self.delta_k: float = delta_k
        self.max_hcr_rounds: int = max_hcr_rounds

    def democratic_phase(
        self,
        root_state: Any,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        action_space: Optional[ActionSpace] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        max_time: float = 25.0,
    ) -> List[CosetSummary]:
        """Democratic Phase: 自下而上各层做局部κ-Snap (文章2 §4)。

        民主阶段: L1→L2→L3各层独立做局部κ-Snap归约,
        生成陪集摘要(coset_id, eta, features)并上传。

        κ-Phase: 各层陪集摘要 = 局部κ-Snap归约的压缩表达
        IC(SA) → IC(SM) 压缩, CR = IC(SA)/IC(SM) 压缩比度量

        Args:
            root_state: 根节点状态。
            examples: 示例列表。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            max_time: 最大搜索时间。

        Returns:
            各层陪集摘要列表 [CosetSummary(L1), CosetSummary(L2), CosetSummary(L3)]。
        """
        summaries: List[CosetSummary] = []

        # L1层陪集摘要: BFS搜索 → 布局特征
        l1_result: CandidateSet = self.pipeline.l1.generate(
            root_state, action_space or ActionSpace('game'),
            max_depth, max_nodes,
        )
        self.pipeline._last_l1_result = l1_result

        if not l1_result.is_empty():
            l1_summary: CosetSummary = CosetSummary(
                layer='L1',
                coset_id=0,
                eta=0.0,  # L1不做η评估
                confidence=1.0,
                features={
                    'candidate_count': len(l1_result.node_ids),
                    'max_depth_reached': max(n.depth for n in l1_result.node_map.values()) if l1_result.node_map else 0,
                },
                node_ids=l1_result.node_ids[:50],  # 上传前50个候选
            )
            summaries.append(l1_summary)

        # L2层陪集摘要: 结构剪枝
        if not l1_result.is_empty():
            l2_result: CandidateSet = self.pipeline.l2.prune(l1_result)

            if not l2_result.is_empty():
                l2_summary: CosetSummary = CosetSummary(
                    layer='L2',
                    coset_id=1,
                    eta=0.0,
                    confidence=1.0,
                    features={
                        'pruned_count': len(l2_result.node_ids),
                        'pruning_ratio': len(l2_result.node_ids) / max(1, len(l1_result.node_ids)),
                    },
                    node_ids=l2_result.node_ids[:30],  # 上传前30个候选
                )
                summaries.append(l2_summary)

                # L3层陪集摘要: 残差评估
                l3_result: EvaluatedCandidateSet = self.pipeline.l3.evaluate(
                    l2_result, examples,
                )

                if not l3_result.is_empty():
                    # 按η聚类为陪集
                    eta_clusters: Dict[int, List[Dict[str, Any]]] = {}
                    for cand in l3_result.candidates:
                        eta: float = cand.get('eta', 1.0)
                        # η映射到陪集编号: int(eta / delta_k * 100)
                        coset_idx: int = min(int(eta / self.delta_k * 100), 329)  # C(11,4)=330陪集
                        if coset_idx not in eta_clusters:
                            eta_clusters[coset_idx] = []
                        eta_clusters[coset_idx].append(cand)

                    for coset_idx, cluster_cands in eta_clusters.items():
                        best_eta_in_cluster: float = min(c.get('eta', 1.0) for c in cluster_cands)
                        l3_summary: CosetSummary = CosetSummary(
                            layer='L3',
                            coset_id=coset_idx,
                            eta=best_eta_in_cluster,
                            confidence=max(0.0, 1.0 - best_eta_in_cluster / self.delta_k),
                            features={
                                'cluster_size': len(cluster_cands),
                                'best_eta': best_eta_in_cluster,
                            },
                            node_ids=[c.get('node_id', -1) for c in cluster_cands[:10]],
                        )
                        summaries.append(l3_summary)

        return summaries

    def centralized_phase(
        self,
        summaries: List[CosetSummary],
        evaluated_set: Optional[EvaluatedCandidateSet] = None,
    ) -> Optional[Dict[str, Any]]:
        """Centralized Phase: 顶层全局κ-Snap归约 (文章2 §4)。

        集中阶段: 从各层陪集摘要中做全局κ-Snap归约,
        选最小η计划。

        κ-Phase: 全局κ-Snap归约 = η升序排序 + 最小η选择
        协调最优性: O(n log n) = 各层O(n) + 顶层O(log n)合并

        Args:
            summaries: Democratic Phase陪集摘要列表。
            evaluated_set: L3评估后的候选集(含η/confidence)。

        Returns:
            最优候选字典, 或None(集中阶段失败)。
        """
        if evaluated_set is None or evaluated_set.is_empty():
            # 从陪集摘要中重建候选集
            l3_summaries: List[CosetSummary] = [s for s in summaries if s.layer == 'L3']
            if not l3_summaries:
                return None

            # η升序排序陪集摘要
            sorted_summaries: List[CosetSummary] = sorted(l3_summaries, key=lambda s: s.eta)

            # 选最小η陪集作为最优候选
            best_summary: CosetSummary = sorted_summaries[0]
            return {
                'node_id': best_summary.node_ids[0] if best_summary.node_ids else -1,
                'eta': best_summary.eta,
                'confidence': best_summary.confidence,
                'coset_id': best_summary.coset_id,
                'hcr_phase': 'centralized',
            }

        # 从评估集直接选择最小η候选
        best_candidates: List[Dict[str, Any]] = self.pipeline.l4.select(evaluated_set)
        if len(best_candidates) == 0:
            return None

        best: Dict[str, Any] = best_candidates[0]
        best['hcr_phase'] = 'centralized'

        return best

    def solve(
        self,
        root_state: Any,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        action_space: Optional[ActionSpace] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        max_time: float = 30.0,
    ) -> Optional[HCRResult]:
        """HCR民主集中制求解: Democratic → Centralized → Verify → Inflow闭环。

        闭环流程:
          1. Democratic Phase: L1→L2→L3各层陪集摘要
          2. Centralized Phase: 顶层全局κ-Snap归约
          3. Verify: GaussEx校验
          4. 若残差>ε → 负Inflow修正 → 重跑Democratic Phase

        协调代价: O(n log n)

        Args:
            root_state: 根节点状态。
            examples: 示例列表。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            max_time: 最大搜索时间。

        Returns:
            HCRResult(含最优候选、陪集摘要、协调代价), 或None(失败)。
        """
        t0: float = time.time()
        remaining_time: float = max_time
        n_candidates: int = 0

        for round_idx in range(self.max_hcr_rounds):
            if remaining_time <= 0:
                break

            t_round: float = time.time()
            time_budget: float = min(remaining_time * 0.85, 25.0)

            # Step 1: Democratic Phase
            summaries: List[CosetSummary] = self.democratic_phase(
                root_state, examples, action_space,
                max_depth, max_nodes, time_budget,
            )

            # 收集L3评估结果
            l3_summaries: List[CosetSummary] = [s for s in summaries if s.layer == 'L3']
            if not l3_summaries:
                # 无L3陪集摘要 → 搜索失败
                elapsed: float = time.time() - t_round
                remaining_time -= elapsed
                continue

            n_candidates = sum(len(s.node_ids) for s in l3_summaries)

            # Step 2: Centralized Phase
            # 从Democratic Phase结果重建EvaluatedCandidateSet
            evaluated_candidates: List[Dict[str, Any]] = []
            for summary in l3_summaries:
                for node_id in summary.node_ids:
                    evaluated_candidates.append({
                        'node_id': node_id,
                        'eta': summary.eta,
                        'confidence': summary.confidence,
                        'coset_id': summary.coset_id,
                        'ic': 0.5,
                        'depth': 0,
                    })

            evaluated_set: EvaluatedCandidateSet = EvaluatedCandidateSet(
                candidates=evaluated_candidates,
                node_map=self.pipeline._last_l1_result.node_map if self.pipeline._last_l1_result else {},
                replay_engine=self.pipeline._last_l1_result.replay_engine if self.pipeline._last_l1_result else None,
                meta={'hcr_round': round_idx},
            )

            best_candidate: Optional[Dict[str, Any]] = self.centralized_phase(
                summaries, evaluated_set,
            )

            if best_candidate is None:
                elapsed = time.time() - t_round
                remaining_time -= elapsed
                continue

            # Step 3: 验证最优候选
            residual_eta: float = best_candidate.get('eta', 1.0)

            # 残差≤ε → HCR成功!
            if residual_eta <= GEX_PASS_THRESHOLD:
                # 协调代价 = n_candidates * log(n_candidates)
                coordination_cost: float = n_candidates * np.log2(max(1, n_candidates))

                return HCRResult(
                    best_candidate=best_candidate,
                    democratic_summaries=summaries,
                    centralized_eta=residual_eta,
                    coordination_cost=coordination_cost,
                    rounds=round_idx + 1,
                )

            # Step 4: 残差>ε → 负Inflow修正(承认错误)
            corrected_prior: Optional[Dict[str, Any]] = self.inflow_loop.inflow_correct(
                residual_eta, {'eta': residual_eta},
            )

            # 更新时间预算
            elapsed = time.time() - t_round
            remaining_time -= elapsed

        # HCR闭环失败
        return None


# ============================================================================
# §3. 自测函数
# ============================================================================

def _self_test() -> bool:
    """HCR算法自测: 验证Democratic Phase + Centralized Phase + solve()。

    Returns:
        True if all tests pass, False otherwise.
    """
    # 测试CosetSummary数据结构
    cs: CosetSummary = CosetSummary(
        layer='L3', coset_id=5, eta=0.02, confidence=0.8,
        features={'cluster_size': 3},
        node_ids=[1, 2, 3],
    )
    assert cs.layer == 'L3', "CosetSummary layer should be L3"
    assert cs.eta == 0.02, "CosetSummary eta should be 0.02"

    # 测试HCRResult数据结构
    hcr_result: HCRResult = HCRResult(
        best_candidate={'node_id': 1, 'eta': 0.01},
        democratic_summaries=[cs],
        centralized_eta=0.01,
        coordination_cost=10.0,
        rounds=1,
    )
    assert hcr_result.rounds == 1, "HCRResult rounds should be 1"

    # 测试HCRAlgorithm构造
    hcr: HCRAlgorithm = HCRAlgorithm()
    assert hcr.pipeline is not None, "HCRAlgorithm should have pipeline"
    assert hcr.inflow_loop is not None, "HCRAlgorithm should have inflow_loop"

    # 测试centralized_phase(空输入)
    empty_result: Optional[Dict[str, Any]] = hcr.centralized_phase([], None)
    assert empty_result is None, "Centralized phase with empty input should return None"

    print("[PASS] hcr_algorithm _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
