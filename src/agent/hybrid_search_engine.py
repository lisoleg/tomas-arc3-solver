"""
src/agent/hybrid_search_engine.py
TOMAS 四层混合搜索管线 — HybridSearchPipeline

四层混合搜索架构: L1(候选生成) → L2(结构剪枝) → L3(残差评估) → L4(决策融合)

核心类:
  - CandidateSet: L1→L2数据载体
  - EvaluatedCandidateSet: L2→L3→L4数据载体 (含η和confidence)
  - PipelineStrategies: 策略配置 (l1/l2/l3/l4策略名)
  - L1CandidateGenerator Protocol: L1候选生成接口
  - L2StructurePruner Protocol: L2结构剪枝接口
  - L3ResidualEvaluator Protocol: L3残差评估接口
  - L4DecisionSelector Protocol: L4决策融合接口
  - HybridSearchPipeline: 管线编排器

Version: v3.18.0 — Hybrid Search Architecture
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from .delta_state import (
    Node,
    ReplayEngine,
    ActionSpace,
    GaussExVerifier,
    LayoutHasher,
    SolverAborted,
    BudgetExceeded,
    GEX_PASS_THRESHOLD,
    DEAD_ZERO_RATIO,
    MAX_RHAE_PER_TASK,
    DEFAULT_MAX_DEPTH,
    STRUCT_BFS_BUDGET,
    PARAM_BFS_BUDGET,
)

# v4.0 — Critique-Self-Loop独立模块 (deepcopy-safe, 泛化接口)
from .critique_loop import (
    CannotConverge,
    CritiqueSelfLoop,
    CritiqueResult,
    CritiqueDiagnosis,
    diagnose_eta,
    adjust_macro_params,
    critique_self_loop,
    log_critique_to_psi_audit,
)


# ============================================================================
# §1c. 群众路线 Inflow 闭环 (文章2 §3)
# ============================================================================

class InflowLoop:
    """群众路线 EML Inflow 闭环 — 感知→归纳→执行→校验→修正 (文章2 §3)。

    κ-Phase: 群众路线 = EML Inflow闭环
      Inflow ← sample(𝒟_env)  → 从环境采样感知数据
      Model ← κ-Snap(Inflow, Prior) → 归纳模型(陪集归约)
      Action ← select_min_η(Model) → 执行最小残差动作
      Obs, Reward ← Env.step(Action) → 校验结果
      若残差>ε → 修正Prior (负Inflow修正, 承认错误)

    允许负Inflow修正(承认错误) — 批评与自我批评的核心机制。
    当κ-Snap残差>ε时, 做Re-Inflow回溯(从SA缓存重跑κ-Snap)。

    Attributes:
        epsilon: 残差阈值ε, 超过则触发Prior修正。
        max_inflow_rounds: 最大Inflow闭环轮数。
        prior: Prior模型(陪集归约结果), 随闭环迭代更新。
        sa_cache: SA缓存(态势感知原始数据), 用于Re-Inflow回溯。
    """

    def __init__(
        self,
        epsilon: float = GEX_PASS_THRESHOLD,
        max_inflow_rounds: int = 5,
    ) -> None:
        """初始化群众路线Inflow闭环。

        Args:
            epsilon: 残差阈值ε, 默认GEX_PASS_THRESHOLD。
            max_inflow_rounds: 最大闭环轮数, 默认5。
        """
        self.epsilon: float = epsilon
        self.max_inflow_rounds: int = max_inflow_rounds
        self.prior: Optional[Dict[str, Any]] = None  # Prior模型
        self.sa_cache: List[Dict[str, Any]] = []  # SA缓存

    def inflow_sample(
        self,
        root_state: Any,
        action_space: Optional[ActionSpace] = None,
    ) -> Dict[str, Any]:
        """Inflow ← sample(𝒟_env): 从环境采样感知数据。

        κ-Phase: 感知 = 从环境获取态势信息(SA原始数据)
        采集Grid统计特征、实体位置、颜色分布等。

        Args:
            root_state: 根节点状态(game/grid)。
            action_space: ActionSpace实例。

        Returns:
            SA原始数据字典: {grid_stats, entity_positions, color_distribution, ...}
        """
        from .delta_state import _extract_game_grid

        sa_data: Dict[str, Any] = {
            'inflow_phase': 'sample',
            'timestamp': time.time(),
        }

        # Grid统计
        grid: Optional[np.ndarray] = None
        if isinstance(root_state, np.ndarray):
            grid = root_state
        else:
            grid = _extract_game_grid(root_state)

        if grid is not None:
            colors, counts = np.unique(grid, return_counts=True)
            total: int = grid.size
            color_dist: Dict[int, float] = {
                int(c): float(cnt) / total for c, cnt in zip(colors, counts)
            }
            sa_data['grid_stats'] = {
                'shape': grid.shape,
                'unique_colors': len(colors),
                'color_distribution': color_dist,
            }

        # 缓存SA数据 (用于Re-Inflow回溯)
        self.sa_cache.append(sa_data)

        return sa_data

    def inflow_model(
        self,
        inflow_data: Dict[str, Any],
        prior: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Model ← κ-Snap(Inflow, Prior): 归纳模型(陪集归约)。

        κ-Phase: 归纳 = κ-Snap陪集归约 + Prior修正
        从Inflow数据中提取陪集特征, 与Prior合并。

        Args:
            inflow_data: Inflow采样数据。
            prior: Prior模型(上一轮归纳结果)。

        Returns:
            归纳模型字典: {coset_features, eta_estimate, confidence, ...}
        """
        # Prior初始化
        if prior is None:
            prior = self.prior or {}

        # κ-Snap陪集归约 (简化版: 从SA数据提取陪集特征)
        model: Dict[str, Any] = {
            'inflow_phase': 'model',
            'prior': prior,
            'inflow_data': inflow_data,
            'coset_features': {},  # κ-Snap陪集归约结果
            'eta_estimate': 0.0,  # 预估残差
            'confidence': 0.0,  # 预估置信度
        }

        # 从Grid统计中提取陪集特征
        grid_stats: Optional[Dict[str, Any]] = inflow_data.get('grid_stats')
        if grid_stats is not None:
            color_dist: Dict[int, float] = grid_stats.get('color_distribution', {})
            # κ-Snap: 颜色分布 → 陪集特征 (330陪集编码)
            sorted_colors: List[int] = sorted(color_dist.keys())
            model['coset_features'] = {
                'sorted_colors': sorted_colors,
                'dominant_color': max(color_dist, key=color_dist.get) if color_dist else None,
                'entropy': -sum(
                    p * np.log2(p + 1e-10) for p in color_dist.values()
                ) if color_dist else 0.0,
            }

        # 更新Prior
        self.prior = model

        return model

    def inflow_execute(
        self,
        model: Dict[str, Any],
        pipeline: HybridSearchPipeline,
        root_state: Any,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        action_space: Optional[ActionSpace] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        max_time: float = 30.0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Action ← select_min_η(Model): 执行最小残差动作。

        κ-Phase: 执行 = κ-优选选择最小η动作
        使用HybridSearchPipeline执行四层混合搜索。

        Args:
            model: 归纳模型。
            pipeline: HybridSearchPipeline实例。
            root_state: 根节点状态。
            examples: 示例列表。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            max_time: 最大搜索时间。

        Returns:
            最优候选列表, 或None(执行失败)。
        """
        return pipeline.solve(
            root_state, examples, action_space,
            max_depth, max_nodes, max_time,
            **{'inflow_model': model},
        )

    def inflow_verify(
        self,
        result: Optional[List[Dict[str, Any]]],
    ) -> Tuple[float, Dict[str, Any]]:
        """Obs, Reward ← Env.step(Action): 校验结果。

        κ-Phase: 校验 = GaussEx残差校验 + κ-Snap自指校验
        检查执行结果的残差η是否≤ε。

        Args:
            result: 执行结果(最优候选列表)。

        Returns:
            (residual_eta, observation_dict) — 残差η和观察数据。
        """
        if result is None or len(result) == 0:
            return 1.0, {'observation': 'no_result', 'reward': 0.0}

        best: Dict[str, Any] = result[0]
        eta: float = best.get('eta', 1.0)
        confidence: float = best.get('confidence', 0.0)

        return eta, {
            'observation': 'evaluated',
            'eta': eta,
            'confidence': confidence,
            'reward': confidence,
            'solved': best.get('gex_result', {}).get('solved', False),
        }

    def inflow_correct(
        self,
        residual_eta: float,
        observation: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """若残差>ε → 修正Prior (负Inflow修正, 承认错误)。

        κ-Phase: 承认错误 = 负Inflow修正 + Re-Inflow回溯
        从SA缓存中回溯, 重新构建Prior模型。

        Args:
            residual_eta: 残差η。
            observation: 观察数据。

        Returns:
            修正后的Prior, 或None(残差≤ε, 无需修正)。
        """
        if residual_eta <= self.epsilon:
            return None  # 残差≤ε → 无需修正

        # 负Inflow修正: 承认错误, 从SA缓存回溯
        corrected_prior: Dict[str, Any] = {
            'correction_type': 'negative_inflow',
            'original_eta': residual_eta,
            'sa_cache_size': len(self.sa_cache),
            'correction_note': '承认错误: κ-Snap残差>ε → Prior需要修正',
        }

        # Re-Inflow回溯: 从最近SA缓存重建Prior
        if self.sa_cache:
            latest_sa: Dict[str, Any] = self.sa_cache[-1]
            corrected_prior['re_inflow_from'] = latest_sa.get('timestamp', 0)

        # 更新Prior
        self.prior = corrected_prior

        return corrected_prior

    def run_full_loop(
        self,
        pipeline: HybridSearchPipeline,
        root_state: Any,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        action_space: Optional[ActionSpace] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        max_time: float = 30.0,
    ) -> Optional[List[Dict[str, Any]]]:
        """执行完整的群众路线Inflow闭环。

        闭环流程: 感知→归纳→执行→校验→修正 (最多max_inflow_rounds轮)

        Args:
            pipeline: HybridSearchPipeline实例。
            root_state: 根节点状态。
            examples: 示例列表。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            max_time: 最大搜索时间。

        Returns:
            最终结果(最优候选列表), 或None(闭环失败)。
        """
        t0: float = time.time()
        remaining_time: float = max_time

        for round_idx in range(self.max_inflow_rounds):
            if remaining_time <= 0:
                break

            # Phase 1: Inflow ← sample(𝒟_env)
            inflow_data: Dict[str, Any] = self.inflow_sample(root_state, action_space)

            # Phase 2: Model ← κ-Snap(Inflow, Prior)
            model: Dict[str, Any] = self.inflow_model(inflow_data, self.prior)

            # Phase 3: Action ← select_min_η(Model)
            time_budget: float = min(remaining_time * 0.8, 25.0)  # 留20%给校验
            result: Optional[List[Dict[str, Any]]] = self.inflow_execute(
                model, pipeline, root_state, examples,
                action_space, max_depth, max_nodes, time_budget,
            )

            # Phase 4: Obs, Reward ← Env.step(Action)
            residual_eta, observation = self.inflow_verify(result)

            # 残差≤ε → 闭环成功!
            if residual_eta <= self.epsilon:
                if result is not None and len(result) > 0:
                    result[0]['inflow_loop'] = {
                        'round_idx': round_idx,
                        'residual_eta': residual_eta,
                        'observation': observation,
                    }
                    return result

            # Phase 5: 若残差>ε → 修正Prior
            corrected_prior: Optional[Dict[str, Any]] = self.inflow_correct(
                residual_eta, observation,
            )

            if corrected_prior is None:
                # 残差≤ε但result为空 → 可能是通关候选
                if result is not None:
                    return result
                break  # 无修正需要但无结果 → 退出闭环

            # 更新时间预算
            elapsed: float = time.time() - t0
            remaining_time = max_time - elapsed

        return None  # 闭环失败


# ============================================================================
# §1. 管线数据载体
# ============================================================================

@dataclass
class CandidateSet:
    """L1→L2 数据载体: L1候选生成结果。

    由L1策略产生的候选集合，包含节点ID列表、ReplayEngine
    和搜索元数据。供L2结构剪枝策略进一步筛选。

    Attributes:
        node_ids: 候选节点ID列表 (来自L1搜索)。
        node_map: 节点编号 → Node对象映射。
        replay_engine: IDO ReplayEngine实例。
        root_state: 根节点状态 (game/grid)。
        action_space: ActionSpace实例 (决定可用动作)。
        meta: 搜索元数据 (L1策略名、耗时等)。
    """

    node_ids: List[int] = field(default_factory=list)
    node_map: Dict[int, Node] = field(default_factory=dict)
    replay_engine: Optional[ReplayEngine] = None
    root_state: Any = None
    action_space: Optional[ActionSpace] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """返回候选数量。"""
        return len(self.node_ids)

    def is_empty(self) -> bool:
        """候选集是否为空。"""
        return len(self.node_ids) == 0


@dataclass
class EvaluatedCandidateSet:
    """L2→L3→L4 数据载体: 评估后的候选集合。

    由L2剪枝+L3评估后的候选集合，包含η残差、confidence置信度
    和GaussEx验证结果。供L4决策融合策略做最终选择。

    Attributes:
        candidates: 评估后的候选列表，每个候选包含:
            - 'node_id': 节点ID
            - 'eta': GaussEx残差η
            - 'confidence': 置信度 = 1 - η/δ_K
            - 'gex_result': GaussEx验证结果字典
            - 'ic': 信息基数IC
            - 'depth': 搜索深度
            - 'liu_score': Liu-Score优先级 (L4计算后)
        node_map: 节点编号 → Node对象映射 (从CandidateSet继承)。
        replay_engine: IDO ReplayEngine实例 (从CandidateSet继承)。
        meta: 评估元数据 (L2/L3策略名、耗时等)。
    """

    candidates: List[Dict[str, Any]] = field(default_factory=list)
    node_map: Dict[int, Node] = field(default_factory=dict)
    replay_engine: Optional[ReplayEngine] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """返回评估后候选数量。"""
        return len(self.candidates)

    def is_empty(self) -> bool:
        """评估集是否为空。"""
        return len(self.candidates) == 0

    def best_eta(self) -> Optional[float]:
        """返回最小残差η (最接近目标的候选)。

        Returns:
            最小η值，或None(空集)。
        """
        if self.is_empty():
            return None
        return min(cand.get('eta', 1.0) for cand in self.candidates)

    def best_confidence(self) -> Optional[float]:
        """返回最大置信度。

        Returns:
            最大confidence值，或None(空集)。
        """
        if self.is_empty():
            return None
        return max(cand.get('confidence', 0.0) for cand in self.candidates)


@dataclass
class PipelineStrategies:
    """管线策略配置: 声明每层使用的策略名。

    每层策略名对应具体策略类:
      - L1: 'macro_draft' | 'wall_bfs' | 'bfs' | 'dfs' | 'delta_replay' | 'direct'
      - L2: 'combo_symmetry' | 'prime_signature' | 'matroid_constraint' | 'pass_through'
      - L3: 'kappa_snap' | 'dead_zero_fuse' | 'gauss_ex' | 'asym_index' | 'pass_through'
      - L4: 'kappa_selector' | 'liu_priority'

    Attributes:
        l1_strategy: L1候选生成策略名。
        l2_strategy: L2结构剪枝策略名。
        l3_strategy: L3残差评估策略名。
        l4_strategy: L4决策融合策略名。
    """

    l1_strategy: str = "bfs"
    l2_strategy: str = "sym_pruner"
    l3_strategy: str = "gauss_ex"
    l4_strategy: str = "kappa_selector"


# ============================================================================
# §2. 四层 Protocol 接口定义
# ============================================================================

@runtime_checkable
class L1CandidateGenerator(Protocol):
    """L1候选生成 Protocol: 从初始状态生成候选节点集。

    L1层负责搜索空间探索，生成候选节点集(CandidateSet)。
    不同L1策略使用不同的搜索方式:
      - 'macro_draft': Macro-Draft宏观候选 (DSpark启发1, K≤8)
      - 'wall_bfs': KA59推箱Wall-BFS (物理原语约束)
      - 'bfs': 标准BFS (结构探索)
      - 'dfs': DFS枚举 (深度优先)
      - 'delta_replay': Δ-State Replay (IDO流贯)
      - 'direct': 直接计算 (简单游戏)

    Protocol方法:
      - generate(root_state, action_space, **kwargs) → CandidateSet
    """

    def generate(
        self,
        root_state: Any,
        action_space: ActionSpace,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        **kwargs: Any,
    ) -> CandidateSet:
        """从初始状态生成候选节点集。

        Args:
            root_state: 根节点状态 (game/grid)。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            **kwargs: 策略特定参数。

        Returns:
            CandidateSet: L1候选生成结果。
        """
        ...


@runtime_checkable
class L2StructurePruner(Protocol):
    """L2结构剪枝 Protocol: 从L1候选中剪枝低价值节点。

    L2层负责结构剪枝，基于对称性、签名、约束等结构特征
    剔除低价值候选。不同L2策略:
      - 'combo_symmetry': 组合对称性剪枝 (KA59推箱)
      - 'prime_signature': 素签名剪枝 (CN04仿射)
      - 'matroid_constraint': 拟阵约束剪枝
      - 'pass_through': 无剪枝(直接传递)

    Protocol方法:
      - prune(candidate_set) → CandidateSet
    """

    def prune(self, candidate_set: CandidateSet) -> CandidateSet:
        """从L1候选中剪枝低价值节点。

        Args:
            candidate_set: L1候选生成结果。

        Returns:
            CandidateSet: 剪枝后的候选集。
        """
        ...


@runtime_checkable
class L3ResidualEvaluator(Protocol):
    """L3残差评估 Protocol: 评估剪枝后候选的残差η和置信度。

    L3层负责残差评估，通过物化节点(Replay)并计算GaussEx
    残差η，转换为EvaluatedCandidateSet。不同L3策略:
      - 'kappa_snap': κ-Snap投影评估 (Octonion内积)
      - 'dead_zero_fuse': Dead-Zero熔断评估 (卞氏阈值)
      - 'gauss_ex': GaussEx残差评估 (像素匹配)
      - 'asym_index': 不对称指数评估
      - 'pass_through': 无评估(直接传递)
      - 'kappa_snap_seq_verify': κ-Snap Sequence-Level Verify (DSpark启发2, 逐步回放+Dead-Zero Fuse+Longest Accept Prefix)

    Protocol方法:
      - evaluate(candidate_set, examples) → EvaluatedCandidateSet
    """

    def evaluate(
        self,
        candidate_set: CandidateSet,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> EvaluatedCandidateSet:
        """评估剪枝后候选的残差η和置信度。

        Args:
            candidate_set: L2剪枝后的候选集。
            examples: 示例列表 [(input, output), ...]。

        Returns:
            EvaluatedCandidateSet: 评估后的候选集。
        """
        ...


@runtime_checkable
class L4DecisionSelector(Protocol):
    """L4决策融合 Protocol: 从评估后候选中选择最优方案。

    L4层负责决策融合，从L3评估后的候选中做最终选择。
    不同L4策略:
      - 'kappa_selector': κ-优选η升序 + Liu-Score双约束
      - 'liu_priority': 纯Liu-Score优先

    Protocol方法:
      - select(evaluated_set) → List[Dict[str, Any]]
    """

    def select(
        self,
        evaluated_set: EvaluatedCandidateSet,
    ) -> List[Dict[str, Any]]:
        """从评估后候选中选择最优方案。

        Args:
            evaluated_set: L3评估后的候选集。

        Returns:
            最优候选列表(按优先级排序)。
        """
        ...

    def confidence(self, eta: float) -> float:
        """计算κ-优选置信度。

        Args:
            eta: GaussEx残差η。

        Returns:
            置信度值 (0~1范围)。
        """
        ...




# ============================================================================
# §3. HybridSearchPipeline — 管线编排器
# ============================================================================

class HybridSearchPipeline:
    """四层混合搜索管线编排器: L1→L2→L3→L4。

    核心架构:
      L1(候选生成) → L2(结构剪枝) → L3(残差评估) → L4(决策融合)

    管线通过PipelineStrategies声明每层策略，HybridSearchPipeline
    负责创建策略实例并编排四层调用。

    设计决策:
      - solve() → 四层管线执行 + 验证
      - _build() → 根据PipelineStrategies创建策略实例
      - _execute() → 四层顺序执行
      - _verify() → GaussEx验证最优候选

    Attributes:
        strategies: PipelineStrategies策略配置。
        l1: L1策略实例。
        l2: L2策略实例。
        l3: L3策略实例。
        l4: L4策略实例。
    """

    # 策略名 → 策略类注册表
    L1_REGISTRY: Dict[str, type] = {}
    L2_REGISTRY: Dict[str, type] = {}
    L3_REGISTRY: Dict[str, type] = {}
    L4_REGISTRY: Dict[str, type] = {}

    def __init__(
        self,
        strategies: Optional[PipelineStrategies] = None,
    ) -> None:
        """初始化四层混合搜索管线。

        Args:
            strategies: PipelineStrategies策略配置。
                如果None，使用默认配置(bfs/sym_pruner/gauss_ex/kappa_selector)。
        """
        self.strategies: PipelineStrategies = strategies or PipelineStrategies()
        self.l1: Optional[L1CandidateGenerator] = None
        self.l2: Optional[L2StructurePruner] = None
        self.l3: Optional[L3ResidualEvaluator] = None
        self.l4: Optional[L4DecisionSelector] = None

        # 保存中间结果供外部提取 (Phase 0.5 plan转换)
        self._last_l1_result: Optional[CandidateSet] = None

        # 构建策略实例
        self._build()

    def _build(self) -> None:
        """根据PipelineStrategies创建策略实例。

        从各层注册表中查找策略名对应的策略类，实例化后赋值。
        注册表由各策略模块(l1_strategies, l2_strategies, etc.)填充。
        """
        # L1策略
        l1_name: str = self.strategies.l1_strategy
        l1_cls: Optional[type] = self.L1_REGISTRY.get(l1_name)
        if l1_cls is not None:
            self.l1 = l1_cls()  # type: ignore[assignment]
        else:
            # 默认: BFSPathCandidateGenerator
            self.l1 = _DefaultL1BFS()

        # L2策略
        l2_name: str = self.strategies.l2_strategy
        l2_cls: Optional[type] = self.L2_REGISTRY.get(l2_name)
        if l2_cls is not None:
            self.l2 = l2_cls()  # type: ignore[assignment]
        else:
            self.l2 = _DefaultL2SymPruner()

        # L3策略
        l3_name: str = self.strategies.l3_strategy
        l3_cls: Optional[type] = self.L3_REGISTRY.get(l3_name)
        if l3_cls is not None:
            self.l3 = l3_cls()  # type: ignore[assignment]
        else:
            self.l3 = _DefaultL3GaussEx()

        # L4策略
        l4_name: str = self.strategies.l4_strategy
        l4_cls: Optional[type] = self.L4_REGISTRY.get(l4_name)
        if l4_cls is not None:
            self.l4 = l4_cls()  # type: ignore[assignment]
        else:
            self.l4 = _DefaultL4Kappa()

    def solve(
        self,
        root_state: Any,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        action_space: Optional[ActionSpace] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        max_time: float = 30.0,
        **kwargs: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        """四层混合搜索管线: L1→L2→L3→L4 → 验证 → 返回最优方案。

        执行流程:
          1. L1: 候选生成 → CandidateSet
          2. L2: 结构剪枝 → CandidateSet (缩减候选)
          3. L3: 残差评估 → EvaluatedCandidateSet (含η/confidence)
          4. L4: 决策融合 → 最优候选列表
          5. 验证: GaussEx校验最优候选

        Args:
            root_state: 根节点状态 (game engine或np.ndarray)。
            examples: 示例列表 [(input, output), ...]。
            action_space: ActionSpace实例 (默认game mode)。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            max_time: 最大搜索时间(秒)。
            **kwargs: 策略特定参数。

        Returns:
            最优候选列表(按优先级排序)，或None(搜索失败)。
        """
        t0: float = time.time()

        if action_space is None:
            action_space = ActionSpace('game')

        # Step 1: L1候选生成
        l1_result: CandidateSet = self.l1.generate(
            root_state, action_space, max_depth, max_nodes, **kwargs,
        )
        l1_result.meta['l1_time'] = time.time() - t0
        self._last_l1_result = l1_result  # 保存供Phase 0.5提取

        if l1_result.is_empty():
            return None  # L1搜索失败 → 无候选

        # Step 2: L2结构剪枝
        t2: float = time.time()
        l2_result: CandidateSet = self.l2.prune(l1_result)
        l2_result.meta['l2_time'] = time.time() - t2

        if l2_result.is_empty():
            return None  # L2剪枝后无候选

        # Step 3: L3残差评估
        t3: float = time.time()
        l3_result: EvaluatedCandidateSet = self.l3.evaluate(
            l2_result, examples,
        )
        l3_result.meta['l3_time'] = time.time() - t3

        if l3_result.is_empty():
            return None  # L3评估后无候选

        # Step 4: L4决策融合
        t4: float = time.time()
        best_candidates: List[Dict[str, Any]] = self.l4.select(l3_result)
        l3_result.meta['l4_time'] = time.time() - t4

        if len(best_candidates) == 0:
            # ★ 升级2+5: κ-优选选不出候选 → 触发critique_self_loop (文章 §14.1)
            # 批评与自我批评 = κ-Snap自指残差校验 + Dead-Zero熔断
            # diagnose(η) → adjust_macro(plan, diagnosis) → 最多max_retry=3次
            # ★ 升级5: 读取底层L4选择器的needs_critique和critique_diagnosis
            critique_diagnosis: str = ""
            if hasattr(self.l4, 'needs_critique'):
                critique_diagnosis = getattr(self.l4, 'critique_diagnosis', '')
            critique_result: Optional[List[Dict[str, Any]]] = self._critique_self_loop(
                l3_result, replay_engine=l3_result.replay_engine, examples=examples,
                root_state=root_state, action_space=action_space,
                max_depth=max_depth, max_nodes=max_nodes,
                max_time_remaining=max_time - (time.time() - t0),
                l4_diagnosis=critique_diagnosis,
            )
            if critique_result is not None:
                return critique_result
            return None  # critique_self_loop也失败 → 真正无法求解

        # Step 5: 验证最优候选
        verified: Optional[List[Dict[str, Any]]] = self._verify(
            best_candidates, l3_result.replay_engine, examples,
        )

        return verified

    def _execute(
        self,
        root_state: Any,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        action_space: Optional[ActionSpace] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        **kwargs: Any,
    ) -> Tuple[CandidateSet, CandidateSet, EvaluatedCandidateSet, List[Dict[str, Any]]]:
        """四层顺序执行(不验证): L1→L2→L3→L4。

        用于调试和中间结果分析。

        Args:
            root_state: 根节点状态。
            examples: 示例列表。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            **kwargs: 策略特定参数。

        Returns:
            (l1_result, l2_result, l3_result, l4_result) 四层中间结果。
        """
        if action_space is None:
            action_space = ActionSpace('game')

        l1_result: CandidateSet = self.l1.generate(
            root_state, action_space, max_depth, max_nodes, **kwargs,
        )
        l2_result: CandidateSet = self.l2.prune(l1_result)
        l3_result: EvaluatedCandidateSet = self.l3.evaluate(l2_result, examples)
        l4_result: List[Dict[str, Any]] = self.l4.select(l3_result)

        return l1_result, l2_result, l3_result, l4_result

    def _verify(
        self,
        best_candidates: List[Dict[str, Any]],
        replay_engine: Optional[ReplayEngine],
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[List[Dict[str, Any]]]:
        """验证最优候选: GaussEx校验 + 置信度过滤。

        对L4选择的最优候选做最终验证:
          1. 物化候选节点 (ReplayEngine.replay)
          2. GaussEx验证 (GaussExVerifier.verify)
          3. 置信度过滤 (confidence ≥ 1/6)
          4. 返回验证通过的候选列表

        Args:
            best_candidates: L4选择的最优候选列表。
            replay_engine: ReplayEngine实例。
            examples: 示例列表。

        Returns:
            验证通过的候选列表，或None(全部失败)。
        """
        if replay_engine is None:
            # 无ReplayEngine → 无法物化验证 → 直接返回候选
            return best_candidates

        verifier: GaussExVerifier = GaussExVerifier()
        verified: List[Dict[str, Any]] = []

        for cand in best_candidates:
            node_id: int = cand.get('node_id', -1)
            eta: float = cand.get('eta', 1.0)
            confidence: float = cand.get('confidence', 0.0)

            # 置信度过滤: confidence ≥ 1/6 (卞氏阈值)
            if confidence < GEX_PASS_THRESHOLD:
                continue

            # 物化候选节点
            try:
                state: Any = replay_engine.replay(node_id)
            except (KeyError, SolverAborted):
                continue  # 物化失败 → 跳过

            # Grid mode: 直接验证
            if replay_engine.mode == 'grid' and isinstance(state, np.ndarray):
                result: Dict[str, Any] = verifier.verify(state, examples)
                cand['final_gex'] = result
                if result['passed']:
                    verified.append(cand)

            # Game mode: 尝试提取Grid后验证
            elif replay_engine.mode == 'game':
                from .delta_state import _extract_game_grid
                grid: Optional[np.ndarray] = _extract_game_grid(state)
                if grid is not None:
                    result = verifier.verify(grid, examples)
                    cand['final_gex'] = result
                    if result['passed']:
                        verified.append(cand)
                else:
                    # 无法提取Grid → 基于confidence判断
                    if confidence >= GEX_PASS_THRESHOLD:
                        verified.append(cand)

        if len(verified) == 0:
            return None

        return verified

    def _critique_self_loop(
        self,
        evaluated_set: EvaluatedCandidateSet,
        replay_engine: Optional[ReplayEngine] = None,
        examples: List[Tuple[np.ndarray, np.ndarray]] = [],
        root_state: Any = None,
        action_space: Optional[ActionSpace] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        max_time_remaining: float = 10.0,
        max_retry: int = 3,
        l4_diagnosis: str = "",
    ) -> Optional[List[Dict[str, Any]]]:
        """critique_self_loop 自纠错循环 — 委托给独立CritiqueSelfLoop引擎 (v4.0)。

        κ-优选选不出候选(η > δ_K)时触发自纠错:
          1. diagnose(η): 分析最大残差来源
          2. adjust_macro: 根据诊断调整搜索策略
          3. 重跑L1→L2→L3→L4管线(放宽参数)
          4. 最多max_retry=3次, 失败则raise CannotConverge

        v4.0: 委托给独立CritiqueSelfLoop模块，通过回调函数接入pipeline。
        κ-Phase: 批评与自我批评 = κ-Snap自指残差校验 + Dead-Zero熔断

        Args:
            evaluated_set: L3评估后的候选集(η都>δ_K)。
            replay_engine: ReplayEngine实例。
            examples: 示例列表。
            root_state: 根节点状态。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            max_time_remaining: 剩余时间预算(秒)。
            max_retry: 最大自纠错轮数, 默认3。
            l4_diagnosis: L4选择器诊断信息。

        Returns:
            自纠错后的最优候选列表, 或None(收敛失败)。
        """
        # ── 构建管线回调函数 ──
        # pipeline_fn: 调用self的L1→L2→L3→L4管线
        def pipeline_fn(depth: int, nodes: int, **kwargs) -> List[Dict[str, Any]]:
            """L1→L2→L3→L4管线回调 — 供CritiqueSelfLoop调用。"""
            try:
                l1_result: CandidateSet = self.l1.generate(
                    root_state, action_space or ActionSpace('game'),
                    depth, nodes,
                    **kwargs,
                )
            except Exception:
                return []  # L1失败 → 空列表

            if l1_result.is_empty():
                return []

            l2_result: CandidateSet = self.l2.prune(l1_result)
            if l2_result.is_empty():
                return []

            l3_result: EvaluatedCandidateSet = self.l3.evaluate(l2_result, examples)
            if l3_result.is_empty():
                return []

            return self.l4.select(l3_result)

        # verify_fn: 调用self._verify验证候选
        def verify_fn(candidates: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
            """验证回调 — 供CritiqueSelfLoop调用。"""
            return self._verify(candidates, replay_engine, examples)

        # ── 委托给独立CritiqueSelfLoop引擎 ──
        engine = CritiqueSelfLoop(
            max_retry=max_retry,
            base_depth=max_depth,
            base_nodes=max_nodes,
            max_time_budget=max_time_remaining,
        )

        # 将evaluated_set.candidates转换为List[Dict]供diagnose_eta使用
        candidates_list = evaluated_set.candidates if hasattr(evaluated_set, 'candidates') \
            else list(evaluated_set)

        result: CritiqueResult = engine.run(
            candidates=candidates_list,
            pipeline_fn=pipeline_fn,
            l4_diagnosis=l4_diagnosis,
            verify_fn=verify_fn,
        )

        # ── 记录ψ-Audit日志 ──
        if result.converged and result.verified_candidates:
            log_critique_to_psi_audit(result, result.verified_candidates[0])
        else:
            log_critique_to_psi_audit(result)

        if result.converged:
            return result.verified_candidates
        return None

    def _diagnose_eta(self, evaluated_set: EvaluatedCandidateSet) -> Dict[str, Any]:
        """diagnose(η) — 委托给独立diagnose_eta函数 (v4.0)。

        κ-Phase: 自指残差校验 = κ-Snap project self_view → KS_GX residual_self
        如果residual_self > δ_K → DZFUSE + INFLOW correction

        Args:
            evaluated_set: L3评估后的候选集。

        Returns:
            诊断字典(兼容旧接口)。
        """
        candidates = evaluated_set.candidates if hasattr(evaluated_set, 'candidates') \
            else list(evaluated_set)
        diag: CritiqueDiagnosis = diagnose_eta(candidates)
        return diag.to_dict()


# ============================================================================
# §4. 默认策略实现 (fallback)
# ============================================================================

class _DefaultL1BFS:
    """默认L1策略: 标准BFS候选生成 + 大网格depth cap + Wall-BFS bounding。

    使用delta_state.structural_bfs做BFS搜索。
    对大网格(H*W >= 64)强制max_depth=2以减少候选数。
    小网格保持原depth=3以保证搜索充分性。

    Attributes:
        LARGE_GRID_THRESHOLD: 大网格阈值，H*W >= 此值视为大网格。
        SMALL_GRID_MAX_DEPTH: 小网格最大搜索深度。
        LARGE_GRID_MAX_DEPTH: 大网格最大搜索深度(强制≤2)。
    """

    LARGE_GRID_THRESHOLD: int = 64
    SMALL_GRID_MAX_DEPTH: int = 3
    LARGE_GRID_MAX_DEPTH: int = 2

    def generate(
        self,
        root_state: Any,
        action_space: ActionSpace,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        **kwargs: Any,
    ) -> CandidateSet:
        """BFS候选生成 + 大网格depth cap。

        对大网格(H*W >= 64)强制max_depth=2，避免生成过多候选。
        小网格保持原depth=3以保证搜索充分性。

        Args:
            root_state: 根节点状态 (game/grid)。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度(会被大网格depth cap覆盖)。
            max_nodes: 最大扩展节点数。
            **kwargs: 策略特定参数。

        Returns:
            CandidateSet: L1候选生成结果。
        """
        from .delta_state import structural_bfs

        # 大网格检测: 强制depth cap
        effective_max_depth: int = max_depth
        if is_large_grid(root_state):
            effective_max_depth = min(max_depth, self.LARGE_GRID_MAX_DEPTH)
        else:
            effective_max_depth = min(max_depth, self.SMALL_GRID_MAX_DEPTH)

        # 创建根节点和node_map
        node_map: Dict[int, Node] = {}
        root_node: Node = Node(id=0, parent_id=-1, action="root", depth=0)
        node_map[0] = root_node

        # 创建ReplayEngine
        replay_engine: ReplayEngine = ReplayEngine(
            root_state=root_state,
            node_map=node_map,
            mode=action_space._mode,
        )

        # 大网格时减少max_nodes预算(避免BFS爆炸)
        effective_max_nodes: int = max_nodes
        if is_large_grid(root_state):
            effective_max_nodes = min(max_nodes, 20)  # 大网格时限制节点数

        # 执行BFS搜索 (structural_bfs内部也受depth控制)
        candidates: List[int] = structural_bfs(
            root_id=0,
            node_map=node_map,
            action_space=action_space,
            max_nodes=effective_max_nodes,
        )

        # 过滤超过effective_max_depth的候选
        if effective_max_depth < DEFAULT_MAX_DEPTH:
            candidates = [
                nid for nid in candidates
                if node_map[nid].depth <= effective_max_depth
            ]

        meta: Dict[str, Any] = {
            'l1_strategy': 'bfs',
            'effective_max_depth': effective_max_depth,
            'effective_max_nodes': effective_max_nodes,
            'is_large_grid': is_large_grid(root_state),
        }

        return CandidateSet(
            node_ids=candidates,
            node_map=node_map,
            replay_engine=replay_engine,
            root_state=root_state,
            action_space=action_space,
            meta=meta,
        )


def is_large_grid(state: Any) -> bool:
    """判断是否为大网格(H*W >= 64)。

    大网格(L1+关卡)搜索空间爆炸风险高，需要depth cap限制。
    对于np.ndarray直接用shape判断；对于game engine对象
    通过提取grid或使用fallback默认值判断。

    Args:
        state: 根节点状态 (game engine或np.ndarray)。

    Returns:
        True如果H*W >= 64 (大网格)，False otherwise。
    """
    if isinstance(state, np.ndarray):
        H: int = state.shape[0]
        W: int = state.shape[1]
    else:
        # game mode: 提取grid
        try:
            from .delta_state import _extract_game_grid
            grid: Optional[np.ndarray] = _extract_game_grid(state)
            if grid is not None:
                H, W = grid.shape[0], grid.shape[1]
            else:
                H, W = 8, 8  # default fallback
        except Exception:
            H, W = 8, 8  # default fallback
    return H * W >= 64


class _DefaultL2SymPruner:
    """默认L2策略: 对称hash剪枝 + Prime-Signature快拒。

    使用8-对称归一化hash(canonical_signature)和(opcode, param_hash)去重。
    替代旧版 _DefaultL2PassThrough (完全不剪枝)。

    对称剪枝:
      - 计算grid的8-对称归一化hash (rot90×4 + mirror, 取min tuple)
      - 对每个候选节点生成 (canon, opcode, param_hash) 签名
      - 同签名候选只保留第一个(去重)

    Attributes:
        _sym_cache: grid_tuple → canonical_sig 缓存 (避免重复计算8-对称)。
        _seen_this_frame: 当前帧已见签名集合 (每次prune清空)。
    """

    def __init__(self) -> None:
        """初始化对称剪枝器。"""
        self._sym_cache: Dict[tuple, tuple] = {}
        self._seen_this_frame: set = set()

    def prune(self, candidate_set: CandidateSet) -> CandidateSet:
        """对称剪枝+去重: 基于8-对称归一化hash和(opcode, param_hash)签名。

        Args:
            candidate_set: L1候选生成结果。

        Returns:
            CandidateSet: 剪枝后的候选集(去除对称重复和签名重复)。
        """
        self._seen_this_frame.clear()

        # 计算root state的canonical signature
        canon: tuple = self._canonical_signature(candidate_set.root_state)

        filtered_ids: List[int] = []
        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            # (canon, opcode, param_hash) 去重签名
            action_str: str = node.action
            meta_hash: int = hash(str(node.meta))
            sig: tuple = (canon, action_str, meta_hash)
            if sig in self._seen_this_frame:
                continue
            self._seen_this_frame.add(sig)
            filtered_ids.append(node_id)

        pruned_count: int = len(candidate_set.node_ids) - len(filtered_ids)
        return CandidateSet(
            node_ids=filtered_ids,
            node_map=dict(candidate_set.node_map),
            replay_engine=candidate_set.replay_engine,
            root_state=candidate_set.root_state,
            action_space=candidate_set.action_space,
            meta={
                'l2_strategy': 'sym_pruner',
                'pruned': pruned_count,
                'l1_meta': candidate_set.meta,
            },
        )

    def _canonical_signature(self, state: Any) -> tuple:
        """计算8-对称归一化hash: rot90×4 + mirror, 取min tuple。

        对grid做4次旋转(rot90)×2(含水平翻转)共8种对称变换，
        取所有变换结果中字典序最小的tuple作为canonical signature。
        同一grid在不同旋转/镜像下会得到相同的canonical signature，
        实现对称去重。

        Args:
            state: 根节点状态 (game engine或np.ndarray)。

        Returns:
            canonical signature tuple (8-对称归一化后的最小表示)。
        """
        grid: Optional[np.ndarray] = None
        if isinstance(state, np.ndarray):
            grid = state
        else:
            from .delta_state import _extract_game_grid
            grid = _extract_game_grid(state)

        if grid is None:
            return ()

        key: tuple = tuple(tuple(r) for r in grid.tolist())
        if key in self._sym_cache:
            return self._sym_cache[key]

        # 8 symmetry variants: 4 rotations × 2 (original + mirror)
        variants: List[tuple] = []
        for i in range(4):
            v: np.ndarray = np.rot90(grid, i)
            variants.append(tuple(tuple(r) for r in v.tolist()))
            variants.append(tuple(tuple(r) for r in np.fliplr(v).tolist()))

        canon: tuple = min(variants)
        self._sym_cache[key] = canon
        return canon


class _DefaultL2PassThrough:
    """旧版L2策略: PassThrough(不剪枝) — 仅用于向后兼容。

    直接传递L1候选集，不做任何剪枝。
    已被 _DefaultL2SymPruner 替代，保留供手动指定 'pass_through' 策略时使用。
    """

    def prune(self, candidate_set: CandidateSet) -> CandidateSet:
        """PassThrough剪枝(不修改候选集)。"""
        return CandidateSet(
            node_ids=list(candidate_set.node_ids),
            node_map=dict(candidate_set.node_map),
            replay_engine=candidate_set.replay_engine,
            root_state=candidate_set.root_state,
            action_space=candidate_set.action_space,
            meta={'l2_strategy': 'pass_through', 'l1_meta': candidate_set.meta},
        )


class _DefaultL3GaussEx:
    """默认L3策略: GaussEx残差评估 + 增量diff-residual优化。

    物化候选节点并计算GaussEx残差η。使用增量diff-residual优化：
    当连续候选的grid变化较小时，只计算变化cell的残差增量，
    避免O(HW)的full GaussEx重计算。同时支持early-stop：
    当η低于DELTA_K_EARLY阈值时立即返回最优候选。

    Attributes:
        DELTA_K_EARLY: early-stop阈值，η低于此值时立即返回候选。
    """

    DELTA_K_EARLY: float = 0.005  # early-stop threshold

    def evaluate(
        self,
        candidate_set: CandidateSet,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> EvaluatedCandidateSet:
        """GaussEx残差评估 + 增量diff-residual + early-stop。

        评估流程:
          1. 第一个候选: full GaussEx验证
          2. 后续候选: 增量diff-residual (只计算变化cell的残差增量)
          3. early-stop: η < DELTA_K_EARLY → 立即返回最优候选

        Args:
            candidate_set: L2剪枝后的候选集。
            examples: 示例列表 [(input, output), ...]。

        Returns:
            EvaluatedCandidateSet: 评估后的候选集(含η/confidence/early_stop标记)。
        """
        from .delta_state import parametric_bfs, _extract_game_grid

        verifier: GaussExVerifier = GaussExVerifier()
        candidates: List[Dict[str, Any]] = []

        replay_engine: Optional[ReplayEngine] = candidate_set.replay_engine
        if replay_engine is None:
            return EvaluatedCandidateSet(
                candidates=candidates,
                node_map=candidate_set.node_map,
                replay_engine=None,
                meta={'l3_strategy': 'gauss_ex_incremental', 'error': 'no_replay_engine'},
            )

        # 提取goal_grid from examples (最终output)
        goal_grid: Optional[np.ndarray] = None
        if len(examples) > 0:
            goal_grid = examples[-1][1] if len(examples[-1]) > 1 else None

        # Track previous state for incremental residual
        prev_eta: Optional[float] = None
        prev_grid: Optional[np.ndarray] = None

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            # 物化节点
            try:
                state: Any = replay_engine.replay(node_id)
            except (KeyError, SolverAborted):
                continue

            # Grid mode: 直接验证
            if replay_engine.mode == 'grid' and isinstance(state, np.ndarray):
                grid: np.ndarray = state

                # Incremental residual optimization
                if (
                    prev_grid is not None
                    and prev_eta is not None
                    and goal_grid is not None
                    and isinstance(prev_grid, np.ndarray)
                    and prev_grid.shape == grid.shape
                    and goal_grid.shape == grid.shape
                ):
                    eta: float = self._incremental_residual(
                        prev_grid, grid, goal_grid, prev_eta,
                    )
                    # For incremental, we don't have full gex_result, compute confidence from eta
                    confidence: float = max(0.0, 1.0 - eta / 0.036)
                    ic: float = node.meta.get('ic', 0.5)
                    cand_dict: Dict[str, Any] = {
                        'node_id': node_id,
                        'eta': eta,
                        'confidence': confidence,
                        'ic': ic,
                        'depth': node.depth,
                    }
                else:
                    # First candidate or shape mismatch: full GaussEx
                    result: Dict[str, Any] = verifier.verify(grid, examples)
                    eta = result.get('max_error', 1.0)
                    confidence = max(0.0, 1.0 - eta / 0.036)
                    ic = node.meta.get('ic', 0.5)
                    cand_dict = {
                        'node_id': node_id,
                        'eta': eta,
                        'confidence': confidence,
                        'gex_result': result,
                        'ic': ic,
                        'depth': node.depth,
                    }

                # Early-stop check
                if eta < self.DELTA_K_EARLY:
                    cand_dict['early_stop'] = True
                    candidates.append(cand_dict)
                    # Immediately return — no need to evaluate remaining candidates
                    return EvaluatedCandidateSet(
                        candidates=candidates,
                        node_map=candidate_set.node_map,
                        replay_engine=replay_engine,
                        meta={
                            'l3_strategy': 'gauss_ex_incremental',
                            'early_stopped': True,
                            'early_stop_node_id': node_id,
                        },
                    )

                candidates.append(cand_dict)

                # Update prev for incremental
                prev_grid = grid.copy()
                prev_eta = eta

            # Game mode: 提取Grid后验证
            elif replay_engine.mode == 'game':
                from .delta_state import _extract_game_grid
                grid = _extract_game_grid(state)
                if grid is not None:
                    # Incremental residual optimization (same logic as grid mode)
                    if (
                        prev_grid is not None
                        and prev_eta is not None
                        and goal_grid is not None
                        and isinstance(prev_grid, np.ndarray)
                        and prev_grid.shape == grid.shape
                        and goal_grid.shape == grid.shape
                    ):
                        eta = self._incremental_residual(
                            prev_grid, grid, goal_grid, prev_eta,
                        )
                        confidence = max(0.0, 1.0 - eta / 0.036)
                        ic = node.meta.get('ic', 0.5)
                        cand_dict = {
                            'node_id': node_id,
                            'eta': eta,
                            'confidence': confidence,
                            'ic': ic,
                            'depth': node.depth,
                        }
                    else:
                        result = verifier.verify(grid, examples)
                        eta = result.get('max_error', 1.0)
                        confidence = max(0.0, 1.0 - eta / 0.036)
                        ic = node.meta.get('ic', 0.5)
                        cand_dict = {
                            'node_id': node_id,
                            'eta': eta,
                            'confidence': confidence,
                            'gex_result': result,
                            'ic': ic,
                            'depth': node.depth,
                        }

                    # Early-stop check
                    if eta < self.DELTA_K_EARLY:
                        cand_dict['early_stop'] = True
                        candidates.append(cand_dict)
                        return EvaluatedCandidateSet(
                            candidates=candidates,
                            node_map=candidate_set.node_map,
                            replay_engine=replay_engine,
                            meta={
                                'l3_strategy': 'gauss_ex_incremental',
                                'early_stopped': True,
                                'early_stop_node_id': node_id,
                            },
                        )

                    candidates.append(cand_dict)
                    prev_grid = grid.copy()
                    prev_eta = eta

        return EvaluatedCandidateSet(
            candidates=candidates,
            node_map=candidate_set.node_map,
            replay_engine=replay_engine,
            meta={'l3_strategy': 'gauss_ex_incremental'},
        )

    def _incremental_residual(
        self,
        prev_grid: np.ndarray,
        new_grid: np.ndarray,
        goal_grid: np.ndarray,
        prev_eta: float,
    ) -> float:
        """增量残差计算: η_new = η_prev + Σ_{changed cells} Δ(cell, goal)。

        只计算变化cell的残差增量，避免O(HW)的full GaussEx重计算。
        对于相邻候选(如BFS扩展的同级节点)，grid变化通常很小，
        只改变少数cell，增量计算复杂度O(changed)而非O(HW)。

        Args:
            prev_grid: 前一个候选的grid (numpy 2D array)。
            new_grid: 当前候选的grid (numpy 2D array)。
            goal_grid: 目标grid (从examples提取的最终output)。
            prev_eta: 前一个候选的η值。

        Returns:
            当前候选的η值 (增量计算结果)。
        """
        if not isinstance(prev_grid, np.ndarray) or not isinstance(new_grid, np.ndarray):
            return prev_eta  # fallback

        # 找到变化的cell位置
        changed = np.where(prev_grid != new_grid)
        delta: float = 0.0
        for r, c in zip(changed[0], changed[1]):
            old_diff: float = abs(int(prev_grid[r, c]) - int(goal_grid[r, c]))
            new_diff: float = abs(int(new_grid[r, c]) - int(goal_grid[r, c]))
            delta += new_diff - old_diff

        # Normalize delta by total grid size (same as GaussExVerifier)
        total_pixels: int = goal_grid.size
        if total_pixels > 0:
            delta /= float(total_pixels)

        return max(0.0, prev_eta + delta)


class _DefaultL4Kappa:
    """默认L4策略: κ-优选η升序选择器 + early-stop + 双阈值通道。

    使用kappa_selector.KappaEtaAscendSelector做最终选择。
    支持early-stop: 当L3标记了early_stop候选或η < DELTA_K_EARLY时，
    立即返回最优候选，不做完整排序。

    ★ 升级6: 双阈值通道 — 硬通道优先, 空时切换创意通道。

    Attributes:
        DELTA_K_EARLY: early-stop阈值，η低于此值时立即返回。
        _selector: KappaEtaAscendSelector实例 (支持双阈值)。
    """

    DELTA_K_EARLY: float = 0.005

    def __init__(self) -> None:
        """初始化L4 κ-优选选择器 (双阈值模式)。"""
        from .kappa_selector import KappaEtaAscendSelector
        self._selector: KappaEtaAscendSelector = KappaEtaAscendSelector(
            dual_threshold=True,  # ★ 升级6: 启用双阈值
        )

    def select(
        self,
        evaluated_set: EvaluatedCandidateSet,
    ) -> List[Dict[str, Any]]:
        """κ-优选η升序选择 + early-stop。

        选择流程:
          1. 检查L3的early_stop标记 → 有则立即返回该候选
          2. η升序排序候选
          3. early-stop: η < DELTA_K_EARLY → 立即返回该候选
          4. 无early-stop → 返回排序后前5个候选

        Args:
            evaluated_set: L3评估后的候选集。

        Returns:
            最优候选列表(按优先级排序)。
        """
        candidates: List[Dict[str, Any]] = evaluated_set.candidates

        # Check for early_stop flag from L3
        for c in candidates:
            if c.get('early_stop', False):
                return [c]  # L3 already found a great candidate

        # Sort by η ascending (lowest residual first)
        sorted_cands: List[Dict[str, Any]] = sorted(
            candidates, key=lambda x: x.get('eta', 1.0),
        )

        # Early-stop: first η < threshold → return immediately
        for c in sorted_cands:
            if c.get('eta', 1.0) < self.DELTA_K_EARLY:
                return [c]

        # No early-stop → delegate to KappaEtaAscendSelector for full selection
        return self._selector.select(candidates)

    def confidence(self, eta: float) -> float:
        """计算置信度。"""
        return self._selector.confidence(eta)


# ============================================================================
# §5. 自测函数
# ============================================================================

def _self_test() -> bool:
    """HybridSearchPipeline 自测: 构造简单Grid，验证四层管线。

    Returns:
        True if all tests pass, False otherwise.
    """
    # 构造简单Grid (4×4)
    grid: np.ndarray = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=int)

    # 构造示例: (input, output)
    examples: List[Tuple[np.ndarray, np.ndarray]] = [
        (grid.copy(), grid.copy()),  # 完全匹配示例
    ]

    # 测试1: CandidateSet数据载体
    cs: CandidateSet = CandidateSet(
        node_ids=[0, 1, 2],
        node_map={0: Node(id=0, parent_id=-1, action="root", depth=0)},
        meta={'test': True},
    )
    assert len(cs) == 3, "CandidateSet len should be 3"
    assert not cs.is_empty(), "CandidateSet should not be empty"

    # 测试2: EvaluatedCandidateSet数据载体
    ecs: EvaluatedCandidateSet = EvaluatedCandidateSet(
        candidates=[
            {'node_id': 0, 'eta': 0.1, 'confidence': 0.8, 'ic': 0.5, 'depth': 2},
        ],
    )
    assert len(ecs) == 1, "EvaluatedCandidateSet len should be 1"
    assert ecs.best_eta() == 0.1, "best_eta should be 0.1"
    assert ecs.best_confidence() == 0.8, "best_confidence should be 0.8"

    # 测试3: PipelineStrategies默认配置
    ps: PipelineStrategies = PipelineStrategies()
    assert ps.l1_strategy == "bfs", "Default L1 should be bfs"
    assert ps.l2_strategy == "sym_pruner", "Default L2 should be sym_pruner"
    assert ps.l3_strategy == "gauss_ex", "Default L3 should be gauss_ex"
    assert ps.l4_strategy == "kappa_selector", "Default L4 should be kappa_selector"

    # 测试4: HybridSearchPipeline构建
    pipeline: HybridSearchPipeline = HybridSearchPipeline(ps)
    assert pipeline.l1 is not None, "L1 strategy should be built"
    assert pipeline.l2 is not None, "L2 strategy should be built"
    assert pipeline.l3 is not None, "L3 strategy should be built"
    assert pipeline.l4 is not None, "L4 strategy should be built"

    # 测试5: 空候选集处理
    empty_cs: CandidateSet = CandidateSet()
    assert empty_cs.is_empty(), "Empty CandidateSet should be empty"
    assert len(empty_cs) == 0, "Empty CandidateSet len should be 0"

    empty_ecs: EvaluatedCandidateSet = EvaluatedCandidateSet()
    assert empty_ecs.is_empty(), "Empty EvaluatedCandidateSet should be empty"
    assert empty_ecs.best_eta() is None, "best_eta of empty set should be None"

    print("[PASS] hybrid_search_engine _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
