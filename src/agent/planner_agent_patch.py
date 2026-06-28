"""src/agent/planner_agent_patch.py
κ-Snap排名求解接口 — 让PlannerAgent支持主动探测

κ-Phase: planner_agent_patch = κ-Snap的排名求解接口软件模拟,
让PlannerAgent支持solve_ranked()方法, 返回多个候选按η升序排列,
供SemiPrivateProber进行主动探测。

核心接口:
  solve_ranked(game, level_idx, max_candidates=3) → List[RankedSolution]
  kappa_snap_perceive_ranked(grid, delta_k=0.036) → List[RankedSolution]
  patch_planner_agent(planner_agent_class) → type (Monkey-patch)

Version: v1.0.0 — ARC-AGI主动探测+IDO/TOMAS复盘框架
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from .kappa_selector import (
    BayesianRHAESSelector,
    KappaEtaAscendSelector,
    KAPPA_DELTA_K,
    kappa_eta_ascend_sort,
    kappa_priority_select,
)

__all__ = [
    'RankedSolution',
    'solve_ranked',
    'kappa_snap_perceive_ranked',
    'patch_planner_agent',
    # v4.0 — κ-Snap排名求解常量 (与__init__.py兼容)
    'MAX_RANKED_CANDIDATES',
    'ETA_MIN_GAP',
    'PROBE_STOP_THRESHOLD',
    'PROBE_TIMEOUT',
]

# v4.0 — κ-Snap排名求解常量 (与src/agent/__init__.py兼容)
MAX_RANKED_CANDIDATES: int = 3  # 最大候选数
ETA_MIN_GAP: float = KAPPA_DELTA_K / 10.0  # η最小间隔 (避免重复候选)
PROBE_STOP_THRESHOLD: float = KAPPA_DELTA_K / 2.0  # 探测停止阈值 (η < threshold → accept)
PROBE_TIMEOUT: float = 30.0  # 探测超时时间 (秒)


# ============================================================================
# §1. 数据结构
# ============================================================================

@dataclass
class RankedSolution:
    """κ-Snap排名候选 — η升序排列的单个解候选。

    κ-Phase: RankedSolution = κ-Snap对单个候选解的η排名记录,
    包含动作序列(plan)、残差(eta)、置信度(confidence)、κ相位标记。

    Attributes:
        plan: 动作序列 (DSL变换列表 或 (action, name) 元组列表).
        eta: κ-残差η (GaussEx一致性度量, η越小→越接近目标).
        confidence: κ-优选置信度 = 1 - η/δ_K.
        kappa_phase: κ-相位标记 (标识候选的来源相位).
        liu_score: Liu-Score优先级 (可选, κ-优选排序后的值).
        bayesian_rhae_score: Bayesian-RHAE融合分数 (可选).
        node_id: κ-Snap节点ID (可选, 用于ψ-Audit追踪).
        task_id: ARC-AGI任务ID (可选).
        timestamp: 候选生成时间戳.
    """

    plan: list = field(default_factory=list)
    eta: float = 1.0
    confidence: float = 0.0
    kappa_phase: str = "kappa_snap_ranked"
    liu_score: float = 0.0
    bayesian_rhae_score: float = 0.0
    node_id: int = -1
    task_id: str = ""
    timestamp: float = 0.0

    def compute_confidence(
        self,
        delta_k: float = KAPPA_DELTA_K,
    ) -> float:
        """计算κ-优选置信度 = 1 - η/δ_K。

        κ-Phase: confidence = κ-优选置信度公式,
        η越小 → confidence越大 → 越接近目标。

        Args:
            delta_k: κ-Snap残差阈值δ_K, 默认0.036.

        Returns:
            置信度值 (0~1范围).
        """
        if delta_k <= 0:
            self.confidence = 0.0
            return 0.0
        self.confidence = max(0.0, 1.0 - self.eta / delta_k)
        return self.confidence

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式.

        Returns:
            包含所有RankedSolution字段的字典.
        """
        return {
            'plan': self.plan,
            'eta': self.eta,
            'confidence': self.confidence,
            'kappa_phase': self.kappa_phase,
            'liu_score': self.liu_score,
            'bayesian_rhae_score': self.bayesian_rhae_score,
            'node_id': self.node_id,
            'task_id': self.task_id,
            'timestamp': self.timestamp,
        }

    def to_probe_candidate(self) -> Dict[str, Any]:
        """转换为κ-Snap候选格式 (供SemiPrivateProber使用).

        Returns:
            κ-Snap候选字典 (包含eta, plan, confidence等).
        """
        return {
            'eta': self.eta,
            'plan': self.plan,
            'confidence': self.confidence,
            'liu_score': self.liu_score,
            'bayesian_rhae_score': self.bayesian_rhae_score,
            'node_id': self.node_id,
            'depth': len(self.plan),
            'ic': 0.5,  # 默认信息基数
            'kappa_phase': self.kappa_phase,
        }

    def is_high_confidence(
        self,
        min_confidence: float = 0.167,
    ) -> bool:
        """判断是否为高置信度候选 (卞氏5/6饱和阈值)。

        κ-Phase: 卞氏阈值 = κ-优选的置信度门槛,
        confidence ≥ 1/6 → 有效候选。

        Args:
            min_confidence: 最小置信度阈值, 默认1/6≈0.167.

        Returns:
            True if confidence ≥ min_confidence.
        """
        return self.confidence >= min_confidence


# ============================================================================
# §2. κ-Snap排名求解接口
# ============================================================================

def solve_ranked(
    game: Any,
    level_idx: int = 0,
    max_candidates: int = 3,
    delta_k: float = KAPPA_DELTA_K,
) -> List[RankedSolution]:
    """κ-Snap排名求解 — 返回η升序候选列表。

    κ-Phase: solve_ranked = κ-Snap的排名求解接口,
    从PlannerAgent的求解结果中提取多个候选, 按η升序排列。

    算法流程:
      1. 从game获取当前状态
      2. 生成多个候选解 (κ-Snap搜索)
      3. 用κ-优选η升序排序
      4. 返回最多max_candidates个最优候选

    Args:
        game: ARC-AGI游戏实例 (或env._game).
        level_idx: 当前关卡索引, 默认0.
        max_candidates: 最大候选数, 默认3.
        delta_k: κ-Snap残差阈值δ_K, 默认0.036.

    Returns:
        η升序排列的RankedSolution列表 (最多max_candidates个).
    """
    # κ-Snap η升序选择器
    selector: KappaEtaAscendSelector = KappaEtaAscendSelector(
        delta_k=delta_k, max_select=max_candidates,
    )

    # Bayesian-RHAE融合选择器 (备用)
    bayesian_selector: BayesianRHAESSelector = BayesianRHAESSelector(
        delta_k=delta_k, max_select=max_candidates,
    )

    # 从game提取候选 (模拟κ-Snap搜索)
    raw_candidates: List[Dict[str, Any]] = _extract_candidates_from_game(
        game, level_idx, delta_k
    )

    # κ-优选η升序排序
    ranked_candidates: List[Dict[str, Any]] = selector.select(raw_candidates)

    # 转换为RankedSolution列表
    solutions: List[RankedSolution] = []
    for idx, cand in enumerate(ranked_candidates[:max_candidates]):
        eta: float = cand.get('eta', 1.0)
        confidence: float = cand.get('confidence', 0.0)
        liu_score: float = cand.get('liu_score', 0.0)
        br_score: float = cand.get('bayesian_rhae_score', 0.0)
        plan: list = cand.get('plan', cand.get('dsl_sequence', []))
        node_id: int = cand.get('node_id', -1)

        sol: RankedSolution = RankedSolution(
            plan=plan,
            eta=eta,
            confidence=confidence,
            kappa_phase="kappa_snap_ranked",
            liu_score=liu_score,
            bayesian_rhae_score=br_score,
            node_id=node_id,
            task_id=f"level_{level_idx}",
            timestamp=time.time(),
        )
        sol.compute_confidence(delta_k)
        solutions.append(sol)

    return solutions


def kappa_snap_perceive_ranked(
    grid: np.ndarray,
    delta_k: float = KAPPA_DELTA_K,
) -> List[RankedSolution]:
    """κ-Snap感知排序 — 从网格直接提取η升序候选。

    κ-Phase: perceive_ranked = κ-Snap的感知排序接口,
    从原始网格的拓扑特征中提取多个候选变换, 按η升序排列。

    算法流程:
      1. 从grid提取拓扑特征 (颜色分布、连通分量等)
      2. 生成候选变换 (基于拓扑特征的规则候选)
      3. 计算每个候选的η (残差估计)
      4. κ-优选η升序排序
      5. 返回最多3个最优候选

    Args:
        grid: 输入网格 (numpy 2D array).
        delta_k: κ-Snap残差阈值δ_K, 默认0.036.

    Returns:
        η升序排列的RankedSolution列表.
    """
    if grid is None or not isinstance(grid, np.ndarray):
        return []

    # 提取拓扑特征
    features: Dict[str, Any] = _extract_grid_features(grid)

    # 生成候选变换
    raw_candidates: List[Dict[str, Any]] = _generate_transform_candidates(
        grid, features, delta_k
    )

    # κ-优选η升序排序
    selector: KappaEtaAscendSelector = KappaEtaAscendSelector(
        delta_k=delta_k, max_select=3,
    )
    ranked: List[Dict[str, Any]] = selector.select(raw_candidates)

    # 转换为RankedSolution
    solutions: List[RankedSolution] = []
    for cand in ranked[:3]:
        sol: RankedSolution = RankedSolution(
            plan=cand.get('plan', []),
            eta=cand.get('eta', 1.0),
            confidence=cand.get('confidence', 0.0),
            kappa_phase="kappa_snap_perceive",
            liu_score=cand.get('liu_score', 0.0),
            bayesian_rhae_score=cand.get('bayesian_rhae_score', 0.0),
            node_id=cand.get('node_id', -1),
            timestamp=time.time(),
        )
        sol.compute_confidence(delta_k)
        solutions.append(sol)

    return solutions


# ============================================================================
# §3. PlannerAgent Monkey-patch
# ============================================================================

def patch_planner_agent(
    planner_agent_class: Type[Any],
) -> Type[Any]:
    """Monkey-patch PlannerAgent — 添加solve_ranked()方法。

    κ-Phase: patch = κ-Snap对PlannerAgent的接口扩展,
    添加solve_ranked()方法让PlannerAgent支持主动探测,
    返回多个候选按η升序排列。

    Patch内容:
      - solve_ranked(game, level_idx, max_candidates=3)
        → 返回η升序RankedSolution列表
      - kappa_snap_perceive_ranked(grid)
        → 返回η升序RankedSolution列表

    使用方法:
      patched_cls = patch_planner_agent(PlannerAgent)
      agent = patched_cls(env=env)
      ranked_solutions = agent.solve_ranked(game, level_idx=0)

    Args:
        planner_agent_class: PlannerAgent类 (原始类).

    Returns:
        Patched后的PlannerAgent类 (添加了solve_ranked方法).
    """
    def solve_ranked_method(
        self: Any,
        game: Any,
        level_idx: int = 0,
        max_candidates: int = 3,
        delta_k: float = KAPPA_DELTA_K,
    ) -> List[RankedSolution]:
        """PlannerAgent.solve_ranked() — κ-Snap排名求解接口。

        κ-Phase: solve_ranked = κ-Snap对PlannerAgent的排名求解扩展,
        从PlannerAgent的求解结果中提取多个候选, 按η升序排列。

        Args:
            game: ARC-AGI游戏实例.
            level_idx: 当前关卡索引.
            max_candidates: 最大候选数.
            delta_k: κ-Snap残差阈值δ_K.

        Returns:
            η升序排列的RankedSolution列表.
        """
        return solve_ranked(game, level_idx, max_candidates, delta_k)

    def kappa_snap_perceive_ranked_method(
        self: Any,
        grid: np.ndarray,
        delta_k: float = KAPPA_DELTA_K,
    ) -> List[RankedSolution]:
        """PlannerAgent.kappa_snap_perceive_ranked() — κ-Snap感知排序。

        Args:
            grid: 输入网格.
            delta_k: κ-Snap残差阈值δ_K.

        Returns:
            η升序排列的RankedSolution列表.
        """
        return kappa_snap_perceive_ranked(grid, delta_k)

    # 添加方法到类
    planner_agent_class.solve_ranked = solve_ranked_method
    planner_agent_class.kappa_snap_perceive_ranked = kappa_snap_perceive_ranked_method

    return planner_agent_class


# ============================================================================
# §4. 内部辅助函数
# ============================================================================

def _extract_candidates_from_game(
    game: Any,
    level_idx: int,
    delta_k: float,
) -> List[Dict[str, Any]]:
    """从game实例提取κ-Snap候选。

    κ-Phase: _extract = κ-Snap从ARC-AGI游戏实例中提取候选变换,
    基于当前游戏状态的拓扑特征生成多个候选。

    Args:
        game: ARC-AGI游戏实例 (或env._game).
        level_idx: 当前关卡索引.
        delta_k: κ-Snap残差阈值δ_K.

    Returns:
        κ-Snap候选列表 (包含eta, plan, confidence等).
    """
    candidates: List[Dict[str, Any]] = []

    # 尝试从game提取网格
    grid: Optional[np.ndarray] = None
    try:
        if hasattr(game, 'current_level') and hasattr(game.current_level, 'pixels'):
            pixels: Any = game.current_level.pixels
            if isinstance(pixels, np.ndarray):
                grid = pixels
    except Exception:
        pass

    if grid is None:
        # 无网格 → 生成默认候选
        for i in range(3):
            eta: float = delta_k * (i + 1) / 3.0
            candidates.append({
                'node_id': i,
                'eta': eta,
                'ic': 0.3 + i * 0.1,
                'depth': 2 + i,
                'plan': [],
                'dsl_sequence': [],
                'confidence': max(0.0, 1.0 - eta / delta_k),
            })
        return candidates

    # 有网格 → 从拓扑特征生成候选
    features: Dict[str, Any] = _extract_grid_features(grid)
    candidates = _generate_transform_candidates(grid, features, delta_k)

    return candidates


def _extract_grid_features(
    grid: np.ndarray,
) -> Dict[str, Any]:
    """从网格提取拓扑特征 (用于候选变换生成)。

    κ-Phase: _extract_grid_features = κ-Snap perceive阶段的拓扑特征提取,
    包括颜色分布、连通分量数、形状、对称性等。

    Args:
        grid: 输入网格 (numpy 2D array).

    Returns:
        拓扑特征字典.
    """
    if grid is None or not isinstance(grid, np.ndarray):
        return {'shape': (0, 0), 'n_colors': 0}

    shape: Tuple[int, ...] = grid.shape
    unique_colors: np.ndarray = np.unique(grid)
    n_colors: int = len(unique_colors)

    # 颜色占比
    color_distribution: Dict[int, float] = {}
    total: int = grid.size
    for color in unique_colors:
        count: int = int(np.sum(grid == color))
        color_distribution[int(color)] = count / total if total > 0 else 0.0

    # 主色
    dominant: int = int(unique_colors[np.argmax([
        np.sum(grid == c) for c in unique_colors
    ])])

    # 非主色像素数 (信息量估计)
    non_dominant_count: int = total - int(np.sum(grid == dominant))

    # 对称性检查 (简版: 行/列对称)
    row_symmetric: bool = False
    col_symmetric: bool = False
    if grid.shape[0] > 1:
        row_symmetric = np.allclose(grid, grid[::-1, :], atol=0)
    if grid.shape[1] > 1:
        col_symmetric = np.allclose(grid, grid[:, ::-1], atol=0)

    # 连通分量估计 (颜色种类 - 1 = 近似)
    n_components: int = max(1, n_colors - 1)

    return {
        'shape': shape,
        'n_colors': n_colors,
        'color_distribution': color_distribution,
        'dominant_color': dominant,
        'non_dominant_count': non_dominant_count,
        'row_symmetric': row_symmetric,
        'col_symmetric': col_symmetric,
        'n_components': n_components,
    }


def _generate_transform_candidates(
    grid: np.ndarray,
    features: Dict[str, Any],
    delta_k: float,
) -> List[Dict[str, Any]]:
    """从拓扑特征生成候选变换列表。

    κ-Phase: _generate = κ-Snap从拓扑特征生成候选变换,
    基于颜色分布、对称性等特征推断可能的变换规则。

    基础变换类型:
      - color_map: 颜色映射变换
      - rotate: 旋转变换
      - flip: 翻转变换
      - scale: 缩放变换
      - fill: 填充变换

    Args:
        grid: 输入网格.
        features: 拓扑特征字典.
        delta_k: κ-Snap残差阈值δ_K.

    Returns:
        κ-Snap候选列表 (每个包含eta, plan, confidence等).
    """
    candidates: List[Dict[str, Any]] = []

    n_colors: int = features.get('n_colors', 2)
    dominant: int = features.get('dominant_color', 0)
    non_dominant: int = features.get('non_dominant_count', 0)
    row_sym: bool = features.get('row_symmetric', False)
    col_sym: bool = features.get('col_symmetric', False)

    # 基于特征生成候选变换
    transform_types: List[str] = ['color_map', 'rotate', 'flip', 'fill', 'scale']

    # 优先级: 非对称 → flip优先; 多色 → color_map优先
    if not row_sym and not col_sym:
        transform_types = ['color_map', 'flip', 'rotate', 'fill', 'scale']
    elif n_colors > 2:
        transform_types = ['color_map', 'rotate', 'flip', 'fill', 'scale']

    # 为每个变换类型生成候选
    for idx, transform in enumerate(transform_types[:3]):
        # η估计: 基于变换类型的典型残差
        eta_estimates: Dict[str, float] = {
            'color_map': delta_k * 0.1,  # 颜色映射通常η较小
            'rotate': delta_k * 0.2,  # 旋转η中等
            'flip': delta_k * 0.15,  # 翻转η较小
            'fill': delta_k * 0.3,  # 填充η较大
            'scale': delta_k * 0.4,  # 缩放η最大
        }
        eta: float = eta_estimates.get(transform, delta_k * (idx + 1) / 3.0)

        # confidence = 1 - η/δ_K
        confidence: float = max(0.0, 1.0 - eta / delta_k) if delta_k > 0 else 0.0

        # IC (信息基数): 基于非主色像素比例
        ic: float = non_dominant / grid.size if grid.size > 0 else 0.5

        candidates.append({
            'node_id': idx,
            'eta': eta,
            'plan': [transform],
            'dsl_sequence': [transform],
            'confidence': confidence,
            'ic': ic,
            'depth': 1 + idx,
            'rhae_score': 0.0,
        })

    # 额外候选: 组合变换 (depth=2+)
    if n_colors > 1:
        combo_eta: float = delta_k * 0.25
        candidates.append({
            'node_id': 3,
            'eta': combo_eta,
            'plan': ['color_map', 'rotate'],
            'dsl_sequence': ['color_map', 'rotate'],
            'confidence': max(0.0, 1.0 - combo_eta / delta_k),
            'ic': 0.6,
            'depth': 3,
            'rhae_score': 0.0,
        })

    return candidates


# ============================================================================
# §5. 自测函数
# ============================================================================

def _self_test() -> bool:
    """planner_agent_patch自测: 验证RankedSolution、solve_ranked、patch。

    Returns:
        True if all tests pass, False otherwise.
    """
    # 测试1: RankedSolution数据结构
    sol: RankedSolution = RankedSolution(
        plan=['color_map', 'rotate'],
        eta=0.01,
        confidence=0.72,
        kappa_phase="kappa_snap_ranked",
        liu_score=0.5,
        bayesian_rhae_score=0.8,
        node_id=1,
        task_id="test_001",
    )

    # compute_confidence
    computed_conf: float = sol.compute_confidence(KAPPA_DELTA_K)
    expected_conf: float = 1.0 - 0.01 / KAPPA_DELTA_K
    assert abs(computed_conf - expected_conf) < 0.01, f"confidence={computed_conf}, expected={expected_conf}"

    # is_high_confidence
    assert sol.is_high_confidence(), "η=0.01 → confidence high"

    # to_dict
    sol_dict: Dict[str, Any] = sol.to_dict()
    assert sol_dict['plan'] == ['color_map', 'rotate'], "to_dict should preserve plan"
    assert sol_dict['kappa_phase'] == "kappa_snap_ranked", "to_dict should preserve kappa_phase"

    # to_probe_candidate
    probe_cand: Dict[str, Any] = sol.to_probe_candidate()
    assert 'eta' in probe_cand, "probe_candidate should have eta"
    assert 'plan' in probe_cand, "probe_candidate should have plan"
    assert 'confidence' in probe_cand, "probe_candidate should have confidence"

    # 测试2: kappa_snap_perceive_ranked — numpy grid输入
    grid: np.ndarray = np.array([
        [1, 2, 1],
        [2, 1, 2],
        [1, 2, 1],
    ])
    ranked: List[RankedSolution] = kappa_snap_perceive_ranked(grid)
    assert isinstance(ranked, list), "perceive_ranked should return list"
    # 应有至少1个有效候选 (如果通过了卞氏阈值)
    if len(ranked) > 0:
        assert ranked[0].eta <= ranked[-1].eta if len(ranked) > 1 else True, \
            "Ranked solutions should be η ascending"
        assert ranked[0].kappa_phase == "kappa_snap_perceive", \
            "kappa_phase should be perceive"

    # 测试3: kappa_snap_perceive_ranked — None输入
    ranked_none: List[RankedSolution] = kappa_snap_perceive_ranked(None)
    assert len(ranked_none) == 0, "None grid → empty list"

    # 测试4: solve_ranked — 模拟game实例
    class MockGame:
        """模拟ARC-AGI游戏实例 (用于测试)."""
        level_index: int = 0
        current_level: Any = None

    mock_game: MockGame = MockGame()
    solutions: List[RankedSolution] = solve_ranked(
        game=mock_game,
        level_idx=0,
        max_candidates=3,
    )
    assert isinstance(solutions, list), "solve_ranked should return list"
    assert len(solutions) <= 3, "solve_ranked should return ≤3 candidates"

    # 测试5: solve_ranked — η升序验证
    if len(solutions) >= 2:
        assert solutions[0].eta <= solutions[1].eta, \
            "solve_ranked should return η ascending solutions"

    # 测试6: patch_planner_agent
    class MockPlannerAgent:
        """模拟PlannerAgent类 (用于patch测试)."""
        def __init__(self, env=None):
            self._env = env

        def choose_action(self, observation):
            return 0

    patched_cls: Type[Any] = patch_planner_agent(MockPlannerAgent)
    assert hasattr(patched_cls, 'solve_ranked'), \
        "Patched class should have solve_ranked method"
    assert hasattr(patched_cls, 'kappa_snap_perceive_ranked'), \
        "Patched class should have kappa_snap_perceive_ranked method"

    # 创建patched实例并调用solve_ranked
    agent: Any = patched_cls(env=None)
    agent_solutions: List[RankedSolution] = agent.solve_ranked(
        game=mock_game, level_idx=0, max_candidates=3
    )
    assert isinstance(agent_solutions, list), \
        "Patched agent.solve_ranked should return list"

    # 测试7: patched实例调用kappa_snap_perceive_ranked
    agent_ranked: List[RankedSolution] = agent.kappa_snap_perceive_ranked(grid)
    assert isinstance(agent_ranked, list), \
        "Patched agent.perceive_ranked should return list"

    # 测试8: RankedSolution η升序 (solve_ranked结果)
    for i in range(len(agent_solutions) - 1):
        assert agent_solutions[i].eta <= agent_solutions[i + 1].eta, \
            "solve_ranked solutions should be η ascending"

    # 测试9: _extract_grid_features
    features: Dict[str, Any] = _extract_grid_features(grid)
    assert 'shape' in features, "features should have shape"
    assert 'n_colors' in features, "features should have n_colors"
    assert 'color_distribution' in features, "features should have color_distribution"
    assert features['n_colors'] == 2, f"Grid has 2 unique colors, got {features['n_colors']}"

    # 测试10: _generate_transform_candidates
    candidates: List[Dict[str, Any]] = _generate_transform_candidates(
        grid, features, KAPPA_DELTA_K
    )
    assert len(candidates) > 0, "Should generate at least 1 candidate"
    for cand in candidates:
        assert 'eta' in cand, "Candidate should have eta"
        assert 'plan' in cand, "Candidate should have plan"

    # 测试11: RankedSolution边界 — η=0 → confidence=1
    sol_zero: RankedSolution = RankedSolution(eta=0.0)
    conf_zero: float = sol_zero.compute_confidence(KAPPA_DELTA_K)
    assert abs(conf_zero - 1.0) < 1e-6, f"η=0 → confidence=1, got {conf_zero}"

    # 测试12: RankedSolution边界 — η=δ_K → confidence=0
    sol_max: RankedSolution = RankedSolution(eta=KAPPA_DELTA_K)
    conf_max: float = sol_max.compute_confidence(KAPPA_DELTA_K)
    assert abs(conf_max - 0.0) < 1e-6, f"η=δ_K → confidence=0, got {conf_max}"

    # 测试13: RankedSolution — is_high_confidence阈值
    sol_low: RankedSolution = RankedSolution(eta=0.8)
    sol_low.compute_confidence(KAPPA_DELTA_K)
    assert not sol_low.is_high_confidence(), \
        "η=0.8 → confidence很低 → not high confidence"

    print("[PASS] planner_agent_patch _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
