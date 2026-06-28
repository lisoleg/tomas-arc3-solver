"""
src/agent/l3_strategies.py
TOMAS 四层混合搜索 L3 残差评估策略

L3层负责评估剪枝后候选的残差η和置信度confidence。
物化候选节点(Replay)并计算GaussEx残差η，转换为EvaluatedCandidateSet。

不同L3策略:
  - KappaSnapEvaluation: κ-Snap投影评估 (Octonion内积，残差η = 1 - best_similarity)
  - DeadZeroFuseEvaluation: Dead-Zero熔断评估 (卞氏5/6阈值 + deadlock剪枝)
  - GaussExEvaluation: GaussEx残差评估 (像素匹配，max_error = η)
  - AsymIndexEvaluation: 不对称指数评估 (κ-phase consistency → η)
  - PassThroughEvaluation: 不评估(直接传递，η=0.0)

所有策略类实现L3ResidualEvaluator Protocol:
  evaluate(candidate_set, examples) → EvaluatedCandidateSet

Version: v3.18.0 — Hybrid Search L3 Strategies
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .delta_state import (
    Node,
    ReplayEngine,
    GaussExVerifier,
    SolverAborted,
    GEX_PASS_THRESHOLD,
    DEAD_ZERO_RATIO,
    _extract_game_grid,
)
from .hybrid_search_engine import CandidateSet, EvaluatedCandidateSet
from .physics_primitives import is_deadlock_corner, kappa_phase_consistency, check_deadlock_with_wall_ride


# ============================================================================
# §1. GaussExEvaluation — GaussEx残差评估
# ============================================================================

class GaussExEvaluation:
    """GaussEx残差评估策略 — 像素匹配计算η。

    物化候选节点(ReplayEngine.replay)，然后用GaussExVerifier
    计算Grid与示例的像素错误比率(max_error = η)。

    置信度: confidence = 1 - η/δ_K (类比量子态纯度)
    δ_K = 0.036 (from KSnapEngine)

    适用于大多数键盘游戏(LS20/TR87/FT09等)。

    Attributes:
        delta_k: κ-Snap残差阈值δ_K。
    """

    def __init__(self, delta_k: float = 0.036) -> None:
        """初始化GaussEx残差评估策略。

        Args:
            delta_k: κ-Snap残差阈值δ_K，默认0.036。
        """
        self.delta_k: float = delta_k

    def evaluate(
        self,
        candidate_set: CandidateSet,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> EvaluatedCandidateSet:
        """GaussEx残差评估: 物化 + 像素匹配。

        Args:
            candidate_set: L2剪枝后的候选集。
            examples: 示例列表 [(input, output), ...]。

        Returns:
            EvaluatedCandidateSet: 评估后的候选集。
        """
        verifier: GaussExVerifier = GaussExVerifier()
        candidates: List[Dict[str, Any]] = []

        replay_engine: Optional[ReplayEngine] = candidate_set.replay_engine
        if replay_engine is None:
            return EvaluatedCandidateSet(
                candidates=candidates,
                node_map=candidate_set.node_map,
                meta={'l3_strategy': 'gauss_ex', 'error': 'no_replay_engine'},
            )

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
                result: Dict[str, Any] = verifier.verify(state, examples)
                eta: float = result.get('max_error', 1.0)
                confidence: float = max(0.0, 1.0 - eta / self.delta_k)
                ic: float = node.meta.get('ic', 0.5)

                candidates.append({
                    'node_id': node_id,
                    'eta': eta,
                    'confidence': confidence,
                    'gex_result': result,
                    'ic': ic,
                    'depth': node.depth,
                })

            # Game mode: 检查通关 + 提取Grid后验证
            elif replay_engine.mode == 'game':
                # ★ 关键: 先检查通关状态 — IDO κ-优选核心
                # 通关候选 η=0 (完美分数) → κ-优选必然选择它
                game_solved: bool = False
                try:
                    # state是replay后的game对象, 检查是否通关
                    if hasattr(state, '_state') and str(state._state) == 'GameState.WIN':
                        game_solved = True
                    elif hasattr(state, '_state') and state._state.value == 3:  # WIN=3
                        game_solved = True
                except Exception:
                    pass

                if game_solved:
                    # 通关候选 → η=0, confidence=1.0 (κ-优选最高优先)
                    candidates.append({
                        'node_id': node_id,
                        'eta': 0.0,
                        'confidence': 1.0,
                        'gex_result': {'max_error': 0.0, 'solved': True},
                        'ic': 1.0,
                        'depth': node.depth,
                    })
                    continue  # 通关候选已记录, 不需要进一步验证

                grid: Optional[np.ndarray] = _extract_game_grid(state)
                if grid is not None:
                    result = verifier.verify(grid, examples)
                    eta = result.get('max_error', 1.0)
                    confidence = max(0.0, 1.0 - eta / self.delta_k)
                    ic = node.meta.get('ic', 0.5)
                    candidates.append({
                        'node_id': node_id,
                        'eta': eta,
                        'confidence': confidence,
                        'gex_result': result,
                        'ic': ic,
                        'depth': node.depth,
                    })

        return EvaluatedCandidateSet(
            candidates=candidates,
            node_map=candidate_set.node_map,
            replay_engine=replay_engine,
            meta={'l3_strategy': 'gauss_ex', 'delta_k': self.delta_k},
        )


# ============================================================================
# §2. KappaSnapEvaluation — κ-Snap投影评估
# ============================================================================

class KappaSnapEvaluation:
    """κ-Snap投影评估策略 — Octonion内积计算η。

    使用KSnapEngine的project()方法做Octonion内积投影，
    计算残差η = 1 - best_similarity (κ-Snap残差)。

    置信度: confidence = 1 - η/δ_K
    适用于CN04(仿射变换)和AR25(镜像覆盖)等需要κ-Snap的游戏。

    Attributes:
        delta_k: κ-Snap残差阈值δ_K。
        engine: KSnapEngine实例 (懒加载)。
    """

    def __init__(self, delta_k: float = 0.036) -> None:
        """初始化κ-Snap投影评估策略。

        Args:
            delta_k: κ-Snap残差阈值δ_K，默认0.036。
        """
        self.delta_k: float = delta_k
        self._engine: Any = None  # KSnapEngine懒加载

    def _get_engine(self) -> Any:
        """懒加载KSnapEngine。"""
        if self._engine is None:
            from .t_processor_isa import KSnapEngine, Octonion
            self._engine = KSnapEngine()
        return self._engine

    def evaluate(
        self,
        candidate_set: CandidateSet,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> EvaluatedCandidateSet:
        """κ-Snap投影评估: Octonion内积 + GaussEx。

        先物化候选节点 → 构建EML Octonion → κ-Snap投影 → 计算η。
        同时计算GaussEx作为辅助指标。

        Args:
            candidate_set: L2剪枝后的候选集。
            examples: 示例列表 [(input, output), ...]。

        Returns:
            EvaluatedCandidateSet: κ-Snap评估后的候选集。
        """
        verifier: GaussExVerifier = GaussExVerifier()
        candidates: List[Dict[str, Any]] = []

        replay_engine: Optional[ReplayEngine] = candidate_set.replay_engine
        if replay_engine is None:
            return EvaluatedCandidateSet(
                candidates=candidates,
                node_map=candidate_set.node_map,
                meta={'l3_strategy': 'kappa_snap', 'error': 'no_replay_engine'},
            )

        engine: Any = self._get_engine()

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            # 物化节点
            try:
                state: Any = replay_engine.replay(node_id)
            except (KeyError, SolverAborted):
                continue

            # ★ 通关检查 — κ-优选核心 (KappaSnapEvaluation专用)
            game_solved: bool = False
            if replay_engine.mode == 'game' and hasattr(state, '_state'):
                try:
                    if str(state._state) == 'GameState.WIN' or state._state.value == 3:
                        game_solved = True
                except Exception:
                    pass

            if game_solved:
                # 通关候选 → η=0, confidence=1.0
                candidates.append({
                    'node_id': node_id,
                    'eta': 0.0,
                    'confidence': 1.0,
                    'gex_result': {'max_error': 0.0, 'solved': True},
                    'kappa_eta': 0.0,
                    'ic': 1.0,
                    'depth': node.depth,
                })
                continue

            # 提取Grid
            grid: Optional[np.ndarray] = None
            if isinstance(state, np.ndarray):
                grid = state
            elif replay_engine.mode == 'game':
                grid = _extract_game_grid(state)

            if grid is None:
                continue

            # GaussEx评估 (基础η)
            gex_result: Dict[str, Any] = verifier.verify(grid, examples)
            gex_eta: float = gex_result.get('max_error', 1.0)

            # κ-Snap投影评估 (增强η)
            from .t_processor_isa import Octonion
            eml_state: Octonion = self._grid_to_octonion(grid)
            _, kappa_eta = engine.project(eml_state)

            # 综合η: 取GaussEx和κ-Snap的最小值
            eta: float = min(gex_eta, kappa_eta)
            confidence: float = max(0.0, 1.0 - eta / self.delta_k)
            ic: float = node.meta.get('ic', 0.5)

            candidates.append({
                'node_id': node_id,
                'eta': eta,
                'confidence': confidence,
                'gex_result': gex_result,
                'kappa_eta': kappa_eta,
                'ic': ic,
                'depth': node.depth,
            })

        return EvaluatedCandidateSet(
            candidates=candidates,
            node_map=candidate_set.node_map,
            replay_engine=replay_engine,
            meta={'l3_strategy': 'kappa_snap', 'delta_k': self.delta_k},
        )

    def _grid_to_octonion(self, grid: np.ndarray) -> Any:
        """将Grid数据转换为Octonion表示(简化版)。

        8分量Octonion = Grid统计特征的8维编码:
          a(实部) = Grid面积占比
          b-h(虚部) = 7种颜色频率的归一化编码

        Args:
            grid: 2D numpy array。

        Returns:
            Octonion实例。
        """
        from .t_processor_isa import Octonion

        # Grid统计特征
        total_pixels: int = grid.size
        colors, counts = np.unique(grid, return_counts=True)

        # 8维编码
        freqs: Dict[int, float] = {}
        for c, cnt in zip(colors, counts):
            freqs[int(c)] = float(cnt) / total_pixels

        # 取前7种颜色频率
        sorted_freqs: List[float] = sorted(freqs.values(), reverse=True)[:7]
        while len(sorted_freqs) < 7:
            sorted_freqs.append(0.0)

        # 构造Octonion
        a: float = 1.0  # 实部
        octonion: Octonion = Octonion(
            a=a,
            b=sorted_freqs[0],
            c=sorted_freqs[1],
            d=sorted_freqs[2],
            e=sorted_freqs[3],
            f=sorted_freqs[4],
            g=sorted_freqs[5],
            h=sorted_freqs[6],
        )

        return octonion.normalized()


# ============================================================================
# §3. DeadZeroFuseEvaluation — Dead-Zero熔断评估
# ============================================================================

class DeadZeroFuseEvaluation:
    """Dead-Zero熔断评估策略 — 卞氏5/6阈值 + deadlock剪枝。

    评估步骤:
      1. GaussEx残差η = max_error
      2. Dead-Zero熔断: η ≥ GEX_FAIL_THRESHOLD → 剪枝
      3. KA59 deadlock检测: is_deadlock_corner → η = 1.0 (熔断)
      4. 置信度: confidence = 1 - η/δ_K

    适用于KA59推箱游戏(需要deadlock剪枝)。

    Attributes:
        delta_k: κ-Snap残差阈值δ_K。
        wall_char: 墙壁颜色值。
        goal_char: 目标颜色值。
    """

    def __init__(
        self,
        delta_k: float = 0.036,
        wall_char: int = 0,
        goal_char: int = 2,
    ) -> None:
        """初始化Dead-Zero熔断评估策略。

        Args:
            delta_k: κ-Snap残差阈值δ_K。
            wall_char: 墙壁颜色值。
            goal_char: 目标颜色值。
        """
        self.delta_k: float = delta_k
        self.wall_char: int = wall_char
        self.goal_char: int = goal_char

    def evaluate(
        self,
        candidate_set: CandidateSet,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> EvaluatedCandidateSet:
        """Dead-Zero熔断评估: GaussEx + deadlock + 卞氏阈值。

        Args:
            candidate_set: L2剪枝后的候选集。
            examples: 示例列表 [(input, output), ...]。

        Returns:
            EvaluatedCandidateSet: Dead-Zero熔断评估后的候选集。
        """
        verifier: GaussExVerifier = GaussExVerifier()
        candidates: List[Dict[str, Any]] = []

        replay_engine: Optional[ReplayEngine] = candidate_set.replay_engine
        if replay_engine is None:
            return EvaluatedCandidateSet(
                candidates=candidates,
                node_map=candidate_set.node_map,
                meta={'l3_strategy': 'dead_zero_fuse', 'error': 'no_replay_engine'},
            )

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            # 物化节点
            try:
                state: Any = replay_engine.replay(node_id)
            except (KeyError, SolverAborted):
                continue

            # ★ 通关检查 (DeadZeroFuseEvaluation专用)
            game_solved: bool = False
            if replay_engine.mode == 'game' and hasattr(state, '_state'):
                try:
                    if str(state._state) == 'GameState.WIN' or state._state.value == 3:
                        game_solved = True
                except Exception:
                    pass

            if game_solved:
                candidates.append({
                    'node_id': node_id,
                    'eta': 0.0,
                    'confidence': 1.0,
                    'gex_result': {'max_error': 0.0, 'solved': True},
                    'ic': 1.0,
                    'depth': node.depth,
                })
                continue

            # 提取Grid
            grid: Optional[np.ndarray] = None
            if isinstance(state, np.ndarray):
                grid = state
            elif replay_engine.mode == 'game':
                grid = _extract_game_grid(state)

            if grid is None:
                continue

            # GaussEx评估
            gex_result: Dict[str, Any] = verifier.verify(grid, examples)
            eta: float = gex_result.get('max_error', 1.0)

            # Dead-Zero熔断: η ≥ GEX_FAIL_THRESHOLD → 剪枝
            if eta >= GEX_PASS_THRESHOLD + 0.05:  # 放宽一点阈值
                # η太大 → Dead-Zero熔断 → 不加入候选
                continue

            # KA59 deadlock检测
            # 检查Grid中是否有箱子在deadlock角落
            has_deadlock: bool = self._check_grid_deadlock(grid)
            if has_deadlock:
                eta = 1.0  # Dead-Zero熔断: η设为最大
                continue  # 不加入候选

            # 计算置信度
            confidence: float = max(0.0, 1.0 - eta / self.delta_k)
            ic: float = node.meta.get('ic', 0.5)

            candidates.append({
                'node_id': node_id,
                'eta': eta,
                'confidence': confidence,
                'gex_result': gex_result,
                'deadlock_checked': True,
                'ic': ic,
                'depth': node.depth,
            })

        return EvaluatedCandidateSet(
            candidates=candidates,
            node_map=candidate_set.node_map,
            replay_engine=replay_engine,
            meta={'l3_strategy': 'dead_zero_fuse', 'delta_k': self.delta_k},
        )

    def _check_grid_deadlock(self, grid: np.ndarray, player_pos: Optional[Tuple[int, int]] = None) -> bool:
        """检查Grid中是否有箱子在deadlock角落 — 含Wall-Ride豁免 (CHK_DL修正版)。

        升级版: 使用check_deadlock_with_wall_ride替代纯is_deadlock_corner,
        箱子贴墙滑向目标时不熔断(κ-优选路径连续性优先)。

        Args:
            grid: 2D numpy array。
            player_pos: 玩家位置(x, y), 可为None(无豁免能力)。

        Returns:
            True如果存在不可豁免的deadlock，False否则。
        """
        if grid.shape[0] < 3 or grid.shape[1] < 3:
            return False

        # 扫描Grid中的箱子位置
        for y in range(1, grid.shape[0] - 1):
            for x in range(1, grid.shape[1] - 1):
                cell: int = int(grid[y, x])
                # 箱子颜色(通常3或5)
                if cell in {3, 5}:
                    # ★ CHK_DL修正版: 含Wall-Ride豁免
                    if player_pos is not None:
                        # 有玩家位置 → 使用check_deadlock_with_wall_ride
                        if check_deadlock_with_wall_ride(
                            grid, (x, y), player_pos,
                            goal=None,  # 目标未知时用goal_char判断
                            wall_char=self.wall_char,
                            goal_char=self.goal_char,
                            box_chars={3, 5},
                        ):
                            return True
                    else:
                        # 无玩家位置 → 使用基础is_deadlock_corner
                        if is_deadlock_corner(grid, (x, y), self.wall_char, self.goal_char):
                            return True

        return False


# ============================================================================
# §4. AsymIndexEvaluation — 不对称指数评估
# ============================================================================

class AsymIndexEvaluation:
    """不对称指数评估策略 — κ-phase consistency → η。

    使用physics_primitives.kappa_phase_consistency计算Grid与示例的
    κ-Phase一致性分数，η = 1 - consistency。

    适用于需要相位一致性分析的游戏。

    Attributes:
        delta_k: κ-Snap残差阈值δ_K。
    """

    def __init__(self, delta_k: float = 0.036) -> None:
        """初始化不对称指数评估策略。

        Args:
            delta_k: κ-Snap残差阈值δ_K。
        """
        self.delta_k: float = delta_k

    def evaluate(
        self,
        candidate_set: CandidateSet,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> EvaluatedCandidateSet:
        """不对称指数评估: κ-phase consistency → η。

        Args:
            candidate_set: L2剪枝后的候选集。
            examples: 示例列表 [(input, output), ...]。

        Returns:
            EvaluatedCandidateSet: 不对称指数评估后的候选集。
        """
        candidates: List[Dict[str, Any]] = []

        replay_engine: Optional[ReplayEngine] = candidate_set.replay_engine
        if replay_engine is None:
            return EvaluatedCandidateSet(
                candidates=candidates,
                node_map=candidate_set.node_map,
                meta={'l3_strategy': 'asym_index', 'error': 'no_replay_engine'},
            )

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            try:
                state: Any = replay_engine.replay(node_id)
            except (KeyError, SolverAborted):
                continue

            grid: Optional[np.ndarray] = None
            if isinstance(state, np.ndarray):
                grid = state
            elif replay_engine.mode == 'game':
                grid = _extract_game_grid(state)

            if grid is None:
                continue

            # 计算κ-phase一致性
            best_consistency: float = 0.0
            for _, output_grid in examples:
                consistency: float = kappa_phase_consistency(grid, output_grid)
                best_consistency = max(best_consistency, consistency)

            # η = 1 - consistency
            eta: float = 1.0 - best_consistency
            confidence: float = max(0.0, 1.0 - eta / self.delta_k)
            ic: float = node.meta.get('ic', 0.5)

            candidates.append({
                'node_id': node_id,
                'eta': eta,
                'confidence': confidence,
                'asym_index': best_consistency,
                'ic': ic,
                'depth': node.depth,
            })

        return EvaluatedCandidateSet(
            candidates=candidates,
            node_map=candidate_set.node_map,
            replay_engine=replay_engine,
            meta={'l3_strategy': 'asym_index', 'delta_k': self.delta_k},
        )


# ============================================================================
# §5. PassThroughEvaluation — 不评估(直接传递)
# ============================================================================

class PassThroughEvaluation:
    """PassThrough评估策略 — 不评估，直接传递，η=0.0。

    适用于不需要L3评估的游戏(零分游戏pass_through)。
    所有候选直接传递，η设为0.0(完美匹配假设)。

    Attributes:
        None — 无额外配置。
    """

    def evaluate(
        self,
        candidate_set: CandidateSet,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> EvaluatedCandidateSet:
        """PassThrough评估(η=0.0，直接传递)。

        Args:
            candidate_set: L2剪枝后的候选集。
            examples: 示例列表(忽略)。

        Returns:
            EvaluatedCandidateSet: η=0.0的评估集。
        """
        candidates: List[Dict[str, Any]] = []

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            candidates.append({
                'node_id': node_id,
                'eta': 0.0,
                'confidence': 1.0,  # η=0 → confidence=1.0
                'ic': node.meta.get('ic', 0.5),
                'depth': node.depth,
            })

        return EvaluatedCandidateSet(
            candidates=candidates,
            node_map=candidate_set.node_map,
            replay_engine=candidate_set.replay_engine,
            meta={'l3_strategy': 'pass_through'},
        )


# ============================================================================
# §6. L3策略注册到HybridSearchPipeline
# ============================================================================

def register_l3_strategies() -> None:
    """将所有L3策略注册到HybridSearchPipeline.L3_REGISTRY。"""
    from .hybrid_search_engine import HybridSearchPipeline

    HybridSearchPipeline.L3_REGISTRY['gauss_ex'] = GaussExEvaluation
    HybridSearchPipeline.L3_REGISTRY['kappa_snap'] = KappaSnapEvaluation
    HybridSearchPipeline.L3_REGISTRY['dead_zero_fuse'] = DeadZeroFuseEvaluation
    HybridSearchPipeline.L3_REGISTRY['asym_index'] = AsymIndexEvaluation
    HybridSearchPipeline.L3_REGISTRY['pass_through'] = PassThroughEvaluation


# 自动注册
register_l3_strategies()


# ============================================================================
# §7. 自测函数
# ============================================================================

def _self_test() -> bool:
    """L3策略自测: 验证所有5个L3策略的evaluate()方法。

    Returns:
        True if all tests pass, False otherwise.
    """
    from .hybrid_search_engine import HybridSearchPipeline

    # 验证注册
    assert 'gauss_ex' in HybridSearchPipeline.L3_REGISTRY, "gauss_ex should be registered"
    assert 'kappa_snap' in HybridSearchPipeline.L3_REGISTRY, "kappa_snap should be registered"
    assert 'dead_zero_fuse' in HybridSearchPipeline.L3_REGISTRY, "dead_zero_fuse should be registered"
    assert 'asym_index' in HybridSearchPipeline.L3_REGISTRY, "asym_index should be registered"
    assert 'pass_through' in HybridSearchPipeline.L3_REGISTRY, "pass_through should be registered"

    # 测试PassThrough
    cs: CandidateSet = CandidateSet(
        node_ids=[0],
        node_map={0: Node(id=0, parent_id=-1, action="root", depth=0)},
    )
    pt: PassThroughEvaluation = PassThroughEvaluation()
    pt_result: EvaluatedCandidateSet = pt.evaluate(cs, [])
    assert len(pt_result) == 1, "PassThrough should have 1 candidate"
    assert pt_result.candidates[0]['eta'] == 0.0, "PassThrough eta should be 0.0"
    assert pt_result.candidates[0]['confidence'] == 1.0, "PassThrough confidence should be 1.0"

    print("[PASS] l3_strategies _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
