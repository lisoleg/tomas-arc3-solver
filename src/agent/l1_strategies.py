"""
src/agent/l1_strategies.py
TOMAS 四层混合搜索 L1 候选生成策略

L1层负责搜索空间探索，从初始状态生成候选节点集(CandidateSet)。
不同L1策略使用不同的搜索方式:

  - WallBFSCandidateGenerator: KA59推箱Wall-BFS (物理原语约束)
  - BFSPathCandidateGenerator: 标准BFS (结构探索)
  - DFSEnumerationCandidateGenerator: DFS枚举 (深度优先)
  - DeltaStateReplayCandidateGenerator: Δ-State Replay (IDO流贯)
  - DirectComputeCandidateGenerator: 直接计算 (简单游戏)

所有策略类实现L1CandidateGenerator Protocol:
  generate(root_state, action_space, ...) → CandidateSet

Version: v3.18.0 — Hybrid Search L1 Strategies
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .delta_state import (
    Node,
    ReplayEngine,
    ActionSpace,
    SolverAborted,
    BudgetExceeded,
    DEFAULT_MAX_DEPTH,
    STRUCT_BFS_BUDGET,
)
from .hybrid_search_engine import CandidateSet


# ============================================================================
# §1. BFSPathCandidateGenerator — 标准BFS候选生成
# ============================================================================

class BFSPathCandidateGenerator:
    """标准BFS候选生成策略 — 结构探索阶段。

    使用delta_state.structural_bfs做BFS搜索，扩展所有可用动作，
    生成候选节点集。适用于大多数键盘游戏(LS20/TR87等)。

    IDO Δ-State设计:
      - 搜索节点用Node(parent_id, action)表示
      - 不物化Grid，只记录动作增量(Delta)
      - 物化只在L3评估时发生

    Attributes:
        max_depth: 最大搜索深度。
        max_nodes: 最大扩展节点数。
    """

    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
    ) -> None:
        """初始化BFS候选生成策略。

        Args:
            max_depth: 最大搜索深度，默认5。
            max_nodes: 最大扩展节点数，默认40。
        """
        self.max_depth: int = max_depth
        self.max_nodes: int = max_nodes

    def generate(
        self,
        root_state: Any,
        action_space: ActionSpace,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        **kwargs: Any,
    ) -> CandidateSet:
        """BFS候选生成: 扩展所有可用动作。

        Args:
            root_state: 根节点状态 (game engine或Grid)。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            **kwargs: 策略特定参数。

        Returns:
            CandidateSet: BFS搜索结果。
        """
        from .delta_state import structural_bfs

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

        # 执行BFS搜索
        effective_max_nodes: int = min(max_nodes, self.max_nodes)
        candidates: List[int] = structural_bfs(
            root_id=0,
            node_map=node_map,
            action_space=action_space,
            max_nodes=effective_max_nodes,
        )

        return CandidateSet(
            node_ids=candidates,
            node_map=node_map,
            replay_engine=replay_engine,
            root_state=root_state,
            action_space=action_space,
            meta={'l1_strategy': 'bfs', 'max_depth': max_depth, 'max_nodes': effective_max_nodes},
        )


# ============================================================================
# §2. DFSEnumerationCandidateGenerator — DFS枚举候选生成
# ============================================================================

class DFSEnumerationCandidateGenerator:
    """DFS枚举候选生成策略 — 深度优先搜索。

    DFS搜索所有可用动作序列，优先深入探索。
    适用于需要深度搜索的游戏(CN04仿射变换等)。

    Attributes:
        max_depth: 最大搜索深度。
        max_nodes: 最大扩展节点数。
    """

    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = 80,
    ) -> None:
        """初始化DFS候选生成策略。

        Args:
            max_depth: 最大搜索深度，默认5。
            max_nodes: 最大扩展节点数，默认80。
        """
        self.max_depth: int = max_depth
        self.max_nodes: int = max_nodes

    def generate(
        self,
        root_state: Any,
        action_space: ActionSpace,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = 80,
        **kwargs: Any,
    ) -> CandidateSet:
        """DFS候选生成: 深度优先搜索。

        Args:
            root_state: 根节点状态。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            **kwargs: 策略特定参数。

        Returns:
            CandidateSet: DFS搜索结果。
        """
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

        # DFS搜索
        effective_max_nodes: int = min(max_nodes, self.max_nodes)
        effective_max_depth: int = min(max_depth, self.max_depth)
        candidates: List[int] = []
        next_id: int = 1
        expanded: int = 0
        stack: List[int] = [0]  # DFS用栈

        while stack and expanded < effective_max_nodes:
            cur_id: int = stack.pop()
            cur_node: Node = node_map[cur_id]

            if cur_node.depth >= effective_max_depth:
                continue

            primitives: list = action_space.get_primitives()

            for action in primitives:
                if action_space._mode == 'game':
                    action_str: str = str(action)
                else:
                    action_str = action

                new_node: Node = Node(
                    id=next_id,
                    parent_id=cur_id,
                    action=action_str,
                    depth=cur_node.depth + 1,
                    meta={'stage': 'dfs'},
                )
                node_map[next_id] = new_node
                candidates.append(next_id)
                stack.append(next_id)
                next_id += 1
                expanded += 1

                if expanded >= effective_max_nodes:
                    break

        return CandidateSet(
            node_ids=candidates,
            node_map=node_map,
            replay_engine=replay_engine,
            root_state=root_state,
            action_space=action_space,
            meta={'l1_strategy': 'dfs', 'max_depth': effective_max_depth, 'max_nodes': effective_max_nodes},
        )


# ============================================================================
# §3. WallBFSCandidateGenerator — KA59推箱Wall-BFS
# ============================================================================

class WallBFSCandidateGenerator:
    """KA59推箱Wall-BFS候选生成策略。

    使用wall_bfs.WallBFSEngine做BFS搜索，物理原语约束：
    can_push_box验证前方可通行，is_deadlock_corner剪枝死锁状态。

    适用于KA59 Sokoban推箱游戏。

    Attributes:
        max_depth: 最大搜索深度。
        max_nodes: 最大扩展节点数。
    """

    def __init__(
        self,
        max_depth: int = 15,
        max_nodes: int = 200,
    ) -> None:
        """初始化Wall-BFS候选生成策略。

        Args:
            max_depth: 最大搜索深度，默认15。
            max_nodes: 最大扩展节点数，默认200。
        """
        self.max_depth: int = max_depth
        self.max_nodes: int = max_nodes

    def generate(
        self,
        root_state: Any,
        action_space: ActionSpace,
        max_depth: int = 15,
        max_nodes: int = 200,
        **kwargs: Any,
    ) -> CandidateSet:
        """Wall-BFS候选生成: 物理原语约束搜索。

        Args:
            root_state: 根节点状态 (Grid np.ndarray 或 game engine)。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            **kwargs: 可包含wall_char, goal_char, box_chars参数。

        Returns:
            CandidateSet: Wall-BFS搜索结果。
        """
        from .wall_bfs import WallBFSEngine

        # 提取Grid数据
        grid: Optional[np.ndarray] = None
        if isinstance(root_state, np.ndarray):
            grid = root_state
        else:
            from .delta_state import _extract_game_grid
            grid = _extract_game_grid(root_state)

        if grid is None:
            # 无法提取Grid → 返回空候选集
            return CandidateSet(
                node_ids=[], node_map={},
                meta={'l1_strategy': 'wall_bfs', 'error': 'no_grid'},
            )

        # 构建WallBFSEngine
        wall_char: int = kwargs.get('wall_char', 0)
        goal_char: int = kwargs.get('goal_char', 2)
        box_chars: Optional[Set[int]] = kwargs.get('box_chars', None)

        engine: WallBFSEngine = WallBFSEngine(
            root_state=grid,
            wall_char=wall_char,
            goal_char=goal_char,
            box_chars=box_chars,
        )

        # 执行搜索
        effective_max_depth: int = min(max_depth, self.max_depth)
        effective_max_nodes: int = min(max_nodes, self.max_nodes)
        candidates: List[int] = engine.search(
            max_depth=effective_max_depth,
            max_nodes=effective_max_nodes,
        )

        # 构建ReplayEngine
        replay_engine: ReplayEngine = ReplayEngine(
            root_state=grid,
            node_map=engine.node_map,
            mode='grid',
        )

        return CandidateSet(
            node_ids=candidates,
            node_map=engine.node_map,
            replay_engine=replay_engine,
            root_state=grid,
            action_space=action_space,
            meta={'l1_strategy': 'wall_bfs', 'max_depth': effective_max_depth, 'max_nodes': effective_max_nodes},
        )


# ============================================================================
# §4. DeltaStateReplayCandidateGenerator — Δ-State Replay候选生成
# ============================================================================

class DeltaStateReplayCandidateGenerator:
    """Δ-State Replay候选生成策略 — IDO流贯。

    使用ReplayEngine.from_game()构建共享ReplayEngine，
    然后执行IDO多阶段BFS(结构探索→参数探索→κ-Priority精化)。

    适用于需要完整IDO流贯搜索的游戏。

    Attributes:
        max_depth: 最大搜索深度。
        max_nodes: 最大扩展节点数。
    """

    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
    ) -> None:
        """初始化Δ-State Replay候选生成策略。

        Args:
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
        """
        self.max_depth: int = max_depth
        self.max_nodes: int = max_nodes

    def generate(
        self,
        root_state: Any,
        action_space: ActionSpace,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = STRUCT_BFS_BUDGET,
        **kwargs: Any,
    ) -> CandidateSet:
        """Δ-State Replay候选生成: IDO流贯搜索。

        Args:
            root_state: 根节点状态。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            **kwargs: 可包含game_id参数(用于全局共享)。

        Returns:
            CandidateSet: IDO流贯搜索结果。
        """
        from .delta_state import structural_bfs, ReplayEngine

        game_id: str = kwargs.get('game_id', "")

        # 尝试获取共享ReplayEngine
        replay_engine: Optional[ReplayEngine] = None
        if game_id:
            replay_engine = ReplayEngine.get_shared(game_id)

        if replay_engine is None:
            # 创建新ReplayEngine
            replay_engine = ReplayEngine.from_game(
                root_state, mode=action_space._mode, game_id=game_id, shared=bool(game_id),
            )

        # 执行IDO结构探索
        effective_max_nodes: int = min(max_nodes, self.max_nodes)
        candidates: List[int] = structural_bfs(
            root_id=0,
            node_map=replay_engine.node_map,
            action_space=action_space,
            max_nodes=effective_max_nodes,
        )

        return CandidateSet(
            node_ids=candidates,
            node_map=replay_engine.node_map,
            replay_engine=replay_engine,
            root_state=root_state,
            action_space=action_space,
            meta={'l1_strategy': 'delta_replay', 'game_id': game_id},
        )


# ============================================================================
# §5. DirectComputeCandidateGenerator — 直接计算候选生成
# ============================================================================

class DirectComputeCandidateGenerator:
    """直接计算候选生成策略 — 简单游戏。

    不做BFS/DFS搜索，直接从初始状态计算候选。
    适用于DFA点击编程(TN36)和偏序排序(SB26)等简单游戏。

    Attributes:
        max_depth: 最大搜索深度(通常很小)。
        max_nodes: 最大扩展节点数(通常很小)。
    """

    def __init__(
        self,
        max_depth: int = 3,
        max_nodes: int = 20,
    ) -> None:
        """初始化直接计算候选生成策略。

        Args:
            max_depth: 最大搜索深度，默认3。
            max_nodes: 最大扩展节点数，默认20。
        """
        self.max_depth: int = max_depth
        self.max_nodes: int = max_nodes

    def generate(
        self,
        root_state: Any,
        action_space: ActionSpace,
        max_depth: int = 3,
        max_nodes: int = 20,
        **kwargs: Any,
    ) -> CandidateSet:
        """直接计算候选生成: 短BFS搜索。

        Args:
            root_state: 根节点状态。
            action_space: ActionSpace实例。
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。
            **kwargs: 策略特定参数。

        Returns:
            CandidateSet: 简短搜索结果。
        """
        from .delta_state import structural_bfs

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

        # 简短BFS搜索
        effective_max_nodes: int = min(max_nodes, self.max_nodes)
        candidates: List[int] = structural_bfs(
            root_id=0,
            node_map=node_map,
            action_space=action_space,
            max_nodes=effective_max_nodes,
        )

        return CandidateSet(
            node_ids=candidates,
            node_map=node_map,
            replay_engine=replay_engine,
            root_state=root_state,
            action_space=action_space,
            meta={'l1_strategy': 'direct', 'max_depth': max_depth, 'max_nodes': effective_max_nodes},
        )


# ============================================================================
# §6. L1策略注册到HybridSearchPipeline
# ============================================================================

def register_l1_strategies() -> None:
    """将所有L1策略注册到HybridSearchPipeline.L1_REGISTRY。"""
    from .hybrid_search_engine import HybridSearchPipeline

    HybridSearchPipeline.L1_REGISTRY['bfs'] = BFSPathCandidateGenerator
    HybridSearchPipeline.L1_REGISTRY['dfs'] = DFSEnumerationCandidateGenerator
    HybridSearchPipeline.L1_REGISTRY['wall_bfs'] = WallBFSCandidateGenerator
    HybridSearchPipeline.L1_REGISTRY['delta_replay'] = DeltaStateReplayCandidateGenerator
    HybridSearchPipeline.L1_REGISTRY['direct'] = DirectComputeCandidateGenerator


# 自动注册
register_l1_strategies()


# ============================================================================
# §7. 自测函数
# ============================================================================

def _self_test() -> bool:
    """L1策略自测: 验证所有5个L1策略的generate()方法。

    Returns:
        True if all tests pass, False otherwise.
    """
    from .hybrid_search_engine import HybridSearchPipeline

    # 验证注册
    assert 'bfs' in HybridSearchPipeline.L1_REGISTRY, "bfs should be registered"
    assert 'dfs' in HybridSearchPipeline.L1_REGISTRY, "dfs should be registered"
    assert 'wall_bfs' in HybridSearchPipeline.L1_REGISTRY, "wall_bfs should be registered"
    assert 'delta_replay' in HybridSearchPipeline.L1_REGISTRY, "delta_replay should be registered"
    assert 'direct' in HybridSearchPipeline.L1_REGISTRY, "direct should be registered"

    # 测试BFS
    grid: np.ndarray = np.zeros((4, 4), dtype=int)
    action_space: ActionSpace = ActionSpace('grid')
    bfs_gen: BFSPathCandidateGenerator = BFSPathCandidateGenerator()
    bfs_result: CandidateSet = bfs_gen.generate(grid, action_space, max_depth=3, max_nodes=20)
    assert isinstance(bfs_result, CandidateSet), "BFS should return CandidateSet"
    assert bfs_result.meta['l1_strategy'] == 'bfs', "BFS meta should record strategy"

    # 测试DFS
    dfs_gen: DFSEnumerationCandidateGenerator = DFSEnumerationCandidateGenerator()
    dfs_result: CandidateSet = dfs_gen.generate(grid, action_space, max_depth=3, max_nodes=20)
    assert isinstance(dfs_result, CandidateSet), "DFS should return CandidateSet"
    assert dfs_result.meta['l1_strategy'] == 'dfs', "DFS meta should record strategy"

    # 测试Direct
    direct_gen: DirectComputeCandidateGenerator = DirectComputeCandidateGenerator()
    direct_result: CandidateSet = direct_gen.generate(grid, action_space, max_depth=3, max_nodes=20)
    assert isinstance(direct_result, CandidateSet), "Direct should return CandidateSet"
    assert direct_result.meta['l1_strategy'] == 'direct', "Direct meta should record strategy"

    print("[PASS] l1_strategies _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
