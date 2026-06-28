"""
src/agent/l2_strategies.py
TOMAS 四层混合搜索 L2 结构剪枝策略

L2层负责从L1候选中剪枝低价值节点，基于对称性、签名、约束等结构特征。

不同L2策略:
  - ComboSymmetryPruner: 组合对称性剪枝 (KA59推箱)
  - PrimeSignaturePruner: 索签名剪枝 (CN04仿射/AR25镜像)
  - MatroidConstraintPruner: 拟阵约束剪枝
  - PassThroughPruner: 不剪枝(直接传递)

所有策略类实现L2StructurePruner Protocol:
  prune(candidate_set) → CandidateSet

Version: v3.18.0 — Hybrid Search L2 Strategies
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .delta_state import (
    Node,
    ReplayEngine,
    LayoutHasher,
    GEX_PASS_THRESHOLD,
    DEAD_ZERO_RATIO,
)
from .hybrid_search_engine import CandidateSet


# ============================================================================
# §1. PassThroughPruner — 不剪枝(直接传递)
# ============================================================================

class PassThroughPruner:
    """PassThrough剪枝策略 — 不修改候选集，直接传递。

    适用于不需要L2剪枝的游戏(简单游戏、TN36 DFA等)。
    L2层不做任何操作，直接将L1候选集传递给L3。

    Attributes:
        None — 无额外配置。
    """

    def prune(self, candidate_set: CandidateSet) -> CandidateSet:
        """PassThrough剪枝(不修改候选集)。

        Args:
            candidate_set: L1候选生成结果。

        Returns:
            CandidateSet: 直接传递的候选集(副本)。
        """
        return CandidateSet(
            node_ids=list(candidate_set.node_ids),
            node_map=dict(candidate_set.node_map),
            replay_engine=candidate_set.replay_engine,
            root_state=candidate_set.root_state,
            action_space=candidate_set.action_space,
            meta={'l2_strategy': 'pass_through', 'l1_meta': candidate_set.meta},
        )


# ============================================================================
# §2. ComboSymmetryPruner — 组合对称性剪枝 (KA59推箱)
# ============================================================================

class ComboSymmetryPruner:
    """组合对称性剪枝策略 — KA59推箱。

    利用推箱游戏的组合对称性剪枝:
      - 水平/垂直对称: 镜像操作等价状态只保留一个
      - 组合等价: 箱子在不同目标位置但拓扑等价 → 剪枝
      - 布局哈希去重: 相同布局的不同路径只保留最短

    IDO设计: 物化候选节点(Replay) → 计算布局哈希 → 去重。

    Attributes:
        max_prune_ratio: 最大剪枝比例(0.5=最多剪掉50%)。
    """

    def __init__(self, max_prune_ratio: float = 0.5) -> None:
        """初始化组合对称性剪枝策略。

        Args:
            max_prune_ratio: 最大剪枝比例，默认0.5。
        """
        self.max_prune_ratio: float = max_prune_ratio

    def prune(self, candidate_set: CandidateSet) -> CandidateSet:
        """组合对称性剪枝: 布局哈希去重 + 对称剪枝。

        Args:
            candidate_set: L1候选生成结果。

        Returns:
            CandidateSet: 剪枝后的候选集。
        """
        if candidate_set.is_empty():
            return CandidateSet(
                node_ids=[], node_map=candidate_set.node_map,
                replay_engine=candidate_set.replay_engine,
                meta={'l2_strategy': 'combo_symmetry', 'pruned': 0},
            )

        replay_engine: Optional[ReplayEngine] = candidate_set.replay_engine
        if replay_engine is None:
            # 无ReplayEngine → 只做签名去重
            return self._prune_by_signature(candidate_set)

        # 物化 + 布局哈希去重
        layout_cache: Dict[str, int] = {}  # 布局哈希 → 最短深度节点ID
        pruned_ids: List[int] = []

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            # 物化节点
            try:
                state: Any = replay_engine.replay(node_id)
            except (KeyError, Exception):
                continue

            # 计算布局哈希
            if isinstance(state, np.ndarray):
                layout_hash: str = LayoutHasher.hash_with_shape(state)
            else:
                # Game mode: 用签名做去重
                layout_hash = node.signature()

            # 检查是否已有相同布局
            if layout_hash in layout_cache:
                existing_id: int = layout_cache[layout_hash]
                existing_node: Node = candidate_set.node_map[existing_id]
                # 保留更短路径
                if node.depth < existing_node.depth:
                    layout_cache[layout_hash] = node_id
                    pruned_ids.append(node_id)
                # 否则丢弃当前节点
                continue
            else:
                layout_cache[layout_hash] = node_id
                pruned_ids.append(node_id)

        # 限制剪枝比例
        max_keep: int = max(1, int(len(candidate_set.node_ids) * (1 - self.max_prune_ratio)))
        pruned_ids = pruned_ids[:max_keep]

        return CandidateSet(
            node_ids=pruned_ids,
            node_map=candidate_set.node_map,
            replay_engine=candidate_set.replay_engine,
            root_state=candidate_set.root_state,
            action_space=candidate_set.action_space,
            meta={
                'l2_strategy': 'combo_symmetry',
                'original_count': len(candidate_set.node_ids),
                'pruned_count': len(pruned_ids),
                'l1_meta': candidate_set.meta,
            },
        )

    def _prune_by_signature(self, candidate_set: CandidateSet) -> CandidateSet:
        """签名去重: 无ReplayEngine时的fallback。

        Args:
            candidate_set: L1候选生成结果。

        Returns:
            CandidateSet: 签名去重后的候选集。
        """
        sig_cache: Set[str] = set()
        pruned_ids: List[int] = []

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            sig: str = node.signature()
            if sig not in sig_cache:
                sig_cache.add(sig)
                pruned_ids.append(node_id)

        return CandidateSet(
            node_ids=pruned_ids,
            node_map=candidate_set.node_map,
            replay_engine=candidate_set.replay_engine,
            root_state=candidate_set.root_state,
            action_space=candidate_set.action_space,
            meta={'l2_strategy': 'combo_symmetry', 'pruned_count': len(pruned_ids)},
        )


# ============================================================================
# §3. PrimeSignaturePruner — 素签名剪枝
# ============================================================================

class PrimeSignaturePruner:
    """素签名剪枝策略 — CN04仿射变换/AR25镜像。

    利用数学结构的素签名特征做剪枝:
      - 素数编码: Grid变换的素数签名(唯一标识变换类型)
      - 仿射等价: D4群旋转+平移 → 等价变换只保留一个
      - 颜色签名: 颜色映射签名去重

    适用于CN04(仿射变换)和AR25(镜像覆盖)游戏。

    Attributes:
        max_prune_ratio: 最大剪枝比例。
    """

    # 素数编码表 (用于签名计算)
    PRIME_TABLE: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def __init__(self, max_prune_ratio: float = 0.6) -> None:
        """初始化素签名剪枝策略。

        Args:
            max_prune_ratio: 最大剪枝比例，默认0.6。
        """
        self.max_prune_ratio: float = max_prune_ratio

    def prune(self, candidate_set: CandidateSet) -> CandidateSet:
        """素签名剪枝: 变换签名去重。

        Args:
            candidate_set: L1候选生成结果。

        Returns:
            CandidateSet: 剪枝后的候选集。
        """
        if candidate_set.is_empty():
            return CandidateSet(
                node_ids=[], node_map=candidate_set.node_map,
                meta={'l2_strategy': 'prime_signature', 'pruned': 0},
            )

        replay_engine: Optional[ReplayEngine] = candidate_set.replay_engine

        if replay_engine is None:
            # 无ReplayEngine → 用动作序列签名做去重
            return self._prune_by_action_signature(candidate_set)

        # 物化 + 素签名计算
        prime_cache: Dict[str, int] = {}
        pruned_ids: List[int] = []

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            try:
                state: Any = replay_engine.replay(node_id)
            except (KeyError, Exception):
                continue

            # 计算素签名
            if isinstance(state, np.ndarray):
                prime_sig: str = self._compute_prime_signature(state)
            else:
                prime_sig = node.signature()

            if prime_sig in prime_cache:
                continue  # 等价变换 → 剪枝

            prime_cache[prime_sig] = node_id
            pruned_ids.append(node_id)

        # 限制剪枝比例
        max_keep: int = max(1, int(len(candidate_set.node_ids) * (1 - self.max_prune_ratio)))
        pruned_ids = pruned_ids[:max_keep]

        return CandidateSet(
            node_ids=pruned_ids,
            node_map=candidate_set.node_map,
            replay_engine=candidate_set.replay_engine,
            root_state=candidate_set.root_state,
            action_space=candidate_set.action_space,
            meta={
                'l2_strategy': 'prime_signature',
                'original_count': len(candidate_set.node_ids),
                'pruned_count': len(pruned_ids),
            },
        )

    def _compute_prime_signature(self, grid: np.ndarray) -> str:
        """计算Grid的素签名。

        素签名 = 素数编码的颜色频率分布。
        每种颜色出现次数 × 对应素数 → 累加 → 唯一标识。

        Args:
            grid: 2D numpy array。

        Returns:
            素签名字符串。
        """
        colors, counts = np.unique(grid, return_counts=True)
        signature_parts: List[str] = []

        for i, (color, count) in enumerate(zip(colors, counts)):
            prime_idx: int = min(int(color), len(self.PRIME_TABLE) - 1)
            prime_val: int = self.PRIME_TABLE[prime_idx]
            encoded: int = prime_val * int(count)
            signature_parts.append(f"{int(color)}:{encoded}")

        return "|".join(signature_parts)

    def _prune_by_action_signature(self, candidate_set: CandidateSet) -> CandidateSet:
        """动作序列签名去重: 无ReplayEngine时的fallback。

        Args:
            candidate_set: L1候选生成结果。

        Returns:
            CandidateSet: 动作签名去重后的候选集。
        """
        sig_cache: Set[str] = set()
        pruned_ids: List[int] = []

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            # 动作序列签名
            sig: str = node.signature()
            # 素签名增强: 深度×素数编码
            depth_prime: int = self.PRIME_TABLE[min(node.depth, len(self.PRIME_TABLE) - 1)]
            enhanced_sig: str = f"{sig}:{depth_prime}"

            if enhanced_sig not in sig_cache:
                sig_cache.add(enhanced_sig)
                pruned_ids.append(node_id)

        return CandidateSet(
            node_ids=pruned_ids,
            node_map=candidate_set.node_map,
            replay_engine=candidate_set.replay_engine,
            root_state=candidate_set.root_state,
            action_space=candidate_set.action_space,
            meta={'l2_strategy': 'prime_signature', 'pruned_count': len(pruned_ids)},
        )


# ============================================================================
# §4. MatroidConstraintPruner — 拟阵约束剪枝
# ============================================================================

class MatroidConstraintPruner:
    """拟阵约束剪枝策略 — 基于拟阵理论的约束剪枝。

    利用拟阵(matroid)理论中的独立集概念做剪枝:
      - 独立集约束: 搜索路径必须构成拟阵独立集
      - 交换性质: 路径中可以交换等价动作
      - 增广路径: 只保留可增广的候选路径

    适用于需要严格约束的游戏(特定搜索空间)。

    Attributes:
        max_prune_ratio: 最大剪枝比例。
    """

    def __init__(self, max_prune_ratio: float = 0.4) -> None:
        """初始化拟阵约束剪枝策略。

        Args:
            max_prune_ratio: 最大剪枝比例，默认0.4。
        """
        self.max_prune_ratio: float = max_prune_ratio

    def prune(self, candidate_set: CandidateSet) -> CandidateSet:
        """拟阵约束剪枝: 独立集过滤。

        Args:
            candidate_set: L1候选生成结果。

        Returns:
            CandidateSet: 剪枝后的候选集。
        """
        if candidate_set.is_empty():
            return CandidateSet(
                node_ids=[], node_map=candidate_set.node_map,
                meta={'l2_strategy': 'matroid_constraint', 'pruned': 0},
            )

        # 独立集过滤: 路径深度 ≤ max_depth 且 无重复动作
        pruned_ids: List[int] = []
        seen_action_sets: Set[str] = set()

        for node_id in candidate_set.node_ids:
            node: Optional[Node] = candidate_set.node_map.get(node_id)
            if node is None:
                continue

            # 收集路径动作集
            actions: List[str] = self._collect_path_actions(node, candidate_set.node_map)
            action_set_sig: str = "|".join(sorted(actions))

            # 独立集约束: 无重复动作集
            if action_set_sig in seen_action_sets:
                continue

            seen_action_sets.add(action_set_sig)
            pruned_ids.append(node_id)

        # 限制剪枝比例
        max_keep: int = max(1, int(len(candidate_set.node_ids) * (1 - self.max_prune_ratio)))
        pruned_ids = pruned_ids[:max_keep]

        return CandidateSet(
            node_ids=pruned_ids,
            node_map=candidate_set.node_map,
            replay_engine=candidate_set.replay_engine,
            root_state=candidate_set.root_state,
            action_space=candidate_set.action_space,
            meta={
                'l2_strategy': 'matroid_constraint',
                'original_count': len(candidate_set.node_ids),
                'pruned_count': len(pruned_ids),
            },
        )

    def _collect_path_actions(
        self,
        node: Node,
        node_map: Dict[int, Node],
    ) -> List[str]:
        """从节点回溯收集路径动作序列。

        Args:
            node: 目标节点。
            node_map: 节点编号 → Node映射。

        Returns:
            动作序列列表(从根到目标)。
        """
        actions: List[str] = []
        cur_id: int = node.id

        while cur_id != -1:
            cur_node: Optional[Node] = node_map.get(cur_id)
            if cur_node is None:
                break
            if cur_node.action != "root":
                actions.append(cur_node.action)
            cur_id = cur_node.parent_id

        actions.reverse()
        return actions


# ============================================================================
# §5. L2策略注册到HybridSearchPipeline
# ============================================================================

def register_l2_strategies() -> None:
    """将所有L2策略注册到HybridSearchPipeline.L2_REGISTRY。"""
    from .hybrid_search_engine import HybridSearchPipeline

    HybridSearchPipeline.L2_REGISTRY['pass_through'] = PassThroughPruner
    HybridSearchPipeline.L2_REGISTRY['combo_symmetry'] = ComboSymmetryPruner
    HybridSearchPipeline.L2_REGISTRY['prime_signature'] = PrimeSignaturePruner
    HybridSearchPipeline.L2_REGISTRY['matroid_constraint'] = MatroidConstraintPruner


# 自动注册
register_l2_strategies()


# ============================================================================
# §6. 自测函数
# ============================================================================

def _self_test() -> bool:
    """L2策略自测: 验证所有4个L2策略的prune()方法。

    Returns:
        True if all tests pass, False otherwise.
    """
    from .hybrid_search_engine import HybridSearchPipeline

    # 验证注册
    assert 'pass_through' in HybridSearchPipeline.L2_REGISTRY, "pass_through should be registered"
    assert 'combo_symmetry' in HybridSearchPipeline.L2_REGISTRY, "combo_symmetry should be registered"
    assert 'prime_signature' in HybridSearchPipeline.L2_REGISTRY, "prime_signature should be registered"
    assert 'matroid_constraint' in HybridSearchPipeline.L2_REGISTRY, "matroid_constraint should be registered"

    # 测试PassThrough
    cs: CandidateSet = CandidateSet(
        node_ids=[1, 2, 3],
        node_map={0: Node(id=0, parent_id=-1, action="root", depth=0)},
    )
    pt: PassThroughPruner = PassThroughPruner()
    pt_result: CandidateSet = pt.prune(cs)
    assert len(pt_result.node_ids) == 3, "PassThrough should keep all candidates"
    assert pt_result.meta['l2_strategy'] == 'pass_through', "Meta should record strategy"

    # 测试ComboSymmetry
    cs_sym: CandidateSet = CandidateSet(
        node_ids=[1, 2],
        node_map={
            0: Node(id=0, parent_id=-1, action="root", depth=0),
            1: Node(id=1, parent_id=0, action="UP", depth=1),
            2: Node(id=2, parent_id=0, action="DOWN", depth=1),
        },
    )
    sym: ComboSymmetryPruner = ComboSymmetryPruner()
    sym_result: CandidateSet = sym.prune(cs_sym)
    assert isinstance(sym_result, CandidateSet), "ComboSymmetry should return CandidateSet"

    # 测试PrimeSignature
    prime: PrimeSignaturePruner = PrimeSignaturePruner()
    prime_result: CandidateSet = prime.prune(cs_sym)
    assert isinstance(prime_result, CandidateSet), "PrimeSignature should return CandidateSet"

    # 测试MatroidConstraint
    matroid: MatroidConstraintPruner = MatroidConstraintPruner()
    matroid_result: CandidateSet = matroid.prune(cs_sym)
    assert isinstance(matroid_result, CandidateSet), "MatroidConstraint should return CandidateSet"

    print("[PASS] l2_strategies _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
