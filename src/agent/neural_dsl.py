"""TOMAS-ARC3 Neural-Inspired DSL Module (v3.16.0)

基于两篇微信文章的神经启发 ARC-3 求解架构:
  - 文章1: "循环动力学作为信息流贯的拓扑囚禁: IDO/TOMAS框架下ARC-3求解架构的重构"
    → RecurrentDSL (repeat_until_converge, fixed_point_iterate)
    → inject_interneurons (EML超图中继节点)
    → estimate_ic_with_motifs (2-cycle/3-cycle motif IC bonus)

  - 文章2: "神经启发 ARC-3 求解架构 (IDO/TOMAS)"
    → GatedDSL (forget_gate, input_gate — LSTM启发)
    → AdvancedDSL (residual_compose, attention_select — ResNet/Transformer启发)
    → neuro_inspired_kps_search (完整搜索算法)

Core design principles (IDO/TOMAS correspondence):
  1. RNN 短循环 = 拓扑囚禁孤子 (Topological Soliton) — 信息位数被限制在回路内,
     无法一步耗散, 构成短期记忆的物理基础.
  2. LSTM Forget Gate = Ψ-Cut (物理熔断) — 主动丢弃低 κ 信息.
  3. ResNet 残差连接 = 信息直通通道 (Identity Mapping) — 防止流贯在深层投影中坍缩.
  4. Transformer Attention = 基于 GaussEx 相容性的动态重加权.
  5. Hopfield Network = 能量收敛机制 (GaussEx 校验).

Weizmann 实验发现: 含 2-环/3-环的网络是最小架构 (Minimal Solver),
短循环通过"时间维度的折叠"在有限节点内创造高维信息结构.

Version: v3.16.0
TOMAS Correspondence: TOMAS Phase II → Neural-Inspired DSL (流贯拓扑囚禁)
IDO Correspondence: IDO 流贯 = 拓扑囚禁孤子 + 门控 + 残差 + 注意力
"""

from __future__ import annotations

import copy
import heapq
import math
import time as _time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================================
# §1. RecurrentDSL — 循环原语 (RNN 短循环 / 拓扑囚禁孤子)
# ============================================================================

class RecurrentDSL:
    """循环控制原语 — 对应 RNN 中的自连接与短循环.

    Weizmann 发现: 2-环/3-环是决定网络计算性能的核心要素.
    IDO 诠释: 短循环 = 拓扑囚禁孤子, 信息位数被限制在回路内无法一步耗散.

    核心洞察: "当求解器学会了像生物神经网络一样利用短循环来留住信息时,
    它就不再是在猜测答案, 而是在进行计算."

    使用场景:
      - Flood Fill 任务 (迭代扩散直到边界)
      - Growth 任务 (迭代生长直到收敛)
      - 奇偶振荡任务 (交替变换直到稳定)
      - 对象集合迭代 (3-环协同演化)
    """

    @staticmethod
    def repeat_until_converge(
        operation: Callable[[np.ndarray], np.ndarray],
        max_steps: int = 10,
        convergence_threshold: float = 0.0,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """循环原语: 反复应用操作直到输出不再变化 (达到不动点).

        对应 RNN 自连接 — 流贯在回路中形成拓扑囚禁孤子,
        信息不因单步传播而耗散, 而是被限制在回路内持续演化.

        Args:
            operation: 单步变换函数 (grid → grid)
            max_steps: 最大迭代步数 (防止无限循环)
            convergence_threshold: 收敛判定阈值 (像素变化量 ≤ 此值视为收敛)

        Returns:
            包装后的函数: grid → converged_grid
        """
        def wrapper(grid_state: np.ndarray) -> np.ndarray:
            prev = None
            curr = grid_state.copy()
            for step_i in range(max_steps):
                prev = curr.copy()
                curr = operation(curr)
                # 收敛判定: 像素变化量 ≤ threshold
                if np.sum(np.abs(curr - prev)) <= convergence_threshold:
                    break
            return curr
        return wrapper

    @staticmethod
    def fixed_point_iterate(
        obj_set: List[Dict],
        transform: Callable[[Dict, List[Dict]], Dict],
        global_condition: Callable[[List[Dict]], bool],
        max_steps: int = 20,
    ) -> List[Dict]:
        """对对象集合进行迭代变换直至全局条件满足.

        对应 3-环结构中的多节点协同演化 — 多个对象通过迭代变换
        彼此交互, 直到整个系统达到全局稳态.

        Args:
            obj_set: 对象集合 (每个对象为字典描述)
            transform: 单对象变换函数 (obj, context → new_obj)
            global_condition: 全局收敛条件 (obj_set → bool)
            max_steps: 最大迭代步数

        Returns:
            收敛后的对象集合
        """
        current_set = [obj.copy() for obj in obj_set]
        for step_i in range(max_steps):
            new_set = []
            for obj in current_set:
                new_obj = transform(obj, current_set)
                new_set.append(new_obj)
            current_set = new_set
            if global_condition(current_set):
                break
        return current_set


# ============================================================================
# §2. GatedDSL — 门控 DSL (LSTM Forget/Input Gate 启发)
# ============================================================================

class GatedDSL:
    """门控 DSL — LSTM Forget Gate / Input Gate 启发.

    LSTM Forget Gate (f_t) = Ψ-Cut (物理熔断):
      主动丢弃低 κ 信息 — 低于阈值的不重要信息被保留旧记忆,
      高于阈值才更新记忆.

    LSTM Input Gate (i_t):
      选择性注入信息 — 通过 feature_map 和 selector 筛选特征,
      再与原状态 combine, 避免全量更新导致信息过载.
    """

    @staticmethod
    def forget_gate(
        context: Callable[[Any], float],
        threshold: float = 0.05,
    ) -> Callable[[Any, Any], Any]:
        """LSTM Forget Gate 启发: 丢弃不重要信息.

        计算信息重要性 (IC 估计), 低于阈值 → 保留旧记忆 (不更新),
        高于阈值 → 用新状态替换 (信息更新).

        TOMAS 语义: Ψ-Cut 物理熔断 — 流贯中低 IC 分支被剪枝,
        高 IC 分支保留并继续演化.

        Args:
            context: IC 估计函数 (state → importance_score)
            threshold: IC 阈值 (低于此值的信息被遗忘)

        Returns:
            门控函数: (grid_state, memory_state) → updated_memory
        """
        def op(grid_state: Any, memory_state: Any) -> Any:
            importance = context(grid_state)
            if importance < threshold:
                return memory_state  # 保留旧记忆 — Ψ-Cut 剪枝
            else:
                return grid_state   # 更新记忆 — 高 IC 分支保留
        return op

    @staticmethod
    def input_gate(
        feature_map: Callable[[Any], Any],
        selector: Callable[[Any], Any],
    ) -> Callable[[Any], Any]:
        """LSTM Input Gate 启发: 选择性注入信息.

        通过 feature_map 提取特征, selector 筛选重要特征,
        再与原状态 combine — 防止全量更新导致信息过载.

        Args:
            feature_map: 特征提取函数 (state → features)
            selector: 特征筛选函数 (features → selected_features)

        Returns:
            门控函数: (grid_state) → combined_state
        """
        def op(grid_state: Any) -> Any:
            if isinstance(grid_state, np.ndarray):
                selected_features = selector(feature_map(grid_state))
                return grid_state + selected_features  # 简单 combine
            else:
                return grid_state  # 非 numpy 类型: 透传
        return op


# ============================================================================
# §3. AdvancedDSL — 残差/注意力 DSL (ResNet/Transformer 启发)
# ============================================================================

class AdvancedDSL:
    """高级 DSL — ResNet 残差连接 + Transformer Attention + RNN 循环迭代.

    ResNet 残差连接 (F(x) + x):
      信息直通通道 — 即使变换 F(x) 无效, 原始信息 x 仍能传递.
      搜索中: 先执行 op1 得到 base, 再在 base 上执行 op2 得到 residual,
      两者 combine 确保信息不因单次操作失败而丢失.

    Transformer Attention:
      基于 GaussEx 相容性的动态重加权 — 对候选操作集计算
      attention_score, 取 argmax 选择最优操作.
      搜索中: 不按固定顺序扩展分支, 而是动态选择最相关的操作.

    RNN 循环迭代 (repeat_until_converge):
      与 RecurrentDSL.repeat_until_converge 相同,
      此处提供 AdvancedDSL 语境下的别名.
    """

    @staticmethod
    def residual_compose(
        op1: Callable[[Any], Any],
        op2: Callable[[Any], Any],
    ) -> Callable[[Any], Any]:
        """ResNet 启发: 残差连接 — F(x) + x.

        先执行 op1 得到 base, 再在 base 上执行 op2 得到 residual.
        combine(base, residual) 确保即使 op2 无效, base 信息仍保留.

        IDO 语义: 信息直通通道 — 防止流贯在深层投影中因梯度消失而坍缩.

        Args:
            op1: 基础变换 (grid → base_grid)
            op2: 残差变换 (base_grid → residual_grid)

        Returns:
            组合函数: grid → base + residual
        """
        def wrapper(grid_state: Any) -> Any:
            if isinstance(grid_state, np.ndarray):
                base = op1(grid_state)
                residual = op2(base)
                return base + residual  # ResNet: F(x) + x
            else:
                # 非 numpy 类型: 顺序应用
                base = op1(grid_state)
                return op2(base)
        return wrapper

    @staticmethod
    def attention_select(
        options: List[Callable[[Any], Any]],
        query_context: Callable[[Any], float],
    ) -> Callable[[Any], Any]:
        """Transformer 启发: 基于上下文选择最优操作.

        对候选操作集计算 attention_score (基于 query_context 和 grid_state),
        取 argmax 选择最优操作 — 不按固定顺序扩展分支,
        而是动态选择最相关的操作.

        IDO 语义: 基于 GaussEx 相容性的动态重加权.

        Args:
            options: 候选操作列表 [(grid → grid), ...]
            query_context: Attention 评分函数 (grid → score)

        Returns:
            选择函数: grid → best_option(grid)
        """
        def wrapper(grid_state: Any) -> Any:
            scores = []
            for opt in options:
                try:
                    result = opt(grid_state)
                    score = query_context(result)
                    scores.append((score, opt, result))
                except Exception:
                    scores.append((-999.0, opt, None))

            if not scores:
                return grid_state

            # argmax: 选择 score 最高的操作
            best_score, best_opt, best_result = max(scores, key=lambda t: t[0])
            if best_result is not None:
                return best_result
            return grid_state
        return wrapper

    @staticmethod
    def repeat_until_converge(
        operation: Callable[[np.ndarray], np.ndarray],
        max_steps: int = 10,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """RNN 启发: 循环迭代直到收敛 (AdvancedDSL 别名).

        与 RecurrentDSL.repeat_until_converge 功能相同,
        此处提供 AdvancedDSL 语境下的便捷接口.
        """
        return RecurrentDSL.repeat_until_converge(operation, max_steps)


# ============================================================================
# §4. inject_interneurons — EML 超图中继节点注入
# ============================================================================

def inject_interneurons(
    eml_nodes: List[Dict],
    eml_edges: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """EML 超图中继节点注入 — Interneuron Analogy.

    Weizmann 证明: 添加少量稀疏连接的中间神经元 (Interneurons)
    能极大提升大网络性能. 在 ARC-3 的 EML 超图感知中,
    这对应在三角形关系边上注入抽象中继节点.

    中继节点不对应具体像素, 而是抽象概念节点:
      - mod (模长) = 0.5 → 低模长代表抽象概念
      - phase = centroid → 三角形重心位置
      - oct_part = 0 → 无色 (纯结构关系)
      - blob_id = "relay_{triangle_id}"

    TOMAS 语义: 流贯在超图结构上的中继传播 — 抽象 witness
    节点捕获对象间的关系模式, 提升信息基数 κ.

    Args:
        eml_nodes: EML 超图节点列表 (每个节点为字典)
        eml_edges: EML 超图边列表 (每条边为字典)

    Returns:
        (new_nodes, new_edges) — 包含中继节点的增强超图
    """
    new_nodes = list(eml_nodes)
    new_edges = list(eml_edges)

    # 检测三角形结构 (3-cycle motifs)
    node_ids = {n.get('blob_id', n.get('id', f'node_{i}')): n
                for i, n in enumerate(eml_nodes)}

    # 构建 adjacency 以检测三角形
    adjacency: Dict[str, Set[str]] = {}
    for edge in eml_edges:
        edge_nodes = edge.get('nodes', [])
        if len(edge_nodes) >= 2:
            n1, n2 = edge_nodes[0], edge_nodes[1]
            adjacency.setdefault(n1, set()).add(n2)
            adjacency.setdefault(n2, set()).add(n1)

    # 检测三角形: 3个节点 A→B→C→A 形成闭合回路
    triangles_found: List[Tuple[str, str, str]] = []
    visited_triangles: Set[str] = set()

    for a_id in adjacency:
        for b_id in adjacency[a_id]:
            if b_id == a_id:
                continue  # 2-cycle (自环), 另行处理
            for c_id in adjacency.get(b_id, set()):
                if c_id != a_id and c_id != b_id and a_id in adjacency.get(c_id, set()):
                    # A→B→C→A 形成三角形
                    tri_key = tuple(sorted([a_id, b_id, c_id]))
                    if tri_key not in visited_triangles:
                        visited_triangles.add(tri_key)
                        triangles_found.append((a_id, b_id, c_id))

    # 同时统计 2-cycle (自环/双向边)
    n2cycles = sum(1 for n_id in adjacency if n_id in adjacency[n_id])
    # 双向边也算 2-cycle
    for edge in eml_edges:
        edge_nodes = edge.get('nodes', [])
        if len(edge_nodes) >= 2:
            n1, n2 = edge_nodes[0], edge_nodes[1]
            if n2 in adjacency.get(n1, set()) and n1 in adjacency.get(n2, set()):
                n2cycles += 1
    n2cycles = n2cycles // 2  # 双向边被计了两次

    n3cycles = len(triangles_found)

    # 为每个三角形注入中继节点
    for tri_idx, (a_id, b_id, c_id) in enumerate(triangles_found):
        # 计算三角形重心
        a_node = node_ids.get(a_id)
        b_node = node_ids.get(b_id)
        c_node = node_ids.get(c_id)

        if a_node and b_node and c_node:
            # 从节点提取位置信息
            positions = []
            for nd in [a_node, b_node, c_node]:
                if 'centroid' in nd:
                    positions.append(nd['centroid'])
                elif 'position' in nd:
                    positions.append(nd['position'])
                elif 'x' in nd and 'y' in nd:
                    positions.append((nd['x'], nd['y']))

            if positions:
                cx = sum(p[0] for p in positions) / len(positions)
                cy = sum(p[1] for p in positions) / len(positions)
            else:
                cx, cy = 0, 0
        else:
            cx, cy = 0, 0

        # 创建中继节点 (抽象概念节点)
        relay_node = {
            'blob_id': f'relay_tri{tri_idx}',
            'mod': 0.5,            # 低模长 → 抽象概念
            'phase': (cx, cy),     # 三角形重心
            'oct_part': 0,         # 无色 → 纯结构关系
            'domain': 'MANIFEST',  # 中继节点在显式域
            'is_relay': True,      # 标记为中继节点
            'triangles_witnessed': [a_id, b_id, c_id],
        }
        new_nodes.append(relay_node)

        # 连接中继节点与实体节点 (抽象 witness 边)
        for vertex_id in [a_id, b_id, c_id]:
            new_edges.append({
                'nodes': (relay_node['blob_id'], vertex_id),
                'relation': 'abstract_witness',
                'weight': 0.3,  # 中继连接权重较低
            })

    return new_nodes, new_edges


# ============================================================================
# §5. estimate_ic_with_motifs — Motif-based IC 估计
# ============================================================================

def estimate_ic_with_motifs(
    prog_fragment: Any,
    eml_nodes: List[Dict],
    eml_edges: List[Dict],
    base_ic_func: Optional[Callable] = None,
) -> float:
    """结合 Weizmann Motif 统计改进信息基数 (IC) 估计.

    Weizmann 发现: 2-环与 3-环的数量是预测网络性能的最佳标量特征.
    IDO 诠释: 短循环 = 拓扑囚禁孤子, 提升 κ 捕获能力.

    Motif IC bonus:
      - 2-cycle (自环/双向边): 0.2 × n2cycles
      - 3-cycle (三角形): 0.3 × n3cycles
      - 这些循环模体在超图中的数量反映了结构复杂性,
        更多循环 = 更强的信息保持能力 = 更高的 κ.

    Args:
        prog_fragment: 程序片段 (用于基础 IC 估计)
        eml_nodes: EML 超图节点列表
        eml_edges: EML 超图边列表
        base_ic_func: 基础 IC 估计函数 (默认使用简单估计)

    Returns:
        改进后的 IC 估计值
    """
    # 基础 IC 估计
    if base_ic_func is not None:
        base_ic = base_ic_func(prog_fragment)
    else:
        # 简单估计: 程序长度 × 0.1 + 常数
        if isinstance(prog_fragment, (list, tuple)):
            base_ic = len(prog_fragment) * 0.1 + 0.5
        elif isinstance(prog_fragment, dict):
            base_ic = len(prog_fragment) * 0.05 + 0.3
        else:
            base_ic = 0.3

    # 统计 EML 超图中的循环模体
    adjacency: Dict[str, Set[str]] = {}
    for edge in eml_edges:
        edge_nodes = edge.get('nodes', [])
        if len(edge_nodes) >= 2:
            n1, n2 = edge_nodes[0], edge_nodes[1]
            adjacency.setdefault(n1, set()).add(n2)
            adjacency.setdefault(n2, set()).add(n1)

    # 统计 2-cycles (自环 + 双向边)
    n2cycles = 0
    for n_id in adjacency:
        if n_id in adjacency.get(n_id, set()):
            n2cycles += 1  # 自环

    # 双向边
    bidirectional_pairs: Set[str] = set()
    for n_id, neighbors in adjacency.items():
        for nb_id in neighbors:
            if n_id in adjacency.get(nb_id, set()):
                pair_key = tuple(sorted([n_id, nb_id]))
                if pair_key not in bidirectional_pairs and n_id != nb_id:
                    bidirectional_pairs.add(pair_key)
                    n2cycles += 1

    # 统计 3-cycles (三角形)
    n3cycles = 0
    visited_triangles: Set[str] = set()
    for a_id in adjacency:
        for b_id in adjacency[a_id]:
            for c_id in adjacency.get(b_id, set()):
                if c_id != a_id and c_id != b_id and a_id in adjacency.get(c_id, set()):
                    tri_key = tuple(sorted([a_id, b_id, c_id]))
                    if tri_key not in visited_triangles:
                        visited_triangles.add(tri_key)
                        n3cycles += 1

    # Weizmann 权重: 循环对性能的贡献远大于单纯连接数
    motif_bonus = 0.2 * n2cycles + 0.3 * n3cycles

    return base_ic + motif_bonus


# ============================================================================
# §6. neuro_inspired_kps_search — 神经启发 κ-优先搜索
# ============================================================================

def neuro_inspired_kps_search(
    game: Any,
    max_depth: int = 40,
    max_nodes: int = 300000,
    max_time: float = 15.0,
    primitives: Optional[List[Any]] = None,
    eml_nodes: Optional[List[Dict]] = None,
    eml_edges: Optional[List[Dict]] = None,
    forget_threshold: float = 0.05,
    energy_threshold: float = 0.167,  # 1/6 ≈ 5/6 饱和
    ic_metric: Any = None,
    gex_constraint: Any = None,
) -> list[tuple] | None:
    """神经启发 κ-优先搜索 — 融合 LSTM/ResNet/Transformer/Hopfield 的搜索算法.

    Pipeline:
      PriorityQueue(IC优先) → forget_gate剪枝 → residual_compose扩展
      → attention_score优先级 → energy收敛检查

    搜索步骤对照 (对 BFS 的改进):
      | 步骤 | 启发来源 | 作用 | 对BFS改进 |
      | PriorityQueue+IC | κ-Snap/IDO | 高κ优先 | 替代盲序BFS |
      | forget_gate检查 | LSTM | 低IC剪枝 | 避免低信息分支浪费算力 |
      | residual_compose | ResNet | 叠加残差 | 保证信息不因单次失败丢失 |
      | attention_score | Transformer | 动态加权 | 替代静态深度优先 |
      | energy收敛 | Hopfield | 能量函数 | 替代单纯输出匹配 |

    Args:
        game: 游戏对象 (不修改, 使用 deepcopy)
        max_depth: 最大搜索深度
        max_nodes: 最大节点数
        max_time: 时间限制 (秒)
        primitives: DSL原语列表 (默认用游戏valid actions)
        eml_nodes: EML超图节点 (用于motif IC估计)
        eml_edges: EML超图边 (用于motif IC估计)
        forget_threshold: Forget Gate IC阈值 (低于此值剪枝)
        energy_threshold: Hopfield能量收敛阈值 (默认1/6)
        ic_metric: 可选ICMetric实例 (用于octonion IC)
        gex_constraint: 可选PhysicalGaussExConstraint (用于GEX)

    Returns:
        动作计划列表 [(GameAction, data), ...] 或 None
    """
    import heapq
    from arcengine import ActionInput

    t0 = _time.time()
    original_level = game._current_level_index
    total_nodes = 0
    counter = 0

    # EML 超图数据 (用于 motif IC 估计)
    _eml_nodes = eml_nodes or []
    _eml_edges = eml_edges or []

    # 构建forget gate
    forget_gate = GatedDSL.forget_gate(
        context=_estimate_ic_simple,
        threshold=forget_threshold,
    )

    # 获取游戏valid actions作为primitives
    if primitives is None:
        try:
            primitives = game._get_valid_action_inputs()
        except (AttributeError, Exception):
            primitives = []
            for aid in [1, 2, 3, 4, 6, 7]:
                try:
                    primitives.append(ActionInput(id=aid, data={}))
                except Exception:
                    pass

    # 初始优先级队列
    initial_hash = _game_state_hash_neuro(game)
    pq: list[tuple[float, int, Any, list[tuple], str]] = []
    heapq.heappush(pq, (0.0, counter, copy.deepcopy(game), [], initial_hash))
    visited: set[str] = {initial_hash}

    best_plan: list[tuple] | None = None
    best_energy: float = float('inf')

    while pq:
        if _time.time() - t0 > max_time:
            break
        if total_nodes > max_nodes:
            break

        neg_priority, cnt, g, path, prev_hash = heapq.heappop(pq)
        total_nodes += 1
        current_depth = len(path)

        # ── Forget Gate 检查 (LSTM 启发) ──
        # 低 IC 分支 → Ψ-Cut 剪枝, 不扩展
        ic_est = _estimate_ic_simple(g)
        if forget_gate(ic_est, ic_est) == ic_est and ic_est < forget_threshold:
            continue  # 保留旧记忆 → 剪枝此分支

        # ── Hopfield 能量收敛检查 ──
        # energy(child_state) < threshold → 找到解
        current_energy = _compute_energy(g, original_level)
        if current_energy < energy_threshold:
            # 能量收敛 → 可能找到解
            if g._current_level_index > original_level:
                return path  # 确实通关
            # 否则继续搜索, 但记录最佳结果
            if current_energy < best_energy:
                best_energy = current_energy
                best_plan = path

        # ── 获取 valid actions ──
        actions = []
        try:
            actions = g._get_valid_action_inputs()
        except (AttributeError, Exception):
            actions = primitives

        for ai in actions:
            if current_depth + 1 > max_depth:
                break

            try:
                test_sim = copy.deepcopy(g)
                test_sim.perform_action(ai)
            except Exception:
                continue

            # 检查通关
            if test_sim._current_level_index > original_level:
                return path + [(ai.id, ai.data if ai.data else None)]

            new_hash = _game_state_hash_neuro(test_sim)
            if new_hash in visited:
                continue
            visited.add(new_hash)

            # ── Attention Score 优先级 (Transformer 启发) ──
            # 基于 motif IC 估计动态计算优先级
            motif_ic = estimate_ic_with_motifs(
                path + [(ai.id, ai.data)],
                _eml_nodes, _eml_edges,
            )

            # 残差组合效果评估 (ResNet启发)
            residual_effect = _compute_residual_effect(test_sim, g)

            # 综合优先级: motif IC + residual effect - depth penalty
            # heapq 是 min-heap → 取负值使得高优先级先出
            attention_score = -(motif_ic + residual_effect - current_depth * 0.05)

            counter += 1
            heapq.heappush(pq, (
                attention_score, counter,
                test_sim,
                path + [(ai.id, ai.data if ai.data else None)],
                new_hash,
            ))

    return best_plan


# ============================================================================
# §7. 辅助函数 — 神经搜索内部使用
# ============================================================================

def _estimate_ic_simple(game: Any) -> float:
    """简单 IC 估计 — 用于 forget gate 和 attention score.

    基于 game state 的结构复杂度:
      - 有效动作数 × 0.02
      - 网格非零像素比例 × 0.5
      + 基数 0.3
    """
    try:
        n_actions = 0
        try:
            n_actions = len(game._get_valid_action_inputs())
        except Exception:
            n_actions = 4

        # 网格信息量 (如果有)
        grid_density = 0.5
        try:
            if hasattr(game, '_get_game_grid'):
                grid = game._get_game_grid()
                if grid is not None:
                    grid_density = np.count_nonzero(grid) / max(grid.size, 1)
        except Exception:
            pass

        return n_actions * 0.02 + grid_density * 0.5 + 0.3
    except Exception:
        return 0.3


def _compute_energy(game: Any, original_level: int) -> float:
    """Hopfield 能量函数 — 用于收敛检查.

    能量 = GEX 残差 (与目标状态的差异度).
    能量低于 threshold → 接近目标 → 可能找到解.
    """
    try:
        # 如果已通关 → 能量=0
        if game._current_level_index > original_level:
            return 0.0

        # 基于网格差异估算能量
        energy = 1.0  # 默认高能量 (远离目标)
        try:
            if hasattr(game, '_get_game_grid'):
                grid = game._get_game_grid()
                if grid is not None:
                    # 网格稀疏度 = 非零元素比例 (越接近目标越低)
                    density = np.count_nonzero(grid) / max(grid.size, 1)
                    energy = abs(density - 0.5) * 2  # 偏离50%密度 → 高能量
        except Exception:
            pass

        return energy
    except Exception:
        return 1.0


def _compute_residual_effect(new_game: Any, old_game: Any) -> float:
    """残差效果评估 — 用于 attention score.

    计算 new_game vs old_game 的变化量 (残差大小).
    大残差 = 显著变化 = 高优先级 (值得深入探索).
    小残差 = 微小变化 = 低优先级 (可能无效操作).
    """
    try:
        # 位置变化 (游戏模式)
        effect = 0.0
        try:
            if hasattr(new_game, '_current_level_index') and hasattr(old_game, '_current_level_index'):
                if new_game._current_level_index > old_game._current_level_index:
                    return 1.0  # 通关 → 最高残差效果

            # 网格变化量
            new_grid = None
            old_grid = None
            try:
                if hasattr(new_game, '_get_game_grid'):
                    new_grid = new_game._get_game_grid()
                if hasattr(old_game, '_get_game_grid'):
                    old_grid = old_game._get_game_grid()
            except Exception:
                pass

            if new_grid is not None and old_grid is not None:
                diff = np.sum(np.abs(new_grid - old_grid))
                total = max(new_grid.size, 1)
                effect = diff / total * 0.5
            else:
                # 无网格 → 用位置变化估算
                effect = 0.1  # 默认小残差
        except Exception:
            effect = 0.1

        return effect
    except Exception:
        return 0.1


def _game_state_hash_neuro(game: Any) -> str:
    """游戏状态哈希 — 用于 visited 去重.

    基于游戏关键属性生成唯一哈希,
    防止搜索中重复访问相同状态.
    """
    try:
        parts = []
        parts.append(str(game._current_level_index))

        # 网格哈希 (如果有)
        try:
            if hasattr(game, '_get_game_grid'):
                grid = game._get_game_grid()
                if grid is not None:
                    parts.append(str(hash(grid.tobytes())))
        except Exception:
            pass

        # 玩家位置 (如果有)
        try:
            if hasattr(game, 'player') and game.player is not None:
                parts.append(f"{game.player.x},{game.player.y}")
        except Exception:
            pass

        # 游戏状态哈希 (如果有)
        try:
            if hasattr(game, '_state_hash'):
                parts.append(str(game._state_hash()))
        except Exception:
            pass

        return '|'.join(parts) if parts else str(hash(id(game)))
    except Exception:
        return str(hash(id(game)))


# ============================================================================
# §8. neurally_gated_expand — 门控扩展函数 (for κ-PS integration)
# ============================================================================

def neurally_gated_expand(
    node_state: Any,
    primitives: List[Any],
    forget_threshold: float = 0.05,
    eml_nodes: Optional[List[Dict]] = None,
    eml_edges: Optional[List[Dict]] = None,
) -> List[Tuple[Any, float]]:
    """门控扩展: forget gate + motif IC → 筛选扩展候选.

    用于 κ-PS Phase 2.5 的扩展步骤:
      1. 计算每个 primitive 的 IC 估计
      2. Forget Gate: 丢弃 IC < threshold 的候选
      3. Motif IC bonus: 2-cycle/3-cycle 增加优先级
      4. 返回筛选后的候选列表 (带 attention score)

    Args:
        node_state: 当前搜索节点状态
        primitives: DSL 原语列表
        forget_threshold: Forget Gate 阈值
        eml_nodes: EML 超图节点
        eml_edges: EML 超图边

    Returns:
        [(expanded_state, attention_score), ...] — 筛选后的候选列表
    """
    candidates: List[Tuple[Any, float]] = []

    for prim in primitives:
        try:
            # 尝试应用 primitive
            test_state = copy.deepcopy(node_state)
            if hasattr(prim, 'id'):
                from arcengine import ActionInput
                ai = ActionInput(id=prim.id, data=prim.data if hasattr(prim, 'data') and prim.data else {})
                test_state.perform_action(ai)
            else:
                test_state = prim(node_state)  # Grid-mode primitive
        except Exception:
            continue

        # Forget Gate: IC < threshold → 丢弃
        ic = _estimate_ic_simple(test_state)
        if ic < forget_threshold:
            continue

        # Motif IC bonus
        motif_ic = estimate_ic_with_motifs(
            [prim], eml_nodes or [], eml_edges or [],
        )

        # Residual effect
        residual = _compute_residual_effect(test_state, node_state)

        # Attention score = motif IC + residual
        attention_score = motif_ic + residual
        candidates.append((test_state, attention_score))

    return candidates


# ============================================================================
# §9. Pipeline 集成点 — 与 game_solvers.py 的衔接
# ============================================================================

# neuro_inspired_kps_search 在 solve_game() pipeline 中的集成:
#   Phase 2.5 (κ-PS) → Phase 2.6 (Neural κ-PS, v3.16.0 NEW)
#   在 κ-PS 之后再尝试一次神经启发搜索,
#   使用 forget gate + residual compose + attention select + energy convergence
#   作为 κ-PS 的补充/替代 (更高 IC 分支优先, 低 IC 分支剪枝)

# neurally_gated_expand 用于 κ-PS 的扩展步骤:
#   在现有 solve_kappa_priority_search() 中,
#   扩展候选时使用 neurally_gated_expand() 筛选,
#   替代原始的 all-actions-expand.
