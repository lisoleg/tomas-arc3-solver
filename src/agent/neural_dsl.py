"""TOMAS-ARC3 Neural-Inspired DSL Module (v3.20.0)

基于两篇微信文章的神经启发 ARC-3 求解架构:
  - 文章1: "循环动力学作为信息流贯的拓扑囚禁: IDO/TOMAS框架下ARC-3求解架构的重构"
    → RecurrentDSL (repeat_until_converge, fixed_point_iterate)
    → inject_interneurons (EML超图中继节点)
    → estimate_ic_with_motifs (2-cycle/3-cycle motif IC bonus)

  - 文章2: "神经启发 ARC-3 求解架构 (IDO/TOMAS)"
    → GatedDSL (forget_gate, input_gate — LSTM启发)
    → AdvancedDSL (residual_compose, attention_select — ResNet/Transformer启发)
    → neuro_inspired_kps_search (完整搜索算法)

  - 文章3: "从流贯拓扑囚禁到天行相变" (v3.17.0 NEW)
    → is_soliton (PTS孤子判别: 强耦合→拓扑孤子, 弱耦合→背景)
    → doga_score (DOGA评分: O秩序锚定/G正向冲量/A异化损耗)
    → YinLongDSL (阴龙运算: 八元数非结合代数核, 保留括号结构)
    → kappa_snap_search Phase 2 增强 (YinLong组合 + DOGA剪枝)

  - 文章4: "天行方程的求解理论" (v3.17.0 NEW)
    → TianxingGaussExVerifier (基于天行方程的GaussEx校验器: Xi=tanh(real(S²)))
    → solve_tianxing (FTX数值求解器: IDO/TOMAS语义)
    → yinlong_tensor_product / estimate_structural_phase / 辅助函数

  - 文章5: "从 κ‑代数到 ARC‑AGI：基于 Tsirelson 界与因果归约的抽象推理统一框架" (v3.20.0 NEW)
    → KappaTsirelsonVerifier (κ-algebra Tsirelson bound CHSH ≤ 2√2 校验)
    → tsirelson_causal_reduction (κ-陪集C(11,4)因果归约: ARC=T-minimization)
    → KappaCausalReductionDSL (κ-legal变换识别 + CHSH→DZFUSE映射)
    → confidence_from_eta (物理纯度: confidence = 1 - η/δ_K)

κ-Tsirelson Locking Theorem (v3.20.0 NEW):
    κ-algebra八元数虚单位球约束 → CHSH最大值=2√2 (Tsirelson界)
    PR-Box (S=4) → 代数非法 → Dead-Zero熔断
    ARC求解 = κ-陪集C(11,4)因果归约 + 最小GaussEx残差η
    η < δ_K → Tsirelson-legal (物理直觉正确)
    η ≥ δ_K → PR-Box非法 (伪规则剪枝)

Core design principles (IDO/TOMAS correspondence):
  1. RNN 短循环 = 拓扑囚禁孤子 (Topological Soliton) — 信息位数被限制在回路内,
     无法一步耗散, 构成短期记忆的物理基础.
  2. LSTM Forget Gate = Ψ-Cut (物理熔断) — 主动丢弃低 κ 信息.
  3. ResNet 残差连接 = 信息直通通道 (Identity Mapping) — 防止流贯在深层投影中坍缩.
  4. Transformer Attention = 基于 GaussEx 相容性的动态重加权.
  5. Hopfield Network = 能量收敛机制 (GaussEx 校验).
  6. PTS Soliton Discrimination = 流贯拓扑囚禁的实体判别 (v3.17.0 NEW).
  7. DOGA Scoring = 秩序锚定/正向冲量/异化损耗的三参数评估 (v3.17.0 NEW).
  8. YinLong Algebra = 八元数非结合代数, 保留括号结构防止信息坍缩 (v3.17.0 NEW).
  9. Tianxing Equation = 波性相干核×粒性实存核→天行相变 (v3.17.0 NEW).
  10. κ-Tsirelson Bound = κ-algebra约束CHSH≤2√2, PR-Box→DZFUSE (v3.20.0 NEW).

Weizmann 实验发现: 含 2-环/3-环的网络是最小架构 (Minimal Solver),
短循环通过"时间维度的折叠"在有限节点内创造高维信息结构.

Version: v3.20.0
TOMAS Correspondence: TOMAS Phase II → Neural-Inspired DSL (流贯拓扑囚禁 → 天行相变 → κ-Tsirelson因果归约)
IDO Correspondence: IDO 流贯 = 拓扑囚禁孤子 + 门控 + 残差 + 注意力 + 天行方程 + κ-Tsirelson界
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
# §10. PTS Soliton Discrimination — 流贯拓扑囚禁孤子判别 (v3.17.0 NEW)
# ============================================================================

def is_soliton(blob: Dict, grid_shape: Tuple[int, int], g_critical: float = 0.3) -> bool:
    """PTS model: strong coupling → topological soliton (object), weak coupling → background.

    流贯拓扑囚禁语义:
      - 强耦合 (coupling_strength > g_critical): 信息被拓扑囚禁, 形成稳定孤子 → 对象节点
      - 弱耦合 (coupling_strength ≤ g_critical): 信息未被囚禁, 随时间耗散 → Dead-Zone 背景

    139 相变阈值: 当 g_critical 设为 0.3 (约 139/466), 覆盖多数 ARC 任务中的对象判别场景.

    Args:
        blob: 字典描述的 blob/sprite, 包含:
            - 'area': blob 的像素面积
            - 'bbox_area': bounding box 面积 (用于计算凸包填充率)
            - 'inner_var': 内部颜色方差
            - 'outer_var': 外部颜色方差
        grid_shape: 网格尺寸 (rows, cols)
        g_critical: PTS 耦合强度临界值 (默认 0.3)

    Returns:
        True = Jinling Sphere node (拓扑囚禁孤子 → 对象),
        False = Dead-Zone (背景噪声)
    """
    area_ratio = blob.get('area', 0) / (grid_shape[0] * grid_shape[1])
    compactness = blob.get('area', 0) / max(blob.get('bbox_area', 1), 1)  # convex hull fill rate
    inner_var = blob.get('inner_var', 0.0)
    outer_var = blob.get('outer_var', 0.0)
    contrast = inner_var / (outer_var + 1e-6)
    coupling_strength = area_ratio * compactness * contrast
    return coupling_strength > g_critical


def is_soliton_from_sprite(sprite: Dict, grid_size: Tuple[int, int]) -> bool:
    """PTS 孤子判别 — 针对 ARC game sprite 适配.

    将 sprite 数据转换为 blob 格式后调用 is_soliton.

    Args:
        sprite: ARC sprite 字典, 包含:
            - 'area' 或 'pixel_count': 像素面积
            - 'bbox_area' 或 'width'*'height': bounding box 面积
            - 'color_var' 或 'inner_color_var': 颜色方差
            - 'outer_color_var' 或 'bg_color_var': 背景颜色方差
        grid_size: 网格尺寸 (rows, cols)

    Returns:
        True = 对象 (拓扑囚禁孤子), False = 背景
    """
    # 统一 sprite → blob 转换
    blob = {
        'area': sprite.get('area', sprite.get('pixel_count', 0)),
        'bbox_area': sprite.get('bbox_area',
                    sprite.get('width', 1) * sprite.get('height', 1)),
        'inner_var': sprite.get('inner_var',
                    sprite.get('color_var',
                    sprite.get('inner_color_var', 0.0))),
        'outer_var': sprite.get('outer_var',
                    sprite.get('outer_color_var',
                    sprite.get('bg_color_var', 0.0))),
    }
    return is_soliton(blob, grid_size)


# ============================================================================
# §11. DOGA Scoring Function — 秩序锚定/正向冲量/异化损耗 (v3.17.0 NEW)
# ============================================================================

def estimate_order_anchoring(prog: Dict, eml_graph: Dict) -> float:
    """秩序锚定 (O): 检查程序是否作用于已识别的对象.

    DOGA O 参数: 程序操作的目标是否与 EML 超图中的对象节点匹配.
    高 O = 程序有明确对象目标 (秩序锚定好).
    低 O = 程序操作散乱 (缺乏锚定).

    Args:
        prog: 程序字典, 包含 'targets_sprites' (目标 sprite ID列表)
        eml_graph: EML 超图字典, 包含 'nodes' (对象节点列表)

    Returns:
        O 值 (0~1), 高值表示秩序锚定好
    """
    targets = prog.get('targets_sprites', [])
    nodes = eml_graph.get('nodes', [])

    if not targets or not nodes:
        return 0.1  # 无目标或无节点 → 低锚定

    # 检查程序目标与 EML 超图节点重叠度
    node_ids = set()
    for n in nodes:
        nid = n.get('blob_id', n.get('id', ''))
        node_ids.add(nid)

    overlap = len(set(targets) & node_ids)
    coverage = overlap / max(len(targets), 1)

    # 对象操作权重: 目标覆盖率 × 对象密度
    obj_density = len(nodes) / max(len(eml_graph.get('edges', [])) + 1, 1)
    O = coverage * (0.5 + 0.5 * min(obj_density, 1.0))

    return min(O, 1.0)


def estimate_alienation(prog: Dict, eml_graph: Dict) -> float:
    """异化损耗 (A): 检查程序是否引入了与 EML 超图无关的实体.

    DOGA A 参数: 程序是否引入了 EML 超图中不存在的实体操作.
    高 A = 程序引入大量无关操作 (异化损耗大).
    低 A = 程序操作与已知对象一致 (异化损耗小).

    Args:
        prog: 程序字典, 包含 'actions' (动作列表)
        eml_graph: EML 超图字典, 包含 'nodes' (对象节点列表)

    Returns:
        A 值 (0~1), 低值表示异化损耗小 (好)
    """
    actions = prog.get('actions', [])
    nodes = eml_graph.get('nodes', [])

    if not actions:
        return 0.0  # 无动作 → 无异化

    # 统计动作中涉及的不在 EML 超图中的实体
    node_ids = set()
    for n in nodes:
        nid = n.get('blob_id', n.get('id', ''))
        node_ids.add(nid)

    # 动作中引用的实体
    referenced_entities = set()
    for act in actions:
        if isinstance(act, dict):
            for key in ['target', 'sprite_id', 'object_id', 'entity']:
                if key in act:
                    referenced_entities.add(str(act[key]))
        elif isinstance(act, (list, tuple)) and len(act) >= 2:
            referenced_entities.add(str(act[1]))

    # 异化比例: 不在 EML 超图中的实体占比
    alien_count = len(referenced_entities - node_ids)
    A = alien_count / max(len(referenced_entities), 1)

    return min(A, 1.0)


def doga_score(prog: Dict, eml_graph: Dict) -> float:
    """DOGA parameters: O(秩序锚定), G(正向冲量), A(异化损耗).

    DOGA 评分: G * O / (A + ε)
      - O > 0.2 且 A < 0.5 → 正常评分 (≥ 139 阈值语境下可接受)
      - O < 0.2 → -1 (秩序锚定太低, 139 threshold)
      - A > 0.5 → -1 (异化损耗太大, 139 critical)

    正向冲量 G = 1/(num_primitives + ε): 程序越简洁越好.

    Args:
        prog: 程序字典, 包含:
            - 'num_primitives': 原语数量 (或从 actions 列表推断)
            - 'targets_sprites': 目标 sprite ID列表
            - 'actions': 动作列表
        eml_graph: EML 超图字典, 包含 'nodes', 'edges'

    Returns:
        DOGA 评分 (正值为合格, -1 为不合格)
    """
    O = estimate_order_anchoring(prog, eml_graph)
    num_prims = prog.get('num_primitives',
               len(prog.get('actions', [])) if isinstance(prog.get('actions', []), list) else 1)
    G = 1.0 / (num_prims + 1e-6)  # simplicity: fewer primitives = higher G
    A = estimate_alienation(prog, eml_graph)

    # 139 相变阈值检查
    if O < 0.2:
        return -1  # O too low — below 139 threshold
    if A > 0.5:
        return -1  # A too large — alienation at 139 critical

    return G * O / (A + 1e-6)


# ============================================================================
# §12. YinLong DSL — 阴龙运算: 八元数非结合代数核 (v3.17.0 NEW)
# ============================================================================

class YinLongDSL:
    """阴龙运算：八元数非结合代数核 — preserves bracket structure.

    八元数 (Octonion) 是唯一既非结合又非交换的可除代数.
    YinLong DSL 保留括号结构 (非结合性), 防止信息在不同组合顺序下坍缩.

    IDO/TOMAS 语义:
      - 非结合性 = 不同括号组合产生不同结果 → 信息保持多元性
      - 括号残差 (associator) = (a∘b)∘c - a∘(b∘c) → 衡量信息重组的差异
      - 这避免了深度投影中信息因顺序不同而坍缩到同一终点

    在 ARC 搜索中的应用:
      - compose: 将两个操作组合, 保留括号结构 (非简单顺序应用)
      - associator: 计算不同括号组合的残差, 用于评估信息保持质量
    """

    def compose(self, op1: Callable, op2: Callable) -> Callable:
        """Compose two operations preserving bracket structure (non-associativity).

        与普通函数组合 f(g(x)) 不同, YinLong compose 保留括号结构标记,
        使得后续 associator 检查能追踪不同组合顺序的信息差异.

        Args:
            op1: 第一操作 (先执行)
            op2: 第二操作 (后执行)

        Returns:
            组合操作 (带括号结构标记的 wrapped function)
        """
        # Preserve bracket structure (non-associativity)
        def wrapped(grid_or_state: Any) -> Any:
            return op2(op1(grid_or_state))
        # 标记括号结构信息 (用于 associator 追踪)
        wrapped._yinlong_bracket = (op1, op2)  # type: ignore
        wrapped._yinlong_depth = 2  # type: ignore
        if hasattr(op1, '_yinlong_depth'):
            wrapped._yinlong_depth = op1._yinlong_depth + 1  # type: ignore
        return wrapped

    def associator(self, a: Callable, b: Callable, c: Callable) -> Tuple[Callable, Callable]:
        """计算八元数非结合残差: (a∘b)∘c - a∘(b∘c).

        非结合性是八元数的关键特征 — 不同括号组合产生不同结果.
        associator 返回两种组合方式, 差异 (residual) 反映信息保持的多元性.

        在 ARC 搜索中:
          - 如果两种组合结果差异大 → 信息未坍缩 (好, 保留多元性)
          - 如果两种组合结果差异小 → 信息可能已坍缩 (需警惕)

        Args:
            a, b, c: 三个操作函数

        Returns:
            (left_association, right_association) — 两种括号组合
            left = (a∘b)∘c, right = a∘(b∘c)
        """
        left = self.compose(self.compose(a, b), c)   # (a∘b)∘c
        right = self.compose(a, self.compose(b, c))   # a∘(b∘c)
        return (left, right)


# ============================================================================
# §13. Tianxing GaussEx Verifier — 天行方程校验器 (v3.17.0 NEW)
# ============================================================================

def yinlong_tensor_product(W: float, P: float) -> float:
    """八元数启发张量积 — 保留括号结构.

    IDO/TOMAS 语义下的天行方程张量积:
      Ψ = W ⊗ P (八元数启发, 非简单乘积)

    八元数张量积保留括号结构信息, 防止 W 和 P 的简单乘积导致信息坍缩.

    Args:
        W: 波性相干核 (structural complexity)
        P: 粒性实存核 (DOGA score)

    Returns:
        Ψ 值 (W 和 P 的八元数启发张量积)
    """
    # 八元数启发: 使用非结合的交叉乘积
    # 在实数域近似: W*P + bracket_residual (括号残差)
    bracket_residual = 0.1 * (W - P) * (W + P) / (W + P + 1e-6)  # 非结合残差项
    Psi = W * P + bracket_residual
    return Psi


def estimate_structural_phase(candidate_program: Dict) -> float:
    """计算波性相干核 W (structural complexity).

    W = 程序结构复杂度, 衡量程序的信息保持能力.
    基于 num_primitives, action_types, 和 sprite 交互数.

    Args:
        candidate_program: 程序字典, 包含:
            - 'num_primitives' 或 'actions'
            - 'action_types' (可选)
            - 'targets_sprites' (可选)

    Returns:
        W 值 (structural complexity)
    """
    num_prims = candidate_program.get('num_primitives',
               len(candidate_program.get('actions', [])))
    actions = candidate_program.get('actions', [])
    targets = candidate_program.get('targets_sprites', [])

    # 结构复杂度 = 操作多样性 × 目标覆盖度 × 操作深度
    action_types = set()
    for act in actions:
        if isinstance(act, dict):
            action_types.add(act.get('type', act.get('action_id', 'unknown')))
        elif isinstance(act, (list, tuple)) and len(act) >= 1:
            action_types.add(str(act[0]))

    diversity = len(action_types) / max(num_prims, 1)
    coverage = len(targets) / max(num_prims + 1, 1)
    depth = min(num_prims * 0.1, 1.0)

    W = diversity * 0.4 + coverage * 0.3 + depth * 0.3
    return min(max(W, 0.0), 1.0)


def solve_tianxing(W_seq: List[float], P_seq: List[float], dt: float = 1.0) -> float:
    """IDO/TOMAS 语义下的天行方程数值求解器.

    天行方程: Xi = tanh(real(S²))
      S = Σ Ψ_i × dt, Ψ_i = W_i ⊗ P_i (八元数启发张量积)
      Xi ∈ [0, 1] (tanh 饱和)

    相变阈值:
      - Xi → 1: 天行相变成功 (程序与 EML 超图强耦合)
      - Xi < 1-5/6 = 1/6: 相变失败 (程序与 EML 超图耦合不足)

    Args:
        W_seq: 波性相干核序列 [W_1, W_2, ...]
        P_seq: 粒性实存核序列 [P_1, P_2, ...]
        dt: 时间步长 (默认 1.0)

    Returns:
        Xi 值 (天行方程解)
    """
    S = 0.0  # accumulated coupling
    for W, P in zip(W_seq, P_seq):
        Psi = yinlong_tensor_product(W, P)  # tensor product (non-associative)
        S += Psi * dt
    # Xi = tanh(real(S²)) — coherent square + real part + tanh saturation
    Xi = math.tanh(S ** 2)
    return Xi


class TianxingGaussExVerifier:
    """基于天行方程的 GaussEx 校验器 — Xi = tanh(real(S²)).

    天行方程将程序验证从单纯输出匹配提升到结构相变检查:
      - W (波性相干核) = 程序结构复杂度
      - P (粒性实存核) = DOGA 评分 (O/G/A)
      - Xi = tanh(S²) = 天行相变指标

    139 相变阈值:
      - residual = 1 - Xi ≤ 1/6 → PASS (相变成功)
      - residual > 1/6 → FAIL (相变失败)

    在 ARC 搜索 pipeline 中的集成:
      Phase 2.6 (Neural κ-PS → 天行验证) — 在 neuro_inspired_kps_search 之后,
      使用 TianxingGaussExVerifier 对候选程序进行结构相变检查,
      替代/补充单纯的输出匹配验证.
    """

    def verify(self, candidate_program: Dict, examples: List[Tuple[Any, Any]]) -> Dict:
        """验证候选程序是否通过天行方程相变检查.

        Args:
            candidate_program: 候选程序字典, 包含:
                - 'num_primitives' 或 'actions'
                - 'targets_sprites'
                - 'action_types' (可选)
            examples: 示例列表 [(input, output), ...]

        Returns:
            验证结果字典:
                - 'passed': bool (是否通过相变检查)
                - 'xi_value': float (Xi 相变指标值)
                - 'residual': float (残差 = 1 - Xi)
                - 'threshold': float (阈值 = 1/6)
        """
        W_history: List[float] = []  # 波性相干核 (program structural complexity)
        P_history: List[float] = []  # 粒性实存核 (DOGA: O/G/A)

        for inp, out in examples:
            W = estimate_structural_phase(candidate_program)
            # P = DOGA score for this example
            P = self._estimate_doga_for_example(candidate_program, inp, out)
            W_history.append(W)
            P_history.append(P)

        # Solve Tianxing equation
        Xi = solve_tianxing(W_history, P_history, dt=1.0)

        # 139 phase transition threshold
        residual = 1.0 - Xi  # or |1 - Xi|
        passed = residual <= (1.0 - 5.0 / 6.0)  # 1/6 tolerance

        return {
            'passed': passed,
            'xi_value': Xi,
            'residual': residual,
            'threshold': 1.0 - 5.0 / 6.0,
        }

    def _estimate_doga_for_example(self, prog: Dict, inp: Any, out: Any) -> float:
        """为单个示例计算 DOGA P 值 (粒性实存核).

        Args:
            prog: 程序字典
            inp: 输入 (grid 或 game state)
            out: 输出 (grid 或 game state)

        Returns:
            P 值 (DOGA score, -1 表示不合格)
        """
        # 从示例构造 eml_graph (简化版)
        eml_graph = {'nodes': [], 'edges': []}
        if isinstance(inp, np.ndarray):
            # 从 grid 提取简单 EML 超图
            try:
                non_zero_positions = np.argwhere(inp != 0)
                for i, pos in enumerate(non_zero_positions[:10]):  # 最多10个节点
                    eml_graph['nodes'].append({
                        'blob_id': f'sprite_{i}',
                        'position': tuple(pos),
                    })
            except Exception:
                pass

        # 计算 DOGA score
        score = doga_score(prog, eml_graph)
        if score < 0:
            return 0.0  # DOGA 不合格 → P=0 (无粒性实存)
        return min(score, 1.0)


# ============================================================================
# §9. Pipeline 集成点 — 与 game_solvers.py 的衔接
# ============================================================================

# neuro_inspired_kps_search 在 solve_game() pipeline 中的集成:
#   Phase 2.5 (κ-PS) → Phase 2.6 (Neural κ-PS, v3.16.0)
#   → Phase 2.7 (天行方程验证, v3.17.0 NEW)
#   在 Neural κ-PS 之后, 使用 TianxingGaussExVerifier 对候选程序进行
#   天行相变检查 (Xi = tanh(S²)), 替代/补充单纯输出匹配验证.

# neurally_gated_expand 用于 κ-PS 的扩展步骤:
#   在现有 solve_kappa_priority_search() 中,
#   扩展候选时使用 neurally_gated_expand() 筛选,
#   替代原始的 all-actions-expand.

# YinLongDSL.compose 用于 kappa_snap_search Phase 2 (6-resonance) 增强 (v3.17.0):
#   候选操作通过 YinLongDSL.compose 组合 (保留括号结构),
#   替代简单顺序应用. DOGA score 用作 Phase 3 (9-convergence) 剪枝标准.

# is_soliton / is_soliton_from_sprite 用于 game_solvers sprite 分类 (v3.17.0):
#   在 sprite 提取阶段, 使用 PTS 孤子判别将 sprite 分类为
#   对象 (Jinling Sphere node) vs 背景 (Dead-Zone).

# κ-Causal Reduction (v3.20.0 NEW):
#   ARCSolver 在 game_solvers.py Phase -0.5κ 中调用,
#   使用 κ-Tsirelson bound + GaussEx 残差 η 对候选变换进行物理合法性检查.
#   η < δ_K → Tsirelson-legal (物理直觉正确)
#   η ≥ δ_K → PR-Box非法 (伪规则剪枝 → Dead-Zero fuse)


# ============================================================================
# §14. κ-Tsirelson Locking Theorem — ARC legality bound (v3.20.0 NEW)
# ============================================================================
#
# κ-algebra (八元数虚单位球) 约束 CHSH 相关性到 Tsirelson 界 2√2.
# PR-Box (S=4) 代数非法 → Dead-Zero 熔断.
# ARC求解 = κ-陪集 C(11,4) 因果归约 + 最小 GaussEx 残差 η.
#
# 在 ARC 上下文: 违反对象拓扑守恒的伪规则 = PR-Box 等价物 → DZFUSE.

_KAPPA_TSIRELSON_BOUND: float = 2 * math.sqrt(2)  # ≈ 2.828
_KAPPA_PR_BOX_VALUE: float = 4.0                    # Prohibited — violates κ-algebra
_KAPPA_DELTA_K: float = 0.036                       # GaussEx residual threshold


def tsirelson_causal_reduction(
    eml_graph: Dict,
    candidate_transforms: List[Dict],
    delta_k: float = _KAPPA_DELTA_K,
) -> Dict:
    """κ-陪集C(11,4)因果归约 — ARC = T-minimization in κ-algebra coset space.

    From article: ARC solving = finding min GaussEx residual η causal
    transform T in κ-algebra coset space C(11,4).

    For each candidate transform T:
        1. Project EML state onto C(11,4) coset (κ-Snap pipeline)
        2. Compute GaussEx residual η for T
        3. If η > δ_K → PR-Box illegal → DZFUSE (skip candidate)
        4. If η ≤ δ_K → Tsirelson-legal → compute confidence

    Args:
        eml_graph: EML hypergraph state (dict with 'nodes', 'edges').
        candidate_transforms: List of candidate transform dicts, each with:
            - 'name': Transform name (e.g., 'ROT90', 'MIRROR_X')
            - 'type': Transform category ('omul', 'mir_x', 'mir_y', 'st_eml', 'fill_cc')
            - 'chsh_value': Hypothetical CHSH correlation for this transform
            - 'apply_fn': Optional callable to apply transform (unused in DSL)
        delta_k: GaussEx residual threshold (default 0.036).

    Returns:
        Dict with:
            - 'best_transform': Name of best (min-η) Tsirelson-legal transform
            - 'best_eta': GaussEx residual of best transform
            - 'confidence': Physical purity = 1 - η/δ_K
            - 'pruned': List of PR-Box-illegal (η > δ_K) transform names
            - 'legal_transforms': List of Tsirelson-legal transform names
            - 'tsirelson_bound': KAPPA_TSIRELSON_BOUND ≈ 2.828
    """
    best_transform: Optional[str] = None
    best_eta: float = float('inf')
    pruned: List[str] = []
    legal_transforms: List[str] = []

    # κ-physically allowed transformations (article Appendix A)
    # Each has a baseline CHSH value based on κ-algebra constraints:
    # - OMUL/MIR_X/MIR_Y/ST_EML/FILL_CC are κ-legal by construction (CHSH ≤ 2√2)
    # - Any transform with CHSH > 2√2 is PR-Box illegal → DZFUSE
    tsirelson_legal_types = {
        'omul': _KAPPA_TSIRELSON_BOUND * 0.5,    # ROT90: half Tsirelson bound
        'mir_x': _KAPPA_TSIRELSON_BOUND * 0.7,   # Mirror X: moderate CHSH
        'mir_y': _KAPPA_TSIRELSON_BOUND * 0.7,   # Mirror Y: moderate CHSH
        'st_eml': _KAPPA_TSIRELSON_BOUND * 0.3,  # Color swap: low CHSH
        'fill_cc': _KAPPA_TSIRELSON_BOUND * 0.4, # Fill CC: moderate-low CHSH
    }

    # Simulate κ-Snap projection for each candidate transform
    # In neural DSL context, we use simplified η estimation
    n_eml_nodes = len(eml_graph.get('nodes', []))

    for candidate in candidate_transforms:
        name: str = candidate.get('name', 'unknown')
        t_type: str = candidate.get('type', 'unknown')
        chsh: float = candidate.get('chsh_value', tsirelson_legal_types.get(t_type, _KAPPA_PR_BOX_VALUE))

        # Step 1: Tsirelson legality check (CHSH ≤ 2√2)
        if abs(chsh) > _KAPPA_TSIRELSON_BOUND + 1e-6:
            # PR-Box illegal → DZFUSE → prune this candidate
            pruned.append(name)
            continue

        # Step 2: Estimate GaussEx residual η for this transform
        # Simplified η estimation based on:
        # - Transform type complexity (κ-algebra constraint)
        # - EML graph size (projection quality scales with node count)
        type_complexity = {
            'omul': 0.02, 'mir_x': 0.015, 'mir_y': 0.015,
            'st_eml': 0.025, 'fill_cc': 0.03,
        }
        base_eta = type_complexity.get(t_type, 0.05)
        # η decreases with more EML nodes (better projection quality)
        node_factor = 1.0 / max(math.log2(n_eml_nodes + 1), 1.0)
        eta = base_eta * node_factor

        # Step 3: Check η against δ_K
        if eta > delta_k:
            pruned.append(name)
            continue

        # Step 4: Tsirelson-legal → compute confidence
        legal_transforms.append(name)
        if eta < best_eta:
            best_eta = eta
            best_transform = name

    # Compute confidence for best transform
    confidence: float = 0.0
    if best_transform is not None and best_eta <= delta_k:
        confidence = 1.0 - (best_eta / delta_k)

    return {
        'best_transform': best_transform,
        'best_eta': best_eta,
        'confidence': confidence,
        'pruned': pruned,
        'legal_transforms': legal_transforms,
        'tsirelson_bound': _KAPPA_TSIRELSON_BOUND,
    }


class KappaTsirelsonVerifier:
    """κ-Tsirelson bound verifier — CHSH ≤ 2√2 校验器 (v3.20.0 NEW).

    Based on article: κ-algebra (Octonion imaginary unit sphere) constrains
    CHSH correlations to Tsirelson bound 2√2, prohibiting PR-Box (S=4).

    In ARC context: pseudo-rules violating object conservation = PR-Box.
    Tsirelson Locking Theorem ensures physical legality of transform candidates.

    Usage:
        verifier = KappaTsirelsonVerifier()
        result = verifier.verify_transform(transform_dict)
        if result['legal']:
            # Tsirelson-legal transform → allow in κ-Snap pipeline
        else:
            # PR-Box illegal → DZFUSE → prune candidate
    """

    TSIRELSON_BOUND: float = _KAPPA_TSIRELSON_BOUND  # 2√2 ≈ 2.828
    PR_BOX_VALUE: float = _KAPPA_PR_BOX_VALUE          # 4.0 (prohibited)
    DELTA_K: float = _KAPPA_DELTA_K                     # 0.036

    # κ-legal transform types (Tsirelson-legal by construction)
    KAPPA_LEGAL_TYPES: Dict[str, float] = {
        'omul': 0.02,       # Octonion multiplication (ROT90)
        'mir_x': 0.015,     # Mirror X
        'mir_y': 0.015,     # Mirror Y
        'st_eml': 0.025,    # EML attribute swap
        'fill_cc': 0.03,    # Fill connected component
        'count_nodes': 0.01, # Count nodes (aggregation)
    }

    def verify_transform(self, transform: Dict) -> Dict:
        """Verify a transform candidate against κ-Tsirelson bound.

        Checks:
            1. CHSH correlation ≤ 2√2 (Tsirelson bound)
            2. GaussEx residual η < δ_K (physical legality)
            3. Transform type ∈ κ-legal set (κ-algebra construction)

        Args:
            transform: Transform dict with:
                - 'name': Transform name
                - 'type': Transform type ('omul', 'mir_x', etc.)
                - 'chsh_value': Hypothetical CHSH correlation (optional)
                - 'eta': GaussEx residual (optional, computed if missing)

        Returns:
            Dict with:
                - 'legal': bool (Tsirelson-legal)
                - 'chsh_legal': bool (CHSH ≤ 2√2)
                - 'eta_legal': bool (η < δ_K)
                - 'confidence': float (1 - η/δ_K, 0.0 if illegal)
                - 'reason': str (reason for legality/illegality)
        """
        name: str = transform.get('name', 'unknown')
        t_type: str = transform.get('type', 'unknown')
        chsh: float = transform.get('chsh_value', self.PR_BOX_VALUE)
        eta: Optional[float] = transform.get('eta')

        # Step 1: CHSH legality (Tsirelson bound)
        chsh_legal: bool = abs(chsh) <= self.TSIRELSON_BOUND + 1e-6

        # Step 2: Compute η if not provided
        if eta is None:
            base_eta = self.KAPPA_LEGAL_TYPES.get(t_type, 0.05)
            # PR-Box equivalent types get η = ∞
            if not chsh_legal:
                eta = float('inf')
            else:
                eta = base_eta

        # Step 3: η legality (GaussEx residual threshold)
        eta_legal: bool = eta < self.DELTA_K

        # Step 4: Overall legality
        legal: bool = chsh_legal and eta_legal

        # Step 5: Confidence computation
        confidence: float = 0.0
        if legal:
            confidence = 1.0 - (eta / self.DELTA_K)

        # Step 6: Reason
        if legal:
            reason = f"κ-legal: CHSH={chsh:.3f}≤2√2, η={eta:.4f}<δ_K={self.DELTA_K}"
        elif not chsh_legal:
            reason = f"PR-Box illegal: CHSH={chsh:.3f}>2√2={self.TSIRELSON_BOUND:.3f} → DZFUSE"
        elif not eta_legal:
            reason = f"GaussEx illegal: η={eta:.4f}≥δ_K={self.DELTA_K} → DZFUSE"
        else:
            reason = "Unknown legality violation"

        return {
            'legal': legal,
            'chsh_legal': chsh_legal,
            'eta_legal': eta_legal,
            'confidence': confidence,
            'reason': reason,
        }

    def verify_batch(self, transforms: List[Dict]) -> List[Dict]:
        """Verify a batch of transform candidates against κ-Tsirelson bound.

        Args:
            transforms: List of transform dicts to verify.

        Returns:
            List of verification result dicts (same order as input).
        """
        return [self.verify_transform(t) for t in transforms]

    def compute_confidence_from_eta(self, eta: float) -> float:
        """Compute physical purity confidence from GaussEx residual η.

        confidence = 1 - η/δ_K (analog of physical purity in quantum mechanics).

        Args:
            eta: GaussEx residual value.

        Returns:
            Confidence value (0.0 if η ≥ δ_K, 1 - η/δ_K otherwise).
        """
        if eta >= self.DELTA_K:
            return 0.0
        return 1.0 - (eta / self.DELTA_K)


class KappaCausalReductionDSL:
    """κ-Causal Reduction DSL — neural-inspired κ-Tsirelson transform search (v3.20.0 NEW).

    Combines κ-Tsirelson bound verification with neural-inspired DSL primitives
    for ARC-AGI causal reduction. Uses:
        - RecurrentDSL for iterative κ-Snap convergence
        - GatedDSL for Tsirelson-legal/illegal gating (forget gate = DZFUSE)
        - AdvancedDSL for residual attention (η-weighted candidate selection)

    Pipeline:
        1. perceive(grid) → EML hypergraph
        2. Generate κ-constrained candidates (OMUL, MIR_X, etc.)
        3. RecurrentDSL.repeat_until_converge → κ-Snap projection convergence
        4. GatedDSL.forget_gate → Tsirelson legality check (forget illegal = DZFUSE)
        5. AdvancedDSL.attention_select → η-weighted candidate ranking
        6. Select min-η transform → confidence = 1 - η/δ_K
    """

    def __init__(self) -> None:
        """Initialize κ-Causal Reduction DSL components."""
        self.verifier = KappaTsirelsonVerifier()
        self.recurrent = RecurrentDSL()
        self.gated = GatedDSL()
        self.advanced = AdvancedDSL()

    def solve_kappa_causal_reduction(
        self,
        grid: np.ndarray,
        max_iterations: int = 5,
    ) -> Dict:
        """κ-Causal Reduction solve — find min-η Tsirelson-legal transform.

        Args:
            grid: 2D input grid (numpy array).
            max_iterations: Maximum κ-Snap convergence iterations (default 5).

        Returns:
            Dict with:
                - 'transform': Best transform name (or None)
                - 'eta': Best GaussEx residual
                - 'confidence': Physical purity (1 - η/δ_K)
                - 'convergence_history': η values per iteration
                - 'pruned': PR-Box-illegal transform names
        """
        # Generate κ-constrained candidate transforms
        candidates = self._generate_kappa_candidates(grid)

        # Build EML graph representation
        eml_graph = self._build_eml_from_grid(grid)

        # Iterative κ-Snap convergence (RecurrentDSL)
        convergence_history: List[float] = []
        best_transform: Optional[str] = None
        best_eta: float = float('inf')
        pruned: List[str] = []

        for iteration in range(max_iterations):
            # Tsirelson legality check (GatedDSL forget gate = DZFUSE)
            verified = self.verifier.verify_batch(candidates)

            # Filter legal candidates
            legal_candidates = []
            for cand, ver in zip(candidates, verified):
                if ver['legal']:
                    legal_candidates.append(cand)
                else:
                    if cand['name'] not in pruned:
                        pruned.append(cand['name'])

            if not legal_candidates:
                # All candidates pruned → DZFUSE
                break

            # η-weighted attention selection (AdvancedDSL)
            eta_weights = [
                self.verifier.compute_confidence_from_eta(
                    cand.get('eta', self.verifier.KAPPA_LEGAL_TYPES.get(cand['type'], 0.05))
                )
                for cand in legal_candidates
            ]

            # Select best candidate (max confidence = min η)
            best_idx = int(np.argmax(eta_weights))
            best_cand = legal_candidates[best_idx]
            cand_eta = best_cand.get('eta', self.verifier.KAPPA_LEGAL_TYPES.get(best_cand['type'], 0.05))

            convergence_history.append(cand_eta)

            if cand_eta < best_eta:
                best_eta = cand_eta
                best_transform = best_cand['name']

            # Check convergence: η < δ_K
            if best_eta < self.verifier.DELTA_K:
                break

        confidence = self.verifier.compute_confidence_from_eta(best_eta)

        return {
            'transform': best_transform,
            'eta': best_eta,
            'confidence': confidence,
            'convergence_history': convergence_history,
            'pruned': pruned,
        }

    def _generate_kappa_candidates(self, grid: np.ndarray) -> List[Dict]:
        """Generate κ-constrained candidate transforms for a grid.

        Creates candidates based on κ-legal transform types from article
        Appendix A: OMUL, MIR_X, MIR_Y, ST_EML, FILL_CC, COUNT_NODES.

        Args:
            grid: 2D input grid for candidate generation.

        Returns:
            List of candidate transform dicts.
        """
        h, w = grid.shape if hasattr(grid, 'shape') else (len(grid), len(grid[0]))

        # κ-legal transforms (article Appendix A)
        kappa_types = [
            ('ROT90', 'omul', self.verifier.KAPPA_LEGAL_TYPES['omul']),
            ('MIRROR_X', 'mir_x', self.verifier.KAPPA_LEGAL_TYPES['mir_x']),
            ('MIRROR_Y', 'mir_y', self.verifier.KAPPA_LEGAL_TYPES['mir_y']),
            ('COLOR_SWAP', 'st_eml', self.verifier.KAPPA_LEGAL_TYPES['st_eml']),
            ('FILL_CC', 'fill_cc', self.verifier.KAPPA_LEGAL_TYPES['fill_cc']),
        ]

        candidates = []
        for name, t_type, base_eta in kappa_types:
            # Adjust η based on grid complexity
            n_colors = len(set(grid.flatten())) if hasattr(grid, 'flatten') else 10
            complexity_factor = min(math.log2(n_colors + 1) / 5.0, 1.0)
            eta = base_eta * (1.0 + complexity_factor * 0.5)

            candidates.append({
                'name': name,
                'type': t_type,
                'chsh_value': self.verifier.TSIRELSON_BOUND * 0.5,  # κ-legal by construction
                'eta': eta,
            })

        return candidates

    def _build_eml_from_grid(self, grid: np.ndarray) -> Dict:
        """Build simplified EML hypergraph from grid for κ-Snap projection.

        Args:
            grid: 2D input grid.

        Returns:
            Dict with 'nodes' and 'edges' representing EML state.
        """
        nodes = []
        edges = []
        h, w = grid.shape if hasattr(grid, 'shape') else (len(grid), len(grid[0]))

        for y in range(min(h, 10)):  # Limit nodes for DSL context
            for x in range(min(w, 10)):
                cell = int(grid[y, x]) if hasattr(grid, '__getitem__') else 0
                if cell != 0:  # Only non-empty cells
                    nodes.append({
                        'id': y * w + x,
                        'position': (x, y),
                        'cell_value': cell,
                    })

        return {'nodes': nodes, 'edges': edges}


def confidence_from_eta(eta: float, delta_k: float = _KAPPA_DELTA_K) -> float:
    """Compute physical purity confidence from GaussEx residual η.

    confidence = 1 - η/δ_K (quantum mechanical purity analog).

    η < δ_K → Tsirelson-legal → confidence > 0
    η ≥ δ_K → PR-Box illegal → confidence = 0 (DZFUSE)

    Args:
        eta: GaussEx residual value.
        delta_k: GaussEx threshold (default 0.036).

    Returns:
        Confidence value (0.0 if illegal, 1 - η/δ_K if legal).
    """
    if eta >= delta_k:
        return 0.0
    return 1.0 - (eta / delta_k)
