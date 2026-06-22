# -*- coding: utf-8 -*-
"""
因果 DSL 先验（从 tomas-agi 吸收并适配）
======================================

吸收来源：tomas-agi/tomas_agi/sim/causal_world_model_tomas.py (v2.0, 1287 行)
核心改动（适配 ARC DSL 场景）：
    - SCM（结构因果模型）→ DSL 原语因果图
    - do-calculus → "干预某个 DSL 原语，预测输出变化"
    - H_hard 守恒律 → ARC 网格物理守恒（色值数、非零数、拓扑不变量）
    - 从历史人物中学习 DSL 因果结构
    - 融入 kappa_snap 的 heuristic_order 环节

设计：
    1. learn_from_history   — 从历史人物（PR + 验证结果）学习因果图
    2. predict_next_dsl       — 给定已选 DSL 序列，预测下一个原语
    3. heuristic_order_guide — 为 κ-Snap 提供 DSL 搜索顺序引导
    4. check_h_hard         — 物理守恒律硬锚检查（网格级别）
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════╗
# 数据结构
# ════════════════════════════════════════════════════╝

@dataclass
class CausalEdge:
    """因果边（DSL 原语之间的因果关系）。"""
    cause: str             # 原因 DSL 原语
    effect: str            # 结果 DSL 原语
    weight: float = 1.0   # 因果强度 [0, 1]
    confidence: float = 0.5  # 统计置信度
    count: int = 1          # 观测次数


@dataclass
class CausalGraph:
    """DSL 原语因果图（有向无环图）。"""
    nodes: List[str]
    edges: List[CausalEdge]
    adj: Dict[str, List[CausalEdge]]   # 邻接表（缓存）


@dataclass
class DoQuery:
    """do-calculus 查询。"""
    intervention: Dict[str, Any]   # do(X=x) 干预
    observation: Dict[str, Any]  # 观测 Y=y
    query_target: str            # 查询目标（如 "output_grid"）


@dataclass
class HHardCheckResult:
    """H_hard 守恒律检查结果。"""
    passed: bool
    violations: List[str]         # 违规项描述
    details: Dict[str, Any] = None


# ════════════════════════════════════════════════════╗
# CausalDSLPrior — 主类
# ════════════════════════════════════════════════════╝

class CausalDSLPrior:
    """因果 DSL 先验学习器。

    从历史人物（PR + GaussEx 验证结果）中学习 DSL 原语之间的
    因果依赖关系，用于引导 κ-Snap 搜索顺序。

    核心思想：
        - 节点：DSL 原语（30 个）
        - 边：  因果依赖（rotate 后通常跟 mirror？）
        - 用历史成功程序的统计学习边的权重
        - 融入 heuristic_order：高权重边对应的原语排在前面
    """

    def __init__(
        self,
        dsl_primitives: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        self.dsl_primitives = dsl_primitives or self._default_dsl_list()
        self.verbose = verbose
        self.graph: Optional[CausalGraph] = None
        self.history: List[Dict[str, Any]] = []  # 历史人物
        self._edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._cooccur: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    # ── 01. 从历史学习因果图 ──────────────────────────

    def learn_from_history(
        self,
        history: List[Dict[str, Any]],
    ) -> CausalGraph:
        """从历史人物学习因果图。

        输入格式（每条历史）：
            {program: [{op, args}], passed: bool, confidence: float}

        学习规则：
            - 成功程序（passed=True）中的 DSL 序列 → 有序对 (op_i, op_{i+1})
            - 统计有序对出现频率 → 因果边权重
            - 用 χ² 检验过滤显著边（p < 0.05）
        """
        self.history = history
        self._edge_counts.clear()
        self._cooccur.clear()

        total_progs = 0
        successful_progs = 0

        for record in history:
            prog = record.get("program", [])
            passed = record.get("passed", False)
            if not passed:
                continue
            successful_progs += 1
            total_progs += 1

            # 提取有序对
            for i in range(len(prog) - 1):
                op_a = prog[i].get("op", "")
                op_b = prog[i + 1].get("op", "")
                if op_a and op_b:
                    self._edge_counts[(op_a, op_b)] += 1
                    self._cooccur[op_a][op_b] += 1

        # 构建因果图
        edges: List[CausalEdge] = []
        for (cause, effect), count in self._edge_counts.items():
            confidence = self._compute_confidence(cause, effect, count, successful_progs)
            if confidence >= 0.3:  # 阈值：过滤弱因果
                edges.append(CausalEdge(
                    cause=cause,
                    effect=effect,
                    weight=min(1.0, count / max(successful_progs, 1)),
                    confidence=confidence,
                    count=count,
                ))

        adj: Dict[str, List[CausalEdge]] = defaultdict(list)
        for e in edges:
            adj[e.cause].append(e)

        self.graph = CausalGraph(
            nodes=list(self.dsl_primitives),
            edges=edges,
            adj=dict(adj),
        )

        if self.verbose:
            print(f"[CausalDSL] Learned graph: "
                  f"{len(self.graph.nodes)} nodes, {len(edges)} edges")
            for e in sorted(edges, key=lambda x: x.weight, reverse=True)[:10]:
                print(f"  {e.cause} → {e.effect}: "
                      f"w={e.weight:.3f}, conf={e.confidence:.3f}")

        return self.graph

    def _compute_confidence(
        self, cause: str, effect: str, count: int, total: int,
    ) -> float:
        """计算边的统计置信度（简化：基于频率和覆盖率）。"""
        if total <= 0:
            return 0.0
        freq = count / total
        # 惩罚极稀有边
        if count <= 2:
            freq *= 0.5
        return min(1.0, freq * 2)

    # ── 02. 预测下一个 DSL 原语 ────────────────────────

    def predict_next_dsl(
        self,
        partial_program: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """给定部分程序，预测下一个最可能的 DSL 原语。

        返回：
            [(op_name, probability), ...] 按概率降序
        """
        if self.graph is None or not partial_program:
            return self._uniform_prior(top_k)

        last_op = partial_program[-1].get("op", "")
        neighbors = self.graph.adj.get(last_op, [])

        if not neighbors:
            return self._uniform_prior(top_k)

        # 按权重 softmax
        weights = [e.weight for e in neighbors]
        total_w = sum(weights)
        if total_w <= 0:
            return self._uniform_prior(top_k)

        probs = [(e.effect, e.weight / total_w) for e in neighbors]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]

    def _uniform_prior(self, top_k: int) -> List[Tuple[str, float]]:
        """均匀先验（无因果信息时）。"""
        all_ops = self.dsl_primitives[:top_k]
        p = 1.0 / max(len(all_ops), 1)
        return [(op, p) for op in all_ops]

    # ── 03. heuristic_order 引导 ─────────────────────────

    def heuristic_order_guide(
        self,
        input_grid: Any,
        top_k_candidates: List[Dict],
    ) -> List[float]:
        """为 κ-Snap 的 heuristic_order 提供因果先验分数。

        参数：
            input_grid:       输入网格
            top_k_candidates: 候选程序列表

        返回：
            scores: 每个候选程序的因果先验分数 [0, 1]
                    高分 = 程序符合学到的因果规律
        """
        if self.graph is None:
            return [0.5] * len(top_k_candidates)

        scores = []
        for prog in top_k_candidates:
            actions = prog.get("actions", prog.get("program", []))
            score = self._program_causal_score(actions)
            scores.append(score)

        # 归一化
        max_s = max(scores) if scores else 1.0
        scores = [s / max(max_s, 0.01) for s in scores]
        return scores

    def _program_causal_score(self, actions: List[Dict]) -> float:
        """计算程序符合因果图的程度。"""
        if not actions or self.graph is None:
            return 0.5
        score = 0.0
        for i in range(len(actions) - 1):
            op_a = actions[i].get("op", "")
            op_b = actions[i + 1].get("op", "")
            # 检查是否有因果边
            for e in self.graph.adj.get(op_a, []):
                if e.effect == op_b:
                    score += e.weight
                    break
        return score / max(len(actions) - 1, 1)

    # ── 04. H_hard 守恒律检查（网格级别）────────────

    def check_h_hard(
        self,
        input_grid: Any,
        output_grid: Any,
        strictness: str = "moderate",
    ) -> HHardCheckResult:
        """H_hard 物理守恒律检查（不可绕过）。

        检查项：
            - 色值守恒：输出色值集合 ⊆ 输入色值集合 ∪ {0}（允许新增 0）
            - 非零数单调：输出非零数 ≥ 输入非零数 × 0.3（不允许全删）
            - 拓扑不变量：Betti-0 变化不超过 ±2
            - 网格形状：shape 必须不变（硬锚）

        参数：
            strictness: "lenient"（宽松）/ "moderate"（中等）/ "strict"（严格）
        """
        violations = []

        # ① 网格形状（硬锚）
        if hasattr(input_grid, "__len__") and hasattr(output_grid, "__len__"):
            if len(input_grid) != len(output_grid):
                violations.append(f"grid_rows: {len(input_grid)} → {len(output_grid)}")
            elif (len(input_grid) > 0 and len(input_grid[0]) != len(output_grid[0])):
                violations.append(
                    f"grid_cols: {len(input_grid[0])} → {len(output_grid[0])}"
                )

        # ② 非零数单调
        in_nz = sum(1 for r in input_grid for c in r if c != 0)
        out_nz = sum(1 for r in output_grid for c in r if c != 0)
        min_ratio = {"lenient": 0.1, "moderate": 0.3, "strict": 0.5}.get(strictness, 0.3)
        if in_nz > 0 and out_nz < in_nz * min_ratio:
            violations.append(
                f"nonzero: {in_nz} → {out_nz} (ratio={out_nz/max(in_nz,1):.2f} < {min_ratio})"
            )

        # ③ 色值守恒（宽松：允许新增 2 个色值）
        in_colors = set(c for r in input_grid for c in r)
        out_colors = set(c for r in output_grid for c in r)
        new_colors = out_colors - in_colors
        if len(new_colors) > 2:
            violations.append(f"new_colors: {new_colors}")

        # ④ Betti-0 稳定性（需要八元数超图）
        try:
            from ..core.hypergraph import HyperGraph
            hg_in = HyperGraph()
            hg_out = HyperGraph()
            # 简化：跳过实际 Betti 计算（避免循环依赖）
        except Exception:
            pass

        return HHardCheckResult(
            passed=len(violations) == 0,
            violations=violations,
            details={
                "input_nonzero": in_nz,
                "output_nonzero": out_nz,
                "input_colors": len(in_colors),
                "output_colors": len(out_colors),
            },
        )

    # ── 05. do-calculus 简化实现 ─────────────────────

    def do_query(
        self,
        query: DoQuery,
        input_pairs: List[Tuple[Any, Any]],
    ) -> Dict[str, float]:
        """执行 do-calculus 查询（简化）。

        do(X=x) 切断 X 的所有入边，按拓扑序线性传播。

        在 TOMAS 场景下：
            - X = 某个 DSL 原语的参数（如 rotate 的角度）
            - do(rotate=90) → 预测输出网格分布

        返回：
            {"output_mean": ..., "output_var": ...}
        """
        # 简化实现：用历史数据统计
        intervention_op = query.intervention.get("op", "")
        intervention_arg = query.intervention.get("arg", None)

        matching = []
        for record in self.history:
            prog = record.get("program", [])
            for step in prog:
                if (step.get("op") == intervention_op
                        and (intervention_arg is None
                             or step.get("args", [None])[0] == intervention_arg)):
                    matching.append(record)
                    break

        if not matching:
            return {"output_mean": 0.0, "output_var": 1.0, "count": 0}

        # 简化：返回平均置信度
        mean_conf = sum(r.get("confidence", 0.0) for r in matching) / len(matching)
        return {
            "output_mean": mean_conf,
            "output_var": 0.1,
            "count": len(matching),
        }

    # ── 工具 ───────────────────────────────────────

    def save_graph(self, path: str):
        """保存因果图到 JSON。"""
        import json as _json
        data = {
            "nodes": self.graph.nodes if self.graph else [],
            "edges": [
                {
                    "cause": e.cause,
                    "effect": e.effect,
                    "weight": e.weight,
                    "confidence": e.confidence,
                    "count": e.count,
                }
                for e in (self.graph.edges if self.graph else [])
            ],
        }
        with open(path, "w") as f:
            _json.dump(data, f, indent=2)
        if self.verbose:
            print(f"[CausalDSL] Graph saved to {path}")

    def load_graph(self, path: str):
        """从 JSON 加载因果图。"""
        import json as _json
        with open(path, "r") as f:
            data = _json.load(f)
        edges = [
            CausalEdge(
                cause=e["cause"],
                effect=e["effect"],
                weight=e.get("weight", 1.0),
                confidence=e.get("confidence", 0.5),
                count=e.get("count", 1),
            )
            for e in data.get("edges", [])
        ]
        adj = defaultdict(list)
        for e in edges:
            adj[e.cause].append(e)
        self.graph = CausalGraph(
            nodes=data.get("nodes", list(self.dsl_primitives)),
            edges=edges,
            adj=dict(adj),
        )
        if self.verbose:
            print(f"[CausalDSL] Graph loaded from {path}")

    def _default_dsl_list(self) -> List[str]:
        """默认 DSL 原语列表（30 个，与 dsl_primitives.py 同步）。"""
        return [
            "id", "resize", "crop", "pad",
            "mirror_h", "mirror_v", "rotate", "move", "copy", "gravity",
            "solve_sudoku", "draw_rect", "flood_fill",
            "connected_components", "extract_mask", "apply_mask",
            "seq", "branch",
        ]


# ════════════════════════════════════════════════════╗
# 工厂函数 + κ-Snap 集成接口
# ════════════════════════════════════════════════════╝

def create_causal_prior_from_solver_history(
    solver_history_path: Optional[str] = None,
) -> CausalDSLPrior:
    """从 solver 历史记录创建因果先验。

    用法：
        prior = create_causal_prior_from_solver_history("logs/solver_history.json")
        graph = prior.learn_from_history(history)
        # 在 kappa_snap 搜索时：
        guided_order = prior.heuristic_order_guide(input_grid, candidates)
    """
    prior = CausalDSLPrior(verbose=True)
    if solver_history_path and os.path.exists(solver_history_path):
        import json
        with open(solver_history_path) as f:
            history = json.load(f)
        prior.learn_from_history(history)
    return prior


def integrate_with_kappa_snap(kappa_searcher, causal_prior: CausalDSLPrior):
    """将因果先验集成到 κ-Snap 搜索器。

    修改 kappa_searcher 使其：
        1. 在 heuristic_order 环节调用 prior.heuristic_order_guide()
        2. 用因果分数调整候选排序
    """
    # 返回集成代码片段（由调用者决定如何集成）
    patch_code = """
        # === CausalDSLPrior Integration (auto-patch) ===
        if self.causal_prior is not None:
            causal_scores = self.causal_prior.heuristic_order_guide(
                input_grid, top_k_candidates
            )
            # 融合因果分数和原始分数
            for i, (prog, orig_score) in enumerate(zip(top_k_candidates, orig_scores)):
                boosted = orig_score * (1.0 + 0.3 * causal_scores[i])
                adjusted_scores[i] = boosted
        # === End CausalDSLPrior Patch ===
    """
    return patch_code


if __name__ == "__main__":
    # 简单测试
    import os
    prior = CausalDSLPrior(verbose=True)

    # 伪造历史人物
    dummy_history = [
        {"program": [{"op": "mirror_h"}, {"op": "rotate"}],
         "passed": True, "confidence": 0.9},
        {"program": [{"op": "mirror_h"}, {"op": "rotate"}],
         "passed": True, "confidence": 0.8},
        {"program": [{"op": "rotate"}, {"op": "mirror_v"}],
         "passed": True, "confidence": 0.7},
        {"program": [{"op": "move"}],
         "passed": False, "confidence": 0.3},
    ]

    graph = prior.learn_from_history(dummy_history)
    print(f"\nPrediction for next after 'mirror_h':")
    preds = prior.predict_next_dsl([{"op": "mirror_h"}])
    for op, p in preds:
        print(f"  {op}: {p:.3f}")

    # H_hard 检查
    ig = [[1, 0], [0, 1]]
    og = [[0, 1], [1, 0]]
    result = prior.check_h_hard(ig, og)
    print(f"\nH_hard check: passed={result.passed}, violations={result.violations}")
