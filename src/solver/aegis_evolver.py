# -*- coding: utf-8 -*-
"""
AEGIS 程序演进引擎（从 tomas-agi 吸收并适配）
=============================================

吸收来源：tomas-agi/tomas_agi/sim/harness_aegis.py (v2.0, 1175 行)
核心改动（适配 TOMAS ARC 场景）：
    - Digester:  候选程序 → 提取 DSL 原语特征向量
    - Planner:  基于特征规划下一轮 DSL 组合
    - Evolver:  基于 Phase B 结果变异 top-K 程序（mutate DSL params）
    - Critic:   用 GaussEx 验证变异后的程序
    - 将 κ-Snap 从"搜索"升级为"进化搜索"

设计：
    四阶段流水线：
        1. Digester  → 消化候选程序，提取特征（DSL 原语频率向量）
        2. Planner   → 基于特征规划下一批候选（MDL 引导）
        3. Evolver   → 基于规划演化程序（mutate params, crossover）
        4. Critic+Gate → ψ-Gate 批判 + GaussEx 门控过滤

可选启用：config 中 `aegis.enabled: false`（默认关闭）
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════╗
# 数据结构
# ════════════════════════════════════════════════════╝

@dataclass
class ProgramFeature:
    """候选程序的特征向量。"""
    program: Dict[str, Any]
    dsl_freq: Dict[str, int]       # DSL 原语频率（如 {"rotate":2, "mirror_h":1}）
    mdl_cost: int                    # MDL 代价
    topo_hash: str                  # 拓扑哈希
    gaussex_passed: bool = False   # GaussEx 验证是否通过
    confidence: float = 0.0
    generation: int = 0             # 进化代数


@dataclass
class EvolutionConfig:
    """AEGIS 进化配置。"""
    population_size: int = 20        # 每代候选程序数
    num_generations: int = 5        # 进化代数
    mutation_rate: float = 0.3      # 变异率
    crossover_rate: float = 0.5      # 交叉率
    elitism_count: int = 3           # 精英保留数（直接进下一代）
    mdl_weight: float = 0.4         # MDL 代价权重
    accuracy_weight: float = 0.6      # 准确率权重
    use_psi_gate: bool = True        # 是否用 ψ-Gate 门控
    verbose: bool = False


@dataclass
class EvolutionResult:
    """单代进化结果。"""
    generation: int
    best_program: Dict[str, Any]
    best_score: float
    population: List[ProgramFeature]
    stats: Dict[str, Any]


# ════════════════════════════════════════════════════╗
# AEGISEngine — 主类
# ════════════════════════════════════════════════════╝

class AEGISEngine:
    """AEGIS 程序演进引擎。

    将 κ-Snap 两阶段搜索升级为进化搜索：
        初始化种群（κ-Snap Phase B 的 top-K）→
        for G in 1..num_generations:
            Digester（提取特征）→
            Planner（规划变异策略）→
            Evolver（执行变异/交叉）→
            Critic（GaussEx 验证 + ψ-Gate 门控）→
            生存选择（MDL + 准确率加权和）

    用法：
        engine = AEGISEngine(config)
        result = engine.evolve(initial_programs, input_pairs)
        best = result.best_program
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
    ) -> None:
        self.config = config or EvolutionConfig()
        self.history: List[EvolutionResult] = []
        self._rng = random.Random(42)

    # ── 主入口 ──────────────────────────────────────

    def evolve(
        self,
        initial_programs: List[Dict[str, Any]],
        input_pairs: List[Tuple[Any, Any]],
    ) -> EvolutionResult:
        """执行进化搜索，返回最优程序。

        参数：
            initial_programs: 初始种群（来自 κ-Snap Phase B top-K）
            input_pairs:       ARC 输入输出对 [ (input, output), ... ]

        返回：
            EvolutionResult（最后一代的最优结果）
        """
        # 初始化种群
        population = self._programs_to_features(initial_programs, input_pairs)

        for gen in range(self.config.num_generations):
            if self.config.verbose:
                print(f"\n[AEGIS] Generation {gen + 1}/{self.config.num_generations}")
                print(f"  Population size: {len(population)}")

            # ① Digester
            features = self._digest(population)

            # ② Planner
            plan = self._plan(features)

            # ③ Evolver
            offspring = self._evolve(population, plan)

            # ④ Critic + Gate
            evaluated = self._critic(offspring, input_pairs)

            # ⑤ 生存选择
            population = self._select(population, evaluated)

            # 记录统计
            best = max(population, key=lambda p: self._fitness(p))
            stats = self._compute_stats(population)
            result = EvolutionResult(
                generation=gen,
                best_program=best.program,
                best_score=self._fitness(best),
                population=population,
                stats=stats,
            )
            self.history.append(result)

            if self.config.verbose:
                print(f"  Best score: {result.best_score:.4f}")
                print(f"  Avg MDL: {stats.get('avg_mdl', 0):.1f}")

        # 返回最后一代的最优程序
        final_best = max(population, key=lambda p: self._fitness(p))
        return EvolutionResult(
            generation=self.config.num_generations - 1,
            best_program=final_best.program,
            best_score=self._fitness(final_best),
            population=population,
            stats=self._compute_stats(population),
        )

    # ── 01. Digester ────────────────────────────────

    def _digest(self, population: List[ProgramFeature]) -> Dict[str, Any]:
        """消化候选程序，提取特征。

        输出：
            - dsl_histogram:   所有程序的 DSL 原语直方图
            - avg_mdl:         平均 MDL 代价
            - topo_hash_clusters: 按 topo_hash 分组的簇
        """
        dsl_hist: Dict[str, int] = defaultdict(int)
        topo_clusters: Dict[str, List[ProgramFeature]] = defaultdict(list)

        for pf in population:
            for op, cnt in pf.dsl_freq.items():
                dsl_hist[op] += cnt
            topo_clusters[pf.topo_hash].append(pf)

        return {
            "dsl_histogram": dict(dsl_hist),
            "avg_mdl": sum(p.mdl_cost for p in population) / max(len(population), 1),
            "topo_hash_clusters": {k: len(v) for k, v in topo_clusters.items()},
            "num_unique_topo": len(topo_clusters),
        }

    def _programs_to_features(
        self,
        programs: List[Dict],
        input_pairs: List[Tuple],
    ) -> List[ProgramFeature]:
        """将原始程序列表转为 ProgramFeature 列表。"""
        features = []
        for prog in programs:
            freq = self._extract_dsl_freq(prog)
            mdl = self._compute_mdl(prog)
            th = self._compute_topo_hash(prog, input_pairs)
            pf = ProgramFeature(
                program=prog,
                dsl_freq=freq,
                mdl_cost=mdl,
                topo_hash=th,
            )
            features.append(pf)
        return features

    def _extract_dsl_freq(self, prog: Dict) -> Dict[str, int]:
        """提取程序的 DSL 原语频率。"""
        freq: Dict[str, int] = defaultdict(int)
        actions = prog.get("actions", prog.get("program", []))
        for step in actions:
            op = step.get("op", "unknown")
            freq[op] += 1
        return dict(freq)

    def _compute_mdl(self, prog: Dict) -> int:
        """计算程序的 MDL 代价。"""
        # 简化：DSL 原语数 × 5（默认 MDL 代价）
        actions = prog.get("actions", prog.get("program", []))
        return len(actions) * 5

    def _compute_topo_hash(self, prog: Dict, input_pairs: List[Tuple]) -> str:
        """计算程序输出（对首个输入）的 topo_hash。"""
        try:
            from ..core.topo_hash import TopoHashFilter
            thf = TopoHashFilter()
            # 简化：用程序本身的 hash 作为代理
            raw = str(prog)
            return hashlib.md5(raw.encode()).hexdigest()[:16]
        except Exception:
            return "0" * 16

    # ── 02. Planner ──────────────────────────────────

    def _plan(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """基于特征规划下一轮的变异策略。

        策略：
            - 如果 avg_mdl 过高 → 增加 mutate_drop（删除冗余操作）的概率
            - 如果 num_unique_topo 过低 → 增加 mutate_insert（插入新操作）的概率
            - 否则 → 均衡变异
        """
        avg_mdl = features.get("avg_mdl", 30)
        num_unique = features.get("num_unique_topo", 1)
        total = max(len(features.get("dsl_histogram", {})), 1)

        plan = {
            "mutate_drop_prob": min(0.5, avg_mdl / 100.0),
            "mutate_insert_prob": min(0.5, 5.0 / max(num_unique, 1)),
            "mutate_param_prob": 0.3,
            "crossover_prob": self.config.crossover_rate,
            "target_dsl_freq": features.get("dsl_histogram", {}),
        }
        return plan

    # ── 03. Evolver ──────────────────────────────────

    def _evolve(
        self,
        population: List[ProgramFeature],
        plan: Dict[str, Any],
    ) -> List[ProgramFeature]:
        """执行变异和交叉，生成子代。"""
        offspring: List[ProgramFeature] = []

        # 精英保留
        elite = sorted(population, key=lambda p: self._fitness(p), reverse=True)
        elite = elite[:self.config.elitism_count]
        offspring.extend(elite)

        while len(offspring) < self.config.population_size:
            # 选择父代（tournament selection）
            parent_a = self._tournament_select(population)
            parent_b = self._tournament_select(population)

            # 交叉
            if self._rng.random() < plan["crossover_prob"]:
                child_prog = self._crossover(parent_a.program, parent_b.program)
            else:
                child_prog = dict(parent_a.program)

            # 变异
            child_prog = self._mutate(child_prog, plan)

            child = ProgramFeature(
                program=child_prog,
                dsl_freq=self._extract_dsl_freq(child_prog),
                mdl_cost=self._compute_mdl(child_prog),
                topo_hash=self._compute_topo_hash(child_prog, []),
                generation=parent_a.generation + 1,
            )
            offspring.append(child)

        return offspring

    def _tournament_select(
        self, population: List[ProgramFeature], k: int = 3,
    ) -> ProgramFeature:
        """Tournament selection。"""
        candidates = self._rng.sample(population, min(k, len(population)))
        return max(candidates, key=lambda p: self._fitness(p))

    def _crossover(self, prog_a: Dict, prog_b: Dict) -> Dict:
        """交叉两个程序（DSL 操作序列）。"""
        actions_a = list(prog_a.get("actions", prog_a.get("program", [])))
        actions_b = list(prog_b.get("actions", prog_b.get("program", [])))
        if not actions_a or not actions_b:
            return dict(prog_a)
        # 单点交叉
        cut = self._rng.randint(1, min(len(actions_a), len(actions_b)))
        new_actions = actions_a[:cut] + actions_b[cut:]
        child = dict(prog_a)
        child["actions"] = new_actions
        return child

    def _mutate(self, prog: Dict, plan: Dict) -> Dict:
        """变异程序。"""
        actions = list(prog.get("actions", prog.get("program", [])))
        mutated = False

        # 删除操作
        if actions and self._rng.random() < plan["mutate_drop_prob"]:
            idx = self._rng.randint(0, len(actions) - 1)
            actions.pop(idx)
            mutated = True

        # 插入操作
        if self._rng.random() < plan["mutate_insert_prob"]:
            idx = self._rng.randint(0, len(actions))
            new_op = self._random_dsl_op()
            actions.insert(idx, new_op)
            mutated = True

        # 修改参数
        for step in actions:
            if self._rng.random() < plan["mutate_param_prob"]:
                self._mutate_op_params(step)
                mutated = True

        result = dict(prog)
        result["actions"] = actions
        if mutated:
            result["mutated"] = True
        return result

    def _random_dsl_op(self) -> Dict:
        """随机生成一个 DSL 操作。"""
        ops = [
            ("mirror_h", []),
            ("mirror_v", []),
            ("rotate", [90]),
            ("move", [1, 0]),
            ("copy", [0, 0, 3, 3]),
            ("gravity", [0]),
        ]
        name, args = self._rng.choice(ops)
        return {"op": name, "args": list(args)}

    def _mutate_op_params(self, step: Dict):
        """随机修改 DSL 操作的参数。"""
        op = step.get("op", "")
        args = step.get("args", [])
        if op == "rotate" and args:
            args[0] = self._rng.choice([90, 180, 270, -90])
        elif op == "move" and len(args) >= 2:
            args[0] = self._rng.randint(-3, 3)
            args[1] = self._rng.randint(-3, 3)
        elif op == "gravity" and args:
            args[0] = self._rng.randint(0, 3)

    # ── 04. Critic + Gate ─────────────────────────────

    def _critic(
        self,
        offspring: List[ProgramFeature],
        input_pairs: List[Tuple],
    ) -> List[ProgramFeature]:
        """用 GaussEx 验证子代程序。

        可选：用 ψ-Gate 做语义门控（过滤低置信度程序）。
        """
        use_gate = self.config.use_psi_gate
        if use_gate:
            try:
                from .psi_fusion_gate import PsiFusionGate, create_default_anchors
                gate = PsiFusionGate(
                    anchors=create_default_anchors(),
                    verbose=self.config.verbose,
                )
            except ImportError:
                use_gate = False

        for pf in offspring:
            if pf.gaussex_passed:  # 已验证过
                continue
            try:
                passed, conf = self._verify_with_gaussex(pf.program, input_pairs)
                pf.gaussex_passed = passed
                pf.confidence = conf

                if use_gate:
                    # ψ-Gate 门控
                    verdict = gate.check_anchors(
                        input_pairs[0][0] if input_pairs else [[0]],
                        input_pairs[0][1] if input_pairs else [[0]],
                        [pf.program],
                    )
                    if verdict.value in ("BLOCK",):
                        pf.gaussex_passed = False
                        pf.confidence *= 0.3
            except Exception as e:
                if self.config.verbose:
                    print(f"[AEGIS] Critic error: {e}")
                pf.gaussex_passed = False

        return offspring

    def _verify_with_gaussex(
        self, program: Dict, input_pairs: List[Tuple],
    ) -> Tuple[bool, float]:
        """用 GaussEx 验证程序。"""
        try:
            from .gaussex_verifier import GaussExVerifier
            verifier = GaussExVerifier()
            passed = verifier.verify(program, input_pairs)
            # 简化置信度
            conf = 0.9 if passed else 0.2
            return passed, conf
        except Exception:
            return False, 0.0

    # ── 05. 生存选择 ─────────────────────────────────

    def _select(
        self,
        parents: List[ProgramFeature],
        offspring: List[ProgramFeature],
    ) -> List[ProgramFeature]:
        """从父代+子代中选择下一代种群（生存选择）。"""
        combined = parents + offspring
        # 按适应度降序排列
        combined.sort(key=lambda p: self._fitness(p), reverse=True)
        return combined[:self.config.population_size]

    def _fitness(self, pf: ProgramFeature) -> float:
        """计算适应度（MDL + 准确率加权和）。"""
        mdl_penalty = self.config.mdl_weight * (pf.mdl_cost / 100.0)
        acc_bonus = self.config.accuracy_weight * (pf.confidence if pf.gaussex_passed else 0.0)
        return acc_bonus - mdl_penalty

    def _compute_stats(self, population: List[ProgramFeature]) -> Dict[str, Any]:
        """计算种群统计。"""
        if not population:
            return {}
        return {
            "avg_mdl": sum(p.mdl_cost for p in population) / len(population),
            "pass_rate": sum(1 for p in population if p.gaussex_passed) / len(population),
            "avg_confidence": sum(p.confidence for p in population) / len(population),
            "num_generations": population[0].generation + 1 if population else 0,
        }


# ════════════════════════════════════════════════════╗
# 工厂函数 + κ-Snap 集成接口
# ════════════════════════════════════════════════════╝

def create_aegis_from_kappa_snap(
    kappa_results: List[Dict[str, Any]],
    config: Optional[EvolutionConfig] = None,
) -> AEGISEngine:
    """从 κ-Snap Phase B 结果创建 AEGIS 引擎。

    用法（在 kappa_snap_searcher.py 中）：
        # Phase B 完成后：
        if config.get("aegis.enabled", False):
            engine = create_aegis_from_kappa_snap(phase_b_results)
            result = engine.evolve(initial_programs, input_pairs)
            best = result.best_program
    """
    engine = AEGISEngine(config)
    return engine


def integrate_with_kappa_snap(
    kappa_searcher,
    input_pairs: List[Tuple],
    config: Optional[Dict] = None,
):
    """将 AEGIS 集成到 κ-Snap 搜索器。

    修改 kappa_searcher 使其：
        1. 正常执行 Phase A + Phase B
        2. 若 Phase B top-K 程序数 ≥ 5 且 aegis.enabled=True
        3. 则启动 AEGIS 进化搜索
        4. 返回进化后的最优程序
    """
    config = config or {}
    if not config.get("aegis.enabled", False):
        return None  # 不启用

    # 在 kappa_searcher 的 search() 方法末尾添加：
    #   if self.config.get("aegis.enabled"):
    #       engine = AEGISEngine(...)
    #       evolution_result = engine.evolve(phase_b_top_k, input_pairs)
    #       return evolution_result.best_program
    #
    # 此函数返回 monkey-patch 代码片段（由调用者决定如何集成）
    patch_code = """
        # === AEGIS Integration (auto-patch) ===
        if hasattr(self, 'config') and self.config.get('aegis.enabled', False):
            from src.solver.aegis_evolver import AEGISEngine, EvolutionConfig
            engine = AEGISEngine(EvolutionConfig(
                population_size=self.config.get('aegis.population_size', 20),
                num_generations=self.config.get('aegis.num_generations', 5),
                verbose=self.config.get('verbose', False),
            ))
            top_k = self._get_phase_b_top_k()
            if len(top_k) >= 3:
                result = engine.evolve(top_k, input_pairs)
                return result.best_program
        # === End AEGIS Patch ===
    """
    return patch_code


if __name__ == "__main__":
    # 简单测试
    cfg = EvolutionConfig(population_size=10, num_generations=3, verbose=True)
    engine = AEGISEngine(cfg)

    # 伪造初始种群
    dummy_programs = [
        {"actions": [{"op": "mirror_h", "args": []}, {"op": "rotate", "args": [90]}]},
        {"actions": [{"op": "mirror_v", "args": []}]},
        {"actions": [{"op": "move", "args": [1, 0]}, {"op": "gravity", "args": [0]}]},
    ]
    dummy_pairs = [([[1,0],[0,1]], [[0,1],[1,0]])]

    result = engine.evolve(dummy_programs, dummy_pairs)
    print(f"\n✅ AEGIS evolution done:")
    print(f"  Best score: {result.best_score:.4f}")
    print(f"  Best program: {result.best_program}")
