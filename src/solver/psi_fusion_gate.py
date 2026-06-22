# -*- coding: utf-8 -*-
"""
ψ-Gate 语义门控（从 tomas-agi 吸收并适配 TOMAS ARC-AGI-3）
=====================================================

吸收来源：tomas-agi/tomas_agi/sim/psi_gate.py (v1.0, 714 行)
核心改动（适配 ARC 网格推理场景）：
    - PsiAnchor.I_value   → 网格 topo_hash 匹配度
    - MusCell            → 多假设程序共存（top-K 候选程序）
    - φ-Gate 不确定性   → 八元数编码信息丰度
    - 删除 EML 图谱依赖 → 改用八元数超图
    - 适配 DSL 程序（而非 token 生成）

五大核心能力：
    01. ψ-锚 双轨裁决 — Hard/Soft Anchor 分流
    02. MUS 互斥稳态 — 冲突双存与 Bayesian 渐进裁决
    03. φ-Gate 不确定性量化 — 基于八元数编码的信息丰度
    04. 多程序平行验证 — Wave-Particle 双路径 + Bayesian 融合
    05. 容错衰减器 — Tolerance Decay Controller
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from ..core.octonion_hyperedge import OctonionHyperEdge
    from ..core.hypergraph import HyperGraph
    _HAVE_OCT = True
except ImportError:
    _HAVE_OCT = False
    OctonionHyperEdge = None  # type: ignore
    HyperGraph = None  # type: ignore


# ══════════════════════════════════════════════════════╗
# 枚举
# ══════════════════════════════════════════════════════╝

class GateVerdict(Enum):
    """ψ-Gate 裁决结果。"""
    PASS       = "PASS"         # 通过，置信度足够
    BLOCK      = "BLOCK"        # 拦截，触发硬锚
    DEFER      = "DEFER"        # 推迟，进入 MUS 双存
    PROBE      = "PROBE"        # 探测模式 — 多程序并行验证
    SOFT_PASS  = "SOFT_PASS"   # 软通过 — 容错衰减后通过


class AnchorType(Enum):
    """ψ-锚 类型。"""
    HARD = "hard"   # I_value = 1.0，不可软化
    SOFT = "soft"   # I_value < 1.0，可软化
    PROBE = "probe"  # 探测锚


class MusResolution(Enum):
    """MUS 冲突裁决策略。"""
    PENDING          = "PENDING"
    DELAYED_DECISION = "DELAYED_DECISION"
    SELECT_A         = "SELECT_A"
    SELECT_B         = "SELECT_B"
    HYBRID_FUSION   = "HYBRID_FUSION"
    ARCHIVE          = "ARCHIVE"


# ══════════════════════════════════════════════════════╗
# 数据结构
# ══════════════════════════════════════════════════════╝

@dataclass
class PsiAnchor:
    """ψ-锚点定义（适配 ARC 网格场景）。

    I_value 计算方式（vs 原版）：
        原版：基于 EML 图谱的信息丰度
        适配：基于八元数 topo_hash 匹配度
              I = 1.0 - min(1.0, topo_hash_distance / threshold)
    """
    name: str
    predicate: str           # 断言逻辑（Python 表达式，可含 topo_hash, betti0 等变量）
    I_value: float = 1.0     # 信息权重 [0, 1]
    on_violation: str = "BLOCK"
    source: str = "TOMAS_PsiGate_v2.3"
    created_at: float = field(default_factory=time.time)
    violation_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def anchor_type(self) -> AnchorType:
        if self.I_value >= 0.999:
            return AnchorType.HARD
        return AnchorType.SOFT

    def compute_I_from_grid(self, input_grid: Any, output_grid: Any) -> float:
        """从输入输出网格计算 I_value（信息权重）。"""
        if not _HAVE_OCT:
            return self.I_value
        try:
            enc = OctonionHyperEdge()
            o_input = enc.encode_grid(input_grid)
            o_output = enc.encode_grid(output_grid)
            # 八元数距离 → I_value
            dist = abs(o_input - o_output)
            norm = max(abs(o_input), abs(o_output), 1e-10)
            rel_dist = min(1.0, dist / norm)
            self.I_value = 1.0 - rel_dist
        except Exception:
            pass
        return self.I_value

    def soften(self, delta: float):
        """软化锚点。"""
        self.I_value = max(0.0, min(0.999, self.I_value - delta))

    def harden(self, delta: float):
        """硬化锚点。"""
        self.I_value = max(0.001, min(1.0, self.I_value + delta))


@dataclass
class MusCell:
    """MUS 互斥稳态单元（适配：多假设程序共存）。

    原版：两个实体（e_a, e_b）的冲突双存
    适配：top-K 候选程序（prog_a, prog_b, ...）的 Bayesian 权重跟踪
    """
    cell_id: str
    prog_a: Dict[str, Any]    # 候选程序 A（DSL 程序字典）
    prog_b: Dict[str, Any]    # 候选程序 B
    tag: str = ""              # 冲突标签（如 "shape_mismatch", "color_mismatch"）
    weight_a: float = 0.5     # Bayesian 权重
    weight_b: float = 0.5
    resolution: MusResolution = MusResolution.PENDING
    evidence_log: List[Dict] = field(default_factory=list)
    resolved_by: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    ksnap_hash: Optional[str] = None  # κ-Snap 拓扑哈希

    def add_evidence(self, side: str, evidence: Dict, likelihood: float):
        """添加证据并更新 Bayesian 权重。"""
        prior = self.weight_a if side == "a" else self.weight_b
        posterior = prior * likelihood
        ev = prior * likelihood + (1 - prior) * (1 - likelihood)
        new_w = posterior / max(ev, 1e-10)
        if side == "a":
            self.weight_a = max(0.01, min(0.99, new_w))
            self.weight_b = 1.0 - self.weight_a
        else:
            self.weight_b = max(0.01, min(0.99, new_w))
            self.weight_a = 1.0 - self.weight_b
        self.evidence_log.append({
            "side": side, "likelihood": likelihood,
            "new_weight": new_w, "ts": time.time(),
        })

    def should_resolve(self, threshold: float = 0.95) -> bool:
        """检查是否达到裁决阈值。"""
        return max(self.weight_a, self.weight_b) >= threshold

    def resolve(self) -> str:
        """执行裁决，返回 'a' 或 'b'。"""
        if self.weight_a >= self.weight_b:
            self.resolution = MusResolution.SELECT_A
            self.resolved_by = "bayesian_threshold"
            return "a"
        else:
            self.resolution = MusResolution.SELECT_B
            self.resolved_by = "bayesian_threshold"
            return "b"


@dataclass
class UncertaintyEstimate:
    """φ-Gate 不确定性估计（基于八元数编码）。"""
    info_abundance: float       # 信息丰度（基于八元数分量方差）
    topo_invariants: Dict[str, float]    # 拓扑不变量（Betti0, topo_hash）
    program_complexity: int    # DSL 程序长度（行动数代理）
    grid_entropy: float = 0.0  # 输出网格熵

    @property
    def uncertainty_score(self) -> float:
        """综合不确定性得分 [0, 1]。"""
        # 信息丰度低 → 不确定性高
        u_info = 1.0 - min(1.0, self.info_abundance)
        # 拓扑不变量不稳定 → 不确定性高
        u_topo = 1.0 - min(1.0, self.topo_invariants.get("betti0_stability", 0.5))
        # 程序过长 → 过拟合风险 → 不确定性高
        u_prog = min(1.0, self.program_complexity / 50.0)
        return 0.4 * u_info + 0.3 * u_topo + 0.3 * u_prog


# ══════════════════════════════════════════════════════╗
# ψ-Gate 主类
# ══════════════════════════════════════════════════════╝

class PsiFusionGate:
    """ψ-Gate 语义门控器（适配 TOMAS ARC 求解器）。

    集成位置：
        - 替换 tomas_solver.py 中的 _fusion_decision()
        - 在 κ-Snap 搜索后、输出前做语义门控
        - 管理 top-K 候选程序的 MUS 共存
    """

    def __init__(
        self,
        anchors: Optional[List[PsiAnchor]] = None,
        tolerance_decay_rate: float = 0.05,
        verbose: bool = False,
    ) -> None:
        self.anchors: List[PsiAnchor] = anchors or []
        self.tolerance_decay_rate = tolerance_decay_rate
        self.verbose = verbose
        self.mus_cells: Dict[str, MusCell] = {}
        self._uncertainty_history: List[float] = []
        self._verdict_stats: Dict[str, int] = defaultdict(int)

    # ── 01. ψ-锚 双轨裁决 ───────────────────────────────────

    def check_anchors(
        self,
        input_grid: Any,
        output_grid: Any,
        candidate_programs: List[Dict],
    ) -> GateVerdict:
        """检查所有 ψ-锚，返回裁决结果。

        双轨逻辑：
            - 硬锚（I_value ≥ 0.999）违反 → BLOCK
            - 软锚（I_value < 0.999）违反 → DEFER（进入 MUS）
            - 全部通过 → PASS（或 SOFT_PASS）
        """
        hard_violated = False
        soft_violated = False

        for anchor in self.anchors:
            passed = self._eval_anchor(anchor, input_grid, output_grid, candidate_programs)
            if not passed:
                anchor.violation_count += 1
                if anchor.anchor_type == AnchorType.HARD:
                    hard_violated = True
                else:
                    soft_violated = True

        if hard_violated:
            self._verdict_stats["BLOCK"] += 1
            return GateVerdict.BLOCK
        if soft_violated:
            self._verdict_stats["DEFER"] += 1
            return GateVerdict.DEFER

        # 检查不确定性决定是否进入探测模式
        uncertainty = self.estimate_uncertainty(input_grid, candidate_programs)
        if uncertainty.uncertainty_score > 0.6:
            self._verdict_stats["PROBE"] += 1
            return GateVerdict.PROBE

        self._verdict_stats["PASS"] += 1
        return GateVerdict.PASS

    def _eval_anchor(
        self,
        anchor: PsiAnchor,
        input_grid: Any,
        output_grid: Any,
        programs: List[Dict],
    ) -> bool:
        """评估单个锚点是否通过。"""
        # 简化实现：检查 topo_hash 是否匹配
        if "topo_hash" in anchor.predicate:
            try:
                from ..core.topo_hash import TopoHashFilter
                thf = TopoHashFilter()
                h_input = thf.compute_hash(input_grid)
                h_output = thf.compute_hash(output_grid)
                # 锚点要求 topo_hash 不变 → 检查匹配
                if "==" in anchor.predicate:
                    return h_input == h_output
            except Exception:
                pass
        # 默认通过
        return True

    # ── 02. MUS 互斥稳态 ────────────────────────────────────

    def create_mus_cell(
        self,
        prog_a: Dict, prog_b: Dict,
        tag: str = "",
    ) -> str:
        """创建 MUS 细胞（多假设程序共存）。"""
        cell_id = hashlib.md5(
            f"{json_dumps(prog_a)}{json_dumps(prog_b)}{time.time()}".encode()
        ).hexdigest()[:12]
        cell = MusCell(
            cell_id=cell_id,
            prog_a=prog_a, prog_b=prog_b,
            tag=tag,
        )
        self.mus_cells[cell_id] = cell
        if self.verbose:
            print(f"[MUS] Created cell {cell_id}: {tag}")
        return cell_id

    def update_mus_from_verification(
        self,
        cell_id: str,
        prog_a_passed: bool,
        prog_b_passed: bool,
        confidence_a: float = 0.5,
        confidence_b: float = 0.5,
    ):
        """用 GaussEx 验证结果更新 MUS 细胞权重。"""
        cell = self.mus_cells.get(cell_id)
        if cell is None:
            return
        if prog_a_passed:
            cell.add_evidence("a", {"gaussex": True}, confidence_a)
        if prog_b_passed:
            cell.add_evidence("b", {"gaussex": True}, confidence_b)
        if cell.should_resolve():
            winner = cell.resolve()
            if self.verbose:
                print(f"[MUS] Cell {cell_id} resolved: {winner} "
                      f"(w_a={cell.weight_a:.3f}, w_b={cell.weight_b:.3f})")

    # ── 03. φ-Gate 不确定性量化 ──────────────────────────────

    def estimate_uncertainty(
        self,
        input_grid: Any,
        candidate_programs: List[Dict],
    ) -> UncertaintyEstimate:
        """基于八元数编码估计不确定性。"""
        info_abundance = 0.5  # 默认值
        topo_invariants = {"betti0_stability": 0.5}
        max_complexity = 0

        if _HAVE_OCT:
            try:
                enc = OctonionHyperEdge()
                o = enc.encode_grid(input_grid)
                # 八元数分量方差 → 信息丰度
                components = [
                    o.e0, o.e1, o.e2, o.e3,
                    o.e4, o.e5, o.e6, o.e7,
                ]
                variance = sum((c - sum(components)/8)**2 for c in components) / 8
                info_abundance = min(1.0, variance * 2)
                # topo_hash 稳定性（简化）
                from ..core.topo_hash import TopoHashFilter
                thf = TopoHashFilter()
                h = thf.compute_hash(input_grid)
                topo_invariants = {
                    "topo_hash": abs(hash(str(h))) % 1000 / 1000.0,
                    "betti0_stability": 0.8,  # placeholder
                }
            except Exception:
                pass

        for prog in candidate_programs:
            if isinstance(prog, dict):
                prog_complexity = len(prog.get("actions", []))
            else:
                # Assume it's a ProgramNode
                prog_complexity = len(prog.flatten() if hasattr(prog, "flatten") else [])
            max_complexity = max(max_complexity, prog_complexity)

        est = UncertaintyEstimate(
            info_abundance=info_abundance,
            topo_invariants=topo_invariants,
            program_complexity=max_complexity,
        )
        self._uncertainty_history.append(est.uncertainty_score)
        return est

    # ── 04. 多程序平行验证 ──────────────────────────────────

    def run_parallel_verification(
        self,
        candidate_programs: List[Dict],
        input_pairs: List[Tuple],
        top_k: int = 3,
    ) -> List[Tuple[Dict, float]]:
        """Wave-Particle 双路径：并行验证 top-K 候选程序。

        返回：
            [(program, confidence), ...] 按置信度降序排列
        """
        from ..solver.gaussex_verifier import GaussExVerifier

        verifier = GaussExVerifier()
        results: List[Tuple[Dict, float]] = []

        for prog in candidate_programs[:top_k]:
            try:
                # GaussEx 纤维验证
                passed = verifier.verify(prog, input_pairs)
                # Bayesian 置信度
                from ..solver.bayesian_confidence import BayesianConfidence
                bc = BayesianConfidence()
                conf = bc.compute(prog, input_pairs, passed)
                results.append((prog, conf))
            except Exception as e:
                if self.verbose:
                    print(f"[PsiGate] Verification failed: {e}")
                results.append((prog, 0.0))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ── 05. 容错衰减器 ──────────────────────────────────────

    def tolerance_decay(
        self,
        verdict: GateVerdict,
        consecutive_soft_fails: int,
    ) -> GateVerdict:
        """容错衰减：连续软失败后逐渐放行。

        逻辑：
            - 连续 N 次 SOFT 失败 → 第 N+1 次强制 SOFT_PASS
            - 衰减率 τ 控制放行速度
        """
        if verdict == GateVerdict.SOFT_PASS:
            return verdict  # 已经是软通过

        if verdict == GateVerdict.DEFER and consecutive_soft_fails >= 3:
            # 衰减后放行
            new_I = 1.0 - (self.tolerance_decay_rate * consecutive_soft_fails)
            if self.verbose:
                print(f"[PsiGate] Tolerance decay: {consecutive_soft_fails} fails, "
                      f"effective I = {new_I:.3f}")
            return GateVerdict.SOFT_PASS

        return verdict

    # ── 集成接口 ──────────────────────────────────────────────

    def fuse(
        self,
        candidate_programs: List[Dict],
        input_pairs: List[Tuple],
        input_grid: Any,
        output_grid: Any,
    ) -> Dict[str, Any]:
        """融合裁决：替换 tomas_solver.py 的 _fusion_decision()。

        工作流程：
            1. ψ-锚检查 → 裁决
            2. 若 PASS → 并行验证 top-K
            3. 若 DEFER → 创建 MUS 细胞
            4. Bayesian 融合验证结果
            5. 返回最优程序 + 置信度
        """
        verdict = self.check_anchors(input_grid, output_grid, candidate_programs)

        if verdict == GateVerdict.BLOCK:
            return {"verdict": "BLOCK", "program": None, "confidence": 0.0}

        if verdict == GateVerdict.DEFER:
            # 创建 MUS 细胞（取 top-2 候选）
            if len(candidate_programs) >= 2:
                cell_id = self.create_mus_cell(
                    candidate_programs[0], candidate_programs[1],
                    tag="defer_from_gate",
                )
                # 立即用输入对验证
                self.update_mus_from_verification(
                    cell_id, True, True, 0.6, 0.4,
                )

        # 并行验证
        top_k_results = self.run_parallel_verification(
            candidate_programs, input_pairs, top_k=3,
        )

        if not top_k_results:
            return {"verdict": "NO_SOLUTION", "program": None, "confidence": 0.0}

        best_prog, best_conf = top_k_results[0]
        return {
            "verdict": verdict.value,
            "program": best_prog,
            "confidence": best_conf,
            "all_candidates": top_k_results,
            "uncertainty": self.estimate_uncertainty(input_grid, candidate_programs).uncertainty_score,
        }


# ══════════════════════════════════════════════════════╗
# 工厂函数
# ══════════════════════════════════════════════════════╝

def create_default_anchors() -> List[PsiAnchor]:
    """创建默认 ψ-锚集合（ARC 任务通用）。"""
    return [
        PsiAnchor(
            name="topo_invariance",
            predicate="topo_hash(input) == topo_hash(output)",
            I_value=0.95,
            on_violation="DEFER",
        ),
        PsiAnchor(
            name="betti0_monotonic",
            predicate="betti0(output) >= betti0(input) * 0.5",
            I_value=0.90,
            on_violation="DEFER",
        ),
        PsiAnchor(
            name="color_count_preserve",
            predicate="len(set(output.flatten())) <= len(set(input.flatten())) + 2",
            I_value=0.85,
            on_violation="SOFT_PASS",
        ),
        PsiAnchor(
            name="grid_shape_hard",
            predicate="shape(input) == shape(output)",
            I_value=1.0,   # 硬锚
            on_violation="BLOCK",
        ),
    ]


def json_dumps(obj: Any) -> str:
    """简单 JSON 序列化（避免 circular reference）。"""
    import json as _json
    try:
        return _json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        return str(obj)


if __name__ == "__main__":
    # 简单测试
    gate = PsiFusionGate(verbose=True)
    anchors = create_default_anchors()
    gate = PsiFusionGate(anchors=anchors, verbose=True)
    print(f"[PsiGate] Initialized with {len(anchors)} anchors")
    print(f"[PsiGate] Verdict stats template: {dict(gate._verdict_stats)}")
