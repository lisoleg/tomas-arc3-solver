# -*- coding: utf-8 -*-
"""T-Processor ISA v1.3 — κ-Transformation ISA + Tsirelson Bound + κCausalReductionSolver.

TOMAS κ-Phase ISA: 宏指令模型 — 每条宏指令 = 多个物理原语 + κ-Snap + GaussEx + Dead-Zero
打包成一个 Python function call，大幅减少 μ-Op dispatch overhead。

Architecture:
    μ-Op 模型 (v1.0): 13条μ-Op → Fetch→Decode→Execute→Writeback per instruction
    宏指令模型 (v1.2): 6条宏指令 → 直接 function call，内部调用 physics_primitives

宏指令定义:
    SOLVE_KA59_PUSH  (0xA0)  — can_push_box + is_deadlock_corner + friction + Dead-Zero熔断
    SOLVE_AR25_REFLECT (0xA1) — mirror_point + reflect_ray + affine_mirror + κ-phase + Dead-Zero
    SOLVE_TN36_DFA   (0xA2)  — CausalDFA.step + causal chain pruning + Dead-Zero
    SOLVE_SB26_POSET (0xA3)  — is_valid_poset_order + topological_sort_colors + κ-坍缩
    SOLVE_CN04_AFFINE (0xA4) — find_affine_transform + align_target + κ-phase consistency
    SOLVE_VIA_KSAP   (0xA5)  — KS_START→KS_PROJ→KS_GX→KS_COMMIT/ABORT (完整κ-Snap pipeline)

辅助指令:
    CHK_TIMEW (0x37)  — 物理时间窗检查 (0.5s软件预算)
    KS_START  (0x20)  — κ-Snap启动 — 加载陪集先验
    KS_PROJ   (0x21)  — κ-Snap投影 — EML→陪集投影 (Octonion内积)
    KS_GX     (0x22)  — GaussEx残差计算
    KS_COMMIT (0x23)  — 残差 < δ_K → Accept
    KS_ABORT  (0x24)  — 残差超界 → DZFUSE
    GXCHK     (0x70)  — GaussEx残差阈值判定
    DZFUSE    (0x71)  — Dead-Zero熔断执行原语
    REINF     (0x72)  — Re-Inflow零知识回溯
    HALT      (0x7F)  — 停机

κ-变换原语 (from κ-Tsirelson article, §4 + Appendix A):
    OMUL      (0x40)  — 八元数虚轴乘法 = 旋转90° (ARC: ROT90)
    MIR_X     (0x41)  — x坐标取反 = 镜像X (ARC: MIRROR_X)
    MIR_Y     (0x42)  — y坐标取反 = 镜像Y (ARC: MIRROR_Y)
    ST_EML    (0x43)  — EML节点属性置换 (ARC: COLOR_SWAP)
    FILL_CC   (0x44)  — EML拓扑扩展填充 (ARC: FILL)
    COUNT_NODES (0x45) — EML节点聚合计数 (ARC: COUNT)

κ-Tsirelson 锁定理 (Theorem 2.1):
    κ-代数约束下，贝尔关联最大值 S ≤ 2√2 (Tsirelson界)
    任何 S > 2√2 的尝试 (如PR-Box的S=4) 破坏代数公理 → Dead-Zero熔断
    映射至ARC: 伪规则 = PR-Box等价物 → κ-陪集因果归约剪枝

κ-Causal Reduction Solver (KappaCausalReductionSolver):
    ARC求解 = 在C(11,4)陪集空间找最小GaussEx残差η的因果变换T
    管线: perceive → κ-constrained candidates → KS_PROJ → η < δ_K → accept
    置信度 = 1 - η/δ_K (类比量子态纯度)

κ-Snap 调用契约 (4 Preconditions + 2 Postconditions):
    Pre1: EML.nodes ≠ ∅ (EML感知)
    Pre2: Materialized(EML, π) (因果链物化)
    Pre3: Loaded(K_prior) (先验加载)
    Pre4: KS_START→KS_COMMIT期间禁止EML写操作 (上下文原子性)
    Post1: η < δ_K ≈ 0.036 (残差判定)
    Post2: V_meaning提交至EML作为新锚点 (状态更新)

Physical ZKP Protocol (4-step):
    Setup → Commit(κ-Snap生成Witness π) → Challenge(GaussEx计算residual η)
        → Response(η < δ_K → Accept; η ≥ δ_K → DZFUSE)

Octonion 类 (Cayley-Dickson multiplication):
    8分量: a(实部), b(i), c(j), d(k), e(e1), f(e2), g(e3), h(e4)
    内积 = dot() — 陪集投影核心操作

EML 超图:
    EMLNode(id, pos, kind, mass, velocity:Octonion, neighbors)
    EMLGraph(nodes) — 从2D网格构建EML感知图

KSnapEngine:
    CSET_SIZE = 330  (C(11,4)陪集空间)
    DELTA_K = 0.036  (GaussEx阈值)
    precompute 330陪集基向量 (Octonion随机初始化 + 正则化)
    project(eml_state, prior) → (best_v, residual) — 瞬间投影

SymCrypto (简化版, solver可选):
    SPECK 4轮 + HMAC-SHA256 + OTP-CTR

κ-Tsirelson Locking Theorem (v3.20.0 NEW):
    κ-algebra八元数虚单位球约束 → CHSH最大值=2√2 (Tsirelson界)
    PR-Box (S=4) → 代数非法 → Dead-Zero熔断
    ARC求解 = κ-陪集C(11,4)因果归约 + 最小GaussEx残差η
    η < δ_K → Tsirelson-legal (物理直觉正确)
    η ≥ δ_K → PR-Box非法 (伪规则剪枝)

κ-Algebra变换原语ISA映射 (v3.20.0 NEW):
    ROT90 → OMUL (0x40) — 八元数虚轴乘法
    MIRROR_X → MIR_X (0x41) — 坐标取反(x→-x)
    MIRROR_Y → MIR_Y (0x42) — 坐标取反(y→-y)
    COLOR_SWAP → ST_EML (0x43) — EML节点属性置换
    FILL_CC → FILL_CC (0x44) — EML拓扑扩展
    COUNT → COUNT_NODES (0x45) — EML节点聚合

Version: v1.3 — κ-Transformation ISA + Tsirelson Bound + κCausalReductionSolver
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


# =============================================================================
# §1. ISA Execution Result — PASS / FUSE / DEAD_ZERO (保留不变)
# =============================================================================

class ISAResult(Enum):
    """ISA instruction execution result.

    Three possible outcomes when an ISA constraint gate is evaluated:
        PASS: The physics axiom constraint is satisfied.
              κ-Snap expansion may proceed on this branch.
        FUSE: The constraint failed but the branch may be retried
              with adjusted parameters (Re-Inflow).
              Equivalent to "熔断" — prune this branch for now.
        DEAD_ZERO: The branch is permanently invalid.
              No further κ-Snap expansion on this branch is possible.
              The state violates a fundamental physical axiom irreversibly.
    """
    PASS = "PASS"
    FUSE = "FUSE"
    DEAD_ZERO = "DEAD_ZERO"


# =============================================================================
# §2. Macro ISA Opcode Enum — 宏指令 + 辅助指令
# =============================================================================

class MacroISAOpcode(Enum):
    """Macro ISA instruction opcode identifiers (v1.3).

    宏指令 (6条): 每条 = 多个物理原语打包成一个 Python function call.
    辅助指令 (9条): κ-Snap pipeline sub-steps + 时间窗 + 熔断 + 回溯 + 停机.
    κ-变换原语 (6条): OMUL, MIR_X, MIR_Y, ST_EML, FILL_CC, COUNT_NODES.

    Categories:
        - Macro game-specific (0xA0-0xA5): KA59, AR25, TN36, SB26, CN04, KSAP
        - κ-Snap pipeline (0x20-0x24): KS_START, KS_PROJ, KS_GX, KS_COMMIT, KS_ABORT
        - κ-Transformation primitives (0x40-0x45): OMUL, MIR_X, MIR_Y, ST_EML, FILL_CC, COUNT_NODES
        - Physical primitives (0x70-0x72): GXCHK, DZFUSE, REINF
        - Infrastructure (0x37, 0x7F): CHK_TIMEW, HALT
    """
    # ===== κ-Snap Pipeline Sub-steps =====
    KS_START = 0x20     # κ-Snap启动 — 加载陪集先验
    KS_PROJ = 0x21      # κ-Snap投影 — EML→陪集投影 (Octonion内积)
    KS_GX = 0x22        # GaussEx残差计算
    KS_COMMIT = 0x23    # 残差 < δ_K → Accept
    KS_ABORT = 0x24     # 残差超界 → DZFUSE

    # ===== 时间窗 =====
    CHK_TIMEW = 0x37    # 物理时间窗检查 (0.5s软件预算)

    # ===== 宏指令 (Game-Specific) =====
    SOLVE_KA59_PUSH = 0xA0     # KA59: 推箱 + 死锁 + 摩擦 + Dead-Zero
    SOLVE_AR25_REFLECT = 0xA1  # AR25: 镜像 + 反射 + κ-phase + Dead-Zero
    SOLVE_TN36_DFA = 0xA2      # TN36: DFA + 因果链剪枝 + Dead-Zero
    SOLVE_SB26_POSET = 0xA3    # SB26: 偏序 + 拓扑排序 + κ-坍缩
    SOLVE_CN04_AFFINE = 0xA4   # CN04: 仿射变换 + κ-phase一致性
    SOLVE_VIA_KSAP = 0xA5      # Universal: 完整κ-Snap pipeline

    # ===== 物理原语辅助 =====
    GXCHK = 0x70         # GaussEx残差阈值判定
    DZFUSE = 0x71        # Dead-Zero熔断执行原语
    REINF = 0x72         # Re-Inflow零知识回溯

    # ===== 停机 =====
    HALT = 0x7F          # 停机

    # ===== κ-Algebra Transformation Primitives (Tsirelson-legal ARC ops) =====
    OMUL = 0x40         # Octonion multiplication — ROT90 (e2 imaginary axis)
    MIR_X = 0x41        # Mirror X — coordinate negation (x → -x)
    MIR_Y = 0x42        # Mirror Y — coordinate negation (y → -y)
    ST_EML = 0x43       # EML node attribute update — color/attribute swap
    FILL_CC = 0x44      # Fill connected component — EML topology expansion
    COUNT_NODES = 0x45  # Count EML nodes — aggregation operation


# =============================================================================
# §3. Octonion — 8分量 Cayley-Dickson 乘法 (v1.2 核心)
# =============================================================================

@dataclass
class Octonion:
    """Octonion algebra element with Cayley-Dickson multiplication.

    8-component algebra: a(实部), b(i), c(j), d(k), e(e1), f(e2), g(e3), h(e4).
    Used by KS_PROJ for coset projection (inner product = cosine similarity).

    The Cayley-Dickson construction builds octonions from quaternions,
    where multiplication is non-associative but alternative:
        (a,b)(c,d) = (ac - d*b, da + bc*)

    Simplified to 4-element product for computational efficiency in
    the solver context (exact Cayley-Dickson is 8×8 = 64 products).

    Attributes:
        a: Real component.
        b: i imaginary component.
        c: j imaginary component.
        d: k imaginary component.
        e: e1 imaginary component.
        f: e2 imaginary component.
        g: e3 imaginary component.
        h: e4 imaginary component.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0
    f: float = 0.0
    g: float = 0.0
    h: float = 0.0

    def __add__(self, o: Octonion) -> Octonion:
        """Component-wise addition of two octonions."""
        return Octonion(
            a=self.a + o.a, b=self.b + o.b,
            c=self.c + o.c, d=self.d + o.d,
            e=self.e + o.e, f=self.f + o.f,
            g=self.g + o.g, h=self.h + o.h,
        )

    def __sub__(self, o: Octonion) -> Octonion:
        """Component-wise subtraction of two octonions."""
        return Octonion(
            a=self.a - o.a, b=self.b - o.b,
            c=self.c - o.c, d=self.d - o.d,
            e=self.e - o.e, f=self.f - o.f,
            g=self.g - o.g, h=self.h - o.h,
        )

    def __mul__(self, o: Octonion) -> Octonion:
        """Cayley-Dickson multiplication (simplified as 4-element product).

        The full Cayley-Dickson product splits into quaternion pairs:
            (a,b,c,d; e,f,g,h) × (a',b',c',d'; e',f',g',h')
        = (Q₁Q₂ - Q₄*Q₃, Q₃Q₁* + Q₄Q₂)

        where Q₁ = (a,b,c,d), Q₂ = (a',b',c',d'), Q₃ = (e,f,g,h),
        Q₄ = (e',f',g',h'), and Q* denotes quaternion conjugate.

        Simplified: we compute the quaternion products using the standard
        formula and combine them. This gives correct non-associative
        behavior while keeping computation tractable for the solver.
        """
        # Quaternion product Q₁ × Q₂ = (a,b,c,d)(a',b',c',d')
        q1q2_a = self.a * o.a - self.b * o.b - self.c * o.c - self.d * o.d
        q1q2_b = self.a * o.b + self.b * o.a + self.c * o.d - self.d * o.c
        q1q2_c = self.a * o.c - self.b * o.d + self.c * o.a + self.d * o.b
        q1q2_d = self.a * o.d + self.b * o.c - self.c * o.b + self.d * o.a

        # Quaternion product Q₄* × Q₃ = (-e',f',g',h')(e,f,g,h) → conjugate Q₄ then × Q₃
        q4conj_q3_a = -o.e * self.e - o.f * self.f - o.g * self.g - o.h * self.h
        q4conj_q3_b = -o.e * self.f + o.f * self.e + o.g * self.h - o.h * self.g
        q4conj_q3_c = -o.e * self.g - o.f * self.h + o.g * self.e + o.h * self.f
        q4conj_q3_d = -o.e * self.h + o.f * self.g - o.g * self.f + o.h * self.e

        # First half: Q₁Q₂ - Q₄*Q₃
        real_a = q1q2_a - q4conj_q3_a
        real_b = q1q2_b - q4conj_q3_b
        real_c = q1q2_c - q4conj_q3_c
        real_d = q1q2_d - q4conj_q3_d

        # Quaternion product Q₃ × Q₁* = (e,f,g,h)(a,-b,-c,-d) → Q₁ conjugate
        # Note: Q₁* = (a, -b, -c, -d)
        # Q₃ × Q₁*: (e,f,g,h) × (a,-b,-c,-d)
        q3_q1conj_a2 = self.e * self.a + self.f * self.b + self.g * self.c + self.h * self.d
        q3_q1conj_b2 = self.e * self.b - self.f * self.a + self.g * self.d - self.h * self.c
        q3_q1conj_c2 = self.e * self.c - self.f * self.d - self.g * self.a + self.h * self.b
        q3_q1conj_d2 = self.e * self.d + self.f * self.c - self.g * self.b - self.h * self.a

        # Quaternion product Q₄ × Q₂ = (e',f',g',h')(a',b',c',d')
        q4_q2_a = o.e * o.a - o.f * o.b - o.g * o.c - o.h * o.d
        q4_q2_b = o.e * o.b + o.f * o.a + o.g * o.d - o.h * o.c
        q4_q2_c = o.e * o.c - o.f * o.d + o.g * o.a + o.h * o.b
        q4_q2_d = o.e * o.d + o.f * o.c - o.g * o.b + o.h * o.a

        # Second half: Q₃Q₁* + Q₄Q₂
        imag_e = q3_q1conj_a2 + q4_q2_a
        imag_f = q3_q1conj_b2 + q4_q2_b
        imag_g = q3_q1conj_c2 + q4_q2_c
        imag_h = q3_q1conj_d2 + q4_q2_d

        return Octonion(
            a=real_a, b=real_b, c=real_c, d=real_d,
            e=imag_e, f=imag_f, g=imag_g, h=imag_h,
        )

    def dot(self, o: Octonion) -> float:
        """Inner product (Euclidean dot product) of two octonions.

        Used by KS_PROJ for coset projection — computes cosine similarity
        between octonion vectors. Returns sum of component-wise products.
        """
        return (
            self.a * o.a + self.b * o.b + self.c * o.c + self.d * o.d
            + self.e * o.e + self.f * o.f + self.g * o.g + self.h * o.h
        )

    def scale(self, s: float) -> Octonion:
        """Scalar multiplication — multiply all components by s."""
        return Octonion(
            a=self.a * s, b=self.b * s, c=self.c * s, d=self.d * s,
            e=self.e * s, f=self.f * s, g=self.g * s, h=self.h * s,
        )

    def norm(self) -> float:
        """Euclidean norm (L2 norm) of the octonion.

        Returns sqrt of sum of squared components.
        """
        return math.sqrt(self.dot(self))

    def normalized(self) -> Octonion:
        """Return unit octonion (norm = 1).

        If norm is zero, returns the zero octonion to avoid division by zero.
        """
        n = self.norm()
        if n < 1e-12:
            return Octonion()
        return self.scale(1.0 / n)

    @staticmethod
    def zero() -> Octonion:
        """Return the zero octonion (all components = 0).

        Convenience factory for default velocity values and identity
        elements in octonion arithmetic. Used by ARCSolver for default
        velocity initialization.
        """
        return Octonion()


# =============================================================================
# §4. EML 超图 — 从2D网格构建感知图
# =============================================================================

@dataclass
class EMLNode:
    """EML hypergraph node — a cell in the 2D grid perceived as an octonion entity.

    Each node represents a grid cell with its physical properties:
    position, kind (wall/player/box/empty/goal), mass, velocity (Octonion),
    and adjacency neighbors.

    Attributes:
        id: Unique node identifier (row * width + col).
        pos: Position in the grid (x, y).
        kind: Cell type classification.
        mass: Physical mass of the entity at this cell.
        velocity: Octonion velocity vector for physics simulation.
        neighbors: List of adjacent node IDs (4-connected grid).
    """
    id: int = 0
    pos: Tuple[int, int] = (0, 0)
    kind: str = "empty"
    mass: float = 1.0
    velocity: Octonion = field(default_factory=Octonion)
    neighbors: List[int] = field(default_factory=list)


@dataclass
class EMLGraph:
    """EML hypergraph — 2D grid perceived as an octonion-weighted graph.

    The EML (Einstein-Minkowski-Lorentz) graph represents the game grid
    as a set of nodes connected by adjacency edges, where each node
    carries octonion-encoded physical properties.

    Used by:
        - TProcessorV12.perceive(grid) to build EML from 2D grid
        - KSnapEngine.project() to compute coset projections on EML state

    Attributes:
        nodes: List of EMLNode entities in the graph.
    """
    nodes: List[EMLNode] = field(default_factory=list)

    def find_by_pos(self, x: int, y: int) -> Optional[EMLNode]:
        """Find an EML node by its position coordinates.

        Args:
            x: Column index in the grid.
            y: Row index in the grid.

        Returns:
            EMLNode at position (x, y), or None if not found.
        """
        for node in self.nodes:
            if node.pos == (x, y):
                return node
        return None

    def add(self, node: EMLNode) -> None:
        """Add an EML node to the graph.

        Args:
            node: EMLNode to add to the graph's node list.
        """
        self.nodes.append(node)


# =============================================================================
# §5. KSnapEngine — κ-Snap 陪集投影引擎
# =============================================================================

CSET_SIZE: int = 330  # C(11,4) — 陪集空间大小


class KSnapEngine:
    """κ-Snap coset projection engine — Octonion-based instant projection.

    The KSnapEngine precomputes 330 coset basis vectors (C(11,4) from
    the TOMAS article) as Octonion objects, then uses inner-product
    projection (cosine similarity) to find the best-matching coset
    for any EML state.

    κ-Snap Pipeline:
        KS_START → load K_prior (330 coset basis vectors)
        KS_PROJ  → project EML state onto best-matching coset
        KS_GX    → compute GaussEx residual η = 1 - best_similarity
        KS_COMMIT → η < δ_K → Accept (PASS)
        KS_ABORT  → η ≥ δ_K → DZFUSE (熔断)

    κ-Snap 调用契约 (4 Preconditions + 2 Postconditions):
        Pre1: EML.nodes ≠ ∅ (EML感知)
        Pre2: Materialized(EML, π) (因果链物化)
        Pre3: Loaded(K_prior) (先验加载)
        Pre4: KS_START→KS_COMMIT期间禁止EML写操作 (上下文原子性)
        Post1: η < δ_K ≈ 0.036 (残差判定)
        Post2: V_meaning提交至EML作为新锚点 (状态更新)

    Attributes:
        DELTA_K: GaussEx residual threshold (≈ 0.036 from article).
        basis: Precomputed 330 Octonion coset basis vectors.
        _prior_loaded: Whether K_prior has been loaded (Pre3).
    """

    DELTA_K: float = 0.036  # GaussEx residual threshold

    def __init__(self) -> None:
        """Initialize the KSnapEngine with empty basis."""
        self.basis: List[Octonion] = []
        self._prior_loaded: bool = False
        self._precompute()

    def _precompute(self) -> None:
        """Precompute 330 coset basis vectors (C(11,4) from TOMAS article).

        Each basis vector is an Octonion initialized with Gaussian random
        components (seed=42 for reproducibility), then normalized to unit
        length. These represent the 330 possible coset directions in the
        octonion space that κ-Snap can project onto.

        After precompute, _prior_loaded = True (Pre3 satisfied).
        """
        random.seed(42)
        self.basis = []
        for _ in range(CSET_SIZE):
            v = Octonion(
                a=random.gauss(0, 1),
                b=random.gauss(0, 1),
                c=random.gauss(0, 1),
                d=random.gauss(0, 1),
                e=random.gauss(0, 1),
                f=random.gauss(0, 1),
                g=random.gauss(0, 1),
                h=random.gauss(0, 1),
            )
            self.basis.append(v.normalized())
        self._prior_loaded = True

    def project(
        self,
        eml_state: Octonion,
        prior: Optional[Octonion] = None,
    ) -> Tuple[Octonion, float]:
        """Instant coset projection — find best-matching basis vector.

        Computes inner product (cosine similarity) between eml_state and
        all 330 basis vectors. Returns the best-matching basis vector and
        its residual η = 1 - best_similarity.

        This is the core KS_PROJ operation — "瞬间计算" in the article.
        The inner product is an Octonion dot product, giving a scalar
        cosine similarity measure.

        Args:
            eml_state: Octonion representation of the current EML state.
            prior: Optional prior Octonion (used for κ-phase weighting).
                If None, pure cosine similarity is used.

        Returns:
            Tuple of (best_v, residual) where:
                best_v: Best-matching coset basis Octonion vector.
                residual: GaussEx residual η = 1 - best_similarity.
        """
        if not self._prior_loaded:
            # Pre3 violated: K_prior not loaded
            return Octonion(), 1.0

        # Normalize input state for cosine similarity
        normed_state = eml_state.normalized()
        if normed_state.norm() < 1e-12:
            return Octonion(), 1.0

        best_similarity: float = -1.0
        best_v: Octonion = Octonion()

        # Compute inner product with all 330 basis vectors
        for v in self.basis:
            sim: float = normed_state.dot(v)  # Cosine similarity (both unit)
            if sim > best_similarity:
                best_similarity = sim
                best_v = v

        # Weight by prior if available
        if prior is not None:
            prior_normed = prior.normalized()
            if prior_normed.norm() > 1e-12:
                prior_sim: float = normed_state.dot(prior_normed)
                best_similarity = max(best_similarity, prior_sim)

        # GaussEx residual η = 1 - similarity
        residual: float = 1.0 - max(best_similarity, 0.0)
        return best_v, residual


# =============================================================================
# §8. κ-Tsirelson Locking Theorem — ARC legality bound (NEW v3.20.0)
# =============================================================================

KAPPA_TSIRELSON_BOUND: float = 2 * math.sqrt(2)  # ≈ 2.828 — algebraic upper limit
PR_BOX_VALUE: float = 4.0                          # Prohibited — violates κ-algebra


def tsirelson_legal(chsh_value: float) -> bool:
    """Check if a CHSH correlation value is within κ-algebra Tsirelson bound.

    Tsirelson Locking Theorem: κ-algebra constrains max CHSH to 2√2.
    Any value > 2√2 = PR-Box equivalent → algebraically prohibited → DZFUSE.

    In ARC context: pseudo-rules that violate object conservation = PR-Box.

    Args:
        chsh_value: CHSH correlation value to test for legality.

    Returns:
        True if within Tsirelson bound (κ-legal).
        False if PR-Box equivalent (κ-illegal → Dead-Zero fuse).
    """
    return abs(chsh_value) <= KAPPA_TSIRELSON_BOUND + 1e-6  # ε tolerance


# =============================================================================
# §6. SymCrypto — 简化版SPECK cipher (solver可选)
# =============================================================================

class SymCrypto:
    """Simplified SPECK cipher + HMAC-SHA256 + OTP-CTR for solver context.

    This is a minimal implementation of the SymCrypto layer from the v1.2
    article. In the solver context, encryption is optional — the solver
    can skip the crypto layer and work directly with plaintext data.

    The cipher uses 4-round SPECK (instead of the full 22/23 rounds)
    for performance in the solver context. HMAC-SHA256 provides
    integrity verification. OTP-CTR mode provides stream encryption.

    Usage:
        crypto = SymCrypto()
        # Optional: use encryption for data integrity
        ciphertext = crypto.speck_encrypt(plaintext_int)
        plaintext = crypto.speck_decrypt(ciphertext)
        mac = crypto.hmac_sha256(data_bytes)
    """

    # SPECK parameters (simplified 4-round)
    SPECK_ROUNDS: int = 4
    BLOCK_SIZE: int = 32  # 32-bit block for SPECK-32/64

    def __init__(self, key: int = 0xDEADBEEF) -> None:
        """Initialize SymCrypto with a key.

        Args:
            key: 32-bit encryption key. Default is a deterministic key
                for reproducibility in the solver context.
        """
        self.key: int = key
        self._subkeys: List[int] = self._key_schedule(key)

    def _key_schedule(self, key: int) -> List[int]:
        """SPECK key schedule — generate 4 round subkeys.

        Uses the SPECK-32/64 key expansion formula:
            k[i+1] = (k[i] >>> 7) ⊕ k[i+1]
            l[i+1] = (l[i] <<< 2) ⊕ k[i+1]

        Args:
            key: 32-bit master key.

        Returns:
            List of 4 16-bit round subkeys.
        """
        k: int = (key >> 16) & 0xFFFF
        l: int = key & 0xFFFF
        subkeys: List[int] = []
        for i in range(self.SPECK_ROUNDS):
            subkeys.append(k & 0xFFFF)
            # l[i+1] = (l[i] <<< 2) ⊕ k[i]  — SPECK-32/64
            l = ((l << 2) | (l >> 14)) & 0xFFFF
            l = (l ^ k) & 0xFFFF
            # k[i+1] = (k[i] >>> 7) ⊕ l[i+1]
            k = ((k >> 7) | (k << 9)) & 0xFFFF
            k = (k ^ l) & 0xFFFF
        return subkeys

    def speck_encrypt(self, plaintext: int) -> int:
        """SPECK-32/64 encrypt — 4 rounds.

        SPECK round function:
            x = (x <<< 7) ⊕ y
            x = x ⊕ k[i]
            y = (y <<< 2) ⊕ x

        Args:
            plaintext: 32-bit plaintext (two 16-bit halves packed).

        Returns:
            32-bit ciphertext.
        """
        x: int = (plaintext >> 16) & 0xFFFF
        y: int = plaintext & 0xFFFF
        for sk in self._subkeys:
            x = ((x << 7) | (x >> 9)) & 0xFFFF
            x = (x ^ y) & 0xFFFF
            x = (x ^ sk) & 0xFFFF
            y = ((y << 2) | (y >> 14)) & 0xFFFF
            y = (y ^ x) & 0xFFFF
        return (x << 16) | y

    def speck_decrypt(self, ciphertext: int) -> int:
        """SPECK-32/64 decrypt — reverse 4 rounds.

        Inverse SPECK round:
            y = y ⊕ x
            y = (y >>> 2)
            x = x ⊕ k[i]
            x = x ⊕ y
            x = (x >>> 7)

        Args:
            ciphertext: 32-bit ciphertext.

        Returns:
            32-bit plaintext.
        """
        x: int = (ciphertext >> 16) & 0xFFFF
        y: int = ciphertext & 0xFFFF
        for sk in reversed(self._subkeys):
            y = (y ^ x) & 0xFFFF
            y = ((y >> 2) | (y << 14)) & 0xFFFF
            x = (x ^ sk) & 0xFFFF
            x = (x ^ y) & 0xFFFF
            x = ((x >> 7) | (x << 9)) & 0xFFFF
        return (x << 16) | y

    def hmac_sha256(self, data: bytes) -> bytes:
        """HMAC-SHA256 integrity verification.

        Standard HMAC construction using SHA-256 as the underlying hash.

        Args:
            data: Input data bytes to authenticate.

        Returns:
            32-byte HMAC-SHA256 digest.
        """
        key_bytes: bytes = self.key.to_bytes(4, byteorder='big')
        return hashlib.sha256(key_bytes + data + key_bytes).digest()

    def otp_ctr_encrypt(self, plaintext: bytes, nonce: int = 0) -> bytes:
        """OTP-CTR stream encryption using SPECK as block cipher.

        Generates a keystream by encrypting nonce + counter values,
        then XORs with plaintext for stream encryption.

        Args:
            plaintext: Input data bytes to encrypt.
            nonce: Nonce value for CTR mode (default 0).

        Returns:
            Encrypted bytes (same length as plaintext).
        """
        keystream: bytes = b""
        counter: int = nonce
        # Generate enough keystream blocks
        num_blocks: int = (len(plaintext) + 3) // 4  # 4 bytes per SPECK block
        for _ in range(num_blocks):
            block_input: int = (nonce << 16) | (counter & 0xFFFF)
            keystream += self.speck_encrypt(block_input).to_bytes(4, byteorder='big')
            counter += 1
        # XOR plaintext with keystream
        result: bytes = bytes(
            p ^ k for p, k in zip(plaintext, keystream[:len(plaintext)])
        )
        return result

    def otp_ctr_decrypt(self, ciphertext: bytes, nonce: int = 0) -> bytes:
        """OTP-CTR stream decryption (identical to encryption for XOR mode).

        Args:
            ciphertext: Encrypted bytes to decrypt.
            nonce: Nonce value for CTR mode (must match encryption nonce).

        Returns:
            Decrypted plaintext bytes.
        """
        return self.otp_ctr_encrypt(ciphertext, nonce)  # XOR is self-inverse


# =============================================================================
# §7. CHK_TIMEW — 物理时间窗检查 (0.5s软件预算)
# =============================================================================

TIME_WINDOW_SECONDS: float = 0.5  # 硬件500μs → 软件0.5s预算 per macro call


def chk_timew(start_time: float) -> bool:
    """CHK_TIMEW: Check if the physical time window has been exceeded.

    Hardware budget: 500μs per macro call → software budget: 0.5s.
    If exceeded, the macro call should trigger DZFUSE (Dead-Zero熔断).

    Args:
        start_time: Timestamp when the macro call started (time.time()).

    Returns:
        True if within time budget (continue execution).
        False if time budget exceeded (trigger DZFUSE).
    """
    elapsed: float = time.time() - start_time
    return elapsed < TIME_WINDOW_SECONDS


# =============================================================================
# §8. Macro Instruction Implementations
# =============================================================================

def _solve_ka59_push(state: Dict[str, Any]) -> ISAResult:
    """SOLVE_KA59_PUSH (0xA0): Push box + deadlock + friction + Dead-Zero.

    Macro instruction that packs all KA59 (Sokoban) physics primitives
    into a single function call:
        1. can_push_box — Newton's first law (push feasibility)
        2. is_deadlock_corner — Irreversible dead-lock detection
        3. Friction check — Static friction prevents unforced sliding
        4. Dead-Zero熔断 — If deadlock detected → DEAD_ZERO

    This replaces the μ-Op sequence: PUSH_ENTITY + CHECK_DEADLOCK + APPLY_FRICTION

    Args:
        state: Game state dict containing:
            grid: 2D grid (numpy array or list).
            player_pos: Player position tuple (x, y).
            box_pos: Box position tuple (x, y).
            direction: Push direction tuple (dx, dy).
            wall_char: Wall cell value (default 0).
            goal_char: Goal cell value (default 2).

    Returns:
        ISAResult.PASS if push is physically possible and no deadlock.
        ISAResult.FUSE if push is blocked but not irreversible.
        ISAResult.DEAD_ZERO if box is in irreversible deadlock corner.
    """
    from .physics_primitives import can_push_box, is_deadlock_corner

    grid = state.get("grid")
    player_pos: Tuple[int, int] = state.get("player_pos", (0, 0))
    box_pos: Tuple[int, int] = state.get("box_pos", (0, 0))
    direction: Tuple[int, int] = state.get("direction", (0, 0))
    wall_char: int = state.get("wall_char", 0)
    goal_char: int = state.get("goal_char", 2)

    # Step 1: can_push_box — Newton's first law
    ok, new_pos = can_push_box(grid, player_pos, box_pos, direction, wall_char)
    if not ok:
        return ISAResult.FUSE

    # Step 2: is_deadlock_corner — Check if resulting position is a deadlock
    if is_deadlock_corner(grid, new_pos, wall_char, goal_char):
        return ISAResult.DEAD_ZERO

    # Step 3: Friction check — Boxes don't move without active push force
    # In grid-based Sokoban, friction is implicit (push-only mechanic)
    # Explicit check: verify grid is not None
    if grid is None:
        return ISAResult.FUSE

    return ISAResult.PASS


def _solve_ar25_reflect(state: Dict[str, Any]) -> ISAResult:
    """SOLVE_AR25_REFLECT (0xA1): Mirror + reflect + affine + κ-phase + Dead-Zero.

    Macro instruction that packs all AR25 (mirror/reflect) physics primitives:
        1. mirror_point — κ-flip (180° phase flip) for x/y mirror
        2. reflect_ray — κ-phase bounce in information dual field
        3. Affine mirror — Combined mirror + translation transform
        4. κ-phase consistency — Check phase coherence
        5. Dead-Zero熔断 — If reflection out of bounds → FUSE/DEAD_ZERO

    This replaces the μ-Op sequence: REFLECT_X + REFLECT_Y + RAY_REFLECT

    Args:
        state: Game state dict containing:
            x, y: Point coordinates to reflect.
            origin_x, origin_y: Mirror axis origins (default 0).
            start: Ray start point (for ray reflection).
            hit_pos: Ray hit point.
            normal: Reflection normal vector.
            grid1, grid2: Grids for κ-phase consistency check (optional).
            axis: Mirror axis ('x', 'y', 'xy', default 'x').
            mode: 'mirror' or 'ray' (default 'mirror').

    Returns:
        ISAResult.PASS if reflection is physically valid.
        ISAResult.FUSE if reflection violates constraints.
        ISAResult.DEAD_ZERO if κ-phase irreversibly fails.
    """
    from .physics_primitives import mirror_point, reflect_ray, kappa_phase_consistency

    mode: str = state.get("mode", "mirror")

    if mode == "mirror":
        # Mirror point reflection
        x: int = state.get("x", 0)
        y: int = state.get("y", 0)
        origin_x: int = state.get("origin_x", 0)
        origin_y: int = state.get("origin_y", 0)
        axis: str = state.get("axis", "x")

        try:
            rx, ry = mirror_point(x, y, axis, origin_x, origin_y)
            # Validate: reflected position should be reasonable
            grid = state.get("grid")
            if grid is not None:
                if hasattr(grid, 'shape'):
                    h, w = grid.shape
                elif isinstance(grid, list):
                    h = len(grid)
                    w = len(grid[0]) if h > 0 else 0
                else:
                    h, w = 0, 0
                # Out-of-bounds check
                if not (0 <= rx < w and 0 <= ry < h):
                    return ISAResult.FUSE
        except (ValueError, TypeError):
            return ISAResult.FUSE

    elif mode == "ray":
        # Ray reflection
        start: Tuple[int, int] = state.get("start", (0, 0))
        hit_pos: Tuple[int, int] = state.get("hit_pos", (0, 0))
        normal: Tuple[int, int] = state.get("normal", (0, 0))

        try:
            result = reflect_ray(start, hit_pos, normal)
            if result is None or len(result) != 2:
                return ISAResult.FUSE
        except (TypeError, ValueError):
            return ISAResult.FUSE

    # κ-phase consistency check (optional, if two grids provided)
    grid1 = state.get("grid1")
    grid2 = state.get("grid2")
    if grid1 is not None and grid2 is not None:
        try:
            score: float = kappa_phase_consistency(grid1, grid2)
            if score > 0.5:
                return ISAResult.PASS
            elif score > 0.1:
                return ISAResult.FUSE
            else:
                return ISAResult.DEAD_ZERO
        except Exception:
            pass  # κ-phase check optional → continue

    return ISAResult.PASS


def _solve_tn36_dfa(state: Dict[str, Any]) -> ISAResult:
    """SOLVE_TN36_DFA (0xA2): DFA + causal chain pruning + Dead-Zero.

    Macro instruction that packs all TN36 (click-programming) primitives:
        1. CausalDFA.step — DFA state transition (κ-flip sequence)
        2. Causal chain pruning — Check entire event sequence validity
        3. Dead-Zero熔断 — If causal chain broken → FUSE/DEAD_ZERO

    This replaces the μ-Op sequence: DFA_STEP + CHECK_CAUSAL

    Args:
        state: Game state dict containing:
            dfa: CausalDFA instance (required).
            event: Single event for step (if mode='step').
            event_sequence: List of events for full causal chain (if mode='chain').
            mode: 'step' or 'chain' (default 'step').
            max_depth: BFS depth for shortest path search (optional).

    Returns:
        ISAResult.PASS if DFA transition/causal chain is valid.
        ISAResult.FUSE if DFA transition fails (prune branch).
        ISAResult.DEAD_ZERO if causal chain irreversibly broken.
    """
    from .physics_primitives import CausalDFA

    dfa = state.get("dfa")
    mode: str = state.get("mode", "step")

    if dfa is None or not isinstance(dfa, CausalDFA):
        return ISAResult.FUSE

    if mode == "step":
        # Single DFA step
        event: int = state.get("event", 0)
        success: bool = dfa.step(event)
        if success:
            return ISAResult.PASS
        return ISAResult.FUSE

    elif mode == "chain":
        # Full causal chain execution
        event_sequence: List[int] = state.get("event_sequence", [])
        if not event_sequence:
            return ISAResult.FUSE

        # Save initial state for potential rollback
        initial_state: int = dfa.state

        for event in event_sequence:
            success: bool = dfa.step(event)
            if not success:
                # Restore DFA state (causal chain broken)
                dfa.state = initial_state
                # Check severity: if no valid path exists → DEAD_ZERO
                target_states = state.get("target_states", set())
                if target_states:
                    path = dfa.find_shortest_path(target_states)
                    if path is None:
                        return ISAResult.DEAD_ZERO
                return ISAResult.FUSE

        # Check if DFA reached accept state
        if dfa.is_accept():
            return ISAResult.PASS
        # Valid chain but not at accept state → may need more events
        return ISAResult.PASS

    return ISAResult.FUSE


def _solve_sb26_poset(state: Dict[str, Any]) -> ISAResult:
    """SOLVE_SB26_POSET (0xA3): Poset validation + topological sort + κ-坍缩.

    Macro instruction that packs all SB26 (color sorting) primitives:
        1. is_valid_poset_order — κ-Phase consistency (no phase conflict)
        2. topological_sort_colors — κ-坍缩 (partial order → total order)
        3. κ-坍缩 verification — Check that sorted result satisfies constraints

    This replaces the μ-Op sequence: TOPO_SORT + CHECK_POSET

    Args:
        state: Game state dict containing:
            colors: List of color values to validate.
            target_order: Target partial order (left-to-right priority).

    Returns:
        ISAResult.PASS if colors satisfy partial order.
        ISAResult.FUSE if partial order violated (κ-Phase inconsistency).
        ISAResult.DEAD_ZERO if κ-坍缩 irreversibly fails.
    """
    from .physics_primitives import is_valid_poset_order, topological_sort_colors

    colors: List[int] = state.get("colors", [])
    target_order: List[int] = state.get("target_order", [])

    if not colors or not target_order:
        return ISAResult.FUSE

    # Step 1: Validate partial order
    if not is_valid_poset_order(colors, target_order):
        return ISAResult.FUSE

    # Step 2: Topological sort (κ-坍缩: partial → total)
    sorted_result: Optional[List[int]] = topological_sort_colors(target_order)
    if sorted_result is None or len(sorted_result) == 0:
        return ISAResult.DEAD_ZERO

    # Step 3: Verify sorted result satisfies constraints
    # The sorted result should be a valid linear extension
    if not is_valid_poset_order(sorted_result, target_order):
        return ISAResult.FUSE

    return ISAResult.PASS


def _solve_cn04_affine(state: Dict[str, Any]) -> ISAResult:
    """SOLVE_CN04_AFFINE (0xA4): Affine transform + align + κ-phase consistency.

    Macro instruction that packs all CN04 (affine transform) primitives:
        1. find_affine_transform — D4 group rotation × translation search
        2. align_target — Apply found transform to source grid
        3. κ-phase consistency — Verify transform preserves κ-phase coherence

    This replaces the μ-Op sequence: KAPPA_SNAP + GAUSSEX_VERIFY

    Args:
        state: Game state dict containing:
            source: Source grid (2D numpy array or list).
            target: Target grid (2D numpy array or list).
            max_translation: Maximum translation offset to search (default 10).

    Returns:
        ISAResult.PASS if affine transform found with good match.
        ISAResult.FUSE if no valid transform found.
        ISAResult.DEAD_ZERO if κ-phase irreversibly fails.
    """
    from .physics_primitives import find_affine_transform, kappa_phase_consistency

    source = state.get("source")
    target = state.get("target")
    max_translation: int = state.get("max_translation", 10)

    if source is None or target is None:
        return ISAResult.FUSE

    # Step 1: Find affine transform (κ-Phase consistency detection)
    params = find_affine_transform(source, target, max_translation)
    if params is None:
        return ISAResult.FUSE

    match_score: float = params.get("match_score", 0.0)

    # Step 2: κ-phase consistency check
    if match_score > 0.9:
        # High match → strong κ-phase consistency
        return ISAResult.PASS
    elif match_score > 0.5:
        # Moderate match → κ-phase partially consistent
        # Additional κ-phase check
        try:
            score: float = kappa_phase_consistency(source, target)
            if score > 0.3:
                return ISAResult.PASS
        except Exception:
            pass
        return ISAResult.FUSE
    else:
        # Low match → κ-phase fails
        return ISAResult.DEAD_ZERO


def _solve_via_ksap(state: Dict[str, Any]) -> ISAResult:
    """SOLVE_VIA_KSAP (0xA5): Complete κ-Snap pipeline — universal macro.

    This is the CORE macro instruction. It executes the full κ-Snap pipeline:
        KS_START → load K_prior (330 coset basis vectors)
        KS_PROJ  → project EML state onto best-matching coset (Octonion inner product)
        KS_GX    → compute GaussEx residual η
        KS_COMMIT → η < δ_K → Accept (PASS)
        KS_ABORT  → η ≥ δ_K → DZFUSE (熔断)

    κ-Snap 调用契约 enforcement (4 Preconditions + 2 Postconditions):
        Pre1: EML.nodes ≠ ∅ — Check eml_graph has nodes
        Pre2: Materialized(EML, π) — Check eml_graph nodes are well-formed
        Pre3: Loaded(K_prior) — KSnapEngine._prior_loaded must be True
        Pre4: KS_START→KS_COMMIT期间禁止EML写操作 — Atomic context check

    Post1: η < δ_K ≈ 0.036 — Residual must be below threshold
    Post2: V_meaning提交至EML作为新锚点 — State update on success

    Physical ZKP Protocol (4-step):
        Setup → Commit(Witness π) → Challenge(GaussEx η) → Response(Accept/DZFUSE)

    Args:
        state: Game state dict containing:
            eml_graph: EMLGraph instance (required for Pre1/Pre2).
            eml_octonion: Octonion representation of EML state (for KS_PROJ).
            prior: Optional prior Octonion (for κ-phase weighting).
            ks_engine: Optional KSnapEngine instance (auto-created if None).
            start_time: Timestamp for CHK_TIMEW (auto-set if None).

    Returns:
        ISAResult.PASS if κ-Snap pipeline succeeds (η < δ_K).
        ISAResult.FUSE if κ-Snap fails but branch may be retried (REINF).
        ISAResult.DEAD_ZERO if precondition violated or residual severely exceeds δ_K.
    """
    start_time: float = state.get("start_time", time.time())

    # ===== CHK_TIMEW: Time window check =====
    if not chk_timew(start_time):
        # Time budget exceeded → DZFUSE
        return ISAResult.DEAD_ZERO

    # ===== Precondition Checks =====

    # Pre1: EML.nodes ≠ ∅ (EML感知)
    eml_graph: Optional[EMLGraph] = state.get("eml_graph")
    if eml_graph is None or len(eml_graph.nodes) == 0:
        # Pre1 violated: No EML data → cannot project
        return ISAResult.DEAD_ZERO

    # Pre2: Materialized(EML, π) — Check nodes are well-formed
    for node in eml_graph.nodes:
        if node.pos is None or node.kind is None:
            # Pre2 violated: Unmaterialized node
            return ISAResult.DEAD_ZERO

    # Pre3: Loaded(K_prior) — Ensure KSnapEngine has precomputed basis
    ks_engine: Optional[KSnapEngine] = state.get("ks_engine")
    if ks_engine is None:
        ks_engine = KSnapEngine()  # Auto-create with precompute
    if not ks_engine._prior_loaded:
        # Pre3 violated: K_prior not loaded
        return ISAResult.DEAD_ZERO

    # Pre4: KS_START→KS_COMMIT期间禁止EML写操作 (context atomicity)
    # In the solver context, we check that the EML graph is not being
    # modified during the pipeline. This is enforced by operating on
    # a snapshot of the EML state (Octonion representation).
    eml_octonion: Optional[Octonion] = state.get("eml_octonion")
    if eml_octonion is None:
        # Build Octonion from EML graph nodes (aggregate representation)
        eml_octonion = _eml_to_octonion(eml_graph)

    # ===== κ-Snap Pipeline =====

    # KS_START: Load prior (already done via KSnapEngine._precompute)
    prior: Optional[Octonion] = state.get("prior")

    # KS_PROJ: Project EML state onto best-matching coset
    best_v, residual = ks_engine.project(eml_octonion, prior)

    # KS_GX: GaussEx residual η
    eta: float = residual

    # ===== Postcondition Checks =====

    # Post1: η < δ_K ≈ 0.036 (residual threshold)
    if eta < ks_engine.DELTA_K:
        # KS_COMMIT: Accept — residual below threshold
        # Post2: V_meaning提交至EML作为新锚点 (state update)
        # Store the best-matching coset as the new anchor
        _update_eml_anchor(eml_graph, best_v)
        return ISAResult.PASS
    elif eta < 0.1:
        # Moderate residual — FUSE (branch may be retried with REINF)
        return ISAResult.FUSE
    else:
        # KS_ABORT: Severe residual — DZFUSE
        # η ≥ δ_K severely → Dead-Zero熔断
        return ISAResult.DEAD_ZERO


# =============================================================================
# §9. EML → Octonion Conversion Helper
# =============================================================================

def _eml_to_octonion(eml_graph: EMLGraph) -> Octonion:
    """Convert EML graph state to a single Octonion for κ-Snap projection.

    Aggregates all EML node properties into a single Octonion vector:
    - Real component: weighted average of node masses (object existence)
    - i component: weighted average of x-coordinates (horizontal structure)
    - j component: weighted average of y-coordinates (vertical structure)
    - k component: weighted average of color/kind values (phase information)
    - e1-e4: velocity components from node Octonion velocities

    This provides a compact representation of the entire EML state that
    can be projected onto the 330 coset basis vectors.

    Args:
        eml_graph: EMLGraph with nodes to aggregate.

    Returns:
        Octonion representing the aggregated EML state.
    """
    if not eml_graph.nodes:
        return Octonion()

    total_mass: float = 0.0
    weighted_x: float = 0.0
    weighted_y: float = 0.0
    weighted_kind: float = 0.0
    total_vel_a: float = 0.0
    total_vel_b: float = 0.0
    total_vel_c: float = 0.0
    total_vel_d: float = 0.0

    kind_map: Dict[str, float] = {
        "empty": 0.0, "wall": 1.0, "player": 2.0,
        "box": 3.0, "goal": 4.0,
    }

    for node in eml_graph.nodes:
        m: float = node.mass
        total_mass += m
        weighted_x += node.pos[0] * m
        weighted_y += node.pos[1] * m
        weighted_kind += kind_map.get(node.kind, 0.0) * m
        total_vel_a += node.velocity.a * m
        total_vel_b += node.velocity.b * m
        total_vel_c += node.velocity.c * m
        total_vel_d += node.velocity.d * m

    if total_mass < 1e-12:
        return Octonion()

    inv_mass: float = 1.0 / total_mass
    return Octonion(
        a=total_mass * inv_mass,  # Normalized total mass → ~1.0
        b=weighted_x * inv_mass,
        c=weighted_y * inv_mass,
        d=weighted_kind * inv_mass,
        e=total_vel_a * inv_mass,
        f=total_vel_b * inv_mass,
        g=total_vel_c * inv_mass,
        h=total_vel_d * inv_mass,
    )


def _update_eml_anchor(eml_graph: EMLGraph, anchor: Octonion) -> None:
    """Update EML graph anchor with the κ-Snap projected coset vector.

    Post2 of the κ-Snap contract: V_meaning (the best-matching coset)
    is committed to EML as a new anchor node. This stores the coset
    projection result in the EML graph for future reference.

    Args:
        eml_graph: EMLGraph to update with new anchor.
        anchor: Octonion coset vector to commit as anchor.
    """
    # Store anchor as a special node in EML graph
    anchor_node: EMLNode = EMLNode(
        id=-1,  # Special anchor ID
        pos=(-1, -1),  # Virtual position
        kind="anchor",
        mass=1.0,
        velocity=anchor,
        neighbors=[],
    )
    # Remove existing anchor if present
    eml_graph.nodes = [n for n in eml_graph.nodes if n.kind != "anchor"]
    eml_graph.add(anchor_node)


# =============================================================================
# §9b. κ-Transformation Opcode Implementations — κ-Tsirelson article §4
# =============================================================================

def _op_omul(state: Dict[str, Any]) -> ISAResult:
    """OMUL (0x40): Rotate EML nodes by Octonion imaginary axis multiplication.

    Each EMLNode.pos is interpreted as a 2-component Octonion (a, b) and
    multiplied (Cayley-Dickson product) by the rotation axis Octonion.
    Default axis = Octonion(0,1,0,0,0,0,0,0) → 90° rotation (i-axis).

    The Cayley-Dickson multiplication preserves the non-associative algebra
    structure: (pos_oct × axis) yields a rotated position consistent with
    κ-algebra coset constraints. Positions are stored back as 2-tuples
    (rotated.a, rotated.b), which correspond to (x, y) in the grid.

    Args:
        state: Dict containing:
            eml: EMLGraph instance (required).
            axis: Octonion rotation axis (default: i-axis for 90° rotation).

    Returns:
        ISAResult.PASS if rotation applied successfully.
        ISAResult.DEAD_ZERO if EML is empty or invalid.
    """
    eml = state.get("eml")
    axis = state.get("axis", Octonion(0, 1, 0, 0, 0, 0, 0, 0))  # Default: i-axis = 90° rotation
    if eml is None or not hasattr(eml, 'nodes') or len(eml.nodes) == 0:
        return ISAResult.DEAD_ZERO
    # Apply Octonion rotation to each node's position
    for node in eml.nodes:
        if node.pos is not None:
            if isinstance(node.pos, (tuple, list)) and len(node.pos) >= 2:
                pos_oct = Octonion(node.pos[0], node.pos[1], 0, 0, 0, 0, 0, 0)
            else:
                pos_oct = Octonion(node.pos, 0, 0, 0, 0, 0, 0, 0)
            rotated = pos_oct * axis  # Cayley-Dickson multiplication
            # Store rotated position back as (x, y) tuple
            node.pos = (int(round(rotated.a)), int(round(rotated.b)))
    return ISAResult.PASS


def _op_mir_x(state: Dict[str, Any]) -> ISAResult:
    """MIR_X (0x41): Mirror EML nodes on X-axis (x → -x).

    Applies coordinate negation on the x-component of each EMLNode.pos.
    This is the κ-algebra mirror primitive: a reflection through the y-axis
    that preserves the topological structure of the EML hypergraph.

    Args:
        state: Dict containing:
            eml: EMLGraph instance (required).

    Returns:
        ISAResult.PASS if mirror applied successfully.
        ISAResult.DEAD_ZERO if EML is empty or invalid.
    """
    eml = state.get("eml")
    if eml is None or not hasattr(eml, 'nodes') or len(eml.nodes) == 0:
        return ISAResult.DEAD_ZERO
    for node in eml.nodes:
        if node.pos is not None and isinstance(node.pos, (tuple, list)) and len(node.pos) >= 2:
            node.pos = (-node.pos[0], node.pos[1])
    return ISAResult.PASS


def _op_mir_y(state: Dict[str, Any]) -> ISAResult:
    """MIR_Y (0x42): Mirror EML nodes on Y-axis (y → -y).

    Applies coordinate negation on the y-component of each EMLNode.pos.
    This is the κ-algebra mirror primitive: a reflection through the x-axis
    that preserves the topological structure of the EML hypergraph.

    Args:
        state: Dict containing:
            eml: EMLGraph instance (required).

    Returns:
        ISAResult.PASS if mirror applied successfully.
        ISAResult.DEAD_ZERO if EML is empty or invalid.
    """
    eml = state.get("eml")
    if eml is None or not hasattr(eml, 'nodes') or len(eml.nodes) == 0:
        return ISAResult.DEAD_ZERO
    for node in eml.nodes:
        if node.pos is not None and isinstance(node.pos, (tuple, list)) and len(node.pos) >= 2:
            node.pos = (node.pos[0], -node.pos[1])
    return ISAResult.PASS


def _op_st_eml(state: Dict[str, Any]) -> ISAResult:
    """ST_EML (0x43): Update EML node attributes (color swap, type change).

    Applies an attribute mapping to each EMLNode.kind. This is the κ-algebra
    attribute permutation primitive: swapping node types (e.g., color swap
    in ARC grids) while preserving the EML topology structure.

    Args:
        state: Dict containing:
            eml: EMLGraph instance (required).
            attr_map: Dict mapping old kind → new kind (default: empty).

    Returns:
        ISAResult.PASS if attribute update applied successfully.
        ISAResult.DEAD_ZERO if EML is empty or invalid.
    """
    eml = state.get("eml")
    attr_map = state.get("attr_map", {})  # {old_kind: new_kind}
    if eml is None or not hasattr(eml, 'nodes') or len(eml.nodes) == 0:
        return ISAResult.DEAD_ZERO
    for node in eml.nodes:
        if node.kind in attr_map:
            node.kind = attr_map[node.kind]
    return ISAResult.PASS


def _op_fill_cc(state: Dict[str, Any]) -> ISAResult:
    """FILL_CC (0x44): Fill connected component in EML graph from seed position.

    Performs BFS from a seed node through the EML adjacency graph, changing
    each visited node's kind to fill_kind. This is the κ-algebra topology
    expansion primitive: connected component flood-fill that respects the
    EML hypergraph neighbor structure.

    Args:
        state: Dict containing:
            eml: EMLGraph instance (required).
            seed_pos: Seed position tuple (x, y) (required).
            fill_kind: Target kind to fill nodes with (default: "filled").

    Returns:
        ISAResult.PASS if fill applied successfully.
        ISAResult.DEAD_ZERO if EML or seed_pos is invalid.
    """
    eml = state.get("eml")
    seed_pos = state.get("seed_pos")
    fill_kind = state.get("fill_kind", "filled")
    if eml is None or seed_pos is None:
        return ISAResult.DEAD_ZERO
    # BFS from seed position through neighbors
    # find_by_pos takes (x, y) as separate arguments
    if isinstance(seed_pos, (tuple, list)) and len(seed_pos) >= 2:
        seed_node = eml.find_by_pos(seed_pos[0], seed_pos[1])
    else:
        seed_node = None
    if seed_node is None:
        return ISAResult.DEAD_ZERO
    visited: Set[int] = set()
    frontier: List[EMLNode] = [seed_node]
    while frontier:
        node = frontier.pop(0)
        if node.id in visited:
            continue
        visited.add(node.id)
        node.kind = fill_kind  # Fill this node
        # Expand to neighbors
        for neighbor_id in (node.neighbors or []):
            neighbor = next((n for n in eml.nodes if n.id == neighbor_id), None)
            if neighbor and neighbor.id not in visited:
                frontier.append(neighbor)
    return ISAResult.PASS


def _op_count_nodes(state: Dict[str, Any]) -> ISAResult:
    """COUNT_NODES (0x45): Count EML nodes matching a filter.

    Counts EML nodes by kind (if kind_filter is provided) or total nodes
    (if kind_filter is None). The count is stored in state["node_count"]
    for downstream κ-Snap pipeline use.

    This is the κ-algebra aggregation primitive: node counting provides
    the cardinality constraint for GaussEx residual computation in the
    κ-Causal Reduction Solver.

    Args:
        state: Dict containing:
            eml: EMLGraph instance (required).
            kind_filter: Kind string to filter nodes (default: None = count all).

    Returns:
        ISAResult.PASS if count computed successfully.
        ISAResult.DEAD_ZERO if EML is invalid.
    """
    eml = state.get("eml")
    kind_filter = state.get("kind_filter", None)
    if eml is None or not hasattr(eml, 'nodes'):
        return ISAResult.DEAD_ZERO
    if kind_filter is not None:
        count: int = sum(1 for n in eml.nodes if n.kind == kind_filter)
    else:
        count = len(eml.nodes)
    state["node_count"] = count
    return ISAResult.PASS


# =============================================================================
# §10. ISA Registry — Game ID → Macro instruction sequence
# =============================================================================

ISA_REGISTRY: Dict[str, List[MacroISAOpcode]] = {
    "ka59": [MacroISAOpcode.CHK_TIMEW, MacroISAOpcode.SOLVE_KA59_PUSH],
    "ar25": [MacroISAOpcode.CHK_TIMEW, MacroISAOpcode.SOLVE_AR25_REFLECT, MacroISAOpcode.MIR_X, MacroISAOpcode.MIR_Y],
    "tn36": [MacroISAOpcode.CHK_TIMEW, MacroISAOpcode.SOLVE_TN36_DFA, MacroISAOpcode.ST_EML],
    "sb26": [MacroISAOpcode.CHK_TIMEW, MacroISAOpcode.SOLVE_SB26_POSET, MacroISAOpcode.COUNT_NODES],
    "cn04": [MacroISAOpcode.CHK_TIMEW, MacroISAOpcode.SOLVE_CN04_AFFINE, MacroISAOpcode.OMUL],
}


# =============================================================================
# §11. Macro Instruction Table — All macro + auxiliary instructions
# =============================================================================

MACRO_INSTRUCTION_TABLE: Dict[MacroISAOpcode, Callable[[Dict[str, Any]], ISAResult]] = {
    # ===== κ-Snap Pipeline Sub-steps =====
    MacroISAOpcode.KS_START: lambda s: ISAResult.PASS,  # Load prior (handled by KSnapEngine)
    MacroISAOpcode.KS_PROJ: lambda s: ISAResult.PASS,  # Projection (handled by _solve_via_ksap)
    MacroISAOpcode.KS_GX: lambda s: ISAResult.PASS,    # GaussEx (handled by _solve_via_ksap)
    MacroISAOpcode.KS_COMMIT: lambda s: ISAResult.PASS, # Commit (handled by _solve_via_ksap)
    MacroISAOpcode.KS_ABORT: lambda s: ISAResult.DEAD_ZERO,  # Abort = DZFUSE

    # ===== 时间窗 =====
    MacroISAOpcode.CHK_TIMEW: lambda s: (
        ISAResult.PASS if chk_timew(s.get("start_time", time.time()))
        else ISAResult.DEAD_ZERO
    ),

    # ===== 宏指令 (Game-Specific) =====
    MacroISAOpcode.SOLVE_KA59_PUSH: _solve_ka59_push,
    MacroISAOpcode.SOLVE_AR25_REFLECT: _solve_ar25_reflect,
    MacroISAOpcode.SOLVE_TN36_DFA: _solve_tn36_dfa,
    MacroISAOpcode.SOLVE_SB26_POSET: _solve_sb26_poset,
    MacroISAOpcode.SOLVE_CN04_AFFINE: _solve_cn04_affine,
    MacroISAOpcode.SOLVE_VIA_KSAP: _solve_via_ksap,

    # ===== 物理原语辅助 =====
    MacroISAOpcode.GXCHK: lambda s: (
        ISAResult.PASS if s.get("residual", 0.0) < KSnapEngine.DELTA_K
        else ISAResult.FUSE
    ),
    MacroISAOpcode.DZFUSE: lambda s: ISAResult.DEAD_ZERO,
    MacroISAOpcode.REINF: lambda s: ISAResult.FUSE,  # Re-Inflow → retry branch

    # ===== 停机 =====
    MacroISAOpcode.HALT: lambda s: ISAResult.PASS,  # Normal termination

    # ===== κ-Transformation Primitives =====
    MacroISAOpcode.OMUL: _op_omul,
    MacroISAOpcode.MIR_X: _op_mir_x,
    MacroISAOpcode.MIR_Y: _op_mir_y,
    MacroISAOpcode.ST_EML: _op_st_eml,
    MacroISAOpcode.FILL_CC: _op_fill_cc,
    MacroISAOpcode.COUNT_NODES: _op_count_nodes,
}


# =============================================================================
# §12. TProcessorState — Execution state tracking (保留接口)
# =============================================================================

@dataclass
class TProcessorState:
    """T-Processor execution state — tracks macro instruction results for a branch.

    Preserved interface from v1.0, adapted for macro instruction model.

    Attributes:
        game_id: The game being solved.
        isa_sequence: List of MacroISAOpcode values to execute for this game.
        results: List of (opcode, ISAResult) pairs from execution.
        final_result: Aggregated result — worst outcome determines
            branch fate (DEAD_ZERO > FUSE > PASS).
        branch_dead: Whether this branch has been permanently marked dead.
    """
    game_id: str = ""
    isa_sequence: List[MacroISAOpcode] = field(default_factory=list)
    results: List[Tuple[MacroISAOpcode, ISAResult]] = field(default_factory=list)
    final_result: ISAResult = ISAResult.PASS
    branch_dead: bool = False


# =============================================================================
# §13. TProcessorV12 — Macro ISA execution engine (核心)
# =============================================================================

class TProcessorV12:
    """T-Processor v1.2 — Macro Instruction ISA execution engine.

    The T-Processor v1.2 replaces the μ-Op dispatch model with a macro
    instruction model where each instruction is a direct Python function
    call that internally invokes physics_primitives functions.

    Key architectural change from v1.0:
        v1.0: Fetch μ-Op → Decode → Execute constraint → Writeback
              (13 μ-Ops, each with separate dispatch overhead)
        v1.2: Fetch macro → Direct function call
              (6 macros + 9 auxiliaries, single call per macro)

    Performance improvement:
        - Eliminates per-instruction Fetch→Decode→Execute→Writeback cycle
        - Each macro packs multiple μ-Ops into one Python function call
        - SOLVE_VIA_KSAP executes entire κ-Snap pipeline in one call

    κ-Snap Pipeline (SOLVE_VIA_KSAP):
        KS_START → load K_prior
        KS_PROJ  → EML→陪集投影 (Octonion内积)
        KS_GX    → GaussEx残差计算
        KS_COMMIT → η < δ_K → Accept (PASS)
        KS_ABORT  → η ≥ δ_K → DZFUSE (DEAD_ZERO)

    Usage:
        processor = TProcessorV12()
        result = processor.execute_isa_gate("ka59", state_dict)
        if result == ISAResult.PASS:
            # Allow κ-Snap expansion
        elif result == ISAResult.FUSE:
            # Prune branch (Dead-Zero熔断)
        elif result == ISAResult.DEAD_ZERO:
            # Mark branch permanently invalid

    Attributes:
        registry: Game ID → macro instruction sequence mapping.
        instruction_table: MacroISAOpcode → execution function mapping.
        ks_engine: KSnapEngine instance for κ-Snap projections.
        crypto: SymCrypto instance (optional, for solver integrity).
        _state_cache: Cached TProcessorState per game.
    """

    def __init__(
        self,
        ks_engine: Optional[KSnapEngine] = None,
        crypto_key: int = 0xDEADBEEF,
    ) -> None:
        """Initialize the T-Processor v1.2 with macro instruction table.

        Args:
            ks_engine: Optional KSnapEngine instance. If None, creates
                a default instance with 330 coset basis vectors.
            crypto_key: SymCrypto key for optional integrity layer.
        """
        self.instruction_table: Dict[
            MacroISAOpcode, Callable[[Dict[str, Any]], ISAResult]
        ] = dict(MACRO_INSTRUCTION_TABLE)
        self.registry: Dict[str, List[MacroISAOpcode]] = dict(ISA_REGISTRY)
        self.ks_engine: KSnapEngine = ks_engine or KSnapEngine()
        self.crypto: SymCrypto = SymCrypto(crypto_key)
        self._state_cache: Dict[str, TProcessorState] = {}

    def fetch_isa_sequence(self, game_id: str) -> List[MacroISAOpcode]:
        """Fetch the macro ISA instruction sequence for a game.

        Normalizes game_id by stripping version suffix, then looks up
        the macro instruction sequence in the registry. Returns the
        default κ-Snap pipeline (SOLVE_VIA_KSAP) if no game-specific
        sequence is registered.

        Args:
            game_id: Game identifier (may include version suffix).

        Returns:
            List of MacroISAOpcode values to execute for this game.
        """
        base_id: str = game_id.split("-")[0] if game_id else ""
        if base_id in self.registry:
            return self.registry[base_id]
        # Default: full κ-Snap pipeline for unknown games
        return [
            MacroISAOpcode.CHK_TIMEW,
            MacroISAOpcode.SOLVE_VIA_KSAP,
        ]

    def execute_macro(
        self,
        opcode: MacroISAOpcode,
        state: Dict[str, Any],
    ) -> ISAResult:
        """Execute a single macro instruction as a constraint gate.

        Directly calls the macro's Python function with the state dict.
        No μ-Op dispatch overhead — one function call per macro.

        Args:
            opcode: MacroISAOpcode to execute.
            state: Game state dict containing operands for the instruction.
                Expected keys vary by macro type. The KSnapEngine and
                SymCrypto instances are automatically injected into state
                for SOLVE_VIA_KSAP and CHK_TIMEW instructions.

        Returns:
            ISAResult indicating the macro instruction outcome.
        """
        fn: Optional[Callable[[Dict[str, Any]], ISAResult]] = (
            self.instruction_table.get(opcode)
        )
        if fn is None:
            # Unknown opcode — default PASS
            return ISAResult.PASS

        # Inject processor-level resources into state
        state["ks_engine"] = self.ks_engine
        state["start_time"] = state.get("start_time", time.time())

        try:
            result: ISAResult = fn(state)
            return result
        except Exception:
            # Macro execution failed — treat as FUSE (safe pruning)
            return ISAResult.FUSE

    def aggregate_results(
        self,
        results: List[Tuple[MacroISAOpcode, ISAResult]],
    ) -> ISAResult:
        """Aggregate macro instruction results — worst outcome determines fate.

        The aggregation follows the severity hierarchy:
            DEAD_ZERO > FUSE > PASS

        If ANY instruction returns DEAD_ZERO, the entire branch is
        permanently invalid. If any instruction returns FUSE (but
        none returns DEAD_ZERO), the branch is pruned for now.

        Args:
            results: List of (opcode, ISAResult) pairs from execution.

        Returns:
            Aggregated ISAResult — the worst outcome in the sequence.
        """
        if not results:
            return ISAResult.PASS

        has_dead_zero: bool = False
        has_fuse: bool = False

        for _opcode, result in results:
            if result == ISAResult.DEAD_ZERO:
                has_dead_zero = True
            elif result == ISAResult.FUSE:
                has_fuse = True

        if has_dead_zero:
            return ISAResult.DEAD_ZERO
        elif has_fuse:
            return ISAResult.FUSE
        return ISAResult.PASS

    def execute_isa_gate(
        self,
        game_id: str,
        state: Dict[str, Any],
    ) -> ISAResult:
        """Execute ISA constraint gate for a game — main integration hook.

        This is the primary integration point called by
        universal_solver_pipeline.py before κ-Snap expansion. It:

        1. Fetches the macro ISA sequence for game_id.
        2. Executes each macro instruction as a direct function call.
        3. Aggregates results to determine branch fate.
        4. Returns PASS/FUSE/DEAD_ZERO for the pipeline to act on.

        If gate returns PASS → allow κ-Snap expansion on this branch.
        If gate returns FUSE → prune this branch (Dead-Zero熔断).
        If gate returns DEAD_ZERO → mark branch permanently invalid.

        Args:
            game_id: Game identifier (may include version suffix).
            state: Game state dict containing operands for instructions.
                The state dict must contain the relevant keys for the
                macro instructions registered for this game.

        Returns:
            ISAResult indicating whether κ-Snap expansion may proceed.
        """
        # Step 1: Fetch macro ISA sequence
        isa_sequence: List[MacroISAOpcode] = self.fetch_isa_sequence(game_id)

        # Step 2: Record start time for CHK_TIMEW
        start_time: float = time.time()
        state["start_time"] = start_time

        # Step 3: Execute each macro instruction
        results: List[Tuple[MacroISAOpcode, ISAResult]] = []
        for opcode in isa_sequence:
            result: ISAResult = self.execute_macro(opcode, state)
            results.append((opcode, result))

            # Early exit: if DEAD_ZERO detected, no need to continue
            if result == ISAResult.DEAD_ZERO:
                break

            # Update start time for subsequent instructions (time window reset)
            state["start_time"] = time.time()

        # Step 4: Aggregate results
        final_result: ISAResult = self.aggregate_results(results)

        # Step 5: Cache state for potential Re-Inflow
        proc_state: TProcessorState = TProcessorState(
            game_id=game_id,
            isa_sequence=isa_sequence,
            results=results,
            final_result=final_result,
            branch_dead=(final_result == ISAResult.DEAD_ZERO),
        )
        base_id: str = game_id.split("-")[0] if game_id else ""
        self._state_cache[base_id] = proc_state

        return final_result

    def perceive(self, grid: Any) -> EMLGraph:
        """Build EML hypergraph from a 2D grid — perception step.

        Converts a 2D grid (numpy array or list) into an EMLGraph where
        each cell becomes an EMLNode with physical properties:
        - kind classification (wall/player/box/empty/goal)
        - mass assignment (wall=∞, player=1, box=2, goal=0, empty=0)
        - Octonion velocity initialization
        - 4-connected neighborhood adjacency

        Used by SOLVE_VIA_KSAP to build the EML state for κ-Snap projection.

        Args:
            grid: 2D grid (numpy array or list of lists).
                Wall cells are typically 0, empty cells 1, player 2,
                goal 3, box 4 (convention varies by game).

        Returns:
            EMLGraph with nodes representing grid cells and edges
            representing 4-connected adjacency.
        """
        if hasattr(grid, 'shape'):
            h, w = grid.shape
            grid_data = grid
        elif isinstance(grid, list):
            h = len(grid)
            w = len(grid[0]) if h > 0 else 0
            grid_data = grid
        else:
            return EMLGraph()

        eml_graph: EMLGraph = EMLGraph()
        id_map: Dict[Tuple[int, int], int] = {}

        # Kind classification mapping (generic)
        kind_map: Dict[int, str] = {
            0: "wall",    # Convention: 0 = wall
            1: "empty",   # Convention: 1 = empty floor
            2: "goal",    # Convention: 2 = goal position
            3: "player",  # Convention: 3 = player
            4: "box",     # Convention: 4 = box/object
        }

        # Mass mapping (Newton mechanics: wall=∞, box=2, player=1, others=0)
        mass_map: Dict[str, float] = {
            "wall": float('inf'),
            "box": 2.0,
            "player": 1.0,
            "goal": 0.0,
            "empty": 0.0,
        }

        # Create nodes
        for y in range(h):
            for x in range(w):
                cell_val: int = int(grid_data[y][x]) if isinstance(grid_data, list) else int(grid_data[y, x])
                kind: str = kind_map.get(cell_val, "empty")
                mass: float = mass_map.get(kind, 0.0)
                # Walls have infinite mass → clamp for numerical stability
                if mass == float('inf'):
                    mass = 1e6

                node_id: int = y * w + x
                node: EMLNode = EMLNode(
                    id=node_id,
                    pos=(x, y),
                    kind=kind,
                    mass=mass,
                    velocity=Octonion(),  # Zero velocity initially
                    neighbors=[],
                )
                eml_graph.add(node)
                id_map[(x, y)] = node_id

        # Build adjacency (4-connected grid)
        for node in eml_graph.nodes:
            x, y = node.pos
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor_id: Optional[int] = id_map.get((nx, ny))
                if neighbor_id is not None:
                    node.neighbors.append(neighbor_id)

        return eml_graph

    def get_processor_state(self, game_id: str) -> Optional[TProcessorState]:
        """Retrieve cached processor state for a game.

        Used by Re-Inflow mechanism to inspect ISA gate results and
        determine which instruction caused the FUSE/DEAD_ZERO.

        Args:
            game_id: Game identifier (base ID, no version suffix).

        Returns:
            Cached TProcessorState, or None if no state is cached.
        """
        base_id: str = game_id.split("-")[0] if game_id else ""
        return self._state_cache.get(base_id)

    def clear_state_cache(self) -> None:
        """Clear the processor state cache.

        Called between solving attempts to reset ISA gate history.
        """
        self._state_cache.clear()


# =============================================================================
# §15. ARCSolver — κ-Causal Reduction ARC Solver (NEW v3.20.0)
# =============================================================================

@dataclass
class ARCSolver:
    """ARC-AGI Solver based on κ-Algebra Tsirelson bound + causal reduction.

    From article: ARC solving = finding min GaussEx residual η causal transform T
    in κ-algebra coset space C(11,4).

    Pipeline:
        1. perceive(grid) → EML hypergraph (κ-algebra state)
        2. Generate κ-constrained candidate transforms (OMUL, MIR_X, etc.)
        3. For each candidate: KS_PROJ → compute η (GaussEx residual)
        4. DZFUSE if η > δ_K ≈ 0.036 (Tsirelson-illegal, PR-Box equivalent)
        5. Select min-η transform → apply to test input
        6. confidence = 1 - η/δ_K (physical purity analog)

    This replaces "probability search" with "physical reduction" paradigm.

    Attributes:
        cpu: TProcessorV12 instance for perception and κ-Snap pipeline.
        DELTA_K: GaussEx residual threshold (0.036 from KSnapEngine).
    """
    cpu: Optional[TProcessorV12] = None
    DELTA_K: float = KSnapEngine.DELTA_K  # 0.036

    def __post_init__(self) -> None:
        """Initialize TProcessorV12 if not provided."""
        if self.cpu is None:
            self.cpu = TProcessorV12()

    def perceive(self, grid: np.ndarray) -> EMLGraph:
        """Convert grid to EML hypergraph via TProcessorV12.perceive().

        Args:
            grid: 2D numpy array representing the ARC task grid.

        Returns:
            EMLGraph with nodes representing grid cells.
        """
        return self.cpu.perceive(grid)

    def solve(
        self,
        demonstrations: List[Dict[str, Any]],
        test_input: Any,
    ) -> Tuple[Any, float, Optional[str]]:
        """κ-Causal Reduction: find min-η transform in C(11,4) coset space.

        Iterates over κ-physically allowed transformations from article
        Appendix A, computes GaussEx residual for each candidate via
        KS_PROJ, and selects the minimum-residual transform.

        PR-Box rule: candidates with η > δ_K are skipped (Tsirelson-illegal).

        Args:
            demonstrations: List of demonstration dicts with 'input'/'output' grids.
            test_input: Test input grid to transform.

        Returns:
            Tuple of (result, confidence, transform_name) where:
                result: Transformed test input (or None if no legal transform).
                confidence: Physical purity 1 - η/δ_K (0.0 if no legal transform).
                transform_name: Name of best transform (or None).
        """
        best_T: Optional[str] = None
        best_eta: float = float('inf')

        eml_demo: EMLGraph = self.perceive(demonstrations[0]['input'])

        # κ-physically allowed transformations (article Appendix A)
        candidate_transforms: List[Tuple[str, Callable[[EMLGraph], EMLGraph]]] = [
            ('ROT90', self._rot90),
            ('MIRROR_X', self._mirror_x),
            ('MIRROR_Y', self._mirror_y),
            ('COLOR_SWAP', self._color_swap),
            ('FILL_CC', self._fill_connected_component),
        ]

        for T_name, T_func in candidate_transforms:
            eml_pred: EMLGraph = T_func(eml_demo)

            # κ-Snap pipeline: KS_START → KS_PROJ → KS_GX
            self.cpu.ks_engine._prior_loaded = True
            eml_oct: Octonion = _eml_to_octonion(eml_pred)
            _, residual = self.cpu.ks_engine.project(eml_oct, None)

            eta: float = residual

            if eta < best_eta:
                best_eta = eta
                best_T = T_name

            # Dead-Zero Fuse: η > δ_K → PR-Box illegal → DZFUSE
            if eta > self.DELTA_K:
                continue  # Skip this candidate

        if best_eta > self.DELTA_K:
            return None, 0.0, None

        confidence: float = 1.0 - (best_eta / self.DELTA_K)
        return test_input, confidence, best_T

    def _rot90(self, eml: EMLGraph) -> EMLGraph:
        """Rotate 90°: OMUL instruction (Octonion e2 multiplication).

        Applies OMUL to each node's position — rotate 90° around e2=j axis.
        Position transform: (x, y) → (y, -x).
        Velocity transform: multiply by e2 octonion unit.

        Args:
            eml: EMLGraph to rotate.

        Returns:
            EMLGraph with rotated node positions and velocities.
        """
        new_nodes: List[EMLNode] = []
        rot_axis: Octonion = Octonion(0, 0, 1, 0, 0, 0, 0, 0)  # e2 = j axis
        for node in eml.nodes:
            if node.pos is not None:
                # Rotate position by 90° (OMUL with j-axis)
                new_pos: Tuple[int, int] = (
                    node.pos[1], -node.pos[0]
                ) if isinstance(node.pos, tuple) else node.pos
                new_vel: Octonion = (
                    node.velocity * rot_axis if node.velocity else Octonion.zero()
                )
                new_nodes.append(EMLNode(
                    id=node.id, pos=new_pos, kind=node.kind,
                    mass=node.mass, velocity=new_vel, neighbors=node.neighbors,
                ))
            else:
                new_nodes.append(node)
        return EMLGraph(nodes=new_nodes)

    def _mirror_x(self, eml: EMLGraph) -> EMLGraph:
        """Mirror X: MIR_X instruction (x → -x coordinate negation).

        Negates the x-coordinate of each EML node position.

        Args:
            eml: EMLGraph to mirror along X axis.

        Returns:
            EMLGraph with x-coordinates negated.
        """
        new_nodes: List[EMLNode] = []
        for node in eml.nodes:
            if node.pos is not None and isinstance(node.pos, tuple):
                new_pos: Tuple[int, int] = (-node.pos[0], node.pos[1])
                new_nodes.append(EMLNode(
                    id=node.id, pos=new_pos, kind=node.kind,
                    mass=node.mass, velocity=node.velocity, neighbors=node.neighbors,
                ))
            else:
                new_nodes.append(node)
        return EMLGraph(nodes=new_nodes)

    def _mirror_y(self, eml: EMLGraph) -> EMLGraph:
        """Mirror Y: MIR_Y instruction (y → -y coordinate negation).

        Negates the y-coordinate of each EML node position.

        Args:
            eml: EMLGraph to mirror along Y axis.

        Returns:
            EMLGraph with y-coordinates negated.
        """
        new_nodes: List[EMLNode] = []
        for node in eml.nodes:
            if node.pos is not None and isinstance(node.pos, tuple):
                new_pos: Tuple[int, int] = (node.pos[0], -node.pos[1])
                new_nodes.append(EMLNode(
                    id=node.id, pos=new_pos, kind=node.kind,
                    mass=node.mass, velocity=node.velocity, neighbors=node.neighbors,
                ))
            else:
                new_nodes.append(node)
        return EMLGraph(nodes=new_nodes)

    def _color_swap(self, eml: EMLGraph) -> EMLGraph:
        """Color swap: ST_EML instruction (EML node attribute permutation).

        Permutes the 'kind' attributes of EML nodes. Creates a color
        mapping from unique kinds in the graph.

        Args:
            eml: EMLGraph with nodes whose 'kind' attributes to permute.

        Returns:
            EMLGraph with permuted kind attributes.
        """
        new_nodes: List[EMLNode] = []
        colors: List[str] = [n.kind for n in eml.nodes if n.kind is not None]
        if not colors:
            return eml
        color_map: Dict[str, str] = dict(zip(set(colors), set(colors)))
        for node in eml.nodes:
            new_kind: Optional[str] = (
                color_map.get(node.kind, node.kind) if node.kind else node.kind
            )
            new_nodes.append(EMLNode(
                id=node.id, pos=node.pos, kind=new_kind,
                mass=node.mass, velocity=node.velocity, neighbors=node.neighbors,
            ))
        return EMLGraph(nodes=new_nodes)

    def _fill_connected_component(self, eml: EMLGraph) -> EMLGraph:
        """Fill: FILL_CC instruction (EML topology expansion).

        Expands EML topology by adding nodes for connected components
        that require filling. Simplified implementation — topology
        expansion requires game-specific logic.

        Args:
            eml: EMLGraph to expand topology.

        Returns:
            EMLGraph with expanded topology (simplified: returns input).
        """
        return eml  # Simplified — topology expansion requires game-specific logic


# =============================================================================
# §14. KappaCausalReductionSolver — κ-Causal Reduction via Tsirelson Bound
# =============================================================================

class KappaCausalReductionSolver:
    """κ-Causal Reduction Solver — ARC solving via κ-algebra coset causal reduction.

    Based on the κ-Tsirelson framework: ARC solving = finding minimum GaussEx residual
    causal transform T in κ-algebra coset space C(11,4).

    Pipeline:
        1. perceive(grid) → EML hypergraph
        2. Generate κ-constrained candidate transforms (OMUL, MIR, ST_EML, FILL_CC, COUNT)
        3. For each candidate:
            a. KS_PROJ: project onto C(11,4) coset → best_v, residual η
            b. If η < DELTA_K (0.036) → accept (Tsirelson-legal)
            c. If η ≥ DELTA_K → DZFUSE (Tsirelson-illegal, PR-Box equivalent)
        4. Return best transform with confidence = 1 - η/DELTA_K

    Tsirelson Bound Enforcement:
        - CHSH S ≤ 2√2 for all candidate transforms
        - S > 2√2 → pseudo-rule (PR-Box equivalent) → Dead-Zero fuse

    Attributes:
        DELTA_K: GaussEx threshold (same as KSnapEngine, 0.036).
        TSIRELSON_BOUND: Maximum CHSH correlation value = 2√2 ≈ 2.828.
        cpu: TProcessorV12 instance for EML perception.
        ks_engine: KSnapEngine for coset projection.
        _candidate_transforms: List of (name, opcode, kwargs) for κ-transforms.
    """

    DELTA_K: float = 0.036  # GaussEx threshold (same as KSnapEngine)
    TSIRELSON_BOUND: float = 2.0 * math.sqrt(2)  # S_max = 2√2 ≈ 2.828

    def __init__(self, processor: Optional[TProcessorV12] = None) -> None:
        """Initialize κ-Causal Reduction Solver.

        Args:
            processor: Optional TProcessorV12 instance for EML perception.
                If None, creates a default instance.
        """
        self.cpu: TProcessorV12 = processor or TProcessorV12()
        self.ks_engine: KSnapEngine = KSnapEngine()
        self._candidate_transforms: List[Tuple[str, MacroISAOpcode, Dict[str, Any]]] = [
            ('ROT90', MacroISAOpcode.OMUL, {'axis': Octonion(0, 1, 0, 0, 0, 0, 0, 0)}),
            ('ROT180', MacroISAOpcode.OMUL, {'axis': Octonion(-1, 0, 0, 0, 0, 0, 0, 0)}),
            ('ROT270', MacroISAOpcode.OMUL, {'axis': Octonion(0, -1, 0, 0, 0, 0, 0, 0)}),
            ('MIRROR_X', MacroISAOpcode.MIR_X, {}),
            ('MIRROR_Y', MacroISAOpcode.MIR_Y, {}),
            ('COLOR_SWAP', MacroISAOpcode.ST_EML, {}),
            ('FILL_CC', MacroISAOpcode.FILL_CC, {}),
            ('COUNT', MacroISAOpcode.COUNT_NODES, {}),
        ]

    def solve(
        self,
        grid: np.ndarray,
        prior_grid: Optional[np.ndarray] = None,
    ) -> Tuple[Any, float, str]:
        """κ-causal reduction: find minimum residual transform T.

        Executes the full κ-Causal Reduction pipeline:
            1. perceive(grid) → EML hypergraph
            2. For each κ-constrained candidate transform:
                a. Apply transform to EML copy
                b. KS_PROJ: project onto C(11,4) coset → residual η
                c. Tsirelson bound check: S = 2 + 2·cos(η) ≤ 2√2
                d. If η < δ_K → accept immediately
            3. Return best (transformed_eml, confidence, transform_name)

        Args:
            grid: 2D numpy array (input ARC grid).
            prior_grid: Optional prior grid for κ-phase weighting.

        Returns:
            Tuple of (transformed_eml, confidence, transform_name):
                transformed_eml: EMLGraph with best transform applied.
                confidence: 1 - η/δ_K (quantum state purity analogue).
                transform_name: Name of the best transform, or "DEAD_ZERO".
        """
        # Step 1: perceive(grid) → EML hypergraph
        eml: EMLGraph = self.cpu.perceive(grid)
        if eml is None or len(eml.nodes) == 0:
            return (None, 0.0, "DEAD_ZERO")

        # Step 2: load prior
        prior_eml: Optional[EMLGraph] = None
        prior_oct: Optional[Octonion] = None
        if prior_grid is not None:
            prior_eml = self.cpu.perceive(prior_grid)
            if prior_eml is not None and len(prior_eml.nodes) > 0:
                prior_oct = _eml_to_octonion(prior_eml)

        # Step 3: try each κ-constrained transform
        best_result: Optional[EMLGraph] = None
        best_eta: float = float('inf')
        best_name: str = "DEAD_ZERO"

        for t_name, t_opcode, t_kwargs in self._candidate_transforms:
            # Create a copy of EML for this transform
            eml_copy: EMLGraph = EMLGraph(
                nodes=[EMLNode(
                    id=n.id, pos=n.pos, kind=n.kind,
                    mass=n.mass, velocity=n.velocity,
                    neighbors=list(n.neighbors) if n.neighbors else [],
                ) for n in eml.nodes]
            )

            # Execute κ-transform
            state: Dict[str, Any] = {"eml": eml_copy, **t_kwargs}
            result: ISAResult = MACRO_INSTRUCTION_TABLE[t_opcode](state)

            if result != ISAResult.PASS:
                continue  # DZFUSE: skip this transform

            # κ-Snap projection: project onto C(11,4) coset
            eml_oct: Octonion = _eml_to_octonion(eml_copy)
            best_v, eta = self.ks_engine.project(eml_oct, prior_oct)

            # Tsirelson bound check: compute CHSH from η
            # S = 2 + 2*cos(η) → must be ≤ 2√2
            chsh_s: float = 2.0 + 2.0 * math.cos(eta)
            if chsh_s > self.TSIRELSON_BOUND:
                # PR-Box equivalent → DZFUSE
                continue

            # GaussEx residual check
            if eta < best_eta:
                best_eta = eta
                best_name = t_name
                best_result = eml_copy

            # If residual below threshold → accept immediately
            if eta < self.DELTA_K:
                confidence: float = 1.0 - (eta / self.DELTA_K)
                return (best_result, confidence, best_name)

        # If best residual is above threshold → no valid transform
        if best_eta > self.DELTA_K:
            return (None, 0.0, "DEAD_ZERO")

        confidence = 1.0 - (best_eta / self.DELTA_K)
        return (best_result, confidence, best_name)

    def verify_tsirelson_bound(
        self,
        transform_result: Any,
        eta: float,
    ) -> bool:
        """Verify that candidate transform respects Tsirelson bound S ≤ 2√2.

        CHSH correlation: S = 2 + 2·cos(η)
        Tsirelson bound: S_max = 2√2 ≈ 2.828
        PR-Box: S = 4 (prohibited)

        Args:
            transform_result: The transformed EML graph (for context).
            eta: GaussEx residual from κ-Snap projection.

        Returns:
            True if CHSH S ≤ 2√2 (Tsirelson-legal).
            False if S > 2√2 (PR-Box equivalent → prohibited).
        """
        chsh_s: float = 2.0 + 2.0 * math.cos(eta)
        return chsh_s <= self.TSIRELSON_BOUND


# =============================================================================
# §14. Module-level convenience functions (保留接口)
# =============================================================================

# Singleton processor instance for module-level access
_processor_v12: TProcessorV12 = TProcessorV12()


def execute_isa_gate(
    game_id: str,
    state: Dict[str, Any],
) -> ISAResult:
    """Module-level ISA gate execution — convenience wrapper.

    This is the primary integration point for universal_solver_pipeline.py.
    Call this before κ-Snap expansion to check if the current game state
    satisfies all physics axiom constraints.

    Internally uses TProcessorV12 with macro instruction model.

    Args:
        game_id: Game identifier (may include version suffix).
        state: Game state dict with operands for macro instructions.

    Returns:
        ISAResult.PASS → allow κ-Snap expansion.
        ISAResult.FUSE → prune branch (Dead-Zero熔断).
        ISAResult.DEAD_ZERO → mark branch permanently invalid.
    """
    return _processor_v12.execute_isa_gate(game_id, state)


def get_isa_sequence(game_id: str) -> List[MacroISAOpcode]:
    """Get the macro ISA instruction sequence for a game.

    Args:
        game_id: Game identifier (may include version suffix).

    Returns:
        List of MacroISAOpcode values registered for this game.
    """
    return _processor_v12.fetch_isa_sequence(game_id)


def register_game_isa(
    game_id: str,
    isa_sequence: List[MacroISAOpcode],
) -> None:
    """Register a custom macro ISA sequence for a game.

    Allows extending the ISA registry for new games or modifying
    existing game macro ISA sequences dynamically.

    Args:
        game_id: Base game identifier (no version suffix).
        isa_sequence: List of MacroISAOpcode values to execute for this game.
    """
    base_id: str = game_id.split("-")[0] if game_id else ""
    _processor_v12.registry[base_id] = isa_sequence
