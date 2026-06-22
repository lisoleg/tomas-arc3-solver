# -*- coding: utf-8 -*-
"""
GAT（广义代数理论）公理系统 — DSL 原语形式化（从 tomas-agi 吸收）
============================================================

吸收来源：tomas-agi/tomas_agi/sim/gat_axioms.py (v3.12, 389 行)
适配改动：
  - 作为 src/core/gatlab_interface.py 的 Pure Python fallback
  - 当 GATLab 不可用时自动切换
  - 保留 ArcDSL_GAT 类（DSL 原语 GAT 签名）
  - 添加 TOMAS 30-DSL 原语的 GAT 签名

设计：
  - GATTheory: GAT 理论基类（Sort / Operation / Axiom / free_model / theory_map）
  - ArcDSL_GAT: ARC DSL 的 GAT 形式化
  - 可作为 gatlab_interface.py 的轻量替代（无需 Julia 运行时）
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════╗
# ║               GATTheory — GAT 理论基类                              ║
# ╚═══════════════════════════════════════════════════╝

class GATTheory:
    """GAT（广义代数理论）基类。

    形式化一个多类代数理论，包含：
    - Sort:       类型 / 集合
    - Operation:  多参数运算
    - Axiom:      等式公理

    可构造自由模型和理论态射。
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.sorts: Dict[str, str] = {}
        self.operations: List[Dict[str, Any]] = []
        self.axioms: List[Dict[str, Any]] = []

    def add_sort(self, name: str, description: str = "") -> str:
        """添加类型（Sort）。"""
        if name in self.sorts:
            raise ValueError(f"Sort '{name}' 在理论 '{self.name}' 中已存在")
        self.sorts[name] = description
        return name

    def add_operation(
        self, name: str, domain: List[str], codomain: str,
    ) -> Dict[str, Any]:
        """添加操作（多参数运算）。"""
        for s in domain:
            if s not in self.sorts:
                raise ValueError(f"操作 '{name}' 的 domain 类型 '{s}' 未定义")
        if codomain not in self.sorts:
            raise ValueError(f"操作 '{name}' 的 codomain 类型 '{codomain}' 未定义")
        op = {"name": name, "domain": list(domain), "codomain": codomain}
        self.operations.append(op)
        return op

    def add_axiom(self, name: str, equation: str) -> Dict[str, Any]:
        """添加公理（等式）。"""
        axiom = {"name": name, "equation": str(equation)}
        self.axioms.append(axiom)
        return axiom

    def free_model(self) -> Dict[str, Any]:
        """构造自由模型（term algebra）。"""
        model: Dict[str, Any] = {
            "theory_name": self.name,
            "sorts": dict(self.sorts),
            "sort_elements": {},
            "operations": [dict(op) for op in self.operations],
            "axioms": [dict(ax) for ax in self.axioms],
            "constant_terms": {},
            "derived_term_count": 0,
        }
        for sort_name in self.sorts:
            elements: List[str] = []
            for op in self.operations:
                if op["codomain"] == sort_name and not op["domain"]:
                    elements.append(op["name"])
            model["sort_elements"][sort_name] = elements
        for op in self.operations:
            if not op["domain"]:
                model["constant_terms"][op["name"]] = op["codomain"]
        model["derived_term_count"] = sum(
            len(elems) for elems in model["sort_elements"].values()
        )
        return model

    def theory_map(
        self, target_theory: "GATTheory", mapping: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构造理论态射 self → target_theory。"""
        sort_map = mapping.get("sorts", {})
        op_map = mapping.get("operations", {})
        morphism: Dict[str, Any] = {
            "source": self.name,
            "target": target_theory.name,
            "sort_map": sort_map,
            "operation_map": op_map,
        }
        return morphism

    def check_term(self, term: Dict) -> bool:
        """检查 term 是否符合该理论的 typng rules。"""
        # 简化实现：检查 term["op"] 是否在 operations 中
        op_name = term.get("op", "")
        return any(op["name"] == op_name for op in self.operations)

    def pretty_print(self) -> str:
        """格式化输出理论。"""
        lines = [
            f"GAT Theory: {self.name}",
            f"  Sorts ({len(self.sorts)}):",
        ]
        for s, desc in self.sorts.items():
            lines.append(f"    {s}: {desc}")
        lines.append(f"  Operations ({len(self.operations)}):")
        for op in self.operations:
            dom = " × ".join(op["domain"]) if op["domain"] else "1"
            lines.append(f"    {op['name']}: {dom} → {op['codomain']}")
        lines.append(f"  Axioms ({len(self.axioms)}):")
        for ax in self.axioms:
            lines.append(f"    {ax['name']}: {ax['equation']}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════╗
# ║          ArcDSL_GAT — ARC DSL 的 GAT 形式化                       ║
# ╚═══════════════════════════════════════════════════╝

class ArcDSL_GAT(GATTheory):
    """ARC DSL 原语的 GAT 形式化（Pure Python，无 Julia 依赖）。

    30 个 DSL 原语（TOMAS v2.3）的 GAT 签名：
      - Grid 类型：输入 / 输出网格
      - 操作签名：每个 DSL 原语的类型签名
      - 等式公理：DSL 原语的可判定等式（如 rotate 逆元）

    用法：
        theory = ArcDSL_GAT()
        model = theory.free_model()
        # 检查程序项是否类型正确
        theory.check_program(program)
    """

    def __init__(self) -> None:
        super().__init__("ArcDSL_GAT_v2.3")

        # ── Sorts ─────────────────────────────────────────
        self.add_sort("Grid",  "ARC 输入输出网格（变长，色值 0-9）")
        self.add_sort("Int",   "整数参数")
        self.add_sort("Bool",  "布尔值")
        self.add_sort("Prog",  "DSL 程序（操作序列）")
        self.add_sort("Topo",  "拓扑不变量（topo_hash, Betti0...）")

        # ── 基础操作 ─────────────────────────────────────
        self.add_operation("id",     [], "Grid")                 # 恒等
        self.add_operation("resize", ["Grid", "Int", "Int"], "Grid")
        self.add_operation("crop",   ["Grid", "Int", "Int", "Int", "Int"], "Grid")
        self.add_operation("pad",    ["Grid", "Int", "Int", "Int", "Int", "Int"], "Grid")
        self.add_operation("mirror_h", ["Grid"], "Grid")
        self.add_operation("mirror_v", ["Grid"], "Grid")
        self.add_operation("rotate", ["Grid", "Int"], "Grid")   # Int ∈ {90,180,270,-90,-180,-270}
        self.add_operation("move",    ["Grid", "Int", "Int"], "Grid")
        self.add_operation("copy",    ["Grid", "Int", "Int", "Int", "Int"], "Grid")
        self.add_operation("gravity", ["Grid", "Int"], "Grid")  # Int: direction

        # ── 高级操作 ─────────────────────────────────────
        self.add_operation("solve_sudoku", ["Grid"], "Grid")
        self.add_operation("draw_rect",   ["Grid", "Int", "Int", "Int", "Int", "Int"], "Grid")
        self.add_operation("flood_fill",  ["Grid", "Int", "Int", "Int"], "Grid")
        self.add_operation("connected_components", ["Grid"], "Grid")
        self.add_operation("extract_mask", ["Grid", "Int"], "Grid")
        self.add_operation("apply_mask",  ["Grid", "Grid"], "Grid")

        # ── 拓扑不变量操作 ───────────────────────────────
        self.add_operation("topo_hash", ["Grid"], "Topo")
        self.add_operation("betti0",    ["Grid"], "Int")

        # ── 程序组合 ─────────────────────────────────────
        self.add_operation("seq", ["Prog", "Prog"], "Prog")   # 顺序组合
        self.add_operation("branch", ["Bool", "Prog", "Prog"], "Prog")

        # ── 公理（等式）───────────────────────────────────
        # rotate 逆元
        self.add_axiom(
            "rotate_inverse",
            "seq(rotate(g, k), rotate(g, -k)) == id(g)",
        )
        # mirror 对合
        self.add_axiom(
            "mirror_h_involution",
            "seq(mirror_h(g), mirror_h(g)) == id(g)",
        )
        self.add_axiom(
            "mirror_v_involution",
            "seq(mirror_v(g), mirror_v(g)) == id(g)",
        )
        # resize 单位元
        self.add_axiom(
            "resize_id",
            "let (h, w) = shape(g) in resize(g, h, w) == g",
        )

    def check_program(self, program: List[Dict]) -> Tuple[bool, str]:
        """检查 DSL 程序的类型正确性。

        参数：
            program: [{op: str, args: [...]}]

        返回：
            (is_valid, error_msg)
        """
        stack: List[str] = ["Grid"]  # 类型栈（简化：仅跟踪输入输出 Grid）

        op_names = {op["name"] for op in self.operations}

        for i, step in enumerate(program):
            op_name = step.get("op", "")
            if op_name not in op_names:
                return False, f"Step {i}: unknown op '{op_name}'"

            # 简化类型检查：仅检查 Grid→Grid 操作
            op = next(o for o in self.operations if o["name"] == op_name)
            # 检查 domain 是否匹配（简化）
            expected_args = len(op["domain"])
            actual_args = len(step.get("args", []))
            # 允许误差：整数参数可缺省
            if op_name in ("rotate", "gravity") and actual_args < expected_args:
                pass  # 使用默认值
            elif actual_args < expected_args:
                return (False,
                        f"Step {i}: op '{op_name}' expects {expected_args} args, "
                        f"got {actual_args}")

        return True, ""

    def free_model(self) -> Dict[str, Any]:
        """扩展自由模型：包含 DSL 原语的 MDL 代价。"""
        model = super().free_model()

        # 附加 MDL 代价（来自 dsl_primitives.py）
        mdl_costs = {
            "id":         1,
            "resize":     5,
            "mirror_h":   5,
            "mirror_v":   5,
            "rotate":     5,
            "move":       5,
            "copy":       5,
            "gravity":    5,
            "flood_fill": 10,
            "solve_sudoku": 10,
        }
        model["mdl_costs"] = mdl_costs
        model["total_mdl"] = lambda prog: sum(
            mdl_costs.get(step.get("op", "id"), 5)
            for step in (prog if isinstance(prog, list) else [])
        )
        return model


# ════════════════════════════════════════════════════╗
# ║          OctonionGAT — 八元数超图的 GAT 形式化                      ║
# ╚═══════════════════════════════════════════════════╝

class OctonionGAT(GATTheory):
    """八元数超图编码的 GAT 形式化。

    将 OctonionHyperEdge 的操作形式化为 GAT 理论：
      - O: 八元数类型（e0..e7）
      - encode_grid:   Grid → O
      - decode_to_grid: O → Grid
      - 公理：encode(decode(o)) = o（在特定条件下）
    """

    def __init__(self) -> None:
        super().__init__("OctonionGAT")

        self.add_sort("O",    "八元数 O ≈ R^8")
        self.add_sort("Grid", "ARC 网格")
        self.add_sort("H",    "超图 HyperGraph")

        self.add_operation("encode_grid",   ["Grid"], "O")
        self.add_operation("decode_to_grid", ["O"], "Grid")
        self.add_operation("topo_invariants", ["O"], "H")
        self.add_operation("oct_mult", ["O", "O"], "O")  # 八元数乘法（非结合）

        # 公理：八元数乘法非结合（Moufang 恒等式）
        self.add_axiom(
            "oct_nonassoc_example",
            "exists a,b,c. (a*b)*c != a*(b*c)",
        )
        # 公理：encode → decode 往返（网格分辨率不变时）
        self.add_axiom(
            "encode_decode_roundtrip",
            "forall g. shape(decode(encode_grid(g))) == shape(g)",
        )


# ════════════════════════════════════════════════════╗
# ║               工具函数                                           ║
# ╚═══════════════════════════════════════════════════╝

def prove_equality(
    theory: GATTheory,
    term_a: Dict,
    term_b: Dict,
    max_depth: int = 10,
) -> Tuple[bool, List[str]]:
    """在给定 GAT 理论中证明两个项相等。

    简化实现：检查项是否完全相同（语法等式）。
    完整实现需要：项重写引擎 + 公理展开。
    """
    if term_a == term_b:
        return True, ["refl"]
    # 尝试用公理重写
    for ax in theory.axioms:
        # 极简：检查公理是否"覆盖"了两个项
        # 实际需 unification 引擎
        pass
    return False, []


def gat_fallback_for_gatlab(gatlab_available: bool) -> GATTheory:
    """GATLab 不可用时返回 Pure Python GAT 理论。

    用法（在 gatlab_interface.py 中）：
        try:
            from .gatlab_interface import GATLabBridge
            theory = GATLabBridge().load_theory("ArcDSL")
        except ImportError:
            theory = gat_fallback_for_gatlab(False)
    """
    if gatlab_available:
        # 尝试导入真实 GATLab 接口
        try:
            from .gatlab_interface import GATLabBridge  # type: ignore
            return GATLabBridge().load_theory("ArcDSL")
        except ImportError:
            pass
    return ArcDSL_GAT()


if __name__ == "__main__":
    # 简单测试
    theory = ArcDSL_GAT()
    print(theory.pretty_print())
    print()

    # 检查示例程序
    sample_prog = [
        {"op": "mirror_h", "args": []},
        {"op": "rotate", "args": [90]},
    ]
    valid, msg = theory.check_program(sample_prog)
    print(f"Program valid: {valid}")
    if not valid:
        print(f"  Error: {msg}")

    # 自由模型
    model = theory.free_model()
    print(f"\nFree model: {model['theory_name']}")
    print(f"  Sorts: {list(model['sorts'].keys())}")
    print(f"  Operations: {len(model['operations'])}")
