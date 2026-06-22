# -*- coding: utf-8 -*-
"""
数学序列工具（统一 Luzhao DNA 数学函数）
======================================

统一来自两个仓库的 Fibonacci / Lucas / Bagua 序列实现：
  - tomas-agi:     luzhao_dna.py（市场波浪基因编码）
  - tomas-arc3:    src/core/luzhao_dna.py（网格拓扑不变量）

此模块提供纯数学工具函数，供两处引用，消除重复代码。

包含：
  - Fibonacci / Lucas 数列（生成 + 缓存）
  - Bagua（八卦）常数（先天 / 后天）
  - 八元数编码辅助函数
  - 拓扑不变量计算（Betti0, topo_hash 简化版）
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import hashlib


# ════════════════════════════════════════════════════╗
# ║          Fibonacci / Lucas 数列                                 ║
# ╚═══════════════════════════════════════════════════╝

_FIB_CACHE: List[int] = [0, 1]   # F[0]=0, F[1]=1
_LUC_CACHE: List[int] = [2, 1]   # L[0]=2, L[1]=1


def fibonacci(n: int) -> int:
    """返回第 n 个 Fibonacci 数（F[0]=0, F[1]=1）。"""
    if n < 0:
        raise ValueError("fibonacci(n) requires n >= 0")
    while len(_FIB_CACHE) <= n:
        _FIB_CACHE.append(_FIB_CACHE[-1] + _FIB_CACHE[-2])
    return _FIB_CACHE[n]


def fibonacci_list(n: int) -> List[int]:
    """返回前 n 个 Fibonacci 数 [F[0], ..., F[n-1]]。"""
    fibonacci(n)  # 确保缓存
    return list(_FIB_CACHE[:n])


def fibonacci_mod(n: int, mod: int) -> int:
    """返回 F[n] mod mod（Pisano 周期优化）。"""
    if mod <= 0:
        raise ValueError("mod must be positive")
    if mod == 1:
        return 0
    # 简化：直接算（适用于小 n）
    return fibonacci(n) % mod


def lucas(n: int) -> int:
    """返回第 n 个 Lucas 数（L[0]=2, L[1]=1）。"""
    if n < 0:
        raise ValueError("lucas(n) requires n >= 0")
    while len(_LUC_CACHE) <= n:
        _LUC_CACHE.append(_LUC_CACHE[-1] + _LUC_CACHE[-2])
    return _LUC_CACHE[n]


def lucas_list(n: int) -> List[int]:
    """返回前 n 个 Lucas 数 [L[0], ..., L[n-1]]。"""
    lucas(n)
    return list(_LUC_CACHE[:n])


def fibonacci_lucas_identity(n: int) -> bool:
    """验证恒等式 L[n] = F[n-1] + F[n+1]。"""
    if n < 1:
        return True
    return lucas(n) == fibonacci(n - 1) + fibonacci(n + 1)


# ════════════════════════════════════════════════════╗
# ║          Bagua（八卦）常数                                     ║
# ╚═══════════════════════════════════════════════════╝

# 先天八卦序数（伏羲八卦次序：乾1、兑2、离3、震4、巽5、坎6、艮7、坤8）
BAGUA_XIANTIAN: Dict[str, int] = {
    "乾": 1, "兑": 2, "离": 3, "震": 4,
    "巽": 5, "坎": 6, "艮": 7, "坤": 8,
}
BAGUA_XIANTIAN_INV: Dict[int, str] = {v: k for k, v in BAGUA_XIANTIAN.items()}

# 后天八卦序数（文王八卦：坎1、坤2、震3、巽4、中5、乾6、兑7、艮8、离9）
BAGUA_HOUTIAN: Dict[str, int] = {
    "坎": 1, "坤": 2, "震": 3, "巽": 4,
    "中": 5, "乾": 6, "兑": 7, "艮": 8, "离": 9,
}
BAGUA_HOUTIAN_INV: Dict[int, str] = {v: k for k, v in BAGUA_HOUTIAN.items()}


def bagua_constant(trigram: str, mode: str = "xiantian") -> int:
    """返回八卦常数。

    参数：
        trigram: 八卦名称（如 "乾", "兑"）
        mode:    "xiantian"（先天）或 "houtian"（后天）

    返回：
        序数（1-8 或 1-9）
    """
    if mode == "xiantian":
        return BAGUA_XIANTIAN.get(trigram, 0)
    elif mode == "houtian":
        return BAGUA_HOUTIAN.get(trigram, 0)
    else:
        raise ValueError(f"Unknown Bagua mode: {mode}")


def bagua_to_color(trigram: str) -> int:
    """将八卦映射为 ARC 色值（0-9）。

    映射规则（自定义，可调整）：
        乾→7(白), 坤→0(黑), 震→1(蓝), 巽→2(红)...
    """
    mapping = {
        "乾": 7, "坤": 0, "震": 1, "巽": 2,
        "坎": 4, "离": 5, "艮": 3, "兑": 6,
    }
    return mapping.get(trigram, 0)


def bagua_from_grid(grid: List[List[int]]) -> List[int]:
    """从网格中提取"八卦签名"（简化）。

    将 3x3 局部模式映射到最近的八卦三角。
    返回 8 维向量（每个八卦的匹配度）。
    """
    # 简化：统计网格中的色值分布，映射到八卦
    from collections import Counter
    flat = [c for row in grid for c in row]
    cnt = Counter(flat)
    total = max(sum(cnt.values()), 1)
    # 按色值 0-7 映射为八卦签名
    signature = []
    for i in range(8):
        ratio = cnt.get(i, 0) / total
        signature.append(int(ratio * 100))
    return signature


# ════════════════════════════════════════════════════╗
# ║          八元数编码辅助                                           ║
# ╚═══════════════════════════════════════════════════╝

def grid_to_octonion_components(
    grid: List[List[int]],
    x: int = 0, y: int = 0,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """将网格（位置 (x,y) 处）编码为八元数 8 个分量。

    简化版（供 gat_axioms.py / luzhao_dna.py 引用）：
      e0 = real_mask:   色值 0 的比例
      e1 = x:           归一化 x 坐标
      e2 = y:           归一化 y 坐标
      e3 = color_phase: 色值相位
      e4 = component_id: 连通分量 ID
      e5 = symmetry:    对称性指标
      e6 = boundary:    边界长度
      e7 = area_ratio:  面积比

    完整实现在 src/core/octonion_hyperedge.py。
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 1
    cx = grid[y][x] if 0 <= y < h and 0 <= x < w else 0

    e0 = (w * h - sum(row.count(0) for row in grid)) / max(w * h, 1)
    e1 = x / max(w - 1, 1)
    e2 = y / max(h - 1, 1)
    e3 = (cx % 16) / 16.0
    e4 = float((x + y * w) % 8)
    e5 = 1.0 if cx == 0 else 0.0  # 简化对称
    e6 = float(sum(1 for r in grid for c in r if c != 0))
    e7 = e0  # 面积比 ≈ 非零比

    return (e0, e1, e2, e3, e4, e5, e6, e7)


def octonion_distance(o1: Tuple, o2: Tuple) -> float:
    """计算两个八元数之间的欧几里得距离。"""
    return sum((a - b) ** 2 for a, b in zip(o1, o2)) ** 0.5


# ════════════════════════════════════════════════════╗
# ║          拓扑不变量（简化版）                                     ║
# ╚═══════════════════════════════════════════════════╝

def betti0(grid: List[List[int]]) -> int:
    """计算 Betti-0（连通分量数）。

    简化实现（完整实现在 hypergraph.py）。
    使用 BFS 染色法。
    """
    if not grid or not grid[0]:
        return 0
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = 0

    from collections import deque
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != 0:
                # BFS
                components += 1
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (0 <= nr < h and 0 <= nc < w
                                and not visited[nr][nc]
                                and grid[nr][nc] != 0):
                            visited[nr][nc] = True
                            q.append((nr, nc))
    return components


def topo_hash(grid: List[List[int]]) -> str:
    """计算拓扑哈希（内容感知）。

    简化实现：基于 (nonzero_count, betti0, color_histogram) 的哈希。
    完整实现在 topo_hash.py。
    """
    if not grid:
        return "0" * 16
    h, w = len(grid), len(grid[0])
    nonzero = sum(1 for r in grid for c in r if c != 0)
    b0 = betti0(grid)
    # 色值直方图
    hist = [0] * 10
    for row in grid:
        for c in row:
            if 0 <= c < 10:
                hist[c] += 1
    raw = f"{h}x{w}|nz={nonzero}|b0={b0}|hist={hist}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def luzhao_dna_encoding(grid: List[List[int]]) -> Dict[str, Any]:
    """计算 Luzhao DNA 不变量（统一接口）。

    返回：
        {fibonacci_sig, lucas_sig, bagua_sig, topo_hash, betti0}
    """
    b0 = betti0(grid)
    th = topo_hash(grid)
    fib = [fibonacci(i) % 10 for i in range(8)]
    luc = [lucas(i) % 10 for i in range(8)]
    bagua = bagua_from_grid(grid)
    return {
        "fibonacci_sig": fib,
        "lucas_sig": luc,
        "bagua_sig": bagua,
        "topo_hash": th,
        "betti0": b0,
    }


# ════════════════════════════════════════════════════╗
# ║          Golden Ratio / φ 工具                                     ║
# ╚═══════════════════════════════════════════════════╝

PHI = (1 + 5 ** 0.5) / 2  # ≈ 1.618


def phi_proportion(a: float, b: float) -> float:
    """返回 a/b 与 φ 的接近程度 [0, 1]。"""
    if b == 0:
        return 0.0
    ratio = abs(a / b)
    return max(0.0, 1.0 - abs(ratio - PHI) / PHI)


if __name__ == "__main__":
    # 测试
    print(f"F[10] = {fibonacci(10)}")
    print(f"L[10] = {lucas(10)}")
    print(f"Bagua 乾 = {bagua_constant('乾')}")
    print(f"Betti-0 test: {betti0([[1,0],[0,1]])}")  # 2
    print(f"φ = {PHI:.6f}")
