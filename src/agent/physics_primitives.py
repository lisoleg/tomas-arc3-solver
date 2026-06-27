"""
src/agent/physics_primitives.py
TOMAS κ-Phase 物理原语纯软件模拟
IDO/TOMAS: 物理直觉的本质是κ-相位感知
ARC-AGI-3 智能体必须识别网格变换中的κ-相位一致性

四类物理原语:
- newton_push: 牛顿刚体推箱 (KA59 Sokoban) — 质量、摩擦、dead-lock冻结
- mirror_geo: 反射几何 (AR25 镜像覆盖) — 八元数仿射镜像、光线追踪
- dfa: 离散因果状态机 (TN36 点击编程) — DFA因果链、最小作用量路径
- poset: 偏序颜色排序 (SB26 物品排序) — Poset拓扑排序、偏序验证

核心架构: solve_via_pipeline() + SBInjector.physics_primitives → κ-Snap搜索调用原语做Dead-Zero剪枝
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Any
import numpy as np

# ============================================================
# 基本类型
# ============================================================

@dataclass
class Entity:
    """物理实体 — TOMAS κ-Unit 的软件模拟"""
    x: int
    y: int
    mass: float = 1.0      # 箱子质量 > 玩家质量 (隐含推不动自身)
    is_box: bool = False
    is_player: bool = False
    color: int = 0          # 网格颜色值
    tags: List[str] = field(default_factory=list)

# ============================================================
# KA59: 牛顿刚体推箱 (质量、单向推动、dead-lock冻结)
# ============================================================

# Dead-lock 冻结: 箱子推入墙角且无目标 → κ-Snap 剪枝 (Dead-Zero 熔断)

def can_push_box(grid: Any, player_pos: Tuple[int, int], box_pos: Tuple[int, int],
                 direction: Tuple[int, int], wall_char: int = 0) -> Tuple[bool, Tuple[int, int]]:
    """
    TOMAS 判位: 箱子必须在player相邻格，前方不能是墙/另一箱
    返回 (ok: bool, new_box_pos: Tuple[int,int])
    """
    dx, dy = direction
    bx, by = box_pos
    tx, ty = bx + dx, by + dy

    # 转换grid为numpy array以便索引
    if hasattr(grid, 'shape'):
        h, w = grid.shape
    elif isinstance(grid, list):
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
    else:
        return False, box_pos

    if not (0 <= tx < w and 0 <= ty < h):
        return False, box_pos

    cell = grid[ty][tx] if isinstance(grid, list) else int(grid[ty, tx])
    if cell == wall_char:
        return False, box_pos

    # 检查前方是否还有箱 (KA59不允许连环推多个箱)
    if is_box_at(grid, tx, ty, wall_char):
        return False, box_pos

    return True, (tx, ty)


def is_box_at(grid: Any, x: int, y: int, wall_char: int = 0) -> bool:
    """判断某坐标是否有箱/对象"""
    if hasattr(grid, 'shape'):
        cell = int(grid[y, x])
    elif isinstance(grid, list):
        cell = grid[y][x]
    else:
        return False
    # 非墙(0)、非空地(1) → 对象
    return cell != wall_char and cell != 1


def is_deadlock_corner(grid: Any, box_pos: Tuple[int, int],
                       wall_char: int = 0, goal_char: int = 2) -> bool:
    """
    TOMAS Dead-Zero 冻结: 箱子贴墙角 + 非目标格 → 剪枝
    κ-Snap搜索中遇此状态将熔断该分支
    """
    x, y = box_pos
    if hasattr(grid, 'shape'):
        h, w = grid.shape
    elif isinstance(grid, list):
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
    else:
        return False

    # 四邻是否有墙
    left_wall  = (x == 0 or _grid_get(grid, y, x-1, wall_char) == wall_char)
    right_wall = (x == w-1 or _grid_get(grid, y, x+1, wall_char) == wall_char)
    top_wall   = (y == 0 or _grid_get(grid, y-1, x, wall_char) == wall_char)
    bottom_wall= (y == h-1 or _grid_get(grid, y+1, x, wall_char) == wall_char)

    # 角落且非目标格 → 死锁
    cell = _grid_get(grid, y, x, 0)
    if (left_wall or right_wall) and (top_wall or bottom_wall):
        if cell != goal_char:
            return True
    return False


def _grid_get(grid, y, x, default):
    """Safe grid access"""
    try:
        if hasattr(grid, 'shape'):
            return int(grid[y, x])
        elif isinstance(grid, list):
            return grid[y][x]
    except (IndexError, KeyError):
        return default
    return default


# ============================================================
# AR25: 反射几何 (八元数风格仿射镜像)
# ============================================================

# κ-Phase: 反射是八元数虚轴乘法的几何投射
# 在信息对偶场中，镜像操作 = κ-flip (κ-相位180°翻转)

def mirror_point(x: int, y: int, axis: str = 'x',
                 origin_x: int = 0, origin_y: int = 0) -> Tuple[int, int]:
    """
    镜像反射 (仿射变换, 对应八元数虚部乘法)
    axis: 'x' (关于竖线origin_x翻转) / 'y' (关于横线origin_y翻转) / 'xy' (180°旋转)
    κ-Phase: 镜像 = κ-flip, 信息相位 180° 翻转
    """
    if axis == 'x':
        return (2 * origin_x - x, y)
    elif axis == 'y':
        return (x, 2 * origin_y - y)
    elif axis == 'xy':
        return (2 * origin_x - x, 2 * origin_y - y)
    raise ValueError(f"Unknown axis: {axis}")


def reflect_ray(start: Tuple[int, int], hit_pos: Tuple[int, int],
                normal: Tuple[int, int]) -> Tuple[int, int]:
    """
    光线反射: (dx,dy) - 2*(dx*normal.x + dy*normal.y)*normal
    κ-Phase: 反射 = 流贯在κ-相位面上的弹射
    """
    sx, sy = start
    hx, hy = hit_pos
    dx, dy = hx - sx, hy - sy
    nx, ny = normal
    dot = dx * nx + dy * ny
    rx = dx - 2 * dot * nx
    ry = dy - 2 * dot * ny
    return (hx + rx, hy + ry)


def multi_mirror_trace(source: Tuple[int, int],
                       mirrors: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                       max_bounces: int = 10) -> List[Tuple[int, int]]:
    """
    多镜面链式光线追踪
    mirrors: [(mirror_pos, normal), ...]
    返回反射路径点列表
    κ-Phase: 每次弹射 = κ-phase在信息对偶场中的投影偏转
    """
    path = [source]
    current = source
    direction = (1, 0)  # 默认向右

    for bounce in range(max_bounces):
        # 找最近击中的镜面
        best_mirror = None
        best_dist = float('inf')
        for mpos, mnorm in mirrors:
            # 简化: 检查光线是否经过镜面位置附近
            # 实际应由游戏引擎提供镜面位置
            pass

        if best_mirror is None:
            break

        # 反射
        reflected = reflect_ray(current, best_mirror[0], best_mirror[1])
        path.append(reflected)
        current = reflected

    return path


# ============================================================
# TN36: 离散因果状态机 (DFA)
# ============================================================

# κ-Phase: 状态转换 = κ-flip序列 (最小作用量路径)
# EML超图中的因果边 = morphism

class CausalDFA:
    """
    最简DFA: state_id → {event: next_state_id}
    TOMAS: 因果边 = morphism in EML超图
    κ-Phase: 状态转换 = κ-flip序列
    """
    def __init__(self, states: Set[int], trans: Dict[Tuple[int, int], int],
                 init: int, accept: Set[int]):
        self.states = states      # set of state ids
        self.trans = trans        # dict[(sid, event)] = next_sid
        self.state = init
        self.accept = accept

    def step(self, event: int) -> bool:
        """执行一次因果转换"""
        key = (self.state, event)
        if key in self.trans:
            self.state = self.trans[key]
            return True
        return False

    def is_accept(self) -> bool:
        """是否到达目标状态"""
        return self.state in self.accept

    def find_shortest_path(self, target_states: Set[int],
                           max_depth: int = 20) -> Optional[List[int]]:
        """
        BFS搜索从当前状态到目标状态的最短因果路径 (最小作用量)
        返回事件序列 (instruction IDs for TN36)
        """
        from collections import deque
        queue = deque([(self.state, [])])
        visited = {self.state}

        while queue:
            current_state, path = queue.popleft()
            if current_state in target_states:
                return path

            if len(path) >= max_depth:
                continue

            # 尝试所有可能的事件
            for event in range(64):  # TN36最多63种指令
                key = (current_state, event)
                if key in self.trans:
                    next_state = self.trans[key]
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [event]))

        return None


# ============================================================
# SB26: 偏序颜色排序 (Poset / Topological Sort)
# ============================================================

# κ-Phase: 排序 = κ-坍缩 (从偏序到全序的相位坍缩)
# 在EML超图中, 颜色 = 不同维度标签, 排序 = 拓扑排序

def is_valid_poset_order(colors: List[int], target_order: List[int]) -> bool:
    """
    target_order 定义偏序 (从左到右优先级递增)
    检查 colors 是否满足该偏序
    κ-Phase: 偏序满足 = κ-相位一致性 (无相位冲突)
    """
    rank = {c: i for i, c in enumerate(target_order)}
    for i in range(len(colors) - 1):
        if rank.get(colors[i], 999) > rank.get(colors[i + 1], 999):
            return False
    return True


def topological_sort_colors(target_order: List[int]) -> List[int]:
    """偏序全显化 (κ-坍缩: 从偏序到全序)"""
    return list(target_order)


def compute_swap_sequence(current_colors: List[int],
                          target_order: List[int],
                          available_actions: List[int] = None) -> Optional[List[Tuple[int, Any]]]:
    """
    计算从当前颜色排列到目标排列的最少交换序列
    κ-Phase: 每次交换 = κ-flip (相位翻转)
    返回 [(action_id, position), ...] 格式的plan
    """
    if not current_colors or not target_order:
        return None

    # 简化: 逐位贪心交换
    plan = []
    current = list(current_colors)

    for i, target_color in enumerate(target_order):
        if current[i] == target_color:
            continue

        # 找到目标颜色的位置
        j = None
        for k in range(i, len(current)):
            if current[k] == target_color:
                j = k
                break

        if j is None:
            continue

        # 交换 current[i] 和 current[j]
        plan.append(('swap', i, j))
        current[i], current[j] = current[j], current[i]

    return plan if plan else None


# ============================================================
# CN04: 仿射变换 (κ-Phase: 旋转+平移 = κ-相位偏转+κ-平移)
# ============================================================

# κ-Phase: 仿射变换 = κ-rotation (相位角度偏转) + κ-translation (空间相位偏移)
# 在EML超图中, 仿射变换 = D4群旋转 × 位移向量
# 高斯唯一性定理: 线性可识别性要求残差为高斯白噪声

def rotate_90(grid: Any, k: int = 1) -> Any:
    """
    κ-旋转: 将网格旋转90°×k (相位角度偏转)
    k=1: 90°, k=2: 180°, k=3: 270°
    κ-Phase: 旋转 = κ-rotation (相位角度偏转)
    """
    if hasattr(grid, 'shape'):
        return np.rot90(np.array(grid), k=k)
    elif isinstance(grid, list):
        g = np.array(grid)
        return np.rot90(g, k=k)
    return grid


def translate_grid(grid: Any, dx: int = 0, dy: int = 0) -> Any:
    """
    κ-平移: 将网格平移(dx, dy)像素 (空间相位偏移)
    κ-Phase: 平移 = κ-translation (空间相位偏移)
    """
    if hasattr(grid, 'shape'):
        g = np.array(grid)
    elif isinstance(grid, list):
        g = np.array(grid)
    else:
        return grid

    h, w = g.shape
    result = np.zeros_like(g)
    for y in range(h):
        for x in range(w):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h:
                result[ny, nx] = g[y, x]
    return result


def find_affine_transform(source: Any, target: Any, max_translation: int = 10) -> Optional[Dict]:
    """
    搜索将source映射到target的最佳仿射变换
    D4群(8种旋转) × 位移向量 → 最多8×(2*max_translation+1)²种组合
    κ-Phase: 仿射变换搜索 = κ-Phase一致性检测的离散版本

    Args:
        source: Source grid (2D)
        target: Target grid (2D)
        max_translation: Max translation offset to search

    Returns:
        Dict with 'rotation' (0-3), 'dx', 'dy', 'match_score' or None
    """
    if hasattr(source, 'shape'):
        s = np.array(source)
        t = np.array(target)
    elif isinstance(source, list):
        s = np.array(source)
        t = np.array(target)
    else:
        return None

    if s.shape != t.shape:
        # Try padding/cropping to match shapes
        # For now, only try same-shape transforms
        return None

    best_score = 0.0
    best_params = None
    total_cells = t.size

    for k in range(4):  # 0°, 90°, 180°, 270°
        rotated = np.rot90(s, k=k)
        if rotated.shape != t.shape:
            continue

        # Zero translation first
        match = np.sum(rotated == t) / total_cells
        if match > best_score:
            best_score = match
            best_params = {'rotation': k, 'dx': 0, 'dy': 0, 'match_score': float(match)}

        # With translations (shift and compare overlapping region)
        for dx in range(-max_translation, max_translation + 1):
            for dy in range(-max_translation, max_translation + 1):
                # Compute overlap match
                r_h, r_w = rotated.shape
                t_h, t_w = t.shape

                # Overlapping region
                r_y_start = max(0, -dy)
                r_x_start = max(0, -dx)
                t_y_start = max(0, dy)
                t_x_start = max(0, dx)

                r_y_end = min(r_h, t_h - dy)
                r_x_end = min(r_w, t_w - dx)
                t_y_end = min(t_h, r_h + dy)
                t_x_end = min(t_w, r_w + dx)

                if r_y_end <= r_y_start or r_x_end <= r_x_start:
                    continue

                overlap_r = rotated[r_y_start:r_y_end, r_x_start:r_x_end]
                overlap_t = t[t_y_start:t_y_end, t_x_start:t_x_end]

                if overlap_r.shape != overlap_t.shape or overlap_r.size == 0:
                    continue

                match = np.sum(overlap_r == overlap_t) / total_cells
                if match > best_score:
                    best_score = match
                    best_params = {'rotation': k, 'dx': dx, 'dy': dy, 'match_score': float(match)}

    if best_params is not None and best_score > 0.5:
        return best_params
    return None


def align_target(source: Any, target: Any) -> Optional[Any]:
    """
    找到仿射变换后直接应用，返回变换后的网格
    κ-Phase: align = κ-Snap归约的几何路径版本
    """
    params = find_affine_transform(source, target)
    if params is None:
        return None
    rotated = rotate_90(source, params['rotation'])
    translated = translate_grid(rotated, params['dx'], params['dy'])
    return translated


# ============================================================
# κ-Phase 一致性检测 (核心理论 — 从URL1采纳)
# ============================================================

# 物理直觉的本质是 κ-相位感知
# ARC-AGI-3 智能体必须识别网格变换中的 κ-相位一致性

def kappa_phase_consistency(grid1: Any, grid2: Any,
                            transformations: List[str] = ['color_flip', 'rotation', 'mirror']) -> float:
    """
    计算两个网格之间的κ-相位一致性程度
    返回一致性分数 0.0 ~ 1.0

    κ-Phase: 颜色翻转 = κ-flip (180°相位翻转)
             旋转 = κ-旋转 (相位角度偏转)
             镜像 = κ-reflection (虚轴乘法投影)

    物理直觉的本质 = 识别这些κ-相位变换的能力
    这解释了为何陪集优先搜索(330 coset)与拉马努金机猜想引擎表现卓越
    """
    try:
        if hasattr(grid1, 'shape'):
            g1 = np.array(grid1)
            g2 = np.array(grid2)
        else:
            g1 = np.array(grid1)
            g2 = np.array(grid2)
    except Exception:
        return 0.0

    if g1.shape != g2.shape:
        return 0.0

    best_score = 0.0

    for transform in transformations:
        if transform == 'color_flip':
            # κ-flip: 颜色翻转一致性
            # 检查是否存在颜色映射使得 g1 → g2
            score = _check_color_flip_consistency(g1, g2)
            best_score = max(best_score, score)

        elif transform == 'rotation':
            # κ-rotation: 旋转一致性
            # 检查90°/180°/270°旋转后是否匹配
            score = _check_rotation_consistency(g1, g2)
            best_score = max(best_score, score)

        elif transform == 'mirror':
            # κ-reflection: 镜像一致性
            score = _check_mirror_consistency(g1, g2)
            best_score = max(best_score, score)

    return best_score


def _check_color_flip_consistency(g1: np.ndarray, g2: np.ndarray) -> float:
    """κ-flip: 检查是否存在颜色映射使得 g1 → g2 完全匹配"""
    if g1.shape != g2.shape:
        return 0.0

    # 构建颜色映射
    color_map = {}
    mismatches = 0
    total = 0

    for y in range(g1.shape[0]):
        for x in range(g1.shape[1]):
            c1 = int(g1[y, x])
            c2 = int(g2[y, x])
            total += 1

            if c1 in color_map:
                if color_map[c1] != c2:
                    mismatches += 1
            else:
                color_map[c1] = c2

    if total == 0:
        return 1.0
    return 1.0 - mismatches / total


def _check_rotation_consistency(g1: np.ndarray, g2: np.ndarray) -> float:
    """κ-rotation: 检查旋转后是否匹配 (90°/180°/270°)"""
    if g1.shape != g2.shape:
        # 对于非方阵，旋转会改变shape
        # 尝试180°旋转（不改变shape）
        rotated_180 = np.rot90(g1, 2)
        if rotated_180.shape == g2.shape:
            match = np.sum(rotated_180 == g2) / (g2.shape[0] * g2.shape[1])
            return float(match)
        return 0.0

    best = 0.0
    for k in [1, 2, 3]:  # 90°, 180°, 270°
        rotated = np.rot90(g1, k)
        if rotated.shape == g2.shape:
            match = np.sum(rotated == g2) / (g2.shape[0] * g2.shape[1])
            best = max(best, float(match))
    return best


def _check_mirror_consistency(g1: np.ndarray, g2: np.ndarray) -> float:
    """κ-reflection: 检查镜像后是否匹配"""
    if g1.shape != g2.shape:
        return 0.0

    best = 0.0
    # 水平镜像
    mirrored_h = np.flip(g1, axis=1)
    match_h = np.sum(mirrored_h == g2) / (g2.shape[0] * g2.shape[1])
    best = max(best, float(match_h))

    # 垂直镜像
    mirrored_v = np.flip(g1, axis=0)
    match_v = np.sum(mirrored_v == g2) / (g2.shape[0] * g2.shape[1])
    best = max(best, float(match_v))

    return best


# ============================================================
# 导出
# ============================================================

PHYSICS_PRIMITIVE_REGISTRY = {
    'newton_push': {
        'functions': [can_push_box, is_box_at, is_deadlock_corner],
        'description': '牛顿刚体推箱 — 质量、摩擦、dead-lock冻结',
        'games': ['ka59'],
    },
    'mirror_geo': {
        'functions': [mirror_point, reflect_ray, multi_mirror_trace],
        'description': '反射几何 — 八元数仿射镜像、光线追踪',
        'games': ['ar25'],
    },
    'dfa': {
        'functions': [CausalDFA],
        'description': '离散因果状态机 — DFA因果链、最小作用量路径',
        'games': ['tn36'],
    },
    'poset': {
        'functions': [is_valid_poset_order, topological_sort_colors, compute_swap_sequence],
        'description': '偏序颜色排序 — Poset拓扑排序、偏序验证',
        'games': ['sb26'],
    },
    'affine_transform': {
        'functions': [rotate_90, translate_grid, find_affine_transform, align_target],
        'description': '仿射变换 — κ-旋转+κ-平移 (D4群×位移)',
        'games': ['cn04'],
    },
}

def get_primitive_functions(primitive_name: str) -> List[Any]:
    """获取指定物理原语的所有函数"""
    entry = PHYSICS_PRIMITIVE_REGISTRY.get(primitive_name)
    if entry is None:
        return []
    return entry['functions']
