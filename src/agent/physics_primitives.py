"""
src/agent/physics_primitives.py
TOMAS κ-Phase 物理原语纯软件模拟
IDO/TOMAS: 物理直觉的本质是κ-相位感知
ARC-AGI-3 智能体必须识别网格变换中的κ-相位一致性

十二类物理/几何原语:
- newton_push: 牛顿刚体推箱 (KA59 Sokoban) — 质量、摩擦、dead-lock冻结
- mirror_geo: 反射几何 (AR25 镜像覆盖) — 八元数仿射镜像、光线追踪
- optics: 光学物理 (AR25 光线追踪) — BFS光线追踪+覆盖图+镜面移动约束
- dfa: 离散因果状态机 (TN36 点击编程) — DFA因果链、最小作用量路径
- poset: 偏序颜色排序 (SB26 物品排序) — Poset拓扑排序、偏序验证
- affine_transform: 仿射变换 (CN04) — κ-旋转+κ-平移 (D4群×位移)
- kinematics: 运动学原语 — 轨迹、速度向量、加速度、抛物线弹道
- collision: 碰撞检测原语 — 刚体碰撞、动量守恒、弹性/非弹性碰撞
- thermodynamics: 热力学原语 — 熵计算、能量梯度、网格热扩散、最小能量路径
- wave: 波动力学原语 — 离散波传播、干涉检测、叠加原理
- euclidean_geometry: 欧几里得几何原语 — 距离度量、线段交点、凸包、几何相似性
- trigonometry_discrete: 离散三角函数原语 — 方向角、向量分解、极坐标转换

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
# KA59 Wall-Ride Push 豁免 (文章 Appendix A.1/A.2)
# ============================================================

# κ-Phase: 推箱子贴墙时, 三面围死但不一定是死锁 —
# 如果箱子正在沿着墙壁滑向目标, 应豁免死锁判定。
# 这是κ-优选物理直觉的核心: 路径连续性 > 静态deadlock判定。

# 四方向常量 (上/下/左/右)
DIRS4: List[Tuple[int, int]] = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# 空地字符 (默认)
EMPTY_CHAR: int = 1


def _in_bounds(pos: Tuple[int, int], grid: Any) -> bool:
    """检查坐标是否在grid边界内。

    Args:
        pos: 坐标(x, y)。
        grid: 2D numpy array或列表。

    Returns:
        True如果在边界内, False否则。
    """
    x, y = pos
    if hasattr(grid, 'shape'):
        h, w = grid.shape
        return 0 <= x < w and 0 <= y < h
    elif isinstance(grid, list):
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        return 0 <= x < w and 0 <= y < h
    return False


def _add_pos(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    """向量加法: add(a, b) = (a.x+b.x, a.y+b.y)。"""
    return (a[0] + b[0], a[1] + b[1])


def _sub_pos(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    """向量减法: sub(a, b) = (a.x-b.x, a.y-b.y)。"""
    return (a[0] - b[0], a[1] - b[1])


def _neg_pos(d: Tuple[int, int]) -> Tuple[int, int]:
    """方向取反: neg(d) = (-d.x, -d.y)。"""
    return (-d[0], -d[1])


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """曼哈顿距离: |a.x-b.x| + |a.y-b.y|。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_wall_ride_push(
    player: Tuple[int, int],
    box: Tuple[int, int],
    grid: Any,
    goal: Optional[Tuple[int, int]] = None,
    wall_char: int = 0,
    box_chars: Set[int] = {3, 5},
    empty_char: int = EMPTY_CHAR,
) -> bool:
    """KA59 Wall-Ride Push 豁免判定 (文章 Appendix A.1)。

    箱子贴墙推时, 豁免三面围死误判。κ-优选核心洞察:
    路径连续性 > 静态deadlock判定。如果箱子正在沿墙滑向目标,
    即使三面围死也不应熔断。

    判定逻辑:
      1. 箱子相邻必须有墙 (adj_walls非空)
      2. push_dir必须在DIRS4中 (标准四方向推)
      3. 前方(push_dir方向)必须是空地
      4. 后方(push反方向)不能是墙或另一箱 (玩家可达)
      5. blocked≥3且非目标格 → 拒绝豁免 (真正死锁)
      6. 如果有目标, push后距离不能远离目标超过1步

    Args:
        player: 玩家位置(x, y)。
        box: 箱子位置(x, y)。
        grid: 2D numpy array或列表。
        goal: 目标位置(x, y), 可为None。
        wall_char: 墙壁颜色值, 默认0。
        box_chars: 箱子颜色值集合, 默认{3, 5}。
        empty_char: 地颜色值, 默认1。

    Returns:
        True表示Wall-Ride豁免(允许推), False表示不豁免。
    """
    # Step 1: 检查箱子相邻是否有墙
    adj_walls: List[Tuple[int, int]] = []
    for d in DIRS4:
        adj_pos: Tuple[int, int] = _add_pos(box, d)
        if not _in_bounds(adj_pos, grid):
            adj_walls.append(d)  # 越界=墙
        else:
            cell_val: int = _grid_get(grid, adj_pos[1], adj_pos[0], wall_char)
            if cell_val == wall_char:
                adj_walls.append(d)

    if not adj_walls:
        return False  # 箱子不贴墙 → 不触发Wall-Ride

    # Step 2: 计算push方向 (从玩家到箱子的方向)
    push_dir: Tuple[int, int] = _sub_pos(box, player)
    if push_dir not in DIRS4:
        return False  # 非标准四方向推 → 不豁免

    # Step 3: 前方(push方向)必须是空地或目标
    front_pos: Tuple[int, int] = _add_pos(box, push_dir)
    if not _in_bounds(front_pos, grid):
        return False  # 前方越界 → 不可推
    front_val: int = _grid_get(grid, front_pos[1], front_pos[0], wall_char)
    if front_val == wall_char or front_val in box_chars:
        return False  # 前方是墙或箱 → 不可推

    # Step 4: 后方(push反方向)不能是墙或箱 (玩家必须可达)
    back_dir: Tuple[int, int] = _neg_pos(push_dir)
    back_pos: Tuple[int, int] = _add_pos(box, back_dir)
    if not _in_bounds(back_pos, grid):
        return False  # 后方越界 → 玩家不可达
    back_val: int = _grid_get(grid, back_pos[1], back_pos[0], wall_char)
    if back_val == wall_char or back_val in box_chars:
        return False  # 后方是墙或箱 → 玩家不可达

    # Step 5: blocked≥3且非目标格 → 拒绝豁免 (真正死锁)
    blocked: int = sum(
        1 for d in DIRS4
        if not _in_bounds(_add_pos(box, d), grid)
        or _grid_get(grid, _add_pos(box, d)[1], _add_pos(box, d)[0], wall_char) in {wall_char, *box_chars}
    )
    if blocked >= 3 and not (goal and box == goal):
        return False  # 三面围死且非目标 → 真正死锁, 不豁免

    # Step 6: 如果有目标, push后不能远离目标超过1步
    if goal is not None:
        cur_d: int = _manhattan(box, goal)
        new_box_pos: Tuple[int, int] = _add_pos(box, push_dir)
        nxt_d: int = _manhattan(new_box_pos, goal)
        if nxt_d > cur_d + 1:
            return False  # 远离目标 → 不豁免

    return True  # Wall-Ride豁免: 允许贴墙推


def check_deadlock_with_wall_ride(
    grid: Any,
    box_pos: Tuple[int, int],
    player_pos: Tuple[int, int],
    goal: Optional[Tuple[int, int]] = None,
    wall_char: int = 0,
    goal_char: int = 2,
    box_chars: Set[int] = {3, 5},
) -> bool:
    """CHK_DL修正版 — 含Wall-Ride豁免 (文章 Appendix A.2)。

    升级版deadlock检查: 保留原有is_deadlock_corner判定,
    但添加Wall-Ride豁免。当箱子贴墙推且正在滑向目标时,
    即使看起来是"死锁"也不熔断。

    κ-优选核心: 物理直觉 = 路径连续性优先于静态判断。
    "箱子贴墙滑向目标" ≠ "箱子推入死角冻死"

    Args:
        grid: 2D numpy array或列表。
        box_pos: 箱子位置(x, y)。
        player_pos: 玩家位置(x, y)。
        goal: 目标位置(x, y), 可为None。
        wall_char: 墙壁颜色值, 默认0。
        goal_char: 目标颜色值, 默认2。
        box_chars: 箱子颜色值集合, 默认{3, 5}。

    Returns:
        True表示死锁(应熔断), False表示非死锁(允许推)。
    """
    # Step 1: 基础blocked计数
    blocked: int = sum(
        1 for d in DIRS4
        if not _in_bounds(_add_pos(box_pos, d), grid)
        or _grid_get(grid, _add_pos(box_pos, d)[1], _add_pos(box_pos, d)[0], wall_char) in {wall_char, *box_chars}
    )

    # Step 2: blocked≥3且非目标格 → 潜在死锁
    cell_val: int = _grid_get(grid, box_pos[1], box_pos[0], wall_char)
    if blocked >= 3 and not (goal and box_pos == goal and cell_val == goal_char):
        # Step 3: Wall-Ride豁免检查
        if is_wall_ride_push(player_pos, box_pos, grid, goal, wall_char, box_chars):
            return False  # 豁免: 箱子正在贴墙滑向目标
        return True  # 真正死锁: 三面围死且不豁免

    # Step 4: 原有is_deadlock_corner检查 (角落判定)
    if is_deadlock_corner(grid, box_pos, wall_char, goal_char):
        # Wall-Ride豁免二次检查
        if is_wall_ride_push(player_pos, box_pos, grid, goal, wall_char, box_chars):
            return False  # 角落但豁免
        return True

    return False  # 非死锁


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
# AR25: 光学物理原语 (optics ray-tracing + coverage)
# κ-Phase: 反射 = 光子在κ-相位面上的弹射
# 光线覆盖游戏: 选择piece → 移动piece → 通过mirror反射覆盖target
# ============================================================

@dataclass
class OpticsMirror:
    """光学镜面 — κ-Phase反射面的软件模拟。

    垂直镜面(vertical): 反射x轴 — ref_x = 2*mirror_x - src_x
    水平镜面(horizontal): 反射y轴 — ref_y = 2*mirror_y - src_y
    移动约束: vertical只上下移动, horizontal只左右移动
    """
    x: int
    y: int
    orientation: str  # 'vertical' or 'horizontal'
    width: int = 1
    height: int = 1
    movable: bool = True
    move_axis: str = 'vertical'  # vertical mirror只能上下, horizontal只能左右

    @property
    def tag(self) -> str:
        """AR25 sprite tag for mirror identification."""
        if self.orientation == 'vertical':
            return '0054kgxrvfihgm'
        else:
            return '0002nuguepuujf'

    def reflect(self, src_x: int, src_y: int) -> Tuple[int, int]:
        """反射坐标 — 精确匹配AR25游戏源码nloqvbouxu()。

        vertical mirror: ref_x = 2*mirror_x - src_x, ref_y = src_y
        horizontal mirror: ref_x = src_x, ref_y = 2*mirror_y - src_y
        """
        if self.orientation == 'vertical':
            return (2 * self.x - src_x, src_y)
        else:
            return (src_x, 2 * self.y - src_y)


@dataclass
class OpticsTarget:
    """光学覆盖目标 — 需要被piece直接或反射覆盖的点。"""
    x: int
    y: int


@dataclass
class OpticsPiece:
    """光学可移动piece — 可以选择并移动的sprite。

    piece与mirror共享sprite (AR25特有):
    0003uqrdzdofso + 0054kgxrvfihgm → vertical mirror/piece
    0003uqrdzdofso + 0002nuguepuujf → horizontal mirror/piece
    """
    x: int
    y: int
    orientation: str  # 'vertical' (上下移动) or 'horizontal' (左右移动)
    width: int = 1
    height: int = 1


def optics_ray_trace(
    source: Tuple[int, int],
    mirrors: List[OpticsMirror],
    max_bounces: int = 12,
    grid_width: int = 64,
    grid_height: int = 64,
) -> List[Tuple[int, int]]:
    """光学光线追踪 — 精确匹配AR25游戏源码nloqvbouxu()的BFS反射逻辑。

    从source出发, BFS遍历所有可达的反射点:
    - 每个像素点尝试被每个mirror反射
    - 反射后继续尝试新mirror
    - 最大深度12 (ythhvclqmk = 12 in game source)
    - 带过滤: reflect_horizontal_only只匹配horizontal mirror,
               0038pnuzypawco只匹配vertical mirror

    κ-Phase: 每次反射 = 光子在κ-相位面的弹射,
    信息相位沿镜面法线方向翻转180°。

    Args:
        source: 光源坐标(piece/pixel位置)
        mirrors: 所有镜面对象
        max_bounces: 最大反射深度(游戏源码ythhvclqmk=12)
        grid_width: 网格宽度
        grid_height: 网格高度

    Returns:
        反射路径点列表(包含source和所有可达反射点)
    """
    from collections import deque

    path_points: List[Tuple[int, int]] = []
    visited: Set[Tuple[int, int]] = set()
    bfs_queue: deque = deque()

    # 初始点
    sx, sy = source
    if 0 <= sx < grid_width and 0 <= sy < grid_height:
        visited.add((sx, sy))
        bfs_queue.append(((sx, sy), 0))
        path_points.append((sx, sy))

    while bfs_queue:
        current_pos, depth = bfs_queue.popleft()
        if depth > max_bounces:
            continue
        cx, cy = current_pos

        # 尝试被每个mirror反射
        for mirror in mirrors:
            # 反射计算 — 精确匹配游戏源码
            if mirror.orientation == 'vertical':
                ref_x = 2 * mirror.x - cx
                ref_y = cy
            else:
                ref_x = cx
                ref_y = 2 * mirror.y - cy

            reflected = (ref_x, ref_y)

            # 已访问则跳过(防止振荡)
            if reflected in visited:
                continue

            visited.add(reflected)
            bfs_queue.append((reflected, depth + 1))

            # 只记录在网格内的反射点
            if 0 <= ref_x < grid_width and 0 <= ref_y < grid_height:
                path_points.append((ref_x, ref_y))

    return path_points


def optics_coverage_map(
    piece_pixels: List[Tuple[int, int]],
    mirrors: List[OpticsMirror],
    max_bounces: int = 12,
    grid_width: int = 64,
    grid_height: int = 64,
) -> np.ndarray:
    """光学覆盖图 — 精确匹配AR25游戏源码nloqvbouxu()。

    对每个piece的每个像素, 执行光线追踪(BFS反射),
    最终合并所有反射结果生成覆盖图(jtowzmaffb)。

    κ-Phase: 覆盖 = κ-Snap归约后的可达集合,
    每个像素的反射路径 = κ-陪集中的因果链。

    Args:
        piece_pixels: piece的所有像素坐标列表[(x, y), ...]
        mirrors: 所有镜面对象
        max_bounces: 最大反射深度
        grid_width: 网格宽度
        grid_height: 网格高度

    Returns:
        覆盖图(int数组, -1=未覆盖, 非负=被覆盖)
    """
    from collections import deque

    coverage = np.full((grid_height, grid_width), -1, dtype=int)
    all_visited: Set[Tuple[int, int]] = set()

    for px, py in piece_pixels:
        # BFS光线追踪: 从piece像素出发, 反射可达的所有点
        visited: Set[Tuple[int, int]] = set()
        bfs_queue: deque = deque()

        start = (px, py)
        if start not in all_visited:
            visited.add(start)
            bfs_queue.append((start, 0))

        while bfs_queue:
            current_pos, depth = bfs_queue.popleft()
            if depth > max_bounces:
                continue
            cx, cy = current_pos

            for mirror in mirrors:
                if mirror.orientation == 'vertical':
                    ref_x = 2 * mirror.x - cx
                    ref_y = cy
                else:
                    ref_x = cx
                    ref_y = 2 * mirror.y - cy

                reflected = (ref_x, ref_y)
                if reflected in visited or reflected in all_visited:
                    continue

                visited.add(reflected)
                bfs_queue.append((reflected, depth + 1))

                if 0 <= ref_x < grid_width and 0 <= ref_y < grid_height:
                    if coverage[ref_y, ref_x] < 0:
                        coverage[ref_y, ref_x] = 0

        # 标记piece像素本身
        if 0 <= px < grid_width and 0 <= py < grid_height:
            coverage[py, px] = 0

        all_visited.update(visited)

    return coverage


def optics_check_win(
    targets: List[OpticsTarget],
    coverage: np.ndarray,
) -> bool:
    """光学胜判定 — 精确匹配AR25游戏源码vplrhaovhr()。

    所有target的(x, y)在覆盖图中值 >= 0 → win

    Args:
        targets: 所有目标位置列表
        coverage: 光学覆盖图

    Returns:
        True如果所有target被覆盖(值>=0)
    """
    for t in targets:
        if 0 <= t.x < coverage.shape[1] and 0 <= t.y < coverage.shape[0]:
            if coverage[t.y, t.x] < 0:
                return False
        else:
            return False
    return True


def optics_mirror_move_constraint(
    mirror: OpticsMirror,
    action_id: int,
) -> Tuple[int, int]:
    """AR25镜面/piece移动约束 — 精确匹配游戏源码。

    vertical mirror/piece (0054kgxrvfihgm): 只能上下移动
      ACTION1(UP): dy=-1, ACTION2(DOWN): dy=+1, dx=0
    horizontal mirror/piece (0002nuguepuujf): 只能左右移动
      ACTION3(LEFT): dx=-1, ACTION4(RIGHT): dx=+1, dy=0

    κ-Phase: 移动约束 = κ-相位守恒,
    piece只能沿其反射轴的垂直方向移动(保持反射几何不变)。
    """
    dx, dy = 0, 0
    if action_id == 1:  # UP
        dy = -1
    elif action_id == 2:  # DOWN
        dy = 1
    elif action_id == 3:  # LEFT
        dx = -1
    elif action_id == 4:  # RIGHT
        dx = 1

    # vertical mirror: 禁止左右移动
    if mirror.orientation == 'vertical':
        dx = 0
    # horizontal mirror: 禁止上下移动
    if mirror.orientation == 'horizontal':
        dy = 0

    return (dx, dy)


# ============================================================
# 运动学原语 (Kinematics) — 轨迹、速度向量、加速度
# κ-Phase: 运动学 = κ-时间相位序列的离散化
# ARC网格中的运动是离散的，但遵守经典力学约束
# ============================================================

@dataclass
class TrajectoryPoint:
    """轨迹点 — 离散化的κ-时间相位"""
    x: int
    y: int
    t: int = 0           # 离散时间步
    vx: float = 0.0      # 速度分量
    vy: float = 0.0


def compute_trajectory(start: Tuple[int, int], velocity: Tuple[float, float],
                       steps: int, bounds: Optional[Tuple[int, int, int, int]] = None
                       ) -> List[TrajectoryPoint]:
    """
    计算离散轨迹 — 匀速运动模型

    κ-Phase: trajectory = κ-时间序列上位置相位的等间隔采样

    Args:
        start: (x, y) 起始位置
        velocity: (vx, vy) 速度向量
        steps: 预测步数
        bounds: (x_min, y_min, x_max, y_max) 边界限制

    Returns:
        轨迹点列表 (包含起始点)
    """
    points = []
    x, y = float(start[0]), float(start[1])
    vx, vy = velocity
    for t in range(steps):
        pt = TrajectoryPoint(x=int(round(x)), y=int(round(y)), t=t, vx=vx, vy=vy)
        if bounds:
            x_min, y_min, x_max, y_max = bounds
            pt.x = max(x_min, min(x_max, pt.x))
            pt.y = max(y_min, min(y_max, pt.y))
        points.append(pt)
        x += vx
        y += vy
    return points


def compute_parabolic_trajectory(start: Tuple[int, int], v0: Tuple[float, float],
                                  gravity: float = 1.0, steps: int = 20,
                                  bounds: Optional[Tuple[int, int, int, int]] = None
                                  ) -> List[TrajectoryPoint]:
    """
    抛物线弹道 — 带重力加速度的离散轨迹

    κ-Phase: parabola = κ-时间序列上的二次相位偏移
    重力使vy随时间线性递增，轨迹呈抛物线

    Args:
        start: (x, y) 起始位置
        v0: (vx, vy0) 初速度
        gravity: 重力加速度（每步vy增量）
        steps: 预测步数
        bounds: 边界限制

    Returns:
        轨迹点列表
    """
    points = []
    x, y = float(start[0]), float(start[1])
    vx, vy = v0[0], v0[1]
    for t in range(steps):
        pt = TrajectoryPoint(x=int(round(x)), y=int(round(y)), t=t, vx=vx, vy=vy)
        if bounds:
            x_min, y_min, x_max, y_max = bounds
            pt.x = max(x_min, min(x_max, pt.x))
            pt.y = max(y_min, min(y_max, pt.y))
        points.append(pt)
        x += vx
        vy += gravity  # 重力加速
        y += vy
    return points


def predict_intercept_position(target_traj: List[TrajectoryPoint],
                                interceptor_speed: float,
                                interceptor_start: Tuple[int, int]
                                ) -> Optional[Tuple[int, int]]:
    """
    拦截位置预测 — 找到最快拦截点

    κ-Phase: intercept = κ-时间序列对齐点（两个相位序列的最早交叉）

    Args:
        target_traj: 目标轨迹
        interceptor_speed: 拦截者速度（曼哈距离每步）
        interceptor_start: 拦截者起始位置

    Returns:
        拦截点坐标 (x, y)，或 None
    """
    ix, iy = interceptor_start
    for pt in target_traj:
        dist = abs(pt.x - ix) + abs(pt.y - iy)
        if dist <= interceptor_speed * pt.t:
            return (pt.x, pt.y)
    return None


def velocity_from_displacement(positions: List[Tuple[int, int]],
                                dt: int = 1
                                ) -> List[Tuple[float, float]]:
    """
    从位移序列推导速度向量

    κ-Phase: velocity = Δposition / Δt = κ-相位差分的标准形式

    Args:
        positions: 位置序列 [(x0,y0), (x1,y1), ...]
        dt: 时间步长

    Returns:
        速度向量序列 [(vx0,vy0), ...]
    """
    velocities = []
    for i in range(len(positions) - 1):
        dx = positions[i+1][0] - positions[i][0]
        dy = positions[i+1][1] - positions[i][1]
        velocities.append((dx / dt, dy / dt))
    return velocities


# ============================================================
# 碰撞检测原语 (Collision) — 刚体碰撞、动量守恒
# κ-Phase: 碰撞 = κ-相位交汇点，动量守恒是κ-总量不变约束
# ============================================================

@dataclass
class CollisionResult:
    """碰撞结果 — κ-相位交汇点"""
    collided: bool
    position: Tuple[int, int]        # 碰撞发生位置
    v1_post: Tuple[float, float]     # 对象1碰后速度
    v2_post: Tuple[float, float]     # 对象2碰后速度


def check_rect_collision(r1: Tuple[int, int, int, int],
                          r2: Tuple[int, int, int, int]) -> bool:
    """
    矩形碰撞检测 — AABB (Axis-Aligned Bounding Box)

    κ-Phase: 碰撞 = κ-相位域重叠（两个实体的相位窗口交集非空）

    Args:
        r1: (x1, y1, w1, h1) 矩形1
        r2: (x2, y2, w2, h2) 矩形2

    Returns:
        True如果两个矩形重叠
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1)


def elastic_collision_1d(v1: float, m1: float, v2: float, m2: float
                         ) -> Tuple[float, float]:
    """
    一维弹性碰撞 — 动量+能量守恒

    κ-Phase: elastic = κ-总量完全守恒（相位总量和幅值总量都不变）
    v1' = ((m1-m2)*v1 + 2*m2*v2) / (m1+m2)
    v2' = ((m2-m1)*v2 + 2*m1*v1) / (m1+m2)

    Args:
        v1, m1: 对象1速度和质量
        v2, m2: 对象2速度和质量

    Returns:
        (v1_post, v2_post) 碰后速度
    """
    total_m = m1 + m2
    if total_m == 0:
        return v1, v2
    v1_post = ((m1 - m2) * v1 + 2 * m2 * v2) / total_m
    v2_post = ((m2 - m1) * v2 + 2 * m1 * v1) / total_m
    return v1_post, v2_post


def inelastic_collision_1d(v1: float, m1: float, v2: float, m2: float,
                            restitution: float = 0.5
                            ) -> Tuple[float, float]:
    """
    一维非弹性碰撞 — 动量守恒，能量损失由恢复系数控制

    κ-Phase: inelastic = κ-幅值部分耗散（相位守恒但幅值衰减）
    恢复系数 e: 1.0=完全弹性, 0.0=完全非弹性

    Args:
        restitution: 恢复系数 (0~1)

    Returns:
        (v1_post, v2_post)
    """
    total_m = m1 + m2
    if total_m == 0:
        return v1, v2
    v_common = (m1 * v1 + m2 * v2) / total_m
    v1_post = v_common + restitution * m2 * (v2 - v1) / total_m
    v2_post = v_common - restitution * m1 * (v2 - v1) / total_m
    return v1_post, v2_post


def grid_collision_detect(grid: Any, moving_pos: Tuple[int, int],
                          direction: Tuple[int, int],
                          obstacle_chars: Set[int] = None
                          ) -> Optional[Tuple[int, int]]:
    """
    网格碰撞检测 — 检查移动方向上的第一个障碍物

    κ-Phase: 网格碰撞 = κ-相位序列上的不可穿越相位壁

    Args:
        grid: 2D网格 (numpy array or list)
        moving_pos: 移动实体位置
        direction: (dx, dy) 移动方向
        obstacle_chars: 障碍物颜色集合

    Returns:
        碰撞位置 (x, y)，或 None（无碰撞）
    """
    if obstacle_chars is None:
        obstacle_chars = {0}  # 默认：0=墙

    dx, dy = direction
    x, y = moving_pos

    if hasattr(grid, 'shape'):
        h, w = grid.shape
    elif isinstance(grid, list):
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
    else:
        return None

    nx, ny = x + dx, y + dy
    if not (0 <= nx < w and 0 <= ny < h):
        return (nx, ny)  # 边界碰撞

    cell = int(grid[ny, nx]) if hasattr(grid, 'shape') else grid[ny][nx]
    if cell in obstacle_chars:
        return (nx, ny)

    return None


# ============================================================
# 热力学原语 (Thermodynamics) — 熵、能量梯度、热扩散
# κ-Phase: 熵 = κ-相位分布的离散度，热扩散 = κ-相位均衡化过程
# ============================================================

def compute_grid_entropy(grid: Any) -> float:
    """
    网格熵计算 — 颜色分布的Shannon熵

    κ-Phase: 熵 = κ-相位分布的不确定性度量
    高熵 = κ-相位均匀分布（多种颜色混杂）
    低熵 = κ-相位集中分布（少数颜色主导）

    Args:
        grid: 2D网格

    Returns:
        Shannon熵值 (0.0 ~ log(n_colors))
    """
    if hasattr(grid, 'shape'):
        arr = np.array(grid).flatten()
    else:
        arr = np.array(grid).flatten()

    if len(arr) == 0:
        return 0.0

    # 频率统计
    unique, counts = np.unique(arr, return_counts=True)
    probs = counts / len(arr)
    # Shannon熵: H = -Σ p_i * log2(p_i)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return float(entropy)


def compute_energy_gradient(grid: Any, energy_fn: Optional[callable] = None
                           ) -> np.ndarray:
    """
    能量梯度场 — 网格每点的局部能量变化率

    κ-Phase: 能量梯度 = κ-相位场的方向导数
    最小能量路径是κ-相位坍缩的自然路径

    Args:
        grid: 2D网格
        energy_fn: 可选自定义能量函数，默认使用颜色值

    Returns:
        梯度数组 (2, H, W) — [∂E/∂x, ∂E/∂y]
    """
    arr = np.array(grid, dtype=float)
    if energy_fn is not None:
        arr = energy_fn(arr)

    # 中心差分梯度
    grad_x = np.zeros_like(arr)
    grad_y = np.zeros_like(arr)

    if arr.ndim == 2:
        h, w = arr.shape
        # ∂E/∂x: 水平梯度
        grad_x[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / 2.0
        grad_x[:, 0] = arr[:, 1] - arr[:, 0]
        grad_x[:, -1] = arr[:, -1] - arr[:, -2]
        # ∂E/∂y: 垂直梯度
        grad_y[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / 2.0
        grad_y[0, :] = arr[1, :] - arr[0, :]
        grad_y[-1, :] = arr[-1, :] - arr[-2, :]

    return np.stack([grad_x, grad_y], axis=0)


def grid_heat_diffusion(grid: Any, iterations: int = 5,
                         diffusion_rate: float = 0.25,
                         source_mask: Optional[np.ndarray] = None
                         ) -> np.ndarray:
    """
    网格热扩散 — 离散化的Fourier热传导方程

    κ-Phase: 热扩散 = κ-相位均衡化过程
    每步: T[i,j] += α * Σ_neighbors(T[n] - T[i,j])
    稳态解 = κ-相位场的全局均衡态

    Args:
        grid: 2D网格（初始温度场）
        iterations: 扩散步数
        diffusion_rate: 扩散系数α (0~0.25, 保证稳定性)
        source_mask: 持续热源掩模（True=固定热源，不随扩散变化）

    Returns:
        扩散后的温度场
    """
    T = np.array(grid, dtype=float)
    if source_mask is None:
        source_mask = np.zeros_like(T, dtype=bool)

    for _ in range(iterations):
        T_new = T.copy()
        # 四邻居差分
        T_new[1:, :] += diffusion_rate * (T[:-1, :] - T[1:, :])   # 上
        T_new[:-1, :] += diffusion_rate * (T[1:, :] - T[:-1, :])   # 下
        T_new[:, 1:] += diffusion_rate * (T[:, :-1] - T[:, 1:])    # 左
        T_new[:, :-1] += diffusion_rate * (T[:, 1:] - T[:, :-1])   # 右
        # 热源位置保持不变
        T_new[source_mask] = T[source_mask]
        T = T_new

    return T


def minimum_energy_path(grid: Any, start: Tuple[int, int],
                         end: Tuple[int, int],
                         obstacle_chars: Set[int] = None
                         ) -> Optional[List[Tuple[int, int]]]:
    """
    最小能量路径 — 梯度引导的Dijkstra最短路径

    κ-Phase: 最小能量路径 = κ-相位场中的自然坍缩路径
    能量 = 颜色值权重 + 曼哈距离 + 障碍惩罚

    Args:
        grid: 2D网格
        start: 起点 (x, y)
        end: 终点 (x, y)
        obstacle_chars: 障碍物颜色集合

    Returns:
        路径坐标列表，或 None
    """
    if obstacle_chars is None:
        obstacle_chars = {0}

    arr = np.array(grid) if hasattr(grid, 'shape') else np.array(grid)
    h, w = arr.shape

    # Dijkstra with energy-weighted edges
    import heapq
    dist_map = {}
    prev_map = {}
    sx, sy = start
    ex, ey = end

    dist_map[(sx, sy)] = 0.0
    heap = [(0.0, sx, sy)]

    while heap:
        d, cx, cy = heapq.heappop(heap)
        if (cx, cy) == (ex, ey):
            # 回溯路径
            path = []
            cur = (ex, ey)
            while cur in prev_map:
                path.append(cur)
                cur = prev_map[cur]
            path.append(start)
            path.reverse()
            return path

        if d > dist_map.get((cx, cy), float('inf')):
            continue

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            cell = int(arr[ny, nx])
            if cell in obstacle_chars:
                continue
            # 能量权重：颜色值越高能量越高（更难穿越）
            edge_weight = 1.0 + float(cell) * 0.1
            new_dist = d + edge_weight
            if new_dist < dist_map.get((nx, ny), float('inf')):
                dist_map[(nx, ny)] = new_dist
                prev_map[(nx, ny)] = (cx, cy)
                heapq.heappush(heap, (new_dist, nx, ny))

    return None  # 不可达


# ============================================================
# 波动力学原语 (Wave) — 离散波传播、干涉、叠加
# κ-Phase: 波 = κ-相位传播的BFS wavefront模型
# 网格上的波是离散传播的，遵守叠加原理
# ============================================================

def wave_propagate_bfs(grid: Any, sources: List[Tuple[int, int]],
                        max_steps: int = 50,
                        speed: float = 1.0,
                        obstacle_chars: Set[int] = None
                        ) -> np.ndarray:
    """
    BFS波传播 — 从多个源点向外扩散的wavefront模型

    κ-Phase: 波传播 = κ-相位以恒定速率从源点向外扩散
    BFS wavefront = κ-相位前沿的离散化
    到达时间 = 相位延迟 = |源距离| / 速度

    Args:
        grid: 2D网格
        sources: 波源位置列表
        max_steps: 最大传播步数
        speed: 波速（曼哈距离/步）
        obstacle_chars: 障碍物颜色集合

    Returns:
        到达时间场 (H, W) — -1表示不可达，0表示源点
    """
    if obstacle_chars is None:
        obstacle_chars = {0}

    arr = np.array(grid) if hasattr(grid, 'shape') else np.array(grid)
    h, w = arr.shape
    arrival = np.full((h, w), -1, dtype=float)

    # BFS多源初始化
    from collections import deque
    queue = deque()
    for sx, sy in sources:
        if 0 <= sx < w and 0 <= sy < h:
            arrival[sy, sx] = 0.0
            queue.append((sx, sy, 0))

    while queue:
        cx, cy, t = queue.popleft()
        if t >= max_steps:
            continue
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            cell = int(arr[ny, nx])
            if cell in obstacle_chars or arrival[ny, nx] >= 0:
                continue
            arrival[ny, nx] = t + 1.0 / speed
            queue.append((nx, ny, t + 1))

    return arrival


def wave_interference_check(wave1: np.ndarray, wave2: np.ndarray,
                             threshold: float = 0.1
                             ) -> np.ndarray:
    """
    波干涉检测 — 两个波场的叠加结果

    κ-Phase: 干涉 = κ-相位叠加（constructive/destructive）
    constructive: Δ相位≈0 → 振幅增强
    destructive: Δ相位≈π → 振幅抵消

    Args:
        wave1, wave2: 两个波场（到达时间）
        threshold: 干涉检测阈值

    Returns:
        干涉图 (H, W): 值>0=增强, 值<0=抵消, 值≈0=无干涉
    """
    # 叠加：两波振幅之和
    # 将到达时间转换为相位: φ = 2π * t / λ
    phase1 = 2 * np.pi * wave1 / 10.0  # λ=10步
    phase2 = 2 * np.pi * wave2 / 10.0

    # 振幅叠加 (简谐波 A*sin(φ))
    amp1 = np.sin(phase1)
    amp2 = np.sin(phase2)

    # 掩模：不可达区域相位=0
    mask = (wave1 >= 0) & (wave2 >= 0)
    interference = np.zeros_like(wave1)
    interference[mask] = amp1[mask] + amp2[mask]

    # 增强区: |叠加| > 单波最大值
    # 抵消区: |叠加| < threshold
    return interference


def wave_superposition(waves: List[np.ndarray]) -> np.ndarray:
    """
    多波叠加原理 — N个波场的振幅线性叠加

    κ-Phase: superposition = κ-相位场的线性叠加（物理基本公理）
    总振幅 = Σ A_i * sin(φ_i)

    Args:
        waves: 波场列表（到达时间数组）

    Returns:
        叠加后的总振幅场
    """
    if len(waves) == 0:
        return np.zeros((1, 1))

    result = np.zeros_like(waves[0])
    for w in waves:
        phase = 2 * np.pi * w / 10.0
        mask = w >= 0
        contribution = np.zeros_like(w)
        contribution[mask] = np.sin(phase[mask])
        result += contribution

    return result


def wave_resonance_positions(wave: np.ndarray, wavelength: int = 10,
                              amplitude_threshold: float = 1.5
                              ) -> List[Tuple[int, int]]:
    """
    波共振位置检测 — 找到叠加振幅超过阈值的点

    κ-Phase: 共振 = κ-相位驻波节点（能量集中点）

    Args:
        wave: 波场
        wavelength: 波长参数
        amplitude_threshold: 振幅阈值

    Returns:
        共振位置列表 [(x, y), ...]
    """
    phase = 2 * np.pi * wave / wavelength
    mask = wave >= 0
    amplitude = np.zeros_like(wave)
    amplitude[mask] = np.abs(np.sin(phase[mask]))

    positions = []
    if amplitude.ndim == 2:
        h, w = amplitude.shape
        for y in range(h):
            for x in range(w):
                if mask[y, x] and amplitude[y, x] >= amplitude_threshold:
                    positions.append((x, y))

    return positions


# ============================================================
# 欧几里得几何原语 (Euclidean Geometry) — 距离、线段、凸包
# κ-Phase: 几何 = κ-相位空间的度量结构
# ARC空间推理的基础：距离度量、形状识别、几何变换
# ============================================================

def distance_l1(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """
    L1距离（曼哈顿距离） — 网格运动的基本度量

    κ-Phase: L1 = κ-相位差绝对值之和（|Δx|+|Δy|）

    Args:
        p1, p2: 两点坐标

    Returns:
        曼哈顿距离
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def distance_l2(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    L2距离（欧几里得距离） — 连续空间的度量

    κ-Phase: L2 = κ-相位差的Euclidean范数

    Args:
        p1, p2: 两点坐标

    Returns:
        欧几里得距离
    """
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


def distance_linf(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """
    L∞距离（切比雪夫距离） — 8方向运动的基本度量

    κ-Phase: L∞ = κ-相位差的最大绝对值（max(|Δx|, |Δy|))

    Args:
        p1, p2: 两点坐标

    Returns:
        切比雪夫距离
    """
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def line_segment_intersection(a1: Tuple[int, int], a2: Tuple[int, int],
                               b1: Tuple[int, int], b2: Tuple[int, int]
                               ) -> Optional[Tuple[float, float]]:
    """
    线段交点检测 — 判断两条线段是否相交并计算交点

    κ-Phase: 线段交点 = κ-相位路径的交汇点

    Args:
        a1, a2: 线段A的端点
        b1, b2: 线段B的端点

    Returns:
        交点坐标 (x, y)，或 None（不相交）
    """
    # 参数化: P = a1 + t*(a2-a1), Q = b1 + s*(b2-b1)
    dx_a = a2[0] - a1[0]
    dy_a = a2[1] - a1[1]
    dx_b = b2[0] - b1[0]
    dy_b = b2[1] - b1[1]

    # 交叉行列式
    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) < 1e-10:
        return None  # 平行或共线

    dx_c = b1[0] - a1[0]
    dy_c = b1[1] - a1[1]

    t = (dx_c * dy_b - dy_c * dx_b) / denom
    s = (dx_c * dy_a - dy_c * dx_a) / denom

    if 0 <= t <= 1 and 0 <= s <= 1:
        ix = a1[0] + t * dx_a
        iy = a1[1] + t * dy_a
        return (ix, iy)

    return None


def convex_hull_2d(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    2D凸包计算 — Graham扫描算法

    κ-Phase: 凸包 = κ-相位空间的最小闭合区域
    对象的凸包是其外轮廓的κ-相位边界

    Args:
        points: 点集 [(x,y), ...]

    Returns:
        凸包顶点列表（逆时针顺序）
    """
    if len(points) <= 2:
        return list(points)

    # 找最低点（y最小，y相同取x最小）
    pivot = min(points, key=lambda p: (p[1], p[0]))

    # 按极角排序
    def polar_angle(p):
        dx = p[0] - pivot[0]
        dy = p[1] - pivot[1]
        return float(np.arctan2(dy, dx))

    sorted_pts = sorted(points, key=polar_angle)

    # Graham扫描
    hull = []
    for p in sorted_pts:
        while len(hull) >= 2:
            # 检查是否左转
            o, a = hull[-2], hull[-1]
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)

    return hull


def polygon_area(vertices: List[Tuple[int, int]]) -> float:
    """
    多边形面积 — Shoelace公式

    κ-Phase: 面积 = κ-相位区域的总幅值（Shoelace = 行列式展开）

    Args:
        vertices: 多边形顶点列表（逆时针）

    Returns:
        面积值（正=逆时针，负=顺时针）
    """
    if len(vertices) < 3:
        return 0.0

    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    return abs(area) / 2.0


def geometric_similarity(shape1: List[Tuple[int, int]],
                          shape2: List[Tuple[int, int]],
                          tolerance: float = 0.1
                          ) -> float:
    """
    几何相似性检测 — D4旋转+平移下的形状匹配

    κ-Phase: similarity = κ-陪集匹配度（D4群×位移陪集中的最佳匹配）
    两形状相似 = 存在κ-变换使得κ-Snap归约误差<tolerance

    Args:
        shape1, shape2: 形状点集
        tolerance: 匹配容差

    Returns:
        最佳匹配分数 (0.0~1.0)
    """
    if len(shape1) == 0 or len(shape2) == 0:
        return 0.0
    if len(shape1) != len(shape2):
        return 0.0

    best_score = 0.0
    s1 = np.array(shape1, dtype=float)

    # D4变换: 8种旋转+镜像
    transforms = [
        lambda p: p,                             # identity
        lambda p: np.column_stack([-p[:,1], p[:,0]]),   # rot90
        lambda p: np.column_stack([-p[:,0], -p[:,1]]),  # rot180
        lambda p: np.column_stack([p[:,1], -p[:,0]]),   # rot270
        lambda p: np.column_stack([p[:,0], -p[:,1]]),   # mirror_x
        lambda p: np.column_stack([-p[:,0], p[:,1]]),   # mirror_y
        lambda p: np.column_stack([-p[:,1], -p[:,0]]),  # mirror_diag1
        lambda p: np.column_stack([p[:,1], p[:,0]]),    # mirror_diag2
    ]

    s2 = np.array(shape2, dtype=float)

    for transform in transforms:
        s2_t = transform(s2)
        # 最优平移: 中心对齐
        center1 = s1.mean(axis=0)
        center2 = s2_t.mean(axis=0)
        offset = center1 - center2
        s2_aligned = s2_t + offset

        # 匹配分数: 归一化距离误差
        dists = np.sqrt(np.sum((s1 - s2_aligned)**2, axis=1))
        max_dist = max(dists.max(), 1.0)
        score = 1.0 - dists.mean() / max_dist
        best_score = max(best_score, score)

    return float(best_score)


def vector_add(v1: Tuple[int, int], v2: Tuple[int, int]) -> Tuple[int, int]:
    """向量加法 — κ-相位合成"""
    return (v1[0] + v2[0], v1[1] + v2[1])


def vector_sub(v1: Tuple[int, int], v2: Tuple[int, int]) -> Tuple[int, int]:
    """向量减法 — κ-相位差"""
    return (v1[0] - v2[0], v1[1] - v2[1])


def vector_scale(v: Tuple[int, int], scalar: float) -> Tuple[float, float]:
    """向量缩放 — κ-相位幅值调整"""
    return (v[0] * scalar, v[1] * scalar)


def vector_dot(v1: Tuple[int, int], v2: Tuple[int, int]) -> int:
    """向量内积 — κ-相位相关性"""
    return v1[0] * v2[0] + v1[1] * v2[1]


def vector_cross_2d(v1: Tuple[int, int], v2: Tuple[int, int]) -> int:
    """2D向量叉积 — κ-相位旋度"""
    return v1[0] * v2[1] - v1[1] * v2[0]


# ============================================================
# 离散三角函数原语 (Trigonometry Discrete)
# κ-Phase: 方向/角度 = κ-相位空间的8方向离散化
# 连续sin/cos对网格ARC过度，但方向/向量转换是必要的
# ============================================================

# 8方向系统 — ARC网格的标准方向集
DIRECTION_8 = {
    0: (0, -1),     # N (上)
    1: (1, -1),     # NE (右上)
    2: (1, 0),      # E (右)
    3: (1, 1),      # SE (右下)
    4: (0, 1),      # S (下)
    5: (-1, 1),     # SW (左下)
    6: (-1, 0),     # W (左)
    7: (-1, -1),    # NW (左上)
}

DIRECTION_4 = {
    0: (0, -1),     # N (上)
    1: (1, 0),      # E (右)
    2: (0, 1),      # S (下)
    3: (-1, 0),     # W (左)
}


def direction_angle_8(dx: int, dy: int) -> int:
    """
    计算8方向角度 — 从位移向量到方向编码

    κ-Phase: 方向编码 = κ-相位角度的8级量化

    Args:
        dx, dy: 位移分量

    Returns:
        方向编码 0-7 (0=N, 1=NE, ..., 7=NW)
    """
    if dx == 0 and dy == 0:
        return 0  # 默认北

    angle = int(np.arctan2(dy, dx) * 180 / np.pi)
    # 映射到8方向: 0°=E, 45°=NE, 90°=N, ...
    # 重新映射: N=0(+y↑→dy<0), 顺时针递增
    if dy < 0 and dx == 0:    return 0  # N
    if dy < 0 and dx > 0:     return 1  # NE
    if dx > 0 and dy == 0:    return 2  # E
    if dy > 0 and dx > 0:     return 3  # SE
    if dy > 0 and dx == 0:    return 4  # S
    if dy > 0 and dx < 0:     return 5  # SW
    if dx < 0 and dy == 0:    return 6  # W
    if dy < 0 and dx < 0:     return 7  # NW
    return 0


def direction_to_vector(dir_code: int, use_8: bool = True) -> Tuple[int, int]:
    """
    方向编码到向量 — 从方向编码还原位移向量

    κ-Phase: 方向编码 → κ-相位位移向量

    Args:
        dir_code: 方向编码
        use_8: 使用8方向(True)或4方向(False)

    Returns:
        (dx, dy) 位移向量
    """
    dirs = DIRECTION_8 if use_8 else DIRECTION_4
    return dirs.get(dir_code % len(dirs), (0, 0))


def polar_to_cartesian(radius: float, angle_deg: float) -> Tuple[float, float]:
    """
    极坐标→笛卡尔坐标转换

    κ-Phase: polar→cartesian = κ-相位从角域到线域的变换

    Args:
        radius: 极径
        angle_deg: 角度（度）

    Returns:
        (x, y) 笛卡尔坐标
    """
    rad = np.deg2rad(angle_deg)
    return (radius * np.cos(rad), radius * np.sin(rad))


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """
    笛卡尔坐标→极坐标转换

    κ-Phase: cartesian→polar = κ-相位从线域到角域的变换

    Args:
        x, y: 笛卡尔坐标

    Returns:
        (radius, angle_deg) 极坐标
    """
    radius = float(np.sqrt(x**2 + y**2))
    angle_deg = float(np.rad2deg(np.arctan2(y, x)))
    return (radius, angle_deg)


def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """
    两向量夹角 — 离散化的角度距离

    κ-Phase: 夹角 = κ-相位间的角度距离
    cos(θ) = (v1·v2) / (|v1||v2|)

    Args:
        v1, v2: 两个向量

    Returns:
        夹角（度）
    """
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = float(np.sqrt(v1[0]**2 + v1[1]**2))
    mag2 = float(np.sqrt(v2[0]**2 + v2[1]**2))
    if mag1 < 1e-10 or mag2 < 1e-10:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return float(np.rad2deg(np.arccos(cos_theta)))


def vector_decompose(v: Tuple[float, float],
                     basis: Tuple[float, float]
                     ) -> Tuple[float, float]:
    """
    向量分解 — 将向量投影到给定基向量方向

    κ-Phase: 分解 = κ-相位沿基方向的投影
    v = proj_along * basis + perp_component

    Args:
        v: 待分解向量
        basis: 基向量方向

    Returns:
        (parallel_component, perpendicular_component) 沿基方向和垂直分量
    """
    b_mag_sq = basis[0]**2 + basis[1]**2
    if b_mag_sq < 1e-10:
        return (0.0, 0.0)
    # 沿基方向投影
    proj_scalar = (v[0]*basis[0] + v[1]*basis[1]) / b_mag_sq
    proj_parallel = (proj_scalar * basis[0], proj_scalar * basis[1])
    # 垂直分量
    proj_perp = (v[0] - proj_parallel[0], v[1] - proj_parallel[1])
    return (proj_scalar, float(np.sqrt(proj_perp[0]**2 + proj_perp[1]**2)))


def discrete_sin(angle_steps: int, total_steps: int = 8) -> float:
    """
    离散正弦 — 8方向量化角度的sin值

    κ-Phase: discrete_sin = κ-相位角度量化后的三角函数值
    只取8个标准角度的sin值，避免连续三角函数的复杂性

    Args:
        angle_steps: 角度步数 (0~total_steps-1)
        total_steps: 总方向数 (默认8)

    Returns:
        sin值近似
    """
    # 8方向sin预计算
    if total_steps == 8:
        sin_table = [0.0, 0.707, 1.0, 0.707, 0.0, -0.707, -1.0, -0.707]
        return sin_table[angle_steps % 8]
    # 4方向sin预计算
    if total_steps == 4:
        sin_table = [1.0, 0.0, -1.0, 0.0]  # N/E/S/W (dy分量)
        return sin_table[angle_steps % 4]
    # 通用：用连续sin
    angle = 2 * np.pi * angle_steps / total_steps
    return float(np.sin(angle))


def discrete_cos(angle_steps: int, total_steps: int = 8) -> float:
    """
    离散余弦 — 8方向量化角度的cos值

    κ-Phase: discrete_cos = κ-相位角度量化后的三角函数值

    Args:
        angle_steps: 角度步数
        total_steps: 总方向数

    Returns:
        cos值近似
    """
    if total_steps == 8:
        cos_table = [1.0, 0.707, 0.0, -0.707, -1.0, -0.707, 0.0, 0.707]
        return cos_table[angle_steps % 8]
    if total_steps == 4:
        cos_table = [0.0, 1.0, 0.0, -1.0]  # N/E/S/W (dx分量)
        return cos_table[angle_steps % 4]
    angle = 2 * np.pi * angle_steps / total_steps
    return float(np.cos(angle))


# ============================================================
# 导出
# ============================================================

PHYSICS_PRIMITIVE_REGISTRY = {
    'newton_push': {
        'functions': [can_push_box, is_box_at, is_deadlock_corner, is_wall_ride_push, check_deadlock_with_wall_ride],
        'description': '牛顿刚体推箱 — 质量、摩擦、dead-lock冻结 + Wall-Ride豁免',
        'games': ['ka59'],
    },
    'mirror_geo': {
        'functions': [mirror_point, reflect_ray, multi_mirror_trace],
        'description': '反射几何 — 八元数仿射镜像、光线追踪',
        'games': ['ar25'],
    },
    'optics': {
        'functions': [OpticsMirror, OpticsTarget, OpticsPiece,
                      optics_ray_trace, optics_coverage_map, optics_check_win,
                      optics_mirror_move_constraint],
        'description': '光学物理原语 — BFS光线追踪 + 覆盖图 + 镜面移动约束 (精确匹配AR25游戏源码)',
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
    'kinematics': {
        'functions': [compute_trajectory, compute_parabolic_trajectory,
                      predict_intercept_position, velocity_from_displacement],
        'description': '运动学原语 — 轨迹计算、速度向量、加速度、抛物线弹道、拦截预测',
        'games': [],  # 通用原语，适用于所有移动类游戏
    },
    'collision': {
        'functions': [check_rect_collision, elastic_collision_1d, inelastic_collision_1d,
                      grid_collision_detect],
        'description': '碰撞检测原语 — AABB矩形碰撞、弹性/非弹性碰撞(动量守恒)、网格碰撞检测',
        'games': [],  # 通用原语，适用于所有碰撞类游戏
    },
    'thermodynamics': {
        'functions': [compute_grid_entropy, compute_energy_gradient,
                      grid_heat_diffusion, minimum_energy_path],
        'description': '热力学原语 — Shannon熵、能量梯度场、Fourier热扩散、Dijkstra最小能量路径',
        'games': [],  # 通用原语，适用于能量/熵优化类搜索
    },
    'wave': {
        'functions': [wave_propagate_bfs, wave_interference_check,
                      wave_superposition, wave_resonance_positions],
        'description': '波动力学原语 — BFS波传播(wavefront)、干涉检测(constructive/destructive)、叠加原理、共振检测',
        'games': [],  # 通用原语，适用于传播/影响域类问题
    },
    'euclidean_geometry': {
        'functions': [distance_l1, distance_l2, distance_linf,
                      line_segment_intersection, convex_hull_2d, polygon_area,
                      geometric_similarity,
                      vector_add, vector_sub, vector_scale, vector_dot, vector_cross_2d],
        'description': '欧几里得几何原语 — L1/L2/L∞距离度量、线段交点、凸包(Graham扫描)、Shoelace面积、D4×平移几何相似性、向量运算',
        'games': [],  # 通用原语，ARC空间推理的基础
    },
    'trigonometry_discrete': {
        'functions': [direction_angle_8, direction_to_vector,
                      polar_to_cartesian, cartesian_to_polar,
                      angle_between_vectors, vector_decompose,
                      discrete_sin, discrete_cos],
        'description': '离散三角函数原语 — 8方向编码/解码、极坐标↔笛卡尔、向量夹角、向量分解、离散sin/cos表',
        'games': [],  # 通用原语，适用于方向/角度类推理
    },
}

def get_primitive_functions(primitive_name: str) -> List[Any]:
    """获取指定物理原语的所有函数"""
    entry = PHYSICS_PRIMITIVE_REGISTRY.get(primitive_name)
    if entry is None:
        return []
    return entry['functions']
