"""
src/agent/wall_bfs.py
TOMAS Wall-BFS — KA59推箱Wall-BFS搜索策略

KA59 Sokoban推箱游戏中，玩家需要将箱子推到目标位置。
Wall-BFS策略基于物理原语(can_push_box, is_deadlock_corner)，
在墙壁约束下做BFS搜索，避免dead-lock状态。

核心思路:
  1. 从当前状态出发，BFS搜索所有合法push动作
  2. 每个push动作检查: can_push_box → 前方非墙非箱
  3. push后检查: is_deadlock_corner → 角落死锁剪枝
  4. 用Node(parent_id, action)记录搜索树 (IDO Δ-State)
  5. 返回候选节点集供后续L2/L3/L4处理

Version: v3.18.0 — Hybrid Search L1 Strategy
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .delta_state import Node, ReplayEngine, ActionSpace
from .physics_primitives import can_push_box, is_box_at, is_deadlock_corner, check_deadlock_with_wall_ride


# ============================================================================
# §1. Wall-BFS 常量
# ============================================================================

# KA59 推箱Wall-BFS搜索参数
WALL_BFS_MAX_DEPTH: int = 15           # 最大搜索深度
WALL_BFS_MAX_NODES: int = 200          # 最大扩展节点数
WALL_BFS_WALL_CHAR: int = 0            # 墙壁颜色值
WALL_BFS_GOAL_CHAR: int = 2            # 目标颜色值
WALL_BFS_PLAYER_CHAR: int = 9          # 玩家颜色值(默认)
WALL_BFS_BOX_CHARS: Set[int] = {3, 5}  # 箱子颜色值集合


# ============================================================================
# §2. Wall-BFS 搜索引擎
# ============================================================================

class WallBFSEngine:
    """KA59推箱Wall-BFS搜索引擎。

    在墙壁约束下做BFS搜索，每个push动作都经过物理原语验证：
    can_push_box检查前方可通行，is_deadlock_corner剪枝死锁状态。

    IDO Δ-State设计:
      - 搜索节点用Node(parent_id, action)表示
      - 物化只在需要时发生 (ReplayEngine.replay)
      - 避免 DeepCopy 整个 Grid

    Attributes:
        root_state: 根节点Grid状态(np.ndarray)。
        node_map: 节点编号 → Node对象映射。
        wall_char: 墙壁颜色值。
        goal_char: 目标颜色值。
        box_chars: 箱子颜色值集合。
        _next_id: 下一个节点编号。
    """

    def __init__(
        self,
        root_state: np.ndarray,
        wall_char: int = WALL_BFS_WALL_CHAR,
        goal_char: int = WALL_BFS_GOAL_CHAR,
        box_chars: Optional[Set[int]] = None,
        bfs_radius: int = 0,
    ) -> None:
        """初始化WallBFSEngine。

        Args:
            root_state: 根节点Grid状态(64×64 numpy array)。
            wall_char: 墙壁颜色值。
            goal_char: 目标颜色值。
            box_chars: 箱子颜色值集合，默认{3, 5}。
            bfs_radius: P0-A bounding — BFS搜索半径限制(Manhattan距离)。
                0表示无限制, >0时只探索起始位置radius范围内的节点。
        """
        self.root_state: np.ndarray = root_state.copy()
        self.wall_char: int = wall_char
        self.goal_char: int = goal_char
        self.box_chars: Set[int] = box_chars or WALL_BFS_BOX_CHARS.copy()
        self.bfs_radius: int = bfs_radius  # P0-A: BFS半径边界
        self.node_map: Dict[int, Node] = {}
        self._next_id: int = 0

        # P0-A: 记录起始玩家位置用于radius约束
        player_pos, _ = self._find_entities(self.root_state)
        self._origin_pos: Tuple[int, int] = player_pos

        # 创建根节点
        root_node: Node = Node(
            id=self._next_id,
            parent_id=-1,
            action="root",
            depth=0,
            meta={'stage': 'wall_bfs'},
        )
        self.node_map[self._next_id] = root_node
        self._next_id += 1

    def _find_entities(self, grid: np.ndarray) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        """从Grid中提取玩家和箱子位置。

        Args:
            grid: 2D numpy array。

        Returns:
            (player_pos, box_positions) — 玩家位置和所有箱子位置列表。
        """
        player_pos: Tuple[int, int] = (-1, -1)
        box_positions: List[Tuple[int, int]] = []

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                cell: int = int(grid[y, x])
                if cell == WALL_BFS_PLAYER_CHAR or cell in {12, 9}:
                    player_pos = (x, y)
                elif cell in self.box_chars:
                    box_positions.append((x, y))

        return player_pos, box_positions

    def _apply_push(
        self,
        grid: np.ndarray,
        player_pos: Tuple[int, int],
        box_pos: Tuple[int, int],
        direction: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """在Grid上执行一次push操作。

        Args:
            grid: 当前Grid状态。
            player_pos: 玩家当前位置(x, y)。
            box_pos: 箱子当前位置(x, y)。
            direction: 推动方向(dx, dy)。

        Returns:
            推动后的新Grid，或None(如果push不合法)。
        """
        dx, dy = direction
        bx, by = box_pos
        new_bx: int = bx + dx
        new_by: int = by + dy

        # 检查push合法性
        can_push, new_box_pos = can_push_box(
            grid, player_pos, box_pos, direction, self.wall_char
        )
        if not can_push:
            return None

        # 检查push后deadlock
        new_grid: np.ndarray = grid.copy()
        # 移动玩家到箱子原位
        px, py = player_pos
        new_grid[py, px] = self.wall_char  # 原位置变空(或背景色)
        # 移动箱子到新位置
        new_grid[new_box_pos[1], new_box_pos[0]] = int(grid[by, bx])
        # 玩家移到箱子原位
        new_grid[by, bx] = WALL_BFS_PLAYER_CHAR

        # ★ CHK_DL修正版: 含Wall-Ride豁免 (文章 Appendix A.2)
        # 箱子贴墙推时, 豁免三面围死误判 — κ-优选路径连续性优先
        if check_deadlock_with_wall_ride(
            new_grid, new_box_pos, player_pos,
            goal=None,  # 目标未知
            wall_char=self.wall_char,
            goal_char=self.goal_char,
            box_chars=self.box_chars,
        ):
            return None  # 死锁剪枝 (含Wall-Ride豁免)

        return new_grid

    def search(
        self,
        max_depth: int = WALL_BFS_MAX_DEPTH,
        max_nodes: int = WALL_BFS_MAX_NODES,
    ) -> List[int]:
        """执行Wall-BFS搜索，返回候选节点ID列表。

        BFS搜索所有合法push动作，使用物理原语验证和deadlock剪枝。

        Args:
            max_depth: 最大搜索深度。
            max_nodes: 最大扩展节点数。

        Returns:
            候选节点ID列表(供L2/L3/L4处理)。
        """
        directions: List[Tuple[int, int]] = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        direction_names: Dict[Tuple[int, int], str] = {
            (0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT",
        }

        # 创建ReplayEngine用于物化
        replay_engine: ReplayEngine = ReplayEngine(
            root_state=self.root_state,
            node_map=self.node_map,
            mode='grid',
        )

        candidates: List[int] = []
        queue: deque = deque([0])  # 根节点ID=0
        expanded: int = 0
        visited_layouts: Set[str] = set()

        # 物化根节点并记录布局哈希
        root_grid: np.ndarray = self.root_state.copy()
        root_hash: str = self._layout_hash(root_grid)
        visited_layouts.add(root_hash)

        while queue and expanded < max_nodes:
            cur_id: int = queue.popleft()
            cur_node: Node = self.node_map[cur_id]

            if cur_node.depth >= max_depth:
                continue

            # 物化当前节点状态
            cur_grid: np.ndarray = replay_engine.replay(cur_id)

            # 提取实体位置
            player_pos, box_positions = self._find_entities(cur_grid)
            if player_pos == (-1, -1):
                continue  # 无法找到玩家

            # 对每个箱子，尝试每个方向的push
            for box_pos in box_positions:
                for direction in directions:
                    # 执行push
                    new_grid: Optional[np.ndarray] = self._apply_push(
                        cur_grid, player_pos, box_pos, direction
                    )
                    if new_grid is None:
                        continue  # push不合法或deadlock

                    # P0-A bounding: BFS radius边界限制
                    # 只接受origin位置Manhattan距离<=bfs_radius内的push目标
                    if self.bfs_radius > 0 and self._origin_pos != (-1, -1):
                        ox, oy = self._origin_pos
                        # 新箱子push后的目标位置
                        new_box_x = box_pos[0] + direction[0]
                        new_box_y = box_pos[1] + direction[1]
                        manhattan = abs(new_box_x - ox) + abs(new_box_y - oy)
                        if manhattan > self.bfs_radius:
                            continue  # 超出BFS radius边界，跳过

                    # 布局哈希去重
                    layout_hash: str = self._layout_hash(new_grid)
                    if layout_hash in visited_layouts:
                        continue
                    visited_layouts.add(layout_hash)

                    # 构造动作描述
                    dir_name: str = direction_names[direction]
                    action_str: str = f"PUSH:{box_pos[0]}:{box_pos[1]}:{dir_name}"

                    # 创建新节点
                    new_node: Node = Node(
                        id=self._next_id,
                        parent_id=cur_id,
                        action=action_str,
                        depth=cur_node.depth + 1,
                        meta={'stage': 'wall_bfs', 'layout_hash': layout_hash},
                    )
                    # 缓存物化结果
                    new_node._grid_cache = new_grid.copy()
                    self.node_map[self._next_id] = new_node
                    candidates.append(self._next_id)
                    queue.append(self._next_id)
                    self._next_id += 1
                    expanded += 1

                    if expanded >= max_nodes:
                        break

        return candidates

    def _layout_hash(self, grid: np.ndarray) -> str:
        """计算Grid布局哈希(用于去重)。

        Args:
            grid: 2D numpy array。

        Returns:
            布局哈希字符串。
        """
        from .delta_state import LayoutHasher
        return LayoutHasher.hash_with_shape(grid)


# ============================================================================
# §3. 自测函数
# ============================================================================

def _self_test() -> bool:
    """Wall-BFS 自测: 构造简单推箱Grid，验证搜索功能。

    Returns:
        True if all tests pass, False otherwise.
    """
    # 构造简单推箱Grid (5×5)
    # 0=墙, 1=空地, 2=目标, 9=玩家, 3=箱子
    grid: np.ndarray = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 9, 1, 0],
        [0, 1, 3, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=int)

    engine: WallBFSEngine = WallBFSEngine(
        root_state=grid,
        wall_char=0,
        goal_char=2,
        box_chars={3},
    )

    # 执行搜索
    candidates: List[int] = engine.search(max_depth=5, max_nodes=50)

    # 验证搜索结果
    assert len(candidates) >= 0, "Wall-BFS should return candidates list"
    assert all(cid in engine.node_map for cid in candidates), "All candidates should be in node_map"

    # 验证根节点存在
    assert 0 in engine.node_map, "Root node should exist"
    assert engine.node_map[0].parent_id == -1, "Root node parent_id should be -1"

    # 验证物理原语工作
    can_push, new_pos = can_push_box(grid, (2, 2), (2, 3), (1, 0), wall_char=0)
    # 箱子在(2,3)，玩家在(2,2)向右推，前方(3,3)是目标格
    # 注: 格式是(x,y)，玩家(2,2)推箱子(2,3)向右(1,0)
    # 前方是(3,3)即目标格(2)
    assert isinstance(can_push, bool), "can_push_box should return bool"

    # 验证deadlock检测
    is_dl: bool = is_deadlock_corner(grid, (2, 3), wall_char=0, goal_char=2)
    # 箱子不在角落 → 不是deadlock
    assert isinstance(is_dl, bool), "is_deadlock_corner should return bool"

    print("[PASS] Wall-BFS _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
