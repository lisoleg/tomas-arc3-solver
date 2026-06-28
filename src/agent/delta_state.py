"""TOMAS-ARC3 Δ-State Engine (v3.18.0)

IDO/TOMAS Framework: 流贯演化 = 不可变基态 + 可变 Δ(Diff)
Zero DeepCopy · Multi-Stage BFS · Lazy Materialization · RHAE Budget Control

Core design principles:
  1. IDO (Immutable Base + Observable Delta): Node 只记录 (parent_id, action),
     不拷贝整个状态；物化(Replay)只在需要时发生。
  2. 流贯 Replay: 从 root_state 开始逐动作 Replay 到目标节点，替代 DeepCopy。
     Game mode 用 deepcopy(root) + perform_action sequence；
     Grid mode 用 copy(root_grid) + transformation sequence。
  3. Multi-Stage BFS: 结构探索(零拷贝) → 参数探索(布局哈希去重) → κ-Priority精化。
  4. RHAE Budget Control: 卞氏 5/6 饱和阈值 (GEX_PASS_THRESHOLD = 1/6 ≈ 0.167)
     控制步数预算和提前终止。
  5. 双模式 ActionSpace: game mode (UP/DOWN/LEFT/RIGHT/ACTION6) 和
     grid mode (rotate/mirror/translate/recolor/noop) 共享统一接口。

Version: v3.18.0
TOMAS Correspondence: TOMAS Phase II → Δ-State IDO Agent
IDO Correspondence: IDO 流贯 = Immutable Base + Observable Delta
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# §1. 异常类 — RHAE 控制器触发的有意识放弃与预算耗尽
# ============================================================================

class SolverAborted(Exception):
    """由 RHAE 控制器触发的有意识放弃。

    当潜在 RHAE 低于 ABORT_RHAE_THRESHOLD (0.04) 时，
    控制器判定继续搜索无意义，主动抛出此异常终止求解。
    这不是错误，而是 IDO 框架的「有意识终止」机制。
    """
    pass


class BudgetExceeded(Exception):
    """步数预算耗尽。

    当已用步数超过 MAX_RHAE_PER_TASK × LS20_BUDGET_MULT 或
    用户指定的 max_steps 时，抛出此异常。
    防止无限搜索，确保计算资源可控。
    """
    pass


# ============================================================================
# §2. 全局常量 — 卞氏 5/6 饱和阈值 + IDO 流贯参数
# ============================================================================

# RHAE (Remaining Heuristic Action Estimate) 最大预算
# 每个任务最多消耗 1.0 个 RHAE 单位
MAX_RHAE_PER_TASK: float = 1.0

# 卞氏 5/6 饱和阈值 (Theorem 1: 5/6 saturation threshold)
# GEX ≥ GEX_PASS_THRESHOLD → 通过 (结构足够接近目标)
# GEX < GEX_FAIL_THRESHOLD → 失败 (偏离太大)
GEX_PASS_THRESHOLD: float = 1.0 - 5.0 / 6.0   # = 1/6 ≈ 0.1667
GEX_FAIL_THRESHOLD: float = 1.0 / 6.0           # = 1/6 ≈ 0.1667 (对称阈值)

# LS20 压力测试乘数 — 预算放大 2.5 倍以应对复杂任务
LS20_BUDGET_MULT: float = 2.5

# IDO 搜索参数
DEFAULT_MAX_DEPTH: int = 5          # 最大搜索深度
MAX_REPLAY_CACHE: int = 128         # Replay 缓存上限 (防止内存溢出)

# 多阶段 BFS 预算
STRUCT_BFS_BUDGET: int = 40         # 阶段1: 结构探索最大扩展节点数
PARAM_BFS_BUDGET: int = 100         # 阶段2: 参数探索最大物化节点数

# RHAE 预算控制阈值
MIN_RHAE_TO_KEEP: float = 0.01      # RHAE 低于此值直接丢弃 (不值得继续)
ABORT_RHAE_THRESHOLD: float = 0.04  # 潜在 RHAE 低于此值提前终止 (SolverAborted)

# 背景剪枝阈值 — 卞氏 5/6 阈值的 1/2
# 用于 κ-Priority 排序中过滤低价值节点
DEAD_ZERO_RATIO: float = (1.0 - 5.0 / 6.0) / 2.0  # ≈ 0.0833 ≈ 8.3%


# ============================================================================
# §3. Node dataclass — IDO-TOMAS 不可变节点 (只记录 Δ)
# ============================================================================

@dataclass
class Node:
    """IDO-TOMAS 节点：用 (parent_id, action) 代表状态，而非拷贝整个 Grid。

    IDO 核心思想：节点不存储完整状态(Grid/Game)，只存储从父节点到本节点的
    动作增量(Delta)。状态物化(Replay)只在需要验证时才发生。

    设计决策：
      - parent_id = -1 代表根节点 (不可变基态)
      - action 在 game mode 是 action_id (int)，在 grid mode 是动作名 (str)
      - _grid_cache 和 _hash_cache 是惰性计算的私有字段，不在 repr 中显示
      - signature() 返回 "parent_id->action" 的唯一标识，用于去重和哈希

    Attributes:
        id: 节点唯一编号。
        parent_id: 父节点编号，-1 代表根节点。
        action: 执行的动作 (game mode: int, grid mode: str)。
        depth: 当前搜索深度。
        meta: 可扩展元数据字典 (IC, GEX, RHAE 等)。
        _grid_cache: 惰性物化后的 Grid 缓存 (仅 grid mode)。
        _hash_cache: 惰性计算的布局哈希缓存。
    """

    id: int
    parent_id: int                 # -1 代表根节点
    action: str                    # 执行的动作
    depth: int                     # 当前深度
    meta: Dict[str, Any] = field(default_factory=dict)
    _grid_cache: Optional[np.ndarray] = field(default=None, repr=False)
    _hash_cache: Optional[str] = field(default=None, repr=False)

    def signature(self) -> str:
        """返回节点的唯一签名标识。

        根节点返回 "root"，非根节点返回 "parent_id->action" 格式。
        此签名用于去重判断和哈希计算。

        Returns:
            节点签名字符串。
        """
        if self.parent_id == -1:
            return "root"
        return f"{self.parent_id}->{self.action}"

    def __hash__(self) -> int:
        """基于签名的哈希值，支持 set/dict 去重。"""
        return hash(self.signature())

    def __eq__(self, other: object) -> bool:
        """基于签名的等价判断。"""
        if not isinstance(other, Node):
            return NotImplemented
        return self.signature() == other.signature()


# ============================================================================
# §4. ActionSpace — 可插拔的 DSL 原语空间 (双模式: game + grid)
# ============================================================================

class ActionSpace:
    """可插拔的 DSL 原语空间 — 双模式支持。

    IDO 框架需要统一的动作接口，同时支持两种场景：
      - Game mode: ARC-AGI-3 游戏动作 (UP/DOWN/LEFT/RIGHT/ACTION6)
      - Grid mode: 通用 ARC grid transformations (rotate/mirror/translate/recolor)

    设计决策：
      - 构造时指定 mode，决定使用哪种原语集
      - get_primitives() 返回当前模式的所有可用动作
      - get_params() 根据动作类型返回可调参数范围
      - apply() 纯函数：对 Grid 施加动作，返回新 Grid (不修改输入)
      - apply_game_action() 直接在 game engine 上执行动作

    Attributes:
        GAME_ACTIONS: 游戏 mode 的动作映射 {action_id: action_name}。
        GRID_PRIMITIVES: Grid mode 的原语列表。
    """

    # Game mode: ARC-AGI-3 标准动作集
    GAME_ACTIONS: Dict[int, str] = {
        1: 'UP',
        2: 'DOWN',
        3: 'LEFT',
        4: 'RIGHT',
        6: 'ACTION6',
    }

    # Grid mode: ARC 通用变换原语集
    GRID_PRIMITIVES: List[str] = [
        'rotate_90',
        'rotate_180',
        'rotate_270',
        'mirror_h',
        'mirror_v',
        'translate',
        'recolor',
        'noop',
    ]

    def __init__(self, mode: str = 'game') -> None:
        """初始化 ActionSpace。

        Args:
            mode: 动作模式，'game' 或 'grid'。
                'game' 使用 GAME_ACTIONS (整数 ID 到动作名映射)；
                'grid' 使用 GRID_PRIMITIVES (字符串动作名列表)。

        Raises:
            ValueError: 如果 mode 不是 'game' 或 'grid'。
        """
        if mode not in ('game', 'grid'):
            raise ValueError(f"ActionSpace mode must be 'game' or 'grid', got '{mode}'")
        self._mode: str = mode
        # game mode: dict {id: name}; grid mode: list of primitive names
        self._primitives: Any = (
            self.GAME_ACTIONS if mode == 'game' else self.GRID_PRIMITIVES
        )

    def get_primitives(self) -> list:
        """返回当前模式的所有可用动作。

        Game mode 返回动作 ID 列表 [1, 2, 3, 4, 6]；
        Grid mode 返回原语名称列表 ['rotate_90', ...]。

        Returns:
            可用动作列表。
        """
        if self._mode == 'game':
            return list(self._primitives.keys())
        return list(self._primitives)

    def get_params(self, action: Any, context: Optional[Dict[str, Any]] = None) -> list:
        """根据动作类型返回可调参数范围。

        Game mode:
          - UP/DOWN/LEFT/RIGHT: 无参数 (纯方向移动)
          - ACTION6: 需要坐标参数 [(x, y)] 从 context 获取可选位置

        Grid mode:
          - rotate: 无参数 (固定角度)
          - mirror: 无参数 (固定轴)
          - translate: 偏移量 [(dx, dy)] 从 grid 尺寸推导
          - recolor: 颜色映射 [(src_color, dst_color)] 从 grid 颜色集推导
          - noop: 无参数

        Args:
            action: 动作标识 (game mode: int, grid mode: str)。
            context: 可选上下文信息 (含 grid 尺寸、颜色集等)。

        Returns:
            参数候选列表，空列表表示无参数动作。
        """
        context = context or {}

        if self._mode == 'game':
            # Game mode: ACTION6 需要 click 坐标，其他方向动作无参数
            if action == 6:
                # 从 context 提取可选目标坐标
                targets: list = context.get('click_targets', [])
                return targets
            return []  # UP/DOWN/LEFT/RIGHT 无参数

        # Grid mode 参数推导
        grid_shape: tuple = context.get('grid_shape', (0, 0))
        unique_colors: list = context.get('unique_colors', [])

        if action == 'translate':
            # 平移偏移范围：基于 grid 尺寸推导
            max_dx = max(0, grid_shape[0] - 1) if grid_shape[0] > 0 else 0
            max_dy = max(0, grid_shape[1] - 1) if grid_shape[1] > 0 else 0
            offsets: list = []
            for dx in range(-max_dx, max_dx + 1):
                for dy in range(-max_dy, max_dy + 1):
                    if dx != 0 or dy != 0:  # (0,0) 已是 noop
                        offsets.append((dx, dy))
            return offsets

        if action == 'recolor':
            # 颜色重映射候选：(src_color, dst_color)
            if len(unique_colors) < 2:
                return []
            mappings: list = []
            for src in unique_colors:
                for dst in unique_colors:
                    if src != dst:
                        mappings.append((src, dst))
            return mappings

        # rotate/mirror/noop: 无参数
        return []

    def apply(self, grid: np.ndarray, action: str, params: Optional[list] = None) -> np.ndarray:
        """纯函数：对 Grid 施加动作，返回新 Grid (不修改输入)。

        仅在 grid mode 下使用。Game mode 使用 apply_game_action()。

        所有变换都是纯函数：输入 Grid 不被修改，返回新的 ndarray。
        这是 IDO 流贯 Replay 的核心：每次 apply 都是一个 Δ 操作。

        Args:
            grid: 输入 numpy array (2D, dtype int)。
            action: 动作名称 (grid mode: 'rotate_90' 等)。
            params: 动作参数列表 (translate 的偏移量, recolor 的映射)。

        Returns:
            变换后的新 numpy array (2D, dtype int)。

        Raises:
            ValueError: 如果 action 不是 grid mode 原语。
        """
        params = params or []

        if action == 'rotate_90':
            return np.rot90(grid, k=1).copy()

        if action == 'rotate_180':
            return np.rot90(grid, k=2).copy()

        if action == 'rotate_270':
            return np.rot90(grid, k=3).copy()

        if action == 'mirror_h':
            return np.flip(grid, axis=1).copy()

        if action == 'mirror_v':
            return np.flip(grid, axis=0).copy()

        if action == 'translate':
            if len(params) < 2:
                return grid.copy()
            dx: int = int(params[0])
            dy: int = int(params[1])
            result: np.ndarray = np.zeros_like(grid)
            rows, cols = grid.shape
            for r in range(rows):
                for c in range(cols):
                    nr: int = r + dy
                    nc: int = c + dx
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr, nc] = grid[r, c]
            return result

        if action == 'recolor':
            if len(params) < 2:
                return grid.copy()
            src_color: int = int(params[0])
            dst_color: int = int(params[1])
            result = grid.copy()
            result[result == src_color] = dst_color
            return result

        if action == 'noop':
            return grid.copy()

        raise ValueError(f"Unknown grid action: '{action}'. Must be one of {self.GRID_PRIMITIVES}")

    def apply_game_action(
        self,
        game: Any,
        action_id: int,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Game mode: perform_action on game engine。

        直接在 game engine 上执行动作，修改 game 状态。
        此方法用于 ReplayEngine 的 game mode replay：
        deepcopy(root_game) 后逐动作 perform_action 到目标节点。

        Args:
            game: ARC-AGI-3 game engine 对象。
            action_id: 动作 ID (1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 6=ACTION6)。
            data: 动作数据字典 (ACTION6 需要 click 坐标)。

        Raises:
            ImportError: 如果 arcengine 包不可用。
            ValueError: 如果 action_id 不在 GAME_ACTIONS 中。
        """
        if action_id not in self.GAME_ACTIONS:
            raise ValueError(
                f"Unknown game action_id: {action_id}. "
                f"Must be one of {list(self.GAME_ACTIONS.keys())}"
            )

        try:
            from arcengine import ActionInput  # type: ignore[import-untyped]
        except ImportError:
            # arcengine 不可用时，尝试通过 game 对象的方法直接执行
            # 大多数 ARC-AGI-3 game 有 perform_action 方法
            if hasattr(game, 'perform_action'):
                game.perform_action(action_id, data or {})
                return
            raise ImportError(
                "arcengine.ActionInput not available and game has no perform_action method"
            )

        ai = ActionInput(id=action_id, data=data or {})
        game.perform_action(ai)


# ============================================================================
# §5. ReplayEngine — IDO 流贯 Replay 引擎 (替代 DeepCopy)
# ============================================================================

class ReplayEngine:
    """IDO 流贯 Replay 引擎 — 支持双模式 (game + grid)。

    IDO 核心优化：不做 DeepCopy，而是从 root_state 开始逐动作 Replay。
    这是「流贯」(flow-through) 的实现：基态不变，Δ(动作序列) 决定终态。

    设计决策：
      - Game mode: deepcopy(root_game) + perform_action sequence
        注意：只在最终验证时才 deepcopy，中间节点不拷贝 game
      - Grid mode: copy(root_grid) + transformation sequence
        Grid 变换是纯函数，中间节点可缓存
      - 有条件缓存：grid mode 物化后的 Grid 缓存在 node._grid_cache
        缓存上限 MAX_REPLAY_CACHE (128)，防止内存溢出
      - 回溯路径：_backtrack_path() 从 target_id 回溯到 root，
        收集动作链然后逆序执行

    v3.18.0 NEW:
      - from_game() 工厂方法: 从game engine和node_map自动构建ReplayEngine
      - 全局共享支持: _shared_engines dict支持跨任务共享ReplayEngine实例

    Attributes:
        root_state: 根节点状态 (game mode: game engine, grid mode: np.ndarray)。
        node_map: 节点编号 → Node 对象的映射。
        mode: Replay 模式 ('game' 或 'grid')。
        _cache_count: 当前已使用的缓存槽位数。
    """

    # v3.18.0: 全局共享ReplayEngine实例池 (支持跨任务共享)
    _shared_engines: Dict[str, 'ReplayEngine'] = {}

    def __init__(
        self,
        root_state: Any,
        node_map: Dict[int, Node],
        mode: str = 'game',
    ) -> None:
        """初始化 ReplayEngine。

        Args:
            root_state: 根节点状态。
                game mode: ARC-AGI-3 game engine 对象 (需要 deepcopy)。
                grid mode: numpy ndarray (只需 copy)。
            node_map: 节点编号 → Node 对象的映射字典。
            mode: Replay 模式，'game' 或 'grid'。
        """
        self.root_state: Any = root_state
        self.node_map: Dict[int, Node] = node_map
        self.mode: str = mode
        self._cache_count: int = 0
        # v4.1: Karma Cache — structural_hash lookup/store for ReplayEngine
        self._karma_cache: Dict[str, Any] = {}

    @classmethod
    def from_game(
        cls,
        game: Any,
        mode: str = 'game',
        game_id: str = "",
        shared: bool = False,
    ) -> 'ReplayEngine':
        """从game engine自动构建ReplayEngine的工厂方法。

        创建根节点和node_map，构建ReplayEngine实例。
        如果shared=True，将实例注册到全局共享池，
        同一game_id共享同一个ReplayEngine。

        Args:
            game: ARC-AGI-3 game engine 对象。
            mode: Replay 模式 ('game' 或 'grid')。
            game_id: 游戏ID，用于全局共享池的键 (可选)。
            shared: 是否注册到全局共享池 (默认False)。

        Returns:
            ReplayEngine实例。
        """
        # 创建根节点
        node_map: Dict[int, Node] = {}
        root_node: Node = Node(
            id=0,
            parent_id=-1,
            action="root",
            depth=0,
            meta={'stage': 'root'},
        )
        node_map[0] = root_node

        # 提取根状态
        root_state: Any = game
        if mode == 'grid':
            # Grid mode: 从game提取Grid数据
            grid: Optional[np.ndarray] = _extract_game_grid(game)
            if grid is not None:
                root_state = grid.copy()
            else:
                # 无法提取Grid → 使用空Grid
                root_state = np.zeros((64, 64), dtype=int)

        # 构建ReplayEngine
        engine: ReplayEngine = cls(
            root_state=root_state,
            node_map=node_map,
            mode=mode,
        )

        # 全局共享
        if shared and game_id:
            cls._shared_engines[game_id] = engine

        return engine

    @classmethod
    def get_shared(cls, game_id: str) -> Optional['ReplayEngine']:
        """获取全局共享的ReplayEngine实例。

        Args:
            game_id: 游戏ID。

        Returns:
            共享的ReplayEngine实例，或None(不存在)。
        """
        return cls._shared_engines.get(game_id)

    @classmethod
    def clear_shared(cls) -> None:
        """清除所有全局共享的ReplayEngine实例。"""
        for engine in cls._shared_engines.values():
            engine.clear_cache()
        cls._shared_engines.clear()

    def replay(self, target_id: int) -> Any:
        """从根节点 Replay 到 target_id，返回最终状态。

        IDO 流贯 Replay 核心方法：
          1. 检查缓存 (grid mode: node._grid_cache)
          2. 回溯构建动作链 (_backtrack_path)
          3. Game mode: deepcopy(root) → perform_action sequence
          4. Grid mode: copy(root_grid) → apply transformation sequence
          5. 有条件缓存结果

        Args:
            target_id: 目标节点的编号。

        Returns:
            最终状态 (game mode: game engine 对象, grid mode: np.ndarray)。

        Raises:
            KeyError: 如果 target_id 不在 node_map 中。
        """
        if target_id not in self.node_map:
            raise KeyError(f"Node id {target_id} not found in node_map")

        node: Node = self.node_map[target_id]

        # 检查缓存 — grid mode 的物化缓存
        if self.mode == 'grid' and node._grid_cache is not None:
            return node._grid_cache

        # 回溯构建动作链
        path: list = self._backtrack_path(target_id)

        if self.mode == 'game':
            # Game mode: deepcopy root_game, then perform_action sequence
            # 注意：deepcopy 只在 Replay 时发生，BFS 扩展阶段不拷贝
            state = copy.deepcopy(self.root_state)
            action_space = ActionSpace('game')
            for action_item in path:
                # path 元素格式取决于 _backtrack_path 的解析
                # game mode: action_item 是 action_id (int)
                if isinstance(action_item, int):
                    action_space.apply_game_action(state, action_item)
                elif isinstance(action_item, tuple) and len(action_item) == 2:
                    # (action_id, data) 格式
                    action_space.apply_game_action(state, action_item[0], action_item[1])
                elif isinstance(action_item, str):
                    # 字符串格式：尝试解析为 int
                    try:
                        action_id = int(action_item)
                        action_space.apply_game_action(state, action_id)
                    except (ValueError, TypeError):
                        pass  # 忽略无法解析的动作
            # Game mode 不缓存结果 (game engine 对象不适合缓存)
            return state

        # Grid mode: copy root_grid, then apply transformations
        state: np.ndarray = self.root_state.copy()
        action_space = ActionSpace('grid')
        for action_item in path:
            # path 元素在 grid mode 是动作描述
            # 格式: (action_name, params) 或 (action_name,)
            if isinstance(action_item, tuple):
                action_name: str = action_item[0]
                action_params: list = list(action_item[1:]) if len(action_item) > 1 else []
                state = action_space.apply(state, action_name, action_params)
            elif isinstance(action_item, str):
                # 单字符串动作名 (无参数)
                state = action_space.apply(state, action_item)

        # 有条件缓存 — 上限 MAX_REPLAY_CACHE
        if self._cache_count < MAX_REPLAY_CACHE and node._grid_cache is None:
            node._grid_cache = state.copy()  # 缓存副本，防止后续修改
            self._cache_count += 1

        return state

    def _backtrack_path(self, target_id: int) -> list:
        """回溯获取从根到目标的动作链。

        从 target_id 开始，沿 parent_id 回溯到根节点 (-1)，
        收集每个节点的 action，然后逆序排列得到执行序列。

        IDO 设计：action 字段在 game mode 存储 action_id 或
        "action_id:data" 格式；在 grid mode 存储 "action_name:param1:param2"
        格式。此方法解析这些格式为可执行的动作描述。

        Args:
            target_id: 目标节点编号。

        Returns:
            动作链列表 (从根到目标的顺序)。
            game mode: [action_id, (action_id, data), ...]
            grid mode: [(action_name, params), (action_name,), ...]
        """
        raw_actions: list = []
        cur_id: int = target_id

        while cur_id != -1:
            node: Node = self.node_map[cur_id]
            raw_actions.append(node.action)
            cur_id = node.parent_id

        # 逆序：从根到目标
        raw_actions.reverse()

        # 解析动作格式
        parsed_path: list = []
        for action_str in raw_actions:
            if action_str == "root":
                continue  # 根节点无动作

            # 解析 "action_name:param1:param2:..." 格式
            parts: list = action_str.split(':')
            if len(parts) == 1:
                # 无参数动作
                # game mode: 尝试解析为整数
                try:
                    parsed_path.append(int(parts[0]))
                except ValueError:
                    parsed_path.append(parts[0])
            else:
                # 有参数动作
                name: str = parts[0]
                params: list = parts[1:]
                # 尝试将参数转为数值
                parsed_params: list = []
                for p in params:
                    try:
                        parsed_params.append(int(p))
                    except ValueError:
                        try:
                            parsed_params.append(float(p))
                        except ValueError:
                            parsed_params.append(p)
                parsed_path.append((name, *parsed_params))

        return parsed_path

    def clear_cache(self) -> None:
        """清除所有 Replay 缓存，释放内存。

        在 BFS 完成后或内存压力时调用。清除所有 node 的 _grid_cache
        并重置缓存计数器。v4.1: 同时清除 Karma 缓存。
        """
        for node in self.node_map.values():
            node._grid_cache = None
            node._hash_cache = None
        self._cache_count = 0
        self._karma_cache.clear()

    # =========================================================================
    # v4.1: Karma Cache — structural_hash lookup/store
    # =========================================================================

    def lookup(self, sig: str) -> Any | None:
        """Look up cached verification result by structural hash signature.

        Karma Cache (v4.1) enables ReplayEngine to cache program verification
        results keyed by program signature, avoiding redundant verification.
        Uses structural_hash from LayoutHasher as the lookup key.

        Typical usage pattern:
          sig = LayoutHasher.hash_with_shape(materialized_grid)
          cached = engine.lookup(sig)
          if cached is not None:
              return cached  # Skip redundant verification
          result = verifier.verify(grid, examples)
          engine.store(sig, result)  # Cache for future lookups

        Args:
            sig: Structural hash signature string. Typically generated
                by LayoutHasher.hash_with_shape() on the materialized grid.

        Returns:
            Cached verification result if found, None otherwise.
            The result type depends on what was stored — typically a
            dict from GaussExVerifier.verify() or similar.
        """
        return self._karma_cache.get(sig)

    def store(self, sig: str, result: Any) -> None:
        """Store verification result in Karma cache.

        Caches the verification result keyed by structural hash signature.
        When the cache exceeds MAX_REPLAY_CACHE (128 entries), the oldest
        entries are evicted (Python 3.7+ dict maintains insertion order).

        Args:
            sig: Structural hash signature string.
            result: Verification result to cache. Can be any type —
                typically a dict from GaussExVerifier.verify() containing
                'passed', 'max_error', 'threshold', 'details', 'margin'.
        """
        self._karma_cache[sig] = result
        # Prune if exceeding MAX_REPLAY_CACHE
        if len(self._karma_cache) > MAX_REPLAY_CACHE:
            # Remove oldest entries (first inserted; dict is ordered in Python 3.7+)
            keys_to_remove: List[str] = list(self._karma_cache.keys())[
                :len(self._karma_cache) - MAX_REPLAY_CACHE
            ]
            for key in keys_to_remove:
                del self._karma_cache[key]


# ============================================================================
# §6. LayoutHasher — 布局哈希 (物化后 Grid 的 MD5)
# ============================================================================

class LayoutHasher:
    """布局哈希：对物化后的 Grid 做 MD5 哈希。

    IDO 多阶段 BFS 阶段2 (参数探索) 使用布局哈希去重：
    同一布局的不同参数组合只保留一个，避免重复验证。

    设计决策：
      - hash() 忽略 shape，只哈希 Grid 数据 (用于同尺寸去重)
      - hash_with_shape() 包含 shape 信息 (用于跨尺寸去重)
      - MD5 速度快，碰撞概率极低 (2^128 空间)

    注意：布局哈希只对物化后的 Grid 计算，IDO 阶段1 (结构探索)
    不需要物化，因此不使用布局哈希。
    """

    @staticmethod
    def hash(grid: np.ndarray) -> str:
        """计算 Grid 的 MD5 哈希 (不含 shape)。

        适用于同尺寸 Grid 之间的去重比较。

        Args:
            grid: numpy array (2D, dtype int)。

        Returns:
            MD5 哈希字符串 (32 hex chars)。
        """
        return hashlib.md5(grid.tobytes()).hexdigest()

    @staticmethod
    def hash_with_shape(grid: np.ndarray) -> str:
        """计算 Grid 的 MD5 哈希 (含 shape 信息)。

        适用于跨尺寸 Grid 之间的去重比较。
        格式: MD5(shape_bytes | grid_bytes)

        Args:
            grid: numpy array (2D, dtype int)。

        Returns:
            MD5 哈希字符串 (32 hex chars)，包含 shape 前缀。
        """
        shape_bytes: bytes = str(grid.shape).encode('utf-8')
        grid_bytes: bytes = grid.tobytes()
        return hashlib.md5(shape_bytes + b'|' + grid_bytes).hexdigest()


# ============================================================================
# §7. GaussExVerifier — IDO GaussEx 验证器 (5/6 饱和阈值)
# ============================================================================

class GaussExVerifier:
    """IDO GaussEx 验证器 — 卞氏 5/6 饱和阈值。

    GaussEx (Gaussian Example Verification) 是 ARC-3 求解中的关键验证指标。
    基于 Theorem 1 (5/6 saturation threshold):
      - GEX ≥ 1/6 ≈ 0.167 → 通过 (结构足够接近目标)
      - GEX < 1/6 → 失败 (偏离太大)

    设计决策：
      - verify() 计算 Grid 与所有示例的错误比率
      - verify_node() 自动 Replay 节点后再验证 (IDO 惰性物化)
      - threshold 默认使用 GEX_PASS_THRESHOLD (1/6)

    Attributes:
        threshold: 通过阈值，默认 GEX_PASS_THRESHOLD ≈ 0.167。
    """

    def __init__(self, threshold: float = GEX_PASS_THRESHOLD) -> None:
        """初始化 GaussExVerifier。

        Args:
            threshold: 通过阈值。默认使用 GEX_PASS_THRESHOLD (1/6 ≈ 0.167)。
        """
        self.threshold: float = threshold

    def verify(
        self,
        grid: np.ndarray,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """验证 Grid 是否通过 GaussEx。

        对每个示例 (input_grid, output_grid)，计算 grid 与 output 的
        结构相似度 (structural similarity ratio)。GEX 定义为最大相似度。

        卞氏定理：当 GEX ≥ 1/6 (即 5/6 的结构一致) 时，验证通过。

        Args:
            grid: 待验证的 Grid (numpy 2D array)。
            examples: 示例列表 [(input, output), ...]。

        Returns:
            验证结果字典，包含：
              - passed: bool, 是否通过阈值
              - max_error: float, 最大错误比率 (0=完全匹配, 1=完全不同)
              - threshold: float, 使用的阈值
              - details: list, 每个示例的详细错误比率
              - margin: float, 通过裕度 (threshold - max_error)
        """
        if len(examples) == 0:
            return {
                'passed': False,
                'max_error': 1.0,
                'threshold': self.threshold,
                'details': [],
                'margin': self.threshold - 1.0,
            }

        details: list = []
        for input_grid, output_grid in examples:
            # 计算 grid 与 output 的错误比率
            # 错误比率 = 不匹配像素占比 (需要相同 shape)
            if grid.shape != output_grid.shape:
                # shape 不匹配 → 完全错误
                error_ratio: float = 1.0
            else:
                total_pixels: int = output_grid.size
                if total_pixels == 0:
                    error_ratio = 0.0  # 空 grid 视为完美匹配
                else:
                    mismatch_count: int = int(np.sum(grid != output_grid))
                    error_ratio = float(mismatch_count) / float(total_pixels)
            details.append({
                'error_ratio': error_ratio,
                'grid_shape': grid.shape,
                'output_shape': output_grid.shape,
                'mismatch_pixels': int(np.sum(grid != output_grid))
                if grid.shape == output_grid.shape else -1,
            })

        # GEX = 1 - max_error (结构一致性)
        max_error: float = max(d['error_ratio'] for d in details)
        # 通过条件：max_error ≤ threshold (即 GEX ≥ 1 - threshold)
        # 卞氏阈值：threshold = 1/6，即 max_error ≤ 1/6 ≈ 0.167
        passed: bool = max_error <= self.threshold
        margin: float = self.threshold - max_error

        return {
            'passed': passed,
            'max_error': max_error,
            'threshold': self.threshold,
            'details': details,
            'margin': margin,
        }

    def verify_node(
        self,
        node_id: int,
        replay_engine: ReplayEngine,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """验证节点（自动 Replay 后再验证）。

        IDO 惰性物化：先 Replay 获取节点的实际 Grid，
        再用 GaussEx 验证。这是 IDO 的「零拷贝 + 惰性物化」策略。

        Args:
            node_id: 待验证的节点编号。
            replay_engine: ReplayEngine 实例 (用于物化节点状态)。
            examples: 示例列表 [(input, output), ...]。

        Returns:
            验证结果字典 (同 verify() 返回格式)。
        """
        # Replay 物化节点状态
        state: Any = replay_engine.replay(node_id)

        # Grid mode: state 是 np.ndarray，直接验证
        if replay_engine.mode == 'grid':
            if isinstance(state, np.ndarray):
                return self.verify(state, examples)
            # 类型不匹配
            return {
                'passed': False,
                'max_error': 1.0,
                'threshold': self.threshold,
                'details': [],
                'margin': self.threshold - 1.0,
            }

        # Game mode: state 是 game engine 对象
        # 从 game 提取 Grid 后验证
        if replay_engine.mode == 'game':
            grid: Optional[np.ndarray] = _extract_game_grid(state)
            if grid is not None:
                return self.verify(grid, examples)
            # 无法提取 Grid → 无法验证
            return {
                'passed': False,
                'max_error': 1.0,
                'threshold': self.threshold,
                'details': [],
                'margin': self.threshold - 1.0,
            }

        return {
            'passed': False,
            'max_error': 1.0,
            'threshold': self.threshold,
            'details': [],
            'margin': self.threshold - 1.0,
        }


def _extract_game_grid(game: Any) -> Optional[np.ndarray]:
    """从 ARC-AGI-3 game engine 对象提取 Grid 数据。

    尝试多种属性路径获取 Grid:
      - game.grid
      - game.current_level.grid
      - game._grid
      - game.current_level._grid

    Args:
        game: ARC-AGI-3 game engine 对象。

    Returns:
        numpy 2D array (Grid 数据)，或 None (无法提取)。
    """
    # 直接属性
    grid_attr = getattr(game, 'grid', None)
    if grid_attr is not None and isinstance(grid_attr, np.ndarray):
        return grid_attr

    # current_level 属性
    cl = getattr(game, 'current_level', None)
    if cl is not None:
        cl_grid = getattr(cl, 'grid', None)
        if cl_grid is not None and isinstance(cl_grid, np.ndarray):
            return cl_grid
        cl_grid = getattr(cl, '_grid', None)
        if cl_grid is not None and isinstance(cl_grid, np.ndarray):
            return cl_grid

    # 私有属性
    priv_grid = getattr(game, '_grid', None)
    if priv_grid is not None and isinstance(priv_grid, np.ndarray):
        return priv_grid

    return None


# ============================================================================
# §8. 多阶段 BFS 函数 — IDO 结构探索 + 参数探索 + κ-Priority 精化
# ============================================================================

def structural_bfs(
    root_id: int,
    node_map: Dict[int, Node],
    action_space: ActionSpace,
    max_nodes: int = STRUCT_BFS_BUDGET,
    visited_global: Optional[set] = None,
) -> List[int]:
    """IDO 结构探索阶段：零拷贝，只探索方向组合。

    阶段1 (Structural BFS) 只扩展动作方向，不物化 Grid。
    每个节点只记录 (parent_id, action)，不存储完整状态。
    这是 IDO 的核心优化：用 O(1) 的 Node 代替 O(n²) 的 Grid。

    设计决策：
      - visited_global 跨任务共享的去重集合 (避免重复探索相同方向)
      - max_nodes 限制扩展节点数 (STRUCT_BFS_BUDGET = 40)
      - 返回候选节点 ID 列表，供参数探索阶段使用

    Args:
        root_id: 根节点编号。
        node_map: 节点编号 → Node 的映射 (会被扩展)。
        action_space: ActionSpace 实例 (决定可用动作)。
        max_nodes: 最大扩展节点数。
        visited_global: 全局去重集合 (可选，跨任务共享)。

    Returns:
        候选节点 ID 列表 (供 parametric_bfs 使用)。
    """
    if visited_global is None:
        visited_global = set()

    candidates: List[int] = []
    queue: deque = deque([root_id])
    next_id: int = max(node_map.keys(), default=0) + 1
    expanded: int = 0

    while queue and expanded < max_nodes:
        cur_id: int = queue.popleft()
        cur_node: Node = node_map[cur_id]

        # 获取可用动作
        primitives: list = action_space.get_primitives()

        for action in primitives:
            # 构造动作字符串 (格式: "action_name" 或 "action_id")
            if action_space._mode == 'game':
                action_str: str = str(action)
            else:
                action_str: str = action

            # 构造新节点签名
            signature: str = f"{cur_id}->{action_str}"

            # 去重检查
            if signature in visited_global:
                continue
            visited_global.add(signature)

            # 创建新节点 (零拷贝！只记录 Δ)
            new_node: Node = Node(
                id=next_id,
                parent_id=cur_id,
                action=action_str,
                depth=cur_node.depth + 1,
                meta={'stage': 'structural'},
            )
            node_map[next_id] = new_node
            candidates.append(next_id)
            queue.append(next_id)
            next_id += 1
            expanded += 1

            if expanded >= max_nodes:
                break

    return candidates


def parametric_bfs(
    candidates: List[int],
    node_map: Dict[int, Node],
    replay_engine: ReplayEngine,
    examples: List[Tuple[np.ndarray, np.ndarray]],
    max_total: int = PARAM_BFS_BUDGET,
    layout_cache: Optional[Dict[str, int]] = None,
) -> List[int]:
    """IDO 参数探索阶段：布局哈希去重 + GaussEx 校验。

    阶段2 (Parametric BFS) 物化候选节点 (Replay)，然后：
      1. 布局哈希去重 (LayoutHasher): 同布局只保留一个
      2. GaussEx 校验 (GaussExVerifier): 验证是否接近目标

    设计决策：
      - layout_cache 布局哈希 → node_id 映射 (去重)
      - max_total 限制物化总数 (PARAM_BFS_BUDGET = 100)
      - 只有 grid mode 才物化和哈希去重
      - game mode 直接在 game engine 上验证

    Args:
        candidates: 结构探索阶段的候选节点 ID 列表。
        node_map: 节点编号 → Node 的映射。
        replay_engine: ReplayEngine 实例 (物化节点)。
        examples: 示例列表 [(input, output), ...]。
        max_total: 最大物化节点数。
        layout_cache: 布局哈希缓存 (可选，跨任务共享)。

    Returns:
        通过 GaussEx 校验的节点 ID 列表 (供 kappa_priority_refine 使用)。
    """
    if layout_cache is None:
        layout_cache = {}

    verifier: GaussExVerifier = GaussExVerifier()
    passed_nodes: List[int] = []
    materialized: int = 0

    for node_id in candidates:
        if materialized >= max_total:
            break

        node: Node = node_map[node_id]

        # Grid mode: 物化 + 布局哈希去重
        if replay_engine.mode == 'grid':
            grid: np.ndarray = replay_engine.replay(node_id)

            # 布局哈希去重
            layout_hash: str = LayoutHasher.hash_with_shape(grid)
            if layout_hash in layout_cache:
                # 同布局已存在，跳过
                continue
            layout_cache[layout_hash] = node_id

            # GaussEx 校验
            result: Dict[str, Any] = verifier.verify(grid, examples)
            node.meta['gex_result'] = result
            node.meta['layout_hash'] = layout_hash

            if result['passed']:
                passed_nodes.append(node_id)

            materialized += 1

        # Game mode: 直接 Replay + GaussEx 校验
        elif replay_engine.mode == 'game':
            result = verifier.verify_node(node_id, replay_engine, examples)
            node.meta['gex_result'] = result

            if result['passed']:
                passed_nodes.append(node_id)

            materialized += 1

    return passed_nodes


def kappa_priority_refine(
    candidates: List[int],
    node_map: Dict[int, Node],
    replay_engine: ReplayEngine,
    examples: List[Tuple[np.ndarray, np.ndarray]],
) -> List[int]:
    """IDO κ-Priority 精化：Liu-Score 排序选优。

    阶段3 (κ-Priority Refine) 对通过 GaussEx 的候选节点排序：
      - Liu-Score = 1 / (S_rel + ε)
        其中 S_rel = 0.1×num_primitives - 0.5×IC + 2.0×GEX
      - ε = 0.01 防止除零
      - 排序后取前 N 个最优节点

    设计决策：
      - Liu mechanism 优先公式 (v3.14.0): priority = 1/(S_rel + ε)
      - ε = 0.01 防止除零
      - DEAD_ZERO_RATIO (~8.3%) 作为背景剪枝阈值
      - 返回排序后的节点 ID 列表

    Args:
        candidates: 通过 GaussEx 校验的节点 ID 列表。
        node_map: 节点编号 → Node 的映射。
        replay_engine: ReplayEngine 实例。
        examples: 示例列表 [(input, output), ...]。

    Returns:
        按 Liu-Score 排序的节点 ID 列表 (最高优先级在前)。
    """
    if len(candidates) == 0:
        return []

    epsilon: float = 0.01  # 防止除零

    # 计算每个候选的 Liu-Score
    scored: List[Tuple[float, int]] = []
    for node_id in candidates:
        node: Node = node_map[node_id]
        gex_result: Dict[str, Any] = node.meta.get('gex_result', {})

        # 从 GEX 结果中提取指标
        max_error: float = gex_result.get('max_error', 1.0)
        gex_score: float = 1.0 - max_error  # GEX = 1 - error (一致性)

        # IC (Information Cardinality) — 从 meta 或默认值
        ic: float = node.meta.get('ic', 0.5)

        # Liu-Score: priority = 1 / (S_rel + ε)
        # S_rel = 0.1×num_primitives - 0.5×IC + 2.0×GEX
        num_primitives: int = node.depth  # 深度 ≈ 使用的原语数
        s_rel: float = 0.1 * num_primitives - 0.5 * ic + 2.0 * gex_score
        liu_score: float = 1.0 / (s_rel + epsilon)

        # 背景剪枝：低于 DEAD_ZERO_RATIO 的丢弃
        if liu_score < DEAD_ZERO_RATIO:
            continue

        scored.append((liu_score, node_id))

    # 按 Liu-Score 降序排序 (高优先级在前)
    scored.sort(key=lambda x: x[0], reverse=True)

    return [node_id for _, node_id in scored]


# ============================================================================
# §9. 金币 proximity position 辅助函数 — LS20 金币收集路径规划
# ============================================================================

def compute_coin_proximity_position(
    start_x: int,
    start_y: int,
    step_size: int,
    coin_x: int,
    coin_y: int,
) -> Tuple[int, int]:
    """计算金币锚点的 proximity position — 玩家包围盒重叠金币锚点的网格位置。

    LS20 (金色猎人) 游戏中，玩家需要移动到金币锚点附近才能收集金币。
    此函数计算玩家应站在哪个网格位置，使得玩家包围盒包含金币锚点。

    玩家包围盒 = [px, px+step_size) × [py, py+step_size)
    金币锚点 (coin_x, coin_y) 必须落入此包围盒。

    公式:
        px = start_x + step_size * ((coin_x - start_x) // step_size)
        py = start_y + step_size * ((coin_y - start_y) // step_size)

    验证条件:
        coin_x ∈ [px, px+step_size) AND coin_y ∈ [py, py+step_size)

    Args:
        start_x: 网格起始 X 坐标 (通常为 0 或地图偏移)。
        start_y: 网格起始 Y 坐标。
        step_size: 网格步长 (玩家每次移动的像素数)。
        coin_x: 金币锚点 X 坐标。
        coin_y: 金币锚点 Y 坐标。

    Returns:
        (proximity_x, proximity_y) — 玩家可以站在此位置收集金币。

    Raises:
        AssertionError: 如果验证条件不满足 (金币不在包围盒内)。
    """
    px: int = start_x + step_size * ((coin_x - start_x) // step_size)
    py: int = start_y + step_size * ((coin_y - start_y) // step_size)

    # 验证金币锚点在玩家包围盒内
    assert coin_x >= px and coin_x < px + step_size, (
        f"Coin X validation failed: coin_x={coin_x}, "
        f"px={px}, step_size={step_size}. "
        f"Expected coin_x ∈ [{px}, {px + step_size})"
    )
    assert coin_y >= py and coin_y < py + step_size, (
        f"Coin Y validation failed: coin_y={coin_y}, "
        f"py={py}, step_size={step_size}. "
        f"Expected coin_y ∈ [{py}, {py + step_size})"
    )

    return (px, py)
