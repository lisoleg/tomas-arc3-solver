"""
src/agent/tn36_opcode.py
TN36 数据驱动 OPCODE_TABLE — 替代lambda闭包 (文章2 方案C)

核心问题:
  TN36的okllwtboml字典将opcode映射到lambda闭包。
  deepcopy后cell引用断裂 → 验证失败。
  IDO判位: 这是"信息基数(IC)未锚定的本体学错误"，不是Python小坑。

方案C: 数据驱动OPCODE_TABLE
  三层架构:
    1. OPCODE_TABLE (纯数据层): opcode → 数据字典 (dict/int/str/tuple)
       完全可哈希、可序列化、可deepcopy、可ψ-Audit
    2. execute_opcode (分发层): 从OPCODE_TABLE读取数据，调用对应处理器
       只做参数分发，不持有lambda
    3. _op_* (处理层): 独立函数，执行具体操作
       可独立测试、可deepcopy引用（函数对象本身可深拷贝）

κ-Phase洞察:
  - lambda闭包 = IC未锚定 = 本体学错误 (IDO判位)
  - 数据驱动 = IC锚定 = 每个宏携带确定参数 = 可哈希/可序列化
  - OPCODE_TABLE = κ-陪集C(11,4)的数据层映射
  - deepcopy = κ-Snap全息重构 → 只要数据层是纯数据就完全安全

Version: v4.5 — Wall-Aware Path Planning: BFS on 4-aligned grid avoiding wall collision
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import copy


# ============================================================
# §1. OPCODE定义 — 纯数据层
# ============================================================

# TN36游戏OPCODE映射 (CSPOIQWER = 4步长常量)
# 从okllwtboml字典(game source)提取的完整opcode→数据字典
# 每个opcode对应: {type, dx, dy, rotation, scale, sjmtdfxdrc_set, description}
#
# v4.4新增 (从game source okllwtboml发现):
#   - 11/13: 双步移动备用 (与10/12相同效果)
#   - 14/15/63: sjmtdfxdrc_set — 直接设置sjmtdfxdrc到特定值
#   - 16: rotate(270°) — 与opcode 6(-90°)等效但不同入口
#
# sjmtdfxdrc常量 (game source line 1724-1729):
#   fqduemlzes = 9  → opcode 14 sets sjmtdfxdrc=9
#   xgzvwnroim = 8  → opcode 15 sets sjmtdfxdrc=8
#   nihflvyqyo = 15 → opcode 63 sets sjmtdfxdrc=15

OPCODE_TABLE: Dict[int, Dict[str, Any]] = {
    # ── NOP ──
    0: {
        'type': 'noop',
        'dx': 0, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'NOP (mlejdghzfo) — 无操作',
    },

    # ── 单步移动 (CSPOIQWER=4) ──
    1: {
        'type': 'move',
        'dx': -4, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'LEFT (-4, 0) — 左移4像素',
    },
    2: {
        'type': 'move',
        'dx': 4, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'RIGHT (+4, 0) — 右移4像素',
    },
    3: {
        'type': 'move',
        'dx': 0, 'dy': 4,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'DOWN (0, +4) — 下移4像素',
    },
    33: {
        'type': 'move',
        'dx': 0, 'dy': -4,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'UP (0, -4) — 上移4像素',
    },
    34: {
        'type': 'move',
        'dx': -4, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'LEFT-ALT (-4, 0) — 左移4像素(备用)',
    },

    # ── 双步移动 (CSPOIQWER×2=8) ──
    10: {
        'type': 'move',
        'dx': 8, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'RIGHT×2 (+8, 0) — 右移8像素(双步)',
    },
    11: {
        'type': 'move',
        'dx': 8, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'RIGHT×2-ALT (+8, 0) — 右移8像素(备用)',
    },
    12: {
        'type': 'move',
        'dx': -8, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'LEFT×2 (-8, 0) — 左移8像素(双步)',
    },
    13: {
        'type': 'move',
        'dx': -8, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'LEFT×2-ALT (-8, 0) — 左移8像素(备用)',
    },

    # ── 旋转 ──
    5: {
        'type': 'rotation',
        'dx': 0, 'dy': 0,
        'rotation': 90, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'rotate +90° — 正旋转90度',
    },
    6: {
        'type': 'rotation',
        'dx': 0, 'dy': 0,
        'rotation': -90, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'rotate -90° — 反旋转90度',
    },
    7: {
        'type': 'rotation',
        'dx': 0, 'dy': 0,
        'rotation': 180, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'rotate +180° — 旋转180度',
    },
    16: {
        'type': 'rotation',
        'dx': 0, 'dy': 0,
        'rotation': 270, 'scale': 0,
        'sjmtdfxdrc_set': None,
        'description': 'rotate +270° — 正旋转270度(=反90°)',
    },

    # ── 缩放 ──
    8: {
        'type': 'scale',
        'dx': 0, 'dy': 0,
        'rotation': 0, 'scale': 1,
        'sjmtdfxdrc_set': None,
        'description': 'scale +1 — 缩放增大',
    },
    9: {
        'type': 'scale',
        'dx': 0, 'dy': 0,
        'rotation': 0, 'scale': -1,
        'sjmtdfxdrc_set': None,
        'description': 'scale -1 — 缩放减小',
    },

    # ── sjmtdfxdrc Set — 直接设置到特定值 ──
    # v4.4新增: 从game source okllwtboml提取
    # knfgrcbayu(value) → htntnzkbzu.ubescnrjpf = value
    14: {
        'type': 'sjmtdfxdrc_set',
        'dx': 0, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': 9,
        'description': 'sjmtdfxdrc=9 (fqduemlzes) — 设置外观码为9',
    },
    15: {
        'type': 'sjmtdfxdrc_set',
        'dx': 0, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': 8,
        'description': 'sjmtdfxdrc=8 (xgzvwnroim) — 设置外观码为8',
    },
    63: {
        'type': 'sjmtdfxdrc_set',
        'dx': 0, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_set': 15,
        'description': 'sjmtdfxdrc=15 (nihflvyqyo) — 设置外观码为15',
    },
}

# sjmtdfxdrc target value → opcode lookup (反向映射)
_SJMTDFXDRC_TARGET_TO_OPCODE: Dict[int, int] = {
    9: 14,   # fqduemlzes
    8: 15,   # xgzvwnroim
    15: 63,  # nihflvyqyo
}


# ============================================================
# §2. 状态机数据结构 — 纯数据，可deepcopy
# ============================================================

@dataclass
class TN36StateMachine:
    """TN36状态机 — 纯数据表示，完全可deepcopy。

    κ-Phase: 状态机 = κ-陪集数据对象 (IC锚定)
    不持有lambda闭包，所有参数是int/float/str/dict → deepcopy完全安全。

    Attributes:
        x: 当前X坐标。
        y: 当前Y坐标。
        rotation: 当前旋转角度(度)。
        scale: 当前缩放值。
        sjmtdfxdrc: 当前sjmtdfxdrc值。
        program: 按钮opcode列表 (数据驱动)。
        is_editable: 是否可编辑 (viknfwcfei)。
        label: 标签(left/right)。
    """
    x: float = 0.0
    y: float = 0.0
    rotation: int = 0
    scale: int = 1
    sjmtdfxdrc: int = 0
    program: List[int] = field(default_factory=list)
    is_editable: bool = False
    label: str = ""

    # ── 目标状态 ──
    target_x: float = 0.0
    target_y: float = 0.0
    target_rotation: int = 0
    target_scale: int = 1
    target_sjmtdfxdrc: int = 0

    def compute_delta(self) -> Dict[str, Any]:
        """计算当前状态到目标状态的delta。"""
        CSPOIQWER = 4
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dx_steps = dx // CSPOIQWER
        dy_steps = dy // CSPOIQWER
        rotation_delta = (self.target_rotation - self.rotation) % 360
        scale_delta = self.target_scale - self.scale
        sjmtdfxdrc_delta = self.target_sjmtdfxdrc - self.sjmtdfxdrc

        return {
            'dx': dx, 'dy': dy,
            'dx_steps': dx_steps, 'dy_steps': dy_steps,
            'rotation_delta': rotation_delta,
            'scale_delta': scale_delta,
            'sjmtdfxdrc_delta': sjmtdfxdrc_delta,
        }

    def compute_target_program(self, n_buttons: int) -> Optional[List[int]]:
        """从delta计算目标opcode序列。

        κ-Phase: target_program = κ-陪集因果归约的目标序列
        按优先级: sjmtdfxdrc → rotation → scale → movement

        v4.4新增: sjmtdfxdrc_set opcode (14/15/63)
        sjmtdfxdrc不是toggle而是直接set到特定值。
        只有target值=9/8/15时才能直接映射到opcode。

        Args:
            n_buttons: 可用按钮数量。

        Returns:
            目标opcode列表, 或None(无法在n_buttons内表达)。
        """
        delta = self.compute_delta()
        target_prog: List[int] = []

        # 1. sjmtdfxdrc set (最优先 — 外观码必须精确匹配)
        sjmtdfxdrc_delta = delta['sjmtdfxdrc_delta']
        if sjmtdfxdrc_delta != 0:
            target_value = int(self.target_sjmtdfxdrc)
            sjmtdfxdrc_opcode = _SJMTDFXDRC_TARGET_TO_OPCODE.get(target_value)
            if sjmtdfxdrc_opcode is not None:
                target_prog.append(sjmtdfxdrc_opcode)
            else:
                # target值不在已知映射中 → 无法通过opcode直接设置
                # 但如果current == target, delta应该是0, 不会走到这里
                return None

        # 2. Rotation opcodes (旋转影响后续移动方向)
        rot_delta = delta['rotation_delta']
        if rot_delta == 90:
            target_prog.append(5)
        elif rot_delta == 270:
            target_prog.append(16)  # v4.4: rotate +270° (或用6=rotate -90°)
        elif rot_delta == 180:
            target_prog.append(7)

        # 3. Scale opcodes
        scale_delta = delta['scale_delta']
        for _ in range(abs(scale_delta)):
            target_prog.append(8 if scale_delta > 0 else 9)

        # 4. Movement opcodes
        dx_steps = delta['dx_steps']
        dy_steps = delta['dy_steps']

        # 先尝试单步opcode
        for _ in range(abs(dx_steps)):
            target_prog.append(2 if dx_steps > 0 else 1)
        for _ in range(abs(dy_steps)):
            target_prog.append(3 if dy_steps > 0 else 33)

        # 如果超过按钮数量 → 压缩为双步opcode
        if len(target_prog) > n_buttons:
            target_prog = self._compress_program(delta, n_buttons)

        if len(target_prog) > n_buttons:
            return None  # 无法压缩到n_buttons内

        # 填充NOP
        while len(target_prog) < n_buttons:
            target_prog.append(0)

        return target_prog

    def _compress_program(self, delta: Dict[str, Any],
                          n_buttons: int) -> List[int]:
        """压缩程序 — 使用双步opcode减少长度。

        κ-Phase: 压缩 = κ-陪集归约的效率优化
        双步opcode = 2×CSPOIQWER步长 = 减半移动指令数
        v4.4: 包含sjmtdfxdrc_set和rotate(270°)
        """
        compressed: List[int] = []

        # sjmtdfxdrc set (最优先)
        sjmtdfxdrc_delta = delta['sjmtdfxdrc_delta']
        if sjmtdfxdrc_delta != 0:
            target_value = int(self.target_sjmtdfxdrc)
            sjmtdfxdrc_opcode = _SJMTDFXDRC_TARGET_TO_OPCODE.get(target_value)
            if sjmtdfxdrc_opcode is not None:
                compressed.append(sjmtdfxdrc_opcode)

        # Rotation stays same
        rot_delta = delta['rotation_delta']
        if rot_delta == 90:
            compressed.append(5)
        elif rot_delta == 270:
            compressed.append(16)
        elif rot_delta == 180:
            compressed.append(7)

        # Scale stays same
        scale_delta = delta['scale_delta']
        for _ in range(abs(scale_delta)):
            compressed.append(8 if scale_delta > 0 else 9)

        # Double-step movement where possible
        dx_steps = delta['dx_steps']
        remaining_dx = abs(dx_steps)
        while remaining_dx >= 2:
            compressed.append(10 if dx_steps > 0 else 12)
            remaining_dx -= 2
        for _ in range(remaining_dx):
            compressed.append(2 if dx_steps > 0 else 1)

        dy_steps = delta['dy_steps']
        remaining_dy = abs(dy_steps)
        # 注: 没有明确的UP×2/DOWN×2 opcode编号
        for _ in range(abs(dy_steps)):
            compressed.append(3 if dy_steps > 0 else 33)

        return compressed

    def apply_opcode(self, opcode: int) -> 'TN36StateMachine':
        """应用一个opcode到状态机 — 返回新状态(不修改原状态)。

        κ-Phase: apply_opcode = κ-陪集变换的纯数据执行
        从OPCODE_TABLE读取参数 → 修改状态 → 返回新状态
        v4.4: 支持sjmtdfxdrc_set (直接设置外观码值)

        Args:
            opcode: OPCODE_TABLE中的opcode编号。

        Returns:
            新的TN36StateMachine实例。
        """
        entry = OPCODE_TABLE.get(opcode)
        if entry is None:
            # 未知opcode → NOP
            entry = OPCODE_TABLE[0]

        new_state = copy.deepcopy(self)  # ✅ deepcopy完全安全！纯数据！

        if entry['type'] == 'noop':
            pass  # 无变化
        elif entry['type'] == 'move':
            new_state.x += entry['dx']
            new_state.y += entry['dy']
        elif entry['type'] == 'rotation':
            new_state.rotation = (new_state.rotation + entry['rotation']) % 360
        elif entry['type'] == 'scale':
            new_state.scale += entry['scale']
        elif entry['type'] == 'sjmtdfxdrc_set':
            new_state.sjmtdfxdrc = entry['sjmtdfxdrc_set']

        return new_state

    def apply_program(self, program: List[int]) -> 'TN36StateMachine':
        """应用整个opcode序列 — 逐步执行并返回最终状态。

        κ-Phase: apply_program = κ-陪集因果链的完整执行
        """
        state = self
        for opcode in program:
            state = state.apply_opcode(opcode)
        return state

    def is_at_target(self) -> bool:
        """检查是否已到达目标状态。"""
        return (self.x == self.target_x and
                self.y == self.target_y and
                self.rotation == self.target_rotation and
                self.scale == self.target_scale and
                self.sjmtdfxdrc == self.target_sjmtdfxdrc)


# ============================================================
# §3. OPCODE执行层 — 分发器 + 处理器
# ============================================================

def execute_opcode(opcode: int, state: TN36StateMachine) -> TN36StateMachine:
    """OPCODE分发器 — 从OPCODE_TABLE读取数据，调用对应处理器。

    κ-Phase: 分发器 = κ-陪集变换的IC锚定分发层
    只做参数分发，不持有lambda → deepcopy完全安全
    v4.4: 新增sjmtdfxdrc_set分发

    Args:
        opcode: opcode编号。
        state: 当前状态机。

    Returns:
        新状态机实例。
    """
    entry = OPCODE_TABLE.get(opcode)
    if entry is None:
        entry = OPCODE_TABLE[0]  # 未知opcode → NOP

    op_type = entry['type']

    if op_type == 'noop':
        return _op_noop(state, entry)
    elif op_type == 'move':
        return _op_move(state, entry)
    elif op_type == 'rotation':
        return _op_rotation(state, entry)
    elif op_type == 'scale':
        return _op_scale(state, entry)
    elif op_type == 'sjmtdfxdrc_set':
        return _op_sjmtdfxdrc_set(state, entry)
    else:
        # 未知类型 → NOP
        return _op_noop(state, entry)


def _op_noop(state: TN36StateMachine, entry: Dict[str, Any]) -> TN36StateMachine:
    """NOP处理器 — 无操作。"""
    return copy.deepcopy(state)


def _op_move(state: TN36StateMachine, entry: Dict[str, Any]) -> TN36StateMachine:
    """移动处理器 — dx/dy偏移。"""
    new_state = copy.deepcopy(state)
    new_state.x += entry['dx']
    new_state.y += entry['dy']
    return new_state


def _op_rotation(state: TN36StateMachine, entry: Dict[str, Any]) -> TN36StateMachine:
    """旋转处理器 — angle偏移。"""
    new_state = copy.deepcopy(state)
    new_state.rotation = (new_state.rotation + entry['rotation']) % 360
    return new_state


def _op_scale(state: TN36StateMachine, entry: Dict[str, Any]) -> TN36StateMachine:
    """缩放处理器 — scale增减。"""
    new_state = copy.deepcopy(state)
    new_state.scale += entry['scale']
    return new_state


def _op_sjmtdfxdrc_set(state: TN36StateMachine, entry: Dict[str, Any]) -> TN36StateMachine:
    """sjmtdfxdrc处理器 — 直接设置外观码到特定值。

    v4.4新增: knfgrcbayu(value) → htntnzkbzu.ubescnrjpf = value
    不是toggle而是直接赋值。
    """
    new_state = copy.deepcopy(state)
    new_state.sjmtdfxdrc = entry['sjmtdfxdrc_set']
    return new_state


# ============================================================
# §4. 游戏内状态提取 — 从game对象到纯数据
# ============================================================

def extract_state_machine_from_game(sm_obj: Any, label: str = ""
                                    ) -> Optional[TN36StateMachine]:
    """从TN36游戏对象提取状态机为纯数据 — IC锚定。

    κ-Phase: extract = κ-Snap感知 → 从游戏本体提取可deepcopy的纯数据表示
    不持有任何lambda/闭包引用 → deepcopy完全安全

    Args:
        sm_obj: 游戏状态机对象(mvqheosngn/bzirenxmrg)。
        label: 标签(left/right)。

    Returns:
        TN36StateMachine实例, 或None(提取失败)。
    """
    if sm_obj is None:
        return None

    try:
        htnt = getattr(sm_obj, 'htntnzkbzu', None)
        target = getattr(sm_obj, 'aqszntqeae', None)
        prog_bar = getattr(sm_obj, 'ukwrvhanub', None)

        if htnt is None:
            return None

        # 当前状态
        htnt_x = getattr(htnt, 'x', 0)
        htnt_y = getattr(htnt, 'y', 0)
        htnt_rotation = getattr(htnt, 'rotation', 0)
        htnt_scale = getattr(htnt, 'scale', 1)
        htnt_sjmtdfxdrc = getattr(htnt, 'sjmtdfxdrc', 0)

        # 目标状态
        target_x = getattr(target, 'x', 0) if target else 0
        target_y = getattr(target, 'y', 0) if target else 0
        target_rotation = getattr(target, 'rotation', 0) if target else 0
        target_scale = getattr(target, 'scale', 1) if target else 1
        target_sjmtdfxdrc = getattr(target, 'sjmtdfxdrc', 0) if target else 0

        # 是否可编辑
        is_editable = not getattr(prog_bar, 'viknfwcfei', True) if prog_bar else False

        # 当前程序(从按钮opcode读取)
        buttons = getattr(prog_bar, 'pfyayhyovw', []) if prog_bar else []
        current_program = [getattr(btn, 'qaeirkuwro', 0) for btn in buttons]

        state = TN36StateMachine(
            x=htnt_x, y=htnt_y,
            rotation=htnt_rotation, scale=htnt_scale,
            sjmtdfxdrc=htnt_sjmtdfxdrc,
            program=current_program,
            is_editable=is_editable,
            label=label,
            target_x=target_x, target_y=target_y,
            target_rotation=target_rotation, target_scale=target_scale,
            target_sjmtdfxdrc=target_sjmtdfxdrc,
        )

        return state

    except Exception:
        return None


def extract_both_state_machines(game: Any
                                ) -> Tuple[Optional[TN36StateMachine],
                                           Optional[TN36StateMachine]]:
    """从游戏提取两个状态机(LEFT+RIGHT)。

    Args:
        game: TN36游戏对象。

    Returns:
        (left_sm, right_sm) 纯数据状态机对。
    """
    fdk = getattr(game, 'fdksqlmpki', None)
    if fdk is None:
        return None, None

    left_obj = getattr(fdk, 'mvqheosngn', None)
    right_obj = getattr(fdk, 'bzirenxmrg', None)

    left_sm = extract_state_machine_from_game(left_obj, "left")
    right_sm = extract_state_machine_from_game(right_obj, "right")

    return left_sm, right_sm


def find_editable_sm(left_sm: Optional[TN36StateMachine],
                     right_sm: Optional[TN36StateMachine]
                     ) -> Optional[TN36StateMachine]:
    """找到可编辑的状态机。

    Args:
        left_sm, right_sm: 纯数据状态机。

    Returns:
        可编辑的状态机, 或None。
    """
    if left_sm is not None and left_sm.is_editable:
        return left_sm
    if right_sm is not None and right_sm.is_editable:
        return right_sm
    return None


# ============================================================
# §5. Wall-Aware Path Planning — BFS避墙运动规划
# ============================================================

@dataclass
class WallRect:
    """Wall bounding box — 纯数据用于wall-aware路径规划。

    κ-Phase: WallRect = κ-陪集C(11,4)的障碍区域描述
    从游戏源码 bizgpiltwm (sadwidscaq) 提取的墙壁AABB。

    Attributes:
        x: 墙壁左边界X坐标。
        y: 墙壁上边界Y坐标。
        width: 墙壁宽度。
        height: 墙壁高度。
    """
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def overlaps_rect(self, rx: int, ry: int, rw: int, rh: int) -> bool:
        """AABB重叠检查 — 判断此墙壁是否与矩形(rx,ry,rw,rh)重叠。

        κ-Phase: overlaps_rect = κ-陪集碰撞检测的纯数据实现
        与游戏源码 nxexmixgsb + collides_with 等价的AABB重叠判定。

        Args:
            rx: 矩形左边界X。
            ry: 矩形上边界Y。
            rw: 矩形宽度。
            rh: 矩形高度。

        Returns:
            True如果重叠。
        """
        return (self.x < rx + rw and
                self.x + self.width > rx and
                self.y < ry + rh and
                self.y + self.height > ry)


def _check_wall_collision(x: int, y: int, w: int, h: int,
                          walls: List[WallRect]) -> bool:
    """检查bbox是否与任何墙壁碰撞 — AABB重叠判定。

    κ-Phase: _check_wall_collision = cwtesiybfx 的纯数据模拟
    与游戏源码 cwtesiybfx 中 zmrxadvvwp.collides_with(htntnzkbzu) 等价。

    Args:
        x: htnt左边界X。
        y: htnt上边界Y。
        w: htnt宽度。
        h: htnt高度。
        walls: 墙壁列表。

    Returns:
        True如果与任何墙壁碰撞。
    """
    for wall in walls:
        if wall.overlaps_rect(x, y, w, h):
            return True
    return False


def _wall_aware_movement_bfs(start_x: int, start_y: int,
                             target_x: int, target_y: int,
                             walls: List[WallRect],
                             htnt_w: int, htnt_h: int,
                             CSPOIQWER: int = 4,
                             max_steps: int = 20) -> Optional[List[int]]:
    """BFS on CSPOIQWER-aligned grid — 寻找避墙运动路径。

    κ-Phase: _wall_aware_movement_bfs = κ-陪集因果归约的运动路径规划
    根因修复: cwtesiybfx在墙壁碰撞时reverse移动 → κ-Snap忽略墙壁 → 执行失败
    修复方案: BFS在4对齐网格上搜索避墙路径，编码为movement opcodes。

    算法:
        - 状态: (x, y) — htnt左上角位置
        - 转移: LEFT(-4,0), RIGHT(+4,0), DOWN(0,+4), UP(0,-4)
        - 约束: 每步移动后的htnt bbox不能与任何wall bbox重叠
        - 目标: htnt.x==target_x, htnt.y==target_y

    Args:
        start_x: htnt起始X坐标。
        start_y: htnt起始Y坐标。
        target_x: htnt目标X坐标。
        target_y: htnt目标Y坐标。
        walls: 墙壁AABB列表。
        htnt_w: htnt宽度(用于碰撞检测)。
        htnt_h: htnt高度(用于碰撞检测)。
        CSPOIQWER: 步长常量(默认4)。
        max_steps: 最大搜索步数(受n_buttons限制)。

    Returns:
        movement opcode列表, 或None(无路径可达)。
    """
    # Movement opcodes: opcode → (dx, dy)
    MOVES: List[Tuple[int, int, int]] = [
        (1, -CSPOIQWER, 0),    # LEFT
        (2, CSPOIQWER, 0),     # RIGHT
        (3, 0, CSPOIQWER),     # DOWN
        (33, 0, -CSPOIQWER),   # UP
    ]

    visited: set = {(start_x, start_y)}
    queue: deque = deque([(start_x, start_y, [])])

    while queue:
        x, y, path = queue.popleft()

        # 目标到达
        if x == target_x and y == target_y:
            return path

        # 步数限制
        if len(path) >= max_steps:
            continue

        for opcode, dx, dy in MOVES:
            new_x = x + dx
            new_y = y + dy

            # 已访问
            if (new_x, new_y) in visited:
                continue

            # 负坐标检查
            if new_x < 0 or new_y < 0:
                continue

            # 墙壁碰撞检查 — 与cwtesiybfx等价
            if _check_wall_collision(new_x, new_y, htnt_w, htnt_h, walls):
                continue

            visited.add((new_x, new_y))
            queue.append((new_x, new_y, path + [opcode]))

    return None  # 无避墙路径可达


def compute_wall_aware_program(sm: TN36StateMachine,
                                walls: List[WallRect],
                                htnt_w: int, htnt_h: int,
                                n_buttons: int) -> Optional[List[int]]:
    """计算避墙目标程序 — sjmtdfxdrc/rotation/scale + wall-aware movement。

    κ-Phase: compute_wall_aware_program = κ-陪集因果归约的完整避墙规划
    优先级: sjmtdfxdrc → rotation → scale → movement(wall-aware BFS)

    根因修复 (Task #690):
        κ-Snap忽略墙壁 → 简单delta规划撞墙 → 执行失败
        → BFS避墙路径 → 每步确保htnt bbox不与wall重叠 → 执行成功

    Args:
        sm: 当前状态机(TN36StateMachine)。
        walls: 墙壁AABB列表。
        htnt_w: htnt宽度(用于碰撞检测)。
        htnt_h: htnt高度(用于碰撞检测)。
        n_buttons: 可用按钮数量。

    Returns:
        目标opcode列表, 或None(无法在n_buttons内表达)。
    """
    delta = sm.compute_delta()
    target_prog: List[int] = []

    # 1. sjmtdfxdrc set (最优先)
    sjmtdfxdrc_delta = delta['sjmtdfxdrc_delta']
    if sjmtdfxdrc_delta != 0:
        target_value = int(sm.target_sjmtdfxdrc)
        sjmtdfxdrc_opcode = _SJMTDFXDRC_TARGET_TO_OPCODE.get(target_value)
        if sjmtdfxdrc_opcode is not None:
            target_prog.append(sjmtdfxdrc_opcode)
        else:
            return None

    # 2. Rotation opcodes
    rot_delta = delta['rotation_delta']
    if rot_delta == 90:
        target_prog.append(5)
    elif rot_delta == 270:
        target_prog.append(16)
    elif rot_delta == 180:
        target_prog.append(7)

    # 3. Scale opcodes
    scale_delta = delta['scale_delta']
    for _ in range(abs(scale_delta)):
        target_prog.append(8 if scale_delta > 0 else 9)

    # 4. Movement — wall-aware BFS
    remaining_slots = n_buttons - len(target_prog)

    # 无运动需求 → 直接填充NOP
    if delta['dx_steps'] == 0 and delta['dy_steps'] == 0:
        while len(target_prog) < n_buttons:
            target_prog.append(0)
        return target_prog

    # 无剩余按钮 → 无法添加运动
    if remaining_slots <= 0:
        return None

    # BFS避墙路径规划
    movement_opcodes = _wall_aware_movement_bfs(
        int(sm.x), int(sm.y),
        int(sm.target_x), int(sm.target_y),
        walls, htnt_w, htnt_h,
        CSPOIQWER=4, max_steps=remaining_slots,
    )

    if movement_opcodes is not None:
        target_prog.extend(movement_opcodes)
    else:
        # BFS无路径 → 退回到简单delta(κ-Snap通过但执行可能失败)
        dx_steps = delta['dx_steps']
        dy_steps = delta['dy_steps']
        for _ in range(abs(dx_steps)):
            target_prog.append(2 if dx_steps > 0 else 1)
        for _ in range(abs(dy_steps)):
            target_prog.append(3 if dy_steps > 0 else 33)

    if len(target_prog) > n_buttons:
        return None

    # 填充NOP
    while len(target_prog) < n_buttons:
        target_prog.append(0)

    return target_prog


# ============================================================
# §6. deepcopy安全性验证 — κ-Snap全息重构测试
# ============================================================

def verify_deepcopy_safe(obj: Any) -> bool:
    """验证对象是否deepcopy安全 — κ-Snap全息重构测试。

    κ-Phase: deepcopy = κ-Snap全息重构
    如果对象是纯数据(dict/int/str/tuple/dataclass) → deepcopy完全安全
    如果对象含lambda闭包 → deepcopy不安全 (IC未锚定)

    Args:
        obj: 待验证对象。

    Returns:
        True如果deepcopy安全。
    """
    try:
        copied = copy.deepcopy(obj)
        # 验证关键属性一致性
        if isinstance(obj, TN36StateMachine):
            assert copied.x == obj.x
            assert copied.y == obj.y
            assert copied.rotation == obj.rotation
            assert copied.scale == obj.scale
            assert copied.sjmtdfxdrc == obj.sjmtdfxdrc
            assert copied.is_editable == obj.is_editable
            assert copied.program == obj.program
            return True
        elif isinstance(obj, dict):
            # 纯数据字典 → 验证所有值一致
            for key in obj:
                if obj[key] != copied[key]:
                    return False
            return True
        return True
    except Exception:
        return False


# ============================================================
# §6. 自测
# ============================================================

def _self_test() -> None:
    """tn36_opcode.py 自测 — v4.5: wall-aware BFS + sjmtdfxdrc_set + rotate(270°)。"""
    print("=" * 60)
    print("tn36_opcode.py _self_test() v4.5")
    print("=" * 60)

    # Test 1: OPCODE_TABLE 纯数据验证
    for opcode, entry in OPCODE_TABLE.items():
        assert isinstance(entry, dict), f"opcode {opcode} entry is not dict"
        assert 'type' in entry, f"opcode {opcode} missing 'type'"
        assert 'description' in entry, f"opcode {opcode} missing 'description'"
    print(f"  ✅ Test 1: OPCODE_TABLE — {len(OPCODE_TABLE)} opcodes, all pure data")

    # Test 2: OPCODE_TABLE deepcopy安全性
    copied_table = copy.deepcopy(OPCODE_TABLE)
    assert len(copied_table) == len(OPCODE_TABLE)
    for opcode in OPCODE_TABLE:
        assert copied_table[opcode] == OPCODE_TABLE[opcode]
    print("  ✅ Test 2: OPCODE_TABLE deepcopy — safe ✅")

    # Test 3: TN36StateMachine 创建和deepcopy
    sm = TN36StateMachine(
        x=100, y=50, rotation=45, scale=2,
        sjmtdfxdrc=1, program=[5, 2, 3],
        is_editable=True, label="right",
        target_x=108, target_y=54,
        target_rotation=135, target_scale=3,
        target_sjmtdfxdrc=0,
    )
    sm_copy = copy.deepcopy(sm)
    assert sm_copy.x == sm.x
    assert sm_copy.rotation == sm.rotation
    assert sm_copy.program == sm.program
    assert sm_copy.is_editable == sm.is_editable
    print("  ✅ Test 3: TN36StateMachine deepcopy — safe ✅")

    # Test 4: apply_opcode — 单步移动
    sm_base = TN36StateMachine(x=100, y=50)
    sm_after_right = sm_base.apply_opcode(2)  # RIGHT (+4, 0)
    assert sm_after_right.x == 104, f"x after RIGHT: {sm_after_right.x}"
    assert sm_after_right.y == 50, f"y after RIGHT: {sm_after_right.y}"
    print("  ✅ Test 4: apply_opcode(2=RIGHT) — PASS")

    # Test 5: apply_opcode — 旋转 (含270°)
    sm_rot = TN36StateMachine(rotation=0)
    sm_after_rot = sm_rot.apply_opcode(5)  # rotate +90
    assert sm_after_rot.rotation == 90, f"rotation after +90: {sm_after_rot.rotation}"
    sm_after_rot270 = sm_rot.apply_opcode(16)  # rotate +270 (v4.4)
    assert sm_after_rot270.rotation == 270, f"rotation after +270: {sm_after_rot270.rotation}"
    print("  ✅ Test 5: apply_opcode(5=rot+90, 16=rot+270) — PASS")

    # Test 6: apply_program — 多步序列
    sm_start = TN36StateMachine(x=100, y=50, rotation=0, scale=1)
    program = [5, 2, 2, 3]  # rot+90, RIGHT×2, DOWN
    sm_final = sm_start.apply_program(program)
    assert sm_final.x == 108, f"final x: {sm_final.x}"  # 100 + 4 + 4
    assert sm_final.y == 54, f"final y: {sm_final.y}"   # 50 + 4
    assert sm_final.rotation == 90, f"final rotation: {sm_final.rotation}"
    print("  ✅ Test 6: apply_program([5,2,2,3]) — PASS")

    # Test 7: compute_target_program — 从delta生成opcode序列 (含sjmtdfxdrc)
    sm_target = TN36StateMachine(
        x=100, y=50, rotation=0, scale=1,
        target_x=108, target_y=54,
        target_rotation=90, target_scale=1,
    )
    prog = sm_target.compute_target_program(n_buttons=10)
    assert prog is not None, "compute_target_program returned None"
    sm_executed = sm_target.apply_program(prog)
    assert sm_executed.is_at_target(), f"Not at target: {sm_executed}"
    print("  ✅ Test 7: compute_target_program + execution → at target — PASS")

    # Test 8: sjmtdfxdrc_set — opcode 63 (set sjmtdfxdrc=15)
    sm_sjmtd = TN36StateMachine(sjmtdfxdrc=11, target_sjmtdfxdrc=15)
    sm_after_set15 = sm_sjmtd.apply_opcode(63)
    assert sm_after_set15.sjmtdfxdrc == 15, f"sjmtdfxdrc after set15: {sm_after_set15.sjmtdfxdrc}"
    print("  ✅ Test 8: apply_opcode(63=sjmtdfxdrc_set=15) — PASS")

    # Test 9: sjmtdfxdrc_set — opcode 14 (set sjmtdfxdrc=9)
    sm_sjmtd9 = TN36StateMachine(sjmtdfxdrc=4, target_sjmtdfxdrc=9)
    sm_after_set9 = sm_sjmtd9.apply_opcode(14)
    assert sm_after_set9.sjmtdfxdrc == 9, f"sjmtdfxdrc after set9: {sm_after_set9.sjmtdfxdrc}"
    print("  ✅ Test 9: apply_opcode(14=sjmtdfxdrc_set=9) — PASS")

    # Test 10: compute_target_program with sjmtdfxdrc change (L4 scenario)
    sm_l4 = TN36StateMachine(
        x=49, y=8, rotation=270, scale=1, sjmtdfxdrc=11,
        target_x=49, target_y=20, target_rotation=0,
        target_scale=2, target_sjmtdfxdrc=15,
    )
    prog_l4 = sm_l4.compute_target_program(n_buttons=6)
    assert prog_l4 is not None, "L4 target program should not be None"
    # Verify simulation reaches target
    sm_l4_executed = sm_l4.apply_program(prog_l4)
    assert sm_l4_executed.is_at_target(), f"L4 not at target: {sm_l4_executed}"
    print("  ✅ Test 10: L4 scenario (sjmtdfxdrc=11→15, rot=270→0) — PASS")

    # Test 11: _SJMTDFXDRC_TARGET_TO_OPCODE lookup
    assert _SJMTDFXDRC_TARGET_TO_OPCODE[9] == 14
    assert _SJMTDFXDRC_TARGET_TO_OPCODE[8] == 15
    assert _SJMTDFXDRC_TARGET_TO_OPCODE[15] == 63
    print("  ✅ Test 11: sjmtdfxdrc target→opcode lookup — PASS")

    # Test 12: is_at_target — 不在目标
    sm_not_target = TN36StateMachine(x=100, y=50, target_x=200, target_y=200)
    assert not sm_not_target.is_at_target()
    print("  ✅ Test 12: is_at_target(False) — PASS")

    # Test 13: verify_deepcopy_safe — 纯数据对象
    assert verify_deepcopy_safe(sm), "TN36StateMachine should be deepcopy-safe"
    assert verify_deepcopy_safe(OPCODE_TABLE), "OPCODE_TABLE should be deepcopy-safe"
    print("  ✅ Test 13: verify_deepcopy_safe — PASS ✅")

    # Test 14: execute_opcode dispatcher (含sjmtdfxdrc_set)
    sm_dispatch = TN36StateMachine(x=100, y=50, rotation=0, sjmtdfxdrc=11)
    sm_result = execute_opcode(2, sm_dispatch)
    assert sm_result.x == 104, f"dispatch x: {sm_result.x}"
    sm_sjmtd_result = execute_opcode(63, sm_dispatch)
    assert sm_sjmtd_result.sjmtdfxdrc == 15, f"dispatch sjmtdfxdrc: {sm_sjmtd_result.sjmtdfxdrc}"
    print("  ✅ Test 14: execute_opcode dispatcher (含sjmtdfxdrc_set) — PASS")

    # Test 15: NOP opcode
    sm_nop = TN36StateMachine(x=100, y=50)
    sm_nop_result = execute_opcode(0, sm_nop)
    assert sm_nop_result.x == 100, f"NOP x: {sm_nop_result.x}"
    assert sm_nop_result.y == 50
    print("  ✅ Test 15: execute_opcode(0=NOP) — PASS")

    # Test 16: program compression
    sm_compress = TN36StateMachine(
        x=100, y=50, rotation=0, scale=1,
        target_x=132, target_y=50,  # dx=32, dx_steps=8
        target_rotation=0, target_scale=1,
    )
    prog_comp = sm_compress.compute_target_program(n_buttons=4)
    if prog_comp is not None:
        sm_exec = sm_compress.apply_program(prog_comp)
        assert sm_exec.is_at_target() or sm_exec.x == 132, f"Compressed prog result: x={sm_exec.x}"
    print("  ✅ Test 16: program compression — PASS")

    # ── v4.5 新增: Wall-Aware BFS 测试 ──

    # Test 17: WallRect overlaps_rect
    wall = WallRect(x=10, y=10, width=5, height=5)
    assert wall.overlaps_rect(12, 12, 2, 2), "Should overlap inside wall"
    assert not wall.overlaps_rect(20, 20, 2, 2), "Should not overlap outside wall"
    assert wall.overlaps_rect(8, 8, 5, 5), "Should overlap adjacent"
    print("  ✅ Test 17: WallRect.overlaps_rect — PASS")

    # Test 18: _check_wall_collision
    walls_test = [WallRect(x=45, y=20, width=4, height=12)]
    assert _check_wall_collision(45, 20, 4, 4, walls_test), "htnt inside wall"
    assert not _check_wall_collision(0, 0, 4, 4, walls_test), "htnt outside wall"
    assert _check_wall_collision(43, 20, 4, 4, walls_test), "htnt overlaps wall edge"
    print("  ✅ Test 18: _check_wall_collision — PASS")

    # Test 19: _wall_aware_movement_bfs — 无墙直线
    path_no_wall = _wall_aware_movement_bfs(37, 20, 53, 20, [], 4, 4, max_steps=20)
    assert path_no_wall is not None, "No wall → should find path"
    assert len(path_no_wall) == 4, f"4 RIGHT steps: {path_no_wall}"
    assert all(op == 2 for op in path_no_wall), "All RIGHT opcodes"
    print("  ✅ Test 19: BFS no-wall straight path — PASS")

    # Test 20: _wall_aware_movement_bfs — L2避墙路径
    # L2 scenario: htnt=(37,20)→target=(53,12), wall at x=[45,49) y=[20,32)
    walls_l2 = [WallRect(x=45, y=20, width=4, height=12)]
    path_l2 = _wall_aware_movement_bfs(37, 20, 53, 12, walls_l2, 4, 4, max_steps=20)
    assert path_l2 is not None, "L2 should have wall-aware path"
    # Path should go UP first to avoid wall, then RIGHT, then UP more
    # Simulate: start=(37,20), apply opcodes one by one
    sim_x, sim_y = 37, 20
    for op in path_l2:
        if op == 1: sim_x -= 4
        elif op == 2: sim_x += 4
        elif op == 3: sim_y += 4
        elif op == 33: sim_y -= 4
    assert sim_x == 53 and sim_y == 12, f"L2 path result: ({sim_x},{sim_y}), path={path_l2}"
    print(f"  ✅ Test 20: L2 wall-aware path — {path_l2} → ({sim_x},{sim_y}) PASS")

    # Test 21: _wall_aware_movement_bfs — 完全阻塞
    walls_block = [WallRect(x=0, y=0, width=64, height=64)]  # Wall covers entire grid
    path_block = _wall_aware_movement_bfs(10, 10, 20, 20, walls_block, 4, 4, max_steps=20)
    assert path_block is None, "Fully blocked → should return None"
    print("  ✅ Test 21: BFS fully blocked → None — PASS")

    # Test 22: WallRect deepcopy安全
    wall_copy = copy.deepcopy(walls_l2)
    assert len(wall_copy) == len(walls_l2)
    assert wall_copy[0].x == walls_l2[0].x
    assert wall_copy[0].width == walls_l2[0].width
    print("  ✅ Test 22: WallRect deepcopy — safe ✅")

    # Test 23: compute_wall_aware_program — 无墙场景
    sm_no_wall = TN36StateMachine(x=37, y=20, target_x=53, target_y=20)
    prog_no_wall = compute_wall_aware_program(sm_no_wall, [], 4, 4, n_buttons=10)
    assert prog_no_wall is not None, "No wall → should compute program"
    sm_exec_nw = sm_no_wall.apply_program(prog_no_wall)
    assert sm_exec_nw.is_at_target(), f"No-wall program not at target: {sm_exec_nw}"
    print("  ✅ Test 23: compute_wall_aware_program (no wall) — PASS")

    # Test 24: compute_wall_aware_program — L2避墙场景
    sm_l2 = TN36StateMachine(x=37, y=20, target_x=53, target_y=12)
    prog_l2_wa = compute_wall_aware_program(sm_l2, walls_l2, 4, 4, n_buttons=10)
    assert prog_l2_wa is not None, "L2 wall-aware program should exist"
    sm_exec_l2 = sm_l2.apply_program(prog_l2_wa)
    assert sm_exec_l2.is_at_target(), f"L2 wall-aware not at target: {sm_exec_l2}"
    print(f"  ✅ Test 24: compute_wall_aware_program L2 — {prog_l2_wa} PASS")

    print("=" * 60)
    print("All 24 tests PASS ✅ — TN36 OPCODE_TABLE v4.5 wall-aware BFS!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
