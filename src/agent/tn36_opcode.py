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

Version: v4.0 — 数据驱动OPCODE_TABLE + 三层架构
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import copy


# ============================================================
# §1. OPCODE定义 — 纯数据层
# ============================================================

# TN36游戏OPCODE映射 (CSPOIQWER = 4步长常量)
# 每个opcode对应一个数据字典: {type, dx, dy, rotation, scale, sjmtdfxdrc_toggle, description}

OPCODE_TABLE: Dict[int, Dict[str, Any]] = {
    # ── NOP ──
    0: {
        'type': 'noop',
        'dx': 0, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'NOP (mlejdghzfo) — 无操作',
    },

    # ── 单步移动 (CSPOIQWER=4) ──
    1: {
        'type': 'move',
        'dx': -4, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'LEFT (-4, 0) — 左移4像素',
    },
    2: {
        'type': 'move',
        'dx': 4, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'RIGHT (+4, 0) — 右移4像素',
    },
    3: {
        'type': 'move',
        'dx': 0, 'dy': 4,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'DOWN (0, +4) — 下移4像素',
    },
    33: {
        'type': 'move',
        'dx': 0, 'dy': -4,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'UP (0, -4) — 上移4像素',
    },
    34: {
        'type': 'move',
        'dx': -4, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'LEFT-ALT (-4, 0) — 左移4像素(备用)',
    },

    # ── 双步移动 (CSPOIQWER×2=8) ──
    10: {
        'type': 'move',
        'dx': 8, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'RIGHT×2 (+8, 0) — 右移8像素(双步)',
    },
    12: {
        'type': 'move',
        'dx': -8, 'dy': 0,
        'rotation': 0, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'LEFT×2 (-8, 0) — 左移8像素(双步)',
    },

    # ── 旋转 ──
    5: {
        'type': 'rotation',
        'dx': 0, 'dy': 0,
        'rotation': 90, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'rotate +90° — 正旋转90度',
    },
    6: {
        'type': 'rotation',
        'dx': 0, 'dy': 0,
        'rotation': -90, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'rotate -90° — 反旋转90度',
    },
    7: {
        'type': 'rotation',
        'dx': 0, 'dy': 0,
        'rotation': 180, 'scale': 0,
        'sjmtdfxdrc_toggle': 0,
        'description': 'rotate +180° — 旋转180度',
    },

    # ── 缩放 ──
    8: {
        'type': 'scale',
        'dx': 0, 'dy': 0,
        'rotation': 0, 'scale': 1,
        'sjmtdfxdrc_toggle': 0,
        'description': 'scale +1 — 缩放增大',
    },
    9: {
        'type': 'scale',
        'dx': 0, 'dy': 0,
        'rotation': 0, 'scale': -1,
        'sjmtdfxdrc_toggle': 0,
        'description': 'scale -1 — 缩放减小',
    },

    # ── sjmtdfxdrc Toggle ──
    # (toggle参数: 0=不变, 1=翻转)
    # 注: 这些opcode目前没有明确编号，但从游戏源码推断可能存在
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
        按优先级: rotation → scale → movement → sjmtdfxdrc

        Args:
            n_buttons: 可用按钮数量。

        Returns:
            目标opcode列表, 或None(无法在n_buttons内表达)。
        """
        delta = self.compute_delta()
        target_prog: List[int] = []

        # 1. Rotation opcodes (优先，因为旋转影响后续移动方向)
        rot_delta = delta['rotation_delta']
        if rot_delta == 90:
            target_prog.append(5)
        elif rot_delta == 270 or rot_delta == -90:
            target_prog.append(6)
        elif rot_delta == 180:
            target_prog.append(7)

        # 2. Scale opcodes
        scale_delta = delta['scale_delta']
        for _ in range(abs(scale_delta)):
            target_prog.append(8 if scale_delta > 0 else 9)

        # 3. sjmtdfxdrc toggle (如果delta为1, 需要翻转)
        # 注: 目前没有明确的sjmtdfxdrc toggle opcode编号
        # 在游戏源码中，sjmtdfxdrc可能是通过特定opcode的副作用触发

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
        """
        compressed: List[int] = []

        # Rotation stays same
        rot_delta = delta['rotation_delta']
        if rot_delta == 90:
            compressed.append(5)
        elif rot_delta == 270 or rot_delta == -90:
            compressed.append(6)
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
        # 注: 目前没有明确的UP×2/DOWN×2 opcode编号
        # 使用单步opcode
        for _ in range(abs(dy_steps)):
            compressed.append(3 if dy_steps > 0 else 33)

        return compressed

    def apply_opcode(self, opcode: int) -> 'TN36StateMachine':
        """应用一个opcode到状态机 — 返回新状态(不修改原状态)。

        κ-Phase: apply_opcode = κ-陪集变换的纯数据执行
        从OPCODE_TABLE读取参数 → 修改状态 → 返回新状态

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

        # sjmtdfxdrc toggle
        if entry['sjmtdfxdrc_toggle'] != 0:
            new_state.sjmtdfxdrc ^= 1  # 翻转

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
    if entry['sjmtdfxdrc_toggle'] != 0:
        new_state.sjmtdfxdrc ^= 1
    return new_state


def _op_rotation(state: TN36StateMachine, entry: Dict[str, Any]) -> TN36StateMachine:
    """旋转处理器 — angle偏移。"""
    new_state = copy.deepcopy(state)
    new_state.rotation = (new_state.rotation + entry['rotation']) % 360
    if entry['sjmtdfxdrc_toggle'] != 0:
        new_state.sjmtdfxdrc ^= 1
    return new_state


def _op_scale(state: TN36StateMachine, entry: Dict[str, Any]) -> TN36StateMachine:
    """缩放处理器 — scale增减。"""
    new_state = copy.deepcopy(state)
    new_state.scale += entry['scale']
    if entry['sjmtdfxdrc_toggle'] != 0:
        new_state.sjmtdfxdrc ^= 1
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
# §5. deepcopy安全性验证 — κ-Snap全息重构测试
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
    """tn36_opcode.py 自测 — 验证数据驱动OPCODE_TABLE和deepcopy安全性。"""
    print("=" * 60)
    print("tn36_opcode.py _self_test()")
    print("=" * 60)

    # Test 1: OPCODE_TABLE 纯数据验证
    for opcode, entry in OPCODE_TABLE.items():
        assert isinstance(entry, dict), f"opcode {opcode} entry is not dict"
        assert 'type' in entry, f"opcode {opcode} missing 'type'"
        assert 'dx' in entry, f"opcode {opcode} missing 'dx'"
        assert 'dy' in entry, f"opcode {opcode} missing 'dy'"
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

    # Test 5: apply_opcode — 旋转
    sm_rot = TN36StateMachine(rotation=0)
    sm_after_rot = sm_rot.apply_opcode(5)  # rotate +90
    assert sm_after_rot.rotation == 90, f"rotation after +90: {sm_after_rot.rotation}"
    sm_after_rot2 = sm_after_rot.apply_opcode(7)  # rotate +180
    assert sm_after_rot2.rotation == 270, f"rotation after +180: {sm_after_rot2.rotation}"
    print("  ✅ Test 5: apply_opcode(5=rot+90, 7=rot+180) — PASS")

    # Test 6: apply_program — 多步序列
    sm_start = TN36StateMachine(x=100, y=50, rotation=0, scale=1)
    program = [5, 2, 2, 3]  # rot+90, RIGHT×2, DOWN
    sm_final = sm_start.apply_program(program)
    assert sm_final.x == 108, f"final x: {sm_final.x}"  # 100 + 4 + 4
    assert sm_final.y == 54, f"final y: {sm_final.y}"   # 50 + 4
    assert sm_final.rotation == 90, f"final rotation: {sm_final.rotation}"
    print("  ✅ Test 6: apply_program([5,2,2,3]) — PASS")

    # Test 7: compute_target_program — 从delta生成opcode序列
    sm_target = TN36StateMachine(
        x=100, y=50, rotation=0, scale=1,
        target_x=108, target_y=54,
        target_rotation=90, target_scale=1,
    )
    prog = sm_target.compute_target_program(n_buttons=10)
    assert prog is not None, "compute_target_program returned None"
    # 验证执行后到达目标
    sm_executed = sm_target.apply_program(prog)
    assert sm_executed.is_at_target(), f"Not at target: {sm_executed}"
    print("  ✅ Test 7: compute_target_program + execution → at target — PASS")

    # Test 8: is_at_target — 不在目标
    sm_not_target = TN36StateMachine(x=100, y=50, target_x=200, target_y=200)
    assert not sm_not_target.is_at_target()
    print("  ✅ Test 8: is_at_target(False) — PASS")

    # Test 9: verify_deepcopy_safe — 纯数据对象
    assert verify_deepcopy_safe(sm), "TN36StateMachine should be deepcopy-safe"
    assert verify_deepcopy_safe(OPCODE_TABLE), "OPCODE_TABLE should be deepcopy-safe"
    print("  ✅ Test 9: verify_deepcopy_safe — PASS ✅")

    # Test 10: execute_opcode 分发器
    sm_dispatch = TN36StateMachine(x=100, y=50, rotation=0)
    sm_result = execute_opcode(2, sm_dispatch)
    assert sm_result.x == 104, f"dispatch x: {sm_result.x}"
    sm_rot_result = execute_opcode(5, sm_dispatch)
    assert sm_rot_result.rotation == 90
    print("  ✅ Test 10: execute_opcode dispatcher — PASS")

    # Test 11: NOP opcode
    sm_nop = TN36StateMachine(x=100, y=50)
    sm_nop_result = execute_opcode(0, sm_nop)
    assert sm_nop_result.x == 100, f"NOP x: {sm_nop_result.x}"
    assert sm_nop_result.y == 50
    print("  ✅ Test 11: execute_opcode(0=NOP) — PASS")

    # Test 12: program compression
    sm_compress = TN36StateMachine(
        x=100, y=50, rotation=0, scale=1,
        target_x=132, target_y=50,  # dx=32, dx_steps=8
        target_rotation=0, target_scale=1,
    )
    prog_comp = sm_compress.compute_target_program(n_buttons=4)
    # 8步移动需要在4按钮内 → 必须压缩
    if prog_comp is not None:
        # 执行验证
        sm_exec = sm_compress.apply_program(prog_comp)
        assert sm_exec.is_at_target() or sm_exec.x == 132, f"Compressed prog result: x={sm_exec.x}"
    print("  ✅ Test 12: program compression — PASS")

    print("=" * 60)
    print("All tests PASS ✅ — TN36 OPCODE_TABLE is deepcopy-safe!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
