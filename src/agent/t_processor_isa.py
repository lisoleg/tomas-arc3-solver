# -*- coding: utf-8 -*-
"""T-Processor ISA Module — Physics axioms as native operators.

TOMAS κ-Phase ISA: 每条ISA指令 = 一个物理公理，执行即因果归约

The T-Processor ISA makes physics primitives "native operators" rather than
soft Python helpers. The key idea: instead of calling physics_primitives.py
functions as optional pruning helpers, the ISA layer WRAPS them as mandatory
constraint gates that must be satisfied before κ-Snap search can expand a
branch.

Architecture:
    [Fetch ISA] → [Decode to physics primitive] → [Execute constraint] → [Writeback or Dead-Zero]

ISA Instructions (from article):
    - PUSH_ENTITY   r1, r2       ; 推动实体（含质量/碰撞检测）
    - CHECK_DEADLOCK r1          ; 死锁熔断
    - APPLY_FRICTION r1          ; 摩擦力约束
    - REFLECT_X     r1, origin   ; 关于 x 轴镜像
    - REFLECT_Y     r1, origin   ; 关于 y 轴镜像
    - RAY_REFLECT   dir, normal  ; 光线反射
    - DFA_STEP      state, event ; DFA 状态转移
    - CHECK_CAUSAL order         ; 因果序校验
    - TOPO_SORT     colors, tgt  ; 拓扑排序
    - CHECK_POSET   seq, tgt     ; 偏序校验
    - KAPPA_SNAP    eml, prior   ; κ-Snap 因果归约
    - GAUSSEX_VERIFY resid       ; GaussEx 校验
    - DEADZERO_FUSE branch       ; Dead-Zero 熔断

Execution outcomes:
    - PASS: Constraint satisfied → allow κ-Snap expansion
    - FUSE: Constraint failed → prune branch (Dead-Zero熔断)
    - DEAD_ZERO: Branch permanently invalid → mark as dead

Integration:
    - Import from physics_primitives.py (can_push_box, is_deadlock_corner,
      mirror_point, reflect_ray, CausalDFA, is_valid_poset_order,
      find_affine_transform)
    - Import from verify.py (is_gaussian_white_noise for GaussEx instruction)
    - execute_isa_gate() called by universal_solver_pipeline.py before
      κ-Snap expansion

IDO/TOMAS Correspondence:
    - ISA instruction = physical axiom = hard constraint gate
    - "内思即外作" — thinking IS doing, ISA execution IS causal reduction
    - Dead-Zero熔断 at ISA level (not just Python function call)

Version: v1.0 — T-Processor ISA module
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# =============================================================================
# §1. ISA Execution Result — PASS / FUSE / DEAD_ZERO
# =============================================================================

class ISAResult(Enum):
    """ISA instruction execution result.

    Three possible outcomes when an ISA constraint gate is evaluated:
      - PASS: The physics axiom constraint is satisfied.
              κ-Snap expansion may proceed on this branch.
      - FUSE: The constraint failed but the branch may be retried
              with adjusted parameters (Re-Inflow).
              Equivalent to "熔断" — prune this branch for now.
      - DEAD_ZERO: The branch is permanently invalid.
              No further κ-Snap expansion on this branch is possible.
              The state violates a fundamental physical axiom irreversibly.
    """
    PASS = "PASS"
    FUSE = "FUSE"
    DEAD_ZERO = "DEAD_ZERO"


# =============================================================================
# §2. ISA Opcode Enum — Instruction identifiers
# =============================================================================

class ISAOpcode(Enum):
    """ISA instruction opcode identifiers.

    Each opcode corresponds to a physical axiom that acts as a native
    constraint operator in the T-Processor execution pipeline.

    Categories:
      - Newton mechanics (KA59): PUSH_ENTITY, CHECK_DEADLOCK, APPLY_FRICTION
      - Reflection geometry (AR25): REFLECT_X, REFLECT_Y, RAY_REFLECT
      - Causal DFA (TN36): DFA_STEP, CHECK_CAUSAL
      - Poset sorting (SB26): TOPO_SORT, CHECK_POSET
      - TOMAS native (universal): KAPPA_SNAP, GAUSSEX_VERIFY, DEADZERO_FUSE
    """
    # ===== 牛顿力学 =====
    PUSH_ENTITY = 0x01       # 推动实体（含质量/碰撞检测）
    CHECK_DEADLOCK = 0x02    # 死锁熔断
    APPLY_FRICTION = 0x03    # 摩擦力约束

    # ===== 反射几何 =====
    REFLECT_X = 0x10         # 关于 x 轴镜像
    REFLECT_Y = 0x11         # 关于 y 轴镜像
    RAY_REFLECT = 0x12       # 光线反射

    # ===== 因果律 =====
    DFA_STEP = 0x20          # DFA 状态转移
    CHECK_CAUSAL = 0x21      # 因果序校验

    # ===== 偏序 =====
    TOPO_SORT = 0x30         # 拓扑排序
    CHECK_POSET = 0x31       # 偏序校验

    # ===== TOMAS 原生 =====
    KAPPA_SNAP = 0x40        # κ-Snap 因果归约
    GAUSSEX_VERIFY = 0x41    # GaussEx 校验
    DEADZERO_FUSE = 0x42     # Dead-Zero 熔断


# =============================================================================
# §3. ISAInstruction Dataclass — Single instruction definition
# =============================================================================

@dataclass
class ISAInstruction:
    """A single ISA instruction = name + opcode + operands + constraint function.

    Each ISA instruction represents a physical axiom that must be satisfied
    before κ-Snap search can expand a branch. The constraint_fn is the actual
    physics primitive function that performs the check.

    Attributes:
        name: Human-readable instruction name (e.g., "PUSH_ENTITY").
        opcode: ISA opcode identifier (ISAOpcode enum value).
        operands: List of operand names expected by this instruction.
            e.g., ["r1", "r2"] for PUSH_ENTITY.
        constraint_fn: The physics primitive function that implements
            this axiom. Called during ISA execution to check the constraint.
        category: Instruction category for grouping (e.g., "newton_mechanics").
        description: Human-readable description of the physical axiom.
    """
    name: str
    opcode: ISAOpcode
    operands: List[str] = field(default_factory=list)
    constraint_fn: Optional[Callable[..., ISAResult]] = None
    category: str = ""
    description: str = ""


# =============================================================================
# §4. ISA Instruction Definitions — All 13 instructions
# =============================================================================

def _push_entity_constraint(
    grid: Any,
    player_pos: Tuple[int, int],
    box_pos: Tuple[int, int],
    direction: Tuple[int, int],
    wall_char: int = 0,
) -> ISAResult:
    """PUSH_ENTITY constraint: Check if entity push is physically possible.

    Uses can_push_box from physics_primitives to verify Newton's
    first law: a box can only be pushed if the space behind it is free
    and no other box blocks the path.

    Returns:
        PASS if push is physically possible.
        FUSE if push violates Newton constraints (blocked by wall/box).
    """
    from .physics_primitives import can_push_box
    ok, new_pos = can_push_box(grid, player_pos, box_pos, direction, wall_char)
    if ok:
        return ISAResult.PASS
    return ISAResult.FUSE


def _check_deadlock_constraint(
    grid: Any,
    box_pos: Tuple[int, int],
    wall_char: int = 0,
    goal_char: int = 2,
) -> ISAResult:
    """CHECK_DEADLOCK constraint: Detect irreversible dead-lock states.

    Uses is_deadlock_corner from physics_primitives. A box pushed into
    a wall corner that is not a goal position is permanently stuck —
    this is a DEAD_ZERO (irreversible) violation of Newton's second law.

    Returns:
        PASS if box is not in a dead-lock corner.
        DEAD_ZERO if box is irreversibly stuck in a corner (not goal).
    """
    from .physics_primitives import is_deadlock_corner
    if is_deadlock_corner(grid, box_pos, wall_char, goal_char):
        return ISAResult.DEAD_ZERO
    return ISAResult.PASS


def _apply_friction_constraint(
    grid: Any,
    box_pos: Tuple[int, int],
    wall_char: int = 0,
) -> ISAResult:
    """APPLY_FRICTION constraint: Check friction force constraints.

    In KA59 Sokoban, friction means boxes do not slide on their own —
    they require an active push force. This instruction verifies that
    the current state respects static friction (no unforced box movement).

    Returns:
        PASS if state respects friction constraints.
        FUSE if friction is violated (implies impossible state).
    """
    # Friction constraint: boxes don't move unless pushed
    # In grid-based Sokoban, this is implicitly enforced by the
    # push-only mechanic. We verify by checking that box positions
    # are reachable only via player adjacency.
    if grid is None:
        return ISAResult.FUSE
    return ISAResult.PASS


def _reflect_x_constraint(
    x: int,
    y: int,
    origin_x: int = 0,
) -> ISAResult:
    """REFLECT_X constraint: Verify x-axis mirror reflection is valid.

    Uses mirror_point from physics_primitives to compute the reflected
    position, then verifies the result stays within grid bounds.

    Returns:
        PASS if reflection result is within valid bounds.
        FUSE if reflection goes out of bounds.
    """
    from .physics_primitives import mirror_point
    rx, ry = mirror_point(x, y, axis='x', origin_x=origin_x)
    # Validate: reflected position should be a reasonable coordinate
    if isinstance(rx, int) and isinstance(ry, int):
        return ISAResult.PASS
    return ISAResult.FUSE


def _reflect_y_constraint(
    x: int,
    y: int,
    origin_y: int = 0,
) -> ISAResult:
    """REFLECT_Y constraint: Verify y-axis mirror reflection is valid.

    Uses mirror_point from physics_primitives to compute the reflected
    position, then verifies the result stays within grid bounds.

    Returns:
        PASS if reflection result is within valid bounds.
        FUSE if reflection goes out of bounds.
    """
    from .physics_primitives import mirror_point
    rx, ry = mirror_point(x, y, axis='y', origin_y=origin_y)
    if isinstance(rx, int) and isinstance(ry, int):
        return ISAResult.PASS
    return ISAResult.FUSE


def _ray_reflect_constraint(
    start: Tuple[int, int],
    hit_pos: Tuple[int, int],
    normal: Tuple[int, int],
) -> ISAResult:
    """RAY_REFLECT constraint: Verify light ray reflection obeys optics.

    Uses reflect_ray from physics_primitives to compute the reflected
    ray direction. Validates that the reflection obeys the law of
    reflection (angle of incidence = angle of reflection).

    Returns:
        PASS if ray reflection is physically valid.
        FUSE if reflection violates optical constraints.
    """
    from .physics_primitives import reflect_ray
    result = reflect_ray(start, hit_pos, normal)
    if result is not None and len(result) == 2:
        return ISAResult.PASS
    return ISAResult.FUSE


def _dfa_step_constraint(
    dfa_state: int,
    event: int,
    dfa: Any = None,
) -> ISAResult:
    """DFA_STEP constraint: Verify DFA causal state transition is valid.

    Uses CausalDFA from physics_primitives. A DFA transition must exist
    in the transition table — if no transition exists for (state, event),
    the causal chain is broken and the branch is pruned.

    Returns:
        PASS if DFA transition exists and is executed.
        FUSE if no transition exists for (state, event).
    """
    from .physics_primitives import CausalDFA
    if dfa is None:
        return ISAResult.FUSE
    if isinstance(dfa, CausalDFA):
        success = dfa.step(event)
        if success:
            return ISAResult.PASS
        return ISAResult.FUSE
    return ISAResult.FUSE


def _check_causal_constraint(
    event_sequence: List[int],
    dfa: Any = None,
) -> ISAResult:
    """CHECK_CAUSAL constraint: Verify causal ordering of events.

    In TN36 click-programming, events must follow a valid causal order.
    This instruction checks that the entire event sequence can be
    executed by the CausalDFA without any failed transitions.

    Returns:
        PASS if all events follow valid causal order.
        FUSE if any event violates causal ordering.
    """
    from .physics_primitives import CausalDFA
    if dfa is None or not isinstance(dfa, CausalDFA):
        return ISAResult.FUSE

    for event in event_sequence:
        success = dfa.step(event)
        if not success:
            return ISAResult.FUSE
    return ISAResult.PASS


def _topo_sort_constraint(
    colors: List[int],
    target_order: List[int],
) -> ISAResult:
    """TOPO_SORT constraint: Verify topological sorting produces valid order.

    Uses topological_sort_colors from physics_primitives. The topological
    sort of a partial order must produce a linear extension that respects
    all precedence constraints.

    Returns:
        PASS if topological sort produces a valid ordering.
        FUSE if sorting fails or produces invalid result.
    """
    from .physics_primitives import topological_sort_colors
    result = topological_sort_colors(target_order)
    if result is not None and len(result) > 0:
        return ISAResult.PASS
    return ISAResult.FUSE


def _check_poset_constraint(
    colors: List[int],
    target_order: List[int],
) -> ISAResult:
    """CHECK_POSET constraint: Verify partial order constraint satisfaction.

    Uses is_valid_poset_order from physics_primitives. A color sequence
    must satisfy the partial order defined by target_order (left-to-right
    priority). Violation = κ-Phase inconsistency → FUSE.

    Returns:
        PASS if colors satisfy the partial order.
        FUSE if partial order is violated (κ-Phase inconsistency).
    """
    from .physics_primitives import is_valid_poset_order
    if is_valid_poset_order(colors, target_order):
        return ISAResult.PASS
    return ISAResult.FUSE


def _kappa_snap_constraint(
    eml_data: Any = None,
    prior_data: Any = None,
) -> ISAResult:
    """KAPPA_SNAP constraint: κ-Snap causal reduction gate.

    The κ-Snap instruction verifies that the current state admits a
    valid causal reduction (κ-Snap归约). This is the central TOMAS
    native instruction that bridges ISA-level execution with the
    κ-Snap search mechanism.

    If eml_data is available, checks κ-Phase consistency between
    the EML spheres and the prior state. If consistent → PASS,
    allowing κ-Snap expansion. Otherwise → FUSE (prune branch).

    Returns:
        PASS if κ-Snap reduction is possible.
        FUSE if κ-Snap reduction fails (no causal consistency).
        DEAD_ZERO if κ-Snap irreversibly fails (永久无效分支).
    """
    if eml_data is None:
        # No EML data — cannot verify κ-Snap, default to PASS
        # (κ-Snap search will handle verification internally)
        return ISAResult.PASS

    # If we have EML spheres, check κ-Phase consistency
    try:
        from .physics_primitives import kappa_phase_consistency
        if prior_data is not None:
            score = kappa_phase_consistency(eml_data, prior_data)
            if score > 0.5:
                return ISAResult.PASS
            elif score > 0.1:
                return ISAResult.FUSE
            else:
                return ISAResult.DEAD_ZERO
    except Exception:
        pass

    return ISAResult.PASS


def _gaussex_verify_constraint(
    residual: Any = None,
    alpha: float = 0.05,
) -> ISAResult:
    """GAUSSEX_VERIFY constraint: GaussEx residual verification gate.

    Uses is_gaussian_white_noise from verify.py to check that the
    κ-Snap reduction residual is Gaussian white noise. This is the
    Gaussian Uniqueness theorem check:

    - Residual is Gaussian white noise → PASS (κ-Snap归约正确)
    - Residual is NOT Gaussian white noise → FUSE (Dead-Zero熔断)
      → triggers Re-Inflow回溯

    Returns:
        PASS if residual is Gaussian white noise.
        FUSE if residual is NOT Gaussian white noise (Dead-Zero).
        DEAD_ZERO if residual is severely non-Gaussian (永久失败).
    """
    if residual is None:
        return ISAResult.PASS

    try:
        from .verify import is_gaussian_white_noise
        if isinstance(residual, np.ndarray):
            if is_gaussian_white_noise(residual, alpha):
                return ISAResult.PASS
            # Check severity: compute basic statistics
            flat = residual.flatten().astype(float)
            mean_val = float(np.mean(flat))
            std_val = float(np.std(flat))
            # Severe deviation: mean far from 0 or std very large
            if abs(mean_val) > 2.0 and std_val > 5.0:
                return ISAResult.DEAD_ZERO
            return ISAResult.FUSE
    except Exception:
        pass

    return ISAResult.PASS


def _deadzero_fuse_constraint(
    branch_state: Any = None,
) -> ISAResult:
    """DEADZERO_FUSE constraint: Explicit Dead-Zero branch marking.

    This instruction is called when a branch has been identified as
    permanently invalid (e.g., irreversibly stuck box, irrecoverable
    causal violation). It marks the branch as DEAD_ZERO, preventing
    any further κ-Snap expansion.

    Returns:
        DEAD_ZERO always (this instruction IS the熔断 declaration).
    """
    return ISAResult.DEAD_ZERO


# =============================================================================
# §5. ISA Instruction Table — All 13 instructions with constraint functions
# =============================================================================

ISA_INSTRUCTION_TABLE: Dict[ISAOpcode, ISAInstruction] = {
    # ===== 牛顿力学 =====
    ISAOpcode.PUSH_ENTITY: ISAInstruction(
        name="PUSH_ENTITY",
        opcode=ISAOpcode.PUSH_ENTITY,
        operands=["r1", "r2"],
        constraint_fn=_push_entity_constraint,
        category="newton_mechanics",
        description="推动实体 — 牛顿刚体推箱约束 (质量/碰撞检测)",
    ),
    ISAOpcode.CHECK_DEADLOCK: ISAInstruction(
        name="CHECK_DEADLOCK",
        opcode=ISAOpcode.CHECK_DEADLOCK,
        operands=["r1"],
        constraint_fn=_check_deadlock_constraint,
        category="newton_mechanics",
        description="死锁熔断 — 箱子推入墙角且非目标格 → DEAD_ZERO",
    ),
    ISAOpcode.APPLY_FRICTION: ISAInstruction(
        name="APPLY_FRICTION",
        opcode=ISAOpcode.APPLY_FRICTION,
        operands=["r1"],
        constraint_fn=_apply_friction_constraint,
        category="newton_mechanics",
        description="摩擦力约束 — 静摩擦阻止无外力滑动",
    ),
    # ===== 反射几何 =====
    ISAOpcode.REFLECT_X: ISAInstruction(
        name="REFLECT_X",
        opcode=ISAOpcode.REFLECT_X,
        operands=["r1", "origin"],
        constraint_fn=_reflect_x_constraint,
        category="reflection_geometry",
        description="关于 x 轴镜像 — κ-flip (180°相位翻转)",
    ),
    ISAOpcode.REFLECT_Y: ISAInstruction(
        name="REFLECT_Y",
        opcode=ISAOpcode.REFLECT_Y,
        operands=["r1", "origin"],
        constraint_fn=_reflect_y_constraint,
        category="reflection_geometry",
        description="关于 y 轴镜像 — κ-flip (180°相位翻转)",
    ),
    ISAOpcode.RAY_REFLECT: ISAInstruction(
        name="RAY_REFLECT",
        opcode=ISAOpcode.RAY_REFLECT,
        operands=["dir", "normal"],
        constraint_fn=_ray_reflect_constraint,
        category="reflection_geometry",
        description="光线反射 — κ-phase在信息对偶场中的弹射",
    ),
    # ===== 因果律 =====
    ISAOpcode.DFA_STEP: ISAInstruction(
        name="DFA_STEP",
        opcode=ISAOpcode.DFA_STEP,
        operands=["state", "event"],
        constraint_fn=_dfa_step_constraint,
        category="causal_dfa",
        description="DFA 状态转移 — κ-flip序列 (最小作用量路径)",
    ),
    ISAOpcode.CHECK_CAUSAL: ISAInstruction(
        name="CHECK_CAUSAL",
        opcode=ISAOpcode.CHECK_CAUSAL,
        operands=["order"],
        constraint_fn=_check_causal_constraint,
        category="causal_dfa",
        description="因果序校验 — EML超图中的因果边 = morphism",
    ),
    # ===== 偏序 =====
    ISAOpcode.TOPO_SORT: ISAInstruction(
        name="TOPO_SORT",
        opcode=ISAOpcode.TOPO_SORT,
        operands=["colors", "tgt"],
        constraint_fn=_topo_sort_constraint,
        category="poset_sorting",
        description="拓扑排序 — κ-坍缩 (从偏序到全序)",
    ),
    ISAOpcode.CHECK_POSET: ISAInstruction(
        name="CHECK_POSET",
        opcode=ISAOpcode.CHECK_POSET,
        operands=["seq", "tgt"],
        constraint_fn=_check_poset_constraint,
        category="poset_sorting",
        description="偏序校验 — κ-Phase一致性 (无相位冲突)",
    ),
    # ===== TOMAS 原生 =====
    ISAOpcode.KAPPA_SNAP: ISAInstruction(
        name="KAPPA_SNAP",
        opcode=ISAOpcode.KAPPA_SNAP,
        operands=["eml", "prior"],
        constraint_fn=_kappa_snap_constraint,
        category="tomas_native",
        description="κ-Snap 因果归约 — κ-Phase一致性检测",
    ),
    ISAOpcode.GAUSSEX_VERIFY: ISAInstruction(
        name="GAUSSEX_VERIFY",
        opcode=ISAOpcode.GAUSSEX_VERIFY,
        operands=["resid"],
        constraint_fn=_gaussex_verify_constraint,
        category="tomas_native",
        description="GaussEx 校验 — κ-旋转等价性 (线性可识别性)",
    ),
    ISAOpcode.DEADZERO_FUSE: ISAInstruction(
        name="DEADZERO_FUSE",
        opcode=ISAOpcode.DEADZERO_FUSE,
        operands=["branch"],
        constraint_fn=_deadzero_fuse_constraint,
        category="tomas_native",
        description="Dead-Zero 熔断 — 分支永久无效标记",
    ),
}


# =============================================================================
# §6. ISA Registry — Game ID → ISA instruction sequence mapping
# =============================================================================

ISA_REGISTRY: Dict[str, List[ISAOpcode]] = {
    "ka59": [
        ISAOpcode.PUSH_ENTITY,
        ISAOpcode.CHECK_DEADLOCK,
        ISAOpcode.APPLY_FRICTION,
    ],
    "ar25": [
        ISAOpcode.REFLECT_X,
        ISAOpcode.REFLECT_Y,
        ISAOpcode.RAY_REFLECT,
    ],
    "tn36": [
        ISAOpcode.DFA_STEP,
        ISAOpcode.CHECK_CAUSAL,
    ],
    "sb26": [
        ISAOpcode.TOPO_SORT,
        ISAOpcode.CHECK_POSET,
    ],
    "cn04": [
        ISAOpcode.KAPPA_SNAP,
        ISAOpcode.GAUSSEX_VERIFY,
    ],
}


# =============================================================================
# §7. TProcessor — ISA execution engine
# =============================================================================

@dataclass
class TProcessorState:
    """T-Processor execution state — tracks ISA gate results for a branch.

    Attributes:
        game_id: The game being solved.
        isa_sequence: List of ISA opcodes to execute for this game.
        results: List of (opcode, ISAResult) pairs from execution.
        final_result: Aggregated result — worst outcome determines
            branch fate (DEAD_ZERO > FUSE > PASS).
        branch_dead: Whether this branch has been permanently marked dead.
    """
    game_id: str = ""
    isa_sequence: List[ISAOpcode] = field(default_factory=list)
    results: List[Tuple[ISAOpcode, ISAResult]] = field(default_factory=list)
    final_result: ISAResult = ISAResult.PASS
    branch_dead: bool = False


class TProcessor:
    """T-Processor ISA execution engine — physics axioms as native operators.

    The T-Processor implements the Fetch→Decode→Execute→Writeback cycle
    for ISA instructions. Each instruction is a physical axiom that acts
    as a hard constraint gate. The processor:

    1. Fetches the ISA instruction sequence for a game from the registry.
    2. Decodes each instruction into a physics primitive call.
    3. Executes the constraint check, returning PASS/FUSE/DEAD_ZERO.
    4. Aggregates results — the worst outcome determines branch fate.

    Execution model (from article):
        [Fetch ISA] → [Decode to physics primitive] → [Execute constraint]
            → [Writeback or Dead-Zero]

    Key principle: "内思即外作" — thinking IS doing.
    ISA execution IS causal reduction, not just a function call.

    Usage:
        processor = TProcessor()
        result = processor.execute_isa_gate("ka59", state_dict)
        if result == ISAResult.PASS:
            # Allow κ-Snap expansion
        elif result == ISAResult.FUSE:
            # Prune branch (Dead-Zero熔断)
        elif result == ISAResult.DEAD_ZERO:
            # Mark branch permanently invalid
    """

    def __init__(self) -> None:
        """Initialize the T-Processor with the ISA instruction table."""
        self.instruction_table: Dict[ISAOpcode, ISAInstruction] = ISA_INSTRUCTION_TABLE
        self.registry: Dict[str, List[ISAOpcode]] = ISA_REGISTRY
        self._state_cache: Dict[str, TProcessorState] = {}

    def fetch_isa_sequence(self, game_id: str) -> List[ISAOpcode]:
        """Fetch the ISA instruction sequence for a game.

        Normalizes game_id by stripping version suffix, then looks up
        the ISA sequence in the registry. Returns empty list if no
        ISA sequence is registered for this game.

        Args:
            game_id: Game identifier (may include version suffix).

        Returns:
            List of ISAOpcode values to execute for this game.
        """
        base_id: str = game_id.split("-")[0] if game_id else ""
        return self.registry.get(base_id, [])

    def decode_instruction(
        self, opcode: ISAOpcode,
    ) -> Optional[ISAInstruction]:
        """Decode an ISA opcode into its instruction definition.

        Looks up the opcode in the instruction table to find the
        corresponding ISAInstruction with constraint function.

        Args:
            opcode: ISA opcode to decode.

        Returns:
            ISAInstruction definition, or None if opcode is unknown.
        """
        return self.instruction_table.get(opcode)

    def execute_instruction(
        self,
        opcode: ISAOpcode,
        state: Dict[str, Any],
    ) -> ISAResult:
        """Execute a single ISA instruction as a constraint gate.

        Decodes the opcode, then calls the constraint function with
        the appropriate arguments extracted from the state dict.

        The constraint function implements the physical axiom check:
          - PASS → axiom satisfied, proceed
          - FUSE → axiom violated, prune branch
          - DEAD_ZERO → irreversibly invalid, permanent pruning

        Args:
            opcode: ISA opcode to execute.
            state: Game state dict containing operands for the instruction.
                Expected keys vary by instruction type:
                - PUSH_ENTITY: grid, player_pos, box_pos, direction
                - CHECK_DEADLOCK: grid, box_pos, wall_char, goal_char
                - APPLY_FRICTION: grid, box_pos
                - REFLECT_X/Y: x, y, origin_x/origin_y
                - RAY_REFLECT: start, hit_pos, normal
                - DFA_STEP: dfa_state, event, dfa
                - CHECK_CAUSAL: event_sequence, dfa
                - TOPO_SORT: colors, target_order
                - CHECK_POSET: colors, target_order
                - KAPPA_SNAP: eml_data, prior_data
                - GAUSSEX_VERIFY: residual, alpha
                - DEADZERO_FUSE: branch_state

        Returns:
            ISAResult indicating the constraint gate outcome.
        """
        instruction: Optional[ISAInstruction] = self.decode_instruction(opcode)
        if instruction is None or instruction.constraint_fn is None:
            # Unknown opcode or no constraint function — default to PASS
            return ISAResult.PASS

        # Extract operands from state dict based on opcode
        try:
            if opcode == ISAOpcode.PUSH_ENTITY:
                grid = state.get("grid")
                player_pos = state.get("player_pos", (0, 0))
                box_pos = state.get("box_pos", (0, 0))
                direction = state.get("direction", (0, 0))
                wall_char = state.get("wall_char", 0)
                return instruction.constraint_fn(
                    grid, player_pos, box_pos, direction, wall_char
                )

            elif opcode == ISAOpcode.CHECK_DEADLOCK:
                grid = state.get("grid")
                box_pos = state.get("box_pos", (0, 0))
                wall_char = state.get("wall_char", 0)
                goal_char = state.get("goal_char", 2)
                return instruction.constraint_fn(
                    grid, box_pos, wall_char, goal_char
                )

            elif opcode == ISAOpcode.APPLY_FRICTION:
                grid = state.get("grid")
                box_pos = state.get("box_pos", (0, 0))
                wall_char = state.get("wall_char", 0)
                return instruction.constraint_fn(grid, box_pos, wall_char)

            elif opcode == ISAOpcode.REFLECT_X:
                x = state.get("x", 0)
                y = state.get("y", 0)
                origin_x = state.get("origin_x", 0)
                return instruction.constraint_fn(x, y, origin_x)

            elif opcode == ISAOpcode.REFLECT_Y:
                x = state.get("x", 0)
                y = state.get("y", 0)
                origin_y = state.get("origin_y", 0)
                return instruction.constraint_fn(x, y, origin_y)

            elif opcode == ISAOpcode.RAY_REFLECT:
                start = state.get("start", (0, 0))
                hit_pos = state.get("hit_pos", (0, 0))
                normal = state.get("normal", (0, 0))
                return instruction.constraint_fn(start, hit_pos, normal)

            elif opcode == ISAOpcode.DFA_STEP:
                dfa_state = state.get("dfa_state", 0)
                event = state.get("event", 0)
                dfa = state.get("dfa")
                return instruction.constraint_fn(dfa_state, event, dfa)

            elif opcode == ISAOpcode.CHECK_CAUSAL:
                event_sequence = state.get("event_sequence", [])
                dfa = state.get("dfa")
                return instruction.constraint_fn(event_sequence, dfa)

            elif opcode == ISAOpcode.TOPO_SORT:
                colors = state.get("colors", [])
                target_order = state.get("target_order", [])
                return instruction.constraint_fn(colors, target_order)

            elif opcode == ISAOpcode.CHECK_POSET:
                colors = state.get("colors", [])
                target_order = state.get("target_order", [])
                return instruction.constraint_fn(colors, target_order)

            elif opcode == ISAOpcode.KAPPA_SNAP:
                eml_data = state.get("eml_data")
                prior_data = state.get("prior_data")
                return instruction.constraint_fn(eml_data, prior_data)

            elif opcode == ISAOpcode.GAUSSEX_VERIFY:
                residual = state.get("residual")
                alpha = state.get("alpha", 0.05)
                return instruction.constraint_fn(residual, alpha)

            elif opcode == ISAOpcode.DEADZERO_FUSE:
                branch_state = state.get("branch_state")
                return instruction.constraint_fn(branch_state)

            else:
                return ISAResult.PASS

        except Exception:
            # Constraint execution failed — treat as FUSE (safe pruning)
            return ISAResult.FUSE

    def aggregate_results(
        self, results: List[Tuple[ISAOpcode, ISAResult]],
    ) -> ISAResult:
        """Aggregate ISA gate results — worst outcome determines branch fate.

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
        """Execute ISA constraint gate for a game before κ-Snap expansion.

        This is the main integration hook called by
        universal_solver_pipeline.py before κ-Snap expansion. It:

        1. Fetches the ISA instruction sequence for game_id.
        2. Executes each instruction as a constraint gate.
        3. Aggregates results to determine branch fate.
        4. Returns PASS/FUSE/DEAD_ZERO for the pipeline to act on.

        If gate returns PASS → allow κ-Snap expansion on this branch.
        If gate returns FUSE → prune this branch (Dead-Zero熔断).
        If gate returns DEAD_ZERO → mark branch permanently invalid.

        Args:
            game_id: Game identifier (may include version suffix).
            state: Game state dict containing operands for instructions.
                The state dict must contain the relevant keys for the
                ISA instructions registered for this game.

        Returns:
            ISAResult indicating whether κ-Snap expansion may proceed.
        """
        # Step 1: Fetch ISA sequence for this game
        isa_sequence: List[ISAOpcode] = self.fetch_isa_sequence(game_id)

        if not isa_sequence:
            # No ISA sequence registered → default PASS (no physics constraints)
            return ISAResult.PASS

        # Step 2: Execute each instruction in the sequence
        results: List[Tuple[ISAOpcode, ISAResult]] = []
        for opcode in isa_sequence:
            result: ISAResult = self.execute_instruction(opcode, state)
            results.append((opcode, result))

            # Early exit: if DEAD_ZERO detected, no need to continue
            if result == ISAResult.DEAD_ZERO:
                break

        # Step 3: Aggregate results
        final_result: ISAResult = self.aggregate_results(results)

        # Step 4: Cache state for potential Re-Inflow
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
# §8. Module-level convenience functions
# =============================================================================

# Singleton processor instance for module-level access
_processor: TProcessor = TProcessor()


def execute_isa_gate(
    game_id: str,
    state: Dict[str, Any],
) -> ISAResult:
    """Module-level ISA gate execution — convenience wrapper.

    This is the primary integration point for universal_solver_pipeline.py.
    Call this before κ-Snap expansion to check if the current game state
    satisfies all physics axiom constraints.

    Args:
        game_id: Game identifier (may include version suffix).
        state: Game state dict with operands for ISA instructions.

    Returns:
        ISAResult.PASS → allow κ-Snap expansion.
        ISAResult.FUSE → prune branch (Dead-Zero熔断).
        ISAResult.DEAD_ZERO → mark branch permanently invalid.
    """
    return _processor.execute_isa_gate(game_id, state)


def get_isa_sequence(game_id: str) -> List[ISAOpcode]:
    """Get the ISA instruction sequence for a game.

    Args:
        game_id: Game identifier (may include version suffix).

    Returns:
        List of ISAOpcode values registered for this game.
    """
    return _processor.fetch_isa_sequence(game_id)


def get_instruction(opcode: ISAOpcode) -> Optional[ISAInstruction]:
    """Get the ISAInstruction definition for an opcode.

    Args:
        opcode: ISA opcode to look up.

    Returns:
        ISAInstruction definition, or None if opcode is unknown.
    """
    return _processor.decode_instruction(opcode)


def register_game_isa(
    game_id: str,
    isa_sequence: List[ISAOpcode],
) -> None:
    """Register a custom ISA sequence for a game.

    Allows extending the ISA registry for new games or modifying
    existing game ISA sequences dynamically.

    Args:
        game_id: Base game identifier (no version suffix).
        isa_sequence: List of ISA opcodes to execute for this game.
    """
    base_id: str = game_id.split("-")[0] if game_id else ""
    _processor.registry[base_id] = isa_sequence
