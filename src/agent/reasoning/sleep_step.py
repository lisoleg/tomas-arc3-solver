"""SleepStep - L4 观测者的睡前总结机制。

从失败轨迹中提取防御性原语：
1. 误检模式 → 禁忌规则（Forbidden Patterns）
2. 高频动作序列 → 防御性 Macro 原语
3. 空间关系 → 感知修正规则

核心检测逻辑：
- 位置突变（穿墙误检）：连续两步 player_pos 曼哈顿距离 > JUMP_THRESHOLD
- 连续同一方向移动后撞墙：连续 3+ 步同一 action 且最后 moved=False
- 感知修正：当检测位置不可靠时，生成 USE_PREV_POSITION 规则
"""

from __future__ import annotations

from typing import List, Set, Tuple, Optional, Dict, Any
from collections import Counter


class SleepStep:
    """从 EpisodeMemory 的失败/成功轨迹中学习防御性规则。

    SleepStep 是 L4 观测者在每局/每关结束后运行的"睡前总结"模块。
    它分析 EpisodeMemory 中归档的轨迹（failures / successes），提取
    防御性原语以供 HeuristicPlanner 在后续关卡中避开已知的误检和死路。

    Attributes:
        forbidden_positions: 已知误检或危险位置集合 (col, row)。
        forbidden_actions: 穿墙方向动作标记为禁止的 action ID 集合。
        defensive_macros: 防御性宏动作，键为宏名称，值为宏描述。
        perception_corrections: 感知修正规则列表。
        pattern_stats: 检测到的模式频次统计。
    """

    #: 位置突变阈值：MOVE_STEP=5，超过 2 个 MOVE_STEP（即 > 10）算突变。
    JUMP_THRESHOLD: int = 10

    #: 连续同方向移动次数阈值，达到此值且最后未移动 → 穿墙误检。
    CONSECUTIVE_SAME_ACTION_THRESHOLD: int = 3

    #: 键盘动作 ID → 方向名称映射（与 HeuristicPlanner.KEYBOARD_ACTIONS 对齐）。
    ACTION_DIRECTIONS: Dict[int, str] = {
        1: "up",
        2: "down",
        3: "left",
        4: "right",
    }

    def __init__(self) -> None:
        """初始化 SleepStep，所有规则集合为空。"""
        self.forbidden_positions: Set[Tuple[int, int]] = set()
        self.forbidden_actions: Set[int] = set()
        self.defensive_macros: Dict[str, dict] = {}
        self.perception_corrections: List[dict] = []
        self.pattern_stats: Counter = Counter()

    # ------------------------------------------------------------------
    # 单条轨迹分析
    # ------------------------------------------------------------------

    def analyze_failure(self, steps: list, info: str = "") -> dict:
        """分析单条失败轨迹（EpisodeStep 列表）。

        检测三类问题并生成对应防御规则：

        1. **位置突变（穿墙误检）**：连续两步 player_pos 曼哈顿距离
           > ``JUMP_THRESHOLD``。突变位置注册为 forbidden_position，
           并生成 perception_correction 规则。

        2. **连续同方向撞墙**：连续 ``CONSECUTIVE_SAME_ACTION_THRESHOLD``
           步执行同一 action，且最后一步 ``moved=False``。该 action
           注册为 forbidden_action。

        3. **感知修正**：当 info 包含 "WALL" 或 "MISDETECT" 时，
           或检测到位置突变时，生成 ``USE_PREV_POSITION`` 规则。

        Args:
            steps: EpisodeStep 列表（来自 EpisodeMemory）。
            info: 失败原因描述（如 "GAME_OVER"、"WALL_MISDETECT" 等）。

        Returns:
            包含本次分析提取规则的字典::
                {
                    "forbidden": List[Tuple[int, int]],   # 新增禁止位置
                    "forbidden_actions": List[int],         # 新增禁止动作
                    "macro": Optional[dict],                # 新增防御性宏
                    "perception_fix": Optional[dict],       # 新增感知修正
                }
        """
        forbidden_new: List[Tuple[int, int]] = []
        forbidden_actions_new: List[int] = []
        macro_new: Optional[dict] = None
        perception_fix_new: Optional[dict] = None

        if not steps or len(steps) < 2:
            return {
                "forbidden": forbidden_new,
                "forbidden_actions": forbidden_actions_new,
                "macro": macro_new,
                "perception_fix": perception_fix_new,
            }

        # --- 检测 1: 位置突变（穿墙误检） ---
        prev_pos: Optional[Tuple[int, int]] = None
        for i, step in enumerate(steps):
            curr_pos = getattr(step, "player_pos", None)
            if curr_pos is None or prev_pos is None:
                prev_pos = curr_pos
                continue

            jump_dist = abs(curr_pos[0] - prev_pos[0]) + abs(
                curr_pos[1] - prev_pos[1]
            )
            if jump_dist > self.JUMP_THRESHOLD:
                # 突变位置注册为禁止位置
                pos_tuple = (int(curr_pos[0]), int(curr_pos[1]))
                if pos_tuple not in self.forbidden_positions:
                    self.forbidden_positions.add(pos_tuple)
                    forbidden_new.append(pos_tuple)

                # 生成感知修正规则
                perception_fix_new = {
                    "condition": f"position_jump > {self.JUMP_THRESHOLD}",
                    "action": "USE_PREV_POSITION",
                    "detected_at_step": getattr(step, "step", i),
                    "jump_distance": jump_dist,
                }
                self.perception_corrections.append(perception_fix_new)
                self.pattern_stats["position_jump"] += 1

            prev_pos = curr_pos

        # --- 检测 2: 连续同方向移动后撞墙 ---
        consecutive_action: Optional[int] = None
        consecutive_count: int = 0
        for i, step in enumerate(steps):
            action = getattr(step, "action", None)
            moved = getattr(step, "moved", True)

            if action == consecutive_action:
                consecutive_count += 1
            else:
                consecutive_action = action
                consecutive_count = 1

            # 达到阈值且最后一步未移动 → 穿墙动作
            if (
                consecutive_count >= self.CONSECUTIVE_SAME_ACTION_THRESHOLD
                and not moved
                and action is not None
                and action in self.ACTION_DIRECTIONS
            ):
                if action not in self.forbidden_actions:
                    self.forbidden_actions.add(action)
                    forbidden_actions_new.append(action)

                    # 生成防御性宏：记录该方向为禁止
                    macro_name = f"avoid_{self.ACTION_DIRECTIONS[action]}"
                    macro_new = {
                        "name": macro_name,
                        "pattern": [],
                        "forbidden_action": action,
                        "direction": self.ACTION_DIRECTIONS[action],
                        "detected_at_step": getattr(step, "step", i),
                    }
                    self.defensive_macros[macro_name] = macro_new

                self.pattern_stats["wall_collision"] += 1
                # 重置连续计数避免重复触发
                consecutive_count = 0

        # --- 检测 3: 基于 info 的感知修正 ---
        info_upper = info.upper() if info else ""
        if ("WALL" in info_upper or "MISDETECT" in info_upper) and perception_fix_new is None:
            perception_fix_new = {
                "condition": "info_contains_wall_or_misdetect",
                "action": "USE_PREV_POSITION",
                "info": info,
            }
            self.perception_corrections.append(perception_fix_new)
            self.pattern_stats["info_triggered_fix"] += 1

        return {
            "forbidden": forbidden_new,
            "forbidden_actions": forbidden_actions_new,
            "macro": macro_new,
            "perception_fix": perception_fix_new,
        }

    # ------------------------------------------------------------------
    # 从 EpisodeMemory 批量学习
    # ------------------------------------------------------------------

    def learn_from_episode(self, memory) -> dict:
        """从 EpisodeMemory 学习所有规则。

        遍历 ``memory.failures`` 和 ``memory.successes`` 中的每条轨迹，
        调用 :meth:`analyze_failure` 提取防御规则。失败轨迹用于提取
        禁忌规则，成功轨迹用于验证（不产生禁止规则，但更新统计）。

        Args:
            memory: EpisodeMemory 实例，需具有 ``failures`` 和
                ``successes`` 属性（均为 ``list[list[EpisodeStep]]``）。

        Returns:
            汇总规则字典::
                {
                    "forbidden_positions": List[Tuple[int, int]],
                    "forbidden_actions": List[int],
                    "macros": List[dict],
                    "perception_fixes": List[dict],
                    "total_trajectories_analyzed": int,
                }
        """
        total_analyzed: int = 0

        # 分析失败轨迹
        failures: list = getattr(memory, "failures", [])
        for traj in failures:
            if not traj:
                continue
            # 推断失败原因：取最后一步的游戏状态
            info = "GAME_OVER"
            self.analyze_failure(traj, info=info)
            total_analyzed += 1

        # 分析成功轨迹（仅更新统计，不产生禁止规则）
        successes: list = getattr(memory, "successes", [])
        for traj in successes:
            if not traj:
                continue
            # 成功轨迹中也可能有位置突变（误检），但不算失败
            self.analyze_failure(traj, info="LEVEL_COMPLETE")
            total_analyzed += 1

        return {
            "forbidden_positions": list(self.forbidden_positions),
            "forbidden_actions": list(self.forbidden_actions),
            "macros": list(self.defensive_macros.values()),
            "perception_fixes": list(self.perception_corrections),
            "total_trajectories_analyzed": total_analyzed,
        }

    # ------------------------------------------------------------------
    # 规则查询
    # ------------------------------------------------------------------

    def get_rules(self) -> dict:
        """返回当前所有防御规则。

        Returns:
            包含所有当前防御规则的字典::
                {
                    "forbidden_positions": List[Tuple[int, int]],
                    "forbidden_actions": List[int],
                    "defensive_macros": Dict[str, dict],
                    "perception_corrections": List[dict],
                    "pattern_stats": Dict[str, int],
                }
        """
        return {
            "forbidden_positions": list(self.forbidden_positions),
            "forbidden_actions": list(self.forbidden_actions),
            "defensive_macros": dict(self.defensive_macros),
            "perception_corrections": list(self.perception_corrections),
            "pattern_stats": dict(self.pattern_stats),
        }

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置所有规则，清空所有集合和统计。"""
        self.forbidden_positions.clear()
        self.forbidden_actions.clear()
        self.defensive_macros.clear()
        self.perception_corrections.clear()
        self.pattern_stats.clear()
