"""TOMAS-ARC3 RHAE Budget Controller (v3.15.0)

RHAE = (H/A)² 导向的算力分配策略
CoinCollector: 金币收集 = 保全局平均分，不是刷单题神话
RHAEBudgetController: 实时监控(H/A)²，超预算/过低→放弃
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple

# 从 delta_state.py 导入 RHAE 相关常量
# TOMAS语义：流贯阈值 — Δ-State拓扑的边界条件
from .delta_state import (
    MAX_RHAE_PER_TASK,
    LS20_BUDGET_MULT,
    MIN_RHAE_TO_KEEP,
    ABORT_RHAE_THRESHOLD,
)


class CoinCollector:
    """LS20 金币收集策略调度器

    TOMAS 语义：流贯在多任务空间的最优算力分配

    核心哲学：RHAE = (H/A)² — 不是追求单题满分，而是最大化全局 mean RHAE
    "金币" = RHAE分 — "收集金币" = 优先保证每题拿到尽可能高的 RHAE
    一道题卡住 → 及时止损 → 算力留给下一题

    三件事:
    ① 步数预算控制 (MAX_STEPS_PER_TASK = H * budget_mult)
    ② 早期退出 (Early-Exit: GEX残差太大→直接放弃)
    ③ 多任务优先 (保平均分，不是刷单题神话)

    Attributes:
        tasks: 待处理的任务列表
        is_ls20: 是否为LS20游戏模式
        budget_mult: 步数预算乘数因子
        results: 各任务的RHAE结果字典
        _task_idx: 当前任务索引指针
    """

    def __init__(
        self,
        tasks: List[Dict],
        is_ls20: bool = False,
        budget_multiplier: float = LS20_BUDGET_MULT,
    ) -> None:
        """初始化金币收集调度器。

        Args:
            tasks: 任务列表，每个任务为包含 task_id、human_steps 等信息的字典
            is_ls20: 是否为LS20游戏模式（LS20额外20%预算）
            budget_multiplier: Agent步数相对于人类步数的预算乘数
        """
        self.tasks: List[Dict] = tasks
        self.is_ls20: bool = is_ls20
        self.budget_mult: float = budget_multiplier
        self.results: Dict[int, Dict] = {}
        self._task_idx: int = 0

    def get_budget(self, task_id: int, human_steps: int) -> int:
        """根据人类步数 H 计算 Agent 步数预算 A。

        TOMAS语义：流贯预算 = H × 算力乘数，LS20额外增加20%容错空间

        Args:
            task_id: 任务标识符
            human_steps: 人类参考步数 H

        Returns:
            Agent步数预算上限（整数）
        """
        base: float = human_steps * self.budget_mult
        if self.is_ls20:
            # LS20 额外 20% — 流贯容错空间，允许更多探索路径
            base = int(base * 1.2)
        return int(base)

    def record_result(
        self,
        task_id: int,
        human_steps: int,
        agent_steps: int,
        solved: bool,
        extra: Optional[Dict] = None,
    ) -> None:
        """记录单题结果。

        RHAE = (H/A)² — 解决则计分，未解决则RHAE=0
        上限截断于 MAX_RHAE_PER_TASK，防止单题膨胀

        Args:
            task_id: 任务标识符
            human_steps: 人类参考步数 H
            agent_steps: Agent实际使用步数 A
            solved: 是否成功解决该题
            extra: 额外信息字典（如GEX残差、搜索深度等）
        """
        if not solved:
            rhae: float = 0.0
        else:
            rhae = min(
                (human_steps / max(agent_steps, 1)) ** 2,
                MAX_RHAE_PER_TASK,
            )
        self.results[task_id] = {
            "H": human_steps,
            "A": agent_steps,
            "rhae": rhae,
            "solved": solved,
        }
        if extra:
            self.results[task_id].update(extra)

    def get_global_rhae(self) -> float:
        """计算全局平均 RHAE。

        mean RHAE = Σ rhae_i / N — 金币收集的核心指标
        不是max(RHAE)，而是mean(RHAE) — 保全局而非刷单题

        Returns:
            全局平均RHAE值（0.0若无结果）
        """
        if not self.results:
            return 0.0
        total: float = sum(r["rhae"] for r in self.results.values())
        return total / len(self.results)

    def get_solved_count(self) -> int:
        """获取已解决任务数量。

        Returns:
            成功解决的任务数
        """
        return sum(1 for r in self.results.values() if r["solved"])

    def should_abort(
        self, task_id: int, human_steps: int, current_agent_steps: int
    ) -> bool:
        """判断是否应放弃当前任务。

        早期退出策略：当潜在RHAE低于阈值时，及时止损
        TOMAS语义：流贯低于Δ-State边界 → 切断该分支，算力流向下一任务

        Args:
            task_id: 任务标识符
            human_steps: 人类参考步数 H
            current_agent_steps: 当前已消耗的Agent步数

        Returns:
            True表示应放弃当前任务，False表示可继续
        """
        if current_agent_steps <= 0:
            return False
        potential_rhae: float = (human_steps / current_agent_steps) ** 2
        # 潜在RHAE低于阈值 → 止损退出
        if potential_rhae < ABORT_RHAE_THRESHOLD:
            return True
        return False

    def summary(self) -> str:
        """生成调度器执行摘要字符串。

        Returns:
            包含总任务数、解决数、解决率、全局RHAE的格式化字符串
        """
        total: int = len(self.results)
        solved: int = self.get_solved_count()
        rhae: float = self.get_global_rhae()
        return (
            f"[CoinCollector] Tasks: {total} | "
            f"Solved: {solved} ({solved / max(total, 1) * 100:.1f}%) | "
            f"Global RHAE: {rhae:.4f}"
        )

    def export_results(self, path: str) -> None:
        """导出结果到 JSON 文件。

        Args:
            path: JSON文件输出路径
        """
        data: Dict[str, Any] = {
            "global_rhae": self.get_global_rhae(),
            "solved": self.get_solved_count(),
            "total": len(self.results),
            "is_ls20": self.is_ls20,
            "details": self.results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


class RHAEBudgetController:
    """RHAE 预算控制器

    实时监控每题的 (H, A) 对
    超过预算 → 触发 SolverAborted
    记录详细日志供分析

    TOMAS语义：Δ-State流贯的实时守卫 — 越界即停

    Attributes:
        cc: 关联的CoinCollector实例，负责结果记录与统计
        min_rhae: 最低可接受RHAE阈值
        abort_threshold: 早期退出的RHAE阈值
        log: 执行日志列表，记录每个关键事件
    """

    def __init__(
        self,
        coin_collector: CoinCollector,
        min_rhae_to_keep: float = MIN_RHAE_TO_KEEP,
        abort_threshold: float = ABORT_RHAE_THRESHOLD,
    ) -> None:
        """初始化RHAE预算控制器。

        Args:
            coin_collector: CoinCollector实例，用于结果记录和预算计算
            min_rhae_to_keep: 最低可保留RHAE值（低于此值的成功仍记录但标记）
            abort_threshold: 早期退出阈值——潜在RHAE低于此值时放弃
        """
        self.cc: CoinCollector = coin_collector
        self.min_rhae: float = min_rhae_to_keep
        self.abort_threshold: float = abort_threshold
        self.log: List[Dict] = []

    def check_budget(
        self, task_id: int, human_steps: int, current_steps: int
    ) -> bool:
        """检查是否超出预算，返回是否可继续执行。

        双重守卫：
        ① 步数超限 → 算力耗尽，必须停止
        ② RHAE过低 → 即使步数未满，也应及时止损

        Args:
            task_id: 任务标识符
            human_steps: 人类参考步数 H
            current_steps: 当前已消耗的Agent步数 A

        Returns:
            True = 可以继续，False = 应终止
        """
        budget: int = self.cc.get_budget(task_id, human_steps)

        # 步数超限 — 算力耗尽
        if current_steps >= budget:
            self.log.append(
                {
                    "task": task_id,
                    "event": "budget_exceeded",
                    "steps": current_steps,
                    "budget": budget,
                }
            )
            return False

        # RHAE 过低 → 提前终止（早期退出）
        if current_steps > 0:
            potential: float = (human_steps / current_steps) ** 2
            if potential < self.abort_threshold:
                self.log.append(
                    {
                        "task": task_id,
                        "event": "rhae_too_low",
                        "potential_rhae": potential,
                        "threshold": self.abort_threshold,
                    }
                )
                return False

        return True

    def record_success(
        self, task_id: int, human_steps: int, agent_steps: int
    ) -> None:
        """记录成功解决的任务。

        Args:
            task_id: 任务标识符
            human_steps: 人类参考步数 H
            agent_steps: Agent实际使用步数 A
        """
        self.cc.record_result(task_id, human_steps, agent_steps, solved=True)
        rhae: float = (human_steps / max(agent_steps, 1)) ** 2
        self.log.append(
            {
                "task": task_id,
                "event": "solved",
                "H": human_steps,
                "A": agent_steps,
                "rhae": rhae,
            }
        )

    def record_failure(
        self, task_id: int, human_steps: int, agent_steps: int
    ) -> None:
        """记录未能解决的任务。

        Args:
            task_id: 任务标识符
            human_steps: 人类参考步数 H
            agent_steps: Agent实际使用步数 A
        """
        self.cc.record_result(task_id, human_steps, agent_steps, solved=False)
        self.log.append(
            {
                "task": task_id,
                "event": "unsolved",
                "H": human_steps,
                "A": agent_steps,
            }
        )

    def print_report(self) -> None:
        """打印执行报告到stdout。

        输出格式化的RHAE预算控制器执行报告，包括：
        - 每题结果（✅成功 / ❌失败 / ⚠️超限 / ⚠️RHAE过低）
        - CoinCollector全局统计摘要
        """
        print("\n" + "=" * 60)
        print("  RHAE Budget Controller — 执行报告")
        print("=" * 60)
        for entry in self.log:
            if entry["event"] == "solved":
                print(
                    f"  ✅ Task {entry['task']}: "
                    f"H={entry['H']} A={entry['A']} "
                    f"RHAE={entry['rhae']:.4f}"
                )
            elif entry["event"] == "unsolved":
                print(
                    f"  ❌ Task {entry['task']}: "
                    f"H={entry['H']} A={entry['A']} "
                    f"RHAE=0.0000"
                )
            elif entry["event"] == "budget_exceeded":
                print(
                    f"  ⚠️ Task {entry['task']}: "
                    f"Budget exceeded ({entry['steps']}/{entry['budget']} steps)"
                )
            elif entry["event"] == "rhae_too_low":
                print(
                    f"  ⚠️ Task {entry['task']}: "
                    f"RHAE too low "
                    f"({entry['potential_rhae']:.4f} < {entry['threshold']})"
                )
        print(f"\n  📊 {self.cc.summary()}")
        print("=" * 60 + "\n")


def create_game_task(
    game_id: str, level_idx: int, human_steps: int
) -> Dict[str, Any]:
    """为游戏创建RHAE追踪任务。

    TOMAS语义：将游戏关卡映射为Δ-State追踪单元
    每个任务携带游戏标识、关卡索引和人类参考步数

    Args:
        game_id: 游戏标识 (e.g., "ls20", "ka59")
        level_idx: 关卡索引（从0开始）
        human_steps: 人类参考步数（如无数据则使用估算值）

    Returns:
        Dict with id, game_id, level_idx, human_steps
    """
    # 基于game_id + level_idx生成稳定哈希ID
    task_hash: int = hash(f"{game_id}_{level_idx}") % 10000
    return {
        "id": task_hash,
        "game_id": game_id,
        "level_idx": level_idx,
        "human_steps": human_steps,
    }


def ls20_estimate_human_steps(level_idx: int) -> int:
    """估算LS20各关卡的人类参考步数。

    基于已知的通关数据:
    - L0: 13步 (已验证)
    - L1: ~25步 (预估, 需金币收集+旋转+导航)
    - L2: 43步 (已验证)
    - L3+: ~30步 (预估)

    TOMAS语义：人类步数H是RHAE计算的基准
    无准确H → RHAE计算失真 → 算力分配偏差

    Args:
        level_idx: LS20关卡索引（从0开始）

    Returns:
        估算的人类参考步数
    """
    # 已验证数据 + 预估数据的混合映射
    estimates: Dict[int, int] = {
        0: 13,   # 已验证
        1: 25,   # 预估: 金币收集+旋转+导航
        2: 43,   # 已验证
        3: 30,   # 预估
        4: 35,   # 预估
        5: 40,   # 预估
        6: 45,   # 预估
    }
    return estimates.get(level_idx, 30)
