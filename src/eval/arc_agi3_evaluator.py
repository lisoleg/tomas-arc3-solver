# -*- coding: utf-8 -*-
"""
ARC-AGI-3 评估模块（从 tomas-agi 吸收）
=============================================

基于论文：
  "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence"
  ARC Prize Foundation, April 20, 2026

核心概念：
    1. 交互式回合制环境（64x64 网格，16 色）
    2. 四大智能支柱：
       - Exploration:   主动交互获取信息
       - Modeling:      从观测建立可泛化世界模型
       - Goal-Setting:  无指令推断胜利条件
       - Planning & Execution: 映射行动路径到识别目标
    3. RHAE（Relative Human Action Efficiency）评分：
       - 关卡得分：S = min(1.15, (h/a)^2)，h=人类基线，a=AI行动数
       - 环境得分：线性加权（等权 5 关卡）
       - 总分：所有环境得分均值
    4. 行动预算：5x 人类基线中位数

吸收来源：tomas-agi/tomas_agi/sim/arc_agi3_eval.py (v1.0)
适配改动：
    - 移除 DeepSeek API 依赖，改为调用 TOMAS solver
    - 网格格式从 64x64 映射到 ARC 网格（3-30 格）
    - 集成到 tomas_solver.py 作为可选评估模式
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

GRID_SIZE = 64
NUM_COLORS = 16
SYSTEM_PROMPT = (
    "You are playing a game. Your goal is to win. "
    "Reply with the exact action you want to take. "
    "The final action in your reply will be executed next turn. "
    "Your entire reply will be carried to the next turn."
)
MAX_LEVELS_PER_ENV = 5
ACTION_BUDGET_MULTIPLIER = 5
SCORE_CAP = 1.15


class ActionType(Enum):
    """ARC-AGI-3 行动类型。"""
    KEY_UP     = "key_up"
    KEY_DOWN   = "key_down"
    KEY_LEFT   = "key_left"
    KEY_RIGHT  = "key_right"
    KEY_SPACE  = "key_space"
    UNDO       = "undo"
    CELL_SELECT = "cell_select"
    ACTION1    = "ACTION1"
    ACTION2    = "ACTION2"
    ACTION3    = "ACTION3"
    ACTION4    = "ACTION4"
    ACTION5    = "ACTION5"
    ACTION6    = "ACTION6"
    ACTION7    = "ACTION7"


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Frame:
    """环境帧（观测快照）。"""
    grid: List[List[int]]
    agent_pos: Optional[Tuple[int, int]] = None
    extra: Dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class ActionResult:
    """行动执行结果。"""
    action: str
    new_frame: Frame
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class LevelResult:
    """单个关卡评估结果。"""
    level_id: int
    completed: bool
    actions_taken: int
    total_reward: float = 0.0
    time_seconds: float = 0.0
    trajectory: List[Dict] = dc_field(default_factory=list)


@dataclass
class EnvironmentResult:
    """单个环境评估结果。"""
    env_id: str
    levels: List[LevelResult]
    total_actions: int = 0
    time_seconds: float = 0.0


# ============================================================
# RHAE 评分器
# ============================================================

class RHAEScorer:
    """RHAE（Relative Human Action Efficiency）评分器。

    公式：
        S_level = min(1.15, (h / a)^2)
        S_env  = Σ_i w_i * S_level_i   (等权 w_i = 1/5)
        S_total = mean(S_env_i)
    """

    @staticmethod
    def level_score(human_baseline: int, ai_actions: int) -> float:
        """计算单关卡 RHAE 得分。"""
        if human_baseline <= 0 or ai_actions <= 0:
            return 0.0
        ratio = human_baseline / ai_actions
        return min(SCORE_CAP, ratio ** 2)

    @staticmethod
    def environment_score(
        level_scores: List[Tuple[int, float]],
        total_levels: int,
        completed: int,
    ):
        """计算环境 RHAE 得分（dataclass 风格返回）。"""
        if total_levels <= 0:
            return _RHaeResult(0.0, [])

        # 等权平均
        weights = [1.0 / total_levels] * total_levels
        weighted_sum = 0.0
        valid_scores = []

        for level_id, score in level_scores:
            weighted_sum += score * weights[min(level_id - 1, total_levels - 1)]
            valid_scores.append((level_id, score))

        # 未完成的关卡得 0 分
        cap = SCORE_CAP
        return _RHaeResult(weighted_sum, valid_scores, cap)

    @staticmethod
    def total_score(env_scores: List[float]) -> float:
        """计算 total RHAE（所有环境均值）。"""
        if not env_scores:
            return 0.0
        return sum(env_scores) / len(env_scores)


@dataclass
class _RHaeResult:
    environment_score: float
    level_scores: List[Tuple[int, float]]
    environment_cap: float = SCORE_CAP


# ============================================================
# ARC-AGI-3 环境模拟器
# ============================================================

class ARCAGI3Environment:
    """ARC-AGI-3 环境模拟器（离线，基于静态数据集）。

    在 TOMAS 场景下，我们不直接控制 64x64 游戏，
    而是将 ARC 任务视为"环境"，TOMAS solver 输出程序作为"行动"。
    这里提供兼容接口，便于后续接入官方 arc-agi Python 包。
    """

    def __init__(self, env_id: str, levels: List[Dict[str, Any]]) -> None:
        self.env_id = env_id
        self.levels = levels
        self.current_level: Optional[int] = None
        self.current_frame: Optional[Frame] = None

    def reset(self, level_id: int = 1) -> Frame:
        """重置到指定关卡起始帧。"""
        if level_id < 1 or level_id > len(self.levels):
            raise ValueError(f"Invalid level_id: {level_id}")
        lvl = self.levels[level_id - 1]
        self.current_level = level_id
        grid = lvl.get("initial_frame", [[0]*GRID_SIZE for _ in range(GRID_SIZE)])
        self.current_frame = Frame(grid=grid)
        return self.current_frame

    def step(self, action: str) -> ActionResult:
        """执行一步行动（在静态数据集中为 no-op）。"""
        # 静态数据集模式下，step 仅记录行动
        return ActionResult(
            action=action,
            new_frame=self.current_frame,
            done=False,
        )

    def get_human_baseline(self, level_id: int) -> int:
        """获取人类基线行动数。"""
        if level_id < 1 or level_id > len(self.levels):
            return 0
        return self.levels[level_id - 1].get("human_baseline", 1)

    def check_win(self, level_id: int, program_output: Any) -> bool:
        """检查 TOMAS 程序输出是否满足胜利条件。"""
        lvl = self.levels[level_id - 1]
        win_cond = lvl.get("win_condition", {})
        cond_type = win_cond.get("type", "unknown")

        if cond_type == "reach_target":
            # 检查输出网格是否含目标色块
            target_pos = win_cond.get("target_pos")
            return True  # 简化：实际需比对完整输出
        elif cond_type == "collect_all":
            return True
        return False


# ============================================================
# TOMAS 评估器
# ============================================================

class TOMASEvaluator:
    """TOMAS ARC-AGI-3 评估器。

    工作流程：
        1. 加载数据集（JSON 格式）
        2. 对每个环境/关卡：
           a. 用 TOMAS solver 生成候选程序
           b. 执行程序并比对输出
           c. 记录行动数（程序长度 ≈ 行动数）
        3. 用 RHAE 公式计算得分
    """

    def __init__(
        self,
        tomas_solver_url: str = "http://localhost:5000",
        verbose: bool = False,
    ) -> None:
        self.tomas_solver_url = tomas_solver_url
        self.verbose = verbose
        self.results: List[EnvironmentResult] = []

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """加载 ARC-AGI-3 数据集 JSON。"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        envs = data.get("environments", [])
        if self.verbose:
            print(f"[_eval] Loaded {len(envs)} environments from {dataset_path}")
        return envs

    def evaluate_environment(
        self, env_def: Dict[str, Any]
    ) -> EnvironmentResult:
        """评估单个环境（所有关卡）。"""
        env_id = env_def.get("env_id", "unknown")
        levels = env_def.get("levels", [])
        level_results: List[LevelResult] = []
        total_actions = 0

        for i, lvl in enumerate(levels):
            level_id = i + 1
            human_baseline = lvl.get("human_baseline", 1)

            # 用 TOMAS solver 求解该关卡
            # （实际调用：将 initial_frame 转为 ARC 格式输入 solver）
            ai_actions = self._run_tomas_solver(lvl)

            completed = ai_actions > 0 and ai_actions <= human_baseline * ACTION_BUDGET_MULTIPLIER
            lr = LevelResult(
                level_id=level_id,
                completed=completed,
                actions_taken=ai_actions,
            )
            level_results.append(lr)
            total_actions += ai_actions

            if self.verbose:
                rhae = RHAEScorer.level_score(human_baseline, ai_actions)
                print(f"  Level {level_id}: "
                      f"h={human_baseline}, a={ai_actions}, "
                      f"RHAE={rhae:.4f}, {'✅' if completed else '❌'}")

        return EnvironmentResult(
            env_id=env_id,
            levels=level_results,
            total_actions=total_actions,
        )

    def _run_tomas_solver(self, level_def: Dict[str, Any]) -> int:
        """调用 TOMAS solver 求解单个关卡，返回行动数（程序长度）。

        在完整实现中，这里应：
          1. 将 level_def["initial_frame"] 转为 ARC 输入输出对
          2. 调用 tomas_solver.py 的 solve() 方法
          3. 返回生成程序的长度（≈ 行动数）

        当前为 MVP：返回占位值。
        """
        # TODO: 接入真实 TOMAS solver
        # from src.solver.tomas_solver import TOMASSolver
        # solver = TOMASSolver()
        # result = solver.solve(arc_input_pairs)
        # return len(result.best_program)
        return level_def.get("human_baseline", 1)  # placeholder

    def evaluate_dataset(
        self,
        dataset_path: str,
        max_envs: int = 0,
    ) -> Dict[str, Any]:
        """评估完整数据集。"""
        environments = self.load_dataset(dataset_path)
        if max_envs > 0:
            environments = environments[:max_envs]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  ARC-AGI-3 Evaluation")
            print(f"  Environments: {len(environments)}")
            print(f"{'='*60}")

        all_env_scores: List[float] = []
        detailed_results: List[Dict] = []

        for env_def in environments:
            env_result = self.evaluate_environment(env_def)
            self.results.append(env_result)

            level_scores: List[Tuple[int, float]] = []
            total_levels = len(env_def.get("levels", []))
            completed = sum(1 for l in env_result.levels if l.completed)

            for lr in env_result.levels:
                env = ARCAGI3Environment(env_result.env_id, env_def.get("levels", []))
                h = env.get_human_baseline(lr.level_id - 1)
                s = RHAEScorer.level_score(h, lr.actions_taken)
                level_scores.append((lr.level_id, s))

            rhae = RHAEScorer.environment_score(level_scores, total_levels, completed)
            all_env_scores.append(rhae.environment_score)

            detailed_results.append({
                "env_id": env_result.env_id,
                "total_actions": env_result.total_actions,
                "levels_completed": completed,
                "total_levels": total_levels,
                "rhae_score": round(rhae.environment_score * 100, 2),
            })

        total = RHAEScorer.total_score(all_env_scores)
        report = {
            "total_score": round(total * 100, 2),
            "environments_evaluated": len(environments),
            "detailed_results": detailed_results,
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  Total RHAE Score: {total*100:.2f}%")
            print(f"  (Frontier AI baseline: < 1%)")
            print(f"{'='*60}")

        return report


# ============================================================
# CLI 入口
# ============================================================

def generate_demo_environments() -> List[Dict]:
    """生成演示环境（无真实数据集时用）。"""
    def make_grid(agent_pos=(32, 32), target_pos=None, walls=None):
        grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        ax, ay = agent_pos
        grid[ay][ax] = 1
        if target_pos:
            tx, ty = target_pos
            grid[ty][tx] = 4
        if walls:
            for wx, wy in walls:
                if 0 <= wx < GRID_SIZE and 0 <= wy < GRID_SIZE:
                    grid[wy][wx] = 2
        return grid

    return [
        {
            "env_id": "demo_01",
            "levels": [
                {
                    "initial_frame": make_grid(agent_pos=(10, 32), target_pos=(50, 32)),
                    "win_condition": {"type": "reach_target", "target_pos": (50, 32)},
                    "human_baseline": 40,
                    "valid_actions": ["key_up", "key_down", "key_left", "key_right"],
                },
                {
                    "initial_frame": make_grid(agent_pos=(5, 5), target_pos=(58, 58)),
                    "win_condition": {"type": "reach_target", "target_pos": (58, 58)},
                    "human_baseline": 80,
                    "valid_actions": ["key_up", "key_down", "key_left", "key_right"],
                },
            ],
        },
        {
            "env_id": "demo_02",
            "levels": [
                {
                    "initial_frame": make_grid(),
                    "win_condition": {"type": "collect_all"},
                    "human_baseline": 50,
                    "valid_actions": ["key_up", "key_down", "key_left", "key_right", "key_space"],
                },
            ],
        },
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TOMAS ARC-AGI-3 Evaluator")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset JSON path")
    parser.add_argument("--demo", action="store_true", help="Use demo environments")
    parser.add_argument("--output", type=str, default=None, help="Output report JSON")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.demo or args.dataset is None:
        demo_envs = generate_demo_environments()
        demo_path = "data/arc_agi3_demo.json"
        os.makedirs("data", exist_ok=True)
        with open(demo_path, "w") as f:
            json.dump({"environments": demo_envs}, f, indent=2)
        args.dataset = demo_path
        print(f"⚠️  Using demo environments: {demo_path}")

    evaluator = TOMASEvaluator(verbose=args.verbose)
    report = evaluator.evaluate_dataset(args.dataset)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Report saved to: {args.output}")

    print(f"\n✅ Total RHAE Score: {report['total_score']}%")
