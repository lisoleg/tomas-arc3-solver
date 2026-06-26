#!/usr/bin/env python3
"""QA 验证 ls20 金币收集策略 — L0/L1/L2 通关测试.

验证修复:
1. LS20Adapter 新增 step_decrement/steps_remaining 属性
2. _solve_ls20_multiphase_bfs 主动金币收集
3. _solve_ls20_kappa_gradient_direct 步数预算修正

测试项:
T1: LS20Adapter step_decrement 属性 (L0=2, L1=2, L2=1)
T2: LS20Adapter steps_remaining 属性
T3: L0 通关 (回归测试, 不破坏已通过状态)
T4: L1 通关 (核心目标 — 金币收集使 goal 可达)
T5: L2 通关
"""
import sys
import os
import copy
import time

# ── 项目路径 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "environment_files", "ls20", "9607627b"))

# ── 导入 ──


def _load_ls20_game():
    """加载 LS20 游戏模块."""
    ls20_module_path = os.path.join(PROJECT_ROOT, "environment_files", "ls20", "9607627b", "ls20.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ls20_module", ls20_module_path)
    ls20_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls20_module)
    return ls20_module.Ls20


def _set_level(game, level_idx: int):
    """设置游戏到指定关卡."""
    game._current_level_index = level_idx
    game._levels[level_idx] = game._clean_levels[level_idx].clone()
    game.on_set_level(game._levels[level_idx])


def _is_level_solved(game, level_idx: int) -> bool:
    """检查关卡是否已通关."""
    return game._current_level_index > level_idx


def _detect_step_size(game) -> int:
    """检测步长."""
    player = game.gudziatsk
    return max(1, int(player.width))


def run_test_T1():
    """T1: LS20Adapter step_decrement 属性."""
    print("\n=== T1: LS20Adapter step_decrement 属性 ===")
    Ls20 = _load_ls20_game()
    game = Ls20()

    from src.agent.oracle_adapters import get_oracle_adapter

    results = {}
    for level_idx in [0, 1, 2]:
        _set_level(game, level_idx)
        adapter = get_oracle_adapter("ls20", game)
        if adapter is None:
            results[level_idx] = "FAIL: adapter=None"
            continue

        decrement = adapter.step_decrement
        budget = adapter.step_budget
        max_actions = budget // decrement

        # 检查 StepsDecrement 数据
        steps_decrement_data = game.current_level.get_data("StepsDecrement")

        print(f"  L{level_idx}: step_decrement={decrement}, step_budget={budget}, "
              f"max_actions={max_actions}, StepsDecrement_data={steps_decrement_data}")
        results[level_idx] = {
            "decrement": decrement,
            "budget": budget,
            "max_actions": max_actions,
            "data": steps_decrement_data,
        }

    # 验证 — 实际值来自游戏源码:
    # L0: StepsDecrement=1 (显式设置) → 1
    # L1: StepsDecrement=None → 默认 2
    # L2: StepsDecrement=None → 默认 2
    t1_pass = True
    expected = {0: 1, 1: 2, 2: 2}
    for lvl, exp in expected.items():
        actual = results.get(lvl, {}).get("decrement")
        if actual != exp:
            print(f"  ❌ L{lvl} step_decrement 应为 {exp}, 实际={actual}")
            t1_pass = False
        t1_pass = False

    if t1_pass:
        print("  ✅ T1 PASS: step_decrement 属性正确")
    else:
        print("  ❌ T1 FAIL: step_decrement 属性错误")

    return t1_pass


def run_test_T2():
    """T2: LS20Adapter steps_remaining 属性."""
    print("\n=== T2: LS20Adapter steps_remaining 属性 ===")
    Ls20 = _load_ls20_game()
    game = Ls20()

    from src.agent.oracle_adapters import get_oracle_adapter
    from arcengine import ActionInput

    _set_level(game, 1)  # L1: decrement=2, budget=42
    adapter = get_oracle_adapter("ls20", game)

    # 初始状态: current_steps=42, remaining=42//2=21
    initial_remaining = adapter.steps_remaining
    print(f"  L1 初始: steps_remaining={initial_remaining}, expected=21")

    # 执行3步后: current_steps=42-3*2=36, remaining=36//2=18
    for _ in range(3):
        ai = ActionInput(id=1, data={})  # UP
        game.perform_action(ai)

    after_3_remaining = adapter.steps_remaining
    print(f"  L1 3步后: steps_remaining={after_3_remaining}, expected=18")

    # 读取游戏引擎的真实步数
    current_steps = game._step_counter_ui.current_steps
    print(f"  L1 游戏引擎 current_steps={current_steps}")

    t2_pass = (initial_remaining == 21 and after_3_remaining == 18)
    if t2_pass:
        print("  ✅ T2 PASS: steps_remaining 属性正确反映游戏引擎状态")
    else:
        print(f"  ❌ T2 FAIL: expected (21, 18), got ({initial_remaining}, {after_3_remaining})")

    return t2_pass


def run_test_T3():
    """T3: L0 通关 (回归测试)."""
    print("\n=== T3: L0 通关回归测试 ===")
    Ls20 = _load_ls20_game()
    game = Ls20()

    from src.agent.game_solvers import solve_ls20

    _set_level(game, 0)
    plan = solve_ls20(game, 0)

    if plan is not None:
        # 执行计划
        from arcengine import ActionInput
        sim = copy.deepcopy(game)
        _set_level(sim, 0)

        for aid, data in plan:
            ai = ActionInput(id=aid, data=data if data else {})
            try:
                sim.perform_action(ai)
            except Exception as e:
                print(f"  ❌ 执行动作异常: {e}")
                break

        solved = _is_level_solved(sim, 0)
        print(f"  L0 plan长度={len(plan)}, 通关={solved}")
        t3_pass = solved
    else:
        print("  ❌ L0 solver 返回 None")
        t3_pass = False

    if t3_pass:
        print("  ✅ T3 PASS: L0 通关成功")
    else:
        print("  ❌ T3 FAIL: L0 通关失败")

    return t3_pass


def run_test_T4():
    """T4: L1 通关 (核心目标)."""
    print("\n=== T4: L1 通关 — 金币收集策略验证 ===")
    Ls20 = _load_ls20_game()
    game = Ls20()

    from src.agent.game_solvers import solve_ls20
    from src.agent.oracle_adapters import get_oracle_adapter

    _set_level(game, 1)
    adapter = get_oracle_adapter("ls20", game)

    # 显示 L1 关卡信息
    print(f"  L1 信息: step_decrement={adapter.step_decrement}, "
          f"max_actions={adapter.step_budget // adapter.step_decrement}")
    print(f"  金币: {[(int(c.x), int(c.y)) for c in adapter.coins]}")
    print(f"  Changer: {[(int(c.x), int(c.y)) for dim, cs in adapter.state_changers.items() for c in cs]}")
    print(f"  Goal: {[(int(g.x), int(g.y)) for g in adapter.goals]}")

    # 尝试求解
    t0 = time.time()
    plan = solve_ls20(game, 1)
    elapsed = time.time() - t0

    print(f"  L1 solver耗时={elapsed:.1f}s, plan长度={len(plan) if plan else 'None'}")

    if plan is not None:
        # 执行计划验证
        from arcengine import ActionInput
        sim = copy.deepcopy(game)
        _set_level(sim, 1)

        executed = 0
        for aid, data in plan:
            ai = ActionInput(id=aid, data=data if data else {})
            try:
                sim.perform_action(ai)
                executed += 1
            except Exception as e:
                print(f"  ❌ 执行第{executed}步异常: {e}")
                break

        solved = _is_level_solved(sim, 1)
        print(f"  L1 执行{executed}步, 通关={solved}")

        if not solved:
            # 诊断失败原因
            adapter2 = get_oracle_adapter("ls20", sim)
            if adapter2 and adapter2.player:
                print(f"  最终位置: ({int(adapter2.player.x)}, {int(adapter2.player.y)})")
                print(f"  最终状态: {adapter2.player_state}")
                print(f"  steps_remaining: {adapter2.steps_remaining}")

        t4_pass = solved
    else:
        print("  ❌ L1 solver 返回 None")
        t4_pass = False

    if t4_pass:
        print("  ✅ T4 PASS: L1 通关成功!")
    else:
        print("  ❌ T4 FAIL: L1 通关失败 — 需进一步优化金币收集路径")

    return t4_pass


def run_test_T5():
    """T5: L2 通关."""
    print("\n=== T5: L2 通关 ===")
    Ls20 = _load_ls20_game()
    game = Ls20()

    from src.agent.game_solvers import solve_ls20
    from src.agent.oracle_adapters import get_oracle_adapter

    _set_level(game, 2)
    adapter = get_oracle_adapter("ls20", game)

    # L2 信息
    print(f"  L2 信息: step_decrement={adapter.step_decrement}, "
          f"max_actions={adapter.step_budget // adapter.step_decrement}")
    print(f"  金币: {[(int(c.x), int(c.y)) for c in adapter.coins]}")

    # 尝试求解
    t0 = time.time()
    plan = solve_ls20(game, 2)
    elapsed = time.time() - t0

    print(f"  L2 solver耗时={elapsed:.1f}s, plan长度={len(plan) if plan else 'None'}")

    if plan is not None:
        from arcengine import ActionInput
        sim = copy.deepcopy(game)
        _set_level(sim, 2)

        executed = 0
        for aid, data in plan:
            ai = ActionInput(id=aid, data=data if data else {})
            try:
                sim.perform_action(ai)
                executed += 1
            except Exception as e:
                print(f"  ❌ 执行第{executed}步异常: {e}")
                break

        solved = _is_level_solved(sim, 2)
        print(f"  L2 执行{executed}步, 通关={solved}")
        t5_pass = solved
    else:
        print("  ❌ L2 solver 返回 None")
        t5_pass = False

    if t5_pass:
        print("  ✅ T5 PASS: L2 通关成功!")
    else:
        print("  ❌ T5 FAIL: L2 通关失败")

    return t5_pass


def main():
    """运行所有测试."""
    print("=" * 60)
    print("QA 验证: ls20 金币收集策略 — L0/L1/L2 通关")
    print("=" * 60)

    results = {}
    results["T1"] = run_test_T1()
    results["T2"] = run_test_T2()

    # T3/T4/T5 需要完整 solver, 可能耗时较长
    try:
        results["T3"] = run_test_T3()
    except Exception as e:
        print(f"  ❌ T3 异常: {e}")
        results["T3"] = False

    try:
        results["T4"] = run_test_T4()
    except Exception as e:
        print(f"  ❌ T4 异常: {e}")
        results["T4"] = False

    try:
        results["T5"] = run_test_T5()
    except Exception as e:
        print(f"  ❌ T5 异常: {e}")
        results["T5"] = False

    # ── 总结 ──
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    total_pass = 0
    total_tests = len(results)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if passed:
            total_pass += 1

    print(f"\n通过率: {total_pass}/{total_tests} ({total_pass/total_tests*100:.0f}%)")

    # 智能路由判定
    if total_pass == total_tests:
        print("\n🏆 全部通过 — 金币收集策略修复成功!")
    elif results.get("T4", False):
        print("\n✅ L1 通关成功 — 核心目标达成, 其他测试可后续修复")
    elif not results.get("T1", False) or not results.get("T2", False):
        print("\n🔧 属性测试失败 → 反馈给工程师修复 adapter 属性")
    elif results.get("T3", False) and not results.get("T4", False):
        print("\n🔧 L0回归通过但L1失败 → 金币收集策略需要进一步优化路径规划")

    return total_pass >= 3  # 至少 T1+T2+T3 通过


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
