"""验证 ls20 L1/L0/L2 通关 — 金币 proximity position 修复后回归测试.

关键修复: 金币锚点 (15,16)/(40,51) 不在玩家步长网格 (29+5k, 40+5m) 上,
使用 _compute_coin_proximity_position() 计算包围盒重叠的 proximity position
作为 BFS 目标, 使金币可通过碰撞检测被收集.
"""
import sys
import os
import importlib.util
import copy

# ── 动态加载 ls20 游戏 ──
GAME_PATH = os.path.join(
    os.path.dirname(__file__), 'environment_files', 'ls20', '9607627b', 'ls20.py'
)
spec = importlib.util.spec_from_file_location('ls20', GAME_PATH)
ls20_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls20_mod)
Ls20 = ls20_mod.Ls20

from arcengine import ActionInput

# ── 导入 solver 函数 ──
sys.path.insert(0, os.path.dirname(__file__))
from src.agent.game_solvers import (
    _compute_coin_proximity_position,
    _detect_game_step,
    solve_ls20,
)

def test_proximity_position():
    """T1: 验证 proximity position 计算"""
    # L1 coin at (15, 16), player start (29, 40), step_size=5
    prox = _compute_coin_proximity_position(29, 40, 5, 15, 16)
    assert prox == (14, 15), f"Expected (14, 15), got {prox}"
    # Verify coin anchor in bounding box
    assert 15 >= 14 and 15 < 14 + 5, "coin_x not in bounding box"
    assert 16 >= 15 and 16 < 15 + 5, "coin_y not in bounding box"
    
    # L1 coin at (40, 51), player start (29, 40), step_size=5
    prox2 = _compute_coin_proximity_position(29, 40, 5, 40, 51)
    assert prox2 == (39, 50), f"Expected (39, 50), got {prox2}"
    assert 40 >= 39 and 40 < 39 + 5, "coin_x not in bounding box"
    assert 51 >= 50 and 51 < 50 + 5, "coin_y not in bounding box"
    
    print("T1 PASS: proximity position calculation correct")
    return True

def test_l0_regression():
    """T2: L0 回归测试 — 应仍然通关"""
    game = Ls20()
    game._current_level_index = 0
    game._levels[0] = game._clean_levels[0].clone()
    game.on_set_level(game._levels[0])
    
    plan = solve_ls20(game, 0)
    if plan is None:
        print("T2 FAIL: L0 solver returned None")
        return False
    
    # Replay plan on pristine copy
    sim = Ls20()
    sim._current_level_index = 0
    sim._levels[0] = sim._clean_levels[0].clone()
    sim.on_set_level(sim._levels[0])
    
    for aid, data in plan[:300]:
        ai = ActionInput(id=aid, data=data if data else {})
        try:
            sim.perform_action(ai)
        except Exception:
            break
    
    solved = sim._current_level_index > 0
    if solved:
        print(f"T2 PASS: L0 通关 in {len(plan)} steps")
    else:
        print(f"T2 FAIL: L0 not solved after {len(plan)} steps")
    return solved

def test_l1_passage():
    """T3: L1 通关测试 — 关键! 金币 proximity position 修复"""
    game = Ls20()
    game._current_level_index = 1
    game._levels[1] = game._clean_levels[1].clone()
    game.on_set_level(game._levels[1])
    
    plan = solve_ls20(game, 1)
    if plan is None:
        print("T3 FAIL: L1 solver returned None")
        return False
    
    # Replay plan on pristine copy
    sim = Ls20()
    sim._current_level_index = 1
    sim._levels[1] = sim._clean_levels[1].clone()
    sim.on_set_level(sim._levels[1])
    
    for aid, data in plan[:300]:
        ai = ActionInput(id=aid, data=data if data else {})
        try:
            sim.perform_action(ai)
        except Exception:
            break
    
    solved = sim._current_level_index > 1
    if solved:
        print(f"T3 PASS: L1 通关 in {len(plan)} steps!")
    else:
        print(f"T3 FAIL: L1 not solved after {len(plan)} steps")
    return solved

def test_l2_regression():
    """T4: L2 回归测试 — 应仍然通关"""
    game = Ls20()
    game._current_level_index = 2
    game._levels[2] = game._clean_levels[2].clone()
    game.on_set_level(game._levels[2])
    
    plan = solve_ls20(game, 2)
    if plan is None:
        print("T4 FAIL: L2 solver returned None")
        return False
    
    # Replay plan on pristine copy
    sim = Ls20()
    sim._current_level_index = 2
    sim._levels[2] = sim._clean_levels[2].clone()
    sim.on_set_level(sim._levels[2])
    
    for aid, data in plan[:300]:
        ai = ActionInput(id=aid, data=data if data else {})
        try:
            sim.perform_action(ai)
        except Exception:
            break
    
    solved = sim._current_level_index > 2
    if solved:
        print(f"T4 PASS: L2 通关 in {len(plan)} steps")
    else:
        print(f"T4 FAIL: L2 not solved after {len(plan)} steps")
    return solved

# ── 运行所有测试 ──
results = []
print("=" * 60)
print("  LS20 金币 Proximity Position 修复验证")
print("=" * 60)

try:
    results.append(("T1", test_proximity_position()))
except Exception as e:
    print(f"T1 ERROR: {e}")
    results.append(("T1", False))

try:
    results.append(("T2", test_l0_regression()))
except Exception as e:
    print(f"T2 ERROR: {e}")
    results.append(("T2", False))

try:
    results.append(("T3", test_l1_passage()))
except Exception as e:
    print(f"T3 ERROR: {e}")
    results.append(("T3", False))

try:
    results.append(("T4", test_l2_regression()))
except Exception as e:
    print(f"T4 ERROR: {e}")
    results.append(("T4", False))

print("\n" + "=" * 60)
print("  测试结果汇总")
print("=" * 60)
for name, passed in results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {name}: {status}")
total_pass = sum(1 for _, p in results if p)
print(f"\n  总计: {total_pass}/{len(results)} 通过")
print("=" * 60)
