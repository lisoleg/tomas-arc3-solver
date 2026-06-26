"""Debug test: run each stage of solve_ls20 separately on L0."""
import sys
import os
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, GameState, ActionInput
import copy

from src.agent.game_solvers import (
    _solve_oracle_replay,
    _solve_ls20_kappa_gradient_direct,
    _solve_ls20_kappa_ps_bfs,
    _is_level_solved,
    _detect_game_step,
)
from src.agent.oracle_adapters import get_oracle_adapter

LS20_BASELINES = [22, 123, 73, 84, 96, 192, 186]


def test_stage1_l0():
    """Test Stage 1 oracle replay on L0."""
    print("=== Stage 1: Oracle Replay on L0 ===")
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    obs = env.step(GameAction.ACTION1)

    game = copy.deepcopy(env._game)
    adapter = get_oracle_adapter("ls20", game)
    print(f"  Player: ({adapter.player.x}, {adapter.player.y})")
    print(f"  Switchers: {len(adapter.switchers)}")
    print(f"  Goals: {len(adapter.goals)}")
    if adapter.goals:
        for g in adapter.goals:
            print(f"    Goal: ({g.x}, {g.y})")
    step_size = _detect_game_step(game)
    print(f"  Step size: {step_size}")

    t0 = time.time()
    plan = _solve_oracle_replay(game, "ls20", 0, max_steps=300, max_time=15.0)
    elapsed = time.time() - t0

    print(f"  Result: plan_len={len(plan) if plan else 0}, time={elapsed:.1f}s")
    if plan:
        # Show first 10 actions
        for i, (aid, data) in enumerate(plan[:10]):
            print(f"    Action {i}: id={aid}, data={data}")
        if len(plan) > 10:
            print(f"    ... ({len(plan)} total)")

        # Verify on pristine copy
        sim = copy.deepcopy(env._game)
        original_level = sim._current_level_index
        for aid, data in plan:
            ai = ActionInput(id=aid, data=data if data else {})
            try:
                sim.perform_action(ai)
            except Exception as e:
                print(f"    Error at {aid}: {e}")
                break
        solved = _is_level_solved(sim, original_level)
        print(f"  Verified: {solved}")


def test_stage2_l0():
    """Test Stage 2 kappa gradient direct on L0."""
    print("\n=== Stage 2: Kappa Gradient Direct on L0 ===")
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    obs = env.step(GameAction.ACTION1)

    game = copy.deepcopy(env._game)
    t0 = time.time()
    plan = _solve_ls20_kappa_gradient_direct(game, 0)
    elapsed = time.time() - t0

    print(f"  Result: plan_len={len(plan) if plan else 0}, time={elapsed:.1f}s")
    if plan:
        # Count ACTION6 clicks
        click_count = sum(1 for aid, _ in plan if aid == 6)
        move_count = len(plan) - click_count
        print(f"  Actions breakdown: {move_count} moves, {click_count} clicks")

        for i, (aid, data) in enumerate(plan[:15]):
            print(f"    Action {i}: id={aid}, data={data}")

        # Verify on pristine copy
        sim = copy.deepcopy(env._game)
        original_level = sim._current_level_index
        for aid, data in plan:
            ai = ActionInput(id=aid, data=data if data else {})
            try:
                sim.perform_action(ai)
            except Exception as e:
                print(f"    Error at {aid}: {e}")
                break
        solved = _is_level_solved(sim, original_level)
        print(f"  Verified: {solved}")


def test_l1_direct():
    """Test solve_ls20 on L1 (the critical level with switcher)."""
    print("\n=== L1: Full pipeline test ===")
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    obs = env.step(GameAction.ACTION1)

    # First solve L0 to reach L1
    game_l0 = copy.deepcopy(env._game)
    from src.agent.game_solvers import solve_ls20
    plan_l0 = solve_ls20(game_l0, 0)

    if plan_l0 is None:
        print("  L0 failed — cannot test L1")
        return

    # Replay L0 on env to advance to L1
    for aid, data in plan_l0:
        ai = ActionInput(id=aid, data=data if data else {})
        try:
            env.step(ai)
        except Exception:
            pass

    if env._game._current_level_index <= 0:
        print("  Could not advance to L1")
        return

    l1_idx = env._game._current_level_index
    print(f"  Advanced to level index {l1_idx}")

    game_l1 = copy.deepcopy(env._game)
    adapter = get_oracle_adapter("ls20", game_l1)
    print(f"  Player: ({adapter.player.x}, {adapter.player.y})")
    print(f"  Switchers: {len(adapter.switchers)}")
    if adapter.switchers:
        for sw in adapter.switchers:
            print(f"    Switcher: ({sw.x}, {sw.y})")
    print(f"  Goals: {len(adapter.goals)}")
    if adapter.goals:
        for g in adapter.goals:
            print(f"    Goal: ({g.x}, {g.y})")

    t0 = time.time()
    plan_l1 = solve_ls20(game_l1, l1_idx)
    elapsed = time.time() - t0

    print(f"  Result: plan_len={len(plan_l1) if plan_l1 else 0}, time={elapsed:.1f}s")

    if plan_l1:
        # Count actions
        click_count = sum(1 for aid, _ in plan_l1 if aid == 6)
        move_count = len(plan_l1) - click_count
        print(f"  Actions breakdown: {move_count} moves, {click_count} clicks")

        for i, (aid, data) in enumerate(plan_l1[:15]):
            print(f"    Action {i}: id={aid}, data={data}")

        # Verify
        sim = copy.deepcopy(env._game)
        original_level = sim._current_level_index
        for aid, data in plan_l1:
            ai = ActionInput(id=aid, data=data if data else {})
            try:
                sim.perform_action(ai)
            except Exception as e:
                print(f"    Error at {aid}: {e}")
                break
        solved = _is_level_solved(sim, original_level)
        print(f"  Verified: {solved}")
    else:
        print("  L1 FAILED: plan is None")


if __name__ == "__main__":
    test_stage1_l0()
    test_stage2_l0()
    test_l1_direct()
