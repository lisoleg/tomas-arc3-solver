#!/usr/bin/env python
"""Regression test: Verify NAR-Conv + TOMASLearner don't break LS20/FT09/TR87 solvers.

Uses PlannerAgent (the real pipeline) instead of direct game manipulation.
This is the same approach as test_tr87_all_levels.py and test_planner_ls20.py.

Expected results (before upgrade):
- LS20: 7/7 levels, RHAE~115.0 each
- FT09: 6/6 levels, RHAE~115.0 each
- TR87: 5/6 levels (L5 stuck)

Usage:
    python tests/test_regression_v321.py
"""
import sys
import os
import time
import json

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Add environment files
env_files = os.path.join(PROJECT_ROOT, "environment_files")
if os.path.exists(env_files):
    sys.path.insert(0, env_files)

import arc_agi
from arcengine import GameAction

# ===== STEP 1: Import verification =====
print("=" * 70)
print("STEP 1: Import verification — NAR-Conv + TOMASLearner")
print("=" * 70)

from src.agent.tomas_learner import TOMASLearner, EpisodeTrace, ActionTrace
from src.encoder.nar_conv import OctonionConv2d, NARGridEncoder, color_index_to_octonion
from src.agent.planner_agent import PlannerAgent
from src.agent.game_solvers import solve_game

print("✅ All imports OK — no import-level regression")

# ===== STEP 2: RHAE regression test =====
print("\n" + "=" * 70)
print("STEP 2: RHAE regression test — LS20 / FT09 / TR87")
print("=" * 70)

GAME_CONFIGS = {
    "ls20": {
        "baselines": [22, 123, 73, 84, 96, 192, 186],
        "num_levels": 7,
        "max_steps_per_level": 500,
    },
    "ft09": {
        "baselines": [17, 51, 38, 42, 56, 34],
        "num_levels": 6,
        "max_steps_per_level": 300,
    },
    "tr87": {
        "baselines": [54, 58, 40, 66, 55, 78],
        "num_levels": 6,
        "max_steps_per_level": 256,
    },
}

def compute_rhae(baseline: int, steps: int) -> float:
    """RHAE = min(115.0, (baseline / max(1, steps))^2 * 100)"""
    if steps <= 0 or baseline <= 0:
        return 0.0
    if steps > baseline * 5:
        return 0.0
    return min(115.0, (baseline / max(1, steps)) ** 2 * 100)

def test_game_with_planner(game_id: str, config: dict) -> dict:
    """Test a game using PlannerAgent pipeline (the real solver)."""
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make(game_id)
    
    baselines = config["baselines"]
    num_levels = config["num_levels"]
    max_steps = config["max_steps_per_level"]
    
    agent = PlannerAgent(
        env=env,
        level_baselines=baselines,
        max_steps=max_steps * num_levels,
        stagnation_threshold=max_steps,
        use_oracle=True,
        game_id=game_id,
    )
    agent._check_oracle_availability()
    print(f"  Oracle available: {agent._oracle_adapter is not None}")
    
    obs = env.step(GameAction.RESET)
    total_steps = 0
    per_level_results = []
    levels_completed = 0
    total_rhae = 0.0
    
    for level_idx in range(num_levels):
        print(f"\n  --- Level {level_idx} ---")
        level_steps = 0
        level_max = max_steps
        
        while level_steps < level_max:
            done = agent.is_done(frames=[obs], latest_frame=obs)
            if done:
                break
            
            action = agent.choose_action(frames=[obs], latest_frame=obs)
            if action is None:
                print(f"    No action at step {level_steps}")
                break
            
            data = agent.get_pending_action_data()
            obs = env.step(action, data=data)
            level_steps += 1
            total_steps += 1
            
            # Check game over
            if hasattr(obs, 'state') and str(obs.state) == 'GameState.GAME_OVER':
                print(f"    GAME_OVER at step {level_steps}")
                break
        
        # Check completion
        if obs.levels_completed > level_idx:
            levels_completed += 1
            baseline = baselines[level_idx] if level_idx < len(baselines) else 50
            rhae = compute_rhae(baseline, level_steps)
            total_rhae += rhae
            print(f"    ✅ L{level_idx} done in {level_steps} steps (baseline={baseline}, RHAE={rhae:.1f})")
            per_level_results.append({
                "level": level_idx, "status": "PASS",
                "steps": level_steps, "baseline": baseline,
                "rhae": rhae,
            })
        else:
            print(f"    ❌ L{level_idx} NOT completed ({level_steps} steps)")
            per_level_results.append({
                "level": level_idx, "status": "FAIL",
                "steps": level_steps,
                "baseline": baselines[level_idx] if level_idx < len(baselines) else 50,
                "rhae": 0.0,
            })
    
    avg_rhae = total_rhae / max(1, levels_completed) if levels_completed > 0 else 0.0
    
    return {
        "game_id": game_id,
        "levels_completed": levels_completed,
        "total_levels": num_levels,
        "avg_rhae": avg_rhae,
        "total_rhae": total_rhae,
        "total_steps": total_steps,
        "per_level": per_level_results,
    }

all_results = {}
for game_id, config in GAME_CONFIGS.items():
    print(f"\n--- Testing {game_id} ---")
    result = test_game_with_planner(game_id, config)
    all_results[game_id] = result
    print(f"  Summary: {result['levels_completed']}/{result['total_levels']} | avg RHAE={result['avg_rhae']:.1f}")

# Regression verification
print("\n" + "=" * 70)
print("REGRESSION VERIFICATION")
print("=" * 70)

EXPECTED = {
    "ls20": {"min_levels": 7, "min_avg_rhae": 100.0},
    "ft09": {"min_levels": 6, "min_avg_rhae": 100.0},
    "tr87": {"min_levels": 5, "min_avg_rhae": 80.0},
}

regression_passed = True
for game_id, result in all_results.items():
    exp = EXPECTED[game_id]
    lc = result["levels_completed"]
    ar = result["avg_rhae"]
    
    lc_ok = lc >= exp["min_levels"]
    rhae_ok = ar >= exp["min_avg_rhae"]
    
    lc_icon = "✅" if lc_ok else "❌"
    rhae_icon = "✅" if rhae_ok else "❌"
    print(f"  {game_id}: levels {lc}/{exp['min_levels']} {lc_icon} | RHAE {ar:.1f}/{exp['min_avg_rhae']} {rhae_icon}")
    
    if not (lc_ok and rhae_ok):
        regression_passed = False

print(f"\n{'🎉 NO REGRESSION!' if regression_passed else '⚠️ REGRESSION — investigate!'}")

# Save results
results_path = os.path.join(PROJECT_ROOT, "test_regression_v321_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to: {results_path}")
