"""
Test the PlannerAgent V5 (Hybrid Planner-Learner) on LS20.

This agent combines:
1. Game Introspection: Read env._game for perfect state knowledge
2. Route Planning: BFS pathfinding + type-order optimization (Oracle v17 core)
3. RL Meta-Learning: Q-learning prioritizes route orderings
4. IRL Trauma Memory: Avoid dangerous positions from GAME_OVER
5. Library Learning: Cross-level macro transfer
6. Alpha-Beta Pruning: Cut routes that can't beat current best
7. Circuit Breaker: Abort failing plans, switch to exploration
8. Pixel Fallback: DopamineExplorer for unknown game types

Usage:
    python tests/test_planner_ls20.py
"""

import sys
import os
import time
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Also add the arc_agi environment files to path
env_files = os.path.join(project_root, "environment_files")
if os.path.exists(env_files):
    sys.path.insert(0, env_files)

# LS20 per-level baseline actions (from environment metadata)
LS20_BASELINES = [22, 123, 73, 84, 96, 192, 186]
LS20_TOTAL_BASELINE = sum(LS20_BASELINES)  # 776


def run_planner_test():
    """Run the PlannerAgent V5 on the LS20 game environment."""
    print("=" * 70)
    print("PlannerAgent V5 (Hybrid Planner-Learner) LS20 Test")
    print("=" * 70)

    # Step 1: Import and create agent
    print("\n[1] Importing PlannerAgent...")
    from src.agent import PlannerAgent
    print("    [PASS] PlannerAgent imported")

    # Step 2: Create environment
    print("\n[2] Creating LS20 environment...")
    import arc_agi
    from arcengine import GameAction, GameState

    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    print(f"    Environment created: ls20")

    for e in arc.available_environments:
        if e.game_id == "ls20":
            print(f"    Tags: {e.tags}")
            print(f"    Baseline actions: {e.baseline_actions}")
            print(f"    Win levels: {len(e.baseline_actions)}")
            break

    # Step 3: Create agent with env reference
    print("\n[3] Creating PlannerAgent...")
    agent = PlannerAgent(
        env=env,
        level_baselines=LS20_BASELINES,
        max_steps=2000,
        stagnation_threshold=500,
    )
    print(f"    Agent created (max_steps=2000, stagnation=500)")

    # Step 4: Initial step
    print("\n[4] Taking initial step (RESET)...")
    obs = env.step(GameAction.RESET)
    frames = [obs]
    print(f"    State: {obs.state}")
    print(f"    Levels: {obs.levels_completed}/{obs.win_levels}")
    print(f"    Available actions: {obs.available_actions}")

    grid = np.array(obs.frame[0])
    print(f"    Grid shape: {grid.shape}")
    print(f"    Grid colors: {sorted(np.unique(grid).tolist())}")

    # Step 5: Run agent loop
    print(f"\n[5] Running agent for up to 2000 steps...")
    print("-" * 70)

    step_count = 0
    prev_levels = obs.levels_completed
    prev_state = obs.state
    start_time = time.time()
    level_start_step = 0
    level_steps_list = []

    while not agent.is_done(frames, obs) and step_count < 2000:
        action = agent.choose_action(frames, obs)
        obs = env.step(action)
        frames.append(obs)
        step_count += 1

        state_changed = obs.state != prev_state
        levels_changed = obs.levels_completed != prev_levels

        if levels_changed:
            level_steps = step_count - level_start_step
            level_steps_list.append(level_steps)
            baseline = LS20_BASELINES[min(prev_levels, len(LS20_BASELINES) - 1)]
            rhae = min(115, ((baseline / max(level_steps, 1)) ** 2) * 100)
            print(f"    >>> Level {prev_levels} completed in {level_steps} steps "
                  f"(baseline: {baseline}, RHAE: {rhae:.1f})")
            level_start_step = step_count

        if state_changed:
            state_name = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
            print(f"    >>> State changed! -> {state_name}")

        prev_levels = obs.levels_completed
        prev_state = obs.state

    elapsed = time.time() - start_time

    # Step 6: Report
    print("\n" + "=" * 70)
    print("[6] Final Results")
    print("=" * 70)

    final_state = obs.state
    state_name = final_state.name if hasattr(final_state, "name") else str(final_state)
    print(f"  Total steps: {step_count}")
    print(f"  Final state: {state_name}")
    print(f"  Levels completed: {obs.levels_completed}/{obs.win_levels}")
    print(f"  Time: {elapsed:.2f}s ({step_count / max(elapsed, 0.01):.1f} steps/s)")

    if level_steps_list:
        print(f"\n  Per-level steps:")
        total_rhae = 0
        for i, ls in enumerate(level_steps_list):
            baseline = LS20_BASELINES[i] if i < len(LS20_BASELINES) else "?"
            if isinstance(baseline, int) and ls > 0:
                ratio = ls / baseline
                eff = min(115, ((baseline / ls) ** 2) * 100)
                total_rhae += eff
                print(f"    Level {i}: {ls} steps (baseline: {baseline}, "
                      f"ratio: {ratio:.2f}x, RHAE: {eff:.1f})")
            else:
                print(f"    Level {i}: {ls} steps (baseline: {baseline})")

    # RHAE calculation
    print(f"\n  RHAE Calculation:")
    if step_count > 0:
        rhae = min(115, (LS20_TOTAL_BASELINE / step_count) ** 2 * 100)
        print(f"    Overall RHAE = min(115, ({LS20_TOTAL_BASELINE}/{step_count})^2 * 100) = {rhae:.1f}")
        if rhae >= 80:
            print(f"    >>> RHAE >= 80: PASS!")
        elif rhae >= 50:
            print(f"    >>> RHAE >= 50: improving...")
        else:
            print(f"    >>> RHAE < 50: need more optimization")

    stats = agent.get_stats()
    print(f"\n  Agent Statistics:")
    print(f"    Total actions: {stats['total_actions']}")
    print(f"    Game over count: {stats['game_over_count']}")
    print(f"    Level game overs: {stats['level_game_overs']}")
    print(f"    Route Q-table size: {stats['route_q_size']}")
    print(f"    Danger walls: {stats['wall_count']}")
    print(f"    Macro count: {stats['macro_count']}")
    print(f"    Using fallback: {stats['using_fallback']}")
    print(f"    Plan attempts: {stats['plan_attempt']}")
    print(f"    Stagnation: {stats['stagnation']}")
    print(f"    Level step counts: {stats['level_steps']}")

    action_seq = agent.memory.get_action_sequence()
    print(f"    Action sequence (last 30): {action_seq[-30:]}")

    print("\n  Outcome: ", end="")
    if final_state == GameState.WIN:
        print("SUCCESS - Game WON!")
    elif final_state == GameState.GAME_OVER:
        print(f"FAILED - Game Over (retries: {stats['game_over_count']})")
    elif stats["stagnation"] >= 500:
        print("STUCK - Agent stuck")
    elif step_count >= 2000:
        print("TIMEOUT - Max steps reached")
    else:
        print("UNKNOWN")

    print("=" * 70)
    return {"steps": step_count, "state": state_name, "levels": obs.levels_completed}


def main():
    """Main entry point."""
    try:
        run_planner_test()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
