"""
Multi-game test for PlannerAgent V5 with grid perception.

Tests the agent across multiple ARC-AGI-3 games:
1. LS20 (oracle mode, keyboard) - backward compatibility check
2. VC33 (grid mode, click) - pure grid inference + click support
3. TR87 (grid mode, keyboard) - different sprite structure
4. Additional games as available

Usage:
    python tests/test_multi_game.py
    python tests/test_multi_game.py --games ls20 vc33 tr87
    python tests/test_multi_game.py --max-steps 500
"""

import sys
import os
import time
import argparse
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Add the arc_agi environment files to path
env_files = os.path.join(project_root, "environment_files")
if os.path.exists(env_files):
    sys.path.insert(0, env_files)

# All 25 game baselines
GAME_BASELINES = {
    "ls20": [22, 123, 73, 84, 96, 192, 186],
    "vc33": [7, 18, 44, 61, 131, 34, 152],
    "tr87": [54, 58, 40, 45, 71, 146],
    "tu93": [19, 16, 34, 42, 123, 80, 14, 23, 111],
    "bp35": [21, 48, 44, 38, 33, 87, 86, 131, 163],
    "dc22": [59, 102, 67, 98, 324, 578],
    "s5i5": [20, 89, 106, 54, 162, 38, 86, 83],
    "sk48": [61, 177, 101, 103, 230, 181, 125, 92],
    "tn36": [32, 72, 26, 40, 30, 55, 62],
    "ft09": [43, 12, 23, 28, 65, 37],
    "su15": [22, 42, 26, 115, 36, 31, 8, 40, 41],
    "lf52": [32, 81, 60, 71, 205, 148, 244, 109, 164, 225],
    "sc25": [36, 6, 32, 83, 143, 50],
    "m0r0": [30, 111, 203, 26, 500, 237],
    "re86": [26, 42, 86, 108, 189, 139, 424, 241],
    "r11l": [22, 33, 51, 26, 52, 49],
    "cn04": [29, 54, 85, 300, 208, 113],
    "lp85": [17, 38, 31, 16, 41, 60, 26, 159],
    "cd82": [55, 8, 41, 21, 23, 23],
    "g50t": [78, 175, 179, 230, 96, 54, 67],
    "sp80": [39, 58, 25, 148, 96, 152],
    "ka59": [28, 109, 51, 51, 33, 132, 326],
    "ar25": [32, 50, 75, 37, 89, 159, 233, 73],
    "wa30": [71, 119, 183, 98, 368, 68, 79, 442, 415],
    "sb26": [18, 28, 18, 19, 31, 23, 58, 18],
}


def run_game_test(game_id: str, max_steps: int = 500, use_oracle: bool = None) -> dict:
    """Run PlannerAgent on a single game.

    Args:
        game_id: Game identifier (e.g., "ls20", "vc33").
        max_steps: Maximum steps for this game.
        use_oracle: Force oracle mode (True), grid mode (False), or auto (None).

    Returns:
        Dict with test results.
    """
    print(f"\n{'=' * 70}")
    print(f"Testing: {game_id}")
    print(f"{'=' * 70}")

    # Import
    from src.agent import PlannerAgent
    import arc_agi
    from arcengine import GameAction, GameState

    arc = arc_agi.Arcade()
    env = arc.make(game_id)

    # Get baselines
    baselines = GAME_BASELINES.get(game_id, [])
    total_baseline = sum(baselines) if baselines else 0

    # Get game info
    obs = env.step(GameAction.RESET)
    grid = np.array(obs.frame[0])
    colors = sorted(np.unique(grid).tolist())
    actions = obs.available_actions
    win_levels = obs.win_levels

    print(f"  Baselines: {baselines} (total={total_baseline})")
    print(f"  Grid: 64x64, colors={colors}")
    print(f"  Actions: {actions}")
    print(f"  Win levels: {win_levels}")

    # Determine action type
    keyboard_actions = {1, 2, 3, 4}
    click_actions = {6, 7}
    has_keyboard = any(a in keyboard_actions for a in actions)
    has_click = any(a in click_actions for a in actions)
    action_type = "keyboard_click" if (has_keyboard and has_click) else \
                  "keyboard" if has_keyboard else \
                  "click" if has_click else "unknown"
    print(f"  Action type: {action_type}")

    # Create agent
    agent = PlannerAgent(
        env=env,
        level_baselines=baselines,
        max_steps=max_steps,
        stagnation_threshold=max_steps // 2,
        use_oracle=use_oracle,
        game_id=game_id,
    )
    # Note: use_oracle is determined on first choose_action call
    # Force check now for reporting
    agent._check_oracle_availability()
    oracle_mode = agent.use_oracle
    print(f"  Oracle mode: {oracle_mode}")

    # Run agent
    print(f"\n  Running for up to {max_steps} steps...")
    print(f"  {'-' * 66}")

    step_count = 0
    prev_levels = obs.levels_completed
    prev_state = obs.state
    level_start_step = 0
    level_steps_list = []
    game_over_count = 0
    start_time = time.time()

    while not agent.is_done(frames=[obs], latest_frame=obs) and step_count < max_steps:
        action = agent.choose_action(frames=[obs], latest_frame=obs)

        # Handle click data for click games
        click_data = agent.get_pending_action_data()

        try:
            if click_data is not None:
                obs = env.step(action, data=click_data)
            else:
                obs = env.step(action)
        except Exception as e:
            print(f"    [ERROR] env.step failed: {e}")
            break

        step_count += 1

        # Track level changes
        levels_changed = obs.levels_completed != prev_levels
        state_changed = obs.state != prev_state

        if levels_changed:
            level_steps = step_count - level_start_step
            level_steps_list.append(level_steps)
            baseline = baselines[min(prev_levels, len(baselines) - 1)] if baselines else "?"
            if isinstance(baseline, int) and level_steps > 0:
                rhae = min(115, ((baseline / level_steps) ** 2) * 100)
                print(f"    >>> Level {prev_levels} done in {level_steps} steps "
                      f"(baseline: {baseline}, RHAE: {rhae:.1f})")
            else:
                print(f"    >>> Level {prev_levels} done in {level_steps} steps")
            level_start_step = step_count

        if state_changed:
            state_name = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
            if state_name == "GAME_OVER":
                game_over_count += 1
                print(f"    >>> GAME_OVER (count={game_over_count})")
            elif state_name == "WIN":
                print(f"    >>> WIN!")

        prev_levels = obs.levels_completed
        prev_state = obs.state

    elapsed = time.time() - start_time

    # Calculate RHAE
    final_state = obs.state
    state_name = final_state.name if hasattr(final_state, "name") else str(final_state)
    levels_completed = obs.levels_completed

    rhae = 0.0
    if total_baseline > 0 and step_count > 0:
        rhae = min(115, ((total_baseline / step_count) ** 2) * 100)

    # Report
    print(f"\n  {'=' * 66}")
    print(f"  Results for {game_id}:")
    print(f"    Steps: {step_count}/{max_steps}")
    print(f"    State: {state_name}")
    print(f"    Levels: {levels_completed}/{win_levels}")
    print(f"    Game overs: {game_over_count}")
    print(f"    Time: {elapsed:.1f}s ({step_count / max(elapsed, 0.01):.1f} steps/s)")
    print(f"    RHAE: {rhae:.1f}")
    print(f"    Oracle: {oracle_mode}")

    if level_steps_list:
        print(f"    Per-level: {level_steps_list}")

    stats = agent.get_stats()
    print(f"    Fallback: {stats.get('using_fallback', False)}")
    print(f"    Plan attempts: {stats.get('plan_attempt', 0)}")

    outcome = "WIN" if final_state == GameState.WIN else \
              "TIMEOUT" if step_count >= max_steps else \
              "GAME_OVER" if final_state == GameState.GAME_OVER else \
              "STUCK" if stats.get("stagnation", 0) >= max_steps // 2 else \
              "UNKNOWN"

    print(f"    Outcome: {outcome}")
    print(f"  {'=' * 66}")

    return {
        "game_id": game_id,
        "steps": step_count,
        "max_steps": max_steps,
        "state": state_name,
        "levels_completed": levels_completed,
        "win_levels": win_levels,
        "game_overs": game_over_count,
        "rhae": rhae,
        "oracle": oracle_mode,
        "action_type": action_type,
        "outcome": outcome,
        "elapsed": elapsed,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-game ARC-AGI-3 test")
    parser.add_argument(
        "--games", nargs="*", default=["ls20", "vc33", "tr87"],
        help="Game IDs to test (default: ls20 vc33 tr87)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Maximum steps per game (default: 500)",
    )
    parser.add_argument(
        "--oracle", choices=["auto", "yes", "no"], default="auto",
        help="Force oracle mode (yes), grid mode (no), or auto-detect (auto)",
    )
    args = parser.parse_args()

    use_oracle = None if args.oracle == "auto" else (args.oracle == "yes")

    print("=" * 70)
    print("Multi-Game PlannerAgent V5 Test")
    print(f"Games: {args.games}")
    print(f"Max steps: {args.max_steps}")
    print(f"Oracle: {args.oracle}")
    print("=" * 70)

    results = []
    for game_id in args.games:
        try:
            result = run_game_test(game_id, args.max_steps, use_oracle)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR testing {game_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "game_id": game_id,
                "outcome": "ERROR",
                "error": str(e),
                "rhae": 0.0,
                "steps": 0,
                "levels_completed": 0,
            })

    # Summary
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Game':<10} {'Steps':<8} {'Levels':<10} {'RHAE':<8} {'Oracle':<8} {'Type':<16} {'Outcome'}")
    print(f"{'-' * 70}")

    total_rhae = 0
    total_wins = 0
    for r in results:
        gid = r["game_id"]
        steps = r.get("steps", 0)
        levels = f"{r.get('levels_completed', 0)}/{r.get('win_levels', '?')}"
        rhae = r.get("rhae", 0.0)
        oracle = str(r.get("oracle", "?"))
        atype = r.get("action_type", "?")
        outcome = r.get("outcome", "?")

        total_rhae += rhae
        if outcome == "WIN":
            total_wins += 1

        print(f"{gid:<10} {steps:<8} {levels:<10} {rhae:<8.1f} {oracle:<8} {atype:<16} {outcome}")

    avg_rhae = total_rhae / max(len(results), 1)
    print(f"\n  Average RHAE: {avg_rhae:.1f}")
    print(f"  Wins: {total_wins}/{len(results)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
