"""Full 25-game RHAE benchmark for ARC-AGI-3.

Runs each game with Oracle mode, reports per-level RHAE and total score.
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import arc_agi
from arcengine import GameAction
from agent.planner_agent import PlannerAgent

# All 25 games with estimated baselines
GAMES = {
    # keyboard_only (6)
    'ls20': [22, 123, 73, 84, 96, 192, 186],
    'tr87': [30, 50, 70, 90, 110, 130, 150],
    'tu93': [30, 50, 70, 90, 110, 130, 150],
    're86': [30, 50, 70, 90, 110, 130, 150],
    'g50t': [30, 50, 70, 90, 110, 130, 150],
    'wa30': [30, 50, 70, 90, 110, 130, 150],
    # click_only (7)
    'vc33': [30, 50, 70, 90, 110, 130, 150],
    's5i5': [30, 50, 70, 90, 110, 130, 150],
    'tn36': [30, 50, 70, 90, 110, 130, 150],
    'ft09': [5, 7, 14, 16, 30, 50, 70],
    'su15': [30, 50, 70, 90, 110, 130, 150],
    'r11l': [30, 50, 70, 90, 110, 130, 150],
    'lp85': [30, 50, 70, 90, 110, 130, 150],
    # keyboard_click (12)
    'bp35': [30, 50, 70, 90, 110, 130, 150],
    'dc22': [30, 50, 70, 90, 110, 130, 150],
    'sk48': [30, 50, 70, 90, 110, 130, 150],
    'lf52': [30, 50, 70, 90, 110, 130, 150],
    'sc25': [30, 50, 70, 90, 110, 130, 150],
    'm0r0': [30, 50, 70, 90, 110, 130, 150],
    'cn04': [30, 50, 70, 90, 110, 130, 150],
    'cd82': [30, 50, 70, 90, 110, 130, 150],
    'sp80': [30, 50, 70, 90, 110, 130, 150],
    'ka59': [30, 50, 70, 90, 110, 130, 150],
    'ar25': [30, 50, 70, 90, 110, 130, 150],
    'sb26': [30, 50, 70, 90, 110, 130, 150],
}

MAX_STEPS = 1000
STAGNATION = 500


def run_game(game_id: str, baselines: list[int]) -> dict:
    """Run a single game and return results."""
    result = {
        'game_id': game_id,
        'levels_completed': 0,
        'total_steps': 0,
        'per_level_rhae': [],
        'error': None,
        'oracle': False,
        'adapter': None,
    }

    try:
        arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
        env = arc.make(game_id)
        obs = env.step(GameAction.RESET)

        agent = PlannerAgent(
            env=env,
            level_baselines=baselines,
            max_steps=MAX_STEPS,
            stagnation_threshold=STAGNATION,
            use_oracle=True,
            game_id=game_id,
        )
        agent._check_oracle_availability()
        result['oracle'] = agent.use_oracle
        result['adapter'] = type(agent._oracle_adapter).__name__ if agent._oracle_adapter else None

        levels = 0
        step = 0
        level_start = 0

        while step < MAX_STEPS and levels < 7:
            action = agent.choose_action(frames=[obs], latest_frame=obs)
            if action is None:
                break
            click_data = agent.get_pending_action_data()
            if click_data:
                try:
                    obs = env.step(action, data=click_data)
                except (KeyError, Exception) as e:
                    print(f"  Error performing step with action {action}: {e}")
                    break
            else:
                try:
                    obs = env.step(action)
                except (Exception) as e:
                    print(f"  Error performing step with action {action}: {e}")
                    break
            
            # Check if obs is valid
            if obs is None:
                print(f"  obs became None at step {step}, stopping")
                break
            
            step += 1

            if obs.levels_completed > levels:
                old_levels = levels
                levels = obs.levels_completed
                level_steps = step - level_start
                level_start = step
                baseline = baselines[old_levels] if old_levels < len(baselines) else 100
                rhae = min(115.0, (baseline / max(1, level_steps)) ** 2 * 100)
                result['per_level_rhae'].append(rhae)
                # Reset click solver for new level
                agent._click_solve_attempted = False
                agent._click_solution = None
                agent._click_solution_idx = 0
                # Don't print every click, just level completion
                print(f"  L{old_levels}: {level_steps} steps, baseline={baseline}, RHAE={rhae:.1f}")

        result['levels_completed'] = levels
        result['total_steps'] = step

    except Exception as e:
        result['error'] = str(e)
        traceback.print_exc()

    return result


def main():
    print("=" * 70)
    print("ARC-AGI-3 Full 25-Game Benchmark")
    print("=" * 70)

    all_results = []
    total_rhae = 0
    total_levels = 0
    max_possible_levels = 25 * 7  # 175

    for game_id, baselines in sorted(GAMES.items()):
        print(f"\n--- {game_id} ---")
        t0 = time.time()
        result = run_game(game_id, baselines)
        elapsed = time.time() - t0
        result['elapsed'] = elapsed

        game_rhae = sum(result['per_level_rhae'])
        print(f"  Result: {result['levels_completed']} levels, "
              f"RHAE={game_rhae:.1f}, "
              f"oracle={result['oracle']}, "
              f"adapter={result['adapter']}, "
              f"time={elapsed:.1f}s")
        if result['error']:
            print(f"  ERROR: {result['error']}")

        all_results.append(result)
        total_rhae += game_rhae
        total_levels += result['levels_completed']

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Game':<8} {'Levels':<8} {'RHAE':<8} {'Oracle':<8} {'Adapter':<16} {'Time':<8}")
    print("-" * 60)
    for r in all_results:
        rhae = sum(r['per_level_rhae'])
        print(f"{r['game_id']:<8} {r['levels_completed']:<8} {rhae:<8.1f} "
              f"{'Y' if r['oracle'] else 'N':<8} {r['adapter'] or 'N/A':<16} {r.get('elapsed', 0):<8.1f}")
    print("-" * 60)
    print(f"{'TOTAL':<8} {total_levels:<8} {total_rhae:<8.1f}")
    print(f"\nMax possible: {max_possible_levels} levels, 2875 RHAE")
    print(f"Achieved: {total_levels} levels, {total_rhae:.1f} RHAE")
    print(f"Coverage: {total_levels}/{max_possible_levels} = {total_levels/max_possible_levels*100:.1f}%")


if __name__ == "__main__":
    main()
