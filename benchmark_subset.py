"""Run benchmark on a subset of games to understand current state."""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import arc_agi
from arcengine import GameAction
from agent.planner_agent import PlannerAgent

# Subset of games: working, partially working, failing
GAMES_SUBSET = {
    # Known working
    'ls20': [22, 123, 73, 84, 96, 192, 186],
    'ft09': [5, 7, 14, 16, 30, 50, 70],
    'tu93': [30, 50, 70, 90, 110, 130, 150],
    # Partially working
    'tr87': [30, 50, 70, 90, 110, 130, 150],
    # Known failing
    'g50t': [30, 50, 70, 90, 110, 130, 150],
    'su15': [30, 50, 70, 90, 110, 130, 150],
}

MAX_STEPS = 1000
STAGNATION = 500


def run_game(game_id: str, baselines: list) -> dict:
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
        arc = arc_agi.Arcade()
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
                obs = env.step(action, data=click_data)
            else:
                obs = env.step(action)
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
                print(f"  L{old_levels}: {level_steps} steps, baseline={baseline}, RHAE={rhae:.1f}")

        result['levels_completed'] = levels
        result['total_steps'] = step

    except Exception as e:
        result['error'] = str(e)
        traceback.print_exc()

    return result


def main():
    print("=" * 70)
    print("ARC-AGI-3 Subset Benchmark (6 games)")
    print("=" * 70)

    all_results = []
    total_rhae = 0
    total_levels = 0

    for game_id, baselines in sorted(GAMES_SUBSET.items()):
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
    print(f"\n{'Game':<8} {'Levels':<8} {'RHAE':<10} {'Oracle':<8} {'Adapter':<20} {'Time':<8}")
    print("-" * 70)
    for r in all_results:
        rhae = sum(r['per_level_rhae'])
        print(f"{r['game_id']:<8} {r['levels_completed']:<8} {rhae:<10.1f} "
              f"{'Y' if r['oracle'] else 'N':<8} {(r['adapter'] or 'N/A')[:20]:<20} {r.get('elapsed', 0):<8.1f}")
    print("-" * 70)
    print(f"{'TOTAL':<8} {total_levels:<8} {total_rhae:<10.1f}")
    print(f"\nSubset max possible: {len(GAMES_SUBSET) * 7} levels, {len(GAMES_SUBSET) * 115 * 7 / 7:.1f} RHAE")
    print(f"Achieved: {total_levels} levels, {total_rhae:.1f} RHAE")


if __name__ == "__main__":
    main()
