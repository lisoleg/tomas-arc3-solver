"""Test tr87 all levels with the cipher solver."""
import sys
sys.path.insert(0, ".")

import arc_agi
from arcengine import GameAction
import time

arc = arc_agi.Arcade()
env = arc.make("tr87")

# Import PlannerAgent
sys.path.insert(0, "src")
from agent.planner_agent import PlannerAgent

# Game profiles baselines
baselines = [54, 58, 40, 66, 55, 78]  # tr87 baselines per level

agent = PlannerAgent(
    env=env,
    level_baselines=baselines,
    max_steps=300,
    stagnation_threshold=300,
    use_oracle=True,
    game_id="tr87",
)

# Check oracle availability
agent._check_oracle_availability()
print(f"Oracle available: {agent._oracle_adapter is not None}")

obs = env.step(GameAction.RESET)
total_steps = 0
levels_completed = 0

for level_idx in range(6):
    print(f"\n{'='*60}")
    print(f"Level {level_idx}")
    print(f"{'='*60}")

    level_steps = 0
    max_steps = 256 if level_idx <= 4 else 300

    while level_steps < max_steps:
        # Check if level changed
        done = agent.is_done(frames=[obs], latest_frame=obs)
        if done:
            break

        # Choose action
        action = agent.choose_action(frames=[obs], latest_frame=obs)
        if action is None:
            print(f"  No action returned at step {level_steps}")
            break

        # Get click data if needed
        data = agent.get_pending_action_data()

        # Execute
        obs = env.step(action, data=data)
        level_steps += 1
        total_steps += 1

        # Check game over via levels_completed or state
        if hasattr(obs, 'state') and str(obs.state) == 'GameState.GAME_OVER':
            print(f"  GAME OVER at step {level_steps}")
            break

    if obs.levels_completed > level_idx:
        levels_completed += 1
        baseline = baselines[level_idx] if level_idx < len(baselines) else 50
        rhae = min(115.0, (baseline / max(level_steps, 1)) ** 2 * 100)
        print(f"  ✅ Level {level_idx} completed in {level_steps} steps "
              f"(baseline={baseline}, RHAE={rhae:.1f})")
    else:
        print(f"  ❌ Level {level_idx} NOT completed ({level_steps} steps)")

print(f"\n{'='*60}")
print(f"Total: {levels_completed}/6 levels, {total_steps} steps")
print(f"{'='*60}")
