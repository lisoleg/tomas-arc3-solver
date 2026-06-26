"""Test s5i5 through PlannerAgent with new solver."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from arcengine import GameAction
from agent.planner_agent import PlannerAgent
import arc_agi

MAX_STEPS = 100
MAX_LEVELS = 4

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
env = arc.make("s5i5")
obs = env.step(GameAction.RESET)

agent = PlannerAgent(
    env=env,
    level_baselines=[],
    max_steps=MAX_STEPS,
    stagnation_threshold=100,
    use_oracle=True,
    game_id="s5i5",
)
agent._check_oracle_availability()
print(f"Oracle: {agent.use_oracle}, Adapter: {type(agent._oracle_adapter).__name__}")

levels = 0
step = 0
last_level_step = 0

while step < MAX_STEPS and levels < MAX_LEVELS:
    action = agent.choose_action(frames=[obs], latest_frame=obs)
    if action is None:
        print(f"Agent returned None at step {step}")
        break

    click_data = agent.get_pending_action_data()
    if click_data:
        obs = env.step(action, data=click_data)
    else:
        obs = env.step(action)

    if obs is None:
        print(f"obs became None at step {step}")
        break

    step += 1
    new_levels = obs.levels_completed
    if new_levels > levels:
        print(f"  ✅ Level {levels}→{new_levels} at step {step} (+{step - last_level_step} steps)")
        levels = new_levels
        last_level_step = step

print(f"\nResult: {levels}/{MAX_LEVELS} levels in {step} steps")
if levels >= MAX_LEVELS:
    print("🎉 ALL LEVELS COMPLETED!")
elif levels > 0:
    print(f"Progress: {levels} levels, stuck at level {levels}")
else:
    print("❌ No levels completed")
