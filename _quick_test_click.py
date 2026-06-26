"""Quick test for click game solvers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from arcengine import GameAction
from agent.planner_agent import PlannerAgent
from agent.game_configs import GAME_CONFIGS
import arc_agi

MAX_STEPS = 200
STAGNATION = 20

# Only test s5i5 first (fastest to debug)
TEST_GAMES = ["s5i5"]  # "r11l", "tn36", "vc33", "su15"]

for game_id in TEST_GAMES:
    print(f"\n{'='*60}")
    print(f"Testing {game_id}")
    print(f"{'='*60}")

    try:
        arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
        env = arc.make(game_id)
        obs = env.step(GameAction.RESET)

        agent = PlannerAgent(
            env=env,
            level_baselines=[],
            max_steps=MAX_STEPS,
            stagnation_threshold=STAGNATION,
            use_oracle=True,
            game_id=game_id,
        )
        agent._check_oracle_availability()
        print(f"  Oracle: {agent.use_oracle}, Adapter: {type(agent._oracle_adapter).__name__}")

        levels = 0
        step = 0

        while step < MAX_STEPS and levels < 4:
            action = agent.choose_action(frames=[obs], latest_frame=obs)
            if action is None:
                print(f"  Agent returned None at step {step}")
                break

            click_data = agent.get_pending_action_data()
            if click_data:
                obs = env.step(action, data=click_data)
            else:
                obs = env.step(action)

            if obs is None:
                print(f"  obs became None at step {step}")
                break

            step += 1
            new_levels = obs.levels_completed
            if new_levels > levels:
                print(f"  ✅ Level {levels}→{new_levels} at step {step}")
                levels = new_levels

        print(f"  Result: {levels} levels in {step} steps")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ERROR: {e}")
