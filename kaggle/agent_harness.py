"""
Kaggle Competition Harness for TOMAS ARC-AGI-3 Agent.

This script runs the TOMAS agent against all 25 ARC-AGI-3 games
in competition mode and generates the submission.

Usage:
    python kaggle/agent_harness.py              # Run all games
    python kaggle/agent_harness.py --game ls20  # Run single game
    python kaggle/agent_harness.py --competition # Competition mode
"""
import argparse
import sys
import time
import numpy as np

def run_agent_on_game(game_name: str, max_steps: int = 500, competition: bool = False):
    """Run the TOMAS agent on a single game.
    
    Args:
        game_name: Name of the game (e.g., "ls20")
        max_steps: Maximum number of actions to take
        competition: If True, use competition mode
        
    Returns:
        Dictionary with results
    """
    import arc_agi
    from arcengine import GameAction, GameState
    from src.agent import TomasAgent

    # Create arcade in appropriate mode
    if competition:
        arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION)
    else:
        arc = arc_agi.Arcade()

    # Create environment
    env = arc.make(game_name)
    
    # Create agent
    agent = TomasAgent()
    
    # Track results
    frames = []
    start_time = time.time()
    levels_completed = 0
    total_actions = 0
    resets = 0
    
    print(f"\n{'='*60}")
    print(f"Running TOMAS Agent on {game_name}")
    print(f"  Available actions: {env.action_space}")
    print(f"  Max steps: {max_steps}")
    print(f"{'='*60}")
    
    # Initial step to get first frame
    obs = env.step(GameAction.ACTION1)
    frames.append(obs)
    total_actions += 1
    
    for step in range(max_steps - 1):
        # Check if done
        if agent.is_done(frames, obs):
            print(f"  Agent says DONE at step {step + 1}")
            break
        
        # Choose action
        action = agent.choose_action(frames, obs)
        
        # Execute action
        obs = env.step(action)
        frames.append(obs)
        total_actions += 1
        
        if action == GameAction.RESET:
            resets += 1
        
        # Track level progress
        if obs.levels_completed > levels_completed:
            print(f"  Level {obs.levels_completed}/{obs.win_levels} completed at step {step + 1}")
            levels_completed = obs.levels_completed
        
        # Check game state
        if obs.state == GameState.WIN:
            print(f"  WIN! Completed all {obs.win_levels} levels in {step + 1} steps")
            break
        elif obs.state == GameState.GAME_OVER:
            print(f"  GAME OVER at step {step + 1}, attempting RESET")
            # Don't break - let agent handle it
        
        # Progress update every 50 steps
        if (step + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step + 1}/{max_steps} | Levels: {obs.levels_completed}/{obs.win_levels} | "
                  f"Actions: {total_actions} | Resets: {resets} | Time: {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    
    # Get scorecard
    try:
        scorecard = arc.get_scorecard()
        score = scorecard.score if hasattr(scorecard, 'score') else 0.0
    except:
        score = 0.0
    
    result = {
        "game": game_name,
        "levels_completed": obs.levels_completed,
        "win_levels": obs.win_levels,
        "won": obs.state == GameState.WIN,
        "total_actions": total_actions,
        "resets": resets,
        "time_seconds": round(elapsed, 2),
        "score": score,
    }
    
    print(f"\n  Result: {result}")
    return result


def run_all_games(max_steps: int = 500, competition: bool = False):
    """Run the agent on all 25 games."""
    import arc_agi
    
    arc = arc_agi.Arcade()
    games = [e.title.lower() for e in arc.available_environments]
    
    print(f"Running on {len(games)} games: {games}")
    
    results = []
    total_score = 0.0
    total_levels = 0
    total_wins = 0
    
    for game in games:
        try:
            result = run_agent_on_game(game, max_steps, competition)
            results.append(result)
            total_score += result["score"]
            total_levels += result["levels_completed"]
            if result["won"]:
                total_wins += 1
        except Exception as e:
            print(f"  ERROR on {game}: {e}")
            results.append({"game": game, "error": str(e)})
    
    avg_score = total_score / len(games) if games else 0
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Games played: {len(results)}")
    print(f"  Games won: {total_wins}")
    print(f"  Total levels completed: {total_levels}")
    print(f"  Average score: {avg_score:.2f}")
    print(f"\n  Per-game results:")
    for r in results:
        if "error" in r:
            print(f"    {r['game']:6s}: ERROR - {r['error']}")
        else:
            print(f"    {r['game']:6s}: {r['levels_completed']}/{r['win_levels']} levels, "
                  f"score={r['score']:.1f}, actions={r['total_actions']}, "
                  f"{'WON' if r['won'] else 'lost'}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOMAS ARC-AGI-3 Agent Harness")
    parser.add_argument("--game", type=str, default=None, help="Single game to run")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per game")
    parser.add_argument("--competition", action="store_true", help="Use competition mode")
    args = parser.parse_args()
    
    if args.game:
        run_agent_on_game(args.game, args.max_steps, args.competition)
    else:
        run_all_games(args.max_steps, args.competition)
