"""Diagnostic for L1 failures: ls20, lp85, s5i5 — direct Oracle replay test."""
import sys
sys.path.insert(0, '.')
import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_game, _solve_oracle_replay, _is_level_solved, SOLVERS
from src.agent.oracle_adapters import get_oracle_adapter
import copy, time

arc = Arcade(operation_mode=OperationMode.OFFLINE)

games_to_test = ['ls20', 'lp85', 's5i5']
max_level = 2  # Test L0 and L1

for gid in games_to_test:
    print(f"\n{'='*60}")
    print(f"  Diagnostic: {gid}")
    print(f"{'='*60}")
    
    for li in range(max_level):
        env = arc.make(gid)
        obs = env.step(GameAction.RESET)
        g = env._game
        
        if li > 0:
            g._current_level_index = li
            if hasattr(g, 'on_set_level'):
                g.on_set_level(g._levels[li])
        
        print(f"\n--- {gid} Level {li} ---")
        print(f"  _current_level_index: {g._current_level_index}")
        print(f"  _levels count: {len(g._levels)}")
        
        # Test Oracle adapter directly
        adapter = get_oracle_adapter(gid, g)
        if adapter:
            player = adapter.player
            goals = adapter.goals if hasattr(adapter, 'goals') else []
            print(f"  Oracle adapter: {type(adapter).__name__}")
            print(f"  Player: {player}")
            print(f"  Goals: {goals}")
        else:
            print(f"  No Oracle adapter for {gid}")
        
        # Test SOLVERS dict directly (with correct level_idx)
        solver = SOLVERS.get(gid)
        if solver:
            print(f"  Solver: {solver.__name__}")
            t0 = time.time()
            try:
                game_copy = copy.deepcopy(g)
                plan = solver(game_copy, li)
                elapsed = time.time() - t0
                steps = len(plan) if plan else 0
                print(f"  Solver result: {steps} steps in {elapsed:.1f}s")
                if plan:
                    # Verify on pristine copy
                    g3 = copy.deepcopy(g)
                    original_level = g3._current_level_index
                    solved = False
                    for step in plan[:300]:
                        aid, data = step
                        ai = ActionInput(id=aid, data=data if data else {})
                        try:
                            g3.perform_action(ai)
                        except Exception as e:
                            print(f"    Action error at step: {e}")
                            break
                        if _is_level_solved(g3, original_level):
                            solved = True
                            break
                    print(f"  Verification: solved={solved}")
                    if not solved:
                        print(f"  FAILED — plan didn't solve level {li}")
                        # Show last few steps
                        for i, s in enumerate(plan[-5:]):
                            print(f"    Step {steps-5+i}: {s}")
                else:
                    print(f"  Solver returned None — no plan found")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  Solver EXCEPTION: {e} ({elapsed:.1f}s)")
        else:
            print(f"  No solver in SOLVERS dict for {gid}")
        
        # Also test solve_game with correct level_idx
        print(f"\n  Testing solve_game(g, '{gid}', level_idx={li})...")
        t0 = time.time()
        plan2 = solve_game(g, gid, level_idx=li)
        elapsed2 = time.time() - t0
        steps2 = len(plan2) if plan2 else 0
        print(f"  solve_game result: {steps2} steps in {elapsed2:.1f}s")
        
        if plan2:
            g4 = copy.deepcopy(g)
            original_level2 = g4._current_level_index
            solved2 = False
            for step in plan2[:300]:
                aid, data = step
                ai = ActionInput(id=aid, data=data if data else {})
                try:
                    g4.perform_action(ai)
                except:
                    pass
                if _is_level_solved(g4, original_level2):
                    solved2 = True
                    break
            print(f"  solve_game verification: solved={solved2}")
