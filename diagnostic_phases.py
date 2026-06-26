"""Diagnostic: trace solve_game phases step-by-step for ls20/s5i5/lp85."""
import sys
sys.path.insert(0, '.')
import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import (
    SOLVERS, _is_level_solved, _get_valid_action_inputs,
    _snap_click_coordinates, solve_generic_bfs, solve_beam_search,
    solve_idfs, solve_generic_dfs, solve_generic_keyboard, solve_random_walk
)
from src.agent.game_profiles import ALL_GAME_BASELINES
import copy, time

arc = Arcade(operation_mode=OperationMode.OFFLINE)

def _normalize_plan(plan):
    if not plan:
        return None
    normalized = []
    for action, click_data in plan:
        if click_data is not None and isinstance(click_data, (tuple, list)):
            normalized.append((action, {"x": int(click_data[0]), "y": int(click_data[1])}))
        else:
            normalized.append((action, click_data))
    return normalized

def _verify_plan(plan, pristine_game, original_level):
    if not plan:
        return False
    from arcengine import ActionInput
    try:
        sim = copy.deepcopy(pristine_game)
        for aid, data in plan[:300]:
            ai = ActionInput(id=aid, data=data if data else {})
            sim.perform_action(ai)
            if _is_level_solved(sim, original_level):
                return True
        return _is_level_solved(sim, original_level)
    except Exception:
        return False

games_to_test = ['ls20', 's5i5', 'lp85']

for gid in games_to_test:
    print(f"\n{'='*60}")
    print(f"  Phase-by-Phase: {gid}")
    print(f"{'='*60}")
    
    for li in range(2):  # L0 and L1
        env = arc.make(gid)
        obs = env.step(GameAction.RESET)
        g = env._game
        if li > 0:
            g._current_level_index = li
            if hasattr(g, 'on_set_level'):
                g.on_set_level(g._levels[li])
        
        original_level = g._current_level_index
        pristine_game = copy.deepcopy(g)
        valid_actions = _get_valid_action_inputs(g)
        n_actions = len(valid_actions)
        
        print(f"\n--- {gid} Level {li} ---")
        print(f"  _current_level_index: {original_level}")
        print(f"  n_actions: {n_actions}, valid_actions: {valid_actions}")
        
        t_start = time.time()
        solved_phase = None
        
        # Phase 0: SOLVERS dict
        solver = SOLVERS.get(gid)
        if solver:
            t0 = time.time()
            try:
                game_copy = copy.deepcopy(g)
                plan = solver(game_copy, li)
                elapsed = time.time() - t0
                steps = len(plan) if plan else 0
                plan_n = _normalize_plan(plan)
                verified = _verify_plan(plan_n, pristine_game, original_level) if plan_n else False
                print(f"  Phase 0 (SOLVERS): {steps} steps, {elapsed:.1f}s, verified={verified}")
                if verified:
                    solved_phase = f"P0({elapsed:.1f}s)"
                    continue
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  Phase 0 (SOLVERS): EXCEPTION {type(e).__name__}: {e} ({elapsed:.1f}s)")
        
        # Phase 1: UniversalSolverPipeline
        t0 = time.time()
        try:
            from src.agent.universal_solver_pipeline import UniversalSolverPipeline
            pipeline = UniversalSolverPipeline(g, gid, max_time=min(30.0, 55.0 - (time.time() - t_start)))
            plan = pipeline.solve()
            elapsed = time.time() - t0
            steps = len(plan) if plan else 0
            plan_n = _normalize_plan(plan)
            verified = _verify_plan(plan_n, pristine_game, original_level) if plan_n else False
            print(f"  Phase 1 (Pipeline): {steps} steps, {elapsed:.1f}s, verified={verified}")
            if verified:
                solved_phase = f"P1({elapsed:.1f}s)"
                continue
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  Phase 1 (Pipeline): EXCEPTION {type(e).__name__}: {str(e)[:80]} ({elapsed:.1f}s)")
        
        # Phase 2: BFS
        if n_actions <= 7:
            t0 = time.time()
            try:
                bfs_time = 20.0 if n_actions <= 4 else 15.0
                bfs_depth = 40 if n_actions <= 4 else 30
                plan = solve_generic_bfs(g, max_depth=bfs_depth, max_nodes=300000, max_time=min(bfs_time, 50.0 - (time.time() - t_start)))
                elapsed = time.time() - t0
                steps = len(plan) if plan else 0
                plan_n = _normalize_plan(plan)
                verified = _verify_plan(plan_n, pristine_game, original_level) if plan_n else False
                print(f"  Phase 2 (BFS): {steps} steps, {elapsed:.1f}s, verified={verified}")
                if verified:
                    solved_phase = f"P2({elapsed:.1f}s)"
                    continue
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  Phase 2 (BFS): EXCEPTION ({elapsed:.1f}s)")
        
        # Phase 3: Beam search
        t0 = time.time()
        try:
            remaining = 45.0 - (time.time() - t_start)
            if remaining > 2.0:
                plan = solve_beam_search(g, max_depth=80, beam_width=12, max_time=min(6.0, remaining - 1.0))
                elapsed = time.time() - t0
                steps = len(plan) if plan else 0
                plan_n = _normalize_plan(plan)
                verified = _verify_plan(plan_n, pristine_game, original_level) if plan_n else False
                print(f"  Phase 3 (Beam): {steps} steps, {elapsed:.1f}s, verified={verified}")
                if verified:
                    solved_phase = f"P3({elapsed:.1f}s)"
                    continue
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  Phase 3 (Beam): EXCEPTION ({elapsed:.1f}s)")
        
        # Phase 4: IDFS
        t0 = time.time()
        try:
            remaining = 50.0 - (time.time() - t_start)
            if remaining > 3.0:
                plan = solve_idfs(g, max_time=min(10.0, remaining - 2.0))
                elapsed = time.time() - t0
                steps = len(plan) if plan else 0
                plan_n = _normalize_plan(plan)
                verified = _verify_plan(plan_n, pristine_game, original_level) if plan_n else False
                print(f"  Phase 4 (IDFS): {steps} steps, {elapsed:.1f}s, verified={verified}")
                if verified:
                    solved_phase = f"P4({elapsed:.1f}s)"
                    continue
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  Phase 4 (IDFS): EXCEPTION ({elapsed:.1f}s)")
        
        # Phase 5: DFS
        t0 = time.time()
        try:
            remaining = 55.0 - (time.time() - t_start)
            if remaining > 2.0:
                plan = solve_generic_dfs(g, max_depth=15, max_nodes=30000, max_time=min(10.0, remaining - 1.0))
                elapsed = time.time() - t0
                steps = len(plan) if plan else 0
                plan_n = _normalize_plan(plan)
                verified = _verify_plan(plan_n, pristine_game, original_level) if plan_n else False
                print(f"  Phase 5 (DFS): {steps} steps, {elapsed:.1f}s, verified={verified}")
                if verified:
                    solved_phase = f"P5({elapsed:.1f}s)"
                    continue
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  Phase 5 (DFS): EXCEPTION ({elapsed:.1f}s)")
        
        if not solved_phase:
            print(f"  ** ALL PHASES FAILED for {gid} L{li} **")
        else:
            print(f"  ** SOLVED by {solved_phase} **")
