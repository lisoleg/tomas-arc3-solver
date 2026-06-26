"""Verify s5i5 and lp85 L1 plans step-by-step to find failure point."""
import sys
sys.path.insert(0, '.')
import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import SOLVERS, _is_level_solved
import copy

arc = Arcade(operation_mode=OperationMode.OFFLINE)

# Test s5i5 L1
print("=" * 60)
print("  s5i5 L1 — Step-by-step plan replay")
print("=" * 60)

env = arc.make('s5i5')
obs = env.step(GameAction.RESET)
g = env._game
g._current_level_index = 1
if hasattr(g, 'on_set_level'):
    g.on_set_level(g._levels[1])

original_level = g._current_level_index

# Get solver plan
solver = SOLVERS.get('s5i5')
game_copy = copy.deepcopy(g)
plan = solver(game_copy, 1)
print(f"Plan: {plan}")
print(f"Plan length: {len(plan) if plan else 0}")

# Replay step by step
if plan:
    sim = copy.deepcopy(g)
    for i, (aid, data) in enumerate(plan):
        # Normalize data
        if isinstance(data, (tuple, list)):
            data = {"x": int(data[0]), "y": int(data[1])}
        elif not data:
            data = {}
        
        ai = ActionInput(id=aid, data=data)
        try:
            result = sim.perform_action(ai)
            solved = _is_level_solved(sim, original_level)
            # Get current blocks/targets positions
            cl = sim.current_level
            blocks = cl.get_sprites_by_tag("0064ocqkuqacti") if hasattr(cl, 'get_sprites_by_tag') else []
            targets = cl.get_sprites_by_tag("0087vvmblxkzdi") if hasattr(cl, 'get_sprites_by_tag') else []
            blk_pos = [(int(b.x), int(b.y)) for b in blocks]
            tgt_pos = [(int(t.x), int(t.y)) for t in targets]
            print(f"  Step {i}: action={aid}, data={data}, solved={solved}, blocks={blk_pos}, targets={tgt_pos}")
            if solved:
                print(f"  ** SOLVED at step {i}! **")
                break
        except Exception as e:
            print(f"  Step {i}: action={aid}, data={data}, EXCEPTION: {e}")
            break
    
    final_solved = _is_level_solved(sim, original_level)
    print(f"  Final: solved={final_solved}, level_index={sim._current_level_index}")

# Test lp85 L1
print("\n" + "=" * 60)
print("  lp85 L1 — Step-by-step plan replay")
print("=" * 60)

env = arc.make('lp85')
obs = env.step(GameAction.RESET)
g = env._game
g._current_level_index = 1
if hasattr(g, 'on_set_level'):
    g.on_set_level(g._levels[1])

original_level = g._current_level_index

# Get solver plan
solver = SOLVERS.get('lp85')
game_copy = copy.deepcopy(g)
plan = solver(game_copy, 1)
print(f"Plan length: {len(plan) if plan else 0}")
if plan:
    # Show first 10 and last 5 steps
    for i, s in enumerate(plan[:10]):
        print(f"  Step {i}: {s}")
    if len(plan) > 10:
        print(f"  ... ({len(plan) - 15} more steps) ...")
        for i, s in enumerate(plan[-5:]):
            print(f"  Step {len(plan)-5+i}: {s}")

# Replay step by step
if plan:
    sim = copy.deepcopy(g)
    for i, (aid, data) in enumerate(plan[:50]):
        if isinstance(data, (tuple, list)):
            data = {"x": int(data[0]), "y": int(data[1])}
        elif not data:
            data = {}
        
        ai = ActionInput(id=aid, data=data)
        try:
            result = sim.perform_action(ai)
            solved = _is_level_solved(sim, original_level)
            # Get button positions
            all_sprites = [s for s in sim.current_level._sprites] if hasattr(sim, 'current_level') and hasattr(sim.current_level, '_sprites') else []
            button_sprites = [s for s in all_sprites if any("button" in t.lower() for t in getattr(s, 'tags', []))]
            btn_pos = [(int(s.x), int(s.y), s.tags) for s in button_sprites[:6]]
            if i % 5 == 0 or solved:
                print(f"  Step {i}: action={aid}, data={data}, solved={solved}, buttons={btn_pos}")
            if solved:
                print(f"  ** SOLVED at step {i}! **")
                break
        except Exception as e:
            print(f"  Step {i}: EXCEPTION: {e}")
            break
    
    final_solved = _is_level_solved(sim, original_level)
    print(f"  Final: solved={final_solved}, level_index={sim._current_level_index}")
