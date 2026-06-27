"""L2 solver decision trace: Follow solver's BFS calls, coin collection, and step budget."""

import sys, os, time, copy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction, ActionInput
from src.agent.game_solvers import solve_ls20, _detect_direction_mapping, _detect_game_step
from src.agent.oracle_adapters import get_oracle_adapter

arc = arc_agi.Arcade()
env = arc.make("ls20")
env.reset()
obs = env.step(GameAction.ACTION1)
game = env._game

# Solve L0 and L1
for lvl in range(2):
    plan = solve_ls20(game, lvl)
    if plan:
        for aid, data in plan:
            game.perform_action(ActionInput(id=aid, data=data if data else {}))
    else:
        print(f"L{lvl} FAILED")
        sys.exit(1)

print(f"L2 reached (level={game._current_level_index})")

# Now instrument the solver's main loop by patching _lightweight_bfs
# We'll add a print statement to see what BFS calls are made and their results

import src.agent.game_solvers as gs_module
from collections import deque

# Save original _lightweight_bfs
original_lightweight_bfs = gs_module._solve_ls20_delta_state_bfs

def instrumented_solver(game, level_idx):
    """Instrumented version of _solve_ls20_delta_state_bfs with decision logging."""
    # This is too complex to rewrite. Let's instead use a simpler approach:
    # Run the solver and trace the game state changes.
    sim = copy.deepcopy(game)
    result = solve_ls20(game, level_idx)
    return result

# Instead, let's manually simulate the solver's key decisions
adapter = get_oracle_adapter("ls20", game)
px, py = int(adapter.player.x), int(adapter.player.y)
remaining = adapter.steps_remaining
state = adapter.player_state
step_size = _detect_game_step(game)
dir_offsets = _detect_direction_mapping(game)

print(f"\nL2 start: pos=({px},{py}), remaining={remaining}, state={state}")
print(f"  step_size={step_size}, dir_offsets={dir_offsets}")
print(f"  goals={[(int(g.x),int(g.y)) for g in adapter.goals or []]}")
print(f"  goal_reqs={adapter.goal_requirements}")
print(f"  changers={[(d, [(int(c.x),int(c.y)) for c in ch]) for d,ch in adapter.state_changers.items()]}")
print(f"  coins={[(int(c.x),int(c.y)) for c in adapter.coins or []]}")

# Step 1: Check if remaining is sufficient for BFS to nearest changer
wall_positions = set((int(w.x), int(w.y)) for w in adapter.walls or [])

# BFS to rotation changer (49,10) with max_steps=remaining
print(f"\n--- BFS to rotation changer (49,10) with max_steps={remaining} ---")
target_ch = (49, 10)
other_changers = set()
for dim, chs in adapter.state_changers.items():
    for ch in chs:
        ch_pos = (int(ch.x), int(ch.y))
        if ch_pos != target_ch:
            other_changers.add(ch_pos)

blocked = set(wall_positions)
blocked.update(other_changers)
blocked.discard(target_ch)

visited = {(px, py)}
visited.update(blocked)
queue = deque([([], px, py)])
path_to_changer = None
nodes_explored = 0

while queue:
    ca, cx, cy = queue.popleft()
    if len(ca) >= remaining:  # max_steps = remaining
        continue
    if cx == target_ch[0] and cy == target_ch[1]:
        path_to_changer = ca
        break
    for d, (dx, dy) in dir_offsets.items():
        nx, ny = cx + dx, cy + dy
        if (nx, ny) in visited:
            continue
        visited.add((nx, ny))
        queue.append((ca + [(d, None)], nx, ny))
        nodes_explored += 1

if path_to_changer:
    print(f"  ✅ Found path: {len(path_to_changer)} steps (within remaining={remaining})")
    print(f"     Nodes explored: {nodes_explored}")
else:
    print(f"  ❌ No path found within remaining={remaining} steps")
    print(f"     Nodes explored: {nodes_explored}")

# BFS to color changer (29,45) with max_steps=remaining
print(f"\n--- BFS to color changer (29,45) with max_steps={remaining} ---")
target_color = (29, 45)
other_changers2 = set()
for dim, chs in adapter.state_changers.items():
    for ch in chs:
        ch_pos = (int(ch.x), int(ch.y))
        if ch_pos != target_color:
            other_changers2.add(ch_pos)

blocked2 = set(wall_positions)
blocked2.update(other_changers2)
blocked2.discard(target_color)

visited2 = {(px, py)}
visited2.update(blocked2)
queue2 = deque([([], px, py)])
path_to_color_changer = None
nodes_explored2 = 0

while queue2:
    ca, cx, cy = queue2.popleft()
    if len(ca) >= remaining:
        continue
    if cx == target_color[0] and cy == target_color[1]:
        path_to_color_changer = ca
        break
    for d, (dx, dy) in dir_offsets.items():
        nx, ny = cx + dx, cy + dy
        if (nx, ny) in visited2:
            continue
        visited2.add((nx, ny))
        queue2.append((ca + [(d, None)], nx, ny))
        nodes_explored2 += 1

if path_to_color_changer:
    print(f"  ✅ Found path: {len(path_to_color_changer)} steps (within remaining={remaining})")
else:
    print(f"  ❌ No path found within remaining={remaining} steps")

# BFS to nearest coin with max_steps=remaining
print(f"\n--- BFS to nearest coin ---")
from src.agent.game_solvers import _compute_coin_proximity_position
for coin_x, coin_y in [(35, 16), (20, 31)]:
    prox = _compute_coin_proximity_position(px, py, step_size, coin_x, coin_y)
    blocked3 = set(wall_positions)
    visited3 = {(px, py)}
    visited3.update(blocked3)
    queue3 = deque([([], px, py)])
    coin_path = None
    while queue3:
        ca, cx, cy = queue3.popleft()
        if len(ca) >= remaining:
            continue
        if cx == prox[0] and cy == prox[1]:
            coin_path = ca
            break
        for d, (dx, dy) in dir_offsets.items():
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited3:
                continue
            visited3.add((nx, ny))
            queue3.append((ca + [(d, None)], nx, ny))
    if coin_path:
        print(f"  Coin ({coin_x},{coin_y}) → prox({prox[0]},{prox[1]}): "
              f"{len(coin_path)} steps ✅ (fits in remaining={remaining})")
    else:
        print(f"  Coin ({coin_x},{coin_y}) → prox({prox[0]},{prox[1]}): "
              f"NO PATH ❌")

# BFS to goal (54,50) avoiding ALL changers
print(f"\n--- BFS to goal (54,50) avoiding all changers, max_steps={remaining} ---")
all_changers = set()
for dim, chs in adapter.state_changers.items():
    for ch in chs:
        all_changers.add((int(ch.x), int(ch.y)))

blocked4 = set(wall_positions)
blocked4.update(all_changers)
blocked4.discard((54, 50))

visited4 = {(px, py)}
visited4.update(blocked4)
queue4 = deque([([], px, py)])
goal_path = None
while queue4:
    ca, cx, cy = queue4.popleft()
    if len(ca) >= remaining:
        continue
    if cx == 54 and cy == 50:
        goal_path = ca
        break
    for d, (dx, dy) in dir_offsets.items():
        nx, ny = cx + dx, cy + dy
        if (nx, ny) in visited4:
            continue
        visited4.add((nx, ny))
        queue4.append((ca + [(d, None)], nx, ny))

if goal_path:
    print(f"  ✅ Found path to goal: {len(goal_path)} steps")
else:
    print(f"  ❌ No path to goal within remaining={remaining}")

# Now simulate: collect coin, then check if BFS to changer succeeds
print(f"\n--- Simulate: collect nearest coin then re-check BFS ---")
sim = copy.deepcopy(game)
adapter_sim = get_oracle_adapter("ls20", sim)
print(f"  Before: pos=({int(adapter_sim.player.x)},{int(adapter_sim.player.y)}), "
      f"remaining={adapter_sim.steps_remaining}")

# Manually navigate to coin (35,16) prox (34,15)
# Need to compute BFS path and execute it
prox1 = _compute_coin_proximity_position(
    int(adapter_sim.player.x), int(adapter_sim.player.y), step_size, 35, 16)
print(f"  Target coin prox: ({prox1[0]},{prox1[1]})")

# BFS to coin
wall_sim = set((int(w.x), int(w.y)) for w in adapter_sim.walls or [])
p_sim = (int(adapter_sim.player.x), int(adapter_sim.player.y))
vis = {p_sim}
vis.update(wall_sim)
q = deque([([], p_sim[0], p_sim[1])])
coin_bfs = None
while q:
    ca, cx, cy = q.popleft()
    if len(ca) >= adapter_sim.steps_remaining:
        continue
    if cx == prox1[0] and cy == prox1[1]:
        coin_bfs = ca
        break
    for d, (dx, dy) in dir_offsets.items():
        nx, ny = cx + dx, cy + dy
        if (nx, ny) in vis:
            continue
        vis.add((nx, ny))
        q.append((ca + [(d, None)], nx, ny))

if coin_bfs:
    print(f"  Coin path: {len(coin_bfs)} steps")
    for aid, data in coin_bfs:
        ai = ActionInput(id=aid, data=data if data else {})
        sim.perform_action(ai)
    
    post = get_oracle_adapter("ls20", sim)
    if post and post.player:
        new_px, new_py = int(post.player.x), int(post.player.y)
        new_remaining = post.steps_remaining
        new_state = post.player_state
        print(f"  After coin: pos=({new_px},{new_py}), remaining={new_remaining}, state={new_state}")
        
        # Check: can we now BFS to changer within new remaining?
        print(f"\n  --- BFS to rotation changer (49,10) with max_steps={new_remaining} ---")
        wall_new = set((int(w.x), int(w.y)) for w in post.walls or [])
        other_ch = set()
        for dim, chs in post.state_changers.items():
            for ch in chs:
                cp = (int(ch.x), int(ch.y))
                if cp != (49, 10):
                    other_ch.add(cp)
        bl = set(wall_new)
        bl.update(other_ch)
        bl.discard((49, 10))
        vis2 = {(new_px, new_py)}
        vis2.update(bl)
        q2 = deque([([], new_px, new_py)])
        bfs_after_coin = None
        while q2:
            ca, cx, cy = q2.popleft()
            if len(ca) >= new_remaining:
                continue
            if cx == 49 and cy == 10:
                bfs_after_coin = ca
                break
            for d, (dx, dy) in dir_offsets.items():
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in vis2:
                    continue
                vis2.add((nx, ny))
                q2.append((ca + [(d, None)], nx, ny))
        
        if bfs_after_coin:
            print(f"  ✅ Path to changer: {len(bfs_after_coin)} steps (within remaining={new_remaining})")
        else:
            print(f"  ❌ No path to changer within remaining={new_remaining}")
else:
    print(f"  ❌ No path to coin")
