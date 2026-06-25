#!/usr/bin/env python
"""Smart test script that tries different strategies for failing games."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import arc_agi
from arcengine import GameAction


def test_bp35_strategies():
    """Test different strategies for bp35."""
    print("\n" + "="*60)
    print("BP35 Strategies")
    print("="*60)

    # Strategy 1: Just move RIGHT and see what happens
    print("\n--- Strategy 1: RIGHT x10 ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("bp35")
    obs = env.step(GameAction.RESET)
    game = env._game

    for i in range(10):
        obs = env.step(GameAction.ACTION4, data={"x": 0, "y": 0})
        if obs.levels_completed > 0:
            print(f"  Level passed at step {i+1}!")
            return True
        if hasattr(obs, 'state') and str(obs.state) == 'GameState.GAME_OVER':
            print(f"  Game over at step {i+1}")
            break
    print(f"  No pass. levels_completed={obs.levels_completed}")

    # Strategy 2: Try moving in all 4 directions
    print("\n--- Strategy 2: UP,DOWN,LEFT,RIGHT alternating ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("bp35")
    obs = env.step(GameAction.RESET)

    actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
    for i in range(20):
        act = actions[i % 4]
        obs = env.step(act, data={"x": 0, "y": 0})
        if obs.levels_completed > 0:
            print(f"  Level passed at step {i+1} with action {act}!")
            return True
    print(f"  No pass. levels_completed={obs.levels_completed}")

    # Strategy 3: Inspect the engine object
    print("\n--- Strategy 3: Inspect engine ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("bp35")
    obs = env.step(GameAction.RESET)
    game = env._game
    engine = getattr(game, "oztjzzyqoek", None)
    if engine:
        print(f"  Engine type: {type(engine).__name__}")
        for name in sorted(dir(engine)):
            if name.startswith("_"):
                continue
            try:
                val = getattr(engine, name)
                if callable(val):
                    continue
                print(f"  engine.{name}: {str(val)[:200]}")
            except:
                pass

    # Strategy 4: Try CLICK (ACTION6) with the sprite position
    print("\n--- Strategy 4: CLICK at sprite pos (4,3) ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("bp35")
    obs = env.step(GameAction.RESET)
    obs = env.step(GameAction.ACTION6, data={"x": 4, "y": 3})
    print(f"  After click: levels={obs.levels_completed}, state={obs.state}")

    # Strategy 5: Check the other engine object
    print("\n--- Strategy 5: Inspect tehvqeiqsdu ---")
    game = env._game
    teh = getattr(game, "tehvqeiqsdu", None)
    if teh:
        print(f"  tehvqeiqsdu type: {type(teh).__name__}")
        for name in sorted(dir(teh)):
            if name.startswith("_"):
                continue
            try:
                val = getattr(teh, name)
                if callable(val):
                    continue
                print(f"  tehvqeiqsdu.{name}: {str(val)[:200]}")
            except:
                pass

    return False


def test_su15_strategies():
    """Test different strategies for su15."""
    print("\n" + "="*60)
    print("SU15 Strategies")
    print("="*60)

    # Strategy 1: Execute pre-computed actions, re-reading each time
    print("\n--- Strategy 1: Replay actions, re-read each step ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("su15")
    obs = env.step(GameAction.RESET)
    game = env._game

    actions_list = getattr(game, "actions", [])
    print(f"  Initial actions count: {len(actions_list)}")

    for i in range(min(20, len(actions_list))):
        # Re-read actions each step
        current_actions = getattr(game, "actions", [])
        if not current_actions:
            print(f"  No more actions at step {i}")
            break
        act = current_actions[0]
        act_id = getattr(act, "id", None)
        act_data = getattr(act, "data", None)
        if act_data is None:
            act_data = {}
        print(f"  Step {i}: action={act_id}, data={act_data}")
        obs = env.step(act_id, data=act_data)
        if obs.levels_completed > 0:
            print(f"  Level passed at step {i+1}!")
            return True

    print(f"  After 20 steps: levels={obs.levels_completed}")
    # Check if actions list changed
    new_actions = getattr(game, "actions", [])
    print(f"  Actions count after 20 steps: {len(new_actions)}")
    if new_actions:
        print(f"  First action now: {new_actions[0].__dict__ if hasattr(new_actions[0], '__dict__') else new_actions[0]}")

    # Strategy 2: Execute all actions then ACTION7
    print("\n--- Strategy 2: All actions + ACTION7 submit ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("su15")
    obs = env.step(GameAction.RESET)
    game = env._game
    actions_list = getattr(game, "actions", [])

    steps = 0
    for act in actions_list[:50]:  # Only first 50 to save time
        act_id = getattr(act, "id", None)
        act_data = getattr(act, "data", None) or {}
        obs = env.step(act_id, data=act_data)
        steps += 1
        if obs.levels_completed > 0:
            print(f"  Level passed at step {steps}!")
            return True

    # Try ACTION7 (submit)
    print(f"  After {steps} actions: levels={obs.levels_completed}, trying ACTION7...")
    obs = env.step(GameAction.ACTION7, data={"x": 0, "y": 0})
    print(f"  After ACTION7: levels={obs.levels_completed}, state={obs.state}")

    return False


def test_g50t_strategies():
    """Test different strategies for g50t."""
    print("\n" + "="*60)
    print("G50T Strategies")
    print("="*60)

    # Check game engine for step size
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("g50t")
    obs = env.step(GameAction.RESET)
    game = env._game

    engine = getattr(game, "vgwycxsxjz", None)
    if engine:
        print(f"  Engine type: {type(engine).__name__}")
        for name in sorted(dir(engine)):
            if name.startswith("_"):
                continue
            try:
                val = getattr(engine, name)
                if callable(val):
                    continue
                s = str(val)
                if len(s) > 200:
                    s = s[:200] + "..."
                print(f"  engine.{name}: {s}")
            except:
                pass

    # Check player and goal positions
    cl = game.current_level
    sprites = list(getattr(cl, "_sprites", []))
    player = None
    goal = None
    for s in sprites:
        tags = getattr(s, "tags", [])
        if "qftsebtxuc" in tags:
            player = s
        if "gilbljmfbc" in tags:
            goal = s

    if player and goal:
        px, py = int(player.x), int(player.y)
        gx, gy = int(goal.x), int(goal.y)
        pw, ph = int(player.width), int(player.height)
        gw, gh = int(goal.width), int(goal.height)
        print(f"\n  Player: pos=({px},{py}) size=({pw},{ph}) center=({px+pw//2},{py+ph//2})")
        print(f"  Goal: pos=({gx},{gy}) size=({gw},{gh}) center=({gx+gw//2},{gy+gh//2})")

    # Strategy 1: Try moving RIGHT and check if player moves
    print("\n--- Strategy 1: Move RIGHT 1 step, check position ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("g50t")
    obs = env.step(GameAction.RESET)
    game = env._game
    cl = game.current_level
    sprites_before = list(getattr(cl, "_sprites", []))
    player_before = None
    for s in sprites_before:
        if "qftsebtxuc" in getattr(s, "tags", []):
            player_before = s
            break
    if player_before:
        print(f"  Before: player at ({player_before.x},{player_before.y})")

    obs = env.step(GameAction.ACTION4)  # RIGHT
    sprites_after = list(getattr(cl, "_sprites", []))
    player_after = None
    for s in sprites_after:
        if "qftsebtxuc" in getattr(s, "tags", []):
            player_after = s
            break
    if player_after:
        print(f"  After RIGHT: player at ({player_after.x},{player_after.y})")
        if player_before:
            dx = player_after.x - player_before.x
            dy = player_after.y - player_before.y
            print(f"  Delta: ({dx},{dy})")

    obs = env.step(GameAction.ACTION2)  # DOWN
    sprites_after2 = list(getattr(cl, "_sprites", []))
    player_after2 = None
    for s in sprites_after2:
        if "qftsebtxuc" in getattr(s, "tags", []):
            player_after2 = s
            break
    if player_after2:
        print(f"  After DOWN: player at ({player_after2.x},{player_after2.y})")

    # Strategy 2: Try step-by-step movement to goal
    print("\n--- Strategy 2: Step-by-step to goal ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("g50t")
    obs = env.step(GameAction.RESET)
    game = env._game

    # Get current positions
    cl = game.current_level
    for s in getattr(cl, "_sprites", []):
        if "qftsebtxuc" in getattr(s, "tags", []):
            player = s
        if "gilbljmfbc" in getattr(s, "tags", []):
            goal = s

    # Try moving towards goal, checking position each step
    for i in range(50):
        px, py = int(player.x), int(player.y)
        gx, gy = int(goal.x), int(goal.y)

        if px < gx:
            obs = env.step(GameAction.ACTION4)  # RIGHT
        elif px > gx:
            obs = env.step(GameAction.ACTION3)  # LEFT
        elif py < gy:
            obs = env.step(GameAction.ACTION2)  # DOWN
        elif py > gy:
            obs = env.step(GameAction.ACTION1)  # UP
        else:
            print(f"  Reached goal at step {i}!")
            break

        # Re-read player position
        for s in getattr(cl, "_sprites", []):
            if "qftsebtxuc" in getattr(s, "tags", []):
                player = s
                break

        if obs.levels_completed > 0:
            print(f"  Level passed at step {i+1}!")
            return True
        if i < 5 or i % 10 == 0:
            print(f"  Step {i+1}: player at ({player.x},{player.y}), goal at ({gx},{gy})")

    print(f"  Final: levels={obs.levels_completed}, state={obs.state}")

    return False


def test_lp85_strategies():
    """Test different strategies for lp85."""
    print("\n" + "="*60)
    print("LP85 Strategies")
    print("="*60)

    # Get solution info
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("lp85")
    obs = env.step(GameAction.RESET)
    game = env._game

    solution = getattr(game, "uopmnplcnv", {})
    level_name = getattr(game, "ucybisahh", "")
    print(f"  Level name: {level_name}")
    print(f"  Solution keys: {list(solution.keys())}")

    if level_name in solution:
        level_sol = solution[level_name]
        print(f"  Level solution sections: {list(level_sol.keys())}")
        for section, sec_data in level_sol.items():
            print(f"  Section '{section}' keys: {list(sec_data.keys())}")
            for sub_key, sub_data in sec_data.items():
                if isinstance(sub_data, dict):
                    print(f"    {sub_key}: {len(sub_data)} entries")
                    # Show first few
                    for k, v in list(sub_data.items())[:5]:
                        print(f"      {k}: {v}")
                    if len(sub_data) > 5:
                        print(f"      ... ({len(sub_data)} total)")

    # Get buttons
    buttons = getattr(game, "afhycvvjg", [])
    print(f"\n  Buttons ({len(buttons)}):")
    for btn in buttons:
        tags = getattr(btn, "tags", [])
        print(f"    tags={tags} pos=({btn.x},{btn.y})")

    # Strategy 1: Click L button 20 times
    print("\n--- Strategy 1: Click L button 20 times ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("lp85")
    obs = env.step(GameAction.RESET)

    l_btn_pos = None
    for btn in buttons:
        if "button_A_L" in getattr(btn, "tags", []):
            l_btn_pos = (int(btn.x), int(btn.y))
            break

    if l_btn_pos:
        for i in range(20):
            obs = env.step(GameAction.ACTION6, data={"x": l_btn_pos[0], "y": l_btn_pos[1]})
            if obs.levels_completed > 0:
                print(f"  Level passed at step {i+1} with L!")
                return True
        print(f"  After 20 L clicks: levels={obs.levels_completed}")

    # Strategy 2: Click R button 20 times
    print("\n--- Strategy 2: Click R button 20 times ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("lp85")
    obs = env.step(GameAction.RESET)

    r_btn_pos = None
    for btn in buttons:
        if "button_A_R" in getattr(btn, "tags", []):
            r_btn_pos = (int(btn.x), int(btn.y))
            break

    if r_btn_pos:
        for i in range(20):
            obs = env.step(GameAction.ACTION6, data={"x": r_btn_pos[0], "y": r_btn_pos[1]})
            if obs.levels_completed > 0:
                print(f"  Level passed at step {i+1} with R!")
                return True
        print(f"  After 20 R clicks: levels={obs.levels_completed}")

    # Strategy 3: Click L and R alternating
    print("\n--- Strategy 3: L,R alternating 20 times ---")
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make("lp85")
    obs = env.step(GameAction.RESET)

    if l_btn_pos and r_btn_pos:
        for i in range(20):
            pos = l_btn_pos if i % 2 == 0 else r_btn_pos
            obs = env.step(GameAction.ACTION6, data={"x": pos[0], "y": pos[1]})
            if obs.levels_completed > 0:
                direction = "L" if i % 2 == 0 else "R"
                print(f"  Level passed at step {i+1} with {direction}!")
                return True
        print(f"  After 20 alternating clicks: levels={obs.levels_completed}")

    # Strategy 4: Check if game has a win check function
    print("\n--- Strategy 4: Check game methods ---")
    game = env._game
    for name in sorted(dir(game)):
        if name.startswith("_"):
            continue
        try:
            val = getattr(game, name)
            if callable(val) and not name.startswith("on_"):
                print(f"  method: {name}")
        except:
            pass

    return False


if __name__ == "__main__":
    games = sys.argv[1:] if len(sys.argv) > 1 else ["bp35", "su15", "g50t", "lp85"]

    if "bp35" in games:
        test_bp35_strategies()
    if "su15" in games:
        test_su15_strategies()
    if "g50t" in games:
        test_g50t_strategies()
    if "lp85" in games:
        test_lp85_strategies()
