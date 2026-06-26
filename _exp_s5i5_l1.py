"""Experiment: s5i5 Level 1 — pure LC(3) RIGHT strategy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ["ARCADE_NO_RENDER"] = "1"

from arcengine import GameAction, ClickData
import arc_agi

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
env = arc.make("s5i5")
obs = env.step(GameAction.RESET)

# Get game internals
game = env.game
from agent.game_solvers import _get_display_coords

# Level 0: quick solve (13 clicks via solver)
from agent.game_solvers import solve_s5i5
plan_l0 = solve_s5i5(game)
print(f"Level 0 plan: {len(plan_l0)} clicks")
for i, (lc, action_type) in enumerate(plan_l0):
    lc_x, lc_y = lc
    cx, cy = _get_display_coords(game, lc_x + lc.w - 2, lc_y + lc.h // 2, 1, 1)
    obs = env.step(GameAction.CLICK, data=ClickData(cx.item(), cy.item()))
    if obs is None:
        print(f"  obs None at click {i}")
        break

print(f"After L0: level={obs.levels_completed}")

# Level 1: Experiment — pure LC(3) RIGHT
# Chain: LC(3,54)→Sel(9,39)[RIGHT]→Sel(12,39)[UP]→LC(33,54)→Sel(12,36)[RIGHT]→Sel(15,36)[DOWN]→Block(15,36)
# LC(3,54) controls Sel(9,39): click right half → GROW RIGHT (+3 pixels)
# LC(33,54) or the LC controlling Sel(12,39) for UP

# Strategy: 12× RIGHT via LC(3), then 2× UP via the UP LC
# Let's find the UP LC...

# First, find all LCs and their selectors
print("\n=== Level 1 state ===")
level = game._levels[game._current_level_index]
print(f"Level index: {game._current_level_index}")

# Get all entities
lcs = [e for e in level.entities if e.tag == '0066ghlkyvdbgg']
selectors = [e for e in level.entities if e.tag == '0001qwdmnlybkb']
blocks = [e for e in level.entities if e.tag == '0064ocqkuqacti']
targets = [e for e in level.entities if e.tag == '0087vvmblxkzdi']

print(f"LCs: {len(lcs)}, Selectors: {len(selectors)}, Blocks: {len(blocks)}, Targets: {len(targets)}")

# Print all LCs with their positions
for lc in lcs:
    print(f"  LC at ({lc.x},{lc.y}) {lc.w}x{lc.h}")

# Print all selectors
for sel in selectors:
    print(f"  Sel at ({sel.x},{sel.y}) {sel.w}x{sel.h}")

# Print block
for blk in blocks:
    print(f"  Block at ({blk.x},{blk.y}) {blk.w}x{blk.h}")

# Print targets
for tgt in targets:
    print(f"  Target at ({tgt.x},{tgt.y}) {tgt.w}x{tgt.h}")

# Get pigtralzpb (LC→Selectors mapping)
pigtralzpb = getattr(level, 'pigtralzpb', None)
if pigtralzpb:
    print("\npigtralzpb (LC→Selectors):")
    for lc_key, sels in pigtralzpb.items():
        lc_key_id = id(lc_key)
        for lc in lcs:
            if id(lc) == lc_key_id:
                sel_infos = [(s.x, s.y, s.w, s.h) for s in sels]
                print(f"  LC({lc.x},{lc.y}) → {sel_infos}")
                break

# Get uricqfoplr (Selector→Children mapping)
uricqfoplr = getattr(level, 'uricqfoplr', None)
if uricqfoplr:
    print("\nuricqfoplr (Selector→Children):")
    for sel_key, children in uricqfoplr.items():
        for sel in selectors:
            if id(sel) == id(sel_key):
                child_infos = [(c.x, c.y, c.w, c.h, c.tag[-6:]) for c in children]
                print(f"  Sel({sel.x},{sel.y}) → {child_infos}")
                break

# Print Block's parent chain
print("\n=== Block parent chain ===")
for blk in blocks:
    print(f"Block({blk.x},{blk.y}) parent chain:")
    entity = blk
    chain = [(entity.x, entity.y)]
    for sel_key, children in uricqfoplr.items():
        for child in children:
            if id(child) == id(entity):
                for sel in selectors:
                    if id(sel) == id(sel_key):
                        entity = sel
                        chain.append((sel.x, sel.y))
                        break
                break
    print(f"  Chain (root→block): {list(reversed(chain))}")

# Now find the UP LC — the one controlling Sel(12,39)
# Sel(12,39)'s rotation should be UP (tracer at [-1,1])
print("\n=== Selector rotations ===")
for sel in selectors:
    px = sel.pixels
    rot = "UNKNOWN"
    if px.shape[0] > 1 and px.shape[1] > 1:
        if px[-1, 1] == 3:
            rot = "UP"
        elif px[1, 0] == 3:
            rot = "RIGHT"
        elif px[0, 1] == 3:
            rot = "DOWN"
        elif px[1, -1] == 3:
            rot = "LEFT"
    print(f"  Sel({sel.x},{sel.y}) {sel.w}x{sel.h} rotation={rot}")

# Find which LC controls which selector
sel_to_lc = {}
for lc in lcs:
    for sel_key, sel_list in pigtralzpb.items():
        if id(lc) == id(sel_key):
            for s in sel_list:
                sel_to_lc[(s.x, s.y)] = (lc.x, lc.y, lc.w, lc.h)

print("\n=== LC→Selector mapping ===")
for sel in selectors:
    lc_info = sel_to_lc.get((sel.x, sel.y), "NONE")
    print(f"  Sel({sel.x},{sel.y}) → LC{lc_info}")
