"""Test script for v3.3.0 泛函宏统一场论 integration."""

import sys
sys.path.insert(0, "src")

from agent.tomas_learner import (
    TOMASLearner, MacroCandidate, LibraryManager,
    GaussExGuard, FastPathDispatcher, extract_topo_features
)
import numpy as np

print("=== v3.3.0 Integration Test ===")

# 1. MacroCandidate new fields
print(f"\nMacroCandidate fields: {[f.name for f in MacroCandidate.__dataclass_fields__.values()]}")
mc = MacroCandidate(
    name="test_macro",
    dsl_sequence=[{"action": "UP"}],
    applicable_topo={"euler_char": 1, "period_rank": 0},
    gaussex_precond="True",
)
print(f"  applicable_topo: {mc.applicable_topo}")
print(f"  gaussex_precond: {mc.gaussex_precond}")

# 2. extract_topo_features
grid = np.zeros((10, 10), dtype=int)
grid[2:4, 3:5] = 1  # A 2x2 block
topo = extract_topo_features(grid)
print(f"\nextract_topo_features result: {topo}")

# 3. GaussExGuard
guard = GaussExGuard()
macro_pass = MacroCandidate(name="test", dsl_sequence=[{"action": "UP"}], gaussex_precond="True")
passes, reason = guard.check_precondition(macro_pass, {"sprites": []})
print(f"\nGaussExGuard (True precond): passes={passes}, reason={reason}")

macro_cond = MacroCandidate(
    name="test2", dsl_sequence=[{"action": "UP"}],
    gaussex_precond="sprite_count('movable') == 1 AND sprite_count('goal') >= 1",
)
passes2, reason2 = guard.check_precondition(macro_cond, {
    "sprites": [{"type": "movable"}, {"type": "goal"}, {"type": "goal"}],
})
print(f"GaussExGuard (conditional precond): passes={passes2}, reason={reason2}")

macro_fail = MacroCandidate(
    name="test3", dsl_sequence=[{"action": "UP"}],
    gaussex_precond="sprite_count('movable') == 2",
)
passes3, reason3 = guard.check_precondition(macro_fail, {
    "sprites": [{"type": "movable"}],
})
print(f"GaussExGuard (should fail): passes={passes3}, reason={reason3}")

# 4. LibraryManager with v3.3.0 schema
lm = LibraryManager(library_path="library.json")
schema_ver = lm._library.get("_schema_version")
print(f"\nLibraryManager schema: {schema_ver}")
macros = lm.get_all_macros()
print(f"Library has {len(macros)} macros")
for m in macros:
    print(f"  {m.name}: applicable_topo={m.applicable_topo}, gaussex_precond={m.gaussex_precond}")

# 5. retrieve_for_topo
candidates = lm.retrieve_for_topo(
    topo_features={"euler_char": 1, "period_rank": 0, "symmetry": [], "density": 0.15},
    game_tags=["navigation_pattern", "sprite_movement", "move_up"],
    k=3,
)
print(f"\nretrieve_for_topo results:")
for macro, score in candidates:
    print(f"  {macro.name}: score={score:.3f}")

# 6. TOMASLearner with Fast-Path
learner = TOMASLearner(library_path="library.json")
print(f"\nTOMASLearner initialized:")
print(f"  fast_path_dispatcher: {type(learner.fast_path_dispatcher).__name__}")
print(f"  gauss_ex_guard: {type(learner.gauss_ex_guard).__name__}")

# 7. try_fast_path
result = learner.try_fast_path(
    grid=grid,
    game_state={"sprites": [{"type": "movable"}, {"type": "goal"}]},
    game_tags=["navigation_pattern", "sprite_movement", "move_up"],
)
print(f"\ntry_fast_path result: {result}")
fp_log = learner.get_fast_path_log()
print(f"Fast-Path log entries: {len(fp_log)}")

# 8. Security check: blocked names
macro_evil = MacroCandidate(
    name="evil", dsl_sequence=[{"action": "UP"}],
    gaussex_precond="exec('import os')",
)
passes_evil, reason_evil = guard.check_precondition(macro_evil, {"sprites": []})
print(f"\nSecurity check (blocked 'exec'): passes={passes_evil}, reason={reason_evil}")

print("\n=== All tests passed! ===")
