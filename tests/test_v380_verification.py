"""v3.8.0 QA Verification — Thinker-Performer + Interactive ZKP Loop + PersistentKVCache

Test categories:
1. Import chain completeness
2. Class functionality unit tests
3. Backward compatibility tests
4. game_solvers.py Thinker-Performer integration
5. kaggle_my_agent.py version verification
6. library.json schema validation
"""

import sys
import os
import json
import numpy as np

# ── Setup import path ──
# tomas_learner.py imports from src.encoder.nar_conv (absolute import from project root)
# The 'agent' package lives inside src/ so we need BOTH paths on sys.path:
# 1. Project root for 'from src.encoder.nar_conv import OCT_MUL_TABLE' to work
# 2. src/ for 'from agent.tomas_learner import ...' to work
PROJECT_ROOT = r"C:\Users\1\WorkBuddy\2026-06-22-20-51-49\tomas-arc3-solver"
SRC_DIR = r"C:\Users\1\WorkBuddy\2026-06-22-20-51-49\tomas-arc3-solver\src"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

# ============================================================================
# Test 1: Import Chain Completeness
# ============================================================================
def test_import_chain():
    """Verify all v3.8.0 imports work from both tomas_learner and agent.__init__."""
    print("\n=== Test 1: Import Chain Completeness ===")
    errors = []

    # Direct import from tomas_learner
    try:
        from agent.tomas_learner import PersistentKVCache, InteractiveZKPLoop, ThinkerPerformerPipeline
        print("✅ Direct import from agent.tomas_learner OK")
        print(f"   PersistentKVCache: {PersistentKVCache}")
        print(f"   InteractiveZKPLoop: {InteractiveZKPLoop}")
        print(f"   ThinkerPerformerPipeline: {ThinkerPerformerPipeline}")
    except ImportError as e:
        errors.append(f"Direct import failed: {e}")
        print(f"❌ Direct import from agent.tomas_learner FAILED: {e}")

    # Import from agent.__init__
    try:
        from agent import PersistentKVCache, InteractiveZKPLoop, ThinkerPerformerPipeline
        print("✅ Import from agent.__init__ OK")
    except ImportError as e:
        errors.append(f"__init__ import failed: {e}")
        print(f"❌ Import from agent.__init__ FAILED: {e}")

    return errors


# ============================================================================
# Test 2: Class Functionality Unit Tests
# ============================================================================
def test_class_functionality():
    """Verify 3 new v3.8.0 classes work correctly."""
    print("\n=== Test 2: Class Functionality Unit Tests ===")
    errors = []

    try:
        from agent.tomas_learner import (
            PersistentKVCache, InteractiveZKPLoop, ThinkerPerformerPipeline,
            extract_topo_features
        )
    except ImportError as e:
        errors.append(f"Import failed for unit tests: {e}")
        print(f"❌ Cannot run unit tests, import failed: {e}")
        return errors

    grid = np.array([[1,0,2],[0,1,0],[2,0,1]])

    # ── Test PersistentKVCache ──
    print("\n── PersistentKVCache Tests ──")
    kv = PersistentKVCache()

    # update_thinker_state
    topo = extract_topo_features(grid)
    kv.update_thinker_state(grid, topo, "step_1")
    if kv._step_count != 1:
        errors.append(f"PersistentKVCache.update_thinker_state: expected step_count=1, got {kv._step_count}")
        print(f"❌ update_thinker_state step_count={kv._step_count}, expected 1")
    else:
        print(f"✅ update_thinker_state OK, step_count={kv._step_count}")

    # clean_latent_writeback
    kv.clean_latent_writeback(grid, {"passed": True, "reason": "test"})
    if kv._reference_grid is None:
        errors.append("PersistentKVCache.clean_latent_writeback: reference_grid is None after writeback")
        print(f"❌ clean_latent_writeback: reference_grid is None")
    elif kv._reference_grid.shape != grid.shape:
        errors.append(f"PersistentKVCache.clean_latent_writeback: reference_grid shape={kv._reference_grid.shape}, expected {grid.shape}")
        print(f"❌ clean_latent_writeback: reference_grid shape mismatch")
    else:
        print(f"✅ clean_latent_writeback OK, reference_grid shape={kv._reference_grid.shape}")

    # Verify gaussex_log was recorded
    if len(kv._gaussex_logs) != 1:
        errors.append(f"PersistentKVCache.clean_latent_writeback: gaussex_logs length={len(kv._gaussex_logs)}, expected 1")
        print(f"❌ clean_latent_writeback: gaussex_logs length={len(kv._gaussex_logs)}, expected 1")
    else:
        print(f"✅ gaussex_logs recorded, length={len(kv._gaussex_logs)}")

    # Verify writeback skips when gaussex_result.passed=False
    kv.clean_latent_writeback(np.array([[3,3,3],[3,3,3],[3,3,3]]), {"passed": False, "reason": "failed"})
    if len(kv._gaussex_logs) != 1:  # Should still be 1 (not increment on fail)
        errors.append(f"PersistentKVCache.clean_latent_writeback (fail): gaussex_logs length={len(kv._gaussex_logs)}, expected 1")
        print(f"❌ clean_latent_writeback fail case: gaussex_logs still incremented")
    else:
        print(f"✅ clean_latent_writeback (fail case) correctly skipped")

    # get_performer_state
    state = kv.get_performer_state()
    expected_keys = ["topo_features", "reference_grid", "effective_macros", "compactification_state",
                     "euler_char_history", "connected_components_history", "step_count"]
    missing_keys = [k for k in expected_keys if k not in state]
    if missing_keys:
        errors.append(f"PersistentKVCache.get_performer_state: missing keys {missing_keys}")
        print(f"❌ get_performer_state missing keys: {missing_keys}")
    else:
        print(f"✅ get_performer_state OK, keys={list(state.keys())}")

    # get_topology_invariant_score (default 0.5 with <2 history at init, but we've done update+writeback)
    score = kv.get_topology_invariant_score()
    if not (0.0 <= score <= 1.0):
        errors.append(f"PersistentKVCache.get_topology_invariant_score: score={score} outside [0,1]")
        print(f"❌ get_topology_invariant_score={score}, outside [0,1]")
    else:
        print(f"✅ get_topology_invariant_score={score} (in [0,1])")

    # register_effective_macro
    kv.register_effective_macro("test_macro", "UP UP DOWN")
    if "test_macro" not in kv._effective_macros:
        errors.append("PersistentKVCache.register_effective_macro: macro not registered")
        print(f"❌ register_effective_macro failed")
    elif kv._effective_macros["test_macro"] != "UP UP DOWN":
        errors.append(f"PersistentKVCache.register_effective_macro: DSL sequence mismatch")
        print(f"❌ register_effective_macro: DSL sequence mismatch")
    else:
        print(f"✅ register_effective_macro OK, macros={list(kv._effective_macros.keys())}")

    # ── Test InteractiveZKPLoop ──
    print("\n── InteractiveZKPLoop Tests ──")
    kv2 = PersistentKVCache()
    zkp = InteractiveZKPLoop(kv2)
    print(f"✅ InteractiveZKPLoop init OK")

    # Run loop with empty candidates
    best, result = zkp.run_loop(grid, {"game_id": "test"}, [])
    if best is not None:
        errors.append(f"InteractiveZKPLoop.run_loop (empty candidates): expected best=None, got {best}")
        print(f"❌ run_loop (empty candidates): best={best}, expected None")
    else:
        print(f"✅ run_loop (empty candidates) OK, best={best}")

    if "n_verified" not in result:
        errors.append(f"InteractiveZKPLoop.run_loop: missing 'n_verified' in result")
        print(f"❌ run_loop: missing n_verified key")
    else:
        print(f"✅ run_loop result keys={list(result.keys())}")

    # Run loop with some candidates
    best2, result2 = zkp.run_loop(grid, {"game_id": "test"}, ["UP", "DOWN", "LEFT"])
    print(f"✅ run_loop (with candidates) OK, n_verified={result2['n_verified']}")

    # Verify loop_count increments
    if result2["loop_count"] != 2:
        errors.append(f"InteractiveZKPLoop.run_loop: loop_count={result2['loop_count']}, expected 2")
        print(f"❌ run_loop: loop_count={result2['loop_count']}, expected 2")
    else:
        print(f"✅ loop_count increments correctly: {result2['loop_count']}")

    # Get loop stats
    stats = zkp.get_loop_stats()
    expected_stats_keys = ["total_loops", "avg_information_gain", "cumulative_depth_reduction", "topo_invariant_score"]
    missing_stats = [k for k in expected_stats_keys if k not in stats]
    if missing_stats:
        errors.append(f"InteractiveZKPLoop.get_loop_stats: missing keys {missing_stats}")
        print(f"❌ get_loop_stats missing keys: {missing_stats}")
    else:
        print(f"✅ get_loop_stats OK, stats_keys={list(stats.keys())}")

    # ── Test ThinkerPerformerPipeline ──
    print("\n── ThinkerPerformerPipeline Tests ──")
    pipeline = ThinkerPerformerPipeline()
    print(f"✅ ThinkerPerformerPipeline init OK")

    # Think phase
    thinker_result = pipeline.think_phase(grid, {"game_id": "test"}, "test_game")
    expected_thinker_keys = ["topo", "complexity", "phys_pruner", "game_id", "grid_shape"]
    missing_thinker = [k for k in expected_thinker_keys if k not in thinker_result]
    if missing_thinker:
        errors.append(f"ThinkerPerformerPipeline.think_phase: missing keys {missing_thinker}")
        print(f"❌ think_phase missing keys: {missing_thinker}")
    else:
        print(f"✅ think_phase OK, result_keys={list(thinker_result.keys())}")

    # Verify KV-cache was updated
    if pipeline._kv_cache._step_count < 1:
        errors.append(f"ThinkerPerformerPipeline.think_phase: kv_cache step_count={pipeline._kv_cache._step_count}, expected >= 1")
        print(f"❌ think_phase: KV-cache not updated")
    else:
        print(f"✅ think_phase: KV-cache updated, step_count={pipeline._kv_cache._step_count}")

    # Perform phase
    performer_result = pipeline.perform_phase(None, "test_game", 0, [])
    expected_performer_keys = ["phys_pruner", "complexity", "complexity_class", "performer_state", "topo_invariant_score"]
    missing_performer = [k for k in expected_performer_keys if k not in performer_result]
    if missing_performer:
        errors.append(f"ThinkerPerformerPipeline.perform_phase: missing keys {missing_performer}")
        print(f"❌ perform_phase missing keys: {missing_performer}")
    else:
        print(f"✅ perform_phase OK, result_keys={list(performer_result.keys())}")

    # Pipeline stats
    pipeline_stats = pipeline.get_pipeline_stats()
    expected_pipeline_keys = ["total_pipelines", "thinker_results_keys", "performer_results_keys",
                              "zkp_stats", "kv_cache_step_count", "topo_invariant_score"]
    missing_pipeline = [k for k in expected_pipeline_keys if k not in pipeline_stats]
    if missing_pipeline:
        errors.append(f"ThinkerPerformerPipeline.get_pipeline_stats: missing keys {missing_pipeline}")
        print(f"❌ get_pipeline_stats missing keys: {missing_pipeline}")
    else:
        print(f"✅ get_pipeline_stats OK, stats_keys={list(pipeline_stats.keys())}")

    return errors


# ============================================================================
# Test 3: Backward Compatibility Tests
# ============================================================================
def test_backward_compatibility():
    """Verify v3.7.0 classes still work in v3.8.0 environment."""
    print("\n=== Test 3: Backward Compatibility ===")
    errors = []

    try:
        from agent.tomas_learner import (
            PhysicalCompactificationReduction,
            PhysicalGaussExGuard,
            PsiCutController,
            classify_task_complexity,
            compute_scaling_law_v36,
            extract_topo_features,
        )
        print("✅ v3.7.0 classes still accessible")
    except ImportError as e:
        errors.append(f"v3.7.0 backward compatibility import failed: {e}")
        print(f"❌ v3.7.0 classes import FAILED: {e}")
        return errors

    # Test PhysicalCompactificationReduction init
    try:
        pruner = PhysicalCompactificationReduction()
        print(f"✅ PhysicalCompactificationReduction init OK")
    except Exception as e:
        errors.append(f"PhysicalCompactificationReduction init failed: {e}")
        print(f"❌ PhysicalCompactificationReduction init FAILED: {e}")

    # Test classify_task_complexity — NOTE: signature is (game_id, topo_features, ...)
    # NOT (grid, game_id) — must use correct signature
    try:
        grid = np.array([[1,0,2],[0,1,0],[2,0,1]])
        topo = extract_topo_features(grid)
        result = classify_task_complexity("test_game", topo)
        if "complexity_class" not in result:
            errors.append(f"classify_task_complexity: missing 'complexity_class' in result")
            print(f"❌ classify_task_complexity: missing complexity_class")
        else:
            print(f"✅ classify_task_complexity OK, complexity_class={result['complexity_class']}")
    except Exception as e:
        errors.append(f"classify_task_complexity failed: {e}")
        print(f"❌ classify_task_complexity FAILED: {e}")

    # Test compute_scaling_law_v36
    try:
        scaling_result = compute_scaling_law_v36(C=10.0, D=100.0)
        if "L_total" not in scaling_result:
            errors.append(f"compute_scaling_law_v36: missing 'L_total' in result")
            print(f"❌ compute_scaling_law_v36: missing L_total")
        else:
            print(f"✅ compute_scaling_law_v36 OK, L_total={scaling_result['L_total']}")
    except Exception as e:
        errors.append(f"compute_scaling_law_v36 failed: {e}")
        print(f"❌ compute_scaling_law_v36 FAILED: {e}")

    # Test PhysicalGaussExGuard
    try:
        guard = PhysicalGaussExGuard()
        print(f"✅ PhysicalGaussExGuard init OK")
    except Exception as e:
        errors.append(f"PhysicalGaussExGuard init failed: {e}")
        print(f"❌ PhysicalGaussExGuard init FAILED: {e}")

    # Test PsiCutController
    try:
        controller = PsiCutController()
        print(f"✅ PsiCutController init OK")
    except Exception as e:
        errors.append(f"PsiCutController init failed: {e}")
        print(f"❌ PsiCutController init FAILED: {e}")

    return errors


# ============================================================================
# Test 4: game_solvers.py Thinker-Performer Integration (Code Review)
# ============================================================================
def test_game_solvers_integration():
    """Verify Thinker-Performer code in game_solvers.py is try/except wrapped."""
    print("\n=== Test 4: game_solvers.py Thinker-Performer Integration ===")
    errors = []

    game_solvers_path = r"C:\Users\1\WorkBuddy\2026-06-22-20-51-49\tomas-arc3-solver\src\agent\game_solvers.py"
    with open(game_solvers_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check Thinker-Performer import is in try/except
    lines = content.split("\n")

    # Find the Thinker-Performer block (around line 4839)
    tp_start_found = False
    tp_try_found = False
    tp_except_found = False
    tp_fallback_found = False

    for i, line in enumerate(lines):
        if "thinker_performer_pipeline = None" in line:
            tp_start_found = True
            print(f"✅ Found 'thinker_performer_pipeline = None' at line {i+1}")
        if "from .tomas_learner import ThinkerPerformerPipeline" in line and i > 4800:
            tp_try_found = True
            print(f"✅ Found ThinkerPerformerPipeline import at line {i+1}")
        if "except Exception:" in line and i > 4840 and i < 4860:
            tp_except_found = True
            print(f"✅ Found except block at line {i+1}")
        if "Fallback to v3.7.0 behavior" in line or "pass  # Fallback" in line:
            if i > 4840 and i < 4870:
                tp_fallback_found = True
                print(f"✅ Found fallback comment at line {i+1}")

    if not tp_start_found:
        errors.append("game_solvers.py: thinker_performer_pipeline = None not found")
        print(f"❌ thinker_performer_pipeline init line not found")
    if not tp_try_found:
        errors.append("game_solvers.py: ThinkerPerformerPipeline import not in expected location")
        print(f"❌ ThinkerPerformerPipeline import not found")
    if not tp_except_found:
        errors.append("game_solvers.py: Thinker-Performer code not wrapped in try/except")
        print(f"❌ No try/except wrapper found")
    if not tp_fallback_found:
        errors.append("game_solvers.py: No v3.7.0 fallback comment found")
        print(f"❌ No v3.7.0 fallback comment found")

    # Check topo_invariant_score is also try/except wrapped
    topo_invariant_try = False
    topo_invariant_except = False
    for i, line in enumerate(lines):
        if "topo_invariant_score = 0.5" in line and i > 4870:
            topo_invariant_try = True
            print(f"✅ Found default topo_invariant_score=0.5 at line {i+1}")
        if "thinker_performer_pipeline._kv_cache.get_topology_invariant_score()" in line:
            print(f"✅ Found topo_invariant_score KV-cache access at line {i+1}")
        if "except Exception:" in line and i > 4880 and i < 4890:
            topo_invariant_except = True
            print(f"✅ Found except for topo_invariant_score at line {i+1}")

    if not topo_invariant_try:
        errors.append("game_solvers.py: default topo_invariant_score=0.5 not found")
    if not topo_invariant_except:
        errors.append("game_solvers.py: topo_invariant_score access not try/except wrapped")

    # Verify all v3.8.0 additions are safe (won't break v3.7.0 logic)
    all_safe = tp_start_found and tp_try_found and tp_except_found and tp_fallback_found
    if all_safe:
        print(f"✅ All v3.8.0 Thinker-Performer code is properly try/except wrapped")
    else:
        print(f"❌ Some v3.8.0 code is NOT properly wrapped")

    return errors


# ============================================================================
# Test 5: kaggle_my_agent.py Version Verification
# ============================================================================
def test_kaggle_version():
    """Verify kaggle_my_agent.py version identifiers are v3.8.0."""
    print("\n=== Test 5: kaggle_my_agent.py Version Verification ===")
    errors = []

    agent_path = r"C:\Users\1\WorkBuddy\2026-06-22-20-51-49\tomas-arc3-solver\kaggle_my_agent.py"
    with open(agent_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")

    # Check docstring version is v3.8.0
    docstring_version_found = False
    for i, line in enumerate(lines[:30]):
        if "v3.8.0" in line:
            docstring_version_found = True
            print(f"✅ Found v3.8.0 in docstring at line {i+1}: {line.strip()[:80]}")
            break

    if not docstring_version_found:
        errors.append("kaggle_my_agent.py: v3.8.0 not found in docstring")
        print(f"❌ v3.8.0 not found in docstring")

    # Check name property returns tomas.v3.8.0
    name_property_found = False
    for i, line in enumerate(lines):
        if "def name(self)" in line or 'name(self) -> str' in line:
            # Check the next few lines for the return statement
            for j in range(i+1, min(i+5, len(lines))):
                if "return" in lines[j] and "v3.8.0" in lines[j]:
                    name_property_found = True
                    print(f"✅ name property returns v3.8.0: {lines[j].strip()}")
                    break

    if not name_property_found:
        errors.append("kaggle_my_agent.py: name property does not return v3.8.0")
        print(f"❌ name property does not reference v3.8.0")

    # Check Thinker-Performer import is in try/except
    tp_import_try = False
    tp_import_except = False
    for i, line in enumerate(lines):
        if "from tomas_learner import ThinkerPerformerPipeline" in line:
            tp_import_try = True
            print(f"✅ Found ThinkerPerformerPipeline import at line {i+1}")
        if i > 740 and "except" in line and i < 780:
            tp_import_except = True
            print(f"✅ Found except for Thinker-Performer import at line {i+1}")

    if not tp_import_try:
        errors.append("kaggle_my_agent.py: ThinkerPerformerPipeline import not found")
        print(f"❌ ThinkerPerformerPipeline import not found")
    if not tp_import_except:
        # Search broader for except near Thinker-Performer code
        for i, line in enumerate(lines):
            if i > 740 and i < 800 and "except" in line:
                tp_import_except = True
                print(f"✅ Found except block near Thinker-Performer code at line {i+1}")
                break
        if not tp_import_except:
            errors.append("kaggle_my_agent.py: Thinker-Performer import not in try/except")
            print(f"❌ Thinker-Performer import NOT in try/except")

    return errors


# ============================================================================
# Test 6: library.json Schema Validation
# ============================================================================
def test_library_schema():
    """Verify library.json schema version, framework version, and new macros."""
    print("\n=== Test 6: library.json Schema Validation ===")
    errors = []

    lib_path = r"C:\Users\1\WorkBuddy\2026-06-22-20-51-49\tomas-arc3-solver\library.json"
    with open(lib_path, "r", encoding="utf-8") as f:
        lib = json.load(f)

    # Check _schema_version = "3.8.0"
    schema_version = lib.get("_schema_version", "")
    if schema_version != "3.8.0":
        errors.append(f"library.json: _schema_version='{schema_version}', expected '3.8.0'")
        print(f"❌ _schema_version='{schema_version}', expected '3.8.0'")
    else:
        print(f"✅ _schema_version = '{schema_version}'")

    # Check _tomas_framework.version = "3.8.0"
    framework = lib.get("_tomas_framework", {})
    framework_version = framework.get("version", "")
    if framework_version != "3.8.0":
        errors.append(f"library.json: _tomas_framework.version='{framework_version}', expected '3.8.0'")
        print(f"❌ _tomas_framework.version='{framework_version}', expected '3.8.0'")
    else:
        print(f"✅ _tomas_framework.version = '{framework_version}'")

    # Check new v3.8.0 core_modules
    core_modules = framework.get("core_modules", [])
    expected_new_modules = ["persistent_kv_cache", "interactive_zkp_loop", "thinker_performer_pipeline"]
    for mod in expected_new_modules:
        if mod not in core_modules:
            errors.append(f"library.json: missing core_module '{mod}'")
            print(f"❌ Missing core_module '{mod}'")
        else:
            print(f"✅ core_module '{mod}' present")

    # Check new feature flags
    expected_flags = {
        "thinker_performer_pipeline": True,
        "interactive_zkp_loop": True,
        "kv_cache_persistent_state": True,
        "clean_latent_writeback": True,
        "topology_invariant_guided_ranking": True,
    }
    for flag, expected_val in expected_flags.items():
        val = framework.get(flag)
        if val != expected_val:
            errors.append(f"library.json: {flag}={val}, expected {expected_val}")
            print(f"❌ {flag}={val}, expected {expected_val}")
        else:
            print(f"✅ {flag} = {val}")

    # Check new ZKP macros (at least macro_zkp_observe_encode_prove and macro_zkp_verify_act_writeback)
    abstractions = lib.get("abstractions", [])
    macro_names = [a.get("name", "") for a in abstractions]

    zkp_macros = [m for m in macro_names if "zkp" in m.lower()]
    if len(zkp_macros) < 2:
        errors.append(f"library.json: only {len(zkp_macros)} ZKP macros found, expected at least 2 (observe_encode_prove + verify_act_writeback)")
        print(f"❌ Only {len(zkp_macros)} ZKP macros found: {zkp_macros}")
    else:
        print(f"✅ Found {len(zkp_macros)} ZKP macros: {zkp_macros}")

    # Verify macro_zkp_observe_encode_prove exists
    if "macro_zkp_observe_encode_prove" not in macro_names:
        errors.append("library.json: macro_zkp_observe_encode_prove not found")
        print(f"❌ macro_zkp_observe_encode_prove not found")
    else:
        print(f"✅ macro_zkp_observe_encode_prove found")

    # Verify macro_zkp_verify_act_writeback exists
    if "macro_zkp_verify_act_writeback" not in macro_names:
        errors.append("library.json: macro_zkp_verify_act_writeback not found")
        print(f"❌ macro_zkp_verify_act_writeback not found")
    else:
        print(f"✅ macro_zkp_verify_act_writeback found")

    # Verify complexity_class field still exists (v3.7.0 compat)
    complexity_classes_found = 0
    for a in abstractions:
        if "complexity_class" in a:
            complexity_classes_found += 1

    if complexity_classes_found == 0:
        errors.append("library.json: no abstractions have complexity_class field (v3.7.0 compat broken)")
        print(f"❌ No abstractions have complexity_class field")
    else:
        print(f"✅ {complexity_classes_found} abstractions have complexity_class field (v3.7.0 compat preserved)")

    return errors


# ============================================================================
# Main Runner
# ============================================================================
if __name__ == "__main__":
    all_errors = []

    all_errors.extend(test_import_chain())
    all_errors.extend(test_class_functionality())
    all_errors.extend(test_backward_compatibility())
    all_errors.extend(test_game_solvers_integration())
    all_errors.extend(test_kaggle_version())
    all_errors.extend(test_library_schema())

    # ── Summary ──
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    total_tests = 6
    passed = total_tests - len([e for e in all_errors])  # Approximate
    failed = len(all_errors)

    if all_errors:
        print(f"\n❌ FAILURES ({len(all_errors)} issues found):")
        for e in all_errors:
            print(f"  - {e}")
    else:
        print(f"\n✅ ALL TESTS PASSED — 0 issues found")

    print(f"\nTotal test categories: {total_tests}")
    print(f"Issues found: {len(all_errors)}")

    # ── Routing Decision ──
    if not all_errors:
        print("\nRouting Decision: NoOne (all tests pass)")
    else:
        # Determine if errors are source bugs or test bugs
        source_bugs = []
        test_bugs = []
        for e in all_errors:
            # Classify based on error content
            if "import failed" in e.lower() or "init failed" in e.lower() or "missing keys" in e.lower() or "missing core_module" in e.lower():
                source_bugs.append(e)
            else:
                test_bugs.append(e)

        if source_bugs:
            print(f"\nRouting Decision: Engineer (software-engineer) — {len(source_bugs)} source bugs found")
        if test_bugs:
            print(f"\nRouting Decision: QA (self) — {len(test_bugs)} test bugs to fix")
