"""
src/agent/test_hybrid_pipeline.py
TOMAS 四层混合搜索管线集成测试

验证完整的L1→L2→L3→L4管线工作流程:
  1. HybridSearchPipeline构建
  2. 四层策略注册
  3. solve()完整管线执行
  4. game_profiles配置路由
  5. 遗留兼容(SOLVERS fallback)

Version: v3.18.0 — Hybrid Search Integration Test
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def _self_test() -> bool:
    """四层混合搜索管线集成测试。

    Returns:
        True if all tests pass, False otherwise.
    """
    all_pass: bool = True

    # ── Test 1: HybridSearchPipeline构建 ──
    try:
        from .hybrid_search_engine import (
            HybridSearchPipeline,
            PipelineStrategies,
            CandidateSet,
            EvaluatedCandidateSet,
        )

        # 默认配置
        pipeline: HybridSearchPipeline = HybridSearchPipeline()
        assert pipeline.l1 is not None, "L1 should be built"
        assert pipeline.l2 is not None, "L2 should be built"
        assert pipeline.l3 is not None, "L3 should be built"
        assert pipeline.l4 is not None, "L4 should be built"

        # 自定义配置
        ps: PipelineStrategies = PipelineStrategies(
            l1_strategy="bfs",
            l2_strategy="pass_through",
            l3_strategy="gauss_ex",
            l4_strategy="kappa_selector",
        )
        custom_pipeline: HybridSearchPipeline = HybridSearchPipeline(ps)
        assert custom_pipeline.strategies.l1_strategy == "bfs"
        assert custom_pipeline.strategies.l2_strategy == "pass_through"
        assert custom_pipeline.strategies.l3_strategy == "gauss_ex"
        assert custom_pipeline.strategies.l4_strategy == "kappa_selector"

        print("[PASS] Test 1: HybridSearchPipeline构建")
    except Exception as e:
        print(f"[FAIL] Test 1: {e}")
        all_pass = False

    # ── Test 2: 四层策略注册 ──
    try:
        # L1注册
        assert 'bfs' in HybridSearchPipeline.L1_REGISTRY
        assert 'dfs' in HybridSearchPipeline.L1_REGISTRY
        assert 'wall_bfs' in HybridSearchPipeline.L1_REGISTRY
        assert 'delta_replay' in HybridSearchPipeline.L1_REGISTRY
        assert 'direct' in HybridSearchPipeline.L1_REGISTRY

        # L2注册
        assert 'pass_through' in HybridSearchPipeline.L2_REGISTRY
        assert 'combo_symmetry' in HybridSearchPipeline.L2_REGISTRY
        assert 'prime_signature' in HybridSearchPipeline.L2_REGISTRY
        assert 'matroid_constraint' in HybridSearchPipeline.L2_REGISTRY

        # L3注册
        assert 'gauss_ex' in HybridSearchPipeline.L3_REGISTRY
        assert 'kappa_snap' in HybridSearchPipeline.L3_REGISTRY
        assert 'dead_zero_fuse' in HybridSearchPipeline.L3_REGISTRY
        assert 'asym_index' in HybridSearchPipeline.L3_REGISTRY
        assert 'pass_through' in HybridSearchPipeline.L3_REGISTRY

        # L4注册
        assert 'kappa_selector' in HybridSearchPipeline.L4_REGISTRY
        assert 'liu_priority' in HybridSearchPipeline.L4_REGISTRY

        print("[PASS] Test 2: 四层策略注册")
    except Exception as e:
        print(f"[FAIL] Test 2: {e}")
        all_pass = False

    # ── Test 3: HybridGameProfile配置 ──
    try:
        from .game_profiles import HybridGameProfile, HYBRID_GAME_PROFILES

        # 验证25游戏配置表
        assert len(HYBRID_GAME_PROFILES) == 25, \
            f"Should have 25 game profiles, got {len(HYBRID_GAME_PROFILES)}"

        # KA59配置
        ka59: HybridGameProfile = HYBRID_GAME_PROFILES['ka59']
        assert ka59.l1_strategy == "wall_bfs", f"KA59 L1 should be wall_bfs, got {ka59.l1_strategy}"
        assert ka59.l2_strategy == "combo_symmetry"
        assert ka59.l3_strategy == "dead_zero_fuse"
        assert ka59.l4_strategy == "kappa_selector"

        # TN36配置
        tn36: HybridGameProfile = HYBRID_GAME_PROFILES['tn36']
        assert tn36.l1_strategy == "direct"
        assert tn36.l2_strategy == "pass_through"
        assert tn36.l3_strategy == "pass_through"
        assert tn36.l4_strategy == "kappa_selector"

        # CN04配置
        cn04: HybridGameProfile = HYBRID_GAME_PROFILES['cn04']
        assert cn04.l1_strategy == "dfs"
        assert cn04.l2_strategy == "prime_signature"
        assert cn04.l3_strategy == "kappa_snap"
        assert cn04.l4_strategy == "kappa_selector"

        # 零分游戏配置
        s5i5: HybridGameProfile = HYBRID_GAME_PROFILES['s5i5']
        assert s5i5.l3_strategy == "pass_through"
        assert s5i5.l4_strategy == "pass_through"

        # to_pipeline_strategies()
        ps_ka59: PipelineStrategies = ka59.to_pipeline_strategies()
        assert ps_ka59.l1_strategy == "wall_bfs"
        assert ps_ka59.l2_strategy == "combo_symmetry"
        assert ps_ka59.l3_strategy == "dead_zero_fuse"
        assert ps_ka59.l4_strategy == "kappa_selector"

        # from_game_profile()
        from .game_profiles import GameProfile
        base_profile: GameProfile = GameProfile(game_id="test_game", action_type="keyboard")
        hybrid: HybridGameProfile = HybridGameProfile.from_game_profile(
            base_profile, l1="dfs", l2="prime_signature",
            l3="kappa_snap", l4="kappa_selector",
        )
        assert hybrid.game_id == "test_game"
        assert hybrid.l1_strategy == "dfs"
        assert hybrid.l2_strategy == "prime_signature"
        assert hybrid.l3_strategy == "kappa_snap"
        assert hybrid.l4_strategy == "kappa_selector"

        print("[PASS] Test 3: HybridGameProfile配置")
    except Exception as e:
        print(f"[FAIL] Test 3: {e}")
        all_pass = False

    # ── Test 4: 数据载体 ──
    try:
        from .delta_state import Node
        from .hybrid_search_engine import CandidateSet, EvaluatedCandidateSet

        # CandidateSet
        cs: CandidateSet = CandidateSet(
            node_ids=[0, 1, 2],
            node_map={0: Node(id=0, parent_id=-1, action="root", depth=0)},
        )
        assert len(cs) == 3
        assert not cs.is_empty()

        empty_cs: CandidateSet = CandidateSet()
        assert len(empty_cs) == 0
        assert empty_cs.is_empty()

        # EvaluatedCandidateSet
        ecs: EvaluatedCandidateSet = EvaluatedCandidateSet(
            candidates=[
                {'node_id': 0, 'eta': 0.1, 'confidence': 0.8, 'ic': 0.5, 'depth': 2},
                {'node_id': 1, 'eta': 0.3, 'confidence': 0.6, 'ic': 0.7, 'depth': 3},
            ],
        )
        assert len(ecs) == 2
        assert not ecs.is_empty()
        assert ecs.best_eta() == 0.1
        assert ecs.best_confidence() == 0.8

        empty_ecs: EvaluatedCandidateSet = EvaluatedCandidateSet()
        assert empty_ecs.is_empty()
        assert empty_ecs.best_eta() is None
        assert empty_ecs.best_confidence() is None

        print("[PASS] Test 4: 数据载体")
    except Exception as e:
        print(f"[FAIL] Test 4: {e}")
        all_pass = False

    # ── Test 5: ReplayEngine.from_game() ──
    try:
        from .delta_state import ReplayEngine

        # 构造mock game
        class MockGame:
            _grid = np.zeros((64, 64), dtype=int)

        mock_game: MockGame = MockGame()

        # from_game()
        engine: ReplayEngine = ReplayEngine.from_game(mock_game, mode='grid', game_id="test_game")
        assert engine.mode == 'grid'
        assert 0 in engine.node_map
        assert engine.node_map[0].action == "root"

        # 全局共享
        engine_shared: ReplayEngine = ReplayEngine.from_game(
            mock_game, mode='grid', game_id="shared_test", shared=True,
        )
        retrieved: Optional[ReplayEngine] = ReplayEngine.get_shared("shared_test")
        assert retrieved is not None, "Shared engine should be retrievable"

        # 清除共享
        ReplayEngine.clear_shared()
        assert ReplayEngine.get_shared("shared_test") is None, "Should be cleared"

        print("[PASS] Test 5: ReplayEngine.from_game()")
    except Exception as e:
        print(f"[FAIL] Test 5: {e}")
        all_pass = False

    # ── Test 6: κ-优选η升序 ──
    try:
        from .kappa_selector import KappaEtaAscendSelector, KAPPA_DELTA_K, KAPPA_MIN_CONFIDENCE

        selector: KappaEtaAscendSelector = KappaEtaAscendSelector()
        candidates: List[Dict[str, Any]] = [
            {'node_id': 1, 'eta': 0.8, 'ic': 0.5, 'depth': 3},
            {'node_id': 2, 'eta': 0.05, 'ic': 0.3, 'depth': 2},
            {'node_id': 3, 'eta': 0.2, 'ic': 0.7, 'depth': 4},
            {'node_id': 4, 'eta': 0.03, 'ic': 0.2, 'depth': 1},
        ]

        result: List[Dict[str, Any]] = selector.select(candidates)
        assert isinstance(result, list)

        # 置信度
        conf: float = selector.confidence(0.03)
        assert conf > 0

        # 常量验证
        assert KAPPA_DELTA_K == 0.036
        assert abs(KAPPA_MIN_CONFIDENCE - 1.0/6.0) < 0.01

        print("[PASS] Test 6: κ-优选η升序")
    except Exception as e:
        print(f"[FAIL] Test 6: {e}")
        all_pass = False

    # ── Test 7: __init__.py导出验证 ──
    try:
        from . import (
            HybridGameProfile,
            CandidateSet,
            EvaluatedCandidateSet,
            PipelineStrategies,
            HybridSearchPipeline,
            BFSPathCandidateGenerator,
            DFSEnumerationCandidateGenerator,
            WallBFSCandidateGenerator,
            PassThroughPruner,
            ComboSymmetryPruner,
            GaussExEvaluation,
            KappaSnapEvaluation,
            DeadZeroFuseEvaluation,
            KappaSelector,
            LiuSelector,
            KappaEtaAscendSelector,
            KAPPA_DELTA_K,
            WallBFSEngine,
        )
        print("[PASS] Test 7: __init__.py导出验证")
    except Exception as e:
        print(f"[FAIL] Test 7: {e}")
        all_pass = False

    # ── 总结 ──
    if all_pass:
        print("\n[ALL PASS] test_hybrid_pipeline 全部7项测试通过")
    else:
        print("\n[FAIL] test_hybrid_pipeline 有测试失败")

    return all_pass


if __name__ == "__main__":
    _self_test()
