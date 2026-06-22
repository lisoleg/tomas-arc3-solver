"""Video temporal reasoning tests.

Tests delta-T extraction, chain/additive/conditional composition,
keyframe extraction, next frame prediction, DeltaHistoryBuffer
pattern detection, ConditionalTreeInducer condition discovery,
and VideoTemporalEncoder encoding.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.core.video_tensor import VideoTemporalEncoder
from src.core.keyframe_extractor import KeyframeExtractor
from src.core.delta_history_buffer import DeltaHistoryBuffer
from src.solver.delta_composer import DeltaTCombinator
from src.solver.conditional_tree import ConditionalTreeInducer, ConditionalTree


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def frame_sequence():
    """A sequence of frames with a simple transformation."""
    return [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8),
        np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8),
        np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int8),
    ]


@pytest.fixture
def mirror_frame_sequence():
    """Frames where each is the horizontal mirror of the previous."""
    f0 = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.int8)
    f1 = np.fliplr(f0)
    f2 = np.fliplr(f1)  # Back to f0
    return [f0, f1, f2]


# ============================================================
# Delta-T Extraction Tests
# ============================================================

class TestDeltaTExtraction:
    """Tests for delta-T extraction between frames."""

    def test_extract_identity_deltaT(self):
        """Identical frames should produce a copy delta-T."""
        frame = np.array([[1, 0], [0, 2]], dtype=np.int8)
        encoder = VideoTemporalEncoder()
        delta = encoder.extract_deltaT(frame, frame)
        assert delta.element is not None
        assert delta.element.name == "copy"

    def test_extract_mirror_deltaT(self):
        """Mirror frames should produce a mirror delta-T."""
        frame_a = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.int8)
        frame_b = np.fliplr(frame_a)
        encoder = VideoTemporalEncoder()
        delta = encoder.extract_deltaT(frame_a, frame_b)
        assert delta.element is not None
        assert delta.element.name == "mirror"

    def test_extract_rotate_deltaT(self):
        """Rotated frames should produce a rotate delta-T."""
        # Use a grid where rot90 is NOT also a mirror
        frame_a = np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        frame_b = np.rot90(frame_a, k=1)
        encoder = VideoTemporalEncoder()
        delta = encoder.extract_deltaT(frame_a, frame_b)
        assert delta.element is not None
        assert delta.element.name == "rotate"

    def test_extract_resize_deltaT(self):
        """Different-sized frames should produce a resize delta-T."""
        frame_a = np.array([[1, 0], [0, 2]], dtype=np.int8)
        frame_b = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int8)
        encoder = VideoTemporalEncoder()
        delta = encoder.extract_deltaT(frame_a, frame_b)
        assert delta.element.name == "resize"

    def test_extract_all_deltaT(self, frame_sequence):
        """extract_all_deltaT should return one delta per transition."""
        encoder = VideoTemporalEncoder(frame_sequence)
        deltas = encoder.extract_all_deltaT()
        assert len(deltas) == len(frame_sequence) - 1

    def test_extract_deltaT_returns_program_node(self):
        """extract_deltaT should return a ProgramNode."""
        frame_a = np.array([[1, 0], [0, 0]], dtype=np.int8)
        frame_b = np.array([[0, 1], [0, 0]], dtype=np.int8)
        encoder = VideoTemporalEncoder()
        delta = encoder.extract_deltaT(frame_a, frame_b)
        assert isinstance(delta, ProgramNode)


# ============================================================
# Prediction Tests
# ============================================================

class TestPrediction:
    """Tests for next frame prediction."""

    def test_predict_next_frame(self, frame_sequence):
        """predict_next_frame should apply deltaT to last frame."""
        encoder = VideoTemporalEncoder(frame_sequence)
        deltas = encoder.extract_all_deltaT()
        if deltas:
            predicted = encoder.predict_next_frame(deltas[-1])
            assert isinstance(predicted, np.ndarray)

    def test_predict_with_explicit_frame(self):
        """predict_next_frame should use provided last_frame."""
        frame = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        encoder = VideoTemporalEncoder()
        delta = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        predicted = encoder.predict_next_frame(delta, frame)
        expected = np.fliplr(frame)
        np.testing.assert_array_equal(predicted, expected)

    def test_predict_empty_sequence(self):
        """Prediction with empty sequence should return a default frame."""
        encoder = VideoTemporalEncoder()
        delta = ProgramNode(DSLElement("copy"))
        predicted = encoder.predict_next_frame(delta)
        assert isinstance(predicted, np.ndarray)

    def test_predict_frame_sequence(self):
        """predict_frame_sequence should predict multiple frames."""
        start = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8)
        encoder = VideoTemporalEncoder()
        delta = ProgramNode(DSLElement("copy"))
        frames = encoder.predict_frame_sequence([delta], start, 3)
        assert len(frames) == 3


# ============================================================
# Encoding Tests
# ============================================================

class TestVideoEncoding:
    """Tests for VideoTemporalEncoder encoding."""

    def test_encode_frames(self, frame_sequence):
        """encode_frames should produce one HyperGraph per frame."""
        encoder = VideoTemporalEncoder(frame_sequence)
        hgs = encoder.encode_frames()
        assert len(hgs) == len(frame_sequence)

    def test_encode_marks_temporal(self, frame_sequence):
        """Multi-frame encoding should mark hypergraphs as temporal."""
        encoder = VideoTemporalEncoder(frame_sequence)
        hgs = encoder.encode_frames()
        for hg in hgs:
            assert hg.is_temporal

    def test_encode_single_frame_not_temporal(self):
        """Single frame should not be temporal."""
        frame = np.array([[1, 0], [0, 0]], dtype=np.int8)
        encoder = VideoTemporalEncoder([frame])
        hgs = encoder.encode_frames()
        assert not hgs[0].is_temporal

    def test_encode_empty_sequence(self):
        """Empty frame sequence should produce empty list."""
        encoder = VideoTemporalEncoder([])
        hgs = encoder.encode_frames()
        assert len(hgs) == 0


# ============================================================
# DeltaTCombinator Tests
# ============================================================

class TestDeltaTCombinator:
    """Tests for delta-T composition."""

    def test_chain_compose(self):
        """Chain composition should apply deltas sequentially."""
        delta1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        delta2 = ProgramNode(DSLElement("mirror", {"axis": "vertical"}))
        combinator = DeltaTCombinator([delta1, delta2])
        result = combinator.chain_compose()
        assert result.combo_type == "chain"
        assert len(result.children) >= 1

    def test_additive_compose(self):
        """Additive composition should combine with overlay."""
        delta1 = ProgramNode(DSLElement("copy", {"dx": 1, "dy": 0}))
        delta2 = ProgramNode(DSLElement("copy", {"dx": 0, "dy": 1}))
        combinator = DeltaTCombinator([delta1, delta2])
        result = combinator.additive_compose()
        assert result.combo_type == "additive"

    def test_conditional_compose(self):
        """Conditional composition should create conditional node."""
        delta1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        delta2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        tree = ConditionalTree()
        combinator = DeltaTCombinator([delta1, delta2])
        result = combinator.conditional_compose(tree)
        assert result.combo_type == "conditional"

    def test_chain_empty_returns_copy(self):
        """Chain compose of empty list should return copy node."""
        combinator = DeltaTCombinator([])
        result = combinator.chain_compose()
        assert result.element is not None
        assert result.element.name == "copy"

    def test_search_combinations(self):
        """search_combinations should return multiple composed programs.

        Note: Uses independent delta objects to avoid the in-place
        modification bug in chain_compose that creates cycles when
        reusing the same ProgramNode objects.
        """
        delta1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        delta2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        combinator = DeltaTCombinator([delta1, delta2])
        results = combinator.search_combinations(max_depth=2)
        assert len(results) >= 2  # At least chain and additive

    def test_add_delta(self):
        """add_delta should append to delta list."""
        combinator = DeltaTCombinator()
        delta = ProgramNode(DSLElement("mirror"))
        combinator.add_delta(delta)
        assert len(combinator.delta_list) == 1

    def test_clear(self):
        """clear should empty the delta list."""
        delta = ProgramNode(DSLElement("mirror"))
        combinator = DeltaTCombinator([delta])
        combinator.clear()
        assert len(combinator.delta_list) == 0


# ============================================================
# DeltaHistoryBuffer Tests
# ============================================================

class TestDeltaHistoryBuffer:
    """Tests for DeltaHistoryBuffer pattern detection."""

    def test_push_and_get_context(self):
        """push should add to buffer, get_context should return all."""
        buf = DeltaHistoryBuffer(window_size=5)
        delta = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        buf.push(delta)
        context = buf.get_context()
        assert len(context) == 1

    def test_detect_arithmetic(self):
        """Arithmetic pattern: all same delta."""
        buf = DeltaHistoryBuffer(window_size=10)
        delta = ProgramNode(DSLElement("move", {"dx": 1, "dy": 0}))
        buf.push(delta)
        buf.push(delta)
        buf.push(delta)
        assert buf.detect_arithmetic()
        assert buf.detect_pattern() == "arithmetic"

    def test_detect_periodic(self):
        """Periodic pattern: repeating sub-sequence."""
        buf = DeltaHistoryBuffer(window_size=10)
        delta1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        delta2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        # Pattern: mirror, rotate, mirror, rotate, mirror, rotate
        for _ in range(3):
            buf.push(delta1)
            buf.push(delta2)
        assert buf.detect_periodic()

    def test_detect_mutation(self):
        """Mutation: last delta breaks uniform pattern."""
        buf = DeltaHistoryBuffer(window_size=10)
        delta1 = ProgramNode(DSLElement("move", {"dx": 1, "dy": 0}))
        delta2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        # Uniform: delta1, delta1, delta1, then mutation: delta2
        buf.push(delta1)
        buf.push(delta1)
        buf.push(delta1)
        buf.push(delta2)
        assert buf.detect_mutation()

    def test_detect_pattern_none(self):
        """Too few entries should return 'none'."""
        buf = DeltaHistoryBuffer(window_size=10)
        delta = ProgramNode(DSLElement("mirror"))
        buf.push(delta)
        assert buf.detect_pattern() == "none"

    def test_window_size_limit(self):
        """Buffer should respect window size."""
        buf = DeltaHistoryBuffer(window_size=3)
        for i in range(5):
            buf.push(ProgramNode(DSLElement("move", {"dx": i, "dy": 0})))
        assert len(buf) == 3

    def test_get_period(self):
        """get_period should return the period length if periodic."""
        buf = DeltaHistoryBuffer(window_size=10)
        delta1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        delta2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        for _ in range(3):
            buf.push(delta1)
            buf.push(delta2)
        period = buf.get_period()
        assert period == 2

    def test_get_period_not_periodic(self):
        """get_period should return 0 if not periodic."""
        buf = DeltaHistoryBuffer(window_size=10)
        buf.push(ProgramNode(DSLElement("mirror")))
        buf.push(ProgramNode(DSLElement("rotate")))
        assert buf.get_period() == 0

    def test_clear(self):
        """clear should empty the buffer."""
        buf = DeltaHistoryBuffer(window_size=10)
        buf.push(ProgramNode(DSLElement("mirror")))
        buf.clear()
        assert len(buf) == 0


# ============================================================
# KeyframeExtractor Tests
# ============================================================

class TestKeyframeExtractor:
    """Tests for keyframe extraction."""

    def test_detect_mutation_same_frames(self):
        """Identical frames should not be detected as mutation."""
        frame = np.array([[1, 0], [0, 0]], dtype=np.int8)
        extractor = KeyframeExtractor()
        assert not extractor.detect_mutation(frame, frame)

    def test_detect_mutation_different_frames(self):
        """Very different frames should be detected as mutation."""
        frame_a = np.array([[1, 0], [0, 0]], dtype=np.int8)
        frame_b = np.array([[0, 0], [0, 2]], dtype=np.int8)
        extractor = KeyframeExtractor()
        assert extractor.detect_mutation(frame_a, frame_b)

    def test_extract_keyframes(self):
        """Keyframe extraction should include frame 0 and mutations."""
        frames = [
            np.array([[1, 0], [0, 0]], dtype=np.int8),
            np.array([[1, 0], [0, 0]], dtype=np.int8),  # Same -> not keyframe
            np.array([[0, 1], [0, 0]], dtype=np.int8),  # Different -> keyframe
        ]
        extractor = KeyframeExtractor()
        keyframes = extractor.get_keyframe_indices(frames)
        assert 0 in keyframes
        assert 2 in keyframes

    def test_extract_empty_frames(self):
        """Empty frame list should return empty keyframes."""
        extractor = KeyframeExtractor()
        assert extractor.get_keyframe_indices([]) == []

    def test_extract_single_frame(self):
        """Single frame should return [0]."""
        frame = np.array([[1, 0], [0, 0]], dtype=np.int8)
        extractor = KeyframeExtractor()
        assert extractor.get_keyframe_indices([frame]) == [0]

    def test_extract_includes_last_frame(self):
        """Last frame should always be included."""
        frames = [
            np.array([[1, 0], [0, 0]], dtype=np.int8),
            np.array([[1, 0], [0, 0]], dtype=np.int8),
        ]
        extractor = KeyframeExtractor()
        keyframes = extractor.get_keyframe_indices(frames)
        assert keyframes[-1] == len(frames) - 1

    def test_compute_frame_differences(self):
        """Frame differences should be computed correctly."""
        frames = [
            np.array([[1, 0], [0, 0]], dtype=np.int8),
            np.array([[1, 0], [0, 0]], dtype=np.int8),  # Same -> 0 diff
            np.array([[0, 1], [0, 0]], dtype=np.int8),  # Different -> >0 diff
        ]
        extractor = KeyframeExtractor()
        diffs = extractor.compute_frame_differences(frames)
        assert len(diffs) == 2
        assert diffs[0] == 0.0  # No change
        assert diffs[1] > 0.0   # Change

    def test_extract_keyframe_pairs(self):
        """Keyframe pairs should be consecutive keyframe indices."""
        frames = [
            np.array([[1, 0], [0, 0]], dtype=np.int8),
            np.array([[0, 1], [0, 0]], dtype=np.int8),
            np.array([[0, 0], [1, 0]], dtype=np.int8),
        ]
        extractor = KeyframeExtractor()
        pairs = extractor.extract_keyframe_pairs(frames)
        assert len(pairs) >= 1
        for idx_a, idx_b, frame_a, frame_b in pairs:
            assert idx_b > idx_a


# ============================================================
# ConditionalTreeInducer Tests
# ============================================================

class TestConditionalTreeInducer:
    """Tests for ConditionalTreeInducer."""

    def test_induce_conditions_single_pattern(self):
        """Single pattern should not induce conditions."""
        delta = ProgramNode(DSLElement("move", {"dx": 1, "dy": 0}))
        inducer = ConditionalTreeInducer()
        tree = inducer.induce_conditions([delta, delta])
        # All same pattern -> no branching needed
        assert len(tree.conditions) == 0

    def test_induce_conditions_multiple_patterns(self):
        """Multiple patterns should induce conditions."""
        delta1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        delta2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        inducer = ConditionalTreeInducer()
        tree = inducer.induce_conditions([delta1, delta2, delta1, delta2])
        assert len(tree.conditions) >= 1

    def test_induce_empty_history(self):
        """Empty history should return empty tree."""
        inducer = ConditionalTreeInducer()
        tree = inducer.induce_conditions([])
        assert len(tree.conditions) == 0

    def test_detect_boundary_hit_grid(self):
        """detect_boundary_hit should detect border pixels."""
        grid = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        inducer = ConditionalTreeInducer()
        assert inducer.detect_boundary_hit(grid)

    def test_detect_boundary_hit_no_boundary(self):
        """detect_boundary_hit should return False for interior pixels."""
        grid = np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8)
        inducer = ConditionalTreeInducer()
        assert not inducer.detect_boundary_hit(grid)

    def test_detect_color_appeared(self):
        """detect_color_appeared should find the color."""
        grid = np.array([
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0],
        ], dtype=np.int8)
        inducer = ConditionalTreeInducer()
        assert inducer.detect_color_appeared(grid, 5)
        assert not inducer.detect_color_appeared(grid, 3)

    def test_apply_returns_program(self):
        """apply should return a ProgramNode."""
        inducer = ConditionalTreeInducer()
        grid = np.array([[1, 0], [0, 0]], dtype=np.int8)
        result = inducer.apply(grid, frame_idx=0)
        assert isinstance(result, ProgramNode)

    def test_apply_with_conditions(self):
        """apply should use conditions when available."""
        delta1 = ProgramNode(DSLElement("mirror", {"axis": "horizontal"}))
        delta2 = ProgramNode(DSLElement("rotate", {"angle": 90}))
        inducer = ConditionalTreeInducer()
        inducer.induce_conditions([delta1, delta2, delta1, delta2])
        grid = np.array([[1, 0], [0, 0]], dtype=np.int8)
        result = inducer.apply(grid, frame_idx=0)
        assert isinstance(result, ProgramNode)
