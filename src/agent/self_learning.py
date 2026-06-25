"""
SelfLearning — After-action review, cognitive reasoning, and operator
accumulation for the ARC-AGI-3 agent.

This module provides three complementary learning systems that enable
the agent to improve its performance across episodes:

Module A — AfterActionReview:
    Analyzes completed episodes to identify success patterns, failure
    points, and extract actionable lessons. Operates on trajectory data
    (state-action-reward sequences) and episode outcomes.

Module B — CognitiveRecursiveDynamics:
    A 5-layer cognitive reasoning system that processes perceptions
    through escalating levels of abstraction:
    L0 Perception -> L1 Tactical -> L2 Strategic -> L3 Metacognition
    -> L4 Recursive Self-Improvement.
    Each layer feeds into the next, with metacognitive reflection and
    recursive improvement closing the loop.

Module C — OperatorAccumulator:
    Extracts reusable operators (macro-actions with preconditions and
    effects) from successful trajectories. Operators are generalized
    from specific instances and can be retrieved by context matching.
    The operator library can be persisted to disk for cross-session
    learning.

Usage:
    from .self_learning import (
        AfterActionReview,
        CognitiveRecursiveDynamics,
        OperatorAccumulator,
        Trajectory,
        Outcome,
    )

    # After-action review
    reviewer = AfterActionReview()
    result = reviewer.review_episode(trajectory, outcome)

    # Cognitive reasoning
    cognitive = CognitiveRecursiveDynamics()
    result = cognitive.think(perception_data, context)

    # Operator accumulation
    accumulator = OperatorAccumulator()
    op = accumulator.extract_operator(trajectory, outcome)
    ops = accumulator.retrieve_operators(context)

Author: TOMAS Team
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import numpy as np


# ============================================================================
# Shared data structures
# ============================================================================

@dataclass
class Trajectory:
    """A trajectory of an episode (sequence of states and actions).

    Attributes:
        states: List of game states (grids, positions, or dicts).
        actions: List of action IDs taken.
        rewards: List of rewards received after each action.
        level: Level index (0-based).
        game_id: Game identifier (e.g., "ls20").
        metadata: Additional metadata (e.g., danger positions, plan info).
    """

    states: list[Any] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    level: int = 0
    game_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of steps in the trajectory."""
        return len(self.actions)

    @property
    def total_reward(self) -> float:
        """Total cumulative reward."""
        return sum(self.rewards)


@dataclass
class Outcome:
    """Outcome of an episode attempt.

    Attributes:
        success: Whether the level was completed.
        steps: Total steps taken.
        game_over_count: Number of game-over events.
        score: Score achieved (0.0 to 1.0).
        baseline: Human baseline steps for this level.
        level: Level index.
        game_id: Game identifier.
        metadata: Additional outcome metadata.
    """

    success: bool = False
    steps: int = 0
    game_over_count: int = 0
    score: float = 0.0
    baseline: int = 0
    level: int = 0
    game_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def efficiency(self) -> float:
        """Efficiency ratio (baseline / actual steps). >1.0 means better than baseline."""
        if self.steps <= 0:
            return 0.0
        return self.baseline / self.steps

    @property
    def rhae(self) -> float:
        """Relative Human Action Efficiency score (0-115)."""
        if self.steps <= 0 or self.baseline <= 0:
            return 0.0
        return min(115.0, ((self.baseline / self.steps) ** 2) * 100.0)


# ============================================================================
# Module A: AfterActionReview
# ============================================================================

@dataclass
class Pattern:
    """A recognized pattern from episode analysis.

    Attributes:
        description: Human-readable description of the pattern.
        conditions: Conditions under which the pattern applies.
        frequency: How many times this pattern has been observed.
        confidence: Confidence score (0.0 to 1.0).
        action_sequence: Associated action sequence (if applicable).
    """

    description: str = ""
    conditions: dict = field(default_factory=dict)
    frequency: int = 1
    confidence: float = 0.5
    action_sequence: list[int] = field(default_factory=list)


@dataclass
class AntiPattern:
    """A recognized anti-pattern (behavior to avoid).

    Attributes:
        description: Human-readable description of the anti-pattern.
        conditions: Conditions under which the anti-pattern occurs.
        frequency: How many times this anti-pattern has been observed.
        confidence: Confidence score (0.0 to 1.0).
        trigger_action: The action that triggered the failure (if applicable).
    """

    description: str = ""
    conditions: dict = field(default_factory=dict)
    frequency: int = 1
    confidence: float = 0.5
    trigger_action: int = -1


@dataclass
class Lesson:
    """A lesson extracted from episode analysis.

    Attributes:
        description: Human-readable lesson description.
        category: Lesson category ('success', 'failure', 'optimization', 'safety').
        confidence: Confidence score (0.0 to 1.0).
        evidence: Supporting evidence strings.
        applicable_contexts: Contexts where this lesson applies.
    """

    description: str = ""
    category: str = "general"
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    applicable_contexts: list[str] = field(default_factory=list)


@dataclass
class ReviewResult:
    """Result of an after-action review.

    Attributes:
        lessons: Extracted lessons.
        patterns: Identified success patterns.
        anti_patterns: Identified failure patterns.
        summary: Human-readable summary.
        success_rate: Overall success rate (0.0 to 1.0).
        efficiency_score: Efficiency score (0.0 to 1.0).
        key_findings: Key findings from the review.
    """

    lessons: list[Lesson] = field(default_factory=list)
    patterns: list[Pattern] = field(default_factory=list)
    anti_patterns: list[AntiPattern] = field(default_factory=list)
    summary: str = ""
    success_rate: float = 0.0
    efficiency_score: float = 0.0
    key_findings: list[str] = field(default_factory=list)


class AfterActionReview:
    """After-action review system for episode analysis.

    Analyzes completed episodes to identify what went well, what went
    wrong, and extracts actionable lessons for future episodes. The
    system maintains a cumulative database of patterns and anti-patterns
    that grows with each review.

    Usage:
        reviewer = AfterActionReview()
        result = reviewer.review_episode(trajectory, outcome)
        # result.lessons, result.patterns, result.anti_patterns
    """

    # N-gram sizes for action pattern detection
    NGRAM_SIZES: list[int] = [2, 3, 4, 5]

    def __init__(self) -> None:
        """Initialize the after-action review system."""
        self._history: list[ReviewResult] = []
        self._pattern_db: dict[str, Pattern] = {}
        self._anti_pattern_db: dict[str, AntiPattern] = {}
        self._lesson_db: dict[str, Lesson] = {}
        self._total_reviews: int = 0
        self._success_count: int = 0

    def review_episode(
        self,
        trajectory: Trajectory,
        outcome: Outcome,
    ) -> ReviewResult:
        """Analyze a completed episode and produce a review.

        Args:
            trajectory: The episode trajectory (states, actions, rewards).
            outcome: The episode outcome (success, steps, etc.).

        Returns:
            ReviewResult with lessons, patterns, and anti-patterns.
        """
        self._total_reviews += 1
        if outcome.success:
            self._success_count += 1

        patterns = self._identify_success_patterns(trajectory, outcome)
        anti_patterns = self._identify_failure_points(trajectory, outcome)
        lessons = self._extract_lessons(patterns, anti_patterns, outcome)

        # Update cumulative databases
        for p in patterns:
            key = p.description
            if key in self._pattern_db:
                self._pattern_db[key].frequency += 1
                self._pattern_db[key].confidence = min(
                    1.0, self._pattern_db[key].confidence + 0.1
                )
            else:
                self._pattern_db[key] = p

        for ap in anti_patterns:
            key = ap.description
            if key in self._anti_pattern_db:
                self._anti_pattern_db[key].frequency += 1
                self._anti_pattern_db[key].confidence = min(
                    1.0, self._anti_pattern_db[key].confidence + 0.1
                )
            else:
                self._anti_pattern_db[key] = ap

        for lesson in lessons:
            key = lesson.description
            if key in self._lesson_db:
                self._lesson_db[key].confidence = min(
                    1.0, self._lesson_db[key].confidence + 0.05
                )
            else:
                self._lesson_db[key] = lesson

        # Build summary
        success_rate = self._success_count / max(1, self._total_reviews)
        efficiency = outcome.efficiency if outcome.baseline > 0 else 0.0

        key_findings: list[str] = []
        if outcome.success:
            key_findings.append(f"Level {outcome.level} completed in {outcome.steps} steps")
            if outcome.baseline > 0 and outcome.steps < outcome.baseline:
                key_findings.append(
                    f"Beat baseline by {outcome.baseline - outcome.steps} steps "
                    f"(RHAE={outcome.rhae:.1f})"
                )
        else:
            key_findings.append(f"Level {outcome.level} failed after {outcome.steps} steps")
            if outcome.game_over_count > 0:
                key_findings.append(f"Game over occurred {outcome.game_over_count} times")

        if patterns:
            key_findings.append(f"Identified {len(patterns)} success patterns")
        if anti_patterns:
            key_findings.append(f"Identified {len(anti_patterns)} failure patterns")

        summary = self._build_summary(
            outcome, patterns, anti_patterns, lessons
        )

        result = ReviewResult(
            lessons=lessons,
            patterns=patterns,
            anti_patterns=anti_patterns,
            summary=summary,
            success_rate=success_rate,
            efficiency_score=efficiency,
            key_findings=key_findings,
        )

        self._history.append(result)
        return result

    def _identify_success_patterns(
        self,
        trajectory: Trajectory,
        outcome: Outcome,
    ) -> list[Pattern]:
        """Identify successful action patterns from the trajectory.

        Analyzes the trajectory for:
        1. N-gram action sequences that appear in successful episodes
        2. Efficient sub-sequences (short paths to sub-goals)
        3. Reward-correlated action patterns

        Args:
            trajectory: The episode trajectory.
            outcome: The episode outcome.

        Returns:
            List of identified success patterns.
        """
        patterns: list[Pattern] = []

        if not outcome.success or trajectory.length < 2:
            return patterns

        actions = trajectory.actions

        # 1. N-gram analysis: find repeated action sequences
        for n in self.NGRAM_SIZES:
            if len(actions) < n:
                continue
            ngrams: Counter = Counter()
            for i in range(len(actions) - n + 1):
                ngram = tuple(actions[i:i + n])
                ngrams[ngram] += 1

            # Find repeated n-grams (appears more than once)
            for ngram, count in ngrams.items():
                if count >= 2:
                    patterns.append(Pattern(
                        description=f"Repeated {n}-action sequence: {list(ngram)}",
                        conditions={
                            'ngram_size': n,
                            'sequence': list(ngram),
                            'repeat_count': count,
                            'game_id': trajectory.game_id,
                            'level': trajectory.level,
                        },
                        frequency=count,
                        confidence=min(0.9, 0.3 + count * 0.15),
                        action_sequence=list(ngram),
                    ))

        # 2. Efficient prefix patterns
        if outcome.baseline > 0 and outcome.steps < outcome.baseline:
            # The entire trajectory is efficient
            patterns.append(Pattern(
                description=(
                    f"Efficient path: {outcome.steps} steps vs "
                    f"{outcome.baseline} baseline"
                ),
                conditions={
                    'steps': outcome.steps,
                    'baseline': outcome.baseline,
                    'efficiency': outcome.efficiency,
                    'game_id': trajectory.game_id,
                    'level': trajectory.level,
                },
                frequency=1,
                confidence=0.8,
                action_sequence=list(actions),
            ))

        # 3. Reward-correlated patterns
        if trajectory.rewards:
            # Find action sequences followed by positive rewards
            for n in [1, 2, 3]:
                if len(actions) < n + 1:
                    continue
                for i in range(len(actions) - n):
                    reward = trajectory.rewards[i] if i < len(trajectory.rewards) else 0.0
                    if reward > 0:
                        seq = actions[i:i + n]
                        patterns.append(Pattern(
                            description=f"Reward-correlated sequence: {seq}",
                            conditions={
                                'sequence': seq,
                                'reward': reward,
                                'position': i,
                                'game_id': trajectory.game_id,
                            },
                            frequency=1,
                            confidence=0.6,
                            action_sequence=list(seq),
                        ))

        # 4. No-game-over pattern (if episode had no game overs)
        if outcome.success and outcome.game_over_count == 0:
            patterns.append(Pattern(
                description="Clean completion (no game overs)",
                conditions={
                    'game_over_count': 0,
                    'steps': outcome.steps,
                    'game_id': trajectory.game_id,
                    'level': trajectory.level,
                },
                frequency=1,
                confidence=0.7,
                action_sequence=[],
            ))

        return patterns

    def _identify_failure_points(
        self,
        trajectory: Trajectory,
        outcome: Outcome,
    ) -> list[AntiPattern]:
        """Identify failure points and anti-patterns from the trajectory.

        Analyzes the trajectory for:
        1. Actions that led to game-over events
        2. Stagnation patterns (repeated actions without progress)
        3. Inefficient sub-sequences (longer than expected)
        4. Oscillation patterns (back-and-forth movement)

        Args:
            trajectory: The episode trajectory.
            outcome: The episode outcome.

        Returns:
            List of identified anti-patterns.
        """
        anti_patterns: list[AntiPattern] = []
        actions = trajectory.actions

        if not actions:
            return anti_patterns

        # 1. Game-over trigger actions
        # (RESET action = 0, or last action before a game-over event)
        if outcome.game_over_count > 0:
            # Find RESET actions in the trajectory
            reset_indices = [i for i, a in enumerate(actions) if a == 0]
            for idx in reset_indices:
                # The action before RESET likely caused the game over
                if idx > 0:
                    trigger = actions[idx - 1]
                    anti_patterns.append(AntiPattern(
                        description=(
                            f"Action {trigger} preceded game-over at step {idx}"
                        ),
                        conditions={
                            'trigger_action': trigger,
                            'step': idx,
                            'game_id': trajectory.game_id,
                            'level': trajectory.level,
                        },
                        frequency=1,
                        confidence=0.7,
                        trigger_action=trigger,
                    ))

        # 2. Stagnation: repeated same action many times
        if len(actions) >= 5:
            max_consecutive = 1
            current_consecutive = 1
            stagnation_action = actions[0]
            stagnation_start = 0

            for i in range(1, len(actions)):
                if actions[i] == actions[i - 1]:
                    current_consecutive += 1
                    if current_consecutive > max_consecutive:
                        max_consecutive = current_consecutive
                        stagnation_action = actions[i]
                        stagnation_start = i - current_consecutive + 1
                else:
                    current_consecutive = 1

            if max_consecutive >= 8:
                anti_patterns.append(AntiPattern(
                    description=(
                        f"Stagnation: action {stagnation_action} repeated "
                        f"{max_consecutive} times from step {stagnation_start}"
                    ),
                    conditions={
                        'action': stagnation_action,
                        'repeat_count': max_consecutive,
                        'start_step': stagnation_start,
                        'game_id': trajectory.game_id,
                    },
                    frequency=1,
                    confidence=0.6,
                    trigger_action=stagnation_action,
                ))

        # 3. Oscillation: alternating between two actions
        if len(actions) >= 6:
            for i in range(len(actions) - 5):
                window = actions[i:i + 6]
                # Check for A-B-A-B-A-B pattern
                if (window[0] == window[2] == window[4]
                        and window[1] == window[3] == window[5]
                        and window[0] != window[1]):
                    anti_patterns.append(AntiPattern(
                        description=(
                            f"Oscillation between actions {window[0]} and "
                            f"{window[1]} from step {i}"
                        ),
                        conditions={
                            'action_a': window[0],
                            'action_b': window[1],
                            'start_step': i,
                            'game_id': trajectory.game_id,
                        },
                        frequency=1,
                        confidence=0.5,
                        trigger_action=window[0],
                    ))
                    break  # Only report first oscillation

        # 4. Inefficient path (if failed and took too many steps)
        if not outcome.success and outcome.baseline > 0:
            if outcome.steps > outcome.baseline * 2:
                anti_patterns.append(AntiPattern(
                    description=(
                        f"Excessive steps: {outcome.steps} vs "
                        f"{outcome.baseline} baseline (2x over)"
                    ),
                    conditions={
                        'steps': outcome.steps,
                        'baseline': outcome.baseline,
                        'ratio': outcome.steps / outcome.baseline,
                        'game_id': trajectory.game_id,
                        'level': trajectory.level,
                    },
                    frequency=1,
                    confidence=0.4,
                    trigger_action=-1,
                ))

        # 5. Failure without game over (ran out of steps)
        if not outcome.success and outcome.game_over_count == 0:
            anti_patterns.append(AntiPattern(
                description=(
                    f"Step budget exhausted: {outcome.steps} steps without "
                    f"completing level {outcome.level}"
                ),
                conditions={
                    'steps': outcome.steps,
                    'level': outcome.level,
                    'game_id': trajectory.game_id,
                },
                frequency=1,
                confidence=0.5,
                trigger_action=-1,
            ))

        return anti_patterns

    def _extract_lessons(
        self,
        patterns: list[Pattern],
        anti_patterns: list[AntiPattern],
        outcome: Outcome,
    ) -> list[Lesson]:
        """Extract actionable lessons from patterns and anti-patterns.

        Args:
            patterns: Identified success patterns.
            anti_patterns: Identified failure patterns.
            outcome: Episode outcome.

        Returns:
            List of extracted lessons.
        """
        lessons: list[Lesson] = []

        # Success lessons
        if outcome.success:
            lessons.append(Lesson(
                description=(
                    f"Level {outcome.level} solvable in {outcome.steps} steps "
                    f"(baseline: {outcome.baseline})"
                ),
                category='success',
                confidence=0.9,
                evidence=[f"Success with {outcome.steps} steps"],
                applicable_contexts=[f"level_{outcome.level}", outcome.game_id],
            ))

            if outcome.baseline > 0 and outcome.steps < outcome.baseline:
                lessons.append(Lesson(
                    description=(
                        f"Optimal path found: {outcome.steps} steps is "
                        f"{outcome.baseline - outcome.steps} below baseline"
                    ),
                    category='optimization',
                    confidence=0.8,
                    evidence=[f"RHAE={outcome.rhae:.1f}"],
                    applicable_contexts=[f"level_{outcome.level}", outcome.game_id],
                ))

            if outcome.game_over_count == 0:
                lessons.append(Lesson(
                    description=(
                        "Avoid game-overs: clean completion is achievable "
                        "on this level"
                    ),
                    category='safety',
                    confidence=0.7,
                    evidence=["Zero game-over events during successful run"],
                    applicable_contexts=[f"level_{outcome.level}", outcome.game_id],
                ))

        # Pattern-based lessons
        for p in patterns:
            if p.frequency >= 2:
                lessons.append(Lesson(
                    description=(
                        f"Reuse action sequence {p.action_sequence}: "
                        f"appeared {p.frequency} times in successful run"
                    ),
                    category='optimization',
                    confidence=p.confidence,
                    evidence=[p.description],
                    applicable_contexts=[outcome.game_id, f"level_{outcome.level}"],
                ))

        # Anti-pattern-based lessons
        for ap in anti_patterns:
            if ap.trigger_action >= 0:
                lessons.append(Lesson(
                    description=(
                        f"Avoid action {ap.trigger_action} in similar "
                        f"contexts: led to failure"
                    ),
                    category='failure',
                    confidence=ap.confidence,
                    evidence=[ap.description],
                    applicable_contexts=[outcome.game_id, f"level_{outcome.level}"],
                ))

            if 'Stagnation' in ap.description:
                lessons.append(Lesson(
                    description=(
                        "Avoid prolonged repetition of the same action: "
                        "indicates being stuck"
                    ),
                    category='failure',
                    confidence=0.6,
                    evidence=[ap.description],
                    applicable_contexts=[outcome.game_id],
                ))

            if 'Oscillation' in ap.description:
                lessons.append(Lesson(
                    description=(
                        "Avoid oscillating between two actions: indicates "
                        "no progress being made"
                    ),
                    category='failure',
                    confidence=0.6,
                    evidence=[ap.description],
                    applicable_contexts=[outcome.game_id],
                ))

        # Failure lessons
        if not outcome.success:
            lessons.append(Lesson(
                description=(
                    f"Level {outcome.level} not solved in {outcome.steps} steps: "
                    f"consider alternative strategies"
                ),
                category='failure',
                confidence=0.7,
                evidence=[f"Failed with {outcome.steps} steps, {outcome.game_over_count} game-overs"],
                applicable_contexts=[f"level_{outcome.level}", outcome.game_id],
            ))

        return lessons

    def _build_summary(
        self,
        outcome: Outcome,
        patterns: list[Pattern],
        anti_patterns: list[AntiPattern],
        lessons: list[Lesson],
    ) -> str:
        """Build a human-readable summary of the review.

        Args:
            outcome: Episode outcome.
            patterns: Success patterns.
            anti_patterns: Failure patterns.
            lessons: Extracted lessons.

        Returns:
            Summary string.
        """
        status = "SUCCESS" if outcome.success else "FAILURE"
        parts: list[str] = [
            f"After-Action Review: {status}",
            f"  Level {outcome.level} ({outcome.game_id})",
            f"  Steps: {outcome.steps}/{outcome.baseline} baseline",
        ]

        if outcome.game_over_count > 0:
            parts.append(f"  Game-overs: {outcome.game_over_count}")

        if outcome.success and outcome.baseline > 0:
            parts.append(f"  RHAE: {outcome.rhae:.1f}")

        if patterns:
            parts.append(f"  Success patterns: {len(patterns)}")
            for p in patterns[:3]:
                parts.append(f"    + {p.description}")

        if anti_patterns:
            parts.append(f"  Failure patterns: {len(anti_patterns)}")
            for ap in anti_patterns[:3]:
                parts.append(f"    - {ap.description}")

        if lessons:
            parts.append(f"  Lessons: {len(lessons)}")
            for l in lessons[:3]:
                parts.append(f"    * [{l.category}] {l.description}")

        return "\n".join(parts)

    def get_cumulative_patterns(self) -> dict:
        """Get cumulative pattern statistics across all reviews.

        Returns:
            Dictionary with pattern databases and statistics.
        """
        return {
            'total_reviews': self._total_reviews,
            'success_count': self._success_count,
            'success_rate': self._success_count / max(1, self._total_reviews),
            'unique_patterns': len(self._pattern_db),
            'unique_anti_patterns': len(self._anti_pattern_db),
            'unique_lessons': len(self._lesson_db),
            'top_patterns': sorted(
                self._pattern_db.values(),
                key=lambda x: x.frequency,
                reverse=True,
            )[:10],
            'top_anti_patterns': sorted(
                self._anti_pattern_db.values(),
                key=lambda x: x.frequency,
                reverse=True,
            )[:10],
        }


# ============================================================================
# Module B: CognitiveRecursiveDynamics
# ============================================================================

@dataclass
class CognitiveResult:
    """Result of cognitive recursive dynamics reasoning.

    Attributes:
        action: The recommended action (int or None).
        confidence: Confidence in the recommendation (0.0 to 1.0).
        reasoning_chain: Chain of reasoning from L0 to L4.
        meta_assessment: Metacognitive assessment of the reasoning quality.
        layer_outputs: Outputs from each cognitive layer.
        improvements: Suggested improvements to the reasoning process.
    """

    action: Optional[int] = None
    confidence: float = 0.5
    reasoning_chain: list[str] = field(default_factory=list)
    meta_assessment: dict = field(default_factory=dict)
    layer_outputs: dict = field(default_factory=dict)
    improvements: list[str] = field(default_factory=list)


class CognitiveRecursiveDynamics:
    """5-layer cognitive reasoning system with recursive self-improvement.

    Processes perceptions through escalating cognitive layers:

    L0 Perception: Extract features from raw perception data.
        - Entity detection, position tracking, spatial relationships.
    L1 Tactical: Identify immediate actions and consequences.
        - Available actions, short-term rewards, collision risks.
    L2 Strategic: Plan multi-step strategies.
        - Goal selection, path planning, resource management.
    L3 Metacognition: Reflect on the plan's quality.
        - Confidence assessment, bias detection, alternative evaluation.
    L4 Recursive Self-Improvement: Improve the reasoning process itself.
        - Identify reasoning flaws, suggest process improvements.

    Each layer feeds into the next, creating a chain of increasingly
    abstract reasoning. The metacognitive layer (L3) can flag issues
    with the strategic plan, and the recursive layer (L4) can suggest
    improvements to the overall reasoning process.

    Usage:
        cognitive = CognitiveRecursiveDynamics()
        result = cognitive.think(perception_data, context)
        # result.action, result.confidence, result.reasoning_chain
    """

    LAYER_NAMES: list[str] = [
        'L0_Perception',
        'L1_Tactical',
        'L2_Strategic',
        'L3_Metacognition',
        'L4_Recursive',
    ]

    def __init__(self) -> None:
        """Initialize the cognitive reasoning system."""
        self._layer_cache: dict[str, Any] = {}
        self._meta_history: list[dict] = []
        self._improvement_count: int = 0
        self._reasoning_flaws: list[str] = []
        self._confidence_calibration: float = 0.5

    def think(
        self,
        perception: dict,
        context: dict,
    ) -> CognitiveResult:
        """Execute the full 5-layer cognitive reasoning pipeline.

        Args:
            perception: Perception data containing entity positions,
                grid state, and other sensory information.
                Expected keys: 'player_pos', 'goal_positions',
                'wall_positions', 'available_actions', 'grid'.
            context: Contextual information about the current situation.
                Expected keys: 'level', 'game_id', 'step', 'budget',
                'previous_actions', 'danger_positions'.

        Returns:
            CognitiveResult with the recommended action, confidence,
            and reasoning chain.
        """
        reasoning_chain: list[str] = []
        layer_outputs: dict = {}

        # L0: Perception
        l0_output = self._perception_layer(perception, context)
        layer_outputs['L0_Perception'] = l0_output
        reasoning_chain.append(
            f"L0: Detected {l0_output.get('entity_count', 0)} entities, "
            f"player at {l0_output.get('player_pos', 'unknown')}"
        )

        # L1: Tactical
        l1_output = self._tactical_layer(l0_output, context)
        layer_outputs['L1_Tactical'] = l1_output
        reasoning_chain.append(
            f"L1: {len(l1_output.get('available_tactics', []))} tactics, "
            f"best immediate: {l1_output.get('best_tactic', 'none')}"
        )

        # L2: Strategic
        l2_output = self._strategic_layer(l1_output, context)
        layer_outputs['L2_Strategic'] = l2_output
        reasoning_chain.append(
            f"L2: Strategy={l2_output.get('strategy', 'none')}, "
            f"target={l2_output.get('target', 'none')}, "
            f"estimated_steps={l2_output.get('estimated_steps', '?')}"
        )

        # L3: Metacognition
        l3_output = self._meta_reflect(l2_output, context)
        layer_outputs['L3_Metacognition'] = l3_output
        reasoning_chain.append(
            f"L3: Confidence={l3_output.get('confidence', 0):.2f}, "
            f"biases={l3_output.get('biases_detected', [])}, "
            f"alternatives={l3_output.get('alternatives_count', 0)}"
        )

        # L4: Recursive Self-Improvement
        l4_output = self._recursive_improve(l3_output)
        layer_outputs['L4_Recursive'] = l4_output
        reasoning_chain.append(
            f"L4: Improvements={l4_output.get('improvement_count', 0)}, "
            f"flaws_addressed={l4_output.get('flaws_addressed', 0)}"
        )

        # Build final result
        action = l2_output.get('recommended_action')
        confidence = l3_output.get('confidence', 0.5)
        improvements = l4_output.get('improvements', [])

        # Adjust confidence based on L4 assessment
        if l4_output.get('flaws_addressed', 0) > 0:
            confidence = max(0.1, confidence - 0.1)

        # Record meta-history
        self._meta_history.append({
            'step': context.get('step', 0),
            'action': action,
            'confidence': confidence,
            'biases': l3_output.get('biases_detected', []),
            'improvements': len(improvements),
        })

        return CognitiveResult(
            action=action,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            meta_assessment=l3_output,
            layer_outputs=layer_outputs,
            improvements=improvements,
        )

    def _perception_layer(
        self,
        perception: dict,
        context: dict,
    ) -> dict:
        """L0: Extract features from raw perception data.

        Args:
            perception: Raw perception data with entity positions.
            context: Current context.

        Returns:
            Dictionary with extracted features:
            - player_pos, goal_positions, wall_positions
            - entity_count, distances, spatial_layout
        """
        player_pos = perception.get('player_pos')
        goal_positions = perception.get('goal_positions', [])
        wall_positions = perception.get('wall_positions', [])
        available_actions = perception.get('available_actions', [])

        # Compute distances to goals
        distances: list[float] = []
        if player_pos is not None:
            px, py = player_pos
            for goal in goal_positions:
                if isinstance(goal, (tuple, list)) and len(goal) >= 2:
                    gx, gy = goal[0], goal[1]
                    dist = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
                    distances.append(dist)

        # Find nearest goal
        nearest_goal = None
        nearest_dist = float('inf')
        if distances:
            min_idx = int(np.argmin(distances))
            nearest_goal = goal_positions[min_idx] if min_idx < len(goal_positions) else None
            nearest_dist = float(distances[min_idx])

        # Spatial layout analysis
        wall_count = len(wall_positions)
        goal_count = len(goal_positions)
        entity_count = wall_count + goal_count + (1 if player_pos else 0)

        return {
            'player_pos': player_pos,
            'goal_positions': goal_positions,
            'wall_positions': wall_positions,
            'available_actions': available_actions,
            'distances_to_goals': distances,
            'nearest_goal': nearest_goal,
            'nearest_distance': nearest_dist,
            'wall_count': wall_count,
            'goal_count': goal_count,
            'entity_count': entity_count,
            'has_walls': wall_count > 0,
            'has_goals': goal_count > 0,
        }

    def _tactical_layer(
        self,
        perception_data: dict,
        context: dict,
    ) -> dict:
        """L1: Identify immediate tactical options.

        Args:
            perception_data: Output from L0 perception layer.
            context: Current context.

        Returns:
            Dictionary with tactical analysis:
            - available_tactics, best_tactic, collision_risks
            - immediate_rewards, recommended_action
        """
        player_pos = perception_data.get('player_pos')
        goals = perception_data.get('goal_positions', [])
        walls = set(perception_data.get('wall_positions', []))
        available_actions = perception_data.get('available_actions', [])
        danger_positions = set(context.get('danger_positions', []))

        tactics: list[dict] = []
        collision_risks: list[int] = []

        # Action mapping: 1=up, 2=down, 3=left, 4=right
        action_deltas = {
            1: (0, -1, 'up'),
            2: (0, 1, 'down'),
            3: (-1, 0, 'left'),
            4: (1, 0, 'right'),
        }

        if player_pos is not None:
            px, py = player_pos

            for action_id in available_actions:
                if action_id in action_deltas:
                    dx, dy, name = action_deltas[action_id]
                    nx, ny = px + dx, py + dy

                    # Check collision
                    is_collision = (nx, ny) in walls
                    is_danger = (nx, ny) in danger_positions

                    if is_collision:
                        collision_risks.append(action_id)
                        continue

                    # Compute tactical value (distance reduction to nearest goal)
                    tactical_value = 0.0
                    if perception_data.get('distances_to_goals'):
                        current_dist = perception_data['nearest_distance']
                        new_dist = float('inf')
                        for goal in goals:
                            if isinstance(goal, (tuple, list)) and len(goal) >= 2:
                                d = ((nx - goal[0]) ** 2 + (ny - goal[1]) ** 2) ** 0.5
                                new_dist = min(new_dist, d)
                        tactical_value = current_dist - new_dist

                    tactics.append({
                        'action': action_id,
                        'name': name,
                        'new_pos': (nx, ny),
                        'tactical_value': tactical_value,
                        'is_danger': is_danger,
                    })

        # Sort tactics by value (highest first)
        tactics.sort(key=lambda x: x['tactical_value'], reverse=True)

        best_tactic = tactics[0] if tactics else None
        recommended_action = best_tactic['action'] if best_tactic else None

        # If all actions are collisions, recommend a non-collision action
        if not tactics and available_actions:
            non_reset = [a for a in available_actions if a != 0]
            recommended_action = non_reset[0] if non_reset else available_actions[0]

        return {
            'available_tactics': tactics,
            'best_tactic': best_tactic['name'] if best_tactic else 'none',
            'collision_risks': collision_risks,
            'recommended_action': recommended_action,
            'tactic_count': len(tactics),
            'danger_actions': [t['action'] for t in tactics if t['is_danger']],
        }

    def _strategic_layer(
        self,
        tactical_data: dict,
        context: dict,
    ) -> dict:
        """L2: Develop multi-step strategy.

        Args:
            tactical_data: Output from L1 tactical layer.
            context: Current context.

        Returns:
            Dictionary with strategic analysis:
            - strategy, target, estimated_steps, recommended_action
        """
        recommended_action = tactical_data.get('recommended_action')
        tactic_count = tactical_data.get('tactic_count', 0)
        collision_risks = tactical_data.get('collision_risks', [])
        budget = context.get('budget', 2000)
        step = context.get('step', 0)
        level = context.get('level', 0)

        # Determine strategy based on tactical analysis
        if tactic_count == 0:
            strategy = 'exploration'
            target = 'unknown'
            estimated_steps = budget
        elif len(collision_risks) >= 3:
            strategy = 'cautious_navigation'
            target = 'safe_position'
            estimated_steps = 10
        elif tactic_count == 1:
            strategy = 'forced_path'
            target = 'only_option'
            estimated_steps = 5
        else:
            strategy = 'greedy_approach'
            target = 'nearest_goal'
            estimated_steps = 5

        # Adjust strategy based on budget
        remaining_budget = budget - step
        if remaining_budget < budget * 0.2:
            strategy = 'emergency'
            target = 'any_progress'

        # Check previous actions for pattern continuation
        prev_actions = context.get('previous_actions', [])
        if len(prev_actions) >= 3:
            last_3 = prev_actions[-3:]
            if len(set(last_3)) == 1 and last_3[0] == recommended_action:
                # Repeating same action — might be on the right track
                strategy = 'continuation'
            elif last_3[0] == last_3[2] and last_3[0] != last_3[1]:
                # Oscillation detected — change strategy
                strategy = 'break_oscillation'
                # Recommend a different action
                available = context.get('available_actions', [])
                alt_actions = [a for a in available if a != recommended_action and a != 0]
                if alt_actions:
                    recommended_action = alt_actions[0]

        return {
            'strategy': strategy,
            'target': target,
            'estimated_steps': estimated_steps,
            'recommended_action': recommended_action,
            'remaining_budget': remaining_budget,
            'budget_ratio': remaining_budget / max(1, budget),
        }

    def _meta_reflect(
        self,
        strategic_data: dict,
        context: dict,
    ) -> dict:
        """L3: Metacognitive reflection on the strategic plan.

        Reflects on the quality of the reasoning:
        - Am I confident in this plan?
        - Are there biases in my reasoning?
        - What alternatives exist?
        - Is my understanding of the problem correct?

        Args:
            strategic_data: Output from L2 strategic layer.
            context: Current context.

        Returns:
            Dictionary with metacognitive assessment:
            - confidence, biases_detected, alternatives_count
            - understanding_quality, risk_assessment
        """
        strategy = strategic_data.get('strategy', 'unknown')
        estimated_steps = strategic_data.get('estimated_steps', 0)
        budget_ratio = strategic_data.get('budget_ratio', 1.0)
        remaining_budget = strategic_data.get('remaining_budget', 0)

        # Confidence assessment
        confidence = 0.5
        if strategy == 'greedy_approach':
            confidence = 0.7
        elif strategy == 'forced_path':
            confidence = 0.4
        elif strategy == 'exploration':
            confidence = 0.3
        elif strategy == 'cautious_navigation':
            confidence = 0.6
        elif strategy == 'continuation':
            confidence = 0.65
        elif strategy == 'emergency':
            confidence = 0.2
        elif strategy == 'break_oscillation':
            confidence = 0.55

        # Budget confidence adjustment
        if budget_ratio < 0.2:
            confidence *= 0.7  # Low budget reduces confidence
        elif budget_ratio > 0.8:
            confidence *= 1.1  # High budget increases confidence

        confidence = min(1.0, max(0.0, confidence))

        # Bias detection
        biases_detected: list[str] = []

        # Confirmation bias: sticking with same strategy
        prev_actions = context.get('previous_actions', [])
        if len(prev_actions) >= 5:
            recent_actions = prev_actions[-5:]
            if len(set(recent_actions)) <= 2:
                biases_detected.append('repetition_bias')

        # Optimism bias: underestimating steps needed
        if estimated_steps > 0 and remaining_budget > 0:
            if estimated_steps > remaining_budget * 0.5:
                biases_detected.append('optimism_bias')

        # Availability bias: over-relying on recent success
        if context.get('recent_success', False) and confidence > 0.8:
            biases_detected.append('availability_bias')

        # Alternatives count
        alternatives_count = 0
        if strategy != 'forced_path':
            alternatives_count += 1
        if strategy != 'exploration':
            alternatives_count += 1
        if budget_ratio > 0.3:
            alternatives_count += 1

        # Understanding quality
        understanding_quality = 'moderate'
        if strategy in ('greedy_approach', 'continuation'):
            understanding_quality = 'good'
        elif strategy in ('exploration', 'emergency'):
            understanding_quality = 'poor'

        # Risk assessment
        risk_level = 'low'
        if budget_ratio < 0.2:
            risk_level = 'high'
        elif budget_ratio < 0.4:
            risk_level = 'medium'
        if biases_detected:
            risk_level = 'high' if len(biases_detected) >= 2 else 'medium'

        # Apply confidence calibration
        confidence = confidence * self._confidence_calibration + confidence * (1 - self._confidence_calibration) * 0.8
        confidence = min(1.0, max(0.0, confidence))

        return {
            'confidence': confidence,
            'biases_detected': biases_detected,
            'alternatives_count': alternatives_count,
            'understanding_quality': understanding_quality,
            'risk_level': risk_level,
            'strategy_assessment': f"Strategy '{strategy}' is {understanding_quality}",
            'estimated_accuracy': confidence,
        }

    def _recursive_improve(
        self,
        meta_data: dict,
    ) -> dict:
        """L4: Recursive self-improvement of the reasoning process.

        Identifies flaws in the reasoning process and suggests
        improvements. This is the highest cognitive layer — it
        reflects on the reasoning itself rather than the problem.

        Args:
            meta_data: Output from L3 metacognitive layer.

        Returns:
            Dictionary with improvement suggestions:
            - improvements, improvement_count, flaws_addressed
            - process_adjustments
        """
        improvements: list[str] = []
        flaws_addressed = 0

        confidence = meta_data.get('confidence', 0.5)
        biases = meta_data.get('biases_detected', [])
        understanding = meta_data.get('understanding_quality', 'moderate')
        risk_level = meta_data.get('risk_level', 'low')

        # Address low confidence
        if confidence < 0.3:
            improvements.append(
                "Low confidence detected: consider gathering more information "
                "before committing to an action"
            )
            flaws_addressed += 1

        # Address biases
        if 'repetition_bias' in biases:
            improvements.append(
                "Repetition bias detected: try exploring alternative actions "
                "to break out of repetitive patterns"
            )
            flaws_addressed += 1

        if 'optimism_bias' in biases:
            improvements.append(
                "Optimism bias detected: revise step estimates upward and "
                "consider conservative alternatives"
            )
            flaws_addressed += 1

        if 'availability_bias' in biases:
            improvements.append(
                "Availability bias detected: recent success may not generalize; "
                "evaluate each situation independently"
            )
            flaws_addressed += 1

        # Address poor understanding
        if understanding == 'poor':
            improvements.append(
                "Poor problem understanding: invest more steps in exploration "
                "to build a better mental model"
            )
            flaws_addressed += 1

        # Address high risk
        if risk_level == 'high':
            improvements.append(
                "High risk situation: prioritize safety over progress; "
                "avoid dangerous positions"
            )
            flaws_addressed += 1

        # Process-level improvements (meta-meta-cognition)
        if len(self._meta_history) >= 5:
            recent = self._meta_history[-5:]
            avg_confidence = sum(r.get('confidence', 0) for r in recent) / len(recent)
            if avg_confidence < 0.4:
                improvements.append(
                    "Systematic low confidence over recent steps: "
                    "consider fundamentally rethinking the approach"
                )
                flaws_addressed += 1

            bias_frequency = sum(
                len(r.get('biases', [])) for r in recent
            ) / len(recent)
            if bias_frequency > 1.0:
                improvements.append(
                    "Frequent bias detection: implement bias-mitigation "
                    "strategies in the tactical layer"
                )
                flaws_addressed += 1

        # Update calibration based on history
        if len(self._meta_history) >= 10:
            recent_confidences = [r.get('confidence', 0.5) for r in self._meta_history[-10:]]
            avg_conf = sum(recent_confidences) / len(recent_confidences)
            # Gradually adjust calibration toward observed average
            self._confidence_calibration = (
                0.9 * self._confidence_calibration + 0.1 * avg_conf
            )

        self._improvement_count += len(improvements)

        # Track reasoning flaws
        if flaws_addressed > 0:
            self._reasoning_flaws.extend(improvements[:2])  # Keep last 2

        return {
            'improvements': improvements,
            'improvement_count': len(improvements),
            'flaws_addressed': flaws_addressed,
            'total_improvements_made': self._improvement_count,
            'process_adjustments': {
                'confidence_calibration': self._confidence_calibration,
                'recent_flaw_count': len(self._reasoning_flaws),
            },
        }

    def get_cognitive_history(self) -> list[dict]:
        """Get the history of cognitive assessments.

        Returns:
            List of historical cognitive assessment dictionaries.
        """
        return list(self._meta_history)

    def get_reasoning_flaws(self) -> list[str]:
        """Get recently identified reasoning flaws.

        Returns:
            List of recent reasoning flaw descriptions.
        """
        return list(self._reasoning_flaws)


# ============================================================================
# Module C: OperatorAccumulator
# ============================================================================

@dataclass
class Operator:
    """A reusable operator extracted from experience.

    Operators represent generalized action sequences that achieve a
    specific sub-goal. They can be retrieved and applied in similar
    contexts to accelerate problem-solving.

    Attributes:
        name: Human-readable operator name.
        precondition: Conditions that must hold for the operator to apply.
        action_sequence: Sequence of actions to execute.
        effect: Expected outcome of executing the operator.
        game_id: Game identifier this operator was learned from.
        confidence: Confidence score (0.0 to 1.0).
        use_count: Number of times this operator has been used.
        success_count: Number of successful applications.
        extraction_level: Level the operator was extracted from.
    """

    name: str = ""
    precondition: dict = field(default_factory=dict)
    action_sequence: list[int] = field(default_factory=list)
    effect: dict = field(default_factory=dict)
    game_id: str = ""
    confidence: float = 0.5
    use_count: int = 0
    success_count: int = 0
    extraction_level: int = -1

    @property
    def success_rate(self) -> float:
        """Success rate (success_count / use_count)."""
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count

    def matches_context(self, context: dict) -> bool:
        """Check if this operator's preconditions match a context.

        Args:
            context: Context dictionary to match against.

        Returns:
            True if all precondition keys are satisfied by the context.
        """
        for key, value in self.precondition.items():
            if key not in context:
                continue
            ctx_value = context[key]
            if isinstance(value, (set, list)):
                if ctx_value not in value:
                    return False
            elif value != ctx_value:
                return False
        return True

    def to_dict(self) -> dict:
        """Convert operator to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the operator.
        """
        return {
            'name': self.name,
            'precondition': dict(self.precondition),
            'action_sequence': list(self.action_sequence),
            'effect': dict(self.effect),
            'game_id': self.game_id,
            'confidence': self.confidence,
            'use_count': self.use_count,
            'success_count': self.success_count,
            'extraction_level': self.extraction_level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Operator':
        """Create an operator from a dictionary.

        Args:
            data: Dictionary with operator fields.

        Returns:
            Operator instance.
        """
        return cls(
            name=data.get('name', ''),
            precondition=data.get('precondition', {}),
            action_sequence=data.get('action_sequence', []),
            effect=data.get('effect', {}),
            game_id=data.get('game_id', ''),
            confidence=data.get('confidence', 0.5),
            use_count=data.get('use_count', 0),
            success_count=data.get('success_count', 0),
            extraction_level=data.get('extraction_level', -1),
        )


class OperatorAccumulator:
    """Accumulates and manages reusable operators from experience.

    Extracts operators (macro-actions with preconditions and effects)
    from successful trajectories. Operators are generalized from
    specific instances and can be retrieved by context matching.

    The operator library can be persisted to disk as JSON for
    cross-session learning.

    Usage:
        accumulator = OperatorAccumulator()
        op = accumulator.extract_operator(trajectory, outcome)
        ops = accumulator.retrieve_operators(context)
        accumulator.save_library("operators.json")
    """

    # Minimum action sequence length for operator extraction
    MIN_SEQUENCE_LENGTH: int = 3
    # Maximum action sequence length for operator extraction
    MAX_SEQUENCE_LENGTH: int = 50
    # N-gram sizes for macro action detection
    MACRO_NGRAM_SIZES: list[int] = [3, 4, 5, 6, 7, 8]
    # Minimum frequency for a macro to be extracted
    MIN_MACRO_FREQUENCY: int = 1

    def __init__(self) -> None:
        """Initialize the operator accumulator."""
        self._library: dict[str, Operator] = {}
        self._macro_db: dict[str, list[tuple[int, int]]] = {}
        self._extraction_count: int = 0

    def extract_operator(
        self,
        trajectory: Trajectory,
        outcome: Outcome,
    ) -> Optional[Operator]:
        """Extract a reusable operator from a trajectory.

        Finds macro-actions in the trajectory and generalizes them
        into an operator with preconditions and effects.

        Args:
            trajectory: The episode trajectory.
            outcome: The episode outcome.

        Returns:
            Extracted Operator, or None if no suitable macro found.
        """
        if not outcome.success or trajectory.length < self.MIN_SEQUENCE_LENGTH:
            return None

        macros = self._find_macro_actions(trajectory)
        if not macros:
            return None

        # Pick the most promising macro (longest, most frequent)
        best_macro = max(macros, key=lambda m: len(m[0]) * m[2])

        operator = self._generalize_operator(best_macro, trajectory, outcome)
        if operator is not None:
            op_key = operator.name
            if op_key in self._library:
                # Update existing operator
                existing = self._library[op_key]
                existing.use_count += 1
                existing.success_count += 1
                existing.confidence = min(
                    1.0, existing.confidence + 0.05
                )
            else:
                operator.use_count = 1
                operator.success_count = 1
                self._library[op_key] = operator

            self._extraction_count += 1

        return operator

    def _find_macro_actions(
        self,
        trajectory: Trajectory,
    ) -> list[tuple[list[int], int, int]]:
        """Find macro-actions (repeated action sequences) in a trajectory.

        Uses n-gram analysis to identify action sequences that appear
        multiple times. Each macro is a candidate for operator extraction.

        Args:
            trajectory: The episode trajectory.

        Returns:
            List of (action_sequence, start_position, frequency) tuples.
        """
        actions = trajectory.actions
        macros: list[tuple[list[int], int, int]] = []

        if len(actions) < self.MIN_SEQUENCE_LENGTH:
            return macros

        for n in self.MACRO_NGRAM_SIZES:
            if len(actions) < n:
                continue

            # Count n-gram frequencies
            ngram_positions: dict[tuple, list[int]] = defaultdict(list)
            for i in range(len(actions) - n + 1):
                ngram = tuple(actions[i:i + n])
                ngram_positions[ngram].append(i)

            # Find repeated n-grams
            for ngram, positions in ngram_positions.items():
                freq = len(positions)
                if freq >= self.MIN_MACRO_FREQUENCY:
                    macros.append((list(ngram), positions[0], freq))

        # Also find the overall trajectory as a macro (if successful)
        if len(actions) <= self.MAX_SEQUENCE_LENGTH:
            macros.append((list(actions), 0, 1))

        # Sort by length * frequency (longer and more frequent = better)
        macros.sort(key=lambda m: len(m[0]) * m[2], reverse=True)

        return macros

    def _generalize_operator(
        self,
        macro: tuple[list[int], int, int],
        trajectory: Trajectory,
        outcome: Outcome,
    ) -> Optional[Operator]:
        """Generalize a macro-action into a reusable operator.

        Converts a specific action sequence into a generalized operator
        by:
        1. Extracting preconditions from the trajectory context
        2. Abstracting the action sequence
        3. Describing the expected effect

        Args:
            macro: (action_sequence, start_position, frequency) tuple.
            trajectory: The source trajectory.
            outcome: The episode outcome.

        Returns:
            Generalized Operator, or None if generalization fails.
        """
        action_sequence, start_pos, frequency = macro

        if len(action_sequence) < self.MIN_SEQUENCE_LENGTH:
            return None

        # Build precondition from trajectory context
        precondition: dict = {
            'game_id': trajectory.game_id,
            'level': trajectory.level,
        }

        # Add state-based preconditions if available
        if start_pos < len(trajectory.states):
            state = trajectory.states[start_pos]
            if isinstance(state, dict):
                # Extract relevant state features
                for key in ['player_pos', 'goal_count', 'wall_count']:
                    if key in state:
                        precondition[key] = state[key]
            elif isinstance(state, np.ndarray):
                precondition['state_shape'] = list(state.shape)

        # Build effect description
        effect: dict = {
            'outcome': 'level_complete' if outcome.success else 'level_failed',
            'steps': len(action_sequence),
            'frequency': frequency,
        }

        # Check if the macro led to positive rewards
        if start_pos < len(trajectory.rewards):
            end_pos = min(start_pos + len(action_sequence), len(trajectory.rewards))
            segment_rewards = trajectory.rewards[start_pos:end_pos]
            if segment_rewards:
                effect['total_reward'] = sum(segment_rewards)
                effect['avg_reward'] = sum(segment_rewards) / len(segment_rewards)
                effect['max_reward'] = max(segment_rewards)

        # Generate operator name
        action_str = '-'.join(str(a) for a in action_sequence[:5])
        if len(action_sequence) > 5:
            action_str += f'-...({len(action_sequence)} actions)'
        name = f"op_{trajectory.game_id}_L{outcome.level}_{action_str}"

        # Compute initial confidence
        confidence = 0.5
        if outcome.success and outcome.game_over_count == 0:
            confidence = 0.7
        if frequency > 1:
            confidence = min(0.9, confidence + 0.1 * frequency)
        if outcome.baseline > 0 and outcome.steps < outcome.baseline:
            confidence = min(0.95, confidence + 0.1)

        return Operator(
            name=name,
            precondition=precondition,
            action_sequence=list(action_sequence),
            effect=effect,
            game_id=trajectory.game_id,
            confidence=confidence,
            extraction_level=outcome.level,
        )

    def retrieve_operators(
        self,
        context: dict,
        max_results: int = 10,
    ) -> list[Operator]:
        """Retrieve applicable operators for a given context.

        Matches the context against operator preconditions and returns
        matching operators sorted by confidence and success rate.

        Args:
            context: Context dictionary with game_id, level, etc.
            max_results: Maximum number of operators to return.

        Returns:
            List of applicable Operators, sorted by relevance.
        """
        matching: list[Operator] = []

        for op in self._library.values():
            if op.matches_context(context):
                matching.append(op)

        # Sort by confidence * success_rate, then by use_count
        matching.sort(
            key=lambda o: (o.confidence * (0.5 + o.success_rate * 0.5), o.use_count),
            reverse=True,
        )

        return matching[:max_results]

    def record_application(
        self,
        operator_name: str,
        success: bool,
    ) -> None:
        """Record the outcome of applying an operator.

        Updates the operator's use and success counts, and adjusts
        its confidence based on the outcome.

        Args:
            operator_name: Name of the operator that was applied.
            success: Whether the application was successful.
        """
        op = self._library.get(operator_name)
        if op is None:
            return

        op.use_count += 1
        if success:
            op.success_count += 1
            op.confidence = min(0.99, op.confidence + 0.02)
        else:
            op.confidence = max(0.05, op.confidence - 0.05)

    def save_library(self, path: str) -> None:
        """Save the operator library to a JSON file.

        Args:
            path: File path to save to.
        """
        data = {
            'version': '1.0',
            'extraction_count': self._extraction_count,
            'operators': [op.to_dict() for op in self._library.values()],
        }

        # Ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_library(self, path: str) -> None:
        """Load the operator library from a JSON file.

        Args:
            path: File path to load from.
        """
        if not os.path.exists(path):
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._extraction_count = data.get('extraction_count', 0)
        self._library.clear()

        for op_data in data.get('operators', []):
            op = Operator.from_dict(op_data)
            self._library[op.name] = op

    def get_operator_stats(self) -> dict:
        """Get statistics about the operator library.

        Returns:
            Dictionary with library statistics.
        """
        total_uses = sum(op.use_count for op in self._library.values())
        total_successes = sum(op.success_count for op in self._library.values())
        avg_confidence = (
            sum(op.confidence for op in self._library.values())
            / max(1, len(self._library))
        )

        # Group by game_id
        by_game: dict[str, int] = defaultdict(int)
        for op in self._library.values():
            by_game[op.game_id] += 1

        # Top operators by use count
        top_ops = sorted(
            self._library.values(),
            key=lambda o: o.use_count,
            reverse=True,
        )[:10]

        return {
            'total_operators': len(self._library),
            'total_extractions': self._extraction_count,
            'total_uses': total_uses,
            'total_successes': total_successes,
            'overall_success_rate': total_successes / max(1, total_uses),
            'avg_confidence': avg_confidence,
            'operators_by_game': dict(by_game),
            'top_operators': [
                {
                    'name': op.name,
                    'use_count': op.use_count,
                    'success_rate': op.success_rate,
                    'confidence': op.confidence,
                    'sequence_length': len(op.action_sequence),
                }
                for op in top_ops
            ],
        }

    def get_all_operators(self) -> list[Operator]:
        """Get all operators in the library.

        Returns:
            List of all Operator instances.
        """
        return list(self._library.values())

    def clear_library(self) -> None:
        """Clear all operators from the library."""
        self._library.clear()
        self._macro_db.clear()
        self._extraction_count = 0


# ============================================================================
# v3.1: Psi Audit System — ψ锚点审计 + MUS双存 + 对齐伪装检测
# ============================================================================

@dataclass
class PsiAuditEntry:
    """A single ψ audit record for traceability and alignment monitoring.

    Each entry captures a decision point in the agent's reasoning
    pipeline, enabling post-hoc analysis of decision quality and
    detection of alignment faking.

    Attributes:
        step: Global step counter.
        node_id: Identifier for the decision node (e.g., program hash).
        fidelity: Information fidelity score from GaussEx Dead-Zero gate.
        decision_basis: What the decision was based on (e.g., "gaussex_pass",
            "dead_zero_reject", "mus_stored", "bayesian_fusion").
        confidence: Agent's self-reported confidence [0, 1].
        psi_hash: ψ-anchor hash for consistency tracking.
        metadata: Additional contextual metadata.
        timestamp: When the entry was recorded (Unix epoch).
    """

    step: int = 0
    node_id: str = ""
    fidelity: float = 0.0
    decision_basis: str = ""
    confidence: float = 0.5
    psi_hash: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PsiAuditor:
    """ψ审计器：追踪决策质量，检测对齐伪装。

    Core capabilities:
    1. **Step-level audit logging**: Records every decision with fidelity,
       confidence, and decision basis for complete traceability.
    2. **ψ-anchor consistency detection**: Monitors whether the agent's
       self-reported confidence is consistent with actual fidelity.
       Inconsistency → Potential Alignment Faking.
    3. **Confidence-Fidelity Gap tracking**: Computes the gap between
       what the agent claims (confidence) and what GaussEx measures
       (fidelity). Systematic gaps indicate misalignment.
    4. **MUS preservation**: Tracks mutually exclusive hypotheses that
       were stored for later resolution.

    Usage:
        auditor = PsiAuditor()
        auditor.log_step(step=0, node_id="prog_abc", fidelity=0.85,
                         decision_basis="gaussex_pass", confidence=0.7)
        # ... many steps later ...
        audit = auditor.get_audit_report()
        if audit["alignment_faking_detected"]:
            print("WARNING: Potential alignment faking!")
    """

    # Threshold for detecting alignment faking
    CONFIDENCE_FIDELITY_GAP_THRESHOLD: float = 0.3
    # Window size for trend detection
    TREND_WINDOW: int = 10

    def __init__(self) -> None:
        """Initialize the ψ auditor."""
        self._audit_trail: list[PsiAuditEntry] = []
        self._total_steps: int = 0
        self._alignment_warnings: list[dict] = []
        self._mus_preserved: list[dict] = []

        # Aggregated statistics
        self._fidelity_sum: float = 0.0
        self._confidence_sum: float = 0.0
        self._dead_zero_count: int = 0
        self._mus_count: int = 0

    def log_step(
        self,
        step: int,
        node_id: str,
        fidelity: float,
        decision_basis: str,
        confidence: float = 0.5,
        metadata: dict | None = None,
    ) -> int:
        """Log a single decision step to the ψ audit trail.

        Args:
            step: Global step counter.
            node_id: Decision node identifier.
            fidelity: GaussEx information fidelity score.
            decision_basis: Basis for the decision.
            confidence: Agent's self-reported confidence [0, 1].
            metadata: Extra contextual data.

        Returns:
            Entry index in audit trail.
        """
        # Compute ψ-hash for consistency tracking
        psi_hash = hashlib.md5(
            f"{node_id}{fidelity:.4f}{decision_basis}{confidence:.4f}".encode()
        ).hexdigest()[:8]

        entry = PsiAuditEntry(
            step=step,
            node_id=node_id,
            fidelity=fidelity,
            decision_basis=decision_basis,
            confidence=confidence,
            psi_hash=psi_hash,
            metadata=metadata or {},
            timestamp=time.time(),
        )

        self._audit_trail.append(entry)
        self._total_steps += 1
        self._fidelity_sum += fidelity
        self._confidence_sum += confidence

        if decision_basis == "dead_zero_reject":
            self._dead_zero_count += 1

        # Real-time alignment faking check
        self._check_alignment_faking(entry)

        return len(self._audit_trail) - 1

    def log_mus_preservation(
        self,
        cell_id: str,
        prog_a_id: str,
        prog_b_id: str,
        reason: str = "",
    ) -> None:
        """Log a MUS (Mutually Exclusive System) preservation event.

        When two contradictory but individually valid programs are
        detected, both are preserved rather than prematurely discarded.
        This records the event for later analysis.

        Args:
            cell_id: MUS cell identifier.
            prog_a_id: First program's node ID.
            prog_b_id: Second program's node ID.
            reason: Why MUS preservation was triggered.
        """
        self._mus_preserved.append({
            "cell_id": cell_id,
            "prog_a_id": prog_a_id,
            "prog_b_id": prog_b_id,
            "reason": reason,
            "timestamp": time.time(),
        })
        self._mus_count += 1

    def _check_alignment_faking(self, entry: PsiAuditEntry) -> None:
        """Check for alignment faking patterns in the audit trail.

        Alignment faking indicators:
        1. **Confidence-Fidelity Gap**: Agent claims high confidence
           but fidelity is low. Gap > threshold → suspicious.
        2. **Trend Divergence**: Confidence increases while fidelity
           decreases → potential self-deception.
        3. **ψ-Hash Inconsistency**: Same node_id/program produces
           different ψ-hashes at different times → inconsistency.

        Args:
            entry: The newly added audit entry.
        """
        # Indicator 1: Confidence-Fidelity Gap
        gap = entry.confidence - entry.fidelity
        if gap > self.CONFIDENCE_FIDELITY_GAP_THRESHOLD:
            self._alignment_warnings.append({
                "type": "confidence_fidelity_gap",
                "step": entry.step,
                "node_id": entry.node_id,
                "gap": gap,
                "confidence": entry.confidence,
                "fidelity": entry.fidelity,
                "severity": "HIGH" if gap > 0.5 else "MEDIUM",
            })

        # Indicator 2: Trend divergence (over recent window)
        if len(self._audit_trail) >= self.TREND_WINDOW:
            recent = self._audit_trail[-self.TREND_WINDOW:]
            confidences = [e.confidence for e in recent]
            fidelities = [e.fidelity for e in recent]

            # Linear trend (simple slope)
            conf_slope = _linear_trend(confidences)
            fid_slope = _linear_trend(fidelities)

            # Confidence rises while fidelity falls → alarm
            if conf_slope > 0.01 and fid_slope < -0.01:
                self._alignment_warnings.append({
                    "type": "trend_divergence",
                    "step": entry.step,
                    "conf_trend": conf_slope,
                    "fid_trend": fid_slope,
                    "severity": "MEDIUM",
                })

        # Indicator 3: ψ-hash inconsistency for same node
        same_node_entries = [e for e in self._audit_trail
                            if e.node_id == entry.node_id and e.psi_hash != entry.psi_hash]
        if len(same_node_entries) >= 3:
            self._alignment_warnings.append({
                "type": "psi_hash_inconsistency",
                "step": entry.step,
                "node_id": entry.node_id,
                "inconsistent_count": len(same_node_entries),
                "severity": "HIGH",
            })

    def detect_psi_inconsistency(self) -> dict[str, Any]:
        """Run full ψ inconsistency detection across the audit trail.

        Checks for:
        - ψ-anchor drift: same program evaluated differently over time
        - Decision reversals: pass → reject → pass cycles
        - Confidence inflation: monotonically increasing confidence
          despite flat/decreasing fidelity

        Returns:
            Dict with detection results and recommendations.
        """
        if len(self._audit_trail) < 5:
            return {
                "inconsistency_detected": False,
                "reason": "insufficient_data",
                "recommendations": [],
            }

        # 1. ψ-anchor drift
        node_groups: dict[str, list[float]] = defaultdict(list)
        for entry in self._audit_trail:
            node_groups[entry.node_id].append(entry.fidelity)

        drift_nodes = []
        for node_id, fidelities in node_groups.items():
            if len(fidelities) >= 3:
                fid_range = max(fidelities) - min(fidelities)
                if fid_range > 0.3:  # Significant fidelity swing
                    drift_nodes.append({
                        "node_id": node_id,
                        "fidelity_range": fid_range,
                        "evaluations": len(fidelities),
                    })

        # 2. Decision reversals
        decisions = [e.decision_basis for e in self._audit_trail]
        reversal_count = 0
        for i in range(len(decisions) - 2):
            if decisions[i] != decisions[i + 1] and decisions[i + 1] != decisions[i + 2]:
                reversal_count += 1

        # 3. Confidence inflation
        if len(self._audit_trail) >= self.TREND_WINDOW:
            recent = self._audit_trail[-self.TREND_WINDOW:]
            confs = [e.confidence for e in recent]
            fids = [e.fidelity for e in recent]
            confs_increasing = all(confs[i] <= confs[i + 1] for i in range(len(confs) - 1))
            avg_fid_trend = _linear_trend(fids)
            confidence_inflation = confs_increasing and avg_fid_trend < 0.01
        else:
            confidence_inflation = False

        recommendations = []
        if drift_nodes:
            recommendations.append(
                f"ψ-anchor drift detected for {len(drift_nodes)} nodes. "
                "Review the evaluation pipeline for non-determinism."
            )
        if reversal_count > len(self._audit_trail) * 0.1:
            recommendations.append(
                "Frequent decision reversals detected. Consider Bayesian "
                "fusion to stabilize decisions."
            )
        if confidence_inflation:
            recommendations.append(
                "Confidence inflation detected: agent self-reports increasing "
                "confidence despite flat fidelity. Potential alignment faking."
            )

        is_inconsistent = bool(drift_nodes) or confidence_inflation or \
            reversal_count > len(self._audit_trail) * 0.15

        return {
            "inconsistency_detected": is_inconsistent,
            "psi_drift_nodes": drift_nodes,
            "decision_reversals": reversal_count,
            "confidence_inflation": confidence_inflation,
            "total_alignment_warnings": len(self._alignment_warnings),
            "recommendations": recommendations,
        }

    def get_audit_report(self) -> dict[str, Any]:
        """Generate a comprehensive ψ audit report.

        Returns:
            Dict with complete audit statistics and alignment assessment.
        """
        avg_fidelity = self._fidelity_sum / max(1, self._total_steps)
        avg_confidence = self._confidence_sum / max(1, self._total_steps)
        gap = avg_confidence - avg_fidelity

        # Alignment faking probability (heuristic)
        # Based on: gap size + warning count + trend divergence count
        gap_score = min(1.0, abs(gap) / self.CONFIDENCE_FIDELITY_GAP_THRESHOLD)
        warning_score = min(1.0, len(self._alignment_warnings) / 10.0)
        trend_divergences = sum(
            1 for w in self._alignment_warnings if w["type"] == "trend_divergence"
        )
        trend_score = min(1.0, trend_divergences / 3.0)

        alignment_faking_prob = 0.4 * gap_score + 0.35 * warning_score + 0.25 * trend_score
        alignment_faking_detected = alignment_faking_prob > 0.5

        # Decision distribution
        decision_counts = defaultdict(int)
        for entry in self._audit_trail:
            decision_counts[entry.decision_basis] += 1

        return {
            "total_steps": self._total_steps,
            "avg_fidelity": avg_fidelity,
            "avg_confidence": avg_confidence,
            "confidence_fidelity_gap": gap,
            "dead_zero_rejects": self._dead_zero_count,
            "mus_preserved": self._mus_count,
            "alignment_faking_probability": alignment_faking_prob,
            "alignment_faking_detected": alignment_faking_detected,
            "alignment_warnings": self._alignment_warnings[-5:],  # Last 5
            "decision_distribution": dict(decision_counts),
            "psi_inconsistency": self.detect_psi_inconsistency(),
            "recent_fidelity_trend": _linear_trend(
                [e.fidelity for e in self._audit_trail[-self.TREND_WINDOW:]]
            ) if len(self._audit_trail) >= self.TREND_WINDOW else 0.0,
        }

    def get_audit_trail(self, max_entries: int = 50) -> list[dict]:
        """Get recent audit trail entries as dicts.

        Args:
            max_entries: Maximum entries to return.

        Returns:
            List of audit entry dicts.
        """
        recent = self._audit_trail[-max_entries:]
        return [
            {
                "step": e.step,
                "node_id": e.node_id,
                "fidelity": e.fidelity,
                "decision_basis": e.decision_basis,
                "confidence": e.confidence,
                "psi_hash": e.psi_hash,
                "timestamp": e.timestamp,
            }
            for e in recent
        ]

    def reset(self) -> None:
        """Reset the audit trail and all statistics."""
        self._audit_trail.clear()
        self._total_steps = 0
        self._alignment_warnings.clear()
        self._mus_preserved.clear()
        self._fidelity_sum = 0.0
        self._confidence_sum = 0.0
        self._dead_zero_count = 0
        self._mus_count = 0


# ============================================================================
# Psi Audit integration into CognitiveRecursiveDynamics
# ============================================================================

class AuditedCognitiveRecursiveDynamics(CognitiveRecursiveDynamics):
    """CognitiveRecursiveDynamics with integrated ψ audit system.

    Extends the 5-layer cognitive reasoning with ψ-anchor audit at
    each layer transition, enabling alignment monitoring and
    confidence calibration based on actual fidelity scores.

    Usage:
        acrd = AuditedCognitiveRecursiveDynamics()
        acrd.attach_auditor()  # Or pass an existing PsiAuditor
        result = acrd.think(perception_data, context)
        # acrd.auditor.get_audit_report() shows alignment status
    """

    def __init__(self) -> None:
        """Initialize with ψ audit support."""
        super().__init__()
        self.auditor: PsiAuditor | None = None
        self._step_counter: int = 0

    def attach_auditor(self, auditor: PsiAuditor | None = None) -> None:
        """Attach a ψ auditor for decision traceability.

        Args:
            auditor: Existing PsiAuditor, or None to create a new one.
        """
        self.auditor = auditor or PsiAuditor()

    def think(
        self,
        perception: dict,
        context: dict,
    ) -> CognitiveResult:
        """Execute 5-layer reasoning with ψ audit at each layer.

        Each layer transition is recorded as a ψ audit entry,
        enabling post-hoc analysis of the reasoning quality.

        Args:
            perception: Perception data.
            context: Contextual information.

        Returns:
            CognitiveResult with audit metadata.
        """
        self._step_counter += 1
        step = self._step_counter

        # Run the standard reasoning pipeline
        result = super().think(perception, context)

        # Attach audit metadata if auditor is active
        if self.auditor is not None and result.action is not None:
            # Compute a synthetic fidelity from confidence and biases
            # (real fidelity would come from GaussEx verifier)
            synthetic_fidelity = result.confidence
            biases = result.meta_assessment.get("biases_detected", [])
            if biases:
                synthetic_fidelity *= max(0.3, 1.0 - 0.15 * len(biases))

            # Determine decision basis
            strategy = result.layer_outputs.get("L2_Strategic", {}).get("strategy", "unknown")
            decision_basis = f"cognitive_{strategy}"

            # Log the decision
            self.auditor.log_step(
                step=step,
                node_id=f"cog_{step}_{result.action}",
                fidelity=synthetic_fidelity,
                decision_basis=decision_basis,
                confidence=result.confidence,
                metadata={
                    "strategy": strategy,
                    "biases": biases,
                    "understanding": result.meta_assessment.get("understanding_quality", "unknown"),
                },
            )

        return result

    def get_audit_report(self) -> dict[str, Any] | None:
        """Get ψ audit report from attached auditor.

        Returns:
            Audit report dict, or None if no auditor attached.
        """
        if self.auditor is None:
            return None
        return self.auditor.get_audit_report()


# ============================================================================
# P1-8: Conditional ΔT Discovery — Auto-discover conditional transformations
# ============================================================================

@dataclass
class ConditionalRule:
    """A conditional transformation rule discovered from experience.

    Represents: IF condition_features THEN apply transformation T
    (ELSE apply transformation T' if specified).

    Attributes:
        condition: Feature conditions that must hold (dict of feature→value).
        transformation: The transformation to apply when condition holds.
        alt_transformation: Alternative transformation when condition fails.
        confidence: Rule confidence [0, 1] based on observed frequency.
        support: Number of trajectory samples supporting this rule.
        game_id: Game identifier this rule was discovered from.
    """
    condition: dict[str, Any] = field(default_factory=dict)
    transformation: str = ""
    alt_transformation: str = ""
    confidence: float = 0.0
    support: int = 0
    game_id: str = ""


class ConditionalDeltaTDiscovery:
    """P1-8: Automatically discover conditional ΔT patterns from trajectories.

    Analyzes successful and failed trajectories to discover conditions under
    which specific transformations are effective. This enables the agent to
    learn rules like:
        "IF grid has symmetry → use symmetry_completion"
        "IF grid has >3 colors → use color_map_transform"
        "IF grid is sparse → use flood_fill"

    Discovery pipeline:
    1. **Feature extraction**: Compute grid features (symmetry, color count,
       sparsity, connectivity) for each state in trajectory.
    2. **Outcome labeling**: Label each (state, action) pair as success/failure
       based on trajectory outcome.
    3. **Conditional clustering**: Group states by feature similarity.
    4. **Rule extraction**: For each cluster, determine the transformation
       with highest success rate → form a conditional rule.
    5. **Rule refinement**: Merge similar rules, prune low-confidence ones.

    Usage:
        discovery = ConditionalDeltaTDiscovery()
        rules = discovery.discover([trajectory1, trajectory2, ...])
        best_transform = discovery.select_transform(current_features, rules)
    """

    # Minimum samples for a rule to be considered reliable
    MIN_SAMPLES: int = 3
    # Minimum confidence for a rule to be kept
    MIN_CONFIDENCE: float = 0.6
    # Maximum number of rules to keep
    MAX_RULES: int = 50

    def __init__(self) -> None:
        """Initialize the conditional ΔT discovery system."""
        self._rules: list[ConditionalRule] = []
        self._feature_history: list[dict[str, Any]] = []
        self._outcome_history: list[bool] = []

    def discover(
        self,
        trajectories: list[Trajectory],
    ) -> list[ConditionalRule]:
        """Discover conditional transformation rules from trajectories.

        Args:
            trajectories: List of episode trajectories with states, actions,
                and outcomes.

        Returns:
            List of discovered ConditionalRule instances sorted by confidence.
        """
        # 1. Extract features and outcomes from all trajectories
        samples: list[tuple[dict[str, Any], str, bool]] = []

        for traj in trajectories:
            for i, (state, action) in enumerate(
                zip(traj.states, traj.actions)
            ):
                features = self._extract_features(state)
                # Determine if this action led to success in the trajectory
                # (simplified: later states closer to goal = success)
                success = self._label_outcome(traj, i)
                action_name = f"action_{action}"
                samples.append((features, action_name, success))

        if len(samples) < self.MIN_SAMPLES:
            return []

        # 2. Group samples by action type
        by_action: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
        for features, action_name, success in samples:
            by_action[action_name].append((features, success))

        # 3. For each action, find discriminating features
        new_rules: list[ConditionalRule] = []
        for action_name, action_samples in by_action.items():
            if len(action_samples) < self.MIN_SAMPLES:
                continue

            # Find features that discriminate success from failure
            discriminating = self._find_discriminating_features(action_samples)

            if discriminating:
                # Create conditional rule
                success_count = sum(1 for _, s in action_samples if s)
                total_count = len(action_samples)
                confidence = success_count / total_count

                if confidence >= self.MIN_CONFIDENCE:
                    rule = ConditionalRule(
                        condition=discriminating,
                        transformation=action_name,
                        confidence=confidence,
                        support=total_count,
                        game_id="",
                    )
                    new_rules.append(rule)

        # 4. Merge similar rules and prune low-confidence ones
        new_rules = self._merge_rules(new_rules)
        new_rules = [r for r in new_rules if r.confidence >= self.MIN_CONFIDENCE]
        new_rules = sorted(new_rules, key=lambda r: r.confidence, reverse=True)
        new_rules = new_rules[:self.MAX_RULES]

        # Update internal rule set
        self._rules.extend(new_rules)
        self._rules = sorted(self._rules, key=lambda r: r.confidence, reverse=True)
        self._rules = self._rules[:self.MAX_RULES]

        return new_rules

    def _extract_features(self, state: Any) -> dict[str, Any]:
        """Extract grid features from a game state.

        Computes structural features that may discriminate when different
        transformations are effective:
        - grid_size: Total cell count
        - unique_colors: Number of distinct colors
        - sparsity: Fraction of non-zero cells
        - has_symmetry: Whether grid has horizontal/vertical symmetry
        - max_object_size: Size of largest connected component

        Args:
            state: Game state (grid array or dict with 'grid' key).

        Returns:
            Feature dictionary.
        """
        features: dict[str, Any] = {}

        # Extract grid from state
        grid = None
        if isinstance(state, np.ndarray):
            grid = state
        elif isinstance(state, dict) and "grid" in state:
            grid = np.array(state["grid"])
        elif hasattr(state, "__array__"):
            grid = np.array(state)

        if grid is None or grid.size == 0:
            return features

        # Basic features
        features["grid_size"] = int(grid.size)
        features["unique_colors"] = int(len(np.unique(grid)))
        features["sparsity"] = float(np.count_nonzero(grid) / max(grid.size, 1))

        # Symmetry detection
        if grid.ndim == 2:
            h, w = grid.shape
            # Horizontal symmetry (flip left-right)
            h_sym = np.array_equal(grid, np.fliplr(grid))
            # Vertical symmetry (flip up-down)
            v_sym = np.array_equal(grid, np.flipud(grid))
            features["has_symmetry"] = bool(h_sym or v_sym)

            # Connected components (simplified: count distinct non-zero regions)
            try:
                from scipy import ndimage
                labeled, num_components = ndimage.label(grid > 0)
                features["num_objects"] = int(num_components)
            except ImportError:
                # Fallback: count non-zero rows as rough object count
                features["num_objects"] = int(np.count_nonzero(grid.any(axis=1)))
        else:
            features["has_symmetry"] = False
            features["num_objects"] = 0

        return features

    def _label_outcome(self, trajectory: Trajectory, step_idx: int) -> bool:
        """Label whether an action at a given step led to a positive outcome.

        Simplified heuristic: if the trajectory eventually succeeds, actions
        in the first half are labeled positive; if it fails, all are negative.

        Args:
            trajectory: The episode trajectory.
            step_idx: Index of the action in the trajectory.

        Returns:
            True if the action is considered successful.
        """
        if not hasattr(trajectory, "rewards") or not trajectory.rewards:
            # No reward data: assume success if trajectory is in later half
            return step_idx > len(trajectory.actions) // 2

        if step_idx < len(trajectory.rewards):
            return trajectory.rewards[step_idx] > 0

        return False

    def _find_discriminating_features(
        self,
        samples: list[tuple[dict[str, Any], bool]],
    ) -> dict[str, Any]:
        """Find features that discriminate success from failure.

        For each feature, compute the success rate when the feature is
        present vs. absent. Features with large success-rate gaps are
        discriminating.

        Args:
            samples: List of (features, success) pairs.

        Returns:
            Dictionary of discriminating feature conditions.
        """
        if not samples:
            return {}

        # Collect all feature keys
        all_keys: set[str] = set()
        for features, _ in samples:
            all_keys.update(features.keys())

        discriminating: dict[str, Any] = {}
        for key in all_keys:
            # Split samples by feature value
            present: list[bool] = []
            absent: list[bool] = []
            for features, success in samples:
                val = features.get(key)
                if val is not None and val != False and val != 0:
                    present.append(success)
                else:
                    absent.append(success)

            if len(present) < 2 or len(absent) < 2:
                continue

            present_rate = sum(present) / len(present)
            absent_rate = sum(absent) / len(absent)

            # Discrimination gap > 0.2 is significant
            gap = abs(present_rate - absent_rate)
            if gap > 0.2:
                # Use the value associated with higher success rate
                if present_rate > absent_rate:
                    # Feature presence correlates with success
                    discriminating[key] = True
                else:
                    # Feature absence correlates with success
                    discriminating[key] = False

        return discriminating

    def _merge_rules(
        self, rules: list[ConditionalRule]
    ) -> list[ConditionalRule]:
        """Merge rules with identical conditions and transformations.

        Args:
            rules: List of rules to merge.

        Returns:
            Merged list with combined support and averaged confidence.
        """
        if len(rules) <= 1:
            return rules

        merged: dict[str, ConditionalRule] = {}
        for rule in rules:
            key = f"{json.dumps(rule.condition, sort_keys=True)}|{rule.transformation}"
            if key in merged:
                existing = merged[key]
                # Combine support and recompute confidence
                total_support = existing.support + rule.support
                weighted_conf = (
                    existing.confidence * existing.support
                    + rule.confidence * rule.support
                ) / max(total_support, 1)
                existing.support = total_support
                existing.confidence = weighted_conf
            else:
                merged[key] = rule

        return list(merged.values())

    def select_transform(
        self,
        current_features: dict[str, Any],
        rules: list[ConditionalRule] | None = None,
    ) -> Optional[str]:
        """Select the best transformation given current features.

        Matches current features against rule conditions and returns
        the transformation from the highest-confidence matching rule.

        Args:
            current_features: Features of the current game state.
            rules: Rules to match against (defaults to internal rules).

        Returns:
            Transformation name, or None if no rule matches.
        """
        if rules is None:
            rules = self._rules

        best_match: Optional[ConditionalRule] = None
        best_confidence = 0.0

        for rule in rules:
            # Check if all conditions are satisfied
            match = True
            for cond_key, cond_val in rule.condition.items():
                actual_val = current_features.get(cond_key)
                if actual_val != cond_val:
                    # Handle boolean comparisons
                    if cond_val is True and not actual_val:
                        match = False
                        break
                    elif cond_val is False and actual_val:
                        match = False
                        break
                    elif actual_val != cond_val:
                        match = False
                        break

            if match and rule.confidence > best_confidence:
                best_match = rule
                best_confidence = rule.confidence

        if best_match is not None:
            return best_match.transformation
        return None

    def get_rules(self) -> list[ConditionalRule]:
        """Get all discovered conditional rules.

        Returns:
            List of ConditionalRule instances.
        """
        return self._rules.copy()

    def clear(self) -> None:
        """Clear all discovered rules and history."""
        self._rules.clear()
        self._feature_history.clear()
        self._outcome_history.clear()


# ============================================================================
# Helper Functions
# ============================================================================

def _linear_trend(values: list[float]) -> float:
    """Compute linear trend slope using simple least squares.

    Args:
        values: List of numeric values.

    Returns:
        Slope of linear fit (positive = increasing, negative = decreasing).
    """
    if len(values) < 2:
        return 0.0
    n = len(values)
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)

    # Simple linear regression: slope = cov(x,y) / var(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)
