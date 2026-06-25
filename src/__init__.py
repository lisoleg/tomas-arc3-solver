"""TOMAS ARC-AGI-3 Solver — Taiyi Mutual-Play framework for ARC-AGI-3.

Modules:
    agent: Interactive agents (PlannerAgent, Oracle adapters, self-learning, TOMAS learner)
    encoder: NAR-Conv octonion encoders (OctonionConv2d, NARGridEncoder)
    core: Core solver logic
    solver: κ-Snap search and beam search
    perception: Grid perception
    eval: Evaluation and benchmarking
    utils: Utility functions
    web: Web dashboard
    verify: GaussEx verification
"""
from __future__ import annotations

__version__ = "3.2.0-dev"
__author__ = "TOMAS Team"
