"""
TOMAS Visualization Data Service

Generates realistic mock data for dashboard visualizations:
- kappa-Snap search tree (D3.js hierarchy)
- GaussEx fiber verification results
- 8-strategy pruning statistics (Recharts)
- Task history with details

When the real solver is running, this service can be enhanced to
capture actual runtime data via callbacks from the solver pipeline.
"""

import random
import hashlib
from typing import Any
from datetime import datetime, timedelta


def generate_search_tree(task_id: str, depth: int = 3) -> dict[str, Any]:
    """Generate a realistic kappa-Snap search tree structure.

    Simulates the Two-Phase search tree:
    - Root: demo pairs input
    - Level 1-3: candidate programs at each depth
    - Phase A pass/fail labels
    - Phase B verify pass/fail labels

    Args:
        task_id: Task identifier.
        depth: Maximum search depth.

    Returns:
        D3.js hierarchy-compatible tree dict.
    """
    rng = random.Random(hash(task_id) & 0xFFFFFFFF)

    dsl_primitives = [
        "resize", "mirror_h", "mirror_v", "rotate90", "rotate180",
        "rotate270", "move", "copy", "gravity_down", "gravity_up",
        "fill", "mask", "extract_color", "count_objects", "tile",
    ]

    def make_node(name: str, level: int, parent_phase: str = "") -> dict:
        node = {
            "name": name,
            "level": level,
            "phase": parent_phase,
            "mdl": rng.randint(5, 50),
            "children": [],
        }
        if level < depth:
            num_children = rng.randint(2, 5) if level < 2 else rng.randint(1, 3)
            for i in range(num_children):
                prim = rng.choice(dsl_primitives)
                phase = "phase_a_pass" if rng.random() > 0.3 else "phase_a_fail"
                child = make_node(f"{prim}_{i}", level + 1, phase)
                if level + 1 == depth:
                    child["verified"] = rng.random() > 0.7
                    child["confidence"] = round(rng.uniform(0.3, 0.99), 3)
                node["children"].append(child)
        return node

    root = make_node("root (demo_pairs)", 0, "init")
    return {
        "task_id": task_id,
        "tree": root,
        "total_candidates": _count_nodes(root) - 1,
        "phase_a_passed": _count_phase(root, "phase_a_pass"),
        "phase_a_failed": _count_phase(root, "phase_a_fail"),
        "phase_b_verified": _count_verified(root),
        "max_depth": depth,
    }


def generate_fiber_verification(task_id: str) -> dict[str, Any]:
    """Generate GaussEx fiber verification results.

    Simulates fiber intersection verification results:
    - List of demo pairs with their fiber verification status
    - CRC32 hash matching results
    - Fiber intersection diagrams data

    Args:
        task_id: Task identifier.

    Returns:
        Fiber verification data dict.
    """
    rng = random.Random(hash(task_id) & 0xFFFFFFFF + 1)

    num_demos = rng.randint(2, 5)
    num_candidates = rng.randint(5, 15)

    demos = []
    for i in range(num_demos):
        demo = {
            "demo_id": f"demo_{i+1}",
            "input_shape": [rng.randint(3, 10), rng.randint(3, 10)],
            "output_shape": [rng.randint(3, 10), rng.randint(3, 10)],
            "fiber_count": rng.randint(2, 8),
            "intersection_size": rng.randint(1, 4),
            "crc32_input": f"{rng.randint(0, 0xFFFFFFFF):08x}",
            "crc32_output": f"{rng.randint(0, 0xFFFFFFFF):08x}",
            "crc32_match": rng.random() > 0.4,
        }
        demos.append(demo)

    candidates = []
    for i in range(num_candidates):
        verified_demos = []
        for j in range(num_demos):
            verified_demos.append({
                "demo_id": f"demo_{j+1}",
                "pass": rng.random() > 0.35,
                "fiber_overlap": round(rng.uniform(0, 1), 3),
                "hash_match": rng.random() > 0.3,
            })
        all_pass = all(d["pass"] for d in verified_demos)
        candidates.append({
            "candidate_id": f"prog_{i+1}",
            "mdl": rng.randint(5, 50),
            "verified_demos": verified_demos,
            "all_pass": all_pass,
            "confidence": round(rng.uniform(0.2, 0.99), 3) if all_pass else 0.0,
        })

    passed = [c for c in candidates if c["all_pass"]]

    return {
        "task_id": task_id,
        "demos": demos,
        "candidates": candidates,
        "total_candidates": len(candidates),
        "verified": len(passed),
        "failed": len(candidates) - len(passed),
        "verification_rate": round(len(passed) / max(len(candidates), 1) * 100, 1),
    }


def generate_pruning_stats(task_id: str) -> dict[str, Any]:
    """Generate 8-strategy pruning pipeline statistics.

    Returns data for Recharts bar chart showing pruning rates:
    1. grid_shape
    2. nonzero_count
    3. color_histogram
    4. betti0
    5. topo_hash
    6. symmetry_dedup
    7. incremental_mdl
    8. heuristic_order

    Args:
        task_id: Task identifier.

    Returns:
        Pruning statistics dict.
    """
    rng = random.Random(hash(task_id) & 0xFFFFFFFF + 2)

    total_initial = rng.randint(500, 2000)

    strategies = [
        {"name": "grid_shape", "label": "Grid Shape Filter", "color": "#1976d2"},
        {"name": "nonzero_count", "label": "Non-zero Count", "color": "#388e3c"},
        {"name": "color_histogram", "label": "Color Histogram", "color": "#f57c00"},
        {"name": "betti0", "label": "Betti0 Invariant", "color": "#d32f2f"},
        {"name": "topo_hash", "label": "Topo Hash (Phase A)", "color": "#7b1fa2"},
        {"name": "symmetry_dedup", "label": "Symmetry Dedup", "color": "#0097a7"},
        {"name": "incremental_mdl", "label": "Incremental MDL", "color": "#689f38"},
        {"name": "heuristic_order", "label": "Heuristic Order", "color": "#e64a19"},
    ]

    remaining = total_initial
    stages = []
    for s in strategies:
        pruned = int(remaining * rng.uniform(0.05, 0.25))
        stages.append({
            "name": s["name"],
            "label": s["label"],
            "color": s["color"],
            "candidates_before": remaining,
            "candidates_after": remaining - pruned,
            "pruned": pruned,
            "prune_rate": round(pruned / max(remaining, 1) * 100, 1),
            "cumulative_rate": round(
                (1 - (remaining - pruned) / total_initial) * 100, 1
            ),
        })
        remaining -= pruned

    return {
        "task_id": task_id,
        "total_initial": total_initial,
        "total_remaining": remaining,
        "total_pruned": total_initial - remaining,
        "overall_prune_rate": round(
            (1 - remaining / total_initial) * 100, 1
        ),
        "stages": stages,
    }


def generate_task_history(limit: int = 20) -> list[dict[str, Any]]:
    """Generate realistic task history entries.

    Args:
        limit: Maximum number of history entries.

    Returns:
        List of history entry dicts.
    """
    rng = random.Random(42)
    modes = ["video", "bayesian", "fusion", "auto"]
    statuses = ["completed", "completed", "completed", "failed", "timeout"]
    tasks = []

    base_time = datetime.now() - timedelta(days=7)

    for i in range(limit):
        mode = rng.choice(modes)
        status = rng.choice(statuses)
        duration = rng.uniform(2.0, 120.0)
        num_candidates = rng.randint(50, 2000)
        num_verified = int(num_candidates * rng.uniform(0.05, 0.3))
        confidence = round(rng.uniform(0.5, 0.99), 3) if status == "completed" else 0.0

        task = {
            "task_id": f"task_{i+1:04d}",
            "input_path": f"data/task_{rng.randint(1, 100):03d}.json",
            "mode": mode,
            "status": status,
            "duration_sec": round(duration, 2),
            "candidates_generated": num_candidates,
            "candidates_verified": num_verified,
            "prune_rate": round(rng.uniform(60, 95), 1),
            "confidence": confidence,
            "mdl_best": rng.randint(5, 45) if status == "completed" else None,
            "timestamp": (base_time + timedelta(hours=i * 3)).isoformat(),
            "psi_gate_enabled": rng.choice([True, False]),
            "aegis_enabled": rng.choice([True, False]),
        }
        tasks.append(task)

    return tasks


def generate_task_detail(task_id: str) -> dict[str, Any]:
    """Generate detailed task information including all visualization data.

    Args:
        task_id: Task identifier.

    Returns:
        Complete task detail dict.
    """
    return {
        "task_id": task_id,
        "search_tree": generate_search_tree(task_id),
        "fiber_verification": generate_fiber_verification(task_id),
        "pruning_stats": generate_pruning_stats(task_id),
    }


# --- Helper functions ---

def _count_nodes(node: dict) -> int:
    """Recursively count nodes in a tree."""
    count = 1
    for child in node.get("children", []):
        count += _count_nodes(child)
    return count


def _count_phase(node: dict, phase: str) -> int:
    """Count nodes with a specific phase label."""
    count = 1 if node.get("phase") == phase else 0
    for child in node.get("children", []):
        count += _count_phase(child, phase)
    return count


def _count_verified(node: dict) -> int:
    """Count verified leaf nodes."""
    if not node.get("children"):
        return 1 if node.get("verified") else 0
    count = 0
    for child in node.get("children", []):
        count += _count_verified(child)
    return count
