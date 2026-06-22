#!/usr/bin/env python
"""TOMAS benchmark with proper config handling.

Fixes:
- Pass full config to solver (not just search section)
- Properly merge config sections for KappaSnapSearcher
- Enable/disable psi_gate, aegis, causal_prior correctly
"""

from __future__ import annotations

import json
import os
import sys
import time
import copy
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ConfigLoader


def merge_config_for_searcher(config: dict[str, Any]) -> dict[str, Any]:
    """Merge nested config sections into flat structure for KappaSnapSearcher.
    
    KappaSnapSearcher expects flat config:
    - max_depth, mdl_threshold, etc. at top level
    - cuda config at config['cuda']
    - pruning config at config['pruning']
    
    But our YAML has them under 'search:' section.
    This function merges them properly.
    """
    merged = copy.deepcopy(config)
    
    # Merge search section into top level
    search_cfg = config.get("search", {})
    for key, value in search_cfg.items():
        merged[key] = value
    
    # Keep psi_gate, aegis, causal_prior at top level (solver will read them)
    # They are already at top level in config, so no need to move them
    
    return merged


def load_tasks(data_dir: str = "data") -> list[dict[str, Any]]:
    """Load all task JSON files from data directory."""
    tasks = []
    data_path = PROJECT_ROOT / data_dir
    if not data_path.exists():
        print(f"[WARN] Data directory not found: {data_path}")
        return tasks

    for json_file in sorted(data_path.glob("*.json")):
        with open(json_file, "r") as f:
            task_data = json.load(f)
        task_data["_file"] = str(json_file.name)
        tasks.append(task_data)

    return tasks


def extract_demo_pairs(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract demo pairs from ARC task JSON."""
    demo_pairs = []
    for train_item in task.get("train", []):
        input_grids = [np.array(g, dtype=np.int8) for g in train_item["input"]]
        output_grids = [np.array(g, dtype=np.int8) for g in train_item["output"]]
        demo_pairs.append({"input": input_grids, "output": output_grids})
    return demo_pairs


def run_single_benchmark(
    config: dict[str, Any],
    demo_pairs: list[dict[str, Any]],
    task_name: str,
    config_name: str,
) -> dict[str, Any]:
    """Run a single benchmark with given config and demo pairs."""
    from src.solver.tomas_solver import TOMASSolver
    
    start_time = time.time()

    result: dict[str, Any] = {
        "task": task_name,
        "config": config_name,
        "status": "running",
        "error": None,
    }

    try:
        # Use TOMASSolver which properly handles config
        solver = TOMASSolver(config)
        
        # Run solve
        search_start = time.time()
        solution = solver.solve(demo_pairs)
        search_time = time.time() - search_start

        # Get stats from solver
        searcher = solver.searcher
        
        # Check correctness
        correct = False
        if solution and solution.get("program"):
            prog = solution["program"]
            try:
                all_correct = True
                for pair in demo_pairs:
                    for i, inp in enumerate(pair["input"]):
                        if i < len(pair["output"]):
                            predicted = prog.apply(inp)
                            expected = pair["output"][i]
                            if not np.array_equal(predicted, expected):
                                all_correct = False
                                break
                    if not all_correct:
                        break
                correct = all_correct
            except Exception:
                correct = False

        total_time = time.time() - start_time
        
        # Get candidate count from searcher
        candidates = getattr(searcher, '_last_candidates', [])
        top_confidence = solution.get("confidence", 0.0) if solution else 0.0
        top_mdl = solution.get("mdl", 0) if solution else 0

        result.update({
            "status": "completed",
            "search_time_sec": round(search_time, 4),
            "total_time_sec": round(total_time, 4),
            "total_candidates": len(candidates),
            "ranked_candidates": len(candidates),
            "top_confidence": round(top_confidence, 4),
            "top_mdl": top_mdl,
            "correct": correct,
            "cuda_backend": getattr(searcher, '_cuda_backend', 'none'),
        })

    except Exception as e:
        result.update({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()[:500],
            "total_time_sec": round(time.time() - start_time, 4),
        })

    return result


def run_psi_gate_benchmark(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run psi-Gate enabled vs disabled benchmark."""
    results = []

    for task in tasks:
        demo_pairs = extract_demo_pairs(task)
        if not demo_pairs:
            continue

        task_name = task.get("_file", "unknown")

        # Config A: psi_gate disabled
        config_disabled = copy.deepcopy(config)
        if "psi_gate" in config_disabled:
            config_disabled["psi_gate"]["enabled"] = False

        result_disabled = run_single_benchmark(
            config_disabled, demo_pairs, task_name, "psi_gate_disabled"
        )
        results.append(result_disabled)

        # Config B: psi_gate enabled
        config_enabled = copy.deepcopy(config)
        if "psi_gate" in config_enabled:
            config_enabled["psi_gate"]["enabled"] = True

        result_enabled = run_single_benchmark(
            config_enabled, demo_pairs, task_name, "psi_gate_enabled"
        )
        results.append(result_enabled)

    return results


def run_aegis_benchmark(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run AEGIS evolution vs normal search benchmark."""
    results = []

    for task in tasks:
        demo_pairs = extract_demo_pairs(task)
        if not demo_pairs:
            continue

        task_name = task.get("_file", "unknown")

        # Config A: AEGIS disabled (normal search)
        config_normal = copy.deepcopy(config)
        if "aegis" in config_normal:
            config_normal["aegis"]["enabled"] = False

        result_normal = run_single_benchmark(
            config_normal, demo_pairs, task_name, "aegis_disabled"
        )
        results.append(result_normal)

        # Config B: AEGIS enabled (evolution search)
        config_aegis = copy.deepcopy(config)
        if "aegis" in config_aegis:
            config_aegis["aegis"]["enabled"] = True

        result_aegis = run_single_benchmark(
            config_aegis, demo_pairs, task_name, "aegis_enabled"
        )
        results.append(result_aegis)

    return results


def run_causal_prior_benchmark(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run causal DSL prior enabled vs disabled benchmark."""
    results = []

    for task in tasks:
        demo_pairs = extract_demo_pairs(task)
        if not demo_pairs:
            continue

        task_name = task.get("_file", "unknown")

        # Config A: causal_prior disabled
        config_disabled = copy.deepcopy(config)
        if "causal_prior" in config_disabled:
            config_disabled["causal_prior"]["enabled"] = False

        result_disabled = run_single_benchmark(
            config_disabled, demo_pairs, task_name, "causal_prior_disabled"
        )
        results.append(result_disabled)

        # Config B: causal_prior enabled
        config_enabled = copy.deepcopy(config)
        if "causal_prior" in config_enabled:
            config_enabled["causal_prior"]["enabled"] = True

        result_enabled = run_single_benchmark(
            config_enabled, demo_pairs, task_name, "causal_prior_enabled"
        )
        results.append(result_enabled)

    return results


def generate_report(
    psi_gate_results: list[dict],
    aegis_results: list[dict],
    causal_results: list[dict],
) -> str:
    """Generate a comprehensive performance report as Markdown."""
    lines = []
    lines.append("# TOMAS ARC-AGI-3 Solver Performance Benchmark Report")
    lines.append("")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Tasks**: {len(set(r.get('task', '') for r in psi_gate_results))}")
    lines.append(f"**Config**: search.max_depth={psi_gate_results[0].get('config', {}).get('max_depth', 'N/A') if psi_gate_results else 'N/A'}")
    lines.append("")

    # Helper to aggregate results
    def aggregate(results: list[dict], config_name: str) -> dict:
        matching = [r for r in results if r.get("config") == config_name]
        completed = [r for r in matching if r.get("status") == "completed"]
        if not completed:
            return {"count": 0}

        correct = sum(1 for r in completed if r.get("correct"))
        avg_time = sum(r.get("total_time_sec", 0) for r in completed) / len(completed)
        avg_candidates = sum(r.get("total_candidates", 0) for r in completed) / len(completed)
        avg_confidence = sum(r.get("top_confidence", 0) for r in completed) / len(completed)
        avg_mdl = sum(r.get("top_mdl", 0) for r in completed) / len(completed)

        return {
            "count": len(completed),
            "correct": correct,
            "accuracy": round(correct / len(completed) * 100, 1),
            "avg_time": round(avg_time, 4),
            "avg_candidates": round(avg_candidates, 1),
            "avg_confidence": round(avg_confidence, 4),
            "avg_mdl": round(avg_mdl, 1),
        }

    # 1. psi-Gate comparison
    lines.append("## 1. psi-Gate Semantic Gating Comparison")
    lines.append("")
    lines.append("| Metric | psi-Gate Disabled | psi-Gate Enabled | Delta |")
    lines.append("|--------|-------------------|------------------|-------|")

    agg_disabled = aggregate(psi_gate_results, "psi_gate_disabled")
    agg_enabled = aggregate(psi_gate_results, "psi_gate_enabled")

    if agg_disabled["count"] > 0 and agg_enabled["count"] > 0:
        metrics = [
            ("Tasks Completed", agg_disabled["count"], agg_enabled["count"]),
            ("Correct", agg_disabled["correct"], agg_enabled["correct"]),
            ("Accuracy (%)", agg_disabled["accuracy"], agg_enabled["accuracy"]),
            ("Avg Time (s)", agg_disabled["avg_time"], agg_enabled["avg_time"]),
            ("Avg Candidates", agg_disabled["avg_candidates"], agg_enabled["avg_candidates"]),
            ("Avg Confidence", agg_disabled["avg_confidence"], agg_enabled["avg_confidence"]),
            ("Avg MDL", agg_disabled["avg_mdl"], agg_enabled["avg_mdl"]),
        ]
        for name, d, e in metrics:
            delta = e - d if isinstance(d, (int, float)) and isinstance(e, (int, float)) else "N/A"
            if isinstance(delta, float):
                delta_str = f"+{delta}" if delta >= 0 else str(delta)
            else:
                delta_str = str(delta)
            lines.append(f"| {name} | {d} | {e} | {delta_str} |")
    else:
        lines.append("| (insufficient data) | | | |")

    lines.append("")

    # 2. AEGIS comparison
    lines.append("## 2. AEGIS Evolution Engine Comparison")
    lines.append("")
    lines.append("| Metric | Normal Search | AEGIS Evolution | Delta |")
    lines.append("|--------|--------------|-----------------|-------|")

    agg_normal = aggregate(aegis_results, "aegis_disabled")
    agg_aegis = aggregate(aegis_results, "aegis_enabled")

    if agg_normal["count"] > 0 and agg_aegis["count"] > 0:
        metrics = [
            ("Tasks Completed", agg_normal["count"], agg_aegis["count"]),
            ("Correct", agg_normal["correct"], agg_aegis["correct"]),
            ("Accuracy (%)", agg_normal["accuracy"], agg_aegis["accuracy"]),
            ("Avg Time (s)", agg_normal["avg_time"], agg_aegis["avg_time"]),
            ("Avg Candidates", agg_normal["avg_candidates"], agg_aegis["avg_candidates"]),
            ("Avg Confidence", agg_normal["avg_confidence"], agg_aegis["avg_confidence"]),
            ("Avg MDL", agg_normal["avg_mdl"], agg_aegis["avg_mdl"]),
        ]
        for name, d, e in metrics:
            delta = e - d if isinstance(d, (int, float)) and isinstance(e, (int, float)) else "N/A"
            if isinstance(delta, float):
                delta_str = f"+{delta}" if delta >= 0 else str(delta)
            else:
                delta_str = str(delta)
            lines.append(f"| {name} | {d} | {e} | {delta_str} |")
    else:
        lines.append("| (insufficient data) | | | |")

    lines.append("")

    # 3. Causal Prior comparison
    lines.append("## 3. Causal DSL Prior Comparison")
    lines.append("")
    lines.append("| Metric | Causal Prior Disabled | Causal Prior Enabled | Delta |")
    lines.append("|--------|----------------------|----------------------|-------|")

    agg_cp_off = aggregate(causal_results, "causal_prior_disabled")
    agg_cp_on = aggregate(causal_results, "causal_prior_enabled")

    if agg_cp_off["count"] > 0 and agg_cp_on["count"] > 0:
        metrics = [
            ("Tasks Completed", agg_cp_off["count"], agg_cp_on["count"]),
            ("Correct", agg_cp_off["correct"], agg_cp_on["correct"]),
            ("Accuracy (%)", agg_cp_off["accuracy"], agg_cp_on["accuracy"]),
            ("Avg Time (s)", agg_cp_off["avg_time"], agg_cp_on["avg_time"]),
            ("Avg Candidates", agg_cp_off["avg_candidates"], agg_cp_on["avg_candidates"]),
            ("Avg Confidence", agg_cp_off["avg_confidence"], agg_cp_on["avg_confidence"]),
            ("Avg MDL", agg_cp_off["avg_mdl"], agg_cp_on["avg_mdl"]),
        ]
        for name, d, e in metrics:
            delta = e - d if isinstance(d, (int, float)) and isinstance(e, (int, float)) else "N/A"
            if isinstance(delta, float):
                delta_str = f"+{delta}" if delta >= 0 else str(delta)
            else:
                delta_str = str(delta)
            lines.append(f"| {name} | {d} | {e} | {delta_str} |")
    else:
        lines.append("| (insufficient data) | | | |")

    lines.append("")

    # 4. Per-task details
    lines.append("## 4. Per-Task Details")
    lines.append("")
    lines.append("### psi-Gate Benchmark")
    lines.append("")
    lines.append("| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |")
    lines.append("|------|--------|--------|----------|------------|------------|---------|")
    for r in psi_gate_results:
        lines.append(
            f"| {r.get('task', '')} | {r.get('config', '')} | {r.get('status', '')} | "
            f"{r.get('total_time_sec', '-')} | {r.get('total_candidates', '-')} | "
            f"{r.get('top_confidence', '-')} | {r.get('correct', '-')} |"
        )

    lines.append("")
    lines.append("### AEGIS Benchmark")
    lines.append("")
    lines.append("| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |")
    lines.append("|------|--------|--------|----------|------------|------------|---------|")
    for r in aegis_results:
        lines.append(
            f"| {r.get('task', '')} | {r.get('config', '')} | {r.get('status', '')} | "
            f"{r.get('total_time_sec', '-')} | {r.get('total_candidates', '-')} | "
            f"{r.get('top_confidence', '-')} | {r.get('correct', '-')} |"
        )

    lines.append("")

    # 5. Conclusions
    lines.append("## 5. Conclusions")
    lines.append("")
    if agg_enabled.get("count", 0) > 0 and agg_disabled.get("count", 0) > 0:
        if agg_enabled["accuracy"] > agg_disabled["accuracy"]:
            lines.append("- **psi-Gate** improves accuracy by "
                        f"{agg_enabled['accuracy'] - agg_disabled['accuracy']:.1f}%")
        if agg_enabled["avg_confidence"] > agg_disabled["avg_confidence"]:
            lines.append("- **psi-Gate** improves confidence by "
                        f"{agg_enabled['avg_confidence'] - agg_disabled['avg_confidence']:.4f}")

    if agg_aegis.get("count", 0) > 0 and agg_normal.get("count", 0) > 0:
        if agg_aegis["accuracy"] > agg_normal["accuracy"]:
            lines.append("- **AEGIS** improves accuracy by "
                        f"{agg_aegis['accuracy'] - agg_normal['accuracy']:.1f}%")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by TOMAS benchmark suite*")

    return "\n".join(lines)


def main():
    """Main benchmark entry point."""
    print("=" * 60)
    print("TOMAS ARC-AGI-3 Solver Performance Benchmark (Fixed)")
    print("=" * 60)

    # Load config
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = ConfigLoader.load(str(config_path))
    print(f"Config loaded from: {config_path}")
    print(f"  search.max_depth: {config.get('search', {}).get('max_depth', 'N/A')}")
    print(f"  psi_gate.enabled: {config.get('psi_gate', {}).get('enabled', 'N/A')}")
    print(f"  aegis.enabled: {config.get('aegis', {}).get('enabled', 'N/A')}")

    # Load tasks
    tasks = load_tasks("data")
    print(f"Loaded {len(tasks)} tasks from data/")

    if not tasks:
        print("[ERROR] No tasks found. Aborting.")
        return

    # Run benchmarks
    print("\n--- Running psi-Gate benchmark ---")
    psi_gate_results = run_psi_gate_benchmark(tasks, config)
    print(f"  Completed: {len(psi_gate_results)} runs")

    print("\n--- Running AEGIS benchmark ---")
    aegis_results = run_aegis_benchmark(tasks, config)
    print(f"  Completed: {len(aegis_results)} runs")

    print("\n--- Running Causal Prior benchmark ---")
    causal_results = run_causal_prior_benchmark(tasks, config)
    print(f"  Completed: {len(causal_results)} runs")

    # Generate report
    print("\n--- Generating report ---")
    report = generate_report(psi_gate_results, aegis_results, causal_results)

    # Save report
    report_path = PROJECT_ROOT / "benchmarks" / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Save raw results as JSON
    raw_path = PROJECT_ROOT / "benchmarks" / "benchmark_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "psi_gate": psi_gate_results,
            "aegis": aegis_results,
            "causal_prior": causal_results,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2, ensure_ascii=False)
    print(f"Raw data saved to: {raw_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)

    def print_summary(name: str, results: list[dict]):
        completed = [r for r in results if r.get("status") == "completed"]
        correct = sum(1 for r in completed if r.get("correct"))
        print(f"  {name}: {len(completed)}/{len(results)} completed, "
              f"{correct}/{len(completed)} correct")

    print_summary("psi-Gate", psi_gate_results)
    print_summary("AEGIS", aegis_results)
    print_summary("Causal Prior", causal_results)

    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
