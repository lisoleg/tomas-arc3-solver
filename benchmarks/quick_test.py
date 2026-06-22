#!/usr/bin/env python
"""Simplified benchmark - test if config changes actually work."""

from __future__ import annotations
import json
import sys
import time
import copy
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.solver.tomas_solver import TOMASSolver

def load_tasks(data_dir="data"):
    tasks = []
    for json_file in sorted(Path(data_dir).glob("*.json")):
        with open(json_file) as f:
            task = json.load(f)
        task["_file"] = json_file.name
        tasks.append(task)
    return tasks

def run_benchmark():
    """Run benchmark comparing psi_gate on/off."""
    from src.utils.config import ConfigLoader
    
    config = ConfigLoader.load(str(PROJECT_ROOT / "config" / "default.yaml"))
    tasks = load_tasks()
    
    print(f"Loaded {len(tasks)} tasks")
    print(f"Config: psi_gate.enabled={config.get('psi_gate', {}).get('enabled')}")
    
    results = {"psi_off": [], "psi_on": []}
    
    for task in tasks[:5]:  # Test with 5 tasks first
        task_name = task["_file"]
        
        # Test with psi_gate OFF
        cfg_off = copy.deepcopy(config)
        cfg_off["psi_gate"]["enabled"] = False
        solver_off = TOMASSolver(cfg_off)
        t0 = time.time()
        sol_off = solver_off.solve(task)
        t_off = time.time() - t0
        
        # Test with psi_gate ON
        cfg_on = copy.deepcopy(config)
        cfg_on["psi_gate"]["enabled"] = True
        solver_on = TOMASSolver(cfg_on)
        t0 = time.time()
        sol_on = solver_on.solve(task)
        t_on = time.time() - t0
        
        # Check correctness
        def check_correct(sol, task):
            if not sol or not sol.get("program"):
                return False
            prog = sol["program"]
            try:
                for train in task.get("train", []):
                    inp = np.array(train["input"][0], dtype=np.int8)
                    out = np.array(train["output"][0], dtype=np.int8)
                    if not np.array_equal(prog.apply(inp), out):
                        return False
                return True
            except:
                return False
        
        correct_off = check_correct(sol_off, task)
        correct_on = check_correct(sol_on, task)
        
        results["psi_off"].append({"task": task_name, "correct": correct_off, "time": t_off})
        results["psi_on"].append({"task": task_name, "correct": correct_on, "time": t_on})
        
        print(f"{task_name}: OFF={'✓' if correct_off else '✗'} ({t_off:.2f}s) | ON={'✓' if correct_on else '✗'} ({t_on:.2f}s)")
    
    # Summary
    off_correct = sum(1 for r in results["psi_off"] if r["correct"])
    on_correct = sum(1 for r in results["psi_on"] if r["correct"])
    print(f"\nSummary: OFF={off_correct}/5 | ON={on_correct}/5")

if __name__ == "__main__":
    run_benchmark()
