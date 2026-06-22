"""CLI entry point for TOMAS ARC-AGI-3 Solver."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="TOMAS ARC-AGI-3 Solver — Taiyi Mutual-Play framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["video", "bayesian", "fusion", "auto"],
        default="auto",
        help="Inference mode: video (fast symbolic), bayesian (posterior), "
        "fusion (multi-modal), auto (time-budget adaptive)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSON file (ARC-AGI-3 task) or directory of tasks",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.json",
        help="Path to output submission file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=80.0,
        help="Time budget per task in seconds (for auto mode)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    # Load configuration
    config = ConfigLoader.load(args.config)

    # Setup logger
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    logger = setup_logger(log_level, config.get("logging", {}).get("log_dir", "logs"))
    logger.info("TOMAS ARC-AGI-3 Solver starting...")
    logger.info(f"Mode: {args.mode}, Input: {args.input}, Output: {args.output}")

    # Import solver here to avoid heavy imports on --help
    from src.solver.tomas_solver import TOMASSolver

    solver = TOMASSolver(config)

    # Load input
    if args.input is None:
        logger.error("No input file specified. Use --input <path>")
        return 1

    input_path = Path(args.input)
    if input_path.is_dir():
        # Batch mode: solve all tasks in directory
        task_files = sorted(input_path.glob("*.json"))
        logger.info(f"Found {len(task_files)} task files in {input_path}")
        results = {}
        for tf in task_files:
            logger.info(f"Solving {tf.name}...")
            with open(tf, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            mode = args.mode
            if mode == "auto":
                mode = solver.auto_select_mode(args.time_budget, len(task_data.get("train", [])))
                logger.info(f"Auto-selected mode: {mode}")
            result = solver.solve(task_data, mode=mode)
            results[tf.stem] = result
            solver._post_solve_learning(result)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        mode = args.mode
        if mode == "auto":
            mode = solver.auto_select_mode(args.time_budget, len(task_data.get("train", [])))
            logger.info(f"Auto-selected mode: {mode}")
        result = solver.solve(task_data, mode=mode)
        results = {input_path.stem: result}
        solver._post_solve_learning(result)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Output written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
