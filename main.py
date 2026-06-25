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
        "--oracle",
        action="store_true",
        help="Enable Oracle mode (uses HybridTaiyiAgent with L2→L3→L4→L5 pipeline)",
    )
    parser.add_argument(
        "--sleep-step",
        action="store_true",
        help="Run Sleep-Step warmup (Library Learning) before solving",
    )
    parser.add_argument(
        "--modifier-hints",
        type=str,
        nargs="*",
        default=None,
        help="Modifier hints for Oracle mode (e.g., --modifier-hints gravity flip)",
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

    if args.oracle:
        from src.agent.hybrid_agent import HybridTaiyiAgent
        logger.info("Oracle mode enabled — using HybridTaiyiAgent")

    solver = TOMASSolver(config)

    # v3.1: Sleep-Step warmup (Library Learning)
    if args.sleep_step:
        logger.info("Running Sleep-Step warmup (Library Learning)...")
        try:
            from src.solver.library_learning import LibraryLearning
            ll = LibraryLearning(config.get("library", {}))
            ll.sleep_step_warmup(warmup_tasks=25)
            logger.info("Sleep-Step warmup complete")
        except Exception as e:
            logger.warning(f"Sleep-Step warmup failed: {e}")

    # Load input
    if args.input is None:
        logger.error("No input file specified. Use --input <path>")
        return 1

    input_path = Path(args.input)

    # v3.1: Per-task timeout protection (Unix only — signal.alarm not on Windows)
    import signal
    _signal_alarm_available = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")

    def _timeout_handler(signum, frame):
        raise TimeoutError("Per-task timeout exceeded")

    timeout_sec = config.get("defensive", {}).get("per_task_timeout", 70)
    old_handler = signal.getsignal(signal.SIGALRM) if _signal_alarm_available else None

    if args.oracle:
        # v3.1: Oracle Mode — use HybridTaiyiAgent with L2→L3→L4→L5 pipeline
        import arc_agi
        agent = HybridTaiyiAgent(use_oracle=True)
        results = {}
        task_list = sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]
        for tf in task_list:
            with open(tf, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            env = arc_agi.make(task_data.get("game_id", ""))
            episode_result = agent.run_episode(
                env,
                game_id=task_data.get("game_id"),
                modifier_hints=args.modifier_hints,
            )
            results[tf.stem] = episode_result
            logger.info(f"Oracle episode: {episode_result}")
    else:
        # Grid-Only Mode: existing solver path
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
                # v3.1: Per-task timeout
                if _signal_alarm_available:
                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(int(timeout_sec))
                try:
                    result = solver.solve(task_data, mode=mode)
                except TimeoutError:
                    logger.warning(f"Task {tf.name} timed out after {timeout_sec}s")
                    result = None
                finally:
                    if _signal_alarm_available:
                        signal.alarm(0)
                results[tf.stem] = result
                solver._post_solve_learning(result)
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            mode = args.mode
            if mode == "auto":
                mode = solver.auto_select_mode(args.time_budget, len(task_data.get("train", [])))
                logger.info(f"Auto-selected mode: {mode}")
            # v3.1: Per-task timeout
            if _signal_alarm_available:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(int(timeout_sec))
            try:
                result = solver.solve(task_data, mode=mode)
            except TimeoutError:
                logger.warning(f"Task {input_path.name} timed out after {timeout_sec}s")
                result = None
            finally:
                if _signal_alarm_available:
                    signal.alarm(0)
            results = {input_path.stem: result}
            solver._post_solve_learning(result)

    # Restore old signal handler
    if _signal_alarm_available:
        signal.signal(signal.SIGALRM, old_handler)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Output written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
