"""Structured logging with kappa-Snap audit chain recording (JSON Lines)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger as _logger


class KappaSnapAuditor:
    """Records kappa-Snap audit chain entries in JSON Lines format.

    Each audit entry contains timestamp, task_id, action, program MDL,
    and the full decision path for reproducibility.
    """

    def __init__(self, audit_log_path: str | Path = "logs/kappa_snap_audit.jsonl") -> None:
        """Initialize the auditor.

        Args:
            audit_log_path: Path to the JSON Lines audit log file.
        """
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        task_id: str,
        action: str,
        program_mdl: int = 0,
        decision_path: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record a single audit entry.

        Args:
            task_id: Identifier of the task being solved.
            action: Description of the decision/action taken.
            program_mdl: MDL cost of the program at this decision point.
            decision_path: Ordered list of decision steps taken.
            extra: Additional metadata to include.
        """
        entry: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "task_id": task_id,
            "action": action,
            "program_mdl": program_mdl,
            "decision_path": decision_path or [],
        }
        if extra:
            entry["extra"] = extra

        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Global auditor instance (lazily initialized)
_auditor: KappaSnapAuditor | None = None


def setup_logger(level: str = "INFO", log_dir: str = "logs") -> Any:
    """Configure and return a loguru logger instance.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.

    Returns:
        Configured loguru logger.
    """
    global _auditor
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    _logger.remove()
    _logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    _logger.add(
        str(log_dir_path / "tomas_{time}.log"),
        level=level,
        rotation="50 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )

    audit_path = log_dir_path / "kappa_snap_audit.jsonl"
    _auditor = KappaSnapAuditor(audit_path)

    return _logger


def get_auditor() -> KappaSnapAuditor:
    """Get the global kappa-Snap auditor instance.

    Returns:
        The global KappaSnapAuditor, initializing a default if needed.
    """
    global _auditor
    if _auditor is None:
        _auditor = KappaSnapAuditor()
    return _auditor
