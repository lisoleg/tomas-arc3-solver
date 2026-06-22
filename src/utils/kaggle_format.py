"""ARC-AGI-3 JSON parsing and submission package generation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class VideoARCTask:
    """Data structure for an ARC-AGI-3 video task.

    Attributes:
        task_id: Unique task identifier.
        demo_pairs: List of (input_frames, output_frames) demo pairs.
        test_frames: List of test input frames to predict.
        demo_frames: Flattened list of all demo frames.
    """

    task_id: str = ""
    demo_pairs: list[dict[str, Any]] = field(default_factory=list)
    test_frames: list[np.ndarray] = field(default_factory=list)
    demo_frames: list[np.ndarray] = field(default_factory=list)

    def parse(self, raw_data: dict[str, Any]) -> None:
        """Parse raw JSON data into structured fields.

        Args:
            raw_data: Raw task dictionary with 'train' and 'test' keys.
        """
        self.demo_pairs = []
        self.demo_frames = []
        for pair in raw_data.get("train", []):
            input_frames = [np.array(f, dtype=np.int8) for f in pair.get("input", [])]
            output_frames = [np.array(f, dtype=np.int8) for f in pair.get("output", [])]
            self.demo_pairs.append({"input": input_frames, "output": output_frames})
            self.demo_frames.extend(input_frames)
            self.demo_frames.extend(output_frames)

        self.test_frames = []
        for test_item in raw_data.get("test", []):
            frames = [np.array(f, dtype=np.int8) for f in test_item.get("input", [])]
            self.test_frames.extend(frames)

    def get_demo_input_grids(self) -> list[np.ndarray]:
        """Return all demo input grids.

        Returns:
            List of input grids from demo pairs.
        """
        grids: list[np.ndarray] = []
        for pair in self.demo_pairs:
            grids.extend(pair["input"])
        return grids

    def get_demo_output_grids(self) -> list[np.ndarray]:
        """Return all demo output grids.

        Returns:
            List of output grids from demo pairs.
        """
        grids: list[np.ndarray] = []
        for pair in self.demo_pairs:
            grids.extend(pair["output"])
        return grids


class KaggleFormatAdapter:
    """Adapts ARC-AGI-3 JSON input/output to internal data structures."""

    def __init__(self, input_format: str = "json") -> None:
        """Initialize the adapter.

        Args:
            input_format: Expected input format identifier.
        """
        self.input_format = input_format

    def parse_input(self, raw_data: dict[str, Any]) -> VideoARCTask:
        """Parse raw JSON task data into a VideoARCTask.

        Args:
            raw_data: Raw dictionary from ARC-AGI-3 JSON.

        Returns:
            Parsed VideoARCTask instance.
        """
        task = VideoARCTask(task_id=raw_data.get("task_id", "unknown"))
        task.parse(raw_data)
        return task

    def generate_submission(self, predictions: dict[str, Any]) -> str:
        """Generate a Kaggle submission JSON string.

        Args:
            predictions: Dictionary mapping task IDs to predicted grids.

        Returns:
            JSON string of the submission.
        """
        submission: dict[str, Any] = {}
        for task_id, pred in predictions.items():
            if isinstance(pred, np.ndarray):
                submission[task_id] = pred.tolist()
            elif isinstance(pred, list):
                submission[task_id] = pred
            else:
                submission[task_id] = pred
        return json.dumps(submission, ensure_ascii=False)

    def validate_output(self, output: dict[str, Any]) -> bool:
        """Validate that an output dictionary conforms to submission format.

        Args:
            output: Output dictionary to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(output, dict):
            return False
        for task_id, pred in output.items():
            if not isinstance(task_id, str):
                return False
            if isinstance(pred, np.ndarray):
                continue
            if isinstance(pred, list):
                continue
            return False
        return True

    def load_task_file(self, file_path: str | Path) -> VideoARCTask:
        """Load a single task from a JSON file.

        Args:
            file_path: Path to the task JSON file.

        Returns:
            Parsed VideoARCTask.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        task_id = Path(file_path).stem
        raw_data["task_id"] = task_id
        return self.parse_input(raw_data)

    def save_submission(
        self, predictions: dict[str, Any], output_path: str | Path
    ) -> None:
        """Save predictions to a submission file.

        Args:
            predictions: Dictionary of predictions.
            output_path: Path to write the submission file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission_str = self.generate_submission(predictions)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(submission_str)
