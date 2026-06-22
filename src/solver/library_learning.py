"""Library Learning: DreamCoder-style subroutine extraction with JSON persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.dsl_primitives import DSLElement, ProgramNode


class LibraryLearning:
    """DreamCoder-style library learning for DSL subroutine extraction.

    Extracts frequently occurring sub-expression patterns from solved
    programs, adds them to a persistent library, and uses them to
    reduce MDL of future programs.

    Attributes:
        library: Dictionary of learned abstractions.
        frequency_threshold: Min frequency for pattern addition.
        persistence_path: Path to JSON persistence file.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the library learning module.

        Args:
            config: Library config with persistence_path, frequency_threshold.
        """
        self.library: dict[str, dict[str, Any]] = {}
        self.frequency_threshold: int = config.get("frequency_threshold", 3)
        self.persistence_path: str = config.get("persistence_path", "library.json")
        self.max_abstractions: int = config.get("max_abstractions", 200)
        self._pattern_counts: dict[str, int] = {}
        self.load()

    def extract_patterns(
        self, solved_programs: list[ProgramNode]
    ) -> list[DSLElement]:
        """Extract frequent sub-expression patterns from solved programs.

        Counts sub-expression frequencies and returns patterns that
        exceed the frequency threshold.

        Args:
            solved_programs: List of successfully solved ProgramNodes.

        Returns:
            List of DSLElement patterns to potentially add to library.
        """
        # Count sub-expression frequencies
        self._pattern_counts = {}

        for program in solved_programs:
            elements = program.flatten()
            # Count individual elements
            for elem in elements:
                key = f"{elem.name}:{json.dumps(elem.params, sort_keys=True)}"
                self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1

            # Count pairs (depth-2 sub-expressions)
            for i in range(len(elements) - 1):
                pair_key = f"{elements[i].name}+{elements[i + 1].name}"
                self._pattern_counts[pair_key] = self._pattern_counts.get(pair_key, 0) + 1

        # Extract patterns above threshold
        new_patterns: list[DSLElement] = []
        for key, count in self._pattern_counts.items():
            if count >= self.frequency_threshold:
                if "+" not in key and key not in self.library:
                    # Single element pattern
                    parts = key.split(":", 1)
                    name = parts[0]
                    params = json.loads(parts[1]) if len(parts) > 1 else {}
                    elem = DSLElement(name, params)
                    new_patterns.append(elem)

        # Add new patterns to library
        for pattern in new_patterns:
            self.add_to_library(pattern)

        return new_patterns

    def add_to_library(self, pattern: DSLElement) -> None:
        """Add a pattern to the library.

        Args:
            pattern: DSLElement pattern to add.
        """
        if len(self.library) >= self.max_abstractions:
            return

        key = f"{pattern.name}:{json.dumps(pattern.params, sort_keys=True)}"
        if key in self.library:
            self.library[key]["frequency"] += 1
        else:
            self.library[key] = {
                "name": pattern.name,
                "pattern": pattern.to_dict(),
                "frequency": self._pattern_counts.get(key, 1),
                "mdl_cost": pattern.mdl_cost,
            }

    def match_library(self, deltaT: ProgramNode) -> DSLElement | None:
        """Match a program against library abstractions.

        If a sub-expression of the program matches a library abstraction,
        return the abstraction (which has reduced MDL).

        Args:
            deltaT: ProgramNode to match.

        Returns:
            Matched DSLElement abstraction, or None.
        """
        elements = deltaT.flatten()
        for elem in elements:
            key = f"{elem.name}:{json.dumps(elem.params, sort_keys=True)}"
            if key in self.library:
                # Return the library abstraction with reduced MDL
                lib_entry = self.library[key]
                matched = DSLElement.from_dict(lib_entry["pattern"])
                matched.mdl_cost = max(1, lib_entry["mdl_cost"] // 2)
                return matched

            # Also check pair patterns
        for i in range(len(elements) - 1):
            pair_key = f"{elements[i].name}+{elements[i + 1].name}"
            if pair_key in self.library:
                lib_entry = self.library[pair_key]
                matched = DSLElement.from_dict(lib_entry["pattern"])
                matched.mdl_cost = max(1, lib_entry["mdl_cost"] // 2)
                return matched

        return None

    def save(self) -> None:
        """Persist the library to JSON file."""
        data = {
            "abstractions": [
                {
                    "name": entry["name"],
                    "pattern": entry["pattern"],
                    "frequency": entry["frequency"],
                    "mdl_cost": entry["mdl_cost"],
                }
                for entry in self.library.values()
            ]
        }

        try:
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Fail silently on save errors

    def load(self) -> None:
        """Load the library from JSON file.

        Handles corrupt files gracefully by initializing an empty library.
        """
        path = Path(self.persistence_path)
        if not path.exists():
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.library = {}
            for abstraction in data.get("abstractions", []):
                key = f"{abstraction['name']}:{json.dumps(abstraction['pattern'].get('params', {}), sort_keys=True)}"
                self.library[key] = {
                    "name": abstraction["name"],
                    "pattern": abstraction["pattern"],
                    "frequency": abstraction.get("frequency", 1),
                    "mdl_cost": abstraction.get("mdl_cost", 5),
                }
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupt file: initialize empty library
            self.library = {}

    def get_abstractions(self) -> list[DSLElement]:
        """Get all library abstractions as DSLElement instances.

        Returns:
            List of DSLElement abstractions.
        """
        return [
            DSLElement.from_dict(entry["pattern"])
            for entry in self.library.values()
        ]

    def get_size(self) -> int:
        """Get the number of abstractions in the library.

        Returns:
            Library size.
        """
        return len(self.library)

    def clear(self) -> None:
        """Clear the library."""
        self.library = {}
        self._pattern_counts = {}
