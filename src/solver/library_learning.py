"""Library Learning: DreamCoder-style subroutine extraction with JSON persistence.

TOMAS v3.0: Sleep-Step continual learning — extract sub-expressions from
solved programs, compute MDL compression gain, and register new primitives
into the DSL registry for monotonic coverage improvement (Theorem 4).
"""
from __future__ import annotations

import hashlib
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

    # ============================================================
    # Sleep-Step: Sub-expression extraction and primitive learning
    # ============================================================

    def extract_subexpressions(
        self,
        program: ProgramNode,
        max_ast_depth: int = 3,
    ) -> list[tuple[str, ProgramNode]]:
        """Recursively extract sub-expressions from ProgramNode AST.

        Traverses the program tree and extracts all chain-composition
        subtrees (the most common composition type in ARC solving).
        Each subtree is assigned a canonical hash for cross-program
        deduplication.

        Args:
            program: ProgramNode to extract from.
            max_ast_depth: Maximum AST depth to traverse.

        Returns:
            List of (canonical_hash, subtree) pairs. canonical_hash is
            a JSON string of the subtree structure for dedup.
        """
        result: list[tuple[str, ProgramNode]] = []

        def _extract(node: ProgramNode, depth: int) -> None:
            """Recursively extract chain subtrees."""
            if depth > max_ast_depth:
                return

            # Extract chain subtrees (most common composition type)
            if node.combo_type == "chain" and node.children:
                try:
                    canonical_hash = json.dumps(
                        node.to_dict(), sort_keys=True
                    )
                except Exception:
                    canonical_hash = repr(node.to_dict())
                result.append((canonical_hash, node.clone()))

            # Recurse into children
            for child in node.children:
                _extract(child, depth + 1)

        _extract(program, 0)
        return result

    def compute_mdl_gain(
        self,
        subexpr_hash: str,
        subexpr: ProgramNode,
        frequency: int,
    ) -> int:
        """Compute MDL compression gain for a sub-expression.

        The gain represents how much description length is saved by
        registering this sub-expression as a new primitive:

            gain = Len(uncompressed) * frequency
                   - Len(compressed) * frequency
                   - registration_cost

        Where:
            - Len(uncompressed) = subexpr.compute_mdl() (original cost)
            - Len(compressed) = 1 (after registration, it's a single primitive)
            - registration_cost = 10 (overhead of adding to the library)

        Args:
            subexpr_hash: Canonical hash of the sub-expression.
            subexpr: The ProgramNode sub-expression.
            frequency: Number of times this sub-expression appears
                across all solved programs.

        Returns:
            Integer MDL gain. Higher is better. Negative means
            registration is not worthwhile.
        """
        uncompressed_len: int = subexpr.compute_mdl()
        compressed_len: int = 1  # After registration: single primitive
        registration_cost: int = 10  # Overhead of library entry

        gain = (
            uncompressed_len * frequency
            - compressed_len * frequency
            - registration_cost
        )
        return gain

    def sleep_step(
        self,
        solved_programs: list[ProgramNode],
        max_ast_depth: int = 3,
        max_total_subexprs: int = 5000,
        min_freq: int = 3,
        mdl_gain_threshold: int = 5,
        max_new: int = 15,
    ) -> list[DSLElement]:
        """Execute Sleep-Step: extract → filter → register new primitives.

        The Sleep-Step is the core of continual learning (Theorem 4:
        ⟨𝓛ₑ₊₁⟩ ⊇ ⟨𝓛ₑ⟩, coverage monotonically non-decreasing). It
        extracts frequently occurring sub-expressions from solved
        programs and registers them as new DSL primitives.

        Pipeline:
            1. Extract sub-expressions from all solved programs
            2. Count frequency across programs (dedup by canonical hash)
            3. Filter by min_freq (must appear in multiple programs)
            4. Compute MDL gain, filter by threshold
            5. Sort by MDL gain descending
            6. Register top max_new as new DSLElement primitives

        Args:
            solved_programs: List of successfully solved ProgramNodes.
            max_ast_depth: Maximum AST depth for sub-expression extraction.
            max_total_subexprs: Maximum total sub-expressions to collect.
            min_freq: Minimum frequency for a sub-expression to be
                considered for registration.
            mdl_gain_threshold: Minimum MDL gain for registration.
            max_new: Maximum number of new primitives to register.

        Returns:
            List of newly registered DSLElement primitives.
        """
        if not solved_programs:
            return []

        # 1. Extract sub-expressions from all solved programs
        # hash -> (subtree, frequency)
        all_subexprs: dict[str, tuple[ProgramNode, int]] = {}

        for program in solved_programs:
            subexprs = self.extract_subexpressions(
                program, max_ast_depth
            )
            for hash_str, subtree in subexprs:
                if hash_str in all_subexprs:
                    existing_subtree, freq = all_subexprs[hash_str]
                    all_subexprs[hash_str] = (existing_subtree, freq + 1)
                else:
                    all_subexprs[hash_str] = (subtree, 1)

            # Early termination if too many sub-expressions
            if len(all_subexprs) > max_total_subexprs:
                break

        # 2. Filter by min_freq
        frequent: dict[str, tuple[ProgramNode, int]] = {
            h: (s, f)
            for h, (s, f) in all_subexprs.items()
            if f >= min_freq
        }

        if not frequent:
            return []

        # 3. Compute MDL gain, filter by threshold
        gains: list[tuple[int, str, ProgramNode, int]] = []
        for hash_str, (subtree, freq) in frequent.items():
            gain = self.compute_mdl_gain(hash_str, subtree, freq)
            if gain >= mdl_gain_threshold:
                gains.append((gain, hash_str, subtree, freq))

        if not gains:
            return []

        # 4. Sort by MDL gain descending
        gains.sort(key=lambda x: x[0], reverse=True)

        # 5. Register top max_new as new DSLElement primitives
        new_primitives: list[DSLElement] = []
        for gain, hash_str, subtree, freq in gains[:max_new]:
            # Generate a short unique name for the learned primitive
            short_hash = hashlib.md5(
                hash_str.encode("utf-8")
            ).hexdigest()[:8]
            prim_name = f"learned_{short_hash}"

            # Skip if already registered
            if prim_name in DSLElement._registry:
                continue

            # Create DSLElement with compressed MDL cost
            elem = DSLElement(prim_name, {})
            elem.mdl_cost = 1  # Compressed cost (single primitive)

            # Register delegate function that executes the subtree
            DSLElement._registry[prim_name] = self._make_learned_delegate(
                subtree
            )

            # Add to library
            self.add_to_library(elem)
            new_primitives.append(elem)

        return new_primitives

    @staticmethod
    def _make_learned_delegate(subtree: ProgramNode) -> Any:
        """Create a delegate function for a learned primitive.

        The delegate function applies the subtree to the input grid,
        allowing the learned sub-expression to be used as a single
        DSL primitive.

        Args:
            subtree: The ProgramNode sub-expression to encapsulate.

        Returns:
            A callable that takes (grid, **kwargs) and returns the
            transformed grid.
        """

        def delegate(grid: Any, **_: Any) -> Any:
            """Apply the learned sub-expression to a grid.

            Args:
                grid: Input grid as numpy ndarray.
                **_: Ignored keyword arguments (for DSL registry compat).

            Returns:
                Transformed grid.
            """
            return subtree.apply(grid)

        return delegate
