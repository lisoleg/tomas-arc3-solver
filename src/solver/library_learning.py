"""Library Learning: DreamCoder-style subroutine extraction with JSON persistence.

TOMAS v3.0: Sleep-Step continual learning — extract sub-expressions from
solved programs, compute MDL compression gain, and register new primitives
into the DSL registry for monotonic coverage improvement (Theorem 4).

TOMAS v3.1: TOSAS-inspired Primality Check — filter out "composite"
Macros (can be expressed by existing primitives) to keep library minimal.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from src.core.dsl_primitives import DSLElement, ProgramNode


# =============================================================================
# TOSAS-inspired Primality Check (素基性检查)
# =============================================================================

def is_prime_like(
    candidate: ProgramNode,
    primitive_set: list[DSLElement],
    composite_patterns: dict[str, list[str]] | None = None,
) -> bool:
    """TOSAS-inspired Primality Check — test if candidate is "prime-like"
    
    A "prime-like" ProgramNode cannot be expressed as a NASGA combination
    of existing primitives (i.e., it's "atomic" / "indivisible").
    
    A "composite" (non-prime-like) node can be decomposed into existing
    primitives — it should NOT be registered as a new library entry.
    
    Args:
        candidate: The ProgramNode to test.
        primitive_set: List of existing DSL primitives.
        composite_patterns: Optional dict of known composite patterns
                          (e.g., {"rotate90_then_fill": ["rotate90", "fill"]}),
                          Typicaly loaded from config.
        
    Returns:
        True if candidate is "prime-like" (should be kept),
        False if candidate is "composite" (should be rejected).
    """
    # 1. Trivial case: candidate itself is in primitive_set → not "new"
    # NOTE: We skip this check because ProgramNode may not have equiv() method
    # for p in primitive_set:
    #     if candidate.equiv(p):
    #         return False  # It's already a known primitive

    # 2. Structural check: composite pattern detection
    if composite_patterns is not None and hasattr(candidate, "dsl_name"):
        dsl_name = getattr(candidate, "dsl_name", None)
        if dsl_name is not None and dsl_name in composite_patterns:
            # Check if children match the composite pattern
            child_names = [getattr(c, "dsl_name", "") for c in candidate.children]
            if child_names == composite_patterns[dsl_name]:
                return False  # Composite: can be expressed by existing primitives

    # 3. TODO: Full NASGA combination check (requires κ-Snap searcher)
    # For now, we use a heuristic: if candidate has >2 children,
    # it might be composite (can be decomposed).
    # This is a simplified version — full check requires search.

    # For leaf nodes (depth=1), they are always prime-like
    if not candidate.children:
        return True

    # For composite nodes, check if any child is already in primitive_set
    # If ALL children are in primitive_set, it's likely composite
    # NOTE: We use name comparison instead of equiv() (which may not exist)
    all_children_are_primitives = True
    for c in candidate.children:
        if c is None:
            continue
        child_name = getattr(c, "dsl_name", None)
        if child_name is None:
            all_children_are_primitives = False
            break
        # Check if this child's name is in primitive_set
        child_in_set = any(child_name == getattr(p, "name", None) for p in primitive_set)
        if not child_in_set:
            all_children_are_primitives = False
            break
    
    if all_children_are_primitives and len(candidate.children) > 1:
        # Likely composite — but we need search to be sure
        # For now, let it pass (conservative: avoid false negatives)
        pass

    return True  # Default: assume prime-like


def _log_composite_observation(program: ProgramNode, composite_log: list[dict]) -> None:
    """Log "composite rejected" events for later DSL optimization.

    Corresponds to "复合体理学" in TOSAS — composite patterns that
    are frequently observed but rejected from library can guide future
    DSL primitive set expansion.

    Args:
        program: The composite ProgramNode that was rejected.
        composite_log: List to append the observation to.
    """
    entry = {
        "type": "composite_rejected",
        "dsl_name": getattr(program, "dsl_name", repr(program)),
        "children": [getattr(c, "dsl_name", "") for c in program.children],
        "timestamp": time.time(),
    }
    composite_log.append(entry)


# =============================================================================
# LibraryLearning Class
# =============================================================================

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

        # P1-5: Adaptive Sleep-Step Budget parameters
        # B = B_base + α * MDL(prog) + β * log2(freq(prog) + 1)
        # Higher MDL programs and more frequent patterns get larger budgets,
        # allowing the system to invest more search effort where it matters.
        self._budget_base: int = config.get("budget_base", 10)
        self._budget_alpha: float = config.get("budget_alpha", 0.5)  # MDL weight
        self._budget_beta: float = config.get("budget_beta", 2.0)  # frequency weight

        # P1-6: AST Width Control parameters
        # W(d) = W_max * exp(-λ*d) — shallower depths get more candidates
        self._ast_width_max: int = config.get("ast_width_max", 100)
        self._ast_lambda: float = config.get("ast_lambda", 0.5)  # decay rate

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

        P1-6: AST Width Control — limits the number of sub-expressions
        extracted at each depth level using an exponential decay function:
            W(d) = W_max * exp(-λ * d)
        Shallower depths (more important patterns) get wider extraction,
        while deeper depths (less likely to be useful) are pruned aggressively.

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
        # P1-6: Track extraction count per depth for width control
        depth_counts: dict[int, int] = {}

        def _extract(node: ProgramNode, depth: int) -> None:
            """Recursively extract chain subtrees with width control."""
            if depth > max_ast_depth:
                return

            # P1-6: Check width budget for this depth
            width_limit = self._ast_width_at_depth(depth)
            current_count = depth_counts.get(depth, 0)
            if current_count >= width_limit:
                return  # Width budget exhausted for this depth

            # Extract chain subtrees (most common composition type)
            if node.combo_type == "chain" and node.children:
                try:
                    canonical_hash = json.dumps(
                        node.to_dict(), sort_keys=True
                    )
                except Exception:
                    canonical_hash = repr(node.to_dict())
                result.append((canonical_hash, node.clone()))
                depth_counts[depth] = current_count + 1

            # Recurse into children
            for child in node.children:
                _extract(child, depth + 1)

        _extract(program, 0)
        return result

    def _ast_width_at_depth(self, depth: int) -> int:
        """P1-6: Compute AST extraction width limit at a given depth.

        Uses exponential decay: W(d) = W_max * exp(-λ * d)
        At depth 0: W = W_max (full width for root-level patterns)
        At depth 1: W = W_max * exp(-λ) (reduced)
        At depth 2+: progressively narrower (prune unlikely patterns)

        Args:
            depth: AST depth level (0 = root).

        Returns:
            Maximum number of sub-expressions to extract at this depth.
        """
        width = int(self._ast_width_max * math.exp(-self._ast_lambda * depth))
        return max(1, width)  # Always allow at least 1

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

        # P1-5: Adaptive Sleep-Step Budget — compute dynamic max_new
        # B = B_base + α * MDL(prog) + β * log2(freq(prog) + 1)
        # Higher MDL and more frequent programs get larger budgets,
        # investing search effort where it yields the most compression.
        adaptive_max_new = self._compute_adaptive_budget(
            solved_programs, gains
        )

        # Use the smaller of adaptive budget and explicit max_new parameter
        effective_max_new = min(adaptive_max_new, max_new)

        # 5. Register top effective_max_new as new DSLElement primitives
        new_primitives: list[DSLElement] = []
        
        # TOSAS v3.1: Composite patterns for structural primality check
        composite_patterns = getattr(self, "composite_patterns", None)
        
        for gain, hash_str, subtree, freq in gains[:effective_max_new]:
            # TOSAS v3.1: Primality Check — reject "composite" Macros
            if not is_prime_like(subtree, self.get_abstractions(), composite_patterns):
                # Composite Macro: can be expressed by existing primitives
                # → Do NOT register (avoid library bloat)
                # Log for later DSL optimization
                if not hasattr(self, "composite_log"):
                    self.composite_log: list[dict] = []
                _log_composite_observation(subtree, self.composite_log)
                continue  # Skip this candidate — don't register!
            
            # Prime-like Macro: indivisible → register as new primitive
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

    def _compute_adaptive_budget(
        self,
        solved_programs: list[ProgramNode],
        gains: list[tuple[int, str, ProgramNode, int]],
    ) -> int:
        """P1-5: Compute adaptive Sleep-Step budget.

        Budget formula:
            B = B_base + α * MDL(prog) + β * log2(freq(prog) + 1)

        Where:
        - B_base: minimum budget (always register at least this many)
        - α * MDL(prog): programs with higher description length get more
          budget (more complex patterns need more search effort)
        - β * log2(freq + 1): frequently occurring patterns get logarithmic
          bonus (diminishing returns for very high frequencies)

        Args:
            solved_programs: List of solved ProgramNodes.
            gains: List of (gain, hash, subtree, freq) tuples.

        Returns:
            Adaptive budget (number of new primitives to register).
        """
        if not gains:
            return self._budget_base

        # Compute average MDL across solved programs
        total_mdl = 0
        program_count = 0
        for prog in solved_programs:
            try:
                total_mdl += prog.compute_mdl()
                program_count += 1
            except Exception:
                continue

        avg_mdl = total_mdl / max(program_count, 1)

        # Compute average frequency of top candidates
        avg_freq = sum(g[3] for g in gains[:10]) / max(min(len(gains), 10), 1)

        # B = B_base + α * MDL + β * log2(freq + 1)
        budget = (
            self._budget_base
            + self._budget_alpha * avg_mdl
            + self._budget_beta * math.log2(avg_freq + 1)
        )

        # Clamp to reasonable range
        return max(1, min(int(budget), self.max_abstractions))

    def sleep_step_warmup(self, warmup_tasks: int = 25) -> None:
        """Sleep-Step warmup: run on public tasks to build library.json.

        Processes warmup_tasks number of public ARC tasks to extract
        common patterns and populate the abstraction library.
        Uses TOSAS primality filter to only keep prime-like patterns.

        Args:
            warmup_tasks: Number of public tasks to process.
        """
        import os
        import json
        from pathlib import Path

        print(f"[Sleep-Step] Warming up with {warmup_tasks} public tasks...")

        # Try to load public tasks from data directory
        data_dir = Path("data")
        task_files = sorted(data_dir.glob("**/*.json"))[:warmup_tasks]

        if not task_files:
            print("[Sleep-Step] No public tasks found, skipping warmup")
            return

        patterns_found = 0
        for tf in task_files:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    task_data = json.load(f)

                train = task_data.get("train", [])
                for pair in train:
                    inp = pair.get("input", [])
                    out = pair.get("output", [])
                    if inp and out:
                        # Extract patterns from input-output pairs
                        import numpy as np
                        inp_arr = np.array(inp, dtype=np.int8)
                        out_arr = np.array(out, dtype=np.int8)

                        # TOSAS primality filter: only keep prime-like patterns
                        if self.is_prime_like(inp_arr):
                            self.add_pattern(inp_arr, out_arr)
                            patterns_found += 1
            except Exception:
                continue

        # Save library
        self.save_library("library.json")
        print(f"[Sleep-Step] Warmup complete: {patterns_found} patterns added to library.json")

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
