"""Configuration loader with YAML parsing and environment variable override."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Loads YAML configuration with environment variable overrides.

    Environment variables in the form ``TOMAS__SECTION__KEY`` override
    corresponding nested keys in the YAML config. For example,
    ``TOMAS__GPU__DEVICE=cuda:0`` sets ``config['gpu']['device']``.
    """

    ENV_PREFIX = "TOMAS__"

    @staticmethod
    def load(config_path: str | Path) -> dict[str, Any]:
        """Load configuration from YAML file with env var overrides.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Configuration dictionary.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        ConfigLoader._apply_env_overrides(config)
        return config

    @staticmethod
    def _apply_env_overrides(config: dict[str, Any]) -> None:
        """Apply environment variable overrides to the config dict in-place.

        Args:
            config: Configuration dictionary to modify.
        """
        for key, value in os.environ.items():
            if not key.startswith(ConfigLoader.ENV_PREFIX):
                continue
            parts = key[len(ConfigLoader.ENV_PREFIX):].lower().split("__")
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            final_key = parts[-1]
            current[final_key] = ConfigLoader._parse_env_value(value)

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse an environment variable string into an appropriate Python type.

        Args:
            value: Raw string value from environment.

        Returns:
            Parsed value (bool, int, float, or str).
        """
        lower = value.lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @staticmethod
    def get(config: dict[str, Any], dot_key: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation.

        Args:
            config: Configuration dictionary.
            dot_key: Dot-separated key path (e.g. "gpu.amp_enabled").
            default: Default value if key not found.

        Returns:
            The value at the key path, or default.
        """
        current: Any = config
        for part in dot_key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
