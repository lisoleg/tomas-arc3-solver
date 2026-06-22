"""Setup configuration for TOMAS ARC-AGI-3 Solver."""
from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

_HERE = Path(__file__).parent

_LONG_DESCRIPTION = (_HERE / "README.md").read_text(encoding="utf-8") if (_HERE / "README.md").exists() else ""

setup(
    name="tomas-arc3-solver",
    version="1.0.0",
    description="TOMAS ARC-AGI-3 Solver — Taiyi Mutual-Play framework for ARC-AGI-3 video reasoning",
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="TOMAS Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "networkx>=3.2",
        "torch>=2.1.0",
        "pyyaml>=6.0",
        "httpx>=0.27.0",
        "loguru>=0.7.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.16.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-cov>=4.1.0", "ruff>=0.4.0", "mypy>=1.10.0"],
    },
    entry_points={
        "console_scripts": [
            "tomas-solve=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
