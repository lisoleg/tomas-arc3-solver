#!/usr/bin/env bash
# Setup environment for TOMAS ARC-AGI-3 Solver
set -euo pipefail

echo "========================================="
echo "  TOMAS ARC-AGI-3 Solver — Environment Setup"
echo "========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "Installing TOMAS solver package..."
pip install -e .

# Create necessary directories
mkdir -p logs runs data config

echo ""
echo "Setup complete! Activate with: source .venv/bin/activate"
echo "Run solver with: python main.py --help"
