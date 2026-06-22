#!/usr/bin/env bash
# One-click solve for Kaggle ARC-AGI-3
set -euo pipefail

INPUT_DIR="${1:-/kaggle/input/arc-agi-3}"
OUTPUT_DIR="${2:-/kaggle/working}"
CONFIG="${3:-config/default.yaml}"

echo "========================================="
echo "  TOMAS ARC-AGI-3 Solver — Kaggle Solve"
echo "========================================="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Config: $CONFIG"
echo ""

# Run solver
python main.py \
    --mode auto \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR/submission.json" \
    --config "$CONFIG" \
    --time-budget 80 \
    --verbose

echo ""
echo "Solve complete!"
echo "Submission: $OUTPUT_DIR/submission.json"
