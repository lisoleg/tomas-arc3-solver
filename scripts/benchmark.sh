#!/usr/bin/env bash
# Performance benchmark for TOMAS ARC-AGI-3 Solver
set -euo pipefail

DATA_DIR="${1:-data/training}"
RESULTS_DIR="${2:-benchmark_results}"
CONFIG="${3:-config/default.yaml}"

mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TOMAS ARC-AGI-3 Solver — Benchmark"
echo "========================================="
echo "Data:    $DATA_DIR"
echo "Results: $RESULTS_DIR"
echo ""

# Count tasks
TASK_COUNT=$(find "$DATA_DIR" -name "*.json" | wc -l)
echo "Found $TASK_COUNT tasks"
echo ""

# Run benchmarks for each mode
for MODE in video bayesian fusion; do
    echo "--- Mode: $MODE ---"
    OUTPUT_FILE="$RESULTS_DIR/${MODE}_submission.json"
    TIME_FILE="$RESULTS_DIR/${MODE}_timing.txt"

    START_TIME=$(date +%s.%N)
    python main.py \
        --mode "$MODE" \
        --input "$DATA_DIR" \
        --output "$OUTPUT_FILE" \
        --config "$CONFIG" \
        --time-budget 80 \
        --verbose 2>&1 | tee "$RESULTS_DIR/${MODE}_log.txt"
    END_TIME=$(date +%s.%N)

    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    AVG_TIME=$(echo "scale=2; $ELAPSED / $TASK_COUNT" | bc)
    echo "Mode: $MODE | Total: ${ELAPSED}s | Avg: ${AVG_TIME}s/task" | tee "$TIME_FILE"
    echo ""
done

echo "Benchmark complete! Results in: $RESULTS_DIR"
