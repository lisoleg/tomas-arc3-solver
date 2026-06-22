#!/usr/bin/env bash
# Auto-shutdown timer — stops the machine after work is done
set -euo pipefail

MAX_HOURS="${1:-9}"
WARN_MINUTES=10

echo "========================================="
echo "  TOMAS Shutdown Timer"
echo "  Max runtime: ${MAX_HOURS}h | Warning: ${WARN_MINUTES}min before"
echo "========================================="

TOTAL_SECONDS=$((MAX_HOURS * 3600))
WARN_SECONDS=$((WARN_MINUTES * 60))
SLEEP_SECONDS=$((TOTAL_SECONDS - WARN_SECONDS))

echo "Sleeping for $((SLEEP_SECONDS / 3600))h $((SLEEP_SECONDS % 3600 / 60))m before warning..."
sleep "$SLEEP_SECONDS"

echo ""
echo "⚠️  WARNING: Machine will shut down in ${WARN_MINUTES} minutes!"
echo "Press Ctrl+C to cancel."
sleep "$WARN_SECONDS"

echo "Shutting down..."
# On Kaggle, this stops the notebook
if [ -f /kaggle/working ]; then
    touch /kaggle/working/__stop__
fi
poweroff 2>/dev/null || shutdown -h now 2>/dev/null || true
