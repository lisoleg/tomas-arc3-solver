#!/usr/bin/env bash
# Download ARC-AGI-3 competition data
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "========================================="
echo "  Downloading ARC-AGI-3 Data"
echo "========================================="

# Try Kaggle CLI first
if command -v kaggle &> /dev/null; then
    echo "Using Kaggle CLI to download competition data..."
    kaggle competitions download -c arc-agi-3 -p "$DATA_DIR" || {
        echo "Kaggle download failed. Trying manual download..."
    }
    if ls "$DATA_DIR"/*.zip 1> /dev/null 2>&1; then
        echo "Extracting data..."
        unzip -o "$DATA_DIR"/*.zip -d "$DATA_DIR"
        rm -f "$DATA_DIR"/*.zip
    fi
else
    echo "Kaggle CLI not found. Please install: pip install kaggle"
    echo "Or manually download data to: $DATA_DIR"
    echo ""
    echo "Expected structure:"
    echo "  $DATA_DIR/"
    echo "    ├── training/"
    echo "    │   └── *.json"
    echo "    ├── evaluation/"
    echo "    │   └── *.json"
    echo "    └── sample_submission.json"
fi

echo ""
echo "Data directory: $DATA_DIR"
ls -la "$DATA_DIR" || true
echo "Done!"
