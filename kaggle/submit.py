#!/usr/bin/env python
"""Kaggle Submission Helper for TOMAS ARC-AGI-3 Solver

This script helps upload the notebook and submit to the ARC-AGI-3 competition.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Get API credentials from https://www.kaggle.com/settings → Create New API Token
3. Place kaggle.json at ~/.kaggle/kaggle.json (Linux/macOS) or %USERPROFILE%\\.kaggle\\kaggle.json (Windows)
4. Join the competition at https://www.kaggle.com/competitions/arc-agi-3

Usage:
    python kaggle/submit.py --notebook kaggle/notebook_template.ipynb
    python kaggle/submit.py --check          # Check credentials and competition status
    python kaggle/submit.py --upload          # Upload notebook as a Kaggle dataset
    python kaggle/submit.py --submit          # Submit notebook to competition
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPETITION = "arc-agi-3"
NOTEBOOK_PATH = PROJECT_ROOT / "kaggle" / "notebook_template.ipynb"


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print("[ERROR] Kaggle credentials not found!")
        print(f"  Expected: {kaggle_json}")
        print("\nTo set up Kaggle credentials:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New API Token' — downloads kaggle.json")
        print(f"  3. Place it at: {kaggle_json}")
        print("  4. Set permissions: chmod 600 ~/.kaggle/kaggle.json (Linux/macOS)")
        print("     On Windows: ensure only your user has read access")
        return False

    print(f"[OK] Kaggle credentials found at: {kaggle_json}")

    # Verify credentials work
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "list", "-s", COMPETITION],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("[OK] Kaggle API connection successful")
            if COMPETITION in result.stdout:
                print(f"[OK] Competition '{COMPETITION}' found")
            else:
                print(f"[WARN] Competition '{COMPETITION}' not found in list")
                print(f"  Make sure you've joined: https://www.kaggle.com/competitions/{COMPETITION}")
            return True
        else:
            print(f"[ERROR] Kaggle API failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("[ERROR] kaggle CLI not found. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"[ERROR] Kaggle API check failed: {e}")
        return False


def upload_notebook() -> bool:
    """Upload the notebook as a Kaggle notebook."""
    if not NOTEBOOK_PATH.exists():
        print(f"[ERROR] Notebook not found: {NOTEBOOK_PATH}")
        return False

    print(f"Uploading notebook: {NOTEBOOK_PATH}")

    # Create metadata for the notebook
    metadata = {
        "id": "tomas-arc-agi3-solver",
        "title": "TOMAS ARC-AGI-3 Solver v2.4",
        "code_file": str(NOTEBOOK_PATH),
        "language": "python",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_type": "notebook",
        "competition_sources": [COMPETITION],
    }

    # Push notebook
    try:
        cmd = [
            "kaggle", "kernels", "push",
            "-p", str(NOTEBOOK_PATH.parent)
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("[OK] Notebook uploaded successfully")
            print(result.stdout)
            return True
        else:
            print(f"[ERROR] Upload failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return False


def submit_to_competition() -> bool:
    """Submit the notebook to the ARC-AGI-3 competition."""
    print(f"Submitting to competition: {COMPETITION}")

    try:
        # List existing submissions
        cmd = ["kaggle", "competitions", "submissions", COMPETITION]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("Current submissions:")
            print(result.stdout)
        else:
            print(f"[WARN] Could not list submissions: {result.stderr}")

        print("\nTo submit:")
        print(f"  1. Go to https://www.kaggle.com/competitions/{COMPETITION}/submit")
        print(f"  2. Upload the notebook: {NOTEBOOK_PATH}")
        print(f"  3. Or use: kaggle kernels push -p {NOTEBOOK_PATH.parent}")
        print(f"  4. Then submit the kernel output as competition submission")

        return True
    except Exception as e:
        print(f"[ERROR] Submit failed: {e}")
        return False


def verify_notebook() -> bool:
    """Verify the notebook has correct v2.4 configuration."""
    if not NOTEBOOK_PATH.exists():
        print(f"[ERROR] Notebook not found: {NOTEBOOK_PATH}")
        return False

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Check notebook cells for v2.4 config
    full_text = ""
    for cell in nb.get("cells", []):
        for source_line in cell.get("source", []):
            full_text += source_line

    checks = [
        ("2.4.0", "Version 2.4.0"),
        ("psi_gate", "psi-Gate configuration"),
        ("aegis", "AEGIS evolver"),
        ("TOMASSolver", "TOMAS Solver import"),
    ]

    all_pass = True
    for keyword, desc in checks:
        if keyword in full_text:
            print(f"  [OK] {desc}")
        else:
            print(f"  [MISS] {desc}")
            all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Kaggle submission helper for TOMAS ARC-AGI-3")
    parser.add_argument("--check", action="store_true", help="Check credentials and notebook")
    parser.add_argument("--upload", action="store_true", help="Upload notebook to Kaggle")
    parser.add_argument("--submit", action="store_true", help="Submit to competition")
    parser.add_argument("--notebook", type=str, help="Notebook file path")

    args = parser.parse_args()

    print("=" * 60)
    print("TOMAS ARC-AGI-3 Kaggle Submission Helper")
    print("=" * 60)

    # Default: check everything
    if not any([args.check, args.upload, args.submit]):
        args.check = True

    if args.check:
        print("\n--- Checking Kaggle Setup ---")
        check_kaggle_credentials()

        print("\n--- Verifying Notebook ---")
        if verify_notebook():
            print("\n[OK] Notebook is ready for submission")
        else:
            print("\n[WARN] Notebook needs updates before submission")

    if args.upload:
        print("\n--- Uploading Notebook ---")
        upload_notebook()

    if args.submit:
        print("\n--- Submitting to Competition ---")
        submit_to_competition()

    print("\n" + "=" * 60)
    print("Manual submission steps:")
    print(f"  1. Ensure kaggle.json is at ~/.kaggle/kaggle.json")
    print(f"  2. Join competition: https://www.kaggle.com/competitions/{COMPETITION}")
    print(f"  3. Upload notebook: kaggle kernels push -p {NOTEBOOK_PATH.parent}")
    print(f"  4. Or upload via web: https://www.kaggle.com/competitions/{COMPETITION}/submit")
    print("=" * 60)


if __name__ == "__main__":
    main()
