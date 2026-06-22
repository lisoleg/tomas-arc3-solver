#!/usr/bin/env python3
"""
TOMAS ARC-AGI-3 Kaggle 提交助手
使用 Kaggle API (Python) 直接提交，绕过 CLI OAuth 问题
"""
import os
import json
import sys

def load_credentials():
    """从 ~/.kaggle/kaggle.json 加载凭证"""
    cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(cred_path):
        print(f"[ERROR] Credentials not found at {cred_path}")
        print("  Please create the file with: {\"username\":\"YOUR_USER\",\"key\":\"YOUR_KEY\"}")
        sys.exit(1)
    
    with open(cred_path) as f:
        creds = json.load(f)
    return creds["username"], creds["key"]

def push_notebook():
    """推送 notebook 到 Kaggle（作为 Kernel）"""
    try:
        import kaggle
    except ImportError:
        print("[ERROR] kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)
    
    username, key = load_credentials()
    
    # 设置环境变量（kaggle API 会读取这些）
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    
    notebook_path = "kaggle/notebook_template.ipynb"
    if not os.path.exists(notebook_path):
        print(f"[ERROR] Notebook not found: {notebook_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("TOMAS ARC-AGI-3 - Kaggle Kernel Push")
    print("=" * 60)
    print(f"\n[INFO] Pushing notebook: {notebook_path}")
    print(f"[INFO] User: {username}")
    
    # 使用 kaggle API 推送 kernel
    # 注意：kaggle kernels push 需要 CLI，这里用 Python API 包装
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "push", "-p", "kaggle"],
        capture_output=True, text=True,
        env={**os.environ, "KAGGLE_USERNAME": username, "KAGGLE_KEY": key}
    )
    
    print("\n--- STDOUT ---")
    print(result.stdout or "(empty)")
    print("--- STDERR ---")
    print(result.stderr or "(empty)")
    
    if result.returncode == 0:
        print("\n[OK] Notebook pushed successfully!")
        print("\nNext steps:")
        print("  1. Go to: https://www.kaggle.com/code/YOUR_USERNAME/tomas-arc3-solver")
        print("  2. Click 'Run All' to execute the notebook")
        print("  3. After completion, click 'Submit to Competition'")
    else:
        print(f"\n[ERROR] Push failed (exit code {result.returncode})")
        return False
    
    return True

def check_competition_access():
    """检查竞赛访问权限"""
    username, key = load_credentials()
    
    print("\n--- Checking Competition Access ---")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "files", "arc-agi-3"],
        capture_output=True, text=True,
        env={**os.environ, "KAGGLE_USERNAME": username, "KAGGLE_KEY": key}
    )
    
    if "404" in result.stderr or "Not Found" in result.stderr:
        print("[WARNING] Competition 'arc-agi-3' not found or not joined")
        print("  Please join at: https://www.kaggle.com/competitions/arc-agi-3/rules")
        return False
    else:
        print("[OK] Competition accessible")
        return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TOMAS ARC-AGI-3 - Kaggle Submission Helper (API)")
    print("=" * 60)
    
    # 1. 检查凭证
    print("\n--- Step 1: Check Credentials ---")
    try:
        username, key = load_credentials()
        print(f"[OK] Credentials loaded (user: {username})")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # 2. 检查竞赛权限
    print("\n--- Step 2: Check Competition Access ---")
    check_competition_access()
    
    # 3. 推送 notebook
    print("\n--- Step 3: Push Notebook ---")
    push_notebook()
