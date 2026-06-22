#!/usr/bin/env python3
"""
使用 Kaggle Python API 直接推送 notebook
绕过 CLI OAuth 问题
"""
import os
import json
import sys
import zipfile
import io

def load_credentials():
    cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(cred_path):
        print(f"[ERROR] Credentials not found: {cred_path}")
        sys.exit(1)
    with open(cred_path) as f:
        return json.load(f)

def push_kernel_via_api():
    """使用 Kaggle API 直接推送"""
    creds = load_credentials()
    
    # 初始化 API
    try:
        from kaggle.api.kaggle_api import KaggleApi
    except ImportError:
        print("[ERROR] Cannot import KaggleApi")
        sys.exit(1)
    
    api = KaggleApi()
    api.authenticate(username=creds["username"], key=creds["key"])
    print(f"[OK] Authenticated as {creds['username']}")
    
    # 读取元数据
    with open("kaggle/kernel-metadata.json") as f:
        metadata = json.load(f)
    
    kernel_slug = metadata["id"].split("/")[-1]
    title = metadata["title"]
    competiton = metadata.get("competition_sources", ["arc-agi-3"])[0]
    
    print(f"\n[INFO] Kernel slug: {kernel_slug}")
    print(f"[INFO] Title: {title}")
    print(f"[INFO] Competition: {competition}")
    
    # 打包 notebook
    notebook_path = "kaggle/notebook_template.ipynb"
    if not os.path.exists(notebook_path):
        print(f"[ERROR] Notebook not found: {notebook_path}")
        sys.exit(1)
    
    # 创建 kernel push 请求
    # Kaggle API 需要通过 HTTP 直接调用
    # 这里使用 kaggle 包的内部方法
    
    print("\n[INFO] Pushing kernel to Kaggle...")
    
    try:
        # 使用 kaggle 包的内部 API
        # kernels_push 需要 zip 文件
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zf:
            zf.write(notebook_path, os.path.basename(notebook_path))
            zf.write("kaggle/kernel-metadata.json", "kernel-metadata.json")
        zip_buf.seek(0)
        
        # 调用 API
        result = api.kernels_push(
            id=metadata["id"],
            title=title,
            code_file=notebook_path,
            language=metadata["language"],
            kernel_type=metadata["kernel_type"],
            is_private=metadata.get("is_private", True),
            enable_gpu=metadata.get("enable_gpu", False),
            enable_internet=metadata.get("enable_internet", False),
            competition_sources=metadata.get("competition_sources", []),
        )
        
        print(f"[OK] Kernel pushed successfully!")
        print(f"\nView at: https://www.kaggle.com/code/{metadata['id']}")
        print(f"\nNext steps:")
        print(f"  1. Go to: https://www.kaggle.com/code/{metadata['id']}")
        print(f"  2. Click 'Run All' to execute the notebook")
        print(f"  3. After completion, click 'Submit to Competition'")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Push failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TOMAS ARC-AGI-3 - Kaggle Push (Python API)")
    print("=" * 60)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    
    push_kernel_via_api()
