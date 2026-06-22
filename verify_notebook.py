import json
import sys

try:
    with open('kaggle/notebook_template.ipynb', 'r') as f:
        nb = json.load(f)
    
    print(f"✅ Notebook JSON is valid")
    print(f"  Cells: {len(nb['cells'])}")
    print(f"  Kernel: {nb['metadata']['kaggle']['accelerator']}")
    print(f"  GPU enabled: {nb['metadata']['kaggle']['isGpuEnabled']}")
    print(f"  Language: {nb['metadata']['kernelspec']['language']}")
    
    # Check CUDA config in code cells
    cuda_mentioned = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'cuda' in source.lower() or 'numba' in source.lower():
                cuda_mentioned = True
                break
    
    print(f"  CUDA mentioned in code: {cuda_mentioned}")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
