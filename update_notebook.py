import json

# Read notebook
with open('kaggle/notebook_template.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update Cell 4 (config cell)
new_source = '''import sys
import json
from pathlib import Path

# Add solver to path
solver_path = '/kaggle/input/tomas-arc3-solver'
if solver_path not in sys.path:
    sys.path.insert(0, solver_path)

from src.utils.config import load_config
from src.solver.tomas_solver import TOMASSolver

# Load config from file (v2.4.1 optimized settings)
config_path = f'{solver_path}/config/default.yaml'
config = load_config(config_path)

# === Kaggle Environment Overrides ===
config['kaggle'] = {
    'input_dir': '/kaggle/input/arc-agi-3',
    'output_dir': '/kaggle/working',
}
config['vl_api'] = {'available': False}  # No external API

# === CUDA GPU Configuration (CRITICAL) ===
config['cuda'] = config.get('cuda', {})
config['cuda']['enabled'] = True
config['cuda']['use_numba'] = True
config['cuda']['max_vram_gb'] = 16  # T4 GPU
config['cuda']['block_size'] = 256
config['cuda']['grid_size'] = 64

# === v2.4.1 Optimized Feature Configuration ===
config['psi_gate'] = config.get('psi_gate', {})
config['psi_gate']['enabled'] = True
config['psi_gate']['tolerance_decay_rate'] = 0.03  # Stricter matching
config['psi_gate']['enable_multi_world'] = True

config['aegis'] = config.get('aegis', {})
config['aegis']['enabled'] = True
config['aegis']['max_generations'] = 5   # Increased from 3
config['aegis']['population_size'] = 30  # Increased from 20
config['aegis']['mutation_rate'] = 0.15

config['causal_prior'] = config.get('causal_prior', {})
config['causal_prior']['enabled'] = True

config['library_learning'] = config.get('library_learning', {})
config['library_learning']['enabled'] = True

config['eval'] = config.get('eval', {})
config['eval']['enabled'] = False

print('=== TOMAS v2.4.1 Configuration (Optimized) ===')
print(f'  CUDA enabled: {config["cuda"]["enabled"]}')
print(f'  ψ-Gate: {"ON" if config["psi_gate"]["enabled"] else "OFF"}')
print(f'    tolerance_decay_rate: {config["psi_gate"].get("tolerance_decay_rate", "N/A")}')
print(f'  AEGIS: {"ON" if config["aegis"]["enabled"] else "OFF"}')
print(f'    max_generations: {config["aegis"].get("max_generations", "N/A")}')
print(f'    population_size: {config["aegis"].get("population_size", "N/A")}')
print(f'  Library Learning: {"ON" if config["library_learning"]["enabled"] else "OFF"}')
'''

for cell in nb['cells']:
    if cell['id'] == 'cell-4':
        cell['source'] = new_source.split('\n')
        print(f"✅ Updated Cell 4 (config cell)")
        break

# Write back
with open('kaggle/notebook_template.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("✅ Notebook updated successfully")
