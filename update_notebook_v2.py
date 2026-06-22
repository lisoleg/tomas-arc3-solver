import json

# Read notebook
with open('kaggle/notebook_template.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find config cell by content (Cell 4 - "Load TOMAS Solver")
new_source = [
    "import sys\n",
    "import json\n",
    "from pathlib import Path\n",
    "\n",
    "# Add solver to path\n",
    "solver_path = '/kaggle/input/tomas-arc3-solver'\n",
    "if solver_path not in sys.path:\n",
    "    sys.path.insert(0, solver_path)\n",
    "\n",
    "from src.utils.config import load_config\n",
    "from src.solver.tomas_solver import TOMASSolver\n",
    "\n",
    "# Load config from file (v2.4.1 optimized settings)\n",
    "config_path = f'{solver_path}/config/default.yaml'\n",
    "config = load_config(config_path)\n",
    "\n",
    "# === Kaggle Environment Overrides ===\n",
    "config['kaggle'] = {\n",
    "    'input_dir': '/kaggle/input/arc-agi-3',\n",
    "    'output_dir': '/kaggle/working',\n",
    "}\n",
    "config['vl_api'] = {'available': False}  # No external API\n",
    "\n",
    "# === CUDA GPU Configuration (CRITICAL) ===\n",
    "config['cuda'] = config.get('cuda', {})\n",
    "config['cuda']['enabled'] = True\n",
    "config['cuda']['use_numba'] = True\n",
    "config['cuda']['max_vram_gb'] = 16  # T4 GPU\n",
    "config['cuda']['block_size'] = 256\n",
    "config['cuda']['grid_size'] = 64\n",
    "\n",
    "# === v2.4.1 Optimized Feature Configuration ===\n",
    "config['psi_gate'] = config.get('psi_gate', {})\n",
    "config['psi_gate']['enabled'] = True\n",
    "config['psi_gate']['tolerance_decay_rate'] = 0.03  # Stricter matching\n",
    "config['psi_gate']['enable_multi_world'] = True\n",
    "\n",
    "config['aegis'] = config.get('aegis', {})\n",
    "config['aegis']['enabled'] = True\n",
    "config['aegis']['max_generations'] = 5   # Increased from 3\n",
    "config['aegis']['population_size'] = 30  # Increased from 20\n",
    "config['aegis']['mutation_rate'] = 0.15\n",
    "\n",
    "config['causal_prior'] = config.get('causal_prior', {})\n",
    "config['causal_prior']['enabled'] = True\n",
    "\n",
    "config['library_learning'] = config.get('library_learning', {})\n",
    "config['library_learning']['enabled'] = True\n",
    "\n",
    "config['eval'] = config.get('eval', {})\n",
    "config['eval']['enabled'] = False\n",
    "\n",
    "print('=== TOMAS v2.4.1 Configuration (Optimized) ===')\n",
    "print(f'  CUDA enabled: {config[\"cuda\"][\"enabled\"]}')\n",
    "print(f'  ψ-Gate: {\"ON\" if config[\"psi_gate\"][\"enabled\"] else \"OFF\"}')\n",
    "print(f'    tolerance_decay_rate: {config[\"psi_gate\"].get(\"tolerance_decay_rate\", \"N/A\")}')\n",
    "print(f'  AEGIS: {\"ON\" if config[\"aegis\"][\"enabled\"] else \"OFF\"}')\n",
    "print(f'    max_generations: {config[\"aegis\"].get(\"max_generations\", \"N/A\")}')\n",
    "print(f'    population_size: {config[\"aegis\"].get(\"population_size\", \"N/A\")}')\n",
    "print(f'  Library Learning: {\"ON\" if config[\"library_learning\"][\"enabled\"] else \"OFF\"}')\n"
]

cell_index = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'Load TOMAS Solver' in ''.join(cell.get('source', [])):
        cell_index = i
        cell['source'] = new_source
        print(f"✅ Updated Cell {i+1} (config cell)")
        break

# Write back
with open('kaggle/notebook_template.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("✅ Notebook updated successfully")
