"""
TOMAS Visualization API

REST API endpoints for dashboard visualizations:
- GET  /api/viz/search-tree/<task_id>      kappa-Snap search tree data
- GET  /api/viz/fiber-verification/<task_id>  GaussEx fiber verification data
- GET  /api/viz/pruning-stats/<task_id>     8-strategy pruning statistics
- GET  /api/viz/task-detail/<task_id>       Complete task visualization data
- GET  /api/viz/history                     Task history list (with filters)
- GET  /api/viz/history/<task_id>           Task history detail
- GET  /api/viz/benchmark                   Performance benchmark data
"""

import json
import os
from flask import Blueprint, jsonify, request

from services.viz_data import (
    generate_search_tree,
    generate_fiber_verification,
    generate_pruning_stats,
    generate_task_history,
    generate_task_detail,
)

viz_bp = Blueprint('viz', __name__)


@viz_bp.route('/search-tree/<task_id>', methods=['GET'])
def get_search_tree(task_id: str):
    """Get kappa-Snap search tree data for D3.js visualization.

    Returns hierarchical tree structure showing:
    - Root: demo pairs
    - Levels 1-3: candidate programs at each depth
    - Phase A pass/fail labels
    - Phase B verify pass/fail labels
    """
    try:
        depth = request.args.get('depth', 3, type=int)
        data = generate_search_tree(task_id, depth=depth)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@viz_bp.route('/fiber-verification/<task_id>', methods=['GET'])
def get_fiber_verification(task_id: str):
    """Get GaussEx fiber verification data.

    Returns fiber intersection verification results:
    - Demo pairs with fiber counts and CRC32 hashes
    - Candidate programs with per-demo verification status
    - Overall verification rate
    """
    try:
        data = generate_fiber_verification(task_id)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@viz_bp.route('/pruning-stats/<task_id>', methods=['GET'])
def get_pruning_stats(task_id: str):
    """Get 8-strategy pruning pipeline statistics.

    Returns data for Recharts bar chart:
    - Per-strategy pruning counts and rates
    - Cumulative pruning rate
    - Before/after candidate counts
    """
    try:
        data = generate_pruning_stats(task_id)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@viz_bp.route('/task-detail/<task_id>', methods=['GET'])
def get_task_detail(task_id: str):
    """Get complete task visualization data (all three visualizations).

    Convenience endpoint that returns search tree, fiber verification,
    and pruning stats in a single response.
    """
    try:
        data = generate_task_detail(task_id)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@viz_bp.route('/history', methods=['GET'])
def get_viz_history():
    """Get task history list with optional filtering.

    Query params:
    - limit: Maximum results (default 20)
    - mode: Filter by inference mode
    - status: Filter by status
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        mode = request.args.get('mode')
        status = request.args.get('status')

        # Try to load real history first
        from config import HISTORY_FILE
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []

        # If no real history, generate sample data
        if not history:
            history = generate_task_history(limit=limit)

        # Apply filters
        if mode:
            history = [h for h in history if h.get('mode') == mode]
        if status:
            history = [h for h in history if h.get('status') == status]

        # Limit results
        history = history[:limit]

        return jsonify({
            'history': history,
            'total': len(history),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@viz_bp.route('/history/<task_id>', methods=['GET'])
def get_viz_history_detail(task_id: str):
    """Get detailed task history entry with all visualization data."""
    try:
        # Try real history first
        from config import HISTORY_FILE
        real_entry = None
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                try:
                    history = json.load(f)
                    for h in history:
                        if h.get('task_id') == task_id:
                            real_entry = h
                            break
                except json.JSONDecodeError:
                    pass

        # Generate visualization data
        detail = generate_task_detail(task_id)
        if real_entry:
            detail['task_info'] = real_entry

        return jsonify(detail)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@viz_bp.route('/history/<task_id>', methods=['DELETE'])
def delete_viz_history(task_id: str):
    """Delete a task history entry."""
    try:
        from config import HISTORY_FILE
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            history = [h for h in history if h.get('task_id') != task_id]
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            return jsonify({'status': 'deleted', 'task_id': task_id})
        return jsonify({'error': 'History file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@viz_bp.route('/benchmark', methods=['GET'])
def get_benchmark():
    """Get performance benchmark comparison data.

    Loads real benchmark results from benchmarks/benchmark_raw.json if available.
    Falls back to generated sample data when no benchmark has been run.

    Returns comparison data for:
    - psi_gate enabled vs disabled
    - AEGIS evolution vs normal search
    """
    try:
        # Try to load real benchmark data
        import os
        benchmark_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..',
            'benchmarks', 'benchmark_raw.json'
        )
        benchmark_path = os.path.abspath(benchmark_path)

        if os.path.exists(benchmark_path):
            with open(benchmark_path, 'r') as f:
                raw = json.load(f)

            # Aggregate real psi_gate results
            psi_results = raw.get('psi_gate', [])
            aegis_results = raw.get('aegis', [])

            def aggregate_real(results, config_name):
                matching = [r for r in results if r.get('config') == config_name and r.get('status') == 'completed']
                if not matching:
                    return None
                correct = sum(1 for r in matching if r.get('correct'))
                return {
                    'accuracy': round(correct / len(matching), 3),
                    'avg_search_time': round(sum(r.get('total_time_sec', 0) for r in matching) / len(matching), 2),
                    'avg_candidates': round(sum(r.get('total_candidates', 0) for r in matching) / len(matching), 1),
                    'avg_confidence': round(sum(r.get('top_confidence', 0) for r in matching) / len(matching), 4),
                    'avg_mdl': round(sum(r.get('top_mdl', 0) for r in matching) / len(matching), 1),
                    'task_count': len(matching),
                }

            psi_disabled = aggregate_real(psi_results, 'psi_gate_disabled')
            psi_enabled = aggregate_real(psi_results, 'psi_gate_enabled')
            aegis_disabled = aggregate_real(aegis_results, 'aegis_disabled')
            aegis_enabled = aggregate_real(aegis_results, 'aegis_enabled')

            if psi_disabled and psi_enabled:
                psi_gate_data = {'enabled': psi_enabled, 'disabled': psi_disabled}
            else:
                psi_gate_data = None

            if aegis_disabled and aegis_enabled:
                aegis_data = {
                    'aegis': {
                        'success_rate': aegis_enabled.get('accuracy', 0),
                        'avg_iterations': 0,
                        'avg_programs_evolved': 0,
                        'convergence_time': aegis_enabled.get('avg_search_time', 0),
                    },
                    'normal': {
                        'success_rate': aegis_disabled.get('accuracy', 0),
                        'avg_iterations': 1,
                        'avg_programs_evolved': 0,
                        'convergence_time': aegis_disabled.get('avg_search_time', 0),
                    },
                }
            else:
                aegis_data = None

            if psi_gate_data or aegis_data:
                return jsonify({
                    'psi_gate': psi_gate_data or {},
                    'aegis': aegis_data or {},
                    'source': 'real_benchmark',
                    'timestamp': raw.get('timestamp', ''),
                })

        # Fall back to sample data
        import random
        rng = random.Random(42)

        psi_gate_data = {
            'enabled': {
                'accuracy': round(rng.uniform(0.75, 0.92), 3),
                'avg_search_time': round(rng.uniform(15, 35), 2),
                'avg_candidates': rng.randint(200, 400),
                'avg_confidence': round(rng.uniform(0.7, 0.9), 3),
                'avg_mdl': rng.randint(12, 25),
            },
            'disabled': {
                'accuracy': round(rng.uniform(0.55, 0.75), 3),
                'avg_search_time': round(rng.uniform(10, 25), 2),
                'avg_candidates': rng.randint(150, 300),
                'avg_confidence': round(rng.uniform(0.5, 0.7), 3),
                'avg_mdl': rng.randint(15, 30),
            },
        }

        aegis_data = {
            'aegis': {
                'success_rate': round(rng.uniform(0.8, 0.95), 3),
                'avg_iterations': rng.randint(3, 8),
                'avg_programs_evolved': rng.randint(50, 150),
                'convergence_time': round(rng.uniform(20, 50), 2),
            },
            'normal': {
                'success_rate': round(rng.uniform(0.6, 0.8), 3),
                'avg_iterations': 1,
                'avg_programs_evolved': 0,
                'convergence_time': round(rng.uniform(10, 30), 2),
            },
        }

        return jsonify({
            'psi_gate': psi_gate_data,
            'aegis': aegis_data,
            'source': 'sample_data',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
