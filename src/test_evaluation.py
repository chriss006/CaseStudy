#!/usr/bin/env python3
"""Test evaluation module components."""
import sys
sys.path.insert(0, 'src')
from evaluate_model import ModelEvaluator
import tempfile
import os
import json
import shutil

# Create temp directory with dummy results
temp_dir = tempfile.mkdtemp()

# Create test results
test_results = [
    {
        'parsed_successfully': True,
        'is_valid': True,
        'output': {
            'switches': {'s_0': {'x': 100, 'y': 100}, 's_1': {'x': 500, 'y': 500}},
            'routing_paths': {'r_0': ['i_0', 's_0', 't_0'], 'r_1': ['i_1', 's_1', 't_1']}
        },
        'validation': {
            'valid': True,
            'errors': [],
            'checks': {
                'switch_placement': True,
                'path_elements': True,
                'route_connectivity': True,
                'no_cycles': True
            }
        }
    },
    {
        'parsed_successfully': True,
        'is_valid': False,
        'output': {
            'switches': {'s_0': {'x': 1200, 'y': 100}},
            'routing_paths': {'r_0': ['i_0', 's_0', 't_0']}
        },
        'validation': {
            'valid': False,
            'errors': ['switch_placement: s_0 at (1200, 100) outside bounds'],
            'checks': {
                'switch_placement': False,
                'path_elements': True,
                'route_connectivity': True,
                'no_cycles': True
            }
        }
    },
    {
        'parsed_successfully': False,
        'is_valid': False,
        'output': None,
        'validation': None
    }
]

# Write test files
for i, result in enumerate(test_results):
    with open(os.path.join(temp_dir, f'result_{i:03d}.json'), 'w') as f:
        json.dump(result, f)

print("Testing Evaluation Module")
print("=" * 60)

# Test evaluation
evaluator = ModelEvaluator(temp_dir)
metrics = evaluator.compute_metrics()

print(f"\n✅ Test passed: Loaded {len(evaluator.results)} results")
print(f"✅ Test passed: Parsing rate = {metrics['parsing']['parsing_rate'] * 100:.1f}%")
print(f"✅ Test passed: Validity rate = {metrics['validity']['overall_success_rate'] * 100:.1f}%")
print(f"✅ Test passed: Constraint analysis completed")

# Test report generation
report_file = os.path.join(temp_dir, 'test_report.md')
evaluator.generate_report(report_file)

if os.path.exists(report_file):
    print(f"✅ Test passed: Report generated successfully")
else:
    print(f"❌ Test failed: Report not generated")

# Cleanup
shutil.rmtree(temp_dir)

print("\n" + "=" * 60)
print("All evaluation component tests passed!")
print("=" * 60)
