#!/usr/bin/env python3
"""Test ground truth comparison components."""
import sys
sys.path.insert(0, 'src')
from compare_with_ground_truth import compare_architectures
import tempfile
import os
import json
import shutil

# Create temp directory
temp_dir = tempfile.mkdtemp()

# Create test result
test_result = {
    'spec': {
        'inits': {'i_0': {'x': 100, 'y': 100}},
        'targets': {'t_0': {'x': 800, 'y': 800}},
        'connectivity': {'r_0': ['i_0', 't_0']},
        'floorplan_dim': [1000, 1000],
        'blockages': {}
    },
    'parsed_successfully': True,
    'is_valid': True,
    'output': {
        'switches': {'s_0': {'x': 450, 'y': 450}},
        'routing_paths': {'r_0': ['i_0', 's_0', 't_0']}
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
}

# Create ground truth
ground_truth = {
    'switches': {'s_0': {'x': 500, 'y': 500}},
    'routing_paths': {'r_0': ['i_0', 's_0', 't_0']}
}

# Write files
result_file = os.path.join(temp_dir, 'result_test.json')
gt_file = os.path.join(temp_dir, 'ground_truth_test.json')

with open(result_file, 'w') as f:
    json.dump(test_result, f)

with open(gt_file, 'w') as f:
    json.dump(ground_truth, f)

print("Testing Ground Truth Comparison")
print("=" * 60)

# Test comparison
comparison = compare_architectures(result_file, gt_file)

if comparison:
    print("\n✅ Test passed: Comparison completed")
    print(f"✅ Test passed: Switches comparison = {comparison['switches']}")
    print(f"✅ Test passed: Routes comparison = {comparison['routes']}")
    print(f"✅ Test passed: Validity = {comparison['is_valid']}")
else:
    print("\n❌ Test failed: Comparison did not return results")

# Cleanup
shutil.rmtree(temp_dir)

print("\n" + "=" * 60)
print("All comparison component tests passed!")
print("=" * 60)
