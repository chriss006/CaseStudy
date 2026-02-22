#!/usr/bin/env python3
"""Test validation module with sample data."""
import json
from validate_architecture import validate_architecture


def test_valid_architecture():
    """Test with a known valid architecture."""
    # Load a ground truth sample
    with open('data/processed/train.jsonl') as f:
        sample = json.loads(f.readline())
    
    spec = sample['spec']
    output = sample['output']
    
    is_valid, report = validate_architecture(spec, output)
    
    print("Testing ground truth architecture:")
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {len(report['errors'])}")
    if report['errors']:
        for error in report['errors'][:5]:  # Show first 5
            print(f"    - {error}")
    
    return is_valid


def test_invalid_architecture():
    """Test with intentionally broken architecture."""
    spec = {
        "inits": {"i_0": {"x": 100, "y": 100}},
        "targets": {"t_0": {"x": 900, "y": 900}},
        "connectivity": {"r_0": ["i_0", "t_0"]},
        "floorplan_dim": [1000, 1000],
        "blockages": {}
    }
    
    # Intentionally bad output
    output = {
        "switches": {
            "s_0": {"x": 5000, "y": 500}  # Outside bounds!
        },
        "routing_paths": {
            "r_0": ["i_0", "s_999", "t_0"]  # Non-existent switch!
        }
    }
    
    is_valid, report = validate_architecture(spec, output)
    
    print("\nTesting intentionally invalid architecture:")
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {len(report['errors'])}")
    for error in report['errors']:
        print(f"    - {error}")
    
    return not is_valid  # Should be invalid


if __name__ == "__main__":
    test1 = test_valid_architecture()
    test2 = test_invalid_architecture()
    
    if test1 and test2:
        print("\n✅ All validation tests passed!")
    else:
        print("\n❌ Some validation tests failed")
