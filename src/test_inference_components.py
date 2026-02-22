#!/usr/bin/env python3
"""
Test script to validate inference module components without requiring trained model.
This tests the prompt generation and validation integration.
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from validate_architecture import validate_architecture


def test_prompt_creation():
    """Test prompt creation from specification."""
    print("Testing prompt creation...")
    
    # Load a test spec
    with open('test_specs/test_spec_00.json') as f:
        spec = json.load(f)
    
    # Create prompt manually (mimicking NoCGenerator.create_prompt)
    spec_str = json.dumps(spec, indent=2)
    
    prompt = f"""You are an expert NoC (Network-on-Chip) physical designer. Given a chip floorplan specification including initiators, targets, connectivity requirements, and blockage zones, generate optimal switch placements and routing paths.

Output ONLY valid JSON with this structure:
{{
  "switches": {{"s_0": {{"x": <int>, "y": <int>}}, ...}},
  "routing_paths": {{"r_0": ["<init>", "<switches>", ..., "<target>"], ...}}
}}

Constraints:
- Switches must be within floorplan bounds and avoid blockage zones
- All routes in connectivity must have valid paths
- Network must be deadlock-free (no cycles)
- Minimize total wirelength

Specification:
{spec_str}

Generated Architecture:"""
    
    print(f"✅ Prompt created successfully ({len(prompt)} characters)")
    print(f"   Spec has {len(spec.get('inits', {}))} initiators, {len(spec.get('targets', {}))} targets")
    print(f"   {len(spec.get('connectivity', {}))} routes to generate")
    
    return True


def test_validation_integration():
    """Test validation integration with ground truth."""
    print("\nTesting validation integration...")
    
    # Load test spec and ground truth
    with open('test_specs/test_spec_00.json') as f:
        spec = json.load(f)
    
    with open('test_specs/test_ground_truth_00.json') as f:
        ground_truth = json.load(f)
    
    # Validate ground truth
    is_valid, report = validate_architecture(spec, ground_truth)
    
    print(f"✅ Ground truth validation: {is_valid}")
    print(f"   Switches: {len(ground_truth.get('switches', {}))}")
    print(f"   Routes: {len(ground_truth.get('routing_paths', {}))}")
    
    if not is_valid:
        print(f"   ⚠️  Errors: {len(report['errors'])}")
        for error in report['errors'][:3]:
            print(f"      - {error}")
    
    return is_valid


def test_json_parsing():
    """Test JSON parsing logic."""
    print("\nTesting JSON parsing...")
    
    # Test cases
    test_cases = [
        # Valid JSON
        ('{"switches": {"s_0": {"x": 100, "y": 200}}, "routing_paths": {}}', True),
        # JSON with surrounding text
        ('Here is the output:\n{"switches": {}, "routing_paths": {}}\nDone!', True),
        # Invalid JSON
        ('This is not JSON', False),
        # Partial JSON
        ('{"switches": {"s_0": {', False),
    ]
    
    import re
    
    passed = 0
    for text, should_parse in test_cases:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                result = True
            except:
                result = False
        else:
            result = False
        
        if result == should_parse:
            passed += 1
    
    print(f"✅ JSON parsing tests: {passed}/{len(test_cases)} passed")
    
    return passed == len(test_cases)


def test_module_imports():
    """Test that all modules can be imported."""
    print("\nTesting module imports...")
    
    try:
        from generate_architecture import NoCGenerator
        print("✅ generate_architecture imports successfully")
        
        from batch_generate import batch_generate
        print("✅ batch_generate imports successfully")
        
        from extract_test_specs import extract_test_specs
        print("✅ extract_test_specs imports successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("INFERENCE MODULE COMPONENT TESTS")
    print("="*60)
    print()
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Prompt Creation", test_prompt_creation),
        ("Validation Integration", test_validation_integration),
        ("JSON Parsing", test_json_parsing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All component tests passed!")
        print("\nNext steps:")
        print("1. Complete model training")
        print("2. Run: python src/generate_architecture.py test_specs/test_spec_00.json --model <model_path>")
        return 0
    else:
        print("\n❌ Some tests failed - please fix before running inference")
        return 1


if __name__ == "__main__":
    sys.exit(main())
