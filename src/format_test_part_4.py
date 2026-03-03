#!/usr/bin/env python3
"""Format test_part_4.jsonl data to match processed_str data format with validation."""
import json
from pathlib import Path
from typing import Dict, Any, List
import sys


def validate_spec_format(spec: Dict[str, Any]) -> bool:
    """Validate that spec contains required fields."""
    required_keys = {'inits', 'targets', 'connectivity', 'floorplan_dim', 'blockages'}
    return isinstance(spec, dict) and required_keys.issubset(spec.keys())


def validate_switches_format(switches: Dict[str, Any]) -> bool:
    """Validate that switches is a dictionary with switch coordinates."""
    if not isinstance(switches, dict):
        return False
    # Each switch should have x, y coordinates
    return all(isinstance(v, dict) and 'x' in v and 'y' in v 
               for v in switches.values())


def validate_routing_paths_format(routing_paths: Dict[str, Any]) -> bool:
    """Validate that routing_paths is a dictionary with path lists."""
    if not isinstance(routing_paths, dict):
        return False
    # Each route should be a list
    return all(isinstance(v, list) for v in routing_paths.values())


def format_test_data(input_path: str, output_path: str) -> tuple[int, int, List[str]]:
    """
    Format test_part_4.jsonl to match processed_str format.
    
    Args:
        input_path: Path to test_part_4.jsonl
        output_path: Path to processed_part_4_test.jsonl
    
    Returns:
        Tuple of (total_records, valid_records, error_messages)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    total_records = 0
    valid_records = 0
    error_messages = []
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, start=1):
            try:
                total_records += 1
                sample = json.loads(line)
                
                # Validate input format - check for required fields
                required_fields = {'spec', 'switches', 'routing_paths'}
                if not required_fields.issubset(sample.keys()):
                    error_messages.append(
                        f"Line {line_num}: Missing required fields. Expected {required_fields}, got {set(sample.keys())}"
                    )
                    continue
                
                spec = sample['spec']
                switches = sample['switches']
                routing_paths = sample['routing_paths']
                
                # Validate spec format
                if not validate_spec_format(spec):
                    error_messages.append(
                        f"Line {line_num}: Invalid spec format - missing required fields"
                    )
                    continue
                
                # Validate switches format
                if not validate_switches_format(switches):
                    error_messages.append(
                        f"Line {line_num}: Invalid switches format"
                    )
                    continue
                
                # Validate routing_paths format
                if not validate_routing_paths_format(routing_paths):
                    error_messages.append(
                        f"Line {line_num}: Invalid routing_paths format"
                    )
                    continue
                
                # Generate ID based on line number
                record_id = f"test_part_4_{line_num-1:03d}"
                
                # Create output field containing switches and routing_paths (matching reference format)
                output_field = {
                    "switches": switches,
                    "routing_paths": routing_paths
                }
                
                # Create output format: spec and output as stringified JSON
                output_sample = {
                    "id": record_id,
                    "spec": json.dumps(spec, ensure_ascii=False),
                    "output": json.dumps(output_field, ensure_ascii=False)
                }
                
                # Write to output
                fout.write(json.dumps(output_sample, ensure_ascii=False) + '\n')
                valid_records += 1
                
            except json.JSONDecodeError as e:
                error_messages.append(f"Line {line_num}: JSON decode error - {str(e)}")
            except Exception as e:
                error_messages.append(f"Line {line_num}: Unexpected error - {str(e)}")
    
    return total_records, valid_records, error_messages


def validate_output_file(output_path: str) -> tuple[bool, List[str]]:
    """
    Validate the output file format and syntax.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    validation_errors = []
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    sample = json.loads(line)
                    
                    # Check required fields
                    required_fields = {'id', 'spec', 'output'}
                    if not required_fields.issubset(sample.keys()):
                        validation_errors.append(
                            f"Line {line_num}: Missing required fields. "
                            f"Expected {required_fields}, got {set(sample.keys())}"
                        )
                        continue
                    
                    # Check id format
                    if not isinstance(sample['id'], str):
                        validation_errors.append(
                            f"Line {line_num}: 'id' should be string, got {type(sample['id'])}"
                        )
                    
                    # Check spec is string and can be parsed
                    if not isinstance(sample['spec'], str):
                        validation_errors.append(
                            f"Line {line_num}: 'spec' should be JSON string, got {type(sample['spec'])}"
                        )
                    else:
                        try:
                            spec_obj = json.loads(sample['spec'])
                            if not validate_spec_format(spec_obj):
                                validation_errors.append(
                                    f"Line {line_num}: spec JSON is missing required fields"
                                )
                        except json.JSONDecodeError as e:
                            validation_errors.append(
                                f"Line {line_num}: spec JSON decode error - {str(e)}"
                            )
                    
                    # Check output is string and can be parsed
                    if not isinstance(sample['output'], str):
                        validation_errors.append(
                            f"Line {line_num}: 'output' should be JSON string, got {type(sample['output'])}"
                        )
                    else:
                        try:
                            output_obj = json.loads(sample['output'])
                            # Check output contains switches and routing_paths
                            if 'switches' not in output_obj or 'routing_paths' not in output_obj:
                                validation_errors.append(
                                    f"Line {line_num}: output should contain 'switches' and 'routing_paths'"
                                )
                            else:
                                if not validate_switches_format(output_obj['switches']):
                                    validation_errors.append(
                                        f"Line {line_num}: switches format is invalid"
                                    )
                                if not validate_routing_paths_format(output_obj['routing_paths']):
                                    validation_errors.append(
                                        f"Line {line_num}: routing_paths format is invalid"
                                    )
                        except json.JSONDecodeError as e:
                            validation_errors.append(
                                f"Line {line_num}: output JSON decode error - {str(e)}"
                            )
                
                except json.JSONDecodeError as e:
                    validation_errors.append(f"Line {line_num}: JSON decode error - {str(e)}")
    
    except FileNotFoundError:
        return False, [f"Output file not found: {output_path}"]
    except Exception as e:
        return False, [f"Unexpected error reading output file: {str(e)}"]
    
    is_valid = len(validation_errors) == 0
    return is_valid, validation_errors


def main():
    base_dir = Path(__file__).parent.parent
    
    input_path = str(base_dir / "data" / "raw_part_4" / "test_part_4.jsonl")
    output_path = str(base_dir / "data" / "processed_str_output_part_4" / "processed_part_4_test.jsonl")
    
    print("=" * 80)
    print("Formatting test_part_4.jsonl to processed_str format...")
    print("=" * 80)
    
    # Check if input file exists
    if not Path(input_path).exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Format the data
    total, valid, format_errors = format_test_data(input_path, output_path)
    
    print(f"\n📊 Formatting Statistics:")
    print(f"  Total records processed: {total}")
    print(f"  Valid records: {valid}")
    print(f"  Invalid records: {total - valid}")
    
    if format_errors:
        print(f"\n⚠️  Formatting Warnings/Errors ({len(format_errors)}):")
        for error in format_errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(format_errors) > 10:
            print(f"  ... and {len(format_errors) - 10} more errors")
    
    print(f"\n📁 Output file created: {output_path}")
    
    # Validate the output
    print("\n" + "=" * 80)
    print("Validating output file format and syntax...")
    print("=" * 80)
    
    is_valid, validation_errors = validate_output_file(output_path)
    
    if is_valid:
        print("\n✅ Validation PASSED!")
        print(f"   All {valid} records have correct format and valid JSON.")
    else:
        print(f"\n❌ Validation FAILED!")
        print(f"   Found {len(validation_errors)} validation errors:")
        for error in validation_errors[:10]:
            print(f"  - {error}")
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more errors")
        sys.exit(1)
    
    # Show sample of output
    print("\n" + "=" * 80)
    print("Sample from output file (first record):")
    print("=" * 80)
    with open(output_path, 'r') as f:
        sample_line = f.readline()
        sample = json.loads(sample_line)
        print(f"ID: {sample['id']}")
        print(f"Spec (first 100 chars): {sample['spec'][:100]}...")
        output_obj = json.loads(sample['output'])
        print(f"Output contains: {list(output_obj.keys())}")
        print(f"Switches count: {len(output_obj['switches'])}")
        print(f"Routing paths count: {len(output_obj['routing_paths'])}")
    
    print("\n" + "=" * 80)
    print("✨ Transformation and validation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
