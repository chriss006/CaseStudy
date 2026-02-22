#!/usr/bin/env python3
"""Validate all samples in the dataset."""
import json
from validate_architecture import validate_architecture


def validate_dataset(data_file):
    """Validate all samples in a JSONL file."""
    valid_count = 0
    invalid_count = 0
    all_errors = []
    
    with open(data_file) as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            is_valid, report = validate_architecture(sample['spec'], sample['output'])
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                all_errors.extend(report['errors'])
                if invalid_count <= 5:  # Show first 5 invalid
                    print(f"\nSample {i} INVALID:")
                    for error in report['errors'][:3]:
                        print(f"  - {error}")
    
    print(f"\n{'='*60}")
    print(f"Dataset: {data_file}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")
    print(f"Total: {valid_count + invalid_count}")
    if valid_count + invalid_count > 0:
        print(f"Validity Rate: {100 * valid_count / (valid_count + invalid_count):.1f}%")
    
    return valid_count, invalid_count, all_errors


if __name__ == "__main__":
    print("Validating training data...")
    train_valid, train_invalid, train_errors = validate_dataset('data/processed/train.jsonl')
    
    print("\n\nValidating validation data...")
    valid_valid, valid_invalid, valid_errors = validate_dataset('data/processed/valid.jsonl')
    
    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"Training: {train_valid}/{train_valid + train_invalid} valid")
    print(f"Validation: {valid_valid}/{valid_valid + valid_invalid} valid")
    print(f"Total: {train_valid + valid_valid}/{train_valid + train_invalid + valid_valid + valid_invalid} valid")
