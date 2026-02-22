#!/usr/bin/env python3
"""Extract some test specifications from the dataset."""
import json
from pathlib import Path


def extract_test_specs(data_file: str, output_dir: str, num_specs: int = 5):
    """
    Extract specifications for testing.
    
    Args:
        data_file: Source JSONL file
        output_dir: Where to save spec files
        num_specs: Number of specs to extract
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {num_specs} test specifications from {data_file}...")
    
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= num_specs:
                break
            
            sample = json.loads(line)
            spec = sample['spec']
            ground_truth = sample['output']
            
            # Save spec
            spec_file = output_path / f"test_spec_{i:02d}.json"
            with open(spec_file, 'w') as out:
                json.dump(spec, out, indent=2)
            
            # Save ground truth separately for comparison
            gt_file = output_path / f"test_ground_truth_{i:02d}.json"
            with open(gt_file, 'w') as out:
                json.dump(ground_truth, out, indent=2)
            
            print(f"  ✅ Saved test case {i}: {spec_file.name}")
    
    print(f"\n✅ Extracted {num_specs} test specifications to {output_dir}/")
    print(f"   - Specification files: test_spec_XX.json")
    print(f"   - Ground truth files: test_ground_truth_XX.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract test specifications from dataset")
    parser.add_argument(
        "--data-file",
        default="data/processed/valid.jsonl",
        help="Source JSONL file (default: data/processed/valid.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        default="test_specs",
        help="Output directory (default: test_specs)"
    )
    parser.add_argument(
        "--num-specs",
        type=int,
        default=5,
        help="Number of specs to extract (default: 5)"
    )
    
    args = parser.parse_args()
    
    extract_test_specs(args.data_file, args.output_dir, args.num_specs)
