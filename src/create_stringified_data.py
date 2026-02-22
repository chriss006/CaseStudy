#!/usr/bin/env python3
"""Create stringified versions of processed data for training."""
import json
from pathlib import Path

def stringify_jsonl(input_path: str, output_path: str):
    """Convert JSONL with JSON objects to JSONL with stringified JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            sample = json.loads(line)
            
            # Create stringified version
            str_sample = {
                "id": sample["id"],
                "spec": json.dumps(sample["spec"], ensure_ascii=False),
                "output": json.dumps(sample["output"], ensure_ascii=False),
            }
            
            fout.write(json.dumps(str_sample, ensure_ascii=False) + '\n')

def main():
    base_dir = Path(__file__).parent.parent
    
    print("Converting train.jsonl to stringified format...")
    stringify_jsonl(
        str(base_dir / "data" / "processed" / "train.jsonl"),
        str(base_dir / "data" / "processed_str" / "train.jsonl")
    )
    
    print("Converting valid.jsonl to stringified format...")
    stringify_jsonl(
        str(base_dir / "data" / "processed" / "valid.jsonl"),
        str(base_dir / "data" / "processed_str" / "valid.jsonl")
    )
    
    print("✅ Stringified versions created successfully!")
    
    # Verify
    with open(base_dir / "data" / "processed_str" / "train.jsonl", 'r') as f:
        sample = json.loads(f.readline())
        print(f"\nVerification - Sample keys: {list(sample.keys())}")
        print(f"Spec type: {type(sample['spec'])}")
        print(f"Output type: {type(sample['output'])}")

if __name__ == "__main__":
    main()
