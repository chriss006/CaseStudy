#!/usr/bin/env python3
"""Batch generation for multiple specifications."""
import json
from pathlib import Path
from generate_architecture import NoCGenerator
from validate_architecture import validate_architecture


def batch_generate(
    data_file: str,
    model_path: str,
    output_dir: str,
    num_samples: int = 10,
    temperature: float = 0.7
):
    """
    Generate architectures for multiple samples.
    
    Args:
        data_file: JSONL file with samples
        model_path: Path to trained model
        output_dir: Directory to save results
        num_samples: Number of samples to generate
        temperature: Sampling temperature
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load generator once
    generator = NoCGenerator(model_path)
    
    # Track statistics
    stats = {
        "total": 0,
        "parsed": 0,
        "valid": 0,
        "invalid": 0,
        "errors_by_type": {}
    }
    
    # Process samples
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            sample = json.loads(line)
            spec = sample['spec']
            
            print(f"\nGenerating {i+1}/{num_samples}...")
            
            result = generator.generate_and_validate(spec, temperature=temperature)
            
            # Update stats
            stats["total"] += 1
            if result["parsed_successfully"]:
                stats["parsed"] += 1
                if result["is_valid"]:
                    stats["valid"] += 1
                else:
                    stats["invalid"] += 1
                    # Track error types
                    if result["validation"] and result["validation"]["errors"]:
                        for error in result["validation"]["errors"]:
                            error_type = error.split(":")[0]
                            stats["errors_by_type"][error_type] = stats["errors_by_type"].get(error_type, 0) + 1
            
            # Save result
            output_file = output_path / f"result_{i:03d}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Print quick status
            if result["parsed_successfully"]:
                status = "✅ Valid" if result["is_valid"] else "❌ Invalid"
            else:
                status = "⚠️  Parse failed"
            print(f"  {status}")
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH GENERATION SUMMARY")
    print("="*60)
    print(f"Total: {stats['total']}")
    print(f"Parsed: {stats['parsed']} ({100*stats['parsed']/stats['total']:.1f}%)")
    print(f"Valid: {stats['valid']} ({100*stats['valid']/stats['total']:.1f}%)")
    print(f"Invalid: {stats['invalid']} ({100*stats['invalid']/stats['total']:.1f}%)")
    
    if stats["errors_by_type"]:
        print(f"\nError Distribution:")
        for error_type, count in sorted(stats["errors_by_type"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
    
    # Save summary
    with open(output_path / "summary.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Results saved to {output_dir}")
    print(f"✅ Summary saved to {output_dir}/summary.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate NoC architectures")
    parser.add_argument("data_file", help="JSONL file with samples")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--output-dir", default="outputs/generated", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    batch_generate(
        args.data_file,
        args.model,
        args.output_dir,
        args.num_samples,
        args.temperature
    )
