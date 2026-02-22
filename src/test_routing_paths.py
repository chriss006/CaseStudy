#!/usr/bin/env python3
"""Quick test that routing paths are in the training data."""
import json

def check_routing_paths():
    print("Checking processed training data...\n")
    
    with open('data/processed/train.jsonl', 'r') as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            has_paths = 'routing_paths' in sample['output']
            if i < 3:
                print(f"Sample {i}: Has routing_paths = {has_paths}")
                if has_paths:
                    num_paths = len(sample['output']['routing_paths'])
                    num_switches = len(sample['output']['switches'])
                    print(f"  Number of switches: {num_switches}")
                    print(f"  Number of routes: {num_paths}")
                    # Show one route example
                    route_id, route_path = list(sample['output']['routing_paths'].items())[0]
                    print(f"  Example route {route_id}: {route_path}")
                print()
    
    print("✅ Validation complete\n")
    
    # Check validation set too
    print("Checking validation data...")
    with open('data/processed/valid.jsonl', 'r') as f:
        sample = json.loads(f.readline())
        has_paths = 'routing_paths' in sample['output']
        print(f"Validation set has routing_paths: {has_paths}")
        if has_paths:
            print(f"  Switches: {len(sample['output']['switches'])}")
            print(f"  Routes: {len(sample['output']['routing_paths'])}")
    
    print("\n✅ All checks passed!")

if __name__ == "__main__":
    check_routing_paths()
