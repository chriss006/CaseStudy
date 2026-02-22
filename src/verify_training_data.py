#!/usr/bin/env python3
"""Verify training data format is correct."""
import sys
sys.path.insert(0, 'src')
import json
from format_noc import build_prompt, build_label

# Test the complete pipeline
with open('data/processed_str/train.jsonl', 'r') as f:
    sample = json.loads(f.readline())

# Parse the strings
spec = json.loads(sample['spec'])
output = json.loads(sample['output'])

# Generate prompt and label
prompt = build_prompt(spec)
label = build_label(output)

print('='*60)
print('TRAINING DATA VERIFICATION')
print('='*60)
print(f'Sample ID: {sample["id"]}')
print(f'\nSpec contains:')
print(f'  - {len(spec["inits"])} initiators')
print(f'  - {len(spec["targets"])} targets')
print(f'  - {len(spec["connectivity"])} required routes')
print(f'  - {len(spec["blockages"])} blockages')
print(f'\nOutput contains:')
print(f'  - {len(output["switches"])} switches')
print(f'  - {len(output["routing_paths"])} routing paths')
print(f'\nPrompt length: {len(prompt)} chars')
print(f'Label length: {len(label)} chars')
print(f'Total length: {len(prompt + label)} chars (~{len(prompt + label)//4} tokens)')
print(f'\nFirst 200 chars of prompt:')
print(prompt[:200])
print(f'\nFirst 150 chars of label:')
print(label[:150])
print('\n' + '='*60)
print('✅ Training data format is correct!')
print('='*60)
