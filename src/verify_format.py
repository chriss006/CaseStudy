#!/usr/bin/env python3
"""Verify that output format matches reference format."""
import json

# Read reference sample
with open('data/processed_str/train.jsonl') as f:
    ref = json.loads(f.readline())

# Read current output sample
with open('data/processed_str_output_part_4/processed_part_4_test.jsonl') as f:
    output = json.loads(f.readline())

print("=" * 80)
print("FORMAT COMPARISON: Reference vs Output")
print("=" * 80)

print("\nFIELD STRUCTURE:")
print("Reference fields:", sorted(ref.keys()))
print("Output fields:   ", sorted(output.keys()))
fields_match = sorted(ref.keys()) == sorted(output.keys())
print("Fields Match:", "YES" if fields_match else "NO")

print("\nDETAILED COMPARISON:")
for field in ['id', 'spec', 'output']:
    ref_type = type(ref.get(field)).__name__
    out_type = type(output.get(field)).__name__
    match = "OK" if ref_type == out_type else "FAIL"
    print("{} {} | Ref: {} | Out: {}".format(match, field, ref_type, out_type))

print("\nSPEC FIELD VALIDATION:")
ref_spec = json.loads(ref['spec'])
out_spec = json.loads(output['spec'])
print("Reference spec keys:", sorted(ref_spec.keys()))
print("Output spec keys:   ", sorted(out_spec.keys()))
spec_match = sorted(ref_spec.keys()) == sorted(out_spec.keys())
print("Spec Match:", "YES" if spec_match else "NO")

print("\nOUTPUT FIELD VALIDATION:")
ref_output = json.loads(ref['output'])
out_output = json.loads(output['output'])
print("Reference output keys:", sorted(ref_output.keys()))
print("Output output keys:   ", sorted(out_output.keys()))
output_match = sorted(ref_output.keys()) == sorted(out_output.keys())
print("Output Match:", "YES" if output_match else "NO")

print("\nOUTPUT SUBFIELDS:")
print("Reference has switches:", 'switches' in ref_output)
print("Reference has routing_paths:", 'routing_paths' in ref_output)
print("Output has switches:", 'switches' in out_output)
print("Output has routing_paths:", 'routing_paths' in out_output)

print("\n" + "=" * 80)
if fields_match and spec_match and output_match:
    print("RESULT: Format matches reference exactly!")
else:
    print("RESULT: Format mismatch detected!")
print("=" * 80)
