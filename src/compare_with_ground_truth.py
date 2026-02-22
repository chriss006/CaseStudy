#!/usr/bin/env python3
"""Compare generated architectures with ground truth."""
import json
from pathlib import Path
from typing import Dict


def compare_architectures(result_file: str, ground_truth_file: str):
    """
    Compare generated architecture with ground truth.
    
    Args:
        result_file: Generated result JSON
        ground_truth_file: Ground truth output JSON
    """
    with open(result_file) as f:
        result = json.load(f)
    
    with open(ground_truth_file) as f:
        ground_truth = json.load(f)
    
    print(f"\n{'='*60}")
    print("COMPARISON: Generated vs Ground Truth")
    print(f"{'='*60}")
    
    # Get outputs
    generated = result.get('output', {})
    
    # Compare switches
    gen_switches = generated.get('switches', {})
    gt_switches = ground_truth.get('switches', {})
    
    print(f"\nSwitches:")
    print(f"  Generated: {len(gen_switches)}")
    print(f"  Ground truth: {len(gt_switches)}")
    print(f"  Difference: {len(gen_switches) - len(gt_switches):+d}")
    
    # Compare routes
    gen_routes = generated.get('routing_paths', {})
    gt_routes = ground_truth.get('routing_paths', {})
    
    print(f"\nRouting Paths:")
    print(f"  Generated: {len(gen_routes)}")
    print(f"  Ground truth: {len(gt_routes)}")
    print(f"  Difference: {len(gen_routes) - len(gt_routes):+d}")
    
    # Route completeness check
    missing_routes = set(gt_routes.keys()) - set(gen_routes.keys())
    extra_routes = set(gen_routes.keys()) - set(gt_routes.keys())
    
    if missing_routes:
        print(f"\n  Missing routes: {sorted(missing_routes)}")
    if extra_routes:
        print(f"  Extra routes: {sorted(extra_routes)}")
    
    # Validation status
    print(f"\nValidation:")
    print(f"  Generated is valid: {result.get('is_valid', False)}")
    
    if not result.get('is_valid'):
        validation = result.get('validation', {})
        print(f"  Errors: {len(validation.get('errors', []))}")
        if validation.get('errors'):
            print(f"\n  Error details:")
            for error in validation['errors'][:5]:  # Show first 5
                print(f"    - {error}")
            if len(validation['errors']) > 5:
                print(f"    ... and {len(validation['errors']) - 5} more")
    
    # Quality comparison
    print(f"\nQuality Metrics:")
    
    # Calculate total wirelength (Manhattan distance)
    def calculate_wirelength(switches: Dict, routes: Dict, spec: Dict) -> float:
        """Calculate total Manhattan wirelength."""
        total = 0
        all_nodes = {**spec.get('inits', {}), **spec.get('targets', {}), **switches}
        
        for route_id, path in routes.items():
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i+1]
                if node1 in all_nodes and node2 in all_nodes:
                    x1, y1 = all_nodes[node1]['x'], all_nodes[node1]['y']
                    x2, y2 = all_nodes[node2]['x'], all_nodes[node2]['y']
                    total += abs(x2 - x1) + abs(y2 - y1)
        
        return total
    
    spec = result.get('spec', {})
    
    if gen_switches and gen_routes and spec:
        gen_wirelength = calculate_wirelength(gen_switches, gen_routes, spec)
        gt_wirelength = calculate_wirelength(gt_switches, gt_routes, spec)
        
        print(f"  Generated wirelength: {gen_wirelength:.1f}")
        print(f"  Ground truth wirelength: {gt_wirelength:.1f}")
        
        if gt_wirelength > 0:
            ratio = gen_wirelength / gt_wirelength
            print(f"  Ratio (gen/gt): {ratio:.2f}x")
            
            if ratio < 1.2:
                print(f"  Assessment: ✅ Excellent (within 20% of optimal)")
            elif ratio < 1.5:
                print(f"  Assessment: ✅ Good (within 50% of optimal)")
            elif ratio < 2.0:
                print(f"  Assessment: ⚠️ Acceptable (within 2x of optimal)")
            else:
                print(f"  Assessment: ❌ Needs improvement (>2x optimal)")
    
    # Route length comparison
    print(f"\nRoute Lengths:")
    
    def avg_route_length(routes: Dict) -> float:
        """Calculate average route length."""
        if not routes:
            return 0
        lengths = [len(path) for path in routes.values()]
        return sum(lengths) / len(lengths)
    
    gen_avg_len = avg_route_length(gen_routes)
    gt_avg_len = avg_route_length(gt_routes)
    
    print(f"  Generated avg: {gen_avg_len:.2f} hops")
    print(f"  Ground truth avg: {gt_avg_len:.2f} hops")
    
    if gt_avg_len > 0:
        print(f"  Difference: {gen_avg_len - gt_avg_len:+.2f} hops")
    
    print(f"\n{'='*60}\n")
    
    return {
        'switches': {
            'generated': len(gen_switches),
            'ground_truth': len(gt_switches),
            'difference': len(gen_switches) - len(gt_switches)
        },
        'routes': {
            'generated': len(gen_routes),
            'ground_truth': len(gt_routes),
            'missing': list(missing_routes),
            'extra': list(extra_routes)
        },
        'is_valid': result.get('is_valid', False),
        'avg_route_length': {
            'generated': gen_avg_len,
            'ground_truth': gt_avg_len
        }
    }


def batch_compare(results_dir: str, ground_truth_dir: str, output_file: str = "comparison_report.md"):
    """
    Compare multiple generated results with ground truth.
    
    Args:
        results_dir: Directory with generated results
        ground_truth_dir: Directory with ground truth files
        output_file: Output markdown report
    """
    results_path = Path(results_dir)
    gt_path = Path(ground_truth_dir)
    
    result_files = sorted(results_path.glob("result_*.json"))
    comparisons = []
    
    print(f"\nComparing {len(result_files)} results...\n")
    
    for result_file in result_files:
        # Extract index from filename (e.g., result_000.json -> 00)
        idx = result_file.stem.split('_')[1]
        gt_file = gt_path / f"test_ground_truth_{idx}.json"
        
        if not gt_file.exists():
            print(f"⚠️ No ground truth for {result_file.name}")
            continue
        
        print(f"Comparing {result_file.name}...")
        comparison = compare_architectures(str(result_file), str(gt_file))
        comparisons.append(comparison)
    
    # Generate summary report
    if comparisons:
        _generate_comparison_report(comparisons, output_file)
    else:
        print("No comparisons performed")


def _generate_comparison_report(comparisons: list, output_file: str):
    """Generate markdown report from comparisons."""
    
    total = len(comparisons)
    valid_count = sum(1 for c in comparisons if c['is_valid'])
    
    avg_switch_diff = sum(c['switches']['difference'] for c in comparisons) / total
    avg_route_diff = sum(c['routes']['generated'] - c['routes']['ground_truth'] 
                         for c in comparisons) / total
    
    report = f"""# Ground Truth Comparison Report

## Summary

**Total Comparisons**: {total}

**Valid Architectures**: {valid_count} ({100*valid_count/total:.1f}%)

---

## Structural Differences

### Switches

- **Average difference from ground truth**: {avg_switch_diff:+.2f} switches
- **Range**: {min(c['switches']['difference'] for c in comparisons)} to {max(c['switches']['difference'] for c in comparisons)}

### Routes

- **Average difference from ground truth**: {avg_route_diff:+.2f} routes
- **Missing routes**: {sum(len(c['routes']['missing']) for c in comparisons)} total
- **Extra routes**: {sum(len(c['routes']['extra']) for c in comparisons)} total

---

## Route Length Analysis

**Average Route Length Comparison**:

- Generated: {sum(c['avg_route_length']['generated'] for c in comparisons)/total:.2f} hops
- Ground Truth: {sum(c['avg_route_length']['ground_truth'] for c in comparisons)/total:.2f} hops

---

## Interpretation

"""
    
    if valid_count / total > 0.7:
        report += "- ✅ **High validity rate**: Most generated architectures satisfy constraints\n"
    
    if abs(avg_switch_diff) < 1:
        report += "- ✅ **Good switch count**: Generated architectures use similar number of switches\n"
    elif abs(avg_switch_diff) > 3:
        report += "- ⚠️ **Switch count variation**: Model tends to use different number of switches\n"
    
    if abs(avg_route_diff) < 0.5:
        report += "- ✅ **Complete routing**: Model generates all required routes\n"
    
    report += "\n---\n\n*Generated by Ground Truth Comparison Tool*\n"
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Comparison report saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare generated with ground truth")
    parser.add_argument("result_file", nargs='?', help="Generated result JSON")
    parser.add_argument("ground_truth_file", nargs='?', help="Ground truth JSON")
    parser.add_argument("--batch", action="store_true", help="Batch comparison mode")
    parser.add_argument("--results-dir", help="Directory with generated results (batch mode)")
    parser.add_argument("--gt-dir", help="Directory with ground truth files (batch mode)")
    parser.add_argument("--output", default="comparison_report.md", help="Output report")
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.results_dir or not args.gt_dir:
            parser.error("--batch requires --results-dir and --gt-dir")
        batch_compare(args.results_dir, args.gt_dir, args.output)
    else:
        if not args.result_file or not args.ground_truth_file:
            parser.error("Single comparison requires result_file and ground_truth_file")
        compare_architectures(args.result_file, args.ground_truth_file)
