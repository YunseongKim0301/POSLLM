#!/usr/bin/env python3
"""
Analyze failure cases from evaluation results
"""

import json
import os

def main():
    base_dir = "/home/user/POSLLM"
    results_path = os.path.join(base_dir, "evaluation_results.json")

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("\n" + "="*80)
    print("DETAILED FAILURE ANALYSIS")
    print("="*80)

    for version in ['v61', 'v70']:
        print(f"\n{version.upper()} Failures:")
        print("-"*80)

        failures = data[f'{version}_failures']

        print(f"\nKey Failure Examples ({len(failures['key_failure_examples'])} shown):")
        for i, f in enumerate(failures['key_failure_examples'], 1):
            print(f"\n{i}. File: {f['filename']}")
            print(f"   Spec Name: {f['spec_name']}")
            print(f"   GT Value: {f['gt_value']}")
            print(f"   Extracted: {f['extracted_value']}")

        print(f"\nValue Failure Examples ({len(failures['value_failure_examples'])} shown):")
        for i, f in enumerate(failures['value_failure_examples'], 1):
            print(f"\n{i}. File: {f['filename']}")
            print(f"   Spec Name: {f['spec_name']}")
            print(f"   GT Value: {f['gt_value']}")
            print(f"   Extracted: {f['extracted_value']}")

    # Identify specific failing specs
    print("\n" + "="*80)
    print("SPECIFIC FAILING SPECIFICATIONS")
    print("="*80)

    v61_results = data['v61']
    for file_result in v61_results['file_results']:
        filename = file_result['filename']
        failures = [d for d in file_result.get('details', [])
                   if not (d['matched_key'] and d['matched_value'])]

        if failures:
            print(f"\n{filename}:")
            for f in failures:
                print(f"  - {f['spec_name']}: GT={f['gt_value']}, Extracted={f.get('extracted_value')}")

if __name__ == "__main__":
    main()
