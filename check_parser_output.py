#!/usr/bin/env python3
"""
Check what HTMLChunkParser actually extracts from failure files
"""

import sys
sys.path.insert(0, '/home/user/POSLLM')

from v61_extractor import HTMLChunkParser
import json
import re

def normalize(text):
    """Normalize text for comparison"""
    if not text:
        return ""
    return re.sub(r'\s+', '', text.upper())

def main():
    # Load ground truth
    with open('ground_truth_auto.json', 'r') as f:
        gt = json.load(f)

    # Files with failures
    failure_files = [
        '2606-POS-0057101_001_02_A4(27).html',
        '2606-POS-0094102_000_02_A4_CONSOLE.html'
    ]

    for html_file in failure_files:
        print("\n" + "="*80)
        print(f"Analyzing: {html_file}")
        print("="*80)

        # Parse with v61
        parser = HTMLChunkParser(file_path=html_file)

        print(f"\nTotal key-value pairs extracted: {len(parser.kv_pairs)}")

        # Get ground truth for this file
        gt_items = gt[html_file]

        # Find which ground truth items are missing
        missing = []
        for gt_item in gt_items:
            gt_spec = normalize(gt_item['spec_name'])
            gt_val = normalize(gt_item['value'])

            found = False
            for kv in parser.kv_pairs:
                kv_key = normalize(kv['key'])
                kv_val = normalize(kv['value'])

                # Simple substring match
                if gt_spec in kv_key or kv_key in gt_spec:
                    if gt_val in kv_val or kv_val in gt_val:
                        found = True
                        break

            if not found:
                missing.append(gt_item)

        print(f"\nMissing ground truth items: {len(missing)}")

        for item in missing[:10]:
            print(f"\n  Spec: {item['spec_name']}")
            print(f"  Value: {item['value']} {item['unit']}")
            print(f"  Raw: {item['raw_text'][:150]}")

        # Show a sample of what was extracted
        print(f"\n\nSample of extracted key-value pairs (first 10):")
        for i, kv in enumerate(parser.kv_pairs[:10], 1):
            print(f"\n{i}. Key: {kv['key'][:80]}")
            print(f"   Val: {kv['value'][:80]}")


if __name__ == "__main__":
    main()
