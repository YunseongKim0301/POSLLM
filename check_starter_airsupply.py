#!/usr/bin/env python3
"""
Check if Starter and Air supplydevices are actually extracted
"""

import sys
sys.path.insert(0, '/home/user/POSLLM')

from v61_extractor import HTMLChunkParser
import json

# Load ground truth
with open('ground_truth_auto.json', 'r') as f:
    gt = json.load(f)

# Parse file
parser = HTMLChunkParser(file_path='2606-POS-0057101_001_02_A4(27).html')

# Find Starter and Air supply
target_specs = ['Starter', 'Air supplydevices']

for spec in target_specs:
    print(f"\nSearching for: '{spec}'")
    print("="*80)

    # Ground truth
    gt_items = [item for item in gt['2606-POS-0057101_001_02_A4(27).html']
               if item['spec_name'] == spec]
    if gt_items:
        print(f"Ground Truth:")
        for item in gt_items:
            print(f"  Value: '{item['value']}'")
            print(f"  Unit: '{item['unit']}'")
            print(f"  Raw: {item['raw_text'][:100]}")

    # Extracted
    print(f"\nExtracted matches:")
    found = False
    for kv in parser.kv_pairs:
        if spec.lower().replace(' ', '') in kv['key'].lower().replace(' ', ''):
            print(f"  Key: '{kv['key']}'")
            print(f"  Val: '{kv['value']}'")
            found = True

    if not found:
        print("  NOT FOUND")
