#!/usr/bin/env python3
"""
Comprehensive evaluation script for v61 and v70 extractors
Tests all HTML files against ground truth and calculates accuracy
"""

import os
import sys
import json
import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import glob

# Import HTMLChunkParser from both versions
sys.path.insert(0, os.path.dirname(__file__))


def aggressive_normalize(text: str) -> str:
    """Aggressive normalization for comparison"""
    if not text:
        return ""
    text = re.sub(r'\s+', '', text)
    text = text.replace('*', '').replace('□', '').replace('■', '')
    text = text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    text = text.replace(',', '')
    return text.upper()


def fuzzy_match_keys(key1: str, key2: str, threshold: float = 0.70) -> bool:
    """Fuzzy matching for keys"""
    norm1 = aggressive_normalize(key1)
    norm2 = aggressive_normalize(key2)

    if norm1 == norm2:
        return True

    if norm1 in norm2 or norm2 in norm1:
        return True

    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    if similarity >= threshold:
        return True

    words1 = set(re.findall(r'[A-Z0-9]{2,}', norm1))
    words2 = set(re.findall(r'[A-Z0-9]{2,}', norm2))

    if words1 and words2:
        common = words1 & words2
        word_similarity = len(common) / max(len(words1), len(words2))
        if word_similarity >= 0.55:
            return True

    return False


def extract_numbers(text: str) -> List[str]:
    """Extract all numbers from text"""
    if not text:
        return []
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return numbers


def extract_kv_with_parser(html_path: str, parser_class) -> List[Dict]:
    """Extract key-value pairs using HTMLChunkParser"""
    parser = parser_class(file_path=html_path)

    extracted = []
    for pair in parser.kv_pairs:
        extracted.append({
            'key': pair['key'],
            'value': pair['value'],
            'norm_key': aggressive_normalize(pair['key']),
            'norm_value': aggressive_normalize(pair['value'])
        })

    return extracted


def compare_with_ground_truth(
    extracted: List[Dict],
    ground_truth: List[Dict]
) -> Tuple[int, int, List[Dict]]:
    """Compare extraction results with ground truth"""
    matched = 0
    details = []

    # Track matched items
    extracted_matched = [False] * len(extracted)

    # Try to match each ground truth item
    for gt_item in ground_truth:
        spec_name = gt_item['spec_name']
        gt_norm_key = aggressive_normalize(spec_name)
        gt_norm_value = aggressive_normalize(gt_item['value'])

        matched_key = False
        matched_value = False
        best_match_idx = -1

        # Try to match with extracted items
        for idx, ext_data in enumerate(extracted):
            if extracted_matched[idx]:
                continue

            # Check key match
            key_match = False
            if gt_norm_key == ext_data['norm_key']:
                key_match = True
            elif fuzzy_match_keys(gt_norm_key, ext_data['norm_key'], threshold=0.70):
                key_match = True

            if not key_match:
                continue

            # Check value match
            value_match = False
            if (gt_norm_value in ext_data['norm_value'] or
                ext_data['norm_value'] in gt_norm_value or
                gt_norm_value == ext_data['norm_value']):
                value_match = True
            elif SequenceMatcher(None, gt_norm_value, ext_data['norm_value']).ratio() >= 0.85:
                value_match = True
            else:
                gt_nums = extract_numbers(gt_norm_value)
                ext_nums = extract_numbers(ext_data['norm_value'])
                if gt_nums and ext_nums:
                    if all(num in ext_nums for num in gt_nums):
                        value_match = True

            if key_match and value_match:
                matched_key = True
                matched_value = True
                best_match_idx = idx
                break

        # Mark matched extraction
        if best_match_idx >= 0:
            extracted_matched[best_match_idx] = True
            matched += 1

        details.append({
            'spec_name': spec_name,
            'gt_value': gt_item['value'],
            'matched_key': matched_key,
            'matched_value': matched_value,
            'extracted_value': extracted[best_match_idx]['value'] if best_match_idx >= 0 else None
        })

    return matched, len(ground_truth), details


def evaluate_file(html_path: str, ground_truth: List[Dict], parser_class) -> Dict:
    """Evaluate a single HTML file"""
    filename = os.path.basename(html_path)

    try:
        extracted = extract_kv_with_parser(html_path, parser_class)
        matched, total, details = compare_with_ground_truth(extracted, ground_truth)

        accuracy = (matched / total * 100) if total > 0 else 0.0

        return {
            'filename': filename,
            'total': total,
            'matched': matched,
            'accuracy': accuracy,
            'details': details
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'filename': filename,
            'total': 0,
            'matched': 0,
            'accuracy': 0.0,
            'error': str(e)
        }


def evaluate_all_files(base_dir: str, ground_truth_path: str, parser_class, version_name: str) -> Dict:
    """Evaluate all HTML files"""
    # Load ground truth
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)

    # Find all HTML files
    html_files = glob.glob(os.path.join(base_dir, "*.html"))

    print(f"\n{'='*80}")
    print(f"Evaluating {version_name}")
    print(f"{'='*80}")
    print(f"Found {len(html_files)} HTML files")
    print(f"Ground truth contains {len(ground_truth_data)} files")

    results = []
    total_matched = 0
    total_specs = 0

    for html_path in sorted(html_files):
        filename = os.path.basename(html_path)

        if filename not in ground_truth_data:
            print(f"  Skipping {filename} (no ground truth)")
            continue

        gt = ground_truth_data[filename]
        result = evaluate_file(html_path, gt, parser_class)
        results.append(result)

        total_matched += result['matched']
        total_specs += result['total']

        print(f"  {filename}: {result['matched']}/{result['total']} ({result['accuracy']:.1f}%)")

    overall_accuracy = (total_matched / total_specs * 100) if total_specs > 0 else 0.0

    print(f"\n{'-'*80}")
    print(f"Overall: {total_matched}/{total_specs} ({overall_accuracy:.2f}%)")
    print(f"{'='*80}\n")

    return {
        'version': version_name,
        'overall_accuracy': overall_accuracy,
        'total_matched': total_matched,
        'total_specs': total_specs,
        'file_results': results
    }


def analyze_failures(results: Dict) -> Dict:
    """Analyze failure patterns"""
    failures = []

    for file_result in results['file_results']:
        for detail in file_result.get('details', []):
            if not (detail['matched_key'] and detail['matched_value']):
                failures.append({
                    'filename': file_result['filename'],
                    'spec_name': detail['spec_name'],
                    'gt_value': detail['gt_value'],
                    'extracted_value': detail.get('extracted_value'),
                    'matched_key': detail['matched_key'],
                    'matched_value': detail['matched_value']
                })

    # Categorize failures
    key_failures = [f for f in failures if not f['matched_key']]
    value_failures = [f for f in failures if f['matched_key'] and not f['matched_value']]

    return {
        'total_failures': len(failures),
        'key_failures': len(key_failures),
        'value_failures': len(value_failures),
        'key_failure_examples': key_failures[:10],
        'value_failure_examples': value_failures[:10]
    }


def main():
    """Main evaluation function"""
    base_dir = "/home/user/POSLLM"
    ground_truth_path = os.path.join(base_dir, "ground_truth_auto.json")

    # Import v61 parser
    from v61_extractor import HTMLChunkParser as V61Parser

    # Import v70 parser
    from v70_extractor import HTMLChunkParser as V70Parser

    # Evaluate v61
    v61_results = evaluate_all_files(base_dir, ground_truth_path, V61Parser, "v61")

    # Evaluate v70
    v70_results = evaluate_all_files(base_dir, ground_truth_path, V70Parser, "v70")

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"v61 Accuracy: {v61_results['overall_accuracy']:.2f}% ({v61_results['total_matched']}/{v61_results['total_specs']})")
    print(f"v70 Accuracy: {v70_results['overall_accuracy']:.2f}% ({v70_results['total_matched']}/{v70_results['total_specs']})")

    diff = v70_results['overall_accuracy'] - v61_results['overall_accuracy']
    print(f"Difference: {diff:+.2f}%")
    print("="*80)

    # Analyze failures for both versions
    print("\n" + "="*80)
    print("FAILURE ANALYSIS - v61")
    print("="*80)
    v61_failures = analyze_failures(v61_results)
    print(f"Total failures: {v61_failures['total_failures']}")
    print(f"  Key failures: {v61_failures['key_failures']}")
    print(f"  Value failures: {v61_failures['value_failures']}")

    print("\n" + "="*80)
    print("FAILURE ANALYSIS - v70")
    print("="*80)
    v70_failures = analyze_failures(v70_results)
    print(f"Total failures: {v70_failures['total_failures']}")
    print(f"  Key failures: {v70_failures['key_failures']}")
    print(f"  Value failures: {v70_failures['value_failures']}")

    # Save detailed results
    output_path = os.path.join(base_dir, "evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'v61': v61_results,
            'v70': v70_results,
            'v61_failures': v61_failures,
            'v70_failures': v70_failures
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")

    return v61_results, v70_results


if __name__ == "__main__":
    main()
