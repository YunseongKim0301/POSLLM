#!/usr/bin/env python3
"""
v61_extractor.py의 HTMLChunkParser 테스트

실제 v61_extractor.py의 HTMLChunkParser를 사용하여 정확도 테스트
"""

import os
import sys
import json
import re
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

# v61_extractor.py에서 HTMLChunkParser import
sys.path.insert(0, os.path.dirname(__file__))
from v61_extractor import HTMLChunkParser


def aggressive_normalize(text: str) -> str:
    """강화된 정규화"""
    if not text:
        return ""
    text = re.sub(r'\s+', '', text)
    text = text.replace('*', '').replace('□', '').replace('■', '')
    text = text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    text = text.replace(',', '')
    return text.upper()


def fuzzy_match_keys(key1: str, key2: str, threshold: float = 0.70) -> bool:
    """퍼지 매칭"""
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
    """텍스트에서 모든 숫자 추출"""
    if not text:
        return []
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return numbers


def extract_kv_from_html_with_parser(html_path: str) -> List[Dict]:
    """HTMLChunkParser를 사용하여 키-값 쌍 추출"""
    parser = HTMLChunkParser(file_path=html_path)

    # kv_pairs를 반환 형식에 맞게 변환
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
    """정답과 비교 (중복 키 처리 포함)"""
    matched = 0
    details = []

    # 추출 데이터는 리스트로 유지
    extracted_list = []
    for item in extracted:
        extracted_list.append({
            'key': item['key'],
            'value': item['value'],
            'norm_key': item['norm_key'],
            'norm_value': item['norm_value']
        })

    # 매칭 추적
    extracted_matched = [False] * len(extracted_list)

    # 정답 항목별로 매칭 시도
    for gt_item in ground_truth:
        spec_name = gt_item['spec_name']
        gt_norm_key = aggressive_normalize(spec_name)
        gt_norm_value = aggressive_normalize(gt_item['value'])

        matched_key = False
        matched_value = False
        best_match_idx = -1

        # 모든 추출 항목을 순회하여 매칭 시도
        for idx, ext_data in enumerate(extracted_list):
            if extracted_matched[idx]:
                continue

            # 키 매칭 확인
            key_match = False
            if gt_norm_key == ext_data['norm_key']:
                key_match = True
            elif fuzzy_match_keys(gt_norm_key, ext_data['norm_key'], threshold=0.70):
                key_match = True

            if not key_match:
                continue

            # 값 매칭 확인
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

        # 매칭 성공 시 해당 추출 항목을 사용됨으로 표시
        if matched_key and matched_value and best_match_idx >= 0:
            extracted_matched[best_match_idx] = True
            matched += 1

        details.append({
            'spec': spec_name,
            'expected_value': gt_item['value'],
            'matched': (matched_key and matched_value)
        })

    return matched, len(ground_truth), details


def run_test(ground_truth_path: str, html_dir: str):
    """테스트 실행"""
    print("=" * 80)
    print("v61_extractor.py HTMLChunkParser 테스트")
    print("=" * 80)

    # 정답지 로드
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_all = json.load(f)

    total_matched = 0
    total_gt = 0

    results = []

    for filename, gt_items in ground_truth_all.items():
        html_path = os.path.join(html_dir, filename)

        if not os.path.exists(html_path):
            print(f"\n⚠️  파일 없음: {filename}")
            continue

        # 추출
        extracted = extract_kv_from_html_with_parser(html_path)

        # 비교
        matched, gt_count, details = compare_with_ground_truth(extracted, gt_items)

        accuracy = (matched / gt_count * 100) if gt_count > 0 else 0

        total_matched += matched
        total_gt += gt_count

        results.append({
            'filename': filename,
            'matched': matched,
            'total_gt': gt_count,
            'total_extracted': len(extracted),
            'accuracy': accuracy
        })

        status = "✓" if accuracy >= 90.0 else "✗"
        print(f"\n{status} {filename}:")
        print(f"  정답: {gt_count}개")
        print(f"  추출: {len(extracted)}개")
        print(f"  매칭: {matched}개")
        print(f"  정확도: {accuracy:.1f}%")

    # 전체 결과
    overall_accuracy = (total_matched / total_gt * 100) if total_gt > 0 else 0

    print("\n" + "=" * 80)
    print("전체 결과")
    print("=" * 80)
    print(f"정답 총 사양: {total_gt}개")
    print(f"매칭: {total_matched}개")
    print(f"전체 정확도: {overall_accuracy:.1f}%")
    print("=" * 80)

    if overall_accuracy >= 90.0:
        print(f"\n✓ 목표 정확도 90% 달성!")
    else:
        print(f"\n✗ 목표 정확도 90%에 미달 (현재: {overall_accuracy:.1f}%)")
        print(f"개선이 필요합니다.")

    return overall_accuracy, results


if __name__ == "__main__":
    ground_truth_path = "/home/user/POSLLM/ground_truth_auto.json"
    html_dir = "/home/user/POSLLM"

    accuracy, results = run_test(ground_truth_path, html_dir)

    sys.exit(0 if accuracy >= 90.0 else 1)
