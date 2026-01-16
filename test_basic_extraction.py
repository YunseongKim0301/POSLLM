#!/usr/bin/env python3
"""
기본 추출 테스트 (PostgreSQL/LLM 없이)

HTMLChunkParser를 사용하여 HTML 파싱 및 키-값 추출을 테스트합니다.
"""

import json
import re
import sys
from typing import Dict, List, Tuple


# HTMLChunkParser를 간단하게 재구현 (BeautifulSoup 사용)
from bs4 import BeautifulSoup


def normalize_text(text: str) -> str:
    """텍스트 정규화"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def extract_kv_pairs_from_html(html_path: str) -> List[Dict]:
    """HTML에서 키-값 쌍 추출"""
    with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    kv_pairs = []
    seen = set()

    for table in soup.find_all('table'):
        rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])

            if len(cells) < 2:
                continue

            # 2-column table: key | value
            for i in range(len(cells) - 1):
                key = normalize_text(cells[i].get_text())
                value = normalize_text(cells[i + 1].get_text())

                if not key or len(key) < 3 or len(key) > 100:
                    continue

                # 노이즈 필터링
                if any(noise in key.upper() for noise in ['GENERAL', 'TABLE', 'SECTION', 'PAGE']):
                    continue

                # 값이 의미 있는 경우만
                if value and len(value) < 100:
                    pair_key = (key.upper(), value.upper())
                    if pair_key not in seen:
                        seen.add(pair_key)
                        kv_pairs.append({"key": key, "value": value})

    return kv_pairs


def compare_with_ground_truth(extracted: List[Dict], ground_truth: List[Dict]) -> Tuple[int, int, int]:
    """
    추출 결과와 정답 비교

    Returns:
        (matched, total_ground_truth, total_extracted)
    """
    matched = 0

    # 정답의 키를 정규화하여 인덱스 생성
    gt_index = {}
    for item in ground_truth:
        key = normalize_text(item['spec_name']).upper()
        value = normalize_text(item['value']).upper()
        gt_index[key] = value

    # 추출 결과를 정답과 비교
    extracted_index = {}
    for item in extracted:
        key = normalize_text(item['key']).upper()
        value = normalize_text(item['value']).upper()
        extracted_index[key] = value

    # 매칭 카운트
    for gt_key, gt_value in gt_index.items():
        if gt_key in extracted_index:
            ext_value = extracted_index[gt_key]

            # 값 비교 (부분 매칭도 허용)
            if gt_value in ext_value or ext_value in gt_value:
                matched += 1

    return matched, len(ground_truth), len(extracted)


def test_single_file(html_path: str, ground_truth: List[Dict]) -> Dict:
    """단일 파일 테스트"""
    extracted = extract_kv_pairs_from_html(html_path)

    matched, total_gt, total_ext = compare_with_ground_truth(extracted, ground_truth)

    accuracy = (matched / total_gt * 100) if total_gt > 0 else 0

    return {
        "html_path": html_path,
        "matched": matched,
        "total_ground_truth": total_gt,
        "total_extracted": total_ext,
        "accuracy": accuracy
    }


def test_all_files(ground_truth_path: str):
    """모든 파일 테스트"""
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_all = json.load(f)

    print("=" * 80)
    print("기본 추출 테스트 (PostgreSQL/LLM 없이)")
    print("=" * 80)

    results = []
    total_matched = 0
    total_gt = 0
    total_ext = 0

    for filename, gt_items in ground_truth_all.items():
        html_path = f"/home/user/POSLLM/{filename}"

        result = test_single_file(html_path, gt_items)
        results.append(result)

        total_matched += result['matched']
        total_gt += result['total_ground_truth']
        total_ext += result['total_extracted']

        print(f"\n{filename}:")
        print(f"  정답: {result['total_ground_truth']}개")
        print(f"  추출: {result['total_extracted']}개")
        print(f"  매칭: {result['matched']}개")
        print(f"  정확도: {result['accuracy']:.1f}%")

    overall_accuracy = (total_matched / total_gt * 100) if total_gt > 0 else 0

    print("\n" + "=" * 80)
    print("전체 결과")
    print("=" * 80)
    print(f"정답 총 사양: {total_gt}개")
    print(f"추출 총 사양: {total_ext}개")
    print(f"매칭: {total_matched}개")
    print(f"전체 정확도: {overall_accuracy:.1f}%")
    print("=" * 80)

    return results, overall_accuracy


if __name__ == "__main__":
    ground_truth_path = "/home/user/POSLLM/ground_truth_auto.json"

    results, accuracy = test_all_files(ground_truth_path)

    print(f"\n최종 정확도: {accuracy:.1f}%")

    if accuracy < 90.0:
        print(f"\n⚠️  목표 정확도 90%에 미달 (현재: {accuracy:.1f}%)")
        print("추가 개선이 필요합니다.")
    else:
        print(f"\n✓ 목표 정확도 90% 달성!")
