#!/usr/bin/env python3
"""
매칭 실패 원인 분석

정답지와 추출 결과를 비교하여 왜 매칭이 실패했는지 분석합니다.
"""

import json
import re
from typing import Dict, List
from bs4 import BeautifulSoup
from collections import defaultdict


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

            for i in range(len(cells) - 1):
                key = normalize_text(cells[i].get_text())
                value = normalize_text(cells[i + 1].get_text())

                if not key or len(key) < 3 or len(key) > 100:
                    continue

                if any(noise in key.upper() for noise in ['GENERAL', 'TABLE', 'SECTION', 'PAGE']):
                    continue

                if value and len(value) < 100:
                    pair_key = (key.upper(), value.upper())
                    if pair_key not in seen:
                        seen.add(pair_key)
                        kv_pairs.append({"key": key, "value": value})

    return kv_pairs


def analyze_failures(html_path: str, ground_truth: List[Dict]):
    """매칭 실패 분석"""
    extracted = extract_kv_pairs_from_html(html_path)

    # 추출 결과 인덱스
    extracted_index = {}
    for item in extracted:
        key = normalize_text(item['key']).upper()
        value = normalize_text(item['value']).upper()
        extracted_index[key] = value

    # 정답 인덱스
    gt_index = {}
    for item in ground_truth:
        key = normalize_text(item['spec_name']).upper()
        value = normalize_text(item['value']).upper()
        gt_index[key] = value

    # 매칭 및 실패 분석
    failures = defaultdict(list)

    for gt_key, gt_value in gt_index.items():
        if gt_key in extracted_index:
            ext_value = extracted_index[gt_key]

            # 값 비교
            if gt_value in ext_value or ext_value in gt_value:
                # 매칭 성공
                pass
            else:
                # 키는 있지만 값이 다름
                failures['value_mismatch'].append({
                    "key": gt_key,
                    "expected_value": gt_value,
                    "extracted_value": ext_value
                })
        else:
            # 키가 아예 없음
            # 부분 매칭 시도
            partial_matches = []
            for ext_key in extracted_index.keys():
                # 단어 단위로 비교
                gt_words = set(gt_key.split())
                ext_words = set(ext_key.split())

                # 공통 단어가 2개 이상이면 후보
                common = gt_words & ext_words
                if len(common) >= 2:
                    partial_matches.append({
                        "extracted_key": ext_key,
                        "common_words": list(common),
                        "extracted_value": extracted_index[ext_key]
                    })

            if partial_matches:
                failures['key_mismatch_partial'].append({
                    "expected_key": gt_key,
                    "expected_value": gt_value,
                    "partial_matches": partial_matches[:3]  # 상위 3개만
                })
            else:
                failures['key_missing'].append({
                    "expected_key": gt_key,
                    "expected_value": gt_value
                })

    return failures


def print_failure_analysis(filename: str, failures: Dict):
    """실패 분석 결과 출력"""
    print(f"\n{'='*80}")
    print(f"실패 분석: {filename}")
    print(f"{'='*80}")

    # 1. 키 누락
    if failures['key_missing']:
        print(f"\n[1] 키 완전 누락: {len(failures['key_missing'])}개")
        for item in failures['key_missing'][:5]:
            print(f"  - {item['expected_key']}: {item['expected_value']}")
        if len(failures['key_missing']) > 5:
            print(f"  ... 외 {len(failures['key_missing']) - 5}개")

    # 2. 키 부분 매칭
    if failures['key_mismatch_partial']:
        print(f"\n[2] 키 부분 매칭 (동의어/약어 가능성): {len(failures['key_mismatch_partial'])}개")
        for item in failures['key_mismatch_partial'][:5]:
            print(f"  정답: {item['expected_key']}")
            print(f"  후보:")
            for match in item['partial_matches']:
                print(f"    - {match['extracted_key']} (공통어: {match['common_words']})")
        if len(failures['key_mismatch_partial']) > 5:
            print(f"  ... 외 {len(failures['key_mismatch_partial']) - 5}개")

    # 3. 값 불일치
    if failures['value_mismatch']:
        print(f"\n[3] 값 불일치 (키는 매칭, 값이 다름): {len(failures['value_mismatch'])}개")
        for item in failures['value_mismatch'][:5]:
            print(f"  - {item['key']}")
            print(f"    정답: {item['expected_value']}")
            print(f"    추출: {item['extracted_value']}")
        if len(failures['value_mismatch']) > 5:
            print(f"  ... 외 {len(failures['value_mismatch']) - 5}개")


if __name__ == "__main__":
    ground_truth_path = "/home/user/POSLLM/ground_truth_auto.json"

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_all = json.load(f)

    # 정확도가 낮은 파일들만 분석
    low_accuracy_files = [
        "2550-POS-0077601_001_02_A4(16).html",  # 58.0%
        "2606-POS-0036329_001_00_A4(22)_FRS.html",  # 58.8%
        "2598-POS-0070307_000_02_A4(28)_FO_SUPPLY_MODULE.html",  # 62.8%
    ]

    for filename in low_accuracy_files:
        if filename in ground_truth_all:
            html_path = f"/home/user/POSLLM/{filename}"
            ground_truth = ground_truth_all[filename]

            failures = analyze_failures(html_path, ground_truth)
            print_failure_analysis(filename, failures)
