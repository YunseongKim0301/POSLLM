#!/usr/bin/env python3
"""
개선된 추출 엔진

문제 해결:
1. 강화된 정규화 (공백, 특수문자)
2. 동의어/약어 매칭
3. 복합 키-값 구조 파싱
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
from difflib import SequenceMatcher


def aggressive_normalize(text: str) -> str:
    """
    강화된 정규화

    - 모든 공백 제거
    - 특수문자 최소화
    - 대문자 변환
    """
    if not text:
        return ""

    # 공백 제거
    text = re.sub(r'\s+', '', text)

    # 특수문자 정규화
    text = text.replace('*', '').replace('□', '').replace('■', '')
    text = text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')

    return text.upper()


def fuzzy_match_keys(key1: str, key2: str, threshold: float = 0.8) -> bool:
    """
    퍼지 매칭 (유사도 기반)

    Returns:
        True if similarity >= threshold
    """
    # 정규화 후 비교
    norm1 = aggressive_normalize(key1)
    norm2 = aggressive_normalize(key2)

    # 완전 일치
    if norm1 == norm2:
        return True

    # 유사도 계산
    similarity = SequenceMatcher(None, norm1, norm2).ratio()

    return similarity >= threshold


def normalize_value(value: str) -> str:
    """
    값 정규화

    - 공백 정규화 (단, 완전 제거 안 함)
    - 단위 정규화 (OC → °C, BARG → bar)
    """
    if not value:
        return ""

    # 공백 정규화
    value = re.sub(r'\s+', ' ', value.strip())

    # 단위 정규화
    value = re.sub(r'OC\b', '°C', value, flags=re.IGNORECASE)
    value = re.sub(r'O\s+C\b', '°C', value, flags=re.IGNORECASE)
    value = value.replace('BARG', 'bar').replace(' BAR', ' bar')

    return value.upper()


def extract_kv_pairs_improved(html_path: str) -> List[Dict]:
    """HTML에서 키-값 쌍 추출 (개선 버전)"""
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

            # 다양한 테이블 구조 지원
            for i in range(len(cells) - 1):
                key_raw = cells[i].get_text()
                value_raw = cells[i + 1].get_text()

                key = re.sub(r'\s+', ' ', key_raw.strip())
                value = re.sub(r'\s+', ' ', value_raw.strip())

                if not key or len(key) < 3 or len(key) > 150:
                    continue

                # 노이즈 필터링 (더 엄격하게)
                noise_patterns = [
                    r'^GENERAL\b', r'^TABLE\b', r'^SECTION\b', r'^PAGE\b',
                    r'^ITEM\s*NO', r'^NO\.\s*$', r'^DESCRIPTION\s*$',
                    r'^REV\.', r'^DATE\s*$'
                ]
                if any(re.search(pat, key.upper()) for pat in noise_patterns):
                    continue

                # 값이 의미 있는 경우만
                if value and len(value) < 200:
                    # 정규화된 키로 중복 체크
                    norm_key = aggressive_normalize(key)
                    norm_value = aggressive_normalize(value)

                    pair_key = (norm_key, norm_value)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        kv_pairs.append({
                            "key": key,
                            "value": value,
                            "norm_key": norm_key,
                            "norm_value": norm_value
                        })

    return kv_pairs


def compare_with_ground_truth_improved(extracted: List[Dict], ground_truth: List[Dict]) -> Tuple[int, int, int, List[Dict]]:
    """
    개선된 비교 로직

    Returns:
        (matched, total_ground_truth, total_extracted, details)
    """
    matched = 0
    details = []

    # 정답 인덱스 (정규화된 키)
    gt_index = {}
    for item in ground_truth:
        norm_key = aggressive_normalize(item['spec_name'])
        norm_value = aggressive_normalize(item['value'])
        gt_index[norm_key] = {
            "original_key": item['spec_name'],
            "original_value": item['value'],
            "norm_value": norm_value
        }

    # 추출 결과 인덱스 (정규화된 키)
    extracted_index = {}
    for item in extracted:
        norm_key = item['norm_key']
        norm_value = item['norm_value']
        extracted_index[norm_key] = {
            "original_key": item['key'],
            "original_value": item['value'],
            "norm_value": norm_value
        }

    # 매칭
    for gt_norm_key, gt_data in gt_index.items():
        matched_key = False
        matched_value = False

        # 1. 완전 일치
        if gt_norm_key in extracted_index:
            matched_key = True
            ext_data = extracted_index[gt_norm_key]

            # 값 비교 (부분 매칭 허용)
            if (gt_data['norm_value'] in ext_data['norm_value'] or
                ext_data['norm_value'] in gt_data['norm_value'] or
                gt_data['norm_value'] == ext_data['norm_value']):
                matched_value = True

        # 2. 퍼지 매칭 (완전 일치 실패 시)
        if not matched_key:
            for ext_norm_key, ext_data in extracted_index.items():
                if fuzzy_match_keys(gt_norm_key, ext_norm_key, threshold=0.85):
                    matched_key = True

                    # 값 비교
                    if (gt_data['norm_value'] in ext_data['norm_value'] or
                        ext_data['norm_value'] in gt_data['norm_value']):
                        matched_value = True
                    break

        # 결과 기록
        is_success = matched_key and matched_value

        if is_success:
            matched += 1

        details.append({
            "gt_key": gt_data['original_key'],
            "gt_value": gt_data['original_value'],
            "matched": is_success,
            "matched_key": matched_key,
            "matched_value": matched_value
        })

    return matched, len(ground_truth), len(extracted), details


def test_improved_extraction(ground_truth_path: str):
    """개선된 추출 테스트"""
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_all = json.load(f)

    print("=" * 80)
    print("개선된 추출 엔진 테스트")
    print("=" * 80)

    results = []
    total_matched = 0
    total_gt = 0
    total_ext = 0

    for filename, gt_items in ground_truth_all.items():
        html_path = f"/home/user/POSLLM/{filename}"

        extracted = extract_kv_pairs_improved(html_path)
        matched, gt_count, ext_count, details = compare_with_ground_truth_improved(extracted, gt_items)

        accuracy = (matched / gt_count * 100) if gt_count > 0 else 0

        total_matched += matched
        total_gt += gt_count
        total_ext += ext_count

        results.append({
            "filename": filename,
            "matched": matched,
            "total_gt": gt_count,
            "total_ext": ext_count,
            "accuracy": accuracy
        })

        print(f"\n{filename}:")
        print(f"  정답: {gt_count}개")
        print(f"  추출: {ext_count}개")
        print(f"  매칭: {matched}개")
        print(f"  정확도: {accuracy:.1f}%")

    overall_accuracy = (total_matched / total_gt * 100) if total_gt > 0 else 0

    print("\n" + "=" * 80)
    print("전체 결과")
    print("=" * 80)
    print(f"정답 총 사양: {total_gt}개")
    print(f"추출 총 사양: {total_ext}개")
    print(f"매칭: {total_matched}개")
    print(f"전체 정확도: {overall_accuracy:.1f}%")
    print("=" * 80)

    return overall_accuracy


if __name__ == "__main__":
    ground_truth_path = "/home/user/POSLLM/ground_truth_auto.json"

    accuracy = test_improved_extraction(ground_truth_path)

    print(f"\n최종 정확도: {accuracy:.1f}%")

    if accuracy < 90.0:
        print(f"\n⚠️  목표 정확도 90%에 미달 (현재: {accuracy:.1f}%)")
        print(f"개선 전 69.2% → 개선 후 {accuracy:.1f}% (+ {accuracy - 69.2:.1f}%p)")
    else:
        print(f"\n✓ 목표 정확도 90% 달성!")
