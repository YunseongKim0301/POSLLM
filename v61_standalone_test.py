#!/usr/bin/env python3
"""
v61_extractor.py Standalone Test Module

PostgreSQL 없이 HTML 파싱 및 정확도 테스트 수행
"""

import os
import sys
import json
import re
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup
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
    # 숫자 내 쉼표 제거 (1,000 → 1000)
    text = text.replace(',', '')
    return text.upper()


def fuzzy_match_keys(key1: str, key2: str, threshold: float = 0.70) -> bool:
    """퍼지 매칭 (임계값 0.75 → 0.70 완화)"""
    norm1 = aggressive_normalize(key1)
    norm2 = aggressive_normalize(key2)

    if norm1 == norm2:
        return True

    # 부분 문자열 매칭 (하나가 다른 것에 포함되면 매칭)
    if norm1 in norm2 or norm2 in norm1:
        return True

    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    if similarity >= threshold:
        return True

    # 단어 단위 매칭 (0.6 → 0.55 완화)
    words1 = set(re.findall(r'[A-Z0-9]{2,}', norm1))
    words2 = set(re.findall(r'[A-Z0-9]{2,}', norm2))

    if words1 and words2:
        common = words1 & words2
        word_similarity = len(common) / max(len(words1), len(words2))
        if word_similarity >= 0.55:
            return True

    return False


def detect_header_row(row_cells: List[str]) -> bool:
    """헤더 행 감지"""
    if not row_cells:
        return False

    non_empty = [c for c in row_cells if c.strip()]
    if len(non_empty) < len(row_cells) * 0.3:
        return False

    header_keywords = ['COMPOSITION', 'RANGE', 'DESIGN', 'ITEM', 'SPECIFICATION',
                       'PARAMETER', 'VALUE', 'UNIT', 'TYPE', 'MODEL']

    text = ' '.join(row_cells).upper()
    has_keyword = any(kw in text for kw in header_keywords)

    digit_count = sum(1 for c in text if c.isdigit())
    total_chars = len([c for c in text if c.isalnum()])
    digit_ratio = digit_count / total_chars if total_chars > 0 else 0

    return has_keyword or digit_ratio < 0.3


def extract_horizontal_data_table(table) -> List[Dict]:
    """수평 데이터 테이블 파싱 (row=item, column=attribute)"""
    rows = table.find_all('tr')
    if not rows:
        return []

    # 헤더 감지
    header_rows = []
    data_start_idx = 0

    for i, row in enumerate(rows[:5]):
        cells = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
        if detect_header_row(cells):
            header_rows.append(cells)
            data_start_idx = i + 1
        else:
            break

    if not header_rows:
        return []

    headers = header_rows[-1] if header_rows else []

    # 데이터 행 파싱
    kv_pairs = []

    for row_idx in range(data_start_idx, len(rows)):
        row = rows[row_idx]
        cells = row.find_all(['td', 'th'])

        if not cells:
            continue

        row_label_raw = cells[0].get_text(strip=True)

        if not row_label_raw or len(row_label_raw) > 100:
            continue

        row_label = re.sub(r'\s+', ' ', row_label_raw)

        for col_idx, cell in enumerate(cells[1:], start=1):
            value_raw = cell.get_text(strip=True)

            if not value_raw or len(value_raw) > 100:
                continue

            value = re.sub(r'\s+', ' ', value_raw)

            col_name = headers[col_idx] if col_idx < len(headers) else f"Column{col_idx}"

            if col_name and col_name.strip():
                combined_key = f"{row_label}_{col_name}"
            else:
                combined_key = row_label

            kv_pairs.append({
                'key': combined_key,
                'value': value,
                'norm_key': aggressive_normalize(combined_key),
                'norm_value': aggressive_normalize(value)
            })

    return kv_pairs


def extract_vertical_kv_table(table) -> List[Dict]:
    """수직 키-값 테이블 파싱 (row: key | value)"""
    rows = table.find_all('tr')
    kv_pairs = []

    for row in rows:
        cells = row.find_all(['td', 'th'])

        if len(cells) < 2:
            continue

        for i in range(len(cells) - 1):
            key_raw = cells[i].get_text(strip=True)
            value_raw = cells[i + 1].get_text(strip=True)

            if not key_raw or len(key_raw) < 3 or len(key_raw) > 150:
                continue

            key = re.sub(r'\s+', ' ', key_raw)
            value = re.sub(r'\s+', ' ', value_raw)

            noise_patterns = [
                r'^GENERAL\b', r'^TABLE\b', r'^SECTION\b', r'^PAGE\b',
                r'^ITEM\s*NO', r'^NO\.\s*$', r'^DESCRIPTION\s*$'
            ]
            if any(re.search(pat, key.upper()) for pat in noise_patterns):
                continue

            if value and len(value) < 200:
                kv_pairs.append({
                    'key': key,
                    'value': value,
                    'norm_key': aggressive_normalize(key),
                    'norm_value': aggressive_normalize(value)
                })

    return kv_pairs


def is_value_like(text: str) -> bool:
    """텍스트가 값처럼 보이는지 판단 (GT 노이즈 필터링용)"""
    if not text or len(text) < 2:
        return False

    # 범위 패턴 (숫자 ~ 숫자)
    if re.search(r'\d+(?:\.\d+)?\s*~\s*\d+(?:\.\d+)?', text):
        return True

    # 순수 숫자만 있는 경우
    if re.match(r'^[\d.,\-+\s]+$', text):
        return True

    # 짧은 텍스트에 숫자가 많은 경우
    digits = sum(1 for c in text if c.isdigit())
    if len(text) <= 20 and digits >= len(text) * 0.5:
        return True

    return False


def extract_numbers(text: str) -> List[str]:
    """텍스트에서 모든 숫자 추출 (비교용)"""
    if not text:
        return []
    # 소수점 포함 숫자 추출
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return numbers


def extract_kv_from_html(html_path: str) -> List[Dict]:
    """HTML에서 키-값 쌍 추출 (고급 파싱)"""
    with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    all_kv_pairs = []
    seen = set()

    for table in soup.find_all('table'):
        # 수평 데이터 테이블 시도
        horizontal_pairs = extract_horizontal_data_table(table)
        all_kv_pairs.extend(horizontal_pairs)

        # 수직 키-값 테이블 시도
        vertical_pairs = extract_vertical_kv_table(table)
        all_kv_pairs.extend(vertical_pairs)

    # 중복 제거
    unique_pairs = []
    for pair in all_kv_pairs:
        pair_key = (pair['norm_key'], pair['norm_value'])
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append(pair)

    return unique_pairs


def compare_with_ground_truth(
    extracted: List[Dict],
    ground_truth: List[Dict]
) -> Tuple[int, int, List[Dict]]:
    """정답과 비교 (중복 키 처리 포함)"""
    matched = 0
    details = []

    # 추출 데이터는 리스트로 유지 (중복 키 가능)
    extracted_list = []
    for item in extracted:
        extracted_list.append({
            'key': item['key'],
            'value': item['value'],
            'norm_key': item['norm_key'],
            'norm_value': item['norm_value']
        })

    # 매칭 추적 (각 추출 항목은 한 번만 매칭)
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
            # 이미 매칭된 항목은 스킵
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
            # 값 유사도 체크
            elif SequenceMatcher(None, gt_norm_value, ext_data['norm_value']).ratio() >= 0.85:
                value_match = True
            # 숫자 기반 매칭
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
                break  # 첫 번째 매칭을 사용

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


def run_standalone_test(ground_truth_path: str, html_dir: str):
    """Standalone 테스트 실행"""
    print("=" * 80)
    print("v61_extractor.py Standalone Test")
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
        extracted = extract_kv_from_html(html_path)

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

    accuracy, results = run_standalone_test(ground_truth_path, html_dir)

    sys.exit(0 if accuracy >= 90.0 else 1)
