#!/usr/bin/env python3
"""
자동 정답지 생성 스크립트

HTML 파일을 파싱하여 테이블에서 사양 항목과 값을 추출하고,
이를 정답지로 저장합니다. 하드코딩 없이 알고리즘 기반으로 동작합니다.
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup


def normalize_text(text: str) -> str:
    """텍스트 정규화"""
    if not text:
        return ""
    # 공백 정규화
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def extract_number_with_unit(text: str) -> Optional[Tuple[str, str]]:
    """
    텍스트에서 숫자와 단위 추출 (알고리즘 기반)

    Returns:
        (value, unit) 튜플 또는 None
    """
    if not text:
        return None

    text = normalize_text(text)

    # 패턴 1: 괄호 안의 값 (예: "(33OC)", "(-163°C)")
    paren_match = re.search(r'\(([^)]+)\)', text)
    if paren_match:
        inner = paren_match.group(1)
        # 숫자 패턴
        num_match = re.search(r'([-+]?\d+(?:[.,~\-]\d+)*)', inner)
        if num_match:
            value = num_match.group(1)
            # 단위 찾기
            unit_match = re.search(r'([A-Za-z°℃%]+)', inner)
            unit = unit_match.group(1) if unit_match else ""
            return (value, unit)

    # 패턴 2: 범위 표기 (예: "5 ~ 8 bar", "10 - 55 O C", "-20 to 70")
    range_patterns = [
        r'([-+]?\d+(?:\.\d+)?)\s*[~\-]\s*([-+]?\d+(?:\.\d+)?)\s*([A-Za-z°℃%/]+)?',
        r'([-+]?\d+(?:\.\d+)?)\s+to\s+([-+]?\d+(?:\.\d+)?)\s*([A-Za-z°℃%/]+)?',
    ]
    for pattern in range_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = f"{match.group(1)} ~ {match.group(2)}"
            unit = match.group(3) if match.group(3) else ""
            return (value, unit)

    # 패턴 3: 단일 숫자+단위 (예: "33°C", "50 bar")
    simple_match = re.search(r'([-+]?\d+(?:\.\d+)?)\s*([A-Za-z°℃%/]+)', text)
    if simple_match:
        return (simple_match.group(1), simple_match.group(2))

    # 패턴 4: 텍스트 값 (예: "SUS316", "STAINLESS STEEL")
    # 단, 숫자가 포함되어 있거나 길이가 짧으면
    if re.search(r'\d', text) or len(text) <= 30:
        # 명확한 사양 값으로 보이는 경우
        if not any(noise in text.upper() for noise in ['GENERAL', 'TABLE', 'SECTION', 'PAGE', 'DOCUMENT']):
            return (text, "")

    return None


def parse_html_tables(html_path: str) -> List[Dict]:
    """
    HTML에서 테이블을 파싱하고 사양 항목 추출

    Returns:
        [{"spec_name": "...", "value": "...", "unit": "...", "raw_text": "..."}, ...]
    """
    with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    spec_items = []

    # 모든 테이블 처리
    for table in soup.find_all('table'):
        rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])

            if len(cells) < 2:
                continue

            # 첫 번째 셀: 사양명 후보
            # 두 번째 셀: 사양값 후보
            for i in range(len(cells) - 1):
                key_cell = cells[i].get_text(strip=True)
                value_cell = cells[i + 1].get_text(strip=True)

                key_cell = normalize_text(key_cell)
                value_cell = normalize_text(value_cell)

                # 사양명 패턴 (대문자 위주, 콜론/하이픈 포함 가능)
                if not key_cell or len(key_cell) < 3:
                    continue

                # 값에서 숫자/단위 추출
                extracted = extract_number_with_unit(value_cell)

                if extracted:
                    value, unit = extracted
                    spec_items.append({
                        "spec_name": key_cell,
                        "value": value,
                        "unit": unit,
                        "raw_text": f"{key_cell} | {value_cell}",
                        "source": os.path.basename(html_path)
                    })

    return spec_items


def filter_valid_specs(spec_items: List[Dict]) -> List[Dict]:
    """
    유효한 사양 항목만 필터링

    필터링 기준:
    1. 사양명이 너무 길지 않음 (100자 이하)
    2. 사양명에 노이즈 키워드가 없음
    3. 값에 숫자가 있어야 함 (헤더 제외)
    4. 중복 제거
    """
    noise_keywords = [
        'GENERAL', 'TABLE', 'SECTION', 'PAGE', 'DOCUMENT', 'REVISION',
        'REVIEWED', 'APPROVED', 'SIGNATURE', 'DATE', 'CONTENTS',
        'ITEM NO', 'NO.', 'DESCRIPTION', 'INDEX', 'REVISION',
        'RANGE', 'DESIGN', 'COMPOSITION', 'MOLE', 'MOL %'
    ]

    # 헤더로 보이는 패턴
    header_patterns = [
        r'^[A-Z\s]+\([A-Z%\s]+\)$',  # "LNG (Mole %)"
        r'^[A-Z\s]+:$',  # "Range:"
        r'^\d+(\.\d+)?$'  # 순수 숫자 (키가 숫자인 경우 제외)
    ]

    filtered = []
    seen = set()

    for item in spec_items:
        spec_name = item['spec_name'].upper()
        value = item['value'].upper()

        # 길이 체크
        if len(spec_name) > 100:
            continue

        # 노이즈 키워드 체크
        if any(noise in spec_name for noise in noise_keywords):
            continue

        # 헤더 패턴 체크
        if any(re.match(pat, spec_name) for pat in header_patterns):
            continue

        # 값이 헤더처럼 보이는지 체크
        if any(re.match(pat, value) for pat in header_patterns):
            # 단, 값에 숫자가 있으면 허용
            if not re.search(r'\d', value):
                continue

        # 값에 숫자가 있어야 함 (텍스트 사양은 제외하지 않음)
        # 이 조건은 완화: 숫자가 없어도 짧고 의미있으면 허용
        if not re.search(r'\d', value) and len(value) > 50:
            continue

        # 중복 체크
        key = (spec_name, value)
        if key in seen:
            continue
        seen.add(key)

        filtered.append(item)

    return filtered


def generate_ground_truth_for_all_files(html_dir: str) -> Dict[str, List[Dict]]:
    """
    모든 HTML 파일에 대해 정답지 생성

    Returns:
        {filename: [spec_items...], ...}
    """
    ground_truth = {}

    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html') and f[0].isdigit()]
    html_files.sort()

    for filename in html_files:
        filepath = os.path.join(html_dir, filename)
        print(f"\n분석 중: {filename}")

        spec_items = parse_html_tables(filepath)
        spec_items = filter_valid_specs(spec_items)

        print(f"  추출된 사양 항목: {len(spec_items)}개")

        ground_truth[filename] = spec_items

    return ground_truth


def save_ground_truth(ground_truth: Dict, output_path: str):
    """정답지를 JSON 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    print(f"\n정답지 저장 완료: {output_path}")


def print_summary(ground_truth: Dict):
    """정답지 요약 출력"""
    print("\n" + "=" * 80)
    print("정답지 생성 요약")
    print("=" * 80)

    total_specs = 0
    for filename, spec_items in ground_truth.items():
        print(f"\n{filename}: {len(spec_items)}개 사양")
        total_specs += len(spec_items)

        # 샘플 5개 출력
        for item in spec_items[:5]:
            print(f"  - {item['spec_name']}: {item['value']} {item['unit']}")

        if len(spec_items) > 5:
            print(f"  ... 외 {len(spec_items) - 5}개")

    print("\n" + "=" * 80)
    print(f"총 {len(ground_truth)}개 파일, {total_specs}개 사양 항목")
    print("=" * 80)


if __name__ == "__main__":
    html_dir = "/home/user/POSLLM"
    output_path = "/home/user/POSLLM/ground_truth_auto.json"

    print("=" * 80)
    print("자동 정답지 생성 시작")
    print("=" * 80)

    ground_truth = generate_ground_truth_for_all_files(html_dir)
    save_ground_truth(ground_truth, output_path)
    print_summary(ground_truth)
