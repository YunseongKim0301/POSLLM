#!/usr/bin/env python3
"""
v61 추출 정확도 테스트 스크립트

HTML 문서에서 사양값을 추출하고 정답지와 비교하여 정확도를 측정합니다.
"""

import json
import re
from typing import Dict, List, Tuple

# 정답지 (HTML 문서를 직접 분석하여 작성)
GROUND_TRUTH = {
    "2597-POS-0039001_000_01_A4_WATER_DRAIN_SYS_FINAL.html": {
        # 온도 관련 (Line 491-492, 436-437, 445-446, 462)
        "MIN. SERVICE TEMPERATURE_SECONDARY INSULATION SPACE": {
            "value": "-163",
            "unit": "°C",
            "source": "Line 462: (-163°C) Minimum cargo temperature"
        },
        "MAX. SERVICE TEMPERATURE_SECONDARY INSULATION SPACE": {
            "value": "70",
            "unit": "°C",
            "source": "Line 492: -20 to 70 O C Open deck"
        },
        "MAX. SERVICE TEMPERATURE_MOTOR ROOM": {
            "value": "55",
            "unit": "°C",
            "source": "Line 491: 10 - 55 O C Machinery space"
        },
        "MAX. SEA WATER TEMPERATURE_SECONDARY INSULATION SPACE": {
            "value": "33",
            "unit": "°C",
            "source": "Line 436-437: Max. (33O C) Sea water temperature"
        },
        "MAX. AIR TEMPERATURE_MOTOR ROOM": {
            "value": "50",
            "unit": "°C",
            "source": "Line 445-446: Max. (50O C) Air temperature"
        },

        # 압력/재질 관련 (Line 563, 567, 631)
        "COMPRESSED AIR_SECONDARY INSULATION SPACE": {
            "value": "5 ~ 8",
            "unit": "bar",
            "source": "Line 563: 5 ~ 8 bar (Service pressure) Compressed air"
        },
        "CASING MATERIAL_SECONDARY INSULATION SPACE": {
            "value": "SUS316",
            "unit": "",
            "source": "Line 567, 631: SUS316 BODY"
        },

        # 기타 사양들 (정답을 모르는 경우 None으로 표시)
        "OUTPUT_SECONDARY INSULATION SPACE": None,
        "QUANTITY_SECONDARY INSULATION SPACE": None,
        "TYPE_SECONDARY INSULATION SPACE": None,
        "VOLTAGE_SECONDARY INSULATION SPACE": None,
        "POWER SOURCE_SECONDARY INSULATION SPACE": None,
        "MOTOR SHAFT SPEED_SECONDARY INSULATION SPACE": None,
        "STARTING METHOD_SECONDARY INSULATION SPACE": None,
        "ENCLOSURE_SECONDARY INSULATION SPACE": None,
        "CAPACITY_SECONDARY INSULATION SPACE": None,
        "HEAD_SECONDARY INSULATION SPACE": None,
        "SEAL_SECONDARY INSULATION SPACE": None,
    }
}


def normalize_value(value: str) -> str:
    """값 정규화 (비교를 위해)"""
    if not value:
        return ""

    # 공백 정규화
    value = re.sub(r'\s+', ' ', value.strip())

    # 범위 표기 정규화
    value = value.replace('~', '-').replace('～', '-')

    return value.upper()


def compare_extraction(extracted: Dict, ground_truth: Dict) -> Tuple[int, int, int, List[Dict]]:
    """
    추출 결과와 정답 비교

    Returns:
        (correct_count, total_count, missing_count, details)
    """
    correct = 0
    total = 0
    missing = 0
    details = []

    for spec_key, truth in ground_truth.items():
        if truth is None:
            # 정답을 모르는 항목은 제외
            continue

        total += 1
        extracted_item = extracted.get(spec_key, {})
        extracted_value = extracted_item.get("value", "")
        extracted_unit = extracted_item.get("unit", "")

        truth_value = truth.get("value", "")
        truth_unit = truth.get("unit", "")

        # 값 비교 (정규화 후)
        value_match = normalize_value(extracted_value) == normalize_value(truth_value)

        # 단위 비교 (정규화 후)
        unit_match = normalize_value(extracted_unit) == normalize_value(truth_unit)

        is_correct = value_match and unit_match

        if is_correct:
            correct += 1
        elif not extracted_value:
            missing += 1

        details.append({
            "spec": spec_key,
            "correct": is_correct,
            "missing": not extracted_value,
            "expected_value": truth_value,
            "expected_unit": truth_unit,
            "extracted_value": extracted_value,
            "extracted_unit": extracted_unit,
            "source": truth.get("source", "")
        })

    return correct, total, missing, details


def print_accuracy_report(correct: int, total: int, missing: int, details: List[Dict]):
    """정확도 리포트 출력"""
    accuracy = (correct / total * 100) if total > 0 else 0

    print("\n" + "=" * 80)
    print("추출 정확도 테스트 결과")
    print("=" * 80)
    print(f"전체 사양 항목: {total}개")
    print(f"정확히 추출: {correct}개 ({correct/total*100:.1f}%)" if total > 0 else "정확히 추출: 0개")
    print(f"값 누락: {missing}개 ({missing/total*100:.1f}%)" if total > 0 else "값 누락: 0개")
    print(f"값 오류: {total - correct - missing}개")
    print(f"\n최종 정확도: {accuracy:.1f}%")
    print("=" * 80)

    # 상세 결과
    print("\n상세 결과:")
    print("-" * 80)

    for detail in details:
        status = "✓" if detail["correct"] else ("✗ (누락)" if detail["missing"] else "✗ (오류)")
        print(f"\n{status} {detail['spec']}")
        print(f"  정답: {detail['expected_value']} {detail['expected_unit']}")
        print(f"  추출: {detail['extracted_value']} {detail['extracted_unit']}")
        if detail['source']:
            print(f"  출처: {detail['source']}")


if __name__ == "__main__":
    print("=" * 80)
    print("v61 추출 정확도 테스트")
    print("=" * 80)
    print("\n이 스크립트는 추출 결과를 JSON 파일로부터 읽어 정확도를 측정합니다.")
    print("실제 추출을 수행하려면 v61_extractor.py를 실행하세요.")
    print("\n정답지 항목 수:", sum(1 for v in GROUND_TRUTH["2597-POS-0039001_000_01_A4_WATER_DRAIN_SYS_FINAL.html"].values() if v is not None))

    # 예시: JSON 결과 파일이 있다면 로드
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            results = json.load(f)

        # 추출 결과를 딕셔너리로 변환
        extracted = {}
        for item in results:
            spec_key = f"{item['spec_name']}_{item.get('equipment', 'UNKNOWN')}"
            extracted[spec_key] = {
                "value": item.get("value", ""),
                "unit": item.get("unit", "")
            }

        # 정확도 비교
        correct, total, missing, details = compare_extraction(
            extracted,
            GROUND_TRUTH["2597-POS-0039001_000_01_A4_WATER_DRAIN_SYS_FINAL.html"]
        )

        print_accuracy_report(correct, total, missing, details)
    else:
        print("\n사용법: python test_extraction_accuracy.py <extraction_results.json>")
        print("\n정답지 내용:")
        for spec_key, truth in GROUND_TRUTH["2597-POS-0039001_000_01_A4_WATER_DRAIN_SYS_FINAL.html"].items():
            if truth:
                print(f"  {spec_key}: {truth['value']} {truth['unit']}")
