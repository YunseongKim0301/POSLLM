#!/usr/bin/env python3
"""
v76 extractor의 파싱 로직 테스트 (DB 연결 없이)
"""
import sys
import os
import re

# v76_extractor를 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_parse_value_unit():
    """_parse_value_unit 함수 테스트"""
    print("=" * 70)
    print("Test: _parse_value_unit")
    print("=" * 70)

    # Test cases
    test_cases = [
        ("SWL 6 tonnes", ("6", "ton")),
        ("Maximum 19 m", ("19", "m")),
        ("One(1) set", ("One(1) set", "")),
        ("6 tonnes", ("6", "ton")),
        ("19 m", ("19", "m")),
        ("6", ("6", "")),
        ("(-163°C)", ("-163", "°C")),
    ]

    # RuleBasedExtractor의 _parse_value_unit을 테스트
    from v76_extractor import RuleBasedExtractor

    extractor = RuleBasedExtractor()

    for raw_value, expected in test_cases:
        result = extractor._parse_value_unit(raw_value)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{raw_value}' → {result} (expected: {expected})")

def test_chunk_parsing():
    """Chunk에서 값 추출 테스트"""
    print("\n" + "=" * 70)
    print("Test: Chunk Parsing")
    print("=" * 70)

    # Test case 1: "Hoisting capacity | SWL 6 tonnes"
    chunk1 = "Hoisting capacity | SWL 6 tonnes"
    print(f"\nTest Chunk 1: '{chunk1}'")

    # 파싱 로직 시뮬레이션
    # Pipe로 분리
    parts = chunk1.split('|')
    print(f"  Pipe split: {parts}")

    if len(parts) >= 2:
        desc = parts[0].strip()
        value_part = parts[1].strip()

        print(f"  Description: '{desc}'")
        print(f"  Value part: '{value_part}'")

        # "SWL 6 tonnes"에서 숫자+단위 추출
        # 패턴: 마지막 숫자+단위를 찾음
        match = re.search(r'\b(\d+(?:[.,]\d+)?(?:\s*[~\-]\s*\d+(?:[.,]\d+)?)?)\s*([a-zA-Z°℃%/³²]+(?:\s*[a-zA-Z°℃%/³²]+)*)?$', value_part)
        if match:
            value = match.group(1).strip()
            unit = match.group(2).strip() if match.group(2) else ""
            print(f"  Extracted value: '{value}'")
            print(f"  Extracted unit: '{unit}'")
        else:
            print(f"  No match found")

    # Test case 2: "Q'ty | One(1) set per ship"
    chunk2 = "Q'ty | One(1) set per ship"
    print(f"\nTest Chunk 2: '{chunk2}'")

    parts = chunk2.split('|')
    print(f"  Pipe split: {parts}")

    if len(parts) >= 2:
        desc = parts[0].strip()
        value_part = parts[1].strip()

        print(f"  Description: '{desc}'")
        print(f"  Value part: '{value_part}'")

        # "One(1) set per ship"에서 값 추출
        # "per ship"은 제거하고 "One(1) set"만 추출
        # 패턴: 숫자 포함 텍스트 + "set" 등의 단위
        match = re.match(r'^(.+?)\s+(set|sets|unit|units|piece|pieces)\s*(?:per\s+\w+)?.*$', value_part, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            unit = match.group(2).strip()
            full_value = f"{value} {unit}"
            print(f"  Extracted value: '{full_value}'")
        else:
            print(f"  Fallback: use full value part")
            print(f"  Extracted value: '{value_part}'")

def main():
    test_parse_value_unit()
    test_chunk_parsing()
    print("\n" + "=" * 70)
    print("테스트 완료")
    print("=" * 70)

if __name__ == "__main__":
    main()
