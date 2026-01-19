#!/usr/bin/env python3
"""
전체 파이프라인 테스트 (LLM 포함)

테스트 항목:
1. NOT_FOUND 처리 확인
2. LLM "다른 표현" 인식 확인
3. 성능 측정 (spec당 시간)
4. v61 vs v70 비교
"""

import os
import sys
import time
import pandas as pd
from typing import Dict, List
import json

sys.path.insert(0, os.path.dirname(__file__))

def test_not_found_handling(extractor_class, version_name: str):
    """
    NOT_FOUND 처리 테스트

    테스트 케이스:
    1. Template에는 있지만 POS에 없는 사양항목
    2. LLM이 찾아야 하는 경우
    """
    print(f"\n{'='*80}")
    print(f"{version_name} NOT_FOUND 테스트")
    print(f"{'='*80}")

    # TODO: 실제 구현
    # 1. Template 로드
    # 2. POS 파일 로드
    # 3. 추출 실행
    # 4. EXTRACTION_FAILED 개수 확인

    print("⚠️  이 테스트는 Ollama + PostgreSQL 환경이 필요합니다")
    print("⚠️  아래 단계를 직접 수행하세요:")
    print()
    print("1. Ollama 시작:")
    print("   $ ollama serve")
    print()
    print("2. PostgreSQL 연결 확인:")
    print("   $ psql -h <host> -U <user> -d <database> -c 'SELECT COUNT(*) FROM pos_dict;'")
    print()
    print("3. v61 추출 실행:")
    print("   $ python v61_extractor.py")
    print()
    print("4. 결과 JSON에서 확인:")
    print("   $ grep 'EXTRACTION_FAILED' extraction_result_*.json | wc -l")


def test_performance(extractor_class, version_name: str, num_specs: int = 100):
    """
    성능 테스트

    Args:
        num_specs: 테스트할 사양 개수

    측정:
        - 평균 시간/spec
        - 300K 추정 시간
        - LLM 호출 횟수
    """
    print(f"\n{'='*80}")
    print(f"{version_name} 성능 테스트 ({num_specs} specs)")
    print(f"{'='*80}")

    print("⚠️  이 테스트는 Ollama + PostgreSQL 환경이 필요합니다")
    print()
    print("수동 테스트 방법:")
    print()
    print(f"1. {num_specs}개 사양 추출:")
    print(f"   $ time python {version_name.lower()}_extractor.py")
    print()
    print("2. 시간 측정:")
    print(f"   총 시간 / {num_specs} = 평균 시간/spec")
    print()
    print("3. 300K 추정:")
    print(f"   평균 시간/spec × 300,000 = 총 예상 시간")
    print()
    print("4. 목표 확인:")
    print("   총 예상 시간 < 3일 (259,200초)")


def compare_v61_v70():
    """v61과 v70 비교"""
    print("\n" + "="*80)
    print("v61 vs v70 비교 테스트")
    print("="*80)

    print("\n비교 항목:")
    print("1. 정확도 (동일한 template + POS로 테스트)")
    print("2. 성능 (spec당 평균 시간)")
    print("3. LLM 호출 횟수")
    print("4. NOT_FOUND 비율")

    print("\n실행 방법:")
    print()
    print("1. 동일한 입력으로 두 버전 실행:")
    print("   $ python v61_extractor.py > v61_output.log 2>&1")
    print("   $ python v70_extractor.py > v70_output.log 2>&1")
    print()
    print("2. 결과 비교:")
    print("   $ python compare_results.py v61_result.json v70_result.json")


def create_test_template():
    """
    테스트용 template 생성

    포함 사양:
    1. POS에 있는 사양 (정확한 표현)
    2. POS에 있는 사양 (다른 표현) - LLM이 찾아야 함
    3. POS에 없는 사양 - NOT_FOUND 예상
    """
    print("\n" + "="*80)
    print("테스트 Template 생성")
    print("="*80)

    test_specs = [
        {
            "umgv_code": "TEST001",
            "umgv_desc": "Output",
            "expected": "FOUND",
            "note": "POS에 정확히 있음"
        },
        {
            "umgv_code": "TEST002",
            "umgv_desc": "Maximum Speed",
            "expected": "FOUND via LLM",
            "note": "POS에 'Max. velocity'로 있음 (다른 표현)"
        },
        {
            "umgv_code": "TEST003",
            "umgv_desc": "Fuel Tank Capacity",
            "expected": "NOT_FOUND",
            "note": "POS에 없음"
        }
    ]

    print("\n테스트 사양 목록:")
    for spec in test_specs:
        print(f"\n{spec['umgv_code']}: {spec['umgv_desc']}")
        print(f"  예상: {spec['expected']}")
        print(f"  비고: {spec['note']}")

    # Save to TSV
    df = pd.DataFrame(test_specs)
    output_path = "test_template.tsv"
    df.to_csv(output_path, sep='\t', index=False)
    print(f"\n✓ 저장됨: {output_path}")

    return test_specs


def main():
    """메인 테스트 실행"""
    print("="*80)
    print("전체 파이프라인 테스트")
    print("="*80)
    print()
    print("⚠️  중요: 이 테스트는 다음 환경이 필요합니다:")
    print("   1. Ollama 실행 중 (gemma3:27b 모델)")
    print("   2. PostgreSQL 연결")
    print("   3. pos_dict, umgv_fin 테이블 데이터")
    print()

    # Check environment
    import subprocess

    print("환경 확인...")
    print()

    # Check Ollama
    print("1. Ollama 상태:")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✓ Ollama 실행 중")
            if 'gemma3:27b' in result.stdout or 'qwen2.5:32b' in result.stdout:
                print("   ✓ LLM 모델 있음")
            else:
                print("   ✗ LLM 모델 없음 (gemma3:27b 또는 qwen2.5:32b 필요)")
        else:
            print("   ✗ Ollama 실행 안 됨")
            print("   $ ollama serve  # 로 시작하세요")
    except FileNotFoundError:
        print("   ✗ Ollama 설치 안 됨")
    except Exception as e:
        print(f"   ✗ 오류: {e}")

    print()
    print("2. PostgreSQL 상태:")
    print("   수동 확인 필요:")
    print("   $ psql -h <host> -U <user> -d <database> -c '\\dt'")

    print()
    print("-"*80)
    print()

    # Create test template
    create_test_template()

    # Run tests
    try:
        from v61_extractor import POSExtractorV61
        test_not_found_handling(POSExtractorV61, "v61")
        test_performance(POSExtractorV61, "v61", num_specs=10)
    except Exception as e:
        print(f"\n✗ v61 테스트 실패: {e}")
        print("   Ollama나 PostgreSQL 환경을 확인하세요")

    try:
        from v70_extractor import POSExtractorV61 as POSExtractorV70
        test_not_found_handling(POSExtractorV70, "v70")
        test_performance(POSExtractorV70, "v70", num_specs=10)
    except Exception as e:
        print(f"\n✗ v70 테스트 실패: {e}")
        print("   Ollama나 PostgreSQL 환경을 확인하세요")

    # Compare
    compare_v61_v70()

    print("\n" + "="*80)
    print("테스트 완료")
    print("="*80)
    print()
    print("다음 단계:")
    print("1. Ollama + PostgreSQL 환경 구성")
    print("2. 소규모 테스트 (10-100 specs)")
    print("3. 결과 분석 및 개선")
    print("4. 대규모 테스트 (300K specs)")


if __name__ == "__main__":
    main()
