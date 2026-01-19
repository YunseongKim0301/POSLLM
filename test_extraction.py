#!/usr/bin/env python3
"""
v61_extractor.py 실제 테스트 스크립트
POS HTML 파일에서 사양 추출 테스트
"""

import sys
import json
from pathlib import Path

# v61_extractor 임포트
from v61_extractor import POSExtractorV61, SpecItem, build_config

def test_extraction():
    """실제 POS 파일로 추출 테스트"""

    # 1. 테스트할 POS 파일 선택
    test_html = "/home/user/POSLLM/2550-POS-0077601_001_02_A4(16).html"

    if not Path(test_html).exists():
        print(f"❌ Error: HTML file not found: {test_html}")
        return

    print(f"📄 Testing with: {test_html}")
    print("=" * 80)

    # 2. Config 생성 (기본 설정 사용)
    config = build_config()
    config.extraction_mode = "light"  # light mode: PostgreSQL 없이 실행
    config.use_llm = False  # LLM 비활성화 (먼저 Rule 기반만 테스트)

    # 3. Extractor 초기화
    print("\n🔧 Initializing extractor...")
    try:
        extractor = POSExtractorV61(
            glossary_path="",  # 빈 문자열로 설정
            specdb_path="",
            config=config
        )
        print("✅ Extractor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 테스트할 사양 정의
    specs = [
        SpecItem(
            spec_name="CAPACITY",
            equipment="",
            expected_unit="m3/h",
            hull="2550"
        ),
        SpecItem(
            spec_name="MOTOR POWER",
            equipment="",
            expected_unit="kW",
            hull="2550"
        ),
        SpecItem(
            spec_name="VOLTAGE",
            equipment="",
            expected_unit="V",
            hull="2550"
        ),
        SpecItem(
            spec_name="FREQUENCY",
            equipment="",
            expected_unit="Hz",
            hull="2550"
        ),
        SpecItem(
            spec_name="MOTOR RPM",
            equipment="",
            expected_unit="rpm",
            hull="2550"
        ),
    ]

    print(f"\n📊 Testing {len(specs)} specifications:")
    for spec in specs:
        print(f"  - {spec.spec_name} ({spec.expected_unit})")

    # 5. 추출 실행
    print("\n🚀 Starting extraction...")
    print("=" * 80)

    try:
        results = extractor.extract_batch(test_html, specs)

        # 6. 결과 출력
        print("\n📈 Extraction Results:")
        print("=" * 80)

        success_count = 0
        fail_count = 0

        for spec in specs:
            result = results.get(spec.spec_name)

            if result and result.value:
                success_count += 1
                print(f"\n✅ {spec.spec_name}:")
                print(f"   Value: {result.value}")
                print(f"   Unit: {result.unit}")
                print(f"   Confidence: {result.confidence:.2f}")
                print(f"   Method: {result.method}")
                if result.evidence:
                    evidence_preview = result.evidence[:100].replace('\n', ' ')
                    print(f"   Evidence: {evidence_preview}...")
            else:
                fail_count += 1
                print(f"\n❌ {spec.spec_name}: NOT FOUND")

        # 7. 통계 출력
        print("\n" + "=" * 80)
        print("📊 Statistics:")
        print(f"   Total: {len(specs)}")
        print(f"   Success: {success_count} ({success_count/len(specs)*100:.1f}%)")
        print(f"   Failed: {fail_count} ({fail_count/len(specs)*100:.1f}%)")
        print("=" * 80)

        # 8. JSON 파일로 저장
        output_file = "/home/user/POSLLM/test_results.json"
        output_data = {}
        for spec in specs:
            result = results.get(spec.spec_name)
            output_data[spec.spec_name] = {
                "value": result.value if result else "",
                "unit": result.unit if result else "",
                "confidence": result.confidence if result else 0.0,
                "method": result.method if result else "",
                "evidence": result.evidence[:200] if result and result.evidence else ""
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")

        return success_count, fail_count

    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, len(specs)

if __name__ == "__main__":
    test_extraction()
