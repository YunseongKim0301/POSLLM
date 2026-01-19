#!/usr/bin/env python3
"""
간단한 HTML 파싱 및 추출 테스트
"""

import sys
from pathlib import Path

from v61_extractor import HTMLChunkParser

def test_html_parsing():
    """HTML 파싱 테스트"""

    test_html = "/home/user/POSLLM/2550-POS-0077601_001_02_A4(16).html"

    if not Path(test_html).exists():
        print(f"❌ Error: HTML file not found: {test_html}")
        return

    print(f"📄 Testing HTML parsing: {Path(test_html).name}")
    print("=" * 80)

    try:
        # HTML 파싱
        print("\n🔧 Parsing HTML...")
        parser = HTMLChunkParser(file_path=test_html)
        print("✅ HTML parsed successfully")

        # Key-Value 쌍 출력
        print(f"\n📊 Found {len(parser.kv_pairs)} key-value pairs")

        # 샘플 출력
        print("\n📝 Sample key-value pairs (first 30):")
        print("=" * 80)
        for i, kv in enumerate(parser.kv_pairs[:30], 1):
            key = kv['key'][:50]
            value = str(kv['value'])[:50] if kv['value'] else "(empty)"
            print(f"{i:2d}. {key:50s} = {value}")

        # 특정 키워드 포함한 KV 찾기
        print("\n🔍 KV pairs containing 'Capacity':")
        print("=" * 80)
        capacity_kvs = [kv for kv in parser.kv_pairs 
                        if 'capacity' in kv['key'].lower() or 
                           ('value' in kv and kv['value'] and 'capacity' in str(kv['value']).lower())]
        for i, kv in enumerate(capacity_kvs[:10], 1):
            print(f"{i}. {kv['key']} = {kv['value']}")

        # 테이블 정보
        print(f"\n📊 Found {len(parser.tables)} tables in the document")
        if parser.tables:
            print(f"   First table has {len(parser.tables[0])} rows")
            if parser.tables[0]:
                print(f"   First row has {len(parser.tables[0][0])} columns")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_html_parsing()
