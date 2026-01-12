#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON 콘솔 출력 예시

PRINT_JSON = True로 설정하면 파일 저장과 동시에 콘솔에도 JSON을 출력합니다.
"""

from v53_extractor import POSExtractorV52, build_config

# 방법 1: 전역 설정 변경
# v53_extractor.py 파일에서:
# PRINT_JSON = True  # 기본값

# 방법 2: Config 객체로 직접 설정
config = build_config()
config.print_json = True  # 콘솔 출력 활성화
# config.print_json = False  # 콘솔 출력 비활성화

# 추출 실행
extractor = POSExtractorV52(config=config)
extractor.initialize()

# Light 모드 (단건 추출)
# result = extractor.extract_single(html_path, spec)
# → JSON이 콘솔에 출력됩니다 (print_json=True인 경우)

# Verify 모드
# EXTRACTION_MODE = "verify"로 설정 후:
# results = extractor.verify_full()
# → 검증 결과 JSON이 콘솔에 출력됩니다

print("""
# 출력 예시:

================================================================================
JSON 추출 결과 (Pretty Print)
================================================================================
[
  {
    "doknr": "2597-POS-0039001",
    "items": [
      {
        "pmg_desc": "PUMP",
        "umgv_desc": "CAPACITY",
        "pos_umgv_value": "20",
        "pos_umgv_uom": "m³/H",
        "umgv_value_edit": "",
        "evidence_fb": "",
        ...
      }
    ]
  }
]
================================================================================

또는 Verify 모드:

================================================================================
Verify 모드 검증 결과 (Pretty Print)
================================================================================
[
  {
    "umgv_desc": "CAPACITY",
    "umgv_value": "20000",
    "umgv_uom": "L/H",
    "pos_umgv_value": "20",
    "pos_umgv_uom": "m³/h",
    "verification_status": "MATCHED",
    "verification_confidence": 1.0,
    "verification_reason": "완전 일치 (m³/h → l/h)"
  }
]
================================================================================
""")
