# v61 추출 정확도 테스트 가이드

## 개선 사항 요약

다음 4가지 주요 문제를 해결했습니다:

### 1. 범위 파싱 로직 개선 (v61_extractor.py:7271-7340)
- LLM 프롬프트에서 MAX/MIN 키워드 자동 감지
- 동적으로 범위 파싱 지시사항 생성
- 예시:
  - "MAX. SERVICE TEMPERATURE" + "10-55°C" → "55" 추출
  - "MIN. SERVICE TEMPERATURE" + "10-55°C" → "10" 추출

### 2. MAX/MIN 키워드 기반 chunk 선택 강화 (v61_extractor.py:5526-5538)
- ChunkQualityScorer에 MAX/MIN 검증 로직 추가
- 사양명과 chunk의 MAX/MIN 불일치 시 강력한 페널티 (-0.5)
- 예시: "MAX. SERVICE TEMPERATURE"는 "Minimum cargo temperature" chunk 선택 방지

### 3. HTML 단위 정규화 (v61_extractor.py:3857-3883)
- HTML 파싱 전에 단위 표기 정규화
- 변환 패턴:
  - `<sup>O</sup>C` → `°C`
  - `O C` → `°C`
  - `OC` → `°C` (단어 경계에서만)

### 4. 텍스트 값 추출 개선 (v61_extractor.py:6138-6181)
- 괄호 안의 값 추출: `(-163°C)` → `-163`, `°C`
- 텍스트+숫자 혼합: `SUS316 BODY`, `5 ~ 8 bar`
- 단위 자동 분리: `5 ~ 8 bar` → value: `5 ~ 8`, unit: `bar`

## 테스트 실행 방법

### 1단계: Astrago 환경에서 추출 실행

```bash
cd /workspace/codes
python v61_extractor.py
```

실행 결과는 JSON 파일로 저장됩니다 (예: `extraction_results_20260116.json`).

### 2단계: 정확도 측정

```bash
# GitHub에서 테스트 스크립트 가져오기
cd /workspace
git pull origin claude/robust-spec-detection-35Ndd

# 정확도 측정
python test_extraction_accuracy.py extraction_results_20260116.json
```

출력 예시:
```
================================================================================
추출 정확도 테스트 결과
================================================================================
전체 사양 항목: 7개
정확히 추출: 6개 (85.7%)
값 누락: 0개 (0.0%)
값 오류: 1개

최종 정확도: 85.7%
================================================================================

상세 결과:
--------------------------------------------------------------------------------
✓ MIN. SERVICE TEMPERATURE_SECONDARY INSULATION SPACE
  정답: -163 °C
  추출: -163 °C
  출처: Line 462: (-163°C) Minimum cargo temperature

✗ (오류) MAX. SERVICE TEMPERATURE_SECONDARY INSULATION SPACE
  정답: 70 °C
  추출: -163 °C
  출처: Line 492: -20 to 70 O C Open deck
...
```

## 정답지 (Ground Truth)

`test_extraction_accuracy.py`에 다음 7개 사양의 정답이 포함되어 있습니다:

| 사양명 | 정답 값 | 단위 | HTML 출처 |
|--------|---------|------|-----------|
| MIN. SERVICE TEMPERATURE_SECONDARY INSULATION SPACE | -163 | °C | Line 462 |
| MAX. SERVICE TEMPERATURE_SECONDARY INSULATION SPACE | 70 | °C | Line 492 |
| MAX. SERVICE TEMPERATURE_MOTOR ROOM | 55 | °C | Line 491 |
| MAX. SEA WATER TEMPERATURE_SECONDARY INSULATION SPACE | 33 | °C | Line 436-437 |
| MAX. AIR TEMPERATURE_MOTOR ROOM | 50 | °C | Line 445-446 |
| COMPRESSED AIR_SECONDARY INSULATION SPACE | 5 ~ 8 | bar | Line 563 |
| CASING MATERIAL_SECONDARY INSULATION SPACE | SUS316 | - | Line 567, 631 |

## 90% 달성을 위한 반복 개선 프로세스

만약 정확도가 90% 미만이면:

### 1. 문제 분석
- 어떤 항목이 실패했는지 확인
- 실패 원인 분류:
  - chunk 선택 오류
  - 값 파싱 오류
  - LLM 거부
  - 단위 정규화 오류

### 2. 해결 방법 설계
- chunk selection 로직 조정
- LLM 프롬프트 개선
- 정규화 패턴 추가
- 키워드 매칭 강화

### 3. 코드 개선
- v61_extractor.py 수정
- GitHub에 커밋 및 push

### 4. 재테스트
- Astrago에서 v61_extractor.py 업데이트
- 다시 실행 및 정확도 측정

### 5. 반복
- 90% 달성할 때까지 1-4 반복

## 추가 테스트 데이터

GitHub에 10개의 추가 HTML 파일이 있습니다:
- 2550-POS-0077601_001_02_A4(16).html
- 2574-POS-0060101_000_02_A4(30).html
- 2598-POS-0070307_000_02_A4(28)_FO_SUPPLY_MODULE.html
- 2606-POS-0036329_001_00_A4(22)_FRS.html
- 2606-POS-0037601_001_02_A4(27)_INERT_GAS_SYSTEM.html
- 2606-POS-0057101_001_02_A4(27).html
- 2606-POS-0065101_000_00_A4(28).html
- 2606-POS-0094102_000_02_A4_CONSOLE.html
- 2606-POS-0094303_000_00_A4(19).html
- 2606-POS-0096121_000_01_A4(45).html

이 파일들로 추가 테스트를 수행하여 일반화 성능을 검증할 수 있습니다.

## 성능 목표

- **정확도**: 90% 이상
- **실행 시간**: 항목당 5초 이하 (현재 11초)
- **테스트 범위**: 11개 HTML 파일

## 문제 발생 시

문제가 발생하면 다음 정보를 제공해주세요:
1. 실행 로그 (전체 또는 오류 부분)
2. 추출 결과 JSON 파일
3. 정확도 측정 결과
4. 실패한 사양 항목의 상세 정보

이 정보를 바탕으로 추가 개선을 진행하겠습니다.
