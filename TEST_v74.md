# v74_extractor.py 테스트 가이드

## 실행 방법

```bash
# v71, v73과 동일한 방식으로 실행
/usr/bin/python v74_extractor.py
```

## 테스트 파일
- `2530-POS-0033101_001_00_A4{12}.html`

## 테스트 대상 사양항목 (3개)

### 1. QUANTITY (Y0330)
- **umgv_desc**: `QUANTITY`
- **HTML 표기**: `Q'ty | One(1) set per ship`
- **기대 결과**:
  - `pos_umgv_desc`: `Q'ty` ✓
  - `pos_umgv_value`: `One(1)` ✓
  - `pos_umgv_uom`: `set per ship` ✓

### 2. CAPACITY(SWL) (Y0064)
- **umgv_desc**: `CAPACITY(SWL)`
- **HTML 표기**: `Hoisting capacity | SWL 6 tonnes`
- **기대 결과**:
  - `pos_umgv_desc`: `Hoisting capacity | SWL` ✓
  - `pos_umgv_value`: `6` ✓
  - `pos_umgv_uom`: `tonnes` ✓

### 3. MAX. WORKING RADIUS (Y0255)
- **umgv_desc**: `MAX. WORKING RADIUS`
- **HTML 표기**: `Working radius | Maximum | 19 m`
- **기대 결과**:
  - `pos_umgv_desc`: `Working radius | Maximum` ✓
  - `pos_umgv_value`: `19` ✓
  - `pos_umgv_uom`: `m` ✓

## 100% 정확도 검증 기준

### Rule 기반 추출 성공률
- 3개 항목 모두 Rule 기반 추출 성공 → 100%
- 또는 최소 2개 Rule 성공 + 1개 LLM Fallback 성공 → 100%

### pos_umgv_desc 정확도
- **QUANTITY**: `Q'ty` (짧은 키워드, 정확한 매칭)
- **CAPACITY**: `Hoisting capacity | SWL` (계층 구조 보존)
- **MAX. WORKING RADIUS**: `Working radius | Maximum` (계층 구조 보존)

### 성공 지표
```
총 항목: 3
Rule 성공: 3 (100%) ← 목표
LLM Fallback: 0 (0%)
실패: 0 (0%)
```

## v71 vs v73 vs v74 비교

| 항목 | v71 | v73 | v74 (목표) |
|------|-----|-----|------------|
| QUANTITY | ✅ `Q'ty` | ❌ 실패 | ✅ `Q'ty` |
| CAPACITY | ⚠️ `CAPACITY(SWL)` | ✅ `Hoisting capacity \| SWL` | ✅ `Hoisting capacity \| SWL` |
| MAX. WORKING RADIUS | ⚠️ `Maximum` | ✅ `Working radius \| Maximum` | ✅ `Working radius \| Maximum` |
| **정확도** | **33%** (1/3) | **67%** (2/3) | **100%** (3/3) |

## 로그 확인 포인트

### 1. 짧은 키워드 감지
```
[DEBUG] Short keyword detected: 'Q'ty' - minimal expansion
```

### 2. 후보 선택 로그
```
[DEBUG] Selected candidate: 'Q'ty' (strategy=minimal, score=4.2)
```
- QUANTITY는 minimal 전략, 높은 점수 (짧은 키워드 보너스)

```
[DEBUG] Selected candidate: 'Hoisting capacity | SWL' (strategy=delimiter, score=3.5)
```
- CAPACITY는 delimiter 전략, 계층 구조 보존

```
[DEBUG] Selected candidate: 'Working radius | Maximum' (strategy=delimiter, score=3.4)
```
- MAX. WORKING RADIUS는 delimiter 전략, 계층 구조 보존

### 3. 최종 추출 통계
```
총 항목: 3
Rule 성공: 3 (100%)  ← 이것이 목표!
LLM Fallback: 0 (0%)
실패: 0 (0%)
```

## 트러블슈팅

### QUANTITY 실패 시
- **원인**: 키워드 확장이 과도하게 적용됨
- **확인**: `is_short_variant` 감지 로직 확인
- **해결**: `len(v) <= 5` 또는 특수문자 감지 로직 조정

### CAPACITY/RADIUS 실패 시
- **원인**: 계층 구조 보존 실패
- **확인**: `delimiter` 전략 점수가 낮음
- **해결**: 계층 구조 보너스 증가 (`|` 포함 시 +0.5)

## 실행 예시

```bash
# v74 실행
/usr/bin/python v74_extractor.py

# 예상 출력 (일부)
[INFO] Rule 기반 추출 성공 → LLM 검증 시작: QUANTITY
[INFO] Rule 기반 추출 성공 → LLM 검증 시작: CAPACITY(SWL)
[INFO] Rule 기반 추출 성공 → LLM 검증 시작: MAX. WORKING RADIUS
...
[INFO] 추출 통계
[INFO]   총 항목: 3
[INFO]   Rule 성공: 3 (100%)
[INFO]   LLM Fallback: 0 (0%)
[INFO]   실패: 0 (0%)
```

## 최종 JSON 검증

```json
{
  "doknr": "2530-POS-0033101",
  "items": [
    {
      "umgv_desc": "CAPACITY(SWL)",
      "pos_umgv_desc": "Hoisting capacity | SWL",  // ✓ 계층 구조
      "pos_umgv_value": "6",
      "pos_umgv_uom": "tonnes"
    },
    {
      "umgv_desc": "MAX. WORKING RADIUS",
      "pos_umgv_desc": "Working radius | Maximum",  // ✓ 계층 구조
      "pos_umgv_value": "19",
      "pos_umgv_uom": "m"
    },
    {
      "umgv_desc": "QUANTITY",
      "pos_umgv_desc": "Q'ty",  // ✓ 짧은 키워드
      "pos_umgv_value": "One(1)",
      "pos_umgv_uom": "set per ship"
    }
  ]
}
```
