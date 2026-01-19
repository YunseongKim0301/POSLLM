# v74_extractor.py 개선 사항

## 목표
v71과 v73의 장점을 결합하여 **범용적이고 보편적으로 적용 가능한** 추출 코드 작성

## 문제 분석

### v71의 특징
- **장점**: 단순한 키워드 매칭으로 "Q'ty" 같은 짧은 사양명 정확히 추출
- **단점**: 괄호 제거 로직으로 계층 구조 손실
  - `Hoisting capacity | SWL` → `CAPACITY(SWL)` ❌
  - `Working radius | Maximum` → `Maximum` ❌

### v73의 특징
- **장점**: 다중 전략으로 계층 구조 보존
  - `Hoisting capacity | SWL` ✓
  - `Working radius | Maximum` ✓
- **단점**: 과도한 키워드 확장으로 짧은 사양명 처리 실패
  - `Q'ty` → 추출 실패 ❌

## 범용적 해결 방안

### 1. 점수 계산 함수 개선 (`_calculate_candidate_score`)

#### 짧은 키워드 자동 감지
```python
is_short_keyword = len(keyword) <= 4 or any(c in keyword for c in ["'", '"', '-', '.'])
```
- 길이 4자 이하
- 또는 특수문자 포함 (', ", -, .)

#### 점수 계산 로직 개선
1. **길이 점수**: 짧은 키워드는 2-10자에 높은 점수 (1.2), 일반 키워드는 5-50자 선호
2. **정확한 매칭**: 대소문자까지 일치 시 1.0 보너스, 대소문자 무시 매칭은 0.6
3. **유사도 가중치 증가**: spec_name과의 단어 겹침에 1.5배 가중치
4. **계층 구조 보너스**: `|` 포함 시 0.5 보너스
5. **짧은 키워드 정확 매칭 특별 보너스**: 짧은 키워드가 정확히 매칭되면 2.0 추가

### 2. 사양명 추출 함수 개선 (`_extract_original_spec_name_from_chunk`)

#### 짧은 키워드 감지 (범용적 기준)
```python
is_short_variant = any(
    len(v) <= 5 or any(c in v for c in ["'", '"', '-', '.'])
    for v in variants
)
```

#### 키워드 확장 전략
- **짧은 키워드**: 확장 최소화 (원본 변형만 사용)
  - 예: `Q'ty` → `Q'ty`만 검색
- **일반 키워드**: 확장 수행
  - 예: `MAX. WORKING RADIUS` → `MAX. WORKING RADIUS`, `MAXIMUM`, `WORKING RADIUS`, `RADIUS` 등

#### 추출 전략 최적화
- **짧은 키워드**:
  1. Minimal 전략 (매칭된 키워드만)
  2. Delimiter 전략 (계층 구조 포착)
  3. 첫 번째 매칭으로 조기 종료

- **일반 키워드**:
  1. Delimiter (구분자 기반)
  2. Word boundary (단어 경계)
  3. Value pattern (값 패턴)
  4. Grammar (문법 기반)
  5. Minimal (최소 확장)

## 예상 결과

### QUANTITY (짧은 키워드)
- 입력 chunk: `Q'ty\n\n\nOne(1) set per ship`
- 키워드: `Q'ty`
- 감지: 짧은 키워드 ✓ (특수문자 포함)
- 전략: Minimal 우선
- 예상 출력: `Q'ty` ✓

### CAPACITY(SWL) (일반 키워드, 계층 구조)
- 입력 chunk: `Hoisting capacity | SWL 6 tonnes`
- 키워드: `CAPACITY`, `SWL` 등
- 전략: Delimiter (계층 구조 보존)
- 예상 출력: `Hoisting capacity | SWL` ✓

### MAX. WORKING RADIUS (일반 키워드, 계층 구조)
- 입력 chunk: `Working radius | Maximum | 19 m`
- 키워드: `MAXIMUM`, `WORKING RADIUS` 등
- 전략: Delimiter (계층 구조 보존)
- 예상 출력: `Working radius | Maximum` ✓

## 범용성 (Generic Approach)

이 개선안은 **특정 사양항목에 국한되지 않고** 모든 사양항목에 적용 가능:

1. **자동 감지**: 키워드 특성(길이, 특수문자)을 자동으로 분석
2. **적응형 전략**: 키워드 특성에 따라 전략을 자동으로 조정
3. **점수 기반 선택**: 하드코딩된 규칙 없이 점수로 최선의 후보 선택
4. **확장 가능**: 새로운 사양항목이 추가되어도 코드 수정 불필요

## 기대 효과

- QUANTITY: v71처럼 정확히 추출 ✓
- CAPACITY: v73처럼 계층 구조 보존 ✓
- MAX. WORKING RADIUS: v73처럼 계층 구조 보존 ✓
- **모든 사양항목에 범용적으로 적용 가능** ✓
