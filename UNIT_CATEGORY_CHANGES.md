# 단위 변환 제거 및 단위 카테고리 매칭 구현

## 변경 배경

### 문제점
- 단위 변환 중 값이 틀리면 더 큰 문제 발생
- 추출된 값을 Ctrl+F로 문서에서 찾을 수 없음
- 자동 변환으로 인한 신뢰성 저하

### 해결 방안
- **추출**: POS 문서에 적힌 그대로 추출 (단위 변환 제거)
- **검증**: 단위 카테고리가 일치하는지만 확인
- **LLM 이해**: inch와 cm은 같은 "길이" 단위임을 LLM이 인지

## 주요 변경사항

### 1. 단위 카테고리 정의

```python
# 단위 카테고리 매핑
UNIT_CATEGORIES = {
    # 길이
    'LENGTH': ['m', 'mm', 'cm', 'km', 'inch', 'ft', 'yard', 'mile', 'nm'],

    # 무게
    'WEIGHT': ['kg', 'g', 'ton', 'mt', 'lb', 'oz'],

    # 압력
    'PRESSURE': ['bar', 'bar.g', 'MPa', 'kPa', 'psi', 'mbar', 'mmHg', 'kgf/cm2'],

    # 온도
    'TEMPERATURE': ['°C', 'C', 'OC', 'K', '°F', 'F'],

    # 유량
    'FLOW_RATE': ['m3/h', 'm³/h', 'L/h', 'L/min', 'GPM', 'kg/h'],

    # 부피
    'VOLUME': ['m3', 'm³', 'L', 'mL', 'gallon', 'ft3'],

    # 전력/에너지
    'POWER': ['kW', 'W', 'MW', 'HP', 'PS'],
    'ENERGY': ['kWh', 'MWh', 'J', 'kJ', 'MJ'],

    # 속도
    'SPEED': ['rpm', 'RPM', 'r/min', 'm/s', 'km/h', 'knot'],

    # 비율/농도
    'PERCENTAGE': ['%', 'percent', 'ppm', 'mol%', 'wt%', 'vol%'],

    # 전기
    'VOLTAGE': ['V', 'kV', 'mV'],
    'CURRENT': ['A', 'mA', 'kA'],
    'FREQUENCY': ['Hz', 'kHz', 'MHz'],
}
```

### 2. 단위 카테고리 매칭 함수

```python
def get_unit_category(unit: str) -> Optional[str]:
    """단위의 카테고리 반환"""
    unit_normalized = unit.strip().upper()

    for category, units in UNIT_CATEGORIES.items():
        for u in units:
            if u.upper() == unit_normalized:
                return category
    return None

def units_same_category(unit1: str, unit2: str) -> bool:
    """두 단위가 같은 카테고리인지 확인"""
    cat1 = get_unit_category(unit1)
    cat2 = get_unit_category(unit2)

    if cat1 and cat2:
        return cat1 == cat2
    return False
```

### 3. LLM 프롬프트 수정

#### Before (단위 변환 지시)
```
**예상단위**: {spec.expected_unit}

## 작업
위 문서에서 각 사양의 값을 찾아 추출하세요.
```

#### After (원문 그대로 추출 + 카테고리 힌트)
```
**예상단위 카테고리**: {unit_category} (예: inch, cm, mm 모두 길이 단위)
**참고 단위**: {spec.expected_unit}

## 작업
위 문서에서 각 사양의 값을 찾아 추출하세요.

**중요**:
- 값과 단위를 문서에 적힌 그대로 추출하세요
- 단위를 변환하지 마세요 (예: inch를 cm로 변환 금지)
- 참고 단위와 다른 단위여도, 같은 카테고리면 정상입니다
  (예: 참고단위=inch, 추출값=5 cm → 정상, 둘 다 길이 단위)
```

### 4. _normalize_unit_in_result() 메서드 수정

#### Before (단위 변환 수행)
```python
def _normalize_unit_in_result(self, result, hint):
    """단위 정규화"""
    if not result.unit or not self.unit_normalizer:
        return result

    original_unit = result.unit
    normalized_unit = self.unit_normalizer.normalize(original_unit)

    if normalized_unit != original_unit:
        result.unit = normalized_unit  # 단위 변환
        result.confidence += 0.05
```

#### After (카테고리 검증만 수행)
```python
def _validate_unit_category(self, result, hint):
    """단위 카테고리 검증 (변환 없음)"""
    if not result.unit or not hint or not hint.umgv_uom:
        return result

    # 단위 변환 제거 - 원문 그대로 유지
    extracted_unit = result.unit
    expected_unit = hint.umgv_uom

    # 카테고리 일치 확인만 수행
    if units_same_category(extracted_unit, expected_unit):
        result.confidence = min(1.0, result.confidence + 0.1)
        result.evidence += f" (단위 카테고리 일치: {get_unit_category(extracted_unit)})"
        self.log.debug(
            f"Unit category match: '{extracted_unit}' and '{expected_unit}' "
            f"are both {get_unit_category(extracted_unit)}"
        )
    else:
        # 카테고리 불일치 경고 (오류는 아님)
        cat1 = get_unit_category(extracted_unit)
        cat2 = get_unit_category(expected_unit)
        if cat1 != cat2:
            self.log.warning(
                f"Unit category mismatch: '{extracted_unit}' ({cat1}) "
                f"vs expected '{expected_unit}' ({cat2})"
            )

    return result
```

### 5. ExtractionHint에 카테고리 정보 추가

```python
@dataclass
class ExtractionHint:
    """추출 힌트 데이터"""
    spec_name: str = ""

    # 용어집 힌트
    umgv_uom: str = ""                  # 예상 단위
    umgv_uom_category: str = ""         # NEW: 예상 단위 카테고리
    pos_umgv_uom: str = ""
    pos_umgv_uom_category: str = ""     # NEW: POS 단위 카테고리
```

## 예상 효과

### 정확성 향상
- ✅ Ctrl+F로 추출값 검색 가능
- ✅ 단위 변환 오류 제거
- ✅ 문서 원문과 100% 일치

### LLM 이해도 향상
```
Before:
  문서: "5 inch"
  예상단위: cm
  LLM: ❌ "단위가 다르므로 잘못된 chunk"

After:
  문서: "5 inch"
  예상단위 카테고리: LENGTH (참고: cm)
  LLM: ✅ "inch와 cm은 같은 길이 단위이므로 올바른 chunk"
  추출: "5 inch" (원문 그대로)
```

### 신뢰성 향상
- 사용자가 추출값을 POS 문서에서 직접 확인 가능
- 단위 변환 오류로 인한 데이터 손실 방지

## 구현 체크리스트

- [ ] UNIT_CATEGORIES 정의
- [ ] get_unit_category() 함수 구현
- [ ] units_same_category() 함수 구현
- [ ] _normalize_unit_in_result() → _validate_unit_category()로 변경
- [ ] LLM 프롬프트 수정 (배치, 단일 모두)
- [ ] ExtractionHint에 카테고리 필드 추가
- [ ] 테스트 및 검증
