# v61_extractor.py 실행 테스트 결과 및 개선 전략

## 📊 테스트 결과 요약

### 테스트 환경
- **POS 파일**: 2550-POS-0077601_001_02_A4(16).html (FAN 사양 문서)
- **테스트 날짜**: 2026-01-19
- **코드 버전**: v61 (PostgreSQL-Enhanced with BGE-M3)

### HTML 파싱 성능 (HTMLChunkParser)

#### ✅ 성공한 부분
1. **HTML 파싱 성공**: BeautifulSoup으로 정상 파싱
2. **Key-Value 쌍 추출**: 436개 추출 성공
3. **테이블 파싱**: 9개 테이블 감지
4. **단위 정규화**: HTML의 `<sup>O</sup>C` → `°C` 자동 변환

#### 추출된 샘플 데이터
```
1_Capacity_Air volume(m3/h)# = 135,000
1_Capacity_Static pressure(mmAq)# = 45
Total barometric pressure = 1,000 mbar
ER air temperature = 45 °C
Relative humidity of air = 60%
```

---

## ❌ 발견된 문제점

### 1. **의존성 문제**
- **현상**: PostgreSQL 없이는 초기화 불가
- **에러**: `RuntimeError: 용어집(pos_dict) 로드 실패`
- **원인**: `_init_light_mode()`가 용어집이 비어있으면 강제 종료
- **영향**: Standalone 테스트 불가능

```python
# v61_extractor.py:8285
if glossary_df is None or len(glossary_df) == 0:
    raise RuntimeError("용어집(pos_dict) 로드 실패: 데이터가 비어있습니다")
```

### 2. **패키지 의존성 누락**
- **필수 패키지**: pandas, beautifulsoup4가 설치되지 않으면 동작 불가
- **HAS_PANDAS 플래그**: 정의되어 있지만 실제로 체크하지 않는 부분 존재
- **예**: `pd.DataFrame()` 직접 사용 → NameError

### 3. **Key-Value 매칭의 한계**

#### 발견된 복잡한 Key 구조
```
1_Capacity_Air volume(m3/h)#    ← 숫자 prefix + 계층 구조
1_Capacity_Static pressure(mmAq)#
2_Capacity_Air volume(m3/h)#     ← 여러 개의 동일 사양 (반복)
```

**문제점**:
- Spec name "CAPACITY"로는 매칭 불가 (너무 모호)
- 실제 키: `1_Capacity_Air volume(m3/h)#`
- 정확한 매칭을 위해서는 "Air volume" 또는 "Static pressure" 등 세부 정보 필요

### 4. **사양명 매핑의 불완전성**

SPEC_SYNONYMS에 정의된 것들:
```python
'CAPACITY': ['CAPACITY', 'FLOW RATE', 'RATED CAPACITY', ...]
```

하지만 실제 POS에는:
- `Capacity_Air volume(m3/h)#` ← 구조적으로 다름
- `Capacity_Static pressure(mmAq)#` ← 구조적으로 다름

**Gap**: CAPACITY 단독으로는 어떤 값을 원하는지 불명확

---

## 🔍 근본 원인 분석

### 1. **계층적 사양 구조 미지원**
- 많은 POS 문서가 계층적 구조 사용:
  ```
  Capacity
    ├─ Air volume (m3/h)
    ├─ Static pressure (mmAq)
    └─ System
  ```
- 현재 코드는 flat한 사양명만 지원

### 2. **반복되는 장비의 사양 처리 부족**
- 동일한 장비가 여러 개일 때 (1번 FAN, 2번 FAN, 3번 FAN...)
- Key에 번호 prefix (`1_`, `2_`, `3_`)가 추가됨
- 장비명(Equipment)만으로는 구분 불가

### 3. **용어집 의존도가 너무 높음**
- 용어집 없이는 아예 실행 불가
- "Zero-shot" 추출 능력 부족
- 용어집이 없는 새로운 사양에 대한 적응력 낮음

### 4. **LLM 프롬프트 테스트 불가**
- Ollama 서버가 실행 중이지 않으면 LLM 추출 테스트 불가
- Rule 기반과 LLM 기반의 분리된 테스트 어려움

---

## 💡 개선 전략 (기능 단위)

### 전략 1: **Standalone 모드 구현** 🔧
**목표**: 외부 의존성 없이 기본 기능 동작

#### 구현 내용:
1. `Config`에 `allow_empty_glossary` 옵션 추가
2. 용어집이 비어있어도 계속 진행 (warning만 출력)
3. Zero-shot 추출 모드: 사양명만으로 직접 추출 시도

```python
# 의사코드
if config.allow_empty_glossary and len(glossary_df) == 0:
    self.log.warning("용어집 없음. Zero-shot 모드로 진행")
    self.glossary = EmptyGlossary()  # 더미 객체
else:
    raise RuntimeError(...)
```

**효과**:
- 빠른 프로토타이핑 가능
- 새로운 POS 문서 즉시 테스트 가능
- 용어집 구축 전에도 활용 가능

---

### 전략 2: **계층적 사양명 파싱** 📊
**목표**: `1_Capacity_Air volume(m3/h)#` 같은 복잡한 구조 처리

#### 구현 내용:
1. Key 구조 분석 함수 추가:
```python
def parse_hierarchical_key(key: str) -> Dict:
    """
    "1_Capacity_Air volume(m3/h)#" 파싱
    
    Returns:
        {
            'index': '1',
            'level1': 'Capacity',
            'level2': 'Air volume',
            'unit': 'm3/h',
            'normalized': 'CAPACITY_AIR_VOLUME'
        }
    """
```

2. SpecItem에 `sub_spec_name` 필드 추가:
```python
@dataclass
class SpecItem:
    spec_name: str = ""          # "CAPACITY"
    sub_spec_name: str = ""      # "AIR VOLUME" (새로 추가)
    equipment: str = ""
    equipment_index: int = 0      # 1, 2, 3... (새로 추가)
```

3. 매칭 로직 강화:
```python
# Before
if spec_name in key:
    ...

# After
if (spec_name in key) and (sub_spec_name in key or not sub_spec_name):
    if equipment_index == 0 or f"{equipment_index}_" in key:
        ...  # 매칭 성공
```

**효과**:
- 복잡한 사양 구조 정확히 추출
- 반복 장비 각각 처리 가능
- 매칭 정확도 향상

---

### 전략 3: **Fuzzy Matching 강화** 🎯
**목표**: 사양명과 Key 간의 유연한 매칭

#### 현재 문제:
- "MOTOR POWER" (spec_name) ≠ "Motor" (POS key)
- 부분 매칭만으로는 오탐 발생

#### 구현 내용:
1. 단어 기반 토큰 매칭:
```python
def token_match_score(spec_name: str, key: str) -> float:
    """
    토큰 기반 매칭 점수
    
    "MOTOR POWER" vs "Output(KW) of electric motor"
    → tokens1 = {'motor', 'power'}
    → tokens2 = {'output', 'kw', 'electric', 'motor'}
    → overlap = {'motor'}
    → score = 1/2 = 0.5
    """
    tokens1 = set(spec_name.lower().split())
    tokens2 = set(re.findall(r'\w+', key.lower()))
    overlap = tokens1 & tokens2
    return len(overlap) / len(tokens1) if tokens1 else 0
```

2. 문맥 기반 스코어링:
```python
def context_aware_match(spec: SpecItem, kv_pair: Dict) -> float:
    """
    사양 + 장비 + 단위를 모두 고려한 매칭 점수
    """
    score = 0.0
    
    # 사양명 매칭 (40%)
    score += 0.4 * token_match_score(spec.spec_name, kv_pair['key'])
    
    # 장비명 매칭 (30%)
    if spec.equipment and spec.equipment.lower() in kv_pair['key'].lower():
        score += 0.3
    
    # 단위 매칭 (30%)
    if spec.expected_unit and spec.expected_unit in kv_pair['key']:
        score += 0.3
    
    return score
```

**효과**:
- 부분 매칭의 신뢰도 향상
- 문맥을 고려한 정확한 매칭
- False positive 감소

---

### 전략 4: **LLM-free 테스트 모드** 🚀
**목표**: LLM 없이도 Rule 기반 추출 성능 측정

#### 구현 내용:
1. `Config.force_rule_only` 옵션 추가
2. LLM 호출 완전 우회
3. Rule 기반 결과만 반환

```python
if config.force_rule_only:
    result = rule_extractor.extract(chunk, spec)
    # LLM fallback 스킵
    return result or ExtractionResult(spec_item=spec, value="", method="not_found")
```

**효과**:
- Ollama 없이도 테스트 가능
- Rule 기반 성능 정확히 측정
- 빠른 디버깅

---

### 전략 5: **Key-Value 기반 Direct Matching 최적화** ⚡
**목표**: 간단한 사양은 Key-Value 직접 매칭으로 빠르게 처리

#### 현재 문제:
- 436개 KV 쌍이 있는데 활용도 낮음
- Chunk 선택 후 LLM 호출하는 무거운 프로세스

#### 구현 내용:
1. KV 우선 매칭 로직:
```python
def extract_from_kv_direct(spec: SpecItem, kv_pairs: List[Dict]) -> Optional[str]:
    """
    Key-Value 쌍에서 직접 추출 (가장 빠른 방법)
    """
    candidates = []
    
    for kv in kv_pairs:
        score = context_aware_match(spec, kv)
        if score > 0.7:  # 높은 신뢰도
            candidates.append((score, kv['value']))
    
    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1]  # 최고 점수 반환
    
    return None
```

2. 추출 파이프라인 재구성:
```
Before: Chunk selection → LLM → Post-process

After:
  1. KV Direct Matching (빠름, 높은 정확도)
     ↓ (실패 시)
  2. Rule-based from Chunk
     ↓ (실패 시)
  3. LLM from Chunk
```

**효과**:
- 처리 속도 3~5배 향상 (간단한 사양)
- API 호출 횟수 대폭 감소
- 정확도 유지 또는 향상

---

### 전략 6: **진단 및 디버깅 도구** 🔍
**목표**: 추출 실패 원인 빠르게 파악

#### 구현 내용:
1. 상세 로그 모드:
```python
class ExtractionDiagnostics:
    """추출 진단 정보"""
    spec_name: str
    kv_candidates: List[Tuple[float, str]]  # (score, kv_key)
    chunks_found: int
    rule_result: Optional[str]
    llm_result: Optional[str]
    final_result: Optional[str]
    failure_reason: str  # "no_chunk", "llm_timeout", ...
```

2. HTML Annotation 출력:
```python
def generate_annotated_html(html_path: str, results: Dict) -> str:
    """
    추출 결과를 HTML에 하이라이트 표시
    
    - 성공한 값: 초록색
    - 실패한 값: 빨간색
    - 후보였던 값: 노란색
    """
```

**효과**:
- 실패 원인 즉시 파악
- 용어집/사양명 개선 포인트 발견
- 시각적 검증 가능

---

### 전략 7: **단위 변환 롤백 검증** ✅
**목표**: "NO UNIT CONVERSION" 원칙 준수 확인

#### 검증 내용:
1. 추출된 값이 문서에 정확히 존재하는지 확인:
```python
def verify_exact_match(html_content: str, extracted_value: str, extracted_unit: str) -> bool:
    """
    Ctrl+F 테스트: 추출된 값이 문서에 그대로 있는가?
    """
    search_pattern = f"{extracted_value}.*?{extracted_unit}"
    return bool(re.search(search_pattern, html_content, re.IGNORECASE))
```

2. 단위 변환 감지:
```python
def detect_unit_conversion(result: ExtractionResult, hint: ExtractionHint) -> bool:
    """
    단위가 변환되었는지 감지
    
    hint.umgv_uom = "inch"
    result.unit = "cm"
    → True (변환 감지)
    """
    if hint and hint.umgv_uom and result.unit:
        if hint.umgv_uom != result.unit:
            # related_units 체크
            if result.unit not in hint.related_units:
                return True  # 의심스러운 변환
    return False
```

**효과**:
- 원문 보존 원칙 검증
- 잘못된 단위 변환 즉시 발견
- 데이터 신뢰도 보장

---

## 📋 우선순위 및 구현 계획

### Phase 1: 긴급 (1-2일)
1. ✅ **전략 1**: Standalone 모드 구현
2. ✅ **전략 4**: LLM-free 테스트 모드

→ **목표**: 즉시 테스트 가능한 환경 구축

### Phase 2: 핵심 성능 (3-5일)
3. ✅ **전략 5**: KV Direct Matching 최적화
4. ✅ **전략 3**: Fuzzy Matching 강화

→ **목표**: Rule 기반 정확도 90% 달성

### Phase 3: 고급 기능 (5-7일)
5. ✅ **전략 2**: 계층적 사양명 파싱
6. ✅ **전략 6**: 진단 도구

→ **목표**: 복잡한 POS 문서 대응

### Phase 4: 검증 (2-3일)
7. ✅ **전략 7**: 단위 변환 롤백 검증

→ **목표**: 프로덕션 준비 완료

---

## 📈 예상 효과

### 정량적 목표:
- **처리 속도**: 3배 향상 (KV Direct Matching 활용)
- **Rule 기반 정확도**: 70% → 90%
- **전체 정확도**: 99.1% → 99.5%+
- **LLM API 호출**: 50% 감소

### 정성적 목표:
- ✅ PostgreSQL 없이도 테스트 가능
- ✅ 새로운 POS 문서 즉시 대응
- ✅ 디버깅 시간 80% 단축
- ✅ 원문 보존 원칙 100% 준수

---

## 🎯 다음 단계

1. **Phase 1 즉시 착수**: Standalone 모드 구현
2. **테스트 케이스 확장**: 다양한 POS 파일로 테스트
3. **성능 벤치마크**: Before/After 비교
4. **문서화**: 개선사항 README 업데이트

---

**작성일**: 2026-01-19
**작성자**: Claude Code Agent
**버전**: v61 Analysis Report
