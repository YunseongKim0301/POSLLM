# v76 Extractor 테스트 가이드

## 주요 개선사항

### 1. 값 파싱 개선 (_parse_value_unit)
- **"SWL 6 tonnes"** → ('6', 'ton') - 앞의 설명 텍스트 제거
- **"Maximum 19 m"** → ('19', 'm') - 형용사 제거
- **"One(1) set"** → ('One(1)', 'set') - QUANTITY 처리
- **"(-163°C)"** → ('-163', '°C') - 음수 지원

### 2. Context 관리 개선
- **우선순위**: result.chunk > chunk_context > get_context_for_value > full_text
- pos_chunk를 LLM 검증에 우선적으로 사용하여 관련성 높은 context 제공

### 3. LLM 검증 프롬프트 개선
- 더 관대하고 실용적인 검증 가이드라인
- QUANTITY 같은 명확한 값에 대한 오탐 방지
- Context가 관련성 높을 때 더 높은 신뢰도 부여

## 실행 방법

### 환경에 맞게 v76_extractor.py 실행

```bash
# 1. v76_extractor.py를 실제 환경에 복사
cp /home/user/POSLLM/v76_extractor.py /workspace/codes/

# 2. LIGHT 모드로 실행
python /workspace/codes/v76_extractor.py

# 또는 기존 실행 방식 그대로 사용
```

## 기대 결과

### CAPACITY(SWL)
- **pos_umgv_desc**: "Hoisting capacity" (✓ 개선)
- **pos_umgv_value**: "6" (✓ 개선)
- **pos_umgv_uom**: "ton" (✓ 유지)
- **pos_chunk**: "Hoisting capacity | SWL 6 tonnes"

### MAX. WORKING RADIUS
- **pos_umgv_desc**: "Working radius | Maximum" 또는 "Maximum" (✓ 개선)
- **pos_umgv_value**: "19" (✓ 개선)
- **pos_umgv_uom**: "m" (✓ 유지)
- **pos_chunk**: "Working radius | Maximum | 19 m"

### QUANTITY
- **pos_umgv_desc**: "Q'ty" 또는 "Quantity"
- **pos_umgv_value**: "One(1)" 또는 "One(1) set" (✓ 추출 성공 기대)
- **pos_umgv_uom**: "set" 또는 "" (✓ 개선)
- **pos_chunk**: "Q'ty | One(1) set per ship"
- **LLM 검증**: PASS (✓ 더 관대한 검증으로 개선)

## 검증 포인트

1. **파싱 개선 확인**:
   - "SWL 6 tonnes"에서 "6"만 추출되는지
   - "Maximum 19 m"에서 "19"만 추출되는지

2. **Context 개선 확인**:
   - "Context 비어있음" 경고가 줄어드는지
   - pos_chunk가 LLM 검증에 잘 사용되는지

3. **LLM 검증 개선 확인**:
   - QUANTITY "One(1) set"이 거부되지 않고 통과하는지
   - 전체적으로 LLM 거부율이 감소하는지

## 문제 발생 시

로그를 확인하여:
- 어떤 단계에서 실패하는지
- LLM 검증 프롬프트와 응답을 확인
- 필요시 추가 개선 수행
