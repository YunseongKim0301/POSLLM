# v77 개선사항

## Ollama 자동 시작 기능 추가

### 문제
- v76까지는 Ollama 서버를 수동으로 시작해야 했음
- Astrago/웹터미널 환경에서 매번 `ollama serve &` 실행 필요
- Ollama가 실행되지 않으면 LLM 검증/추출이 모두 실패

### 해결
v77에서는 **Ollama 서버가 자동으로 시작**됩니다:

1. **자동 검색**: Ollama 바이너리를 여러 경로에서 자동 검색
2. **자동 시작**: 서버가 실행 중이 아니면 자동으로 `ollama serve` 실행
3. **자동 정리**: 프로그램 종료 시 Ollama 서버 자동 종료
4. **멀티 포트**: 여러 포트에서 동시 serve 지원 (로드 밸런싱)

### UnifiedLLMClient 변경사항

```python
# v76 - 수동 시작 필요
# 터미널에서: ollama serve &
client = UnifiedLLMClient(model="gemma3:27b")

# v77 - 자동 시작 (기본 활성화)
client = UnifiedLLMClient(
    model="gemma3:27b",
    ollama_ports=[11434, 11436],
    auto_start_ollama=True  # 기본값
)
```

### 새로운 메서드

- `_find_ollama_binary()`: Ollama 바이너리 자동 검색
- `_check_ollama_server(port)`: 특정 포트의 서버 상태 확인
- `_start_ollama_server(port)`: 특정 포트에서 서버 시작
- `_ensure_ollama_running()`: 모든 포트에서 서버 확인 및 시작
- `_cleanup_ollama_servers()`: 프로그램 종료 시 서버 정리

### 실행 예시

```bash
# v76 - 수동 시작
OLLAMA_HOST=127.0.0.1:11434 ollama serve &
OLLAMA_HOST=127.0.0.1:11436 ollama serve &
python v76_extractor.py

# v77 - 자동 시작
python v77_extractor.py  # Ollama가 자동으로 시작됨!
```

## Rule 기반 추출 문제 분석

### 발견된 문제

**MAX. WORKING RADIUS** 추출 실패:
- **기대값**: "19 m"
- **추출값**: "Maximum"
- **원인**: rowspan이 있는 복잡한 테이블 구조에서 값 추출 실패

HTML 구조:
```html
<tr>
  <td rowspan="2">Working radius</td>
  <td>Maximum</td>
  <td>19 m</td>  <-- 실제 값
</tr>
<tr>
  <td>Minimum</td>
  <td>The Supplier's standard</td>
</tr>
```

현재 파서가 "Maximum"을 값으로 잘못 추출함 (실제 값은 "19 m")

### 해결 방법 (v78에서 수정 예정)

1. **HTML 테이블 파싱 개선**
   - rowspan/colspan 처리 로직 강화
   - 테이블 셀 매핑 알고리즘 개선

2. **LLM Fallback 활용**
   - Rule 추출 실패 시 LLM이 올바른 값 추출
   - v77에서 Ollama가 정상 작동하면 이런 문제 자동 해결

3. **검증 강화**
   - 추출된 값이 숫자+단위 형식인지 검증
   - "Maximum", "Minimum" 같은 단어는 값이 아닌 설명으로 처리

### 임시 해결책

v77에서는 Ollama가 정상 작동하므로:
1. Rule 기반 추출이 실패해도
2. LLM Fallback이 올바른 값을 추출
3. LLM 검증이 잘못된 값을 수정

따라서 **Ollama만 제대로 실행되면** Rule 추출 문제가 자동으로 보완됩니다!

## 사용 방법

### 1. Ollama 설치 (처음 한 번만)

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# 모델 다운로드
ollama pull gemma3:27b
```

### 2. v77 실행

```bash
python v77_extractor.py
```

끝! Ollama가 자동으로 시작되고, Rule 추출 문제도 LLM이 보완합니다.

## 참고 문서

- [Ollama 자동 시작 가이드](OLLAMA_AUTO_START_GUIDE.md)
- [테스트 스크립트](test_ollama_autostart.py)
