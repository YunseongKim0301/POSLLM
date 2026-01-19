# Ollama Context Length 설정 가이드

## 문제 상황

기본 Ollama context length는 **4096 tokens** (약 16KB 텍스트)로 설정되어 있습니다.
하지만 v76 extractor는:
- USER_MAX_EVIDENCE_CHARS = 8000 (8KB)
- 프롬프트 템플릿 + 시스템 메시지 = 2-4KB
- **총 10-12KB 텍스트 = 약 2.5-3K tokens**

Gemma3는 스펙상 **128K context**를 지원하지만, 기본 설정으로는 활용하지 못합니다.

## 해결 방법

### 1. Ollama 서비스 시작 시 설정

```bash
# 32K context로 설정 (권장)
OLLAMA_NUM_CTX=32768 ollama serve

# 또는 64K (더 여유있게)
OLLAMA_NUM_CTX=65536 ollama serve
```

### 2. 환경 변수로 영구 설정

**Linux/Mac (~/.bashrc 또는 ~/.zshrc)**:
```bash
export OLLAMA_NUM_CTX=32768
```

**Docker 환경**:
```bash
docker run -d \
  -e OLLAMA_NUM_CTX=32768 \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama
```

### 3. 실행 중인 Ollama 재시작

```bash
# Ollama 프로세스 종료
pkill ollama

# 새 설정으로 시작
OLLAMA_NUM_CTX=32768 ollama serve &
```

## v76 Extractor 설정

v76_extractor.py는 이미 다음과 같이 설정되어 있습니다:

```python
# Config 클래스
ollama_num_ctx: int = 32768  # Context length (tokens)

# UnifiedLLMClient에 자동 전달
payload = {
    "model": self.model,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": self.temperature,
        "num_ctx": self.num_ctx  # ← 이것이 Ollama API로 전달됨
    }
}
```

## 확인 방법

### Ollama 로그 확인
```bash
# Ollama 로그에서 num_ctx 확인
# 정상적으로 설정되면 로그에 표시됨
ollama serve
# 출력 예시: "num_ctx=32768"
```

### 테스트 API 호출
```bash
curl -X POST http://localhost:11434/api/generate \
  -d '{
    "model": "gemma3:27b",
    "prompt": "Hello",
    "options": {"num_ctx": 32768}
  }'
```

## 권장 설정 값

| 모델 | VRAM | 권장 num_ctx | 비고 |
|------|------|-------------|------|
| gemma3:27b | 40GB | 32768 | 기본 권장 |
| gemma3:27b | 80GB+ | 65536 | 더 여유있게 |
| qwen2.5:32b | 40GB | 32768 | 기본 권장 |
| qwen2.5:32b | 80GB+ | 65536 | 더 여유있게 |

## 기대 효과

1. **LLM Chunk Selection 정확도 향상**
   - 긴 문서 컨텍스트 전체 처리 가능
   - 잘림 없이 완전한 정보 제공

2. **LLM Validation 정확도 향상**
   - 충분한 context로 정확한 검증 가능
   - "Context 비어있음" 경고 감소

3. **LLM Fallback 성공률 향상**
   - 복잡한 추출 케이스도 충분한 context로 처리

## 문제 해결

### "context length exceeded" 오류
→ OLLAMA_NUM_CTX 값을 더 크게 설정 (예: 65536)

### VRAM 부족 오류
→ OLLAMA_NUM_CTX 값을 줄이거나 (16384), USER_MAX_EVIDENCE_CHARS를 줄임 (4000)

### 설정이 적용 안 됨
→ Ollama 프로세스 완전 재시작 필요
```bash
pkill -9 ollama
OLLAMA_NUM_CTX=32768 ollama serve &
```
