# Ollama 자동 시작 가이드 (v77)

## 개요

v77_extractor.py는 **Ollama 서버를 자동으로 시작**하는 기능이 추가되었습니다.
Astrago와 같은 SaaS 웹터미널 환경에서 코드만 실행하면 Ollama가 자동으로 serve되도록 설계되었습니다.

## 주요 개선사항

### 1. Ollama 자동 시작 (Auto-Start)

UnifiedLLMClient가 초기화될 때:
1. 설정된 포트에서 Ollama 서버가 실행 중인지 확인
2. 실행되지 않으면 자동으로 `ollama serve` 실행
3. 서버가 정상적으로 시작될 때까지 대기 (최대 30초)
4. 프로그램 종료 시 자동으로 Ollama 서버 정리

### 2. Ollama 바이너리 자동 검색

다음 경로에서 Ollama 바이너리를 자동으로 검색:
- `/usr/local/bin/ollama`
- `/usr/bin/ollama`
- `/opt/ollama/bin/ollama`
- `ollama` (PATH에 있는 경우)

### 3. 멀티 포트 지원

여러 포트에서 Ollama를 동시에 실행 가능:
- 포트 11434, 11436 등에서 동시 serve
- 자동 로드 밸런싱

## Ollama 설치 (Astrago 환경)

### 1. Ollama 설치

```bash
# Ollama 설치 (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# 설치 확인
ollama --version
```

### 2. 모델 다운로드

```bash
# gemma3:27b 모델 다운로드 (v77에서 사용)
ollama pull gemma3:27b

# 다운로드 확인
ollama list
```

### 3. v77_extractor.py 실행

이제 **코드만 실행하면 Ollama가 자동으로 시작**됩니다!

```bash
python v77_extractor.py
```

## 실행 로그 예시

```
[INFO] UnifiedLLMClient 초기화: gemma3:27b (ports: [11434, 11436])
[INFO] Ollama 서버 상태 확인: 2개 포트
[INFO] ✗ Ollama 서버가 포트 11434에서 실행되지 않음 → 자동 시작 시도
[INFO] Ollama 바이너리 발견: /usr/local/bin/ollama
[INFO] Ollama 서버 시작 중: /usr/local/bin/ollama serve (포트 11434)
[INFO] ✓ Ollama 서버가 포트 11434에서 정상 시작됨 (3초 소요)
[INFO] ✗ Ollama 서버가 포트 11436에서 실행되지 않음 → 자동 시작 시도
[INFO] Ollama 서버 시작 중: /usr/local/bin/ollama serve (포트 11436)
[INFO] ✓ Ollama 서버가 포트 11436에서 정상 시작됨 (3초 소요)
```

## 문제 해결

### Ollama가 설치되지 않은 경우

```
[WARNING] Ollama 바이너리를 찾을 수 없음
[ERROR] Ollama 바이너리를 찾을 수 없어 포트 11434에서 시작 불가
```

**해결:** 위의 "Ollama 설치" 섹션 참조

### 모델이 없는 경우

Ollama 서버는 시작되지만 LLM 호출 시 실패:

```
[WARNING] Ollama 오류: model 'gemma3:27b' not found
```

**해결:** `ollama pull gemma3:27b`

### 포트가 이미 사용 중인 경우

```
[ERROR] Ollama 서버 시작 실패 (포트 11434): address already in use
```

**해결:**
1. 기존 Ollama 프로세스 종료: `pkill ollama`
2. 다른 포트 사용: v77_extractor.py 상단의 `ollama_ports` 설정 변경

### 자동 시작 비활성화

자동 시작을 원하지 않는 경우:

```python
# v77_extractor.py에서 UnifiedLLMClient 생성 시
client = UnifiedLLMClient(
    model="gemma3:27b",
    ollama_ports=[11434, 11436],
    auto_start_ollama=False  # 자동 시작 비활성화
)
```

그런 다음 수동으로 Ollama 서버 시작:

```bash
# 포트 11434
OLLAMA_HOST=127.0.0.1:11434 ollama serve &

# 포트 11436
OLLAMA_HOST=127.0.0.1:11436 ollama serve &
```

## v76과의 차이점

| 기능 | v76 | v77 |
|------|-----|-----|
| Ollama 서버 시작 | 수동 (`ollama serve &`) | **자동** |
| Ollama 설치 확인 | 없음 | **자동 검색** |
| 포트 상태 확인 | 없음 | **자동 확인** |
| 서버 정리 | 수동 (`pkill ollama`) | **자동 (atexit)** |

## 추가 정보

- v77은 v76의 모든 기능을 포함합니다
- Ollama 자동 시작은 **기본적으로 활성화**되어 있습니다
- 여러 v77 인스턴스가 동시에 실행되어도 Ollama 서버는 중복 시작되지 않습니다 (클래스 레벨 프로세스 추적)

## 참고 문서

- [Ollama 공식 문서](https://ollama.com)
- [v76 extractor README](/home/user/POSLLM/v61_extractor_README.md)
