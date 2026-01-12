# Windows PC에서 POS 추출 코드 실행 가이드

## 📋 전제 조건

- Windows 10/11 (64-bit)
- 관리자 권한
- 최소 16GB RAM (27B 모델 사용 시)
- 최소 50GB 여유 디스크 공간

---

## 🚀 빠른 시작 (WSL2 사용)

### 1단계: WSL2 설치

**PowerShell (관리자 권한):**

```powershell
wsl --install -d Ubuntu-22.04
```

설치 후 **재부팅**

### 2단계: Ubuntu 설정

재부팅 후 Ubuntu 터미널이 열리면:

```bash
# 사용자명/비밀번호 설정
# 예: posllm / posllm123

# 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지
sudo apt install -y python3 python3-pip python3-venv git curl wget
```

### 3단계: 코드 및 데이터 복사

```bash
# 작업 디렉토리 생성
mkdir -p ~/POSLLM
cd ~/POSLLM

# Git clone (공개 저장소인 경우)
git clone -b claude/enhance-v53-extractor-aVD9t https://github.com/YunseongKim0301/POSLLM.git .

# 또는 Windows에서 파일 복사
mkdir -p test_data uploaded_files

# 용어집 복사 (Windows 경로 예시)
cp /mnt/c/Users/YunseongKim/Desktop/pos/gpt업로드/용어집.txt ./test_data/pos_dict.txt
cp /mnt/c/Users/YunseongKim/Desktop/pos/gpt업로드/사양값DB.txt ./test_data/umgv_fin.txt
cp /mnt/c/Users/YunseongKim/Desktop/pos/gpt업로드/사양값추출_template ./test_data/ext_tmpl.txt

# POS 파일 복사 (1-2개 테스트용)
cp /mnt/c/Users/YunseongKim/Desktop/pos/workspace/POS/phase3/phase3_formatted_new/*.html ./uploaded_files/
```

### 4단계: Python 환경 구성

```bash
cd ~/POSLLM

# 가상환경 생성
python3 -m venv venv

# 활성화
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install pandas numpy beautifulsoup4 lxml requests psycopg2-binary sentence-transformers torch
```

### 5단계: Ollama 설치

```bash
# Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 백그라운드 실행
ollama serve > /tmp/ollama.log 2>&1 &

# 모델 다운로드 (선택: gemma2:27b 또는 qwen2.5:32b)
ollama pull qwen2.5:32b

# 확인
ollama list
```

### 6단계: 테스트 실행

```bash
cd ~/POSLLM
source venv/bin/activate

# 테스트 실행
python test_extraction.py
```

---

## 📁 파일 구조

```
~/POSLLM/
├── v53_extractor.py          # 메인 추출 코드
├── test_extraction.py         # 테스트 스크립트
├── test_data/
│   ├── pos_dict.txt          # 용어집
│   ├── umgv_fin.txt          # 사양값 DB
│   └── ext_tmpl.txt          # 추출 템플릿
├── uploaded_files/            # POS HTML 파일들
│   └── *.html
├── output/                    # 결과 저장
│   ├── test_results.json
│   └── *.csv
└── venv/                      # Python 가상환경
```

---

## 🔧 트러블슈팅

### WSL2 설치 오류

```powershell
# 수동 설치
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 재부팅 후
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

### Ollama 모델 다운로드 느림

```bash
# 더 작은 모델 사용
ollama pull gemma:7b

# Config에서 모델명 변경
config.ollama_model = "gemma:7b"
```

### 메모리 부족

```bash
# WSL 메모리 제한 설정
# Windows 사용자 폴더에 .wslconfig 생성
notepad.exe ~/.wslconfig

# 내용:
[wsl2]
memory=12GB
processors=4
```

### 파일 인코딩 문제

```bash
# UTF-8 변환
iconv -f CP949 -t UTF-8 test_data/pos_dict.txt > test_data/pos_dict_utf8.txt
mv test_data/pos_dict_utf8.txt test_data/pos_dict.txt
```

---

## 📊 결과 확인

```bash
# 결과 파일 보기
cat output/test_results.json | python -m json.tool | head -50

# 성공률 확인
grep -c "pos_umgv_value" output/test_results.json
```

---

## 🎯 정확도 85-90% 달성 팁

1. **용어집 품질 확인**: pos_dict.txt에 충분한 동의어 매핑
2. **사양값 DB 활용**: umgv_fin.txt에 과거 추출 값 존재
3. **Voting 활성화**: config.vote_enabled = True
4. **LLM 검증 활성화**: 모든 추출 결과 LLM 검증
5. **Section/Table 힌트 활용**: pos_dict의 section_num, table_text 활용

---

## 📞 도움말

문제 발생 시:
1. 로그 확인: `cat /tmp/ollama.log`
2. Python 에러: `python test_extraction.py 2>&1 | tee error.log`
3. GitHub Issues 제출
