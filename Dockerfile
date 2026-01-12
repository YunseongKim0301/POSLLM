FROM python:3.10-slim

# 작업 디렉토리
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ollama 설치
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python 패키지 설치
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    beautifulsoup4 \
    lxml \
    requests \
    psycopg2-binary \
    sentence-transformers \
    torch

# 코드 복사
COPY . /app

# Ollama 서비스 시작 스크립트
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "test_extraction.py"]
