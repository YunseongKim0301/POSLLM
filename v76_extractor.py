# -*- coding: utf-8 -*-
"""
POS Specification Value Extractor (PostgreSQL-Enhanced)
========================================================

PostgreSQL 기반 동적 지식 통합 및 성능 최적화

주요 기능:
1. PostgreSQL 전용 모드
2. 동적 지식 베이스 (pos_dict, umgv_fin 활용)
3. In-memory 캐싱으로 보안 네트워크 대응
4. Enhanced chunk selection (7-stage)
5. LLM 후처리 (범위 파싱, 단위 정규화)
6. 병렬 처리 최적화
7. 출력 형식 개선 (nested JSON, 디버그 CSV)

추출 모드:
- FULL: Template의 모든 POS 추출
- LIGHT: POS 폴더 파일 기준, 해당 row만 추출 (빠른 초기화)
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import glob
import pickle
import shutil
import hashlib
import logging
import traceback
import threading
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 서드파티 라이브러리
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # tqdm 없으면 dummy function 사용
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_SENTENCE_TRANSFORMER = True
except ImportError:
    HAS_SENTENCE_TRANSFORMER = False

try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_BGE_M3 = True
except ImportError:
    HAS_BGE_M3 = False


# #############################################################################
# ██████╗ ██╗   ██╗███╗   ██╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗
# ██╔══██╗██║   ██║████╗  ██║    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝
# ██████╔╝██║   ██║██╔██╗ ██║    ██╔████╔██║██║   ██║██║  ██║█████╗  
# ██╔══██╗██║   ██║██║╚██╗██║    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  
# ██║  ██║╚██████╔╝██║ ╚████║    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗
# ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
#                                                                    
# 이 섹션의 변수만 변경하여 실행 방식을 결정합니다.
# #############################################################################

# =============================================================================
# [1] 추출 모드 설정 (가장 중요!)
# =============================================================================
# "full"   : 전체 추출 모드 - Template의 모든 POS 추출
# "light"  : 소량 추출 모드 - POS 폴더 파일 기준으로 해당 row만 추출
# "verify" : 검증 모드 - 사양값DB의 기존 값을 POS 문서와 대조하여 검증
EXTRACTION_MODE = "light"

# =============================================================================
# [2] 데이터 소스 모드
# =============================================================================
# "file" : 파일 기반 입출력 (로컬 파일)
# "db"   : PostgreSQL 기반 입출력
DATA_SOURCE_MODE = "db"

# =============================================================================
# [3] 출력 설정
# =============================================================================
SAVE_JSON = True           # nested JSON 저장
SAVE_CSV = False            # 기본 CSV 저장
SAVE_DEBUG_CSV = False      # 디버그 CSV (참조 정보 포함) 저장
SAVE_TO_DB = False         # PostgreSQL에 결과 저장
PRINT_JSON = True          # JSON 결과를 콘솔에도 출력 (pretty print)

# =============================================================================
# [4] Light 모드 전용 설정
# =============================================================================
# Light 모드에서 POS 문서를 찾을 폴더 (Full 모드와 다름)
LIGHT_MODE_POS_FOLDER = "/workspace/server/uploaded_files"

# Light 모드에서 배치 처리 비활성화 (단건 처리)
LIGHT_MODE_BATCH_DISABLED = True

# Light 모드에서 Hybrid glossary match 스킵 (초기화 시간 단축)
LIGHT_MODE_SKIP_HYBRID_MATCH = True

# Light 모드 병렬 처리
LIGHT_MODE_WORKERS = 4  # 병렬 worker 수 (3-6 권장, 현재 27sec → 목표 5-10sec)

# =============================================================================
# [5] pos_embedding DB 활용 설정
# =============================================================================
# pos_embedding 테이블에서 사전 임베딩 활용
USE_PRECOMPUTED_EMBEDDINGS = True
EMBEDDING_TABLE_NAME = "pos_embedding"
EMBEDDING_TOP_K = 5                    # 유사도 검색 시 상위 K개
EMBEDDING_SIMILARITY_THRESHOLD = 0.65  # 유사도 임계값

# =============================================================================
# [6] Verify 모드 전용 설정
# =============================================================================
# Verify 모드에서 사양값DB 테이블
VERIFY_MODE_SPECDB_TABLE = "umgv_fin"

# 단위 변환 허용 오차 (%)
VERIFY_UNIT_CONVERSION_TOLERANCE = 5.0

# 검증 신뢰도 임계값
VERIFY_CONFIDENCE_THRESHOLD = 0.7


# #############################################################################
# 일반 사용자 설정 (자주 변경하지 않음)
# #############################################################################

# =============================================================================
# 파일 모드 경로 설정
# =============================================================================
# Full 모드용 POS 문서 폴더
USER_BASE_FOLDER = "/workspace/pos/phase3/phase3_formatted_new"
USER_GLOSSARY_PATH = "/workspace/data/용어집.txt"
USER_SPEC_PATH = "/workspace/data/사양값추출_ongoing_sample.txt"
USER_SPECDB_PATH = "/workspace/data/사양값DB.txt"
USER_OUTPUT_PATH = "/workspace/results/ongoing/samples"
USER_PARTIAL_OUTPUT_PATH = "/workspace/results/ongoing/samples/partial"

USER_ENABLE_CHECKPOINT = True

# =============================================================================
# PostgreSQL 설정
# =============================================================================
USER_DB_HOST = "10.131.132.116"
USER_DB_PORT = 5432
USER_DB_NAME = "managesys"
USER_DB_USER = "postgres"
USER_DB_PASSWORD = "pmg_umg!@"

USER_DB_TABLE_GLOSSARY = "pos_dict"
USER_DB_TABLE_SPECDB = "umgv_fin"
USER_DB_TABLE_TEMPLATE = "ext_tmpl"

# =============================================================================
# LLM 설정
# =============================================================================
USER_USE_LLM = True
USER_OLLAMA_MODEL = "gemma3:27b"
USER_OLLAMA_BIN = "/workspace/ollama/bin/ollama"
USER_OLLAMA_MODELS_DIR = "/workspace/models"
USER_OLLAMA_TIMEOUT_SEC = 180

USER_OLLAMA_HOST = "127.0.0.1"
USER_OLLAMA_PORTS = [11434, 11436]
USER_AUTO_START_OLLAMA_SERVE = True
USER_OLLAMA_SERVE_START_GRACE_SEC = 10

# 다수결 투표 설정
USER_VOTE_ENABLED = True
USER_VOTE_K = 2
USER_VOTE_MIN_AGREEMENT = 2

# 병렬 처리 설정
USER_ENABLE_PARALLEL = True
USER_NUM_WORKERS = 30
USER_LLM_WORKERS = 2

# 배치 처리 설정
USER_BATCH_SIZE = 15

# LLM Context 설정
# CRITICAL: Ollama 기본 context는 4096 tokens (약 16KB text)로 부족함
# Gemma3는 128K context 지원하므로 32K로 설정 권장
# 설정 방법: OLLAMA_NUM_CTX=32768 ollama serve
USER_MAX_EVIDENCE_CHARS = 8000  # 8KB (프롬프트 포함 약 10-12KB = 2.5-3K tokens)
USER_OLLAMA_NUM_CTX = 32768  # Ollama context length (tokens)

# LLM 호출 안정화
USER_LLM_RATE_LIMIT_SEC = 0.3
USER_LLM_MAX_RETRIES = 3
USER_LLM_RETRY_SLEEP_SEC = 1.5
USER_LLM_TEMPERATURE = 0.0

# =============================================================================
# LLM Audit 설정
# =============================================================================
USER_ENABLE_LLM_AUDIT = True
USER_AUDIT_CONFIDENCE_THRESHOLD = 0.85
USER_AUDIT_ALWAYS_FOR_LLM = True

# Rule 기반 추출이 실패할 때 LLM Fallback 활성화
USER_ENABLE_LLM_FALLBACK = True
USER_RULE_CONF_THRESHOLD = 0.72

# =============================================================================
# 시멘틱 검색 설정 (BGE-M3)
# =============================================================================
USER_ENABLE_SEMANTIC_SEARCH = True
USER_SEMANTIC_MODEL_PATH = "/workspace/bge-m3"
USER_SEMANTIC_DEVICE = "cuda"
USER_SEMANTIC_SIMILARITY_THRESHOLD = 0.65
USER_SEMANTIC_CODE_MATCH_BOOST = 0.2
USER_SEMANTIC_EXACT_MATCH_THRESHOLD = 0.90
USER_SEMANTIC_TOP_K = 5
USER_ENABLE_LLM_SIMILARITY_VALIDATION = False
USER_SEMANTIC_CACHE_DIR = "/workspace/cache/embeddings"

# =============================================================================
# 값 검증 설정
# =============================================================================
ENABLE_VALUE_VALIDATION = True
# NUMERIC_VARIANCE_THRESHOLD 제거: 실제 사양값은 과거 값과 전혀 다를 수 있으므로
# 과거 값 대비 variance check는 부적절함
MIN_VALUE_LENGTH = 1
MAX_VALUE_LENGTH = 200

# =============================================================================
# 복합값/범위값 파싱 설정
# =============================================================================
SPLIT_COMPOUND_VALUES = True  # 슬래시(/) 구분 복합값 분리 여부
SPLIT_RANGE_VALUES = True     # 범위(~, -) 값 분리 여부


# =============================================================================
# 로깅 설정
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("POSExtractor")


# ############################################################################
# 체크박스 패턴 (하드코딩 허용 - 표준화된 UI 패턴)
# ############################################################################

CHECKBOX_PATTERNS = {
    'YNQ_BRACKET': re.compile(r'\(([YNQ])\)', re.IGNORECASE),
    'CHECKED_SQUARE': re.compile(r'\[x\]|\[X\]|■|☑|✓|✔|√'),
    'UNCHECKED_SQUARE': re.compile(r'\[\s*\]|□|☐'),
    'CHECKED_CIRCLE': re.compile(r'●|◉'),
    'UNCHECKED_CIRCLE': re.compile(r'○|◯'),
    'OX_CHECKED': re.compile(r'\(O\)|\(o\)'),
    'OX_UNCHECKED': re.compile(r'\(X\)|\(x\)'),
}


# =============================================================================
# 출력 스키마 정의
# =============================================================================

# 새 출력 스키마 (사용자 요청 기준)
OUTPUT_SCHEMA_COLUMNS = [
    # 템플릿에서 가져오는 컬럼
    "pmg_desc", "pmg_code", "umg_desc", "umg_code",
    "mat_attr_desc", "extwg", "matnr", "doknr",
    "umgv_desc", "umgv_code", "umgv_uom",
    # POS 문서 관련
    "file_name",
    # 추출 결과 컬럼
    "section_num", "table_text", "value_format",
    "pos_mat_attr_desc", "pos_umgv_desc",
    "pos_umgv_value", "umgv_value_edit", "pos_umgv_uom",
    "pos_chunk",
    "evidence_fb",  # 빈값 고정
    # 타임스탬프
    "created_on", "updated_on",
]

# CSV용 추가 컬럼 (디버깅/분석용)
CSV_DEBUG_COLUMNS = [
    *OUTPUT_SCHEMA_COLUMNS,
    # 메타데이터
    "_method", "_confidence",
    # 참조 정보
    "_ref_pmg_code", "_ref_pmg_desc",
    "_ref_umg_code", "_ref_umg_desc",
    "_ref_umgv_code", "_ref_umgv_desc",
    "_ref_extwg", "_ref_extwg_desc",
    "_ref_similarity", "_ref_match_type",
    # 추출 근거
    "_evidence",
]


# =============================================================================
# GPU VRAM 모니터링
# =============================================================================

def get_gpu_memory_info() -> Dict[str, Any]:
    """
    nvidia-smi를 사용해 GPU VRAM 사용량 조회

    Returns:
        {
            'available': True/False,
            'gpus': [
                {
                    'id': 0,
                    'name': 'GPU name',
                    'memory_used_mb': 1024,
                    'memory_total_mb': 40960,
                    'memory_free_mb': 39936,
                    'utilization_percent': 25
                },
                ...
            ]
        }
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return {'available': False, 'gpus': [], 'error': 'nvidia-smi failed'}

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpus.append({
                    'id': int(parts[0]),
                    'name': parts[1],
                    'memory_used_mb': float(parts[2]),
                    'memory_total_mb': float(parts[3]),
                    'memory_free_mb': float(parts[4]),
                    'utilization_percent': float(parts[5])
                })

        return {'available': True, 'gpus': gpus}

    except FileNotFoundError:
        return {'available': False, 'gpus': [], 'error': 'nvidia-smi not found'}
    except Exception as e:
        return {'available': False, 'gpus': [], 'error': str(e)}


def log_gpu_memory(logger: logging.Logger):
    """GPU VRAM 사용량을 로그에 출력"""
    gpu_info = get_gpu_memory_info()

    if not gpu_info['available']:
        logger.debug(f"GPU 정보 없음: {gpu_info.get('error', 'unknown')}")
        return

    for gpu in gpu_info['gpus']:
        logger.info(
            f"GPU {gpu['id']} ({gpu['name']}): "
            f"VRAM {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB "
            f"({gpu['memory_used_mb']/gpu['memory_total_mb']*100:.1f}% used), "
            f"Util: {gpu['utilization_percent']:.0f}%"
        )


def log_extraction_hint(logger: logging.Logger, spec_name: str, hint, source: str = ""):
    """추출 시 사용된 힌트 정보 로깅"""
    if not hint:
        logger.debug(f"[{source}] {spec_name}: 힌트 없음")
        return

    hint_info = []

    if hasattr(hint, 'section_num') and hint.section_num:
        hint_info.append(f"Section={hint.section_num[:30]}")

    if hasattr(hint, 'historical_values') and hint.historical_values:
        examples = ', '.join(hint.historical_values[:2])
        hint_info.append(f"HistValues=[{examples}]")

    if hasattr(hint, 'pos_umgv_desc') and hint.pos_umgv_desc:
        hint_info.append(f"Synonym={hint.pos_umgv_desc}")

    if hasattr(hint, 'table_text') and hint.table_text:
        hint_info.append(f"TableText={hint.table_text}")

    if hasattr(hint, 'value_format') and hint.value_format:
        hint_info.append(f"Format={hint.value_format}")

    if hint_info:
        logger.debug(f"[{source}] {spec_name}: 힌트 활용 - {', '.join(hint_info)}")
    else:
        logger.debug(f"[{source}] {spec_name}: 힌트 객체 있으나 정보 없음")


# =============================================================================
# 유틸리티 함수
# =============================================================================

def norm(x: Any) -> str:
    """
    값 정규화: None, NaN, 빈 문자열 등을 빈 문자열로 변환
    Null-safe 처리를 위해 모든 데이터 접근에 사용
    """
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null", "<na>", ""):
        return ""
    return s


def safe_get(data: Dict, key: str, default: str = "") -> str:
    """딕셔너리에서 안전하게 값 가져오기"""
    if data is None:
        return default
    val = data.get(key, default)
    return norm(val) if val is not None else default


def safe_float(val: Any, default: float = 0.0) -> float:
    """안전한 float 변환"""
    try:
        if val is None:
            return default
        if isinstance(val, str):
            val = val.replace(',', '').strip()
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_numeric_value(text: str) -> Optional[float]:
    """텍스트에서 숫자값 추출"""
    if not text:
        return None
    # 숫자와 소수점, 쉼표만 추출
    nums = re.findall(r'[\d,]+\.?\d*', text)
    if nums:
        try:
            return float(nums[0].replace(',', ''))
        except ValueError:
            return None
    return None


def is_numeric_spec(spec_name: str, value_format: str = "", confidence_level: bool = False) -> bool:
    """
    숫자형 사양인지 판단 (DB 기반 + 키워드 힌트)

    우선순위:
    1. value_format (용어집 DB) - 절대 규칙
    2. numeric_keywords - 참조 힌트 (유연한 판단)

    Args:
        spec_name: 사양명
        value_format: 용어집의 value_format 컬럼 (NUMERIC/TEXT/MIXED)
        confidence_level: True이면 (is_numeric, confidence) 튜플 반환

    Returns:
        bool 또는 (bool, float) - confidence_level=True인 경우
    """
    # 1. value_format 우선 (DB 기반, 신뢰도 높음)
    if value_format:
        vf_upper = value_format.upper()
        if vf_upper == "NUMERIC":
            return (True, 1.0) if confidence_level else True
        elif vf_upper == "TEXT":
            return (False, 1.0) if confidence_level else False
        elif vf_upper == "MIXED":
            return (True, 0.7) if confidence_level else True  # Mixed는 부분적으로 숫자

    # 2. 키워드 힌트 (참조용, 신뢰도 중간)
    numeric_keywords = [
        'capacity', 'head', 'power', 'pressure', 'temperature',
        'flow', 'speed', 'rpm', 'voltage', 'frequency', 'weight',
        'qty', 'quantity', 'no.', 'number', 'length', 'width', 'height',
        'diameter', 'thickness', 'volume', 'area', 'mcr', 'ncr',
        'current', 'ampere', 'resistance', 'time', 'rate', 'efficiency'
    ]

    spec_lower = spec_name.lower()
    keyword_match = any(kw in spec_lower for kw in numeric_keywords)

    if confidence_level:
        if keyword_match:
            return (True, 0.6)  # 키워드 매칭: 중간 신뢰도
        else:
            return (False, 0.5)  # 불확실: 낮은 신뢰도
    else:
        return keyword_match


def detect_checkbox_selection(text: str) -> Optional[str]:
    """체크박스 선택 상태 감지"""
    if not text:
        return None

    ynq_match = CHECKBOX_PATTERNS['YNQ_BRACKET'].search(text)
    if ynq_match:
        return ynq_match.group(1).upper()

    if CHECKBOX_PATTERNS['CHECKED_SQUARE'].search(text):
        return "Y"
    if CHECKBOX_PATTERNS['CHECKED_CIRCLE'].search(text):
        return "Y"
    if CHECKBOX_PATTERNS['OX_CHECKED'].search(text):
        return "Y"

    if CHECKBOX_PATTERNS['UNCHECKED_SQUARE'].search(text):
        return "N"
    if CHECKBOX_PATTERNS['UNCHECKED_CIRCLE'].search(text):
        return "N"
    if CHECKBOX_PATTERNS['OX_UNCHECKED'].search(text):
        return "N"

    return None


def parse_compound_value(raw_value: str, split_enabled: bool = True) -> List[Tuple[str, str]]:
    """복합값 파싱 - 슬래시 구분"""
    if not raw_value or not raw_value.strip():
        return []

    raw_value = raw_value.strip()

    if not split_enabled:
        return [(raw_value, "")]

    # 단위 보호 패턴
    protected_units = ['m3/h', 'kg/h', 'l/h', 'nm3/h', 'kj/kg', 'w/m2', 'kg/m3', 'l/min', 'kg/cm2']
    temp_value = raw_value
    placeholders = {}

    for i, unit in enumerate(protected_units):
        placeholder = f"__UNIT_{i}__"
        if unit.lower() in temp_value.lower():
            temp_value = re.sub(re.escape(unit), placeholder, temp_value, flags=re.IGNORECASE)
            placeholders[placeholder] = unit

    # 슬래시로 분리 (보호된 단위 제외)
    parts = re.split(r'\s*/\s*', temp_value)

    if len(parts) == 1:
        return [(raw_value, "")]

    results = []
    for part in parts:
        part = part.strip()
        # 플레이스홀더 복원
        for ph, unit in placeholders.items():
            part = part.replace(ph, unit)

        if not part:
            continue

        # 숫자와 단위 분리
        match = re.match(r'^([0-9,.\-\s]+)\s*([a-zA-Z°℃%/\d]+.*)?$', part)
        if match:
            val = match.group(1).strip()
            unit = match.group(2).strip() if match.group(2) else ""
            results.append((val, unit))
        else:
            results.append((part, ""))

    return results if results else [(raw_value, "")]


def parse_range_value(raw_value: str, split_enabled: bool = True) -> List[Tuple[str, str]]:
    """범위형 값 파싱"""
    if not raw_value or not raw_value.strip():
        return []

    raw_value = raw_value.strip()

    if not split_enabled:
        unit_match = re.search(r'([a-zA-Z°℃%/]+)\s*$', raw_value)
        unit = unit_match.group(1) if unit_match else ""
        return [(raw_value, unit)]

    range_match = re.match(r'^([0-9,.\s]+)\s*[~\-]\s*([0-9,.\s]+)\s*([a-zA-Z°℃%]+)?$', raw_value)

    if range_match:
        val1 = range_match.group(1).strip()
        val2 = range_match.group(2).strip()
        unit = range_match.group(3) or ""
        return [(val1, unit), (val2, unit)]

    # 범위가 아니면 단일 값
    unit_match = re.search(r'([a-zA-Z°℃%/]+)\s*$', raw_value)
    unit = unit_match.group(1) if unit_match else ""
    return [(raw_value, unit)]


def build_composite_text(*texts) -> str:
    """
    복합 텍스트 생성 (임베딩용)
    비어있지 않은 값만 결합
    """
    parts = [norm(t) for t in texts if norm(t)]
    return " | ".join(parts)


def extract_hull_from_matnr(matnr: str) -> str:
    """matnr에서 hull 번호 추출 (예: 2377AYS36315 → 2377)"""
    matnr = norm(matnr)
    if not matnr:
        return ""
    match = re.match(r'^(\d{4})', matnr)
    return match.group(1) if match else ""


def extract_pos_from_doknr(doknr: str) -> str:
    """doknr에서 POS 번호 추출 (예: 2377-POS-0036331 → 0036331)"""
    doknr = norm(doknr)
    if not doknr:
        return ""
    match = re.search(r'POS-(\d+)', doknr)
    return match.group(1) if match else ""


def ensure_parent_dir(file_path: str) -> None:
    """부모 디렉토리 생성"""
    parent = os.path.dirname(file_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# =============================================================================
# 단위 변환 유틸리티 (Verify 모드용)
# =============================================================================

# 단위 변환 매핑 (표준 단위 → 배율)
UNIT_CONVERSION_TABLE = {
    # 부피 유량 (기준: L/H)
    'm3/h': 1000.0,
    'm³/h': 1000.0,
    'l/h': 1.0,
    'l/min': 60.0,
    'l/s': 3600.0,

    # 압력 (기준: bar)
    'bar': 1.0,
    'mpa': 10.0,
    'kpa': 0.01,
    'kg/cm2': 0.980665,
    'kg/cm²': 0.980665,
    'psi': 0.0689476,

    # 온도 (기준: °C) - 차이값만 변환 가능
    '°c': 1.0,
    '℃': 1.0,
    'c': 1.0,

    # 전력 (기준: kW)
    'kw': 1.0,
    'w': 0.001,
    'mw': 1000.0,
    'hp': 0.7457,
    'ps': 0.7355,

    # 회전수 (기준: rpm)
    'rpm': 1.0,
    'min-1': 1.0,

    # 전압 (기준: V)
    'v': 1.0,
    'kv': 1000.0,

    # 주파수 (기준: Hz)
    'hz': 1.0,
    'khz': 1000.0,

    # 질량 유량 (기준: kg/h)
    'kg/h': 1.0,
    'kg/s': 3600.0,
    't/h': 1000.0,

    # 길이 (기준: mm)
    'mm': 1.0,
    'cm': 10.0,
    'm': 1000.0,
    'inch': 25.4,
    'in': 25.4,

    # 무게 (기준: kg)
    'kg': 1.0,
    'g': 0.001,
    't': 1000.0,
    'ton': 1000.0,
}

# =============================================================================
# 단위 카테고리 매핑 (단위 변환 없이 카테고리 일치만 확인)
# =============================================================================

def normalize_unit(unit: str) -> str:
    """
    단위 정규화 (대소문자, 특수문자 통일)

    Args:
        unit: 원본 단위 문자열

    Returns:
        정규화된 단위
    """
    if not unit:
        return ""

    # 소문자 변환
    normalized = unit.lower().strip()

    # 공백 제거
    normalized = normalized.replace(' ', '')

    # 특수문자 통일
    normalized = normalized.replace('³', '3')
    normalized = normalized.replace('²', '2')
    normalized = normalized.replace('º', '°')

    return normalized


def convert_unit_value(
    value: float,
    from_unit: str,
    to_unit: str,
    tolerance_percent: float = 5.0
) -> Tuple[float, bool]:
    """
    단위 변환

    Args:
        value: 원본 값
        from_unit: 원본 단위
        to_unit: 목표 단위
        tolerance_percent: 허용 오차 (%)

    Returns:
        (변환된 값, 변환 성공 여부)
    """
    # 단위 정규화
    from_unit_norm = normalize_unit(from_unit)
    to_unit_norm = normalize_unit(to_unit)

    # 동일 단위
    if from_unit_norm == to_unit_norm:
        return value, True

    # 변환 테이블 확인
    if from_unit_norm not in UNIT_CONVERSION_TABLE:
        return value, False

    if to_unit_norm not in UNIT_CONVERSION_TABLE:
        return value, False

    # 기준 단위로 변환 후 목표 단위로 변환
    from_multiplier = UNIT_CONVERSION_TABLE[from_unit_norm]
    to_multiplier = UNIT_CONVERSION_TABLE[to_unit_norm]

    # 변환
    base_value = value * from_multiplier
    converted_value = base_value / to_multiplier

    return converted_value, True


def values_match_with_unit_conversion(
    value1: str,
    unit1: str,
    value2: str,
    unit2: str,
    tolerance_percent: float = 5.0
) -> Tuple[bool, float, str]:
    """
    두 값이 단위 변환을 고려하여 일치하는지 확인

    Args:
        value1, unit1: 첫 번째 값과 단위 (POS에서 추출)
        value2, unit2: 두 번째 값과 단위 (DB에 등록)
        tolerance_percent: 허용 오차 (%)

    Returns:
        (일치 여부, 유사도 점수, 설명)
    """
    # 값 추출
    try:
        val1 = extract_numeric_value(value1)
        val2 = extract_numeric_value(value2)

        if val1 is None or val2 is None:
            # 숫자가 아닌 경우 문자열 비교
            if value1.strip().upper() == value2.strip().upper():
                return True, 1.0, "텍스트 완전 일치"
            else:
                return False, 0.0, "텍스트 불일치"

        # 단위가 없는 경우
        if not unit1 or not unit2:
            # 단위 없이 값만 비교
            diff_percent = abs(val1 - val2) / max(abs(val2), 0.001) * 100
            if diff_percent <= tolerance_percent:
                return True, 1.0 - (diff_percent / tolerance_percent) * 0.3, f"값 일치 (오차 {diff_percent:.1f}%)"
            else:
                return False, 0.0, f"값 불일치 (오차 {diff_percent:.1f}%)"

        # 단위 변환 시도
        converted_val, success = convert_unit_value(val1, unit1, unit2, tolerance_percent)

        if not success:
            # 변환 실패 - 단위가 호환되지 않음
            return False, 0.0, f"단위 변환 불가 ({unit1} → {unit2})"

        # 변환된 값과 비교
        diff_percent = abs(converted_val - val2) / max(abs(val2), 0.001) * 100

        if diff_percent <= tolerance_percent:
            if diff_percent < 0.1:
                return True, 1.0, f"완전 일치 ({unit1} → {unit2})"
            else:
                return True, 1.0 - (diff_percent / tolerance_percent) * 0.3, f"변환 후 일치 (오차 {diff_percent:.1f}%, {unit1} → {unit2})"
        else:
            return False, 0.0, f"변환 후 불일치 (오차 {diff_percent:.1f}%, {unit1} → {unit2})"

    except Exception as e:
        return False, 0.0, f"비교 오류: {e}"


def load_tsv(file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """TSV 파일 로드 (안전한 파싱)"""
    if not os.path.exists(file_path):
        logger.warning("파일 없음: %s", file_path)
        return pd.DataFrame()
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        logger.info("TSV 로드: %s (%d bytes)", file_path, len(content))
        
        df = pd.read_csv(
            StringIO(content),
            sep='\t',
            dtype=str,
            on_bad_lines='skip',
            engine='python'
        )
        
        # 컬럼명 정규화
        df.columns = [col.strip().lower() for col in df.columns]
        
        # 빈 값 처리
        df = df.fillna('')
        
        logger.info("TSV 로드 완료: %s (%d rows)", file_path, len(df))
        return df
        
    except Exception as e:
        logger.error("TSV 로드 실패 %s: %s", file_path, e)
        return pd.DataFrame()


def load_specdb_with_repair(file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    사양값DB 로드 (컬럼 수 불일치 복구)
    
    일부 행이 컬럼 수가 맞지 않을 수 있음 → 복구 시도
    """
    if not os.path.exists(file_path):
        logger.warning("파일 없음: %s", file_path)
        return pd.DataFrame()
    
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            lines = f.readlines()
        
        if not lines:
            return pd.DataFrame()
        
        # 헤더 파싱
        header_line = lines[0].strip()
        headers = [h.strip().lower() for h in header_line.split('\t')]
        expected_cols = len(headers)
        
        # 데이터 행 파싱 (컬럼 수 복구)
        rows = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) == expected_cols:
                rows.append(parts)
            elif len(parts) > expected_cols:
                # 초과 컬럼 병합
                rows.append(parts[:expected_cols-1] + ['\t'.join(parts[expected_cols-1:])])
            elif len(parts) < expected_cols:
                # 부족 컬럼 패딩
                rows.append(parts + [''] * (expected_cols - len(parts)))
        
        df = pd.DataFrame(rows, columns=headers)
        df = df.fillna('')
        
        logger.info("사양값DB 복구 로드: %s (%d rows)", file_path, len(df))
        return df
        
    except Exception as e:
        logger.error("사양값DB 로드 실패 %s: %s", file_path, e)
        return pd.DataFrame()


# =============================================================================
# Config 데이터클래스
# =============================================================================

@dataclass
class Config:
    """설정 데이터클래스"""
    # 추출 모드
    extraction_mode: str = "light"
    data_source_mode: str = "file"
    
    # 출력 설정
    save_json: bool = True
    save_csv: bool = True
    save_debug_csv: bool = True
    save_to_db: bool = False
    print_json: bool = True
    
    # 경로
    base_folder: str = ""
    light_mode_pos_folder: str = ""
    glossary_path: str = ""
    spec_path: str = ""
    specdb_path: str = ""
    output_path: str = ""
    partial_output_path: str = ""
    enable_checkpoint: bool = True
    
    # DB 설정
    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""
    db_table_glossary: str = ""
    db_table_specdb: str = ""
    db_table_template: str = ""
    db_table_result: str = ""
    
    # pos_embedding 설정
    use_precomputed_embeddings: bool = True
    embedding_table_name: str = "pos_embedding"
    embedding_top_k: int = 5
    embedding_similarity_threshold: float = 0.65
    
    # LLM 설정
    use_llm: bool = True
    ollama_model: str = "qwen2.5:32b"
    ollama_bin: str = ""
    ollama_models_dir: str = ""
    ollama_timeout: int = 180
    ollama_host: str = "127.0.0.1"
    ollama_ports: List[int] = field(default_factory=lambda: [11434])
    ollama_num_ctx: int = 32768  # Context length (tokens) - Gemma3 supports up to 128K
    auto_start_ollama: bool = True
    ollama_start_grace_sec: int = 10
    
    # 투표 설정
    vote_enabled: bool = True
    vote_k: int = 2
    vote_min_agreement: int = 2
    
    # 병렬 처리
    enable_parallel: bool = True
    num_workers: int = 30
    llm_workers: int = 2
    
    # 배치 처리
    batch_size: int = 15
    max_evidence_chars: int = 8000  # 8KB (with prompt ~10-12KB = 2.5-3K tokens)
    
    # LLM 안정화
    llm_rate_limit_sec: float = 0.3
    llm_max_retries: int = 3
    llm_retry_sleep_sec: float = 1.5
    llm_temperature: float = 0.0
    
    # LLM Audit
    enable_llm_audit: bool = True
    audit_confidence_threshold: float = 0.85
    audit_always_for_llm: bool = True
    enable_llm_fallback: bool = True
    rule_conf_threshold: float = 0.72
    
    # 시멘틱 검색
    enable_semantic_search: bool = True
    semantic_model_path: str = "/workspace/bge-m3"
    semantic_device: str = "cuda"
    semantic_similarity_threshold: float = 0.65
    semantic_code_match_boost: float = 0.2
    semantic_exact_match_threshold: float = 0.90
    semantic_top_k: int = 5
    enable_llm_similarity_validation: bool = False
    semantic_cache_dir: str = "/workspace/cache/embeddings"
    
    # Light 모드 최적화
    light_mode_batch_disabled: bool = True
    light_mode_skip_hybrid_match: bool = True
    light_mode_workers: int = 4  # 병렬 worker 수 (3-6 권장)

    # 값 검증 설정
    enable_value_validation: bool = True
    # numeric_variance_threshold 제거 - 과거 값 대비 variance check는 부적절
    min_value_length: int = 1
    max_value_length: int = 200

    # 복합값/범위값 파싱
    split_compound_values: bool = True
    split_range_values: bool = True

    # Full 모드 전용 설정
    full_mode_batch_size: int = 15
    full_mode_checkpoint_interval: int = 50
    full_mode_voting_enabled: bool = True
    full_mode_vote_k: int = 3
    full_mode_audit_enabled: bool = True
    full_mode_process_all_files: bool = True
    full_mode_checkpoint_dir: str = ""


def build_config() -> Config:
    """사용자 설정을 Config 객체로 변환"""
    return Config(
        # 추출 모드 (맨 위에서 설정)
        extraction_mode=EXTRACTION_MODE,
        data_source_mode=DATA_SOURCE_MODE,
        
        # 출력 설정
        save_json=SAVE_JSON,
        save_csv=SAVE_CSV,
        save_debug_csv=SAVE_DEBUG_CSV,
        save_to_db=SAVE_TO_DB,
        print_json=PRINT_JSON,
        
        # 경로
        base_folder=USER_BASE_FOLDER,
        light_mode_pos_folder=LIGHT_MODE_POS_FOLDER,
        glossary_path=USER_GLOSSARY_PATH,
        spec_path=USER_SPEC_PATH,
        specdb_path=USER_SPECDB_PATH,
        output_path=USER_OUTPUT_PATH,
        partial_output_path=USER_PARTIAL_OUTPUT_PATH,
        enable_checkpoint=USER_ENABLE_CHECKPOINT,
        
        # DB 설정
        db_host=USER_DB_HOST,
        db_port=USER_DB_PORT,
        db_name=USER_DB_NAME,
        db_user=USER_DB_USER,
        db_password=USER_DB_PASSWORD,
        db_table_glossary=USER_DB_TABLE_GLOSSARY,
        db_table_specdb=USER_DB_TABLE_SPECDB,
        db_table_template=USER_DB_TABLE_TEMPLATE,
        db_table_result="",
        
        # pos_embedding 설정
        use_precomputed_embeddings=USE_PRECOMPUTED_EMBEDDINGS,
        embedding_table_name=EMBEDDING_TABLE_NAME,
        embedding_top_k=EMBEDDING_TOP_K,
        embedding_similarity_threshold=EMBEDDING_SIMILARITY_THRESHOLD,
        
        # LLM 설정
        use_llm=USER_USE_LLM,
        ollama_model=USER_OLLAMA_MODEL,
        ollama_bin=USER_OLLAMA_BIN,
        ollama_models_dir=USER_OLLAMA_MODELS_DIR,
        ollama_timeout=USER_OLLAMA_TIMEOUT_SEC,
        ollama_host=USER_OLLAMA_HOST,
        ollama_ports=USER_OLLAMA_PORTS,
        auto_start_ollama=USER_AUTO_START_OLLAMA_SERVE,
        ollama_start_grace_sec=USER_OLLAMA_SERVE_START_GRACE_SEC,
        vote_enabled=USER_VOTE_ENABLED,
        vote_k=USER_VOTE_K,
        vote_min_agreement=USER_VOTE_MIN_AGREEMENT,
        enable_parallel=USER_ENABLE_PARALLEL,
        num_workers=USER_NUM_WORKERS,
        llm_workers=USER_LLM_WORKERS,
        batch_size=USER_BATCH_SIZE,
        max_evidence_chars=USER_MAX_EVIDENCE_CHARS,
        llm_rate_limit_sec=USER_LLM_RATE_LIMIT_SEC,
        llm_max_retries=USER_LLM_MAX_RETRIES,
        llm_retry_sleep_sec=USER_LLM_RETRY_SLEEP_SEC,
        llm_temperature=USER_LLM_TEMPERATURE,
        enable_llm_audit=USER_ENABLE_LLM_AUDIT,
        audit_confidence_threshold=USER_AUDIT_CONFIDENCE_THRESHOLD,
        audit_always_for_llm=USER_AUDIT_ALWAYS_FOR_LLM,
        enable_llm_fallback=USER_ENABLE_LLM_FALLBACK,
        rule_conf_threshold=USER_RULE_CONF_THRESHOLD,
        
        # 시멘틱 검색
        enable_semantic_search=USER_ENABLE_SEMANTIC_SEARCH,
        semantic_model_path=USER_SEMANTIC_MODEL_PATH,
        semantic_device=USER_SEMANTIC_DEVICE,
        semantic_similarity_threshold=USER_SEMANTIC_SIMILARITY_THRESHOLD,
        semantic_code_match_boost=USER_SEMANTIC_CODE_MATCH_BOOST,
        semantic_exact_match_threshold=USER_SEMANTIC_EXACT_MATCH_THRESHOLD,
        semantic_top_k=USER_SEMANTIC_TOP_K,
        enable_llm_similarity_validation=USER_ENABLE_LLM_SIMILARITY_VALIDATION,
        semantic_cache_dir=USER_SEMANTIC_CACHE_DIR,
        
        # Light 모드 최적화
        light_mode_batch_disabled=LIGHT_MODE_BATCH_DISABLED,
        light_mode_skip_hybrid_match=LIGHT_MODE_SKIP_HYBRID_MATCH,
        light_mode_workers=LIGHT_MODE_WORKERS,

        # 값 검증 설정
        enable_value_validation=ENABLE_VALUE_VALIDATION,
        # numeric_variance_threshold 제거
        min_value_length=MIN_VALUE_LENGTH,
        max_value_length=MAX_VALUE_LENGTH,

        # 복합값/범위값 파싱
        split_compound_values=SPLIT_COMPOUND_VALUES,
        split_range_values=SPLIT_RANGE_VALUES,
    )


# =============================================================================
# SpecItem 데이터클래스
# =============================================================================

@dataclass
class SpecItem:
    """추출할 사양 항목"""
    spec_name: str = ""          # umgv_desc
    spec_code: str = ""          # umgv_code
    equipment: str = ""          # mat_attr_desc
    category: str = ""           # umg_desc
    pmg_desc: str = ""
    pmg_code: str = ""
    umg_code: str = ""
    expected_unit: str = ""      # umgv_uom
    hull: str = ""               # 호선 번호
    pos: str = ""                # POS 번호
    matnr: str = ""
    extwg: str = ""
    file_path: str = ""          # HTML 파일 경로
    raw_data: Dict = field(default_factory=dict)
    
    def get_composite_text(self) -> str:
        """복합 메타데이터 텍스트 반환 (임베딩용)"""
        return build_composite_text(
            self.pmg_desc, self.category, self.equipment, self.spec_name
        )


@dataclass
class ExtractionResult:
    """추출 결과"""
    spec_item: SpecItem
    value: str = ""
    unit: str = ""
    chunk: str = ""
    method: str = ""
    confidence: float = 0.0
    evidence: str = ""
    errors: List[str] = field(default_factory=list)
    similarity_info: Dict = field(default_factory=dict)
    reference_source: str = ""
    found: bool = True 
    validation_status: str = ""  # "valid", "invalid", "warning", ""
    validation_message: str = "" 
    compound_values: List[Tuple[str, str]] = field(default_factory=list) 
    # POS 원문 텍스트 보존
    original_spec_name: str = ""  # POS에 적힌 그대로의 사양명 (소문자, 특수문자 등 보존)
    original_unit: str = ""  # POS에 적힌 그대로의 단위
    original_equipment: str = ""  # POS에 적힌 그대로의 장비명


# ############################################################################
# 동의어 관리자 (DB 기반 - 하드코딩 제거)
# ############################################################################

class SynonymManager:
    """
    DB 기반 동의어 관리자
    - pos_dict의 umgv_desc(표준명) ↔ pos_umgv_desc(유의어) 매핑
    - 하드코딩 동의어 제거
    """

    def __init__(self):
        self.logger = logging.getLogger("SynonymManager")

        # 표준명 → 유의어 목록
        self.std_to_synonyms: Dict[str, Set[str]] = defaultdict(set)

        # 유의어 → 표준명 역매핑
        self.synonym_to_std: Dict[str, str] = {}

        # umgv_code → 모든 관련 용어
        self.code_to_terms: Dict[str, Set[str]] = defaultdict(set)

    def build_from_glossary(self, df_glossary: pd.DataFrame):
        """용어집에서 동의어 매핑 구축"""
        self.std_to_synonyms.clear()
        self.synonym_to_std.clear()
        self.code_to_terms.clear()

        if df_glossary.empty:
            self.logger.warning("빈 용어집으로 동의어 매핑 구축 불가")
            return

        # 필요한 컬럼 확인
        required_cols = ['umgv_desc', 'pos_umgv_desc']
        for col in required_cols:
            if col not in df_glossary.columns:
                self.logger.warning(f"용어집에 {col} 컬럼 없음")
                return

        for _, row in df_glossary.iterrows():
            std_name = norm(row.get('umgv_desc', ''))
            pos_name = norm(row.get('pos_umgv_desc', ''))
            umgv_code = norm(row.get('umgv_code', ''))

            if not std_name:
                continue

            std_upper = std_name.upper()

            # 표준명 자체도 검색어로 등록
            self.std_to_synonyms[std_upper].add(std_name)
            self.code_to_terms[umgv_code].add(std_name)

            # 유의어 등록
            if pos_name and pos_name != std_name:
                pos_upper = pos_name.upper()

                # 표준명 → 유의어
                self.std_to_synonyms[std_upper].add(pos_name)

                # 유의어 → 표준명
                self.synonym_to_std[pos_upper] = std_name

                # 코드 → 용어
                self.code_to_terms[umgv_code].add(pos_name)

        total_mappings = sum(len(v) for v in self.std_to_synonyms.values())
        self.logger.info(f"동의어 매핑 구축: {len(self.std_to_synonyms)}개 표준명, {total_mappings}개 총 매핑")

    def get_synonyms(self, term: str) -> List[str]:
        """용어의 동의어 목록 반환"""
        if not term:
            return []

        term_upper = term.upper().strip()
        synonyms = set()

        # 1. 표준명으로 검색
        if term_upper in self.std_to_synonyms:
            synonyms.update(self.std_to_synonyms[term_upper])

        # 2. 유의어로 검색 (역매핑)
        if term_upper in self.synonym_to_std:
            std_name = self.synonym_to_std[term_upper]
            synonyms.update(self.std_to_synonyms.get(std_name.upper(), set()))

        # 3. 부분 매칭 (표준명에 포함된 경우)
        for std, syns in self.std_to_synonyms.items():
            if term_upper in std or std in term_upper:
                synonyms.update(syns)

        # 원본 제외하고 반환
        synonyms.discard(term)
        return list(synonyms)

    def get_all_search_terms(self, term: str, umgv_code: str = "") -> List[str]:
        """검색에 사용할 모든 용어 반환 (우선순위순)"""
        terms = []
        term_upper = term.upper().strip() if term else ""

        # 1순위: 원본 용어
        if term:
            terms.append(term)

        # 2순위: umgv_code로 연결된 모든 용어
        if umgv_code and umgv_code in self.code_to_terms:
            for t in self.code_to_terms[umgv_code]:
                if t not in terms:
                    terms.append(t)

        # 3순위: 동의어
        synonyms = self.get_synonyms(term)
        for syn in synonyms:
            if syn not in terms:
                terms.append(syn)

        # 4순위: 토큰화된 부분 (3글자 이상)
        if term:
            parts = re.split(r'[_\s\-/()]+', term)
            for p in parts:
                if len(p) >= 3 and p not in terms:
                    terms.append(p)

        # 중복 제거 및 길이순 정렬 (긴 것이 더 구체적)
        seen = set()
        unique = []
        for t in terms:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique.append(t)

        unique.sort(key=len, reverse=True)
        return unique[:20]  # 최대 20개


# ############################################################################
# 값 검증기 (Phase 4)
# ############################################################################

class ValueValidator:
    """
    추출된 값 검증 (Phase 4)
    - 숫자형 사양 검증
    - 단위 비교
    - 과거 값 범위 검증
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("ValueValidator")

    def validate(self, result: ExtractionResult, spec: SpecItem) -> ExtractionResult:
        """추출 결과 검증 (과거 값 비교 제외 - 사양값은 변경 가능)"""
        if not self.config.enable_value_validation:
            return result

        if not result.found or not result.value:
            return result

        validation_issues = []

        # 1. 길이 검증
        if len(result.value) < self.config.min_value_length:
            validation_issues.append("값이 너무 짧음")
        if len(result.value) > self.config.max_value_length:
            validation_issues.append("값이 너무 김")
            result.confidence *= 0.5

        # 2. 숫자형 사양 검증
        if is_numeric_spec(spec.spec_name):
            numeric_val = extract_numeric_value(result.value)

            if numeric_val is None:
                # 숫자형인데 숫자가 없음 → 잘못된 추출 가능성
                validation_issues.append("숫자형 사양이나 숫자 없음")
                result.confidence *= 0.6

        # 3. 단위 검증
        if spec.expected_unit and result.unit:
            if not self._units_compatible(spec.expected_unit, result.unit):
                validation_issues.append(f"단위 불일치: 기대={spec.expected_unit}, 추출={result.unit}")
                result.confidence *= 0.8

        # 4. 값이 키워드 자체인지 확인 (잘못된 추출)
        if self._is_likely_keyword(result.value, spec):
            validation_issues.append("추출된 값이 키워드일 가능성")
            result.confidence *= 0.4

        # 결과 업데이트
        if validation_issues:
            result.validation_status = "warning" if result.confidence > 0.5 else "invalid"
            result.validation_message = "; ".join(validation_issues)
        else:
            result.validation_status = "valid"

        return result

    def _units_compatible(self, expected: str, actual: str) -> bool:
        """단위 호환성 검사"""
        if not expected or not actual:
            return True

        expected_norm = expected.lower().replace(' ', '').replace('.', '')
        actual_norm = actual.lower().replace(' ', '').replace('.', '')

        # 정확히 일치
        if expected_norm == actual_norm:
            return True

        # 포함 관계
        if expected_norm in actual_norm or actual_norm in expected_norm:
            return True

        # 동등한 단위 (DB에서 로드하는 것이 이상적이나, 기본적인 것만)
        unit_equivalents = {
            ('m3/h', 'm³/h'), ('m3', 'm³'), ('℃', '°c', 'degc'),
            ('kw', 'kilowatt'), ('rpm', 'r/min', 'rev/min'),
            ('bar', 'barg', 'bara'), ('mm', 'millimeter'),
        }

        for equiv_group in unit_equivalents:
            if expected_norm in equiv_group and actual_norm in equiv_group:
                return True

        return False

    def _is_likely_keyword(self, value: str, spec: SpecItem) -> bool:
        """추출된 값이 키워드(사양명)일 가능성 체크"""
        value_upper = value.upper().strip()

        # 사양명과 유사
        if value_upper == spec.spec_name.upper():
            return True

        # 일반적인 헤더 키워드
        header_keywords = ['type', 'qty', "q'ty", 'remark', 'unit', 'item', 'description',
                          'spec', 'specification', 'parameter', 'value', 'no.', 'no']
        if value_upper in [kw.upper() for kw in header_keywords]:
            return True

        return False


# =============================================================================
# PostgreSQL 연결 (pos_embedding 테이블 연동)
# =============================================================================

class PostgresEmbeddingLoader:
    """
    PostgreSQL에서 pos_embedding 테이블 조회
    
    사전 임베딩된 데이터를 활용하여 유사도 검색 수행
    """
    
    def __init__(self, config: Config, logger: logging.Logger = None):
        self.config = config
        self.log = logger or logging.getLogger("PGEmbedding")
        self.conn = None
        self._connect()
    
    def _connect(self) -> bool:
        """PostgreSQL 연결"""
        if not HAS_PSYCOPG2:
            self.log.warning("psycopg2 미설치. DB 임베딩 비활성화")
            return False
        
        try:
            self.conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                dbname=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
                connect_timeout=10
            )
            self.log.info("PostgreSQL 연결 성공: %s:%d/%s", 
                         self.config.db_host, self.config.db_port, self.config.db_name)
            return True
        except Exception as e:
            self.log.error("PostgreSQL 연결 실패: %s", e)
            return False
    
    def search_similar_embeddings(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        threshold: float = 0.65
    ) -> List[Dict[str, Any]]:
        """
        코사인 유사도로 유사한 임베딩 검색
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            top_k: 상위 K개 반환
            threshold: 유사도 임계값
            
        Returns:
            유사한 레코드 리스트
        """
        if not self.conn:
            return []
        
        try:
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # PostgreSQL에서 코사인 유사도 계산
            # pgvector 확장이 있다면 <=> 연산자 사용 가능
            # 없으면 Python에서 계산
            
            query = f"""
                SELECT doknr, dokvr, matnr, extwg, embedding_key, embedding,
                       created_on, updated_on
                FROM {self.config.embedding_table_name}
                LIMIT 1000
            """
            cur.execute(query)
            rows = cur.fetchall()
            cur.close()
            
            if not rows:
                return []
            
            # Python에서 코사인 유사도 계산
            results = []
            for row in rows:
                emb_str = row.get('embedding', '')
                if not emb_str:
                    continue
                
                # 임베딩 문자열 파싱
                try:
                    if isinstance(emb_str, str):
                        # "[0.1, 0.2, ...]" 형식
                        emb_str = emb_str.strip('[]')
                        db_embedding = [float(x) for x in emb_str.split(',')]
                    else:
                        db_embedding = list(emb_str)
                except:
                    continue
                
                # 코사인 유사도 계산
                similarity = self._cosine_similarity(query_embedding, db_embedding)
                
                if similarity >= threshold:
                    results.append({
                        'doknr': row['doknr'],
                        'matnr': row['matnr'],
                        'extwg': row['extwg'],
                        'embedding_key': row['embedding_key'],
                        'similarity': similarity
                    })
            
            # 유사도 내림차순 정렬
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            self.log.error("임베딩 검색 실패: %s", e)
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def get_embedding_by_key(self, embedding_key: str) -> Optional[Dict]:
        """embedding_key로 레코드 조회"""
        if not self.conn:
            return None

        try:
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            query = f"""
                SELECT * FROM {self.config.embedding_table_name}
                WHERE embedding_key = %s
                LIMIT 1
            """
            cur.execute(query, (embedding_key,))
            row = cur.fetchone()
            cur.close()
            return dict(row) if row else None
        except Exception as e:
            self.log.error("임베딩 조회 실패: %s", e)
            return None

    def search_by_key_with_fallback(
        self,
        search_key: str,
        query_text: str = "",
        embedding_model: Any = None,
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        하이브리드 검색 - search_key exact match → embedding similarity fallback

        Schema (pos_embedding table):
        - search_key: hull_pmg_code_umg_code_extwg (exact match용)
        - embedding_key: hull_pmg_desc_umg_desc_mat_attr_desc (similarity용)
        - embedding: BGE-M3 vector

        Args:
            search_key: 정확 매칭용 키 (hull_pmg_code_umg_code_extwg)
            query_text: 유사도 검색용 텍스트 (embedding_key 생성)
            embedding_model: BGE-M3 모델 (유사도 검색 시 사용)
            top_k: 반환할 최대 결과 수
            similarity_threshold: 유사도 임계값

        Returns:
            매칭된 레코드 리스트 (exact match 우선, 없으면 similarity)
        """
        if not self.conn:
            return []

        # Step 1: search_key exact match 시도
        try:
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            query = f"""
                SELECT * FROM {self.config.embedding_table_name}
                WHERE search_key = %s
                LIMIT {top_k}
            """
            cur.execute(query, (search_key,))
            rows = cur.fetchall()
            cur.close()

            if rows:
                self.log.debug(f"search_key exact match 성공: {search_key} ({len(rows)} rows)")
                return [dict(row) for row in rows]

        except Exception as e:
            self.log.debug(f"search_key 조회 실패: {e}")

        # Step 2: Embedding similarity fallback
        if not query_text or not embedding_model:
            self.log.debug("query_text 또는 embedding_model 없음, similarity fallback 생략")
            return []

        try:
            # Query text를 embedding으로 변환
            query_embedding = embedding_model.encode([query_text])[0].tolist()

            # embedding_key 기반 유사도 검색
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            query = f"""
                SELECT * FROM {self.config.embedding_table_name}
                LIMIT 500
            """
            cur.execute(query)
            rows = cur.fetchall()
            cur.close()

            if not rows:
                return []

            # Python에서 코사인 유사도 계산
            results = []
            for row in rows:
                emb_str = row.get('embedding', '')
                if not emb_str:
                    continue

                try:
                    if isinstance(emb_str, str):
                        emb_str = emb_str.strip('[]')
                        db_embedding = [float(x) for x in emb_str.split(',')]
                    else:
                        db_embedding = list(emb_str)
                except:
                    continue

                similarity = self._cosine_similarity(query_embedding, db_embedding)

                if similarity >= similarity_threshold:
                    row_dict = dict(row)
                    row_dict['similarity'] = similarity
                    results.append(row_dict)

            # 유사도 내림차순 정렬
            results.sort(key=lambda x: x['similarity'], reverse=True)

            if results:
                self.log.debug(f"Embedding similarity fallback 성공: {len(results)} matches")

            return results[:top_k]

        except Exception as e:
            self.log.error(f"Embedding similarity fallback 실패: {e}")
            return []
    
    def load_template_from_db(self, table_name: str = "") -> pd.DataFrame:
        """DB에서 추출 템플릿 로드"""
        if not self.conn:
            return pd.DataFrame()
        
        table_name = table_name or self.config.db_table_template
        
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.conn)
            df.columns = [col.lower() for col in df.columns]
            self.log.info("템플릿 로드 (DB): %d rows", len(df))
            return df
        except Exception as e:
            self.log.error("템플릿 로드 실패: %s", e)
            return pd.DataFrame()
    
    def close(self):
        """연결 종료"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def load_glossary_from_db(self, table_name: str = "") -> pd.DataFrame:
        """
        DB에서 용어집(pos_dict) 로드
        
        테이블 구조: umgv_code, umgv_desc, ...
        """
        if not self.conn:
            return pd.DataFrame()
        
        table_name = table_name or self.config.db_table_glossary
        
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.conn)
            df.columns = [col.lower() for col in df.columns]
            self.log.info("용어집 로드 (DB): %s → %d rows", table_name, len(df))
            return df
        except Exception as e:
            self.log.error("용어집 DB 로드 실패: %s", e)
            return pd.DataFrame()
    
    def load_specdb_from_db(self, table_name: str = "") -> pd.DataFrame:
        """
        DB에서 사양값DB(umgv_fin) 로드
        
        테이블 구조: matnr, umgv_desc, umgv_value_edit, ...
        """
        if not self.conn:
            return pd.DataFrame()
        
        table_name = table_name or self.config.db_table_specdb
        
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.conn)
            df.columns = [col.lower() for col in df.columns]
            self.log.info("사양값DB 로드 (DB): %s → %d rows", table_name, len(df))
            return df
        except Exception as e:
            self.log.error("사양값DB DB 로드 실패: %s", e)
            return pd.DataFrame()


# =============================================================================
# 개선된 Pre-Check 로직 (섹션 번호 vs 소수점 구분)
# =============================================================================

class ImprovedPreChecker:
    """
    개선된 사전 검사기
    
    주요 개선:
    - 섹션 번호 vs 소수점 구분 강화
    - HTML 구조 및 chunk 위치 고려
    - 연결된 숫자 판단 개선
    """
    
    # 수치형 사양 키워드
    NUMERIC_SPECS = [
        'CAPACITY', 'SPEED', 'PRESSURE', 'HEAD', 'LENGTH', 'WIDTH', 'HEIGHT',
        'THICKNESS', 'OUTPUT', 'POWER', 'TEMPERATURE', 'DENSITY', 'FLOW',
        'RATE', 'VOLUME', 'DIAMETER', 'WEIGHT', 'MASS', 'VOLTAGE', 'CURRENT',
        'FREQUENCY', 'RPM', 'QUANTITY', 'QTY', 'NO.', 'NUMBER'
    ]
    
    # 텍스트형 사양 키워드 (정확 매칭 필요 - 단독 TYPE만)
    # "Digital type gyro repeater"는 수량 사양이므로 제외
    TEXT_SPECS = [
        'MATERIAL', 'METHOD', 'PLACE', 'REFRIGERANT TYPE',
        'DRIVEN TYPE', 'CLASSIFICATION', 'STARTING METHOD', 'COVER TOP TYPE',
        'COUPLING TYPE', 'MEASURING PRINCIPLE', 'TIER III TECH',
        'EQUIPMENT TYPE', 'MODEL TYPE', 'ENGINE TYPE', 'SHIP TYPE',
        'PROPELLER TYPE', 'PUMP TYPE', 'TYPE & MAKER'
    ]
    
    # 수량/장비 관련 키워드 (숫자 값이 정상인 경우)
    QUANTITY_SPECS = [
        'REPEATER', 'INDICATOR', 'UNIT', 'SET', 'DISPLAY', 'TELEPHONE',
        'LEVER', 'COVER', 'BOARD', 'PANEL', 'ABSORBER', 'CYLINDER'
    ]
    
    # 소수점 숫자로 흔히 사용되는 패턴 (섹션이 아님)
    DECIMAL_PATTERNS = [
        r'^\d+\.\d+\s*(mm|m|cm|kg|bar|kW|MW|%|°C|℃)$',  # 단위 붙은 소수
        r'^\d+\.\d{1,3}$',  # 소수점 3자리 이하
        r'^[+-]?\d+\.\d+$',  # 부호 있는 소수
    ]
    
    # 섹션 번호 특성 (HTML 위치 기반)
    SECTION_INDICATORS = [
        r'^<h[1-6]',        # 헤딩 태그 시작
        r'^\d+\.\d+\s+[A-Z]',  # 섹션 번호 후 대문자 (제목)
        r'SECTION\s+\d+',   # SECTION 키워드
        r'CHAPTER\s+\d+',   # CHAPTER 키워드
    ]
    
    def __init__(self, specdb_index=None):
        self.specdb = specdb_index
    
    def check(
        self, 
        umgv_desc: str, 
        value: str, 
        unit: str = "", 
        equipment: str = "",
        chunk_context: str = "",
        html_context: str = ""
    ) -> List[str]:
        """
        추출 결과 사전 검사
        
        Args:
            umgv_desc: 사양 설명
            value: 추출된 값
            unit: 추출된 단위
            equipment: 장비명
            chunk_context: chunk 문맥 (값 주변 텍스트)
            html_context: HTML 문맥 (태그 정보)
        
        Returns:
            오류 코드 리스트 (빈 리스트면 유효)
        """
        errors = []
        
        if not value:
            return errors
        
        value = value.strip()
        unit = unit.strip() if unit else ""
        umgv_upper = umgv_desc.upper()
        
        # A1: 수치형 사양에 텍스트
        if self._is_numeric_spec(umgv_upper):
            if not self._has_number(value) and not self._is_applicability(umgv_upper):
                # 예외: 제어 신호 패턴 (BMS 문서)
                # "Au", "AuL", "AuBCCECC*", "Au*" 같은 제어 신호 약어 조합
                if not self._is_control_signal_value(value):
                    errors.append("A1_NUMERIC_SPEC_TEXT_VALUE")
        
        # A2: 텍스트형 사양에 순수 숫자
        if self._is_text_spec(umgv_upper) and self._is_pure_numeric(value):
            errors.append("A2_TEXT_SPEC_NUMERIC_VALUE")
        
        # B1: 섹션 번호 형식 (개선된 판단)
        if self._is_section_number_improved(value, chunk_context, html_context):
            errors.append("B1_SECTION_NUMBER")
        
        # B2: 연결된 숫자 (개선된 판단)
        # x.x.x 형태가 섹션인지 연결 오류인지 구분
        if self._is_concatenated_number_improved(value, chunk_context):
            errors.append("B2_CONCATENATED_NUMBER")
        
        # B3: 제어문자
        if self._has_control_chars(value):
            errors.append("B3_CONTROL_CHARS")
        
        # E3: 날짜 패턴
        if self._is_date_pattern(value):
            errors.append("E3_DATE_PATTERN")
        
        # F1: 한글 오염 (허용 한글 제외)
        if self._has_korean_contamination(value):
            errors.append("F1_KOREAN_CONTAMINATION")
        
        # G1: 너무 긴 값 (200자 초과) - 문장이 추출된 경우 (완화)
        # 주의: TYPE 같은 사양은 긴 설명 값을 가질 수 있음
        if len(value) > 200 and not self._is_text_spec(umgv_upper):
            errors.append("G1_VALUE_TOO_LONG")
        
        # G2: 불완전한 값 (끝이 잘린 경우) - 완화
        # 심각한 경우만 감지 (괄호 불일치)
        if value.count('(') != value.count(')') or value.count('[') != value.count(']'):
            errors.append("G2_INCOMPLETE_VALUE")
        
        # G3: 사양명/헤더가 값에 포함된 경우 - 비활성화 (너무 많은 false positive)
        # if self._contains_spec_name_in_value(umgv_upper, value):
        #     errors.append("G3_SPEC_NAME_IN_VALUE")
        
        return errors
    
    def _is_incomplete_value(self, value: str) -> bool:
        """불완전한 값 감지 (끝이 잘린 경우)"""
        # 열린 괄호로 끝나는 경우
        if value.endswith('(') or value.endswith('['):
            return True
        # 괄호 불일치
        if value.count('(') != value.count(')'):
            return True
        if value.count('[') != value.count(']'):
            return True
        # 숫자 뒤에 단위 시작 문자만 있는 경우 (예: "70 m" → 불완전)
        if re.search(r'\d+\s+[a-zA-Z]$', value):
            # 단, 유효한 단위인지 확인
            if not re.search(r'\d+\s*(m|mm|cm|kg|kW|MW|bar|rpm|Hz|V|A|EA|ea|set|pcs|kVA)$', value, re.I):
                return True
        # 잘린 단어 감지 (소문자로 끝나고 공백이 없는 3자 이하 단어)
        last_word = value.split()[-1] if value.split() else ""
        if len(last_word) <= 3 and last_word.islower() and not last_word.isdigit():
            # 유효한 단위가 아닌 경우
            valid_short_units = {'m', 'mm', 'cm', 'kg', 'kw', 'mw', 'bar', 'rpm', 'hz', 'v', 'a', 'ea', 'pcs', 'set'}
            if last_word.lower() not in valid_short_units:
                return True
        # "roo", "thi" 같은 잘린 단어 패턴
        if re.search(r'\b[a-z]{2,4}$', value) and not re.search(r'\d', value[-5:]):
            # 마지막 단어가 짧은 소문자이고 숫자가 없으면 잘린 것으로 판단
            last_part = value.split()[-1] if value.split() else ""
            if len(last_part) < 5 and last_part.islower():
                return True
        return False
    
    def _contains_spec_name_in_value(self, spec_name: str, value: str) -> bool:
        """사양명이 값에 포함된 경우 감지"""
        value_upper = value.upper()
        
        # 일반적인 사양명 패턴
        spec_keywords = [
            'SPECIFIC HEAT', 'THERMAL CONDUCTIVITY', 'DENSITY',
            'CAPACITY', 'OUTPUT', 'POWER', 'VOLTAGE', 'CURRENT',
            'TEMPERATURE', 'PRESSURE', 'HEAD', 'SPEED', 'RPM'
        ]
        
        # 값이 사양명으로 시작하는 경우 (잘못된 추출)
        for kw in spec_keywords:
            if value_upper.startswith(kw) and len(value) > len(kw) + 10:
                return True
        
        # 사양명 자체가 값에 있는 경우
        if spec_name in value_upper and len(spec_name) > 5:
            # 단, 사양명이 짧은 경우는 제외
            return True
        
        return False
    
    def _is_numeric_spec(self, umgv_upper: str) -> bool:
        """수치형 사양 여부"""
        return any(kw in umgv_upper for kw in self.NUMERIC_SPECS)
    
    def _is_text_spec(self, umgv_upper: str) -> bool:
        """텍스트형 사양 여부 (개선)"""
        # 수량/장비 관련 키워드가 있으면 텍스트 사양이 아님
        if any(kw in umgv_upper for kw in self.QUANTITY_SPECS):
            return False
        return any(kw in umgv_upper for kw in self.TEXT_SPECS)
    
    def _is_applicability(self, umgv_upper: str) -> bool:
        """적용성 사양 (Y/N/Q)"""
        return 'APPLICABILITY' in umgv_upper
    
    def _is_control_signal_value(self, value: str) -> bool:
        """
        제어 신호 값 여부 (BMS 문서용)
        
        "Au", "AuL", "AuBCCECC*", "Au*LCP" 같은 제어 신호 약어 조합을 인식
        
        제어 신호 약어:
        - Au: Automatic
        - L: Local
        - BCC: Bridge Control Console
        - ECC: Engine Control Console
        - ECR: Engine Control Room
        - LCP: Local Control Panel
        - Ind: Indicator
        - Alm: Alarm
        - Stp: Stop
        - *: 특수 표시
        """
        if not value:
            return False
        
        # 짧은 대문자 약어 조합 패턴
        # 예: "Au", "AuL", "AuBCCECC*", "Au*LCP*", "ECC (on CAMS*)ECR*LCP*"
        control_pattern = r'^[A-Za-z\*\(\)\s,\-]+$'
        
        if not re.match(control_pattern, value):
            return False
        
        # 제어 신호 키워드 포함 여부
        control_keywords = [
            'AU', 'BCC', 'ECC', 'ECR', 'LCP', 'IND', 'ALM', 'STP',
            'TRIP', 'SLDN', 'L', 'M', 'S', 'T', 'CAMS'
        ]
        
        value_upper = value.upper()
        keyword_count = sum(1 for kw in control_keywords if kw in value_upper)
        
        # 최소 1개 이상의 제어 키워드가 있으면 제어 신호로 판단
        return keyword_count >= 1
    
    def _has_number(self, value: str) -> bool:
        """숫자 포함 여부"""
        return bool(re.search(r'\d', value))
    
    def _is_pure_numeric(self, value: str) -> bool:
        """순수 숫자 여부"""
        return bool(re.match(r'^[\d.,\-+~]+$', value.strip()))
    
    def _is_section_number_improved(
        self, 
        value: str, 
        chunk_context: str = "",
        html_context: str = ""
    ) -> bool:
        """
        개선된 섹션 번호 판단
        
        HTML 구조와 chunk 문맥을 함께 고려
        """
        value = value.strip()
        
        # 기본 패턴: x.x 또는 x.x.x
        if not re.match(r'^\d+\.\d+(\.\d+)*$', value):
            return False
        
        # 단위가 붙어있으면 소수점 숫자 (섹션 아님)
        for pattern in self.DECIMAL_PATTERNS:
            # value + 주변 컨텍스트에서 단위 확인
            if chunk_context:
                # 값 바로 뒤에 단위가 있는지 확인
                value_pos = chunk_context.find(value)
                if value_pos >= 0:
                    after_value = chunk_context[value_pos + len(value):value_pos + len(value) + 10]
                    if re.match(r'^\s*(mm|m|cm|kg|bar|kW|MW|%|°C|℃|rpm|Hz)', after_value, re.I):
                        return False  # 단위 있음 → 소수점
        
        # HTML 태그 컨텍스트 확인
        if html_context:
            for indicator in self.SECTION_INDICATORS:
                if re.search(indicator, html_context, re.I):
                    return True  # 섹션 태그 안에 있음
        
        # x.x 형태에서 소수점 자릿수로 판단
        parts = value.split('.')
        if len(parts) == 2:
            # 정수부.소수부 형태
            int_part, dec_part = parts
            # 소수부가 1-3자리면 소수점 숫자일 가능성 높음
            if len(dec_part) <= 3 and len(dec_part) >= 1:
                # 추가 휴리스틱: 정수부가 작은 숫자(1-20)면 섹션 가능성
                if int_part.isdigit() and int(int_part) <= 20:
                    # chunk 문맥에서 섹션 키워드 확인
                    if chunk_context:
                        section_keywords = ['section', 'chapter', 'part', 'clause', '항', '조']
                        chunk_lower = chunk_context.lower()
                        if any(kw in chunk_lower for kw in section_keywords):
                            return True
                    # 애매하면 False (섹션 아님으로 판단)
                    return False
                else:
                    return False  # 정수부가 크면 소수점
        
        # x.x.x 이상 형태는 섹션일 가능성 높음
        if len(parts) >= 3:
            return True
        
        return False
    
    def _is_concatenated_number_improved(
        self, 
        value: str,
        chunk_context: str = ""
    ) -> bool:
        """
        개선된 연결 숫자 판단
        
        x.x.x 형태가 의도적 형식인지 오류인지 구분
        """
        value = value.strip()
        
        # 명백한 연결 오류: ..
        if '..' in value:
            return True
        
        # 명백한 연결 오류: 소수점이 여러 개인데 패턴이 불규칙
        # 예: 4.55.4 (중간 숫자가 2자리 이상이고 의미 없음)
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', value)
        if match:
            first, second, third = match.groups()
            # 버전 형태 (1.2.3)는 괜찮음
            if len(first) <= 2 and len(second) <= 2 and len(third) <= 2:
                return False  # 버전 또는 섹션 번호
            # 중간 숫자가 큰 경우 연결 오류일 가능성
            if len(second) >= 2 and int(second) > 20:
                return True
        
        # 4자리 이상 중간 숫자
        if re.match(r'^\d+\.\d{4,}\.\d+', value):
            return True
        
        return False
    
    def _has_control_chars(self, value: str) -> bool:
        """제어문자 포함 여부"""
        return bool(re.search(r'[\x00-\x1f\x7f-\x9f]', value))
    
    def _is_date_pattern(self, value: str) -> bool:
        """날짜 패턴 여부"""
        patterns = [
            r'^\d{4}[-/]\d{2}[-/]\d{2}$',
            r'^\d{2}[-/]\d{2}[-/]\d{4}$',
            r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
        ]
        value_lower = value.lower()
        return any(re.match(p, value_lower) for p in patterns)
    
    def _has_korean_contamination(self, value: str, source: str = "pos") -> bool:
        """
        한글 오염 여부 판단
        
        정책:
        - POS 문서에서 추출된 한글: 허용 (source="pos")
        - 용어집/사양값DB에서 온 한글 오염: 검사 (source="db")
        
        한글 오염의 정의:
        - 깨진 문자열 (예: "ì ´í¸°ê°") - 인코딩 문제
        - 매우 긴 한글 문장 (40자 이상) - 설명문이 섞인 경우
        
        허용되는 한글:
        - 단위: 개, 대, 식, 세트 등
        - 회사명: 강림, 삼성, 현대 등
        - 일반적인 한글 단어 (40자 미만)
        - 괄호 안의 한글
        
        Args:
            value: 검사할 값
            source: 값의 출처 ("pos" 또는 "db")
            
        Returns:
            True = 오염됨, False = 정상
        """
        # POS 문서에서 추출된 값은 한글 오염으로 보지 않음
        if source == "pos":
            # 단, 인코딩 깨진 문자는 검사
            if self._has_encoding_corruption(value):
                return True
            return False
        
        # DB/용어집에서 온 값에 대한 한글 오염 검사
        korean_chars = re.findall(r'[가-힣]+', value)
        
        if not korean_chars:
            return False
        
        # 인코딩 깨진 문자 패턴 검사
        if self._has_encoding_corruption(value):
            return True
        
        # 매우 긴 한글 문장 (40자 이상 연속 한글)
        for korean_word in korean_chars:
            if len(korean_word) > 40:
                return True
        
        # 한글 비율이 너무 높은 경우 (80% 이상)
        total_korean = sum(len(w) for w in korean_chars)
        if len(value) > 10 and total_korean / len(value) > 0.8:
            return True
        
        return False
    
    def _has_encoding_corruption(self, value: str) -> bool:
        """
        인코딩 깨짐 여부 판단
        
        깨진 UTF-8/EUC-KR 등에서 나타나는 특수 패턴 감지
        """
        # 깨진 문자 패턴 (UTF-8을 다른 인코딩으로 잘못 읽은 경우)
        corruption_patterns = [
            r'[ì|í|ë|ê|î|ï|ã|â|ä|å]+',  # UTF-8 → Latin-1
            r'[\x80-\x9f]+',  # 제어 문자
            r'M-[A-Z]M-',  # octal escape 패턴
        ]
        
        for pattern in corruption_patterns:
            if re.search(pattern, value):
                return True
        
        return False


# =============================================================================
# embedding_key 개선 제안
# =============================================================================

def generate_improved_embedding_key(
    hull: str,
    pmg_code: str = "",
    umg_code: str = "",
    extwg: str = "",
    pmg_desc: str = "",
    umg_desc: str = "",
    mat_attr_desc: str = ""
) -> str:
    """
    개선된 embedding_key 생성
    
    기존: hull_PMG_UMG_EXTWG (코드 기반, 정보 부족)
    개선: hull_PMG_UMG_EXTWG_DESC (설명 포함)
    
    제안:
    1. 코드 기반 key: 빠른 매칭용 (현재)
    2. 텍스트 기반 key: 임베딩 유사도용 (개선)
    """
    # 코드 기반 key (기존 호환)
    code_key = f"{hull}_{pmg_code}_{umg_code}_{extwg}"
    
    # 텍스트 기반 key (임베딩 품질 향상)
    # 장비명 + PMG설명 + UMG설명을 결합
    text_parts = []
    if mat_attr_desc:
        text_parts.append(mat_attr_desc.strip()[:50])
    if pmg_desc:
        text_parts.append(pmg_desc.strip()[:30])
    if umg_desc:
        text_parts.append(umg_desc.strip()[:30])
    
    text_key = " ".join(text_parts)
    
    return {
        'code_key': code_key,
        'text_key': text_key,
        'combined': f"{code_key} | {text_key}"
    }


# 더 나은 embedding_key 제안
EMBEDDING_KEY_RECOMMENDATION = """
## embedding_key 개선 제안

### 현재 문제
- 코드 기반 (`2357_961C_96109_YS96109`): 의미론적 정보 부족
- 유사한 장비라도 코드가 다르면 유사도 낮음

### 개선 방안 1: 텍스트 기반 임베딩
```
embedding_text = "{mat_attr_desc} {pmg_desc} {umg_desc}"
```
예: "MAIN ENGINE CONTROL SYSTEM ENGINE CONTROL"

### 개선 방안 2: 하이브리드 테이블
```sql
ALTER TABLE pos_embedding ADD COLUMN embedding_text TEXT;
ALTER TABLE pos_embedding ADD COLUMN embedding_vector_text VECTOR(1024);

-- 텍스트 기반 임베딩 추가
UPDATE pos_embedding SET 
    embedding_text = mat_attr_desc || ' ' || pmg_desc || ' ' || umg_desc,
    embedding_vector_text = embed(embedding_text);
```

### 개선 방안 3: 코드 + 텍스트 복합 키
현재 embedding_key 유지하면서, 별도 텍스트 임베딩 컬럼 추가
- 코드 매칭: 정확한 동일 장비 찾기
- 텍스트 매칭: 유사 장비 찾기
"""


# =============================================================================
# SemanticMatcher 클래스 (BGE-M3)
# =============================================================================

@dataclass
class SimilarityMatch:
    """유사도 매칭 결과"""
    record: Dict[str, Any]
    similarity_score: float
    code_match_score: float
    combined_score: float
    match_type: str
    match_details: str
    is_valid: bool


class SemanticMatcher:
    """
    시멘틱 유사도 기반 참조 매칭 시스템
    
    개선사항:
    - pos_embedding DB 연동
    - BGE-M3 + GPU 지원
    - 코드 매칭 + 시멘틱 매칭 하이브리드
    """
    
    def __init__(
        self,
        model_path: str = "/workspace/bge-m3",
        device: str = "cuda",
        similarity_threshold: float = 0.65,
        code_match_boost: float = 0.2,
        exact_match_threshold: float = 0.90,
        top_k: int = 5,
        cache_dir: str = "",
        pg_loader: PostgresEmbeddingLoader = None,
        logger: logging.Logger = None
    ):
        self.model_path = model_path
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.code_match_boost = code_match_boost
        self.exact_match_threshold = exact_match_threshold
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.pg_loader = pg_loader
        self.log = logger or logging.getLogger("SemanticMatcher")
        
        # 임베딩 모델 로드
        self.model = None
        self._model_loaded = False
        self._load_model()
        
        # 캐시
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def _load_model(self) -> None:
        """임베딩 모델 로드 (BGE-M3 + GPU)"""
        if not HAS_SENTENCE_TRANSFORMER:
            self.log.warning("sentence-transformers 미설치. 키워드 기반 폴백 사용")
            return
        
        try:
            self.log.info("임베딩 모델 로딩: %s (device: %s)", self.model_path, self.device)
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device
            )
            self._model_loaded = True
            self.log.info("임베딩 모델 로드 완료")
        except Exception as e:
            self.log.warning("임베딩 모델 로드 실패: %s. 키워드 기반 폴백 사용", e)
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """텍스트 임베딩 생성 (캐시 지원)"""
        if not text:
            return None
        
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if not self._model_loaded or self.model is None:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            self.log.warning("임베딩 생성 실패: %s", e)
            return None
    
    def search_similar_from_db(
        self, 
        query_text: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        DB에서 유사한 임베딩 검색 (pos_embedding 테이블 활용)
        """
        if not self.pg_loader:
            return []
        
        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            return []
        
        return self.pg_loader.search_similar_embeddings(
            query_embedding,
            top_k=top_k or self.top_k,
            threshold=self.similarity_threshold
        )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """키워드 기반 Jaccard 유사도 (폴백)"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(re.findall(r'\w+', text1.upper()))
        words2 = set(re.findall(r'\w+', text2.upper()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_code_match_score(
        self,
        query_pmg: str,
        query_umg: str,
        query_extwg: str,
        target_pmg: str,
        target_umg: str,
        target_extwg: str
    ) -> float:
        """코드 매칭 점수 계산"""
        score = 0.0
        
        if query_pmg and target_pmg and query_pmg == target_pmg:
            score += 0.3
        if query_umg and target_umg and query_umg == target_umg:
            score += 0.3
        if query_extwg and target_extwg and query_extwg == target_extwg:
            score += 0.4
        
        return score


# =============================================================================
# 간소화된 인덱스 클래스 (Light 모드 최적화)
# =============================================================================

class LightweightGlossaryIndex:
    """
    경량 용어집 인덱스 (Light 모드용)
    
    기능:
    1. umgv_code 기반 조회
    2. umgv_desc ↔ pos_umgv_desc 동의어 매핑 지원
    
    동의어 매핑 활용:
    - umgv_desc: 추출하려는 사양항목명 (표준명)
    - pos_umgv_desc: 과거 POS에 적혀 있던 사양항목명 (유의어)
    """
    
    def __init__(self, file_path: str = "", df: pd.DataFrame = None):
        self.file_path = file_path
        self.df = df
        self._index: Dict[str, Dict] = {}
        
        # 동의어 인덱스: pos_umgv_desc → umgv_desc 매핑
        self._synonym_index: Dict[str, str] = {}
        # 역방향 인덱스: umgv_desc → [pos_umgv_desc, ...]
        self._reverse_synonym_index: Dict[str, List[str]] = {}
        
        if file_path and os.path.exists(file_path):
            self.df = load_tsv(file_path)
        
        if self.df is not None and not self.df.empty:
            self._build_simple_index()
            self._build_synonym_index()
    
    def _build_simple_index(self):
        """단순 인덱스 구축 (빠른 초기화)"""
        for _, row in self.df.iterrows():
            key = norm(row.get('umgv_code', ''))
            if key:
                self._index[key] = row.to_dict()
    
    def _build_synonym_index(self):
        """
        동의어 인덱스 구축
        
        용어집의 umgv_desc와 pos_umgv_desc를 매핑하여
        POS 문서에서 다양한 표현을 인식할 수 있도록 함
        """
        for _, row in self.df.iterrows():
            umgv_desc = str(row.get('umgv_desc', '')).strip().upper()
            pos_umgv_desc = str(row.get('pos_umgv_desc', '')).strip().upper()
            
            # 유효한 동의어 쌍만 처리
            if umgv_desc and pos_umgv_desc and umgv_desc != pos_umgv_desc:
                # pos_umgv_desc → umgv_desc 매핑
                if pos_umgv_desc not in self._synonym_index:
                    self._synonym_index[pos_umgv_desc] = umgv_desc
                
                # 역방향: umgv_desc → [pos_umgv_desc, ...]
                if umgv_desc not in self._reverse_synonym_index:
                    self._reverse_synonym_index[umgv_desc] = []
                if pos_umgv_desc not in self._reverse_synonym_index[umgv_desc]:
                    self._reverse_synonym_index[umgv_desc].append(pos_umgv_desc)
    
    def lookup(self, umgv_code: str) -> Optional[Dict]:
        """코드로 조회"""
        return self._index.get(norm(umgv_code))
    
    def get_standard_name(self, name: str) -> str:
        """
        동의어를 표준명(umgv_desc)으로 변환
        
        Args:
            name: POS 문서에서 추출된 사양명
            
        Returns:
            표준명 (umgv_desc) 또는 원본
        """
        name_upper = name.strip().upper()
        return self._synonym_index.get(name_upper, name_upper)
    
    def get_synonyms(self, umgv_desc: str) -> List[str]:
        """
        표준명에 대한 모든 동의어 반환
        
        Args:
            umgv_desc: 표준 사양명
            
        Returns:
            동의어 리스트 (pos_umgv_desc들)
        """
        return self._reverse_synonym_index.get(umgv_desc.upper(), [])
    
    def is_synonym_match(self, spec_name: str, doc_key: str) -> bool:
        """
        사양명과 문서 키가 동의어 관계인지 확인
        
        Args:
            spec_name: 추출하려는 사양명 (umgv_desc)
            doc_key: POS 문서에서 추출된 키
            
        Returns:
            동의어 관계 여부
        """
        spec_upper = spec_name.strip().upper()
        doc_upper = doc_key.strip().upper()
        
        # 직접 매칭
        if spec_upper == doc_upper:
            return True
        
        # doc_key가 spec_name의 동의어인지 확인
        synonyms = self.get_synonyms(spec_upper)
        if doc_upper in synonyms:
            return True
        
        # doc_key를 표준명으로 변환 후 비교
        standard = self.get_standard_name(doc_upper)
        if standard == spec_upper:
            return True
        
        return False
    
    def get_all_records(self) -> List[Dict]:
        """모든 레코드 반환"""
        if self.df is None:
            return []
        return self.df.to_dict('records')
    
    def get_synonym_count(self) -> int:
        """동의어 쌍 수 반환"""
        return len(self._synonym_index)


class LightweightSpecDBIndex:
    """
    경량 사양값DB 인덱스 (Light 모드용)
    """
    
    def __init__(self, file_path: str = "", df: pd.DataFrame = None):
        self.file_path = file_path
        self.df = df
        self._index: Dict[str, List[Dict]] = {}
        
        if file_path and os.path.exists(file_path):
            self.df = load_specdb_with_repair(file_path)
        
        if self.df is not None and not self.df.empty:
            self._build_simple_index()
    
    def _build_simple_index(self):
        """단순 인덱스 구축"""
        for _, row in self.df.iterrows():
            hull = extract_hull_from_matnr(row.get('matnr', ''))
            umgv_desc = norm(row.get('umgv_desc', ''))
            
            if hull and umgv_desc:
                key = f"{hull}_{umgv_desc.upper()}"
                if key not in self._index:
                    self._index[key] = []
                self._index[key].append(row.to_dict())
    
    def lookup(self, hull: str, umgv_desc: str) -> List[Dict]:
        """hull + umgv_desc로 조회"""
        key = f"{hull}_{umgv_desc.upper()}"
        return self._index.get(key, [])
    
    def get_historical_values(self, hull: str, umgv_desc: str) -> List[str]:
        """과거 사양값 조회"""
        records = self.lookup(hull, umgv_desc)
        values = [norm(r.get('umgv_value_edit', '')) for r in records]
        return [v for v in values if v]


# =============================================================================
# Dynamic Knowledge Base (동적 지식 베이스 - PostgreSQL 기반)
# =============================================================================

class PostgresKnowledgeLoader:
    """
    PostgreSQL에서 동적 지식 로드 (인메모리 캐싱)

    보안망 환경을 고려하여 파일/SQLite 대신 인메모리 dict 사용:
    - 프로그램 시작 시 PostgreSQL에서 1회 로드
    - Python dict로 O(1) 조회
    - 프로세스 종료 시 자동 삭제

    테이블:
    - pos_dict: 용어집 (umgv_desc, pos_umgv_desc, umgv_uom, pos_umgv_uom, mat_attr_desc)
    - umgv_fin: 사양값DB (과거 추출 결과)
    """

    def __init__(self, config: Config = None, conn=None, logger: logging.Logger = None):
        """
        Args:
            config: DB 설정 (config 또는 conn 중 하나 필요)
            conn: 기존 PostgreSQL 연결 (재사용)
            logger: 로거
        """
        self.config = config
        self.conn = conn
        self.log = logger or logging.getLogger("PGKnowledgeLoader")
        self._own_connection = False

        # 인메모리 캐시
        self.synonym_forward = {}  # {standard: [variant1, variant2, ...]}
        self.synonym_reverse = {}  # {variant: standard}
        self.unit_forward = {}     # {standard_unit: [variant1, variant2, ...]}
        self.unit_reverse = {}     # {variant_unit: standard_unit}
        self.abbreviations = {}    # {abbrev: [full_form1, full_form2, ...]}
        self.mat_attr_index = {}   # {umgv_desc: mat_attr_desc}

        # 로드 상태
        self._loaded = False

        # 연결 확립
        if not self.conn and config:
            self._connect()

    def _connect(self) -> bool:
        """PostgreSQL 연결"""
        if not HAS_PSYCOPG2:
            self.log.warning("psycopg2 미설치. DB 지식 로더 비활성화")
            return False

        try:
            self.conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                dbname=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
                connect_timeout=10
            )
            self._own_connection = True
            self.log.info("PostgreSQL 지식 로더 연결 성공")
            return True
        except Exception as e:
            self.log.error("PostgreSQL 지식 로더 연결 실패: %s", e)
            return False

    def load_all(self) -> bool:
        """
        모든 지식을 PostgreSQL에서 로드 (1회 호출)

        Returns:
            성공 여부
        """
        if self._loaded:
            self.log.debug("이미 로드됨, 스킵")
            return True

        if not self.conn:
            self.log.error("DB 연결 없음")
            return False

        try:
            start_time = time.time()

            # 1. 동의어 로드 (pos_dict 테이블)
            self._load_synonyms()

            # 2. 단위 변형 로드 (pos_dict 테이블)
            self._load_units()

            # 3. mat_attr_desc 인덱스 로드 (pos_dict 테이블)
            self._load_mat_attr_index()

            # 4. 약어 로드 (pos_dict 테이블 + 기본 약어)
            self._load_abbreviations()

            elapsed = time.time() - start_time
            self.log.info(
                f"지식 로드 완료: {elapsed:.2f}초 "
                f"(동의어: {len(self.synonym_reverse)}, "
                f"단위: {len(self.unit_reverse)}, "
                f"약어: {len(self.abbreviations)})"
            )

            self._loaded = True
            return True

        except Exception as e:
            self.log.error(f"지식 로드 실패: {e}")
            self.log.debug(traceback.format_exc())
            return False

    def _load_synonyms(self):
        """pos_dict에서 동의어 매핑 로드"""
        try:
            cur = self.conn.cursor()

            # pos_dict 테이블에서 umgv_desc, pos_umgv_desc 조회
            query = """
                SELECT DISTINCT
                    UPPER(TRIM(umgv_desc)) as standard_term,
                    UPPER(TRIM(pos_umgv_desc)) as variant_term
                FROM pos_dict
                WHERE umgv_desc IS NOT NULL
                  AND umgv_desc != ''
                  AND pos_umgv_desc IS NOT NULL
                  AND pos_umgv_desc != ''
                  AND UPPER(TRIM(umgv_desc)) != UPPER(TRIM(pos_umgv_desc))
            """

            cur.execute(query)
            rows = cur.fetchall()
            cur.close()

            count = 0
            for standard, variant in rows:
                # Forward mapping
                if standard not in self.synonym_forward:
                    self.synonym_forward[standard] = []
                if variant not in self.synonym_forward[standard]:
                    self.synonym_forward[standard].append(variant)

                # Reverse mapping
                self.synonym_reverse[variant] = standard
                count += 1

            self.log.debug(f"동의어 로드: {count}개 매핑")

        except Exception as e:
            self.log.warning(f"동의어 로드 실패: {e}")

    def _load_units(self):
        """pos_dict에서 단위 변형 매핑 로드"""
        try:
            # 기본 단위 변형 (하드코딩, 최소한만)
            basic_variants = {
                '°C': ['OC', 'oc', 'degC', 'deg C', 'degree C', 'celsius'],
                'kW': ['KW', 'kw', 'kilowatt', 'kilowatts'],
                'rpm': ['RPM', 'r/min', 'rev/min', 'revolutions per minute'],
                'mm': ['MM', 'millimeter', 'millimeters'],
                'm': ['M', 'meter', 'meters'],
                'kg': ['KG', 'kilogram', 'kilograms'],
                'bar': ['BAR', 'Bar'],
                'L': ['l', 'liter', 'litre', 'liters', 'litres'],
                'V': ['v', 'volt', 'volts'],
                'A': ['a', 'amp', 'amps', 'ampere', 'amperes'],
                'Hz': ['HZ', 'hz', 'hertz'],
                'MPa': ['MPA', 'mpa', 'megapascal'],
                'kPa': ['KPA', 'kpa', 'kilopascal'],
            }

            # 기본 변형 추가
            for standard, variants in basic_variants.items():
                self.unit_forward[standard] = list(variants)
                for variant in variants:
                    self.unit_reverse[variant] = standard
                # 표준 단위 자신도 매핑
                self.unit_reverse[standard] = standard

            # pos_dict에서 추가 변형 로드
            cur = self.conn.cursor()

            query = """
                SELECT DISTINCT
                    TRIM(umgv_uom) as standard_unit,
                    TRIM(pos_umgv_uom) as variant_unit
                FROM pos_dict
                WHERE umgv_uom IS NOT NULL
                  AND umgv_uom != ''
                  AND pos_umgv_uom IS NOT NULL
                  AND pos_umgv_uom != ''
                  AND TRIM(umgv_uom) != TRIM(pos_umgv_uom)
            """

            cur.execute(query)
            rows = cur.fetchall()
            cur.close()

            count = len(self.unit_reverse)
            for umgv_uom, pos_umgv_uom in rows:
                # 표준 단위로 정규화 (기본 변형 사전에서 찾기)
                standard_unit = self.unit_reverse.get(umgv_uom, umgv_uom)

                if standard_unit not in self.unit_forward:
                    self.unit_forward[standard_unit] = []

                # pos_umgv_uom 추가
                if pos_umgv_uom not in self.unit_forward[standard_unit]:
                    self.unit_forward[standard_unit].append(pos_umgv_uom)
                self.unit_reverse[pos_umgv_uom] = standard_unit

            new_count = len(self.unit_reverse) - count
            self.log.debug(f"단위 변형 로드: 기본 {count}개 + DB {new_count}개")

        except Exception as e:
            self.log.warning(f"단위 변형 로드 실패: {e}")

    def _load_mat_attr_index(self):
        """pos_dict에서 mat_attr_desc 인덱스 로드"""
        try:
            cur = self.conn.cursor()

            query = """
                SELECT DISTINCT
                    UPPER(TRIM(umgv_desc)) as umgv_desc,
                    TRIM(mat_attr_desc) as mat_attr_desc
                FROM pos_dict
                WHERE umgv_desc IS NOT NULL
                  AND umgv_desc != ''
                  AND mat_attr_desc IS NOT NULL
                  AND mat_attr_desc != ''
            """

            cur.execute(query)
            rows = cur.fetchall()
            cur.close()

            for umgv_desc, mat_attr_desc in rows:
                # 하나의 umgv_desc가 여러 mat_attr_desc를 가질 수 있으므로
                # 첫 번째 것만 저장 (또는 리스트로 관리)
                if umgv_desc not in self.mat_attr_index:
                    self.mat_attr_index[umgv_desc] = mat_attr_desc

            self.log.debug(f"mat_attr 인덱스 로드: {len(self.mat_attr_index)}개")

        except Exception as e:
            self.log.warning(f"mat_attr 인덱스 로드 실패: {e}")

    def _load_abbreviations(self):
        """약어 로드 (기본 + pos_dict에서 패턴 추출)"""
        # 기본 약어 (하드코딩, 최소한만)
        basic_abbrevs = {
            'M/E': ['Main Engine', 'Marine Engine'],
            'G/E': ['Generator Engine', 'Generating Engine'],
            'A/E': ['Auxiliary Engine'],
            'MCR': ['Maximum Continuous Rating'],
            'NCR': ['Normal Continuous Rating'],
            'RPM': ['revolutions per minute', 'rev/min', 'r/min'],
            'KW': ['kilowatt', 'kW'],
            'HP': ['horsepower', 'bhp'],
        }

        self.abbreviations.update(basic_abbrevs)

        # pos_dict에서 패턴 추출 (향후 구현 가능)
        # 예: pos_umgv_desc에서 "Full Form (ABBREV)" 패턴
        try:
            cur = self.conn.cursor()

            query = """
                SELECT DISTINCT TRIM(pos_umgv_desc) as pos_desc
                FROM pos_dict
                WHERE pos_umgv_desc IS NOT NULL
                  AND pos_umgv_desc LIKE '%(%'
                LIMIT 1000
            """

            cur.execute(query)
            rows = cur.fetchall()
            cur.close()

            for (pos_desc,) in rows:
                # 패턴: "Full Form (ABBREV)"
                match = re.search(r'(.+?)\s*\(([A-Z/]+)\)', pos_desc)
                if match:
                    full_form = match.group(1).strip()
                    abbrev = match.group(2).strip()

                    if abbrev not in self.abbreviations:
                        self.abbreviations[abbrev] = []
                    if full_form not in self.abbreviations[abbrev]:
                        self.abbreviations[abbrev].append(full_form)

            self.log.debug(f"약어 로드: {len(self.abbreviations)}개")

        except Exception as e:
            self.log.warning(f"약어 로드 실패: {e}")

    # === 조회 메서드 (기존 FuzzyMatcher/UnitNormalizer와 호환) ===

    def get_standard_term(self, variant: str) -> str:
        """변형 용어를 표준 용어로 변환"""
        if not variant:
            return variant
        variant_upper = norm(variant).upper()
        return self.synonym_reverse.get(variant_upper, variant)

    def get_synonyms(self, standard_term: str) -> List[str]:
        """표준 용어의 모든 동의어 반환"""
        standard_upper = norm(standard_term).upper()
        return self.synonym_forward.get(standard_upper, [])

    def is_synonym(self, term1: str, term2: str) -> bool:
        """두 용어가 동의어 관계인지 확인"""
        std1 = self.get_standard_term(term1)
        std2 = self.get_standard_term(term2)
        return std1.upper() == std2.upper()

    def normalize_unit(self, unit: str) -> str:
        """단위를 표준 형태로 정규화"""
        if not unit:
            return unit

        # 직접 매칭
        if unit in self.unit_reverse:
            return self.unit_reverse[unit]

        # 대소문자 무시 검색
        unit_lower = unit.lower()
        for variant, standard in self.unit_reverse.items():
            if variant.lower() == unit_lower:
                return standard

        # 매칭 실패 시 원본 반환
        return unit

    def get_unit_variants(self, unit: str) -> List[str]:
        """표준 단위의 모든 변형 반환"""
        standard = self.normalize_unit(unit)
        return self.unit_forward.get(standard, [unit])

    def is_unit_variant_of(self, unit1: str, unit2: str) -> bool:
        """두 단위가 동일한지 확인 (변형 포함)"""
        return self.normalize_unit(unit1) == self.normalize_unit(unit2)

    def get_abbreviation_expansions(self, text: str) -> List[str]:
        """약어를 모든 가능한 확장으로 변환"""
        expansions = [text]

        for abbrev, full_forms in self.abbreviations.items():
            if abbrev in text.upper():
                for full_form in full_forms:
                    expanded = re.sub(abbrev, full_form, text, flags=re.IGNORECASE)
                    if expanded not in expansions:
                        expansions.append(expanded)

        return expansions

    def get_mat_attr_desc(self, umgv_desc: str) -> str:
        """umgv_desc에 대한 mat_attr_desc 반환"""
        umgv_upper = norm(umgv_desc).upper()
        return self.mat_attr_index.get(umgv_upper, "")

    def close(self):
        """연결 종료 (자신이 생성한 연결만)"""
        if self._own_connection and self.conn:
            try:
                self.conn.close()
                self.log.debug("PostgreSQL 지식 로더 연결 종료")
            except:
                pass


class KnowledgeCacheBuilder:
    """
    ⚠️ DEPRECATED: 파일 기반 캐시 빌더 (PostgresKnowledgeLoader 사용 권장)

    보안망 환경에서는 PostgresKnowledgeLoader를 사용하세요.
    """

    @staticmethod
    def build_synonym_cache(glossary_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        용어집에서 동의어 캐시 생성

        Returns:
            {standard_term: [variant1, variant2, ...]}
            예: {"OUTPUT": ["POWER OUTPUT", "RATED OUTPUT", "M/E OUTPUT"]}
        """
        synonym_map = {}
        reverse_map = {}

        if glossary_df is None or glossary_df.empty:
            return synonym_map

        for _, row in glossary_df.iterrows():
            umgv_desc = norm(row.get('umgv_desc', ''))
            pos_umgv_desc = norm(row.get('pos_umgv_desc', ''))

            if not umgv_desc:
                continue

            # 표준명 → 변형들
            if umgv_desc not in synonym_map:
                synonym_map[umgv_desc] = []

            # pos_umgv_desc가 있고 다르면 추가
            if pos_umgv_desc and pos_umgv_desc != umgv_desc:
                if pos_umgv_desc not in synonym_map[umgv_desc]:
                    synonym_map[umgv_desc].append(pos_umgv_desc)

            # 역방향 매핑 (변형 → 표준명)
            if pos_umgv_desc:
                reverse_map[pos_umgv_desc] = umgv_desc

        # 역방향 매핑도 함께 저장 (검색 편의)
        result = {
            'forward': synonym_map,  # 표준 → 변형들
            'reverse': reverse_map   # 변형 → 표준
        }

        return result

    @staticmethod
    def build_unit_cache(glossary_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        용어집에서 단위 변형 캐시 생성

        Returns:
            {standard_unit: [variant1, variant2, ...]}
            예: {"°C": ["OC", "oc", "degC"]}
        """
        unit_map = {}
        reverse_map = {}

        # 기본 단위 변형 (하드코딩, 최소한만)
        basic_variants = {
            '°C': ['OC', 'oc', 'degC', 'deg C', 'degree C', 'celsius'],
            'kW': ['KW', 'kw', 'kilowatt', 'kilowatts'],
            'rpm': ['RPM', 'r/min', 'rev/min', 'revolutions per minute'],
            'mm': ['MM', 'millimeter', 'millimeters'],
            'm': ['M', 'meter', 'meters'],
            'kg': ['KG', 'kilogram', 'kilograms'],
            'bar': ['BAR', 'Bar'],
            'L': ['l', 'liter', 'litre', 'liters', 'litres'],
            'V': ['v', 'volt', 'volts'],
            'A': ['a', 'amp', 'amps', 'ampere', 'amperes'],
            'Hz': ['HZ', 'hz', 'hertz'],
            'MPa': ['MPA', 'mpa', 'megapascal'],
            'kPa': ['KPA', 'kpa', 'kilopascal'],
        }

        for standard, variants in basic_variants.items():
            unit_map[standard] = list(variants)
            for variant in variants:
                reverse_map[variant] = standard
            # 표준 단위 자신도 매핑
            reverse_map[standard] = standard

        # 용어집에서 추가 변형 학습
        if glossary_df is not None and not glossary_df.empty:
            for _, row in glossary_df.iterrows():
                umgv_uom = norm(row.get('umgv_uom', ''))
                pos_umgv_uom = norm(row.get('pos_umgv_uom', ''))

                if not umgv_uom:
                    continue

                # 표준 단위로 정규화 (기본 변형 사전에서 찾기)
                standard_unit = reverse_map.get(umgv_uom, umgv_uom)

                if standard_unit not in unit_map:
                    unit_map[standard_unit] = []

                # pos_umgv_uom이 있고 다르면 추가
                if pos_umgv_uom and pos_umgv_uom != standard_unit:
                    if pos_umgv_uom not in unit_map[standard_unit]:
                        unit_map[standard_unit].append(pos_umgv_uom)
                    reverse_map[pos_umgv_uom] = standard_unit

        result = {
            'forward': unit_map,
            'reverse': reverse_map
        }

        return result

    @staticmethod
    def build_abbreviation_cache(glossary_df: pd.DataFrame = None) -> Dict[str, List[str]]:
        """
        약어 캐시 생성

        Returns:
            {abbreviation: [full_form1, full_form2, ...]}
            예: {"M/E": ["Main Engine", "Marine Engine"]}
        """
        abbrev_map = {}

        # 기본 약어 (하드코딩, 최소한만)
        basic_abbrevs = {
            'M/E': ['Main Engine', 'Marine Engine'],
            'G/E': ['Generator Engine', 'Generating Engine'],
            'A/E': ['Auxiliary Engine'],
            'MCR': ['Maximum Continuous Rating'],
            'NCR': ['Normal Continuous Rating'],
            'RPM': ['revolutions per minute', 'rev/min', 'r/min'],
            'KW': ['kilowatt', 'kW'],
            'HP': ['horsepower', 'bhp'],
        }

        abbrev_map.update(basic_abbrevs)

        # 용어집에서 학습 (향후 구현 가능)
        # pos_umgv_desc에서 "Full Form (ABBREV)" 패턴 추출
        if glossary_df is not None and not glossary_df.empty:
            for _, row in glossary_df.iterrows():
                pos_desc = norm(row.get('pos_umgv_desc', ''))
                if not pos_desc:
                    continue

                # 패턴: "Full Form (ABBREV)"
                match = re.search(r'(.+?)\s*\(([A-Z/]+)\)', pos_desc)
                if match:
                    full_form = match.group(1).strip()
                    abbrev = match.group(2).strip()

                    if abbrev not in abbrev_map:
                        abbrev_map[abbrev] = []
                    if full_form not in abbrev_map[abbrev]:
                        abbrev_map[abbrev].append(full_form)

        return abbrev_map

    @staticmethod
    def save_cache(cache_data: dict, cache_name: str):
        """캐시를 JSON 파일로 저장"""
        cache_path = KnowledgeCacheBuilder.get_cache_path(cache_name)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save cache {cache_name}: {e}")

    @staticmethod
    def load_cache(cache_name: str) -> dict:
        """캐시 파일 로드"""
        cache_path = KnowledgeCacheBuilder.get_cache_path(cache_name)
        if not os.path.exists(cache_path):
            return {}

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load cache {cache_name}: {e}")
            return {}

    @staticmethod
    def is_cache_valid(cache_name: str, max_age_hours: int = 24) -> bool:
        """캐시가 유효한지 확인 (생성된 지 24시간 이내)"""
        cache_path = KnowledgeCacheBuilder.get_cache_path(cache_name)
        if not os.path.exists(cache_path):
            return False

        # 파일 수정 시간 확인
        mtime = os.path.getmtime(cache_path)
        age_hours = (time.time() - mtime) / 3600

        return age_hours < max_age_hours

    @staticmethod
    def rebuild_all_caches(glossary_path: str = None, glossary_df: pd.DataFrame = None):
        """모든 캐시 재생성"""
        if glossary_df is None and glossary_path:
            try:
                glossary_df = pd.read_excel(glossary_path)
            except Exception as e:
                print(f"[ERROR] Failed to load glossary: {e}")
                return

        print("[INFO] Building synonym cache...")
        synonym_cache = KnowledgeCacheBuilder.build_synonym_cache(glossary_df)
        KnowledgeCacheBuilder.save_cache(synonym_cache, 'synonyms_cache.json')

        print("[INFO] Building unit cache...")
        unit_cache = KnowledgeCacheBuilder.build_unit_cache(glossary_df)
        KnowledgeCacheBuilder.save_cache(unit_cache, 'units_cache.json')

        print("[INFO] Building abbreviation cache...")
        abbrev_cache = KnowledgeCacheBuilder.build_abbreviation_cache(glossary_df)
        KnowledgeCacheBuilder.save_cache(abbrev_cache, 'abbreviations_cache.json')

        print("[INFO] All caches built successfully")


class UnitNormalizer:
    """
    단위 정규화 (PostgreSQL 기반)

    PostgresKnowledgeLoader를 래핑하여 기존 인터페이스 유지
    """

    def __init__(self, pg_loader: PostgresKnowledgeLoader = None):
        """
        Args:
            pg_loader: PostgresKnowledgeLoader 인스턴스 (공유)
        """
        self.pg_loader = pg_loader

    def normalize(self, unit: str) -> str:
        """단위를 표준 형태로 정규화"""
        if not self.pg_loader:
            return unit
        return self.pg_loader.normalize_unit(unit)

    def get_variants(self, unit: str) -> List[str]:
        """표준 단위의 모든 변형 반환"""
        if not self.pg_loader:
            return [unit]
        return self.pg_loader.get_unit_variants(unit)

    def is_variant_of(self, unit1: str, unit2: str) -> bool:
        """두 단위가 동일한지 확인 (변형 포함)"""
        if not self.pg_loader:
            return unit1 == unit2
        return self.pg_loader.is_unit_variant_of(unit1, unit2)


class FuzzyMatcher:
    """
    Fuzzy string matching (PostgreSQL 기반)

    PostgresKnowledgeLoader를 래핑하여 기존 인터페이스 유지
    """

    def __init__(self, pg_loader: PostgresKnowledgeLoader = None):
        """
        Args:
            pg_loader: PostgresKnowledgeLoader 인스턴스 (공유)
        """
        self.pg_loader = pg_loader

    def get_standard_term(self, variant: str) -> str:
        """변형 용어를 표준 용어로 변환"""
        if not self.pg_loader:
            return variant
        return self.pg_loader.get_standard_term(variant)

    def get_synonyms(self, standard_term: str) -> List[str]:
        """표준 용어의 모든 동의어 반환"""
        if not self.pg_loader:
            return []
        return self.pg_loader.get_synonyms(standard_term)

    def is_synonym(self, term1: str, term2: str) -> bool:
        """두 용어가 동의어 관계인지 확인"""
        if not self.pg_loader:
            return False
        return self.pg_loader.is_synonym(term1, term2)

    def ratio(self, s1: str, s2: str) -> float:
        """
        두 문자열의 유사도 계산 (0.0 ~ 1.0)

        먼저 동의어 관계 확인, 그 다음 Levenshtein distance 계산
        """
        if not s1 or not s2:
            return 0.0

        # 동의어 관계면 높은 점수
        if self.is_synonym(s1, s2):
            return 0.95

        # Levenshtein distance 기반 유사도
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def find_best_match(self, query: str, candidates: List[str], threshold: float = 0.7) -> tuple:
        """
        후보 중 가장 유사한 것 찾기

        Returns:
            (best_match, similarity_score) or (None, 0.0)
        """
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = self.ratio(query, candidate)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate

        return best_match, best_score


class AbbreviationExpander:
    """
    약어 확장 (PostgreSQL 기반)

    PostgresKnowledgeLoader를 래핑하여 기존 인터페이스 유지
    """

    def __init__(self, pg_loader: PostgresKnowledgeLoader = None):
        """
        Args:
            pg_loader: PostgresKnowledgeLoader 인스턴스 (공유)
        """
        self.pg_loader = pg_loader

    def expand(self, text: str) -> List[str]:
        """약어를 모든 가능한 확장으로 변환"""
        if not self.pg_loader:
            return [text]
        return self.pg_loader.get_abbreviation_expansions(text)

    def get_full_forms(self, abbrev: str) -> List[str]:
        """약어의 모든 전체 형태 반환"""
        if not self.pg_loader:
            return []
        return self.pg_loader.abbreviations.get(abbrev.upper(), [])


# =============================================================================
# Reference Hint Engine (용어집/사양값DB 참조 힌트)
# =============================================================================

@dataclass
class ExtractionHint:
    """추출 힌트 데이터"""
    spec_name: str = ""

    # 용어집 힌트
    section_num: str = ""          # 섹션 번호 (예: "2.2.1 Main particulars")
    table_text: str = ""           # 테이블/텍스트 구분
    value_format: str = ""         # 값 형식 (숫자형/문자형/혼합형)
    pos_umgv_desc: str = ""        # POS에서의 사양명 (동의어)
    mat_attr_desc: str = ""        # mat_attr_desc (장비명)
    umgv_uom: str = ""             # umgv_uom (단위)
    pos_umgv_uom: str = ""         # pos_umgv_uom (POS에서의 단위)

    # 사양값DB 힌트
    historical_values: List[str] = field(default_factory=list)  # 과거 값들
    value_patterns: List[str] = field(default_factory=list)     # 값 패턴

    # 유사 POS 힌트
    similar_pos_hints: List[Dict] = field(default_factory=list)  # 유사 POS 정보

    # 단위 관련 힌트 (데이터 기반)
    related_units: List[str] = field(default_factory=list)  # 사양값DB/용어집에서 수집한 관련 단위들

    # 메타데이터 (신뢰도 평가용)
    metadata: Dict = field(default_factory=dict)  # 힌트 출처 및 신뢰도 정보


class ReferenceHintEngine:
    """
    참조 힌트 엔진
    
    용어집과 사양값DB를 참조하여 추출에 필요한 힌트를 제공합니다.
    
    효율성 전략:
    1. Hull 단위 배치 로딩 - 동일 hull의 모든 사양 힌트를 한 번에 로드
    2. LRU 캐시 - 최근 사용된 hull/사양 정보 캐싱
    3. Lazy Loading - 실제 필요 시에만 로드
    
    사용 예시:
        hint_engine = ReferenceHintEngine(glossary, specdb, pg_loader)
        
        # 파일 처리 전 배치 로드 (선택적, 최적화용)
        hint_engine.preload_for_hull("2377")
        
        # 개별 사양별 힌트 조회
        hint = hint_engine.get_hints("2377", "CAPACITY")
    """
    
    # 캐시 크기 설정
    MAX_HULL_CACHE_SIZE = 50
    MAX_SPEC_CACHE_SIZE = 200
    
    def __init__(
        self,
        glossary: LightweightGlossaryIndex = None,
        specdb: LightweightSpecDBIndex = None,
        pg_loader = None,  # PostgresEmbeddingLoader
        logger: logging.Logger = None
    ):
        self.glossary = glossary
        self.specdb = specdb
        self.pg_loader = pg_loader
        self.log = logger or logging.getLogger(__name__)
        
        # 캐시
        self._hull_cache: Dict[str, Dict[str, ExtractionHint]] = {}
        self._spec_cache: Dict[str, ExtractionHint] = {}
        self._hull_cache_order: List[str] = []  # LRU 순서
        
        # 용어집 인덱스 구축 (사양명 → 힌트 정보)
        self._glossary_index: Dict[str, List[Dict]] = {}
        self._build_glossary_index()
        
        self.log.debug("ReferenceHintEngine 초기화 완료")
    
    def _build_glossary_index(self):
        """
        용어집 인덱스 구축 (사양명별 힌트 정보)
        
        umgv_desc를 키로 section_num, table_text, value_format 등을 인덱싱
        """
        if not self.glossary or self.glossary.df is None:
            return
        
        for _, row in self.glossary.df.iterrows():
            spec_name = str(row.get('umgv_desc', '')).strip().upper()
            if not spec_name:
                continue
            
            if spec_name not in self._glossary_index:
                self._glossary_index[spec_name] = []
            
            self._glossary_index[spec_name].append({
                'section_num': str(row.get('section_num', '')),
                'table_text': str(row.get('table_text', '')),
                'value_format': str(row.get('value_format', '')),
                'pos_umgv_desc': str(row.get('pos_umgv_desc', '')),
                'hull': extract_hull_from_matnr(str(row.get('matnr', ''))),
                'extwg': str(row.get('extwg', '')),
            })
    
    def preload_for_hull(self, hull: str) -> Dict[str, ExtractionHint]:
        """
        특정 hull의 모든 사양 힌트를 배치 로드 (효율성 최적화)
        
        파일 처리 시작 시 호출하여 해당 hull의 모든 힌트를 캐시에 로드합니다.
        이후 개별 사양별 조회는 캐시에서 O(1)로 수행됩니다.
        
        Args:
            hull: 호선 번호
            
        Returns:
            사양명 → ExtractionHint 매핑
        """
        if hull in self._hull_cache:
            # LRU 업데이트
            if hull in self._hull_cache_order:
                self._hull_cache_order.remove(hull)
            self._hull_cache_order.append(hull)
            return self._hull_cache[hull]
        
        hull_hints: Dict[str, ExtractionHint] = {}
        
        # 1. 사양값DB에서 해당 hull의 모든 과거 값 로드
        historical_by_spec: Dict[str, List[str]] = {}
        if self.specdb and self.specdb.df is not None:
            hull_df = self.specdb.df[
                self.specdb.df['matnr'].astype(str).str.startswith(hull)
            ]
            for _, row in hull_df.iterrows():
                spec_name = str(row.get('umgv_desc', '')).strip().upper()
                value = str(row.get('umgv_value_edit', '')).strip()
                if spec_name and value:
                    if spec_name not in historical_by_spec:
                        historical_by_spec[spec_name] = []
                    if value not in historical_by_spec[spec_name]:
                        historical_by_spec[spec_name].append(value)
        
        # 2. 용어집에서 해당 사양들의 힌트 정보 매핑
        all_specs = set(historical_by_spec.keys())
        
        # 용어집의 모든 사양도 포함
        all_specs.update(self._glossary_index.keys())
        
        for spec_name in all_specs:
            hint = self._build_hint(spec_name, hull, historical_by_spec.get(spec_name, []))
            hull_hints[spec_name] = hint
        
        # 캐시에 저장 (LRU)
        if len(self._hull_cache) >= self.MAX_HULL_CACHE_SIZE:
            oldest_hull = self._hull_cache_order.pop(0)
            del self._hull_cache[oldest_hull]
        
        self._hull_cache[hull] = hull_hints
        self._hull_cache_order.append(hull)
        
        return hull_hints
    
    def _build_hint(
        self,
        spec_name: str,
        hull: str,
        historical_values: List[str] = None
    ) -> ExtractionHint:
        """
        개별 사양의 힌트 구성 (임베딩 기반 유사 사양 검색 포함)

        개선사항:
        1. pos_embedding 활용하여 유사 사양 검색
        2. 상세 로그 기록 (참조 row, 유사도)
        3. 힌트 신뢰도 메타데이터 추가
        """
        hint = ExtractionHint(spec_name=spec_name)
        hint_metadata = {
            'glossary_source': None,
            'embedding_sources': [],
            'historical_count': 0
        }
        related_units_set = set()  # 중복 제거를 위한 set

        # 용어집 힌트
        glossary_entries = self._glossary_index.get(spec_name.upper(), [])
        if glossary_entries:
            # hull이 일치하는 항목 우선, 없으면 첫 번째 사용
            matched = None
            for entry in glossary_entries:
                if entry.get('hull') == hull:
                    matched = entry
                    break

            if not matched:
                matched = glossary_entries[0]

            hint.section_num = matched.get('section_num', '')
            hint.table_text = matched.get('table_text', '')
            hint.value_format = matched.get('value_format', '')
            hint.pos_umgv_desc = matched.get('pos_umgv_desc', '')

            # 용어집에서 단위 정보 수집 (umgv_uom, pos_umgv_uom)
            hint.umgv_uom = matched.get('umgv_uom', '')
            hint.pos_umgv_uom = matched.get('pos_umgv_uom', '')

            if hint.umgv_uom:
                related_units_set.add(hint.umgv_uom)
            if hint.pos_umgv_uom:
                related_units_set.add(hint.pos_umgv_uom)

            hint_metadata['glossary_source'] = {
                'hull': matched.get('hull', ''),
                'extwg': matched.get('extwg', ''),
                'section': hint.section_num
            }

            self.log.debug(
                f"[HINT] Glossary: spec={spec_name}, hull={matched.get('hull')}, "
                f"section={hint.section_num}, pos_desc={hint.pos_umgv_desc}, "
                f"units={hint.umgv_uom}/{hint.pos_umgv_uom}"
            )

        # 과거 값 (사양값DB)
        if historical_values:
            hint.historical_values = historical_values[:10]  # 최대 10개
            hint.value_patterns = self._extract_value_patterns(historical_values)
            hint_metadata['historical_count'] = len(historical_values)

            self.log.debug(
                f"[HINT] Historical: spec={spec_name}, hull={hull}, "
                f"values_count={len(historical_values)}, "
                f"samples={historical_values[:3]}"
            )

        # 임베딩 기반 유사 사양 검색 (새로 추가)
        if self.pg_loader:
            try:
                # search_key 생성: hull_pmg_code_umg_code_extwg
                # embedding_key 생성: hull_pmg_desc_umg_desc_mat_attr_desc
                query_text = f"{hull} {spec_name}"

                # BGE-M3 모델 가져오기 (있으면 사용, 없으면 None)
                embedding_model = getattr(self.pg_loader, 'embedding_model', None)

                similar_specs = self.pg_loader.search_by_key_with_fallback(
                    search_key=f"{hull}_{spec_name}",  # 간단한 키
                    query_text=query_text,
                    embedding_model=embedding_model,
                    top_k=3,
                    similarity_threshold=0.7
                )

                if similar_specs:
                    for idx, similar in enumerate(similar_specs):
                        embedding_key = similar.get('embedding_key', '')
                        similarity = similar.get('similarity', 0.0)

                        hint_metadata['embedding_sources'].append({
                            'embedding_key': embedding_key,
                            'similarity': similarity,
                            'rank': idx + 1
                        })

                        # 유사 사양의 단위 정보 수집
                        similar_uom = similar.get('umgv_uom', '')
                        if similar_uom:
                            related_units_set.add(similar_uom)

                        self.log.debug(
                            f"[HINT] Embedding: spec={spec_name}, "
                            f"similar_key={embedding_key[:50]}..., "
                            f"similarity={similarity:.3f}, uom={similar_uom}"
                        )

                        # 유사 사양의 값도 힌트에 추가 (최대 3개)
                        similar_value = similar.get('umgv_value_edit', '')
                        if similar_value and similar_value not in hint.historical_values:
                            hint.historical_values.append(similar_value)

            except Exception as e:
                self.log.debug(f"[HINT] Embedding search failed: {e}")

        # related_units를 list로 변환하여 저장
        hint.related_units = sorted(list(related_units_set))

        # 힌트 메타데이터 저장 (신뢰도 평가용)
        hint.metadata = hint_metadata

        return hint
    
    def _extract_value_patterns(self, values: List[str]) -> List[str]:
        """값들에서 패턴 추출"""
        patterns = []
        
        for value in values[:5]:
            # 숫자+단위 패턴
            if re.search(r'\d+.*?(kW|rpm|bar|mm|m3|℃|°C|%)', value, re.I):
                patterns.append("숫자+단위")
            # 텍스트 패턴
            elif re.match(r'^[A-Za-z]', value) and not re.search(r'\d', value):
                patterns.append("텍스트")
            # 혼합 패턴
            elif re.search(r'\d', value) and re.search(r'[A-Za-z]', value):
                patterns.append("혼합")
            # 순수 숫자
            elif re.match(r'^[\d.,\-~]+$', value):
                patterns.append("숫자")
        
        return list(set(patterns))
    
    def get_hints(self, hull: str, spec_name: str) -> ExtractionHint:
        """
        특정 사양의 힌트 조회
        
        캐시된 경우 O(1), 아니면 동적 생성
        
        Args:
            hull: 호선 번호
            spec_name: 사양명
            
        Returns:
            ExtractionHint 객체
        """
        spec_upper = spec_name.upper()
        
        # 1. hull 캐시 확인
        if hull in self._hull_cache:
            if spec_upper in self._hull_cache[hull]:
                return self._hull_cache[hull][spec_upper]
        
        # 2. spec 캐시 확인
        cache_key = f"{hull}_{spec_upper}"
        if cache_key in self._spec_cache:
            return self._spec_cache[cache_key]
        
        # 3. 동적 생성
        historical_values = []
        if self.specdb:
            historical_values = self.specdb.get_historical_values(hull, spec_name)
        
        hint = self._build_hint(spec_upper, hull, historical_values)
        
        # spec 캐시에 저장
        if len(self._spec_cache) >= self.MAX_SPEC_CACHE_SIZE:
            # 가장 오래된 항목 제거 (간단한 FIFO)
            oldest_key = next(iter(self._spec_cache))
            del self._spec_cache[oldest_key]
        
        self._spec_cache[cache_key] = hint
        
        return hint
    
    def evaluate_hint_confidence(
        self,
        hint: ExtractionHint,
        chunk_text: str,
        extracted_value: str = ""
    ) -> Dict[str, float]:
        """
        힌트 신뢰도 평가 (하이브리드 접근)

        Rule 기반 평가:
        1. 빈도 기반: 힌트의 historical_value가 chunk에 있는가?
        2. 거리 기반: 추출된 값과 힌트 값의 차이가 합리적인가?
        3. 구조 기반: 원본 사양명이 chunk에 있으면 동의어 불필요

        Returns:
            {
                'overall': 0.0~1.0 (전체 신뢰도),
                'frequency_score': 0.0~1.0,
                'distance_score': 0.0~1.0,
                'structure_score': 0.0~1.0,
                'should_use': True/False
            }
        """
        scores = {
            'overall': 0.0,
            'frequency_score': 0.0,
            'distance_score': 0.0,
            'structure_score': 0.0,
            'should_use': False
        }

        if not hint or not chunk_text:
            return scores

        chunk_upper = chunk_text.upper()
        spec_name_upper = hint.spec_name.upper()

        # 1. 빈도 기반 평가: 힌트 값이 chunk에 존재하는가?
        if hint.historical_values:
            matched_count = 0
            for hist_val in hint.historical_values[:5]:  # 최대 5개만 체크
                hist_upper = hist_val.upper()
                # 완전 일치 또는 부분 일치
                if hist_upper in chunk_upper or chunk_upper.find(hist_upper) >= 0:
                    matched_count += 1

            scores['frequency_score'] = matched_count / min(5, len(hint.historical_values))

        # 2. 거리 기반 평가: 추출된 값과 힌트 값의 차이
        if extracted_value and hint.historical_values:
            try:
                # 숫자 값인 경우 수치 비교
                extracted_num = float(re.sub(r'[^\d.-]', '', extracted_value))
                hint_nums = []

                for hist_val in hint.historical_values[:3]:
                    num_match = re.search(r'([-+]?\d+(?:\.\d+)?)', hist_val)
                    if num_match:
                        hint_nums.append(float(num_match.group(1)))

                if hint_nums:
                    # 가장 가까운 힌트 값과의 차이
                    min_diff = min(abs(extracted_num - h) for h in hint_nums)
                    avg_hint = sum(hint_nums) / len(hint_nums)

                    # 차이 비율 계산 (10% 이내면 높은 점수)
                    if avg_hint != 0:
                        diff_ratio = min_diff / abs(avg_hint)
                        scores['distance_score'] = max(0, 1.0 - diff_ratio)
                    else:
                        scores['distance_score'] = 1.0 if min_diff == 0 else 0.5

            except (ValueError, ZeroDivisionError):
                # 텍스트 값이거나 파싱 실패 시 문자열 유사도
                if extracted_value.upper() in [h.upper() for h in hint.historical_values]:
                    scores['distance_score'] = 1.0
                else:
                    scores['distance_score'] = 0.3  # 중립

        # 3. 구조 기반 평가: 원본 사양명 vs 동의어
        if spec_name_upper in chunk_upper:
            # 원본 사양명이 이미 chunk에 있으면 높은 점수
            scores['structure_score'] = 1.0
        elif hint.pos_umgv_desc and hint.pos_umgv_desc.upper() in chunk_upper:
            # 동의어가 chunk에 있으면 중간 점수
            scores['structure_score'] = 0.7
        else:
            # 둘 다 없으면 낮은 점수
            scores['structure_score'] = 0.3

        # 전체 신뢰도 계산 (가중 평균)
        weights = {
            'frequency_score': 0.4,
            'distance_score': 0.3,
            'structure_score': 0.3
        }

        scores['overall'] = sum(
            scores[key] * weights[key]
            for key in weights.keys()
        )

        # 사용 여부 결정 (임계값: 0.5)
        scores['should_use'] = scores['overall'] >= 0.5

        # 로그 기록
        self.log.debug(
            f"[HINT_EVAL] spec={hint.spec_name}, "
            f"overall={scores['overall']:.2f}, "
            f"freq={scores['frequency_score']:.2f}, "
            f"dist={scores['distance_score']:.2f}, "
            f"struct={scores['structure_score']:.2f}, "
            f"should_use={scores['should_use']}"
        )

        return scores

    def filter_hints_by_confidence(
        self,
        hint: ExtractionHint,
        chunk_text: str,
        min_confidence: float = 0.5
    ) -> ExtractionHint:
        """
        신뢰도 기반 힌트 필터링

        신뢰도가 낮은 힌트는 제거하거나 약화시킵니다.

        Args:
            hint: 원본 힌트
            chunk_text: 추출 대상 chunk
            min_confidence: 최소 신뢰도 (기본 0.5)

        Returns:
            필터링된 힌트 (신뢰도 낮으면 일부 정보 제거)
        """
        if not hint:
            return hint

        # 신뢰도 평가 (extracted_value 없이 사전 평가)
        confidence = self.evaluate_hint_confidence(hint, chunk_text, "")

        if confidence['should_use']:
            # 신뢰도 높으면 그대로 사용
            self.log.debug(f"[HINT_FILTER] Using hint: confidence={confidence['overall']:.2f}")
            return hint
        else:
            # 신뢰도 낮으면 일부 정보만 사용
            filtered_hint = ExtractionHint(spec_name=hint.spec_name)

            # section_num과 value_format은 유지 (안전한 정보)
            filtered_hint.section_num = hint.section_num
            filtered_hint.value_format = hint.value_format

            # historical_values는 제거 (오히려 혼란 가능)
            # pos_umgv_desc도 제거 (동의어가 맞지 않을 수 있음)

            self.log.debug(
                f"[HINT_FILTER] Filtered hint: confidence={confidence['overall']:.2f}, "
                f"removed historical_values and pos_umgv_desc"
            )

            return filtered_hint

    def get_section_search_hints(self, spec_name: str) -> List[str]:
        """
        섹션 검색 힌트 반환

        section_num에서 검색에 유용한 키워드 추출
        예: "2.2.1 Main particulars" → ["2.2.1", "Main particulars"]
        """
        hints = []

        glossary_entries = self._glossary_index.get(spec_name.upper(), [])
        for entry in glossary_entries:
            section = entry.get('section_num', '')
            if section:
                # 섹션 번호 추출 (예: "2.2.1")
                sec_match = re.match(r'^(\d+(?:\.\d+)*)', section)
                if sec_match:
                    hints.append(sec_match.group(1))

                # 섹션 제목 추출 (예: "Main particulars")
                title_match = re.search(r'\d+(?:\.\d+)*\s+(.+)', section)
                if title_match:
                    hints.append(title_match.group(1).strip())

        return list(set(hints))
    
    def get_similar_pos_hints(
        self, 
        hull: str, 
        spec_name: str,
        embedding_key: str = ""
    ) -> List[Dict]:
        """
        유사 POS의 힌트 정보 조회 (embedding 기반)
        
        주의: DB 쿼리 필요, 필요할 때만 호출
        
        Args:
            hull: 현재 호선 번호
            spec_name: 사양명
            embedding_key: 임베딩 검색용 키
            
        Returns:
            유사 POS의 값/힌트 정보 리스트
        """
        if not self.pg_loader:
            return []
        
        # TODO: embedding_key 기반 유사 POS 검색 구현
        # 현재는 사양값DB의 모든 hull에서 값 수집
        hints = []
        
        if self.specdb and self.specdb.df is not None:
            spec_upper = spec_name.upper()
            spec_df = self.specdb.df[
                self.specdb.df['umgv_desc'].str.upper() == spec_upper
            ]
            
            # 다른 hull의 값들 수집 (현재 hull 제외)
            for _, row in spec_df.head(20).iterrows():
                row_hull = extract_hull_from_matnr(str(row.get('matnr', '')))
                if row_hull and row_hull != hull:
                    value = str(row.get('umgv_value_edit', '')).strip()
                    if value:
                        hints.append({
                            'hull': row_hull,
                            'value': value,
                            'extwg': str(row.get('extwg', '')),
                        })
        
        return hints[:5]  # 최대 5개
    
    def clear_cache(self):
        """캐시 초기화"""
        self._hull_cache.clear()
        self._spec_cache.clear()
        self._hull_cache_order.clear()
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계"""
        return {
            'hull_cache_size': len(self._hull_cache),
            'spec_cache_size': len(self._spec_cache),
            'glossary_index_size': len(self._glossary_index),
        }

class HTMLChunkParser:
    """
    HTML 문서 파싱 및 청킹
    
    개선사항:
    - 테이블 키-값 쌍 추출 강화
    - 사양명 동의어 확장
    - 정규화된 값/단위 분리
    """
    
    # 사양명 동의어 매핑
    # 일반화된 동의어로 다양한 POS 문서 커버
    SPEC_SYNONYMS = {
        # 출력/용량 관련
        'OUTPUT': ['OUTPUT', 'POWER OUTPUT', 'RATED OUTPUT', 'MOTOR OUTPUT', 
                   'RATING OF ELECTRIC MOTOR', 'RATED POWER', 'MOTOR POWER',
                   'ELECTRIC MOTOR OUTPUT', 'ELECTRIC MOTOR'],
        'CAPACITY': ['CAPACITY', 'FLOW RATE', 'RATED CAPACITY', 'NOMINAL CAPACITY',
                    'HANDLING CAPACITY', 'CARGO LOADING', 'CARGO UNLOADING',
                    'CAPACITY & HEAD'],
        'QUANTITY': ['QUANTITY', 'QTY', 'QTY.', 'NO. OF UNIT', 'NO. OF SET',
                    'NO OF UNIT', 'NO OF SET', 'Q\'TY', 'NUMBER OF', 'NO. OF UNITS'],
        
        # 타입/모델 관련
        'TYPE': ['TYPE', 'MODEL TYPE', 'EQUIPMENT TYPE', 'SEAL TYPE', 'DEVICE TYPE'],
        'TYPE & MAKER': ['TYPE & MAKER', 'TYPE AND MAKER', 'MAKER & TYPE', 'MAKER AND TYPE'],
        'MODEL': ['MODEL', 'MODEL NO.', 'MODEL NO', 'MODEL NUMBER'],
        
        # 전기/동력 관련
        'POWER SOURCE': ['POWER SOURCE', 'POWER SUPPLY', 'ELECTRIC SOURCE'],
        'VOLTAGE': ['VOLTAGE', 'RATED VOLTAGE', 'WORKING VOLTAGE'],
        'MOTOR SHAFT SPEED': ['MOTOR SHAFT SPEED', 'SHAFT SPEED', 'MOTOR SPEED'],
        'SPEED': ['SPEED', 'RPM', 'REVOLUTION', 'ROTATION SPEED'],
        'STARTING METHOD': ['STARTING METHOD', 'START METHOD', 'STARTING TYPE'],
        
        # 치수/크기 관련
        'DIMENSION': ['DIMENSION', 'DIMENSIONS', 'ESTIMATE DIMENSION', 'SIZE', 'OPENING SIZE'],
        'THICKNESS': ['THICKNESS', 'INSULATION THICKNESS', 'INSUL. THICKNESS', 'PLATE THICKNESS'],
        'DIAMETER': ['DIAMETER', 'SHAFT DIAMETER', 'PIPE DIAMETER', 'NOMINAL DIAMETER'],
        
        # 압력/온도 관련
        'PRESSURE': ['PRESSURE', 'WORKING PRESSURE', 'DESIGN PRESSURE', 
                    'MAX. PRESSURE', 'OPERATING PRESSURE', 'BACK PRESSURE', 'BACK-PRESSURE',
                    'MAXIMUM WORKING PRESSURE', 'STUFFING BOX PRESSURE', 'DISCHARGE PRESSURE',
                    'SUCTION PRESSURE', 'BAROMETRIC PRESSURE', 'TOTAL BAROMETRIC PRESSURE'],
        'TEMPERATURE': ['TEMPERATURE', 'WORKING TEMPERATURE', 'DESIGN TEMPERATURE',
                       'MAX. TEMPERATURE', 'INLET TEMPERATURE', 'OUTLET TEMPERATURE',
                       'MAXIMUM WORKING TEMPERATURE', 'AIR TEMPERATURE', 'ER AIR TEMPERATURE',
                       'SEA WATER TEMPERATURE', 'AMBIENT TEMPERATURE', 'AMBIENT AIR TEMPERATURE'],
        
        # 재질/물성 관련
        'MATERIAL': ['MATERIAL', 'MATERIALS', 'INSULATION MATERIAL', 'PIPE MATERIAL'],
        'DENSITY': ['DENSITY', 'FOAM DENSITY', 'BULK DENSITY', 'SPECIFIC GRAVITY'],
        'THERMAL CONDUCTANCE': ['THERMAL CONDUCTANCE', 'THERMAL CONDUCTIVITY'],
        
        # 펌프/유체 관련
        'HEAD': ['HEAD', 'TOTAL HEAD', 'DISCHARGE HEAD', 'PUMP HEAD', 'WATER HEAD', 'TH'],
        'FLOW': ['FLOW', 'FLOW RATE', 'VOLUMETRIC FLOW'],
        'DRIVEN BY': ['DRIVEN BY', 'DRIVE METHOD', 'DRIVEN TYPE'],
        'SEAL': ['SEAL', 'SEAL RING', 'MECHANICAL SEAL', 'SEAL TYPE'],
        'API PIPING PLAN': ['API PIPING PLAN', 'API PIPINGPLAN', 'PIPING PLAN'],
        
        # 엔진 관련
        'MCR': ['MCR', 'NOMINAL MCR', 'SPECIFIED MCR', 'MAXIMUM CONTINUOUS RATING'],
        'NCR': ['NCR', 'NORMAL CONTINUOUS RATING'],
        'SFOC': ['SFOC', 'SPECIFIC FUEL OIL CONSUMPTION', 'FUEL CONSUMPTION'],
        'BACK PRESSURE': ['BACK PRESSURE', 'BACK-PRESSURE', 'BACKPRESSURE',
                         'ESTIMATED BACK-PRESSURE', 'ESTIMATED BACK PRESSURE'],
        'STEAM PRESSURE': ['STEAM PRESSURE', 'STEAM PRESS'],
        'TURBOCHARGER TYPE': ['TURBOCHARGER TYPE', 'TURBOCHARGER TYPE & NUMBER', 
                              'TURBOCHARGER TYPE AND NUMBER', 'T/C TYPE'],
        
        # 선박 관련
        'CRANE CAPACITY': ['CRANE CAPACITY', 'MAX. CRANE CAPACITY', 'MAX CRANE CAPACITY',
                          'SHIPYARD\'S MAX. CRANE CAPACITY', 'SHIPYARD MAX CRANE CAPACITY'],
        'GYRO REPEATER': ['GYRO REPEATER', 'DIGITAL TYPE GYRO REPEATER', 'GYRO COMPASS REPEATER'],
        'NOMINAL THRUST': ['NOMINAL THRUST', 'THRUST', 'RATED THRUST'],
        'BLADE': ['BLADE', 'BLADES', 'PROPELLER BLADE'],
        
        # 냉동/냉장 시스템 관련 (일반화)
        'VOLUME': ['VOLUME', 'CHAMBER VOLUME', 'ROOM VOLUME', 'STORAGE VOLUME'],
        'MEAT ROOM': ['MEAT ROOM', 'MEAT CHAMBER', 'MEAT STORAGE'],
        'FISH ROOM': ['FISH ROOM', 'FISH CHAMBER', 'FISH STORAGE'],
        'VEGETABLE ROOM': ['VEGETABLE ROOM', 'VEGETABLE CHAMBER', 'VEGETABLE STORAGE'],
        'DAIRY ROOM': ['DAIRY ROOM', 'DAIRY CHAMBER', 'DAIRY STORAGE'],
        'LOBBY': ['LOBBY', 'ANTE ROOM', 'ANTEROOM'],
        'COMPRESSOR': ['COMPRESSOR', 'COMPRESSORS', 'REFRIGERANT COMPRESSOR'],
        'UNIT COOLER': ['UNIT COOLER', 'UNIT COOLERS', 'EVAPORATOR', 'COOLING UNIT'],
        'CONDENSING UNIT': ['CONDENSING UNIT', 'CONDENSER UNIT', 'CONDENSERUNIT'],
        'SUB-COOLER': ['SUB-COOLER', 'SUBCOOLER', 'SUB COOLER'],
        'REFRIGERANT': ['REFRIGERANT', 'REFRIGERANT TYPE', 'COOLANT'],
        'COOLING SYSTEM': ['COOLING SYSTEM', 'COOLINGSYSTEM', 'REFRIGERATION SYSTEM'],
        'DEFROSTING SYSTEM': ['DEFROSTING SYSTEM', 'DEFROSTINGSYSTEM', 'DEFROST SYSTEM'],
        'MAIN COMPONENT': ['MAIN COMPONENT', 'MAINCOMPONENT', 'MAIN COMPONENTS'],
        'ELECTRIC EQUIPMENT': ['ELECTRIC EQUIPMENT', 'ELECTRICEQUIPMENT', 'ELECTRICAL EQUIPMENT'],
        'TEMPERATURE SENSOR': ['TEMPERATURE SENSOR', 'TEMPERATURE SENSORS', 'TEMP SENSOR'],
        'PIPE MATERIAL': ['PIPE MATERIAL', 'PIPEMATERIAL', 'PIPING MATERIAL'],
        'COOLING WATER LINES': ['COOLING WATER LINES', 'COOLING WATER LINE', 'CW LINES'],
        'REFRIGERANT LINES': ['REFRIGERANT LINES', 'REFRIGERANT LINE', 'REF LINES'],
        
        # 기타
        'LOCATION': ['LOCATION', 'MOUNTING PLACE', 'INSTALLATION PLACE', 'MOUNTING LOCATION'],
        'APPLICATION': ['APPLICATION', 'INSTALLATION METHOD', 'USE'],
        'PRIME MOVER': ['PRIME MOVER', 'DRIVER'],
        'RATING TIME': ['RATING TIME', 'DUTY CYCLE', 'CONTINUOUS OPERATION'],
        'HUMIDITY': ['HUMIDITY', 'RELATIVE HUMIDITY', 'RELATIVE HUMIDITY OF AIR'],
        'BLOWING AGENT': ['BLOWING AGENT', 'FOAMING AGENT'],
        'SURFACE PROTECTION': ['SURFACE PROTECTION', 'COATING', 'SURFACE COATING'],
        'ENCLOSURE': ['ENCLOSURE', 'ENCLOSURE TYPE', 'PROTECTION CLASS'],
        'ACCELERATION': ['ACCELERATION', 'VERTICAL ACCELERATION', 'TRANSVERSAL ACCELERATION',
                        'LONGITUDINAL ACCELERATION'],
        'SUPPLIER': ['SUPPLIER', 'MAKER', 'MANUFACTURER', 'VENDOR'],
        'DOCUMENTATION': ['DOCUMENTATION', 'DOCUMENTS', 'DOCUMENT'],
        'SCOPE OF SUPPLY': ['SCOPE OF SUPPLY', 'SCOPE OFSUPPLY', 'SUPPLY SCOPE'],
        'FOUNDATION': ['FOUNDATION', 'FOUNDATION OF EQUIPMENT', 'EQUIPMENT FOUNDATION'],
        'NAME PLATE': ['NAME PLATE', 'NAMEPLATE', 'NAME PLATES'],
        'ACCESSORIES': ['ACCESSORIES', 'ACCESSORY', 'ACCESSORIESFOR'],
        'TOTAL': ['TOTAL', 'TOTAL VOLUME', 'TOTAL CAPACITY'],
    }
    
    # 정확 매칭 키워드 (이 키워드는 정확히 매칭되어야 함)
    EXACT_MATCH_KEYS = {'NO.', 'NO', 'TYPE', 'SPEED', 'POWER', 'OUTPUT', 'NCR', 'MCR'}
    
    def __init__(self, html_content: str = "", file_path: str = ""):
        self.html_content = html_content
        self.file_path = file_path
        self.soup = None
        self.tables = []           # 원시 테이블 (2D 배열)
        self.table_structures = [] # 테이블 구조 정보 (헤더, 데이터 행 위치)
        self.kv_pairs = []         # 키-값 쌍 리스트
        self.kv_index = {}         # KV Direct Matching 최적화: 정규화된 키 -> KV 매핑
        self.text_chunks = []

        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                self.html_content = f.read()

        # HTML 정규화: <sup>O</sup>C → °C
        if self.html_content:
            self.html_content = self._normalize_html_units(self.html_content)

        if self.html_content:
            self._parse()
    
    def _parse(self):
        """HTML 파싱"""
        if not HAS_BS4:
            logger.warning("BeautifulSoup 미설치")
            return

        self.soup = BeautifulSoup(self.html_content, 'html.parser')
        self._extract_tables()
        self._extract_kv_pairs()
        self._extract_text_chunks()

    def _normalize_html_units(self, html_content: str) -> str:
        """
        HTML 단위 표기 정규화

        변환 패턴:
        - <sup>O</sup>C → °C
        - O C → °C
        - OC → °C (단, 단어 경계에서만)

        Args:
            html_content: 원본 HTML

        Returns:
            정규화된 HTML
        """
        # Pattern 1: <sup>O</sup>C 또는 <sup>o</sup>C → °C
        html_content = re.sub(r'<sup>\s*[Oo]\s*</sup>\s*C', '°C', html_content)

        # Pattern 2: O C 또는 o C (공백 포함) → °C
        html_content = re.sub(r'(?<!\w)[Oo]\s+C(?!\w)', '°C', html_content)

        # Pattern 3: OC 또는 oC (단어 경계) → °C
        # 주의: "DOCUMENT", "PROCEDURE" 같은 단어는 건드리지 않음
        html_content = re.sub(r'(?<![A-Za-z])OC(?![A-Za-z])', '°C', html_content)
        html_content = re.sub(r'(?<![A-Za-z])oC(?![A-Za-z])', '°C', html_content)

        return html_content

    def _extract_tables(self):
        """테이블 추출"""
        if not self.soup:
            return

        self.tables = []
        self.table_structures = []

        for table in self.soup.find_all('table'):
            table_data = []
            rows = table.find_all('tr')

            # 구조 정보 초기화
            structure = {
                'header_row_idx': -1,
                'header_cols': [],
                'data_start_row': 0,
                'col_count': 0
            }

            for row_idx, row in enumerate(rows):
                cells = []
                cell_tags = row.find_all(['td', 'th'])

                for cell in cell_tags:
                    text = cell.get_text(strip=True)
                    text = re.sub(r'\s+', ' ', text).strip()
                    cells.append(text)

                if cells and any(c for c in cells):  # 비어있지 않은 행만
                    table_data.append(cells)

                    # 헤더 행 감지
                    has_th = row.find('th') is not None
                    if has_th or (row_idx == 0 and structure['header_row_idx'] == -1):
                        # 헤더 행인지 추가 검증
                        if self._is_likely_header_row(cells):
                            structure['header_row_idx'] = len(table_data) - 1  # table_data 인덱스 사용
                            structure['header_cols'] = cells
                            structure['data_start_row'] = len(table_data)  # 다음 행부터 데이터

                    structure['col_count'] = max(structure['col_count'], len(cells))

            if table_data:
                self.tables.append(table_data)
                self.table_structures.append(structure)
    
    def _extract_kv_pairs(self):
        """
        테이블에서 키-값 쌍 추출

        개선사항:
        1. 수평 데이터 테이블 지원 (row=item, column=attribute)
        2. 수직 키-값 테이블 지원 (traditional key|value)
        3. 멀티 레이어 헤더 감지
        4. 중복 제거 강화
        """
        all_pairs = []

        # BeautifulSoup로 직접 파싱 (더 정확한 테이블 구조 파악)
        if not self.soup:
            return

        for table in self.soup.find_all('table'):
            # 수평 데이터 테이블 시도
            horizontal_pairs = self._extract_horizontal_data_table(table)
            all_pairs.extend(horizontal_pairs)

            # 수직 키-값 테이블 시도
            vertical_pairs = self._extract_vertical_kv_table(table)
            all_pairs.extend(vertical_pairs)

        # 중복 제거
        seen = set()
        self.kv_pairs = []
        for pair in all_pairs:
            norm_key = self._aggressive_normalize(pair['key'])
            norm_value = self._aggressive_normalize(pair['value'])
            pair_signature = (norm_key, norm_value)

            if pair_signature not in seen:
                seen.add(pair_signature)
                self.kv_pairs.append(pair)

        # KV Direct Matching 인덱스 구축 (O(1) 조회)
        self._build_kv_index()

    def _build_kv_index(self):
        """
        KV Direct Matching 최적화: 정규화된 키를 인덱스로 저장

        여러 정규화 버전을 저장하여 다양한 매칭 시도를 빠르게 처리:
        1. 완전 정규화 (공백/특수문자 제거)
        2. 번호 prefix 제거
        3. 계층적 키의 spec 부분만
        """
        self.kv_index = {}

        def normalize_key_simple(key: str) -> str:
            """간단한 정규화 - 번호 prefix 제거, 공백 정리"""
            key = key.strip()
            key = re.sub(r'^[A-Z]?\d+[\)\.\:\-]\s*', '', key)  # 번호 제거
            key = re.sub(r'[_\-\s]+', ' ', key).strip()  # 공백 정규화
            return key.upper()

        for kv in self.kv_pairs:
            original_key = kv['key']

            # 1. 완전 정규화 버전 (aggressive)
            norm_key = self._aggressive_normalize(original_key)
            if norm_key and norm_key not in self.kv_index:
                self.kv_index[norm_key] = kv

            # 2. 간단한 정규화 버전 (공백 유지)
            simple_norm = normalize_key_simple(original_key)
            if simple_norm and simple_norm not in self.kv_index:
                self.kv_index[simple_norm] = kv

            # 3. 계층적 키의 spec 부분만 추출
            if '_' in original_key or '(' in original_key:
                parsed = self._parse_hierarchical_key(original_key)
                if parsed['spec'] and parsed['spec'] not in self.kv_index:
                    self.kv_index[parsed['spec']] = kv

    def _aggressive_normalize(self, text: str) -> str:
        """강화된 정규화"""
        if not text:
            return ""
        text = re.sub(r'\s+', '', text)
        text = text.replace('*', '').replace('□', '').replace('■', '')
        text = text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        # 숫자 내 쉼표 제거 (1,000 → 1000)
        text = text.replace(',', '')
        return text.upper()

    def verify_value_in_document(self, value: str, unit: str = "") -> bool:
        """
        단위 변환 검증: 추출된 값이 원본 문서에 실제로 존재하는지 확인

        Purpose:
        - 단위 변환이 발생하지 않았는지 검증 (Ctrl+F 테스트)
        - 예: "1000" 추출 시 문서에 "1000", "1,000" 등이 있는지 확인

        Args:
            value: 추출된 값
            unit: 추출된 단위 (선택)

        Returns:
            True if value exists in document, False otherwise
        """
        if not value or not self.html_content:
            return True  # 검증 불가 시 통과

        # 검색 패턴 생성
        search_patterns = []

        # 1. 값 + 단위 조합
        if unit:
            # "6 tonnes", "6tonnes", "6 ton"
            search_patterns.append(f"{value}\\s*{unit}")
            search_patterns.append(f"{value}{unit}")

        # 2. 값만 (쉼표 포함/제외)
        # "1000" → ["1000", "1,000"]
        clean_value = value.replace(',', '')
        if clean_value.isdigit() and len(clean_value) >= 4:
            # 천 단위 쉼표 추가
            formatted = "{:,}".format(int(clean_value))
            search_patterns.append(clean_value)
            search_patterns.append(formatted)
        else:
            search_patterns.append(value)

        # HTML 컨텐츠에서 검색 (대소문자 무시)
        html_lower = self.html_content.lower()

        for pattern in search_patterns:
            # 정규식으로 검색
            if re.search(re.escape(pattern.lower()), html_lower):
                return True

        # 추가: 소수점 형식 변환 (3.5 vs 3,5)
        if '.' in value:
            comma_version = value.replace('.', ',')
            if re.search(re.escape(comma_version.lower()), html_lower):
                return True

        # 어떤 패턴도 찾지 못함 - 변환이 발생했을 가능성
        return False

    def _detect_header_row(self, row_cells: List[str]) -> bool:
        """헤더 행 감지"""
        if not row_cells:
            return False

        non_empty = [c for c in row_cells if c.strip()]
        if len(non_empty) < len(row_cells) * 0.3:
            return False

        header_keywords = ['COMPOSITION', 'RANGE', 'DESIGN', 'ITEM', 'SPECIFICATION',
                           'PARAMETER', 'VALUE', 'UNIT', 'TYPE', 'MODEL']

        text = ' '.join(row_cells).upper()
        has_keyword = any(kw in text for kw in header_keywords)

        digit_count = sum(1 for c in text if c.isdigit())
        total_chars = len([c for c in text if c.isalnum()])
        digit_ratio = digit_count / total_chars if total_chars > 0 else 0

        return has_keyword or digit_ratio < 0.3

    def _extract_horizontal_data_table(self, table) -> List[Dict]:
        """
        수평 데이터 테이블 파싱 (row=item, column=attribute)

        다층 헤더 지원
        """
        rows = table.find_all('tr')
        if not rows:
            return []

        # 헤더 감지 (다층 헤더 지원)
        header_row_objects = []  # BeautifulSoup row objects
        header_texts = []  # 텍스트 리스트
        data_start_idx = 0

        for i, row in enumerate(rows[:5]):
            cells = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
            if self._detect_header_row(cells):
                header_row_objects.append(row)
                header_texts.append(cells)
                data_start_idx = i + 1
            else:
                break

        if not header_row_objects:
            return []

        # 다층 헤더 병합
        merged_headers = self._merge_multi_layer_headers(header_row_objects, header_texts)

        # 데이터 행 파싱
        kv_pairs = []

        for row_idx in range(data_start_idx, len(rows)):
            row = rows[row_idx]
            cells = row.find_all(['td', 'th'])

            if not cells:
                continue

            row_label_raw = cells[0].get_text(strip=True)

            if not row_label_raw or len(row_label_raw) > 100:
                continue

            row_label = re.sub(r'\s+', ' ', row_label_raw)

            for col_idx, cell in enumerate(cells[1:], start=1):
                value_raw = cell.get_text(strip=True)

                if not value_raw or len(value_raw) > 200:
                    continue

                value = re.sub(r'\s+', ' ', value_raw)

                # 병합된 헤더 사용
                col_name = merged_headers[col_idx] if col_idx < len(merged_headers) else f"Column{col_idx}"

                if col_name and col_name.strip():
                    combined_key = f"{row_label}_{col_name}"
                else:
                    combined_key = row_label

                kv_pairs.append({
                    'key': combined_key,
                    'value': value,
                    'row': [row_label, value]
                })

        return kv_pairs

    def _merge_multi_layer_headers(self, header_rows: List, header_texts: List[List[str]]) -> List[str]:
        """
        다층 헤더 병합 (colspan, rowspan 지원)

        Args:
            header_rows: BeautifulSoup row objects
            header_texts: 각 행의 텍스트 리스트

        Returns:
            병합된 헤더 리스트
        """
        if not header_rows:
            return []

        if len(header_rows) == 1:
            # 단일 헤더
            return header_texts[0]

        # 다층 헤더 처리 (rowspan, colspan 고려)
        first_row = header_rows[0]
        first_cells = first_row.find_all(['td', 'th'])

        # 각 셀의 colspan, rowspan 분석
        col_info = []  # [(text, colspan, rowspan), ...]
        for cell in first_cells:
            text = cell.get_text(strip=True)
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            col_info.append((text, colspan, rowspan))

        # 두 번째 헤더 행이 있으면 병합
        if len(header_rows) >= 2:
            second_row = header_rows[1]
            second_cells = second_row.find_all(['td', 'th'])
            second_texts = [c.get_text(strip=True) for c in second_cells]

            # 병합된 헤더 생성
            merged = []
            second_idx = 0

            for parent_text, colspan, rowspan in col_info:
                if rowspan >= 2:
                    # rowspan=2 이상이면, 이 셀은 두 행에 걸쳐있음
                    # 하위 헤더 없이 그대로 사용
                    merged.append(parent_text)
                elif colspan == 1:
                    # colspan=1이고 rowspan=1이면 하위 헤더 1개
                    # 하지만 실제로는 두 번째 행에 대응하는 셀이 없을 수도 있음
                    # 안전하게 parent만 사용
                    merged.append(parent_text)
                else:
                    # colspan > 1이면, 이 parent 아래에 colspan개의 하위 헤더가 있음
                    for _ in range(colspan):
                        if second_idx < len(second_texts):
                            child_text = second_texts[second_idx]
                            if child_text:
                                # 상위_하위 형식으로 병합
                                merged.append(f"{parent_text}_{child_text}")
                            else:
                                merged.append(parent_text)
                            second_idx += 1
                        else:
                            merged.append(parent_text)

            return merged

        return header_texts[0]

    def _extract_value_from_long_text(self, text: str) -> str:
        """
        긴 텍스트에서 첫 번째 의미있는 값 추출 (보편적 방법)

        패턴:
        - 숫자 + 단위 (예: "100%", "50 bar", "25°C")
        - 순수 숫자 + 단위 단어 (예: "3 SET", "5 units")

        Args:
            text: 추출할 텍스트

        Returns:
            추출된 값 (없으면 원본 텍스트의 앞부분)
        """
        if not text or len(text) < 200:
            return text

        # 패턴 1: 숫자 + % (가장 흔한 패턴)
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if match:
            return match.group(0)

        # 패턴 2: 숫자 + 단위 (bar, psi, °C, °F, etc.)
        match = re.search(r'(\d+(?:\.\d+)?)\s*(bar|psi|°C|°F|Pa|MPa|kPa|kg|mm|cm|m|kW|MW|V|A|Hz|rpm)', text, re.IGNORECASE)
        if match:
            return match.group(0)

        # 패턴 3: 숫자 + SET/LOT/UNIT 등
        match = re.search(r'(\d+)\s*(SET|LOT|UNIT|UNITS|EA|PCS|PC)\b', text, re.IGNORECASE)
        if match:
            return match.group(0)

        # 패턴 4: 범위 (예: "5 ~ 10", "20-30")
        match = re.search(r'(\d+(?:\.\d+)?)\s*[~\-]\s*(\d+(?:\.\d+)?)', text)
        if match:
            return match.group(0)

        # 패턴 5: 순수 숫자 (마지막 수단)
        match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
        if match:
            return match.group(0)

        # 추출 실패: 앞부분 200자만 반환
        return text[:200]

    def _extract_vertical_kv_table(self, table) -> List[Dict]:
        """
        수직 키-값 테이블 파싱 (row: key | value)

        개선사항:
        - 긴 값(>200자)도 처리: 첫 번째 의미있는 값 추출
        - 다중 컬럼 지원: Cell[i+1]이 너무 길면 Cell[i+2]도 확인
        - 최대 값 길이: 1000자 (안전 제한)
        """
        rows = table.find_all('tr')
        kv_pairs = []

        for row in rows:
            cells = row.find_all(['td', 'th'])

            if len(cells) < 2:
                continue

            for i in range(len(cells) - 1):
                key_raw = cells[i].get_text(strip=True)

                if not key_raw or len(key_raw) < 3 or len(key_raw) > 150:
                    continue

                key = re.sub(r'\s+', ' ', key_raw)

                noise_patterns = [
                    r'^GENERAL\b', r'^TABLE\b', r'^SECTION\b', r'^PAGE\b',
                    r'^ITEM\s*NO', r'^NO\.\s*$', r'^DESCRIPTION\s*$'
                ]
                if any(re.search(pat, key.upper()) for pat in noise_patterns):
                    continue

                # 다중 컬럼 값 시도
                values_to_try = []

                # Cell[i+1] (primary value)
                if i + 1 < len(cells):
                    values_to_try.append(cells[i + 1].get_text(strip=True))

                # Cell[i+2] (secondary value, if primary is very long)
                if i + 2 < len(cells):
                    primary_len = len(values_to_try[0]) if values_to_try else 0
                    if primary_len > 500:  # Very long primary value
                        values_to_try.append(cells[i + 2].get_text(strip=True))

                # 각 값 후보에 대해 처리
                for value_raw in values_to_try:
                    if not value_raw or len(value_raw) > 1000:  # Safety limit
                        continue

                    value = re.sub(r'\s+', ' ', value_raw)

                    # 긴 값 처리: 의미있는 부분 추출
                    if len(value) >= 200:
                        value = self._extract_value_from_long_text(value)

                    if value:
                        kv_pairs.append({
                            'key': key,
                            'value': value,
                            'row': [key, value]
                        })
                        break  # 첫 번째 유효한 값을 찾았으므로 중단

        return kv_pairs
    
    def _normalize_cell_key(self, key: str) -> str:
        """
        셀 키 정규화 (일반화된 처리)
        
        1. CamelCase 분리: "MainComponent" → "Main Component"
        2. 특정 패턴만 안전하게 분리
        3. 연속 공백 정리
        
        주의: 일반적인 전치사 분리는 하지 않음 (오탐 방지)
        """
        if not key:
            return key
        
        original_key = key
        
        # CamelCase 분리 (소문자 뒤에 대문자가 오는 경우만)
        # "MainComponent" → "Main Component"
        key = re.sub(r'([a-z])([A-Z])', r'\1 \2', key)
        
        # 연속 공백 정리
        key = re.sub(r'\s+', ' ', key).strip()
        
        return key
    
    def _compute_fuzzy_match_score(self, key: str, spec_name: str, equipment: str = "",
                                    expected_unit: str = "") -> float:
        """
        개선된 Fuzzy Matching 점수 계산 (토큰 기반)

        가중치:
        - Spec name: 40%
        - Equipment: 30%
        - Unit: 30%

        Args:
            key: 테이블 키 (예: "Motor Power_Main Engine")
            spec_name: 대상 사양명
            equipment: 대상 장비명 (선택)
            expected_unit: 예상 단위 (선택)

        Returns:
            0.0 ~ 1.0 점수 (높을수록 유사)
        """
        # 계층적 키 파싱
        parsed = self._parse_hierarchical_key(key)

        score = 0.0
        weights = {'spec': 0.4, 'equipment': 0.3, 'unit': 0.3}

        # 1. Spec name 매칭 (40%)
        if parsed['spec']:
            spec_tokens = set(parsed['spec'].upper().split())
            target_tokens = set(spec_name.upper().split())

            if spec_tokens == target_tokens:
                score += weights['spec']  # 완벽 매칭
            else:
                # Token overlap ratio
                common = spec_tokens & target_tokens
                union = spec_tokens | target_tokens
                if union:
                    overlap_ratio = len(common) / len(union)
                    score += weights['spec'] * overlap_ratio

        # 2. Equipment 매칭 (30%)
        if equipment and parsed['equipment']:
            equip_tokens = set(parsed['equipment'].upper().split())
            target_equip_tokens = set(equipment.upper().split())

            if equip_tokens == target_equip_tokens:
                score += weights['equipment']
            else:
                common = equip_tokens & target_equip_tokens
                union = equip_tokens | target_equip_tokens
                if union:
                    overlap_ratio = len(common) / len(union)
                    score += weights['equipment'] * overlap_ratio
        elif not equipment or not parsed['equipment']:
            # Equipment가 둘 다 없으면 가중치를 spec에 재분배
            score += weights['equipment'] * 0.5  # 중립적 점수

        # 3. Unit 매칭 (30%)
        if expected_unit and parsed['unit']:
            unit_norm = parsed['unit'].upper().replace(' ', '').replace('/', '')
            expected_norm = expected_unit.upper().replace(' ', '').replace('/', '')

            if unit_norm == expected_norm:
                score += weights['unit']
            elif unit_norm in expected_norm or expected_norm in unit_norm:
                score += weights['unit'] * 0.7  # 부분 매칭
        elif not expected_unit or not parsed['unit']:
            # Unit이 둘 다 없으면 가중치를 spec에 재분배
            score += weights['unit'] * 0.5

        return score

    def _parse_hierarchical_key(self, key: str) -> Dict[str, str]:
        """
        계층적 키 파싱

        형식: "<index>_<spec>_<equipment>(<unit>)#" 또는 "<spec>_<equipment>"

        예시:
        - "1_Capacity_Air volume(m3/h)#" → {index: "1", spec: "CAPACITY", equipment: "Air volume", unit: "m3/h"}
        - "Motor Power_Main Engine" → {spec: "MOTOR POWER", equipment: "Main Engine"}
        - "CAPACITY(m³/h)" → {spec: "CAPACITY", unit: "m³/h"}

        Returns:
            Dictionary with keys: index, spec, equipment, unit (missing keys have empty strings)
        """
        result = {
            'index': '',
            'spec': '',
            'equipment': '',
            'unit': ''
        }

        if not key:
            return result

        key = key.strip().rstrip('#')  # Remove trailing #

        # Extract unit from parentheses if present
        unit_match = re.search(r'\(([^)]+)\)\s*$', key)
        if unit_match:
            result['unit'] = unit_match.group(1).strip()
            key = key[:unit_match.start()].strip()

        # Split by underscore
        parts = key.split('_')

        if len(parts) == 1:
            # Simple key: just spec name (or spec + unit in parens)
            result['spec'] = parts[0].strip().upper()
        elif len(parts) == 2:
            # Two parts: could be "index_spec" or "spec_equipment"
            first = parts[0].strip()
            second = parts[1].strip()

            # Check if first part is a number (index)
            if re.match(r'^\d+$', first):
                result['index'] = first
                result['spec'] = second.upper()
            else:
                # spec_equipment
                result['spec'] = first.upper()
                result['equipment'] = second
        elif len(parts) >= 3:
            # Three or more parts: "index_spec_equipment" or "spec_subspec_equipment"
            first = parts[0].strip()

            if re.match(r'^\d+$', first):
                # index_spec_equipment
                result['index'] = first
                result['spec'] = parts[1].strip().upper()
                result['equipment'] = '_'.join(parts[2:]).strip()
            else:
                # spec_subspec_equipment (treat first N-1 parts as spec)
                result['spec'] = '_'.join(parts[:-1]).upper()
                result['equipment'] = parts[-1].strip()

        return result

    def _is_likely_header_row(self, cells: List[str]) -> bool:
        """
        헤더 행인지 판단

        개선사항:
        - 헤더 키워드 수 카운트
        - 숫자값 비율 확인
        - 더 정교한 판단
        """
        if not cells:
            return False

        # 헤더 키워드
        header_keywords = [
            'type', 'item', 'description', 'spec', 'specification', 'parameter',
            'unit', 'value', 'qty', "q'ty", 'quantity', 'remark', 'no.', 'no',
            'name', 'model', 'capacity', 'material', 'maker', 'size'
        ]

        keyword_count = 0
        numeric_count = 0

        for cell in cells:
            cell_lower = cell.lower().strip()

            # 헤더 키워드 확인
            if any(kw in cell_lower for kw in header_keywords):
                keyword_count += 1

            # 숫자값 확인 (데이터 행일 가능성)
            if re.match(r'^[\d,.\-]+\s*[a-zA-Z]*$', cell.strip()):
                numeric_count += 1

        # 키워드가 많고 숫자가 적으면 헤더
        return keyword_count >= 2 or (keyword_count >= 1 and numeric_count == 0)

    def _is_header_row_table(self, table: List[List[str]]) -> bool:
        """
        테이블의 첫 행이 헤더인지 판단

        헤더 판단 기준:
        - 첫 행의 모든 셀이 짧은 텍스트 (헤더 키워드)
        - 첫 행에 숫자+단위가 없음 (데이터 행이면 숫자가 있음)
        """
        if not table or len(table) < 2:
            return False

        first_row = table[0]
        if not first_row:
            return False

        # 첫 행에 숫자+단위 패턴이 있으면 데이터 행 (헤더 아님)
        for cell in first_row:
            # 숫자+단위 패턴: "1,000 mbar", "45℃", "120 mm" 등
            if re.search(r'\d+[,.]?\d*\s*(mbar|bar|mm|m|℃|°C|%|kg|kW|rpm|Hz|m3/h|g/kWh)', cell, re.I):
                return False
            # 순수 숫자 값 (3자리 이상)
            if re.match(r'^[\d.,]+$', cell.strip()) and len(cell.strip()) >= 3:
                return False
        
        # 첫 행의 모든 셀이 헤더 키워드인지 확인
        header_keywords = {
            'TYPE', 'NAME', 'NO.', 'NO', 'INDEX', 'Q\'TY', 'QTY', 'QUANTITY',
            'DESCRIPTION', 'PARTICULARS', 'ITEM', 'SPEC', 'UNIT', 'REMARK',
            'RATE', 'CONDITION', 'VALUE', 'MATERIAL', 'SIZE', 'CAPACITY',
            'MODEL', 'MAKER', 'SUPPLIER', 'LOCATION', 'APPLICATION'
        }
        
        header_count = 0
        for cell in first_row:
            cell_upper = cell.upper().strip()
            # 비어있거나 짧은 텍스트이고 숫자 없으면 헤더 가능성
            if not cell_upper:
                header_count += 1
            elif cell_upper in header_keywords:
                header_count += 1
            elif len(cell_upper) <= 12 and not re.search(r'\d', cell_upper):
                header_count += 1
        
        # 70% 이상이 헤더 특성이면 헤더 행으로 판단
        return header_count >= len(first_row) * 0.7
    
    def _is_valid_kv_pair(self, key: str, value: str, seen_pairs: set) -> bool:
        """
        유효한 키-값 쌍인지 판단 (일반화된 검증)
        
        검증 기준:
        1. 키가 비어있지 않고 적절한 길이
        2. 값이 비어있지 않음
        3. 키가 숫자만으로 구성되지 않음
        4. 키가 값처럼 보이지 않음
        5. 값이 헤더 키워드가 아님
        6. 키와 값이 동일하지 않음
        """
        if not key or not value:
            return False
        
        # 키 길이 제한 (너무 짧거나 너무 긴 키 제외)
        if len(key) < 2 or len(key) > 100:
            return False
        
        # 숫자만으로 구성된 키 제외
        if key.isdigit() or re.match(r'^[\d.,\-+%\s]+$', key):
            return False
        
        # 키가 값처럼 보이면 제외 (숫자+단위 패턴)
        if self._is_likely_value(key):
            return False
        
        # 값이 헤더 키워드면 제외 (헤더 행 잘못 추출 방지)
        if self._is_header_keyword(value):
            return False
        
        # 키와 값이 동일하면 제외
        if key.strip().upper() == value.strip().upper():
            return False
        
        # 키가 특수문자로만 구성되면 제외
        if re.match(r'^[\s\-\*\#\.\:\,]+$', key):
            return False
        
        return True
    
    def _is_header_keyword(self, text: str) -> bool:
        """헤더 키워드인지 판단"""
        header_keywords = {
            'TYPE', 'NAME', 'NO.', 'NO', 'INDEX', 'Q\'TY', 'QTY', 'QUANTITY',
            'DESCRIPTION', 'PARTICULARS', 'ITEM', 'SPEC', 'SPECIFICATION',
            'UNIT', 'REMARK', 'REMARKS', 'RATE', 'CONDITION', 'VALUE',
            'MATERIAL', 'SIZE', 'CAPACITY', 'MODEL', 'MAKER', 'SUPPLIER',
            'LOCATION', 'APPLICATION', 'DESIGN', 'PRESSURE', 'TEMPERATURE',
            'FLOW', 'HEAD', 'POWER', 'OUTPUT', 'SPEED', 'RPM', 'VOLTAGE',
            'FLUID', 'CERT.', 'CERTIFICATE', 'ELECTRIC MOTOR', 'PARTICULAR'
        }
        text_upper = text.upper().strip()
        return text_upper in header_keywords
    
    def _is_likely_value(self, text: str) -> bool:
        """
        값으로 보이는지 판단 (일반화된 패턴)
        
        값의 특징:
        1. 숫자로 시작
        2. 단위 포함
        3. 숫자+단위 패턴
        4. 수량 패턴 (1 set, 2 ea 등)
        
        예외:
        - 번호 prefix (예: "1) Type", "2) Specified MCR")
        """
        text = text.strip()
        
        # 번호 prefix 패턴은 값이 아님 (키의 일부)
        # 예: "1) Type", "2) Specified MCR", "A01: Hull"
        if re.match(r'^[A-Z]?\d+[\)\.\:\-]\s*[A-Za-z]', text):
            return False
        
        # 숫자로 시작하면 값 (날짜 패턴 제외)
        if re.match(r'^[\d.,\-+]', text):
            # 날짜 패턴은 키일 수 있음 (예: "2024-01-01")
            if not re.match(r'^\d{4}[-/]\d{2}[-/]\d{2}', text):
                return True
        
        # 단위가 포함되면 값
        unit_pattern = r'\b(kW|MW|HP|rpm|bar|mbar|m3|m³|mm|cm|m\b|kg|ton|V|Hz|°C|℃|%|l/min|m3/h|g/kWh|kg/m³|W/mK|Kcal)\b'
        if re.search(unit_pattern, text, re.I):
            return True
        
        # 수량 패턴 (숫자 + 단위)
        if re.match(r'^\d+\s*(set|sets|ea|pcs|units?)\b', text, re.I):
            return True
        
        # 범위 패턴 (예: "25~40", "15~20")
        if re.match(r'^\d+\s*[~\-]\s*\d+', text):
            return True
        
        # 소수점 숫자 (예: "0.45", "1.025")
        if re.match(r'^[\d.]+$', text) and '.' in text:
            return True
        
        return False
    
    def _extract_text_chunks(self):
        """텍스트 청크 추출"""
        if not self.soup:
            return
        
        # 원본 soup 복사 (테이블 제거용)
        soup_copy = BeautifulSoup(str(self.soup), 'html.parser')
        for table in soup_copy.find_all('table'):
            table.decompose()
        
        text = soup_copy.get_text(separator='\n', strip=True)
        lines = text.split('\n')
        current_chunk = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_chunk:
                    self.text_chunks.append('\n'.join(current_chunk))
                    current_chunk = []
            else:
                current_chunk.append(line)
        
        if current_chunk:
            self.text_chunks.append('\n'.join(current_chunk))

    def search_in_tables_enhanced(self, keywords: List[str]) -> List[Dict]:
        """
        테이블 검색

        개선사항:
        - 헤더/데이터 구분
        - 열 위치 기반 값 추출
        - Case 1: 키워드가 헤더에 있으면 해당 열의 데이터 추출
        - Case 2: 키워드가 데이터에 있으면 인접 셀 추출
        """
        results = []

        for t_idx, table in enumerate(self.tables):
            structure = self.table_structures[t_idx] if t_idx < len(self.table_structures) else {}
            header_row_idx = structure.get('header_row_idx', -1)
            header_cols = structure.get('header_cols', [])
            data_start_row = structure.get('data_start_row', 0)

            for r_idx, row in enumerate(table):
                row_text = ' '.join(row).lower()

                for keyword in keywords:
                    kw_lower = keyword.lower()

                    if kw_lower not in row_text:
                        continue

                    # 키워드가 어느 셀에 있는지 찾기
                    match_col_idx = -1
                    for c_idx, cell in enumerate(row):
                        if kw_lower in cell.lower():
                            match_col_idx = c_idx
                            break

                    if match_col_idx == -1:
                        continue

                    # Case 1: 키워드가 헤더에 있는 경우 → 해당 열의 데이터 추출
                    if r_idx == header_row_idx:
                        # 데이터 행에서 해당 열의 값 추출
                        for data_row_idx in range(data_start_row, len(table)):
                            data_row = table[data_row_idx]
                            if match_col_idx < len(data_row):
                                value = data_row[match_col_idx]
                                if value and not self._is_likely_header_keyword(value):
                                    results.append({
                                        'table_idx': t_idx,
                                        'row_idx': data_row_idx,
                                        'col_idx': match_col_idx,
                                        'row': data_row,
                                        'value': value,
                                        'match_type': 'header_column'
                                    })

                    # Case 2: 키워드가 데이터 행에 있는 경우 → 인접 셀 추출
                    else:
                        # 오른쪽 셀 확인
                        if match_col_idx + 1 < len(row):
                            value = row[match_col_idx + 1]
                            if value and not self._is_likely_header_keyword(value):
                                results.append({
                                    'table_idx': t_idx,
                                    'row_idx': r_idx,
                                    'col_idx': match_col_idx + 1,
                                    'row': row,
                                    'value': value,
                                    'match_type': 'adjacent_cell'
                                })

        return results

    def _is_likely_header_keyword(self, value: str) -> bool:
        """헤더 키워드일 가능성 체크 (값이 아닌 헤더)"""
        value_upper = value.upper().strip()

        # 일반적인 헤더 키워드
        header_keywords = ['type', 'qty', "q'ty", 'remark', 'unit', 'item', 'description',
                          'spec', 'specification', 'parameter', 'value', 'no.', 'no']
        if value_upper.lower() in [kw.lower() for kw in header_keywords]:
            return True

        return False

    def find_value_in_tables(
        self, 
        spec_name: str, 
        equipment: str = ""
    ) -> Optional[Tuple[str, str, str]]:
        """
        테이블에서 사양값 찾기
        
        개선사항:
        - 번호 prefix 제거 ("1) Type" → "Type")
        - 괄호 suffix 구분 ("NCR" vs "NCR(85%)")
        - 부분 매칭 강화
        
        1. 키-값 쌍에서 직접 검색 (정확 매칭)
        2. 번호 prefix 제거 후 매칭
        3. 테이블 행 검색
        4. 동의어 매칭
        """
        spec_upper = spec_name.upper().strip()
        
        # 동의어 목록 생성
        variants = self._get_spec_variants(spec_upper)
        
        # 키 정규화 함수
        def normalize_key(key: str) -> str:
            """키 정규화 - 번호 prefix 제거, 공백 정리"""
            key = key.strip()
            # "1) Type" → "Type", "A01: Hull" → "Hull"
            key = re.sub(r'^[A-Z]?\d+[\)\.\:\-]\s*', '', key)
            # 공백 정규화
            key = re.sub(r'[_\-\s]+', ' ', key).strip()
            return key.upper()
        
        # 스펙 이름 정규화
        spec_normalized = normalize_key(spec_upper)

        # 0a. KV Direct Matching (O(1) 인덱스 조회 - 가장 빠름)
        # 여러 정규화 버전으로 시도
        for variant in variants:
            # Aggressive normalization
            norm_variant = self._aggressive_normalize(variant)
            if norm_variant in self.kv_index:
                kv = self.kv_index[norm_variant]
                value = kv['value']
                clean_value, unit = self._parse_value_unit(value)
                if clean_value and self._is_valid_value(clean_value, spec_upper):
                    # 단위 변환 검증
                    if self.verify_value_in_document(clean_value, unit):
                        return (clean_value, unit, f"{kv['key']} | {value}")

            # Simple normalization
            simple_norm = normalize_key(variant)
            if simple_norm in self.kv_index:
                kv = self.kv_index[simple_norm]
                value = kv['value']
                clean_value, unit = self._parse_value_unit(value)
                if clean_value and self._is_valid_value(clean_value, spec_upper):
                    return (clean_value, unit, f"{kv['key']} | {value}")

        # 0. 계층적 키 매칭 (가장 먼저 시도)
        # "1_Capacity_Air volume(m3/h)#" 같은 복잡한 키 처리
        for kv in self.kv_pairs:
            if '_' in kv['key'] or '(' in kv['key']:
                parsed = self._parse_hierarchical_key(kv['key'])

                # Spec name이 일치하는지 확인
                if parsed['spec'] and parsed['spec'] in variants:
                    # Equipment 매칭도 확인 (있는 경우)
                    if equipment:
                        # Equipment가 키의 equipment 부분과 유사한지 확인
                        if parsed['equipment']:
                            equip_normalized = parsed['equipment'].upper()
                            if equipment.upper() in equip_normalized or equip_normalized in equipment.upper():
                                value = kv['value']
                                clean_value, unit = self._parse_value_unit(value)
                                # Unit hint가 있으면 우선 사용
                                if parsed['unit'] and not unit:
                                    unit = parsed['unit']
                                if clean_value and self._is_valid_value(clean_value, spec_upper):
                                    return (clean_value, unit, f"{kv['key']} | {value}")
                        # Equipment가 명시되지 않은 경우도 매칭
                        elif not parsed['equipment']:
                            value = kv['value']
                            clean_value, unit = self._parse_value_unit(value)
                            if parsed['unit'] and not unit:
                                unit = parsed['unit']
                            if clean_value and self._is_valid_value(clean_value, spec_upper):
                                return (clean_value, unit, f"{kv['key']} | {value}")
                    else:
                        # Equipment 조건 없음 - spec만 매칭
                        value = kv['value']
                        clean_value, unit = self._parse_value_unit(value)
                        if parsed['unit'] and not unit:
                            unit = parsed['unit']
                        if clean_value and self._is_valid_value(clean_value, spec_upper):
                            return (clean_value, unit, f"{kv['key']} | {value}")

        # 1. 정확 매칭 먼저 시도 (번호 prefix 제거 후)
        for kv in self.kv_pairs:
            key_normalized = normalize_key(kv['key'])
            
            # 정확 매칭
            if key_normalized == spec_normalized:
                value = kv['value']
                clean_value, unit = self._parse_value_unit(value)
                if clean_value and self._is_valid_value(clean_value, spec_upper):
                    return (clean_value, unit, f"{kv['key']} | {value}")
        
        # 2. 괄호 없는 버전으로 정확 매칭 (NCR vs NCR(85%) 구분)
        spec_no_paren = re.sub(r'\([^)]*\)', '', spec_normalized).strip()
        
        for kv in self.kv_pairs:
            key_normalized = normalize_key(kv['key'])
            key_no_paren = re.sub(r'\([^)]*\)', '', key_normalized).strip()
            
            # 괄호 제거 후 정확 매칭 (단, 원본에 괄호가 없어야 함)
            if key_no_paren == spec_no_paren:
                # 키에 괄호가 없고 스펙에도 괄호가 없는 경우만 매칭
                if '(' not in kv['key'] and '(' not in spec_name:
                    value = kv['value']
                    clean_value, unit = self._parse_value_unit(value)
                    if clean_value and self._is_valid_value(clean_value, spec_upper):
                        return (clean_value, unit, f"{kv['key']} | {value}")

        # 2a. Enhanced fuzzy matching (토큰 기반, 계층적 키 지원)
        best_match = None
        best_score = 0.0

        for kv in self.kv_pairs:
            # 토큰 기반 fuzzy matching score 계산
            fuzzy_score = self._compute_fuzzy_match_score(
                key=kv['key'],
                spec_name=spec_name,
                equipment=equipment,
                expected_unit=""  # Unit은 나중에 파싱에서 확인
            )

            if fuzzy_score > best_score and fuzzy_score >= 0.6:  # 임계값 0.6
                value = kv['value']
                clean_value, unit = self._parse_value_unit(value)

                if clean_value and self._is_valid_value(clean_value, spec_upper):
                    best_match = (clean_value, unit, f"{kv['key']} | {value}")
                    best_score = fuzzy_score

        if best_match and best_score >= 0.7:  # 높은 신뢰도면 즉시 반환
            return best_match

        # 3. 키-값 쌍에서 유사 매칭 (기존 방식, fallback)
        # best_match는 위에서 이미 선언됨
        # best_score = 0.0  # 초기화하지 않고 기존 값 유지
        
        for kv in self.kv_pairs:
            key = normalize_key(kv['key'])
            score = self._match_score(key, variants)
            
            # 괄호 불일치 패널티: 스펙에 괄호 없는데 키에 괄호 있으면 점수 감소
            if '(' not in spec_name and '(' in kv['key']:
                score *= 0.5
            
            if score > best_score and score >= 0.7:  # 임계값 0.7
                value = kv['value']
                
                # 값 정규화
                clean_value, unit = self._parse_value_unit(value)
                
                # 값이 유효한지 확인
                if clean_value and self._is_valid_value(clean_value, spec_upper):
                    best_match = (clean_value, unit, f"{kv['key']} | {value}")
                    best_score = score
        
        if best_match:
            return best_match
        
        # 4. 테이블 직접 검색 (폴백)
        for table in self.tables:
            for row in table:
                for i, cell in enumerate(row):
                    cell_normalized = normalize_key(cell)
                    score = self._match_score(cell_normalized, variants)
                    
                    # 괄호 불일치 패널티
                    if '(' not in spec_name and '(' in cell:
                        score *= 0.5
                    
                    if score > best_score and score >= 0.5:
                        # 다음 셀에서 값 추출
                        if i + 1 < len(row):
                            value = row[i + 1]
                            clean_value, unit = self._parse_value_unit(value)
                            
                            if clean_value and self._is_valid_value(clean_value, spec_upper):
                                best_match = (clean_value, unit, ' | '.join(row))
                                best_score = score
        
        return best_match
    
    def _get_spec_variants(self, spec_name: str) -> List[str]:
        """사양명 변형 목록 생성"""
        variants = [spec_name]
        
        # 괄호 내용 제거 버전
        no_paren = re.sub(r'\([^)]*\)', '', spec_name).strip()
        if no_paren and no_paren != spec_name:
            variants.append(no_paren)
        
        # 동의어 매핑에서 찾기
        for canonical, synonyms in self.SPEC_SYNONYMS.items():
            # 사양명이 동의어 목록에 있으면 모든 동의어 추가
            for syn in synonyms:
                if syn in spec_name or spec_name in syn:
                    variants.extend(synonyms)
                    break
        
        # 긴 키 이름 부분 매칭용 키워드 추출
        # "Estimated Back-pressure from after T/C to funnel" → ["BACK-PRESSURE", "BACK PRESSURE"]
        words = spec_name.split()
        if len(words) > 1:
            for word in words:
                if len(word) >= 4 and word not in ('TYPE', 'SIZE', 'RATE', 'FROM', 'AFTER', 'WITH'):
                    variants.append(word)
            
            # 연속된 2~3 단어 조합 추가 (핵심 키워드 추출)
            for i in range(len(words) - 1):
                two_word = ' '.join(words[i:i+2])
                if len(two_word) >= 6:
                    variants.append(two_word)
                
                if i + 2 < len(words):
                    three_word = ' '.join(words[i:i+3])
                    if len(three_word) >= 10:
                        variants.append(three_word)
        
        # "Type & Maker" 같은 특수 패턴 처리
        if '&' in spec_name:
            variants.append(spec_name.replace('&', 'AND'))
            variants.append(spec_name.replace('&', ' AND '))
        if ' AND ' in spec_name.upper():
            variants.append(spec_name.upper().replace(' AND ', ' & '))
        
        return list(set(variants))
    
    def _match_score(self, text: str, variants: List[str]) -> float:
        """
        매칭 점수 계산
        
        개선사항:
        1. 핵심 키워드 기반 매칭 강화
        2. 부분 문자열 매칭 유연화
        3. 약어/축약어 매칭 지원
        """
        text = text.strip().upper()
        
        if not text:
            return 0.0
        
        # 텍스트 정규화 (공백, 특수문자 정리)
        text_normalized = re.sub(r'[_\-\s]+', ' ', text).strip()
        
        max_score = 0.0
        
        for variant in variants:
            variant_upper = variant.upper()
            variant_normalized = re.sub(r'[_\-\s]+', ' ', variant_upper).strip()
            
            # 1. 정확 매칭 (최고 점수)
            if text_normalized == variant_normalized:
                return 1.0
            
            # 2. 키가 사양명으로 정확히 시작 (예: "OUTPUT (kW)" → OUTPUT)
            if text_normalized.startswith(variant_normalized + ' ') or \
               text_normalized.startswith(variant_normalized + '('):
                score = 0.95
                if score > max_score:
                    max_score = score
            
            # 3. 사양명이 키에 정확히 포함 (단어 경계 확인)
            if variant_normalized in text_normalized:
                # 단어 경계 확인
                pattern = r'\b' + re.escape(variant_normalized) + r'\b'
                if re.search(pattern, text_normalized, re.I):
                    score = 0.9
                    if score > max_score:
                        max_score = score
            
            # 4. 키가 사양명에 포함 (역방향)
            if text_normalized in variant_normalized:
                if len(text_normalized) >= 5:  # 최소 5자 이상
                    score = 0.85
                    if score > max_score:
                        max_score = score
        
        # 5. 핵심 키워드 기반 매칭 (부분 매칭 강화)
        for variant in variants:
            variant_upper = variant.upper()
            
            # 긴 키워드(6자 이상)가 텍스트에 포함
            if len(variant_upper) >= 6:
                if variant_upper in text.upper():
                    score = 0.75
                    if score > max_score:
                        max_score = score
            
            # 짧은 키워드(4-5자)도 단어 경계 확인하여 매칭
            elif len(variant_upper) >= 4:
                pattern = r'\b' + re.escape(variant_upper) + r'\b'
                if re.search(pattern, text.upper()):
                    score = 0.65
                    if score > max_score:
                        max_score = score
        
        # 6. 단어 집합 오버랩 매칭
        text_words = set(re.findall(r'[A-Z]{3,}', text.upper()))
        for variant in variants:
            variant_words = set(re.findall(r'[A-Z]{3,}', variant.upper()))
            if variant_words and text_words:
                overlap = len(text_words & variant_words)
                if overlap >= 2:  # 최소 2개 단어 일치
                    score = 0.6 + (overlap * 0.05)
                    if score > max_score:
                        max_score = min(score, 0.8)  # 최대 0.8
        
        return max_score
    
    def _parse_value_unit(self, raw: str) -> Tuple[str, str]:
        """
        값과 단위 분리
        
        예시:
        - "2,000 kW (bow)" → ("2,000", "kW")
        - "440V, 3 Phase, 60 Hz" → ("440V, 3 Phase, 60 Hz", "")
        - "120 mm" → ("120", "mm")
        - "40(±10) kg/m³" → ("40(±10)", "kg/m³")
        - "Approx. 3.5m3/h" → ("3.5", "m3/h")
        - "Tunnel, CPP" → ("Tunnel, CPP", "")
        - "Two(2) sets, each 100% capacity" → ("Two(2) sets", "")
        - "420mLCat the design cargo" → ("420", "mLC")
        - "Electric motor (single speed)" → ("Electric motor", "")
        """
        if not raw:
            return "", ""
        
        raw = raw.strip()
        original_raw = raw  # 원본 보존

        # 0a. 사양명 접두어 제거 (테이블 값에서 중복 사양명 제거)
        # 예: "SWL 6 tonnes" → "6 tonnes", "MCR 1000 kW" → "1000 kW"
        # 흔한 사양명 약어: SWL, MCR, NCR, RPM, BHP, SHP 등
        spec_prefix_pattern = r'^(SWL|MCR|NCR|BHP|SHP|TDH|NPSHr?|FLOW|HEAD|CAPACITY|POWER|VOLTAGE|FREQUENCY|SPEED|PRESSURE|TEMPERATURE|RPM|TH)\s+(\d)'
        spec_prefix_match = re.match(spec_prefix_pattern, raw, re.I)
        if spec_prefix_match:
            # 사양명 제거, 숫자부터 시작하도록
            prefix_len = len(spec_prefix_match.group(1)) + 1  # +1 for space
            raw = raw[prefix_len:].strip()

        # 0. 텍스트 수량 패턴 우선 처리 (One, Two, Three 등)
        text_qty_match = re.match(
            r'^(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s*\(\d+\)\s*(sets?|units?|ea|pcs)?',
            raw, re.I
        )
        if text_qty_match:
            # 쉼표 이전까지만 반환
            value = text_qty_match.group(0)
            return value, ""
        
        # 1. 쉼표로 분리된 부가 설명 제거 (단, 값이 완결된 경우만)
        # "Two(2) sets, each 100% capacity" → "Two(2) sets"
        if ', ' in raw:
            parts = raw.split(', ')
            first_part = parts[0].strip()
            # 첫 부분이 완결된 값인지 확인 (단위 포함 또는 텍스트 수량)
            if re.search(r'(sets?|units?|ea|pcs|bar|kW|rpm|mm|℃|°C|%)\s*$', first_part, re.I):
                raw = first_part
            elif re.search(r'\)\s*$', first_part):  # 괄호로 끝남
                raw = first_part
            elif len(raw) > 50:
                raw = first_part
        
        # 2. 단위 뒤에 붙은 설명 분리
        # "420mLCat the design cargo" → "420mLC" + "at the design cargo"
        # "Electric motor (single speed)" → "Electric motor"
        unit_boundary = re.search(
            r'(mLC|m3|m³|kW|rpm|bar|mm|cm|kg|ton|%|℃|°C|V|Hz|set|ea)\s*(at|by|for|with|and|or|\()',
            raw, re.I
        )
        if unit_boundary:
            raw = raw[:unit_boundary.end(1)]
        
        # 3. "Approx.", "About" 등 접두어 제거 (값 추출 전)
        value_prefix = ""
        prefix_match = re.match(r'^(Approx\.?|About|Approximately|Abt\.?)\s*', raw, flags=re.I)
        if prefix_match:
            value_prefix = prefix_match.group(0)
            raw = raw[len(value_prefix):]
        
        # 4. 텍스트 모델명/타입 감지 (숫자 추출 전에 체크)
        # "TCT40-MLx2 sets", "MAN-ES", "B&W" 같은 모델명은 그대로 반환
        # "Electric motor" 같은 텍스트 값도 그대로 반환
        if re.match(r'^[A-Za-z][A-Za-z0-9\-&/\s]+', raw):
            # 모델명 패턴: 영문자 시작, 숫자/문자 혼합, 하이픈 포함
            if re.search(r'[A-Za-z]+[\-][A-Za-z0-9]+', raw) or \
               re.search(r'^[A-Z]{2,}[\-\s]?[A-Z0-9]+', raw) or \
               re.search(r'x\d+\s*(set|ea|pcs)', raw, re.I):  # "x2 sets" 패턴
                # 모델명으로 판단
                # 괄호 이후 내용 제거
                if ' (' in raw:
                    raw = raw.split(' (')[0].strip()
                return raw, ""
            
            # 순수 텍스트 값 (Electric motor, Carbon / SiC 등)
            if not re.search(r'\d', raw) or re.search(r'^[A-Z][a-z]+\s+[a-z]+', raw):
                # 괄호 이후 내용 제거
                if ' (' in raw:
                    raw = raw.split(' (')[0].strip()
                return raw, ""
        
        # 4. 단위 패턴 매칭 (순서 중요 - 구체적인 것부터)
        unit_patterns = [
            # 열전도율 (가장 구체적)
            (r'([\d.,]+)\s*(W/mK|Kcal\s*/\s*mh°?C|W/m[·.]K)', 2),
            # 밀도
            (r'([\d.,()±]+)\s*(kg/m[³3]|kg/m³|g/cm[³3])', 2),
            # 유량
            (r'([\d.,]+)\s*(m[³3]/h|m³/h|m3/h|m³/hr)', 2),
            (r'([\d.,]+)\s*(l/min|L/min|lpm|LPM)', 2),
            # 출력
            (r'([\d.,]+)\s*(kW|MW|HP|kVA)', 2),
            # 회전수
            (r'([\d.,]+)\s*(rpm|RPM|r/min|rev/min)', 2),
            # 압력
            (r'([\d.,]+)\s*(bar|Bar|MPa|kPa|kg/cm2|mbar|mmWC|mmHg|psi)', 2),
            # 소비율
            (r'([\d.,]+)\s*(g/kWh)', 2),
            # 길이 (mm 우선)
            (r'([\d.,]+)\s*(mm)', 2),
            (r'([\d.,]+)\s*(cm)', 2),
            (r'([\d.,]+)\s*(m)\b', 2),
            # 온도
            (r'([\d.,\-~]+)\s*([°℃]C|℃|°C)', 2),
            # 습도
            (r'([\d.,]+)\s*(%)', 2),
            # 전기
            (r'([\d.,]+)\s*(kV)', 2),
            (r'([\d.,]+)\s*(V)\b', 2),
            (r'([\d.,]+)\s*(Hz)', 2),
            (r'([\d.,]+)\s*(A)\b', 2),
            # 무게
            (r'([\d.,]+)\s*(kg|tonnes?|ton|t)\b', 2),
            # 수량 (단독 숫자만, "x2" 같은 형태 제외)
            (r'^([\d.,]+)\s*(EA|ea|set|sets|pcs|units)', 2),
        ]
        
        for pattern, unit_group in unit_patterns:
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                value = match.group(1)
                unit = match.group(unit_group)
                
                # 괄호 내용 처리 (예: "(bow)" 제거)
                rest = raw[match.end():].strip()
                if rest.startswith('(') and ')' in rest:
                    # 괄호 내용 제거
                    pass
                
                return value, unit
        
        # 5. 단위 없이 순수 숫자만 있는 경우
        pure_num = re.match(r'^([\d.,\-~]+)$', raw)
        if pure_num:
            return pure_num.group(1), ""
        
        # 6. 텍스트 값 (Type, Material 등)
        # 너무 긴 값은 첫 부분만
        if len(raw) > 80:
            # 첫 문장 또는 첫 절만
            cut = re.split(r'[.\-]', raw)[0].strip()
            if len(cut) > 10:
                raw = cut
        
        return raw, ""
    
    def _is_valid_value(self, value: str, spec_name: str) -> bool:
        """
        값이 유효한지 검사
        
        Args:
            value: 추출된 값
            spec_name: 사양명 (대문자)
        """
        if not value:
            return False
        
        # 너무 긴 값은 무효 (200자로 완화)
        if len(value) > 200:
            return False
        
        # 섹션 번호 패턴 (예: "2.4.2 Temperature control system")
        if re.match(r'^\d+\.\d+(\.\d+)?\s+[A-Za-z]', value):
            return False
        
        # "Size/Location" 같은 헤더 패턴
        if re.match(r'^[A-Z][a-z]+/[A-Z][a-z]+$', value):
            return False
        
        # 빈 값이나 특수문자만 있는 경우
        if not re.search(r'[a-zA-Z0-9가-힣]', value):
            return False
        
        # 사양 타입별 타당성 검증
        return self._validate_value_for_spec(value, spec_name)
    
    def _validate_value_for_spec(self, value: str, spec_name: str) -> bool:
        """사양 타입에 맞는 값인지 검증"""
        value_upper = value.upper()
        
        # OUTPUT/POWER 관련: kW, MW, HP 단위가 있어야 함
        if 'OUTPUT' in spec_name or 'POWER' in spec_name:
            # dB 단위는 OUTPUT이 아님 (소음 레벨)
            if 'DB' in value_upper or 'DECIBEL' in value_upper:
                return False
            # 시간/기간 값은 OUTPUT이 아님
            if 'MONTH' in value_upper or 'YEAR' in value_upper or 'WEEK' in value_upper:
                return False
        
        # CAPACITY: 숫자 + 유량 단위가 있어야 함
        if spec_name == 'CAPACITY':
            # "Specific heat" 같은 다른 사양이 아닌지 확인
            if 'HEAT' in value_upper or 'CONDUCTIVITY' in value_upper:
                return False
        
        # DENSITY: 밀도 단위가 있어야 함
        if 'DENSITY' in spec_name:
            # "Specific heat" 같은 다른 사양이 아닌지 확인
            if 'HEAT' in value_upper or 'KCAL' in value_upper:
                return False
        
        # SPEED: rpm 또는 숫자 + 회전 관련 값이어야 함
        if 'SPEED' in spec_name:
            # 시간/기간 값은 SPEED가 아님
            if 'MINUTE' in value_upper or 'CONTINUOUS' in value_upper:
                return False
            if 'MONTH' in value_upper or 'YEAR' in value_upper:
                return False
        
        # TYPE: 텍스트 설명이어야 함 (숫자만 있으면 안됨)
        if spec_name == 'TYPE':
            # 순수 숫자는 TYPE이 아님
            if re.match(r'^[\d.,\-+%]+$', value.strip()):
                return False
        
        return True
    
    def find_value_by_keyword_search(
        self, 
        spec_name: str,
        equipment: str = ""
    ) -> Optional[Tuple[str, str, str]]:
        """키워드 기반 텍스트 검색"""
        full_text = self.get_full_text()
        
        patterns = [
            rf'{re.escape(spec_name)}\s*[:：]\s*([^\n]+)',
            rf'{re.escape(spec_name)}\s+([0-9][^\n]*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                raw_value = match.group(1).strip()
                value, unit = self._parse_value_unit(raw_value)
                if value and self._is_valid_value(value, spec_name.upper()):
                    start = max(0, match.start() - 50)
                    end = min(len(full_text), match.end() + 50)
                    chunk = full_text[start:end]
                    return (value, unit, chunk)
        
        return None
    
    def find_value_in_section(
        self,
        spec_name: str,
        section_hints: List[str],
        equipment: str = ""
    ) -> Optional[Tuple[str, str, str]]:
        """
        섹션 힌트 기반 검색
        
        용어집의 section_num 정보를 활용하여 특정 섹션 내에서 우선 검색합니다.
        
        Args:
            spec_name: 사양명
            section_hints: 섹션 검색 힌트 (예: ["2.2.1", "Main particulars"])
            equipment: 장비명
            
        Returns:
            (값, 단위, chunk) 또는 None
        """
        if not section_hints or not self.html_content:
            return None
        
        full_text = self.get_full_text()
        
        for hint in section_hints:
            if not hint:
                continue
            
            # 1. 섹션 힌트 위치 찾기
            hint_pattern = re.escape(hint)
            hint_match = re.search(hint_pattern, full_text, re.IGNORECASE)
            
            if not hint_match:
                continue
            
            # 2. 해당 섹션 영역 추출 (힌트 위치부터 다음 섹션 또는 1000자까지)
            section_start = hint_match.start()
            
            # 다음 섹션 번호 찾기 (예: 다음 "2.3" 또는 "3.")
            next_section_pattern = r'\n\s*\d+\.\d*\s+[A-Z]'
            next_match = re.search(next_section_pattern, full_text[section_start + 100:])
            
            if next_match:
                section_end = section_start + 100 + next_match.start()
            else:
                section_end = min(len(full_text), section_start + 2000)
            
            section_text = full_text[section_start:section_end]
            
            # 3. 섹션 내에서 사양명 검색
            spec_pattern = rf'{re.escape(spec_name)}\s*[:：]?\s*([^\n]+)'
            spec_match = re.search(spec_pattern, section_text, re.IGNORECASE)
            
            if spec_match:
                raw_value = spec_match.group(1).strip()
                value, unit = self._parse_value_unit(raw_value)
                
                if value and self._is_valid_value(value, spec_name.upper()):
                    # chunk 생성
                    chunk_start = max(0, spec_match.start() - 30)
                    chunk_end = min(len(section_text), spec_match.end() + 50)
                    chunk = section_text[chunk_start:chunk_end]
                    
                    return (value, unit, f"[Section: {hint}] {chunk}")
        
        return None
    
    def get_full_text(self) -> str:
        """전체 텍스트 반환"""
        return '\n'.join(self.text_chunks)
    
    def get_context_for_value(self, value: str, window: int = 100) -> str:
        """값 주변 컨텍스트 반환"""
        full_text = self.get_full_text()
        pos = full_text.find(value)
        if pos < 0:
            return ""
        
        start = max(0, pos - window)
        end = min(len(full_text), pos + len(value) + window)
        return full_text[start:end]


# =============================================================================
# 4-Stage Chunk Selection Components
# =============================================================================

@dataclass
class HTMLSection:
    """HTML 섹션 정보"""
    section_num: str           # "1", "2", "2.A", "2.2.1", etc.
    section_title: str         # "GENERAL", "TECHNICAL PARTICULARS", etc.
    section_level: int         # 1, 2, 3 (depth)
    start_pos: int            # 전체 텍스트에서의 시작 위치
    end_pos: int              # 전체 텍스트에서의 끝 위치
    content: str              # 섹션 전체 컨텐츠
    tables: List[List[List[str]]] = field(default_factory=list)  # 섹션 내 테이블들
    text_chunks: List[str] = field(default_factory=list)    # 섹션 내 텍스트 청크들


class HTMLSectionParser:
    """
    HTML 문서를 섹션 단위로 파싱

    POS 문서의 섹션 구조를 분석하여 Section 2 (TECHNICAL PARTICULARS) 우선 검색 지원
    """

    def __init__(self, html_content: str = "", file_path: str = "", chunk_parser: HTMLChunkParser = None):
        self.html_content = html_content
        self.file_path = file_path
        self.chunk_parser = chunk_parser  # 기존 HTMLChunkParser 재사용
        self.soup = None
        self.sections: List[HTMLSection] = []
        self.section_index: Dict[str, HTMLSection] = {}  # 빠른 조회용

        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                self.html_content = f.read()

        if self.html_content:
            self._parse()

    def _parse(self):
        """HTML 파싱 및 섹션 추출"""
        if not HAS_BS4:
            return

        self.soup = BeautifulSoup(self.html_content, 'html.parser')
        self._extract_sections()
        self._build_section_index()

    def _extract_sections(self):
        """
        섹션 추출

        전략:
        1. <h1>, <h2>, <h3> 태그로 섹션 헤더 식별
        2. 섹션 번호 패턴: "1.", "2.-A", "2.2.1", etc.
        3. 각 섹션의 컨텐츠 범위 결정
        """
        full_text = self.soup.get_text()

        # 섹션 헤더 패턴 매칭
        # 예: "1. GENERAL", "2.-A. TECHNICAL PARTICULARS", "2.2.1 Main particulars"
        section_pattern = re.compile(
            r'^\s*(\d+(?:\.-?[A-Z]|\.\d+)*)\s*\.?\s+([A-Z][A-Z\s]+)',
            re.MULTILINE
        )

        matches = list(section_pattern.finditer(full_text))

        for i, match in enumerate(matches):
            section_num = match.group(1).strip()
            section_title = match.group(2).strip()
            start_pos = match.start()

            # 다음 섹션 시작 또는 문서 끝까지
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(full_text)

            content = full_text[start_pos:end_pos]

            # 섹션 레벨 계산 (1, 1.A, 1.1, 1.1.1 등)
            level = section_num.count('.') + section_num.count('-')

            # 섹션 내 테이블 추출 (chunk_parser 활용)
            tables = []
            if self.chunk_parser:
                # chunk_parser의 테이블 중 이 섹션에 포함된 것만
                for table in self.chunk_parser.tables:
                    # 간단한 휴리스틱: 테이블 첫 셀이 content에 있으면 포함
                    if table and table[0] and any(cell in content for row in table[:2] for cell in row):
                        tables.append(table)

            section = HTMLSection(
                section_num=section_num,
                section_title=section_title,
                section_level=level,
                start_pos=start_pos,
                end_pos=end_pos,
                content=content,
                tables=tables,
                text_chunks=[content]  # 간단하게 전체를 하나의 청크로
            )

            self.sections.append(section)

    def _build_section_index(self):
        """섹션 인덱스 구축 (빠른 조회용)"""
        for section in self.sections:
            self.section_index[section.section_num] = section
            # 정규화된 번호도 인덱싱 ("2.-A" → "2.A")
            normalized = section.section_num.replace('.-', '.')
            if normalized != section.section_num:
                self.section_index[normalized] = section

    def get_section_by_number(self, section_num: str) -> Optional[HTMLSection]:
        """섹션 번호로 조회"""
        return self.section_index.get(section_num)

    def get_technical_sections(self) -> List[HTMLSection]:
        """
        Section 2 (TECHNICAL PARTICULARS) 및 하위 섹션 반환

        가장 중요: 대부분의 spec은 여기서 찾아야 함
        """
        technical = []
        for section in self.sections:
            # Section 2로 시작하는 모든 섹션
            if section.section_num.startswith('2'):
                technical.append(section)
        return technical

    def search_in_section(
        self,
        section_num: str,
        keywords: List[str],
        context_chars: int = 200
    ) -> List[Tuple[str, int]]:
        """
        특정 섹션 내에서 키워드 검색

        Returns:
            [(matching_text, position), ...] 리스트
        """
        section = self.get_section_by_number(section_num)
        if not section:
            return []

        matches = []
        for keyword in keywords:
            # 대소문자 무시 검색
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            for match in pattern.finditer(section.content):
                start = max(0, match.start() - context_chars // 2)
                end = min(len(section.content), match.end() + context_chars // 2)
                chunk = section.content[start:end]
                matches.append((chunk, match.start()))

        return matches


@dataclass
class ChunkCandidate:
    """Chunk 후보 정보"""
    text: str                  # Chunk 텍스트
    source: str                # 출처: "section_2a_table", "section_2_text", "keyword_search", etc.
    section_num: str = ""      # 소속 섹션 번호
    quality_score: float = 0.0  # 품질 점수 (0-1.0)
    keywords_found: List[str] = field(default_factory=list)  # 발견된 키워드들
    has_numeric: bool = False   # 숫자 포함 여부
    is_table: bool = False      # 테이블 출처 여부
    start_pos: int = 0          # 원본에서의 시작 위치 (확장용)
    end_pos: int = 0            # 원본에서의 끝 위치 (확장용)
    metadata: Dict[str, Any] = field(default_factory=dict)   # 추가 메타데이터


class ChunkCandidateGenerator:
    """
    다양한 소스에서 chunk 후보 생성

    전략:
    1. Section 2 테이블 우선 검색
    2. Hint의 section_num 활용
    3. 사양명/장비명 키워드 검색
    4. 동의어 확장 검색
    """

    def __init__(
        self,
        section_parser: HTMLSectionParser,
        chunk_parser: HTMLChunkParser,
        glossary: LightweightGlossaryIndex = None,
        logger: logging.Logger = None,
        use_dynamic_knowledge: bool = True,
        pg_knowledge_loader: PostgresKnowledgeLoader = None  # Enhanced
    ):
        self.section_parser = section_parser
        self.chunk_parser = chunk_parser
        self.glossary = glossary
        self.log = logger or logging.getLogger(__name__)

        # Dynamic knowledge components (PostgreSQL 기반)
        self.use_dynamic_knowledge = use_dynamic_knowledge
        self.pg_knowledge_loader = pg_knowledge_loader
        self.fuzzy_matcher = None
        self.unit_normalizer = None
        self.abbreviation_expander = None

        if use_dynamic_knowledge and pg_knowledge_loader:
            self._init_dynamic_components()

    def _init_dynamic_components(self):
        """동적 지식 컴포넌트 초기화 (PostgreSQL 기반)"""
        try:
            if not self.pg_knowledge_loader:
                self.log.warning("PostgresKnowledgeLoader not provided, disabling dynamic knowledge")
                self.use_dynamic_knowledge = False
                return

            # PostgresKnowledgeLoader를 공유하는 래퍼 생성
            self.fuzzy_matcher = FuzzyMatcher(self.pg_knowledge_loader)
            self.unit_normalizer = UnitNormalizer(self.pg_knowledge_loader)
            self.abbreviation_expander = AbbreviationExpander(self.pg_knowledge_loader)

            self.log.debug("Dynamic knowledge components initialized (PostgreSQL-based)")
        except Exception as e:
            self.log.warning(f"Failed to initialize dynamic components: {e}")
            self.use_dynamic_knowledge = False

    def generate_candidates(
        self,
        spec: SpecItem,
        hint: ExtractionHint = None,
        max_candidates: int = 15  # 15로 증가 (더 많은 후보 생성)
    ) -> List[ChunkCandidate]:
        """
        Spec에 대한 chunk 후보들 생성

        Returns:
            최대 max_candidates개의 후보 (중복 제거)
        """
        candidates = []
        seen_texts = set()  # 중복 방지

        # 1. Section 2 테이블에서 검색 (최우선)
        candidates.extend(
            self._search_in_technical_tables(spec, hint, seen_texts)
        )

        # 2. Hint의 section_num 활용
        if hint and hint.section_num:
            candidates.extend(
                self._search_in_hint_section(spec, hint, seen_texts)
            )

        # 3. 사양명 키워드 검색 (Section 2 우선)
        candidates.extend(
            self._keyword_search(spec, hint, seen_texts)
        )

        # 4. 동의어 확장 검색 (기존 용어집)
        if self.glossary:
            candidates.extend(
                self._synonym_search(spec, hint, seen_texts)
            )

        # === 동적 지식 기반 검색 ===
        if self.use_dynamic_knowledge:
            # 5. Fuzzy 매칭 검색 (75% 이상 유사)
            candidates.extend(
                self._fuzzy_match_search(spec, hint, seen_texts)
            )

            # 6. 약어 확장 검색
            candidates.extend(
                self._abbreviation_search(spec, hint, seen_texts)
            )

            # 7. 단위 변형 검색 (umgv_uom 있는 경우)
            if hint and hint.umgv_uom:
                candidates.extend(
                    self._unit_normalized_search(spec, hint, seen_texts)
                )

        # 8. 중복 제거 및 제한
        unique_candidates = self._deduplicate(candidates)
        return unique_candidates[:max_candidates]

    def _search_in_technical_tables(
        self,
        spec: SpecItem,
        hint: ExtractionHint,
        seen_texts: Set[str]
    ) -> List[ChunkCandidate]:
        """Section 2의 테이블에서 검색"""
        candidates = []
        technical_sections = self.section_parser.get_technical_sections()

        spec_upper = spec.spec_name.upper()
        equipment_upper = spec.equipment.upper() if spec.equipment else ""

        for section in technical_sections:
            for table in section.tables:
                # 테이블에서 spec_name 검색
                table_text = "\n".join([" | ".join(row) for row in table])

                if spec_upper in table_text.upper():
                    # 관련된 행 찾기
                    for row in table:
                        row_text = " | ".join(row)
                        if spec_upper in row_text.upper():
                            # 중복 확인
                            if row_text not in seen_texts:
                                seen_texts.add(row_text)

                                candidates.append(ChunkCandidate(
                                    text=row_text,
                                    source=f"section_{section.section_num}_table",
                                    section_num=section.section_num,
                                    is_table=True,
                                    has_numeric=bool(re.search(r'\d', row_text)),
                                    keywords_found=[spec.spec_name]
                                ))

        return candidates

    def _search_in_hint_section(
        self,
        spec: SpecItem,
        hint: ExtractionHint,
        seen_texts: Set[str]
    ) -> List[ChunkCandidate]:
        """Hint의 section_num에서 검색"""
        candidates = []

        # section_num에서 실제 번호 추출 ("2.2.1 Main particulars" → "2.2.1")
        section_match = re.match(r'^(\d+(?:\.-?[A-Z]|\.\d+)*)', hint.section_num)
        if not section_match:
            return candidates

        section_num = section_match.group(1)
        matches = self.section_parser.search_in_section(
            section_num,
            [spec.spec_name, spec.equipment or ""],
            context_chars=300
        )

        for match_text, pos in matches:
            if match_text not in seen_texts:
                seen_texts.add(match_text)

                candidates.append(ChunkCandidate(
                    text=match_text,
                    source=f"hint_section_{section_num}",
                    section_num=section_num,
                    has_numeric=bool(re.search(r'\d', match_text)),
                    keywords_found=[spec.spec_name],
                    start_pos=pos
                ))

        return candidates

    def _keyword_search(
        self,
        spec: SpecItem,
        hint: ExtractionHint,
        seen_texts: Set[str]
    ) -> List[ChunkCandidate]:
        """사양명 키워드 검색 (Section 2 우선)"""
        candidates = []

        # Section 2에서 먼저 검색
        technical_sections = self.section_parser.get_technical_sections()
        for section in technical_sections:
            matches = self.section_parser.search_in_section(
                section.section_num,
                [spec.spec_name],
                context_chars=250
            )

            for match_text, pos in matches[:3]:  # 최대 3개
                if match_text not in seen_texts:
                    seen_texts.add(match_text)

                    candidates.append(ChunkCandidate(
                        text=match_text,
                        source=f"keyword_section_{section.section_num}",
                        section_num=section.section_num,
                        has_numeric=bool(re.search(r'\d', match_text)),
                        keywords_found=[spec.spec_name],
                        start_pos=pos
                    ))

        return candidates

    def _synonym_search(
        self,
        spec: SpecItem,
        hint: ExtractionHint,
        seen_texts: Set[str]
    ) -> List[ChunkCandidate]:
        """동의어 확장 검색"""
        candidates = []

        # 동의어 가져오기
        synonyms = []
        if hint and hint.pos_umgv_desc:
            synonyms.append(hint.pos_umgv_desc)

        if self.glossary and hasattr(self.glossary, 'get_synonyms'):
            synonyms.extend(self.glossary.get_synonyms(spec.spec_name))

        # 동의어로 검색 (Section 2만)
        for synonym in synonyms[:3]:  # 최대 3개 동의어
            if not synonym:
                continue

            technical_sections = self.section_parser.get_technical_sections()
            for section in technical_sections:
                matches = self.section_parser.search_in_section(
                    section.section_num,
                    [synonym],
                    context_chars=250
                )

                for match_text, pos in matches[:2]:  # 동의어당 최대 2개
                    if match_text not in seen_texts:
                        seen_texts.add(match_text)

                        candidates.append(ChunkCandidate(
                            text=match_text,
                            source=f"synonym_{section.section_num}",
                            section_num=section.section_num,
                            has_numeric=bool(re.search(r'\d', match_text)),
                            keywords_found=[synonym],
                            metadata={'synonym': synonym}
                        ))

        return candidates

    def _deduplicate(self, candidates: List[ChunkCandidate]) -> List[ChunkCandidate]:
        """중복 제거 (텍스트 기준)"""
        seen = set()
        unique = []

        for cand in candidates:
            # 텍스트의 처음 100자로 중복 판단
            key = cand.text[:100]
            if key not in seen:
                seen.add(key)
                unique.append(cand)

        return unique

    # === Enhanced Search Methods ===

    def _fuzzy_match_search(
        self,
        spec: SpecItem,
        hint: ExtractionHint,
        seen_texts: Set[str]
    ) -> List[ChunkCandidate]:
        """
        Fuzzy matching 기반 검색 (75% 이상 유사도)

        용어집의 동의어 캐시를 활용하여 유사한 용어 검색
        """
        if not self.fuzzy_matcher:
            return []

        candidates = []
        spec_name = spec.spec_name

        # 1. 동의어 가져오기
        synonyms = self.fuzzy_matcher.get_synonyms(spec_name)

        # 2. 각 동의어로 검색
        for synonym in synonyms[:5]:  # 최대 5개
            if not synonym:
                continue

            # Section 2에서 검색
            technical_sections = self.section_parser.get_technical_sections()
            for section in technical_sections:
                matches = self.section_parser.search_in_section(
                    section.section_num,
                    [synonym],
                    context_chars=250
                )

                for match_text, pos in matches[:2]:  # 각 동의어당 최대 2개
                    if match_text not in seen_texts:
                        seen_texts.add(match_text)

                        # 유사도 계산
                        similarity = self.fuzzy_matcher.ratio(spec_name, synonym)

                        candidates.append(ChunkCandidate(
                            text=match_text,
                            source=f"fuzzy_match_{similarity:.2f}",
                            section_num=section.section_num,
                            has_numeric=bool(re.search(r'\d', match_text)),
                            keywords_found=[synonym],
                            start_pos=pos,
                            metadata={'fuzzy_score': similarity, 'synonym': synonym}
                        ))

        return candidates

    def _abbreviation_search(
        self,
        spec: SpecItem,
        hint: ExtractionHint,
        seen_texts: Set[str]
    ) -> List[ChunkCandidate]:
        """
        약어 확장 검색

        예: "M/E OUTPUT" → "Main Engine OUTPUT", "Marine Engine OUTPUT"
        """
        if not self.abbreviation_expander:
            return []

        candidates = []
        spec_name = spec.spec_name

        # 1. 약어 확장
        expansions = self.abbreviation_expander.expand(spec_name)

        # 2. 확장된 형태로 검색
        for expansion in expansions:
            if expansion == spec_name:  # 원본은 이미 검색했으므로 스킵
                continue

            # Section 2에서 검색
            technical_sections = self.section_parser.get_technical_sections()
            for section in technical_sections:
                matches = self.section_parser.search_in_section(
                    section.section_num,
                    [expansion],
                    context_chars=250
                )

                for match_text, pos in matches[:2]:  # 각 확장당 최대 2개
                    if match_text not in seen_texts:
                        seen_texts.add(match_text)

                        candidates.append(ChunkCandidate(
                            text=match_text,
                            source="abbreviation_expansion",
                            section_num=section.section_num,
                            has_numeric=bool(re.search(r'\d', match_text)),
                            keywords_found=[expansion],
                            start_pos=pos,
                            metadata={'expanded_from': spec_name, 'expansion': expansion}
                        ))

        return candidates

    def _unit_normalized_search(
        self,
        spec: SpecItem,
        hint: ExtractionHint,
        seen_texts: Set[str]
    ) -> List[ChunkCandidate]:
        """
        단위 변형 검색

        예: umgv_uom="°C" → "OC", "oc", "degC" 등으로도 검색
        """
        if not self.unit_normalizer or not hint.umgv_uom:
            return []

        candidates = []

        # 1. 단위 변형들 가져오기
        unit_variants = self.unit_normalizer.get_variants(hint.umgv_uom)

        # 2. 각 변형으로 검색
        for variant in unit_variants[:5]:  # 최대 5개 변형
            if not variant:
                continue

            # Section 2에서 단위가 포함된 텍스트 검색
            technical_sections = self.section_parser.get_technical_sections()
            for section in technical_sections:
                # 사양명 + 단위 조합으로 검색
                search_terms = [
                    f"{spec.spec_name}",  # 일단 사양명만
                ]

                matches = self.section_parser.search_in_section(
                    section.section_num,
                    search_terms,
                    context_chars=300
                )

                # 매칭된 텍스트에 단위 변형이 포함되어 있는지 확인
                for match_text, pos in matches[:3]:
                    if variant.lower() in match_text.lower():
                        if match_text not in seen_texts:
                            seen_texts.add(match_text)

                            candidates.append(ChunkCandidate(
                                text=match_text,
                                source="unit_variant_match",
                                section_num=section.section_num,
                                has_numeric=bool(re.search(r'\d', match_text)),
                                keywords_found=[spec.spec_name, variant],
                                start_pos=pos,
                                metadata={'unit_variant': variant, 'standard_unit': hint.umgv_uom}
                            ))

        return candidates


class ChunkQualityScorer:
    """
    Chunk 후보의 품질 평가

    평가 기준:
    1. 길이 적정성 (100-3000 chars)
    2. 키워드 존재
    3. 숫자 패턴
    4. 테이블 구조
    5. 섹션 관련성
    """

    def __init__(
        self,
        glossary: LightweightGlossaryIndex = None,
        logger: logging.Logger = None
    ):
        self.glossary = glossary
        self.log = logger or logging.getLogger(__name__)

    def score_candidate(
        self,
        candidate: ChunkCandidate,
        spec: SpecItem,
        hint: ExtractionHint = None
    ) -> float:
        """
        후보 chunk의 품질 점수 계산

        Returns:
            0.0-1.0 사이의 점수
        """
        score = 0.0

        # 1. 길이 점수 (최대 0.2)
        score += self._score_length(candidate)

        # 2. 키워드 점수 (최대 0.3)
        score += self._score_keywords(candidate, spec, hint)

        # 3. 숫자 패턴 점수 (최대 0.15)
        score += self._score_numeric_pattern(candidate, spec, hint)

        # 4. 구조 점수 (최대 0.15)
        score += self._score_structure(candidate, hint)

        # 5. 섹션 관련성 점수 (최대 0.2)
        score += self._score_section_relevance(candidate, hint)

        return min(1.0, score)

    def _score_length(self, candidate: ChunkCandidate) -> float:
        """길이 점수"""
        length = len(candidate.text)

        if 100 <= length <= 3000:
            return 0.2
        elif 50 <= length < 100:
            return 0.1
        elif length > 3000:
            return 0.15
        else:
            return 0.0

    def _score_keywords(
        self,
        candidate: ChunkCandidate,
        spec: SpecItem,
        hint: ExtractionHint
    ) -> float:
        """키워드 점수 (fuzzy matching 지원)"""
        score = 0.0
        text_upper = candidate.text.upper()
        spec_upper = spec.spec_name.upper()

        # MAX/MIN 키워드 감지 및 검증
        is_max_spec = any(kw in spec_upper for kw in ['MAX', 'MAXIMUM', 'UPPER', 'HIGH'])
        is_min_spec = any(kw in spec_upper for kw in ['MIN', 'MINIMUM', 'LOWER', 'LOW'])

        # Chunk에 반대 키워드가 있으면 강력한 페널티
        if is_max_spec:
            # MAX 사양인데 chunk에 MINIMUM/MIN 키워드가 있으면 페널티
            if re.search(r'\b(MINIMUM|MIN\.?)\b', text_upper):
                return -0.5  # 강력한 페널티
        elif is_min_spec:
            # MIN 사양인데 chunk에 MAXIMUM/MAX 키워드가 있으면 페널티
            if re.search(r'\b(MAXIMUM|MAX\.?)\b', text_upper):
                return -0.5  # 강력한 페널티

        # Exact match
        if spec_upper in text_upper:
            score += 0.15

            # 장비명도 함께 있으면 보너스
            if spec.equipment and spec.equipment.upper() in text_upper:
                score += 0.1
        else:
            # Fuzzy matching - 약어 및 부분 단어 매칭
            # 사양명을 단어로 분리 (예: "DIAMETER X LENGTH" → ["DIAMETER", "LENGTH"])
            spec_words = [w for w in re.findall(r'[A-Z]{3,}', spec_upper) if len(w) >= 3]

            if spec_words:
                # 약어 매칭 (예: DIAMETER → DIA, TEMPERATURE → TEMP)
                abbrev_patterns = {
                    'DIAMETER': r'\bDIA\.?',
                    'TEMPERATURE': r'\bTEMP\.?',
                    'QUANTITY': r'\bQTY\.?',
                    'MAXIMUM': r'\bMAX\.?',
                    'MINIMUM': r'\bMIN\.?',
                    'LENGTH': r'\bLEN\.?|LENGTH',
                    'PRESSURE': r'\bPRES\.?|PRESS\.?',
                }

                matched = False
                for word in spec_words:
                    # Exact word match
                    if re.search(rf'\b{word}\b', text_upper):
                        matched = True
                        break
                    # Abbreviation match
                    if word in abbrev_patterns:
                        if re.search(abbrev_patterns[word], text_upper):
                            matched = True
                            break

                if matched:
                    score += 0.10  # Fuzzy match는 exact보다 낮은 점수

                    # 여러 키워드 매칭 시 추가 점수
                    matched_count = sum(1 for w in spec_words if re.search(rf'\b{w}\b', text_upper))
                    if matched_count >= 2:
                        score += 0.05
                else:
                    # 어떤 키워드도 매칭 안 되면 매우 낮은 점수
                    return 0.02  # 완전히 0은 아니지만 매우 낮음

        # 동의어
        if hint and hint.pos_umgv_desc:
            if hint.pos_umgv_desc.upper() in text_upper:
                score += 0.05

        return score

    def _score_numeric_pattern(
        self,
        candidate: ChunkCandidate,
        spec: SpecItem,
        hint: ExtractionHint
    ) -> float:
        """숫자 패턴 점수"""
        value_format = hint.value_format if hint else ""
        is_numeric = is_numeric_spec(spec.spec_name, value_format)

        has_number = re.search(r'\d', candidate.text)

        if is_numeric:
            return 0.15 if has_number else 0.0
        else:
            return 0.1

    def _score_structure(
        self,
        candidate: ChunkCandidate,
        hint: ExtractionHint
    ) -> float:
        """구조 점수"""
        score = 0.0

        if candidate.is_table:
            score += 0.1

            # table_text hint와 일치
            if hint and hint.table_text and hint.table_text.upper() == "Y":
                score += 0.05

        return score

    def _score_section_relevance(
        self,
        candidate: ChunkCandidate,
        hint: ExtractionHint
    ) -> float:
        """섹션 관련성 점수"""
        score = 0.0
        section_num = candidate.section_num
        text_upper = candidate.text.upper()

        # GENERAL section 및 무의미한 chunk 강력 필터링
        # 이러한 패턴은 사양값이 없는 메타 정보
        if any(pattern in text_upper for pattern in [
            'GENERAL', 'REVIEWED', '[DOCUMENT EXCERPT]', 'TABLE OF CONTENTS',
            'REVISION', 'APPROVAL', 'SIGNATURE'
        ]):
            return -0.5  # 강력한 페널티

        # 너무 짧고 의미없는 chunk
        if len(candidate.text.strip()) < 20:
            return -0.3

        # Section 2 가산점
        if section_num and section_num.startswith('2'):
            score += 0.15

        # Section 1 감점
        if section_num and section_num.startswith('1'):
            score -= 0.1

        # Hint section 일치
        if hint and hint.section_num:
            hint_section = hint.section_num.split()[0]
            if section_num and section_num in hint_section:
                score += 0.05

        return max(0.0, score)


class LLMChunkSelector:
    """
    LLM 기반 최적 chunk 선택

    Top N 후보 중 LLM이 가장 관련성 높은 chunk 선택
    """

    def __init__(
        self,
        llm_client: 'UnifiedLLMClient',
        logger: logging.Logger = None
    ):
        self.llm_client = llm_client
        self.log = logger or logging.getLogger(__name__)

    def select_best_chunk(
        self,
        candidates: List[ChunkCandidate],
        spec: SpecItem,
        hint: ExtractionHint = None,
        top_k: int = 5
    ) -> Optional[ChunkCandidate]:
        """
        최적 chunk 선택

        Args:
            candidates: 후보 리스트 (이미 quality score로 정렬됨)
            spec: 추출 대상 사양
            hint: 참조 힌트
            top_k: LLM에 제시할 후보 수

        Returns:
            선택된 chunk (없으면 None)
        """
        if not candidates:
            return None

        # 후보가 1개면 그대로 반환
        if len(candidates) == 1:
            return candidates[0]

        # Top K만 선택
        top_candidates = candidates[:top_k]

        # 프롬프트 생성
        prompt = self._build_selection_prompt(top_candidates, spec, hint)

        # LLM 호출
        try:
            response_text, _, _ = self.llm_client.generate(prompt)
            selected_idx = self._parse_selection_response(response_text, len(top_candidates))

            if selected_idx is not None and 0 <= selected_idx < len(top_candidates):
                selected = top_candidates[selected_idx]
                self.log.debug(
                    "LLM selected chunk %d: source=%s, score=%.2f",
                    selected_idx, selected.source, selected.quality_score
                )
                return selected
            else:
                # 파싱 실패 시 최고 점수 후보 반환
                self.log.debug("LLM selection parse failed, using top score")
                return top_candidates[0]

        except Exception as e:
            self.log.debug("LLM selection error: %s, using top score", e)
            return top_candidates[0]

    def _build_selection_prompt(
        self,
        candidates: List[ChunkCandidate],
        spec: SpecItem,
        hint: ExtractionHint
    ) -> str:
        """선택 프롬프트 생성"""

        # 후보 목록 구성
        candidate_list = []
        for idx, cand in enumerate(candidates):
            preview = cand.text[:200] + "..." if len(cand.text) > 200 else cand.text
            candidate_list.append(
                f"[{idx}] (Section: {cand.section_num}, Source: {cand.source})\n{preview}\n"
            )

        candidates_text = "\n".join(candidate_list)

        # Hint information
        hint_text = ""
        if hint:
            hint_parts = []
            if hint.historical_values:
                hint_parts.append(f"Historical values: {', '.join(hint.historical_values[:2])}")
            if hint.pos_umgv_desc:
                hint_parts.append(f"Alternative name: {hint.pos_umgv_desc}")
            if hint.section_num:
                hint_parts.append(f"Expected section: {hint.section_num}")

            if hint_parts:
                hint_text = "\nHints: " + ", ".join(hint_parts)

        prompt = f"""You are a technical specialist in the shipbuilding industry, expert in selecting the most relevant text chunk for specification extraction from POS documents.

**Context**: Select the chunk that most clearly contains the specification value for "{spec.spec_name}".

**Target Specification:**
- Spec Name: {spec.spec_name}
- Equipment: {spec.equipment or 'N/A'}
- Expected Unit: {spec.expected_unit or 'N/A'}{hint_text}

**Candidate Chunks:**
{candidates_text}

**Task:**
Select the index of the chunk that is MOST suitable for extracting the value of "{spec.spec_name}".

**Note**: Chunks from Section 2 (TECHNICAL PARTICULARS) are typically most accurate for machinery specifications.

**Output Format:**
Respond EXACTLY in the following format:
SELECTED: [index_number]
CONFIDENCE: [0.0-1.0]

Example:
SELECTED: 2
CONFIDENCE: 0.9"""

        return prompt

    def _parse_selection_response(
        self,
        response: str,
        num_candidates: int
    ) -> Optional[int]:
        """LLM 응답 파싱"""
        # "SELECTED: 2" 패턴 찾기
        match = re.search(r'SELECTED:\s*(\d+)', response, re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < num_candidates:
                return idx

        return None


class ChunkExpander:
    """
    짧은 chunk를 주변 컨텍스트로 확장

    전략:
    1. 길이 < 100 chars이면 확장
    2. 테이블 chunk: 전체 테이블 포함
    3. 텍스트 chunk: ±500 chars 추가
    """

    def __init__(
        self,
        section_parser: HTMLSectionParser,
        chunk_parser: HTMLChunkParser,
        logger: logging.Logger = None
    ):
        self.section_parser = section_parser
        self.chunk_parser = chunk_parser
        self.log = logger or logging.getLogger(__name__)

    def expand_if_needed(
        self,
        chunk: str,
        candidate: ChunkCandidate,
        max_size: int = 5000
    ) -> str:
        """
        필요시 chunk 확장

        Args:
            chunk: 원본 chunk 텍스트
            candidate: Chunk 후보 정보
            max_size: 최대 크기

        Returns:
            확장된 chunk (또는 원본)
        """
        # 이미 충분히 긴 경우
        if len(chunk) >= 100:
            return chunk

        self.log.debug(
            "Expanding short chunk (%d chars) from %s",
            len(chunk), candidate.source
        )

        # 섹션 컨텐츠에서 주변 확장
        section = self.section_parser.get_section_by_number(candidate.section_num)
        if section:
            # chunk가 섹션 내 어디 있는지 찾기
            pos = section.content.find(chunk)
            if pos >= 0:
                # ±500 chars 확장
                start = max(0, pos - 500)
                end = min(len(section.content), pos + len(chunk) + 500)
                expanded = section.content[start:end]

                # 크기 제한
                if len(expanded) > max_size:
                    expanded = expanded[:max_size] + "...[truncated]"

                self.log.debug("Expanded to %d chars", len(expanded))
                return expanded

        # 확장 실패 시 원본 반환
        return chunk


# =============================================================================
# Rule 기반 추출기 (개선)
# =============================================================================

class RuleBasedExtractor:
    """
    Rule 기반 사양값 추출
    
    개선사항:
    1. 용어집 동의어 매핑 활용 (umgv_desc ↔ pos_umgv_desc)
    2. 다중 사양명으로 검색 시도
    3. ReferenceHintEngine 통합 - 섹션 힌트, 과거값 참조
    """
    
    def __init__(
        self,
        glossary: LightweightGlossaryIndex = None,
        specdb: LightweightSpecDBIndex = None,
        hint_engine: ReferenceHintEngine = None,
        logger: logging.Logger = None,
        enable_enhanced_chunk_selection: bool = True
    ):
        self.glossary = glossary
        self.specdb = specdb
        self.hint_engine = hint_engine
        self.log = logger or logging.getLogger(__name__)
        self.enable_enhanced_chunk_selection = enable_enhanced_chunk_selection

        # 힌트 엔진이 없으면 생성 (lazy initialization)
        if not self.hint_engine and (glossary or specdb):
            self.hint_engine = ReferenceHintEngine(
                glossary=glossary,
                specdb=specdb
            )

        # Enhanced Chunk Selection 컴포넌트 (lazy loading)
        self.section_parser = None
        self.candidate_generator = None
        self.quality_scorer = None
        self.chunk_expander = None
        self._enhanced_components_initialized = False
    
    def _init_enhanced_components(self, parser: HTMLChunkParser):
        """
        Enhanced chunk selection 컴포넌트 초기화 (lazy loading)

        Args:
            parser: HTML 파서 (section parser 생성에 필요)
        """
        if self._enhanced_components_initialized:
            return

        try:
            # HTMLSectionParser 생성
            self.section_parser = HTMLSectionParser(
                html_content=parser.html_content,
                file_path=parser.file_path,
                chunk_parser=parser
            )

            # ChunkCandidateGenerator 생성
            self.candidate_generator = ChunkCandidateGenerator(
                section_parser=self.section_parser,
                chunk_parser=parser,
                glossary=self.glossary,
                logger=self.log
            )

            # ChunkQualityScorer 생성
            self.quality_scorer = ChunkQualityScorer(
                glossary=self.glossary,
                logger=self.log
            )

            # ChunkExpander 생성
            self.chunk_expander = ChunkExpander(
                section_parser=self.section_parser,
                chunk_parser=parser,
                logger=self.log
            )

            self._enhanced_components_initialized = True
            self.log.debug("Enhanced chunk selection components initialized")

        except Exception as e:
            self.log.warning(f"Failed to initialize enhanced components: {e}")
            self.enable_enhanced_chunk_selection = False

    def preload_hints_for_hull(self, hull: str):
        """
        특정 hull의 힌트 정보를 배치 로드 (효율성 최적화)

        파일 처리 시작 전 호출하면 이후 개별 추출이 빨라집니다.

        Args:
            hull: 호선 번호
        """
        if self.hint_engine:
            self.hint_engine.preload_for_hull(hull)
    
    def _extract_with_enhanced_selection(
        self,
        parser: HTMLChunkParser,
        spec: SpecItem,
        hint: ExtractionHint = None
    ) -> Optional[ExtractionResult]:
        """
        Enhanced 4-stage chunk selection으로 추출 시도

        4 Stages:
        1. HTMLSectionParser: 섹션 구조 파싱
        2. ChunkCandidateGenerator: 다양한 소스에서 후보 생성
        3. ChunkQualityScorer: 각 후보 품질 평가
        4. ChunkExpander: 짧은 chunk 확장

        Returns:
            ExtractionResult 또는 None (실패 시)
        """
        try:
            # 컴포넌트 초기화 (lazy)
            self._init_enhanced_components(parser)

            if not self._enhanced_components_initialized:
                return None

            # Stage 1-2: 후보 생성
            candidates = self.candidate_generator.generate_candidates(spec, hint, max_candidates=10)

            if not candidates:
                self.log.debug(f"No chunk candidates for spec={spec.spec_name}")
                return None

            # Stage 3: 품질 평가
            for candidate in candidates:
                candidate.quality_score = self.quality_scorer.score_candidate(
                    candidate, spec, hint
                )

            # 점수순 정렬
            candidates.sort(key=lambda c: c.quality_score, reverse=True)

            # 최고 점수 chunk 선택
            best_candidate = candidates[0]

            self.log.debug(
                f"Best chunk: source={best_candidate.source}, "
                f"section={best_candidate.section_num}, score={best_candidate.quality_score:.2f}"
            )

            # 점수가 너무 낮으면 실패
            if best_candidate.quality_score < 0.3:
                self.log.debug(f"Quality score too low: {best_candidate.quality_score:.2f}")
                return None

            # Stage 4: 필요시 확장
            chunk_text = self.chunk_expander.expand_if_needed(
                best_candidate.text, best_candidate, max_size=3000
            )

            # 값 추출 시도 (chunk에서 직접 추출)
            value, unit = self._extract_value_from_chunk(chunk_text, spec, hint)

            if value:
                confidence = best_candidate.quality_score * 0.9  # quality score 기반
                # chunk에서 원본 사양명 추출
                original_spec_name = self._extract_original_spec_name_from_chunk(
                    chunk_text, spec, hint
                )
                return ExtractionResult(
                    spec_item=spec,
                    value=value,
                    unit=unit or spec.expected_unit,
                    chunk=chunk_text[:500],  # 500자 제한
                    method="RULE_ENHANCED_CHUNK",
                    confidence=confidence,
                    reference_source=f"enhanced:{best_candidate.source}",
                    original_spec_name=original_spec_name,
                    original_unit=unit  # 원본 단위 보존
                )

            return None

        except Exception as e:
            self.log.warning(f"Enhanced chunk selection error: {e}")
            return None

    def _extract_value_from_chunk(
        self,
        chunk: str,
        spec: SpecItem,
        hint: ExtractionHint = None
    ) -> Tuple[str, str]:
        """
        Chunk에서 값 추출 (간단한 패턴 매칭)

        Returns:
            (value, unit) 튜플
        """
        # 사양명 뒤의 값 찾기
        spec_upper = spec.spec_name.upper()
        chunk_upper = chunk.upper()

        # 패턴 1: "SPEC_NAME : VALUE"
        pattern = rf'{re.escape(spec_upper)}\s*[:：]\s*([^\n\|]+)'
        match = re.search(pattern, chunk_upper, re.IGNORECASE)

        if match:
            raw_value = match.group(1).strip()
            # 값과 단위 분리
            value, unit = self._parse_value_unit(raw_value)
            if value:
                return value, unit

        # 패턴 2: 테이블 구조 "SPEC_NAME | VALUE"
        pattern2 = rf'{re.escape(spec_upper)}\s*\|\s*([^\n\|]+)'
        match2 = re.search(pattern2, chunk_upper, re.IGNORECASE)

        if match2:
            raw_value = match2.group(1).strip()
            value, unit = self._parse_value_unit(raw_value)
            if value:
                return value, unit

        # 패턴 3: "SPEC_NAME<whitespace>VALUE" (단위 포함 값)
        pattern3 = rf'{re.escape(spec_upper)}\s+([0-9.,\-~]+\s*[a-zA-Z°℃%/]+)'
        match3 = re.search(pattern3, chunk_upper, re.IGNORECASE)

        if match3:
            raw_value = match3.group(1).strip()
            value, unit = self._parse_value_unit(raw_value)
            if value:
                return value, unit

        return "", ""

    def _parse_value_unit(self, raw_value: str) -> Tuple[str, str]:
        """
        값과 단위 분리

        개선사항:
        - "SWL 6 tonnes" → ("6", "tonnes") - 앞의 설명 텍스트 제거
        - "One(1) set" → ("One(1) set", "") - 괄호 안 숫자 포함 텍스트 처리
        - "Maximum 19 m" → ("19", "m") - 앞의 형용사 제거

        Returns:
            (value, unit) 튜플
        """
        if not raw_value:
            return "", ""

        raw_value = raw_value.strip()

        # 괄호 안의 값 추출 (예: "(-163°C)" → "-163", "°C")
        # 단, "One(1) set" 같은 경우는 제외
        paren_only_match = re.match(r'^\(([^)]+)\)$', raw_value)
        if paren_only_match:
            raw_value = paren_only_match.group(1).strip()

        # 패턴 1: "텍스트 숫자 단위" 형태 (예: "SWL 6 tonnes", "Maximum 19 m")
        # 앞의 텍스트는 설명이므로 제거하고 숫자+단위만 추출
        # 음수 지원: -163°C 등 (단어 경계 \b 대신 음수 포함 패턴 사용)
        text_num_unit = re.search(r'(-?\d+(?:[.,]\d+)?(?:\s*[~\-]\s*-?\d+(?:[.,]\d+)?)?)\s*([a-zA-Z°℃%/³²]+(?:\s*[a-zA-Z°℃%/³²]+)*)?$', raw_value)
        if text_num_unit:
            value = text_num_unit.group(1).strip()
            unit = text_num_unit.group(2).strip() if text_num_unit.group(2) else ""
            # 단위 정리 (tonnes, ton, tons 등)
            if unit:
                unit = unit.replace('tonnes', 'ton').replace('tons', 'ton')
            return value, unit

        # 패턴 2: 숫자로 시작하는 경우 (예: "6 tonnes", "19 m", "-163°C")
        match = re.match(r'^(-?[0-9.,\-~\s]+)\s*([a-zA-Z°℃%/³²]+.*)?$', raw_value)
        if match:
            value = match.group(1).strip()
            unit = match.group(2).strip() if match.group(2) else ""
            return value, unit

        # 패턴 3: 순수 숫자
        if re.match(r'^[0-9.,\-~\s]+$', raw_value):
            return raw_value.strip(), ""

        # 패턴 4: 텍스트+숫자 혼합 - QUANTITY 등에 사용
        # "One(1) set", "Two(2) units" 등
        if re.search(r'\d', raw_value):
            # 단위가 명시적으로 뒤에 있는 경우 분리
            unit_match = re.match(r'^(.+?)\s+(bar|mm|kg|kW|RPM|Hz|V|A|°C|degrees?|%|MPa|kPa|m³/h|L/min|set|sets|unit|units|piece|pieces).*$', raw_value, re.IGNORECASE)
            if unit_match:
                return unit_match.group(1).strip(), unit_match.group(2).strip()

            # 숫자 포함 텍스트를 그대로 값으로 반환
            return raw_value, ""

        # 패턴 5: 순수 텍스트 값 (숫자 없음, 예: "SUS316 BODY", "STAINLESS STEEL")
        # 단, 너무 긴 텍스트는 제외 (50자 이하만)
        if len(raw_value) <= 50:
            return raw_value, ""

        return "", ""

    def _get_spec_name_variants(self, spec_name: str, hint: ExtractionHint = None) -> List[str]:
        """
        사양명의 모든 변형 반환 (동의어 포함)

        용어집의 umgv_desc ↔ pos_umgv_desc 매핑을 활용하여
        검색할 사양명 후보를 확장합니다.

        Args:
            spec_name: 원본 사양명
            hint: 추출 힌트 (있으면 pos_umgv_desc 추가)

        Returns:
            검색할 사양명 리스트 (원본 + 동의어들)
        """
        variants = [spec_name]

        # 힌트에서 pos_umgv_desc 추가
        if hint and hint.pos_umgv_desc and hint.pos_umgv_desc not in variants:
            variants.append(hint.pos_umgv_desc)

        # 용어집에서 동의어 조회
        if self.glossary and hasattr(self.glossary, 'get_synonyms'):
            synonyms = self.glossary.get_synonyms(spec_name)
            for syn in synonyms:
                if syn and syn not in variants:
                    variants.append(syn)

        return variants

    def _expand_spec_keywords(self, spec_name: str) -> List[str]:
        """
        사양명을 키워드로 분해하여 검색 범위 확장

        예시:
        - "CAPACITY(SWL)" → ["CAPACITY(SWL)", "CAPACITY", "SWL"]
        - "MAX. WORKING RADIUS" → ["MAX. WORKING RADIUS", "MAXIMUM", "WORKING RADIUS", "RADIUS"]
        - "M/E OUTPUT" → ["M/E OUTPUT", "OUTPUT", "M/E"]

        Args:
            spec_name: 원본 사양명

        Returns:
            확장된 키워드 리스트 (우선순위 순)
        """
        keywords = [spec_name]  # 원본 우선

        # 1. 괄호 처리: "CAPACITY(SWL)" → ["CAPACITY", "SWL"]
        if '(' in spec_name:
            # 괄호 앞부분
            base = spec_name.split('(')[0].strip()
            if base and base not in keywords:
                keywords.append(base)

            # 괄호 안 내용
            paren_match = re.search(r'\(([^)]+)\)', spec_name)
            if paren_match:
                paren_content = paren_match.group(1).strip()
                if paren_content and paren_content not in keywords:
                    keywords.append(paren_content)

        # 2. 점(.) 처리: "MAX. WORKING RADIUS" → ["MAXIMUM", "WORKING RADIUS"]
        if '.' in spec_name:
            # "MAX." → "MAXIMUM" 확장
            expanded = spec_name.replace('MAX.', 'MAXIMUM').replace('MIN.', 'MINIMUM')
            if expanded != spec_name and expanded not in keywords:
                keywords.append(expanded)

            # 점 제거
            no_dot = spec_name.replace('.', '').strip()
            if no_dot and no_dot not in keywords:
                keywords.append(no_dot)

        # 3. 주요 단어 추출 (3글자 이상)
        words = re.findall(r'[A-Z][A-Z]+', spec_name)  # 대문자 연속
        for word in words:
            if len(word) >= 3 and word not in keywords:
                keywords.append(word)

        # 4. 복합어 분리: "WORKING RADIUS" 추출
        if ' ' in spec_name:
            parts = spec_name.split()
            # 뒤에서 2개 단어 조합
            if len(parts) >= 2:
                last_two = ' '.join(parts[-2:])
                if last_two not in keywords:
                    keywords.append(last_two)

        return keywords

    def _extract_by_delimiters(self, chunk: str, match_start: int, match_end: int, keyword: str) -> Optional[str]:
        """
        전략 1: 구분자 기반 추출 (|, :, /, = 등)

        예: "Working radius | Maximum | 19 m" → "Working radius | Maximum"
        """
        line_start = chunk.rfind('\n', 0, match_start) + 1
        end_pos = match_end

        while end_pos < len(chunk):
            char = chunk[end_pos]

            if char.isdigit():
                break
            if char == '\n':
                break

            if char in ['|', ':', '/', '=']:
                end_pos += 1
                while end_pos < len(chunk) and chunk[end_pos].isspace():
                    end_pos += 1

                next_word_end = end_pos
                while next_word_end < len(chunk) and (chunk[next_word_end].isalnum() or chunk[next_word_end] in ['.', "'"]):
                    next_word_end += 1

                next_word = chunk[end_pos:next_word_end].strip()
                if next_word and next_word[0].isdigit():
                    break
                if next_word and next_word[0].isalpha():
                    end_pos = next_word_end
                    continue
                break

            if char.isalnum() or char.isspace() or char in ['.', "'", '-', '(', ')']:
                end_pos += 1
            else:
                break

        context = chunk[line_start:end_pos].strip()

        # | 기반 정리
        parts = context.split('|')
        result_parts = []
        for part in parts:
            part = part.strip()
            if not part or (part and part[0].isdigit()):
                break
            result_parts.append(part)

        result = ' | '.join(result_parts) if result_parts else context
        return result if result else None

    def _extract_by_word_boundaries(self, chunk: str, match_start: int, match_end: int, keyword: str) -> Optional[str]:
        """
        전략 2: 단어 경계 기반 추출 (구분자 없어도 작동)

        예: "Hoisting capacity 6 tonnes" → "Hoisting capacity"
            "Type Electro-hydraulic" → "Type"
        """
        # 앞쪽 확장: 인접한 알파벳 단어들 포함
        start = match_start
        while start > 0:
            prev_char = chunk[start - 1]
            # 알파벳이나 공백이면 계속
            if prev_char.isalpha() or (prev_char.isspace() and start > 1 and chunk[start - 2].isalpha()):
                start -= 1
            else:
                break

        # 뒤쪽 확장: 값 패턴 전까지
        end = match_end
        while end < len(chunk):
            char = chunk[end]

            # 숫자 시작하면 중단
            if char.isdigit():
                break

            # 줄바꿈, 강한 구분자 중단
            if char in ['\n', '\t', '|', '=']:
                break

            # 동사 패턴 감지 (is, are, should, must 등)
            remaining = chunk[end:end+10].lower()
            if any(remaining.startswith(v) for v in [' is ', ' are ', ' should ', ' must ', ' will ']):
                break

            # 알파벳, 공백, 일부 특수문자 계속
            if char.isalpha() or char.isspace() or char in ['.', "'", '-', '(', ')']:
                end += 1
            else:
                break

        result = chunk[start:end].strip()

        # 정리: 앞뒤 특수문자 제거
        result = result.strip('.,;:')

        return result if len(result) >= 2 else None

    def _extract_until_value_pattern(self, chunk: str, match_start: int, match_end: int, keyword: str) -> Optional[str]:
        """
        전략 3: 값 패턴 감지까지 추출

        값 패턴:
        - 숫자
        - 숫자+단위 (6 tonnes, 19m, 1200kW)
        - 특정 키워드 (not less than, approximately, about)
        """
        # 라인 시작부터
        line_start = chunk.rfind('\n', 0, match_start) + 1

        # 값 패턴까지
        end = match_end

        # 단위 패턴 (일반적인 것들)
        units = ['mm', 'cm', 'm', 'km', 'kg', 'ton', 'tonne', 'kw', 'mw', 'rpm', 'hz', 'bar',
                 'mpa', 'kpa', 'degrees', '°c', '%', 'l/min', 'm3/h', 'm³/h', 'litre', 'liter']

        while end < len(chunk):
            char = chunk[end]

            # 숫자 발견
            if char.isdigit():
                break

            # 줄바꿈
            if char == '\n':
                break

            # 값 관련 키워드
            remaining = chunk[end:end+20].lower()
            value_keywords = ['not less than', 'not more than', 'approximately',
                             'about', 'abt.', 'min.', 'max.', 'approx']
            if any(remaining.startswith(kw) for kw in value_keywords):
                break

            end += 1

        result = chunk[line_start:end].strip()

        # : / = 뒤 부분만 (있으면)
        for delim in [':', '/', '=']:
            if delim in result:
                parts = result.split(delim)
                # 마지막 부분이 비어있지 않으면
                if len(parts) > 1 and parts[-1].strip():
                    result = parts[-1].strip()
                    break

        return result if len(result) >= 2 else None

    def _extract_by_grammar(self, chunk: str, match_start: int, match_end: int, keyword: str) -> Optional[str]:
        """
        전략 4: 문법 기반 추출 (문장 구조 분석)

        예: "The hoisting capacity is rated at 6 tonnes" → "hoisting capacity"
            "Capacity should be 6 tonnes" → "Capacity"
        """
        # 문장 내에서 주어-동사 구조 찾기
        # 매칭 키워드가 주어 부분에 있다고 가정

        line_start = chunk.rfind('\n', 0, match_start) + 1
        line_end = chunk.find('\n', match_end)
        if line_end == -1:
            line_end = len(chunk)

        sentence = chunk[line_start:line_end]

        # 동사 패턴 찾기
        verbs = [' is ', ' are ', ' was ', ' were ', ' should ', ' must ', ' will ', ' shall ', ' rated ']

        verb_pos = -1
        for verb in verbs:
            pos = sentence.lower().find(verb)
            if pos > 0:
                verb_pos = pos
                break

        if verb_pos > 0:
            # 동사 전까지가 주어
            subject = sentence[:verb_pos].strip()

            # 관사 제거 (The, A, An)
            for article in ['The ', 'the ', 'A ', 'a ', 'An ', 'an ']:
                if subject.startswith(article):
                    subject = subject[len(article):]
                    break

            return subject if len(subject) >= 2 else None

        # 동사가 없으면 단순히 : 이나 - 전
        for delim in [':', ' - ']:
            if delim in sentence:
                before_delim = sentence.split(delim)[0].strip()
                # 관사 제거
                for article in ['The ', 'the ', 'A ', 'a ', 'An ', 'an ']:
                    if before_delim.startswith(article):
                        before_delim = before_delim[len(article):]
                        break
                return before_delim if len(before_delim) >= 2 else None

        return None

    def _calculate_candidate_score(self, candidate: str, spec_name: str, keyword: str) -> float:
        """
        후보의 품질 점수 계산 (범용적 개선 버전)

        고려 요소:
        - 길이 (짧은 키워드도 적절히 평가)
        - spec_name과의 유사도
        - keyword 포함 여부 (정확한 매칭 우대)
        - 문법적 완결성
        - 계층 구조 보너스
        """
        score = 0.0

        # 짧은 키워드 감지 (3자 이하 또는 특수문자 포함)
        is_short_keyword = len(keyword) <= 4 or any(c in keyword for c in ["'", '"', '-', '.'])

        # 1. 길이 점수 (짧은 키워드에 유연한 점수)
        length = len(candidate)
        if is_short_keyword:
            # 짧은 키워드: 정확한 매칭 우대
            if 2 <= length <= 10:
                score += 1.2  # 높은 점수
            elif 10 < length <= 30:
                score += 0.9
            elif 30 < length <= 100:
                score += 0.6
            else:
                score += 0.3
        else:
            # 일반 키워드: 적절한 길이 선호
            if 5 <= length <= 50:
                score += 1.0
            elif 2 <= length < 5:
                score += 0.5
            elif 50 < length <= 100:
                score += 0.7
            else:
                score += 0.2

        # 2. 키워드 정확한 매칭 (대소문자 구분)
        if keyword in candidate:
            score += 1.0  # 정확한 매칭 높은 보너스
        elif keyword.lower() in candidate.lower():
            score += 0.6  # 대소문자 무시 매칭

        # 3. spec_name의 단어들과 유사도
        spec_words = set(re.findall(r'\w+', spec_name.upper()))
        cand_words = set(re.findall(r'\w+', candidate.upper()))

        if spec_words and cand_words:
            overlap = len(spec_words & cand_words)
            similarity = overlap / len(spec_words)
            score += similarity * 1.5  # 유사도 가중치 증가

        # 4. 구조적 특징 (계층 구조 있으면 보너스)
        if '|' in candidate:
            score += 0.5  # 계층 구조 보너스 증가

        # 5. 완결성 (문장 부호로 끝나지 않으면 보너스)
        if candidate and candidate[-1].isalnum():
            score += 0.3

        # 6. 짧은 키워드 정확 매칭 특별 보너스
        if is_short_keyword and candidate.strip().upper() == keyword.strip().upper():
            score += 2.0  # 짧은 키워드 정확 매칭 시 매우 높은 보너스

        return score

    def _select_best_candidate(self, candidates: List[Tuple[str, str]], spec_name: str, keyword: str) -> Optional[str]:
        """
        여러 후보 중 최선 선택

        Args:
            candidates: [(strategy_name, extracted_text), ...]
            spec_name: 원본 사양명
            keyword: 매칭된 키워드

        Returns:
            최선의 후보 문자열
        """
        if not candidates:
            return None

        # 검증: 길이 및 기본 조건
        valid = []
        for strategy, text in candidates:
            if text and 2 <= len(text) <= 200:
                valid.append((strategy, text))

        if not valid:
            return None

        # 점수 계산
        scored = []
        for strategy, text in valid:
            score = self._calculate_candidate_score(text, spec_name, keyword)
            scored.append((score, strategy, text))

        # 정렬 (점수 높은 순)
        scored.sort(reverse=True, key=lambda x: x[0])

        # 최고 점수 반환
        best_score, best_strategy, best_text = scored[0]

        self.log.debug(
            f"Selected candidate: '{best_text}' (strategy={best_strategy}, score={best_score:.2f})"
        )

        return best_text

    def _extract_original_spec_name_from_chunk(
        self,
        chunk: str,
        spec: SpecItem,
        hint: ExtractionHint = None
    ) -> str:
        """
        Chunk 텍스트에서 원본 사양명 추출 (다중 전략 버전)

        목표: 패턴 독립적으로 chunk에서 사양명 추출

        다중 전략:
        1. 구분자 기반 (|, :, /, =) - 계층 구조 인식
        2. 단어 경계 기반 - 구분자 없어도 작동
        3. 값 패턴 기반 - 숫자/단위 전까지
        4. 문법 기반 - 동사, 접속사 전
        5. 최소 확장 - 매칭된 키워드만

        모든 전략을 시도하고 점수 기반으로 최선 선택

        예시:
        - "Hoisting capacity | SWL 6 tonnes" → "Hoisting capacity"
        - "Working radius | Maximum | 19 m" → "Working radius | Maximum"
        - "Type Electro-hydraulic" → "Type"
        - "The hoisting capacity is rated at 6 tonnes" → "hoisting capacity"
        - "Capacity 6 tonnes" → "Capacity"

        Args:
            chunk: POS 문서 chunk 텍스트
            spec: 사양 항목
            hint: 추출 힌트

        Returns:
            원본 사양명 또는 빈 문자열
        """
        if not chunk:
            return ""

        # 1단계: 검색 키워드 생성
        variants = self._get_spec_name_variants(spec.spec_name, hint)

        # 짧은 키워드 감지 (범용적 기준)
        is_short_variant = any(
            len(v) <= 5 or any(c in v for c in ["'", '"', '-', '.'])
            for v in variants
        )

        # 키워드 확장 전략 (짧은 키워드는 확장 제한)
        if is_short_variant:
            # 짧은 키워드: 확장 최소화 (원본 변형만 사용)
            search_terms = variants
            self.log.debug(f"Short keyword detected: '{variants[0]}' - minimal expansion")
        else:
            # 일반 키워드: 확장 수행
            expanded_keywords = []
            for variant in variants:
                keywords = self._expand_spec_keywords(variant)
                for kw in keywords:
                    if kw and kw not in expanded_keywords:
                        expanded_keywords.append(kw)

            # 우선순위: 원본 변형 → 확장 키워드
            search_terms = variants + [kw for kw in expanded_keywords if kw not in variants]

        # 2단계: 모든 후보 수집
        all_candidates = []

        for keyword in search_terms:
            # 대소문자 무시 검색
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            match = pattern.search(chunk)

            if not match:
                continue

            match_start = match.start()
            match_end = match.end()

            # 짧은 키워드에 대한 최적화 전략
            if is_short_variant:
                # 전략 5: 최소 확장 우선 (매칭된 키워드만)
                minimal = chunk[match_start:match_end]
                if minimal:
                    all_candidates.append(("minimal", minimal))

                # 전략 1: 구분자 기반 (계층 구조 포착)
                c1 = self._extract_by_delimiters(chunk, match_start, match_end, keyword)
                if c1:
                    all_candidates.append(("delimiter", c1))

                # 다른 전략은 스킵 (과도한 확장 방지)
                break  # 첫 번째 매칭으로 충분
            else:
                # 일반 키워드: 모든 전략 시도
                # 전략 1: 구분자 기반
                c1 = self._extract_by_delimiters(chunk, match_start, match_end, keyword)
                if c1:
                    all_candidates.append(("delimiter", c1))

                # 전략 2: 단어 경계 기반
                c2 = self._extract_by_word_boundaries(chunk, match_start, match_end, keyword)
                if c2:
                    all_candidates.append(("word_boundary", c2))

                # 전략 3: 값 패턴 기반
                c3 = self._extract_until_value_pattern(chunk, match_start, match_end, keyword)
                if c3:
                    all_candidates.append(("value_pattern", c3))

                # 전략 4: 문법 기반
                c4 = self._extract_by_grammar(chunk, match_start, match_end, keyword)
                if c4:
                    all_candidates.append(("grammar", c4))

                # 전략 5: 최소 확장 (매칭된 키워드만)
                minimal = chunk[match_start:match_end]
                if minimal:
                    all_candidates.append(("minimal", minimal))

        # 3단계: 중복 제거 (동일한 텍스트는 한 번만)
        unique_candidates = []
        seen_texts = set()
        for strategy, text in all_candidates:
            text_normalized = text.strip().upper()
            if text_normalized not in seen_texts:
                seen_texts.add(text_normalized)
                unique_candidates.append((strategy, text))

        # 4단계: 최선의 후보 선택
        if not unique_candidates:
            return ""

        # 가장 많이 사용된 키워드로 선택 (여러 전략에서 나온 것)
        best_keyword = search_terms[0] if search_terms else spec.spec_name

        best = self._select_best_candidate(unique_candidates, spec.spec_name, best_keyword)

        return best if best else ""

    def extract(
        self,
        parser: HTMLChunkParser,
        spec: SpecItem,
        hint: ExtractionHint = None
    ) -> Optional[ExtractionResult]:
        """
        테이블에서 사양값 추출 (Enhanced chunk selection 통합)

        전략 순서:
        0. Enhanced 4-stage chunk selection
        1. [힌트] 섹션 힌트 기반 검색 (section_num 활용)
        2. 표준 사양명으로 테이블 검색
        3. 동의어로 테이블 검색 (용어집 활용)
        4. 키워드 기반 텍스트 검색
        5. [힌트] 과거 값 형식으로 재검색

        Args:
            parser: HTML 파서
            spec: 추출 대상 사양 정보
            hint: 참조 힌트 (선택적, 없으면 동적 조회)
        """
        # 힌트 조회 (없으면 동적 조회)
        if not hint and self.hint_engine and spec.hull:
            hint = self.hint_engine.get_hints(spec.hull, spec.spec_name)

        # 힌트 로깅
        log_extraction_hint(self.log, spec.spec_name, hint, source="RuleBasedExtractor")

        # Enhanced chunk selection 시도
        if self.enable_enhanced_chunk_selection:
            enhanced_result = self._extract_with_enhanced_selection(parser, spec, hint)
            if enhanced_result:
                return enhanced_result

        # 기존 로직 (fallback)
        # 사양명 변형 목록 (동의어 + 힌트의 pos_umgv_desc)
        spec_variants = self._get_spec_name_variants(spec.spec_name, hint)
        
        # 전략 0: 섹션 힌트 기반 검색 (가장 정확)
        if hint and hint.section_num:
            section_hints = self._parse_section_hints(hint.section_num)
            if section_hints:
                for variant in spec_variants:
                    result = parser.find_value_in_section(variant, section_hints, spec.equipment)
                    if result:
                        value, unit, chunk = result
                        if value:
                            confidence = self._calculate_confidence(spec, value, unit, hint)
                            # chunk에서 원본 사양명 추출
                            original_spec_name = self._extract_original_spec_name_from_chunk(
                                chunk, spec, hint
                            )
                            return ExtractionResult(
                                spec_item=spec,
                                value=value,
                                unit=unit or spec.expected_unit,
                                chunk=chunk,
                                method="RULE_SECTION_HINT",
                                confidence=confidence,
                                reference_source=f"section:{hint.section_num}",
                                original_spec_name=original_spec_name,
                                original_unit=unit
                            )

        # table_text 힌트에 따라 검색 우선순위 조정
        # Y: 테이블 우선 검색, N: 텍스트 우선 검색
        prefer_table = True
        if hint and hint.table_text:
            prefer_table = (hint.table_text.upper() == "Y")

        # 전략 1a: 테이블 검색 (table_text=Y이거나 힌트 없음)
        if prefer_table:
            for variant in spec_variants:
                result = parser.find_value_in_tables(variant, spec.equipment)

                if result:
                    value, unit, chunk = result
                    if value:
                        confidence = self._calculate_confidence(spec, value, unit, hint)
                        # table_text 힌트와 일치하면 신뢰도 상승
                        if hint and hint.table_text and hint.table_text.upper() == "Y":
                            confidence *= 1.05  # 5% 보너스
                        # 동의어로 찾은 경우 약간 낮은 신뢰도
                        if variant != spec.spec_name:
                            confidence *= 0.95
                        # chunk에서 원본 사양명 추출
                        original_spec_name = self._extract_original_spec_name_from_chunk(
                            chunk, spec, hint
                        )
                        return ExtractionResult(
                            spec_item=spec,
                            value=value,
                            unit=unit or spec.expected_unit,
                            chunk=chunk,
                            method="RULE_TABLE_DIRECT",
                            confidence=confidence,
                            reference_source=f"table_text:{hint.table_text}" if hint else "",
                            original_spec_name=original_spec_name,
                            original_unit=unit
                        )

        # 전략 1b: 텍스트 검색 (table_text=N이면 우선)
        if not prefer_table:
            for variant in spec_variants:
                result = parser.find_value_by_keyword_search(variant, spec.equipment)

                if result:
                    value, unit, chunk = result
                    if value:
                        confidence = self._calculate_confidence(spec, value, unit) * 0.9
                        # table_text 힌트와 일치하면 신뢰도 상승
                        if hint and hint.table_text and hint.table_text.upper() == "N":
                            confidence *= 1.1  # 10% 보너스
                        if variant != spec.spec_name:
                            confidence *= 0.95
                        # chunk에서 원본 사양명 추출
                        original_spec_name = self._extract_original_spec_name_from_chunk(
                            chunk, spec, hint
                        )
                        return ExtractionResult(
                            spec_item=spec,
                            value=value,
                            unit=unit or spec.expected_unit,
                            chunk=chunk,
                            method="RULE_TEXT_SEARCH",
                            confidence=confidence,
                            reference_source=f"table_text:{hint.table_text}" if hint else "",
                            original_spec_name=original_spec_name,
                            original_unit=unit
                        )

        # 전략 2: 반대 방향 검색 (테이블 우선이었으면 텍스트, 텍스트 우선이었으면 테이블)
        if prefer_table:
            # 테이블에서 못 찾았으면 텍스트 시도
            for variant in spec_variants:
                result = parser.find_value_by_keyword_search(variant, spec.equipment)

                if result:
                    value, unit, chunk = result
                    if value:
                        confidence = self._calculate_confidence(spec, value, unit) * 0.85  # 힌트 불일치로 낮은 신뢰도
                        if variant != spec.spec_name:
                            confidence *= 0.95
                        # chunk에서 원본 사양명 추출
                        original_spec_name = self._extract_original_spec_name_from_chunk(
                            chunk, spec, hint
                        )
                        return ExtractionResult(
                            spec_item=spec,
                            value=value,
                            unit=unit or spec.expected_unit,
                            chunk=chunk,
                            method="RULE_TEXT_FALLBACK",
                            confidence=confidence,
                            original_spec_name=original_spec_name,
                            original_unit=unit
                        )
        else:
            # 텍스트에서 못 찾았으면 테이블 시도
            for variant in spec_variants:
                result = parser.find_value_in_tables(variant, spec.equipment)

                if result:
                    value, unit, chunk = result
                    if value:
                        confidence = self._calculate_confidence(spec, value, unit, hint) * 0.85  # 힌트 불일치로 낮은 신뢰도
                        if variant != spec.spec_name:
                            confidence *= 0.95
                        # chunk에서 원본 사양명 추출
                        original_spec_name = self._extract_original_spec_name_from_chunk(
                            chunk, spec, hint
                        )
                        return ExtractionResult(
                            spec_item=spec,
                            value=value,
                            unit=unit or spec.expected_unit,
                            chunk=chunk,
                            method="RULE_TABLE_FALLBACK",
                            confidence=confidence,
                            original_spec_name=original_spec_name,
                            original_unit=unit
                        )
        
        return None
    
    def _parse_section_hints(self, section_num: str) -> List[str]:
        """
        섹션 번호에서 검색 힌트 추출
        
        예: "2.2.1 Main particulars" → ["2.2.1", "Main particulars"]
        """
        if not section_num:
            return []
        
        hints = []
        
        # 섹션 번호 추출 (예: "2.2.1")
        sec_match = re.match(r'^(\d+(?:\.\d+)*)', section_num)
        if sec_match:
            hints.append(sec_match.group(1))
        
        # 섹션 제목 추출 (예: "Main particulars")
        title_match = re.search(r'\d+(?:\.\d+)*\s+(.+)', section_num)
        if title_match:
            title = title_match.group(1).strip()
            # 첫 20자만 (너무 긴 제목 방지)
            if len(title) > 20:
                title = title[:20]
            hints.append(title)
        
        return hints
    
    def _calculate_confidence(
        self, 
        spec: SpecItem, 
        value: str, 
        unit: str,
        hint: ExtractionHint = None
    ) -> float:
        """
        신뢰도 계산
        
        신뢰도 결정 요소:
        1. 과거 값과 일치/유사
        2. 단위 일치
        3. 힌트의 value_format과 일치
        4. 힌트의 historical_values 참조
        """
        confidence = 0.80
        
        # 1. 힌트의 과거 값 참조 (캐시된 정보 활용)
        historical = []
        if hint and hint.historical_values:
            historical = hint.historical_values
        elif self.specdb:
            historical = self.specdb.get_historical_values(spec.hull, spec.spec_name)
        
        if historical:
            if value in historical:
                confidence = 0.95
            else:
                # 유사한 형식의 값이 있는지 확인
                for hist_val in historical:
                    if self._similar_format(value, hist_val):
                        confidence = 0.90
                        break
        
        # 2. 단위 일치 시 보너스
        if unit and spec.expected_unit:
            if unit.lower() == spec.expected_unit.lower():
                confidence = min(1.0, confidence + 0.05)
        
        # 3. value_format 검증 (힌트 활용)
        if hint and hint.value_patterns:
            value_type = self._determine_value_type(value)
            if value_type in hint.value_patterns:
                confidence = min(1.0, confidence + 0.03)
        
        return confidence
    
    def _determine_value_type(self, value: str) -> str:
        """값의 타입 결정"""
        if re.match(r'^[\d.,\-~]+$', value):
            return "숫자"
        elif re.search(r'\d+.*?(kW|rpm|bar|mm|m3|℃|°C|%)', value, re.I):
            return "숫자+단위"
        elif re.search(r'\d', value) and re.search(r'[A-Za-z]', value):
            return "혼합"
        else:
            return "텍스트"
    
    def _similar_format(self, val1: str, val2: str) -> bool:
        """값 형식 유사성 확인"""
        # 숫자 형식 비교
        num1 = re.findall(r'[\d.,]+', val1)
        num2 = re.findall(r'[\d.,]+', val2)
        
        if num1 and num2:
            # 자릿수가 비슷하면 유사
            if abs(len(num1[0]) - len(num2[0])) <= 2:
                return True
        
        return False


# =============================================================================
# UnifiedLLMClient
# =============================================================================

class UnifiedLLMClient:
    """
    Ollama LLM 클라이언트 (포트 로테이션 지원)

    개선사항:
    - 포트 로테이션으로 부하 분산
    - 스레드 안전 포트 선택
    - 토큰 추적
    """

    def __init__(self, ollama_host: str = "127.0.0.1", ollama_ports: List[int] = None,
                 model: str = "gemma3:27b", timeout: int = 180,
                 temperature: float = 0.0, max_retries: int = 3,
                 retry_sleep: float = 1.5, rate_limit: float = 0.3,
                 num_ctx: int = 32768,
                 logger: logging.Logger = None):
        self.host = ollama_host
        self.ports = ollama_ports or [11434]
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.rate_limit = rate_limit
        self.num_ctx = num_ctx
        self.logger = logger or logging.getLogger("UnifiedLLMClient")

        # 포트 로테이션
        self.port_index = 0
        self.port_lock = threading.Lock()

        # 토큰 추적
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def generate(self, prompt: str) -> Tuple[str, int, int]:
        """
        LLM 응답 생성

        Returns:
            (response_text, input_tokens, output_tokens)
        """
        if not HAS_REQUESTS:
            self.logger.warning("requests 라이브러리 없음")
            return "", 0, 0

        # 포트 로테이션 (스레드 안전)
        with self.port_lock:
            port = self.ports[self.port_index]
            self.port_index = (self.port_index + 1) % len(self.ports)

        url = f"http://{self.host}:{port}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx
            }
        }

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit)

                response = requests.post(url, json=payload, timeout=self.timeout)

                if response.status_code == 200:
                    data = response.json()
                    text = data.get('response', '')

                    # 토큰 추정 (4 chars = 1 token)
                    input_tokens = len(prompt) // 4
                    output_tokens = len(text) // 4

                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.total_calls += 1

                    return text, input_tokens, output_tokens

            except requests.exceptions.Timeout:
                self.logger.warning(f"Ollama 타임아웃 (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                self.logger.warning(f"Ollama 오류: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_sleep)

        return "", 0, 0

    def generate_with_voting(self, prompt: str, vote_k: int = 3, min_agreement: int = 2, parallel_workers: int = 2) -> Tuple[str, int, int]:
        """
        Voting 기능을 사용한 LLM 응답 생성 (병렬 처리)

        여러 번 호출하여 가장 많이 나온 응답 선택
        2개 포트를 동시 활용하여 병렬 처리 (40GB VRAM에서 gemma3:27b 2개 동시 serve 가능)

        Args:
            prompt: 프롬프트
            vote_k: 투표 횟수
            min_agreement: 최소 일치 수
            parallel_workers: 병렬 호출 수 (기본 2, gemma3:27b 2개 포트)

        Returns:
            (most_common_response, total_input_tokens, total_output_tokens)
        """
        if vote_k <= 1:
            return self.generate(prompt)

        responses = []
        total_input = 0
        total_output = 0

        # 병렬 처리로 속도 향상
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def call_generate():
            return self.generate(prompt)

        # vote_k개를 parallel_workers개씩 나눠서 처리
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(call_generate) for _ in range(vote_k)]

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    text, in_tok, out_tok = future.result(timeout=self.timeout)
                    if text:
                        responses.append(text)
                        total_input += in_tok
                        total_output += out_tok
                    else:
                        self.logger.warning(f"Voting 호출 {i}/{vote_k} 실패")
                except Exception as e:
                    self.logger.warning(f"Voting 호출 {i}/{vote_k} 오류: {e}")

        if not responses:
            return "", 0, 0

        # 가장 많이 나온 응답 선택
        from collections import Counter
        counter = Counter(responses)
        most_common_response, count = counter.most_common(1)[0]

        if count >= min_agreement:
            self.logger.info(f"Voting 성공: {count}/{vote_k}개 일치 (병렬 처리: {parallel_workers} workers)")
            return most_common_response, total_input, total_output
        else:
            self.logger.warning(f"Voting 합의 실패: 최대 {count}/{vote_k}개 일치 (최소 {min_agreement}개 필요)")
            # 합의 실패 시 가장 많이 나온 것 반환
            return most_common_response, total_input, total_output

    def get_stats(self) -> Dict:
        """통계 반환"""
        return {
            'total_calls': self.total_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens
        }


# =============================================================================
# LLM 검증 클래스 (모든 추출 결과 검증)
# =============================================================================

class LLMValidator:
    """
    LLM을 사용한 추출 결과 검증

    Rule-based 추출 결과도 LLM으로 검증하여 정확도 향상
    """

    def __init__(self, llm_client: UnifiedLLMClient, config: Config, logger: logging.Logger = None):
        self.llm_client = llm_client
        self.config = config
        self.logger = logger or logging.getLogger("LLMValidator")

    def validate_extraction(self, spec: SpecItem, extracted_value: str, extracted_unit: str,
                           html_context: str, hint: 'ExtractionHint' = None, use_voting: bool = True) -> Dict[str, Any]:
        """
        추출된 값을 LLM으로 검증

        Args:
            spec: 사양 항목
            extracted_value: 추출된 값
            extracted_unit: 추출된 단위
            html_context: HTML 컨텍스트 (테이블 등)
            hint: 추출 힌트 (related_units 포함)
            use_voting: Voting 사용 여부

        Returns:
            {
                'is_valid': bool,  # 검증 통과 여부
                'confidence': float,  # 신뢰도 (0-1)
                'llm_extracted_value': str,  # LLM이 추출한 값 (다를 경우)
                'llm_extracted_unit': str,  # LLM이 추출한 단위
                'reason': str  # 검증 이유
            }
        """
        prompt = self._build_validation_prompt(spec, extracted_value, extracted_unit, html_context, hint)

        if use_voting and self.config.vote_enabled:
            response, _, _ = self.llm_client.generate_with_voting(
                prompt,
                vote_k=self.config.vote_k,
                min_agreement=self.config.vote_min_agreement
            )
        else:
            response, _, _ = self.llm_client.generate(prompt)

        if not response:
            self.logger.warning(f"LLM 검증 실패: spec={spec.spec_name}, 응답 없음")
            return {
                'is_valid': True,  # 검증 실패 시 통과로 간주 (보수적)
                'confidence': 0.5,
                'llm_extracted_value': extracted_value,
                'llm_extracted_unit': extracted_unit,
                'reason': 'LLM 응답 없음'
            }

        return self._parse_validation_response(response, extracted_value, extracted_unit)

    def _build_validation_prompt(self, spec: SpecItem, extracted_value: str,
                                 extracted_unit: str, html_context: str, hint: 'ExtractionHint' = None) -> str:
        """검증 프롬프트 생성 (개선: 더 관대한 검증)"""
        # Related units 정보 준비
        related_units_text = "N/A"
        if hint and hint.related_units:
            related_units_text = ', '.join(hint.related_units)

        prompt = f"""You are a quality assurance specialist in the shipbuilding industry, expert in validating extracted specifications from POS (Purchase Order Specification) documents for marine equipment.

**Context**: Verify that the extracted value is correct and matches the specification in the original document. Be REASONABLE and PRACTICAL in your validation - if the value clearly exists in the context and relates to the specification, accept it.

**Specification**:
- Name: {spec.spec_name}
- Equipment: {spec.equipment if spec.equipment else 'N/A'}
- Expected Unit: {spec.expected_unit if spec.expected_unit else 'N/A'}
- Related Units (from historical data): {related_units_text}
  → These units have been used interchangeably in similar specifications and should be treated as equivalent

**Extracted Result to Validate**:
- Value: {extracted_value}
- Unit: {extracted_unit}

**Original Document Context** (This is the MOST RELEVANT section where the value was found):
```
{html_context[:2000]}
```

**Validation Guidelines** (IMPORTANT - Read carefully):
1. **Primary Check**: Does the extracted value exist in the document context?
   - Look for the EXACT value OR semantically equivalent expressions
   - For QUANTITY: "One(1) set", "1 set", "One set" are all valid
   - For numeric values: "6 tonnes", "6 ton", "6t" are equivalent

2. **Specification Matching**: Does it relate to "{spec.spec_name}"?
   - Use FLEXIBLE matching: "CAPACITY(SWL)" matches "SWL", "Capacity", "Hoisting capacity"
   - Use CONTEXT clues: if value is in the same row/section as spec name, it's valid

3. **Unit Validation**: Is the unit appropriate?
   - Accept expected_unit OR any related_units
   - Accept equivalent units: "ton"="tonnes"="tons", "m"="meter"="metres"
   - For QUANTITY: unit can be empty or "set", "sets", "unit", "units", "piece", "pieces"

4. **Format Check**: Is the value format reasonable?
   - Numeric specs should have numeric values
   - Text specs can have alphanumeric values
   - Ranges (e.g., "5~8") are valid

**CRITICAL - Common False Rejections to AVOID**:
- DO NOT reject if the value is clearly present but in a slightly different format
- DO NOT reject QUANTITY values like "One(1) set" - these are VALID quantity expressions
- DO NOT be overly strict about exact wording - use semantic understanding
- If the context is the MOST RELEVANT section (not general doc text), TRUST it

**Decision Logic**:
- If extracted value CLEARLY exists in context → is_valid = true
- If context is relevant to the spec and value is present → is_valid = true
- Only reject if value is OBVIOUSLY wrong or completely absent

**Output Format** (JSON only):
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "llm_extracted_value": "corrected value if different, or same value",
  "llm_extracted_unit": "corrected unit if different, or same unit",
  "reason": "brief explanation of validation result"
}}

Respond with JSON only:"""

        return prompt

    def _parse_validation_response(self, response: str, original_value: str,
                                   original_unit: str) -> Dict[str, Any]:
        """LLM 검증 응답 파싱"""
        try:
            # JSON 추출 시도
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return {
                    'is_valid': result.get('is_valid', True),
                    'confidence': float(result.get('confidence', 0.7)),
                    'llm_extracted_value': result.get('llm_extracted_value', original_value),
                    'llm_extracted_unit': result.get('llm_extracted_unit', original_unit),
                    'reason': result.get('reason', '')
                }
        except Exception as e:
            self.logger.warning(f"LLM 검증 응답 파싱 실패: {e}, response={response[:200]}")

        # 파싱 실패 시 기본값
        return {
            'is_valid': True,  # 보수적으로 통과
            'confidence': 0.6,
            'llm_extracted_value': original_value,
            'llm_extracted_unit': original_unit,
            'reason': '응답 파싱 실패'
        }


# =============================================================================
# LLM Fallback 클래스
# =============================================================================

class LLMFallbackExtractor:
    """
    LLM 기반 사양값 추출 (Fallback)
    
    Rule 기반 추출이 실패했을 때 LLM으로 추출 시도
    """
    
    def __init__(
        self,
        ollama_host: str = "127.0.0.1",
        ollama_ports: List[int] = None,
        model: str = "qwen2.5:32b",
        timeout: int = 180,  # 120 → 180 (timeout 빈번 발생)
        logger: logging.Logger = None,
        llm_client: 'UnifiedLLMClient' = None,
        use_voting: bool = True,
        glossary: LightweightGlossaryIndex = None,
        enable_enhanced_chunk_selection: bool = True,
        use_dynamic_knowledge: bool = True  # 동적 지식 사용
    ):
        self.host = ollama_host
        self.ports = ollama_ports or [11434]
        self.model = model
        self.timeout = timeout
        self.log = logger or logging.getLogger("LLMFallback")
        self._current_port_idx = 0
        self.llm_client = llm_client  # UnifiedLLMClient for voting support
        self.use_voting = use_voting  # Enable voting for improved accuracy
        self.glossary = glossary
        self.enable_enhanced_chunk_selection = enable_enhanced_chunk_selection

        # Enhanced Chunk Selection 컴포넌트 (lazy loading)
        self.section_parser = None
        self.candidate_generator = None
        self.quality_scorer = None
        self.llm_chunk_selector = None  # LLM 기반 chunk 선택
        self.chunk_expander = None
        self._enhanced_components_initialized = False

        # 동적 지식 컴포넌트 (PostgreSQL 기반)
        self.use_dynamic_knowledge = use_dynamic_knowledge
        self.pg_knowledge_loader = None  # POSExtractor에서 주입
        self.unit_normalizer = None
    
    def _init_enhanced_components(self, parser: HTMLChunkParser):
        """
        Enhanced chunk selection 컴포넌트 초기화 (lazy loading)

        Args:
            parser: HTML 파서
        """
        if self._enhanced_components_initialized:
            return

        try:
            # HTMLSectionParser 생성
            self.section_parser = HTMLSectionParser(
                html_content=parser.html_content,
                file_path=parser.file_path,
                chunk_parser=parser
            )

            # ChunkCandidateGenerator 생성
            self.candidate_generator = ChunkCandidateGenerator(
                section_parser=self.section_parser,
                chunk_parser=parser,
                glossary=self.glossary,
                logger=self.log
            )

            # ChunkQualityScorer 생성
            self.quality_scorer = ChunkQualityScorer(
                glossary=self.glossary,
                logger=self.log
            )

            # LLMChunkSelector 생성 (LLM 기반 선택)
            if self.llm_client:
                self.llm_chunk_selector = LLMChunkSelector(
                    llm_client=self.llm_client,
                    logger=self.log
                )

            # ChunkExpander 생성
            self.chunk_expander = ChunkExpander(
                section_parser=self.section_parser,
                chunk_parser=parser,
                logger=self.log
            )

            self._enhanced_components_initialized = True
            self.log.debug("LLM Fallback: Enhanced chunk selection components initialized")

        except Exception as e:
            self.log.warning(f"Failed to initialize enhanced components: {e}")
            self.enable_enhanced_chunk_selection = False

    def _get_ollama_url(self) -> str:
        """현재 Ollama URL 반환"""
        port = self.ports[self._current_port_idx % len(self.ports)]
        return f"http://{self.host}:{port}/api/generate"

    def _rotate_port(self):
        """다음 포트로 전환"""
        self._current_port_idx = (self._current_port_idx + 1) % len(self.ports)
    
    def extract(
        self,
        parser: HTMLChunkParser,
        spec: SpecItem,
        hint: ExtractionHint = None,
        max_chunk_chars: int = 5000
    ) -> Optional[ExtractionResult]:
        """
        LLM으로 사양값 추출 (Enhanced chunk selection 통합)

        Args:
            parser: HTML 파서
            spec: 추출할 사양 항목
            hint: 참조 힌트 (용어집/사양값DB)
            max_chunk_chars: 최대 청크 크기

        Returns:
            ExtractionResult 또는 None
        """
        if not HAS_REQUESTS:
            self.log.warning("requests 모듈 없음. LLM Fallback 비활성화")
            return None

        # 힌트 로깅
        log_extraction_hint(self.log, spec.spec_name, hint, source="LLMFallbackExtractor")

        # Enhanced chunk selection 시도
        chunk = None
        if self.enable_enhanced_chunk_selection:
            chunk = self._get_enhanced_chunk(parser, spec, hint, max_chunk_chars)

        # Fallback to legacy chunk selection
        if not chunk:
            chunk = self._get_relevant_chunk(parser, spec, max_chunk_chars, hint)

        # 최종 fallback: 전체 텍스트 사용 (약어나 다른 이유로 chunk를 찾지 못한 경우)
        if not chunk:
            self.log.warning("LLM Fallback: chunk 없음, 전체 텍스트 사용: spec=%s", spec.spec_name)
            chunk = parser.get_full_text()[:max_chunk_chars]

        if not chunk:
            self.log.debug("LLM Fallback 스킵: 텍스트 자체가 없음 (spec=%s)", spec.spec_name)
            return None
        
        # 프롬프트 생성 (힌트 정보 포함)
        prompt = self._build_prompt(spec, chunk, hint)

        # LLM 호출 (Voting 사용 가능 시 정확도 향상)
        self.log.debug("LLM 호출 시작: spec=%s, chunk_len=%d", spec.spec_name, len(chunk))

        response = None
        if self.llm_client and self.use_voting:
            # UnifiedLLMClient의 voting 기능 사용
            self.log.info("LLM Fallback with Voting: spec=%s", spec.spec_name)
            try:
                response, _, _ = self.llm_client.generate_with_voting(
                    prompt=prompt,
                    vote_k=2,  # 2번 호출 (성능 최적화, 2 ports × 1 = 2 votes)
                    min_agreement=2  # 2개 이상 일치 필요
                )
            except Exception as e:
                self.log.warning("Voting 실패, 일반 호출로 fallback: %s", e)
                response = self._call_ollama(prompt)
        else:
            # 기존 방식: 단일 호출
            response = self._call_ollama(prompt)

        if response:
            result = self._parse_llm_response(response, spec, chunk)
            if result:
                self.log.debug("LLM 응답 파싱 성공: spec=%s, value=%s",
                             spec.spec_name, result.value)
                # Voting 사용 시 method 표시
                if self.llm_client and self.use_voting:
                    result.method = "llm_fallback_voting"

                # 후처리 (범위 파싱, 단위 정규화)
                if self.use_dynamic_knowledge:
                    result = self._post_process_result(result, spec, hint)

            else:
                self.log.debug("LLM 응답 파싱 실패: spec=%s, response=%s...",
                             spec.spec_name, response[:100] if response else "")
            return result
        else:
            self.log.debug("LLM 응답 없음: spec=%s", spec.spec_name)

        return None

    def extract_batch(
        self,
        parser: HTMLChunkParser,
        specs: List[SpecItem],
        hints: Dict[str, ExtractionHint] = None,
        max_chunk_chars: int = 8000
    ) -> List[Optional[ExtractionResult]]:
        """
        배치 추출: 여러 사양값을 하나의 LLM 호출로 처리 (대량 추출 최적화)

        30만개 사양값을 2-3일 내 처리를 위한 핵심 기능
        15개 spec을 하나의 프롬프트로 전송하여 속도 향상

        Args:
            parser: HTML 파서
            specs: 추출할 사양 항목 리스트 (최대 15개 권장)
            hints: 사양별 힌트 딕셔너리 {spec_name: hint}
            max_chunk_chars: 최대 청크 크기

        Returns:
            ExtractionResult 리스트 (실패 시 None)
        """
        if not HAS_REQUESTS:
            self.log.warning("requests 모듈 없음. Batch 추출 비활성화")
            return [None] * len(specs)

        if len(specs) > 20:
            self.log.warning("Batch 크기가 너무 큼 (%d개). 20개 이하 권장", len(specs))

        # 관련 청크 추출 (전체 문서 사용)
        chunk = parser.get_full_text()[:max_chunk_chars]

        if not chunk:
            self.log.debug("Batch 추출 스킵: 청크 없음")
            return [None] * len(specs)

        # 배치 프롬프트 생성
        prompt = self._build_batch_prompt(specs, chunk, hints or {})

        # LLM 호출 (Voting 사용 시 정확도 향상)
        self.log.debug("Batch LLM 호출 시작: %d개 spec", len(specs))
        if self.llm_client and self.use_voting:
            try:
                response, _, _ = self.llm_client.generate_with_voting(
                    prompt=prompt,
                    vote_k=2,  # Batch는 voting 횟수 줄임 (속도 우선)
                    min_agreement=2
                )
            except Exception as e:
                self.log.warning("Batch voting 실패, 일반 호출로 fallback: %s", e)
                response = self._call_ollama(prompt)
        else:
            response = self._call_ollama(prompt)

        if response:
            results = self._parse_batch_response(response, specs, chunk)
            success_count = sum(1 for r in results if r and r.value)
            self.log.debug("Batch 파싱 완료: %d/%d 성공", success_count, len(specs))
            return results
        else:
            self.log.debug("Batch LLM 응답 없음")

        return [None] * len(specs)

    def _build_batch_prompt(
        self,
        specs: List[SpecItem],
        chunk: str,
        hints: Dict[str, ExtractionHint]
    ) -> str:
        """
        배치 추출용 프롬프트 생성

        15개 사양을 하나의 프롬프트로 구성
        """
        # Build specification list
        spec_list = []
        for idx, spec in enumerate(specs, 1):
            hint = hints.get(spec.spec_name)
            hint_text = ""

            if hint:
                hint_parts = []
                if hint.historical_values:
                    examples = ', '.join(hint.historical_values[:2])
                    hint_parts.append(f"examples: {examples}")
                if hint.pos_umgv_desc and hint.pos_umgv_desc != spec.spec_name:
                    hint_parts.append(f"alt_name: {hint.pos_umgv_desc}")
                if hint.related_units:
                    units_str = ', '.join(hint.related_units[:5])  # max 5 units
                    hint_parts.append(f"related_units: {units_str}")

                if hint_parts:
                    hint_text = f" ({', '.join(hint_parts)})"

            spec_list.append(
                f"{idx}. Spec: {spec.spec_name}, "
                f"Equipment: {spec.equipment or 'N/A'}, "
                f"Expected Unit: {spec.expected_unit or 'N/A'}{hint_text}"
            )

        spec_section = "\n".join(spec_list)

        prompt = f"""You are a technical specialist in the shipbuilding industry, expert in batch extraction of multiple specifications from POS (Purchase Order Specification) documents for marine equipment.

**Context**: POS documents are technical specifications for ship machinery. Extract multiple specification values efficiently while preserving EXACT notation from the original document.

## Target Specifications ({len(specs)} specs)
{spec_section}

## Document Content
```
{chunk}
```

## Task
Extract values for ALL specifications listed above from the document.

## Critical Instructions
1. **NO UNIT CONVERSION**: Extract values and units EXACTLY as written in the document
   - Example: If document says "5 inch" → unit: "inch" (do NOT convert to cm)
   - If unit differs from expected but appears in related_units, it's normal

2. If value not found, use empty string ("")

3. NEVER fabricate values not present in the document

4. Use historical examples (if provided) as format guidance

## Output Format (JSON Array)
Respond ONLY in the following JSON Array format:
```json
[
  {{"spec_index": 1, "value": "extracted_value1", "unit": "unit1", "confidence": 0.9}},
  {{"spec_index": 2, "value": "extracted_value2", "unit": "unit2", "confidence": 0.8}},
  ...
]
```

**MANDATORY**: Return EXACTLY {len(specs)} results in the JSON Array (one for each spec_index 1 to {len(specs)})."""

        return prompt

    def _parse_batch_response(
        self,
        response: str,
        specs: List[SpecItem],
        chunk: str
    ) -> List[Optional[ExtractionResult]]:
        """
        배치 응답 파싱

        JSON Array를 파싱하여 각 spec에 대한 ExtractionResult 생성
        """
        results = [None] * len(specs)

        try:
            # JSON 추출
            json_match = re.search(r'\[[\s\S]*\]', response)
            if not json_match:
                self.log.warning("Batch 응답에서 JSON Array를 찾을 수 없음")
                return results

            json_str = json_match.group(0)
            parsed = json.loads(json_str)

            if not isinstance(parsed, list):
                self.log.warning("Batch 응답이 Array가 아님")
                return results

            # 각 결과를 ExtractionResult로 변환
            for item in parsed:
                if not isinstance(item, dict):
                    continue

                spec_index = item.get('spec_index', 0) - 1  # 0-based index
                if spec_index < 0 or spec_index >= len(specs):
                    continue

                value = str(item.get('value', '')).strip()
                unit = str(item.get('unit', '')).strip()
                confidence = float(item.get('confidence', 0.0))

                if value:
                    spec = specs[spec_index]
                    results[spec_index] = ExtractionResult(
                        value=value,
                        unit=unit,
                        confidence=confidence,
                        method="llm_batch",
                        chunk=chunk[:500],
                        evidence=f"Batch extraction from {len(specs)} specs"
                    )

        except json.JSONDecodeError as e:
            self.log.warning("Batch 응답 JSON 파싱 실패: %s", e)
        except Exception as e:
            self.log.warning("Batch 응답 파싱 오류: %s", e)

        return results

    def _get_enhanced_chunk(
        self,
        parser: HTMLChunkParser,
        spec: SpecItem,
        hint: ExtractionHint,
        max_chars: int
    ) -> str:
        """
        Enhanced 4-stage chunk selection으로 chunk 추출

        4 Stages:
        1. HTMLSectionParser: 섹션 구조 파싱
        2. ChunkCandidateGenerator: 다양한 소스에서 후보 생성
        3. ChunkQualityScorer + LLMChunkSelector: 최적 chunk 선택
        4. ChunkExpander: 짧은 chunk 확장

        Returns:
            최적 chunk 텍스트 (없으면 빈 문자열)
        """
        try:
            # 컴포넌트 초기화 (lazy)
            self._init_enhanced_components(parser)

            if not self._enhanced_components_initialized:
                return ""

            # Stage 1-2: 후보 생성
            candidates = self.candidate_generator.generate_candidates(spec, hint, max_candidates=10)

            if not candidates:
                self.log.debug(f"No chunk candidates for LLM (spec={spec.spec_name})")
                return ""

            # Stage 3: 품질 평가
            for candidate in candidates:
                candidate.quality_score = self.quality_scorer.score_candidate(
                    candidate, spec, hint
                )

            # 점수순 정렬
            candidates.sort(key=lambda c: c.quality_score, reverse=True)

            # Stage 3b: LLM 기반 최적 chunk 선택 (Top 5 중에서)
            best_candidate = None
            if self.llm_chunk_selector:
                self.log.debug(f"Using LLM chunk selector for {spec.spec_name}")
                best_candidate = self.llm_chunk_selector.select_best_chunk(
                    candidates, spec, hint, top_k=5
                )

            # LLM 선택 실패 시 최고 점수 사용
            if not best_candidate:
                best_candidate = candidates[0]

            self.log.debug(
                f"Selected chunk: source={best_candidate.source}, "
                f"section={best_candidate.section_num}, score={best_candidate.quality_score:.2f}"
            )

            # 점수가 너무 낮으면 실패
            if best_candidate.quality_score < 0.2:
                self.log.debug(f"Quality score too low: {best_candidate.quality_score:.2f}")
                return ""

            # Stage 4: 필요시 확장
            chunk_text = self.chunk_expander.expand_if_needed(
                best_candidate.text, best_candidate, max_size=max_chars
            )

            return chunk_text

        except Exception as e:
            self.log.warning(f"Enhanced chunk selection error: {e}")
            return ""

    def _get_relevant_chunk(
        self,
        parser: HTMLChunkParser,
        spec: SpecItem,
        max_chars: int,
        hint: ExtractionHint = None
    ) -> str:
        """
        관련 청크 추출

        개선사항:
        1. 힌트의 section_num으로 섹션 검색
        2. pos_umgv_desc (동의어)로도 검색
        3. 부분 매칭으로 검색 범위 확대
        4. 청크가 없으면 전체 문서 일부 제공 (fallback)
        """
        chunks = []
        
        # 검색할 키워드 목록 구성 (사양명 + 동의어)
        search_keywords = [spec.spec_name.upper()]
        if hint and hint.pos_umgv_desc:
            search_keywords.append(hint.pos_umgv_desc.upper())
        
        # 사양명에서 핵심 키워드 추출 (예: "M/E LO inlet temperature" → ["LO", "INLET", "TEMPERATURE"])
        spec_words = [w for w in re.findall(r'[A-Za-z]{3,}', spec.spec_name.upper()) if len(w) >= 3]
        
        # 1. 힌트의 section_num으로 섹션 검색 (우선)
        if hint and hint.section_num:
            full_text = parser.get_full_text()
            section_match = re.search(
                re.escape(hint.section_num[:20]), full_text, re.IGNORECASE
            )
            if section_match:
                start = max(0, section_match.start() - 100)
                end = min(len(full_text), section_match.end() + 2000)
                chunks.append(f"[Section: {hint.section_num[:30]}]\n{full_text[start:end]}")
        
        # 2. 테이블 텍스트 검색 (정확 매칭)
        for table in parser.tables:
            table_text = '\n'.join([' | '.join(row) for row in table])
            table_upper = table_text.upper()
            
            # 사양명 또는 동의어 포함 여부
            for keyword in search_keywords:
                if keyword in table_upper:
                    chunks.append(table_text)
                    break
            else:
                # 장비명 포함 여부
                if spec.equipment and spec.equipment.upper() in table_upper:
                    chunks.append(table_text)
        
        # 3. 테이블 텍스트 검색 (부분 매칭 - 핵심 키워드 2개 이상)
        if not chunks and spec_words:
            for table in parser.tables:
                table_text = '\n'.join([' | '.join(row) for row in table])
                table_upper = table_text.upper()
                
                matched_words = sum(1 for w in spec_words if w in table_upper)
                if matched_words >= 2:  # 2개 이상 키워드 매칭
                    chunks.append(table_text)
        
        # 4. 텍스트 청크 검색
        for text in parser.text_chunks:
            text_upper = text.upper()
            for keyword in search_keywords:
                if keyword in text_upper:
                    chunks.append(text)
                    break
        
        # 5. Fallback: 청크가 없으면 전체 문서 일부 제공
        if not chunks:
            full_text = parser.get_full_text()
            if full_text:
                # GENERAL section은 절대 반환하지 않음
                # Section 2 (TECHNICAL PARTICULARS)를 찾아서 반환
                section2_match = re.search(r'2\.\s*TECHNICAL\s+PARTICULARS', full_text, re.IGNORECASE)
                if section2_match:
                    start = section2_match.start()
                    # Section 3 또는 문서 끝까지
                    section3_match = re.search(r'\n\s*3\.\s+[A-Z]', full_text[start:], re.IGNORECASE)
                    if section3_match:
                        end = min(start + section3_match.start(), start + 3000)
                    else:
                        end = min(start + 3000, len(full_text))
                    chunks.append(f"[Section 2 excerpt]\n{full_text[start:end]}")
                else:
                    # Section 2도 없으면 아예 반환하지 않음 (GENERAL 반환 방지)
                    self.log.debug("No Section 2 found, skipping fallback chunk")
                    return ""  # 빈 문자열 반환
        
        # 청크 결합 및 크기 제한
        combined = '\n---\n'.join(chunks)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."
        
        return combined
    
    def _build_prompt(self, spec: SpecItem, chunk: str, hint: ExtractionHint = None) -> str:
        """
        LLM 프롬프트 생성
        
        힌트 정보:
        - historical_values: 과거 값 예시
        - value_patterns: 값 형식
        - pos_umgv_desc: POS에서의 사양명 (동의어)
        """
        # Build hint section
        hint_section = ""
        if hint:
            hint_parts = []

            if hint.historical_values:
                examples = ', '.join(hint.historical_values[:3])
                hint_parts.append(f"- Historical values: {examples}")

            if hint.pos_umgv_desc and hint.pos_umgv_desc != spec.spec_name:
                hint_parts.append(f"- Alternative name in POS: {hint.pos_umgv_desc}")

            if hint.section_num:
                hint_parts.append(f"- Reference section: {hint.section_num[:50]}")

            # Add related units (data-driven)
            if hint.related_units:
                units_str = ', '.join(hint.related_units)
                hint_parts.append(f"- Related units (from historical data): {units_str}")
                hint_parts.append(f"  → These units have been used in similar specifications (same type)")

            if hint_parts:
                hint_section = "\n## Reference Hints\n" + "\n".join(hint_parts) + "\n"
        
        # MAX/MIN 키워드 감지
        spec_name_lower = spec.spec_name.lower()
        is_max_spec = any(kw in spec_name_lower for kw in ['max', 'maximum', 'upper', 'high'])
        is_min_spec = any(kw in spec_name_lower for kw in ['min', 'minimum', 'lower', 'low'])

        # Range parsing instruction
        range_instruction = ""
        if is_max_spec:
            range_instruction = """
**CRITICAL - Range Value Handling**:
- The specification name contains "MAX" or "MAXIMUM"
- If the document shows a range (e.g., "10 - 55", "-20 to 70"), extract ONLY the **upper bound (larger value)**
- Example: "10 - 55 OC" → value: "55", unit: "°C"
- Example: "-20 to 70 OC" → value: "70", unit: "°C"
"""
        elif is_min_spec:
            range_instruction = """
**CRITICAL - Range Value Handling**:
- The specification name contains "MIN" or "MINIMUM"
- If the document shows a range (e.g., "10 - 55", "-20 to 70"), extract ONLY the **lower bound (smaller value)**
- Example: "10 - 55 OC" → value: "10", unit: "°C"
- Example: "-20 to 70 OC" → value: "-20", unit: "°C"
"""

        prompt = f"""You are a technical specialist in the shipbuilding industry, expert in extracting specifications from POS (Purchase Order Specification) documents for marine equipment.

**Context**: POS documents are technical specifications for ship machinery (pumps, boilers, engines, etc.). Your task is to accurately extract specification values while preserving the EXACT notation from the original document.

## Target Specification
- Spec Name: {spec.spec_name}
- Equipment: {spec.equipment or '(Not specified)'}
- Expected Unit: {spec.expected_unit or '(Not specified)'}
{hint_section}
## Document Content
```
{chunk}
```

## Task
Extract the value for specification "{spec.spec_name}" from the above document.
{range_instruction}
## Output Format (JSON)
Respond ONLY in the following JSON format:
```json
{{
  "value": "extracted_value",
  "unit": "unit",
  "confidence": 0.0~1.0,
  "original_spec_name": "exact spec name from POS",
  "original_unit": "exact unit from POS",
  "original_equipment": "exact equipment name from POS"
}}
```

If value not found:
```json
{{"value": "", "unit": "", "confidence": 0.0, "original_spec_name": "", "original_unit": "", "original_equipment": ""}}
```

## Critical Instructions
1. **MANDATORY**: Verify the extracted value EXACTLY exists in the document. NEVER fabricate values!

2. **NO UNIT CONVERSION**: Extract values and units EXACTLY as written in the document!
   - Example: If document says "5 inch" → value: "5", unit: "inch" (do NOT convert to cm)
   - If the unit differs from expected unit but appears in "Related units", it's normal

3. Extract ONLY the value, not the specification name

4. Separate numbers and units (e.g., "70 m3/h" → value: "70", unit: "m3/h")

5. Check values in parentheses (e.g., "(34)mm" → value: "34", unit: "mm")

6. If multiple values exist, select the one most relevant to "{spec.spec_name}"

7. Return empty strings if uncertain or value not clearly present

8. Reference historical values from hints for format guidance

9. **PRESERVE ORIGINAL NOTATION**: For original_spec_name, original_unit, original_equipment:
   - Keep exact case, spacing, special characters
   - Example: "capacity" stays "capacity", "CAPACITY" stays "CAPACITY"

10. **EXTRACT ORIGINAL SPEC NAME FROM CHUNK CONTEXT** (CRITICAL):
    - Do NOT reuse the target spec name "{spec.spec_name}" directly
    - Find and extract the EXACT text from the chunk that describes this specification
    - Preserve hierarchical structure: "Working radius | Maximum | 19 m" → original_spec_name: "Working radius | Maximum"
    - Include contextual parent levels when delimiters (|, :, /) are present
    - Extract complete descriptive phrase, not just a keyword
    - Examples:
      * Chunk: "Hoisting capacity | SWL 6 tonnes" → original_spec_name: "Hoisting capacity"
      * Chunk: "Working radius | Maximum | 19 m" → original_spec_name: "Working radius | Maximum"
      * Chunk: "Type: Electro-hydraulic" → original_spec_name: "Type"
      * Chunk: "The hoisting capacity is rated at 6 tonnes" → original_spec_name: "hoisting capacity"
    - If the chunk is just a value without context, use the most descriptive text available
    - The goal is to capture the EXACT specification name as it appears in the POS document, maintaining all original formatting

## Example
Document: "Capacity: 700 m³/h, Type: VC"
Spec: "CAPACITY"
Output: {{"value": "700", "unit": "m³/h", "confidence": 0.95, "original_spec_name": "Capacity", "original_unit": "m³/h", "original_equipment": ""}}
"""
        return prompt
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Ollama API 호출"""
        url = self._get_ollama_url()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 200,
            }
        }
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                self.log.warning("Ollama 응답 오류: %d", response.status_code)
                self._rotate_port()
                return None
                
        except requests.exceptions.Timeout:
            self.log.warning("Ollama 타임아웃")
            self._rotate_port()
            return None
        except requests.exceptions.ConnectionError:
            self.log.warning("Ollama 연결 실패")
            self._rotate_port()
            return None
        except Exception as e:
            self.log.error("Ollama 호출 오류: %s", e)
            return None
    
    def _expand_spec_keywords(self, spec_name: str) -> List[str]:
        """사양명을 키워드로 분해 (RuleBasedExtractor와 동일)"""
        keywords = [spec_name]
        if '(' in spec_name:
            base = spec_name.split('(')[0].strip()
            if base and base not in keywords:
                keywords.append(base)
            paren_match = re.search(r'\(([^)]+)\)', spec_name)
            if paren_match:
                paren_content = paren_match.group(1).strip()
                if paren_content and paren_content not in keywords:
                    keywords.append(paren_content)
        if '.' in spec_name:
            expanded = spec_name.replace('MAX.', 'MAXIMUM').replace('MIN.', 'MINIMUM')
            if expanded != spec_name and expanded not in keywords:
                keywords.append(expanded)
            no_dot = spec_name.replace('.', '').strip()
            if no_dot and no_dot not in keywords:
                keywords.append(no_dot)
        words = re.findall(r'[A-Z][A-Z]+', spec_name)
        for word in words:
            if len(word) >= 3 and word not in keywords:
                keywords.append(word)
        if ' ' in spec_name:
            parts = spec_name.split()
            if len(parts) >= 2:
                last_two = ' '.join(parts[-2:])
                if last_two not in keywords:
                    keywords.append(last_two)
        return keywords

    def _extract_by_delimiters(self, chunk: str, match_start: int, match_end: int, keyword: str) -> Optional[str]:
        """전략 1: 구분자 기반 (RuleBasedExtractor와 동일)"""
        line_start = chunk.rfind('\n', 0, match_start) + 1
        end_pos = match_end
        while end_pos < len(chunk):
            char = chunk[end_pos]
            if char.isdigit() or char == '\n':
                break
            if char in ['|', ':', '/', '=']:
                end_pos += 1
                while end_pos < len(chunk) and chunk[end_pos].isspace():
                    end_pos += 1
                next_word_end = end_pos
                while next_word_end < len(chunk) and (chunk[next_word_end].isalnum() or chunk[next_word_end] in ['.', "'"]):
                    next_word_end += 1
                next_word = chunk[end_pos:next_word_end].strip()
                if next_word and next_word[0].isdigit():
                    break
                if next_word and next_word[0].isalpha():
                    end_pos = next_word_end
                    continue
                break
            if char.isalnum() or char.isspace() or char in ['.', "'", '-', '(', ')']:
                end_pos += 1
            else:
                break
        context = chunk[line_start:end_pos].strip()
        parts = context.split('|')
        result_parts = []
        for part in parts:
            part = part.strip()
            if not part or (part and part[0].isdigit()):
                break
            result_parts.append(part)
        result = ' | '.join(result_parts) if result_parts else context
        return result if result else None

    def _extract_by_word_boundaries(self, chunk: str, match_start: int, match_end: int, keyword: str) -> Optional[str]:
        """전략 2: 단어 경계 기반 (RuleBasedExtractor와 동일)"""
        start = match_start
        while start > 0:
            prev_char = chunk[start - 1]
            if prev_char.isalpha() or (prev_char.isspace() and start > 1 and chunk[start - 2].isalpha()):
                start -= 1
            else:
                break
        end = match_end
        while end < len(chunk):
            char = chunk[end]
            if char.isdigit() or char in ['\n', '\t', '|', '=']:
                break
            remaining = chunk[end:end+10].lower()
            if any(remaining.startswith(v) for v in [' is ', ' are ', ' should ', ' must ', ' will ']):
                break
            if char.isalpha() or char.isspace() or char in ['.', "'", '-', '(', ')']:
                end += 1
            else:
                break
        result = chunk[start:end].strip().strip('.,;:')
        return result if len(result) >= 2 else None

    def _extract_until_value_pattern(self, chunk: str, match_start: int, match_end: int, keyword: str) -> Optional[str]:
        """전략 3: 값 패턴 감지 (RuleBasedExtractor와 동일)"""
        line_start = chunk.rfind('\n', 0, match_start) + 1
        end = match_end
        while end < len(chunk):
            char = chunk[end]
            if char.isdigit() or char == '\n':
                break
            remaining = chunk[end:end+20].lower()
            value_keywords = ['not less than', 'not more than', 'approximately', 'about', 'abt.', 'min.', 'max.', 'approx']
            if any(remaining.startswith(kw) for kw in value_keywords):
                break
            end += 1
        result = chunk[line_start:end].strip()
        for delim in [':', '/', '=']:
            if delim in result:
                parts = result.split(delim)
                if len(parts) > 1 and parts[-1].strip():
                    result = parts[-1].strip()
                    break
        return result if len(result) >= 2 else None

    def _extract_by_grammar(self, chunk: str, match_start: int, match_end: int, keyword: str) -> Optional[str]:
        """전략 4: 문법 기반 (RuleBasedExtractor와 동일)"""
        line_start = chunk.rfind('\n', 0, match_start) + 1
        line_end = chunk.find('\n', match_end)
        if line_end == -1:
            line_end = len(chunk)
        sentence = chunk[line_start:line_end]
        verbs = [' is ', ' are ', ' was ', ' were ', ' should ', ' must ', ' will ', ' shall ', ' rated ']
        verb_pos = -1
        for verb in verbs:
            pos = sentence.lower().find(verb)
            if pos > 0:
                verb_pos = pos
                break
        if verb_pos > 0:
            subject = sentence[:verb_pos].strip()
            for article in ['The ', 'the ', 'A ', 'a ', 'An ', 'an ']:
                if subject.startswith(article):
                    subject = subject[len(article):]
                    break
            return subject if len(subject) >= 2 else None
        for delim in [':', ' - ']:
            if delim in sentence:
                before_delim = sentence.split(delim)[0].strip()
                for article in ['The ', 'the ', 'A ', 'a ', 'An ', 'an ']:
                    if before_delim.startswith(article):
                        before_delim = before_delim[len(article):]
                        break
                return before_delim if len(before_delim) >= 2 else None
        return None

    def _calculate_candidate_score(self, candidate: str, spec_name: str, keyword: str) -> float:
        """후보 점수 계산 (RuleBasedExtractor와 동일)"""
        score = 0.0
        length = len(candidate)
        if 5 <= length <= 50:
            score += 1.0
        elif 2 <= length < 5:
            score += 0.5
        elif 50 < length <= 100:
            score += 0.7
        else:
            score += 0.2
        if keyword.lower() in candidate.lower():
            score += 0.5
        spec_words = set(re.findall(r'\w+', spec_name.upper()))
        cand_words = set(re.findall(r'\w+', candidate.upper()))
        if spec_words and cand_words:
            overlap = len(spec_words & cand_words)
            similarity = overlap / len(spec_words)
            score += similarity
        if '|' in candidate:
            score += 0.3
        if candidate and candidate[-1].isalnum():
            score += 0.2
        return score

    def _select_best_candidate(self, candidates: List[Tuple[str, str]], spec_name: str, keyword: str) -> Optional[str]:
        """최선 후보 선택 (RuleBasedExtractor와 동일)"""
        if not candidates:
            return None
        valid = [(s, t) for s, t in candidates if t and 2 <= len(t) <= 200]
        if not valid:
            return None
        scored = [(self._calculate_candidate_score(t, spec_name, keyword), s, t) for s, t in valid]
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][2]

    def _improve_original_spec_name(self, chunk: str, spec: SpecItem, llm_suggested: str) -> str:
        """
        LLM 제안을 다중 전략으로 개선

        LLM이 제안한 사양명을 기반으로 chunk에서 더 나은 매칭 시도

        Issue 3 Fix: 범용적인 추출 방법 추가 (특정 패턴에만 의존하지 않음)
        """
        if not chunk or not spec.spec_name:
            return llm_suggested

        # 검색 키워드 생성
        keywords = self._expand_spec_keywords(spec.spec_name)
        if llm_suggested and llm_suggested not in keywords:
            keywords.insert(0, llm_suggested)  # LLM 제안 우선

        # 모든 후보 수집
        all_candidates = []

        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            match = pattern.search(chunk)
            if not match:
                continue

            match_start = match.start()
            match_end = match.end()

            # 기존 5가지 전략
            c1 = self._extract_by_delimiters(chunk, match_start, match_end, keyword)
            if c1:
                all_candidates.append(("delimiter", c1))

            c2 = self._extract_by_word_boundaries(chunk, match_start, match_end, keyword)
            if c2:
                all_candidates.append(("word_boundary", c2))

            c3 = self._extract_until_value_pattern(chunk, match_start, match_end, keyword)
            if c3:
                all_candidates.append(("value_pattern", c3))

            c4 = self._extract_by_grammar(chunk, match_start, match_end, keyword)
            if c4:
                all_candidates.append(("grammar", c4))

            minimal = chunk[match_start:match_end]
            if minimal:
                all_candidates.append(("minimal", minimal))

        # ===== Issue 3 Fix: 범용적인 전략 추가 =====
        # 전략 6: 문맥 기반 추출 (구분자가 없어도 작동)
        # spec_name 키워드가 포함된 라인 전체를 후보로 추가
        for line in chunk.split('\n'):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) > 200:
                continue

            # spec_name의 주요 키워드가 2개 이상 포함된 라인
            spec_words = [w.upper() for w in spec.spec_name.split() if len(w) >= 3]
            line_upper = line_stripped.upper()
            matched_count = sum(1 for w in spec_words if w in line_upper)

            if matched_count >= 2:
                # 숫자로 시작하지 않고, 적절한 길이의 텍스트
                if not line_stripped[0].isdigit() and 5 <= len(line_stripped) <= 150:
                    # 구분자가 있으면 앞부분만 추출
                    for delim in [':', '|', '/', '=']:
                        if delim in line_stripped:
                            candidate = line_stripped.split(delim)[0].strip()
                            if 3 <= len(candidate) <= 150:
                                all_candidates.append(("context_line", candidate))
                            break
                    else:
                        # 구분자가 없으면 라인 전체 (값 부분 제외)
                        # 숫자가 많으면 제외 (값이 포함된 것으로 간주)
                        digit_count = sum(1 for c in line_stripped if c.isdigit())
                        if digit_count < len(line_stripped) * 0.3:
                            all_candidates.append(("context_line", line_stripped))

        # 중복 제거
        unique_candidates = []
        seen_texts = set()
        for strategy, text in all_candidates:
            text_normalized = text.strip().upper()
            if text_normalized not in seen_texts:
                seen_texts.add(text_normalized)
                unique_candidates.append((strategy, text))

        # 최선 선택
        if unique_candidates:
            best = self._select_best_candidate(unique_candidates, spec.spec_name, keywords[0])
            if best:
                return best

        # 개선 실패시 LLM 제안 사용
        return llm_suggested

    def _parse_llm_response(
        self,
        response: str,
        spec: SpecItem,
        chunk: str
    ) -> Optional[ExtractionResult]:
        """LLM 응답 파싱 (개선 버전)"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]+\}', response)
            if not json_match:
                self.log.warning("LLM 응답에서 JSON을 찾을 수 없음: spec=%s, response=%.200s...",
                               spec.spec_name, response)
                return None

            data = json.loads(json_match.group())

            value = data.get("value", "").strip()
            unit = data.get("unit", "").strip()
            confidence = float(data.get("confidence", 0.0))

            # POS 원문 텍스트 추출 (LLM 제안)
            llm_original_spec = data.get("original_spec_name", "").strip()
            original_unit = data.get("original_unit", "").strip()
            original_equipment = data.get("original_equipment", "").strip()

            # ★ 개선: LLM 제안을 chunk에서 검증 및 확장
            original_spec_name = self._improve_original_spec_name(
                chunk, spec, llm_original_spec
            )

            if not value:
                self.log.warning("LLM이 빈 값 반환: spec=%s, JSON=%s",
                               spec.spec_name, data)
                return None

            # LLM 환각 방지 - 추출된 값이 chunk에 실제로 있는지 검증
            chunk_upper = chunk.upper()
            value_upper = value.upper()

            # 숫자 값인 경우 더 엄격하게 검증
            if re.match(r'^-?\d+\.?\d*$', value):  # 순수 숫자
                # 괄호 포함 여부 확인 (예: (34), 34)
                value_patterns = [
                    rf'\b{re.escape(value)}\b',  # 정확한 숫자
                    rf'\({re.escape(value)}\)',  # 괄호 안 숫자
                    rf'{re.escape(value)}\s*[A-Za-z]',  # 숫자 + 단위
                ]

                found = any(re.search(pattern, chunk, re.IGNORECASE) for pattern in value_patterns)

                if not found:
                    self.log.warning(
                        f"LLM 환각 감지: 추출값 '{value}'이(가) chunk에 없음. "
                        f"Chunk 내용: {chunk[:200]}..."
                    )
                    # confidence를 크게 낮춤
                    confidence *= 0.3
            elif value_upper not in chunk_upper:
                # 텍스트 값도 chunk에 있는지 확인
                self.log.warning(
                    f"LLM 환각 가능성: 추출값 '{value}'이(가) chunk에 정확히 없음"
                )
                confidence *= 0.5

            return ExtractionResult(
                spec_item=spec,
                value=value,
                unit=unit or spec.expected_unit,
                chunk=chunk[:500],
                method="LLM_FALLBACK",
                confidence=confidence * 0.9,  # LLM 결과는 약간 낮은 신뢰도
                original_spec_name=original_spec_name,
                original_unit=original_unit,
                original_equipment=original_equipment
            )

        except json.JSONDecodeError:
            self.log.warning("LLM 응답 JSON 파싱 실패")
            return None
        except Exception as e:
            self.log.error("LLM 응답 파싱 오류: %s", e)
            return None

    # === Post-processing Methods ===

    def _post_process_result(
        self,
        result: ExtractionResult,
        spec: SpecItem,
        hint: ExtractionHint = None
    ) -> ExtractionResult:
        """
        추출 결과 후처리

        1. 범위 표기 파싱 (예: "-15~60" → "-15" or "60")
        2. 단위 정규화 (예: "OC" → "°C")
        """
        if not result or not result.value:
            return result

        # 1. 범위 표기 파싱
        result = self._parse_range_notation(result, spec)

        # 2. 단위를 문서 그대로 유지 (변환 없음)
        result = self._validate_unit_as_is(result, hint)

        return result

    def _parse_range_notation(
        self,
        result: ExtractionResult,
        spec: SpecItem
    ) -> ExtractionResult:
        """
        범위 표기 파싱

        예:
        - spec_name = "minimum operating temperature", value = "-15~60" → value = "-15"
        - spec_name = "maximum operating temperature", value = "-15~60" → value = "60"
        """
        value = result.value
        spec_name_lower = spec.spec_name.lower()

        # 범위 패턴 감지
        range_patterns = [
            r'([-\d.]+)\s*~\s*([-\d.]+)',      # -15~60
            r'([-\d.]+)\s+to\s+([-\d.]+)',     # -15 to 60
            r'([-\d.]+)\s*-\s*([-\d.]+)',      # -15 - 60 (단, 음수와 구분)
            r'([-\d.]+)\s*/\s*([-\d.]+)',      # -15 / 60
        ]

        for pattern in range_patterns:
            match = re.match(pattern, value.strip())
            if match:
                lower_val = match.group(1)
                upper_val = match.group(2)

                # spec_name에 따라 선택
                if any(kw in spec_name_lower for kw in ['minimum', 'min', 'lower', 'from']):
                    result.value = lower_val
                    self.log.debug(f"Range parsed: '{value}' → '{lower_val}' (minimum)")
                elif any(kw in spec_name_lower for kw in ['maximum', 'max', 'upper', 'to']):
                    result.value = upper_val
                    self.log.debug(f"Range parsed: '{value}' → '{upper_val}' (maximum)")
                else:
                    # 기본값: 범위 그대로 유지 (어느 쪽인지 불명확)
                    self.log.debug(f"Range detected but unclear which end to use: '{value}'")

                break

        return result

    def _validate_unit_as_is(
        self,
        result: ExtractionResult,
        hint: ExtractionHint = None
    ) -> ExtractionResult:
        """
        단위를 문서 그대로 유지 (변환 없음)

        - 단위 변환을 수행하지 않음 (POS 문서에 적힌 그대로 유지)
        - LLM이 프롬프트에 제공된 단위 관련 힌트를 바탕으로 적절한 chunk를 선택했으므로
          추출된 단위를 그대로 신뢰
        """
        # 단위 변환 제거 - 원문 그대로 유지
        # hint의 related_units는 LLM 프롬프트에서 이미 활용됨

        return result


# =============================================================================
# POSExtractor 메인 클래스
# =============================================================================

class POSExtractor:
    """
    POS 사양값 추출기

    주요 특징:
    - PostgreSQL 전용 모드 (동적 지식 베이스)
    - In-memory 캐싱 (보안 네트워크 대응)
    - Enhanced chunk selection (7-stage)
    - LLM 후처리 (범위 파싱, 단위 정규화)
    - 병렬 처리 최적화
    """

    def __init__(
        self,
        glossary_path: str = "",
        specdb_path: str = "",
        config: Config = None,
    ):
        self.config = config or build_config()
        self.log = logging.getLogger("POSExtractor")

        # LLM 관련 속성 사전 초기화 (모드 초기화 전에 선언)
        self.llm_client = None
        self.llm_validator = None
        self.llm_fallback = None
        self.pg_knowledge_loader = None

        # 모드별 초기화 (glossary, specdb, pg_knowledge_loader 생성)
        if self.config.extraction_mode == "light":
            self._init_light_mode(glossary_path, specdb_path)
        elif self.config.extraction_mode == "verify":
            self._init_verify_mode(glossary_path, specdb_path)
        else:
            self._init_full_mode(glossary_path, specdb_path)

        # 공통 컴포넌트
        self.pre_checker = ImprovedPreChecker()
        self.rule_extractor = RuleBasedExtractor(self.glossary, self.specdb)

        # ValueValidator 초기화
        self.value_validator = ValueValidator(self.config)
        self.log.info("ValueValidator 초기화 완료")

        # UnifiedLLMClient 초기화 (모든 LLM 호출에 사용)
        if self.config.use_llm:
            self.llm_client = UnifiedLLMClient(
                ollama_host=self.config.ollama_host,
                ollama_ports=self.config.ollama_ports,
                model=self.config.ollama_model,
                timeout=self.config.ollama_timeout,
                temperature=self.config.llm_temperature,
                max_retries=self.config.llm_max_retries,
                retry_sleep=self.config.llm_retry_sleep_sec,
                rate_limit=self.config.llm_rate_limit_sec,
                num_ctx=self.config.ollama_num_ctx,
                logger=self.log
            )
            self.log.info("UnifiedLLMClient 초기화: %s (ports: %s)",
                         self.config.ollama_model, self.config.ollama_ports)

            # LLMValidator 초기화 (모든 추출 결과 검증)
            self.llm_validator = LLMValidator(self.llm_client, self.config, self.log)
            self.log.info("LLMValidator 초기화 완료 (모든 추출 결과 LLM 검증)")

        # LLM Fallback 초기화 (모드 초기화 후, glossary 사용 가능)
        if self.config.use_llm and self.config.enable_llm_fallback:
            self.llm_fallback = LLMFallbackExtractor(
                ollama_host=self.config.ollama_host,
                ollama_ports=self.config.ollama_ports,
                model=self.config.ollama_model,
                timeout=self.config.ollama_timeout,
                logger=self.log,
                llm_client=self.llm_client,
                use_voting=self.config.vote_enabled,
                glossary=self.glossary,  # 모드 초기화에서 생성됨
                enable_enhanced_chunk_selection=True,
                use_dynamic_knowledge=True
            )
            voting_status = "Voting 활성화" if self.config.vote_enabled else "단일 호출"
            self.log.info("LLM Fallback 초기화: %s (ports: %s, %s)",
                         self.config.ollama_model, self.config.ollama_ports, voting_status)

            # PostgresKnowledgeLoader를 LLMFallbackExtractor에 주입
            if self.pg_knowledge_loader:
                self.llm_fallback.pg_knowledge_loader = self.pg_knowledge_loader
                self.llm_fallback.unit_normalizer = UnitNormalizer(self.pg_knowledge_loader)
                self.log.info("LLMFallbackExtractor: 동적 지식 통합 완료")

        # BGE-M3 Embedding Model 초기화
        self.bge_m3_model = None
        if HAS_BGE_M3:
            try:
                self.log.info("BGE-M3 모델 로딩 시작...")

                # 로컬 모델 경로 우선 시도
                local_model_path = "/workspace/bge-m3"
                if os.path.exists(local_model_path):
                    self.log.info(f"로컬 모델 사용: {local_model_path}")
                    self.bge_m3_model = BGEM3FlagModel(local_model_path, use_fp16=True)
                else:
                    self.log.info("HuggingFace Hub에서 다운로드: BAAI/bge-m3")
                    self.bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

                self.log.info("BGE-M3 모델 로딩 완료")

                # PostgresKnowledgeLoader에 모델 주입
                if self.pg_knowledge_loader:
                    self.pg_knowledge_loader.embedding_model = self.bge_m3_model
                    self.log.info("PostgresKnowledgeLoader에 BGE-M3 모델 주입 완료")
            except Exception as e:
                self.log.warning(f"BGE-M3 모델 로딩 실패: {e}")
                self.log.warning("임베딩 검색 없이 계속 진행합니다")
                self.bge_m3_model = None
        else:
            self.log.warning("FlagEmbedding 패키지가 설치되지 않음. 임베딩 검색 비활성화")
            self.log.warning("설치 방법: pip install FlagEmbedding")

        # 통계
        self.stats = {
            'total': 0,
            'rule_success': 0,
            'llm_fallback': 0,
            'failed': 0,
            'semantic_match': 0,
            'keyword_fallback': 0,
            'no_reference': 0,
        }

        # 파서 캐시
        self._parser_cache: Dict[str, HTMLChunkParser] = {}
    
    def _init_light_mode(self, glossary_path: str, specdb_path: str):
        """
        Light 모드 초기화 (PostgreSQL 전용)

        변경사항:
        - 파일 모드 제거 (DB 모드만 지원)
        - PostgresKnowledgeLoader 초기화 (동적 지식)
        """
        self.log.info("Light 모드 초기화 시작")
        start = time.time()

        # === DB 모드 전용 (파일 모드 제거) ===
        if self.config.data_source_mode != "db":
            self.log.error("Enhanced는 DB 모드만 지원합니다.")
            raise RuntimeError(
                "파일 모드는 더 이상 지원되지 않습니다. "
                "config.data_source_mode='db'로 설정하세요."
            )

        self.log.info("데이터 소스: DB 모드 (PostgreSQL)")

        # PostgreSQL 연결
        try:
            self.pg_loader = PostgresEmbeddingLoader(self.config, self.log)
        except Exception as e:
            self.log.error(f"PostgreSQL 연결 실패: {e}")
            raise RuntimeError(f"PostgreSQL 연결 필수: {e}")

        # 용어집 로드 (pos_dict 테이블)
        glossary_df = self.pg_loader.load_glossary_from_db()
        if glossary_df.empty:
            self.log.error("용어집(pos_dict) 로드 실패")
            raise RuntimeError("용어집(pos_dict) 로드 실패: 데이터가 비어있습니다")
        self.glossary = LightweightGlossaryIndex(df=glossary_df)
        self.log.info(f"용어집 로드 완료: {len(glossary_df)}행")

        # 사양값DB 로드 (umgv_fin 테이블)
        specdb_df = self.pg_loader.load_specdb_from_db()
        if specdb_df.empty:
            self.log.error("사양값DB(umgv_fin) 로드 실패")
            raise RuntimeError("사양값DB(umgv_fin) 로드 실패: 데이터가 비어있습니다")
        self.specdb = LightweightSpecDBIndex(df=specdb_df)
        self.log.info(f"사양값DB 로드 완료: {len(specdb_df)}행")

        # === 동적 지식 로더 초기화 ===
        self.pg_knowledge_loader = PostgresKnowledgeLoader(
            conn=self.pg_loader.conn,  # 기존 연결 재사용
            logger=self.log
        )

        # 지식 로드 (1회, 수초 소요)
        if not self.pg_knowledge_loader.load_all():
            self.log.warning("동적 지식 로드 실패, 기본 기능만 사용")
        else:
            self.log.info(
                f"동적 지식 로드 완료: "
                f"동의어 {len(self.pg_knowledge_loader.synonym_reverse)}개, "
                f"단위 {len(self.pg_knowledge_loader.unit_reverse)}개, "
                f"약어 {len(self.pg_knowledge_loader.abbreviations)}개"
            )

        # ReferenceHintEngine 초기화 (용어집/사양값DB 참조)
        self.hint_engine = ReferenceHintEngine(
            glossary=self.glossary,
            specdb=self.specdb,
            pg_loader=self.pg_loader,
            logger=self.log
        )
        self._hint_engine_initialized = True
        self.log.info("ReferenceHintEngine 초기화 완료")

        # SynonymManager 초기화 (Lazy loading)
        self.synonym_manager = None
        self._synonym_manager_initialized = False
        self.log.info("SynonymManager: Lazy loading 모드")

        # SemanticMatcher (Lazy loading)
        self.semantic_matcher = None
        self._semantic_matcher_initialized = False
        self.log.info("SemanticMatcher: Lazy loading 모드")

        elapsed = time.time() - start
        self.log.info(f"Light 모드 초기화 완료: {elapsed:.2f}초")

    def _init_verify_mode(self, glossary_path: str, specdb_path: str):
        """
        Verify 모드 초기화

        사양값DB의 기존 값을 POS 문서와 대조하여 검증
        Light 모드와 동일하게 초기화하되, 검증 전용 컴포넌트 추가
        """
        self.log.info("Verify 모드 초기화 시작")
        start = time.time()

        # Light 모드와 동일한 초기화 (용어집, 사양값DB 로드)
        self._init_light_mode(glossary_path, specdb_path)

        # Verify 전용 설정
        self.verify_tolerance = VERIFY_UNIT_CONVERSION_TOLERANCE
        self.verify_confidence_threshold = VERIFY_CONFIDENCE_THRESHOLD

        elapsed = time.time() - start
        self.log.info("Verify 모드 초기화 완료: %.2f초", elapsed)

    def _init_full_mode(self, glossary_path: str, specdb_path: str):
        """
        Full 모드 초기화 (PostgreSQL 전용)

        변경사항:
        - 파일 모드 제거 (DB 모드만 지원)
        - PostgresKnowledgeLoader 초기화 (동적 지식)
        """
        self.log.info("Full 모드 초기화 시작")
        start = time.time()

        # === DB 모드 전용 (파일 모드 제거) ===
        if self.config.data_source_mode != "db":
            self.log.error("Enhanced는 DB 모드만 지원합니다.")
            raise RuntimeError(
                "파일 모드는 더 이상 지원되지 않습니다. "
                "config.data_source_mode='db'로 설정하세요."
            )

        self.log.info("데이터 소스: DB 모드 (PostgreSQL)")

        # PostgreSQL 연결
        try:
            self.pg_loader = PostgresEmbeddingLoader(self.config, self.log)
        except Exception as e:
            self.log.error(f"PostgreSQL 연결 실패: {e}")
            raise RuntimeError(f"PostgreSQL 연결 필수: {e}")

        # 용어집 로드 (pos_dict 테이블)
        glossary_df = self.pg_loader.load_glossary_from_db()
        if glossary_df.empty:
            self.log.error("용어집(pos_dict) 로드 실패")
            raise RuntimeError("용어집(pos_dict) 로드 실패: 데이터가 비어있습니다")
        self.glossary = LightweightGlossaryIndex(df=glossary_df)
        self.log.info(f"용어집 로드 완료: {len(glossary_df)}행")

        # 사양값DB 로드 (umgv_fin 테이블)
        specdb_df = self.pg_loader.load_specdb_from_db()
        if specdb_df.empty:
            self.log.error("사양값DB(umgv_fin) 로드 실패")
            raise RuntimeError("사양값DB(umgv_fin) 로드 실패: 데이터가 비어있습니다")
        self.specdb = LightweightSpecDBIndex(df=specdb_df)
        self.log.info(f"사양값DB 로드 완료: {len(specdb_df)}행")

        # === 동적 지식 로더 초기화 ===
        self.pg_knowledge_loader = PostgresKnowledgeLoader(
            conn=self.pg_loader.conn,  # 기존 연결 재사용
            logger=self.log
        )

        # 지식 로드 (1회, 수초 소요)
        if not self.pg_knowledge_loader.load_all():
            self.log.warning("동적 지식 로드 실패, 기본 기능만 사용")
        else:
            self.log.info(
                f"동적 지식 로드 완료: "
                f"동의어 {len(self.pg_knowledge_loader.synonym_reverse)}개, "
                f"단위 {len(self.pg_knowledge_loader.unit_reverse)}개, "
                f"약어 {len(self.pg_knowledge_loader.abbreviations)}개"
            )

        # ReferenceHintEngine 초기화 (용어집/사양값DB 참조)
        self.hint_engine = ReferenceHintEngine(
            glossary=self.glossary,
            specdb=self.specdb,
            pg_loader=self.pg_loader,
            logger=self.log
        )
        self._hint_engine_initialized = True
        self.log.info("ReferenceHintEngine 초기화 완료")

        # SynonymManager 초기화 (Lazy loading)
        self.synonym_manager = None
        self._synonym_manager_initialized = False
        self.log.info("SynonymManager: Lazy loading 모드")

        # SemanticMatcher (Lazy loading)
        self.semantic_matcher = None
        self._semantic_matcher_initialized = False
        self.log.info("SemanticMatcher: Lazy loading 모드")

        elapsed = time.time() - start
        self.log.info(f"Full 모드 초기화 완료: {elapsed:.2f}초")
    
    def _get_parser(self, file_path: str) -> HTMLChunkParser:
        """HTML 파서 캐시 조회/생성"""
        if file_path not in self._parser_cache:
            self._parser_cache[file_path] = HTMLChunkParser(file_path=file_path)
        return self._parser_cache[file_path]
    
    def extract_single(self, html_path: str, spec: SpecItem) -> Dict[str, Any]:
        """
        단일 사양값 추출 (Rule + LLM Fallback)
        
        추출 순서:
        1. Rule 기반 추출 (테이블 + 키워드)
        2. Pre-Check 검증
        3. 실패 시 LLM Fallback
        
        ReferenceHintEngine 힌트 활용
        """
        self.stats['total'] += 1
        
        # 파일 존재 확인
        if not os.path.exists(html_path):
            self.stats['failed'] += 1
            return self._create_empty_result(spec, "FILE_NOT_FOUND")
        
        # HTML 파싱
        parser = self._get_parser(html_path)
        
        # 힌트 조회 (캐시된 경우 O(1), 아니면 동적 생성)
        # Lazy loading: 첫 사용 시 자동 초기화
        hint = None
        hint_engine = self._get_hint_engine()
        if hint_engine and spec.hull:
            hint = hint_engine.get_hints(spec.hull, spec.spec_name)
        
        # 1. Rule 기반 추출 시도 (힌트 활용)
        result = self.rule_extractor.extract(parser, spec, hint)
        
        if result and result.value:
            # Pre-Check
            chunk_context = parser.get_context_for_value(result.value)
            errors = self.pre_checker.check(
                spec.spec_name, result.value, result.unit,
                chunk_context=chunk_context
            )

            if not errors:
                # CRITICAL: 모든 추출 결과는 LLM 평가를 거쳐야 함
                # Rule 기반 추출이 성공해도 LLM 검증 필수
                if self.llm_validator:
                    self.log.info("Rule 기반 추출 성공 → LLM 검증 시작: %s", spec.spec_name)

                    # HTML 컨텍스트 준비 (우선순위: result.chunk > chunk_context > full_text)
                    html_context = None

                    # 1순위: result.chunk (가장 관련성 높음)
                    if result.chunk and result.chunk.strip():
                        html_context = result.chunk
                    # 2순위: chunk_context
                    elif chunk_context and chunk_context.strip():
                        html_context = chunk_context
                    # 3순위: get_context_for_value
                    elif result.value:
                        html_context = parser.get_context_for_value(result.value)

                    # 마지막 fallback: 전체 텍스트
                    if not html_context or html_context.strip() == "":
                        self.log.warning("Context 비어있음, 전체 텍스트 fallback 사용: %s", spec.spec_name)
                        html_context = parser.get_full_text()[:2000]

                    if len(html_context) > 2000:
                        html_context = html_context[:2000]

                    # LLM 검증 (voting 활성화)
                    validation = self.llm_validator.validate_extraction(
                        spec=spec,
                        extracted_value=result.value,
                        extracted_unit=result.unit,
                        html_context=html_context,
                        hint=hint,  # related_units 포함
                        use_voting=True  # Voting으로 정확도 향상
                    )

                    if validation['is_valid']:
                        # LLM 검증 성공
                        self.log.info("LLM 검증 성공: %s (confidence: %.2f)",
                                    spec.spec_name, validation['confidence'])

                        # LLM이 값을 수정한 경우 반영
                        if validation['llm_extracted_value'] != result.value:
                            self.log.info("LLM 값 보정: '%s' → '%s'",
                                        result.value, validation['llm_extracted_value'])
                            result.value = validation['llm_extracted_value']
                            result.unit = validation['llm_extracted_unit']
                            result.method = "rule+llm_corrected"
                        else:
                            result.method = "rule+llm_validated"

                        result.confidence = validation['confidence']
                        self.stats['rule_success'] += 1
                        return self._create_result(result, spec, html_path, hint)
                    else:
                        # LLM이 Rule 결과를 거부 → LLM Fallback으로 진행
                        self.log.warning("LLM이 Rule 결과 거부: %s (이유: %s)",
                                       spec.spec_name, validation['reason'])
                        self.log.info("LLM Fallback으로 재추출 시도")
                        # 아래 LLM Fallback 섹션으로 진행
                else:
                    # LLMValidator가 없으면 Rule 결과 그대로 사용 (비권장)
                    self.log.warning("LLMValidator 없음 - Rule 결과를 검증 없이 사용")
                    self.stats['rule_success'] += 1
                    return self._create_result(result, spec, html_path, hint)
            else:
                # Pre-Check 실패 로그
                self.log.debug("Pre-Check 실패: %s -> %s (errors: %s)",
                             spec.spec_name, result.value, errors)
        
        # 2. LLM Fallback 시도 (힌트를 프롬프트에 포함)
        if self.llm_fallback:
            llm_result = self.llm_fallback.extract(parser, spec, hint)
            
            if llm_result and llm_result.value:
                # LLM 결과도 Pre-Check
                errors = self.pre_checker.check(
                    spec.spec_name, llm_result.value, llm_result.unit,
                    chunk_context=""
                )

                if not errors:
                    self.stats['llm_fallback'] += 1
                    return self._create_result(llm_result, spec, html_path, hint)
                else:
                    self.log.warning("LLM Fallback Pre-Check 실패: %s -> %s (errors: %s)",
                                   spec.spec_name, llm_result.value, errors)
            else:
                if llm_result:
                    self.log.warning("LLM Fallback 결과 있으나 value 비어있음: %s (result: %s)",
                                   spec.spec_name, llm_result)
                else:
                    self.log.warning("LLM Fallback 결과 없음 (None 반환): %s", spec.spec_name)
        
        # 3. 모두 실패
        self.stats['failed'] += 1
        return self._create_empty_result(spec, "EXTRACTION_FAILED")

    def _get_synonym_manager(self) -> 'SynonymManager':
        """
        Lazy loading: SynonymManager 초기화 (첫 호출 시)

        초기화 시간 단축을 위해 필요할 때만 생성
        """
        if not self._synonym_manager_initialized:
            self.synonym_manager = SynonymManager()

            if self.glossary and hasattr(self.glossary, 'df') and not self.glossary.df.empty:
                self.synonym_manager.build_from_glossary(self.glossary.df)
                self.log.info("SynonymManager 초기화 완료 (Lazy)")
            elif self.config.data_source_mode == "db" and self.pg_loader:
                glossary_df = self.pg_loader.load_glossary_from_db()
                if not glossary_df.empty:
                    self.synonym_manager.build_from_glossary(glossary_df)
                    self.log.info("SynonymManager 초기화 완료 (DB, Lazy)")

            self._synonym_manager_initialized = True

        return self.synonym_manager

    def _get_hint_engine(self) -> 'ReferenceHintEngine':
        """
        Lazy loading: ReferenceHintEngine 초기화 (첫 호출 시)

        초기화 시간 단축을 위해 필요할 때만 생성
        """
        if not self._hint_engine_initialized:
            if self.glossary or self.specdb:
                self.hint_engine = ReferenceHintEngine(
                    glossary=self.glossary,
                    specdb=self.specdb,
                    pg_loader=self.pg_loader,
                    logger=self.log
                )
                self.log.info("ReferenceHintEngine 초기화 완료 (Lazy)")
            self._hint_engine_initialized = True

        return self.hint_engine

    def _get_semantic_matcher(self) -> 'SemanticMatcher':
        """
        Lazy loading: SemanticMatcher 초기화 (첫 호출 시)

        초기화 시간 단축을 위해 필요할 때만 생성
        """
        if not self._semantic_matcher_initialized:
            if self.config.enable_semantic_search:
                try:
                    self.semantic_matcher = SemanticMatcher(
                        model_path=self.config.semantic_model_path,
                        device=self.config.semantic_device,
                        similarity_threshold=self.config.semantic_similarity_threshold,
                        pg_loader=self.pg_loader,
                        logger=self.log
                    )
                    self.log.info("SemanticMatcher 초기화 완료 (Lazy)")
                except Exception as e:
                    self.log.warning("SemanticMatcher 초기화 실패: %s", e)
                    self.semantic_matcher = None

            self._semantic_matcher_initialized = True

        return self.semantic_matcher

    def preload_hints_for_file(self, file_path: str, hull: str = ""):
        """
        특정 파일 처리 전 힌트 배치 로드 (효율성 최적화)

        파일 처리 루프에서 첫 파일 처리 전 호출하면
        해당 hull의 모든 사양 힌트가 캐시에 로드됩니다.

        Args:
            file_path: HTML 파일 경로
            hull: 호선 번호 (없으면 파일명에서 추출 시도)
        """
        # Lazy loading: 첫 사용 시 자동 초기화
        hint_engine = self._get_hint_engine()
        if not hint_engine:
            return

        # hull 추출
        if not hull:
            filename = os.path.basename(file_path)
            hull = self._extract_hull_from_filename(filename)

        if hull:
            hint_engine.preload_for_hull(hull)
            self.log.debug("힌트 배치 로드 완료: hull=%s", hull)
    
    def _extract_hull_from_filename(self, filename: str) -> str:
        """파일명에서 hull 추출"""
        # 파일명 패턴: XXXX-POS-XXXXXXX... (앞 4자리가 hull)
        match = re.match(r'^(\d{4})', filename)
        if match:
            return match.group(1)
        return ""

    def _extract_spec_name_from_chunk(
        self,
        chunk: str,
        extracted_value: str,
        spec_name: str,
        hint: ExtractionHint = None
    ) -> str:
        """
        Chunk에서 사양명 추출 (LLM이 실패했을 때 fallback)

        문제: LLM이 original_spec_name을 추출하지 못하면 spec.spec_name이 그대로 pos_umgv_desc에 들어감
        해결: chunk에서 직접 추출 시도

        전략:
        1. 값 주변의 텍스트에서 사양명 찾기
        2. 힌트의 pos_umgv_desc 활용
        3. 구분자 패턴 (: | / =) 활용
        4. 문장 구조 분석
        5. 범용적인 키워드 매칭 (특정 패턴에 의존하지 않음)
        """
        if not chunk:
            return ""

        candidates = []

        # 전략 1: 힌트의 pos_umgv_desc가 chunk에 있으면 사용
        if hint and hint.pos_umgv_desc:
            pos_desc_upper = hint.pos_umgv_desc.upper()
            chunk_upper = chunk.upper()
            if pos_desc_upper in chunk_upper:
                candidates.append(('hint', hint.pos_umgv_desc, 1.0))

        # 전략 2: 값 주변 텍스트 분석
        if extracted_value:
            # 값이 있는 위치 찾기
            value_pos = chunk.upper().find(extracted_value.upper())
            if value_pos > 0:
                # 값 앞의 텍스트 (최대 200자)
                before_value = chunk[max(0, value_pos - 200):value_pos]

                # 구분자로 분리 (: | / =)
                for delimiter in [':', '|', '/', '=']:
                    if delimiter in before_value:
                        parts = before_value.split(delimiter)
                        if len(parts) >= 2:
                            # 구분자 직전 부분
                            spec_candidate = parts[-1].strip()
                            # 너무 짧거나 길면 제외
                            if 3 <= len(spec_candidate) <= 150:
                                # 숫자로 시작하지 않으면 후보로 추가
                                if not spec_candidate[0].isdigit():
                                    candidates.append(('delimiter', spec_candidate, 0.8))

        # 전략 3: spec_name 키워드가 chunk에 있는지 확인
        spec_keywords = spec_name.upper().split()
        for line in chunk.split('\n'):
            line_upper = line.upper()
            # 2개 이상의 키워드가 포함된 라인
            matched_keywords = sum(1 for kw in spec_keywords if kw in line_upper)
            if matched_keywords >= 2:
                # 라인에서 구분자 앞 부분 추출
                for delimiter in [':', '|', '/']:
                    if delimiter in line:
                        before_delim = line.split(delimiter)[0].strip()
                        if 3 <= len(before_delim) <= 150 and not before_delim[0].isdigit():
                            score = 0.6 + (matched_keywords * 0.1)
                            candidates.append(('keyword_match', before_delim, score))
                        break

        # 전략 4: 범용적인 패턴 - 숫자 앞의 텍스트
        # 예: "Capacity 700 m3/h" → "Capacity"
        number_pattern = r'([A-Za-z\s\(\)/\-]{3,100})\s+(\d+\.?\d*)'
        matches = re.finditer(number_pattern, chunk)
        for match in matches:
            potential_spec = match.group(1).strip()
            # 뒤의 숫자가 extracted_value와 일치하면 높은 점수
            number = match.group(2)
            if number in extracted_value:
                candidates.append(('number_context', potential_spec, 0.7))

        # 후보가 없으면 빈 문자열 반환 (spec.spec_name을 넣지 않음)
        if not candidates:
            return ""

        # 점수가 가장 높은 후보 선택
        candidates.sort(key=lambda x: -x[2])
        best_candidate = candidates[0][1]

        # 추가 검증: spec_name과 너무 유사하면 제외 (단순 복사 방지)
        if best_candidate.upper() == spec_name.upper():
            # 두 번째 후보 확인
            if len(candidates) > 1:
                return candidates[1][1]
            else:
                return ""  # spec_name과 같으면 빈 문자열

        return best_candidate

    def _create_result(
        self,
        result: ExtractionResult,
        spec: SpecItem,
        html_path: str = "",
        hint: ExtractionHint = None
    ) -> Dict[str, Any]:
        """결과 딕셔너리 생성 (힌트 정보 포함)"""
        raw = spec.raw_data or {}

        # 힌트에서 section_num 가져오기
        section_num = ""
        if hint and hint.section_num:
            section_num = hint.section_num[:100]  # 최대 100자

        # ===== Issue 1 Fix: pos_umgv_desc 개선 =====
        # LLM이 original_spec_name을 추출하지 못했거나 spec.spec_name과 동일한 경우,
        # chunk에서 직접 추출 시도
        pos_umgv_desc = result.original_spec_name or ""
        if not pos_umgv_desc or pos_umgv_desc == spec.spec_name:
            # chunk에서 재추출 시도
            extracted_from_chunk = self._extract_spec_name_from_chunk(
                chunk=result.chunk or "",
                extracted_value=result.value,
                spec_name=spec.spec_name,
                hint=hint
            )
            if extracted_from_chunk and extracted_from_chunk != spec.spec_name:
                pos_umgv_desc = extracted_from_chunk
            # 재추출에도 실패하면 빈 문자열 (spec.spec_name을 넣지 않음)

        # ===== Issue 2 Fix: pos_chunk 개선 =====
        # 500자 → 1500자로 확대, 문장 단위로 자르기 (맥락 보존)
        pos_chunk = ""
        if result.chunk:
            max_chars = 1500
            if len(result.chunk) <= max_chars:
                pos_chunk = result.chunk
            else:
                # 1500자에서 가장 가까운 문장 끝(.!?) 찾기
                truncated = result.chunk[:max_chars]
                # 마지막 문장 끝 찾기
                last_period = max(
                    truncated.rfind('.'),
                    truncated.rfind('!'),
                    truncated.rfind('?'),
                    truncated.rfind('\n')
                )
                if last_period > max_chars * 0.7:  # 70% 이상이면 사용
                    pos_chunk = result.chunk[:last_period + 1]
                else:
                    # 문장 끝을 찾지 못하면 그냥 자르기
                    pos_chunk = truncated + "..."

        return {
            'pmg_desc': safe_get(raw, 'pmg_desc'),
            'pmg_code': safe_get(raw, 'pmg_code'),
            'umg_desc': safe_get(raw, 'umg_desc'),
            'umg_code': safe_get(raw, 'umg_code'),
            'extwg': safe_get(raw, 'extwg'),
            'extwg_desc': safe_get(raw, 'extwg_desc', spec.equipment),  # Required field
            'matnr': spec.matnr,
            'doknr': safe_get(raw, 'doknr'),
            'umgv_code': spec.spec_code,
            'umgv_desc': spec.spec_name,
            'section_num': section_num,
            'table_text': 'Y' if result.chunk else '',
            'value_format': self._detect_format(result.value),
            'umgv_uom': spec.expected_unit,
            'pos_chunk': pos_chunk,
            # POS 원문 텍스트 보존 (대소문자, 특수문자 등 그대로)
            'pos_mat_attr_desc': result.original_equipment or spec.equipment,
            'pos_umgv_desc': pos_umgv_desc,  # 개선된 로직 적용
            'pos_umgv_value': result.value,
            'umgv_value_edit': '',  # Always empty - user feedback before validation
            'pos_umgv_uom': result.original_unit or result.unit,
            'evidence_fb': '',
            # Additional metadata fields (not in required spec but useful)
            'file_name': os.path.basename(html_path) if html_path else '',
            'mat_attr_desc': spec.equipment,
            '_method': result.method,
            '_confidence': result.confidence,
            '_evidence': result.evidence,
            '_reference_source': result.reference_source if hasattr(result, 'reference_source') else '',
        }
    
    def _create_empty_result(self, spec: SpecItem, method: str) -> Dict[str, Any]:
        """빈 결과 생성"""
        raw = spec.raw_data or {}

        return {
            'pmg_desc': safe_get(raw, 'pmg_desc'),
            'pmg_code': safe_get(raw, 'pmg_code'),
            'umg_desc': safe_get(raw, 'umg_desc'),
            'umg_code': safe_get(raw, 'umg_code'),
            'extwg': safe_get(raw, 'extwg'),
            'extwg_desc': safe_get(raw, 'extwg_desc', ''),  # Required field
            'matnr': spec.matnr,
            'doknr': safe_get(raw, 'doknr'),
            'umgv_code': spec.spec_code,
            'umgv_desc': spec.spec_name,
            'section_num': '',
            'table_text': '',
            'value_format': '',
            'umgv_uom': spec.expected_unit,
            'pos_chunk': '',
            'pos_mat_attr_desc': '',
            'pos_umgv_desc': '',
            'pos_umgv_value': '',
            'umgv_value_edit': '',
            'pos_umgv_uom': '',
            'evidence_fb': '',
            # Additional metadata fields (not in required spec but useful)
            'file_name': '',
            'mat_attr_desc': spec.equipment,
            '_method': method,
            '_confidence': 0.0,
            '_evidence': '',
        }

    def extract_full(
        self,
        html_folder: str = None,
        checkpoint_file: str = None
    ) -> Dict[str, Any]:
        """
        Full 모드: 디렉토리 내 모든 HTML 파일에 대해 일괄 추출

        특징:
        1. 자동 템플릿 로드 (DB의 ext_tmpl 테이블)
        2. 배치 처리 및 체크포인트
        3. Voting 활성화로 정확도 향상
        4. 진행상황 및 상세 감사 로그

        Args:
            html_folder: HTML 파일이 있는 폴더 (없으면 config.light_mode_pos_folder 사용)
            checkpoint_file: 체크포인트 파일 경로 (없으면 자동 생성)

        Returns:
            전체 추출 결과 딕셔너리
        """
        self.log.info("=" * 80)
        self.log.info("Full 모드 추출 시작")
        self.log.info("=" * 80)

        # 폴더 설정
        folder = html_folder or self.config.light_mode_pos_folder
        if not folder or not os.path.exists(folder):
            raise ValueError(f"HTML 폴더를 찾을 수 없습니다: {folder}")

        # 체크포인트 파일 설정
        if not checkpoint_file:
            checkpoint_dir = self.config.full_mode_checkpoint_dir or self.config.output_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_file = os.path.join(checkpoint_dir, f"full_mode_checkpoint_{timestamp}.json")

        # HTML 파일 목록 수집
        html_files = []
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith('.html'):
                html_files.append(os.path.join(folder, filename))

        self.log.info("총 %d개 HTML 파일 발견", len(html_files))

        if not html_files:
            self.log.warning("처리할 HTML 파일이 없습니다")
            return {"status": "no_files", "total_files": 0, "results": []}

        # 체크포인트 로드 (이전 실행 재개)
        processed_files = set()
        all_results = []
        checkpoint_data = self._load_checkpoint(checkpoint_file)
        if checkpoint_data:
            processed_files = set(checkpoint_data.get('processed_files', []))
            all_results = checkpoint_data.get('results', [])
            self.log.info("체크포인트 로드: %d개 파일 이미 처리됨", len(processed_files))

        # 파일별 처리
        total_extracted = 0
        total_failed = 0
        batch_results = []

        for idx, html_file in enumerate(html_files, 1):
            if html_file in processed_files:
                self.log.debug("스킵 (이미 처리됨): %s", os.path.basename(html_file))
                continue

            self.log.info("-" * 80)
            self.log.info("[%d/%d] 처리 중: %s", idx, len(html_files), os.path.basename(html_file))
            self.log.info("-" * 80)

            try:
                # 파일에 대한 템플릿 로드
                file_specs = self._load_template_for_file(html_file)

                if not file_specs:
                    self.log.warning("템플릿 없음, 스킵: %s", os.path.basename(html_file))
                    processed_files.add(html_file)
                    continue

                self.log.info("템플릿 로드 완료: %d개 사양 항목", len(file_specs))

                # 힌트 배치 로드 (성능 최적화)
                hull = self._extract_hull_from_filename(os.path.basename(html_file))
                if hull:
                    self.preload_hints_for_file(html_file, hull)

                # 힌트 딕셔너리 준비 (Lazy loading)
                hints_dict = {}
                hint_engine = self._get_hint_engine()
                if hint_engine:
                    for spec in file_specs:
                        hint = hint_engine.get_hints(spec.hull, spec.spec_name)
                        if hint:
                            hints_dict[spec.spec_name] = hint

                # 배치 추출 (15개씩 묶어서 처리)
                file_results = []
                batch_size = self.config.full_mode_batch_size  # 기본 15개
                parser = self._get_parser(html_file)

                for batch_start in range(0, len(file_specs), batch_size):
                    batch_specs = file_specs[batch_start:batch_start + batch_size]
                    batch_end = min(batch_start + batch_size, len(file_specs))

                    self.log.info("  Batch 추출: [%d-%d/%d]", batch_start + 1, batch_end, len(file_specs))

                    # Batch 추출 시도 (LLM Fallback이 있는 경우)
                    batch_results = None
                    if self.llm_fallback and len(batch_specs) > 1:
                        try:
                            batch_results = self.llm_fallback.extract_batch(
                                parser=parser,
                                specs=batch_specs,
                                hints=hints_dict
                            )
                        except Exception as e:
                            self.log.warning("Batch 추출 실패, 개별 추출로 fallback: %s", e)

                    # Batch 실패 시 개별 추출
                    if not batch_results or all(r is None for r in batch_results):
                        self.log.debug("  개별 추출로 fallback")
                        for spec_idx, spec in enumerate(batch_specs, batch_start + 1):
                            result = self.extract_single(html_file, spec)
                            file_results.append(result)

                            if result.get('pos_umgv_value'):
                                total_extracted += 1
                            else:
                                total_failed += 1
                    else:
                        # Batch 결과 처리
                        for spec_idx, (spec, extraction_result) in enumerate(zip(batch_specs, batch_results), batch_start + 1):
                            if extraction_result and extraction_result.value:
                                # LLM Batch 결과를 Dict로 변환
                                result_dict = self._create_result(extraction_result, spec, html_file, hints_dict.get(spec.spec_name))
                                file_results.append(result_dict)
                                total_extracted += 1
                                self.log.debug("    [%d] %s: %s", spec_idx, spec.spec_name, extraction_result.value)
                            else:
                                # Batch에서 실패한 항목은 개별 추출 시도
                                self.log.debug("    [%d] Batch 실패, 개별 추출: %s", spec_idx, spec.spec_name)
                                result = self.extract_single(html_file, spec)
                                file_results.append(result)

                                if result.get('pos_umgv_value'):
                                    total_extracted += 1
                                else:
                                    total_failed += 1

                all_results.extend(file_results)
                batch_results.extend(file_results)
                processed_files.add(html_file)

                self.log.info("파일 처리 완료: 추출 %d개 / 실패 %d개",
                            len([r for r in file_results if r.get('pos_umgv_value')]),
                            len([r for r in file_results if not r.get('pos_umgv_value')]))

                # 체크포인트 저장 (주기적)
                if self.config.enable_checkpoint and len(processed_files) % self.config.full_mode_checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint_file, list(processed_files), all_results)
                    self.log.info("체크포인트 저장: %s", checkpoint_file)

            except Exception as e:
                self.log.error("파일 처리 실패: %s - %s", os.path.basename(html_file), e)
                processed_files.add(html_file)  # 실패해도 스킵 처리
                continue

        # 최종 체크포인트 저장
        if self.config.enable_checkpoint:
            self._save_checkpoint(checkpoint_file, list(processed_files), all_results)
            self.log.info("최종 체크포인트 저장: %s", checkpoint_file)

        # 결과 요약
        self.log.info("=" * 80)
        self.log.info("Full 모드 추출 완료")
        self.log.info("=" * 80)
        self.log.info("처리 파일: %d / %d", len(processed_files), len(html_files))
        self.log.info("추출 성공: %d", total_extracted)
        self.log.info("추출 실패: %d", total_failed)

        # 결과 저장
        saved_files = self._save_full_mode_results(all_results)

        return {
            "status": "completed",
            "mode": "full",
            "total_files": len(html_files),
            "processed_files": len(processed_files),
            "total_specs": len(all_results),
            "extracted": total_extracted,
            "failed": total_failed,
            "checkpoint_file": checkpoint_file,
            "saved_files": saved_files,
            "results": all_results
        }

    def verify_full(
        self,
        html_folder: str = None
    ) -> Dict[str, Any]:
        """
        Verify 모드: 사양값DB의 기존 값을 POS 문서와 대조하여 검증

        프로세스:
        1. 사양값DB에서 기존 값 로드 (umgv_value, umgv_uom)
        2. POS 문서에서 실제 값 추출 (pos_umgv_value, pos_umgv_uom)
        3. 단위 변환을 고려하여 값 비교
        4. 명칭 차이 고려 (umgv_desc vs pos_umgv_desc)
        5. 검증 결과 반환 (일치/불일치, 신뢰도)

        Returns:
            검증 결과 딕셔너리
        """
        self.log.info("=" * 80)
        self.log.info("Verify 모드 검증 시작")
        self.log.info("=" * 80)

        # 폴더 설정
        folder = html_folder or self.config.light_mode_pos_folder
        if not folder or not os.path.exists(folder):
            raise ValueError(f"HTML 폴더를 찾을 수 없습니다: {folder}")

        # HTML 파일 목록
        html_files = []
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith('.html'):
                html_files.append(os.path.join(folder, filename))

        self.log.info("총 %d개 HTML 파일 발견", len(html_files))

        if not html_files:
            self.log.warning("처리할 HTML 파일이 없습니다")
            return {"status": "no_files", "total_files": 0, "verification_results": []}

        # 검증 결과
        all_verifications = []
        total_verified = 0
        total_matched = 0
        total_mismatched = 0
        total_db_missing = 0

        for idx, html_file in enumerate(html_files, 1):
            self.log.info("-" * 80)
            self.log.info("[%d/%d] 검증 중: %s", idx, len(html_files), os.path.basename(html_file))
            self.log.info("-" * 80)

            try:
                # 사양값DB에서 기존 값 로드
                db_values = self._load_db_values_for_file(html_file)

                if not db_values:
                    self.log.warning("사양값DB에 데이터 없음, 스킵: %s", os.path.basename(html_file))
                    continue

                self.log.info("사양값DB 로드: %d개 항목", len(db_values))

                # 각 DB 값에 대해 검증
                for db_item in db_values:
                    verification = self._verify_single_item(html_file, db_item)
                    all_verifications.append(verification)

                    total_verified += 1
                    if verification['verification_status'] == 'MATCHED':
                        total_matched += 1
                    elif verification['verification_status'] == 'MISMATCHED':
                        total_mismatched += 1
                    elif verification['verification_status'] == 'DB_MISSING':
                        total_db_missing += 1

                    # 로그 출력
                    status = verification['verification_status']
                    confidence = verification['verification_confidence']
                    self.log.info(
                        "  %s: %s (신뢰도: %.2f) - %s",
                        db_item['umgv_desc'],
                        status,
                        confidence,
                        verification['verification_reason']
                    )

            except Exception as e:
                self.log.error("파일 검증 실패 %s: %s", os.path.basename(html_file), e)
                continue

        # 검증 통계
        match_rate = (total_matched / total_verified * 100) if total_verified > 0 else 0.0
        mismatch_rate = (total_mismatched / total_verified * 100) if total_verified > 0 else 0.0

        self.log.info("=" * 80)
        self.log.info("검증 완료")
        self.log.info("총 검증: %d개", total_verified)
        self.log.info("일치: %d개 (%.1f%%)", total_matched, match_rate)
        self.log.info("불일치: %d개 (%.1f%%)", total_mismatched, mismatch_rate)
        self.log.info("DB 누락: %d개", total_db_missing)
        self.log.info("=" * 80)

        # 결과 저장
        self._save_verification_results(all_verifications)

        return {
            "status": "success",
            "total_files": len(html_files),
            "total_verified": total_verified,
            "total_matched": total_matched,
            "total_mismatched": total_mismatched,
            "total_db_missing": total_db_missing,
            "match_rate": match_rate,
            "mismatch_rate": mismatch_rate,
            "verification_results": all_verifications
        }

    def _load_db_values_for_file(self, html_file: str) -> List[Dict]:
        """
        HTML 파일에 대응하는 사양값DB 데이터 로드

        Returns:
            사양값DB 항목 리스트 (umgv_value, umgv_uom 포함)
        """
        if not self.pg_loader:
            self.log.warning("PostgreSQL 연결 없음 - 사양값DB 로드 불가")
            return []

        filename = os.path.basename(html_file)
        hull = self._extract_hull_from_filename(filename)

        if not hull:
            self.log.warning("Hull 추출 실패: %s", filename)
            return []

        try:
            # 사양값DB 로드 (umgv_fin 테이블)
            query = f"""
                SELECT
                    umgv_desc, umgv_code, umgv_value, umgv_uom,
                    pos_mat_attr_desc, pos_umgv_desc,
                    matnr, doknr
                FROM {VERIFY_MODE_SPECDB_TABLE}
                WHERE matnr LIKE %s
                ORDER BY umgv_desc
            """

            conn = self.pg_loader.get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query, (f"{hull}%",))
            rows = cursor.fetchall()
            cursor.close()

            db_values = [dict(row) for row in rows]
            return db_values

        except Exception as e:
            self.log.error("사양값DB 로드 실패: %s", e)
            return []

    def _verify_single_item(self, html_file: str, db_item: Dict) -> Dict:
        """
        단일 항목 검증

        Args:
            html_file: POS HTML 파일 경로
            db_item: 사양값DB 항목 (umgv_value, umgv_uom 포함)

        Returns:
            검증 결과 딕셔너리
        """
        # DB 값
        db_value = norm(db_item.get('umgv_value', ''))
        db_unit = norm(db_item.get('umgv_uom', ''))
        umgv_desc = norm(db_item.get('umgv_desc', ''))
        pos_umgv_desc = norm(db_item.get('pos_umgv_desc', ''))

        # SpecItem 생성
        spec = SpecItem(
            spec_name=umgv_desc,
            spec_code=db_item.get('umgv_code', ''),
            equipment=db_item.get('pos_mat_attr_desc', ''),
            expected_unit=db_unit,
            matnr=db_item.get('matnr', ''),
            hull=self._extract_hull_from_filename(os.path.basename(html_file)),
            raw_data=db_item
        )

        # POS 문서에서 값 추출
        extraction_result = self.extract_single(html_file, spec)
        pos_value = norm(extraction_result.get('pos_umgv_value', ''))
        pos_unit = norm(extraction_result.get('pos_umgv_uom', ''))

        # DB에 값이 없는 경우
        if not db_value:
            return {
                **db_item,
                'pos_umgv_value': pos_value,
                'pos_umgv_uom': pos_unit,
                'verification_status': 'DB_MISSING',
                'verification_confidence': 0.0,
                'verification_reason': 'DB에 값 없음'
            }

        # POS에서 값을 찾지 못한 경우
        if not pos_value:
            return {
                **db_item,
                'pos_umgv_value': pos_value,
                'pos_umgv_uom': pos_unit,
                'verification_status': 'POS_NOT_FOUND',
                'verification_confidence': 0.0,
                'verification_reason': 'POS 문서에서 값 추출 실패'
            }

        # 값 비교 (단위 변환 고려)
        matched, confidence, reason = values_match_with_unit_conversion(
            pos_value, pos_unit,
            db_value, db_unit,
            tolerance_percent=self.verify_tolerance
        )

        # 명칭 차이 고려 (pos_umgv_desc vs umgv_desc)
        name_match = (umgv_desc.upper() == pos_umgv_desc.upper()) if pos_umgv_desc else True
        if not name_match:
            reason += f" (명칭 차이: {umgv_desc} vs {pos_umgv_desc})"

        status = 'MATCHED' if matched and confidence >= self.verify_confidence_threshold else 'MISMATCHED'

        return {
            **db_item,
            'pos_umgv_value': pos_value,
            'pos_umgv_uom': pos_unit,
            'verification_status': status,
            'verification_confidence': confidence,
            'verification_reason': reason
        }

    def _save_verification_results(self, verifications: List[Dict]):
        """검증 결과 저장"""
        if not self.config.output_path:
            return

        os.makedirs(self.config.output_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # JSON 저장
        json_path = os.path.join(self.config.output_path, f"verification_result_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(verifications, f, ensure_ascii=False, indent=2)
        self.log.info("검증 결과 JSON 저장: %s", json_path)

        # 콘솔 출력 (PRINT_JSON 옵션)
        if self.config.print_json:
            print("\n" + "=" * 80)
            print("Verify 모드 검증 결과 (Pretty Print)")
            print("=" * 80)
            print(json.dumps(verifications, ensure_ascii=False, indent=2))
            print("=" * 80 + "\n")

        # CSV 저장
        if HAS_PANDAS:
            csv_path = os.path.join(self.config.output_path, f"verification_result_{timestamp}.csv")
            df = pd.DataFrame(verifications)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            self.log.info("검증 결과 CSV 저장: %s", csv_path)

    def _load_template_for_file(self, html_file: str) -> List[SpecItem]:
        """
        HTML 파일에 대한 템플릿(사양 항목 목록) 로드

        DB 모드: ext_tmpl 테이블에서 로드
        파일 모드: spec 파일에서 로드
        """
        if not self.pg_loader:
            self.log.warning("PostgreSQL 연결 없음 - 템플릿 로드 불가")
            return []

        # 파일명에서 매칭 정보 추출
        filename = os.path.basename(html_file)

        # ext_tmpl 테이블에서 템플릿 로드
        try:
            tmpl_df = self.pg_loader.load_template_from_db()
            if tmpl_df.empty:
                self.log.error("템플릿 테이블(ext_tmpl)이 비어있습니다")
                return []

            # 파일명 기반 필터링 (doknr 매칭)
            # 파일명 패턴: XXXX-POS-YYYYYYY...
            # doknr에서 XXXX 또는 YYYYYYY 부분 매칭
            hull = self._extract_hull_from_filename(filename)

            # hull로 필터링
            if hull:
                filtered_df = tmpl_df[tmpl_df['doknr'].str.contains(hull, na=False, case=False)]
            else:
                # hull 없으면 전체 사용 (비권장)
                filtered_df = tmpl_df

            if filtered_df.empty:
                self.log.error("템플릿 필터링 결과 없음: %s", filename)
                return []

            # SpecItem 객체 생성
            specs = []
            for _, row in filtered_df.iterrows():
                spec = SpecItem(
                    spec_name=row.get('umgv_desc', ''),
                    spec_code=row.get('umgv_code', ''),
                    equipment=row.get('pos_mat_attr_desc', ''),
                    expected_unit=row.get('umgv_uom', ''),
                    matnr=row.get('matnr', ''),
                    hull=hull,
                    raw_data=row.to_dict()
                )
                specs.append(spec)

            return specs

        except Exception as e:
            self.log.error("템플릿 로드 실패: %s", e)
            return []

    def _load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """체크포인트 로드"""
        if not os.path.exists(checkpoint_file):
            return None

        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log.warning("체크포인트 로드 실패: %s", e)
            return None

    def _save_checkpoint(self, checkpoint_file: str, processed_files: List[str], results: List[Dict]):
        """체크포인트 저장"""
        try:
            checkpoint_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processed_files": processed_files,
                "total_processed": len(processed_files),
                "total_results": len(results),
                "results": results
            }

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.log.error("체크포인트 저장 실패: %s", e)

    def _save_full_mode_results(self, results: List[Dict]) -> Dict[str, str]:
        """Full 모드 결과 저장 (JSON, CSV)"""
        saved_files = {}

        if not self.config.output_path:
            self.log.warning("출력 경로 미설정 - 결과 저장 스킵")
            return saved_files

        os.makedirs(self.config.output_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # JSON 저장
        if self.config.save_json:
            json_file = os.path.join(self.config.output_path, f"full_mode_results_{timestamp}.json")
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                saved_files['json'] = json_file
                self.log.info("JSON 저장 완료: %s", json_file)
            except Exception as e:
                self.log.error("JSON 저장 실패: %s", e)

        # CSV 저장
        if self.config.save_csv:
            csv_file = os.path.join(self.config.output_path, f"full_mode_results_{timestamp}.csv")
            try:
                df = pd.DataFrame(results)
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                saved_files['csv'] = csv_file
                self.log.info("CSV 저장 완료: %s", csv_file)
            except Exception as e:
                self.log.error("CSV 저장 실패: %s", e)

        return saved_files

    def extract_verify(
        self,
        html_folder: str = None,
        verify_sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Verify 모드: umgv_fin 기반 검증

        기존 추출 결과(umgv_fin)를 검증하여 정확도 측정
        POS 문서에서 값을 찾아 맥락 기반으로 검증

        Args:
            html_folder: HTML 파일이 있는 폴더
            verify_sample_size: 검증할 샘플 수 (0=전체)

        Returns:
            검증 결과 딕셔너리
        """
        self.log.info("=" * 80)
        self.log.info("Verify 모드 시작")
        self.log.info("=" * 80)

        # 폴더 설정
        folder = html_folder or self.config.light_mode_pos_folder
        if not folder or not os.path.exists(folder):
            raise ValueError(f"HTML 폴더를 찾을 수 없습니다: {folder}")

        # umgv_fin 테이블에서 검증 대상 로드
        if not self.pg_loader:
            raise RuntimeError("PostgreSQL 연결이 필요합니다 (DB 모드)")

        specdb_df = self.pg_loader.load_specdb_from_db()
        if specdb_df.empty:
            raise RuntimeError("umgv_fin 테이블이 비어있습니다")

        self.log.info("umgv_fin에서 %d개 레코드 로드", len(specdb_df))

        # 샘플링
        if verify_sample_size > 0 and len(specdb_df) > verify_sample_size:
            specdb_df = specdb_df.sample(n=verify_sample_size, random_state=42)
            self.log.info("샘플링: %d개 레코드 선택", len(specdb_df))

        # 검증 통계
        total = 0
        verified = 0
        mismatched = 0
        not_found = 0
        errors = 0

        verification_results = []

        for idx, row in specdb_df.iterrows():
            total += 1

            try:
                # POS 문서 경로 추출
                doknr = row.get('doknr', '')
                if not doknr:
                    errors += 1
                    continue

                # HTML 파일 찾기
                html_file = self._find_html_file(folder, doknr)
                if not html_file:
                    not_found += 1
                    self.log.debug("[%d/%d] POS 파일 없음: %s", total, len(specdb_df), doknr)
                    continue

                # 검증 수행
                verify_result = self._verify_single(html_file, row)

                verification_results.append(verify_result)

                if verify_result['status'] == 'verified':
                    verified += 1
                elif verify_result['status'] == 'mismatched':
                    mismatched += 1
                    self.log.warning("[%d/%d] 불일치: %s - 기대 '%s', 발견 '%s'",
                                    total, len(specdb_df),
                                    verify_result['spec_name'],
                                    verify_result['expected_value'],
                                    verify_result['found_value'])
                else:
                    not_found += 1

            except Exception as e:
                errors += 1
                self.log.error("검증 오류: %s", e)

        # 결과 요약
        accuracy = (verified / total * 100) if total > 0 else 0.0

        self.log.info("=" * 80)
        self.log.info("Verify 모드 완료")
        self.log.info("=" * 80)
        self.log.info("총 검증: %d", total)
        self.log.info("일치: %d (%.1f%%)", verified, verified / total * 100 if total > 0 else 0)
        self.log.info("불일치: %d (%.1f%%)", mismatched, mismatched / total * 100 if total > 0 else 0)
        self.log.info("미발견: %d (%.1f%%)", not_found, not_found / total * 100 if total > 0 else 0)
        self.log.info("오류: %d", errors)
        self.log.info("정확도: %.1f%%", accuracy)

        return {
            "status": "completed",
            "mode": "verify",
            "total": total,
            "verified": verified,
            "mismatched": mismatched,
            "not_found": not_found,
            "errors": errors,
            "accuracy": accuracy,
            "results": verification_results
        }

    def _find_html_file(self, folder: str, doknr: str) -> Optional[str]:
        """
        doknr를 기반으로 HTML 파일 찾기

        파일명 패턴: XXXX-POS-YYYYYYY...
        """
        # doknr에서 hull 추출
        hull_match = re.search(r'(\d{4})', doknr)
        if not hull_match:
            return None

        hull = hull_match.group(1)

        # 파일명 매칭
        for filename in os.listdir(folder):
            if filename.lower().endswith('.html') and hull in filename:
                # doknr의 주요 부분이 파일명에 포함되어 있는지 확인
                if doknr[:10] in filename or hull in filename[:4]:
                    return os.path.join(folder, filename)

        return None

    def _verify_single(self, html_file: str, row: pd.Series) -> Dict[str, Any]:
        """
        단일 사양값 검증

        맥락 기반 단계적 탐색:
        1. Section 탐색 (section_num 기준)
        2. Table 탐색 (테이블 구조 기반)
        3. Full document 탐색

        Args:
            html_file: HTML 파일 경로
            row: umgv_fin 레코드

        Returns:
            검증 결과 딕셔너리
        """
        spec_name = row.get('umgv_desc', '')
        expected_value = str(row.get('umgv_value', '')).strip()
        expected_unit = str(row.get('umgv_uom', '')).strip()

        parser = self._get_parser(html_file)

        # 1단계: Section 기반 탐색
        section_num = row.get('section_num', '')
        if section_num:
            found = self._search_in_section(parser, spec_name, section_num, expected_value, expected_unit)
            if found:
                return {
                    'status': 'verified',
                    'spec_name': spec_name,
                    'expected_value': expected_value,
                    'found_value': found['value'],
                    'method': 'section_search'
                }

        # 2단계: Table 기반 탐색
        found = self._search_in_tables(parser, spec_name, expected_value, expected_unit)
        if found:
            return {
                'status': 'verified',
                'spec_name': spec_name,
                'expected_value': expected_value,
                'found_value': found['value'],
                'method': 'table_search'
            }

        # 3단계: Full document 탐색
        found = self._search_in_full_document(parser, spec_name, expected_value, expected_unit)
        if found:
            return {
                'status': 'verified',
                'spec_name': spec_name,
                'expected_value': expected_value,
                'found_value': found['value'],
                'method': 'full_document_search'
            }

        # 값을 찾지 못함
        return {
            'status': 'not_found',
            'spec_name': spec_name,
            'expected_value': expected_value,
            'found_value': '',
            'method': 'none'
        }

    def _search_in_section(
        self,
        parser: HTMLChunkParser,
        spec_name: str,
        section_num: str,
        expected_value: str,
        expected_unit: str
    ) -> Optional[Dict]:
        """Section 내에서 값 탐색"""
        full_text = parser.get_full_text()

        # Section 위치 찾기
        section_match = re.search(re.escape(section_num[:20]), full_text, re.IGNORECASE)
        if not section_match:
            return None

        # Section 주변 텍스트 추출 (±500자)
        start = max(0, section_match.start() - 500)
        end = min(len(full_text), section_match.end() + 1500)
        section_text = full_text[start:end]

        # 사양명과 값 찾기
        if spec_name.upper() in section_text.upper():
            # 유사값 탐색 (단위 변환, 정제 고려)
            if self._is_similar_value(expected_value, section_text):
                return {'value': expected_value, 'confidence': 0.8}

        return None

    def _search_in_tables(
        self,
        parser: HTMLChunkParser,
        spec_name: str,
        expected_value: str,
        expected_unit: str
    ) -> Optional[Dict]:
        """Table 내에서 값 탐색"""
        # HTMLChunkParser의 테이블 검색 활용
        keywords = [spec_name] + spec_name.upper().split()
        table_results = parser.search_in_tables_enhanced(keywords[:3])

        for result in table_results:
            value = result.get('value', '')
            if self._is_similar_value(expected_value, value):
                return {'value': value, 'confidence': 0.9}

        return None

    def _search_in_full_document(
        self,
        parser: HTMLChunkParser,
        spec_name: str,
        expected_value: str,
        expected_unit: str
    ) -> Optional[Dict]:
        """전체 문서에서 값 탐색"""
        full_text = parser.get_full_text()

        # 사양명 근처에서 값 찾기
        spec_pattern = re.escape(spec_name[:20])
        matches = list(re.finditer(spec_pattern, full_text, re.IGNORECASE))

        for match in matches:
            context_start = max(0, match.start() - 100)
            context_end = min(len(full_text), match.end() + 200)
            context = full_text[context_start:context_end]

            if self._is_similar_value(expected_value, context):
                return {'value': expected_value, 'confidence': 0.7}

        return None

    def _is_similar_value(self, expected: str, text: str) -> bool:
        """
        값 유사도 판단 (단위 변환, 정제 고려)

        예: "60°C" vs "60 deg C", "15~20" vs "15 to 20"
        """
        expected_norm = expected.upper().replace(' ', '').replace('~', '-')

        # 숫자만 추출하여 비교
        expected_nums = re.findall(r'[\d.]+', expected)
        text_nums = re.findall(r'[\d.]+', text)

        # 주요 숫자가 포함되어 있는지 확인
        for num in expected_nums:
            if num in text_nums or num in text:
                return True

        # 정규화된 문자열 포함 여부
        if expected_norm and expected_norm in text.upper().replace(' ', ''):
            return True

        return False

    def _detect_format(self, value: str) -> str:
        """값 형식 감지"""
        if not value:
            return ""
        if re.match(r'^[\d.,\-+~]+$', value):
            return "NUMERIC"
        if re.search(r'\d', value):
            return "MIXED"
        return "TEXT"
    
    # =========================================================================
    # Light 모드 전용 메서드
    # =========================================================================
    
    @staticmethod
    def scan_pos_files(pos_folder: str, extensions: List[str] = None) -> List[str]:
        """POS 폴더 내 파일 목록 스캔"""
        if not os.path.isdir(pos_folder):
            return []
        
        extensions = extensions or ['.html', '.htm']
        files = []
        
        for fname in os.listdir(pos_folder):
            fpath = os.path.join(pos_folder, fname)
            if os.path.isfile(fpath):
                _, ext = os.path.splitext(fname)
                if ext.lower() in extensions:
                    files.append(fname)
        
        return files
    
    @staticmethod
    def extract_doknr_from_filename(filename: str) -> str:
        """파일명에서 doknr 추출"""
        match = re.search(r'(\d+)-POS-(\d+)', filename)
        if match:
            return f"{match.group(1)}-POS-{match.group(2)}"
        return ""
    
    def filter_template_by_files(
        self,
        template_df: pd.DataFrame,
        pos_files: List[str],
        doknr_column: str = "doknr",
        matnr_column: str = "matnr"
    ) -> pd.DataFrame:
        """
        파일 목록 기반 템플릿 필터링
        
        개선사항:
        - hull + POS 번호 조합으로 정확한 매칭
        - 파일별로 다른 템플릿이 매칭되도록 함
        
        매칭 로직:
        1. 파일명: 4508-POS-0055401 → hull=4508, pos=0055401
        2. 템플릿: doknr=xxxx-POS-0055401, matnr=4508Axxxxxx
        3. 매칭: pos 번호 일치 AND matnr 앞 4자리(hull) 일치
        """
        if template_df is None or template_df.empty:
            return pd.DataFrame()
        
        # 파일명에서 hull + POS 번호 추출
        file_info = {}  # (hull, pos) -> filename
        file_pos_only = {}  # pos -> [hulls]
        
        for fname in pos_files:
            # 파일명 파싱: XXXX-POS-YYYYYYY
            match = re.match(r'^(\d{4})-POS-(\d+)', fname)
            if match:
                hull = match.group(1)
                pos_num = match.group(2)
                file_info[(hull, pos_num)] = fname
                
                # POS 번호만으로도 검색 가능하도록
                if pos_num not in file_pos_only:
                    file_pos_only[pos_num] = []
                file_pos_only[pos_num].append(hull)
                
                # 앞의 0 제거 버전
                pos_stripped = pos_num.lstrip('0')
                if pos_stripped and pos_stripped != pos_num:
                    file_info[(hull, pos_stripped)] = fname
                    if pos_stripped not in file_pos_only:
                        file_pos_only[pos_stripped] = []
                    file_pos_only[pos_stripped].append(hull)
        
        if not file_info:
            self.log.warning("파일 목록에서 hull/POS 정보 추출 실패")
            return pd.DataFrame()
        
        self.log.info("파일에서 추출된 hull-POS 조합: %d개", len(file_info))
        for (hull, pos), fname in list(file_info.items())[:5]:
            self.log.debug("  %s-%s: %s", hull, pos, fname)
        
        # 매칭 함수 (hull + POS 조합)
        def match_row(row) -> bool:
            doknr = str(row.get(doknr_column, '')).strip()
            matnr = str(row.get(matnr_column, '')).strip()
            
            # 템플릿에서 POS 번호 추출
            pos_match = re.search(r'POS-(\d+)', doknr)
            if not pos_match:
                return False
            template_pos = pos_match.group(1)
            template_pos_stripped = template_pos.lstrip('0')
            
            # 템플릿에서 hull 추출 (matnr 앞 4자리)
            template_hull = matnr[:4] if len(matnr) >= 4 else ""
            
            # hull + POS 조합 매칭 (정확 매칭)
            if template_hull and template_pos:
                if (template_hull, template_pos) in file_info:
                    return True
                if template_pos_stripped and (template_hull, template_pos_stripped) in file_info:
                    return True
            
            # hull이 없으면 POS만으로 매칭 (fallback)
            if not template_hull:
                if template_pos in file_pos_only or template_pos_stripped in file_pos_only:
                    return True
            
            return False
        
        # 필터링 적용
        filtered = template_df[template_df.apply(match_row, axis=1)]
        
        # hull별 통계
        if not filtered.empty and matnr_column in filtered.columns:
            hull_counts = filtered[matnr_column].str[:4].value_counts()
            self.log.info("hull별 템플릿 수:")
            for hull, cnt in hull_counts.head(10).items():
                self.log.info("  %s: %d rows", hull, cnt)
        
        self.log.info("템플릿 필터링: %d/%d rows (파일 %d개)", 
                     len(filtered), len(template_df), len(pos_files))
        
        return filtered
    
    def run_light_mode(
        self,
        pos_folder: str = "",
        template_df: pd.DataFrame = None,
        template_path: str = ""
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        소량 추출 모드 실행
        
        1. POS 폴더 파일 스캔
        2. 해당 파일에 대한 템플릿 row만 필터링
        3. 순차 추출 (batch 없음)
        """
        start_time = time.time()
        self.log.info("=" * 60)
        self.log.info("소량 추출 모드 (LIGHT) 시작")
        self.log.info("=" * 60)

        # GPU VRAM 모니터링
        log_gpu_memory(self.log)
        
        # POS 폴더 결정
        pos_folder = pos_folder or self.config.light_mode_pos_folder
        if not pos_folder or not os.path.isdir(pos_folder):
            self.log.error("POS 폴더 없음: %s", pos_folder)
            return [], {}
        
        # POS 파일 스캔
        pos_files = self.scan_pos_files(pos_folder)
        if not pos_files:
            self.log.error("POS 폴더에 파일 없음: %s", pos_folder)
            return [], {}
        
        self.log.info("POS 파일 발견: %d개", len(pos_files))
        for f in pos_files[:5]:
            self.log.info("  - %s", f)
        if len(pos_files) > 5:
            self.log.info("  ... 외 %d개", len(pos_files) - 5)
        
        # 템플릿 로드
        if template_df is None:
            if template_path and os.path.exists(template_path):
                template_df = load_tsv(template_path)
            elif self.config.data_source_mode == "db" and self.pg_loader:
                template_df = self.pg_loader.load_template_from_db()
            else:
                template_df = load_tsv(self.config.spec_path)
        
        if template_df is None or template_df.empty:
            self.log.error("템플릿 로드 실패")
            return [], {}
        
        # 템플릿 필터링
        filtered_df = self.filter_template_by_files(template_df, pos_files)
        if filtered_df.empty:
            self.log.error("템플릿 필터링 결과 없음: POS 파일에 대한 템플릿 정보가 없습니다.")
            self.log.error("추출 대상 사양 항목이 불명확하므로 작업을 종료합니다.")
            return [], {}
        
        # 추출 실행
        all_results = []

        # tqdm 진행도 표시
        total_specs = len(filtered_df)
        for fname in tqdm(pos_files, desc="POS 파일 처리", unit="file"):
            html_path = os.path.join(pos_folder, fname)
            file_doknr = self.extract_doknr_from_filename(fname)
            
            if not file_doknr:
                self.log.warning("doknr 추출 실패: %s", fname)
                continue
            
            # 파일명에서 hull + POS 번호 추출
            file_match = re.match(r'^(\d{4})-POS-(\d+)', fname)
            if not file_match:
                # fallback: doknr에서 POS 번호만 추출
                pos_match = re.search(r'POS-(\d+)', file_doknr)
                if not pos_match:
                    continue
                file_hull = ""
                file_pos_num = pos_match.group(1)
            else:
                file_hull = file_match.group(1)
                file_pos_num = file_match.group(2)
            
            file_pos_stripped = file_pos_num.lstrip('0')
            
            # 해당 파일의 템플릿 rows 찾기 (hull + POS 조합 매칭)
            def match_file_template(row):
                template_doknr = str(row.get('doknr', '')).strip()
                template_matnr = str(row.get('matnr', '')).strip()
                
                # 템플릿에서 POS 번호 추출
                t_match = re.search(r'POS-(\d+)', template_doknr)
                if not t_match:
                    return False
                t_pos = t_match.group(1)
                t_pos_stripped = t_pos.lstrip('0')
                
                # POS 번호 매칭 확인
                pos_matched = (t_pos == file_pos_num or t_pos_stripped == file_pos_stripped)
                if not pos_matched:
                    return False
                
                # hull 매칭 확인 (있으면)
                if file_hull:
                    t_hull = template_matnr[:4] if len(template_matnr) >= 4 else ""
                    if t_hull and t_hull != file_hull:
                        return False  # hull 불일치
                
                return True
            
            file_rows = filtered_df[filtered_df.apply(match_file_template, axis=1)]

            if file_rows.empty:
                self.log.warning("해당 POS의 템플릿 정보 없음: %s (hull=%s, POS=%s), 다음 파일로 넘어갑니다.", fname, file_hull, file_pos_num)
                continue

            self.log.info("처리 중: %s (hull=%s, %d specs)", fname, file_hull, len(file_rows))

            # 병렬 처리 (3-6 workers)
            specs_to_extract = []
            for _, row in file_rows.iterrows():
                spec = self._row_to_spec_item(row, html_path)
                specs_to_extract.append((html_path, spec, fname))

            # 병렬 추출
            workers = self.config.light_mode_workers
            if workers > 1 and len(specs_to_extract) > 1:
                self.log.debug(f"병렬 추출 시작: {workers} workers, {len(specs_to_extract)} specs")
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    # Submit all tasks
                    future_to_spec = {
                        executor.submit(self._extract_single_with_filename, html_path, spec, fname): spec
                        for html_path, spec, fname in specs_to_extract
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_spec):
                        try:
                            result = future.result()
                            all_results.append(result)
                        except Exception as e:
                            spec = future_to_spec[future]
                            self.log.error(f"병렬 추출 실패: {spec.spec_name}, error: {e}")
            else:
                # Sequential fallback (single worker or single spec)
                for html_path, spec, fname in specs_to_extract:
                    result = self.extract_single(html_path, spec)
                    result['file_name'] = fname
                    all_results.append(result)
        
        # 결과 저장
        saved = self.save_results(all_results)
        
        elapsed = time.time() - start_time
        self.log.info("=" * 60)
        self.log.info("소량 추출 완료: %.1f초 소요", elapsed)
        self.log.info("  처리 파일: %d개", len(pos_files))
        self.log.info("  추출 결과: %d건", len(all_results))
        self.log.info("=" * 60)

        # GPU VRAM 모니터링 (완료 후)
        log_gpu_memory(self.log)

        self.print_stats()
        
        return all_results, saved
    
    def _row_to_spec_item(self, row: pd.Series, file_path: str = "") -> SpecItem:
        """DataFrame row를 SpecItem으로 변환"""
        d = row.to_dict()
        return SpecItem(
            spec_name=safe_get(d, 'umgv_desc'),
            spec_code=safe_get(d, 'umgv_code'),
            equipment=safe_get(d, 'mat_attr_desc'),
            category=safe_get(d, 'umg_desc'),
            pmg_desc=safe_get(d, 'pmg_desc'),
            pmg_code=safe_get(d, 'pmg_code'),
            umg_code=safe_get(d, 'umg_code'),
            expected_unit=safe_get(d, 'umgv_uom'),
            hull=extract_hull_from_matnr(safe_get(d, 'matnr')),
            pos=extract_pos_from_doknr(safe_get(d, 'doknr')),
            matnr=safe_get(d, 'matnr'),
            extwg=safe_get(d, 'extwg'),
            file_path=file_path,
            raw_data=d
        )

    def _extract_single_with_filename(
        self,
        html_path: str,
        spec: SpecItem,
        fname: str
    ) -> Dict[str, Any]:
        """
        병렬 처리를 위한 wrapper 메서드

        extract_single을 호출하고 file_name을 추가합니다.
        Thread-safe한 단위 작업입니다.
        """
        try:
            result = self.extract_single(html_path, spec)
            result['file_name'] = fname
            return result
        except Exception as e:
            self.log.error(f"추출 실패: {fname} - {spec.spec_name}, error: {e}")
            # 빈 결과 반환
            empty_result = self._create_empty_result(spec, "ERROR")
            empty_result['file_name'] = fname
            return empty_result
    
    # =========================================================================
    # 결과 저장
    # =========================================================================
    
    def save_results(self, results: List[Dict]) -> Dict[str, str]:
        """결과 저장"""
        saved = {}
        
        if self.config.save_json:
            json_path = self._save_json(results)
            saved['json'] = json_path
        
        if self.config.save_csv:
            csv_path = self._save_csv(results)
            saved['csv'] = csv_path
        
        if self.config.save_debug_csv:
            debug_path = self._save_debug_csv(results)
            saved['debug_csv'] = debug_path
        
        return saved
    
    def _save_json(self, results: List[Dict]) -> str:
        """nested JSON 저장"""
        output_dir = self.config.output_path or "/mnt/user-data/outputs"
        ensure_parent_dir(output_dir + "/")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"extraction_result_{timestamp}.json")
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # doknr 기준 그룹화
        grouped: Dict[str, List[Dict]] = {}
        for r in results:
            doknr = r.get('doknr', '') or 'UNKNOWN'
            if doknr not in grouped:
                grouped[doknr] = []
            
            # 출력 스키마 컬럼만
            formatted = {}
            for col in OUTPUT_SCHEMA_COLUMNS:
                if col == 'evidence_fb':
                    formatted[col] = ""
                elif col in ('created_on', 'updated_on'):
                    formatted[col] = now_str
                else:
                    formatted[col] = r.get(col, "")
            
            grouped[doknr].append(formatted)
        
        # nested 구조
        nested_results = [
            {"doknr": doknr, "items": items}
            for doknr, items in grouped.items()
        ]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(nested_results, f, ensure_ascii=False, indent=2)

        total_items = sum(len(g["items"]) for g in nested_results)
        self.log.info("JSON 저장: %s (%d doknr, %d items)",
                     output_path, len(nested_results), total_items)

        # 콘솔 출력 (PRINT_JSON 옵션)
        if self.config.print_json:
            print("\n" + "=" * 80)
            print("JSON 추출 결과 (Pretty Print)")
            print("=" * 80)
            print(json.dumps(nested_results, ensure_ascii=False, indent=2))
            print("=" * 80 + "\n")

        return output_path
    
    def _save_csv(self, results: List[Dict]) -> str:
        """기본 CSV 저장"""
        output_dir = self.config.output_path or "/mnt/user-data/outputs"
        ensure_parent_dir(output_dir + "/")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"extraction_result_{timestamp}.csv")
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        rows = []
        for r in results:
            row = {}
            for col in OUTPUT_SCHEMA_COLUMNS:
                if col == 'evidence_fb':
                    row[col] = ""
                elif col in ('created_on', 'updated_on'):
                    row[col] = now_str
                else:
                    row[col] = r.get(col, "")
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=OUTPUT_SCHEMA_COLUMNS)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.log.info("CSV 저장: %s (%d건)", output_path, len(df))
        return output_path
    
    def _save_debug_csv(self, results: List[Dict]) -> str:
        """디버그 CSV 저장 (참조 정보 포함)"""
        output_dir = self.config.output_path or "/mnt/user-data/outputs"
        ensure_parent_dir(output_dir + "/")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"extraction_debug_{timestamp}.csv")
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        rows = []
        for r in results:
            row = {}
            for col in CSV_DEBUG_COLUMNS:
                if col == 'evidence_fb':
                    row[col] = ""
                elif col in ('created_on', 'updated_on'):
                    row[col] = now_str
                else:
                    row[col] = r.get(col, "")
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=CSV_DEBUG_COLUMNS)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self.log.info("Debug CSV 저장: %s (%d건)", output_path, len(df))
        return output_path
    
    def print_stats(self):
        """통계 출력"""
        self.log.info("=" * 60)
        self.log.info("추출 통계")
        self.log.info("=" * 60)
        
        total = self.stats['total']
        if total > 0:
            self.log.info("  총 항목: %d", total)
            self.log.info("  Rule 성공: %d (%.1f%%)", 
                         self.stats['rule_success'], 
                         self.stats['rule_success']/total*100)
            self.log.info("  LLM Fallback: %d (%.1f%%)", 
                         self.stats['llm_fallback'],
                         self.stats['llm_fallback']/total*100)
            self.log.info("  실패: %d (%.1f%%)", 
                         self.stats['failed'],
                         self.stats['failed']/total*100)
        
        self.log.info("=" * 60)


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    config = build_config()

    logger.info("=" * 70)
    logger.info("POS Extractor (PostgreSQL-Enhanced)")
    logger.info("=" * 70)
    logger.info("추출 모드: %s", config.extraction_mode.upper())
    logger.info("데이터 소스: %s", config.data_source_mode.upper())
    logger.info("출력: JSON=%s, CSV=%s, DB=%s",
               config.save_json, config.save_csv, config.save_to_db)
    logger.info("=" * 70)

    # 추출기 초기화
    extractor = POSExtractor(config=config)
    
    # 모드별 실행
    if config.extraction_mode == "light":
        # 소량 추출 모드
        results, saved = extractor.run_light_mode()
        
        if saved:
            logger.info("\n저장된 파일:")
            for key, path in saved.items():
                logger.info("  %s: %s", key, path)
    else:
        # 전체 추출 모드
        logger.info("전체 추출 모드 실행")
        # TODO: Full 모드 구현
        logger.warning("Full 모드는 아직 구현되지 않음")
    
    logger.info("\n실행 완료")


if __name__ == "__main__":
    main()
