# -*- coding: utf-8 -*-
"""
POS Specification Value Extractor v52 (Optimized)
==================================================

v51 기반 + 7가지 요구사항 반영

주요 개선사항 (v52):
1. Light 모드 전용 POS 경로 분리
2. 변수 기반 실행 (CLI 인자 제거)
3. 섹션 번호 vs 소수점 구분 개선 (HTML 구조 분석)
4. pos_embedding DB 연동 (사전 임베딩 활용)
5. embedding_key 개선 제안 적용
6. Light 모드 초기화 최적화 (불필요한 인덱스 빌드 스킵)
7. 출력 형식 개선 (nested JSON, 디버그 CSV)

추출 모드:
- FULL: Template의 모든 POS 추출 (기존 방식)
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
USER_OLLAMA_MODEL = "gemma3n:e4b"
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
USER_MAX_EVIDENCE_CHARS = 15000

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
# 값 검증 설정 (v2에서 추가)
# =============================================================================
ENABLE_VALUE_VALIDATION = True
# NUMERIC_VARIANCE_THRESHOLD 제거: 실제 사양값은 과거 값과 전혀 다를 수 있으므로
# 과거 값 대비 variance check는 부적절함 (v53에서 제거)
MIN_VALUE_LENGTH = 1
MAX_VALUE_LENGTH = 200

# =============================================================================
# 복합값/범위값 파싱 설정 (v2에서 추가)
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
logger = logging.getLogger("POSExtractorV52")


# ############################################################################
# 체크박스 패턴 (하드코딩 허용 - 표준화된 UI 패턴) (v2에서 추가)
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
# 출력 스키마 정의 (v51.1 기준)
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
    """안전한 float 변환 (v2에서 추가)"""
    try:
        if val is None:
            return default
        if isinstance(val, str):
            val = val.replace(',', '').strip()
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_numeric_value(text: str) -> Optional[float]:
    """텍스트에서 숫자값 추출 (v2에서 추가)"""
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
    """체크박스 선택 상태 감지 (v2에서 추가)"""
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
    """복합값 파싱 - 슬래시 구분 (v2에서 추가)"""
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
    """범위형 값 파싱 (v2에서 추가)"""
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
    max_evidence_chars: int = 15000
    
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

    # 값 검증 설정 (v2에서 추가)
    enable_value_validation: bool = True
    # numeric_variance_threshold 제거 (v53) - 과거 값 대비 variance check는 부적절
    min_value_length: int = 1
    max_value_length: int = 200

    # 복합값/범위값 파싱 (v2에서 추가)
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

        # 값 검증 설정 (v2에서 추가)
        enable_value_validation=ENABLE_VALUE_VALIDATION,
        # numeric_variance_threshold 제거 (v53)
        min_value_length=MIN_VALUE_LENGTH,
        max_value_length=MAX_VALUE_LENGTH,

        # 복합값/범위값 파싱 (v2에서 추가)
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
    found: bool = True  # v2에서 추가
    validation_status: str = ""  # v2에서 추가: "valid", "invalid", "warning", ""
    validation_message: str = ""  # v2에서 추가
    compound_values: List[Tuple[str, str]] = field(default_factory=list)  # v2에서 추가
    # v53에서 추가: POS 원문 텍스트 보존
    original_spec_name: str = ""  # POS에 적힌 그대로의 사양명 (소문자, 특수문자 등 보존)
    original_unit: str = ""  # POS에 적힌 그대로의 단위
    original_equipment: str = ""  # POS에 적힌 그대로의 장비명


# ############################################################################
# 동의어 관리자 (DB 기반 - 하드코딩 제거) (v2에서 추가)
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
# 값 검증기 (Phase 4) (v2에서 추가)
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
    개선된 사전 검사기 (v52)
    
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
        추출 결과 사전 검사 (v52 개선)
        
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
        개선된 섹션 번호 판단 (v52)
        
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
        개선된 연결 숫자 판단 (v52)
        
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
        한글 오염 여부 판단 (v52.3 개선)
        
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
    개선된 embedding_key 생성 (v52)
    
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
    시멘틱 유사도 기반 참조 매칭 시스템 (v52)
    
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
    
    # 사양값DB 힌트
    historical_values: List[str] = field(default_factory=list)  # 과거 값들
    value_patterns: List[str] = field(default_factory=list)     # 값 패턴
    
    # 유사 POS 힌트
    similar_pos_hints: List[Dict] = field(default_factory=list)  # 유사 POS 정보


class ReferenceHintEngine:
    """
    참조 힌트 엔진 (v52.3)
    
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
        """개별 사양의 힌트 구성"""
        hint = ExtractionHint(spec_name=spec_name)
        
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
        
        # 과거 값
        if historical_values:
            hint.historical_values = historical_values[:10]  # 최대 10개
            hint.value_patterns = self._extract_value_patterns(historical_values)
        
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
    HTML 문서 파싱 및 청킹 (v52 개선)
    
    개선사항:
    - 테이블 키-값 쌍 추출 강화
    - 사양명 동의어 확장
    - 정규화된 값/단위 분리
    """
    
    # 사양명 동의어 매핑 (v52.2 대폭 확장)
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
        self.table_structures = [] # v2에서 추가: 테이블 구조 정보 (헤더, 데이터 행 위치)
        self.kv_pairs = []         # 키-값 쌍 리스트
        self.text_chunks = []
        
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                self.html_content = f.read()
        
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
    
    def _extract_tables(self):
        """테이블 추출 (v2 개선: 구조 정보 포함)"""
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

                    # 헤더 행 감지 (v2: 개선된 로직)
                    has_th = row.find('th') is not None
                    if has_th or (row_idx == 0 and structure['header_row_idx'] == -1):
                        # 헤더 행인지 추가 검증
                        if self._is_likely_header_row_v2(cells):
                            structure['header_row_idx'] = len(table_data) - 1  # table_data 인덱스 사용
                            structure['header_cols'] = cells
                            structure['data_start_row'] = len(table_data)  # 다음 행부터 데이터

                    structure['col_count'] = max(structure['col_count'], len(cells))

            if table_data:
                self.tables.append(table_data)
                self.table_structures.append(structure)
    
    def _extract_kv_pairs(self):
        """
        테이블에서 키-값 쌍 추출 (v52.2 일반화 개선)
        
        개선사항:
        1. 헤더 행 감지 및 제외
        2. 값 유효성 검사 강화
        3. 중복 키-값 방지
        4. 테이블 구조 자동 인식
        5. CamelCase 키 분리 (예: "Maincomponent" → "Main component")
        """
        seen_pairs = set()  # 중복 방지용
        
        for table in self.tables:
            # 테이블 구조 분석: 첫 행이 헤더인지 확인
            is_header_row_table = self._is_header_row_table(table)
            
            for row_idx, row in enumerate(table):
                # 헤더 행 스킵 (첫 행이 헤더인 경우)
                if is_header_row_table and row_idx == 0:
                    continue
                
                # 2셀 이상 구조 처리
                if len(row) >= 2:
                    # 첫 번째 셀이 키, 두 번째가 값인 기본 구조
                    key = self._normalize_cell_key(row[0].strip())
                    value = row[1].strip()
                    
                    if self._is_valid_kv_pair(key, value, seen_pairs):
                        pair_key = f"{key}|{value}"
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            self.kv_pairs.append({
                                'key': key,
                                'value': value,
                                'row': row
                            })
                
                # 멀티 컬럼 구조 분석 (3셀 이상인 경우)
                if len(row) >= 3:
                    for i in range(len(row) - 1):
                        key = self._normalize_cell_key(row[i].strip())
                        value = row[i + 1].strip()
                        
                        if self._is_valid_kv_pair(key, value, seen_pairs):
                            pair_key = f"{key}|{value}"
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                self.kv_pairs.append({
                                    'key': key,
                                    'value': value,
                                    'row': row
                                })
    
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
    
    def _is_likely_header_row_v2(self, cells: List[str]) -> bool:
        """
        헤더 행인지 판단 (v2 개선: 키워드 카운트 기반)

        개선사항:
        - 헤더 키워드 수 카운트
        - 숫자값 비율 확인
        - 더 정교한 판단
        """
        if not cells:
            return False

        # 헤더 키워드 (v2)
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
        테이블의 첫 행이 헤더인지 판단 (v52.2 개선)

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
        테이블 검색 (v2 개선: 위치 기반 값 추출)

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
        테이블에서 사양값 찾기 (v52.1 개선)
        
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
        
        # 3. 키-값 쌍에서 유사 매칭
        best_match = None
        best_score = 0.0
        
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
        매칭 점수 계산 (v52.2 일반화 개선)
        
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
        값과 단위 분리 (v52.3 개선)
        
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
            (r'([\d.,]+)\s*(kg|ton|t)\b', 2),
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
        값이 유효한지 검사 (v52 개선 - 사양 타입별 검증)
        
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
        섹션 힌트 기반 검색 (v52.3 추가)
        
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
# 4-Stage Chunk Selection Components (v53 Enhancement)
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
    HTML 문서를 섹션 단위로 파싱 (v53 신규)

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
    다양한 소스에서 chunk 후보 생성 (v53 신규)

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
        logger: logging.Logger = None
    ):
        self.section_parser = section_parser
        self.chunk_parser = chunk_parser
        self.glossary = glossary
        self.log = logger or logging.getLogger(__name__)

    def generate_candidates(
        self,
        spec: SpecItem,
        hint: ExtractionHint = None,
        max_candidates: int = 10
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

        # 4. 동의어 확장 검색
        if self.glossary:
            candidates.extend(
                self._synonym_search(spec, hint, seen_texts)
            )

        # 5. 중복 제거 및 제한
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


class ChunkQualityScorer:
    """
    Chunk 후보의 품질 평가 (v53 신규)

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
        """키워드 점수"""
        score = 0.0
        text_upper = candidate.text.upper()

        # Spec name
        if spec.spec_name.upper() in text_upper:
            score += 0.15

        # Equipment
        if spec.equipment and spec.equipment.upper() in text_upper:
            score += 0.1

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
    LLM 기반 최적 chunk 선택 (v53 신규)

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
            response = self.llm_client.generate(prompt)
            selected_idx = self._parse_selection_response(response, len(top_candidates))

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

        # 힌트 정보
        hint_text = ""
        if hint:
            hint_parts = []
            if hint.historical_values:
                hint_parts.append(f"과거값 예시: {', '.join(hint.historical_values[:2])}")
            if hint.pos_umgv_desc:
                hint_parts.append(f"다른이름: {hint.pos_umgv_desc}")
            if hint.section_num:
                hint_parts.append(f"예상 섹션: {hint.section_num}")

            if hint_parts:
                hint_text = "\n힌트: " + ", ".join(hint_parts)

        prompt = f"""당신은 POS 문서에서 사양값을 추출하는 전문가입니다.
아래의 후보 chunk 중에서 "{spec.spec_name}" 사양값을 추출하기에 가장 적합한 chunk를 선택하세요.

**추출 대상:**
- 사양명: {spec.spec_name}
- 장비: {spec.equipment or '미지정'}
- 예상단위: {spec.expected_unit or '미지정'}{hint_text}

**후보 Chunks:**
{candidates_text}

**작업:**
위 후보 중 "{spec.spec_name}" 값을 추출하기에 가장 적합한 chunk의 번호를 선택하세요.
주의: Section 2 (TECHNICAL PARTICULARS)의 chunk가 일반적으로 가장 정확합니다.

**출력 형식:**
정확히 다음 형식으로만 응답하세요:
SELECTED: [번호]
CONFIDENCE: [0.0-1.0]

예:
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
    짧은 chunk를 주변 컨텍스트로 확장 (v53 신규)

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
    Rule 기반 사양값 추출 (v52.3 개선)
    
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

        # v53 Enhanced Chunk Selection 컴포넌트 (lazy loading)
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
        Enhanced 4-stage chunk selection으로 추출 시도 (v53)

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
                return ExtractionResult(
                    spec_item=spec,
                    value=value,
                    unit=unit or spec.expected_unit,
                    chunk=chunk_text[:500],  # 500자 제한
                    method="RULE_ENHANCED_CHUNK",
                    confidence=confidence,
                    reference_source=f"enhanced:{best_candidate.source}"
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

        Returns:
            (value, unit) 튜플
        """
        if not raw_value:
            return "", ""

        raw_value = raw_value.strip()

        # 숫자+단위 패턴
        match = re.match(r'^([0-9.,\-~\s]+)\s*([a-zA-Z°℃%/]+.*)?$', raw_value)
        if match:
            value = match.group(1).strip()
            unit = match.group(2).strip() if match.group(2) else ""
            return value, unit

        # 순수 숫자
        if re.match(r'^[0-9.,\-~\s]+$', raw_value):
            return raw_value.strip(), ""

        # 텍스트 값 (숫자 없음)
        return raw_value, ""

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
    
    def extract(
        self,
        parser: HTMLChunkParser,
        spec: SpecItem,
        hint: ExtractionHint = None
    ) -> Optional[ExtractionResult]:
        """
        테이블에서 사양값 추출 (v53: Enhanced chunk selection 통합)

        전략 순서:
        0. [v53 NEW] Enhanced 4-stage chunk selection
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

        # v53: Enhanced chunk selection 시도
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
                            return ExtractionResult(
                                spec_item=spec,
                                value=value,
                                unit=unit or spec.expected_unit,
                                chunk=chunk,
                                method="RULE_SECTION_HINT",
                                confidence=confidence,
                                reference_source=f"section:{hint.section_num}"
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
                        return ExtractionResult(
                            spec_item=spec,
                            value=value,
                            unit=unit or spec.expected_unit,
                            chunk=chunk,
                            method="RULE_TABLE_DIRECT",
                            confidence=confidence,
                            reference_source=f"table_text:{hint.table_text}" if hint else ""
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
                        return ExtractionResult(
                            spec_item=spec,
                            value=value,
                            unit=unit or spec.expected_unit,
                            chunk=chunk,
                            method="RULE_TEXT_SEARCH",
                            confidence=confidence,
                            reference_source=f"table_text:{hint.table_text}" if hint else ""
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
                        return ExtractionResult(
                            spec_item=spec,
                            value=value,
                            unit=unit or spec.expected_unit,
                            chunk=chunk,
                            method="RULE_TEXT_FALLBACK",
                            confidence=confidence
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
                        return ExtractionResult(
                            spec_item=spec,
                            value=value,
                            unit=unit or spec.expected_unit,
                            chunk=chunk,
                            method="RULE_TABLE_FALLBACK",
                            confidence=confidence
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
        신뢰도 계산 (v52.3 개선 - 힌트 활용)
        
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
# UnifiedLLMClient (v2에서 추가 - Ollama 전용)
# =============================================================================

class UnifiedLLMClient:
    """
    Ollama LLM 클라이언트 (포트 로테이션 지원)

    개선사항 (v2):
    - 포트 로테이션으로 부하 분산
    - 스레드 안전 포트 선택
    - 토큰 추적
    """

    def __init__(self, ollama_host: str = "127.0.0.1", ollama_ports: List[int] = None,
                 model: str = "gemma3n:e4b", timeout: int = 180,
                 temperature: float = 0.0, max_retries: int = 3,
                 retry_sleep: float = 1.5, rate_limit: float = 0.3,
                 logger: logging.Logger = None):
        self.host = ollama_host
        self.ports = ollama_ports or [11434]
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.rate_limit = rate_limit
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
            "options": {"temperature": self.temperature}
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
                           html_context: str, use_voting: bool = True) -> Dict[str, Any]:
        """
        추출된 값을 LLM으로 검증

        Args:
            spec: 사양 항목
            extracted_value: 추출된 값
            extracted_unit: 추출된 단위
            html_context: HTML 컨텍스트 (테이블 등)
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
        prompt = self._build_validation_prompt(spec, extracted_value, extracted_unit, html_context)

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
                                 extracted_unit: str, html_context: str) -> str:
        """검증 프롬프트 생성"""
        prompt = f"""You are a technical specification validator.

**Task**: Validate if the extracted value is correct for the given specification.

**Specification Name**: {spec.spec_name}
**Equipment**: {spec.equipment if spec.equipment else 'N/A'}
**Expected Unit**: {spec.expected_unit if spec.expected_unit else 'N/A'}

**Extracted Value**: {extracted_value}
**Extracted Unit**: {extracted_unit}

**HTML Context**:
```
{html_context[:2000]}
```

**Instructions**:
1. Check if the extracted value matches the specification name in the context
2. Check if the unit is appropriate
3. Check if the value format is correct (number, text, etc.)

**Output Format** (JSON):
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "llm_extracted_value": "corrected value if different",
  "llm_extracted_unit": "corrected unit if different",
  "reason": "brief explanation"
}}

**Output**:"""

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
        timeout: int = 120,
        logger: logging.Logger = None,
        llm_client: 'UnifiedLLMClient' = None,
        use_voting: bool = True,
        glossary: LightweightGlossaryIndex = None,
        enable_enhanced_chunk_selection: bool = True
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

        # v53 Enhanced Chunk Selection 컴포넌트 (lazy loading)
        self.section_parser = None
        self.candidate_generator = None
        self.quality_scorer = None
        self.llm_chunk_selector = None  # LLM 기반 chunk 선택
        self.chunk_expander = None
        self._enhanced_components_initialized = False
    
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
        LLM으로 사양값 추출 (v53: Enhanced chunk selection 통합)

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

        # v53: Enhanced chunk selection 시도
        chunk = None
        if self.enable_enhanced_chunk_selection:
            chunk = self._get_enhanced_chunk(parser, spec, hint, max_chunk_chars)

        # Fallback to legacy chunk selection
        if not chunk:
            chunk = self._get_relevant_chunk(parser, spec, max_chunk_chars, hint)

        if not chunk:
            self.log.debug("LLM Fallback 스킵: 관련 청크 없음 (spec=%s)", spec.spec_name)
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
                    vote_k=3,  # 3번 호출하여 투표
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
        # 사양 목록 구성
        spec_list = []
        for idx, spec in enumerate(specs, 1):
            hint = hints.get(spec.spec_name)
            hint_text = ""

            if hint:
                hint_parts = []
                if hint.historical_values:
                    examples = ', '.join(hint.historical_values[:2])
                    hint_parts.append(f"예시: {examples}")
                if hint.pos_umgv_desc and hint.pos_umgv_desc != spec.spec_name:
                    hint_parts.append(f"다른이름: {hint.pos_umgv_desc}")

                if hint_parts:
                    hint_text = f" ({', '.join(hint_parts)})"

            spec_list.append(
                f"{idx}. 사양명: {spec.spec_name}, "
                f"장비: {spec.equipment or '미지정'}, "
                f"예상단위: {spec.expected_unit or '미지정'}{hint_text}"
            )

        spec_section = "\n".join(spec_list)

        prompt = f"""당신은 POS 문서에서 여러 사양값을 한 번에 추출하는 전문가입니다.

## 추출 대상 ({len(specs)}개)
{spec_section}

## 문서 내용
```
{chunk}
```

## 작업
위 문서에서 각 사양의 값을 찾아 추출하세요.
값을 찾지 못한 경우 빈 문자열("")로 표시하세요.

## 출력 형식 (JSON Array)
정확히 다음 형식으로만 응답하세요:
```json
[
  {{"spec_index": 1, "value": "추출된값1", "unit": "단위1", "confidence": 0.9}},
  {{"spec_index": 2, "value": "추출된값2", "unit": "단위2", "confidence": 0.8}},
  ...
]
```

중요: 반드시 {len(specs)}개의 결과를 JSON Array로 반환하세요."""

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
        Enhanced 4-stage chunk selection으로 chunk 추출 (v53)

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
        관련 청크 추출 (v52.4 - Legacy fallback)

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
                # 문서 앞부분 (보통 사양 정보가 있음)
                chunks.append(f"[Document excerpt]\n{full_text[:3000]}")
        
        # 청크 결합 및 크기 제한
        combined = '\n---\n'.join(chunks)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."
        
        return combined
    
    def _build_prompt(self, spec: SpecItem, chunk: str, hint: ExtractionHint = None) -> str:
        """
        LLM 프롬프트 생성 (v52.3 힌트 포함)
        
        힌트 정보:
        - historical_values: 과거 값 예시
        - value_patterns: 값 형식
        - pos_umgv_desc: POS에서의 사양명 (동의어)
        """
        # 힌트 정보 구성
        hint_section = ""
        if hint:
            hint_parts = []
            
            if hint.historical_values:
                examples = ', '.join(hint.historical_values[:3])
                hint_parts.append(f"- 과거 값 예시: {examples}")
            
            if hint.pos_umgv_desc and hint.pos_umgv_desc != spec.spec_name:
                hint_parts.append(f"- POS에서 사용되는 다른 이름: {hint.pos_umgv_desc}")
            
            if hint.section_num:
                hint_parts.append(f"- 참조 섹션: {hint.section_num[:50]}")
            
            if hint_parts:
                hint_section = "\n## 참조 힌트\n" + "\n".join(hint_parts) + "\n"
        
        prompt = f"""당신은 POS(Purchase Order Specification) 문서에서 사양값을 추출하는 전문가입니다.

## 추출 대상
- 사양명: {spec.spec_name}
- 장비: {spec.equipment or '(미지정)'}
- 예상 단위: {spec.expected_unit or '(미지정)'}
{hint_section}
## 문서 내용
```
{chunk}
```

## 작업
위 문서에서 "{spec.spec_name}" 사양의 값을 찾아 추출하세요.

## 출력 형식 (JSON)
정확히 다음 형식으로만 응답하세요:
```json
{{
  "value": "추출된값",
  "unit": "단위",
  "confidence": 0.0~1.0,
  "original_spec_name": "POS에 적힌 사양명 그대로",
  "original_unit": "POS에 적힌 단위 그대로",
  "original_equipment": "POS에 적힌 장비명 그대로"
}}
```

값을 찾지 못한 경우:
```json
{{"value": "", "unit": "", "confidence": 0.0, "original_spec_name": "", "original_unit": "", "original_equipment": ""}}
```

주의사항:
1. 값만 추출하고 사양명은 포함하지 마세요
2. 숫자와 단위를 분리하세요 (예: "70 m3/h" → value: "70", unit: "m3/h")
3. 여러 값이 있으면 가장 관련성 높은 것을 선택하세요
4. 확실하지 않으면 confidence를 낮게 설정하세요
5. 참조 힌트의 과거 값 예시를 참고하여 비슷한 형식으로 추출하세요
6. **중요**: original_spec_name, original_unit, original_equipment는 POS 문서에 적힌 그대로를 추출하세요
   - 대소문자, 띄어쓰기, 특수문자 등을 정확히 보존하세요
   - 예: POS에 "capacity"로 적혀있으면 "capacity", "CAPACITY"로 적혀있으면 "CAPACITY"
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
    
    def _parse_llm_response(
        self,
        response: str,
        spec: SpecItem,
        chunk: str
    ) -> Optional[ExtractionResult]:
        """LLM 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]+\}', response)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            value = data.get("value", "").strip()
            unit = data.get("unit", "").strip()
            confidence = float(data.get("confidence", 0.0))

            # v53: POS 원문 텍스트 추출
            original_spec_name = data.get("original_spec_name", "").strip()
            original_unit = data.get("original_unit", "").strip()
            original_equipment = data.get("original_equipment", "").strip()

            if not value:
                return None

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


# =============================================================================
# POSExtractorV52 메인 클래스
# =============================================================================

class POSExtractorV52:
    """
    POS 사양값 추출기 v52 (최적화 버전)
    
    주요 특징:
    - Light 모드 최적화 (빠른 초기화)
    - pos_embedding DB 연동
    - 개선된 Pre-Check
    - 변수 기반 실행
    """
    
    def __init__(
        self,
        glossary_path: str = "",
        specdb_path: str = "",
        config: Config = None,
    ):
        self.config = config or build_config()
        self.log = logging.getLogger("POSExtractorV52")
        
        # 모드별 초기화
        if self.config.extraction_mode == "light":
            self._init_light_mode(glossary_path, specdb_path)
        elif self.config.extraction_mode == "verify":
            self._init_verify_mode(glossary_path, specdb_path)
        else:
            self._init_full_mode(glossary_path, specdb_path)
        
        # 공통 컴포넌트
        self.pre_checker = ImprovedPreChecker()
        self.rule_extractor = RuleBasedExtractor(self.glossary, self.specdb)

        # ValueValidator 초기화 (v2에서 추가)
        self.value_validator = ValueValidator(self.config)
        self.log.info("ValueValidator 초기화 완료")

        # UnifiedLLMClient 초기화 (모든 LLM 호출에 사용)
        self.llm_client = None
        self.llm_validator = None
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
                logger=self.log
            )
            self.log.info("UnifiedLLMClient 초기화: %s (ports: %s)",
                         self.config.ollama_model, self.config.ollama_ports)

            # LLMValidator 초기화 (모든 추출 결과 검증)
            self.llm_validator = LLMValidator(self.llm_client, self.config, self.log)
            self.log.info("LLMValidator 초기화 완료 (모든 추출 결과 LLM 검증)")

        # LLM Fallback 초기화 (UnifiedLLMClient와 연동하여 Voting 지원)
        self.llm_fallback = None
        if self.config.use_llm and self.config.enable_llm_fallback:
            self.llm_fallback = LLMFallbackExtractor(
                ollama_host=self.config.ollama_host,
                ollama_ports=self.config.ollama_ports,
                model=self.config.ollama_model,
                timeout=self.config.ollama_timeout,
                logger=self.log,
                llm_client=self.llm_client,  # UnifiedLLMClient 전달
                use_voting=self.config.vote_enabled  # Config의 voting 설정 사용
            )
            voting_status = "Voting 활성화" if self.config.vote_enabled else "단일 호출"
            self.log.info("LLM Fallback 초기화: %s (ports: %s, %s)",
                         self.config.ollama_model, self.config.ollama_ports, voting_status)
        
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
        Light 모드 초기화 (최적화)
        
        DATA_SOURCE_MODE에 따라:
        - "file": 파일에서 용어집/사양값DB 로드
        - "db": PostgreSQL에서 용어집/사양값DB 로드
        
        v52.3: ReferenceHintEngine 추가 (용어집/사양값DB 참조 힌트)
        """
        self.log.info("Light 모드 초기화 시작 (최적화)")
        start = time.time()
        
        # DATA_SOURCE_MODE에 따라 용어집/사양값DB 로드
        if self.config.data_source_mode == "db":
            # DB 모드: PostgreSQL에서 로드
            self.log.info("데이터 소스: DB 모드")

            # pos_embedding DB 연결
            try:
                self.pg_loader = PostgresEmbeddingLoader(self.config, self.log)
            except Exception as e:
                self.log.error("PostgreSQL 연결 실패: %s", e)
                self.log.error("DB 모드에서는 PostgreSQL 연결이 필수입니다. 프로그램을 종료합니다.")
                raise RuntimeError(f"PostgreSQL 연결 실패: {e}")

            # 용어집 로드 (pos_dict 테이블)
            glossary_df = self.pg_loader.load_glossary_from_db()
            if glossary_df.empty:
                self.log.error("용어집(pos_dict) 로드 실패: 데이터가 비어있습니다.")
                raise RuntimeError("용어집 로드 실패")
            self.glossary = LightweightGlossaryIndex(df=glossary_df)

            # 사양값DB 로드 (umgv_fin 테이블)
            specdb_df = self.pg_loader.load_specdb_from_db()
            if specdb_df.empty:
                self.log.error("사양값DB(umgv_fin) 로드 실패: 데이터가 비어있습니다.")
                raise RuntimeError("사양값DB 로드 실패")
            self.specdb = LightweightSpecDBIndex(df=specdb_df)
        else:
            # 파일 모드: 로컬 파일에서 로드
            self.log.info("데이터 소스: 파일 모드")
            self.pg_loader = None

            gpath = glossary_path or self.config.glossary_path
            if not gpath or not os.path.exists(gpath):
                self.log.error(f"용어집 파일 없음: {gpath}")
                raise RuntimeError(f"용어집 파일 없음: {gpath}")
            self.glossary = LightweightGlossaryIndex(file_path=gpath)

            spath = specdb_path or self.config.specdb_path
            if not spath or not os.path.exists(spath):
                self.log.error(f"사양값DB 파일 없음: {spath}")
                raise RuntimeError(f"사양값DB 파일 없음: {spath}")
            self.specdb = LightweightSpecDBIndex(file_path=spath)

            # 파일 모드에서 임베딩이 필요한 경우에만 DB 연결 시도
            if self.config.use_precomputed_embeddings:
                try:
                    self.pg_loader = PostgresEmbeddingLoader(self.config, self.log)
                except Exception as e:
                    self.log.warning("PostgreSQL 연결 실패 (임베딩 사용 불가): %s", e)
                    self.pg_loader = None
        
        # SynonymManager 초기화 (Lazy loading으로 변경 - 초기화 시간 단축)
        self.synonym_manager = None
        self._synonym_manager_initialized = False
        self.log.info("SynonymManager: Lazy loading 모드 (첫 사용 시 초기화)")

        # ReferenceHintEngine 초기화 (Lazy loading - 첫 사용 시 초기화)
        self.hint_engine = None
        self._hint_engine_initialized = False
        self.log.info("ReferenceHintEngine: Lazy loading 모드 (첫 사용 시 초기화)")

        # SemanticMatcher (Lazy loading - 첫 사용 시 초기화)
        self.semantic_matcher = None
        self._semantic_matcher_initialized = False
        self.log.info("SemanticMatcher: Lazy loading 모드 (첫 사용 시 초기화)")
        
        elapsed = time.time() - start
        self.log.info("Light 모드 초기화 완료: %.2f초", elapsed)

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
        Full 모드 초기화
        
        DATA_SOURCE_MODE에 따라:
        - "file": 파일에서 용어집/사양값DB 로드
        - "db": PostgreSQL에서 용어집/사양값DB 로드
        """
        self.log.info("Full 모드 초기화 시작")
        start = time.time()

        # DATA_SOURCE_MODE에 따라 용어집/사양값DB 로드
        if self.config.data_source_mode == "db":
            # DB 모드: PostgreSQL에서 로드
            self.log.info("데이터 소스: DB 모드")

            # pos_embedding DB 연결
            try:
                self.pg_loader = PostgresEmbeddingLoader(self.config, self.log)
            except Exception as e:
                self.log.error("PostgreSQL 연결 실패: %s", e)
                self.log.error("DB 모드에서는 PostgreSQL 연결이 필수입니다. 프로그램을 종료합니다.")
                raise RuntimeError(f"PostgreSQL 연결 실패: {e}")

            # 용어집 로드 (pos_dict 테이블)
            glossary_df = self.pg_loader.load_glossary_from_db()
            if glossary_df.empty:
                self.log.error("용어집(pos_dict) 로드 실패: 데이터가 비어있습니다.")
                raise RuntimeError("용어집 로드 실패")
            self.glossary = LightweightGlossaryIndex(df=glossary_df)

            # 사양값DB 로드 (umgv_fin 테이블)
            specdb_df = self.pg_loader.load_specdb_from_db()
            if specdb_df.empty:
                self.log.error("사양값DB(umgv_fin) 로드 실패: 데이터가 비어있습니다.")
                raise RuntimeError("사양값DB 로드 실패")
            self.specdb = LightweightSpecDBIndex(df=specdb_df)
        else:
            # 파일 모드: 로컬 파일에서 로드
            self.log.info("데이터 소스: 파일 모드")
            self.pg_loader = None

            gpath = glossary_path or self.config.glossary_path
            if not gpath or not os.path.exists(gpath):
                self.log.error(f"용어집 파일 없음: {gpath}")
                raise RuntimeError(f"용어집 파일 없음: {gpath}")
            self.glossary = LightweightGlossaryIndex(file_path=gpath)

            spath = specdb_path or self.config.specdb_path
            if not spath or not os.path.exists(spath):
                self.log.error(f"사양값DB 파일 없음: {spath}")
                raise RuntimeError(f"사양값DB 파일 없음: {spath}")
            self.specdb = LightweightSpecDBIndex(file_path=spath)

            # 파일 모드에서 임베딩이 필요한 경우에만 DB 연결 시도
            if self.config.use_precomputed_embeddings:
                try:
                    self.pg_loader = PostgresEmbeddingLoader(self.config, self.log)
                except Exception as e:
                    self.log.warning("PostgreSQL 연결 실패 (임베딩 사용 불가): %s", e)
                    self.pg_loader = None

        # SynonymManager 초기화 (Lazy loading으로 변경 - 초기화 시간 단축)
        self.synonym_manager = None
        self._synonym_manager_initialized = False
        self.log.info("SynonymManager: Lazy loading 모드 (첫 사용 시 초기화)")

        # SemanticMatcher (Lazy loading - 첫 사용 시 초기화)
        self.semantic_matcher = None
        self._semantic_matcher_initialized = False
        self.log.info("SemanticMatcher: Lazy loading 모드 (첫 사용 시 초기화)")
        
        elapsed = time.time() - start
        self.log.info("Full 모드 초기화 완료: %.2f초", elapsed)
    
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
        
        v52.3: ReferenceHintEngine 힌트 활용
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

                    # HTML 컨텍스트 준비 (최대 2000자)
                    html_context = chunk_context if chunk_context else parser.get_context_for_value(result.value)
                    if len(html_context) > 2000:
                        html_context = html_context[:2000]

                    # LLM 검증 (voting 활성화)
                    validation = self.llm_validator.validate_extraction(
                        spec=spec,
                        extracted_value=result.value,
                        extracted_unit=result.unit,
                        html_context=html_context,
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
                    self.log.debug("LLM Fallback Pre-Check 실패: %s -> %s (errors: %s)",
                                 spec.spec_name, llm_result.value, errors)
            else:
                self.log.debug("LLM Fallback 결과 없음: %s", spec.spec_name)
        
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
            'pos_chunk': result.chunk[:500] if result.chunk else '',
            # v53: POS 원문 텍스트 보존 (대소문자, 특수문자 등 그대로)
            'pos_mat_attr_desc': result.original_equipment or spec.equipment,
            'pos_umgv_desc': result.original_spec_name or spec.spec_name,
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
        파일 목록 기반 템플릿 필터링 (v52.4 개선)
        
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
        
        for fname in pos_files:
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
            
            for _, row in file_rows.iterrows():
                spec = self._row_to_spec_item(row, html_path)
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
        self.log.info("v52 추출 통계")
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
    logger.info("POS Extractor v52 (Optimized)")
    logger.info("=" * 70)
    logger.info("추출 모드: %s", config.extraction_mode.upper())
    logger.info("데이터 소스: %s", config.data_source_mode.upper())
    logger.info("출력: JSON=%s, CSV=%s, DB=%s", 
               config.save_json, config.save_csv, config.save_to_db)
    logger.info("=" * 70)
    
    # 추출기 초기화
    extractor = POSExtractorV52(config=config)
    
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