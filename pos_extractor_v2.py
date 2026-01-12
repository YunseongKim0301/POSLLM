# -*- coding: utf-8 -*-
"""
POS Specification Value Extractor (Enhanced Version)
=====================================================

조선 산업용 POS 문서에서 사양값을 추출하는 고도화된 시스템

핵심 개선사항:
1. 테이블 추출 로직 개선 - 헤더/데이터 구분, 열 위치 기반 추출
2. 템플릿 매칭 수정 - Hull+POS 번호 조합, extwg 보조 매칭
3. 동의어 매핑 DB 로드 - 하드코딩 제거, pos_dict에서 동적 구축
4. 값 검증 로직 추가 - 숫자형 검증, 단위 비교, 과거 값 범위 검증
5. 4-Layer 취소선 제거
6. 체크박스 맥락 인식 (Y/N/Q 패턴)
7. 복합값/범위형 처리 (토글 방식)
8. 대표호선 캐싱 (검증 강화)
9. NOT_FOUND 명시적 처리
10. Claude API 지원

설계 원칙:
- 하드코딩 최소화: 체크박스 패턴 외 모든 매핑/동의어는 DB에서 로드
- 용어집(pos_dict)의 umgv_desc ↔ pos_umgv_desc를 동의어로 활용
- 사양값DB(umgv_fin)의 과거 값을 힌트/검증에 활용

Author: Claude AI Assistant
Date: 2026-01-12
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
import hashlib
import logging
import traceback
import threading
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# ============================================================================
# 서드파티 라이브러리 임포트
# ============================================================================

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARNING] pandas not installed")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("[WARNING] beautifulsoup4 not installed")

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
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        """tqdm 대체 클래스"""
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
        def update(self, n=1):
            self.n += n
        def set_postfix(self, **kwargs):
            pass
        def set_postfix_str(self, s):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ============================================================================
# 로깅 설정
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("POSExtractor")


# ############################################################################
# 설정값 (Settings) - 하드코딩 최소화
# ############################################################################

# =============================================================================
# [1] 모드 설정
# =============================================================================
EXTRACTION_MODE = "light"  # "full", "light", "verify"
DATA_SOURCE_MODE = "db"    # "file" 또는 "db"

# =============================================================================
# [2] LLM 백엔드 설정
# =============================================================================
LLM_BACKEND = "ollama"  # "ollama" 또는 "claude"

# =============================================================================
# [3] 복합형 사양값 처리 설정 (토글)
# =============================================================================
SPLIT_COMPOUND_VALUES = False  # True: 분리, False: 통째로 저장
SPLIT_RANGE_VALUES = False     # True: 범위 분리, False: 통째로 저장

# =============================================================================
# [4] Light 모드 전용 설정
# =============================================================================
LIGHT_MODE_MAX_FILES = 500
LIGHT_MODE_DEFAULT_SPECS = 6
LIGHT_MODE_VOTING_DISABLED = True
LIGHT_MODE_AUDIT_DISABLED = True
LIGHT_MODE_CHECKPOINT_DISABLED = True

# =============================================================================
# [5] Full 모드 전용 설정
# =============================================================================
FULL_MODE_BATCH_SIZE = 15
FULL_MODE_CHECKPOINT_INTERVAL = 50
FULL_MODE_VOTING_ENABLED = True
FULL_MODE_VOTE_K = 2
FULL_MODE_AUDIT_ENABLED = True

# =============================================================================
# [6] Verify 모드 전용 설정
# =============================================================================
VERIFY_MODE_BATCH_SIZE = 10
VERIFY_MODE_CONFIDENCE_THRESHOLD = 0.7

# =============================================================================
# [7] Ollama LLM 설정
# =============================================================================
USER_OLLAMA_MODEL = "gemma3:27b"
USER_OLLAMA_HOST = "127.0.0.1"
USER_OLLAMA_PORTS = [11434, 11436, 11438, 11440]
USER_OLLAMA_TIMEOUT_SEC = 180
USER_LLM_TEMPERATURE = 0.0
USER_LLM_RATE_LIMIT_SEC = 0.2
USER_LLM_MAX_RETRIES = 3
USER_LLM_RETRY_SLEEP_SEC = 1.5

# =============================================================================
# [8] Claude API 설정
# =============================================================================
USER_CLAUDE_MODEL = "claude-sonnet-4-20250514"
USER_CLAUDE_MAX_TOKENS = 1024
USER_CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# =============================================================================
# [9] 병렬 처리 설정
# =============================================================================
USER_ENABLE_PARALLEL = True
USER_NUM_WORKERS = 4
USER_LLM_WORKERS = 2

# =============================================================================
# [10] PostgreSQL 설정
# =============================================================================
USER_DB_HOST = "10.131.132.116"
USER_DB_PORT = 5432
USER_DB_NAME = "managesys"
USER_DB_USER = "postgres"
USER_DB_PASSWORD = "pmg_umg!@"

# DB 테이블명 (스키마 없이 테이블명만)
USER_DB_TABLE_GLOSSARY = "pos_dict"
USER_DB_TABLE_SPECDB = "umgv_fin"
USER_DB_TABLE_TEMPLATE = "ext_tmpl"
USER_DB_TABLE_RESULT = "ext_rslt"

# =============================================================================
# [11] 파일 모드 경로 설정
# =============================================================================
USER_BASE_FOLDER = "/workspace/pos/phase3/phase3_formatted_new"
USER_GLOSSARY_PATH = "/mnt/project/용어집.txt"
USER_SPEC_PATH = "/mnt/project/사양값추출_template.txt"
USER_SPECDB_PATH = "/mnt/project/사양값DB.txt"
USER_OUTPUT_PATH = "/home/claude/results"
USER_PARTIAL_OUTPUT_PATH = "/home/claude/logging"

# =============================================================================
# [12] 추출 설정
# =============================================================================
USER_RULE_CONF_THRESHOLD = 0.72
USER_FORCE_LLM_ON_ALL = False
USER_MAX_EVIDENCE_CHARS = 15000
USER_EVIDENCE_MAX_TABLES = 10

# =============================================================================
# [13] 대표호선 시스템 설정
# =============================================================================
SERIES_HULL_RANGE = 10
USE_REPRESENTATIVE_HULL_HINT = True
REPRESENTATIVE_HULL_CACHE_ENABLED = True
CACHE_MIN_CONFIDENCE = 0.80  # 캐싱 최소 신뢰도 (Phase 4: 0.7→0.8 상향)

# =============================================================================
# [14] 값 검증 설정 (Phase 4)
# =============================================================================
ENABLE_VALUE_VALIDATION = True
NUMERIC_VARIANCE_THRESHOLD = 0.5  # 과거 값 대비 50% 이내 허용
MIN_VALUE_LENGTH = 1
MAX_VALUE_LENGTH = 200

# =============================================================================
# [15] 디버그 설정
# =============================================================================
USER_DEBUG = True


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


# ############################################################################
# 유틸리티 함수
# ############################################################################

def norm(val: Any) -> str:
    """값 정규화 - None이나 nan을 빈 문자열로 변환"""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


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


def extract_hull_from_filename(filename: str) -> str:
    """파일명에서 Hull 번호 추출 (예: 5533)"""
    match = re.search(r'^(\d{4})', filename)
    return match.group(1) if match else ""


def extract_doknr_from_filename(filename: str) -> str:
    """파일명에서 문서번호(doknr) 추출 (예: 5533-POS-0070101)"""
    match = re.match(r'(\d{4}-POS-\d+)', filename)
    return match.group(1) if match else ""


def extract_pos_number(doknr: str) -> str:
    """doknr에서 POS 번호만 추출 (예: 5533-POS-0070101 → 0070101)"""
    match = re.search(r'POS-(\d+)', doknr)
    return match.group(1) if match else ""


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


def is_numeric_spec(spec_name: str, value_format: str = "") -> bool:
    """숫자형 사양인지 판단"""
    numeric_keywords = [
        'capacity', 'head', 'power', 'pressure', 'temperature', 
        'flow', 'speed', 'rpm', 'voltage', 'frequency', 'weight',
        'qty', 'quantity', 'no.', 'number', 'length', 'width', 'height',
        'diameter', 'thickness', 'volume', 'area', 'mcr', 'ncr'
    ]
    spec_lower = spec_name.lower()
    return any(kw in spec_lower for kw in numeric_keywords)


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
    """복합값 파싱 (슬래시 구분)"""
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
    
    return [(raw_value, "")]


# ############################################################################
# 데이터 클래스
# ############################################################################

@dataclass
class Config:
    """추출기 설정"""
    mode: str = EXTRACTION_MODE
    data_source: str = DATA_SOURCE_MODE
    llm_backend: str = LLM_BACKEND
    
    split_compound_values: bool = SPLIT_COMPOUND_VALUES
    split_range_values: bool = SPLIT_RANGE_VALUES
    
    ollama_model: str = USER_OLLAMA_MODEL
    ollama_host: str = USER_OLLAMA_HOST
    ollama_ports: List[int] = field(default_factory=lambda: USER_OLLAMA_PORTS.copy())
    ollama_timeout: int = USER_OLLAMA_TIMEOUT_SEC
    llm_temperature: float = USER_LLM_TEMPERATURE
    llm_rate_limit: float = USER_LLM_RATE_LIMIT_SEC
    llm_max_retries: int = USER_LLM_MAX_RETRIES
    llm_retry_sleep: float = USER_LLM_RETRY_SLEEP_SEC
    
    claude_model: str = USER_CLAUDE_MODEL
    claude_max_tokens: int = USER_CLAUDE_MAX_TOKENS
    claude_api_key: str = USER_CLAUDE_API_KEY
    
    enable_parallel: bool = USER_ENABLE_PARALLEL
    num_workers: int = USER_NUM_WORKERS
    llm_workers: int = USER_LLM_WORKERS
    
    db_host: str = USER_DB_HOST
    db_port: int = USER_DB_PORT
    db_name: str = USER_DB_NAME
    db_user: str = USER_DB_USER
    db_password: str = USER_DB_PASSWORD
    
    base_folder: str = USER_BASE_FOLDER
    glossary_path: str = USER_GLOSSARY_PATH
    spec_path: str = USER_SPEC_PATH
    specdb_path: str = USER_SPECDB_PATH
    output_path: str = USER_OUTPUT_PATH
    partial_output_path: str = USER_PARTIAL_OUTPUT_PATH
    
    rule_conf_threshold: float = USER_RULE_CONF_THRESHOLD
    force_llm_on_all: bool = USER_FORCE_LLM_ON_ALL
    max_evidence_chars: int = USER_MAX_EVIDENCE_CHARS
    evidence_max_tables: int = USER_EVIDENCE_MAX_TABLES
    
    series_hull_range: int = SERIES_HULL_RANGE
    use_representative_hull_hint: bool = USE_REPRESENTATIVE_HULL_HINT
    representative_hull_cache_enabled: bool = REPRESENTATIVE_HULL_CACHE_ENABLED
    cache_min_confidence: float = CACHE_MIN_CONFIDENCE
    
    enable_value_validation: bool = ENABLE_VALUE_VALIDATION
    numeric_variance_threshold: float = NUMERIC_VARIANCE_THRESHOLD
    
    debug: bool = USER_DEBUG


@dataclass
class SpecItem:
    """추출할 사양 항목"""
    pmg_code: str = ""
    pmg_desc: str = ""
    umg_code: str = ""
    umg_desc: str = ""
    extwg: str = ""
    extwg_desc: str = ""
    matnr: str = ""
    doknr: str = ""
    umgv_code: str = ""
    umgv_desc: str = ""
    umgv_uom: str = ""
    spec_name: str = ""
    existing_value: str = ""
    value_format: str = ""  # 값 형식 (숫자형, 문자형 등)


@dataclass
class ExtractionHint:
    """추출 힌트 정보 (DB에서 동적 로드)"""
    match_tier: int = 0
    
    # 동의어 (DB에서 로드)
    synonyms: List[str] = field(default_factory=list)
    
    # 용어집 정보
    pos_umgv_desc: str = ""
    section_num: str = ""
    table_text: str = ""
    value_format: str = ""
    
    # 과거 값 정보 (사양값DB에서)
    historical_values: List[str] = field(default_factory=list)
    historical_units: List[str] = field(default_factory=list)
    historical_numeric_values: List[float] = field(default_factory=list)
    
    # 대표호선 정보
    representative_hull: str = ""
    representative_value: str = ""
    representative_unit: str = ""
    representative_position: str = ""


@dataclass
class ExtractionResult:
    """추출 결과"""
    value: str = ""
    unit: str = ""
    confidence: float = 0.0
    method: str = ""
    found: bool = True
    evidence: str = ""
    position: str = ""
    compound_values: List[Tuple[str, str]] = field(default_factory=list)
    validation_status: str = ""  # "valid", "invalid", "warning", ""
    validation_message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "method": self.method,
            "found": self.found,
            "evidence": self.evidence[:200] if self.evidence else "",
            "position": self.position,
            "compound_values": self.compound_values,
            "validation_status": self.validation_status,
            "validation_message": self.validation_message
        }


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
    
    def validate(self, result: ExtractionResult, spec: SpecItem, 
                 hint: ExtractionHint) -> ExtractionResult:
        """추출 결과 검증"""
        if not self.config.enable_value_validation:
            return result
        
        if not result.found or not result.value:
            return result
        
        validation_issues = []
        
        # 1. 길이 검증
        if len(result.value) < MIN_VALUE_LENGTH:
            validation_issues.append("값이 너무 짧음")
        if len(result.value) > MAX_VALUE_LENGTH:
            validation_issues.append("값이 너무 김")
            result.confidence *= 0.5
        
        # 2. 숫자형 사양 검증
        if is_numeric_spec(spec.spec_name, hint.value_format):
            numeric_val = extract_numeric_value(result.value)
            
            if numeric_val is None:
                # 숫자형인데 숫자가 없음 → 잘못된 추출 가능성
                validation_issues.append("숫자형 사양이나 숫자 없음")
                result.confidence *= 0.6
            elif hint.historical_numeric_values:
                # 과거 값과 비교
                avg_historical = sum(hint.historical_numeric_values) / len(hint.historical_numeric_values)
                if avg_historical > 0:
                    variance = abs(numeric_val - avg_historical) / avg_historical
                    if variance > self.config.numeric_variance_threshold:
                        validation_issues.append(f"과거 값({avg_historical:.1f})과 차이 큼({variance:.1%})")
                        result.confidence *= 0.7
        
        # 3. 단위 검증
        if spec.umgv_uom and result.unit:
            if not self._units_compatible(spec.umgv_uom, result.unit):
                validation_issues.append(f"단위 불일치: 기대={spec.umgv_uom}, 추출={result.unit}")
                result.confidence *= 0.8
        
        # 4. 값이 키워드 자체인지 확인 (잘못된 추출)
        if self._is_likely_keyword(result.value, spec, hint):
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
    
    def _is_likely_keyword(self, value: str, spec: SpecItem, hint: ExtractionHint) -> bool:
        """추출된 값이 키워드(사양명)일 가능성 체크"""
        value_upper = value.upper().strip()
        
        # 사양명과 유사
        if value_upper == spec.spec_name.upper():
            return True
        if value_upper == spec.umgv_desc.upper():
            return True
        
        # 동의어와 유사
        for syn in hint.synonyms:
            if value_upper == syn.upper():
                return True
        
        # 일반적인 헤더 키워드
        header_keywords = ['type', 'qty', 'q\'ty', 'remark', 'unit', 'item', 'description', 
                          'spec', 'specification', 'parameter', 'value', 'no.', 'no']
        if value_upper in [kw.upper() for kw in header_keywords]:
            return True
        
        return False


# ############################################################################
# PostgreSQL 데이터 로더 (Phase 2: 템플릿 매칭 개선)
# ############################################################################

class PostgresLoader:
    """PostgreSQL 데이터 로더"""
    
    def __init__(self, config: Config):
        self.config = config
        self.conn = None
        self.logger = logging.getLogger("PostgresLoader")
        
        # 동의어 관리자
        self.synonym_manager = SynonymManager()
        
    def connect(self) -> bool:
        """DB 연결"""
        if not HAS_PSYCOPG2:
            self.logger.warning("psycopg2 not installed")
            return False
        
        try:
            self.conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                dbname=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            self.logger.info(f"PostgreSQL 연결 성공: {self.config.db_host}:{self.config.db_port}")
            return True
        except Exception as e:
            self.logger.error(f"PostgreSQL 연결 실패: {e}")
            return False
    
    def disconnect(self):
        """DB 연결 해제"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def load_glossary(self) -> pd.DataFrame:
        """용어집 로드 및 동의어 매핑 구축"""
        if not HAS_PANDAS:
            return pd.DataFrame()
        
        df = pd.DataFrame()
        
        # DB에서 로드 시도
        if self.conn:
            try:
                query = f"SELECT * FROM {USER_DB_TABLE_GLOSSARY}"
                df = pd.read_sql(query, self.conn)
                self.logger.info(f"DB 용어집 로드 완료: {len(df)}건")
            except Exception as e:
                self.logger.warning(f"DB 용어집 로드 실패: {e}")
        
        # 파일에서 로드 (폴백)
        if df.empty and os.path.exists(self.config.glossary_path):
            try:
                df = pd.read_csv(self.config.glossary_path, sep='\t', encoding='utf-8',
                                on_bad_lines='skip')
                self.logger.info(f"파일 용어집 로드 완료: {len(df)}건")
            except Exception as e:
                self.logger.warning(f"파일 용어집 로드 실패: {e}")
        
        # 동의어 매핑 구축
        if not df.empty:
            self.synonym_manager.build_from_glossary(df)
        
        return df
    
    def load_specdb(self) -> pd.DataFrame:
        """사양값DB 로드"""
        if not HAS_PANDAS:
            return pd.DataFrame()
        
        df = pd.DataFrame()
        
        if self.conn:
            try:
                query = f"SELECT * FROM {USER_DB_TABLE_SPECDB}"
                df = pd.read_sql(query, self.conn)
                self.logger.info(f"DB 사양값DB 로드 완료: {len(df)}건")
            except Exception as e:
                self.logger.warning(f"DB 사양값DB 로드 실패: {e}")
        
        if df.empty and os.path.exists(self.config.specdb_path):
            try:
                df = pd.read_csv(self.config.specdb_path, sep='\t', encoding='utf-8',
                                on_bad_lines='skip')
                self.logger.info(f"파일 사양값DB 로드 완료: {len(df)}건")
            except Exception as e:
                self.logger.warning(f"파일 사양값DB 로드 실패: {e}")
        
        return df
    
    def load_template(self) -> pd.DataFrame:
        """템플릿 로드"""
        if not HAS_PANDAS:
            return pd.DataFrame()
        
        df = pd.DataFrame()
        
        if self.conn:
            try:
                query = f"SELECT * FROM {USER_DB_TABLE_TEMPLATE}"
                df = pd.read_sql(query, self.conn)
                self.logger.info(f"DB 템플릿 로드 완료: {len(df)}건")
            except Exception as e:
                self.logger.warning(f"DB 템플릿 로드 실패: {e}")
        
        if df.empty and os.path.exists(self.config.spec_path):
            try:
                df = pd.read_csv(self.config.spec_path, sep='\t', encoding='utf-8',
                                on_bad_lines='skip')
                self.logger.info(f"파일 템플릿 로드 완료: {len(df)}건")
            except Exception as e:
                self.logger.warning(f"파일 템플릿 로드 실패: {e}")
        
        return df
    
    def get_template_for_pos(self, doknr: str, hull: str, df_template: pd.DataFrame) -> pd.DataFrame:
        """
        특정 POS 문서에 대한 템플릿 반환 (Phase 2: 개선된 매칭)
        
        매칭 우선순위:
        1. doknr 정확 매칭
        2. Hull + POS번호 조합 매칭 (matnr/extwg)
        3. POS번호만 매칭
        """
        if df_template.empty:
            return pd.DataFrame()
        
        pos_num = extract_pos_number(doknr)
        
        # 1순위: doknr 정확 매칭
        if 'doknr' in df_template.columns:
            mask = df_template['doknr'].apply(
                lambda x: doknr == str(x).strip() if pd.notna(x) else False
            )
            filtered = df_template[mask]
            if not filtered.empty:
                self.logger.debug(f"템플릿 매칭 (doknr 정확): {doknr}")
                return filtered
        
        # 2순위: Hull + POS번호 조합 (matnr/extwg에서 찾기)
        if hull and pos_num:
            # matnr 패턴: 5533AYS70101 또는 5533AYS0070101
            for col in ['matnr', 'extwg']:
                if col in df_template.columns:
                    # Hull번호 + POS번호 패턴
                    pattern1 = f"{hull}.*{pos_num[-5:]}"  # 마지막 5자리
                    pattern2 = f"{hull}.*{pos_num}"       # 전체 POS번호
                    
                    mask = df_template[col].apply(
                        lambda x: bool(re.search(pattern1, str(x)) or re.search(pattern2, str(x))) 
                        if pd.notna(x) else False
                    )
                    filtered = df_template[mask]
                    if not filtered.empty:
                        self.logger.debug(f"템플릿 매칭 ({col}): hull={hull}, pos={pos_num}")
                        return filtered
        
        # 3순위: POS번호만 매칭 (doknr에 포함)
        if pos_num and 'doknr' in df_template.columns:
            # YS-POS-0070101 형태
            mask = df_template['doknr'].apply(
                lambda x: pos_num in str(x) if pd.notna(x) else False
            )
            filtered = df_template[mask]
            if not filtered.empty:
                self.logger.debug(f"템플릿 매칭 (POS번호): {pos_num}")
                return filtered
        
        return pd.DataFrame()
    
    def get_synonyms(self, spec_name: str) -> List[str]:
        """사양명의 동의어 목록 반환"""
        return self.synonym_manager.get_synonyms(spec_name)
    
    def get_all_search_terms(self, spec_name: str, umgv_code: str = "") -> List[str]:
        """검색에 사용할 모든 용어 반환"""
        return self.synonym_manager.get_all_search_terms(spec_name, umgv_code)


# ############################################################################
# 힌트 엔진 (3-Tier 매칭 + 대표호선)
# ############################################################################

class ReferenceHintEngine:
    """용어집과 사양값DB 기반 힌트 엔진"""
    
    def __init__(self, config: Config, db_loader: PostgresLoader):
        self.config = config
        self.db_loader = db_loader
        self.logger = logging.getLogger("ReferenceHintEngine")
        
        # 3-Tier 인덱스
        self.glossary_full_idx: Dict[str, List[Dict]] = defaultdict(list)
        self.glossary_partial_idx: Dict[str, List[Dict]] = defaultdict(list)
        self.glossary_umgv_idx: Dict[str, List[Dict]] = defaultdict(list)
        
        # 사양값DB 인덱스
        self.specdb_idx: Dict[str, List[Dict]] = defaultdict(list)
        self.specdb_by_hull: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        
        # 대표호선 캐시
        self.hull_series_cache: Dict[str, str] = {}
        self.hull_results_cache: Dict[str, Dict[str, ExtractionResult]] = defaultdict(dict)
        
    def build_indexes(self, df_glossary: pd.DataFrame, df_specdb: pd.DataFrame):
        """인덱스 구축"""
        # 용어집 인덱스
        for _, row in df_glossary.iterrows():
            entry = row.to_dict()
            
            pmg_code = norm(entry.get('pmg_code', ''))
            umg_code = norm(entry.get('umg_code', ''))
            extwg = norm(entry.get('extwg', ''))
            umgv_code = norm(entry.get('umgv_code', ''))
            
            # Tier 1: 정확 매칭
            if pmg_code and umg_code and extwg and umgv_code:
                key1 = f"{pmg_code}_{umg_code}_{extwg}_{umgv_code}"
                self.glossary_full_idx[key1].append(entry)
            
            # Tier 2: 부분 매칭 (extwg 제외)
            if pmg_code and umg_code and umgv_code:
                key2 = f"{pmg_code}_{umg_code}_{umgv_code}"
                self.glossary_partial_idx[key2].append(entry)
            
            # Tier 3: 최소 매칭 (umgv_code만)
            if umgv_code:
                self.glossary_umgv_idx[umgv_code].append(entry)
        
        # 사양값DB 인덱스
        for _, row in df_specdb.iterrows():
            entry = row.to_dict()
            umgv_code = norm(entry.get('umgv_code', ''))
            matnr = norm(entry.get('matnr', ''))
            
            if umgv_code:
                self.specdb_idx[umgv_code].append(entry)
            
            # Hull별 인덱스
            hull = extract_hull_from_filename(matnr)
            if hull and umgv_code:
                self.specdb_by_hull[hull][umgv_code].append(entry)
        
        self.logger.info(
            f"힌트 인덱스 구축: "
            f"glossary_full={len(self.glossary_full_idx)}, "
            f"glossary_partial={len(self.glossary_partial_idx)}, "
            f"glossary_umgv={len(self.glossary_umgv_idx)}, "
            f"specdb={len(self.specdb_idx)}, "
            f"specdb_hulls={len(self.specdb_by_hull)}"
        )
    
    def get_hint(self, spec: SpecItem, hull: str = "") -> ExtractionHint:
        """사양 항목에 대한 힌트 조회"""
        hint = ExtractionHint()
        
        # 동의어 추가 (DB에서)
        search_terms = self.db_loader.get_all_search_terms(spec.spec_name, spec.umgv_code)
        hint.synonyms = search_terms
        
        # Tier 1: 정확 매칭
        key1 = f"{spec.pmg_code}_{spec.umg_code}_{spec.extwg}_{spec.umgv_code}"
        entries = self.glossary_full_idx.get(key1, [])
        
        if entries:
            hint.match_tier = 1
            self._fill_hint_from_entries(hint, entries, spec)
        else:
            # Tier 2: 부분 매칭
            key2 = f"{spec.pmg_code}_{spec.umg_code}_{spec.umgv_code}"
            entries = self.glossary_partial_idx.get(key2, [])
            
            if entries:
                hint.match_tier = 2
                self._fill_hint_from_entries(hint, entries, spec)
            else:
                # Tier 3: 최소 매칭
                entries = self.glossary_umgv_idx.get(spec.umgv_code, [])
                
                if entries:
                    # 동일 PMG/UMG 우선
                    same_pmg_umg = [e for e in entries 
                                   if norm(e.get('pmg_code', '')) == spec.pmg_code 
                                   and norm(e.get('umg_code', '')) == spec.umg_code]
                    if same_pmg_umg:
                        entries = same_pmg_umg
                    
                    hint.match_tier = 3
                    self._fill_hint_from_entries(hint, entries, spec)
        
        # 사양값DB에서 과거 값 수집
        self._collect_historical_values(hint, spec, hull)
        
        return hint
    
    def _fill_hint_from_entries(self, hint: ExtractionHint, entries: List[Dict], spec: SpecItem):
        """용어집 엔트리에서 힌트 정보 채우기"""
        if not entries:
            return
        
        first = entries[0]
        hint.pos_umgv_desc = norm(first.get('pos_umgv_desc', ''))
        hint.section_num = norm(first.get('section_num', ''))
        hint.table_text = norm(first.get('table_text', ''))
        hint.value_format = norm(first.get('value_format', ''))
        
        # 용어집의 pos_umgv_desc를 동의어에 추가
        if hint.pos_umgv_desc and hint.pos_umgv_desc not in hint.synonyms:
            hint.synonyms.insert(0, hint.pos_umgv_desc)
        
        # 과거 값 수집
        for entry in entries:
            val = norm(entry.get('pos_umgv_value', ''))
            unit = norm(entry.get('pos_umgv_uom', ''))
            if val:
                if val not in hint.historical_values:
                    hint.historical_values.append(val)
                    hint.historical_units.append(unit)
                    
                    # 숫자값 추출
                    num_val = extract_numeric_value(val)
                    if num_val is not None:
                        hint.historical_numeric_values.append(num_val)
    
    def _collect_historical_values(self, hint: ExtractionHint, spec: SpecItem, hull: str):
        """사양값DB에서 과거 값 수집"""
        # umgv_code로 검색
        specdb_entries = self.specdb_idx.get(spec.umgv_code, [])
        
        for entry in specdb_entries[:10]:
            val = norm(entry.get('umgv_value_edit', ''))
            unit = norm(entry.get('umgv_uom', ''))
            if val and val not in hint.historical_values:
                hint.historical_values.append(val)
                hint.historical_units.append(unit)
                
                num_val = extract_numeric_value(val)
                if num_val is not None:
                    hint.historical_numeric_values.append(num_val)
        
        # 동일 Hull의 값 우선
        if hull and hull in self.specdb_by_hull:
            hull_entries = self.specdb_by_hull[hull].get(spec.umgv_code, [])
            for entry in hull_entries[:3]:
                val = norm(entry.get('umgv_value_edit', ''))
                if val and val not in hint.historical_values:
                    hint.historical_values.insert(0, val)
    
    def get_representative_hull(self, hull: str, all_hulls: List[str]) -> str:
        """대표호선 반환"""
        if not hull or not all_hulls:
            return hull
        
        if hull in self.hull_series_cache:
            return self.hull_series_cache[hull]
        
        try:
            hull_num = int(hull)
            series_hulls = []
            
            for h in all_hulls:
                try:
                    h_num = int(h)
                    if abs(h_num - hull_num) <= self.config.series_hull_range:
                        series_hulls.append(h)
                except ValueError:
                    continue
            
            if series_hulls:
                representative = min(series_hulls, key=lambda x: int(x))
                for h in series_hulls:
                    self.hull_series_cache[h] = representative
                return representative
        except ValueError:
            pass
        
        return hull
    
    def cache_result(self, hull: str, spec_key: str, result: ExtractionResult, all_hulls: List[str]):
        """추출 결과 캐시 (검증 강화)"""
        if not self.config.representative_hull_cache_enabled:
            return
        
        # 최소 신뢰도 검사 (Phase 4: 0.7→0.8 상향)
        if result.confidence < self.config.cache_min_confidence:
            return
        
        if not result.found:
            return
        
        # 검증 상태 확인
        if result.validation_status == "invalid":
            return
        
        representative = self.get_representative_hull(hull, all_hulls)
        self.hull_results_cache[representative][spec_key] = result
    
    def get_cached_result(self, hull: str, spec_key: str, all_hulls: List[str]) -> Optional[ExtractionResult]:
        """캐시된 결과 조회"""
        if not self.config.representative_hull_cache_enabled:
            return None
        
        representative = self.get_representative_hull(hull, all_hulls)
        cached = self.hull_results_cache.get(representative, {}).get(spec_key)
        
        if cached:
            return ExtractionResult(
                value=cached.value,
                unit=cached.unit,
                confidence=cached.confidence * 0.95,
                method="cache",
                found=cached.found,
                evidence=f"[Cached from Hull {representative}] {cached.evidence}",
                position=cached.position,
                compound_values=cached.compound_values,
                validation_status=cached.validation_status
            )
        
        return None


# ############################################################################
# HTML 청크 파서 (4-Layer 취소선 제거)
# ############################################################################

class HTMLChunkParser:
    """HTML 문서 파서"""
    
    def __init__(self, config: Config):
        self.config = config
        self.soup = None
        self.tables = []
        self.table_structures = []  # Phase 1: 테이블 구조 정보
        self.text_chunks = []
        self.full_text = ""
        
    def parse(self, html_content: str) -> bool:
        """HTML 파싱"""
        if not HAS_BS4:
            return False
        
        try:
            self.soup = BeautifulSoup(html_content, 'html.parser')
            self._remove_strikethrough_4layer()
            self._extract_tables_with_structure()
            self._extract_text_chunks()
            self.full_text = self.soup.get_text(separator=' ', strip=True)
            return True
        except Exception as e:
            logger.error(f"HTML 파싱 실패: {e}")
            return False
    
    def _remove_strikethrough_4layer(self):
        """4-Layer 취소선 제거"""
        if not self.soup:
            return
        
        # Layer 1: 태그
        for tag_name in ['strike', 'del', 's']:
            for tag in self.soup.find_all(tag_name):
                tag.decompose()
        
        # Layer 2: 인라인 스타일
        style_pattern = re.compile(r'text-decoration:\s*line-through', re.IGNORECASE)
        for tag in self.soup.find_all(style=style_pattern):
            tag.decompose()
        
        # Layer 3: CSS 클래스
        for cls in ['strikethrough', 'line-through', 'deleted', 'struck', 'strike']:
            for tag in self.soup.find_all(class_=re.compile(cls, re.IGNORECASE)):
                tag.decompose()
        
        # Layer 4: 텍스트 패턴
        for text_node in self.soup.find_all(string=True):
            original = str(text_node)
            modified = re.sub(r'\[delete\].*?\[/delete\]', '', original, flags=re.IGNORECASE | re.DOTALL)
            modified = re.sub(r'~~[^~]+~~', '', modified)
            if modified != original:
                text_node.replace_with(modified)
    
    def _extract_tables_with_structure(self):
        """테이블 추출 (Phase 1: 구조 정보 포함)"""
        self.tables = []
        self.table_structures = []
        
        for table in self.soup.find_all('table'):
            table_data = []
            rows = table.find_all('tr')
            
            # 구조 정보
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
                    cells.append(text)
                
                if cells:
                    table_data.append(cells)
                    
                    # 헤더 행 감지 (첫 번째 행 또는 th 태그가 있는 행)
                    has_th = row.find('th') is not None
                    if has_th or (row_idx == 0 and structure['header_row_idx'] == -1):
                        # 헤더 행인지 추가 검증
                        if self._is_likely_header_row(cells):
                            structure['header_row_idx'] = row_idx
                            structure['header_cols'] = cells
                            structure['data_start_row'] = row_idx + 1
                    
                    structure['col_count'] = max(structure['col_count'], len(cells))
            
            if table_data:
                self.tables.append(table_data)
                self.table_structures.append(structure)
    
    def _is_likely_header_row(self, cells: List[str]) -> bool:
        """헤더 행인지 판단"""
        if not cells:
            return False
        
        # 헤더 키워드
        header_keywords = [
            'type', 'item', 'description', 'spec', 'specification', 'parameter',
            'unit', 'value', 'qty', 'q\'ty', 'quantity', 'remark', 'no.', 'no',
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
    
    def _extract_text_chunks(self):
        """텍스트 청크 추출"""
        self.text_chunks = []
        
        for heading in self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            section_title = heading.get_text(strip=True)
            content_parts = []
            
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                text = sibling.get_text(strip=True)
                if text:
                    content_parts.append(text)
            
            if content_parts:
                self.text_chunks.append({
                    'title': section_title,
                    'content': ' '.join(content_parts)
                })
        
        if not self.text_chunks:
            for p in self.soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 50:
                    self.text_chunks.append({'title': '', 'content': text})
    
    def search_in_tables_enhanced(self, keywords: List[str]) -> List[Dict]:
        """
        테이블 검색 (Phase 1: 개선된 로직)
        - 헤더/데이터 구분
        - 열 위치 기반 값 추출
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
                                        'match': keyword,
                                        'match_type': 'header_column'
                                    })
                    
                    # Case 2: 키워드가 데이터 행에 있는 경우 → 같은 행의 다음 셀에서 값 추출
                    else:
                        # 다음 셀(들)에서 값 후보 찾기
                        for val_col_idx in range(match_col_idx + 1, len(row)):
                            value = row[val_col_idx]
                            if value and not self._is_likely_header_keyword(value):
                                results.append({
                                    'table_idx': t_idx,
                                    'row_idx': r_idx,
                                    'col_idx': val_col_idx,
                                    'row': row,
                                    'value': value,
                                    'match': keyword,
                                    'match_type': 'same_row'
                                })
                                break  # 첫 번째 유효한 값만
                    
                    break  # 첫 번째 매칭 키워드만
        
        return results
    
    def _is_likely_header_keyword(self, text: str) -> bool:
        """텍스트가 헤더 키워드인지 판단"""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        header_keywords = [
            'type', 'item', 'description', 'spec', 'specification', 'parameter',
            'unit', 'value', 'qty', 'q\'ty', 'quantity', 'remark', 'no.', 'no',
            'name', 'model', 'capacity', 'material', 'maker', 'size', 'head',
            'motor', 'power', 'pressure', 'temperature', 'flow', 'speed'
        ]
        
        return text_lower in header_keywords
    
    def search_value_in_doc_with_context(self, value: str, context_chars: int = 100) -> Optional[str]:
        """문서에서 값 검색"""
        if not value or not self.full_text:
            return None
        
        idx = self.full_text.find(value)
        if idx == -1:
            num_only = re.sub(r'[^0-9.]', '', value)
            if num_only:
                idx = self.full_text.find(num_only)
        
        if idx != -1:
            start = max(0, idx - context_chars)
            end = min(len(self.full_text), idx + len(value) + context_chars)
            return self.full_text[start:end]
        
        return None
    
    def get_evidence_text(self, max_chars: int = 15000) -> str:
        """LLM용 증거 텍스트"""
        evidence_parts = []
        
        for i, table in enumerate(self.tables[:self.config.evidence_max_tables]):
            table_text = f"\n[Table {i+1}]\n"
            for row in table:
                table_text += ' | '.join(row) + '\n'
            evidence_parts.append(table_text)
        
        for chunk in self.text_chunks[:10]:
            if chunk['title']:
                evidence_parts.append(f"\n[Section: {chunk['title']}]\n{chunk['content']}")
            else:
                evidence_parts.append(chunk['content'])
        
        full_evidence = '\n'.join(evidence_parts)
        
        if len(full_evidence) > max_chars:
            full_evidence = full_evidence[:max_chars] + "..."
        
        return full_evidence


# ############################################################################
# 규칙 기반 추출기 (Phase 1: 개선된 테이블 추출)
# ############################################################################

class ImprovedRuleExtractor:
    """규칙 기반 사양값 추출기"""
    
    def __init__(self, config: Config, db_loader: PostgresLoader):
        self.config = config
        self.db_loader = db_loader
        self.logger = logging.getLogger("RuleExtractor")
        
    def extract(self, parser: HTMLChunkParser, spec: SpecItem, 
                hint: ExtractionHint) -> ExtractionResult:
        """규칙 기반 추출"""
        # 검색 키워드 (DB 동의어 사용)
        keywords = hint.synonyms if hint.synonyms else [spec.spec_name]
        
        # 1단계: 개선된 테이블 검색
        table_results = parser.search_in_tables_enhanced(keywords)
        
        if table_results:
            result = self._extract_from_table_results_enhanced(table_results, spec, hint, parser)
            if result.found and result.validation_status != "invalid":
                return result
        
        # 2단계: 텍스트 청크 검색
        text_result = self._extract_from_text_chunks(parser, keywords, spec, hint)
        if text_result.found:
            return text_result
        
        # 3단계: 전체 텍스트 검색
        return self._extract_from_full_text(parser, keywords, spec, hint)
    
    def _extract_from_table_results_enhanced(self, results: List[Dict], spec: SpecItem,
                                             hint: ExtractionHint, parser: HTMLChunkParser) -> ExtractionResult:
        """개선된 테이블 결과 처리 (Phase 1)"""
        # 매칭 타입별 우선순위 정렬
        # header_column > same_row
        results_sorted = sorted(results, key=lambda x: (
            0 if x.get('match_type') == 'header_column' else 1,
            -len(x.get('match', ''))  # 긴 키워드 우선
        ))
        
        for result in results_sorted:
            value = result.get('value', '').strip()
            row = result.get('row', [])
            match_type = result.get('match_type', '')
            
            if not value:
                continue
            
            # 값 검증: 헤더 키워드가 아닌지 확인
            if parser._is_likely_header_keyword(value):
                continue
            
            # 체크박스 감지
            checkbox = detect_checkbox_selection(value)
            if checkbox:
                return ExtractionResult(
                    value=checkbox,
                    unit="",
                    confidence=0.90,
                    method="rule_checkbox",
                    found=True,
                    evidence=' | '.join(row),
                    position=f"Table {result['table_idx'] + 1}, Row {result['row_idx'] + 1}"
                )
            
            # 복합값 파싱
            compound = parse_compound_value(value, self.config.split_compound_values)
            
            if compound:
                if len(compound) == 1:
                    val, unit = compound[0]
                    
                    # 범위형 체크
                    if '~' in val or (re.search(r'\d\s*-\s*\d', val) and 'POS' not in val.upper()):
                        range_vals = parse_range_value(value, self.config.split_range_values)
                        return ExtractionResult(
                            value=value if not self.config.split_range_values else range_vals[0][0],
                            unit=spec.umgv_uom or unit,
                            confidence=0.85,
                            method="rule_range",
                            found=True,
                            evidence=' | '.join(row),
                            position=f"Table {result['table_idx'] + 1}, Row {result['row_idx'] + 1}",
                            compound_values=range_vals if self.config.split_range_values else []
                        )
                    
                    # 신뢰도 조정: header_column 매칭이 더 높음
                    conf = 0.88 if match_type == 'header_column' else 0.82
                    
                    return ExtractionResult(
                        value=val,
                        unit=spec.umgv_uom or unit,
                        confidence=conf,
                        method=f"rule_table_{match_type}",
                        found=True,
                        evidence=' | '.join(row),
                        position=f"Table {result['table_idx'] + 1}, Row {result['row_idx'] + 1}"
                    )
                else:
                    # 복합값
                    if self.config.split_compound_values:
                        return ExtractionResult(
                            value=compound[0][0],
                            unit=compound[0][1],
                            confidence=0.80,
                            method="rule_compound",
                            found=True,
                            evidence=' | '.join(row),
                            position=f"Table {result['table_idx'] + 1}, Row {result['row_idx'] + 1}",
                            compound_values=compound
                        )
                    else:
                        return ExtractionResult(
                            value=value,
                            unit="",
                            confidence=0.80,
                            method="rule_compound",
                            found=True,
                            evidence=' | '.join(row),
                            position=f"Table {result['table_idx'] + 1}, Row {result['row_idx'] + 1}"
                        )
        
        return ExtractionResult(found=False, method="rule_table_fail")
    
    def _extract_from_text_chunks(self, parser: HTMLChunkParser, keywords: List[str],
                                  spec: SpecItem, hint: ExtractionHint) -> ExtractionResult:
        """텍스트 청크에서 추출"""
        for chunk in parser.text_chunks:
            content = chunk['content']
            content_lower = content.lower()
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    # 키워드 뒤의 값 추출
                    pattern = re.compile(
                        rf'{re.escape(keyword)}\s*[:\-]?\s*([0-9,.\-\s~]+)\s*([a-zA-Z°℃%/]+)?',
                        re.IGNORECASE
                    )
                    match = pattern.search(content)
                    
                    if match:
                        val = match.group(1).strip()
                        unit = match.group(2) or ""
                        
                        return ExtractionResult(
                            value=val,
                            unit=spec.umgv_uom or unit.strip(),
                            confidence=0.75,
                            method="rule_text",
                            found=True,
                            evidence=content[:200],
                            position=f"Section: {chunk['title']}" if chunk['title'] else "Text"
                        )
        
        return ExtractionResult(found=False, method="rule_text_fail")
    
    def _extract_from_full_text(self, parser: HTMLChunkParser, keywords: List[str],
                                spec: SpecItem, hint: ExtractionHint) -> ExtractionResult:
        """전체 텍스트에서 추출"""
        full_text = parser.full_text
        
        for keyword in keywords:
            pattern = re.compile(
                rf'{re.escape(keyword)}\s*[:\-]?\s*([0-9,.\-\s~]+)\s*([a-zA-Z°℃%/]+)?',
                re.IGNORECASE
            )
            match = pattern.search(full_text)
            
            if match:
                val = match.group(1).strip()
                unit = match.group(2) or ""
                
                start = max(0, match.start() - 50)
                end = min(len(full_text), match.end() + 50)
                context = full_text[start:end]
                
                return ExtractionResult(
                    value=val,
                    unit=spec.umgv_uom or unit.strip(),
                    confidence=0.70,
                    method="rule_fulltext",
                    found=True,
                    evidence=context,
                    position="Full text search"
                )
        
        return ExtractionResult(found=False, method="rule_fail")


# ############################################################################
# 통합 LLM 클라이언트
# ############################################################################

class UnifiedLLMClient:
    """Ollama/Claude 통합 LLM 클라이언트"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("UnifiedLLMClient")
        
        self.port_index = 0
        self.port_lock = threading.Lock()
        
        self.claude_client = None
        if config.llm_backend == "claude" and HAS_ANTHROPIC and config.claude_api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=config.claude_api_key)
            except Exception as e:
                self.logger.warning(f"Claude 초기화 실패: {e}")
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
    
    def generate(self, prompt: str) -> Tuple[str, int, int]:
        """LLM 응답 생성"""
        if self.config.llm_backend == "claude":
            return self._generate_claude(prompt)
        else:
            return self._generate_ollama(prompt)
    
    def _generate_ollama(self, prompt: str) -> Tuple[str, int, int]:
        """Ollama API 호출"""
        if not HAS_REQUESTS:
            return "", 0, 0
        
        with self.port_lock:
            port = self.config.ollama_ports[self.port_index]
            self.port_index = (self.port_index + 1) % len(self.config.ollama_ports)
        
        url = f"http://{self.config.ollama_host}:{port}/api/generate"
        
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.llm_temperature}
        }
        
        for attempt in range(self.config.llm_max_retries):
            try:
                time.sleep(self.config.llm_rate_limit)
                
                response = requests.post(url, json=payload, timeout=self.config.ollama_timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    text = data.get('response', '')
                    
                    input_tokens = len(prompt) // 4
                    output_tokens = len(text) // 4
                    
                    self.total_input_tokens += input_tokens
                    self.total_output_tokens += output_tokens
                    self.total_calls += 1
                    
                    return text, input_tokens, output_tokens
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Ollama 타임아웃 (attempt {attempt + 1})")
            except Exception as e:
                self.logger.warning(f"Ollama 오류: {e}")
            
            if attempt < self.config.llm_max_retries - 1:
                time.sleep(self.config.llm_retry_sleep)
        
        return "", 0, 0
    
    def _generate_claude(self, prompt: str) -> Tuple[str, int, int]:
        """Claude API 호출"""
        if not self.claude_client:
            return "", 0, 0
        
        try:
            time.sleep(self.config.llm_rate_limit)
            
            message = self.claude_client.messages.create(
                model=self.config.claude_model,
                max_tokens=self.config.claude_max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = message.content[0].text
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_calls += 1
            
            return text, input_tokens, output_tokens
            
        except Exception as e:
            self.logger.error(f"Claude API 오류: {e}")
            return "", 0, 0


# ############################################################################
# LLM 기반 추출기
# ############################################################################

class ImprovedLLMExtractor:
    """LLM 기반 사양값 추출기"""
    
    def __init__(self, config: Config, llm_client: UnifiedLLMClient, db_loader: PostgresLoader):
        self.config = config
        self.llm_client = llm_client
        self.db_loader = db_loader
        self.logger = logging.getLogger("LLMExtractor")
    
    def extract(self, parser: HTMLChunkParser, spec: SpecItem, 
                hint: ExtractionHint) -> ExtractionResult:
        """LLM 기반 추출"""
        prompt = self._build_prompt(parser, spec, hint)
        response, in_tokens, out_tokens = self.llm_client.generate(prompt)
        
        if not response:
            return ExtractionResult(found=False, method="llm_fail")
        
        return self._parse_response(response, spec)
    
    def _build_prompt(self, parser: HTMLChunkParser, spec: SpecItem, 
                      hint: ExtractionHint) -> str:
        """프롬프트 구성"""
        hint_section = self._build_hint_section(spec, hint)
        evidence = parser.get_evidence_text(self.config.max_evidence_chars)
        
        # 동의어 (DB에서 로드된)
        search_terms = ', '.join(hint.synonyms[:5]) if hint.synonyms else spec.spec_name
        
        prompt = f"""You are a shipbuilding specification value extractor.

## Specification to Extract
- Standard Name: {spec.umgv_desc or spec.spec_name}
- Search Terms: {search_terms}
- Expected Unit: {spec.umgv_uom or "unknown"}

{hint_section}

## Document Content
{evidence}

## Instructions
1. Search for the specification using the search terms
2. Extract the exact value as written in the document
3. Do NOT extract header labels or keywords as values
4. If value contains multiple parts (e.g., "28,260 kW / 72.0 rpm"), extract as-is
5. For checkbox patterns (Y), (N), [x], return the selection

## Response Format (JSON only)
If value IS FOUND:
{{"value": "<extracted value>", "unit": "<unit>", "confidence": <0.0-1.0>, "found": true, "position": "<location>"}}

If value IS NOT FOUND:
{{"value": "NOT_FOUND", "unit": "", "confidence": 0.0, "found": false, "position": ""}}

Respond with JSON only."""
        
        return prompt
    
    def _build_hint_section(self, spec: SpecItem, hint: ExtractionHint) -> str:
        """힌트 섹션 구성"""
        parts = []
        
        if hint.match_tier > 0:
            parts.append(f"## Reference Hints (Tier {hint.match_tier} match)")
        else:
            parts.append("## Reference Hints")
        
        if hint.pos_umgv_desc and hint.pos_umgv_desc != spec.spec_name:
            parts.append(f"- May appear as: \"{hint.pos_umgv_desc}\"")
        
        if hint.historical_values:
            parts.append("- Example values from similar documents:")
            for val, unit in zip(hint.historical_values[:3], hint.historical_units[:3]):
                parts.append(f"  - {val} {unit}".strip())
        
        if hint.value_format:
            parts.append(f"- Value format: {hint.value_format}")
        
        return '\n'.join(parts) if len(parts) > 1 else ""
    
    def _parse_response(self, response: str, spec: SpecItem) -> ExtractionResult:
        """응답 파싱"""
        try:
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if not json_match:
                return ExtractionResult(found=False, method="llm_parse_fail")
            
            data = json.loads(json_match.group())
            
            value = str(data.get('value', '')).strip()
            unit = str(data.get('unit', '')).strip()
            confidence = float(data.get('confidence', 0.0))
            found = data.get('found', True)
            position = str(data.get('position', '')).strip()
            
            if value.upper() == "NOT_FOUND" or not found:
                return ExtractionResult(
                    value="",
                    unit="",
                    confidence=0.0,
                    method="NOT_FOUND",
                    found=False,
                    position=position
                )
            
            compound_values = []
            if '/' in value and self.config.split_compound_values:
                compound_values = parse_compound_value(value, True)
            
            return ExtractionResult(
                value=value,
                unit=spec.umgv_uom or unit,
                confidence=confidence,
                method="llm",
                found=True,
                position=position,
                compound_values=compound_values
            )
            
        except json.JSONDecodeError:
            return ExtractionResult(found=False, method="llm_parse_fail")
        except Exception as e:
            self.logger.warning(f"응답 파싱 오류: {e}")
            return ExtractionResult(found=False, method="llm_parse_fail")


# ############################################################################
# 토큰 추적기
# ############################################################################

class TokenTracker:
    """토큰 사용량 및 추출 통계 추적"""
    
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.calls = 0
        self.rule_success = 0
        self.llm_fallback = 0
        self.not_found = 0
        self.failures = 0
        self.cache_hits = 0
        self.validation_warnings = 0
        self.validation_invalids = 0
        self.start_time = time.time()
    
    def add_tokens(self, input_tokens: int, output_tokens: int):
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.calls += 1
    
    def add_rule_success(self):
        self.rule_success += 1
    
    def add_llm_fallback(self):
        self.llm_fallback += 1
    
    def add_not_found(self):
        self.not_found += 1
    
    def add_failure(self):
        self.failures += 1
    
    def add_cache_hit(self):
        self.cache_hits += 1
    
    def add_validation_warning(self):
        self.validation_warnings += 1
    
    def add_validation_invalid(self):
        self.validation_invalids += 1
    
    def get_elapsed(self) -> float:
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict:
        total = self.rule_success + self.llm_fallback + self.not_found + self.failures
        return {
            "total_items": total,
            "rule_success": self.rule_success,
            "rule_success_pct": (self.rule_success / total * 100) if total > 0 else 0,
            "llm_fallback": self.llm_fallback,
            "llm_fallback_pct": (self.llm_fallback / total * 100) if total > 0 else 0,
            "not_found": self.not_found,
            "not_found_pct": (self.not_found / total * 100) if total > 0 else 0,
            "failures": self.failures,
            "failures_pct": (self.failures / total * 100) if total > 0 else 0,
            "cache_hits": self.cache_hits,
            "validation_warnings": self.validation_warnings,
            "validation_invalids": self.validation_invalids,
            "llm_calls": self.calls,
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "elapsed_seconds": self.get_elapsed()
        }
    
    def get_progress_str(self) -> str:
        total = self.rule_success + self.llm_fallback + self.not_found + self.failures
        elapsed = int(self.get_elapsed())
        return (f"[{LLM_BACKEND}] "
                f"[IN:{self.total_input:,} OUT:{self.total_output:,}] "
                f"[LLM:{self.calls}] "
                f"[Rule:{self.rule_success}/{total}] "
                f"[Cache:{self.cache_hits}] "
                f"[NotFound:{self.not_found}] "
                f"[{elapsed}s]")


# ############################################################################
# 파일 탐색기
# ############################################################################

class FileFinder:
    """POS 파일 탐색"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("FileFinder")
    
    def find_pos_files(self) -> List[str]:
        """POS HTML 파일 목록 반환"""
        files = []
        
        upload_dir = "/mnt/user-data/uploads"
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                if f.endswith('.html') and 'POS' in f.upper():
                    files.append(os.path.join(upload_dir, f))
        
        if os.path.exists(self.config.base_folder):
            for root, dirs, filenames in os.walk(self.config.base_folder):
                for f in filenames:
                    if f.endswith('.html'):
                        files.append(os.path.join(root, f))
        
        if EXTRACTION_MODE == "light" and len(files) > LIGHT_MODE_MAX_FILES:
            files = files[:LIGHT_MODE_MAX_FILES]
        
        self.logger.info(f"POS 파일 {len(files)}개 발견")
        return files


# ############################################################################
# 메인 추출기 클래스
# ############################################################################

class POSExtractor:
    """POS 사양값 추출기 메인 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("POSExtractor")
        
        self.db_loader = PostgresLoader(config)
        self.hint_engine = None
        self.llm_client = None
        self.rule_extractor = None
        self.llm_extractor = None
        self.value_validator = None
        self.file_finder = FileFinder(config)
        self.token_tracker = TokenTracker()
        
        self.df_glossary = pd.DataFrame()
        self.df_specdb = pd.DataFrame()
        self.df_template = pd.DataFrame()
        
        self.logger.info(f"POSExtractor 초기화 (모드: {config.mode}, LLM: {config.llm_backend})")
    
    def initialize(self):
        """초기화"""
        self.logger.info("초기화 시작...")
        
        if self.config.data_source == "db":
            self.db_loader.connect()
        
        self.df_glossary = self.db_loader.load_glossary()
        self.df_specdb = self.db_loader.load_specdb()
        self.df_template = self.db_loader.load_template()
        
        self.hint_engine = ReferenceHintEngine(self.config, self.db_loader)
        self.hint_engine.build_indexes(self.df_glossary, self.df_specdb)
        
        self.llm_client = UnifiedLLMClient(self.config)
        
        self.rule_extractor = ImprovedRuleExtractor(self.config, self.db_loader)
        self.llm_extractor = ImprovedLLMExtractor(self.config, self.llm_client, self.db_loader)
        self.value_validator = ValueValidator(self.config)
        
        os.makedirs(self.config.output_path, exist_ok=True)
        os.makedirs(self.config.partial_output_path, exist_ok=True)
        
        self.logger.info("초기화 완료")
    
    def cleanup(self):
        """정리"""
        self.db_loader.disconnect()
    
    def run(self) -> Dict:
        """추출 실행"""
        if self.config.mode == "light":
            return self._run_light_mode()
        elif self.config.mode == "full":
            return self._run_full_mode()
        elif self.config.mode == "verify":
            return self._run_verify_mode()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")
    
    def _run_light_mode(self) -> Dict:
        """Light 모드 실행"""
        self.logger.info("Light 모드 시작")
        
        pos_files = self.file_finder.find_pos_files()
        
        if not pos_files:
            return {"status": "error", "message": "POS 파일 없음"}
        
        results = []
        pbar = tqdm(pos_files, desc="추출 진행")
        
        for filepath in pbar:
            try:
                file_result = self._process_single_file(filepath)
                results.append(file_result)
                pbar.set_postfix_str(self.token_tracker.get_progress_str())
            except Exception as e:
                self.logger.error(f"파일 처리 오류 {filepath}: {e}")
                results.append({
                    "file": os.path.basename(filepath),
                    "status": "error",
                    "message": str(e)
                })
        
        pbar.close()
        
        output_file = self._save_results(results)
        summary = self.token_tracker.get_summary()
        self._print_statistics(summary)
        
        return {
            "status": "success",
            "mode": "light",
            "total_items": summary['total_items'],
            "elapsed_seconds": summary['elapsed_seconds'],
            "saved_files": {"json": output_file},
            "statistics": summary
        }
    
    def _process_single_file(self, filepath: str) -> Dict:
        """단일 파일 처리"""
        filename = os.path.basename(filepath)
        doknr = extract_doknr_from_filename(filename)
        hull = extract_hull_from_filename(filename)
        
        parser = HTMLChunkParser(self.config)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except Exception as e:
            return {"file": filename, "status": "read_error", "message": str(e)}
        
        if not parser.parse(html_content):
            return {"file": filename, "status": "parse_error"}
        
        # Phase 2: 개선된 템플릿 매칭
        template = self.db_loader.get_template_for_pos(doknr, hull, self.df_template)
        
        if template.empty:
            self.logger.info(f"{filename}: 템플릿 없음, 기본 추출 시도")
            specs = self._generate_default_specs_from_db(hull)
        else:
            specs = self._template_to_specs(template)
        
        extractions = []
        all_hulls = self._get_all_hulls()
        
        for spec in specs:
            spec_key = f"{spec.umg_code}_{spec.umgv_code}"
            
            # 캐시 확인
            cached = self.hint_engine.get_cached_result(hull, spec_key, all_hulls)
            
            if cached:
                result = cached
                self.token_tracker.add_cache_hit()
            else:
                # 힌트 조회
                hint = self.hint_engine.get_hint(spec, hull)
                
                # 규칙 기반 추출
                result = self.rule_extractor.extract(parser, spec, hint)
                
                # Phase 4: 값 검증
                if result.found:
                    result = self.value_validator.validate(result, spec, hint)
                
                if result.found and result.validation_status != "invalid":
                    self.token_tracker.add_rule_success()
                    if result.validation_status == "warning":
                        self.token_tracker.add_validation_warning()
                else:
                    # LLM 폴백
                    result = self.llm_extractor.extract(parser, spec, hint)
                    
                    if result.found:
                        result = self.value_validator.validate(result, spec, hint)
                        self.token_tracker.add_llm_fallback()
                        if result.validation_status == "warning":
                            self.token_tracker.add_validation_warning()
                    elif result.method == "NOT_FOUND":
                        self.token_tracker.add_not_found()
                    else:
                        self.token_tracker.add_failure()
                
                # 캐시 저장 (검증된 결과만)
                if result.found and result.validation_status != "invalid":
                    self.hint_engine.cache_result(hull, spec_key, result, all_hulls)
            
            extractions.append({
                "spec_name": spec.spec_name,
                "umgv_code": spec.umgv_code,
                "umg_desc": spec.umg_desc,
                **result.to_dict()
            })
        
        return {
            "file": filename,
            "doknr": doknr,
            "hull": hull,
            "status": "success",
            "extractions": extractions
        }
    
    def _generate_default_specs_from_db(self, hull: str) -> List[SpecItem]:
        """DB에서 기본 사양 항목 생성 (하드코딩 제거)"""
        specs = []
        
        # 사양값DB에서 가장 많이 사용되는 umgv_code 찾기
        if not self.df_specdb.empty and 'umgv_code' in self.df_specdb.columns:
            # Hull별 또는 전체에서 상위 사양 추출
            if hull and 'matnr' in self.df_specdb.columns:
                hull_mask = self.df_specdb['matnr'].apply(
                    lambda x: hull in str(x) if pd.notna(x) else False
                )
                hull_specs = self.df_specdb[hull_mask]
                if not hull_specs.empty:
                    top_codes = hull_specs['umgv_code'].value_counts().head(LIGHT_MODE_DEFAULT_SPECS)
                else:
                    top_codes = self.df_specdb['umgv_code'].value_counts().head(LIGHT_MODE_DEFAULT_SPECS)
            else:
                top_codes = self.df_specdb['umgv_code'].value_counts().head(LIGHT_MODE_DEFAULT_SPECS)
            
            for umgv_code in top_codes.index:
                if not umgv_code:
                    continue
                
                # umgv_code에 해당하는 사양 정보 찾기
                mask = self.df_specdb['umgv_code'] == umgv_code
                rows = self.df_specdb[mask]
                
                if not rows.empty:
                    row = rows.iloc[0]
                    specs.append(SpecItem(
                        umgv_code=umgv_code,
                        umgv_desc=norm(row.get('umgv_desc', '')),
                        umg_code=norm(row.get('umg_code', '')),
                        umg_desc=norm(row.get('umg_desc', '')),
                        spec_name=norm(row.get('umgv_desc', ''))
                    ))
        
        # 사양값DB가 비어있으면 용어집에서 추출
        if not specs and not self.df_glossary.empty and 'umgv_code' in self.df_glossary.columns:
            top_codes = self.df_glossary['umgv_code'].value_counts().head(LIGHT_MODE_DEFAULT_SPECS)
            
            for umgv_code in top_codes.index:
                if not umgv_code:
                    continue
                
                mask = self.df_glossary['umgv_code'] == umgv_code
                rows = self.df_glossary[mask]
                
                if not rows.empty:
                    row = rows.iloc[0]
                    specs.append(SpecItem(
                        umgv_code=umgv_code,
                        umgv_desc=norm(row.get('umgv_desc', '')),
                        umg_code=norm(row.get('umg_code', '')),
                        umg_desc=norm(row.get('umg_desc', '')),
                        spec_name=norm(row.get('umgv_desc', ''))
                    ))
        
        return specs[:LIGHT_MODE_DEFAULT_SPECS]
    
    def _template_to_specs(self, template: pd.DataFrame) -> List[SpecItem]:
        """템플릿을 SpecItem 리스트로 변환"""
        specs = []
        
        for _, row in template.iterrows():
            spec = SpecItem(
                pmg_code=norm(row.get('pmg_code', '')),
                pmg_desc=norm(row.get('pmg_desc', '')),
                umg_code=norm(row.get('umg_code', '')),
                umg_desc=norm(row.get('umg_desc', '')),
                extwg=norm(row.get('extwg', '')),
                extwg_desc=norm(row.get('extwg_desc', '')),
                matnr=norm(row.get('matnr', '')),
                doknr=norm(row.get('doknr', '')),
                umgv_code=norm(row.get('umgv_code', '')),
                umgv_desc=norm(row.get('umgv_desc', '')),
                umgv_uom=norm(row.get('umgv_uom', '')),
                spec_name=norm(row.get('umgv_desc', ''))
            )
            specs.append(spec)
        
        return specs
    
    def _get_all_hulls(self) -> List[str]:
        """모든 Hull 번호 목록"""
        hulls = set()
        
        for col in ['extwg', 'matnr', 'doknr']:
            if col in self.df_template.columns:
                for val in self.df_template[col].dropna():
                    hull = extract_hull_from_filename(str(val))
                    if hull:
                        hulls.add(hull)
        
        return list(hulls)
    
    def _save_results(self, results: List[Dict]) -> str:
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.mode}_extraction_result_{timestamp}.json"
        filepath = os.path.join(self.config.output_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"JSON 저장: {filepath}")
        return filepath
    
    def _print_statistics(self, summary: Dict):
        """통계 출력"""
        self.logger.info("=" * 60)
        self.logger.info("추출 통계")
        self.logger.info("=" * 60)
        self.logger.info(f"총 항목: {summary['total_items']}")
        self.logger.info(f"Rule 성공: {summary['rule_success']} ({summary['rule_success_pct']:.1f}%)")
        self.logger.info(f"LLM Fallback: {summary['llm_fallback']} ({summary['llm_fallback_pct']:.1f}%)")
        self.logger.info(f"NOT_FOUND: {summary['not_found']} ({summary['not_found_pct']:.1f}%)")
        self.logger.info(f"실패: {summary['failures']} ({summary['failures_pct']:.1f}%)")
        self.logger.info(f"캐시 히트: {summary['cache_hits']}")
        self.logger.info(f"검증 경고: {summary['validation_warnings']}")
        self.logger.info(f"검증 실패: {summary['validation_invalids']}")
        self.logger.info(f"LLM 호출: {summary['llm_calls']}회")
        self.logger.info(f"토큰: IN={summary['input_tokens']:,} OUT={summary['output_tokens']:,}")
        self.logger.info(f"소요 시간: {summary['elapsed_seconds']:.1f}초")
        self.logger.info("=" * 60)
    
    def _run_full_mode(self) -> Dict:
        """Full 모드 (추후 구현)"""
        self.logger.info("Full 모드는 추후 구현 예정")
        return {"status": "not_implemented", "mode": "full"}
    
    def _run_verify_mode(self) -> Dict:
        """Verify 모드 (추후 구현)"""
        self.logger.info("Verify 모드는 추후 구현 예정")
        return {"status": "not_implemented", "mode": "verify"}


# ############################################################################
# 메인 실행
# ############################################################################

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("POS Specification Value Extractor (Enhanced)")
    print("=" * 70)
    
    config = Config()
    extractor = POSExtractor(config)
    
    try:
        extractor.initialize()
        result = extractor.run()
        
        print("\n" + "=" * 60)
        print("실행 완료")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        
        if 'saved_files' in result:
            print("\n저장된 파일:")
            for file_type, file_path in result['saved_files'].items():
                print(f"  {file_type}: {file_path}")
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단됨")
        return {"status": "interrupted"}
        
    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
        
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()
