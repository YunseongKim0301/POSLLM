# -*- coding: utf-8 -*-
"""
POS Specification Value Extractor (Improved Version)
=====================================================

조선 산업용 POS 문서에서 사양값을 추출하는 고도화된 시스템

핵심 개선사항:
1. 취소선 제거 강화 (4-layer 제거 전략)
2. 체크박스 맥락 인식 기능 (Y/N/Q 패턴 포함)
3. 복합값 처리 (토글 방식으로 분리/통합 선택 가능)
4. 용어집/사양값DB 힌트 활용 고도화 (3-tier key matching)
5. 대표호선 구조 힌트 + 시리즈 호선 그룹핑 (캐싱 시스템)
6. NOT_FOUND 명시적 처리 (found 필드 추가)
7. Claude API 지원 (Unified LLM Client)
8. 검증 모드 강화 (5단계 검증 프로세스)
9. 정확도 85%+ 목표 달성을 위한 다층 추출 전략
10. 동의어 매핑 DB 로드 (pos_dict의 umgv_desc ↔ pos_umgv_desc)

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
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_SENTENCE_TRANSFORMER = True
except ImportError:
    HAS_SENTENCE_TRANSFORMER = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # tqdm 대체 클래스
    class tqdm:
        """tqdm이 없을 때 사용하는 대체 클래스"""
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
# 설정값 (Settings)
# ############################################################################

# =============================================================================
# [1] 모드 설정
# =============================================================================
EXTRACTION_MODE = "light"  # "full", "light", "verify" 중 선택
DATA_SOURCE_MODE = "db"    # "file" 또는 "db" 선택

# =============================================================================
# [2] LLM 백엔드 설정
# =============================================================================
LLM_BACKEND = "ollama"  # "ollama" 또는 "claude" 선택

# =============================================================================
# [3] 복합형 사양값 처리 설정 (토글)
# =============================================================================
# True: 복합값을 분리 (예: "28,260 kW / 72.0 rpm" → [("28,260", "kW"), ("72.0", "rpm")])
# False: 복합값을 통째로 저장 (예: "28,260 kW / 72.0 rpm" 그대로 저장, 단위는 빈 값)
SPLIT_COMPOUND_VALUES = False

# 범위형 사양값 처리
# True: 범위를 분리 (예: "10 ~ 20" → [("10", ""), ("20", "")])
# False: 범위를 통째로 저장 (예: "10 ~ 20" 그대로 저장)
SPLIT_RANGE_VALUES = False

# =============================================================================
# [4] Light 모드 전용 설정
# =============================================================================
LIGHT_MODE_MAX_FILES = 500            # 처리할 최대 파일 수
LIGHT_MODE_DEFAULT_SPECS = 6          # 템플릿 없을 때 추출할 기본 사양수
LIGHT_MODE_VOTING_DISABLED = True     # Voting 비활성화
LIGHT_MODE_AUDIT_DISABLED = True      # 2차 감사 비활성화
LIGHT_MODE_CHECKPOINT_DISABLED = True # checkpoint 비활성화

# =============================================================================
# [5] Full 모드 전용 설정
# =============================================================================
FULL_MODE_BATCH_SIZE = 15             # 배치 크기
FULL_MODE_CHECKPOINT_INTERVAL = 50    # checkpoint 저장 주기
FULL_MODE_VOTING_ENABLED = True       # Voting 활성화
FULL_MODE_VOTE_K = 2                  # Voting 포트 수
FULL_MODE_AUDIT_ENABLED = True        # 2차 감사 활성화

# =============================================================================
# [6] Verify 모드 전용 설정
# =============================================================================
VERIFY_MODE_BATCH_SIZE = 10
VERIFY_MODE_CONFIDENCE_THRESHOLD = 0.7

# =============================================================================
# [7] Ollama LLM 설정
# =============================================================================
USER_OLLAMA_MODEL = "gemma3:27b"      # 또는 "qwen2.5:32b-instruct"
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

# DB 테이블명 (스키마 없이 테이블명만 사용)
USER_DB_TABLE_GLOSSARY = "pos_dict"       # 용어집 테이블
USER_DB_TABLE_SPECDB = "umgv_fin"         # 사양값DB 테이블
USER_DB_TABLE_TEMPLATE = "ext_tmpl"       # 템플릿 테이블
USER_DB_TABLE_RESULT = "ext_rslt"         # 결과 저장 테이블
USER_DB_TABLE_EMBEDDING_MST = "embedding_mst"
USER_DB_TABLE_POS_EMBEDDING = "pos_embedding"

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
# [12] 임베딩 설정
# =============================================================================
USER_EMBEDDING_MODEL_PATH = "/workspace/bge-m3"
USER_EMBEDDING_DEVICE = "cuda"
USE_PRECOMPUTED_EMBEDDINGS = True
EMBEDDING_TOP_K = 5
EMBEDDING_SIMILARITY_THRESHOLD = 0.65

# =============================================================================
# [13] Rule 기반 추출 설정
# =============================================================================
USER_RULE_CONF_THRESHOLD = 0.72
USER_FORCE_LLM_ON_ALL = False

# =============================================================================
# [14] 2차 감사 설정
# =============================================================================
USER_AUDIT_CONF_LOW = 0.5
USER_AUDIT_CONF_HIGH = 0.9
USER_AUDIT_MAX_FRACTION = 0.3

# =============================================================================
# [15] Evidence 설정
# =============================================================================
USER_MAX_EVIDENCE_CHARS = 15000
USER_EVIDENCE_MAX_TABLES = 10

# =============================================================================
# [16] 대표호선 시스템 설정
# =============================================================================
SERIES_HULL_RANGE = 10                # ±10 범위 내 시리즈 호선 그룹핑
USE_REPRESENTATIVE_HULL_HINT = True   # 대표호선 힌트 사용 여부
REPRESENTATIVE_HULL_CACHE_ENABLED = True  # 캐싱 활성화

# =============================================================================
# [17] 디버그 설정
# =============================================================================
USER_DEBUG = True


# ############################################################################
# 체크박스 패턴 (하드코딩 - 표준화된 패턴이므로 DB 불필요)
# ############################################################################

CHECKBOX_PATTERNS = {
    # Y/N/Q 괄호 패턴
    'YNQ_BRACKET': re.compile(r'\(([YNQ])\)', re.IGNORECASE),
    
    # 체크된 박스 패턴
    'CHECKED_SQUARE': re.compile(r'\[x\]|\[X\]|■|☑|✓|✔|√'),
    'UNCHECKED_SQUARE': re.compile(r'\[\s*\]|□|☐'),
    
    # 동그라미 패턴
    'CHECKED_CIRCLE': re.compile(r'●|◉'),
    'UNCHECKED_CIRCLE': re.compile(r'○|◯'),
    
    # O/X 패턴
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
    """파일명에서 Hull 번호 추출"""
    match = re.search(r'^(\d{4})', filename)
    return match.group(1) if match else ""


def extract_doknr_from_filename(filename: str) -> str:
    """파일명에서 문서번호(doknr) 추출 - 예: 5533-POS-0070101"""
    match = re.match(r'(\d{4}-POS-\d+)', filename)
    return match.group(1) if match else ""


def detect_checkbox_selection(text: str) -> Optional[str]:
    """
    체크박스 선택 상태 감지
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        "Y", "N", "Q", 또는 None
    """
    if not text:
        return None
    
    # Y/N/Q 괄호 패턴 - 가장 우선
    ynq_match = CHECKBOX_PATTERNS['YNQ_BRACKET'].search(text)
    if ynq_match:
        return ynq_match.group(1).upper()
    
    # 체크된 박스 감지
    if CHECKBOX_PATTERNS['CHECKED_SQUARE'].search(text):
        return "Y"
    if CHECKBOX_PATTERNS['CHECKED_CIRCLE'].search(text):
        return "Y"
    if CHECKBOX_PATTERNS['OX_CHECKED'].search(text):
        return "Y"
    
    # 체크 안된 박스 감지
    if CHECKBOX_PATTERNS['UNCHECKED_SQUARE'].search(text):
        return "N"
    if CHECKBOX_PATTERNS['UNCHECKED_CIRCLE'].search(text):
        return "N"
    if CHECKBOX_PATTERNS['OX_UNCHECKED'].search(text):
        return "N"
    
    return None


def parse_compound_value(raw_value: str, split_enabled: bool = True) -> List[Tuple[str, str]]:
    """
    복합값 파싱 (슬래시로 구분된 여러 값)
    
    Args:
        raw_value: 원본 값 문자열 (예: "28,260 kW / 72.0 rpm")
        split_enabled: 분리 활성화 여부
        
    Returns:
        [(값, 단위), ...] 리스트
    """
    if not raw_value or not raw_value.strip():
        return []
    
    raw_value = raw_value.strip()
    
    # 분리 비활성화 시 통째로 반환
    if not split_enabled:
        return [(raw_value, "")]
    
    # 단위 보호 패턴 (m3/h, kg/h 등은 분리하지 않음)
    protected_units = ['m3/h', 'kg/h', 'l/h', 'nm3/h', 'kj/kg', 'w/m2', 'kg/m3']
    protected_placeholders = {}
    temp_value = raw_value.lower()
    
    for i, unit in enumerate(protected_units):
        placeholder = f"__UNIT_{i}__"
        temp_value = temp_value.replace(unit, placeholder)
        protected_placeholders[placeholder] = unit
    
    # 슬래시로 분리
    parts = re.split(r'\s*/\s*', raw_value)
    
    if len(parts) == 1:
        # 단일 값
        return [(raw_value, "")]
    
    results = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # 숫자와 단위 분리
        match = re.match(r'^([0-9,.\-\s]+)\s*([a-zA-Z°℃%/]+.*)?$', part)
        if match:
            val = match.group(1).strip()
            unit = match.group(2).strip() if match.group(2) else ""
            results.append((val, unit))
        else:
            results.append((part, ""))
    
    return results if results else [(raw_value, "")]


def parse_range_value(raw_value: str, split_enabled: bool = True) -> List[Tuple[str, str]]:
    """
    범위형 값 파싱 (예: "10 ~ 20", "10-20")
    
    Args:
        raw_value: 원본 값 문자열
        split_enabled: 분리 활성화 여부
        
    Returns:
        [(값, 단위), ...] 리스트
    """
    if not raw_value or not raw_value.strip():
        return []
    
    raw_value = raw_value.strip()
    
    # 분리 비활성화 시 통째로 반환
    if not split_enabled:
        # 단위 추출 시도
        unit_match = re.search(r'([a-zA-Z°℃%/]+)\s*$', raw_value)
        unit = unit_match.group(1) if unit_match else ""
        return [(raw_value, unit)]
    
    # 범위 패턴 (~ 또는 - 로 구분)
    range_match = re.match(r'^([0-9,.\s]+)\s*[~\-]\s*([0-9,.\s]+)\s*([a-zA-Z°℃%]+)?$', raw_value)
    
    if range_match:
        val1 = range_match.group(1).strip()
        val2 = range_match.group(2).strip()
        unit = range_match.group(3) or ""
        return [(val1, unit), (val2, unit)]
    
    return [(raw_value, "")]


# ############################################################################
# 데이터 클래스 (Data Classes)
# ############################################################################

@dataclass
class Config:
    """추출기 설정"""
    # 모드 설정
    mode: str = EXTRACTION_MODE
    data_source: str = DATA_SOURCE_MODE
    llm_backend: str = LLM_BACKEND
    
    # 복합값 처리 설정
    split_compound_values: bool = SPLIT_COMPOUND_VALUES
    split_range_values: bool = SPLIT_RANGE_VALUES
    
    # Ollama 설정
    ollama_model: str = USER_OLLAMA_MODEL
    ollama_host: str = USER_OLLAMA_HOST
    ollama_ports: List[int] = field(default_factory=lambda: USER_OLLAMA_PORTS.copy())
    ollama_timeout: int = USER_OLLAMA_TIMEOUT_SEC
    llm_temperature: float = USER_LLM_TEMPERATURE
    llm_rate_limit: float = USER_LLM_RATE_LIMIT_SEC
    llm_max_retries: int = USER_LLM_MAX_RETRIES
    llm_retry_sleep: float = USER_LLM_RETRY_SLEEP_SEC
    
    # Claude 설정
    claude_model: str = USER_CLAUDE_MODEL
    claude_max_tokens: int = USER_CLAUDE_MAX_TOKENS
    claude_api_key: str = USER_CLAUDE_API_KEY
    
    # 병렬 처리 설정
    enable_parallel: bool = USER_ENABLE_PARALLEL
    num_workers: int = USER_NUM_WORKERS
    llm_workers: int = USER_LLM_WORKERS
    
    # DB 설정
    db_host: str = USER_DB_HOST
    db_port: int = USER_DB_PORT
    db_name: str = USER_DB_NAME
    db_user: str = USER_DB_USER
    db_password: str = USER_DB_PASSWORD
    
    # 경로 설정
    base_folder: str = USER_BASE_FOLDER
    glossary_path: str = USER_GLOSSARY_PATH
    spec_path: str = USER_SPEC_PATH
    specdb_path: str = USER_SPECDB_PATH
    output_path: str = USER_OUTPUT_PATH
    partial_output_path: str = USER_PARTIAL_OUTPUT_PATH
    
    # 임베딩 설정
    embedding_model_path: str = USER_EMBEDDING_MODEL_PATH
    embedding_device: str = USER_EMBEDDING_DEVICE
    use_precomputed_embeddings: bool = USE_PRECOMPUTED_EMBEDDINGS
    embedding_top_k: int = EMBEDDING_TOP_K
    embedding_similarity_threshold: float = EMBEDDING_SIMILARITY_THRESHOLD
    
    # 추출 설정
    rule_conf_threshold: float = USER_RULE_CONF_THRESHOLD
    force_llm_on_all: bool = USER_FORCE_LLM_ON_ALL
    max_evidence_chars: int = USER_MAX_EVIDENCE_CHARS
    evidence_max_tables: int = USER_EVIDENCE_MAX_TABLES
    
    # 대표호선 설정
    series_hull_range: int = SERIES_HULL_RANGE
    use_representative_hull_hint: bool = USE_REPRESENTATIVE_HULL_HINT
    representative_hull_cache_enabled: bool = REPRESENTATIVE_HULL_CACHE_ENABLED
    
    # 디버그 설정
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
    umgv_desc: str = ""       # 표준 사양명
    umgv_uom: str = ""        # 표준 단위
    spec_name: str = ""       # 검색에 사용할 사양명
    existing_value: str = ""  # 검증 모드에서 비교할 기존 값


@dataclass
class ExtractionHint:
    """추출 힌트 정보"""
    # 3-Tier 매칭 정보
    match_tier: int = 0  # 0: 없음, 1: 정확, 2: 부분, 3: 최소
    
    # 용어집 정보
    pos_umgv_desc: str = ""      # POS 문서에서 사용하는 용어 (유의어)
    section_num: str = ""
    table_text: str = ""
    value_format: str = ""
    
    # 과거 값 정보
    historical_values: List[str] = field(default_factory=list)
    historical_units: List[str] = field(default_factory=list)
    
    # 대표호선 정보
    representative_hull: str = ""
    representative_value: str = ""
    representative_unit: str = ""
    representative_position: str = ""
    
    # 동의어 목록
    synonyms: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """추출 결과"""
    value: str = ""
    unit: str = ""
    confidence: float = 0.0
    method: str = ""           # "rule", "llm", "NOT_FOUND", "cache"
    found: bool = True         # 값을 찾았는지 여부
    evidence: str = ""         # 근거 텍스트
    position: str = ""         # 문서 내 위치
    compound_values: List[Tuple[str, str]] = field(default_factory=list)  # 복합값
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "method": self.method,
            "found": self.found,
            "evidence": self.evidence[:200] if self.evidence else "",
            "position": self.position,
            "compound_values": self.compound_values
        }


# ############################################################################
# PostgreSQL 데이터 로더
# ############################################################################

class PostgresLoader:
    """PostgreSQL 데이터 로더"""
    
    def __init__(self, config: Config):
        self.config = config
        self.conn = None
        self.logger = logging.getLogger("PostgresLoader")
        
        # 동의어 매핑 캐시 (umgv_desc → [pos_umgv_desc, ...])
        self.synonym_map: Dict[str, List[str]] = defaultdict(list)
        
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
        """
        용어집 로드 및 동의어 매핑 구축
        
        Returns:
            용어집 DataFrame
        """
        if not HAS_PANDAS:
            return pd.DataFrame()
        
        # DB에서 로드 시도
        if self.conn:
            try:
                # 올바른 SQL 쿼리 (테이블명만 사용, IP 주소 제외)
                query = f"SELECT * FROM {USER_DB_TABLE_GLOSSARY}"
                df = pd.read_sql(query, self.conn)
                self.logger.info(f"DB 용어집 로드 완료: {len(df)}건")
                
                # 동의어 매핑 구축
                self._build_synonym_map(df)
                
                return df
            except Exception as e:
                self.logger.warning(f"DB 용어집 로드 실패, 파일로 대체: {e}")
        
        # 파일에서 로드
        if os.path.exists(self.config.glossary_path):
            try:
                df = pd.read_csv(self.config.glossary_path, sep='\t', encoding='utf-8', 
                                on_bad_lines='skip')
                self.logger.info(f"파일 용어집 로드 완료: {len(df)}건")
                
                # 동의어 매핑 구축
                self._build_synonym_map(df)
                
                return df
            except Exception as e:
                self.logger.warning(f"파일 용어집 로드 실패: {e}")
        else:
            self.logger.warning(f"용어집 파일 없음: {self.config.glossary_path}")
        
        return pd.DataFrame()
    
    def _build_synonym_map(self, df: pd.DataFrame):
        """
        동의어 매핑 구축 (umgv_desc → pos_umgv_desc)
        
        Args:
            df: 용어집 DataFrame
        """
        self.synonym_map.clear()
        
        if 'umgv_desc' not in df.columns or 'pos_umgv_desc' not in df.columns:
            self.logger.warning("용어집에 umgv_desc 또는 pos_umgv_desc 컬럼 없음")
            return
        
        for _, row in df.iterrows():
            std_name = norm(row.get('umgv_desc', ''))
            pos_name = norm(row.get('pos_umgv_desc', ''))
            
            if std_name and pos_name and std_name != pos_name:
                # 표준명 → 유의어 매핑
                if pos_name not in self.synonym_map[std_name.upper()]:
                    self.synonym_map[std_name.upper()].append(pos_name)
                
                # 유의어 → 표준명 역매핑도 추가
                if std_name not in self.synonym_map[pos_name.upper()]:
                    self.synonym_map[pos_name.upper()].append(std_name)
        
        self.logger.info(f"동의어 매핑 구축 완료: {len(self.synonym_map)}개 그룹")
    
    def get_synonyms(self, spec_name: str) -> List[str]:
        """
        사양명의 동의어 목록 반환
        
        Args:
            spec_name: 사양명
            
        Returns:
            동의어 목록
        """
        if not spec_name:
            return []
        
        key = spec_name.upper().strip()
        return self.synonym_map.get(key, [])
    
    def load_specdb(self) -> pd.DataFrame:
        """사양값DB 로드"""
        if not HAS_PANDAS:
            return pd.DataFrame()
        
        if self.conn:
            try:
                query = f"SELECT * FROM {USER_DB_TABLE_SPECDB}"
                df = pd.read_sql(query, self.conn)
                self.logger.info(f"DB 사양값DB 로드 완료: {len(df)}건")
                return df
            except Exception as e:
                self.logger.warning(f"DB 사양값DB 로드 실패, 파일로 대체: {e}")
        
        if os.path.exists(self.config.specdb_path):
            try:
                df = pd.read_csv(self.config.specdb_path, sep='\t', encoding='utf-8',
                                on_bad_lines='skip')
                self.logger.info(f"파일 사양값DB 로드 완료: {len(df)}건")
                return df
            except Exception as e:
                self.logger.warning(f"파일 사양값DB 로드 실패: {e}")
        else:
            self.logger.warning(f"사양값DB 파일 없음: {self.config.specdb_path}")
        
        return pd.DataFrame()
    
    def load_template(self) -> pd.DataFrame:
        """템플릿 로드"""
        if not HAS_PANDAS:
            return pd.DataFrame()
        
        if self.conn:
            try:
                query = f"SELECT * FROM {USER_DB_TABLE_TEMPLATE}"
                df = pd.read_sql(query, self.conn)
                self.logger.info(f"DB 템플릿 로드 완료: {len(df)}건")
                return df
            except Exception as e:
                self.logger.warning(f"DB 템플릿 로드 실패, 파일로 대체: {e}")
        
        if os.path.exists(self.config.spec_path):
            try:
                df = pd.read_csv(self.config.spec_path, sep='\t', encoding='utf-8',
                                on_bad_lines='skip')
                self.logger.info(f"파일 템플릿 로드 완료: {len(df)}건")
                return df
            except Exception as e:
                self.logger.warning(f"파일 템플릿 로드 실패: {e}")
        else:
            self.logger.warning(f"템플릿 파일 없음: {self.config.spec_path}")
        
        return pd.DataFrame()
    
    def get_template_for_pos(self, doknr: str, df_template: pd.DataFrame) -> pd.DataFrame:
        """
        특정 POS 문서에 대한 템플릿 반환
        
        Args:
            doknr: 문서번호 (예: 5533-POS-0070101)
            df_template: 전체 템플릿 DataFrame
            
        Returns:
            해당 문서의 템플릿 DataFrame
        """
        if df_template.empty:
            return pd.DataFrame()
        
        # doknr로 필터링
        if 'doknr' in df_template.columns:
            # 정확한 매칭
            mask = df_template['doknr'].apply(lambda x: doknr in str(x) if pd.notna(x) else False)
            filtered = df_template[mask]
            
            if not filtered.empty:
                return filtered
            
            # POS 번호만으로 매칭 시도
            pos_num = re.search(r'(\d+)$', doknr)
            if pos_num:
                pos_num = pos_num.group(1)
                mask = df_template['doknr'].apply(
                    lambda x: pos_num in str(x) if pd.notna(x) else False
                )
                filtered = df_template[mask]
                if not filtered.empty:
                    return filtered
        
        return pd.DataFrame()
    
    def save_result(self, results: List[Dict]) -> bool:
        """결과 DB 저장"""
        if not self.conn or not results:
            return False
        
        try:
            cursor = self.conn.cursor()
            
            for result in results:
                # INSERT 또는 UPDATE
                cursor.execute(f"""
                    INSERT INTO {USER_DB_TABLE_RESULT} 
                    (doknr, umgv_code, extracted_value, extracted_unit, 
                     confidence, method, extracted_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doknr, umgv_code) 
                    DO UPDATE SET 
                        extracted_value = EXCLUDED.extracted_value,
                        extracted_unit = EXCLUDED.extracted_unit,
                        confidence = EXCLUDED.confidence,
                        method = EXCLUDED.method,
                        extracted_at = EXCLUDED.extracted_at
                """, (
                    result.get('doknr', ''),
                    result.get('umgv_code', ''),
                    result.get('value', ''),
                    result.get('unit', ''),
                    result.get('confidence', 0.0),
                    result.get('method', ''),
                    datetime.now()
                ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"결과 저장 실패: {e}")
            self.conn.rollback()
            return False


# ############################################################################
# 힌트 엔진 (Reference Hint Engine)
# ############################################################################

class ReferenceHintEngine:
    """
    용어집과 사양값DB를 활용한 3-Tier 힌트 매칭 엔진
    """
    
    def __init__(self, config: Config, db_loader: PostgresLoader):
        self.config = config
        self.db_loader = db_loader
        self.logger = logging.getLogger("ReferenceHintEngine")
        
        # 3-Tier 인덱스
        self.glossary_full_idx: Dict[str, List[Dict]] = defaultdict(list)   # Tier 1: 정확 매칭
        self.glossary_partial_idx: Dict[str, List[Dict]] = defaultdict(list) # Tier 2: 부분 매칭
        self.glossary_umgv_idx: Dict[str, List[Dict]] = defaultdict(list)    # Tier 3: 최소 매칭
        
        # 사양값DB 인덱스
        self.specdb_idx: Dict[str, List[Dict]] = defaultdict(list)
        
        # 대표호선 캐시
        self.hull_series_cache: Dict[str, str] = {}  # hull -> representative hull
        self.hull_results_cache: Dict[str, Dict[str, ExtractionResult]] = defaultdict(dict)
        
    def build_indexes(self, df_glossary: pd.DataFrame, df_specdb: pd.DataFrame):
        """
        3-Tier 인덱스 구축
        """
        # 용어집 인덱스 구축
        for _, row in df_glossary.iterrows():
            entry = row.to_dict()
            
            pmg_code = norm(entry.get('pmg_code', ''))
            umg_code = norm(entry.get('umg_code', ''))
            extwg = norm(entry.get('extwg', ''))
            umgv_code = norm(entry.get('umgv_code', ''))
            
            # Tier 1: 정확 매칭 (pmg_code + umg_code + extwg + umgv_code)
            if pmg_code and umg_code and extwg and umgv_code:
                key1 = f"{pmg_code}_{umg_code}_{extwg}_{umgv_code}"
                self.glossary_full_idx[key1].append(entry)
            
            # Tier 2: 부분 매칭 (pmg_code + umg_code + umgv_code, extwg 제외)
            if pmg_code and umg_code and umgv_code:
                key2 = f"{pmg_code}_{umg_code}_{umgv_code}"
                self.glossary_partial_idx[key2].append(entry)
            
            # Tier 3: 최소 매칭 (umgv_code만)
            if umgv_code:
                self.glossary_umgv_idx[umgv_code].append(entry)
        
        # 사양값DB 인덱스 구축
        for _, row in df_specdb.iterrows():
            entry = row.to_dict()
            umgv_code = norm(entry.get('umgv_code', ''))
            if umgv_code:
                self.specdb_idx[umgv_code].append(entry)
        
        self.logger.info(
            f"힌트 인덱스 구축: glossary_full={len(self.glossary_full_idx)}, "
            f"glossary_partial={len(self.glossary_partial_idx)}, "
            f"glossary_umgv={len(self.glossary_umgv_idx)}, "
            f"specdb={len(self.specdb_idx)}"
        )
    
    def get_hint(self, spec: SpecItem) -> ExtractionHint:
        """
        사양 항목에 대한 힌트 조회 (3-Tier 매칭)
        """
        hint = ExtractionHint()
        
        # Tier 1: 정확 매칭 시도
        key1 = f"{spec.pmg_code}_{spec.umg_code}_{spec.extwg}_{spec.umgv_code}"
        entries = self.glossary_full_idx.get(key1, [])
        
        if entries:
            hint.match_tier = 1
            self._fill_hint_from_entries(hint, entries, spec)
            return hint
        
        # Tier 2: 부분 매칭 시도
        key2 = f"{spec.pmg_code}_{spec.umg_code}_{spec.umgv_code}"
        entries = self.glossary_partial_idx.get(key2, [])
        
        if entries:
            hint.match_tier = 2
            self._fill_hint_from_entries(hint, entries, spec)
            return hint
        
        # Tier 3: 최소 매칭 시도 (umgv_code만)
        entries = self.glossary_umgv_idx.get(spec.umgv_code, [])
        
        if entries:
            # 동일 PMG/UMG 엔트리 우선
            same_pmg_umg = [e for e in entries 
                           if norm(e.get('pmg_code', '')) == spec.pmg_code 
                           and norm(e.get('umg_code', '')) == spec.umg_code]
            
            if same_pmg_umg:
                entries = same_pmg_umg
            
            hint.match_tier = 3
            self._fill_hint_from_entries(hint, entries, spec)
            return hint
        
        # 매칭 실패 - 동의어만 추가
        hint.synonyms = self.db_loader.get_synonyms(spec.spec_name)
        
        return hint
    
    def _fill_hint_from_entries(self, hint: ExtractionHint, entries: List[Dict], spec: SpecItem):
        """힌트 엔트리에서 정보 채우기"""
        if not entries:
            return
        
        # 첫 번째 엔트리에서 기본 정보
        first = entries[0]
        hint.pos_umgv_desc = norm(first.get('pos_umgv_desc', ''))
        hint.section_num = norm(first.get('section_num', ''))
        hint.table_text = norm(first.get('table_text', ''))
        hint.value_format = norm(first.get('value_format', ''))
        
        # 모든 엔트리에서 과거 값 수집
        for entry in entries:
            val = norm(entry.get('pos_umgv_value', ''))
            unit = norm(entry.get('pos_umgv_uom', ''))
            if val and val not in hint.historical_values:
                hint.historical_values.append(val)
                hint.historical_units.append(unit)
        
        # 동의어 추가
        hint.synonyms = self.db_loader.get_synonyms(spec.spec_name)
        if hint.pos_umgv_desc and hint.pos_umgv_desc not in hint.synonyms:
            hint.synonyms.insert(0, hint.pos_umgv_desc)
        
        # 사양값DB에서 추가 값 수집
        specdb_entries = self.specdb_idx.get(spec.umgv_code, [])
        for entry in specdb_entries[:5]:
            val = norm(entry.get('umgv_value_edit', ''))
            unit = norm(entry.get('umgv_uom', ''))
            if val and val not in hint.historical_values:
                hint.historical_values.append(val)
                hint.historical_units.append(unit)
    
    def get_representative_hull(self, hull: str, all_hulls: List[str]) -> str:
        """
        대표호선 반환 (시리즈 내 가장 작은 번호)
        """
        if not hull or not all_hulls:
            return hull
        
        # 캐시 확인
        if hull in self.hull_series_cache:
            return self.hull_series_cache[hull]
        
        try:
            hull_num = int(hull)
            
            # ±SERIES_HULL_RANGE 범위 내 호선 찾기
            series_hulls = []
            for h in all_hulls:
                try:
                    h_num = int(h)
                    if abs(h_num - hull_num) <= self.config.series_hull_range:
                        series_hulls.append(h)
                except ValueError:
                    continue
            
            if series_hulls:
                # 가장 작은 번호가 대표호선
                representative = min(series_hulls, key=lambda x: int(x))
                
                # 캐시에 저장
                for h in series_hulls:
                    self.hull_series_cache[h] = representative
                
                return representative
        except ValueError:
            pass
        
        return hull
    
    def cache_result(self, hull: str, spec_key: str, result: ExtractionResult, all_hulls: List[str]):
        """
        추출 결과 캐시 (대표호선 기준)
        """
        if not self.config.representative_hull_cache_enabled:
            return
        
        # 성공적인 추출만 캐시 (confidence >= 0.7)
        if result.confidence < 0.7 or not result.found:
            return
        
        representative = self.get_representative_hull(hull, all_hulls)
        self.hull_results_cache[representative][spec_key] = result
    
    def get_cached_result(self, hull: str, spec_key: str, all_hulls: List[str]) -> Optional[ExtractionResult]:
        """
        캐시된 결과 조회 (대표호선 기준)
        """
        if not self.config.representative_hull_cache_enabled:
            return None
        
        representative = self.get_representative_hull(hull, all_hulls)
        cached = self.hull_results_cache.get(representative, {}).get(spec_key)
        
        if cached:
            # 캐시 결과 복사 (method 표시)
            return ExtractionResult(
                value=cached.value,
                unit=cached.unit,
                confidence=cached.confidence * 0.95,  # 약간 낮은 신뢰도
                method="cache",
                found=cached.found,
                evidence=f"[Cached from Hull {representative}] {cached.evidence}",
                position=cached.position,
                compound_values=cached.compound_values
            )
        
        return None


# ############################################################################
# HTML 청크 파서
# ############################################################################

class HTMLChunkParser:
    """
    HTML 문서 파서 - 4-Layer 취소선 제거 포함
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.soup = None
        self.tables = []
        self.text_chunks = []
        self.full_text = ""
        
    def parse(self, html_content: str) -> bool:
        """HTML 파싱 및 취소선 제거"""
        if not HAS_BS4:
            return False
        
        try:
            self.soup = BeautifulSoup(html_content, 'html.parser')
            
            # 4-Layer 취소선 제거
            self._remove_strikethrough_4layer()
            
            # 테이블 추출
            self._extract_tables()
            
            # 텍스트 청크 추출
            self._extract_text_chunks()
            
            # 전체 텍스트
            self.full_text = self.soup.get_text(separator=' ', strip=True)
            
            return True
            
        except Exception as e:
            logger.error(f"HTML 파싱 실패: {e}")
            return False
    
    def _remove_strikethrough_4layer(self):
        """
        4-Layer 취소선 제거
        
        Layer 1: 태그 (<strike>, <del>, <s>)
        Layer 2: 인라인 스타일 (text-decoration: line-through)
        Layer 3: CSS 클래스 (.strikethrough, .line-through, .deleted)
        Layer 4: 텍스트 패턴 ([delete]...[/delete], ~~...~~)
        """
        if not self.soup:
            return
        
        # Layer 1: 태그 제거
        for tag_name in ['strike', 'del', 's']:
            for tag in self.soup.find_all(tag_name):
                tag.decompose()
        
        # Layer 2: 인라인 스타일 제거
        style_pattern = re.compile(r'text-decoration:\s*line-through', re.IGNORECASE)
        for tag in self.soup.find_all(style=style_pattern):
            tag.decompose()
        
        # Layer 3: CSS 클래스 제거
        class_patterns = ['strikethrough', 'line-through', 'deleted', 'struck', 'strike']
        for cls in class_patterns:
            for tag in self.soup.find_all(class_=re.compile(cls, re.IGNORECASE)):
                tag.decompose()
        
        # Layer 4: 텍스트 패턴 제거
        for text_node in self.soup.find_all(string=True):
            original = str(text_node)
            # [delete]...[/delete] 패턴
            modified = re.sub(r'\[delete\].*?\[/delete\]', '', original, flags=re.IGNORECASE | re.DOTALL)
            # ~~...~~ 패턴 (마크다운 스타일)
            modified = re.sub(r'~~[^~]+~~', '', modified)
            
            if modified != original:
                text_node.replace_with(modified)
    
    def _extract_tables(self):
        """테이블 추출"""
        self.tables = []
        
        for table in self.soup.find_all('table'):
            table_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                cells = []
                for cell in row.find_all(['td', 'th']):
                    text = cell.get_text(strip=True)
                    cells.append(text)
                
                if cells:
                    table_data.append(cells)
            
            if table_data:
                self.tables.append(table_data)
    
    def _extract_text_chunks(self):
        """텍스트 청크 추출 (섹션별)"""
        self.text_chunks = []
        
        # 제목 태그로 섹션 구분
        for heading in self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            section_title = heading.get_text(strip=True)
            
            # 다음 제목까지의 내용 수집
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
        
        # 섹션이 없으면 단락별로 추출
        if not self.text_chunks:
            for p in self.soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 50:  # 너무 짧은 단락 제외
                    self.text_chunks.append({
                        'title': '',
                        'content': text
                    })
    
    def search_in_tables(self, keywords: List[str]) -> List[Dict]:
        """
        테이블에서 키워드 검색
        
        Returns:
            [{"table_idx": int, "row_idx": int, "row": [...], "match": str}, ...]
        """
        results = []
        
        for t_idx, table in enumerate(self.tables):
            for r_idx, row in enumerate(table):
                row_text = ' '.join(row).lower()
                
                for keyword in keywords:
                    if keyword.lower() in row_text:
                        results.append({
                            'table_idx': t_idx,
                            'row_idx': r_idx,
                            'row': row,
                            'match': keyword
                        })
                        break
        
        return results
    
    def search_value_in_doc_with_context(self, value: str, context_chars: int = 100) -> Optional[str]:
        """
        문서에서 특정 값을 검색하고 주변 컨텍스트 반환
        """
        if not value or not self.full_text:
            return None
        
        # 정확한 값 검색
        idx = self.full_text.find(value)
        if idx == -1:
            # 숫자만 추출하여 검색
            num_only = re.sub(r'[^0-9.]', '', value)
            if num_only:
                idx = self.full_text.find(num_only)
        
        if idx != -1:
            start = max(0, idx - context_chars)
            end = min(len(self.full_text), idx + len(value) + context_chars)
            return self.full_text[start:end]
        
        return None
    
    def get_evidence_text(self, max_chars: int = 15000) -> str:
        """
        LLM에 제공할 증거 텍스트 생성
        """
        evidence_parts = []
        
        # 테이블 증거
        for i, table in enumerate(self.tables[:self.config.evidence_max_tables]):
            table_text = f"\n[Table {i+1}]\n"
            for row in table:
                table_text += ' | '.join(row) + '\n'
            evidence_parts.append(table_text)
        
        # 텍스트 청크 증거
        for chunk in self.text_chunks[:10]:
            if chunk['title']:
                evidence_parts.append(f"\n[Section: {chunk['title']}]\n{chunk['content']}")
            else:
                evidence_parts.append(chunk['content'])
        
        full_evidence = '\n'.join(evidence_parts)
        
        # 길이 제한
        if len(full_evidence) > max_chars:
            full_evidence = full_evidence[:max_chars] + "..."
        
        return full_evidence


# ############################################################################
# 규칙 기반 추출기
# ############################################################################

class ImprovedRuleExtractor:
    """
    규칙 기반 사양값 추출기 (체크박스, 복합값 지원)
    """
    
    def __init__(self, config: Config, db_loader: PostgresLoader):
        self.config = config
        self.db_loader = db_loader
        self.logger = logging.getLogger("RuleExtractor")
        
    def extract(self, parser: HTMLChunkParser, spec: SpecItem, 
                hint: ExtractionHint) -> ExtractionResult:
        """
        규칙 기반 추출 실행
        """
        # 키워드 구축 (힌트의 동의어 포함)
        keywords = self._build_keywords(spec, hint)
        
        # 1단계: 테이블 검색
        table_results = parser.search_in_tables(keywords)
        
        if table_results:
            result = self._extract_from_table_results(table_results, spec, hint)
            if result.found:
                return result
        
        # 2단계: 텍스트 청크 검색
        text_result = self._extract_from_text_chunks(parser, keywords, spec, hint)
        if text_result.found:
            return text_result
        
        # 3단계: 전체 텍스트 검색
        full_text_result = self._extract_from_full_text(parser, keywords, spec, hint)
        
        return full_text_result
    
    def _build_keywords(self, spec: SpecItem, hint: ExtractionHint) -> List[str]:
        """
        검색 키워드 구축 (우선순위 기반)
        """
        keywords = []
        
        # Priority 1: 힌트의 POS 용어 (최고 신뢰도)
        if hint.pos_umgv_desc:
            keywords.append(hint.pos_umgv_desc)
        
        # Priority 2: 원본 사양명
        if spec.spec_name:
            keywords.append(spec.spec_name)
        if spec.umgv_desc and spec.umgv_desc != spec.spec_name:
            keywords.append(spec.umgv_desc)
        
        # Priority 3: 동의어 (DB에서 로드된)
        keywords.extend(hint.synonyms)
        
        # Priority 4: 토큰화된 부분 (3글자 이상)
        if spec.spec_name:
            parts = re.split(r'[_\s\-]', spec.spec_name)
            keywords.extend([p for p in parts if len(p) >= 3])
        
        # 중복 제거 및 길이순 정렬 (긴 것이 더 구체적)
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        unique_keywords.sort(key=len, reverse=True)
        
        return unique_keywords[:15]
    
    def _extract_from_table_results(self, results: List[Dict], spec: SpecItem,
                                    hint: ExtractionHint) -> ExtractionResult:
        """테이블 검색 결과에서 값 추출"""
        for result in results:
            row = result['row']
            
            if len(row) < 2:
                continue
            
            # 매칭된 키워드 위치 찾기
            match_keyword = result['match'].lower()
            match_idx = -1
            
            for i, cell in enumerate(row):
                if match_keyword in cell.lower():
                    match_idx = i
                    break
            
            if match_idx == -1:
                continue
            
            # 다음 셀에서 값 추출 시도
            for val_idx in range(match_idx + 1, len(row)):
                cell_value = row[val_idx].strip()
                
                if not cell_value:
                    continue
                
                # 체크박스 감지
                checkbox = detect_checkbox_selection(cell_value)
                if checkbox:
                    return ExtractionResult(
                        value=checkbox,
                        unit="",
                        confidence=0.9,
                        method="rule_checkbox",
                        found=True,
                        evidence=' | '.join(row),
                        position=f"Table {result['table_idx'] + 1}, Row {result['row_idx'] + 1}"
                    )
                
                # 복합값 파싱
                compound = parse_compound_value(cell_value, self.config.split_compound_values)
                
                if compound:
                    if len(compound) == 1:
                        val, unit = compound[0]
                        # 범위형 체크
                        if '~' in val or (re.search(r'\d\s*-\s*\d', val) and 'POS' not in val):
                            range_vals = parse_range_value(cell_value, self.config.split_range_values)
                            return ExtractionResult(
                                value=cell_value if not self.config.split_range_values else range_vals[0][0],
                                unit=spec.umgv_uom or unit,
                                confidence=0.85,
                                method="rule_range",
                                found=True,
                                evidence=' | '.join(row),
                                position=f"Table {result['table_idx'] + 1}, Row {result['row_idx'] + 1}",
                                compound_values=range_vals if self.config.split_range_values else []
                            )
                        
                        return ExtractionResult(
                            value=val,
                            unit=spec.umgv_uom or unit,
                            confidence=0.85,
                            method="rule_table",
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
                            # 통째로 저장
                            return ExtractionResult(
                                value=cell_value,
                                unit="",
                                confidence=0.80,
                                method="rule_compound",
                                found=True,
                                evidence=' | '.join(row),
                                position=f"Table {result['table_idx'] + 1}, Row {result['row_idx'] + 1}"
                            )
        
        return ExtractionResult(found=False, method="rule_fail")
    
    def _extract_from_text_chunks(self, parser: HTMLChunkParser, keywords: List[str],
                                  spec: SpecItem, hint: ExtractionHint) -> ExtractionResult:
        """텍스트 청크에서 값 추출"""
        for chunk in parser.text_chunks:
            content = chunk['content']
            content_lower = content.lower()
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    # 키워드 뒤의 값 추출 시도
                    pattern = re.compile(
                        rf'{re.escape(keyword)}\s*[:\-]?\s*([0-9,.\-\s]+)\s*([a-zA-Z°℃%/]+)?',
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
        
        return ExtractionResult(found=False, method="rule_fail")
    
    def _extract_from_full_text(self, parser: HTMLChunkParser, keywords: List[str],
                                spec: SpecItem, hint: ExtractionHint) -> ExtractionResult:
        """전체 텍스트에서 값 추출"""
        full_text = parser.full_text
        
        for keyword in keywords:
            # 키워드 뒤의 값 패턴
            pattern = re.compile(
                rf'{re.escape(keyword)}\s*[:\-]?\s*([0-9,.\-\s~]+)\s*([a-zA-Z°℃%/]+)?',
                re.IGNORECASE
            )
            match = pattern.search(full_text)
            
            if match:
                val = match.group(1).strip()
                unit = match.group(2) or ""
                
                # 컨텍스트 추출
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
    """
    Ollama와 Claude API를 통합한 LLM 클라이언트
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("UnifiedLLMClient")
        
        # Ollama 포트 라운드 로빈
        self.port_index = 0
        self.port_lock = threading.Lock()
        
        # Claude 클라이언트
        self.claude_client = None
        if config.llm_backend == "claude" and HAS_ANTHROPIC and config.claude_api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=config.claude_api_key)
                self.logger.info("Claude API 클라이언트 초기화 완료")
            except Exception as e:
                self.logger.warning(f"Claude API 초기화 실패: {e}")
        
        # 토큰 추적
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
    
    def generate(self, prompt: str) -> Tuple[str, int, int]:
        """
        LLM 응답 생성
        
        Returns:
            (응답 텍스트, 입력 토큰, 출력 토큰)
        """
        if self.config.llm_backend == "claude":
            return self._generate_claude(prompt)
        else:
            return self._generate_ollama(prompt)
    
    def _generate_ollama(self, prompt: str) -> Tuple[str, int, int]:
        """Ollama API 호출"""
        if not HAS_REQUESTS:
            return "", 0, 0
        
        # 포트 선택 (라운드 로빈)
        with self.port_lock:
            port = self.config.ollama_ports[self.port_index]
            self.port_index = (self.port_index + 1) % len(self.config.ollama_ports)
        
        url = f"http://{self.config.ollama_host}:{port}/api/generate"
        
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.llm_temperature
            }
        }
        
        for attempt in range(self.config.llm_max_retries):
            try:
                # Rate limit
                time.sleep(self.config.llm_rate_limit)
                
                response = requests.post(
                    url, 
                    json=payload, 
                    timeout=self.config.ollama_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    text = data.get('response', '')
                    
                    # 토큰 추정 (Ollama는 정확한 토큰 수를 제공하지 않을 수 있음)
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
            self.logger.warning("Claude 클라이언트가 초기화되지 않음")
            return "", 0, 0
        
        try:
            # Rate limit
            time.sleep(self.config.llm_rate_limit)
            
            message = self.claude_client.messages.create(
                model=self.config.claude_model,
                max_tokens=self.config.claude_max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
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
    """
    LLM 기반 사양값 추출기 (NOT_FOUND 명시적 처리)
    """
    
    def __init__(self, config: Config, llm_client: UnifiedLLMClient, db_loader: PostgresLoader):
        self.config = config
        self.llm_client = llm_client
        self.db_loader = db_loader
        self.logger = logging.getLogger("LLMExtractor")
    
    def extract(self, parser: HTMLChunkParser, spec: SpecItem, 
                hint: ExtractionHint) -> ExtractionResult:
        """LLM 기반 추출 실행"""
        
        # 프롬프트 구성
        prompt = self._build_prompt(parser, spec, hint)
        
        # LLM 호출
        response, in_tokens, out_tokens = self.llm_client.generate(prompt)
        
        if not response:
            return ExtractionResult(found=False, method="llm_fail")
        
        # 응답 파싱
        result = self._parse_response(response, spec)
        
        return result
    
    def _build_prompt(self, parser: HTMLChunkParser, spec: SpecItem, 
                      hint: ExtractionHint) -> str:
        """추출 프롬프트 구성"""
        
        # 힌트 섹션
        hint_section = self._build_hint_section(spec, hint)
        
        # 증거 텍스트
        evidence = parser.get_evidence_text(self.config.max_evidence_chars)
        
        # 동의어 목록
        synonyms_list = ', '.join(hint.synonyms[:5]) if hint.synonyms else spec.spec_name
        
        prompt = f"""You are a shipbuilding specification value extractor. Extract the value for the following specification.

## Specification to Extract
- Standard Name: {spec.umgv_desc or spec.spec_name}
- Search Terms: {synonyms_list}
- Expected Unit: {spec.umgv_uom or "unknown"}

{hint_section}

## Document Content
{evidence}

## Instructions
1. Search for the specification in the document
2. Extract the exact value as written
3. If the value contains multiple parts (e.g., "28,260 kW / 72.0 rpm"), extract as compound value
4. If checkbox pattern found (Y), (N), [x], etc., return the selection

## Response Format (JSON only)
If value IS FOUND:
{{"value": "<extracted value>", "unit": "<unit>", "confidence": <0.0-1.0>, "found": true, "position": "<where found>"}}

If value IS NOT FOUND in the document:
{{"value": "NOT_FOUND", "unit": "", "confidence": 0.0, "found": false, "position": ""}}

Respond with JSON only, no explanation."""
        
        return prompt
    
    def _build_hint_section(self, spec: SpecItem, hint: ExtractionHint) -> str:
        """힌트 섹션 구성"""
        parts = []
        
        if hint.match_tier > 0:
            parts.append(f"## Reference Hints (Tier {hint.match_tier} match)")
        else:
            parts.append("## Reference Hints")
        
        # POS 용어 변형
        if hint.pos_umgv_desc and hint.pos_umgv_desc != spec.spec_name:
            parts.append(f"- May appear as: \"{hint.pos_umgv_desc}\"")
        
        # 과거 값 예시
        if hint.historical_values:
            parts.append("- Previous values from similar documents:")
            for i, (val, unit) in enumerate(zip(hint.historical_values[:3], hint.historical_units[:3])):
                parts.append(f"  - Example {i+1}: {val} {unit}".strip())
        
        # 대표호선 참조
        if hint.representative_value:
            parts.append(f"- From similar hull {hint.representative_hull}: {hint.representative_value} {hint.representative_unit}")
        
        # 값 형식
        if hint.value_format:
            parts.append(f"- Value format: {hint.value_format}")
        
        return '\n'.join(parts) if len(parts) > 1 else ""
    
    def _parse_response(self, response: str, spec: SpecItem) -> ExtractionResult:
        """LLM 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if not json_match:
                return ExtractionResult(found=False, method="llm_parse_fail")
            
            data = json.loads(json_match.group())
            
            value = str(data.get('value', '')).strip()
            unit = str(data.get('unit', '')).strip()
            confidence = float(data.get('confidence', 0.0))
            found = data.get('found', True)
            position = str(data.get('position', '')).strip()
            
            # NOT_FOUND 처리
            if value.upper() == "NOT_FOUND" or not found:
                return ExtractionResult(
                    value="",
                    unit="",
                    confidence=0.0,
                    method="NOT_FOUND",
                    found=False,
                    position=position
                )
            
            # 복합값 파싱
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
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON 파싱 실패: {e}")
            return ExtractionResult(found=False, method="llm_parse_fail")
        except Exception as e:
            self.logger.warning(f"응답 파싱 오류: {e}")
            return ExtractionResult(found=False, method="llm_parse_fail")


# ############################################################################
# 검증 추출기 (5-Step Verification)
# ############################################################################

class VerificationExtractor:
    """
    5단계 검증 프로세스를 통한 사양값 검증
    """
    
    def __init__(self, config: Config, rule_extractor: ImprovedRuleExtractor,
                 llm_extractor: ImprovedLLMExtractor):
        self.config = config
        self.rule_extractor = rule_extractor
        self.llm_extractor = llm_extractor
        self.logger = logging.getLogger("VerificationExtractor")
    
    def verify(self, spec: SpecItem, parser: HTMLChunkParser, 
               hint: ExtractionHint) -> Dict:
        """
        5단계 검증 실행
        
        Returns:
            {"status": "CORRECT"|"INCORRECT"|"UNCERTAIN", 
             "existing_value": str, "extracted_value": str, ...}
        """
        existing_value = spec.existing_value
        
        # Step 1: 문서에서 기존 값 검색 및 컨텍스트 확인
        found_context = parser.search_value_in_doc_with_context(existing_value, 100)
        
        # Step 2: 규칙 기반 재추출
        rule_result = self.rule_extractor.extract(parser, spec, hint)
        
        # Step 3: (규칙 실패 시) LLM 기반 재추출
        if not rule_result.found:
            llm_result = self.llm_extractor.extract(parser, spec, hint)
            extracted_result = llm_result
        else:
            extracted_result = rule_result
        
        # Step 4 & 5: 비교 및 최종 판정
        if not extracted_result.found:
            return {
                "status": "UNCERTAIN",
                "reason": "값을 추출할 수 없음",
                "existing_value": existing_value,
                "extracted_value": "",
                "extracted_method": extracted_result.method,
                "context_found": found_context is not None
            }
        
        # 값 비교
        status, reason = self._compare_values(existing_value, extracted_result.value)
        
        return {
            "status": status,
            "reason": reason,
            "existing_value": existing_value,
            "extracted_value": extracted_result.value,
            "extracted_unit": extracted_result.unit,
            "extracted_method": extracted_result.method,
            "confidence": extracted_result.confidence,
            "context_found": found_context is not None,
            "evidence": extracted_result.evidence
        }
    
    def _compare_values(self, existing: str, extracted: str) -> Tuple[str, str]:
        """값 비교 및 판정"""
        if not existing or not extracted:
            return "UNCERTAIN", "비교할 값 없음"
        
        existing = existing.strip()
        extracted = extracted.strip()
        
        # 정확히 일치
        if existing == extracted:
            return "CORRECT", "정확 일치"
        
        # 대소문자 무시 비교
        if existing.lower() == extracted.lower():
            return "CORRECT", "대소문자 무시 일치"
        
        # 숫자만 비교
        existing_nums = re.sub(r'[^0-9.]', '', existing)
        extracted_nums = re.sub(r'[^0-9.]', '', extracted)
        
        if existing_nums and extracted_nums:
            try:
                if abs(float(existing_nums) - float(extracted_nums)) < 0.01:
                    return "CORRECT", "숫자 일치 (포맷 차이)"
            except ValueError:
                pass
        
        # 포함 관계
        if existing in extracted or extracted in existing:
            return "UNCERTAIN", "부분 일치"
        
        return "INCORRECT", "값 불일치"


# ############################################################################
# 토큰 추적기
# ############################################################################

class TokenTracker:
    """LLM 토큰 사용량 및 추출 통계 추적"""
    
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.calls = 0
        self.rule_success = 0
        self.llm_fallback = 0
        self.not_found = 0
        self.failures = 0
        self.start_time = time.time()
    
    def add_tokens(self, input_tokens: int, output_tokens: int):
        """토큰 추가"""
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.calls += 1
    
    def add_rule_success(self):
        """규칙 성공 카운트"""
        self.rule_success += 1
    
    def add_llm_fallback(self):
        """LLM 폴백 카운트"""
        self.llm_fallback += 1
    
    def add_not_found(self):
        """NOT_FOUND 카운트"""
        self.not_found += 1
    
    def add_failure(self):
        """실패 카운트"""
        self.failures += 1
    
    def get_elapsed(self) -> float:
        """경과 시간"""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict:
        """요약 반환"""
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
            "llm_calls": self.calls,
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "elapsed_seconds": self.get_elapsed()
        }
    
    def get_progress_str(self) -> str:
        """진행 상황 문자열"""
        total = self.rule_success + self.llm_fallback + self.not_found + self.failures
        elapsed = int(self.get_elapsed())
        return (f"[Model: {LLM_BACKEND}] "
                f"[IN:{self.total_input:,} OUT:{self.total_output:,}] "
                f"[LLM:{self.calls}] "
                f"[Rule:{self.rule_success}/{total}] "
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
        
        # uploads 폴더 확인
        upload_dir = "/mnt/user-data/uploads"
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                if f.endswith('.html') and 'POS' in f.upper():
                    files.append(os.path.join(upload_dir, f))
        
        # base_folder 확인
        if os.path.exists(self.config.base_folder):
            for root, dirs, filenames in os.walk(self.config.base_folder):
                for f in filenames:
                    if f.endswith('.html'):
                        files.append(os.path.join(root, f))
        
        # 제한
        if EXTRACTION_MODE == "light" and len(files) > LIGHT_MODE_MAX_FILES:
            files = files[:LIGHT_MODE_MAX_FILES]
        
        self.logger.info(f"POS 파일 {len(files)}개 발견")
        return files


# ############################################################################
# 메인 추출기 클래스
# ############################################################################

class POSExtractor:
    """
    POS 사양값 추출기 메인 클래스
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("POSExtractor")
        
        # 컴포넌트 초기화
        self.db_loader = PostgresLoader(config)
        self.hint_engine = None
        self.llm_client = None
        self.rule_extractor = None
        self.llm_extractor = None
        self.verification_extractor = None
        self.file_finder = FileFinder(config)
        self.token_tracker = TokenTracker()
        
        # 데이터
        self.df_glossary = pd.DataFrame()
        self.df_specdb = pd.DataFrame()
        self.df_template = pd.DataFrame()
        
        self.logger.info(f"POSExtractor 초기화 (모드: {config.mode}, LLM: {config.llm_backend})")
    
    def initialize(self):
        """초기화"""
        self.logger.info("초기화 시작...")
        
        # DB 연결
        if self.config.data_source == "db":
            self.db_loader.connect()
        
        # 데이터 로드
        self.df_glossary = self.db_loader.load_glossary()
        self.df_specdb = self.db_loader.load_specdb()
        self.df_template = self.db_loader.load_template()
        
        # 힌트 엔진 초기화
        self.hint_engine = ReferenceHintEngine(self.config, self.db_loader)
        self.hint_engine.build_indexes(self.df_glossary, self.df_specdb)
        
        # LLM 클라이언트 초기화
        self.llm_client = UnifiedLLMClient(self.config)
        
        # 추출기 초기화
        self.rule_extractor = ImprovedRuleExtractor(self.config, self.db_loader)
        self.llm_extractor = ImprovedLLMExtractor(self.config, self.llm_client, self.db_loader)
        self.verification_extractor = VerificationExtractor(
            self.config, self.rule_extractor, self.llm_extractor
        )
        
        # 출력 디렉토리 생성
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
        
        # POS 파일 찾기
        pos_files = self.file_finder.find_pos_files()
        
        if not pos_files:
            return {"status": "error", "message": "POS 파일 없음"}
        
        results = []
        
        # 진행 표시줄
        pbar = tqdm(pos_files, desc="추출 진행")
        
        for filepath in pbar:
            try:
                file_result = self._process_single_file(filepath)
                results.append(file_result)
                
                # 진행 상황 업데이트
                pbar.set_postfix_str(self.token_tracker.get_progress_str())
                
            except Exception as e:
                self.logger.error(f"파일 처리 오류 {filepath}: {e}")
                results.append({
                    "file": os.path.basename(filepath),
                    "status": "error",
                    "message": str(e)
                })
        
        pbar.close()
        
        # 결과 저장
        output_file = self._save_results(results)
        
        # 통계 출력
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
        
        # HTML 파싱
        parser = HTMLChunkParser(self.config)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except Exception as e:
            return {"file": filename, "status": "read_error", "message": str(e)}
        
        if not parser.parse(html_content):
            return {"file": filename, "status": "parse_error"}
        
        # 템플릿 조회
        template = self.db_loader.get_template_for_pos(doknr, self.df_template)
        
        if template.empty:
            self.logger.info(f"{filename}: 템플릿 없음, 기본 추출 시도")
            specs = self._generate_default_specs(parser, hull)
        else:
            specs = self._template_to_specs(template)
        
        # 사양별 추출
        extractions = []
        all_hulls = self._get_all_hulls()
        
        for spec in specs:
            # 캐시 확인
            spec_key = f"{spec.umg_code}_{spec.umgv_code}"
            cached = self.hint_engine.get_cached_result(hull, spec_key, all_hulls)
            
            if cached:
                result = cached
                self.token_tracker.add_rule_success()  # 캐시도 규칙 성공으로 카운트
            else:
                # 힌트 조회
                hint = self.hint_engine.get_hint(spec)
                
                # 규칙 기반 추출
                result = self.rule_extractor.extract(parser, spec, hint)
                
                if result.found:
                    self.token_tracker.add_rule_success()
                else:
                    # LLM 폴백
                    result = self.llm_extractor.extract(parser, spec, hint)
                    self.token_tracker.add_tokens(
                        self.llm_client.total_input_tokens - self.token_tracker.total_input,
                        self.llm_client.total_output_tokens - self.token_tracker.total_output
                    )
                    
                    if result.found:
                        self.token_tracker.add_llm_fallback()
                    elif result.method == "NOT_FOUND":
                        self.token_tracker.add_not_found()
                    else:
                        self.token_tracker.add_failure()
                
                # 캐시 저장
                if result.found:
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
    
    def _generate_default_specs(self, parser: HTMLChunkParser, hull: str) -> List[SpecItem]:
        """템플릿이 없을 때 기본 사양 항목 생성"""
        # 테이블에서 주요 사양 패턴 탐지
        default_patterns = [
            ("TYPE", "Y0646"),
            ("CAPACITY", "Y0062"),
            ("HEAD", "Y0145"),
            ("MOTOR_KW", "Y0215"),
            ("MATERIAL", "Y0200"),
            ("QUANTITY", "Y0330"),
        ]
        
        specs = []
        for spec_name, umgv_code in default_patterns[:LIGHT_MODE_DEFAULT_SPECS]:
            specs.append(SpecItem(
                spec_name=spec_name,
                umgv_code=umgv_code,
                umgv_desc=spec_name
            ))
        
        return specs
    
    def _template_to_specs(self, template: pd.DataFrame) -> List[SpecItem]:
        """템플릿 DataFrame을 SpecItem 리스트로 변환"""
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
        """모든 Hull 번호 목록 반환"""
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
        self.logger.info(f"LLM 호출: {summary['llm_calls']}회")
        self.logger.info(f"토큰 사용: IN={summary['input_tokens']:,} OUT={summary['output_tokens']:,}")
        self.logger.info(f"소요 시간: {summary['elapsed_seconds']:.1f}초")
        self.logger.info("=" * 60)
    
    def _run_full_mode(self) -> Dict:
        """Full 모드 실행 (추후 구현)"""
        self.logger.info("Full 모드는 추후 구현 예정")
        return {"status": "not_implemented", "mode": "full"}
    
    def _run_verify_mode(self) -> Dict:
        """Verify 모드 실행 (추후 구현)"""
        self.logger.info("Verify 모드는 추후 구현 예정")
        return {"status": "not_implemented", "mode": "verify"}


# ############################################################################
# 메인 실행
# ############################################################################

def main():
    """
    메인 실행 함수
    상단의 설정 변수에 따라 추출을 실행합니다.
    """
    print("\n" + "=" * 70)
    print("POS Specification Value Extractor")
    print("=" * 70)
    
    # 설정 생성
    config = Config()
    
    # 추출기 생성
    extractor = POSExtractor(config)
    
    try:
        # 초기화
        extractor.initialize()
        
        # 실행
        result = extractor.run()
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("실행 완료")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        
        # 저장된 파일 안내
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
