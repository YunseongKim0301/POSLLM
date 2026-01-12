# -*- coding: utf-8 -*-
"""
POS 사양값/사양단위 추출기 
"""

from __future__ import annotations

import os
import re
import sys
import json
import urllib.request
import urllib.error
import time
import glob
import math
import pickle
import queue
import shutil
import signal
import hashlib
import logging
import subprocess
import threading
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union, Set
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

# numpy (선택적 - 벡터 연산용)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# requests 라이브러리 (선택적 - 없으면 urllib로 대체)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# PostgreSQL 연결 (선택적 - 없으면 파일 모드만 사용)
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

# Sentence Transformers (선택적 - 벡터 유사도 검색용)
try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_SENTENCE_TRANSFORMER = True
except ImportError:
    HAS_SENTENCE_TRANSFORMER = False


# =============================================================================
# 사용자 설정 영역 (여기만 수정)
# =============================================================================

# --------------------------------
# 실행 모드 설정
# --------------------------------
# "file": TXT/HTML 파일 입력, CSV 출력
# "db": PostgreDB 입력, JSON 출력 + DB 업로드
USER_MODE = "file"

# --------------------------------
# 파일 모드 경로 설정
# --------------------------------
# POS HTML 파일들이 들어있는 루트 폴더
USER_BASE_FOLDER = "/workspace/pos/phase3/phase3_formatted_new"

# 용어집 TXT (탭 구분)
USER_GLOSSARY_PATH = "/workspace/data/용어집.txt"

# 사양값추출 대상 TXT (탭 구분)
USER_SPEC_PATH = "/workspace/data/사양값추출_ongoing_sample.txt"

# 사양값DB (탭 구분; 깨진 개행/따옴표가 섞일 수 있어 repair loader 사용)
USER_SPECDB_PATH = "/workspace/data/사양값DB.txt"

# 최종 결과 파일 CSV/JSON (폴더만 주면 default 파일명으로 저장)
USER_OUTPUT_PATH = "/workspace/results/ongoing/samples"

# 체크포인트 저장 경로 (폴더)
USER_PARTIAL_OUTPUT_PATH = "/workspace/logging/ongoing/samples"

# --------------------------------
# DB 모드 설정 (PostgreSQL)
# --------------------------------
USER_DB_HOST = "10.131.132.116"
USER_DB_PORT = 5432
USER_DB_NAME = "managesys"
USER_DB_USER = "postgres"
USER_DB_PASSWORD = "pmg_umg!@"

# DB 테이블명
USER_DB_TABLE_GLOSSARY = "pos_dict"
USER_DB_TABLE_SPECDB = "umgv_fin"
USER_DB_TABLE_TEMPLATE = "ext_tmpl"
USER_DB_TABLE_RESULT = "ext_rslt"

# --------------------------------
# 체크포인트 기능 on/off
# --------------------------------
USER_ENABLE_CHECKPOINT = True

# --------------------------------
# LLM 모델/서버 설정
# --------------------------------
USER_USE_LLM = True
USER_OLLAMA_MODEL = "llama3.1:70b"
USER_OLLAMA_BIN = "/workspace/ollama/bin/ollama"
USER_OLLAMA_MODELS_DIR = "/workspace/models"
USER_OLLAMA_TIMEOUT_SEC = 300

# 멀티 Ollama serve 포트 풀
USER_OLLAMA_HOST = "127.0.0.1"
USER_OLLAMA_PORTS = [11434, 11436, 11438, 11440]  # 1개만 쓰면 [11434]

# 코드에서 멀티 serve 자동 실행(on/off)
USER_AUTO_START_OLLAMA_SERVES = True
USER_OLLAMA_SERVE_START_GRACE_SEC = 8

# --------------------------------
# 병렬 처리 설정(프로세스 수)
# --------------------------------
USER_ENABLE_PARALLEL = True
USER_NUM_WORKERS = 32

# --------------------------------
# 배치 처리 설정
# --------------------------------
USER_BATCH_SIZE = 15                   # 1회 LLM에 묶어 보내는 스펙 개수
USER_MAX_EVIDENCE_CHARS = 18000        # 프롬프트에 포함되는 evidence 최대 길이

# --------------------------------
# LLM 호출 안정화
# --------------------------------
USER_LLM_RATE_LIMIT_MIN_INTERVAL_SEC = 0.2   # 포트당 최소 호출 간격
USER_LLM_MAX_RETRIES = 3
USER_LLM_RETRY_BASE_SLEEP_SEC = 2.0

# --------------------------------
# Consistency vote 설정
# --------------------------------
USER_VOTE_ENABLED = True
USER_VOTE_K = 2               # 동일 배치 프롬프트를 몇 개의 서로 다른 포트로 보내 투표할지
USER_VOTE_TIEBREAK = "confidence"  # "confidence" or "first"

# --------------------------------
# 2차 검증(배치) on/off
# --------------------------------
USER_ENABLE_SECOND_AUDIT = True

# --------------------------------
# Sentence Transformer 설정 (벡터 유사도 검색)
# --------------------------------
USER_SENTENCE_TRANSFORMER_MODEL = "/workspace/bge-m3"
USER_SENTENCE_TRANSFORMER_DEVICE = "cuda"  # "cuda" or "cpu"
USER_ENABLE_VECTOR_SEARCH = True  # 벡터 기반 유사 POS 검색 활성화

# --------------------------------
# 병렬 + 멀티 serve 환경에서 일관성(재현성) 강화
# --------------------------------
USER_LLM_TEMPERATURE = 0.0

# --------------------------------
# 개선(성능/정확도) 파라미터
# --------------------------------
# 동일 POS의 다른 호선 파일 fallback 허용 여부(기본 False: 잘못된 파일 선택 방지)
USER_ALLOW_CROSS_HULL_FALLBACK = False

# evidence 구성(청크 선택) 파라미터
USER_EVIDENCE_MAX_TABLES = 15
USER_EVIDENCE_INCLUDE_MARKDOWN_TABLES = True
USER_EVIDENCE_MAX_MD_TABLES = 4
USER_EVIDENCE_MAX_CHUNKS = 10

# Rule 기반 추출이 일정 신뢰도 이상이면 LLM 호출을 생략
USER_RULE_CONF_THRESHOLD = 0.72
USER_FORCE_LLM_ON_ALL = False

# 2차 감사(LLM) 대상 선택 기준
USER_AUDIT_CONF_LOW = 0.50
USER_AUDIT_CONF_HIGH = 0.82
USER_AUDIT_MAX_FRACTION = 0.55  # 배치에서 audit 수행 최대 비율(0~1)

# --------------------------------
# 추가 설정 (누락된 상수들)
# --------------------------------
# 파일 모드 입출력 경로 (별칭)
USER_INPUT_PATH = USER_SPEC_PATH
USER_OUTPUT_DIR = USER_OUTPUT_PATH
USER_LOG_DIR = USER_PARTIAL_OUTPUT_PATH
USER_HTML_DIRS = [USER_BASE_FOLDER]

# 모델 설정 (별칭)
USER_MODEL_NAME = USER_OLLAMA_MODEL
USER_LLM_TIMEOUT = USER_OLLAMA_TIMEOUT_SEC

# Rule 기반 추출 설정
USER_RULE_ENABLE = True

# 디버그 모드
USER_DEBUG = False

# Glossary 하이브리드 매칭 최대 처리 수
USER_GLOSSARY_HYBRID_LIMIT = 200000

# DB 모드 테이블명 (추가)
USER_DB_INPUT_TABLE = USER_DB_TABLE_TEMPLATE
USER_DB_INPUT_CONDITIONS = ""  # 예: "WHERE status = 'pending'"
USER_DB_OUTPUT_TABLE = USER_DB_TABLE_RESULT


# =============================================================================
# 컬럼명 정의 (영문 통일 - 코드_작성_규칙.txt 기준)
# =============================================================================

# 입력 템플릿 컬럼 (사양값추출_template.txt)
COL_PMG_LV1 = "pmg_lv1"
COL_PMG_LV2 = "pmg_lv2"
COL_PMG_CODE = "pmg_code"
COL_PMG_DESC = "pmg_desc"
COL_UMG_CODE = "umg_code"
COL_UMG_DESC = "umg_desc"
COL_EXTWG = "extwg"
COL_EXTWG_DESC = "mat_attr_desc"
COL_MATNR = "matnr"
COL_DOKNR = "doknr"
COL_UMGV_CODE = "umgv_code"
COL_UMGV_DESC = "umgv_desc"
COL_UMGV_UOM = "umgv_uom"

# 레거시 호환을 위한 한글 컬럼명 (입력 파일이 한글일 경우 매핑)
COL_HULL = "호선"
COL_POS = "POS"
COL_PMG = "PMG_NAME"
COL_UMG = "UMG_NAME"
COL_MAT_GROUP = "자재속성그룹명"
COL_SPEC_NAME = "사양항목명"
COL_EXIST_VAL = "사양값"
COL_EXIST_UNIT = "사양단위"

# =============================================================================
# 추출 결과 컬럼 (코드_작성_규칙.txt 출력 스키마 기준)
# =============================================================================
# POS 문서에서 추출해야 하는 필수 컬럼들

COL_SECTION_NUM = "section_num"       # 사양값이 추출된 섹션 번호
COL_TABLE_TEXT = "table_text"         # 테이블에서 추출 여부 ("Y" or "N")
COL_VALUE_FORMAT = "value_format"     # 값 형식 ("NUMERIC", "NUMERIC_LIKE", "TEXT", "MIXED")
COL_POS_CHUNK = "pos_chunk"           # 사양값이 추출된 POS 문서의 청크 텍스트
COL_POS_EXTWG_DESC = "pos_mat_attr_desc" # POS 문서의 자재속성그룹명
COL_POS_UMGV_DESC = "pos_umgv_desc"   # POS 문서의 사양항목명 (실제 표기)
COL_POS_UMGV_VALUE = "pos_umgv_value" # 추출된 사양값
COL_UMGV_VALUE_EDIT = "umgv_value_edit"  # 편집된 사양값 (단위변환 등, 기본값=pos_umgv_value)
COL_POS_UMGV_UOM = "pos_umgv_uom"     # 추출된 사양단위
COL_EVIDENCE_FB = "evidence_fb"       # 사용자 피드백 (초기값 빈 문자열)

# Few-shot 힌트 저장용 컬럼 (디버깅/검증용)
COL_FEWSHOT_HINTS = "_fewshot_hints"  # 사용된 few-shot 힌트 (JSON)
COL_GLOSSARY_HINTS = "_glossary_hints"  # 사용된 용어집 힌트 (JSON)
COL_SPECDB_HINTS = "_specdb_hints"    # 사용된 사양값DB 힌트 (JSON)
COL_EVIDENCE_STRATEGY = "_evidence_strategy"  # 사용된 evidence 선택 전략

# 내부 처리용 컬럼 (최종 출력에서 제외 가능)
COL_CONFIDENCE = "confidence"         # 신뢰도 (0~1)
COL_METHOD = "method"                 # 추출 방식 (RULE_TABLE, LLM_VOTE 등)
COL_AUDIT = "audit"                   # 감사 결과 (PASS, FAIL)

# v41 추가 컬럼
COL_REFERENCE_INFO = "_reference_info"      # v41: 참조 정보 (JSON)
COL_VALIDATION_NOTE = "_validation_note"    # v41: 검증 메모

# 하위 호환성 별칭
COL_SPEC_VALUE = COL_POS_UMGV_VALUE
COL_SPEC_UNIT = COL_POS_UMGV_UOM
COL_EVIDENCE = COL_POS_CHUNK


# =============================================================================
# 컬럼명 매핑 (한글 -> 영문)
# =============================================================================

COLUMN_MAPPING_KR_TO_EN = {
    "호선": "hull",
    "POS": "doknr",
    "PMG_NAME": "pmg_desc",
    "UMG_NAME": "umg_desc",
    "자재속성그룹명": "mat_attr_desc",
    "자재속성그룹": "extwg",
    "사양항목명": "umgv_desc",
    "사양값": "pos_umgv_value",
    "사양단위": "pos_umgv_uom",
    "자재번호": "matnr",
    "PMG코드": "pmg_code",
    "UMG코드": "umg_code",
    "관리Spec명": "umgv_desc",
}

# =============================================================================
# 출력 JSON 스키마 정의 (코드_작성_규칙.txt 기준)
# =============================================================================
# 출력 JSON의 각 요소가 가져야 할 컬럼 순서

OUTPUT_SCHEMA_COLUMNS = [
    # 템플릿에서 가져오는 컬럼
    "pmg_code", "pmg_desc", "umg_code", "umg_desc", 
    "extwg", "mat_attr_desc", "matnr", "doknr",
    "umgv_code", "umgv_desc",
    # POS 문서에서 추출하는 컬럼
    "section_num", "table_text", "value_format", "umgv_uom",
    "pos_chunk", "pos_mat_attr_desc", "pos_umgv_desc", 
    "pos_umgv_value", "umgv_value_edit", "pos_umgv_uom",
    "evidence_fb",
]

# 추출 대상 컬럼 (템플릿에 없고 POS에서 추출해야 하는 컬럼)
EXTRACTION_TARGET_COLUMNS = [
    "section_num",      # 섹션 번호
    "table_text",       # 테이블 추출 여부 (Y/N)
    "value_format",     # 값 형식 (NUMERIC/TEXT/MIXED)
    "pos_chunk",        # 추출된 청크
    "pos_mat_attr_desc",   # POS 문서의 자재속성그룹명
    "pos_umgv_desc",    # POS 문서의 사양항목명
    "pos_umgv_value",   # 추출된 사양값
    "umgv_value_edit",  # 편집된 사양값
    "pos_umgv_uom",     # 추출된 사양단위
    "evidence_fb",      # 사용자 피드백
]


def create_empty_extraction_result() -> Dict[str, str]:
    """빈 추출 결과 딕셔너리를 생성합니다."""
    return {
        "section_num": "",
        "table_text": "",
        "value_format": "",
        "pos_chunk": "",
        "pos_mat_attr_desc": "",
        "pos_umgv_desc": "",
        "pos_umgv_value": "",
        "umgv_value_edit": "",
        "pos_umgv_uom": "",
        "evidence_fb": "",
        # v41 추가 컬럼
        "_reference_info": "",
        "_validation_note": "",
    }


# =============================================================================
# v41 검증 함수들 (2026-01-05)
# =============================================================================

def check_korean_contamination_v41(value: str) -> bool:
    """
    한글 오염 검사 (v41)
    
    POS 문서는 영어로만 작성되므로, 추출된 값에 한글이 포함되어 있으면
    용어집 few-shot이 결과에 혼입된 것으로 판단
    
    Args:
        value: 추출된 값
        
    Returns:
        True if 한글 포함 (오염됨)
    """
    if not value:
        return False
    return bool(re.search(r'[가-힣]', str(value)))


def validate_value_in_evidence_v41(
    value: str, 
    evidence: str, 
    umgv_desc: str = "",
) -> Tuple[bool, str, float]:
    """
    추출된 값이 evidence에 실제로 존재하는지 검증 (v41)
    
    검증 항목:
    1. 한글 오염 검사 (용어집 few-shot 혼입)
    2. evidence 내 값 존재 확인
    3. 타입 일관성 (CAPACITY는 양수, TYPE은 숫자가 아니어야 함)
    
    Args:
        value: 추출된 값
        evidence: 증거 텍스트 (POS chunk)
        umgv_desc: 사양항목명
        
    Returns:
        (유효여부, 검증메모, 신뢰도 조정 계수)
    """
    if not value:
        return True, "빈값_정상", 1.0
    if not evidence:
        return False, "증거없음", 0.5
    
    issues = []
    penalty = 0.0
    
    # 1. 한글 오염 검사
    if check_korean_contamination_v41(value):
        issues.append("한글오염_용어집혼입")
        penalty += 0.5
    
    # 2. evidence 내 값 존재 확인
    value_str = str(value).strip()
    value_normalized = value_str.replace(",", "").replace(" ", "")
    
    # 직접 존재 확인
    if value_str in evidence:
        pass  # OK
    elif value_normalized in evidence.replace(",", "").replace(" ", ""):
        pass  # OK (정규화 후 매칭)
    elif re.match(r'^[\d\.]+$', value_str):
        # 숫자인 경우 패턴 매칭
        pattern = r'\b' + re.escape(value_str) + r'\b'
        if not re.search(pattern, evidence):
            issues.append(f"숫자'{value_str}'_evidence미발견")
            penalty += 0.4
    else:
        # 텍스트인 경우 부분 매칭 (처음 15자)
        if len(value_str) > 5 and value_str[:15].lower() not in evidence.lower():
            issues.append(f"텍스트'{value_str[:20]}'_evidence미발견")
            penalty += 0.4
    
    # 3. 타입 일관성 검사
    umgv_lower = umgv_desc.lower() if umgv_desc else ""
    
    # CAPACITY는 양수여야 함
    if 'capacity' in umgv_lower:
        try:
            num = float(value_str.replace(',', ''))
            if num <= 0:
                issues.append(f"CAPACITY비양수({value_str})")
                penalty += 0.3
        except ValueError:
            pass
    
    # TYPE은 순수 숫자가 아니어야 함 (3자리 이상)
    if 'type' in umgv_lower:
        if re.match(r'^\d{3,}$', value_str):
            issues.append(f"TYPE숫자({value_str})")
            penalty += 0.3
    
    # TYPE이 너무 길면 문장이 추출된 것
    if 'type' in umgv_lower:
        if len(value_str) > 50:
            issues.append("TYPE너무김_문장추출의심")
            penalty += 0.3
    
    is_valid = len(issues) == 0
    note = "; ".join(issues) if issues else "OK"
    confidence_adj = max(0.0, 1.0 - penalty)
    
    return is_valid, note, confidence_adj


def clean_extracted_value_v41(value: str, umgv_desc: str) -> str:
    """
    추출된 값 정제 (v41)
    
    - 주석 제거 (*1, *2 등)
    - 개정 표시 제거 (Rev.1 등)
    - CAPACITY의 경우 첫 번째 숫자만 추출 (범위가 있는 경우)
    
    Args:
        value: 추출된 원본 값
        umgv_desc: 사양항목명
        
    Returns:
        정제된 값
    """
    if not value:
        return ""
    
    value = str(value)
    
    # 주석 제거 (*1, *2, *3 등)
    value = re.sub(r'\s*\*\d+', '', value)
    
    # 개정 표시 제거 (Rev.1, Rev.2 등)
    value = re.sub(r'\s*Rev\.\d+', '', value, flags=re.I)
    
    # CAPACITY의 경우 첫 번째 숫자만 추출 (범위가 있는 경우)
    umgv_lower = umgv_desc.lower() if umgv_desc else ""
    if 'capacity' in umgv_lower:
        # "1 - 2080" 같은 경우 첫 번째 숫자만
        match = re.match(r'([\d\.]+)', value)
        if match:
            value = match.group(1)
    
    # TYPE이 너무 길면 쉼표 전까지만
    if 'type' in umgv_lower and len(value) > 40:
        if ',' in value:
            value = value.split(',')[0].strip()
    
    return value.strip()


# =============================================================================
# v41 참조 정보 관리
# =============================================================================

class ReferenceLoggerV41:
    """
    Few-shot 참조 정보 기록 및 저장 (v41)
    
    각 추출 건별로 어떤 사양값DB, 용어집 정보를 참조했는지 기록
    """
    
    def __init__(self, output_dir: str = ""):
        self.output_dir = output_dir
        self.references: List[Dict] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_reference(
        self,
        extwg: str,
        matnr: str,
        pmg_desc: str,
        umg_desc: str,
        umgv_desc: str,
        # 사양값DB 참조 정보
        specdb_pos: str = "",
        specdb_matnr: str = "",
        specdb_hull: str = "",
        specdb_value: str = "",
        specdb_uom: str = "",
        specdb_similarity: float = 0.0,
        # 용어집 참조 정보
        glossary_pos: str = "",
        glossary_matnr: str = "",
        glossary_section: str = "",
        glossary_table_text: str = "",
        glossary_format: str = "",
        glossary_std_uom: str = "",
        # 벡터 유사도 검색 결과
        vector_top1_hull: str = "",
        vector_top1_score: float = 0.0,
        vector_top1_value: str = "",
    ):
        """참조 정보 추가"""
        self.references.append({
            # 현재 추출 대상
            'extwg': extwg,
            'matnr': matnr,
            'pmg_desc': pmg_desc,
            'umg_desc': umg_desc,
            'umgv_desc': umgv_desc,
            # 사양값DB 참조 정보
            'specdb_pos': specdb_pos,
            'specdb_matnr': specdb_matnr,
            'specdb_hull': specdb_hull,
            'specdb_value': specdb_value,
            'specdb_uom': specdb_uom,
            'specdb_similarity': specdb_similarity,
            # 용어집 참조 정보
            'glossary_pos': glossary_pos,
            'glossary_matnr': glossary_matnr,
            'glossary_section': glossary_section,
            'glossary_table_text': glossary_table_text,
            'glossary_format': glossary_format,
            'glossary_std_uom': glossary_std_uom,
            # 벡터 유사도 검색 결과
            'vector_top1_hull': vector_top1_hull,
            'vector_top1_score': vector_top1_score,
            'vector_top1_value': vector_top1_value,
        })
    
    def save(self, filename: str = "") -> str:
        """참조 로그 CSV 저장"""
        if not self.references:
            return ""
        
        df = pd.DataFrame(self.references)
        
        if not filename:
            filename = f"reference_log_{self.timestamp}.csv"
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, filename)
        else:
            filepath = filename
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath
    
    def get_count(self) -> int:
        return len(self.references)


# 전역 참조 로거
_reference_logger_v41: Optional[ReferenceLoggerV41] = None

def get_reference_logger_v41(output_dir: str = "") -> ReferenceLoggerV41:
    """전역 참조 로거 획득"""
    global _reference_logger_v41
    if _reference_logger_v41 is None:
        _reference_logger_v41 = ReferenceLoggerV41(output_dir)
    return _reference_logger_v41

def reset_reference_logger_v41(output_dir: str = "") -> ReferenceLoggerV41:
    """참조 로거 리셋 및 새로 생성"""
    global _reference_logger_v41
    _reference_logger_v41 = ReferenceLoggerV41(output_dir)
    return _reference_logger_v41


def determine_value_format(value: str) -> str:
    """
    추출된 값의 형식을 판단합니다.
    
    Returns:
        "NUMERIC": 순수 숫자 (예: "123", "45.6", "1,234")
        "NUMERIC_LIKE": 숫자와 단위/기호 혼합 (예: "123 kg", "45.6 m")
        "TEXT": 순수 텍스트 (예: "YES", "BLUE")
        "MIXED": 숫자와 텍스트 복합 (예: "2 x 100 kW")
    """
    if not value:
        return ""
    
    v = str(value).strip()
    
    # 순수 숫자 (콤마, 소수점 포함)
    if re.match(r'^[\d,\.]+$', v.replace(' ', '')):
        return "NUMERIC"
    
    # 숫자로 시작하고 단위만 붙은 경우
    if re.match(r'^[\d,\.]+\s*[a-zA-Z°%℃℉]+$', v):
        return "NUMERIC_LIKE"
    
    # 숫자가 전혀 없는 경우
    if not re.search(r'\d', v):
        return "TEXT"
    
    # 그 외 (숫자와 텍스트 혼합)
    return "MIXED"


# =============================================================================
# 공통 유틸리티 함수
# =============================================================================

def norm(x: Any) -> str:
    """
    문자열 공백/개행을 정리해 비교 가능한 형태로 만듭니다.
    None이나 NaN 값도 안전하게 빈 문자열로 변환합니다.
    """
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x)
    s = s.replace("\u00a0", " ")  # Non-breaking space 제거
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_parent_dir(path: str) -> None:
    """파일의 상위 폴더를 자동 생성합니다."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_output_path(path_or_dir: str, default_name: str, extension: str = ".csv") -> str:
    """출력 경로가 폴더면 기본 파일명을 붙여 반환합니다."""
    if path_or_dir.lower().endswith(extension):
        return path_or_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = default_name.replace(extension, "")
    return os.path.join(path_or_dir, f"{base_name}_{timestamp}{extension}")


def setup_logger() -> None:
    """과도한 로그를 억제하고, 핵심 이벤트만 INFO로 출력합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # 시끄러운 라이브러리 로그 억제
    for noisy in ["bs4", "urllib3", "chardet", "psycopg2"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def fast_hash(s: str) -> str:
    """동일 입력에 대해 고정된 짧은 키를 생성합니다."""
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def extract_hull_from_matnr(matnr: str) -> str:
    """
    matnr(자재번호)에서 호선번호를 추출합니다.
    matnr의 앞 4자리 숫자가 호선번호입니다.
    예: "2606AYS57111" -> "2606"
    예: "2377-POS-0036331" -> "2377"
    """
    if not matnr:
        return ""
    # 앞 4자리 숫자 추출
    match = re.match(r"^(\d{4})", str(matnr))
    if match:
        return match.group(1)
    # fallback: 모든 숫자 추출 후 앞 4자리
    digits = re.sub(r"\D", "", str(matnr))
    return digits[:4] if len(digits) >= 4 else digits


def extract_pos_from_matnr(matnr: str) -> str:
    """
    matnr(자재번호)에서 POS 번호를 추출합니다.
    패턴: XXXX-POS-YYYYYYY... → YYYYYYY (7자리 숫자)
    예: "2377-POS-0036331_000_02" -> "0036331"
    예: "4516-POS-0040401_000_01_A4" -> "0040401"
    """
    if not matnr:
        return ""
    matnr_str = str(matnr)
    
    # POS- 패턴 찾기
    match = re.search(r"POS[_\-]?(\d{7})", matnr_str, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # 파일명 패턴: XXXX-POS-YYYYYYY
    match = re.search(r"\d{4}[_\-]POS[_\-](\d{7})", matnr_str, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # fallback: 5자리 이상 연속 숫자 중 두번째 그룹
    digits_groups = re.findall(r"\d{5,}", matnr_str)
    if len(digits_groups) >= 2:
        return digits_groups[1][:7]
    elif len(digits_groups) == 1 and len(digits_groups[0]) >= 7:
        # 앞 4자리가 hull이면 나머지가 POS
        return digits_groups[0][4:11] if len(digits_groups[0]) > 4 else ""
    
    return ""


def extract_unit_from_header(header_text: str, spec_name: str) -> str:
    """
    v42: 테이블 헤더에서 단위를 추출합니다.
    예: "Capacity (m3/h)" → "m3/h"
    예: "Capacity (m 3 /h)" → "m3/h" (공백 정규화)
    예: "TH" → "" (단위 없음)
    예: "Temp.(℃)" → "℃"
    
    Args:
        header_text: 테이블 헤더 텍스트 (예: "Capacity (m 3 /h)")
        spec_name: 사양항목명 (예: "CAPACITY")
        
    Returns:
        추출된 단위 또는 빈 문자열
    """
    if not header_text:
        return ""
    
    # 공백 정규화
    text = re.sub(r"\s+", " ", str(header_text).strip())
    
    # 패턴 1: 괄호 안의 단위 (m3/h), (kW), (℃) 등
    # v42: 중첩 괄호 또는 여러 괄호 처리
    paren_matches = re.findall(r"\(([^)]+)\)", text)
    for paren_content in paren_matches:
        unit = paren_content.strip()
        # 공백 제거 (m 3 /h → m3/h)
        unit = re.sub(r"\s+", "", unit)
        # 유효한 단위인지 확인
        if re.match(r"^[a-zA-Z°℃³²/]+$", unit) or unit in ["m3/h", "kW", "kN", "mm", "m", "℃", "°C"]:
            return unit
    
    # 패턴 2: 점 다음의 단위 Temp.℃
    dot_match = re.search(r"\.\s*([^\s.]+)\s*$", text)
    if dot_match:
        unit = dot_match.group(1).strip()
        if unit in ["℃", "°C", "°F", "mm", "m", "kg", "kW", "bar", "MPa"]:
            return unit
    
    # 패턴 3: spec_name과 함께 제공된 단위
    # "CAPACITY (m3/h)" spec_name에서 단위 추출
    spec_paren_match = re.search(r"\(([^)]+)\)", spec_name)
    if spec_paren_match:
        spec_unit = spec_paren_match.group(1).strip()
        # 공백 제거
        spec_unit = re.sub(r"\s+", "", spec_unit)
        # | 로 구분된 경우 첫 번째 단위
        if "|" in spec_unit:
            spec_unit = spec_unit.split("|")[0].strip()
        if spec_unit and re.match(r"^[a-zA-Z°℃³²/]+$", spec_unit):
            return spec_unit
    
    # 패턴 4: 알려진 단위 직접 매칭
    known_units = {
        "m3/h": ["m3/h", "m³/h", "m 3 /h", "㎥/h", "m3h"],
        "kW": ["kW", "kw", "KW"],
        "m": ["m", "meter", "meters"],
        "mm": ["mm", "MM"],
        "bar": ["bar", "Bar", "BAR"],
        "MPa": ["MPa", "Mpa", "mpa"],
        "℃": ["℃", "°C", "deg C", "degC"],
        "rpm": ["rpm", "RPM", "r/min"],
        "ton": ["ton", "tons", "t"],
        "kg": ["kg", "KG"],
        "cSt": ["cSt", "cst", "CST"],
        "cP": ["cP", "cp", "CP"],
        "micron": ["micron", "μm", "um"],
    }
    
    for std_unit, variants in known_units.items():
        for variant in variants:
            if variant in text:
                return std_unit
    
    return ""


def extract_value_from_text_pattern(
    text: str, 
    spec_name: str,
    known_aliases: Optional[List[str]] = None,
) -> Optional[Dict[str, str]]:
    """
    v42: 텍스트에서 패턴 기반으로 사양값을 추출합니다.
    
    지원 패턴:
    1. "X : Y" 또는 "X: Y" (콜론 패턴)
    2. "X of Y (unit)" (of 패턴, 예: NPSH required of 4.5 m)
    3. "X = Y" (등호 패턴)
    4. "X shall be Y" (shall be 패턴)
    5. v42: "X" of Y unit (따옴표 포함 패턴)
    
    Args:
        text: 검색할 텍스트
        spec_name: 사양항목명 (예: "NPSH", "MATERIAL - CASING")
        known_aliases: 알려진 별칭 리스트 (예: ["NPSHr", "NPSH required"])
        
    Returns:
        {"value": ..., "unit": ..., "matched_desc": ..., "pattern": ...} 또는 None
    """
    if not text or not spec_name:
        return None
    
    # 검색 키워드 준비
    keywords = [spec_name.lower()]
    if known_aliases:
        keywords.extend([a.lower() for a in known_aliases])
    
    # spec_name에서 핵심 단어 추출
    # "MATERIAL - CASING" → ["material", "casing"]
    # "NPSH (NPSHa | NPSHr)" → ["npsh", "npsha", "npshr"]
    core_words = re.findall(r"[a-zA-Z]+", spec_name.lower())
    
    text_lower = text.lower()
    
    # v42: 패턴 0: "X required" of Y unit (NPSH 특화 패턴)
    # 예: 'designed to have the "NPSH required" of 4.5 m'
    npsh_patterns = [
        r'"([^"]*npsh[^"]*?)"\s*(?:of|=|is)\s*([\d.]+)\s*([a-zA-Z°℃³/]*)',
        r"'([^']*npsh[^']*?)'\s*(?:of|=|is)\s*([\d.]+)\s*([a-zA-Z°℃³/]*)",
        r'(?:have|has)\s+(?:the\s+)?["\']?([^"\']*npsh[^"\']*?)["\']?\s*(?:of)?\s*([\d.]+)\s*([a-zA-Z°℃³/]*)',
    ]
    if any(kw in ["npsh", "npshr", "npsha"] for kw in core_words + keywords):
        for pattern in npsh_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return {
                    "value": match.group(2),
                    "unit": match.group(3) if match.group(3) else "m",  # NPSH 기본 단위는 m
                    "matched_desc": match.group(1).strip(),
                    "pattern": "NPSH_SPECIAL",
                }
    
    # 패턴 1: "X : Y" 콜론 패턴
    # 예: "- Casing : Cast iron"
    for kw in keywords + core_words:
        # 콜론 앞뒤에 키워드가 있는 패턴
        pattern = rf"[-•]?\s*{re.escape(kw)}\s*:\s*([^\n\r,]+)"
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1).strip()
            # 값에서 단위 분리
            unit_match = re.search(r"([\d.]+)\s*([a-zA-Z°℃³/]+)", value)
            if unit_match:
                return {
                    "value": unit_match.group(1),
                    "unit": unit_match.group(2),
                    "matched_desc": kw,
                    "pattern": "COLON",
                }
            # 단위 없는 텍스트 값
            if value and not value.startswith((":", "-")):
                # 첫 단어만 또는 전체
                value_clean = re.sub(r"\s*[\(\[].*$", "", value).strip()
                if value_clean:
                    # v42: 원본 대소문자 유지를 위해 원본 텍스트에서 재검색
                    orig_pattern = rf"[-•]?\s*{re.escape(kw)}\s*:\s*([^\n\r,]+)"
                    orig_match = re.search(orig_pattern, text, re.IGNORECASE)
                    if orig_match:
                        orig_value = orig_match.group(1).strip()
                        orig_value_clean = re.sub(r"\s*[\(\[].*$", "", orig_value).strip()
                        return {
                            "value": orig_value_clean,
                            "unit": "",
                            "matched_desc": kw,
                            "pattern": "COLON",
                        }
                    return {
                        "value": value_clean.title() if not value_clean[0].isdigit() else value_clean,
                        "unit": "",
                        "matched_desc": kw,
                        "pattern": "COLON",
                    }
    
    # 패턴 2: "X of Y (unit)" 또는 "X" of Y unit
    # 예: 'NPSH required" of 4.5 m'
    for kw in keywords + core_words:
        pattern = rf'{re.escape(kw)}["\']?\s*(?:of|=|is|be)\s*([\d.]+)\s*([a-zA-Z°℃³/]*)'
        match = re.search(pattern, text_lower)
        if match:
            return {
                "value": match.group(1),
                "unit": match.group(2) if match.group(2) else "",
                "matched_desc": kw,
                "pattern": "OF_PATTERN",
            }
    
    # 패턴 3: "shall have X of Y" 
    # 예: 'designed to have the "NPSH required" of 4.5 m'
    for kw in keywords + core_words:
        pattern = rf'(?:have|has)\s+(?:the\s+)?["\']?{re.escape(kw)}["\']?\s*(?:of)?\s*([\d.]+)\s*([a-zA-Z°℃³/]*)'
        match = re.search(pattern, text_lower)
        if match:
            return {
                "value": match.group(1),
                "unit": match.group(2) if match.group(2) else "",
                "matched_desc": kw,
                "pattern": "HAVE_PATTERN",
            }
    
    # 패턴 4: "X : Y" 대문자 버전 (원본 텍스트에서)
    for kw in [spec_name] + (known_aliases or []):
        pattern = rf"[-•]?\s*{re.escape(kw)}\s*:\s*([^\n\r]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # 여러 값이 있으면 첫 번째만
            value = value.split(",")[0].strip()
            value = re.sub(r"\s*[\(\[].*$", "", value).strip()
            if value and len(value) < 100:
                return {
                    "value": value,
                    "unit": "",
                    "matched_desc": kw,
                    "pattern": "COLON_ORIGINAL",
                }
    
    return None


def build_spec_aliases(spec_name: str, glossary_hints: Optional[Dict] = None) -> List[str]:
    """
    v41: 사양항목명의 별칭 목록을 생성합니다.
    
    Args:
        spec_name: 사양항목명
        glossary_hints: 용어집에서 가져온 힌트 정보
        
    Returns:
        별칭 리스트
    """
    aliases = []
    
    # 1. 괄호 안의 별칭 추출
    # "NPSH (NPSHa | NPSHr)" → ["NPSHa", "NPSHr"]
    paren_match = re.search(r"\(([^)]+)\)", spec_name)
    if paren_match:
        inner = paren_match.group(1)
        # | 또는 / 로 분리
        parts = re.split(r"[|/]", inner)
        aliases.extend([p.strip() for p in parts if p.strip()])
    
    # 2. 알려진 약어 확장
    abbrev_map = {
        "TH": ["Total Head", "Head"],
        "NPSH": ["NPSH required", "NPSHr", "NPSHa", "NPSH available"],
        "FW": ["Fresh Water", "Freshwater"],
        "SW": ["Sea Water", "Seawater"],
        "CFW": ["Cooling Fresh Water", "Cooling FW"],
        "HC": ["Horizontal Centrifugal", "Horizontal"],
        "VC": ["Vertical Centrifugal", "Vertical"],
        "CIRC": ["Circulation", "Circulating"],
        "P/P": ["Pump"],
        "m3/h": ["m³/h", "㎥/h", "cubic meter per hour"],
    }
    
    for abbrev, expansions in abbrev_map.items():
        if abbrev.lower() in spec_name.lower():
            aliases.extend(expansions)
    
    # 3. 하이픈/언더스코어 변환
    if "-" in spec_name:
        aliases.append(spec_name.replace("-", " ").strip())
    if "_" in spec_name:
        aliases.append(spec_name.replace("_", " ").strip())
    
    # 4. 용어집 힌트에서 추가
    if glossary_hints:
        pos_umgv_desc = glossary_hints.get("pos_umgv_desc", "")
        if pos_umgv_desc and pos_umgv_desc != spec_name:
            aliases.append(pos_umgv_desc)
    
    return list(set(aliases))


def find_evidence_chunk_for_empty_value(
    doc_chunks: List,
    spec_name: str,
    known_aliases: Optional[List[str]] = None,
) -> Optional[str]:
    """
    v41: 값이 비어있는 경우, 해당 사양항목이 왜 비어있는지 근거가 되는 chunk를 찾습니다.
    
    예: MOTOR POWER (kW)가 POS에 "Motor output (rating) : kW" 형태로만 있고,
        실제 값이 비어있는 경우, 이 정보가 담긴 chunk를 반환합니다.
    
    Args:
        doc_chunks: 문서 청크 리스트
        spec_name: 사양항목명
        known_aliases: 별칭 리스트
        
    Returns:
        근거 chunk 텍스트 또는 None
    """
    if not doc_chunks:
        return None
    
    # 검색 키워드
    keywords = [spec_name.lower()]
    if known_aliases:
        keywords.extend([a.lower() for a in known_aliases])
    
    # 핵심 단어
    core_words = re.findall(r"[a-zA-Z]{3,}", spec_name.lower())
    search_terms = keywords + core_words
    
    best_chunk = None
    best_score = 0
    
    for chunk in doc_chunks:
        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        chunk_lower = chunk_text.lower()
        
        # 매칭 점수 계산
        score = 0
        for term in search_terms:
            if term in chunk_lower:
                score += 2 if len(term) > 4 else 1
        
        # 빈값 패턴 확인 (: kW, : %, : m 등)
        empty_patterns = [
            r":\s*[a-zA-Z°℃³/%]+\s*$",  # ": kW" 로 끝남
            r":\s*$",                      # ":" 로 끝남
            r":\s*[-–—]\s*$",              # ": -" 로 끝남
            r"\(\s*\)\s*:",                # "( ) :"
        ]
        for pattern in empty_patterns:
            if re.search(pattern, chunk_text, re.MULTILINE):
                score += 3
        
        if score > best_score:
            best_score = score
            best_chunk = chunk_text
    
    return best_chunk if best_score >= 3 else None


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 컬럼명을 영문으로 통일합니다.
    한글 컬럼명이 있으면 영문으로 매핑합니다.
    """
    rename_map = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in COLUMN_MAPPING_KR_TO_EN:
            rename_map[col] = COLUMN_MAPPING_KR_TO_EN[col_stripped]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# =============================================================================
# HTTP JSON 요청 유틸리티
# =============================================================================

def http_json_request(
    method: str, 
    url: str, 
    payload: Optional[Dict[str, Any]] = None, 
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    JSON 기반 HTTP 요청을 전송하고 응답 JSON을 반환합니다.
    오류 발생 시 예외를 던지지 않고 빈 딕셔너리 또는 에러 정보를 반환합니다.
    """
    data = None
    headers = {'Accept': 'application/json'}
    
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        headers['Content-Type'] = 'application/json; charset=utf-8'
    
    req = urllib.request.Request(
        url=url, 
        data=data, 
        headers=headers, 
        method=method.upper()
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        
        if not raw:
            return {}
        
        try:
            return json.loads(raw.decode('utf-8', errors='replace'))
        except json.JSONDecodeError:
            # JSON이 아닌 응답은 최소 정보로 감쌉니다.
            return {'_raw': raw.decode('utf-8', errors='replace')}
    except Exception as e:
        return {'_error': str(e)}


# =============================================================================
# 설정 클래스
# =============================================================================

@dataclass
class Config:
    """실행 전반 설정값을 보관합니다."""
    # 실행 모드
    mode: str = "file"
    
    # 파일 경로
    base_folder: str = ""
    glossary_path: str = ""
    spec_path: str = ""
    specdb_path: str = ""
    output_path: str = ""
    partial_output_path: str = ""
    enable_checkpoint: bool = True

    # DB 설정
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""
    db_table_glossary: str = "glossary"
    db_table_specdb: str = "spec_db"
    db_table_template: str = "extraction_template"
    db_table_result: str = "extraction_result"

    # LLM 설정
    use_llm: bool = True
    ollama_model: str = "llama3.1:70b"
    ollama_models_dir: str = ""
    ollama_bin: str = ""
    ollama_timeout_sec: int = 300
    ollama_host: str = "127.0.0.1"
    ollama_ports: List[int] = field(default_factory=lambda: [11434])
    llm_temperature: float = 0.0

    # Ollama 자동 실행
    auto_start_ollama_serves: bool = True
    ollama_serve_start_grace_sec: int = 8

    # 병렬 처리
    enable_parallel: bool = True
    num_workers: int = 32

    # 배치 처리
    batch_size: int = 15
    max_evidence_chars: int = 18000

    # LLM 호출 안정화
    rate_limit_min_interval_sec: float = 0.2
    max_retries: int = 3
    retry_base_sleep_sec: float = 2.0

    # 투표
    vote_enabled: bool = True
    vote_k: int = 2
    vote_tiebreak: str = "confidence"

    # 2차 감사
    enable_second_audit: bool = True

    # 벡터 유사도 검색 설정
    enable_vector_search: bool = True
    sentence_transformer_model: str = "/workspace/bge-m3"
    sentence_transformer_device: str = "cuda"

    # 개선 파라미터
    allow_cross_hull_fallback: bool = False
    evidence_max_tables: int = 15
    evidence_include_markdown_tables: bool = True
    evidence_max_md_tables: int = 4
    evidence_max_chunks: int = 10
    rule_enable: bool = True              # Rule 기반 추출 활성화
    rule_conf_threshold: float = 0.72
    force_llm_on_all: bool = False
    audit_conf_low: float = 0.50
    audit_conf_high: float = 0.82
    audit_max_fraction: float = 0.55
    
    # 추가 설정 (POSSpecProcessor 호환용)
    debug: bool = False
    glossary_hybrid_limit: int = 200000
    db_input_conditions: str = ""
    
    # 편의 속성 (별칭)
    @property
    def log_dir(self) -> str:
        """partial_output_path의 별칭"""
        return self.partial_output_path
    
    @property
    def input_path(self) -> str:
        """spec_path의 별칭"""
        return self.spec_path
    
    @property
    def output_dir(self) -> str:
        """output_path의 별칭"""
        return self.output_path
    
    @property
    def html_dirs(self) -> List[str]:
        """base_folder를 리스트로 반환"""
        return [self.base_folder] if self.base_folder else []
    
    @property
    def model_name(self) -> str:
        """ollama_model의 별칭"""
        return self.ollama_model
    
    @property
    def llm_timeout(self) -> int:
        """ollama_timeout_sec의 별칭"""
        return self.ollama_timeout_sec
    
    @property
    def db_input_table(self) -> str:
        """db_table_template의 별칭"""
        return self.db_table_template
    
    @property
    def db_output_table(self) -> str:
        """db_table_result의 별칭"""
        return self.db_table_result


def build_config() -> Config:
    """USER_* 상수를 Config 객체로 변환합니다."""
    return Config(
        mode=USER_MODE,
        
        base_folder=USER_BASE_FOLDER,
        glossary_path=USER_GLOSSARY_PATH,
        spec_path=USER_SPEC_PATH,
        specdb_path=USER_SPECDB_PATH,
        output_path=USER_OUTPUT_PATH,
        partial_output_path=USER_PARTIAL_OUTPUT_PATH,
        enable_checkpoint=bool(USER_ENABLE_CHECKPOINT),

        db_host=USER_DB_HOST,
        db_port=int(USER_DB_PORT),
        db_name=USER_DB_NAME,
        db_user=USER_DB_USER,
        db_password=USER_DB_PASSWORD,
        db_table_glossary=USER_DB_TABLE_GLOSSARY,
        db_table_specdb=USER_DB_TABLE_SPECDB,
        db_table_template=USER_DB_TABLE_TEMPLATE,
        db_table_result=USER_DB_TABLE_RESULT,

        use_llm=bool(USER_USE_LLM),
        ollama_model=USER_OLLAMA_MODEL,
        ollama_models_dir=USER_OLLAMA_MODELS_DIR,
        ollama_bin=USER_OLLAMA_BIN,
        ollama_timeout_sec=int(USER_OLLAMA_TIMEOUT_SEC),
        ollama_host=USER_OLLAMA_HOST,
        ollama_ports=list(USER_OLLAMA_PORTS),
        llm_temperature=float(USER_LLM_TEMPERATURE),

        auto_start_ollama_serves=bool(USER_AUTO_START_OLLAMA_SERVES),
        ollama_serve_start_grace_sec=int(USER_OLLAMA_SERVE_START_GRACE_SEC),

        enable_parallel=bool(USER_ENABLE_PARALLEL),
        num_workers=int(USER_NUM_WORKERS),

        batch_size=int(USER_BATCH_SIZE),
        max_evidence_chars=int(USER_MAX_EVIDENCE_CHARS),

        rate_limit_min_interval_sec=float(USER_LLM_RATE_LIMIT_MIN_INTERVAL_SEC),
        max_retries=int(USER_LLM_MAX_RETRIES),
        retry_base_sleep_sec=float(USER_LLM_RETRY_BASE_SLEEP_SEC),

        vote_enabled=bool(USER_VOTE_ENABLED),
        vote_k=int(USER_VOTE_K),
        vote_tiebreak=str(USER_VOTE_TIEBREAK),

        enable_second_audit=bool(USER_ENABLE_SECOND_AUDIT),

        # 벡터 유사도 검색
        enable_vector_search=bool(USER_ENABLE_VECTOR_SEARCH),
        sentence_transformer_model=USER_SENTENCE_TRANSFORMER_MODEL,
        sentence_transformer_device=USER_SENTENCE_TRANSFORMER_DEVICE,

        allow_cross_hull_fallback=bool(USER_ALLOW_CROSS_HULL_FALLBACK),
        evidence_max_tables=int(USER_EVIDENCE_MAX_TABLES),
        evidence_include_markdown_tables=bool(USER_EVIDENCE_INCLUDE_MARKDOWN_TABLES),
        evidence_max_md_tables=int(USER_EVIDENCE_MAX_MD_TABLES),
        evidence_max_chunks=int(USER_EVIDENCE_MAX_CHUNKS),
        rule_conf_threshold=float(USER_RULE_CONF_THRESHOLD),
        force_llm_on_all=bool(USER_FORCE_LLM_ON_ALL),
        audit_conf_low=float(USER_AUDIT_CONF_LOW),
        audit_conf_high=float(USER_AUDIT_CONF_HIGH),
        audit_max_fraction=float(USER_AUDIT_MAX_FRACTION),
    )


def resolve_ollama_models_dir(user_path: str) -> str:
    """
    Ollama 모델 스토어 경로를 점검해 실제로 모델이 존재하는 디렉터리를 선택합니다.
    """
    cand = []
    if user_path:
        cand.append(user_path)
    # 흔히 쓰는 기본 경로 후보를 추가합니다.
    cand.append(os.path.expanduser('~/.ollama/models'))
    cand.append('/root/.ollama/models')
    cand.append('/workspace/models')

    def _looks_like_store(p: str) -> bool:
        if not p or not os.path.isdir(p):
            return False
        mdir = os.path.join(p, 'manifests')
        bdir = os.path.join(p, 'blobs')
        if not (os.path.isdir(mdir) and os.path.isdir(bdir)):
            return False
        # manifests에 무언가 존재하면 모델이 있다고 판단합니다.
        try:
            for root, dirs, files in os.walk(mdir):
                if files:
                    return True
            return False
        except Exception:
            return False

    for p in cand:
        if _looks_like_store(p):
            return p
    return user_path or os.path.expanduser('~/.ollama/models')


# =============================================================================
# 체크포인트 클래스
# =============================================================================

class Checkpoint:
    """중단/재시작을 위해 처리 완료 키를 저장합니다."""
    
    def __init__(self, path: str, enabled: bool, logger: logging.Logger):
        self.path = path
        self.enabled = enabled
        self.logger = logger
        self.file = os.path.join(self.path, "checkpoint.pkl")

    def load(self) -> Dict[str, Any]:
        """저장된 완료 목록을 복구합니다."""
        if not self.enabled:
            return {}
        try:
            if os.path.exists(self.file):
                with open(self.file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning("체크포인트 로드 실패: %s", e)
        return {}

    def save(self, state: Dict[str, Any]) -> None:
        """완료 목록을 주기적으로 저장합니다."""
        if not self.enabled:
            return
        try:
            os.makedirs(self.path, exist_ok=True)
            with open(self.file, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            self.logger.warning("체크포인트 저장 실패: %s", e)


# =============================================================================
# PostgreSQL 데이터 로더
# =============================================================================

class PostgresLoader:
    """PostgreSQL에서 데이터를 로드하고 결과를 업로드합니다."""
    
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.conn = None
    
    def connect(self) -> bool:
        """DB 연결을 수립합니다."""
        if not HAS_PSYCOPG2:
            self.logger.error("psycopg2 모듈이 설치되어 있지 않습니다.")
            return False
        try:
            self.conn = psycopg2.connect(
                host=self.cfg.db_host,
                port=self.cfg.db_port,
                dbname=self.cfg.db_name,
                user=self.cfg.db_user,
                password=self.cfg.db_password,
            )
            self.logger.info("PostgreSQL 연결 성공: %s:%s/%s", 
                           self.cfg.db_host, self.cfg.db_port, self.cfg.db_name)
            return True
        except Exception as e:
            self.logger.error("PostgreSQL 연결 실패: %s", e)
            return False
    
    def disconnect(self) -> None:
        """DB 연결을 종료합니다."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def load_table(self, table_name: str, where_clause: str = "") -> pd.DataFrame:
        """테이블에서 데이터를 로드합니다."""
        if not self.conn:
            return pd.DataFrame()
        try:
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            df = pd.read_sql_query(query, self.conn)
            self.logger.info("[DB] %s 로드 완료: %d rows", table_name, len(df))
            return df
        except Exception as e:
            self.logger.error("[DB] %s 로드 실패: %s", table_name, e)
            return pd.DataFrame()
    
    def upload_results(self, results: List[Dict[str, Any]]) -> bool:
        """추출 결과를 DB에 업로드합니다."""
        if not self.conn or not results:
            return False
        try:
            # 결과 테이블에 UPSERT
            cursor = self.conn.cursor()
            for row in results:
                # JSON 형태로 저장
                cursor.execute(
                    f"""
                    INSERT INTO {self.cfg.db_table_result} 
                    (matnr, doknr, extwg, umgv_code, result_json, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (matnr, doknr, extwg, umgv_code) 
                    DO UPDATE SET result_json = EXCLUDED.result_json, updated_at = NOW()
                    """,
                    (
                        row.get("matnr", ""),
                        row.get("doknr", ""),
                        row.get("extwg", ""),
                        row.get("umgv_code", ""),
                        json.dumps(row, ensure_ascii=False),
                    )
                )
            self.conn.commit()
            cursor.close()
            self.logger.info("[DB] 결과 %d건 업로드 완료", len(results))
            return True
        except Exception as e:
            self.logger.error("[DB] 결과 업로드 실패: %s", e)
            self.conn.rollback()
            return False


# =============================================================================
# TXT 파일 로더
# =============================================================================

class TxtTableLoader:
    """TSV 파일을 안정적으로 읽습니다."""
    
    def __init__(self, path_or_logger, logger: Optional[logging.Logger] = None):
        """
        초기화 방법:
        1) TxtTableLoader(logger) - 기존 방식, 이후 load_tsv(path) 호출
        2) TxtTableLoader(path) - 경로 지정, 이후 load() 호출
        3) TxtTableLoader(path, logger) - 둘 다 지정
        """
        if isinstance(path_or_logger, logging.Logger):
            # 기존 방식: logger만 받음
            self.path = ""
            self.logger = path_or_logger
        elif isinstance(path_or_logger, str):
            # 새 방식: 경로를 받음
            self.path = path_or_logger
            self.logger = logger or logging.getLogger("TxtTableLoader")
        else:
            self.path = ""
            self.logger = logging.getLogger("TxtTableLoader")

    def load(self) -> pd.DataFrame:
        """저장된 경로에서 TSV를 로드합니다."""
        if not self.path:
            self.logger.error("[load] 경로가 설정되지 않았습니다.")
            return pd.DataFrame()
        return self.load_tsv(self.path)

    def load_tsv(self, path: str) -> pd.DataFrame:
        """일반 TSV를 UTF-8-SIG로 로드합니다."""
        if not os.path.exists(path):
            self.logger.warning("[load_tsv] 파일이 존재하지 않음: %s", path)
            return pd.DataFrame()
        
        size = os.path.getsize(path)
        self.logger.info("[load_tsv] %s size=%s", path, f"{size:,}")
        
        try:
            df = pd.read_csv(
                path, 
                sep="\t", 
                encoding="utf-8-sig", 
                dtype=str, 
                keep_default_na=False,
                on_bad_lines='warn'
            )
            # 컬럼명 정규화
            df = normalize_column_names(df)
            return df
        except Exception as e:
            self.logger.error("[load_tsv] 로드 실패: %s - %s", path, e)
            return pd.DataFrame()

    def load_specdb_repaired(self, path: str, max_lines_scan: int = 0) -> pd.DataFrame:
        """
        비정상 개행/따옴표가 섞인 대용량 TSV를 라인 병합으로 복구하여 로드합니다.
        """
        if not os.path.exists(path):
            self.logger.warning("[load_specdb] 파일이 존재하지 않음: %s", path)
            return pd.DataFrame()
        
        size = os.path.getsize(path)
        self.logger.info("[load_specdb] %s size=%s", path, f"{size:,}")

        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline().rstrip("\n")
        
        if "\t" not in header:
            self.logger.warning("[specdb] header에 탭이 없습니다. specdb를 무시합니다.")
            return pd.DataFrame()

        expected_tabs = header.count("\t")
        rows: List[str] = []
        buf = ""
        biglines = 0

        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            _ = f.readline()  # 헤더 스킵
            for i, line in enumerate(f, start=1):
                if max_lines_scan > 0 and i > max_lines_scan:
                    break
                s = line.rstrip("\n")
                buf = s if not buf else (buf + "\n" + s)

                if buf.count("\t") >= expected_tabs:
                    rows.append(buf)
                    buf = ""

                if len(s) > 2_000_000:
                    biglines += 1

                if i % 200000 == 0:
                    self.logger.info(
                        "[specdb] scanning lines=%s rows=%s biglines=%s", 
                        f"{i:,}", f"{len(rows):,}", biglines
                    )

        if buf:
            rows.append(buf)

        text = header + "\n" + "\n".join(rows)
        df = pd.read_csv(StringIO(text), sep="\t", dtype=str, keep_default_na=False)
        df = normalize_column_names(df)
        
        self.logger.info(
            "[specdb] repair 완료 rows=%s cols=%s biglines=%s", 
            f"{len(df):,}", len(df.columns), biglines
        )
        return df


# =============================================================================
# POS HTML 파일 탐색
# =============================================================================

class FileFinder:
    """hull/POS 조합으로 POS HTML 파일을 찾습니다."""
    
    def __init__(
        self, 
        base_folder: str = "",
        base_dirs: Optional[List[str]] = None,
        allow_cross_hull_fallback: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            base_folder: 단일 기본 폴더 (기존 호환성)
            base_dirs: 다중 기본 폴더 리스트 (새 인터페이스)
            allow_cross_hull_fallback: 다른 호선 파일 허용 여부
            logger: 로거 인스턴스
        """
        # base_dirs가 있으면 우선 사용, 없으면 base_folder 사용
        if base_dirs:
            self.base_dirs = list(base_dirs) if isinstance(base_dirs, (list, tuple)) else [base_dirs]
        elif base_folder:
            self.base_dirs = [base_folder]
        else:
            self.base_dirs = []
        
        # 기존 호환성: self.base 속성 유지
        self.base = self.base_dirs[0] if self.base_dirs else ""
        
        self.allow_cross_hull_fallback = bool(allow_cross_hull_fallback)
        self.logger = logger or logging.getLogger("FileFinder")

    @staticmethod
    def _version_key(fn: str) -> Tuple[int, int, int]:
        """파일명 내 버전/회차 숫자를 뽑아 최신을 우선 정렬합니다."""
        base = os.path.basename(fn)
        m = re.search(r"_(\d{3})_(\d{2})_", base)
        a = int(m.group(1)) if m else 0
        b = int(m.group(2)) if m else 0
        m2 = re.search(r"\((\d+)\)", base)
        c = int(m2.group(1)) if m2 else 0
        return (a, b, c)

    def find_file(self, hull: str, pos: str) -> Optional[str]:
        """
        hull/POS로 HTML 파일을 찾습니다 (새 인터페이스).
        find_latest_html의 별칭입니다.
        """
        return self.find_latest_html(hull, pos)

    def find_latest_html(self, hull: str, pos: str) -> Optional[str]:
        """
        hull/POS로 후보 파일을 모아 최신 버전부터 반환합니다.

        중요 버그 픽스 (v21 개선):
        - 기존 코드는 "*{pos}*.html" fallback 때문에, 동일 POS의 다른 호선 파일을 잘못 잡는 경우가 발생했습니다.
        - 기본값으로 cross-hull fallback은 차단하고, 필요 시 cfg.allow_cross_hull_fallback=True로만 허용합니다.
        """
        hull_digits = re.sub(r"\D", "", str(hull))
        pos_norm = str(pos).strip()
        
        # POS 번호 정규화 (하이픈 제거 등)
        pos_norm_clean = re.sub(r"[^0-9A-Za-z]", "", pos_norm)

        # 모든 base_dirs에서 검색
        cand: List[str] = []
        
        for base in self.base_dirs:
            if not base or not os.path.isdir(base):
                continue
                
            patterns = []
            if hull_digits and pos_norm:
                patterns.extend([
                    os.path.join(base, f"*{hull_digits}*{pos_norm}*.html"),
                    os.path.join(base, f"*{hull_digits}*{pos_norm_clean}*.html"),
                    os.path.join(base, f"*{pos_norm}*{hull_digits}*.html"),
                ])
            elif pos_norm:
                patterns.append(os.path.join(base, f"*{pos_norm}*.html"))
            
            for pat in patterns:
                cand.extend(glob.glob(pat))

        if not cand and not pos_norm:
            return None

        # 중복 제거
        cand = list({c: 1 for c in cand}.keys())

        # hull_digits가 있으면 1차로 해당 digits 포함 파일만 우선
        if hull_digits:
            cand_hull = [c for c in cand if hull_digits in os.path.basename(c)]
            if cand_hull:
                cand = cand_hull
            else:
                # 마지막 안전장치: pos-only fallback (기본은 차단)
                if not self.allow_cross_hull_fallback:
                    self.logger.debug("No file found for hull=%s, pos=%s (cross-hull disabled)", hull, pos)
                    return None

        if not cand:
            self.logger.debug("No file found for hull=%s, pos=%s", hull, pos)
            return None

        cand.sort(key=lambda x: self._version_key(x), reverse=True)
        return cand[0]


# =============================================================================
# POS HTML 파서 (v21에서 누락된 클래스 추가)
# =============================================================================

class POSHTMLReader:
    """
    POS HTML 문서를 파싱하여 텍스트와 테이블을 추출합니다.
    파일 경로 또는 HTML 문자열을 입력으로 받을 수 있습니다.
    """
    
    def __init__(
        self, 
        path_or_html: Optional[str] = None, 
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            path_or_html: 파일 경로 또는 HTML 문자열 (None이면 나중에 설정)
            logger: 로거 객체
        """
        self.logger = logger or logging.getLogger("POSHTMLReader")
        self._html: str = ""
        self._texts: List[str] = []
        self._tables: List[Tuple[str, List[List[str]]]] = []
        self._chunks: List["DocChunk"] = []
        self._parsed = False
        
        if path_or_html:
            if os.path.isfile(path_or_html):
                # 파일 경로인 경우
                self._load_file(path_or_html)
            else:
                # HTML 문자열인 경우
                self._html = path_or_html
    
    def _load_file(self, file_path: str) -> None:
        """HTML 파일 로드"""
        try:
            # 여러 인코딩 시도
            for enc in ["utf-8", "cp949", "euc-kr", "latin-1"]:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        self._html = f.read()
                    self.logger.debug("Loaded %s with encoding %s", file_path, enc)
                    break
                except UnicodeDecodeError:
                    continue
            if not self._html:
                self.logger.warning("Failed to load %s with any encoding", file_path)
        except Exception as e:
            self.logger.error("Error loading file %s: %s", file_path, e)
    
    def _parse(self) -> None:
        """HTML 파싱 (한 번만 실행)"""
        if self._parsed or not self._html:
            return
        
        self._texts, self._tables = self._html_to_texts_and_tables(self._html)
        self._chunks = self._build_chunks()
        self._parsed = True
    
    def _html_to_texts_and_tables(
        self, 
        html: str
    ) -> Tuple[List[str], List[Tuple[str, List[List[str]]]]]:
        """
        HTML에서 텍스트 블록과 테이블을 분리합니다.
        취소선(<s>, <del>, <strike>, style에 line-through) 처리된 내용은 제외합니다.
        """
        soup = BeautifulSoup(html, "html.parser")
        
        # 취소선 태그 제거
        for strike_tag in soup.find_all(['s', 'del', 'strike']):
            strike_tag.decompose()
        
        # style에 line-through가 있는 요소 제거
        for tag in soup.find_all(style=re.compile(r'line-through', re.I)):
            tag.decompose()

        # 텍스트 블록 추출
        texts: List[str] = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div"]):
            # 테이블 내부 요소는 건너뛰기
            if tag.find_parent("table"):
                continue
            t = norm(tag.get_text(" ", strip=True))
            if t and len(t) > 2:
                texts.append(t)

        # 테이블 추출
        tables: List[Tuple[str, List[List[str]]]] = []
        for ti, table in enumerate(soup.find_all("table")):
            rows: List[List[str]] = []
            for tr in table.find_all("tr"):
                cells = []
                for td in tr.find_all(["th", "td"]):
                    # 셀 내 취소선 제거 후 텍스트 추출
                    for strike in td.find_all(['s', 'del', 'strike']):
                        strike.decompose()
                    cell_text = norm(td.get_text(" ", strip=True))
                    cells.append(cell_text)
                if any(cells):
                    rows.append(cells)
            if rows:
                tables.append((f"table_{ti}", rows))
        
        return texts, tables
    
    def _build_chunks(self) -> List["DocChunk"]:
        """텍스트와 테이블을 DocChunk 리스트로 변환"""
        chunks: List["DocChunk"] = []
        
        # 텍스트 청크
        for i, text in enumerate(self._texts):
            if not text.strip():
                continue
            chunk_id = fast_hash(f"text_{i}_{text[:50]}")
            chunks.append(DocChunk(
                ctype="text",
                text=text,
                raw_rows=None,
            ))
        
        # 테이블 청크
        for table_name, rows in self._tables:
            if not rows:
                continue
            # 테이블을 markdown 형식으로도 저장
            md_text = table_to_markdown(rows)
            chunk_id = fast_hash(f"{table_name}_{md_text[:50]}")
            chunks.append(DocChunk(
                ctype="table_md",
                text=md_text,
                raw_rows=rows,
            ))
        
        return chunks
    
    def get_chunks(self) -> List["DocChunk"]:
        """문서 청크 리스트 반환"""
        self._parse()
        return self._chunks
    
    def get_text_blob(self) -> str:
        """전체 텍스트를 하나의 문자열로 반환"""
        self._parse()
        parts = []
        
        # 텍스트 추가
        for text in self._texts:
            if text.strip():
                parts.append(text)
        
        # 테이블 텍스트 추가
        for table_name, rows in self._tables:
            for row in rows:
                row_text = " | ".join(str(cell) for cell in row if cell)
                if row_text.strip():
                    parts.append(row_text)
        
        return "\n".join(parts)
    
    def get_tables(self) -> List[Tuple[str, List[List[str]]]]:
        """테이블 리스트 반환"""
        self._parse()
        return self._tables
    
    def get_texts(self) -> List[str]:
        """텍스트 블록 리스트 반환"""
        self._parse()
        return self._texts
    
    # 하위 호환성을 위한 메서드
    def html_to_texts_and_tables(
        self, 
        html: str
    ) -> Tuple[List[str], List[Tuple[str, List[List[str]]]]]:
        """하위 호환성: HTML 문자열을 직접 받아 파싱"""
        return self._html_to_texts_and_tables(html)


# 모듈 레벨에서도 사용 가능하도록 함수 형태로도 제공
def html_to_texts_and_tables(html: str) -> Tuple[List[str], List[Tuple[str, List[List[str]]]]]:
    """HTML에서 텍스트 블록과 테이블을 분리합니다."""
    reader = POSHTMLReader(logging.getLogger("POSHTMLReader"))
    return reader.html_to_texts_and_tables(html)


# =============================================================================
# 문서 청킹
# =============================================================================

@dataclass
class DocChunk:
    """문서 청크를 텍스트/테이블 단위로 보관합니다."""
    ctype: str                                    # "text", "table_kv", "table_md", "hint", "snippet"
    text: str                                     # 청크 내용
    raw_rows: Optional[List[List[str]]] = None   # 테이블인 경우 원본 행 데이터


def table_to_markdown(rows: List[List[str]], max_body_rows: int = 60) -> str:
    """테이블 구조(헤더/열)를 보존한 markdown 표 문자열을 만듭니다."""
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    header = rows[0] + [""] * (ncol - len(rows[0]))

    def pad(r: List[str]) -> List[str]:
        return r + [""] * (ncol - len(r))

    md: List[str] = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * ncol) + " |")
    for r in rows[1: 1 + max_body_rows]:
        rr = pad(r)
        md.append("| " + " | ".join(rr) + " |")
    return "\n".join(md)


def table_to_header_value_lines(matrix_rows: List[List[str]]) -> Tuple[List[str], List[str]]:
    """
    테이블을 헤더-값 쌍([Header: Value]) 행 문자열로 직렬화하고 헤더 목록도 함께 반환합니다.
    이 형식은 LLM이 테이블 구조를 이해하기 쉽고 토큰도 절약됩니다.
    """
    if not matrix_rows:
        return [], []
    headers = [norm(x) for x in matrix_rows[0]]
    if not headers:
        return [], []
    if all(not h for h in headers):
        headers = [f"Col_{i+1}" for i in range(len(headers))]
    
    out_lines: List[str] = []
    for r in matrix_rows[1:]:
        parts: List[str] = []
        for i, v in enumerate(r):
            h = headers[i] if i < len(headers) else f"Extra_Col_{i+1}"
            vv = norm(v)
            if vv:
                parts.append(f"[{h}: {vv}]")
        if parts:
            out_lines.append(" | ".join(parts))
    return out_lines, headers


def make_doc_chunks(
    texts: List[str],
    tables: List[Tuple[str, List[List[str]]]],
    max_tables: int = 15,
    include_markdown_tables: bool = True,
    max_md_tables: int = 4,
) -> List[DocChunk]:
    """
    LLM에 제공할 핵심 evidence 청크를 구성합니다.

    - TEXT: 문서 본문(헤더/문단/리스트)을 하나의 blob으로 합쳐 상단 맥락 제공
    - TABLE_KV: 테이블을 "[Header: Value]" 라인 형태로 직렬화(LLM 친화 + 토큰 절약)
    - TABLE_MD: 일부 테이블은 markdown 표로도 제공(구조가 중요한 경우 보조)
    """
    chunks: List[DocChunk] = []

    # 텍스트 블록을 하나로 합침
    if texts:
        blob = "\n".join(texts[:1000])  # 상한
        chunks.append(DocChunk(
            kind="text",
            section="doc_text",
            chunk_id=fast_hash(blob)[:12],
            text=blob[:12000],  # 최대 길이 제한
        ))

    # 테이블 처리
    md_added = 0
    for name, rows in tables[:max_tables]:
        # 1) compact KV lines
        table_lines, _headers = table_to_header_value_lines(rows)
        if table_lines:
            body = "\n".join(table_lines)
            chunks.append(DocChunk(
                kind="table_kv",
                section=name,
                chunk_id=fast_hash(body)[:12],
                text=body[:16000],
            ))

        # 2) markdown (optional, limited)
        if include_markdown_tables and md_added < max_md_tables:
            md_table = table_to_markdown(rows, max_body_rows=50)
            if md_table:
                chunks.append(DocChunk(
                    kind="table_md",
                    section=name,
                    chunk_id=fast_hash(md_table)[:12],
                    text=md_table[:16000],
                ))
                md_added += 1

    return chunks


# =============================================================================
# Glossary 하이브리드 매칭 (편집거리 + SimHash 기반 임베딩 유사도)
# =============================================================================

def norm_key(s: Any) -> str:
    """검색 키용 문자열 정규화"""
    s = norm(s)
    s = s.lower()
    
    # 단위 내 공백 정규화 (m 3 /h -> m3/h)
    s = re.sub(r'm\s*3\s*/\s*h', 'm3/h', s, flags=re.I)
    s = re.sub(r'm\s*3', 'm3', s, flags=re.I)
    s = re.sub(r'k\s*w', 'kw', s, flags=re.I)
    s = re.sub(r'k\s*g', 'kg', s, flags=re.I)
    s = re.sub(r'm\s*/\s*s', 'm/s', s, flags=re.I)
    s = re.sub(r'm\s*/\s*min', 'm/min', s, flags=re.I)
    
    s = re.sub(r"[^0-9a-z가-힣\s/_\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_tokens(s: str) -> List[str]:
    """문자열을 토큰 리스트로 분할"""
    s = norm_key(s)
    if not s:
        return []
    return [t for t in re.split(r"[\s/_\-]+", s) if t]


def simhash64(tokens: List[str]) -> int:
    """
    초경량 '임베딩 대체' 시그니처.
    - 완전한 의미 임베딩은 아니지만, 대규모 후보 축소 + 유사도 정렬에 큰 도움이 됩니다.
    """
    if not tokens:
        return 0
    v = [0] * 64
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest(), 16)
        for i in range(64):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i, w in enumerate(v):
        if w >= 0:
            out |= (1 << i)
    return out


def hamming64(a: int, b: int) -> int:
    """64비트 해밍 거리"""
    return (a ^ b).bit_count()


def edit_ratio(a: str, b: str) -> float:
    """difflib 기반 편집거리 비율 (Levenshtein과 100% 동일하지는 않지만, 빠르고 의존성이 없습니다)"""
    import difflib
    a2, b2 = norm_key(a), norm_key(b)
    if not a2 or not b2:
        return 0.0
    return difflib.SequenceMatcher(None, a2, b2).ratio()


@dataclass
class GlossaryMatch:
    """Glossary 매칭 결과"""
    score: float
    row: Dict[str, Any]


class GlossaryIndex:
    """
    대규모 사양항목명(예: 30만)에서도 돌아가도록,
    후보를 토큰 인덱스로 줄인 뒤
    (편집거리 + SimHash) 혼합 점수로 최종 매칭합니다.
    """
    
    def __init__(self, glossary_df: pd.DataFrame):
        self.rows: List[Dict[str, Any]] = []
        self.by_code: Dict[str, Dict[str, Any]] = {}
        self.by_desc_exact: Dict[str, Dict[str, Any]] = {}
        self.token_to_ids: Dict[str, List[int]] = {}
        self._build(glossary_df)

    def _choose_better(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """pos_chunk / pos_umgv_desc / evidence_fb 가 더 풍부한 쪽을 선호"""
        def richness(x: Dict[str, Any]) -> int:
            score = 0
            for k in ["pos_chunk", "pos_umgv_desc", "table_text", "evidence_fb", 
                      "value_format", "umgv_uom", "pos_umgv_uom", "section_num"]:
                if norm(x.get(k, "")):
                    score += 1
            return score
        return a if richness(a) >= richness(b) else b

    def _build(self, df: pd.DataFrame) -> None:
        """인덱스 구축"""
        if df is None or df.empty:
            return
        df = df.fillna("")
        
        for _, r in df.iterrows():
            d = {k: r.get(k, "") for k in df.columns}
            idx = len(self.rows)
            self.rows.append(d)

            # code 기반 인덱스
            code = norm_key(d.get("umgv_code", ""))
            if code:
                prev = self.by_code.get(code)
                self.by_code[code] = d if prev is None else self._choose_better(prev, d)

            # desc 기반 인덱스
            desc = norm_key(d.get("umgv_desc", ""))
            if desc:
                prev = self.by_desc_exact.get(desc)
                self.by_desc_exact[desc] = d if prev is None else self._choose_better(prev, d)

            # 후보 축소용 토큰 인덱스: umgv_desc + pos_umgv_desc + pos_chunk
            txt = " ".join([
                str(d.get("umgv_desc", "")),
                str(d.get("pos_umgv_desc", "")),
                str(d.get("pos_chunk", "")),
            ])
            for t in set(simple_tokens(txt)):
                self.token_to_ids.setdefault(t, []).append(idx)

        # token list 중복 제거
        for t, ids in list(self.token_to_ids.items()):
            if len(ids) > 1:
                self.token_to_ids[t] = list(sorted(set(ids)))

    def match(
        self, 
        spec_name: str, 
        umgv_code: str = "", 
        umgv_desc: str = "", 
        top_k: int = 3
    ) -> List[GlossaryMatch]:
        """사양항목명에 대해 가장 유사한 Glossary 항목을 찾습니다."""
        
        # 1) code exact
        code = norm_key(umgv_code)
        if code and code in self.by_code:
            return [GlossaryMatch(score=1.0, row=self.by_code[code])]

        # 2) desc exact
        desc_key = norm_key(umgv_desc or spec_name)
        if desc_key and desc_key in self.by_desc_exact:
            return [GlossaryMatch(score=0.98, row=self.by_desc_exact[desc_key])]

        # 3) hybrid: token 후보 축소 -> (edit + simhash) 스코어
        q = umgv_desc or spec_name
        q_tokens = simple_tokens(q)
        if not q_tokens:
            return []

        cand_ids: List[int] = []
        for t in set(q_tokens[:8]):  # 상위 몇 토큰만
            cand_ids.extend(self.token_to_ids.get(t, []))
        if not cand_ids:
            return []

        cand_ids = list(sorted(set(cand_ids)))
        q_sig = simhash64(q_tokens)

        scored: List[GlossaryMatch] = []
        for cid in cand_ids[:5000]:  # 상한 (안전장치)
            row = self.rows[cid]
            cand_txt = " ".join([
                str(row.get("umgv_desc", "")), 
                str(row.get("pos_umgv_desc", "")), 
                str(row.get("pos_chunk", ""))
            ])
            r_edit = edit_ratio(q, cand_txt)

            c_sig = simhash64(simple_tokens(cand_txt))
            ham = hamming64(q_sig, c_sig)
            r_sim = max(0.0, 1.0 - (ham / 64.0))  # 0~1

            # 혼합 점수 (편집거리 비중↑, simhash는 동률 깨기)
            score = (0.70 * r_edit) + (0.30 * r_sim)
            if score >= 0.50:
                scored.append(GlossaryMatch(score=score, row=row))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def lookup(
        self, 
        umgv_desc: str = "",
        umgv_code: str = "",
        top_k: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        사양항목명/코드로 Glossary에서 가장 유사한 항목을 찾아 딕셔너리로 반환합니다.
        
        Args:
            umgv_desc: 사양항목명
            umgv_code: 사양항목코드
            top_k: 반환할 최대 결과 수 (기본 1)
        
        Returns:
            가장 유사한 항목의 딕셔너리 또는 None
        """
        # match 메서드 활용
        matches = self.match(
            spec_name=umgv_desc,
            umgv_code=umgv_code,
            umgv_desc=umgv_desc,
            top_k=top_k,
        )
        
        if matches and len(matches) > 0:
            return matches[0].row
        
        return None
    
    def lookup_with_context(
        self, 
        umgv_desc: str,
        umgv_code: str = "",
        hull: str = "",
        umg_code: str = "",
        pmg_code: str = "",
        top_k: int = 5,
    ) -> List[GlossaryMatch]:
        """
        맥락 정보(호선, UMG, PMG)를 활용한 유사 레코드 검색.
        같은 UMG/PMG의 다른 호선 레코드에서 힌트를 찾습니다.
        """
        # 기본 검색
        matches = self.lookup(umgv_desc=umgv_desc, umgv_code=umgv_code, top_k=top_k * 3)
        
        if not matches:
            return []
        
        # 맥락 기반 점수 조정
        scored = []
        for m in matches:
            bonus = 0.0
            
            # 같은 UMG면 보너스
            row_umg = str(m.row.get("umg_code", ""))
            if umg_code and row_umg == umg_code:
                bonus += 0.1
            
            # 같은 PMG면 보너스
            row_pmg = str(m.row.get("pmg_code", ""))
            if pmg_code and row_pmg == pmg_code:
                bonus += 0.05
            
            # pos_umgv_value가 있으면 보너스 (추출 경험이 있는 레코드)
            if m.row.get("pos_umgv_value"):
                bonus += 0.15
            
            # section_num, table_text 정보가 있으면 보너스
            if m.row.get("section_num"):
                bonus += 0.05
            if m.row.get("table_text") == "Y":
                bonus += 0.05
            
            adjusted_score = min(m.score + bonus, 1.0)
            scored.append(GlossaryMatch(score=adjusted_score, row=m.row))
        
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


# =============================================================================
# 벡터 기반 유사 POS 검색 시스템 (Sentence Transformer)
# =============================================================================

@dataclass
class SimilarPOSMatch:
    """유사 POS 매칭 결과"""
    score: float                    # 유사도 점수 (0~1)
    hull: str                       # 호선번호
    pos: str                        # POS 번호
    pmg_desc: str                   # PMG 설명
    umg_desc: str                   # UMG 설명
    umgv_desc: str                  # UMGV 설명
    umgv_uom: str                   # 표준 단위
    pos_umgv_value: str             # 추출된 값
    pos_umgv_uom: str               # POS 문서 단위
    pos_umgv_desc: str              # POS 문서 설명
    section_num: str                # 섹션 번호
    table_text: str                 # 테이블 여부
    pos_chunk: str                  # 추출 청크
    row_data: Dict[str, Any]        # 원본 데이터


class VectorSimilaritySearch:
    """
    Sentence Transformer를 활용한 벡터 기반 유사 POS 검색.
    
    사용 시나리오:
    1. 현재 추출 대상의 (hull, pmg, umg, umgv, uom) 정보를 쿼리로 생성
    2. 사양값DB/용어집에서 벡터 유사도가 높은 과거 POS 레코드 검색
    3. 검색된 레코드의 (section_num, table_text, pos_umgv_uom 등)을 힌트로 활용
    """
    
    def __init__(
        self,
        model_path: str = "/workspace/bge-m3",
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger("VectorSearch")
        self.model_path = model_path
        self.device = device
        self.model = None
        self.embeddings = None
        self.rows: List[Dict[str, Any]] = []
        self._is_initialized = False
    
    def _load_model(self) -> bool:
        """Sentence Transformer 모델 로드"""
        if not HAS_SENTENCE_TRANSFORMER:
            self.logger.warning("sentence_transformers not installed. Vector search disabled.")
            return False
        
        if self.model is not None:
            return True
        
        try:
            self.logger.info("Loading Sentence Transformer model: %s", self.model_path)
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device,
            )
            self.logger.info("Model loaded successfully. Device: %s", self.device)
            return True
        except Exception as e:
            self.logger.error("Failed to load Sentence Transformer: %s", e)
            return False
    
    def build_index(
        self,
        df: pd.DataFrame,
        text_columns: List[str] = None,
        batch_size: int = 32,
    ) -> bool:
        """
        DataFrame에서 벡터 인덱스 구축.
        
        Args:
            df: 사양값DB 또는 용어집 DataFrame
            text_columns: 임베딩할 컬럼들 (기본: pmg_desc, umg_desc, umgv_desc)
            batch_size: 배치 크기
        """
        if not self._load_model():
            return False
        
        if df is None or df.empty:
            self.logger.warning("Empty DataFrame. Cannot build index.")
            return False
        
        # 기본 텍스트 컬럼 (v41: mat_attr_desc 추가)
        if text_columns is None:
            text_columns = ["pmg_desc", "umg_desc", "umgv_desc", "umgv_uom", "mat_attr_desc"]
        
        df = df.fillna("")
        self.rows = []
        texts = []
        
        for _, row in df.iterrows():
            # 행 데이터 저장
            row_dict = row.to_dict()
            self.rows.append(row_dict)
            
            # 임베딩용 텍스트 생성 (컬럼들을 연결)
            text_parts = []
            
            # v41: matnr 앞 4자리 (호선번호) 추가
            matnr = str(row.get("matnr", ""))
            if matnr and len(matnr) >= 4:
                hull = matnr[:4]
                text_parts.append(f"Hull:{hull}")
            
            for col in text_columns:
                val = str(row.get(col, "")).strip()
                if val and val != "nan":
                    text_parts.append(val)
            
            # 추가 컨텍스트 (있으면)
            for extra_col in ["pos_umgv_desc", "pos_chunk"]:
                val = str(row.get(extra_col, "")).strip()
                if val and val != "nan" and len(val) < 500:
                    text_parts.append(val[:200])
            
            text = " | ".join(text_parts) if text_parts else "unknown"
            texts.append(text)
        
        # 임베딩 생성
        self.logger.info("Encoding %d texts...", len(texts))
        try:
            self.embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,  # cosine similarity용
            )
            self._is_initialized = True
            self.logger.info("Index built: %d vectors, shape=%s", 
                           len(self.embeddings), self.embeddings.shape)
            return True
        except Exception as e:
            self.logger.error("Failed to encode texts: %s", e)
            return False
    
    def search(
        self,
        query_hull: str = "",
        query_pmg: str = "",
        query_umg: str = "",
        query_umgv: str = "",
        query_uom: str = "",
        query_mat_attr_desc: str = "",  # v41: 자재속성그룹명 추가
        top_k: int = 5,
        exclude_same_hull: bool = True,
    ) -> List[SimilarPOSMatch]:
        """
        쿼리와 유사한 과거 POS 레코드 검색.
        
        Args:
            query_hull: 현재 호선번호 (유사 검색 대상에서 제외)
            query_pmg: PMG 설명
            query_umg: UMG 설명
            query_umgv: UMGV 설명
            query_uom: 표준 단위
            query_mat_attr_desc: v41 - 자재속성그룹명 (mat_attr_desc)
            top_k: 반환할 최대 결과 수
            exclude_same_hull: 같은 호선 제외 여부
        
        Returns:
            유사도 높은 순으로 정렬된 SimilarPOSMatch 리스트
        """
        if not self._is_initialized or self.model is None:
            self.logger.warning("Vector search not initialized.")
            return []
        
        # 쿼리 텍스트 생성 (v41: hull과 mat_attr_desc 추가)
        query_parts = []
        if query_hull:
            query_parts.append(f"Hull:{query_hull}")
        if query_pmg:
            query_parts.append(query_pmg)
        if query_umg:
            query_parts.append(query_umg)
        if query_mat_attr_desc:
            query_parts.append(query_mat_attr_desc)
        if query_umgv:
            query_parts.append(query_umgv)
        if query_uom:
            query_parts.append(query_uom)
        
        if not query_parts:
            return []
        
        query_text = " | ".join(query_parts)
        
        # 쿼리 임베딩
        try:
            query_embedding = self.model.encode(
                [query_text],
                normalize_embeddings=True,
            )[0]
        except Exception as e:
            self.logger.error("Failed to encode query: %s", e)
            return []
        
        # 코사인 유사도 계산 (정규화된 벡터이므로 내적 = 코사인 유사도)
        if HAS_NUMPY:
            similarities = np.dot(self.embeddings, query_embedding)
        else:
            # numpy 없으면 수동 계산
            similarities = []
            for emb in self.embeddings:
                sim = sum(a * b for a, b in zip(emb, query_embedding))
                similarities.append(sim)
            similarities = similarities
        
        # 상위 결과 추출
        if HAS_NUMPY:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            row = self.rows[idx]
            row_hull = str(row.get("hull", "") or row.get("matnr", "")[:4] if row.get("matnr") else "")
            
            # 같은 호선 제외
            if exclude_same_hull and row_hull and row_hull == query_hull:
                continue
            
            # pos_umgv_value가 있는 레코드만 (추출 경험이 있는)
            if not row.get("pos_umgv_value"):
                continue
            
            score = float(similarities[idx]) if HAS_NUMPY else similarities[idx]
            
            match = SimilarPOSMatch(
                score=score,
                hull=row_hull,
                pos=str(row.get("doknr", "") or row.get("pos", ""))[:10],
                pmg_desc=str(row.get("pmg_desc", "")),
                umg_desc=str(row.get("umg_desc", "")),
                umgv_desc=str(row.get("umgv_desc", "")),
                umgv_uom=str(row.get("umgv_uom", "")),
                pos_umgv_value=str(row.get("pos_umgv_value", "")),
                pos_umgv_uom=str(row.get("pos_umgv_uom", "")),
                pos_umgv_desc=str(row.get("pos_umgv_desc", "")),
                section_num=str(row.get("section_num", "")),
                table_text=str(row.get("table_text", "")),
                pos_chunk=str(row.get("pos_chunk", ""))[:500],
                row_data=row,
            )
            results.append(match)
        
        return results
    
    def get_few_shot_hints(
        self,
        query_hull: str,
        query_pmg: str,
        query_umg: str,
        query_umgv: str,
        query_uom: str,
        query_mat_attr_desc: str = "",  # v41: 자재속성그룹명 추가
        top_k: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Few-shot 프롬프트용 힌트 생성.
        
        Returns:
            [{"umgv_desc": ..., "value": ..., "unit": ..., "section": ..., "table": ...}, ...]
        """
        matches = self.search(
            query_hull=query_hull,
            query_pmg=query_pmg,
            query_umg=query_umg,
            query_umgv=query_umgv,
            query_uom=query_uom,
            query_mat_attr_desc=query_mat_attr_desc,
            top_k=top_k,
            exclude_same_hull=True,
        )
        
        hints = []
        for m in matches:
            if m.pos_umgv_value:
                hints.append({
                    "hull": m.hull,
                    "umgv_desc": m.umgv_desc,
                    "value": m.pos_umgv_value,
                    "unit": m.pos_umgv_uom or m.umgv_uom,
                    "section": m.section_num,
                    "table": m.table_text,
                    "score": f"{m.score:.3f}",
                })
        
        return hints


# 글로벌 벡터 검색 인스턴스 (싱글톤)
_vector_search_instance: Optional[VectorSimilaritySearch] = None


def get_vector_search(
    model_path: str = None,
    device: str = None,
    logger: logging.Logger = None,
) -> Optional[VectorSimilaritySearch]:
    """VectorSimilaritySearch 싱글톤 인스턴스 반환"""
    global _vector_search_instance
    
    if not HAS_SENTENCE_TRANSFORMER:
        return None
    
    if _vector_search_instance is None:
        _vector_search_instance = VectorSimilaritySearch(
            model_path=model_path or USER_SENTENCE_TRANSFORMER_MODEL,
            device=device or USER_SENTENCE_TRANSFORMER_DEVICE,
            logger=logger,
        )
    
    return _vector_search_instance


# =============================================================================
# Locator Hint (레퍼런스 POS 위치 힌트)
# =============================================================================

@dataclass
class LocatorHint:
    """레퍼런스 POS에서 값/단위가 나타난 위치를 요약합니다."""
    ref_hull: str
    ref_pos: str
    ref_file: str
    found_in: str           # "table" or "text"
    table_index: Optional[int]
    row_index: Optional[int]
    col_index: Optional[int]
    anchor_text: str

    def to_prompt_hint(self) -> str:
        """힌트를 프롬프트 상단에 삽입할 한 줄 문자열로 만듭니다."""
        parts = [
            f"ref_hull={self.ref_hull}",
            f"ref_pos={self.ref_pos}",
            f"ref_file={os.path.basename(self.ref_file)}",
            f"found_in={self.found_in}",
        ]
        if self.table_index is not None:
            parts.append(f"table_index={self.table_index}")
        if self.row_index is not None and self.col_index is not None:
            parts.append(f"cell=({self.row_index},{self.col_index})")
        if self.anchor_text:
            parts.append(f"anchor='{self.anchor_text[:150]}'")
        return "LOCATOR_HINT: " + ", ".join(parts)


class EvidenceLocator:
    """POS HTML에서 레퍼런스 value/unit의 위치를 빠르게 찾습니다."""
    
    def __init__(
        self, 
        finder: Optional[FileFinder] = None, 
        reader: Optional[POSHTMLReader] = None, 
        glossary_index: Optional[GlossaryIndex] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.finder = finder
        self.reader = reader
        self.glossary_index = glossary_index
        self.logger = logger or logging.getLogger("EvidenceLocator")
        self._cache: Dict[str, Optional[LocatorHint]] = {}

    def select_evidence(
        self,
        spec_name: str,
        label_variants: List[str],
        doc_chunks: List[DocChunk],
        doc_blob: str,
        hints: Dict[str, Any],
        max_chars: int = 18000,
    ) -> str:
        """
        Evidence 텍스트만 반환합니다 (하위 호환성).
        """
        evidence, _ = self.select_evidence_with_meta(
            spec_name=spec_name,
            label_variants=label_variants,
            doc_chunks=doc_chunks,
            doc_blob=doc_blob,
            hints=hints,
            max_chars=max_chars,
        )
        return evidence

    def select_evidence_with_meta(
        self,
        spec_name: str,
        label_variants: List[str],
        doc_chunks: List[DocChunk],
        doc_blob: str,
        hints: Dict[str, Any],
        max_chars: int = 18000,
        equipment_number: Optional[int] = None,
        mat_attr_desc: str = "",  # 자재속성그룹명 (핵심 식별자)
        umg_desc: str = "",    # UMG 설명
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Evidence 텍스트와 메타정보(section_num, table_text 등)를 함께 반환합니다.
        
        개선사항:
        1. mat_attr_desc(자재속성그룹명)를 최우선 키워드로 사용
        2. 장비 번호 인식 (WASHING MACHINE 1, 2, 3...)
        3. umgv_desc가 chunk에 포함되도록 우선순위 조정
        4. 테이블 구조 개선 (LLM 이해도 향상)
        5. "1. GENERAL" 같은 보일러플레이트 청크 회피
        
        Returns:
            (evidence_text, {
                "section_num": "3.2",
                "table_text": "Y" or "N",
                "strategy": "FOCUSED" or "GLOBAL" or "LOCATOR",
                "matched_chunk_count": int,
                "keyword_found": bool,  # 핵심 키워드가 chunk에 있는지
            })
        """
        meta = {
            "section_num": "",
            "table_text": "N",
            "strategy": "",
            "matched_chunk_count": 0,
            "keyword_found": False,
        }
        
        if not doc_chunks and not doc_blob:
            return "", meta
        
        # 장비 번호 파싱 (WASHING MACHINE 1 -> base, 1)
        base_spec_name, equip_num = parse_equipment_number(spec_name)
        if equipment_number is not None:
            equip_num = equipment_number
        
        # ========================================
        # 핵심 키워드 추출 (이 키워드들이 chunk에 반드시 있어야 함)
        # ========================================
        core_keywords = []
        
        # 1. spec_name에서 핵심 단어 추출 (괄호/단위 제거)
        clean_spec = re.sub(r'\([^)]*\)', '', spec_name).strip()  # 괄호 내용 제거
        clean_spec = re.sub(r'\s+\d+$', '', clean_spec).strip()   # 끝의 숫자 제거
        
        # 핵심 단어들 추출
        spec_words = re.findall(r'[A-Za-z]+', clean_spec)
        for w in spec_words:
            if len(w) >= 3 and w.upper() not in ['THE', 'AND', 'FOR', 'WITH']:
                core_keywords.append(w.lower())
        
        # 2. mat_attr_desc(자재속성그룹명)에서 핵심 키워드 추출 - 최우선!
        # 예: "ME FO SUPPLY MODULE" → ["me", "fo", "supply", "module"]
        extwg_keywords = []
        if mat_attr_desc:
            extwg_words = re.findall(r'[A-Za-z]+', mat_attr_desc)
            for w in extwg_words:
                w_lower = w.lower()
                if len(w) >= 2 and w_lower not in ['for', 'the', 'and', 'with', 'of', 'in']:
                    extwg_keywords.append(w_lower)
                    if w_lower not in core_keywords:
                        core_keywords.insert(0, w_lower)  # 최우선순위로 추가
        
        # 3. umg_desc에서 키워드 추출
        if umg_desc:
            umg_words = re.findall(r'[A-Za-z]+', umg_desc)
            for w in umg_words:
                w_lower = w.lower()
                if len(w) >= 3 and w_lower not in core_keywords:
                    core_keywords.append(w_lower)
        
        # 4. 언더스코어로 분리된 부분 추가
        if "_" in spec_name:
            parts = spec_name.split("_")
            for p in parts:
                p_clean = re.sub(r'\([^)]*\)', '', p).strip()
                p_clean = re.sub(r'\s+\d+$', '', p_clean).strip()
                words = re.findall(r'[A-Za-z]+', p_clean)
                for w in words:
                    if len(w) >= 3 and w.lower() not in core_keywords:
                        core_keywords.append(w.lower())
        
        # label_variants에서도 키워드 수집
        all_labels = list(label_variants)
        if base_spec_name and base_spec_name not in all_labels:
            all_labels.append(base_spec_name)
        
        for label in all_labels:
            if "_" in label:
                parts = label.split("_")
                for p in parts:
                    if p and len(p) > 2 and p not in all_labels:
                        all_labels.append(p)
        
        self.logger.debug("Core keywords for '%s': %s, extwg_keywords: %s", 
                         spec_name, core_keywords[:10], extwg_keywords)
        
        # ========================================
        # 보일러플레이트 청크 감지 함수
        # ========================================
        def is_boilerplate_chunk(chunk_text: str) -> bool:
            """
            "1. GENERAL" 같은 일반적인 문서 시작 부분인지 감지.
            이런 청크는 실제 스펙 값이 없을 가능성이 높음.
            """
            text_lower = chunk_text[:500].lower()
            boilerplate_markers = [
                "1. general",
                "drawings and instruction manuals",
                "after the official submission",
                "cyber resilience",
                "asbestos-free declaration",
                "general specifications",
            ]
            marker_count = sum(1 for m in boilerplate_markers if m in text_lower)
            return marker_count >= 2
        
        # ========================================
        # chunk 품질 평가 함수 (개선됨)
        # ========================================
        def score_chunk_relevance(chunk_text: str) -> Tuple[int, bool]:
            """
            chunk가 spec_name/mat_attr_desc와 얼마나 관련있는지 평가.
            Returns: (score, has_core_keyword)
            """
            text_lower = chunk_text.lower()
            score = 0
            has_core = False
            
            # 보일러플레이트 감지 → 점수 대폭 하락
            if is_boilerplate_chunk(chunk_text):
                return -100, False
            
            # ========================================
            # 장비 유형 식별 (ME/GE/AB 구분) - 최우선!
            # ========================================
            # mat_attr_desc에서 장비 유형 추출
            extwg_upper = mat_attr_desc.upper() if mat_attr_desc else ""
            is_me_equipment = "ME " in extwg_upper or "M/E" in extwg_upper or "MAIN ENGINE" in extwg_upper
            is_ge_equipment = "GE " in extwg_upper or "G/E" in extwg_upper or "GENERATOR" in extwg_upper or "DIESEL" in extwg_upper
            is_ab_equipment = "A/B" in extwg_upper or "BOILER" in extwg_upper or "AUX BOILER" in extwg_upper
            
            # 청크 내 장비 유형 확인
            has_me_content = "M/E FO" in text_lower or "M/E LSMGO" in text_lower or "ME FO" in text_lower or "m/e fo" in text_lower
            has_ge_content = "G/E FO" in text_lower or "G/E LSMGO" in text_lower or "GE FO" in text_lower or "g/e fo" in text_lower
            has_ab_content = "A/B FO" in text_lower or "a/b fo" in text_lower
            
            # 장비 유형 매칭 점수 (매우 높음)
            equipment_matched = False
            if is_me_equipment and has_me_content and not has_ge_content:
                score += 100  # 정확한 ME 매칭
                equipment_matched = True
                has_core = True
            elif is_ge_equipment and has_ge_content and not has_me_content:
                score += 100  # 정확한 GE 매칭
                equipment_matched = True
                has_core = True
            elif is_ab_equipment and has_ab_content:
                score += 100  # 정확한 A/B 매칭
                equipment_matched = True
                has_core = True
            
            # 부분 매칭 (낮은 점수)
            if not equipment_matched:
                if is_me_equipment and has_me_content:
                    score += 50
                    has_core = True
                if is_ge_equipment and has_ge_content:
                    score += 50
                    has_core = True
                if is_ab_equipment and has_ab_content:
                    score += 50
                    has_core = True
            
            # 혼합 테이블 페널티 (ME와 GE 모두 포함)
            # → 어느 장비인지 명확하지 않으므로 점수 감소
            if has_me_content and has_ge_content:
                if is_me_equipment or is_ge_equipment:
                    score -= 30
            
            # ========================================
            # 기존 키워드 매칭 로직
            # ========================================
            # 1. mat_attr_desc 완전 매칭
            if mat_attr_desc and mat_attr_desc.lower() in text_lower:
                score += 30
                has_core = True
            
            # 2. extwg_keywords 모두 존재
            if extwg_keywords:
                extwg_match_count = sum(1 for kw in extwg_keywords if kw in text_lower)
                if extwg_match_count == len(extwg_keywords):
                    score += 20  # 모든 키워드 매칭
                    has_core = True
                elif extwg_match_count >= len(extwg_keywords) * 0.7:
                    score += 15  # 70% 이상 매칭
                    has_core = True
            
            # 3. 핵심 키워드 존재 여부
            core_matches = 0
            for kw in core_keywords[:8]:  # 상위 8개 핵심 키워드
                if kw in text_lower:
                    core_matches += 1
                    score += 3
            
            if core_matches >= 2:
                has_core = True
            
            # 4. 정확한 spec_name 매칭 (보너스)
            if spec_name.lower() in text_lower:
                score += 10
                has_core = True
            
            # 5. base_spec_name 매칭 (보너스)
            if base_spec_name and base_spec_name.lower() in text_lower:
                score += 7
                has_core = True
            
            # 6. 일반 label 매칭
            for label in all_labels[:5]:
                if label.lower() in text_lower:
                    score += 2
            
            return score, has_core
        
        # ========================================
        # 전략 1: 테이블 청크에서 키워드 매칭
        # ========================================
        table_evidence = []
        for chunk in doc_chunks:
            if chunk.ctype in ("table_md", "table_kv") and chunk.text:
                score, has_core = score_chunk_relevance(chunk.text)
                
                # 장비 번호 보너스
                if equip_num:
                    if re.search(rf'\|\s*{equip_num}\s*\|', chunk.text):
                        score += 5
                    elif re.search(rf'(?:No\.?|#)\s*{equip_num}', chunk.text, re.I):
                        score += 3
                
                # 핵심 키워드가 있거나 높은 점수인 경우만 후보로
                if has_core or score >= 5:
                    table_evidence.append((score, has_core, chunk))
        
        if table_evidence:
            # 핵심 키워드가 있는 것 우선, 그 다음 점수 순
            table_evidence.sort(key=lambda x: (x[1], x[0]), reverse=True)
            best_score, best_has_core, best_chunk = table_evidence[0]
            
            meta["table_text"] = "Y"
            meta["strategy"] = "TABLE_MATCH"
            meta["matched_chunk_count"] = len(table_evidence)
            meta["keyword_found"] = best_has_core
            
            # 섹션 번호 추출 시도 (청크 텍스트에서)
            section_match = re.search(r'(\d+(?:\.\d+)*)\s*[\.:\-]', best_chunk.text[:100])
            if section_match:
                meta["section_num"] = section_match.group(1)
            
            # 테이블 구조 개선 (LLM 이해도 향상)
            evidence = enhance_table_for_llm(best_chunk.text[:max_chars])
            
            return evidence, meta
        
        # ========================================
        # 전략 2: 텍스트 청크에서 키워드 매칭 (개선)
        # ========================================
        text_chunks = [c for c in doc_chunks if c.ctype == "text"]
        
        # 핵심 키워드가 포함된 청크만 선택
        focused_chunks = []
        for chunk in text_chunks:
            score, has_core = score_chunk_relevance(chunk.text)
            
            # 핵심 키워드가 있는 경우만 후보로 (중요!)
            if has_core:
                focused_chunks.append((score, True, chunk))
            elif score >= 5:
                # 핵심 키워드는 없지만 점수가 높은 경우 낮은 우선순위로 추가
                focused_chunks.append((score, False, chunk))
        
        if focused_chunks:
            # 핵심 키워드 있는 것 우선, 그 다음 점수 순
            focused_chunks.sort(key=lambda x: (x[1], x[0]), reverse=True)
            
            # 핵심 키워드가 있는 청크만 선택 (최대 3개)
            core_chunks = [c for c in focused_chunks if c[1]]
            if core_chunks:
                selected = core_chunks[:3]
                meta["keyword_found"] = True
            else:
                # 핵심 키워드가 없으면 점수 높은 것 선택
                selected = focused_chunks[:3]
                meta["keyword_found"] = False
            
            combined_text = "\n\n---\n\n".join([c[2].text for c in selected])
            
            evidence = combined_text[:max_chars]
            meta["table_text"] = "N"
            meta["strategy"] = "FOCUSED_MULTI"
            meta["matched_chunk_count"] = len(focused_chunks)
            
            # 섹션 번호 추출 시도
            section_match = re.search(r'(\d+(?:\.\d+)*)\s*[\.:\-]', evidence[:100])
            if section_match:
                meta["section_num"] = section_match.group(1)
            
            return evidence, meta
        
        # ========================================
        # 전략 3: 전체 텍스트에서 키워드 주변 윈도우
        # ========================================
        combined_text = " ".join([c.text for c in text_chunks])
        
        # 핵심 키워드로 먼저 검색
        for kw in core_keywords[:3]:
            windows = _windows_around(combined_text, kw, window=600, max_windows=2)
            if windows:
                evidence = "\n".join(windows)[:max_chars]
                meta["table_text"] = "N"
                meta["strategy"] = "WINDOW_CORE"
                meta["keyword_found"] = True
                
                section_match = re.search(r'(\d+(?:\.\d+)*)\s*[\.:\-]', evidence[:100])
                if section_match:
                    meta["section_num"] = section_match.group(1)
                
                return evidence, meta
        
        # 일반 라벨로 검색
        for label in all_labels[:5]:
            windows = _windows_around(combined_text, label, window=500, max_windows=2)
            if windows:
                evidence = "\n".join(windows)[:max_chars]
                meta["table_text"] = "N"
                meta["strategy"] = "WINDOW"
                
                section_match = re.search(r'(\d+(?:\.\d+)*)\s*[\.:\-]', evidence[:100])
                if section_match:
                    meta["section_num"] = section_match.group(1)
                
                return evidence, meta
        
        # ========================================
        # 전략 4: 관련 chunk 없음 - NO_RELEVANT_CHUNK
        # ========================================
        # 핵심 키워드가 전혀 없는 경우, 빈 evidence 반환하지 않고
        # 경고와 함께 blob의 일부를 반환 (하지만 keyword_found=False로 표시)
        if doc_blob:
            meta["table_text"] = "N"
            meta["strategy"] = "GLOBAL_FALLBACK"
            meta["keyword_found"] = False
            self.logger.warning(
                "No relevant chunk found for '%s'. Using global fallback. Core keywords: %s",
                spec_name, core_keywords[:5]
            )
            return doc_blob[:max_chars], meta
        
        return "", meta

    def locate(
        self, 
        ref_hull: str, 
        ref_pos: str, 
        spec_name: str, 
        value: str, 
        unit: str
    ) -> Optional[LocatorHint]:
        """value/unit이 등장한 테이블 셀이나 텍스트 스니펫을 찾아 힌트로 반환합니다."""
        if not self.finder or not self.reader:
            return None
            
        key = f"{ref_hull}||{ref_pos}||{spec_name}||{value}||{unit}"
        if key in self._cache:
            return self._cache[key]

        html_path = self.finder.find_latest_html(ref_hull, ref_pos)
        if not html_path or not os.path.exists(html_path):
            self._cache[key] = None
            return None

        try:
            html = open(html_path, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            self._cache[key] = None
            return None

        texts, tables = self.reader.html_to_texts_and_tables(html)
        v = norm(value)
        u = norm(unit)
        s = norm(spec_name)
        tokens = [t for t in [v, u, s] if t]

        # 테이블에서 우선 검색
        for ti, (_, rows) in enumerate(tables):
            for ri, r in enumerate(rows[:800]):
                for ci, c in enumerate(r[:80]):
                    cell = norm(c)
                    if not cell:
                        continue
                    hit = 0
                    if v and v in cell:
                        hit += 1
                    if u and u in cell:
                        hit += 1
                    if s and s.lower() in cell.lower():
                        hit += 1
                    if hit >= 2 or (hit >= 1 and v and v in cell):
                        hint = LocatorHint(
                            ref_hull=str(ref_hull),
                            ref_pos=str(ref_pos),
                            ref_file=str(html_path),
                            found_in="table",
                            table_index=ti,
                            row_index=ri,
                            col_index=ci,
                            anchor_text=cell,
                        )
                        self._cache[key] = hint
                        return hint

        # 텍스트에서 검색
        joined = "\n".join(texts)
        if v and v in joined:
            idx = joined.find(v)
            anchor = joined[max(0, idx - 150): idx + 150]
            hint = LocatorHint(
                ref_hull=str(ref_hull),
                ref_pos=str(ref_pos),
                ref_file=str(html_path),
                found_in="text",
                table_index=None,
                row_index=None,
                col_index=None,
                anchor_text=anchor,
            )
            self._cache[key] = hint
            return hint

        self._cache[key] = None
        return None


# =============================================================================
# Evidence 선택 (다중 전략 + 내부 voting)
# =============================================================================

def _count_hits(text: str, needles: List[str]) -> int:
    """텍스트에서 키워드 등장 횟수를 카운트합니다."""
    t = text.lower()
    hits = 0
    for n in needles:
        nn = norm_key(n)
        if not nn or len(nn) < 3:
            continue
        hits += t.count(nn)
    return hits


def parse_equipment_number(spec_name: str) -> Tuple[str, Optional[int]]:
    """
    사양항목명에서 장비 번호를 추출합니다.
    
    예시:
        "CAPACITY_WASHING MACHINE 1 (kg)" -> ("CAPACITY_WASHING MACHINE", 1)
        "CAPACITY_DRYING TUMBLER 2 (kg)" -> ("CAPACITY_DRYING TUMBLER", 2)
        "CAPACITY (ton)" -> ("CAPACITY", None)
    
    Returns:
        (base_name, equipment_number)
    """
    if not spec_name:
        return spec_name, None
    
    # 패턴 1: "NAME 1 (unit)" 형태
    m = re.match(r'^(.+?)\s+(\d+)\s*\([^)]*\)$', spec_name)
    if m:
        return m.group(1).strip(), int(m.group(2))
    
    # 패턴 2: "NAME 1" 형태 (단위 없음)
    m = re.match(r'^(.+?)\s+(\d+)$', spec_name)
    if m:
        base = m.group(1).strip()
        num = int(m.group(2))
        # 숫자가 너무 크면 장비 번호가 아닐 수 있음
        if num <= 10:
            return base, num
    
    # 패턴 3: "NAME_SUBNAME 1" 형태 (언더스코어 구분)
    m = re.match(r'^(.+?)_(.+?)\s+(\d+)\s*(?:\([^)]*\))?$', spec_name)
    if m:
        prefix = m.group(1).strip()
        suffix = m.group(2).strip()
        num = int(m.group(3))
        if num <= 10:
            return f"{prefix}_{suffix}", num
    
    return spec_name, None


def find_equipment_context_in_table(
    table_text: str,
    base_name: str,
    equipment_number: int,
    expected_unit: str = "",
) -> Optional[Tuple[str, str, str]]:
    """
    테이블에서 특정 장비 번호에 해당하는 값을 찾습니다.
    
    복잡한 테이블에서 장비 번호(1, 2, 3...)로 구분된 여러 장비 중
    올바른 행을 식별합니다.
    
    Args:
        table_text: 테이블 텍스트 (markdown 또는 plain)
        base_name: 기본 사양명 (예: "CAPACITY_WASHING MACHINE")
        equipment_number: 장비 번호 (예: 1, 2, 3)
        expected_unit: 기대 단위
    
    Returns:
        (value, unit, context) 또는 None
    """
    if not table_text or not base_name:
        return None
    
    # base_name에서 키워드 추출
    keywords = []
    if "_" in base_name:
        parts = base_name.split("_")
        keywords.extend(parts)
    keywords.append(base_name.replace("_", " "))
    
    # 테이블을 행별로 분리
    lines = table_text.split("\n")
    
    # 번호가 포함된 행 찾기 (NO. 열 또는 번호 패턴)
    equipment_rows = []
    current_equipment_num = 0
    
    for i, line in enumerate(lines):
        # Markdown 구분선 스킵
        if re.match(r'^\s*\|?\s*[-:]+\s*\|', line):
            continue
        
        # 행에서 장비 번호 탐지
        # 패턴: | 1 | 또는 | NO. 1 | 또는 첫 열이 숫자
        cells = re.split(r'\s*\|\s*', line)
        cells = [c.strip() for c in cells if c.strip()]
        
        if cells:
            first_cell = cells[0]
            # 첫 번째 셀이 숫자인 경우 (테이블 NO. 열)
            if re.match(r'^\d+$', first_cell):
                current_equipment_num = int(first_cell)
            elif first_cell.upper() == "NO.":
                continue  # 헤더 행
        
        # 키워드가 포함된 행인지 확인
        line_lower = line.lower()
        has_keyword = any(kw.lower() in line_lower for kw in keywords)
        
        if has_keyword:
            equipment_rows.append({
                "line_num": i,
                "line": line,
                "equipment_num": current_equipment_num,
                "cells": cells,
            })
    
    # 장비 번호에 맞는 행 찾기
    target_row = None
    for row in equipment_rows:
        if row["equipment_num"] == equipment_number:
            target_row = row
            break
    
    # 번호로 못 찾으면 순서로 추정
    if target_row is None and equipment_rows:
        if equipment_number <= len(equipment_rows):
            target_row = equipment_rows[equipment_number - 1]
    
    if target_row is None:
        return None
    
    # 값 추출
    line = target_row["line"]
    
    # 숫자 + 단위 패턴 찾기
    value_patterns = [
        rf'([\d,\.]+)\s*({re.escape(expected_unit)})' if expected_unit else None,
        r'(?:Abt\.?\s*)?([\d,\.]+)\s*(kg|ltr|l|L|m|mm|kW|ton|t)',
        r'(?:Capacity|Cap\.?)\s*[:：]?\s*(?:Abt\.?\s*)?([\d,\.]+)\s*(\w+)',
    ]
    
    for pat in value_patterns:
        if pat is None:
            continue
        m = re.search(pat, line, re.IGNORECASE)
        if m:
            return (m.group(1), m.group(2), line[:200])
    
    return None


def enhance_table_for_llm(table_text: str) -> str:
    """
    복잡한 테이블을 LLM이 더 잘 이해하도록 변환합니다.
    
    개선사항:
    1. 병합 셀 표시 추가
    2. 행/열 번호 명시
    3. 구조 설명 추가
    """
    if not table_text:
        return table_text
    
    lines = table_text.strip().split("\n")
    if len(lines) < 2:
        return table_text
    
    enhanced_lines = []
    row_num = 0
    
    for line in lines:
        # Markdown 구분선 처리
        if re.match(r'^\s*\|?\s*[-:]+(\s*\|\s*[-:]+)*\s*\|?\s*$', line):
            enhanced_lines.append(line)
            continue
        
        # 셀 분리
        if '|' in line:
            cells = line.split('|')
            # 앞뒤 빈 셀 제거
            if cells and not cells[0].strip():
                cells = cells[1:]
            if cells and not cells[-1].strip():
                cells = cells[:-1]
            
            # 행 번호 추가 (헤더 제외)
            if row_num > 0:
                # 각 셀에 열 번호 힌트 추가 (비어있지 않은 경우)
                annotated_cells = []
                for col_num, cell in enumerate(cells):
                    cell_text = cell.strip()
                    if cell_text:
                        annotated_cells.append(f" {cell_text} ")
                    else:
                        annotated_cells.append(cell)
                
                enhanced_line = f"Row{row_num}: |" + "|".join(annotated_cells) + "|"
                enhanced_lines.append(enhanced_line)
            else:
                enhanced_lines.append(line)
            
            row_num += 1
        else:
            enhanced_lines.append(line)
    
    # 테이블 구조 설명 추가
    result = "### Table Structure (Row numbers added for reference)\n"
    result += "\n".join(enhanced_lines)
    
    return result


def _windows_around(text: str, needle: str, window: int = 250, max_windows: int = 3) -> List[str]:
    """텍스트에서 키워드 주변 윈도우를 추출합니다."""
    out: List[str] = []
    if not text or not needle:
        return out
    n = norm_key(needle)
    if not n or len(n) < 3:
        return out
    tl = norm_key(text)
    if n not in tl:
        return out
    
    # 보수적으로: split 후 일부 주변만
    parts = text.split(needle)
    if len(parts) <= 1:
        return out
    
    # 가장 앞쪽 등장 중심으로만 사용
    prefix = parts[0]
    idx = len(prefix)
    out.append(text[max(0, idx - window): idx + window])
    return out[:max_windows]


def build_label_variants(row: Dict[str, Any]) -> List[str]:
    """추출 대상 row에서 가능한 label 후보 목록을 생성합니다."""
    v: List[str] = []
    
    # 주요 label 소스
    for k in [COL_SPEC_NAME, "umgv_desc", "__gl_pos_umgv_desc", "__gl_pos_chunk", 
              "__gl_table_text", "mat_attr_desc", COL_MAT_GROUP]:
        vv = norm(row.get(k, ""))
        if vv and vv not in v:
            v.append(vv)
    
    # feedback이 있으면 추가
    fb = norm(row.get("__gl_evidence_fb", ""))
    if fb and fb not in v:
        v.append(fb)
    
    # 너무 긴 chunk 문구는 label로 쓰기엔 무거우니 앞부분만
    out = []
    for x in v:
        if len(x) > 150:
            out.append(x[:150])
        else:
            out.append(x)
    return out


def select_evidence_for_batch(
    doc_text_blob: str,
    chunks: List[DocChunk],
    locator_hint: Optional[LocatorHint],
    label_variants: List[str],
    max_chars: int,
    max_chunks: int = 10,
) -> Tuple[str, str]:
    """
    다중 전략으로 evidence 후보를 만들고, 내부 점수로 best 1개를 선택합니다.
    반환: (evidence_text, strategy_name)
    """
    labels = [x for x in label_variants if x and len(x) >= 3]
    
    # Strategy A: global top chunks (label 히트 수 기반)
    scored = []
    for ch in chunks:
        hits = _count_hits(ch.text, labels)
        # 텍스트 청크에 약간의 보너스
        bonus = 2 if ch.ctype == "text" else 0
        scored.append((hits + bonus, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks_a = [c for s, c in scored[:max_chunks] if s > 0] or [c for s, c in scored[:max(4, max_chunks//2)]]

    # Strategy B: focused windows (본문에서 label 주변만)
    win_parts: List[str] = []
    for lb in labels[:15]:
        win_parts.extend(_windows_around(doc_text_blob, lb))
    win_parts = [norm(x) for x in win_parts if norm(x)]
    win_text = "\n...\n".join(win_parts)[:8000] if win_parts else ""

    # Strategy C: locator hint 중심
    loc_chunks: List[DocChunk] = []
    if locator_hint and locator_hint.found_in == "table" and locator_hint.table_index is not None:
        want_section = f"table_{int(locator_hint.table_index)}"
        for ch in chunks:
            if ch.ctype == want_section:
                loc_chunks.append(ch)
    if locator_hint and locator_hint.anchor_text:
        anchor = locator_hint.anchor_text
        for lb in labels[:10]:
            if lb and lb in anchor:
                win_parts.insert(0, anchor)
                break

    def build(parts_chunks: List[DocChunk], extra_top: List[DocChunk], strategy: str) -> Tuple[str, str, int]:
        use_chunks = []
        if locator_hint:
            use_chunks.append(DocChunk(
                ctype="hint", 
                text=locator_hint.to_prompt_hint(),
                raw_rows=None,
            ))
        use_chunks.extend(parts_chunks)
        use_chunks.extend(extra_top)

        # 중복 제거(같은 텍스트 해시)
        seen = set()
        final_chunks: List[DocChunk] = []
        for c in use_chunks:
            chunk_hash = fast_hash(c.text[:100]) if c.text else ""
            if chunk_hash in seen:
                continue
            seen.add(chunk_hash)
            final_chunks.append(c)

        # evidence 텍스트 구성
        parts: List[str] = []
        for ch in final_chunks:
            ctype_label = ch.ctype.upper() if ch.ctype else "TEXT"
            parts.append(f"--- [{ctype_label}] ---")
            parts.append(ch.text)
            parts.append("")
        ev = "\n".join(parts)
        ev = ev[:max_chars] if len(ev) > max_chars else ev
        
        # 스코어 계산
        score = _count_hits(ev, labels) + (5 if locator_hint else 0) + (4 if strategy == "FOCUSED" and win_text else 0)
        return ev, strategy, score

    ev_a, st_a, sc_a = build([], top_chunks_a, "GLOBAL")
    ev_b, st_b, sc_b = build(
        [DocChunk(ctype="snippet", text=win_text, raw_rows=None)] if win_text else [], 
        top_chunks_a[:4], 
        "FOCUSED"
    )
    ev_c, st_c, sc_c = build(loc_chunks, top_chunks_a[:4], "LOCATOR")

    best = max([(sc_a, ev_a, st_a), (sc_b, ev_b, st_b), (sc_c, ev_c, st_c)], key=lambda x: x[0])
    return best[1], best[2]


# =============================================================================
# Rule 기반 값/단위 추출 (LLM 호출 비율을 줄이기 위한 핵심)
# =============================================================================

# 단위 정규 표현
# 복합 단위를 먼저 정의 (긴 것부터 매칭되도록)
_UNIT_COMPLEX = [
    # 유량 단위
    "m3/h", "m³/h", "㎥/h", "Nm3/h", "Nm³/h", "l/min", "L/min", "l/h", "L/h",
    "m3/min", "m³/min", "l/s", "L/s",
    # 압력 단위  
    "kg/cm2", "kg/cm²", "kgf/cm2", "kgf/cm²", 
    # 속도/회전 단위
    "m/min", "m/s", "km/h", "r/min", "rpm",
    # 전력/토크 단위
    "kW", "MW", "kVA", "VA", "kN·m", "kNm", "N·m", "Nm",
    # 기타 복합 단위
    "mLC", "mTH", "mWC", "mAq",
]

_UNIT_SIMPLE = [
    "mm", "cm", "m", "kg", "g", "t", "kn", "kN", "W", "V", "A", "Hz",
    "bar", "MPa", "kPa", "Pa", "deg", "°", "C", "℃", "F", "℉",
    "N", "L", "l", "m3", "㎥", "m²", "㎡", "m2",
    "sec", "s", "min", "h", "hr", "day", "days", 
    "EA", "ea", "pcs", "pc", "set", "sets", "%", "ppm",
    "persons", "person", "ton", "tons",
]

# 복합 단위를 먼저, 그 다음 단순 단위 (길이 역순 정렬)
_UNIT_CANON = _UNIT_COMPLEX + _UNIT_SIMPLE
_UNIT_RE = re.compile(
    r"(" + "|".join(re.escape(u) for u in sorted(set(_UNIT_CANON), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)


# =============================================================================
# 단위 동의어 관리 시스템 (외부 YAML 파일 기반)
# =============================================================================

class UnitSynonymManager:
    """
    단위 동의어를 외부 YAML 파일에서 로드하고 관리합니다.
    
    기능:
    1. YAML 파일에서 동의어 로드
    2. 런타임 중 새 동의어 추가
    3. 사용자 피드백 후 파일 저장
    4. 표준 단위 <-> 동의어 매핑
    
    YAML 구조:
    ```yaml
    ton:
      synonyms: ["ton", "t", "톤"]
      compatible: ["kN"]
    ```
    """
    
    DEFAULT_YAML_PATH = os.path.join(os.path.dirname(__file__), "unit_synonyms.yaml")
    
    def __init__(self, yaml_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.yaml_path = yaml_path or self.DEFAULT_YAML_PATH
        self.logger = logger or logging.getLogger("UnitSynonymManager")
        
        # 표준단위 -> {synonyms: set, compatible: set}
        self.unit_data: Dict[str, Dict[str, set]] = {}
        
        # 역방향 매핑: 동의어 -> 표준단위
        self.synonym_to_standard: Dict[str, str] = {}
        
        # 새로 발견된 동의어 (저장 대기)
        self.pending_synonyms: Dict[str, set] = {}
        
        # 로드
        self._load_from_yaml()
    
    def _load_from_yaml(self):
        """YAML 파일에서 동의어 로드"""
        if not os.path.exists(self.yaml_path):
            self.logger.warning("Unit synonyms file not found: %s. Using defaults.", self.yaml_path)
            self._load_defaults()
            return
        
        try:
            import yaml
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            for std_unit, info in data.items():
                if not isinstance(info, dict):
                    continue
                
                std_key = std_unit.lower().strip()
                self.unit_data[std_key] = {
                    "synonyms": set(s.lower() for s in info.get("synonyms", [])),
                    "compatible": set(c.lower() for c in info.get("compatible", [])),
                }
                
                # 역방향 매핑 생성
                for syn in self.unit_data[std_key]["synonyms"]:
                    self.synonym_to_standard[syn] = std_key
            
            self.logger.info("Loaded %d unit definitions from %s", len(self.unit_data), self.yaml_path)
            
        except Exception as e:
            self.logger.warning("Failed to load unit synonyms: %s. Using defaults.", e)
            self._load_defaults()
    
    def _load_defaults(self):
        """기본 동의어 (YAML 로드 실패 시)"""
        defaults = {
            "ton": ({"ton", "tons", "t", "톤", "mt"}, {"kn"}),
            "kn": ({"kn", "kilonewton"}, {"ton"}),
            "m3/h": ({"m3/h", "m³/h", "㎥/h", "m 3 /h", "m/3"}, {"nm3/h"}),
            "nm3/h": ({"nm3/h", "nm³/h"}, {"m3/h"}),
            "m": ({"m", "meter", "meters", "미터"}, set()),
            "mm": ({"mm", "millimeter", "밀리미터"}, set()),
            "cm": ({"cm", "centimeter"}, set()),
            "kw": ({"kw", "kilowatt"}, set()),
            "rpm": ({"rpm", "r/min", "rev/min"}, set()),
            "mlc": ({"mlc", "mth", "mwc", "maq"}, set()),
            "bar": ({"bar", "bars"}, set()),
            "kg/cm2": ({"kg/cm2", "kg/cm²", "kgf/cm2"}, set()),
            "v": ({"v", "volt", "volts"}, set()),
            "hz": ({"hz", "hertz"}, set()),
            "set": ({"set", "sets", "세트"}, set()),
            "ea": ({"ea", "pcs", "pc", "개"}, set()),
        }
        
        for std_unit, (synonyms, compatible) in defaults.items():
            self.unit_data[std_unit] = {
                "synonyms": synonyms,
                "compatible": compatible,
            }
            for syn in synonyms:
                self.synonym_to_standard[syn] = std_unit
    
    def get_standard_unit(self, unit: str) -> str:
        """단위를 표준 형식으로 변환"""
        if not unit:
            return ""
        u = unit.lower().strip()
        u = re.sub(r'\s+', '', u)
        return self.synonym_to_standard.get(u, u)
    
    def are_synonyms(self, unit1: str, unit2: str) -> bool:
        """두 단위가 동의어인지 확인"""
        if not unit1 or not unit2:
            return True
        
        std1 = self.get_standard_unit(unit1)
        std2 = self.get_standard_unit(unit2)
        
        return std1 == std2
    
    def are_compatible(self, unit1: str, unit2: str) -> bool:
        """두 단위가 호환되는지 확인 (동의어 또는 호환 관계)"""
        if self.are_synonyms(unit1, unit2):
            return True
        
        std1 = self.get_standard_unit(unit1)
        std2 = self.get_standard_unit(unit2)
        
        # 호환 관계 확인
        if std1 in self.unit_data:
            if std2 in self.unit_data[std1].get("compatible", set()):
                return True
        if std2 in self.unit_data:
            if std1 in self.unit_data[std2].get("compatible", set()):
                return True
        
        return False
    
    def add_synonym(self, standard_unit: str, new_synonym: str):
        """
        새 동의어 추가 (pending 상태).
        save_pending()을 호출해야 파일에 저장됨.
        """
        std_key = standard_unit.lower().strip()
        syn = new_synonym.lower().strip()
        
        if not std_key or not syn:
            return
        
        # 이미 등록된 동의어인지 확인
        if syn in self.synonym_to_standard:
            return
        
        # pending에 추가
        if std_key not in self.pending_synonyms:
            self.pending_synonyms[std_key] = set()
        self.pending_synonyms[std_key].add(syn)
        
        # 바로 사용할 수 있도록 메모리에도 추가
        if std_key in self.unit_data:
            self.unit_data[std_key]["synonyms"].add(syn)
        else:
            self.unit_data[std_key] = {"synonyms": {syn, std_key}, "compatible": set()}
        self.synonym_to_standard[syn] = std_key
        
        self.logger.info("New synonym pending: '%s' -> '%s'", syn, std_key)
    
    def discover_from_extraction(self, expected_unit: str, extracted_unit: str):
        """
        추출 결과에서 새 동의어 발견.
        expected_unit이 표준 단위이고, extracted_unit이 새로운 동의어일 때 호출.
        """
        if not expected_unit or not extracted_unit:
            return
        
        std_key = expected_unit.lower().strip()
        ext = extracted_unit.lower().strip()
        
        # 이미 알려진 동의어면 무시
        if ext in self.synonym_to_standard:
            return
        
        # 표준 단위가 등록되어 있으면 새 동의어로 추가
        if std_key in self.unit_data or std_key in self.synonym_to_standard:
            actual_std = self.get_standard_unit(std_key)
            self.add_synonym(actual_std, ext)
    
    def save_pending(self) -> bool:
        """
        pending 동의어를 YAML 파일에 저장.
        사용자 피드백 완료 후 호출.
        """
        if not self.pending_synonyms:
            return True
        
        try:
            import yaml
            
            # 기존 파일 로드
            if os.path.exists(self.yaml_path):
                with open(self.yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}
            
            # pending 동의어 추가
            for std_key, new_syns in self.pending_synonyms.items():
                if std_key not in data:
                    data[std_key] = {"synonyms": [std_key]}
                
                existing = set(data[std_key].get("synonyms", []))
                existing.update(new_syns)
                data[std_key]["synonyms"] = sorted(existing)
            
            # 저장
            with open(self.yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=True)
            
            self.logger.info("Saved %d new synonyms to %s", 
                           sum(len(s) for s in self.pending_synonyms.values()),
                           self.yaml_path)
            
            self.pending_synonyms.clear()
            return True
            
        except Exception as e:
            self.logger.error("Failed to save synonyms: %s", e)
            return False
    
    def get_pending_count(self) -> int:
        """저장 대기 중인 동의어 수"""
        return sum(len(s) for s in self.pending_synonyms.values())


# 글로벌 인스턴스 (싱글톤)
_unit_manager: Optional[UnitSynonymManager] = None


def get_unit_manager(yaml_path: Optional[str] = None) -> UnitSynonymManager:
    """UnitSynonymManager 싱글톤 인스턴스 반환"""
    global _unit_manager
    if _unit_manager is None:
        _unit_manager = UnitSynonymManager(yaml_path=yaml_path)
    return _unit_manager


def get_standard_unit(unit: str) -> str:
    """단위를 표준 형식으로 변환"""
    return get_unit_manager().get_standard_unit(unit)


def are_units_compatible(unit1: str, unit2: str) -> bool:
    """두 단위가 호환되는지 확인 (동의어 관계)"""
    return get_unit_manager().are_compatible(unit1, unit2)


def _normalize_unit_spaces(text: str) -> str:
    """
    단위 표기에서 공백을 정규화합니다.
    예: "m 3 /h" -> "m3/h", "Nm 3 /h" -> "Nm3/h"
    """
    # m 3 /h -> m3/h
    text = re.sub(r'\bm\s*3\s*/\s*h\b', 'm3/h', text, flags=re.I)
    text = re.sub(r'\bNm\s*3\s*/\s*h\b', 'Nm3/h', text, flags=re.I)
    text = re.sub(r'\bl\s*/\s*min\b', 'l/min', text, flags=re.I)
    text = re.sub(r'\bm\s*/\s*min\b', 'm/min', text, flags=re.I)
    text = re.sub(r'\bm\s*/\s*s\b', 'm/s', text, flags=re.I)
    text = re.sub(r'\bkg\s*/\s*cm\s*2\b', 'kg/cm2', text, flags=re.I)
    text = re.sub(r'\br\s*/\s*min\b', 'r/min', text, flags=re.I)
    text = re.sub(r'\bm\s*3\b', 'm3', text, flags=re.I)
    text = re.sub(r'\bm\s*2\b', 'm2', text, flags=re.I)
    return text


def split_value_unit(text: str, expected_unit: str = "") -> Tuple[str, str]:
    """
    텍스트에서 값과 단위를 분리합니다.
    
    개선사항:
    - 복합 단위 (m3/h, Nm3/h 등) 우선 인식
    - 공백 포함 단위 정규화 (m 3 /h -> m3/h)
    - 숫자로 시작하는 패턴만 값으로 인정
    - "Approx.", "약" 등 접두사 처리
    - expected_unit 유사 매칭 지원
    - 괄호 안의 값 추출 (예: "SWL ( 4.0 ) t")
    """
    if not text:
        return "", ""
    
    t = norm(text)
    if not t:
        return "", ""
    
    # 먼저 숫자가 있는지 확인
    if not re.search(r'\d', t):
        return "", ""
    
    # 단위 공백 정규화
    t = _normalize_unit_spaces(t)
    
    # 패턴 1: 괄호 안의 숫자 + 괄호 밖의 단위 (예: "SWL ( 4.0 ) t", "Min. ( 17 ) m")
    paren_pattern = re.compile(
        r'(?:SWL|Min\.?|Max\.?|Approx\.?)?\s*\(\s*([\d,\.~\-]+)\s*\)\s*([a-zA-Z]+(?:/[a-zA-Z]+)?\.?)',
        flags=re.IGNORECASE
    )
    m_paren = paren_pattern.search(t)
    if m_paren:
        val = m_paren.group(1).strip()
        unit = m_paren.group(2).strip().rstrip('.')
        return val, unit
    
    # expected unit 처리
    eu = norm(expected_unit)
    if eu:
        eu = _normalize_unit_spaces(eu)
        eu_variants = _get_unit_variants(eu)
        
        for eu_var in eu_variants:
            # 정확한 매칭 시도
            pattern = re.compile(
                r'([\d,\.]+(?:\s*[xX×]\s*[\d,\.]+)?)\s*' + re.escape(eu_var),
                flags=re.IGNORECASE
            )
            m = pattern.search(t)
            if m:
                val = m.group(1).strip()
                # Approx. 등 접두사가 앞에 있으면 제거
                prefix_match = re.match(r'^(approx\.?|약|ca\.?|~)\s*', t[:m.start()], re.I)
                if prefix_match:
                    pass  # 접두사는 이미 값에서 제외됨
                return val, eu_var
    
    # 숫자+단위 패턴 직접 찾기
    # 패턴: [접두사] + 숫자 (선택적 소수점, 콤마) + 공백 + 단위
    value_unit_pattern = re.compile(
        r'(?:approx\.?\s*|약\s*|ca\.?\s*|~\s*)?'  # 선택적 접두사
        r'([\d,\.]+(?:\s*[xX×]\s*[\d,\.]+)?)\s*'  # 값 (123 또는 123 x 456)
        r'(' + '|'.join(re.escape(u) for u in sorted(_UNIT_CANON, key=len, reverse=True)) + r')',
        flags=re.IGNORECASE
    )
    
    m = value_unit_pattern.search(t)
    if m:
        val = m.group(1).strip()
        unit = m.group(2).strip()
        return val, unit
    
    # 일반 유닛 분리 (fallback)
    m2 = _UNIT_RE.search(t)
    if m2:
        u = m2.group(1)
        val = norm(t[:m2.start()])
        # Approx. 등 접두사 제거
        val = re.sub(r'^(approx\.?|약|ca\.?|~)\s*', '', val, flags=re.I).strip()
        # 값에 숫자가 있고 합리적인 길이인 경우만 반환
        if val and re.search(r'\d', val) and len(val) < 30:
            return val, u
    
    # 숫자만 있는 경우
    num_match = re.match(r'^(?:approx\.?\s*|약\s*)?([\d,\.]+)', t, re.I)
    if num_match:
        return num_match.group(1), ""
    
    return "", ""


def _get_unit_variants(unit: str) -> List[str]:
    """단위의 다양한 표기 변형을 반환합니다."""
    if not unit:
        return []
    
    variants = [unit]
    u = unit.lower().strip()
    
    # 공백 변형
    if ' ' not in u:
        # m3/h -> m 3 /h, m 3/h 등
        spaced = re.sub(r'(\d)', r' \1 ', u).strip()
        spaced = re.sub(r'\s+', ' ', spaced)
        if spaced != u:
            variants.append(spaced)
    
    # 대소문자 변형
    variants.append(u.upper())
    variants.append(u.lower())
    
    # 특수 문자 변형
    if '3' in u:
        variants.append(u.replace('3', '³'))
    if '2' in u:
        variants.append(u.replace('2', '²'))
    
    # Nm3/h <-> m3/h 관계
    if u.startswith('nm'):
        variants.append(u[1:])  # Nm3/h -> m3/h
    elif u.startswith('m') and not u.startswith('mm'):
        variants.append('n' + u)  # m3/h -> Nm3/h
    
    return list(set(variants))


# =============================================================================
# 일반 라벨 목록 (다중 매칭 시 추가 검증 필요)
# =============================================================================

GENERIC_LABELS = {
    "type", "material", "capacity", "size", "weight", "power", "voltage",
    "quantity", "number", "no", "model", "rating", "speed", "pressure",
    "temperature", "dimension", "length", "width", "height", "diameter",
    "head", "flow", "sets", "units", "ea", "pcs",
}


def _is_generic_label(label: str) -> bool:
    """
    일반적인 라벨인지 확인 (여러 곳에 등장할 가능성 높음)
    
    - 단위가 포함된 라벨은 specific으로 간주 (예: "CAPACITY (m3/h)")
    - 접두사/접미사가 있는 라벨은 specific으로 간주 (예: "NO. OF SET_135L FOAM")
    """
    lk = norm_key(label)
    if not lk:
        return False
    
    # 단위가 포함되어 있으면 specific
    unit_patterns = [
        r'm3/h', r'kw', r'ton', r'kg', r'mm', r'm/s', r'm/min',
        r'bar', r'mpa', r'kpa', r'rpm', r'hz', r'l/min',
        r'npsh', r'℃', r'°c',
    ]
    lk_lower = lk.lower()
    for pat in unit_patterns:
        if pat in lk_lower:
            return False
    
    # 괄호 안에 단위가 있으면 specific (예: "HEIGHT (m)", "SPEED (m/s)")
    if re.search(r'\([^)]*[a-z]+[^)]*\)', label, re.I):
        return False
    
    # 복합 라벨 (언더스코어, 하이픈 포함)은 specific
    if "_" in label or " - " in label or " / " in label:
        return False
    
    # 단어 단위로 체크
    words = lk.split()
    for w in words:
        if w in GENERIC_LABELS:
            return True
    
    return len(lk) <= 8  # 짧은 라벨도 일반적으로 취급


def _normalize_unit_for_match(unit: str) -> str:
    """
    단위를 비교용으로 정규화.
    동의어 시스템을 활용하여 표준 단위로 변환.
    """
    if not unit:
        return ""
    u = unit.lower().strip()
    # 공백 제거
    u = re.sub(r'\s+', '', u)
    # 슬래시 정규화
    u_no_slash = u.replace('/', '')
    # 특수문자 제거
    u_no_slash = re.sub(r'[³²]', lambda m: '3' if m.group() == '³' else '2', u_no_slash)
    
    # 표준 단위로 변환 시도
    std = get_standard_unit(u)
    if std != u:
        return std
    
    return u_no_slash


def units_match(expected: str, extracted: str) -> bool:
    """
    예상 단위와 추출 단위가 매칭되는지 확인.
    동의어 관계까지 고려.
    """
    if not expected or not extracted:
        return True  # 단위가 없으면 매칭으로 간주
    
    # 정규화
    exp_norm = _normalize_unit_for_match(expected)
    ext_norm = _normalize_unit_for_match(extracted)
    
    # 직접 매칭
    if exp_norm == ext_norm:
        return True
    
    # 포함 매칭
    if exp_norm in ext_norm or ext_norm in exp_norm:
        return True
    
    # 동의어 매칭
    if are_units_compatible(expected, extracted):
        return True
    
    return False


def _score_extraction_candidate(
    value: str,
    unit: str,
    expected_unit: str,
    label_cell: str,
    context_cells: List[str],
    is_right: bool,
) -> float:
    """
    추출 후보의 품질 점수를 계산합니다.
    
    점수 요소:
    - 단위 일치: +0.3
    - 숫자 값: +0.2
    - 우측 셀: +0.1 (하단보다 신뢰도 높음)
    - 합리적인 값 길이: +0.1
    - 라벨과 값이 같은 행: +0.1
    
    감점 요소:
    - 단위 불일치: -0.2
    - 순수 텍스트 값: -0.3
    - 너무 긴 값: -0.2
    - 콤마/특수문자만 있는 값: -0.4
    - expected_unit 없는데 유량/압력 등 특수 단위: -0.3
    """
    score = 0.5  # 기본 점수
    
    # 값 검증
    if not value or len(value.strip()) == 0:
        return 0.0
    
    # 콤마, 점 등만 있는 경우 (잘못된 추출)
    if re.match(r'^[\s,\.\-:;]+$', value):
        return 0.0
    
    # 1) 단위 일치 확인 (동의어 시스템 활용)
    if expected_unit and unit:
        if units_match(expected_unit, unit):
            # 정확한 동의어 매칭
            score += 0.3
        else:
            # 단위가 있지만 일치하지 않음
                score -= 0.15
    elif expected_unit and not unit:
        # 예상 단위가 있는데 추출 안 됨
        score -= 0.1
    elif not expected_unit and unit:
        # 예상 단위가 없는데 유량/압력 등 특수 단위가 추출됨
        # -> 이 값이 진짜 이 라벨의 값인지 의심스러움
        unit_lower = unit.lower()
        special_units = ['m3/h', 'nm3/h', 'l/min', 'mpa', 'kpa', 'bar', 'mlc', 'mth']
        if any(su in unit_lower for su in special_units):
            score -= 0.25
    
    # 2) 숫자 값 여부
    has_digit = bool(re.search(r'\d', value))
    if has_digit:
        score += 0.2
        # 순수 숫자+단위면 추가 점수
        if re.match(r'^[\d,\.\s]+$', value.replace(' ', '')):
            score += 0.15
    else:
        # 숫자 없는 값은 크게 감점
        score -= 0.25
    
    # 3) 위치 (우측 > 하단)
    if is_right:
        score += 0.1
    
    # 4) 값 길이 검사
    val_len = len(value)
    if val_len > 100:
        score -= 0.3
    elif val_len > 50:
        score -= 0.2
    elif val_len < 20 and has_digit:
        score += 0.1
    
    # 5) 값이 다른 라벨처럼 보이면 감점
    # 순수 알파벳+공백이고 단위가 없으면
    if re.match(r'^[A-Za-z\s\-]+$', value) and len(value) > 3:
        if not unit:
            score -= 0.3
        else:
            score -= 0.1
    
    # 6) 값과 라벨이 너무 비슷하면 감점 (자기 자신을 추출한 경우)
    val_key = norm_key(value)
    label_key = norm_key(label_cell)
    if val_key and label_key and (val_key in label_key or label_key in val_key):
        score -= 0.2
    
    return min(max(score, 0.0), 1.0)


def rule_extract_by_extwg(
    table_chunks: List[DocChunk],
    extwg: str,
    umgv_desc: str,
    mat_attr_desc: str = "",
    glossary_keywords: Optional[List[str]] = None,
) -> Optional[Tuple[str, str, float, str]]:
    """
    v40 개선된 extwg 기반 테이블 추출.
    
    개선사항:
    - mat_attr_desc 유사도 매칭 지원 (Jaccard 유사도)
    - 다중 헤더 테이블 지원
    - 정확한 컬럼 인덱스 매핑
    - 약어 정규화 (CFW→COOLING FW, P/P→PUMP 등)
    
    Args:
        table_chunks: 테이블 청크 리스트
        extwg: Part No. (예: YS57312)
        umgv_desc: 추출할 사양항목명 (예: "CAPACITY (m3/h)")
        mat_attr_desc: 자재속성그룹명 (예: "COOLING FW BOOSTER PUMP")
        glossary_keywords: 용어집에서 가져온 추가 키워드
        
    Returns:
        (value, unit, confidence, reason) 또는 None
    """
    if not umgv_desc:
        return None
    
    # ========================================
    # 헬퍼 함수들
    # ========================================
    def normalize_desc(text: str) -> str:
        """Description 정규화 - 약어 확장 및 특수문자 제거"""
        if not text:
            return ""
        text = str(text).upper()
        # 약어 확장
        text = text.replace('CFW', 'COOLING FW')
        text = text.replace('CSW', 'COOLING SW')
        text = text.replace('P/P', 'PUMP')
        text = text.replace('CIRC.', 'CIRCULATION')
        text = text.replace('DISCH.', 'DISCHARGE')
        text = text.replace('A/C', 'AIRCON')
        text = text.replace('ACCOM.', 'ACCOM')
        # 특수문자 제거
        text = re.sub(r'[^A-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_keywords(text: str) -> set:
        """핵심 키워드 추출 - 불용어 제거"""
        words = set(normalize_desc(text).split())
        stopwords = {'FOR', 'THE', 'AND', 'WITH', 'TYPE', 'IN', 'OF', 'UNIT', 'SYSTEM'}
        return words - stopwords
    
    def match_desc_similarity(template_desc: str, pos_part_desc: dict, threshold: float = 0.35):
        """Jaccard 유사도 기반 mat_attr_desc 매칭"""
        template_keywords = extract_keywords(template_desc)
        best_match, best_score = None, 0.0
        
        for part_no, pos_desc in pos_part_desc.items():
            pos_keywords = extract_keywords(pos_desc)
            if template_keywords and pos_keywords:
                intersection = template_keywords & pos_keywords
                union = template_keywords | pos_keywords
                score = len(intersection) / len(union) if union else 0
                if score > best_score:
                    best_score, best_match = score, part_no
        
        return (best_match, best_score) if best_score >= threshold else (None, 0)
    
    def parse_table_data(table_text: str) -> dict:
        """테이블에서 Part No.별 데이터 추출"""
        lines = table_text.strip().split('\n')
        data = {}
        
        for line in lines:
            if '|' not in line or '---' in line:
                continue
            
            cells = [c.strip() for c in line.split('|')]
            
            # Part No. 찾기 (YS로 시작하는 5자리 숫자)
            for i, cell in enumerate(cells):
                if re.match(r'^YS\d{5}$', cell):
                    data[cell] = {
                        'cells': cells,
                        'part_idx': i,
                        'description': cells[i+1] if i+1 < len(cells) else '',
                    }
                    break
        
        return data
    
    # ========================================
    # umgv_desc → 컬럼 인덱스 매핑
    # ========================================
    UMGV_COL_MAP = {
        'CAPACITY': 6,
        'TYPE': 5,
        'EQUIPMENT TYPE': 5,
        'WATER HEAD': 9,
        'HEAD': 9,
        'TH': 9,
        "Q'TY": 4,
        'QUANTITY': 4,
    }
    
    umgv_upper = umgv_desc.upper()
    umgv_base = re.sub(r'\s*\([^)]*\)', '', umgv_desc).strip().upper()
    
    # ========================================
    # 테이블 처리
    # ========================================
    for chunk in table_chunks:
        if chunk.ctype not in ("table_md", "table_kv"):
            continue
        
        table_text = chunk.text
        table_data = parse_table_data(table_text)
        
        if not table_data:
            continue
        
        pos_part_desc = {p: d['description'] for p, d in table_data.items()}
        
        # 1. extwg 직접 매칭
        matched_part = None
        match_method = "DIRECT"
        
        if extwg and extwg in table_data:
            matched_part = extwg
        
        # 2. mat_attr_desc 유사도 매칭
        if not matched_part and mat_attr_desc:
            similar_part, score = match_desc_similarity(mat_attr_desc, pos_part_desc)
            if similar_part:
                matched_part = similar_part
                match_method = f"SIMILAR({score:.2f})"
        
        if not matched_part:
            continue
        
        # 3. umgv_desc → 컬럼 값 추출
        row_data = table_data[matched_part]
        cells = row_data['cells']
        
        # 3-1. 고정 인덱스 매핑
        col_idx = None
        for key in [umgv_upper, umgv_base]:
            for map_key, idx in UMGV_COL_MAP.items():
                if map_key in key:
                    col_idx = idx
                    break
            if col_idx:
                break
        
        if col_idx and col_idx < len(cells):
            value = cells[col_idx]
            if value and value not in ['-', '', 'N/A']:
                unit = ""
                num_match = re.search(r'([\d\.]+)\s*([a-zA-Z/³²°℃%]*)', value)
                if num_match:
                    extracted_value = num_match.group(1)
                    unit = num_match.group(2) if num_match.group(2) else ""
                else:
                    extracted_value = value
                
                # v41: 헤더에서 단위 추출 시도 (값에 단위가 없는 경우)
                if not unit:
                    header_line = table_text.split('\n')[0] if '\n' in table_text else ''
                    header_cells = [c.strip() for c in header_line.split('|')]
                    if col_idx < len(header_cells):
                        header_unit = extract_unit_from_header(header_cells[col_idx], umgv_desc)
                        if header_unit:
                            unit = header_unit
                
                confidence = 0.92 if match_method == "DIRECT" else 0.85
                reason = f"RULE_EXTWG_V40({matched_part},{match_method})"
                
                return (extracted_value, unit, confidence, reason)
        
        # 3-2. 헤더 매칭 시도
        header_line = table_text.split('\n')[0] if '\n' in table_text else ''
        header_cells = [c.strip() for c in header_line.split('|')]
        
        keywords = [umgv_upper.lower(), umgv_base.lower()]
        if glossary_keywords:
            keywords.extend([k.lower() for k in glossary_keywords if k])
        
        for i, h in enumerate(header_cells):
            h_norm = re.sub(r'[^a-z0-9]', '', h.lower())
            for kw in keywords:
                kw_norm = re.sub(r'[^a-z0-9]', '', kw)
                if kw_norm and h_norm and (kw_norm in h_norm or h_norm in kw_norm):
                    if i < len(cells):
                        value = cells[i]
                        if value and value not in ['-', '', 'N/A']:
                            unit = ""
                            num_match = re.search(r'([\d\.]+)', value)
                            if num_match:
                                extracted_value = num_match.group(1)
                            else:
                                extracted_value = value
                            
                            # v41: 헤더에서 단위 추출
                            header_unit = extract_unit_from_header(h, umgv_desc)
                            if header_unit:
                                unit = header_unit
                            
                            confidence = 0.88 if match_method == "DIRECT" else 0.80
                            reason = f"RULE_EXTWG_V40({matched_part},{match_method},header={h[:15]})"
                            
                            return (extracted_value, unit, confidence, reason)
    
    return None




def rule_extract_from_tables(
    tables_or_chunks: Union[List[Tuple[str, List[List[str]]]], List[DocChunk]],
    label_variants: List[str],
    expected_unit: str = "",
    context_keywords: Optional[List[str]] = None,
) -> Optional[Tuple[str, str, float, str]]:
    """
    테이블에서 label cell을 찾고, 인접 셀(우측/하단)에서 값을 추출합니다.
    
    개선사항:
    - 다중 후보 수집 후 점수 기반 최적 선택
    - 일반 라벨(Type, Material 등) + 단위 없음: LLM에게 위임
    - expected_unit 일치 시 가중치 부여
    
    Args:
        tables_or_chunks: 테이블 튜플 리스트 또는 DocChunk 리스트
        label_variants: 찾을 라벨 변형 목록
        expected_unit: 예상 단위
        context_keywords: 컨텍스트 확인용 키워드 (예: 장비명)
        
    Returns:
        (value, unit, confidence, reason) 또는 None
    """
    labels = [norm_key(x) for x in label_variants if norm_key(x) and len(norm_key(x)) >= 3]
    if not labels:
        return None
    
    # 일반 라벨 여부 확인
    is_generic = any(_is_generic_label(lv) for lv in label_variants)
    
    # 일반 라벨 + 예상 단위 없음: Rule로는 정확한 추출이 어려움 -> LLM에게 위임
    if is_generic and not expected_unit:
        return None

    # DocChunk 리스트를 테이블 형식으로 변환
    tables: List[Tuple[str, List[List[str]]]] = []
    for item in tables_or_chunks:
        if len(tables) >= 200:
            break
        if isinstance(item, DocChunk):
            if item.raw_rows:
                tables.append((f"chunk_{len(tables)}", item.raw_rows))
        elif isinstance(item, tuple) and len(item) == 2:
            tables.append(item)

    # 후보 수집 (일반 라벨이면 여러 개, 아니면 첫 번째만)
    candidates: List[Tuple[str, str, float, str, float]] = []  # (value, unit, base_conf, reason, score)
    max_candidates = 10 if is_generic else 3

    for ti, (_tname, rows) in enumerate(tables):
        if not rows:
            continue
        if len(candidates) >= max_candidates:
            break
            
        nrow = len(rows)
        ncol = max(len(r) for r in rows) if rows else 0
        
        # normalize cells
        mat = []
        for r in rows:
            rr = [norm(r[i]) if i < len(r) else "" for i in range(ncol)]
            mat.append(rr)

        for ri in range(nrow):
            if len(candidates) >= max_candidates:
                break
                
            for ci in range(ncol):
                cell = mat[ri][ci]
                cell_key = norm_key(cell)
                if not cell_key:
                    continue
                
                hit = None
                for lb in labels[:10]:
                    # 정확한 매칭 또는 포함 매칭
                    if cell_key == lb or lb in cell_key:
                        hit = lb
                        break
                    # 셀 키가 라벨의 첫 단어를 포함하면 매칭 (예: 'capacity' in 'capacity ton')
                    if cell_key in lb:
                        hit = lb
                        break
                    # 라벨의 첫 단어가 셀 키와 일치하면 매칭
                    lb_first = lb.split()[0] if lb.split() else lb
                    if lb_first and (cell_key == lb_first or lb_first in cell_key):
                        hit = lb
                        break
                if not hit:
                    continue

                # 후보1: 우측 셀
                right = ""
                right_cells = []
                for cj in range(ci + 1, min(ncol, ci + 5)):
                    if mat[ri][cj]:
                        right_cells.append(mat[ri][cj])
                        if not right and norm_key(mat[ri][cj]) not in labels:
                            right = mat[ri][cj]

                # 후보2: 하단 셀
                down = ""
                down_cells = []
                for rj in range(ri + 1, min(nrow, ri + 4)):
                    if mat[rj][ci]:
                        down_cells.append(mat[rj][ci])
                        if not down and norm_key(mat[rj][ci]) not in labels:
                            down = mat[rj][ci]

                # 우측 후보 평가
                if right:
                    v, u = split_value_unit(right, expected_unit=expected_unit)
                    if v:
                        base_conf = 0.80
                        score = _score_extraction_candidate(
                            v, u, expected_unit, cell, right_cells, is_right=True
                        )
                        candidates.append((v, u, base_conf, f"RULE_TABLE(t{ti},r{ri},c{ci})", score))

                # 하단 후보 평가
                if down and down != right:
                    v, u = split_value_unit(down, expected_unit=expected_unit)
                    if v:
                        base_conf = 0.74
                        score = _score_extraction_candidate(
                            v, u, expected_unit, cell, down_cells, is_right=False
                        )
                        candidates.append((v, u, base_conf, f"RULE_TABLE(t{ti},r{ri},c{ci}:down)", score))
    
    if not candidates:
        return None
    
    # 점수 기반 정렬
    candidates.sort(key=lambda x: x[4], reverse=True)
    
    # 최고 점수 후보 선택
    best = candidates[0]
    v, u, base_conf, reason, score = best
    
    # 일반 라벨 + 예상 단위 없음: 더 보수적으로 (LLM에게 맡김)
    if is_generic and not expected_unit:
        min_score = 0.75
    elif is_generic:
        min_score = 0.65
    else:
        min_score = 0.55
    
    if score < min_score:
        return None
    
    # 최종 신뢰도 = 기본 신뢰도 * 점수
    final_conf = base_conf * score
    
    # 일반 라벨이고 점수가 낮으면 신뢰도 추가 감소
    if is_generic and score < 0.7:
        final_conf *= 0.8
    
    return v, u, final_conf, reason


def rule_extract_from_text(
    doc_blob: str, 
    label_variants: List[str], 
    expected_unit: str = ""
) -> Optional[Tuple[str, str, float, str]]:
    """
    본문에서 'label : value' 패턴을 찾아 추출합니다.
    """
    if not doc_blob:
        return None
    
    labels = [x for x in label_variants if x and len(x) >= 3]
    for lb in labels[:10]:
        # label 이후 80자 정도에서 값 추출 (과도한 regex는 위험하니 보수적으로)
        pat = re.compile(re.escape(lb) + r"\s*[:=\-]\s*([^\n\r]{1,100})", flags=re.I)
        m = pat.search(doc_blob)
        if m:
            raw = m.group(1)
            v, u = split_value_unit(raw, expected_unit=expected_unit)
            if v:
                return v, u, 0.68, "RULE_TEXT"
    
    return None


def local_audit(evidence: str, value: str, unit: str) -> bool:
    """
    LLM 결과의 최소 검증: evidence 내에 값이 직접 존재하는지(근사) 확인
    """
    v = norm(value)
    if not v:
        return False
    
    ev = norm_key(evidence)
    
    # 숫자는 콤마/공백 제거해서도 확인
    v1 = norm_key(v)
    if v1 and v1 in ev:
        return True
    
    v_num = re.sub(r"[,\s]", "", v1)
    if v_num and v_num in re.sub(r"[,\s]", "", ev):
        return True
    
    # 단위까지 같이 확인(있다면)
    u = norm_key(unit)
    if u and (v1 + " " + u) in ev:
        return True
    
    return False


# =============================================================================
# Glossary 필드 부착 (spec_df에 __gl_* 컬럼 추가)
# =============================================================================

def attach_glossary_fields(
    spec_df: pd.DataFrame,
    glossary_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
    hybrid_min_score: float = 0.68,
    hybrid_top_k: int = 1,
    hybrid_limit: int = 250000,
) -> pd.DataFrame:
    """
    spec_df에 Glossary 정보를 __gl_* 컬럼으로 붙입니다.

    1) umgv_code exact 매칭
    2) umgv_desc exact 매칭
    3) 나머지는 하이브리드(편집거리+SimHash) 매칭(상한 hybrid_limit)

    붙이는 컬럼(존재할 때):
    - __gl_pos_umgv_desc, __gl_pos_chunk, __gl_table_text, __gl_evidence_fb, 
      __gl_value_format, __gl_umgv_uom, __gl_section_num
    """
    if spec_df is None or spec_df.empty or glossary_df is None or glossary_df.empty:
        return spec_df

    log = logger or logging.getLogger("glossary")
    df = spec_df.copy()
    gl = glossary_df.fillna("").copy()

    # 풍부한 행 우선순위로 dedup
    def _rich(row: pd.Series) -> int:
        s = 0
        for k in ["pos_chunk", "pos_umgv_desc", "table_text", "evidence_fb", 
                  "value_format", "umgv_uom", "pos_umgv_uom", "section_num"]:
            if norm(row.get(k, "")):
                s += 1
        return s

    gl["__rich"] = gl.apply(_rich, axis=1)
    gl["__code_key"] = gl["umgv_code"].apply(norm_key) if "umgv_code" in gl.columns else ""
    gl["__desc_key"] = gl["umgv_desc"].apply(norm_key) if "umgv_desc" in gl.columns else ""

    # code best (가장 풍부한 행 우선)
    if "__code_key" in gl.columns:
        gl_code = gl.sort_values(["__code_key", "__rich"], ascending=[True, False])
        gl_code = gl_code[gl_code["__code_key"] != ""].drop_duplicates("__code_key")
    else:
        gl_code = pd.DataFrame()

    # desc best
    if "__desc_key" in gl.columns:
        gl_desc = gl.sort_values(["__desc_key", "__rich"], ascending=[True, False])
        gl_desc = gl_desc[gl_desc["__desc_key"] != ""].drop_duplicates("__desc_key")
    else:
        gl_desc = pd.DataFrame()

    # spec keys
    if "umgv_code" in df.columns:
        df["__code_key"] = df["umgv_code"].apply(norm_key)
    else:
        df["__code_key"] = ""

    if "umgv_desc" in df.columns:
        df["__desc_key"] = df["umgv_desc"].apply(norm_key)
    elif COL_SPEC_NAME in df.columns:
        df["__desc_key"] = df[COL_SPEC_NAME].apply(norm_key)
    else:
        df["__desc_key"] = ""

    # __gl_* 컬럼 초기화
    gl_fields = ["pos_umgv_desc", "pos_chunk", "table_text", "evidence_fb", 
                 "value_format", "umgv_uom", "pos_umgv_uom", "pos_umgv_value", "section_num"]
    for f in gl_fields:
        df[f"__gl_{f}"] = ""

    # code_key -> dict 형태로 변환(빠른 조회)
    code_lookup: Dict[str, Dict[str, Any]] = {}
    if not gl_code.empty:
        for _, row in gl_code.iterrows():
            ck = row.get("__code_key", "")
            if ck:
                code_lookup[ck] = {f: row.get(f, "") for f in gl_fields}

    desc_lookup: Dict[str, Dict[str, Any]] = {}
    if not gl_desc.empty:
        for _, row in gl_desc.iterrows():
            dk = row.get("__desc_key", "")
            if dk:
                desc_lookup[dk] = {f: row.get(f, "") for f in gl_fields}

    # 1) code exact match
    matched_mask = pd.Series([False] * len(df), index=df.index)
    for idx, row in df.iterrows():
        ck = row.get("__code_key", "")
        if ck and ck in code_lookup:
            for f in gl_fields:
                df.at[idx, f"__gl_{f}"] = code_lookup[ck].get(f, "")
            matched_mask[idx] = True

    # 2) desc exact match (code 미매칭만)
    for idx, row in df.iterrows():
        if matched_mask[idx]:
            continue
        dk = row.get("__desc_key", "")
        if dk and dk in desc_lookup:
            for f in gl_fields:
                df.at[idx, f"__gl_{f}"] = desc_lookup[dk].get(f, "")
            matched_mask[idx] = True

    # 3) 하이브리드 매칭 (남은 것만, 상한 적용)
    missing_idx = df.index[~matched_mask].tolist()
    
    if missing_idx and len(missing_idx) <= hybrid_limit:
        log.info("Glossary exact match miss: %d rows. Hybrid matching...", len(missing_idx))
        gl_index = GlossaryIndex(gl)

        # 캐시(동일 desc 반복 대응)
        cache: Dict[str, Optional[Dict[str, Any]]] = {}

        try:
            from tqdm import tqdm as _tqdm
            iters = _tqdm(missing_idx, desc="Hybrid glossary match")
        except ImportError:
            iters = missing_idx

        for idx in iters:
            dk = df.at[idx, "__desc_key"]
            if not dk:
                continue
            
            # 캐시 확인
            if dk in cache:
                if cache[dk]:
                    for f in gl_fields:
                        df.at[idx, f"__gl_{f}"] = cache[dk].get(f, "")
                continue
            
            # 하이브리드 검색 (match 메서드 사용)
            results = gl_index.match(spec_name=dk, umgv_desc=dk, top_k=hybrid_top_k)
            # results는 GlossaryMatch 리스트이며, score >= min_score인 것만 사용
            valid_results = [r for r in results if r.score >= hybrid_min_score]
            if valid_results:
                best = valid_results[0].row  # GlossaryMatch.row가 실제 데이터 딕셔너리
                fld_vals = {f: best.get(f, "") for f in gl_fields}
                cache[dk] = fld_vals
                for f in gl_fields:
                    df.at[idx, f"__gl_{f}"] = fld_vals.get(f, "")
            else:
                cache[dk] = None
    elif missing_idx:
        log.warning("Hybrid glossary skipped: %d rows > limit %d", len(missing_idx), hybrid_limit)

    # 임시 컬럼 정리
    df.drop(columns=["__code_key", "__desc_key"], errors="ignore", inplace=True)
    
    return df


# =============================================================================
# Ollama Serve Manager (자동 서버 시작)
# =============================================================================

class OllamaServeManager:
    """
    Ollama 서버를 자동으로 시작하고 관리합니다.
    지정된 포트들에 대해 serve 프로세스를 자동 실행/유지합니다.
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.host = config.ollama_host
        self.ports = list(config.ollama_ports)
        self.procs: Dict[int, subprocess.Popen] = {}
        self._log_files: Dict[int, Any] = {}
    
    def _find_ollama_binary(self) -> str:
        """Ollama 바이너리 경로를 찾습니다."""
        # 설정에서 지정된 경로
        if self.config.ollama_bin and os.path.isfile(self.config.ollama_bin):
            return self.config.ollama_bin
        
        # 일반적인 경로들
        candidates = [
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
            os.path.expanduser("~/.ollama/bin/ollama"),
            "/opt/ollama/bin/ollama",
            "ollama",  # PATH에서 찾기
        ]
        
        for path in candidates:
            if path == "ollama":
                # which 명령어로 찾기
                try:
                    result = subprocess.run(
                        ["which", "ollama"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return result.stdout.strip()
                except Exception:
                    pass
            elif os.path.isfile(path):
                return path
        
        return "ollama"  # 기본값
    
    def _is_port_in_use(self, port: int) -> bool:
        """포트가 사용 중인지 확인합니다."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((self.host, port))
                return False
            except socket.error:
                return True
    
    def _is_ollama_alive(self, port: int) -> bool:
        """특정 포트의 Ollama 서버가 응답하는지 확인합니다."""
        try:
            url = f"http://{self.host}:{port}/api/tags"
            if HAS_REQUESTS:
                r = requests.get(url, timeout=5)
                return r.status_code == 200
            else:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    def _wait_for_server(self, port: int, timeout: int = 60) -> bool:
        """서버가 준비될 때까지 대기합니다."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_ollama_alive(port):
                return True
            time.sleep(2)
        return False
    
    def start_all(self) -> Dict[int, bool]:
        """
        모든 포트에서 Ollama 서버를 시작합니다.
        
        Returns:
            {port: success} 딕셔너리
        """
        if not self.config.auto_start_ollama_serves:
            self.logger.info("[OllamaServeManager] auto_start disabled, skipping")
            return {p: self._is_ollama_alive(p) for p in self.ports}
        
        ollama_bin = self._find_ollama_binary()
        self.logger.info("[OllamaServeManager] Using ollama binary: %s", ollama_bin)
        
        # 로그 디렉토리 생성
        log_dir = self.config.partial_output_path
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 모델 디렉토리 설정
        models_dir = self.config.ollama_models_dir
        if models_dir:
            os.makedirs(models_dir, exist_ok=True)
        
        results = {}
        
        for port in self.ports:
            # 이미 실행 중인지 확인
            if self._is_ollama_alive(port):
                self.logger.info("[OllamaServeManager] Port %d already alive", port)
                results[port] = True
                continue
            
            # 포트가 다른 프로세스에 의해 사용 중인지 확인
            if self._is_port_in_use(port):
                self.logger.warning("[OllamaServeManager] Port %d in use by another process", port)
                results[port] = False
                continue
            
            # 환경 변수 설정
            env = os.environ.copy()
            env["OLLAMA_HOST"] = f"0.0.0.0:{port}"  # 모든 인터페이스에서 수신
            if models_dir:
                env["OLLAMA_MODELS"] = models_dir
            
            # 로그 파일
            log_path = os.path.join(log_dir, f"ollama_serve_{port}.log") if log_dir else f"/tmp/ollama_serve_{port}.log"
            
            try:
                # 로그 파일 열기
                log_file = open(log_path, "ab", buffering=0)
                self._log_files[port] = log_file
                
                # serve 프로세스 시작
                cmd = [ollama_bin, "serve"]
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    env=env,
                    start_new_session=True,  # 독립 세션으로 실행
                )
                self.procs[port] = proc
                
                self.logger.info(
                    "[OllamaServeManager] Started port=%d pid=%d log=%s",
                    port, proc.pid, log_path
                )
                
            except Exception as e:
                self.logger.error(
                    "[OllamaServeManager] Failed to start port=%d: %s",
                    port, e
                )
                results[port] = False
                continue
        
        # 서버들이 준비될 때까지 대기
        grace_sec = self.config.ollama_serve_start_grace_sec or 10
        self.logger.info("[OllamaServeManager] Waiting %d seconds for servers to start...", grace_sec)
        time.sleep(grace_sec)
        
        # 각 포트 상태 확인
        for port in self.ports:
            if port not in results:  # 아직 결과가 없는 포트만
                alive = self._is_ollama_alive(port)
                results[port] = alive
                if alive:
                    self.logger.info("[OllamaServeManager] Port %d ready", port)
                else:
                    self.logger.warning("[OllamaServeManager] Port %d not responding", port)
        
        return results
    
    def ensure_model_loaded(self, model: str, port: int) -> bool:
        """특정 포트에서 모델이 로드되어 있는지 확인하고, 없으면 로드합니다."""
        try:
            url = f"http://{self.host}:{port}/api/tags"
            if HAS_REQUESTS:
                r = requests.get(url, timeout=10)
                if r.status_code != 200:
                    return False
                data = r.json()
            else:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            
            # 모델 목록 확인
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if model in model_names:
                return True
            
            # 모델이 없으면 pull 시도
            self.logger.info("[OllamaServeManager] Model %s not found on port %d, pulling...", model, port)
            
            pull_url = f"http://{self.host}:{port}/api/pull"
            payload = {"name": model, "stream": False}
            
            if HAS_REQUESTS:
                r = requests.post(pull_url, json=payload, timeout=600)  # 10분 타임아웃
                return r.status_code == 200
            else:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    pull_url, data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return resp.status == 200
                    
        except Exception as e:
            self.logger.warning("[OllamaServeManager] Model check failed: %s", e)
            return False
    
    def stop_all(self) -> None:
        """모든 관리 중인 서버 프로세스를 종료합니다."""
        for port, proc in list(self.procs.items()):
            try:
                proc.terminate()
                proc.wait(timeout=10)
                self.logger.info("[OllamaServeManager] Stopped port=%d", port)
            except Exception as e:
                self.logger.warning("[OllamaServeManager] Stop failed port=%d: %s", port, e)
                try:
                    proc.kill()
                except Exception:
                    pass
        
        # 로그 파일 닫기
        for port, log_file in self._log_files.items():
            try:
                log_file.close()
            except Exception:
                pass
        
        self.procs.clear()
        self._log_files.clear()
    
    def __del__(self):
        """소멸자: 관리 중인 프로세스 정리"""
        # 기본적으로 프로세스를 유지 (다른 세션에서도 사용 가능하도록)
        # 명시적으로 stop_all()을 호출해야 종료됨
        pass


# =============================================================================
# LLM Client (Ollama 호출)
# =============================================================================

class LLMClient:
    """
    Ollama 서버에 LLM 요청을 보내는 클라이언트.
    rate limiting, retry, timeout 지원.
    requests가 없으면 urllib로 대체합니다.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "llama3.1:70b",
        timeout: float = 180.0,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        self.host = host
        self.port = port
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.log = logger or logging.getLogger("llm_client")
        self.base_url = f"http://{host}:{port}"
        
        # 헬스 체크용
        self._last_health_check = 0.0
        self._is_healthy = False

    def _http_get(self, url: str, timeout: float = 10.0) -> Tuple[int, str]:
        """HTTP GET 요청 (requests 또는 urllib 사용)"""
        if HAS_REQUESTS:
            try:
                r = requests.get(url, timeout=timeout)
                return r.status_code, r.text
            except Exception as e:
                return 0, str(e)
        else:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.status, resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                return 0, str(e)

    def _http_post_json(self, url: str, payload: Dict, timeout: float) -> Tuple[int, str]:
        """HTTP POST JSON 요청 (requests 또는 urllib 사용)"""
        if HAS_REQUESTS:
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                return r.status_code, r.text
            except requests.exceptions.Timeout:
                return 0, "TIMEOUT"
            except requests.exceptions.RequestException as e:
                return 0, str(e)
        else:
            try:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    url, data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.status, resp.read().decode("utf-8", errors="replace")
            except urllib.error.URLError as e:
                return 0, str(e)
            except Exception as e:
                return 0, str(e)

    def health_check(self) -> bool:
        """서버 상태 확인"""
        try:
            url = f"{self.base_url}/api/tags"
            status, _ = self._http_get(url, timeout=10)
            self._is_healthy = status == 200
            self._last_health_check = time.time()
            return self._is_healthy
        except Exception as e:
            self.log.debug("Health check failed for %s:%d - %s", self.host, self.port, e)
            self._is_healthy = False
            return False

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """
        LLM 응답 생성.
        
        Args:
            prompt: 사용자 프롬프트
            system: 시스템 프롬프트
            temperature: 샘플링 온도
            max_tokens: 최대 토큰 수
            
        Returns:
            응답 텍스트 또는 None(실패 시)
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            payload["system"] = system

        for attempt in range(self.max_retries):
            status, text = self._http_post_json(url, payload, self.timeout)
            
            if status == 200:
                try:
                    data = json.loads(text)
                    return data.get("response", "")
                except json.JSONDecodeError:
                    self.log.warning("LLM JSON parse failed (port=%d)", self.port)
            elif text == "TIMEOUT":
                self.log.warning(
                    "LLM timeout (port=%d, attempt=%d/%d)", 
                    self.port, attempt + 1, self.max_retries
                )
            else:
                self.log.warning(
                    "LLM request failed (port=%d, attempt=%d/%d): status=%d, error=%s",
                    self.port, attempt + 1, self.max_retries, status, text[:100]
                )
            
            # 재시도 전 대기
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        
        return None

    def __repr__(self) -> str:
        return f"LLMClient({self.host}:{self.port}, model={self.model})"


class LLMClientPool:
    """
    여러 Ollama 포트에 대한 클라이언트 풀.
    라운드로빈 또는 건강한 서버 선택 지원.
    """

    def __init__(
        self,
        host: str,
        ports: List[int],
        model: str,
        timeout: float = 180.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.host = host
        self.ports = ports
        self.model = model
        self.log = logger or logging.getLogger("llm_pool")
        
        # 각 포트별 클라이언트 생성
        self.clients: Dict[int, LLMClient] = {}
        for port in ports:
            self.clients[port] = LLMClient(
                host=host, port=port, model=model, 
                timeout=timeout, logger=self.log
            )
        
        self._rr_index = 0  # 라운드로빈 인덱스
        self._lock = threading.Lock()

    def check_health_all(self) -> Dict[int, bool]:
        """모든 포트 상태 확인"""
        results = {}
        for port, client in self.clients.items():
            results[port] = client.health_check()
        return results

    def get_healthy_clients(self) -> List[LLMClient]:
        """건강한 클라이언트만 반환"""
        return [c for c in self.clients.values() if c._is_healthy]

    def get_next_client(self) -> Optional[LLMClient]:
        """라운드로빈으로 다음 클라이언트 반환"""
        with self._lock:
            healthy = self.get_healthy_clients()
            if not healthy:
                # 헬스 체크 후 재시도
                self.check_health_all()
                healthy = self.get_healthy_clients()
            
            if not healthy:
                return None
            
            client = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return client

    def generate_with_voting(
        self,
        prompt: str,
        system: str = "",
        k: int = 2,
        temperature: float = 0.0,
    ) -> List[Tuple[int, Optional[str]]]:
        """
        k개의 서버에서 동시 생성 후 결과 반환 (투표용).
        
        Args:
            prompt: 프롬프트
            system: 시스템 프롬프트
            k: 투표 수
            temperature: 온도
            
        Returns:
            [(port, response), ...] 리스트
        """
        healthy = self.get_healthy_clients()
        if not healthy:
            self.check_health_all()
            healthy = self.get_healthy_clients()
        
        if not healthy:
            return []
        
        # k개 선택 (부족하면 있는 만큼)
        selected = healthy[:min(k, len(healthy))]
        
        results: List[Tuple[int, Optional[str]]] = []
        
        # 병렬 호출
        def _call(client: LLMClient) -> Tuple[int, Optional[str]]:
            resp = client.generate(prompt, system=system, temperature=temperature)
            return (client.port, resp)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as ex:
            futures = [ex.submit(_call, c) for c in selected]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    self.log.warning("Voting call failed: %s", e)
        
        return results


# =============================================================================
# 다수결 투표 (Majority Voting)
# =============================================================================

def parse_llm_response(response: str) -> Dict[str, str]:
    """
    LLM 응답 파싱. 다양한 형식 지원:
    - JSON: {"value": "...", "unit": "...", "evidence": "..."}
    - 태그: <value>...</value> <unit>...</unit>
    - 키-값: value: ... / unit: ...
    """
    result = {"value": "", "unit": "", "evidence": "", "confidence": ""}
    
    if not response:
        return result
    
    # 1) JSON 시도
    try:
        # JSON 블록 추출
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            result["value"] = str(data.get("value", data.get("val", ""))).strip()
            result["unit"] = str(data.get("unit", data.get("uom", ""))).strip()
            result["evidence"] = str(data.get("evidence", data.get("ev", ""))).strip()
            result["confidence"] = str(data.get("confidence", data.get("conf", ""))).strip()
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 2) XML 태그 시도
    for key in ["value", "unit", "evidence", "confidence"]:
        pattern = rf'<{key}>\s*(.*?)\s*</{key}>'
        m = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if m:
            result[key] = m.group(1).strip()
    
    if result["value"]:
        return result
    
    # 3) 키-값 패턴
    patterns = {
        "value": [r'(?:value|val|값)\s*[:=]\s*([^\n,]+)', r'^([^\n:]+)$'],
        "unit": [r'(?:unit|uom|단위)\s*[:=]\s*([^\n,]+)'],
        "evidence": [r'(?:evidence|ev|근거)\s*[:=]\s*([^\n]+)'],
    }
    
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, response, re.IGNORECASE)
            if m:
                result[key] = m.group(1).strip()
                break
    
    return result


def majority_vote(
    responses: List[Tuple[int, Optional[str]]],
    min_agreement: int = 2,
) -> Tuple[Dict[str, str], float, str]:
    """
    여러 LLM 응답에서 다수결 투표.
    
    Args:
        responses: [(port, response_text), ...]
        min_agreement: 최소 일치 수
        
    Returns:
        (best_result, confidence, vote_detail)
        - best_result: {"value": ..., "unit": ..., "evidence": ...}
        - confidence: 0.0 ~ 1.0
        - vote_detail: 투표 상세 정보
    """
    if not responses:
        return {"value": "", "unit": "", "evidence": ""}, 0.0, "NO_RESPONSES"
    
    # 파싱
    parsed: List[Tuple[int, Dict[str, str]]] = []
    for port, resp in responses:
        if resp:
            p = parse_llm_response(resp)
            parsed.append((port, p))
    
    if not parsed:
        return {"value": "", "unit": "", "evidence": ""}, 0.0, "ALL_PARSE_FAILED"
    
    # 값 기준 그룹화
    value_groups: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    for port, p in parsed:
        v = norm_key(p.get("value", ""))
        if v not in value_groups:
            value_groups[v] = []
        value_groups[v].append((port, p))
    
    # 빈 값 제외하고 가장 많은 그룹 찾기
    best_group = None
    best_count = 0
    for v, group in value_groups.items():
        if v and len(group) > best_count:
            best_count = len(group)
            best_group = group
    
    # 빈 값만 있는 경우
    if not best_group:
        if "" in value_groups:
            # 모든 응답이 빈 값 = "값 없음" 합의
            return {"value": "", "unit": "", "evidence": ""}, 0.5, "AGREED_EMPTY"
        return {"value": "", "unit": "", "evidence": ""}, 0.0, "NO_VALID_VALUES"
    
    # 합의 확인
    total = len(parsed)
    conf = best_count / total if total > 0 else 0.0
    
    if best_count >= min_agreement:
        # 그룹 내에서 가장 완전한 결과 선택
        best_result = max(best_group, key=lambda x: len(x[1].get("evidence", "")))[1]
        ports_agreed = [str(p) for p, _ in best_group]
        detail = f"VOTE_OK({best_count}/{total}):ports={','.join(ports_agreed)}"
        return best_result, conf, detail
    else:
        # 합의 실패 - 첫 번째 비어있지 않은 값 반환
        for port, p in parsed:
            if norm_key(p.get("value", "")):
                detail = f"VOTE_WEAK({best_count}/{total}):fallback_port={port}"
                return p, conf * 0.7, detail
        
        return {"value": "", "unit": "", "evidence": ""}, 0.0, "VOTE_ALL_FAILED"


# =============================================================================
# 프롬프트 빌더
# =============================================================================

# =============================================================================
# LLM Prompts (English for llama3.1 optimization)
# =============================================================================

SYSTEM_PROMPT_EXTRACTION = """You are an expert in extracting specification values from shipyard POS (Purchase Order Specification) documents.

Your task is to extract the requested specification item's value and unit from the provided document evidence.

CRITICAL RULES:
1. ONLY extract values that LITERALLY EXIST in the "Document Evidence" section below
2. DO NOT use values from "Reference Hints" or "Examples from Similar POS" - those are ONLY for understanding the pattern/format
3. If the target value/unit CANNOT BE FOUND in the evidence, you MUST return empty strings: {"value": "", "unit": "", "pos_desc": ""}
4. NEVER fabricate, guess, or use values from hints/examples as the actual extraction result
5. The evidence MUST contain the actual specification label and value you extract

VALUE EXTRACTION RULES:
- Keep numbers in original format (e.g., "1,000" stays as "1,000")
- Keep ranges as-is (e.g., "10~20" stays as "10~20")
- For tables, the value is typically in the cell adjacent to or below the label
- Extract units only when they appear WITH the value in the evidence

RESPONSE FORMAT (JSON only, no other text):
{"value": "extracted value or empty", "unit": "unit or empty", "pos_desc": "label as written in evidence"}

IMPORTANT: If you cannot find the specification in the evidence, respond with:
{"value": "", "unit": "", "pos_desc": ""}"""


def build_extraction_prompt(
    spec_name: str,
    spec_code: str,
    evidence: str,
    hints: Optional[Dict[str, str]] = None,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    mat_attr_desc: str = "",  # 자재속성그룹명 (핵심 장비 식별자)
    umg_desc: str = "",    # UMG 설명
) -> str:
    """
    Build extraction prompt for LLM (v41 enhanced).
    
    v41 개선사항:
    - 과거 데이터(few-shot)와 현재 POS chunk를 명확히 구분
    - 한글 값 사용 금지 명시
    - 참조 정보는 패턴 참고용임을 강조
    - 값이 evidence에 반드시 존재해야 함을 강조
    
    Args:
        spec_name: Specification item name (umgv_desc)
        spec_code: Specification code (umgv_code)
        evidence: Document evidence
        hints: Additional hints from glossary/specDB
        few_shot_examples: 유사 POS에서 추출된 예시 [{umgv_desc, value, unit, section, table}, ...]
        mat_attr_desc: 자재속성그룹명 (예: "ME FO SUPPLY MODULE")
        umg_desc: UMG 설명 (예: "FO SUPPLY MODULE")
        
    Returns:
        Prompt string
    """
    prompt_parts = []
    
    # v41: 기본 지시사항 강화
    prompt_parts.append("# SPECIFICATION VALUE EXTRACTION TASK")
    prompt_parts.append("")
    prompt_parts.append("## CRITICAL RULES")
    prompt_parts.append("1. Extract values ONLY from the 'DOCUMENT EVIDENCE' section below")
    prompt_parts.append("2. NEVER use values from 'Reference Hints' or 'Examples' sections - they are OTHER ships' data")
    prompt_parts.append("3. If the value is not found in DOCUMENT EVIDENCE, return empty strings")
    prompt_parts.append("4. NEVER return Korean text (한글) - POS documents are in English only")
    prompt_parts.append("")
    
    # Basic info - mat_attr_desc를 최우선으로 표시
    prompt_parts.append("## Target Specification Item")
    if mat_attr_desc:
        prompt_parts.append(f"- Equipment/Material: {mat_attr_desc}")
    if umg_desc and umg_desc != mat_attr_desc:
        prompt_parts.append(f"- Category: {umg_desc}")
    prompt_parts.append(f"- Specification Name: {spec_name}")
    if spec_code:
        prompt_parts.append(f"- Code: {spec_code}")
    
    # v41: Hints - 명확히 참고용임을 더욱 강조
    if hints:
        prompt_parts.append("")
        prompt_parts.append("## Reference Hints (⚠️ FOR FORMAT REFERENCE ONLY - DO NOT COPY VALUES)")
        prompt_parts.append("These hints show expected FORMAT/PATTERN from other vessels, NOT actual values:")
        if hints.get("gl_desc"):
            prompt_parts.append(f"- Typical label: {hints['gl_desc']}")
        if hints.get("gl_value_format"):
            prompt_parts.append(f"- Expected format: {hints['gl_value_format']}")
        if hints.get("gl_uom"):
            prompt_parts.append(f"- Expected unit: {hints['gl_uom']}")
        if hints.get("similar_value"):
            prompt_parts.append(f"- Similar vessel value (DO NOT USE): {hints['similar_value']}")
        prompt_parts.append("⚠️ These are from OTHER ships. Find values in THIS document below.")
    
    # v41: Few-shot examples - 명확히 패턴 참고용임을 강조
    if few_shot_examples:
        prompt_parts.append("")
        prompt_parts.append("## Examples from Other Vessels (⚠️ PATTERN REFERENCE ONLY)")
        prompt_parts.append("These patterns are from OTHER ships. DO NOT use these values:")
        for i, ex in enumerate(few_shot_examples[:3], 1):
            ex_line = f"{i}. Ship {ex.get('hull', '?')}: \"{ex.get('umgv_desc', spec_name)}\" = \"{ex.get('value', '')}\" {ex.get('unit', '')}"
            if ex.get('section'):
                ex_line += f" (section {ex['section']})"
            prompt_parts.append(ex_line)
        prompt_parts.append("⚠️ Search for similar patterns in THIS document, but extract THIS document's values.")
    
    # v41: Evidence - 여기서만 추출해야 함을 강조 (더 명확한 구분)
    prompt_parts.append("")
    prompt_parts.append("=" * 70)
    prompt_parts.append("## DOCUMENT EVIDENCE (✓ EXTRACT FROM HERE ONLY)")
    prompt_parts.append("=" * 70)
    prompt_parts.append(evidence)
    prompt_parts.append("=" * 70)
    prompt_parts.append("")
    
    # v41: Instruction - 더욱 명확하게
    prompt_parts.append("## Your Task")
    if mat_attr_desc:
        prompt_parts.append(f"1. Find '{mat_attr_desc}' (or similar equipment) in the DOCUMENT EVIDENCE above")
        prompt_parts.append(f"2. Locate '{spec_name}' specification for that equipment")
        prompt_parts.append("3. Extract the value and unit that appear WITH the label")
    else:
        prompt_parts.append(f"1. Search for '{spec_name}' (or similar label) in DOCUMENT EVIDENCE above")
        prompt_parts.append("2. Extract the value and unit that appear WITH the label")
    
    prompt_parts.append("")
    prompt_parts.append("## Response Requirements")
    prompt_parts.append("- The value MUST exist literally in the DOCUMENT EVIDENCE above")
    prompt_parts.append("- If NOT found in DOCUMENT EVIDENCE, return empty strings")
    prompt_parts.append("- Korean text (한글) indicates an error - never use it")
    prompt_parts.append("")
    prompt_parts.append("Respond with JSON only:")
    prompt_parts.append('{"value": "extracted_value_or_empty", "unit": "extracted_unit_or_empty", "pos_desc": "label_as_written_or_empty"}')
    
    return "\n".join(prompt_parts)


def build_batch_extraction_prompt(
    specs: List[Dict[str, Any]],
    evidence: str,
) -> str:
    """
    Build batch extraction prompt for multiple specification items.
    
    Args:
        specs: [{"name": ..., "code": ..., "hints": ...}, ...]
        evidence: Shared document evidence
        
    Returns:
        Prompt string
    """
    prompt_parts = []
    
    prompt_parts.append("## Specification Items to Extract")
    for i, spec in enumerate(specs, 1):
        name = spec.get("name", "")
        code = spec.get("code", "")
        hints = spec.get("hints", {})
        
        line = f"{i}. {name}"
        if code:
            line += f" (code: {code})"
        if hints.get("gl_uom"):
            line += f" [expected unit: {hints['gl_uom']}]"
        prompt_parts.append(line)
    
    prompt_parts.append("\n## Document Evidence")
    prompt_parts.append(evidence)
    
    prompt_parts.append("\n## Task")
    prompt_parts.append("Extract values and units for each item from the evidence above.")
    prompt_parts.append("Respond with JSON array only:")
    prompt_parts.append('[{"name": "item name", "value": "extracted value", "unit": "unit", "pos_desc": "label as in POS"}, ...]')
    prompt_parts.append("For items not found, set value to empty string.")
    
    return "\n".join(prompt_parts)


SYSTEM_PROMPT_AUDIT = """You are an auditor verifying specification value extractions.

Verify that:
1. The extracted value actually exists in the evidence
2. The extracted value corresponds to the requested specification item
3. The unit is correctly extracted

RESPONSE FORMAT (JSON only):
{"valid": true/false, "reason": "verification result", "corrected_value": "corrected value if needed", "corrected_unit": "corrected unit if needed"}"""


def build_audit_prompt(
    spec_name: str,
    extracted_value: str,
    extracted_unit: str,
    evidence: str,
) -> str:
    """
    Build audit prompt for extraction verification.
    """
    prompt_parts = []
    
    prompt_parts.append("## Verification Target")
    prompt_parts.append(f"- Specification item: {spec_name}")
    prompt_parts.append(f"- Extracted value: {extracted_value}")
    prompt_parts.append(f"- Extracted unit: {extracted_unit}")
    
    prompt_parts.append("\n## Original Evidence")
    prompt_parts.append(evidence)
    
    prompt_parts.append("\n## Task")
    prompt_parts.append("Verify if the extraction is correct and respond with JSON only.")
    
    return "\n".join(prompt_parts)




# =============================================================================
# 사양값 DB 인덱스 (시리즈 선박 참조용)
# =============================================================================

class SpecDBIndex:
    """
    사양값 DB에서 시리즈 선박의 이전 추출값을 검색.
    동일 umgv_code를 가진 시리즈 선박의 값을 힌트로 활용.
    """

    def __init__(
        self,
        specdb_df: pd.DataFrame,
        logger: Optional[logging.Logger] = None,
    ):
        self.log = logger or logging.getLogger("specdb_index")
        self.df = specdb_df.fillna("") if specdb_df is not None else pd.DataFrame()
        
        # 인덱스 구축: (hull, umgv_code) -> {value, unit, ...}
        self._index: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._hull_series: Dict[str, List[str]] = {}  # hull -> series hulls
        
        if not self.df.empty:
            self._build_index()

    def _build_index(self):
        """
        인덱스 구축 (v42 개선)
        - umgv_code가 비어있는 경우 umgv_desc를 키로 사용
        - (hull, umgv_desc) 조합으로 인덱싱
        - v42: 컬럼명 매칭 강화, 빈 값 검사 개선
        """
        # 컬럼명 확인 (v42: 실제 사양값DB 컬럼에 맞게 수정)
        hull_col = None
        code_col = None       # umgv_code (비어있을 수 있음)
        desc_col = None       # umgv_desc (주 키로 사용)
        value_col = None
        unit_col = None
        pmg_col = None
        umg_col = None
        extwg_col = None      # extwg 컬럼
        extwg_desc_col = None # extwg_desc 또는 mat_attr_desc
        doknr_col = None
        
        # v42: 정확한 컬럼명 매칭 (소문자 비교)
        col_lower_map = {c.lower().strip(): c for c in self.df.columns}
        
        # hull 컬럼 (matnr 우선)
        for cn in ["matnr", "hull", "hull_no", "호선"]:
            if cn in col_lower_map:
                hull_col = col_lower_map[cn]
                break
        
        # umgv_code 컬럼
        for cn in ["umgv_code", "code", "사양코드"]:
            if cn in col_lower_map:
                code_col = col_lower_map[cn]
                break
        
        # umgv_desc 컬럼
        for cn in ["umgv_desc", "spec_desc", "사양항목명"]:
            if cn in col_lower_map:
                desc_col = col_lower_map[cn]
                break
        
        # umgv_value 컬럼 (umgv_value_edit 우선)
        for cn in ["umgv_value_edit", "umgv_value", "value", "사양값"]:
            if cn in col_lower_map:
                value_col = col_lower_map[cn]
                break
        
        # unit 컬럼
        for cn in ["umgv_uom", "unit", "단위", "pos_umgv_uom"]:
            if cn in col_lower_map:
                unit_col = col_lower_map[cn]
                break
        
        # pmg_desc 컬럼
        for cn in ["pmg_desc", "pmg_name"]:
            if cn in col_lower_map:
                pmg_col = col_lower_map[cn]
                break
        
        # umg_desc 컬럼
        for cn in ["umg_desc", "umg_name"]:
            if cn in col_lower_map:
                umg_col = col_lower_map[cn]
                break
        
        # extwg 컬럼
        if "extwg" in col_lower_map:
            extwg_col = col_lower_map["extwg"]
        
        # extwg_desc 또는 mat_attr_desc 컬럼
        for cn in ["extwg_desc", "mat_attr_desc"]:
            if cn in col_lower_map:
                extwg_desc_col = col_lower_map[cn]
                break
        
        # doknr 컬럼
        for cn in ["doknr", "pos", "pos_no"]:
            if cn in col_lower_map:
                doknr_col = col_lower_map[cn]
                break
        
        # v42: umgv_desc를 기본 키로 사용 (umgv_code는 대부분 비어있음)
        key_col = desc_col  # v42: 기본적으로 desc 사용
        use_desc_as_key = True
        
        # umgv_code가 충분히 있으면 code 사용
        if code_col and desc_col:
            try:
                non_empty_codes = self.df[code_col].astype(str).str.strip().ne("").sum()
                non_empty_descs = self.df[desc_col].astype(str).str.strip().ne("").sum()
                self.log.info("SpecDB column stats: umgv_code=%d, umgv_desc=%d non-empty", 
                             non_empty_codes, non_empty_descs)
                if non_empty_codes > non_empty_descs * 0.5:  # code가 desc의 50% 이상이면 code 사용
                    key_col = code_col
                    use_desc_as_key = False
            except Exception as e:
                self.log.warning("SpecDB column check error: %s", e)
        elif code_col and not desc_col:
            key_col = code_col
            use_desc_as_key = False
        
        if not hull_col:
            self.log.warning("SpecDB missing hull column. Found columns: %s", list(self.df.columns))
            return
        
        if not key_col:
            self.log.warning("SpecDB missing key column (umgv_desc or umgv_code). Found columns: %s", 
                           list(self.df.columns))
            return
        
        self.log.info("SpecDB index building: hull=%s, key=%s (use_desc=%s), value=%s, unit=%s", 
                      hull_col, key_col, use_desc_as_key, value_col, unit_col)
        
        # v41: 추가 인덱스 - (hull, extwg, umgv_desc) -> data
        self._extwg_index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        
        indexed_count = 0
        extwg_indexed = 0
        
        for _, row in self.df.iterrows():
            # v41: matnr에서 hull 추출
            raw_hull = str(row.get(hull_col, ""))
            hull = extract_hull_from_matnr(raw_hull)
            
            # 키 값 (umgv_desc 또는 umgv_code)
            key_value = norm_key(str(row.get(key_col, "")))
            
            if not hull or not key_value:
                continue
            
            # 데이터 구성
            data = {
                "value": str(row.get(value_col, "")).strip() if value_col else "",
                "unit": str(row.get(unit_col, "")).strip() if unit_col else "",
                "pmg_desc": str(row.get(pmg_col, "")).strip() if pmg_col else "",
                "umg_desc": str(row.get(umg_col, "")).strip() if umg_col else "",
                "umgv_desc": str(row.get(desc_col, "")).strip() if desc_col else "",
                "extwg": str(row.get(extwg_col, "")).strip() if extwg_col else "",
                "extwg_desc": str(row.get(extwg_desc_col, "")).strip() if extwg_desc_col else "",
                "doknr": str(row.get(doknr_col, "")).strip() if doknr_col else "",
                "hull": hull,
                "matnr": raw_hull,
            }
            
            # 메인 인덱스: (hull, key_value)
            main_key = (hull, key_value)
            if main_key not in self._index:
                self._index[main_key] = data
                indexed_count += 1
            
            # v41: extwg 인덱스 추가 - (hull, extwg, umgv_desc) -> data
            extwg = data.get("extwg", "")
            umgv_desc = data.get("umgv_desc", "")
            if extwg and umgv_desc:
                extwg_key = (hull, extwg, norm_key(umgv_desc))
                if extwg_key not in self._extwg_index:
                    self._extwg_index[extwg_key] = data
                    extwg_indexed += 1
        
        self.log.info("SpecDB index built: %d main entries, %d extwg entries", indexed_count, extwg_indexed)
        
        # v41: 사용 가능한 키 유형 저장
        self._use_desc_as_key = use_desc_as_key

    def set_series_mapping(self, series_map: Dict[str, List[str]]):
        """
        시리즈 선박 매핑 설정.
        
        Args:
            series_map: {hull: [series_hull1, series_hull2, ...]}
        """
        self._hull_series = series_map

    def lookup(
        self,
        hull: str,
        umgv_code: str = "",
        umgv_desc: str = "",
        include_series: bool = True,
    ) -> Optional[Dict[str, str]]:
        """
        사양값 조회 (v41 개선: umgv_desc 기반 조회 지원)
        
        Args:
            hull: 선박 번호
            umgv_code: 사양 코드 (비어있을 수 있음)
            umgv_desc: 사양 설명 (주 키로 사용)
            include_series: 시리즈 선박도 검색할지
            
        Returns:
            {"value": ..., "unit": ..., "pmg_desc": ..., ...} 또는 None
        """
        # v41: hull에서 숫자만 추출
        hull_clean = extract_hull_from_matnr(hull) if hull else ""
        
        # v41: umgv_desc를 우선 키로 사용
        key_value = norm_key(umgv_desc) if umgv_desc else norm_key(umgv_code)
        if not key_value:
            return None
        
        # 직접 조회
        key = (hull_clean, key_value)
        if key in self._index:
            return self._index[key]
        
        # 시리즈 선박 조회
        if include_series and hull_clean in self._hull_series:
            for series_hull in self._hull_series[hull_clean]:
                key = (series_hull, key_value)
                if key in self._index:
                    result = self._index[key].copy()
                    result["from_series"] = series_hull  # v41: 시리즈 출처 표시
                    return result
        
        return None
    
    def lookup_by_extwg(
        self,
        hull: str,
        extwg: str,
        umgv_desc: str,
    ) -> Optional[Dict[str, str]]:
        """
        v41: extwg + umgv_desc 조합으로 조회
        
        Args:
            hull: 선박 번호
            extwg: 자재속성그룹 코드 (예: YS57312)
            umgv_desc: 사양 설명
            
        Returns:
            매칭된 데이터 또는 None
        """
        hull_clean = extract_hull_from_matnr(hull) if hull else ""
        desc_key = norm_key(umgv_desc)
        
        if not hull_clean or not extwg or not desc_key:
            return None
        
        key = (hull_clean, extwg, desc_key)
        if key in self._extwg_index:
            return self._extwg_index[key]
        
        # 시리즈 선박 조회
        if hull_clean in self._hull_series:
            for series_hull in self._hull_series[hull_clean]:
                key = (series_hull, extwg, desc_key)
                if key in self._extwg_index:
                    result = self._extwg_index[key].copy()
                    result["from_series"] = series_hull
                    return result
        
        return None
    
    def lookup_by_desc(
        self,
        pmg_desc: str = "",
        umg_desc: str = "",
        umgv_desc: str = "",
        exclude_hull: str = "",
    ) -> List[Dict[str, str]]:
        """
        v41: 설명 기반 조회 (유사 사양값 검색용)
        
        Args:
            pmg_desc: PMG 설명
            umg_desc: UMG 설명
            umgv_desc: UMGV 설명
            exclude_hull: 제외할 호선
            
        Returns:
            매칭된 레코드 리스트
        """
        results = []
        
        pmg_lower = pmg_desc.lower() if pmg_desc else ""
        umg_lower = umg_desc.lower() if umg_desc else ""
        umgv_lower = umgv_desc.lower() if umgv_desc else ""
        exclude_hull_clean = extract_hull_from_matnr(exclude_hull) if exclude_hull else ""
        
        for (hull, code), data in self._index.items():
            # 같은 호선 제외
            if exclude_hull_clean and hull == exclude_hull_clean:
                continue
            
            # 설명 매칭
            match_score = 0
            if pmg_lower and pmg_lower in data.get("pmg_desc", "").lower():
                match_score += 1
            if umg_lower and umg_lower in data.get("umg_desc", "").lower():
                match_score += 1
            if umgv_lower and umgv_lower in data.get("umgv_desc", "").lower():
                match_score += 2  # umgv_desc 매칭에 더 높은 가중치
            
            if match_score >= 2:  # 최소 2점 이상
                result = data.copy()
                result["match_score"] = match_score
                results.append(result)
        
        # 점수 높은 순 정렬
        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return results[:5]  # 상위 5개


# =============================================================================
# 배치 처리 함수
# =============================================================================

def process_single_spec(
    spec: Dict[str, Any],
    doc_chunks: List[DocChunk],
    doc_blob: str,
    glossary_index: Optional[GlossaryIndex],
    specdb_index: Optional[SpecDBIndex],
    llm_pool: LLMClientPool,
    config: Config,
    logger: logging.Logger,
    vector_search: Optional[VectorSimilaritySearch] = None,
) -> Dict[str, Any]:
    """
    단일 사양항목 처리.
    
    1) 장비 번호 파싱 (WASHING MACHINE 1, 2, 3...)
    2) Rule-based 추출 시도
    3) 실패 시 LLM 투표 (벡터 유사도 기반 few-shot 힌트 포함)
    4) Local audit
    5) 결과 반환 (few-shot 힌트 정보 포함)
    
    출력 필수 컬럼:
    - section_num, table_text, value_format
    - pos_chunk, pos_mat_attr_desc, pos_umgv_desc
    - pos_umgv_value, umgv_value_edit, pos_umgv_uom
    - evidence_fb
    - _fewshot_hints, _glossary_hints, _specdb_hints (디버깅용)
    """
    spec_name = spec.get(COL_UMGV_DESC, "") or spec.get(COL_SPEC_NAME, "")
    if not isinstance(spec_name, str):
        spec_name = str(spec_name) if spec_name else ""
    
    spec_code = spec.get("umgv_code", "")
    if not isinstance(spec_code, str):
        spec_code = str(spec_code) if spec_code else ""
    
    expected_unit = spec.get("__gl_umgv_uom", "") or spec.get("__gl_pos_umgv_uom", "") or spec.get("umgv_uom", "")
    if not isinstance(expected_unit, str):
        expected_unit = str(expected_unit) if expected_unit and str(expected_unit) != "nan" else ""
    
    # ========================================
    # 핵심 식별자 추출 (chunk 선택에 사용)
    # ========================================
    # mat_attr_desc: 자재속성그룹명 (예: "ME FO SUPPLY MODULE", "COOLING FW PUMP FOR AIR-CON. SYSTEM")
    # 이 값이 POS 문서에서 해당 장비를 식별하는 핵심 키워드임
    mat_attr_desc = spec.get("mat_attr_desc", "") or spec.get("__mat_attr_desc", "")
    if not isinstance(mat_attr_desc, str):
        mat_attr_desc = str(mat_attr_desc) if mat_attr_desc else ""
    
    # umg_desc: UMG 설명 (예: "CENTRIFUGAL PUMPS", "FO SUPPLY MODULE")
    umg_desc = spec.get("umg_desc", "") or spec.get("__umg_desc", "")
    if not isinstance(umg_desc, str):
        umg_desc = str(umg_desc) if umg_desc else ""
    
    # pmg_desc: PMG 설명 (예: "CENTRIFUGAL PUMPS", "TUBULAR TYPE HEAT EXCHANGER")
    pmg_desc = spec.get("pmg_desc", "") or spec.get("__pmg_desc", "")
    if not isinstance(pmg_desc, str):
        pmg_desc = str(pmg_desc) if pmg_desc else ""
    
    # matnr: 자재번호 (호선 + 자재코드, 예: "2606AYS57111")
    matnr = spec.get("matnr", "")
    if not isinstance(matnr, str):
        matnr = str(matnr) if matnr else ""
    
    # 장비 번호 파싱 (WASHING MACHINE 1 -> base, 1)
    base_spec_name, equipment_number = parse_equipment_number(spec_name)
    
    # 결과 딕셔너리 초기화 (규칙 스키마 준수)
    result = create_empty_extraction_result()
    result[COL_CONFIDENCE] = 0.0
    result[COL_METHOD] = ""
    result[COL_AUDIT] = ""
    
    # 힌트 저장용 (디버깅/검증)
    result[COL_FEWSHOT_HINTS] = ""
    result[COL_GLOSSARY_HINTS] = ""
    result[COL_SPECDB_HINTS] = ""
    result[COL_EVIDENCE_STRATEGY] = ""
    
    # ========================================
    # 라벨 변형 목록 (chunk 선택에 사용)
    # 우선순위: mat_attr_desc > umg_desc > spec_name > base_spec_name
    # ========================================
    label_variants = []
    
    # 1순위: mat_attr_desc (자재속성그룹명) - 가장 구체적인 장비 식별자
    if mat_attr_desc:
        label_variants.append(mat_attr_desc)
        # mat_attr_desc에서 핵심 단어 추출 (예: "ME FO SUPPLY MODULE" -> ["ME", "FO", "SUPPLY", "MODULE"])
        extwg_words = re.findall(r'[A-Za-z]+', mat_attr_desc)
        for word in extwg_words:
            if len(word) >= 2 and word.upper() not in ["FOR", "THE", "AND", "WITH", "OF", "IN"]:
                if word not in label_variants:
                    label_variants.append(word)
    
    # 2순위: umg_desc (UMG 설명)
    if umg_desc and umg_desc not in label_variants:
        label_variants.append(umg_desc)
    
    # 3순위: spec_name (사양항목명)
    if spec_name and spec_name not in label_variants:
        label_variants.append(spec_name)
    
    if base_spec_name and base_spec_name != spec_name and base_spec_name not in label_variants:
        label_variants.append(base_spec_name)
    
    # 4순위: 언더스코어로 분리된 키워드
    if "_" in spec_name:
        parts = spec_name.split("_")
        for p in parts:
            p_clean = re.sub(r'\s*\d+\s*$', '', p).strip()  # 끝의 숫자 제거
            p_clean = re.sub(r'\([^)]*\)', '', p_clean).strip()  # 괄호 내용 제거
            if p_clean and len(p_clean) > 2 and p_clean not in label_variants:
                label_variants.append(p_clean)
    
    gl_desc = spec.get("__gl_pos_umgv_desc", "")
    if gl_desc and gl_desc not in label_variants:
        label_variants.append(gl_desc)
    
    # 용어집 힌트 수집
    glossary_hints_data = {}
    if glossary_index:
        gl_match = glossary_index.lookup(umgv_desc=spec_name, umgv_code=spec_code)
        if gl_match:
            glossary_hints_data = {
                "umgv_desc": gl_match.get("umgv_desc", ""),
                "pos_umgv_desc": gl_match.get("pos_umgv_desc", ""),
                "pos_umgv_value": gl_match.get("pos_umgv_value", ""),
                "pos_umgv_uom": gl_match.get("pos_umgv_uom", ""),
                "section_num": gl_match.get("section_num", ""),
                "table_text": gl_match.get("table_text", ""),
            }
    result[COL_GLOSSARY_HINTS] = json.dumps(glossary_hints_data, ensure_ascii=False) if glossary_hints_data else ""
    
    # SpecDB 힌트 수집
    specdb_hints_data = {}
    if specdb_index:
        hull = spec.get("__hull", "")
        db_result = specdb_index.lookup(hull, spec_code, include_series=True)
        if db_result:
            specdb_hints_data = {
                "hull": db_result.get("hull", ""),
                "value": db_result.get("value", ""),
                "uom": db_result.get("uom", ""),
            }
    result[COL_SPECDB_HINTS] = json.dumps(specdb_hints_data, ensure_ascii=False) if specdb_hints_data else ""
    
    # 벡터 유사도 기반 few-shot 힌트 수집 (v41: mat_attr_desc 추가)
    fewshot_hints = []
    if vector_search and config.enable_vector_search:
        try:
            fewshot_hints = vector_search.get_few_shot_hints(
                query_hull=spec.get("__hull", ""),
                query_pmg=spec.get("pmg_desc", ""),
                query_umg=spec.get("umg_desc", ""),
                query_umgv=base_spec_name or spec_name,
                query_uom=expected_unit,
                query_mat_attr_desc=mat_attr_desc,  # v41: 자재속성그룹명 추가
                top_k=3,
            )
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
    result[COL_FEWSHOT_HINTS] = json.dumps(fewshot_hints, ensure_ascii=False) if fewshot_hints else ""
    
    # v41: 참조 정보 로깅
    try:
        ref_logger = get_reference_logger_v41()
        ref_logger.add_reference(
            extwg=spec.get("extwg", ""),
            matnr=spec.get("matnr", ""),
            pmg_desc=spec.get("pmg_desc", ""),
            umg_desc=spec.get("umg_desc", ""),
            umgv_desc=spec_name,
            # 사양값DB 참조 정보
            specdb_pos=specdb_hints_data.get("pos", "") if specdb_hints_data else "",
            specdb_matnr=specdb_hints_data.get("matnr", "") if specdb_hints_data else "",
            specdb_hull=specdb_hints_data.get("hull", "") if specdb_hints_data else "",
            specdb_value=specdb_hints_data.get("value", "") if specdb_hints_data else "",
            specdb_uom=specdb_hints_data.get("uom", "") if specdb_hints_data else "",
            # 용어집 참조 정보
            glossary_pos=glossary_hints_data.get("pos", "") if glossary_hints_data else "",
            glossary_matnr=glossary_hints_data.get("matnr", "") if glossary_hints_data else "",
            glossary_section=glossary_hints_data.get("section_num", "") if glossary_hints_data else "",
            glossary_table_text=glossary_hints_data.get("table_text", "") if glossary_hints_data else "",
            glossary_format=glossary_hints_data.get("value_format", "") if glossary_hints_data else "",
            glossary_std_uom=glossary_hints_data.get("pos_umgv_uom", "") if glossary_hints_data else "",
            # 벡터 유사도 검색 결과
            vector_top1_hull=fewshot_hints[0].get("hull", "") if fewshot_hints else "",
            vector_top1_score=float(fewshot_hints[0].get("score", 0)) if fewshot_hints else 0.0,
            vector_top1_value=fewshot_hints[0].get("value", "") if fewshot_hints else "",
        )
    except Exception as e:
        logger.debug("Reference logging failed: %s", e)
    
    # extwg 추출 (spec에서 가져오기)
    extwg = spec.get("extwg", "") or ""
    if not isinstance(extwg, str):
        extwg = str(extwg) if extwg else ""
    
    # 용어집에서 pos_umgv_desc 키워드 가져오기
    glossary_keywords = []
    if glossary_hints_data.get("pos_umgv_desc"):
        glossary_keywords.append(glossary_hints_data["pos_umgv_desc"])
    
    # 1) Rule-based 추출 (extwg 기반 우선!)
    if config.rule_enable:
        # 1-1) extwg(Part No.) 기반 테이블 추출 (최우선)
        if extwg:
            extwg_result = rule_extract_by_extwg(
                table_chunks=[c for c in doc_chunks if c.ctype in ("table_md", "table_kv")],
                extwg=extwg,
                umgv_desc=spec_name,
                mat_attr_desc=mat_attr_desc,
                glossary_keywords=glossary_keywords,
            )
            
            if extwg_result:
                v, u, conf, method = extwg_result
                if conf >= config.rule_conf_threshold:
                    # v41: 한글 오염 검증
                    if check_korean_contamination_v41(v):
                        logger.warning(
                            "Korean contamination detected in rule extraction: '%s' for %s. Skipping.",
                            v, spec_name
                        )
                        # 한글 오염된 값은 사용하지 않고 다음 추출 방법으로 진행
                    else:
                        # v41: 값 정제
                        v = clean_extracted_value_v41(v, spec_name)
                        
                        result[COL_POS_UMGV_VALUE] = v
                        result[COL_POS_UMGV_UOM] = u
                        result[COL_UMGV_VALUE_EDIT] = v
                        result[COL_TABLE_TEXT] = "Y"
                        result[COL_VALUE_FORMAT] = determine_value_format(v)
                        result[COL_POS_UMGV_DESC] = gl_desc or spec_name
                        result[COL_POS_CHUNK] = f"Extracted by extwg={extwg}: {v} {u}"
                        result[COL_CONFIDENCE] = conf
                        result[COL_METHOD] = method
                        
                        # v41: 검증 노트 추가
                        result[COL_VALIDATION_NOTE] = "RULE_EXTWG_OK"
                        
                        if expected_unit and u and expected_unit.lower() != u.lower():
                            if are_units_compatible(expected_unit, u):
                                get_unit_manager().discover_from_extraction(expected_unit, u)
                        
                        return result
        
        # 1-2) 기존 label 기반 테이블 추출
        rule_result = rule_extract_from_tables(doc_chunks, label_variants, expected_unit)
        
        # 장비 번호가 있으면 테이블에서 해당 번호의 값 찾기
        if rule_result and equipment_number:
            # 매칭된 테이블에서 장비 번호에 맞는 값 추출 시도
            for chunk in doc_chunks:
                if chunk.ctype in ("table_md", "table_kv"):
                    equip_result = find_equipment_context_in_table(
                        chunk.text, base_spec_name, equipment_number, expected_unit
                    )
                    if equip_result:
                        v, u, context = equip_result
                        result[COL_POS_UMGV_VALUE] = v
                        result[COL_POS_UMGV_UOM] = u
                        result[COL_UMGV_VALUE_EDIT] = v
                        result[COL_TABLE_TEXT] = "Y"
                        result[COL_VALUE_FORMAT] = determine_value_format(v)
                        result[COL_POS_UMGV_DESC] = gl_desc or spec_name
                        result[COL_POS_CHUNK] = f"Equipment #{equipment_number}: {context}"
                        result[COL_CONFIDENCE] = 0.85
                        result[COL_METHOD] = f"RULE_EQUIPMENT({equipment_number})"
                        return result
        
        if rule_result:
            v, u, conf, method = rule_result
            if conf >= config.rule_conf_threshold:
                # v42: 테이블에서 값은 찾았지만 단위가 없고, expected_unit이 있으면
                # 텍스트 패턴에서 단위와 함께 추출할 수 있는지 확인
                if not u and expected_unit:
                    spec_aliases = build_spec_aliases(spec_name, glossary_hints_data)
                    text_pattern_result = extract_value_from_text_pattern(
                        text=doc_blob,
                        spec_name=spec_name,
                        known_aliases=spec_aliases,
                    )
                    if text_pattern_result:
                        txt_v = text_pattern_result.get("value", "")
                        txt_u = text_pattern_result.get("unit", "")
                        # 텍스트에서 단위까지 추출되었고, expected_unit과 호환되면 텍스트 결과 사용
                        if txt_v and txt_u and are_units_compatible(expected_unit, txt_u):
                            v = txt_v
                            u = txt_u
                            conf = 0.82  # 텍스트 패턴 신뢰도
                            method = f"RULE_TEXT_PATTERN_OVERRIDE({text_pattern_result.get('pattern', '')})"
                            logger.info("v42: Text pattern with unit overrides table (unit=%s)", txt_u)
                
                result[COL_POS_UMGV_VALUE] = v
                result[COL_POS_UMGV_UOM] = u
                result[COL_UMGV_VALUE_EDIT] = v
                result[COL_TABLE_TEXT] = "Y"
                result[COL_VALUE_FORMAT] = determine_value_format(v)
                result[COL_POS_UMGV_DESC] = gl_desc or spec_name
                result[COL_POS_CHUNK] = f"Rule extracted from table: {v} {u}"
                result[COL_CONFIDENCE] = conf
                result[COL_METHOD] = method
                
                if expected_unit and u and expected_unit.lower() != u.lower():
                    if are_units_compatible(expected_unit, u):
                        get_unit_manager().discover_from_extraction(expected_unit, u)
                
                return result
        
        # 본문에서 추출 시도
        rule_result = rule_extract_from_text(doc_blob, label_variants, expected_unit)
        if rule_result:
            v, u, conf, method = rule_result
            if conf >= config.rule_conf_threshold:
                result[COL_POS_UMGV_VALUE] = v
                result[COL_POS_UMGV_UOM] = u
                result[COL_UMGV_VALUE_EDIT] = v
                result[COL_TABLE_TEXT] = "N"
                result[COL_VALUE_FORMAT] = determine_value_format(v)
                result[COL_POS_UMGV_DESC] = gl_desc or spec_name
                result[COL_POS_CHUNK] = f"Rule extracted from text: {v} {u}"
                result[COL_CONFIDENCE] = conf
                result[COL_METHOD] = method
                
                if expected_unit and u and expected_unit.lower() != u.lower():
                    if are_units_compatible(expected_unit, u):
                        get_unit_manager().discover_from_extraction(expected_unit, u)
                
                return result
    
    # v41: 텍스트 패턴 매칭 시도 (테이블 추출 실패 시)
    # 예: "NPSH required" of 4.5 m, "Casing : Cast iron" 등
    if config.rule_enable:
        # 별칭 구축
        spec_aliases = build_spec_aliases(spec_name, glossary_hints_data)
        
        # 문서 전체에서 텍스트 패턴 매칭
        text_pattern_result = extract_value_from_text_pattern(
            text=doc_blob,
            spec_name=spec_name,
            known_aliases=spec_aliases,
        )
        
        if text_pattern_result:
            v = text_pattern_result.get("value", "")
            u = text_pattern_result.get("unit", "")
            matched_desc = text_pattern_result.get("matched_desc", "")
            pattern = text_pattern_result.get("pattern", "")
            
            if v and not check_korean_contamination_v41(v):
                v = clean_extracted_value_v41(v, spec_name)
                
                # 해당 값이 포함된 chunk 찾기
                matched_chunk = ""
                for chunk in doc_chunks:
                    if v in chunk.text or (matched_desc and matched_desc.lower() in chunk.text.lower()):
                        matched_chunk = chunk.text[:1500]
                        break
                
                if not matched_chunk:
                    # doc_blob에서 해당 부분 추출
                    search_pattern = rf".{{0,200}}{re.escape(v)}.{{0,200}}"
                    match = re.search(search_pattern, doc_blob)
                    if match:
                        matched_chunk = match.group(0)
                
                result[COL_POS_UMGV_VALUE] = v
                result[COL_POS_UMGV_UOM] = u
                result[COL_UMGV_VALUE_EDIT] = v
                result[COL_TABLE_TEXT] = "N"
                result[COL_VALUE_FORMAT] = determine_value_format(v)
                result[COL_POS_UMGV_DESC] = matched_desc or gl_desc or spec_name
                result[COL_POS_CHUNK] = matched_chunk or f"Text pattern: {matched_desc} = {v} {u}"
                result[COL_CONFIDENCE] = 0.78  # 텍스트 패턴 매칭은 약간 낮은 신뢰도
                result[COL_METHOD] = f"RULE_TEXT_PATTERN({pattern})"
                result[COL_VALIDATION_NOTE] = f"TextPattern:{matched_desc}"
                
                return result
    
    # 2) Evidence 선택 (장비 번호 전달)
    evidence_locator = EvidenceLocator(
        glossary_index=glossary_index,
        logger=logger,
    )
    
    hints = {
        "gl_desc": gl_desc,
        "gl_uom": expected_unit,
        "gl_value_format": spec.get("__gl_value_format", ""),
        "gl_chunk": spec.get("__gl_pos_chunk", ""),
    }
    
    # SpecDB 힌트 추가
    if specdb_hints_data.get("value"):
        hints["similar_value"] = specdb_hints_data["value"]
    
    evidence, evidence_meta = evidence_locator.select_evidence_with_meta(
        spec_name=spec_name,
        label_variants=label_variants,
        doc_chunks=doc_chunks,
        doc_blob=doc_blob,
        hints=hints,
        max_chars=config.max_evidence_chars,
        equipment_number=equipment_number,
        mat_attr_desc=mat_attr_desc,  # 자재속성그룹명 전달
        umg_desc=umg_desc,      # UMG 설명 전달
    )
    
    result[COL_EVIDENCE_STRATEGY] = evidence_meta.get("strategy", "")
    keyword_found = evidence_meta.get("keyword_found", False)
    
    if not evidence:
        result[COL_METHOD] = "NO_EVIDENCE"
        return result
    
    # 핵심 키워드가 evidence에 없는 경우 경고
    if not keyword_found:
        logger.warning(
            "Keyword not found in evidence for '%s'. Strategy: %s. "
            "LLM may hallucinate values from hints.",
            spec_name, evidence_meta.get("strategy", "")
        )
    
    # evidence 메타정보 저장
    result[COL_SECTION_NUM] = evidence_meta.get("section_num", "")
    result[COL_TABLE_TEXT] = evidence_meta.get("table_text", "N")
    
    # 3) LLM 추출 (few-shot 힌트 포함)
    prompt = build_extraction_prompt(
        spec_name=spec_name,
        spec_code=spec_code,
        evidence=evidence,
        hints=hints,
        few_shot_examples=fewshot_hints if fewshot_hints else None,
    )
    
    responses = llm_pool.generate_with_voting(
        prompt=prompt,
        system=SYSTEM_PROMPT_EXTRACTION,
        k=config.vote_k,
        temperature=0.0,
    )
    
    voted_result, vote_conf, vote_detail = majority_vote(responses, min_agreement=2)
    
    extracted_value = voted_result.get("value", "")
    extracted_unit = voted_result.get("unit", "")
    
    # v41: 추출된 값 검증 강화
    validation_note = ""
    
    if extracted_value:
        # v41: 한글 오염 검증 (최우선)
        if check_korean_contamination_v41(extracted_value):
            logger.warning(
                "Korean contamination detected in LLM extraction: '%s' for %s. Clearing result.",
                extracted_value, spec_name
            )
            extracted_value = ""
            extracted_unit = ""
            vote_detail = "KOREAN_CONTAMINATION"
            vote_conf = 0.0
            validation_note = "한글오염_용어집혼입"
        else:
            # v41: 값 정제
            extracted_value = clean_extracted_value_v41(extracted_value, spec_name)
            
            # v41: evidence 내 존재 검증
            is_valid, note, conf_adj = validate_value_in_evidence_v41(
                extracted_value, evidence, spec_name
            )
            validation_note = note
            
            if not is_valid:
                vote_conf *= conf_adj
                if conf_adj < 0.5:
                    logger.warning(
                        "Value validation failed for '%s': %s. Clearing result.",
                        spec_name, note
                    )
                    extracted_value = ""
                    extracted_unit = ""
                    vote_detail = f"VALIDATION_FAILED:{note}"
                    vote_conf = 0.0
    
    # 기존 검증 (keyword + evidence 존재 확인)
    value_in_evidence = False
    if extracted_value:
        value_normalized = extracted_value.replace(",", "").strip()
        if value_normalized in evidence or extracted_value in evidence:
            value_in_evidence = True
    
    # 키워드도 없고 값도 evidence에 없으면 환각 의심
    if not keyword_found and extracted_value and not value_in_evidence:
        logger.warning(
            "Suspected hallucination for '%s': extracted value '%s' not found in evidence. "
            "Clearing result.",
            spec_name, extracted_value
        )
        extracted_value = ""
        extracted_unit = ""
        vote_detail = "HALLUCINATION_SUSPECTED"
        vote_conf = 0.0
        validation_note = "환각의심_evidence미발견"
    
    # v41: 빈값일 때 근거 chunk 저장 (값이 비어있는 이유를 설명하는 chunk 찾기)
    if not extracted_value:
        # 별칭 구축
        spec_aliases = build_spec_aliases(spec_name, glossary_hints_data)
        
        # 빈값의 근거가 되는 chunk 찾기 (예: "Motor output (rating) : kW" 형태로만 있는 경우)
        evidence_for_empty = find_evidence_chunk_for_empty_value(
            doc_chunks=doc_chunks,
            spec_name=spec_name,
            known_aliases=spec_aliases,
        )
        
        if evidence_for_empty:
            # 근거 chunk가 발견됨 - 값은 비어있지만 그 이유를 알 수 있음
            result[COL_POS_CHUNK] = f"[값 없음 근거] {evidence_for_empty[:1500]}"
            validation_note = "빈값_근거발견"
        elif evidence:
            # 기존 evidence 사용
            result[COL_POS_CHUNK] = evidence[:2000]
        else:
            result[COL_POS_CHUNK] = ""
    else:
        result[COL_POS_CHUNK] = evidence[:2000]
    
    result[COL_POS_UMGV_VALUE] = extracted_value
    result[COL_POS_UMGV_UOM] = extracted_unit
    result[COL_UMGV_VALUE_EDIT] = extracted_value
    result[COL_VALUE_FORMAT] = determine_value_format(extracted_value)
    result[COL_POS_UMGV_DESC] = voted_result.get("pos_desc", "") or gl_desc or spec_name
    result[COL_CONFIDENCE] = vote_conf
    result[COL_VALIDATION_NOTE] = validation_note  # v41: 검증 노트 추가
    
    # method에 keyword_found 정보 추가
    method_suffix = "" if keyword_found else ":NO_KW"
    result[COL_METHOD] = f"LLM:{vote_detail}{method_suffix}"
    
    # 새 동의어 발견 시 등록
    if expected_unit and extracted_unit and expected_unit.lower() != extracted_unit.lower():
        if are_units_compatible(expected_unit, extracted_unit):
            get_unit_manager().discover_from_extraction(expected_unit, extracted_unit)
    
    # 4) Local Audit
    if extracted_value:
        audit_pass = local_audit(evidence, extracted_value, extracted_unit)
        result[COL_AUDIT] = "PASS" if audit_pass else "FAIL"
        
        if not audit_pass:
            result[COL_CONFIDENCE] *= 0.7
    
    return result


def process_batch(
    specs: List[Dict[str, Any]],
    doc_chunks: List[DocChunk],
    doc_blob: str,
    glossary_index: Optional[GlossaryIndex],
    specdb_index: Optional[SpecDBIndex],
    llm_pool: LLMClientPool,
    config: Config,
    logger: logging.Logger,
    vector_search: Optional[VectorSimilaritySearch] = None,
) -> List[Dict[str, Any]]:
    """
    배치 사양항목 처리.
    """
    results = []
    
    for spec in specs:
        try:
            result = process_single_spec(
                spec=spec,
                doc_chunks=doc_chunks,
                doc_blob=doc_blob,
                glossary_index=glossary_index,
                specdb_index=specdb_index,
                llm_pool=llm_pool,
                config=config,
                logger=logger,
                vector_search=vector_search,
            )
            # 원본 정보 병합
            merged = {**spec, **result}
            results.append(merged)
        except Exception as e:
            logger.error("Error processing spec %s: %s", spec.get(COL_SPEC_NAME, ""), e)
            error_result = {
                **spec,
                COL_SPEC_VALUE: "",
                COL_SPEC_UNIT: "",
                COL_EVIDENCE: "",
                COL_CONFIDENCE: 0.0,
                COL_METHOD: f"ERROR:{str(e)[:100]}",
                COL_AUDIT: "",
            }
            results.append(error_result)
    
    return results


# =============================================================================
# POS 그룹 워커 (멀티프로세싱용)
# =============================================================================

def worker_process_pos_group(
    args: Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    단일 POS 그룹 처리 워커.
    
    Args:
        args: (hull, pos, specs_list, worker_config)
        
    Returns:
        처리 결과 리스트
    """
    hull, pos, specs, worker_config = args
    
    # 설정 복원
    config = Config(**worker_config.get("config", {}))
    
    # 로거 설정
    logger = logging.getLogger(f"worker_{hull}_{pos}")
    logger.setLevel(logging.INFO)
    
    # HTML 파일 찾기
    file_finder = FileFinder(
        base_dirs=config.html_dirs,
        allow_cross_hull_fallback=False,
        logger=logger,
    )
    
    html_path = file_finder.find_file(hull, pos)
    if not html_path:
        # 파일 없음 - 모든 사양에 대해 FILE_NOT_FOUND 반환
        results = []
        for spec in specs:
            results.append({
                **spec,
                COL_SPEC_VALUE: "",
                COL_SPEC_UNIT: "",
                COL_EVIDENCE: "",
                COL_CONFIDENCE: 0.0,
                COL_METHOD: "FILE_NOT_FOUND",
                COL_AUDIT: "",
            })
        return results
    
    # HTML 파싱
    reader = POSHTMLReader(html_path, logger=logger)
    doc_chunks = reader.get_chunks()
    doc_blob = reader.get_text_blob()
    
    if not doc_chunks and not doc_blob:
        results = []
        for spec in specs:
            results.append({
                **spec,
                COL_SPEC_VALUE: "",
                COL_SPEC_UNIT: "",
                COL_EVIDENCE: "",
                COL_CONFIDENCE: 0.0,
                COL_METHOD: "EMPTY_DOCUMENT",
                COL_AUDIT: "",
            })
        return results
    
    # Glossary 인덱스 (워커별 생성 또는 공유)
    glossary_index = None
    if worker_config.get("glossary_df") is not None:
        glossary_df = pd.DataFrame(worker_config["glossary_df"])
        glossary_index = GlossaryIndex(glossary_df)
    
    # SpecDB 인덱스
    specdb_index = None
    if worker_config.get("specdb_df") is not None:
        specdb_df = pd.DataFrame(worker_config["specdb_df"])
        specdb_index = SpecDBIndex(specdb_df)
    
    # LLM Pool (워커별 생성)
    llm_pool = LLMClientPool(
        host=config.ollama_host,
        ports=config.ollama_ports,
        model=config.model_name,
        timeout=config.llm_timeout,
        logger=logger,
    )
    llm_pool.check_health_all()
    
    # 배치 처리
    results = process_batch(
        specs=specs,
        doc_chunks=doc_chunks,
        doc_blob=doc_blob,
        glossary_index=glossary_index,
        specdb_index=specdb_index,
        llm_pool=llm_pool,
        config=config,
        logger=logger,
    )
    
    return results




# =============================================================================
# 메인 프로세서 클래스
# =============================================================================

class POSSpecProcessor:
    """
    POS 사양값 추출 메인 프로세서.
    
    모드:
    - file: TXT 입력 → CSV 출력
    - db: PostgreSQL 입력 → JSON → PostgreSQL 출력
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        
        # 데이터 홀더
        self.spec_df: Optional[pd.DataFrame] = None
        self.glossary_df: Optional[pd.DataFrame] = None
        self.specdb_df: Optional[pd.DataFrame] = None
        self.results: List[Dict[str, Any]] = []
        
        # 인덱스
        self.glossary_index: Optional[GlossaryIndex] = None
        self.specdb_index: Optional[SpecDBIndex] = None
        
        # LLM Pool
        self.llm_pool: Optional[LLMClientPool] = None
        
        # 체크포인트
        self.checkpoint_file = os.path.join(config.log_dir, "checkpoint.pkl")
        self.done_keys: Set[Tuple[str, str, str]] = set()  # (hull, pos, spec_code)

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logger = logging.getLogger("POSSpecProcessor")
        logger.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        logger.handlers = []
        
        # 파일 핸들러
        fh = logging.FileHandler(
            os.path.join(self.config.log_dir, f"extraction_{datetime.now():%Y%m%d_%H%M%S}.log"),
            encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(fh)
        
        # 콘솔 핸들러
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)
        
        return logger

    def load_checkpoint(self) -> bool:
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "rb") as f:
                    data = pickle.load(f)
                self.done_keys = data.get("done_keys", set())
                self.results = data.get("results", [])
                self.logger.info("Checkpoint loaded: %d done, %d results", 
                               len(self.done_keys), len(self.results))
                return True
            except Exception as e:
                self.logger.warning("Checkpoint load failed: %s", e)
        return False

    def save_checkpoint(self):
        """체크포인트 저장"""
        try:
            data = {
                "done_keys": self.done_keys,
                "results": self.results,
                "timestamp": datetime.now().isoformat(),
            }
            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning("Checkpoint save failed: %s", e)

    def load_data(self):
        """데이터 로드 (모드에 따라)"""
        self.logger.info("Loading data (mode=%s)...", self.config.mode)
        
        if self.config.mode == "file":
            self._load_data_from_files()
        else:
            self._load_data_from_db()
        
        # Glossary 로드 (공통)
        if os.path.exists(self.config.glossary_path):
            gl_loader = TxtTableLoader(self.config.glossary_path, self.logger)
            self.glossary_df = gl_loader.load()
            self.logger.info("Glossary loaded: %d rows", len(self.glossary_df))
            self.glossary_index = GlossaryIndex(self.glossary_df)
        
        # SpecDB 로드 (공통) - 깨진 파일 복구 로직 사용
        if os.path.exists(self.config.specdb_path):
            sd_loader = TxtTableLoader(self.logger)
            self.specdb_df = sd_loader.load_specdb_repaired(self.config.specdb_path)
            self.logger.info("SpecDB loaded: %d rows", len(self.specdb_df))
            self.specdb_index = SpecDBIndex(self.specdb_df, logger=self.logger)

    def _load_data_from_files(self):
        """파일 모드: TXT에서 사양 목록 로드"""
        loader = TxtTableLoader(self.config.input_path, self.logger)
        self.spec_df = loader.load()
        
        if self.spec_df is None or self.spec_df.empty:
            raise ValueError(f"Failed to load spec data from {self.config.input_path}")
        
        # 컬럼명 표준화
        col_map = {
            "사양항목명": COL_SPEC_NAME,
            "사양값": COL_SPEC_VALUE,
            "단위": COL_SPEC_UNIT,
            "matnr": "matnr",
            "사양코드": "umgv_code",
        }
        self.spec_df.rename(columns={k: v for k, v in col_map.items() if k in self.spec_df.columns}, 
                           inplace=True)
        
        self.logger.info("Spec data loaded: %d rows", len(self.spec_df))

    def _load_data_from_db(self):
        """DB 모드: PostgreSQL에서 사양 목록 로드"""
        pg_loader = PostgresLoader(
            host=self.config.db_host,
            port=self.config.db_port,
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            logger=self.logger,
        )
        
        self.spec_df = pg_loader.load_specs(
            table_name=self.config.db_input_table,
            conditions=self.config.db_input_conditions,
        )
        
        if self.spec_df is None or self.spec_df.empty:
            raise ValueError("Failed to load spec data from database")
        
        self.logger.info("Spec data loaded from DB: %d rows", len(self.spec_df))

    def prepare_data(self):
        """데이터 전처리"""
        self.logger.info("Preparing data...")
        
        # hull, pos 컬럼 추가
        # hull은 matnr에서, pos는 doknr 또는 matnr에서 추출
        if "matnr" in self.spec_df.columns:
            self.spec_df["__hull"] = self.spec_df["matnr"].apply(extract_hull_from_matnr)
        else:
            self.spec_df["__hull"] = ""
        
        # POS 추출: doknr 우선, 없으면 matnr에서 시도
        if "doknr" in self.spec_df.columns:
            self.spec_df["__pos"] = self.spec_df["doknr"].apply(extract_pos_from_matnr)
            self.logger.info("POS extracted from 'doknr' column")
        elif "matnr" in self.spec_df.columns:
            self.spec_df["__pos"] = self.spec_df["matnr"].apply(extract_pos_from_matnr)
            self.logger.info("POS extracted from 'matnr' column")
        else:
            self.spec_df["__pos"] = ""
            self.logger.warning("No 'doknr' or 'matnr' column found for POS extraction")
        
        # __pos가 비어있으면 matnr에서 다시 시도
        if "matnr" in self.spec_df.columns:
            mask = self.spec_df["__pos"] == ""
            if mask.any():
                self.spec_df.loc[mask, "__pos"] = self.spec_df.loc[mask, "matnr"].apply(extract_pos_from_matnr)
                self.logger.info("POS fallback from 'matnr' for %d rows", mask.sum())
        
        # Hull/POS 추출 결과 샘플 출력
        sample_size = min(5, len(self.spec_df))
        if sample_size > 0:
            self.logger.info("Hull/POS extraction sample (first %d rows):", sample_size)
            for i in range(sample_size):
                row = self.spec_df.iloc[i]
                self.logger.info("  [%d] hull='%s', pos='%s', umgv_desc='%s'", 
                               i, row.get("__hull", ""), row.get("__pos", ""), 
                               row.get("umgv_desc", "")[:30])
        
        # Hull/POS 통계
        empty_hull = (self.spec_df["__hull"] == "").sum()
        empty_pos = (self.spec_df["__pos"] == "").sum()
        self.logger.info("Hull/POS stats: %d rows total, %d empty hull, %d empty pos",
                        len(self.spec_df), empty_hull, empty_pos)
        
        # Glossary 필드 부착
        if self.glossary_df is not None:
            self.spec_df = attach_glossary_fields(
                self.spec_df,
                self.glossary_df,
                logger=self.logger,
                hybrid_min_score=0.68,
                hybrid_limit=self.config.glossary_hybrid_limit,
            )
        
        self.logger.info("Data preparation complete")

    def setup_llm(self):
        """LLM Pool 설정 (필요시 Ollama 서버 자동 시작)"""
        self.logger.info("Setting up LLM pool...")
        
        # 1) Ollama 서버 자동 시작 (설정에 따라)
        self.ollama_manager = OllamaServeManager(self.config, self.logger)
        server_status = self.ollama_manager.start_all()
        
        started_count = sum(1 for v in server_status.values() if v)
        self.logger.info("Ollama servers: %d/%d available", started_count, len(self.config.ollama_ports))
        
        # 2) LLM 클라이언트 풀 생성
        self.llm_pool = LLMClientPool(
            host=self.config.ollama_host,
            ports=self.config.ollama_ports,
            model=self.config.model_name,
            timeout=self.config.llm_timeout,
            logger=self.logger,
        )
        
        # 3) 헬스 체크
        health = self.llm_pool.check_health_all()
        healthy_count = sum(1 for v in health.values() if v)
        
        self.logger.info("LLM health check: %d/%d ports healthy", 
                        healthy_count, len(self.config.ollama_ports))
        
        for port, status in health.items():
            self.logger.info("  Port %d: %s", port, "OK" if status else "FAIL")
        
        # 4) 모든 포트가 실패하면 재시도
        if healthy_count == 0:
            self.logger.warning("No healthy ports. Waiting additional 30 seconds and retrying...")
            time.sleep(30)
            
            health = self.llm_pool.check_health_all()
            healthy_count = sum(1 for v in health.values() if v)
            
            if healthy_count == 0:
                self.logger.error("Still no healthy Ollama servers after retry")
                self.logger.error("Please ensure:")
                self.logger.error("  1. Ollama is installed: which ollama")
                self.logger.error("  2. Model is available: ollama list")
                self.logger.error("  3. Ports are not blocked: netstat -tlnp | grep 11434")
                raise RuntimeError("No healthy Ollama servers available")
            
            self.logger.info("Retry successful: %d/%d ports healthy", 
                           healthy_count, len(self.config.ollama_ports))

    def group_specs_by_pos(self) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """사양을 (hull, pos) 기준으로 그룹화"""
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        
        for _, row in self.spec_df.iterrows():
            hull = row.get("__hull", "")
            pos = row.get("__pos", "")
            key = (hull, pos)
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append(row.to_dict())
        
        return groups

    def run(self):
        """메인 실행"""
        self.logger.info("=" * 60)
        self.logger.info("POS Spec Extractor v42 Started")
        self.logger.info("=" * 60)
        
        # 데이터 로드
        self.load_data()
        self.prepare_data()
        
        # LLM 설정
        self.setup_llm()
        
        # 체크포인트 로드
        self.load_checkpoint()
        
        # POS 그룹화
        groups = self.group_specs_by_pos()
        self.logger.info("Total groups: %d (hull, pos)", len(groups))
        
        # 그룹 샘플 출력 (디버깅용)
        sample_groups = list(groups.items())[:5]
        for (hull, pos), specs in sample_groups:
            self.logger.info("  Sample group: hull='%s', pos='%s', specs=%d", hull, pos, len(specs))
        
        # 파일 매칭 사전 확인
        self.logger.info("Checking file availability...")
        found_count = 0
        not_found_count = 0
        file_finder = FileFinder(
            base_dirs=self.config.html_dirs,
            allow_cross_hull_fallback=False,
            logger=self.logger,
        )
        for (hull, pos), specs in groups.items():
            path = file_finder.find_file(hull, pos)
            if path:
                found_count += 1
            else:
                not_found_count += 1
                if not_found_count <= 3:  # 처음 3개만 로깅
                    self.logger.warning("  File not found: %s-POS-%s", hull, pos)
        
        self.logger.info("File check: %d found, %d not found (out of %d groups)", 
                        found_count, not_found_count, len(groups))
        
        if found_count == 0:
            self.logger.error("NO FILES FOUND! Check base_folder=%s", self.config.base_folder)
            self.logger.error("  Expected file pattern: <hull>-POS-<pos>*.html")
            return
        
        # 진행률 표시
        try:
            from tqdm import tqdm
            group_iter = tqdm(groups.items(), desc="Processing POS groups")
        except ImportError:
            group_iter = groups.items()
        
        # FILE_NOT_FOUND 결과는 재처리 대상으로 표시
        # (이전에 파일이 없었지만 지금은 있을 수 있음)
        keys_to_retry = set()
        for r in self.results:
            if r.get(COL_METHOD) == "FILE_NOT_FOUND":
                hull = r.get("__hull", "")
                pos = r.get("__pos", "")
                code = r.get("umgv_code", "") or r.get(COL_SPEC_NAME, "")
                key = (hull, pos, code)
                keys_to_retry.add(key)
        
        if keys_to_retry:
            self.logger.info("Retrying %d FILE_NOT_FOUND results...", len(keys_to_retry))
            self.done_keys -= keys_to_retry
            # 결과에서도 제거
            self.results = [r for r in self.results if r.get(COL_METHOD) != "FILE_NOT_FOUND"]
        
        # 그룹별 처리
        for (hull, pos), specs in group_iter:
            # 이미 완료된 항목 필터링
            remaining_specs = []
            for spec in specs:
                code = spec.get("umgv_code", "") or spec.get(COL_SPEC_NAME, "")
                key = (hull, pos, code)
                if key not in self.done_keys:
                    remaining_specs.append(spec)
            
            if not remaining_specs:
                continue
            
            self.logger.debug("Processing %s/%s: %d specs", hull, pos, len(remaining_specs))
            
            # HTML 파일 찾기
            file_finder = FileFinder(
                base_dirs=self.config.html_dirs,
                allow_cross_hull_fallback=False,
                logger=self.logger,
            )
            
            html_path = file_finder.find_file(hull, pos)
            
            if not html_path:
                # 파일 없음 - 로깅 추가
                self.logger.warning("FILE_NOT_FOUND: hull='%s', pos='%s', dirs=%s", 
                                   hull, pos, self.config.html_dirs)
                for spec in remaining_specs:
                    result = {
                        **spec,
                        COL_SPEC_VALUE: "",
                        COL_SPEC_UNIT: "",
                        COL_EVIDENCE: "",
                        COL_CONFIDENCE: 0.0,
                        COL_METHOD: "FILE_NOT_FOUND",
                        COL_AUDIT: "",
                    }
                    self.results.append(result)
                    code = spec.get("umgv_code", "") or spec.get(COL_SPEC_NAME, "")
                    self.done_keys.add((hull, pos, code))
                continue
            
            self.logger.debug("Found HTML: %s", html_path)
            
            # HTML 파싱
            reader = POSHTMLReader(html_path, logger=self.logger)
            doc_chunks = reader.get_chunks()
            doc_blob = reader.get_text_blob()
            
            # 배치 처리
            batch_results = process_batch(
                specs=remaining_specs,
                doc_chunks=doc_chunks,
                doc_blob=doc_blob,
                glossary_index=self.glossary_index,
                specdb_index=self.specdb_index,
                llm_pool=self.llm_pool,
                config=self.config,
                logger=self.logger,
            )
            
            # 결과 저장
            for result in batch_results:
                self.results.append(result)
                code = result.get("umgv_code", "") or result.get(COL_SPEC_NAME, "")
                self.done_keys.add((hull, pos, code))
            
            # 주기적 체크포인트
            if len(self.results) % 100 == 0:
                self.save_checkpoint()
        
        # 최종 저장
        self.save_checkpoint()
        self.save_results()
        
        self.logger.info("=" * 60)
        self.logger.info("Processing complete: %d results", len(self.results))
        self.logger.info("=" * 60)

    def save_results(self):
        """결과 저장 (모드에 따라 CSV+JSON 또는 JSON+DB)"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if self.config.mode == "file":
            self._save_results_to_files()
        else:
            self._save_results_to_json_and_db()

    def _format_result_for_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        결과를 출력 스키마에 맞게 포맷팅합니다.
        코드_작성_규칙.txt의 출력 JSON 구조를 따릅니다.
        """
        formatted = {}
        
        # 템플릿에서 복사해야 할 컬럼
        template_cols = [
            "pmg_code", "pmg_desc", "umg_code", "umg_desc",
            "extwg", "mat_attr_desc", "matnr", "doknr",
            "umgv_code", "umgv_desc", "umgv_uom",
        ]
        for col in template_cols:
            formatted[col] = result.get(col, "")
        
        # 추출 결과 컬럼
        formatted["section_num"] = result.get(COL_SECTION_NUM, "")
        formatted["table_text"] = result.get(COL_TABLE_TEXT, "")
        formatted["value_format"] = result.get(COL_VALUE_FORMAT, "")
        formatted["pos_chunk"] = result.get(COL_POS_CHUNK, "")
        formatted["pos_mat_attr_desc"] = result.get(COL_POS_EXTWG_DESC, "")
        formatted["pos_umgv_desc"] = result.get(COL_POS_UMGV_DESC, "")
        formatted["pos_umgv_value"] = result.get(COL_POS_UMGV_VALUE, "")
        formatted["umgv_value_edit"] = result.get(COL_UMGV_VALUE_EDIT, "") or result.get(COL_POS_UMGV_VALUE, "")
        formatted["pos_umgv_uom"] = result.get(COL_POS_UMGV_UOM, "")
        formatted["evidence_fb"] = result.get(COL_EVIDENCE_FB, "")  # 초기값 빈 문자열
        
        # 디버깅용 메타 컬럼 (선택적)
        formatted["_method"] = result.get(COL_METHOD, "")
        formatted["_confidence"] = result.get(COL_CONFIDENCE, 0.0)
        
        return formatted

    def _save_results_to_files(self):
        """파일 모드: CSV와 JSON 모두 저장"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 출력 스키마에 맞게 포맷팅
        formatted_results = [self._format_result_for_output(r) for r in self.results]
        
        # JSON 저장 (주요 출력)
        json_path = os.path.join(self.config.output_dir, f"extraction_result_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=2)
        self.logger.info("JSON results saved to: %s", json_path)
        
        # CSV 저장 (보조)
        result_df = pd.DataFrame(formatted_results)
        csv_path = os.path.join(self.config.output_dir, f"extraction_result_{timestamp}.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        self.logger.info("CSV results saved to: %s", csv_path)
        
        # 요약 통계
        total = len(result_df)
        extracted = len(result_df[result_df["pos_umgv_value"] != ""])
        self.logger.info("Summary: %d total, %d extracted (%.1f%%)", 
                        total, extracted, 100 * extracted / total if total else 0)

    def _save_results_to_json_and_db(self):
        """DB 모드: JSON 저장 후 사용자 확인 대기, 이후 DB 업로드"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 출력 스키마에 맞게 포맷팅
        formatted_results = [self._format_result_for_output(r) for r in self.results]
        
        # JSON 저장 (사용자 검토용)
        json_path = os.path.join(self.config.output_dir, f"extraction_result_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=2)
        self.logger.info("JSON results saved to: %s", json_path)
        self.logger.info("Please review the results. After feedback, run upload_to_db() method.")
        
        # 업로드 대기 상태 저장
        pending_path = os.path.join(self.config.output_dir, f"pending_upload_{timestamp}.json")
        pending_info = {
            "json_path": json_path,
            "timestamp": timestamp,
            "record_count": len(formatted_results),
            "status": "pending_review",
        }
        with open(pending_path, "w", encoding="utf-8") as f:
            json.dump(pending_info, f, ensure_ascii=False, indent=2)
        
        # 요약 통계
        total = len(formatted_results)
        extracted = sum(1 for r in formatted_results if r.get("pos_umgv_value"))
        self.logger.info("Summary: %d total, %d extracted (%.1f%%)", 
                        total, extracted, 100 * extracted / total if total else 0)

    def upload_reviewed_results(self, json_path: str, feedback_updates: Optional[Dict] = None):
        """
        사용자가 검토한 JSON 결과를 DB에 업로드합니다.
        
        Args:
            json_path: 검토가 완료된 JSON 파일 경로
            feedback_updates: {record_index: {"evidence_fb": "피드백 내용", ...}} 형태의 수정사항
        
        단위 변환 구조:
            - pos_umgv_value: POS 문서 원본 값
            - pos_umgv_uom: POS 문서 원본 단위
            - umgv_uom: 시스템 표준 단위 (DB에서 로드)
            - umgv_value_edit: 사용자가 변환한 값 (pos_umgv_uom -> umgv_uom 변환 시)
        """
        if not HAS_PSYCOPG2:
            self.logger.error("psycopg2 not installed. Cannot upload to DB.")
            return
        
        # JSON 로드
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # 피드백 반영
        if feedback_updates:
            for idx_str, updates in feedback_updates.items():
                idx = int(idx_str)
                if 0 <= idx < len(results):
                    results[idx].update(updates)
        
        # 피드백 완료 후 발견된 동의어 저장
        unit_manager = get_unit_manager()
        pending_count = unit_manager.get_pending_count()
        if pending_count > 0:
            self.logger.info("Saving %d new unit synonyms discovered during extraction...", pending_count)
            if unit_manager.save_pending():
                self.logger.info("Unit synonyms saved successfully")
            else:
                self.logger.warning("Failed to save unit synonyms")
        
        # DB 업로드
        pg_loader = PostgresLoader(
            host=self.config.db_host,
            port=self.config.db_port,
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            logger=self.logger,
        )
        
        result_df = pd.DataFrame(results)
        
        success = pg_loader.save_results(
            df=result_df,
            table_name=self.config.db_output_table,
        )
        
        if success:
            self.logger.info("Results uploaded to DB table: %s (%d records)", 
                           self.config.db_output_table, len(results))
            
            # 용어집 업데이트 (피드백이 있는 경우)
            if feedback_updates:
                self._update_glossary_from_feedback(results, feedback_updates)
        else:
            self.logger.error("DB upload failed")

    def _update_glossary_from_feedback(self, results: List[Dict], feedback_updates: Dict):
        """
        사용자 피드백을 용어집에 반영합니다.
        """
        self.logger.info("Updating glossary from %d feedback entries...", len(feedback_updates))
        # TODO: 용어집 DB 테이블에 피드백 반영 로직 구현
        # 현재는 로그만 남김
        for idx_str, updates in feedback_updates.items():
            if updates.get("evidence_fb"):
                self.logger.debug("  Record %s: %s", idx_str, updates.get("evidence_fb")[:50])


# =============================================================================
# 메인 실행 함수
# =============================================================================

def main():
    """
    메인 실행 진입점.
    
    USER_* 상수를 Config로 변환하고 실행합니다.
    """
    print("=" * 70)
    print("POS Specification Value Extractor v42")  # v41 버전
    print("=" * 70)
    
    # 설정 생성 (build_config 사용)
    config = build_config()
    
    # v41: 참조 로거 초기화
    output_dir = config.output_path if os.path.isdir(config.output_path) else os.path.dirname(config.output_path)
    reset_reference_logger_v41(output_dir)
    
    # 설정 출력
    print(f"\nConfiguration:")
    print(f"  Mode: {config.mode}")
    print(f"  Input (spec_path): {config.spec_path}")
    print(f"  Output: {config.output_path}")
    print(f"  HTML folder: {config.base_folder}")
    print(f"  Glossary: {config.glossary_path}")
    print(f"  SpecDB: {config.specdb_path}")
    print(f"  Ollama: {config.ollama_host}:{config.ollama_ports}")
    print(f"  Model: {config.ollama_model}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Vote K: {config.vote_k}")
    print(f"  Rule threshold: {config.rule_conf_threshold}")
    print(f"  Workers: {config.num_workers}")
    print()
    
    # 프로세서 생성 및 실행
    processor = POSSpecProcessor(config)
    
    try:
        processor.run()
        
        # v41: 참조 로그 저장
        ref_logger = get_reference_logger_v41()
        if ref_logger.get_count() > 0:
            ref_filepath = ref_logger.save()
            print(f"\n[v41] Reference log saved: {ref_filepath} ({ref_logger.get_count()} entries)")
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Saving checkpoint...")
        processor.save_checkpoint()
        
        # v41: 참조 로그도 저장
        ref_logger = get_reference_logger_v41()
        if ref_logger.get_count() > 0:
            ref_logger.save("reference_log_interrupted.csv")
        
        print("[!] Checkpoint saved. Resume with same settings to continue.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        processor.save_checkpoint()
        sys.exit(1)


# =============================================================================
# 엔트리 포인트
# =============================================================================

if __name__ == "__main__":
    main()
